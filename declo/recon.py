"""DeClotH steps (2)-(4), (7): two DMTets (cloth, body) in the canonical A-pose, posed by SMPL-X LBS, optimised
against the photo's targets (paper Sec. 3.3, 3.6). SDS terms (Sec. 3.4-3.5) plug in through `extra_losses`.

L_geo = L_cloth-t + L_body-t + [L_cloth-SDS + L_human-SDS] + L_normal + L_sil + L_pen

* L_*-t   : silhouettes of each DMTet vs its template, photo camera + a bank of orbit cameras
* L_sil   : cloth render vs SAM cloth mask, cloth+body composite vs SAM human mask (front; the back view's
            silhouette is the same mask under our "same rays, far surface" back view)
* L_normal: composite front/back normals vs ECON's normal maps (when installed)
* L_pen   : body surface inside the cloth and cloth surface inside the body, via the other DMTet's SDF
Schedule: Adam, lr 1e-3 * 0.1^(step/steps) (paper).

Ours, not the paper's: the cloth DMTet is a tear-free shell (dmtet.ShellDMTet) on its own finer grid; the body
DMTet is a plain DMTet. Both share the hash-grid MLP design.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from . import dmtet, render
from .cameras import Orbit, to_cam_frame
from .skinning import Skinning

DEFAULTS = dict(cell_body=0.01, band_body=0.06, cell_cloth=0.006, band_cloth_out=0.045, band_cloth_in=0.01,
                tau=0.0055, mid_gap=0.0015, pen_gap=0.001, body_tmpl_inside=0.1, res=384, steps=500, lr=1e-3, n_random_views=1, n_view_bank=24,
                w_sil=1.0, w_tmpl=0.5, w_tmpl_cloth_standin=0.1, w_normal=0.05, w_pen=100.0, w_sdf_reg=0.01, w_nc=0.0, w_keep=0.001,
                deform=True, hash_max_res=1024, seed=0)


def cloth_patch_from_mask(body: dict, cloth_mask: np.ndarray, K: np.ndarray, smplx_model, human_mask=None,
                          edge_px=4, min_facing=0.3, smooth_iters=2) -> np.ndarray:
    """Stand-in for ClothNet until ClothWild runs: which body-template vertices are under the garments.
    Returns a bool per SMPL-X vertex.

    The photo is only trusted where it can be: vertices that are visible in the photo (z-buffer), face the camera
    (n.view > min_facing) and project inside the person mask away from its edge. Every other vertex (hidden behind
    an arm, grazing at the silhouette, on the back) takes the label of the geodesically nearest trusted vertex
    (Dijkstra over the canonical SMPL-X surface). Plain projection labelled the hip sides behind the arms as
    skin (the arm is in front in the photo) and shoulder tops as background (grazing, slightly misaligned), which
    cut holes in the cloth shell where the body then showed through. Head, hands and feet are never cloth."""
    import cv2
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    from . import raster
    V, F = np.asarray(body["V_cam"], np.float64), np.asarray(body["F"])
    H, W = cloth_mask.shape
    uvw = V @ K.T
    z = uvw[:, 2]
    uv = uvw[:, :2] / z[:, None]
    x = np.clip(np.floor(uv[:, 0]).astype(int), 0, W - 1)
    y = np.clip(np.floor(uv[:, 1]).astype(int), 0, H - 1)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, depth = raster.rasterize(torch.tensor(V, dtype=torch.float32, device=dev), F, K, H, W)
    depth = depth.cpu().numpy()
    visible = z <= depth[y, x] + 0.01
    n = np.asarray(raster.vertex_normals(torch.tensor(V, dtype=torch.float32), torch.tensor(F)).numpy(), np.float64)
    facing = np.einsum("ij,ij->i", n, -V / np.linalg.norm(V, axis=1, keepdims=True)) > min_facing
    inside = np.ones(len(V), bool)
    if human_mask is not None:
        core = cv2.erode(human_mask.astype(np.uint8), np.ones((2 * edge_px + 1, 2 * edge_px + 1)))
        inside = core[y, x] > 0
    Wl = smplx_model.lbs_weights.cpu().numpy()
    extremity = Wl[:, [7, 8, 10, 11, 12, 15, 20, 21, 22, 23, 24] + list(range(25, 55))].sum(1) > 0.5
    cloth = cv2.dilate(cloth_mask.astype(np.uint8), np.ones((3, 3)))[y, x] > 0
    trusted = visible & facing & inside & ~extremity
    # geodesic nearest trusted vertex over the canonical surface (pose-independent distances)
    Vc = np.asarray(body["canon"]["V"], np.float64)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    w = np.linalg.norm(Vc[e[:, 0]] - Vc[e[:, 1]], axis=1) + 1e-9
    G = coo_matrix((w, (e[:, 0], e[:, 1])), shape=(len(V), len(V))).tocsr()
    src = np.nonzero(trusted)[0]
    label = cloth & trusted
    if len(src):
        _, _, nearest = dijkstra(G, directed=False, indices=src, min_only=True, return_predecessors=True)
        ok = nearest >= 0
        label = np.where(trusted, cloth, False)
        label[ok & ~trusted] = cloth[nearest[ok & ~trusted]]
    label &= ~extremity
    A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(V), len(V))).tocsr()
    A = ((A + A.T) > 0).astype(np.float64)
    deg = np.asarray(A.sum(1)).ravel() + 1
    for _ in range(smooth_iters):                          # neighbour majority: no single-vertex islands / holes
        label = ((A @ label.astype(np.float64) + label) / deg > 0.5) & ~extremity
    return label


def shell_targets(X, Vb, Fb, patch, mid, garment_pts=None, mid_max=0.08):
    """Initial shell fields at points X (canonical): f0 = body SDF - mid (mid-surface `mid` outside the body);
    g0 = garment region via the nearest body vertex: -(distance to the patch edge) on the patch, +distance outside.
    garment_pts (canonical ClothWild vertices): mid follows the garment - its body SDF at the nearest garment
    point (a loose shirt 3 cm off the body gets a shell 3 cm out). Limitation: a skirt between the legs becomes
    a shell around each leg (the offset is measured from the nearest body part)."""
    from scipy.spatial import cKDTree
    sdf_b = dmtet.mesh_sdf(X, Vb, Fb).numpy()
    if garment_pts is not None and len(garment_pts):
        h = dmtet.mesh_sdf(garment_pts, Vb, Fb).numpy()               # garment offset from the body
        _, gi = cKDTree(garment_pts).query(np.asarray(X), workers=-1)
        mid = np.clip(h[gi], mid, mid_max)
    Fp = Fb[patch[Fb].sum(1) > 0]
    edge = np.unique(Fp[~patch[Fp].all(1)])               # vertices of faces that straddle the patch edge
    _, nn = cKDTree(Vb).query(np.asarray(X), workers=-1)
    d_edge, _ = cKDTree(Vb[edge]).query(Vb[nn], workers=-1)
    g0 = np.where(patch[nn], -d_edge, d_edge)
    f = lambda a: torch.tensor(a, dtype=torch.float32)
    return f(sdf_b - mid), f(g0)


class Reconstruction:
    def __init__(self, prep: dict, smplx_model, cloth_patch=None, cfg=None, log=print):
        """prep: Preprocessor output (with prep["cloth_t"] from ClothNet when SMPL is installed: its garments are
        the cloth template for L_cloth-t and shape the initial shell). cloth_patch: bool per SMPL-X vertex;
        None -> ClothNet's, or without ClothNet the SAM-mask stand-in."""
        from scipy.spatial import cKDTree
        self.cfg = {**DEFAULTS, **(cfg or {})}
        c, dev = self.cfg, "cuda"
        self.dev, self.log = dev, log
        t0 = time.time()
        body = prep["body"]
        self.sk = Skinning(smplx_model, body)
        Vfull, Ffull = body["canon"]["V"], body["F"]
        f32 = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
        ct = prep.get("cloth_t")
        self.cloth_template = None                        # ClothWild garments (minus shoes): posed + canonical
        if ct is not None and any(k != "shoes" for k in ct["garments"]):
            Vs, Fs, o = [], [], 0
            for k, (V, F) in ct["garments"].items():
                if k != "shoes":
                    Vs.append(V); Fs.append(F + o); o += len(V)
            Vg, Fg = np.concatenate(Vs), np.concatenate(Fs)
            Vg_can = self.sk.to_canon(f32(Vg), self.sk.weights_posed(f32(Vg), body["V_cam"])).cpu().numpy()
            self.cloth_template = (Vg, Fg, Vg_can)
            if cloth_patch is None:
                cloth_patch = ct["patch"]
        if cloth_patch is None:
            cloth_patch = cloth_patch_from_mask(body, prep["masks"]["cloth"], prep["K"], smplx_model,
                                                human_mask=prep["masks"]["human"])
        Vb_t, Fb_t = dmtet.main_component(Vfull, Ffull)
        _, idx = cKDTree(Vfull).query(Vb_t)                # patch flags onto the eyeball-free body by position
        patch = cloth_patch[idx]
        self.patch = patch

        # body DMTet: 1 cm grid in a band around the body template (+ its inside)
        kb = dmtet.band_keep([(Vb_t, Fb_t)], band=c["band_body"], solid=(Vb_t, Fb_t))
        lo_b, hi_b = Vb_t.min(0) - 0.08, Vb_t.max(0) + 0.08
        vb, tb = dmtet.tet_grid(lo_b, hi_b, c["cell_body"], kb)
        self.lookup_body = dmtet.GridLookup(lo_b, hi_b, c["cell_body"], kb)
        # cloth DMTet: finer grid in a shell band around the garment part of the body (-1 cm .. +4.5 cm)
        Pv = Vb_t[patch]
        inside = dmtet.inside_voxels(Vb_t, Fb_t)
        tree = cKDTree(Pv)

        def kc(centres):
            d, _ = tree.query(centres, workers=-1, distance_upper_bound=c["band_cloth_out"])
            near = np.isfinite(d)
            deep = near & inside(centres)
            keep = near.copy()
            if deep.any():
                keep[deep] = dmtet.mesh_sdf(centres[deep], Vb_t, Fb_t).numpy() > -c["band_cloth_in"]
            return keep
        lo_c, hi_c = Pv.min(0) - 0.06, Pv.max(0) + 0.06
        vc, tc = dmtet.tet_grid(lo_c, hi_c, c["cell_cloth"], kc)
        self.lookup_cloth = dmtet.GridLookup(lo_c, hi_c, c["cell_cloth"], kc)

        self.nets = {"body": dmtet.DMTet(vb.to(dev), tb.to(dev), c["cell_body"], max_res=c["hash_max_res"],
                                         deform=c["deform"]).to(dev),
                     "cloth": dmtet.ShellDMTet(vc.to(dev), tc.to(dev), c["cell_cloth"], c["tau"],
                                               max_res=c["hash_max_res"], deform=c["deform"]).to(dev)}
        self.sdf0 = {"body": dmtet.mesh_sdf(vb, Vb_t, Fb_t).to(dev)}
        dmtet.fit_sdf(self.nets["body"], self.sdf0["body"], log=log)
        f0, g0 = shell_targets(vc.numpy(), Vb_t, Fb_t, patch, c["tau"] + c["mid_gap"],
                               garment_pts=None if self.cloth_template is None else self.cloth_template[2])
        dmtet.fit_shell(self.nets["cloth"], f0.to(dev), g0.to(dev), log=log)
        with torch.no_grad():
            self.sdf0["cloth"] = self.nets["cloth"].query(self.nets["cloth"].verts)[0]
        for k in self.sdf0:
            self.sdf0[k] = self.sdf0[k].clamp(-4 * c["cell_body"], 4 * c["cell_body"])
        self.Wg = {k: self.sk.weights(n.verts) for k, n in self.nets.items()}
        log(f"grids: body {len(vb) / 1e3:.0f}k verts / {len(tb) / 1e6:.2f}M tets, cloth {len(vc) / 1e3:.0f}k verts / "
            f"{len(tc) / 1e6:.2f}M tets; DMTets initialised ({time.time() - t0:.0f}s)")

        # cameras: photo (front + back) at res, a bank of orbit views for the template loss
        res = c["res"]
        s = res / prep["frame"].res
        self.K = torch.tensor(np.diag([s, s, 1.0]) @ prep["K"], dtype=torch.float32, device=dev)
        self.res = res
        self.orbit = Orbit(body["V_cam"], body["R"] @ np.array([0, 1, 0.0]), res=res)
        self.rng = np.random.default_rng(c["seed"])
        rs = lambda m: torch.nn.functional.interpolate(f32(m)[None, None], size=(res, res), mode="area")[0, 0]
        self.S_cloth, self.S_human = rs(prep["masks"]["cloth"]), rs(prep["masks"]["human"])
        # where the photo shows an arm, it cannot tell whether cloth is behind it (the hip under a hanging arm):
        # the cloth-only silhouette term ignores those pixels instead of pulling the hidden cloth away
        import cv2
        arms = np.isin(prep["labels"], [14, 15]).astype(np.uint8)          # SegFormer Left-arm / Right-arm
        self.arm_core = rs(cv2.erode(arms, np.ones((3, 3)))) > 0.5          # surely arm skin in the photo
        arms = cv2.dilate(arms, np.ones((5, 5)))
        self.cloth_sil_w = 1 - rs(arms)
        self.N = None
        if prep.get("normals") is not None:
            n = torch.nn.functional.interpolate(prep["normals"].float(), size=(res, res), mode="bilinear")
            self.N = n.permute(0, 2, 3, 1).to(dev)                          # (2,res,res,3) front/back, ECON frame
        # templates: body = SMPL-X; cloth = the initial shell (what ClothNet's template would be); posed once
        with torch.no_grad():
            Vc0, Fc0, oc0 = self.posed("cloth")
        if len(Fc0) == 0:
            raise RuntimeError(f"cloth shell empty after initialisation (garment region {int(patch.sum())} SMPL-X "
                               f"vertices, cloth grid {len(vc)} verts): no garment found on the body")
        self.templates = {"body": (Vb_t, Fb_t), "cloth": (oc0["V"].cpu().numpy(), Fc0.cpu().numpy())}
        self.T_posed = {"body": (self.sk.to_cam(f32(Vb_t), self.sk.weights(f32(Vb_t))).detach(),
                                 torch.tensor(Fb_t, device=dev)),
                        "cloth": (Vc0.detach(), Fc0) if self.cloth_template is None else
                                 (f32(self.cloth_template[0]), torch.tensor(self.cloth_template[1], device=dev))}
        # cloth template weight relative to the body's: ClothWild's garment -> full; stand-in (body + offset) ->
        # w_tmpl_cloth_standin / w_tmpl (it held shoulder flaps wider than the photo against the silhouette loss)
        self.cloth_tmpl_scale = 1.0 if self.cloth_template is not None else (c["w_tmpl_cloth_standin"] / c["w_tmpl"]) ** 0.5
        self.views = [(self.K, None, None)]
        for _ in range(c["n_view_bank"]):
            _, (Ko, R, t) = self.orbit.sample(self.rng)
            self.views.append((f32(Ko), R, t))
        self.T_masks = []
        with torch.no_grad():
            for Kk, R, t in self.views:
                tf = (lambda V: V) if R is None else (lambda V: to_cam_frame(V, R, t))
                self.T_masks.append({k: render.render(tf(V), F, Kk, res)["mask"] for k, (V, F) in self.T_posed.items()})
        params = [p for n in self.nets.values() for p in n.parameters()]
        self.opt = torch.optim.Adam(params, lr=c["lr"])
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda i: 0.1 ** (i / c["steps"]))
        self.step_i = 0

    # ------------------------------------------------------------------ meshes
    def posed(self, name):
        o = self.nets[name]()
        ev, w, Wg = o["edge_verts"], o["edge_w"], self.Wg[name]
        W = Wg[ev[:, 0]] * (1 - w)[:, None] + Wg[ev[:, 1]] * w[:, None]
        return self.sk.to_cam(o["V"], W), o["F"], o

    @staticmethod
    def _sdf_reg(o, tets):
        """nvdiffrec's SDF regulariser: penalise sign changes across tet edges (fewer floaters / thin sheets)."""
        sdf = o["sdf"] / o["cell"]                                  # logits in cells (SDF is in metres)
        e = tets[:, [0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3]].reshape(-1, 2)
        a, b = sdf[e[:, 0]], sdf[e[:, 1]]
        flip = (a > 0) != (b > 0)
        if not flip.any():
            return sdf.sum() * 0
        bce = torch.nn.functional.binary_cross_entropy_with_logits
        return bce(a[flip], (b[flip] > 0).float()) + bce(b[flip], (a[flip] > 0).float())

    def _pen(self, Vc_can, Vb_can, n=20000, gap=0.002):
        """L_pen in the canonical frame, each net evaluated only where its grid exists (elsewhere never trained):
        (1) body surface in the garment region (g < 0) must lie inside the cloth's INNER surface by `gap`:
            f(x) < -(tau + gap), f = the shell's signed mid-surface field. A plain "body inside the shell volume"
            test misses body that went all the way through (it is outside the thin shell again): that showed as
            body through the sleeve caps.
        (2) cloth surface must stay outside the body by `gap` (body DMTet SDF)."""
        sub = lambda V: V[torch.randperm(len(V), device=V.device)[:n]]
        Xb, Xc = sub(Vb_can), sub(Vc_can)
        Xb, Xc = Xb[self.lookup_cloth(Xb)], Xc[self.lookup_body(Xc)]
        zero = Vc_can.sum() * 0
        cloth = self.nets["cloth"]
        l1 = zero
        if len(Xb):
            f, g, _ = cloth.fields(Xb)
            under = (g < 0).detach()
            if under.any():
                l1 = torch.relu(f[under] + cloth.tau + gap).mean()
        l2 = torch.relu(-self.nets["body"].query(Xc)[0] + gap).mean() if len(Xc) else zero
        return l1 + l2

    # ------------------------------------------------------------------ one step
    def step(self, extra_losses=None):
        c, res, K = self.cfg, self.res, self.K
        Vc, Fc, oc = self.posed("cloth")
        Vb, Fb, ob = self.posed("body")
        L = {}
        Vall = torch.cat([Vc, Vb])
        Fall = torch.cat([Fc, Fb + len(Vc)])
        rc = render.render(Vc, Fc, K, res)
        ra = render.render(Vall, Fall, K, res)
        # ...but cloth IN FRONT of the arm (frontmost face in the composite is cloth where the photo shows arm skin)
        # is a sleeve over the visible arm: penalised again (a ClothWild template with long sleeves kept them)
        sleeve = self.arm_core & (ra["face"] >= 0) & (ra["face"] < len(Fc))
        w_sil = torch.where(sleeve, torch.ones_like(self.cloth_sil_w), self.cloth_sil_w)
        L["sil"] = (w_sil * (rc["mask"] - self.S_cloth).pow(2)).mean() + (ra["mask"] - self.S_human).pow(2).mean()
        if self.N is not None:                     # L_normal: front (ECON or Sapiens) and back (ECON) when present
            m = (self.S_human > 0.5) & (self.N[0].abs().sum(-1) > 0)
            L["normal"] = (ra["normal"] - self.N[0])[m].pow(2).sum(-1).mean()
            if self.N[1].abs().sum() > 0:
                rb = render.render(Vall, Fall, K, res, far=True)
                mb = (self.S_human > 0.5) & (self.N[1].abs().sum(-1) > 0)
                L["normal"] = L["normal"] + (rb["normal"] - self.N[1])[mb].pow(2).sum(-1).mean()
        pick = [0] + list(self.rng.integers(1, len(self.views), c["n_random_views"]))
        lt = 0
        for i in pick:
            Kk, R, t = self.views[i]
            tf = (lambda V: V) if R is None else (lambda V: to_cam_frame(V, R, t))
            for name, (V, F) in (("cloth", (Vc, Fc)), ("body", (Vb, Fb))):
                diff = render.render(tf(V), F, Kk, res)["mask"] - self.T_masks[i][name]
                if name == "body":     # one-sided: the body under clothes may be slimmer than its template, not
                    diff = torch.where(diff < 0, diff * c["body_tmpl_inside"] ** 0.5, diff)   # wider (gave way to the cloth)
                if name == "cloth":   # stand-in cloth template = SMPL-X + offset: as wide as the body fit at the
                    diff = diff * self.cloth_tmpl_scale    # shoulders, so it gets a low weight until ClothWild
                lt = lt + diff.pow(2).mean()
        L["tmpl"] = lt / len(pick)
        L["pen"] = self._pen(oc["V"], ob["V"], gap=c["pen_gap"])
        L["sdf_reg"] = self._sdf_reg(oc, self.nets["cloth"].tets) + self._sdf_reg(ob, self.nets["body"].tets)
        if c["w_nc"]:
            L["nc"] = dmtet.normal_consistency(oc["V"], oc["F"]) + dmtet.normal_consistency(ob["V"], ob["F"])
        if c["w_keep"]:      # weak 3D template prior: sparse (silhouette-only) gradients + hash collisions -> noise
            L["keep"] = 0
            for k, o in (("cloth", oc), ("body", ob)):
                idx = torch.randint(0, len(o["sdf"]), (65536,), device=self.dev)
                L["keep"] = L["keep"] + ((o["sdf"][idx] - self.sdf0[k][idx]) / o["cell"]).pow(2).mean()
        total = (c["w_sil"] * L["sil"] + c["w_tmpl"] * L["tmpl"] + c["w_pen"] * L["pen"] + c["w_sdf_reg"] * L["sdf_reg"]
                 + c["w_normal"] * L.get("normal", 0) + c["w_nc"] * L.get("nc", 0) + c["w_keep"] * L.get("keep", 0))
        if extra_losses is not None:
            # SDS on detached copies of the posed meshes: the guidance back-propagates each term into these leaves
            # straight away (one diffusion graph alive at a time); one surrogate term then carries the summed
            # mesh gradients back through LBS, Marching Tetrahedra and the DMTet networks.
            Vc_d, Vb_d = Vc.detach().requires_grad_(), Vb.detach().requires_grad_()
            for k, (w, v) in extra_losses(self, {"cloth": (Vc_d, Fc), "body": (Vb_d, Fb)}).items():
                L[k] = v
                if v.requires_grad:                                 # a term that did not back-propagate itself
                    v = (w * v)
                    v.backward(retain_graph=True)
            for V, Vd in ((Vc, Vc_d), (Vb, Vb_d)):
                if Vd.grad is not None:
                    total = total + (V * Vd.grad).sum()
        self.opt.zero_grad()
        total.backward()
        self.opt.step()
        self.sched.step()
        self.step_i += 1
        return {k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in L.items()}

    @torch.no_grad()
    def meshes(self, posed=True):
        out = {}
        for name in ("cloth", "body"):
            V, F, o = self.posed(name)
            out[name] = ((V if posed else o["V"]).cpu().numpy(), F.cpu().numpy())
        return out

    @torch.no_grad()
    def metrics(self):
        """Front-view IoU of cloth / composite against the SAM masks."""
        Vc, Fc, _ = self.posed("cloth")
        Vb, Fb, _ = self.posed("body")
        rc = render.render(Vc, Fc, self.K, self.res)["mask"] > 0.5
        o = render.render(torch.cat([Vc, Vb]), torch.cat([Fc, Fb + len(Vc)]), self.K, self.res)
        ra = o["mask"] > 0.5
        iou = lambda a, b: ((a & b).sum() / (a | b).sum()).item()
        v = self.cloth_sil_w > 0.5                                   # cloth: pixels not covered by an arm in the photo
        sleeve = self.arm_core & (o["face"] >= 0) & (o["face"] < len(Fc))
        return {"iou_cloth": iou(rc & v, (self.S_cloth > 0.5) & v), "iou_human": iou(ra, self.S_human > 0.5),
                "cloth_over_arm": (sleeve.sum() / self.arm_core.sum().clamp_min(1)).item()}
