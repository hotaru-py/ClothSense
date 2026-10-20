"""DeClotH step (8): texture stage (paper Sec. 3.6, Eq. 10). NOT RUN YET.

L_tex = L_cloth-SDS^rgb + L_human-SDS^rgb + L_recon^tex,  L_recon^tex = L2 + LPIPS(rendered RGB, photo)

The geometry from the geometry stage is frozen. Two colour fields (Psi_cloth, Psi_human) map a point of the
canonical A-pose surface to RGB; they are queried per pixel (the rasteriser interpolates each pixel's canonical
position), so texture detail is not limited by the mesh's ~1 cm vertex spacing. Canonical inputs mean the
texture moves with the cloth when it is re-posed or swapped onto another body (M6).
Reconstruction targets: the photo in the front view only. The paper also lists a back view; one photo has no
back RGB, so the back is left to the SDS terms (our reading; the paper does not say where a back RGB would come from).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from . import render
from .cameras import to_cam_frame
from .captions import view_suffix
from .dmtet import HashGrid
from .sds import CN_OPENPOSE

CN_SEG = "lllyasviel/control_v11p_sd15_seg"


class ColorField(nn.Module):
    """Paper (supp. S4): one fully-connected hidden layer of 32 (ReLU) on a hash encoding with max resolution 2048."""

    def __init__(self, lo, hi, hidden=32):
        super().__init__()
        self.enc = HashGrid(lo, hi, levels=16, feats=2, log2_size=19, base_res=16, max_res=2048)
        self.mlp = nn.Sequential(nn.Linear(self.enc.out_dim, hidden), nn.ReLU(), nn.Linear(hidden, 3))

    def forward(self, x):
        return torch.sigmoid(self.mlp(self.enc(x)))


class Background(nn.Module):
    """Paper (supp. S4): an MLP with two fully-connected layers mapping the camera to an adaptive background
    colour. Input: the camera's viewing direction in the photo frame."""

    def __init__(self, hidden=32):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(3, hidden), nn.ReLU(), nn.Linear(hidden, 3))

    def forward(self, R):
        d = torch.as_tensor(np.asarray(R)[2], dtype=torch.float32, device=self.mlp[0].weight.device)
        return torch.sigmoid(self.mlp(d))


class TextureStage:
    def __init__(self, rec, prep, guidance=None, steps=500, lr=1e-2, res=512, w_l2=1.0, w_lpips=0.5,
                 w_sds=(1e-5, 5e-6), guidance_scale=15.0, w_anchor=10.0, seed=0):
        """rec: Reconstruction after the geometry stage. guidance: declo.guidance.Guidance (shares the SD +
        ControlNets); None -> reconstruction losses only.
        guidance_scale: CFG for RGB SDS. 50 (the geometry stage's) over-saturated unseen sides (a navy tee's back
        turned electric blue). w_anchor: the cloth's mean colour in a random view stays near its mean in the photo."""
        dev = rec.dev
        self.rec, self.g, self.res, self.dev = rec, guidance, res, dev
        self.w = {"l2": w_l2, "lpips": w_lpips, "sds_cloth": w_sds[0], "sds_human": w_sds[1]}
        self.mesh = {}
        with torch.no_grad():
            for name in ("cloth", "body"):
                V, F, o = rec.posed(name)
                self.mesh[name] = (V.detach(), F, o["V"].detach())          # posed (camera), faces, canonical
        allc = torch.cat([m[2] for m in self.mesh.values()]).cpu().numpy()
        lo, hi = allc.min(0) - 0.02, allc.max(0) + 0.02
        self.nets = nn.ModuleDict({k: ColorField(lo, hi) for k in self.mesh}).to(dev)
        self.bg = Background().to(dev)
        self.opt = torch.optim.Adam(list(self.nets.parameters()) + list(self.bg.parameters()), lr=lr)
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda i: 0.1 ** (i / max(steps, 1)))
        s = res / prep["frame"].res
        self.K = torch.tensor(np.diag([s, s, 1.0]) @ prep["K"], dtype=torch.float32, device=dev)
        f = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
        rs = lambda a, mode: torch.nn.functional.interpolate(a, size=(res, res), mode=mode)
        self.photo = rs(f(prep["rgb"]).permute(2, 0, 1)[None] / 255.0, "bilinear")[0].permute(1, 2, 0)
        self.m_human = rs(f(prep["masks"]["human"])[None, None], "area")[0, 0]
        self.m_cloth = rs(f(prep["masks"]["cloth"])[None, None], "area")[0, 0]
        import lpips
        self.lpips = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval().requires_grad_(False)
        self.rng = np.random.default_rng(seed)
        self.cfg_scale, self.w_anchor = guidance_scale, w_anchor
        # per-garment colour anchors: label cloth vertices upper / lower by projecting them into the photo's SAM
        # masks (a garment's front and back fall in the same image region), and take each garment's photo mean.
        # One mean for all the cloth dragged a white pair of shorts to the navy of the tee.
        mu = rs(f(prep["masks"]["upper"])[None, None], "area")[0, 0] > 0.5
        ml = rs(f(prep["masks"]["lower"])[None, None], "area")[0, 0] > 0.5
        Vc = self.mesh["cloth"][0]
        uv = Vc @ self.K.T
        uv = (uv[:, :2] / uv[:, 2:3]).long().clamp(0, res - 1)
        up, lo_ = mu[uv[:, 1], uv[:, 0]], ml[uv[:, 1], uv[:, 0]]
        lab = torch.where(lo_ & ~up, 1.0, 0.0)
        if (up | lo_).any() and (~(up | lo_)).any():         # unlabelled: copy the nearest labelled vertex
            from scipy.spatial import cKDTree               # (a GPU cdist here needed 7 GB)
            known = torch.nonzero(up | lo_).squeeze(1)
            unk = torch.nonzero(~(up | lo_)).squeeze(1)
            V_np = Vc.cpu().numpy()
            _, j = cKDTree(V_np[known.cpu().numpy()]).query(V_np[unk.cpu().numpy()], workers=-1)
            lab[unk] = lab[known[torch.as_tensor(j, device=lab.device)]]
        self.cloth_label = lab
        self.garment_mean = {k: self.photo[m].mean(0) for k, m in ((0, mu), (1, ml)) if m.any()}

    def render(self, names, K, R=None, t=None, bg=1.0):
        """RGB of the given meshes (composited in depth order), (res,res,3), the antialiased mask, and which mesh
        each pixel shows (index into `names`, -1 = background)."""
        Vs, Fs, attr, off = [], [], [], 0
        for i, n in enumerate(names):
            V, F, Vc = self.mesh[n]
            Vs.append(V if R is None else to_cam_frame(V, R, t))
            Fs.append(F + off)
            lab = self.cloth_label[:, None] if n == "cloth" and hasattr(self, "cloth_label") else torch.full_like(Vc[:, :1], -1.0)
            attr.append(torch.cat([Vc, torch.full_like(Vc[:, :1], float(i)), lab], 1))   # canonical xyz, mesh id, garment
            off += len(V)
        o = render.render(torch.cat(Vs), torch.cat(Fs), K, self.res, colors=torch.cat(attr))
        fg = o["face"] >= 0
        P = o["rgb"][fg]
        rgb = torch.zeros(self.res, self.res, 3, device=self.dev)
        vals = torch.zeros(len(P), 3, device=self.dev)
        mid = P[:, 3].round().long()
        for i, n in enumerate(names):
            sel = mid == i
            if sel.any():
                vals[sel] = self.nets[n](P[sel, :3])
        rgb[fg] = vals
        ids = torch.full((self.res, self.res), -1, dtype=torch.long, device=self.dev)
        ids[fg] = mid
        self.last_garment = torch.full((self.res, self.res), -1.0, device=self.dev)
        self.last_garment[fg] = P[:, 4].detach()
        m = o["mask"][..., None]
        return rgb * m + bg * (1 - m), o["mask"], ids

    def _lpips(self, a, b, mask):
        ys, xs = torch.nonzero(mask > 0.5, as_tuple=True)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        crop = lambda im: im[y0:y1, x0:x1].permute(2, 0, 1)[None] * 2 - 1
        return self.lpips(crop(a), crop(b)).mean()

    def step(self):
        self.opt.zero_grad()            # first: the SDS terms below back-propagate as they go
        L = {}
        # front view reconstruction: composite vs photo (human mask), cloth alone vs photo (cloth mask)
        rgb_all, _, ids = self.render(("cloth", "body"), self.K)
        rgb_cloth, _, _ = self.render(("cloth",), self.K)
        mh, mc = self.m_human[..., None], self.m_cloth[..., None]
        # region-aware: a rendered body pixel learns the photo only where the photo shows skin (not clothes);
        # otherwise the body under a slightly narrow cloth mesh learns the shirt colour and shows it from the side
        w = torch.where((ids == 0)[..., None], mc, torch.where((ids == 1)[..., None], mh * (1 - mc), mh))
        L["l2"] = ((rgb_all - self.photo) * w).pow(2).mean() + ((rgb_cloth - self.photo) * mc).pow(2).mean()
        tgt = self.photo * mh + (1 - mh)
        L["lpips"] = self._lpips(rgb_all, tgt, self.m_human)
        total = self.w["l2"] * L["l2"] + self.w["lpips"] * L["lpips"]
        if self.g is not None:                                   # RGB SDS from a random orbit view
            g, rec = self.g, self.rec
            (az, el), (K, R, t) = rec.orbit.sample(self.rng)
            K = np.diag([self.res / rec.res, self.res / rec.res, 1.0]) @ K
            Kt = torch.tensor(K, dtype=torch.float32, device=self.dev)
            pose, seg, _, _ = g.cond_images(K, R, t)
            P = {k: (f"{p}, {view_suffix(az)}", n) for k, (p, n) in g.prompts.items()}
            def term(w, names, prompt, cond, anchor=False):   # one diffusion graph alive at a time (8 GB)
                bg = self.bg(R)                                # adaptive background (per term: own graph)
                rgb, mask, _ = self.render(names, Kt, R, t, bg=bg)
                loss = g.sds(rgb.permute(2, 0, 1)[None], *prompt, cond, guidance=self.cfg_scale)
                extra = 0
                if anchor:
                    gl = self.last_garment
                    a = 0
                    for k, mean in self.garment_mean.items():
                        sel = (mask > 0.5) & ((gl - k).abs() < 0.5)
                        if sel.sum() > 50:
                            a = a + (rgb[sel].mean(0) - mean).pow(2).sum()
                    if torch.is_tensor(a):
                        L["anchor"] = a
                        extra = self.w_anchor * a
                total_t = w * loss + extra
                if total_t.requires_grad:                      # nothing of these meshes in view: skip
                    total_t.backward()
                return loss.detach()

            L["sds_cloth"] = term(self.w["sds_cloth"], ("cloth",), P["cloth"], {CN_SEG: seg}, anchor=True)
            L["sds_human"] = (term(self.w["sds_human"], ("body",), P["body"], {CN_OPENPOSE: pose})
                              + term(self.w["sds_human"], ("cloth", "body"), P["human"], {CN_OPENPOSE: pose}))
        total.backward()
        self.opt.step()
        self.sched.step()
        return {k: float(v.detach()) for k, v in L.items()}

    @torch.no_grad()
    def vertex_colors(self):
        """Per-vertex colours for export (GLB/PLY): the colour field at each canonical vertex."""
        return {n: (self.nets[n](Vc).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                for n, (_, _, Vc) in self.mesh.items()}
