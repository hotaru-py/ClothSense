"""DeClotH step (1): 3D templates.

BodyNet  - paper: PIXIE modified to output SMPL+H. Ours: Multi-HMR -> SMPL-X (declo.hmr), which
           contains SMPL+H's body and hands and takes the camera intrinsics as input.
ClothNet - paper and ours: ClothWild (Moon et al., ECCV 2022) -> SMPLicit garments. ClothWild is built
           on SMPL, so the SMPL-X body is converted first (to_smpl): same 21 body-joint rotations, SMPL
           shape fitted to the SMPL-X rest mesh, translation matched at the pelvis.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from . import hmr as B

from .frame import K_original_from_hmr

ROOT = Path(__file__).resolve().parents[1]
CLOTHWILD = ROOT / "work/ClothWild"


class BodyNet:
    def __init__(self, device=None, smplx_root=ROOT / "models", hmr_repo=ROOT / "work/multi-hmr",
                 hmr_name="multiHMR_896_L"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.smplx = B.load_smplx(smplx_root, self.device)
        self.hmr = B.MultiHMR(hmr_repo, smplx_root, hmr_name, self.device)

    def __call__(self, pil, person_mask=None, focal_px=None, hfov="auto") -> dict:
        """pil: upright photo. focal_px: original-image focal (EXIF); else hfov (deg or "auto")."""
        focal, src, fov_scores = focal_px or B.focal_px_from_exif(pil), "EXIF", None
        if focal is None:
            if hfov == "auto":
                assert person_mask is not None, "auto-FOV needs the person mask"
                hfov, fov_scores = B.estimate_hfov(self.hmr, self.smplx, pil, person_mask)
                src = f"auto-FOV {hfov:g} deg"
            else:
                src = f"assumed HFOV {float(hfov):g} deg"
            focal = B.focal_px_from_hfov(pil.size[0], float(hfov))
        hmr = self.hmr.predict(pil, focal_px=focal)
        hmr["shape"] = np.asarray(hmr["shape"]).reshape(-1)[:10]
        R, c, _ = B.camera_from_hmr(self.smplx, hmr)
        canon = B.canonical_apose(self.smplx, hmr["shape"], scale=1.0)
        return {"V_cam": hmr["v3d"].astype(np.float64), "F": self.smplx.faces.astype(np.int64),
                "K_orig": K_original_from_hmr(hmr), "focal_px": float(focal), "focal_src": src,
                "fov_scores": fov_scores, "hmr": hmr, "R": R, "c": c, "betas": hmr["shape"], "canon": canon}


def align_body(body: dict, human_mask: np.ndarray, labels: np.ndarray, K: np.ndarray, steps=120, res=256,
               log=print) -> dict:
    """Shift the fitted body (3D translation in the camera frame) so it sits in the photo: penalise body pixels
    outside the person mask and visible skin (face, arms, legs) the body does not cover. Clothes cannot bias it
    (the naked body only has to fit inside the clothed outline). Multi-HMR's mesh was off by a few cm sideways on
    Sachin's photo (13 % of the mask area outside), and everything built from it - cloth template, garment region,
    template losses - inherited the offset (shoulder flaps). Updates body["V_cam"] and the rigid map c in place."""
    import cv2
    from . import render
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    s_ = res / human_mask.shape[0]
    Kt = torch.tensor(np.diag([s_, s_, 1.0]) @ K, dtype=torch.float32, device=dev)
    rs = lambda m: torch.tensor(cv2.resize(m.astype(np.float32), (res, res), interpolation=cv2.INTER_AREA), device=dev)
    H = rs(human_mask > 0)
    skin = rs(np.isin(labels, [11, 12, 13, 14, 15]))              # face, legs, arms (SegFormer ATR)
    V = torch.tensor(body["V_cam"], dtype=torch.float32, device=dev)
    F = torch.tensor(body["F"], dtype=torch.long, device=dev)
    d = torch.zeros(3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([d], lr=2e-3)

    def losses():
        m = render.render(V + d, F, Kt, res)["mask"]
        out = (m * (1 - H)).sum() / H.sum()
        uncovered = (skin * (1 - m)).sum() / skin.sum().clamp_min(1)
        return m, out, uncovered

    with torch.no_grad():
        m0, o0, u0 = losses()
    for _ in range(steps):
        _, out, unc = losses()
        loss = out + unc + 10 * d[2] ** 2                            # depth changes the size: keep it modest
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        m1, o1, u1 = losses()
    shift = d.detach().cpu().numpy().astype(np.float64)
    iou = lambda m: ((m > 0.5) & (H > 0.5)).sum().item() / ((m > 0.5) | (H > 0.5)).sum().item()
    log(f"align_body: shift {np.round(1000 * shift, 1)} mm; body outside the person mask {100 * o0:.1f} -> {100 * o1:.1f} %, "
        f"visible skin uncovered {100 * u0:.1f} -> {100 * u1:.1f} %; IoU {iou(m0):.3f} -> {iou(m1):.3f}")
    body["V_cam"] = body["V_cam"] + shift
    body["c"] = body["c"] + shift
    body["align_shift"] = shift
    return body


# --------------------------------------------------------------------------- SMPL-X -> SMPL

def load_smpl(model_root=ROOT / "models", device="cpu", gender="neutral"):
    """models/smpl/SMPL_NEUTRAL.pkl (tools/fetch_mpi.py)."""
    import smplx
    return smplx.create(str(model_root), model_type="smpl", gender=gender).to(device)


def to_smpl(smplx_model, smpl_model, body: dict, steps=300, log=print) -> dict:
    """SMPL parameters equivalent to the Multi-HMR SMPL-X body, in the same camera frame.

    pose : SMPL joints 1-21 = SMPL-X body joints 1-21 (same kinematic tree); SMPL hand joints 22-23 = 0.
    shape: 10 SMPL betas fitted in the rest pose to the SMPL-X rest mesh (symmetric chamfer + the 22
           shared body joints), with a free offset: the two rest poses sit ~16 cm apart (pelvis y -0.37 vs
           -0.22); without it the betas absorbed the offset (|beta| 12, 74 mm surface error -> 11 mm, |beta| 1.5).
           The two models have different vertex orders, hence chamfer, not L2.
    trans: v_cam = R v_local + c for SMPL-X; SMPL rotates about its own pelvis J0, so
           trans = R J0_smplx + c - J0_smpl.
    """
    dev = smpl_model.shapedirs.device
    hmr = body["hmr"]
    with torch.no_grad():
        ox = B.smplx_forward(smplx_model, body["betas"])
        Vx = ox.vertices[0].to(dev)
        Jx = ox.joints[0, :22].to(dev)
    beta = torch.zeros(1, 10, device=dev, requires_grad=True)
    off = torch.zeros(1, 3, device=dev, requires_grad=True)       # rest-pose offset, only for the shape fit
    opt = torch.optim.Adam([beta, off], lr=0.05)
    zero = torch.zeros(1, 69, device=dev)
    for i in range(steps):
        o = smpl_model(betas=beta, body_pose=zero, global_orient=torch.zeros(1, 3, device=dev))
        Vs, Js = o.vertices[0] + off, o.joints[0, :22] + off
        d = torch.cdist(Vs, Vx)
        cham = d.min(1).values.pow(2).mean() + d.min(0).values.pow(2).mean()
        loss = cham + (Js - Jx).pow(2).sum(1).mean() + 1e-5 * beta.pow(2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        o = smpl_model(betas=beta, body_pose=zero, global_orient=torch.zeros(1, 3, device=dev))
        d = torch.cdist(o.vertices[0] + off, Vx)
        err = 0.5 * (d.min(1).values.mean() + d.min(0).values.mean()).item()
        J0s = o.joints[0, 0].cpu().numpy().astype(np.float64)
    log(f"to_smpl: SMPL shape fit, mean surface distance {1000 * err:.1f} mm")
    rv = np.asarray(hmr["rotvec"], np.float64)
    pose = np.zeros((24, 3))
    pose[:22] = rv[:22]
    J0x = Jx[0].cpu().numpy().astype(np.float64)
    trans = body["R"] @ J0x + body["c"] - J0s
    return {"pose": pose.reshape(-1), "shape": beta.detach().cpu().numpy().reshape(-1).astype(np.float64),
            "trans": trans, "fit_mm": 1000 * err}


def smpl_vertices(smpl_model, sp: dict) -> np.ndarray:
    dev = smpl_model.shapedirs.device
    t = lambda a: torch.tensor(np.asarray(a)[None], dtype=torch.float32, device=dev)
    with torch.no_grad():
        o = smpl_model(betas=t(sp["shape"]), global_orient=t(sp["pose"][:3]), body_pose=t(sp["pose"][3:]),
                       transl=t(sp["trans"]))
    return o.vertices[0].cpu().numpy().astype(np.float64)


# --------------------------------------------------------------------------- ClothNet

class _isolated_imports:
    """Import a repo whose top-level module names clash with already-imported ones (ClothWild vs Multi-HMR:
    both have `model` and `utils`). Inside: the clashing modules are hidden and the repo dirs lead sys.path;
    on exit the repo's modules are removed again and the previous ones restored (objects keep their refs)."""

    def __init__(self, names, paths):
        self.names, self.paths = tuple(names), [str(p) for p in paths]

    def _ours(self, k):
        return any(k == n or k.startswith(n + ".") for n in self.names)

    def __enter__(self):
        self.saved = {k: sys.modules.pop(k) for k in list(sys.modules) if self._ours(k)}
        self.old_path = list(sys.path)
        sys.path[:0] = self.paths
        import types                                # a regular package elsewhere on sys.path (Multi-HMR's utils/
        for p in self.paths:                        # has __init__.py) beats a namespace dir that comes first
            for n in self.names:
                d = Path(p) / n
                if d.is_dir() and not (d / "__init__.py").exists() and n not in sys.modules:
                    m = types.ModuleType(n)
                    m.__path__ = [str(d)]
                    sys.modules[n] = m
        return self

    def __exit__(self, *exc):
        for k in [k for k in sys.modules if self._ours(k)]:
            sys.modules.pop(k)
        sys.modules.update(self.saved)
        sys.path[:] = self.old_path
        return False


class ClothNet:
    """ClothWild (Moon et al., ECCV 2022) -> SMPLicit cloth template, in the photo's camera frame.

    ClothWild reads the person crop (256x192) to pick the garment types and their SMPLicit codes; SMPLicit
    then builds each garment around the SMPL body given as input and adds the camera translation, so with the
    SMPL parameters from to_smpl() the garments land directly in our camera frame.
    Needs models/smpl/SMPL_{NEUTRAL,MALE,FEMALE}.pkl (tools/fetch_mpi.py smpl) and the released weights
    work/ClothWild/demo/snapshot_7.pth.tar. NOT RUN YET (SMPL files pending).

    Compatibility, applied here without editing the ClothWild repo: numpy-2 aliases (np.float/np.bool/np.int),
    torchvision's removed resnet.model_urls, and a stub for its pytorch3d demo renderer (utils.postprocessing).
    """
    TYPES = ("uppercloth", "coat", "pants", "skirts", "shoes")

    def __init__(self, device=None):
        smpl = ROOT / "models/smpl/SMPL_NEUTRAL.pkl"
        if not smpl.exists():
            raise FileNotFoundError(f"{smpl} missing: register at https://smpl.is.tue.mpg.de and run "
                                    f"tools/fetch_mpi.py smpl")
        ckpt = CLOTHWILD / "demo/snapshot_7.pth.tar"
        if not ckpt.exists():
            raise FileNotFoundError(f"{ckpt} missing (Google Drive id 1-gafc-6V1ma7L5NS1JphzbhTdO_pQFuL)")
        self._prepare_data()
        with _isolated_imports(("model", "utils", "config", "nets", "base"), [CLOTHWILD / "main", CLOTHWILD / "common"]):
            self._compat()                                       # inside: its utils.postprocessing stub must survive
            from config import cfg                               # ClothWild's main/config.py
            from model import get_model                          # (Multi-HMR also has top-level model/utils)
            from utils.preprocessing import generate_patch_image, process_bbox
            self.model = get_model("test").cuda().eval()
        self.cfg, self._patch, self._bbox = cfg, generate_patch_image, process_bbox
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)["network"]
        sd = {k[7:] if k.startswith("module") else k: v for k, v in sd.items()}
        missing, _ = self.model.load_state_dict(sd, strict=False)
        missing = [k for k in missing if not k.startswith(("smpl_layer", "smplicit_layer"))]
        if missing:
            raise RuntimeError(f"ClothWild checkpoint: {len(missing)} missing keys, e.g. {missing[:3]}")

    @staticmethod
    def _prepare_data():
        """ClothWild/SMPLicit expect data/base_data/human_models/smpl/SMPL_*.pkl and data/base_data/smplicit/."""
        import shutil
        hm = CLOTHWILD / "data/base_data/human_models/smpl"
        hm.mkdir(parents=True, exist_ok=True)
        for g in ("NEUTRAL", "MALE", "FEMALE"):
            src, dst = ROOT / f"models/smpl/SMPL_{g}.pkl", hm / f"SMPL_{g}.pkl"
            if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
                shutil.copy2(src, dst)
        si = CLOTHWILD / "data/base_data/smplicit"
        for sub in ("checkpoints", "clusters"):
            if not (si / sub).exists():
                shutil.copytree(ROOT / "work/SMPLicit" / sub, si / sub)

    @staticmethod
    def _compat():
        import types
        for name, val in (("float", float), ("bool", bool), ("int", int)):
            if name not in np.__dict__:
                setattr(np, name, val)
        import torchvision.models.resnet as tvr
        if not hasattr(tvr, "model_urls"):
            tvr.model_urls = {}
        if "utils.postprocessing" not in sys.modules:            # demo renderer needs pytorch3d; unused here
            stub = types.ModuleType("utils.postprocessing")
            stub.renderer = None
            stub.merge_mesh = stub.rasterize_mesh_given_cam_param = stub.save_proj_faces = None
            stub.read_valid_point = stub.pa_mpjpe = stub.pairwise_distances = None
            sys.modules["utils.postprocessing"] = stub

    def _patch_img(self, rgb_full, person_mask):
        ys, xs = np.nonzero(person_mask)
        H, W = person_mask.shape
        bbox = self._bbox(np.array([xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min()], np.float32), W, H)
        patch, _, _, _ = self._patch(rgb_full.astype(np.float32), bbox, 1.0, 0.0, False, self.cfg.input_img_shape)
        return torch.from_numpy(patch).permute(2, 0, 1)[None].cuda() / 255.0

    @torch.no_grad()
    def gender(self, rgb_full: np.ndarray, person_mask: np.ndarray) -> str:
        """ClothWild builds the garments on the SMPL of the gender it predicts (male 1 / female 2), so the SMPL
        shape must be fitted on that model (to_smpl), not on the neutral one (42-73 mm off otherwise)."""
        g, _, _, _ = self.model.cloth_net(self.model.backbone(self._patch_img(rgb_full, person_mask)))
        return ("male", "female")[int(torch.argmax(g[0]))]

    @torch.no_grad()
    def __call__(self, rgb_full: np.ndarray, person_mask: np.ndarray, sp: dict) -> dict:
        """rgb_full (H,W,3) uint8 photo, person_mask (H,W), sp = to_smpl(...) result.
        Returns {"garments": {type: (V (n,3) camera frame, F)}, "scores", "gender", "smpl_mesh" (ClothWild's
        own SMPL in the camera frame, for an alignment check)}."""
        img = self._patch_img(rgb_full, person_mask)
        t = lambda a: torch.tensor(np.asarray(a, np.float32)[None]).cuda()
        meta = {"smpl_pose": t(sp["pose"]), "smpl_shape": t(sp["shape"]), "cam_trans": t(sp["trans"])}
        out = self.model({"img": img}, {}, meta, "test")
        garments = {}
        for k in self.TYPES:
            m = out[f"{k}_mesh"][0]
            if m is not None:
                garments[k] = (np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64))
        return {"garments": garments, "valid": out["pred_clothes"][0].cpu().numpy().tolist(),
                "gender": int(out["pred_gender"][0]), "smpl_mesh": out["smpl_mesh"][0].cpu().numpy()}

    def free(self):
        self.model = None
        torch.cuda.empty_cache()


def cloth_patch(body_V_cam: np.ndarray, garments: dict, smplx_model, within=0.05) -> np.ndarray:
    """Which SMPL-X vertices are under the ClothWild garments (not shoes): a garment surface within `within`
    of the body vertex. Feeds the cloth DMTet's garment region (recon.Reconstruction(cloth_patch=...))."""
    from scipy.spatial import cKDTree
    pts = [V for k, (V, F) in garments.items() if k != "shoes"]
    if not pts:
        return np.zeros(len(body_V_cam), bool)
    d, _ = cKDTree(np.concatenate(pts)).query(body_V_cam, workers=-1)
    Wl = smplx_model.lbs_weights.cpu().numpy()
    extremity = Wl[:, [7, 8, 10, 11, 12, 15, 20, 21, 22, 23, 24] + list(range(25, 55))].sum(1) > 0.5
    return (d < within) & ~extremity
