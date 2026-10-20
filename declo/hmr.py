"""Body template plumbing: Multi-HMR -> SMPL-X, camera, canonical A-pose (ported from the v1 code in bak/).

* MultiHMR       : Multi-HMR (Baradel et al., ECCV 2024) single-shot SMPL-X regression; takes the camera
                   intrinsics as input, so the focal length matters (EXIF or estimate_hfov).
* camera_from_hmr: rigid map from the zero-orientation SMPL-X body to Multi-HMR's camera frame.
* canonical_apose: the person's body in the canonical frame the DMTets live in (A-pose, y-up, feet at y=0,
                   facing +z, torso centred; the frame GarmentCode garments are placed in, kept for sizing later).
* estimate_hfov  : camera FOV for photos without EXIF focal length (mesh-vs-mask IoU over 35-95 deg).
Needs models/smplx/SMPLX_NEUTRAL.npz (free registration at https://smpl-x.is.tue.mpg.de). Units: metres.
"""
from __future__ import annotations

import math
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch

REF_ARM_ANGLE_DEG = 45.48

REF_TORSO_CENTER_XZ = (0.0, 0.01)

SMPLX_JOINT = {
    "pelvis": 0, "left_hip": 1, "right_hip": 2, "spine1": 3, "left_knee": 4, "right_knee": 5,
    "spine2": 6, "left_ankle": 7, "right_ankle": 8, "spine3": 9, "left_foot": 10,
    "right_foot": 11, "neck": 12, "left_collar": 13, "right_collar": 14, "head": 15,
    "left_shoulder": 16, "right_shoulder": 17, "left_elbow": 18, "right_elbow": 19,
    "left_wrist": 20, "right_wrist": 21,
}

ARM_JOINTS = [16, 17, 18, 19, 20, 21] + list(range(25, 55))

MULTIHMR_URLS = [
    "https://download.europe.naverlabs.com/ComputerVision/MultiHMR/{name}.pt",
    "http://download.europe.naverlabs.com/multihmr/{name}.pt",
]

MEAN_PARAMS_URL = ("https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/"
                   "smpl_mean_params.npz?versionId=CAEQHhiBgICN6M3V6xciIDU1MzUzNjZjZGNiOTQ3OWJiZTJmNThiZmY4NmMxMTM4")



def focal_px_from_exif(pil_img) -> float | None:
    """Focal length in pixels of the ORIGINAL image from EXIF FocalLengthIn35mmFilm.

    35 mm-equivalent focal length is defined against the 36x24 mm frame diagonal (43.27 mm),
    so f_px = f35 * image_diagonal_px / 43.27 (correct for any aspect ratio).
    """
    try:
        exif = pil_img.getexif()
        f35 = exif.get_ifd(0x8769).get(41989) or exif.get(41989)
    except Exception:
        f35 = None
    if not f35:
        return None
    w, h = pil_img.size
    return float(f35) * math.hypot(w, h) / math.hypot(36.0, 24.0)

def focal_px_from_hfov(width_px: int, hfov_deg: float) -> float:
    return width_px / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))

def load_smplx(model_root, device="cpu", num_betas=10):
    """model_root must contain smplx/SMPLX_NEUTRAL.npz (same flags as Multi-HMR uses)."""
    import smplx
    return smplx.create(str(model_root), model_type="smplx", gender="neutral", use_pca=False,
                        flat_hand_mean=True, num_betas=num_betas).to(device)

def _zeros(model, n, b=1):
    return torch.zeros(b, n, device=model.shapedirs.device)

def smplx_forward(model, betas, body_pose=None, expression=None):
    """SMPL-X with zero global orientation (Multi-HMR applies the root rotation itself)."""
    dev = model.shapedirs.device
    betas = torch.as_tensor(betas, dtype=torch.float32, device=dev).reshape(1, -1)
    bp = _zeros(model, 63) if body_pose is None else torch.as_tensor(
        body_pose, dtype=torch.float32, device=dev).reshape(1, 63)
    ex = _zeros(model, 10) if expression is None else torch.as_tensor(
        expression, dtype=torch.float32, device=dev).reshape(1, -1)
    return model(betas=betas, body_pose=bp, global_orient=_zeros(model, 3),
                 expression=ex, jaw_pose=_zeros(model, 3), leye_pose=_zeros(model, 3),
                 reye_pose=_zeros(model, 3), left_hand_pose=_zeros(model, 45),
                 right_hand_pose=_zeros(model, 45))

def apose_body_pose(model, betas, arm_angle_deg=REF_ARM_ANGLE_DEG, iters=6):
    """body_pose (63,) that lowers both arms to `arm_angle_deg` below horizontal.

    Solved numerically (rotate the shoulders about the forward axis and re-measure the
    shoulder->wrist direction) so it does not depend on SMPL-X's rest-pose arm angle.
    """
    def angles(bp):
        J = smplx_forward(model, betas, bp.reshape(-1)).joints[0].detach().cpu().numpy()
        out = []
        for sh, wr in ((16, 20), (17, 21)):
            d = J[wr] - J[sh]
            out.append(math.degrees(math.atan2(-d[1], abs(d[0]))))   # degrees below horizontal
        return np.array(out)

    bp = np.zeros((21, 3))
    idx = (15, 16)                      # body_pose rows of left / right shoulder (joints 16, 17)
    for _ in range(iters):              # Newton steps with a numerical derivative per arm
        cur = angles(bp)
        if np.all(np.abs(cur - arm_angle_deg) < 0.5):
            break
        eps = 0.05
        for k, row in enumerate(idx):
            bp2 = bp.copy(); bp2[row, 2] += eps
            slope = (angles(bp2)[k] - cur[k]) / eps
            if abs(slope) > 1e-3:
                bp[row, 2] += (arm_angle_deg - cur[k]) / slope
    print(f"A-pose arm angles below horizontal: {np.round(angles(bp), 1)} (target {arm_angle_deg})")
    return bp.reshape(-1)

def canonical_apose(model, betas, scale=1.0, arm_angle_deg=REF_ARM_ANGLE_DEG):
    """Body in the canonical frame: A-pose, y-up, feet at y=0, facing +z, torso centred.

    Returns dict(V, F, joints{name: xyz}, body_pose, offset, scale, arm_mask).
    `scale` is the metric correction (1.0 = trust the regressor's absolute size).
    """
    bp = apose_body_pose(model, betas, arm_angle_deg)
    out = smplx_forward(model, betas, bp)
    V = out.vertices[0].detach().cpu().numpy().astype(np.float64) * scale
    J = out.joints[0].detach().cpu().numpy().astype(np.float64) * scale
    F = model.faces.astype(np.int64)
    offset = np.array([-J[0, 0], -V[:, 1].min(), 0.0])
    V, J = V + offset, J + offset
    # centre the torso at chest height on the reference torso centre: centroid of the torso vertices in a 2 cm
    # band (v1 used the girth loop of its measurement code; same reference point, no slicing needed)
    W = model.lbs_weights.detach().cpu().numpy()
    arm_mask = W[:, ARM_JOINTS].sum(1) > 0.5
    y_chest = 0.5 * (J[SMPLX_JOINT["spine2"], 1] + J[SMPLX_JOINT["spine3"], 1])
    band = (np.abs(V[:, 1] - y_chest) < 0.01) & ~arm_mask
    if band.any():
        c = V[band][:, [0, 2]].mean(0)
        shift = np.array([REF_TORSO_CENTER_XZ[0] - c[0], 0.0, REF_TORSO_CENTER_XZ[1] - c[1]])
        V, J, offset = V + shift, J + shift, offset + shift
    joints = {k: J[i] for k, i in SMPLX_JOINT.items()}
    joints["head_top"] = V[V[:, 1].argmax()]
    return {"V": V, "F": F, "joints": joints, "body_pose": bp, "offset": offset,
            "scale": scale, "arm_mask": arm_mask, "betas": np.asarray(betas, float).reshape(-1)}

def _download(urls, dst):
    dst = Path(dst)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for u in urls:
        try:
            print(f"downloading {u}")
            urllib.request.urlretrieve(u, dst)
            if dst.stat().st_size > 0:
                return dst
        except Exception as e:     # try the next mirror
            last = e
    raise RuntimeError(f"could not download {dst.name}: {last}")

class MultiHMR:
    """Thin wrapper around naver/multi-hmr that avoids demo.py (which imports optional deps)."""

    def __init__(self, repo_dir, smplx_root, model_name="multiHMR_896_L", device=None):
        if sys.platform.startswith("linux"):          # headless Linux (Kaggle); Windows uses WGL
            os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.repo = Path(repo_dir).resolve()
        models = self.repo / "models"
        (models / "smplx").mkdir(parents=True, exist_ok=True)
        link = models / "smplx" / "SMPLX_NEUTRAL.npz"
        src = Path(smplx_root) / "smplx" / "SMPLX_NEUTRAL.npz"
        if not link.exists():
            assert src.exists(), f"SMPL-X model not found at {src}"
            try:
                os.symlink(src.resolve(), link)
            except OSError:                           # Windows without developer mode
                import shutil
                shutil.copy2(src, link)
        _download([MEAN_PARAMS_URL], models / "smpl_mean_params.npz")
        ckpt = _download([u.format(name=model_name) for u in MULTIHMR_URLS],
                         models / "multiHMR" / f"{model_name}.pt")

        cwd = os.getcwd()
        sys.path.insert(0, str(self.repo))
        os.chdir(self.repo)                       # Multi-HMR resolves 'models/...' relatively
        try:
            from model import Model
            from utils import normalize_rgb
            self._normalize_rgb = normalize_rgb
            ck = torch.load(ckpt, map_location=self.device, weights_only=False)
            kwargs = dict(vars(ck["args"]))
            kwargs["type"] = ck["args"].train_return_type
            kwargs["img_size"] = ck["args"].img_size[0]
            self.model = Model(**kwargs).to(self.device).eval()
            self.model.load_state_dict(ck["model_state_dict"], strict=False)
        finally:
            os.chdir(cwd)
        self.img_size = int(self.model.img_size)

    def letterbox(self, pil_img):
        """Resize longest side to img_size and centre-pad to a square (Multi-HMR's input)."""
        from PIL import Image
        W, H = pil_img.size
        s = self.img_size / max(W, H)
        w, h = round(W * s), round(H * s)
        canvas = Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))
        ox, oy = (self.img_size - w) // 2, (self.img_size - h) // 2
        canvas.paste(pil_img.convert("RGB").resize((w, h)), (ox, oy))
        return canvas, {"scale": s, "ox": ox, "oy": oy, "W": W, "H": H}

    @torch.no_grad()
    def predict(self, pil_img, focal_px=None, hfov_deg=60.0, det_thresh=0.3):
        """Returns the most confident person. focal_px is for the ORIGINAL image."""
        sq, lb = self.letterbox(pil_img)
        f = (focal_px * lb["scale"]) if focal_px else focal_px_from_hfov(round(lb["W"] * lb["scale"]), hfov_deg)
        S = self.img_size
        K = torch.tensor([[f, 0, S / 2], [0, f, S / 2], [0, 0, 1]], dtype=torch.float32,
                         device=self.device)[None]
        x = torch.from_numpy(self._normalize_rgb(np.asarray(sq))).unsqueeze(0).to(self.device)
        with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
            humans = self.model(x, is_training=False, nms_kernel_size=3, det_thresh=det_thresh, K=K)
        if not humans:
            raise RuntimeError("no person detected - stand fully in frame, plain background helps")
        best = max(humans, key=lambda h: float(h["scores"]))
        out = {k: (v.float().cpu().numpy() if torch.is_tensor(v) else v) for k, v in best.items()}
        out["K"] = K[0].cpu().numpy()
        out["letterbox"] = lb
        out["square_image"] = sq
        return out

def camera_from_hmr(model, hmr: dict):
    """Rigid map local-SMPL-X -> Multi-HMR camera frame: v_cam = R v_local + c.

    Multi-HMR rotates the zero-orient SMPL-X body by R(root rotvec) and anchors it at a
    person-centre joint; we recover the anchor offset c from its own vertices so the
    convention does not have to be hard-coded. Prints the residual as a sanity check.
    """
    import roma
    R = roma.rotvec_to_rotmat(torch.tensor(hmr["rotvec"][0])).numpy().astype(np.float64)
    rv = hmr["rotvec"]
    body_pose = rv[1:22].reshape(-1)
    out = model(betas=torch.tensor(hmr["shape"][None], dtype=torch.float32, device=model.shapedirs.device),
                body_pose=torch.tensor(body_pose[None], dtype=torch.float32, device=model.shapedirs.device),
                global_orient=_zeros(model, 3),
                left_hand_pose=torch.tensor(rv[22:37].reshape(1, -1), dtype=torch.float32, device=model.shapedirs.device),
                right_hand_pose=torch.tensor(rv[37:52].reshape(1, -1), dtype=torch.float32, device=model.shapedirs.device),
                jaw_pose=torch.tensor(rv[52:53].reshape(1, -1), dtype=torch.float32, device=model.shapedirs.device),
                expression=torch.tensor(hmr["expression"].reshape(1, -1), dtype=torch.float32, device=model.shapedirs.device),
                leye_pose=_zeros(model, 3), reye_pose=_zeros(model, 3))
    v_local = out.vertices[0].detach().cpu().numpy().astype(np.float64)
    c = (hmr["v3d"] - v_local @ R.T).mean(0)
    resid = np.abs(v_local @ R.T + c - hmr["v3d"]).max()
    print(f"camera_from_hmr: max residual {1000 * resid:.2f} mm (should be ~0)")
    return R, c, v_local

def estimate_hfov(hmr: MultiHMR, smplx_model, pil, person_mask, coarse=(35, 45, 55, 65, 75, 85, 95), fine_step=2.5):
    """Camera FOV for a photo without EXIF focal length: the HFOV whose Multi-HMR mesh, projected with that
    camera, best overlaps the person mask (IoU). Coarse grid, then +/- 2 fine steps. Multi-HMR takes K as input
    and its depth/size trade-off only lines up with the image near the true focal; the optimum is broad
    (treat the value as +/- ~5 deg). Returns (hfov_deg, {hfov: IoU})."""
    import cv2
    W, H = pil.size
    s = 400.0 / H                                            # compare at 400 px height
    mask = cv2.resize((person_mask > 0).astype(np.uint8), (round(W * s), round(H * s)),
                      interpolation=cv2.INTER_NEAREST).astype(bool)
    F = smplx_model.faces.astype(np.int64)
    cache = {}

    def iou(h):
        if h not in cache:
            hm = hmr.predict(pil, focal_px=focal_px_from_hfov(W, h))
            uvw = hm["v3d"] @ np.asarray(hm["K"]).T
            uv = uvw[:, :2] / uvw[:, 2:3]
            lb = hm["letterbox"]
            uv = np.stack([(uv[:, 0] - lb["ox"]) / lb["scale"], (uv[:, 1] - lb["oy"]) / lb["scale"]], 1)
            pts = np.round(uv * s).astype(np.int32)
            sil = np.zeros(mask.shape, np.uint8)
            for tri in pts[F]:
                cv2.fillConvexPoly(sil, tri, 1)
            sil = sil.astype(bool)
            cache[h] = float((sil & mask).sum() / max((sil | mask).sum(), 1))
        return cache[h]

    best = max(coarse, key=iou)
    for h in (best - 2 * fine_step, best - fine_step, best + fine_step, best + 2 * fine_step):
        if 25 <= h <= 110:
            iou(h)
    best = max(cache, key=cache.get)
    return best, {k: round(v, 3) for k, v in sorted(cache.items())}
