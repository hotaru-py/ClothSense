"""DeClotH step (9) / M6: separate cloth + body meshes -> re-posing and garment swap onto another person.

A reconstruction is saved as a *bundle* (canonical cloth and body meshes, their colours, the SMPL-X parameters,
camera and photo crop). `export_garment` makes a shareable garment-only file from it (cloth on the zero-shape body,
no photo, no person-specific body). Garment swap A -> B (paper: "try-on onto another body"):
  1. A's cloth lives in A's canonical A-pose. Every cloth point moves with its nearest canonical body vertices
     (k-NN, inverse-distance weights): x_B = x_A + sum_k w_k (V_B[i_k] - V_A[i_k]). Both bodies are SMPL-X, so
     vertex i is the same body point on both; this is the shape-displacement transfer SMPL's LBS implies.
  2. Points that end up inside B's body (B is bigger there) are pushed out along B's normal to a small gap.
  3. B's skinning poses it into B's photo (or any pose via Skinning.set_pose).
Loose garments keep their absolute offset from the body, so a shirt that is loose on A is less loose on a bigger B.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from . import dmtet, render
from .skinning import Skinning


# --------------------------------------------------------------------------- bundles

def save_bundle(path, prep: dict, rec, colors: dict | None = None):
    """Everything needed to re-pose / swap / render one reconstruction later (no GPU models needed to load it)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    b = prep["body"]
    out = {"rgb": prep["rgb"], "K": prep["K"], "smplx_canon_V": b["canon"]["V"], "smplx_F": b["F"],
           "betas": b["betas"], "rotvec": b["hmr"]["rotvec"], "expression": b["hmr"]["expression"],
           "R": b["R"], "c": b["c"], "V_cam": b["V_cam"], "canon_body_pose": b["canon"]["body_pose"],
           "canon_offset": b["canon"]["offset"], "canon_scale": np.float64(b["canon"]["scale"])}
    for name, (V, F) in rec.meshes(posed=False).items():
        out[f"{name}_V"], out[f"{name}_F"] = V, F
        if colors and name in colors:
            out[f"{name}_colors"] = colors[name]
    np.savez_compressed(path, **out)
    (path.with_suffix(".json")).write_text(json.dumps({"captions": prep.get("captions"),
                                                        "frame": prep["frame"].to_dict()}, indent=1))
    return path


def load_bundle(path) -> dict:
    z = dict(np.load(path))
    if "garment_only" in z:                      # export_garment file: cloth on the zero-shape body, nothing else
        return z
    body = {"betas": z["betas"], "R": z["R"], "c": z["c"], "V_cam": z["V_cam"], "F": z["smplx_F"],
            "hmr": {"rotvec": z["rotvec"], "expression": z["expression"]},
            "canon": {"V": z["smplx_canon_V"], "body_pose": z["canon_body_pose"], "offset": z["canon_offset"],
                      "scale": float(z["canon_scale"])}}
    z["body"] = body
    return z


# --------------------------------------------------------------------------- garment-only files (shippable)

def neutral_canon(smplx_model, n_betas=10) -> np.ndarray:
    """Canonical A-pose SMPL-X body with zero shape (betas = 0): the body shipped garments are expressed on."""
    from .hmr import canonical_apose
    return canonical_apose(smplx_model, np.zeros(n_betas))["V"]


def export_garment(bundle: dict, smplx_model, path, caption=None):
    """Garment only: the cloth moved onto the neutral body (transfer_garment), its faces and colours, a caption and
    a thumbnail. No photo, no person-specific body (shape, texture or pose), so it can be shared."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    V0 = neutral_canon(smplx_model, len(bundle["betas"]))
    X, _ = transfer_garment(bundle["cloth_V"], bundle["smplx_canon_V"], V0, bundle["smplx_F"])
    out = {"cloth_V": X, "cloth_F": bundle["cloth_F"], "garment_only": np.array(True)}
    if "cloth_colors" in bundle:
        out["cloth_colors"] = bundle["cloth_colors"]
    np.savez_compressed(path, **out)
    path.with_suffix(".json").write_text(json.dumps({"captions": {"cloth": caption}}, indent=1))
    from PIL import Image
    Image.fromarray(garment_thumbnail(out)).save(path.with_suffix(".png"))
    return path


def garment_thumbnail(bundle: dict, res=256) -> np.ndarray:
    """Front view of the canonical garment (y up, facing +z), coloured, on white: (res, res, 4) RGBA uint8."""
    V = np.asarray(bundle["cloth_V"], np.float64)
    lo, hi = V.min(0), V.max(0)
    c = 0.5 * (lo + hi)
    size = max(hi[0] - lo[0], hi[1] - lo[1]) * 1.1
    d = 3.0
    Vc = np.stack([V[:, 0] - c[0], -(V[:, 1] - c[1]), d - (V[:, 2] - c[2])], 1)   # camera in front, OpenCV axes
    f = res * d / size
    K = np.array([[f, 0, res / 2], [0, f, res / 2], [0, 0, 1.0]])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t = lambda a, dt=torch.float32: torch.tensor(np.asarray(a), dtype=dt, device=dev)
    cols = bundle.get("cloth_colors")
    with torch.no_grad():
        o = render.render(t(Vc), t(bundle["cloth_F"], torch.long), t(K), res,
                          colors=None if cols is None else t(cols)[:, :3] / 255.0)
    img = o["rgb"] if cols is not None else (o["normal"] + 1) * 0.5
    rgba = torch.cat([img.clamp(0, 1), o["mask"][..., None]], -1)
    return (rgba.cpu().numpy() * 255).astype(np.uint8)


# --------------------------------------------------------------------------- transfer

def transfer_garment(X_A, Vbody_A, Vbody_B, F_body, k=4, gap=0.003):
    """Canonical cloth points of person A -> person B's canonical space (see module docstring).
    Vbody_*: the two SMPL-X canonical bodies (same topology). Returns X_B and how many points were pushed out."""
    from scipy.spatial import cKDTree
    import trimesh
    d, nn = cKDTree(Vbody_A).query(X_A, k=k, workers=-1)
    w = 1.0 / (d + 1e-4)
    w /= w.sum(1, keepdims=True)
    X_B = X_A + np.einsum("nk,nkj->nj", w, Vbody_B[nn] - Vbody_A[nn])
    Vb, Fb = dmtet.main_component(Vbody_B, F_body)
    sdf = dmtet.mesh_sdf(X_B, Vb, Fb).numpy()
    inside = sdf < gap
    if inside.any():
        nB = trimesh.Trimesh(Vb, Fb, process=False).vertex_normals
        _, j = cKDTree(Vb).query(X_B[inside], workers=-1)
        X_B[inside] += (gap - sdf[inside])[:, None] * nB[j]
    return X_B, int(inside.sum())


def pose(smplx_model, bundle: dict, X_canon: np.ndarray, rotvec=None) -> np.ndarray:
    """Canonical points (in this bundle's canonical frame) -> its photo camera frame (or another pose)."""
    sk = Skinning(smplx_model, bundle["body"])
    if rotvec is not None:
        sk.set_pose(rotvec)
    X = torch.tensor(X_canon, dtype=torch.float32, device=sk.device)
    return sk.to_cam(X, sk.weights(X)).detach().cpu().numpy()


def try_on(smplx_model, A: dict, B: dict, res=512):
    """A's cloth on B's body, in B's photo pose. Returns posed cloth (V,F), B's posed body (V,F) and an
    overlay image on B's photo crop (cloth in A's colours if A has them, else normals)."""
    VA = A["smplx_canon_V"] if "smplx_canon_V" in A else neutral_canon(smplx_model, len(B["betas"]))  # garment-only file
    X_B, pushed = transfer_garment(A["cloth_V"], VA, B["smplx_canon_V"], B["smplx_F"])
    Vc = pose(smplx_model, B, X_B)
    Vb = pose(smplx_model, B, B["body_V"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=dev)
    K = t(np.diag([res / B["rgb"].shape[0], res / B["rgb"].shape[0], 1.0]) @ B["K"])
    Vall = torch.cat([t(Vc), t(Vb)])
    Fall = torch.cat([t(A["cloth_F"], torch.long), t(B["body_F"], torch.long) + len(Vc)])
    cols = None
    if "cloth_colors" in A and "body_colors" in B:
        cols = torch.cat([t(A["cloth_colors"]), t(B["body_colors"])]) / 255.0
    with torch.no_grad():
        o = render.render(Vall, Fall, K, res, colors=cols)
    img = o["rgb"] if cols is not None else (o["normal"] + 1) * 0.5
    m = o["mask"][..., None]
    photo = torch.nn.functional.interpolate(t(B["rgb"]).permute(2, 0, 1)[None] / 255.0, size=(res, res),
                                            mode="bilinear")[0].permute(1, 2, 0)
    over = (img * m + photo * (1 - m)).clamp(0, 1).cpu().numpy()
    return {"cloth": (Vc, A["cloth_F"]), "body": (Vb, B["body_F"]), "overlay": (over * 255).astype(np.uint8),
            "pushed_out": pushed}
