"""Minimal z-buffer rasteriser in PyTorch (no gradients).

Used in M1 for previews and for the SMPL-X normal maps that ECON's normal net takes as input
(T_normal_F / T_normal_B). The optimisation (M2+) renders with nvdiffrast instead.

"Back" view = the same pixel rays, keeping the farthest surface (far=True). This is what ECON's
back normal map is: rendered from behind and mirrored so it lines up pixel-for-pixel with the photo.
"""
from __future__ import annotations

import numpy as np
import torch


def vertex_normals(V: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    fn = torch.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]], dim=1)   # area-weighted
    n = torch.zeros_like(V).index_add_(0, F.reshape(-1), fn.repeat_interleave(3, 0))
    return torch.nn.functional.normalize(n, dim=1)


@torch.no_grad()
def rasterize(V_cam, F, K, H, W, far=False, chunk=8_000_000):
    """V_cam (N,3) OpenCV camera frame, F (M,3), K (3,3). Pixel (i, j) is sampled at (j+.5, i+.5).

    Returns face (H,W) long (-1 = empty), bary (H,W,3) perspective-correct, depth (H,W) (inf = empty).
    """
    dev = V_cam.device
    V_cam, F = torch.as_tensor(V_cam, dtype=torch.float32, device=dev), torch.as_tensor(F, dtype=torch.long, device=dev)
    K = torch.as_tensor(K, dtype=torch.float32, device=dev)
    z = V_cam[:, 2]
    uv = (V_cam @ K.T)[:, :2] / z[:, None].clamp_min(1e-6)
    tri, tz = uv[F], z[F]                                                  # (M,3,2), (M,3)
    lo = tri.min(1).values - 0.5
    hi = tri.max(1).values - 0.5
    x0, y0 = lo[:, 0].ceil().clamp(0, W - 1).long(), lo[:, 1].ceil().clamp(0, H - 1).long()
    x1, y1 = hi[:, 0].floor().clamp(-1, W - 1).long(), hi[:, 1].floor().clamp(-1, H - 1).long()
    w, h = (x1 - x0 + 1).clamp_min(0), (y1 - y0 + 1).clamp_min(0)
    ok = (tz > 1e-4).all(1) & (w > 0) & (h > 0)
    fid = torch.nonzero(ok).squeeze(1)
    cnt = (w * h)[fid]

    best = torch.full((H * W,), float("-inf") if far else float("inf"), device=dev)
    keep = []                                                              # (pix, face, b0, b1, b2, depth)
    starts = torch.cumsum(cnt, 0) - cnt
    bounds = [0]
    tot = 0
    for i, c in enumerate(cnt.tolist()):                                   # chunk by pixel count
        tot += c
        if tot > chunk:
            bounds.append(i + 1)
            tot = 0
    bounds.append(len(fid))
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b <= a:
            continue
        f = fid[a:b]
        n = cnt[a:b]
        rf = torch.repeat_interleave(torch.arange(len(f), device=dev), n)
        local = torch.arange(int(n.sum()), device=dev) - (starts[a:b] - starts[a])[rf]
        ff = f[rf]
        px = x0[ff] + local % w[ff]
        py = y0[ff] + local // w[ff]
        p = torch.stack([px + 0.5, py + 0.5], 1).float()
        t = tri[ff]
        e0 = t[:, 1] - t[:, 0]
        e1 = t[:, 2] - t[:, 0]
        d = p - t[:, 0]
        den = e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0]
        good = den.abs() > 1e-12
        b1 = (d[:, 0] * e1[:, 1] - d[:, 1] * e1[:, 0]) / torch.where(good, den, torch.ones_like(den))
        b2 = (e0[:, 0] * d[:, 1] - e0[:, 1] * d[:, 0]) / torch.where(good, den, torch.ones_like(den))
        bb = torch.stack([1 - b1 - b2, b1, b2], 1)
        inside = good & (bb >= -1e-6).all(1)
        bb, ff, px, py = bb[inside], ff[inside], px[inside], py[inside]
        zinv = (bb / tz[ff]).sum(1)
        dep = 1.0 / zinv
        bb = (bb / tz[ff]) / zinv[:, None]
        pix = py * W + px
        best.scatter_reduce_(0, pix, dep, "amax" if far else "amin")
        keep.append((pix, ff, bb, dep))

    face = torch.full((H * W,), -1, dtype=torch.long, device=dev)
    bary = torch.zeros(H * W, 3, device=dev)
    for pix, ff, bb, dep in keep:
        win = dep == best[pix]
        face.scatter_reduce_(0, pix[win], ff[win], "amax")
    for pix, ff, bb, dep in keep:
        sel = (dep == best[pix]) & (ff == face[pix])
        bary[pix[sel]] = bb[sel]
    depth = torch.where(face >= 0, best, torch.full_like(best, float("inf")))
    return face.view(H, W), bary.view(H, W, 3), depth.view(H, W)


def interpolate(attr: torch.Tensor, F: torch.Tensor, face: torch.Tensor, bary: torch.Tensor) -> torch.Tensor:
    """Per-vertex attribute (N,C) -> image (H,W,C); zeros where face == -1."""
    f = face.clamp_min(0)
    out = (attr[F[f]] * bary[..., None]).sum(-2)
    return out * (face >= 0)[..., None]


@torch.no_grad()
def normal_maps(V_cam, F, K, res, frame="econ"):
    """Front and back normal maps of a mesh, (2,3,res,res) in [-1,1], 0 outside, and the masks (2,res,res).

    frame="econ": ECON's world frame (x right, y up, z towards the camera), the space its normal net
    reads and predicts. frame="cam": OpenCV camera frame.
    """
    V_cam = torch.as_tensor(V_cam, dtype=torch.float32)
    dev = V_cam.device
    F = torch.as_tensor(F, dtype=torch.long, device=dev)
    n = vertex_normals(V_cam, F)
    if frame == "econ":
        n = n * torch.tensor([1.0, -1.0, -1.0], device=dev)
    outs, masks = [], []
    for far in (False, True):
        face, bary, _ = rasterize(V_cam, F, K, res, res, far=far)
        img = torch.nn.functional.normalize(interpolate(n, F, face, bary), dim=-1) * (face >= 0)[..., None]
        outs.append(img.permute(2, 0, 1))
        masks.append(face >= 0)
    return torch.stack(outs), torch.stack(masks)


def to_rgb(nmap: torch.Tensor) -> np.ndarray:
    """(3,H,W) normals in [-1,1] -> uint8 RGB image (the usual (n+1)/2 colouring, black background)."""
    m = (nmap.abs().sum(0) > 0)
    img = ((nmap + 1) * 0.5 * m).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return (img * 255).astype(np.uint8)
