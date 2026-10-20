"""DeClotH step (3): differentiable renders (silhouette, normal, RGB) of meshes in a pinhole camera.

Two backends behind one function:
* "torch"      : works now, no compiler. The triangle per pixel comes from declo.raster (no gradient);
                 barycentrics, normals and colours are recomputed from the vertex positions with autograd; the
                 silhouette gets an analytic edge antialias (nvdiffrast's idea: where the mesh outline crosses
                 between two pixel centres, both pixels get partial coverage, with gradients to the edge's
                 vertices). Without that, a mask loss has no gradient with respect to the geometry.
* "nvdiffrast" : nvdiffrast rasterises and interpolates (CUDA, ~7x faster than "torch" with backward on a 557k-face
                 cloth); the silhouette keeps our edge antialias. nvdiffrast's own antialias needs the triangle
                 that owns a silhouette edge to cover a pixel centre: on DMTet meshes finer than the pixels it
                 left 114 of 681 edge pixels and no gradient at all on a 20k-face sphere
                 (work/declo/test_render_backends.py).

Back view ("far=True"): same pixel rays, farthest surface. ECON's back normal map is defined that way
(rendered from behind and mirrored), so photo-aligned targets and renders match pixel for pixel.
Normals are returned in ECON's frame (x right, y up, z towards the camera) unless frame="cam".
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from . import raster

_FLIP = torch.tensor([1.0, -1.0, -1.0])


def _project(V, K):
    z = V[:, 2:3]
    return (V @ K.T)[:, :2] / z.clamp_min(1e-6), V[:, 2]


def _bary(uv, F_, face, res):
    """Differentiable screen-space barycentrics of the pixel centres inside their (fixed) faces."""
    H = W = res
    idx = torch.nonzero(face >= 0)                          # (P,2) row, col
    f = face[idx[:, 0], idx[:, 1]]
    p = torch.stack([idx[:, 1] + 0.5, idx[:, 0] + 0.5], 1).to(uv.dtype)
    a, b, c = uv[F_[f, 0]], uv[F_[f, 1]], uv[F_[f, 2]]
    e0, e1, d = b - a, c - a, p - a
    den = e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0]
    den = torch.where(den.abs() < 1e-12, torch.full_like(den, 1e-12), den)
    b1 = (d[:, 0] * e1[:, 1] - d[:, 1] * e1[:, 0]) / den
    b2 = (e0[:, 0] * d[:, 1] - e0[:, 1] * d[:, 0]) / den
    return idx, f, torch.stack([1 - b1 - b2, b1, b2], 1)


def _perspective(bs, z, F_, f):
    """Screen-space -> perspective-correct barycentrics."""
    w = bs / z[F_[f]]
    return w / w.sum(1, keepdim=True)


def silhouette_edges(V, F_):
    """(M,3) bool: edge k of face f (vertices k -> k+1) is a silhouette edge seen from the camera at the origin:
    an open boundary edge, or shared by a front- and a back-facing triangle."""
    M = len(F_)
    e = torch.cat([F_[:, [0, 1]], F_[:, [1, 2]], F_[:, [2, 0]]])
    _, inv, cnt = torch.unique(torch.sort(e, 1).values, dim=0, return_inverse=True, return_counts=True)
    with torch.no_grad():
        fn = torch.cross(V[F_[:, 1]] - V[F_[:, 0]], V[F_[:, 2]] - V[F_[:, 0]], dim=1)
        front = ((fn * V[F_].mean(1)).sum(1) < 0).float()
    nfront = torch.zeros(len(cnt), device=V.device).index_add_(0, inv, front.repeat(3))
    sil = (cnt == 1) | ((nfront > 0) & (nfront < cnt))
    return sil[inv].view(3, M).T


def _cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _crossing(A, B, p, r):
    """Where the segment p -> p + r crosses the edge A -> B (s in [0,1] along p -> p + r), -1 if it does not."""
    e = B - A
    den = _cross2(r.expand_as(e), e)
    ok = den.abs() > 1e-9
    den = torch.where(ok, den, torch.ones_like(den))
    ap = A - p
    s_ = _cross2(ap, e) / den
    u_ = _cross2(ap, r.expand_as(e)) / den
    ok = ok & (s_ >= 0) & (s_ <= 1) & (u_ >= 0) & (u_ <= 1)
    return torch.where(ok, s_, torch.full_like(s_, -1.0))


def _edge_aa(mask, uv, F_, face, sil, tile=8, max_pairs=20_000_000):
    """Analytic antialias of a binary silhouette (nvdiffrast's idea), differentiable in uv.
    For each 4-neighbour pixel pair p (inside) / q (outside), the mesh outline crosses the segment p->q at
    s in [0,1]: the last crossing with a projected silhouette edge (the rim triangles that own those edges are
    often thinner than a pixel, so they are searched among all silhouette edges, not among the triangles under p).
    q gains coverage max(s - 0.5, 0), p loses max(0.5 - s, 0); gradients flow to the crossing edge's vertices.
    Edges are binned into `tile` x `tile` pixel tiles (box padded by 1 px); a pair only tests the edges in its
    own tile. (Testing every pair against every edge ran out of memory on noisy surfaces with many edges.)"""
    H, W = mask.shape
    dev = uv.device
    fg = face >= 0
    fi, ki = torch.nonzero(sil, as_tuple=True)
    if len(fi) == 0:
        return mask
    A, B = uv[F_[fi, ki]], uv[F_[fi, (ki + 1) % 3]]              # (E,2) silhouette edges in pixels
    with torch.no_grad():
        lo, hi = torch.minimum(A, B) - 1, torch.maximum(A, B) + 1
        nt_x, nt_y = (W + tile - 1) // tile, (H + tile - 1) // tile
        tx0 = (lo[:, 0] / tile).floor().long().clamp(0, nt_x - 1); tx1 = (hi[:, 0] / tile).floor().long().clamp(0, nt_x - 1)
        ty0 = (lo[:, 1] / tile).floor().long().clamp(0, nt_y - 1); ty1 = (hi[:, 1] / tile).floor().long().clamp(0, nt_y - 1)
        nx, ny = tx1 - tx0 + 1, ty1 - ty0 + 1
        cnt = nx * ny
        e_rep = torch.repeat_interleave(torch.arange(len(A), device=dev), cnt)
        local = torch.arange(int(cnt.sum()), device=dev) - (torch.cumsum(cnt, 0) - cnt)[e_rep]
        tid = (ty0[e_rep] + local // nx[e_rep]) * nt_x + tx0[e_rep] + local % nx[e_rep]
        order = torch.argsort(tid)
        tid_s, e_s = tid[order], e_rep[order]
    gains, ids = [], []
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        y0, y1, x0, x1 = max(0, -dy), H - max(0, dy), max(0, -dx), W - max(0, dx)
        pr = torch.nonzero(fg[y0:y1, x0:x1] & ~fg[y0 + dy:y1 + dy, x0 + dx:x1 + dx])
        if len(pr) == 0:
            continue
        py, px = pr[:, 0] + y0, pr[:, 1] + x0
        p = torch.stack([px + 0.5, py + 0.5], 1).to(uv.dtype)
        r = torch.tensor([dx, dy], dtype=uv.dtype, device=dev)
        with torch.no_grad():
            ptid = (py // tile) * nt_x + px // tile
            st = torch.searchsorted(tid_s, ptid, right=False)
            n = torch.searchsorted(tid_s, ptid, right=True) - st
        # search without autograd (memory ~ pairs x edges in the tile; with autograd every chunk's intermediates
        # stayed alive until backward and long runs ran out of memory), then recompute the winning crossing per
        # pixel with gradients (amax only passes gradient to the winner anyway)
        best = torch.full((len(p),), -1.0, dtype=uv.dtype, device=dev)
        best_e = torch.full((len(p),), -1, dtype=torch.long, device=dev)
        csum = torch.cumsum(n, 0)
        bounds, start = [0], 0
        while start < len(p):
            lim = (csum[start - 1] if start else 0) + max_pairs
            end = int(torch.searchsorted(csum, lim, right=True))
            end = max(end, start + 1)
            bounds.append(min(end, len(p)))
            start = bounds[-1]
        with torch.no_grad():
            Ad, Bd, pd = A.detach(), B.detach(), p
            for c0, c1 in zip(bounds[:-1], bounds[1:]):
                nn_ = n[c0:c1]
                if int(nn_.sum()) == 0:
                    continue
                pi = torch.repeat_interleave(torch.arange(c0, c1, device=dev), nn_)
                off = torch.arange(len(pi), device=dev) - (torch.cumsum(nn_, 0) - nn_)[pi - c0]
                ei = e_s[st[pi] + off]
                s_ = _crossing(Ad[ei], Bd[ei], pd[pi], r)
                best = best.scatter_reduce(0, pi, s_, "amax")
                win = (s_ >= 0) & (s_ == best[pi])
                best_e[pi[win]] = ei[win]
        hit = (best >= 0) & (best_e >= 0)
        s_ = _crossing(A[best_e[hit]], B[best_e[hit]], p[hit], r)       # with gradients to the winning edge
        gains += [(s_ - 0.5).clamp_min(0), -(0.5 - s_).clamp_min(0)]
        ids += [(py[hit] + dy) * W + (px[hit] + dx), py[hit] * W + px[hit]]
    if not ids:
        return mask
    delta = torch.zeros(H * W, dtype=uv.dtype, device=dev).index_add(0, torch.cat(ids), torch.cat(gains))
    return (mask + delta.view(H, W)).clamp(0, 1)


BACKEND = os.environ.get("DECLO_RENDER", "torch")   # or "nvdiffrast"; test: work/declo/test_render_backends.py
_glctx = None


def _render_nvdiffrast(V, F_, K, res, far, colors, frame, aa, near=0.05, farplane=20.0):
    """Clip space from the OpenCV camera: x_ndc = 2u/W - 1, y_ndc = 2v/H - 1, so row 0 of nvdiffrast's output is the
    image's top row; far=True negates depth so the depth test keeps the far surface."""
    import nvdiffrast.torch as dr
    global _glctx
    if _glctx is None:
        _glctx = dr.RasterizeCudaContext()
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    a, b = (farplane + near) / (farplane - near), -2 * farplane * near / (farplane - near)
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    zc = (a * z + b) * (-1 if far else 1)
    clip = torch.stack([2 * fx / res * x + (2 * cx / res - 1) * z, 2 * fy / res * y + (2 * cy / res - 1) * z, zc, z], 1)[None].contiguous()
    Fi = F_.int().contiguous()
    rast, _ = dr.rasterize(_glctx, clip, Fi, (res, res))
    face = rast[0, ..., 3].long() - 1
    fg = (face >= 0)[..., None]
    n = raster.vertex_normals(V, F_)
    if frame == "econ":
        n = n * _FLIP.to(V.device)
    attr = torch.cat([n, z[:, None]] + ([colors] if colors is not None else []), 1)[None].contiguous()
    ip = dr.interpolate(attr, rast, Fi)[0][0]
    nimg = F.normalize(ip[..., :3], dim=-1) * fg
    mask = (face >= 0).to(V.dtype)
    if aa:
        uv, _ = _project(V, K)
        mask = _edge_aa(mask, uv, F_, face, silhouette_edges(V, F_))
    out = {"mask": mask, "normal": nimg, "face": face, "depth": torch.where(fg[..., 0], ip[..., 3], torch.inf).detach()}
    if colors is not None:
        out["rgb"] = ip[..., 4:] * fg
    return out


def render(V, F_, K, res, far=False, colors=None, frame="econ", aa=True):
    """V (N,3) camera frame (OpenCV), torch, may require grad. F_ (M,3) long. K (3,3).
    Returns dict: mask (H,W) antialiased silhouette, normal (H,W,3) unit normals (0 outside),
    face (H,W), depth (H,W), and rgb (H,W,3) if per-vertex colors (N,3) are given."""
    dev = V.device
    K = torch.as_tensor(K, dtype=V.dtype, device=dev)
    F_ = torch.as_tensor(F_, dtype=torch.long, device=dev)
    if BACKEND == "nvdiffrast":
        return _render_nvdiffrast(V, F_, K, res, far, colors, frame, aa)
    face, _, depth = raster.rasterize(V.detach(), F_, K, res, res, far=far)
    uv, z = _project(V, K)
    idx, f, bs = _bary(uv, F_, face, res)
    bp = _perspective(bs, z, F_, f)
    n = raster.vertex_normals(V, F_)
    if frame == "econ":
        n = n * _FLIP.to(dev)
    nimg = torch.zeros(res, res, 3, device=dev, dtype=V.dtype)
    nimg = nimg.index_put((idx[:, 0], idx[:, 1]), F.normalize((n[F_[f]] * bp[..., None]).sum(1), dim=-1))
    mask = (face >= 0).to(V.dtype)
    if aa:
        mask = _edge_aa(mask, uv, F_, face, silhouette_edges(V, F_))
    out = {"mask": mask, "normal": nimg, "face": face, "depth": depth}
    if colors is not None:
        rgb = torch.zeros(res, res, colors.shape[1], device=dev, dtype=colors.dtype)
        out["rgb"] = rgb.index_put((idx[:, 0], idx[:, 1]), (colors[F_[f]] * bp[..., None]).sum(1))
    return out
