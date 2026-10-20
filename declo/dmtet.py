"""DeClotH step (2): Deep Marching Tetrahedra (Shen et al., NeurIPS 2021), one for cloth and one for the body.

A deformable tetrahedral grid in the canonical A-pose; an MLP on a multiresolution hash encoding
(Mueller et al. 2022) predicts per grid vertex a signed distance and a small offset; Marching
Tetrahedra turns the zero level set into a triangle mesh, differentiably in the vertex positions.

Grid: regular cubes split into 6 tetrahedra (Kuhn/Freudenthal, conforming across cubes), kept only in
a band around the templates. At 1 cm cells a full 2 m box would be ~8 M tets; the band keeps ~1-2 M.
Hash grid: plain PyTorch (tiny-cuda-nn needs the CUDA toolkit); fine at this point count.
Sign convention: SDF < 0 inside.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import torch
import torch.nn as nn

# --------------------------------------------------------------------------- grid

_KUHN = [list(p) for p in itertools.permutations(range(3))]      # 6 axis orders -> 6 tets per cube


def tet_grid(lo, hi, cell, keep_cube=None):
    """Tetrahedral grid over the box [lo, hi] (metres). keep_cube(centres (C,3)) -> bool mask selects cubes.
    Returns verts (N,3) float32, tets (T,4) int64, positively oriented."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    n = np.ceil((hi - lo) / cell).astype(int)
    ii, jj, kk = np.meshgrid(*(np.arange(k) for k in n), indexing="ij")
    cubes = np.stack([ii, jj, kk], -1).reshape(-1, 3)
    if keep_cube is not None:
        cubes = cubes[keep_cube(lo + (cubes + 0.5) * cell)]
    tets = []
    for order in _KUHN:                                           # path 000 -> e_a -> e_a+e_b -> 111
        path = [np.zeros(3, int)]
        for ax in order:
            step = path[-1].copy()
            step[ax] = 1
            path.append(step)
        tets.append(np.stack([cubes + p for p in path], 1))       # (C,4,3) lattice coords
    tets = np.concatenate(tets)                                   # (6C,4,3)
    stride = np.array([(n[1] + 1) * (n[2] + 1), n[2] + 1, 1])
    lin = (tets * stride).sum(-1)                                 # (T,4) lattice ids
    uniq, inv = np.unique(lin.reshape(-1), return_inverse=True)
    tets = inv.reshape(-1, 4)
    lat = np.stack([uniq // stride[0], (uniq // stride[1]) % (n[1] + 1), uniq % (n[2] + 1)], 1)
    verts = lo + lat * cell
    a, b, c, d = (verts[tets[:, i]] for i in range(4))
    neg = np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) < 0
    tets[neg] = tets[neg][:, [0, 2, 1, 3]]
    return torch.tensor(verts, dtype=torch.float32), torch.tensor(tets, dtype=torch.long)


class GridLookup:
    """Is a point inside the kept cubes of a band-limited tet grid? (the DMTet's field is only trained there)"""

    def __init__(self, lo, hi, cell, keep_cube, device="cuda"):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        n = np.ceil((hi - lo) / cell).astype(int)
        ijk = np.stack(np.meshgrid(*(np.arange(k) for k in n), indexing="ij"), -1).reshape(-1, 3)
        occ = keep_cube(lo + (ijk + 0.5) * cell).reshape(tuple(n))
        self.occ = torch.tensor(occ, device=device)
        self.lo = torch.tensor(lo, dtype=torch.float32, device=device)
        self.n = torch.tensor(n, device=device)
        self.cell = cell

    def __call__(self, X):
        q = ((X - self.lo) / self.cell).floor().long()
        ok = ((q >= 0) & (q < self.n)).all(1)
        out = torch.zeros(len(X), dtype=torch.bool, device=X.device)
        qq = q[ok]
        out[ok] = self.occ[qq[:, 0], qq[:, 1], qq[:, 2]]
        return out


# --------------------------------------------------------------------------- marching tetrahedra

_TRI_TABLE = torch.tensor([
    [-1, -1, -1, -1, -1, -1], [1, 0, 2, -1, -1, -1], [4, 0, 3, -1, -1, -1], [1, 4, 2, 1, 3, 4],
    [3, 1, 5, -1, -1, -1], [2, 3, 0, 2, 5, 3], [1, 4, 0, 1, 5, 4], [4, 2, 5, -1, -1, -1],
    [4, 5, 2, -1, -1, -1], [4, 1, 0, 4, 5, 1], [3, 2, 0, 3, 5, 2], [1, 3, 5, -1, -1, -1],
    [4, 1, 2, 4, 3, 1], [3, 0, 4, -1, -1, -1], [2, 0, 1, -1, -1, -1], [-1, -1, -1, -1, -1, -1]])
_NUM_TRI = torch.tensor([0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0])
_TET_EDGES = torch.tensor([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3])
FLIP_FACES = True     # table orientation vs our positively oriented tets, checked in tests (outward normals)


def marching_tets(pos, sdf, tets):
    """pos (N,3), sdf (N,), tets (T,4) -> verts (V,3), faces (F,3), and for every output vertex the two grid
    vertices it lies between plus the interpolation weight (to carry per-grid-vertex attributes, e.g. LBS)."""
    dev = pos.device
    with torch.no_grad():
        occ = sdf < 0                                            # inside
        occ4 = occ[tets]
        s = occ4.sum(-1)
        valid = (s > 0) & (s < 4)
        edges = tets[valid][:, _TET_EDGES.to(dev)].reshape(-1, 2)
        edges = torch.stack([edges.min(1).values, edges.max(1).values], 1)
        uniq, idx_map = torch.unique(edges, dim=0, return_inverse=True)
        cross = occ[uniq].sum(-1) == 1
        mapping = torch.full((len(uniq),), -1, dtype=torch.long, device=dev)
        mapping[cross] = torch.arange(int(cross.sum()), device=dev)
        idx_map = mapping[idx_map].reshape(-1, 6)
        ev = uniq[cross]                                         # (V,2) grid vertex pairs
        code = (occ4[valid].long() * (2 ** torch.arange(4, device=dev))).sum(-1)
        ntri = _NUM_TRI.to(dev)[code]
        table = _TRI_TABLE.to(dev)[code]
        faces = torch.cat([
            torch.gather(idx_map[ntri == 1], 1, table[ntri == 1][:, :3]).reshape(-1, 3),
            torch.gather(idx_map[ntri == 2], 1, table[ntri == 2][:, :6]).reshape(-1, 3)])
        if FLIP_FACES:
            faces = faces[:, [0, 2, 1]]
    s0, s1 = sdf[ev[:, 0]], sdf[ev[:, 1]]
    w = s0 / (s0 - s1)                                           # zero crossing along the edge
    verts = pos[ev[:, 0]] * (1 - w)[:, None] + pos[ev[:, 1]] * w[:, None]
    return verts, faces, ev, w.detach()


# --------------------------------------------------------------------------- hash grid + MLP

class HashGrid(nn.Module):
    """Multiresolution hash encoding; all levels and the 8 cell corners in one gather."""
    PRIMES = (1, 2654435761, 805459861)

    def __init__(self, lo, hi, levels=16, feats=2, log2_size=19, base_res=16, max_res=2048):
        super().__init__()
        self.register_buffer("lo", torch.as_tensor(lo, dtype=torch.float32))
        self.register_buffer("ext", torch.as_tensor(np.asarray(hi) - np.asarray(lo), dtype=torch.float32).max())
        g = math.exp((math.log(max_res) - math.log(base_res)) / (levels - 1))
        res = [int(math.floor(base_res * g ** l)) for l in range(levels)]
        self.size, self.L, self.Fd = 2 ** log2_size, levels, feats
        self.register_buffer("res", torch.tensor(res, dtype=torch.float32))
        self.register_buffer("dense", torch.tensor([(r + 1) ** 3 <= self.size for r in res]))
        self.register_buffer("lvl_off", torch.arange(levels) * self.size)
        self.table = nn.Parameter(torch.empty(levels * self.size, feats).uniform_(-1e-4, 1e-4))
        self.out_dim = levels * feats
        self.register_buffer("corners", torch.tensor(list(itertools.product((0, 1), repeat=3))))

    def _idx_w(self, u):
        """Table rows and trilinear weights of the 8 cell corners at every level: (N,L,8) each."""
        p = u[:, None, :] * self.res[None, :, None]                   # (N,L,3)
        p0 = p.floor()
        f = p - p0
        q = p0.long()[:, :, None, :] + self.corners[None, None]       # (N,L,8,3)
        r1 = (self.res.long() + 1)[None, :, None]
        dense = (q[..., 0] * r1 + q[..., 1]) * r1 + q[..., 2]
        hashed = ((q[..., 0] * self.PRIMES[0]) ^ (q[..., 1] * self.PRIMES[1]) ^ (q[..., 2] * self.PRIMES[2])) % self.size
        idx = torch.where(self.dense[None, :, None], dense, hashed) + self.lvl_off[None, :, None]
        c = self.corners[None, None].bool()                            # (1,1,8,3)
        w = torch.where(c, f[:, :, None, :], 1 - f[:, :, None, :]).prod(-1)   # (N,L,8)
        return idx, w

    def forward(self, x):
        u = ((x - self.lo) / self.ext).clamp(0, 1)                    # isotropic cells
        if x.requires_grad:                    # gradient w.r.t. the points (penetration loss): plain autograd
            idx, w = self._idx_w(u)
            feat = self.table[idx.reshape(-1)].view(*idx.shape, self.Fd)
            return (w[..., None] * feat).sum(2).reshape(len(x), -1)
        return _HashEncode.apply(u, self.table, self)              # fixed points: memory-light


class _HashEncode(torch.autograd.Function):
    """Hash encoding of fixed points (grid vertices, canonical surface points) that keeps only the points for
    backward and recomputes indices / weights in chunks. The plain version stores an (N,16,8) index + weight
    tensor per query, ~1 GB per DMTet at our grid sizes, which pushed M4 past 8 GB into shared memory."""
    CHUNK = 32768

    @staticmethod
    def forward(ctx, u, table, grid):
        out = torch.empty(len(u), grid.out_dim, device=u.device, dtype=table.dtype)
        for i in range(0, len(u), _HashEncode.CHUNK):
            idx, w = grid._idx_w(u[i:i + _HashEncode.CHUNK])
            feat = table[idx.reshape(-1)].view(*idx.shape, grid.Fd)
            out[i:i + _HashEncode.CHUNK] = (w[..., None] * feat).sum(2).reshape(len(idx), -1)
        ctx.save_for_backward(u)
        ctx.grid, ctx.n_rows = grid, table.shape[0]
        return out

    @staticmethod
    def backward(ctx, g):
        (u,) = ctx.saved_tensors
        grid = ctx.grid
        gt = torch.zeros(ctx.n_rows, grid.Fd, device=g.device, dtype=g.dtype)
        for i in range(0, len(u), _HashEncode.CHUNK):
            idx, w = grid._idx_w(u[i:i + _HashEncode.CHUNK])
            gi = g[i:i + _HashEncode.CHUNK].view(len(idx), grid.L, 1, grid.Fd)
            gt.index_add_(0, idx.reshape(-1), (w[..., None] * gi).reshape(-1, grid.Fd))
        return None, gt, None


class DMTet(nn.Module):
    """Tet grid + hash-grid MLP -> (sdf, offset) per grid vertex -> mesh by Marching Tetrahedra."""

    def __init__(self, verts, tets, cell, hidden=32, max_res=1024, deform=True):
        """Paper (supp. S4): two fully-connected layers, 32 hidden, ReLU; hash encoding with 16 levels and a
        maximum resolution of 1024 over the grid box normalised to [-0.5, 0.5]."""
        super().__init__()
        self.deform = deform
        self.register_buffer("verts", verts)
        self.register_buffer("tets", tets)
        self.cell = cell
        lo, hi = verts.min(0).values - cell, verts.max(0).values + cell
        self.enc = HashGrid(lo.cpu().numpy(), hi.cpu().numpy(), levels=16, max_res=max_res)
        self.mlp = nn.Sequential(nn.Linear(self.enc.out_dim, hidden), nn.ReLU(), nn.Linear(hidden, 4))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def query(self, x):
        o = self.mlp(self.enc(x))
        off = torch.tanh(o[:, 1:]) * 0.5 * self.cell if self.deform else torch.zeros_like(o[:, 1:])
        return o[:, 0], off

    def forward(self):
        sdf, off = self.query(self.verts)
        pos = self.verts + off
        V, F, ev, w = marching_tets(pos, sdf, self.tets)
        return {"V": V, "F": F, "edge_verts": ev, "edge_w": w, "sdf": sdf, "offset": off, "cell": self.cell}


class ShellDMTet(DMTet):
    """Cloth as a shell that cannot tear: sdf = max(|f| - tau, g).
    f: signed field whose zero set is the garment's mid-surface (a continuous f always has a band |f| < tau around
    its zero crossing, so the shell can move but not open); g: garment region (< 0 inside), which makes the
    openings (neck, sleeves, hem). Plain "udf - tau" shells tore under SDS at our grid resolution.
    Thickness 2 tau must exceed the longest tet edge (~1.7 cells) or Marching Tetrahedra skips the shell."""

    def __init__(self, verts, tets, cell, tau, hidden=32, max_res=1024, deform=True):
        super().__init__(verts, tets, cell, hidden, max_res, deform)
        self.tau = tau
        self.mlp[-1] = nn.Linear(hidden, 5)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def fields(self, x):
        o = self.mlp(self.enc(x))
        off = torch.tanh(o[:, 2:]) * 0.5 * self.cell if self.deform else torch.zeros_like(o[:, 2:])
        return o[:, 0], o[:, 1], off

    def query(self, x):
        f, g, off = self.fields(x)
        return torch.maximum(f.abs() - self.tau, g), off


def fit_shell(net: ShellDMTet, f0, g0, steps=400, lr=1e-2, batch=65536, log=print, seed=0):
    """Initialise f and g of a ShellDMTet (targets truncated at +-4 cells)."""
    b = 4 * net.cell
    f0, g0 = f0.clamp(-b, b), g0.clamp(-b, b)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.1 ** (i / steps))
    dev = net.verts.device
    gen = torch.Generator(device=dev).manual_seed(seed)
    near = torch.nonzero((f0.abs() < 2 * net.cell + net.tau) & (g0 < 2 * net.cell)).squeeze(1)
    for i in range(steps):
        idx = torch.cat([torch.randint(0, len(f0), (batch // 2,), device=dev, generator=gen),
                         near[torch.randint(0, len(near), (batch // 2,), device=dev, generator=gen)]])
        f, g, off = net.fields(net.verts[idx])
        loss = (f - f0[idx]).pow(2).mean() + (g - g0[idx]).pow(2).mean() + 1e-2 * off.pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        sdf, _ = net.query(net.verts)
        tgt = torch.maximum(f0.abs() - net.tau, g0)
        agree = ((sdf < 0) == (tgt < 0)).float().mean().item()
    log(f"  fit_shell {steps} steps: inside/outside agreement {100 * agree:.3f} %")
    return net


def fit_sdf(net: DMTet, target_sdf: torch.Tensor, steps=300, lr=1e-2, log=print, band=None, batch=65536, seed=0):
    """Initialise a DMTet to a template: regress the grid-vertex SDF onto the template's SDF (target truncated
    at +-band). Random vertex batches per step; the final line reports all vertices."""
    band = band or 4 * net.cell
    tgt = target_sdf.clamp(-band, band)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.1 ** (i / steps))
    dev = net.verts.device
    g = torch.Generator(device=dev).manual_seed(seed)
    n = len(net.verts)
    near = torch.nonzero(tgt.abs() < 2 * net.cell).squeeze(1)       # half of every batch near the surface
    for i in range(steps):
        idx = torch.cat([torch.randint(0, n, (batch // 2,), device=dev, generator=g),
                         near[torch.randint(0, len(near), (batch // 2,), device=dev, generator=g)]])
        sdf, off = net.query(net.verts[idx])
        loss = (sdf - tgt[idx]).pow(2).mean() + 1e-2 * off.pow(2).mean()          # clamp the target only
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        sdf, _ = net.query(net.verts)
        rmse = (sdf - tgt).pow(2).mean().sqrt().item()
        sign_ok = ((sdf < 0) == (tgt < 0)).float().mean().item()
    log(f"  fit_sdf {steps} steps: rmse {1000 * rmse:.2f} mm, sign agreement {100 * sign_ok:.3f} %")
    return net


# --------------------------------------------------------------------------- template SDFs

def inside_voxels(V, F, voxel=0.005, n_samples=2_000_000, seed=0):
    """Solid voxelisation: surface samples -> voxels (dilated once, which closes holes smaller than a voxel
    such as SMPL-X's mouth slit) -> fill everything not reachable from the box boundary. Returns a lookup."""
    import trimesh
    from scipy import ndimage
    m = trimesh.Trimesh(np.asarray(V), np.asarray(F), process=False)
    pts, _ = trimesh.sample.sample_surface(m, n_samples, seed=seed)
    lo = pts.min(0) - 3 * voxel
    n = np.ceil((pts.max(0) + 3 * voxel - lo) / voxel).astype(int) + 1
    occ = np.zeros(n, bool)
    ijk = np.floor((pts - lo) / voxel).astype(int)
    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    occ = ndimage.binary_dilation(occ)
    solid = ndimage.binary_fill_holes(occ)

    def lookup(P):
        q = np.floor((np.asarray(P) - lo) / voxel).astype(int)
        ok = ((q >= 0) & (q < n)).all(1)
        out = np.zeros(len(q), bool)
        out[ok] = solid[q[ok, 0], q[ok, 1], q[ok, 2]]
        return out
    return lookup


def _point_triangle_distance(P, T):
    """Exact distance from points P (n,3) to triangles T (n,3,3), vectorised (Ericson, Real-Time Collision Detection)."""
    a, b, c = T[:, 0], T[:, 1], T[:, 2]
    ab, ac, ap = b - a, c - a, P - a
    d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
    bp = P - b
    d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
    cp = P - c
    d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    denom = va + vb + vc
    denom = np.where(np.abs(denom) < 1e-20, 1e-20, denom)
    v, w = vb / denom, vc / denom
    q = a + ab * v[:, None] + ac * w[:, None]                         # interior projection
    sd = lambda x, y: np.where(np.abs(y) < 1e-20, 0.0, x / np.where(np.abs(y) < 1e-20, 1, y))
    # edge / vertex regions
    t_ab = np.clip(sd(d1, d1 - d3), 0, 1)
    t_ac = np.clip(sd(d2, d2 - d6), 0, 1)
    t_bc = np.clip(sd(d4 - d3, (d4 - d3) + (d5 - d6)), 0, 1)
    cand = np.stack([q, a + ab * t_ab[:, None], a + ac * t_ac[:, None], b + (c - b) * t_bc[:, None]], 1)
    inside = (va >= 0) & (vb >= 0) & (vc >= 0)
    dist = np.linalg.norm(cand - P[:, None], axis=-1)
    dist[:, 0] = np.where(inside, dist[:, 0], np.inf)
    return dist.min(1)


def mesh_sdf(points, V, F, n_samples=300_000, thin=None, seed=0, voxel=0.005, exact_within=0.05):
    """Signed distance of points to a mesh (< 0 inside): distance to a dense surface sample; sign from the
    nearest sample's normal close to the surface (< 3 voxels) and from a flood-filled voxelisation
    further away, where the normal test fails (open mouth, eyeballs, concave regions).
    thin=t treats the mesh as a shell of half-thickness t (open garments)."""
    import trimesh
    from scipy.spatial import cKDTree
    m = trimesh.Trimesh(np.asarray(V), np.asarray(F), process=False)
    pts, fid = trimesh.sample.sample_surface(m, n_samples, seed=seed)
    P = np.asarray(points.detach().cpu() if torch.is_tensor(points) else points, np.float64)
    d, i = cKDTree(pts).query(P, workers=-1)
    near = d < exact_within
    if near.any():                       # exact distance to the nearest sample's triangle (samples alone: ~1 mm noise)
        d[near] = np.minimum(d[near], _point_triangle_distance(P[near], np.asarray(V)[np.asarray(F)[fid[i[near]]]]))
    if thin is not None:
        return torch.tensor(d - thin, dtype=torch.float32)
    near_in = np.einsum("ij,ij->i", P - pts[i], m.face_normals[fid][i]) < 0
    inside = np.where(d < 3 * voxel, near_in, inside_voxels(V, F, voxel)(P))
    return torch.tensor(np.where(inside, -d, d), dtype=torch.float32)


def main_component(V, F):
    """Largest connected piece of a mesh (drops SMPL-X's eyeballs, which sit inside the head)."""
    import trimesh
    parts = trimesh.Trimesh(np.asarray(V), np.asarray(F), process=False).split(only_watertight=False)
    m = max(parts, key=lambda q: len(q.faces))
    return np.asarray(m.vertices), np.asarray(m.faces)


def band_keep(surfaces, band, solid=None):
    """keep_cube for tet_grid: cubes within `band` of any surface (list of (V,F)), plus cubes inside the
    closed mesh `solid` (the body template), so no grid boundary sits inside the body."""
    import trimesh
    from scipy.spatial import cKDTree
    pts = np.concatenate([trimesh.sample.sample_surface(trimesh.Trimesh(V, F, process=False), 200_000, seed=1)[0]
                          for V, F in surfaces])
    tree = cKDTree(pts)
    inside = inside_voxels(*solid) if solid is not None else None

    def keep(centres):
        d, _ = tree.query(centres, workers=-1, distance_upper_bound=band * 1.01)
        k = np.isfinite(d)
        if inside is not None:
            k |= inside(centres)                  # inside/outside only: the voxel lookup, not a full SDF
        return k
    return keep


def laplacian_loss(V, F):
    """Uniform Laplacian (vertex minus neighbour mean), squared, averaged: surface smoothness."""
    e = torch.cat([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = torch.cat([e, e.flip(1)])
    deg = torch.zeros(len(V), device=V.device).index_add_(0, e[:, 0], torch.ones(len(e), device=V.device))
    nb = torch.zeros_like(V).index_add_(0, e[:, 0], V[e[:, 1]])
    return (V - nb / deg.clamp_min(1)[:, None]).pow(2).sum(1).mean()


def normal_consistency(V, F):
    """1 - cos between the normals of faces sharing an edge (bending only; MT's irregular triangles make a
    uniform Laplacian push vertices sideways)."""
    M = len(F)
    e = torch.sort(torch.cat([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1).values
    _, inv, cnt = torch.unique(e, dim=0, return_inverse=True, return_counts=True)
    fid = torch.arange(3 * M, device=F.device) % M
    order = torch.argsort(inv)
    inv_s, fid_s = inv[order], fid[order]
    pair = (inv_s[1:] == inv_s[:-1])                               # consecutive rows of the same edge
    f1, f2 = fid_s[:-1][pair], fid_s[1:][pair]
    n = torch.nn.functional.normalize(torch.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]], dim=1), dim=1)
    return (1 - (n[f1] * n[f2]).sum(1)).mean()
