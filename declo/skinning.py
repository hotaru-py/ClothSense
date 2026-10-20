"""Canonical A-pose <-> photo pose with SMPL-X linear blend skinning (DeClotH: "LBS pre-defined in SMPL+H").

Every point borrows the skinning weights of its nearest canonical body vertex (the standard trick for
clothed / implicit humans). Weights are computed once per tet-grid vertex; Marching Tetrahedra vertices
interpolate them along their tet edge, so posing stays differentiable in the vertex positions.

Same maths as v1's GarmentSkinner (bak/) (un-pose the A-pose blend, re-pose with the photo pose, then
Multi-HMR's rigid map v_cam = R v + c), in PyTorch.
"""
from __future__ import annotations

import numpy as np
import torch


class Skinning:
    def __init__(self, smplx_model, body: dict, device=None):
        from smplx.lbs import batch_rigid_transform, batch_rodrigues, blend_shapes, vertices2joints
        self.device = device or smplx_model.shapedirs.device
        m = smplx_model
        dev = m.shapedirs.device
        b = torch.tensor(np.asarray(body["betas"], float)[None], dtype=torch.float32, device=dev)
        J = vertices2joints(m.J_regressor, m.v_template[None] + blend_shapes(b, m.shapedirs))

        def A(full):                                               # (55,3) axis-angles -> (55,4,4)
            rot = batch_rodrigues(torch.tensor(full, dtype=torch.float32, device=dev)).reshape(1, 55, 3, 3)
            return batch_rigid_transform(rot, J, m.parents)[1][0]

        canon, rv = body["canon"], np.asarray(body["hmr"]["rotvec"], float)
        ap = np.zeros((55, 3))
        ap[1:22] = np.asarray(canon["body_pose"]).reshape(21, 3)
        ph = np.zeros((55, 3))
        ph[1:22], ph[22], ph[25:40], ph[40:55] = rv[1:22], rv[52], rv[22:37], rv[37:52]
        f32 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=self.device)
        self._A = A
        self.A_ap, self.A_ph = A(ap).to(self.device), A(ph).to(self.device)
        self.W_body = m.lbs_weights.to(self.device)                 # (10475,55)
        self.offset, self.scale = f32(canon["offset"]), float(canon["scale"])
        self.R, self.c = f32(body["R"]), f32(body["c"])
        self.canon_V = f32(canon["V"])                              # canonical body, frame coords

    def set_pose(self, rotvec, R=None, c=None):
        """Re-pose (M6): a new pose in Multi-HMR's layout (53,3): root, 21 body joints, 15+15 hand joints, jaw.
        The root rotation is NOT taken from rotvec[0]: pass the rigid map R, c (camera frame) explicitly, or keep
        the photo's. Everything posed after this call uses the new pose."""
        rv = np.asarray(rotvec, float).reshape(-1, 3)
        ph = np.zeros((55, 3))
        ph[1:22] = rv[1:22]
        if len(rv) >= 53:
            ph[22], ph[25:40], ph[40:55] = rv[52], rv[22:37], rv[37:52]
        self.A_ph = self._A(ph).to(self.device)
        f32 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=self.device)
        if R is not None:
            self.R = f32(R)
        if c is not None:
            self.c = f32(c)

    @torch.no_grad()
    def weights(self, X_canon, chunk=20000):
        """Nearest canonical body vertex's weights for points (N,3) in the canonical frame."""
        idx = torch.cat([torch.cdist(X_canon[i:i + chunk], self.canon_V).argmin(1)
                         for i in range(0, len(X_canon), chunk)])
        return self.W_body[idx]

    def _T(self, W):
        return torch.einsum("nj,jab->nab", W, self.A_ap), torch.einsum("nj,jab->nab", W, self.A_ph)

    def to_cam(self, X_canon, W):
        """Canonical (frame coords) -> photo camera frame. Differentiable in X_canon."""
        Xl = (X_canon - self.offset) / self.scale
        T_ap, T_ph = self._T(W)
        Xh = torch.cat([Xl, torch.ones_like(Xl[:, :1])], 1)[..., None]
        X = (T_ph @ torch.linalg.solve(T_ap, Xh))[:, :3, 0]
        return X @ self.R.T + self.c

    def to_canon(self, X_cam, W):
        """Photo camera frame -> canonical (e.g. to un-pose the ClothWild template)."""
        Xp = (X_cam - self.c) @ self.R
        T_ap, T_ph = self._T(W)
        Xh = torch.cat([Xp, torch.ones_like(Xp[:, :1])], 1)[..., None]
        X = (T_ap @ torch.linalg.solve(T_ph, Xh))[:, :3, 0]
        return X * self.scale + self.offset

    @torch.no_grad()
    def weights_posed(self, X_cam, V_body_cam, chunk=20000):
        """Weights for points given in the photo pose: nearest posed body vertex."""
        Vb = torch.as_tensor(V_body_cam, dtype=torch.float32, device=self.device)
        idx = torch.cat([torch.cdist(X_cam[i:i + chunk], Vb).argmin(1) for i in range(0, len(X_cam), chunk)])
        return self.W_body[idx]
