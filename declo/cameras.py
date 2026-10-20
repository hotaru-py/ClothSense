"""DeClotH step (3) cameras: the photo camera (front; "back" = same rays, far surface) and random orbit cameras
for the SDS views. A camera is (K, Rw2c, t): x_cam = Rw2c x + t, OpenCV axes, everything in the photo's
camera frame as "world" (the frame the posed meshes live in after LBS).
"""
from __future__ import annotations

import math

import numpy as np
import torch


def intrinsics(fov_deg, res):
    f = res / (2 * math.tan(math.radians(fov_deg) / 2))
    return np.array([[f, 0, res / 2], [0, f, res / 2], [0, 0, 1.0]])


def look_at(eye, target, up):
    """OpenCV extrinsics (R, t) for a camera at eye looking at target; up is a world direction."""
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)                        # image x to the right, image y = -up (OpenCV, y down)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ eye


class Orbit:
    """Cameras orbiting the person: azimuth around the body's up axis (0 = the photo view), elevation
    above the horizontal, distance chosen so the body (height h) fills `fill` of the frame."""

    def __init__(self, V_cam: np.ndarray, up_cam: np.ndarray, res=512, fov_deg=40.0, fill=0.85):
        self.res, self.fov = res, fov_deg
        self.K = intrinsics(fov_deg, res)
        self.up = up_cam / np.linalg.norm(up_cam)
        lo, hi = V_cam.min(0), V_cam.max(0)
        self.center = 0.5 * (lo + hi)
        h = float(np.ptp(V_cam @ self.up))
        self.dist = 0.5 * h / fill / math.tan(math.radians(fov_deg) / 2)
        # azimuth 0 looks from where the photo camera is (camera origin, projected onto the horizontal)
        to_cam = -self.center
        to_cam -= (to_cam @ self.up) * self.up
        self.front = to_cam / np.linalg.norm(to_cam)
        self.side = np.cross(self.up, self.front)

    def camera(self, azim_deg, elev_deg=0.0, r=1.0, origin=None):
        """r scales the framing distance (paper: spherical r; 1 = the whole body fills the frame); origin
        defaults to the body centre (head views use the head keypoint)."""
        a, e = math.radians(azim_deg), math.radians(elev_deg)
        d = math.cos(e) * (math.cos(a) * self.front + math.sin(a) * self.side) + math.sin(e) * self.up
        o = self.center if origin is None else np.asarray(origin, float)
        R, t = look_at(o + r * self.dist * d, o, self.up)
        return self.K, R, t

    head = None          # head keypoint (photo camera frame); set by the guidance to enable head views

    def sample(self, rng, elev=(-30, 30), r=(0.7, 1.3), p_head=0.2):
        """Paper (supp. S4): r in [0.7, 1.3], elevation in [-30, 30] deg, azimuth over 360 deg, uniform; plus
        zoomed-in views around the head keypoint with r in [0.3, 0.4] and elevation / azimuth in [-90, 90] deg
        (clamped to +-85 so look_at stays defined). Head views are drawn with probability p_head (the paper
        does not give the ratio)."""
        if self.head is not None and rng.uniform() < p_head:
            az, el = rng.uniform(-85, 85), rng.uniform(-85, 85)
            return (az % 360, el), self.camera(az, el, rng.uniform(0.3, 0.4), origin=self.head)
        az, el = rng.uniform(0, 360), rng.uniform(*elev)
        return (az, el), self.camera(az, el, rng.uniform(*r))


def project(X, K, R, t):
    Xc = X @ np.asarray(R).T + np.asarray(t)
    uv = Xc @ np.asarray(K).T
    return uv[:, :2] / uv[:, 2:3], Xc[:, 2]


def to_cam_frame(V, R, t):
    """Torch: points in the photo frame -> this camera's frame (for the rasterisers)."""
    R = torch.as_tensor(R, dtype=V.dtype, device=V.device)
    t = torch.as_tensor(t, dtype=V.dtype, device=V.device)
    return V @ R.T + t
