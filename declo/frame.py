"""Work frame: the square crop every DeClotH stage works in.

ECON (whose normal estimator DeClotH uses) crops the person so the larger side of the person box spans
0.9 of a 512 px square, centred. We use the same crop for masks, normals and the optimisation renders,
with a pinhole camera K for it (Multi-HMR gives a perspective camera, ECON assumes orthographic; at
2.5-3 m the difference is a few pixels at the edges).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class WorkFrame:
    x0: float          # crop square in original-image pixels
    y0: float
    side: float
    res: int = 512     # output resolution

    @classmethod
    def from_mask(cls, mask: np.ndarray, res=512, fill=0.9):
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            raise RuntimeError("empty person mask")
        x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        side = max(x2 - x1, y2 - y1) / fill
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        return cls(float(cx - side / 2), float(cy - side / 2), float(side), res)

    def A(self, res=None):
        """2x3 affine: original pixels -> crop pixels at `res`."""
        s = (res or self.res) / self.side
        return np.array([[s, 0, -self.x0 * s], [0, s, -self.y0 * s]], np.float64)

    def warp(self, img: np.ndarray, res=None, nearest=False):
        r = res or self.res
        flags = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA if self.side > r else cv2.INTER_LINEAR
        return cv2.warpAffine(img, self.A(r), (r, r), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def K(self, K_orig: np.ndarray, res=None):
        A = np.vstack([self.A(res), [0, 0, 1]])
        return A @ K_orig

    def to_dict(self):
        return {"x0": self.x0, "y0": self.y0, "side": self.side, "res": self.res}


def K_original_from_hmr(hmr: dict) -> np.ndarray:
    """Multi-HMR's K lives on its letterboxed square; map it back to the original photo's pixels."""
    K, lb = hmr["K"].astype(np.float64), hmr["letterbox"]
    s = lb["scale"]
    return np.array([[K[0, 0] / s, 0, (K[0, 2] - lb["ox"]) / s],
                     [0, K[1, 1] / s, (K[1, 2] - lb["oy"]) / s],
                     [0, 0, 1]])
