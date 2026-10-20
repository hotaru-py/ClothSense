"""Human skeleton S_skel for the SDS conditions (DeClotH Eq. 5-7): OpenPose COCO-18 keypoints from the
posed SMPL-X body template, drawn the way ControlNet's OpenPose annotator draws them (v1.1).
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import torch

from .cameras import project

# OpenPose COCO-18 <- SMPL-X joints (smplx package order; 55-59 = nose, reye, leye, rear, lear)
COCO18_FROM_SMPLX = [55, None, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7, 56, 57, 58, 59]   # 1 (neck) = mid-shoulders
LIMBS = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11], [2, 12], [12, 13], [13, 14],
         [2, 1], [1, 15], [15, 17], [1, 16], [16, 18]]
COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


def posed_joints(smplx_model, body: dict) -> np.ndarray:
    """All SMPL-X joints (incl. nose/eyes/ears) of the BodyNet result, in the photo camera frame."""
    from .hmr import _zeros
    hmr, dev = body["hmr"], smplx_model.shapedirs.device
    rv = np.asarray(hmr["rotvec"], np.float32)
    t = lambda a: torch.tensor(np.asarray(a, np.float32).reshape(1, -1), device=dev)
    with torch.no_grad():
        o = smplx_model(betas=t(body["betas"]), body_pose=t(rv[1:22]), global_orient=_zeros(smplx_model, 3),
                        left_hand_pose=t(rv[22:37]), right_hand_pose=t(rv[37:52]), jaw_pose=t(rv[52:53]),
                        expression=t(hmr["expression"]), leye_pose=_zeros(smplx_model, 3),
                        reye_pose=_zeros(smplx_model, 3))
    J = o.joints[0].cpu().numpy().astype(np.float64)
    return J @ body["R"].T + body["c"]


def coco18(J_cam: np.ndarray) -> np.ndarray:
    out = np.zeros((18, 3))
    for i, j in enumerate(COCO18_FROM_SMPLX):
        out[i] = J_cam[j] if j is not None else 0.5 * (J_cam[16] + J_cam[17])
    return out


def draw(kp3d: np.ndarray, K, R, t, res=512, stick=4, face_visible=None) -> np.ndarray:
    """OpenPose-style skeleton image (res,res,3) uint8. Face points (nose/eyes/ears) are dropped when they
    face away from the camera, as the OpenPose detector would miss them on a back view."""
    uv, z = project(kp3d, K, R, t)
    vis = np.ones(18, bool)
    if face_visible is not None:
        vis[[0, 14, 15, 16, 17]] = face_visible
    canvas = np.zeros((res, res, 3), np.uint8)
    for (a, b), col in zip(LIMBS, COLORS):
        a, b = a - 1, b - 1
        if not (vis[a] and vis[b]):
            continue
        (x0, y0), (x1, y1) = uv[a], uv[b]
        length = math.hypot(x1 - x0, y1 - y0)
        ang = math.degrees(math.atan2(y0 - y1, x0 - x1))
        poly = cv2.ellipse2Poly((int((x0 + x1) / 2), int((y0 + y1) / 2)), (int(length / 2), stick), int(ang), 0, 360, 1)
        cv2.fillConvexPoly(canvas, poly, [int(c * 0.6) for c in col])
    for i, col in enumerate(COLORS):
        if vis[i]:
            cv2.circle(canvas, (int(uv[i, 0]), int(uv[i, 1])), 4, col, thickness=-1)
    return canvas


def face_visible(J_cam: np.ndarray, R, t) -> bool:
    """Is the head's front towards this camera? (nose ahead of the ears' midpoint, seen from the camera)"""
    nose, ears = J_cam[55], 0.5 * (J_cam[58] + J_cam[59])
    cam_center = -np.asarray(R).T @ np.asarray(t)
    return float((nose - ears) @ (cam_center - ears)) > 0
