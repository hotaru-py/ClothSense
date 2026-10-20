"""GLB export shared by the scripts and the web service.

Vertices come in the OpenCV camera frame (x right, y down, z forward) and are written y-up, facing +z (glTF).
Colours come as sRGB (photo colours); glTF's COLOR_0 is linear, so they are converted, otherwise viewers
brighten them a second time (the first web test looked washed out).
"""
from __future__ import annotations

import numpy as np


def srgb_to_linear(rgb_u8: np.ndarray) -> np.ndarray:
    c = np.asarray(rgb_u8, np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return np.round(lin * 255).astype(np.uint8)


def normal_colors(V_cam, F) -> np.ndarray:
    """(n+1)/2 colouring of the glTF-frame vertex normals, RGBA uint8 (already 'linear' data, not a photo)."""
    import trimesh
    n = trimesh.Trimesh(np.asarray(V_cam) * np.array([1.0, -1.0, -1.0]), F, process=False).vertex_normals
    rgb = ((n + 1) * 127.5).clip(0, 255).astype(np.uint8)
    return np.concatenate([rgb, np.full((len(rgb), 1), 255, np.uint8)], 1)


def glb_bytes(meshes) -> bytes:
    """meshes: list of (V camera frame, F, colours) with colours = per-vertex RGB/RGBA uint8 in sRGB, one RGBA
    tuple, or the string "normals"."""
    import trimesh
    scene = trimesh.Scene()
    for i, (V, F, col) in enumerate(meshes):
        if isinstance(col, str) and col == "normals":
            rgba = normal_colors(V, F)
        else:
            col = np.asarray(col, np.uint8)
            if col.ndim == 1:
                col = np.tile(col[None], (len(V), 1))
            if col.shape[1] == 3:
                col = np.concatenate([col, np.full((len(col), 1), 255, np.uint8)], 1)
            rgba = np.concatenate([srgb_to_linear(col[:, :3]), col[:, 3:]], 1)
        scene.add_geometry(trimesh.Trimesh(np.asarray(V) * np.array([1.0, -1.0, -1.0]), F, vertex_colors=rgba,
                                           process=False), node_name=f"mesh_{i}")
    return scene.export(file_type="glb")
