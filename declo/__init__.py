"""declo - DeClotH (Nam et al., CVPR 2025) rebuilt from released parts, for the 3DVT course project.

One photo -> a separate 3D body and 3D garment (two DMTets refined with diffusion), textured, re-posable and
swappable between people. Plan, decisions and measured numbers: PLAN_DECLOTH.md.

  parsing    SegFormer clothes parsing (ATR labels)
  masks      (1) human / cloth silhouettes: SAM prompted from the parsing
  hmr        Multi-HMR -> SMPL-X, camera, canonical A-pose, auto-FOV
  templates  (1) BodyNet (Multi-HMR) and ClothNet (ClothWild -> SMPLicit), SMPL-X -> SMPL bridge
  normals    (1) ECON front / back normal maps        captions  prompts (BLIP)
  frame / raster / render / cameras / skeleton        crop, rasterisers, differentiable renderer, views, OpenPose
  dmtet / skinning                                    (2) DMTets (body; tear-free cloth shell), LBS
  recon / sds / guidance                              (3)-(7) geometry stage with template, silhouette, SDS losses
  texture                                             (8) texture stage
  swap / export                                       (9) bundles, re-posing, garment swap, GLB export
  prep                                                M1 orchestration

Conventions: camera frame = OpenCV (x right, y down, z forward), metres. Canonical frame: A-pose at 45.5 deg,
y up, +z facing forward, feet at y=0, torso centred (the frame GarmentCode garments use, kept for sizing later).
"""
__version__ = "0.1.0"


def cap_vram(fraction=0.9):
    """Keep PyTorch under `fraction` of the GPU's dedicated memory: its caching allocator then frees and reuses
    cached blocks instead of reserving more. On Windows (WDDM) going past dedicated memory does not fail, it
    spills into shared system memory and runs 4-6x slower (seen in M4); with the cap a real shortage raises OOM.
    Expandable segments against fragmentation (a 4000-step run died with 630 MB reserved but unused); must be
    set before the first CUDA allocation."""
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction)
