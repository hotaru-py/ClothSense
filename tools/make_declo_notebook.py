"""Builds declo.ipynb, the DeClotH-rework notebook (script so it stays reproducible and diffable).

    .venv/Scripts/python tools/make_declo_notebook.py declo.ipynb
"""
import json
import sys

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)})


md(r"""
# DeClotH, rebuilt: decomposed 3D cloth + body from one photo

**3D Vision Techniques, semester project.** This follows the DeClotH pipeline (Nam et al., CVPR 2025) step by step.
The paper released no code, so each step is built from released parts (`PLAN_DECLOTH.md` lists every substitution).
The earlier feed-forward sizing app (v1) is archived in `bak/v1_fitlab_2026-10-06/`.

```
photo ─► (1) preprocessing ─┬─ silhouettes S (human, cloth)  ── SAM, prompted from SegFormer parsing
                            ├─ normal maps N (front, back)    ── ECON normal estimator
                            ├─ body template  M_body^t        ── BodyNet  = Multi-HMR → SMPL-X
                            └─ cloth template M_cloth^t       ── ClothNet = ClothWild → SMPLicit
      (2) two DMTets (cloth, body) in canonical A-pose ─► Marching Tetrahedra ─► SMPL-X LBS ─► photo pose
      (3) differentiable renders (silhouette / normal / RGB) from the photo camera + random cameras
      (4) template loss  (5) cloth SDS  (6) human SDS  (7) geometry stage  (8) texture stage
      (9) separate textured cloth + body meshes → re-pose, swap garments between people
```

| milestone | status |
|---|---|
| M1 preprocessing (this notebook, §1–2) | masks, body template, body normal renders: run. ECON normals: need ICON data. Cloth template (ClothWild): written, needs SMPL |
| M2 DMTet ×2 + LBS + renderer | run (pure-PyTorch renderer; nvdiffrast backend written, not run) |
| M3–M5 geometry, diffusion, texture | written; M3 run, M4 partly run, M5 not run |
| M6–M7 re-posing, swap, web app | not started |
""")

md("## 0. Setup")
code(r"""
import sys, json, time
from pathlib import Path
import numpy as np, torch
from IPython.display import display, Image as IPImage

PROJECT = Path.cwd()
sys.path.insert(0, str(PROJECT))
import declo; declo.cap_vram()          # stay inside dedicated VRAM (Windows spills to system RAM, 4-6x slower)
print("torch", torch.__version__, "CUDA", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
need = {
    "SMPL-X (BodyNet)":            PROJECT / "models/smplx/SMPLX_NEUTRAL.npz",
    "SAM ViT-H (silhouettes)":     PROJECT / "models/sam/sam_vit_h_4b8939.pth",
    "ECON normal net (normals)":   PROJECT / "work/ECON/data/ckpt/normal.ckpt",
    "SMPL (ClothNet)":             PROJECT / "models/smpl/SMPL_NEUTRAL.pkl",
    "ClothWild weights (ClothNet)": PROJECT / "work/ClothWild/demo/snapshot_7.pth.tar",
}
for k, p in need.items():
    print(f"  {'ok     ' if p.exists() else 'MISSING'} {k:30s} {p.relative_to(PROJECT)}")
try:
    import nvdiffrast.torch  # noqa: F401
    print("  ok      nvdiffrast (M2+)")
except ImportError:
    print("  MISSING nvdiffrast (M2+): needs CUDA Toolkit 12.6 + VS 2022 Build Tools, then pip install")
""")

md(r"""
## 1. Photo
Whole body, arms slightly away from the torso, 2.5–3 m away. The focal length comes from EXIF
(`FocalLengthIn35mmFilm`); without it the camera FOV is estimated (mesh-vs-mask IoU, v1's auto-FOV).
""")
code(r"""
from declo.prep import load_photo
IMG_PATH = PROJECT / "assets/test.jpeg"
pil, FOCAL = load_photo(IMG_PATH)
print(pil.size, "EXIF focal px:", FOCAL)
pil.resize((pil.width * 400 // pil.height, 400))
""")

md(r"""
## 2. Preprocessing: DeClotH Sec. 3.2 (M1)
* **Work crop**: ECON's convention. The person box spans 0.9 of a 512 px square; every map and render below lives on it.
* **Silhouettes S**: the paper prompts SAM with text. SAM itself takes boxes and points, so we ground
  the "text" with SegFormer clothes parsing: each region's box, interior points (positive) and neighbouring
  regions (negative) go to SAM ViT-H. The printed IoU says how much SAM agrees with the parsing.
  *cloth* = upper ∪ lower, the region of DeClotH's single cloth DMTet.
* **BodyNet**: Multi-HMR → SMPL-X, the paper's PIXIE → SMPL+H. Also gives the camera. Its front/back
  normal renders are the extra input of ECON's normal net.
* **Normals N**: ECON's normal estimator, front and back (the back map is mirrored to line up with the photo).
* **ClothNet**: ClothWild → SMPLicit cloth template, in the same camera frame.
""")
code(r"""
from declo.prep import Preprocessor, overview
pre = Preprocessor()
OUT = PROJECT / "work/declo" / IMG_PATH.stem
t = time.time()
D = pre(pil, focal_px=FOCAL, out_dir=OUT)
print(f"preprocessing {time.time() - t:.1f}s -> {OUT.relative_to(PROJECT)}")
""")
code(r"""
overview(D, OUT / "m1_overview.png")
display(IPImage(filename=str(OUT / "m1_overview.png")))
""")

md(r"""
## 3. Prompts (BLIP)
DeClotH writes its prompts with BLIP. BLIP describes the garments best on the whole person
("a photo of a person wearing …"); the upper garment's own caption adds detail. Cloth prompts get the
garment-only wording of our ClothDiffusion stand-in ("ghost mannequin … clothes only") plus person/skin negatives.
""")
code(r"""
from declo.captions import prompts
print(D["captions"])
for k, (p, n) in prompts(D["captions"]).items():
    print(f"{k:6s} + {p}
       - {n}")
""")

md(r"""
## 4. Geometry stage without diffusion (M3)
Two DMTets (cloth, body) on one band-limited tet grid (1 cm cells) in the canonical A-pose, initialised from the
templates, Marching Tetrahedra, SMPL-X LBS to the photo pose, differentiable renders (pure-PyTorch renderer with
an analytic silhouette antialias; nvdiffrast once the CUDA toolkit is installed). Losses: template silhouettes
(photo + orbit views), SAM silhouettes (cloth; cloth+body composite vs the human mask), penetration, SDF
regulariser; ECON normals join when installed. Until ClothWild runs, the cloth template is a stand-in: the body
surface under the SAM cloth mask, 1 cm outward.
""")
code(r"""
from declo.recon import Reconstruction
SMPLX = pre.body.smplx
pre.free()
rec = Reconstruction(D, SMPLX, cfg={"steps": 300})
print("start", rec.metrics())
t = time.time()
for i in range(rec.cfg["steps"]):
    L = rec.step()
    if i % 50 == 0:
        print(i, {k: round(v, 4) for k, v in L.items()})
print("end", rec.metrics(), f"{time.time() - t:.0f}s")
""")

md(r"""
## 5. Geometry stage with diffusion (M4)
Adds DeClotH's two SDS losses on rendered **normal maps** from a random orbit camera each step:
cloth SDS (ControlNet seg with the cloth template's silhouette painted ADE20K "apparel", garment-only prompt:
our ClothDiffusion stand-in) and human SDS (ControlNet OpenPose with the template skeleton, on the body and
on cloth+body). SDS weights 1e-5 / 5e-6 balance its gradients against the reconstruction losses (measured,
provisional). `encoder="taesd"` is ~6x faster than SD's VAE for previews; use "vae" for final results.
Run this instead of section 4 (both optimise the same Reconstruction).
""")
code(r"""
from declo.guidance import Guidance
G = Guidance(rec, D, SMPLX, encoder="taesd")
print(G.prompts["cloth"][0])
t = time.time()
for i in range(rec.cfg["steps"]):
    L = rec.step(extra_losses=G)
    if i % 50 == 0:
        print(i, {k: round(v, 4) for k, v in L.items()}, f"{(time.time() - t) / (i + 1):.2f} s/step")
print("end", rec.metrics())
""")

md(r"""
## 6. Texture stage (M5)
Geometry frozen. Two colour fields (cloth, person) on the canonical surface, queried per pixel; L2 + LPIPS
against the photo (front view) + RGB SDS through the same ControlNets.
""")
code(r"""
from declo.texture import TextureStage
TX = TextureStage(rec, D, guidance=G, steps=300)
t = time.time()
for i in range(300):
    L = TX.step()
    if i % 50 == 0:
        print(i, {k: round(v, 4) for k, v in L.items()}, f"{(time.time() - t) / (i + 1):.2f} s/step")
""")

md(r"""
## 7. Export
Separate textured meshes in the photo pose (GLB, per-vertex colour), the cloth and the body each on their own.
""")
code(r"""
from declo.export import glb_bytes                      # OpenCV -> glTF frame, sRGB -> linear colours
cols = TX.vertex_colors()
for name in ("cloth", "body"):
    V, F, _ = TX.mesh[name]
    (OUT / f"{name}.glb").write_bytes(glb_bytes([(V.cpu().numpy(), F.cpu().numpy(), cols[name])]))
print("wrote", [str((OUT / f"{n}.glb").relative_to(PROJECT)) for n in ("cloth", "body")])
""")

md(r"""
## 8. Next milestones
* **M6**: re-posing (canonical meshes + SMPL-X LBS) and garment swap onto another person's body.
* **M7**: web app (capture -> job -> streamed meshes -> 3D viewer: body / cloth / both, clothing options).
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for i, cell in enumerate(nb["cells"]):
    cell["id"] = f"c{i:02d}"
json.dump(nb, open(sys.argv[1], "w"), indent=1)
print("wrote", sys.argv[1], len(cells), "cells")
