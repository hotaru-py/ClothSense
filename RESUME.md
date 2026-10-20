# Where we stopped (20 Oct 2026): SMPL / ECON / nvdiffrast in; ClothWild opt-in; long-run drift under study

v1 (feed-forward sizing app) is archived in `bak/v1_fitlab_2026-10-06/`. Plan, decisions and every measured number:
`PLAN_DECLOTH.md`. Code: `declo/`; notebook `declo.ipynb` (M1-M5 + export); tests and outputs: `work/declo/`.

## Run it
```
.venv/Scripts/python webapp/serve.py                  # then http://localhost:8000  (~10 min per photo)
.venv/Scripts/python tools/run_full.py <photo>   # same pipeline headless -> work/declo/<stem>/ (GLBs, bundle)
.venv/Scripts/python tools/run_full.py <photo> 0 200   # texture stage only, reusing the saved geometry
.venv/Scripts/python work/declo/test_webapp.py        # browser test against a running --declo server
```
Clothing options = bundles in `work/declo/library/` (demo_test, demo_sachin + every finished web job).

## State (all run unless marked)
- Photo -> SAM masks, Multi-HMR body, BLIP prompts (33 s) -> two DMTets (body; tear-free cloth shell) -> 150 steps
  silhouette + template + penetration + cloth/human SDS (4.2 min) -> 200 texture steps (3.5 min) -> GLBs. 9-11 min,
  peak 6.4 GB. Web app streams the meshes every 25 steps; garment swap from the library in ~4 s.
- ECON front + back normals in use. ClothWild runs (opt-in, DECLO_CLOTHWILD=1) but hurts. nvdiffrast installed,
  hybrid path (DECLO_RENDER=nvdiffrast) converged worse -> default renderer "torch".

## Known weak spots
- (Fixed 7 Oct) body poking through / cloth cut off: garment region, shell-aware penetration loss, arm-aware
  silhouette, one-sided body template, body aligned to the photo. Poke-through now ~1.5 % of cloth pixels (mostly
  rims at the openings). Diagnostics: work/declo/diag_pokethrough.py, diag_sil.py, diag_patch.py, diag_skin.py.
- Surfaces still bumpy: paper hash grid (max res 1024) is rougher at 150 steps than the old net; Sapiens front
  normals (w_normal 0.05) help the cloth a bit but widen it (PLAN_DECLOTH.md, 18 Oct table). Dark specks from SDS.
- Web job once failed with an empty cloth shell / no-grad SDS (guarded, clear error now); root cause not found,
  the same photo is fine offline.
- Swaps expose body regions that were under the person's own clothes: their texture is SDS-only and not skin yet.
- Photo overlay of a swapped garment shows the original clothes where the new one is narrower (needs inpainting).
- GLBs are big (try-on 13.7 MB); decimate for phones.

## Next
0. Decide: keep paper net + normals, or lower w_normal (0.02) / start it after ~50 steps; one 10-min run each.
1. Owner: SMPL + ICON registrations, CUDA Toolkit 12.6 + VS 2022 Build Tools (see PLAN_DECLOTH.md).
2. ECON normals into L_normal; ClothWild template; nvdiffrast backend (render.BACKEND) -> faster, smoother.
3. Longer offline schedule for showcase results; phone test. Later: garment sizing on the reconstruction
   (v1's measure / fit / draper code is in bak/).
