"""Whole pipeline on one photo: preprocessing -> geometry stage with SDS (M4) -> texture stage (M5) -> export.

    python tools/run_full.py <photo> [geo_steps=150] [tex_steps=200] [encoder=taesd]
-> work/declo/<stem>/: full_overview.png (photo, textured renders around, normals), cloth.glb, body.glb,
   bundle_full.npz (for garment swaps), full.log numbers on stdout
"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import declo; declo.cap_vram()
import os; os.environ.setdefault("HF_HUB_OFFLINE", "1")   # models are cached; no network round-trips
import numpy as np, torch, trimesh
from PIL import Image
from declo.prep import Preprocessor, load_photo
from declo.recon import Reconstruction
from declo.guidance import Guidance
from declo.texture import TextureStage
from declo.swap import save_bundle

photo = sys.argv[1]
geo = int(sys.argv[2]) if len(sys.argv) > 2 else 150
tex = int(sys.argv[3]) if len(sys.argv) > 3 else 200
enc = sys.argv[4] if len(sys.argv) > 4 else "taesd"
OUT = ROOT / "work/declo" / Path(photo).name.split(".")[0]; OUT.mkdir(parents=True, exist_ok=True)
T = {}
t = time.time()
pre = Preprocessor(log=lambda *a: None)
pil, focal = load_photo(photo)
d = pre(pil, focal_px=focal, log=lambda *a: None)
smplx_model = pre.body.smplx
pre.free(); del pre; torch.cuda.empty_cache()
T["prep"] = time.time() - t
print("captions:", d["captions"], flush=True)

t = time.time()
import json                                                     # experiments: DECLO_CFG='{"w_normal": 0.2}'
rec = Reconstruction(d, smplx_model, cfg={"steps": max(geo, 1), **json.loads(os.environ.get("DECLO_CFG", "{}"))},
                     log=lambda *a: None)
G = Guidance(rec, d, smplx_model, encoder=enc)
T["init"] = time.time() - t
t = time.time()
state = OUT / "geo_state.pt"
if geo == 0 and state.exists():                      # texture-only rerun: reuse the saved geometry stage
    for k, sd in torch.load(state, weights_only=False).items():
        rec.nets[k].load_state_dict(sd)
    print("geometry: loaded", state, flush=True)
rough = lambda V, F: np.degrees(trimesh.Trimesh(V, F, process=False).face_adjacency_angles).mean()
every = int(os.environ.get("DECLO_METRICS_EVERY", 0))
PEAK = 0        # long runs: metrics along the way
stop = int(os.environ.get("DECLO_GEO_STOP", geo))          # diagnostics: the long schedule, cut short
for i in range(min(geo, stop)):
    L = rec.step(extra_losses=G)
    if i % 50 == 0 or i == geo - 1:
        base = torch.cuda.memory_allocated()                     # between steps: grows only if something leaks
        with torch.no_grad():
            from declo.render import silhouette_edges
            nf, ns = {}, {}
            for k in ("cloth", "body"):
                V, F, _ = rec.posed(k)
                nf[k], ns[k] = len(F), int(silhouette_edges(V, F).sum())
        print(f"  geo {i:4d} sil {L['sil']:.4f} tmpl {L['tmpl']:.4f}  ({(time.time() - t) / (i + 1):.2f} s/step)  "
              f"faces {nf}, silhouette edges (photo view) {ns}, VRAM between steps {base / 1e9:.2f} GB, "
              f"peak since last {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)
        PEAK = max(PEAK, torch.cuda.max_memory_allocated())
        torch.cuda.reset_peak_memory_stats()
    if every and i and i % every == 0:
        with torch.no_grad():
            m = {k: round(v, 4) for k, v in rec.metrics().items()}
            r = {k: round(rough(*v), 1) for k, v in rec.meshes(posed=False).items()}
        print(f"  @{i}: {m} roughness {r}", flush=True)
if geo:
    torch.save({k: n.state_dict() for k, n in rec.nets.items()}, state)
T["geometry"] = time.time() - t
print("geometry:", rec.metrics(), "roughness", {k: round(rough(*v), 1) for k, v in rec.meshes(posed=False).items()}, flush=True)

t = time.time()
tx = TextureStage(rec, d, guidance=G, steps=tex)
for i in range(tex):
    L = tx.step()
    if i % 50 == 0 or i == tex - 1:
        print(f"  tex {i:4d} " + " ".join(f"{k} {v:.4g}" for k, v in L.items()) + f"  ({(time.time() - t) / (i + 1):.2f} s/step)", flush=True)
T["texture"] = time.time() - t

cols = tx.vertex_colors()
from declo.export import glb_bytes
for name in ("cloth", "body"):
    V, F, _ = tx.mesh[name]
    (OUT / f"{name}.glb").write_bytes(glb_bytes([(V.cpu().numpy(), F.cpu().numpy(), cols[name])]))
save_bundle(OUT / "bundle_full.npz", d, rec, colors=cols)

with torch.no_grad():                                            # overview: photo | front textured | 4 orbit views
    img = lambda names, K, R=None, t=None: tx.render(names, K, R, t)[0].clamp(0, 1).cpu().numpy()
    row1 = [np.asarray(Image.fromarray(d["rgb"]).resize((tx.res, tx.res))) / 255.0, img(("cloth", "body"), tx.K),
            img(("cloth",), tx.K), img(("body",), tx.K)]
    row2 = []
    for az in (45, 135, 180, 270):
        K, R, tt = rec.orbit.camera(az, 5)
        K = np.diag([tx.res / rec.res, tx.res / rec.res, 1.0]) @ K
        row2.append(img(("cloth", "body"), torch.tensor(K, dtype=torch.float32, device="cuda"), R, tt))
pic = np.concatenate([np.concatenate(row1, 1), np.concatenate(row2, 1)], 0)
Image.fromarray((pic * 255).astype(np.uint8)).resize((4 * 384, 2 * 384)).save(OUT / "full_overview.png")
print("times:", {k: round(v) for k, v in T.items()}, f"total {sum(T.values()) / 60:.1f} min, "
      f"peak {max(PEAK, torch.cuda.max_memory_allocated()) / 1e9:.2f} GB", flush=True)
print("wrote", OUT / "full_overview.png", OUT / "cloth.glb", OUT / "body.glb", OUT / "bundle_full.npz")
