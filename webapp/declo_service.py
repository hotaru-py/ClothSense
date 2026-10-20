"""M7 job service for the DeClotH demo: photo -> preprocessing -> draft meshes -> geometry stage (snapshots
streamed as GLB) -> texture stage -> final textured cloth + body, saved to a library for garment swaps.
Run and browser-tested (work/declo/test_webapp.py). Loaded by webapp/serve.py.

Job state (GET /api/jobs/<id>): status, stage, progress, step/total, latest snapshot version, captions, metrics.
Files (GET /api/jobs/<id>/<name>): photo.png, v<k>_cloth.glb, v<k>_body.glb (v0 = templates), final_*.glb,
tryon_<other job or library id>.glb (computed on request).
Clothing options = garment-only files shipped in garments/ (ids "g_<name>", tracked in git) + this machine's
finished reconstructions in work/declo/library/ (ids = job ids; full bundles with the photo, never shipped).

VRAM: preprocessing (~4.6 GB peak) and SD + 2 ControlNets (~3.8 GB) do not fit together on 8 GB, so each job
loads the preprocessing models, frees them, then loads the diffusion stack (~1 min of loading per job).
"""
from __future__ import annotations

import io
import queue
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "work/declo/library"
GARMENTS = ROOT / "garments"
_ID = __import__("re").compile(r"^[A-Za-z0-9_]{1,64}$")


def item_path(item_id: str) -> Path:
    """Library id -> bundle file (FileNotFoundError for unknown or malformed ids)."""
    if not _ID.match(item_id):
        raise FileNotFoundError(item_id)
    p = GARMENTS / f"{item_id[2:]}.npz" if item_id.startswith("g_") else LIBRARY / f"{item_id}.npz"
    if not p.exists():
        raise FileNotFoundError(item_id)
    return p
SKIN = (200, 170, 150, 255)


def _glb(meshes) -> bytes:
    from declo.export import glb_bytes                  # sRGB -> linear colours, OpenCV -> glTF frame
    return glb_bytes(meshes)


class DeclothJob:
    def __init__(self, photo: bytes, hfov):
        self.id = uuid.uuid4().hex[:12]
        self.photo, self.hfov = photo, hfov
        self.state = {"id": self.id, "status": "queued", "stage": "Waiting for the GPU", "progress": 0.0,
                      "step": 0, "total": 0, "version": None, "captions": None, "metrics": None, "error": None}
        self.files = {}
        self.bundle = None


class DeclothService:
    kind = "declo"

    def __init__(self, geo_steps=150, tex_steps=200, encoder="taesd", snapshot_every=25):
        sys.path.insert(0, str(ROOT))
        self.geo_steps, self.tex_steps, self.encoder, self.every = geo_steps, tex_steps, encoder, snapshot_every
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")    # all models are cached: a dropped connection killed a run
        import declo
        declo.cap_vram()
        from declo import hmr as B
        self.smplx = B.load_smplx(ROOT / "models", "cuda")       # small; kept for re-posing / try-on requests
        self.jobs, self.queue = {}, queue.Queue()
        LIBRARY.mkdir(parents=True, exist_ok=True)
        threading.Thread(target=self._worker, daemon=True).start()

    def submit(self, photo, hfov="auto"):
        job = DeclothJob(photo, hfov)
        self.jobs[job.id] = job
        self.queue.put(job)
        return job

    def library(self):
        """Clothing options: id, the garment caption (BLIP) and whether it is a shipped garment."""
        import json
        out = []
        for d, shipped in ((GARMENTS, True), (LIBRARY, False)):
            for p in sorted(d.glob("*.npz"), key=lambda q: q.stat().st_mtime):
                meta = p.with_suffix(".json")
                caps = (json.loads(meta.read_text()).get("captions") or {}) if meta.exists() else {}
                out.append({"id": f"g_{p.stem}" if shipped else p.stem, "shipped": shipped,
                            "caption": caps.get("cloth") or caps.get("wearing") or p.stem})
        return out

    def thumbnail(self, item_id) -> bytes:
        """Front view of the garment (PNG with alpha), made once and cached next to the bundle."""
        p = item_path(item_id).with_suffix(".png")
        if not p.exists():
            from PIL import Image
            from declo.swap import garment_thumbnail, load_bundle
            Image.fromarray(garment_thumbnail(load_bundle(p.with_suffix(".npz")))).save(p)
        return p.read_bytes()

    def _worker(self):
        while True:
            job = self.queue.get()
            try:
                self._run(job)
            except Exception as e:
                import traceback
                traceback.print_exc()
                job.state.update(status="error", error=f"{type(e).__name__}: {e}", stage="Failed")
            finally:
                job.photo = None

    def _snapshot(self, job, rec, version, colors=None):
        for name, (V, F) in rec.meshes().items():
            job.files[f"v{version}_{name}.glb"] = _glb([(V, F, colors[name] if colors else "normals")])
        job.state["version"] = version

    def _run(self, job):
        import torch
        from PIL import Image
        from declo.guidance import Guidance
        from declo.prep import Preprocessor, load_photo
        from declo.recon import Reconstruction
        from declo.swap import save_bundle
        from declo.texture import TextureStage
        st, t0 = job.state, time.time()
        st["status"] = "running"
        log = lambda s: st.update(stage=s)
        st.update(stage="Loading the photo models", progress=0.02)
        pre = Preprocessor(log=lambda *a: None)
        pil, focal = load_photo(io.BytesIO(job.photo))
        st.update(stage="Reading the photo", progress=0.05)
        d = pre(pil, focal_px=focal, hfov=job.hfov, log=lambda *a: None)
        st.update(captions=d["captions"], prep_s=round(time.time() - t0, 1))
        b = io.BytesIO()
        Image.fromarray(d["rgb"]).save(b, "PNG")
        job.files["photo.png"] = b.getvalue()
        pre.free()
        del pre
        torch.cuda.empty_cache()

        st.update(stage="Building the cloth and body volumes", progress=0.12)
        rec = Reconstruction(d, self.smplx, cfg={"steps": self.geo_steps}, log=lambda *a: None)
        Vb, Fb = rec.T_posed["body"]
        Vc, Fc = rec.T_posed["cloth"]
        job.files["v0_body.glb"] = _glb([(Vb.cpu().numpy(), Fb.cpu().numpy(), "normals")])
        job.files["v0_cloth.glb"] = _glb([(Vc.cpu().numpy(), Fc.cpu().numpy(), "normals")])
        st["version"] = 0
        st.update(stage="Loading the diffusion models", progress=0.15)
        G = Guidance(rec, d, self.smplx, encoder=self.encoder)
        total = self.geo_steps + self.tex_steps
        st["total"] = total
        for i in range(self.geo_steps):
            rec.step(extra_losses=G)
            st.update(step=i + 1, stage=f"Shaping cloth and body ({i + 1}/{self.geo_steps})",
                      progress=0.15 + 0.6 * (i + 1) / total)
            if (i + 1) % self.every == 0 or i + 1 == self.geo_steps:
                self._snapshot(job, rec, i + 1)
        st["metrics"] = rec.metrics()
        tx = TextureStage(rec, d, guidance=G, steps=self.tex_steps)
        for i in range(self.tex_steps):
            tx.step()
            k = self.geo_steps + i + 1
            st.update(step=k, stage=f"Painting ({i + 1}/{self.tex_steps})", progress=0.15 + 0.85 * k / total)
            if (i + 1) % (2 * self.every) == 0 or i + 1 == self.tex_steps:
                self._snapshot(job, rec, k, colors=tx.vertex_colors())
        cols = tx.vertex_colors()
        for name, (V, F) in rec.meshes().items():
            job.files[f"final_{name}.glb"] = _glb([(V, F, cols[name])])
        job.bundle = save_bundle(LIBRARY / f"{job.id}.npz", d, rec, colors=cols)
        self.thumbnail(job.id)
        st.update(status="done", stage="Done", progress=1.0, total_s=round(time.time() - t0, 1))

    def tryon(self, job_id, other_id) -> bytes:
        """The other person's garment on this job's person (DeClotH-style swap), as one GLB."""
        from declo.swap import load_bundle, try_on
        A, B = load_bundle(item_path(other_id)), load_bundle(item_path(job_id))
        r = try_on(self.smplx, A, B)
        ca, cb = A.get("cloth_colors"), B.get("body_colors")
        return _glb([(r["cloth"][0], r["cloth"][1], ca if ca is not None else "normals"),
                     (r["body"][0], r["body"][1], cb if cb is not None else "normals")])
