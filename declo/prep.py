"""M1: DeClotH preprocessing (paper Sec. 3.2) - photo -> optimisation targets and templates.

    pre = Preprocessor()                      # loads SegFormer, SAM, Multi-HMR (+ ECON, ClothWild if present)
    d = pre(load_photo("assets/test.jpeg"), out_dir="work/declo/test")

Outputs, all on the 512 px work crop: image, human / cloth / upper / lower silhouettes (S), body
template (SMPL-X, camera frame + canonical A-pose), its normal renders, ECON normal maps (N) and the
ClothWild cloth template when their weights are installed. Saved to out_dir/prep.npz + prep.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from . import hmr as B
from .parsing import HumanParser

from . import raster
from .frame import WorkFrame
from .masks import Segmenter, iou
from .templates import BodyNet

ROOT = Path(__file__).resolve().parents[1]
SAM_RES = 1024


def load_photo(path):
    """Upright RGB photo + focal length in px from EXIF (None without FocalLengthIn35mmFilm).
    The 35 mm focal is defined on the diagonal, so it survives the EXIF rotation."""
    raw = Image.open(path)
    focal = B.focal_px_from_exif(raw)
    return ImageOps.exif_transpose(raw).convert("RGB"), focal


class Preprocessor:
    def __init__(self, device=None, log=print):
        t = time.time()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.parser = HumanParser(device=self.device)
        self.seg = Segmenter(device=self.device, parser=self.parser)
        self.body = BodyNet(self.device)
        from .captions import Captioner
        self.captioner = Captioner(self.device)
        self.normals = self.cloth = None
        try:
            from .normals import NormalEstimator
            self.normals = NormalEstimator(device=self.device).to("cpu")   # on the GPU only while it runs
        except FileNotFoundError as e:
            log(f"[skip] ECON normals: {e}")
        self.front_normals = None
        if self.normals is None:                 # ECON's released code takes the front normal from Sapiens
            try:
                from .normals import SapiensNormal
                self.front_normals = SapiensNormal(device=self.device)
                log("front normals: Sapiens-normal 0.3B (back normal needs ECON)")
            except Exception as e:                # not downloaded / offline
                log(f"[skip] Sapiens normals: {type(e).__name__}: {e}")
        # ClothWild template: opt-in (DECLO_CLOTHWILD=1). Measured worse here: its garments are looser and longer
        # than the photo's (sleeves over the forearms: cloth over arm 40 % vs 7 %, cloth IoU 0.81 vs 0.94 at 150
        # steps; still 31 % at 4000 steps). Default = the mask-based stand-in (recon.cloth_patch_from_mask).
        import os
        try:
            if not os.environ.get("DECLO_CLOTHWILD"):
                raise NotImplementedError("off by default (set DECLO_CLOTHWILD=1); see PLAN_DECLOTH.md, 19-20 Oct")
            from .templates import ClothNet, load_smpl
            self.cloth = ClothNet(device=self.device)
            self.smpl = {g: load_smpl(ROOT / "models", self.device, g) for g in ("male", "female")}
        except (FileNotFoundError, NotImplementedError) as e:
            log(f"[skip] ClothNet: {e}")
        log(f"Preprocessor ready in {time.time() - t:.0f}s")

    def free(self):
        """Release every preprocessing model (the optimisation needs the GPU for SD + ControlNet)."""
        for k in ("parser", "seg", "body", "normals", "front_normals", "cloth", "captioner", "smpl"):
            setattr(self, k, None)
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    def __call__(self, pil, focal_px=None, hfov="auto", out_dir=None, log=print) -> dict:
        T = {}
        t = time.time()
        rgb_full = np.asarray(pil)
        self.seg.sam.model.to(self.device)                       # parked in RAM by the previous call
        person = self.parser(pil)["person"]
        frame = WorkFrame.from_mask(person, res=512)
        T["parse"] = time.time() - t

        t = time.time()                                          # (1) BodyNet
        body = self.body(pil, person_mask=person, focal_px=focal_px, hfov=hfov)
        K = frame.K(body["K_orig"])
        T["body"] = time.time() - t
        log(f"body: focal {body['focal_px']:.0f} px from {body['focal_src']}")

        t = time.time()                                          # (1) silhouettes, SAM on a 1024 crop
        crop_hi = frame.warp(rgb_full, SAM_RES)
        Jr = self.body.smplx.J_regressor.detach().cpu().numpy()
        J = Jr @ body["V_cam"]                                   # posed joints, camera frame
        hips = frame.K(body["K_orig"], SAM_RES) @ J[[1, 2]].T
        hip_row = float((hips[1] / hips[2]).mean())
        m = self.seg(crop_hi, hip_row=hip_row)
        masks = {k: frame_down(m[k]) for k in ("human", "cloth", "upper", "lower")}
        from .templates import align_body                     # put the body template where the photo has it
        align_body(body, masks["human"], frame_down(m["labels"], nearest=True), K, log=log)
        T["sam"] = time.time() - t
        log(f"masks: SAM vs parsing IoU {json.dumps(m['scores'])}")

        t = time.time()                                          # prompts (paper: BLIP captions)
        caps = self.captioner(frame.warp(rgb_full), masks)
        T["captions"] = time.time() - t
        log(f"captions: {caps}")

        t = time.time()                                          # body template renders on the crop
        Vb = torch.tensor(body["V_cam"], dtype=torch.float32, device=self.device)
        Tn, Tm = raster.normal_maps(Vb, body["F"], K, frame.res)
        sil_iou = iou(Tm[0].cpu().numpy(), masks["human"])
        log(f"body template silhouette vs SAM human mask: IoU {sil_iou:.3f}")
        T["render"] = time.time() - t

        rgb = frame.warp(rgb_full)
        N = None
        if self.normals is not None:                             # (1) ECON normal maps
            t = time.time()
            self.seg.sam.reset_image()                           # SAM is done: swap it out for ECON (8 GB)
            self.seg.sam.model.cpu()
            torch.cuda.empty_cache()
            self.normals.to(self.device)
            N = self.normals(rgb, masks["human"], Tn)
            self.normals.to("cpu")
            T["normals"] = time.time() - t
        elif self.front_normals is not None:                     # front only (back stays 0 = no target)
            t = time.time()
            Nf = self.front_normals(crop_hi, masks["human"])
            N = torch.stack([Nf, torch.zeros_like(Nf)])
            T["normals"] = time.time() - t
        cloth_t = None
        if self.cloth is not None:                               # (1) ClothNet
            t = time.time()
            from .templates import cloth_patch, smpl_vertices, to_smpl
            g = self.cloth.gender(rgb_full, person)              # ClothWild dresses the gendered SMPL
            sp = to_smpl(self.body.smplx, self.smpl[g], body, log=log)
            cloth_t = self.cloth(rgb_full, person, sp)
            cloth_t["smpl"] = sp
            cloth_t["patch"] = cloth_patch(body["V_cam"], cloth_t["garments"], self.body.smplx)
            off = np.abs(cloth_t["smpl_mesh"] - smpl_vertices(self.smpl[g], sp)).max()
            log(f"ClothNet: {sorted(cloth_t['garments'])}, gender {cloth_t['gender']}; its SMPL vs ours max "
                f"{1000 * off:.1f} mm (should be ~0: same parameters), {cloth_t['patch'].sum()} SMPL-X vertices under cloth")
            T["cloth"] = time.time() - t

        d = {"rgb": rgb, "masks": masks, "labels": frame_down(m["labels"], nearest=True), "frame": frame,
             "K": K, "body": body, "T_normal": Tn.cpu(), "T_mask": Tm.cpu(), "normals": N, "cloth_t": cloth_t,
             "sam_scores": m["scores"], "body_sil_iou": sil_iou, "captions": caps, "times": T}
        log("times: " + ", ".join(f"{k} {v:.1f}s" for k, v in T.items()))
        if out_dir:
            save(d, out_dir)
        return d


def frame_down(mask_hi: np.ndarray, res=512, nearest=False) -> np.ndarray:
    if nearest:
        return cv2.resize(mask_hi, (res, res), interpolation=cv2.INTER_NEAREST)
    return (cv2.resize(mask_hi.astype(np.float32), (res, res), interpolation=cv2.INTER_AREA) > 0.5).astype(np.uint8)


def save(d: dict, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    b = d["body"]
    arr = {"rgb": d["rgb"], "labels": d["labels"], "K": d["K"], "T_normal": d["T_normal"].numpy(),
           "body_V_cam": b["V_cam"], "body_F": b["F"], "body_R": b["R"], "body_c": b["c"], "betas": b["betas"],
           "rotvec": b["hmr"]["rotvec"], "expression": b["hmr"]["expression"],
           "canon_V": b["canon"]["V"], "canon_body_pose": b["canon"]["body_pose"], "canon_offset": b["canon"]["offset"]}
    arr.update({f"mask_{k}": v for k, v in d["masks"].items()})
    if d["normals"] is not None:
        arr["normals"] = d["normals"].numpy()
    if d["cloth_t"] is not None:
        for k, (V, F) in d["cloth_t"]["garments"].items():
            arr[f"cloth_{k}_V"], arr[f"cloth_{k}_F"] = V, F
        arr["cloth_patch"] = d["cloth_t"]["patch"]
        arr.update({f"smpl_{k}": v for k, v in d["cloth_t"]["smpl"].items() if isinstance(v, np.ndarray)})
    np.savez_compressed(out / "prep.npz", **arr)
    meta = {"frame": d["frame"].to_dict(), "focal_px": b["focal_px"], "focal_src": b["focal_src"],
            "fov_scores": b["fov_scores"], "sam_scores": d["sam_scores"], "body_sil_iou": d["body_sil_iou"],
            "captions": d["captions"], "times": d["times"], "has_normals": d["normals"] is not None, "has_cloth_template": d["cloth_t"] is not None}
    (out / "prep.json").write_text(json.dumps(meta, indent=1, default=float))
    Image.fromarray(d["rgb"]).save(out / "rgb.png")


def overview(d: dict, path=None):
    """One figure with every M1 output (the notebook shows it; the web app will stream parts of it)."""
    import matplotlib
    matplotlib.use("Agg") if path else None
    import matplotlib.pyplot as plt
    rgb = d["rgb"]
    tiles = [("photo crop", rgb)]
    tint = lambda m, c: (rgb * 0.45 + np.array(c) * 0.55 * m[..., None] + rgb * 0.55 * (1 - m[..., None])).astype(np.uint8)
    tiles += [("human (SAM)", tint(d["masks"]["human"], (255, 80, 80))),
              ("cloth (SAM)", tint(d["masks"]["cloth"], (60, 160, 255))),
              ("upper / lower", tint(d["masks"]["upper"], (60, 200, 90)) * (d["masks"]["lower"][..., None] == 0)
               + tint(d["masks"]["lower"], (240, 190, 40)) * (d["masks"]["lower"][..., None] > 0))]
    bodyF = raster.to_rgb(d["T_normal"][0])
    over = rgb.copy()
    over[d["T_mask"][0].numpy()] = (0.5 * rgb + 0.5 * bodyF)[d["T_mask"][0].numpy()].astype(np.uint8)
    tiles += [("body template on photo", over), ("body normals front", bodyF),
              ("body normals back", raster.to_rgb(d["T_normal"][1]))]
    if d.get("cloth_t") is not None and d["cloth_t"]["garments"]:
        Vs, Fs, o = [], [], 0
        for V, F in d["cloth_t"]["garments"].values():
            Vs.append(V); Fs.append(F + o); o += len(V)
        nm, mk = raster.normal_maps(torch.tensor(np.concatenate(Vs), dtype=torch.float32), np.concatenate(Fs),
                                    d["K"], rgb.shape[0])
        over = rgb.copy()
        m0 = mk[0].numpy()
        over[m0] = (0.4 * rgb + 0.6 * raster.to_rgb(nm[0]))[m0].astype(np.uint8)
        tiles += [("cloth template (ClothWild)", over)]
    if d["normals"] is not None:
        tiles += [("ECON normals front", raster.to_rgb(d["normals"][0])),
                  ("ECON normals back", raster.to_rgb(d["normals"][1]))]
    n = len(tiles)
    cols = 5 if n > 8 else 4
    rows = -(-n // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows))
    for ax in axs.flat:
        ax.axis("off")
    for ax, (title, img) in zip(axs.flat, tiles):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=110)
        plt.close(fig)
    return fig
