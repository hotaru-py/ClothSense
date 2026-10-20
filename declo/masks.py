"""DeClotH step (1): human and cloth silhouettes.

The paper runs SAM with text prompts (t_cloth, t_human). SAM itself takes boxes and points, so a text
prompt needs a grounding model in front of it. We ground with SegFormer clothes parsing instead
(declo.parsing): each region's box plus a few interior points (positive) and points of
neighbouring regions (negative) go to SAM ViT-H, and SAM's mask that agrees best with the parsing wins.
SAM supplies the crisp boundary; the parsing supplies "which pixels are cloth".

Regions: human (everything on the person), upper (upper clothes, dress, scarf), lower (pants, skirt).
cloth = upper | lower, as in DeClotH's single cloth DMTet. Shoes, hats and bags stay with the human.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .parsing import HumanParser

ROOT = Path(__file__).resolve().parents[1]
SAM_CKPT = ROOT / "models/sam/sam_vit_h_4b8939.pth"

REGIONS = {                       # SegFormer (ATR) label ids
    "upper": [4, 17],             # upper clothes, scarf
    "lower": [5, 6],              # skirt, pants
}
DRESS = 7                         # split at the hip line: SegFormer calls shirt + shorts a "dress" too
NOT_HUMAN = [0, 16]               # background, bag


def _points(mask: np.ndarray, k: int, rng) -> np.ndarray:
    """k well-spread interior points: distance-transform weighted, then farthest-point sampling."""
    if mask.sum() == 0:
        return np.zeros((0, 2))
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    ys, xs = np.nonzero(dt > 0.5 * dt.max())
    if len(xs) == 0:
        ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], 1).astype(np.float64)
    out = [pts[np.argmax(dt[ys, xs])]]
    d = np.linalg.norm(pts - out[0], axis=1)
    for _ in range(k - 1):
        out.append(pts[np.argmax(d)])
        d = np.minimum(d, np.linalg.norm(pts - out[-1], axis=1))
    return np.array(out)


def _box(mask: np.ndarray, pad=0.02):
    ys, xs = np.nonzero(mask)
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    p = pad * max(x2 - x1, y2 - y1)
    H, W = mask.shape
    return np.array([max(x1 - p, 0), max(y1 - p, 0), min(x2 + p, W - 1), min(y2 + p, H - 1)])


def iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    return float((a & b).sum() / max((a | b).sum(), 1))


class Segmenter:
    def __init__(self, sam_ckpt=SAM_CKPT, model_type="vit_h", device=None, parser=None):
        from segment_anything import SamPredictor, sam_model_registry
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.parser = parser or HumanParser(device=self.device)
        sam = sam_model_registry[model_type](checkpoint=str(sam_ckpt)).to(self.device).eval()
        if self.device == "cuda":
            sam.image_encoder.half()                 # 1.3 GB less; masks unchanged (IoU vs fp32 checked)
        self.sam = SamPredictor(sam)

    def parse(self, rgb: np.ndarray) -> np.ndarray:
        from PIL import Image
        return self.parser(Image.fromarray(rgb))["labels"]

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray, labels: np.ndarray | None = None, hip_row=None, n_pos=4, min_px=400,
                 seed=0) -> dict:
        """rgb (H,W,3) uint8 (already the work crop, ideally ~1024 px). hip_row: image row of the body's hip
        line, where "dress" pixels are split into upper / lower. Returns uint8 masks + parsing."""
        rng = np.random.default_rng(seed)
        lab = self.parse(rgb) if labels is None else labels
        human_p = ~np.isin(lab, NOT_HUMAN)
        parsed = {"human": human_p}
        for k, ids in REGIONS.items():
            parsed[k] = np.isin(lab, ids)
        dress = lab == DRESS
        if dress.any():
            below = np.arange(lab.shape[0])[:, None] >= (hip_row if hip_row is not None else lab.shape[0])
            parsed["upper"] |= dress & ~below
            parsed["lower"] |= dress & below
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
            self.sam.set_image(rgb)
        out, scores = {}, {}
        for name, m in parsed.items():
            if m.sum() < min_px:
                out[name] = np.zeros(m.shape, np.uint8)
                scores[name] = None
                continue
            pos = _points(m, n_pos, rng)
            if name == "human":                        # negatives: background near the box corners
                bg = ~human_p
                neg = _points(cv2.erode(bg.astype(np.uint8), np.ones((15, 15))) > 0, 4, rng) if bg.any() else np.zeros((0, 2))
            else:                                      # negatives: the person's non-cloth parts
                other = human_p & ~m & ~np.isin(lab, sum(REGIONS.values(), [DRESS]))
                neg = _points(other, 3, rng) if other.sum() > min_px else np.zeros((0, 2))
            pts = np.concatenate([pos, neg])
            lbl = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
                masks, q, _ = self.sam.predict(point_coords=pts, point_labels=lbl, box=_box(m), multimask_output=True)
            agree = [iou(mm, m) for mm in masks]
            j = int(np.argmax(agree))
            out[name] = masks[j].astype(np.uint8)
            scores[name] = {"iou_vs_parsing": round(agree[j], 3), "sam_score": round(float(q[j]), 3)}
        out["upper"] &= out["human"]
        out["lower"] &= out["human"]                   # upper/lower may overlap (tucked hem); cloth is the union
        out["cloth"] = ((out["upper"] | out["lower"]) & out["human"]).astype(np.uint8)
        out["labels"] = lab
        out["scores"] = scores
        return out

    def free(self):
        del self.sam
        torch.cuda.empty_cache()
