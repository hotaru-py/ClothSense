"""Human parsing: which pixels are person, garment and bare skin (ported from the v1 code in bak/).

SegFormer-B2 fine-tuned for clothes parsing (Hugging Face: mattmdjaga/segformer_b2_clothes, ATR labels);
pip-installable via `transformers`, no registration. declo.masks uses the labels to prompt SAM.
"""
from __future__ import annotations

import numpy as np

LABELS = ["Background", "Hat", "Hair", "Sunglasses", "Upper-clothes", "Skirt", "Pants", "Dress",
          "Belt", "Left-shoe", "Right-shoe", "Face", "Left-leg", "Right-leg", "Left-arm",
          "Right-arm", "Bag", "Scarf"]

SKIN = [11, 12, 13, 14, 15]

UPPER_GARMENT = [4, 7, 17]

NOT_PERSON = [0, 16]

class HumanParser:
    def __init__(self, name="mattmdjaga/segformer_b2_clothes", device=None):
        import torch
        from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = SegformerImageProcessor.from_pretrained(name)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(name).to(self.device).eval()

    def __call__(self, pil_img) -> dict:
        torch = self.torch
        with torch.no_grad():
            inputs = self.proc(images=pil_img.convert("RGB"), return_tensors="pt").to(self.device)
            logits = self.model(**inputs).logits
            up = torch.nn.functional.interpolate(logits, size=pil_img.size[::-1], mode="bilinear",
                                                 align_corners=False)
            lab = up.argmax(1)[0].cpu().numpy().astype(np.uint8)
        return masks_from_labels(lab)

def masks_from_labels(lab: np.ndarray) -> dict:
    person = ~np.isin(lab, NOT_PERSON)
    return {"labels": lab, "person": person.astype(np.uint8),
            "skin": np.isin(lab, SKIN).astype(np.uint8),
            "upper_garment": np.isin(lab, UPPER_GARMENT).astype(np.uint8)}

def colorize(lab: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(3)
    pal = (rng.uniform(0.2, 1.0, (len(LABELS), 3)) * 255).astype(np.uint8)
    pal[0] = 0
    return pal[lab]
