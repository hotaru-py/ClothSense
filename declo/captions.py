"""Text prompts for the SDS losses (DeClotH: BLIP captions describe cloth colour, shape and style).

P_cloth : BLIP on the person, prompted "a photo of a person wearing", + BLIP on the upper garment alone for detail.
P_human : "a person wearing <the same garments>".
P_body  : the unclothed body for human SDS on M_body (written, not captioned: BLIP cannot see under clothes).
View-dependent suffixes (DreamFusion) are added per camera in the SDS loop.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

BLIP = "Salesforce/blip-image-captioning-large"


def _clean(c: str) -> str:
    import re
    c = re.sub(r" (with|on|in front of|against) (a )?(plain )?white background", "", c)
    for junk in (" on a mannequin dummy", " on a mannequin", " is shown", " are shown"):
        c = c.replace(junk, "")
    return c.replace(" - ", "-").replace(" ' s", "'s").removeprefix("a photo of ").strip()


class Captioner:
    def __init__(self, device=None):
        from transformers import BlipForConditionalGeneration, BlipProcessor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = BlipProcessor.from_pretrained(BLIP)
        self.model = BlipForConditionalGeneration.from_pretrained(BLIP, torch_dtype=torch.float16).to(self.device).eval()

    @torch.no_grad()
    def caption(self, rgb: np.ndarray, mask: np.ndarray, prefix="a photo of") -> str:
        img = np.where(mask[..., None] > 0, rgb, 255).astype(np.uint8)
        ys, xs = np.nonzero(mask)
        pad = 8
        img = img[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]
        inp = self.proc(Image.fromarray(img), prefix, return_tensors="pt").to(self.device, torch.float16)
        out = self.model.generate(**inp, max_new_tokens=40, num_beams=3, repetition_penalty=1.5)
        return self.proc.decode(out[0], skip_special_tokens=True).strip()

    def __call__(self, rgb, masks) -> dict:
        """BLIP sees the garments best on the whole person ("a photo of a person wearing ..."): cut-out shorts
        alone become "scissors" or "toilet paper". The upper garment's own caption adds detail (collar, zip)."""
        wearing = _clean(self.caption(rgb, masks["human"], "a photo of a person wearing")).removeprefix("a person wearing ").strip()
        cloth = wearing
        if masks["upper"].sum() > 200 and " and " in wearing:
            upper = _clean(self.caption(rgb, masks["upper"]))
            cloth = f"{upper} and {wearing.split(' and ', 1)[1]}"
        return {"cloth": cloth, "wearing": wearing, "human": f"a person wearing {wearing}"}

    def free(self):
        self.model = None
        torch.cuda.empty_cache()


def prompts(caps: dict) -> dict:
    """The SDS prompt set. Cloth: garment-only wording + negatives (ClothDiffusion stand-in)."""
    cloth = caps["cloth"]
    return {
        "cloth": (f"ghost mannequin product photo of {cloth}, clothes only, plain white background",
                  "person, human, man, woman, skin, face, head, neck, arms, hands, legs, feet, body, mannequin, hanger, "
                  "oversaturated, neon colors"),
        "human": (f"a photo of {caps['human']}, full body, high quality",
                  "blurry, low quality, deformed, bad anatomy, oversaturated, neon colors"),
        "body": ("a photo of a person in tight grey underwear, full body, high quality",
                 "clothes, shirt, pants, dress, skirt, jacket, blurry, low quality, deformed, oversaturated, neon colors"),
    }


VIEWS = ((45, "front view"), (135, "side view"), (225, "back view"), (315, "side view"))


def view_suffix(azim_deg: float) -> str:
    a = azim_deg % 360
    if a < 45 or a >= 315:
        return "front view"
    if 135 <= a < 225:
        return "back view"
    return "side view"
