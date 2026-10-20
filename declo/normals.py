"""DeClotH step (1): front and back normal maps with ECON's normal estimator.

ECON (Xiu et al., CVPR 2023) predicts the clothed person's front and back normal maps from the image
plus normal renders of the fitted SMPL-X body (T_normal_F / T_normal_B), with two pix2pixHD generators.
The weights come with ECON's data (econ_data.zip, ICON registration -> tools/fetch_mpi.py econ).

Frame: ECON's (x right, y up, z towards the camera), on the 512 px work crop; the back map is
mirrored so it lines up with the photo (the far surface along each pixel ray, declo.raster).
Newer ECON code can swap the front map for Sapiens-normal; DeClotH used ECON's own net, so do we.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ECON = ROOT / "work/ECON"
NORMAL_CKPT = ECON / "data/ckpt/normal.ckpt"


def _stub_lightning():
    """ECON's generators subclass pl.LightningModule but use nothing from it."""
    if "pytorch_lightning" not in sys.modules:
        pl = types.ModuleType("pytorch_lightning")
        pl.LightningModule = torch.nn.Module
        sys.modules["pytorch_lightning"] = pl


class NormalEstimator:
    def __init__(self, ckpt=NORMAL_CKPT, device=None):
        ckpt = Path(ckpt)
        if not ckpt.exists():
            raise FileNotFoundError(f"{ckpt} missing: register at https://icon.is.tue.mpg.de and run "
                                    f"tools/fetch_mpi.py econ")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _stub_lightning()
        sys.path.insert(0, str(ECON))
        try:
            from lib.net.FBNet import define_G
        finally:
            sys.path.remove(str(ECON))
        # in_nml = image + T_normal_F -> netF (6 ch), image + T_normal_B -> netB (6 ch); configs/econ.yaml
        self.netF = define_G(6, 3, 64, "global", 4, 9, 1, 3, "instance")
        self.netB = define_G(6, 3, 64, "global", 4, 9, 1, 3, "instance")
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        for name, net in (("netF.", self.netF), ("netB.", self.netB)):
            sub = {k.split(name, 1)[1]: v for k, v in sd.items() if name in k}
            missing, unexpected = net.load_state_dict(sub, strict=False)
            if missing:
                raise RuntimeError(f"normal.ckpt: {len(missing)} missing keys for {name} e.g. {missing[:3]}")
            net.to(self.device).eval()

    @torch.no_grad()
    def __call__(self, rgb512: np.ndarray, mask512: np.ndarray, T_normal: torch.Tensor):
        """rgb512 (512,512,3) uint8 work crop, mask512 human mask, T_normal (2,3,512,512) body normals.
        Returns (2,3,512,512) predicted front/back normals in [-1,1], 0 outside the mask."""
        img = torch.from_numpy(rgb512).float().permute(2, 0, 1) / 255.0
        img = ((img - 0.5) / 0.5) * torch.from_numpy(mask512 > 0).float()[None]
        img = img[None].to(self.device)
        T = T_normal.to(self.device).float()
        nF = self.netF(torch.cat([img, T[0:1]], 1))
        nB = self.netB(torch.cat([img, T[1:2]], 1))
        m = (img.abs().sum(1, keepdim=True) != 0).float()
        nF = torch.nn.functional.normalize(nF, dim=1) * m
        nB = torch.nn.functional.normalize(nB, dim=1) * m
        return torch.cat([nF, nB]).cpu()

    def to(self, device):
        """Park in system RAM between uses (1.5 GB; with SAM's activations everything did not fit in 8 GB)."""
        self.netF.to(device)
        self.netB.to(device)
        self.device = device
        return self

    def free(self):
        del self.netF, self.netB
        torch.cuda.empty_cache()


SAPIENS = ("facebook/sapiens-normal-0.3b-torchscript", "sapiens_0.3b_normal_render_people_epoch_66_torchscript.pt2")


class SapiensNormal:
    """Front normal map with Sapiens-normal (Khirodkar et al., ECCV 2024), as ECON's released inference does for
    its front normal (apps/infer.py, cfg.sapiens.use=True); no registration. Same preprocessing as ECON's
    apps/sapiens.py: resize to 1024x768 (H x W), mean/std (123.5,116.5,103.5)/(58.5,57.0,57.5). ECON feeds the
    result straight in as normal_F, so it is in ECON's frame. We use the 0.3B model (ECON's config uses 2B, or
    1B if out of memory; both too big beside our other models on 8 GB).
    The back normal still needs ECON's own net (ICON registration)."""

    def __init__(self, device=None):
        from huggingface_hub import hf_hub_download
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        path = hf_hub_download(*SAPIENS)
        self.model = torch.jit.load(path, map_location=self.device).eval()
        self.mean = torch.tensor([123.5, 116.5, 103.5], device=self.device).view(1, 3, 1, 1) / 255
        self.std = torch.tensor([58.5, 57.0, 57.5], device=self.device).view(1, 3, 1, 1) / 255

    @torch.no_grad()
    def __call__(self, rgb_sq: np.ndarray, mask512: np.ndarray) -> torch.Tensor:
        """rgb_sq: the square work crop at any resolution (person at 0.9 of the side). Returns (3,512,512) in
        ECON's frame, 0 outside the person mask. The 4:3 portrait centre of the square goes to the model, so the
        person is not squashed; the result is pasted back into the square."""
        S = rgb_sq.shape[0]
        w = int(round(S * 0.75))
        x0 = (S - w) // 2
        img = torch.from_numpy(np.ascontiguousarray(rgb_sq[:, x0:x0 + w])).to(self.device).permute(2, 0, 1)[None].float() / 255
        img = torch.nn.functional.interpolate(img, size=(1024, 768), mode="bilinear", align_corners=False)
        out = self.model((img - self.mean) / self.std)
        out = torch.nn.functional.interpolate(out.float(), size=(S, w), mode="bilinear", align_corners=False)
        full = torch.zeros(1, 3, S, S, device=self.device)
        full[..., x0:x0 + w] = out
        full = torch.nn.functional.interpolate(full, size=mask512.shape, mode="bilinear", align_corners=False)[0]
        full = torch.nn.functional.normalize(full, dim=0)
        return (full * torch.from_numpy(mask512 > 0).to(self.device)[None]).cpu()

    def free(self):
        self.model = None
        torch.cuda.empty_cache()
