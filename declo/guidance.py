"""DeClotH steps (5)-(6) wired into the geometry stage: cloth SDS and human SDS on rendered normal maps.

  L_cloth-SDS^norm = SDS(N_cloth^k,      S_cloth+skel^k, P_cloth)       ours: seg ControlNet on S_cloth^k only
  L_human-SDS^norm = SDS(N_body^k,       S_skel^k,       P_body)        ControlNet OpenPose
                   + SDS(N_cloth+body^k, S_skel^k,       P_cloth+body)
(paper Eq. 5-6; k = a random orbit camera per step; DreamFusion-style view suffix on every prompt).
The cloth stand-in drops the skeleton: with it, SD 1.5 drew a person in 3-4 of 6 views; with the seg
silhouette alone, garments only in 5 of 6 (work/declo/skeleton/cloth_conditions.png).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from . import render, skeleton
from .cameras import to_cam_frame
from .captions import prompts as make_prompts, view_suffix
from .sds import CN_OPENPOSE, SDS

CN_SEG = "lllyasviel/control_v11p_sd15_seg"
PAL = json.load(open(Path(__file__).with_name("ade20k_palette.json")))
WALL, APPAREL = PAL[0], PAL[92]


class Guidance:
    def __init__(self, rec, prep, smplx_model, encoder="vae", guidance=50.0, res=512, w_cloth=1e-5, w_human=5e-6,
                 t_range=(0.02, 0.98)):
        """w_*: SDS gradients on the DMTet parameters are ~1e4-1e5x the silhouette/template ones (measured at step 0,
        work/declo/test_gradnorms.py); 1e-5 / 5e-6 bring them to the same scale. 1e-3 tore the cloth apart.
        Provisional until a full run confirms them."""
        self.rec, self.res = rec, res
        self.sds = SDS(controlnets=(CN_OPENPOSE, CN_SEG), encoder=encoder, t_range=t_range)
        self.prompts = make_prompts(prep["captions"])
        for p, n in self.prompts.values():                 # embed every prompt x view once, then drop CLIP
            for v in ("front view", "side view", "back view"):
                self.sds.embed(f"{p}, {v}")
            self.sds.embed(n)
        self.sds.free_text_encoder()
        J = skeleton.posed_joints(smplx_model, prep["body"])
        self.J, self.kp = J, skeleton.coco18(J)
        rec.orbit.head = J[15]                              # paper: zoomed-in views around the head keypoint
        self.cloth_t = rec.T_posed["cloth"]                 # posed cloth template (S_cloth^k comes from it)
        self.guidance, self.w = guidance, {"cloth": w_cloth, "human": w_human}
        self.rng = np.random.default_rng(1)

    def _img(self, V, F, K):
        """Normal map as an image in [0,1], white background, (1,3,res,res)."""
        o = render.render(V, F, K, self.res)
        n = (o["normal"] + 1) * 0.5
        m = o["mask"][..., None]
        return (n * m + (1 - m)).permute(2, 0, 1)[None]

    def cond_images(self, K, R, t):
        sk = skeleton.draw(self.kp, K, R, t, res=self.res, face_visible=skeleton.face_visible(self.J, R, t))
        Vc, Fc = self.cloth_t
        with torch.no_grad():
            m = render.render(to_cam_frame(Vc, R, t), Fc, torch.tensor(K, dtype=torch.float32, device=Vc.device),
                              self.res, aa=False)["mask"].cpu().numpy() > 0.5
        seg = np.empty((self.res, self.res, 3), np.uint8)
        seg[:] = WALL
        seg[m] = APPAREL
        to_t = lambda a: torch.tensor(a, dtype=torch.float32, device=self.rec.dev).permute(2, 0, 1)[None] / 255.0
        return to_t(sk), to_t(seg), sk, seg

    def __call__(self, rec, meshes):
        """meshes: posed cloth / body as detached leaf tensors (Reconstruction.step). Each SDS term is
        back-propagated right away into those leaves, so only one render + encoder graph is alive at a time
        (all three at once overflowed 8 GB). Returns detached values for logging."""
        (az, el), (K, R, t) = rec.orbit.sample(self.rng)
        s = self.res / rec.res
        K = np.diag([s, s, 1.0]) @ K                         # orbit intrinsics are for rec.res
        Kt = torch.tensor(K, dtype=torch.float32, device=rec.dev)
        pose, seg, _, _ = self.cond_images(K, R, t)
        Vc, Fc = meshes["cloth"]
        Vb, Fb = meshes["body"]
        Vc, Vb = to_cam_frame(Vc, R, t), to_cam_frame(Vb, R, t)
        view = view_suffix(az)
        P = {k: (f"{p}, {view}", n) for k, (p, n) in self.prompts.items()}
        g = self.guidance

        def term(w, img_fn, prompt, cond):
            loss = self.sds(img_fn(), *prompt, cond, guidance=g)
            if loss.requires_grad:                                 # nothing of this mesh in view: no gradient
                (w * loss).backward(retain_graph=True)             # frees this branch once `loss` is dropped
            return loss.detach()

        l_cloth = term(self.w["cloth"], lambda: self._img(Vc, Fc, Kt), P["cloth"], {CN_SEG: seg})
        l_body = term(self.w["human"], lambda: self._img(Vb, Fb, Kt), P["body"], {CN_OPENPOSE: pose})
        l_all = term(self.w["human"], lambda: self._img(torch.cat([Vc, Vb]), torch.cat([Fc, Fb + len(Vc)]), Kt),
                     P["human"], {CN_OPENPOSE: pose})
        return {"sds_cloth": (self.w["cloth"], l_cloth), "sds_human": (self.w["human"], l_body + l_all)}
