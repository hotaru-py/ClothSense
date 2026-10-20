"""DeClotH steps (5) and (6): score distillation sampling (DreamFusion, Poole et al. 2023) with ControlNet.

grad_x L_SDS = w(t) (eps_hat(x_t; c, P, t) - eps) dx/dphi   (paper Eq. 3)

* Human SDS: SD 1.5 + ControlNet OpenPose on the skeleton rendered from the body template
  (the paper's HumanDiffusion is exactly "pre-trained ControlNet weights").
* Cloth SDS: the paper's ClothDiffusion (ControlNet fine-tuned on cloth images, conditioned on cloth
  silhouette + skeleton) is not released. Stand-in: the same SD 1.5 + ControlNet stack with garment-only
  prompts and person/skin negatives (more ControlNets can be stacked, e.g. a silhouette one).

One UNet (fp16) is shared by every ControlNet; the text encoder runs once per prompt and is then freed.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

SD = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CN_OPENPOSE = "lllyasviel/control_v11p_sd15_openpose"


class SDS:
    def __init__(self, controlnets=(CN_OPENPOSE,), device="cuda", dtype=torch.float16, t_range=(0.02, 0.98),
                 encoder="vae"):
        """encoder="vae": SD's VAE (197 ms encode+backward at 512 px on the 3070 Ti, offline quality);
        "taesd": tiny autoencoder (30 ms, latents corr ~0.85 with the VAE's) for the live preview."""
        from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer
        self.device, self.dtype = device, dtype
        kw = dict(torch_dtype=dtype, variant="fp16")
        if encoder == "taesd":
            from diffusers import AutoencoderTiny
            self.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=dtype).to(device).eval()
        else:
            self.vae = AutoencoderKL.from_pretrained(SD, subfolder="vae", **kw).to(device).eval()
        self.tiny = encoder == "taesd"
        self.unet = UNet2DConditionModel.from_pretrained(SD, subfolder="unet", **kw).to(device).eval()
        self.cn = {name: ControlNetModel.from_pretrained(name, **kw).to(device).eval() for name in controlnets}
        for m in [self.vae, self.unet, *self.cn.values()]:
            m.requires_grad_(False)
        self.tok = CLIPTokenizer.from_pretrained(SD, subfolder="tokenizer")
        self.te = CLIPTextModel.from_pretrained(SD, subfolder="text_encoder", **kw).to(device).eval()
        self.sched = DDPMScheduler.from_pretrained(SD, subfolder="scheduler")
        self.alphas = self.sched.alphas_cumprod.to(device)
        self.t_min, self.t_max = (int(r * self.sched.config.num_train_timesteps) for r in t_range)
        self._emb = {}

    @torch.no_grad()
    def embed(self, prompt: str):
        if prompt not in self._emb:
            ids = self.tok(prompt, padding="max_length", max_length=self.tok.model_max_length, truncation=True,
                           return_tensors="pt").input_ids.to(self.device)
            self._emb[prompt] = self.te(ids)[0]
        return self._emb[prompt]

    def free_text_encoder(self):
        self.te = None
        torch.cuda.empty_cache()

    def encode(self, img):
        """img (B,3,512,512) in [0,1] -> latents (B,4,64,64), differentiable."""
        x = img.to(self.dtype) * 2 - 1
        if self.tiny:
            return self.vae.encode(x).latents
        return self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor

    def __call__(self, img, prompt, negative, cond: dict, guidance=50.0, cond_scale=1.0, t=None):
        """img: render (B,3,512,512) in [0,1] with grad. cond: {controlnet name: (B,3,512,512) in [0,1]}.
        Returns a loss whose gradient w.r.t. img is the SDS gradient."""
        B = img.shape[0]
        lat = self.encode(img)
        if t is None:
            t = torch.randint(self.t_min, self.t_max + 1, (B,), device=self.device)
        emb = torch.cat([self.embed(negative).expand(B, -1, -1), self.embed(prompt).expand(B, -1, -1)])
        with torch.no_grad():
            noise = torch.randn_like(lat)
            x_t = self.sched.add_noise(lat, noise, t)
            x2, t2 = torch.cat([x_t] * 2), torch.cat([t] * 2)
            down, mid = None, None
            for name, image in cond.items():
                d, m = self.cn[name](x2, t2, encoder_hidden_states=emb,
                                     controlnet_cond=torch.cat([image] * 2).to(self.dtype),
                                     conditioning_scale=cond_scale, return_dict=False)
                down = d if down is None else [a + b for a, b in zip(down, d)]
                mid = m if mid is None else mid + m
            eps = self.unet(x2, t2, encoder_hidden_states=emb, down_block_additional_residuals=down,
                            mid_block_additional_residual=mid).sample
            eu, ec = eps.chunk(2)
            eps_hat = eu + guidance * (ec - eu)
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
            grad = torch.nan_to_num(w * (eps_hat - noise))
        target = (lat - grad).detach()
        return 0.5 * F.mse_loss(lat.float(), target.float(), reduction="sum") / B
