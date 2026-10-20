"""One-time download of everything the pipeline needs that does not need a registration.

    .venv\\Scripts\\python tools\\setup_models.py

* Hugging Face models into the HF cache (the pipeline then runs offline, HF_HUB_OFFLINE=1): Stable Diffusion 1.5
  (fp16 UNet / VAE / text encoder), ControlNet openpose + seg, TAESD, BLIP-large, SegFormer clothes parser,
  Sapiens-normal 0.3B (front-normal fallback).
* SAM ViT-H checkpoint -> models/sam/sam_vit_h_4b8939.pth
* git clones: naver/multi-hmr -> work/multi-hmr (its checkpoint downloads itself on first use),
  YuliangXiu/ECON -> work/ECON, and ECON's normal estimator (as loaded by ECON's own Hugging Face demo, from
  Yuliang/ICON) -> work/ECON/data/ckpt/normal.ckpt
Not here (registration): SMPL-X -> models/smplx/SMPLX_NEUTRAL.npz. Optional ClothWild template: see README.
About 12 GB in total. Re-running skips what is already there.
"""
import subprocess
import sys
from pathlib import Path

import requests
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
SD = "stable-diffusion-v1-5/stable-diffusion-v1-5"
HF = [
    (SD, ["model_index.json", "scheduler/*", "tokenizer/*", "unet/config.json", "unet/*.fp16.safetensors",
          "vae/config.json", "vae/*.fp16.safetensors", "text_encoder/config.json", "text_encoder/*.fp16.safetensors"]),
    ("lllyasviel/control_v11p_sd15_openpose", ["config.json", "*.fp16.safetensors"]),
    ("lllyasviel/control_v11p_sd15_seg", ["config.json", "*.fp16.safetensors"]),
    ("madebyollin/taesd", ["config.json", "*.safetensors"]),
    ("Salesforce/blip-image-captioning-large", None),
    ("mattmdjaga/segformer_b2_clothes", None),
]
SAPIENS = ("facebook/sapiens-normal-0.3b-torchscript", "sapiens_0.3b_normal_render_people_epoch_66_torchscript.pt2")
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
REPOS = [("https://github.com/naver/multi-hmr", ROOT / "work/multi-hmr"),
         ("https://github.com/YuliangXiu/ECON", ROOT / "work/ECON")]


def weights_only(repo):
    """transformers repos: configs + one copy of the weights (safetensors if the repo has them, else .bin)."""
    files = HfApi().list_repo_files(repo)
    w = "*.safetensors" if any(f.endswith(".safetensors") for f in files) else "*.bin"
    return ["*.json", "*.txt", w]


def download(url, dst: Path):
    if dst.exists() and dst.stat().st_size > 0:
        print(f"  have {dst.relative_to(ROOT)}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total, done = int(r.headers.get("content-length", 0)), 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                done += len(chunk)
                print(f"\r  {dst.name}: {done / 1e6:8.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dst)


def main():
    print("Hugging Face models (cache):")
    for repo, allow in HF:
        snapshot_download(repo, allow_patterns=allow or weights_only(repo))
        print(f"  ok {repo}")
    hf_hub_download(*SAPIENS)
    print(f"  ok {SAPIENS[0]}")
    print("SAM ViT-H:")
    download(SAM_URL, ROOT / "models/sam/sam_vit_h_4b8939.pth")
    print("Repositories:")
    for url, dst in REPOS:
        if dst.exists():
            print(f"  have {dst.relative_to(ROOT)}")
        else:
            subprocess.run(["git", "clone", "--depth", "1", url, str(dst)], check=True)
    print("ECON normal estimator:")
    ck = ROOT / "work/ECON/data/ckpt/normal.ckpt"
    if ck.exists():
        print(f"  have {ck.relative_to(ROOT)}")
    else:
        ck.parent.mkdir(parents=True, exist_ok=True)
        src = hf_hub_download("Yuliang/ICON", "normal.ckpt")
        import shutil
        shutil.copy2(src, ck)
        print(f"  ok {ck.relative_to(ROOT)}")
    smplx = ROOT / "models/smplx/SMPLX_NEUTRAL.npz"
    print(f"SMPL-X: {'ok' if smplx.exists() else 'MISSING - register at https://smpl-x.is.tue.mpg.de and put SMPLX_NEUTRAL.npz in models/smplx/'}")


if __name__ == "__main__":
    sys.exit(main())
