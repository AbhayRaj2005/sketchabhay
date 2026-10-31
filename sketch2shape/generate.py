"""Stage 3 - sketch-guided image generation: Stable Diffusion 1.5 + ControlNet (scribble).

The sketch is the *structural* condition, the optional prompt only steers appearance. Several
candidates are generated and the caller re-ranks them by sketch fidelity (see pipeline.py).
Models are downloaded from the Hugging Face Hub on first use (see docs/MODELS.md).
"""
from __future__ import annotations
import os
from PIL import Image

BASE_MODEL = os.environ.get("S2S_SD_MODEL", "stable-diffusion-v1-5/stable-diffusion-v1-5")
CONTROLNET = os.environ.get("S2S_CONTROLNET", "lllyasviel/sd-controlnet-scribble")
_PIPE = {}


def available() -> bool:
    try:
        import torch, diffusers  # noqa: F401
        return True
    except ImportError:
        return False


def _load(device: str):
    if device in _PIPE:
        return _PIPE[device]
    import torch
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
    dtype = torch.float16 if device == "cuda" else torch.float32
    cn = ControlNetModel.from_pretrained(CONTROLNET, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(BASE_MODEL, controlnet=cn, torch_dtype=dtype,
                                                             safety_checker=None)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if device == "cuda":
        try:
            props = torch.cuda.get_device_properties(0)
            if props.total_memory < 8 * 2**30:
                pipe.enable_model_cpu_offload()            # < 8 GB VRAM
            else:
                pipe.to("cuda")
        except Exception:
            pipe.to("cuda")
    else:
        pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    _PIPE[device] = pipe
    return pipe


def generate_candidates(scribble: Image.Image, prompt: str, negative: str, n: int = 3, steps: int = 20,
                        guidance: float = 7.5, control_scale: float = 1.0, seed: int = 0,
                        device: str = "cuda", size: int = 512) -> list[Image.Image]:
    import torch
    pipe = _load(device)
    gen = [torch.Generator(device="cpu").manual_seed(seed + i) for i in range(n)]
    res = pipe([prompt] * n, image=[scribble.resize((size, size))] * n, negative_prompt=[negative] * n,
               num_inference_steps=steps, guidance_scale=guidance, controlnet_conditioning_scale=control_scale,
               generator=gen)
    return list(res.images)
