"""GPU backend: Microsoft TRELLIS image -> textured GLB (same family as the AWS sample's TRELLIS.2).

Setup: clone https://github.com/microsoft/TRELLIS, follow its install, then
    export S2S_TRELLIS_DIR=/path/to/TRELLIS
Weights (microsoft/TRELLIS-image-large) download on first use. Needs a CUDA GPU with >= 16 GB VRAM.
Not exercised in the authoring sandbox (no GPU) - see README 'Validation status'.
"""
from __future__ import annotations
import os, sys
from PIL import Image
from ..meshio import read_glb

_PIPE = None


def available() -> bool:
    d = os.environ.get("S2S_TRELLIS_DIR")
    if not d or not os.path.isdir(d):
        return False
    try:
        import torch
        return (torch.cuda.is_available()
                and torch.cuda.get_device_properties(0).total_memory >= 16 * 2**30)
    except ImportError:
        return False


def reconstruct(image: Image.Image, out_glb: str, seed: int = 1, simplify: float = 0.95, texture_size: int = 1024):
    global _PIPE
    d = os.environ["S2S_TRELLIS_DIR"]
    if d not in sys.path:
        sys.path.insert(0, d)
    os.environ.setdefault("SPCONV_ALGO", "native")
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import postprocessing_utils
    if _PIPE is None:
        _PIPE = TrellisImageTo3DPipeline.from_pretrained(os.environ.get("S2S_TRELLIS_MODEL", "microsoft/TRELLIS-image-large"))
        _PIPE.cuda()
    out = _PIPE.run(image.convert("RGB"), seed=seed,
                    sparse_structure_sampler_params={"steps": 12, "cfg_strength": 7.5},
                    slat_sampler_params={"steps": 12, "cfg_strength": 3})
    glb = postprocessing_utils.to_glb(out["gaussian"][0], out["mesh"][0], simplify=simplify, texture_size=texture_size)
    os.makedirs(os.path.dirname(os.path.abspath(out_glb)), exist_ok=True)
    glb.export(out_glb)
    return read_glb(out_glb), out_glb
