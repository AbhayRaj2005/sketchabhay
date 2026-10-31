"""Optional slow, high-fidelity backend: Stable-DreamFusion (Zero-1-to-3 SDS), bundled in third_party/.

Pipeline (from the bundled repo's README): preprocess RGBA -> `main.py --image ... --iters N` -> `--test --save_mesh`.
Needs a CUDA GPU, the compiled extensions (`scripts/install_ext.sh`) and `pretrained/zero123/zero123-xl.ckpt`.
Expect 10-40 minutes - use it for the final hero asset, not for live demos.
"""
from __future__ import annotations
import glob, os, subprocess, sys
from PIL import Image

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.environ.get("S2S_DREAMFUSION_DIR", os.path.join(HERE, "third_party", "stable-dreamfusion"))


def available() -> bool:
    ck = os.path.join(REPO, "pretrained", "zero123", "zero123-xl.ckpt")
    try:
        import torch
        return os.path.isfile(os.path.join(REPO, "main.py")) and os.path.isfile(ck) and torch.cuda.is_available()
    except ImportError:
        return False


def reconstruct(rgba: Image.Image, workdir: str, iters: int = 3000, timeout: int = 3 * 3600) -> str:
    os.makedirs(workdir, exist_ok=True)
    rgba_path = os.path.abspath(os.path.join(workdir, "input_rgba.png")); rgba.save(rgba_path)
    ws = os.path.abspath(os.path.join(workdir, "trial"))
    base = [sys.executable, "main.py", "-O", "--image", rgba_path, "--workspace", ws, "--iters", str(iters)]
    for cmd in (base, base + ["--test", "--save_mesh"]):
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            raise RuntimeError(f"Stable-DreamFusion failed: {r.stderr[-400:]}")
    objs = glob.glob(os.path.join(ws, "mesh", "*.obj"))
    if not objs:
        raise RuntimeError("Stable-DreamFusion produced no mesh/*.obj")
    return objs[0]
