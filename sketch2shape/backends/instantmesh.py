"""GPU backend: TencentARC InstantMesh via its own CLI (multi-view diffusion + LRM, textured OBJ).

Setup: clone https://github.com/TencentARC/InstantMesh, install its requirements, then
    export S2S_INSTANTMESH_DIR=/path/to/InstantMesh
Best-effort wrapper - it shells out to `run.py ... --export_texmap` and picks up the produced OBJ.
"""
from __future__ import annotations
import glob, os, subprocess, sys, tempfile
from PIL import Image


def available() -> bool:
    d = os.environ.get("S2S_INSTANTMESH_DIR")
    if not d or not os.path.isfile(os.path.join(d, "run.py")):
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def reconstruct(image: Image.Image, out_dir: str, timeout: int = 1800) -> str:
    d = os.environ["S2S_INSTANTMESH_DIR"]
    os.makedirs(out_dir, exist_ok=True)
    src = os.path.join(tempfile.mkdtemp(), "input.png"); image.save(src)
    cmd = [sys.executable, "run.py", "configs/instant-mesh-large.yaml", src, "--output_path", out_dir, "--export_texmap"]
    r = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=timeout)
    objs = sorted(glob.glob(os.path.join(out_dir, "**", "*.obj"), recursive=True), key=os.path.getmtime)
    if r.returncode != 0 or not objs:
        raise RuntimeError(f"InstantMesh failed (rc={r.returncode}): {r.stderr[-400:]}")
    return objs[-1]
