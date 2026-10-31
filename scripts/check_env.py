"""Environment doctor: shows which pipeline stages / backends will run on THIS machine. Run before a demo."""
import importlib, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sketch2shape import hardware, generate, background
from sketch2shape.backends import trellis, instantmesh, dreamfusion

def has(m):
    try: importlib.import_module(m); return True
    except Exception: return False

hw = hardware.probe()
print("Hardware:", {k: hw[k] for k in ("platform", "cpu_count", "ram_gb", "gpu", "vram_gb", "torch", "device")})
if hw["device"] == "cuda":
    import torch
    tensor = torch.arange(16, device="cuda", dtype=torch.float32).reshape(4, 4)
    print("CUDA tensor test: PASS (matrix sum =", (tensor @ tensor).sum().item(), ")")
rows = [("Core (numpy/scipy/opencv/skimage/PIL)", all(has(m) for m in ("numpy", "scipy", "cv2", "skimage", "PIL")), "required"),
        ("Gradio UI", has("gradio"), "python app.py"),
        ("Diffusers + Torch (SD+ControlNet)", generate.available(), "pip install -r requirements-gpu.txt"),
        ("Accelerator for diffusion", hw["device"] in ("cuda", "mps"), "diffusion auto-disabled on CPU"),
        ("rembg background removal", background.rembg_available(), "falls back to flood-fill matting"),
        ("TRELLIS backend", trellis.available(), "S2S_TRELLIS_DIR + CUDA"),
        ("InstantMesh backend", instantmesh.available(), "S2S_INSTANTMESH_DIR + CUDA"),
        ("Stable-DreamFusion backend", dreamfusion.available(), "third_party/stable-dreamfusion + zero123 ckpt + CUDA"),
        ("trimesh (extra validation)", has("trimesh"), "optional"),
        ("Anthropic API captioning", bool(os.environ.get("ANTHROPIC_API_KEY")) and has("anthropic"), "optional")]
for name, ok, hint in rows:
    print(f"[{'OK' if ok else '--'}] {name:42s} {'' if ok else '-> ' + hint}")
auto = "trellis" if trellis.available() else "inflate (CPU)"
print(f"\n'auto' will use: image generator = {'SD+ControlNet' if generate.available() and hw['device'] in ('cuda','mps') else 'offline stylizer'}, 3D backend = {auto}")
