"""Run with `python -m pytest tests -q` or simply `python tests/test_pipeline.py` (no pytest needed)."""
import os, sys, tempfile
import cv2, numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sketch2shape import preprocess, silhouette, validate
from sketch2shape.backends import inflate
from sketch2shape.mesh import Mesh
from sketch2shape.meshio import read_glb, write_glb, write_obj, read_obj
from sketch2shape.pipeline import Options, run


def _ring_img():
    im = np.full((400, 400), 255, np.uint8)
    cv2.circle(im, (200, 200), 150, 0, 4); cv2.circle(im, (200, 200), 60, 0, 4)
    return Image.fromarray(im)


def test_preprocess_polarity_and_blank():
    inv = Image.fromarray(255 - np.array(_ring_img()))
    a, b = preprocess.clean_sketch(_ring_img()), preprocess.clean_sketch(inv)
    assert abs(a.stroke_mask.mean() - b.stroke_mask.mean()) < 0.01
    try:
        preprocess.clean_sketch(Image.new("RGB", (100, 100), "white")); assert False
    except ValueError:
        pass


def test_topology_nested_vs_fill():
    s = preprocess.clean_sketch(_ring_img()).stroke_mask
    assert silhouette.topology(silhouette.silhouette_from_strokes(s, "fill"))["holes"] == 0
    assert silhouette.topology(silhouette.silhouette_from_strokes(s, "nested"))["holes"] == 1


def test_inflate_is_closed_3d_solid():
    m = np.zeros((300, 400), np.uint8); cv2.ellipse(m, (200, 150), (150, 90), 0, 0, 360, 1, -1)
    mesh = inflate.reconstruct(m > 0, np.full((300, 400, 3), 128, np.uint8))
    v = validate.validate_mesh(mesh)
    assert v["ok"] and v["watertight"] and v["euler_characteristic"] == 2 and v["depth_ratio"] > 0.3
    assert validate.silhouette_iou(mesh, m > 0)["silhouette_iou"] > 0.9


def test_glb_obj_roundtrip():
    m = np.zeros((200, 200), np.uint8); cv2.circle(m, (100, 100), 70, 1, -1)
    mesh = inflate.reconstruct(m > 0, np.full((200, 200, 3), (200, 50, 50), np.uint8), resolution=96)
    with tempfile.TemporaryDirectory() as d:
        g = write_glb(mesh, os.path.join(d, "a.glb")); o = write_obj(mesh, os.path.join(d, "a.obj"))
        r, r2 = read_glb(g), read_obj(o)
        assert len(r.vertices) == len(mesh.vertices) and r.texture is not None and r.uv is not None
        assert len(r2.faces) == len(mesh.faces) and r2.texture is not None
        assert validate.validate_mesh(g)["opens_with_reader"]


def test_validator_rejects_flat_and_empty():
    flat = Mesh(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32), np.array([[0, 1, 2]], np.int32))
    assert not validate.validate_mesh(flat)["ok"]
    empty = Mesh(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32))
    assert not validate.validate_mesh(empty)["ok"]


def test_end_to_end_cpu():
    with tempfile.TemporaryDirectory() as d:
        r = run(_ring_img(), Options(prompt="blue ring", holes="nested", out_dir=d, run_name="t", generator="none", backend="inflate"))
        assert os.path.getsize(r["glb"]) > 1000 and os.path.exists(r["obj"]) and os.path.exists(r["report_md"])
        assert r["validation"]["ok"] and r["analysis"]["holes"] == 1 and r["validation"]["euler_characteristic"] == 0


def test_five_sample_categories_generate_geometry():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    names = ("bottle", "donut", "lamp", "rocket", "tennis_racket")
    with tempfile.TemporaryDirectory() as d:
        for name in names:
            image = Image.open(os.path.join(root, "samples", "inputs", f"{name}.png"))
            result = run(image, Options(prompt="", backend="inflate", generator="none", resolution=96,
                                        out_dir=d, run_name=name))
            assert result["validation"]["ok"], name
            assert result["validation"]["vertices"] > 0 and result["validation"]["faces"] > 0, name
            assert os.path.getsize(result["glb"]) > 1000 and os.path.getsize(result["obj"]) > 1000, name


def test_gradio_callback_returns_preview_and_downloads():
    import app
    previous_out = app.OUT
    try:
        with tempfile.TemporaryDirectory() as d:
            app.OUT = d
            result = app.generate(None, _ring_img(), "", "inflate", "none", "nested", "Fast",
                                  progress=lambda *args, **kwargs: None)
            model, clean, preview, files, summary = result
            assert os.path.isfile(model) and clean.size == (512, 512) and preview.size == (512, 512)
            assert len(files) == 4 and all(os.path.isfile(path) for path in files)
            assert "Backend:" in summary and "inflate" in summary
            clean.close()
            preview.close()
    finally:
        app.OUT = previous_out


def test_missing_backend_falls_back():
    with tempfile.TemporaryDirectory() as d:
        r = run(_ring_img(), Options(out_dir=d, run_name="t", generator="none", backend="trellis"))
        assert r["backend_used"] == "inflate" and any("trellis" in n.lower() for n in r["notes"])


def test_diffusion_auto_disabled_below_vram_requirement():
    from sketch2shape.pipeline import _can_run_diffusion
    assert not _can_run_diffusion({"device": "cuda", "vram_gb": 4.0})
    assert _can_run_diffusion({"device": "cuda", "vram_gb": 8.0})
    assert not _can_run_diffusion({"device": "cuda", "vram_gb": None})


def _fake_generation(monkeypatch_targets, candidates):
    """Pretend a GPU + diffusers exist and return `candidates` instead of running Stable Diffusion."""
    from sketch2shape import generate, hardware
    saved = (generate.available, generate.generate_candidates, hardware.probe)
    generate.available = lambda: True
    generate.generate_candidates = lambda *a, **k: candidates
    real = hardware.probe
    hardware.probe = lambda: {**real(), "device": "cuda", "gpu": "MOCK", "vram_gb": 8.0}
    return saved


def _restore(saved):
    from sketch2shape import generate, hardware
    generate.available, generate.generate_candidates, hardware.probe = saved


def _product_img(kind):
    im = np.full((512, 512, 3), 255, np.uint8)
    if kind == "ring":
        cv2.circle(im, (256, 256), 190, (200, 60, 60), -1); cv2.circle(im, (256, 256), 76, (255, 255, 255), -1)
    else:
        cv2.rectangle(im, (60, 200), (450, 300), (60, 60, 200), -1)            # unrelated object
    return Image.fromarray(im)


def test_generation_branch_ranks_by_sketch_fidelity():
    saved = _fake_generation(None, [_product_img("bar"), _product_img("ring"), _product_img("bar")])
    try:
        with tempfile.TemporaryDirectory() as d:
            r = run(_ring_img(), Options(out_dir=d, run_name="g", holes="nested", backend="inflate"))
        assert r["generator_used"] == "sd_controlnet"
        best = max(r["candidates"], key=lambda c: c["sketch_iou"])
        assert best["index"] == 1 and best["sketch_iou"] > 0.8
        assert r["validation"]["ok"] and r["validation"]["euler_characteristic"] == 0   # hole came from the matte
    finally:
        _restore(saved)


def test_drift_guard_uses_sketch_silhouette():
    saved = _fake_generation(None, [_product_img("bar")])
    try:
        with tempfile.TemporaryDirectory() as d:
            r = run(_ring_img(), Options(out_dir=d, run_name="g", holes="nested", backend="inflate"))
        assert any("drifted" in n for n in r["notes"]) and r["fidelity"]["silhouette_iou"] > 0.85
    finally:
        _restore(saved)


def test_generation_failure_falls_back_to_stylizer():
    from sketch2shape import generate
    saved = _fake_generation(None, [])
    def boom(*a, **k): raise RuntimeError("CUDA out of memory")
    generate.generate_candidates = boom
    try:
        with tempfile.TemporaryDirectory() as d:
            r = run(_ring_img(), Options(out_dir=d, run_name="g", backend="inflate"))
        assert r["generator_used"] == "offline-stylizer" and r["validation"]["ok"]
    finally:
        _restore(saved)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f(); print("PASS", f.__name__)
    print(f"{len(fns)} tests passed")
