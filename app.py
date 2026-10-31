"""Gradio web app:  python app.py   ->  http://127.0.0.1:7860

Draw on the canvas (brush / eraser / clear) OR upload a sketch image, optionally add a text prompt,
press Generate, inspect the model in the 3D viewer and download .glb / .obj / run report.
"""
from __future__ import annotations
import os, sys, zipfile
import gradio as gr
from PIL import Image
from sketch2shape import hardware
from sketch2shape.pipeline import Options, run
from sketch2shape.preprocess import to_pil

OUT = os.environ.get("S2S_OUT", "outputs")


def _canvas_has_ink(val) -> bool:
    try:
        import numpy as np
        im = to_pil(val); a = np.array(im.convert("RGBA"))
        return bool((a[..., 3] > 0).any() and (a[..., :3].min(-1) < 200).any())
    except Exception:
        return False


def generate(canvas, upload, prompt, backend, generator, holes, quality, progress=gr.Progress()):
    src = upload if upload is not None else (canvas if _canvas_has_ink(canvas) else None)
    if src is None:
        raise gr.Error("Draw a sketch on the canvas or upload a sketch image first.")
    try:
        r = run(src, Options(prompt=prompt or "", backend=backend, generator=generator, holes=holes,
                             resolution={"Fast": 160, "Balanced": 224, "Detailed": 320}[quality], out_dir=OUT),
                progress=lambda f, m: progress(f, desc=m))
    except gr.Error:
        raise
    except Exception as ex:
        raise gr.Error(f"Generation failed ({type(ex).__name__}): {ex}") from ex
    zp = os.path.join(r["run_dir"], "bundle.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in (r["glb"], r["obj"], r["obj"][:-4] + ".mtl", r["obj"][:-4] + "_texture.png", r["report_md"]):
            if os.path.exists(f): z.write(f, os.path.basename(f))
    v, fid = r["validation"], r["fidelity"]
    summary = (f"**Backend:** `{r['backend_used']}` | **Generator:** `{r['generator_used']}` | **Time:** {r['timings']['total']} s  \n"
               f"**Mesh:** {v['vertices']:,} verts / {v['faces']:,} faces | watertight: {v['watertight']} | textured: {v['has_texture']}  \n"
               f"**Sketch fidelity (silhouette IoU):** {fid['silhouette_iou']}  \n"
               + "".join(f"- {n}\n" for n in r["notes"]))
    return (r["glb"], Image.open(os.path.join(r["run_dir"], "01_clean_sketch.png")), Image.open(r["preview_image"]),
            [r["glb"], r["obj"], r["report_md"], zp], summary)


with gr.Blocks(title="Sketch2Shape AI") as demo:
    hw = hardware.probe()
    if hw["device"] == "cuda" and (hw.get("vram_gb") or 0) < 8:
        runtime = (f"CUDA detected ({hw['gpu']}, {hw['vram_gb']} GB VRAM), but this is below the local "
                   "learned-generation minimum. The default uses CPU silhouette inflation: a textured, closed 3D "
                   "mesh, not learned full-object reconstruction.")
    else:
        runtime = (f"Hardware: {hw.get('gpu') or 'CPU only'}. The default backend is CPU silhouette inflation; "
                   "learned models run only when separately installed and configured.")
    gr.Markdown("# Sketch2Shape AI - from a 2D sketch to a 3D asset\n"
                + runtime)
    with gr.Row():
        with gr.Column():
            with gr.Tab("Draw"):
                canvas = gr.Sketchpad(label="Canvas (brush / eraser / clear)", type="pil", image_mode="RGBA",
                                      canvas_size=(512, 512)) if hasattr(gr, "Sketchpad") else gr.ImageEditor(type="pil")
            with gr.Tab("Upload"):
                upload = gr.Image(label="Upload a sketch or object image (PNG / JPG / JPEG)", type="pil",
                                  sources=["upload"])
            prompt = gr.Textbox(label="Optional prompt", placeholder="e.g. red ceramic mug, glossy  (not required)")
            with gr.Accordion("Options", open=False):
                backend = gr.Dropdown(["auto", "inflate", "trellis", "instantmesh", "dreamfusion"], value="auto", label="3D backend")
                generator = gr.Dropdown(["auto", "sd_controlnet", "none"], value="auto", label="Sketch-guided image generator")
                holes = gr.Radio(["fill", "nested"], value="fill", label="Holes",
                                 info="'nested': an outline drawn inside another outline becomes a through-hole")
                quality = gr.Radio(["Fast", "Balanced", "Detailed"], value="Balanced", label="CPU mesh detail")
            btn = gr.Button("Generate 3D model", variant="primary")
        with gr.Column():
            model = gr.Model3D(label="3D preview (drag to rotate)", clear_color=(0.96, 0.96, 0.96, 1.0))
            with gr.Row():
                clean = gr.Image(label="Cleaned sketch", height=180)
                obj_img = gr.Image(label="Object used for 3D", height=180)
            files = gr.File(label="Downloads (.glb, .obj, report, bundle.zip)", file_count="multiple")
            info = gr.Markdown()
    btn.click(generate, [canvas, upload, prompt, backend, generator, holes, quality], [model, clean, obj_img, files, info])

if __name__ == "__main__":
    demo.queue().launch(server_name=os.environ.get("HOST", "127.0.0.1"), server_port=int(os.environ.get("PORT", 7860)))
