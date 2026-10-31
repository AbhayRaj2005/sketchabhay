# Models, services and third-party components (disclosure)

Pretrained weights are **downloaded at first use**; nothing is bundled. Verify each licence before redistribution.

| Component | Used for | Where | Licence (as published upstream - please re-check) | Required? |
|---|---|---|---|---|
| Stable Diffusion v1.5 (`stable-diffusion-v1-5/stable-diffusion-v1-5`) | sketch-guided image generation | Hugging Face Hub | CreativeML OpenRAIL-M | No (GPU path) |
| ControlNet scribble (`lllyasviel/sd-controlnet-scribble`) | structural conditioning from the sketch | Hugging Face Hub | CreativeML OpenRAIL-M | No (GPU path) |
| rembg + U²-Net | background removal | PyPI / github.com/danielgatis/rembg | MIT (code); U²-Net Apache-2.0 | No (flood-fill fallback) |
| Microsoft TRELLIS (`microsoft/TRELLIS-image-large`) | image → textured 3D | github.com/microsoft/TRELLIS | MIT | No (GPU path) |
| TencentARC InstantMesh | alternative image → 3D | github.com/TencentARC/InstantMesh | Apache-2.0 | No |
| Stability AI / Tripo AI TripoSR | pretrained single-image → 3D candidate | `stabilityai/TripoSR` on Hugging Face | MIT | No; 1.68 GB checkpoint, upstream README reports about 6 GB VRAM; not integrated or downloaded |
| Stable-DreamFusion + Zero-1-to-3 (`zero123-xl.ckpt`) | optional slow image → 3D | `third_party/` (code), Columbia download (weights) | Apache-2.0 (code) / MIT (Zero-1-to-3) | No |
| Anthropic API (`S2S_CLAUDE_MODEL`, default `claude-haiku-4-5`) | optional object naming when prompt is empty | external service, only if `ANTHROPIC_API_KEY` set **and** `--use-claude` / `use_claude=True` | commercial API | No - **off by default; sketch image is sent to the API when enabled** |
| Gradio, PyTorch, Diffusers, OpenCV, SciPy, scikit-image, NumPy, Pillow, trimesh | libraries | PyPI | Apache-2.0 / BSD / MIT / HPND | core ones yes |

**No external service is contacted in the default configuration** after dependencies are installed. The CPU path uses no
pretrained model at all.

On the inspected RTX 3050 Laptop GPU, available VRAM is 4 GB. The CUDA PyTorch tensor path works, but this is below the
published TripoSR requirement (~6 GB) and TRELLIS minimum (16 GB). TripoSR is listed as a candidate, not an available
backend. Its checkpoint has not been downloaded; a local installation needs explicit approval because it is a 1.68 GB
download and is still over the GPU's documented memory budget. The default `inflate` backend creates watertight
silhouette-driven 2.5D geometry and is not learned image-to-3D reconstruction.

## Ideas adapted from supplied reference repositories
* `aws-samples/sample-retail-sketch-to-3d` (MIT-0): pipeline staging (cleanup → image generation → matting → TRELLIS → GLB) and
  adaptive-threshold sketch cleaning. Code re-written; no AWS services used.
* `Sketch-to-3d-Model` (K. Hofer, master's thesis): flood-fill silhouette, Euler-number hole counting, silhouette IoU as a metric.
  The repository ships no licence, so **no code was copied**; the algorithms are textbook (connected components, Euler number).
* `ashawkey/stable-dreamfusion` (Apache-2.0): bundled unmodified under `third_party/` for the optional backend.
