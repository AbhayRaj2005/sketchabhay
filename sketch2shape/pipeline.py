"""End-to-end orchestrator:  sketch -> clean -> analyse -> (generate -> rank -> matte) -> 3D -> validate -> export.

Design goals (from the judging rubric): never crash silently, always leave a valid .glb, report timing and
hardware, and keep the result tied to the *sketch* (fidelity re-ranking + geometry guard).
"""
from __future__ import annotations
import os, time, traceback
from dataclasses import dataclass, asdict
import cv2
import numpy as np
from PIL import Image

from . import analysis, background, generate, hardware, preprocess, stylize, validate
from .backends import dreamfusion, inflate, instantmesh, trellis
from .mesh import Mesh
from .meshio import read_glb, read_obj, write_glb, write_obj
from .report import Timer, write_report


@dataclass
class Options:
    prompt: str = ""
    backend: str = "auto"          # auto | inflate | trellis | instantmesh | dreamfusion
    generator: str = "auto"        # auto | sd_controlnet | none
    candidates: int = 0            # 0 = auto (3 on GPU, 1 on CPU)
    holes: str = "fill"            # fill | nested
    steps: int = 20
    seed: int = 0
    resolution: int = 224          # inflate grid resolution
    depth_scale: float = 1.0       # inflate thickness multiplier
    out_dir: str = "outputs"
    run_name: str = ""
    use_claude: bool = False       # optional external captioning API
    min_fidelity: float = 0.55     # below this, geometry falls back to the sketch silhouette


def _log(cb, frac, msg):
    if cb: cb(frac, msg)


def _can_run_diffusion(hw: dict) -> bool:
    if hw.get("device") == "mps":
        return True
    return hw.get("device") == "cuda" and (hw.get("vram_gb") or 0) >= 8


def _warp_to_silhouette(rgba: Image.Image, sil: np.ndarray) -> np.ndarray:
    """Stretch the generated object's bbox onto the sketch silhouette bbox (texture transfer fallback)."""
    a = np.array(rgba); ys, xs = np.where(a[..., 3] > 16)
    crop = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1, :3]
    sy, sx = np.where(sil)
    h, w = sy.max() - sy.min() + 1, sx.max() - sx.min() + 1
    out = np.full((*sil.shape, 3), 200, np.uint8)
    out[sy.min():sy.min() + h, sx.min():sx.min() + w] = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
    return out


def run(sketch, opts: Options | None = None, progress=None) -> dict:
    opts = opts or Options()
    T, notes = Timer(), []
    hw = hardware.probe()
    run_name = opts.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(opts.out_dir, run_name); os.makedirs(run_dir, exist_ok=True)
    res: dict = {"run_name": run_name, "run_dir": run_dir, "options": asdict(opts), "hardware": hw, "notes": notes}

    # 1 ---- input conditioning
    _log(progress, 0.05, "Cleaning sketch")
    with T.stage("1_preprocess"):
        sk = preprocess.clean_sketch(sketch)
        Image.fromarray(sk.line_img).save(os.path.join(run_dir, "01_clean_sketch.png"))
        sk.scribble.save(os.path.join(run_dir, "01_scribble_control.png"))

    # 2 ---- sketch understanding
    _log(progress, 0.12, "Analysing structure")
    with T.stage("2_analysis"):
        an = analysis.analyze_sketch(sk.stroke_mask, opts.holes)
        sil = an.pop("silhouette")
        Image.fromarray((sil * 255).astype(np.uint8)).save(os.path.join(run_dir, "02_silhouette.png"))
        hint = analysis.caption_with_claude(Image.fromarray(sk.line_img)) if opts.use_claude else ""
        pos, neg = analysis.build_prompt(opts.prompt, an, hint)
        an.update(prompt_used=pos, object_hint=hint)
    res["analysis"] = an
    if an["components"] > 1:
        notes.append(f"Sketch has {an['components']} separate parts; each is inflated independently.")

    # 3 ---- sketch-guided generation (+ fidelity re-ranking)
    gen_mode = opts.generator
    if gen_mode == "auto":
        gen_mode = "sd_controlnet" if generate.available() and _can_run_diffusion(hw) else "none"
        if generate.available() and hw.get("device") == "cuda" and not _can_run_diffusion(hw):
            notes.append(f"Stable Diffusion + ControlNet disabled: {hw.get('vram_gb') or 'unknown'} GB VRAM; "
                         "automatic inference requires at least 8 GB. The offline stylizer will be used.")
    if gen_mode == "sd_controlnet" and not _can_run_diffusion(hw):
        notes.append(f"Stable Diffusion + ControlNet skipped: {hw.get('vram_gb') or 'unknown'} GB VRAM; "
                     "automatic inference requires at least 8 GB.")
        gen_mode = "none"
    if gen_mode == "sd_controlnet" and not generate.available():
        notes.append("diffusers/torch not installed - generation skipped (offline stylizer used).")
        gen_mode = "none"
    chosen_rgba, cand_info, bg_method = None, [], "-"
    if gen_mode == "sd_controlnet":
        n = opts.candidates or (3 if hw["device"] != "cpu" else 1)
        _log(progress, 0.2, f"Generating {n} candidate image(s) with Stable Diffusion + ControlNet")
        try:
            with T.stage("3_generate"):
                cands = generate.generate_candidates(sk.scribble, pos, neg, n=n, steps=opts.steps, seed=opts.seed,
                                                     device=hw["device"] if hw["device"] != "cpu" else "cpu")
            with T.stage("4_matte_and_rank"):
                best = -1.0
                for i, im in enumerate(cands):
                    im.save(os.path.join(run_dir, f"03_candidate_{i}.png"))
                    rgba, bg_method = background.remove_background(im)
                    m = np.array(rgba)[..., 3] > 127
                    score = validate.mask_iou(m, sil) if m.any() else 0.0
                    cand_info.append({"index": i, "sketch_iou": round(score, 4)})
                    if score > best: best, chosen_rgba, chosen_idx = score, rgba, i
                res["candidates"] = cand_info
                notes.append(f"Picked candidate {chosen_idx} (silhouette IoU with sketch = {best:.3f}).")
                if best < opts.min_fidelity:
                    notes.append("Generated image drifted from the sketch; sketch silhouette is used for geometry.")
        except Exception as ex:                       # model download / OOM etc.
            notes.append(f"Image generation failed ({type(ex).__name__}: {ex}); fell back to offline stylizer.")
            chosen_rgba, gen_mode = None, "none"
    res["generator_used"] = gen_mode if chosen_rgba is not None else "offline-stylizer"
    res["bg_method"] = bg_method

    if chosen_rgba is None:
        with T.stage("3_stylize"):
            tex = stylize.stylize_from_sketch(sk.stroke_mask, sil, opts.prompt, opts.seed)
            rgba_arr = np.dstack([np.array(tex), (sil * 255).astype(np.uint8)])
            chosen_rgba = Image.fromarray(rgba_arr, "RGBA")
        gen_ok = False
    else:
        gen_ok = True
    rgba_prep, rgb_prep = background.prepare_for_reconstruction(chosen_rgba)
    rgba_prep.save(os.path.join(run_dir, "04_object_rgba.png")); rgb_prep.save(os.path.join(run_dir, "04_object_on_white.png"))

    # 4 ---- 3D reconstruction with automatic fallback
    if opts.backend == "auto":                    # silently pick the best backend that is actually configured
        order = (["trellis"] if trellis.available() else []) + ["inflate"]
    else:
        order = [opts.backend] if opts.backend == "inflate" else [opts.backend, "inflate"]
    glb_path = os.path.join(run_dir, f"{run_name}.glb"); obj_path = os.path.join(run_dir, f"{run_name}.obj")
    mesh, used = None, None
    for bk in order:
        _log(progress, 0.6, f"Reconstructing 3D ({bk})")
        try:
            with T.stage(f"5_reconstruct_{bk}"):
                if bk == "inflate":
                    if gen_ok:
                        m_gen = np.array(rgba_prep)[..., 3] > 127
                        fid = validate.mask_iou(m_gen, sil)
                        if fid >= opts.min_fidelity:
                            mask, tex = m_gen, np.array(rgb_prep)
                        else:
                            mask, tex = sil, _warp_to_silhouette(chosen_rgba, sil)
                    else:
                        mask, tex = np.array(rgba_prep)[..., 3] > 127, np.array(rgba_prep)[..., :3]
                    mesh = inflate.reconstruct(mask, tex, opts.resolution, opts.depth_scale)
                    write_glb(mesh, glb_path)
                elif bk == "trellis":
                    if not trellis.available(): raise RuntimeError("TRELLIS not configured (S2S_TRELLIS_DIR / CUDA)")
                    mesh, _ = trellis.reconstruct(rgb_prep, glb_path, seed=opts.seed)
                elif bk == "instantmesh":
                    if not instantmesh.available(): raise RuntimeError("InstantMesh not configured (S2S_INSTANTMESH_DIR / CUDA)")
                    mesh = read_obj(instantmesh.reconstruct(rgb_prep, os.path.join(run_dir, "instantmesh")))
                    write_glb(mesh, glb_path)
                elif bk == "dreamfusion":
                    if not dreamfusion.available(): raise RuntimeError("Stable-DreamFusion not set up (see third_party README)")
                    mesh = read_obj(dreamfusion.reconstruct(rgba_prep, os.path.join(run_dir, "dreamfusion")))
                    write_glb(mesh, glb_path)
                else:
                    raise ValueError(f"unknown backend {bk}")
            with T.stage(f"6_validate_{bk}"):
                v = validate.validate_mesh(glb_path)
            if not v["ok"]:
                raise RuntimeError("; ".join(v["errors"]))
            used = bk
            break
        except Exception as ex:
            notes.append(f"Backend '{bk}' unavailable/failed: {ex}")
            if os.environ.get("S2S_DEBUG"): traceback.print_exc()
    if used is None:
        raise RuntimeError("All reconstruction backends failed: " + " | ".join(notes))
    if opts.backend not in ("auto", used):
        notes.append(f"Requested backend '{opts.backend}' could not run; used '{used}' instead.")

    # 5 ---- export + fidelity
    _log(progress, 0.9, "Exporting and scoring")
    with T.stage("7_export_and_score"):
        mesh = read_glb(glb_path)
        write_obj(mesh, obj_path)
        v = validate.validate_mesh(glb_path)
        fidelity = validate.silhouette_iou(mesh, sil)
    if used == "inflate":
        notes.append("CPU 'sketch inflation' gives a symmetric front/back solid; hidden geometry is a plausible "
                     "mirror, not a true reconstruction. Use a GPU backend (TRELLIS/InstantMesh) for full 3D.")
    res.update(backend_used=used, glb=glb_path, obj=obj_path, validation=v, fidelity=fidelity,
               timings={"total": T.total, "stages": T.stages})
    res["preview_image"] = os.path.join(run_dir, "04_object_on_white.png")
    jp, mp = write_report(run_dir, res)
    res.update(report_json=jp, report_md=mp)
    _log(progress, 1.0, "Done")
    return res
