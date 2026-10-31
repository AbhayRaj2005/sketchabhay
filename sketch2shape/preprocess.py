"""Stage 1 - input conditioning: any drawing/photo of a sketch -> clean square line-art.

Ideas adapted from the AWS sample's `preprocess_sketch` (grayscale -> adaptive threshold ->
crop) and extended with polarity detection, speck removal and a ControlNet scribble map.
"""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
from PIL import Image


@dataclass
class SketchData:
    line_img: np.ndarray      # (S,S) uint8, strokes=0, paper=255
    stroke_mask: np.ndarray   # (S,S) bool, True on strokes
    scribble: Image.Image     # RGB, white strokes on black (ControlNet scribble format)
    info: dict


def to_pil(obj) -> Image.Image:
    """Accept PIL / ndarray / path / Gradio ImageEditor dict and return a PIL image."""
    if obj is None:
        raise ValueError("No sketch supplied. Draw on the canvas or upload an image.")
    if isinstance(obj, dict):                       # gradio ImageEditor / Sketchpad value
        for key in ("composite", "background", "image"):
            if obj.get(key) is not None:
                obj = obj[key]; break
        else:
            layers = obj.get("layers") or []
            if not layers:
                raise ValueError("Canvas is empty.")
            obj = layers[0]
    if isinstance(obj, (str, bytes)) or hasattr(obj, "__fspath__"):
        obj = Image.open(obj)
    if isinstance(obj, np.ndarray):
        obj = Image.fromarray(obj.astype(np.uint8))
    if not isinstance(obj, Image.Image):
        raise TypeError(f"Unsupported sketch type: {type(obj)}")
    return obj


def _flatten_on_white(img: Image.Image) -> np.ndarray:
    """Composite transparency on white. A transparent canvas with dark strokes works too;
    a transparent canvas with *white* strokes is detected and inverted."""
    if img.mode in ("RGBA", "LA", "P"):
        rgba = np.array(img.convert("RGBA")).astype(np.float32)
        a = rgba[..., 3:4] / 255.0
        rgb = rgba[..., :3]
        if a.mean() < 0.999:                        # genuinely transparent canvas
            opaque = a[..., 0] > 0.1
            if opaque.any() and rgb[opaque].mean() > 200:   # white ink on transparent
                rgb = 255 - rgb
            return (rgb * a + 255 * (1 - a)).astype(np.uint8)
        return rgb.astype(np.uint8)
    return np.array(img.convert("RGB"))


def clean_sketch(image, size: int = 512, margin: float = 0.08, min_speck: float = 2e-5) -> SketchData:
    img = to_pil(image)
    rgb = _flatten_on_white(img)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    scale = 1024 / max(h, w)
    if scale < 1:                                   # keep processing cheap on big photos
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    info = {"input_size": (w, h)}

    # polarity: sketches are normally dark-on-light. Blackboard style (light-on-dark) -> invert.
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    if np.median(border) < 110:
        gray = 255 - gray; info["inverted"] = True

    # contrast + noise reduction
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    lo, hi = np.percentile(gray, 1), np.percentile(gray, 99)
    gray = np.clip((gray.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1), 0, 255).astype(np.uint8)

    paper = gray[gray >= np.percentile(gray, 50)]
    clean_paper = paper.size == 0 or np.std(paper) < 6   # digital canvas / scan
    if clean_paper:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        info["threshold"] = "otsu"
    else:                                           # uneven lighting (phone photo of paper)
        bs = max(15, (min(gray.shape) // 16) | 1)
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, bs, 12)
        info["threshold"] = "adaptive"
    if (bw > 0).mean() > 0.5:                       # threshold picked the paper -> flip
        bw = 255 - bw

    # remove specks
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    keep = np.zeros(n, bool); keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(4, min_speck * bw.size)
    bw = (keep[lab] * 255).astype(np.uint8)
    if not bw.any():
        raise ValueError("No strokes detected. Draw darker/thicker lines on a plain background.")

    # crop to content, pad to square, resize
    ys, xs = np.where(bw > 0)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = bw[y0:y1, x0:x1]
    side = int(max(crop.shape) * (1 + 2 * margin))
    canvas = np.zeros((side, side), np.uint8)
    oy, ox = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
    canvas[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    canvas = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    mask = canvas > 96
    # normalise stroke width so thin pencil and fat marker both give usable control images
    thick = cv2.dilate(mask.astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    scribble = Image.fromarray(np.stack([thick] * 3, -1))
    info["stroke_ratio"] = float(mask.mean())
    return SketchData(255 - canvas, mask, scribble, info)
