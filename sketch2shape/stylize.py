"""Offline texture synthesis used when no diffusion model is available (CPU-only machines).

Colour comes from the optional prompt (colour / material words), shading from a distance-field
ambient-occlusion term, and the user's own pencil lines are multiplied back in so the sketch's
inner details stay visible on the 3D surface.
"""
from __future__ import annotations
import cv2
import numpy as np
from PIL import Image
from .analysis import prompt_color


def stylize_from_sketch(stroke_mask: np.ndarray, silhouette: np.ndarray, prompt: str = "", seed: int = 0) -> Image.Image:
    rgb = np.array(prompt_color(prompt), np.float32)
    dt = cv2.distanceTransform(silhouette.astype(np.uint8), cv2.DIST_L2, 5)
    d = np.clip(dt / max(dt.max() * 0.6, 1e-6), 0, 1)
    shade = 0.62 + 0.38 * np.sqrt(d)                              # darker towards the rim (fake AO)
    rng = np.random.default_rng(seed)
    grain = cv2.GaussianBlur(rng.normal(0, 1, silhouette.shape).astype(np.float32), (0, 0), 1.2) * 0.035
    img = rgb[None, None, :] * (shade + grain)[..., None]
    ink = cv2.GaussianBlur(stroke_mask.astype(np.float32), (0, 0), 0.8)
    img = img * (1 - 0.55 * ink[..., None])
    img[~silhouette] = rgb * 0.8
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
