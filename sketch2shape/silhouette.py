"""Filled silhouette + topology (genus) from line art.

Concept inspired by the flood-fill / Euler-number stage of Hofer's thesis
(Sketch-to-3d-Model); this is an independent, vectorised implementation.
"""
from __future__ import annotations
import cv2
import numpy as np
from skimage.measure import euler_number


def _fill_exterior(strokes: np.ndarray) -> np.ndarray:
    pad = np.pad(strokes, 2)
    n, lab = cv2.connectedComponents((pad == 0).astype(np.uint8), connectivity=4)
    exterior = lab[0, 0]
    return (lab != exterior)[2:-2, 2:-2]


def _nested_holes(strokes: np.ndarray) -> np.ndarray:
    """Parity rule: region inside an outline = solid, inside an outline inside an outline = hole,
    and so on. Matches how people draw rings, frames, donuts and racket heads."""
    cnts, hier = cv2.findContours(strokes, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        return _fill_exterior(strokes)
    hier = hier[0]
    depth = np.zeros(len(cnts), int)
    for i in range(len(cnts)):
        p, d = hier[i][3], 0
        while p != -1:
            d += 1; p = hier[p][3]
        depth[i] = d
    canvas = np.zeros(strokes.shape, np.uint8)
    for i in np.argsort(depth):
        if depth[i] % 2 == 1:                       # contour of an enclosed (background) region
            cv2.drawContours(canvas, cnts, i, 255 if depth[i] % 4 == 1 else 0, thickness=-1)
    return (canvas > 0) | (strokes > 0)


def silhouette_from_strokes(stroke_mask: np.ndarray, holes: str = "fill") -> np.ndarray:
    """holes: 'fill' (all enclosed areas solid) | 'nested' (inner outlines carve through-holes)."""
    s = stroke_mask.astype(np.uint8) * 255
    k0 = max(3, int(round(min(s.shape) * 0.012)) | 1)
    bbox_area = max(1, np.ptp(np.where(s)[0]) * np.ptp(np.where(s)[1]))
    sil = None
    for mult in (1, 2, 3, 5):                       # progressively close gaps in open outlines
        k = k0 * mult | 1
        closed = cv2.morphologyEx(s, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        sil = _nested_holes(closed) if holes == "nested" else _fill_exterior(closed)
        if sil.sum() > 0.04 * bbox_area:
            break
    else:                                           # still open -> convex hull of the strokes
        pts = cv2.findNonZero(s)
        hull = cv2.convexHull(pts)
        sil = np.zeros_like(s); cv2.fillConvexPoly(sil, hull, 255); sil = sil > 0
    # keep components bigger than 0.5% of the largest
    n, lab, st, _ = cv2.connectedComponentsWithStats(sil.astype(np.uint8), connectivity=8)
    if n > 2:
        big = st[1:, cv2.CC_STAT_AREA].max()
        ok = np.zeros(n, bool); ok[1:] = st[1:, cv2.CC_STAT_AREA] >= 0.005 * big
        sil = ok[lab]
    return sil


def topology(mask: np.ndarray) -> dict:
    """Connected components, through-holes (genus-like count) via the Euler number."""
    e = int(euler_number(mask, connectivity=2))
    comps = int(cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1)
    return {"components": comps, "euler": e, "holes": max(0, comps - e)}
