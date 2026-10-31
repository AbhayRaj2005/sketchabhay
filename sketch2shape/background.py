"""Stage 4 - foreground isolation. rembg (U2-Net) when installed, classical flood-fill otherwise."""
from __future__ import annotations
import cv2
import numpy as np
from PIL import Image

_SESSION = None


def rembg_available() -> bool:
    try:
        import rembg  # noqa: F401
        return True
    except ImportError:
        return False


def _classical(img: Image.Image, tol: int = 28) -> np.ndarray:
    """Generated product shots have a near-white studio background: flood-fill it from the borders."""
    rgb = np.array(img.convert("RGB"))
    h, w = rgb.shape[:2]
    bg = np.median(np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]]), axis=0)
    diff = np.abs(rgb.astype(int) - bg[None, None, :]).max(-1)
    cand = (diff <= tol).astype(np.uint8)
    n, lab = cv2.connectedComponents(cand, connectivity=4)
    border_labels = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    bgmask = np.isin(lab, [l for l in border_labels if l != 0 and cand[lab == l][0] == 1])
    # enclosed regions that are *exactly* background-coloured and sizeable are through-holes (rings, handles)
    strict = (diff <= 8).astype(np.uint8)
    ns, ls, sts, _ = cv2.connectedComponentsWithStats(strict, connectivity=4)
    for l in range(1, ns):
        if sts[l, cv2.CC_STAT_AREA] > 0.005 * h * w:
            bgmask |= ls == l
    fg = ~bgmask
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
    if n > 1:
        fg = lab == (1 + np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return fg


def remove_background(img: Image.Image, use_rembg: bool = True) -> tuple[Image.Image, str]:
    """Returns (RGBA image, method)."""
    global _SESSION
    img = img.convert("RGB")
    if use_rembg and rembg_available():
        try:
            from rembg import new_session, remove
            if _SESSION is None:
                _SESSION = new_session("u2net")
            out = remove(img, session=_SESSION).convert("RGBA")
            if np.array(out)[..., 3].mean() > 8:
                return out, "rembg-u2net"
        except Exception:
            pass
    fg = _classical(img)
    rgba = np.dstack([np.array(img), (fg * 255).astype(np.uint8)])
    return Image.fromarray(rgba, "RGBA"), "classical-floodfill"


def prepare_for_reconstruction(rgba: Image.Image, size: int = 512, border: float = 0.12,
                               background=(255, 255, 255)) -> tuple[Image.Image, Image.Image]:
    """Crop to object, pad to a square with margin, return (RGBA, RGB-on-white) at `size`."""
    a = np.array(rgba)
    ys, xs = np.where(a[..., 3] > 16)
    if len(ys) == 0:
        raise ValueError("Background removal produced an empty foreground.")
    crop = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    side = int(max(crop.shape[:2]) * (1 + 2 * border))
    canvas = np.zeros((side, side, 4), np.uint8)
    oy, ox = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
    canvas[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    out = Image.fromarray(canvas, "RGBA").resize((size, size), Image.LANCZOS)
    flat = Image.new("RGB", (size, size), background)
    flat.paste(out, mask=out.split()[3])
    return out, flat
