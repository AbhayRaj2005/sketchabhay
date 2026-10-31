"""Stage 2 - sketch understanding (offline, classical CV) + optional Claude captioning.

Produces structural descriptors that (a) are shown to the user, (b) enrich the diffusion prompt so
the generated image keeps the sketch's structure, and (c) are logged in the run report.
"""
from __future__ import annotations
import base64, io, os, json
import numpy as np
from PIL import Image
from .silhouette import silhouette_from_strokes, topology

COLOR_WORDS = {"red": (200, 40, 40), "green": (60, 160, 80), "blue": (50, 90, 200), "yellow": (240, 210, 60),
               "orange": (235, 130, 40), "purple": (130, 70, 170), "pink": (240, 130, 180), "white": (240, 240, 238),
               "black": (35, 35, 38), "gray": (140, 140, 145), "grey": (140, 140, 145), "brown": (120, 80, 50),
               "gold": (212, 175, 55), "silver": (190, 192, 198), "teal": (40, 150, 150), "beige": (215, 195, 160)}
MATERIAL_WORDS = {"wood": (150, 105, 65), "wooden": (150, 105, 65), "metal": (170, 175, 182), "steel": (170, 175, 182),
                  "ceramic": (235, 232, 225), "plastic": (200, 200, 205), "glass": (170, 205, 215),
                  "leather": (110, 70, 45), "clay": (190, 120, 90), "stone": (150, 148, 140), "rubber": (50, 50, 52)}


def prompt_color(prompt: str, default=(196, 190, 180)):
    """First colour/material keyword in the prompt -> RGB (used by the offline texture stylizer)."""
    words = [w.strip(".,;:!").lower() for w in (prompt or "").split()]
    for w in words:
        if w in COLOR_WORDS: return COLOR_WORDS[w]
    for w in words:
        if w in MATERIAL_WORDS: return MATERIAL_WORDS[w]
    return default


def analyze_sketch(stroke_mask: np.ndarray, holes: str = "fill") -> dict:
    sil = silhouette_from_strokes(stroke_mask, holes)
    ys, xs = np.where(sil)
    h, w = np.ptp(ys) + 1, np.ptp(xs) + 1
    crop = sil[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    sym = float((crop & crop[:, ::-1]).sum() / max(1, (crop | crop[:, ::-1]).sum()))
    topo = topology(sil)
    fill = float(crop.mean())
    words = []
    aspect = w / h
    words.append("tall" if aspect < 0.6 else "wide" if aspect > 1.7 else "compact")
    if sym > 0.85: words.append("symmetrical")
    if fill > 0.72: words.append("rounded, solid")
    if topo["holes"] > 0: words.append(f"with {topo['holes']} through-hole{'s' if topo['holes'] > 1 else ''}")
    return {"silhouette": sil, "aspect": round(aspect, 3), "symmetry": round(sym, 3), "fill_ratio": round(fill, 3),
            "components": topo["components"], "holes": topo["holes"], "euler": topo["euler"],
            "descriptors": words}


def build_prompt(user_prompt: str, analysis: dict, object_hint: str = "") -> tuple[str, str]:
    """Return (positive, negative). User prompt stays optional; descriptors keep structure."""
    base = (user_prompt or "").strip() or object_hint or "a single object"
    struct = ", ".join(analysis.get("descriptors", []))
    pos = (f"{base}, {struct}, product design render, single object centered, three-quarter view, "
           f"soft studio lighting, plain white background, highly detailed, clean surfaces")
    neg = ("text, watermark, signature, multiple objects, cropped, out of frame, blurry, low quality, "
           "background clutter, floor, shadow on wall, people, hands")
    return pos, neg


def caption_with_claude(image: Image.Image, timeout: float = 20.0) -> str:
    """OPTIONAL external API (disabled unless ANTHROPIC_API_KEY is set). Returns e.g. 'a ceramic mug'."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return ""
    try:
        import anthropic
        buf = io.BytesIO(); image.convert("RGB").save(buf, "PNG")
        client = anthropic.Anthropic(api_key=key, timeout=timeout)
        msg = client.messages.create(
            model=os.environ.get("S2S_CLAUDE_MODEL", "claude-haiku-4-5"), max_tokens=40,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": base64.b64encode(buf.getvalue()).decode()}},
                {"type": "text", "text": "This is a rough line sketch of one object. Reply with only a short noun "
                                         "phrase naming it (max 6 words), e.g. 'a wooden rocking chair'."}]}])
        return msg.content[0].text.strip().strip(".")
    except Exception:
        return ""
