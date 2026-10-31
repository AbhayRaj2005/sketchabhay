"""Generate hand-drawn-looking sample sketches and run the full pipeline on them (reproducible demo data)."""
import os, sys, cv2, numpy as np
from PIL import Image
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sketch2shape.pipeline import Options, run
from scripts.render_preview import sheet

rng = np.random.default_rng(7)


def wobble(pts, amp=1.6):
    return (np.asarray(pts, np.float32) + rng.normal(0, amp, np.shape(pts))).astype(np.int32)


def stroke(img, pts, closed=False, th=4):
    pts = np.asarray(pts, np.float32)
    if closed: pts = np.vstack([pts, pts[:1]])
    dense = np.vstack([np.linspace(pts[i], pts[i + 1], 12, endpoint=False) for i in range(len(pts) - 1)] + [pts[-1:]])
    cv2.polylines(img, [wobble(dense)], False, 0, th, cv2.LINE_AA)


def ell(img, c, ax, th=4, a0=0, a1=360):
    t = np.radians(np.linspace(a0, a1, 90)); stroke(img, np.stack([c[0] + ax[0] * np.cos(t), c[1] + ax[1] * np.sin(t)], 1), th=th)


def canvas(): return np.full((512, 512), 255, np.uint8)


def donut():
    im = canvas(); ell(im, (256, 256), (200, 190)); ell(im, (256, 256), (70, 66)); return im


def bottle():
    im = canvas()
    stroke(im, [(220, 60), (292, 60), (292, 140), (330, 200), (340, 260), (340, 450), (320, 470), (192, 470), (172, 450), (172, 260), (182, 200), (220, 140)], closed=True)
    stroke(im, [(200, 300), (312, 300)], th=3); stroke(im, [(200, 400), (312, 400)], th=3); return im


def lamp():
    im = canvas()
    stroke(im, [(150, 120), (362, 120), (420, 250), (92, 250)], closed=True)
    stroke(im, [(244, 250), (268, 250), (268, 400), (244, 400)], closed=True)
    ell(im, (256, 430), (110, 28))
    return im


def rocket():
    im = canvas()
    stroke(im, [(256, 40), (320, 150), (332, 330), (180, 330), (192, 150)], closed=True)
    stroke(im, [(192, 270), (130, 380), (192, 340)], closed=True); stroke(im, [(320, 270), (382, 380), (320, 340)], closed=True)
    stroke(im, [(220, 330), (240, 400), (272, 400), (292, 330)], closed=True); ell(im, (256, 190), (34, 34)); return im


def racket():
    im = canvas(); ell(im, (256, 190), (130, 160)); ell(im, (256, 190), (95, 122))
    stroke(im, [(236, 345), (276, 345), (270, 480), (242, 480)], closed=True); return im


SAMPLES = [("donut", donut, "pink glazed donut", "nested"), ("bottle", bottle, "blue glass bottle", "fill"),
           ("lamp", lamp, "yellow desk lamp", "fill"), ("rocket", rocket, "red toy rocket", "fill"),
           ("tennis_racket", racket, "orange tennis racket", "nested")]

if __name__ == "__main__":
    for name, fn, prompt, holes in SAMPLES:
        p = os.path.join(ROOT, "samples", "inputs", f"{name}.png"); Image.fromarray(fn()).save(p)
        r = run(Image.open(p), Options(prompt=prompt, holes=holes, out_dir=os.path.join(ROOT, "samples", "outputs"),
                                       run_name=name, backend="inflate", generator="none"))
        sheet(r["glb"], os.path.join(r["run_dir"], "preview_turntable.png"))
        print(f"{name:14s} {r['timings']['total']:5.2f}s faces={r['validation']['faces']:>7} watertight={r['validation']['watertight']} "
              f"IoU={r['fidelity']['silhouette_iou']} holes={r['analysis']['holes']}")
