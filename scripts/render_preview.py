"""Software renderer (numpy + OpenCV) -> turntable contact sheet of a .glb. No GPU / GL needed."""
from __future__ import annotations
import os, sys, cv2, numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sketch2shape.meshio import read_glb


def render(mesh, yaw_deg=0, pitch_deg=15, size=360):
    v = mesh.vertices.astype(np.float64); f = mesh.faces
    ry, rx = np.radians(yaw_deg), np.radians(pitch_deg)
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    p = v @ Ry.T @ Rx.T
    tri = p[f]; n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    col = np.full((len(f), 3), 190.0)
    if mesh.uv is not None and mesh.texture is not None:
        t = np.array(mesh.texture.convert("RGB")); uvc = mesh.uv[f].mean(1)
        col = t[np.clip((uvc[:, 1] * (t.shape[0] - 1)).astype(int), 0, t.shape[0] - 1),
                np.clip((uvc[:, 0] * (t.shape[1] - 1)).astype(int), 0, t.shape[1] - 1)].astype(float)
    light = np.array([-0.4, 0.6, 0.7]); light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.clip(np.abs(n @ light), 0, 1)
    vis = n[:, 2] > 0
    order = np.argsort(tri[:, :, 2].mean(1))                       # painter's algorithm, far -> near
    s = size * 0.8 / max(np.ptp(p[:, 0]), np.ptp(p[:, 1]))
    xy = np.stack([p[:, 0] * s + size / 2, size / 2 - p[:, 1] * s], 1)
    img = np.full((size, size, 3), 244, np.uint8)
    pts = np.round(xy[f]).astype(np.int32)
    for i in order:
        if vis[i]:
            cv2.fillConvexPoly(img, pts[i], (col[i] * shade[i]).clip(0, 255).tolist())
    return img


def sheet(glb, out, size=300):
    m = read_glb(glb)
    ims = [render(m, y, 15, size) for y in (0, 40, 90, 180)]
    Image.fromarray(np.hstack(ims)).save(out)


if __name__ == "__main__":
    sheet(sys.argv[1], sys.argv[2])
