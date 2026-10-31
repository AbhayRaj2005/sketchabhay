"""CPU backend: "sketch inflation" - silhouette -> watertight, textured 3D solid.

Always available (numpy / scipy / OpenCV only). It guarantees a real, closed 3D mesh
(front + back surfaces, side walls, thickness profile from a distance field) so the pipeline
works end-to-end on any evaluation machine. The profile is a circular cross-section scaled by the
local inscribed radius, i.e. a 'Teddy'-style inflation, followed by Taubin smoothing.
"""
from __future__ import annotations
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy import sparse
from ..mesh import Mesh


def available() -> bool:
    return True


def _taubin(v: np.ndarray, f: np.ndarray, iters: int = 12, lam: float = 0.5, mu: float = -0.53) -> np.ndarray:
    n = len(v)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    A = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.float64)
    deg = np.asarray(A.sum(1)).ravel(); deg[deg == 0] = 1
    W = sparse.diags(1 / deg) @ A
    v = v.astype(np.float64)
    for _ in range(iters):
        v = v + lam * (W @ v - v)
        v = v + mu * (W @ v - v)
    return v


def reconstruct(mask: np.ndarray, texture_rgb: np.ndarray, resolution: int = 224, depth_scale: float = 1.0,
                profile_cap: float = 0.7, smooth_iters: int = 12) -> Mesh:
    """mask: (H,W) bool silhouette.  texture_rgb: (H,W,3) uint8 aligned with the mask."""
    ys, xs = np.where(mask)
    if len(ys) < 20:
        raise ValueError("Silhouette is empty - cannot inflate.")
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    m = mask[y0:y1, x0:x1].astype(np.float32)
    tex = texture_rgb[y0:y1, x0:x1]
    sc = resolution / max(m.shape)
    nh, nw = max(8, int(round(m.shape[0] * sc))), max(8, int(round(m.shape[1] * sc)))
    m = cv2.resize(cv2.GaussianBlur(m, (0, 0), 1.0 / max(sc, 0.25)), (nw, nh), interpolation=cv2.INTER_AREA) > 0.5
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0   # drop <3px slivers
    n_cc, lab_cc, st_cc, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=4)
    if n_cc > 2:                                                            # drop specks (<0.5% of biggest part)
        ok = np.zeros(n_cc, bool); ok[1:] = st_cc[1:, cv2.CC_STAT_AREA] >= 0.005 * st_cc[1:, cv2.CC_STAT_AREA].max()
        m = ok[lab_cc]
    m = np.pad(m, 2)                                                        # border of empty pixels
    H, W = m.shape
    # fill texture outside silhouette with nearest colour so rim vertices never sample paper white
    full_m = mask[y0:y1, x0:x1]
    _, idx = ndi.distance_transform_edt(~full_m, return_indices=True)
    tex = tex[idx[0], idx[1]]

    eroded = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    boundary = m & ~eroded
    dt = cv2.distanceTransform(eroded.astype(np.uint8), cv2.DIST_L2, 5)     # 0 on rim, grows inwards
    R = max(float(dt.max()), 1.0)
    s = np.clip(dt / (profile_cap * R), 0, 1)
    z = depth_scale * R * np.sqrt(1 - (1 - s) ** 2)
    zw = ndi.gaussian_filter(z * m, 1.2) / np.maximum(ndi.gaussian_filter(m.astype(np.float32), 1.2), 1e-3)
    z = np.where(m, zw, 0.0); z[boundary] = 0.0

    weld = m & (z <= 1e-4)
    fid = -np.ones((H, W), np.int64); bid = -np.ones((H, W), np.int64)
    yy, xx = np.where(m)
    nv = len(yy)
    fid[yy, xx] = np.arange(nv)
    back_new = ~weld[yy, xx]
    bid[yy, xx] = fid[yy, xx]
    bid[yy[back_new], xx[back_new]] = nv + np.arange(back_new.sum())
    total = nv + int(back_new.sum())

    verts = np.zeros((total, 3), np.float64)
    ws = 1.0 / max(H, W)                                                     # world units per pixel
    px, py = (xx - W / 2.0) * ws, -(yy - H / 2.0) * ws
    verts[:nv] = np.stack([px, py, z[yy, xx] * ws], 1)
    verts[bid[yy, xx][back_new]] = np.stack([px[back_new], py[back_new], -z[yy, xx][back_new] * ws], 1)
    uv = np.zeros((total, 2), np.float32)
    uv0 = np.stack([(xx - 2 + 0.5) / (W - 4), (yy - 2 + 0.5) / (H - 4)], 1).clip(0, 1)   # image convention
    uv[:nv] = uv0; uv[bid[yy, xx][back_new]] = uv0[back_new]

    full = m[:-1, :-1] & m[:-1, 1:] & m[1:, :-1] & m[1:, 1:]
    by, bx = np.where(full)
    wa, wb, wc, wd = weld[by, bx], weld[by, bx + 1], weld[by + 1, bx], weld[by + 1, bx + 1]
    keep = ~(wa & wb & wc & wd)                                             # zero-thickness block -> skip
    by, bx, wb, wc = by[keep], bx[keep], wb[keep], wc[keep]
    flip = wb & wc                                                          # avoid doubled b-c diagonal on the rim
    q = lambda I: (I[by, bx], I[by, bx + 1], I[by + 1, bx], I[by + 1, bx + 1])
    a, b, c, d = q(fid)
    ffA = np.where(flip[:, None], np.stack([a, c, d], 1), np.stack([a, c, b], 1))
    ffB = np.where(flip[:, None], np.stack([a, d, b], 1), np.stack([b, c, d], 1))
    ff = np.concatenate([ffA, ffB])                                         # front, CCW seen from +z
    a, b, c, d = q(bid)
    fbA = np.where(flip[:, None], np.stack([a, d, c], 1), np.stack([a, b, c], 1))
    fbB = np.where(flip[:, None], np.stack([a, b, d], 1), np.stack([b, d, c], 1))
    fb = np.concatenate([fbA, fbB])                                         # back, reversed winding
    faces = np.concatenate([ff, fb])
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]

    # stitch side walls on every boundary edge that is used by exactly one face
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    key = np.minimum(e[:, 0], e[:, 1]) * total + np.maximum(e[:, 0], e[:, 1])
    uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    open_e = e[cnt[inv] == 1]
    # only front-surface boundary edges need a wall; their back twin is generated by the walls themselves
    is_front = (open_e[:, 0] < nv) & (open_e[:, 1] < nv)
    fe = open_e[is_front]
    back_of = np.arange(total); back_of[:nv] = bid[yy, xx]
    uf, vf = fe[:, 0], fe[:, 1]
    ub, vb = back_of[uf], back_of[vf]
    side = np.concatenate([np.stack([vf, uf, ub], 1), np.stack([vf, ub, vb], 1)])
    side = side[(side[:, 0] != side[:, 1]) & (side[:, 1] != side[:, 2]) & (side[:, 0] != side[:, 2])]
    faces = np.concatenate([faces, side]).astype(np.int32)

    if smooth_iters:
        verts = _taubin(verts, faces, smooth_iters)
    verts -= (verts.max(0) + verts.min(0)) / 2
    verts /= np.ptp(verts, axis=0).max()                                     # longest side = 1 unit
    mesh = Mesh(verts.astype(np.float32), faces, uv, texture=Image.fromarray(tex))
    mesh.fix_orientation().compute_normals()
    mesh.meta.update(backend="inflate", grid=(H, W))
    return mesh
