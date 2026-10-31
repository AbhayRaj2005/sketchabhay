"""Mesh validation + sketch-fidelity scoring (judging: 'non-empty geometry, opens, usable export')."""
from __future__ import annotations
import itertools, os
import cv2
import numpy as np
from .mesh import Mesh
from .meshio import read_glb


def _edge_stats(f: np.ndarray):
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    lo, hi = np.minimum(e[:, 0], e[:, 1]), np.maximum(e[:, 0], e[:, 1])
    key = lo.astype(np.int64) * (int(f.max()) + 1) + hi
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    boundary = int((cnt == 1).sum()); nonman = int((cnt > 2).sum())
    # consistent orientation: each undirected edge used twice must be traversed in opposite directions
    dkey = e[:, 0].astype(np.int64) * (int(f.max()) + 1) + e[:, 1]
    _, dcnt = np.unique(dkey, return_counts=True)
    return boundary, nonman, int((dcnt > 1).sum()), int(len(cnt))


def validate_mesh(mesh_or_path, check_file: bool = True) -> dict:
    path = mesh_or_path if isinstance(mesh_or_path, str) else None
    out = {"ok": False, "errors": [], "warnings": []}
    try:
        mesh = read_glb(path) if path and path.lower().endswith(".glb") else mesh_or_path
        if path and path.lower().endswith(".glb"):
            out["file_size_kb"] = round(os.path.getsize(path) / 1024, 1)
            out["opens_with_reader"] = True
    except Exception as ex:                                   # file did not open
        out["errors"].append(f"cannot open: {ex}"); out["opens_with_reader"] = False
        return out
    v, f = mesh.vertices, mesh.faces
    out.update(vertices=int(len(v)), faces=int(len(f)))
    if len(v) == 0 or len(f) == 0:
        out["errors"].append("empty geometry"); return out
    if not np.isfinite(v).all(): out["errors"].append("NaN/Inf vertices")
    if f.min() < 0 or f.max() >= len(v): out["errors"].append("face index out of range"); return out
    ext = v.max(0) - v.min(0)
    out["bbox_extent"] = [round(float(x), 4) for x in ext]
    flat = float(ext.min() / max(ext.max(), 1e-9))
    out["depth_ratio"] = round(flat, 4)                       # ~0 => flat plane (not acceptable)
    if flat < 0.02: out["errors"].append("geometry is flat (depth ratio < 2%)")
    tri = v[f]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    out["degenerate_faces"] = int((area < 1e-12).sum())
    b, nm, same_dir, ne = _edge_stats(f)
    out.update(boundary_edges=b, non_manifold_edges=nm, watertight=bool(b == 0 and nm == 0 and same_dir == 0))
    out["euler_characteristic"] = int(len(np.unique(f)) - ne + len(f))
    out["volume"] = round(mesh.signed_volume(), 6)
    out["has_uv"] = mesh.uv is not None
    out["has_texture"] = mesh.texture is not None
    if not out["watertight"]: out["warnings"].append("mesh is not watertight (fine for viewing, not for 3D printing)")
    if not out["has_texture"]: out["warnings"].append("no texture embedded")
    try:                                                      # optional second opinion
        import trimesh
        if path:
            tm = trimesh.load(path, force="mesh"); out["trimesh_loads"] = bool(len(tm.vertices) > 0)
    except ImportError:
        pass
    except Exception as ex:
        out["errors"].append(f"trimesh failed to load: {ex}")
    out["ok"] = not out["errors"]
    return out


# ---------------------------------------------------------------- sketch fidelity
def _fit(mask: np.ndarray, size: int = 256) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(ys) == 0: return np.zeros((size, size), bool)
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8)
    s = (size - 8) / max(crop.shape)
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size), np.uint8)
    oy, ox = (size - crop.shape[0]) // 2, (size - crop.shape[1]) // 2
    out[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    return out > 0


def _raster(v2: np.ndarray, f: np.ndarray, size: int = 256) -> np.ndarray:
    mn, mx = v2.min(0), v2.max(0)
    s = (size - 8) / max((mx - mn).max(), 1e-9)
    p = (v2 - (mn + mx) / 2) * s + size / 2
    p[:, 1] = size - p[:, 1]                                   # image y points down
    tri = np.round(p[f]).astype(np.int32)
    img = np.zeros((size, size), np.uint8)
    cv2.fillPoly(img, list(tri), 255)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0


def silhouette_iou(mesh: Mesh, sketch_mask: np.ndarray, max_faces: int = 60000) -> dict:
    """Best IoU between the sketch silhouette and any axis-aligned orthographic silhouette of the mesh
    (12 candidates: 6 ordered axis pairs x mirror). Orientation-agnostic, so it also scores GPU backends."""
    f = mesh.faces
    if len(f) > max_faces:
        f = f[np.random.default_rng(0).choice(len(f), max_faces, replace=False)]
    ref = _fit(sketch_mask)
    best, best_view = 0.0, None
    for (a, b), flip in itertools.product(itertools.permutations(range(3), 2), (False, True)):
        v2 = mesh.vertices[:, [a, b]].astype(np.float64).copy()
        if flip: v2[:, 0] *= -1
        sil = _fit(_raster(v2, f))
        iou = float((sil & ref).sum() / max(1, (sil | ref).sum()))
        if iou > best: best, best_view = iou, f"axes({a},{b}){'/mirror' if flip else ''}"
    return {"silhouette_iou": round(best, 4), "best_view": best_view}


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two binary masks after normalising position and scale (aspect preserved)."""
    fa, fb = _fit(a), _fit(b)
    return float((fa & fb).sum() / max(1, (fa | fb).sum()))
