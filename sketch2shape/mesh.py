"""Tiny mesh container shared by every backend."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from PIL import Image


@dataclass
class Mesh:
    vertices: np.ndarray                    # (N,3) float32
    faces: np.ndarray                       # (M,3) int32
    uv: Optional[np.ndarray] = None         # (N,2) float32, image convention (v grows downward)
    normals: Optional[np.ndarray] = None    # (N,3) float32
    texture: Optional[Image.Image] = None   # RGB(A) PIL image
    meta: dict = field(default_factory=dict)

    def compute_normals(self) -> "Mesh":
        v, f = self.vertices.astype(np.float64), self.faces
        fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])  # area weighted
        n = np.zeros_like(v)
        for k in range(3):
            np.add.at(n, f[:, k], fn)
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        ln[ln == 0] = 1
        self.normals = (n / ln).astype(np.float32)
        return self

    def signed_volume(self) -> float:
        v, f = self.vertices.astype(np.float64), self.faces
        return float(np.sum(np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]]))) / 6.0)

    def fix_orientation(self) -> "Mesh":
        if self.signed_volume() < 0:
            self.faces = self.faces[:, ::-1].copy()
        return self
