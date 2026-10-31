"""Dependency-free GLB / OBJ writers and a GLB reader (used for validation).

Written with numpy only so the export + validation path works on any machine,
even where trimesh / GPU libraries are not installed.
"""
from __future__ import annotations
import io, json, os, struct
import numpy as np
from PIL import Image
from .mesh import Mesh

_GLB_MAGIC, _JSON, _BIN = 0x46546C67, 0x4E4F534A, 0x004E4942


def _pad(b: bytes, fill: bytes = b"\x00") -> bytes:
    return b + fill * ((4 - len(b) % 4) % 4)


def write_glb(mesh: Mesh, path: str, name: str = "sketch2shape") -> str:
    v = np.ascontiguousarray(mesh.vertices, dtype="<f4")
    f = np.ascontiguousarray(mesh.faces, dtype="<u4")
    if mesh.normals is None:
        mesh.compute_normals()
    n = np.ascontiguousarray(mesh.normals, dtype="<f4")
    has_tex = mesh.uv is not None and mesh.texture is not None
    chunks, views, accessors = [], [], []

    def add_view(data: bytes, target=None):
        off = sum(len(c) for c in chunks)
        chunks.append(_pad(data))
        bv = {"buffer": 0, "byteOffset": off, "byteLength": len(data)}
        if target:
            bv["target"] = target
        views.append(bv)
        return len(views) - 1

    add_view(v.tobytes(), 34962)
    accessors.append({"bufferView": 0, "componentType": 5126, "count": len(v), "type": "VEC3",
                      "min": v.min(0).tolist(), "max": v.max(0).tolist()})
    add_view(n.tobytes(), 34962)
    accessors.append({"bufferView": 1, "componentType": 5126, "count": len(n), "type": "VEC3"})
    attrs = {"POSITION": 0, "NORMAL": 1}
    if has_tex:
        uv = np.ascontiguousarray(mesh.uv, dtype="<f4")
        add_view(uv.tobytes(), 34962)
        accessors.append({"bufferView": 2, "componentType": 5126, "count": len(uv), "type": "VEC2"})
        attrs["TEXCOORD_0"] = 2
    iv = add_view(f.tobytes(), 34963)
    accessors.append({"bufferView": iv, "componentType": 5125, "count": int(f.size), "type": "SCALAR",
                      "min": [int(f.min())], "max": [int(f.max())]})
    prim = {"attributes": attrs, "indices": len(accessors) - 1, "mode": 4}
    gltf = {"asset": {"version": "2.0", "generator": "Sketch2Shape AI"},
            "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": name}],
            "meshes": [{"name": name, "primitives": [prim]}]}
    if has_tex:
        buf = io.BytesIO()
        mesh.texture.convert("RGBA" if mesh.texture.mode == "RGBA" else "RGB").save(buf, "PNG")
        img_view = add_view(buf.getvalue())
        gltf["images"] = [{"bufferView": img_view, "mimeType": "image/png"}]
        gltf["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}]
        gltf["textures"] = [{"sampler": 0, "source": 0}]
        gltf["materials"] = [{"name": "baked", "doubleSided": True, "pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0}, "metallicFactor": 0.0, "roughnessFactor": 0.85}}]
        prim["material"] = 0
    else:
        gltf["materials"] = [{"name": "clay", "doubleSided": True, "pbrMetallicRoughness": {
            "baseColorFactor": [0.78, 0.76, 0.72, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.9}}]
        prim["material"] = 0
    binary = b"".join(chunks)
    gltf["bufferViews"], gltf["accessors"], gltf["buffers"] = views, accessors, [{"byteLength": len(binary)}]
    js = _pad(json.dumps(gltf, separators=(",", ":")).encode(), b" ")
    total = 12 + 8 + len(js) + 8 + len(binary)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", _GLB_MAGIC, 2, total))
        fh.write(struct.pack("<II", len(js), _JSON) + js)
        fh.write(struct.pack("<II", len(binary), _BIN) + binary)
    return path


def read_glb(path: str) -> Mesh:
    """Minimal GLB reader (single/multi primitive, float32 + uint16/uint32 indices)."""
    data = open(path, "rb").read()
    magic, ver, length = struct.unpack_from("<III", data, 0)
    if magic != _GLB_MAGIC:
        raise ValueError("not a GLB file")
    off, js, binary = 12, None, b""
    while off < length:
        clen, ctype = struct.unpack_from("<II", data, off)
        body = data[off + 8: off + 8 + clen]
        if ctype == _JSON:
            js = json.loads(body.decode())
        elif ctype == _BIN:
            binary = body
        off += 8 + clen
    ctypes = {5121: "u1", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
    ncomp = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}

    def acc(i):
        a = js["accessors"][i]; bv = js["bufferViews"][a["bufferView"]]
        s = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        k = ncomp[a["type"]]
        arr = np.frombuffer(binary, dtype=ctypes[a["componentType"]], count=a["count"] * k, offset=s)
        return arr.reshape(-1, k) if k > 1 else arr

    V, F, UV, base = [], [], [], 0
    tex = None
    for mesh in js["meshes"]:
        for p in mesh["primitives"]:
            v = acc(p["attributes"]["POSITION"]).astype(np.float32)
            idx = acc(p["indices"]).astype(np.int64).reshape(-1, 3) if "indices" in p else \
                np.arange(len(v)).reshape(-1, 3)
            V.append(v); F.append(idx + base); base += len(v)
            if "TEXCOORD_0" in p["attributes"]:
                UV.append(acc(p["attributes"]["TEXCOORD_0"]).astype(np.float32))
    if js.get("images"):
        im = js["images"][0]
        if "bufferView" in im:
            bv = js["bufferViews"][im["bufferView"]]
            s = bv.get("byteOffset", 0)
            tex = Image.open(io.BytesIO(binary[s:s + bv["byteLength"]])); tex.load()
    m = Mesh(np.concatenate(V), np.concatenate(F).astype(np.int32),
             uv=np.concatenate(UV) if len(UV) == len(V) else None, texture=tex)
    m.meta["materials"] = len(js.get("materials", []))
    return m


def write_obj(mesh: Mesh, path: str, name: str = "sketch2shape") -> str:
    base = os.path.splitext(path)[0]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    has_tex = mesh.uv is not None and mesh.texture is not None
    if mesh.normals is None:
        mesh.compute_normals()
    with open(path, "w") as fh:
        fh.write(f"# Sketch2Shape AI\n")
        if has_tex:
            fh.write(f"mtllib {os.path.basename(base)}.mtl\n")
        fh.write(f"o {name}\n")
        np.savetxt(fh, mesh.vertices, fmt="v %.6f %.6f %.6f")
        if has_tex:
            uv = mesh.uv.copy(); uv[:, 1] = 1.0 - uv[:, 1]       # OBJ origin is bottom-left
            np.savetxt(fh, uv, fmt="vt %.6f %.6f")
        np.savetxt(fh, mesh.normals, fmt="vn %.5f %.5f %.5f")
        if has_tex:
            fh.write("usemtl baked\n")
        f = mesh.faces + 1
        for a, b, c in f:
            fh.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n" if has_tex else f"f {a}//{a} {b}//{b} {c}//{c}\n")
    if has_tex:
        mesh.texture.convert("RGB").save(base + "_texture.png")
        with open(base + ".mtl", "w") as fh:
            fh.write(f"newmtl baked\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nmap_Kd {os.path.basename(base)}_texture.png\n")
    return path


def read_obj(path: str) -> Mesh:
    """Small OBJ(+MTL) loader: triangulates polygons, unifies v/vt pairs, loads map_Kd texture."""
    V, VT, F, mtl = [], [], [], None
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            p = line.split()
            if not p: continue
            if p[0] == "v": V.append([float(x) for x in p[1:4]])
            elif p[0] == "vt": VT.append([float(x) for x in p[1:3]])
            elif p[0] == "mtllib" and mtl is None: mtl = p[1]
            elif p[0] == "f":
                idx = []
                for tok in p[1:]:
                    a = tok.split("/")
                    vi = int(a[0]); vi = vi - 1 if vi > 0 else len(V) + vi
                    ti = None
                    if len(a) > 1 and a[1]:
                        ti = int(a[1]); ti = ti - 1 if ti > 0 else len(VT) + ti
                    idx.append((vi, ti))
                for k in range(1, len(idx) - 1):
                    F.append((idx[0], idx[k], idx[k + 1]))
    V = np.array(V, np.float32); VT = np.array(VT, np.float32) if VT else None
    remap, verts, uvs, faces = {}, [], [], []
    for tri in F:
        row = []
        for key in tri:
            if key not in remap:
                remap[key] = len(verts); verts.append(V[key[0]])
                uvs.append(VT[key[1]] if (VT is not None and key[1] is not None) else (0.0, 0.0))
            row.append(remap[key])
        faces.append(row)
    mesh = Mesh(np.array(verts, np.float32), np.array(faces, np.int32))
    if VT is not None:
        uv = np.array(uvs, np.float32); uv[:, 1] = 1.0 - uv[:, 1]      # back to image convention
        mesh.uv = uv
        tex_path = None
        if mtl and os.path.isfile(os.path.join(os.path.dirname(path), mtl)):
            for line in open(os.path.join(os.path.dirname(path), mtl), errors="ignore"):
                if line.strip().startswith("map_Kd"):
                    tex_path = os.path.join(os.path.dirname(path), line.split(None, 1)[1].strip()); break
        if tex_path and os.path.isfile(tex_path):
            mesh.texture = Image.open(tex_path).convert("RGB")
    return mesh.compute_normals()
