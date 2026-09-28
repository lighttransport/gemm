"""A small pure-Python glTF 2.0 binary (GLB) writer and reader.

Enough for the eye assets: several meshes and nodes, PBR materials with
embedded PNG textures and KHR material extensions (transmission, ior,
volume), and TANGENT attributes.
"""
from __future__ import annotations

import io
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image

GLB_MAGIC, JSON_CHUNK, BIN_CHUNK = 0x46546C67, 0x4E4F534A, 0x004E4942
FLOAT, UINT32 = 5126, 5125
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963
CLAMP, REPEAT, LINEAR, LINEAR_MIPMAP_LINEAR = 33071, 10497, 9729, 9987


def png_bytes(image: np.ndarray, *, compress: int = 6) -> bytes:
    """uint8 (H, W[, C]) -> PNG."""
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(image)).save(buf, format="PNG", compress_level=compress)
    return buf.getvalue()


class GLBBuilder:
    def __init__(self, generator: str = "vhuman eye"):
        self.doc: dict = {"asset": {"version": "2.0", "generator": generator}, "scene": 0,
                          "scenes": [{"nodes": []}], "nodes": [], "meshes": [], "materials": [],
                          "accessors": [], "bufferViews": [], "buffers": [], "images": [],
                          "textures": [], "samplers": []}
        self.bin = bytearray()
        self.extensions: set[str] = set()

    def _view(self, data: bytes, target: int | None = None) -> int:
        while len(self.bin) % 4:
            self.bin.append(0)
        view = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        self.bin.extend(data)
        self.doc["bufferViews"].append(view)
        return len(self.doc["bufferViews"]) - 1

    def accessor(self, array: np.ndarray, *, indices: bool = False) -> int:
        if indices:
            data = np.ascontiguousarray(array.reshape(-1), dtype="<u4")
            kind, comp, target = "SCALAR", UINT32, ELEMENT_ARRAY_BUFFER
        else:
            data = np.ascontiguousarray(array, dtype="<f4")
            kind = {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4"}[data.shape[1]]
            comp, target = FLOAT, ARRAY_BUFFER
        acc = {"bufferView": self._view(data.tobytes(), target), "componentType": comp,
               "count": int(data.shape[0]), "type": kind}
        if not indices:
            acc["min"] = [float(v) for v in data.min(0)]
            acc["max"] = [float(v) for v in data.max(0)]
        self.doc["accessors"].append(acc)
        return len(self.doc["accessors"]) - 1

    def texture(self, image: np.ndarray, name: str, *, wrap: int = CLAMP, compress: int = 6) -> int:
        sampler = {"magFilter": LINEAR, "minFilter": LINEAR_MIPMAP_LINEAR, "wrapS": wrap, "wrapT": wrap}
        if not self.doc["samplers"] or self.doc["samplers"][-1] != sampler:
            self.doc["samplers"].append(sampler)
        view = self._view(png_bytes(image, compress=compress))
        self.doc["images"].append({"name": name, "bufferView": view, "mimeType": "image/png"})
        self.doc["textures"].append({"sampler": len(self.doc["samplers"]) - 1,
                                     "source": len(self.doc["images"]) - 1, "name": name})
        return len(self.doc["textures"]) - 1

    def texture_png(self, png: bytes, name: str, *, wrap: int = REPEAT, mipmaps: bool = True) -> int:
        """A texture from ready-made PNG bytes (e.g. copied from another GLB)."""
        sampler = {"magFilter": LINEAR, "minFilter": LINEAR_MIPMAP_LINEAR if mipmaps else LINEAR,
                   "wrapS": wrap, "wrapT": wrap}
        if not self.doc["samplers"] or self.doc["samplers"][-1] != sampler:
            self.doc["samplers"].append(sampler)
        view = self._view(png)
        self.doc["images"].append({"name": name, "bufferView": view, "mimeType": "image/png"})
        self.doc["textures"].append({"sampler": len(self.doc["samplers"]) - 1,
                                     "source": len(self.doc["images"]) - 1, "name": name})
        return len(self.doc["textures"]) - 1

    def material(self, material: dict) -> int:
        for ext in material.get("extensions", {}):
            self.extensions.add(ext)
        self.doc["materials"].append(material)
        return len(self.doc["materials"]) - 1

    def mesh(self, mesh, material: int) -> int:
        attributes = {"POSITION": self.accessor(mesh.positions), "NORMAL": self.accessor(mesh.normals),
                      "TEXCOORD_0": self.accessor(mesh.uvs)}
        if mesh.tangents is not None:
            attributes["TANGENT"] = self.accessor(mesh.tangents)
        prim = {"attributes": attributes, "indices": self.accessor(mesh.indices, indices=True),
                "material": material, "mode": 4}
        self.doc["meshes"].append({"name": mesh.name, "primitives": [prim]})
        return len(self.doc["meshes"]) - 1

    def node(self, name: str, *, mesh: int | None = None, translation=None, rotation=None, scale=None,
             children=None, root: bool = True, extras: dict | None = None) -> int:
        node: dict = {"name": name}
        if mesh is not None:
            node["mesh"] = mesh
        if translation is not None:
            node["translation"] = [float(v) for v in translation]
        if rotation is not None:
            node["rotation"] = [float(v) for v in rotation]
        if scale is not None:
            node["scale"] = [float(v) for v in scale]
        if children:
            node["children"] = list(children)
        if extras:
            node["extras"] = extras
        self.doc["nodes"].append(node)
        index = len(self.doc["nodes"]) - 1
        if root:
            self.doc["scenes"][0]["nodes"].append(index)
        return index

    def to_bytes(self) -> bytes:
        doc = dict(self.doc)
        doc["buffers"] = [{"byteLength": len(self.bin)}]
        for key in ("images", "textures", "samplers"):
            if not doc[key]:
                doc.pop(key)
        if self.extensions:
            doc["extensionsUsed"] = sorted(self.extensions)
        body = json.dumps(doc, separators=(",", ":")).encode()
        body += b" " * (-len(body) % 4)
        binary = bytes(self.bin) + b"\0" * (-len(self.bin) % 4)
        total = 12 + 8 + len(body) + 8 + len(binary)
        return (struct.pack("<III", GLB_MAGIC, 2, total) + struct.pack("<II", len(body), JSON_CHUNK) + body
                + struct.pack("<II", len(binary), BIN_CHUNK) + binary)

    def write(self, path) -> int:
        data = self.to_bytes()
        path = Path(path)
        partial = path.with_name(path.name + ".partial")
        partial.write_bytes(data)
        partial.replace(path)
        return len(data)


class GLB:
    """Reader: the JSON document plus typed access to accessors and images."""

    def __init__(self, data: bytes):
        magic, version, total = struct.unpack_from("<III", data)
        if magic != GLB_MAGIC or version != 2 or total != len(data):
            raise ValueError("not a glTF 2.0 binary")
        length, kind = struct.unpack_from("<II", data, 12)
        if kind != JSON_CHUNK:
            raise ValueError("first chunk is not JSON")
        self.doc = json.loads(data[20:20 + length])
        blen, btype = struct.unpack_from("<II", data, 20 + length)
        if btype != BIN_CHUNK:
            raise ValueError("second chunk is not BIN")
        self.bin = data[28 + length:28 + length + blen]

    @classmethod
    def load(cls, path) -> "GLB":
        return cls(Path(path).read_bytes())

    def accessor(self, index: int) -> np.ndarray:
        acc = self.doc["accessors"][index]
        width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[acc["type"]]
        dtype = {FLOAT: "<f4", UINT32: "<u4", 5123: "<u2", 5121: "u1"}[acc["componentType"]]
        if "bufferView" in acc:
            view = self.doc["bufferViews"][acc["bufferView"]]
            arr = np.frombuffer(self.bin, dtype=dtype, count=acc["count"] * width,
                                offset=view.get("byteOffset", 0) + acc.get("byteOffset", 0))
        else:
            arr = np.zeros(acc["count"] * width, dtype=dtype)
        if "sparse" in acc:
            arr = arr.copy().reshape(-1, width)
            sparse = acc["sparse"]
            iv = self.doc["bufferViews"][sparse["indices"]["bufferView"]]
            vv = self.doc["bufferViews"][sparse["values"]["bufferView"]]
            it = {UINT32: "<u4", 5123: "<u2", 5121: "u1"}[sparse["indices"]["componentType"]]
            ii = np.frombuffer(self.bin, dtype=it, count=sparse["count"],
                               offset=iv.get("byteOffset", 0) + sparse["indices"].get("byteOffset", 0))
            val = np.frombuffer(self.bin, dtype=dtype, count=sparse["count"] * width,
                                offset=vv.get("byteOffset", 0) + sparse["values"].get("byteOffset", 0))
            arr[ii] = val.reshape(-1, width)
        return arr.reshape(-1, width) if width > 1 else arr

    def sparse_accessor(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return nonzero rows of a sparse glTF accessor without expanding it."""
        acc = self.doc["accessors"][index]
        if "sparse" not in acc:
            dense = self.accessor(index)
            ids = np.flatnonzero(np.any(dense != 0, axis=1))
            return ids, dense[ids]
        sparse = acc["sparse"]
        iv = self.doc["bufferViews"][sparse["indices"]["bufferView"]]
        vv = self.doc["bufferViews"][sparse["values"]["bufferView"]]
        it = {UINT32: "<u4", 5123: "<u2", 5121: "u1"}[sparse["indices"]["componentType"]]
        width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[acc["type"]]
        ids = np.frombuffer(self.bin, dtype=it, count=sparse["count"],
                            offset=iv.get("byteOffset", 0) + sparse["indices"].get("byteOffset", 0))
        vals = np.frombuffer(self.bin, dtype={FLOAT: "<f4"}[acc["componentType"]],
                             count=sparse["count"] * width,
                             offset=vv.get("byteOffset", 0) + sparse["values"].get("byteOffset", 0))
        return ids, vals.reshape(-1, width)

    def image(self, index: int) -> np.ndarray:
        view = self.doc["bufferViews"][self.doc["images"][index]["bufferView"]]
        start = view.get("byteOffset", 0)
        return np.asarray(Image.open(io.BytesIO(self.bin[start:start + view["byteLength"]])))

    def mesh_arrays(self, name: str) -> dict:
        mesh = next(m for m in self.doc["meshes"] if m["name"] == name)
        prim = mesh["primitives"][0]
        out = {k: self.accessor(v) for k, v in prim["attributes"].items()}
        out["indices"] = self.accessor(prim["indices"]).reshape(-1, 3)
        out["material"] = self.doc["materials"][prim["material"]]
        return out
