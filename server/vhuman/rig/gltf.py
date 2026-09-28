"""Skinned, morph-targeted glTF export of an assembled rig (build.RigAsset).

Joints are a node hierarchy with rest TRS; every skinned primitive uses one
skin. Morph targets are sparse accessors (only moving vertices stored),
POSITION and, for the skin, NORMAL; target names go to mesh.extras
(three.js reads them into morphTargetDictionary)."""
from __future__ import annotations

import json

import numpy as np

from ..eye.glb import ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER, FLOAT, GLBBuilder
from .common import quat_from_matrix

UNSIGNED_SHORT, UNSIGNED_INT = 5123, 5125


class RigGLB(GLBBuilder):
    def raw_accessor(self, data: np.ndarray, comp: int, kind: str, target=None, minmax=False, normalized=False):
        view = self._view(np.ascontiguousarray(data).tobytes(), target)
        acc = {"bufferView": view, "componentType": comp, "count": int(data.shape[0]), "type": kind}
        if minmax:
            flat = data.reshape(data.shape[0], -1)
            acc["min"] = [float(v) for v in flat.min(0)]
            acc["max"] = [float(v) for v in flat.max(0)]
        if normalized:
            acc["normalized"] = True
        self.doc["accessors"].append(acc)
        return len(self.doc["accessors"]) - 1

    def sparse_vec3(self, count: int, idx: np.ndarray, vals: np.ndarray) -> int:
        acc = {"componentType": FLOAT, "count": int(count), "type": "VEC3"}
        if len(idx):
            iv = self._view(np.ascontiguousarray(idx, "<u4").tobytes())
            vv = self._view(np.ascontiguousarray(vals, "<f4").tobytes())
            acc["sparse"] = {"count": int(len(idx)), "indices": {"bufferView": iv, "componentType": UNSIGNED_INT},
                             "values": {"bufferView": vv}}
            mn = np.minimum(vals.min(0), 0.0)
            mx = np.maximum(vals.max(0), 0.0)
        else:
            mn = mx = np.zeros(3)
        acc["min"] = [float(v) for v in mn]
        acc["max"] = [float(v) for v in mx]
        self.doc["accessors"].append(acc)
        return len(self.doc["accessors"]) - 1


def write(asset, out_path) -> dict:
    b = RigGLB("vhuman rig")
    skel = asset.skeleton
    names = [j["name"] for j in skel["joints"]]
    # joint nodes
    jnode = {}
    for j in skel["joints"]:
        R = np.asarray(j["rest_rotation"])
        q = quat_from_matrix(R)
        jnode[j["name"]] = b.node(j["name"], translation=j["rest_translation"], rotation=q.tolist(), root=False)
    for j in skel["joints"]:
        if j["parent"]:
            b.doc["nodes"][jnode[j["parent"]]].setdefault("children", []).append(jnode[j["name"]])
    b.doc["scenes"][0]["nodes"].append(jnode[names[0]])
    inv = np.stack([np.linalg.inv(np.asarray(j["bind"])).T.reshape(-1) for j in skel["joints"]]).astype("<f4")
    ibm = b.raw_accessor(inv, FLOAT, "MAT4")
    b.doc.setdefault("skins", []).append({"joints": [jnode[n] for n in names], "inverseBindMatrices": ibm,
                                          "skeleton": jnode[names[0]], "name": "face"})
    # textures and materials
    mats = {}
    for key, spec in asset.materials.items():
        # spec: {"gltf": material with texture-info {"index": k}, "images": {k: png bytes}}
        m = json.loads(json.dumps(spec["gltf"]))
        remap = {int(k): b.texture_png(png, f"{key}_{k}", mipmaps=True) for k, png in spec.get("images", {}).items()}

        def walk(o):
            if isinstance(o, dict):
                for k2, v in o.items():
                    if k2.endswith("Texture") and isinstance(v, dict) and "index" in v:
                        v["index"] = remap[int(v["index"])]
                    else:
                        walk(v)
            elif isinstance(o, list):
                for x in o:
                    walk(x)
        walk(m)
        mats[key] = b.material(m)
    stats = {"meshes": [], "morph_targets": 0}
    for part in asset.parts:
        attrs = {"POSITION": b.accessor(part.positions.astype(np.float32)),
                 "NORMAL": b.accessor(part.normals.astype(np.float32))}
        if part.uv is not None:
            attrs["TEXCOORD_0"] = b.accessor(part.uv.astype(np.float32))
        if part.tangents is not None:
            attrs["TANGENT"] = b.accessor(part.tangents.astype(np.float32))
        J = np.asarray(part.joints, np.uint16)
        W = np.asarray(part.weights, np.float32)
        W = W / np.maximum(W.sum(1, keepdims=True), 1e-9)
        attrs["JOINTS_0"] = b.raw_accessor(J, UNSIGNED_SHORT, "VEC4", ARRAY_BUFFER)
        attrs["WEIGHTS_0"] = b.raw_accessor(W.astype("<f4"), FLOAT, "VEC4", ARRAY_BUFFER)
        prim = {"attributes": attrs, "indices": b.accessor(part.tris.astype(np.uint32), indices=True),
                "material": mats[part.material], "mode": 4}
        tnames = []
        if part.shapes:
            prim["targets"] = []
            # glTF: every target of a primitive has the same attributes; targets
            # without normal deltas get an empty (all-zero) sparse NORMAL
            with_normals = any(dn is not None for _, _, dn in part.shapes.values())
            for name, (idx, d, dn) in part.shapes.items():
                t = {"POSITION": b.sparse_vec3(len(part.positions), idx, d)}
                if dn is not None:
                    t["NORMAL"] = b.sparse_vec3(len(part.positions), idx, dn)
                elif with_normals:
                    t["NORMAL"] = b.sparse_vec3(len(part.positions), np.zeros(0, np.int64), np.zeros((0, 3)))
                prim["targets"].append(t)
                tnames.append(name)
        mesh = {"name": part.name, "primitives": [prim]}
        if tnames:
            mesh["weights"] = [0.0] * len(tnames)
            mesh["extras"] = {"targetNames": tnames}
        b.doc["meshes"].append(mesh)
        b.node(part.name, mesh=len(b.doc["meshes"]) - 1, extras={"rig_part": part.name})
        b.doc["nodes"][-1]["skin"] = 0
        stats["meshes"].append({"name": part.name, "vertices": int(len(part.positions)),
                                "triangles": int(len(part.tris)), "targets": len(tnames)})
        stats["morph_targets"] += len(tnames)
    b.doc["asset"]["extras"] = {"rig": "rig.json", "units": "metres", "frame": "+Y up, face +Z"}
    size = b.write(out_path)
    stats["bytes"] = size
    return stats
