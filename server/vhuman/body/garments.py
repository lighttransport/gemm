"""Register isolated Pixal3D clothing meshes to an MHR bind pose."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..eye.glb import GLB
from ..rig.build import ExportPart
from ..rig.common import normalize, vertex_normals
from .assemble import FLIP_PIXAL, _project


def _source_parts(glb_path: Path):
    glb = GLB.load(glb_path)
    for mesh in glb.doc["meshes"]:
        for prim in mesh["primitives"]:
            a = prim["attributes"]
            if "TEXCOORD_0" not in a or "POSITION" not in a:
                continue
            mat = glb.doc["materials"][prim.get("material", 0)]
            images = {}
            def collect(obj):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if key.endswith("Texture") and isinstance(value, dict) and "index" in value:
                            ti = int(value["index"])
                            src = glb.doc["textures"][ti]["source"]
                            bv = glb.doc["bufferViews"][glb.doc["images"][src]["bufferView"]]
                            start = bv.get("byteOffset", 0)
                            images[ti] = bytes(glb.bin[start:start + bv["byteLength"]])
                        else:
                            collect(value)
                elif isinstance(obj, list):
                    for value in obj:
                        collect(value)
            collect(mat)
            yield (glb.accessor(a["POSITION"]).astype(np.float64) * FLIP_PIXAL,
                   glb.accessor(a["NORMAL"]).astype(np.float64) * FLIP_PIXAL
                   if "NORMAL" in a else None,
                   glb.accessor(a["TEXCOORD_0"]).astype(np.float32),
                   glb.accessor(prim["indices"]).reshape(-1, 3).astype(np.int32),
                   copy.deepcopy(mat), images)


def import_garments(out: Path, vertices: np.ndarray, faces: np.ndarray,
                    joints: np.ndarray, weights: np.ndarray,
                    image: np.ndarray, meta: dict):
    from scipy.spatial import cKDTree
    manifest = out / "garments.json"
    if not manifest.is_file():
        return [], {}, []
    entries = json.loads(manifest.read_text())
    projected = _project(vertices, meta)
    bx = np.rint(projected[:, 0]).astype(int)
    by = np.rint(projected[:, 1]).astype(int)
    valid = (bx >= 0) & (by >= 0) & (bx < image.shape[1]) & (by < image.shape[0])
    bx = np.clip(bx, 0, image.shape[1] - 1)
    by = np.clip(by, 0, image.shape[0] - 1)
    body_normals = vertex_normals(vertices, faces)
    tree = cKDTree(vertices)
    parts, materials, report = [], {}, []
    for item in entries:
        rec = dict(item)
        name = rec["name"]
        try:
            if rec.get("status") != "reconstructed":
                raise ValueError(rec.get("reason", "no garment reconstruction"))
            mask = np.asarray(Image.open(out / rec["mask"]).convert("L")) > 127
            body_ids = np.flatnonzero(valid & mask[by, bx])
            if len(body_ids) < 30:
                raise ValueError("garment mask does not overlap the MHR body")
            target = vertices[body_ids]
            source = list(_source_parts(out / rec["glb"]))
            if not source:
                raise ValueError("Pixal3D garment has no textured mesh")
            src_points = np.concatenate([s[0] for s in source])
            ss = np.percentile(src_points, 95, axis=0) - np.percentile(src_points, 5, axis=0)
            ts = np.percentile(target, 95, axis=0) - np.percentile(target, 5, axis=0)
            scale = float(np.median(ts[:2] / np.maximum(ss[:2], 1e-6)))
            if not .01 < scale < 10:
                raise ValueError("garment reconstruction scale is invalid")
            src_center = np.median(src_points, axis=0)
            target_center = np.median(target, axis=0)
            candidates = []
            for si, (p, n, uv, tri, mat, images) in enumerate(source):
                placed = (p - src_center) * scale + target_center
                dist, near = tree.query(placed, workers=-1)
                if float(np.median(dist)) > .08 or float(np.percentile(dist, 90)) > .18:
                    raise ValueError("reconstructed garment is too far from the body")
                surface = vertices[near]
                normal = body_normals[near]
                signed = np.einsum("ij,ij->i", placed - surface, normal)
                deep = signed < -.01
                placed[deep] += (-signed[deep] + .005)[:, None] * normal[deep]
                if n is None:
                    n = vertex_normals(placed, tri)
                name_i = f"garment_{name}_{si}"
                mat_i = f"garment_{name}_{si}"
                materials[mat_i] = {"gltf": mat, "images": images}
                candidates.append(ExportPart(name_i, mat_i, placed.astype(np.float32),
                                              normalize(n).astype(np.float32), uv, None, tri,
                                              joints[near].astype(np.uint16), weights[near].astype(np.float32)))
            parts.extend(candidates)
            rec.update(status="accepted", vertices=sum(len(p.positions) for p in candidates),
                       scale=round(scale, 4))
        except (OSError, KeyError, ValueError, IndexError) as exc:
            rec.update(status="fallback_body", reason=str(exc))
        report.append(rec)
    return parts, materials, report
