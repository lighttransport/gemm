"""Build one skinned avatar from SAM 3D Body, Pixal3D and an existing face rig.

The SAM result is an MHR *posed* mesh. Its decoded model parameters let the
official local MHR model reproduce that pose and the 127 bind transforms;
using a zero-pose bind would detach the generated clothing and photograph.
All exported geometry is metres, +Y up, face +Z. Pose correctives are baked
at the bind pose; interchangeable GLB/USD skins use linear blend skinning.
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import math
import shutil
import struct
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..eye.glb import GLB, GLBBuilder, png_bytes
from ..rig import gltf, usd
from ..rig.build import ExportPart, RigAsset
from ..rig.common import normalize, quat_from_matrix, vertex_normals

DEFAULT_MODEL = Path("/mnt/disk1/models/sam3d-body/dinov3/assets/mhr_model.pt")
DEFAULT_HEAD = Path("/mnt/disk1/models/sam3d-body/safetensors/sam3d_body_mhr_head.safetensors")
FLIP_CAMERA = np.array([1.0, -1.0, -1.0])
FLIP_PIXAL = np.array([-1.0, 1.0, -1.0])


def _safetensor(path: Path, name: str) -> np.ndarray:
    """Read one tensor without importing safetensors into the rig interpreter."""
    with path.open("rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
    spec = header[name]
    dtype = {"F32": "<f4", "I64": "<i8", "I32": "<i4"}[spec["dtype"]]
    start, stop = spec["data_offsets"]
    return np.memmap(path, dtype=dtype, mode="r", offset=8 + size + start,
                     shape=tuple(spec["shape"]))


def _state_matrix(state: np.ndarray) -> np.ndarray:
    """MHR global state: translation cm, xyzw quaternion, uniform scale."""
    x, y, z, w = state[3:7]
    r = np.array([[1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
                  [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
                  [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]])
    out = np.eye(4)
    out[:3, :3] = r * float(state[7])
    out[:3, 3] = state[:3] * .01
    return out


def _joint_entry(name, parent, bind, parent_bind=None):
    local = bind if parent_bind is None else np.linalg.inv(parent_bind) @ bind
    scale = float(np.cbrt(np.linalg.det(local[:3, :3])))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"invalid MHR joint scale at {name}")
    rot = local[:3, :3] / scale
    return {"name": name, "parent": parent, "bind": bind.tolist(),
            "rest_translation": local[:3, 3].tolist(), "rest_rotation": rot.tolist(),
            "rest_scale": scale}


def _similarity(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    """Upright similarity fit, excluding reflections and large image-pose errors."""
    a, b = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((source - a).T @ (target - b))
    fix = np.diag([1, 1, np.linalg.det(u @ vt)])
    rot = (u @ fix @ vt).T
    scale = float(np.sum((target - b) * ((source - a) @ rot.T)) /
                  np.maximum(np.sum((source - a) ** 2), 1e-12))
    if not .6 <= scale <= 1.6:
        raise ValueError(f"face/body landmark scale {scale:.3f} is implausible")
    out = np.eye(4)
    out[:3, :3] = rot * scale
    out[:3, 3] = b - rot @ a * scale
    return out, float(np.mean(np.linalg.norm(source @ out[:3, :3].T + out[:3, 3] - target, axis=1)))


def _face_alignment(face_dir: Path, vertices: np.ndarray, state: np.ndarray,
                    mapping: np.ndarray) -> tuple[np.ndarray, float]:
    feat = json.loads((face_dir / "features.json").read_text())
    eyes = {e["side"]: e["center"] for e in feat["eyes"]}
    pts = feat["points"]
    src = np.asarray([pts["nose_tip"], eyes["left"], eyes["right"],
                      pts["ear_left"], pts["ear_right"]], np.float64)
    combined = np.vstack([vertices, state[:, :3] * .01])
    dst = np.asarray(mapping[:5], np.float64) @ combined
    # MHR70 is nose, left eye, right eye, left ear, right ear. In MHR's
    # +Y-up frame the face points toward +Z, like vhuman's head frame.
    return _similarity(src, dst)


def _png_material(image: np.ndarray, name: str) -> dict:
    return {"gltf": {"name": name, "pbrMetallicRoughness":
                     {"baseColorTexture": {"index": 0}, "metallicFactor": 0.0,
                      "roughnessFactor": .85}, "doubleSided": True},
            "images": {0: png_bytes(image)}}


def _mesh_texture(glb_path: Path | None, target: np.ndarray) -> tuple[np.ndarray, dict]:
    """Nearest textured Pixal3D surface sample after silhouette-scale alignment.

    The generated front photograph is the visible-area anchor; these colours
    fill side and rear texels. A poor Pixal3D geometry fit is reported rather
    than being treated as body shape or a second rig.
    """
    fallback = np.tile(np.array([125, 119, 111], np.uint8), (len(target), 1))
    if glb_path is None or not glb_path.is_file():
        return fallback, {"source": "neutral_fallback"}
    from scipy.spatial import cKDTree
    g = GLB.load(glb_path)
    pos, uv, rgb = [], [], []
    for mesh in g.doc["meshes"]:
        for prim in mesh["primitives"]:
            attrs = prim["attributes"]
            if "TEXCOORD_0" not in attrs:
                continue
            p = g.accessor(attrs["POSITION"]).astype(np.float32)
            t = g.accessor(attrs["TEXCOORD_0"]).astype(np.float32)
            mat = g.doc["materials"][prim.get("material", 0)]
            info = mat.get("pbrMetallicRoughness", {}).get("baseColorTexture")
            if info is None:
                continue
            im = g.image(g.doc["textures"][info["index"]]["source"])
            if im.ndim == 2:
                im = np.repeat(im[..., None], 3, axis=2)
            im = im[..., :3]
            xy = np.clip(t * [im.shape[1] - 1, im.shape[0] - 1], 0, [im.shape[1] - 1, im.shape[0] - 1]).astype(int)
            pos.append(p * FLIP_PIXAL)
            rgb.append(im[xy[:, 1], xy[:, 0]])
    if not pos:
        return fallback, {"source": "neutral_fallback", "reason": "Pixal3D has no UV base colour"}
    p = np.concatenate(pos)
    c = np.concatenate(rgb)
    if len(p) > 250_000:
        ids = np.linspace(0, len(p) - 1, 250_000).astype(np.int64)
        p, c = p[ids], c[ids]
    p_center, t_center = np.median(p, axis=0), np.median(target, axis=0)
    p_size = np.percentile(p, 97, axis=0) - np.percentile(p, 3, axis=0)
    t_size = np.percentile(target, 97, axis=0) - np.percentile(target, 3, axis=0)
    scale = float(np.median(t_size[:2] / np.maximum(p_size[:2], 1e-6)))
    aligned = (p - p_center) * scale + t_center
    distance, ids = cKDTree(aligned).query(target, workers=-1)
    return c[ids], {"source": "pixal3d", "median_distance_m": round(float(np.median(distance)), 4),
                    "scale": round(scale, 4)}


def _project(vertices: np.ndarray, meta: dict) -> np.ndarray:
    v = vertices * FLIP_CAMERA
    camera = np.asarray(meta["cam_t"], np.float64)
    z = v[:, 2] + camera[2]
    if np.median(z) <= 0:
        raise ValueError("SAM 3D Body camera has invalid depth")
    f = float(meta["focal_px"])
    w, h = meta["image"]["width"], meta["image"]["height"]
    return np.column_stack([f * (v[:, 0] + camera[0]) / z + w * .5,
                            f * (v[:, 1] + camera[1]) / z + h * .5,
                            z])


def _atlas_body(vertices: np.ndarray, faces: np.ndarray, joints: np.ndarray, weights: np.ndarray,
                image: np.ndarray, meta: dict, pixal_colors: np.ndarray, res: int, neck_y: float):
    """Stable per-triangle atlas; duplicate seam vertices and bake photo/Pixal colour."""
    # Keep the neck below the attached facial head. Its lower skin is moved
    # toward the MHR surface before export, so this hidden overlap avoids a gap.
    faces = faces[vertices[faces].mean(axis=1)[:, 1] <= neck_y + .025]
    if not len(faces):
        raise ValueError("MHR body vanished at neck trim")
    grid = math.ceil(math.sqrt(len(faces)))
    idx = np.arange(len(faces))
    col, row = idx % grid, idx // grid
    tile = 1.0 / grid
    uv = np.stack([np.stack([col + .09, row + .09], -1),
                   np.stack([col + .87, row + .09], -1),
                   np.stack([col + .09, row + .87], -1)], 1).reshape(-1, 2) * tile
    positions = vertices[faces].reshape(-1, 3).astype(np.float32)
    normal = vertex_normals(vertices, faces)[faces].reshape(-1, 3).astype(np.float32)
    part = ExportPart("body_skin", "body", positions, normal, uv.astype(np.float32), None,
                      np.arange(len(positions), dtype=np.int32).reshape(-1, 3),
                      joints[faces].reshape(-1, 4).astype(np.uint16),
                      weights[faces].reshape(-1, 4).astype(np.float32))
    tex = np.zeros((res, res, 3), np.uint8)
    tex[:] = [125, 119, 111]
    photo = np.asarray(image, np.uint8)
    proj = _project(vertices, meta)
    front_count = 0
    # At 2048 this visits about one million texels. Interpolating projected
    # pixels retains detail that vertex-colour transfer would discard.
    for i, tri in enumerate(faces):
        cell_x, cell_y = int(col[i] * res / grid), int(row[i] * res / grid)
        next_x, next_y = int((col[i] + 1) * res / grid), int((row[i] + 1) * res / grid)
        xs = np.arange(cell_x, next_x)
        ys = np.arange(cell_y, next_y)
        if not len(xs) or not len(ys):
            continue
        xx, yy = np.meshgrid(xs, ys)
        u = xx / res * grid - col[i]
        v = yy / res * grid - row[i]
        b1, b2 = (u - .09) / .78, (v - .09) / .78
        b0 = 1 - b1 - b2
        inside = (b0 >= 0) & (b1 >= 0) & (b2 >= 0)
        if not inside.any():
            continue
        # Fill the entire cell with clamped edge colours, including a small
        # gutter. Mip filtering then cannot bleed the neutral atlas background
        # into every triangle edge.
        bc = np.maximum(np.stack([b0, b1, b2], -1).reshape(-1, 3), 0)
        bc /= np.maximum(bc.sum(1, keepdims=True), 1e-8)
        fill = bc @ pixal_colors[tri].astype(np.float32)
        cam_tri = vertices[tri] * FLIP_CAMERA
        nz = np.cross(cam_tri[1] - cam_tri[0], cam_tri[2] - cam_tri[0])[2]
        if nz < 0:
            q = bc @ proj[tri]
            ix = np.rint(q[:, 0]).astype(int)
            iy = np.rint(q[:, 1]).astype(int)
            ok = (ix >= 0) & (iy >= 0) & (ix < photo.shape[1]) & (iy < photo.shape[0])
            ix = np.clip(ix, 0, photo.shape[1] - 1)
            iy = np.clip(iy, 0, photo.shape[0] - 1)
            ok &= photo[iy, ix, 3] > 127
            fill[ok] = photo[iy[ok], ix[ok], :3]
            front_count += int((ok & inside.reshape(-1)).sum())
        tex[yy, xx] = np.clip(fill, 0, 255).astype(np.uint8).reshape(len(ys), len(xs), 3)
    return part, tex, {"triangles": int(len(faces)), "atlas": res, "photo_texels": front_count}


def _face_parts(face_dir: Path, alignment: np.ndarray, joints_offset: int,
                body_vertices: np.ndarray, neck_y: float):
    from scipy.spatial import cKDTree
    g = GLB.load(face_dir / "rig.glb")
    rot = alignment[:3, :3]
    scale = float(np.cbrt(np.linalg.det(rot)))
    rotation = rot / scale
    neck = body_vertices[(body_vertices[:, 1] > neck_y - .06) &
                         (body_vertices[:, 1] < neck_y + .06)]
    tree = cKDTree(neck) if len(neck) else None
    mats, parts = {}, []
    for mi, mat in enumerate(g.doc["materials"]):
        images = {}
        def collect(obj):
            if isinstance(obj, dict):
                for k, value in obj.items():
                    if k.endswith("Texture") and isinstance(value, dict) and "index" in value:
                        ti = int(value["index"])
                        src = g.doc["textures"][ti]["source"]
                        bv = g.doc["bufferViews"][g.doc["images"][src]["bufferView"]]
                        start = bv.get("byteOffset", 0)
                        images[ti] = bytes(g.bin[start:start + bv["byteLength"]])
                    else:
                        collect(value)
            elif isinstance(obj, list):
                for value in obj:
                    collect(value)
        collect(mat)
        mats[f"face_{mi}"] = {"gltf": copy.deepcopy(mat), "images": images}
    for mesh in g.doc["meshes"]:
        for prim in mesh["primitives"]:
            a = prim["attributes"]
            pos = g.accessor(a["POSITION"]).astype(np.float64) @ rot.T + alignment[:3, 3]
            normal = normalize(g.accessor(a["NORMAL"]).astype(np.float64) @ rotation.T)
            uv = g.accessor(a["TEXCOORD_0"]).astype(np.float32) if "TEXCOORD_0" in a else None
            tangents = g.accessor(a["TANGENT"]).astype(np.float32) if "TANGENT" in a else None
            if tangents is not None:
                tangents[:, :3] = tangents[:, :3] @ rotation.T
            triangles = g.accessor(prim["indices"]).reshape(-1, 3).astype(np.int32)
            if mesh["name"] == "head_skin" and tree is not None:
                # Morphs at this band are effectively zero. Make its lower
                # neck coincide with MHR before the body-head cut.
                band = pos[:, 1] < neck_y + .045
                if band.any():
                    _, near = tree.query(pos[band], workers=-1)
                    blend = np.clip((pos[band, 1] - (neck_y + .01)) / .035, 0, 1)[:, None]
                    pos[band] = neck[near] * (1 - blend) + pos[band] * blend
                # The face template has a wide lower collar. Faces touching
                # that collar stretch into visible shoulder spikes after the
                # head is registered to a different body. Keep the narrow
                # neck section, which still overlaps MHR at the cut.
                triangles = triangles[np.min(pos[triangles, 1], axis=1) >= neck_y + .02]
            shapes = {}
            for name, target in zip(mesh.get("extras", {}).get("targetNames", []), prim.get("targets", [])):
                ids, delta = g.sparse_accessor(target["POSITION"])
                ni = g.sparse_accessor(target["NORMAL"])[1] if "NORMAL" in target else None
                shapes[name] = (ids.astype(np.int64), (delta @ rot.T).astype(np.float32),
                                (ni @ rotation.T).astype(np.float32) if ni is not None and len(ni) == len(ids) else None)
            parts.append(ExportPart(mesh["name"], f"face_{prim['material']}", pos.astype(np.float32),
                                    normal.astype(np.float32), uv, tangents, triangles,
                                    g.accessor(a["JOINTS_0"]).astype(np.uint16) + joints_offset,
                                    g.accessor(a["WEIGHTS_0"]).astype(np.float32), shapes))
    return parts, mats


def _mhr_rig(model, state: np.ndarray):
    names = ["mhr_" + n for n in model.get_joint_names()]
    parents = model.character_torch.skeleton.joint_parents.cpu().numpy().astype(int)
    bind = np.stack([_state_matrix(s) for s in state])
    joints = [_joint_entry(n, names[parents[i]] if parents[i] >= 0 else None,
                           bind[i], bind[parents[i]] if parents[i] >= 0 else None)
              for i, n in enumerate(names)]
    return joints, names, bind


def _face_rig(face_dir: Path, alignment: np.ndarray, body_head_bind: np.ndarray,
              body_head_name: str):
    definition = json.loads((face_dir / "rig.json").read_text())
    source = definition["joints"]
    joint_by_name = {j["name"]: j for j in source}
    joints = []
    for j in source:
        bind = alignment @ np.asarray(j["bind"], np.float64)
        parent = j["parent"] if j["parent"] is not None else body_head_name
        parent_bind = (alignment @ np.asarray(joint_by_name[parent]["bind"], np.float64)
                       if parent in joint_by_name else body_head_bind)
        joints.append(_joint_entry(j["name"], parent, bind, parent_bind))
    return definition, joints


def assemble(head_dir: Path, out: Path, model_path: Path = DEFAULT_MODEL,
             head_assets: Path = DEFAULT_HEAD, res: int = 2048) -> dict:
    import torch
    start = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    face_dir = head_dir / "rig"
    meta = json.loads((out / "body_mhr.glb.json").read_text())
    model_params = np.asarray(meta["model_params"], np.float32)
    shape = np.asarray(meta["shape"], np.float32)
    if model_params.shape != (204,) or shape.shape != (45,):
        raise ValueError("SAM 3D Body sidecar lacks decoded MHR pose or identity")
    from server.vhuman.runtime import torch_device
    device = torch_device(torch)
    model = torch.jit.load(str(model_path), map_location=device)
    with torch.no_grad():
        vertices_t, state_t = model(torch.from_numpy(shape[None]).to(device),
                                    torch.from_numpy(model_params[None]).to(device), torch.zeros((1, 72), device=device))
    vertices = vertices_t[0].cpu().numpy().astype(np.float64) * .01
    state = state_t[0].cpu().numpy().astype(np.float64)
    faces = np.asarray(_safetensor(head_assets, "head_pose.faces"), np.int32)
    parity_max = None
    if (out / "body_mhr.glb").is_file():
        source = GLB.load(out / "body_mhr.glb")
        prim = source.doc["meshes"][0]["primitives"][0]
        predicted = source.accessor(prim["attributes"]["POSITION"])
        if predicted.shape != vertices.shape:
            raise ValueError("SAM 3D Body and MHR vertex counts differ")
        parity_max = float(np.max(np.abs(predicted * FLIP_CAMERA - vertices)))
        if parity_max > .02:
            raise ValueError(f"decoded MHR mesh differs from SAM 3D Body by {parity_max:.4f} m")
    if not (out / "body_mhr.glb").is_file():
        raw = GLBBuilder("MHR body mock geometry")
        prim = {"attributes": {"POSITION": raw.accessor(vertices.astype(np.float32))},
                "indices": raw.accessor(faces, indices=True), "mode": 4}
        raw.doc["meshes"].append({"name": "body", "primitives": [prim]})
        raw.node("body", mesh=0)
        raw.write(out / "body_mhr.glb")
    joint_ids, skin_weights = model.get_lbsw()
    joint_ids, skin_weights = joint_ids.cpu().numpy(), skin_weights.cpu().numpy()
    order = np.argsort(-skin_weights, axis=1)[:, :4]
    joints4 = np.take_along_axis(joint_ids, order, 1).astype(np.uint16)
    weights4 = np.take_along_axis(skin_weights, order, 1).astype(np.float64)
    weights4 /= np.maximum(weights4.sum(1, keepdims=True), 1e-9)
    body_joints, names, bind = _mhr_rig(model, state)
    head_index = names.index("mhr_c_head")
    neck_index = names.index("mhr_c_neck")
    neck_y = float(bind[neck_index, 1, 3])
    mapping = _safetensor(head_assets, "head_pose.keypoint_mapping")
    alignment, landmark_error = _face_alignment(face_dir, vertices, state, mapping)
    if landmark_error > .035:
        raise ValueError(f"face/body landmark registration error {landmark_error:.3f} m")
    definition, face_joints = _face_rig(face_dir, alignment, bind[head_index], names[head_index])
    # Keep the original facial control namespace; body joints have distinct
    # names so speech tracks and 52 expression morphs remain addressable.
    definition["joints"] = body_joints + face_joints
    definition["body"] = {"model": "MHR", "joints": len(body_joints),
                          "bind_pose": "SAM 3D Body prediction", "pose_correctives": "baked at bind pose"}
    for item in ([definition["ml_deformer"]["model"]] if definition.get("ml_deformer") else []):
        shutil.copyfile(face_dir / item, out / item)
    for item in (entry["file"] for entry in definition.get("wrinkles", {}).get("maps", [])):
        shutil.copyfile(face_dir / item, out / item)
    rgba = np.asarray(Image.open(out / "body_image.png").convert("RGBA"))
    pixal_glb = out / "pixal3d_full.glb"
    colors, pixal_report = _mesh_texture(pixal_glb if pixal_glb.is_file() else None, vertices)
    body_part, tex, bake_report = _atlas_body(vertices, faces, joints4, weights4, rgba, meta, colors, res, neck_y)
    Image.fromarray(tex).save(out / "body_basecolor.png")
    face_parts, face_mats = _face_parts(face_dir, alignment, len(body_joints), vertices, neck_y)
    materials = {"body": _png_material(tex, "body")} | face_mats
    parts = [body_part] + face_parts
    # Optional garments are registered against this exact MHR bind pose.
    from .garments import import_garments
    garment_parts, garment_mats, garment_report = import_garments(out, vertices, faces, joints4, weights4,
                                                                  rgba, meta)
    parts.extend(garment_parts)
    materials.update(garment_mats)
    info = {"rig_name": "avatar.json"}
    if definition.get("ml_deformer"):
        from ..rig.mlruntime import MLDeformer
        info["ml"] = MLDeformer(face_dir)
    asset = RigAsset({"joints": definition["joints"], "scale": 1.0}, parts, materials, definition, info)
    (out / "avatar.json").write_text(json.dumps(definition, indent=1))
    gltf_stats = gltf.write(asset, out / "avatar.glb")
    usd_stats = usd.write(asset, out, name="avatar.usda")
    # UsdSkel package includes its source texture paths and rig definition.
    import zipfile
    with zipfile.ZipFile(out / "avatar_usd.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out / "avatar.usda", "avatar.usda")
        z.write(out / "avatar.json", "avatar.json")
        for extra in ("deformer.lrm", "wm_smile.png", "wm_brow_up.png", "wm_brow_down.png", "wm_mouth.png"):
            if (out / extra).is_file():
                z.write(out / extra, extra)
        for tex_path in sorted((out / "textures").glob("*.png")):
            z.write(tex_path, "textures/" + tex_path.name)
    report = {"version": 1, "body_vertices": len(vertices), "body_triangles": len(body_part.tris),
              "body_joints": len(body_joints), "face_joints": len(face_joints),
              "landmark_error_m": round(landmark_error, 5),
              "mhr_parity_max_m": round(parity_max, 6) if parity_max is not None else None,
              "pixal3d": pixal_report,
              "texture": bake_report, "garments": garment_report, "glb": gltf_stats, "usd": usd_stats,
              "seconds": round(time.perf_counter() - start, 2),
              "limitations": ["MHR pose correctives are baked at the generated bind pose",
                              "A single source view does not determine unseen garment geometry"]}
    (out / "body_report.json").write_text(json.dumps(report, indent=1, default=float))
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("head_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--head-assets", type=Path, default=DEFAULT_HEAD)
    ap.add_argument("--res", type=int, choices=(1024, 2048, 4096), default=2048)
    args = ap.parse_args(argv)
    print(json.dumps(assemble(args.head_dir, args.out, args.model, args.head_assets, args.res), default=float),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
