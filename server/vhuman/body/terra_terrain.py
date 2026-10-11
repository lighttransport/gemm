"""Portable Terra box geometry. No simulator/framework imports at runtime."""
from __future__ import annotations

import numpy as np


Y_TO_Z = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])


def validate_spec(value):
    if not isinstance(value, dict) or set(value) - {"boxes", "provenance"}:
        raise ValueError("terrain must be a Terra TerrainSpec object")
    boxes = value.get("boxes")
    if not isinstance(boxes, list) or len(boxes) > 128:
        raise ValueError("terrain needs a bounded boxes array")
    result = []
    for index, item in enumerate(boxes):
        if not isinstance(item, dict) or set(item) - {"pos", "size", "yaw", "pitch", "name"}:
            raise ValueError("invalid terrain box fields")
        pos, size = np.asarray(item.get("pos"), dtype=float), np.asarray(item.get("size"), dtype=float)
        yaw, pitch = float(item.get("yaw", 0)), float(item.get("pitch", 0))
        if (pos.shape != (3,) or size.shape != (3,) or not np.isfinite(np.r_[pos, size, yaw, pitch]).all()
                or np.any(size <= 0) or np.max(size) > 100 or np.max(abs(pos)) > 1000
                or abs(pitch) > np.pi / 3):
            raise ValueError("invalid terrain box dimensions or angles")
        name = item.get("name", "")
        if not isinstance(name, str) or len(name) > 128 or "seat" in name.lower():
            raise ValueError("seated support is not supported by the MHR Terra backend")
        # MuscleMimic selects collision geometry by this prefix.
        if not name.startswith("terrain_box"):
            name = f"terrain_box_{index}"
        result.append({"pos": pos.tolist(), "size": size.tolist(), "yaw": yaw, "pitch": pitch, "name": name})
    return {"boxes": result, "provenance": value.get("provenance", {})}


def box_rotation(yaw, pitch):
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    return np.array([[cy * cp, -sy, cy * sp], [sy * cp, cy, sy * sp], [-sp, 0., cp]])


class Terrain:
    """Canonical Z-up Terra geometry with one uniform transform into avatar Y-up."""
    def __init__(self, spec, *, scale=1., offset=(0., 0., 0.)):
        self.spec = validate_spec(spec)
        self.scale = float(scale)
        self.offset = np.asarray(offset, dtype=float)
        if not np.isfinite(self.scale) or self.scale <= 0 or self.offset.shape != (3,) or not np.isfinite(self.offset).all():
            raise ValueError("invalid terrain transform")

    def sample(self, points):
        points = np.asarray(points, dtype=float)
        if points.shape[-1] != 3 or not np.isfinite(points).all():
            raise ValueError("invalid terrain sample points")
        source = (points - self.offset) @ Y_TO_Z.T / self.scale
        height = np.zeros(source.shape[:-1])
        normal = np.broadcast_to([0., 0., 1.], source.shape).copy()
        for box in self.spec["boxes"]:
            rotation = box_rotation(box["yaw"], box["pitch"])
            pos, size = np.asarray(box["pos"]), np.asarray(box["size"])
            # Ray/slab intersection agrees with Terra/MuJoCo at pitched end walls.
            origin = source.copy(); origin[..., 2] = 0
            local = (origin - pos) @ rotation
            direction = rotation[2]
            enter, leave = np.full(height.shape, -np.inf), np.full(height.shape, np.inf)
            face = np.full(height.shape, 2, dtype=int)
            sign = np.ones(height.shape)
            for axis in range(3):
                if abs(direction[axis]) < 1e-12:
                    miss = abs(local[..., axis]) > size[axis]
                    enter = np.where(miss, np.inf, enter)
                    continue
                a = (-size[axis] - local[..., axis]) / direction[axis]
                b = (size[axis] - local[..., axis]) / direction[axis]
                low, high = np.minimum(a, b), np.maximum(a, b)
                closer = high < leave
                face = np.where(closer, axis, face)
                sign = np.where(closer, np.sign(direction[axis]), sign)
                enter, leave = np.maximum(enter, low), np.minimum(leave, high)
            selected = (enter <= leave) & (leave > height)
            height = np.where(selected, leave, height)
            normals = np.moveaxis(rotation[:, face], 0, -1) * sign[..., None]
            normal = np.where(selected[..., None], normals, normal)
        return height * self.scale + self.offset[1], normal @ Y_TO_Z

    def height_at(self, points):
        return self.sample(points)[0]

    def clearance(self, points):
        return np.asarray(points)[..., 1] - self.height_at(points)

    def document(self):
        from scipy.spatial.transform import Rotation
        boxes = []
        for b in self.spec["boxes"]:
            rotation = Y_TO_Z.T @ box_rotation(b["yaw"], b["pitch"])
            boxes.append({"position": (np.asarray(b["pos"]) @ Y_TO_Z * self.scale + self.offset).tolist(),
                          "half_extents": (np.asarray(b["size"]) * self.scale).tolist(),
                          "rotation": Rotation.from_matrix(rotation).as_quat().tolist(), "name": b["name"]})
        return {"version": 1, "units": "m", "up_axis": "Y", "boxes": boxes,
                "floor_height": float(self.offset[1]), "source": self.spec,
                "transform": {"scale": self.scale, "offset": self.offset.tolist()}}


def append_glb(path, terrain):
    """Embed terrain as static meshes in the same scene as the animated avatar."""
    from ..eye.glb import GLB, GLBBuilder, FLOAT
    source = GLB.load(path)
    builder = GLBBuilder("vhuman Terra motion")
    builder.doc, builder.bin = source.doc, bytearray(source.bin)
    builder.extensions = set(builder.doc.get("extensionsUsed", []))
    for name in ("images", "textures", "samplers", "meshes", "materials", "accessors"):
        builder.doc.setdefault(name, [])
    material = len(builder.doc.setdefault("materials", []))
    builder.doc["materials"].append({"name": "terra_support", "pbrMetallicRoughness": {
        "baseColorFactor": [.24, .38, .48, 1.], "metallicFactor": 0., "roughnessFactor": .85}})
    document = terrain.document()
    boxes = list(document["boxes"])
    boxes.append({"position": [terrain.offset[0], document["floor_height"] - .01, terrain.offset[2]],
                  "half_extents": [10 * terrain.scale, .01, 10 * terrain.scale],
                  "rotation": [0, 0, 0, 1], "name": "terra_floor"})
    triangles = ((0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
                 (3, 7, 6), (3, 6, 2), (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5))
    cube = np.array([[-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
                     [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]], dtype=float)
    for box in boxes:
        vertices = cube[np.asarray(triangles)].reshape(-1, 3).astype("<f4")
        normal = np.cross(vertices[1::3] - vertices[::3], vertices[2::3] - vertices[::3])
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        attributes = {}
        for name, values in (("POSITION", vertices), ("NORMAL", np.repeat(normal, 3, axis=0))):
            view = builder._view(values.astype("<f4").tobytes())
            accessor = {"bufferView": view, "componentType": FLOAT, "count": len(values), "type": "VEC3"}
            if name == "POSITION":
                accessor.update(min=values.min(axis=0).tolist(), max=values.max(axis=0).tolist())
            attributes[name] = len(builder.doc["accessors"]); builder.doc["accessors"].append(accessor)
        mesh = len(builder.doc["meshes"])
        builder.doc["meshes"].append({"primitives": [{"attributes": attributes, "material": material}]})
        node = len(builder.doc["nodes"])
        builder.doc["nodes"].append({"name": box["name"] or "terra_support", "mesh": mesh,
            "translation": box["position"], "rotation": box["rotation"], "scale": box["half_extents"]})
        builder.doc["scenes"][builder.doc.get("scene", 0)]["nodes"].append(node)
    builder.write(path)
