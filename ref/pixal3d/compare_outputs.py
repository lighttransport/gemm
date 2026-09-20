"""Compare native/reference GLB geometry and optional rendered PNG views."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import trimesh

p = argparse.ArgumentParser()
p.add_argument("native", type=Path)
p.add_argument("reference", type=Path)
p.add_argument("--samples", type=int, default=100000)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--native-renders", type=Path)
p.add_argument("--reference-renders", type=Path)
p.add_argument("--output", type=Path)
a = p.parse_args()
assert 100 <= a.samples <= 1000000


def load_mesh(path):
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise RuntimeError(f"no triangle mesh in {path}")
    return mesh


def sample(mesh, count, rng):
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    vectors = np.cross(triangles[:, 1] - triangles[:, 0],
                       triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(vectors, axis=1) * .5
    if not np.isfinite(areas).all() or areas.sum() <= 0:
        raise RuntimeError("mesh has invalid triangle areas")
    faces = rng.choice(len(triangles), count, p=areas / areas.sum())
    uv = rng.random((count, 2))
    root = np.sqrt(uv[:, :1])
    bary = np.concatenate((1 - root, root * (1 - uv[:, 1:]),
                           root * uv[:, 1:]), axis=1)
    points = (triangles[faces] * bary[:, :, None]).sum(axis=1)
    normals = vectors[faces]
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
    return points, normals


def directed(source, source_normals, target, target_normals):
    distance, nearest = cKDTree(target).query(source, workers=-1)
    cosine = np.sum(source_normals * target_normals[nearest], axis=1)
    absolute_cosine = np.abs(cosine)
    return {"mean": float(distance.mean()),
            "rms": float(np.sqrt(np.mean(distance * distance))),
            "p95": float(np.quantile(distance, .95)),
            "max": float(distance.max()),
            "normal_cosine_mean": float(cosine.mean()),
            "normal_cosine_p05": float(np.quantile(cosine, .05)),
            "normal_abs_cosine_mean": float(absolute_cosine.mean()),
            "normal_abs_cosine_p05": float(np.quantile(absolute_cosine, .05))}


native = load_mesh(a.native)
reference = load_mesh(a.reference)
# Reset the deterministic stream for each mesh. Identical geometry then yields
# exact zero, while differing meshes retain the same area-quantile coverage.
native_points, native_normals = sample(native, a.samples, np.random.default_rng(a.seed))
reference_points, reference_normals = sample(reference, a.samples, np.random.default_rng(a.seed))
nr = directed(native_points, native_normals, reference_points, reference_normals)
rn = directed(reference_points, reference_normals, native_points, native_normals)
result = {
    "samples": a.samples, "seed": a.seed,
    "native": {"path": str(a.native), "vertices": len(native.vertices),
               "triangles": len(native.faces), "bounds": native.bounds.tolist()},
    "reference": {"path": str(a.reference), "vertices": len(reference.vertices),
                  "triangles": len(reference.faces), "bounds": reference.bounds.tolist()},
    "geometry": {"native_to_reference": nr, "reference_to_native": rn,
                 "symmetric_chamfer_rms": math.sqrt((nr["rms"] ** 2 + rn["rms"] ** 2) / 2)},
}

if bool(a.native_renders) != bool(a.reference_renders):
    raise ValueError("provide both render directories")
if a.native_renders:
    # preview_glb.py also emits source textures and a tiled contact sheet.
    # Compare only the independently rendered, camera-matched views.
    native_images = {path.name: path for path in a.native_renders.glob("view-*.png")}
    reference_images = {path.name: path for path in a.reference_renders.glob("view-*.png")}
    names = sorted(native_images.keys() & reference_images.keys())
    if not names:
        raise RuntimeError("render directories have no matching PNG names")
    images = []
    for name in names:
        native_image = Image.open(native_images[name])
        reference_image = Image.open(reference_images[name])
        x = np.asarray(native_image.convert("RGBA"), dtype=np.float32) / 255
        y = np.asarray(reference_image.convert("RGBA"), dtype=np.float32) / 255
        if x.shape != y.shape:
            raise RuntimeError(f"render shape mismatch for {name}: {x.shape} vs {y.shape}")
        delta = x[..., :3] - y[..., :3]
        mse = float(np.mean(delta * delta))
        suffix = name.removeprefix("view-")
        native_mask = a.native_renders / f"mask-{suffix}"
        reference_mask = a.reference_renders / f"mask-{suffix}"
        silhouette_iou = None
        if native_mask.is_file() and reference_mask.is_file():
            xa = np.asarray(Image.open(native_mask).convert("L")) > 127
            ya = np.asarray(Image.open(reference_mask).convert("L")) > 127
            union = np.logical_or(xa, ya).sum()
            silhouette_iou = float(np.logical_and(xa, ya).sum() / max(1, union))
        elif native_image.mode in ("LA", "RGBA") and reference_image.mode in ("LA", "RGBA"):
            xa, ya = x[..., 3] > .5, y[..., 3] > .5
            if xa.min() != xa.max() and ya.min() != ya.max():
                union = np.logical_or(xa, ya).sum()
                silhouette_iou = float(np.logical_and(xa, ya).sum() / max(1, union))
        images.append({"name": name, "rgb_mae": float(np.mean(np.abs(delta))),
                       "rgb_rmse": math.sqrt(mse),
                       "rgb_psnr": None if mse == 0 else -10 * math.log10(mse),
                       "silhouette_iou": silhouette_iou})
    result["renders"] = images

encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
if a.output:
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(encoded)
print(encoded, end="")
