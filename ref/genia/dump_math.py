"""Capture pinned upstream DDA plus independent NumPy guidance/fusion oracles.

No Torch/Numba/model downloads are required. The three upstream DDA functions
are read from the pinned checkout, with only JIT decorators removed. Their
function bodies are unchanged. Generated files belong under repo tmp/.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=ROOT / "tmp/genia-upstream")
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/genia-ref/math")
    args = parser.parse_args()
    lock = json.loads((Path(__file__).parent / "lock.json").read_text())
    revision = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != lock["revision"]:
        raise ValueError("GenIA checkout does not match the reference lock")
    source = (args.upstream / "core/utils/visibility.py").read_text()
    if hashlib.sha256(source.encode()).hexdigest() != lock["visibility_source_sha256"]:
        raise ValueError("pinned GenIA visibility source was modified")
    names = {"_ray_box_intersection", "_check_ray_occluded", "_compute_self_occlusion_jit"}
    functions = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            functions.append(node)
    if len(functions) != len(names):
        raise ValueError("pinned DDA function layout changed")
    namespace = {"np": np, "math": math, "Tuple": Tuple, "numba": SimpleNamespace(prange=range)}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "pinned_genia_dda", "exec"), namespace)
    rng = np.random.default_rng(42)
    coords = np.unique(rng.integers(0, 64, size=(1024, 3), dtype=np.int32), axis=0)
    occupancy = np.zeros((64, 64, 64), np.uint8)
    occupancy[tuple(coords.T)] = 1
    cameras = np.asarray([[-64, 32, 32], [128, 32, 32], [32, -64, 32], [32, 32, 128], [32, 32, 32], [-64, -64, -64]], np.float32)
    visibility = np.stack([namespace["_compute_self_occlusion_jit"](
        coords.astype(np.int64), camera.astype(np.float64), occupancy.astype(bool), 64, 16.)
        for camera in cameras]).astype(np.float32)
    velocities = rng.normal(size=(6, len(coords), 8)).astype(np.float32)
    scores = visibility
    logits = (30*scores).astype(np.float64)
    probabilities = np.exp(logits-logits.max(axis=0, keepdims=True))
    probabilities /= probabilities.sum(axis=0, keepdims=True)
    weights = np.maximum(probabilities, .001)
    weights /= weights.sum(axis=0, keepdims=True)
    fused = np.sum(weights[..., None]*velocities, axis=0).astype(np.float32)
    x = rng.normal(size=(len(coords), 8)).astype(np.float32)
    gradient = rng.normal(size=x.shape).astype(np.float32)
    clean = x+np.float32(.75)*fused
    guided = (fused.astype(np.float64)-.5*np.linalg.norm(fused.astype(np.float64))/np.linalg.norm(gradient.astype(np.float64))*gradient).astype(np.float32)
    args.output.mkdir(parents=True, exist_ok=True)
    arrays = dict(coords=coords, occupancy=occupancy, cameras=cameras, visibility=visibility,
                  velocities=velocities, weights=weights.astype(np.float32), fused=fused,
                  latent=x, gradient=gradient, clean=clean, guided=guided)
    files = {}
    for name, array in arrays.items():
        path = args.output / (name + ".npy")
        np.save(path, array, allow_pickle=False)
        files[path.name] = {"shape": list(array.shape), "dtype": str(array.dtype),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {"format": "gemm.genia_math_reference.v1", "lock": lock,
                "visibility_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "seed": 42, "neighbor_tolerance": 4., "temperature": 30., "floor_weight": .001,
                "time": .25, "guidance_weight": .5, "files": files}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "voxels": len(coords), "files": len(files)}))


if __name__ == "__main__":
    main()
