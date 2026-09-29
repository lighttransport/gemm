"""Validate our sparse skinning math against Meta's pinned PyTorch reference.

This is an offline research check, not a runtime dependency. The original
Apache-2.0 source stays in ignored tmp/vhuman-rig/compskin; only its functions
are executed here, unmodified, on a bounded sample of a vhuman head.
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from ..eye.glb import GLB

ROOT = Path(__file__).resolve().parents[3]
REFERENCE = ROOT / "tmp/vhuman-rig/compskin"
REVISION = "22e6f5fa19e533d84916bb1abd9dc88c4ac3054e"
SHAPES = ("mouthSmileLeft", "mouthSmileRight", "mouthPucker", "mouthFunnel",
          "mouthStretchLeft", "mouthStretchRight", "cheekPuff", "jawOpen")


def _reference(path: Path) -> dict:
    source = path.read_text()
    tree = ast.parse(source)
    names = {"buildTR", "compBX"}
    selected = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names],
                          type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(selected, str(path), "exec"), namespace)
    if not all(n in namespace for n in names):
        raise ValueError("reference implementation changed its skinning interface")
    return namespace


def _our_proxy(weights: torch.Tensor, transforms: torch.Tensor,
               rest: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
    """Independent tensor contraction of the reference's sparse LBS model."""
    matrices = (transforms * axes).sum(0)
    prediction = torch.einsum("bpij,pv,vj->bvi", matrices, weights, rest)
    return prediction.permute(0, 2, 1).reshape(-1, rest.shape[0])


def validate(rig_glb: Path, reference: Path = REFERENCE, steps: int = 200) -> dict:
    if not 1 <= steps <= 2000:
        raise ValueError("steps must be in [1, 2000]")
    revision = subprocess.check_output(["git", "-C", str(reference), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError(f"Compressed Skinning reference is {revision}, expected {REVISION}")
    functions = _reference(reference / "model_fit.py")
    glb = GLB.load(rig_glb)
    mesh = next(m for m in glb.doc["meshes"] if m.get("name") == "head_skin")
    names = mesh["extras"]["targetNames"]
    chosen = [(name, names.index(name)) for name in SHAPES if name in names]
    if len(chosen) < 4:
        raise ValueError("rig lacks the reference facial shape sample")
    primitive = mesh["primitives"][0]
    vertices = glb.accessor(primitive["attributes"]["POSITION"]).astype(np.float32)
    ids = np.unique(np.linspace(0, len(vertices) - 1, min(512, len(vertices)), dtype=np.int32))
    rest = torch.from_numpy(np.concatenate((vertices[ids], np.ones((len(ids), 1), np.float32)), axis=1))
    delta = np.stack([glb.accessor(primitive["targets"][i]["POSITION"])[ids]
                      for _, i in chosen]).astype(np.float32)
    goal = torch.from_numpy(delta.transpose(0, 2, 1).reshape(-1, len(ids)))
    count = 12
    generator = torch.Generator().manual_seed(23)
    weights = torch.nn.Parameter(torch.rand((count, len(ids)), generator=generator) * .1 + .01)
    transforms = torch.nn.Parameter(torch.randn((6, len(chosen), count, 1, 1), generator=generator) * 1e-4)
    axes = functions["buildTR"]("cpu")
    functions["rest_pose"] = rest
    optimizer = torch.optim.Adam([weights, transforms], lr=.003)
    for _ in range(steps):
        optimizer.zero_grad()
        normalized = weights / weights.sum(0, keepdim=True).clamp_min(1e-8)
        prediction, _, _ = functions["compBX"](normalized, transforms, axes, len(chosen), count)
        loss = ((prediction - goal) ** 2).mean() * 1e6
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            weights.clamp_(min=0)
            cutoff = torch.topk(weights, k=4, dim=0).values[-1]
            weights.masked_fill_(weights < cutoff, 0)
            keep = len(chosen) * count * 3
            threshold = torch.topk(transforms.abs().flatten(), k=keep).values[-1]
            transforms.masked_fill_(transforms.abs() < threshold, 0)
    with torch.no_grad():
        normalized = weights / weights.sum(0, keepdim=True).clamp_min(1e-8)
        reference_prediction = functions["compBX"](normalized, transforms, axes, len(chosen), count)[0]
        ours = _our_proxy(normalized, transforms, rest, axes)
        parity = float((ours - reference_prediction).abs().max())
        rmse = float(torch.sqrt(((ours - goal) ** 2).mean()) * 1000)
        baseline = float(torch.sqrt((goal ** 2).mean()) * 1000)
    if parity > 1e-7:
        raise AssertionError(f"reference/port mismatch: {parity}")
    return {"format": "vhuman.compskin_validation.v1", "reference_commit": revision,
            "reference_license": "Apache-2.0", "shapes": [name for name, _ in chosen],
            "sample_vertices": len(ids), "proxy_bones": count, "steps": steps,
            "reference_port_max_abs_m": parity, "baseline_rmse_mm": baseline,
            "proxy_rmse_mm": rmse,
            "original_delta_bytes": int(delta.nbytes),
            "proxy_parameter_bytes_approx": int(4 * len(ids) * (4 + 2) +
                                                (transforms.abs() > 0).sum().item() * (4 + 2))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rig_glb", type=Path)
    ap.add_argument("--reference", type=Path, default=REFERENCE)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    result = validate(args.rig_glb, args.reference, args.steps)
    if args.out:
        args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
