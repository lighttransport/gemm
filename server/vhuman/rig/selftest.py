"""End-to-end rig check on a mock head (no Qwen/Pixal3D; CPU or GPU PyTorch).

    python -m server.vhuman.rig.selftest OUT_DIR

Builds the synthetic portrait + ellipsoid head of head/pipeline.py, fits the
eyes, rigs it (small atlas, few iterations) and prints a JSON summary with
invariants the unit tests assert (template topology, weights, shapes, files).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def template_invariants(tm) -> dict:
    from .common import boundary_edges
    from . import template as T
    e = np.concatenate([tm.tris[:, [0, 1]], tm.tris[:, [1, 2]], tm.tris[:, [2, 0]]])
    key = np.sort(e, 1)
    _, cnt = np.unique(key, axis=0, return_counts=True)
    directed = {tuple(x) for x in e.tolist()}
    return {"vertices": int(tm.n), "triangles": int(len(tm.tris)),
            "nonmanifold_edges": int((cnt > 2).sum()), "boundary_edges": int(len(boundary_edges(tm.tris))),
            "expected_boundary": 2 * T.EYE_N + T.NECK_N,
            "duplicate_directed": int(len(e) - len(directed)),
            "unused_vertices": int(tm.n - len(np.unique(tm.tris)))}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out = Path(argv[0])
    out.mkdir(parents=True, exist_ok=True)
    from ..head import pipeline, fit
    from . import build, template
    head = out / "head"
    head.mkdir(exist_ok=True)
    portrait = pipeline.mock_portrait(head / "portrait.png", seed=0)
    pipeline.mock_head_glb(head / "pixal3d.glb", portrait, 20.0)
    fit.fit_head(portrait, head / "pixal3d.glb", head, fov_deg=20.0, res=256)
    rep = build.assemble(head, head / "rig", res=512, iters=60, cache_dir=out / "cache", preview=False,
                         log=lambda m: None, deformer_samples=192)
    tm = template.get(out / "cache")
    rig = json.loads((head / "rig" / "rig.json").read_text())
    summary = {"template": template_invariants(tm), "report": {k: rep[k] for k in ("shapes", "controls", "joints")},
               "fallback_mouth": rep["features"]["mouth"]["fallback"],
               "files": sorted(p.name for p in (head / "rig").iterdir()),
               "rig_controls": len(rig["controls"]), "blendshapes": [b["name"] for b in rig["blendshapes"]],
               "register": rep["register"], "gltf": rep["gltf"], "usd": rep["usd"], "deformer": rep["deformer"],
               "ml_targets": len(rig.get("ml_deformer", {}).get("targets", []))}
    print(json.dumps(summary, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
