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


def template_invariants(tm, layout=None) -> dict:
    from .common import boundary_edges
    from . import template as T
    e = np.concatenate([tm.tris[:, [0, 1]], tm.tris[:, [1, 2]], tm.tris[:, [2, 0]]])
    key = np.sort(e, 1)
    _, cnt = np.unique(key, axis=0, return_counts=True)
    directed = {tuple(x) for x in e.tolist()}
    return {"vertices": int(tm.n), "triangles": int(len(tm.tris)),
            "nonmanifold_edges": int((cnt > 2).sum()), "boundary_edges": int(len(boundary_edges(tm.tris))),
            "expected_boundary": 2 * (layout or T.Layout()).eye_n + (layout or T.Layout()).neck_n,
            "duplicate_directed": int(len(e) - len(directed)),
            "unused_vertices": int(tm.n - len(np.unique(tm.tris)))}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out = Path(argv[0])
    out.mkdir(parents=True, exist_ok=True)
    from ..head import pipeline, fit
    from . import build, template
    from . import template as T
    head = out / "head"
    head.mkdir(exist_ok=True)
    portrait = pipeline.mock_portrait(head / "portrait.png", seed=0)
    pipeline.mock_head_glb(head / "pixal3d.glb", portrait, 20.0)
    fit.fit_head(portrait, head / "pixal3d.glb", head, fov_deg=20.0, res=256)
    # two synthetic "expression portraits" (smooth warps of the neutral one)
    from PIL import Image
    from . import exprdata
    ex = head / "rig" / "expressions"
    ex.mkdir(parents=True, exist_ok=True)
    im = Image.open(portrait).convert("RGBA")
    bg = Image.new("RGBA", im.size, (128, 128, 128, 255))
    bg.alpha_composite(im)
    bg.convert("RGB").save(ex / "ref.png")
    man = {"ref": "ref.png", "expressions": {}}
    for name in ("smile", "brows_up"):
        exprdata._mock_expression(ex / "ref.png", ex / f"{name}.png", name)
        prompt, controls, group = exprdata.EXPRESSIONS[name]
        man["expressions"][name] = {"file": f"{name}.png", "prompt": prompt, "controls": controls,
                                    "wrinkle_group": group}
    (ex / "manifest.json").write_text(json.dumps(man))
    rep = build.assemble(head, head / "rig", res=512, iters=60, cache_dir=out / "cache", preview=False,
                         log=lambda m: None, deformer_samples=192)
    tm = template.get(out / "cache")
    rig = json.loads((head / "rig" / "rig.json").read_text())
    summary = {"template": template_invariants(tm), "report": {k: rep[k] for k in ("shapes", "controls", "joints")},
               "fallback_mouth": rep["features"]["mouth"]["fallback"],
               "files": sorted(p.name for p in (head / "rig").iterdir()),
               "rig_controls": len(rig["controls"]), "blendshapes": [b["name"] for b in rig["blendshapes"]],
               "register": rep["register"], "gltf": rep["gltf"], "usd": rep["usd"], "deformer": rep["deformer"],
               "ml_targets": len(rig.get("ml_deformer", {}).get("targets", [])),
               "expressions": rep["expressions"], "wrinkles": rep["wrinkles"], "lods": rep["lods"],
               "lod_templates": {}}
    from . import lod
    for lv in rep["lods"]:
        tl = lod.get(int(lv), out / "cache")
        summary["lod_templates"][str(lv)] = template_invariants(tl, lod.LAYOUTS[int(lv)])
        idx, w = lod.mapping(tm, tl, int(lv))
        ring = tl.kind != T.KIND["free"]
        summary["lod_templates"][str(lv)]["exact_ring_mapping"] = bool((w[ring, 0] == 1.0).all())
    print(json.dumps(summary, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
