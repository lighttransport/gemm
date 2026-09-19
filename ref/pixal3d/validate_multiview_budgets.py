"""Validate completed full multiview runs across CUDA memory budgets."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("runs", nargs="+", type=Path)
a = p.parse_args()
records = []
for directory in a.runs:
    mesh = directory / "mesh.glb"
    stats = json.loads((directory / "stats.json").read_text())
    profile = json.loads((directory / "profile.json").read_text())
    assert mesh.is_file() and mesh.stat().st_size > 0, directory
    assert stats["vertices"] > 0 and stats["triangles"] > 0 and stats["shape_tokens"] > 0
    assert profile["execution"] == "resident"
    assert profile["effective_budget_bytes"] >= profile["peak_reserved_device_bytes"]
    records.append({
        "directory": str(directory),
        "sha256": hashlib.sha256(mesh.read_bytes()).hexdigest(),
        "bytes": mesh.stat().st_size,
        "vertices": stats["vertices"], "triangles": stats["triangles"],
        "shape_tokens": stats["shape_tokens"], "seconds": stats["seconds"],
        "effective_budget_mib": profile["effective_budget_bytes"] / 2**20,
        "peak_reserved_mib": profile["peak_reserved_device_bytes"] / 2**20,
        "peak_active_mib": profile["peak_active_device_bytes"] / 2**20,
    })
expected = records[0]
for record in records[1:]:
    for key in ("sha256", "bytes", "vertices", "triangles", "shape_tokens"):
        assert record[key] == expected[key], (key, expected[key], record[key])
print(json.dumps({"runs": records}, indent=2))
print("Full multiview CUDA budget validation PASS")
