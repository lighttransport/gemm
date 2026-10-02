#!/usr/bin/env python3
"""Framework-free parity check of all four native RMBG2 Swin feature maps."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", type=Path, required=True)
    ap.add_argument("--runner", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()
    root = args.fixture
    spec = json.loads((root / "fixture.json").read_text())
    if spec["kind"] != "rmbg2_swin_l_backbone":
        raise ValueError("Wrong fixture kind")
    if sha256(spec["weights"]) != spec["weights_sha256"] or sha256(root / "input.f32") != spec["input_sha256"]:
        raise ValueError("Fixture/checkpoint checksum mismatch")
    out = root / ("native" if args.backend == "cpu" else "native-cuda")
    out.mkdir(exist_ok=True)
    run = subprocess.run([str(args.runner.resolve()), "--model", spec["weights"],
                          "--input", str(root / "input.f32"), "--height", str(spec["height"]),
                          "--width", str(spec["width"]), "--threads", str(args.threads),
                          "--backend", args.backend, "--device", str(args.device),
                          "--output-dir", str(out)], capture_output=True, text=True, check=True)
    checks = []
    for i, shape in enumerate(spec["shapes"]):
        ref = np.fromfile(root / f"reference_{i}.f32", dtype="<f4")
        got = np.fromfile(out / f"feature_{i}.f32", dtype="<f4")
        if got.size != np.prod(shape) or ref.shape != got.shape or not np.isfinite(got).all() or not np.isfinite(ref).all():
            raise AssertionError(f"Invalid feature {i}")
        error = np.abs(got.astype(np.float64) - ref.astype(np.float64))
        # Fixed FP32 feature gates, applied equally to every shape and stage.
        checks.append({"stage": i, "shape": shape, "max_abs": float(error.max()),
                       "mean_abs": float(error.mean()), "max_gate": 1e-3, "mean_gate": 1e-4,
                       "pass": bool(error.max() < 1e-3 and error.mean() < 1e-4)})
    report = {"pass": all(c["pass"] for c in checks), "checks": checks,
              "runner": json.loads(run.stdout),
              "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}
    (out / "parity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
