"""Benchmark CUDA resident flow budgets and require identical predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from safetensors.numpy import load_file


ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser()
p.add_argument("--dump-dir", type=Path, required=True)
p.add_argument("--output-dir", type=Path, default=ROOT / "tmp/pixal3d/budget-matrix")
p.add_argument("--model-dir", type=Path, default=Path("/mnt/disk2/models/Pixal3D"))
p.add_argument("--repeats", type=int, default=3)
p.add_argument("--budgets", type=int, nargs="+", default=(7168, 12288, 14336))
a = p.parse_args()
if a.repeats < 1 or any(value <= 512 or value > 14336 for value in a.budgets):
    p.error("invalid repeat count or VRAM budget")

a.output_dir.mkdir(parents=True, exist_ok=True)
reports = []
reference = None
for budget in a.budgets:
    output = a.output_dir / f"shape1024-{budget}.safetensors"
    command = [
        sys.executable, str(Path(__file__).with_name("benchmark_flow_block.py")),
        "--backend", "cuda", "--dump-dir", str(a.dump_dir), "--output", str(output),
        "--model-dir", str(a.model_dir), "--stage", "shape1024", "--blocks", "30",
        "--repeats", str(a.repeats), "--gpu-execution", "resident",
        "--gpu-kernels", "auto", "--gpu-flow-precision", "mixed",
        "--vram-budget-mib", str(budget),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    report = json.loads(output.with_suffix(".json").read_text())
    values = load_file(output)["feats"]
    if reference is None:
        reference = values
    else:
        np.testing.assert_array_equal(reference, values)
    report["sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    reports.append(report)

summary = {"backend": "cuda", "stage": "shape1024", "outputs_exact": True,
           "runs": reports}
(a.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary))
print("CUDA budget matrix PASS")
