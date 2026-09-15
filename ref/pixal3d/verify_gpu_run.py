"""Run bounded, independently resumable PyTorch-oracle checks for one GPU run."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("--backend", choices=("cuda", "rocm"), required=True)
p.add_argument("--dump-dir", type=Path)
p.add_argument("--fov", type=float)
p.add_argument("--model-dir", type=Path, default=Path("/mnt/disk2/models/Pixal3D"))
p.add_argument("--reference-device", choices=("cpu", "cuda"), default="cpu")
p.add_argument("--gpu-execution", choices=("legacy", "resident"), default="resident")
p.add_argument("--gpu-kernels", choices=("auto", "blas", "mma"), default="auto")
p.add_argument("--output", type=Path, default=Path("tmp/pixal3d/verification/results.jsonl"))
p.add_argument("--skip-complete", action="store_true")
p.add_argument("--primitives", action="store_true")
p.add_argument("--conditioning", action="store_true")
p.add_argument("--flow", action="store_true")
p.add_argument("--decoders", action="store_true")
p.add_argument("--all", action="store_true")
a = p.parse_args()
if not any((a.all, a.primitives, a.conditioning, a.flow, a.decoders)):
    a.all = True
if (a.conditioning or a.flow or a.all) and not a.dump_dir:
    p.error("--dump-dir is required for conditioning/flow/all")
if (a.conditioning or a.all) and a.fov is None:
    p.error("--fov is required for conditioning/all")

def has_dump(name):
    return a.dump_dir is not None and (a.dump_dir / (name + ".safetensors")).is_file()

checks = []
common = ["--backend", a.backend, "--reference-device", a.reference_device,
          "--gpu-execution", a.gpu_execution, "--gpu-kernels", a.gpu_kernels]
oracle = ["--backend", a.backend, "--reference-device", a.reference_device]
if a.all or a.primitives:
    checks.append(("primitives", ["validate.py", *common]))
if a.all or a.conditioning:
    for stage in ("structure", "shape512", "shape1024", "texture"):
        if has_dump(stage + "_projected"):
            checks.append(("conditioning-" + stage, ["validate_full_conditioning.py", *oracle,
                "--dump-dir", str(a.dump_dir), "--stage", stage, "--fov", str(a.fov)]))
if a.all or a.flow:
    for stage in ("structure", "shape512", "shape1024", "texture"):
        if has_dump(stage + "_step_12"):
            checks.append(("flow-" + stage, ["validate_flow_stage.py", *oracle,
                "--dump-dir", str(a.dump_dir), "--stage", stage, "--step", "12",
                "--model-dir", str(a.model_dir)]))
if a.all or a.decoders:
    for stage in ("structure", "shape"):
        checks.append(("decoder-" + stage, ["validate_decoders.py", *common, "--stage", stage,
            "--model-dir", str(a.model_dir)]))

a.output.parent.mkdir(parents=True, exist_ok=True)
done = set()
if a.skip_complete and a.output.is_file():
    for line in a.output.read_text().splitlines():
        try:
            row = json.loads(line)
            if row.get("status") == "PASS":
                done.add(row["check"])
        except (ValueError, KeyError):
            pass
with a.output.open("a") as log:
    for name, args in checks:
        if name in done:
            print(json.dumps({"check": name, "status": "SKIP"}), flush=True)
            continue
        started = time.monotonic()
        proc = subprocess.run([sys.executable, str(ROOT / args[0]), *args[1:]],
                              cwd=ROOT.parent.parent, text=True, capture_output=True)
        row = {"check": name, "status": "PASS" if proc.returncode == 0 else "FAIL",
               "backend": a.backend, "reference_device": a.reference_device,
               "elapsed_seconds": time.monotonic() - started, "returncode": proc.returncode,
               "results": []}
        for line in proc.stdout.splitlines():
            try:
                value = json.loads(line)
                if isinstance(value, dict):
                    row["results"].append(value)
            except ValueError:
                pass
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps(row), flush=True)
        if proc.returncode:
            sys.stderr.write(proc.stdout + proc.stderr)
            raise SystemExit(proc.returncode)
print(f"GPU verification PASS: {len(checks)} checks ({a.backend}, PyTorch {a.reference_device} oracle)", flush=True)
