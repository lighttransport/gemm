"""Bounded repeated CUDA recovery checks for the Pixal3D native pipeline."""
import argparse
import ctypes as C
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser()
p.add_argument("--dump-dir", type=Path, required=True)
p.add_argument("--multiview-runs", nargs=2, type=Path, required=True)
p.add_argument("--repeats", type=int, default=2)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
assert 1 <= a.repeats <= 10


class Memory(C.Structure):
    _fields_ = [("total", C.c_ulonglong), ("free", C.c_ulonglong),
                ("used", C.c_ulonglong)]


nvml = C.CDLL("libnvidia-ml.so.1")
assert nvml.nvmlInit_v2() == 0
handle = C.c_void_p()
assert nvml.nvmlDeviceGetHandleByIndex_v2(0, C.byref(handle)) == 0


def device_used():
    memory = Memory()
    assert nvml.nvmlDeviceGetMemoryInfo(handle, C.byref(memory)) == 0
    return memory.used


def run(name, command):
    started = time.monotonic()
    baseline = device_used()
    child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
    process = psutil.Process(child.pid)
    peak_rss = peak_device = 0
    while child.poll() is None:
        try:
            rss = process.memory_info().rss
            rss += sum(item.memory_info().rss for item in process.children(recursive=True))
            peak_rss = max(peak_rss, rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        peak_device = max(peak_device, device_used())
        time.sleep(.1)
    output = child.stdout.read()
    child.stdout.close()
    if child.returncode:
        raise RuntimeError(f"{name} failed ({child.returncode}):\n{output[-4000:]}")
    return {"name": name, "seconds": time.monotonic() - started,
            "baseline_device_bytes": baseline,
            "peak_device_bytes": peak_device, "peak_host_rss": peak_rss,
            "output_tail": output.strip()[-1000:]}


checks = []
for repeat in range(a.repeats):
    checks.append(run(f"resident-{repeat + 1}", [
        sys.executable, str(ROOT / "ref/pixal3d/validate_resident.py"),
        "--backend", "cuda"]))
for precision in ("mixed", "fp32"):
    checks.append(run(f"structure-{precision}", [
        sys.executable, str(ROOT / "ref/pixal3d/validate_flow_precision.py"),
        "--backend", "cuda", "--dump-dir", str(a.dump_dir),
        "--stage", "structure", "--precision", precision]))
checks.append(run("native-invalid-input-recovery", [
    sys.executable, str(ROOT / "ref/pixal3d/validate_api.py")]))
checks.append(run("web-queue-cancellation", [
    sys.executable, "-W", "error::ResourceWarning", "-m", "unittest",
    "server.pixal3d.test_app"]))
checks.append(run("multiview-budget-artifacts", [
    sys.executable, str(ROOT / "ref/pixal3d/validate_multiview_budgets.py"),
    *(str(path) for path in a.multiview_runs)]))

time.sleep(.5)
result = {"backend": "cuda", "repeats": a.repeats,
          "device_bytes_before": checks[0]["baseline_device_bytes"],
          "device_bytes_after": device_used(), "checks": checks}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(result, indent=2) + "\n")
nvml.nvmlShutdown()
print(json.dumps(result, indent=2))
print("Pixal3D CUDA reliability soak PASS")
