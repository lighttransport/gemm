"""The rig as a server job: run rig/build.py in the rig interpreter (numpy,
scipy, PyTorch) as a subprocess, relay its progress, honour cancellation.
The registration uses CUDA when present, under the shared device lock."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

from .. import gpu
from ..service import ROOT

DEFAULT_PYTHON = ROOT / "tmp/vhuman-rig-venv/bin/python"
RIG_FILES = ("rig.glb", "rig.json", "rig.usda", "rig_usd.zip", "preview.png", "rig_report.json", "features.json",
             "rig_basecolor.png", "rig_normal.png", "rig_orm.png", "rig_mouth.png")
MIN_FREE_MIB = 1536


def availability(python=None) -> dict:
    py = Path(python) if python else DEFAULT_PYTHON
    if not py.exists():
        return {"available": False, "reason": f"no rig interpreter ({py}); see server/vhuman/requirements-rig.txt"}
    return {"available": True, "python": str(py)}


def rig_job(service, request: dict, progress, cancel, python=None, mock: bool = False) -> dict:
    """{head_id, res (1024|2048|4096), iters}"""
    head_id = request.get("head_id")
    folder = service.head_file(head_id, "head.json").parent
    for name in ("head_eyes.glb", "fit.json", "portrait.png"):
        service.head_file(head_id, name)
    res = int(request.get("res", 2048))
    if res not in (1024, 2048, 4096):
        raise ValueError("res must be 1024, 2048 or 4096")
    iters = int(request.get("iters", 600))
    if not 50 <= iters <= 3000:
        raise ValueError("iters must be in [50, 3000]")
    py = Path(python) if python else DEFAULT_PYTHON
    if not py.exists():
        raise ValueError(availability(python)["reason"])
    out = folder / "rig"
    cmd = [str(py), "-m", "server.vhuman.rig.build", str(folder), "--out", str(out), "--res", str(res),
           "--iters", str(iters), "--cache", str(service.work / "cache" / "rig"), "--progress"]
    progress(0.01, "waiting for the GPU")
    lock = (service.work / "mock-gpu.lock") if mock else gpu.LOCK_PATH
    tail = []
    # PyTorch falls back to the CPU without a CUDA device; then only the lock is taken
    check = not mock and gpu.gpu_status() is not None
    with gpu.device_session(MIN_FREE_MIB, cancel, lock_path=lock, check_memory=check):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        stop = threading.Event()

        def watch():
            while not stop.wait(0.5):
                if cancel.is_set() and proc.poll() is None:
                    proc.terminate()
        threading.Thread(target=watch, daemon=True).start()
        try:
            for line in proc.stdout:
                line = line.rstrip()
                if line.startswith("@progress "):
                    _, f, msg = line.split(" ", 2)
                    progress(0.02 + 0.96 * float(f), msg)
                elif line:
                    tail = (tail + [line])[-30:]
            rc = proc.wait()
        finally:
            stop.set()
    if cancel.is_set():
        raise gpu.Cancelled("cancelled")
    if rc != 0:
        raise RuntimeError("rig build failed: " + " | ".join(tail[-6:]))
    report = json.loads((out / "rig_report.json").read_text())
    return {"id": head_id, "seconds": report.get("seconds"), "glb_url": f"/v1/heads/{head_id}/rig/rig.glb",
            "usd_url": f"/v1/heads/{head_id}/rig/rig_usd.zip"}


def main():                                 # pragma: no cover - convenience
    print(json.dumps(availability(sys.argv[1] if len(sys.argv) > 1 else None)))


if __name__ == "__main__":
    main()
