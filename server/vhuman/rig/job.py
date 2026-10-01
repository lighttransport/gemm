"""The rig as a server job: run rig/build.py in the rig interpreter (numpy,
scipy, PyTorch) as a subprocess, relay its progress, honour cancellation.
Registration uses the configured CPU/CUDA/ROCm device under the shared lock."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import shutil
import uuid
from pathlib import Path

from .. import gpu
from ..service import ROOT

DEFAULT_PYTHON = ROOT / "tmp/vhuman-rig-venv/bin/python"
RIG_FILES = ("rig.glb", "rig.json", "rig.usda", "rig_usd.zip", "preview.png", "rig_report.json", "features.json",
             "rig_basecolor.png", "rig_normal.png", "rig_orm.png", "rig_mouth.png", "deformer.lrm",
             "deformer_basis.safetensors", "deformer.json", "rig_deformer.safetensors", "viz.json", "wm_brow_up.png", "wm_brow_down.png",
             "wm_smile.png", "wm_mouth.png", "expressions/manifest.json", "rig_lod1.glb", "rig_lod2.glb",
             "rig_lod1.usda", "rig_lod2.usda", "rig_deformer_lod1.safetensors", "rig_deformer_lod2.safetensors",
             "viz_lod1.json", "viz_lod2.json")
RIG_FILES += ("soft_deformer.safetensors", "soft_deformer_lod1.safetensors",
              "soft_deformer_lod2.safetensors", "soft_deformer_report.json")
RIG_FILES += ("skin_material.json", "rig_coverage.png", "rig_confidence.png", "rig_specular.png")
MIN_FREE_MIB = 1536


def availability(python=None) -> dict:
    py = Path(python) if python else DEFAULT_PYTHON
    if not py.exists():
        return {"available": False, "reason": f"no rig interpreter ({py}); see server/vhuman/requirements-rig.txt"}
    return {"available": True, "python": str(py)}


def rig_job(service, request: dict, progress, cancel, python=None, mock: bool = False) -> dict:
    """{head_id, res (1024|2048|4096), iters, face_model?}."""
    from .face_models import SOURCES
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
    candidate = request.get("reconstruction_run")
    if candidate:
        manifest = service.reconstruction_file(head_id, candidate, "manifest.json")
        out = manifest.parent / "rig"
    old_report = out / "rig_report.json"
    saved_model = json.loads(manifest.read_text()).get("face_model") if candidate else None
    if old_report.exists():
        saved_model = json.loads(old_report.read_text()).get("face_model", "procedural")
    face_model = request.get("face_model") or saved_model or "gnm_v3"
    if face_model not in SOURCES:
        raise ValueError(f"face_model must be one of {', '.join(SOURCES)}")
    build_out = out.parent / (".rig-" + uuid.uuid4().hex + ".partial") if candidate else out
    cmd = [str(py), "-m", "server.vhuman.rig.build", str(folder), "--out", str(build_out), "--res", str(res),
           "--iters", str(iters), "--cache", str(service.work / "cache" / "rig"),
           "--face-model", face_model, "--progress"]
    if candidate:
        cmd.extend(["--reconstruction", str(manifest.parent)])
    try:
        progress(0.01, "waiting for the GPU")
        lock = (service.work / "mock-gpu.lock") if mock else gpu.LOCK_PATH
        tail = []
        # Auto selection uses CPU when no GPU is detected; explicit GPU requests stay strict.
        check = not mock and gpu.gpu_status() is not None
        with gpu.device_session(MIN_FREE_MIB, cancel, lock_path=lock, check_memory=check):
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            from ..runtime import python_command
            proc = subprocess.Popen(python_command(cmd), cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
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
        report = json.loads((build_out / "rig_report.json").read_text())
        if candidate:
            backup = out.parent / (".rig-" + uuid.uuid4().hex + ".backup")
            if out.exists():
                out.replace(backup)
            try:
                build_out.replace(out)
            except BaseException:
                if backup.exists():
                    backup.replace(out)
                raise
            shutil.rmtree(backup, ignore_errors=True)
    except BaseException:
        if candidate:
            shutil.rmtree(build_out, ignore_errors=True)
        raise
    base = f"/v1/heads/{head_id}/" + (f"reconstruction/{candidate}/rig/" if candidate else "rig/")
    return {"id": head_id, "seconds": report.get("seconds"), "glb_url": base + "rig.glb",
            "usd_url": base + "rig_usd.zip"}


def main():                                 # pragma: no cover - convenience
    print(json.dumps(availability(sys.argv[1] if len(sys.argv) > 1 else None)))


if __name__ == "__main__":
    main()
