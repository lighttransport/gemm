"""GPU discovery, shared backend/device locks, and a free-memory gate.

The flock only orders jobs; it does not free memory. The Pixal3D server's
studio can keep Qwen-Image resident for minutes after its lock is
released, so every GPU session here also checks free device memory and
refuses with a clear message instead of running into an out-of-memory
failure. Same lock file as server/pixal3d/app.py (cancellable_file_lock).
"""
from __future__ import annotations

import fcntl
import contextvars
import json
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = ROOT / "tmp/pixal3d/device-locks/cuda-0.lock"
QWEN_MIN_FREE_MIB = 12288      # Qwen-Image 2.1 fast12 at 1024^2: measured peak ~14.6 GB (study run)
PIXAL3D_MIN_FREE_MIB = 7680    # Pixal3D native, 7168 MiB budget + headroom
_CONFIG = contextvars.ContextVar("vhuman_gpu", default=("auto", 0, Path("/mnt/disk1/models")))


def configure(backend="auto", device=0, models_root="/mnt/disk1/models"):
    if backend not in ("auto", "cpu", "cuda", "rocm"):
        raise ValueError("backend must be auto, cpu, cuda or rocm")
    if isinstance(device, bool) or not isinstance(device, int) or device < 0:
        raise ValueError("device must be a non-negative integer")
    return _CONFIG.set((backend, device, Path(models_root).resolve()))


def model_path(relative):
    return _CONFIG.get()[2] / relative


def device_index():
    return _CONFIG.get()[1]


def backend():
    requested = _CONFIG.get()[0]
    if requested != "auto":
        return requested
    status = gpu_status()
    return status.get("backend", "cuda") if status else "cpu"


@contextmanager
def execution(selected="auto", device=0, models_root="/mnt/disk1/models"):
    token = configure(selected, device, models_root)
    try:
        resolved = backend()
        configure(resolved, device, models_root)
        yield resolved
    finally:
        _CONFIG.reset(token)


class GpuBusy(RuntimeError):
    pass


class Cancelled(RuntimeError):
    pass


def gpu_status(device: int | None = None, selected: str | None = None) -> dict | None:
    """Free/total device memory (MiB) and name, from nvidia-smi or a native HIP probe."""
    device = device_index() if device is None else device
    selected = selected or _CONFIG.get()[0]
    if selected == "cpu":
        return None
    if selected in ("auto", "rocm"):
        if selected == "auto":
            cuda = gpu_status(device, "cuda")
            if cuda:
                return cuda
        try:
            result = subprocess.run([str(ROOT / "rdna4/rocew_probe"), "--json", "--device", str(device)],
                                    capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return json.loads(result.stdout)
        except (OSError, subprocess.TimeoutExpired, ValueError):
            pass
        return None
    try:
        out = subprocess.run(["nvidia-smi", f"--id={device}", "--query-gpu=name,memory.free,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    name, free, total = [s.strip() for s in out.stdout.strip().splitlines()[0].split(",")]
    return {"backend": "cuda", "device": device, "name": name,
            "free_mib": int(float(free)), "total_mib": int(float(total))}


@contextmanager
def file_lock(path: Path, timeout: float, cancel: threading.Event | None = None):
    """Process-shared advisory lock with bounded, cancellable waiting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    deadline = time.monotonic() + timeout
    try:
        while True:
            if cancel is not None and cancel.is_set():
                raise Cancelled("cancelled while waiting for the GPU")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise GpuBusy(f"the GPU device stayed locked for {timeout:g}s") from None
                time.sleep(0.1)
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


@contextmanager
def device_session(min_free_mib: int, cancel: threading.Event | None = None, timeout: float = 900.0,
                   lock_path: Path = LOCK_PATH, check_memory: bool = True):
    """Hold the shared CUDA lock and require `min_free_mib` free."""
    if lock_path == LOCK_PATH:
        lock_path = LOCK_PATH.parent / f"{backend()}-{device_index()}.lock"
    with file_lock(lock_path, timeout, cancel):
        if check_memory:
            status = gpu_status()
            if status is None:
                raise GpuBusy(f"no {backend()} device; build rdna4/rocew_probe for AMD detection")
            if status["free_mib"] < min_free_mib:
                raise GpuBusy(f"only {status['free_mib']} MiB of GPU memory free, {min_free_mib} MiB needed; "
                              "another process (e.g. the Pixal3D demo's resident Qwen model) holds it")
        yield
