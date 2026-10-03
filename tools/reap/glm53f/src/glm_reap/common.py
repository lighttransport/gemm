from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import time

GIB = 2**30


def execution_options(config):
    import psutil
    physical = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    def option(key, default, cast):
        return cast(os.environ.get("REAP_"+key.upper(), config.get(key, default)))
    result = {
        "memory_mode": option("memory_mode", "ram", str),
        "cpu_threads": option("cpu_threads", max(1, physical//2), int),
        "source_cache_gib": option("source_cache_gib", 112, float),
        "calibration_batch_windows": option("calibration_batch_windows", 8, int),
        "candidate_dir": option("candidate_dir", str(Path(config.get("work_dir", config["output"]))/"candidates"), str),
    }
    if result["memory_mode"] not in ("ram", "mmap"):
        raise ValueError("memory_mode must be ram or mmap")
    if result["cpu_threads"] < 1 or result["calibration_batch_windows"] < 1 or result["source_cache_gib"] < 0:
        raise ValueError("Invalid CPU/cache/batch setting")
    return result


def configure_cpu(config):
    options = execution_options(config)
    threads = options["cpu_threads"]
    import torch
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != threads:
        try:
            torch.set_num_interop_threads(threads)
        except RuntimeError:
            pass  # PyTorch permits this setting only before parallel work starts.
    # This also controls already-loaded NumPy/OpenBLAS pools.
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=threads)
    return options


def load_config(path):
    config = json.loads(Path(path).read_text())
    root = Path(__file__).resolve().parents[2]
    for key in ("source", "output", "llama_cpp", "ggml_library", "corpus", "quality_report", "work_dir"):
        if key not in config:
            continue
        p = Path(config[key]).expanduser()
        config[key] = str(p if p.is_absolute() else root / p)
    if not 8 <= config["keep_experts"] <= 288:
        raise ValueError("keep_experts must be between top-k=8 and 288")
    if abs(sum(config["corpus_mix"].values()) - 1) > 1e-9:
        raise ValueError("corpus_mix must sum to one")
    return config


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def existing_parent(path):
    path = Path(path)
    while not path.exists():
        path = path.parent
    return path


def require_disk(path, extra, reserve=4 * GIB):
    free = shutil.disk_usage(existing_parent(path)).free
    if free < extra + reserve:
        raise RuntimeError(f"Need {extra / GIB:.2f} GiB plus {reserve / GIB:.2f} GiB reserve; only {free / GIB:.2f} GiB free at {path}")


def memory_guard(config):
    import resource
    import torch
    # VmRSS is current resident memory; ru_maxrss is a historical peak.
    rss = 0
    if Path("/proc/self/status").exists():
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) * 1024
    else:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if rss > config["host_limit_gib"] * GIB:
        raise MemoryError(f"Host RSS {rss / GIB:.2f} GiB exceeds configured limit")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        used = torch.cuda.memory_allocated()
        if used > min(config["gpu_limit_gib"] * GIB, total * 0.85):
            raise MemoryError(f"CUDA allocation {used / GIB:.2f} GiB exceeds configured limit")


def configure_cuda(config, device):
    """Apply the local GPU allocator cap before allocating model tensors."""
    if not str(device).startswith("cuda"):
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is not exposed to this process")
    free, total = torch.cuda.mem_get_info()
    limit = min(config["gpu_limit_gib"]*GIB, total*.85, free-512*2**20)
    if limit <= 0:
        raise MemoryError("No GPU headroom after reserving 512 MiB")
    torch.cuda.set_per_process_memory_fraction(limit/total)


def log(event, **fields):
    print(json.dumps({"time": time.time(), "event": event, **fields}, allow_nan=False), flush=True)
