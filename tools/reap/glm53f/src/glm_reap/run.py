"""Sequential CUDA calibration and compression with durable status and logs."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import time
import traceback

from .common import load_config, atomic_json, fingerprint, configure_cpu


def run(config_path="configs/glm53f.json"):
    from .pipeline import calibrate, compress
    from .checkpoint import Checkpoint
    config = load_config(config_path)
    work = Path(config.get("work_dir", config["output"]))
    work.mkdir(parents=True, exist_ok=True)
    with (work / "run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(work / "run-config.json", config)
        options = configure_cpu(config)
        atomic_json(work / "execution-options.json", options)
        started = time.time()
        phase = "calibration"
        def status(state, **fields):
            atomic_json(work / "run-state.json", {"state": state, "phase": phase, "pid": os.getpid(), "started": started, "updated": time.time(), **fields})
        try:
            status("running")
            target = Path(config["output"]) / "glm53f-reap-gsq-rco.gguf"
            if target.exists():
                import hashlib
                import sys
                import numpy as np
                sys.path.insert(0, str(Path(config["llama_cpp"])/"gguf-py"))
                import gguf
                saved = json.loads(target.with_suffix(".manifest.json").read_text())
                if saved.get("config") != fingerprint(config) or saved.get("source") != fingerprint(Checkpoint(config["source"]).identity()):
                    raise ValueError("Existing final GGUF source/config changed")
                if target.stat().st_size != saved["file_bytes"]:
                    raise ValueError("Existing final GGUF size changed")
                reader = gguf.GGUFReader(target)
                hashes = {tensor.name: hashlib.sha256(memoryview(np.ascontiguousarray(tensor.data).view(np.uint8))).hexdigest() for tensor in reader.tensors}
                if hashes != saved["sha256_tensors"]:
                    raise ValueError("Existing final GGUF tensor hashes changed")
                phase = "compression"
                status("complete", gguf=str(target), verified_existing=True)
                return
            if options["memory_mode"] == "mmap":
                from .common import require_disk, GIB
                existing = sum(p.stat().st_blocks*512 for p in Path(options["candidate_dir"]).glob("*.bin"))
                estimate = Checkpoint(config["source"]).inventory(config["keep_experts"])["candidate_bank_estimate_gib"]*GIB
                require_disk(options["candidate_dir"], max(0, estimate-existing), config["reserve_disk_gib"]*GIB)
            manifest = work / "reap.json"
            if manifest.exists():
                existing = json.loads(manifest.read_text())
                if existing["config"] != fingerprint(config) or existing["source"] != fingerprint(Checkpoint(config["source"]).identity()):
                    raise ValueError("Existing REAP manifest does not match current source/config")
            else:
                calibrate(config, config["corpus"], "cuda")
            phase = "pruned_quality"
            status("running")
            from .quality import check
            quality_config = {**config, "quality_report": str(work / "quality-pruned.json")}
            check(quality_config, config["corpus"], "cuda", 256, 1, 45, "pruned", str(manifest))
            phase = "compression"
            status("running")
            target = compress(config, config["corpus"], "cuda")
            status("complete", gguf=target)
        except BaseException as error:
            status("failed", error=f"{type(error).__name__}: {error}")
            traceback.print_exc()
            raise
