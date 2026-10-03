from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

from .checkpoint import Checkpoint, expert_key, protected
from .common import atomic_json, load_config, existing_parent, configure_cuda, GIB


def preflight(config):
    import importlib.util
    import os
    c = Checkpoint(config["source"])
    from .native import FORMATS, nbytes
    minimum = 0
    for name in c.weight_names():
        shape = c.tensors[name].shape
        key = expert_key(name)
        if key and key[1] >= config["keep_experts"]:
            continue
        kinds = ["F16" if "kv_b_proj" in name or "conv1d" in name else "F32"] if protected(name, shape) else config["expert_types"] if key else config["dense_types"]
        kinds = [kind for kind in kinds if shape[-1] % FORMATS[kind][1] == 0]
        if ".mlp.gate." in name:
            shape = (config["keep_experts"], *shape[1:])
        minimum += min(nbytes(shape, kind) for kind in kinds)
    problems = []
    if minimum+32*2**20 > config["weight_budget_bytes"]:
        problems.append("Minimum native payload plus metadata reserve exceeds byte cap")
    if not Path(config["ggml_library"]).exists():
        problems.append("Native ggml library missing; run scripts/setup.sh")
    dependencies = {name: importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "datasets", "PIL", "safetensors", "scipy")}
    cuda = {}
    if dependencies["torch"]:
        import torch
        cuda = {"torch": torch.__version__, "toolkit": torch.version.cuda, "available": torch.cuda.is_available()}
        if cuda["available"]:
            free, total = torch.cuda.mem_get_info()
            cuda.update(free_bytes=free, total_bytes=total, device=torch.cuda.get_device_name())
        else:
            problems.append("CUDA unavailable; GPU pilot cannot run in this session")
    problems.extend("Missing Python dependency: "+name for name, present in dependencies.items() if not present)
    parent = existing_parent(config["output"])
    free = shutil.disk_usage(parent).free
    required = config["weight_budget_bytes"]+1127254016+(config["corpus_limit_gib"]+config["activation_budget_gib"]+config["reserve_disk_gib"])*GIB
    if free < required:
        problems.append(f"Insufficient disk for strict one-output pipeline: need {required/GIB:.2f} GiB")
    if not os.access(parent, os.W_OK):
        problems.append("Output parent is not writable")
    return {"inventory": c.inventory(config["keep_experts"]), "minimum_payload_bytes": minimum, "upgrade_budget_after_metadata_bytes": config["weight_budget_bytes"]-minimum-32*2**20, "source_fingerprint": __import__("glm_reap.common", fromlist=["fingerprint"]).fingerprint(c.identity()), "dependencies": dependencies, "cuda": cuda, "output_free_gib": free/GIB, "required_disk_gib": required/GIB, "problems": problems}


def pilot(config, device, tokens, layers):
    import resource
    import torch
    from .runtime import Runtime
    from .native import NativeCodec
    from .gsq import refine
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inaccessible; run this pilot directly on the host")
    configure_cuda(config, device)
    from .common import configure_cpu
    configure_cpu(config)
    start = time.monotonic()
    runtime = Runtime(Checkpoint(config["source"]), config, device=device)
    runtime.layers = runtime.layers[:layers]
    ids = list(range(1000, 1000+tokens))
    with torch.no_grad():
        hidden = runtime.hidden(ids)
    elapsed = time.monotonic()-start
    codec = NativeCodec(config["ggml_library"])
    rng = __import__("numpy").random.default_rng(42)
    w = rng.normal(size=(8, 256)).astype("float32")
    x = rng.normal(size=(32, 256)).astype("float32")
    _, metrics = refine(codec, w, "Q2_K", x[:24], x[24:], {**config, "gsq_epochs": 2}, "pilot", device)
    return {"layers": layers, "tokens": tokens, "elapsed_seconds": elapsed, "partial_tokens_per_second": tokens/elapsed, "host_peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/GIB, "cuda_peak_gib": torch.cuda.max_memory_allocated()/GIB if torch.cuda.is_available() else None, "finite_hidden": bool(torch.isfinite(hidden).all()), "native_gsq": metrics, "acceptance": "partial pilot only; full calibration and 128K deployment are unverified"}


def main():
    parser = argparse.ArgumentParser(description="GLM-5.3-Flash local REAP/native GSQ/task RCO tools")
    parser.add_argument("--config", default="configs/glm53f.json")
    parser.add_argument("--output", help="Override artifact output (does not modify source)")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("preflight")
    check.add_argument("--report")
    corpus = sub.add_parser("download-corpus")
    corpus.add_argument("--destination")
    quality = sub.add_parser("quality-check")
    quality.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    quality.add_argument("--corpus")
    quality.add_argument("--tokens", type=int, default=256)
    quality.add_argument("--windows", type=int, default=1)
    quality.add_argument("--layers", type=int, default=45)
    quality.add_argument("--mode", choices=("source", "pruned", "native"), default="native")
    quality.add_argument("--reap")
    quality.add_argument("--report")
    for name in ("pilot", "calibrate", "compress"):
        p = sub.add_parser(name)
        p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
        if name == "pilot":
            p.add_argument("--tokens", type=int, default=16)
            p.add_argument("--layers", type=int, default=1)
        else:
            p.add_argument("--corpus", default="artifacts/corpus")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.output:
        config["output"] = str(Path(args.output).resolve())
    if args.command == "preflight":
        result = preflight(config)
        if args.report:
            atomic_json(args.report, result)
    elif args.command == "download-corpus":
        from .corpus import download
        destination = args.destination or config["corpus"]
        download(config, destination)
        result = {"corpus": str(Path(destination).resolve())}
    elif args.command == "quality-check":
        from .quality import check
        if args.report:
            config["quality_report"] = str(Path(args.report).resolve())
        result = check(config, args.corpus or config["corpus"], args.device, args.tokens, args.windows, args.layers, args.mode, args.reap)
    elif args.command == "pilot":
        result = pilot(config, args.device, args.tokens, args.layers)
        atomic_json(Path(config["output"])/"pilot.json", result)
    elif args.command == "calibrate":
        from .pipeline import calibrate
        result = calibrate(config, args.corpus, args.device)
    else:
        from .pipeline import compress
        result = {"gguf": compress(config, args.corpus, args.device)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
