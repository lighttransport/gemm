from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from .bank import Bank, Candidate
from .checkpoint import Checkpoint, expert_key, group_name, protected
from .common import atomic_json, fingerprint, GIB, log, memory_guard, configure_cuda, require_disk
from .corpus import windows
from .native import NativeCodec, nbytes, FORMATS
from .reap import select


class Reservoirs:
    def __init__(self, capacity, seed, byte_cap):
        self.capacity, self.byte_cap = capacity, byte_cap
        self.rng = np.random.default_rng(seed)
        self.values, self.keys = {}, {}

    def capture(self, group, tensor):
        x = tensor.detach().float().reshape(-1, tensor.shape[-1]).cpu().numpy()
        keys = self.rng.random(len(x))
        if group in self.values:
            x = np.concatenate((self.values[group], x))
            keys = np.concatenate((self.keys[group], keys))
        selected = np.argsort(keys)[-self.capacity:]
        self.values[group], self.keys[group] = x[selected].copy(), keys[selected]
        if sum(v.nbytes for v in self.values.values()) > self.byte_cap:
            raise MemoryError("Activation reservoir byte cap exceeded")

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        names = {}
        for group, value in self.values.items():
            name = fingerprint(group)+".npy"
            np.save(directory / name, value, allow_pickle=False)
            names[group] = name
        atomic_json(directory / "index.json", names)


def calibrate(config, corpus, device):
    import torch
    from .runtime import Runtime
    configure_cuda(config, device)
    from .common import configure_cpu
    configure_cpu(config)
    checkpoint = Checkpoint(config["source"])
    output = Path(config.get("work_dir", config["output"]))
    require_disk(output, config["activation_budget_gib"]*GIB, config["reserve_disk_gib"]*GIB)
    runtime = Runtime(checkpoint, config, device=device)
    runtime.begin_saliency()
    source_id = fingerprint(checkpoint.identity())
    config_id = fingerprint(config)
    corpus_id = fingerprint([(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(Path(corpus).glob("*.jsonl"))])
    identity = fingerprint([source_id, config_id, corpus_id])
    progress_path = output / "calibration-progress.json"
    count, previous_elapsed = 0, 0.0
    if progress_path.exists():
        state = json.loads(progress_path.read_text())
        if state.get("identity") != identity:
            raise ValueError("Calibration checkpoint source/config/corpus changed")
        count, previous_elapsed = state["completed_windows"], state["elapsed_seconds"]
        for layer, score in state["scores"].items():
            accumulator = runtime.saliency[int(layer)]
            accumulator.count[:] = score["count"]
            accumulator.weighted_norm[:] = score["weighted_norm_sum"]
        log("calibration_resume", completed=count)
    start = time.monotonic()
    def source_layer(layer):
        atomic_json(output / "current-layer.json", {"phase": "source_calibration", "window": count+1, "window_end": min(count+batch_size, config["calibration_windows"]), "layer": layer, "source_cache_gib": getattr(checkpoint, "cache_bytes", 0)/GIB, "elapsed_seconds": previous_elapsed+time.monotonic()-start})
        log("calibration_layer", window=count+1, layer=layer)
    runtime.layer_progress = source_layer
    scores = {str(layer): score.result() for layer, score in runtime.saliency.items()}
    from .common import execution_options
    from itertools import islice
    options = execution_options(config)
    batch_size = options["calibration_batch_windows"] if options["memory_mode"] == "ram" and hasattr(runtime, "hidden_batch") else 1
    records = iter(windows(corpus, config["sequence_length"], limit=config["calibration_windows"], config=config))
    for _ in range(count):
        next(records, None)
    with torch.no_grad():
        while batch := list(islice(records, batch_size)):
            if batch_size > 1:
                runtime.hidden_batch(batch)
            else:
                record = batch[0]
                runtime.hidden(record[0], vision=record[2] if len(record) > 2 else None)
            count += len(batch)
            scores = {str(layer): score.result() for layer, score in runtime.saliency.items()}
            atomic_json(progress_path, {"identity": identity, "completed_windows": count, "source": source_id, "scores": scores, "elapsed_seconds": previous_elapsed+time.monotonic()-start})
            log("calibration_window", completed=count, target=config["calibration_windows"])
    if count < config["calibration_windows"]:
        raise ValueError(f"Only {count} usable windows; need {config['calibration_windows']}")
    selected = {layer: select(result["scores"], config["keep_experts"]) for layer, result in scores.items()}
    del runtime
    import gc
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    # GSQ inputs must reflect the pruned routers and residual distributions.
    runtime = Runtime(checkpoint, config, selected, device=device)
    reservoirs = Reservoirs(config["activation_reservoir"], config["seed"], config["activation_budget_gib"]*GIB)
    runtime.capture = reservoirs.capture
    activation_path = output / "activation-state.npz"
    completed = 0
    if activation_path.exists():
        with np.load(activation_path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"]))
            if metadata["identity"] != identity:
                raise ValueError("Activation checkpoint source/config/corpus changed")
            completed = metadata["completed"]
            reservoirs.rng.bit_generator.state = metadata["rng"]
            for index, group in enumerate(metadata["groups"]):
                reservoirs.values[group] = saved[f"v{index}"].copy()
                reservoirs.keys[group] = saved[f"k{index}"].copy()
        log("activation_resume", completed=completed)
    def activation_layer(layer):
        atomic_json(output / "current-layer.json", {"phase": "activation_calibration", "window": completed+1, "layer": layer})
        log("activation_layer", window=completed+1, layer=layer)
    runtime.layer_progress = activation_layer
    with torch.no_grad():
        for index, record in enumerate(windows(corpus, config["sequence_length"], limit=config["quant_windows"], config=config)):
            if index < completed:
                continue
            hidden = runtime.hidden(record[0], vision=record[2] if len(record) > 2 else None)
            reservoirs.capture("lm_head.weight", hidden)
            completed += 1
            if completed % 8 == 0 or completed == config["quant_windows"]:
                groups = list(reservoirs.values)
                arrays = {f"v{i}": reservoirs.values[g] for i, g in enumerate(groups)}
                arrays.update({f"k{i}": reservoirs.keys[g] for i, g in enumerate(groups)})
                arrays["metadata"] = np.array(json.dumps({"identity": identity, "completed": completed, "groups": groups, "rng": reservoirs.rng.bit_generator.state}))
                temporary = activation_path.with_suffix(".npz.tmp")
                require_disk(output, sum(a.nbytes for a in arrays.values()), config["reserve_disk_gib"]*GIB)
                with temporary.open("wb") as stream:
                    np.savez(stream, **arrays)
                    import os
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(activation_path)
            log("activation_window", completed=completed, target=config["quant_windows"])
    if completed < config["quant_windows"]:
        raise ValueError(f"Only {completed} usable activation windows")
    reservoirs.save(output / "activations")
    manifest = {"source": fingerprint(checkpoint.identity()), "config": fingerprint(config), "selected": selected, "scores": scores, "completed_windows": count}
    atomic_json(output / "reap.json", manifest)
    if hasattr(checkpoint, "set_cache_limit"):
        checkpoint.set_cache_limit(0)
    del runtime
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return manifest


def candidates(config, checkpoint, selected, device):
    from .gsq import refine
    from .resume import TileStore
    root = Path(config.get("work_dir", config["output"]))
    index = json.loads((root / "activations/index.json").read_text())
    codec = NativeCodec(config["ggml_library"])
    activation_identity = [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in sorted((root / "activations").glob("*"))]
    import hashlib
    codec_hash = hashlib.sha256(Path(config["ggml_library"]).read_bytes()).hexdigest()
    identity = fingerprint([checkpoint.identity(), config, selected, activation_identity, codec_hash, Path(__file__).with_name("gsq.py").read_text()])
    import os
    store = TileStore(root / "gsq-tiles.sqlite", identity, float(os.environ.get("GSQ_CHECKPOINT_GIB", config.get("gsq_checkpoint_gib", 8)))*GIB, config["reserve_disk_gib"]*GIB)
    from .common import execution_options
    options = execution_options(config)
    bank = Bank(options["memory_mode"], options["candidate_dir"], config["reserve_disk_gib"]*GIB)
    checkpoint.configure(config)
    if options["memory_mode"] == "mmap":
        existing = sum(p.stat().st_blocks*512 for p in Path(options["candidate_dir"]).glob("*.bin"))
        require_disk(options["candidate_dir"], max(0, checkpoint.inventory(config["keep_experts"])["candidate_bank_estimate_gib"]*GIB-existing), config["reserve_disk_gib"]*GIB)
    report = {}
    previous_layer = None
    for name in checkpoint.weight_names():
        checkpoint.configure(config, bank.bytes)
        import re
        match = re.search(r"\.layers\.(\d+)\.", name)
        layer = int(match[1]) if match else None
        if layer is not None and layer != previous_layer:
            checkpoint.prefetch_layer(layer, selected)
        previous_layer = layer
        info = checkpoint.tensors[name]
        key = expert_key(name)
        if key and key[1] not in selected[str(key[0])]:
            continue
        protected_tensor = protected(name, info.shape)
        types = ["F16" if "kv_b_proj" in name or "conv1d" in name else "F32"] if protected_tensor else config["expert_types"] if key else config["dense_types"]
        # KDA f_b projections have 128 input columns: K blocks require 256.
        types = [kind for kind in types if info.shape[-1] % FORMATS[kind][1] == 0]
        if not types:
            raise ValueError(f"No legal native format for {name}: {info.shape}")
        shape = info.shape
        rows = None
        if ".mlp.gate." in name:
            layer = name.split(".layers.")[1].split(".")[0]
            rows = selected[layer]
            shape = (len(rows), *shape[1:])
        if protected_tensor:
            value = checkpoint.read(name)
            if rows is not None:
                value = value[rows]
            bank.add(name, [Candidate(types[0], shape, codec.encode(value, types[0]))])
            continue
        group = group_name(name)
        if group in index:
            inputs = np.load(root / "activations" / index[group], mmap_mode="r", allow_pickle=False)
        elif name == "model.language_model.embed_tokens.weight":
            # Isotropic activation objective for embedding rows; task RCO follows.
            rng = np.random.default_rng(config["seed"])
            inputs = rng.standard_normal((128, shape[-1]), dtype=np.float32)
        else:
            raise ValueError(f"No calibration inputs for {name}; cannot silently substitute RTN")
        split = max(1, int(len(inputs)*0.875))
        if len(inputs)-split < 1:
            raise ValueError(f"Insufficient train/validation activation samples: {group}")
        entries = []
        for kind in types:
            raw = bank.allocate(name, kind, nbytes(shape, kind))
            row_bytes = len(raw)//shape[0]
            tile_reports = []
            for start in range(0, shape[0], config["tile_rows"]):
                stop = min(start+config["tile_rows"], shape[0])
                value = checkpoint.read(name, start, stop)
                tile_key = f"{name}:{kind}:{start}"
                baseline = codec.encode(value, kind, np.mean(np.square(inputs[:split]), axis=0))
                saved = store.restore(tile_key, baseline)
                if saved is None:
                    packed, metrics = refine(codec, value, kind, inputs[:split], inputs[split:], config, tile_key, device)
                    store.save(tile_key, baseline, packed, metrics)
                else:
                    packed, metrics = saved
                raw[start*row_bytes:stop*row_bytes] = packed
                tile_reports.append(metrics)
                memory_guard(config)
            entries.append(Candidate(kind, shape, raw))
            report[f"{name}:{kind}"] = {"tiles": len(tile_reports), "improved_tiles": sum(m["improved"] for m in tile_reports), "validation_mse": float(np.mean([m["selected_mse"] for m in tile_reports]))}
        bank.add(name, entries)
        log("candidate_tensor", name=name, bank_gib=bank.bytes/GIB)
        atomic_json(root / "gsq-progress.json", report)
        memory_guard(config)
    store.close()
    return bank, report


def compress(config, corpus, device):
    import torch
    from .runtime import Runtime
    from .rco import optimize
    from .export import write
    configure_cuda(config, device)
    from .common import configure_cpu
    configure_cpu(config)
    checkpoint = Checkpoint(config["source"])
    manifest = json.loads((Path(config.get("work_dir", config["output"]))/"reap.json").read_text())
    if manifest["source"] != fingerprint(checkpoint.identity()) or manifest["config"] != fingerprint(config):
        raise ValueError("Source/config changed since REAP calibration")
    selected = manifest["selected"]
    bank, report = candidates(config, checkpoint, selected, device)
    runtime = Runtime(checkpoint, config, selected, bank, device)
    batches = windows(corpus, config["sequence_length"], limit=config["quant_windows"], config=config)
    # Offload saved attention activations. Lazy linears save their own CPU inputs.
    with torch.autograd.graph.saved_tensors_hooks(lambda x: (x.device, x.cpu()), lambda saved: saved[1].to(saved[0])):
        rco = optimize(bank, runtime, batches, config, overhead=32*2**20)
    atomic_json(Path(config.get("work_dir", config["output"]))/"rco.json", rco)
    target = write(config, checkpoint, bank, selected)
    log("export_complete", path=target)
    return target
