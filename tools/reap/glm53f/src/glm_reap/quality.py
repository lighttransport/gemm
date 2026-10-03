from __future__ import annotations

import json
from pathlib import Path
import time
import numpy as np

from .checkpoint import Checkpoint, expert_key, protected
from .common import atomic_json, memory_guard, configure_cuda, log, GIB
from .native import NativeCodec, decode


def difference(reference, candidate):
    import torch
    a, b = reference.float().flatten(), candidate.float().flatten()
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise ValueError("Non-finite layer output in quality check")
    mse = (a-b).square().mean()
    return {"mse": float(mse.cpu()), "relative_mse": float((mse/a.square().mean().clamp_min(1e-20)).cpu()), "cosine": float(torch.nn.functional.cosine_similarity(a[None], b[None]).cpu())}


def check(config, corpus, device="cuda", length=256, limit=1, layers=45, mode="native", reap=None):
    """Two hidden streams, bounded-row weights, no candidate bank or GGUF writes."""
    import torch
    from .runtime import Runtime, StreamLinear
    from .corpus import windows
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device is not exposed to this process; use a GPU-enabled host session")
    configure_cuda(config, device)
    checkpoint = Checkpoint(config["source"])
    selected = json.loads(Path(reap).read_text())["selected"] if reap else None
    if mode == "pruned" and selected is None:
        raise ValueError("Pruned quality check requires --reap with calibrated expert IDs")
    reference = Runtime(checkpoint, config, device=device)
    candidate = Runtime(checkpoint, config, selected=selected, device=device)
    if mode == "native":
        codec = NativeCodec(config["ggml_library"])
        original_read = candidate.weight
        def quantized_read(name, start, stop, choice=None):
            value = original_read(name, start, stop, choice)
            if protected(name, value.shape):
                return value
            kind = config["expert_types"][0] if expert_key(name) else config["dense_types"][0]
            if value.shape[-1] % 256:
                kind = "Q8_0"
            return decode(codec.encode(value, kind), kind, value.shape)
        candidate.weight = quantized_read
    count = min(layers, len(reference.layers))
    if count < 1:
        raise ValueError("Layer count must be positive")
    from .common import configure_cpu
    configure_cpu(config)
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    report = {"mode": mode, "quantizer": "native RTN baseline; not GSQ/RCO" if mode == "native" else None, "device": str(device), "torch_cuda": torch.version.cuda, "layers": count, "full_model": count == len(reference.layers), "corpus": str(Path(corpus).resolve()), "reap": reap, "windows": []}
    start_time = time.monotonic()
    with torch.no_grad():
        for row in windows(corpus, length, split="heldout", limit=limit, config=config):
            ids, answer_mask = row[:2]
            ids_tensor = torch.tensor(ids, device=reference.device)[None]
            h_ref = reference.embedding(ids_tensor)
            h_cmp = candidate.embedding(ids_tensor)
            if len(row) > 2:
                h_ref = reference._vision(h_ref, ids_tensor, row[2])
                # Both share the identical retained vision tower.
                candidate.visual = reference.visual
                h_cmp = candidate._vision(h_cmp, ids_tensor, row[2])
            h_ref = h_ref.unsqueeze(2).expand(-1, -1, reference.text_config.hc_mult, -1).contiguous()
            h_cmp = h_cmp.unsqueeze(2).expand_as(h_ref).contiguous()
            mask = torch.ones(ids_tensor.shape, device=reference.device, dtype=torch.bool)
            positions = torch.arange(len(ids), device=reference.device)[None]
            index_ref, index_cmp = None, None
            metrics = {"tokens": len(ids), "layers": []}
            for layer in range(count):
                h_ref, index_ref = reference.layers[layer](h_ref, attention_mask=mask, position_ids=positions, prev_topk_indices=index_ref, use_cache=False, chunk_size=config.get("kda_chunk_size", 16))
                h_cmp, index_cmp = candidate.layers[layer](h_cmp, attention_mask=mask, position_ids=positions, prev_topk_indices=index_cmp, use_cache=False, chunk_size=config.get("kda_chunk_size", 16))
                metrics["layers"].append({"layer": layer, **difference(h_ref, h_cmp)})
                memory_guard(config)
                log("quality_layer", window=len(report["windows"]), **metrics["layers"][-1])
            if report["full_model"]:
                valid = torch.tensor(answer_mask[1:], device=reference.device, dtype=torch.bool)
                labels = ids_tensor[0, 1:][valid]
                for name, runtime, hidden in (("reference", reference, h_ref), ("candidate", candidate, h_cmp)):
                    x = runtime.norm(hidden.mean(2))[0, :-1][valid]
                    # Chunk answer positions as well as weight rows to bound logits.
                    total = 0.0
                    for offset in range(0, len(x), 32):
                        logits = StreamLinear(runtime, "lm_head.weight", runtime.text_config.vocab_size)(x[offset:offset+32]).float()
                        total += float(torch.nn.functional.cross_entropy(logits, labels[offset:offset+32], reduction="sum").cpu())
                    metrics[name+"_answer_nll"] = total/len(labels)
                metrics["answer_tokens"] = len(labels)
            report["windows"].append(metrics)
            report["elapsed_seconds"] = time.monotonic()-start_time
            report["cuda_peak_gib"] = torch.cuda.max_memory_allocated()/GIB if str(device).startswith("cuda") else None
            atomic_json(config["quality_report"], report)
    if not report["windows"]:
        raise ValueError("No usable heldout windows; download corpus first")
    return report


def selected_quality(config, checkpoint, bank, selected):
    """Evaluate final allocated bytes against source with bounded CUDA reads."""
    import torch
    from .runtime import Runtime
    from .corpus import windows
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reference = Runtime(checkpoint, config, device=device)
    candidate = Runtime(checkpoint, config, selected, bank, device)
    report = {"device": device, "quantizer": "final selected GSQ/RCO native bytes", "windows": [], "acceptance": "heldout answer-loss check; coding benchmarks and V100 128K deployment remain unverified"}
    started = time.monotonic()
    with torch.no_grad():
        for row in windows(config["corpus"], 256, split="heldout", limit=2, config=config):
            vision = row[2] if len(row) > 2 else None
            source_loss = reference.loss(row[0], row[1], vision=vision)
            final_loss = candidate.loss(row[0], row[1], vision=vision)
            if not bool(torch.isfinite(source_loss) and torch.isfinite(final_loss)):
                raise ValueError("Non-finite final allocated model answer loss")
            report["windows"].append({"tokens": len(row[0]), "source_answer_nll": float(source_loss.cpu()), "final_answer_nll": float(final_loss.cpu()), "nll_delta": float((final_loss-source_loss).cpu())})
            log("final_quality_window", **report["windows"][-1])
    if not report["windows"]:
        raise ValueError("No heldout examples for final allocated quality check")
    report["elapsed_seconds"] = time.monotonic()-started
    report["cuda_peak_gib"] = torch.cuda.max_memory_allocated()/GIB if device == "cuda" else None
    work = Path(config.get("work_dir", config["output"]))
    atomic_json(work / "quality-final.json", report)
    return report
