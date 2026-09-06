#!/usr/bin/env python3
import argparse
import concurrent.futures
import csv
import heapq
import json
import math
import os
import re
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from batch_quant_report import SafeFile, load_sample, metrics, qsym, quant_block, quant_fp4_block, f32_to_f16


NF4 = (
    -1.0, -0.6961928, -0.5250731, -0.3949175,
    -0.2844414, -0.1847734, -0.0910500, 0.0,
    0.0795803, 0.1609302, 0.2461123, 0.3379152,
    0.4407098, 0.5626170, 0.7229568, 1.0,
)


def qsym_block_clip(w, rows, cols, bits, block, clip_ratio):
    denom = (1 << (bits - 1)) - 1
    out = [0.0] * len(w)
    for rb in range(0, rows, block):
        re = min(rows, rb + block)
        for cb in range(0, cols, block):
            ce = min(cols, cb + block)
            mx = 0.0
            for r in range(rb, re):
                off = r * cols
                for c in range(cb, ce):
                    mx = max(mx, abs(w[off + c]))
            clip = mx * clip_ratio
            s = clip / denom if clip else 1.0
            for r in range(rb, re):
                off = r * cols
                for c in range(cb, ce):
                    x = max(-clip, min(clip, w[off + c]))
                    out[off + c] = qsym(x, denom, s) * s
    return out


def qsym_tensor(w, bits):
    denom = (1 << (bits - 1)) - 1
    mx = max((abs(x) for x in w), default=0.0)
    s = mx / denom if mx else 1.0
    return [qsym(x, denom, s) * s for x in w]


def qsym_row(w, rows, cols, bits):
    denom = (1 << (bits - 1)) - 1
    out = [0.0] * len(w)
    for r in range(rows):
        off = r * cols
        mx = max((abs(w[off + c]) for c in range(cols)), default=0.0)
        s = mx / denom if mx else 1.0
        for c in range(cols):
            out[off + c] = qsym(w[off + c], denom, s) * s
    return out


def col_scale(w, rows, cols, alpha):
    cs = []
    for c in range(cols):
        rms = 0.0
        wmax = 0.0
        for r in range(rows):
            v = abs(w[r * cols + c])
            rms += v * v
            wmax = max(wmax, v)
        rms = math.sqrt(rms / rows) if rows else 0.0
        s = ((rms + 1e-12) ** alpha) / ((wmax + 1e-12) ** (1.0 - alpha))
        cs.append(s if math.isfinite(s) and s > 0.0 else 1.0)
    return cs


def qsym_colscale(w, rows, cols, bits, block, alpha):
    denom = (1 << (bits - 1)) - 1
    return quant_block(w, rows, cols, block, denom, col_scale(w, rows, cols, alpha))


def codebook_block(w, rows, cols, block, codebook, signed_absmax):
    out = [0.0] * len(w)
    max_code = max(abs(x) for x in codebook) if signed_absmax else max(codebook)
    for rb in range(0, rows, block):
        re = min(rows, rb + block)
        for cb in range(0, cols, block):
            ce = min(cols, cb + block)
            mx = 0.0
            for r in range(rb, re):
                off = r * cols
                for c in range(cb, ce):
                    mx = max(mx, abs(w[off + c]))
            s = mx / max_code if mx else 1.0
            for r in range(rb, re):
                off = r * cols
                for c in range(cb, ce):
                    x = w[off + c] / s
                    q = min(codebook, key=lambda y: abs(x - y))
                    out[off + c] = q * s
    return out


def q_fp16(w):
    return [f32_to_f16(x) for x in w]


def config_space():
    for bits in (3, 4, 5, 6, 8, 16):
        yield {"method": "sym_tensor", "bits": bits, "scale": "tensor", "block": 0, "alpha": "", "clip": ""}
        yield {"method": "sym_row", "bits": bits, "scale": "row", "block": 0, "alpha": "", "clip": ""}
        for block in (16, 32, 64, 128, 256):
            yield {"method": "sym_block", "bits": bits, "scale": "block", "block": block, "alpha": "", "clip": ""}
    for bits in (4, 5, 6, 8):
        for block in (32, 64, 128, 256):
            for clip in (0.80, 0.85, 0.90, 0.95, 0.975, 1.0):
                yield {"method": "clip_block", "bits": bits, "scale": "block", "block": block, "alpha": "", "clip": clip}
            for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
                yield {"method": "colscale_block", "bits": bits, "scale": "colscale_block", "block": block, "alpha": alpha, "clip": ""}
    for block in (16, 32, 64, 128, 256):
        yield {"method": "fp4_e2m1", "bits": 4, "scale": "block", "block": block, "alpha": "", "clip": ""}
        yield {"method": "nf4", "bits": 4, "scale": "block", "block": block, "alpha": "", "clip": ""}
    yield {"method": "fp16", "bits": 16, "scale": "none", "block": 0, "alpha": "", "clip": ""}


def estimate_cost(cfg, rows, cols):
    method = cfg["method"]
    bits = int(cfg["bits"])
    scale = cfg["scale"]
    block = int(cfg["block"] or 0)
    n = rows * cols
    if scale == "tensor" or scale == "none":
        scale_count = 1 if scale == "tensor" else 0
    elif scale == "row":
        scale_count = rows
    elif scale == "block" and block:
        scale_count = ((rows + block - 1) // block) * ((cols + block - 1) // block)
    elif scale == "colscale_block" and block:
        scale_count = ((rows + block - 1) // block) * ((cols + block - 1) // block) + cols
    else:
        scale_count = 0
    bytes_per_weight = bits / 8.0
    if method == "fp16":
        ops = 0.0
        path = "fp32_fma_after_widen"
    elif bits == 8 and method in ("sym_block", "clip_block", "colscale_block"):
        ops = 1.0 if scale != "colscale_block" else 2.0
        path = "a64fx_sdot"
    elif bits == 16:
        ops = 1.0
        path = "sve_i16_mac"
    elif method in ("fp4_e2m1", "nf4"):
        ops = 3.0
        path = "unpack_lookup_then_fma"
    else:
        ops = 2.0 if bits < 8 else 1.0
        path = "unpack_to_i8_then_dot" if bits < 8 else "scalar_or_sdot_candidate"
    return {
        "bytes_per_weight": bytes_per_weight,
        "scale_count": scale_count,
        "scales_per_1k": 1000.0 * scale_count / max(n, 1),
        "dequant_ops_per_value": ops,
        "a64fx_path": path,
    }


def apply_config(w, rows, cols, cfg):
    method = cfg["method"]
    bits = int(cfg["bits"])
    block = int(cfg["block"] or 0)
    if method == "sym_tensor":
        return qsym_tensor(w, bits)
    if method == "sym_row":
        return qsym_row(w, rows, cols, bits)
    if method == "sym_block":
        return quant_block(w, rows, cols, block, (1 << (bits - 1)) - 1)
    if method == "clip_block":
        return qsym_block_clip(w, rows, cols, bits, block, float(cfg["clip"]))
    if method == "colscale_block":
        return qsym_colscale(w, rows, cols, bits, block, float(cfg["alpha"]))
    if method == "fp4_e2m1":
        return quant_fp4_block(w, rows, cols, block)
    if method == "nf4":
        return codebook_block(w, rows, cols, block, NF4, True)
    if method == "fp16":
        return q_fp16(w)
    raise ValueError(method)


def iter_tensors(model, include, max_tensors):
    shards = sorted(f for f in os.listdir(model) if f.endswith(".safetensors"))
    count = 0
    for shard in shards:
        sf = SafeFile(os.path.join(model, shard))
        try:
            for name, meta in sf.tensors.items():
                if max_tensors and count >= max_tensors:
                    return
                if not include.match(name):
                    continue
                if name.endswith("_scale_inv") or name.endswith(".scale") or name.endswith(".scale_inv"):
                    continue
                if len(meta.get("shape", [])) != 2:
                    continue
                if meta["dtype"] not in ("BF16", "F16", "FP16", "F32", "F8_E4M3", "F8_E4M3FN"):
                    continue
                count += 1
                yield sf, shard, name, meta
        finally:
            sf.close()


def collect_tensor_tasks(model, include, max_tensors):
    tasks = []
    shards = sorted(f for f in os.listdir(model) if f.endswith(".safetensors"))
    for shard in shards:
        sf = SafeFile(os.path.join(model, shard))
        try:
            for name, meta in sf.tensors.items():
                if max_tensors and len(tasks) >= max_tensors:
                    return tasks
                if not include.match(name):
                    continue
                if name.endswith("_scale_inv") or name.endswith(".scale") or name.endswith(".scale_inv"):
                    continue
                if len(meta.get("shape", [])) != 2:
                    continue
                if meta["dtype"] not in ("BF16", "F16", "FP16", "F32", "F8_E4M3", "F8_E4M3FN"):
                    continue
                tasks.append((shard, name, meta))
        finally:
            sf.close()
    return tasks


def evaluate_tensor_task(args):
    model, model_name, shard, name, meta, rows_cap, cols_cap, max_elements, configs, topk = args
    sf = SafeFile(os.path.join(model, shard))
    try:
        w, sr, sc, _ = load_sample(sf, name, rows_cap, cols_cap, max_elements)
    finally:
        sf.close()
    rows = []
    heap = []
    seq = 0
    evaluated_configs = 0
    for cfg in configs:
        if cfg["block"] and (cfg["block"] > max(sr, sc) * 2):
            continue
        evaluated_configs += 1
        t0 = time.perf_counter()
        got = apply_config(w, sr, sc, cfg)
        quant_ms = (time.perf_counter() - t0) * 1000.0
        m = metrics(w, got)
        row = {
            "model": model_name,
            "shard": shard,
            "tensor": name,
            "dtype": meta["dtype"],
            "rows": meta["shape"][0],
            "cols": meta["shape"][1],
            "sample_rows": sr,
            "sample_cols": sc,
            **cfg,
            **m,
            "quant_ms": quant_ms,
            **estimate_cost(cfg, sr, sc),
        }
        if topk:
            item = (-row["rel_l2"], seq, row)
            seq += 1
            if len(heap) < topk:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
        else:
            rows.append(row)
        del got
    if topk:
        rows = [x[2] for x in heap]
    rows.sort(key=lambda r: r["rel_l2"])
    return rows, len(w), evaluated_configs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--cols", type=int, default=512)
    ap.add_argument("--max-elements-per-tensor", type=int, default=32768)
    ap.add_argument("--include", default=r".*\.weight$")
    ap.add_argument("--max-tensors", type=int, default=0)
    ap.add_argument("--topk", type=int, default=0, help="write only top-k configs per tensor by rel_l2")
    ap.add_argument("--summary-json", default="", help="write model-level timing summary JSON")
    ap.add_argument("--threads", type=int, default=0,
                    help="CPU workers. Default is half of os.cpu_count(); use 1 for sequential.")
    args = ap.parse_args()

    start = time.perf_counter()
    model = os.path.expanduser(args.model)
    model_name = os.path.basename(model.rstrip("/"))
    include = re.compile(args.include)
    configs = list(config_space())
    threads = args.threads if args.threads > 0 else max(1, (os.cpu_count() or 2) // 2)
    fields = [
        "model", "shard", "tensor", "dtype", "rows", "cols", "sample_rows", "sample_cols",
        "method", "bits", "scale", "block", "alpha", "clip",
        "rmse", "mae", "max_abs", "rel_l2", "cosine", "sqnr_db",
        "quant_ms", "bytes_per_weight", "scale_count", "scales_per_1k",
        "dequant_ops_per_value", "a64fx_path",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    processed = 0
    evaluated_configs = 0
    sampled_elements = 0
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        if threads == 1:
            for sf, shard, name, meta in iter_tensors(model, include, args.max_tensors):
                rows, elems, evals = evaluate_tensor_task(
                    (model, model_name, shard, name, meta, args.rows, args.cols,
                     args.max_elements_per_tensor, configs, args.topk)
                )
                writer.writerows(rows)
                processed += 1
                sampled_elements += elems
                evaluated_configs += evals
                if processed % 10 == 0:
                    f.flush()
                    print(f"{model_name}: searched {processed} tensors x {len(configs)} configs", file=sys.stderr)
        else:
            tasks = collect_tensor_tasks(model, include, args.max_tensors)
            worker_args = [
                (model, model_name, shard, name, meta, args.rows, args.cols,
                 args.max_elements_per_tensor, configs, args.topk)
                for shard, name, meta in tasks
            ]
            with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as ex:
                futs = [ex.submit(evaluate_tensor_task, wa) for wa in worker_args]
                for fut in concurrent.futures.as_completed(futs):
                    rows, elems, evals = fut.result()
                    writer.writerows(rows)
                    processed += 1
                    sampled_elements += elems
                    evaluated_configs += evals
                    if processed % 10 == 0:
                        f.flush()
                        print(f"{model_name}: searched {processed}/{len(tasks)} tensors x {len(configs)} configs", file=sys.stderr)
    elapsed = time.perf_counter() - start
    summary = {
        "model": model_name,
        "model_path": model,
        "out": args.out,
        "rows_cap": args.rows,
        "cols_cap": args.cols,
        "max_elements_per_tensor": args.max_elements_per_tensor,
        "max_tensors": args.max_tensors,
        "topk": args.topk,
        "threads": threads,
        "processed_tensors": processed,
        "config_space_size": len(configs),
        "evaluated_configs": evaluated_configs,
        "sampled_elements": sampled_elements,
        "elapsed_sec": elapsed,
        "elapsed_min": elapsed / 60.0,
        "tensors_per_sec": processed / elapsed if elapsed > 0.0 else 0.0,
        "configs_per_sec": evaluated_configs / elapsed if elapsed > 0.0 else 0.0,
        "sampled_elements_per_sec": sampled_elements / elapsed if elapsed > 0.0 else 0.0,
    }
    if args.summary_json:
        with open(args.summary_json, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write("\n")
    print(
        f"{model_name}: wrote search results for {processed} tensors to {args.out}; "
        f"elapsed={elapsed:.3f}s configs={evaluated_configs} configs/s={summary['configs_per_sec']:.2f}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
