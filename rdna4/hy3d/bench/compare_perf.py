#!/usr/bin/env python3
"""Side-by-side perf comparison: PyTorch+ROCm reference vs HIP runner.

Reads two timing JSON files (produced by run_full_pipeline.py
--per-stage-timing and test_hip_hy3d --per-stage-timing) and emits a markdown
table on stdout. Optionally folds in a rocprofv3 kernel-trace JSON to list the
top-N HIP kernels by cumulative GPU time.
"""
import argparse
import json
import os
import sys


def fmt_ms(x):
    return f"{x:8.2f}"


def fmt_x(a, b):
    if a is None or b is None or b == 0:
        return "   n/a"
    return f"{a / b:5.2f}x"


def load(path):
    with open(path, "r") as f:
        return json.load(f)


def parse_rocprof_top(path, top_n):
    """rocprofv3 kernel-trace JSON: schema varies by version. We try a few
    common shapes and fall back to printing nothing on shape mismatch."""
    try:
        with open(path, "r") as f:
            blob = json.load(f)
    except Exception as exc:
        return None, f"could not read {path}: {exc}"

    # Try several shapes
    candidates = []
    # rocprofv3 SDK output is a dict whose `rocprofiler-sdk-tool` value can be
    # either a dict or a list of dicts (one per process).
    sdk_blocks = []
    if isinstance(blob, dict) and "rocprofiler-sdk-tool" in blob:
        sdk = blob["rocprofiler-sdk-tool"]
        sdk_blocks = sdk if isinstance(sdk, list) else [sdk]

    # Build a kernel_id -> name map from `kernel_symbols` and a strings table
    # from buffer_records. Names live under buffer_records.kernel_symbols /
    # `strings.callback_records.kernel_dispatch.kernel_names`, depending on
    # the schema.
    for blk in sdk_blocks:
        bufrec = blk.get("buffer_records", {}) or {}
        # rocprofv3: kernel_symbols at top level of the SDK block
        sym_map = {}
        for s in (blk.get("kernel_symbols") or []):
            kid = s.get("kernel_id")
            kname = s.get("demangled_kernel_name") or s.get("kernel_name") \
                    or s.get("formatted_kernel_name")
            if kid is not None and kname:
                sym_map[kid] = kname
        # Fallbacks
        for s in (bufrec.get("kernel_symbols") or []):
            kid = s.get("kernel_id")
            kname = s.get("demangled_kernel_name") or s.get("kernel_name")
            if kid is not None and kname and kid not in sym_map:
                sym_map[kid] = kname

        for r in (bufrec.get("kernel_dispatch") or []):
            kd = r.get("dispatch_info", {}) or r
            kid = kd.get("kernel_id") or r.get("kernel_id")
            name = sym_map.get(kid) or r.get("kernel_name") or kd.get(
                "kernel_name") or f"kernel_{kid}"
            start = r.get("start_timestamp") or r.get("start") or 0
            end = r.get("end_timestamp") or r.get("end") or 0
            candidates.append((str(name), float(end - start) / 1e6))

    # Fallback: traceEvents
    if not candidates and isinstance(blob, dict) and "traceEvents" in blob:
        for ev in blob["traceEvents"]:
            if ev.get("ph") != "X":
                continue
            name = ev.get("name", "?")
            dur_us = ev.get("dur", 0.0)
            candidates.append((str(name), float(dur_us) / 1000.0))

    if not candidates:
        return None, "no kernel records recognized"

    agg = {}
    for name, ms in candidates:
        agg[name] = agg.get(name, 0.0) + ms
    top = sorted(agg.items(), key=lambda kv: -kv[1])[:top_n]
    return top, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pytorch", required=True, help="timings_pytorch.json")
    ap.add_argument("--hip",     required=True, help="timings_hip.json")
    ap.add_argument("--rocprof", default=None,
                    help="rocprofv3 kernel-trace JSON (optional)")
    ap.add_argument("--top-n", type=int, default=5)
    args = ap.parse_args()

    py = load(args.pytorch)
    hp = load(args.hip)

    print(f"# rdna4/hy3d perf — PyTorch+ROCm vs HIP runner")
    print()
    print(f"- PyTorch: device={py.get('device','?')} dtype={py.get('dtype','?')} "
          f"steps={py.get('steps','?')} guidance={py.get('guidance','?')} "
          f"octree={py.get('octree','?')}")
    print(f"- HIP:     steps={hp.get('steps','?')} guidance={hp.get('guidance','?')} "
          f"grid_res={hp.get('grid_res','?')}")
    print()

    rows = [
        ("DINOv2 image encoder",
         py.get("dino_ms"), hp.get("dino_ms")),
        ("DiT total (all steps, CFG)",
         py.get("dit_total_ms"), hp.get("dit_total_ms")),
        ("DiT per-step mean",
         py.get("dit_step_ms_mean"), hp.get("dit_step_ms_mean")),
        ("ShapeVAE decode + SDF",
         py.get("vae_ms"), hp.get("vae_ms")),
        ("End-to-end (e2e)",
         py.get("e2e_ms"), hp.get("e2e_ms")),
    ]

    print("| Stage | PyTorch+ROCm (ms) | HIP (ms) | Gap (HIP / PyT) |")
    print("|-------|-------------------|----------|-----------------|")
    for label, p, h in rows:
        ps = fmt_ms(p) if p is not None else "    n/a"
        hs = fmt_ms(h) if h is not None else "    n/a"
        gx = fmt_x(h, p) if (p is not None and h is not None) else "  n/a"
        print(f"| {label} | {ps} | {hs} | {gx} |")
    print()

    if args.rocprof:
        top, err = parse_rocprof_top(args.rocprof, args.top_n)
        if err:
            print(f"_rocprof: {err}_")
        else:
            print(f"## Top-{args.top_n} HIP kernels by cumulative GPU time")
            print()
            print("| Kernel | Total (ms) |")
            print("|--------|-----------:|")
            for name, ms in top:
                short = name if len(name) <= 80 else name[:77] + "..."
                print(f"| `{short}` | {ms:.2f} |")
            print()


if __name__ == "__main__":
    sys.exit(main() or 0)
