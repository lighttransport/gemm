#!/usr/bin/env python3
"""Native FP8 diagnostics for rdna4/qimg.

The default commands use pinned 256x256 Qwen-Image artifacts already present in
the repo and avoid a full diffusers run. Torch is optional for latent and
quantization checks, but required for the GEMM probe.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
QIMG = Path(__file__).resolve().parents[1]
DEFAULT_REF = ROOT / "ref/qwen_image/final_latent_packed_256.bin"
DEFAULT_CANDIDATES = [
    QIMG / "hip_bf16fp8_256.bin",
    QIMG / "hip_fp8fp8_256.bin",
    QIMG / "hip_qimg_final_pinned.bin",
    QIMG / "hip_qimg_orig_check.bin",
]


def fp8_e4m3_to_f32_byte(v: int) -> float:
    if v == 0x00:
        return 0.0
    if v == 0x80:
        return -0.0
    sign = -1.0 if (v & 0x80) else 1.0
    exp = (v >> 3) & 0x0F
    mant = v & 0x07
    if exp == 0x0F and mant == 0x07:
        return 0.0
    if exp == 0:
        return sign * math.ldexp(mant / 8.0, -6)
    return sign * math.ldexp(1.0 + mant / 8.0, exp - 7)


FP8_LUT = np.array([fp8_e4m3_to_f32_byte(i) for i in range(256)], dtype=np.float32)
POS_CODES = np.array([i for i in range(0x7F)], dtype=np.uint8)
POS_VALUES = FP8_LUT[POS_CODES].astype(np.float32)


def load_f32(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.float32)
    if data.size == 0:
        raise ValueError(f"{path}: no float32 data")
    return data


def metrics(ref: np.ndarray, got: np.ndarray) -> dict[str, float]:
    n = min(ref.size, got.size)
    if ref.size != got.size:
        print(f"warning: size mismatch ref={ref.size} got={got.size}; using first {n}", file=sys.stderr)
    r = ref.reshape(-1)[:n].astype(np.float64)
    g = got.reshape(-1)[:n].astype(np.float64)
    d = g - r
    mse = float(np.mean(d * d))
    mae = float(np.mean(np.abs(d)))
    rmse = math.sqrt(mse)
    max_i = int(np.argmax(np.abs(d)))
    maxd = float(abs(d[max_i]))
    cos = float(np.dot(r, g) / (np.linalg.norm(r) * np.linalg.norm(g) + 1e-30))
    peak = float(np.max(np.abs(r)))
    psnr = 20.0 * math.log10(max(peak, 1e-20)) - 10.0 * math.log10(max(mse, 1e-30))
    return {
        "n": float(n),
        "cos": cos,
        "mae": mae,
        "rmse": rmse,
        "max": maxd,
        "max_i": float(max_i),
        "psnr_peak": psnr,
    }


def print_metrics(label: str, m: dict[str, float]) -> None:
    print(
        f"{label}: n={int(m['n'])} cos={m['cos']:.9f} mae={m['mae']:.8g} "
        f"rmse={m['rmse']:.8g} max={m['max']:.8g}@{int(m['max_i'])} "
        f"psnr_peak={m['psnr_peak']:.2f}dB"
    )


def quantize_e4m3_nearest(x: np.ndarray) -> np.ndarray:
    """Round finite float32 values to E4M3FN bytes using nearest representable value."""
    arr = np.asarray(x, dtype=np.float32)
    ax = np.nan_to_num(np.abs(arr), nan=0.0, posinf=448.0, neginf=448.0)
    ax = np.minimum(ax, np.float32(448.0))
    idx_hi = np.searchsorted(POS_VALUES, ax, side="left")
    idx_hi = np.clip(idx_hi, 0, POS_VALUES.size - 1)
    idx_lo = np.maximum(idx_hi - 1, 0)
    hi = POS_VALUES[idx_hi]
    lo = POS_VALUES[idx_lo]
    choose_hi = np.abs(hi - ax) < np.abs(ax - lo)
    codes = np.where(choose_hi, POS_CODES[idx_hi], POS_CODES[idx_lo]).astype(np.uint8)
    return np.where(arr < 0, codes | np.uint8(0x80), codes).astype(np.uint8)


def dequantize_e4m3(q: np.ndarray) -> np.ndarray:
    return FP8_LUT[np.asarray(q, dtype=np.uint8)]


def row_quantize(x: np.ndarray, scale_div: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    if x.ndim != 2:
        raise ValueError("row_quantize expects a 2D array")
    row_max = np.max(np.abs(x), axis=1).astype(np.float32)
    scales = np.maximum(row_max / np.float32(scale_div), np.float32(1e-12))
    scaled = x / scales[:, None]
    q = quantize_e4m3_nearest(scaled)
    recon = dequantize_e4m3(q) * scales[:, None]
    err = np.abs(recon - x)
    stats = {
        "row_max_avg": float(np.mean(row_max)),
        "row_max_max": float(np.max(row_max)),
        "mae": float(np.mean(err)),
        "rmse": float(math.sqrt(float(np.mean(err * err)))),
        "p99": float(np.quantile(err.reshape(-1), 0.99)),
        "zeros_pct": float(100.0 * np.mean((q & 0x7F) == 0)),
        "sat_pct": float(100.0 * np.mean(np.abs(scaled) >= 448.0)),
    }
    return q, scales, recon.astype(np.float32), stats


def cmd_compare_latents(args: argparse.Namespace) -> int:
    ref = load_f32(args.ref)
    for path in args.candidate:
        got = load_f32(path)
        print_metrics(path.name, metrics(ref, got))
    return 0


def cmd_quant_sweep(args: argparse.Namespace) -> int:
    data = load_f32(args.input)
    if data.size % args.cols != 0:
        raise ValueError(f"{args.input}: {data.size} floats is not divisible by --cols {args.cols}")
    x = data.reshape(data.size // args.cols, args.cols)
    print(f"input: {args.input} shape={x.shape} mean={x.mean():.6g} std={x.std():.6g}")
    for div in args.div:
        _, _, recon, st = row_quantize(x, div)
        m = metrics(x, recon)
        print(
            f"div={div:g}: cos={m['cos']:.9f} psnr={m['psnr_peak']:.2f}dB "
            f"mae={st['mae']:.7g} rmse={st['rmse']:.7g} p99={st['p99']:.7g} "
            f"row_max_avg={st['row_max_avg']:.7g} row_max_max={st['row_max_max']:.7g} "
            f"zeros={st['zeros_pct']:.3f}% sat={st['sat_pct']:.3f}%"
        )
    return 0


def import_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("torch is required for gemm-probe; install it in rdna4/qimg/.venv") from exc
    return torch


def load_safetensor_tensor(path: Path, name: str):
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as f:
        if name not in f.keys():
            matches = [k for k in f.keys() if name in k]
            hint = f"; substring matches: {matches[:12]}" if matches else ""
            raise KeyError(f"{name!r} not found in {path}{hint}")
        return f.get_tensor(name)


def read_safetensors_header(path: Path) -> dict:
    with path.open("rb") as f:
        n = int.from_bytes(f.read(8), "little")
        return json.loads(f.read(n))


def cmd_list_tensors(args: argparse.Namespace) -> int:
    pat = re.compile(args.pattern) if args.pattern else None
    header = read_safetensors_header(args.safetensors)
    for key in sorted(k for k in header.keys() if k != "__metadata__"):
        if pat and not pat.search(key):
            continue
        info = header[key]
        print(f"{key}\tshape={info.get('shape')}\tdtype={info.get('dtype')}")
    return 0


def cmd_gemm_probe(args: argparse.Namespace) -> int:
    torch = import_torch()
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    rng = np.random.default_rng(args.seed)

    if args.x_bin:
        x_np = load_f32(args.x_bin)
        if x_np.size % args.k != 0:
            raise ValueError(f"{args.x_bin}: {x_np.size} floats is not divisible by --k {args.k}")
        x_np = x_np.reshape(x_np.size // args.k, args.k)
    else:
        x_np = (rng.standard_normal((args.m, args.k)).astype(np.float32) * args.x_std)

    if args.w_safetensors:
        if not args.tensor:
            raise ValueError("--tensor is required with --w-safetensors")
        w = load_safetensor_tensor(args.w_safetensors, args.tensor).to(torch.float32)
        if w.ndim != 2:
            raise ValueError(f"{args.tensor}: expected 2D weight, got {tuple(w.shape)}")
        if w.shape[1] != x_np.shape[1]:
            raise ValueError(f"K mismatch: X has {x_np.shape[1]}, W has {w.shape[1]}")
        if args.n and args.n < w.shape[0]:
            w = w[: args.n].contiguous()
    else:
        w_np = (rng.standard_normal((args.n, x_np.shape[1])).astype(np.float32) * args.w_std)
        w = torch.from_numpy(w_np)

    x = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)
    w = w.to(device=device, dtype=torch.float32)
    print(f"device={device} torch={torch.__version__} x={tuple(x.shape)} w={tuple(w.shape)}")

    ref = x @ w.t()
    bf16 = x.to(torch.bfloat16) @ w.to(torch.bfloat16).t()
    print_metrics("bf16_vs_f32", metrics(ref.detach().cpu().numpy(), bf16.float().detach().cpu().numpy()))

    q, scales_np, x_recon_np, st = row_quantize(x_np.astype(np.float32), args.scale_div)
    x_recon = torch.from_numpy(x_recon_np).to(device=device)
    w_fp8_recon = w.to(torch.float8_e4m3fn).to(torch.float32)
    fp8_recon = x_recon @ w_fp8_recon.t()
    print(
        f"x_quant div={args.scale_div:g}: mae={st['mae']:.7g} p99={st['p99']:.7g} "
        f"zeros={st['zeros_pct']:.3f}% sat={st['sat_pct']:.3f}%"
    )
    print_metrics("fp8_recon_matmul_vs_f32", metrics(ref.detach().cpu().numpy(), fp8_recon.detach().cpu().numpy()))

    if device == "cuda" and hasattr(torch, "_scaled_mm"):
        try:
            scales = torch.from_numpy(scales_np.astype(np.float32)).to(device=device).reshape(-1, 1)
            x_fp8 = torch.clamp(x / scales, -448.0, 448.0).to(torch.float8_e4m3fn)
            w_fp8 = torch.clamp(w, -448.0, 448.0).to(torch.float8_e4m3fn)
            scale_b = torch.ones((1, w_fp8.shape[0]), device=device, dtype=torch.float32)
            try:
                native = torch._scaled_mm(
                    x_fp8, w_fp8.t(), scale_a=scales, scale_b=scale_b, out_dtype=torch.float32
                )
                native_label = "torch_scaled_mm_f32_vs_f32"
            except Exception as exc:
                if "BFloat16 output" not in str(exc):
                    raise
                native = torch._scaled_mm(
                    x_fp8, w_fp8.t(), scale_a=scales, scale_b=scale_b, out_dtype=torch.bfloat16
                )
                native_label = "torch_scaled_mm_bf16_vs_f32"
            print_metrics(native_label, metrics(ref.detach().cpu().numpy(), native.float().detach().cpu().numpy()))
        except Exception as exc:
            print(f"torch_scaled_mm skipped: {type(exc).__name__}: {exc}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("compare-latents", help="Compare HIP final latent bins against a reference bin")
    p.add_argument("--ref", type=Path, default=DEFAULT_REF)
    p.add_argument("candidate", type=Path, nargs="*", default=DEFAULT_CANDIDATES)
    p.set_defaults(func=cmd_compare_latents)

    p = sub.add_parser("quant-sweep", help="Sweep qimg per-row FP8 activation scale divisors")
    p.add_argument("--input", type=Path, default=DEFAULT_REF)
    p.add_argument("--cols", type=int, default=64)
    p.add_argument("--div", type=float, nargs="+", default=[448.0, 480.0, 512.0, 576.0, 896.0])
    p.set_defaults(func=cmd_quant_sweep)

    p = sub.add_parser("list-tensors", help="List safetensors keys, optionally filtered by regex")
    p.add_argument("safetensors", type=Path)
    p.add_argument("--pattern", default="")
    p.set_defaults(func=cmd_list_tensors)

    p = sub.add_parser("gemm-probe", help="Compare BF16 and native FP8 GEMM against F32")
    p.add_argument("--x-bin", type=Path)
    p.add_argument("--m", type=int, default=256)
    p.add_argument("--n", type=int, default=3072)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--x-std", type=float, default=1.0)
    p.add_argument("--w-std", type=float, default=0.02)
    p.add_argument("--scale-div", type=float, default=512.0)
    p.add_argument("--w-safetensors", type=Path)
    p.add_argument("--tensor")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(func=cmd_gemm_probe)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
