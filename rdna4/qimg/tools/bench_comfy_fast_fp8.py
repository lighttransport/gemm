#!/usr/bin/env python3
"""Low-memory throughput benchmark of ComfyUI's two Qwen-Image FP8 paths.

ComfyUI keeps the FP8 e4m3 DiT weights and either:
  - default: dequant/cast the weight to bf16 and run F.linear (bf16 x bf16), or
  - --fast fp8_matrix_mult: clamp/cast activations to fp8 and call
    torch._scaled_mm(x_fp8, w_fp8.t(), scale_a=1, scale_b=1).

This times both on the real qimg DiT GEMM shapes (hidden=3072, mlp_h=12288),
one GEMM at a time so peak VRAM stays tiny (no full pipeline -> no OOM).
"""
import torch, torch.nn.functional as F, time

dev = "cuda"
assert torch.cuda.is_available(), "no GPU"
print(f"torch={torch.__version__} hip={torch.version.hip} gpu={torch.cuda.get_device_name(0)}", flush=True)

# (label, N, K) for one Qwen-Image MMDiT block (img side); mod excluded (tiny M path).
GEMMS = [
    ("attn_q/k/v/out", 3072, 3072),
    ("mlp_fc1",       12288, 3072),
    ("mlp_fc2",        3072, 12288),
    ("mod",           18432, 3072),
]
TOKENS = [("256x256", 256), ("1024x1024", 4096)]
ITERS, WARM = 50, 10


def bench(fn):
    for _ in range(WARM): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(ITERS): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / ITERS


for res, M in TOKENS:
    print(f"\n=== M={M} tokens ({res}) ===", flush=True)
    print(f"{'gemm':16s} {'shape(MxNxK)':22s} {'bf16 TF/s':>10s} {'fp8 TF/s':>10s} {'fp8 ms':>9s} {'speedup':>8s}")
    for label, N, K in GEMMS:
        flop = 2.0 * M * N * K
        xb = (torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 0.1)
        wb = (torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.05)
        t_bf16 = bench(lambda: F.linear(xb, wb))
        # comfy fast-fp8: scale_a=scale_b=1, activations clamped/cast to e4m3
        xf = torch.clamp(xb.float(), -448, 448).to(torch.float8_e4m3fn).contiguous()
        wf = wb.to(torch.float8_e4m3fn).contiguous()
        s = torch.ones((), device=dev, dtype=torch.float32)
        def fp8_call():
            y = torch._scaled_mm(xf, wf.t(), scale_a=s, scale_b=s, out_dtype=torch.bfloat16)
            return y[0] if isinstance(y, tuple) else y
        t_fp8 = bench(fp8_call)
        print(f"{label:16s} {f'{M}x{N}x{K}':22s} {flop/t_bf16/1e12:10.1f} {flop/t_fp8/1e12:10.1f} "
              f"{t_fp8*1e3:9.4f} {t_bf16/t_fp8:7.2f}x", flush=True)
        del xb, wb, xf, wf
        torch.cuda.empty_cache()
