# SAM 3.1 CUDA runner — performance notes

End-to-end runtime is currently ~4× the PyTorch reference on the same
GPU while producing slightly better-quality masks (IoU 0.98 vs ref,
score 0.877 vs 0.841). This document captures the measured hot spots
and the optimizations prioritized to close the gap.

## Current state (fp16 default)

- Weights: F16 on-device (single upload, cached in ctx).
- Activations: F32.
- GEMMs: two paths — `gemm_f16_f32` MMA m16n8k16 (sm_80+) or
  `gemm_tiled_f16_f32` shared-memory fallback. Selected by `c->use_mma`
  which is driven by `SAM3_GEMM` env var and the new
  `cuda_sam3_1_config.precision` field.
- FPN conv projections deliberately kept F32 (1x1 and 3x3 over Ci up
  to 1024 — F16 compounds to ~0.5 max_abs drift here).
- RoPE tables precomputed once at ctx creation.
- No cuBLAS / no flash-attention / no LN+GEMM fusion.

## Where the time goes (qualitative, based on 32 ViT blocks × 4 GEMMs)

ViT dominates the pipeline. Per block:
1. `ln1` (F32).
2. QKV projection GEMM (1024 → 3072).
3. Windowed-or-global attention (K/V transpose + flash-attention-style
   kernel).
4. Output projection (1024 → 1024).
5. `ln2`.
6. MLP fc1 (1024 → 4736).
7. GELU (separate kernel).
8. MLP fc2 (4736 → 1024).

5184 tokens × 32 blocks × 4 large GEMMs = the bulk of wall time.

## Precision knob (implemented)

`cuda_sam3_1_config.precision` / `params.precision` / `--precision`:
- `fp16` (default): MMA tensor cores, fastest, ~1e-4 per-GEMM drift.
- `fp32`: forces tiled F32-accum GEMM. Slower but avoids drift — use
  this for reference-level accuracy checks.
- `bf16`, `fp8`: accepted but log a warning and fall back to fp16.
  Landing them requires new kernel variants (see "Follow-up work").

## Recommended optimizations (to close the 4× gap)

Ordered by expected impact vs refactor cost. None are implemented yet.

1. **cuBLAS/cuBLASLt for ViT GEMMs** — the MMA m16n8k16 custom kernel
   lacks persistent scheduling and sophisticated tiling; cuBLAS
   typically wins 2–3× on these shapes (Co=1024/3072/4736). Medium
   refactor: wire `cublasHandle_t` into the ctx, dispatch
   `cublasGemmEx`/`cublasLtMatmul` for F16×F16→F32. Est. ~30 % of
   total runtime.
2. **Flash-Attention v2 kernel** — current `r_kvtx()` + `r_fa()` pair
   re-reads K/V from HBM per head/window. A fused FA2 kernel with
   cp.async double-buffering is ~20–30 % of ViT time. Medium-large
   refactor (~300 LOC new kernel).
3. **Fuse LN + GEMM** — `ln1` output is written to `d_tmp`, then read
   back by the QKV GEMM. Fusing eliminates one HBM round-trip per
   block (×32). Medium refactor; ~10–15 %.
4. **Fuse GELU into fc1 epilogue** — drop a separate grid launch per
   block. Small refactor; ~3–5 %.
5. **cp.async double-buffer in tiled GEMM** — relevant only for
   `fp32` precision path and pre-sm_80 GPUs. Small refactor; ~5 % on
   the tiled code path.
6. **Persistent kernel for attention heads** — collapses
   16 heads × 9 windows = 144 launches into one grid. Large
   refactor; ~8–12 %.
7. **FPN convs in F16** — FPN is a small fraction of total runtime
   and its F32 weights are a deliberate accuracy choice. Revisit
   only if ViT wins aren't enough.

## Follow-up work: actual bf16/fp8 kernels

Landing bf16 end-to-end requires:
- `upload_bf16()` helper (mirror of `upload_f16()` with F32→bf16
  conversion on host).
- bf16 variants of `gemm_*_f32`, `ln`, `rope`, attention kernels
  (tensor cores on sm_80+ support bf16 × bf16 → f32 natively, so this
  is mostly a template/dtype parameter swap in kernel source).
- Verification against the PyTorch reference's bf16 autocast path.

fp8 (e4m3/e5m2) would additionally require per-tensor scale tracking
and sm_90+ (H100) for native tensor-core support. Defer until bf16 is
stable and a use case appears.
