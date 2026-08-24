# a64fx/vlm — status and milestones

A64FX SVE vision encoder for Qwen3-VL (drop-in replacement for
`vision_encode()`). Numeric ground truth lives at
`common/vision_encoder.h:554-1012`.

> **For the current performance profile, the optimization notes, and the
> "at its practical compute limit" conclusion, see [`vlm.md`](vlm.md)** — that
> page is the authoritative, kept-up-to-date doc. This file only tracks the
> build/run entry points and the milestone history.

Build: `make CC=fcc clean && make CC=fcc`        # fcc + OpenMP (default)
Test:  `make CC=fcc test`                          # kernel unit tests (regression gate)
Run:   `OMP_NUM_THREADS=48 ./build/vlm_runner <model.gguf> <mmproj.gguf> \
       --image-size 384 --threads 48 --bench 8 --dtype int8`

Reference model: **Qwen3VL-2B-Instruct-F16**, `~/fujisan.jpg` 384×256 → 384
patches → **96 merged tokens**. Headline: int8 ~1300–1650 tok/s (48T),
2.7–3.4× vs fp16; int8 norm 452.0263, fp16 455.6237.

---

## Milestones

| ID | What | Where |
|----|------|-------|
| M1 | FP32 baseline drop-in encode | `src/vit_a64fx.c` |
| M2 | SVE kernels + cache (LayerNorm, FEXPA softmax/GELU, attn) | `kernels/norm_sve.c`, `src/vit_a64fx.c` |
| M3 | BF16 storage + asm microkernel (8×48 tile, LD1H+LSL#16) | `kernels/{bf16_gemm.c,micro_kernel_bf16B_8x3.S}` |
| M4 | FP16 storage + asm microkernel (LD1H+FCVT) | `kernels/{fp16_gemm.c,micro_kernel_fp16B_8x3.S}` |
| M5 | Parallel cache build + CMG NUMA infra + per-CMG replication | `src/cmg_pool.{c,h}`, `src/vit_a64fx.c` |
| M6 | Fused dual-conv2d SVE patch-embedding (bit-identical, ~2.8×) | `kernels/conv2d_sve.{c,h}`, `tools/test_conv2d_sve.cpp` |
| **M7** | **INT8 W8A8 SDOT GEMM** (fused dequant kernel + SVE activation quantize) — the big win, ~2.5× on the GEMM, ~3× end-to-end vs fp16 | `kernels/{int8_gemm.c,kernel_6x4_int8.S}`, `tools/test_int8_gemm.c` |
| **M8** | **OpenMP backend by default for fcc/fccpx** — the C11-thrd fallback silently serialized the attention (~1.6× lost); now `make CC=fcc` defaults to OpenMP | `Makefile`, `src/vlm_parallel.c` |

Current state: the encoder is at its **practical compute limit** (int8 SDOT
GEMMs HBM-bound, fp32-FMA attention memory/LLC-bound in-situ with int8 rejected
as too lossy, GELU tanh-compute-bound). See `vlm.md` §2–3.6 for the full
analysis and the list of levers that were tried and rejected.

---

## Files of interest

```
include/vit_a64fx.h               — public API
src/vit_a64fx.c                   — encoder pipeline, cache build/replicate, dispatch
src/cmg_pool.{c,h}                — CMG NUMA primitives (mbind, pin, barrier)
src/vlm_parallel.c                — OpenMP / C11-thrd parallel backends
src/vlm_runner.c                  — CLI + VLM_NUMA env gate
kernels/{fp16,bf16,fp32}_gemm.c   — fp16/bf16/fp32 packed-B GEMM driver
kernels/{fp16,bf16,fp32}_*.S      — 8×48 asm microkernels
kernels/int8_gemm.c + kernel_6x4_int8.S  — INT8 SDOT GEMM (M7)
kernels/conv2d_sve.{c,h}          — fused dual-conv2d patch-embedding (M6)
kernels/norm_sve.c                — SVE LayerNorm batch kernel
tools/test_int8_gemm.c            — int8/int16 GEMM correctness vs fp32 ref + fused bit-identical check
tools/test_conv2d_sve.cpp         — conv2d SVE correctness vs scalar ref
tools/test_cmg_pool.c             — CMG NUMA sanity test
tools/bench_attn.c                — attention per-core micro-benchmark
tools/tensor_diff.c               — VLMD dump validator
../../common/vision_encoder.h     — bit-exact reference (CPU)
```
