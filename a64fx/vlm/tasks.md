# a64fx/vlm — status and remaining work

A64FX-optimized Qwen3-VL vision encoder. Numeric ground truth lives at
`common/vision_encoder.h:554-1012`. Validate via `build/tensor_diff` against
VLMD dumps from `cpu/vlm/test_vision`.

Build: `make CC=fcc OPENMP=1 clean && make CC=fcc OPENMP=1`
Run:   `OMP_NUM_THREADS=12 OMP_PROC_BIND=close OMP_PLACES=cores \
       ./build/vlm_runner <model.gguf> <mmproj.gguf> --image-size 384 \
       --threads 12 --bench 3 --dtype fp16`

## Current status (2026-05-13)

### Done

| Milestone | What | Where |
|-----------|------|-------|
| M1 | FP32 baseline drop-in encode | `src/vit_a64fx.c` |
| M2 | SVE kernels + cache (LayerNorm, FEXPA softmax/GELU, sve_dot_hd attn) | `kernels/norm_sve.c`, `src/vit_a64fx.c` |
| M3 | BF16 storage + asm microkernel (8x48 tile, LD1H+LSL#16) | `kernels/{bf16_gemm.c,micro_kernel_bf16B_8x3.S}` |
| M4 | FP16 storage + asm microkernel (LD1H+FCVT) | `kernels/{fp16_gemm.c,micro_kernel_fp16B_8x3.S}` |
| M5a | Parallel cache build (block-level OpenMP) | `src/vit_a64fx.c:vit_a64fx_cache_build` |
| M6 B3.1 | CMG NUMA infrastructure (mmap+mbind, pin, barrier) | `src/cmg_pool.{c,h}`, `tools/test_cmg_pool.c` |
| M6 B3.2 | Per-CMG weight replication via `vit_a64fx_cache_replicate` | `src/vit_a64fx.c` (`btp_repl_take`, `btp_repl_free`) |
| M6 B3.3 | CMG-aware GEMM dispatch (`gemm_{fp16,bf16}_BTP_cmg`) | `kernels/{fp16,bf16}_gemm.c`, `src/vit_a64fx.c:gemm_BT_dispatch` |
| M6 B3.5 | OMP thread pinning at encode entry (`cmg_pin_omp_threads`) | `src/vit_a64fx.c:vit_a64fx_encode` |

### Benchmarks @ 384x384, fp16, bit-identical norm=745.3280

| Config | Time | tok/s | Notes |
|--------|------|-------|-------|
| 12T no-NUMA (M4 baseline) | 2.74 s | 47.5 | `OMP_PROC_BIND=close OMP_PLACES=cores` |
| **48T no-NUMA** | **1.82 s** | **79.2** | current best, `OMP_PROC_BIND=close OMP_PLACES=cores` |
| 48T NUMA-replicated (`VLM_NUMA=4`) | 2.23 s | 64.6 | infrastructure works but slower than naive 48T |

Headline win this iteration: 12T → 48T = **1.67× (47.5 → 79.2 tok/s)**.

### Stage breakdown (M2 reference, 12T, 384x384)

attn 35.5% · ffn_down 21.1% · ffn_up 12.4% · patch_embed 10.1% ·
deepstack 7.0% · qkv 7.0% · attn_out 3.3% · mm_proj 2.3% · others <1.5%.

---

## Remaining work

### High priority — close out M6 NUMA

- [ ] **B3-INVESTIGATE: why does CMG-replicated 48T regress vs naive 48T?**
  Three hypotheses, in order of likelihood:
  1. `pthread_setaffinity_np` from `cmg_pin_omp_threads` fights the OMP runtime's
     own placement (the NUMA run does not set `OMP_PROC_BIND`/`OMP_PLACES`).
     **Try first:** `OMP_PROC_BIND=close OMP_PLACES=cores VLM_NUMA=4 ...`
  2. Extra `#pragma omp parallel` region inside `gemm_*_BTP_cmg` (vs implicit
     `parallel for` in the no-NUMA path) has spawn overhead the runtime doesn't
     amortize across 144 GEMM calls/encode.
     **Try:** flatten to `parallel for` with a `firstprivate` CMG index computed
     from `tid * n_cmgs / nthr` inside the loop body (avoid the explicit team).
  3. Cross-CMG bandwidth isn't the bottleneck at these sizes; `pack_A_fp32`
     (sequential, ~2.3 MB per QKV call × 144 calls ≈ 330 MB serial) dominates.
     **Try:** parallelize `pack_A_fp32`, or move it onto a per-CMG path.
  Add instrumentation: assert `cmg_self()` matches expected `my_cmg` for one
  thread per CMG on first iter, count cross-CMG remote loads.

- [ ] **B3.4 — attention with head-partition redistribute.**
  Attention is 35% of encode (the biggest single stage). M-partition can't
  parallelize attention cleanly because attention is per-head, not per-token.
  Plan:
  - All-gather Q,K,V from M-parallel to head-parallel at attention entry
    (each CMG holds 4 of 16 heads, full token dim).
  - Compute attention per-head on local CMG (sve_dot_hd → softmax → sve_axpy_hd).
  - Scatter back to M-parallel at attention exit.
  - 4-CMG barrier between layers (`cmg_barrier_wait` already exists).
  This is the most likely big-win item once B3-INVESTIGATE is resolved.

### Medium priority — algorithmic / kernel improvements

- [ ] **Flash-attention-style fused softmax+matmul.** Current attention is
  3-pass: QK → softmax → AV. A fused single-pass kernel would cut attention
  memory traffic ~3×. Worth ~10-15% on encode if attention stays at 35%.

- [ ] **Reduce ffn_down GEMM cost (21% of encode).** Largest weight matrix
  (4096×1024 per layer × 24 layers = 96 MB fp16). Candidates:
  - Better B prefetch in the asm kernel (currently no explicit PRFM).
  - K-split parallelism (current MR×NR tiles only parallelize over M×N).

- [ ] **Cache build is still slow (~30s warm pagecache, ~110s cold).**
  M5a got 1.5× from block-level OpenMP but hits a single-CMG input BW
  ceiling because the GGUF read happens on the main thread (all weight pages
  first-touch on CMG0). Investigate:
  - NUMA_DISTRIBUTE-style pread that spreads weight pages across CMGs
    before per-CMG worker threads do their deq+pack.
  - mmap with MAP_POPULATE? Or pre-pread()? Or use `mbind` MPOL_INTERLEAVE
    on the GGUF mapping before the worker loop?

### Lower priority — quality of life

- [ ] **Stage timer for NUMA paths.** `VLM_STAGE_TIMING=1` works but doesn't
  separately attribute pack_A vs microkernel vs add_bias cost. Helpful for
  the B3-INVESTIGATE work.

- [ ] **`tensor_diff` flags.** Currently exits non-zero on the first mismatch.
  An option to dump the per-tensor norm summary even on mismatch (so a single
  run shows the full diff profile) would speed up regression hunting.

- [ ] **`--bench` jitter.** First iteration is usually 5-10% slower (warm-up
  on the OMP thread team / page faults on fresh stage buffers). Either
  discard the first iter automatically or pre-warm.

- [ ] **Drop the unused unpacked-BT fallbacks.** `vit_gemm_bias_BT_fp16_unpacked_mt`
  and `vit_gemm_bias_BT_bf16_unpacked_mt` (and their `gemm_*_BT` cousins) are
  no longer on any default path. Keep only one of (unpacked C-intrinsic, asm)
  per dtype to cut binary size and confusion.

### Future / speculative

- [ ] **Q8_0 / Q4_K storage with on-the-fly dequant in the microkernel.**
  Cuts weight memory to ~25% of fp16. A64FX has no native int4/int8 ops, but
  an LD1H+SHIFT+FMA path similar to BF16's LSL#16 could work for Q8_0 by
  treating it as bf16-like (scale per block of 32).

- [ ] **Half-precision accumulators in FMA.** Risk: long K-dim accumulator
  drift. M4 chose FP32 accum specifically to match the CPU reference
  bit-for-bit. Could be a runtime flag for inference-only contexts where
  ~1e-3 relative error is acceptable.

- [ ] **Image preprocessor on A64FX.** `--image-size` currently synthesizes
  the test image on the host; production needs JPEG/PNG decode + resize +
  mean/std normalize. Cheap to port but not yet wired.

- [ ] **Multi-image batched encode.** Current API encodes one image; LLaVA-style
  pipelines often want 4-16 images per request. The cache is already
  shareable; just need a batch dimension on the M-parallel split.

---

## Files of interest

```
include/vit_a64fx.h               — public API
src/vit_a64fx.c                   — encoder pipeline, cache build/replicate, dispatch
src/cmg_pool.{c,h}                — CMG NUMA primitives (mbind, pin, barrier)
src/vlm_runner.c                  — CLI + VLM_NUMA env gate
kernels/fused_gemm.c              — FP32 packed-B GEMM driver
kernels/{fp16,bf16}_gemm.c        — FP16/BF16 packed-B GEMM driver + CMG variant
kernels/micro_kernel_fp32_8x3.S   — 8x48 FP32 asm microkernel
kernels/micro_kernel_fp16B_8x3.S  — 8x48 FP16-load asm microkernel
kernels/micro_kernel_bf16B_8x3.S  — 8x48 BF16-load asm microkernel
kernels/norm_sve.c                — SVE LayerNorm batch kernel
tools/test_cmg_pool.c             — CMG NUMA sanity test (mbind + bandwidth)
tools/tensor_diff.c               — VLMD dump validator
../../common/vision_encoder.h     — bit-exact reference (CPU)
```
