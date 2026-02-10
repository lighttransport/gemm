# Fused GEMM and Fused Attention on A64FX

Hand-tuned GEMM micro-kernels and fused attention pipeline for
Fujitsu A64FX (Fugaku), targeting single-core fp16 performance on
512-bit SVE.

## Hardware

- **A64FX** @ 2.0 GHz, 512-bit SVE (VL=64 bytes)
- 2 FP pipes: fp32 peak = 128 GFLOPS, fp16 peak = 256 GFLOPS
- L1D 64 KB (4-way, 256-byte lines, 64 sets), L2 8 MB
- In-order pipeline, 4-wide decode

## What It Computes

### Fused GEMM: (Q @ K^T) @ V

Pure double matrix multiplication without softmax. The intermediate
S = Q @ K^T is never fully materialized — instead, tiles of S are
computed and immediately consumed by the second GEMM.

```
E[M,N] = (A[M,K1] @ B[K1,K2]) @ C[K2,N]
```

Implementation: `fused_gemm.c`, `bench_fused.c`. Uses L2-resident
intermediate tiles. Target: **90%+ of peak** GEMM efficiency for
large problem sizes.

### Fused Attention: softmax(S) @ V

Adds softmax between the two GEMMs for full attention:

```
O[M,d] = softmax(S[M,L]) @ V[L,d]
```

where `softmax(S)_ij = exp2(S_ij - max_i(S)) / sum_j exp2(S_ij - max_i(S))`.

Implementation: `bench_2pass.c`. Takes pre-computed S (attention scores)
and V (value matrix), applies fused softmax + GEMM.

- S: attention scores (fp32, column-major, M x L)
- V: value matrix (fp32, row-major, L x d), pre-packed into fp16 panel format
- O: output (fp16 or fp32, row-major, M x d)
- `exp2` via A64FX FEXPA instruction (~7-bit accuracy, single-cycle)

Typical parameters: M=12 (MR), L=4096..32768, d=256, NR=64 (2 SVE vectors).

## Fused Attention Pipeline Strategies

### 2-Pass (Interleaved)

The baseline fused pipeline:

1. **Phase 0 (max-find):** Scan all L columns of S to find per-row max. SVE `FMAX` reduction, ~68 us for L=32768.
2. **Phase 1+2 (interleaved):** Pipelined producer-consumer with ring buffer:
   - Producer: `exp2(S - max)` via FEXPA, convert to fp16, write to Pp ring slot
   - Consumer: GEMM `O += Pp @ Vp` using hand-coded SVE micro-kernel
   - `pipe_ahead` controls how many blocks the producer leads
3. **Phase 3 (normalize):** `O /= row_sum` using SVE fp32-widening multiply

The interleaving keeps Pp in L1 between producer and consumer, but at
large L the pass1 memory traffic (S reads from L2, Pp writes) competes
with GEMM's B streaming, polluting L1 and degrading GEMM throughput.

### 3-Pass (Separated)

Eliminates L1 pollution by fully separating softmax from GEMM:

1. **Phase 0:** max-find (same as 2-pass)
2. **Phase 1:** Compute ALL Pp blocks sequentially, writing to an L2 buffer (L x MR x 2 bytes, e.g. 786 KB for L=32768)
3. **Phase 2:** Pure GEMM over all k-blocks with clean L1. Pp streams from L2 via hardware prefetcher. A re-touch at tile 2 (after 2 x 32 KB B tiles fill L1, the 6 KB A panel may be evicted)
4. **Phase 3:** normalize

### Online (FlashAttention-style)

Single-pass with running max and O rescaling:

1. For each k-block: find local max, update global max
2. If max changed: rescale O and row_sum by `exp2(old_max - new_max)`
3. Compute Pp, immediately run GEMM (Pp stays in L1)
4. Normalize at end

Eliminates the max-find pass entirely, but the O rescaling cost
(fp32-widening load/multiply/narrow-store over M x d elements per max
change) dominates with random data where max changes frequently.

## Micro-Kernels

All kernels are hand-written SVE assembly (.S files). Naming convention:
`micro_kernel_{fp32,fp16}_{MR}x{NR_vecs}[_variant][_accum].S`

### FP32 Kernels (C accumulated in fp32)

| Name | Shape | Notes |
|------|-------|-------|
| `fp32_8x3` | 8 x 3vec (8x48) | 3 B-vector variant |
| `fp32_6x4_bcast` | 6 x 4vec (6x64) | BCAST lane approach |
| `fp32_10x2` | 10 x 2vec (10x32) | |
| `fp32_11x2` | 11 x 2vec (11x32) | |
| `fp32_12x2` | 12 x 2vec (12x32) | Baseline |
| `fp32_12x2_swp` | 12 x 2vec (12x32) | Software-pipelined loads |

### FP16 Mixed-Precision Kernels (A,B in fp16, C accumulated in fp32)

These kernels compute FMLA in fp16 but accumulate C in fp32 registers.
An epilogue converts fp32 C back to fp16 after the K-loop.

| Name | Shape | Notes |
|------|-------|-------|
| `fp16_12x2_swp` | 12 x 2vec (12x64) | SWP loads, fp32 C accumulation |
| `fp16_12x2_csplit` | 12 x 2vec (12x64) | C-split variant |
| `fp16_12x2_dswp` | 12 x 2vec (12x64) | Double-SWP |
| `fp16_12x2_4k` | 12 x 2vec (12x64) | 4K-unrolled |
| `fp16_12x2_bpre` | 12 x 2vec (12x64) | B-prefetch variant |
| `fp16_8x3` | 8 x 3vec (8x96) | |
| `fp16_6x4` | 6 x 4vec (6x128) | |

### FP16 NOEPI Kernels (No Epilogue — A,B,C all in fp16)

NOEPI = No Epilogue. These kernels accumulate C directly in fp16
registers, eliminating the fp32→fp16 conversion epilogue. Fewer
instructions per kernel call, but lower numerical precision than
the mixed-precision variant.

| Name | Shape | Notes |
|------|-------|-------|
| `fp16_12x2_noepi` | 12 x 2vec (12x64) | Baseline NOEPI |
| `fp16_12x2_noepi4` | 12 x 2vec (12x64) | 4K-unrolled (K%4==0) |
| `fp16_12x2_noepi4_prfm` | 12 x 2vec (12x64) | 4K-unrolled + PRFM (experimental) |

Each kernel has `_init` (writes C) and `_accum` (loads+adds C) variants.

Kernel interface:
```c
void micro_kernel(const T *A, const T *B, T *C,
                  int64_t K, int64_t unused, int64_t ldc_bytes);
```

A is column-major (K x MR), B is panel-major (K x NR), C is row-major (MR x ld_o).

### Kernel Efficiency Analysis (noepi4, 12x2)

Per 4K unrolled iteration:
- 96 FMLA instructions (24 per K step, each 32-wide fp16 = 64 FLOPS)
- 48 `ld1rh` (A broadcast loads) + 8 `ld1h` (B vector loads) = 56 loads
- FMA-limited: 96 FMLA / 2 pipes = 48 cycles
- Theoretical: 6144 FLOPS / 48 cycles = 128 FLOPS/cycle = 256 GF (100%)

Measured GEMM-only: **85% of peak** (218 GF). The gap comes from:
- **Load-use stalls:** Each K step starts with 6 consecutive `ld1rh` loads.
  On A64FX's in-order pipeline (~8-cycle load latency), the first FMLA
  stalls waiting for the first loaded register. The SWP kernel variant
  pre-loads next-K's A data during current-K's FMAs to hide this latency.
- **Tile overhead:** Function call, epilogue (accum: 24 ld+fadd+st), and
  k-block/n-tile loop overhead amortized over only M=12 rows.
- **A re-touch:** At n-tile boundaries where B tiles evict A from L1.

A software-pipelined NOEPI kernel (loading next-K A during batch-2
FMAs) would close the gap toward 90%+.

## L1 Working Set (fp16 NOEPI, Kc=256)

| Component | Size | Description |
|-----------|------|-------------|
| A (Pp block) | MR x Kc x 2 = 6 KB | Softmax output, col-major |
| B (Vp tile) | Kc x NR x 2 = 32 KB | Pre-packed V panel |
| C (O tile) | MR x NR x 2 = 1.5 KB | Output accumulator |
| **Total** | **~39.5 KB** | Fits in 64 KB L1D |

With 4 N-tiles (d=256, NR=64): tiles 0-1 fit fresh in L1, tile 2 needs
A re-touch (B tiles 0-1 = 64 KB likely evicted A's 6 KB).

## Key Techniques

- **FEXPA** for exp2: A64FX-specific instruction, ~7-bit accuracy, single-cycle. Sufficient for softmax (downstream GEMM averages errors).
- **FZ16 (FPCR bit 19):** Flushes fp16 denormals to zero. At large L, softmax outputs are often < 6.1e-5 (fp16 min normal); without FZ16, each denormal triggers a ~100-cycle trap.
- **Ring buffer with L1 conflict padding:** 2-pass pipeline uses `pipe_ahead` ring slots with padding to avoid L1 set conflicts between slots.
- **A re-touch:** Prefetch A panel back into L1 after B tiles evict it (at tile boundary where cumulative B > 40 KB).
- **SVE fp32-widening for fp16 I/O:** Clang `_Float16*` doesn't resolve SVE `svld1` overloads. All fp16 loads use inline asm `ld1h {z.s}` (widen to fp32), stores use `fcvt z.h, p/m, z.s` + `st1h {z.s}`.

## Benchmark Results

Single-core, A64FX @ 2.0 GHz, d=256, fp16 peak = 256 GFLOPS, FZ16=1.

### GEMM-only (P@V, no softmax)

Best micro-kernel GEMM-only performance for the P@V multiply (M=12
skinny case, pre-packed fp16). This is the ceiling for the fused
attention pipeline — softmax overhead comes on top.

| Kernel | Kc | L | GEMM-only (GF) | Peak % |
|--------|-----|-------|----------------|--------|
| h12n4 (noepi4) | 256 | 4096 | 218.0 | 85.2% |
| h12n4 (noepi4) | 256 | 32768 | 216.9 | 84.7% |

Current ceiling is ~85%. A software-pipelined NOEPI kernel should
reach 90%+ by hiding load-use stalls.

### Best Fused Attention Pipeline (Kc=256)

| Pipeline | L | Total (GF) | Peak % | GEMM-only (GF) | GEMM % |
|----------|-------|-----------|--------|----------------|--------|
| 2-pass h12n4 | 4096 | 162.1 | 63.3% | 218.0 | 85.2% |
| 2-pass h12n4 | 8192 | 167.1 | 65.3% | 217.7 | 85.1% |
| 2-pass h12n4 | 16384 | 167.8 | 65.5% | 213.9 | 83.5% |
| 2-pass h12n4 | 32768 | 170.8 | 66.7% | 216.9 | 84.7% |
| **3-pass h12n4** | **4096** | **169.6** | **66.3%** | 218.0 | 85.2% |
| **3-pass h12n4** | **8192** | **175.4** | **68.5%** | 217.7 | 85.1% |
| **3-pass h12n4** | **16384** | **179.0** | **69.9%** | 213.9 | 83.5% |
| **3-pass h12n4** | **32768** | **181.1** | **70.7%** | 216.9 | 84.7% |

### Timing Breakdown (Kc=256, L=32768)

| Phase | 2-pass (us) | 3-pass (us) | Change |
|-------|------------|------------|--------|
| Phase 0+1 (softmax) | 67.7 | 166.2 | +98 (no overlap) |
| Phase 2 (GEMM) | 1100.1 | 934.4 | -166 (clean L1) |
| Phase 3 (normalize) | 10.9 | 10.9 | same |
| **Total** | **1178.7** | **1111.6** | **-67 (5.7% faster)** |

The 3-pass trades softmax-GEMM overlap for a clean L1 during GEMM.
GEMM runs 15% faster (1100 -> 934 us), more than compensating for
the un-overlapped softmax cost.

### Approaches That Didn't Work

| Approach | Result | Why |
|----------|--------|-----|
| PRFM kernel (h12np) | 87 GF at L>=16384 (vs 168 baseline) | `prfm pldl1keep` for B pollutes L1, evicts A. GEMM-only drops from 214 to 98 GF. Works fine at L<=8192. |
| Online softmax | 135 GF at L=32768 (vs 171 baseline) | O rescaling cost (fp32-widening multiply over M x d_out per max change) dominates with random data. Would help with monotonic score distributions. |

### Remaining Gap

```
GEMM-only ceiling:  218 GF (85%)   <- needs SWP NOEPI kernel for 90%+
3-pass pipeline:    181 GF (70.7%) <- softmax overhead = 19% of GEMM time
softmax overhead:   166 us (max-find + exp2 + fp16 convert)
normalize:          11 us
```

Two paths to improve:
1. **SWP NOEPI kernel:** Software-pipeline A loads to hide latency, targeting 90%+ GEMM-only (currently 85%)
2. **Fuse Q@K^T with softmax:** Compute S tiles on-the-fly from Q,K instead of reading pre-computed S from L2, eliminating the S bandwidth bottleneck

## Build

Requires Fujitsu compiler on A64FX (native).

```sh
# Build the original fused (Q@K^T)@V benchmark
make bench_fused

# Build the fused attention (softmax) benchmark
make bench_2pass

# Run (native on A64FX)
./bench_2pass

# Run fp16 section only (skip fp32 benchmarks)
FP16_ONLY=1 ./bench_2pass

# Submit as batch job on Fugaku
make submit_2pass
```

Cross-compilation from Intel host: change `CC = fcc` / `2PASS_CC = fcc` to `fccpx` in Makefile.

The bench_2pass benchmark uses `-Nclang` mode for `arm_sve.h` intrinsics and FEXPA inline assembly.

## Files

```
fused_gemm.{c,h}                       Fused GEMM (Q@K^T)@V library (fp32)
bench_fused.c                           Fused double-GEMM benchmark
pack_matrices.c                         Matrix packing routines

bench_2pass.c                           Fused attention benchmark (all softmax strategies)

micro_kernel_fp32_*.S                   FP32 SVE micro-kernels
micro_kernel_fp16_*_swp.S              FP16 mixed (fp32 C) kernels
micro_kernel_fp16_*_noepi*.S           FP16 NOEPI (fp16 C) kernels
micro_kernel_fp16_*_noepi4*.S          4K-unrolled NOEPI variants
micro_kernel_fp16_*_noepi4_prfm*.S     NOEPI4 + PRFM (experimental)

Makefile                                Build rules (fcc native / fccpx cross)
```
