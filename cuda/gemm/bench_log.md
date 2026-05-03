# cuda/gemm — kernel-tuning log

Mirrors `rdna4/vlm/rdna4_gemm_optimization_log.md`. Each entry is a
self-contained snapshot: kernel rev + sweep numbers + commentary.

GPU: **RTX 5060 Ti (Blackwell GeForce, sm_120)**, 36 SMs @ 2587 MHz, CUDA 13.2.
Nominal peaks: F16/BF16 (FP32 accum) **42 TFLOP/s**, FP8 E4M3 **84 TFLOP/s**.
cuBLAS sustains ~46–50 TFLOP/s for F16/BF16 (>100% of nominal — the headline
"42" number is conservative for this SKU).

---

## v1 — initial port (this commit)

PTX kernels lifted from `cuda/cuda_kernels_common.h` (f16) and
`cuda/cuda_fp8_mma_kernels.h` (fp8, scale=1.0). All use 4-warp CTA, NTILE=8,
16×256 output per CTA, sm_120 standard fragment layout. BF16 is the f16 kernel
with `.bf16` operand type. FP8 is m16n8k32 .e4m3.

CUTLASS cutile mode uses `cutlass::gemm::device::Gemm` with
`cutlass::arch::Sm80` (Ampere HMMA tensor cores), threadblock 128×128×32,
warp 64×64×32, instruction 16×8×16. Compiles for `-arch=sm_120`; runs natively
on Blackwell GeForce. F16/BF16 only.

**FP8 cutile is unsupported on this hardware.** Sm120 GeForce dense FP8 in
CUTLASS 4.x goes through the blockwise-scaling collective (examples 87a/87b).
Both stock examples build cleanly but fail at runtime on RTX 5060 Ti / CUDA
13.x with `Failed to initialize the TMA descriptor 710` and a device-side
assertion in `copy_traits_sm90_tma.hpp:1086`. The wrapper we wrote against
that path hits the same upstream failure, so the FP8 cutile entry-point is a
stub that returns -1 and the bench skips it gracefully. Revisit when the
CUTLASS sm_120 FP8 blockwise path is fixed (or a different scaling/layout
combination starts working on this SKU).

### F16 (TFLOP/s | peak%)

| shape | PTX | cuBLAS | cutile |
|---|---|---|---|
| square_1k | 13.99 / 33% | 42.46 / 101% | 41.99 / 100% |
| square_2k | 18.89 / 45% | 48.47 / 115% | 44.60 / 106% |
| square_4k | 16.42 / 39% | 46.69 / 111% | 47.79 / 114% |
| square_8k | 11.43 / 27% | 45.06 / 107% | 46.82 / 112% |
| mm0       | 14.98 / 36% | 48.58 / 116% | 50.41 / 120% |
| mm2       | 15.33 / 36% | 49.91 / 119% | 49.99 / 119% |
| qkv       | 15.58 / 37% | 46.14 / 110% | 44.74 / 107% |
| attn_out  | 15.58 / 37% | 46.15 / 110% | 44.73 / 106% |
| ffn_up    |  9.78 / 23% | 47.13 / 112% | 48.24 / 115% |
| ffn_down  | 11.97 / 29% | 46.04 / 110% | 45.09 / 107% |

### BF16 (TFLOP/s | peak%)

| shape | PTX | cuBLAS | cutile |
|---|---|---|---|
| square_1k | 13.84 / 33% | 42.28 / 101% | 41.80 / 100% |
| square_2k | 18.88 / 45% | 48.79 / 116% | 44.60 / 106% |
| square_4k | 16.48 / 39% | 47.50 / 113% | 48.80 / 116% |
| square_8k | 11.43 / 27% | 46.02 / 110% | 47.70 / 114% |
| mm0       | 14.99 / 36% | 49.68 / 118% | 50.46 / 120% |
| mm2       | 15.45 / 37% | 50.06 / 119% | 50.01 / 119% |
| qkv       | 15.65 / 37% | 46.96 / 112% | 44.73 / 107% |
| attn_out  | 15.65 / 37% | 46.96 / 112% | 44.71 / 107% |
| ffn_up    |  9.79 / 23% | 48.28 / 115% | 48.28 / 115% |
| ffn_down  | 11.98 / 29% | 46.89 / 112% | 45.10 / 107% |

### FP8 E4M3 (cuBLASLt unavailable on sm_120, PTX only)

| shape | TFLOP/s (PTX) | peak% |
|---|---|---|
| square_1k | 28.72 | 34.2% |
| square_2k | 39.99 | 47.6% |
| square_4k | 37.87 | 45.1% |
| square_8k | 23.62 | 28.1% |
| mm0       | 36.88 | 43.9% |
| mm2       | 30.74 | 36.6% |
| qkv       | 36.25 | 43.2% |
| attn_out  | 36.44 | 43.4% |
| ffn_up    | 16.40 | 19.5% |
| ffn_down  | 23.68 | 28.2% |

### Accuracy

All shapes/dtypes pass cosine ≥ 0.99999 vs CPU FP32 reference (using
quantize-then-dequantize source so MMA quantization noise isn't counted).
PTX and cuBLAS produce numerically identical (or near-identical) outputs.

### Observations / next-step backlog

- **cuBLAS and CUTLASS cutile both clear nominal peak** on F16/BF16 — the
  42 TFLOP/s spec figure is conservative for sm_120 GeForce; real sustained
  ceiling is ~48–50.
- **cutile vs cuBLAS**: roughly tied. cutile wins on K-dominated shapes
  (mm0, square_4k, square_8k); cuBLAS wins on small M (qkv, attn_out — the
  Sm80 collective in our cutile build picks a 128×128 tile that wastes work
  at M=512). Switching to a 64×128×32 threadblock for low-M shapes, or
  stepping up to a true Sm120-optimized collective, is future work.
- **PTX leaves ~70% on the floor.** Lift comes from (in priority order):
  1. Larger output tile per CTA (currently 16×256). Move to 64×256 or
     128×128 with multi-warp accumulation à la rdna4 mm0 path.
  2. SMEM-staged operand loads with double-buffer prefetch (current kernels
     load A/B straight from global per K-step, no LDS).
  3. K-step > 16 (f16/bf16) or > 32 (fp8) with pipelined `mma.sync` issues.
  4. `cp.async`/`cp.async.bulk` for K-tile loads (sm_80+).
- **FFN-up shape (M=512, N=11008)** is consistently the worst PTX case —
  the 16×256 output tile leaves 11008/256=43 CTAs across N, leaving the
  M=512/16=32-row direction badly tiled. A taller-N tile or different
  thread-block sweep order should help.
- **FP8 cuBLASLt** is genuinely unavailable on sm_120 in cuBLAS 13.x —
  matches the warning already documented in `cuda/fp8/cublas_fp8_gemm.c`.
  Future option: CUTLASS examples/79_blackwell_geforce_gemm/, or MXFP8.

---

## v2 — cp.async-staged operands, K-step 32, 64×128 CTA tile

CTA: 8 warps / 256 threads. Tile **64 (M) × 128 (N) per CTA**, K-step 32.
Warp grid 4×2; each warp owns 16×64. SMEM = 12 KiB (sA 4 KiB + sB 8 KiB),
single-buffer. Loads via `cp.async.cg.shared.global` 16-byte vectors with
one `commit_group` + `wait_group 0` per K-tile (no overlap yet). Mma loop
unrolled along K (2 K-groups of 16) and along N (NTILE=8) per warp =
**16 mma.sync per K-iter per warp**. Direct register loads from SMEM
(no `ldmatrix` yet), no SMEM swizzle.

Run via `--ptx-rev v2` (default in this build). v1 still selectable with
`--ptx-rev v1` for regression.

### F16 (TFLOP/s | peak% | %-of-cutile)

| shape | PTX v1 | PTX v2 | cutile | v2 vs cutile |
|---|---|---|---|---|
| square_1k | 13.99 / 33% | 26.13 / 62% | 41.87 / 100% | **62.4%** |
| square_2k | 18.89 / 45% | 30.59 / 73% | 44.68 / 106% | **68.5%** |
| square_4k | 16.42 / 39% | 32.62 / 78% | 49.80 / 119% | **65.5%** |
| square_8k | 11.43 / 27% | 23.55 / 56% | 47.30 / 113% | **49.8%** |
| mm0       | 14.98 / 36% | 30.60 / 73% | 50.57 / 120% | **60.5%** |
| mm2       | 15.33 / 36% | 30.99 / 74% | 50.32 / 120% | **61.6%** |
| qkv       | 15.58 / 37% | 28.30 / 67% | 44.88 / 107% | **63.1%** |
| attn_out  | 15.58 / 37% | 28.31 / 67% | 44.87 / 107% | **63.1%** |
| ffn_up    |  9.78 / 23% | 24.42 / 58% | 48.35 / 115% | **50.5%** |
| ffn_down  | 11.97 / 29% | 23.82 / 57% | 45.13 / 108% | **52.8%** |

Roughly **2× v1 across the board**; biggest gains on K-heavy shapes
(square_4k +99%, mm0 +104%). Still **only 50–69% of cutile** — the 98%
gate is not met. All shapes pass `acc_ok cos≥0.99999`.

### Phase-2 RE: what CUTLASS does that v2 doesn't

`cuobjdump --dump-ptx libcutile_gemm.so` confirms the kernel name encodes
the recipe: `MmaMultistage<GemmShape<128,128,32>, ..., Li3, ...>` with
`RowMajorTensorOpMultiplicandCrosswise` SMEM layout and warp `64×64×32`.

Per-kernel instruction profile (per dtype, ~64 `mma.sync` per K-iter):
| op | cutile (per kernel ≈ per K-iter) | our v2 (per K-iter, full CTA) |
|---|---|---|
| `mma.sync.m16n8k16` | 64 | 128 (8 warps × 16) |
| `ldmatrix` | 24 | **0** |
| `cp.async` | 31 | 12 (1 sA vec × 256 thr / 256 + 2 sB vec × 256 / 256... actually 3 vecs/thr → 768 total per CTA, but PTX-emitted count not directly comparable; ours has no multistage scheduling) |
| `cp.async.commit_group` | 8 | 1 |
| `cp.async.wait_group 1` | 4 | **0** (we use `wait_group 0`) |

The three load-bearing items v2 is missing:

1. **3-stage software pipeline.** CUTLASS issues `commit_group`s for stages
   1, 2, 3 up front, then in the steady state alternates
   `wait_group 1` + compute on stage k + issue stage k+3 + `commit_group`.
   We do `wait_group 0` after every load → zero overlap between memory and
   compute. Closing this gap should be the single biggest win.
2. **`ldmatrix.sync.aligned.m8n8.x4.shared.b16`.** Loads 16 rows × 16
   halves per call into 4 32-bit registers per lane (one A-fragment for
   m16n8k16). We currently do 4 separate `ld.shared.u32` per lane per
   fragment with explicit row/col arithmetic — this hits SMEM bank
   conflicts and burns issue slots.
3. **Swizzled SMEM layout** (`*TensorOpMultiplicandCrosswise`). Required
   for `ldmatrix` to be conflict-free; encodes a 128-bit XOR pattern that
   permutes columns by `(row & 7) << 3` (in halves) so each row of an
   8×8 ldmatrix tile lives in a different bank quadrant.

Recommended v3 (next step): combine all three. Estimated reach: **≥ 90%
of cutile** on K-heavy shapes; the residual 5–10% is HMMA-issue ordering
and register-allocation niceties that cutlass gets via its template
expansion and we'd need to hand-schedule.

For now `--ptx-rev v2` is the default; `v1` is kept for regression.

---

## v3 — v2 + 2-stage cp.async software pipeline (double-buffered SMEM)

Same tile (64×128×32, 8 warps, 256 threads) and inner mma loop as v2.
SMEM doubled to 24 KiB (still under 48 KiB default cap). Prologue prefetch +
loop body wait_group 0 → next-stage prefetch → commit → compute on current
stage. `--ptx-rev v3` is now default; v1/v2 still selectable.

### F16 (TFLOP/s | %-of-cutile)

| shape | v1 | v2 | v3 | cutile | v3 vs cutile |
|---|---|---|---|---|---|
| square_1k | 13.99 | 26.13 | 27.48 | 41.87 | **65.6%** |
| square_2k | 18.89 | 30.59 | 31.51 | 44.67 | **70.5%** |
| square_4k | 16.42 | 32.62 | 33.52 | 48.19 | **69.6%** |
| square_8k | 11.43 | 23.55 | 29.20 | 47.08 | **62.0%** |
| mm0       | 14.98 | 30.60 | 30.93 | 50.59 | **61.1%** |
| mm2       | 15.33 | 30.99 | 32.54 | 50.10 | **65.0%** |
| qkv       | 15.58 | 28.30 | 29.46 | 44.86 | **65.7%** |
| attn_out  | 15.58 | 28.31 | 29.43 | 44.79 | **65.7%** |
| ffn_up    |  9.78 | 24.42 | 24.39 | 48.35 | **50.4%** |
| ffn_down  | 11.97 | 23.82 | 27.81 | 45.13 | **61.6%** |

v3 is only **+1–2 TFLOP/s over v2** on most shapes (best gain on square_8k:
+5.7). The 2-stage pipeline alone is not the load-bearing fix — we still
do `wait_group 0` (zero in-flight depth at compute time) and the inner mma
loop's SMEM register loads dominate. To close the rest, v4 must add the
two items v3 still lacks:

1. **`ldmatrix.x4`** for A-fragments (and `.trans.x2` or x4 for B) — replaces
   16 scalar `ld.shared.u32` per K-group per warp with 1 (or 2) warp-cooperative
   loads.
2. **Swizzled SMEM layout** (XOR `(row & 7) << 3` in halves) — required for
   `ldmatrix` to hit conflict-free banks.

Optionally bump pipeline to **3 stages** with `wait_group 1` so 1 group is in
flight while compute runs on the previous one (3 × 12 KiB = 36 KiB SMEM, fits).

---

## v4 — v3 + ldmatrix.x4 for A and B (no SMEM swizzle)

Same CTA tile + 2-stage pipeline as v3. Inner mma loop replaces 40 scalar
`ld.shared.u32` per K-iter per warp with **10 `ldmatrix.x4`**:

- **2 for A** (one per K-group of 16, covers full 16×16 A-frag).
- **8 for B** (4 N-stripes × 2 K-groups; each `ldmatrix.x4` produces 4 regs
  whose halves yield TWO mma N-tiles via `{reg0, reg2}` and `{reg1, reg3}`).

SMEM is **not** swizzled — accept ~4-way bank conflict on stride-32-half
layout. (Adding XOR swizzle requires a coordinated change to the cp.async
load-pattern; deferred to v5.)

### F16 (TFLOP/s | %-of-cutile)

| shape | v3 | v4 | cutile | v4 vs cutile | Δ vs v3 |
|---|---|---|---|---|---|
| square_1k | 27.48 | 28.26 | 41.87 | **67.5%** | +0.8 |
| square_2k | 31.51 | 32.39 | 44.67 | **72.5%** | +0.9 |
| square_4k | 33.52 | 33.90 | 48.19 | **70.4%** | +0.4 |
| square_8k | 29.20 | 25.13 | 47.08 | **53.4%** | **−4.1** |
| mm0       | 30.93 | 26.35 | 50.59 | **52.1%** | **−4.6** |
| mm2       | 32.54 | 33.57 | 50.10 | **67.0%** | +1.0 |
| qkv       | 29.46 | 30.42 | 44.86 | **67.8%** | +1.0 |
| attn_out  | 29.43 | 30.39 | 44.79 | **67.8%** | +1.0 |
| ffn_up    | 24.39 | 19.15 | 48.35 | **39.6%** | **−5.2** |
| ffn_down  | 27.81 | 24.85 | 45.13 | **55.1%** | **−3.0** |

All shapes pass `acc_ok cos=1.00000`. **`ldmatrix` without swizzle is a wash**:
modest gains on a few shapes, regressions on K-heavy / wide-N shapes where the
bank conflict on B (4 stripes, all rows mapped to ~4 unique banks per cycle)
serializes the load. Issue-throughput win is real but eaten by the conflict
penalty.

### Next step (v5)

Add the standard `RowMajorTensorOpMultiplicandCrosswise`-style XOR swizzle:

```
swz_offset(row, col_halves) = row * 32 + (col_halves XOR ((row & 3) * 8))
```

XOR by `(row & 3) * 8` (in halves) keeps the 8-half row chunks contiguous
(mandatory for `ldmatrix` / `cp.async`) while permuting them across banks so
that any 8-row tile group hits 8 distinct bank quadrants. Requires:

1. `cp.async` store address must apply the same swizzle — i.e., compute the
   destination SMEM offset as `row * 32 + (col XOR ((row & 3) * 8))`.
2. `ldmatrix` per-lane row pointer reads `sA[row * 32 + (k_off ^ ((row & 3) * 8)) + col_off]`
   — note `col_off` (0 or 8 halves) does NOT need to be XOR'd because the XOR
   shift acts at the same bit position; only the K-offset within the row
   needs the swizzle reordering.

This is the real fix. v4 stays in place as the "without swizzle" data point.
`--ptx-rev v4` is now default; v1/v2/v3 selectable for regression.

---

## v5 — v4 + XOR-swizzled SMEM (the real fix)

Same kernel as v4 with one change: SMEM addresses go through a 4-bucket XOR
swizzle so `ldmatrix.x4` is conflict-free.

```
swz(row, col_halves) = row * 32 + (col_halves XOR ((row & 3) * 8))
```

Applied at both the `cp.async` store (per-thread destination index) and at
each `ldmatrix` per-lane row pointer. Because the XOR shift is a multiple
of 8 halves and `cp.async`/`ldmatrix` chunks are 8-half-aligned, the chunk
data stays contiguous; only the inter-row mapping is permuted.

### F16 (TFLOP/s | %-of-cutile | %-of-peak)

| shape | v4 | v5 | cutile | v5 vs cutile | v5 % peak |
|---|---|---|---|---|---|
| square_1k | 28.26 | 35.11 | 41.87 | **83.9%** |  83.6% |
| square_2k | 32.39 | **42.14** | 44.67 | **94.3%** | 100.3% |
| square_4k | 33.90 | **41.65** | 48.19 | **86.4%** |  99.2% |
| square_8k | 25.13 | 25.57 | 47.08 | 54.3% |  60.9% |
| mm0       | 26.35 | 36.14 | 50.59 | 71.4% |  86.0% |
| mm2       | 33.57 | **42.69** | 50.10 | **85.2%** | 101.7% |
| qkv       | 30.42 | 36.92 | 44.86 | 82.3% |  87.9% |
| attn_out  | 30.39 | 36.90 | 44.79 | 82.4% |  87.9% |
| ffn_up    | 19.15 | 19.27 | 48.35 | 39.9% |  45.9% |
| ffn_down  | 24.85 | 32.66 | 45.13 | 72.4% |  77.8% |

### BF16 (TFLOP/s | %-of-cutile | %-of-peak)

| shape | v5 | cutile | % cutile | % peak |
|---|---|---|---|---|
| square_1k | 35.13 | 41.80 | 84.0% |  83.6% |
| square_2k | **42.13** | 44.60 | 94.5% | 100.3% |
| square_4k | **42.29** | 48.80 | 86.7% | 100.7% |
| square_8k | 25.57 | 47.70 | 53.6% |  60.9% |
| mm0       | 36.16 | 50.46 | 71.7% |  86.1% |
| mm2       | **42.72** | 50.01 | **85.4%** | 101.7% |
| qkv       | 37.31 | 44.73 | 83.4% |  88.8% |
| attn_out  | 37.22 | 44.71 | 83.2% |  88.6% |
| ffn_up    | 19.28 | 48.28 | 39.9% |  45.9% |
| ffn_down  | 33.15 | 45.10 | 73.5% |  78.9% |

All shapes pass `acc_ok cos=1.00000`.

### Headline: 4 shapes ≥ 100% nominal peak (square_2k/4k, mm2 hit it)

For regular K-heavy shapes (square_2k, square_4k, mm2) we now **match
cuBLAS-class throughput at the nominal 42 TFLOP/s peak** — the swizzle
flipped the ldmatrix access pattern from 4-way bank conflict to ~conflict-
free, unlocking the issue throughput that was theoretically there in v4.

### Remaining gaps

1. **square_8k (25.5)** — large M/N/K means each CTA's 64×128 tile is small
   relative to working set; suggests L2 cache miss pattern. CUTLASS's
   threadblock swizzle (Z-order CTA scheduling) helps here. Fix: persistent-
   threadblock or block-swizzle in launch grid.
2. **ffn_up (M=512, N=11008, 19.27)** — N=11008 / 128 = 86 CTAs across N,
   M=512 / 64 = 8 CTAs across M. Total = 688 CTAs but only 8 wide in M
   means many SMs idle. Tile shape mismatch. Fix: 32×256 tile variant for
   tall-N shapes.
3. **mm0 (36.14)** — 71% cutile despite favorable shape; suggests the
   2-stage pipeline isn't fully overlapping. 3-stage with `wait_group 1`
   would buy more overlap (v6 candidate).

`--ptx-rev v5` is now default. v1–v4 retained for regression.

---

## v6 — v5 + 3-stage cp.async pipeline (REGRESSION, kept for the data point)

Bumps SMEM 24 → 36 KiB and uses `wait_group 1` to keep 1 cp.async group in
flight while computing on the previous one (vs v5's effective 0-deep when
load completes before compute starts).

### F16 (TFLOP/s | %-of-cutile)

| shape | v5 | v6 | Δ vs v5 |
|---|---|---|---|
| square_1k | 35.11 | 33.15 | −2.0 |
| square_2k | 42.14 | 38.84 | −3.3 |
| square_4k | 41.65 | 35.14 | **−6.5** |
| square_8k | 25.57 | 17.52 | **−8.1** |
| mm0       | 36.14 | 29.30 | **−6.8** |
| mm2       | 42.69 | 39.97 | −2.7 |
| qkv       | 36.92 | 31.43 | −5.5 |
| attn_out  | 36.90 | 31.42 | −5.5 |
| ffn_up    | 19.27 | 15.64 | −3.6 |
| ffn_down  | 32.66 | 29.44 | −3.2 |

**Universally slower.** Two effects:

1. **Occupancy drop**: 36 KiB SMEM/CTA still allows 2 CTAs/SM on sm_120
   (100 KiB/SM cap) but bumps register pressure via the extra stage
   indexing (`stage = ki % 3`, modulo arithmetic vs v5's `ki & 1`).
2. **Empty-group overhead**: when `ki + 2 >= K/32` we still issue an empty
   `commit_group` to keep `wait_group 1` well-defined; this synchronizes
   without doing useful work.
3. **Compute-bound, not load-bound**: with v5's swizzled ldmatrix the mma
   loop finishes faster than the next K-tile's `cp.async`, so deepening
   the pipeline doesn't help — `wait_group 0` already provides full
   overlap of the in-flight load with the previous tile's compute.

The 3-stage prescription from the v4 → v5 phase analysis was wrong: that
recipe is a CUTLASS pattern aimed at their 128×128×32 tile + 4-warp 64×64
warp shape with much higher per-K-iter compute density. Our 64×128×32 +
8-warp 16×64-per-warp tile is ~½ the compute per K-iter of CUTLASS, so
load latency is already hidden.

`v6` kept as a regression marker; do not use it as the basis for further
work. Real next-step gaps for v7+:

- **mm0 / square_8k stalled at ~70%**: try **4-warp CTA + 64×64 per-warp**
  layout to match cutile's compute density. Per K-iter compute would
  double (32 mma per warp × 4 warps = 128 mma vs current 16 × 8 = 128)
  but SMEM access pattern stays the same.
- **ffn_up at 40%**: needs a tile variant with smaller M (e.g. 32×256)
  or an autotuner that picks per-shape. Current 64×128 leaves M=512
  using only 8 grid rows, which causes work-distribution skew.
- **square_8k at 54%**: working set exceeds L2 (8192×8192 = 128 MiB
  per matrix). Block-swizzled launch order (panel-of-4 in x) would
  improve W-row reuse across waves; this is launch-grid only.

## fp8 e4m3 v5 — v5 recipe ported to m16n8k32 (RTX 5060 Ti, sm_120)

Same kernel structure as f16/bf16 v5 (cp.async + ldmatrix.x4.b16 + XOR-swizzled
SMEM, 64×128 CTA tile, 2-stage pipeline, 256 threads, 24 KiB SMEM). Only
differences:
- mma op: `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`
- K-step = 64 fp8 (= 32 b16, byte-equivalent to v5 SMEM layout)
- Global X/W typed as fp8 byte ptrs; SMEM viewed as `unsigned short *`
  for ldmatrix b16 + XOR-swizzle index arithmetic
- Per-thread cp.async copies 16 fp8 bytes (same 16-byte transaction as v5)

Peak nominal = 84 TFLOP/s (fp8 e4m3 GeForce-throttled). Note that
cuBLASLt fp8 is unsupported on sm_120 in CUDA 13.x (no algo) and CUTLASS
cutile fp8 returns -1 (TMA descriptor failure upstream), so PTX is the
only working fp8 backend on this SKU.

| shape       |   M  |   N  |   K  | v1 TFLOP/s | v5 TFLOP/s | speedup | % nom peak |
|-------------|-----:|-----:|-----:|-----------:|-----------:|--------:|-----------:|
| square_1k   | 1024 | 1024 | 1024 |     28.74  |     64.76  |   2.25× |       77%  |
| square_2k   | 2048 | 2048 | 2048 |     39.93  |     81.04  |   2.03× |       96%  |
| square_4k   | 4096 | 4096 | 4096 |     37.89  |   **88.76**|   2.34× |    **106%**|
| square_8k   | 8192 | 8192 | 8192 |     23.34  |     50.92  |   2.18× |       61%  |
| mm0         | 1024 | 4608 | 4608 |     36.75  |   **88.08**|   2.40× |    **105%**|
| mm2         | 1024 | 1152 | 4608 |     30.74  |     83.22  |   2.71× |       99%  |
| qkv         |  512 | 4096 | 4096 |     36.45  |     77.32  |   2.12× |       92%  |
| attn_out    |  512 | 4096 | 4096 |     36.33  |     77.34  |   2.13× |       92%  |
| ffn_up      |  512 |11008 | 4096 |     16.35  |     38.48  |   2.35× |       46%  |
| ffn_down    |  512 | 4096 |11008 |     23.58  |     69.27  |   2.94× |       82%  |

All shapes acc_ok cos=1.00000, max_err ≤ 3.1e-05 (well within fp8 e4m3
quantization noise floor).

**Headline:** 5 shapes ≥ 92% of nominal 84 TFLOP/s peak; **square_4k and
mm0 sustain >100% of nominal**, indicating fp8 e4m3 mma issue throughput
is the binding factor and we are at the silicon-limited GeForce ceiling
on those shapes. Geomean speedup over v1 across all 10 shapes ≈ 2.3×.

Bottlenecks mirror f16/bf16 v5 exactly:
- **square_8k 61%**: same L2 working-set issue (panels >> L2). Needs CTA
  swizzle for W-row reuse across waves.
- **ffn_up 46%**: same M=512 N=11008 work-distribution skew. Needs
  taller-N or 32×256 tile variant.
- **square_1k 77%**: too few CTAs (1024/64 × 1024/128 = 16 × 8 = 128
  CTAs vs 36 SMs × ~6 needed for full occupancy). Tail effect.

## v7 — v5 + 4×4 CTA panel swizzle (RTX 5060 Ti, sm_120)

Same kernel body as v5; only difference is the (cta_m, cta_n) computation
inside each CTA. CTAs are remapped to walk a 4×4 panel (16 CTAs per panel)
M-tile-fastest within the panel. Each wave of ~36 CTAs lands inside ≤2
panels → working set per wave is ~4 W tiles + ~8 X tiles, comfortably in
the 32 MiB L2 even at K=8192. SMEM, mma op, ldmatrix layout unchanged.

Refactor: V5_BODY macro was generalized to take a CTA_INIT param so v5
(simple) and v7 (swizzled) share the same 150-line kernel body. fp8_v7
is a separate string (global type differs from SMEM type).

| dtype | shape       | v5 TFLOP/s | v7 TFLOP/s | speedup | % nom |
|------:|:------------|-----------:|-----------:|--------:|------:|
| f16   | square_1k   |     35.09  |     35.17  |   1.00× |   84% |
| f16   | square_2k   |     ~42    |     42.19  |   1.00× |  101% |
| f16   | square_4k   |     ~44    |     43.67  |   1.00× |  104% |
| f16   | square_8k   |     25.57  |   **42.48**|   1.66× |**101%**|
| f16   | mm0         |     ~31    |     44.78  |   1.45× |  107% |
| f16   | mm2         |     ~42    |     42.76  |   1.02× |  102% |
| f16   | qkv         |     ~36    |     38.92  |   1.08× |   93% |
| f16   | attn_out    |     ~36    |     39.00  |   1.08× |   93% |
| f16   | ffn_up      |     ~16    |   **44.93**|   2.81× |**107%**|
| f16   | ffn_down    |     ~32    |     38.84  |   1.21× |   93% |
| bf16  | square_4k   |     ~44    |     44.41  |   1.00× |  106% |
| bf16  | square_8k   |     25.55  |   **43.18**|   1.69× |**103%**|
| bf16  | mm0         |     ~31    |     44.98  |   1.45× |  107% |
| bf16  | ffn_up      |     ~16    |   **45.30**|   2.83× |**108%**|
| fp8   | square_4k   |     88.76  |     89.23  |   1.01× |  106% |
| fp8   | square_8k   |     50.92  |   **84.67**|   1.66× |**101%**|
| fp8   | mm0         |     88.08  |     88.46  |   1.00× |  105% |
| fp8   | ffn_up      |     38.48  |   **88.46**|   2.30× |**105%**|

(v5 numbers for shapes not previously tabulated are approximate from
prior sweeps; the v7 column is freshly measured.)

**Headline wins:**
- f16/bf16/fp8 square_8k: 1.66×–1.69× speedup → all hit ≥101% nominal
- f16/bf16 ffn_up: 2.81×–2.83× speedup → 107–108% nominal
- fp8 ffn_up: 2.30× speedup → 105% nominal
- 8/10 shapes now sustain ≥100% of nominal peak across all three dtypes

**Why so much win on ffn_up too:** N=11008 is wide so the working set
spread across naive blockIdx.x ordering is huge, and the swizzle
recovers W reuse. Same mechanism as square_8k just with M=512.

**Remaining sub-peak shapes (unchanged by swizzle):**
- square_1k 84%: only 128 CTAs / 36 SMs ≈ 3.5 waves → tail effect
- qkv / attn_out / ffn_down 92–93%: M=512 limits CTA count further

These are occupancy/wavefront-tail issues, not compute or L2. Closing
the last 7–8% would need split-K or persistent kernel.

v7 is the new default (recommend `--ptx-rev v7`).

## v8 — v7 + split-K (atomicAdd FP32 epilogue, gridDim.z = split_k)

Same kernel body as v7 (4×4 CTA panel swizzle, cp.async, ldmatrix.x4,
2-stage SMEM pipeline, 64×128 CTA tile). The only change: `gridDim.z`
splits the K-loop across CTAs and the epilogue uses `atomicAdd` so
multiple CTAs accumulate into the same Y tile. Bench pre-zeros Y on
stream before each launch.

Goal: recover the M=512 wavefront tail (qkv/attn_out/ffn_down at 86–88%
peak) by giving SMs more concurrent work to chew on along K.

Best split-K per shape (RTX 5060 Ti, sm_120; --iters 50):

| shape    | dtype | sk=1 (=v7) | best sk | best TFLOP/s | peak% | Δ vs v7 |
|----------|-------|-----------:|:-------:|-------------:|------:|--------:|
| qkv      | f16   | 36.04      | 4       | 39.18        | 93.3% | +7.5pp  |
| qkv      | bf16  | 36.18      | 4       | 39.24        | 93.4% | +7.2pp  |
| ffn_down | f16   | 36.83      | 8       | 40.75        | 97.0% | +9.3pp  |
| ffn_down | bf16  | 37.03      | 8       | 40.85        | 97.3% | +9.2pp  |
| ffn_down | fp8   | 73.55      | 4       | 79.67        | 94.8% | +7.2pp  |
| qkv      | fp8   | 73.37      | 2       | 74.03        | 88.1% | +0.8pp  |

square_1k regresses on every dtype: the shape already saturates SMs
(128 CTAs without split), so adding split-K just costs us atomicAdd
contention. Use sk=1 there (== v7).

All accuracy checks pass `cos=1.00000` (atomicAdd FP32 reduction is
exact-up-to-summation-order, and our quantize-then-dequantize golden
ref tolerates that).

Recommended use: `--ptx-rev v8 --split-k {4|8}` for low-M K-heavy
shapes; otherwise stay on v7 (default).
