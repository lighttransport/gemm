# Cross-Entropy Loss Kernel Optimization for A64FX

## Overview

SVE-optimized cross-entropy loss (forward and backward) for large vocabulary LLMs on A64FX.
Uses FEXPA for fast exp2 approximation (~0.4% relative error).

```
loss = -logits[target] + max(logits) + log(sum(exp(logits - max)))
grad[i] = softmax(logits)_i - 1{i == target}
```

Files: `a64fx/cross-entropy/{cross_entropy_sve.c, bench_cross_entropy.c, sve_math.h}`

## Performance Results (V=151936, single core, 2 GHz)

| Variant | Time (us) | Eff. B/cy (CPU) | L2 BW Util | Speedup vs Baseline |
|---------|-----------|-----------------|------------|---------------------|
| **fp32 fwd 2-pass+pf+8x** | **22.7** | 26.7 | 42% | 1.07x |
| fp32 fwd blocked | 25.3 | 24.1 | 38% | 0.97x |
| fp16 fwd (fixed) | 29.9 | 10.2 | -- | 1.13x (was broken) |
| **fp32 fwd+bwd 3-pass+pf+8x** | **44.4** | 20.5 | 32% | 1.07x |
| fp32 fwd+bwd blocked | 47.0 | 19.4 | 30% | 1.01x |

Baseline (before optimization): fp32 fwd = 24.4 us, fp32 fwd+bwd = 47.6 us, fp16 fwd = 33.8 us (broken).

Timer is 100 MHz (cntvct_el0). CPU B/cy = timer_B_cy / 20 (2 GHz / 100 MHz).

## Optimizations Applied

### 1. SW Prefetch (pldl1keep)

Added `__builtin_prefetch(ptr, 0, 3)` to all streaming passes.

- Distance: 8 cache lines = 2 KB ahead (PF_DIST_LINES=8)
- Two prefetches per 8-wide iteration (covers 512 bytes = 2 cache lines)
- Applied to pass 1 (max), pass 2 (exp+sum), pass 3 (grad), and fp16 paths

A64FX L2 latency is ~40 cy. At 25 B/cy streaming rate, 8 lines (2 KB) gives ~80 cy
lookahead — enough to cover L2 latency with margin.

### 2. 8x Unroll with Independent Accumulators

Increased from 4x to 8x unroll in pass 2 (exp+sum) and pass 3 (grad):

- 8 independent SVE sum accumulators (vs0-vs7) — better OoO scheduling
- 8x max accumulators (vm0-vm7) already existed in pass 1
- Tree reduction: 3 levels of vertical add/max (12 cy total)
- Matches A64FX's 2-wide FLA pipe with 9 cy latency (need >= 18 independent FMAs)

### 3. FP16 Path Fix

The original fp16 inner loop had a `switch(u)` statement that prevented compiler optimization:

```c
// BEFORE (broken): switch in inner loop
for (int u = 0; u < 8; u++) {
    svfloat16_t h = svld1_f16(...);
    switch (u) {  // prevents vectorization
    case 0: vm0 = svmax(..., lo); vm1 = svmax(..., hi); break;
    ...
    }
}
```

Fixed with explicit 4x fp16-vec unroll — each iteration loads 4 fp16 vectors,
unpacks to 8 fp32 vectors, updates 8 independent accumulators:

```c
// AFTER: explicit unroll, no switch
for (; i < V4; i += 4 * VL16) {
    h0 = svld1_f16(pg16, ptr + i);
    h1 = svld1_f16(pg16, ptr + i + VL16);
    h2 = svld1_f16(pg16, ptr + i + 2*VL16);
    h3 = svld1_f16(pg16, ptr + i + 3*VL16);
    // unpack each to lo/hi -> 8 fp32 vectors
    // update 8 accumulators (vm0-vm7 or vs0-vs7)
}
```

Result: 33.8 us -> 29.9 us (1.13x). Still slower than fp32 due to fcvt+unpack overhead.

### 4. Blocked Forward (1 L2 pass)

Fuses max-find and exp+sum into a single L2 pass per block:

```
for each block (8192 floats = 32 KB):
    1a. Stream from L2, find block_max -> data now in L1
    1b. Update global_max; if changed: running_sum *= fexpa((old-new)*LOG2E + shift)
    1c. Re-read block from L1, accumulate exp(x - global_max) into running_sum
```

Total: 1 L2 read + 1 L1 read per element (vs 2 L2 reads in 2-pass).

Block size = 8192 floats (32 KB = half L1D). The block must fit in L1 so the
re-read in step 1c hits L1 instead of L2.

**Result**: Slightly slower than 2-pass+prefetch (25.3 vs 22.7 us). The L1
re-read adds overhead, and SW prefetch already makes the 2-pass L2 streaming
efficient enough that eliminating one L2 pass doesn't compensate.

### 5. Blocked Forward + Backward

- Phase 1: Blocked max+exp+sum (same as blocked forward)
- Phase 2: Stream logits from L2, compute grad = exp(x-max)/sum, write grad

Total: 2 L2 reads + 1 write (vs 3 reads + 1 write in 3-pass).

**Result**: 47.0 us vs 44.4 us for 3-pass — blocked is again slightly slower.

## Bottleneck Analysis

### Why Blocking Didn't Win

The 2-pass+prefetch approach achieves 534 B/cy at 100 MHz timer = 26.7 B/cy at 2 GHz.
This is already 42% of L2 bandwidth (~64 B/cy theoretical), which is good for a
compute+load mixed workload (FMLA+FEXPA+FADD = 3 FLA ops per element competing
for the same pipes as loads).

The blocked approach saves 1 L2 pass but adds:
- Block max extraction (scalar reduce per block)
- Global max correction (fexpa multiplication per block)
- L1 re-read overhead (still consumes load bandwidth even from L1)

Net effect: the L2 pass saved (~12 us) is offset by the additional overhead.

### Compute vs Memory Ceiling

Pass 2 inner loop per element: 1 FMLA + 1 FEXPA + 1 FADD = 3 FLA uops.
At 2 FLA/cy throughput: 1.5 cy/element = 2.67 B/cy (4 B read / 1.5 cy).

L2 streaming ceiling: ~64 B/cy -> 1/16 cy/element (4 B / 64 B/cy).

Compute ceiling (2.67 B/cy) << L2 ceiling (64 B/cy): **compute-bound in pass 2**.
But pass 1 (max) is purely load+compare at ~1 FLA/element -> 0.5 cy/element ->
8 B/cy, which IS L2-limited. SW prefetch helps pass 1 the most.

## Accuracy

All variants produce identical results within FEXPA tolerance (~0.04% relative error):
- Blocked matches 2-pass exactly for V <= 32000 (single block covers all)
- For V > 32000, blocked has negligible additional error from max correction step
- Gradient checks pass (finite difference, abs < 5e-4 or rel < 1%)
- grad sum ≈ 0 (< 1e-7)

## API

```c
// Forward (2-pass, SW prefetch + 8x unroll) — RECOMMENDED
float cross_entropy_fwd_f32(const float *logits, int target, int V);

// Forward with fp16 input (4x fp16-vec unrolled)
float cross_entropy_fwd_f16(const uint16_t *logits_f16, int target, int V);

// Forward blocked (1 L2 pass via blocking)
float cross_entropy_fwd_blocked_f32(const float *logits, int target, int V);

// Forward + Backward (3-pass, SW prefetch + 8x unroll) — RECOMMENDED
float cross_entropy_fwd_bwd_f32(const float *logits, int target, int V, float *grad);

// Forward + Backward blocked (2 L2 reads + 1 write)
float cross_entropy_fwd_bwd_blocked_f32(const float *logits, int target, int V, float *grad);

// Batch forward (OpenMP)
void cross_entropy_batch_f32(const float *logits, const int *targets,
                             float *losses, int batch_tokens, int V);
```

## Build

```bash
cd a64fx/cross-entropy
make clean && make
OMP_NUM_THREADS=1 ./bench_cross_entropy
```

## Tuning Parameters

- `PF_DIST_LINES`: SW prefetch distance in cache lines (default 8, try 4/16/32)
- `BLOCK_SIZE`: floats per block for blocked variants (default 8192 = 32 KB)
