# cuda/fa2 — benchmark log

Device: **NVIDIA GeForce RTX 5060 Ti**, sm_120, 36 SMs @ 2587 MHz.
Peaks (this card): **BF16/FP16 ≈ 42–49 TFLOP/s**, **FP8 (e4m3) ≈ 84–92 TFLOP/s**.

Method: `bench_cuda_fa2`, 50 timed iters + 5 warmup, CUDA-event timing, mma mode.
FLOP count `4·B·H·D · pairs`, `pairs = S²` (non-causal) or `S(S+1)/2` (causal).
Accuracy = cosine vs CPU FP32 reference over the first 8 query rows (all dtypes
validate `cos = 1.00000` to 5 dp; `max_err` column shows the true rounding gap).

Inputs are uniform random in [-1, 1]. FP8 uses per-tensor max-abs scaling
(`s = max|·|/448`); the random spread quantizes cleanly, so `max_err ≈ 1–3e-4`.
Real-model tensors with outliers will need a tighter (e.g. per-head) scale to
hold that — the per-tensor scheme here is the floor, not a guarantee.

## Non-causal (TFLOP/s, mma)

| shape       |  D  |   f16 |  bf16 |   fp8 | fp8/bf16 | fp8 max_err |
|-------------|:---:|------:|------:|------:|:--------:|------------:|
| qwen3_512   | 128 | 29.98 | 30.81 | 37.33 |   1.21×  |   3.3e-4    |
| qwen3_2k    | 128 | 39.94 | 39.78 | 54.02 |   1.36×  |   1.7e-4    |
| qwen3_4k    | 128 | 41.94 | 41.88 | 57.08 |   1.36×  |   1.2e-4    |
| dit_1k      |  64 | 39.42 | 38.78 | 48.06 |   1.24×  |   2.1e-4    |
| sd_8k_d64   |  64 | 43.87 | 43.31 | 54.82 |   1.27×  |   6.9e-5    |
| qwen35_2k   | 256 | 38.16 | 38.41 | 49.32 |   1.28×  |   1.7e-4    |
| qwen35_512  | 256 | 29.34 | 29.37 | 36.31 |   1.24×  |   3.5e-4    |

## Causal (TFLOP/s, mma)

| shape       |  D  |   f16 |  bf16 |   fp8 | fp8/bf16 |
|-------------|:---:|------:|------:|------:|:--------:|
| qwen3_512   | 128 | 20.52 | 20.31 | 25.58 |   1.26×  |
| qwen3_2k    | 128 | 35.40 | 34.94 | 44.97 |   1.29×  |
| qwen3_4k    | 128 | 38.89 | 38.74 | 50.77 |   1.31×  |
| dit_1k      |  64 | 30.04 | 29.89 | 35.23 |   1.18×  |
| sd_8k_d64   |  64 | 40.01 | 39.61 | 48.16 |   1.22×  |
| qwen35_2k   | 256 | 33.07 | 34.93 | 42.38 |   1.21×  |
| qwen35_512  | 256 | 20.74 | 20.74 | 24.50 |   1.18×  |

(Causal TFLOP/s use the triangular FLOP count, so they are not directly
comparable to the non-causal column — half the work, plus per-block diagonal
masking overhead and load imbalance on the last block.)

## Reading the numbers

- **bf16/f16 ≈ parity** everywhere (within noise) — expected, both are
  `m16n8k16` and share the entire pipeline; the dtype only changes the `cvt`.
  At long sequence (qwen3_4k, sd_8k_d64) they reach **42–44 TFLOP/s ≈ 90 %+ of
  the ~42–49 peak**, tracking `cuda/fa` v4 (the direct ancestor). exp2f gives a
  small edge over `fa`'s expf.
- **fp8 ≈ 1.2–1.36× bf16.** This is the *expected ceiling*, not a shortfall:
  attention is ~half Q@Kᵀ flops and ~half P@V flops, and only Q@Kᵀ runs in FP8
  (2× tensor-core rate) while **P@V stays bf16** (1×). So `(2·½ + 1·½) = 1.5×`
  ideal, eroded by softmax/cp.async/epilogue overhead to the measured ~1.3×.
  Pushing past this needs FP8 P@V too — i.e. quantizing the softmax probs, which
  the chosen accuracy-preserving scheme deliberately avoids.
- **Speedup grows with sequence length** (1.21× → 1.36× over qwen3_512 → 4k):
  longer S amortises the fixed prologue/epilogue and keeps the tensor cores fed,
  so the FP8 Q@Kᵀ advantage shows through more cleanly. Short shapes (512) are
  launch/occupancy-bound for all dtypes.
- **D=256** runs at `BC=16` (vs 32) to fit the O-accumulator register budget and
  SMEM; it lands ~38 TFLOP/s bf16 / ~49 fp8 at 2k — solid given the heavier
  per-thread accumulator (`D/8·4 = 128` f32/thread).

## Status / not-yet-done (perf headroom)

The kernels are **blog-faithful in structure but not yet fully blog-optimal**.
Carried over from `cuda/fa` v4 rather than the blog's final v5:

- **`ldmatrix.x2`** for the bf16/f16 K/V loads (blog uses **`.x4`**) and **SMEM
  padding** instead of the blog's **XOR address swizzle**. The FP8 K path is
  already conflict-free via the `D+16` byte padding.
- **Double-buffered V** (blog v5 single-buffers V and uses the freed SMEM for a
  larger `BLOCK_Q=128`).
- No split-K / no persistent-CTA scheduling.

These are the levers to close the last ~10 % to bf16 peak and to lift D=256;
correctness and the FP8 win are already in hand.
