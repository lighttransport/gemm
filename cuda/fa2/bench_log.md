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

KV-tile size `BC` is tuned **per (dtype, D)** by `pick_BC()` (see below); `BR=64`
fixed (4 warps × 16 q-rows). The `--br`/`--bc` CLI knobs override both for A/B.

## Non-causal (TFLOP/s, mma)

| shape       |  D  |  BC |   f16 |  bf16 |   fp8 | fp8/bf16 | fp8 max_err |
|-------------|:---:|:---:|------:|------:|------:|:--------:|------------:|
| qwen3_512   | 128 | 16/32 | 30.28 | 30.00 | 37.39 |   1.25×  |   3.3e-4    |
| qwen3_2k    | 128 | 16/32 | 41.29 | 41.08 | 53.76 |   1.31×  |   1.7e-4    |
| qwen3_4k    | 128 | 16/32 | 43.60 | 43.22 | 56.94 |   1.32×  |   1.2e-4    |
| dit_1k      |  64 | 32  | 39.19 | 38.76 | 47.77 |   1.23×  |   2.1e-4    |
| sd_8k_d64   |  64 | 32  | 43.74 | 43.19 | 54.83 |   1.27×  |   6.9e-5    |
| qwen35_2k   | 256 | 16  | 38.03 | 38.23 | 49.24 |   1.29×  |   1.7e-4    |
| qwen35_512  | 256 | 16  | 29.24 | 29.23 | 36.16 |   1.24×  |   3.5e-4    |

(`BC` column "16/32" = 16 for bf16/f16, 32 for fp8 — see tuning note.)

## Causal (TFLOP/s, mma)

| shape       |  D  |   f16 |  bf16 |   fp8 | fp8/bf16 |
|-------------|:---:|------:|------:|------:|:--------:|
| qwen3_512   | 128 | 21.08 | 20.77 | 25.66 |   1.24×  |
| qwen3_2k    | 128 | 35.68 | 35.14 | 44.96 |   1.28×  |
| qwen3_4k    | 128 | 40.00 | 39.67 | 50.57 |   1.27×  |
| dit_1k      |  64 | 30.02 | 29.61 | 34.96 |   1.18×  |
| sd_8k_d64   |  64 | 40.01 | 39.56 | 48.40 |   1.22×  |
| qwen35_2k   | 256 | 33.12 | 34.93 | 42.31 |   1.21×  |
| qwen35_512  | 256 | 20.56 | 20.60 | 24.41 |   1.18×  |

(Causal TFLOP/s use the triangular FLOP count, so they are not directly
comparable to the non-causal column — half the work, plus per-block diagonal
masking overhead and load imbalance on the last block.)

## Reading the numbers

- **bf16/f16 ≈ parity** everywhere (within noise) — expected, both are
  `m16n8k16` and share the entire pipeline; the dtype only changes the `cvt`.
  At long sequence (qwen3_4k, sd_8k_d64) they reach **43–44 TFLOP/s ≈ 90 %+ of
  the ~42–49 peak**. exp2f gives a small edge over `fa`'s expf.
- **fp8 ≈ 1.2–1.32× bf16.** This is the *expected ceiling*, not a shortfall:
  attention is ~half Q@Kᵀ flops and ~half P@V flops, and only Q@Kᵀ runs in FP8
  (2× tensor-core rate) while **P@V stays bf16** (1×). So `(2·½ + 1·½) = 1.5×`
  ideal, eroded by softmax/cp.async/epilogue overhead to the measured ~1.3×.
  Pushing past this needs FP8 P@V too — i.e. quantizing the softmax probs, which
  the chosen accuracy-preserving scheme deliberately avoids.
- **Speedup grows with sequence length** (1.25× → 1.32× over qwen3_512 → 4k):
  longer S amortises the fixed prologue/epilogue and keeps the tensor cores fed.
  Short shapes (512) are launch/occupancy-bound for all dtypes.
- **D=256** runs at `BC=16` to fit the O-accumulator register budget and SMEM;
  it lands ~38 TFLOP/s bf16 / ~49 fp8 at 2k — solid given the heavier
  per-thread accumulator (`D/8·4 = 128` f32/thread).

## Tuning: KV-tile size `BC` (the win that landed)

`pick_BC(D, dtype)` chooses the KV tile:

| dtype     | D=64 | D=128 | D=256 |
|-----------|:----:|:-----:|:-----:|
| bf16/f16  |  32  | **16**|  16   |
| fp8       |  32  |  32   |  16   |

Per-block SMEM ≈ `4·BC·(D+8)·2 B` (bf16) — i.e. set by **BC, not BR**. The card
has 100 KB SMEM/SM, so smaller BC ⇒ more resident blocks ⇒ more latency hiding:

- **bf16/f16, D=128:** BC=32 → 34.8 KB → only **2 blocks/SM** (occupancy-starved).
  Dropping to BC=16 → 17.4 KB → **~5 blocks/SM** lifts qwen3_2k/4k **+3.0–3.2 %**
  (39.8→41.1, 42.0→43.2) despite the extra loop overhead. This kernel is
  **occupancy/latency-bound**, so more concurrent blocks beats a bigger tile.
- **fp8, D=128:** the opposite — fp8 Q@Kᵀ is `m16n8k32` (2× dense) so it is more
  compute-bound and already has enough occupancy at BC=32; the **larger** tile
  amortises the per-tile fixed cost and wins (BC=16 *loses* ~10 %: 57→52).
- **D=64:** BC=32 already gives ≥5 blocks/SM for both dtypes, so the larger tile
  wins; **D=256:** BCK≥1 forces BC≥16, and reg/SMEM forces BC≤16.

## Occupancy (mma, `--verbose 1` prints regs / blocks-per-SM)

| config            | regs | SMEM   | blocks/SM | limiter   |
|-------------------|:----:|-------:|:---------:|-----------|
| bf16 D=128 BC=16  | 164  | 17.4 KB|    3      | registers |
| bf16 D=64  BC=32  | 124  | 18.4 KB|    4      | registers |
| bf16 D=256 BC=16  | 254  | 33.8 KB|    2      | registers (near 255 cap) |
| fp8  D=128 BC=32  | 161  | 26.6 KB|    3      | registers |
| fp8  D=256 BC=16  | 238  | 25.6 KB|    2      | registers |

Every config is **register-limited, not SMEM-limited** — the O-accumulator
(`D/8·4` f32/thread = 64 at D=128, 128 at D=256) plus Q/score/P fragments dominate.

## Headroom levers that did NOT transfer (measured, then reverted)

The blog's last-10 %-to-peak levers were all tried and **measured on this card**;
all are net-negative here because the kernel is **register/occupancy-bound**, not
LSU- or DRAM-bandwidth-bound (the regime the blog optimises for):

- **`BLOCK_Q = 128` (BR 64→128, 8 warps/CTA):** −13 % (qwen3_4k 42.0→36.6). Halves
  the CTA count and drops occupancy to ~1 block/SM; the L2 re-read saving (K+V per
  head ≈ 2 MB fits the ~32 MB L2, so DRAM was never the limiter) does not pay for
  the lost parallelism. Smaller `BR` (16/32) is also worse on short seqs (qwen3_512
  bf16: BR16 19.2, BR32 29.9, BR64 30.0) — the short-seq floor is per-CTA
  prologue/epilogue overhead, not CTA starvation. `BR=64` is optimal at every size.
  Left as the opt-in `--br` knob.
- **`ldmatrix.x2 → .x4`** (both matmuls; a lane→matrix mapping that reproduces the
  exact x2 fragments, `cos` stayed 1.0): a **wash** — +1–2 % on D=64/256 but
  **−2 % on D=128** (stable across repeats: 43.2→42.4). Working through the bank
  math: under **either** `D+8` padding **or** an XOR swizzle, *each* x4 matrix is
  internally conflict-free (rows differing by 8 fall in separate load phases), so
  x4's regression is **not** bank conflicts — it's that the kernel isn't LSU-bound.
  Reverted; kernel stays on the proven x2.
- **`__launch_bounds__` min-blocks hint** (cap registers to force more blocks/SM):
  −16 % at 4 blocks, −41 % at 5 blocks (bf16 D=128: 164 regs/3 blk = 43.4 → 128
  regs/4 blk = 36.5 → 96 regs/5 blk = 25.8). Capping registers spills the
  O-accumulator to local memory; the DRAM traffic swamps the occupancy gain.
  **3 blocks/SM is the sweet spot, and the *only* free way to reach it is shrinking
  SMEM (pick_BC), not capping registers** — which is exactly why BC=16 worked and
  this doesn't. Left as the opt-in `--minblk` knob; default = no hint.

## Status / remaining headroom (not done)

The kernel is at its **register/occupancy optimum** for this layout; the blog's
v5 levers don't transfer (above). What's left would be a different algorithm, not
a tweak:

- **XOR address swizzle** — definitively *not* the missing piece for x4 (the bank
  analysis above shows x4 is already conflict-free under padding); it would only
  shave the ~1 KB/block padding, worth at most one more block/SM on bf16 D=128 —
  and forcing more blocks there *loses* (min-blocks result). Closed.
- **FP8 P@V** (quantize the softmax probs to e4m3) is the one lever that could
  break the ~1.3× fp8 ceiling toward ~1.5×, but it's deliberately excluded by the
  accuracy-preserving design (probs in [0,1] quantize poorly to e4m3).
- **Persistent CTAs / split-KV** would lift the short-seq (S=512, ~30 TFLOP/s)
  floor, but that regime isn't the target and it adds a reduction pass.
- No split-K / no persistent-CTA scheduling.

Correctness was re-verified beyond the default 8-row window: **full output across
8 q-CTAs**, **partial last blocks** (S=500, S=333), and **causal diagonal masking**
(S=130) all validate `cos = 1.00000`.
