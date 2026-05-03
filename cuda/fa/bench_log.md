# cuda/fa benchmark log

Device: RTX 5060 Ti, sm_120, 36 SMs @ 2587 MHz
Peaks (assumed, GeForce-throttled): F16/BF16 = 42 TFLOP/s, FP8 = 84 TFLOP/s.

FlashAttention forward: O = softmax(Q·K^T / sqrt(D)) · V, non-causal, self-attention.

## v1 — tiled online softmax, scalar fma (1 thread per D-coord, 1 row per CTA)

CTA = D threads (128), one block per query row. K/V tile (FA_BKV=64) staged
to SMEM in fp32. Per-tile inner loop computes FA_BKV=64 scores serially with
__syncthreads twice per j (2·64 = 128 syncs per K-tile).

Initial measurement removed the per-j `__syncthreads` by aggregating warp-sums
once per K-tile (2 syncs vs 128), but that only shaved ~20% off — score loop
is still scalar fma.

| shape       | ms     | TFLOP/s | % of 42 |
|-------------|-------:|--------:|--------:|
| qwen3_512   | 32.5   | 0.07    | 0.17%   |
| qwen3_2k    | 522.1  | 0.07    | 0.17%   |

## v2 — one warp per query row, f16 SMEM, all reductions warp-local

CTA = 8 warps × 32 threads = 256 threads, 8 query rows per CTA. Each thread
holds D/32 D-coords of Q in registers; warp shuffle reduces across D for
each (j) score. K/V tile in **SMEM as f16** (half SMEM bytes vs fp32, halved
load BW), converted to fp32 on read. No `__syncthreads` inside the score
loop — only 2 per K-tile (load + use boundary).

| shape       | ms      | TFLOP/s | vs v1   | vs ref |
|-------------|--------:|--------:|--------:|-------:|
| qwen3_512   |   3.37  | 0.64    |  9.7×   | 1.4×   |
| qwen3_2k    |  54.00  | 0.64    |  9.7×   | 1.4×   |
| qwen3_4k    | 215.2   | 0.64    |   —     |  —     |
| dit_1k      |  16.98  | 0.38    |   —     |  —     |
| sd_8k_d64   | 359.7   | 0.38    |   —     |  —     |

Throughput plateaus at ~0.64 TFLOP/s for D=128, ~0.38 for D=64 — consistent
with the kernel being memory-bound on K/V loads + scalar-fma on compute, not
arithmetic-bound. We're at ~1.5% of the 42 TFLOP/s F16 peak.

## ref — naive non-tiled (correctness oracle)

Materializes the full S-score buffer in SMEM, single thread does softmax,
then full block does P·V row-parallel.

| shape       | ms     | TFLOP/s |
|-------------|-------:|--------:|
| qwen3_512   |   4.7  | 0.46    |
| qwen3_2k    |  75.6  | 0.45    |

## v3 — tensor-core mma.sync, 4 warps, 64-row Q-tile per CTA

CTA = 4 warps × 32 = 128 threads, BR=64 query rows, BC=32 K/V tile, D=128
fixed. Q kept in registers across all K-tiles as 8×4 b32 frags per warp.
O accumulator in registers as 16×4 fp32 frags per warp (64 fp32/thread).
SMEM: sK[BC,D] row-major + sVT[D,BC] (V transposed at load). Q@K^T uses
32 `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` per warp per K-tile;
P@V uses 32 mma per warp. Online softmax row-group reduction across
4-thread groups via `__shfl_xor` offsets 1,2.

| shape       | ms     | TFLOP/s | vs v2  | % of 42 |
|-------------|-------:|--------:|-------:|--------:|
| qwen3_512   |  0.153 |  14.07  | 22.0×  |  33%    |
| qwen3_2k    |  2.023 |  16.98  | 26.7×  |  40%    |
| qwen3_4k    |  7.825 |  17.56  | 27.5×  |  42%    |

### v3.1 — uint4 (8-half) vectorized cooperative load

Replaces the per-element K/V cooperative load (32 iters × single half) with
4 uint4 loads per thread per matrix per tile. V transpose stays scalar (8
per-element SMEM stores per uint4 read) since the destination is
column-strided. ~7% wallclock win.

| shape       | ms     | TFLOP/s | vs v3   | % of 42 |
|-------------|-------:|--------:|--------:|--------:|
| qwen3_512   |  0.140 |  15.31  | 1.09×   |  36%    |
| qwen3_2k    |  1.907 |  18.02  | 1.06×   |  43%    |
| qwen3_4k    |  7.327 |  18.76  | 1.07×   |  45%    |

Still single-buffered; load and compute serialize on `__syncthreads`.
Note: BC=64 (not BC=32) was tried first and **regressed** to 11.4 TFLOP/s on
qwen3_2k — softmax/pfrag register pressure (32 fp32 sfrag/thread,
16 b32 pfrag/thread) outweighs amortized load BW. Keeping BC=32.

All cos=1.00000 vs CPU FP32 reference. v3 is D=128 only (qwen3_*); D=64
shapes (dit_1k, sd_8k_d64) remain on v2 until a D=64 variant lands.

Throughput rises with S (more K-tiles amortize the per-CTA Q load), saturating
near ~17.5 TFLOP/s. Gap to peak is set by single-buffered SMEM loads (no
`cp.async` pipelining yet) and `ldmatrix`-less fragment loads — same recipe
that closed the gemm gap in cuda/gemm/v5+.

## v3.2 — pad sK / sVT row strides to break LDS-32 bank conflicts (BIG WIN)

The Q@K^T B-frag access `sK[n_row * D + kbase + col_g]` has lane pattern
`bank = (lane>>2) * (D/2) + (lane&3)` where (D/2) = 64 = 0 mod 32 — every
4-thread row group hits the SAME bank → **8-way conflict** on every LDS.32.
Same conflict on sVT (BC=32 → 16 mod 32 → 2-way).

Fix: pad row strides so `(stride_halves/2) mod 32 ∈ {4, 12, 20, 28}` (each row
lands on a fresh quad of banks, all 32 lanes hit distinct banks):

  - sK row stride: 128 → 136 halves  (272B = 68 4B-words = 4 mod 32) ✓
  - sVT row stride: 32 → 40 halves   ( 80B = 20 4B-words = 20 mod 32) ✓

Total SMEM: 32×136 + 128×40 halves = 9472 halves = ~18.5 KiB (was 16 KiB).

| shape       | ms     | TFLOP/s | vs v3.1 | % of 42 |
|-------------|-------:|--------:|--------:|--------:|
| qwen3_512   |  0.104 |  20.63  |  1.35×  |  49%    |
| qwen3_2k    |  1.337 |  25.71  |  1.43×  |  61%    |
| qwen3_4k    |  5.118 |  26.86  |  1.43×  |  64%    |

cos=1.00000 vs CPU ref. We've gone 4.4× over v2 since v3 landed; from 33%
peak to **64% peak** by addressing bank conflicts alone (no ldmatrix yet).

## v4 (final) — cp.async + bank-conflict-free swizzle + ldmatrix (97.5% PEAK)

Final v4 evolution. Replaces explicit u32 LDS B-frag loads with `ldmatrix.x2`
(Q@K^T) and `ldmatrix.x2.trans` (P@V), and eliminates the V transpose pass
entirely. Critical: **sV must also be padded** (row stride 128→136) for
ldmatrix to be conflict-free; without that pad ldmatrix issues serialize on
the same D/2≡0 mod 32 collision and we lose all the gain.

SMEM (34 KiB):
  - sK [2][BC, DP=136] = 8.5 KiB × 2 = 17 KiB
  - sV [2][BC, DP=136] = 8.5 KiB × 2 = 17 KiB

Pipeline: prologue cp.async tile 0 → each iter cp.async tile (t+1), wait on t,
then Q@K^T via ldmatrix.x2 from sK, online softmax, P@V via ldmatrix.x2.trans
from sV (no transpose pass, no sVT buffer).

| shape       | ms     | TFLOP/s | vs prior v4 | % of 42 |
|-------------|-------:|--------:|------------:|--------:|
| qwen3_512   |  0.073 |  29.52  |  1.23×      |  **70%** |
| qwen3_2k    |  0.886 |  38.77  |  1.35×      |  **92%** |
| qwen3_4k    |  3.358 |  40.93  |  1.34×      |  **97.5%** |

cos=1.00000 vs CPU FP32 reference. We're at parity with the published F16
matmul peak on the qwen3_4k shape — softmax and online-rescale overhead are
fully hidden by the cp.async pipeline + tensor-core throughput.

## v4 (interim) — cp.async + transpose pass + bank-conflict-free swizzle

Initial v4 (without sK/sVT pad) ran 5% **slower** than v3.1 because the separate
transpose pass cost more than cp.async hid. After applying the same row-stride
pad as v3.2 (sK stride 128→136, sVT stride 32→40 halves) so LDS-32 is conflict-
free, v4 now wins decisively over v3.2 — cp.async fully overlaps with compute.

SMEM (43 KiB):
  - sK [2][BC, DP=136]  = 8.5 KiB × 2 = 17 KiB
  - sVR[2][BC, D=128]   = 8 KiB × 2   = 16 KiB
  - sVT   [D, BCP=40]   =               10 KiB

Pipeline: prologue issues tile 0 → each iter issues tile (t+1), waits on tile t,
sync transposes sVR[cur]→sVT, runs Q@K^T + P@V from sK[cur] and sVT.

| shape       | ms     | TFLOP/s | vs v3.2 | % of 42 |
|-------------|-------:|--------:|--------:|--------:|
| qwen3_512   |  0.090 |  23.92  |  1.16×  |  57%    |
| qwen3_2k    |  1.200 |  28.63  |  1.11×  |  68%    |
| qwen3_4k    |  4.490 |  30.61  |  1.14×  |  73%    |

cos=1.00000 vs CPU FP32 reference. Default `--mode ptx` now maps to v4.

**End-to-end progression** for qwen3_2k (D=128, S=2048):

| rev   | ms     | TFLOP/s | speedup    | % of 42 peak |
|-------|-------:|--------:|-----------:|-------------:|
| v1    | 522    |  0.07   |    1×      |   0.2%       |
| v2    |  54    |  0.64   |    9.7×    |   1.5%       |
| v3    |  2.02  |  16.98  |  258×      |  40%         |
| v3.1  |  1.91  |  18.02  |  274×      |  43%         |
| v3.2  |  1.34  |  25.71  |  390×      |  61%         |
| v4 (interim) | 1.20 | 28.63 | 435× | 68%       |
| **v4 final** | **0.886** | **38.77** | **589×** | **92%** |

## Status / next

All three kernels validate `cos=1.00000` against CPU FP32 reference (with
quantize→dequantize golden so f16 rounding matches GPU). All 3 are far below
the 42 TFLOP/s F16 peak — this is **scalar-fma territory**, not tensor cores.

The real gap (~150–200×) closes only with **mma.sync tensor cores**. Plan
for v3:

- 64×64 query tile per CTA (4 warps, each owns 16 query rows)
- Q@K^T via `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
- P@V via the same mma instruction with a register-resident P frag
- `cp.async.cg.shared.global` + `ldmatrix.sync.aligned.m8n8.x4` for K/V stage
- XOR-swizzled SMEM (same recipe as cuda/gemm/v5+)
- Online softmax across the row-fragment of S between Q@K^T and P@V

Expected v3 throughput: 25–35 TFLOP/s at qwen3_2k+ (60–80% of f16 peak), in
line with what cuda/gemm/v5/v7 achieves on the same architecture for the
underlying matmuls.
