# cuda/fa2 — FlashAttention-2 forward (BF16 / FP16 / FP8) for sm_120

A clean FlashAttention-2 forward kernel in the NVRTC-based C CUDA runner style,
following the gau-nernst **"fa-5090"** recipe
(<https://gau-nernst.github.io/fa-5090/>). The blog deliberately uses **only
Ampere-class tensor-core features** — `mma.sync.m16n8k16`, `cp.async`,
`ldmatrix`, SMEM bank-conflict avoidance — and reaches ~94 % of BF16 peak on a
5090 without TMA / WGMMA / tcgen05. That recipe ports directly to consumer
Blackwell **sm_120** (RTX 5060 Ti: 36 SMs @ 2587 MHz).

This is the FP8-capable sibling of `cuda/fa` (which is F16-only, no causal, no
FP8). `fa2` keeps the same proven m16n8k16 fragment layout and online-softmax
reduction but generalises over:

| axis        | values                                                    |
|-------------|-----------------------------------------------------------|
| dtype       | `f16`, `bf16`, **`fp8`** (e4m3 Q@Kᵀ + bf16 P@V)           |
| head dim    | `64`, `128`, `256`                                        |
| masking     | non-causal, **causal** (`FA2_CAUSAL`)                     |

Math: `O[b,h,i,d] = Σ_j softmax_j(Q[i]·K[j] · scale) · V[j]`, `scale = 1/√D`.
Tensors `Q,K,V,O` are `[B·H, S, D]` row-major.

## Files

| file                  | role                                                       |
|-----------------------|------------------------------------------------------------|
| `cuda_fa2_kernels.h`  | NVRTC device-kernel source strings (3 kernels, below)      |
| `bench_cuda_fa2.c`    | host driver — gcc + cuew, NVRTC at runtime, CPU FP32 oracle |
| `Makefile`            | gcc build, links `../cuew.c ../cublasew.c`                 |
| `bench_log.md`        | perf journal (measured TFLOP/s vs peak, vs `cuda/fa`)      |

## Kernels (in `cuda_fa2_kernels.h`)

| kernel string         | entry          | what it does                                                            |
|-----------------------|----------------|-------------------------------------------------------------------------|
| `k_fa2_attn_src`      | `fa2_attn`     | f16/bf16 FA2 (`m16n8k16` both matmuls). `#define FA2_BF16` selects bf16. |
| `k_fa2_attn_fp8_src`  | `fa2_attn_fp8` | e4m3 Q@Kᵀ (`m16n8k32`) + bf16 P@V (`m16n8k16`). Output bf16.             |
| `k_fa2_ref_src`       | `fa2_ref`      | naïve non-tiled GPU reference (causal-aware), correctness oracle.       |

The host **prepends a `#define` header** to the chosen source string before
`cu_compile_kernels` (NVRTC). Specialization knobs:

```
#define FA2_D      128   /* head dim: 64 | 128 | 256                 */
#define FA2_BR      64   /* query rows per CTA ((BR/16) warps × 16)   */
#define FA2_BC      16   /* KV tile — tuned per (dtype,D), see below  */
#define FA2_CAUSAL   0   /* 0 | 1                                     */
#define FA2_BF16     1   /* present → bf16; absent → f16 (fa2_attn)   */
```

**KV-tile size `BC` is tuned per (dtype, D)** by the host `pick_BC()`:

| dtype     | D=64 | D=128 | D=256 |
|-----------|:----:|:-----:|:-----:|
| bf16/f16  |  32  | **16**|  16   |
| fp8       |  32  |  32   |  16   |

Per-block SMEM ≈ `4·BC·(D+8)·2 B` is set by `BC` (not `BR`), and this kernel is
**occupancy/latency-bound** on sm_120: at D=128 bf16, BC=32 leaves only ~2
blocks/SM, so BC=16 (~5 blocks/SM) wins **+3 %**. FP8 Q@Kᵀ is `m16n8k32` (2×
dense, more compute-bound) and prefers the larger BC=32 tile. D=256 is pinned to
16 by the O-accumulator reg/SMEM budget. See `bench_log.md` for the full study,
including two blog levers (`BLOCK_Q=128`, `ldmatrix.x4`) that were measured and
reverted as net-negative on this card.

### Design (shared by `fa2_attn` / `fa2_attn_fp8`)

- **4 warps / 128 threads**, `BR = 64` (each warp owns 16 query rows). Grid
  `(⌈S/BR⌉, B·H)`.
- **Q resident in registers** across all KV tiles (A-fragment, loaded once).
- **2-stage `cp.async.cg` double-buffered** K/V SMEM pipeline (prefetch tile
  `t+1` while computing tile `t`; `cp.async.wait_group 1`).
- **Online softmax** over the 4-thread MMA row-group: row-max / row-sum reduced
  with `__shfl_xor` offsets 1,2; running max `m`, running sum `l`, rescaled `O`.
  Softmax done in **base-2** (`exp2f`, `lscale = scale·log2e`) — softmax is
  base-invariant and `exp2f` is a hair faster than `expf`.
- **P@V via `ldmatrix.x2.trans`** on the row-major V SMEM (avoids a transpose
  pass); P is repacked f32→16-bit straight into the P@V A-fragment layout (the
  C-frag and A-frag layouts coincide → no permute).
- **SMEM** uses **padding** (row stride `D+8` halves) for the K/V f16/bf16 tiles
  so `ldmatrix` is conflict-free.
- **Causal:** skip fully-masked KV tiles (`s_eff = min(S, blk·BR+BR)`); on the
  diagonal tile, set `S = −∞` where `kv_col > q_row` (folded into the OOB mask).

### FP8 specifics (`fa2_attn_fp8`)

- **Q, K stored as e4m3 bytes** with **per-tensor max-abs scales** `s_q, s_k`
  (host computes `s = max|·|/448`, quantizes `x/s`). The descale `s_q·s_k` is
  folded into the `scale` argument by the host, so the kernel's softmax is
  unchanged.
- **Q@Kᵀ** uses `mma.sync.m16n8k32.f32.e4m3.e4m3.f32`. The e4m3 A/B fragments are
  read **directly from SMEM as `uint32`** (4 bytes = 4 e4m3) — `ldmatrix` is a
  16-bit op and does not apply. K rows are padded to **`D+16` bytes** so the
  strided `uint32` reads land **conflict-free** (`b_col·(D+16)/4 mod 32` cancels
  to the lane index). sm_120 uses the standard m16n8k32 `a0,a1,a2,a3` ordering.
- **V stays bf16; P@V is the identical bf16 `m16n8k16` path.** The softmax
  probabilities live in [0,1] and quantize poorly to e4m3, so keeping P/V in
  bf16 (SageAttention-style mixed precision) preserves accuracy — only the
  Q,K e4m3 rounding contributes error (`max_err ≈ 1–3e-4` on these shapes).

## Build & run

```sh
cd cuda/fa2
make                 # builds ./bench_cuda_fa2 (gcc, no nvcc)
make smoke           # quick: bf16 qwen3_512, non-causal + causal
make run-bf16        # bf16, all shapes, non-causal
make run-fp8         # fp8,  all shapes, non-causal
```

CLI:

```
./bench_cuda_fa2 [--dtype f16|bf16|fp8] [--mode mma|ref|all] [--shape NAME|all]
                 [--causal 0|1] [--batch B --heads H --seqlen S --head-dim D]
                 [--br N] [--bc N] [--minblk N] [--iters N] [--warmup N]
                 [--verify 0|1] [--verify-qrows N] [--verbose N]
```

- `--br` / `--bc` / `--minblk` override the auto-picked query block / KV tile /
  `__launch_bounds__` min-blocks hint (else `BR=64`, `BC=pick_BC(D,dtype)`, no
  hint) — handy for A/B sweeps. All three are tuned to their measured optimum by
  default; the knobs exist to reproduce the sweeps in `bench_log.md`.
- `--verbose 1` also prints register count and resident blocks/SM per config.

- `--mode ref` / `all` runs the naïve GPU oracle too; for `--dtype fp8` the ref
  is auto-skipped (it's a 16-bit kernel) — FP8 validates against the **CPU FP32
  reference** instead.
- Built-in shapes: `qwen3_512/2k/4k` (D=128), `dit_1k` (D=64), `sd_8k_d64`
  (D=64), `qwen35_2k/512` (D=256). Override with `--batch/--heads/--seqlen/--head-dim`.
- `validate()` reports cosine similarity vs the FP32 reference (pass ≥ 0.999 for
  f16/bf16, ≥ 0.99 for fp8) plus max abs error.

## Results (RTX 5060 Ti, sm_120)

See `bench_log.md` for the full table + tuning study. Headline (non-causal,
mma): bf16/f16 reach **≈ 43–44 TFLOP/s** at long seq (~90 %+ of the ~42–49
TFLOP/s BF16 peak, after the per-(dtype,D) `BC` tuning lifted D=128 +3 %);
**fp8 reaches ~57 TFLOP/s** (qwen3_4k), a **~1.32× speedup over bf16** — matching
theory, since only Q@Kᵀ is FP8 (P@V stays bf16, so the upper bound is ~1.33×
not 2×). All configs validate cos ≈ 1.0.
