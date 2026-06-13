# Squeezing 21× out of a Vision Encoder on RDNA4

*How a profile-driven, layered attack on the Qwen3.6 vision encoder
took it from 77 to 1638 tokens/s at 2048² on a single RX 9070 XT.*

## The starting point

When we first ran the Qwen3.6-27B vision encoder end-to-end on an RX
9070 XT (gfx1201, RDNA4), the headline number was sobering: **260
tokens/s at 1024², 77 tokens/s at 2048²**. For a card whose **BF16/F16
WMMA** peak is 195 TFLOP/s (the F32 path has no WMMA — it falls back
to regular VALU and tops out around an order of magnitude lower) and
whose DRAM peak is 640 GB/s, that felt embarrassingly slow.

The first instinct on any GPU port is to look at the GEMMs. They are,
after all, where most of the FLOPs live. But the profile told a
different story:

```
flash_attn_dyn_f32      : 87.8% of GPU time (10.29s of 11.72s)
gemm_wmma_bf16_f32      :  ~7%
everything else         :  ~5%
```

The vision transformer's attention — running as a *scalar F32*
fallback because no WMMA path existed for `head_dim = 80` — was
eating almost all of the time. The GEMMs, the thing every "GPU
optimization" guide leads with, were already a rounding error. The
real question was: how do you write a flash-attention kernel for a
head dimension that doesn't divide cleanly by the hardware tile size?

This is the story of the eight optimizations that followed, why each
one mattered, and where the floor still is.

## RDNA4 WMMA in 60 seconds

RDNA4's WMMA does a 16×16×16 matmul per wave in ~16 cycles, dual-
issuing across the SIMD pair. **Inputs must be BF16 or F16, with F32
accumulation.** F32 inputs have no WMMA path — they fall back to
VALU at roughly an order of magnitude lower throughput. For
`head_dim = 80`, pad to 80 (5 K-tiles, last partial).

## Step 1 — A WMMA flash-attention kernel that actually exists

The first version was the simplest thing that could possibly work: 1
wave per workgroup, `BQ = 16` queries per block, `BKV = 16` keys per
block, head dimension padded to 80. The kernel:

1. Loads its 16 queries from F32, scales, and casts to **BF16** in
   registers (an F16 variant exists too).
2. For each KV tile: loads K and V into LDS as BF16, runs Q·Kᵀ via
   5 back-to-back WMMAs with **F32 accumulation**, online softmax
   with `__shfl_xor` cross-lane reductions, then S·V via 5 more.
3. Writes the F32 output once at the end.

This dropped 1024² from 3930 ms to 1085 ms — **3.6×**, just from
moving off scalar F32 onto BF16 WMMA. Cosine vs llama.cpp:
0.99996665 (F16 variant: 0.99998676 at the same runtime). That's
the precision floor we held the whole way.

## Step 2 — Two waves, one workgroup

A 1-wave WG underutilizes per-CU concurrency. The 2-wave variant
doubles `BQ` to 32 — each wave still owns 16 queries, but they share
one KV tile load. DRAM K/V traffic per query halves; the 64-thread
cooperative LDS load hides issue rate better.

1024²: 1085 → 758 ms (1.4×). Cosine unchanged.

## Step 3 — Pre-pack K and V to BF16

Inside the FA inner loop, every KV tile load reads `qkv` (F32) and
casts to BF16 in LDS — repeated 256× per query block at 1024², and
wasting bandwidth (read 4 B, use 2 B).

A one-shot `kv_transpose_bf16` kernel run once per attention block
packs F32 → BF16 in `[heads, n_tok, head_dim]` layout. FA reads BF16
directly; DRAM K/V traffic halves. 758 → 713 ms.

## Step 4 — A patch embedding the easy way

Qwen3-VL's patch embedding runs *two* conv2ds on the input image and
sums them. Algebraically, `(W₀ + W₁) ⊛ x = (W₀ ⊛ x) + (W₁ ⊛ x)`. So
why are we computing two convolutions?

We pre-fold `W₀ + W₁` once on the host at model load time. The kernel
shrinks to a single conv. **Single rounding**, so cosine actually
improves: 0.99996665 → 0.99996930.

Then we go further: a conv2d with a small kernel and stride-equals-
kernel-size is a `im2col` followed by a GEMM. Write a `patch_unfold`
kernel that produces the unfolded matrix in F16, then route through
the existing WMMA F16 GEMM. patch_embed time: **113 ms → 0.85 ms.
133× on this one stage.**

Important footnote: BF16 *fails* the accuracy gate here. The 7-bit
mantissa loses too much precision on small normalized RGB inputs
(values in [-2.6, 2.7] range, lots near 0.001). F16 with 10 mantissa
bits passes comfortably. This was the only place in the encoder
where the choice mattered.

## Step 5 — Stop reinventing the GEMM wheel

By this point, the per-block ViT GEMMs (qkv projection, attn_out,
ffn_w0, ffn_w1) had risen to ~61% of total time. We were running
them through a hand-written WMMA kernel achieving 3-14 TFLOP/s on
shapes the kernel wasn't tuned for.

hipBLASLt has ~50,000 pre-tuned kernels in its library, plus runtime
heuristics. The reason we hadn't been using it for the ViT block was
historical: an earlier measurement said "ViT FFN-up is slower with
hipBLASLt." That measurement was stale — true at the time, but only
because *other* parts of the pipeline were dominating. Once
flash-attn was fast, the GEMMs' relative position changed.

The fix was a 16-slot per-shape plan cache keyed on
`(dtype, m, n, k, epilogue)`. First call to a shape pays the
hipBLASLt heuristic cost (~ms); every subsequent call is a hash
lookup. Epilogues map cleanly: `BIAS` for qkv, `BIAS+residual` via
β=1 with `C=residual` for attn_out, `GELU+BIAS` for ffn_w0,
`BIAS+residual` for ffn_w1.

**Encoder GEMM total: 350 ms → 32 ms. 11×.** Average kernel
throughput jumps from ~10 to ~100 TFLOP/s.

## Step 6 — The integer divide that cost a third of the kernel

After step 5, the profile tipped: hipBLASLt GEMMs dropped to 18%, and
flash-attn was 79% — at 188 ms across 27 calls, 7.0 ms per call.

Reading the FA hot loop carefully:

```c
for (int e = tid; e < 16 * FA_WM2_HD_PAD; e += 64) {
    int kv_row = e / FA_WM2_HD_PAD;   // FA_WM2_HD_PAD = 80
    int d = e % FA_WM2_HD_PAD;
    ...
}
```

`HD_PAD = 80`. Eighty is not a power of two. On AMD, that's a *real
integer divide* — 20–30 cycles each. Two of them per loop iteration,
called 20 times per thread per KV tile, 256 KV tiles per block. The
arithmetic is brutal: 10,240 divides per thread per workgroup.

The fix is structural. Replace runtime `e/80` and `e%80` with a
fixed thread-to-row-and-column layout:

```c
int ld_row  = tid >> 2;        // 0..15
int ld_col0 = (tid & 3) * 20;  // 0, 20, 40, 60
```

Each thread always handles a fixed 20-column stripe of one row.
There are no divides. There's a small cost — adjacent threads now
read addresses 40 bytes apart instead of 2, hurting DRAM coalescing
— but L2 absorbs it because the tiles are tiny.

**FA: 7.0 → 4.92 ms/call. -30% on the kernel that was 79% of the
trace.** End-to-end: 278 ms → 215 ms at 1024². The single biggest
single-knob win in the entire optimization.

This is the kind of bug a profiler doesn't tell you about. You read
the loop, notice the modulus by a non-power-of-two constant, and
remember what that compiles to on this ISA.

## Step 7 — Double-buffer the K/V loads

The remaining FA cost is dominated by the per-iteration `__sync` +
LDS-load + WMMA + softmax cycle. Each iteration *waits* for K/V
LDS-stores before it can read them.

Double-buffering breaks the dependency. Allocate two K and two V LDS
slots. A prologue loads tile 0 into buffer 0. Then each iteration:

```
iter t:
    prefetch tile t+1 into buffer 1-(t&1)   # async, doesn't block compute
    Q · K[buffer t&1] · softmax · S · V    # reads "current" buffer
    __syncthreads                           # ensure prefetch landed
```

The prefetch's VMEM and ds_store ops are issued before WMMA, so they
overlap. We pay ~2× the LDS budget (~11 KB/WG, well under the 64
KB/CU cap) and one prologue but get continuous WMMA throughput.

FA: 4.92 → 4.60 ms/call (-7%). The gain is larger at 2048² (where
`n_kv` is twice as long, so the prefetch/compute overlap repeats more
times): 2826 → 2500 ms (-12%).

Total at this point: **202 ms / 4793 tok/s at 1024², 2500 ms / 1638
tok/s at 2048².**

## The journey in one table

| Step | What changed | 1024² ms | 2048² ms |
|---|---|---:|---:|
| 0  | Scalar F32 attention (no WMMA)   | 3930 | 53180 |
| 1  | BF16 WMMA, BQ=16, 1-wave         | 1085 | 11776 |
| 2  | BF16 WMMA, BQ=32, 2-wave         |  758 |  5702 |
| 3  | KV pre-pack F32→BF16             |  713 |  5078 |
| 4a | W₀+W₁ host fold                  |  645 |  4683 |
| 4b | patch_embed im2col + F16 WMMA    |  608 |  4596 |
| 5  | hipBLASLt BF16 ViT GEMMs         |  278 |  3281 |
| 6  | FA fixed-stride loads            |  215 |  2826 |
| 7  | FA double-buffered KV            |  202 |  2500 |

End-to-end: **18.4× at 1024², 21.3× at 2048²**, cosine ≥ 0.99996
throughout.

## What this teaches

A few things worth pulling out of the line-by-line:

**The bottleneck moves.** Step 1 was a 3.6× win on attention. That
made GEMMs the bottleneck. Step 5 was an 11× win on GEMMs. That made
attention *again* the bottleneck — but a different shape of
attention bottleneck (instruction stream pressure, not compute
throughput). Each round, the profile told us a different story than
the last.

**Read the assembly when in doubt.** Step 6 — the divide-by-80 — is
invisible in C. The C looks fine. The ISA tells you `s_div_u32` is
running on a hot path 10,000 times per workgroup. You only see it if
you (a) know the ISA's slow ops or (b) look at the assembly.

**Use the vendor library, but check it.** Step 5's gain came from
*using* hipBLASLt, not avoiding it. The earlier "hipBLASLt is slower
here" measurement was correct in its day and stale by the time we
were in step 5. Always re-measure the gating assumption when the
surrounding system has changed.

**The accuracy floor protects you.** Every step was gated on cosine
≥ 0.99996 vs llama.cpp. When step 4 tried BF16 patch_embed and
failed (rel_l2 jumped from 8e-3 to 3.1e-2), the gate caught it
immediately. Without the gate, that change ships and someone six
months later wonders why generated captions are subtly worse.

## What's still on the table

After 7 steps, FA is still 71% of GPU time at ~13 TFLOP/s — only ~7%
of the 195 TFLOP/s **BF16 WMMA** peak. The remaining headroom, ranked
by expected ROI:

- **BQ=64 4-wave FA.** Halves grid_y; one KV tile shared across 4
  waves instead of 2. Estimated -10 to -20%.
- **b128 LDS vectorization.** Pad rows to 96 or 128 so K/V loads
  become single 16-byte LDS ops. -5 to -10%.
- **Pre-transpose V in the pack step** so smV's `[d × kv]` layout
  matches DRAM read pattern.
- **88- or 96-stride padding** to break LDS bank conflicts on the
  current 80-byte rows. Mostly a follow-on of b128 vectorization.
- **Custom F32→BF16 cast** for hipBLASLt inputs (currently 4.3% of
  total at 88 GB/s — far below DRAM peak, indicating launch-grid
  inefficiency). Or fuse the cast into the previous kernel.
- **Port double-buffering to the F16 FA path.** Mechanical port.

If all of these landed, FA would plausibly hit ~3 ms/call (from
4.6), pushing the total below 150 ms at 1024² — closer to **6500
tokens/s, ~25× over scalar**. Whether that's worth the engineering
hours depends on what's downstream of the encoder.

## Reproduce

```sh
# WMMA is now the default; no env var needed.
./test_hip_vision \
    /path/to/mmproj-BF16.gguf \
    --image fujisan.jpg --image-size 1024 \
    --ref baseline_f32attn.bin --bf16 \
    --warmup 2 --iters 5
```

The full code is in `rdna4/vlm/hip_vision_encoder.c`. Detailed
per-step commentary is in `rdna4_gemm_optimization_log.md`. The
generic recipe distilled from this work — for any GEMM port targeting
~90% of peak — is in `peak_efficiency_playbook.md`.

## Addendum (2026-05-15) — WMMA flash-attention enabled by default

For most of this doc, the WMMA flash-attention kernels were opt-in via
`HIP_VLM_FA=wmma_bf16_*`. With no env var, the dispatcher fell through to
the scalar `flash_attn_dyn_f32` — meaning any user running the encoder
out-of-the-box was getting the slow path. For the Qwen3.6-27B / Qwen3-VL-30B-A3B
tower (`head_dim = 72`), that scalar fallback was ~10× slower than the
already-implemented WMMA kernels, which had supported head_dim=72 all along
via LDS padding to `HD_PAD = 80`.

The fix in `hip_vision_encoder.c` (~L3945) adds an `fa_auto` branch: when
`HIP_VLM_FA` is unset and `head_dim ≤ 80`, default to
`flash_attn_wmma_bf16_4w_pre` (BQ=64, 4-wave, double-buffered pre-packed K/V).
`HIP_VLM_FA=tiled` still restores the scalar path for debugging.

Measured impact on the same model + GPU (BF16, `test_hip_vision`, warm):

| size  | before    | after    | speedup |
|-------|----------:|---------:|--------:|
| 512²  |   306 ms  |   34.4 ms | 8.9×  |
| 1024² |  3233 ms  |  190.3 ms | 17.0× |
| 2048² | 50797 ms  | 1831.3 ms | 27.7× |

That puts the HIP encoder **faster** than a PyTorch/ROCm reference on the same
tower at every size (1.15× to 2.95×). See `vlm-pytorch-compare.md` for the full
comparison and the reusable `bench_pytorch_vision.py` script. Correctness vs
the scalar-F32 path: cosine 0.99994 at 1024².

The phase-2 levers above (V pre-transpose, 128-byte LDS vectorization, fused
F32→BF16 cast, etc.) are no longer urgent — the encoder is already comfortably
ahead of the reference.
