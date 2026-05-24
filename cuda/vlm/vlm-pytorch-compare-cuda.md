# CUDA VLM Vision Encoder — CUDA vs PyTorch (NVIDIA)

Performance comparison of the `cuda/vlm` CUDA vision encoder against a PyTorch
reference, on **identical GPU**. Companion to `rdna4/vlm/vlm-pytorch-compare.md`
(which did the same on AMD/ROCm).

Date: 2026-05-24 · GPU: NVIDIA GeForce RTX 5060 Ti (Blackwell, sm_120, 16 GB) ·
CUDA 12.9/13.x · dtype: **f16** (the CUDA encoder has no bf16 path; RDNA4 doc used bf16)

## Model under test

The Qwen3.6-27B / Qwen3-VL-30B-A3B visual tower (depth=27, hidden=1152, ffn=4304,
heads=16, head_dim=72, patch=16, merge=2, ~539M params).

- **CUDA**: `cuda/vlm/test_cuda_vision --f16` on
  `/mnt/disk01/models/qwen36/27b/mmproj-F16.gguf`. The GGUF mmproj has **deepstack
  disabled** (`0 deepstack layers`).
- **PyTorch**: `rdna4/vlm/bench_pytorch_vision.py` loads `model.visual.*` from
  HF `Qwen/Qwen3-VL-30B-A3B-Instruct` (shard 13), class `Qwen3VLMoeVisionModel`,
  transformers 5.9.0, torch 2.11.0+cu128. Runs the **full** tower **including the 3
  deepstack mergers** (layers 8/16/24) — i.e. slightly *more* work than the GGUF path.

Synthetic square inputs, warmed up, steady-state timed. `tokens = (N/16)² / 4`.

## Results (f16, same GPU)

Numbers below are **after six fixes**: the Blackwell cuBLAS path, flash attention,
the patch-embed im2col+GEMM, **tensor-core (cuBLAS) full attention + a rewritten
windowed-attention kernel**, a **cuBLASLt fused bias/GELU epilogue + F16 FFN
intermediate**, and a **materialize-scores windowed-attention kernel** (see "Fix
applied" sections). Progression of CUDA mean ms (512² / 768²):
session-start built-in kernels **503.6 / 3516** → +cuBLAS F16×F16 **471.8 / 2218** →
+flash attention **122.0 / 448.1** → +patch im2col+GEMM **90.0 / 345.9** → +tensor-core
attn & window rewrite **43.9 / 116.8** → +cuBLASLt fused bias/GELU epilogue & F16 FFN
**41.9 / 104.7** → +windowed-attn WW_WARPS sweep **41.4 / 102.9** →
**+materialize-scores windowed kernel 38.1 / 95.8**
(**13.2× / 36.7× faster end-to-end** than session start).

| image  | tokens | CUDA mean ms | CUDA min ms | CUDA tok/s | PyTorch mean ms | PyTorch min ms | PyTorch tok/s | CUDA vs PyTorch |
|-------:|-------:|-------------:|------------:|-----------:|----------------:|---------------:|--------------:|----------------:|
| 512²   |   256  |         38.1 |        37.7 |       6727 |            96.0 |           69.6 |          2606 | **2.52× faster** |
| 768²   |   576  |         95.8 |        95.4 |       6013 |           212.7 |          184.5 |          2712 | **2.22× faster** |
| 1024²  |  1024  | — (unsupported) |          |            |           438.1 |          401.2 |          4354 |        —        |
| 2048²  |  4096  | — (unsupported) |          |            |          1518.7 |         1515.3 |          2701 |        —        |

CUDA: `--warmup 5 --iters 20`. PyTorch: `--warmup 5 --iters 20`.

> The CUDA encoder is now **>2× faster than PyTorch SDPA on the same GPU** at both 512²
> (2.52×) and 768² (2.22×) — a full reversal of the original 4.9× / 10.4× deficit. (And
> PyTorch additionally runs the 3 deepstack mergers the GGUF path omits, so the real-work
> gap is even larger.) At 768² the dominant remaining cost is the FFN/projection cuBLAS
> GEMMs (≈52%, tensor-core, near-irreducible); the windowed-attention kernel is down to
> ≈12% and the 6-layer full-attention tensor-core path (QK + softmax_rows + P·V + extract)
> is ≈17%. `add_bias`, the standalone `gelu`, the cuBLASLt GELU side kernel, and the large
> FFN-down input cast are all gone (fused / written F16 in-pass); what remains is
> `gelu_f32_to_f16` (≈5.2%) + the smaller residual `cast_f32_f16` (≈2.4%).

## Fix applied #6: materialize-scores windowed-attention kernel

After Fix #5, the windowed-attention kernel (`attn_window_warp_f32`) was the clear #2
cost — **14.8 ms/encode, 18% at 768²**. It used the flash online-softmax algorithm
**one warp per query**, but that serializes the softmax across a window's 36 keys: each
key does a dependent `__shfl_xor` reduction (to finish the Q·K dot split across the
warp's lanes) followed by an `__expf`, and the running max/denominator chain each
iteration onto the previous, so the 36 keys cannot overlap. The kernel was issue-bound
on that dependency chain, not on memory (per-launch stddev 0.3%).

The windows here are tiny and fixed (≤36 tokens, head_dim 72), so the new
`attn_window_tile_f32` **materializes the whole score matrix** instead. One block per
`(window, head)`: stage Q/K/V into shared once, then compute the full `S[size,size]`
matrix with **each thread owning complete (i,j) dot products** (no warp reductions, no
cross-key dependency); softmax **each row exactly once** (warp-per-row, two reductions);
then `P·V` straight from shared. The online-softmax serial chain is gone entirely.
Dynamic shared = `(3·max_win·head_dim + max_win²)` floats ≈ **35 KB** at `max_win=36`
(under the 48 KB default, no opt-in needed). Block size `WT_THREADS` swept
64/96/128/192/256 → **192 best** (the block is shared-memory-bound, not thread-bound, so
adding threads speeds the parallel score/PV phases for free until launch overhead bites);
the `#define` and the host launch block-dim are kept in sync.

The two-pass max replaces the online running max, so it is numerically equivalent —
rel_L2 vs CPU is **0.6394** (a hair *closer* than the warp kernel's 0.6395). Default ON;
`VLM_WINDOW_TILE=0` reverts to `attn_window_warp_f32`. Profile effect at 768²: windowed
attention **14.8 → 9.2 ms/encode (−38%)**, taking the encoder **101.8 → 95.8 ms** and
512² **41.1 → 38.1 ms**. The 6-layer full-attention tensor-core path (≈17%) is now the
larger attention cost, but it already runs on tensor cores; the FFN/projection cuBLAS
GEMMs (≈52%) remain the floor.

## Fix applied #5: cuBLASLt fused bias + GELU epilogue

After Fix #4, the per-GEMM scaffolding around each cuBLAS call was the largest non-GEMM,
non-attention cost: at 768², `add_bias_f32` (9.2%, ~8.9 ms) + the standalone `gelu`
(5.8%, ~5.6 ms) were separate kernel launches wrapping every GEMM. Both are now folded
into a **cuBLASLt matmul epilogue**: new
`cublasew_gemm_f16_f16_f32_lt_bias_rowmajor_nt(ctx, Y, W_f16, X_f16, bias_f32, gelu, …)`
sets `CUBLASLT_EPILOGUE_BIAS` (or `GELU_BIAS` when the caller requests GELU) plus the
bias pointer, so the bias add (and tanh-GELU) happen inside the matmul. `vlm_gemm` became
`vlm_gemm_ex(…, int do_gelu)` (with `vlm_gemm` a `do_gelu=0` wrapper): on the Blackwell
F16×F16 path it tries the fused-epilogue call first when cuBLASLt is available and a bias
is present, and only falls back to the plain GEMM + separate `add_bias`/`gelu` kernels if
the LT call returns failure. The FFN up-projection and the two mmproj/merger fc1/mm0
GEMMs pass `do_gelu=1`; all other biased GEMMs (qkv, attn_out, ffn_down, mm2) fuse
bias-only.

The encoder's GELU is the **tanh approximation** — exactly what `CUBLASLT_EPILOGUE_GELU`
computes — so the fusion is numerically transparent: rel_L2 vs CPU held **0.6395**. Profile
effect at 768²: `add_bias` **8.9 → 0.03 ms**, `gelu` **5.6 → 0 ms**; cuBLASLt additionally
chose faster algos for the bias-fused shapes (the two `cutlass …_tn` GEMMs dropped
**~48.6 → ~41.9 ms**). The bias-only epilogues fuse inline; the `GELU_BIAS` epilogue runs
as a cuBLASLt side kernel (`globalKernel`, 4.4 ms) — still cheaper than the 5.6 ms
standalone GELU it replaced. Net: 768² **116.8 → 110.0 ms**, 512² **43.9 → 42.6 ms**.

**Refinement — F16 FFN intermediate (no GELU side kernel, no FFN-down recast).** The FFN
operates on the *unmerged* patch tokens (2304 at 768²), so the down-projection's input
cast (`[2304, 4304]` ≈ 9.9 M elts, ~148 µs each × 26 layers ≈ 3.9 ms) was the single
largest `cast_f32_f16` contributor, and the `GELU_BIAS` epilogue ran the 4.4 ms side
kernel. A cuBLASLt `GELU_BIAS` epilogue with **FP16 output** would have killed both, but
it returns `CUBLAS_STATUS_NOT_SUPPORTED` on this card (verified via `CUBLASEW_DEBUG_LT`).
Instead the up-projection now uses a **BIAS-only** epilogue (which fuses *inline* in the
GEMM — no side kernel), followed by a new `gelu_f32_to_f16` kernel that applies the
tanh-GELU and writes **F16 straight into `d_ffn_buf_f16`** in one pass; the down-projection
(`vlm_gemm_x_f16`) then consumes that F16 buffer with **no recast**. So per FFN layer the
old (`globalKernel` 148 µs + down-input cast 148 µs) ≈ 296 µs becomes one 148 µs
`gelu_f32_to_f16`. Profile at 768²: `globalKernel` **4.4 → 0 ms**, `cast_f32_f16`
**46.8 → 14.7 ms** (the big casts gone), new `gelu_f32_to_f16` **4.0 ms**. The
`cublasew_…_lt_bias_…_nt` wrapper gained a `y_f16` output-dtype flag (forces FP32 bias via
`CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE`) for the F16-output attempt; the non-cuBLAS / F32
path keeps the in-place F32 GELU. rel_L2 unchanged (0.6395). Net: 768² **110.0 → 104.7 ms**,
512² **42.6 → 41.9 ms**.

## Fix applied #4: tensor-core attention + windowed-attention rewrite

The flash-attention kernel (Fix #2) was a CUDA-core kernel and, after the patch-embed
fix, became the dominant cost again — at 768² `flash_attn_f32` was **69.7%** of runtime
(37.4 ms × 6 full-attention layers ≈ 224 ms/encode), running at only ~1.5% of peak
because of its per-key warp-reduction. The fix routes the 6 full-attention layers through
**cuBLAS tensor cores**, processing one head at a time so the scores scratch stays small:

1. `attn_extract_heads` deinterleaves `qkv[N,3·dim]` (post-RoPE) into head-contiguous F16
   `Q/K/V [n_heads, N, head_dim]`.
2. **QK^T**: `cublasew_gemm_f16_f16_f32_rowmajor_nt` → `S[N,N]` F32 (reuses the existing
   NT wrapper; per-head buffers are contiguous so `ld = head_dim`).
3. `attn_softmax_rows` does a one-block-per-row softmax of `S` (scale folded) → F16 `P[N,N]`.
4. **P·V**: new `cublasew_gemm_f16_f16_f32_rowmajor_nn` (an F16 clone of the F32 NN
   wrapper, with `ld_y`) writes `O[N,head_dim]` F32 directly into the interleaved
   `attn_out` (`ld_y = dim`).

All four steps share the runner's stream, so a single `[N,N]` F32 scores buffer (21 MB at
768²) + `[N,N]` F16 probs buffer are safely reused across heads. Gated on `use_f16` (the
F32 correctness path keeps the bit-exact CUDA-core `flash_attn_f32`). This alone took
768² **345.9 → 129.1 ms** and 512² **90.0 → 48.7 ms**, eliminating `flash_attn_f32` from
the profile entirely.

The windowed-attention kernel then became the #2 cost (25.6% at 768²). The old
`attn_window_f32` looped queries **serially** and did two full 256-thread block reductions
(~16 `__syncthreads`) **per query** with only ~36 of 256 threads active. The replacement
`attn_window_warp_f32` is the flash online-softmax algorithm bounded to each window's keys,
**one warp per query** (no block barriers), grid `(n_windows, n_heads)`; it also stages the
window's K/V into shared memory once per block so the 8 warps don't re-read them from global
per query. That took the windowed layers from **28 → 16.3 ms/encode** at 768²
(1330 → 777 µs/layer). rel_L2 vs CPU unchanged (0.6395) throughout both rewrites.

*Warps-per-block tuning (`WW_WARPS`):* the kernel's per-launch time is nearly constant
(stddev 0.3%), i.e. it is issue-bound on the per-key `__shfl_xor` reductions, not memory.
At 768² each window holds exactly 36 query tokens, so a `WW_WARPS` that divides 36 evenly
balances the per-warp query load. Sweeping 8→12→16→18 (block 256→384→512→576 threads) found
**`WW_WARPS=12`** (3 queries/warp) best, ~104.7 → 102.9 ms / 41.9 → 41.4 ms. **(This warp
kernel is now superseded by the materialize-scores `attn_window_tile_f32` — see Fix #6.)**

## Fix applied #3: patch embedding via im2col + cuBLAS GEMM

The original `patch_embed_dual_f32` had catastrophically uncoalesced weight reads
(`w0[d*ks+ki]` strided by `ks = ps²·3 = 768`) and re-read each pixel `dim` times → ~19.7 ms.
Replaced with a coalesced `patch_im2col_f32` kernel (gather each patch's pixels into a
`[n_patches, ps²·3]` matrix) followed by `vlm_gemm`, **folding the dual conv weights** into
one (`w0 += w1` at load, since both convs hit the same pixels). Patch embedding dropped
**19.7 → 0.3 ms**, taking 768² 448.1 → 345.9 ms and 512² 122.0 → 90.0 ms.

## Fix applied #2: flash attention for the full-attention layers

The old `attn_full_f32` launched only **`n_heads`=16 blocks** (one per head), each looping
over *all* N queries serially and re-reading every K/V from global memory per query —
leaving most SMs idle and incurring O(N²) global traffic. It dominated runtime, especially
at 768² (7 of 27 layers use full attention; the rest use cheap windowed attention with
~36-token windows, already well-parallelized — left unchanged).

New `flash_attn_f32` (NVRTC kernel in `cuda_vision_encoder.c`): online-softmax flash
attention, **one warp per query**, with K/V tiled into shared memory (`FA_TILE_K=64`) and
reused across the `FA_WARPS=16` queries per block. Grid = `(ceil(N/16), n_heads)` → 1024s
of blocks instead of 16. The per-query softmax stats (m, l) and the head_dim output
accumulator are kept in registers, distributed across the warp's lanes (handles the
awkward head_dim=72 via `ceil(72/32)=3` regs/lane); scores reduced with `__shfl_xor_sync`.
F16-vs-CPU rel_L2 unchanged (0.640) — the kernel is mathematically identical to the prior
full-softmax path. `FA_WARPS=16` measured best (vs 8 → 174/710 ms, vs 32 → 126/463 ms);
`FA_WARPS`/`FA_TILE_K` are mirrored between the kernel `#define`s and the host launch.

## Fix applied: cuBLAS F16×F16 path for Blackwell

The root cause of the slowness was **not** a 768²-specific cliff. `vlm_gemm` only called
`cublasew_gemm_f16_f32_rowmajor_nt` (mixed F16-weights × F32-activations `cublasGemmEx`),
which `cublasGemmEx` **rejects on Blackwell sm_120 at every size**. So the encoder fell
back to slow built-in F16 MMA kernels at *all* sizes (512² included); the 768²-vs-512²
slowdown was just O(n²) attention, not a cuBLAS failure. A standalone probe
(`/tmp/cublas_probe.c`) confirmed mixed F16×F32 fails but F16×F16→F32 succeeds on this
card at all relevant shapes.

`cuda/vlm/cuda_vision_encoder.c` now: adds a `cast_f32_f16` NVRTC kernel + a `d_x_f16`
device scratch buffer; `vlm_gemm` tries mixed F16×F32 once, and on failure (latched via
`cublas_mixed_ok`) casts the F32 activations to F16 and calls
`cublasew_gemm_f16_f16_f32_rowmajor_nt` (weights are already F16). It only falls to the
built-in kernels if cuBLAS itself is unavailable. F16 output rel_L2 vs the CPU ref is
unchanged (0.642), i.e. the new path matches the prior built-in F16 output bit-for-shape.

## CUDA-side problems remaining (RTX 5060 Ti / sm_120)

1. ~~cuBLAS GEMM fails / falls back~~ **FIXED** — see "Fix applied" above. cuBLAS now
   engages at every size via the F16×F16→F32 path.
2. **Image size capped at 768².** `cuda_vision_init()` takes no max-size argument; the
   runner sizes its device scratch from the model's config (`image=768`,
   `max_patches=2304`). 1024² (4096 patches) and 2048² (16384) error with
   `too many patches N (max 2304)`. Reaching them needs a runner change to allocate
   for a larger max image.
3. ~~CUDA ~5–10× slower; unfused attention dominates~~ **FIXED + REVERSED** by flash
   attention (Fix #2), then patch im2col+GEMM (Fix #3), tensor-core attention + the
   windowed-attention rewrite (Fix #4), the cuBLASLt fused bias/GELU epilogue +
   F16 FFN intermediate (Fix #5), and the materialize-scores windowed kernel (Fix #6).
   CUDA is now **2.52× faster at 512² and 2.22× faster at 768²** than PyTorch on the same
   GPU. At 768² the only large reducible cost left is the FFN/projection cuBLAS GEMMs
   (≈52%, tensor-core — near the hardware floor); the windowed kernel is down to ≈12% and
   the 6-layer full-attention tensor-core path is ≈17% (spread across QK/softmax/P·V/extract,
   already on tensor cores). Smaller residual levers: `gelu_f32_to_f16` (≈5.2%) and the
   `cast_f32_f16` (≈2.4%, qkv/attn-out input casts) could fold into the layernorm/attn-output
   kernels, but each is minor.

   *Tried and rejected — batching the full-attention heads.* The 6 full-attn layers run
   the QK / softmax / P·V chain **one head at a time** (16 heads × 3 launches × 6 layers).
   The obvious next step was `cublasGemmStridedBatchedEx` to do all 16 heads' QK^T and P·V
   in one launch each (plus a head-batched softmax) — fewer launches, more parallel GEMM
   work submitted at once. **It measured slower** (97.5 → 99.6 ms @768², 38.2 → 38.9 @512²),
   because the per-head loop keeps a single `[N,N]` scores buffer (21 MB at 768²) **hot in
   this GPU's ~32 MB L2** across QK→softmax→P·V, whereas batching materializes all 16 heads'
   scores (≈340 MB) to DRAM and re-reads them twice. So the per-head loop is already
   well-matched to the L2; batching would only help on a GPU with much larger L2 (or smaller
   N). Reverted.
4. **Numeric divergence vs the CPU reference.** `test_cuda_vision` (no `--no-cpu`)
   reports F32 rel_L2 ≈ 0.64 (Qwen3.6-27B mmproj) and ≈ 0.99 (Qwen3-VL-4B mmproj) —
   deepstack is off in *both* CPU and CUDA, so it is not a deepstack artifact. Root
   cause not investigated. Note the canonical correctness check for this encoder is
   `compare_vs_llamacpp` (vs llama.cpp CLIP embeddings), not the CPU `vision_encode`.

## Reproduce

```sh
cd cuda/vlm && make test_cuda_vision
MM=/mnt/disk01/models/qwen36/27b/mmproj-F16.gguf
# CUDA (supported sizes only; --no-cpu skips the slow 1-thread reference)
for N in 512 768; do
  ./test_cuda_vision "$MM" --f16 --no-cpu --warmup 5 --iters 20 --image-size $N
done

# PyTorch reference (CUDA torch venv)
DL=/mnt/disk01/models/hf_cache/Qwen3-VL-30B-A3B-Instruct   # config.json + shard 13
../../ref/qwen_image/.venv/bin/python ../../rdna4/vlm/bench_pytorch_vision.py \
  --model-dir "$DL" --sizes 512,768,1024,2048 --warmup 5 --iters 20 --dtype f16
```

## Harness change

`test_cuda_vision.c` gained `--warmup N`, `--iters N`, and `--no-cpu` (steady-state
loop reporting mean/min ms + tok/s, mirroring `bench_pytorch_vision.py`). Each
`cuda_vision_encode` returns a fresh buffer (freed per iter); the runner reuses its
persistent device scratch, so the loop is safe.
