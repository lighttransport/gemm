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
| 512²   |   256  |         32.4 |        32.2 |       7892 |            96.0 |           69.6 |          2606 | **2.96× faster** |
| 768²   |   576  |         81.4 |        80.3 |       7078 |           212.7 |          184.5 |          2712 | **2.61× faster** |
| 1024²  |  1024  | — (unsupported) |          |            |           438.1 |          401.2 |          4354 |        —        |
| 2048²  |  4096  | — (unsupported) |          |            |          1518.7 |         1515.3 |          2701 |        —        |

CUDA: `--warmup 5 --iters 20`. PyTorch: `--warmup 5 --iters 20`.

> The CUDA encoder is now **>2.6× faster than PyTorch SDPA on the same GPU** at both 512²
> (2.96×, nearly 3×) and 768² (2.61×) — a full reversal of the original 4.9× / 10.4×
> deficit. (And
> PyTorch additionally runs the 3 deepstack mergers the GGUF path omits, so the real-work
> gap is even larger.) At 768² the dominant remaining cost is the FFN/projection cuBLAS
> GEMMs (≈52%, tensor-core, near-irreducible); the windowed-attention kernel is down to
> ≈12% and the 6-layer full-attention tensor-core path (QK + softmax_rows + P·V + extract)
> is ≈17%. `add_bias`, the standalone `gelu`, the cuBLASLt GELU side kernel, and the large
> FFN-down input cast are all gone (fused / written F16 in-pass); what remains is
> `gelu_f32_to_f16` (≈5.2%) + the smaller residual `cast_f32_f16` (≈2.4%).

## Fix applied #10: cache the interpolated position embedding per size

With the host gap closed (Fix #9), the per-phase profile exposed an anomaly: `pos` was
**2.9 ms at 512² but 0.1 ms at 768²** — backwards (bigger image, less pos cost). Cause:
the pos-embedding step has two paths. When the encode grid matches the model's native grid
(`gw == image_size/patch`, i.e. 48×48 at 768²) it uses a cheap direct-indirection kernel
(0.1 ms). At any other size (512² → 32×32 ≠ 48) it **bilinearly interpolates the pos
embedding on the CPU** — a `n_patches × dim` = 1024×1152 ≈ 1.2M-element triple-nested host
loop — then HtoDs the ~4.7 MB result. That interpolation was redone **every encode**, even
though the result is identical for a fixed image size.

Fix: cache it. `d_pos_interp` is a persistent device buffer that nothing else writes, so the
last interpolation stays resident. Added `pos_interp_w/h` (last interpolated grid, init −1);
when the current `(gw, gh)` matches, skip the CPU loop **and** the HtoD and just relaunch the
`add_pos_embd_direct` kernel reading the resident buffer. First encode at a size still
interpolates; a size change re-interpolates. Untouched by the CUDA graph (pos runs before the
captured block loop).

Result (f16, RTX 5060 Ti, `--warmup 5 --iters 20`): **512² 35.2 → 32.4 ms** mean (min 32.2,
**7892 tok/s**, `pos` 2.9 → 0.0 ms) → **2.96× faster than PyTorch** (nearly 3×). 768²
unchanged (81.4 ms — it uses the exact-match path, no interpolation). rel_L2 **bit-identical
0.6394475** (same interpolated bytes, just reused). Helps every non-native size.

## Fix applied #9: collapse the host-side result assembly (the real residual lever)

Fix #8 proved the in-loop launch overhead was tiny (~1.3 ms) and that the bulk of the
wall-vs-GPU gap was **untimed per-encode host work** in the encode tail. This fix removes
it. The old tail unconditionally did, for the 11.8 MB output (576 tokens × 5120 floats):
`calloc(total)` (zero-fill) **+** `malloc(mm_host)` (a second 11.8 MB buffer) **+** a
synchronous pageable `cuMemcpyDtoH` into `mm_host` **+** a per-token interleave-copy from
`mm_host` into `result` **+** two `free`s — even when there are no DeepStack features to
interleave. For this GGUF, `n_deepstack == 0` **always** (`ds_count == 0`), so all of that
staging+zero-fill+copy was pure overhead.

New tail: when `ds_count == 0`, `cuMemcpyDtoH(result, d_mm_out, …)` straight into the one
malloc'd output buffer — no zero-fill, no second buffer, no interleave, no extra frees. The
DeepStack interleave path is kept verbatim behind the `else` for models that actually
populate it. Measured tail cost dropped ~17 ms → ~2.4 ms; the win scales with output size
(more tokens = bigger savings), so 768² gains far more than 512².

Result (f16, RTX 5060 Ti, `--warmup 5 --iters 20`):

| size | before (mean / min) | after (mean / min) |
|------|---------------------|--------------------|
| 512² | 37.7 / 37.2 ms      | **35.2 / 34.9 ms** |
| 768² | 95.2 / 94.6 ms      | **81.5 / 80.5 ms** |

rel_L2 vs CPU of a replayed warm encode = **0.6394475**, bit-identical — the result bytes
are the same, only the path that assembles them changed. With this, the `t_*` phases now
**sum to the wall** at 768² (rgb 1.5 + patch 0.5 + pos 0.1 + rope 0.8 + vit 73.5 + mm 1.3
+ tail ~2.4 ≈ 80 ms): the untimed gap is closed and the `vit` GPU loop (≈73.5 ms) is the
floor. The only sizeable remainders are RGB HtoD (~1.5 ms) and the residual DtoH (~2.4 ms),
both candidates for pinned memory but small now.

## Fix applied #8: CUDA graph capture of the ViT block loop

The 27 ViT blocks issue ~288 host launches/encode (per-head full attention dominates:
6 full-attn layers × 16 heads × 3 launches). Those launches are now captured **once per
image size** into a CUDA graph and replayed with a single `cuGraphLaunch`, collapsing the
per-launch host overhead.

Mechanics (`cuda_vision_encoder.c`): the block loop was factored into
`vlm_run_vit_blocks()` so it can be issued normally, recorded under capture, or skipped
for replay. Encode **#1** runs normally (settles `cublas_mixed_ok`, warms cuBLAS
algos/workspace); encode **#2** captures via `cuStreamBeginCapture_v2`(THREAD_LOCAL) →
`cuStreamEndCapture` → `cuGraphInstantiateWithFlags`, then replays once to execute;
encode **#3+** replay. The pre/post window reorder+swaps are *net-identity* per encode, so
the loop sees the same physical `d_hidden`/`d_hidden2` every time — the baked graph
pointers stay valid across replays. cuBLAS is bound to the capture stream
(`cublasewCreate(.., r->stream)`) and its workspace is pre-allocated + algo-cached, so no
illegal on-stream alloc during capture. Capture is gated on `verbose<2` (in-loop debug
syncs/DtoH would break capture) and `n_deepstack==0` (the host-side `ds_count` can't
survive replay); any capture/instantiate failure disables graphs and falls back to
per-layer launches. `VLM_CUDA_GRAPH=0` disables; default ON.

Result (f16, RTX 5060 Ti, `--warmup 3 --iters 10`):

| size | graph off (mean / min) | graph on (mean / min) |
|------|------------------------|-----------------------|
| 512² | 38.2 / 37.6 ms         | **37.7 / 37.2 ms**    |
| 768² | 96.3 / 94.6 ms         | **95.2 / 94.6 ms**    |

rel_L2 vs CPU of a **replayed** warm encode = **0.6394475**, bit-identical to the
graph-off warm path — replay is numerically faithful.

**The launch overhead was only ~1.3 ms, not the ~20 ms #7 implied.** Graphs cut the
genuine in-loop launch cost (≈4.5 µs × 288) and reduce mean jitter, but the 768² *min*
is unchanged (94.6 ms) — the GPU loop (`vit` phase ≈ 73.4 ms) is the floor. The larger
gap the #7 nsys "21% idle" figure pointed at is **untimed per-encode host work**, not
launch latency: the encode tail does `calloc(11.8 MB)` + `malloc(11.8 MB)` + a synchronous
pageable `cuMemcpyDtoH(11.8 MB)` + a host interleave-copy for the result, plus the RGB
HtoD up front (none of which is in the `t_*` phase timers). That host-side result assembly
(≈17 ms at 768²) + transfers is the real residual lever — **fixed in Fix #9** by skipping
the staging buffer, zero-fill, and interleave when `n_deepstack==0` (768² 95.2 → 81.5 ms).
The window-map + rope precompute (recomputed every encode though size-invariant) is a
smaller cache-per-size opportunity that remains.

## Fix applied #7: fold the layernorm→GEMM input cast into LayerNorm (F16 output)

On the Blackwell F16×F16 path the qkv- and ffn-up-projection GEMMs each cast their F32
activation input to F16 before the GEMM (`cast_f32_f16`). Those two inputs are the LN1
and LN2 outputs, so a new VLM kernel `layernorm_f32_f16` writes the normalized result
**straight to F16** into `d_ln_buf_f16`, and the GEMM consumes it via `vlm_gemm_x_f16`
(no separate cast). Gated on `!cublas_mixed_ok` — the exact predicate under which
`vlm_gemm_ex` would have cast — latched once per layer so the LN output format and the
GEMM's expectation never disagree (layer-0's pre-flip LN stays F32, harmless). The F16
store replaces the F32 store, so LayerNorm costs the same; the cast pass disappears.

Profile at 768²: `cast_f32_f16` **83 → 29 launches/encode, 1.8 → 0.81 ms** (the 54
LN-fed casts gone); `layernorm_f32_f16` (53/enc, 2.04 ms) ≈ the old F32 LayerNorm.
rel_L2 vs CPU **bit-identical (0.6394475)** — the cast was happening either way.

**GELU could not be folded further.** `gelu_f32_to_f16` is already a *fused* gelu+cast
single pass; the only way to shrink it is to have the up-proj emit F16 so GELU reads
F16, but **cuBLASLt on sm_120 has no F16-output epilogue** — `BIAS` *and* `GELU_BIAS`
both fail the algo heuristic with F16 D (status 7). Probed and confirmed; that hardware
limit is exactly why `gelu_f32_to_f16` exists. So GELU stays at 4.0 ms (≈5.3%).

**Diagnostic — the encoder is now host-launch-bound.** Summing the nsys kernel trace at
768²: **GPU busy ≈ 75.7 ms/encode vs wall ≈ 95.5 ms → ~20 ms (21%) GPU idle** waiting on
CPU launches. The cast fold removed 54 launches/encode and ~1 ms GPU time, but the wall
barely moved because the GPU was already starved. The dominant launch source is the
**per-head full-attention loop: 6 layers × 16 heads × 3 launches (QK + softmax + PV) =
288 launches/encode**. Further per-kernel folds (e.g. the remaining 27 attn-out casts,
≈0.4 ms, which would need a duplicate F16-output attention kernel) will not move the wall
until that host overhead is collapsed — the real next lever is **CUDA graphs** (capture
the per-layer kernel sequence once, replay per layer) or batching the per-head launches.

> **Correction (see Fix #8):** the in-loop launch overhead turned out to be only ~1.3 ms,
> not ~20 ms. CUDA graphs recovered exactly that. The bulk of the "21% idle" was untimed
> per-encode **host** work (result `calloc`+pageable-`DtoH`+interleave of the 11.8 MB
> output, RGB HtoD), which the nsys GPU-busy-vs-wall comparison mis-attributed to launch
> starvation. The next lever is that host-side result assembly, not the launch path.

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
   CUDA is now **2.96× faster at 512² and 2.61× faster at 768²** than PyTorch on the same
   GPU. At 768² the only large reducible cost left is the FFN/projection cuBLAS GEMMs
   (≈52%, tensor-core — near the hardware floor); the windowed kernel is down to ≈12% and
   the 6-layer full-attention tensor-core path is ≈17% (spread across QK/softmax/P·V/extract,
   already on tensor cores). The qkv/ffn-up input casts are now folded into LayerNorm
   (Fix #7, F16 output). `gelu_f32_to_f16` (≈5.3%) **cannot** be folded further — sm_120
   cuBLASLt has no F16-output epilogue. The 288/encode per-layer launches are now captured
   into a **CUDA graph** and replayed (Fix #8) — that recovered the real in-loop launch
   overhead (~1.3 ms). The remaining wall-vs-GPU gap was **untimed host work** (11.8 MB
   result `calloc`+pageable-`DtoH`+interleave, RGB HtoD), not GPU launch starvation — now
   collapsed in **Fix #9** (skip the staging/zero-fill/interleave when `n_deepstack==0`),
   which closed the gap (768² 95.2 → 81.5 ms) so the `t_*` phases sum to the wall and the
   `vit` GPU loop (≈73.5 ms) is the floor. At non-native sizes the per-encode CPU pos-embed
   interpolation (≈2.9 ms at 512²) is now cached per size (**Fix #10**, 512² 35.2 → 32.4 ms,
   2.96× vs PyTorch). Residual untimed bits (RGB HtoD ~1.5 ms, output DtoH ~2.4 ms) are
   pinned-memory candidates but small.

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
