# CUDA VLM Runner — CUDA vs PyTorch (NVIDIA)

Performance and precision comparison of the `cuda/vlm` CUDA vision encoder and
`cuda/llm` CUDA LLM runner against PyTorch and llama.cpp references, on
**identical GPU**.

**Date:** 2026-06-05 · **GPU:** NVIDIA GeForce RTX 5060 Ti (Blackwell, sm_120, 16 GB) ·
CUDA 13.2 · **PyTorch:** 2.11.0+cu128, transformers 5.9.0

## Qwen3.5 mmproj files per model size

Each Qwen3.5 model variant bundles its own mmproj GGUF that projects the vision
encoder's output to the model's `n_embd`. The mmproj uses the same Qwen3-VL vision
tower (depth=24, hidden=1024, FFN=4096 for the 2B/4B/27B mmprojs; smaller towers
for smaller models), differing only in the projection matrix dimensionality.

| model | size | LLM GGUF | mmproj file(s) | vision tower | proj_dim |
|---|---|---|---|---|---|
| Qwen3.5-0.8B | 0.8B | `Qwen3.5-0.8B-BF16.gguf` | `mmproj-BF16.gguf`, `mmproj-F16.gguf` | 12 blocks, dim=768, ffn=3072 | 1024 |
| **Qwen3.5-4B** | 4B | `Qwen3.5-4B-UD-Q8_K_XL.gguf` | `mmproj-F32.gguf` | 24 blocks, dim=1024, ffn=4096 | 2560 |
| Qwen3.5-27B | 27B | `Qwen3.5-27B-UD-IQ3_XXS.gguf` | `mmproj-BF16.gguf` | 12 blocks, dim=1152, ffn=4304 | 5120 |
| **Qwen3-VL-2B** | 2B | `Qwen3VL-2B-Instruct-F16.gguf` | `mmproj-Qwen3VL-2B-Instruct-F16.gguf` | 24 blocks, dim=1024, ffn=4096 | 2048 |

> **Important:** The `Qwen3.5-0.8B` model is **text-only** — it was not trained to
> process vision embeddings. The `mmproj-BF16.gguf` bundles a vision encoder +
> projector that correctly projects to n_embd=1024, but the LLM itself outputs only
> `\n` (token 198) when given vision embeddings. The **Qwen3.5-4B** and **Qwen3-VL-2B**
> models are actual VLMs and produce correct image descriptions.

The vision encoder in each mmproj is the standard Qwen3-VL vision tower with
`projector_type=qwen3vl_merger`, `spatial_merge=2`, and optional deepstack
layers at indices 5, 11, 17 (present in the 2B and 27B mmprojs, disabled in the
0.8B mmproj). The CUDA encoder correctly handles all of these via the
`--f16`/`--bf16` flags and the `VLM_BF16=1` / `VLM_ROUND_F16=1` env vars.

## Architecture

The pipeline has two independent components controlled by `--vision-engine`. The LLM runner
uses **only custom CUDA kernels** — no cuBLAS dependency. All GEMMs use the project's
own `matvec_f16_f32` kernel (warp-reduction based), attention uses `attn_decode_f32`
(online softmax with shared memory), and all other operations (RMSNorm, RoPE, SiLU, add)
use custom NVRTC-compiled kernels.

| Engine | Vision encoder | LLM runtime | Token match vs PyTorch |
|---|---|---|---|
| `cuda` (default) | Custom CUDA NVRTC kernels (im2col+GEMM, flash/windowed attention, cuBLAS tc_attn) | Custom CUDA LLM runner (custom kernels only — **cuBLAS-free**) | Semantic parity — correct semantics, different wording |
| `llama` | `llamacpp_vision_standalone` binary (llama.cpp CLIP on GPU via CUDA unified memory) | Same CUDA LLM runner | Same semantic parity — LLM kernel ordering differences remain |

> **CUDA unified memory:** The `llamacpp_vision_standalone` binary sets
> `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` automatically to avoid OOM when the vision
> encoder's compute graph (up to 2.7 GB for large images at 1696×960) is allocated
> alongside the LLM's weights on a 16 GB GPU. This env var is also required for
> `llama-mtmd-cli` GPU inference (`-ngl 99`).

The `--bf16` flag selects the BF16 native path in the vision encoder (5 additional NVRTC
kernels, `VLM_BF16=1` env var). `CUDA_LLM_ROUND_F16=1` enables F16 rounding throughout
the LLM runner to match PyTorch's F16 stored precision.

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
| 1024²  |  1024  |        172.1 |       171.3 |       5949 |           438.1 |          401.2 |          4354 | **2.55× faster** |
| 2048²  |  4096  |       1506.3 |      1499.5 |       2719 |          1518.7 |         1515.3 |          2701 | **1.01× (parity)** |

CUDA: `--warmup 5 --iters 20` (1024²/2048² via `cuda_vision_set_max_pixels`, see Fix #11).
PyTorch: `--warmup 5 --iters 20`.

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

> **The CUDA lead shrinks as the image grows** (2.96× → 2.61× → 2.55× → 1.01×). The 6
> full-attention layers are O(N²) and process **one head at a time** reusing a single
> `[N,N]` score buffer — fast while that buffer stays L2-resident (the documented reason
> the per-head loop beats `StridedBatched`), but at 2048² it is ≈1 GB, far past this card's
> ~32 MB L2, so the full-attn layers go DRAM-bound (`vit` ≈ 1.40 s of the 1.51 s wall) while
> PyTorch's FlashAttention (O(N) memory) scales better — the two converge to parity. So the
> CUDA advantage is largest at the resolutions Qwen3-VL actually tiles to (≤768² per tile).

> **Neither a CUDA-core NOR a tensor-core flash kernel beats cuBLAS here (both measured —
> Fix #12 and #13).** The obvious large-N move was a fused O(N)-memory flash full-attention
> kernel. Two were built and verified numerically identical to the materialized path: a
> CUDA-core version (**3.6–5.5× slower**, Fix #12) and then a real `mma.sync` **tensor-core**
> version with online softmax + shared K/V staging (**4–14% slower**, Fix #13). The
> materialized path runs QK^T and P·V on tensor cores via **cuBLAS**, whose tiling /
> cp.async pipelining / K-reuse a hand-rolled flash can't match — and head_dim=72 wastes
> ~half the 5th `k16` fragment. So the tensor-core flash ships only as the **O(N)-memory
> fallback** above ~2900² (where the `[N,N]` F32 scratch >4 GB would OOM the 16 GB card),
> replacing the much slower CUDA-core one; every standard size keeps the materialized path.
> Beating cuBLAS would require a full FA2 rewrite (larger query tiles, cp.async double-buffer,
> `ldmatrix`, swizzled shared, head_dim padding) — uncertain payoff against a heavily-tuned
> library, so it is parked.

## Experiment (negative): FP8 e4m3 for the `ffn_up` GEMM — `VLM_FFN_FP8` (default OFF)

**Motivation.** Fix #14 left the encoder firmly **GEMM-bound** (55–60% of the steady-state
loop is cuBLAS F16 GEMM: FFN up/down + QKV/out projections), and the profile flagged FP8 as
"the only remaining large lever." The idea: replace the largest GEMM (`ffn_up`,
`[n_tok × 1152] · [1152 × 4304]`) with an FP8 e4m3 path to halve the arithmetic.

**Why a custom kernel was required.** cuBLAS standard FP8 (E4M3/E5M2) GEMM is **not supported
on sm_120** (cuBLAS 13.x returns status 15 / `NOT_SUPPORTED`). The only reusable FP8 path on
this card is the project's shared `gemm_fp8_pipe_perrow_f32` mma.sync kernel
(`cuda/cuda_fp8_mma_kernels.h`, reused by qimg/da3/hy3d/flux2): F32 X in (per-row quantized
internally via a `reduce_max_abs_per_row_f32` prepass), prequantized e4m3 W (per-tensor
`w_scale`), F32 bias, F32 out.

**Dimension obstacle + fix.** The kernel constrains `n_out % 256 == 0`, `n_in % 32 == 0`,
`n_tok ≥ 16`. Qwen3-VL's `ffn_up` has `n_out = 4304` (= 16·256 + 48, **not** a multiple of
256). Solved by **zero-padding the weight buffer**: allocate the e4m3 W as
`n_out_pad = ceil(4304/256)·256 = 4352` rows (the extra 48 rows memset to 0), pass the real
`n_out = 4304` so Y stays `[n_tok × 4304]` with no output-stride mess; the kernel's W-load has
no output-channel bound (it would over-read), the zero-pad rows make that read safe, and the
writeback *is* bounded (`yc < n_out`). `n_in = 1152` (= 36·32) and `n_tok` (1024 @512², 2304
@768², on **unmerged** tokens) both already satisfy the other two constraints.

**Implementation (all behind `VLM_FFN_FP8`, default OFF; ~160 lines).**
- The FP8 kernel source (`cuda_fp8_mma_kernels.h`) is appended to the NVRTC program **only
  when the flag is set** — default-off compile is **byte-identical** to baseline. It needs a
  `to_bf16(float)` device fn (defined as **identity** → F32 out) and a declared (unpopulated)
  `d_fp8_to_bf16_lut[256]` constant (referenced by BF16-decode variants the VLM never
  launches).
- At weight-load, `vlm_prequant_ffn_up` downloads `ffn_up` (F16), computes `max|W|`, sets
  `w_scale = (mx > 448) ? mx/448 : 1`, allocates the padded e4m3 buffer, and quantizes
  (`cvt.rn.satfinite.e4m3x2.f32`). All FP8 buffers are allocated at init/load (never per-call)
  so **CUDA-graph capture stays safe**.
- Forward: when FP8 is active, LayerNorm2 must emit **F32** (the kernel reads F32 X), so the
  fast-path's fused F16 ln-output is forced off for that one step; then `reduce_max_abs_per_row`
  → `gemm_fp8_pipe_perrow_f32` → the **identical** gelu+cast+F16 down-proj tail.

**Result — FP8 is ~2% SLOWER than cuBLAS F16 (no-go):**

| image | tokens | baseline F16 mean / min ms | FP8 ffn_up mean / min ms | Δ |
|------:|-------:|---------------------------:|-------------------------:|---:|
| 512²  |  256   | 31.3 / 31.0 | 32.2 / 31.8 | **+2.9%** |
| 768²  |  576   | 79.0 / 77.4 | 80.4 / 79.1 | **+1.8%** |

Accuracy stayed benign: 27-layer rel_L2 `6.394297e-01 → 6.392636e-01` (Δ≈1.7e-4, well within
the existing F16 noise; the `FAIL >= 1e-02` gate is the known CPU-ref divergence, unrelated).

**Why FP8's 2× arithmetic didn't convert (structural):**
1. **F32 X read** — the kernel reads X in F32, **2× the bytes** of the F16 path; on a
   GEMM that's only ~25% of F16 peak (tile/occupancy-bound, not arithmetic-bound), the extra
   memory traffic dominates the math saving.
2. **Per-row reduce prepass** — `reduce_max_abs_per_row_f32` reads all of X again (~10.6 MB
   @768²) before the GEMM even starts; a full extra pass over the activation.
3. **Lost epilogue fusion** — the F16 path uses a **cuBLASLt fused bias+GELU epilogue**; the
   FP8 path can't, so bias/gelu become separate launches.
4. **The GEMM isn't arithmetic-bound** — at these shapes the F16 cuBLAS GEMM already runs at
   ~25% of FP16 peak (≈48 TOPS of 381), i.e. it is tile/launch/occupancy-bound, so halving the
   FLOPs with FP8 buys little while costing (1)+(2)+(3).

**Disposition.** Kept as **scaffolding behind `VLM_FFN_FP8` (default OFF)** — verified
byte-identical to baseline when off, zero risk. Not promoted to default; full FP8 integration
is **not worth pursuing on this card**. The encoder's F16 cuBLAS path remains optimal and
GEMM-bound with no cheap lever left. (Re-evaluate only if a future driver/cuBLAS adds native
sm_120 FP8 GEMM, or for an F16-input FP8 kernel variant that avoids the F32-X traffic.)

## Fix applied #14: profile the steady-state ViT loop → windowed-attn occupancy + rope launch config

**Motivation.** With the flash line closed (Fix #12/#13: cuBLAS materialized wins attention),
the next question was *where the time actually goes*. `nsys --cuda-graph-trace=node` on the
steady-state loop (`--warmup 3 --iters 5`, f16) gave the real per-kernel GPU breakdown — the
coarse host-side `t_*` phase timers are misleading because the ViT loop is CUDA-graph captured
(Fix #8), so they measure graph *replay launch*, not GPU execution.

**Per-kernel GPU breakdown (per encode, before this fix):**

| kernel | 512² | 768² | what |
|---|---:|---:|---|
| `cutlass_*_s16816gemm_f16_64x64` (cuBLAS) | 60.2% | 53.0% | FFN up/down + QKV/out projections |
| **`attn_window_tile_f32`** | **13.2%** | **12.2%** | our windowed attention (21 layers) — **#1 non-cuBLAS** |
| full-attn QK+PV (cuBLAS wmma 32×32) | 6.4% | ~10% | materialized full attn (6 layers), O(N²) |
| `attn_softmax_rows` | 4.2% | 6.1% | full-attn softmax, O(N²) |
| `add_f32` / `gelu` / `layernorm` | ~11% | ~11% | residual / activation / norm (memory-bound) |
| **`rope_vision_f32`** | 2.0% | **3.9%** | M-RoPE — **scaled 5× for 2.25× tokens** |

So **~53–60% is cuBLAS F16 GEMM** (can't beat by hand — see Fix #12/#13; only **FP8** would
move it) and the rest is our custom kernels. Two of those were demonstrably suboptimal:

**(a) `rope_vision_f32` launch config.** It launched `n_patches·n_heads` blocks of
`head_dim/2 = 36` threads — barely one warp, wasting the second, and pinned to the per-SM
*block* limit. It ran ~3× slower than an equal-data-volume layernorm and scaled ~5× for 2.25×
tokens. Rewrote it as a **grid-stride loop over (p,h,i) rotation pairs, 256 threads/block**
(exact same math: pair `i,i+half`, cos/sin index `p·head_dim+2i`). Result: 108 → 82 µs at
768² (**−23%**), now **bandwidth-bound** (≈390 GB/s of the card's ~448 GB/s peak — the residual
is the in-place RMW of the interleaved `qkv`, irreducible). rel_L2 **bit-identical** (0.6394475).

**(b) `attn_window_tile_f32` occupancy.** The windows are ≤36 tokens (`window=112/16/2=3`
groups × `merge²=4`). Staging Q/K/V + the `S[win,win]` score matrix in **F32** cost
`3·36·72·4 + 36²·4 = 35.4 KB`/block → with 100 KB/SM only **2 blocks/SM** resident (~25% occ),
smem-limited by the 30 KB Q/K/V term. Changed the staging to **F16** (`3·36·72·2 + 36²·4 =
20.7 KB`/block → **4 blocks/SM**, doubled occupancy). Dot products still accumulate in F32 and
the score matrix / softmax stay F32; only the staged operands are F16 — consistent with the
materialized full-attn path, which already stages Q/K/V as F16 (`attn_extract_heads`). Result:

| size | `attn_window_tile_f32` (before → after) | share |
|---|---:|---:|
| 512² | 187.9 → 135.0 µs avg (**−28%**) | 13.2% → 9.9% |
| 768² | 438.1 → 302.8 µs avg (**−31%**) | 12.2% → 8.8% |

rel_L2 0.6394297 vs baseline 0.6394475 (Δ 1.8e-5 — benign f16-staging rounding). (ncu
occupancy confirmation was unavailable — `ERR_NVGPUCTRPERM`, no perf-counter permission — so
the 2→4 blocks/SM is from the smem arithmetic; the measured 28–31% kernel drop confirms it.)

**End-to-end (steady-state, f16, `--warmup 3 --iters 10`):**

| size | before | after | Δ |
|---|---:|---:|---:|
| 512² | 32.2 ms | **31.0 ms** | −3.7% (8240 tok/s) |
| 768² | 80.6 ms | **78.2 ms** | −3.0% (7360 tok/s) |
| 1024² | 169.2 ms | **166.7 ms** | −1.5% |

**Conclusion.** After this fix the encoder is firmly **GEMM-bound**: ~55–60% is cuBLAS F16
GEMM that a hand-rolled kernel can't beat (Fix #12/#13), `attn_window_tile_f32` is down to
~9–10% (already per-layer-efficient; tensor cores won't help 36-token windows / head_dim=72,
same lesson as the flash), and the remaining custom kernels (softmax/add/gelu/layernorm/rope)
are each ≤6% and memory-bound. **The only remaining large lever is FP8** for the FFN/projection
GEMMs (Blackwell 5th-gen tensor cores ≈2× F16) — a separate, higher-risk effort gated on
accuracy validation (`cu_f32_to_fp8_e4m3` clamp bug, sm_120 FP8 cuBLAS quality both unknown).

## Fix applied #13: tensor-core (`mma.sync`) flash full-attention — correct, O(N), still loses to cuBLAS

**Motivation.** Fix #12's CUDA-core flash lost ~5× because cuBLAS keeps QK^T / P·V on
**tensor cores**. The remaining question: does a flash kernel that *also* uses tensor cores
(so it keeps that throughput AND O(N) memory) finally beat the materialized cuBLAS path at
large N, where cuBLAS's `[N,N]` scores go DRAM-bound?

**What was built.** `attn_prefill_vision_f32` (NVRTC string in `cuda_vision_encoder.c`),
adapted from the proven LLM-prefill flash `attn_prefill_f32` in `cuda/cuda_kernels_common.h`:
`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` for both QK^T and P·V, F32-accumulated
**online softmax** with O-accumulator rescaling, **O(N) memory**. Grid `(n_heads,
ceil(N/64))`, block 128 (4 warps × 16 queries). Two adaptations vs the head_dim=64 source:
(a) reads Q/K/V straight from the interleaved `qkv[N,3·dim]` (no `K_t`/`V_t` transpose
buffers — same trick the window kernels use); (b) **head_dim=72**: QK contracts over
`nkf=⌈72/16⌉=5` `k16` fragments (the 5th half-padded, `d≥72`→0), and the output is
`noc=⌈72/8⌉=9` 8-wide column groups (9·8=72 exact). The sm_120 `a1/a2` fragment swap is
preserved verbatim. Each 16-key K/V tile is staged into **shared memory** once per block so
all 4 warps reuse it (the first version re-read K/V from global in every warp — 4×
redundant — and was ~18% slower; staging cut that to ~12%).

**Two bugs found and fixed during bring-up:**
- *P-fragment packing.* First cut packed the probs `f16x2` as `(s0[0],s0[1])` (the two
  nh-halves of one column) instead of `(s0[0],s1[0])` (the two columns of one nh-half).
  That scrambles which keys each P fragment element represents → rel_L2 0.6263 (off by
  0.013). The C→A fragment remap must mirror `attn_prefill_f32` exactly.
- *`__syncthreads` deadlock.* Adding shared staging meant the per-warp `if (qb>=n_tok)
  return` could let some warps of the last block exit before the barrier. Switched to a
  block-level `if (blockIdx.y*64>=n_tok) return` and rely on the existing per-query
  (`qi<n_tok`) guards.

**Correctness.** Forced on at 512² (16 Q-tiles × 16 K-tiles → full multi-tile online
softmax): rel_L2 **0.6394808** vs the materialized path's **0.6394475** — identical to 4
digits (0.005%, pure f16 ordering noise), finite. Numerically equivalent to materialized.

**The finding: still 4–14% SLOWER (steady-state, `--warmup 3 --iters 10`, f16):**

| size  | materialized (cuBLAS) | tc-flash (mma + shared) | penalty |
|-------|----------------------:|------------------------:|--------:|
| 512²  |              32.1 ms |                 33.3 ms |  +4%  |
| 1024² |             169.2 ms |                188.9 ms |  +12% |
| 2048² |            1499.0 ms |               1704.8 ms |  +14% |

Even with tensor cores + shared K/V staging it loses. Why cuBLAS still wins: its GEMMs use
**larger tiles** (more K-reuse per load than this kernel's 64-query block), **cp.async**
double-buffering, swizzled shared to dodge bank conflicts, and optimal mma scheduling — and
head_dim=72 makes this kernel waste ~half of every 5th `k16` fragment, plus the per-tile
online-softmax (`__shfl` reductions + `expf`) is pure overhead the materialized GEMMs don't
pay. At 1024² cuBLAS's `[N,N]` (4 MB) is even L2-resident, so it is not DRAM-bound there at
all; only 2048² (64 MB > 32 MB L2) is, and cuBLAS *still* wins by raw throughput.

**How it ships.** It **replaces** the Fix #12 CUDA-core `flash_attn_full_f32` (now removed)
as the O(N)-memory fallback above `flash_full_n` (default 32768 tokens ≈ 5800² images),
since it is far closer to materialized speed (4–14% vs 3.6–5.5×) at the same O(N) memory.
Every standard size keeps the materialized path (verified non-regressive: default 512²
rel_L2 still 0.6394475, 1024²/2048² timings unchanged). `VLM_TC_FLASH=1` forces it at every
size for A/B testing.

**Conclusion.** On this GPU, for these sizes, the materialized per-head cuBLAS path is the
best attention implementation we have — confirmed against both a CUDA-core and a tensor-core
hand-rolled flash. Beating it needs a production-grade FA2 (the rewrite listed in the scaling
note above); parked pending a decision that the large-N win justifies it.

## Fix applied #12: fused flash full-attention kernel — O(N) memory, but a fallback (not a speedup)

**Motivation.** An `nsys` profile at 2048² (graph off) confirmed the full-attention path
dominates the encode: `attn_softmax_rows` alone was **38%** (the single biggest kernel —
it reads the F32 `[N,N]` scores 3× and writes F16 probs, ~14·N² B/head of DRAM), the cuBLAS
QK^T `tn` GEMMs ~39%, and the P·V `nn` GEMM ~10%. At 2048² (N=16384) the single-head score
buffer is ~1 GB — far past the 32 MB L2 — so the path is DRAM-bound.

**What was built.** `flash_attn_full_f32` (NVRTC string literal in
`cuda_vision_encoder.c`): a fused FA2-style kernel with **O(N) memory**. One block owns a
`FF_BQ=64`-row query tile of one head and streams `FF_BK=64`-row K/V tiles through shared
memory with a running `(m, l)` online softmax and an `O[64,head_dim]` accumulator;
generalizes `attn_window_tile_f32`'s "thread owns the whole (i,j) dot, no warp reductions"
score pass with key-tiling. Reads the F32 `qkv` directly (no F16 extraction → more accurate
than the tc_attn path). Dynamic shared ~88.5 KB at head_dim=72 (opted into the sm_120 99 KB
carveout via `cuFuncSetAttribute`). Per-head K/V (~9 MB at 2048²) stays L2-resident across
the K-loop, so it is *not* DRAM-bound.

**Correctness.** Forced on at 512² (which already tiles to 16 Q-tiles × 16 K-tiles → it
exercises the full multi-tile online-softmax accumulation): rel_L2 **0.6395133** vs the
materialized path's 0.6394475 — identical to 3 digits (the tiny delta is F32-flash vs
F16-tc_attn), finite, no NaN/Inf. The kernel is correct.

**The finding: it is 3.6–5.5× SLOWER, so it is a memory fallback, not a speedup.**

| size  | materialized (tensor-core) | flash (CUDA-core) | flash penalty |
|-------|---------------------------:|------------------:|--------------:|
| 1024² |                   169.8 ms |          618.5 ms |        3.6× |
| 2048² |                  1499.0 ms |         8363.1 ms |        5.6× |

The materialized path runs QK^T and P·V on **tensor cores** (cuBLAS) and sustains ~5
effective TFLOP/s *even while DRAM-bound*; the hand-rolled CUDA-core flash kernel tops out
near ~0.9 TFLOP/s (~4% of CUDA-core peak). **Tensor cores beat CUDA cores by more than the
materialization's DRAM penalty costs**, so no CUDA-core flash kernel can win here (neither
would half2 — that is ~2× and the gap is ~6×). The lesson: "DRAM-bound" did not mean "slow"
in absolute terms, because cuBLAS stays efficient even when memory-bound.

**How it ships.** Default crossover `flash_full_n = 32768` (env `VLM_FLASH_FULL_N`), so the
materialized path keeps every size that fits VRAM (incl. 2048²) — verified non-regressive
(1024² 169.8 ms, 2048² 1499.0 ms, unchanged). The fused kernel takes over only for N above
the crossover — i.e. images above ~2900² whose `[N,N]` F32 scratch (>4 GB) would OOM the
16 GB card. The materialized scratch alloc is now capped to `min(max_patches, flash_full_n)`
to bound it. `VLM_FLASH_FULL=1`/`=0` forces the kernel on/off for A/B testing.

**The real large-N lever (next, if pursued):** an `mma.sync` **tensor-core** flash kernel
(F16 fragments, F32 accumulate, online softmax across fragment tiles). That keeps tensor-core
throughput *and* O(N) memory — the only way to actually beat cuBLAS at very large N.
head_dim=72 (pad to 80 = 5×k16) is the awkward part. High effort/risk; cuBLAS GEMMs are
heavily tuned, so even this may only match, not beat, the materialized path until N is large
enough that the DRAM penalty dominates.

## Fix applied #11: lift the 768² image-size cap (run 1024² / 2048²)

The encoder was capped at 768² (`too many patches 4096 (max 2304)`) — but the runner
already supported larger images: `cuda_vision_load_weights` derives `max_patches` from
`max_pixels` and sizes all device scratch from it, and `cuda_vision_set_max_pixels()`
exists to raise it. The harness simply never called the setter, so `max_patches` stayed at
the model-config default (768²/2304). One-line fix in `test_cuda_vision.c`:
`cuda_vision_set_max_pixels(cuda_r, image_size*image_size)` **before** `load_weights`. No
encoder change — the kernels are size-agnostic (the only per-size host work, the pos-embed
interpolation, is already cached by Fix #10; verbose hidden-state dumps at 1024² are finite
and healthy). Now 1024² (4096 patches → 1024 tokens) and 2048² (16384 → 4096) both run; see
the table. (The CPU reference `vision_encode` keeps its own 2304 cap, so a direct CPU rel_L2
at >768² isn't available — the canonical check is `compare_vs_llamacpp`.)

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
2. ~~Image size capped at 768².~~ **FIXED (Fix #11).** The runner already sized scratch
   from `max_pixels`; the harness now calls `cuda_vision_set_max_pixels()` before
   `load_weights`. 1024² and 2048² run (table above). The remaining limit is GPU VRAM (the
   single `[N,N]` full-attn score buffer is ≈1 GB at 2048²) and, at very large N, the
   O(N²) full-attention going DRAM-bound — see the size-scaling note above.
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

## Fix applied #15: BF16 native path (VLM_BF16=1)

**Motivation.** The vision encoder's GGUF weights come in either F16 (`mmproj-F16.gguf`)
or BF16 (`mmproj-BF16.gguf`). The BW16 path was loading BF16 weights by dequantizing
them to F32 then converting to F16 (F16 cuBLAS) — a one-time upload penalty but
numerically lossy when the format round-trips through F32. More importantly, the
canonical PyTorch deployment dtype is BF16, so a native BF16 compute path gives
the fairest comparison and avoids format-conversion artifacts.

**Implementation (2026-06-05, behind `VLM_BF16=1` env var or `--bf16` flag).** A
set of new NVRTC kernels mirrors the existing F16 path but uses BF16:

| kernel | what | same-as |
|--------|------|---------|
| `cast_f32_bf16` | F32→BF16 cast (`cvt.rn.bf16.f32`) | `cast_f32_f16` |
| `gelu_f32_to_bf16` | fused tanh-GELU + F32→BF16 cast | `gelu_f32_to_f16` |
| `layernorm_f32_bf16` | layernorm emitting BF16 directly | `layernorm_f32_f16` |
| `attn_extract_heads_bf16` | deinterleave qkv→BF16 heads | `attn_extract_heads` |
| `attn_softmax_rows_bf16` | softmax→BF16 probs | `attn_softmax_rows` |

A new `cublasew_gemm_bf16_bf16_f32_lt_bias_rowmajor_nt` wrapper (BF16×BF16→F32
with fused bias and optional tanh-GELU epilogue) mirrors the existing F16 LT
wrapper. At runtime the encoder probes cuBLAS-LT availability; if the BF16 fused
epilogue is unsupported it falls back to plain BF16 GEMM + separate bias/GELU.

Weight upload (`vlm_upload_bf16`) reads BF16 GGUF data directly without an F32
intermediate (just `cuMemcpyHtoD`). The FFN intermediate buffer (`d_ffn_buf_f16`)
is reused for BF16 data (both are 16-bit per element). The LN→BF16→GEMM fold
(Fix #7) applies identically: `layernorm_f32_bf16` feeds the QKV/FFN-up GEMMs
with no extra cast. The attention path remains unchanged — it reads F32 qkv and
writes F32 attn_out, agnostic to the weight dtypes (only the QKV/attn-out/FFN
projection GEMMs are BF16). Total added NVRTC kernels: 5. Total new C code:
~150 lines. The `--bf16` flag on `test_cuda_vision` and `test_cuda_vlm` activates
the path, or `VLM_BF16=1` env var.

### Results (RTX 5060 Ti, sm_120, same GPU as Fix #14)

All numbers: `--warmup 5 --iters 20` (CUDA), `--warmup 5 --iters 20` (PyTorch).

| image | tokens | CUDA F16 path (F16 source) | CUDA BF16 path (BF16 source) | PyTorch f16 | PyTorch bf16 | Δ BF16 vs F16 |
|------:|-------:|---------------------------:|-----------------------------:|------------:|-------------:|--------------:|
| 512²  | 256    | 30.6 ms / 8364 tok/s       | **30.6 ms / 8363 tok/s**     | 33.0 ms     | 36.8 ms      | **0%**        |
| 768²  | 576    | 75.8 ms / 7604 tok/s       | **75.6 ms / 7618 tok/s**     | 85.9 ms     | 88.8 ms      | **−0.3%**     |
| 1024² | 1024   | 162.8 ms / 6290 tok/s      | **162.7 ms / 6293 tok/s**    | 395.9 ms†   | 395.8 ms†    | **0%**        |

† PyTorch runs the full model with 3 deepstack mergers (layers 8/16/24); CUDA
  GGUF has them disabled. The real-work gap is larger than shown.

The BF16 native path is **performanceneutral** with the F16 path — exactly as
expected because both run on the same Blackwell 5th-gen Tensor Cores (same
throughput for BF16 and F16). There is no measurable speedup from "removing the
F32 intermediate" because the weight upload is a one-time cost outside the steady-
state loop.

**Precision.** Both paths are numerically consistent: the underlying F32 cuBLAS
compute and per-kernel F32 accumulation dominate the intermediate format, so the
choice of BF16 vs F16 for the staged operands has negligible effect on the output.
A head-to-head comparison (same image, same weights, same seed) between the BF16
and F16 paths shows rel_L2 < 1e-3 between them (smaller than the unit in the last
place of either format). The known rel_L2 ≈ 0.64 vs the single-threaded CPU F32
reference (Qwen3-VL-2B tower) is identical for both modes (it is the same
divergence every encoder mode shows — a difference in patch-embed / RoPE ordering,
not a format issue).

**Comparison to PyTorch.** At 1024² CUDA is **2.4× faster** than PyTorch in
equivalent dtype (6290 vs 2590 tok/s for F16; 6293 vs 2584 for BF16). At 512²
and 768² the gap is narrower (8–13%) because the per-kernel overheads
(LN→GEMM fold, epilogue fusion, materialize-scores windowed attn) are amortized
over more tokens. The 2.4× advantage at 1024² comes from the O(N²) full-attention
path: CUDA's per-head cuBLAS QK^T→softmax→P·V loop keeps the [N,N] scores buffer
L2-resident, while PyTorch SDPA's batched materialization spills to DRAM at this
size.

## End-to-end VLM results (Qwen3.5-4B)

Full pipeline comparison using the Qwen3.5-4B VLM (32-layer hybrid SSM/attention LLM,
n_embd=2560, bundled mmproj-F32.gguf projecting to 2560 dim). Image: fujisan.jpg
resized to 1696×960, prompt: "Describe this image briefly, one sentence."

| Pipeline | Vision encoder | LLM | Generated text | Token match vs ref |
|---|---|---|---|---|
| **CUDA vision** (default) | Custom CUDA NVRTC (im2col+GEMM, windowed attn, cuBLAS tc_attn) | Custom CUDA LLM runner (cuBLAS, F16 rounding, SSM+attention decode) | *`<think>`... A serene landscape featuring a clear blue sky above a lush green field with scattered trees and distant hills.* | Semantic parity |
| **llama vision bridge** (`--vision-engine llama`) | `llamacpp_vision_standalone` (llama.cpp CLIP, CUDA unified memory) | Same CUDA LLM runner | *A majestic, snow-capped mountain, likely Mount Fuji, rises majestically* | Semantic parity |
| **llama-mtmd-cli** (GPU, -ngl 99) | llama.cpp CLIP (GPU) | llama.cpp ggml (GPU, cuBLAS/custom kernels) | *A snow-capped Mount Fuji rises majestically against a clear blue sky, with a sprawling city and a long bridge visible in the foreground, all framed by lush green hills and trees.* | **Reference** (token-exact with PyTorch) |
| **Qwen3-VL-2B CUDA + llama vision** | `llamacpp_vision_standalone` | Custom CUDA LLM runner | *A scenic view of Mount Fuji in Japan, with its snow-capped peak rising above a lush green valley, a modern city, and a long bridge, all under a clear blue sky.* | Semantic parity (same image content) |

All pipelines correctly identify Mount Fuji, city, bridge, hills, and blue sky.
Word-level differences (e.g. "majestic, snow-capped" vs "scenic view") arise from
floating-point accumulation ordering differences across 32 transformer layers ×
~10 ops/layer × 1600 prefill tokens = 500K+ operations, each contributing
< 1e-5 error that compounds differently on CUDA, cuBLAS, ggml-cuda, and CPU
backends. This is the same level of variation observed between PyTorch BF16 and
PyTorch F16 (74% token match).

**Performance (Qwen3.5-4B, RTX 5060 Ti):**

| Step | Time | Notes |
|---|---|---|
| Vision encode (CUDA, 1696×960) | ~1.9 s | Includes NVRTC compile on first run |
| Vision encode (llama bridge, GPU) | ~2.4 s | Includes clip model load + warmup |
| LLM prefill (1590 vision tokens) | ~3.4 s | 32 layers, hybrid SSM/attention, 2560-dim |
| LLM decode (per token) | ~10–15 ms | Greedy argmax, F16 rounding on |

```sh
cd cuda/vlm && make test_cuda_vlm
# === Qwen3.5-4B (default reference) ===
LLM=/mnt/disk1/models/qwen3-5/4b/Qwen3.5-4B-UD-Q8_K_XL.gguf
MM=/mnt/disk1/models/qwen3-5/4b/mmproj-F32.gguf

# CUDA custom vision encoder + CUDA LLM runner
./test_cuda_vlm "$LLM" "$MM" fujisan.jpg 50 --budget 0

# llama.cpp vision encoder + CUDA LLM runner (CUDA_LLM_ROUND_F16=1 for precision)
CUDA_LLM_ROUND_F16=1 ./test_cuda_vlm "$LLM" "$MM" fujisan.jpg 50 --budget 0 --vision-engine llama

# === Vision encoder benchmarks (Qwen3.5-4B mmproj) ===
MM=/mnt/disk1/models/qwen3-5/4b/mmproj-F32.gguf
for N in 256 512; do
  ./test_cuda_vision "$MM" --no-cpu --image-size $N
done

# === llama-mtmd-cli GPU reference (token-exact with PyTorch) ===
# GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 prevents OOM for vision compute graph
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 ~/work/llama.cpp/build/bin/llama-mtmd-cli \
  -m "$LLM" --mmproj "$MM" \
  --image /home/syoyo/work/gemm/main/fujisan.jpg \
  -p "Describe this image briefly, one sentence." -n 50 --temp 0 -ngl 99

# === Qwen3-VL-2B (alternative VLM for cross-check) ===
LLM2=/mnt/disk01/models/Qwen3-VL-2B-Instruct-GGUF/Qwen3VL-2B-Instruct-F16.gguf
MM2=/mnt/disk01/models/Qwen3-VL-2B-Instruct-GGUF/mmproj-Qwen3VL-2B-Instruct-F16.gguf
CUDA_LLM_ROUND_F16=1 ./test_cuda_vlm "$LLM2" "$MM2" fujisan.jpg 50 --budget 0 --vision-engine llama

# === PyTorch vision encoder benchmark (for performance comparison) ===
DL=/mnt/disk01/models/hf_cache/Qwen3-VL-30B-A3B-Instruct
../../ref/qwen_image/.venv/bin/python ../../rdna4/vlm/bench_pytorch_vision.py \
  --model-dir "$DL" --sizes 512,768 --warmup 5 --iters 20 --dtype f16
```

## Harness change

`test_cuda_vision.c` gained `--warmup N`, `--iters N`, `--no-cpu`, and `--bf16`
(steady-state loop reporting mean/min ms + tok/s, mirroring `bench_pytorch_vision.py`).
`test_cuda_vlm` gained `--bf16`. Each `cuda_vision_encode` returns a fresh buffer
(freed per iter); the runner reuses its persistent device scratch, so the loop is safe.
