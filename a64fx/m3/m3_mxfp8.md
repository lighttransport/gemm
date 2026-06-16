# M3-MXFP8 execution plan (Fugaku A64FX)

## Context / why
`~/models/m3-fp8` is MiniMax-M3 quantized to **MXFP8**: **451 GB** (≈ half the 869 GB bf16),
31 shards. This roughly **halves per-rank memory and staging time** at near-bf16 quality —
the lever that makes **1M context** (currently blocked by the ~120 GB/rank KV) and/or **fewer
EP nodes** tractable, and helps the multi-stream (M=N) weight read. The bf16 port already
works end-to-end (generates correct text @96n); MXFP8 is a precision/memory variant of the
same graph, so it reuses everything except the weight load + matvec.

## Verified format (config.json + safetensors headers, 2026-06-16)
- `quantization_config`: `quant_method=mxfp8`, `activation_scheme=dynamic`, `weight_block_size=[1,32]`.
- Each quantized weight `X.weight` = **F8_E4M3** `[rows,cols]` (1 byte/elem) + companion
  `X.weight_scale_inv` = **U8 (E8M0)** `[rows, cols/32]` — one power-of-2 block scale per
  32 contiguous columns per row. Dequant: `w[r,c] = fp8_e4m3(W[r,c]) * 2^(e8m0(scale[r,c/32]) - bias)`.
  (This is the OCP MX format; the **same E8M0 per-32 scaling the DS4P MXFP4 kernel uses**,
  with FP8 values instead of FP4 nibbles.)
- Quantized: `q/k/v/o_proj`, `index_q/k_proj`, experts `w1/w2/w3`, shared `gate/up/down_proj`
  (22422 weights). NOT quantized (bf16/f32, 676 weights): all norms, `q/k/index norms`,
  `block_sparse_moe.gate` (router, F32), `e_score_correction_bias`, `lm_head`, `embed_tokens`,
  `model.norm`. (Some early-layer MoE gates are also in `ignored_layers`.)
- Per-rank memory @96n: owned experts ≈ 2 × 57L × ~57 MB ≈ **6.5 GB** (vs 12.9 GB bf16) +
  E8M0 scales (~1/32 extra). Arena ≈ **7–8 GB/rank** (vs 13.2). ⇒ ~half.

## Reuse (almost everything)
The whole stack carries over: EP/TP sharding, uTofu all-reduce, pjsub, the GQA+MSA+sigmoid-MoE
forward, multi-stream batched decode, the tokenizer/gen path. Only 4 spots change.

## Implementation (4 changes)
1. **Type (`common/m3.h`)** — add `M3_MXFP8` (the enum already reserves `M3_FP8`/`M3_MXFP4`
   slots). `m3_wbytes(MXFP8,r,c)=r*c` (1 B/elem); `m3_sbytes(MXFP8,r,c)=r*(c/32)` (E8M0).
   `m3_tensor.scale` (uint8*) holds the E8M0 block scales.
2. **Matvec kernel (`common/ggml_dequant.h` or `m3_impl.h`)** — `matvec_mxfp8_8row`: per output
   row, per 32-col block, dequant 32 FP8 (existing E4M3 LUT/`SVE` widen) × the block's E8M0
   scale (existing `e8m0` helper), FMA with x. Adapt the existing `matvec_mxfp4_8row` (same
   E8M0 loop, read 1 byte/value not a nibble) + `matvec_fp8e4m3` widen. Also a batched
   `m3_gemm_mxfp8` for multi-stream (M=N), mirroring `m3_gemm_bf16`.
3. **Stager (`common/m3_stage.c`)** — for each kept quantized weight, ALSO copy its
   `*.weight_scale_inv` tensor (manifest records both, dtype tags F8_E4M3 / U8). `classify()`
   unchanged (expert `e%N`; skip vision/projector); `M3_NSHARDS=31`. bf16 ignored-layers stay
   as-is. Per-rank blob ≈ half.
4. **Loader + forward dispatch (`common/m3_impl.h`)** — `m3_load_real`: a weight with a
   `_scale_inv` companion in the manifest → `M3_MXFP8` (load FP8 bytes + E8M0 scale), else
   bf16. TP slice: FP8 weight slices by row/col (1 B/elem, same offsets); the E8M0 scale
   row-shards fully and col-shards by `/32`. Forward: change the matvec call sites to take the
   `m3_tensor` and dispatch on `.type` (bf16 → `m3_mv_bf16`/`m3_gemm_bf16`; MXFP8 → the new
   kernels). This is the only invasive edit — the forward currently casts `.w` to `uint16_t*`;
   wrap as `m3_matvec(m, y, &L->wq, x, rows, cols)`.

## Validation (mirrors the bf16 bring-up)
1. **fccpx cross-compile** of the kernel + loader.
2. **Single-node real-weight smoke** (no MPI): stage layers 0–3 of `~/models/m3-fp8` (shards
   1–3) ep_size=1 → `/local/m3fp8`, `m3_real_test` load + forward → NaNs=0, load OK. Confirms
   the FP8+E8M0 dequant + mixed-precision loader on real weights.
3. **Numeric spot-check**: dequant one MXFP8 tile in C vs a Python reference (fp8×2^e8m0) to
   pin the E8M0 bias/sign convention before scaling up.
4. **96-node stage + generate** (`pjsub_m3_gen_96n.sh` with `M3_MODEL_DIR=~/models/m3-fp8`,
   `M3_NSHARDS=31`, `M3_STAGE_DIR=/local/m3fp8`): coherent text (the quality gate — should
   match the bf16 "Paris"/summary/code outputs closely) + measure decode/prefill tok/s and
   per-rank arena (target ~7–8 GB → headroom for more KV/context or fewer nodes).
5. **Memory win demo**: re-run at **fewer nodes** (e.g. 48–64) now that experts fit, and/or
   higher `M3_MAXPOS`, to show the context/node-count headroom MXFP8 buys.

## Effort / risk
- New kernel (`matvec_mxfp8` + batched) = the main work; close to existing MXFP4/FP8 kernels.
- Loader mixed-precision + forward dispatch refactor = moderate, mechanical.
- Stager scale-copy = small.
- Risk: E8M0 bias/sign convention (validate with the numeric spot-check #3 first).
- Decode tok/s: half the weight bytes, but decode is dispatch+comm bound (per the ceiling),
  so single-stream speedup is modest; the real wins are MEMORY (1M/fewer nodes) and quality.
  Multi-stream (M=N) benefits more (weight read is the amortized term there).
