# qimg INT4 W4A16 (Nunchaku/SVDQuant) — status

Goal: kill the FP8 DiT's block-streaming (20.4 GB) by keeping a ~4-bit DiT VRAM-resident on the 16 GB RX 9070 XT.

## Done & verified
- **Fits 16 GB**: 60 blocks resident, 14.0 GB / 3.1 GB free, no streaming. mod re-quantized to our int4 (cos 0.993 vs FP8), wscale BF16. Convert: `tools/nunchaku_convert_logical.py`.
- **Numeric**: full SVDQuant linear `(W·wscale/smooth)@x + lora_up@(lora_down@x) + bias` cos 1.0 vs host; fused `gemm_int4w_bf16a_wmma_t` cos 0.999997. Gate: `--test-int4-dequant`.
- **Wiring**: all 12 block GEMMs + mod route through `op_int4_linear` when `--int4`; attention stays BF16.

## Limitation (render not yet perf-viable)
- The **rank-128 lora residual uses the scalar f32 `op_gemm`** (bf16→f32 expand + two scalar GEMMs). At 1024²/256² (16k tok) this is intractable — a step does not complete; main fused GEMM is fine, lora is the tar pit.
- **Fix**: route lora through BF16 WMMA (`op_wgemm_bf16`-class); weights are already BF16. Mechanical.

## Max resolution (memory only): 1024² fits; ~1280–1536² edge (lora dly buffer ~n_out·n_tok dominates). Render needs the lora fix first.

## 2026-10-07: tiled WMMA INT4 path (default; `QIMG_INT4_GEMM=legacy` reverts)
- `qimg_gemm_wmma.hip`: a port of `rdna4/llm/gemm_wmma.hip` (main, 2422e4d8), compiled as a second HIPRTC module.
  - Register prefetch and double-buffered kslot-major LDS.
  - `LdInt4G64` loader (nibble × BF16 g64 scale decoded in the tile loader).
  - Grouped rasterization (GROUP_M=8): keeps the X slab in the Infinity Cache.
    Without it, K=12288 re-streamed 108 MB of X per N column (63 → 17.6 ms).
  - Split-K + bf16 reduce for `lora_down`.
  - **`lora_up` fused into the main GEMM** as a second dense K phase: `Y = [X/s | dt]·[Wq | lu]ᵀ + b`.
- `lora_down` is pre-multiplied by `smooth` at load (`qimg_fold_smooth_into_lora`), so both phases share the
  smoothed BF16 activations.
- Gates (`QIMG_TILED_SELFTEST=1`, `QIMG_LORA_SELFTEST=1`): cos(tiled, legacy) = 0.999993 on every DiT shape.

| shape (M×K→N, rank 128) | legacy | tiled |
|---|---|---|
| 4396×3072→3072 | 25.2 ms | 4.9 ms (5.2×) |
| 4396×3072→12288 | 77.9 ms | 12.0 ms (6.5×) |
| 4396×12288→3072 | 109.8 ms | 24.1 ms (4.6×) |
| 300×3072→12288 | 6.2 ms | 0.76 ms (8.1×) |

Estimated linears per 1024² step (60 blocks): about 3.5 s, down from about 18 s. Still 15–30 TFLOPS of the 195 peak.
Next levers: `lora_down` at K=12288 (6 ms), and larger tiles with an async LDS pipeline.
