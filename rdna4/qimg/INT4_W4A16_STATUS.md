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
