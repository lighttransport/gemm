# qimg HIP vs CUDA fp8 — layer-by-layer parity findings

Verifying the RDNA4 HIP runner against the CUDA fp8 reference dumps
(`/mnt/disk1/models/qwen-image/dumps/`, prompt "a red apple on a white table",
256², 20 steps, cfg 2.5, shift 3.1). HIP produces a slightly wobbly image; CUDA
does not. Tooling: `QIMG_VAE_DUMP_PREFIX` + `--latent-bin` + `tools/compare_cuda_dumps.py`.

## VAE — exonerated
Fed `cuda_latent.bin` into the HIP VAE; all 19 stages match CUDA at cosine
0.992–0.9999, PSNR 34–63 dB, no NaN. Not the cause.

## DiT — pinned exact (init + cuda_txt_pos/neg + cuda_sigmas)
Per-step cosine 0.99999 → 0.9956 (final 29.7 dB), monotonic compounding drift,
no single bad block.

Bisection (final-latent cosine vs CUDA):
| Config | final cos | PSNR |
|---|---|---|
| Default BF16×FP8 GEMM + WMMA attn | 0.9956 | 29.7 |
| F32 GEMM + WMMA attn | 0.9956 | 29.6 |
| F32 GEMM + F32 scalar attn | 0.9958 | 29.8 |
| broad FP8×FP8, clamp scale | — | 24.7 |
| broad FP8×FP8, perrow max/448 | 0.20 (collapse) | 0.7 |

## Conclusion
Removing every HIP approximation (fp8→f32 GEMM, WMMA→f32 attn) barely moves the
result. The drift is NOT attention or MLP/GEMM precision. The only structural
difference left: **CUDA quantizes activations to FP8 per-tensor (max/448) for all
projections; HIP keeps activations bf16/f32**. HIP is the *more* precise impl, so
forcing HIP toward CUDA's fp8 acts makes it worse/collapses (HIP fp8×fp8 kernel is
lossier than cuBLASLt). Byte-parity with CUDA isn't reachable without copying that
loss. Wobble vs ground truth is fp8 *weight* precision compounding — lever is the
weights (int4/svdquant), not GEMM/attn internals.

## Repro: emulate CUDA fp8 acts (QIMG_ACT_FP8_RT=1)
Per-row fp8/448 act roundtrip + accurate GEMM = final cos 0.86 (vs 0.9956 default), WORSE. CUDA fp8 acts are extra loss HIP avoids; 0.9956 is the parity floor. Gated diagnostic, default off.
