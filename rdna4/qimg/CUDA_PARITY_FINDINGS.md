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

## int8 vs bf16 end-to-end (vs CUDA-fp8 dumps, F32-streaming roundtrip)
| weights | cos vs CUDA | PSNR |
|---|---|---|
| fp8 e4m3 (current) | 0.9956 | 29.7 |
| bf16 lossless | 0.9818 | 23.8 |
| int8 g64 | 0.9811 | 23.7 |

int8 g64 == bf16 (lossless). Both FARTHER from CUDA than fp8 because CUDA itself
is fp8 — the dumps are not a ground-truth yardstick for wobble. int8 g64 is the
right quant to reduce *visible* wobble (bf16-equivalent at ~half VRAM), but only
pays off with a real int8 streaming GEMM; the F32-stream validation here is
207s/step (unusable for perf, quality-only). QIMG_FORCE_F32W enables bf16/f32
ckpt via f32 streaming; QIMG_W_INT8_RT adds int8-g64 weight roundtrip.

## int4 W4A16 status — BROKEN (single OOB root cause)
Running the logical-v3 SVDQuant int4 ckpt produces flat-gray images for every
prompt. Diagnosed via the harness:
- DiT latent vs CUDA: step0 cos 0.9999 (first forward ~correct) then explodes —
  step10 cos -0.97, final max 8.1e7.
- HIP error probe: **err=700 (illegal address) is set during denoise, on the
  FIRST forward, at block 0** — confirmed even with --steps 1 --cfg 1.
- LoRA disabled (QIMG_INT4_NOLORA, temp): still faults (hangs at post-denoise
  sync) → the OOB is in the fused `gemm_int4w_bf16a_wmma_t` kernel, not the LoRA
  residual path.

So one bug, two symptoms: the fused int4 GEMM commits an out-of-bounds access →
(1) corrupts the context so the latent blows up over steps, and (2) sets sticky
err 700 so every later hipMalloc "fails" — which is the gray VAE output (NOT a
real OOM). Fix needed: bounds/indexing in gemm_int4w_bf16a_wmma_t (block 0
shapes: to_q 3072x3072, add_* x3584, mlp x12288 — all /128, so likely a
wscale[n_out,n_in/64] or LDS tile edge index). VAE-free-before-decode is wired
(hip_qimg_unload_dit in vae_decode) as a necessary prerequisite for once the
kernel is fixed (int4 is 14GB resident, no streaming).
