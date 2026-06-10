# Qwen3.6-35B-A3B: RDNA4 HIP runner vs llama.cpp (ROCm)

Date: 2026-06-10. GPU: AMD Radeon RX 9070 XT (gfx1201, 16 GB), ROCm 7.2.2.
Model: `Qwen3.6-35B-A3B-UD-IQ3_S.gguf` (12.73 GiB, arch `qwen35moe` — hybrid
Delta-Net SSM + MoE, 40 blocks, 256 experts/8 used, n_embd 2048, head_dim 256).
mmproj: `mmproj-F16.gguf` (Qwen3-VL vision tower).

- HIP runner: `rdna4/llm/test_hip_llm`, `rdna4/vlm/test_hip_vlm` (IQ3_S tensor types
  F32/Q6_K/IQ2_S/IQ3_S/IQ4_XS all natively uploaded; no F16 fallback).
- llama.cpp: `build_rocm722_rdna4_fa/bin/llama-bench` (FA), `build_hip/bin/llama-mtmd-cli`,
  build `549b9d843 (9307)`.
- Image: `fujisan.jpg` (640×427) upscaled → `fujisan_1024.png` (1024×683) → 672 vision tokens.

## Text prefill/decode (no image)

| test   | HIP test_hip_llm | llama.cpp llama-bench | llama.cpp / HIP |
| ------ | ---------------: | --------------------: | --------------: |
| pp512  |        31.2 tok/s |         897.5 tok/s   |        **28.8×** |
| pp1024 |        30.9 tok/s |         887.9 tok/s   |        **28.7×** |
| tg128  |        28.7 tok/s |          83.0 tok/s   |         **2.9×** |

HIP cmd: `LLM_PREFILL_WARMUP=2 ./test_hip_llm <model> -s 1300 --bench --gpu-only-bench --prefill-len {512,1024} --decode 128`
llama: `llama-bench -m <model> --device ROCm0 -ngl 99 -fa on -p 512,1024 -n 128 -r 3`

### Decode optimization progress (2026-06-10)

Decode (greedy, after a 64-token prefill, graph capture on):

| build | decode tok/s | vs llama.cpp 83 |
| ----- | -----------: | --------------: |
| session start (commit db4a2e9)            | 29.1 | 2.85× slower |
| + GPU-side sync-free MoE dispatch + graph (01aee4d) | 36.4 | 2.28× |
| + full warp/thread util IQ2_S/IQ3_S/Q6_K matvec (4bb7e79) | **45.5** | **1.82×** |

Net **+56%** so far; verified bit-exact (`--verify-quant-kernels` 18/18 PASS) and
output unchanged (VLM still identifies Mt. Fuji).

**Diagnostics (rocprofv3 kernel trace, decode, graph off):**
- HIP-graph capture only adds ~4% → decode is **GPU-compute-bound**, not launch-bound.
  The ~20% win from MoE dispatch came from removing 80 host syncs/token, not launches.
- The dequant matvecs dominate. Original kernels strided the K dim by `n_blocks`
  (nb), but MoE/attn/LM-head shapes have nb=2–8, leaving 6–25% of lanes/threads idle.
  Restructuring to one lane/thread per element-group restored full occupancy:
  Q6_K (LM head + shared + attn, 43% of decode) 424→213 ms in the trace.
- **Remaining hotspots**: `matvec_iq3_s_expert` 25%, `matvec_iq2_s_expert` 15% are
  **compute-bound on grid-lookup dequant + per-element sign handling** (running at
  ~3% of mem BW). `deltanet_step_f32` is 9%.

**DP4A port (commit 8472584) — tried, NEGATIVE, gated off.** Ported llama.cpp's
int8-dot (`quantize_q8_32` per-32-block q8 activation + `__builtin_amdgcn_sudot4`,
half-split for full lane occupancy). Validated correct (rel_l2 vs float ref: IQ2_S
1.6e-3, IQ3_S 4.7e-3 = pure q8-activation error). But **44.6 vs 45.4 tok/s — ~2%
slower**: these IQ matvecs are grid-lookup-dequant bound, not multiply bound, so
DP4A (accelerates only the MAC) + q8-quantize + scalar sign-build is a net loss.
Default `LLM_DECODE_DP4A=0`; F32 full-util kernels stay the decode path. To actually
speed IQ2_S/IQ3_S one must cut the grid-lookup cost (LDS-cached grids — untried) or
do SIMD sign (`__vsub4`/`__vcmpne4` are absent under HIPRTC).

## VLM (image) — fujisan_1024.png, 672 vision tokens, prompt "Describe this image in detail."

| stage                         | HIP test_hip_vlm | llama-mtmd-cli | llama.cpp / HIP |
| ----------------------------- | ---------------: | -------------: | --------------: |
| vision encode (672 tok)       |       1294 ms    |     1462 ms    |   **0.88×** (HIP faster) |
| LLM prefill (~688 tok)        |      ~29.5 tok/s |    363.9 tok/s |        **12.3×** |
| decode                        |      ~28.7 tok/s |     77.5 tok/s |         **2.7×** |

HIP cmd: `LLM_PREFILL_WARMUP=2 ./test_hip_vlm <model> <mmproj> fujisan_1024.png -n 64 --vision-bf16 --resize dynamic`
llama: `llama-mtmd-cli -m <model> --mmproj <mmproj> --device ROCm0 -ngl 99 -c 2048 --temp 0 --image-min-tokens 672 --image-max-tokens 672 --image fujisan_1024.png -p "Describe this image in detail." -n 64`

(HIP VLM decode not printed by the driver; equals the text-path per-token decode, ~28.7 tok/s.)

## Correctness — output agreement (temp=0), PASS

Both runners correctly identify the subject as **Mount Fuji, Japan**:
- HIP:   "…large, snow-capped mountain… classic conical shape. This is unmistakably Mount Fuji in Japan."
- llama: "…large, snow-capped mountain. It's clearly Mount Fuji due to its iconic symmetrical cone shape."

## Analysis

- **Prefill is the gap (~29× text, ~12× VLM).** The HIP runner logs
  `Phase-2 batched path disabled (hybrid/moe/quant)` and processes prefill
  **token-by-token** (33–34 ms/tok ≈ its own decode speed) for this hybrid-SSM+MoE+quant
  model. llama.cpp uses a true batched prefill GEMM. This is the known TODO
  ("batched quantized prefill GEMM" for the hybrid path), not a regression.
- **Decode gap ~2.9×** is per-token kernel efficiency (matvec/MoE dispatch/Delta-Net).
- **Vision encoder is competitive** — HIP's WMMA vision tower is ~12% faster than
  llama.cpp's clip encode on the same 672-token image.
- VRAM: 12.73 GiB model fits 16 GB on both. HIP `test_hip_vlm` frees vision weights
  before loading the LLM; llama.cpp keeps mmproj (0.86 GiB) co-resident, ran fine at -c 2048.

## Biggest lever

Batched prefill for the qwen35moe hybrid path: today prefill ≈ decode speed because
it is per-token. Closing this would move HIP prefill from ~30 tok/s toward the
hundreds, the single largest win vs llama.cpp.
