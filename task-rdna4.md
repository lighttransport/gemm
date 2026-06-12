# Task: Optimize RDNA4 HIP runner to match llama.cpp on Qwen3.6-35B-A3B

Resume doc for the `rdna4/llm` + `rdna4/vlm` HIP runner perf work vs llama.cpp ROCm.
GPU: AMD Radeon RX 9070 XT (gfx1201, 16 GB), ROCm 7.2.2.
Model: `/mnt/disk1/models/qwen36/35b/Qwen3.6-35B-A3B-UD-IQ3_S.gguf` (12.73 GiB, arch
`qwen35moe` — hybrid Delta-Net SSM (30 layers) + gated-attn (10) + 256-expert/8-used
MoE on all 40 layers; n_embd 2048, head_dim 256, vocab 248320).
Detailed report: `rdna4/llm/qwen36_35b_vs_llamacpp.md`.

## Current state (vs llama.cpp)

| metric | HIP @session-start | **HIP now** | llama.cpp | gap now |
| --- | ---: | ---: | ---: | ---: |
| prefill pp512  | 31.2 | **1055 t/s** | 902 | **1.17× FASTER** |
| prefill pp1024 | 30.9 | **1536 t/s** | 883 | **1.74× FASTER** |
| decode  tg128  | 28.7 | **128.5 t/s**  |  83.0 | **1.55× FASTER** |
| vision prefill (672 tok) | 33.9 ms/tok | **1.54 ms/tok** | — | — |

Session 2 (commits 0abc74a..5a42442): fused all-expert decode MoE + SSM aux fusion
(decode 45.5->75.7); self-owned 128x128 WMMA bf16 GEMM default (LLM_GEMM, HIPBLASLT=0
build fully works at 990 t/s); grouped all-expert dequant+GEMM (blockIdx.z=expert) +
GPU top-K/histogram/scatter (prefill 401->1001). Decode = launch-latency-bound.

All changes bit-exact (`--verify-quant-kernels` 18/18 PASS) and output-preserving
(VLM still describes Mt. Fuji). Latest commit: `0f3b6b4` (and `ed8a737` = the mmq win).
Branch `main`. NOTHING PUSHED (per repo rule — ask before push).

## Repro / benchmark commands

```bash
cd /mnt/disk1/work/gemm/main/rdna4/llm && make            # build test_hip_llm
cd /mnt/disk1/work/gemm/main/rdna4/vlm && make            # build test_hip_vlm (links runner)

# Decode + prefill bench (default = all opts on). -s must exceed prefill+decode.
LLM_PREFILL_WARMUP=2 ./test_hip_llm /mnt/disk1/models/qwen36/35b/Qwen3.6-35B-A3B-UD-IQ3_S.gguf \
  -s 1300 --bench --gpu-only-bench --prefill-len 512 --decode 128

# Kernel correctness (bit-exact base IQ/Q6_K matvecs)
./test_hip_llm --verify-quant-kernels

# VLM e2e correctness (must say "Mount Fuji"); also exercises chunked batched prefill
LLM_PREFILL_WARMUP=1 ./test_hip_vlm <model> /mnt/disk1/models/qwen36/35b/mmproj-F16.gguf \
  /mnt/disk1/work/gemm/main/fujisan_1024.png -n 40 --vision-bf16

# Profile (find hotspots). Graph capture is decode-only; export env so rocprof child sees it.
export LLM_MOE_PREFILL=1
rocprofv3 --kernel-trace --output-format csv -d /tmp/rp -- ./test_hip_llm <model> \
  -s 600 --bench --gpu-only-bench --prefill-len 256 --decode 1

# llama.cpp reference
cd /home/syoyo/work/llama.cpp && ./build_rocm722_rdna4_fa/bin/llama-bench \
  -m <model> --device ROCm0 -ngl 99 -fa on -p 512,1024 -n 128 -r 3
```

Env flags (all default to the fast path): `LLM_MOE_PREFILL=1` (batched mmq prefill),
`LLM_MOE_CHUNK=256` (Tensile-safe sub-batch), `LLM_DECODE_DP4A=0`, `LLM_LDS_GRID=0`,
`LLM_GRAPH_DISABLE`, `LLM_BATCH_DISABLE`, `LLM_DEBUG_DISPATCH=1` (per-layer dispatch).

## What was done (commits, newest first)

- `ed8a737` **Fused quantized MoE GEMM (mmq) — THE prefill win, default on.**
  `mmq_iq2s_f32`/`mmq_iq3s_f32`: `Y[cnt,N]=X[cnt,K]×W[N,K]^T`, W stays QUANTIZED; one
  warp/weight-row dequants each group ONCE and reuses across all `cnt` tokens
  (amortizes grid-lookup over the tile; no bf16 materialization). Wired into the
  token-grouped expert path (gather→mmq gate/up/silu/down→scatter-accum). PLUS the
  dispatch fixes that finally route this model through the batched path:
  (a) `layer_is_batched_eligible`/`ssm_layer_is_batched_eligible` skip the dense
  `ffn_*` type checks for MoE layers; (b) `batch_qtype_ok`+`get_bf16_weight` accept
  F32 (this model's SSM gate/alpha/beta/out are F32); (c) prefill CHUNKED to ≤256
  (hipBLASLt/Tensile throws on the large SSM/MoE shapes at M>256).
- `4bb7e79` Full warp/thread-utilization IQ2_S/IQ3_S/Q6_K decode matvecs (decode
  36→45.5; old kernels strided K by n_blocks → 6-25% lanes idle for MoE/LM-head).
- `01aee4d` GPU-side sync-free MoE dispatch (`moe_topk_softmax_gpu`,
  `sigmoid_scalar_f32`, `scale_add_dev_f32`, expert-indexed matvecs) + graph capture
  (decode 29→36).
- Negative (gated off, documented): `8472584` DP4A IQ matvec (decode is grid-lookup
  bound, not multiply); `ef3bd38` LDS-cached grids (grids hot in L2 already);
  `ee12d25` per-row batched MoE prefill; `bc05e2e` dequant+hipBLASLt batched experts
  (256 experts/8-used → too few tokens/expert; bf16 materializes full weight).

## Gap analysis / next steps (highest ROI first)

1. **Prefill 2.2× gap.** Profile the *batched* path (export LLM_MOE_PREFILL=1) to see
   the new bottleneck. Candidates: per-layer host top-K sync (1 hipStreamSynchronize/
   layer); router done as 1 GEMM but shared-gate is small; the ≤256 chunk cap limits
   GEMM efficiency — try raising `LLM_MOE_CHUNK` if a Tensile fix/larger-safe-M is
   found (root-cause the "Could not initialize Tensile host: unordered_map::at" — may
   be a specific shape e.g. N=1 shared-gate or N=8192 SSM qkv; pre-warming those exact
   (M,N,K) plans may avoid it and allow M=1024 chunks). mmq tile is BT=cnt≤32 with one
   warp/row — a real WMMA int8 tile (q8_1 act + dp4a, mma 16×16) would push closer to
   llama.cpp's mmq.
2. **Decode 1.82× gap.** IQ2_S/IQ3_S/Q6_K matvecs are at the practical floor for the
   per-token scalar-dequant design (DP4A and LDS both net-negative — see negatives).
   The only decode lever left is amortization via batching: CFG batches 2 sequences →
   a decode-time mmq-batch (M=2) would amortize the dequant 2×. Or accept the floor.
   deltanet_step_f32 is ~9% of decode (SSM recurrence) — a fusion could trim it.
3. **mmq accuracy:** mmq is float-accumulate with exact IQ dequant (more accurate than
   the bf16+hipBLASLt path it replaced). If a quality bar matters, compare velocity
   cos vs the per-token path (LLM_MOE_PREFILL=0).

## Key files & GOTCHAs

- `rdna4/llm/hip_llm_runner.c` — everything (HIPRTC kernel source string at top;
  decode `forward_one_layer`/`forward_moe_ffn`; batched `forward_block_batched_dense`
  + `forward_moe_ffn_batched`; eligibility `*_is_batched_eligible`; `get_bf16_weight`;
  batch entries `hip_llm_forward_batch_logits`/`_embd`).
- `rdna4/llm/mm_blaslt_bridge.cpp` — `mm_blaslt_run_bf16(y[M,N], w[N,K], x[M,K],…)`.
- GOTCHAs: `-s` default 256 rejects prefill>256 (raise it). Use IQ3_S not IQ2_M
  (IQ2_M hits CPU→F16 fallback, VRAM blowup). hipBLASLt/Tensile fails at M>256 here →
  chunking. iq2s/iq3s grids must stay `__device__ const` (NOT `__constant__` —
  divergent lookups serialize). HIP kernel arg array MUST exactly match the kernel
  signature (silent err 700 otherwise). rocprofv3 needs env `export`ed to reach child.
  DP4A helps prefill-GEMM (weight reused) but NOT decode-matvec (M=1, no reuse).
