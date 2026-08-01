# Gemma4-31B IQ2_XXS CUDA — decode/prefill optimization (resume guide)

## Goal
Push the gemma4-31B UD-IQ2_XXS model on RTX 5060 Ti (sm_120, 16 GB) toward
**decode 30 tok/s** and **prefill pp512 960 tok/s** (llama.cpp on the same model+GPU
does decode 32.6 / pp512 961, so both are achievable).

## Model
`/mnt/disk01/models/gemma4/31b/gemma-4-31B-it-UD-IQ2_XXS.gguf`
- n_embd=5376, 60 layers, n_heads=32, n_kv_heads varies [16…,4 every 6th], head_dim_full=512,
  head_dim_swa=256, swa_window=1024, n_ff=21504, softcap=30, weight-tied output (token_embd=Q3_K).
- UD-mixed quant per tensor (verified via `python3 cpu/qwen_image/inspect_gguf.py <gguf>`):
  - attn_q/k/o, ffn_gate/up = **IQ2_XXS**; attn_v = **IQ3_XXS** (every layer);
    ffn_down = **Q2_K** (early ~7 layers) / **IQ2_XXS** (rest); a few layers carry **IQ3_S/IQ2_S**.

## Current state (committed this session, tree clean)
| metric | session start | now |
|---|---|---|
| prefill pp512 | 116 | **418 t/s** |
| prefill pp1024 | — | 467 |
| decode tg | 18.9 | **22.3 t/s** |

Commits (newest first): `0bf3f30` dense IQ2_XXS MMQ int8 TC prefill (265→418) ·
`197b4d5` rmsnorm 256→1024 thr (decode 22.0→22.3) · `5ed93b4` coalesce
iq3_xxs/iq2_s/iq3_s decode matvecs (18.9→22.0) · `4bd5e9f` IQ2_S/F16 prefill→cuBLAS (116→265).

## Build / run / measure (cwd = cuda/llm)
```bash
cd /home/syoyo/work/gemm/main/cuda/llm && make            # NVRTC source is in cuda_llm_runner.c
# prefill pp512:
./test_cuda_llm <gguf> --large-bench 512 -s 1024 --large-bench-random --large-bench-seed 7
# decode + correctness oracle (token-0 hidden must stay norm=2238.3601, byte-identical):
./test_cuda_llm <gguf> -t "Hi" -n 1            # prints "Decode: .. tok/s" and "GPU [..] norm="
# real-prompt prefill correctness (batched MMQ must match cuBLAS top-5):
./test_cuda_llm <gguf> -t "The capital of France is" -n 1   # "Sequential top-5" line
```
GOTCHAS: cwd resets to repo root between tool calls — `cd cuda/llm` each Bash call.
NVRTC PTX cache `/tmp/cuda_llm_sm_120_v25_*.ptx` keys on source hash (auto-recompiles on edit;
`rm` only if debugging stale parity). Profile decode with a sqlite time-window query on the LAST
~2.75s (decode bench = 5 warm + 50 timed); `nsys ... -t cuda` then query
`CUPTI_ACTIVITY_KIND_KERNEL` where `start > max(end)-2.75e9`. `--cuda-graph-trace=node` needed for
decode (graph hides kernels). ncu blocked (ERR_NVGPUCTRPERM). The harness `rel_L2_vs_seq` is
UNRELIABLE (2-bit × 60 layers → run-to-run flips); trust token-0 byte-equality + top-5 match instead.
Sanitizer baseline = **240 benign CUDA_ERROR_INVALID_VALUE launch warnings** in --large-bench
(present identically in baseline b9aacf9); only NEW `invalid read/write`/`misaligned` lines matter.

## Where the time goes now
**Prefill pp512 (1.22s):** `mmq_iq2xxs_grouped8` = 70% (re-reads each weight ~4×, n_work=ceil(512/128));
dequant→cuBLAS for the minority types ~260ms (iq3_xxs v_proj 120ms + q2_K 100 + iq3_s 81 + iq2_s 80);
mmq_quant_q8_1 (activation) 4%. **pp2048 = 131 t/s is ATTENTION-bound** (O(N²) full-attn d512), a
separate axis — not the matvec path.
**Decode (44.8ms/tok, fully GPU-bound, NO host gap):** iq2_xxs_coal 30.6ms (58%, ~248 GB/s — already
matches llama.cpp's per-weight rate), lm_head F16 6.7ms, rest ~7ms.

## Next levers (highest leverage first)
1. **MMQ for the minority prefill types** (tractable, ~150ms → ~480 t/s). Adapt `mmq_iq2xxs_grouped8`
   for IQ3_XXS first (v_proj, every layer, biggest at 120ms). IQ3_XXS = 98B blocks, uint32
   `iq3xxs_grid_dev`, scales_and_signs[32]@66 — decode already worked out in the coalesced decode
   kernel `matvec_iq3_xxs_q8_1_dp4a`. Then Q2_K/IQ3_S/IQ2_S (word-based, harder). Wire into
   `launch_batch_matvec` like the IQ2_XXS case (`launch_mmq_iq2xxs_dense` is the template, ~line 11793).
2. **Read-once MMQ GEMM for IQ2_XXS** (bigger lever toward 960, harder). The grouped kernel
   re-reads the weight n_work=ceil(n_tok/(8·TG))× and is register-capped at TG=16 (TG=32 spills,
   f[32][4]). A true GEMM tiling that K-splits and keeps the decoded weight tile resident across all
   token columns would read the weight ~once. This is the llama.cpp mmq.cuh approach (MMQ_X/MMQ_Y
   tiling). Substantial.
3. **Prefill attention** for pp≥2048 (O(N²) full-attn d512 dominates there) — separate from matvec.
4. **Decode 30** needs a full MMVQ-class rewrite (uniformly-efficient skinny matvecs); the dominant
   iq2_xxs_coal already matches llama.cpp's rate so there's no single hot spot. Tractable-but-modest:
   coalesce q2_K decode matvec (~0.8ms); a Q3_K lm_head needs both a coalesced matvec_q3_K AND an
   embed_q3_K kernel + 4-site rewire (matvec_q3_K is strided; a strided Q3_K lm_head reading 1GB@66GB/s
   would be SLOWER than the 2.8GB F16@424GB/s) — net ~+1 tok/s, deprioritized.

## Key code anchors (cuda_llm_runner.c)
- `launch_mmq_iq2xxs_dense` (~11793) — dense MMQ helper; quant act + 1-expert worklist + launch.
  `use8`/TG=16 path via `mmq_iq2xxs_grouped8`; env `CUDA_LLM_NO_MMQ_DENSE`, `CUDA_LLM_MMQ_TG4`.
- `mmq_iq2xxs_grouped8` (NVRTC, ~4935) — TG=16 prefill MMQ (copy of MoE `mmq_iq2xxs_grouped` ~4858).
- `mmq_quant_q8_1` (~4836) — activation→q8_1 for MMQ; struct scratch `d_mmqd_*` (~6430).
- `launch_batch_matvec` (~11860) — per-type prefill dispatch; IQ2_XXS case prefers MMQ then
  `launch_dequant_gemm_f16` (~11730, dequant→F16→cuBLAS) fallback. IQ2_S/F16/IQ3_S in the f16 helper.
- Coalesced decode matvecs: `matvec_iq2_xxs_q8_1_dp4a_coal` (~1856),
  `matvec_iq3_xxs/iq2_s/iq3_s_q8_1_dp4a` (coalesced this session). Strided word-based: q2_K (~903),
  q3_K (~1018). `launch_rmsnorm` (~9235, now 1024 thr).
- Dead-ends (do NOT retry): on-the-fly block-major repack bm=1 (slower at TG≥8), 2-rows-per-warp
  decode coal2 (regressed), prefill dequant/GEMM overlap (shares bandwidth), #pragma unroll on coal,
  Q3_K lm_head (strided matvec_q3_K slower than F16; needs coalesce + embed_q3_K rewire first).

## Full optimization ladder (for context)
- **Decode 3.1→22.3 over the project:** (a) dp4a INT8 matvecs for all 4 IQ-codebook types
  (`matvec_{iq2_xxs,iq3_xxs,iq2_s,iq3_s}_q8_1_dp4a`): quantize act→Q8_1 + INT8 `dp4a`, codebook signs
  applied branchlessly `signed = __vadd4(grid^mask, mask&0x01010101u)` where `mask` expands 4 sign
  bits to 4 bytes (zero-centered → no Q8_1 offset term) → 3.1→15.9. (b) Coalesced IQ2_XXS matvec
  (`_coal`): lane→qs-uint16 (NOT lane→block) so 32 lanes read 32 consecutive shorts = one coalesced
  64B load; decode g=lane/4=ib32, sub=lane%4, aux gathered via intra-4-lane-group `__shfl` →
  219→295 GB/s, 15.9→19.2. (c) THIS session: same coalescing applied to iq3_xxs/iq2_s/iq3_s
  (18.9→22.0, byte-identical) + rmsnorm 1024 threads (→22.3).
- **Prefill 116→418 over this session:** (a) IQ2_S/F16 weights were on a naive per-token
  `vision_linear_f16` (53% of GPU) → routed to cuBLAS like IQ2_XXS (116→265). (b) Dense IQ2_XXS MMQ
  int8 tensor-core (reuse the MoE `mmq_iq2xxs_grouped`, mma.sync.m16n8k32.s8.s8.s32, weights read as
  2-bit) → 265→312; TG=16 prefill variant (128 tok/block, fewer weight re-reads) → 312→418.

## IQ2_XXS block layout (66 bytes / 256 elements) — for new MMQ/decode kernels
```
offset 0:  d (half) super-block scale ; offset 2: qs[64] = 32 uint16
per 256-block: 8 sub-blocks (ib32). For ib32: a0 = qs[4*ib]|(qs[4*ib+1]<<16),
  a1 = qs[4*ib+2]|(qs[4*ib+3]<<16); db = d*(0.5 + (a1>>28))*0.25;
  for l in 0..3: gv = iq2xxs_grid_dev[(a0>>(8*l))&255] (uint64, 8 int8 mags);
    signs = ksigns_iq2xs_dev[(a1>>(7*l))&127]; 8 weights = ±(gv byte j).
```
IQ3_XXS (98B): d + qs[64]@2 (uint8 idx into uint32 `iq3xxs_grid_dev`) + scales_and_signs[32]@66
  (uint32/ib32: db=d*(0.5+(sas>>28))*0.5, signs via ksigns). IQ2_S (82B): d + qs[32]@2 + signs[32]@34
  + qh[8]@66 + scales[8]@74 (10-bit grid idx via qh, two scales/ib32). IQ3_S (110B): d + qs[64]@2 +
  qh[8]@66 + signs[32]@74 + scales[4]@106 (9-bit idx via qh). Q2_K/Q3_K are word-based (load_u8x4).

## Benchmark calibration (llama.cpp, same model+GPU)
`~/work/llama.cpp/build/bin/llama-bench -m <gguf> -ngl 99 -p 512 -n 64` → pp512=961, tg=32.6.
Both targets are real; the gap is the read-once MMQ GEMM (prefill) and a full MMVQ rewrite (decode).
