---
name: project_mmq_moe
description: CUDA MMQ mul_mat_id kernel for Qwen3.5-35B-A3B IQ2_XXS MoE prefill — optimization state + dead-ends
metadata:
  type: project
---

MMQ (mul_mat_q) port replaces the per-expert dequant+cuBLAS MoE loop with a kernel
that consumes IQ2_XXS weights directly + q8_1 int8 activations via `mma.sync
m16n8k32.s8.s8.s32` tensor cores (sm_120, RTX 5060 Ti). Lives in
`cuda/llm/cuda_llm_runner.c` (NVRTC string kernel `mmq_iq2xxs_grouped` + dispatch
~line 11349, gated `CUDA_LLM_MOE_MMQ=1`). Dev harness: `cuda/llm/mmq/mmq_iq2xxs_moe_test.cu`
(grouped_v1..v7, all bit-identical, rel_L2 vs oracle 0.000000). Target = task.md's 104ms FFN / 2526 tok/s.

## Optimization ladder (Qwen3.5-35B IQ2_XXS, 512-tok prefill, RTX 5060 Ti) — now ~2320 tok/s
- decode-amortized (TG=4, reuse 1 weight decode across 32 tokens): 1273→1578 tok/s (beats cuBLAS)
- 32-lane decode (2 lanes/row, was lane<16 half-warp): 1578→1744, FFN 224→192ms
- flattened work-list grid: 1744→1811. Routing is HEAVY-skewed (256 exp/512 tok top-8:
  mean 16 but maxtok=163, 36 experts=0 tok, 104 experts≤8 tok). Old grid
  (N/64, ceil(maxtok/32), n_experts) launched 6 grid.y blocks for ALL experts (~80%
  early-return) + hot-expert tail. Host builds worklist[] = one packed (e<<16)|group per
  real (expert,group); grid=(N/64,n_work,1). `CUDA_LLM_MOE_DUMP_DIST=1` dumps the histogram.
- shared-mem codebook staging: 1811→1840. Stage iq2xxs_grid (2KB)+ksigns (128B) to shared.
  Modest in runner (global table already L2-hot) but 1.4× in harness (its __constant__ table
  serializes divergent gather — constant-vs-global lesson).
- **branchless vectorized sign decode (v8): 1840→2010, FFN 179→153ms.** Replace per-byte
  `(int)g[j]*((s&(1<<j))?-1:1)` over j<8 with __vsub4 two's-complement negate: (g^mask)-mask
  where mask = 8-byte 0x00/0xFF sign expansion (built in-kernel from ksigns). ~4× fewer decode
  inst + 2 word stores vs 8 byte stores. Harness v8 = 1.22× vs v6.
- **direct aligned int loads (v10): 2010→2272, FFN 153→126ms.** sW stored via *(uint32_t*),
  so the 6 mmq_pack4/MMA (per-byte reassembly) are replaced by *(const int*)&sW[r][k*4].
  Harness v10 = 1.23-1.25× vs v8. BIGGEST of the v8-v11 wins.
- **vectorized activation staging (v11): 2272→2306, FFN 126→122ms.** sX copy 4 bytes/thread
  (int) instead of 1 byte. Harness v11 = 1.06-1.09× vs v10.
- All v8-v11 bit-exact (rel_L2 vs v1 oracle = 0.000000). Parity rel_L2_vs_seq ~0.14-0.15 PASS.

## Bottleneck diagnosis (current)
grouped ~40% of GPU, per-instance 0.74ms, ~22× off int8-TC peak. After v8-v11 the decode
and load wastes are gone; the residual is INHERENT to MoE: ~16 tok/expert means the
per-sub-block FP rescale `f += wr_scale*x_scale*(float)c` (both scales vary per 32-block,
can't defer) dominates the actual MMA (~16:1 instruction ratio) → low TC utilization.
A dense q8 GEMM amortizes rescale over a large token-tile; MoE can't. Matching llama.cpp's
2620 likely needs their exact tiled/pipelined mul_mat_id kernel (multi-day port).
ncu blocked by ERR_NVGPUCTRPERM.

## DEAD-ENDS — do not re-attempt
- **Full-tile coalesced weight staging (harness grouped_v7)**: BIT-IDENTICAL but MUCH SLOWER
  (runner 1840→941 tok/s). 33KB dynamic shared crushes occupancy AND saves no traffic;
  runner's strided reads are L2-absorbed, NOT DRAM-bound.
- **gather_quant activation dedup (runner, reverted)**: quantize the 512 UNIQUE token
  activations once (not 8× per (token,expert)) and have gate/up read via idx_map=ids_token.
  Bit-exact, removed the 8× redundant gather_quant (~15ms) BUT SLOWER (2306→2145, FFN
  122→138ms): grouped then reads activations SCATTERED by token instead of sequential-
  compacted → worse hot-loop coalescing, costs more than the gather saved. Same lesson as v7.
- **double-buffer software pipeline (harness grouped_v9)**: ~1.02× no-op. Inter-block
  occupancy already hides per-sub-block latency; intra-block pipelining is redundant.
- **v4 32-lane → bigger TG**: diminishing (TG already amortizes past avg expert token count).
- **llama.cpp tile-structure port (8 warps/256 thr, MMQ_Y=128 rows/block, grid.x=N/128)**:
  bit-exact, FLAT in runner A/B (2327 vs 2327, env CUDA_LLM_MMQ_TILED). llama.cpp's MMQ
  scale-application is per-K-block = IDENTICAL to ours, so its only structural edge is tile
  shape/occupancy — and that's a no-op here, proving the kernel is NOT occupancy/compute-bound.

## THE WIN: block-major weight repack (DONE, default-on, ~2510 tok/s)
The wall was IQ2 weight DRAM traffic ~2× the floor: a warp reads 16 rows' 66-byte blocks at
row_bytes(528B) stride → cache-line over-fetch. FIX (llama.cpp's actual technique, the one
memory change that WORKED): repack IQ2_XXS expert weights row-major [N][nb][66] → block-major
[nb][N][66] at load (host transpose in `upload_3d_kquant_raw_ex`, temp buffer, same device size,
no extra VRAM), so a row-tile's read of one 256-block is contiguous. Kernel `bp = We + bg*N*66 +
n*66`. FFN 122→106ms, 2330→~2510 tok/s (hits task.md's 104ms/2526 target). Session 1578→2510.
- MMQ kernel is layout-parameterized (`bm` flag); harness `grouped_v13` validates bit-exact
  (rel_L2 vs v1 oracle = 0.000000).
- Decode also fixed: `matvec_iq2_xxs_f32` got a `bm` flag, the 4 decode MoE matvecs route via
  `launch_moe_expert_matvec` → block-major for repacked IQ2 experts. Restores generation
  correctness (REPACK-on parity 0.77 broken → 0.14, logits 0.09). DECODE BREAKS if you repack
  but forget this routing — every consumer of repacked weights must be bm-aware.
- DEFAULT ON when MMQ on; `CUDA_LLM_MMQ_REPACK=0` opts out (→ row-major 2314). Repack requires
  MMQ (cuBLAS/dequant fallback stays row-major, only reached when MMQ off). Differs from the v7
  dead-end (which coalesced via 33KB shared = occupancy death); load-time repack coalesces the
  GLOBAL read directly, no staging.
WHY the earlier tile-port/coalesce/pipeline were no-ops but THIS worked: they tried to fix a
memory-LAYOUT problem with occupancy/tile/shared-staging knobs; only changing the actual byte
layout fixed it.

## DeltaNet scan optimization (2026-06, push toward 3000 — landed ~2810, 3000 NOT reached)
Fresh warm profile (actual model config: dt_rank=32 d_state=128 n_group=16 d_inner=4096,
10 attn / 30 ssm layers — NOT the dt_rank=48 in older notes): 512-tok prefill total ~182ms
@ ~2760 tok/s warm (cold runs read ~2500; GPU boost clock matters, always warm up + interleave A/B).
Detail (ms): ffn experts(MMQ)=88, **scan=33.8**, gemm projections=35, attn core=15, conv=7, topk=6.5.
The DeltaNet scan is the #2 kernel and the only big non-MMQ lever.
- **WIN: pack W=4 independent warps/block (commit 9e37fb6): scan 33.6->30.2ms, 2760->2810 tok/s
  warm, bit-correct.** The d_state==128 register path ran 1 warp/block = grid(dt_rank,128);
  1 warp/block caps occupancy at the 32-blocks/SM limit = 32 warps/SM (half the 64 ceiling).
  Change `r = blockIdx.y*blockDim.y + threadIdx.y`, launch grid(dt_rank,128/W) block(32,W), NO
  shared/NO __syncthreads (warps fully independent). CUDA_LLM_SCAN_W={1,2,4} default 4.
- **DEAD-END: shared-q/k scan (R rows/block share per-token q/k via shared + barrier)**: SLOWER
  at every R (scan 33->46ms@R4, 57ms@R2). The 128x-redundant q/k reads are L2-cached & cheap;
  the per-token __syncthreads serializes the R warps and kills the latency hiding that many
  independent single-warp blocks provide. The scan is LATENCY-bound on the sequential recurrence,
  NOT bandwidth-bound. Same lesson as v7/coalescing: don't fight latency with shared-staging.
- **Projection-GEMM fusion (analyzed, NOT done)**: q/k/v (and ssm qkv/gate/alpha/beta) share input
  d_batch_xb. Fusing into one concat-weight GEMM is FLOPs-bound (no FLOP saving) AND the per-token
  row-major cuBLAS output [n_tokens, sum_dims] makes each projection's slice strided — downstream
  conv1d/l2norm/deltanet/attn all assume contiguous [n_tokens, dim], so fusion needs invasive
  stride plumbing or a de-interleave copy that eats the launch savings. Low ROI; skipped.
- **3000 is NOT reachable.** The wall: scan (30ms) is latency-bound, MMQ FFN (~68-88ms) is at its
  MoE floor, projections are FLOPs-bound. Realistic warm ceiling ~2810.
- **DEAD-END: chunked-parallel scan rewrite (UT-transform/gated-delta).** Attempted per user request.
  CPU prototype `cuda/llm/scan/deltanet_chunk_test.c` validated the math (rel_L2 3e-7 vs sequential).
  CUDA port validated bit-approx (rel_L2_vs_seq 0.165) after the key numerical fix: P(j->t)=Gamma_t/
  Gamma_j MUST be the incremental product prod gamma_i (always <=1), never a division — dividing by
  cumulative decay = 0/0 NaN when a head's decay underflows (sequential form has no such division;
  flooring Gam is a lossy hack that pushed rel_L2 to 0.27, the product form restores 0.165).
  BUT 9x SLOWER (scan 30->279ms, 2765->1172 tok/s): one-block-per-head = only dt_rank=32 blocks
  (8192 thr) on 36 SMs (~15% occupancy). Chunked trades the original's 4096 independent (h,r) warp
  recurrences for fewer/larger matmuls, but 32 heads is far too small a batch to fill the GPU.
  Chunked linear-attn needs a large head/batch dim or big-batch TC GEMMs to win. Reverted from runner
  (prototype + finding kept). NVRTC dynamic shared >0 needs cuFuncSetAttribute(8, bytes) opt-in;
  a failed scan launch left d_batch_mid garbage -> router/MMQ read 0x0 (looked like an mmq bug).
- **NVRTC cache gotcha**: /tmp/cuda_llm_sm_120_*.ptx cache hash did NOT distinguish an edited
  kernel source from a prior build in one iterate-revert-rebuild cycle → served a STALE cubin →
  rel_L2 jumped to 1.3 (looked like a correctness bug). `rm -f /tmp/cuda_llm_sm_120_*.ptx` fixed it.
  When iterating on NVRTC kernels and parity looks broken, clear the PTX cache before debugging.

## Harness caveat
`mmq_iq2xxs_moe_test.cu` uses small random weights that FIT IN L2, so it is valid for
CORRECTNESS only — it cannot measure runner memory effects (codebook, coalescing both
showed misleading harness numbers). Only the runner `--large-bench 512` A/B is valid for perf.

## Parity
e2e rel_L2_vs_seq ~0.14-0.15 = expected F16-TC-projection precision + atomicAdd-scatter
nondeterminism (varies run-to-run), NOT a bug. Result: PASS.
See [[feedback_git_push]] — always ask before pushing.
