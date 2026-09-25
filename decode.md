# Qwen3.8-27B FP4 / true FP6 decode on one A64FX node: q38d checkpoint

Updated: 2026-09-25 JST, interactive allocation 51895461 (host c25-3104b,
48 cores, 2.0 GHz, `eco_state=0`). Root `resume.md` and the older
`qwen38-fp4-resume.md` concern other work; this file is current.

## Result and acceptance boundary

A dedicated decode engine, `a64fx/llm/q38d`, replaces the generic
`transformer.h` decode path for this model. Serial N=1 greedy decode,
A16 activations, same repeated sky-blue prompt as the earlier runner:

| Format (bits/weight) | 128+256 tok/s | 1024+256 tok/s (3 trials) | 4096+256 tok/s | Target |
| --- | ---: | ---: | ---: | ---: |
| FP4 (source NVFP4, 4.5; Q6_K head as exact int8) | 39.20 | **38.88 / 38.81 / 38.83** | 35.70 | 40 |
| true FP6 E2M3 from BF16 (6.25) | 29.56 | **29.47 / 29.43 / 28.73** | 27.36 | 30 |

Previous best (runner, FP4 A8): 8.675 tok/s at 1024+256. **Neither target is
met**: FP4 is ~0.75 ms/token (~3%) and FP6 ~0.6 ms/token (~2%) short at 1024
context (128/4096 columns are from the previous build, v37). Prefill (same per-token path) runs at 41.8 tok/s (FP4) and 30.9
tok/s (FP6) for 1024 tokens.

Node variance: the same v41 binary runs at 36.20 tok/s (FP4, 1024+256) on
node f29-6000c (job 51909571; every phase ~7% slower, global barrier 4.0 us
vs 2.5 us) against 38.8 on c25-3104b. Compare variants only within one
allocation.

Token identity. Every optimized run is compared position-by-position with an
F32-activation, exact-weight reference of the same weights
(`compare_tokens.py`, regex over `n= pos= id=`). The 1024-context F32
references are kept on shared storage as `tmp/q38-fast-final/fp{4,6}-f32-1024-ref.log`:

- FP4 A16 1024+256: 256/256 IDs equal to the **transformer.h runner's F32
  run** (independent implementation) in every trial; q38d's own F32 mode
  also matches the runner 256/256 (max logit diff 0.0025).
- FP6 A16 1024+256: 256/256 equal to q38d's FP6 F32 reference run.
- A8 activations fail: the runner's per-32 A8 diverges at n=237, q38d's
  per-16 A8 at n=4. A16 (centered radix-256 digits) is required.
- Sensitivity: FP4 128+256 and FP6 128+256 / 4096+256 also match their
  q38d F32 references 256/256 (max logit diff 0.0013 / 0.0036 / 0.099).
  FP4 4096+256: 256/256 (max logit diff 0.121).
- Runs are deterministic: all trials give identical IDs and logits (the
  sum of squares of K-split boundary rows is kept in the boundary slot and
  added in a fixed order, not by whichever worker finishes second).

Held-out quality evaluation (BF16 oracle, perplexity) and the qlair
simulator calibration from the previous checkpoint were **not** worked on.

## Engine (a64fx/llm/q38d)

`q38d_engine.c`: 48 persistent workers (CPUs 12..59), weights repacked in
place from the existing low-bit image (`qwen38_lowbit_model`), per layer:

- SSM layer: in-proj (qkv, z as a dual-matrix pass, alpha, beta) |
  per-head conv + gated delta rule (state transposed, no reductions) +
  gated RMSNorm | out-proj into the residual.
- Attention layer: in-proj (q+gate, k, v) | per-CMG KV head: norm, RoPE,
  cache, 12-way position split with 4-position x 6-head blocked QK, PV,
  merge, sigmoid gate | out-proj.
- FFN: gate+up in one dual-matrix pass (8-row partition; split 16-row
  activation units quantized by the second finisher) | down-proj K-chunked
  by 4 with an item-balanced partition (a worker range may end inside a
  group; the two partial sums meet in a split slot). The 6144-column
  out-projections stay unchunked (`out_kch=1`; 3 was slower).
- SSM conv history and conv weights are kept CMG-local per head.
- CMG-aligned in-projection (FP4 image; `ssm_perm`): CMG c owns SSM key
  groups 4c..4c+3 and their value heads {g, g+16, g+32}; qkv/z row blocks
  are permuted at load and alpha/beta (raw K-quant) re-gathered, so the
  in-proj -> SSM core step and the attention in-proj -> core step (already
  head-aligned) end with a CMG barrier instead of a global one. The FP6
  image has low-bit alpha/beta and keeps the global barrier.
- SSM state update in one sweep (`ssm_lazy`): the stored state is the
  decayed matrix without the last rank-1 update (kept next to it) and
  o = A q + delta (k . q).
- Each "|" is a counter barrier (2.5 us). The norm of the residual uses
  per-worker sums of squares and per-CMG cooperative quantization.
- Output head: Q6_K expanded exactly to signed bytes ("Q8K", +28% bytes, no
  decode); argmax per worker, then global. F32 KV cache.

`q38d_kern.h` / `q38d_kern_sve.S`: formats and kernels.

- Pair-interleaved 8-row groups: one SDOT result covers two 16-column units
  (lane 2r+u), so scales need no gather and activations load as 8-byte
  broadcasts (compact 1 or 2 bytes/column instead of the old 8x copy).
- UE4M3 scales are re-encoded at repack as E5M3 so that `as_float(u<<20)`
  is always a normal float. **A64FX pays ~70 cycles per instruction on
  subnormal FP operands**; this single change took decode from 7.3 to 29.9
  tok/s.
- Hand-scheduled, software-pipelined kernels (load / decode / SDOT /
  combine / convert+FMLA stages across slots). FP4: two pairs per slot
  (variant 7) and a dual-matrix variant sharing activation broadcasts
  (variant 8). FP6: 192-byte {codes, high plane} records so the weights are
  one stream, and a zip-free 6-bit decode. Q8K, Q4_K assembly paths.
- Exact-weight FP32 path (reference mode) and a double-precision C reference.

Measured on A64FX (PMU, `pmu_kern.c`, `pmu_stream.c`, `bench_insn.c`):
TBL/ZIP/UZP/AND-immediate/MOVI are FLA-only (1/cycle); SDOT, shifts, vector
AND/ORR, FMUL, SCVTF 2/cycle; MOVPRFX before SDOT is free; SDOT/SCVTF/FMLA
latency ~9, TBL ~6, integer ~4. Scalar integer ops steal FP issue slots.
FP4 A16 costs ~24 FP ops per 256 weights; the kernels reach ~76% of FP issue
and are latency-bound (commit waits on FP ops), not load-bound. With 12
cores streaming: FP4 A16 ~15 GB/s/core (~182 GB/s/CMG); FP6 A16 ~14 GB/s/core
after the single-stream layout (was 11.6, with 7.7 cycles/pair of L2-miss
stalls from three separate streams).

## Where the time goes (FP4 A16, 1024+256, ms/token)

ssm_in 3.91, ssm_core 0.98, ssm_out 1.66, attn_in 1.40, attn_core 0.63,
attn_out 0.53, ffn_gateup 9.79, ffn_down 5.23, head 1.58 (total 25.7).
Effective projection bandwidth ~560-850 GB/s. Per-phase overhead is about
10 us (barrier, norm/quantize, imbalance, stream cold start) x 5 phases x 64
layers; worker imbalance alone is ~1.4 ms/token (FFN units, 5% granularity
of the 5120-row out/down matrices; the item-balanced down removed part of
it). FP6 (34.0 ms, earlier build): ffn_gateup 13.89,
ffn_down 6.98, ssm_in 5.33, ssm_out 2.39, attn_in 1.77, head 1.47.

Tried and rejected (measured): deeper C-intrinsics pipelines (register
spills/extra scalar ops), four-ahead loads, SDOT chains of four (latency),
epoch barriers without RMW (4.8 us vs 2.5), producer-side normalization into
four CMG copies (remote stores), pairing groups of one matrix in the dual
kernel, larger warm-up prefetch, dynamic tail for FFN gate/up (job
51909571; the last 24-64 groups per CMG handed out one group at a time from
a CMG-local counter: balance became near-perfect, max-mean 0.28 -> 0.05 ms,
but each grab restarts the dual-kernel pipeline and a cold stream, ~3.5 us
vs ~2.3 us of work, so gate/up got 0.5-1.1 ms/token slower; FFN gate/up
runs at ~85% of the practical per-CMG stream rate, so its remaining
imbalance, ~0.3 ms from 45/46 groups and a slower CMG3, is not worth
chasing), `Q38D_COST_Q8K`/`Q38D_DUAL_COST` retuning (no gain), in-proj CMG barriers
plus the one-sweep SSM update (kept, but only ssm_in -0.13 ms and
ssm_core -0.05 ms, within the ~0.3 ms run-to-run noise of node f29-6000c),
SSM state prefetch in the sweep (distances 4-32 rows) or into L2 during
in-proj (no gain: the state, 37 MB/CMG, is HBM-latency-bound per core),
`Q38D_PROD_NORM` (now gives wrong tokens with the split-slot down path), a min-makespan
plan partition for ssm_in/attn_in (calibrated costs from per-lane busy
times: dual 1.88x F4, Q4K 34, Q8K 28 cycles/pair; lanes evened out but mean
busy rose, ssm_in 3.80 -> 3.86 and attn_in 1.32 -> 1.37 ms, and the max did
not drop, because the lanes that finish early give bandwidth to the others;
`Q38D_PLAN_DUMP=1` prints the plans), A64FX
hardware barrier (2.2 us, ~0.1
ms/token; would need dynamic libhwb).

## Speculative decoding (FP4): verification-kernel study

Target: 1.5-2x tok/s with speculative decoding while keeping A16
exactness. Findings so far (single core, `bench_multi.c`, generator
`gen/gen_n4.py` -> `q38d_kern_n4.S`):

- Drafters. Prompt lookup (n-gram, `ngram_sim.py` over the 256 generated
  IDs) accepts too little: 1.08-1.11 tokens per verify pass for k = 1..8.
  The NVFP4 GGUF contains the NextN/MTP layer `blk.64` (eh_proj Q4_K
  10240->5120, full attention layer with its own KV cache, NVFP4 FFN, shared
  head). `Q38D_MTP=1` runs it as a measurement-only drafter after every
  token: input [enorm(emb(x_{p+1})); hnorm(h_p)] -> eh_proj -> layer 64 at
  position p -> shared_head_norm -> head predicts x_{p+2}; chained drafts
  feed the drafter's own output hidden. FP4 A16, 1024+256 (main tokens
  still 256/256): depth-1 acceptance 223/255 = 0.875, first two 0.693,
  first three 0.486; tokens per verify pass 1.875 (k = 1), 2.60 (k = 2),
  3.07 (k = 3).
- A64FX throughput model that fits every kernel measured here: about two
  vector-register-writing instructions retire per cycle (loads, SDOT, TBL,
  FP; a fused MOVPRFX is free). KERNEL7 (N = 1) has ~36 writes per pair
  -> 18.5 cycles predicted, 17.7-18.7 measured. Measured instruction
  mixes: 8 LD1RD + 8 SDOT = 7.2 cycles; indexed SDOT (`sdot z.s, z.b,
  z.b[i]`) issues at only ~0.9/cycle, so a 16-row layout with LD1RQ +
  indexed SDOT is slower.
- Multi-token F4 A16 kernels (cycles per pair per token; N = 1 is 18.7):

  | kernel | N | L2 | HBM stream |
  | --- | ---: | ---: | ---: |
  | one group, depth-4 chains (n4a) | 4 | 14.6 | 14.5 |
  | one group, depth-2 chains (nb4) | 4 | 15.6 | 15.5 |
  | one group, alternating chain sets (n4c) | 4 | 14.9 | 14.7 |
  | two groups share each broadcast, depth-9 chains (g2c) | 2 | 17.4 | 19.5 |
  | same (g2c) | 4 | **12.2** | **13.1** |

  Every activation broadcast (one LD1RD per SDOT in the pair layout) costs
  as much as the SDOT it feeds, and A16 needs 8 SDOTs per pair and token,
  so the per-token verification cost cannot fall far below ~9-10 cycles
  (model for 4 tokens x 2 groups: 9.5).
- Full machine (`Q38D_VBENCH=T`: every layer's FP4 projections for T
  tokens with the engine's partitions and 5 barriers per layer, cores
  skipped; node f29-6000c): T = 1 23.6 ms per pass, T = 2 40.6 ms
  (20.3 ms/token), T = 4 54.4 ms (13.6 ms/token, 1.73x less per token).
  The ratio is better than on one core because T = 1 is bandwidth-bound
  with 48 cores.
- Consequence. The single-token kernels already stream at ~185 GB/s per
  CMG (the practical HBM rate), so verification pays extra compute
  without saving bytes: a 4-token pass costs ~2.6x a 1-token pass in the
  projections (~20 of 25.7 ms), plus per-token SSM/attention cores, norms
  and head. Estimated: ~1.7x at 100% acceptance of 3 drafts; with
  the measured MTP acceptance (3.07 tokens per pass for k = 3) and the
  full-machine T = 4 projections (54.4 ms): + ~1 ms chunked SSM core,
  ~1 ms 4-query attention, ~2 ms norms, ~2.5 ms 4-token head, ~3 ms of
  drafting with a cheap draft head -> ~64 ms per pass, ~20.8 ms/token vs
  27.7 on the same node, ~1.33x. A
  kernel at the model limit (~9.5 cycles) would give ~1.55x. So 1.5x needs
  (a) the verification kernel within ~5% of the model, (b) a cheap draft
  head (vocabulary subset), and (c) batched SSM (chunked delta rule),
  multi-query attention and a 4-token head with little overhead.

## Next steps (in order)

1. Remove the remaining ~0.7 ms/token. The in-projection phases are
   limited by per-CMG bandwidth, not imbalance (ssm_in ~150 GB/s/CMG,
   attn_in ~135, ffn_gateup ~171 incl. per-phase overhead); what separates
   them from gate/up is fixed per-layer cost (norm, barrier, cold stream
   start). Candidates: dataflow flags instead of the SSM
   in-proj->core barrier, cheaper SSM prep (conv/norm), hardware barrier.
2. Long context: attention core grows to ~2.3 ms at 4096 (K/V prefetch in
   the in-proj tail); overlap or restructure.
3. Held-out quality: FP6 vs NVFP4 vs BF16 logits (the old evaluator
   prototype in `qwen38_lowbit_eval.c` remains unvalidated).
4. Commit hygiene: dev probes (`bench_*`, `pmu_*`, `q38d_pipe.h`) could move
   under a tools directory.

## Reproduction

Model/image paths (Fugaku):

```text
FP4 source: ~/models/qwen38/27b/Qwen3.8-27B-NVFP4-Quality-v2.gguf
FP4 image : ~/work/gemm/qwen38-27b/tmp/q38-lowbit-20260924/hw-51893515/fp4-v1.image (16.05 GB)
FP6 source: ~/models/qwen38/27b/bf16/Qwen3.8-27B-BF16-00001-of-00002.gguf (+00002)
FP6 image : ~/work/gemm/qwen38-27b/tmp/q38-fast-images/fp6-e2m3-skipblk64-v1.image (21.02 GB)
            written with Q38_LOWBIT_SKIP_PREFIX=blk.64. (must be set to load it)
```

Stage images into the allocation's `/local` first (`dd bs=8M iflag=direct
oflag=direct`, ~190 MB/s). `a64fx/llm/q38d/stage_hook.sh` does this, copies
the binary and restores the F32 references; use it as `READY_HOOK` of
`auto_resubmit_interactive.sh` (see "Automatic resubmission" in
`a64fx/remote-dev-procedure.md`). Build natively (`make -C a64fx/llm CC=fcc q38d
q38d_test`) or cross (`clang --target=aarch64-linux-gnu -static -O2
-march=armv8.2-a+sve -mcpu=a64fx -DPF1_DIST=2048 -DPF2_DIST=32768`, as used
for the measurements). Runs:

```sh
q38d $FP4_GGUF --fmt fp4 --image /local/q38/fp4.image --act a16 \
     --prompt-tokens 1024 --gen 256        # --act f32 for the reference
Q38_LOWBIT_SKIP_PREFIX=blk.64. q38d $FP6_GGUF --fmt fp6 \
     --image /local/q38/fp6.image --act a16 --prompt-tokens 1024 --gen 256
python3 a64fx/llm/q38d/compare_tokens.py REF.log CAND.log
```

Conversion from BF16 (`--write-image PATH`) takes ~19 min with the parallel
packer. Tests: `test_q38d_kern` (QEMU or native), `test_qwen38_lowbit_model`
(ASan/UBSan). Logs of this session: `tmp/q38-fast-final/` (Fugaku shared
storage) and `tmp/q38-fast/` locally.

## Resume prompt

```text
Resume single-node A64FX Qwen3.8-27B decode in a64fx/llm/q38d. Read
decode.md first. Current: FP4 A16 38.8-38.9 tok/s, true FP6 A16 28.7-29.5
tok/s at 1024+256 (targets 40/30), all generated IDs equal to F32
exact-weight references. Remaining ~0.6-0.75 ms/token: worker imbalance in
gate/up and out-proj phases, SSM in-proj->core barrier, SSM prep, attention
at long context. Images: FP4 fp4-v1.image, FP6 fp6-e2m3-skipblk64-v1.image
(set Q38_LOWBIT_SKIP_PREFIX=blk.64.). Keep A16 (A8 diverges). Validate every
change with compare_tokens.py against the F32 runs. Use /local or repo
tmp/, never /tmp. Do not push without explicit authorization.
```
