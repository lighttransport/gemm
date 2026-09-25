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
chasing), `Q38D_COST_Q8K`/`Q38D_DUAL_COST` retuning (no gain), a min-makespan
plan partition for ssm_in/attn_in (calibrated costs from per-lane busy
times: dual 1.88x F4, Q4K 34, Q8K 28 cycles/pair; lanes evened out but mean
busy rose, ssm_in 3.80 -> 3.86 and attn_in 1.32 -> 1.37 ms, and the max did
not drop, because the lanes that finish early give bandwidth to the others;
`Q38D_PLAN_DUMP=1` prints the plans), A64FX
hardware barrier (2.2 us, ~0.1
ms/token; would need dynamic libhwb).

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
