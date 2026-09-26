# Resume: optimized multi-node prefill for Qwen3.8-27B on A64FX

The plan is in `/home/syoyo/.claude/plans/idempotent-frolicking-bengio.md`.
Code is in `a64fx/llm/q38p/`: `q38p_prefill.inc` is included into
`q38d/q38d_engine.c` and enabled with `Q38P=1`. The kernel generator is
`gen_pf.py`, which produces `q38p_kern.S`.

## Latest measured status

See the final continuation: **80.6 sustained prefill tok/s/node at 32k depth**,
**1103 aggregate tok/s for the full 32,572-token coding prompt**. The 130/node
sustained target is not yet reached. Packed QK and minimax pipeline cuts are
implemented; the corrected long-context arithmetic ceiling is documented below.

## Goal (user decisions)

- Latest long-context target: **sustained 130+ tok/s/node near 32k input**.
  Use a depth-controlled random-prefix benchmark for fast iteration and verify
  on the full coding prompt; see the final continuation section.

- An optimized prefill that scales over 1-128 A64FX nodes (one rank per
  mixer/FFN unit); near-term configurations are 6, 8, and 12 nodes. The
  current allocation and benchmark work stops at 12 nodes.
- Prefill and decode time-share the same nodes, e.g. 12-node prefill plus 3 TP4 decode groups.
- Coding-task target: a real code review prompt around 32k input tokens and up
  to 8k generated tokens. Use all 12 nodes for the prefill pass (the fast
  implementation is PP12), then TP4 or TP2 for autoregressive decode.
- The metric is **steady-state prefill throughput** with several prompts in
  flight; single-prompt latency is reported as well.
- **Historical short-context estimate** (see corrected mixed-arithmetic ceiling
  in the final section): the 16-bit arithmetic peak, 6.144 T MAC/s per node
  (48 cores x 2 GHz x 64 int16 MAC/cycle).
  - Work per token is 24.35 G MAC of weights, plus 0.10 G attention at
    L=1024 and 0.11 G SSM.
  - Bound: 250 tok/s per node. The target is 90% of it:

| nodes | bound tok/s (L=1024) | 90% target |
| ---: | ---: | ---: |
| 1 | 250 | 225 |
| 4 | 1000 | 900 |
| 6 | 1501 | 1351 |
| 8 | 2000 | 1800 |
| 12 | 3001 | 2701 |

- Practical pipeline target: **150 tok/s per node**. That corresponds to
  900 tok/s on 6 nodes, 1200 on 8, and 1800 on 12. The latest
  six/eight/twelve-node steady rates are 802.7/1074.8/1577.164 tok/s, or
  133.8/134.3/131.430 tok/s per node, below target. Single-request 32k
  prefill is 590.459 tok/s (49.205/node); TP4/TP2 decode independently
  measures 57.958/34.831 tok/s. See the September 26 handoff results below.

- A different weight layout from decode is allowed as long as everything
  fits in HBM2 (about 28 GB usable per node).
- Correctness gate: decode after prefill must match the F32 reference
  256/256 (`compare_tokens.py` against `/local/q38/ref-f32.log`).

## Done

**Phase 0 findings** (microbenchmarks in `tmp/q38p/ub`, `tmp/q38p/wstat`):
- **int16 SDOT (`sdot z.d, z.h, z.h`) runs at 2/cycle with 9-cycle latency,**
  so the 16-bit peak is real. The indexed SDOT/FMLA forms run at 1/cycle.
- **Every NVFP4 row of all 371 matrices spans ≤ 6 binades of per-16 scales.**
  Each weight row is therefore exactly int16 (≤ 12*15*2^6) times 2^(b_row-21).
- **Microkernel** `q38p_mk_r4t5` (32 rows x 5 tokens, weight tile in L1):
  - Accuracy: exact against a scalar reference.
  - Speed: 89.4% on 1 core, 87.9% on 48 cores, including the per-512-column
    epilogue (convert to fp64 and fma into memory).
  - Without the epilogue it reaches about 93.4%.
  - Transposed layouts, and more LD1RD or vector loads, are worse. The
    vector-load port (about one 64 B load per cycle) is the limit.
- **Streaming from L2** with a 4 KB prefetch and one epilogue per full K
  reaches about 90%. That would need a per-token activation scale, which is
  riskier for accuracy.

**Phase 1 (single node) engine, committed as 1c71d4d6:**
- Activations are int16 per (token, 512 columns). Accumulation is exact
  int64 per block and fp64 across blocks.
- F4 tiles expand on the fly from the decode F4 groups (SVE, bit-exact,
  including the K-chunked down matrix). Q4_K/Q8_K matrices (alpha/beta,
  attention k/v) are converted once to int16 tiles.
- GEMMs use dynamic 32-row tiles.
- The SSM core runs decode's `ssm_head_io` per token, one head per worker.
- Attention runs per (KV head, token) with decode's kernels.
- KV cache, SSM state and conv history are left in decode's layout; decode
  continues unchanged. The head for the last token uses `head_argmax`.
- Self-test `Q38P_TEST=1` checks the tile expansion against decode weights.

**Results** (1 node, FP4, 1024 prompt + 256 decode, all 256/256):

| version | prefill tok/s | note |
| --- | ---: | --- |
| old decode path | 41.8 | |
| v1 | 122.8 | 1200 tokens computed because of chunk padding |
| v2 | 142.9 | occupied groups only, vector epilogue, chunk 480 |
| v3 | **150.2** | tail splitting (`Q38P_TAIL=1`) and faster activation packing, commit 41025342 |
| v4 | **154.7** | token-major GEMM output stores, job 51917132 |

- The max logit diff against F32 is 0.886 (decode path: 0.197). It is
  identical across chunk sizes.
- v3 breakdown at chunk 480 (6.82 s):
  - gemm 5.81 s: worker-mean kernel 4.80, expand 0.41, the rest is epilogue,
    imbalance and barriers
  - norm+quant 0.20
  - ssm_core 0.31
  - attn 0.15
  - The kernel MAC rate is 84.6% of peak.
- The overall rate is 60% of the bound.

**2026-09-26 continuation, job 51917132:** At chunk 480, the GEMM
worker-mean profile before the change was expansion 0.407 s, kernel 4.793 s,
output epilogue 0.467 s, zeroing 0.020 s, dispatch 0.003 s and barrier waits
0.069 s. Reordering the epilogue to write all 32 adjacent rows of one token
before moving to the next cut epilogue time to 0.304 s and GEMM wall time
from 5.795 to 5.668 s. Throughput rose from 151.2 to 154.7 tok/s, with
256/256 decode agreement and unchanged maximum logit difference (0.8857).
Chunk 640 tied at 154.8 tok/s; chunk 1025 was slower at 152.3 tok/s, so
chunk 480 remains the measured setting. At 4096 prompt tokens, the same
change produced 142.2 tok/s and matched a fresh F32 reference 256/256
(maximum logit difference 0.5504). Attention took 4.791 s of the 28.809 s
prefill, up from 0.150 s at 1024 tokens. Logs are preserved in
`tmp/q38p/job51917132/`. The final compiled source repeated the 1024-token
gate at 154.5 tok/s and 256/256 agreement.

**Phase 2 pipeline prototype, job 51917132:** `Q38P_MPI` adds balanced
mixer/FFN stage cuts, MPI residual handoff, repeated independent prompts and
an optional state gather to rank 0 for full decode validation. The verified
8-prompt, 1024-token steady rates at chunk 160 are **275.1 tok/s on 2 nodes,
411.3 on 3, and 544.9 on 4**. The 3-node cuts split layers at units 43 and
85. All three configurations passed 256/256 decode against F32 after the
state gather. On 4 nodes, single-prompt latency was 2.55 s and end-to-end
throughput including fill/drain was 521.7 tok/s. Build/run instructions and
the chunk sweep are in `a64fx/llm/q38p/README.md`.

**Six/eight/twelve-node validation, interactive job 51918651:** At 8 prompts,
1024 tokens, and chunk 160, calibrated-partition runs measured 802.7/1074.8/
1562.8 steady tok/s and 743.0/964.2/1316.6 end-to-end tok/s for 6/8/12 nodes.
Each decode comparison matched 256/256 tokens. Fitting relative FFN, SSM-mixer,
and attention-mixer costs to the per-rank times adjusted the stage cuts; the
12-node steady result improved only 0.5%. A 12-node chunk sweep found chunk
160 best among 120/160/200/240/320/480. The 150 tok/s/node target remains
unmet; no 24-node job was submitted. Detailed results are in
`a64fx/llm/q38p/README.md`.

**Single-node 200 tok/s investigation, job 51917132:** A fresh 1024-token
baseline was 154.6 tok/s (6.624 s); GEMM was 5.668 s, including 4.820 s in
the int16 kernel, 0.408 s in F4 expansion and 0.314 s in output epilogues.
The 200 tok/s budget is 5.120 s, so raising the kernel from 84% to 90% of
peak alone would only save about 0.31 s. The remaining phases also need
large reductions. The following are exploratory measurements at chunk 480;
only the alias-safe panel copy was retained:

| Experiment | Prefill tok/s | Decode vs F32 | Finding |
| --- | ---: | ---: | --- |
| FP32 output epilogue for F4 | 153.8 | not run | Slower than baseline |
| Weight prefetch, 1024 B | 154.6 | residual hash unchanged | No full-engine gain |
| Token-major accumulator layout | 154.7 | residual hash unchanged | No full-engine gain |
| Precomputed F4 base vectors | 154.2 | 256/256 | Within noise |
| 1024-column K blocks | 134.1 | 5/256 | Kernel rate fell from 84% to 71% of peak |
| INT8 rounding of both operands | 153.4 | 0/256 | Insufficient accuracy |
| INT8 rounding of activations only | 155.0 | 4/256 | Insufficient accuracy |
| INT8 rounding of F4 weights only | 154.3 | 0/256 | Insufficient accuracy |
| Power-of-two activation scales | 154.4 | 129/256 | Insufficient accuracy |
| One activation scale per full K | 154.5 | 256/256 | Accurate, no speed gain alone |
| Int64 accumulation across K with one scale | 154.0 | 256/256 | GEMM faster, other cost offset it |

An `-O3` build initially appeared to reach 162.1 tok/s but diverged at the
first generated token. Per-layer hashes located the first difference in the
activation panel before layer 0 GEMM: an `int16_t` buffer had been read and
written through `uint64_t` pointers, violating C aliasing rules. Replacing
those accesses with `memcpy` restored the exact O2 panel and 256/256 decode
agreement under `-O3`. The corrected `-O3` run was 154.1 tok/s. This was a
false optimization result caused by missing work, not a usable speedup.
At 4096 tokens, the corrected `-O3` build reached 142.8 tok/s and matched
the fresh F32 reference 256/256 (maximum logit difference 0.5504).

## 2026-09-26 continuation: Q38D multi-node decode TP

Using the existing 12-node allocation (PJM job 51918651), extended
`q38d_tp.h` collectives to arbitrary TP sizes through 12 and added Q38D
support for TP6/8/9/10/12. TP2/4 keep their optimized path; other sizes
replicate SSM and attention, shard FFN rows/down-projection columns, and shard
the vocabulary head. The replicated-mixer path preserves the residual once
around the FFN all-reduce. Gate/up row boundaries are aligned with the same
64-column FP4 tiles used by the down projection, and CMG boundaries preserve
16-row activation units.

The debugging sequence found and fixed three correctness issues: replicated
SSM permutation still used a rank offset; uneven FFN row partitions could
split a 16-row quantization unit at a CMG boundary; and gate/up rows initially
did not match down-projection column shards. A 32-token prompt / 2-token
decode comparison matches TP1 token IDs for TP6/8/9/10/12 (2/2 each).
Collective correctness was also checked at TP9 (5,120 FP32 values, 100
reductions, 48.75 us average).

Steady decode sample (32-token prompt, 16 generated tokens):

| TP ranks | 6 | 8 | 9 | 10 | 12 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| decode tok/s | 58.061 | 60.893 | 60.590 | 61.433 | 62.312 |

The short-context rate is nearly flat across these sizes. The 12-node
32,768-token prompt / 8,192-token generation run completed through position
40,959: prefill took 1,031.478 s (31.768 tok/s), and decode took 430.460 s
(19.031 tok/s) at long context. The runner repeats the supplied prompt token
sequence to create the requested prompt length, so this measures the context
shape rather than an independently prepared 32k-token code corpus. Peak rank
RSS was about 22.3 GB with MemAvailable steady near 8.5 GB. Its profile shows
attention core at 31.798 ms/token, the main long-context decode bottleneck.
The rank-zero log is under `tmp/tp9/q38d-tp12-32k8k.rank0.long.log` on the
Fugaku checkout.

### TP12 V-cache reuse experiment

Tested an `attn_pv6` loop that loaded each 32-column V tile once and reused it
across all six query heads. Direct-GGUF TP1 and TP12 matched the first two
short-prompt IDs (`271, 760`), and the experimental and baseline TP12 runs
matched all 16 generated IDs at a 32,768-token prompt. Short-context decode
was 62.513 tok/s with reuse and 62.284 tok/s with the original loop, a
negligible difference.

A controlled direct-GGUF TP12 comparison at prompt 32,768 / generation 16
showed the V-reuse loop was slower. The original loop measured 20.661 tok/s
decode and 28.328 ms/token attention core; V reuse measured 19.165 tok/s and
32.108 ms/token. Worker-0 per-layer scores-plus-PV was 1,740.90 us baseline
versus 1,982.92 us with reuse. The prefill was also faster with the original
loop (1,029.237 s / 31.837 tok/s) than with reuse (1,087.724 s / 30.125
tok/s). The experimental loop was removed, and the main source and binary use
the original kernel again.

The older 8,192-token decode run used a prebuilt `/local` image that vanished
with its allocation. That image-backed TP1 reference generated token 71093 at
position 33, while direct-GGUF TP1 generates 760 there. Use the new direct-GGUF
baseline for future comparisons; do not compare those token sequences as
though the weights and conversion path were identical.

A second candidate reused each V tile for four query heads with a smaller
accumulator tile. It preserved all 16 generated IDs against the baseline, but
the 32k run regressed further: 16.489 tok/s decode, 40.245 ms/token attention
core, and 26.930 tok/s prefill. That candidate was also discarded. Continue
from the original two-head / 128-column PV loop; both V-reuse layouts increased
the measured long-context attention time.

## Handoff implementation (completed job 51929545)

Work on 2026-09-26 uses exactly 12 nodes, with the bash-over-HTTP bridge on
login1 port 32390 / local port 42392. Other users' or sessions' allocations
are untouched. The completed allocation was released with `pjdel 51929545`;
start a fresh allocation and regenerate topology/stage `/local` before resuming.

The fast producer is **PP12**, the existing pipeline over mixer/FFN units.
It is not Q38D tensor-parallel TP12, whose replicated mixers perform slow
sequential prefill. The intended combination is PP12 -> TP4 or TP2.

Implemented in `q38d/q38d_state.inc` and `q38p/run_handoff.sh`:

- `--state-out NEW_DIR` exports stage-owned global-head SSM state, deferred
  SSM updates, convolution rings, chronological KV, final residual and first
  output token. The producer publishes metadata after every stage finishes.
- `--state-in DIR` loads only a TP1/2/4 consumer's heads and maps KV time
  blocks onto its CMG position partitions. State import never counts as
  prefill throughput or decode time.
- Header identities cover the validated image inventory/payload checksums,
  prompt token sequence, format and arithmetic. Record checksums detect
  accidental corruption. Version 1 is local little-endian FP32 state.
- `--prompt-file PATH --prompt-tokens 0` consumes a complete file, preserving
  chat framing; positive token counts select a prefix without repetition.
- Independent prefill, export, import/readiness, generation-only rates, and
  1024-output-token decode blocks. Runner logs also contain cold process wall
  time and peak RSS. Cold process reloads remain in this prototype, so the
  sum of phase timings is not measured serving latency.
- Portable `test_q38d_state.c` passes all TP1/2/4 ranks, a 65-token KV tail,
  prompt-identity rejection and corrupted-payload rejection.

Persistent image source, recovered after the old `/local` copy disappeared:
`tmp/q38-lowbit-20260924/hw-51893515/fp4-v1.image`.
All new handoff comparisons use this same image, staged with direct I/O.
The earlier difference between image-backed and direct-GGUF token IDs is
observed; its cause has not been established.

**Correctness investigation:** arbitrary-TP work had changed the SSM setup
loop to skip TP2/TP4. Restoring `ssm_shard_layer` for every enabled SSM
permutation fixes the immediate next-token corruption. A new invariant checks
QKV/Z/alpha/beta/output shard dimensions before inference. With this fix,
PP12 -> TP2 matches 256/256 against `tmp/q38-fast-final/fp4-f32-1024-ref.log`;
PP12 -> TP1 also matches 256/256. TP4 still diverges at output token 5 (26/256
matched); a direct TP4 1024-token prefill matches the first 32 reference tokens.
Conservative flags/collectives do not fix the imported-state divergence.
First-token residual traces show identical values across all four TP ranks,
with gradual TP1/TP4 differences rather than a gross layout discrepancy.
Replaying the last 64 prompt tokens (PP12 prefix 960 -> TP4 prompt 1024)
restores **256/256 F32 agreement**. The real coding workload also matches
8,192/8,192 across TP2/TP4, both before and after the optimization below.
Logs: `tmp/q38p/handoff/{small65,check1024,fixed1024,diagnose}/` remotely.

The complete coding prompt has 32,572 tokens, 80,471 bytes, and SHA256
`77288bc1785a47c37f7db5399d404d60f8032939f82b9ab7ae2240526e262f2c`.
It is a fixed C-source excerpt with review instructions and chat framing:
local `tmp/q38-handoff/review-c-source.txt`, remote `tmp/q38p/review-c-source.txt`.
It is a throughput workload, not an evaluated code-review-quality benchmark.

### Measured optimizations and coding workload

Selected defaults: SVE softmax maximum, CMG-local prefill attention queries,
and 256-position PV cache blocks. A/B controls are `Q38D_SCORE_MAX=0`,
`Q38P_ATTN_CMG=0`, and `Q38D_ATT_PV_BLOCK=0`. Hardware unit tests cover
reduction tails/NaNs/infinities and bit-exact PV accumulation, offsets and
strided output. PV blocking preserves each output's FMA order.

Full 32,572-token prompt, PP12, chunk 160, one request:

| Configuration | Prefill seconds | Total tok/s | tok/s/node |
| --- | ---: | ---: | ---: |
| Scalar maximum, global query queue, unblocked PV | 86.440910 | 376.812 | 31.401 |
| SVE maximum | 84.330578 | 386.242 | 32.187 |
| Plus CMG-local queries | 72.710355 | 447.969 | 37.331 |
| Plus PV block 256 | 55.317369 | 588.820 | 49.068 |

All 2,434 state headers/record checksums agree across these configurations;
consumer import verifies payload checksums. Final full-prompt prefill improves
throughput 56.3%, but **does not meet 150 tok/s/node**. Two-attention-layer
stages are expensive at long context; batch QK/PV across query tiles and
retune stage boundaries using long-context measurements next.

For robust handoff, export a 32,508-token prefix and replay the final 64 known
prompt tokens in each decoder. Replay has a separate timer and is excluded
from generation throughput. This producer measured **55.055448 s = 590.459
tok/s total = 49.205/node**; snapshot export took 5.684308 s.

| Decode group | Baseline full-prefix decode tok/s | PV256 + 64-token replay decode tok/s | Replay seconds | 8192 output IDs vs baseline TP2 |
| --- | ---: | ---: | ---: | --- |
| TP2 | 29.356 | 34.831 | 1.684803 | 8192/8192 |
| TP4 | 50.059 | 57.958 | 1.013122 | 8192/8192 |

Decode measures all 8,192 generated tokens, with context growing from 32,572
to 40,764; it excludes prefill, export, import, replay and model loading.
TP2 improves 18.7%, TP4 15.8%. Baseline TP2/TP4 also matched all 8,192 IDs
(996 distinct IDs; maximum selected-logit difference 0.027112). This is
cross-configuration agreement, **not an 8k-output F32 oracle validation**.
Measured optimized import/readiness was TP2 16.237/19.380 s and TP4
8.226/10.223 s; these older readiness clocks include rank preparation skew. Maximum selected-logit differences from the baseline are
0.028494 (TP2) and 0.045350 (TP4), despite exact token IDs.
The final source synchronizes the group before starting import timing.
Both primary binaries were rebuilt. Final default TP2 and TP4 runs using
prefix 960 + replay 64 each pass **256/256 F32 reference tokens**; logs are
`tmp/q38p/handoff/final-check1024/`. Selected logits differ from F32
(maximum 0.454562 / 0.461564), so the gate is token agreement.
Portable state layout/integrity tests and shell syntax checks also pass.
Baseline peak RSS was 24.975 GiB/node for TP2, 20.205 GiB for TP4;
full-prompt producer peak was about 20.622 GiB/node.

The short streaming regression (eight independent 1024-token prompts) reaches
**1577.164 steady tok/s = 131.430/node**, 1339.300 end-to-end tok/s,
first prompt 1.572 s, with **256/256** gathered TP1 outputs against F32.
This also remains below the 1800 total target; do not substitute streaming
throughput for a single 32k-request rate.

Remote logs: `tmp/q38p/handoff/code32k-{scalar,vector,cmg,block256,prefix64}/`;
streaming logs: `tmp/q38p/pp_runs/handoff_blocked_12x8.rank*.log`.
Local copies: `tmp/q38-handoff/results/`.

Reproduce on an existing 12-node allocation, after staging the same image:

```sh
export TMPDIR=/local/q38/tmp
bash a64fx/llm/q38p/build_pp.sh
make -C a64fx/llm q38d CC=fcc Q38D_TP=1
HANDOFF_PREFIX_TOKENS=32508 HANDOFF_TPS='4 2' \
  bash a64fx/llm/q38p/run_handoff.sh tmp/q38p/handoff/NEW_RUN \
  0 8192 160 tmp/q38p/review-c-source.txt
python3 a64fx/llm/q38p/check_handoff.py tmp/q38p/handoff/NEW_RUN \
  --reference tmp/q38p/handoff/code32k-cmg/decode.tp2.rank0.log --require-exact
```

For synthetic 1024-token validation, export prefix 960, then import with
`HANDOFF_PREFIX_TOKENS=960`, full prompt 1024 and generation 256. A new run
directory is required. `run_decode_state.sh` can reuse an existing snapshot
without rerunning prefill. Tail replay is an explicit workaround for observed
sensitivity, not proof of numerical equivalence for every prompt.

## Next serving work: PP12 prefill followed by TP4/TP2 coding-task decode

Treat this as a phase handoff for one request: use all 12 nodes to ingest the
~32k-token code/review context, then reconfigure the allocation into a smaller
TP4 or TP2 group for the ~8k-token generation. A TP4 decode group leaves room
for up to three independent decode contexts on 12 nodes; TP2 leaves room for
up to six. Measure per-request decode speed and aggregate throughput at those
concurrency levels, since unused nodes only help if other requests can use
them. These are target configurations: current Q38D topology setup treats an
allocation as one collective group, so simultaneous groups need explicit group
membership, isolated collective state and rank-zero ownership.

The handoff must carry the prompt's final residual/token position and all
per-layer recurrent state: K/V cache, SSM state, and convolution history. The
PP12 producer owns state by pipeline layer, while TP4/TP2 decode owns state
by head. The canonical snapshot explicitly maps it into the decode group's
layout, including KV position partitions and SSM/conv head ownership. Do not assume that copying rank 0's buffers is sufficient. Load or
stage the TP4/TP2 weight shards before handoff so model loading is not included
in the interactive request latency.

Use a real, fixed code-review prompt corpus for the main run. The existing
`--prompt-tokens` test repeats a short prompt and is useful for shape/perf
checks only. Keep all comparisons on the same GGUF/conversion path: the
previous `/local` image and direct-GGUF load generated different token IDs.
Report prefill latency/rate, state-transfer time and bytes, time to first
decode token, steady decode tok/s, total 32k+8k latency, peak memory, and
aggregate throughput for 1xTP4, 3xTP4, 1xTP2, and 6xTP2 where applicable.
Correctness gate: compare the post-handoff generated IDs and logits against a
same-weight TP2/TP4 run that starts with the complete prompt state available.

## Remaining items

### Single node (toward 225 tok/s)
1. Continue reducing the GEMM gap. The epilogue is now about 0.30 s;
   barriers 0.07 s and zeroing 0.02 s. Profile per-GEMM imbalance next.
2. Kernel: gain the last points (epilogue cost about 4%).
   - `movprfx` from a zero register instead of `dup` zeroing.
   - Overlap group boundaries.
   - Keep the 512-column activation block: 1024 columns reduced the kernel
     rate to 71% of peak and failed the 256-token decode gate.
   - Try a per-token scale with outlier columns handled separately.
     This is accuracy-sensitive; check logit diffs.
3. Tile expansion is 0.41 s (about 8.5% of the kernel). Optimize it, or use
   a larger chunk. For multi-node, pre-expand int16 tiles in HBM: 4.1 GB per
   node at 12 stages, so no expansion at all.
4. SSM core (0.31 s): batch the per-token loop (chunked delta rule / WY), or
   overlap it with GEMMs. For now only 48 heads run on 48 workers, per token.
5. Attention (0.15 s at L=1024, grows with L): block queries so K/V are
   reused from L1/L2 (flash-style, several queries per K block).
6. norm+quant (0.20 s): fuse with the GEMM epilogue or residual add.
7. L=4096 passed 256/256 on job 51917132. Test the FP6 image (the path
   currently assumes an FP4 model with a Q6_K embedding).

### Multi-node (Phases 2-4)
8. The PP12 -> TP4/TP2 snapshot path and 32.6k+8k benchmark are implemented
   and measured above. Next reduce long-context attention cost with query
   tiling and rebalance pipeline stages; preserve independent phase clocks.
9. Extend correctness coverage to additional coding prompts and no-handoff
   references. Preserve SSM sharding invariants and validate the explicit
   64-token replay against F32; synthetic zero-replay TP4 remains sensitive.
10. Measure decode-group packing: one TP4 group, three concurrent TP4 groups,
    one TP2 group and six concurrent TP2 groups. Compare per-request latency,
    aggregate tok/s, and headroom on the 12 nodes.
11. Keep the existing 6/8/12 Q38P prefill sweep and 150 tok/s/node target for
    batched independent prompts. Its latest chunk-160 12-node result is 1577.164 tok/s
    aggregate; improve stage balance and end-to-end fill/drain separately
    from the single-request TP12 prefill handoff.
12. Replace MPI handoff with uTofu Put plus MRQ completion after the state
    snapshot shape is stable. `utofu-tests/pp_handoff_bench.c` measured
    3.2 us for a 12 KB hop and 6.3 GB/s per link; MPI nonblocking sends without
    a progress thread were slower.
13. Stage-shard the FP4 image and pre-expand int16 tiles (48.7 GB total)
    instead of replicating the image on every node. Verify HBM2 headroom for
    TP12 prefill shards and co-resident TP4/TP2 decode groups.

## How to build and run (1 node)

```
# cross build (local)
$(cat tmp/q38-fast/build2.cmd) a64fx/llm/q38p/q38p_kern.S -o tmp/q38-fast/q38p_vN
rsync -a tmp/q38-fast/q38p_vN fugaku1:work/gemm/qwen38-27b/tmp/q38-fast/
# run through the bridge (tmp/q38-fast/r.py); helper tmp/q38p/run1.sh defines
#   pf <tag> ENV...  -> prefill tok/s, decode tok/s, token match
(echo "BIN=q38p_vN; MD5=$(md5sum tmp/q38-fast/q38p_vN | cut -c1-32)"; cat tmp/q38p/run1.sh;
 echo 'pf x Q38P=1 Q38P_CHUNK=480') | python3 tmp/q38-fast/r.py 2400
```

Environment switches:

| switch | effect |
| --- | --- |
| `Q38P=1` | enable the batched prefill |
| `Q38P_CHUNK=N` | prompt chunk size (rounded to a multiple of 5) |
| `Q38P_TAIL=0/1` | GEMM tail splitting off/on |
| `Q38P_TEST=1` | expansion self-test |

- Kernel microbenchmark: `a64fx/llm/q38p/bench_mk.c` (build it with
  `q38p_kern.S`).
- Remote runs: `export TMPDIR=/local/q38/tmp` before fcc.
- Wait for the shared filesystem with an md5 loop before building (r.py
  runs remotely).

## Resume prompt

> Continue the A64FX Qwen3.8-27B prefill work: read resume-prefill.md and the
> plan in /home/syoyo/.claude/plans/idempotent-frolicking-bengio.md. The
> latest verified state includes a 6/8/12-node MPI pipeline with 256/256
> decode agreement. At 12 nodes and chunk 160 it sustains 1577.164 tok/s across
> eight prompts; see `a64fx/llm/q38p/README.md`.
> Continue with the remaining items:
> - single-node GEMM gap, kernel epilogue, expansion, SSM, attention and
>   norm costs, toward 225 tok/s (90% of the 250 tok/s/node int16 bound);
> - reduce the remaining gap to 150 tok/s/node, then stage-shard weights and
>   replace MPI handoff with uTofu;
> - continue the implemented PP12 -> TP4/TP2 snapshot handoff: 32.6k input +
>   8k output now measures 590.459 prefill tok/s, TP4 57.958 decode tok/s, and
>   TP2 34.831 decode tok/s with 64-token prompt replay. Improve long-context
>   attention and stage balance; concurrent decode groups remain future work.
>
> Keep every change 256/256 against the F32 reference and commit each unit.
> Do not use /tmp and do not push.

## Long-context depth benchmark continuation (job 51930941)

Completed interactive allocation: 12 nodes, normal 2 GHz, bridge local 42392 /
login1 32390. Job 51930941 was released after validation. Same FP4 image restaged to `/local/q38/fp4.image` on all nodes.
Read `q38p/README.md` for the new `run_depth.sh` / `summarize_depth.py` workflow.
`--bench-depth N` adds an untimed deterministic random-token prefix and measures
only `--prompt-tokens` new suffix tokens. `--bench-fill tokens` evaluates the
prefix once and checkpoints recurrent state for repeat reuse; `synthetic-kv`
creates artificial prefix KV directly for fast performance screening. Neither
permits snapshot export or decode. Total depth + suffix is limited to 49152.

Initial screening at depth 32768, suffix 1024, 8 repeats, chunk 160, PP12:

| Query tile | KV tile | Steady aggregate tok/s | Per node |
| ---: | ---: | ---: | ---: |
| 1 (baseline) | original PV 256 | 359.422 | 29.952 |
| 4 | 256 | 451.213 | 37.601 |
| 4 | 32 | 562.455 | 46.871 |
| 8 | 32 | 635.131 | 52.928 |

These are **synthetic KV screening results**, not full 32k input rates.
All final residual hashes match `c94d96b11cf1bc7e`, and each run's eight
repetitions agree. The sustained 130+/node target is still unmet. Raising the
attention partition weight to 2000 gives 636.802 tok/s, not a meaningful gain.
An intrinsic six-head/64-column PV tile regressed to 528.141 tok/s; do not
select it. The explicit SVE PV version also regressed (594.305 tok/s) and was removed.
A bit-exact six-head QK SVE kernel improves the same case to 666.754 tok/s
(55.563/node). It shares K loads and keeps 24 accumulators in registers.
Remote logs: `tmp/q38p/depth/`; local copies: `tmp/q38-depth/results/`.
Selected source defaults are query tile 8, KV tile 32, and six-head QK;
query tiling activates at context >=4096, preserving one-query short-context
tasks. `Q38P_ATTN_GEMM=0 Q38P_ATTN_QTILE=1 Q38P_ATTN_QK6=0` selects the baseline.
Decode retains its existing QK implementation. Both primary binaries have
been rebuilt. Kernel tests pass bit-exact scalar/lane references.

Selected long-context configuration: chunk 480, query tile 8, KV tile 32,
six-head QK. With depth 32768, suffix 1920, six repetitions:

- Synthetic KV screening: **814.064 steady tok/s = 67.839/node**.
- Evaluated random-token prefix: **821.824 steady tok/s = 68.485/node**;
  matched baseline (same depth/suffix/repeats, query tile 1, QK6 off, chunk 160)
  is **364.467 tok/s = 30.372/node**, so sustained throughput improves **2.255x**;
  one-time prefix setup 32.034601 s; measured six-suffix wall 19.064536 s,
  first suffix 7.383 s. Repeated final residual/token checks pass.
- Full fixed 32,572-token coding prompt: **31.707200 s = 1027.275 tok/s =
  85.606/node**, versus 588.820 tok/s before this continuation (**+74.5%**).
  Snapshot export is separately timed at 6.902624 s. All **2,434** state
  headers/record checksums agree with the prior full prompt snapshot.

The full-prompt rate averages context growth from zero; it is not sustained
throughput at 32k depth. **The sustained 130+/node target remains unmet.**
The useful next step is a larger attention GEMM tile that reuses K/V across
query heads/tokens in registers; the remaining QK/PV kernels dominate.
Cache tiling alone and contiguous stage-cost retuning cannot close this gap.

Logs: `tmp/q38p/depth/final-tokens32k/`,
`tmp/q38p/handoff/code32k-querytiles/`; local copies under
`tmp/q38-depth/results/`. The short regression passes **256/256 F32 IDs**
and sustains 1578.182 tok/s (131.515/node). Prefix-checkpoint validation also passes: baseline, optimized cached-prefix,
and direct full 34,688-token evaluation all end at residual hash
`e4b288d2d51e5f88`. The direct test evaluates one untimed token and the remaining
34,687 tokens continuously, so it does not restore the 32k checkpoint.
Logs: `tmp/q38p/depth/{baseline-tokens32k,final-tokens32k,direct-tokens34688}/`.

Reproduce the selected depth benchmark on a fresh 12-node allocation:

```sh
bash a64fx/llm/q38p/run_depth.sh tmp/q38p/depth/NEW_RUN 32768 1920 6 tokens 480 12
python3 a64fx/llm/q38p/summarize_depth.py tmp/q38p/depth/NEW_RUN
# Fast screening only (artificial KV, not evaluated prefix state):
bash a64fx/llm/q38p/run_depth.sh tmp/q38p/depth/NEW_FAST 32768 1920 6 synthetic-kv 480 12
# Matched baseline:
Q38P_ATTN_GEMM=0 Q38P_ATTN_QTILE=1 Q38P_ATTN_QK6=0 \
  bash a64fx/llm/q38p/run_depth.sh tmp/q38p/depth/NEW_BASE 32768 1920 6 tokens 160 12
```

No concurrent decode groups or 24-node allocation were used. Source changes
are confined to the prefill benchmark/attention path and its build dependency;
preexisting unrelated work remains untouched.

## Cached packed-QK continuation (job 51931098)

Job **51931098 released** after validation; `pjstat 51931098` returns no active
job. Its local bridge tunnel is closed. Allocate a fresh 12-node job to resume.

Used an interactive **12-node**, normal-mode allocation and the existing
loopback bash-over-HTTP bridge (local 42392 / login1 32390). Same validated FP4
image on every node. No 24-node allocation or concurrent decode groups.

Implemented and retained:

- `q38p_sgemm12x32`: sequential-FMA FP32 QK, two queries × six GQA heads ×
  32 key positions. `q38p_gemm_attention.h` packs Q/K and scatters scores.
- Persistent packed keys only for each PP stage's owned attention layers,
  CMG-local allocation, incremental refresh of touched 32-key panels. Prefix
  setup also fills packed keys; repeated suffixes reuse the packed prefix.
  Peak measured process RSS rises from 20.880 to **21.147 GiB**.
- Minimax contiguous stage partitioning, with portable exhaustive-reference
  tests and nonempty coverage tests for every rank count from 1 through 128.
- Benchmark output and `summarize_depth.py` record GEMM/cache selection.

New default: `Q38P_ATTN_GEMM=1 Q38P_ATTN_CACHE=1`, query tile 8, KV tile 32.
Use `Q38P_ATTN_GEMM=0` for the previous QK reduction; `Q38P_ATTN_CACHE=0`
recomputes packed keys per task. **QK rounding changes** from a 16-lane
reduction to sequential FP32 FMAs. PV, softmax and decode retain their math.
Snapshot hashes therefore differ; do not use old bit-exact state hashes as a
correctness criterion for the new GEMM path.

Measured depth 32768, suffix 1920, six repeats, chunk 480:

| Configuration | Fill | Steady tok/s | Per node |
| --- | --- | ---: | ---: |
| Previous kernel/cuts, same job | synthetic KV | 823.052 | 68.588 |
| Packed QK, repack per task | synthetic KV | 880.113 | 73.343 |
| Cached QK, previous cuts | synthetic KV | 923.151 | 76.929 |
| Cached QK + minimax cuts, attention cost 3500 | synthetic KV | 978.424 | 81.535 |
| Selected configuration | evaluated random tokens | **967.713** | **80.643** |

The evaluated prefix takes 31.813683 s once, outside timing. Six measured
suffixes take 16.525489 s including fill/drain; first suffix is 6.605 s.
Repeated residual/token checks pass (`8670bff077ce0b29`). This is **17.8%**
above the previous evaluated-prefix rate of 68.485/node. **130/node remains
unmet.** Synthetic and evaluated-prefix rates remain close.

The complete 32,572-token coding prompt, with attention cost 2000, takes
**29.530393 s = 1102.999 tok/s = 91.917/node**, versus 1027.275 before this
continuation (+7.4%). Snapshot export adds 5.552113 s, independently timed.
TP2 checks the first 256 baseline continuation IDs exactly, with maximum
selected-logit difference 0.0474892; generation-only rate is 36.387 tok/s
for this bounded run. Do not compare a 256-output decode rate directly with
an 8192-output average at growing context.

Validation:

- A64FX packed/cached QK unit test versus scalar sequential FMA, odd query
  counts 1/3/15, even counts 2/8/16, partial 73-key panels. Microbenchmark:
  old QK 22.996 ms, packed 15.932 ms, cached 11.038 ms for eight queries at
  32k; max old/new score difference 1.1920929e-6. This is kernel-only timing.
- 1024-token F32 check: 256/256 IDs, max selected-logit difference 0.7177.
- 4096-token F32 check: 256/256 IDs, max selected-logit difference 0.5393.
- Final default short pipeline: 1563.739 steady tok/s = 130.312/node,
  256/256 F32 IDs. Both MPI producer and main TP decoder rebuilt with fcc -O2.
- Portable partition test: `cc -O2 -Wall -Wextra -Wpedantic
  a64fx/llm/q38p/test_q38p_partition.c -lm -o tmp/q38-packed/test_partition`.

Rejected candidates: packed-P/packed-V PV, direct twelve-head PV, packed-P
with direct V, query tile 16. Direct PV suffered severe power-of-two score
stride cache conflicts (~190 ms kernel time, ~23 ms after padding), but the
existing PV kernel is still faster (~18 ms). Chunk 960 gets 82.321/node but
first-suffix latency rises to 11.100 s; retain 480. Experimental PV sources
are only preserved in `tmp/q38-packed/`, not in the production path.

Reproduction (fresh existing 12-node allocation, image staged):

```sh
bash a64fx/llm/q38p/build_pp.sh
Q38P_PP_ATTN_COST=3500 bash a64fx/llm/q38p/run_depth.sh \
  tmp/q38p/depth/NEW_PACKED 32768 1920 6 tokens 480 12
python3 a64fx/llm/q38p/summarize_depth.py tmp/q38p/depth/NEW_PACKED
Q38P_PP_ATTN_COST=2000 HANDOFF_PREFIX_TOKENS=32508 HANDOFF_TPS='4 2' \
  bash a64fx/llm/q38p/run_handoff.sh tmp/q38p/handoff/NEW_PACKED \
  0 8192 480 tmp/q38p/review-c-source.txt
```

Remote logs: `tmp/q38p/packed/`, `tmp/q38p/handoff/code32k-packed/`,
`tmp/q38p/handoff/code32k-packed-prefix64/`, `tmp/q38p/pp_runs/packed_*`.
Local copies: `tmp/q38-packed/results/`. Keep the fixed coding prompt and
validated FP4 image identity for comparisons.

### Corrected arithmetic ceiling at long context

The early ~250 tok/s/node estimate is a short-context approximation. Counting
**both QK and PV**, the current model's 16 attention layers × 24 query heads ×
256 dimensions require `2 * 16 * 24 * 256 * context` FP32 MAC/token: 0.2013 G
at 1024 and **6.4425 G at 32768**. FP32 peak at 2 GHz is 3.072 T MAC/s/node
(6.144 TFLOP/s), versus 6.144 T MAC/s for the int16 dot-product weight path.
Using the existing estimates of 24.35 G weight MAC/token and 0.11 G FP32 SSM
MAC/token gives the compute-only bound:

`1 / (24.35e9 / 6.144e12 + (6.442450944e9 + 0.11e9) / 3.072e12)`

= **164.0 tok/s/node at depth 32768**, or ~162.4 at the midpoint of the
measured suffix. Packing, quantization, softmax, synchronization, memory and
pipeline imbalance reduce practical throughput. Thus 130/node requires about
80% of this mixed-arithmetic ceiling, rather than 52% of the old 250 figure.
The measured 80.6/node is ~50% of that ceiling. This is an arithmetic estimate,
not a hardware-counter measurement or a promise of achievable throughput.

Next optimization priorities: the cost-3500 profile has two-attention stages
at ~3.9 s GEMM + 7.3 s attention, and one-attention/heavier-FFN stages at
~7.0 s GEMM + 3.7 s attention (six suffixes). Even perfect PV alone cannot
close the gap; improve projection GEMM work and attention together, then
refit stage costs. Existing `a64fx/exp2-sve/TWO_PASS_FLASH_ATTENTION.md` and
its packed 8×48 kernels are useful candidates, but their approximate exp2
must not replace this model's softmax. Any reuse needs the same end-to-end
packing costs and correctness gates; standalone peak numbers are insufficient.

One untested refinement of the rejected PV experiments remains: pad score-row
strides *before probability packing*, since the pack loop also reads twelve
strided score streams. The packed-P microbenchmarks used an unpadded 128 KiB
head stride; the separate direct-PV experiment proved that this stride can
cause cache conflicts. Retest packing-inclusive timing with padding before
concluding that every packed-P layout is inherently slower. Keep the current
PV path until a full-model gain and correctness check are demonstrated.

### Final full 32k-input / 8k-output validation

Final sources rebuilt and tested on job 51931098. The 32,508-token producer
prefix takes **29.486634 s = 1102.466 aggregate prefill tok/s** (91.872/node),
with snapshot export separately timed at 5.438342 s. Each decoder replays the
remaining 64 known prompt tokens before generating 8192 outputs:

| Decode group | Replay s | Generation s | Decode tok/s | Baseline IDs |
| --- | ---: | ---: | ---: | ---: |
| TP4 | 1.015371 | 141.577 | **57.863** | **8192/8192** |
| TP2 | 1.687514 | 235.698 | **34.756** | **8192/8192** |

Both match `code32k-cmg/decode.tp2.rank0.log`, with maximum selected-logit
absolute differences 0.102028 (TP4) and 0.101627 (TP2). This is full-length
agreement with the existing quantized baseline, not a full-length F32 check.
Decode throughput is within 0.3% of the previous 57.958 / 34.831 measurements.
Imports take 7.761373 s (TP4) / 21.817327 s (TP2), outside generation timing;
shared-filesystem snapshot transfer is still not a production handoff.

Validation command:

```sh
python3 a64fx/llm/q38p/check_handoff.py \
  tmp/q38p/handoff/code32k-packed-prefix64 \
  --reference tmp/q38p/handoff/code32k-cmg/decode.tp2.rank0.log --require-exact
```

### Follow-up: long-context PV and exactness (12-node job 51931363)

The current default remains cached FP32 QK + the original FP32 PV kernel,
query tile 8, key tile 32, chunk 480. The job used the same validated FP4
image on 12 normal 2 GHz A64FX nodes. It did not allocate 24 nodes. With the
final source build, depth 32768, suffix 1920, six prompts and synthetic KV,
the default reaches **970.166 aggregate / 80.847 tok/s/node** steady. A
1024-token prefill still matches **256/256 F32 continuation IDs** (maximum
selected-logit difference 0.7177). The earlier full coding-prompt default
validation above remains the exact 8192-token TP4/TP2 reference.

An optional int16 PV path caches value panels in the existing SDOT weight
layout and packs exponentiated probabilities per query tile. Its unit test
checks causal and partial panels; the maximum unnormalized PV error in its
four cases is 0.00125 without value residual and 0.00082 with residual.
`Q38P_ATTN_PV_INT16=1` enables it; `Q38P_ATTN_PV_RESIDUAL=1` adds a second
int16 panel for the value rounding residual. Both default to zero. **Neither
is validated for the full coding task: leave them disabled for exact TP2/TP4
continuations.** The single-panel 32k kernel microbenchmark, including SVE
probability packing, was 5.4 ms versus 18.7 ms for the old FP32 PV loop, but
full-model and decode checks determine whether that speed is usable.

| Prefill candidate | 32k steady tok/s/node | 32,508-token coding prefix | Full 8192-token baseline comparison |
| --- | ---: | ---: | --- |
| Default FP32, final rebuild | **80.847** | 1102.466 aggregate (prior full run) | TP4 8192/8192; TP2 8192/8192 |
| Int16 PV, fixed probability scale, cost 2250 | 99.266 synthetic; 99.757 evaluated prefix | 1178.283 aggregate | TP4 and TP2 first diverge at output 11 |
| Int16 PV, per-512-key probability scale | not separately depth-timed | 1178.912 aggregate | TP4 8192/8192; TP2 diverges at 7018 |
| Int16 PV, per-256-key probability scale | 98.124 synthetic | 1171.296 aggregate | TP2 diverges at 4377 |
| Int16 PV, per-512-key probability scale + value residual | 91.953 synthetic | 1132.036 aggregate | TP2 diverges at 7018 |

The current experimental header uses 512-key panels and per-panel
probability scaling. The 256-key row was a temporary compile-time variant,
now restored to 512. The residual build's TP2 generation is 34.708 tok/s;
the first per-512-key build gave TP4 57.831 and TP2 34.820 tok/s, consistent
with the validated default decode rates. The optimized producer snapshot and
its decode timings are independent: the failed ID checks reflect prefill
state differences, not a decode-speed regression. A 256-token agreement is
insufficient; the 512-key variant passed 256/256 before failing at 7018.

Other screened paths on this job did not beat the exact default:

| Exact FP32 setting | 32k steady tok/s/node |
| --- | ---: |
| Key tile 16 / 64 / 128 (query tile 8) | 75.328 / 71.833 / 64.565 |
| Query tile 4 / 12 (key tile 32) | 76.103 / 76.286 |
| Packed FP32 PV, key tile 32 | 76.838 |
| Packed FP32 PV, key tile 256 + score padding | 80.914 (noise-level) |

Score padding alone was neutral. Direct exact FP32 multi-query PV kernels
were bit-identical in the microbenchmark but slower: at 32k the existing
loop took 34.5 ms, a three-head/64-column kernel 50.4 ms, two queries
102.9 ms and four queries 208.7 ms. An isolated int16 QK prototype reached
10.025 ms versus 10.760 ms for cached FP32 QK at 32k (maximum score error
3.86e-5), before its packing overhead; it was not integrated.

The profile explains the remaining 130/node gap. Across six 1920-token
suffixes, the exact default's measured unit work sums to **124.467 s** over
all pipeline stages. Even perfect repartition would need each of 12 stages
below `11520 / (12 * 130) = 7.385 s`, hence total work below 88.62 s:
**at least 28.8% less exact-path work**. The observed best contiguous cut
from the profile is 11.807 s, less than 1% better than the current cuts.
For the faster but non-exact int16 path the total work is 103.65 s and the
best cut is 9.57 s, so partition tuning also cannot get that path to 130.
The remaining high-value target is the A16/F4 projection GEMM: on a heavy
stage its profiled kernel time is about 6.18 s versus 0.48 s for expansion.

Reproduce the exact default on a fresh staged 12-node allocation:

```sh
bash a64fx/llm/q38p/build_pp.sh
Q38P_PP_ATTN_COST=3500 bash a64fx/llm/q38p/run_depth.sh \
  tmp/q38p/depth/NEW_EXACT 32768 1920 6 synthetic-kv 480 12
Q38P_PP_ATTN_COST=3500 Q38P_RUN_TAG=NEW_EXACT_SHORT \
  Q38P_REF=tmp/q38-fast-final/fp4-f32-1024-ref.log \
  bash a64fx/llm/q38p/run_pp.sh 12 1 1024 160 1
```

Remote experimental logs: `tmp/q38p/int16/`,
`tmp/q38p/handoff/code32k-int16*`, `tmp/q38p/fp32-tile/` and
`tmp/q38p/fp32-qtile/`. Local selected copies and isolated prototypes are
under `tmp/q38-next/`. The int16 test command is:

```sh
fcc -Nclang -O2 -march=armv8.2-a+sve -mcpu=a64fx -Wall -Wextra -Wpedantic \
  -Ia64fx/llm/q38p a64fx/llm/q38p/test_pv_int16.c \
  a64fx/llm/q38p/q38p_kern.S -lm -o /local/q38/bin/test_pv_int16
/local/q38/bin/test_pv_int16
```

### Job 51931991: int16 probability residual (12 nodes)

The optional `Q38P_ATTN_PV_P_RESIDUAL=1` packs the int16 probability
rounding error per 512-key panel. It requires `Q38P_ATTN_PV_INT16=1` and
defaults to off. Combined with the existing value residual flag it computes
P0V0 + P1V0 + P0V1 in three SDOT passes; the P1V1 cross term was below
output-float resolution in the unit cases. Fixed half-step probability
residual scaling removed one input scan, with essentially unchanged measured
32k throughput (87.020 to 87.094 tok/s/node). The four-pass version before
removing P1V1 reached only 71.570/node.

| Candidate | 32k steady tok/s/node | 32,508-token coding prefix aggregate tok/s | TP2 8192-token baseline comparison |
| --- | ---: | ---: | --- |
| Exact FP32 default | 80.847 | 1102.466 | 8192/8192 |
| Int16 base, previous fixed P scale | 99.266 | 1178.283 | first mismatch at 11 |
| Int16 base, per-512-key P scale | not separately timed | 1178.912 | first mismatch at 7018 |
| Int16 + P residual, original row scan | 87.020 | 1125.701 | first mismatch at 4377 |
| Int16 + P residual, fixed scale | 87.094 | not run | not run |
| Int16 + P and V residual, three passes | 78.559 | 1041.679 | first mismatch at 4377 |

The long TP2 checks used identical 32,572-token coding prompts, 32,508
prefill tokens, 64 replay tokens and 8192 generated tokens. P-residual and
three-pass snapshots both matched the first 256 IDs, then failed the full
gate. TP2 generation rates remained 35.047 and 35.093 tok/s respectively.
The three-pass unit result has maximum unnormalized PV error 2.83e-6
against a double sum at a 1535-key prefix; the original FP32 FMA summation
itself differs from double by up to 5.64e-5 there. The more accurate SDOT
result still changes long autoregressive IDs. This is evidence of different
floating-point reduction order as a possible contributor, not a proof of
which layer caused the divergence. Keep int16 PV off by default.

Remote logs: `tmp/q38p/presidual2/{p_only,both,both3,p_only_fixed}/` and
`tmp/q38p/handoff/code32k-int16-{pres,both3}*`. Rank-zero copies are in
`tmp/q38-next/results/job51931991/`; rerunning `compare_tokens.py` on those
copies confirms the same 4377-token matched prefix for both full candidates.
The target of 130+
tok/s/node at depth 32k remains unmet. Because the best int16 single-panel
stage work is 103.65 s across twelve stages, even perfect partitioning
cannot reach 130/node without reducing the A16/F4 projection GEMM work.

### Qwen3.8-27B NVFP4 multi-context follow-up (12-node job 51932259)

The current code can run three independent TP4 decode groups on the 12-node
allocation. Each group holds independent SSM/convolution and packed INT6 KV
state for multiple contexts; decode visits the slots round-robin. PP12 owns
only its stage-local attention KV and can write version-2 compressed state
for TP4 import. The grouped importer passed six distinct short C-source
contexts: all six output hashes matched separate direct TP4 runs for eight
generated IDs. See `a64fx/llm/q38d/Q38D_MULTICTX_12N.md` for commands and
capacity details.

On one **evaluated** 32,768-token C-source review prompt, FP32 PP12 prefill
plus INT6 export processed 1,069.68 tok/s total (89.14 tok/s/node), exported
in 5.39 s, and TP4 imported in 2.95 s with an exact first-token match.
TP4 generated 256 tokens at 35.66 tok/s. A separate 1,024-token C-source
prompt matched the FP32 state path for all 64 generated token IDs. The
32K one-request prefill remains below the 150 tok/s/node target. TP4
generated 8,192 tokens from the same INT6 state at 33.32 tok/s. An FP32
state from the identical evaluated prompt generated at 57.37 tok/s. The
continuations matched the first 275 IDs, then diverged. FP32 remains the
appropriate 32K coding path where memory permits; INT6 is an opt-in
deep-context capacity experiment, pending output-quality evaluation.
The 32K state was imported into eight independent decode slots (3/3/2)
on the same 12 nodes. Each group generated 256 tokens per slot at about
36 aggregate tok/s; all eight 256-token sequences exactly matched the
single-context trace. Those eight slots reused one evaluated state; a
separate six-state short run verified distinct C-review prompts.

The three requested simultaneous context shapes pass allocation and
synthetic-depth decode screens with `--kv-i6 --prune-model`:

| Synthetic depth × contexts | TP4 group sizes | Busiest-group aggregate decode tok/s | Busiest post-fill RSS |
| --- | --- | ---: | ---: |
| 262,144 × 32 | 11/11/10 | 7.03 | 23.33 GiB |
| 524,288 × 16 | 6/5/5 | 3.31 | 24.63 GiB |
| 1,048,576 × 8 | 3/3/2 | 1.80 | 24.49 GiB |

These rows filled the earlier KV positions with finite zero rows; they do
not include real prefill or prove long-context quality. The vectorized INT6
K/V scan improved the 512K and 1M groups from 0.67 and 0.34 aggregate
tok/s respectively. At 1M PP12 synthetic depth, one evaluated suffix token
used at most 19.68 GiB RSS on the worst stage. Splitting the packed-KV
single-query attention scan across 12 workers per KV head cut end-to-end
latency from 18.97 to 1.80 s with the same first token (96066); reduction
hashes differ. Neither the 1M history nor the three multi-context shapes
have been fully evaluated from real input.

Further on the same job, packed INT6 PP12 prefill of a **fully evaluated**
32,768-token C-source prompt reached 317.13 tok/s total (26.43/node),
versus 1,069.68 tok/s total for FP32 PP12 followed by INT6 export. The
packed producer's TP4 INT6 decoder matched the FP32 producer's first
generated token but diverged at output index 1 (333 versus 271), even though
both decode caches used INT6. This makes real long-input quality a blocking
unknown for the 1M-capacity path. Eight-slot FP32 batched decode at 32K
reached about 60 aggregate tok/s per TP4 group. For repeated prompt/state
slots, a checked first import followed by resident state clones reduced
three-slot state readiness from 93.39 to 38.78 s, with all eight 256-token
output sequences matching the single-context FP32 reference. INT6 cloning
also passed eight slots; each extra state copy took about 0.05 s. An INT8 PP
cache with INT6 state export was briefly screened on a 1,024-token C-source
prompt: 330.49 prefill tok/s total, with only two output IDs matching the
FP32 producer before divergence. It was slower than FP32 prefill and did not
solve the quality gate; the experimental code was removed.

Eight distinct 32,768-token C-source review prompts were then evaluated
sequentially on PP12 with FP32 K/V. Each prefill reached 1,065.05–1,072.53
aggregate tok/s (88.75–89.38 per node) in 30.55–30.77 s of compute; each
state export took 5.54–5.96 s. The three TP4 groups imported separate states
for 3/3/2 contexts and generated 256 outputs/context at
60.54/61.44/61.23 aggregate tok/s. All 12 ranks passed the grouped checker
and all eight output hashes differed. Separate state imports took
129.38/126.89/80.32 s per group, much longer than resident cloning of one
state. `a64fx/llm/q38d/run_multictx_handoff_12n.sh` now automates distinct
PP12 exports and grouped decode for 6–32 prompt files; packed INT6 PP/KV is
opt-in for depths above FP32 capacity. Its decode loop remains round-robin,
without fused model projections across contexts.
With `Q38_MULTI_KV_I6=1`, six distinct 1,024-token prompts (two per TP4
group) passed the full PP12 version-2 export and grouped TP4 import/decode
check. Packed prefill measured 569.26–580.34 tok/s total; the groups
generated 64 tokens per context at 85.31/85.90/86.55 aggregate tok/s,
with six distinct hashes and all 12 ranks passing. This is a short-depth
wiring check, not a deep-context quality result.
The same eight distinct FP32 states then generated **8,192 tokens per
context**. All 12 ranks passed, with eight distinct output hashes; the
3/3/2 groups delivered 58.05/58.94/59.19 aggregate tok/s over
423.40/417.00/276.81 s of decode. The busiest group's full run took
592.80 s including model startup and 131.03 s of separate state imports.
This validates long grouped decode at 32K, not the 256K–1M real-input
quality/throughput target.

An isolated A64FX INT6 row-unpack microbenchmark compared the current
scalar code to a NEON deinterleave plus SVE conversion path. Both produced
identical 256-value rows; the scalar path reached 11.62 million rows/s and
the NEON/SVE path 7.37 million rows/s on one worker. The slower candidate
was not integrated. A second 4,096-entry pair lookup plus SVE conversion
candidate was also exact but slower: 5.38 versus 11.64 million rows/s.
Neither unpack variant was integrated into production.

### Exact FP4 preexpansion on 12-node job 51934716

The 32,768-token C-source review prompt was **fully evaluated**, one request
through PP12, with Qwen3.8-27B NVFP4 and FP32 KV. The producer's prefill
timer excludes model loading, state export, TP4 import, and decode. A new
`Q38P_PREEXPAND_F4_MIB` option expands selected stage-owned FP4 weight
matrices once into the same exact INT16 panels used by the existing GEMM. It
defaults to zero and is limited to 2048 MiB per rank. The expanded panels are
first-touched by workers on their owning CMG.

| Configuration | Chunk | Prefill s | Total tok/s | Tok/s/node |
| --- | ---: | ---: | ---: | ---: |
| No preexpansion, original cuts | 160 | 31.354952 | 1045.066 | 87.089 |
| No preexpansion, original cuts | 480 | 29.753986 | 1101.298 | 91.775 |
| Full 3.5–4.0 GiB preexpansion, original cuts | 160 | 28.792208 | 1138.086 | 94.840 |
| 2048 MiB cap, attention cost 2000 | 160 | 29.073732 | 1127.065 | 93.922 |
| 2048 MiB cap, attention cost 1500 | 160 | 28.476962 | 1150.685 | 95.890 |
| 2048 MiB cap, attention cost 1500 | 320 | 28.206088 | 1161.735 | 96.811 |
| **2048 MiB cap, attention cost 1500** | **480** | **28.084219** | **1166.776** | **97.231** |

The selected 480-token configuration is 5.94% faster than the same-chunk
control. Its final residual hash (`e0e955e58b6353af`) and all 256 TP4
continuation IDs and reported logits match the non-preexpanded control exactly.
State export took 5.382 s, TP4 import 7.812 s, and 256-token generation
ran at 60.481 tok/s; these are separate from prefill throughput.
The 2 GiB cap preexpanded about 1.97–1.99 GiB and 15–17 FP4 matrices per
rank. An unrestricted 3.5–4.0 GiB variant completed once at chunk 160, but
subsequent runs at chunks 160 and 480 triggered real per-node OOM kills on
different ranks, despite total RSS around 18 GiB. The A64FX HBM NUMA-domain
limit makes that configuration unsafe. The 2 GiB cap completed the measured
160, 320, and 480 runs.

The profile at 480 has a 23.283 s busiest stage (16.025 s GEMM, 5.154 s
attention), with 28.084 s total pipeline latency. Refitting the attention
partition cost from 2000 to 1500 reduced long-stage imbalance. An experimental
160-token first chunk followed by 480-token chunks gave 97.137 tok/s/node,
so the extra scheduling option was removed. Exact prefill remains below the
150 tok/s/node target; the stage's projection GEMM and attention are the
largest measured costs.

One more exact attention screen used `Q38P_ATTN_PV_GEMM=1` and
`Q38P_ATTN_SCORE_PAD=64` with the selected preexpansion/cuts/chunk. It
retained the same residual hash and 256/256 TP4 IDs with zero reported logit
difference, but slowed prefill to 1128.362 tok/s total (94.030/node).
On a two-attention stage, PV worker time rose from about 4.94 to 5.42 s.
Keep the default FP32 PV kernel.

Reproduce on a staged 12-node allocation:

```sh
bash a64fx/llm/q38p/build_pp.sh
Q38P_PP_ATTN_COST=1500 Q38P_PREEXPAND_F4_MIB=2048 \
  HANDOFF_KV_I6=0 HANDOFF_TPS=4 \
  bash a64fx/llm/q38p/run_handoff.sh tmp/q38p/new-preexpand32k \
  32768 256 480 tmp/q38d-context-prompts/review_q38d_32768.txt
```

Run directories and per-rank profiles from job 51934716 are under remote
`tmp/q38p/preexpand-*51934716/`. Use fresh output directories when rerunning.
