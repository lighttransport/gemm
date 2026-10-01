# Strata-inspired GLM53F optimization: implementation and validation

Updated 2026-10-02. Targets are **100 delivered decode tokens/s and 2000
prefill tokens/s**, on twelve A64FX nodes, the complete 45-layer
UD-Q4_K_XL model, top-8 routing, and the saved roughly 8K coding prompt.
These targets have **not been demonstrated** by the changes below.

## October 2 continuation (qualification pending)

The new opt-in candidates preserve the existing arithmetic and reduction
order: four-key index scoring reuses query vectors, MLA values keep four
64-column accumulators in SVE registers, and a derived FP16 latent view
avoids repeated rounding and selected-row packing. The canonical FP32
cache remains available for reference and state traces. The derived view
is allocated only for native replicated caches at the A64FX vector length;
context-parallel caches retain their existing path. Appends refresh each
row, and rollback hides future rows using the canonical context length.
Cache memory accounting includes the additional two bytes per latent.

Native fast/conservative value and conversion checks pass at 1/12/47/48
threads. The actual prefill logits/softmax/value kernels match byte for
byte across 60 cases (all 1–6 head counts, contiguous/reversed selections,
key counts 1/7/128/2048/2051, and output guards). Index arithmetic also
passes the native fast/conservative thread-count matrix. These arithmetic
results do not establish a whole-model throughput improvement.

PJM 52075759 is staging boundedly before a sequential campaign. The campaign
will compare the previously qualified page-none/47-thread configuration,
a rebuilt control, independent index/value ablations, and both cache/index
combinations. Complete output IDs, sampled memory headroom, independent
five-trial confirmation, 1024-transition decode, short128, and synthetic
32K are required before promotion. A complete-state legacy/persistent
checker can exercise the new kernels using diagnostic
`GLM53F_EXECUTOR_INDEX_KERNEL=2 GLM53F_EXECUTOR_MLA_KERNEL=3`; it frees the
reference before loading a fresh cache-enabled candidate.
See the active allocation and queue in `../../resume-strata.md`.

## Implementation

The source studied is `~/work/Strata`, branch `glm53f`, revision `e486a95`.
The implementations here are independent; no Strata source was copied.

| Strata technique | A64FX implementation |
| --- | --- |
| Reusable CPU pool with epoch publication | `glm53f_team.c`: one persistent OpenMP team, controller participation, padded per-worker completion, stack argument lifetime through completion |
| Weight reuse across verification positions | `glm53f_iq_bridge.c` and `glm53f_iq_fast.h`: pair Q4/Q5 rows across up to four positions, preserve independent accumulators and scalar route order |
| Head-owned causal recurrent state | KDA verification owns a head across chronological positions and records every accepted-prefix state |
| Suffix drafting and measured policy | `glm53f_lookup_spec_12n.c`: depth 1–4 suffix proposals, target verification, prefix rollback, bonus/correction token, and fallback based on elapsed time per delivered token |
| Bounded communication ownership | `glm53f_collective_12n.c`: optional serialized owner for mixed MPI/uTofu requests while draining published prefill slabs |
| Reuse across independent heads | Sparse index scores reuse key vectors across eight independent head accumulators, retaining each head's reduction order |
| Keep intermediate accumulators in registers | Native MLA absorbed-query projection uses four SVE accumulators per 64-column tile instead of repeatedly updating scratch |

Persistent decode outlines mHC, projections, KDA, sparse attention, dense FFN,
routed/shared experts and the vocabulary head into reusable team callbacks.
The normalized mHC output can feed the router in the same team dispatch.
MoE combine has an optional SVE row kernel and a producer/collective overlap
mode. Profiling uses a local monotonic clock so worker timing does not enter
MPI while the serialized communication owner is active.

All new production paths are opt-in. Existing default execution remains
available for reference. CLI switches:

```
--decode-executor legacy|persistent
--router-kernel legacy|fused
--verify-kernel legacy|grouped
--collective-owner legacy|serialized
--moe-combine-kernel legacy|vector|overlap
--index-kernel legacy|heads|keys4
--mla-kernel legacy|registers|values|fp16-cache
```

MTP retains its legacy executor and rejects `--decode-executor persistent`.
Its resident trial mode restores target/MTP state, sweeps draft depths, and
buffers output outside the timed interval. Lookup speculation is currently
exposed through the resident benchmark; it is not a serving API.

## Reproduce the full-model gates and measurements

Run serially in one twelve-node allocation after bounded staging completes.
Use separate frozen baseline and candidate binaries compiled with identical
math flags. Preserve prompt, paging, thread count and collectives between
runs. Never overlap a staging MPI job with these launches.

```bash
export GLM53F_BUILD=0 OMP_NUM_THREADS=47
export XOS_MMM_L_PAGING_POLICY=demand:demand:demand
bash a64fx/glm5/build_glm53f_integrated_12n.sh all
bash a64fx/glm5/run_glm53f_12n.sh check
bash a64fx/glm5/run_glm53f_12n.sh executor-check tmp/prompt8k.ids 32

# Use the frozen baseline GLM53F_BIN_DIR for the first command.
unset XOS_MMM_L_HPAGE_TYPE
bash a64fx/glm5/run_glm53f_12n.sh benchmark tmp/prompt8k.ids tmp/baseline.ids \
  --transitions 256 --repetitions 3 > tmp/baseline.log 2>&1
# Switch GLM53F_BIN_DIR to the candidate binaries.
export XOS_MMM_L_HPAGE_TYPE=none GLM53F_NATIVE_Q8_PANEL=0
export GLM53F_SPARSE_ASYNC=0 GLM53F_KDA_ASYNC=0
bash a64fx/glm5/run_glm53f_12n.sh benchmark tmp/prompt8k.ids tmp/candidate.ids \
  --transitions 256 --repetitions 3 --decode-executor persistent \
  --router-kernel fused --verify-kernel grouped --index-kernel heads \
  --mla-kernel registers --collective-owner serialized \
  --moe-combine-kernel vector > tmp/candidate.log 2>&1
python3 a64fx/glm5/compare_glm53f_runs.py \
  --baseline tmp/baseline.log --candidate tmp/candidate.log \
  --baseline-ids tmp/baseline.ids --candidate-ids tmp/candidate.ids \
  --output tmp/comparison.json
```

The resident benchmark loads once, prewarms, runs an untimed plain trial,
then three timed trials restored from the empty state. Prefill includes all
prompt tokens and the final head. Decode counts **transitions after the
first prompt prediction**; 256 transitions produce 257 recorded IDs.
Speculative timing includes drafting, verification, rollback and fallback.
It rejects insufficient memory headroom and inconsistent tokens across
trials. Logs record binary/prompt/topology hashes and explicit settings.
The comparison tool requires complete trials, valid token accounting,
identical generated IDs, and at least 2 GiB sampled memory headroom.
Promotion requires at least 5% median improvement with no greater than 2%
regression in the other phase. Target attainment is a separate report field.

Lookup options: `--speculation lookup --draft-depth 1..4 --spec-policy
adaptive|always`. Adaptive policy evaluates every sixteen cycles, comparing
maximum rank elapsed time per completed token with the plain warm trial.
For the MTP sweep, stage MTP first, enable
`GLM53F_SPEC_SELF_REFERENCE=1`, then run:

```bash
bash a64fx/glm5/run_glm53f_q4_mtp_12n.sh tmp/prompt8k.ids tmp/mtp.ids 128 4 \
  --repetitions 3 --draft-sweep --ignore-eos --verify-kernel grouped > tmp/mtp.log 2>&1
python3 a64fx/glm5/report_glm53f_spec_runs.py --log tmp/mtp.log \
  --ids-prefix tmp/mtp.ids --cycles 128 --output tmp/mtp-report.json
```

The report requires
the final completion record, one warmup and at least three timed trials per
depth, exact cycle/delivery accounting, every delivered ID matching the
greedy reference, and the memory guard. Use `--ignore-eos` for fixed cycles.

A speculative trial is evidence of greedy equivalence only when the
reference check covers every delivered token. Acceptance rate alone is
insufficient. Validate the serialized communication owner with
`bench_glm53f_async_reduce 300 1 512` before enabling overlap beyond the
2048-token sparse window.

## Evidence so far

Native A64FX measurements, job 52068253, normal 2 GHz, compact 2×3×2:

| Check | Result |
| --- | --- |
| Persistent team, delayed workers and repeated stack contexts | PASS at 1, 12, 47 and 48 threads |
| Grouped Q4/Q5 experts, mixed Q6 fallback, repeated routes and tails | Bit-exact PASS at 1, 12, 47 and 48 threads, IQ modes 0 and 1 |
| Persistent mHC versus legacy, complete scratch and stream output | Bit-exact PASS at 1, 12, 47 and 48 threads |
| Conservative builds of expert/mHC tests | Same thread/mode checks PASS without fast-math |
| Vector MoE combine, all 256 route masks and 63-row ragged tiles | Bit-exact PASS at 1, 12, 47 and 48 threads |
| Eight-head sparse index, 8193 synthetic pool entries | Bit-exact PASS; 47 threads 0.374730 → 0.143590 ms (2.610×); 48 threads 0.366679 → 0.138360 ms (2.650×) |
| Register MLA absorbed query, six heads | Bit-exact PASS at 1/12/47/48 threads in fast and conservative builds; 47 threads 15.439987 → 8.599758 µs (1.795×), conservative 1.785× |
| Warm persistent team, internal timer, 4000 jobs/repeat | 12 threads 1.576126 µs/job; 47 threads 4.595131; 48 threads 4.694641 |
| Full-state executor, 32 saved prompt positions | Hidden outputs and complete attention state bit-exact PASS with persistent/fused/index/MLA paths |
| Serialized MPI/uTofu owner stress | 300 iterations PASS, 515-token ragged slab stream, interleaved MPI sum/byte gather/uTofu sum |
| Launcher contracts | 16 local tests PASS |
| Strict resident MTP reporting | Six local tests PASS, including incomplete/reference/token/accounting failures |
| Lookup rejection/rollback controller | 29 cases PASS locally and on twelve ranks, including late verification-cost growth |

The index result is a kernel microbenchmark, not whole-model acceleration.
Baseline and candidate full-model batch/prefill checks PASS, including
every-prefix continuation in the candidate. At the 2048-token sparse
boundary, scalar-reference MLA and rollback have zero error; batched MLA
relative L2 is 7.55019511e-05 and rollback error is zero. Exact checks pin
projection GEMM and fused front off to isolate MLA from nonidentical
projection accumulation. The initial frozen baseline check failed with
projection GEMM enabled; this was also present before the Strata changes.
The launcher now prints the actual sparse log from inside its subshell.

Staging completed at 00:07 JST. Full-state executor and mixed-transport
stress gates PASS. The stress test initializes its OpenMP compute team
before creating the helper: Fujitsu establishes master affinity on the
first parallel region. Before that, the helper correctly refuses to guess
a spare core from a broad CPU mask.

Completed resident 8K results (8049 prompt IDs, 256 decode transitions,
one warm trial plus three timed trials, 47 threads):

| Path | Decode median (min–max), tok/s | Prefill median (min–max), tok/s | Baseline IDs |
| --- | --- | --- | --- |
| Frozen a90f7972 | 29.989745 (29.853840–29.994162) | 303.243600 (302.496074–303.640411) | Reference |
| Persistent executor + fused router | 30.764120 (30.440834–30.774790) | 303.341836 (302.805230–303.411221) | All 257 identical |

Additional exact 8K comparisons:

| Path | Decode median, tok/s | Prefill median, tok/s |
| --- | --- | --- |
| Persistent + fused + index heads + vector combine | 30.713757 | 321.368333 |
| Above + register MLA | 30.894098 | 318.750066 |
| Above + `XOS_MMM_L_HPAGE_TYPE=none` | 33.066761 | 327.644514 |

The page control is the best balanced 47-thread exact result: +10.26% decode and
+8.05% prefill against the frozen baseline. Q8 panels fail baseline-ID
comparison and remain rejected. Combined sparse/MoE overlap also fails
IDs (first divergence at position 18); its throughput does not qualify.
Sparse overlap's MPI index detour changed reduction order. The corrected
serialized-owner path keeps index sums on their original uTofu reduction;
its native regular-vs-async boundary check is bit-exact, with zero rollback
error. The owner stress also passes 300 iterations. Corrected sparse-only
async matches all 257 IDs: 30.879418 decode / 325.856357 prefill tok/s.
With page type `none`, it reaches 32.826162 / 335.438950, versus the simpler
page candidate's 33.066761 / 327.644514. This is a 2.38% prefill improvement
and a 0.73% decode regression, below the 5% promotion threshold. MoE overlap
still changes the baseline large MPI sum to uTofu slabs and is excluded.

The 48-thread page candidate passes the same stream at 33.402963 decode /
317.928989 prefill tok/s. Its own 48-thread frozen baseline with page type
`none` reaches 31.326396 / 302.705849 (+6.63% decode, +5.03% prefill).
Versus 47 threads, decode gains only 1.02% while prefill loses 2.97%; 47
threads remain selected. The 48-thread prefill range is 300.170450–320.935217.

Lookup depths 1–4 all match IDs but original decode medians are 30.305971,
30.247893, 29.430768 and 26.709574 tok/s. The one-shot adaptive check missed
later expensive verification. Recurring 16-cycle cost checks pass the
new delayed-cost regression; the old implementation fails it. All 29
controller cases pass natively. The revised depth-4 median is 30.549583
(30.485636–30.557937), 14.38% faster than the old policy, but below plain
decode. Lookup remains opt-in.

The complete MTP depth sweep passes every delivered ID against its own
plain greedy reference, with 128 cycles, one warmup and three timed trials
per depth. The reference delivers 768 tokens at 30.685 tok/s:

| MTP depth | Delivered tokens/trial | Acceptance | Decode median (min–max), tok/s |
| --- | --- | --- | --- |
| 1 | 352 | 96/128 | 28.988085 (28.987217–28.988481) |
| 2 | 403 | 147/256 | 26.196257 (26.194406–26.196963) |
| 3 | 419 | 163/384 | 22.001181 (22.000102–22.001270) |
| 4 | 419 | 163/512 | 17.897930 (17.896856–17.899092) |

MTP uses scalar prefill through 8048 positions and retains the last prompt
ID as its next target input. The resident benchmark uses batched prefill
through all 8049 positions. These recipes produce different greedy streams
(first observed difference at prediction 14). The MTP equivalence claim
uses its scalar-prefill reference, not the batched-prefill baseline. The
best MTP depth is slower than its own reference, so it is not promoted.
Minimum MTP memory headroom is above 10.3 GiB.

The selected page candidate passes the 128-ID short-prompt comparison:
frozen baseline 36.575832 decode / 267.477754 prefill tok/s; candidate
40.431636 (40.362345–40.440925) / 282.637846 (282.577122–282.934887).
All 257 IDs match, improving decode 10.54% and prefill 5.67%.

The synthetic long-context fixture repeats the saved 8049 IDs four times
(32196 positions). Its first baseline attempt failed during prefill because
the benchmark reserved only 32×4096 floats for collectives. Packed sparse
index scores require up to 32×floor(context/4) floats. The harness now
reserves the larger count while preserving the original reservation through
16K. Frozen baseline kernels and candidate kernels receive the identical
harness fix (commit `071fdcf3`). Corrected qualification passes all 257 IDs:

| Synthetic 32196-token context | Decode median (min–max), tok/s | Prefill median (min–max), tok/s |
| --- | --- | --- |
| Frozen baseline | 27.747994 (27.739213–27.766317) | 265.965590 (265.657743–266.020143) |
| Candidate, page type none | 31.450699 (30.927964–31.469004) | 287.156697 (286.600822–287.428617) |

This improves decode 13.34% and prefill 7.97%. Minimum sampled headroom is
10656768 KiB for baseline and 10699520 KiB for candidate, both above 10 GiB.
The locally recomputed report is byte-identical to the remote report.

All measured model configurations retain 45 layers and top-8 routing.
Neither 100 decode nor 2000 prefill tok/s has been reached. Historical rates
were roughly 29–30 decode and 285–301 prefill tok/s at 8K; the frozen
three-trial baseline above supersedes those historical measurements.
Reports, logs and token outputs are retained locally in
`tmp/glm53f-strata-evidence-20261001/` and remotely in `tmp/strata-v10b/`.
Committed comparison reports and binary/prompt/topology metadata are in
`strata-validation-20261002.json`. The 8K page trial has owner enabled but
inactive because async is off; short/32K trials have owner off.

## Remaining architecture decision

The measured TP12 candidate remains below both targets. The next architecture
evaluation is PP3×TP4 using newly staged TP4 expert parts, explicit layer ownership, stage-local collectives,
and full four-stream handoffs. Current stage images and runtime assume
twelve ranks; a PP configuration cannot be selected by changing a launch
flag. PP3×TP4 is not implemented or qualified. For a single dependent decode
request, account for inter-stage latency rather than assuming threefold
pipeline speedup. Short and synthetic 32K qualification pass. All new paths
remain opt-in; the synthetic fixture does not replace a real long coding prompt.
