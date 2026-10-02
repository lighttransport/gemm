# Strata-inspired GLM53F optimization: implementation and validation

Updated 2026-10-02. Targets are **100 delivered decode tokens/s and 2000
prefill tokens/s**, on twelve A64FX nodes, the complete 45-layer
UD-Q4_K_XL model, top-8 routing, and the saved roughly 8K coding prompt.
These targets have **not been demonstrated** by the changes below.

## October 2 qualification (complete)

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

PJM 52075759 completed bounded staging at 06:30 JST. Uncontended native
units and byte-exact derived-cache prefill/rollback checks pass; the
128-position full-model executor comparison also passes bit-exactly.
Initial same-allocation 8K cache-heads medians are 34.079982 decode and
345.582371 prefill tok/s versus 32.731477 / 329.018991 for the frozen
control (+4.1% / +5.0%), with complete output IDs exact. Independent
five-trial confirmation also passes: 33.939857 / 346.234747 versus
32.610212 / 327.987043 (+4.08% / +5.56%). Full1024-transition
stress, short128 and repeated32K complete-ID qualification also passes.
The cache-heads configuration qualifies for promotion. Synthetic32K
medians are 32.385736 / 316.830563 versus 31.326356 / 286.622911
(+3.4% / +10.5%); short128 rates are essentially unchanged. The campaign
compared the previously qualified page-none/47-thread configuration, a
rebuilt control, independent index/value ablations, and both cache/index
combinations. Complete output IDs, sampled memory headroom, independent
five-trial confirmation, 1024-transition decode, short128, and synthetic
32K are required before promotion. A complete-state legacy/persistent
checker can exercise the new kernels using diagnostic
`GLM53F_EXECUTOR_INDEX_KERNEL=2 GLM53F_EXECUTOR_MLA_KERNEL=3`; it frees the
reference before loading a fresh cache-enabled candidate.
Replicated decode scoring removes the score allreduce but yielded no
whole-model improvement. Both `replicated-heads` and `replicated-keys4`
passed strict sparse gates and complete 257-ID comparisons; decode ratios
were 0.996676 and 0.997061 against cache-heads. They remain experimental.

### Selected selector/panel configuration

`--pool-selector partition4k` uses bounded ID partitioning with the exact
original descending-score/ascending-ID order, sorts only the selected 512
entries, and reuses dead reduction scratch. NaNs, bad pivots and more than
4096 pools retain the heap. Default selection remains `heap`; default
attention width remains 32. Native fast/conservative 120-case oracle/bounds
checks, local ASan/UBSan, and native strict prefill/decode/rollback gates pass.

The selected configuration adds a 47-token attention panel to the qualified
cache-heads runtime: build `build_glm53f_integrated_12n.sh check 47`, then use
`--index-kernel heads --mla-kernel fp16-cache --pool-selector partition4k`
alongside persistent decode, fused router, grouped verification and vector
MoE combine. Keep 47 threads, page type `none`, Q8 panel 0, sparse async 0 and
KDA async 1. All 45 layers and top-8 routing remain active.

Same-allocation ablations reject replicated decode scoring, confirm the
rebuilt 32/heap control is unchanged, and select 47 over 64. Independent
confirmation and every context check complete with exact output IDs.
The control below is the immutable qualified cache-v3 binary; candidate is
candidate-panel47-v1. Every rate is a median of complete rank-max trials.

| Workload | Decode control → candidate (tok/s) | Prefill control → candidate (tok/s) | Decode gain | Prefill gain |
| --- | --- | --- | --- | --- |
| 8K, 256 transitions, 5 trials | 33.793083 → 35.742137 | 345.065496 → 370.754010 | +5.77% | +7.44% |
| 8K, 1024 transitions, 3 trials | 34.040300 → 35.746584 | 344.113402 → 371.903716 | +5.01% | +8.08% |
| Short 128, 256 transitions, 3 trials | 40.436245 → 40.660046 | 282.005617 → 285.918131 | +0.55% | +1.39% |
| Synthetic 32196, 256 transitions, 3 trials | 32.275942 → 32.185018 | 316.484767 → 341.961472 | -0.28% | +8.05% |

All 257 IDs match for 256 transitions and all 1025 match for stress1024.
Minimum sampled memory headroom across these runs is **9.831 GiB**.
32K uses the 8K prompt repeated four times; it is synthetic context evidence.
Targets 100/2000 remain unmet. All new settings stay opt-in.

Strict native panel gates compare 32 against 47/64 over 91 positions at
warm 2046/8049, including rollback; selector gates cover warm 2046/2051
prefill 32 and warm 8049 decode 4. All comparisons require memcmp equality
on every rank. The benchmark records its compiled panel width and sizes
collective reservations for both output and packed-score payloads.
The normal generation/prefill entrypoints share the same bounded integer
reservation helper. Native normal generation with 8049 prompt IDs matches
all 32 generated IDs between panels 32 and 47; local capacity/overflow tests
pass for 32/47/48/64. The normal runner retains its existing scalar final
prompt token; its gate is separate from resident benchmark throughput.

Implementation commit `c461079c` matches the frozen runtime source archives.
The committed [kernel qualification record](strata-kernel-validation-20261002.json)
contains comparisons, settings, binary/source/evidence hashes and rejected
experiments. See [resume](../../resume-strata.md) for the allocation and
immutable remote build paths. A subsequent Q8 campaign is active; see the continuation below and `resume-strata.md`.

Rank-zero diagnostic medians identify the remaining work: decode KDA 6.405,
mHC 5.764 and MoE 7.582 ms/token; prefill MoE stays near 1.02 ms/token. Sparse
prefill falls 0.966→0.760 ms/token and decode index 3.459→2.108 ms/token.
These nested component timers overlap and are separate from the rank-max
throughput measurements above.

## Q8 continuation (October 2 afternoon, pending full-run qualification)

PJM 52085859 provides another six-hour 12-node allocation, approximately
12:17–18:17 JST. Bounded restaging and the native build passed. Best qualified
throughput remains **35.742137 decode /370.754010 prefill tok/s**.

The candidates add exact assembly 4×4 and 2×8 row/position tiles, an eight-row
C decode kernel, and a fused scheduler for independent MLA head projections.
The assembly keeps the original per-64-column integer dot, scale product,
FP32 FMA, and final lane reduction order. It avoids the compiler's many
accumulator spills without changing weight/activation formats. Runtime
switches are opt-in: `--q8-row-kernel rows8`,
`--q8-prefill-kernel tile4x4-asm|tile2x8-asm`, and
`--mla-projection-kernel fused`. Original kernels remain the defaults.

Native assembly arithmetic passed 32 configurations (fast/conservative FCC,
1/12/47/48 threads and four tile modes), 126 cases per configuration with
mixed formats and tails. Staging-time diagnostics show encouraging ASM4
projection times, but are **not performance qualification**. A no-inline-only
experiment was slower and rejected. The earlier decode probe inherited page
settings and lacked NUMA interleave, so it cannot select a production kernel.

The immutable `candidate-q8-v6` integrated build passed all 64 configurations
(fast/conservative ×1/12/47/48 threads ×2 row modes ×4 tile modes), including
independently prepared head inputs. Its Q8 and MLA kernels match the committed
3529192d implementation; later MTP/mHC changes have separate frozen archives.
The six-head decode projection probe is byte-exact and measures 19.893→6.001 µs
for rows4 separate→fused during staging; it needs an uncontended repeat.
The record is [native Q8 evidence](strata-q8-native-20261002.json).
A serial post-stage campaign is queued for uncontended probes, sparse
prefill/decode/rollback, complete-state
comparison and full 8K/short/32K output and throughput gates. Local evidence
is `tmp/strata-q8-20261002/`; current progress and artifact hashes are in
`native-progress.json`. `resume-strata.md` records live PIDs and source paths.
Neither target has been met; no new runtime setting has been promoted.

All seven 8K Q8 ablations return the exact 257-ID stream. Independent five-trial
confirmation of ASM4 plus fused MLA projections measures **35.540871 decode /
377.077157 prefill tok/s**, against **35.609336 /371.931478** for the fresh frozen
control: −0.19% decode /+1.38% prefill. This misses the 5% promotion threshold.
Stress1024, short128 and repeated32K also match all IDs; the complete
campaign passes at15:48. Repeated32K measures32.101995→31.984868 decode and
343.931969→345.949648 prefill tok/s, below the promotion threshold.
Completed reports are recorded in [Q8 full-model progress](strata-q8-full-20261002.json).

The next experiment increases outer prefill workspace capacity to 4096 while
keeping 47-position attention and all arithmetic unchanged. Larger route
cohorts reuse expert expansion across more positions. The default remains
512; the third build argument bounds capacity and `--prefill-chunk` selects
512/1024/2048/4096 during trials. Native capacity/checker compilation and
16 local capacity/panel configurations pass. The queued serial campaign
first requires exact final hidden streams and complete KDA/sparse state
against 512, then full IDs, fresh controls, confirmation and context stress.
No larger-chunk throughput or memory-headroom result exists yet.

## Batched-prefill MTP candidate (October 2)

The resident benchmark now accepts `--speculation mtp` with explicit
`--mtp-routed-stage` and `--mtp-shared-stage` paths. It captures every prompt
position's stream mean using the same batched prefill recipe, applies the
output norm, and teacher-forces the draft cache with the next prompt token.
The first decode parent is copied from the actual target head's normalized
hidden. Chained proposals use the actual MTP shared-head normalized hidden.
This follows the post-norm target export in the
[GLM5-Next graph](https://github.com/ggml-org/llama.cpp/blob/master/src/models/glm5-next.cpp)
and the normalized parent/draft hiddens in local Strata `glm_decode.cpp`.
The historical scalar-prefill MTP runner's raw-hidden contract is preserved.

Each verification window processes the already predicted input together with
its drafts. It restores the accepted target snapshot, retains the first exact
MTP pair, and replays accepted draft inputs with actual verified parent
hiddens. The bonus input is left for the next window. A recurring rank-wide
cost check can switch to a persistent plain suffix and clears the stale draft
cache. Draft/verify work remains outside the persistent target team; those
kernels are not all aware of persistent dispatch.

Prefill timing includes hidden capture, normalization and MTP teacher forcing.
Decode timing includes draft, verify, restore and replay. The benchmark counts
only delivered target transitions, compares every ID with a plain warmup on
its actual batched prompt stream, and can additionally compare complete
KDA/sparse endpoint state with `--decode-state-check TRACE_PREFIX` outside the
timed interval. The prompt checker adds `--capture-hidden` and an optional
`--reference-chunk N` to check every captured position against the original
prefill call at the selected chunk.

Local controller tests pass **896** cases: all acceptance/rejection prefixes,
depths 1–4, ragged delivery, context-dependent target rollback, approximate
draft hiddens, normalized-hidden selection, teacher forcing, observer bounds
and adaptive fallback. Address/undefined-behavior sanitizers pass with leak
detection disabled because LeakSanitizer cannot run under this environment's
tracing. Native output-norm/export tests pass **eight** fast/conservative ×
1/12/47/48-thread configurations, 18 cases each through capacity4096. The
integrated normalized MTP build passes. Evidence is
[strata MTP native record](strata-mtp-native-20261002.json).

Immutable native artifacts are `candidate-mtp-v3` plus the diagnostic-only
checker update in `candidate-mtp-v4`. A serial queue waits for Q8, capacity
and lookup qualification, then stages only checkpoint layer45 and runs
prompt-hidden, cache replay, short/8K full-state and timed depth gates.
There is **no new full-model MTP throughput or promotion result yet**.
`tmp/strata-mtp-20261002/` contains frozen source hashes, build logs and
campaign scripts; `resume-strata.md` records the current queue state.

## mHC synchronization candidate (October 2)

`--mhc-kernel fused-sync` removes the barrier between mixing logits and their
coefficient calculation. Each worker retains the original BF16 dot chain and
contiguous row allocation. An acquire/release completion counter transfers the
completed logits to their last owner, which computes sigmoid/Sinkhorn before
the remaining publication barrier. Norm partitions, FP64 post accumulation,
collapse, and normalized-input publication remain unchanged. The default is
`legacy`; this candidate applies to the validated `GLM53F_MHC_FAST=1` path.

Native arithmetic passes 14 fast/conservative ×1/3/12/23/24/47/48-thread
configurations, each with 384 chained calls. Every stream endpoint, complete
scratch and published normalized input matches byte-for-byte across ordinary
OpenMP and persistent teams. The prior team test also passes all 14 settings.
Tests require finite stream and normalized values.

A separate one-node PJM **52089999** ran these gates and a 90-site probe without
contending with the 12-node campaign. At 47 threads with fast math and the
persistent executor, six-trial medians are **47.444470→46.666629 µs/call**
(+1.67%). Synthetic weight placement differs from the production model;
these numbers do not establish full-model throughput. Requested `FLIB_BARRIER=HARD`
overrides `OMP_PROC_BIND`; unsupported thread counts use the runtime's software
barrier fallback. Full evidence and hashes are in
[mHC native record](strata-mhc-native-20261002.json).

FCC cross compilation on the login node avoids compute contention. Immutable
full-model artifacts are `candidate-mhc-sync-v4`, capacity4096 /attention47,
from `tmp/strata-mhc-sync-20261002/source-v3.tar.gz`. The full-model queue waits
for existing Q8/capacity/lookup/MTP work, then requires zero hidden bit mismatches
and complete KDA/sparse state before timing. Fresh frozen/rebuilt controls,
independent confirmation and qualifying-candidate context stress follow.
Allocation guards defer incomplete work; no full-model mHC result exists yet.

Reproduce the arithmetic and probe after building the integrated `check` tools:

```bash
OMP_NUM_THREADS=47 FLIB_BARRIER=HARD ./test_glm53f_mhc_sync
fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp \
    -ffast-math -fno-math-errno -Wall -Wextra -I. -I../../common \
    bench_glm53f_mhc_sync.c glm53f_team.c -lm -lpthread -o bench_glm53f_mhc_sync
OMP_NUM_THREADS=47 OMP_PROC_BIND=close OMP_PLACES=cores OMP_WAIT_POLICY=active \
    FLIB_BARRIER=HARD XOS_MMM_L_HPAGE_TYPE=none \
    XOS_MMM_L_PAGING_POLICY=demand:demand:demand ./bench_glm53f_mhc_sync
```

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
--index-kernel legacy|heads|keys4|replicated-heads|replicated-keys4
--mla-kernel legacy|registers|values|fp16-cache
--pool-selector heap|partition4k
```

The historical `glm53f_spec_decode_12n` MTP runner retains its legacy executor
and rejects `--decode-executor persistent`. Its resident trial mode restores target/MTP state, sweeps draft depths, and
buffers output outside the timed interval. Lookup speculation is currently
exposed through the resident benchmark; it is not a serving API.

## Reproduce the full-model gates and measurements

Run serially in one twelve-node allocation after bounded staging completes.
Use separate control and candidate binaries compiled with identical math
flags. The completed campaign also checks the rebuilt 32/heap control against
the immutable cache-v3 binary before comparing the selector/panels. Preserve
prompt, paging, thread count and collectives between runs. Never overlap a staging MPI job with these launches.

```bash
unset OPAL_PREFIX OMPI_CC OMPI_CXX
export GLM53F_MPICC=mpifcc
export GLM53F_BUILD=0 OMP_NUM_THREADS=47
export OMP_PROC_BIND=close OMP_PLACES=cores OMP_WAIT_POLICY=active FLIB_BARRIER=HARD
export GLM53F_PREWARM=1 GLM53F_PROFILE=1
export GLM53F_FAST_MATH=1 GLM53F_NO_MATH_ERRNO=1
export XOS_MMM_L_PAGING_POLICY=demand:demand:demand XOS_MMM_L_HPAGE_TYPE=none
export GLM53F_NATIVE_Q8_PANEL=0 GLM53F_COMM_OWNER=0
export GLM53F_SPARSE_ASYNC=0 GLM53F_KDA_ASYNC=1
export GLM53F_BIN_DIR="$PWD/tmp/glm53f-panel32-bin"
bash a64fx/glm5/build_glm53f_integrated_12n.sh check 32
export GLM53F_BIN_DIR="$PWD/tmp/glm53f-panel47-bin"
bash a64fx/glm5/build_glm53f_integrated_12n.sh check 47

# Sequential runs; use unique output names for each independent repeat.
export GLM53F_BIN_DIR="$PWD/tmp/glm53f-panel32-bin"
bash a64fx/glm5/run_glm53f_12n.sh benchmark tmp/prompt8k.ids tmp/control.ids \
  --transitions 256 --repetitions 5 --decode-executor persistent \
  --router-kernel fused --verify-kernel grouped --index-kernel heads \
  --mla-kernel fp16-cache --moe-combine-kernel vector --pool-selector heap \
  > tmp/control.log 2>&1
export GLM53F_BIN_DIR="$PWD/tmp/glm53f-panel47-bin"
bash a64fx/glm5/run_glm53f_12n.sh benchmark tmp/prompt8k.ids tmp/candidate.ids \
  --transitions 256 --repetitions 5 --decode-executor persistent \
  --router-kernel fused --verify-kernel grouped --index-kernel heads \
  --mla-kernel fp16-cache --moe-combine-kernel vector --pool-selector partition4k \
  > tmp/candidate.log 2>&1
python3 a64fx/glm5/compare_glm53f_runs.py \
  --baseline tmp/control.log --candidate tmp/candidate.log \
  --baseline-ids tmp/control.ids --candidate-ids tmp/candidate.ids \
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
