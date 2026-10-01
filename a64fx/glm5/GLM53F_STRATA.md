# Strata-inspired GLM53F optimization: implementation and validation

Updated 2026-10-01. Targets are **100 delivered decode tokens/s and 2000
prefill tokens/s**, on twelve A64FX nodes, the complete 45-layer
UD-Q4_K_XL model, top-8 routing, and the saved roughly 8K coding prompt.
These targets have **not been demonstrated** by the changes below.

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
--index-kernel legacy|heads
--mla-kernel legacy|registers
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
bash a64fx/glm5/run_glm53f_12n.sh benchmark tmp/prompt8k.ids tmp/baseline.ids \
  --transitions 256 --repetitions 3 > tmp/baseline.log 2>&1
# Switch GLM53F_BIN_DIR to the candidate binaries.
bash a64fx/glm5/run_glm53f_12n.sh benchmark tmp/prompt8k.ids tmp/candidate.ids \
  --transitions 256 --repetitions 3 --decode-executor persistent \
  --router-kernel fused --verify-kernel grouped --index-kernel heads \
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
adaptive|always`. Adaptive policy evaluates after sixteen cycles, comparing
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
| Launcher contracts | 16 local tests PASS |
| Strict resident MTP reporting | Six local tests PASS, including incomplete/reference/token/accounting failures |
| Lookup rejection/rollback controller | 28 cases PASS with a local serial MPI shim and native twelve-rank MPI |

The index result is a kernel microbenchmark, not whole-model acceleration.
Baseline and candidate full-model batch/prefill checks PASS, including
every-prefix continuation in the candidate. At the 2048-token sparse
boundary, scalar-reference MLA and rollback have zero error; batched MLA
relative L2 is 7.55019511e-05 and rollback error is zero. Exact checks pin
projection GEMM and fused front off to isolate MLA from nonidentical
projection accumulation. The initial frozen baseline check failed with
projection GEMM enabled; this was also present before the Strata changes.
The launcher now prints the actual sparse log from inside its subshell.

Full-state executor, mixed-transport stress, resident 8K throughput and MTP
depth sweep are running serially after staging completed at 00:07 JST.
Historical full-run rates were roughly 29–30 decode tokens/s and 285–301
prefill tokens/s at 8K. Those are earlier measurements, not candidate results.

The queue also compares `XOS_MMM_L_HPAGE_TYPE=none` with the original page
policy, then the existing `GLM53F_NATIVE_Q8_PANEL=1` path with the row layout
under that same page policy. The panel can change floating-point reduction
order; promotion still requires identical generated IDs and no material
prefill regression. These controls are separate from the register MLA kernel.

## Remaining architecture decision

First measure the TP12 candidate with exact generated-ID checks and complete
phase profiles. If TP12 remains insufficient, evaluate PP3×TP4 using newly
staged TP4 expert parts, explicit layer ownership, stage-local collectives,
and full four-stream handoffs. Current stage images and runtime assume
twelve ranks; a PP configuration cannot be selected by changing a launch
flag. PP3×TP4 is not implemented or qualified. For a single dependent decode
request, account for inter-stage latency rather than assuming threefold
pipeline speedup. Short prompts, derived 32K prompts, and memory headroom
also require qualification before promoting defaults.
