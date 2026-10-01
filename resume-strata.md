# Resume: GLM53F Strata-inspired optimization, 12 A64FX nodes

Updated 2026-10-01 23:20 JST. Implementation is experimental and opt-in.
Targets: complete 45-layer UD-Q4_K_XL/top-8, saved ~8K single request,
100+ delivered decode tokens/s and 2000+ prefill tokens/s. Neither target
has been demonstrated. See `a64fx/glm5/GLM53F_STRATA.md` for code map,
validated kernel results, flags and exact measurement commands.

## Live allocation and isolated deployment

- PJM job **52068253**, twelve nodes, compact 2×3×2, normal 2 GHz, eco 0,
  six-hour allocation started about 20:40 JST; compute node `d26-2014c`.
- SSH endpoint `fugaku1`, login1 account u14346.
- Isolated snapshot `$HOME/work/gemm/glm53f-strata-20261001`.
  Leave the original remote `$HOME/work/gemm/glm53f` checkout untouched.
- Local tmux session `glm53f-strata`; bridge state and helper in
  `tmp/bash-http-glm53f-strata/`. Local HTTP port 42446 → reverse port
  32446 → compute port 21264. Helper reads a Bash program from stdin:
  `python3 tmp/bash-http-glm53f-strata/remote.py`.
- Four-node job 52067029 belongs to other work; leave it alone.

## Running jobs

**Do not launch another mpiexec while staging PID 719 is alive.**

- Staging PID **719**, log `tmp/stage.log`, rank logs
  `tmp/glm53f-q4-52068253/routed-stage-719.1.{rank}`.
  At 23:20 JST rank 0 completed layer 40 (~14.0 GB), out of 45. Shared filesystem staging is slow.
  Wait for `SENTINEL glm53f_stage_12n=OK`; do not restart the partial blob.
  Native dense/sparse/KDA/shared/core staging follows routed staging.
- Full candidate-v7 build PID **6483** completed PASS, `tmp/build-strata-v7.log`.
  Objects `/local/glm53f-strata-v7-52068253`, shared binaries
  `a64fx/glm5/build/candidate-v7`.
- Timer-safe incremental candidate-v8 build PID **6572** completed PASS,
  `tmp/build-strata-v8.log`, script `tmp/glm53f-strata-build-v8.sh`.
  It waited for v7, then extracted its archive and compiled changed files.
  Objects `/local/glm53f-strata-v8-52068253`, shared binaries
  `a64fx/glm5/build/candidate-v8`. Unchanged tools symlink to v7.
- Candidate-v9 MTP completion/report build PID **7842** completed PASS,
  `tmp/build-strata-v9.log`. Core binaries symlink to v8; MTP main is relinked
  with a final completion record. Binaries `a64fx/glm5/build/candidate-v9`.
- Waiting campaign PID **7846**, script `tmp/glm53f-strata-campaign-v9.sh`,
  log `tmp/campaign-strata-v9.log`. It waits for staging and v9 build,
  then runs serially: frozen baseline gates and resident 8K trials;
  candidate gates and full-state executor comparison; persistent 8K;
  300-iteration mixed MPI/uTofu owner stress; overlap 8K; adaptive lookup
  depths 1–4; index/vector-combine 8K; MTP staging and resident depths 1–4.
  Outputs `tmp/strata-v9/`, including a strict reference-checked MTP report. Every baseline/candidate benchmark has one warm
  trial and three timed trials, 256 decode transitions, strict generated IDs.
  `set -e` stops at the first failed gate; inspect logs and fix failures.
- Earlier waiting campaigns v5/v6/v7/v8 were canceled before MPI started.
  Never modify a running build script or source while its compiler reads it.

## Frozen reference and archives

Baseline source revision **a90f7972**. Frozen binaries
`a64fx/glm5/build/baseline`, objects `/local/glm53f-build-52068253`.
Frozen original check script `a64fx/glm5/run_glm53f_baseline_check.sh`.

Local archive SHA256:

- Baseline `tmp/glm53f-strata-base.tar`:
  `d9940e19d7324cbd26ef1357b3a9760c0f113dea23cc327bc8ff41d387968168`.
- Full v7 `tmp/glm53f-strata-v7.tar`:
  `cc0fbcb4f2ec2a1dbe41d54a1b5ac6bf13dfe7e03a4d64d463a5fc2b13c05f7a`.
- Incremental v8 `tmp/glm53f-strata-v8-timers.tar`:
  `02322782e6542a320c952950597d767cb5d1dfd8f20ceff31bf66068e9bf3381`.
  Includes target/head/lookup/benchmark/prefill/spec timers and the
  fast-math-safe finite checks in the executor diagnostic.

Saved remote prompt `tmp/prompt8k.ids`: **8049 IDs**. Historical prefill
reported 8048 positions because it excluded the final scalar position;
new resident benchmark includes all 8049 and the final vocabulary head.
Also `tmp/prompt4k.ids`. Short and derived 32K qualification remain pending.

## Evidence and outstanding work

Retrieved native test logs live locally in
`tmp/glm53f-strata-evidence-20261001/`. Persistent expert/mHC tests pass
bit-exact at 1/12/47/48 threads in fast and conservative builds. Vector MoE
combine passes all 256 route masks and ragged tails. Sparse index microbench
passes bit-exact and is ~2.6× faster at 47/48 threads. Warm pool dispatch
is ~4.6 µs/job at 47 threads, measured internally, excluding ELF startup.
Local launcher 16 tests, strict MTP reporter six tests, and lookup controller
28 mock-state cases pass. Implementation commit **86fd586f**; all 311 deployed
source fingerprints matched before the MTP completion/report addition.

Full-model gates, owner stress, throughput and MTP results are **pending**.
Do not promote defaults or claim the target from kernel timings. Verify all
accepted-prefix states, generated-ID comparisons and sampled memory before
interpreting the campaign. PP3×TP4 is approved for evaluation if TP12 remains
insufficient, but is not implemented; it needs new TP4 stage images,
stage-local communicators, explicit layer ranges and four-stream handoffs.

No push is authorized. Preserve unrelated untracked q38fn and tmp artifacts.
