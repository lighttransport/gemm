# Resume: GLM53F Strata-inspired optimization, 12 A64FX nodes

Updated 2026-10-02 18:46 JST. Capacity4096 remains the promoted recipe:
35.462134 decode /412.273634 prefill tok/s. Neither 100/2000 target is met.
The normalized MTP constructor failure is diagnosed and fixed: compact-core
strictness was cached from target creation and ignored temporary checkpoint
fallback for layer45. Local regression, old-header counterfactual, sanitizers,
18 launcher tests and isolated A64FX cross-build pass. Fresh PJM52097252 is
restaging; corrected serial qualification PID597 follows stage PID134.
No new acceptance or full-model MTP throughput result is available yet.

Previous campaign: TP12 implementation, native validation and
short/8K/synthetic 32K qualification complete. All new paths remain opt-in.
Targets: complete 45-layer UD-Q4_K_XL/top-8, saved ~8K single
request, 100+ delivered decode and 2000+ prefill tok/s. Neither target met.
See `a64fx/glm5/GLM53F_STRATA.md` for implementation, gates and commands.

## Active continuation (October 2, afternoon)

**Current loader-policy fix and queue.** `common/glm53f_safetensors.h` caches
only the core image and reads `GLM53F_REPACK_REQUIRE` per request. The preserved
rank4 log exposes the missing `model.language_model.layers.45.enorm.weight`
entry before the constructor failure. Strict target loading initializes the
cache; MTP then temporarily allows layer45 checkpoint reads and restores
strictness. The old reader retained strictness throughout. Regression fixture
uses distinct target core bytes and draft checkpoint bytes, row and column
reads, both initialization policies and repeated strict/optional transitions:
40 exact checks pass; both old-header runs fail. ASan/UBSan and18 launcher
checks pass. See [loader evidence](a64fx/glm5/strata-repack-policy-20261002.json).

Immutable **candidate-mtp-v8** rebuilds the shared reader implementation in
KDA and relinks the diagnostic benchmark/checker, normal runner and MTP cache
checker, preserving other math objects. Cross-build sentinel passes. Benchmark
SHA `35a1777695f8926c4c9f897859bcfa25cd7f6e1c6a41e3419f068854a762b24a`,
checker `c48d409f5459b195dc82bfab7c9361e676c52e6c3cdaa2836481edfee941d3ce`.
Frozen update-v1 SHA
`37760e30b8d2bbef0b3aafebd699e0d5b5127c5f2293b29eb3a937be255fb79c`.

Fresh **PJM52097252**,12 nodes compact2×3×2, normal2GHz/eco0, starts18:18
and expires00:18 JST October3. Bridge host`c30-7008c`,42446→32446→21264;
local tmux`glm53f-strata-next4`. Same isolated remote checkout and helper.
**Stage PID134** owns `tmp/strata-next-20261002-1818/stage.sh`; its PID file
contains `STRATA_NEXT4_STAGE_PID=134`. Driver requires routed/native/embed/
head/core/shared and layer45 sentinels before native validation. Fresh stage
paths have suffix52097252;52085859 paths are expired. Do not overlap MPI.

**Qualification PID597**, `tmp/strata-repack-20261002/driver.{sh,pid,log}`,
waits for staging then runs native fast/conservative policy checks, controller/
cache gates, corrected8049 prompt means, short128/8K MTP fullstate and fresh/
rebuilt controls. Depths1–4 adaptive/always count only delivered transitions
and include teacher forcing in prefill. Independent five-trial confirmation
precedes context stress for candidates meeting1.05 decode/.98 prefill ratios.
Independent mHC state/timing qualification follows even if MTP rejects. New
outputs are `tmp/strata-repack-20261002/results/` and
`tmp/strata-mhc-sync-20261002/results-next4/`; new-run guard00:00 JST October3.
The first queued PID544 exited before MPI because its PID guard parsed the
labelled staging PID file as a number; repaired PID597 is the current queue.

The following earlier allocation records are retained as historical evidence.

The promoted recipe compiles capacity 4096 and attention panel 47, then selects
`--prefill-chunk 4096`. Keep 47 threads, page `none`, Q8 panel 0, sparse async 0,
persistent decode, fused router, grouped verification, vector MoE, index
`heads`, MLA `fp16-cache`, pool `partition4k`, Q8 rows4/Ctile4x4 and legacy
MLA projection. Source defaults remain 512. Independent five-trial confirmation
is **35.462134 decode /412.273634 prefill tok/s** versus **35.141258 /372.683205**:
+0.91% decode and +10.62% prefill. All endpoint gates (512/1024/2048/4096),
complete 257-ID comparisons, stress1024 (1025 IDs), short128 and repeated32196
pass. `CAPACITY_CAMPAIGN_PASS` and promote=true are observed. See
[capacity qualification](a64fx/glm5/strata-capacity-full-20261002.json).

Immutable capacity artifacts are **candidate-capacity4096-v1**. Benchmark SHA
`938d7b7c7a41aa6c2b9c2a51aecee897a2b423430b4eaf0de4ccddc335db4ec5`, runner
`a7cc2018557611dabd4e4cc55c42073486aac5e2be93f138ee817ff98584b14f`, checker
`dc6c19d9ee74d9a2fb9eb9f0279460fb11c69104031a8c514042bf86577eaab4`.
Sources/build logs remain in `tmp/strata-q8-20261002/` locally and remotely.

Q8 qualification completed at 15:48: all seven ablations and context gates are
exact, but ASM4/fused confirmation gains only 1.38% prefill and −0.19% decode.
No Q8 setting was promoted. The lookup sweep completed at 17:24 using qualified
capacity4096. Depths 1–4 adaptive and depth4 always all match 257 IDs; decode
ratios 0.999623/0.999351/0.988769/0.990379/0.860653 reject promotion. See
[Q8 full record](a64fx/glm5/strata-q8-full-20261002.json) and
[lookup full record](a64fx/glm5/strata-lookup-full-20261002.json).

Three native probes were rejected and their production prototypes removed:

- Three-token register-FP GEMM: all 2400 arithmetic/guard cases pass, but all
  47/48-thread timing shapes regress. Separate PJM52093052. See
  [GEMM record](a64fx/glm5/strata-gemm-acc-native-20261002.json).
- Whole expert groups 96→192: all 51 reference-part gates and ordered hashes
  pass, but 47-thread/C4096 medians regress 23.262978→25.341034 ms. Keep 96.
  Separate PJM52093332; parallel first-touch differs from production placement.
  See [group record](a64fx/glm5/strata-group-limit-native-20261002.json).
- Expert-major panel schedules 64/64 and 256/512: each passes 210 mixed-format
  native chain/guard cases but regresses at 47 threads. Separate PJM52094362
  and 52094882; independent idle-allocation repeat confirms the coarse-panel
  regression. Frozen sources/logs remain in `tmp/strata-moe-panels-20261002/`.

**New opt-in `--moe-prefill-layout padded`** keeps the existing 96-token
scheduler and adds 64 floats to private gate/up output strides. All arithmetic
is retained; default remains `tight`. Added scratch is about 1.1 MiB/rank at
47 workers. Same-host native expert-chain medians at 47 threads/C4096 improve
21.781921→21.356106 ms (~2%), excluding router and collective work. Native
Q4/Q5 gate/up, Q5/Q6 down, intermediate256/512, ragged5–385 and route guards
pass 210 fast/conservative comparisons under OMP settings1/3/12/47/48 in
PJM52095321. The unit exercises a serial worker chain; real parallel benchmark
hashes also match. See [layout evidence](a64fx/glm5/strata-moe-layout-native-20261002.json).

Integrated layout cross-build passes. **candidate-moe-layout-v2** retains v1
model/benchmark binaries and changes only the diagnostic checker. Benchmark
SHA `e8be0a52f165052d430b4e144d13d9cf87de74ca2fa7f398e5a202206e5ff587`, checker
`32a437f13e4f020e0cdbcb22d003f3ad492b468cb00fa9f5be07da7ca7217cc7`.
Frozen update-v1 SHA `4c09c011af39a9b294d8a13dd968ef2071cb0251e7260ef6c72c1233b57c6ac2`.
The full8049-position prompt gate passes zero mean/final-stream mismatches
and complete bit-exact KDA/sparse state, with minimum headroom7.844 GiB.

**Completed layout PID19989**, `tmp/strata-moe-layout-20261002/`
`campaign-resume-v3.{sh,log}` / `campaign-v3.py`: fresh frozen control,
rebuilt-tight control and padded three-trial timings, then qualifying independent
confirmation/context checks. Start guard17:45; new-run guard18:00 JST.
Independent confirmation is 34.979754 /410.735112 vs35.073216 /410.559835,
ratios0.997335 /1.000427. All IDs exact; no promotion and no contextstress.
`MOE_LAYOUT_CAMPAIGN_PASS` observed18:00; tight remains selected. See
[full layout record](a64fx/glm5/strata-moe-layout-full-20261002.json).

Initial layout/mHC launches stopped on missing topology helpers. The verified
capacity helper was copied into their isolated builds (SHA
`ea0fbfbc060645bd98f1630e52572ac303a9c64134221ae5196a95c10e26a19b`).
The first layout retry and MTP checker reached model-state checks: final
streams/state/token were exact, but the serial reference mean disagreed with
the export's flattened OpenMP loop under FCC fast math. Changing only the
reference loop eliminated all8049-position discrepancies in the layout gate;
strict memcmp remains. Old logs and traces are preserved.

Normalized MTP layer45 staging passes, as do 896 native controller cases and
2051-position full/cache-only/rollback hidden-bit checks. Its corrected
prompt checker now passes all8049-position means, final streams and complete
state with zero bit mismatches (minimum7.835 GiB); short/8K state checks are
next, followed by independent timed controls/depths. **candidate-mtp-v5** retains v3 math/v4 benchmark, with checker SHA
`6367bc1999ea2fc7a6cf63799f4e77305f4c08d559c10f9ec6f0012b07bef618`.
Benchmark SHA `c6c1a6381ccb30adbdf186c67d09f9f44b45c2aa79afee73dc560b2b7efd9040`.
Frozen source-v3 SHA `337e92cc60624c75078f78f3d4feea6693c2fa846215f408565b84cdda21e618`.
The previous stages `/local/glm53f-mtp-{routed,shared}-52085859` belonged to the expired allocation; current stages use52097252.
`campaign-v2.log` failed. **Completed MTP state campaign PID20602 (failed short construction)**,
`state-campaign-v3.{sh,py,log}`, uses new `state-results-v3/` outputs. It checks
corrected prompt means, short128 and8K depth4-always full decode state with
one timed128-transition trial each. This bounded state campaign is diagnostic,
not promotion; guards18:07 start/18:08 new-run, ahead of18:17 expiry. The short128 run aborts before timing. Diagnostic retry PID21455 also exits
with `GLM53F_BENCH_CREATE_FAIL rank=4 phase=mtp context=0 workspace=1 hidden=1`.
No IDs/acceptance/throughput result exists. Candidate-mtp-v7 adds tensor-read /
component-pointer failure logs; its cross-build passes. Benchmark SHA
`48f42109d53905515396c0cf8d7109bbe757364cdc6649d7b70cd32cdda418e1`.
This constructor diagnosis is complete and fixed in candidate-mtp-v8 above; short/8K state and multitrial qualification remain pending. See [MTP progress](a64fx/glm5/strata-mtp-progress-20261002.json). Include every teacher-forcing cost in prefill
and count only delivered target transitions in decode. No MTP throughput or
acceptance result exists yet. See [native MTP record](a64fx/glm5/strata-mtp-native-20261002.json).

**Deferred mHC resume PID20607 (exit75)** followed MTP PID20602 serially:
`tmp/strata-mhc-sync-20261002/campaign-resume-v3.{sh,pid,log}`. Artifact
**candidate-mhc-sync-v4** has passed cross-build and all14 fast/conservative
native configurations (384 chained calls each), separate PJM52089999. Synthetic
fast47/persistent47.444470→46.666629 µs is +1.67%, not full-model tok/s.
Full-model128-position exact-state gate precedes fresh/rebuilt controls and
fused-sync timings. Start/new-run guard18:00; likely deferred after MTP. Previous waiting
PID20136 was stopped before any MPI to prioritize bounded MTP state checks. Benchmark SHA
`50fa19bbbca7a2f6a92ce5d870ce41735c3b613055bbd08cb101e0df431e1a32`.
See [native mHC record](a64fx/glm5/strata-mhc-native-20261002.json).

Previous allocation **PJM52085859**,12 nodes2×3×2, requested normal2GHz/eco0,
starts12:17:07 and expires18:17:07 JST. Host`l31-4004b`; local tmux socket
`tmp/tmux-glm53f`, session`glm53f-strata-next3`; bridge ports42446→32446→21264.
Isolated remote repo`~/work/gemm/glm53f-strata-20261001`; helper
`python3 tmp/bash-http-glm53f-strata/remote.py`. Do not overlap helper calls.
Routed staging completed14:22,15,456,534,528 bytes/rank; native/embed/head
stage paths unchanged. Cross-builds use login-nodefccpx/mpifccpx and repo tmp.
Native probes use separate jobs or the verified idle interval17:03:02–17:04:10;
lookup controller was resumed and finished normally.

Objective remains100+ delivered decode /2000+ prefill for the complete45-layer
saved8049-ID single request. Current qualified pair35.462134 /412.273634;
historical panel47 pair35.742137 /370.754010. Targets unmet; goal stays active.
Original remote checkout and unrelated q38fn/CUDA/procedure changes are
untouched. No push authorized or performed.

## Completed continuation (October 2)

- PJM **52075759**, 12 nodes, 2×3×2, normal 2 GHz / eco 0, approximately
  04:05–10:05 JST; compute bridge on `f28-0008c`, tmux `glm53f-strata-next2`.
  Same isolated remote snapshot and HTTP forwarding as below.
- Bounded stage PID **685**, log `tmp/strata-20261002/stage.log`; routed
  rank-zero log `tmp/glm53f-q4-52075759/routed-stage-keys4-stage.1.0`.
  All ranks completed layer 44; stage finished at 06:30 with the OK sentinel.
- Immutable builds **candidate-keys4-v1** and **candidate-values-v1** complete.
  Build logs `tmp/strata-20261002/build-{keys4-v2,values-v1}.log` remotely.
  Native fast/conservative index keys4 and MLA register-value arithmetic
  checks pass at 1/12/47/48 threads. Measurements during staging are contended;
  no new whole-model throughput result or promotion yet.
- Committed candidates add opt-in `--index-kernel keys4` and `--mla-kernel values`.
  Derived FP16 latent cache integrated behind `--mla-kernel fp16-cache`;
  **candidate-cache-v3** builds, native fast/conservative value/conversion
  gates pass at 1/12/47/48 threads. Actual prefill attention has 60 byte-exact
  cases across all head counts, selections and key tails, both compilers.
  Aligned native value outputs now match runtime cache-line boundaries;
  updated fast/conservative arithmetic still passes. Earlier unaligned
  scratch timing could include false sharing and is not promotion evidence.
  Uncontended native units passed. Sparse scalar/batch and strict derived-cache
  prefill/rollback comparisons passed; 128-position full executor state
  gate passed with zero hidden/state bit mismatches.
  Wider mHC tiles rejected
  for slower native probes; no runtime mHC tile path retained.
- Historical wrapper setup cleared `OPAL_PREFIX OMPI_CC OMPI_CXX`. For the
  current allocation, clear `OMPI_CC OMPI_CXX` and set
  `OPAL_PREFIX=/opt/FJSVxtclanga/tcsds-1.2.43`; the unconfigured default is missing.
  Never use global `set -e` in the persistent bridge shell; use child scripts.
- Detached campaign PID **3675**, `tmp/strata-20261002/kernel-campaign.log`,
  waits for stage completion, then runs uncontended native arithmetic/timing,
  sparse scalar/batch/rollback and byte-exact derived-cache comparison gates,
  followed by a 128-position complete-state legacy/candidate comparison.
  Scripts `tmp/glm53f-kernel-campaign-20261002.{sh,py}` contain sequential
  8K ablations, five-trial confirmation, 1024-transition stress, short128 and
  repeated32K qualification. Same-allocation 8K medians: old-best
  **32.731477 / 329.018991**, rebuilt control **32.777221 / 326.879995**,
  keys4 **32.685680 / 328.203548** decode/prefill tok/s. Both compared
  candidates match all 257 output IDs; neither is promoted. Value/cache
  ablations finished: values **33.357720 / 329.070713**, keys4-values
  **33.628723 / 328.824199**, cache-heads **34.079982 / 345.582371**,
  cache-keys4 **34.034069 / 345.518427**. All complete 257-ID streams exact.
  Independent five-trial confirmation passed: control **32.610212 /
  327.987043**, cache-heads **33.939857 / 346.234747** (+4.08% / +5.56%).
  All IDs exact. Stress1024 also passes (1025 IDs): **34.311308 /
  345.120441** vs **32.991032 / 329.546116**. Short128 exact: **40.651945 /
  281.451159** vs **40.450590 / 282.981541** (essentially unchanged).
  Repeated32K qualification passes: **32.385736 / 316.830563** vs
  **31.326356 / 286.622911** (+3.4% / +10.5%). All complete IDs exact;
  `ALL_KERNEL_QUALIFICATIONS_PASS` / `KERNEL_CAMPAIGN_PASS` at 07:52.
  Cache-heads now qualifies for promotion; targets remain unmet.
- Replicated-index PID **6995** completed at 08:08. Both decode variants
  match all 257 IDs but are rejected for throughput: replicated-heads
  decode/prefill ratios **0.996676 / 1.003960**, replicated-keys4
  **0.997061 / 1.002729** against cache-heads. Strict sparse comparisons
  passed at warm2046/2051 (prefill32) and warm8049 (decode4/rollback).
  Log `tmp/strata-replicated-20261002/campaign.log`.
- Selector/panel campaign **PID 10777 completed**, log
  `tmp/strata-panels-20261002/campaign.log`; `PANEL_QUALIFICATIONS_PASS` /
  `PANEL_CAMPAIGN_PASS`. Immutable builds candidate-panel32-v1 /
  candidate-panel47-v1 / candidate-panel64-v1 pass. Panel 47 selected.
- Normal-runner reservation follow-up **PID 13836 completed**, log
  `tmp/panel-runner-check.log`, `PANEL_RUNNER_CHECK_PASS`. Shared integer
  reservation helper covers compiled output panels and packed pool scores
  in all fast-prefill entrypoints; invalid/overflow counts are rejected.
  Config tests pass locally at 32/47/48/64 and natively at 32/47. Native normal
  generation panel 32-vs47 matches all 32 generated IDs after 8049 prompt IDs.
  This separate gate retains the normal runner's scalar final prompt token;
  the resident benchmark throughput binary remains unchanged.
  Normal runner bin `a64fx/glm5/build/candidate-panel47-runner-v1/`.
  Evidence `tmp/glm53f-strata-evidence-20261002/runner/`; frozen correction
  `tmp/glm53f-panel-runner-reservation-fix.tar.gz`.
- Implementation **c461079c**, 13 files, 276+/47-. Opt-in bounded selector
  `--pool-selector partition4k` uses 513..4096 pools, exact original ordering,
  dead score-reduction scratch and original heap fallback. Default heap and
  attention panel 32 remain available; build argument 47 selects the winner.
- Independent five-trial confirmation: control **33.793083 /345.065496**,
  panel 47 **35.742137 /370.754010** decode/prefill tok/s, **+5.77% /+7.44%**,
  all 257 IDs exact. Promotion decision true; 100/2000 targets remain unmet.
- Stress1024: control **34.040300 /
  344.113402**, candidate **35.746584 /
  371.903716**, all 1025 IDs exact.
- Short128: control **40.436245 /
  282.005617**, candidate **40.660046 /
  285.918131**, all 257 IDs exact, no regression.
- Repeated32K (32196 promptIDs): control
  **32.275942 /316.484767**, candidate
  **32.185018 /341.961472**,
  all 257 IDs exact. Synthetic 8K repeated 4x. Minimum across panel campaign
  **9.831 GiB** sampled headroom.
- All local 120-case selector oracle/bounds, ASan/UBSan (LeakSanitizer
  disabled), config 32/47/48/64 and launcher 16 checks pass. Native fast and
  conservative selector 120-case tests pass after timing; final unit waiter
  PID 12302 completed, `selector-final-units.log`: `SELECTOR_FINAL_UNITS_PASS`.
  Strict selector prefill/decode/rollback and 32-vs47/64 panel gates pass.
- Qualified bin `a64fx/glm5/build/candidate-panel47-v1/bench_glm53f_run_12n`;
  native objects `/local/glm53f-panel47-build-52075759`, full frozen source
  `/local/glm53f-panel47-build-52075759/src/a64fx/glm5`.
  Flags: persistent decode, fused router, grouped verify, vector MoE,
  heads index, fp16-cache MLA, partition4k selector; 47 threads/page none,
  Q8 panel 0/sparse async 0/KDA async 1/collective owner 0. Build-time panel 47.
- Rank-zero diagnostic medians: sparse prefill 0.966→0.760 ms/token;
  decode index 3.459→2.108 ms/token. Candidate KDA 6.405, mHC 5.764,
  MoE 7.582 ms/token decode; prefill MoE about 1.02 ms/token unchanged.
  Parent/child timers overlap and are not rank-max throughput.
- Evidence copied locally to `tmp/glm53f-strata-evidence-20261002/`;
  `replicated/` holds rejected replicas, `panels/` holds native gates,
  complete logs/IDs/reports and binary/prompt/topology metadata. All 10
  follow-up reports recomputed locally and match remote JSON exactly.
  Committed record `a64fx/glm5/strata-kernel-validation-20261002.json`.
- Immutable source archives: `tmp/glm53f-replicated-index-update.tar.gz`,
  `tmp/glm53f-prefill-panels-update.tar.gz`, `tmp/glm53f-selector-update.tar.gz`.
  Scripts `tmp/build-glm53f-prefill-panels.sh` and
  `tmp/glm53f-prefill-panels-campaign-20261002.{sh,py}` run serially.
  No push. Allocation 52075759 remains live until approximately 10:05 JST;
  stages disappear when it expires. Do not overlap subsequent timed MPI jobs.

## Previous allocation and isolated deployment (expired)


- PJM **52068253**, 12 nodes, compact 2×3×2, normal 2 GHz, eco 0; six hours
  from about 20:40 JST October 1 to 02:40 JST October 2. Node `d26-2014c`.
- SSH `fugaku1`, login1 u14346. Isolated remote snapshot
  `$HOME/work/gemm/glm53f-strata-20261001`; preserve original `~/work/gemm/glm53f`.
- Local tmux `glm53f-strata`. Bridge state/helper:
  `tmp/bash-http-glm53f-strata/remote.py`, stdin Bash, 50-second timeout
  does not terminate a remote command. HTTP 42446 → reverse 32446 → 21264.
  Use nohup for long jobs and never overlap MPI launches.
- Job 52067029 belongs to other work; leave it alone. No `/tmp` use.
- Staging finished 00:07 JST: `SENTINEL glm53f_stage_12n=OK`.
  Allocation-local routed weights `/local/glm53f-q4-routed-52068253`,
  native `/local/glm53f-q4-native-52068253-{core,shared,dense,sparse,kda,shexp}`,
  embed/head `/local/glm53f-q4-{embed,head}-52068253`.
  MTP staged `/local/glm53f-mtp-{routed,shared}-52068253`.
  All `/local` stages disappear after allocation restart; stage boundedly.

## Completed queue

No builds, staging or inference MPI jobs remain running in this campaign.
The allocation/bridge may remain live until about 02:40 JST.

- Final hardware controls v17 PID15784 **PASS**: sparse+pages and 48-thread
  baselines/candidates. All candidate IDs match, including 47 vs 48 threads.
- Final synthetic 32196-position qualification v19 PID16787 **PASS**, log
  `tmp/qualification-32k-v19.log`. All three trials plus warmup finish, all
  257 IDs match frozen baseline. Median baseline 27.747994 decode / 265.965590
  prefill; candidate 31.450699 / 287.156697 (+13.34% / +7.97%). Minimum
  headroom baseline 10656768 KiB, candidate 10699520 KiB, both above 10 GiB.
  Outputs `tmp/strata-v10b/repeated32k-{baseline,candidate}-v19.*`, report
  `repeated32k-v19.json`. Report recomputed locally byte-identical.
- Initial v16 32K attempt failed beyond 16K because packed index-score
  reduction exceeded the fixed 131072-float benchmark reservation. Commit
  **071fdcf3** sizes it to max(131072,32*floor(prompt/4)); same harness fix
  applied to frozen and candidate kernels. No relaxed correctness gate.
- v18 build **PASS**, objects `/local/glm53f-strata-v18-52068253`, bins
  `a64fx/glm5/build/{baseline,candidate}-v18`. Baseline uses frozen a90
  objects plus scalar-loop compatibility `baseline_sequence.c`; lookup off.
  Candidate retains v15 objects. Build script `tmp/glm53f-strata-32k-v18.sh`;
  fresh run script `tmp/glm53f-strata-32k-v19.sh`. New dirs initially missed
  `tofu_topo_helper`; symlinks supplied before v19, which completed PASS.
- Previous campaigns/build/stage PIDs exited. Never overwrite running
  compiler sources/scripts or existing exclusive outputs. Use fresh tags.

## Source and binary provenance

Base **a90f7972**, frozen bins `a64fx/glm5/build/baseline`, objects
`/local/glm53f-build-52068253`. Original checker is
`a64fx/glm5/run_glm53f_baseline_check.sh`. Code commits:

- **86fd586f** persistent executor, grouped experts, fused router, serialized
  communication owner, lookup controller, native kernels and resident harness.
- **58e3539e** strict complete MTP reference validation.
- **ba6a2140** register MLA absorbed query.
- **d3644d79** exact sparse gate isolates projection GEMM and reports real logs.
- **f9992c7b** owner stress initializes OpenMP before helper affinity selection.
- **a70a9994** reporter reads actual `.greedy` reference suffix.
- **bfcf404e** sparse owner preserves original index uTofu reduction order;
  adds direct regular-vs-async exact boundary check.
- **174945d3** recurring 16-cycle lookup cost checks and late-cost regression.
- **071fdcf3** prompt-sized collective reservation and prefill failure context.

Object sets: full v7 `/local/glm53f-strata-v7-52068253`; timer-safe target,
head and lookup v8; v9 MTP completion; v10 MLA; v14 sparse/collective owner
correction; v15 lookup policy. Latest general bins `candidate-v15` symlink
unchanged v14/v10/v9/v8/v7 tools. Native build warnings are existing unused
static helpers. Final audit checks 422 tracked code/script files; all match
after refreshing the single stale reporter test (runtime source already
matched). Strata studied locally at `~/work/Strata`,
`glm53f` revision e486a95, independent implementation, no source copied.

Archives SHA256:
- `tmp/glm53f-strata-base.tar`:
  `d9940e19d7324cbd26ef1357b3a9760c0f113dea23cc327bc8ff41d387968168`.
- `tmp/glm53f-strata-v7.tar`:
  `cc0fbcb4f2ec2a1dbe41d54a1b5ac6bf13dfe7e03a4d64d463a5fc2b13c05f7a`.
- `tmp/glm53f-strata-v8-timers.tar`:
  `02322782e6542a320c952950597d767cb5d1dfd8f20ceff31bf66068e9bf3381`.

## Validated results

Saved fixture `tmp/prompt8k.ids` has **8049 IDs**. Benchmark counts all
positions and final head; older prefill reports counted 8048. All throughput
comparisons have one warm + three timed trials, rank-max elapsed, 256 decode
transitions / 257 output IDs, sampled memory guards and strict completeness.
Primary 47-thread results (decode / prefill median tok/s):

| Configuration | Decode | Prefill | Exact IDs |
| --- | --- | --- | --- |
| Frozen a90 | 29.989745 | 303.243600 | Reference |
| Persistent/fused | 30.764120 | 303.341836 | PASS |
| Above + index heads/vector combine | 30.713757 | 321.368333 | PASS |
| Above + register MLA | 30.894098 | 318.750066 | PASS |
| Above + page type none | **33.066761** | **327.644514** | PASS |
| Corrected sparse-only async, original pages | 30.879418 | 325.856357 | PASS |
| Corrected sparse-only async, pages none | 32.826162 | 335.438950 | PASS |

Best balanced 47-thread candidate improves frozen decode 10.26%, prefill
8.05%; retains row-Q8 layout, no MoE/sparse async. Owner setting was serialized
in its 8K trial but never active (no async begin). Short/32K use owner off.
Sparse+pages is below incremental promotion threshold. 48-thread baseline
with pages none: 31.326396 / 302.705849; candidate 33.402963 / 317.928989,
exact, +6.63%/+5.03%. Compared with 47-thread candidate, +1.02% decode but
-2.97% prefill, so keep 47 for qualification. Prefill trial variation is larger
at 48 threads (300.170450–320.935217).

128-ID short fixture: baseline 36.575832 / 267.477754; candidate
40.431636 / 282.637846, all 257 IDs exact, +10.54%/+5.67%.

Lookup depths 1–4 original medians: 30.305971, 30.247893, 29.430768, 26.709574,
all exact; no promotion. Revised recurring-cost depth 4 **30.549583**, exact,
14.38% faster than old depth 4, below plain. Unit 29 PASS locally and 12 ranks;
old policy fails delayed-cost regression.

MTP complete 128-cycle/depth 1–4 sweep passes every delivered token against
its own scalar-prefill plain reference: 768 tokens at 30.685 tok/s. Medians
28.988085, 26.196257, 22.001181, 17.897930; delivered 352, 403, 419, 419/trial.
Acceptance 96/128, 147/256, 163/384, 163/512. Scalar MTP prefill and resident
batched prefill have different greedy streams (first difference prediction 14);
MTP exactness does not claim the batched baseline stream. No MTP speedup.
MTP min headroom 10.3 GiB; resident 8K above 10.9 GiB.

Rejected: combined sparse/MoE overlap diverges at ID 18 (1852/198); Q8 panel 1
at ID 6 (61102/563). Never promote their throughput. Original sparse async
MPI detour changes reduction order; fixed owner retains uTofu, full 8K exact.
MoE overlap still replaces whole-chunk MPI with slabs and remains excluded.

Native gates: team/expert/mHC/vector/index/MLA exact at 1/12/47/48 threads,
fast and conservative where relevant. Index kernel ~2.6×; MLA query ~1.8×;
these are not whole-model rates. Full-state 32 positions hidden bit mismatches 0,
complete KDA/sparse state BIT_EXACT. Every-prefix verification continuation
PASS. Sparse layer 3 warm 2046 tokens 32: reference error 0, batched MLA rel-L2
7.55019511e-05 < 2e-4, rollback 0. Exact gate pins projection GEMM / fused front 0;
the original baseline also fails with projection GEMM 1. New direct native
regular-vs-owner async check rel-L2=0 / rollback 0. Mixed MPI/uToFu owner stress 300
PASS after first-team affinity initialization. Local launcher 16 / report 6 PASS.

## Evidence and next architecture work

Local `tmp/glm53f-strata-evidence-20261001/` holds reports, rank-0 logs,
metadata hashes and IDs; remote `tmp/strata-v10b/` holds complete campaign.
Final 32K report/logs/IDs/scripts are retrieved. Committed aggregate:
`a64fx/glm5/strata-validation-20261002.json`. Use strict reporting, never
acceptance alone. Review `GLM53F_STRATA.md` for exact commands.

Best 8K rank-0 phase profile: decode ~29.8 ms summed local phases, attention 15.7
(KDA 6.27 / sparse 9.44), mHC 5.73, FFN 7.89; end-to-end rank maximum ~30.2 ms.
Prefill ~3.07 ms/position, attention 1.60 (sparse 1.11), FFN 1.17, mHC 0.247.
100/2000 require a larger architecture/kernel change than pool overhead.
PP3×TP4 is approved for evaluation if TP12 remains insufficient, but is not
implemented/qualified. Needs TP4 stage images, layer ownership, stage-local
collectives, four-stream handoffs and full-state exact gates. Single dependent
decode latency includes every stage; do not assume 3× pipeline speedup.

No push authorized. Preserve unrelated untracked q38fn and tmp artifacts.

Current allocation is **PJM52097252**, same12 nodes2×3×2, normal2GHz/eco0,
starts18:18 JST and expires about00:18 JST October3. Host`c30-7008c`; tmux
`glm53f-strata-next4`; same42446→32446→21264 ports. Fresh bridge session obtained;
old ID archived as `tmp/bash-http-glm53f-strata/session-52085859-final`.
**Target/MTP staging is active, PID134**, see
`tmp/strata-next-20261002-1818/{stage.pid,driver.log,stage.log,mtp-stage.log}`.
It uses immutable candidate-v15 and validates every rank sentinel before
staging only layer45. All `/local` prefixes use newjob52097252. Do not overlap
MPI/build/native timing with stage. Require `STRATA_NEXT4_STAGE_PASS`; inspect
owned processes before resuming constructor diagnostic/mHC qualification.
No overlap of12-node allocations. Old52085859 has expired and all its queues
have exited. Preserve its completed reports and failed diagnostic traces.
