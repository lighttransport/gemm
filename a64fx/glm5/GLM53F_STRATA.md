# Strata-inspired GLM53F optimization: implementation and validation

Updated 2026-10-03. Targets are **100 delivered decode tokens/s and 2000
prefill tokens/s**, on twelve A64FX nodes, the complete 45-layer
UD-Q4_K_XL model, top-8 routing, and the saved roughly 8K coding prompt.
These targets have **not been demonstrated** by the changes below.

## MLA prefill head and column tiles (October3; small gain, no promotion)

`--mla-prefill-heads legacy|split6|values32`, default`legacy`, adds two opt-in
native prefill experiments. Only the four six-head TP12 slices change;
five-head and PP slices keep their existing kernels. `split6` runs independent
three-head groups and retains all six logit rows. `values32` keeps six-head
cache reuse and reduces the value tile from64 to32 columns. Query/key lane
reductions, ascending selected-key FMAs, FP16 cache rounding and masked
softmax tails retain their original order. Allocation size and collectives
are unchanged. Scalar decode and speculative verification are unaffected.

Strata's absorbed-query/cache/value tiling in
`src/program/glm_decode.cpp:2965..3040` on local branch`glm53f` at`3bbb469`
motivated examining tile shape. FCC assembly has vector spills in MLA logits
and values; the diagnostic records static source-attributed counts, with the
explicit limitation that these are neither dynamic traffic nor speedups.

FCC fast math builds pass`-Wall -Wextra -Werror`, attentionpanel47 and
capacity4096. Strong native fixtures pass1584(split6)/1848(values32) cases per
rank, on all12 ranks. They compare complete logits/values and canaries at
512/768-float strides, varied signs/mantissas and rank seeds, both cache views,
ascending/reversed/permuted selected keys, and11 counts through2052. Finite
checks inspect IEEE exponent bits under fast math. Five alternating native
tile sweeps give six-head component medians**1.142844× split6** and
**1.017734× values32**. Values32 fails the5% component gate; model timing was
skipped. It remains an experimental option with native-only evidence.

Five fresh alternating same-binary whole-model pairs give median paired
ratios**1.017089 prefill /1.000828 decode**. Absolute rate medians are
406.956430→413.551356 prefill and34.181202→34.209521 decode tok/s. These
percentages come from paired ratios, not ratios of absolute medians. Every
prefill pair improves1.54–1.75%; decode fluctuates and is effectively
unchanged. The first timed sparse-prefill profile falls0.806→0.766ms/token,
while routedFFN stays0.840ms/token. Warmup and timed profiles are separately
retained in their original order.

Fresh frozen control:412.574268 prefill/34.808562 decode. All13 completed
8049-prompt model runs pass complete129/257 IDs against that frozen reference;
minimum available8.871399GiB. Full-prompt/128-transition endpoint states match
byte for byte across legacy/split6 on all12 ranks, in addition to internal
warmup/trial checks. The report retains each rank's file length and SHA256.
No short-prompt,1024-token or32K candidate qualification is claimed. Prefill
gain remains below5%; keep the qualified35.462134 decode/412.273634 prefill
recipe. Neither100/2000 target is met.

Runtime binaries are immutable`candidate-mla-head-tiles-v2` (split6) and
`candidate-mla-head-tiles-v4` (adds values32); strong split-only fixturev3 is
separate. Fixturev5 changes indentation only and passes the final1848 cases
per rank; its source/binary SHA is recorded separately. V4 build source hashes
were verified before that formatting; v2 source hashes are
explicitly labelled reconstructed from the recorded v4 delta. The new
`test_glm53f_mla_head_tiles` is included in integrated`checkbuild`. Local
`cc -D_GNU_SOURCE -std=c11 -O2 -Wall -Wextra -Werror
 a64fx/glm5/test_glm53f_prefill_config.c` passes its five new parser cases;
all18 launcher tests pass. Native runs use `mpi_run mla-head-tiles
 /usr/bin/numactl --interleave=4,5,6,7 TEST_BINARY` after canonical runtime
preparation, with47 threads and the existing FCC binding policy. Exact build,
driver and model commands, all paired trials, hashes, native timings and
profiles are in [the evidence report](strata-mla-head-tiles-20261003.json).
Scratch/scripts are under`tmp/strata-mla-head-tiles-20261003/`.

## Batched MTP priming and verification weights (October3; qualified MTP gain)

Three independent opt-in candidates are implemented. The benchmark's
`--mtp-prime-batch 1|64` batches teacher-forced prompt pairs. It retains the
existing first-position mask and norm reductions, uses the existing exact
BF16 4×4 fusion kernel, gathers rank-major shards once per tile, and appends
FP8/BF16 cache projections with identical scalar chains. Native/CP cache
formats use the original scalar fallback. Scratch is allocated lazily and
bounded at8MiB per rank; timing includes complete prompt priming.

`--verify-head-kernel legacy|shared` shares FP32 vocabulary weight loads across
2–5 verification positions. Every row retains the original single-accumulator
SVE chain and horizontal sum; ordinary scalar readout is unchanged.
`--embedding-batch-kernel legacy|packed` packs broadcasts per vocabulary owner
for TP12 prefill. Both FP32/BF16 row storage are supported; PP retains its
existing packed behavior. All TP12 defaults remain legacy/1.

PJM52128881,12 A64FX nodes/normal2GHz/eco0,47 threads and FCC fast math:
all20 embedding and36 actual-weight MTP state/rollback cases pass on every
rank. Head tests pass150 cases per fast/conservative1/47/48-thread setting,
both with the original inputs and a stronger dataset varying signs, all23
mantissa bits and16 exponents independently per rank (21600 case instances).
Five2051-position priming pairs give2.663894×. The stronger head fixture's
47-thread fast paired medians are1.863/2.809/3.257/3.387× for2/3/4/5 positions.
These are component measurements. See
[native evidence](strata-mtp-batch-native-20261003.json).

Short128/full8049 speculative endpoint states pass BIT_EXACT. All25 complete
model runs/51 timed trials pass their expected129/257/1025-ID counts; every
screen and long output matches the frozen plain reference. Minimum available
memory is7.748GiB. Five alternating fresh pairs compare scalar MTP priming/
legacy head with batch64/shared head, using the same candidate binary and
packed target embedding in both. Acceptance remains exactly122/133 in every
pair. Median paired gains are **+6.6175% MTP prefill /+1.2797% MTP decode**.
Median absolute rates are:

| Same-binary MTP depth1/always | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| Prime1, legacy head |365.011296|33.059193|
| Prime64, shared head |389.126088|33.331813|

The percentage uses the median of five paired ratios, rather than the ratio
of absolute-rate medians. The last pair improves prefill only4.62%; all five
improve decode, and the paired medians pass the5%/2% feature threshold.
**This qualifies an improvement within MTP, not a replacement for plain decode.**
The fresh plain screen is34.978736 decode/410.478727 prefill; packed plain
is34.490333/407.402491. MTP depths2/4 adaptive and both lookup policies lose.
At1024 transitions, plain is34.929742/412.428817; adaptive MTP is33.990409/
389.203736 and always MTP is34.387190/389.345006. All1025 IDs match. Adaptive
MTP falls back for902 transitions, so the always-on run separately measures
sustained speculation. Depth1 still spends about54ms per verification cycle;
high acceptance does not remove that cost. Packed embedding cuts its prefill
phase0.050→0.023ms/token, but attention/FFN dominate. The rebuilt default
control also has higher sparse-MLA time than the frozen binary; both controls
and phase profiles are preserved.

**No plain promotion; retain the qualified35.462134 decode/412.273634 prefill
recipe. Neither100/2000 target is met.** New paths stay opt-in; no context-wide
promotion or32K MTP qualification is claimed. Full commands, binary/source
hashes, all trials, paired results and profiles are in
[full-model evidence](strata-mtp-batch-full-20261003.json). Implementation is
commit`79d67a35`; the runtime benchmark remained immutable throughout.

Native fixture reproduction after staging, with the shared `check` build:

```sh
BIN="$PWD/a64fx/glm5/build/mtp-check"
export OMP_NUM_THREADS=47 OMP_PROC_BIND=close OMP_PLACES=cores FLIB_BARRIER=HARD
GLM53F_BIN_DIR="$BIN" GLM53F_FAST_MATH=1 GLM53F_NO_MATH_ERRNO=1 \
  bash a64fx/glm5/build_glm53f_integrated_12n.sh check 47 4096
mpiexec -n 12 /usr/bin/numactl --interleave=4,5,6,7 "$BIN/test_glm53f_embedding_batch_12n"
mpiexec -n 12 /usr/bin/numactl --interleave=4,5,6,7 "$BIN/test_glm53f_head_verify"
GLM53F_REPACK_REQUIRE=0 mpiexec -n 12 /usr/bin/numactl --interleave=4,5,6,7 \
  "$BIN/test_glm53f_mtp_prime_12n" "$MODEL" "$MTP_ROUTED" "$MTP_SHARED" /local/mtp-prime-check
```

Select the measured MTP feature with `--speculation mtp --draft-depth 1
--spec-policy always --mtp-prime-batch 64 --verify-head-kernel shared
--embedding-batch-kernel packed`, plus both MTP stage paths and the recorded
baseline options. The priming option is currently benchmark-only.

## Native dense prefill tiles (October3; exact but no promotion)

`--dense-prefill-tile 4|16|32|64` selects the native dense batching capacity
before model construction. Default4 and FP8 behavior stay unchanged. The
wider path retains four-token Q8 row kernels and four-token reduction slabs;
it amortizes OpenMP setup and activation-allocation calls. Scratch is bounded
at64 positions; the PP memory inventory reserves6MiB per dense layer for its
buffers and prepared-activation peak. Decode and verification use their
existing scalar/small-batch paths.

On PJM52116293, 12 A64FX nodes/normal2GHz/eco0,47 threads and fast FCC, all126
real-weight bit-exact cases pass: layers0–2, tiles16/32/64 and14 aligned/tail
counts through129. Five paired128-position component trials give tile64
median speedups1.331170,1.302043,1.429393 across the three layers. These are
component measurements, not whole-model tok/s. The full45-layer short128/
8049 prompt-mean, final-stream, persistent-state and first-token gates pass
**BIT_EXACT**, minimum headroom10.056885/8.118774GiB. All15 fresh screen/
confirmation runs match all257 frozen output IDs. Five alternating fresh
baseline/candidate pairs select tile64 but measure median paired ratios
**0.995332 prefill /0.981420 decode**. Median rates are frozen410.902710 /
34.809173 and candidate409.116828 /34.265047 tok/s (prefill/decode).
**No promotion; keep default4 and qualified capacity4096.** The targets remain
unmet. The final126-case unit also passes setter/capacity rejection guards.
Evidence: [dense tile report](strata-dense-prefill-tile-20261003.json).

Native reproduction after bounded staging:

```sh
mpiexec -n 12 test_glm53f_dense_prefill_tiles "$TP12_DENSE_STAGE"
mpiexec -n 12 glm53f_prefill_chunk_check_12n "$MODEL" "$ROUTED" "$SHARED" \
  "$PROMPT_IDS" "$TRACE" 4096 --reference-chunk 4096 --capture-hidden \
  --compare-dense-prefill-tile --dense-prefill-tile 64 \
  --prefill-mode fast --prefill-features 27 --prefill-slab 32 \
  --prefill-collective mtni
```

The checker constructs the candidate allocation once, selects tile4 for the
reference, restores the empty model snapshot and selects the wider tile for
the candidate. Its endpoints and all prompt means must remain bit-exact.
The new test programs are built by the shared `check` build.

## Routed expansion cache probe (October3; rejected)

A bounded prototype expands256 columns at a time into18KiB, while retaining
both the original512-column correction boundary and ordered FP32 GEMM FMAs.
Fast/conservative native Q4/Q5 gate-up and Q5/Q6 down route/guard/finite checks
pass21 cases per math mode on every rank. Five alternating paired synthetic
192-part timings at1/12/47/48 threads remain output-bit-exact. At the larger
114-position cohort corresponding to4096-token prefill, every thread setting
regresses about2.4–2.5%;47-thread median baseline/candidate ratio0.975628.
The14-position cohorts improve about2.8%, which does not justify changing the
selected4096 recipe. The production header and runtime are unchanged. The
fixture includes per-call allocation/output copies, unlike resident runtime
workers; these measurements do not establish full-model throughput.
See [cache-probe evidence](strata-moe-expansion-native-20261003.json).

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
immutable remote build paths. The Q8, capacity and lookup continuations are complete; see below and `resume-strata.md`.

Rank-zero diagnostic medians identify the remaining work: decode KDA 6.405,
mHC 5.764 and MoE 7.582 ms/token; prefill MoE stays near 1.02 ms/token. Sparse
prefill falls 0.966→0.760 ms/token and decode index 3.459→2.108 ms/token.
These nested component timers overlap and are separate from the rank-max
throughput measurements above.

## Q8 and prefill-capacity qualification (October 2 afternoon)

PJM 52085859 provides another six-hour 12-node allocation, approximately
12:17–18:17 JST. Bounded restaging and the native build passed. The newly qualified capacity4096 recipe measures **35.462134 decode /412.273634
prefill tok/s** in independent confirmation. The earlier panel47 recipe measured
35.742137 /370.754010 on its previous allocation.

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
The serial post-stage campaign completed uncontended probes, sparse
prefill/decode/rollback gates, complete-state comparison and full 8K/short/32K
output and throughput checks. Local evidence is `tmp/strata-q8-20261002/`;
`resume-strata.md` records artifact hashes and remote paths. No Q8 kernel was
promoted; neither target has been met.

All seven 8K Q8 ablations return the exact 257-ID stream. Independent five-trial
confirmation of ASM4 plus fused MLA projections measures **35.540871 decode /
377.077157 prefill tok/s**, against **35.609336 /371.931478** for the fresh frozen
control: −0.19% decode /+1.38% prefill. This misses the 5% promotion threshold.
Stress1024, short128 and repeated32K also match all IDs; the complete
campaign passes at15:48. Repeated32K measures32.101995→31.984868 decode and
343.931969→345.949648 prefill tok/s, below the promotion threshold.
Completed reports are recorded in [Q8 full-model progress](strata-q8-full-20261002.json).

Increasing outer prefill capacity to 4096 reuses expert expansion across larger
route cohorts while retaining 47-position attention panels and all arithmetic.
The source defaults remain 512. Build with
`build_glm53f_integrated_12n.sh check 47 4096` and explicitly use
`--prefill-chunk 4096` with the qualified runtime recipe above.

All four endpoint gates (512/1024/2048/4096) pass exact final hidden streams and
complete KDA/sparse state. Initial three-trial 8K prefill rates are 393.653658,
404.138154 and 409.149771 tok/s at chunks 1024, 2048 and 4096. Independent
five-trial confirmation qualifies 4096 for promotion:

| Workload | Decode control → candidate (tok/s) | Prefill control → candidate (tok/s) |
| --- | --- | --- |
| 8K, 256 transitions, 5 trials | 35.141258 → 35.462134 | 372.683205 → 412.273634 |
| 8K, 1024 transitions, 3 trials | 35.314212 → 35.212835 | 373.726670 → 411.918596 |
| Short 128, 256 transitions, 3 trials | 39.920756 → 39.901375 | 284.522204 → 285.306876 |
| Repeated 32196, 256 transitions, 3 trials | 31.789555 → 31.700983 | 343.959284 → 372.389918 |

The confirmation gains are +0.91% decode and +10.62% prefill. All 257 output
IDs match in 256-transition runs and all 1025 match in stress1024. Context
qualification and `CAPACITY_CAMPAIGN_PASS` complete before promotion. See
[capacity full-model evidence](strata-capacity-full-20261002.json) for exact
controls, settings, reports and hashes.

A subsequent lookup sweep uses this capacity4096 recipe. All five variants
(depths 1–4 adaptive and depth4 always, with ASM4 verification/fused projections)
match all IDs but none improves decode within the guard. Decode ratios are
0.999623, 0.999351, 0.988769, 0.990379 and 0.860653 against a fresh qualified
plain control. No lookup setting is promoted. See
[lookup full-model evidence](strata-lookup-full-20261002.json).

## Rejected packed GEMM register-accumulation probe (October 2)

An isolated three-token ×64-row SVE tile keeps12 FP32 sums in registers
across scale blocks; two calls consume each unchanged six-token packed
input group. The original per-block integer dot and ordered scale/FMAs
are retained, including initial sums from preceding K chunks. Native
one-node PJM52093052 (normal2GHz/eco0 requested) completes independently
of the twelve-node campaign. All2400 fast/conservative ×1/3/12/47/48-thread
cases pass every output/guard bit against the original assembly and an
independent ordered-FMA scalar reference. Cases cover scale blocks16–256,
K up to4096, nonzero initial sums, segmented K loops and varied strides.

Seven alternating-order trials per shape reject the candidate. At47
threads, representative legacy→candidate medians are45.210→50.227µs for
sb32/K4096/18tokens,128.145→140.294µs for48tokens, and390.669→486.910µs for
sb256/192tokens. Fewer integer accumulators and duplicated weight loads
are plausible causes, inferred from the implementation rather than counters.
No production kernel or default changes. The immutable source archive,
logs, all33 shape/thread medians and raw trials are identified in
[the native rejection record](strata-gemm-acc-native-20261002.json).

A second isolated probe (PJM52093332) compares96 versus192 whole-group
tokens in the original native Q4_K/Q5_K chain, including SwiGLU/requant.
All51 split/unsplit reference-part checks and all ordered output hashes
pass. At47threads/C4096, seven-trial medians23.262978→25.341034ms regress
8.94%; at48threads they improve22.965908→22.023916ms (+4.28%). The probe
uses parallel first-touch weights, so these are diagnostic timings and
cannot qualify production NUMA placement. Keep the96-token production
limit. See [group-limit evidence](strata-group-limit-native-20261002.json).
Expert-major panel scheduling was also rejected. Separate one-node probes
pass 210 mixed-format native chain/guard cases per scheduler, but 47-thread
C4096 medians regress 23.663998→26.950121 ms for 64-row panels and
23.608920→25.372030 ms for coarser gate/up256/down512 panels. An independent
idle-allocation repeat confirms the latter regression (21.792890→23.293020 ms).
The prototype is retained in immutable scratch archives rather than production.

## Experimental MoE prefill output padding (October 2)

`--moe-prefill-layout padded` adds 64 floats to each private gate/up output
stride while retaining the existing 96-token expert scheduler and every
GEMM, SwiGLU and quantization operation. The default remains `tight`. Added
scratch is about 1.1 MiB per rank at 47 workers. A same-host, uncontended
synthetic expert-chain probe improves 21.781921→21.356106 ms at 47 threads
and C4096 (~2%); it excludes router/collective work and establishes no
full-model tok/s gain. All parallel ordered-output hashes remain exact.

The committed mixed-format unit covers Q4/Q5 gate/up, Q5/Q6 down,
intermediates 256/512, ragged cohorts 5–385 and untouched route guards. All
210 native comparisons pass fast/conservative builds under OMP settings
1/3/12/47/48; this unit exercises a serial worker chain. Integrated FCC
cross-build passes. Evidence and immutable source/log hashes are in
[native layout record](strata-moe-layout-native-20261002.json).

Full-model qualification uses `--compare-moe-prefill-layout --capture-hidden`
with equal reference/candidate chunk4096. It requires every prompt mean,
final hidden stream and complete KDA/sparse state to match byte-for-byte.
Fresh frozen and rebuilt-tight controls precede padded timings, independent
confirmation and context gates. The first checker used a serial reference
mean loop that disagreed with the export under FCC fast math despite exact
final streams/state. A diagnostic-only revision uses the export's flattened
OpenMP loop structure; it retains strict memcmp. The revised checker passes all 8049 prompt means, final streams and complete
state with zero bit mismatches (7.844 GiB minimum sampled headroom). Independent five-trial
confirmation measures 34.979754 decode /410.735112 prefill tok/s versus
35.073216 /410.559835 for a fresh control: −0.27% /+0.04%. All 257 IDs match.
The initial three-trial +0.77% prefill gain does not survive confirmation, so
tight layout remains selected. Context stress is not run for this rejected
candidate. See [full-model layout record](strata-moe-layout-full-20261002.json).

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

Immutable native artifacts retain `candidate-mtp-v3` math and the benchmark
from `candidate-mtp-v4`. Layer45 staging, the 896-case native controller and
2051-position full/cache-only/rollback gates pass. The prompt mean diagnostic
stopped with exact final streams/state but mismatched per-position means;
`candidate-mtp-v5` contains the revised checker described above. The revised prompt checker now passes all 8049 means, final streams and
complete state with zero bit mismatches (minimum7.835 GiB). The short128 state run then fails during rank4 MTP context creation before
timing; a benchmark diagnostic retry confirms context=0 with workspace and
hidden allocation successful. Tensor/component diagnostics are cross-built
in candidate-mtp-v7. Reading the complete preserved rank4 log identifies a
compact-core miss for `layers.45.enorm.weight`: the reader cached strictness
from target initialization and ignored the benchmark's temporary MTP fallback.
The reader now caches the core image but evaluates `GLM53F_REPACK_REQUIRE`
on each read. A distinct-payload fixture checks target core reuse, exact draft
checkpoint fallback and restored strict rejection in both row/column paths,
from strict-first and optional-first initialization. All40 checks, ASan/UBSan
and18 launcher tests pass; both old-header counterfactuals fail.

Immutable candidate-mtp-v8 cross-build passes with the repaired shared reader.
Fresh PJM52097252 staging is complete on all12 ranks, including layer45.
Native loader policy passes80 cases, controller896 cases per rank and the
2051-position cache/rollback gate passes. The8049 prompt means, final streams
and complete state pass exactly, with minimum9.318 GiB available. Both short128
and8K depth4-always runs pass complete target state for warmup and timed trials.
Their single-trial acceptance is75/208 and87/163; these correctness fixtures
measure20.412241 and22.443641 decode tok/s and establish no speedup.

Fresh three-trial capacity control measures34.731006 decode /410.553527
prefill tok/s; rebuilt plain control measures34.761139 /409.565019, with257
identical IDs. The completed three-trial sweep retains all257 IDs for every
variant, with medians below the fresh control:

| Depth/policy | Decode tok/s | Prefill tok/s |
| --- | --- | --- |
| 1 adaptive /always | 33.987156 /34.053932 | 369.971770 /368.136477 |
| 2 adaptive /always | 32.665228 /32.874727 | 369.006941 /368.602888 |
| 3 adaptive /always | 32.214399 /29.439538 | 369.467560 /368.798539 |
| 4 adaptive /always | 30.309639 /24.220955 | 369.454096 /364.506573 |

Depth1 always accepts122/133 proposals in the first trial (91.7%), but its
two-position verification cost still exceeds the delivered plain-decode cost.
Teacher forcing lowers prefill throughput for every variant. There is **no
MTP promotion**; no candidate enters confirmation or context stress.
See [completed normalized sweep](strata-mtp-normalized-full-20261002.json),
[loader regression/build record](strata-repack-policy-20261002.json) and
[MTP progress](strata-mtp-progress-20261002.json). `resume-strata.md` records
the active staging and qualification owners; old allocation paths are expired.

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
from `tmp/strata-mhc-sync-20261002/source-v3.tar.gz`. Full-model qualification
completes on PJM52097252:128-position hidden/full-state gate passes exactly,
and all257 delivered IDs match in three timed8K trials. Fresh control measures
34.574158 decode /409.300622 prefill tok/s; fused-sync measures34.337024 /
408.123134 (ratios0.99314 /0.99712). The rebuilt legacy control remains stable.
This candidate is rejected for promotion; confirmation and context stress are
not run because no initial gain qualifies. See
[full-model mHC synchronization record](strata-mhc-sync-full-20261002.json).

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

## Small-batch mHC verification candidate (October 2 evening)

`--mhc-verify-kernel team` keeps one OpenMP team across 2–5-position mHC
verification. Each position retains the original FP64 norm partition and
four-position BF16 SVE dot chain. Coefficients are independent across tokens;
one collapsed token/dimension workshare replaces per-token teams. Residual
copies and serial RMS normalization keep their original order. Default is
`legacy`; scalar decode and larger prefill retain their existing paths.

The first isolated prototype passes exactness but regresses: extra publication
barriers erase the saved team overhead. A second prototype coalesces
coefficients/collapse workshares and improves its same-job 47-thread synthetic
medians22–33%. The integrated public selector is measured independently:
fast47 medians are73.972013→66.419442 µs for2 positions,
136.360857→116.944313 µs for4, and176.522467→147.008234 µs for5:
11.37%/16.60%/20.08% component speedups. Ninety distinct synthetic sites,
four rounds and seven alternating trials exclude model collectives and use
serial first-touch weights; these are not delivered model tok/s.

Separate one-node PJM52099051 passes14 fast/conservative ×1/3/12/23/24/47/48
configurations,672 cases each (9408 total). Every scratch byte and normalized
output matches, with finite outputs and stride/tail guards. Cases cover1–7
positions and prefill mode0/1, including single-token and larger-batch fallback.
The local warning-clean runtime parser and18 launcher tests pass.

Immutable `candidate-mhc-batch-v1` cross-build passes, with capacity4096 and
attention47. It retains the normalized MTP objects and repaired shared reader,
and rebuilds the target object. Benchmark SHA
`2127762f8e8c7440cbdc238ebcec439eebca5b2bf6493ec27e114b3726598714`.
Serial full-model qualification PID2343 follows PID597 on PJM52097252:
8049 prompt capture, short/8K fullstate, fresh/rebuilt controls, fresh legacy
MTP controls, then the depth sweep. Promotion still requires independent
confirmation and qualifying context stress. See
[native/build record](strata-mhc-batch-native-20261002.json).

The prefill MoE probe of existing L1/L2 assembly prefetch variants also passes
all210 mixed-format chain/guard cases and parallel output hashes. All three
variants regress at selected47-thread/C4096: ratios0.9661/0.8215/0.8585.
No production prefetch change is made. See
[native rejection](strata-moe-prefetch-native-20261002.json).

Reproduce the small-batch native checks and component sweep:

```bash
fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp \
    -ffast-math -fno-math-errno -Wall -Wextra -I. -I../../common \
    test_glm53f_mhc_batch.c glm53f_team.c -lm -lpthread -o test_glm53f_mhc_batch
OMP_NUM_THREADS=47 FLIB_BARRIER=HARD ./test_glm53f_mhc_batch
fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp \
    -ffast-math -fno-math-errno -Wall -Wextra -I. -I../../common \
    bench_glm53f_mhc_batch.c glm53f_team.c -lm -lpthread -o bench_glm53f_mhc_batch
OMP_NUM_THREADS=47 OMP_PROC_BIND=close OMP_PLACES=cores OMP_WAIT_POLICY=active \
    FLIB_BARRIER=HARD XOS_MMM_L_HPAGE_TYPE=none ./bench_glm53f_mhc_batch
```

## KDA column recurrence candidate (October 2 evening)

`--kda-decode-kernel columns` and `--kda-prefill-kernel columns` are separate
opt-in selectors; both default to `legacy`. A64FX512-bit SVE owns one64-value
half of each canonical128×128 head state. Four vector accumulators retain
the original chronological key FMA chains and scaled-query multiplication.
Decode computes the same scalar `expf` factors within the existing norm/decay
workshare, then dispatches10/12 column tasks for5/6 local heads. Prefill keeps
the existing factor/normalization preparation and removes packed-state copy
and unpack workshares. Small verification batches retain their existing path.
Snapshot layout and arithmetic precision are unchanged.

Integrated native PJM52100343 passes14 fast/conservative ×1/3/12/23/24/47/48
thread configurations and four benchmark prechecks:18×528=9504 exact cases.
The test compares every output/state float bit, finite IEEE representations
and state/packed/output guards over5/6 heads,12 input distributions and11
lengths through129. Parallel benchmark hashes also match. Runtime parser,
18 launcher tests, six reporting tests and warning-clean cross-build pass.

Seven alternating fast47 synthetic trials measure13.722314→7.478396 µs/token
for5-head scalar decode and13.690525→7.576413 for6 heads (1.835×/1.807×).
The **actual packed16 prefill baseline** at47 positions measures
2.358822→2.764641 for5 heads (regression), and3.276987→2.871168 for6 heads
(1.141×). At64 positions the ratios are0.835/1.127. These include factor
preparation and packing, exclude model projections/normalization/collectives,
and use synthetic serial first-touch; they do not establish model tok/s.
Requested `FLIB_BARRIER=HARD` overrides `OMP_PROC_BIND` toFALSE.

Immutable `candidate-kda-columns-v1` is built with capacity4096 /attention47.
Frozen source archive SHA
`21cabb8aa7210fb233ba0eefbfe58f4aef435ca64c5b9c7af46ac960773a2b8c`;
benchmark SHA `5f3cf4fcb7cf9522fc3322ba4ec881274c66d9c8c3735b8797a730007dd24143`.
Serial PID8008 follows owned PID2343 on PJM52097252. Decode hidden/fullstate
and all8049 prefill means/streams/fullstate gates precede fresh controls and
decode-only, prefill-only and combined measurements. Independent five-trial
confirmation and qualifying context stress precede promotion. No KDA setting
is promoted. See [native/build record](strata-kda-columns-native-20261002.json).

Reproduce the native recurrence checks after an integrated build:

```bash
OMP_NUM_THREADS=47 FLIB_BARRIER=HARD ./test_glm53f_kda_columns
OMP_NUM_THREADS=47 FLIB_BARRIER=HARD ./test_glm53f_kda_columns --bench
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
--moe-prefill-layout tight|padded
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

## Prepared native weight probes (October 2 evening)

Strata's prepared CPU weights and budgeted expert caches motivate two bounded
native probes. Neither changes the selected model recipe. Reusing exact
expanded int8 panels first regresses with serial placement/repeated lookup.
Parallel first touch and one lookup per GEMM improve warm direct reuse:
fast47/C4096 original23.120880→21.759033 ms (1.063×), fast48/C4096
22.960901→21.479845 ms (1.069×), with full-output hashes exact. Mixed-format
chain/output guards pass840 cases across the two probes.

The synthetic layer costs720MiB and about62ms to prepare. At47 threads,
roughly46 chunk uses are needed to repay preparation; the8049 request has
only two outer chunks. Forty-two synthetic layers would need29.5GiB of extra
HBM on top of resident weights. The bounded warm component gain does not
justify a model-wide expanded cache. See
[panel-cache native evidence](strata-moe-panel-cache-native-20261002.json).

A smaller sidecar keeps native Q4/Q5 payload and prepares only scale/minimum
metadata. Its5880 native cases and output hashes pass exactly. At47 threads,
warm4096-column kernels improve1.23–1.24×, while shorter-column gains are
smaller. The24-byte sidecar adds16.7% Q4 or13.6% Q5 bytes. A192-matrix
streaming probe passes6144 exact matrix cases, but47-thread gate/up gains
fall to4–13%; two-block Q5 down regresses4%. The16-byte layout passes7560
unit cases and12288 streamed matrix cases, but both two-block down shapes
regress about9%. Prepared metadata remains outside the model. A direct native
byte-extraction probe checks whether fewer live vector constants can help
without a sidecar. It passes5880 primitive and6144 streamed matrix cases;
Q5 component ratios at47 threads are1.057/1.061 for4096-column gate/up,
1.039/1.039 for256/512-column down. Q4 stays on the original path.

`--moe-scale-kernel words`, default`legacy`, now exposes this Q5 extraction in
eligible scalar routed+shared decode. Native integrated PJM52103308 passes8400
primitive cases across ordinary/persistent teams,400 mixed Q4/Q5/Q6 expert
chains and14200 paired-row comparisons. Two math modes and threads1/3/12/47/48
pass, as do local parser,18 launcher and six reporting checks. New primitive
and diagnostic tools keep-Werror; only pre-existing GLM5 graph warnings are
suppressed in its bridge/grouped translation units. The initial failed build
log is preserved. Immutable candidate-iq-scale-words-v2 has explicit128-step
executor and8049 prompt-hidden/full-state gates before fresh/rebuilt controls.
PID16341 waits for serial MPI diagnostics on the current allocation. No new
model gain is claimed. See [scale extraction evidence](strata-iq-scales-native-20261002.json).

A separate12-rank slab diagnostic waits for the mHC-batch and KDA campaigns.
It compares every FP32 result against the legacy512-token MPI sum over six
input distributions and three chunk lengths before timing eligible slabs.
A nonblocking pipeline probe retains the original512-token boundaries and
compares windows1/2/4/8 against blocking MPI across90 cases. Only fully exact
candidates enter seven rotating64MiB timing trials. PID16162 follows the slab
probe PID9686. No collective size or algorithm has changed in the model. See
[MPI diagnostic queue](strata-mpi-progress-20261002.json). The first pipeline
queue had a prerequisite-name typo; overlap guards stopped both new drivers
before MPI. Corrected owners16162/16341 are verified waiting on their expected
predecessors, with the original refusal logs preserved.

## Remaining architecture decision

The measured TP12 candidate remains below both targets. PP3×TP4 now connects
owned constructors/stagers, layer ownership, stage-local collectives and
four-stream handoffs. Native components, slicing and distribution fixtures
pass, but real-model cross-layout output first differs atID index19, so PP
is not qualified. Exact component/virtual-TP12 isolation results and the
remaining early-layer replay are recorded in`resume-strata.md` and
`strata-cross-layout-isolation-20261003.json`. For a single dependent decode
request, account for inter-stage latency rather than assuming threefold
pipeline speedup. The promoted TP12 recipe passes short/synthetic32K checks;
this does not qualify the new MTP feature at32K or replace a real long coding
prompt. All new paths remain opt-in.

The completed mHC verification-team full sweep passes captured8049 prompt
means and short128/8K target state, with every257-ID output exact. Fresh plain
control34.455058 /408.171302 tok/s outperforms all eight team+MTP variants.
Teams recover only1.69%/0.92% versus fresh legacy MTP depths2/4; promotion
is rejected. See [full mHC team results](strata-mhc-batch-full-20261002.json).
KDA decode128 and8049 prompt-hidden/full-state gates also pass; its fresh
control and candidate timing comparisons continue onPJM52097252.

`--mla-softmax-kernel parallel`, default`legacy`, revisits scalar native MLA
softmax. Its original prototype failed because forced sequential summation
changed FCC fast-math bits. The new path retains scalar expf and compiler
sum policy, distributes64-key tasks, then computes per-head sums. Integrated
native17820 cases pass for each logit stride2052/2056 across two math modes,
five thread counts, nine inputs and33 token/tail counts. At47 threads and
production stride2052, five/six-head timings improve2.654×/2.421×, with
seven alternating90-step component trials including input restoration.
The opt-in sparse worker adds8-float maximum scratch and retains legacy for
small<128 or one-thread calls. Wide prefill softmax is unchanged. Explicit
reference/candidate flags in the executor and captured prompt/state checker
prevent same-candidate comparisons. Strict-Werror integrated build, parser,
18 launcher and six reporting checks pass. Candidate-mla-softmax-v1 waits
behind Q5 in the serial qualification queue asPID19483. No model gain is
claimed. See [softmax native evidence](strata-mla-softmax-native-20261002.json).
