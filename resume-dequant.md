# Goal: finish Qwen3.8 rank-local k-quant decode integration

Continue the Qwen3.8-27B UD-Q4_K_XL A64FX work from the verified Q5R/IQ4R
layouts and rank-local sidecar builder to a safe, measured `tp_runner` decode
path. The finished path must preserve compact weights for prefill and fallback,
load only the local rank's decode sidecar, retain CMG-local row ownership, and
pass exact 128- and 256-token greedy-output gates before promotion.

Do not reset the branch or discard unrelated work. Do not push without explicit
permission in the current user request.

## Definition of done

The goal is complete only when all of the following are true:

1. A real rank-local `Q38TP` stage can be planned and converted to a versioned
   `Q38KQC1` sidecar with bounded memory and I/O.
2. The TP loader validates source identity, layout version, shapes, offsets,
   entry-table hashes, and payload hashes before making a cache entry usable.
3. Decode dispatch uses Q5R/IQ4R only for validated compatible tensors and
   falls back to the unchanged compact kernel for every unsupported, missing,
   invalid, or tail case. Prefill continues to use its required compact form.
4. Cache pages are first-touched or mapped consistently with the persistent
   pool's row/CMG ownership; the eight-row scheduler neither crosses ownership
   boundaries nor drops tail rows.
5. Compact and cached paths produce identical greedy token hashes for the same
   prompts at 128 and 256 generated tokens. Q5R's small floating-point
   accumulation differences are not grounds to waive this token gate.
6. A clean-node run records per-rank stage sizes, peak/steady `MemAvailable`,
   sidecar load time, total decode tok/s, and compact-versus-cache performance.
7. `qwen-q8.md` contains exact commands and representative correctness,
   memory, and performance output, and the focused work is committed without
   unrelated files.

## Verified starting point

The initial Q5 layout benchmark is commit `b383fc0b`, the shared exact-cache
milestone is `7d37cd3c`, the rank-local sidecar builder milestone is
`8cf312cf`, and strict real-sidecar validation is `cfcdf019`. The current
branch may advance beyond those commits; they are landmarks, not reset
targets.

Relevant files:

- `a64fx/llm/bench_qwen38_kquants.c`
- `a64fx/llm/kquant_decode_cache.h`
- `a64fx/llm/test_qwen38_kquant_cache.c`
- `a64fx/llm/qwen38_kquant_stage.[ch]`
- `a64fx/llm/qwen38_kquant_load.[ch]`
- `a64fx/llm/test_qwen38_kquant_stage.c`
- `a64fx/llm/qwen38_tp_stage.h`
- `a64fx/llm/Makefile`
- `qwen-q8.md`, `Q5_K row-interleaved decode probe` and child sections

The real model used for the isolated benchmark is:

```text
/home/u14346/models/qwen38/27b/Qwen3.8-27B-UD-Q4_K_XL.gguf
```

Q5R interleaves eight rows by 256 columns, expands Q5 values once, and retains
the original affine scales/minima. It occupies 280 bytes per original
row/block versus 176 bytes for Q5_K. IQ4R expands the original nonlinear
palette exactly and occupies 272 bytes per row/block versus 136 bytes for
IQ4_XS. Both layouts have version 1.

Validated real layer-0 results at 2.0 GHz and 48 threads:

| Projection | Compact A8 | Row-interleaved | Speedup/effective BW |
|---|---:|---:|---:|
| Q5_K up, 17408 x 5120 | 1.100 ms | 0.244 ms | 4.5x / 251.5 GB/s |
| Q5_K down, 5120 x 17408 | 0.881 ms | 0.265 ms | 3.3x / 230.9 GB/s |
| IQ4_XS gate, 17408 x 5120 | 0.466 ms | 0.213 ms | 2.2x / 222.3 GB/s |

IQ4R is bit-identical to native A8 over wave, sparse, high-dynamic-range, and
deterministic random inputs. Q5R retains native-A8 NRMSE; observed differences
from native A8 were `2.83e-8`--about `1e-6` normalized RMS depending on shape
and pattern, with a worst recorded absolute difference of `1.53e-5`.

The metadata scan reports:

```text
model tensors=866 tensor_bytes=17.912GB Q5_K=325/12.936GB Q5R_eligible=325/20.581GB Q5R_delta=7.644GB projected=25.557GB
IQ4_XS=65/3.078GB IQ4R_eligible=65/6.155GB IQ4R_delta=3.078GB Q5R_IQ4R_projected=28.634GB
```

A combined single-node replacement leaves only about 3.4 GB before runtime
state, and an additive cache cannot fit. The selected design therefore builds
a separate sidecar from each already-sharded compact `Q38TP` rank file.

`qwen38_kquant_stage` writes `rankNN.kquant` through a PID-qualified partial
file and atomic rename. Its header records rank/size, source stage identity,
layout version, tensor metadata, and source/cache checksums. Conversion holds
only one compact tensor and its packed result at a time, uses positioned I/O,
and calls `POSIX_FADV_DONTNEED`. `--plan` performs no output writes. Valid
sidecars are reused; inconsistent header metadata or hashes cause a rebuild.

The bounded synthetic stage test verifies plan/build/reuse, exact Q5R/IQ4R
payload bytes, all relevant hashes, and rebuild after deliberate header
corruption:

```text
SENTINEL qwen38_kquant_stage=OK entries=2 q5r=2240 iq4r=2176 reuse=1 corrupt_rebuild=1 loader_rejects=9
```

The independent read-only loader validates the compact source identity, fixed
headers, entry-table hashes, unique names, type/format pairs, local shapes,
monotonic non-overlapping extents, source checksums, and every payload hash.
It uses a fixed 1 MiB hashing buffer, evicts validation reads, and maps the
sidecar only after every entry passes. Tests require rejection of bad magic,
version, layout, truncation, offset, duplicate name, source entry, source file
size, and payload data. The runtime attachment described below now wires this
loader into the TP runner and persistent-pool dispatch without weakening those
checks.

No real compact `rank00.blob` was present under `/local/u14346` during the
sidecar-builder milestone. Later bounded runs built and checked the real
mixed-Q4 artifact on all four TP4 nodes. All ranks have 866 compact entries, a
5,757,905,920-byte compact file, 390 cache entries, and a 7,809,826,816-byte
sidecar. Job 51815999 validated every payload in 31.056--32.304 seconds per
rank while leaving `MemAvailable` at about 31.3--31.5 GB after staging.

The decode-only runtime attachment is now implemented. `tp_runner` accepts
`--kquant-stage DIR`; every rank validates its local compact identity and all
sidecar hashes, then votes before retaining the mapping. Cache pointers are
attached only after compact prefill and a second all-rank vote. Any validation
or attachment mismatch coherently returns all ranks to compact dispatch. The
persistent pool statically partitions complete eight-row groups, while partial
or tail extents explicitly use the compact path. File-backed demand paging was
pathologically slow in four-node acceptance, so after strict validation each
worker now `pread`s its own groups into a read-only anonymous arena and evicts
the source pages. This makes the sidecar resident before decode and preserves
CMG-local first touch.
Synthetic coverage includes uneven three-worker ownership, unaligned ranged
dispatch, a 15-row compact fallback, invalid format rejection, selective
IQ4-only anonymous materialization with zero Q5 pages, and model attach/detach.
Fujitsu builds of the focused test and full runner pass. The
full Q5R+IQ4R TP4 path reached 25.40 tok/s versus 11.73 compact and matched 128
tokens, but failed the 256-token gate at token 17. An exact Q5R scheduling
experiment was slower than compact and was rejected. Q5R is therefore skipped
by default and available only through explicit `--kquant-q5`; the default
validated sidecar path materializes and attaches only exact IQ4R entries.

## Constraints and safety rules

- Read `AGENTS.md` and this entire file before changing code.
- On the current native A64FX node, work directly in this Git checkout. Use
  repository-local `tmp/` only for disposable compiler scratch or synthetic
  stage data; do not redirect this work through `/local` or `/tmp`.
- Never `cat` or make an interactive full copy of a multi-gigabyte model.
  Keep model and stage access lazy or bounded, use positioned/chunked I/O, and
  discard page cache as work progresses.
- Run `qwen38_kquant_stage --plan` and establish a per-rank compact + cache +
  KV/state/scratch budget before building or loading real sidecars.
- Monitor `MemAvailable`; do full-load speed work detached or in a batch job.
- Production tuning selection must be a runner argument, not a new production
  environment-variable switch. Environment variables may remain diagnostics.
- Preserve compact prefill and correctness fallbacks. Never silently accept a
  cache whose source identity, format, dimensions, offsets, or checksum fails.
- Preserve all unrelated dirty-worktree files. In particular,
  `common/transformer.h` already contains unrelated user changes. Inspect its
  diff before editing it and keep this work separable; if runtime attachment
  cannot be isolated safely, stop and ask rather than overwriting or staging
  those changes.

## Remaining work, in order

### Completed on one node

The five previously listed single-node tasks are complete. The native
48-thread sweep covered all 65 real IQ4_XS tensors under all four activation
patterns: 260/260 compact-A8 versus IQ4R outputs were byte-identical. Aggregate
best-of-three time was 113.246 ms compact versus 55.318 ms IQ4R (2.047x), with
222.5 GB/s effective compact bandwidth; the slowest IQ4R observation was
`blk.38.ffn_gate.weight`/random at 0.257 ms. All tensors had the eligible
17408-by-5120 shape; the separate synthetic 15-row test retains tail-fallback
coverage.

The bounded loader matrix now rejects 11 invalid cases, including missing and
unsupported sidecars, and asserts completely empty loader state after every
rejection. An 8 MiB IQ4 plus 8 MiB skipped-Q5 materialization test records
time, detach time, exact resident-byte accounting, and `MemAvailable`; it
materializes exactly one IQ4 entry and verifies every skipped Q5 byte remains
zero. The benchmark, cache test, stage builder/test, checker, and full
`tp_runner` all build with native `fcc`; the fresh synthetic stage and checker
pass. Commands and representative output are recorded in `qwen-q8.md` under
“Single-node IQ4R model sweep and fault acceptance.”

The exact 128/256-token TP4 hash gate, an injected all-rank fallback vote, and
clean four-rank throughput remain inherently multi-node and are listed below;
they cannot be accepted from a single-node run.

### Work that still requires four nodes

1. **Run the IQ4-only correctness gate on TP4.** Re-run
   `a64fx/llm/pjsub_qwen38_q4_kquant_tp4.sh` on four clean nodes. The updated
   job explicitly keeps `TP_KQUANT_Q5=0`, labels results `cached_iq4_128` and
   `cached_iq4_256`, and requires byte-identical token files at both lengths.
   Confirm each rank reports `prepared 65 tensors` and post-prefill
   `decode attach OK entries=65`.

2. **Complete fallback acceptance.** Exercise one missing/corrupt sidecar on a
   disposable synthetic or staged copy and confirm the all-rank vote selects
   compact fallback. Do not corrupt the validated rank artifacts.

3. **Keep Q5R quarantined.** The fast layout is useful only as an explicit
   diagnostic until a new implementation is both faster than compact and
   passes exact 128/256-token hashes. Do not waive the failed token gate or
   promote the slower exact scheduling experiment.

4. **Measure and document the promoted IQ4 path.** Record its sidecar-load
   time, peak/steady memory, forward/decode timing, total tok/s, and per-rank
   resident bytes. Add representative output to `qwen-q8.md`, run
   `git diff --check`, commit only focused follow-up files, and report the hash.
   Do not push.

## Revalidation commands

```sh
mkdir -p tmp/dequant
TMPDIR="$PWD/tmp/dequant" \
  make -B -C a64fx/llm \
  qwen38_kquant_bench qwen38_kquant_test \
  qwen38_kquant_stage qwen38_kquant_stage_test qwen38_kquant_check \
  CC=fcc OPENMP=1

OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./a64fx/llm/build/test_qwen38_kquant_cache

OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./a64fx/llm/build/test_qwen38_kquant_stage \
  ./a64fx/llm/build/qwen38_kquant_stage \
  "$PWD/tmp/dequant/kquant-stage-test"

Q38TP_RANK=0 Q38TP_SIZE=4 \
  ./a64fx/llm/build/qwen38_kquant_check \
  /path/to/qwen38-q4-tp4 \
  /path/to/qwen38-q4-tp4-kquant

OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
  numactl --interleave=all ./a64fx/llm/build/bench_qwen38_kquants \
  /home/u14346/models/qwen38/27b/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  7 blk.0.ffn_gate.weight wave
```

For an existing real compact stage, substitute its actual directories and
rank count only after verifying them:

```sh
Q38TP_RANK=0 Q38TP_SIZE=4 ./a64fx/llm/build/qwen38_kquant_stage \
  --plan /path/to/ACTUAL_COMPACT_STAGE \
  /path/to/ACTUAL_KQUANT_STAGE
```

## Resume prompts

### Current task: W8A16 / W8A32 at 220--230 GB/s

The user requested FP8 and signed INT8 weights with FP16/FP32 activations
and matching FMA accumulation, explicitly including both E4M3FN and E5M2.
Work directly on this native A64FX node; keep scratch/logs in `tmp/dequant/`.
This is the active single-CMG kernel task, not the separate Qwen TP4 goal.

Implemented and numerically verified all six combinations in
`a64fx/dequant-pipe/w8.S`, `w8.h`, and `bench_w8.c`. Fixed K=128, K-major
byte weights `[K][N]`, N=256 for FP16 and N=128 for FP32, shared activations,
sequential K-order FMA, no scales/tails or activation quantization. FP8
subnormals/specials are supported; no flush-to-zero is selected.

Three-launch medians (GB/s):

| weights | FP16 accumulation | FP32 accumulation |
|:--------|:------------------|:------------------|
| INT8 | 229.04 / 230.30 / 229.17 | 229.37 / 229.37 / 228.67 |
| E4M3FN | 143.20 / 143.19 / 143.18 | 92.25 / 92.24 / 92.24 |
| E5M2 | 116.89 / 116.88 / 116.87 | 200.52 / 200.46 / 200.44 |

**INT8 passes; FP8 target is still unmet.** Every run qualified on paired
reads (227.63--229.77 GB/s), 2 MiB pages, NUMA node 4, CPUs 12--23 at 2 GHz,
FPCR=0. Logs: `tmp/dequant/w8-acceptance.gFdxIs/`.
`run_w8_acceptance.sh` requires all eighteen medians >=220 GB/s and correctly
returns failure for twelve FP8 misses. Do not lower the gate or omit
subnormal weights to claim success.

`make -C a64fx/dequant-pipe test CC=fcc` passes all new exhaustive code tests
and existing W4/full-range INT16 regressions. Use `TMPDIR="$PWD/tmp/dequant"`.
The verifier isolates special codes (including E5M2 infinities), tests
sequential random/cancellation/overflow cases, and requires finite bit
identity and NaN classification. `--normal-weights` is a labeled diagnostic,
not acceptance: the native kernel reached 203.33 GB/s, versus 16.13 on all-finite
weights. FP32 was nearly unchanged, isolating a large subnormal-sensitive
FP16 arithmetic cost in the original native pipeline; no assist PMU counter
was collected. The selected guarded rescaling now raises all-finite FP16 to
116.87--116.89 GB/s. It uses integer corrections to multiply subnormal weights
by four, and divides the associated activation lanes by four only when that
is exact and remains normal. A tile-wide activation exponent check (>=3 for
finite nonzero values) plus FPCR=0 precondition protects the transform;
otherwise it calls `w8_e5m2_f16_native`. Native/scalar bit comparisons include
both sides of the guard, zero, infinities, and tiny values. No product or
FMA-rounding change is permitted.

Continue with lower-cost exact E4M3 conversion and E5M2 FP32 conversion
scheduling; investigate E5M2 FP16 subnormal handling without weakening its
numerical contract. Current E4M3 FP16 uses a wrapped correction table;
FP32 uses parallel byte tables producing exact BF16 bits, then zero-extends
into FP32 bits. INT8/E5M2 FP32 and the native FP16 fallback use cross-K prologue/drain
pipelines; guarded FP16 rescaling is the selected E5M2 path.
See RESULTS.md for rejected schedules, instruction counts, all commands,
and source snapshots. The current exact FP8 paths must remain experimental
until they meet the performance gate. Keep the accepted W4 paths passing,
preserve unrelated edits, commit focused milestones, and do not push.


### A64FX W4A16 fused-kernel follow-up (2026-09-20)

Both requested single-CMG targets now pass for INT4 and E2M1 FP4. Three fresh
launches (240 MiB packed weights, 12 cores, ten iterations, five trials) gave:

| path | INT4 median range, GB/s | FP4 median range, GB/s |
|:-----|-----------------------:|----------------------:|
| full-range SDOT, `int16x8-full/opt` | 219.44--220.09 | 214.23--215.45 |
| sequential FP16, `fp16/opt2` | 171.84--172.04 | 172.08--172.18 |
| native INT16 SDOT, `int16/opt` | 201.46--201.58 | 173.21--173.36 |

The full-range route uses `a = lo + 256*hi + 128` and a 256-byte signed
weight-sum trailer per 8192-byte supertile. Metadata reads and correction are
timed; packed GB/s excludes metadata from its numerator. All INT16 inputs are
representable. FP16 table decoding eliminates integer widening/conversion
and preserves sequential FMA rounding bit-for-bit.

Correctness covers all 65,536 INT16 values and both formats, signed extremes,
random weights, constant nibble codes, and scalar/original FP16 comparisons.
The prior radix-256 rejection at 119.92 GB/s was placement-confounded:
affinity before XOS startup raised the unchanged bounded kernel above
200 GB/s. The acceptance script now records page/NUMA backing and brackets
each kernel with reads on the same allocation; all eighteen runs qualified.

Reproduce with `bash a64fx/dequant-pipe/run_w4a16_acceptance.sh`.
Recorded logs: `tmp/dequant/w4a16-acceptance.ByB7iv/`; details are in
`a64fx/dequant-pipe/RESULTS.md` and `a64fx/doc/fused-dequant.md`.
These are unscaled M=1 kernel rates. Production scale epilogues, arbitrary
tails, model integration, and wider FP accumulation remain separate work.

### Completed task: FP16 FMA above 200 GB/s

Completed natively on 2026-09-20. The selected `--path fp16 --kernel f16pipe`
uses a genuine cross-K pipeline with alternating current FMA / next TBL pairs,
preserving the original layout and ascending-K FP16 rounding bit-for-bit.
Three fresh launch medians (GB/s): INT4 **204.07 / 204.13 / 204.11**;
FP4 **204.02 / 204.07 / 204.03**. Paired reads: 227.96--229.77 GB/s;
2 MiB pages, NUMA node 4, CPUs 12--23, all at 2.0 GHz.
The raised >200 GB/s FP16 and full-range SDOT gates pass in all launches:
`acceptance=PASS qualified=18/18 SDOT=6 FP16=6 failed_targets=0`.

Reproduce with `TMPDIR="$PWD/tmp/dequant" make -C a64fx/dequant-pipe test CC=fcc`
and `bash a64fx/dequant-pipe/run_w4a16_acceptance.sh`.
Tests include scalar/original bit-exact FP16 and exhaustive full-range INT16.
Raw acceptance: `tmp/dequant/w4a16-acceptance.s5EThb/`.
`RESULTS.md` records all medians, commands, and rejected schedules: putting
all lookups at the end reached ~198 GB/s; reducing activation pointer updates
regressed to ~187 GB/s. Both had qualified placement. Original kernels remain
available; no layout changes or 64-byte-load experiment was needed.
Production scales, tails, model integration, and wider accumulation remain
separate future work. The following is the completed task's original brief.

```text
Continue optimizing the A64FX fused W4A16 FP16-FMA kernel toward more than
200 GB/s of packed weight input for BOTH signed INT4 and E2M1 FP4 on one
12-core CMG. Work directly on the native A64FX node in this Git repository.
Read AGENTS.md, this resume file, and the focused sources/docs first; inspect
git status/diffs and preserve unrelated edits. Use repository-local
tmp/dequant/ for scratch and logs, never /tmp or /local. This is single-node
kernel work; do not resume the unrelated Qwen TP4 task below.

Starting point: commit a6d8c053, "Reach A64FX W4A16 SDOT and FP16 bandwidth
targets". Both original targets are accepted: full-range SDOT reaches
219.44--220.09 GB/s for INT4 and 214.23--215.45 for FP4; FP16 reaches
171.84--172.04 and 172.08--172.18 respectively. The new task raises the FP16
target from 150 to 200 GB/s. More than 200 GB/s for FP16 is plausible but has
NOT been demonstrated.

Read a64fx/dequant-pipe/kernels_opt.S, bench_fused_sdot.c, fused_opt.c/.h,
run_w4a16_acceptance.sh, README.md, RESULTS.md, and
a64fx/doc/fused-dequant.md. The selected FP16 path is --path fp16 --kernel
opt2; opt is the one-K comparator, and super is the original conversion
kernel. The fixed SVE512 layout interleaves four K=128, N=64 blocks. Each
K/block consumes 32 bytes; low/high nibbles encode columns 0--31/32--63.
LD1B .h widens bytes, register AND and LSR extract codes, TBL .h produces
exact FP16 weight bits, and FMLA .h updates eight output accumulators.
FP4 uses the doubled E2M1 lattice with the eventual scale factor left to the
caller. Existing benchmarks omit model block scales.

Preserve ascending-K sequential FP16 FMA rounding BIT-FOR-BIT. Do not use
reassociated partial sums, quantize activations, change accumulation type,
or clip inputs to claim the target. Keep the accepted SDOT paths correct.

Attack in this order:
1. Implement a genuine cross-K software pipeline. The current opt2 kernel
   mainly unrolls two steps using separate temporary registers; it does not
   explicitly interleave next-step loads/lookup with current-step FMA.
   Preserve prologue/drain correctness and the per-output K order.
2. Reduce activation-load/address overhead by loading upcoming activations
   early and using offsets to reduce pointer updates. Inspect the actual
   instructions and register allocation; avoid spills and indexed-FMA
   substitutions that cost more on A64FX.
3. If still short, compare full 64-byte packed loads and alternative nibble
   extraction/layout schedules. Keep any changed layout explicit and include
   its correct packer/reference; never benchmark one layout using another's
   interpretation. Preserve original kernels as comparison paths.

The current loop has about 48 arithmetic instructions per 256 packed bytes,
including 16 table lookups. The previous arithmetic-only roofline retained
about 230 GB/s at 48 operations, but this does NOT prove the mixed lookup/FMA
loop can do so: TBL pressure, load latency, and dependencies must be measured.
Use instruction/port analysis and native measurements to choose schedules.

Validate each candidate against scalar half-precision FMADD and the original
kernel using the existing bit-exact tests: fractional inputs, signed zero,
subnormals, cancellation, and overflow. Keep the exhaustive full-range SDOT
checks passing. Build/test with:
  TMPDIR="$PWD/tmp/dequant" make -C a64fx/dequant-pipe test CC=fcc

Benchmark with startup affinity established BEFORE XOS initialization:
  taskset -c 12 env \
    LD_PRELOAD=/opt/FJSVxos/mmm/lib64/libmpg.so.1 \
    XOS_MMM_L_HPAGE_TYPE=hugetlbfs XOS_MMM_L_HUGETLB_SZ=2M \
    XOS_MMM_L_HUGE_MALLOC=1 XOS_MMM_L_FORCE_MMAP_THRESHOLD=1 \
    XOS_MMM_L_HUGETLB_FALLBACK=0 \
    ./a64fx/dequant-pipe/bench_fused_sdot \
    --format int4 --path fp16 --kernel opt2 \
    --cores 12 --core-base 12 --mib 240 --iterations 10 --trials 5 \
    --paired-baseline --compare-kernels
Repeat for --format fp4 and substitute the candidate selector as needed.
Record actual page size, NUMA placement, and CPU frequency. The accepted
baseline used 2 MiB pages, NUMA node 4, CPUs 12--23, and 2.0 GHz.

Acceptance: both formats must exceed 200 GB/s MEDIAN packed-weight bandwidth
in each of three fresh launches, five timed trials per launch. Require
same-allocation read controls before/after to exceed 220 GB/s; retain and
label all slow-placement runs. Do not mistake logical Gweight/s or best-only
measurements for acceptance. The old roughly 120 GB/s radix-256 rejection
was caused by placement; startup pinning resolved it.

Extend run_w4a16_acceptance.sh for the new FP16 candidate and >200 GB/s gate.
Its existing FP16 >150 GB/s PASS is only the old regression threshold.
Keep full-range SDOT >200 GB/s checks. Previous raw acceptance logs are in
tmp/dequant/w4a16-acceptance.ByB7iv/.

Once measured, update RESULTS.md, README.md, the fused-dequant blog, and this
resume file with commands, numerical guarantees, all launch medians, and
rejected schedules with controlled evidence. If the target remains unmet,
state the best validated result and measured bottleneck explicitly. Run
git diff --check, commit only focused changes, report the hash, and do not
push.
```

### Separate Qwen TP4 task (not the current single-node objective)

```text
Continue the active goal in resume-dequant.md: finish and accept safe rank-local
Q5R/IQ4R decode for Qwen3.8 on A64FX. Read AGENTS.md and the whole goal file
first, inspect git status/diffs, and preserve every unrelated dirty-worktree
change. The exact layouts, kernel tests, versioned Q38KQC1 sidecar builder,
strict loader, explicit `tp_runner --kquant-stage DIR` attachment, all-rank
fallback votes, decode-only activation, and persistent eight-row ownership are
implemented. Job 51815999 physically built and hash-validated all four TP4
ranks. The full Q5R+IQ4R path matched 128 tokens and reached 25.40 tok/s versus
11.73 compact, but failed the 256-token gate at token 17. The exact Q5R
scheduling experiment was slower than compact, so Q5R is now explicit
diagnostic opt-in only. Revalidate the focused tests, then continue at the
first unfinished item: run the updated IQ4-only 128/256-token gate in a fresh
four-node allocation and record its `prepared 65 tensors`/`decode attach OK
entries=65` diagnostics, memory, load time, and throughput.

Do not create a full single-node additive cache. Plan real per-rank memory
before conversion, use bounded I/O and repository-local tmp/ scratch, preserve
compact prefill/fallback, and verify the `prepared`/post-prefill `decode attach
OK` diagnostics on every rank. common/transformer.h may retain unrelated user
edits, so never overwrite or stage them. Require exact 128/256-token compact
versus IQ4R greedy hashes plus clean-node sidecar-load, memory, bandwidth, and
tok/s evidence. Keep Q5R disabled unless a faster exact implementation passes
the same gates. Update qwen-q8.md, commit only focused files, report the commit
hash, and do not push.
```
