# Qwen3.8-27B FP4 / true FP6 decode: checkpoint and resume prompt

Updated: 2026-09-24 JST, after wrapping up native allocation 51893515.
This replaces the earlier expanded-FP4 handoff. Root `resume.md` concerns
unrelated work; use this file for the low-bit decode task.

## Current result and acceptance boundary

The compact FP4 path now runs the real model on one 48-core A64FX node.
The latest measured serial N=1 decode is **8.675 tok/s at 1024 input + 256
generated tokens**, and **9.693 tok/s at 128 + 256**. These are single trials,
not stable repeated throughput results. **The FP4 40+ tok/s and true FP6
30+ tok/s targets have not been reached.** Full-model FP6, held-out quality,
and unapproximated greedy-reference validation remain open.

The user-approved scope is:

- One 48-core A64FX node, including the untimed BF16 oracle. No multi-node
  reference and no speculative decoding for the throughput gates.
- FP4 >= 40 tok/s; true FP6 E2M3 >= 30 tok/s. The primary workload is
  1024 input + 256 generated tokens; sensitivity cases are 128 and 4096.
- FP6 is quantized from BF16. The older `packed6` expanded-FP4 payload is
  a separate format and is not a true FP6 result.
- Held-out perplexity and BF16-logit error must be no worse than existing
  NVFP4. Optimized execution must match every generated position/token ID
  from an unapproximated reference for the same weights.
- Bounded direct-to-CMG loading, a reusable final model image, shared
  activation preparation, and a one-node layer-streamed BF16 oracle.
- Simulator acceptance requires matched objects/settings, two native
  allocations, at least ten samples, CV/drift <= 2%, and < 3% cycle and
  bandwidth error. Effective source bytes and FAPP physical traffic are
  separate measurements.

## Implemented checkpoint

All new code lives under `a64fx/llm/`, with opt-in dispatch hooks in
`common/transformer.h` and CLI integration in `qwen38_runner.c`.

| Component | Files / behavior |
| --- | --- |
| Formats and portable reference | `qwen38_lowbit.h`, `.c`: raw NVFP4 repack, BF16-to-FP6 E2M3 conversion, scalar activation preparation and reference math |
| SVE kernels | `qwen38_lowbit_sve.c`, `qwen38_lowbit_scale.inc`: 512-bit SVE, 8 output rows, code lookup + SDOT, FP32 scale/accumulation; F32 activation dequant/FMA path |
| Loader and image | `qwen38_lowbit_model.h`, `.c`, `qwen38_lowbit_image.inc`: bounded conversion, CMG-local anonymous storage, descriptor registry, validated cache image |
| Correctness and measurements | `test_qwen38_lowbit.c`, `test_qwen38_lowbit_model.c`, `bench_qwen38_lowbit.c`, `bench_qwen38_lowbit_prepare.c` |
| Quality prototype | `qwen38_lowbit_eval.c`, `compare_qwen38_lowbit_quality.py`, `test_qwen38_lowbit_quality.py` |

FP4 stores 512 weights in 288 bytes including UE4M3 scales: **4.5 total
bits/weight**, with a lossless repack of source NVFP4. True FP6 uses packed
low-four/high-two-bit planes and E8M0 block-32 scales: 400 bytes per 512
weights, **6.25 total bits/weight**. E2M3 finite magnitudes span 0.125 to 7.5;
conversion handles round-to-nearest-even, signed zero, saturation and tails.
Nonfinite BF16 values are rejected.

A8 activation blocks use symmetric max 127. A16 uses centered radix 256,
`q = lo + 256 * hi`, max 32639; per-block accumulation stays within INT32.
The portable double reference and F32 activation path remain available.
A8/A16 quantization is an approximation: kernel tests do not establish
full-model greedy equivalence to F32 activations.

SVE activation preparation shares a double reciprocal per 32-element block
and corrects the residual at half ties. It matches scalar preparation
byte-for-byte on tested tails, half ties and finite exponent extremes.
Preparation is shared explicitly across workers with begin/end barriers;
there is no pointer-address cache heuristic.

The loader uses <= 8 MiB source reads, <= 16 MiB BF16 scratch and four final
anonymous CMG segments. It pins first touch to CPUs 12/24/36/48, binds nodes
4..7 and samples placement every 2 MiB. Fugaku needs `mbind` maxnode 64,
not 8. The runner forces metadata-only source mmap to prevent GGUF from
materializing a second source copy under NUMA distribution. The current
reserve is 6 GiB; the measured v4/v5 runner object used the prior 4 GiB
reserve and had ample observed headroom. Restore pointers/affinity on
failure; keep the GGUF context alive until the low-bit model is freed.

The v1 cache has canonical metadata, source inventory/file-size/mtime
identity, per-payload 64-bit checksums, bounded writeback and atomic rename.
It rejects stale/truncated/corrupt images without silently converting again.
This detects accidental corruption; source identity is not a cryptographic
content hash. Raw unsupported weights are loaded once; the Q6_K output
head remains on the existing path. Auxiliary `nextn.`, `v.`, `mm.`, `mtp.`
weights stay lazy, but the `blk.64` NextN weights are still loaded/converted.

CLI: `--lowbit fp4|fp6 --lowbit-activation f32|a8|a16` (default F32), with
optional `--lowbit-image PATH` or `--lowbit-write-image PATH`, mutually
exclusive. The new runner rejects legacy repacks, speculative/batched
modes, approximate SiLU and partial-head settings. NUMA mode requires 48
threads; `--lowbit-no-numa` is for portable validation. KV uses F32.

## Native A64FX results

Allocation **51893515**, host **a25-2201c**, one 48-core node, 2 GHz,
`eco_state=0`. Fujitsu `fcc -Nclang`, SVE enabled. The job was explicitly
released on wrap-up; its shell and allocation-local paths cannot be reused.

The same repeated sky-blue seed, 256 generated tokens, one trial and no
warmup were used for each row:

| Runner | Main change | 128-input tok/s | 1024-input tok/s |
| --- | --- | ---: | ---: |
| v3 | Scalar activation preparation; experimental fully unrolled kernel | 3.920 | 3.750 |
| v4 | SVE preparation with vector FP64 division; LUT kernel | 8.901 | 8.054 |
| v5 | Shared reciprocal with exact residual correction; LUT kernel | **9.693** | **8.675** |

All **512 generated (position, token ID) pairs and selected logit bits**
match v3 versus v4 and v3 versus v5. This establishes A8 implementation
consistency only. No full F32-activation serial replay has been completed.
An earlier eight-token FP4 A16 smoke run measured 3.364 tok/s; it is not a
throughput or quality acceptance result. No full-model FP6 rate is available.

Final v5 at 1024 + 256 took 29.511084 s of decode, or 115.278 ms/token:

| Stage | ms/token |
| --- | ---: |
| Attention QKV | 5.929 |
| Attention core | 15.743 |
| Attention output | 2.098 |
| SSM input | 11.104 |
| SSM preparation | 3.290 |
| SSM core | 2.975 |
| SSM output | 6.221 |
| FFN gate/up | 28.153 |
| FFN down | 13.988 |
| Q6_K output head | 20.203 |

The CSV `effective_gb_s=0` is a placeholder caused by the runner passing
zero resident-byte accounting for this path. It is not measured bandwidth;
fix accounting before using that column.

Activation preparation, 1000 passes on native hardware:

| Input columns | Scalar A8 us | Scalar A16 us | Final SVE A8 us | Final SVE A16 us |
| --- | ---: | ---: | ---: | ---: |
| 5120 | 342.709 | 402.656 | 27.192 | 28.012 |
| 6144 | 411.010 | 482.315 | 32.568 | 33.526 |
| 17408 | 1164.355 | 1367.871 | 91.886 | 94.586 |

Synthetic 17408 x 5120, 48 cores, 500 passes, placement verified:
FP4 A8 ~394-396 GB/s, A16 ~294.8 GB/s; FP6 A8 ~397.5 GB/s, A16
~283.8 GB/s. Matched scan controls reach ~844 and ~866-867 GB/s.
These are source-byte/makespan rates. Worker-only durations are also in
logs; do not mix boundaries. Pthread startup skew is about 2.65 ms,
amortized over 500 passes. Full-tile manual unrolling did not improve the
kernel and is not the selected implementation.

Original direct conversion planned 18.212 GB of anonymous allocation,
371 converted matrices, and took 382.029 s. Conversion plus image export
took 417.267 s; validated image reload took **27.896 s**. The final image
contains 16,047,620,416 bytes. Padding makes planned anonymous bytes larger
than touched payload. Do not infer peak resident memory from image size.

The prior expanded-FP4 study's 852-859 GB/s result remains historical:
see `a64fx/llm/qwen38_nvfp4_expanded.md`. It uses another representation
and does not establish true FP6 performance or a real-model token rate.

## qlair commit and calibration status

Clair checkout: `/home/syoyo/work/clair/a64fx`.
Committed **`a18ee475` — Fix A64FX compiler emulation and NUMA placement
queries** (6 files, 352 insertions, 6 deletions). No push was performed.
The commit includes focused regressions and
`tools/qlair/a64fx-lowbit-validation.md`.

Fixed:

- Single LDR/STR pre/post-index SP writeback, previously discarded via XZR.
- NEON FCVTN/N2 and FCVTL/L2 F64/F32 support and aliased operands.
- NEON LD1/ST1 single-lane B/H/S/D insertion/extraction and post-indexing.
- SVE WHILE N flag: first predicate element, not last. FCC's
  `whilelo; b.mi` copy lost the final 32 bytes of a 2080-byte activation.
- `get_mempolicy(MPOL_F_NODE | MPOL_F_ADDR)` simulated placement queries,
  with invalid/unmapped-query error handling.

Working-checkout tests passed **537/537**, including pre-existing
uncommitted tests. Matched FCC driver/kernel replay now gives `correct=1`
for FP4 A8 and FP6 A16. Existing timing/cache/thread/profiler changes and
untracked calibration work in Clair were deliberately left uncommitted.
Do not stage or discard them as part of this task without inspecting them.

A placement-checked 48-core FP4 A8 simulation, 17408 x 5120, four marked
passes, reports 4,696,273 maximum worker ticks at 2 GHz, or 85.404 effective
GB/s. Native ~394 GB/s above uses 500 passes and a makespan boundary. This
is an unresolved diagnostic discrepancy, not a matched calibration result.
No timing constants were fitted. An out-of-order simulation was stopped
at wrap-up and has no usable completed result. The full calibration gate
remains open.

## Validation completed and quality prototype limits

- Native SVE numerical/activation tests passed, including half ties,
  exponent extremes, codes, signs and tail shapes (`test_v5.log`).
- Final local QEMU SVE tests passed after the last scalar FP6 conversion
  simplification (per-block power-of-two inverse). That last conversion
  change has not been benchmarked natively.
- ASan/UBSan loader/image tests passed for both formats and F32/A8/A16,
  including stale, truncated, corrupt and partial-failure cases.
- Quality comparison Python tests: 3 passed. Evaluator cross-compiles.
- `git diff --check` passed. Existing `gguf_loader.h` compiler warnings
  remain; do not claim a globally warning-clean build.

`qwen38_lowbit_eval.c` is an **unvalidated prototype**, not a completed
BF16 oracle. It streams BF16/raw NVFP4 one layer at a time on one node,
retains hidden states for a fixed teacher-forced sequence and caps staging
at 4 GiB with a 6 GiB available-memory guard. Low-bit resident mode also
supports `--serial` to compare traversal order. It emits bounded logit
files and JSON NLL/perplexity, relative L2, maximum error, KL, top-1 and
token-matching diagnostics, tied to token/reference hashes.

It was cross-compiled but **not linked/run on native hardware**. First
validate layer-major versus serial execution on identical resident FP4/F32
weights, then validate BF16 staging/mapping on real shards. No held-out
corpus has been selected or evaluated. The Python comparator requires a
source-NVFP4/F32 baseline and the same BF16 logit reference; it rejects
missing/nonfinite/mismatched records and regressions in NLL/L2/max/KL.
Passing those unit tests does not establish model quality or greedy equality.
The logit file ABI currently assumes little-endian hosts such as A64FX.

## Reproduction and preserved artifacts

Local scratch: `tmp/q38-lowbit-20260924/`.
Native logs/objects/source archive:
`tmp/q38-lowbit-20260924/hw-51893515/`.
Shared remote copy on `fugaku1`:
`~/work/gemm/qwen38-27b/tmp/q38-lowbit-20260924/hw-51893515/`.

The shared directory also contains **`fp4-v1.image`**, preserved with
bounded 8 MiB writeback before releasing the allocation. Do not download
this 15 GiB image to the workstation: local free space was only ~15 GB.
Exclude `fp4-v1.image*` when syncing small logs. Stage it into a new
allocation's `/local` with bounded reads/writes, fsync and fadvise; source
identity must still match. Never assume old `/local` data survives.

Useful evidence:

- `fp4_a8_bench_v3.log`, `fp4_a8_bench_v4.log`, `fp4_a8_bench_v5.log`.
- `prepare-v4-trace-check.json`, `prepare-v5-trace-check.json` (local).
- `prepare_old.log`, `prepare_v4.log`, `prepare_v5.log`, `test_v5.log`.
- `lut48.log`, `full48.log`, `baseline48.log`, `fp4_a16_smoke3.log`.
- `native-sources-v5.tgz`, `runner_main_v4.o`, `model_v4.o`, `portable.o`,
  `lowbit_v4.o`, `lowbit_v5.o`; matched benchmark `bench.o`,
  `qwen38_lowbit.o`, `lowbit_lut.o`.
- Local `final-sve-tests.log`, `final-model-tests.log`, qlair regression
  logs, `bench_lut_sim`, `profile-fp4-48.log` and aggregate JSON.

Raw evidence is untracked scratch; preserve it before any cleanup.
`native-sources-v5.tgz` records the measured source snapshot; the final
working tree additionally contains the evaluator prototype and later
portable-converter/reserve edits. Do not attribute every current source
line to the measured binary.

Model sources on Fugaku (verify before use):

```text
~/models/qwen38/27b/Qwen3.8-27B-NVFP4-Quality-v2.gguf
~/models/qwen38/27b/bf16/Qwen3.8-27B-BF16-00001-of-00002.gguf
~/models/qwen38/27b/bf16/Qwen3.8-27B-BF16-00002-of-00002.gguf
```

Build targets from the remote repo root, with build temporaries in `/local`:

```sh
make -C a64fx/llm CC=fcc OPENMP=1 qwen38_lowbit_runner \
  qwen38_lowbit_sve_test qwen38_lowbit_prepare_bench qwen38_lowbit_eval
```

The measured runner was built from separate objects (runner O2, kernel
O3); the Makefile's default O3 runner rebuild must be revalidated before
comparing timing. Current target names are for resumption, not a claim
that this exact one-line command produced the archived v5 binary.

Measured command in the now-released allocation:

```sh
OMP_NUM_THREADS=48 TF_DPROF=1 TF_DUMP_TOKENS=1 \
./runner_lowbit_v5 \
  "$HOME/models/qwen38/27b/Qwen3.8-27B-NVFP4-Quality-v2.gguf" \
  --lowbit fp4 --lowbit-activation a8 \
  --lowbit-image /local/q38-lowbit-51893515/fp4.image \
  --prompt 'Explain why the sky is blue.' --max-seq 1344 --threads 48 \
  --bench --bench-prompt 128,1024 --bench-gen 256 \
  --bench-runs 1 --bench-warmup 0 --bench-csv
```

Final local checks (scratch directory must already exist):

```sh
export TMPDIR="$PWD/tmp/q38-lowbit-20260924"
clang --target=aarch64-linux-gnu --gcc-toolchain=/usr -fuse-ld=lld \
  -static -O2 -march=armv8.2-a+sve \
  a64fx/llm/test_qwen38_lowbit.c a64fx/llm/qwen38_lowbit.c \
  a64fx/llm/qwen38_lowbit_sve.c -lm -o "$TMPDIR/test_final_sve"
qemu-aarch64 -cpu max,sve512=on "$TMPDIR/test_final_sve"
cc -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  -Wno-unused-function -Icommon -Ia64fx/llm \
  a64fx/llm/test_qwen38_lowbit_model.c a64fx/llm/qwen38_lowbit_model.c \
  a64fx/llm/qwen38_lowbit.c -lm -o "$TMPDIR/test_final_model"
"$TMPDIR/test_final_model"
python3 -m unittest discover -s a64fx/llm -p test_qwen38_lowbit_quality.py
```

Clair regression/replay commands are in its committed validation note.
Read `a64fx/remote-dev-procedure.md` before allocating. Use targeted rsync
and versioned staging. Never use `/tmp`, hold two full resident models,
or copy a model into a model-sized dirty page cache. Use `/local` on the
node and repository `tmp/` locally, bounded I/O and memory monitoring.
No allocation or simulation from this checkpoint is intended to remain
running. Preserve unrelated work. No push without a new explicit request.

## Next work, in order

1. Validate the evaluator's layer-major traversal against serial FP4/F32
   execution on identical fixed tokens, then a bounded native BF16 smoke.
   Audit tensor mapping and causal recurrent state before trusting metrics.
2. Choose/record held-out text and tokenization; obtain single-node BF16
   reference logits, source-NVFP4 baseline, FP4 and true BF16-derived FP6
   metrics. Check the complete FP6 memory plan before loading it.
3. Run full unapproximated F32 serial replays for each optimized A8/A16
   candidate. Compare every generated position and token ID, report the
   first divergence, and keep this gate separate from held-out quality.
4. Optimize measured costs: Q6_K output head (~20 ms/token), FFN gate/up
   (~28), down (~14), SSM projections, then context-dependent attention
   (~16 at 1024). Kernel source GB/s alone does not imply 40/30 tok/s.
5. Collect matched-object native/simulator profiles with identical pass
   counts, timing boundaries and manifests across two allocations, ten
   samples, then address instruction/scheduling/memory discrepancies.
   Fit only to measurements; keep FAPP traffic separate from source bytes.
6. Repeat 1024+256 and 128/4096 sensitivities after correctness gates.
   Fix resident-byte reporting and consider skipping unused NextN weights.
   Report actual rates and remaining budget if targets are still unmet.

## Copy-ready resuming prompt

```text
Resume the compact FP4 / true FP6 E2M3 Qwen3.8-27B single-stream decode
work on one 48-core A64FX node. Read AGENTS.md, root decode.md and
a64fx/remote-dev-procedure.md, then inspect both working trees. Root
resume.md and the old qwen38-fp4-resume.md are not current for this task.

Goals: plain serial N=1 FP4 >=40 tok/s and true BF16-derived FP6 >=30
at 1024 input +256 generated tokens, with 128/4096 sensitivity. The
BF16 oracle must also stay on one node. No speculative rate substitutes.
Quality must be no worse than source NVFP4 against shared BF16 logits,
and each optimized path must match every generated position/token ID
from its unapproximated F32-activation serial reference.

Current native FP4 A8 v5: 8.675 tok/s at 1024+256, 9.693 at 128+256,
one trial each. All 512 positions/IDs/selected logits match the earlier
A8 implementation; F32-reference equality is still untested. No full
FP6 model run or held-out quality result exists. Compact FP4 is 4.5
bits/weight including scales; true E2M3 FP6 is 6.25. Older expanded
packed6 FP4 bandwidth results are a different format.

qlair fixes are committed in /home/syoyo/work/clair/a64fx as a18ee475:
SP writeback, NEON conversions/lane transfers, SVE WHILE flags and
NUMA placement queries. Working-checkout tests passed 537/537 and
matched FCC FP4/FP6 numerical replay passes. Pre-existing simulator
changes remain dirty: preserve them. Timing calibration is incomplete;
85.4 simulated GB/s versus ~394 native is not a matched comparison.
Use two allocations, ten samples, CV/drift <=2%, <3% cycles/BW error.

First validate the new qwen38_lowbit_eval prototype (cross-compiled,
not yet run natively): layer-major versus serial FP4/F32 on fixed
tokens, then bounded BF16 streaming. Continue quality and complete
serial greedy gates before treating A8/A16 speed as accepted. Next
profile/optimize Q6_K head, FFN and attention using measured budgets.
Do not imply that the 40/30 tok/s targets have already been met.

Allocation 51893515 was released. Evidence and native source/object
snapshots are under tmp/q38-lowbit-20260924/hw-51893515 locally and
on Fugaku shared storage. Shared fp4-v1.image is 16,047,620,416 bytes;
image reload took 27.896 s. Do not download it to the space-limited
workstation. Stage bounded chunks to a fresh /local allocation and
verify source identity. Models and exact commands are in decode.md.
Use /local or repo tmp/, never /tmp. Preserve raw evidence and unrelated
dirty files. Do not push without explicit per-action authorization.
```
