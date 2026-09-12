# DS4.1-Flash A64FX bring-up

## Full text runner

`ds41f_run` executes the real 40-layer model on 12 A64FX nodes, one MPI rank per
node, with MPI startup/bootstrap and uTofu reductions (optional native MPI broadcasts). Expert weights
are resident by `expert_id % 12`, dense layers by `layer % 12`, and Engram rows
remain on `/local`. Do not launch concurrent MPI programs in the allocation.

The measured exact FP8 path reaches **6.920 tok/s at approximately 1K history**
(positions 1000–1104), compared with 5.765 tok/s on the same allocation before
the latest changes. Enable `--engram-prefetch --engram-scale-cache --hc-mix-sve
--shared-overlap --weights-local-pages --mpi-broadcast` for this result. All 1,105 token steps and nine saved logit
arrays match the baseline exactly. Optimization has resumed, including an
experimental FP8-to-INT8 SDOT path measures **10.36–10.39 tok/s** in two
profiling-disabled repeats with the additional `--fp8-int8-block 32` flag
(10.208 tok/s with profiling). The 20+ tok/s target
remains open, and INT8 fails the full-model numerical gates.
See [the resumed measurements](../doc/ds41f.md#resumed-int8-optimization-job-51569201).

The initial stage and dense ownership phase are complete for job 51569201.
From a **new shared results directory** inside that allocation:

```sh
env XOS_MMM_L_PAGING_POLICY=demand:demand:demand \
  OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
  mpiexec -np 12 /absolute/repo/a64fx/ds41f/ds41f_run \
  --stage-root /local/u14346/ds41f-51569201 \
  --prompt-ids /absolute/path/prompt.ids --generate 32 --max-context 4096
```

Prompt files contain whitespace-separated token IDs. Prefix plain text with BOS
ID 0; EOS is ID 1. `--ignore-eos` is an explicit benchmark option. `--trace`
records per-layer residual norms/routes. `--logits-prefix PATH` writes FP32
logits, optionally bounded by `--logits-start P --logits-count N` (start defaults
to zero). Disable these diagnostics for
speed measurements. Rank-local logs are `inference.rank00.log` through `11`.

For bounded operator profiling, add `--profile-start 16 --profile-count 1089`
to the 1,100-output, six-token-prompt benchmark. Profiling is disabled by
default; the start defaults to the first generated input and the count is
limited to 4,096 positions. Each rank keeps main-thread wall-clock spans in
memory and writes `profile.rank<R>.bin` plus JSON metadata after the timed
inference loop. No profiling collectives or intermediate disk writes are added.
The 1,089-position profile occupies about 22 MB per rank. Use a new results
directory: existing profile files are rejected.

Report actual history near 1K separately from warm-up and shorter histories:

```sh
OPENBLAS_NUM_THREADS=2 python3 a64fx/ds41f/profile_report.py RESULTS \
  --start 1000 --stop 1105 --json RESULTS/profile-1k.json
```

The report uses dense-owner timings and the slowest parallel expert stage;
it does not add collective wait times across ranks. Reduction remainders
include rendezvous and rank skew, so they are not pure network time. Nested
attention/index/kernel spans overlap their parents. The unattributed residual
checks how closely this reconstruction matches measured token latency.

`--engram-prefetch` starts one local I/O worker per rank. At token start it
fetches both Engram layers' rows into two bounded 24-by-256 FP32 buffers;
layers 1 and 14 wait for their own buffer only when needed. The worker uses
no MPI or uTofu calls. MPI requests `MPI_THREAD_FUNNELED`, and runtime
collectives enable receive acknowledgments to protect receive-slot reuse
under the tested scheduling skew. The latter is necessary when the I/O worker
preempts a receiver. Profiling state is
thread-local; `ENGRAM_PREFETCH` reports overlapping background work and
must not be added to the decode critical path.

`--engram-scale-cache` retains each node's two original Engram scale shards
(about 512 MB total) in HBM, removing the scale-file read from each row lookup.
Loading is chunked with page-cache eviction, and admission preserves the
2 GiB MemAvailable floor. `--hc-mix-sve` vectorizes the FP64 norm used by mHC;
the original matrix-reduction order is retained. `--shared-overlap` computes
the owner's shared expert before the routed-expert reduction, with the same
48-thread team. Its output is still added after that reduction. The profile
reports `EXPERTS_AND_SHARED` as one parallel stage when this option is used.

`--weights-local-pages` allocates original resident tensors on fresh anonymous
pages before the existing bounded `/local` reads. INT8 packed weights always
use fresh pages. This bypasses the Fugaku malloc pool so conversion's parallel
first writes can place rows near their consuming CMG. It is anonymous HBM,
not a file-backed model mapping. Reusing pooled pages had placed entire output
matrices on one CMG and limited INT8 decode to about 8.3 tok/s. The attention
row scratch buffer is now allocated once and reused.
A matched original-weight pool control reaches 10.20 tok/s versus 10.36–10.39
with fresh original weights, and has about 0.42 GB less final memory headroom.

`--mpi-broadcast` uses MPI_Bcast for owner-produced activations and retains
uTofu sums for experts/Engram. It preserves the old broadcast's signed-zero
normalization and runs only on the main thread. `test_broadcast` checks both
modes with all 12 owners, delayed receivers and sizes crossing the chunk
boundary. Selection tests cover first-tie behavior and nonfinite rejection;
the SVE vocabulary argmax and parallel gate scoring retain the scalar choices.

### Experimental INT8 projections

`--fp8-int8-block 32` requantizes resident FP8 matrices to signed INT8 with
per-row, per-K-block FP32 scales, then uses SVE SDOT for decode GEMV, including
the grouped output projection. Blocks 64, 128 and 256 are also accepted;
zero/default keeps FP8. `--fp8-int8-scope projections` limits conversion to
attention `wq_b`, `wo_a`, `wo_b` and shared experts; the default scope is `all`.
Original FP8 activation quantization and BF16 output boundaries remain in place.
This is a single-token GEMV implementation; batched INT8 GEMM is not integrated.

Conversion happens once after loading from `/local`, releasing each original
FP8 tensor as its packed replacement becomes ready. Block 32 adds 12.5% scale
storage plus at most three padded rows. Admission includes the transient source
and destination tensor; when the source uses a malloc pool, it conservatively
budgets all replacements because freed source pages may remain pooled. Full-model conversion takes 0.27–0.50 s/rank
versus 53–57 s resident startup, so no offline INT8 weight files are required.
The original shared-storage weights remain the source of truth.

INT8 is lossy and remains opt-in. On identical fixed-token history, next-token
choices match at 1,065/1,105 positions overall and 104/105 positions near 1K.
All nine saved near-1K argmax choices agree, but minimum logit cosine is 0.902770
and maximum relative RMS is 44.0%, failing the 0.999/1% gates. Narrowing the conversion scope and adding a second residual INT8 plane
did not resolve that drift; the residual-plane experiment is not retained.
Do not present INT8 throughput as a numerically accepted replacement for FP8.

Use `test_int8` for SDOT/reference, grouped-shape, tail and invalid-input checks;
`bench_int8 STAGE BASE ROWS COLS BLOCK GROUP_ROWS` measures an actual staged
matrix. `test_int8_attention STAGE DUMP_PREFIX LAYER POSITIONS BLOCK` reports
same-input attention comparisons (its exit gate is cosine >= 0.999 only).
For full-run comparisons with identical input history, including generated
inputs, use:

```sh
OPENBLAS_NUM_THREADS=2 python3 a64fx/ds41f/compare_run_logits.py BASELINE ACTUAL \
  --start 1000 --count 9 --json ACTUAL/comparison-1k.json
```

Capture that range in both runs and replay the same fixed token history.
The comparison rejects missing/divergent histories and checks cosine >= 0.999,
relative RMS <= 0.01 and matching argmax. `--require-exact` additionally checks
the FP32 bit patterns. The existing `compare_logits.py` independently compares
CPU-reference prefixes and retains its original interface.

The sparse-attention weighted-value loop processes four SVE vectors at once,
using a full A64FX cache line per selected row while preserving each output
lane's accumulation order. Short dimensions retain the predicated tail path.
The default FP8 path and routed MXFP4 experts retain their checkpoint
representation; the experimental INT8 path replaces only FP8 matrices.

For operator debugging, `--dump-prefix PATH --dump-count N` records the first
N positions (default 1, maximum 64) in owner-written
`PATH.pos<P>.layer<L>.bin` files. Each file contains the residual streams,
mHC coefficients, normalized inputs, attention/FFN outputs and routing data.
Existing dump files are rejected. These writes add substantial shared-storage
latency; use a separate run for performance measurements.

`replay_intermediates.py` independently recomputes operators from their recorded
inputs and the original safetensors. Run it on the frontend with NumPy and a
bounded BLAS thread count, not alongside resident inference on a compute node:

```sh
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python3 a64fx/ds41f/replay_intermediates.py \
  --model /home/u14346/models/ds41f --metadata RESULTS/engram_meta.bin \
  --dump-prefix RESULTS/state --prompt-ids REPLAY_INPUT.ids \
  --positions 3 --layers 0,1,2,3,4,5,6,7,8,14,20,24,28,32,36,39
```

Copy the small `engram_meta.bin` from a staged rank into the results directory.
`REPLAY_INPUT.ids` must contain the actual inputs for the replayed positions,
including generated inputs when replay extends beyond the prompt. Dump all
layers: the replay reconstructs shared attention history from the source-layer
inputs even when only selected layers are compared. The default gates require
cosine >= 0.999 and relative RMS error <= 0.01; routing IDs must match exactly.
Passing with identical operator inputs does not establish autoregressive logit
agreement, since that check excludes accumulated upstream rounding differences.

`--max-context 1048576` commits approximately 944 MB of packed KV/index cache
and FP32 windows on every rank. The runner reserves 4 GiB during weight
admission and checks a 2 GiB MemAvailable floor after initialization. A
1,100-output run passed with the 1M allocation and >3.8 GB remaining, but execution
at an actual 1M-token history is not yet validated. Prefill currently follows
the sequential reference path; the new batched GEMM module is not integrated
into the full runner yet.

The first real prompt generated ` Paris. In the`. The completed independent
NumPy reference agrees on all nine next-token choices in `infer-v2`, but five
positions fail the current 0.999 logit-cosine gate (lowest 0.994041139).
This is partial CPU-reference validation, not official GPU parity. See
`a64fx/doc/ds41f.md` for measurements and remaining validation gates.
After correcting mHC post arithmetic, a fresh three-position full-graph check
passes the same 0.999 cosine/argmax gate (minimum cosine 0.999578117). The full
nine-position check with the corrected reference remains to be rerun.
The combined-broadcast runner completes 1,100 outputs in 309.411 s (3.555
generated tokens/s, excluding loading), 3.8% faster than its matched baseline,
with all 1,105 input/next-token triples unchanged. It also passes 726 bounded
operator replay checks and two chat smoke prompts; see the continuation logs
and exact commands in the notes.

Cross-build the runner with `make -C a64fx/ds41f ds41f_run A64FX_MPICC=mpifccpx`.
Build the runner and all A64FX component checks with `make -C a64fx/ds41f a64fx
A64FX_CC=fccpx A64FX_MPICC=mpifccpx` (build only, not execution).
Set TMPDIR to the repository's `tmp/ds41f` on the frontend. Stage initial shards
with `stage_backbone.py`, then run `stage_dense.py` per rank before loading.
Never edit a shell script or overwrite a binary while it is running; copy the
built binary into a versioned results directory for each test run. Shared-file
caches can retain old script contents even after a frontend edit: publish
changed scripts to new paths and compare hashes from frontend and compute.

For plain completion text, run `tokenizer_io.py --tokenizer MODEL/tokenizer.json
encode --text 'The capital of France is' --output NEW_PROMPT.ids` on the
frontend (requires the `tokenizers` Python package). This adds BOS but does not
apply a chat template. Inside the allocation, `sh run_inference.sh STAGE_ROOT
PROMPT_IDS NEW_SHARED_RESULTS_DIR --generate 32` snapshots the executable and
runs one rank/node. Decode its rank-zero log with `tokenizer_io.py --tokenizer
MODEL/tokenizer.json decode --rank0-log RESULTS/inference.rank00.log
--prompt-ids PROMPT_IDS`. Run the wrapper detached for full resident loading.

For chat, pass `encode --messages-json MESSAGES.json --encoding
MODEL/encoding/encoding.py --thinking-mode chat --output NEW_PROMPT.ids`
instead of `--text`. This uses the trusted checkpoint's own formatter, which
already inserts BOS and the assistant generation header. Only string/text
message content is accepted; vision execution is not implemented. The optional
thinking mode and numeric reasoning effort are formatting options, not a claim
of additional validated model capabilities.

This directory contains the V4.1-specific foundation. It is intentionally
separate from `a64fx/llm/ds4f_*`, whose tensor contract is the older V4 model.

## Local checks

```sh
make -C a64fx/ds41f test
make -C a64fx/ds41f validate DS41F_MODEL_DIR=/home/u14346/models/ds41f
```

`validate_model.py` reads only safetensors headers and verifies the 40-layer,
384-expert checkpoint contract without loading the 510 GB model.

The current modules provide:

- `ds41f_model.*`: fixed V4.1 geometry and 12-way expert ownership;
- `ds41f_kernels.*`: reference FP8 E4M3/E8M0, INT8/E8M0, BF16 and RMSNorm kernels;
- `ds41f_moe.*`: deterministic top-k routing and rank filtering;
- `ds41f_kv.*`: compressed-KV/index/window memory sizing;
- `ds41f_runtime.*`: original rank-local contract used by component tests;
- `ds41f_run.c`, `ds41f_attention.*`, `ds41f_comm.*`: full text execution;
- `ds41f_engram.*`: `/local` Engram staging and row access.

SVE BF16, FP8 and adjacent-nibble MXFP4 matvec paths now exist. The runtime
contract alone is not a model runner; `ds41f_run` now connects the modules.

## Compute validation (job 51562789)

The kernel suite now directly exercises BF16 matvec and RMSNorm SVE dispatch
against FP64 accumulations at 12 lengths including 1, 15, 16, 17, 33, 1280,
2304, 5120 and 5121. Output canaries check bounds. Tail accumulation uses
merging predication so inactive lanes retain preceding partial sums.
The A64FX binary reports `backend=SVE vector_bits=512`.

FP8 regression checks include subnormals, exponent-15 finite values, NaN
codes and 32x32 scale indexing across row/column boundaries. The generic
INT8 helper is not the routed-expert decoder: checkpoint `I8` tensors hold
packed MXFP4 and require a separate decoder before model execution.

Run `mpiexec -np 12 sh /absolute/path/a64fx/ds41f/run_rank_tests.sh` from a
fresh shared results directory to produce `kernel.rank0.log` through
`kernel.rank11.log`. This uses scheduler placement, one rank per node.
The separate uTofu test has passed 200 sum/max reductions on each of 12
distinct hosts, using MPI for bootstrap only.

These are module tests, not evidence of full-model inference.

## Staged-weight bring-up (2026-09-12)

`stage_backbone.py` completed in job 51562789, one process/node. It staged
501,382,611,408 exact text-backbone bytes into node-local rank directories.
Experts use `expert_id % 12`; Engram uses contiguous row shards; dense tensors
have a canonical disk copy on rank zero. `stage_dense.py` subsequently
distributed dense ownership; dense weights are not replicated on every rank.

`plan_residency.py MODEL --output NEW_JSON` inventories the actual headers:
dense layer ownership is `layer % 12`, embeddings rank zero, head/norm rank 11.
The resulting compressed weight footprint is 24.575–26.080 GB/node, with
16.897 GB/node Engram on disk. Runtime resident loading has validated that
distribution on all 12 nodes.

Implemented and tested on the initial compute node:

- exact-size bounded anonymous tensor loading with source-cache eviction;
- real expert projection and full w1/w3/SwiGLU/w2 chain, including group-32
  FP8 activation quantization and BF16 boundaries;
- FP8 finite-code round trips and midpoint ties-to-even, scale-block tails;
- routing, mHC mixing, SwiGLU, RoPE and sparse-attention unit cases;
- Engram compressed-token history, pad mapping and disjoint hash buckets.

`test_staged_expert STAGE_DIR EXPERT_ID` compares SVE against scalar reference
using checkpoint layer-zero weights. It does not compare against GPU logits.
`ds41f_utofu_test --staged-experts STAGE_ROOT` adds a deterministic six-expert
combine, comparing uTofu with an MPI test oracle. This new distributed case is
passed on all 12 nodes after staging, as did the synthetic 200-reduction test.

Cross-build all current checks on the frontend:

```sh
TMPDIR="$PWD/tmp/ds41f" make -C a64fx/ds41f \
  ds41f_sve_test test_ops test_engram test_staged_expert ds41f_utofu_test \
  A64FX_CC=fccpx A64FX_MPICC=mpifccpx
python3 a64fx/ds41f/test_staging.py
```

`run_after_staging.sh STAGING_LOG_DIR NEW_RESULTS_DIR` waits for 12 staging
completion records before running the rank-local and uTofu suites. Fugaku
rejects overlapping `mpiexec` launches in this allocation (`PLE 0008`), so
do not launch these while the staging MPI program remains active.

The sections above record the earlier component bring-up. Dense loading,
attention/cache execution, distributed Engram reads, and 40-layer generation
are now connected by the full runner. Remaining work includes broader reference
validation, batched prefill integration and kernel efficiency targets; no
official GPU parity or actual 1M-history throughput is established.
