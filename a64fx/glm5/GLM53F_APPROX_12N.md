# GLM-5.3-Flash TP12 approximation experiments

Experimental work on normal-mode job 51383005. The established exact result
remains median **29.861 tok/s** (three 1,024-token runs after the fixed 8K
prompt), and 29.133 tok/s for the 7,189-token full continuation. See
[GLM53F_Q4_12N.md](GLM53F_Q4_12N.md). The frozen approximation passes the paired
Python pass-rate gate, but both models fail the separate C++ regression.
60 tok/s has not been established.
The subsequent frozen approximate MTP full run also fails scalar equivalence;
use the established exact executable as the fallback, not this approximation.

The source has since been corrected for a concrete MTP rebase bug: the old
path restored `mtp_base+1` and appended `first_token` twice. The fix restores
`mtp_base`, appends committed inputs once only on rejection, and retains the
speculative cache on full acceptance. Runtime validation requires a fresh
allocation and is not represented by the rejected measurements above.

## Implemented controls

Append these argument pairs after the existing ten positional arguments:

```
--expert-top-k 2|4|6|8
--sparse-top-k 256|512|1024|2048
--index-refresh 1|2|4|8
--kda-proj bf16|int8
--prompt-list MANIFEST
```

Defaults remain 8 experts, 2048 sparse tokens, refresh every position, and BF16
KDA projections. Approximation starts only after exact prompt prefill. The
shared expert remains enabled. Selected routed weights retain the existing
normalization. INT8 uses symmetric 128-element blocks for Q/K/V and output
projections only; convolution, recurrence, gates, and state keep their original
representations. Original BF16 projection weights remain resident so switching
back for the next exact prefill does not reload them.

The aggressive exploratory policy `--expert-top-k 2 --sparse-top-k 256
--index-refresh 8` is accepted for non-context-parallel layers. It is not a
quality-tested default and must pass scalar-reference checks and the paired
Python gate before use.

Snapshots include the model-owned policy and sparse-selection history. The
selection-history ring covers the maximum five-position verification batch.
Non-default sparse policies are rejected for context-parallel layers.

Draft count zero is a scalar decode baseline. One through four retain the
existing MTP implementation. The draft model is unchanged by target
approximation flags.

## Resident requests

Manifest lines contain `id prompt_ids_path output_ids_path` with no whitespace
inside fields. An optional fourth field is a diagnostic policy name: `exact`,
`top4`, `sparse512`, `refresh`, `aggressive`, `int8`, `no-int8`, `no-refresh`, or `all`.
Repeated identical prompt paths reuse an exact-prefill snapshot; different
prompts reset both models and run exact prefill again. Do not modify prompt
files during a resident run. Capacity is sized from the largest manifest
prompt and the common generation limit. Model weights load once per process.

`GLM53F_SPEC_SELF_REFERENCE=1` checks each request against its own approximate
scalar reference. This is an implementation-correctness check, **not** a model
quality measurement. Any mismatch rejects the associated speed result.

MTP rebasing now restores the pre-cycle cache length and appends each committed
input once after rejection. Fully accepted cycles retain the speculative cache,
append only the final draft input, and skip target snapshot restoration because
the verifier state is already committed. This avoids duplicate-first-token
state and redundant replay/copy work. The verifier now also accepts a NULL
final snapshot destination, so the final full-acceptance KDA state is not
materialized into either a snapshot or temporary buffer.

## Opt-in exact kernel experiments

- `GLM53F_Q4_ROW_TILE=1`: four-row Q4/Q5 kernels for one/two activation rows;
  other quantizations and larger token groups retain the reference path.
  Activation loads and per-lane sums are shared without changing accumulation
  order. No full INT8 expert-weight expansion is performed.
  Value `2` uses two two-row blocks to reduce register pressure.
- `GLM53F_Q4_SCALE_CACHE_MB=2048`: model-owned lazy cache of 24-byte decoded
  Q4/Q5 scale/minimum records per 256-weight block. The requested budget is
  shared in proportion to layer count, capped at 2 GiB for the target/MTP
  configuration. Unsupported types and exhausted budgets use the original
  arithmetic. Cache allocation and release follow the MoE context lifetime.
- `GLM53F_Q4_ACT_SUMS=1`: with the metadata cache, precompute signed 16-bit
  activation lane sums; the valid range is only -508..508. This adds small
  per-call scratch, not another resident weight representation.
- `GLM53F_MHC_BATCH_TEAM=1`: persistent OpenMP team for batched pre-mixing.
- `GLM53F_MHC_POST_TILES=1`: split post-mixing into 256-element work items.
- `GLM53F_KDA_PROJECTION_QUEUE=1`: combine independent batched BF16 front
  projections in one work queue, then combine the two dependent gate
  projections in a second queue. INT8 Q/K/V retain their separate quantized
  queue. The reference four-row kernels and scalar remainder arithmetic are
  unchanged; scalar decode is unaffected.
- `GLM53F_SPEC_QUEUE_SWEEP=1`: same-prefill scheduling on/off crossed with
  one/two drafts (requires positional draft count 2). Mutually exclusive with
  the other sweeps; supports repeated timing and scalar self-reference.
  Value `2` also tests the repaired mHC team with the KDA queue enabled,
  producing six variants; each has its own reference check.
- `GLM53F_SPEC_TILE_SWEEP=1`: same-prefill untiled/row/post comparisons, compatible
  with self-reference and repeated timing. Do not combine with other sweeps.

These remain opt-in experiments. Metadata and activation-sum paths pass
bit-exact expert tests at 1 MiB and 64 MiB budgets, including partial-cache
fallback. The KDA projection queue passes bit-exact standalone tests for
1..4 tokens at 1, 12, and 47 threads. Their subsequent full-model sweep is
rejected for the reference failure described below.

## Validation and rejected results

Initial checks passed sparse batch/rollback at warm lengths 2050, 2051, 8190,
and 8191 with sparse512/refresh4. The five-token full-target batch/rollback test
passed with top4/sparse512/refresh4/INT8. A synthetic 2051-position, 128-token
self-reference run passed at 40.160 tok/s. Resident duplicate coding prompts
produced identical 34-token responses and matched scalar references.

The subsequent fixed-8K, 1024-token run **failed** scalar/MTP equivalence at
position 17 (`got=12`, `expected=5239`). Three measured rates were
38.557/38.557/38.561 tok/s, but all are rejected. The combined approximation and
new mHC path must not be promoted on this evidence. Individual-policy isolation
passed all eight policies at 128 tokens with mHC disabled (combined policy:
39.715 tok/s). A stronger mHC unit test using non-uniform norm weights reproduces
an absolute difference of 2.38418579e-7 in the first team implementation.
Reusing the reference normalization loop fixes this test bit-for-bit for
1..5 positions. The revised pre-mixing path remains disabled pending full-model
validation.

The fresh-tree sweep subsequently passed all 1024 scalar-reference tokens in
all nine runs. Untiled rates were 39.371/39.395/39.391 tok/s (median 39.391).
Four-row tiling was slower: 37.673/37.680/37.672; adding post tiling to that
variant gave 37.795/37.814/37.818. These tiling variants are not selected.
Draft acceptance was 508/515 (98.6408%) throughout. The scalar reference ran
at 32.903 tok/s. The 60 tok/s target still requires roughly a further one-third
reduction in cycle time, not merely higher draft acceptance.

### Rejected final kernel sweep (18:42 JST)

Immutable binary `tmp/source-1810/a64fx/glm5/glm53f_spec_queue1810` (SHA256
`07fb8f9178a770a7a7708adf5f87a32ef8112f6edd218b9e64dd1b02d2d26175`)
compared metadata caching with serial scheduling, the KDA queue, and the KDA
queue plus repaired mHC, crossed with one/two drafts. Each used three resident
1024-token requests sharing one exact 8K prefill. Raw rates, **not accepted
performance results**, were:

| Variant | Three raw tok/s measurements | Reference status |
| --- | --- | --- |
| Metadata, one draft | 37.102 / 38.807 / 38.811 | PASS / PASS / FAIL |
| + KDA queue, one draft | 39.209 / 39.225 / 39.224 | PASS / PASS / FAIL |
| + repaired mHC, one draft | 38.905 / 38.912 / 38.925 | PASS / PASS / FAIL |
| Metadata, two drafts | 37.004 / 37.016 / 37.027 | PASS / PASS / FAIL |
| + KDA queue, two drafts | 37.326 / 37.339 / 37.357 | PASS / PASS / FAIL |
| + repaired mHC, two drafts | 37.536 / 37.552 / 37.559 | PASS / PASS / FAIL |

The first two scalar references are identical, but differ from the frozen
8K continuation at one-based token 84 (1978 versus 1057). The third scalar
reference differs from the first two at zero-based token 626 (expected 600
versus MTP 3443), causing all six third-repetition checks to fail. The cause
is not isolated; do not attribute it specifically to metadata or mHC without
further evidence. No variant is promoted, and these rates do not replace the
validated 39.391 tok/s result.

Rank 0's metadata cache used 2,004,254,744 bytes. Minimum available memory
across all 12 ranks was 8,571,840 KiB. Logs and output IDs are archived in
`tmp/approx-51383005-evidence/final1810*`. Reproducibility and the C++ failure
must be resolved before further speed experiments are promoted.

### Rejected frozen approximate full run (18:56 JST)

The quality-tested `glm53f_quality_1710` executable was then run with all new
kernels disabled, the frozen top4/sparse512/refresh4/INT8 policy, and one draft.
Scalar generation reached EOS 154827 after 5896 tokens at 32.514 tok/s. MTP
reached EOS after 5895 tokens at a raw 38.857 tok/s, but failed its own scalar
reference at zero-based position 5266 (`got=16`, `expected=17`). Its 98.1176%
draft acceptance does not make this a passing run. **38.857 tok/s is rejected.**

The independently prefilled scalar prefix also differs from the earlier
frozen 1024-token reference at one-based token 412 (341 versus 2224). This
shows that the cross-run reproducibility problem is not confined to the new
kernel build; its cause remains unisolated. Do not claim deterministic
long-form approximate MTP from the earlier short tests or the Python scores.

Minimum available memory over all 12 ranks was 9,815,872 KiB. Full logs and
both output ID files are saved under `tmp/approx-51383005-evidence/frozen-full*`.
The earlier 39.391 tok/s measurement is a passing short, same-prefill result
only—not a validated long-run operating point. The established exact fallback
remains 29.861 tok/s median over 3x1024 and 29.133 tok/s for its byte-identical
7189-token full continuation; see [GLM53F_Q4_12N.md](GLM53F_Q4_12N.md).

The existing 8K throughput fixture is a C-header generation task. It is not
the separate `task_queue.cpp` regression prompt; do not treat its continuation
as a standalone task-queue program.

## Coding-quality tooling

`prepare_glm53f_eval.py` prepares the first 32 MBPP+ task IDs in numerical order
and all 164 HumanEval+ tasks with the canonical tokenizer. Versions: EvalPlus
0.3.1, tokenizers 0.21.4, MBPP+ v0.2.0, HumanEval+ v0.1.10. Dataset and tokenizer
SHA256 values are recorded in generated `metadata.json`. A common greedy,
non-thinking template and 2048-token limit are used for Python.

`decode_glm53f_eval.py` decodes and sanitizes without executing code.
`run_glm53f_eval_sandbox.sh` executes EvalPlus inside a network-disabled
bubblewrap namespace with no host home or repository mount. Its dedicated
work directory must contain `samples.jsonl` and the selected `problems.jsonl`.
Time/address-space/process limits apply. The reference-solution smoke test
passes. Do not run generated code outside this sandbox.

`run_glm53f_cpp_sandbox.sh` compiles the separate task-queue response with strict
C++17 warnings and runs a deterministic 10,001-response stateful fuzz test.
The prompt does not specify ADD-success wording, so that wording is not scored;
state effects and the other responses are checked. Use the complete response
from `cpp-task.ids`, not the unrelated 8K header fixture.

The gate is at most eight fewer HumanEval+ passes out of 164 than the paired
exact baseline. MBPP+ calibration is complete: exact 29/32, candidate 29/32,
with the same failures (56, 57, 74). The candidate was frozen before holdout:
top4/sparse512/refresh4/INT8, one draft, exact prefill, all tiling and metadata
experiments disabled. The frozen record is in
`tmp/glm53f-quality/candidate-frozen-1712.json`. The holdout completed at
18:24 JST with the following paired results:

| Fixture | Exact | Frozen approximation |
| --- | ---: | ---: |
| MBPP+ calibration | 29/32 | 29/32 |
| HumanEval base | 156/164 | 152/164 |
| HumanEval+ base + extra | 150/164 (91.46%) | 149/164 (90.85%) |
| Separate C++ strict compilation | FAIL | FAIL |

HumanEval+ net loss is one task (0.61 percentage points), below the permitted
eight-task loss. Lost tasks: 46, 103, 125, 161. Gained tasks: 89, 99, 158.
The candidate was not retuned on this holdout. Both C++ responses reached
8192 tokens, included prose and unfinished revisions, and failed compilation;
the stateful fuzz test therefore did not run. These are real failures, not
passing C++ results. The aggregate implementation is not promoted as meeting
all requested gates.

Scored evidence: `tmp/glm53f-quality/humaneval164-{exact,candidate}-r1/`,
`tmp/glm53f-quality/cpp-{exact,candidate}-r1/`, and the frozen record above.
The evaluator applies its process cap inside the private user namespace so
unrelated host-user threads cannot prevent sandbox creation. Network, file,
memory, and time restrictions remain enabled for both model outputs.

The paired pass-rate gate applies to these fixed coding fixtures only. Most
Python requests have short contexts; passing this gate would not establish
general long-context quality. Conversely, 8K scalar/MTP token equivalence
checks implementation correctness under the selected approximate policy,
not equivalence to the exact model's answers.

The manifest also accepts a fifth field containing a positive per-request
generation limit after the diagnostic policy field. Python holdout requests
use 2048 tokens; the C++ request uses 8192. The common cycle budget must be
large enough for the largest requested limit.

The [Unsloth guide](https://unsloth.ai/docs/jp/moderu/glm-5.3-flash) and its
[implementation PR](https://github.com/ggml-org/llama.cpp/pull/27754) motivate
short-draft sweeps and careful pooled-key caching. The published speedups are
GPU measurements, not an A64FX projection. This runner already caches completed
pools and has no dynamic graph reconstruction, so those gains cannot simply
be multiplied into its measured rate.

An opt-in one-draft deferred verifier (`GLM53F_SPEC_DEFER_VERIFY=1`) is present
as an experiment: on rejection it commits the first verifier output and rolls
the MTP cache back to the pre-cycle prefix. A first probe found and fixed an
incorrect draft-token consumption in that rollback path. It has not yet passed
a fresh 12-node exactness run, so it is deliberately not part of the promoted
policy or any reported speed measurement.

The opt-in carry path also now performs rejection lookahead: after restoring a
rejected draft it evaluates and emits the fallback's next target token, then
carries that token into the next verifier cycle. This is intended to eliminate
the scalar target step on accepted cycles, but remains unpromoted until a fresh
long-prefix exactness run verifies the target/MTP state invariant.

## Deployment caveat

The compute node was observed reading an older source checksum than login4
at the same shared-filesystem path, even after deployment. Reusing an existing
filename can therefore silently build stale code. Subsequent builds use a new
immutable source directory and `sha256sum -c SHA256SUMS` on the compute node,
plus uniquely named executables. Do not infer implementation identity from a
successful compiler exit or login-node checksum alone.

Current fresh tree: `~/work/gemm/glm53f/tmp/source-1644/`. Rank-local model
stages remain under `/local/*-51383005`; do not restage or duplicate them within
this allocation. Never overlap MPI launches on this interactive allocation.
### Allocation 51387278 revalidation (2026-09-06)

The corrected target/MTP cache state machine was rebuilt and tested against the full 8K-prefix self-reference. Frozen top4/sparse512/refresh4 reached 36.882 tok/s at alpha 0.946768 with 1024/1024 output tokens matching. The opt-in aggressive top2/sparse256/refresh8 policy also matched 1024/1024 and reached 40.378 tok/s at alpha 0.878899. Neither reaches the 50+ tok/s goal; verification remains the dominant phase (43.955 ms/cycle aggressive), so further gains require a token-correct batched verifier or a higher-acceptance draft path.

### External-reference audit (2026-09-07)

The apparent long-prefix failure was traced to the reference, not the batched graph.  A fresh 12-node run with `GLM53F_BATCH_SCALAR=1` (the established scalar target step for every verifier position) was run through the same 8,378-token prefix.  It emitted token 0 = `785` and token 1 = `2701`; the archived reference incorrectly expected `1196`.  The normal batched verifier and all-layer scalar-order diagnostic gates likewise emit `2701`.  Therefore the archived `...greedy` file is stale or was generated under a different decode policy and must not be used as the correctness oracle.  Differential checks should first generate a scalar reference with the identical model, policy, and prefix.

### Exact-order KDA verifier experiment

Long-prefix comparison showed that the multi-token KDA verifier's 4x4 BF16
front kernel can accumulate in a different order from the scalar 8-row path.
`GLM53F_KDA_BATCH_EXACT=1` is an opt-in diagnostic gate that reuses the
scalar-order projection kernel for each verifier position. It is not enabled
by the performance policy because it adds barriers and is expected to reduce
throughput. A fresh 8K first-token probe is running to determine whether this
arithmetic drift explains the `2701` versus `1196` mismatch.

### Exact deferred-verifier fast path (2026-09-07)

The one-draft deferred verifier now uses the established scalar target step for
its single required verification position, followed by an explicit target
snapshot. This avoids the batched scratch/team setup for a one-position call;
the state transition and token decision are unchanged. On a fresh 4,137-token
prefix, binary `f72f5dd…` produced `64/64 PASS`, with target phase 28.628 ms,
MTP draft 2.114 ms, exact scalar verify 29.894 ms, and 32.805 tok/s at alpha
0.25. The prior batch-one verifier measured 32.763 ms in the same profile
family. This is an exactness-preserving improvement, but 50+ tok/s still
requires reducing the target graph or amortizing more than one target position
per cycle.

The same-prefix two-position batch verifier was separately checked against its
fresh scalar self-reference. Although it reduced batch cost to 22.799
ms/position, it diverged at output position 4 (`got=51`, `expected=1161`), so
that fast path is not correctness-safe and remains diagnostic only.

An explicit `omp simd` annotation on the KDA recurrent inner loops was also
tested (`e6ca7d…`). It remained exact (`64/64 PASS`) but regressed the scalar
target profile from 28.996 ms/position to 29.953 ms/position and is therefore
not retained.

Allocation 51393730 long-prefix validation completed with top-2/sparse-256/
refresh-8, one-draft deferred scalar verification: 64/64 reference match after
the full 8,378-token prefill, alpha 0.593750, 25.195 tok/s. A two-draft run
with `GLM53F_BATCH_SCALAR=1` still diverged at generated position 7 (220 versus
537), even though every verifier position used the scalar target graph. The
remaining multi-draft rollback/commit error is therefore not solely fused batch
arithmetic; that configuration is rejected for performance claims.

### MTP rollback state and reference audit (2026-09-07)

The MTP context has its own sparse index-selection ring. Restoring only its
cache length left speculative selection state alive across rejected cycles. The
runner now saves/restores the MTP selection and eight-entry history ring around
each speculative cycle. With a freshly generated scalar reference, the
two-draft scalarized verifier passes `32/32` (the old external reference was
stale for the current tree and incorrectly reported `56936` where the current
target emits `198`). The scalarized diagnostic is correctness evidence, not a
throughput result; fused verification still needs a fresh-reference benchmark.

Replacement allocation 51393730 completed Q4 routed/shared/core and MTP staging
(`SENTINEL_ALL_Q4_MTP_STAGED`). Its fresh runner aborted during model
initialization before decode, so no throughput number is claimed from that
allocation. The deferred one-draft path now skips its redundant rejected-suffix
snapshot allocation and save, reducing memory pressure without changing its
token/state semantics.

### Lazy multi-draft rollback (opt-in)

`GLM53F_SPEC_LAZY_SNAPSHOT=1` is available for `ndraft > 1`. It captures one
target snapshot after the cycle's first target token, runs fused verification
without per-position KDA captures, and replays only the committed prefix if a
draft is rejected. Full-acceptance cycles therefore avoid the extra snapshot
copies. This path is not performance- or quality-claimed until it passes the
fresh internal greedy-reference gate on A64FX.
