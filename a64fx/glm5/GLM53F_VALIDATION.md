# GLM-5.3 Flash validation

The supported native Q4 workflow is in [README.md](README.md).
This file contains the current acceptance gates and evidence, not a running
experiment diary. Historical details are recoverable from Git at
`a7ad56ff` and the shared log directories below.

## Acceptance gates

1. Kernel tests pass in conservative and fast-math builds. Compare against
   independently dequantized reference weights and scalar activation contracts.
2. Native batches preserve decode weights, causal selection, recurrent state
   and rollback. Check ragged batches and the sparse boundary at position 2048.
3. Full-model scalar and batched replay produce matching token/logit results,
   including a subsequent decode after snapshot restoration.
4. Preserve scalar per-layer traces for a fixed prompt, and compare against a
   valid llama.cpp reference separately. Native/scalar equality alone does not
   establish llama.cpp equality.
5. Measure throughput and minimum MemAvailable serially on the same allocation.
   Compile/test generated programs before claiming coding quality.

The integrated check command runs kernel tests, 32-token KDA, all three dense
layers, sparse prefill crossing position 2048, and full-model five-/32-position
verification:

```bash
bash a64fx/glm5/run_glm53f_12n.sh check
```

It reuses staged weights and does not create another full model copy.
For both math modes, run the kernel tests separately from performance work:

```bash
export OPAL_PREFIX=/opt/FJSVxtclanga/tcsds-1.2.43
export MPI_HOME=$OPAL_PREFIX TMPDIR=/local
export PATH="$MPI_HOME/bin:$PATH"
for math in conservative fast; do
    flags=()
    if [ "$math" = fast ]; then flags=(-ffast-math -fno-math-errno); fi
    for test in kquant native_batch; do
        mpifcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast \
          -fopenmp -Wall -Wextra "${flags[@]}" \
          a64fx/glm5/test_glm53f_${test}.c -lm \
          -o /local/test_glm53f_${test}_${math}
        OMP_NUM_THREADS=47 /local/test_glm53f_${test}_${math}
    done
done
```

Metadata-only preflight is safe without loading model payloads:

```bash
python3 a64fx/glm5/glm53f_preflight.py --check-tensors "$HOME/models/glm53f"
```

## Native batch result: job 51909852, commit a7ad56ff

The previous KDA batch path substituted compact BF16 projections and an older
normalization formula; grouped prefill and short verification also substituted
the compact shared expert. The old layer-44 four-token check failed with
relative L2 0.00866510901 and mismatching recurrent snapshots.

The native four-row/four-token Q8_0R SDOT bridge reuses weights while retaining
scalar per-lane FMA order. Dense FFN, KDA, sparse projections and shared experts
now use it. KDA retains chronological recurrence and decode normalization.
Sparse prefill preserves per-query causal selection and the F16 cache view.
Older native stages lacking auxiliary projections fall back to scalar native
calls instead of substituting compact weights.

Validated on twelve native A64FX ranks:

- K-quants: 300 cases; worst normalized error 2.72467e-8.
- Q8_0: 42 cases, including 30 bit-exact repacked cases.
- Native batching: 63 shape/batch combinations, 32--4096 columns, 37 ragged
  rows, batches 1/2/3/4/5/7/16/31/32, padded activation strides, zero input
  and mixed Q8_0/Q4_K. Bit-exact against scalar calls in both math modes.
- KDA layer 44: 2/4/5/16/32 positions, zero output difference, bit-exact
  snapshots and snapshot-free final state. Batch-disabled layer 0 also passes.
- Dense layers 0--2: four-position relative L2 below 7e-8.
- Sparse layer 3: four/32 positions, including warm=2046; output and rollback
  differences zero.
- Full-model five-position verifier and 32-position prefill pass. The latter
  matches token/logit 42 / 19.0221443 and subsequent decode 1 / 12.5834875.
- All 45 scalar token-1234 layer traces are byte-identical to the pre-change
  native binary. Final token/logit remain 220 / 15.31567.

### Whole-model measurements

Same allocation, 47 threads, `demand:demand:prepage`, 8049 prompt-only tokens,
256 new tokens, fast recipe:
`--prefill-chunk 512 --prefill-mode fast --prefill-features 27
--prefill-slab 16 --prefill-collective tree-packed --decode-window 128 (historical recipe; see README for the current one)`.

| Binary/prefill | Prompt tok/s | Decode tok/s | Min MemAvailable |
| --- | ---: | ---: | ---: |
| Before batch fix, fast | 33.733 | 21.407 | 10.253 GiB |
| Native batch fix, fast | 41.179 | 20.272 | 10.228 GiB |
| Native batch fix, scalar legacy | 21.035 | 20.588 | 10.371 GiB |

Prefill improves **22.1%**. All 256 fast-prefill output IDs match fresh scalar
replay; the old fast-prefill output matched only the first 66. The two fast
arms decode different sequences, so their decode timings cannot isolate a
kernel regression.

Separate identical-sequence controls:

| Control | Before tok/s | After tok/s |
| --- | ---: | ---: |
| 128 tokens, first pair | 24.665 | 23.268 |
| 128 tokens, reverse-order pair | 24.428 | 21.996 |
| 512 tokens, profiling enabled | 22.858 | 22.491 |
| 1024 tokens, uninstrumented, candidate first | 23.376 | 22.998 |

Every printed token/logit pair matches in the 512-/1024-token runs. Both
longer controls show a **1.6% decode slowdown**, with a larger short-run gap.
The cause remains unresolved. In the 512-token profile, before/after KDA cost
is 10.601/11.044 ms per position; sparse is 8.217/8.369; MoE is 16.286/16.347.
The sparse MLA inner phase is 1.291/1.291 ms. Do not claim unchanged decode
performance or the still-unmet 150 tok/s native prefill target.

### Coding-quality probe

The repository queue prompt used 671 input tokens and reached EOS after
1712 output tokens. Prompt-only ingestion was 51.292 tok/s. It **failed** the
source-only formatting requirement and, after removing only Markdown fences,
failed `g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic`: `Job` was used
before declaration and outside its nested class scope. The CLI harness was
not run. Broader semantic/coding-quality improvement is not established.

Artifacts: `tmp/glm53f-native-batch-51909852/` contains separate baseline and
candidate binaries, `benchmark.sh`, `decode_controls.sh`, `unit-final.log`,
`parity.txt`, generation IDs, and the failed `queue.cpp`/`queue-build.log`.
MPI timing logs are `tmp/glm53f-q4-51909852/batch-*`. Load time is excluded
from reported prompt/decode phase rates.

## Reference correctness and retained diagnostics

The streamed llama.cpp probe previously aliased a zero-size dummy KV buffer
onto its reused compute buffer. Each sparse layer's 512-element F16 latent
row overwrote the first 1024 bytes of the hidden residual input. The bug was
independent of model weights. Old Q2 claims citing final token 29656 or a
0.234513 mHC-post discrepancy relied on that corrupted reference and must
be re-derived.

The local llama.cpp fix gives dummy-buffer tensors private, zeroed, persistent
storage in `ggml/src/ggml-backend.cpp`. Verify that fix exists in the reference
tree before rebuilding `glm53f_stream_graph_probe.cpp`; do not use an older
probe binary. In job 51909852 it was an uncommitted external-tree change.

Against the corrected token-1234 reference, native Q4 relative L2 is:

| Layer | Native | Hybrid |
| --- | ---: | ---: |
| 0 | 0.00590 | 0.00972 |
| 3 | 0.00856 | 0.00999 |
| 15 | 0.00808 | 0.00963 |
| 20 | 0.0303 | 0.0362 |
| 21 | 0.136 | 0.141 |

Layer 21 has a router near-tie (8th/9th biased score margin 0.0065), selecting
one different expert. Both native production and corrected reference select
final token 220. This is single-position evidence, not multi-token parity.

Useful opt-in diagnostics are retained:

- `GLM53F_PROFILE=1`: phase timing in MPI rank logs.
- `GLM53F_LAYER_TRACE_DIR`: first scalar step's 45 F32 layer outputs.
- `GLM53F_SUBLAYER_TRACE_DIR` and `GLM53F_SUBLAYER_TRACE_LAYER`: selected
  sublayer boundaries.
- `GLM53F_KDA_DUMP_PREFIX`: rank/layer KDA intermediates.
- `GLM53F_SPARSE_DUMP_PREFIX` and `GLM53F_SPARSE_DUMP_LAYER`: sparse
  intermediates.

Create fresh output directories and use short bounded runs. Dumps are not
performance configurations. The obsolete per-load `GLM53F_MEMTRACE_DIR`
files were removed; resident/run memory summaries remain enabled.

## Remaining work

1. Isolate the decode slowdown before claiming a throughput-neutral change.
2. Profile remaining native sparse attention and routed K-quant work before
   adding more batching. Preserve scalar activation, recurrence and selection.
3. Extend the corrected llama.cpp reference to multi-token prompts. Remaining
   contracts include Q8_0 query absorption, F16 query/probability rounding and
   the compact mHC projection.
4. Qualify native MTP end-to-end with same-state greedy reference, then longer
   context. Component and finite-logit passes are not sufficient.
5. Repeat coding tasks through EOS and compile/test the raw generated source.

## Runner cleanup qualification (job 51909852)

The consolidated launcher uses shared binaries, node-local stages, and one
environment/preflight implementation for target and MTP. The old recursive
native-copy deletion was replaced by the bounded core copier. Five stale
planning/job-diary documents and three chained validation scripts were
removed; Git at `a7ad56ff` preserves their contents.

The cleanup also exposed a diagnostic bug: under `-ffast-math`, `isfinite`
was removed from the numeric state comparator, and NaN could produce PASS.
The comparator now checks IEEE-754 exponent bits. Its test rejects NaN and
both infinities in either input or reference in conservative and fast builds.
This does not change inference arithmetic or the byte-exact state comparator.

Qualification commands, with this allocation's existing native stage paths
exported as described in the README:

```bash
bash a64fx/glm5/run_glm53f_12n.sh build all
python3 a64fx/glm5/test_glm53f_launcher.py -v
GLM53F_BUILD=0 bash a64fx/glm5/run_glm53f_12n.sh check
```

The full build passes (existing unused-function warnings remain in included
reference/developer code). Eleven model-free launcher tests pass, including
argument forwarding, missing-rank rejection, path guards, native/hybrid
isolation, MTP configuration, staging sentinels and document links.
The real 12-node check passes all kernels, 32-token KDA, dense layers 0--2,
sparse warm=2046/batch=32, and full-model five-/32-position verification.
Its final 32-token and next-decode logits match the a7ad56ff record above.
State-I/O negative probes intentionally print mismatch/FAIL diagnostics;
the test's final line must be `STATE_IO typed_float_and_exact_metadata PASS`.

Build/test logs are under `tmp/glm53f-native-batch-51909852/cleanup-*`;
MPI logs are under `tmp/glm53f-cleanup-51909852/`. Existing weight images
were reused. Staging orchestration and MTP wrappers were tested with mocks;
this cleanup did not redo the large conversion or qualify native MTP/Q8
generation end to end.

The new `decode 1 16` command also passes with all sixteen token IDs and
printed logits identical to the retained pre-cleanup candidate. `generate`
on the 671-token queue prompt with the fast recipe completes sixteen output
tokens, all matching the previous native output prefix. Minimum available
memory is 10.54 GiB or higher in these smoke runs. These verify orchestration
and sampled numerical stability, not a new performance or coding-quality gain.
