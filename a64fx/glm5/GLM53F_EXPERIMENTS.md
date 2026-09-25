# Alternate GLM-5.3 Flash workflows

Use [README.md](README.md) for the native Q4 default. The workflows below
are retained for comparison and development; their historical performance
claims do not qualify the current native runner.

## Hybrid and Q2

The common launcher also supports compact safetensors-derived attention,
dense FFN and shared experts, with GGUF routed experts and vocabulary
boundaries:

```bash
GLM53F_NATIVE=0 bash a64fx/glm5/run_glm53f_12n.sh run
GLM53F_QUANT=q2 bash a64fx/glm5/run_glm53f_12n.sh run
```

Q2 defaults to the hybrid path and the first shard under
`$HOME/models/glm53f-gguf/`. Override it with `GLM53F_GGUF` or the legacy
`GLM53F_Q2_MODEL`; Q4 also accepts `GLM53F_Q4_MODEL`.
Q2 routed images occupy 8,252,817,408 bytes per rank, versus
15,456,534,528 for Q4. Both split experts into eight 256-column parts over
twelve ranks, preserving GGUF quantization blocks.

Hybrid mode explicitly clears inherited native projection stage variables.
Keep native and hybrid compact images separate. Conservative and fast-math
greedy sequences can differ; do not compare them as identical workloads.

## MTP / speculative decoding

The draft uses safetensors layer 45, not a Q4 conversion of that layer.
Stage it separately after each allocation restart:

```bash
bash a64fx/glm5/run_glm53f_12n.sh build all
bash a64fx/glm5/run_glm53f_mtp_stage_12n.sh
GLM53F_BUILD=0 GLM53F_SPEC_SELF_REFERENCE=1 \
  bash a64fx/glm5/run_glm53f_q4_mtp_12n.sh prompt.ids output.ids 128 1
```

The target stages must already exist. This wrapper now shares the native
target environment, manifest preflight, binary directory and fresh topology
with the main launcher. Export the same path overrides for both wrappers.
Use `GLM53F_NATIVE=0` explicitly to reproduce the older hybrid target.

Useful verifier controls:

- `GLM53F_SPEC_SELF_REFERENCE=1`: generate a scalar reference from the same
  warmed state; save `OUTPUT_IDS.greedy` and compare delivered tokens.
- `GLM53F_SPEC_REFERENCE_IDS=FILE`: compare against a retained reference;
  do not combine with self-reference.
- `GLM53F_SPEC_FULL_REPLAY=1`: use full draft replay instead of cache-only replay.
- `GLM53F_SPEC_DRAFT_SWEEP=1`: compare draft counts from the same state.
- `GLM53F_SPEC_COMPARE_BATCH=1`: compare baseline/optimized verifier modes;
  do not combine with the draft sweep.

The native five-position verifier passes, but the current native full
draft/verify/rollback loop still needs an end-to-end same-state run. Earlier
hybrid measurements with high draft acceptance did not establish native
greedy equivalence or a speculative speedup. A plain PASS is not that proof.

## Q8 resident llama.cpp path

This is a separate backend in `$HOME/work/llama.cpp`, not the native C
target runner. Build the optional stage tools with `build all`, then point
the existing Q8 wrappers at the shared binary directory:

```bash
bash a64fx/glm5/run_glm53f_12n.sh build all
export OPAL_PREFIX=/opt/FJSVxtclanga/tcsds-1.2.43
export MPI_HOME=$OPAL_PREFIX
export PATH="/opt/local/mpiexec:$MPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$MPI_HOME/lib64:${LD_LIBRARY_PATH:-}"
export GLM53F_Q8_STAGE_BIN="$PWD/a64fx/glm5/build/glm53f/glm53f_q8_stage"
export GLM53F_Q8_COPY_BIN="$PWD/a64fx/glm5/build/glm53f/glm53f_core_stage"
export GLM53F_Q8_RESIDENT_BIN="$PWD/a64fx/glm5/build/glm53f/test_glm53f_q8_resident"
q8="$HOME/models/glm53f-gguf-all/Q8_0/GLM-5.3-Flash-Q8_0-00001-of-00008.gguf"
images="$HOME/models/glm53f-q8-rank12-v1"
bash a64fx/glm5/run_glm53f_q8_stage_12n.sh "$q8"
bash a64fx/glm5/run_glm53f_q8_local_stage_12n.sh "$images"
bash a64fx/glm5/run_glm53f_q8_resident_mpi_12n.sh \
  "$q8" "$images" 'The capital of France is' 16 12 \
  "$PWD/tmp/glm53f-q8-$PJM_JOBID"
```

Set `GLM53F_Q8_IMAGE_ROOT` on the shared-stage command if using a different
image root. Each rank image is about 28.72 GB. Do not fully load it plus a
generation context into 32-GiB HBM. The wrapper defaults to lazy expert
loading and expert parallelism (`GGML_MPI_IMAGE_LAZY=1`,
`GGML_MPI_EXPERT_EP=1`). Monitor MemAvailable and retain the memory guard;
lowering it caused rank SIGKILL during earlier long generations.

The external fork's bounded expert cache uses eviction when the memory guard
rejects an admission. Confirm that implementation is present before long
runs. Require twelve `MPI Q8 resident-lazy` lines, per-step
`ranks_exact=yes`, and a final `MPI PASS`. Exact agreement between MPI ranks
does not establish agreement with the native C runner.

The external `llama-mpi --server` mode and tensor-dump interface remain
available in that tree. They are not a native HTTP server provided by this
module. Historical server experiments and detailed Q8 traces are in Git at
`a7ad56ff` (`GLM53F_Q8_RESIDENT.md`) and `tmp/glm53f-q8-*`.

## Developer tools kept outside the default build

`build all` retains MTP, Q8 resident tools, reference probes and specialized
INT8/prefill tests. Single-layer benchmarks, offline repacking and llama.cpp
reference programs remain available as source. They are diagnostic tools,
not additional supported production entry points.

The obsolete post-stage/final/spec-validation shell chain was removed: it
waited on process-name matches and embedded historical expected tokens.
Use the main `check` command and explicit MTP self-reference instead. Old
plans, handoff prompts and chronological job diaries are recoverable with
`git show a7ad56ff:a64fx/glm5/GLM53F_DECODE_PLAN.md` (and the former Q2/Q4,
preflight and validation documents). Current caveats are preserved in
[GLM53F_VALIDATION.md](GLM53F_VALIDATION.md).
