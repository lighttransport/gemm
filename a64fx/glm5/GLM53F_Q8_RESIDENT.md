# GLM-5.3-Flash Q8_0 rank images

`glm53f_q8_stage` builds one contiguous image for one MPI rank from the first
shard of a split GGUF. It opens all eight shard headers in metadata-only mode;
model payloads are copied with bounded `pread` calls, so staging does not mmap
or fault the 310 GiB model. Dense matrices are row-sharded, routed expert
tensors are assigned contiguous expert ranges, and small tensors are
replicated.

Build the stage tool with the native Fugaku MPI wrapper after unloading the
LLVM module:

```sh
module unload LLVM/llvmorg-21.1.0
unset OPAL_PREFIX
GLM53F_MPICC=mpifcc ./build_glm53f_integrated_12n.sh
```

The build emits `glm53f_q8_stage` and `test_glm53f_q8_resident`. The converter
supports a metadata-only check that is safe in an interactive allocation:

```sh
./glm53f_q8_stage \
  "$HOME/models/glm53f-gguf-all/Q8_0/GLM-5.3-Flash-Q8_0-00001-of-00008.gguf" \
  tmp/q8-rank00 0 12 --dry-run
```

After all twelve dry runs pass, stage the real images once per allocation:

```sh
GLM53F_Q8_IMAGE_ROOT="$HOME/models/glm53f-q8-rank12-v1" \
  ./run_glm53f_q8_stage_12n.sh \
  "$HOME/models/glm53f-gguf-all/Q8_0/GLM-5.3-Flash-Q8_0-00001-of-00008.gguf"
```

The image root is shared storage. A later `/local` copier should copy only
`rank%02d.blob` and its manifest to the matching node in bounded chunks, then
call `glm53f_q8_resident_load()` on that local directory. The loader allocates
anonymous memory, uses 64 MiB reads, drops source page cache as it progresses,
checks `MemAvailable` before committing the next chunk, and validates the
manifest FNV-1a checksum and every tensor record before publishing pointers.

The repository includes that copier and an optional load-only HBM test:

```sh
GLM53F_Q8_RESIDENT_TEST=1 \
  ./run_glm53f_q8_local_stage_12n.sh \
  "$HOME/models/glm53f-q8-rank12-v1"
```

The metadata dry run currently reports approximately 28.72 GB per rank for
this checkpoint: 27.59 GB routed experts, 0.79 GB row-sharded matrices, and
0.33 GB replicated tensors. The resident loader is deliberately a separate
API from the existing safetensors GLM path; the llama.cpp MPI bridge below
maps its manifest entries to GGML tensors for the resident path.

## llama.cpp graph binding

The MPI CPU cache in `~/work/llama.cpp` accepts `GGML_MPI_IMAGE_DIR`. For
`mode=0` (replicated) tensors it returns a pointer directly into the anonymous
resident image. For `mode=1` (row-TP) tensors it materializes a full-shaped
zero-backed tensor and copies only the rank-owned rows; existing MPI
`MUL_MAT` row-gather semantics are unchanged. The GGUF mapping remains as
metadata/fallback while weight bytes come from `/local` and stay resident in
HBM2.

`mode=2` expert tensors use an opt-in contiguous-range `MUL_MAT_ID` path in
the updated MPI backend. Each rank maps only its local expert slice, skips
non-owned experts, and allreduces the output. With the opt-in disabled they
remain on the safe GGUF fallback.

After local staging, use `run_glm53f_q8_resident_mpi_12n.sh` to set the image
binding and run the existing MPI CLI. The wrapper enables
`GGML_MPI_EXPERT_EP=1` and lazy expert loading (`GGML_MPI_IMAGE_LAZY=1`); set
either to `0` to retain the old expert fallback or to force the full-image
loader for a residency-only test. Lazy mode loads dense/replicated bytes up
front and faults only selected local expert slices into HBM2. Those slices
remain cached for later requests in the same process, avoiding rereads of
already-used weights. Keep monitoring `MemAvailable`: a long generation can
exhaust the remaining HBM2 while new experts are demand-loaded.

## Direct execution inside an A64FX 12-node interactive job

When the shell is already inside a Fugaku interactive allocation, run the
workflow directly; no bash-over-HTTP bridge is needed. This procedure is
intentionally guarded for AArch64 and the 12-rank/12-node allocation:

```sh
test "$(uname -m)" = aarch64
test "${PJM_NODE:?}" -eq 12
test "${PJM_MPI_PROC:?}" -eq 12
grep -q '^CPU implementer[[:space:]]*: 0x46' /proc/cpuinfo

module unload LLVM/llvmorg-21.1.0
unset OPAL_PREFIX
export OPAL_PREFIX=/opt/FJSVxtclanga/tcsds-1.2.43
export MPI_HOME="$OPAL_PREFIX"
export PATH="/opt/local/mpiexec:$OPAL_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$OPAL_PREFIX/lib64:${LD_LIBRARY_PATH:-}"
export TMPDIR="/local/$USER/llama-mpi-tmp"
mkdir -p "$TMPDIR"

cd "$HOME/work/gemm/glm53f"
q8="$HOME/models/glm53f-gguf-all/Q8_0/GLM-5.3-Flash-Q8_0-00001-of-00008.gguf"
images="$HOME/models/glm53f-q8-rank12-v1"
local_root="/local/glm53f-q8-$PJM_JOBID"
run="$PWD/tmp/glm53f-q8-direct-12n-$PJM_JOBID"
mkdir -p "$run"

GLM53F_Q8_LOCAL_ROOT="$local_root" \
  a64fx/glm5/run_glm53f_q8_local_stage_12n.sh "$images"

GGML_MPI_IMAGE_LAZY=1 GGML_MPI_EXPERT_EP=1 \
GLM53F_Q8_LOCAL_ROOT="$local_root" \
  a64fx/glm5/run_glm53f_q8_resident_mpi_12n.sh \
  "$q8" "$images" 'The capital of France is' 16 12 "$run" \
  2>&1 | tee "$run/launch.log"
```

Require twelve `MPI Q8 resident-lazy` lines, `ranks_exact=yes` for every
step, and a final `MPI PASS`. The local staging step is idempotent for a
completed image and keeps the 28.7-GB rank image node-local; do not copy or
concatenate the eight GGUF shards.

The sixth argument above is the run directory expected by the wrapper. The
wrapper creates `logits.f32` and the per-rank `out.*`/`err.*` files inside it.

The direct 12-node continuation was validated in interactive job `51843198`:
all twelve ranks reported `MPI Q8 resident-lazy`, all sixteen steps reported
`ranks_exact=yes`, and the run ended with `MPI PASS ranks=12 elapsed=56.728 s`.
The five-token prompt took 18.124 s to prefill; the fifteen generated tokens
took 38.559 s, or 0.389 tokens/s. The output began:

```text
Paris. In French, Paris is spelled "Paris", but the pronunciation is different
```

## Persistent HTTP/background mode

For repeated prompts in the same 12-node allocation, start the MPI CLI in
server mode after local staging. It keeps the llama model and MPI image cache
alive between requests. Use lazy mode for this 32-GiB/rank GLM Q8 image;
full-image mode (`GGML_MPI_IMAGE_LAZY=0`) loads all 28.7 GB but leaves too
little headroom for a generation context on the tested nodes:

```sh
model="$HOME/models/glm53f-gguf-all/Q8_0/GLM-5.3-Flash-Q8_0-00001-of-00008.gguf"
local_root="/local/glm53f-q8-$PJM_JOBID"
log="$PWD/tmp/glm53f-http-$PJM_JOBID"
mkdir -p "$log"
export GGML_MPI_IMAGE_DIR="$local_root"
export GGML_MPI_IMAGE_LAZY=1 GGML_MPI_EXPERT_EP=1
nohup mpiexec -np 12 -stdout-proc "$log/out" -stderr-proc "$log/err" \
  "$HOME/work/llama.cpp/build-a64fx-mpi/bin/llama-mpi" \
  --server "$model" 18080 12 >"$log/launcher.log" 2>&1 &
echo $! >"$log/server.pid"
```

Use `GET /health`, then send plain-text prompts with `POST /generate` and an
optional `X-Tokens: 1..1024` header. Requests are serialized. `/shutdown`
terminates all twelve ranks and releases HBM2; do not kill the launcher while
a request is active.

This mode was tested directly in AArch64 interactive job `51843198` on port
18084: `/health` returned `ready`, two serialized one-token requests returned
`Paris` and `a`, and `/shutdown` returned `shutting down`. The twelve
`MPI Q8 resident-lazy` lines appeared only at startup, confirming one
long-lived MPI process handled both requests. A concise C++17 coding request
with `X-Tokens: 1024` reached 810 exact-rank decode steps with
`GGML_MPI_MEM_GUARD_GIB=1`; rank 11 was then SIGKILLed while admitting another
expert slice. The default 2-GiB guard stops earlier at 666 steps. The server
buffers the response until completion, so neither guarded request returned a
partial body; a complete ~1k response needs a larger-memory node or a shorter
coding completion.

A shorter C++ coding task completed through the same persistent service with
the 1-GiB override: 442 exact-rank tokens, a 1,546-byte HTTP body, and
345.26 s elapsed. After removing the model's Markdown fences, native
`mpiFCC -Nclang -std=c++17 -O2 -Wall -Wextra -Werror -pedantic` compiled it;
the sample graph test returned `2`, `2`, `2`, `-1`. Complete output is
therefore possible below the HBM limit, but before bounded eviction an
unconstrained ~1k completion was not safe on the current 32-GiB/rank Q8 image.

The lazy cache now evicts least-recently-used expert slices with
`madvise(MADV_DONTNEED)` when the `MemAvailable` guard rejects the next
bounded read. Partial expert loads that cross the guard are discarded before
eviction and retried. A per-expert worker barrier prevents eviction while a
slower compute thread still consumes the prior slice. In 12-node job
`51848233`, an 8-GiB guard deliberately forced the path: eviction began at
decode step 120, steps 121--127 remained exact across all ranks, and the run
ended with `MPI PASS ranks=12 elapsed=320.158 s`.

## Intermediate tensor dumps

The MPI runner can dump named graph intermediates for comparison with another
runner. Tensor dumping is disabled unless `LLAMA_MPI_DUMP_DIR` is set.
`LLAMA_MPI_DUMP_MODULES` is a required comma-separated list of module names or
shell patterns. Layer and decode-step filters accept comma-separated integers
and inclusive ranges; `*` selects all values. Final tensors such as
`result_norm` and `result_output` use layer `-1`.
Use a new dump directory for each invocation; the runner refuses to overwrite
an existing `manifest.tsv`.

```sh
export LLAMA_MPI_DUMP_DIR="$PWD/tmp/glm53f-dump"
export LLAMA_MPI_DUMP_MODULES='kda_qkv,kda_out,dsa_*,ffn_moe_topk,ffn_out,l_last,result_*'
export LLAMA_MPI_DUMP_LAYERS='-1,0-2,44'
export LLAMA_MPI_DUMP_STEPS='0-1'
```

Only rank 0 writes files, but all ranks use the same evaluation callback
boundaries to preserve MPI collective ordering. Each `.bin` file contains the
complete tensor in native GGML dtype and byte layout. `manifest.tsv` records
the request, decode step, layer, module and tensor names, GGML operation,
dtype, four dimensions, four byte strides, byte count, and file name. A
persistent server increments `request` for each generation, so later requests
do not overwrite earlier dumps. Dump only the modules, layers, and steps under
investigation because observing a node synchronizes graph execution and full
tensors can consume substantial storage.
