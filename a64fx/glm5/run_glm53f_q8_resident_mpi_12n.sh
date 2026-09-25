#!/bin/bash
set -euo pipefail

# Run the llama.cpp MPI graph against rank-local Q8 images.  Each rank sees
# the same path name, but /local is node-local, so rank N resolves rankNN.blob
# on its own A64FX node.
model=${1:?first Q8_0 GGUF shard is required}
image_root=${2:?shared image root is required}
prompt=${3:?prompt is required}
tokens=${4:-1}
threads=${5:-12}
run_dir=${6:-$PWD/tmp/glm53f-q8-resident-${PJM_JOBID:-local}}
ranks=${PJM_MPI_PROC:-12}
mpi_bin=${GLM53F_MPI_BIN:-$HOME/work/llama.cpp/build-a64fx-mpi/bin/llama-mpi}
job=${PJM_JOBID:-manual}
local_root=${GLM53F_Q8_LOCAL_ROOT:-/local/glm53f-q8-$job}

test "$ranks" -eq 12
test -x "$mpi_bin"
test -f "$image_root/COMPLETE"
test -d "$local_root"

# /local is node-private: rank00 is visible here, while rank01..rank11 live on
# their respective nodes.  Validate each pair from the rank that will consume
# it instead of assuming all twelve files are visible on the submit node.
mpiexec -np "$ranks" sh -c '
  r=${PMIX_RANK:-${PJM_MPI_RANK:-${OMPI_COMM_WORLD_RANK:-0}}}
  tag=$(printf "%02d" "$r")
  test -s "$1/rank${tag}.blob"
  test -s "$1/rank${tag}.manifest"
' sh "$local_root"

mkdir -p "$run_dir"
run_dir=$(cd "$run_dir" && pwd)
export GGML_MPI_IMAGE_DIR="$local_root"
export GGML_MPI_EXPERT_EP=${GGML_MPI_EXPERT_EP:-1}
export GGML_MPI_IMAGE_LAZY=${GGML_MPI_IMAGE_LAZY:-1}
export NUMA_INTERLEAVE=${NUMA_INTERLEAVE:-1}
export TMPDIR=${TMPDIR:-/local/$USER/llama-mpi-tmp}
mkdir -p "$TMPDIR"

mpiexec -np "$ranks" -stdout-proc "$run_dir/out" -stderr-proc "$run_dir/err" \
    "$mpi_bin" "$model" "$prompt" "$tokens" "$threads" "$run_dir/logits.f32"

printf 'SENTINEL glm53f_q8_resident_mpi_12n=OK image=%s local=%s run=%s\n' \
    "$image_root" "$local_root" "$run_dir"
