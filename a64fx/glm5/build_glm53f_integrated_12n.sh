#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
if ! grep -q 'int glm53f_st_read_columns' ../../common/glm53f_safetensors.h; then
    echo "error: canonical glm53f_safetensors.h with column reads is required" >&2
    exit 2
fi
if [ -d /local ]; then
    build_dir=${GLM53F_BUILD_DIR:-/local/glm53f-integrated-build-${PJM_JOBID:-manual}}
else
    build_dir=${GLM53F_BUILD_DIR:-../../tmp/glm53f-integrated-build}
fi
mkdir -p "$build_dir"

# The Fugaku login and compute environments can expose different MPI trees.
# Do not mix mpi.h from one tree with libmpi from another: use its compiler
# wrapper for both.  On the current OSS-CN LLVM environment mpiFCC is broken
# by OPAL_PREFIX, while mpiclang is the supported MPI C wrapper.  A caller may
# select another site wrapper explicitly through GLM53F_MPICC.
if [ -n "${GLM53F_MPICC:-}" ]; then
    cc=$GLM53F_MPICC
elif command -v mpiclang >/dev/null 2>&1; then
    cc=mpiclang
elif command -v mpiFCC >/dev/null 2>&1 && mpiFCC --showme:compile >/dev/null 2>&1; then
    cc=mpiFCC
elif command -v mpicc >/dev/null 2>&1; then
    cc=mpicc
else
    echo "error: no working MPI compiler wrapper (set GLM53F_MPICC)" >&2
    exit 2
fi

cflags=(-O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp
        -Wall -Wextra -I. -I../../common)
if [ "${GLM53F_FAST_MATH:-0}" = 1 ]; then
    cflags+=(-ffast-math)
fi
if [ "${GLM53F_NO_MATH_ERRNO:-0}" = 1 ]; then
    cflags+=(-fno-math-errno)
fi
if [ "${GLM53F_MHC_FUSED:-0}" = 1 ]; then
    cflags+=(-DGLM53F_MHC_FUSED=1)
fi
if [ "${GLM53F_MHC_POST_FLOAT:-0}" = 1 ]; then
    cflags+=(-DGLM53F_MHC_POST_FLOAT=1)
fi
if [ -n "${GLM53F_MOE_FUSED_WEIGHTED:-}" ]; then
    cflags+=("-DGLM53F_MOE_FUSED_WEIGHTED=$GLM53F_MOE_FUSED_WEIGHTED")
fi
# The MPI wrapper supplies its matching MPI include and link flags.  uTofu is
# deliberately explicit because the model's collective implementation uses it
# directly when GLM53F_UTOFU=1.
ldflags=(-lm -ltofucom)
external=(-DGLM53F_EXTERNAL_ST_IMPLEMENTATION)

TMPDIR="$build_dir" "$cc" "${cflags[@]}" \
    -c glm53f_collective_12n.c -o "$build_dir/collective.o"

TMPDIR="$build_dir" "$cc" "${cflags[@]}" -DGLM53F_KDA_NO_MAIN \
    -c glm53f_kda_layer_12n.c -o "$build_dir/kda.o"
TMPDIR="$build_dir" "$cc" "${cflags[@]}" "${external[@]}" \
    -DGLM53F_SPARSE_NO_MAIN -c glm53f_sparse_layer_12n.c -o "$build_dir/sparse.o"
TMPDIR="$build_dir" "$cc" "${cflags[@]}" "${external[@]}" \
    -DGLM53F_DENSE_NO_MAIN -c glm53f_dense_ffn_12n.c -o "$build_dir/dense.o"
TMPDIR="$build_dir" "$cc" "${cflags[@]}" "${external[@]}" \
    -DGLM53F_EXPERT_NO_MAIN -c glm53f_expert_decode_12n.c -o "$build_dir/moe.o"
TMPDIR="$build_dir" "$cc" "${cflags[@]}" "${external[@]}" \
    -DGLM53F_TARGET_HEAD_NO_MAIN -c glm53f_target_head_12n.c -o "$build_dir/head.o"
TMPDIR="$build_dir" "$cc" "${cflags[@]}" "${external[@]}" \
    -c glm53f_embedding_12n.c -o "$build_dir/embedding.o"
TMPDIR="$build_dir" "$cc" "${cflags[@]}" \
    -DGLM53F_TARGET_MODEL_NO_MAIN -c glm53f_target_decode_12n.c -o "$build_dir/target.o"
TMPDIR="$build_dir" "$cc" "${cflags[@]}" "${external[@]}" \
    -c glm53f_mtp_12n.c -o "$build_dir/mtp.o"

objects=("$build_dir/collective.o" "$build_dir/kda.o" "$build_dir/sparse.o" "$build_dir/dense.o"
         "$build_dir/moe.o" "$build_dir/head.o" "$build_dir/embedding.o")
TMPDIR="$build_dir" "$cc" "${cflags[@]}" glm53f_target_decode_12n.c \
    "${objects[@]}" "${ldflags[@]}" -o glm53f_target_decode_12n
TMPDIR="$build_dir" "$cc" "${cflags[@]}" glm53f_spec_decode_12n.c \
    "${objects[@]}" "$build_dir/target.o" "$build_dir/mtp.o" \
    "${ldflags[@]}" -o glm53f_spec_decode_12n
TMPDIR="$build_dir" "$cc" "${cflags[@]}" glm53f_target_batch_check_12n.c \
    "${objects[@]}" "$build_dir/target.o" \
    "${ldflags[@]}" -o glm53f_target_batch_check_12n
TMPDIR="$build_dir" "$cc" "${cflags[@]}" glm53f_prefill_12n.c \
    "${objects[@]}" "$build_dir/target.o" \
    "${ldflags[@]}" -o glm53f_prefill_12n
TMPDIR="$build_dir" "$cc" "${cflags[@]}" glm53f_kda_callback_check.c \
    "$build_dir/kda.o" "$build_dir/collective.o" "${ldflags[@]}" -o glm53f_kda_callback_check
TMPDIR="$build_dir" "$cc" "${cflags[@]}" \
    -DGLM53F_SPARSE_NO_MAIN glm53f_sparse_batch_check.c \
    glm53f_sparse_layer_12n.c "$build_dir/collective.o" \
    "${ldflags[@]}" -o glm53f_sparse_batch_check

echo "SENTINEL glm53f_integrated_build_12n=OK cc=$cc build_dir=$build_dir"
