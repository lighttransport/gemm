#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
if [ -d /local ]; then
    build_dir=${GLM53F_BUILD_DIR:-/local/glm53f-integrated-build-${PJM_JOBID:-manual}}
else
    build_dir=${GLM53F_BUILD_DIR:-../../tmp/glm53f-integrated-build}
fi
mkdir -p "$build_dir"

mpi_include=${FJMPI_INCLUDE:-/opt/FJSVxtclanga/tcsds-1.2.43/include/mpi/fujitsu}
mpi_lib=${FJMPI_LIB:-/opt/FJSVxtclanga/tcsds-1.2.43/lib64}
cc=${FJCC:-fcc}
cflags=(-Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp
        -Wall -Wextra -I"$mpi_include")
ldflags=(-L"$mpi_lib" -lmpi -lm)
external=(-DGLM53F_EXTERNAL_ST_IMPLEMENTATION)

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

objects=("$build_dir/kda.o" "$build_dir/sparse.o" "$build_dir/dense.o"
         "$build_dir/moe.o" "$build_dir/head.o" "$build_dir/embedding.o")
TMPDIR="$build_dir" "$cc" "${cflags[@]}" glm53f_target_decode_12n.c \
    "${objects[@]}" "${ldflags[@]}" -o glm53f_target_decode_12n
TMPDIR="$build_dir" "$cc" "${cflags[@]}" glm53f_spec_decode_12n.c \
    "${objects[@]}" "$build_dir/target.o" "$build_dir/mtp.o" \
    "${ldflags[@]}" -o glm53f_spec_decode_12n
TMPDIR="$build_dir" "$cc" "${cflags[@]}" glm53f_target_batch_check_12n.c \
    "${objects[@]}" "$build_dir/target.o" \
    "${ldflags[@]}" -o glm53f_target_batch_check_12n
TMPDIR="$build_dir" "$cc" "${cflags[@]}" glm53f_kda_callback_check.c \
    "$build_dir/kda.o" "${ldflags[@]}" -o glm53f_kda_callback_check
TMPDIR="$build_dir" "$cc" "${cflags[@]}" \
    -DGLM53F_SPARSE_NO_MAIN glm53f_sparse_batch_check.c \
    glm53f_sparse_layer_12n.c "${ldflags[@]}" -o glm53f_sparse_batch_check

echo "SENTINEL glm53f_integrated_build_12n=OK build_dir=$build_dir"
