#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
module unload LLVM/llvmorg-21.1.0 2>/dev/null || true
unset OPAL_PREFIX
mkdir -p tmp/q38p/buildtmp
if [[ -d /local/q38/tmp ]]; then export TMPDIR=/local/q38/tmp
else export TMPDIR="$PWD/tmp/q38p/buildtmp"
fi

"${MPICC:-mpifcc}" -Nclang -O2 -march=armv8.2-a+sve -mcpu=a64fx \
    -DPF1_DIST=2048 -DPF2_DIST=32768 -DQ38P_MPI \
    -Wall -Wno-unused-function -Wno-unused-variable -Icommon -Ia64fx/llm \
    a64fx/llm/q38d/q38d_engine.c a64fx/llm/qwen38_lowbit_model.c \
    a64fx/llm/qwen38_lowbit.c a64fx/llm/qwen38_lowbit_sve.c \
    a64fx/llm/q38d/q38d_kern_sve.S a64fx/llm/q38d/q38d_kern_n4.S \
    a64fx/llm/q38p/q38p_kern.S a64fx/llm/q38p/q38p_attention.S -lpthread -lm -o tmp/q38p/q38p_pp
