#!/bin/bash
# Cross-compile cpu LLM/VLM for A64FX using Fujitsu fccpx/FCCpx
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_a64fx"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake -DCMAKE_TOOLCHAIN_FILE="${SCRIPT_DIR}/toolchain-a64fx-cross.cmake" \
      "${SCRIPT_DIR}"

echo ""
echo "Configured. To build:"
echo "  cd ${BUILD_DIR} && make"
