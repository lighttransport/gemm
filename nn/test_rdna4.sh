#!/bin/sh
# SPDX-License-Identifier: MIT
# Optional hardware correctness sweep; no SDK is needed by the default build.
set -eu
bench=${1:-nn/build/bench_rdna4}
for ta in 0 1; do
    for tb in 0 1; do
        "$bench" 35 67 49 "$ta" "$tb" 5
    done
done
"$bench" 1296 256 2304 0 1 30
"$bench" 256 2304 1296 1 0 30
"$bench" 1296 2304 256 0 0 30
