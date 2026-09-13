#!/bin/sh
# SPDX-License-Identifier: MIT
# Optional GPU diagnostics. Approximation error is reported separately from
# exact integer accumulation, BF16 representable cases and packing checks.
set -eu
bench=${1:-nn/build/bench_rdna4_precision}
"$bench" check
for ta in 0 1; do
    for tb in 0 1; do
        "$bench" 35 67 49 "$ta" "$tb" 5
        "$bench" 3 2 1 "$ta" "$tb" 5
    done
done
"$bench" 1296 256 2304 0 1 100
"$bench" 256 2304 1296 1 0 100
"$bench" 1296 2304 256 0 0 100
