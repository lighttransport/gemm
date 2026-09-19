#!/bin/sh
set -eu

epochs=${EPOCHS:-20}
output=${OUTPUT:-hbm-color-epochs.csv}
binary=${BINARY:-./bench_hbm_color}

: > "$output"
epoch=0
while [ "$epoch" -lt "$epochs" ]; do
    printf '# epoch=%d\n' "$epoch" | tee -a "$output"
    "$binary" --cores 12 --core-base 12 --mib 240 \
        --iterations 5 --trials 3 --min-skew-kib 0 --max-skew-kib 64 \
        --step-bytes 16384 "$@" | tee -a "$output"
    epoch=$((epoch + 1))
done
