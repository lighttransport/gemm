#!/bin/bash
# Direct execution on native A64FX host
set -e

cd "$(dirname "$0")"

make clean
make
./bench_cache_partition
