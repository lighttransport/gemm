#!/bin/sh
# M1 build helper — runs `make CC=fcc` and tees output for inspection.
set -eu
cd "$(dirname "$0")/.."
make CC=fcc "$@" 2>&1 | tee /tmp/vlm_m1_build.log
