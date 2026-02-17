#!/bin/bash
set -e

BLIS_DIR=/home/syoyo/work/gemm/ref/blis

echo "=== Step 1: Configure BLIS for zen2 (no threading) ==="
cd "$BLIS_DIR"
./configure --enable-threading=no zen2

echo ""
echo "=== Step 2: Build BLIS ==="
make -j$(nproc)

echo ""
echo "=== Step 3: Compile benchmark ==="
BLIS_INC="$BLIS_DIR/include/zen2"
BLIS_LIB="$BLIS_DIR/lib/zen2/libblis.a"

cd /home/syoyo/work/gemm/zen2

clang -O3 -march=znver2 -mavx2 -mfma -ffp-contract=fast \
    -I"$BLIS_INC" \
    -o bench_blis bench_blis.c "$BLIS_LIB" -lm -lpthread

echo ""
echo "=== Step 4: Run benchmark ==="
OMP_NUM_THREADS=1 ./bench_blis
