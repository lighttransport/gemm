#!/bin/bash
# Build sector cache test using FCC traditional mode (required for OCL pragmas)
#
# Usage:
#   ./build_sector_test.sh          # cross-compile (login node)
#   ./build_sector_test.sh native   # native compile (compute node)

set -e

CC=fcc

SRC=test_sector_cache.c
BIN=test_sector_cache
ASM=test_sector_cache.s

echo "=== Building sector cache test with $CC ==="

# Compile to assembly (to inspect emitted sector cache instructions)
echo "Generating assembly..."
$CC -Nnoclang -O2 -Kocl,hpctag -S -o $ASM $SRC 2>&1
echo "  -> $ASM"

# Show key sector cache instructions in assembly
echo ""
echo "=== Sector Cache Instructions in Assembly ==="
echo ""
echo "--- __jwe_xset_sccr calls (sector cache control register setup) ---"
grep -n "__jwe_xset_sccr\|__jwe_check_hpctag" $ASM || true
echo ""
echo "--- ORR tagged address (bit 56 = sector 1) ---"
grep -n "orr.*72057594037927936\|orr.*0x01000000" $ASM || true
echo ""
echo "--- SCCR config values (way partition) ---"
grep -n "131074\|196608\|mov.*w0, 256\|mov.*w0, 512" $ASM || true

# Compile to executable
echo ""
echo "Compiling executable..."
$CC -Nnoclang -O2 -Kocl,hpctag -o $BIN $SRC 2>&1
echo "  -> $BIN"

# Also generate compiler optimization listing
echo ""
echo "Generating optimization listing..."
$CC -Nnoclang -O2 -Kocl,hpctag -Nsrc -S -o ${ASM%.s}_listing.s $SRC 2>&1
echo "  -> ${ASM%.s}_listing.s"

echo ""
echo "=== Build complete ==="
echo "Run on compute node: ./test_sector_cache"
echo "Or submit: pjsub run_sector_test.sh"
