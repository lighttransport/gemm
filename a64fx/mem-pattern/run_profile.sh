#!/bin/bash
# Profiling script for memory access pattern analysis
# Uses Linux perf and Fujitsu fapp for A64FX detailed analysis

cd /vol0006/mdt0/data/hp250467/work/gemm/a64fx/mem-pattern

echo "=============================================="
echo "Memory Access Pattern Profiling"
echo "=============================================="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo ""

mkdir -p perf_output fapp_output

# ============================================
# Section 1: Linux perf with hardware events
# ============================================
echo "=== Linux perf stat - Cycles & Instructions ==="
perf stat -e cycles,instructions,stalled-cycles-frontend,stalled-cycles-backend \
    ./bench_mem_pattern -i 10000 2>&1 | tee perf_output/perf_cycles.txt

echo ""
echo "=== Linux perf stat - L1 Data Cache ==="
perf stat -e L1-dcache-loads,L1-dcache-load-misses,cache-references,cache-misses \
    ./bench_mem_pattern -i 10000 2>&1 | tee perf_output/perf_l1d.txt

echo ""
echo "=== Linux perf stat - Branch Prediction ==="
perf stat -e branch-loads,branch-load-misses,br_pred,br_mis_pred \
    ./bench_mem_pattern -i 10000 2>&1 | tee perf_output/perf_branch.txt

echo ""
echo "=== Linux perf stat - TLB ==="
perf stat -e dTLB-load-misses,iTLB-load-misses,L1-icache-loads,L1-icache-load-misses \
    ./bench_mem_pattern -i 10000 2>&1 | tee perf_output/perf_tlb.txt

# ============================================
# Section 2: Fujitsu fapp profiling
# ============================================
echo ""
echo "=== Fujitsu fapp - CPU Performance Analysis ==="
rm -rf fapp_output/cpupa
fapp -C -d fapp_output/cpupa -Icpupa ./bench_mem_pattern -i 10000 2>&1
fapppx -A -d fapp_output/cpupa -Icpupa -ttext -o fapp_output/cpupa.txt 2>&1
cat fapp_output/cpupa.txt

# ============================================
# Section 3: Summary Analysis
# ============================================
echo ""
echo "=============================================="
echo "PROFILING SUMMARY"
echo "=============================================="
echo ""
echo "Key Metrics from perf:"
grep -E "cycles|instructions|L1-dcache|cache-misses|stalled" perf_output/*.txt 2>/dev/null | grep -v "^--"

echo ""
echo "Key Metrics from fapp:"
echo "  - IPC (Instructions Per Cycle)"
echo "  - SIMD instruction rate"
echo "  - Memory throughput"
grep -E "IPC|SIMD|Mem throughput|GIPS" fapp_output/cpupa.txt 2>/dev/null | head -10

echo ""
echo "=============================================="
echo "BOTTLENECK ANALYSIS"
echo "=============================================="
echo ""
echo "Expected behavior for memory-access-only kernel:"
echo "  - 0 GFLOPS (NOPs replace SDOT)"
echo "  - High IPC (~3.5) due to simple NOP+load pattern"
echo "  - Low L1 cache miss rate (working set fits in L1)"
echo "  - Memory throughput limited by load unit throughput"
echo ""
echo "Observed: ~1335 cycles for 34KB loads = 26 bytes/cycle"
echo "Peak: 128 bytes/cycle (2x64B load pipes)"
echo "Efficiency: ~20%"
echo ""
echo "Bottleneck hypothesis:"
echo "  1. Load latency not hidden (11 cycles L1, no compute to overlap)"
echo "  2. Address generation overhead (pointer arithmetic)"
echo "  3. Loop control overhead (branch, counter update)"
echo ""
echo "=============================================="
echo "Profiling complete at $(date)"
echo "=============================================="
