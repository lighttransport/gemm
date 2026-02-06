# A64FX Hardware Prefetch Benchmark

Benchmark suite to measure the effects of hardware prefetch (L1 and L2) on Fujitsu A64FX processor using `hwpfctl`.

## Overview

The A64FX has configurable hardware prefetchers for both L1 and L2 cache. This benchmark measures:
- Bandwidth achieved with different prefetch settings
- Impact of prefetch on various memory access patterns
- Optimal prefetch distance for different working set sizes

## A64FX Cache Architecture

| Level | Size | Line Size | Latency | Peak BW |
|-------|------|-----------|---------|---------|
| L1 | 64KB | 256B (sector) | ~5 cycles | 128 B/cycle |
| L2 | 8MB (per CMG) | 256B | ~40 cycles | ~42 B/cycle |
| HBM2 | 32GB | - | ~260 cycles | ~10 B/cycle/core |

## Building

```bash
# On login node (cross-compile)
make

# On compute node (native compile)
make native
```

## Usage

### Basic usage

```bash
# Run single pattern
./hwpf_bench --pattern seq_read_64 --size 1M

# Run all patterns at 1MB
./hwpf_bench --all --size 1M

# CSV output
./hwpf_bench --all --size 1M --csv > results.csv
```

### With hwpfctl

```bash
# Disable L1 prefetch
hwpfctl --disableL1 -- ./hwpf_bench --all --size 1M

# Disable both prefetchers
hwpfctl --disableL1 --disableL2 -- ./hwpf_bench --all --size 1M

# Set prefetch distances
hwpfctl --distL1 4 --distL2 16 -- ./hwpf_bench --all --size 64M

# Weak prefetch mode
hwpfctl --weakL1 --weakL2 -- ./hwpf_bench --all --size 1M
```

### Job submission

```bash
# Full test suite (all configurations)
pjsub run_all_tests.sh

# Prefetch distance sweep
pjsub run_prefetch_sweep.sh
```

## Access Patterns

### Basic Patterns

| Pattern | Description | Prefetch Benefit |
|---------|-------------|------------------|
| `seq_read_64` | Sequential 64B vector loads | High |
| `seq_read_256` | Sequential 256B cache line loads | High |
| `seq_write_64` | Sequential 64B stores | Moderate |
| `seq_write_256` | Sequential 256B stores | Moderate |
| `seq_copy` | Sequential read+write | High |
| `stride_128` | 128-byte stride | High |
| `stride_256` | 256-byte stride (cache line) | Moderate |
| `stride_512` | 512-byte stride | Low |
| `stride_1024` | 1KB stride | Low |
| `stride_4096` | 4KB stride (page) | Minimal |
| `random` | Random 64B access | Minimal |
| `ptr_chase` | Pointer chase (latency) | None |

### Software Prefetch Patterns

| Pattern | Description | PRFM Hint |
|---------|-------------|-----------|
| `swpf_l1keep` | SW prefetch to L1, temporal | `pldl1keep` |
| `swpf_l1strm` | SW prefetch to L1, streaming | `pldl1strm` |
| `swpf_l2keep` | SW prefetch to L2, temporal | `pldl2keep` |
| `swpf_l2strm` | SW prefetch to L2, streaming | `pldl2strm` |
| `swpf_dist4` | SW prefetch, 4 lines ahead (1KB) | `pldl1keep` |
| `swpf_dist8` | SW prefetch, 8 lines ahead (2KB) | `pldl1keep` |
| `swpf_dist16` | SW prefetch, 16 lines ahead (4KB) | `pldl1keep` |
| `swpf_l1l2` | Combined L1+L2 SW prefetch | both |

### GEMM Sector Cache Patterns

| Pattern | Description | A hint | B/C hint |
|---------|-------------|--------|----------|
| `gemm_baseline` | No hints, no prefetch | - | - |
| `gemm_swpf` | All KEEP prefetch (no sector) | `pldl1keep` | `pldl1keep` |
| `gemm_sector` | Sector differentiation | `pldl1strm` | `pldl1keep` |
| `gemm_optimal` | Sector + two-level prefetch | `pldl1strm` + `pldl2keep` | `pldl1keep` |
| `sector_mixed` | A=STRM scan + B=KEEP resident | `pldl1strm` | `pldl1keep` |
| `nosect_mixed` | A=KEEP scan + B=KEEP resident | `pldl1keep` | `pldl1keep` |

## hwpfctl Options

| Option | Description |
|--------|-------------|
| `--disableL1` | Disable L1 hardware prefetch |
| `--disableL2` | Disable L2 hardware prefetch |
| `--distL1 <n>` | Set L1 prefetch distance (1-15) |
| `--distL2 <n>` | Set L2 prefetch distance (4-60) |
| `--weakL1` | Enable weak L1 prefetch mode |
| `--weakL2` | Enable weak L2 prefetch mode |

## Benchmark Results and Findings

### HW Prefetch Impact (Sequential Read @ 64MB)

| Configuration | Bandwidth (B/cycle) | vs Baseline |
|---------------|---------------------|-------------|
| HW prefetch enabled | 14.3 | 1.0x |
| HW prefetch disabled | 8.0 | 0.56x (1.8x slower) |

**Key observation**: Disabling HW prefetch causes ~1.8x bandwidth loss for streaming workloads.

### SW Prefetch vs HW Prefetch (64MB Working Set)

| Pattern | HW Enabled | HW Disabled | Resilience |
|---------|------------|-------------|------------|
| `seq_read_256` (no SW) | 14.3 B/cy | 8.0 B/cy | 1.8x loss |
| `swpf_l1keep` | 11.1 B/cy | 10.7 B/cy | 1.03x loss |
| `swpf_l2keep` | **16.0 B/cy** | **14.1 B/cy** | 1.13x loss |
| `swpf_l1l2` | 14.3 B/cy | 14.0 B/cy | 1.02x loss |

### SW Prefetch Distance Impact (64MB, HW Disabled)

| Distance | Lookahead | Bandwidth |
|----------|-----------|-----------|
| 4 lines | 1KB | 8.2 B/cy |
| 8 lines | 2KB | 11.0 B/cy |
| 16 lines | 4KB | 13.3 B/cy |

**Optimal distance**: 16 cache lines (4KB) ahead provides best bandwidth.

### Key Findings

1. **SW L2 prefetch outperforms HW-only by 12%**
   - `swpf_l2keep`: 16.0 B/cycle
   - `seq_read_256` with HW: 14.3 B/cycle
   - Explicit `prfm pldl2keep` is more effective than relying on HW prefetcher

2. **SW prefetch is resilient when HW is disabled**
   - Without SW prefetch: 1.8x bandwidth loss
   - With SW L2 prefetch: only 1.1x loss
   - SW prefetch recovers most performance

3. **L2 prefetch beats L1 prefetch for streaming**
   - `swpf_l2keep`: 16.0 B/cy (best)
   - `swpf_l1keep`: 11.1 B/cy
   - L2 prefetch has more time to complete

4. **Combined L1+L2 is most stable**
   - `swpf_l1l2` achieves ~14.3 B/cy regardless of HW setting
   - Good choice when HW prefetch state is unknown

### Bandwidth by Working Set Size

| Size | Cache Level | seq_read_256 | swpf_l2keep |
|------|-------------|--------------|-------------|
| 256KB | L2 | 15.8 B/cy | 15.9 B/cy |
| 1MB | L2 | 16.0 B/cy | 16.0 B/cy |
| 8MB | L2 boundary | 14.1 B/cy | 16.0 B/cy |
| 64MB | Memory | 14.3 B/cy | 16.0 B/cy |

**Note**: SW prefetch shows most benefit at L2 boundary and beyond (>8MB).

### Pointer Chase Latency (cycles/access)

| Size | Latency | Cache Level |
|------|---------|-------------|
| 32KB | ~4 cy | L1 |
| 256KB | ~50 cy | L2 |
| 64MB | ~250 cy | Memory |

## Sector Cache for GEMM

### A64FX Sector Cache Architecture

```
A64FX L1D: 64KB, 4-way set associative, 256B cache line
           = 64 sets x 4 ways x 256B

Sector partitioning (2 effective sectors on A64FX):
  Sector 0 (STRM):  ~1 way  = 16KB   ← streaming data (A matrix)
  Sector 1 (KEEP):  ~3 ways = 48KB   ← resident data  (B + C tiles)

PRFM hint controls sector assignment:
  pldl1strm → streaming sector (limited way occupancy)
  pldl1keep → normal sector    (full associativity)
```

### GEMM Way Budget

For `C[m,n] += A[m,k] * B[k,n]`:

```
Matrix │ Role                  │ Sector │ Ways │ Budget │ Hint
───────┼───────────────────────┼────────┼──────┼────────┼──────────
A      │ Streams along K       │ STRM   │  1   │ 16KB   │ pldl1strm
B      │ Reused across M rows  │ KEEP   │  2   │ 32KB   │ pldl1keep
C      │ Accumulator tile      │ KEEP   │  1   │ 16KB   │ pldl1keep
Total  │                       │        │  4   │ 64KB   │
```

### GEMM Benchmark Results (B/cycle)

| Pattern | 8M HW On | 8M HW Off | 64M HW On | 64M HW Off |
|---------|----------|-----------|-----------|------------|
| `gemm_baseline` (no hints) | **4.60** | 2.30 | **4.67** | 2.31 |
| `gemm_swpf` (all KEEP) | 3.87 | 3.67 | 3.65 | 3.64 |
| `gemm_sector` (A=STRM) | 3.90 | 1.02 | 3.18 | 1.01 |
| `gemm_optimal` (sector+L2) | 2.59 | 2.36 | 2.38 | 2.35 |

### GEMM Analysis

1. **HW prefetch alone wins on raw bandwidth** (4.6 B/cy)
   - A64FX HW prefetcher is very effective at streaming sequential A data
   - No SW instruction overhead

2. **`gemm_swpf` (all KEEP) is HW-resilient** (~3.7 B/cy regardless)
   - SW prefetch compensates when HW is disabled
   - But all data competes equally for L1 → lower than HW-only

3. **`gemm_sector` (STRM for A) requires HW prefetch** (3.9 vs 1.0 B/cy)
   - STRM hint alone limits A's cache residency
   - Without HW prefetch to bring A in, A stalls badly
   - **STRM without a prefetch pipeline is dangerous**

4. **`gemm_optimal` (sector + two-level prefetch) is most stable** (~2.4 B/cy)
   - L2 KEEP prefetch brings A data far ahead
   - L1 STRM near-prefetch isolates A from B/C sector
   - Most resilient to HW prefetch on/off

### Sector Isolation Test (STRM vs KEEP for A, B=32KB resident)

Direct comparison: does PLDL1STRM actually protect B's L1 residency?

| A Size | A=STRM (sector) | A=KEEP (no sector) | Difference |
|--------|------------------|--------------------|------------|
| 128K | 33.3 B/cy | 32.9 B/cy | +1.2% |
| 1M | 30.5 B/cy | 30.5 B/cy | 0% |
| 8M | 10.8 B/cy | 11.1 B/cy | -2.7% |
| 64M | 11.0 B/cy | 11.0 B/cy | 0% |

**Finding: On Fugaku, STRM vs KEEP shows no measurable sector partitioning
effect.** The L1D sector cache feature (IMP_SCTLR_EL1.L1SECTORE) may not
be enabled in Fugaku's kernel configuration, or the A64FX replacement
policy already handles the A/B working set without explicit partitioning.

### Recommendations for A64FX GEMM Prefetch

| Priority | Strategy | Effect |
|----------|----------|--------|
| 1 | **Use HW prefetch** (default on) | Best raw bandwidth for sequential A |
| 2 | Add `pldl2keep` prefetch for A, 16 lines ahead | Resilience when HW fails |
| 3 | Pre-load B/C tiles with `pldl1keep` before K-loop | Ensures L1 residency |
| 4 | Use `pldl1strm` for A if sector is kernel-enabled | Protects B/C from eviction |

**Bottom line**: On current Fugaku, HW prefetch + explicit B/C pre-loading
is the winning strategy. STRM-based sector partitioning has no observable
effect on this system, but the code infrastructure is ready if sector cache
is enabled in future kernel configurations.

## FCC OCL Sector Cache Pragmas

### Overview

FCC traditional mode (`-Nnoclang`) supports OCL (Optimization Control Line) pragmas
to control A64FX sector cache at the source level. The compiler emits:
1. Tagged addresses with top-byte bit 56 set for sector 1 data
2. Runtime calls to `__jwe_xset_sccr` to program the sector cache control register

### Build Requirements

```bash
# Must use traditional mode (NOT -Nclang) for OCL pragma support
fcc -Nnoclang -O2 -Kocl,hpctag -o test test.c

# Generate assembly to inspect sector cache instructions
fcc -Nnoclang -O2 -Kocl,hpctag -S -o test.s test.c
```

### Pragma Syntax (C)

```c
// Partition L1D 4 ways: sector0 gets S0 ways, sector1 gets S1 ways (S0+S1=4)
#pragma statement cache_sector_size(S0, S1)

// Assign array pointer to sector 1 (tagged with bit 56)
#pragma statement cache_subsector_assign(ptr_name)

    // ... loop body: loads from ptr_name use tagged address ...

#pragma statement end_cache_subsector
#pragma statement end_cache_sector_size
```

### Example: GEMM with Sector Cache

```c
void gemm_sector(double *A, double *B, double *C, int M, int N, int K) {
    int i, j, k;

    // A streams through K → sector 1 (1 way = 16KB)
    // B/C reused          → sector 0 (3 ways = 48KB)
#pragma statement cache_sector_size(3, 1)
#pragma statement cache_subsector_assign(A)
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            double sum = 0.0;
            for (k = 0; k < K; k++) {
                sum += A[i*K + k] * B[k*N + j];
            }
            C[i*N + j] += sum;
        }
    }
#pragma statement end_cache_subsector
#pragma statement end_cache_sector_size
}
```

### Generated Assembly Analysis

The compiler emits three key components:

**1. SCCR Register Programming**
```asm
// Begin sector cache: command=256 (0x100)
mov   w0, 256
str   w0, [x22]           // command = 0x100 (start)
str   wzr, [x22, 4]       // reserved = 0
mov   w0, 65536            // sector1_ways << 16
movk  w0, 3, lsl #0       // | sector0_ways = 0x00010003
str   w0, [x22, 8]        // config = sector0_ways | (sector1_ways << 16)
bl    __jwe_xset_sccr      // program SCCR

// End sector cache: command=512 (0x200)
mov   w0, 512
str   w0, [x22]
bl    __jwe_xset_sccr      // restore default
```

**2. Tagged Address for Sector 1**
```asm
// A loads use tagged address: bit 56 = sector 1
orr   x_tagged, x_A, #0x0100000000000000   // = 72057594037927936
ldr   d0, [x_tagged, x_offset, lsl #3]     // load from sector 1
```

**3. HPC Tag Check at Init**
```asm
// Compiler inserts .init_array function to verify hpctag
__fjc_check_hpctag:
    bl    __jwe_check_hpctag    // verify HPC tag override is enabled
```

### SCCR Config Encoding

| Config | SCCR Value | Sector 0 | Sector 1 |
|--------|------------|----------|----------|
| `(1,3)` | `0x00030001` | 1 way (16KB) | 3 ways (48KB) |
| `(2,2)` | `0x00020002` | 2 ways (32KB) | 2 ways (32KB) |
| `(3,1)` | `0x00010003` | 3 ways (48KB) | 1 way (16KB) |

SCCR = `sector0_ways | (sector1_ways << 16)`, always totaling 4 ways.

### Sector Cache Test Results (Fugaku)

```
Test 1: L1 Streaming A (128KB) + Resident B (32KB)  [cache_sector_size(3,1)]
  No sector:   ~6.06M cycles
  With sector: ~6.07M cycles
  Speedup:     1.00x

Test 2: GEMM 64x64x64  [cache_sector_size(3,1) + cache_subsector_assign(A)]
  No sector:   ~2.41M cycles (434 MFLOP/s)
  With sector: ~2.42M cycles (434 MFLOP/s)
  Speedup:     1.00x

Test 3: Way Partition Configurations (8KB data)
  Config (1,3): ~2.99M cycles
  Config (2,2): ~2.79M cycles
  Config (3,1): ~3.16M cycles

Test 4: L2 Sector Cache  [scache_isolate_way L2=5 L1=1]
  A=4MB stream, B=512KB resident
  No sector:   ~87.5M cycles
  With sector: ~94.4M cycles
  Speedup:     0.93x  (sector overhead, no actual partitioning)
```

**Finding**: The sector cache SCCR programming and tagged addresses are
correctly emitted by the compiler, but **no measurable performance benefit**
is observed. L2 sector shows 7% overhead from the SCCR runtime calls without
actual partitioning. This confirms that `IMP_SCTLR_EL1.L1SECTORE` is not
enabled in Fugaku's kernel configuration.

### L1+L2 Sector with scache_isolate_way

For L2-level sector partitioning (in addition to L1), use `#pragma procedure`:

```c
void my_function(double *streaming, double *reuse, int n) {
#pragma procedure scache_isolate_way L2=5 L1=1
#pragma procedure scache_isolate_assign streaming
    int i;
    for (i = 0; i < n; i++) {
        reuse[i % M] += streaming[i];
    }
}
```

This emits a 3-word SCCR config struct:
- `[+4]` = L1 config: `sector0_ways | (sector1_ways << 16)` (S0+S1=4)
- `[+8]` = L2 config: `sector0_ways | (sector1_ways << 16)`

Note: `#pragma procedure` must be placed at the start of the function body
(before variable declarations), unlike `#pragma statement` which can appear
before any loop.

### Files

```
test_sector_cache.c     # Standalone sector cache test program
build_sector_test.sh    # Build script (generates ASM + executable)
run_sector_test.sh      # Job submission script
test_sector_cache.s     # Generated assembly (inspect sector instructions)
```

## Output Format

CSV columns:
- `pattern`: Access pattern name
- `size_kb`: Working set size in KB
- `total_bytes`: Total bytes accessed
- `cycles`: CPU cycles elapsed
- `cycles_per_access`: Cycles per memory access
- `bytes_per_cycle`: Effective bandwidth
- `efficiency_l1`: Percentage of L1 peak (128 B/cy)
- `efficiency_l2`: Percentage of L2 peak (42.6 B/cy)

## File Structure

```
a64fx/hw-prefetch/
├── README.md                 # This file
├── Makefile                  # Build system
├── hwpf_bench.h              # Common header
├── hwpf_bench.c              # Main driver
├── access_patterns.h         # Pattern declarations
├── access_patterns.c         # Pattern implementations
├── test_sector_cache.c       # Standalone sector cache test (FCC OCL pragmas)
├── build_sector_test.sh      # Build script for sector cache test
├── run_sector_test.sh        # Job script for sector cache test
├── run_all_tests.sh          # Full test suite
├── run_prefetch_sweep.sh     # Distance sweep
└── results/                  # Output directory
```

## Optimization Guidelines

### When to Use SW Prefetch

1. **Large streaming workloads (>8MB)**: SW L2 prefetch (`pldl2keep`) provides 12% better bandwidth than HW-only
2. **When HW prefetch is unreliable**: SW prefetch maintains performance regardless of HW settings
3. **GEMM/FFT kernels**: Explicit prefetch for matrix tiles improves memory-bound performance

### Recommended SW Prefetch Strategy

```c
// Simple streaming: L2 prefetch 16 cache lines (4KB) ahead
for (i = 0; i < n; i++) {
    if (i + 16 < n) {
        __asm__ volatile("prfm pldl2keep, [%0]" : : "r"(ptr + (i+16)*256));
    }
    svfloat64_t v = svld1_f64(svptrue_b64(), ptr + i*256);
    // ... compute ...
}
```

### Recommended GEMM Prefetch Strategy

```c
// GEMM: sector-aware prefetch for A(stream), B(reuse), C(accum)
// Pre-load B and C tiles
for (bi = 0; bi < B_LINES; bi++)
    __asm__ volatile("prfm pldl1keep, [%0]" : : "r"(B + bi*256));
for (ci = 0; ci < C_LINES; ci++)
    __asm__ volatile("prfm pldl1keep, [%0]" : : "r"(C + ci*256));

for (m = 0; m < M; m++) {
    for (k = 0; k < K; k++) {
        // A: L2 far ahead + L1 STRM near (sector isolation)
        __asm__ volatile("prfm pldl2keep, [%0]"  : : "r"(A + (a_off+16)*256));
        __asm__ volatile("prfm pldl1strm, [%0]"  : : "r"(A + a_off*256));
        // B: re-touch with KEEP to reinforce residency
        __asm__ volatile("prfm pldl1keep, [%0]"  : : "r"(B + b_off*256));
        // ... SDOT/FMLA micro-kernel ...
    }
}
```

### PRFM Instruction Reference

| Hint | Description | Use Case |
|------|-------------|----------|
| `pldl1keep` | Prefetch to L1, temporal | Data reused soon |
| `pldl1strm` | Prefetch to L1, streaming | One-time access |
| `pldl2keep` | Prefetch to L2, temporal | **Best for streaming** |
| `pldl2strm` | Prefetch to L2, streaming | Large non-reused data |
| `pstl1keep` | Prefetch for store, L1 | Write-intensive |
| `pstl2keep` | Prefetch for store, L2 | Large writes |

### General Tips

1. **Working set size matters**: Prefetch helps most when data exceeds L2 (>8MB)
2. **Stride matters**: HW prefetcher works best for sequential/small-stride; use SW for large strides
3. **Distance tuning**: 16 cache lines (4KB) ahead works well for L2 prefetch
4. **Weak mode**: Use `--weakL2` for workloads with poor locality to reduce cache pollution
5. **Combine HW+SW**: SW prefetch complements HW prefetch, doesn't replace it
6. **STRM needs prefetch**: Never use `pldl1strm` without a prefetch pipeline - STRM alone
   limits cache residency without providing data, causing stalls
7. **Sector cache status**: A64FX supports sector cache but Fugaku may not enable
   `IMP_SCTLR_EL1.L1SECTORE`; STRM hint still useful as replacement policy hint
