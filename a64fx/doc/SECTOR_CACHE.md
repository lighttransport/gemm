# A64FX Sector Cache - FCC OCL Pragma Reference

## Overview

The A64FX processor supports **sector cache**, a mechanism to partition L1D and L2 cache
ways into independent sectors. Each data structure in a program can be assigned to a sector,
preventing streaming data from evicting resident data.

The Fujitsu C compiler (FCC) provides **OCL (Optimization Control Line) pragmas** to control
sector cache from C source code. These pragmas are **only available in FCC traditional mode**
(`-Nnoclang`), not in the clang-compatible mode.

## Hardware Architecture

### L1 Data Cache
- 64 KB, 4-way set-associative, 256-byte cache line, 64 sets
- 4 flat sectors (hardware), 2 sectors usable via FCC pragmas
- Per-core: each PE has independent L1 sector capacity registers
- Register: `IMP_SCCR_L1_EL0` (sys_reg(3, 3, 11, 8, 2))

### L2 Cache
- 8 MB, 16-way set-associative, 256-byte cache line
- Shared per CMG (Core Memory Group, 12-13 cores)
- 2 groups of 2 sectors: {sector0, sector1} and {sector2, sector3}
- Sector IDs 0,1 and 2,3 cannot be active simultaneously in the same CMG
- Register changes on one PE affect all PEs in the same CMG
- Not programmble from app: Registers: `IMP_SCCR_SET0_L2_EL1` (sys_reg(3, 0, 15, 8, 2))
- IMP_SCCR_VSCCR_L2_EL0

### Sector Mechanism

```
                    L1D: 64KB, 4-way
               ┌────────────────────────────┐
               │  Way 0  │  Way 1  │  Way 2  │  Way 3  │
               ├─────────┴─────────┼─────────┴─────────┤
               │   Sector 0        │    Sector 1        │
               │   (default data)  │    (tagged data)   │
               │   B, C in GEMM    │    A in GEMM       │
               └───────────────────┴────────────────────┘

               L2: 8MB, 16-way (shared per CMG)
               ┌─────────────────────────────────────────┐
               │ Way 0..10 (Sector 0)│ Way 11..15 (Sect1)│
               │ reuse data          │ streaming data     │
               └─────────────────────┴────────────────────┘
```

Sector assignment is controlled by a **tagged address**: the compiler sets bit 56
of the virtual address (top byte) to indicate sector 1. The A64FX HPC tag address
override function interprets this bit to route the access to the correct sector.

### Prerequisites

- **`-Khpctag`** compiler flag must be active (enabled by default with `-KA64FX`)
- **`IMP_SCTLR_EL1.L1SECTORE`** must be enabled in the kernel for actual partitioning
- **Fugaku status**: Sector cache is NOT enabled in the Fugaku kernel. The compiler
  correctly emits all instructions, but the hardware ignores the sector hints.

## Compiler Flags

```bash
fcc -Nnoclang -O2 -Kocl,hpctag -o output source.c
```

| Flag | Purpose | Default |
|------|---------|---------|
| `-Nnoclang` | Use FCC traditional mode (required for OCL pragmas) | off |
| `-Kocl` | Enable OCL (Optimization Control Lines) | on |
| `-Khpctag` | Enable HPC tag address override (sector cache + HW prefetch assist) | on with `-KA64FX` |

To generate assembly for inspection:
```bash
fcc -Nnoclang -O2 -Kocl,hpctag -S -o output.s source.c
```

## Pragma Syntax

### Method 1: L1-Only Partitioning (`#pragma statement`)

Scoped to a block of code (typically a loop nest). Can appear anywhere in a function.

```c
#pragma statement cache_sector_size(sector0_ways, sector1_ways)
#pragma statement cache_subsector_assign(array_ptr)
    // ... code: loads from array_ptr are tagged for sector 1 ...
#pragma statement end_cache_subsector
#pragma statement end_cache_sector_size
```

**Parameters:**
- `sector0_ways`: Number of L1D ways for sector 0 (default/untagged data). Range: 1-3.
- `sector1_ways`: Number of L1D ways for sector 1 (tagged data). Range: 1-3.
- Constraint: `sector0_ways + sector1_ways = 4`
- `array_ptr`: Name of a pointer parameter or variable whose loads should be tagged.

**Example: GEMM with A streaming, B/C reusing**

```c
void gemm_sector(double *A, double *B, double *C, int M, int N, int K) {
    int i, j, k;

    // A streams through K dimension → sector 1 (1 way = 16KB)
    // B reused across M rows, C accumulated → sector 0 (3 ways = 48KB)
#pragma statement cache_sector_size(3, 1)
#pragma statement cache_subsector_assign(A)
    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            double sum = 0.0;
            for (k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] += sum;
        }
    }
#pragma statement end_cache_subsector
#pragma statement end_cache_sector_size
}
```

### Method 2: L1+L2 Partitioning (`#pragma procedure`)

Scoped to an entire function. Must be placed at the start of the function body,
before variable declarations.

```c
void my_function(double *stream_data, double *reuse_data, int n) {
#pragma procedure scache_isolate_way L2=N2 [L1=N1]
#pragma procedure scache_isolate_assign stream_data
    int i;
    // ... entire function body: loads from stream_data are tagged ...
}
```

**Parameters:**
- `L2=N2`: Number of L2 ways for sector 1 (tagged/streaming). Range: 1-15 (out of 16).
- `L1=N1`: (Optional) Number of L1 ways for sector 1. Range: 1-3 (out of 4).
- If `L1` is omitted, only L2 partitioning is configured.
- Multiple arrays can be assigned: `scache_isolate_assign A, B`

**Example: SpMV with streaming values/indices, reusing vector**

```c
void spmv(double *val, int *col, int *rowptr, double *x, double *y, int nrows) {
#pragma procedure scache_isolate_way L2=5 L1=1
#pragma procedure scache_isolate_assign val, col
    int i, j;
    for (i = 0; i < nrows; i++) {
        double sum = 0.0;
        for (j = rowptr[i]; j < rowptr[i + 1]; j++) {
            sum += val[j] * x[col[j]];
        }
        y[i] = sum;
    }
}
```

## Generated Assembly

The compiler emits three components for sector cache control.

### 1. SCCR Register Programming

At the start of a sector-cached region, the compiler calls `__jwe_xset_sccr()` with
a configuration struct:

```
struct sccr_config {
    uint32_t command;    // [+0] 256 = start sector, 512 = end sector
    uint32_t l1_config;  // [+4] L1: sector0_ways | (sector1_ways << 16)
    uint32_t l2_config;  // [+8] L2: sector0_ways | (sector1_ways << 16)
};
```

**Method 1** (L1-only, `cache_sector_size`):
```asm
// Start sector cache: cache_sector_size(3,1)
mov   w0, 256               // command = 0x100 (start)
str   w0, [sp]              // config.command = start
str   wzr, [sp, 4]          // config.l1_config = 0 (unused slot)
mov   w0, 65536             // 0x00010000 = 1 << 16
movk  w0, 3, lsl #0         // | 3 = 0x00010003
str   w0, [sp, 8]           // config.l2_config = SCCR value
                             // (confusingly stored at [+8] even for L1-only)
bl    __jwe_xset_sccr       // program SCCR

// End sector cache
mov   w0, 512               // command = 0x200 (end)
str   w0, [sp]
bl    __jwe_xset_sccr       // restore default
```

**Method 2** (L1+L2, `scache_isolate_way L2=5 L1=1`):
```asm
// Start sector cache with L1 and L2 config
mov   w0, 256               // command = start
str   w0, [sp]              // config.command
mov   w0, 65536             // 0x00010000
movk  w0, 3, lsl #0         // L1 config = 0x00010003 (sector0=3, sector1=1)
str   w0, [sp, 4]           // config.l1_config
mov   w1, 327680            // 0x00050000
movk  w1, 9, lsl #0         // L2 config = 0x00050009 (sector0=9, sector1=5)
str   w1, [sp, 8]           // config.l2_config
bl    __jwe_xset_sccr       // program SCCR
```

### 2. Tagged Address (bit 56)

Loads from the assigned array use a tagged pointer with bit 56 set:

```asm
// Original pointer: x_A = address of A
orr   x_tagged, x_A, #0x0100000000000000   // set bit 56 = sector 1 tag
ldr   d0, [x_tagged, x_offset, lsl #3]     // load from tagged address

// 0x0100000000000000 = 72057594037927936 in decimal
```

The A64FX HPC tag address override interprets bit 56 as the sector ID.
The actual load address is unchanged (top byte is ignored for virtual address
translation), but the cache controller uses the tag to select the sector.

### 3. Init-time HPC Tag Check

The compiler inserts a constructor function in `.init_array` to verify that
the HPC tag address override is enabled:

```asm
.section .init_array,"aw"
    .xword  __fjc_check_hpctag

__fjc_check_hpctag:
    bl    __jwe_check_hpctag    // runtime check: is hpctag enabled?
```

## SCCR Config Encoding

### L1 Configurations (4-way)

| Pragma | SCCR Value | Sector 0 (default) | Sector 1 (tagged) |
|--------|------------|--------------------|--------------------|
| `cache_sector_size(1, 3)` | `0x00030001` | 1 way = 16 KB | 3 ways = 48 KB |
| `cache_sector_size(2, 2)` | `0x00020002` | 2 ways = 32 KB | 2 ways = 32 KB |
| `cache_sector_size(3, 1)` | `0x00010003` | 3 ways = 48 KB | 1 way = 16 KB |

Encoding: `SCCR = sector0_ways | (sector1_ways << 16)`

### L2 Configurations (16-way)

| Pragma | L1 Config | L2 Config |
|--------|-----------|-----------|
| `scache_isolate_way L2=5` | (none) | `0x00050009` |
| `scache_isolate_way L2=5 L1=1` | `0x00010003` | `0x00050009` |

## GEMM Way Budget Design

For `C[M,N] += A[M,K] * B[K,N]` with L1 sector cache:

```
Matrix  Role                   Sector   Ways   Budget   Pragma
──────  ─────────────────────  ───────  ─────  ───────  ─────────────────
A       Streams along K        1 (tag)  1      16 KB    cache_subsector_assign(A)
B       Reused across M rows   0 (def)  2      32 KB    (default)
C       Accumulator tile       0 (def)  1      16 KB    (default)
Total                                   4      64 KB    cache_sector_size(3, 1)
```

## Benchmark Results (Fugaku)

### Test Program

See `a64fx/hw-prefetch/test_sector_cache.c` for a complete standalone test.

Build and run:
```bash
fcc -Nnoclang -O2 -Kocl,hpctag -o test_sector_cache test_sector_cache.c
./test_sector_cache
```

### Results

| Test | Description | No Sector | With Sector | Speedup |
|------|-------------|-----------|-------------|---------|
| 1 | L1: Stream A(128KB) + Resident B(32KB) | 6.06M cy | 6.07M cy | 1.00x |
| 2 | GEMM 64x64x64 (L1 sector) | 2.41M cy | 2.42M cy | 1.00x |
| 3a | L1 config (1,3) | 2.99M cy | - | - |
| 3b | L1 config (2,2) | 2.79M cy | - | - |
| 3c | L1 config (3,1) | 3.16M cy | - | - |
| 4 | L2: Stream A(4MB) + Resident B(512KB) | 87.5M cy | 94.4M cy | 0.93x |

### Analysis

1. **No measurable sector partitioning effect on Fugaku.**
   The `IMP_SCTLR_EL1.L1SECTORE` register is not enabled in the Fugaku kernel.
   The compiler correctly emits all sector cache instructions (`__jwe_xset_sccr`,
   tagged addresses), but the hardware does not enforce way partitioning.

2. **L2 sector shows 7% overhead** from the `__jwe_xset_sccr()` runtime calls
   without actual partitioning benefit.

3. **Test 3 variation (~5-10%) is measurement noise**, not real sector partitioning.
   The (2,2) config appearing fastest is inconsistent with any sector model.

4. **Correctness verified**: GEMM with sector cache produces identical results
   to the non-sector version (max difference = 0).

## Comparison: Sector Cache vs PRFM Hints

Since Fugaku's sector cache is not kernel-enabled, the alternative approach is
using PRFM hint encoding (KEEP vs STRM) as a software replacement policy hint:

| Approach | Mechanism | Fugaku Effect |
|----------|-----------|---------------|
| `cache_sector_size` + `cache_subsector_assign` | SCCR register + tagged address | None (not enabled) |
| `prfm pldl1strm` | Replacement policy hint | Minimal (no way partitioning) |
| `prfm pldl1keep` + pre-loading | Brings data into L1 early | Effective |
| HW prefetch (default) | Automatic sequential/stride prefetch | Best for streaming |

**Recommendation**: On current Fugaku, use HW prefetch + explicit `pldl1keep`
pre-loading for B/C tiles rather than sector cache pragmas.

## References

- A64FX Specification HPC Extension v1 (Fujitsu, 2020)
- A64FX Microarchitecture Manual v1.3
- Alappat et al., "ECM modeling and performance tuning of SpMV and Lattice QCD on A64FX" (Wiley, 2022)
- Alappat et al., "Modelling Data Locality of SpMV on the A64FX" (SC'23 Workshop)
  https://epub.ub.uni-muenchen.de/126883/1/3624062.3624198.pdf
- Linux kernel RFC: A64FX cache driver (LKML, 2021)
- Fujitsu Software Compiler Package Manual J2UL-2583
