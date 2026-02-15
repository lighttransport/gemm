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
- Registers: `IMP_SCCR_SET0_L2_EL1` (sys_reg(3, 0, 15, 8, 2))

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

**Compile-time:**
- **`-Khpctag`** compiler flag must be active (enabled by default with `-KA64FX`)
- **`-Kocl`** enables OCL pragmas (required for sector cache syntax)

**Runtime (CRITICAL):**
- **`export FLIB_HPCFUNC=TRUE`** — enables HPC function features (tag addressing)
  - The Fujitsu runtime library writes '1' to `/sys/kernel/xos_hpc/hwpf_uaccess`
  - Without this, `__jwe_check_hpctag` (CRT init) finds '0' → SCCR calls become no-ops
- **`export FLIB_SCCR_CNTL=TRUE`** — enables SCCR register management
- **`export FLIB_L1_SCCR_CNTL=FALSE`** — disable L1 SCCR fallback (optional)

**Kernel:**
- **`IMP_SCTLR_EL1.L1SECTORE`** must be enabled for L1 sector partitioning

### CRT Mechanism (`__jwe_check_hpctag`)

The FCC compiler inserts an `.init_array` constructor that runs before `main()`:

1. **CPU check**: Reads `midr_el1`, verifies implementer=0x46 (Fujitsu) and part=A64FX. **Aborts** if not running on A64FX.
2. **HPC tag check**: Opens `/sys/kernel/xos_hpc/hwpf_uaccess` and reads one byte with `fgetc()`:
   - If file contains `'1'` (0x31) → sets `hpctag_enabled = 1`
   - If file doesn't exist → sets `hpctag_enabled = 1` (assumes enabled)
   - If file contains anything else → `hpctag_enabled = 0` (disabled)
3. Later calls to `__jwe_xset_sccr()` check this flag. If disabled, they are no-ops.

The `FLIB_HPCFUNC=TRUE` environment variable causes the Fujitsu runtime library to
write `'1'` to the sysfs file before `__jwe_check_hpctag` runs, enabling the feature.

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

### Method 2: L1+L2 Partitioning (`scache_isolate_way`)

Three pragma levels are available:

**Region-based way partitioning** (`#pragma statement`):
```c
#pragma statement scache_isolate_way L2=N2 [L1=N1]
    // ... code with sector cache active ...
#pragma statement end_scache_isolate_way
```

**Function-level array assignment** (`#pragma procedure`):
```c
void func(double *stream, double *reuse) {
#pragma procedure scache_isolate_assign stream
    // ... entire function body: loads from stream are tagged ...
}
```

**Statement-level array assignment** (`#pragma statement`):
```c
#pragma statement scache_isolate_assign ptr
    load = ptr[i];  // this load tagged for sector 1
#pragma statement end_scache_isolate_assign
```

**Parameters:**
- `L2=N2`: Number of L2 ways for sector 1 (tagged/streaming). Range: 1-15 (out of 16).
- `L1=N1`: (Optional) Number of L1 ways for sector 1. Range: 1-3 (out of 4).
- If `L1` is omitted, only L2 partitioning is configured.
- Multiple arrays can be assigned: `scache_isolate_assign A, B`

**Example: SpMV with streaming values/indices, reusing vector**
(Based on `a64fx/ref/ellspmv/ellspmv.c`)

```c
// Global region: set up way partitioning around compute loop
#pragma statement scache_isolate_way L2=4
    for (iter = 0; iter < niter; iter++) {
        spmv(val, col, rowptr, x, y, nrows);
    }
#pragma statement end_scache_isolate_way

// Kernel function: assign streaming arrays to sector 1
void spmv(double *val, int *col, int *rowptr, double *x, double *y, int nrows) {
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

**Build flags** (from reference implementation):
```bash
CC=fcc CFLAGS="-Kfast -Kocl -DUSE_A64FX_SECTOR_CACHE -DA64FX_SECTOR_CACHE_L2_WAYS=4" make
```

**Runtime environment** (from reference `run_a64fx.sh`):
```bash
export FLIB_HPCFUNC=TRUE
export FLIB_SCCR_CNTL=TRUE
export FLIB_L1_SCCR_CNTL=FALSE
export XOS_MMM_L_HPAGE_TYPE=hugetlbfs
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
# Compile
fcc -Nnoclang -O2 -Kocl,hpctag -o test_sector_cache test_sector_cache.c

# Run (CRITICAL: set env vars first!)
export FLIB_HPCFUNC=TRUE
export FLIB_SCCR_CNTL=TRUE
export FLIB_L1_SCCR_CNTL=FALSE
./test_sector_cache
```

Or submit via `pjsub run_sector_test.sh`.

### Results (without FLIB_HPCFUNC)

Previous results were measured **without** `FLIB_HPCFUNC=TRUE`, so SCCR was not
actually programmed. The `__jwe_xset_sccr` calls were no-ops:

| Test | Description | No Sector | With Sector | Speedup |
|------|-------------|-----------|-------------|---------|
| 1 | L1: Stream A(128KB) + Resident B(32KB) | 6.06M cy | 6.07M cy | 1.00x |
| 2 | GEMM 64x64x64 (L1 sector) | 2.41M cy | 2.42M cy | 1.00x |
| 3a | L1 config (1,3) | 2.99M cy | - | - |
| 3b | L1 config (2,2) | 2.79M cy | - | - |
| 3c | L1 config (3,1) | 3.16M cy | - | - |
| 4 | L2: Stream A(4MB) + Resident B(512KB) | 87.5M cy | 94.4M cy | 0.93x |

### Results (with FLIB_HPCFUNC=TRUE)

Re-run with `FLIB_HPCFUNC=TRUE`, `FLIB_SCCR_CNTL=TRUE`, `FLIB_L1_SCCR_CNTL=TRUE`:

| Test | Description | No Sector | With Sector | Speedup |
|------|-------------|-----------|-------------|---------|
| 1 | L1: Stream A(128KB) + Resident B(32KB) | 6.07M cy | 6.07M cy | 1.00x |
| 2 | GEMM 64x64x64 (L1 sector) | 2.41M cy | 2.42M cy | 1.00x |
| 3a | L1 config (1,3) | 3.07M cy | - | - |
| 3b | L1 config (2,2) | 2.84M cy | - | - |
| 3c | L1 config (3,1) | 3.10M cy | - | - |
| 4 | L2: Stream A(4MB) + Resident B(512KB) | 87.5M cy | 87.5M cy | 1.00x |

Still no sector partitioning effect (unchanged from previous run).

### Analysis

1. **`/sys/kernel/xos_hpc/hwpf_uaccess` does not exist** on the tested node.
   - `__jwe_check_hpctag` CRT code treats missing file as "enabled" (hpctag=1)
   - But the absence of the XOS HPC sysfs module suggests the kernel lacks full
     HPC function support (IMP_SCTLR_EL1.L1SECTORE not enabled)

2. **SCCR library path confirmed**: `__jwe_xset_sccr` lives in `libfj90i.so.1`.
   - `__jwe_xsccr_init` reads `FLIB_SCCR_CNTL` via `getenv()` → enables/disables
   - When enabled, it reads/writes `IMP_SCCR_L1_EL0` (s3_3_c11_c8_2) directly
   - But SCCR writes are silently ignored if kernel hasn't enabled sector cache

3. **CRT mechanism fully decoded** (see Prerequisites section above):
   - `__jwe_check_hpctag` → CPU check + sysfs check
   - `__jwe_xsccr_init` → env var check (FLIB_SCCR_CNTL, FLIB_L1_SCCR_CNTL, etc.)
   - `__jwe_xset_sccr` → actual SCCR register programming

4. **Test 3 variation (~8%) is measurement noise**, not real partitioning. The
   (2,2) config appearing fastest is inconsistent with any sector model.

5. **To test on Fugaku proper**: Submit via `pjsub` to the `small` resource group
   which may have XOS HPC fully enabled (check for `/sys/kernel/xos_hpc/`).

6. **Correctness verified**: GEMM with sector cache produces identical results
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
  https://dl.acm.org/doi/fullHtml/10.1145/3624062.3624198
- Linux kernel RFC: A64FX cache driver (LKML, 2021)
- Fujitsu Software Compiler Package Manual J2UL-2583
- Reference implementation: `a64fx/ref/ellspmv/` (ELLPACK SpMV with sector cache)
