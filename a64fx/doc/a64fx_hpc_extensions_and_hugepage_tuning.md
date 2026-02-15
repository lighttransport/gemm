# A64FX HPC Extensions & Large Page Tuning Guide

Consolidated from Fujitsu official documentation:
- *A64FX Specification HPC Extension v1* (EN, Nov 2020)
- *Job Operation Software End User Guide HPC Extension Edition* (j2ul-2535-01z0, JA, Sep 2022)
- *Processors Programming Guide* (EN/JA, 2021)
- *Tuning Programming Guide* (JA, v2.2, Mar 2023)
- *C Language User's Guide* (j2ul-2560-01z0, JA)

---

## Part I: HPC Extensions Overview

The A64FX processor implements five proprietary Fujitsu HPC extensions, succeeding features from SPARC64 VIIIfx/IXfx/XIfx:

1. **HPC Tag Address Override** — Control sector cache and hardware prefetch from applications
2. **Sector Cache** — Cache partitioning by data access characteristics (temporal locality)
3. **Hardware Prefetch Assistance** — Software hints for complex prefetch patterns
4. **Hardware Barrier** — Inter-thread synchronization support
5. **Large Page Library (libmpg)** — Efficient HugeTLBfs utilization

---

## Part II: HPC Tag Address Override

### 2.1 Mechanism

The HPC tag address override repurposes the upper 8 bits of 64-bit virtual addresses (normally unused in ARMv8-A Tagged Addressing) as hardware performance hints. When enabled, these bits are not used for addressing but instead control:

- **Sector cache** behavior (which sector to place data in)
- **Hardware prefetch** behavior (enable/disable, stream detect hints)

This is compatible with ARMv8-A Tagged Addressing at the ISA level — both features use the same top-8-bit mechanism, but they are **mutually exclusive** in practice.

### 2.2 Enable/Disable Settings

| TBI (ARMv8-A) | TBO (Fujitsu) | Behavior |
|:-:|:-:|---|
| 0 | — | Both disabled. Full 64-bit address used. |
| 1 | 0 | ARMv8-A Tagged Addressing enabled. Top 8 bits ignored for addressing. |
| 1 | 1 | **HPC tag address override enabled.** Top 8 bits = performance hints. |

Additional control bits within TBO:
- **SCE** (Sector Cache Enable) — per-region enable/disable
- **PFE** (hardware PreFetch Enable) — per-region enable/disable

When HPC tag override is disabled (TBI=0 or TBO=0), sector cache uses the Default Sector from register settings, and hardware prefetch operates in Stream Detect mode (configured via `IMP_PF_STREAM_DETECT_CTRL_EL0`).

### 2.3 System Registers

| Register | EL | Shared Domain |
|----------|:--:|:---:|
| `IMP_FJ_TAG_ADDRESS_CTRL_EL1` | 1–3 | PE |
| `IMP_FJ_TAG_ADDRESS_CTRL_EL2` | 2–3 | PE |
| `IMP_FJ_TAG_ADDRESS_CTRL_EL3` | 3 | PE |
| `IMP_FJ_TAG_ADDRESS_CTRL_EL12` | 2–3 (alias) | PE |

ARMv8.1 virtualization host extension (HCR_EL2.E2H=1, SCR_EL3.NS=1) adds PFE1/SCE1/TBO1 bits to EL2 and redirects EL1 accesses to EL2.

### 2.4 Tag Address Bit Allocation

The upper 8 bits of the virtual address (bits [63:56]) are divided among HPC features when the override is active. Specific bit assignments control sector ID selection and prefetch hints per load/store/prefetch instruction.

### 2.5 fhetbo Command

The `fhetbo` command controls HPC tag address override at the job level:

```bash
/opt/FJSVxos/fhehpc/bin/fhetbo {enable|disable}
```

- **Scope:** All cores assigned to the current job on the executing node
- **Default:** Enabled (HPC override active, ARMv8-A tagging disabled)
- **Persistence:** Reverts to enabled at job termination
- **Multi-node:** Use via `mpiexec` for multi-node jobs

**Job script example:**
```bash
#!/bin/bash
#PJM -L "node=256"
#PJM -L "elapse=86400"
export NR_PROCS=256
mpiexec -n $NR_PROCS /opt/FJSVxos/fhehpc/bin/fhetbo disable  # Disable HPC override
mpiexec -n $NR_PROCS ./a.out                                   # Run application
mpiexec -n $NR_PROCS /opt/FJSVxos/fhehpc/bin/fhetbo enable   # Re-enable
```

**Messages:**
- `[INFO] xos FHE 1113` — Tag override enabled with core mask
- `[INFO] xos FHE 1114` — Tag override disabled with core mask
- Core mask example: `0xffffffffffff000` = cores 12–59 (job cores)

---

## Part III: Sector Cache

### 3.1 Overview

Sector cache is a **cache partitioning** function that divides L1D and L2 caches into sectors. Each sector can be assigned a different capacity weight, allowing data with different temporal locality characteristics to be placed in appropriately sized cache partitions.

This prevents streaming data from evicting frequently reused data, and vice versa.

### 3.2 Control via HPC Tag Override

When HPC tag address override is enabled, each memory access instruction can specify which sector to place its data in through the tag bits. This is controlled at the instruction level, giving fine-grained control over cache utilization.

### 3.3 Software Control Methods

**Fujitsu compiler OCL (Optimization Control Lines):**
```fortran
!OCL CACHE_SECTOR_SIZE(sector_sizes)
!OCL CACHE_SUBSECTOR_ASSIGN(assignments)
```

**Environment variables:**
```bash
export FLIB_SCCR_CNTL=TRUE        # Enable sector cache
export FLIB_SCCR_ASSIGN="..."     # Sector assignment
```

**Compiler options** can also control sector cache behavior for specific loops and data arrays.

---

## Part IV: Large Page (HugeTLB) Feature Summary

### 4.1 Background — Memory Address Translation and TLB

When applications access memory, virtual addresses must be translated to physical addresses via the page table. The CPU caches recent translations in the **TLB (Translation Look-aside Buffer)**:

1. Load/store unit receives a virtual address
2. TLB lookup → **hit**: physical address returned immediately
3. TLB lookup → **miss**: expensive page table walk in main memory

For HPC applications with large data, TLB misses become a major bottleneck. Large pages reduce this by covering more memory per TLB entry.

### 4.2 Page Sizes

| Page Type | Size | Use Case |
|-----------|------|----------|
| Normal page | 64 KiB | Default OS page size |
| Large page | 2 MiB | Default for HPC (via libmpg) |
| McKernel large page | 32 MiB | McKernel mode only |
| McKernel extended | 1 GiB, 16 GiB | McKernel-specific |

**Trade-offs:**

| Metric | 64 KiB (Normal) | 2 MiB (Large) |
|--------|:---------------:|:--------------:|
| TLB miss rate | High | **Low** |
| Memory init cost | **Small** | Large |
| Memory usage efficiency | **High** | Low |

### 4.3 Memory Regions and Large Page Coverage

`libmpg.so` selectively applies 2 MiB pages:

| Memory Region | Large Page | Page Size | Notes |
|---------------|:----------:|-----------|-------|
| .text (code) | ✗ | 64 KiB | Always normal pages |
| .data (static, initialized) | ✓ | 2 MiB | Always prepaged |
| .bss (static, uninitialized) | ✓ | 2 MiB | Paging policy configurable |
| Heap (brk/sbrk) | ✗ | 64 KiB | Kernel limitation |
| mmap region (malloc) | ✓ | 2 MiB | Primary path with libmpg |
| Thread heap | ✓ | 2 MiB | Sub-thread heaps |
| Process/main thread stack | ✓ | 2 MiB | Requires `LPG_MODE=base+stack` |
| Thread stack | ✓ | 2 MiB | Requires `LPG_MODE=base+stack` |
| Shared memory | ✗ | 64 KiB | Inter-process shared |

**Key:** When libmpg is linked, `malloc(3)` goes through `mmap(2)` (large-paged), not `brk/sbrk` (normal-paged). Do **not** mix `brk/sbrk` with libmpg.

### 4.4 Variable Placement by Language

**Fortran:**
```fortran
real*8 a(N)                ! → .bss  (or stack with -Kauto/-Kthreadsafe)
real*8 :: b(N) = 1.0       ! → .data
allocatable :: c(:)
allocate(c(N))             ! → dynamic memory (mmap)
```

**C/C++:**
```c
double a[N];               // Global uninitialized → .bss
double b[N] = {1.0};       // Global initialized → .data
double *c = malloc(...);   // → dynamic memory (mmap)
double e[N];               // Local in main → process stack
// In thread function:
double f[N];               // → thread stack
```

**C++ (std::vector, new):** All dynamic → mmap region.

---

## Part V: Compilation and Linking

### Fujitsu Compiler
```bash
fcc -Klargepage program.c        # Enable large pages (default)
fcc -Knolargepage program.c      # Disable large pages
```

### GCC / General Compilers
```bash
gcc -Wl,-T/opt/FJSVxos/mmm/util/bss-2mb.lds \
    -L/opt/FJSVxos/mmm/lib64 \
    -lmpg -lc -lpthread \
    -no-pie \
    test_program.c
```

**Critical rules:**
1. **Link order:** `-lmpg` must precede `-lc` and `-lpthread`.
2. **PIE incompatible:** `.data/.bss` not large-paged if PIE. Use `-no-pie`. Verify: `readelf -h a.out` → `e_type` must be `ET_EXEC`.
3. **Linker script required:** `bss-2mb.lds` aligns `.data/.bss` to 2 MiB boundaries.

---

## Part VI: Environment Variables — Complete Reference

### 6.1 Basic Settings

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `XOS_MMM_L_HPAGE_TYPE` | `hugetlbfs` \| `none` | `hugetlbfs` | Master enable/disable. `none` disables all XOS_MMM_L_ variables. In McKernel+`none`, uses extended THP mode. |
| `XOS_MMM_L_LPG_MODE` | `base+stack` \| `base` | `base+stack` | `base` = only .data/.bss and mmap. `base+stack` adds stack/thread-stack. |
| `XOS_MMM_L_HUGETLB_FALLBACK` | `0` \| `1` | `0` | `1` = fall back to normal pages on failure (vs OOM kill). Only for mmap-region malloc. Requires: HPAGE_TYPE=hugetlbfs, PAGING_POLICY=\*:\*:prepage, ARENA_LOCK_TYPE=1, MAX_ARENA_NUM=1. |
| `XOS_MMM_L_PRINT_ENV` | `on`/`1` \| `off`/`0` | `0` | Print all tuning variables to stderr before main(). |
| `XOS_MMM_L_HUGETLB_SZ` | `2M` \| `32M` | `2M` | **McKernel only.** 2 MiB or 32 MiB page size. |

### 6.2 Paging Policy

| Variable | Format | Default |
|----------|--------|---------|
| `XOS_MMM_L_PAGING_POLICY` | `<bss>:<stack>:<mmap>` | `prepage:demand:prepage` |

Each field is `demand` or `prepage`. The `.data` region is **always prepaged** regardless.

**Paging modes:**

| Mode | Behavior | Pros | Cons |
|------|----------|------|------|
| **Prepage** | Physical pages allocated upfront at region creation | Fewer subsequent page faults, stable performance | Data lands on CMG0, cross-NUMA traffic |
| **Demand** | Physical pages allocated on first access | NUMA-local allocation, pages on accessing CMG | Initial page faults, slight variability |

### 6.3 Tuning Settings (libmpg-specific)

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `XOS_MMM_L_ARENA_FREE` | `1` \| `2` | `1` | **1**: freed memory returned to OS immediately. **2**: all memory pooled/reused (never freed). `2` implies LOCK_TYPE=1, MAX_ARENA=1, MMAP_THRESHOLD=ULONG_MAX, TRIM_THRESHOLD=ULONG_MAX. |
| `XOS_MMM_L_ARENA_LOCK_TYPE` | `0` \| `1` | `1` | **0**: allocation performance priority — thread heaps created on contention (parallel malloc). **1**: memory efficiency priority — serialized through shared arena. |
| `XOS_MMM_L_MAX_ARENA_NUM` | 1–INT_MAX | `1` | Max arenas (process heap + thread heaps). Only with LOCK_TYPE=1. Default `1` = no thread heaps. ≥2 allows thread heap creation. |
| `XOS_MMM_L_HEAP_SIZE_MB` | ≥2×MMAP_THRESHOLD | 2×MMAP_THRESHOLD | Thread heap allocation/expansion size (MiB). Reduce to save memory. |
| `XOS_MMM_L_COLORING` | `0` \| `1` | `1` | Cache coloring for mmap allocations ≥ MMAP_THRESHOLD. Reduces L1 conflicts. Disable if app implements own coloring. |
| `XOS_MMM_L_FORCE_MMAP_THRESHOLD` | `0` \| `1` | `0` | **0**: search heap free-list first for large allocs. **1**: always use mmap (guarantees cache coloring). |

### 6.4 glibc Settings (used by libmpg)

| Variable | Default | Description |
|----------|---------|-------------|
| `MALLOC_MMAP_THRESHOLD_` | 134217728 (128 MiB) | Allocations ≥ threshold use mmap; smaller use heap. |
| `MALLOC_TRIM_THRESHOLD_` | 134217728 (128 MiB) | Contiguous free space ≥ threshold at heap top triggers OS release. |
| `MALLOC_TOP_PAD_` | 131072 (128 KiB) | Heap growth increment. Rounded up to page size. |
| `MALLOC_MMAP_MAX_` | 2097152 | Max outstanding mmap allocations. 0 = force heap only. |
| `MALLOC_CHECK_` | 3 | Debug: 0=ignore, 1=print, 2=abort, 3=print+trace+abort, 5=brief, 7=brief+trace+abort. |
| `MALLOC_PERTURB_` | 0 | Debug: fills alloc'd memory (complement of low byte) and freed memory (low byte). |

---

## Part VII: Tuning Scenarios and Benchmarks

### 7.1 TLB Miss Reduction — Stream Triad

**Config:** 1 thread, n=83,880,960

| Metric | Normal (64 KiB) | Large (2 MiB) | Improvement |
|--------|:---:|:---:|:---:|
| Execution time (s) | 0.435 | 0.347 | 1.25× |
| Throughput (GB/s) | 61.76 | 77.29 | **+25%** |
| L1D TLB miss rate | 0.098% | 0.003% | **30× lower** |
| L2D TLB miss rate | 0.098% | 0.0003% | **296× lower** |

### 7.2 PAGING_POLICY — Multi-CMG NUMA Locality

**Config:** Stream kernel, 48 threads across 4 CMGs, ~1 GB data.

| Paging Policy | Memory Throughput | Ratio |
|:---:|:---:|:---:|
| `prepage` (default for mmap) | 93 GB/s | 1× |
| `demand` | 804 GB/s | **8.6×** |

**Root cause:** Prepaging allocates all data on CMG0 at startup. Other CMGs suffer remote NUMA latency. Demand paging places pages on the CMG that first touches them.

**Compiler options used:** `-Kfast,openmp -Kprefetch_sequential=soft -Kprefetch_line=9 -Kprefetch_line_L2=70 -Kzfill=18`

### 7.3 ARENA_FREE — malloc/free Cycle Optimization

**Config:** 1024 × 8 MiB malloc, then free all, loop twice.

| Operation | FREE=1 (default) | FREE=2 (pool) | Speedup |
|-----------|:---:|:---:|:---:|
| 1st malloc | 0.501 s | 0.511 s | ~1× |
| 1st free | 0.260 s | 0.0005 s | **524×** |
| 2nd malloc | 0.501 s | 0.0003 s | **1,033×** |
| 2nd free | 0.260 s | 0.0003 s | **1,625×** |

**Trade-off:** Memory never returned to OS — peak consumption persists.

### 7.4 ARENA_LOCK_TYPE — Parallel malloc

**Config:** 16 threads × malloc(64 KiB) × 20,480 times, 10 iterations.

| Setting | Time | Speedup |
|:---:|:---:|:---:|
| LOCK_TYPE=1 (default, serialized) | 0.56 s | 1× |
| LOCK_TYPE=0 (parallel) | 0.35 s | **1.60×** |

### 7.5 MALLOC_MMAP_THRESHOLD_ — Cache Thrashing Fix

**Problem:** Multiple identically-sized dynamic arrays align to the same L1 cache sets (e.g., 256×256×8B = 512 KiB = 32×16 KiB boundary). All streams conflict → **L1D cache thrashing**.

**Fortran example:** 8 arrays of 256×256 doubles, 12 threads.

| Metric | Before | After (THRESHOLD=204800) | Change |
|--------|:---:|:---:|:---:|
| L1D miss rate | 0.26 | 0.13 | **2× lower** |
| L1D demand miss % | 51.52% | 9.47% | **5.4× lower** |

**Mechanism:** Changing the mmap threshold shifts allocation base addresses, breaking power-of-2 alignment and eliminating cache set conflicts — **implicit padding without code changes**.

---

## Part VIII: NUMA Architecture

### FX Server NUMA Layout

A64FX has 4 CMGs (Core Memory Groups). In FX server mode, the OS creates a split configuration:

| NUMA Nodes | Cores | Purpose |
|:---:|:---:|---|
| #0–3 | 0–11 | System (OS) — unavailable to jobs |
| #4–7 | 12–59 | Job (application) — 12 cores per CMG |

**User-accessible CPU numbers start at 12.** For manual affinity:
```c
cpuid[i] = i + 12;  // Logical core 0 = OS core 12
```

Use `sched_getaffinity(2)` for portable CPU discovery.

---

## Part IX: Startup Behavior and Caveats

### 9.1 Startup Memory Overhead

At startup, libmpg temporarily uses **2× combined .data + .bss size** for remapping to large pages. Applications with very large static data may be **SIGKILL'd (OOM)** before `main()`.

**Workaround:**
```bash
export XOS_MMM_L_PAGING_POLICY=demand:demand:prepage
# Or via job submission:
pjsub -x XOS_MMM_L_PAGING_POLICY=demand:demand:prepage jobscript.sh
```

Check static data size: `readelf -S a.out | grep -E '\.data|\.bss'`

### 9.2 brk/sbrk Incompatibility

brk(2)/sbrk(2) memory is never large-paged. Mixing brk/sbrk with libmpg malloc can cause ENOMEM due to normal/large page coexistence. Use `malloc(3)` exclusively with libmpg.

### 9.3 Signal Handler + Prepaging Conflict

When using `timer_create(2)` with frequent SIGALRM/SIGVTALRM, `fork(2)`/`clone(2)` may loop indefinitely due to ERESTARTNOINTR during prepaging. Fix:
```bash
export XOS_MMM_L_PAGING_POLICY=demand:demand:demand
```

### 9.4 Stack Alignment Caveats

Large-paged stack sizes are aligned up to HugeTLBfs page boundaries. If the aligned stack overlaps adjacent VMAs, large page mapping is skipped (warning emitted, execution continues with normal pages).

---

## Part X: Debugging and Diagnostics

### 10.1 Print All Settings
```bash
export XOS_MMM_L_PRINT_ENV=on
# → all XOS_MMM_L_ and MALLOC_ values printed to stderr before main()
```

### 10.2 /proc Inspection

```
# Normal heap:
00430000-00890000 rw-p 00000000 00:00 0  [heap]
# Large-paged:
aaaae2400000-aaaae2600000 rw-p 00000000 00:0e 452989  /anon_hugepage (deleted)

# Normal stack:
fffffffd0000-1000000000000 rw-p 00000000 00:00 0  [stack]
# Large-paged stack:
fffaeb800000-1000000000000 rw-p 00000000 00:0e 274404  /memfd: [stack] by libmpg (deleted)
```

### 10.3 Valgrind Compatibility

| Tool | With libmpg | Notes |
|------|:-----------:|-------|
| memcheck | ✓ | |
| cachegrind | ✓ | |
| callgrind | ✓ | |
| helgrind | **✗** | Conflicts with malloc hooks. May OOM due to unbounded RLIMIT_DATA/RLIMIT_STACK. Set `pjsub --proc-data=<limit> --proc-stack=<limit>`. |

### 10.4 malloc_stats(3)

glibc's `malloc_stats()` prints per-arena statistics. Requires adding the call to source code.

### 10.5 Job Statistics (HUGETLB_FALLBACK)

```ini
# papjmstats.conf:
Item {
    ItemName=hugetlb_fallback
    ItemNameDisp=HUGETLB_FALLBACK_LPG
    RecordNameList=JN
    DataType=PJMX_DATATYPE_UINT8
    DispFormat=dec
}
```
Output: `HUGETLB_FALLBACK_LPG : 0` (not triggered) or `1` (fallback occurred).

---

## Part XI: Warning Messages Reference

| Code | Meaning | Action |
|------|---------|--------|
| **FHE 1113** | Tag override enabled (core mask shown) | Informational |
| **FHE 1114** | Tag override disabled (core mask shown) | Informational |
| **LPG 2002** | PIE binary — .data/.bss not large-paged | Recompile with `-no-pie` |
| **LPG 2003** | Segment overlap — .data/.bss not large-paged | Use proper linker script |
| **LPG 2004** | Custom stack address in pthread_attr — thread stack not large-paged | Don't set stack address in attr |
| **LPG 2005** | Multi-thread detected during process stack remap | Ensure single-thread at init |
| **LPG 2006** | Process stack bottom overlaps next VMA | Avoid mmap near stack bottom |
| **LPG 2007** | Process stack top overlaps previous VMA | Avoid mmap near stack top |
| **LPG 9999** | Fallback to normal pages occurred | Informational (FALLBACK=1) |

---

## Part XII: Performance Tuning Decision Tree

### Step 1: Is TLB a bottleneck?
Check PA reports for high **mDTLB miss rate** (thrashing) or **uDTLB miss rate** (capacity). Ensure large pages are active (no PIE warnings, proper linking).

### Step 2: Multi-CMG performance far below expectations?
```bash
export XOS_MMM_L_PAGING_POLICY=demand:demand:demand
```

### Step 3: malloc/free overhead in parallel regions?
```bash
export XOS_MMM_L_ARENA_LOCK_TYPE=0
export XOS_MMM_L_ARENA_FREE=2
```

### Step 4: L1D cache thrashing with identically-sized arrays?
```bash
export MALLOC_MMAP_THRESHOLD_=204800
```

### Step 5: Startup OOM with large static data?
```bash
export XOS_MMM_L_PAGING_POLICY=demand:demand:prepage
```

### Step 6: Signal handler loops with prepaging?
```bash
export XOS_MMM_L_PAGING_POLICY=demand:demand:demand
```

---

## Part XIII: Recommended Configurations

### Default (single CMG, simple workload)
```bash
# All defaults — no changes needed
# Large pages enabled, prepage:demand:prepage
```

### Multi-CMG HPC (most common for Fugaku)
```bash
export XOS_MMM_L_PAGING_POLICY=demand:demand:prepage
```

### Memory-intensive with repeated alloc/free
```bash
export XOS_MMM_L_ARENA_FREE=2
export XOS_MMM_L_ARENA_LOCK_TYPE=0
```

### Maximum memory safety
```bash
export XOS_MMM_L_HUGETLB_FALLBACK=1
export XOS_MMM_L_PAGING_POLICY=demand:demand:prepage
```

### Debugging
```bash
export XOS_MMM_L_PRINT_ENV=on
export MALLOC_CHECK_=3
export MALLOC_PERTURB_=0xAA
```

### Sector cache tuning (Fujitsu compiler)
```bash
# In job script:
export FLIB_SCCR_CNTL=TRUE
# In source (OCL directives):
# !OCL CACHE_SECTOR_SIZE(...)
# Compiler: -Ksector_cache
```

---

## Part XIV: Embedding Lookup — Hugepage Tuning Results

### 14.1 Workload Characteristics

Embedding forward pass: `output[i,:] = emb_table[indices[i],:]`
- **Embedding table**: random-access read-only (vocab=151936 × dim × 4B FP32)
  - dim=1024: 0.58 GB, dim=2048: 1.16 GB, dim=4096: 2.37 GB
- **Output**: sequential write (each thread writes its contiguous chunk)
- **TLB pressure**: Each random table access hits a different page → high TLB miss rate

### 14.2 Page Size Impact

Measured with `stream` kernel, seq=64K, dim=4096, vocab=151936:

| Page Size | 1 thread | 12 threads (1 CMG) | 48 threads (4 CMG) |
|:---------:|:--------:|:------------------:|:------------------:|
| **2 MiB (hugepage)** | **42.3 GB/s** | 123.3 GB/s | 77.7 GB/s |
| **64 KiB (normal)** | **33.0 GB/s** | 125.0 GB/s | 71.7 GB/s |
| **Hugepage uplift** | **+28%** | ~0% | ~8% |

### 14.3 Analysis

**Why hugepages help single-core (+28%):**
- Table = 2.37 GB → 37K entries with 64KB pages, only ~1200 with 2MB pages
- A64FX L2 TLB: 1024 entries, 4-way associative
- 64KB pages: TLB covers 64MB → constant misses into 2.37GB table
- 2MB pages: TLB covers 2GB → near-complete coverage

**Why hugepages don't help at 12+ threads:**
- 12 threads × OoO execution window = enough concurrency to hide TLB miss latency (~260 cycles)
- HBM2 bandwidth becomes the bottleneck, not address translation
- Multi-threaded memory-level parallelism masks TLB penalties

### 14.4 Paging Policy Impact

| Paging Policy | 12 threads | 48 threads |
|:---:|:---:|:---:|
| prepage (default for malloc) | 123.3 GB/s | 77.7 GB/s |
| demand (NUMA-local) | 123.3 GB/s | 69.8 GB/s |

Demand paging does **not** help embedding lookups because:
- The embedding table is **shared read-only** — all CMGs need access to the full table
- Demand paging distributes pages across CMGs, but random lookups cross all NUMA domains regardless
- Output is already NUMA-local (sequential writes, each thread owns its chunk)
- The Stream Triad benchmark (section 7.2) benefits from demand paging because each thread reads/writes its own region; embedding is the opposite pattern.

### 14.5 Zipfian Distribution + L2 Cache Effect

With realistic Zipfian index distributions (popular tokens accessed more frequently), L2 cache amplification dominates over TLB effects:

| Distribution | Unique/64K | 12 threads, dim=4096 | vs uniform |
|:---:|:---:|:---:|:---:|
| Uniform (α=0.0) | 51K | 123 GB/s | baseline |
| Zipfian α=0.8 | 32K | 128 GB/s | +4% |
| Zipfian α=1.0 | 19K | 144 GB/s | +17% |
| Zipfian α=1.2 | 8K | **169 GB/s** | **+37%** |

At α=1.2, only 8K unique rows are accessed. Hot rows stay in the 8MB L2 cache, reducing HBM2 traffic. The "effective bandwidth" of 169 GB/s exceeds the per-CMG HBM2 share (~256 GB/s) because many lookups are served from L2.

### 14.6 Recommended Settings for Embedding Workloads

```bash
# Use default hugepage settings (already optimal)
# fcc links -Klargepage by default — no changes needed

# For single-core or low-thread-count work:
# Hugepages give +28% — ensure libmpg is linked and binary is ET_EXEC (not PIE)

# For 12+ threads:
# Hugepages have negligible effect — TLB latency hidden by thread-level parallelism
# Default paging policy (prepage for malloc) is fine

# Verify settings:
export XOS_MMM_L_PRINT_ENV=on
./bench_embedding  # check stderr for libmpg parameters

# Verify binary type (must be ET_EXEC, not ET_DYN):
readelf -h bench_embedding | grep Type
```

---

## Part XV: NUMA-Replicated Multi-CMG Scaling

### 15.1 Problem: Inter-CMG Ring Bottleneck

When all 48 cores access a single embedding table, the inter-CMG ring bus becomes the bottleneck:

```
                 ┌───────────┐
                 │   CMG 0   │ ← table lives here (first-touch by main thread)
                 │ 12 cores  │
                 │ ~256 GB/s │
                 └─────┬─────┘
        ring ←─────────┼──────────→ ring
        ~100 GB/s      │           ~100 GB/s
  ┌─────┴─────┐        │        ┌─────┴─────┐
  │   CMG 3   │        │        │   CMG 1   │
  │ 12 cores  │        │        │ 12 cores  │
  └─────┬─────┘        │        └─────┬─────┘
        ring ←─────────┼──────────→ ring
                 ┌─────┴─────┐
                 │   CMG 2   │
                 │ 12 cores  │
                 └───────────┘
```

With a shared table, 48-thread embedding achieves only **78 GB/s** — the ring's ~100 GB/s shared bandwidth is the ceiling. Each CMG has ~256 GB/s local read bandwidth, but remote access is bottlenecked at ~25 GB/s per CMG (100/4).

### 15.2 Solution: Full NUMA Replication

Replicate **all** data structures to each CMG's local HBM2:

1. **Embedding table** — N copies in FP16 (1.24 GB/replica × 4 = 4.96 GB, fits in 32 GB HBM2)
2. **Output buffer** — separate per-CMG allocation (each CMG writes only to its own)
3. **Index array** — per-CMG copy of that CMG's index chunk

Each CMG operates on purely local data with zero inter-CMG traffic:

```
CMG c:
  read:  cmg_indices[c][0..seq/N-1]           → LOCAL HBM2  (256 KB)
  read:  cmg_tables[c][idx * dim .. +dim-1]   → LOCAL HBM2  (1.24 GB)
  write: cmg_outputs[c][tok * dim .. +dim-1]   → LOCAL HBM2  (537 MB)
```

### 15.3 Implementation Details

**First-touch NUMA placement** with demand paging (`XOS_MMM_L_PAGING_POLICY=demand:demand:demand`):

```c
// Thread-to-CMG mapping: OMP_PROC_BIND=close OMP_PLACES=cores
//   threads 0-11  → CMG0
//   threads 12-23 → CMG1
//   threads 24-35 → CMG2
//   threads 36-47 → CMG3

// Each CMG's 12 threads first-touch their own allocations:
#pragma omp parallel
{
    int cmg = omp_get_thread_num() / 12;
    int local_tid = omp_get_thread_num() % 12;

    // 12-way parallel memcpy → pages placed on this CMG's HBM2
    memcpy(cmg_tables[cmg] + my_chunk, master_table + my_chunk, ...);
    memcpy(cmg_indices[cmg] + my_chunk, indices + global_offset + my_chunk, ...);
    memset(cmg_outputs[cmg] + my_chunk, 0, ...);
}
```

**FP16 precision** halves memory per replica (2 bytes vs 4 bytes per element), enabling 4 table copies in ~5 GB vs ~10 GB for FP32. The FP16 stream ASM kernel uses SVE `ld1h`/`st1h` with `.h` predicates and 8-row deep-prefetch pipelining.

**Sequence partitioning**: `seq/N_CMG` tokens per CMG, each CMG's 12 threads further divide their chunk. No overlap, no communication.

### 15.4 Results: Linear Scaling Achieved

Test: vocab=151936, dim=4096, seq=256K, FP16, best-of-5 timing.

| Config | Threads | Time (ms) | BW (GB/s) | Scaling | Target |
|:---|:---:|:---:|:---:|:---:|:---:|
| FP16 1 CMG | 12 | 39.2 | 109.5 | 1.00x | 1x |
| FP16 2 CMG | 24 | 19.6 | 218.6 | **2.00x** | 2x |
| FP16 3 CMG | 36 | 13.1 | 326.9 | **2.98x** | 3x |
| FP16 4 CMG | 48 | 9.9 | **434.6** | **3.97x** | 4x |
| FP32 1 CMG (ref) | 12 | 59.0 | 145.4 | — | — |

All targets achieved within 1% of ideal linear scaling.

### 15.5 Without NUMA Replication (Before Fix)

For comparison, a single shared output buffer with thread-distributed first-touch:

| Config | BW (GB/s) | Scaling | Issue |
|:---|:---:|:---:|:---|
| 1 CMG | 109.5 | 1.00x | OK — all local |
| 2 CMG | 141.8 | 1.30x | Output pages on wrong CMG |
| 3 CMG | 160.5 | 1.47x | Ring-limited output writes |
| 4 CMG | 138.2 | 1.26x | Severe ring contention |

**Root cause**: The shared output buffer's first-touch init distributed pages evenly by thread index, but the timed run partitioned by CMG. CMG1's output pages were physically on CMG0's HBM2, causing all writes to cross the ring.

### 15.6 Key Findings

1. **All three data streams must be NUMA-local**: table (read), output (write), *and* indices (read). Even small cross-CMG traffic (1 MB indices) destabilizes 4-CMG timing due to ring contention under full load.

2. **Per-CMG output buffers are critical**: A single contiguous output array with "distributed first-touch" does NOT work. The first-touch pattern must exactly match the timed-run access pattern. Separate `malloc()` per CMG is the cleanest solution.

3. **Best-of-N timing filters OS noise**: With all 48 cores active, OS tasks occasionally preempt compute threads. Taking the minimum of 5 iterations (vs averaging) gives stable, reproducible results.

4. **FP16 enables 4-way replication**: At dim=4096, FP16 table = 1.24 GB × 4 = 4.96 GB. FP32 would require 9.92 GB for 4 replicas. Both fit in 32 GB HBM2, but FP16 leaves more room for other data.

5. **Per-CMG BW is constant**: Each CMG independently achieves ~109 GB/s regardless of how many CMGs are active, confirming zero inter-CMG interference with full replication.

### 15.7 Recommended Environment

```bash
# NUMA-local page placement (essential for multi-CMG)
export XOS_MMM_L_PAGING_POLICY=demand:demand:demand

# Pin threads to cores, close-packed within CMGs
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_NUM_THREADS=48  # 4 CMG × 12 cores

# Run
./bench_embedding_omp --cmg-only
```
