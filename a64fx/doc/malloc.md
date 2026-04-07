# A64FX malloc / Hugepage Tuning for LLM Inference

## Problem

Large language models (e.g. Qwen3.5-9B BF16 = 17GB) mapped with standard 4KB pages cause severe TLB thrashing on A64FX. The micro-DTLB has only 32 entries and the main DTLB has 1024 entries — far too few for millions of 4KB pages. This makes kernel TLB refill the dominant cost, not the actual compute.

**Measured impact** (Qwen3.5-9B BF16, single thread, 1 token):

| Configuration | Time/token | Bottleneck |
|--------------|-----------|------------|
| mmap + 4KB pages | 6.9s | ~40% kernel (TLB misses) |
| malloc + 2MB hugepages (libmpg) | **1.92s** | ~26% SVE matvec (compute-bound) |

**3.6x speedup** from hugepages alone.

## Setup

### 1. Use malloc instead of mmap for model loading

Set `NO_MMAP=1` to force the GGUF loader to use `malloc()` instead of `mmap()`:

```bash
export NO_MMAP=1
```

When linked with libmpg, `malloc()` allocations go through `mmap(2)` with 2MB hugepage backing (via HugeTLBfs), eliminating TLB pressure.

### 2. Link with libmpg

**Fujitsu compiler (fcc/FCC):** Hugepages enabled by default (`-Klargepage`).

```bash
fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -o test test.c -lm -lpthread
```

Verify libmpg is linked:
```bash
ldd ./test | grep mpg
# Expected: libmpg.so.1 => /opt/FJSVxos/mmm/lib64/libmpg.so.1
```

**GCC / other compilers:** Link explicitly.

```bash
gcc -O3 -march=armv8.2-a+sve -no-pie \
    -Wl,-T/opt/FJSVxos/mmm/util/bss-2mb.lds \
    -L/opt/FJSVxos/mmm/lib64 \
    -o test test.c -lmpg -lc -lpthread -lm
```

Critical rules:
- `-lmpg` must precede `-lc` and `-lpthread`
- `-no-pie` required (PIE disables hugepages for .data/.bss)
- Verify: `readelf -h test` must show `Type: EXEC` (not `DYN`)

### 3. Environment variables

```bash
# Enable 2MB hugepages (default, but explicit for clarity)
export XOS_MMM_L_HPAGE_TYPE=hugetlbfs

# Paging policy: demand paging for NUMA-local allocation
export XOS_MMM_L_PAGING_POLICY=demand:demand:prepage

# Pool freed memory for reuse (avoids repeated mmap/munmap)
export XOS_MMM_L_ARENA_FREE=2

# Disable mmap for model loading (use hugepage-backed malloc)
export NO_MMAP=1
```

### 4. Complete run script

```bash
#!/bin/bash
export XOS_MMM_L_HPAGE_TYPE=hugetlbfs
export XOS_MMM_L_PAGING_POLICY=demand:demand:prepage
export XOS_MMM_L_ARENA_FREE=2
export NO_MMAP=1
OMP_NUM_THREADS=1 ./test_transformer model.gguf "Hello" 32 1
```

## Why mmap with file-backed pages is slow

- `mmap(fd, MAP_PRIVATE)` uses the kernel page cache with 4KB pages
- `MADV_HUGEPAGE` only works for **anonymous** mappings (THP), not file-backed mmap
- `MAP_HUGETLB` requires the file to reside on a HugeTLBfs mount (not available for regular files)
- `MAP_POPULATE` pre-faults pages but still uses 4KB granularity

For a 17GB model: 17GB / 4KB = ~4.4M page table entries. A64FX main DTLB holds 1024 entries, so every pass through the weights triggers massive TLB refills.

With `malloc()` + libmpg: 17GB / 2MB = ~8,500 entries. Fits comfortably in the DTLB with minimal misses.

## Perf profile comparison

### Before (mmap, 4KB pages)
```
21.05%  tf_qmatvec              # SVE BF16 matvec
~40%    kernel (TLB refill)     # Page table walks
 1.57%  tf_ssm_recurrence_worker
```

### After (malloc, 2MB hugepages)
```
26.38%  tf_qmatvec              # SVE BF16 matvec (now dominant)
 1.65%  tf_ssm_recurrence_worker
 0.85%  bpe_hm_set
 0.43%  tf_ssm_deltanet_forward
```

The kernel overhead drops from ~40% to ~57% but shifts from TLB refill to the one-time cost of reading 17GB from disk into hugepage-backed memory. On subsequent tokens, the kernel contribution is negligible.

## SVE BF16 kernel hotspot (with hugepages)

```
20.9%  fmla z1.s, p0/m, z17.s, z5.s   # FP32 fused multiply-add
18.8%  lsl  z17.s, p0/m, z17.s, #16   # BF16 -> FP32 conversion
16.2%  ld1h {z16.s}, p0/z, [x3]       # Load BF16 weight (512-bit)
11.5%  ld1h {z17.s}, p0/z, [x4]       # Load BF16 weight (512-bit)
 5.7%  fmla z4.s, p0/m, z6.s, z5.s    # FMA row 3
 3.0%  fmla z2.s, p0/m, z16.s, z5.s   # FMA row 2
```

The kernel is now cleanly memory-bandwidth-bound. Each token streams ~17GB of weights through the SVE pipeline at ~8.5 GB/s, approaching single-CMG HBM2 bandwidth.

## NUMA-aware weight distribution across 4 CMGs

A64FX has 4 CMGs (Core Memory Groups), each with directly attached HBM2 (~7GB each). Cross-CMG memory access has higher latency. For multi-threaded inference with 1 core per CMG, weight pages should be placed on the CMG that will access them.

### The problem

With `NO_MMAP=1`, `fread` reads the entire model from the main thread. First-touch NUMA policy places all pages on the main thread's CMG. Other CMGs must do cross-CMG reads.

### The solution: `NUMA_DISTRIBUTE=1`

Set `NUMA_DISTRIBUTE=1` to enable parallel weight loading. Each thread reads its row partition of every weight tensor via `pread`, triggering first-touch page placement on the thread's own CMG. This aligns with how the matvec thread pool distributes rows.

### Benchmark (Qwen3.5-2B BF16, 4 threads)

| Config | cores | memory | Time/tok |
|--------|-------|--------|----------|
| 1 thread, 1 CMG | `-C12` | `-m4` | 0.429s |
| 4 threads, 1 CMG | `-C12,13,14,15` | `-m4` | 0.125s |
| 4 threads, 4 CMGs, no NUMA | `-C12,24,36,48` | `-m4-7` | **16.7s** |
| 4 threads, 4 CMGs, NUMA distribute | `-C12,24,36,48` | `-m4-7` | **0.132s** |

Without NUMA distribute, cross-CMG placement causes 130x slowdown. With NUMA distribute, performance matches same-CMG.

### Run script (multi-CMG)

```bash
#!/bin/bash
export XOS_MMM_L_HPAGE_TYPE=hugetlbfs
export XOS_MMM_L_PAGING_POLICY=demand:demand:demand
export XOS_MMM_L_ARENA_FREE=2
export NO_MMAP=1
export NUMA_DISTRIBUTE=1
# 1 core per CMG: core 12 (CMG0), 24 (CMG1), 36 (CMG2), 48 (CMG3)
numactl -C12,24,36,48 -m4-7 \
    ./test_transformer model.gguf "Hello" 32 4
```

### A64FX NUMA topology

```
Node 4: cores 12-23, ~7GB HBM2 (CMG0)
Node 5: cores 24-35, ~7GB HBM2 (CMG1)
Node 6: cores 36-47, ~7GB HBM2 (CMG2)
Node 7: cores 48-59, ~7GB HBM2 (CMG3)
```

Node distances (10=local, 20=adjacent, 30=remote):
```
     4   5   6   7
 4: 10  20  30  30
 5: 20  10  30  30
 6: 30  30  10  20
 7: 30  30  20  10
```

## Memory alignment

All scratch buffers are allocated with 256-byte alignment (`posix_memalign`) for SVE compatibility (512-bit = 64 bytes, but 256 is safer for future extensions). Model data uses 2MB alignment for hugepage boundaries.

## Thread pool: master-worker pattern

The thread pool uses a master-worker pattern: the main thread executes worker 0's task directly instead of spin-waiting. This eliminates one idle core and reduces cross-CMG atomic synchronization overhead. Only `n_threads - 1` background workers are created.

## NUMA allocator: `transformer_numa_setup`

The NUMA allocator runs 4 phases after model load and thread pool creation:

1. **Weight distribution** — parallel `pread` per tensor, row-partitioned across threads. Each thread reads its row partition from the GGUF file, triggering first-touch page placement on its CMG.
2. **Thread scratch** — per-thread dequant buffers (`thread_tmp[t]`) first-touched by their owning thread. Allocated with `posix_memalign` without `memset` (notouch), so demand paging places pages on the first-touching worker.
3. **Logits buffer** — row-partitioned first-touch across threads (248K rows = ~1MB, split 4 ways).
4. **SSM recurrent state** — head-partitioned first-touch. Each thread's `dt_rank/4` heads placed on its CMG (~3MB per SSM layer).

### NUMA allocator env vars

| Variable | Default | Description |
|----------|---------|-------------|
| `NUMA_DISTRIBUTE` | unset | Set to `1` to enable NUMA distribution |
| `NUMA_N_CMGS` | `4` | Number of CMGs |
| `NUMA_CMG_BUDGET_GB` | `6` | Usable GB per CMG (7GB total, 1GB reserved) |
| `NUMA_ALIGNMENT` | `2097152` | Allocation alignment in bytes (2MB) |

### Memory budget output

The allocator prints per-CMG usage at startup:
```
numa: per-CMG usage: CMG0=898.1MB CMG1=897.7MB CMG2=897.7MB CMG3=897.7MB (budget=6.0GB)
```

### What is NOT distributed (stays shared)

Small scratch buffers (x, xb, xb2, q, k, v, att, ffn_buf*) are <40KB each and stay on CMG0. Cross-CMG access for these is negligible. KV cache is accessed by all threads during attention, so it stays shared.

## Per-CMG Memory Pool with mbind (recommended)

The highest-performance approach uses `mmap` + `mbind` to create per-CMG memory pools with explicit NUMA node binding. This guarantees zero cross-CMG weight access.

### Why mbind instead of first-touch

First-touch (demand paging) relies on which thread touches each page first. This is fragile — a single `memset` or `fread` from the wrong thread places pages on the wrong CMG. `mbind` is explicit and deterministic.

### Key system facts

- **A64FX page size: 64KB** (not 4KB). Use `getpagesize()` to confirm.
- **NUMA nodes**: 4-7 correspond to CMG0-3 (each with ~7GB HBM2).
- **mbind works**: verified on Fugaku compute nodes. No special permissions needed.

### Implementation

```c
#include <sys/mman.h>
#include <sys/syscall.h>
#define mbind(a,l,m,nm,mx,f) syscall(SYS_mbind,(a),(l),(m),(nm),(mx),(f))
#define MPOL_BIND 2

/* Allocate pool pinned to a specific CMG */
void *cmg_pool_alloc(int cmg, size_t size) {
    void *p = mmap(NULL, size, PROT_READ|PROT_WRITE,
                   MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return NULL;
    unsigned long mask = 1UL << (4 + cmg);  /* nodes 4-7 */
    mbind(p, size, MPOL_BIND, &mask, 64, 0);
    return p;
}
```

Combine with `pthread_setaffinity_np` to pin each thread to the correct CMG's cores:
```c
cpu_set_t cpuset;
CPU_ZERO(&cpuset);
CPU_SET(12 + cmg * 12 + local_tid, &cpuset);  /* cores 12-23, 24-35, 36-47, 48-59 */
pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
```

### Benchmark results (FP16 4r×2vl matvec [12288, 4096] = 96MB)

| Config | Total GB/s | Per-CMG GB/s | % of 833 streaming peak |
|--------|-----------|-------------|------------------------|
| 1CMG 4T | 105 | 105 | — |
| 1CMG 8T | 181 | 181 | — |
| 1CMG 12T | 211 | 211 | — |
| 4CMG 16T (4/CMG) | 399 | 100 | 48% |
| 4CMG 32T (8/CMG) | 655 | 164 | 79% |
| 4CMG 40T (10/CMG) | 741 | 185 | 89% |
| **4CMG 48T (12/CMG)** | **756** | **189** | **91%** |

**756 GB/s aggregate = 91% of A64FX HBM2 streaming peak (833 GB/s).**

Each CMG achieves 189 GB/s independently — matching single-CMG results with zero cross-CMG penalty.

### Why first-touch fails for multi-CMG

With a single `posix_memalign` buffer, weight rows for different CMGs are on adjacent pages in the same allocation. Even with per-thread `memset` for first-touch, the 64KB page granularity can cause rows intended for different CMGs to share pages. With `mbind` per-CMG pools, each CMG's weights are in a completely separate virtual memory region with guaranteed node binding.

### Multi-CMG barrier: WFE intra-CMG only

The WFE (wait-for-event) barrier works within a single CMG. For multi-CMG, each CMG runs independently with its own WFE barrier group. Cross-CMG synchronization (when needed for the full inference pipeline) uses a lightweight atomic barrier between CMG leaders only — 4 threads instead of 48.

### BF16 vs FP16 kernel performance

| Kernel | Per-core GB/s | % of 32 GB/s peak | Notes |
|--------|-------------|-------------------|-------|
| BF16 `ld1uh_u32 + lsl + fmla32` | 18.0 | 56% | Current production |
| FP16 `ld1h + fmla16` (8r×1vl) | 24.3 | 76% | No conversion overhead |
| **FP16 `ld1h + fmla16` (4r×2vl)** | **27.8** | **87%** | **Best kernel** |
| FP32 ceiling `ld1w + fmla32` | 24.9 | 78% | Theoretical limit for 32-bit |

FP16 4r×2vl exceeds FP32 ceiling because FP16 processes 32 elements per SVE vector (64 bytes) vs 16 for FP32 — same memory bandwidth, 2× compute throughput.

For production LLM inference: pre-convert BF16 weights to FP16 at load time (one-time cost, same memory footprint), then use FP16 kernel throughout decode.

## Additional tuning options

| Variable | Value | Purpose |
|----------|-------|---------|
| `XOS_MMM_L_ARENA_LOCK_TYPE=0` | Parallel malloc | Use if multi-threaded allocation |
| `XOS_MMM_L_COLORING=1` | Cache coloring | Reduces L1D conflict misses (default on) |
| `XOS_MMM_L_PRINT_ENV=on` | Debug | Print all libmpg settings at startup |
| `XOS_MMM_L_HUGETLB_FALLBACK=1` | Safety | Fall back to 4KB pages instead of OOM kill |
