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

## Additional tuning options

| Variable | Value | Purpose |
|----------|-------|---------|
| `XOS_MMM_L_ARENA_LOCK_TYPE=0` | Parallel malloc | Use if multi-threaded allocation |
| `XOS_MMM_L_COLORING=1` | Cache coloring | Reduces L1D conflict misses (default on) |
| `XOS_MMM_L_PRINT_ENV=on` | Debug | Print all libmpg settings at startup |
| `XOS_MMM_L_HUGETLB_FALLBACK=1` | Safety | Fall back to 4KB pages instead of OOM kill |
