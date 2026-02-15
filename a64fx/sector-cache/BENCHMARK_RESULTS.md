# A64FX Sector Cache Benchmark Results

Measured on Fugaku (A64FX @ 2.0 GHz, 1 CMG = 12 cores, 8 MB L2 shared).
All runs use `FLIB_SCCR_CNTL=TRUE FLIB_L1_SCCR_CNTL=TRUE FLIB_L2_SCCR_CNTL_EX=TRUE`.

---

## 1. Gather-Stream Benchmark (`bench_gather_sector.c`)

### 1.1 Pattern

```
result += values[i] * table[indices[i]]
```

- **table**: medium array, random access (keep in cache, sector 0)
- **values + indices**: large sequential stream, 128 MB total (discard, sector 1)
- Models the SpMV inner loop: sequential val/col_idx stream, random access into x vector

### 1.2 Sector Cache Configurations Tested

| Label | Tag | L1 SCCR | L2 SCCR | Description |
|-------|:---:|:-------:|:-------:|-------------|
| nohint | none | default | default | Baseline (no sector partitioning) |
| tag 0xA | 0xA | default | default | Address tag with bit 59=1, sector_id=2 |
| tag 0x2 | 0x2 | default | default | Address tag with bit 59=0, sector_id=2 |
| manual L1(3,1) L2(13,1) | 0xA | sec0=3,sec1=1 | sec0=13,sec1=1 | Manual SCCR: max table ways |
| manual L1(3,1) L2(12,2) | 0xA | sec0=3,sec1=1 | sec0=12,sec1=2 | Manual SCCR: balanced |
| manual L1(2,2) L2(9,5) | 0xA | sec0=2,sec1=2 | sec0=9,sec1=5 | Manual SCCR: equal split |

### 1.3 Single-Thread Results

```
table=512KB (1 way), stream=128MB
  nohint (baseline)                            1.04 GB/s  1.000x
  tagged B=0xA (SCE+sec2)                      1.15 GB/s  1.102x
  manual L1(3,1) L2(13,1) B=0xA                1.16 GB/s  1.109x

table=2MB (4 ways), stream=128MB
  nohint (baseline)                            1.02 GB/s  1.000x
  tagged B=0xA (SCE+sec2)                      1.14 GB/s  1.114x
  manual L1(3,1) L2(13,1) B=0xA                1.15 GB/s  1.118x

table=4MB (8 ways), stream=128MB
  nohint (baseline)                            1.02 GB/s  1.000x
  tagged B=0xA (SCE+sec2)                      1.14 GB/s  1.117x
  manual L1(3,1) L2(13,1) B=0xA                1.14 GB/s  1.120x

table=6MB (12 ways), stream=128MB
  nohint (baseline)                            0.90 GB/s  1.000x
  tagged B=0xA (SCE+sec2)                      0.87 GB/s  0.967x  ← NEGATIVE
  manual L1(3,1) L2(12,2) B=0xA                0.87 GB/s  0.971x  ← NEGATIVE
```

**Single-thread summary**: 10-12% improvement for tables up to 4MB (1-8 L2 ways). Table at 6MB (12 ways) causes degradation — table is too large relative to L2 (14 usable ways), leaving insufficient room for the stream sector.

### 1.4 12-Thread Results (1 CMG, numactl --cpunodebind=4 --membind=4)

```
table=512KB (1 way), stream=128MB, 12 threads
  nohint (baseline)                            9.15 GB/s  1.000x
  tagged B=0xA (SCE+sec2)                      9.75 GB/s  1.066x
  manual L1(3,1) L2(13,1) B=0xA               10.01 GB/s  1.095x  ← best

table=2MB (4 ways), stream=128MB, 12 threads
  nohint (baseline)                            8.75 GB/s  1.000x
  tagged B=0xA (SCE+sec2)                     10.34 GB/s  1.182x  ← best
  manual L1(2,2) L2(9,5) B=0xA               10.34 GB/s  1.182x

table=4MB (8 ways), stream=128MB, 12 threads
  nohint (baseline)                            9.20 GB/s  1.000x
  tagged B=0xA (SCE+sec2)                     10.94 GB/s  1.189x  ← best
  manual L1(2,2) L2(9,5) B=0xA               10.94 GB/s  1.189x

table=6MB (12 ways), stream=128MB, 12 threads
  nohint (baseline)                            9.02 GB/s  1.000x
  tagged B=0xA (SCE+sec2)                      9.86 GB/s  1.093x
  manual L1(2,2) L2(9,5) B=0xA                9.84 GB/s  1.090x
```

**12-thread summary**: Up to **19% improvement** at 4MB table (8 L2 ways). The benefit scales with thread count because 12 threads sharing L2 create more contention. Even the 6MB table improves by 9% (unlike single-thread where it degraded).

### 1.5 Best Configurations by Table Size

| Table | L2 Ways | 1-Thread Best | Speedup | 12-Thread Best | Speedup |
|------:|--------:|:--------------|--------:|:---------------|--------:|
| 512KB | 1 | manual L1(3,1) L2(13,1) | 1.11x | manual L1(3,1) L2(13,1) | **1.10x** |
| 2MB | 4 | manual L1(3,1) L2(13,1) | 1.12x | tagged 0xA | **1.18x** |
| 4MB | 8 | manual L1(3,1) L2(13,1) | 1.12x | tagged 0xA | **1.19x** |
| 6MB | 12 | — (negative) | 0.97x | tagged 0xA | **1.09x** |

---

## 2. fapp PMU Profiling (Gather-Stream)

### 2.1 SC1: Cache Performance Events

Single-thread, table=2MB+4MB combined (2 calls per region).

Events: `0x0011` CPU_CYCLES, `0x0003` L1D_CACHE_REFILL, `0x0004` L1D_CACHE, `0x0008` INST_RETIRED, `0x0200` L2D_CACHE_REFILL, `0x0015` L1D_CACHE_WB, `0x0184` LD_COMP_WAIT, `0x0180` LD_COMP_WAIT_L2_MISS.

| Region | CPU_CYC | L1D_REFILL | L1D_ACC | INST_RET | L2D_REFILL | L1D_WB | LD_WAIT | LD_WAIT_L2M |
|--------|--------:|-----------:|--------:|---------:|-----------:|-------:|--------:|------------:|
| nohint | 7,359M | 507M | 2,681M | 1,997M | 479M | 41.8M | 2,967M | 398M |
| tag_0x2 | 6,557M | 479M | 2,176M | 5,167M | 479M | 14.6K | 1,845M | 1,434M |
| tag_0xA | 6,557M | 479M | 2,176M | 5,167M | 479M | 14.8K | 1,844M | 1,434M |
| manual_13_1 | 6,543M | 476M | 2,173M | 5,167M | 476M | 16.1K | 1,839M | 1,431M |

#### Derived Metrics

| Metric | nohint | sector (tag_0x2) | Change |
|--------|-------:|-----------------:|:------:|
| L1D miss rate | 18.9% | 22.0% | +3.1pp |
| L2D miss rate | 120.3% | 33.4% | **-86.9pp** |
| L1D_WB | 41,796,450 | 14,636 | **-99.96%** |
| IPC | 0.27 | 0.79 | **+193%** |

**Key insight**: L1 miss rate slightly increases (22% vs 19%) because more of L1 is used for table data vs stream. But L2 miss rate drops dramatically (120%→33%) because the table is **protected in L2** — misses that previously went to HBM now hit L2. The 99.96% drop in L1D_WB confirms stream data is no longer polluting the table's cache lines.

The L2D_REFILL count is nearly identical (479M both cases) — the same total volume of data comes from HBM either way. The difference is **which data gets evicted**: without sector cache, the table gets evicted and every random table access goes to HBM; with sector cache, only the stream gets evicted (cheap, sequential, prefetchable).

### 2.2 SC3: Sector Cache Tag Events

Single-thread, table=2MB+4MB combined (2 calls per region).

Events: `0x0011` CPU_CYCLES, `0x0240` L1_PIPE0_VAL, `0x0241` L1_PIPE1_VAL, `0x0250` L1_PIPE0_VAL_IU_TAG_ADRS_SCE, `0x0252` L1_PIPE1_VAL_IU_TAG_ADRS_SCE, `0x02a0` L1_PIPE0_VAL_IU_NOT_SEC0, `0x02a1` L1_PIPE1_VAL_IU_NOT_SEC0, `0x0260` L1_PIPE0_COMP.

| Region | L1_TAG_SCE | L1_SCE_active | L1_NOT_SEC0 | L2_SC_MISS |
|--------|----------:|-----------:|------------:|-----------:|
| nohint | 5,966M | 2,680M | **0** | 367M |
| tag_0x2 | 5,136M | 2,263M | **1,231M** | 707M |
| tag_0xA | 5,136M | 2,263M | **1,231M** | 707M |
| manual_13_1 | 5,119M | 2,261M | **1,233M** | 705M |

**Key observations:**
- **L1_NOT_SEC0 = 0 for nohint** confirms no sector tags applied at baseline
- **L1_NOT_SEC0 = 1.23B for sector variants** confirms address tags are correctly routing ~24% of accesses to non-default sectors (the stream data)
- **L2_SC_MISS increases** from 367M to 707M with sector cache — this is expected: stream data in sector 1 gets evicted faster (intended behavior)
- **SCE_active ratio is ~44% for ALL variants including nohint** — this counter tracks system-level SCE (always ON on Fugaku), not per-address bits

---

## 3. SCE Bit Investigation

### 3.1 Question

Does bit 59 of the virtual address act as a per-access SCE (Sector Cache Enable)?

The address tag format was believed to be bits [59:56] = `{SCE, PFE, sec_id[1:0]}`. If true, tag `0xA` (0b1010, SCE=1) should behave differently from `0x2` (0b0010, SCE=0).

### 3.2 Experiment

Gather-stream benchmark with fapp PMU profiling. Compared:
- **tag 0xA** = 0b**1**010 → bit 59=1, sector_id=2
- **tag 0x2** = 0b**0**010 → bit 59=0, sector_id=2

All other conditions identical.

### 3.3 Results

| PMU Counter | tag 0xA | tag 0x2 | Difference |
|-------------|--------:|--------:|:----------:|
| CPU_CYCLES | 6,560,972,693 | 6,563,190,723 | -0.034% |
| L1_SCE_active (0x0250+0x0252) | 2,263,047,023 | 2,263,224,226 | -0.008% |
| L1_NOT_SEC0 (0x02a0+0x02a1) | 1,231,006,377 | 1,231,109,507 | -0.008% |
| L2_SC_MISS (0x0260) | 706,700,108 | 706,675,684 | +0.003% |

All counters match within noise (<0.04%). **Bit 59 has zero effect.**

### 3.4 Additional Evidence

PMU event 0x0250 ("TAG_ADRS_SCE") shows ~44% for **all variants including nohint**:

| Region | SCE_active / L1_TAG (ratio) |
|--------|:---------------------------:|
| nohint (no tags) | 44.9% |
| tag 0x2 (bit 59=0) | 44.1% |
| tag 0xA (bit 59=1) | 44.1% |
| manual SCCR | 44.2% |

This proves the PMU event counts the **system-level SCE state** (from `IMP_FJ_TAG_ADDRESS_CTRL_EL1`), not a per-address bit. Since HPC tag override is always enabled on Fugaku, system SCE is always active.

### 3.5 Conclusion

**SCE (Sector Cache Enable) is a system register control, NOT a per-address bit.**

- SCE lives in `IMP_FJ_TAG_ADDRESS_CTRL_EL1`, always ON on Fugaku
- Only bits [57:56] of the virtual address (sector_id) affect sector cache behavior
- Bit 59 is either unused or assigned to a different HPC feature (possibly prefetch control)
- The FCC compiler correctly uses only `orr Xn, Xbase, #(1<<56)` for sector 1

### 3.6 With vs Without FLIB Environment Variables

| Config | nohint→sector speedup |
|--------|:---------------------:|
| With FLIB env vars | **1.12x** |
| Without FLIB env vars | **1.06x** |

Without env vars, sector cache still works (address tags are recognized, SCCR writes succeed after first pragma call enables el0ae) but the **L2 way partition is not enforced**. The `FLIB_L2_SCCR_CNTL_EX=TRUE` variable triggers the runtime to program L2 SCCR with the pragma-specified partition instead of the default (sec0=14, sec1=0 = no partition).

PMU evidence: L2_SC_MISS and L1_NOT_SEC0 counts are nearly identical with/without env vars — the tags work identically. The difference is purely in CPU cycles because L2 enforcement protects the table more effectively.

---

## 4. GEMM Benchmark (`bench_gemm_sector.c`)

### 4.1 Pattern

Standard GEMM: `C[M×N] += A[M×K] × B[K×N]`, fp32, SVE microkernel (8×3 tile, unroll-4).

### 4.2 Single-Thread Results (M=N=K varied)

| M=N=K | Baseline GFLOPS | Best Sector GFLOPS | Speedup | % of 128 GFLOPS peak |
|------:|----------------:|-------------------:|--------:|---------------------:|
| 256 | 106.2 | 106.5 | 1.003x | 83% |
| 512 | 107.9 | 107.6 | 0.997x | 84% |
| 1024 | 113.5 | 113.3 | 0.998x | 89% |
| 2048 | 102.1 | 101.8 | 0.997x | 80% |

**No benefit.** All configurations within ±0.3% of baseline.

### 4.3 12-Thread Shared-A Results (M=N=K varied)

| M=N=K | Baseline GFLOPS | Best Sector GFLOPS | Speedup |
|------:|----------------:|-------------------:|--------:|
| 256 | 816 | 807 | 0.989x |
| 512 | 720 | 724 | 1.006x |
| 1024 | 585 | 580 | 0.991x |
| 2048 | 537 | 534 | 0.994x |

**No benefit.** Sector cache slightly negative at most sizes.

### 4.4 12-Thread Independent-A Results (batched GEMM, each thread has own A/B/C)

| M=N=K | Baseline GFLOPS | Best Sector GFLOPS | Speedup |
|------:|----------------:|-------------------:|--------:|
| 256 | 808 | 800 | 0.990x |
| 512 | 632 | 626 | 0.990x |
| 1024 | 534 | 530 | 0.993x |

**No benefit.** Even with independent data per thread (no shared-A cache advantage), sector cache shows negative effect.

### 4.5 Why Sector Cache Doesn't Help GEMM

GEMM's tiled access pattern has inherent sequential locality that the hardware already handles optimally:

1. **A panel** is reused across N-tiles — it stays hot in L1/L2 naturally because it's accessed repeatedly before moving to the next K-block. LRU replacement keeps it resident without sector cache.

2. **B panel** is streamed sequentially — the hardware prefetcher detects and prefetches the stream perfectly. Sector cache can't improve on this.

3. **Compute-to-bandwidth ratio** is ~3.4 FLOPS/byte for fp32 GEMM. Single-thread achieves 80-89% of peak (compute-bound), meaning the memory subsystem is not the bottleneck.

4. **L1 bypass (tag 0xB)** actively hurts at large K on 12 cores (0.93x) because it forces B loads to skip L1, increasing L2 pressure without reducing A eviction.

---

## 5. Pointer-Chase Benchmark (Baseline Verification)

### 5.1 L1 Way Partition

16-32KB keep data, 96KB evict stream, both aliasing same L1 cache sets.

| Config | sec0 | sec1 | Reload cyc/load | Speedup |
|--------|:----:|:----:|:---------------:|:-------:|
| nohint | — | — | 17.2 (L2) | 1.00x |
| sector | 2 | 2 | 5.9 (L1) | **2.91x** |
| sector | 3 | 1 | 5.1 (L1) | **3.08x** |

### 5.2 L2 Way Partition

1-3.5MB keep data, 5-7MB evict stream.

| Config | sec0 | sec1 | Reload cyc/load | Speedup |
|--------|:----:|:----:|:---------------:|:-------:|
| nohint | — | — | 165 (Memory) | 1.00x |
| sector L2=5 | 9 | 5 | 51 (L2) | **3.25x** |
| manual | 14 | 1 | 51 (L2) | **3.25x** |

---

## 6. Environment & Build

### Build Commands

```bash
# Gather-stream benchmark (with OpenMP)
fcc -Nnoclang -O2 -Kocl,hpctag,openmp -o bench_gather_sector bench_gather_sector.c -lm

# Gather-stream with fapp instrumentation
fcc -Nnoclang -O2 -Kocl,hpctag -DUSE_FAPP -o bench_gather_sector_fapp bench_gather_sector.c -lm

# GEMM benchmark (with OpenMP + assembly microkernel)
fcc -Nnoclang -O2 -Kocl,hpctag,openmp -o bench_gemm_sector \
    bench_gemm_sector.c ../fused-gemm/micro_kernel_8x3.S -lm
```

### Run Commands

```bash
# Single-thread
FLIB_SCCR_CNTL=TRUE FLIB_L1_SCCR_CNTL=TRUE FLIB_L2_SCCR_CNTL_EX=TRUE \
    numactl --cpunodebind=4 --membind=4 ./bench_gather_sector

# 12-thread (1 CMG)
OMP_NUM_THREADS=12 OMP_STACKSIZE=64M \
    FLIB_SCCR_CNTL=TRUE FLIB_L1_SCCR_CNTL=TRUE FLIB_L2_SCCR_CNTL_EX=TRUE \
    numactl --cpunodebind=4 --membind=4 ./bench_gather_sector
```

### fapp Profiling

```bash
# SC3: Sector cache tag events
fapp -C -d prof_gather_sc3 -Icpupa \
    -Hevent_raw=0x0011,0x0240,0x0241,0x0250,0x0252,0x02a0,0x02a1,0x0260,method=fast,mode=user \
    ./bench_gather_sector_fapp

# SC1: Cache performance events
fapp -C -d prof_gather_sc1 -Icpupa \
    -Hevent_raw=0x0011,0x0003,0x0004,0x0008,0x0200,0x0015,0x0184,0x0180,method=fast,mode=user \
    ./bench_gather_sector_fapp

# Export CSV
fapppx -A -Icpupa -tcsv -o prof_gather_sc3.csv -d prof_gather_sc3
fapppx -A -Icpupa -tcsv -o prof_gather_sc1.csv -d prof_gather_sc1
```

### OMP Stack Size Note

FCC `-Nnoclang` with OpenMP requires `OMP_STACKSIZE=64M` (or larger) for loops with >1M iterations per thread. The default OMP stack size is insufficient and causes segfaults at ~16M iterations.
