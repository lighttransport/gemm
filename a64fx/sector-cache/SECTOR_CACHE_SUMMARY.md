# A64FX Sector Cache Hint (Tagged Pointer) Test Summary

## Overview

A64FX L1 Data Cache has a **sector cache** feature:
- L1D cache: 64KB, 4-way set associative
- 2 sectors: Sector 0 (way 0,1) for data reuse, Sector 1 (way 2,3) for streaming
- Tagged pointer: bits 59:56 of virtual address control sector assignment

## Tag Values

| Tag | Value | Description |
|-----|-------|-------------|
| TAG_NORMAL | 0x0 | No hint, uses both sectors |
| TAG_SECTOR0 | 0x1 | Sector 0 preferred (reuse data) |
| TAG_SECTOR1 | 0x2 | Sector 1 preferred (streaming) |
| TAG_SECTOR0_S | 0x9 | Sector 0 strong hint |
| TAG_SECTOR1_S | 0xA | Sector 1 strong hint |
| TAG_BYPASS_L1 | 0xB | Bypass L1 cache (go directly to L2) |

## Usage

Apply tag to pointer:
```c
#define APPLY_TAG(ptr, tag) ((void*)((uint64_t)(ptr) | ((tag) << 56)))

float* streaming_data = (float*)APPLY_TAG(data, 0x2);  // Sector 1
float* reuse_data = (float*)APPLY_TAG(data, 0x1);      // Sector 0
```

## Test Results

### Test 1: L1 Bypass (TAG 0xB)

**This clearly works:**

| Access Type | Bandwidth | Cycles |
|-------------|-----------|--------|
| Normal | 12.76 GB/s | 102,749 |
| L1 Bypass (0xB) | **24.14 GB/s** | 54,290 |

**Speedup: 1.9x** - Bypassing L1 avoids cache pollution for streaming data.

### Test 2: Sector Hints (TAG 0x1/0x2)

Mixed workload test (reuse data + streaming data):

| Configuration | Cycles |
|---------------|--------|
| No hints | 35,188 |
| With hints (Sector 0 + Sector 1) | 35,622 |

**Result: ~1% difference** - Minimal measurable effect.

### Test 3: DGEMM Kernel Comparison

Fujitsu reference DGEMM kernel (with sector hints) vs modified version (without):

| Matrix Size | With Hints | Without Hints | Speedup |
|-------------|------------|---------------|---------|
| 64x80x64 | 133 GFLOPS | 130 GFLOPS | 1.02x |
| 128x160x256 | 330 GFLOPS | 330 GFLOPS | 1.00x |
| 256x160x512 | 399 GFLOPS | 398 GFLOPS | 1.00x |

**Result: 0-2% difference**

## Fujitsu Reference Kernel Sector Usage

From `dgemm.kernel.s`:
```assembly
mov  x22, 0x2          // Tag 0x2 = Sector 1 (streaming)
lsl  x22, x22, 56
orr  x6, x6, x22       // Apply to C matrix (output)

mov  x22, 0x41         // Tag 0x41 = prefetch + Sector 0
lsl  x22, x22, 56
orr  x3, x3, x22       // Apply to A matrix (reuse)
```

- **A matrix (reused):** Sector 0 + L1 prefetch hint (tag 0x41)
- **C matrix (output):** Sector 1 for streaming writes (tag 0x2)
- **B matrix:** Uses explicit prefetch instructions (`prfm pldl1keep`)

## Conclusions

1. **L1 Bypass (TAG 0xB) is effective** - Shows clear 1.9x bandwidth improvement for streaming data.

2. **Sector hints (TAG 0x1/0x2) show minimal effect** in tested scenarios:
   - A64FX hardware prefetcher is highly effective
   - L2 cache (8MB) absorbs most working sets
   - Regular GEMM access patterns benefit more from prefetch than sector hints

3. **When sector hints may help:**
   - Irregular access patterns where prefetch doesn't help
   - Mixed workloads with clear hot/cold data separation
   - When L1 capacity is critical and data reuse is important

4. **Recommended usage:**
   - Use TAG 0xB (L1 bypass) for large streaming arrays that won't be reused
   - Use TAG 0x1 (Sector 0) for small frequently-accessed data structures
   - Use TAG 0x2 (Sector 1) for output data that's written once

## Files

- `bench_sector_cache.c` - Basic sector cache tests
- `bench_sector_v2.c` - Improved tests with cache flush
- `bench_dgemm_sector.c` - DGEMM kernel comparison
- `dgemm.kernel.s` - Fujitsu reference kernel (with hints)
- `dgemm_no_sector.s` - Modified kernel (without hints)

## References

- A64FX Microarchitecture Manual
- Fujitsu A64FX sample code: https://github.com/fujitsu/A64FX
