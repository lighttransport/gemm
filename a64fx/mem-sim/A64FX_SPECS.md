# A64FX Memory Hierarchy - Accurate Specifications

## Overview

This document describes the accurate A64FX memory hierarchy specifications as implemented in the simulator.

## L1 Data Cache

### Configuration
- **Size**: 64 KiB per CMG (Core Memory Group)
- **Associativity**: 4-way set associative
- **Line size**: **256 bytes** (not 64 bytes like typical x86!)
- **Latency**: 11 cycles
- **Policy**: Writeback
- **Coherency**: MESI protocol

### Index Calculation
```
index = (address mod 16384) / 256
```

Simple modulo-based indexing with 64 sets (64 KiB / 256 bytes / 4 ways = 64 sets).

### Key Features
- **Large cache lines**: 256-byte lines provide excellent spatial locality
- Each SVE load (64 bytes) benefits from 4× the spatial reuse
- Sequential access patterns get ~75-100% hit rates within a line

## L2 Cache

### Configuration
- **Size**: **7 MiB** usable (8 MiB total, 1 MiB reserved for assistant core)
- **Associativity**: **14-way** set associative
- **Line size**: **256 bytes** (same as L1)
- **Latency**: 35-45 cycles (**average 40 cycles**)
- **Policy**: Writeback
- **Coherency**: MESI protocol

### Index Calculation (XOR-based Hashing)
```
index<10:0> = ((PA<36:34> xor PA<32:30> xor PA<31:29> xor PA<27:25> xor PA<23:21>) << 8)
              xor PA<18:8>
```

This complex XOR-based hashing helps:
- Reduce cache conflicts
- Distribute addresses more uniformly
- Improve hit rates for strided access patterns

### Implementation
```c
int cache_l2_index(uint64_t addr) {
    // Extract bit fields
    uint64_t pa_36_34 = (addr >> 34) & 0x7;
    uint64_t pa_32_30 = (addr >> 30) & 0x7;
    uint64_t pa_31_29 = (addr >> 29) & 0x7;
    uint64_t pa_27_25 = (addr >> 25) & 0x7;
    uint64_t pa_23_21 = (addr >> 21) & 0x7;
    uint64_t pa_18_8  = (addr >> 8)  & 0x7FF;

    // XOR upper bits
    uint64_t xor_upper = pa_36_34 ^ pa_32_30 ^ pa_31_29 ^ pa_27_25 ^ pa_23_21;

    // Compute index
    uint64_t index = (xor_upper << 8) ^ pa_18_8;

    return (int)(index & 0x7FF);  // 11-bit index
}
```

### Key Features
- **2048 sets**: 7 MiB / 256 bytes / 14 ways = 2048 sets
- **XOR hashing**: Better distribution than simple modulo
- **Shared by CMG**: All cores in a CMG share the L2

## HBM2 Main Memory

### Configuration
- **Size**: 8 GiB per CMG
- **Type**: HBM2 (High Bandwidth Memory)
- **Latency**: **260 cycles** (not 200!)
- **Bandwidth**:
  - Load: **256 GB/s**
  - Store: **128 GB/s** (asymmetric!)

### Key Features
- **High latency**: ~130 ns at 2 GHz
- **High bandwidth**: But only if you can exploit parallelism
- **Asymmetric**: Stores are half the bandwidth of loads
- **NUMA**: Each CMG has its own HBM2 stack

## Zero-fill (zfill) Operation

### Purpose
Efficiently initialize cache lines to zero without fetching from memory.

### Traditional Approach (Slow)
```
1. Load cache line from DRAM (260 cycles)
2. Modify to zeros in cache
3. Write back to DRAM (if evicted)
Total: ~260+ cycles
```

### Zero-fill Approach (Fast)
```
1. Allocate L1 cache line directly (11 cycles)
2. Mark as zero-filled
3. No DRAM fetch needed
Total: 11 cycles
```

### Use Cases
- Initializing output buffers
- Clearing temporary arrays
- Allocating zero-initialized data structures

### Example
```
# Traditional (slow)
ld1 0x10000 64    # Load from DRAM (260 cycles)
# ... set to zero ...
st1 0x10000 64    # Write back

# Zero-fill (fast)
zfill 0x10000     # Allocate L1 line (11 cycles)
st1 0x10000 64    # Write data (L1 hit)
```

**Speedup**: ~23× faster (260 cycles → 11 cycles)

## Comparison with Typical x86 Systems

| Feature | A64FX | Typical x86 (e.g., Skylake) |
|---------|-------|------------------------------|
| L1 line size | **256 bytes** | 64 bytes |
| L2 line size | **256 bytes** | 64 bytes |
| L1 latency | 11 cycles | ~4-5 cycles |
| L2 latency | 35-45 cycles | ~12-15 cycles |
| L2 size | 7 MiB (shared) | 256 KB (private) |
| DRAM latency | 260 cycles | ~200 cycles |
| DRAM type | HBM2 | DDR4 |
| L2 indexing | XOR-based | Simple modulo |

## Key Takeaways

1. **Large cache lines are a major advantage**:
   - 256 bytes means 4× the spatial locality
   - Sequential access patterns benefit greatly
   - BUT: Wastes bandwidth if you don't use the whole line

2. **L2 latency is significant**:
   - 35-45 cycles is 3-4× higher than x86
   - Software prefetching is more important
   - Cache blocking is critical

3. **HBM2 latency hurts**:
   - 260 cycles is painful
   - Must maximize L1/L2 hit rates
   - Prefetching is essential for DRAM access

4. **XOR-based L2 indexing helps**:
   - Better than simple modulo for strided patterns
   - Reduces conflict misses
   - Important for scientific workloads

5. **Zero-fill is a major optimization**:
   - 23× speedup for initialization
   - Use whenever allocating buffers you'll overwrite
   - Common in output buffers, temporary storage

## Performance Guidelines

### For Best L1 Hit Rates
- Keep working set < 16 KB
- Reuse data within 256-byte boundaries
- Sequential access patterns work well

### For Best L2 Hit Rates
- Keep working set < 2 MB
- Use blocking/tiling for large arrays
- Prefetch 2-4 iterations ahead

### For Hiding DRAM Latency
- Prefetch distance: ~16-32 KB ahead
- Use zfill for initialization
- Overlap computation with memory access

## Simulator Accuracy

### What the Simulator Models Correctly
✓ Cache sizes and associativity
✓ Cache line sizes (256 bytes)
✓ L1 and L2 latencies
✓ DRAM latency (260 cycles)
✓ L1 simple indexing
✓ L2 XOR-based indexing
✓ LRU replacement policy
✓ Zero-fill optimization

### What the Simulator Simplifies
✗ MESI protocol (assumes perfect coherency)
✗ Out-of-order execution
✗ Instruction-level parallelism
✗ Memory bandwidth limits
✗ TLB misses
✗ Prefetch queue depth
✗ Store buffer behavior
✗ Multi-core interference

Despite simplifications, the simulator provides valuable insights into cache behavior and memory access patterns.

## References

- Fujitsu A64FX Microarchitecture Manual
- A64FX White Paper
- ARM Neoverse N1 TRM (similar architecture)
- "Understanding A64FX Performance Characteristics" (various papers)

## Changelog from Previous Version

### Changed
- L1 line size: 64 → **256 bytes**
- L2 size: 8 MiB → **7 MiB** (1 MiB reserved)
- L2 ways: 16 → **14-way**
- L2 latency: 31 → **40 cycles** (35-45 range)
- L2 line size: 64 → **256 bytes**
- DRAM latency: 200 → **260 cycles**
- DRAM type: Generic → **HBM2**

### Added
- **zfill** instruction support
- Accurate L1 indexing formula
- Accurate L2 XOR-based indexing
- HBM2 bandwidth specifications (256 GB/s ld, 128 GB/s st)

### Impact
- **Higher hit rates** due to larger cache lines
- **Lower total cycles** for sequential access
- **More accurate** cache conflict modeling with XOR indexing
- **Better understanding** of initialization costs (zfill)
