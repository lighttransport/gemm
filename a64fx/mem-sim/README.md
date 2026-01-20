# A64FX Memory Access Simulator

A simple cycle-accurate memory access simulator for exploring cache behavior, prefetching strategies, and memory access patterns on the Fujitsu A64FX architecture without requiring access to actual hardware.

## Overview

This simulator models the A64FX memory hierarchy and provides detailed statistics on cache behavior, memory stalls, and bandwidth utilization. It's designed to help optimize memory access patterns for INT8 GEMM kernels and other compute-intensive workloads.

## Features

- **Cycle-accurate cache simulation**: Models L1 and L2 cache with accurate latencies
- **LRU replacement policy**: Simulates realistic cache eviction behavior
- **Software prefetch support**: Models `prfm` prefetch instructions
- **Detailed statistics**: Cache hit/miss rates, bandwidth, stalls, bottleneck analysis
- **Simple input format**: Pseudo-assembly instructions for easy experimentation

## A64FX Cache Configuration (Accurate Specifications)

The simulator models the following cache hierarchy based on actual A64FX specifications:

| Level | Size | Associativity | Line Size | Latency | Notes |
|-------|------|---------------|-----------|---------|-------|
| L1    | 64 KiB | 4-way | **256 bytes** | 11 cycles | MESI, writeback |
| L2    | **7 MiB** | **14-way** | **256 bytes** | 40 cycles (35-45) | MESI, writeback, XOR index |
| HBM2  | 8 GiB | -     | -        | **260 cycles** | 256 GB/s ld, 128 GB/s st |

**Key Features**:
- **Large cache lines**: 256 bytes (4× typical x86 systems)
- **L2 size**: 7 MiB usable (1 MiB reserved for assistant core)
- **L2 indexing**: XOR-based hashing to reduce conflicts
- **L1 indexing**: `index = (addr mod 16384) / 256`
- **MESI protocol**: Cache coherency (not fully modeled in simulator)

**SVE Vector Size**: 512 bits (64 bytes)

## Building

```bash
# Build the simulator
make

# Run all examples
make test

# Run individual examples
make run_sequential
make run_d256
make run_d512
make run_d512_prefetch

# Clean build artifacts
make clean
```

## Usage

```bash
./memsim <input_file>
```

### Input File Format

The simulator accepts pseudo-assembly instructions in text format:

```
# Comments start with #

# Memory operations
ld1 <addr> <size>         # Load operation
st1 <addr> <size>         # Store operation
prfm <addr>               # Prefetch operation
zfill <addr>              # Zero-fill cache line (A64FX optimization)

# Arithmetic operations (treated as 1-cycle compute)
sdot                      # INT8 SDOT instruction
fmla                      # FP multiply-add
fadd                      # FP add
# ... any other instruction
```

**Example**:
```
# Load 64 bytes from address 0x10000
ld1 0x10000 64

# Perform SDOT computation (1 cycle)
sdot

# Prefetch address 0x20000 into cache
prfm 0x20000

# Zero-fill cache line (allocate without memory fetch)
zfill 0x30000
```

## Examples

### 1. Sequential Access (`example_sequential.txt`)

Simple sequential vector processing pattern. Demonstrates ideal cache behavior with 100% L1 hit rate.

```bash
./memsim example_sequential.txt
```

**Expected**: Nearly perfect cache behavior, minimal stalls.

### 2. GEMM D=256 (`example_gemm_d256.txt`)

6-row INT8 GEMM kernel with D=256 hidden dimension. Working set: ~4 KB.

```bash
./memsim example_gemm_d256.txt
```

**Expected**: ~90% L1 hit rate, good performance.

### 3. GEMM D=512 (`example_gemm_d512.txt`)

6-row INT8 GEMM kernel with D=512 hidden dimension. Working set: ~8 KB.

```bash
./memsim example_gemm_d512.txt
```

**Expected**: ~75% L1 hit rate, more L2 accesses, lower efficiency.

### 4. GEMM D=512 with Prefetching (`example_gemm_d512_prefetch.txt`)

Same as D=512 but with software prefetching 2 K-groups ahead.

```bash
./memsim example_gemm_d512_prefetch.txt
```

**Expected**: Reduced memory stalls due to prefetching hiding L2 latency.

### 5. Zero-fill Optimization (`example_zfill.txt`)

Demonstrates using `zfill` to initialize output buffers efficiently.

```bash
./memsim example_zfill.txt
```

**Expected**: Zfill avoids DRAM fetches (260 cycles) when allocating cache lines that will be overwritten, providing significant speedup for initialization patterns.

## Output

The simulator produces three types of output:

### 1. Basic Statistics

```
Summary Statistics:
  Total cycles:          1234
  Compute cycles:        800
  Memory stall cycles:   434
  Instructions:          100
  IPC (effective):       0.08
```

### 2. Detailed Cache Statistics

```
L1 Cache Statistics:
  Size:              64 KB
  Associativity:     4-way
  Line size:         64 bytes
  Latency:           11 cycles

  Hits:              450
  Misses:            50
  Hit rate:          90.0%
  Evictions:         10

  Bytes read:        25600
  Bytes written:     0
  Bandwidth:         20.8 GB/s

L2 Cache Statistics:
  ...
```

### 3. Bottleneck Analysis

```
Bottleneck Analysis:
  Memory-bound: YES (35% of cycles are memory stalls)

  Time breakdown:
    Compute:         65%
    Memory stalls:   35%

  L1 cache efficiency: 90.0%
  L2 cache efficiency: 95.0%

  Recommendation: Good L1 hit rate. Consider increasing compute intensity.
```

## Understanding Results

### Cache Hit Rate

- **> 95%**: Excellent - working set fits in cache
- **85-95%**: Good - minor cache pressure
- **70-85%**: Moderate - some cache thrashing
- **< 70%**: Poor - significant cache misses

### Memory Stalls

- **< 20%**: Compute-bound (good for compute kernels)
- **20-40%**: Balanced
- **> 40%**: Memory-bound (optimization needed)

### Bandwidth

- **L1**: Up to ~200 GB/s (64 bytes * 2 GHz / 11 cycles ≈ 11.6 GB/s per stream)
- **L2**: Up to ~100 GB/s (64 bytes * 2 GHz / 31 cycles ≈ 4.1 GB/s per stream)
- **DRAM**: Up to ~32 GB/s

## Use Cases

### 1. Explore Prefetch Strategies

Create input files with different prefetch distances and patterns:
```
# Prefetch 1 K-group ahead
prfm 0x10100
ld1 0x10000 64
sdot

# Prefetch 2 K-groups ahead
prfm 0x10200
ld1 0x10100 64
sdot
```

### 2. Test Memory Layouts

Try different data layouts (row-major vs column-major):
```
# Row-major access
ld1 0x10000 64
ld1 0x10040 64
ld1 0x10080 64

# Column-major access (may cause cache conflicts)
ld1 0x10000 8
ld1 0x11000 8
ld1 0x12000 8
```

### 3. Optimize K-tiling

Test different K-tile sizes for blocking:
```
# Small tiles (better L1 locality)
# ... load 256 bytes ...
# ... compute ...

# Large tiles (better L2 locality, may thrash L1)
# ... load 2048 bytes ...
# ... compute ...
```

## Limitations

- **Simplified timing model**: Does not model out-of-order execution, instruction dependencies, or pipeline stalls
- **No prefetch queue**: Real hardware has limited prefetch buffers
- **No memory bandwidth limits**: Each access is independent
- **No TLB modeling**: Assumes infinite TLB
- **No cache coherency**: Single-threaded only

Despite these simplifications, the simulator provides useful insights into cache behavior and memory access patterns.

## Implementation Details

- **Cache implementation**: Set-associative with LRU replacement
- **Address mapping**: `set_index = (addr / line_size) % num_sets`
- **Tag comparison**: `tag = (addr / line_size) / num_sets`
- **Prefetch**: Brings line into L1, marks as useful if later accessed

## Files

- `memsim.h` - API header with configuration constants
- `memsim.c` - Core simulator implementation
- `memsim_main.c` - Main program and CLI
- `Makefile` - Build configuration
- `example_*.txt` - Example input files
- `README.md` - This documentation

## Performance Tips

Based on simulation results:

1. **D=256 kernels**: Don't use prefetching (working set fits in L1)
2. **D=512+ kernels**: Use prefetching 2-4 K-groups ahead (~16 KB distance)
3. **K-tiling**: For large D, process K in tiles of 32-64 K-groups
4. **Memory layout**: Keep frequently-accessed data within 4 KB for best L1 utilization

## References

- A64FX Microarchitecture Manual
- ARM SVE Programmer's Guide
- "Anatomy of High-Performance Matrix Multiplication" (Goto & van de Geijn)
- BLIS Framework cache blocking strategies

## Author

This simulator was created to support INT8 GEMM optimization work on the Fujitsu A64FX processor for LLM inference workloads.

## License

This code is for research and educational purposes.
