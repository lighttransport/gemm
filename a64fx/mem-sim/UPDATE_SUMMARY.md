# A64FX Memory Simulator - Update Summary

## Changes Made

The simulator has been updated with accurate A64FX specifications based on official documentation.

## Major Specification Updates

### L1 Data Cache
| Parameter | Old Value | New Value | Change |
|-----------|-----------|-----------|--------|
| Size | 64 KB | 64 KiB | (same) |
| Associativity | 4-way | 4-way | (same) |
| Line size | **64 bytes** | **256 bytes** | **4× larger!** |
| Latency | 11 cycles | 11 cycles | (same) |
| Index function | Simple modulo | `(addr mod 16384) / 256` | Formula specified |

### L2 Cache
| Parameter | Old Value | New Value | Change |
|-----------|-----------|-----------|--------|
| Size | **8 MB** | **7 MiB** | 1 MiB reserved for assistant core |
| Associativity | **16-way** | **14-way** | Reduced |
| Line size | **64 bytes** | **256 bytes** | **4× larger!** |
| Latency | **31 cycles** | **40 cycles** | More accurate (35-45 range) |
| Index function | Simple modulo | **XOR-based hashing** | **Major change!** |

### Main Memory (HBM2)
| Parameter | Old Value | New Value | Change |
|-----------|-----------|-----------|--------|
| Type | Generic DRAM | **HBM2** | Specified |
| Size | Unspecified | **8 GiB** | Specified |
| Latency | **200 cycles** | **260 cycles** | +30% higher! |
| Bandwidth (load) | Unspecified | **256 GB/s** | Specified |
| Bandwidth (store) | Unspecified | **128 GB/s** | **Asymmetric!** |

## New Features

### 1. Zero-fill (zfill) Instruction

**Purpose**: Efficiently initialize cache lines to zero without fetching from memory.

**Syntax**:
```
zfill <addr>        # Simple format
zfill [x0]          # Assembly format
```

**Performance**:
- Traditional: ~260 cycles (load from DRAM)
- Zero-fill: ~11 cycles (allocate L1 line directly)
- **Speedup: 23×**

**Use Cases**:
- Initializing output buffers
- Clearing temporary arrays
- Allocating zero-initialized structures

### 2. Accurate L2 Index Calculation

**Formula**:
```
index<10:0> = ((PA<36:34> xor PA<32:30> xor PA<31:29> xor PA<27:25> xor PA<23:21>) << 8)
              xor PA<18:8>
```

**Benefits**:
- Reduces conflict misses
- Better distribution for strided access
- More accurate simulation of real hardware

### 3. Accurate L1 Index Calculation

**Formula**:
```
index = (address mod 16384) / 256
```

Simple but correctly specified for A64FX.

## Impact on Simulation Results

### Hit Rates Changed Significantly

With 256-byte cache lines instead of 64-byte:
- **Sequential access**: 0% → **75%** L1 hit rate
- **Spatial locality**: 4× improvement (more data per line)
- **Total cycles**: Reduced by 2-3× for sequential patterns

### Example: Sequential Access Pattern

**Old (64-byte lines)**:
```
L1 hit rate:   0%
Total cycles:  3,888
Memory stalls: 95.1%
```

**New (256-byte lines)**:
```
L1 hit rate:   75%
Total cycles:  1,392
Memory stalls: 86.2%
```

**Improvement**: 2.8× fewer cycles!

### Why the Difference?

With 256-byte cache lines:
- Each line holds **4 SVE vectors** (64 bytes each)
- Sequential loads hit in the same line
- Fewer cold misses
- Better spatial locality

## Updated Examples

### 1. example_zfill.txt (NEW)

Demonstrates zero-fill optimization for buffer initialization.

**Key metrics**:
- L1 hit rate: 68%
- Total cycles: 813
- Shows benefit of zfill vs traditional load/store

### 2. Existing examples now more accurate

All existing examples (sequential, GEMM D=256, GEMM D=512, prefetch) now produce more realistic results with:
- Larger cache lines
- Accurate L2 indexing
- Correct latencies

## Code Changes

### memsim.h
- Updated cache configuration constants
- Added `ACCESS_ZFILL` to `access_type_t` enum
- Added `cache_l1_index()` and `cache_l2_index()` function declarations

### memsim.c
- Implemented `cache_l1_index()` with modulo formula
- Implemented `cache_l2_index()` with XOR-based hashing
- Modified `cache_access()` to use A64FX-specific index functions
- Added zfill support in `memsim_access()`
- Added zfill parsing in `memsim_execute_instruction()`

### Documentation
- Updated README.md with accurate specs
- Created A64FX_SPECS.md with detailed specifications
- Updated QUICKSTART.md with zfill and new specs
- Created this UPDATE_SUMMARY.md

## Validation

### Build Test
```bash
make clean && make
```
✓ Compiles without warnings

### Functional Test
```bash
./memsim example_zfill.txt
```
✓ Runs successfully
✓ Zfill operations tracked correctly
✓ Hit rates reasonable

### All Examples Test
```bash
make test
```
✓ All 5 examples run successfully

## Performance Guidelines (Updated)

### For Best L1 Hit Rates (with 256-byte lines)
- Keep working set < 32 KB (was: 16 KB)
- Sequential access within 256-byte boundaries is ideal
- Expect 75-100% hit rate for good sequential patterns

### For Best L2 Hit Rates
- Keep working set < 3 MB (was: 2 MB, accounting for 7 MiB L2)
- XOR indexing reduces conflicts for power-of-2 strides
- Prefetch 2-4 iterations ahead

### For Hiding DRAM Latency (updated latency)
- Prefetch distance: ~20-40 KB ahead (was: 16-32 KB)
- 260-cycle latency means prefetching is even more critical
- Use zfill for initialization (23× speedup)

## Backward Compatibility

### Breaking Changes
- Hit rates will be different (generally higher)
- Cycle counts will be different (generally lower for sequential)
- L2 indexing produces different conflict patterns

### Migration Guide
If you have existing input files:
1. Results will change but patterns should be similar
2. Prefetch benefits may be larger (higher DRAM latency)
3. Consider adding zfill for initialization patterns

## References

- Fujitsu A64FX Microarchitecture Manual (official specs)
- A64FX White Paper (cache hierarchy details)
- User-provided specifications (confirmed accurate)

## Files Modified

### Core Implementation
- `memsim.h` - Cache configuration and API
- `memsim.c` - Index functions, zfill support

### Examples
- `example_zfill.txt` - NEW

### Documentation
- `README.md` - Updated specs and examples
- `QUICKSTART.md` - Updated quick reference
- `A64FX_SPECS.md` - NEW (detailed specifications)
- `UPDATE_SUMMARY.md` - NEW (this file)

### Build System
- `Makefile` - Added zfill example

## Testing Recommendations

1. **Rerun existing simulations** with new specs
2. **Compare hit rates**: Should generally improve with larger lines
3. **Test zfill**: For initialization-heavy patterns
4. **Validate L2 indexing**: Check conflict patterns for your workload

## Conclusion

The simulator now accurately models the A64FX memory hierarchy including:
✓ Correct cache sizes and associativity
✓ Accurate 256-byte cache lines
✓ Proper L1 and L2 index functions
✓ Accurate latencies (L1, L2, HBM2)
✓ Zero-fill optimization support

These updates make the simulator a more accurate tool for exploring A64FX memory access patterns and optimization strategies.
