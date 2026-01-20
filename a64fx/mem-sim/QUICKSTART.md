# A64FX Memory Simulator - Quick Start Guide

## Build and Run (30 seconds)

```bash
# Build the simulator
make

# Run all examples
make test

# Run individual example
./memsim example_gemm_d512_prefetch.txt
```

## Create Your Own Input

Create a text file with pseudo-assembly instructions:

```bash
cat > my_pattern.txt << 'EOF'
# My memory access pattern
ld1 0x10000 64    # Load 64 bytes from 0x10000
sdot              # Computation (1 cycle)
ld1 0x10040 64    # Load next 64 bytes
sdot
prfm 0x10080      # Prefetch ahead
ld1 0x10080 64
sdot
EOF

# Run simulation
./memsim my_pattern.txt
```

## Supported Instructions

### Memory Operations
```
ld1 <addr> <size>         # Load operation
st1 <addr> <size>         # Store operation
prfm <addr>               # Prefetch operation
zfill <addr>              # Zero-fill (A64FX optimization)
```

### Compute Operations (1 cycle each)
```
sdot                      # INT8 SDOT
fmla                      # FP multiply-add
fadd                      # FP add
add                       # Integer add
mul                       # Integer multiply
```

### Assembly Format (also supported)
```
ld1 z0.b, p0/z, [x0]      # Load with register addressing
st1 z1.s, p1/z, [x1, #64] # Store with offset
prfm pldl1keep, [x2]      # Prefetch with hint
```

## Understanding Output

### Cache Hit Rate
- **> 90%**: Excellent
- **70-90%**: Good
- **50-70%**: Moderate
- **< 50%**: Poor

### Memory-Bound Check
- **< 20% stalls**: Compute-bound (good)
- **20-50% stalls**: Balanced
- **> 50% stalls**: Memory-bound (optimize memory!)

### Prefetch Effectiveness
Compare cycles with/without prefetch:
- Speedup > 1.5×: Very effective
- Speedup 1.1-1.5×: Moderately effective
- Speedup < 1.1×: Not effective

## Common Use Cases

### 1. Test Prefetch Distance

```bash
# Create variants with different prefetch distances
# distance = 1 K-group
prfm 0x10100
ld1 0x10000 64

# distance = 2 K-groups
prfm 0x10200
ld1 0x10000 64

# distance = 4 K-groups
prfm 0x10400
ld1 0x10000 64
```

### 2. Compare Memory Layouts

```bash
# Row-major access (good locality)
ld1 0x10000 64
ld1 0x10040 64
ld1 0x10080 64

# Column-major access (poor locality)
ld1 0x10000 8
ld1 0x11000 8
ld1 0x12000 8
```

### 3. Find Optimal Block Size

```bash
# Small blocks (fits in L1, good locality)
# ... load 2 KB ...
# ... compute ...

# Large blocks (exceeds L1, may thrash)
# ... load 128 KB ...
# ... compute ...
```

## Interpreting Results

### Example: Good Cache Behavior
```
L1 hit rate:  90.0%
Memory stalls: 20.0%
CPI: 5.0
→ Good! Working set fits in L1
```

### Example: Poor Cache Behavior
```
L1 hit rate:  30.0%
Memory stalls: 80.0%
CPI: 100.0
→ Bad! Working set too large, add prefetch or reduce size
```

### Example: Effective Prefetch
```
Without prefetch:
  Cycles: 10000
  L1 hit rate: 60%

With prefetch:
  Cycles: 7000 (1.4× faster)
  L1 hit rate: 75%
→ Prefetch is working!
```

## Tips for Writing Patterns

1. **Use realistic addresses**: Start at 0x10000 or similar
2. **Model actual access patterns**: Match your kernel's memory layout
3. **Include computation**: Add `sdot` between loads
4. **Test multiple iterations**: Repeat pattern to see hit rate improve
5. **Try prefetch variants**: Compare with/without prefetch

## Files in This Directory

- `memsim` - Main simulator executable
- `memsim.h`, `memsim.c` - Simulator implementation
- `memsim_main.c` - CLI program
- `example_*.txt` - Example input files
- `README.md` - Full documentation
- `RESULTS_SUMMARY.md` - Validation results
- `QUICKSTART.md` - This guide
- `Makefile` - Build system

## Next Steps

1. **Run examples**: `make test`
2. **Read full docs**: See `README.md`
3. **Create your pattern**: Model your kernel's access pattern
4. **Experiment**: Try different optimizations
5. **Validate**: Compare with real hardware results

## Getting Help

See `README.md` for:
- Detailed usage instructions
- Cache configuration details
- Interpretation guidelines
- Simulator limitations

## Quick Reference

**Build**: `make`
**Test**: `make test`
**Run**: `./memsim <file>`
**Clean**: `make clean`
**Help**: `./memsim` (no args shows usage)

**A64FX Cache Specs** (accurate):
- L1: 64 KiB, 4-way, **256-byte lines**, 11 cycles
- L2: **7 MiB**, **14-way**, **256-byte lines**, 40 cycles
- HBM2: **260 cycles** (256 GB/s ld, 128 GB/s st)

**Good L1 hit rate**: > 85%
**Good memory stalls**: < 30%
**Effective prefetch**: 1.2-3× speedup
