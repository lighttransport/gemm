# A64FX SVE SWFP4/FP8 skinny GEMM

This directory is a self-contained A64FX SVE1 kernel lab derived from the
dataflow and weight contracts in `ref/v100-skinny`.  It keeps quantized weights
compressed in HBM, accumulates in FP32, and provides an FP16 boundary adapter.
It does not alter or depend on a model runtime.

## Formats

The public API in `swfp4fp8.h` supports:

- `NVFP4_G16`: E2M1 nibbles, E4M3 scale per K=16, and a global FP32 scale.
- `QPN8_TILE32`: E4M3 bytes and one FP32 scale per N=32 tile.  The packer folds
  the V100 decoder's `/256` compensation into that scale.
- `MXFP4_G32`: OCP E2M1 nibbles with E8M0 scale per output row/K=32 block.
- `FP8_BLOCK128`: E4M3 bytes with E8M0 scale per 128x128 block.

The NVFP4 and QPN8 packers accept either canonical row-major codes or the exact
V100 `[N/32][K/16][lane32][8 or 16 bytes]` QPN permutation.  Packing changes
only byte/nibble position.  It never requantizes an exact-path weight.

The optimized representation is `[N/16 panel][K][16 SVE lanes]`.  FP4 retains
two codes per byte.  A canonical compressed copy is also retained by this lab
to benchmark the row-vectorized alternative; it is not an expanded-weight
cache.  `swfp4fp8_matrix_bytes()` reports bytes streamed by one selected
compressed representation, not the sum of the two experimental layouts.

## Kernels

- `auto`/`panel`: N=16 output panel in SVE lanes, specialized for M=1/2/4/8.
  Larger M is composed from spill-free 8-row blocks.  FCC spills the 16-row
  form badly, so it is deliberately not dispatched.
- `row`: K-vectorized eight-output baseline retained for comparison.
- `fp8-ftz`: explicitly lossy arithmetic E4M3 decoder.  It flushes subnormals
  and maps NaN encodings to finite values.
- `fp4-sdot`: explicitly lossy FP4 path.  E2M1 weights are represented exactly
  as doubled integers; activations are dynamically quantized to int8 per
  K=16/K=32 scale group and consumed with SVE `sdot`.

`auto` never selects a lossy kernel.  Current A64FX measurements select the
panel path; the row override remains useful for experiments.

## Build and run

```sh
make -C a64fx/swfp4fp8 CC=fcc
make -C a64fx/swfp4fp8 test CC=fcc
./a64fx/swfp4fp8/bench_swfp4fp8 --quick --threads 48
./a64fx/swfp4fp8/bench_swfp4fp8 --scaling --threads 12
./a64fx/swfp4fp8/bench_swfp4fp8 --full --threads 48
```

The Makefile uses FCC's native SVE target and routes compiler temporary files
to `a64fx/swfp4fp8/tmp`; it does not require `/tmp`.  The benchmark binds
workers to compute cores 12-59.  Both packed weights and the 384 MiB read
ceiling use fresh anonymous mappings first-touched by the eventual owner, which
avoids the A64FX recycled-heap one-CMG placement trap.

`--quick` covers three representative shapes. `--full` adds the V100 trunk,
lm-head, and A64FX DS4F shapes. `--scaling` limits execution to the
`N=5120,K=1536,M=1` shape and includes exact panel/row and lossy controls.

See `RESULTS.md` for measurements from the implementation node.
See `FORMAT_GUIDE.md` for the format-to-SVE instruction analysis and the
recommended MXFP4/QPN8 mixed-format policy.
