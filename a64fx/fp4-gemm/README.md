# A64FX SVE FP4 GEMM

This directory implements weight-only MXFP4 and NVFP4 GEMM for A64FX SVE 1.0.
The kernel computes `C[M,N] = A_fp16[M,K] * W_fp4[N,K]^T`, expands weights to
FP16 in registers, uses SVE FP16 FMA/FP16 accumulators, and returns FP32 output.

Supported weight formats:

- `mxfp4`: E2M1 values, one E8M0 scale per 32 consecutive K values.
- `nvfp4-1d`: E2M1 values, one unsigned E4M3 scale per 16 K values, and one
  FP32 tensor scale.
- `nvfp4-2d`: E2M1 values, one unsigned E4M3 scale per 16x16 weight tile, and
  one FP32 tensor scale.

`promotion_k` controls the accuracy tradeoff. Values 32, 64, 128, and 256
convert FP16 vector accumulators to an FP32 shadow at that K interval. Zero
keeps FP16 accumulators across the full K dimension.

## Build and unit test

```sh
make -C a64fx/fp4-gemm
make -C a64fx/fp4-gemm test
```

The implementation requires `N` and `K` divisible by 32. Arbitrary `M` is
handled in groups of at most six activation rows.

## Stage DeepSeek-V4-Flash data

The staging tool reads safetensors headers, selects layer-0 attention and all
256 routed experts, then copies payloads in bounded chunks. It never reads a
whole shard into memory and evicts each source range from page cache.

```sh
python3 a64fx/fp4-gemm/stage_ds4f_fp4.py --plan-only
python3 a64fx/fp4-gemm/stage_ds4f_fp4.py
```

The current checkpoint selection is 1,548 tensors and 3.287 GiB. The benchmark
loads this raw subset into anonymous HBM, retaining at least 12 GiB of
`MemAvailable`. Only one transient dequantized/converted matrix is allocated at
a time, so peak use remains far below the 16 GiB task limit.

Native expert MXFP4 is copied bit-for-bit into the runtime matrix. NVFP4 expert
variants are requantized from that native MXFP4 value set. Attention weights
are dequantized from the checkpoint's FP8 E4M3 plus 128x128 E8M0 scales and
quantized to each FP4 format.

## Benchmark

```sh
cd a64fx/fp4-gemm
# Focused single-core output-vectorized kernel
OMP_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores ./bench_fp4_single

# Real checkpoint shapes/data, using the N32 compute layout
OMP_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./bench_fp4_gemm --kernel n32 --threads 1 --experts 1 --full

# Original output-row kernel, for multi-core comparison
OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./bench_fp4_gemm --threads 48 --experts 256 --reps 1

# Full attention M sweep without repeating experts
OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./bench_fp4_gemm --threads 48 --experts 0 --full --skip-experts

# Full W1/W2/W3 M sweep for expert zero
OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./bench_fp4_gemm --threads 48 --experts 1 --full --skip-attention
```

Reported relative L2 error compares the SVE result with FP32 accumulation of
the same FP16 activation and FP16-rounded dequantized FP4 weights. Thus it
isolates accumulation error from format conversion error.

The `n32` path repacks 32 output channels together. Each K step expands 32
E2M1 values with SVE table lookup, applies a predecoded FP16 scale vector, then
reuses that vector across a six-row activation microtile. Scale decoding and
repacking are setup costs and are outside timed GEMM execution.

See [RESULTS.md](RESULTS.md) for measured A64FX results.
