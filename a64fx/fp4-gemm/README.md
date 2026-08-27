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

# Scalar base-ISA arithmetic microbenchmark (no SIMD/SVE/FP/TBL)
OMP_NUM_THREADS=1 OMP_PROC_BIND=close OMP_PLACES=cores \
  ./a64fx/fp4-gemm/bench_fp4_ex

# 72 MiB cold-sized FP4 stream, local to one HBM NUMA domain
numactl --physcpubind=12 --membind=4 ./a64fx/fp4-gemm/bench_fp4_stream

# fapp region contains only warmed, repeated fused M=1 calls.
make -C a64fx/fp4-gemm fapp
cd a64fx/fp4-gemm
OMP_PROC_BIND=close OMP_PLACES=cores \
  numactl --physcpubind=12-23 --membind=4 \
  fapp -C -d fapp_fp4_pa1 -Icpupa -Hevent=pa1 \
  ./bench_fp4_fapp 32768 4096 256 400 12
```

The implementation requires `N` and `K` divisible by 32. Arbitrary `M` is
supported. The direct packed path uses groups of at most six activation rows;
the persistent decoded-weight path uses padded 12-row by 64-column tiles.

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

`fp4_gemm_f16_l1` is an experimental assembly path testing A64FX pipeline
separation. Its producer uses only scalar AArch64 GPR/load/store instructions
and a 256-entry packed-byte LUT; its consumer uses SVE FP16 multiply/FMA. Two
alternating 64-byte L1 buffers carry decoded weights between them. It is not
the default because the scalar producer is slower than direct SVE dequant.

`fp4_gemm_f16_l1panel` uses one promotion-sized panel: K=256 occupies 16 KiB.
It ties the full L2 panel at small M but loses at large M because repeated block
transitions and partial-output traffic outweigh the smaller workspace.

`fp4_gemm_f16_l2_omp` is the fused CMG baseline. Each worker owns a private
256 KiB panel and statically owns disjoint N32 tiles, so all 12 cores decode and
compute concurrently without queues, locks, duplicate decoding, or output
sharing. Bind both CPUs and memory to one CMG when benchmarking it.

`fp4_gemm_f16_n32_omp` contains the optimized decode/matvec path. For M=1 it
processes four N32 tiles concurrently, broadcasts each activation once, and
uses four vector FMLAs instead of indexed scalar FMLAs. The MXFP4 fast path is
hand-scheduled assembly and pipelines two K steps so the second set of decode
operations covers part of the six-cycle `ZIP1`/`TBL` latency. Recommended CMG
dispatch is direct N32 for M<=6 and the decoded L2 panel for M>=12.

`fp4_matrix_prepare_u8` plus `fp4_gemm_f16_u8tbl_omp` is an opt-in experiment
that stores one nibble per byte. Groups of four N32 tiles are interleaved at
each K step to form a sequential A64FX stream. The hand-scheduled MXFP4 kernel
pipelines two K steps and needs only `LD1B`, `TBL`, scale multiply, and FMA for
each weight vector. It is the fastest M=1 path, but doubles code storage and
HBM traffic; normal preparation does not allocate this sidecar.

`fp4_matrix_prepare_bitplane` plus `fp4_gemm_f16_bitplane_omp` is a same-density
M=1 experiment. It transposes each N32 group into four 32-bit bitplanes. Its
inner loop uses broadcast loads, shifts, masks, and a magic `0x3800` multiply
to construct FP16 bits, with no per-K `ZIP1` or `TBL`. It remains opt-in because
the added FL arithmetic is slower than the packed decoder.

`fp4_gemm_f16_l2` instead dequantizes one packed `N32 x K` weight panel into a
K-major FP16 workspace and reuses it for every activation row. At K=4096 the
workspace is 256 KiB, so it resides in L2 rather than the 64 KiB L1. Use direct
N32 FP4 for M<12 and the L2 panel path for M>=12. GEMM timing includes panel
dequantization and its L2 write/read traffic.

`fp4_matrix_prepare_bf16` (legacy API name) builds a persistent K-major FP16
sidecar arranged as `[N/64][K][64]`. `fp4_gemm_f16_bf16cache_omp` packs each
12-row activation tile as `[K][12]` and invokes a hand-scheduled 12x64 SVE
microkernel with 24 FP16 accumulators. K-block boundaries convert to FP32 and
accumulate into a contiguous 3 KiB FP32 shadow tile; the final tile is
scattered to row-major C only once. This avoids L1 set conflicts when ldc is a
large power-of-two stride. Activation packing and GEMM share one OpenMP region
to avoid a second fork/join. This is preferred when a matrix is reused for
multiple M>1 GEMMs: it costs 4x the packed weight storage but removes FP4
decode from the timed repeated GEMM. Activation packing remains inside the
reported kernel time. Padded M and N tails use a private output tile, so the
public requirement remains only that N and K are divisible by 32. The sidecar
is validated against the FP4 reference for MXFP4 and both NVFP4 layouts.

On A64FX the cache path tags packed activations for strong L1 reuse and tags
the decoded-weight stream to bypass L1. This is the default because each
weight vector is consumed by the register tile and then discarded, while the
activation K block must survive the stream. Set `FP4_NO_SECTOR_TAGS=1` for the
untagged control. `FP4_A_TAG` and `FP4_B_TAG` override the raw tag values for
experiments; the selected defaults are `0x9` and `0xb`. An MR8x64 assembly
tail avoids executing four padded rows when the final M block has at most
eight live rows.

## INT8 SDOT re-encoding

`fp4_matrix_prepare_sdot` maps E2M1 exactly to signed integers by multiplying
the value by two (`0, +/-1, +/-2, +/-3, +/-4, +/-6, +/-8, +/-12`) and packs four
adjacent K values per SVE lane. `fp4_gemv_i8_sdot_omp` uses INT8 activations,
INT32 SDOT accumulation, and converts/rescales to FP32 only at the activation
group boundary. Per-output MXFP4/NVFP4 scales remain FP16 and are unpacked
after each integer dot group. The representation uses one byte per weight,
the same capacity as the byte-expanded table path, and introduces no weight
requantization error.

For a 128 MiB expanded MXFP4 stream on one 12-core CMG:

| Activation K group | GFLOP/s | Source GB/s | DeepSeek expert worst rel-L2 |
|---:|---:|---:|---:|
| 32 | 419.8 | 223.0 | 1.122% |
| 16 | 417.5 | 221.8 | 0.799% |
| 8 | 416.8 | 221.4 | 0.570% |
| 4 | **409.1** | **217.3** | **0.394%** |

K4 is the accuracy-qualified default because it remains above 400 GFLOP/s and
passes the 0.5% gate over W1/W2/W3 from DeepSeek layer 0, with minimum cosine
similarity 0.999992. Set the fifth `bench_fp4_stream` argument to select another
group; `FP4_SDOT_ONLY=1` skips unrelated kernels. `bench_fp4_sdot_error`
validates the three native expert projections directly from the staged raw
file without loading the full 3.3 GiB subset.

## Packed pair-LUT experiment

`fp4_matrix_prepare_pair` keeps the weight stream at four bits per element and
orders it as `[weight block][N128][K pair][8 SVE vectors]`. For each activation
pair, `fp4_pair_activation_prepare` builds a 256-entry INT16 table containing
the exact two-term E2M1 dot product. `fp4_gemv_pair_lut_omp` uses eight
independent indexed-load/INT32-add chains, followed by FP32 scale accumulation.
It is numerically identical to the expanded SDOT path for the same activation
quantizer (relative difference below 5e-8 in the unit test).

The indexed-load path does not reach the packed-stream roof on A64FX. With
N=32768, K=4096 on one 12-core CMG it measures 177--179 GFLOP/s and about
50 GB/s of packed source traffic for activation groups K32, K8, and K4. The
flat result across promotion intervals identifies `LD1SH` gather throughput,
not FP32 conversion or HBM2, as the bottleneck. Set `FP4_PAIR_ONLY=1` to run
this control. The expanded K4 SDOT kernel remains the production fallback at
409 GFLOP/s because its sequential byte stream reaches roughly 217 GB/s.
The alternative `fp4_gemv_pair_tbl_omp` replaces gathers with two 16-entry
tables per packed K pair. It is also exact, but doubles the table-operation
count needed to cover 128 rows and reaches only 84.5 GFLOP/s at K4; it is a
pipeline diagnostic rather than a selected kernel.

The selected packed MXFP4 M1 path additionally interleaves eight N32 tiles at
each K step. It prepares one shared 32-lane activation-scaled E2M1 table per K,
so the 12 cores only load the finished table instead of redundantly scaling it
inside every output kernel. Eight FP16 accumulator and row-scale vectors stay
resident while two four-vector decode batches stream sequentially. This raises
the N=32768, K=4096, K256-promotion result from roughly 280 to **400 GFLOP/s**
(112.5 GB/s packed source) on one 12-core CMG.

Two variants were rejected: applying row scales only at K32 boundaries costs
extra FP16 workspace traffic and reaches 383.5 GFLOP/s; scalar-offset SVE loads
in place of sequential pointer updates constrain address generation and reach
383.7 GFLOP/s.

`fp4_gemv_i8_sdot4_omp` is an exact-density packed K4 SDOT experiment. Each
64-byte load contains 32 rows by four K values; byte `TBL`, `ZIP1/ZIP2`, and
two SDOTs produce 32 INT32 outputs. Four chunks are kept in flight and FP32
promotion is transposed across eight independent vectors. It matches expanded
K4 SDOT within 1e-7 relative L2, but A64FX byte-table/permutation execution
limits it to 83.5 GFLOP/s (about 26 GB/s including its FP32 scale sidecar).
It is retained as a negative control, not selected over the 400 GFLOP/s t8
FP16 path or 410 GFLOP/s expanded SDOT path.

Three further exact M1 controls attack the measured FLA completion stalls.
`fp4_gemm_f16_half_omp` pairs rows 0--15 with rows 16--31 and uses two
overlapping half-predicate loads plus `ORR` instead of vector `ZIP1`.
`fp4_gemm_f16_t12_omp` widens the original decoder to twelve N32 tiles.
`fp4_gemm_f16_affine_omp` re-encodes E2M1 so FP16 bits can be constructed with
shift/add/predicate operations and removes `TBL`. Set `FP4_HALF_ONLY=1`,
`FP4_T12_ONLY=1`, or `FP4_AFFINE_ONLY=1` on `bench_fp4_stream` to isolate them.
They are diagnostics rather than default dispatch paths.

### Scalar EX packed-FP4 producer

`bench_fp4_ex` also measures a packed-FP4 producer using only scalar AArch64
integer/load/store instructions. Its input is reordered offline so eight low
nibbles and eight high nibbles can be emitted without a permutation. The
nibble-only upper bound needs one load, two masks/shifts, and two stores; the
exact path additionally maps E2M1 to signed SDOT coefficients with SWAR bit
operations. On 12 CMG cores with a 64 MiB packed stream:

| Scalar producer | Packed GB/s | Expanded GB/s | Maximum FP4 GFLOP/s |
|---|---:|---:|---:|
| Nibble split only | 40.6 | 81.1 | 162.3 |
| Exact E2M1 signed bytes | 6.06 | 12.1 | 24.2 |

The exact L1-resident single-core producer reaches only 1.01 billion weights/s.
Consequently scalar EX production cannot feed either the 409 GFLOP/s expanded
SDOT kernel or an 800 GFLOP/s packed-stream target, even with perfect EX/FL
overlap. The fused double-buffer design was rejected at this producer gate.

See [RESULTS.md](RESULTS.md) for measured A64FX results.
