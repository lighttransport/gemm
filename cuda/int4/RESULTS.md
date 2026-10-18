# RTX 5060 Ti E0M3 results

Measured 2026-10-18 on NVIDIA GeForce RTX 5060 Ti, SM120, 16 GB, driver
615.71.09. Compiler: CUDA 13.2.86, `-O3 -std=c++17 -arch=sm_120a -lineinfo`.
cuBLAS: 13.4.1. CUBIN verification used CUDA 13.1.115 `cuobjdump`, which names
E0M3; CUDA 13.2's disassembler does not. Clocks were not locked; the recorded
initial SM clock was 2070 MHz, with a reported maximum of 3105 MHz.

These are measurements of this implementation and timing loop. Instruction
throughput is distinct from matrix throughput; neither is a hardware peak claim.

The `E2M1` columns below are FP4 GEMM measurements. A follow-up comparison
directly reuses the original kernels from `cuda/fp4`, with fixed tiling for the
new FP4/E0M3 pair: [FP4 comparison results](../fp4/RESULTS.md).
The [FP8 comparison](../fp8/RESULTS.md) also measures corrected existing FP8
kernels and cuBLAS E4M3 (135.01 TOp/s at 4096³, 141.84 TOp/s at 8192³).
The corrected [cuBLAS NVFP4 follow-up](../fp4/RESULTS.md) measures 236.82
TOp/s with FP32 output at 8192³, establishing further optimization headroom
for the custom FP4/E0M3 kernels.

## Optimized E0M3 at 8192 cubed: 280 TOp/s target reached

Measured **2026-10-19**, on the same 36-SM RTX 5060 Ti, driver 615.71.09,
CUDA 13.2.86, and cuBLASLt 13.4.1. GPU clocks and the configured 150 W power
limit were unchanged; before/after clocks in the benchmark metadata were
approximately **2070 MHz**. The new optional path reuses the tuned FP4
CUTLASS **256x128x128** tile, three mainloop stages, CLC scheduling,
AlongN raster order, and swizzle 4.

Both rows below use packed inputs, **FP32 accumulation and BF16 output**,
varying signed nibble codes, and varying positive K16 UE4M3 scales. Dense
throughput is `2*M*N*K/time`, with no sparsity multiplier.

| Format / same tuned kernel | M,N,K | Median ms | Min-max ms | Dense TOp/s |
|---|---|---:|---:|---:|
| **E0M3 x E0M3** | 8192,8192,8192 | **3.915753** | 3.915477-3.920933 | **280.79** |
| E2M1 x E2M1 FP4 control | 8192,8192,8192 | 3.916435 | 3.916085-3.918182 | 280.74 |

The throughput is effectively equal, and the **280 TOp/s E0M3 target is
met**. An initial independent run measured 280.90 TOp/s. The earlier 55.01
TOp/s E0M3 result below uses the original FP32-output kernel; the new headline
uses BF16 output, matching the preceding FP4 tuning target. It is not a
precision-matched speedup comparison with that older kernel.

```sh
make -C cuda/int4 cutlass-test
make -C cuda/int4 cutlass-sanitize
make -C cuda/int4 cutlass-bench
make -C cuda/int4 cutlass-control
```

`CUTLASS_DIR=/absolute/path/to/cutlass` overrides the default local checkout,
`tmp/flash-attention-src/csrc/cutlass`. The tested CUTLASS version was 4.6.1,
commit `e05f953a5b3d38adc240df2ff928e0421c2abba3`. Host/device adapter source is
shared with `cuda/fp4/cutlass_fp4.cu`; input generation, references, timing and
CSV reporting reuse `cuda/fp4/cublas_fp4_gemm.c` with `--format e0m3`.

The baseline shared object is built with `-O3 -DNDEBUG`, binary-only
`--generate-code=arch=compute_120a,code=sm_120a`, and `--compress-mode none`.
`patch_cutlass.py` parses its fatbinary entries and changes **only 128 bytes**:
the A/B format bits of **128 OMMA instructions**, including mainloop and tail
copies. Each byte changes by XOR `0xc0`, selecting `E0M3.E0M3`. All other host
ELF, device ELF, layout, scheduling and instruction bits remain identical.
Individual CUDA ELFs and the final shared object are disassembled with CUDA
13.1.115 `cuobjdump`. There is **no PTX fallback** that could silently execute
unpatched FP4. The external CUTLASS checkout was left untouched.

The optimized adapter uses the FP4 path's **128x4 scale layout**. The earlier
`int4_matrix` API uses linear scale rows and FP32 output and remains unchanged.
Input packing, scale swizzling, transfers, decoded-reference preparation,
algorithm setup and validation are excluded from timing. There is no
dequantization in the timed E0M3 kernel. Five warmups precede five rounds of
100 CUDA-event-timed launches; each candidate is checked before timing, and a
fresh launch is checked afterward.

Correctness and validation:

- Every one of the **67,108,864** E0M3 outputs is bit-identical to a cuBLAS
  **BF16-input** GEMM on exactly decoded E0M3 values, with FP32 accumulation
  and BF16 output. This reference is outside timing and is not a native
  cuBLAS E0M3 implementation. The decoded inputs are asserted to be exactly
  representable in BF16 before upload.
- Sixty-four FP64 CPU dot products independently passed. Relative L2 against
  their unrounded values was 0.00166253, reflecting BF16 output rounding.
  Output is filled with NaNs before verification launches.
- **13 new tests passed:** six container/format-patch integrity tests and seven
  GPU/CLI tests. These cover all 256 signed-nibble pairs with varying scales,
  asymmetric 128x256x128 and tail 384x128x320 shapes, unit scales, FP4 controls,
  and rejection of both an unpatched kernel and unverified E0M3 execution.
- Compute Sanitizer reported **zero memory errors** and **zero race hazards**
  at 384x128x320. The existing numeric/raw-patcher CPU tests also passed.
- Shared-runner regressions passed with
  `make -C cuda/fp4 cublas-test cutlass-test`; their log is
  `build/cutlass_shared_runner_regression.log` in this directory.

Artifacts are `build/cutlass_e0m3_8k.csv`,
`build/cutlass_fp4_control_8k.csv`, their `.metadata.txt` sidecars,
`build/libcutlass_e0m3.patch.json` (binary hashes and tool version), and
`build/libcutlass_e0m3.sass`. Tests and sanitizer output are in
`build/cutlass_test.log`, `build/cutlass_validation_bench.log`, and
`build/cutlass_cpu_regression.log`. Sanitizer timing is excluded from the table.

## Further tuning and GLM5.3-Flash decode (2026-10-19)

The **290 TOp/s target at 8192 cubed remains unreached**. Larger epilogue
tiles measured 270.75 TOp/s (64x64) and 271.26 TOp/s (128x64). Moving the
last register MMA before the next TMA-stage wait measured 278.99 TOp/s,
with every output bit-identical to native cuBLAS NVFP4. These experiments
used release builds, 256x128x128 mainloop tiles, FP32 accumulation, BF16
output, AlongN raster order, and swizzle 4. Epilogue trials used 30 launches
x 3 rounds; the MMA scheduling trial used 50 x 5. Their scratch headers
never modified the external CUTLASS checkout. A two-CTA register-budget
experiment stalled, was terminated, and its override was discarded. The
original approximately 281 TOp/s bulk kernel remains selected.

The final paired repeat at **2070 MHz** measured **280.85 TOp/s E0M3**
(3.914960 ms) and **280.81 TOp/s FP4** (3.915556 ms), with five rounds of
100 launches. Each format's 67,108,864 outputs matched its cuBLAS reference
bit for bit. Evidence is `build/further_e0m3_8k.csv` and
`build/further_fp4_8k.csv`, with metadata and full-comparison logs.

The new decode adapter tunes **60 candidates**: 11 shared-memory tile
shapes with asynchronous staging, FP32 global split-K, direct-register MMA,
reduction within a block, and an E0M3 DP4A GEMV. Shared rows have 16 bytes of
padding to avoid bank conflicts. The adapter needs no CUTLASS headers and
returns BF16. Its binary-only build contains **308 format-patched OMMA
instructions**; the manifest verifies that only those 308 bytes changed.
The unpatched library reuses these MMA kernels for FP4 and excludes the
E0M3-only DP4A candidate when that format is selected.

Shapes come from Strata's GLM5.3-Flash tensor schema in
`include/strata/artifact/glm_native.hpp`, including hidden=4096, expert=2048,
dense FFN=12288, KDA=8192, MLA ranks 1536/512 and expanded width 16384,
and vocabulary=154880. Inputs are synthetic signed nibbles with varying
positive K16 UE4M3 scales. This is resident packed W4A4 GEMM, without model
loading, quantization, routing, weight transfers, SwiGLU, or model-quality
validation. For routed experts, M counts rows assigned to one expert.

Timings use graph replay, 5 warmups, **100 launches x 5 rounds**, and GPU
events. Warm mode repeats the same operands. Evicted mode writes **64 MiB
before each GEMM**, with explicit graph event nodes timing only the GEMM
and any reduction. Eviction can create memory writeback pressure; this is
a synthetic cache experiment rather than a captured model decode trace.
Packing, allocation, graph creation/upload, and validation are excluded.
Tuning uses three pilots of 32 launches for core projections. The vocabulary
M=64 decode/cold cases use four-launch pilots; its earlier cases used 32.
Reproduction uses four for the vocabulary projection throughout, keeping
100 x 5 final launches for every case. Core-sweep before/after SM-clock samples
ranged from **2040 to 2077 MHz**, near the requested 2.07 GHz; observed memory
clock was **7001 MHz**, versus reported maximum 14001 MHz. Clock settings
and the configured 150 W power limit were unchanged.

Single-token latency, in microseconds. Bulk is the previous 256x128x128
BF16-output winner; comparisons share packed data, scale layout, output
precision, timing, and independent references. Warm does not imply that
large operands fit in L2.

| Projection | K -> N | Warm bulk | Warm decode | Evicted bulk | Evicted decode |
|---|---|---:|---:|---:|---:|
| expert_gate_up | 4096 -> 2048 | 38.11 | 7.65 | 52.16 | 32.50 |
| expert_gate_up_fused | 4096 -> 4096 | 39.24 | 12.62 | 64.61 | 56.77 |
| expert_down | 2048 -> 4096 | 23.02 | 7.57 | 41.13 | 32.57 |
| dense_gate_up | 4096 -> 12288 | 110.13 | 31.47 | 160.75 | 150.31 |
| dense_down | 12288 -> 4096 | 104.16 | 32.26 | 172.53 | 155.86 |
| kda_qkv_projection | 4096 -> 8192 | 75.52 | 21.34 | 112.32 | 104.18 |
| kda_output | 8192 -> 4096 | 71.61 | 22.54 | 112.95 | 103.28 |
| mla_q_a | 4096 -> 1536 | 38.04 | 6.56 | 51.81 | 25.83 |
| mla_q_b | 1536 -> 16384 | 65.08 | 17.13 | 95.50 | 84.22 |
| mla_kv_a | 4096 -> 512 | 37.56 | 3.98 | 46.11 | 13.97 |
| mla_output | 16384 -> 4096 | 186.64 | 173.65 | 209.38 | 194.61 |
| indexer_q_b | 1536 -> 4096 | 18.99 | 6.00 | 34.31 | 24.95 |
| lm_head | 4096 -> 154880 | 1681.77 | 1665.26 | 1783.88 | 1745.73 |

MLA KV-A improves **9.43x warm / 3.30x evicted** at M=1. Expert gate/up
improves **4.98x / 1.60x**. Summing two single expert projection latencies
gives 15.31 us warm, versus **12.62 us** for the concatenated gate+up matrix
(about 17.5% lower). The analogous evicted comparison is 65.00 versus
56.77 us. These compare GEMM latencies and exclude the SwiGLU operation.
The 154880-wide output matrix occupies about 357 MB including scales;
its M=1 decode timing corresponds to roughly **214 GB/s** of weight and
scale traffic. It shows little improvement from changing the compute tile.
The unpatched FP4 control on the expert 4096->2048, M=1 shape measured
**7.62 us warm / 31.00 us evicted**, using the corresponding E0M3-selected
tiles; all 2048 outputs matched native cuBLAS NVFP4 exactly.

Larger M, with cold-cache fallback applied:

| Projection | M | Warm decode us | Warm TOp/s | Warm speedup over bulk | Evicted selected us | Evicted library |
|---|---:|---:|---:|---:|---:|---|
| expert_gate_up | 16 | 8.57 | 31.33 | 4.45x | 34.09 | decode |
| expert_gate_up | 64 | 16.66 | 64.47 | 2.29x | 41.68 | decode |
| expert_gate_up_fused | 16 | 14.08 | 38.13 | 2.76x | 60.09 | decode |
| expert_gate_up_fused | 64 | 27.08 | 79.31 | 1.44x | 66.04 | bulk |
| expert_down | 16 | 7.94 | 33.80 | 2.87x | 35.90 | decode |
| expert_down | 64 | 16.04 | 66.96 | 1.40x | 42.19 | bulk |

Cold M=64 sometimes regresses: expert down is 44.29 us with the small-M
adapter versus 42.19 us with bulk. `dispatch.csv` therefore retains bulk
unless decode is at least 2% faster. It records library and candidate index
for each shape, M, and cache mode; the raw summary preserves both timings.
The table is an offline selection aid, not a Strata runtime integration.

Reproduce from the repository root:

```sh
make -C cuda/int4 decode-test decode-sanitize
make -C cuda/int4 decode-bench
# Resume an interrupted default sweep without discarding completed rows:
python3 cuda/int4/bench_decode.py --resume
```

All **156 final rows** (13 projections x three M values x two cache modes
x two libraries) passed, with every output bit-identical to decoded-BF16
cuBLAS and 64 independent FP64 CPU samples per row. The smaller unit cases
check every output of every supported candidate, all 256 nibble-code pairs,
M/N/K tails, FP4 reuse, graph dispatch, and both reduction strategies.
Nine decode tests, 13 bulk regression tests, numeric/raw-patcher CPU checks,
and native cuBLAS tests passed. Compute Sanitizer at 17x40x576 reported
**zero memory errors and zero race hazards**, including all split-K factors
and empty tail partitions. The DP4A single-token path also passed memcheck.

Raw core-projection evidence is in `build/glm_decode/`, vocabulary evidence
in `build/glm_head/`, with per-candidate logs, CSV metadata, binary hashes,
and full reference comparisons. The default reproduction command writes a
combined sweep to `build/decode/`. Generated binaries and logs remain local.

## Commands and correctness

From the repository root:

```sh
make -C cuda/int4 all test-cpu test-gpu -j2
make -C cuda/int4 sanitize
cd cuda/int4
./build/bench_int4 --shape all --mode all --csv build/results.csv
./build/bench_int4 --shape all --scales unit --mode packed --csv build/unit.csv
./build/bench_int4 --mode probe --csv build/probe.csv
```

All runs used defaults: five warmups, 100 launches per timed round, five rounds,
and rotated measurement order. Each matrix shape independently selected the
fastest validated variant from the one-warp reference and eight tiled kernels.
Reported times are round medians. Allocation, transfer, tuning, validation, and
compilation are excluded; activation/full modes include their device quantizers.

Validation passed:

- CPU codebook, all finite positive scale-code round trips and adjacent-code
  midpoint ties, integer rounding, padding, zero blocks, and input checks.
- Five patcher tests: exact changed bits/addresses and rejection of corrupted,
  duplicated, already-patched, or unknown instructions.
- CPU/GPU quantizer byte agreement, including clipping counters and K tails.
- Every output of nine GEMM variants in all four operand-format combinations,
  including all 256 pairs of nibble codes, asymmetric/zero/subnormal/maximum scales, odd M/N, multiple
  K tiles, output guards, and nondefault CUDA streams.
- Compute Sanitizer: `ERROR SUMMARY: 0 errors`; racecheck: `0 errors, 0 warnings`.
- All 84 scaled benchmark rows and 48 unit-scale rows passed the 64-output FP64
  reference check. Arithmetic relative L2 error was zero for these synthetic
  inputs. All outputs of all seven instruction-probe modes were checked.

The compiled baseline contains 427 OMMA instructions. Each copied CUBIN patched
and re-disassembled all 427; the original CUBIN was preserved. The generated
manifest retains exact instruction locations, tool version, and binary hashes.

## K16-scaled matrix and quantization throughput

Inputs are deterministic uniform FP32 `[-2,2]`. E0M3 and E2M1 independently
quantize the same inputs into their respective codebooks and K16 UE4M3 scales.
BF16 cuBLAS consumes decoded E0M3 values and accumulates into FP32, providing
a baseline for the same arithmetic as E0M3.

All throughput columns use dense `2MNK/time`, in TOp/s. Activation includes
quantizing A with B already packed; full includes quantizing both A and B.

| Shape | M | N | K | E0M3 packed | E2M1 packed | BF16 | E0M3 activation | E0M3 full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| square_1k | 1024 | 1024 | 1024 | 63.49 | 64.43 | 30.73 | 35.31 | 24.49 |
| square_2k | 2048 | 2048 | 2048 | 77.10 | 77.12 | 36.31 | 49.37 | 34.96 |
| square_4k | 4096 | 4096 | 4096 | 86.14 | 86.15 | 37.38 | 64.40 | 52.55 |
| square_8k | 8192 | 8192 | 8192 | 55.01 | 54.64 | 37.77 | 49.99 | 45.96 |
| mm0 | 1024 | 4608 | 4608 | 88.20 | 88.20 | 37.10 | 65.25 | 33.62 |
| mm2 | 1024 | 1152 | 4608 | 84.76 | 84.79 | 36.81 | 44.37 | 24.90 |
| qkv | 512 | 4096 | 4096 | 75.91 | 75.96 | 35.50 | 61.63 | 21.21 |
| ffn_up | 512 | 11008 | 4096 | 82.40 | 82.44 | 36.92 | 69.11 | 22.63 |
| ffn_down | 512 | 4096 | 11008 | 83.34 | 83.29 | 35.92 | 58.29 | 21.76 |
| small_m1 | 1 | 4096 | 4096 | 1.36 | 1.40 | 0.21 | 1.29 | 0.06 |
| small_m16 | 16 | 4096 | 4096 | 16.77 | 16.76 | 10.43 | 15.42 | 0.99 |
| small_m64 | 64 | 4096 | 4096 | 62.42 | 62.44 | 25.63 | 51.33 | 3.90 |

For `4096^3`, E0M3 packed time was **1.595502 ms**, activation plus GEMM
**2.134069 ms**, and full quantization plus GEMM **2.615244 ms**. BF16 took
**3.676699 ms**. The selected kernel was `gemm_a128_128`: BM=64, BN=128, BK=128,
eight warps, and double-buffered asynchronous staging. Packed E0M3 was 2.30×
the BF16 throughput on this shape; the full pipeline was 1.41×.

Quantization is material, especially when repeatedly quantizing large weights
for small M. Prepacking B gives substantially better inference-like latency.
The lower 8K-square result also remains visible; this first implementation
does not sustain its best throughput across every shape.

Quantization error is evaluated separately against original FP32 input products,
using the same 64 output samples. Across the twelve shapes, E0M3 quantization
relative L2 ranged from **0.0867–0.1224**, versus **0.1138–0.1852** for E2M1.
At `4096^3`, the values were **0.09919** and **0.16481**, respectively. These
figures describe the synthetic uniform distribution, without a model-quality
claim. Arithmetic correctness uses the decoded packed inputs instead.

## Unit-scale INT8 comparison

Inputs are integers `[-7,7]`. E0M3, BF16, and INT8 represent them exactly.
cuBLAS INT8 computes an INT32 output without K16 scaling. E2M1 uses nearest
unit-scale FP4 values and can have quantization error. These measurements are
separate from the scaled-input sweep above.

| Shape | E0M3 packed TOp/s | E2M1 packed | BF16 | cuBLAS INT8 |
|---|---:|---:|---:|---:|
| square_4k | 86.15 | 86.15 | 37.38 | 101.98 |
| square_8k | 55.01 | 54.64 | 37.78 | 143.35 |
| ffn_up | 82.39 | 67.44 | 36.93 | 130.26 |

The short tuning pilots select variants independently for each format. On the
unit-scale FFN-up case, E2M1 selected a slower sustained configuration than
E0M3. Force a shared `--variant` for a comparison with identical tiling; the
instruction-loop comparison below fixes the instruction schedule across formats.

cuBLAS INT8 is faster than the current E0M3 kernels on these shapes, despite
the higher E0M3 instruction throughput below. Operand storage, scale loading,
tiling, and the achieved execution schedule still matter. The FP4/E0M3 sweep
must not be interpreted as a tuned GEMM peak. Scaled INT8 configurations are
explicitly skipped because this baseline has no equivalent K16-scaled reduction.

## Register instruction throughput

4096 blocks, one warp per block, 512 loop iterations, and eight independent
accumulator tiles. Operands arrive at runtime and alternate sign each outer
iteration, preventing constant-folded legacy `s4` unpacking. Product signs stay
positive. Warmups, round rotation, and timed launch counts match the matrix runs.

| Instruction operands | Median ms | Dense effective TOp/s |
|---|---:|---:|
| E2M1 × E2M1 | 0.9239 | 297.52 |
| E0M3 × E2M1 | 0.9240 | 297.50 |
| E2M1 × E0M3 | 0.9240 | 297.50 |
| E0M3 × E0M3 | 0.9240 | 297.50 |
| BF16 | 1.8337 | 37.47 |
| INT8 | 0.9223 | 149.02 |
| Legacy PTX s4 | 17.5910 | 15.63 |

The four E0M3/E2M1 combinations have essentially equal throughput. This loop
includes loop control, sign changes, and output stores, but no matrix loading
or shared-memory GEMM work. Legacy `s4` is the compiler-lowered integer path;
it is distinct from format-patched E0M3 OMMA.

Raw CSVs, metadata, SASS dumps, hashes, build output, and sanitizer logs remain
local under the ignored `build/` directory. Reproduce them with the commands
above; `make clean` removes generated build files.
