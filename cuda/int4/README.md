# E0M3 near-INT4 GEMM on SM120

Experimental dense GEMM for the RTX 5060 Ti: signed-magnitude E0M3 inputs,
FP32 accumulation/output, and optional per-K16 UE4M3 scales. Includes an FP32
quantizer, nine GEMM variants, correctness tests, and reproducible benchmarks.
Measured results and validation commands are in [RESULTS.md](RESULTS.md).

**Optimized packed GEMM:** the optional CUTLASS adapter reaches **about
281 TOp/s at 8192³ and 2070 MHz**, with FP32 accumulation and BF16 output.
It reuses the tuned FP4 256x128x128 tile and patches only its E0M3 instruction
format bits. All 67,108,864 outputs matched cuBLAS BF16 on exactly decoded
E0M3 inputs. This meets the same 280 TOp/s target as the FP4 path.

The format bits were identified by
[revollllt/sm120-e0m3-mma](https://github.com/revollllt/sm120-e0m3-mma).
The kernels here adapt the fragment mapping and staging from this repository's
FP4 experiments. The original CUBIN kernels have no CUTLASS, Python package,
or model dependency; the optional optimized adapter requires CUTLASS headers.

## Optimized CUTLASS adapter

```sh
make -C cuda/int4 cutlass-test
make -C cuda/int4 cutlass-sanitize
make -C cuda/int4 cutlass-bench
make -C cuda/int4 cutlass-control  # same binary before patching, interpreted as FP4
```

The default `CUTLASS_DIR` is repository-local
`tmp/flash-attention-src/csrc/cutlass`; override it with an absolute checkout
path. Tested with CUTLASS 4.6.1 and CUDA 13.2. CUDA 13.1 `cuobjdump` is used
to verify the named E0M3 instructions. Only these explicit targets build the
optional adapter; the original `make all` and `int4_gemm` API are unchanged.

This path consumes row-major packed signed-magnitude `-7..7` codes and K16
UE4M3 scales in the **128x4 scale layout**, as in the tuned FP4 benchmark.
The earlier `int4_matrix` API uses linear scale rows and FP32 output; its scale
buffers cannot be passed directly to this adapter. Packing, scale swizzling,
reference decoding, and validation are outside the packed-GEMM timing.
No integer unpacking or GEMM on decoded values occurs in the timed E0M3 path.

`patch_cutlass.py` walks the host ELF's fatbinary entries, verifies every
embedded CUDA ELF, and changes only bits 14/15 in the high 64-bit word of
the designated OMMA instructions. The selected kernel contains 128 such
instructions. Both individual CUDA ELFs and the final shared object are
re-disassembled. PTX fallback, compressed CUDA ELFs, wrong architectures,
unknown OMMA encodings, and changes outside the designated format bytes are
rejected. `build/libcutlass_e0m3.patch.json` records binary hashes, tool version,
and instruction counts; `build/libcutlass_e0m3.sass` preserves the evidence.

The runner reuses `cuda/fp4/cublas_fp4_gemm.c` with `--format e0m3`; it always
checks decoded E0M3 references. `--compare-cublas` additionally decodes both
inputs exactly to BF16 outside timing and compares every output against a
cuBLAS BF16-input GEMM with FP32 accumulation. This is a correctness reference,
not cuBLAS support for the experimental E0M3 format. `--data codebook` at
16x16x64 checks all 256 signed-nibble pairs with varying scales. Negative
controls ensure an unpatched FP4 adapter fails E0M3 validation.

## GLM5.3-Flash decode projections

`decode.cu` supplies smaller tiles for M=1/16/64, with padded shared-memory
rows, asynchronous double buffering, and FP32 split-K reductions. It also
compares direct-register MMA, reduction within a block, and an E0M3-only DP4A
GEMV. The 60 candidates are selected using verified timing pilots. This
standalone adapter needs no CUTLASS headers; the bulk comparison does.

```sh
make -C cuda/int4 decode-test
make -C cuda/int4 decode-sanitize
make -C cuda/int4 decode-bench
# Shorter expert-only comparison, after building:
python3 cuda/int4/bench_decode.py --suite experts --output build/decode_experts
```

Shapes follow Strata's validated GLM5.3-Flash tensor schema in
`include/strata/artifact/glm_native.hpp`: hidden width 4096, expert width 2048,
dense FFN width 12288, KDA width 8192, MLA ranks 1536/512 with expanded
width 16384, and vocabulary width 154880. The suite benchmarks each resident projection at M=1/16/64;
for an expert, M means the number of token rows routed to that expert.
It does not run a model or modify Strata. Arithmetic uses synthetic signed
codes and varying positive K16 scales; no model-quality claim is made.

The shared runner's `--graph` option removes CPU launch gaps from repeated
decode timings. `--flush-mb 64` additionally writes 64 MiB before **each**
GEMM. In graph mode, explicit GPU event nodes enclose only the GEMM and its
reduction; eviction, graph creation, and graph upload are excluded. These
are cache-eviction measurements, not streamed-weight or host-transfer timings.
The default sweep records both modes, full decoded-BF16 cuBLAS comparisons,
per-candidate logs, library hashes, and `summary.csv` under `build/decode/`.
Output is checked after the actual timed graph runs.
`dispatch.csv` records the selected library and candidate for each shape/cache
case, retaining the bulk kernel unless decode improves its paired timing by
at least 2%. This is a measured configuration table, not integration into Strata.
`--resume` preserves completed rows after an interruption and checks their GPU
library hashes. Large vocabulary matrices use shorter tuning pilots while
retaining the same final iteration count.

The unpatched `build/libdecode_base.so` can also benchmark NVFP4 using
`--format nvfp4 --output bf16 --cutlass PATH`; the runner disables the
E0M3-only DP4A candidate for that format. Other tiles and reductions are
shared. Like the optimized bulk adapter, decode consumes the 128x4 scale
layout and returns BF16; packing and activation quantization are excluded.

## Build and test

Requires Linux, GCC, Python 3, a CUDA toolkit supporting `sm_120a`, and an SM120
GPU for GPU tests. The tested setup has CUDA 13.2, cuBLAS 13.4.1, and driver
615.71.09. Host programs load CUDA/cuBLAS dynamically through the repository's
existing wrappers; `nvcc` compiles the device-only CUBINs.

```sh
make -C cuda/int4 -j2
make -C cuda/int4 test-cpu
make -C cuda/int4 test-gpu
make -C cuda/int4 sanitize
```

CUDA 13.1's `cuobjdump` names the patched instructions as E0M3; CUDA 13.2's
disassembler does not. The Makefile prefers the installed
`/usr/local/cuda-13.1/bin/cuobjdump`. Override `CUOBJDUMP=/path/to/cuobjdump`
with a version that recognizes E0M3. Compilation still uses CUDA 13.2. The
patcher deliberately fails if it cannot verify every resulting instruction.

Build products are ignored under `build/`. Compiler/disassembler scratch files
use the repository's `tmp/int4/` directory; override with `INT4_TMP_DIR` when
needed. The patcher also accepts `--scratch-dir`. No `/tmp` directory is used.

## Numeric contract and API

```text
C[m,n] = sum_k decode(A[m,k]) * scale_A[m,k/16]
                 * decode(B[n,k]) * scale_B[n,k/16]
```

A and B have row-major logical shapes `[M,K]` and `[N,K]`. C is row-major FP32
`[M,N]`, with alpha=1 and beta=0. K is padded to `kp=roundup(K,64)`; code padding
must be zero. M and N need no alignment. Each packed row contains `kp/2` bytes,
with the earlier K element in the low nibble. Scale rows contain `kp/16` bytes
in ordinary K order, without a cuBLAS scale swizzle.

E0M3 codes are `0..7 => 0..7`, `8 => -0`, and `9..15 => -1..-7`. They are
incompatible with two's-complement `s4`, and cannot represent -8. Quantization
emits canonical positive zero. A finite UE4M3 scale is a byte in `0x00..0x7e`;
the encoding for scale 1 is `0x38`. Storage, including K16 scales, is 4.5 bits
per value before row padding. Accumulation follows floating-point MMA behavior.

`int4.h` provides:

- `int4_padded_k`, `int4_codes_size`, and `int4_scales_size` for storage sizing.
- `int4_quantize_cpu` and `int4_decode_cpu` for host packing and references.
- `int4_init`/`int4_destroy` for CUBIN loading into a caller-owned CUDA context.
- `int4_quantize_f32` and `int4_gemm` for asynchronous launches on a caller-owned
  stream, with no allocation or tuning in those functions.

The caller allocates device code/scale/output buffers and fills `int4_matrix`.
Code buffers require 16-byte alignment, scales/FP32 buffers 4-byte alignment,
and statistics 8-byte alignment; normal CUDA allocations satisfy all three.
Use format `INT4_E0M3` for quantization and operand pair `INT4_E0_E0` for GEMM.
Other pairs exist for FP4 controls and mixed-format validation. Variant 0 is
the one-warp reference. Variants 1–8 use BM=64, BN=64/128, BK=64/128, and
synchronous or double-buffered `cp.async` staging. See `int4_variant_names`.
Synchronize the stream before reading results or releasing its buffers/context.
Initialization requires SM120 and an already initialized `cuew` CUDA loader.

For each K16 block, the quantizer rounds `amax/7` to a positive finite UE4M3
scale, clamps it to `[2^-9,448]`, and rounds `x/decoded_scale` to nearest-even
in `[-7,7]`. Zero blocks use scale 1 and zero codes. FP4 controls use `amax/6`
and nearest-even E2M1 encoding. Inputs must be finite FP32. CPU quantization
rejects nonfinite values; finite input is a precondition for GPU quantization.
Very small/large blocks can lose range when the scale is clamped; there is no
additional per-tensor FP32 scale.

Optional device statistics accumulate into two caller-zeroed uint64 counters:
values outside the decoded representable range and blocks whose requested
scale is outside the finite positive scale range. Pass zero for no statistics.
Statistics are collected outside timed benchmark regions.

## CUBIN patching

PTX declares the documented `m16n8k64 ... e2m1.e2m1 ... ue4m3` operation.
`patch_formats.py` obtains the full 128-bit instruction from disassembly, maps
its address into the matching ELF `.text` section, verifies the original bytes,
and sets bits 14/15 of the second 64-bit word for E0M3 A/B. It patches all
compiler-generated copies in the designated kernels, including peeled loops.
The baseline is preserved, and no PTX JIT fallback is used.

Three copied CUBINs implement E0M3×E2M1, E2M1×E0M3, and E0M3×E0M3. A second
disassembly verifies their exact expected instructions. `patch_manifest.json`
records tool version, SHA256 hashes, instruction counts, kernel names, and
addresses; `.sass` files retain the evidence. This is an experimental SASS
interface, with no portability promise across architectures or toolkit changes.

## Benchmarks

```sh
make -C cuda/int4 probe
make -C cuda/int4 bench

cd cuda/int4
./build/bench_int4 --shape square_4k --mode all --csv build/4k.csv
./build/bench_int4 --shape square_4k --scales unit --csv build/unit-4k.csv
./build/bench_int4 --m 67 --n 131 --k 129 --mode all
make -C ../fp4 bench  # includes the original cuda/fp4 kernels
./build/bench_int4 --help
```

Defaults: five warmups, 100 launches per round, five rounds, and rotated job
order. `--variant auto` validates and times all nine variants on that shape,
then chooses the lowest three-launch pilot time outside measurement. These short
pilots can select different variants for E0M3 and E2M1. `--variant 0..8` forces
the same tiling when comparing formats directly.
Use `--warmup`, `--iters`, and odd `--rounds` to change timing settings.

| Mode | Timed device work |
|---|---|
| `probe` | Register MMA loop with eight independent accumulator tiles |
| `packed` | GEMM on packed operands |
| `activation` | FP32 A quantization + GEMM, with B prepacked |
| `full` | FP32 A and B quantization + GEMM |
| `all` | The three matrix modes above |

No timed path includes allocation, input transfer, compilation, tuning, or
validation. Pipeline throughput uses the same dense `2MNK/time` count as GEMM
throughput, so the quantization cost appears in the elapsed time.

The instruction probe compares all four operand pairs, BF16, INT8, and legacy
PTX `s4`. Operands arrive as runtime arguments and alternate sign between loop
iterations to prevent constant-folding of the `s4` unpacking. Each output is
checked. This is register-loop throughput, not optimized GEMM or a peak claim;
the loop also includes sign changes and output stores.

`--backend all|e0m3|fp4|e2m1|bf16|int8` selects matrix comparisons; `fp4` is an
alias for E2M1 FP4 and CSVs retain the `e2m1` label. Instruction probes
always compare all seven instruction modes. BF16 cuBLAS consumes
decoded E0M3 values and produces FP32 output, matching the E0M3 arithmetic.
INT8 cuBLAS uses unpacked E0M3 integers with unit scales and INT32 output; it
is explicitly skipped for K16 scales. `--scales unit` supports packed timing
only (also when `--mode all` is used). No unsupported scaled INT8 comparison
is presented as equivalent. Optional library failures print a skip reason.

`--legacy-fp4 1` adds the actual reference, shared-memory, and `cp.async`
GEMM kernels from `cuda/fp4` as packed-only rows (`fp4_ref`, `fp4_sync`,
`fp4_async`). Their original source strings live in a shared header and are
compiled once through NVRTC, outside timing. They consume the exact same FP4
codes and K16 scales as the `e2m1` row. Their existing dimension requirements
are preserved: even N, plus M%16=N%8=0 for the reference; unsupported shapes
print explicit skips. Use this option with `--mode packed` or `all`.
`make -C cuda/fp4 bench` runs the comparison from the repository root and sets
compiler scratch storage to repository-local `tmp/fp4/`. See the
[FP4 comparison results](../fp4/RESULTS.md).

Matrix correctness uses 64 deterministic output samples against FP64 sums of
decoded packed values, with relative L2 ≤1e-4. The independent small-matrix
test suite checks every output of every variant. `quant_L2` separately compares
the packed-input reference to the original FP32 inputs, on the same samples.
The synthetic FP32 input distribution is deterministic uniform `[-2,2]`;
unit-scale experiments instead use integers `[-7,7]`. These accuracy results
describe those inputs, rather than model quality.

CSV records median/min/max milliseconds, effective dense TOp/s, dimensions,
selected kernel, settings, accuracy, clipping counts, and verification. Each
CSV has a `.metadata.txt` sidecar with invocation and GPU/driver/cuBLAS/clock
information. `build/toolchain.txt` records compiler and disassembler versions
and flags. Clocks are not locked. Standard shapes cover 1K–8K squares,
vision projections, attention/FFN, and contextual M=1/16/64 cases.
