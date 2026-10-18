# 4-bit on consumer Blackwell (sm_120) — FP4 vs INT4, hardware or not?

**E0M3 update:** the INT4 conclusions below describe the legacy PTX `s4` path.
SM120 also executes signed-magnitude E0M3 (`-7..7`, FP32 accumulation) through
format-patched `OMMA.SF` instructions. See [cuda/int4](../int4/README.md) for the
validated RTX 5060 Ti implementation, quantization, and GEMM benchmarks.

**FP4 GEMM comparison:** `make -C cuda/fp4 bench` benchmarks the actual FP4
reference/shared-memory/`cp.async` kernels from this directory alongside the new
FP4 and E0M3 kernels, with common inputs, verification, and timing. See
[measured results and reproduction commands](RESULTS.md).

**cuBLAS correction (2026-10-18/19):** NVFP4 and per-tensor FP8 both work on
this SM120 card with cuBLASLt 13.4.1. The old probe used incorrect transpose
attribute IDs (`0/1` instead of `3/4`), producing false unsupported results.
`make -C cuda/fp4 cublas-bench` now measures verified, tuned 8192³ NVFP4 GEMM:
**236.82 TOp/s with FP32 output**, **261.08 with BF16 output**, and **260.94
with FP16 output**, all with FP32 accumulation. Current SM clocks are about
2070 MHz. The current FP4/E0M3 instruction probe measures about **297 TOp/s**;
the older 406 TOp/s probe below came from an earlier faster-clock run.

**300 TOp/s tuning attempt (2026-10-19):** the new optional CUTLASS path reaches
**280.8 TOp/s at 8192³**, with FP32 accumulation and BF16 output, at the same
approximately 2070 MHz clocks. This is **7.8% faster** than the paired cuBLAS
control, with all 67,108,864 outputs bit-identical to cuBLAS. The 300 TOp/s
target has not been reached. `make -C cuda/fp4 cutlass-bench` builds the selected
256x128x128 tile using unmodified CUTLASS headers; `CUTLASS_DIR` can override
the local checkout. `cutlass-sweep`, `cutlass-test`, and `cutlass-sanitize`
provide the full tile sweep, reference checks, and sanitizer checks.
See [the tuning results](RESULTS.md#attack-on-300-tops-at-current-clocks).

The same tuned kernel now also reaches **280.8 TOp/s with E0M3 inputs**,
using a verified format-only SASS patch. Build and benchmark it with
`make -C cuda/int4 cutlass-bench`; see the
[E0M3 results and validation](../int4/RESULTS.md#optimized-e0m3-at-8192-cubed-280-tops-target-reached).

The follow-up **290 TOp/s** attempt retains that bulk kernel: larger epilogues
and a different MMA/TMA handoff were slower. The new
[GLM5.3-Flash decode adapter](../int4/README.md#glm53-flash-decode-projections)
also reuses its unpatched MMA tiles for FP4. The common runner now supports
`--graph`, `--flush-mb`, and `--pilot-iters` for comparable short-kernel timing.

This directory revisits an earlier "FP4 is software-emulated on sm_120" assumption and tests
two hypotheses: **maybe NVFP4 is HW-accelerated while MXFP4 is software**, and (follow-up)
**maybe INT4 is available and HW-accelerated too.**

Answered empirically on an **RTX 5060 Ti (sm_120, 36 SMs, 2587 MHz boost)** with **CUDA 13.2**
ptxas/NVRTC and **cuBLASLt 13.4**.

---

## TL;DR

- **FP4 (e2m1) IS hardware-accelerated on consumer sm_120**, through the warp-level
  `mma.sync.aligned.m16n8k64 … e2m1 … block_scale` 5th-gen tensor-core instruction. (sm_120
  has *no* `tcgen05.mma`; that block-level instruction is sm_100/datacenter only.) The
  earlier raw probe reached ~406 TOPS, 4× its legacy FP8 MMA probe and 8× BF16.
- **NVFP4 == MXFP4** in throughput, within 0.1%. That hypothesis is **refuted** — both ride
  the same tensor-core datapath and differ only in *scale granularity*, not HW vs SW.
- **INT4 (s4) is the opposite story: available but SOFTWARE-EMULATED.** The `s4` mma still
  assembles at `sm_120a`, but ptxas lowers it to **2× `IMMA.s8` + ~180 integer unpack ops**
  (confirmed in SASS), so it runs at only ~5% of the INT8 mma rate (21.6 TOPS — *slower* than
  BF16). This describes two's-complement `s4`; the E0M3 path above uses a different
  codebook and FP32 accumulation. (INT8 itself is native and fast: 203 TOPS, 2× FP8.)
- **cuBLASLt supports NVFP4 on this SM120 card.** Correct transpose attributes,
  `VEC16_UE4M3` scales in the documented 128x4 layout, and FP32 compute select
  native kernels. The FP4 + `VEC32_UE8M0` combination returns invalid-value
  status 7 in the current library; its documented FP4 API uses K16/UE4M3.

> **Takeaway:** on consumer Blackwell the *new* 4-bit format (FP4) got a hardware datapath,
> while the *old* 4-bit format (INT4) lost its native tensor core and is now emulated on top
> of INT8.

---

## 1. What FP4 is

### 1.1 The e2m1 element (4 bits)

FP4 data is the **e2m1** format: 1 sign bit, 2 exponent bits, 1 mantissa bit. Two e2m1
values are packed per byte. The full representable set is just 16 codes:

```
0, 0.5, 1, 1.5, 2, 3, 4, 6   (and their negatives)
```

With so little dynamic range, FP4 is only usable with a **per-block scale factor** that
rescales each small run of K-elements — this is what makes it "block-scaled" FP4.

### 1.2 NVFP4 vs MXFP4 — the only difference is the scale

Both store the same e2m1 data; they differ purely in how the block scale is defined:

| format | block size (K) | scale type | scale enum (`cublasLtMatmulMatrixScale_t`) |
|--------|---------------:|------------|--------------------------------------------|
| **NVFP4** | 16 elements | `UE4M3` (8-bit, e4m3 magnitude — a real fp8 number) | `VEC16_UE4M3 = 1` |
| **MXFP4** | 32 elements | `UE8M0` (8-bit, exponent-only power-of-two) | `VEC32_UE8M0 = 2` |

NVFP4's finer 16-element block + fp8-precision scale gives better accuracy; MXFP4's 32-element
block + power-of-two scale is the open OCP "microscaling" standard. **Neither implies a
different compute path** — see the results below.

The encoding of 1.0 for each scale (used to fill the benchmark scale buffers):
`UE4M3 = 0x38`, `UE8M0 = 0x7F`.

---

## 2. How FP4 executes on sm_120

### 2.1 Warp-level `mma.sync`, not `tcgen05`

Datacenter Blackwell (sm_100, B200) runs block-scaled GEMM through `tcgen05.mma` (a
block/tile-level tensor-core instruction with a dedicated tensor-memory). **Consumer Blackwell
(sm_120) has no `tcgen05`** — FP4 instead goes through the classic warp-collective
`mma.sync` path, the same family used for BF16/FP8 since Ampere/Ada, extended with a
`.block_scale` modifier.

### 2.2 The instruction

```
NVFP4:  mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3
MXFP4:  mma.sync.aligned.m16n8k64.row.col.kind::mxf4    .block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0
```

- `m16n8k64` — one warp computes a 16×8 output tile with a K-depth of 64 (vs k32 for FP8,
  k16 for BF16). The deeper K is why one FP4 mma does 2× the FLOPs of one FP8 mma.
- `scale_vec::4X` (NVFP4, block 16 → 4 scales across k64) vs `scale_vec::2X` (MXFP4, block 32
  → 2 scales across k64).
- Operand registers per lane (identical fragment shape to BF16/FP8 mma):
  - **D / C** accumulator: 4 × f32
  - **A**: 4 × b32 (32 e2m1 packed)
  - **B**: 2 × b32 (16 e2m1 packed)
  - **scale-A**, **scale-B**: 1 × b32 each, followed by a `{byte-id, thread-id}` immediate
    selector pair that picks which packed scale byte/lane this thread contributes.

### 2.3 You must target `sm_120a`

The block-scale FP4 mma only assembles for the **architecture-specific** `sm_120a` target
(not plain `sm_120`). The repo's shared `cu_compile_kernels()` hardcodes
`--gpu-architecture=sm_%d` (→ `sm_120`), so `mma_fp4_probe` does its own NVRTC compile at
`--gpu-architecture=sm_120a`. A `sm_120a` CUBIN runs on the sm_120 GPU fine.

---

## 3. Benchmarks

### 3.1 `mma_fp4_probe` — raw `mma.sync` issue rate (the silicon test)

The decisive HW test. For each of BF16 / FP8 / NVFP4 / MXFP4 / INT8 / INT4 it generates a
kernel that issues a long stream of warp-level MMAs into `NACC` independent accumulator tiles
(to saturate the tensor cores), compiles it at `sm_120a`, and times it. Integer formats use an
s32 accumulator (`+r`), float formats an f32 accumulator (`+f`).

Two reasons this is definitive:
- **Assembly is the existence test.** ptxas only emits a tensor-core SASS op if the *target
  arch* supports it. If `…e2m1…` assembles at `sm_120a`, the datapath is in silicon; if ptxas
  rejects it ("not supported on target"), it is not.
- **Garbage operands are fine.** This measures *issue throughput*, not a correct GEMM, so the
  A/B/scale registers are seeded from `threadIdx` and never need a valid layout.

FLOPs per warp-level instruction: BF16 `2·16·8·16 = 4096`, FP8 `2·16·8·32 = 8192`,
FP4 `2·16·8·64 = 16384`.

### 3.2 `cublas_fp4_gemm` — cuBLASLt support probe + BF16 baseline

Drives `cublasLtMatmul` for BF16 (works), per-tensor FP8, and block-scaled NVFP4/MXFP4. To
test real scale indexing, the default inputs are varying signed FP4 codes and
positive K16 scales, packed in the documented 128x4 scale layout. `--data ones`
retains the original constant-input check. Allocation and packing are excluded
from timing. The runner searches up to 32 heuristic candidates (six were
returned for 8192³), verifies them against FP64 CPU sums, and records five
rounds of 100 launches with FP32, BF16, or FP16 output.

---

## 4. Results (RTX 5060 Ti, FP32 accumulate, throughput-bound: NACC 8 ≡ 16)

### 4.1 `mma_fp4_probe`

| format | mma shape | TOP/s | G mma/s | vs BF16 | vs FP8 | datapath |
|--------|-----------|------:|--------:|--------:|-------:|----------|
| BF16  | m16n8k16              |  51 | 12.5 | 1.0× | 0.5× | native |
| FP8   | m16n8k32 (e4m3)       | 102 | 12.5 | 2.0× | 1.0× | native |
| **NVFP4** | m16n8k64 (vec16/ue4m3) | **406** | 24.8 | 7.9× | 4.0× | **native** |
| **MXFP4** | m16n8k64 (vec32/ue8m0) | **406** | 24.8 | 7.9× | 4.0× | **native** |
| INT8  | m16n8k32 (s8)         | 203 | 24.8 | 3.9× | 2.0× | native (2× FP8!) |
| **INT4** | m16n8k64 (s4)      |  **22** | **1.3** | 0.4× | 0.2× | **EMULATED** |

(TOP/s = 1e12 ops/s; float rows are TFLOP/s, integer rows are TOPS — same unit.)

Read the **G mma/s** column (instruction issue rate), not just TOP/s:

1. **FP4 mma issues at 2× the FP8/BF16 rate** (24.8 vs 12.5) — a real, separate FP4 datapath,
   not a microcoded fallback. Plus each FP4 mma covers 2× the K of an FP8 mma → **4× FP8**.
2. **INT8 mma issues at 24.8 G mma/s — 2× the FP8 rate.** GeForce throttles FP8 to the FP16
   rate, but INT8 runs full-rate, so INT8 (203 TOPS) beats FP8 (102 TFLOP/s) here.
3. **INT4 mma issues at only 1.3 G mma/s — ~5% of INT8.** It is *not* a native op (see §4.3).

(Absolute BF16 ~51 TFLOP/s reflects GeForce's halved FP32-accumulate tensor rate; the *ratios*
are the clean signal.)

### 4.3 SASS evidence — FP4 is one native op, INT4 is emulated

Compiling single mmas with `nvcc -arch=sm_120a -cubin` and dumping SASS
(`cuobjdump --dump-sass`) shows the actual tensor-core opcode per format:

| PTX mma | SASS tensor op(s) | total SASS instrs | verdict |
|---------|-------------------|------------------:|---------|
| `…e2m1…block_scale…ue4m3` (NVFP4) | **1× `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`** | 32 | **native** |
| `…s8.s8…` m16n8k32 (INT8) | **1× `IMMA.16832.S8.S8`** | — | native |
| `…s4.s4…` m16n8k64 (INT4) | **2× `IMMA.16832.S8.S8`** + ~180 ALU | 224 | **emulated** |
| `…s4.s4…` m16n8k32 (INT4) | **2× `IMMA.16816.S8.S8`** + ALU | 128 | **emulated** |

- FP4 maps to a *single* native block-scaled tensor-core op `OMMA.SF…E2M1…` (`.SF` =
  scale-factor; `16864` = m16n8k64; `4X` = scale_vec). Real silicon.
- *Both* INT4 shapes lower to **INT8** `IMMA` ops (2×) plus ~180 `LOP3`/`SHF`/`IMAD` ops that
  unpack 4-bit → 8-bit. **No `IMMA.*.S4` opcode is emitted for sm_120 at all** — Turing/Ampere
  had one, Blackwell dropped it. The ~90× instruction blow-up is the measured ~19× slowdown.

### 4.4 Would writing the same legacy `s4` instruction directly help?

A natural follow-up: if inline-asm INT4 got emulated, can a hand-written PTX (or SASS) kernel
avoid this lowering? The emulation is not a frontend artifact:

1. `nvcc -arch=sm_120a -ptx` emits the PTX **verbatim**: it literally contains
   `mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32`. The CUDA-C frontend does *not* emulate.
2. Feeding that exact PTX straight to **`ptxas -arch=sm_120a`** produces the same
   `2× IMMA.S8 + unpack` SASS. **ptxas is the lowering stage** that emulates — so a hand-written
   PTX with the identical instruction is byte-for-byte identical.
3. This does not rule out another SASS instruction family. The E0M3 experiment in
   [cuda/int4](../int4/README.md) uses format bits in `OMMA.SF`, with signed-magnitude
   `-7..7` inputs and FP32 accumulation, rather than `IMMA.*.S4`.

FP4 needs no such path — it is already a single native `OMMA` regardless of route (inline asm,
hand PTX, or direct `ptxas`).

### 4.2 `cublas_fp4_gemm`

```sh
make -C cuda/fp4 cublas-bench
make -C cuda/fp4 cublas-test
make -C cuda/fp4 cublas-sanitize
```

For individual configurations, use `--format nvfp4 --output f32|bf16|f16`,
`--data random|ones`, `--algos 32`, `--workspace-mb 256`, and `--csv PATH`.
The default is 8192³ with FP32 output and varying signed codes/scales. The
matrix is `C[M,N] = A[M,K] * B[N,K]^T`; packing is outside timing. Results
include median/min/max, selected heuristic, workspace, and decoded-input
precision errors. A `.metadata.txt` sidecar records library/device details
and selected opaque algorithms. See [results](RESULTS.md) for the full setup.

| format | status | note |
|--------|--------|------|
| BF16 | supported | FP32 compute, TN layout |
| FP8 (per-tensor) | supported | see [FP8 measurements](../fp8/RESULTS.md) |
| NVFP4 (VEC16_UE4M3) | supported | 236.82 TOp/s F32 / 261.08 BF16 output at 8192³ |
| FP4 + VEC32_UE8M0 | invalid-value status 7 | this FP4 scale-mode combination is not exposed |

The earlier unsupported results were caused by incorrect descriptor IDs.
They do not describe cuBLAS support on this GPU. The current runner checks
descriptor-call status and validates the nonuniform scale layout. The raw
4× ratio above compares to legacy FP8 MMA, not the newer cuBLAS FP8 kernel.

---

## 5. Conclusions

1. **FP4 is hardware-accelerated on consumer sm_120**, at ~406 TOPS in the earlier
   raw probe — 4× its legacy FP8 probe and 8× BF16.
   The "FP4 is software-emulated" assumption was about the dequant→bf16/fp8 path; native
   `mma.sync` e2m1 is real silicon.
2. **NVFP4 and MXFP4 are equally hardware-accelerated** (identical to 0.1%). They share one
   datapath; NVFP4 vs MXFP4 is purely an accuracy/standard trade-off (block 16 + ue4m3 vs
   block 32 + ue8m0), *not* HW vs SW. Hypothesis refuted.
3. **INT4 (s4) is available but software-emulated** — it assembles, but ptxas lowers it to
   2× INT8 `IMMA` + integer unpacking (SASS-confirmed), so it runs ~19× slower than INT8 and
   even slower than BF16. E0M3 can instead use the native `OMMA.SF` datapath with a
   different integer codebook and FP32 accumulation; see the update above.
4. **INT8 is native** (203 TOPS in the earlier issue-rate probe). Its 2× ratio
   to FP8 concerns the legacy FP8 instruction tested there. Current cuBLAS FP8
   uses a faster path and reaches roughly 142 TOp/s at the current clocks.
5. **cuBLAS exposes NVFP4 on SM120.** The 8192³ library GEMM is substantially
   faster than the current custom FP4 kernels; use it as the optimization baseline.

---

## 6. Build & run

No CUDA SDK needed at compile time — pure C + `cuew` (driver API), cuBLASLt `dlopen`'d, and
the probe JIT-compiles its kernels with NVRTC at `sm_120a` at runtime.

```sh
make                  # builds cublas_fp4_gemm and mma_fp4_probe

./mma_fp4_probe                       # the definitive silicon probe
./mma_fp4_probe --iters 40000 --bpsm 8

./cublas_fp4_gemm                     # cuBLASLt support matrix + BF16 baseline (default 4096³)
./cublas_fp4_gemm -m 1024 -n 1024 -k 1024   # smaller, with the 1.0-fill verify on
```

`mma_fp4_probe` flags: `--iters N` (mma loop trip count), `--bpsm N` (blocks per SM).
`NACC` (independent accumulator tiles, default 16) is a compile-time `-DNACC=`.

---

## 7. Gotchas (worth remembering)

- **`sm_120a` required** for the block-scale FP4 mma; plain `sm_120` will not assemble it.
- **mma fragment shape** is shared across BF16/FP8/FP4 (D:4×f32, A:4×b32, B:2×b32, C:4×f32);
  FP4 just appends `scale-reg, {byte-id,thread-id}` per operand (selectors are literal `{0,0}`
  here since every scale is 1.0).
- **cuBLASLt descriptor IDs:** `TRANSA=3`, `TRANSB=4`. The previous `0/1`
  values refer to different attributes and caused misleading unsupported results.
  The corrected TN setup works for BF16, FP8, and NVFP4.
- **`cublasLtMatmulHeuristicResult_t.algo` is `uint64_t[8]` (64 B), not a `void*`** — the older
  fp8/bf16 templates mis-declared it as a pointer, which showed up as a bogus ~4 GB workspace
  print. Fixed here.
- Exact enums (CUDA 13.2 `library_types.h` / `cublasLt.h`): `CUDA_R_4F_E2M1 = 33`,
  `CUDA_R_8F_UE8M0 = 30`, scale modes `VEC16_UE4M3 = 1` / `VEC32_UE8M0 = 2`, desc attrs
  `A/B_SCALE_MODE = 31/32`.

## 8. References

- PTX ISA — Warp-level MMA with block scaling (`mma.sync … .block_scale`, `kind::mxf4nvf4` /
  `kind::mxf4`).
- NVIDIA CUDALibrarySamples — `cuBLASLt/LtNvfp4Matmul` (the layout/scale-mode reference).
- cuBLAS docs §"16/32-Element 1D Block Scaling for FP8 and FP4 Data Types".
- [cuBLAS FP4 requirements and scale layout](https://docs.nvidia.com/cuda/cublas/#d-block-scaling-factors-layout).
- [Current cuBLAS FP8 results](../fp8/RESULTS.md).
