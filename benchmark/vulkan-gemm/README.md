# Vulkan GEMM benchmark

Small C11 Vulkan 1.3 compute benchmark for row-major `C = A * B`. FP32 defaults
to adaptive output tiles up to 128x128, with up to 8x8 outputs per invocation. Other types
use the original 16x16 shared-memory kernel. Both FP32 kernels remain selectable.
Both the program and vkew loader compile as C. vkew resolves `vulkan-1.dll` at runtime,
so no Vulkan import library or SDK is needed to run the built benchmark.

## Build on Windows

Run these commands from the repository root. CMake and a shader compiler are required
at build time. The installed RenderDoc `glslangValidator` is discovered automatically;
alternatively set `-DSHADER_COMPILER=C:/path/to/glslangValidator.exe` or `glslc.exe`.
SPIR-V is generated in each build's `shaders/` directory. Keep that directory at its
configured location when running the executable; its absolute path is compiled in.

LLVM-MinGW, using the requested installation:

```powershell
cmake -S benchmark/vulkan-gemm -B benchmark/vulkan-gemm/build-mingw -G "MinGW Makefiles" `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_C_COMPILER=N:/local/llvm-mingw-20260602-ucrt-x86_64/bin/x86_64-w64-mingw32-clang.exe `
  -DCMAKE_MAKE_PROGRAM=N:/local/llvm-mingw-20260602-ucrt-x86_64/bin/mingw32-make.exe
cmake --build benchmark/vulkan-gemm/build-mingw --parallel 8
```

MSVC2022, x64:

```powershell
cmake -S benchmark/vulkan-gemm -B benchmark/vulkan-gemm/build-msvc -G "Visual Studio 17 2022" -A x64
cmake --build benchmark/vulkan-gemm/build-msvc --config Release --parallel 8
```

The agent environment on this PC contained both `PATH` and `Path` entries. MSBuild
failed during compiler discovery with a duplicate environment-key error. The following
equivalent commands normalize environment-key casing only in the child process:

```powershell
python -c "import os,subprocess; env={k.upper():v for k,v in os.environ.items()}; raise SystemExit(subprocess.run(['cmake','-S','benchmark/vulkan-gemm','-B','benchmark/vulkan-gemm/build-msvc','-G','Visual Studio 17 2022','-A','x64'],env=env).returncode)"
python -c "import os,subprocess; env={k.upper():v for k,v in os.environ.items()}; raise SystemExit(subprocess.run(['cmake','--build','benchmark/vulkan-gemm/build-msvc','--config','Release','--parallel','8'],env=env).returncode)"
```

## Run

```powershell
benchmark/vulkan-gemm/build-mingw/bench_vulkan_gemm.exe --info
benchmark/vulkan-gemm/build-mingw/bench_vulkan_gemm.exe
benchmark/vulkan-gemm/build-msvc/Release/bench_vulkan_gemm.exe
benchmark/vulkan-gemm/build-mingw/bench_vulkan_gemm.exe --type fp32 --m 512 --n 1024 --k 768 --iterations 20
```

Defaults: device 0, all six types, `M=N=K=1024`, two warmups, ten timed iterations.
`--device INDEX` selects a device from the printed inventory. `--type` accepts `all`,
`int8`, `int16`, `int32`, `fp16`, `fp32`, or `fp64`. `--help` lists the arguments.

| Path | Input storage | Multiply / accumulate | Output | Required optional features |
|---|---|---|---|---|
| int8 | signed 8-bit | signed 32-bit | signed 32-bit | storageBuffer8BitAccess, shaderInt8 |
| int16 | signed 16-bit | signed 32-bit | signed 32-bit | storageBuffer16BitAccess, shaderInt16 |
| int32 | signed 32-bit | signed 32-bit | signed 32-bit | none |
| fp16 | IEEE binary16 | FP32 | FP32 | storageBuffer16BitAccess |
| fp32 | IEEE binary32 | FP32 | FP32 | none |
| fp64 | IEEE binary64 | FP64 | FP64 | shaderFloat64 |

INT8/INT16 are storage-to-INT32 GEMM paths, not measurements of narrow integer
multiply throughput. FP16 is storage-to-FP32 GEMM and does not require or claim
native FP16 arithmetic. Unsupported paths are skipped in `all` mode; explicitly
selecting an unsupported path fails.

Before each type's benchmark, a 19x23x29 GEMM checks all 437 outputs against a CPU
reference. Requested matrices with at most 4096 outputs, or at most 65536 outputs with K <= 128,
also receive a full check;
larger matrices check 64 deterministic samples including the first and last output.
Inputs are reproducible signed values. Floating inputs use values divided by 17,
with explicit binary16 rounding for the FP16 reference. Integer inputs lie in
[-16,16], and K is bounded to avoid INT32 overflow. Integer comparison is exact;
floating absolute tolerance is `max(1,sum(abs(products))) * 2e-6` for FP32 arithmetic
and `* 1e-12` for FP64. Non-finite GPU results fail.

Reported mean/min/max GPU duration uses timestamp queries around each dispatch,
honors `timestampValidBits`/`timestampPeriod`, and excludes transfers, pipeline
creation, CPU references, and warmups. Rates count `2*M*N*K` operations. Matrix
buffers live in device-local memory; transfers reuse an 8 MiB host staging buffer.

## VRAM capacity verification

This runs separately from GEMM. All chunks remain allocated simultaneously. Each
32-bit word gets an address/chunk-dependent GPU pattern, then every word is checked
on the GPU. A second pass uses the complementary pattern. All chunks are written
before any chunk is verified, helping detect aliases. Transfers/dispatches are
bounded to avoid a single long-running display-GPU command.

```powershell
# Default: request 14 GiB with an extra 512 MiB live-budget reserve.
benchmark/vulkan-gemm/build-mingw/bench_vulkan_gemm.exe --vram-test

# Exact 14 GiB check validated on this PC, using the driver's live budget directly.
benchmark/vulkan-gemm/build-mingw/bench_vulkan_gemm.exe --vram-test 14 --vram-reserve-mib 0
benchmark/vulkan-gemm/build-msvc/Release/bench_vulkan_gemm.exe --vram-test 14 --vram-reserve-mib 0

# Smaller capacity check.
benchmark/vulkan-gemm/build-mingw/bench_vulkan_gemm.exe --vram-test 1
```

The program uses the largest device-local heap, queries `VK_EXT_memory_budget`,
respects allocation/buffer/storage-range limits, and enables
`VK_AMD_memory_overallocation_behavior` with overallocation disallowed when available.
It allocates up to 1 GiB per chunk, reducing final chunks to fit a changing budget.
System memory and the small host-visible VRAM aperture are excluded. The extra
reserve is controlled by `--vram-reserve-mib`; it defaults to 512. A zero extra
reserve still checks the live budget before each allocation. Budgets can change
after allocation, so a reserve is a planning margin rather than a residency guarantee.

`PASS` requires the full requested byte count to be allocated and verified with
both patterns. A smaller verified total prints `LIMITED` and exits 1. Allocation
success plus GPU access confirms usable Vulkan device-local capacity; Vulkan does
not guarantee that all allocations stay physically resident under Windows memory
management. `--vram-test` requires the memory-budget extension.

Exit status: 0 success; 1 runtime/correctness/insufficient-capacity failure; 2 invalid
arguments. A GPU fence has a 60-second timeout; timeout/device loss stops the process
without freeing resources that might still be in flight.

## RX570 research and observed results

AMD's [RX570 specification](https://www.amd.com/en/support/downloads/drivers.html/graphics/radeon-600-500-400/radeon-rx-500-series/radeon-rx-570.html)
lists fourth-generation GCN, 2048 stream processors, up to 1244 MHz boost, up to
5.1 TFLOPS, 256-bit GDDR5, and up to 224 GB/s bandwidth. Standard boards are listed
with at most 8 GB. This PC's 16 GB variant must therefore be checked through its
installed driver rather than inferred from the standard board specification.
AMD's [Polaris sub-dword discussion](https://gpuopen.com/learn/using-sub-dword-addressing-on-amd-gpus-with-rocm/)
describes 8/16-bit extraction from 32-bit registers; it does not imply modern
matrix-unit acceleration.

Vulkan [storage and arithmetic feature bits](https://docs.vulkan.org/guide/latest/extensions/shader_features.html)
are independent. In particular, `storageBuffer16BitAccess=true` does not imply
`shaderFloat16=true`. See also the [memory-budget contract](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceMemoryBudgetPropertiesEXT.html)
and [allocation/buffer limits](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceMaintenance4Properties.html).

Observed on 2026-10-05:

- GPU: Radeon RX 570 Series, AMD vendor 0x1002, device 0x67df.
- Vulkan API 1.3.260; AMD proprietary driver 26.5.2 (`vulkaninfo --summary`).
- LLVM-MinGW Clang 22.1.7; MSVC 19.44.35228, Visual Studio 2022 v143; glslang 14.3.0.
- INT8 arithmetic/storage, INT16 arithmetic/storage, FP64 arithmetic: supported.
  FP32/INT32: core. FP16 storage: supported; native FP16 arithmetic: unsupported.
- `gemm_3.spv` declares `Shader` and `StorageBuffer16BitAccess`, with no `Float16`
  arithmetic capability (`spirv-dis` inspection).
- Device-local heap 0: 15.750 GiB. Separate aperture heap 2: 0.250 GiB.
  Initial heap-0 budget approximately 14.992 GiB, varying during allocation.
- Maximum allocation and buffer: 2,147,483,648 bytes each. Storage buffer range:
  4,294,967,295 bytes. Shared memory: 32 KiB. Compute queue 1; 64 timestamp bits,
  40 ns timestamp period.

Initial 16x16 baseline 1024x1024x1024 means, ten iterations after two warmups:

| Path | LLVM-MinGW ms | Rate | MSVC2022 ms | Rate |
|---|---:|---:|---:|---:|
| int8 -> int32 | 3.1309 | 685.890 GOP/s | 2.7730 | 774.419 GOP/s |
| int16 -> int32 | 2.8780 | 746.185 GOP/s | 2.7511 | 780.603 GOP/s |
| int32 | 2.6683 | 804.817 GOP/s | 3.1335 | 685.326 GOP/s |
| fp16 -> fp32 | 1.9428 | 1105.364 GFLOP/s | 2.5102 | 855.495 GFLOP/s |
| fp32 | 2.5894 | 829.329 GFLOP/s | 2.0850 | 1029.988 GFLOP/s |
| fp64 | 7.3668 | 291.508 GFLOP/s | 7.3473 | 292.282 GFLOP/s |

Both builds passed every reference check. They use the same compiled shaders;
timing differences reflect run-to-run GPU behavior, not different CPU-generated GPU
machine code. The FP32 optimization comparison below uses longer runs.

Both builds passed the exact 14 GiB test with `--vram-reserve-mib 0`:

```text
VRAM PASS requested=15032385536 allocated=15032385536 verified=15032385536 bytes
```

Final budgets/usages: LLVM-MinGW 14.340/14.001 GiB; MSVC 14.278/14.001 GiB.
The conservative default can return `LIMITED` because of a changing driver budget:
one MSVC run fully verified 13.975 GiB while retaining the default reserve policy.
A 128 MiB-reserve LLVM-MinGW run fully verified 13.884 GiB and correctly returned
`LIMITED`. These are observations from separate runs, not fixed hardware limits.

## FP32 optimization results

For large matrices, the register kernel uses 128x128 workgroup tiles, 8x8 outputs per invocation,
K steps of 16, vector global loads/stores for aligned shapes, and an unrolled load
loop that keeps the next tile's global reads in flight during arithmetic. It uses
128 VGPRs, 16 KiB LDS, and zero scratch bytes on this driver. Ragged dimensions use
bounds checks; integer and FP16/FP64 kernels retain their original implementation.

Three runs per compiler, each with 20 warmups and 100 timed iterations, gave the
following **median of run averages**. All reference checks passed. GPU timing excludes
setup, input uploads, output download, and CPU validation. No A packing is used.

| Shape | Kernel | LLVM-MinGW TFLOP/s | MSVC2022 TFLOP/s |
|---|---|---:|---:|
| 2048 cubed | baseline | 1.008 | 0.976 |
| 2048 cubed | register | 3.633 | 3.582 |
| 4096 cubed | register | 3.595 | 3.597 |

At 2048 cubed, the optimized run averages ranged from 3.619 to 3.656 TFLOP/s for
LLVM-MinGW and 3.530 to 3.587 for MSVC. The fastest individual iterations reached
approximately 4.15 TFLOP/s. **A sustained average of 4 TFLOP/s has not been achieved.**
A separate 1000-iteration run averaged 3.628 TFLOP/s. Read-only AMD ADL sampling
observed loaded core clocks around 1.11 GHz, with a maximum of 1.139 GHz during that
run; the nominal 1244 MHz boost was not observed. This clock difference lowers the
available peak, but does not explain the entire remaining gap.

```powershell
benchmark/vulkan-gemm/build-mingw/bench_vulkan_gemm.exe --type fp32 `
  --m 2048 --n 2048 --k 2048 --warmup 20 --iterations 100
benchmark/vulkan-gemm/build-mingw/bench_vulkan_gemm.exe --type fp32 `
  --m 2048 --n 2048 --k 2048 --warmup 20 --iterations 100 --fp32-kernel baseline
```

Optional tuning controls: `--fp32-tile auto|64|128`, `--fp32-rows auto|64|128`,
`--fp32-kstep 8|16|32`, `--fp32-pad 0|1`, `--fp32-prefetch 0|1`,
`--fp32-lds-prefetch 0|1`, and `--fp32-pack-a 0|1`. Large-matrix defaults are 128x128,
Kstep=16, pad=0, global prefetch=1, LDS prefetch=0, packing=0.
Tile dimensions now default to `auto`; the rules below describe the selected shapes.
The packing experiment transposes A on the GPU and reports its cost separately,
as well as packing plus one GEMM. A trial reached 3.75 TFLOP/s for the packed kernel,
excluding packing, with no consistent advantage at larger sizes. LDS prefetch and
larger register tiles caused register spills or slower execution and were rejected
as defaults. Changing queue family, padding B, changing accumulator orientation,
and grouping workgroups for cache reuse did not improve the sustained rate.
`--fp32-isa logs/fp32.isa` exports the actual driver disassembly when
`VK_AMD_shader_info` is available. Logs and full commands are in ignored
`logs/optimized-performance.json` and `logs/optimized-validation.log`.

### Further tuning on 2026-10-08

Vector loads, stores, and prefetching now support partial output tiles and partial
K tiles. For row-major A, N and K must be divisible by four. For packed A, M and N
must be divisible by four, while K can be arbitrary. Every partial-tile read is
masked to zero and output stores are bounded. A separate specialization removes
these predicates for full tiles, preserving the original fast path. All other
shapes retain scalar loads and stores with bounds checks.

Default tile selection uses 64 output columns when M*N <= 262144 or N <= 64,
otherwise 128. Default row tiles are 64 when M*N <= 131072 or M <= 64, otherwise
128. This supplies more workgroups for small matrices. An explicit `--fp32-tile`
uses that dimension for rows too unless `--fp32-rows` is supplied. Explicit numeric
settings remain reproducible; `auto` resets selection to the default rules.

Paired LLVM-MinGW runs compared the committed `876fa234` kernel against the new
code. Each pair used 20 warmups and 100 timed iterations; the table reports the
median of three run averages. Inputs, timestamps, compiler, and GPU were shared,
and packing was disabled. The committed code and shader were built separately in
ignored `logs/before-build/` so shader replacement could not change the reference.

| M x N x K | Before TFLOP/s | After TFLOP/s | Ratio |
|---|---:|---:|---:|
| 512 x 512 x 512 | 0.834 | 0.928 | 1.11x |
| 1024 x 1024 x 1024 | 1.864 | 1.823 | 0.98x |
| 2048 x 2048 x 2048 | 3.634 | 3.647 | 1.00x |
| 2049 x 2048 x 2048 | 2.478 | 3.217 | 1.30x |
| 2048 x 2052 x 2048 | 2.498 | 3.216 | 1.29x |
| 2048 x 2048 x 2052 | 2.665 | 3.504 | 1.31x |

The near-2048 shapes gain 29-32%. The aligned 2048 kernel remains around
3.65 TFLOP/s; **sustained 4+ TFLOP/s is still not achieved**. Small-shape rates vary
substantially between short runs: a separate warm-state 512-cubed trial reached
2.09 TFLOP/s with the new 128x64 tile, but this was not sustained in the paired
short-run comparison. The 1024-cubed kernel is unchanged and its small difference
in this table should not be interpreted as a consistent regression. The generated AMD
ISA for the full 128x128-tile path is byte-identical to the committed kernel.

Fresh MSVC2022 runs (20 warmups, 100 iterations each) measured 3.625 TFLOP/s
for 2048 cubed, and 3.231 / 3.224 / 3.517 TFLOP/s for the M / N / K partial-tile
shapes respectively. These agree with the LLVM-MinGW results.

Further experiments with alternating LDS buffers, component-major LDS layouts,
row-major A in LDS, and shape constants did not improve aligned 2048-cubed
throughput and were discarded. Only the measured edge-path and tile-selection
improvements were retained.

Optional `--batch N` (1..1024, default 1) records several sequential GEMMs per
submission and timestamps the whole batch. GPU write-after-write barriers separate
the dispatches. Mean time is total measured GPU time divided by the iteration
count; min/max are per-GEMM batch averages. Inter-dispatch barriers are included,
so batched rates must be reported separately from the default per-submit rates.
Warmups and timed iterations never share a batch, and partial final batches are
counted correctly. Work is capped at 137438953472 operations per submission,
with at least one GEMM, to bound submission size. The actual batch limit is printed.
Batching did not raise the aligned 2048 rate above 4 TFLOP/s and is not the default.

```powershell
benchmark/vulkan-gemm/build-mingw/bench_vulkan_gemm.exe --type fp32 `
  --m 2049 --n 2048 --k 2048 --warmup 20 --iterations 100
benchmark/vulkan-gemm/build-mingw/bench_vulkan_gemm.exe --type fp32 `
  --m 512 --n 512 --k 512 --fp32-tile auto --fp32-rows auto
benchmark/vulkan-gemm/build-mingw/bench_vulkan_gemm.exe --type fp32 `
  --m 1024 --n 1024 --k 1024 --warmup 256 --iterations 512 --batch 8
```

Paired measurements and exact arguments are in `logs/edge-performance.json`;
correctness and argument-check output is in `logs/edge-validation.log`.

### Address-calculation WIP (2026-10-09)

Recovered the measured candidate from the tuning logs after experiments had
restored the production shader. It computes invariant A/B lane addresses once,
then adds uniform offsets for subsequent load strips. The same decomposition
simplifies LDS addresses. These identities rely on the supported K steps
(8/16/32) and vector widths (16/32), each of which divides the 256-thread group.
The defaults and tile-selection rules are unchanged.

Paired LLVM-MinGW measurements taken while the RX570 was installed used three
trials, 100 warmups and 200 timed iterations per run. Median run averages:

| Shape | Kstep | Before TFLOP/s | WIP TFLOP/s | Gain |
|---|---:|---:|---:|---:|
| 2048 cubed | 8 | 3.113 | 3.605 | 15.8% |
| 2048 cubed | 16 (default) | 3.624 | 3.704 | 2.2% |
| 4096 cubed | 8 | 3.149 | 3.611 | 14.7% |
| 4096 cubed | 16 (default) | 3.524 | 3.586 | 1.8% |

All these runs passed the full 19x23x29 reference and the requested-shape sampled
checks. The AMD shader report used 124 VGPRs for the default aligned path, versus
128 previously, with no scratch memory. Exact commands and results are in ignored
`logs/address-paired-performance.json`. The device also reported 32 active compute
units, four SIMDs per compute unit, 64-lane waves, and 256 VGPRs per SIMD through
`VK_AMD_shader_core_properties` and `VK_AMD_shader_core_properties2`.

**WIP limitation:** the AMD GPU has been removed. The previously reported 112-case
suite predates this address change; a full runtime regression of ragged, packed-A,
and alternate-tile modes remains pending. Only compilation/static checks can be
repeated now. Sustained 4+ TFLOP/s is still unmet. One-wave workgroups, 96x96 tiles,
and additional LDS prefetch experiments were slower and are not included.

## Validation commands

```powershell
python benchmark/vulkan-gemm/validate.py `
  benchmark/vulkan-gemm/build-mingw/bench_vulkan_gemm.exe `
  benchmark/vulkan-gemm/build-msvc/Release/bench_vulkan_gemm.exe
```

Result: **56 cases per compiler, 112 total PASS**. Covers non-tile-aligned and
rectangular shapes, 1x1x1, all supported types, malformed/overflowing arguments,
invalid device selection, oversized buffers/indexing, integer overflow prevention,
a 32 MiB capacity test, and insufficient VRAM budget. Full 14 GiB checks are the
separate commands above. Logs remain under ignored `logs/`.

Existing C++ loader compatibility was also checked without warnings:

```powershell
N:/local/llvm-mingw-20260602-ucrt-x86_64/bin/x86_64-w64-mingw32-clang++.exe `
  -std=c++17 -Wall -Wextra -Wpedantic -Werror -c vulkan/deps/vkew.cc `
  -o benchmark/vulkan-gemm/build-mingw/vkew_cpp.obj
```

From an MSVC2022 x64 developer prompt:

```cmd
cl /nologo /TP /std:c++17 /W4 /WX /c vulkan\deps\vkew.cc /Fobenchmark\vulkan-gemm\build-msvc\vkew_cpp.obj
```

`vkew.cc` now includes the canonical C implementation for compatibility. Compile
either `vkew.c` or `vkew.cc` into a target, never both.
