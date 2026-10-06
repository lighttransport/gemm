# Vulkan GEMM benchmark

Small C11 Vulkan 1.3 compute benchmark for row-major `C = A * B`. FP32 defaults
to a 128x128 output tile with 8x8 register tiles per invocation. Other types
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

The default register kernel uses 128x128 workgroup tiles, 8x8 outputs per invocation,
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

Optional tuning controls: `--fp32-tile 64|128`, `--fp32-rows 64|128`,
`--fp32-kstep 8|16|32`, `--fp32-pad 0|1`, `--fp32-prefetch 0|1`,
`--fp32-lds-prefetch 0|1`, and `--fp32-pack-a 0|1`. Defaults are 128x128,
Kstep=16, pad=0, global prefetch=1, LDS prefetch=0, packing=0.
The packing experiment transposes A on the GPU and reports its cost separately,
as well as packing plus one GEMM. A trial reached 3.75 TFLOP/s for the packed kernel,
excluding packing, with no consistent advantage at larger sizes. LDS prefetch and
larger register tiles caused register spills or slower execution and were rejected
as defaults. Changing queue family, padding B, changing accumulator orientation,
and grouping workgroups for cache reuse did not improve the sustained rate.
`--fp32-isa logs/fp32.isa` exports the actual driver disassembly when
`VK_AMD_shader_info` is available. Logs and full commands are in ignored
`logs/optimized-performance.json` and `logs/optimized-validation.log`.

## Validation commands

```powershell
python benchmark/vulkan-gemm/validate.py `
  benchmark/vulkan-gemm/build-mingw/bench_vulkan_gemm.exe `
  benchmark/vulkan-gemm/build-msvc/Release/bench_vulkan_gemm.exe
```

Result: **31 cases per compiler, 62 total PASS**. Covers non-tile-aligned and
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
