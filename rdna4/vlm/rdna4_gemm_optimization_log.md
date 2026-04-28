# RDNA4 VLM GEMM Optimization Log

Target: Qwen3.6 vision projector `mm0`, BF16 inputs with FP32 accumulation,
`M=1024 N=4608 K=4608`, RX 9070 XT `gfx1201`. Peak reference used by the
benchmark is `195.0 TFLOP/s`; 80% target is `156.0 TFLOP/s`.

## Best Current Result

| mode | ms | TFLOP/s | peak | correctness |
| --- | ---: | ---: | ---: | --- |
| `mm0asm` generated external CO, LDS stride 144 | 0.3074 best/latest | 141.481 best/latest | 72.6% best/latest | sampled `PASS`, `max_abs=0`, `rms=0`, `cosine=1.000000000` |
| `mm0asm` generated external CO, old LDS stride 128 | 0.3217 latest control | 135.196 latest control | 69.3% | sampled `PASS`, `max_abs=0`, `rms=0`, `cosine=1.000000000` |
| `mm0pipe` HIPRTC | 0.3220 best/latest | 135.070 best/latest | 69.3% best/latest | sampled `PASS`, `max_abs=0`, `rms=0`, `cosine=1.000000000` |

Command:

```sh
rdna4/vlm/bench_vlm_gemm --dtype bf16 --shape mm0 --mode mm0pipe --iters 200 --check
VLM_GEMM_ASM_CO=rdna4/vlm/generated/mm0_bf16_asm.co rdna4/vlm/bench_vlm_gemm --dtype bf16 --shape mm0 --mode mm0asm --iters 200 --check
```

The default generated `mm0asm` target now uses `MM0_ASM_LDS_STRIDE ?= 144`.
Use `make -C rdna4/vlm MM0_ASM_LDS_STRIDE=128 generated/mm0_bf16_asm.co` to
rebuild the old unpadded baseline.

## Optimization Timeline

| step | change | ms | TFLOP/s | peak | contribution |
| --- | --- | ---: | ---: | ---: | --- |
| baseline | `mm0vec`, vectorized global/LDS staging but poor schedule | 3.8724 | 11.230 | 5.8% | established that vector width alone is not enough |
| pipe explicit stores | `mm0pipe`, double-buffered LDS, explicit epilogue stores | 0.7837 | 55.489 | 28.5% | removed scratch/private epilogue traffic from accumulator pointer tables |
| launch bounds | `__launch_bounds__(128, 1)` on `mm0pipe` | 0.6688 | 65.026 | 33.3% | let compiler use high VGPR count without spills |
| barrier cleanup | removed redundant pre-store CTA barrier | 0.6479 | 67.124 | 34.4% | inactive LDS buffer did not need a pre-store barrier |
| transposed LDS + split barrier | LDS layout `kslot*128+row`, inline `s_barrier_signal/wait` | 0.3235 | 134.436 | 68.9% | main speedup; fixed wave-local LDS access pattern and avoided broad `__syncthreads()` invalidation |
| no-bias diagnostic | passed null bias pointer | 0.3263 | 133.256 | 68.3% | epilogue/bias load is not the bottleneck |
| pipe8 transposed | 8-wave CTA, transposed LDS, split barrier | 0.3601 | 120.760 | 61.9% | correct, but rereads A across more N-side waves |
| pipe4n64 transposed | 128x64 tile, transposed LDS, split barrier | 0.4519 | 96.232 | 49.3% | correct, but increases A reload traffic |
| pipek64 | 128x128 tile, K tile 64, double-buffered 64 KiB LDS | 0.3539 | 122.890 | 63.0% | correct, but larger LDS footprint and wait schedule lose more than halved barriers gain |
| pipemid | K32 tile, store/signal next LDS buffer after first half of WMMA group | 0.3540 | 122.831 | 63.0% | correct, but source-level mid-store scheduling forces extra waits instead of hiding handoff |
| pipegl | K32 tile, source-level global-load interleaving before WMMA groups | 0.8322 | 52.253 | 26.8% | correct, but LLVM forms a much worse control/dependency schedule; volatile version was worse at 3.3936 ms |
| split-tail diagnostic | moved final K32 stage out of the hot loop, then restored | 0.3588 | 121.214 | 62.2% | correct, but losing the original loop shape costs more than removing `has_next` control flow |
| generated external CO | `gen_mm0_bf16_asm.py` emits `gemm_mm0_bf16_asm`, loaded by `mm0asm` | 0.3215 latest, 0.3212 best | 135.262 latest, 135.376 best | 69.4% | establishes external handwritten-code-object ABI and reproducible generator flow |
| LDS stride padding | changed default generated LDS stride from 128 to 144 for both A and B operands | 0.3074 best/latest | 141.481 best/latest | 72.6% best/latest | +4.6% versus the same-session stride-128 control; reduces LDS bank/schedule pressure even though LDS grows from 32 KiB to 36 KiB |
| asymmetric LDS stride sweep | tested A/B stride pairs around 136-152 | 0.3094 best long-run cross candidate | 140.541 | 72.1% | one-sided A padding helped more than one-sided B padding, but symmetric 144 remained the best long-run result |
| generated `.s` round-trip | `mm0_bf16_asm.s` assembled and linked with ROCm clang | 0.3268 | 133.082 | 68.2% | correct and directly editable as AMDGPU assembly, but slower than hipcc `--genco` fatbin |
| asm halfsplit | split the 16 LDS fragment loads into two 8-load/16-WMMA halves | 0.3276 | 132.757 | 68.1% | correct, but delaying the second LDS half starves WMMA issue instead of hiding latency |
| asm storeearly20 | moved next-buffer LDS stores after the first 20 WMMAs and left the barrier after all WMMAs | 0.3303 | 131.643 | 67.5% | correct, but added mid-loop control/dependency cost and did not hide handoff |
| asm bfirst2 | reordered second K16 LDS loads so B4-B7 arrive before A5-A7 | 0.3300 | 131.792 | 67.6% | correct, but no gain over the editable `.s` baseline in the current run |
| direct-A/LDS-B diagnostic | generated kernel keeps A fragments in VGPRs from global loads and stages only B through LDS | 0.6724 | 64.677 | 33.2% | correct, but naïve wave-fragment global loads are too inefficient; hipBLASLt's direct operand path also depends on its packed/global-read mapping |

Current-session control measurements:

- `mm0pipe` HIPRTC re-run: `0.3323 ms`, `130.882 TFLOP/s`, correctness
  `PASS`. The earlier `0.3220 ms` path was not reproducible in this thermal/run
  state.
- `mm0asm` old LDS stride-128 control: `0.3217 ms`, `135.196 TFLOP/s`,
  correctness `PASS`.
- `mm0asm` optimized LDS stride-144 default: `0.3074 ms`,
  `141.481 TFLOP/s`, correctness `PASS`.
- hipBLASLt BF16 `mm0`, epilogue `none`, targeted algo `73624`:
  `0.2629 ms`, `165.420 TFLOP/s`, `84.8%` of the benchmark peak.
- hipBLASLt BF16 `mm0`, epilogue `none`, `--algos 64 --workspace-mb 512`:
  algo `73778`, `0.2340 ms`, `185.825 TFLOP/s`, `95.3%` of the benchmark peak.
  This is a stronger target than the earlier `0.2675 ms` GELU-bias run and
  shows the handwritten path is missing a deeper pipeline, not a minor wait
  shuffle.

LDS stride sweep from the profiler-guided pass:

| variant | LDS bytes | iters | ms | TFLOP/s | correctness |
| --- | ---: | ---: | ---: | ---: | --- |
| stride 128 control | 32768 | 200 | 0.3217 | 135.196 | `PASS` |
| stride 136 symmetric | 34816 | 200 | 0.3082 | 141.107 | `PASS` |
| stride 144 symmetric | 36864 | 200 | 0.3074 | 141.447 | `PASS` |
| stride 152 symmetric | 38912 | 200 | 0.3086 | 140.926 | `PASS` |
| stride 160 symmetric | 40960 | 200 | 0.3080 | 141.170 | `PASS` |
| stride 176 symmetric | 45056 | 100 | 0.3644 | 119.326 | `PASS` |
| stride 192 symmetric | 49152 | 100 | 0.3546 | 122.651 | `PASS` |
| A144/B128 | 34816 | 100 | 0.3168 | 137.251 | `PASS` |
| A128/B144 | 34816 | 100 | 0.3271 | 132.930 | `PASS` |
| A144/B136 | 35840 | 100 | 0.3180 | 136.743 | `PASS` |
| A136/B144 | 35840 | 100 | 0.3203 | 135.772 | `PASS` |
| A144/B152 | 37888 | 100 | 0.3219 | 135.078 | `PASS` |
| A152/B144 | 37888 | 200 | 0.3094 | 140.541 | `PASS` |

The useful padding window is narrow. Stride 136-160 breaks the worst LDS bank
pattern without changing the mainloop, while 176+ is too much padding and
regresses. Asymmetric padding shows A-side padding is more useful than B-side
padding alone, but both operands still need the same 144 stride for the best
long-run timing.

## Math Model

For `mm0`, the workload is:

- FLOPs: `2*M*N*K = 43,486,543,872` FLOPs.
- CTAs: `(1024/128)*(4608/128) = 288` CTAs.
- K32 stages per CTA: `4608/32 = 144`.
- Dynamic WMMA instructions for `mm0pipe`: `288 CTAs * 4 waves * 144 stages * 32 WMMA = 5,308,416` wave-level WMMA instructions.
- Each wave-level `v_wmma_f32_16x16x16_bf16` represents `16*16*16*2 = 8192` FLOPs.

At `0.3074 ms`, the current optimized generated kernel issues about `17.27 G`
wave-level WMMA instructions/s. Peak `195 TFLOP/s` corresponds to `23.80 G`
such instructions/s, the 80% target corresponds to `19.04 G`, and hipBLASLt
algo `73624` at `0.2629 ms` corresponds to `20.19 G`. The current kernel is
therefore still issue-limited relative to peak; reaching the 80% target needs
about `1.10x` more speed, while matching the latest hipBLASLt algo run needs
about `1.17x`.

Tile-reloaded traffic, ignoring cache effects:

- X traffic: `M*K*2*(N/128) = 339.7 MB`.
- W traffic: `N*K*2*(M/128) = 339.7 MB`.
- Y traffic: `M*N*4 = 18.9 MB`.
- Total tile traffic: `~698 MB`, or `~2.27 TB/s` at `0.3074 ms`.

This is high but still not the primary limit: `--no-bias` does not improve
time, and K64/midstore variants that target handoff overhead regress even
though they preserve or reduce nominal data movement. The limiting factor is
how well LDS reads, waits, stores, and WMMA issue are scheduled.

Assuming 64 CUs, 288 CTAs means about `4.5` CTA waves over the chip. The current
`0.3074 ms` implies roughly `68.3 us` per CTA, or `~474 ns` per K32 stage. The
latest hipBLASLt algo `73624` reference at `0.2629 ms` implies `~406 ns` per
K32 stage, while the stronger full-sweep hipBLASLt run at `0.2340 ms` implies
`~361 ns` per K32 stage. The near-term missing performance is therefore about
`68 ns` per K32 stage versus the selected hipBLASLt algo, while the best
library path shows an even larger schedule/dataflow gap.

## Current Static Resource Snapshot

Static instruction counts are from disassembly after compiling with
`VLM_GEMM_DUMP=1`. Resource metadata for the external code object is from the
`rocprofv3` kernel trace.

| kernel | LDS | VGPR | SGPR | scratch/private | max WG | static WMMA | static gld128 | static ds_load128 | static ds_store128 | static barriers | static waits |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gemm_mm0_bf16_pipe4w` | 32768 | 238 | 14 | 0 | 128 | 32 | 16 | 16 | 16 | 4 | 121 |
| `gemm_mm0_bf16_asm` generated CO, LDS stride 144 | 36864 | 248 | 128 | 0 | 128 | 32 | 16 | 16 | 16 | 4 | 121 |
| `gemm_mm0_bf16_pipe4w_glint` | 32768 | 238 | 15 | 0 | 128 | 32 | 16 | 16 | 16 | 4 | 129 |
| `gemm_mm0_bf16_pipe4w_midstore` | 32768 | 231 | 15 | 0 | 128 | 32 | 16 | 16 | 16 | 4 | 122 |
| `gemm_mm0_bf16_pipe8w` | 32768 | 159 | 14 | 0 | 256 | 32 | 12 | 24 | 12 | 6 | 98 |
| `gemm_mm0_bf16_pipe4w_k64` | 65536 | 229 | 14 | 0 | 128 | 64 | 32 | 32 | 32 | 4 | 153 |
| `gemm_mm0_bf16_pipe4w_n64` | 24576 | 168 | 17 | 0 | 128 | 16 | 16 | 12 | 22 | 4 | 175 |

Static counts are from disassembly counters and are not dynamic instruction
counts. They are still useful for comparing compiler schedule shape.

## Interpretation

The dominant gain came from the transposed LDS layout. The old layout made each
wave read fragments with poor lane locality; the transposed layout turns local
fragment reads into contiguous row slices for RDNA4 WMMA.

The no-bias test shows the scalar output epilogue is not the primary limiter.
The K64 test shows barrier count alone is not the primary limiter either:
doubling the K stage increases LDS to 64 KiB and grows the generated wait/load
schedule enough to lose performance.

The remaining gap to hipBLASLt is mainloop dataflow and scheduling. The current
best hipBLASLt control run reaches `185.825 TFLOP/s`, while the handwritten path
is now around `140-141 TFLOP/s` depending on run state. The compiler still groups
many LDS loads before WMMA and emits many waits. A competitive handwritten path
should implement the mainloop in GCN assembly, following the hipBLASLt pattern:
issue next global loads early, stage with `ds_store_b128`, interleave small LDS
read groups with WMMA, and use split barriers only at true buffer handoff
points.

The `pipegl` and split-tail diagnostics strengthen that conclusion. Source-level
attempts to coerce the hipBLASLt schedule do not merely fail to improve the
kernel; they make LLVM create extra waits/branches or lose the favorable loop
shape. The target schedule needs explicit control over `global_load_b128`,
`ds_store_b128`, `ds_load_b128`, `s_wait_*`, and `v_wmma` placement.

## Generated Assembly Path

`gen_mm0_bf16_asm.py` now emits:

- `rdna4/vlm/generated/mm0_bf16_asm.hip`
- `rdna4/vlm/generated/mm0_bf16_asm.s`
- `rdna4/vlm/generated/mm0_bf16_asm.co`

The benchmark mode `mm0asm` loads `VLM_GEMM_ASM_CO` and launches
`gemm_mm0_bf16_asm` with the same ABI as `mm0pipe`. This isolates code-object
iteration from the benchmark harness. The current generator still emits HIP
source and lets hipcc lower the final GCN, but its register families and WMMA
schedule are generated from tables. The next step is to replace the generated
HIP mainloop with direct `.s` emission while keeping the same kernel metadata,
symbol, and launch ABI.

The `.s` round-trip is already functional:

```sh
make -C rdna4/vlm generated/mm0_bf16_asm_from_s.co
VLM_GEMM_ASM_CO=rdna4/vlm/generated/mm0_bf16_asm_from_s.co rdna4/vlm/bench_vlm_gemm --dtype bf16 --shape mm0 --mode mm0asm --iters 100 --check
```

That path measured `0.3268 ms`, `133.082 TFLOP/s`, correctness `PASS`. It is
slower than the hipcc `--genco` fatbin path but gives us a directly editable
AMDGPU assembly file with valid `.amdhsa_kernel` metadata.

Patched assembly targets now build from `patch_mm0_asm_schedule.py`:

- `make -C rdna4/vlm bench-mm0-asm-halfsplit`
- `make -C rdna4/vlm bench-mm0-asm-storeearly20`
- `make -C rdna4/vlm bench-mm0-asm-bfirst2`

The direct-A diagnostic target builds from `gen_mm0_bf16_directa.py`:

- `make -C rdna4/vlm bench-mm0-directa`

## ROCm Profiler Counter Pass

ROCm profiler setup on ROCm `7.2.1`:

- Installed tools used: `rocprofv3`, `rocprofv3-avail`.
- Required runtime library package: `hsa-amd-aqlprofile`.
- `rocprof-compute` is present, but its Python dependencies are not installed.
- Legacy `rocprof --list-basic/--list-derived` aborts on gfx1201 in this
  environment.

`rocprofv3-avail -d 0 info` sees the RX 9070 XT correctly as:

- `gfx_target_version=120001`
- `cu_count=64`
- `simd_count=128`
- `wave_front_size=32`

Commands used for the useful counter group:

```sh
/opt/rocm/bin/rocprofv3 --kernel-trace \
  --pmc SQ_WAVES SQ_BUSY_CYCLES \
  --kernel-include-regex gemm_mm0_bf16_asm \
  -d /tmp/rocprof-mm0asm-wait -o mm0asm -f csv -- \
  env VLM_GEMM_ASM_CO=rdna4/vlm/generated/mm0_bf16_asm.co \
  rdna4/vlm/bench_vlm_gemm --dtype bf16 --shape mm0 --mode mm0asm --iters 20 --check

/opt/rocm/bin/rocprofv3 --kernel-trace \
  --pmc SQ_WAVES SQ_BUSY_CYCLES \
  --kernel-include-regex Cijk_Alik_Bljk \
  -d /tmp/rocprof-hblt-wait -o hblt -f csv -- \
  rdna4/vlm/bench_vlm_hipblaslt --dtype bf16 --shape mm0 --epilogue none \
  --iters 20 --algos 64 --algo-index 73624 --workspace-mb 512
```

Several accepted RDNA4 counters currently report zero through `rocprofv3`:
`SQ_INSTS_VALU`, `SQ_INSTS_LDS`, `SQ_WAIT_ANY`, `GL2C_EA_RDREQ`, `TCP_REQ`,
`VALUInsts`, `LdsUtil`, `VALUBusy`, and occupancy-derived metrics. Treat those
as unavailable in this ROCm build, not as real zero activity.

Usable counter comparison. The stride-128 and hipBLASLt rows are from the first
21-dispatch pass; the stride-144 row is from a 51-dispatch confirmation pass.

| kernel | profiled trace median | LDS | scratch | VGPR | waves | `SQ_BUSY_CYCLES` | busy/wave |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mm0pipe` | `358.9 us` | `32768` | `0` | `248` | `1152` | `31.929 M` | `27716` |
| `mm0asm` | `357.2 us` | `32768` | `0` | `248` | `1152` | `29.865 M` | `25925` |
| `mm0asm` LDS stride 144 | `329.4 us` avg, `319.8 us` median | `36864` | `0` | `248` | `1152` | `29.153 M` avg, `28.492 M` median | `25306` avg, `24733` median |
| hipBLASLt algo `73624` | `291.3 us` | `25600` | `0` | `256` | `1152` | `25.319 M` | `21979` |

Non-profiled timing controls from the same session:

- `mm0asm` stride 128: `0.3254 ms`, `133.658 TFLOP/s`, `68.5%`.
- `mm0asm` stride 144: `0.3074 ms`, `141.481 TFLOP/s`, `72.6%`.
- hipBLASLt algo `73624`: `0.2629 ms`, `165.420 TFLOP/s`, `84.8%`.

Interpretation:

- All kernels do the same number of wave dispatches (`1152`) for this shape.
- The selected hipBLASLt kernel uses `25 KiB` LDS per CTA versus our optimized
  `36 KiB`.
- hipBLASLt spends about `15.2%` fewer busy cycles than `mm0asm`
  (`21979 / 25925`) and about `20.7%` fewer than `mm0pipe` per wave.
- LDS stride 144 improves the observed non-profiled runtime by about `3.9-4.5%`
  against the same-session stride-128 control, but the profiler busy-cycle
  average only drops about `2.4%` and remains noisy. Median busy/wave drops more
  clearly to `24733`, supporting the LDS-conflict interpretation.
- The measurable bottleneck is therefore per-wave pipeline length, not grid
  occupancy or scratch spilling. This matches the assembly inspection:
  hipBLASLt overlaps next global reads, LDS stores/loads, barrier handoff, and
  WMMA work more tightly, while our generated path serializes more of the
  global-to-LDS-to-WMMA chain.

Because memory and instruction mix counters are currently zero through ROCm
`7.2.1` on this gfx1201 stack, the profiler cannot yet prove a DRAM/L2
bottleneck directly. The available evidence still points away from bandwidth:
same wave count, no scratch, lower hipBLASLt LDS footprint, and substantially
lower SQ busy cycles per wave.
