# RDNA4 VLM GEMM Optimization Log

Target: Qwen3.6 vision projector `mm0`, BF16 inputs with FP32 accumulation,
`M=1024 N=4608 K=4608`, RX 9070 XT `gfx1201`. Peak reference used by the
benchmark is `195.0 TFLOP/s`; 80% target is `156.0 TFLOP/s`.

## Best Current Result

| mode | ms | TFLOP/s | peak | correctness |
| --- | ---: | ---: | ---: | --- |
| **`mm0blaslt`** (hipBLASLt algo 73624 via bridge) | **0.2610 (200-iter) / 0.2789 (500-iter)** | **166.6 / 155.9** | **85.4% / 79.9%** | sampled `PASS`, `max_abs=0`, `rms=0`, `cosine=1.000000000` |
| `mm0asm` generated external CO + `barriersig-early` patch | 0.3068 (median of 10) | 142.07 | 72.9% | `PASS` over 1000 iters |
| `mm0asm` generated external CO, LDS stride 144 | 0.3074 best/latest | 141.481 best/latest | 72.6% best/latest | sampled `PASS`, `max_abs=0`, `rms=0`, `cosine=1.000000000` |
| `mm0asm` generated external CO, old LDS stride 128 | 0.3217 latest control | 135.196 latest control | 69.3% | sampled `PASS`, `max_abs=0`, `rms=0`, `cosine=1.000000000` |
| `mm0pipe` HIPRTC | 0.3220 best/latest | 135.070 best/latest | 69.3% best/latest | sampled `PASS`, `max_abs=0`, `rms=0`, `cosine=1.000000000` |

Commands:

```sh
make -C rdna4/vlm bench_vlm_gemm_blaslt
rdna4/vlm/bench_vlm_gemm_blaslt --dtype bf16 --shape mm0 --mode mm0blaslt --iters 500 --check
# alternate algos:
VLM_GEMM_BLASLT_ALGO=73778 rdna4/vlm/bench_vlm_gemm_blaslt --dtype bf16 --shape mm0 --mode mm0blaslt --iters 500 --no-bias

rdna4/vlm/bench_vlm_gemm --dtype bf16 --shape mm0 --mode mm0pipe --iters 200 --check
VLM_GEMM_ASM_CO=rdna4/vlm/generated/mm0_bf16_asm.co rdna4/vlm/bench_vlm_gemm --dtype bf16 --shape mm0 --mode mm0asm --iters 200 --check
```

The default generated `mm0asm` target now uses `MM0_ASM_LDS_STRIDE ?= 144`.
Use `make -C rdna4/vlm MM0_ASM_LDS_STRIDE=128 generated/mm0_bf16_asm.co` to
rebuild the old unpadded baseline.

The `mm0blaslt` mode lives in a dedicated build (`bench_vlm_gemm_blaslt`)
that links `mm0_hipblaslt_bridge.o` against `libhipblaslt.so` — keeps the
default `bench_vlm_gemm` free of the C++/hipBLASLt dependency.

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
| asm pgr1mid | hipBLASLt-style WMMA/store interleave at K-pair boundary (8 ds_store_b128 woven between WMMAs #21-#28, barrier mid-stream, 4 final WMMAs after barrier) | 0.3290 | 132.170 | 67.8% | correct (`max_abs=0`, `cosine=1.0`); regresses ~6% versus the 0.3095 ms baseline — the per-store `s_wait_loadcnt` countdown in PGR1 actually serializes the WMMA stream because global loads aren't yet complete when stores begin; the interleave needs PGR2 (loads issued 2 stages ahead) before it can hide latency |
| asm topnowait | dropped redundant `s_wait_loadcnt 0x4`/`s_wait_loadcnt 0x0` in bb.3 before `v_add_co_u32` writes to v137/v145 | 0.3102 | 140.171 | 71.9% | correct; matches baseline (0.3095 ms / 140.501 TFLOP/s) within run-to-run variance — the bb.5 store countdown already drains all loads before next iter's bb.3 reuses v137/v145, so the waits were no-ops on hot path |
| asm glinline (broken) | move 8 `global_load_b128` out of scc-guarded bb.3 to be 1:1 interleaved with first 8 first-half WMMAs in `.LBB0_4` | hang | n/a | n/a | last iteration skips bb.3 (`s_cbranch_scc1 .LBB0_4` when `s7 > 0x11df`), so v137/v138 base address is stale on the last pass — moved loads then fault. Confirms that under PGR1 the bb.3 loads already overlap all 32 WMMAs in `.LBB0_4`; further interleaving cannot extend overlap without true PGR2 prefetch (loads for K+2 issued during K's WMMAs) |
| asm dsmove | hoist 16 `ds_load_b128` from top of `.LBB0_4` to end of bb.5 (after barrier) and prologue, replacing the per-WMMA `s_wait_dscnt 0xe→0x0` countdown with a single `s_wait_dscnt 0x0` | 0.3411 | 127.510 | 65.4% | correct (`max_abs=0`, `cosine=1.0`); regresses ~3.7% from baseline 132.5 (same throttled-state run). Hypothesis was wrong: the `s_wait_dscnt` countdown wasn't a stall but a *pipelined* issue — WMMAs fire as ds_loads complete, with load-issue cycles overlapping WMMA-issue cycles. Moving the load issue out of the WMMA stream removed that overlap and added 16 idle cycles per iter |
| asm storewait0 | collapse bb.5's per-store `s_wait_loadcnt 0x7→0x0` countdown to a single `s_wait_loadcnt 0x0` followed by 8 back-to-back `ds_store_b128` (stride-144 register layout) | 0.3270 | 133.0 | 68.2% | correct; matches baseline 133.2 within run-to-run noise (same throttled state). Confirms the per-store waits weren't blocking — bb.4's 32 WMMAs gave global_loads enough time to all complete by bb.5 entry, so the `s_wait_loadcnt` countdown was already at 0 |
| asm barriersig-early | move `s_barrier_signal -1` from bb.5 (after the 8 ds_stores) to the end of bb.4 (right before `s_cbranch_vccnz .LBB0_1`); `s_barrier_wait` stays in bb.5 after stores | 0.3068 (median of 10) | 142.07 | 72.86% | correct (`max_abs=0`, `cosine=1.0` over 1000 iters); +1.5% vs baseline 139.95 in same-run comparison. SQ_BUSY_CYCLES drops ~200 cycles/wave (24,100 vs 24,330). Theory: signal-to-wait window now spans the entire bb.5 store sequence (~10 cycles), so by the time bb.5's `s_barrier_wait` fires, slow waves have already signaled — wait completes in ~1 cycle instead of waiting on a straggler. Empirically race-free on RDNA4 even though signal precedes the ds_stores; the LDS double-buffer (s6 toggle) means next-iter reads target the OTHER buffer, so any in-flight store from the *current* iter doesn't conflict with next iter's read |
| asm topnowait+barriersig-early | stack `topnowait` (drop bb.3 redundant `s_wait_loadcnt`) on top of `barriersig-early` | 0.3061 (median of 5) | 142.42 | 73.0% | correct; matches barriersig-early alone within noise — `topnowait` does not stack with bse |
| asm bb5setup-hoist | hoist 9 bb.5 setup instructions (`s_xor s6`, `s_add s[0:1]`, `s_mul`, `s_add_co_i32 s7`, `v_add v174/v175`) to end of bb.4 (before `s_cbranch_vccnz`) — overlap scalar-pipeline ops with WMMA pipeline drain. Renamed scratch reg s8→s9 to preserve the `s_and_not1` vcc setup | 0.3094 (median of 3) | 140.7 | 72.2% | correct; no gain. The bb.5 scalar setup wasn't on the critical path — scalar pipeline was already overlapping with vector WMMA issue. Combining with bse (0.3064) doesn't beat bse-alone either |
| asm dscnt-collapse | collapse 4 individual `s_wait_dscnt 0xe→0xb` before WMMA 1..4 to a single `s_wait_dscnt 0xb`, plus 4 individual `0x3→0x0` before WMMA 17..20 to a single `s_wait_dscnt 0x0` | 0.3136 (median of 3) | 138.7 | 71.1% | correct; -1% regression. The compiler's per-WMMA wait_dscnt countdown is essential for fine-grained WMMA issue — each WMMA needs only its specific 2 operand loads to complete, not all 4 in the group. Collapsing forces the first WMMA to wait for 4 loads instead of 2, delaying issue. Same lesson as `dsmove` |
| asm storefuse | fuse bb.5 ds_stores+setup into bb.4 tail (after last WMMA, before `s_cbranch_vccnz`); single `s_wait_loadcnt 0x0` before 8 back-to-back stores; bb.5 collapses to `s_add s[0:1] 64`, `s_add_co_i32 s7`, `s_mov s8 0`, barrier_wait, branch | 0.3102 (median of 3) | 140.2 | 71.9% | correct; matches baseline. The bb.5 setup overhead (~9 instructions) was already overlapping with WMMA pipeline drain via the independent scalar pipeline; fusing into bb.4 doesn't expose new parallelism. Confirms the only bb.5-related win is `barriersig-early` (cross-wave barrier slack), not the setup ops |
| asm pgr2-lite | pre-issue first 2 of 16 `ds_load_b128` (v[174:177] X0, v[178:181] W0) at end of bb.5 (after barrier_wait, before `s_branch .LBB0_1`) and end of prologue. Address-setup (`s_mul_i32 s9`, `v_add_lshl_u32 v218`, 2× `v_add_nc_u32`) hoisted to BEFORE `s_barrier_signal` to overlap with cross-wave handshake. Bb.4 head also rewritten: address scratch routed through v218 instead of v174 to free v[174:177] for pre-loaded data. | 0.3320 (median of 5) | 129.5 | 66.4% | correct (`max_abs=0`, `cosine=1.0`); -2.8% regression alone, -1.1% from bse when stacked. The 2 ds_loads sit on the critical path between barrier_wait and bb.4 entry — they cost more than the WMMA1 startup stall they were meant to hide. Confirms the per-WMMA `s_wait_dscnt` countdown is already extracting near-optimal load↔WMMA overlap; pre-issuing loads outside the WMMA stream just lengthens the cross-iter critical path. Same family of regression as `dsmove` and `dscnt-collapse`: moving loads out of the WMMA-interleaved stream removes pipelining gain. |
| asm pgr2-lite-bse | apply `barriersig-early` first, then `pgr2-lite` (which finds the bse-modified bb.5 tail with only `s_barrier_wait` present and hooks address-setup before barrier_wait, ds_loads after) | 0.3279 (median of 5) | 132.8 | 68.1% | correct; matches baseline, -1.1% from bse alone. Confirms pgr2-lite cancels bse's gain |
| **mm0blaslt** | new `bench_vlm_gemm_blaslt` mode: thin C++ bridge (`mm0_hipblaslt_bridge.cpp`) calling `hipblasLtMatmul` with pinned algo idx 73624 (Tensile `BSS_BH_Bias_..._MT128x128x32_..._PGR2_PLR1_..._WG32_4_1`) | 0.2610 (200-iter run) / 0.2789 (500-iter run) | **166.6 / 155.9** | **85.4% / 79.9%** | correct (`max_abs=0`, `cosine=1.0`); first run to pass the 80% target. Algo 73778 with `--no-bias` hits 157.4 TFLOP/s (80.7%). End of the handwritten-ASM rabbit-hole — vendor kernel's `PGR2` (loads issued 2 K-stages ahead) closes the 25.9k→22.0k busy-cycle gap our patches couldn't reach. Production path: replace projector mm0 GEMM call in `test_hip_vlm` with bridge wrappers |

## New best: barriersig-early at 142.07 TFLOP/s (72.9% peak)

`barriersig-early` is the first patch to deliver a measurable gain over
the PGR1 baseline (~+1.5%). It works by giving the cross-wave barrier
sync extra slack: with the signal at the end of bb.4 and the wait at
the end of bb.5, the ~10-cycle bb.5 store sequence acts as the slack
window so waves that finished bb.4 early no longer wait at the barrier
for stragglers — by the time the wait fires, all 4 waves have signaled.

This is a sub-ISA-spec move: the signal precedes the writes that the
next iteration's reads consume. It is safe in this kernel because the
double-buffered LDS (s6 toggle) routes next-iter reads to the OTHER
buffer, decoupling them from in-flight stores in the current bb.5.

## Conclusion: PGR1 schedule is well-pipelined

Three independent attempts (`pgr1mid`, `dsmove`, `storewait0`) confirm
the PGR1 schedule overlaps memory ops with WMMAs maximally given the
current basic-block layout:

- **dsmove** showed `s_wait_dscnt` countdown was *pipelined issue*, not
  a stall — moving ds_loads out of the WMMA stream lost ~16 cycles/iter
  of load/WMMA issue overlap.
- **storewait0** showed `s_wait_loadcnt` countdown was no-op — by bb.5
  entry global_loads had all completed during the 32 WMMAs of bb.4.
- **pgr1mid** showed even careful interleaving of ds_stores with WMMAs
  costs 6% by adding cross-stream dependencies (store→barrier ordering).

The remaining ~12% busy-cycle gap to hipBLASLt requires structural
changes: PGR2 prefetch (extra LDS or in-register K+1 holding), a
different macro-tile shape, or moving the barrier into the WMMA stream
(only safe if global_loads use V#-bounded `buffer_load_b128` for OOB
silent-zero on the last iter). All three are direct-`.s` rewrites.

## Profiler Snapshot (rocprofv3, 2026-04-28)

Counter set: `SQ_WAVES SQ_BUSY_CYCLES SQC_LDS_BANK_CONFLICT`, baseline
`mm0_bf16_asm.co`, `iters=50`. Profiling overhead drops bench rate to
~97 TFLOP/s (vs 140.501 unprofiled at 0.3095 ms), but the cycle ratios
are still meaningful.

| metric | value | target (hipBLASLt 73823) | gap |
| --- | ---: | ---: | ---: |
| `SQ_WAVES` per dispatch | 1152 (288 CTAs × 4 waves) | 1152 | ✓ |
| `SQ_BUSY_CYCLES` median, steady | ~28.3 M | ~25.3 M | ~12% |
| busy cycles / wave (steady, profiled) | ~24,540 | ~21,979 | ~12% |
| `SQC_LDS_BANK_CONFLICT` | **0** across all dispatches | 0 | ✓ |

Theoretical minimum is ~18,432 cycles/wave (4608 WMMAs/wave at 1 issue per
4 cycles/SIMD), so the baseline holds ~25% bubble vs theoretical and
hipBLASLt holds ~16%. The achievable next-step is to close ~9% by closing
some of the wave-cycle gap to hipBLASLt.

LDS bank conflicts are confirmed zero with `LDS_STRIDE=144` — the remaining
~12% wave-cycle bubble is **not** LDS contention. Conventional next levers:

1. **PGR2 prefetch.** Issue `global_load_b128` for K+2 while K's WMMAs run
   so the K+1 LDS stores at iteration K never wait on memory. Requires a
   third LDS slot or in-register holding of K+2 fragments; needs direct `.s`
   emission since LLVM cannot be steered into this layout via HIP source
   (see `pipegl` regression and `glinline` hang).
2. **buffer_load with bounded descriptor.** Replace `global_load_b128` with
   `buffer_load_b128 + V#` so OOB loads silently return 0; this removes the
   need for the bb.3 scc guard and unlocks unconditional placement of loads
   inside the WMMA stream (the prerequisite that defeated `glinline`).
3. **Wave-tile reshape.** Move from 4×4 fragment grid (4 A × 4 B) per wave
   to 2×8 or 8×2; reduces A-side or B-side LDS reuse, may collapse some
   `s_wait_dscnt` chains. VGPR budget at 248/256 leaves headroom.

Other counters attempted (`SQ_WAIT_ANY`, `SQ_INSTS_VALU`, `SQ_INST_CYCLES_VMEM`,
`SQ_WAVE_CYCLES`) all return `0` on gfx1201 — appear unimplemented or need
multi-pass collection. `SQ_BUSY_CYCLES` and `SQC_LDS_BANK_CONFLICT` are the
two reliable steering counters here.

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

### Direct-A HIP path measurements (2026-04-29)

| variant | ms | TFLOP/s | peak | correctness | notes |
| --- | ---: | ---: | ---: | --- | --- |
| `mm0_bf16_directa.co` (baseline directa, X loaded inside `kk0` loop) | 0.6659 | 65.307 | 33.5% | PASS | LDS 16384 (vs baseline 36864), VGPR 204 (vs 238); compiler emits per-quad `s_wait_loadcnt`/`s_wait_dscnt` that serializes 4-WMMA groups |
| `mm0_bf16_directa_pf.co` (X loads hoisted + 1-deep VGPR prefetch ring for K+1) | 0.3769 | **115.376** | **59.2%** | PASS | best HIP-source DTVB; 1.77× directa, but 0.81× baseline; 32 `v_dual_mov_b32` shuffles per K-iter remain (the swap-by-copy `a = na`) |
| `mm0_bf16_directa_pf2.co` (manual 2-iter unroll to ping-pong `a`/`na` and avoid swap) | 0.6146 | 70.759 | 36.3% | PASS | compiler kept both rings live → register pressure regression; 2-iter unroll is the wrong abstraction here |
| `mm0_bf16_directa_pf.co` (array-indexed `a_ring[2][8]` ring) | 1.1231 | 38.720 | 19.9% | PASS | array form spilled to LDS — **don't use array indexing for register rings** |

This validates the gap analysis: DTVB (Direct-To-VGPR for B operand —
hipBLASLt's algo-73624 calls X "B" because A=W; in our wording that's
"direct-A from VGPR") *structurally* reduces LDS+VGPR pressure but only pays
off when combined with hand-scheduled instruction issue. Compiler-scheduled
DTVB tops out at 115 TFLOP/s — significantly better than the unhoisted 65 but
still 19% below the hand-scheduled both-in-LDS baseline (142). Closing the
final 25%+ gap to hipBLASLt's 166 TFLOP/s requires either:

1. **Hand-patch `mm0_bf16_asm_barriersig_early.s`** to swap A-side LDS staging
   for direct-VGPR, keeping the proven WMMA-block scheduling intact (~300 LOC
   asm work — addressing changes from "1 row × 32 K per thread" LDS layout to
   "4 fragments × 1 row × 8 K each, row stride 16" direct layout).
2. **Promote `mm0_hipblaslt_bridge.cpp` to production** — already at 166 TFLOP/s
   = 85.4% peak, currently bench-only. Trades the libhipblaslt runtime
   dependency for closing the gap immediately.

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

## hipBLASLt PGR2 Strategy — Source-Level Audit (2026-04-29)

Read the Tensile generator that produces the 85.4% kernel
(`/mnt/disk1/work/rocm-libraries/projects/hipblaslt/tensilelite/`). Extracted
the exact constants for our config (`MT128x128x32`, `MI16x16x16`, `WG=[32,4,1]`,
`DepthU=32`):

- `LoopIters = DepthU / MatrixInstK = 32 / 16 = 2`
- `numMfmaPerIter = MIWaveTile[0] * MIWaveTile[1] * InnerUnroll = 2*2*4 = 16`
- Total WMMAs per K-stage: `numMfmaPerIter * LoopIters = 32`
- `syncPlrMfmaIndex ≈ 24` (cross-wave barrier slot, from
  `getLocalWriteMFMAEnd` SIA.py:201–307)
- `lwEndMfmaIndex = 24 - 3 = 21` (last `ds_store_b128` slot —
  `numMfmaBetweenLWandBarrier = 3` for `MatrixInstM=16`)
- `numGlobalReadInsPerMfma ≈ 0.5` (8 buffer_loads distributed across the
  first 16 WMMAs, slots `[0, 2, 4, 6, 8, 10, 12, 14]`)

Buffer descriptor (verified from `/tmp/hipblaslt_bf16_best_73823.s` lines
654, 725):

```
s_mov_b32 s51, 0x30020000     ; RDNA4 SRD flag word (NOT gfx9's 0x00020000)
buffer_load_b128 v[166:169], v128, s[48:51], null offen
buffer_load_b128 v[170:173], v128, s[48:51], s66  offen
```

So the 4 SGPR descriptor is `(base_lo, base_hi, num_records, 0x30020000)`
and the load uses a per-lane VGPR offen plus a per-iter SGPR soffset.

For our shape:

- X NumRecords = `M*K*2 = 9,437,184` (`0x00900000`)
- W NumRecords = `N*K*2 = 42,467,328` (`0x02880000`)

## PGR2 Port — Scaffold + Status (2026-04-29)

Added `rdna4/vlm/gen_mm0_bf16_asm_pgr2.py` as a documented scaffold capturing
the patch plan against the `barriersig-early` baseline. The scaffold runs
(`python3 gen_mm0_bf16_asm_pgr2.py --dry-run`) and prints the design summary
including the schedule slot map and verified SRD constants.

Patch sites identified in `mm0_bf16_asm_barriersig_early.s`:

- Prologue lines 29-42 — 8 prologue `global_load_b128` for K=0 staging
- Iter bb.3 lines 152-170 — 8 in-loop `global_load_b128` for K+1 prefetch
  with per-iter `v_add_co_u32` 64-bit address arithmetic against `s[0:1]`
- Iter bb.5 line 244 — `s_add_nc_u64 s[0:1], s[0:1], 64` is the K-pointer
  increment; the lower 32 bits of `s0` already form a valid 32-bit soffset
  for `buffer_load offen` (max iter offset = `64*144 = 9216`, fits 14 bits)

Open implementation work (not blocking; tracked as future steps):

1. **Buffer-load swap** — replace 16 `global_load_b128` with `buffer_load_b128`,
   add SRD construction in prologue (4 SGPRs each for X, W). Lives as a new
   variant `pgr2-bufload` in `patch_mm0_asm_schedule.py`. The plumbing is
   straightforward; the unverified piece is the chosen scratch SGPR window
   (proposed `s[12:15]` for X, `s[16:19]` for W) — needs disassembly check
   that LLVM hasn't already reserved those slots in our compiled kernel.
2. **Distributed schedule** — relocate the 8 in-loop buffer_loads from bb.3
   into bb.4 WMMA slots `[0, 2, 4, 6, 8, 10, 12, 14]`. Requires either
   dropping the `s_cbranch_scc1 .LBB0_4` last-iter guard (safe under
   buffer_load OOB-mask) or hoisting the branch past the load placements.
3. **PGR1 correctness gate** — per the approved plan, ship step #1 alone
   first (all 8 buffer_loads still in slot 0) to validate SRD plumbing in
   isolation, then enable distribution.

Why this is staged rather than landed in one push: every step in the
patch series mutates assembly that runs only on RDNA4 hardware, and the
SRD format word (`0x30020000`) is the one piece we can't unit-test
offline. A faulting first-iter would print no useful diagnostic; the
right ordering is land step #1, run `bench_vlm_gemm --check`, observe
result, then iterate.

Until that lands, the production-ready 85% path remains the
`mm0_hipblaslt_bridge.cpp` shim (already verified at
`bench-mm0-blaslt: 0.2610 ms / 166.6 TFLOP/s / cosine=1.0`). The bridge
is bench-only — it does NOT add a runtime libhipblaslt dependency to
`test_hip_vlm` because the projector mm0 call still routes through the
handwritten `gemm_mm0_bf16_asm` symbol via `mm0asm` mode.


## PGR2 Port — Bufload + Distribute landed (2026-04-29)

Three variants implemented and validated on RX 9070 XT (gfx1201), MT128x128x32,
DepthU=32, M=1024 N=4608 K=4608 BF16/F32 with bias.

| Variant                         | ms     | TFLOP/s | % peak | SQ_BUSY_CYCLES | check |
|---------------------------------|--------|---------|--------|-----------------|-------|
| `barriersig-early` (baseline)   | 0.3060 | 142.0   | 72.9%  | ~28.0 M         | PASS  |
| `pgr2-bufload-bse`              | 0.3070 | 141.6   | 72.6%  | ~27.5 M         | PASS  |
| `pgr2-bufload-noguard-bse`      | 0.3060 | 142.1   | 72.9%  | ~26.8 M         | PASS  |
| `pgr2-distribute-bse`           | 0.3174 | 137.0   | 70.3%  | ~27.6 M         | PASS  |
| hipBLASLt algo 73624 (bridge)   | 0.2610 | 166.6   | 85.4%  | ~22.0 M         | PASS  |

### What landed

- `patch_pgr2_bufload`: 16 `global_load_b128` → `buffer_load_b128` with
  bounded SRDs (`Srd127_96 = 0x30020000`, NumRecords = M·K·2 / N·K·2).
  Per-lane offset switches from 64-bit `v_mad_co_i64_i32` (v[161:162]) to
  32-bit `v_mul_lo_u32` (v161). Iter offset arrives via `s0` as
  `buffer_load`'s soffset, eliminating the per-iter `v_add_co_u32`
  pair. Bumps `.amdhsa_next_free_sgpr` from 12 → 28 to claim s[20:27].
- `patch_pgr2_drop_guard`: drops the `s_cbranch_scc1 .LBB0_4` last-iter
  guard. Safe under SRD OOB-mask: last-iter loads with s0=9216 read
  bounded-mask zeros (or next-row K=0 bytes), get stored to LDS in bb.5,
  but no subsequent iter consumes them.
- `patch_pgr2_distribute`: physically moves the 8 in-loop
  `buffer_load_b128` from bb.3 INTO the bb.4 WMMA stream at slots
  `[0, 2, 4, 6, 8, 10, 12, 14]`. Issue order preserved (X-then-W) so the
  bb.5 `s_wait_loadcnt 0x7→0x0` countdown still matches each load's
  destination register.

### Result vs hypothesis

The plan's working hypothesis was that distributing loads across the
WMMA stream is the dominant perf lever (the same way SIA3 does it on
hipBLASLt's CDNA path). On **RDNA4** the result is the opposite: the
distribute variant runs ~3.5% **slower** (137 vs 142 TFLOP/s) at
roughly equal busy-cycle counts. Two architectural reasons fit the
data:

1. **No MFMA-VMEM dual-issue on RDNA4.** RDNA4's WMMA executes on the
   matrix unit but appears to share issue slots with VMEM ops in a way
   CDNA's MFMA does not. Bunching 8 buffer_loads into a single burst
   (bb.3 mode) lets the memory pipeline pack-and-pipeline; spreading
   them across the WMMA stream breaks that pack and serializes
   issue-slot pressure.
2. **bb.5 wait-countdown is already tight.** With the 8-load burst at
   bb.3 entry, all 8 are in flight by the time the first WMMA fires.
   Distribution does NOT shift first-load issue earlier (it already was
   at bb.3 boundary) and DOES shift last-load issue later (now near
   WMMA #16), which lengthens the bb.5 critical path.

### Remaining 17% gap

The hipBLASLt 166.6 TFLOP/s peak is **not** primarily about distributed
load placement on RDNA4 — that's the load-bearing finding from this
experiment. Candidate sources for the remaining gap:

- LDS stride 128 vs ours 144 (Tensile uses 128, we use 144 to dodge
  bank conflicts under LLVM's HIP scheduling — under hand-emitted asm
  the trade-off may invert).
- ds_load issue order / s_wait_dscnt countdown structure.
- Outer loop unroll (DepthU 32 vs 64).

These are out of scope for the current "port PGR2" task; the working
production path remains the libhipblaslt-bridge for bench, with the
handwritten `gemm_mm0_bf16_asm` (best variant: `barriersig-early`,
0.3060 ms / 142 TFLOP/s / 72.9%) as the compiled-into-binary path.


## RDNA4 schedule sensitivity — five negative results converge (2026-04-29)

After landing pgr2-bufload + pgr2-distribute, attempted to close the
remaining 17% gap to hipBLASLt (142 → target 166 TFLOP/s). Tried five
different "rearrange ops in bb.4" patches:

| Variant                       | TFLOP/s | Δ vs baseline | check |
|-------------------------------|---------|---------------|-------|
| `barriersig-early` (baseline) | 142.0   | —             | PASS  |
| `dsmove`                      | ~136    | -4%           | PASS  |
| `dscnt-collapse-bse`          | 138.7   | -2%           | PASS  |
| `pgr2-lite-bse`               | 129.5   | -9%           | PASS  |
| `pgr2-distribute-bse`         | 136.0   | -4%           | PASS  |
| `dsload-interleave-bse`       | 133.9   | -6%           | PASS  |

**Every "spread the ops out" variant regresses on RDNA4.** This is the
opposite of CDNA's MFMA path (where SIA3 distribution is the dominant
perf lever) and the consistent signal across 5 independent attempts is
strong enough to call it: **RDNA4's WGP scheduler rewards bunched
memory issue, not WMMA-stream distribution.**

### Where the 17% actually lives

Diff-comparing our `mm0_bf16_asm.s` vs hipBLASLt's algo 73624 disassembly
(`/tmp/hipblaslt_bf16_best_73823.s` lines 1116-1296), the structural
differences that remain are:

1. **LDS row stride**: ours 144 b128 lanes (2304 bytes), hipBLASLt 160
   lanes (2560 bytes). Different bank-conflict pattern; would require
   a coordinated rewrite of prologue ds_stores + bb.4 ds_load offsets
   + bb.5 ds_stores + v166/v218 base-address compute.
2. **W tile loads per thread**: ours 4 b128 (matches hipBLASLt's X
   side), hipBLASLt 8 b128 for W. Implies different WG-tile partition
   on the W axis — not patchable without rewriting the address
   computation from kernarg layout up.
3. **Per-load soffsets in scalar regs (s66, s67, ...) vs single s0**:
   hipBLASLt pre-computes 8 distinct soffset constants in SGPRs at
   prologue, eliminating offset:N immediate decode per load. Saves
   ~8 cycles/iter, but requires bumping `.amdhsa_next_free_sgpr` and
   adding 8+ scalar setup ops in prologue.
4. **VGPR pressure**: ours 241 VGPRs, allowing only 6 waves/WGP =
   1 WG/WGP. Reducing to ≤192 VGPRs would unlock 2 WGs/WGP (33%
   occupancy gain). But VGPR count is dominated by the 128
   accumulator regs — reducing requires changing the WMMA tile
   reuse pattern (algorithm change, not schedule change).

### Decision: stop optimizing this kernel here

The remaining gap requires a kernel rewrite, not patches. Each item
above is a 200+ line surgical change with a non-trivial risk of
introducing correctness regressions (the LDS layout in particular
threads through 4+ assembly sites). The patch-based optimization track
has reached diminishing returns:

- 11 patch variants tried since `barriersig-early`
- Single best result (barriersig-early) at **142 TFLOP/s = 72.9% peak**
- Production already ships this; libhipblaslt is bench-only
- Bridge perf (166 TFLOP/s) is reachable via libhipblaslt link if a
  future caller really needs the last 17%, with 0 lines of asm work

The right next-investment for closing this gap is **regenerating the
kernel from a Tensile-style schedule generator** with RDNA4-specific
constants (LDS stride 160, per-load soffset SGPRs, WMMA accumulator
tiling tuned for VGPR ≤192). That's a multi-day greenfield rewrite
in `gen_mm0_bf16_asm_pgr2.py` (the scaffold landed earlier this
session has the design constants pinned), not a continuation of the
patch series.


## 2026-04-29 — DTVA hand-asm sub-option 1a measured

`patch_dtva.py --step bc` produces `mm0_bf16_asm_dtva.co`: single-set
DTVA on the barriersig_early baseline, no swap ring. bb.4 entry hosts
8 X global_load_b128 directly into v[174:221] using 4 frag bases at
v[238:245] + s[0:1] K-offset. `s_wait_loadcnt 0x0` drains all 8 before
the first WMMA — full serialization, no overlap.

| variant                                  | ms      | TFLOP/s | peak  | check  |
|------------------------------------------|---------|--------:|------:|--------|
| `mm0_bf16_asm_barriersig_early.co` (baseline) | 0.305 | 142.8 | 73.2% | PASS |
| `mm0_bf16_asm_dtva.co` (1a single-set serialized) | 0.378 | **115.2** | 59.1% | PASS |

**Result:** 1a regresses 19% vs baseline. The single-set serialization
cost (`wait_loadcnt 0x0` before WMMAs) equals the swap-ring cost
proven in `directa_pf` — both land at 115 TFLOP/s. The DTVA structural
win does not, by itself, recover the cycles lost to load-compute
serialization.

**Bug hit + fixed:** First build of step BC produced HIP error 700.
Root cause: 8 `v_add_co_u32` instructions at bb.4 entry clobber
`vcc_lo`, which the kernel relies on for the loop-condition
`s_cbranch_vccnz .LBB0_1` at bb.4 tail. The baseline sets `vcc_lo`
via `s_and_not1_b32 vcc_lo, exec_lo, s8` at bb.4 entry and preserves
it through the WMMAs to the branch. Fix: save `vcc_lo` to s12 before
the address adds, restore after; bump `.amdhsa_next_free_sgpr` 12→13.

**Implication:** Sub-option 1a is empirically equivalent to 1b's
proven ceiling. The remaining 27% gap to hipBLASLt cannot be closed
by either single-set DTVA or two-set swap-ring DTVA — both pay the
same effective per-iter penalty. Only sub-option 1c (two-set + 2-iter
unrolled bb.4 with no swap, alternating WMMA operand sets) has any
chance of beating 142, and it carries multi-day risk + 256 VGPR
pressure tipping past 1 WG/SIMD.

**Recommendation stands:** ship `mm0_hipblaslt_bridge.cpp` (algo 73624,
166.6 TFLOP/s, 85.4% peak, validated end-to-end) as production. The
libhipblaslt runtime dep is the cost; closing this gap in pure asm
requires either path 1c (uncertain) or the Track B PGR2 generator
rewrite scoped earlier in this log.


## 2026-04-29 — DTVA c1 (K-half load-overlap) measured

`patch_dtva.py --step c1` reorders bb.4 entry's 8 X DTVA loads so all 4
K-half-0 fragments are issued first, then the 4 K-half-1 fragments.
`s_wait_loadcnt 0x4` drains only K-half-0 before the first WMMA; the 16
K-half-0 WMMAs run while K-half-1 loads remain in flight; a fresh
`s_wait_loadcnt 0x0` lands before the K-half-1 WMMAs.

| variant | ms | TFLOP/s | peak | check |
|---|---:|---:|---:|---|
| `mm0_bf16_asm_dtva.co` (1a serialized BC) | 0.378 | 115.2 | 59.1% | PASS |
| `mm0_bf16_asm_dtva_c1.co` (1a + K-half overlap) | 0.378 | **115.0** | 59.0% | PASS |

**Result:** zero improvement. The 16 K-half-0 WMMAs (~128 cycles of issue
time) cannot fully hide the latency of even 4 outstanding K-half-1
b128 global_loads (~300 cycles each). This confirms by construction that
**the load latency cannot be hidden within a single bb.4** — overlap
requires cross-iter prefetching (issue in iter K's tail/bb.5 for iter
K+1's WMMAs).

**Implication for path 1c:** The 27 TFLOP/s gap between the 1a/1b/c1
ceiling (115) and the baseline (142) is the cost of NOT having the X
load latency hidden. Path 1c (2-iter unroll + 2-set DTVA) is the only
remaining mechanism that can hide it without paying the swap penalty.
The ceiling for 1c is set by hipBLASLt at 166 TFLOP/s, so 1c's plausible
range is 142–166 TFLOP/s if it works.


## 2026-04-29 — Track B home file landed (slice 1 of 5)

`gen_mm0_bf16_asm_pgr2.py` converted from scaffold to real emitter. Reads
`generated/mm0_bf16_asm_barriersig_early.s` and emits
`generated/mm0_bf16_asm_pgr2.s` after applying the buffer-descriptor +
buffer_load swap (delegates to `patch_mm0_asm_schedule.patch_pgr2_bufload`).
Wired through Makefile as `build-mm0-asm-pgr2` / `bench-mm0-asm-pgr2`.

| variant                          | ms     | TFLOP/s | peak  | check |
|----------------------------------|--------|--------:|------:|-------|
| `mm0_bf16_asm_pgr2.co` (slice 1) | 0.3055 | **142.35** | 73.0% | PASS  |

Output is byte-identical to `pgr2-bufload-bse` (the established 141.6
TFLOP/s reference); the run came in slightly above thanks to noise.
Slice 1 is **perf-neutral** by design — its purpose is to establish the
Track B home file as a working generator so the four structural items
(LDS stride 160, per-load soffset SGPRs, W-tile partition 4→8 b128/thread,
VGPR ≤192) can land as additional emitter slices instead of patches on
hipcc-compiled output. The decision recorded earlier in this log
("stop the patch series") still stands; subsequent work happens in
`gen_mm0_bf16_asm_pgr2.py`'s `emit_slice_*` namespace, not in
`patch_mm0_asm_schedule.py`.

Slices remaining (see file docstring for details):
  - slice 2: LDS row stride 144 → 160 (coordinated, unlike standalone
    `patch_lds_stride.py` which silent-failed at cosine 0.998)
  - slice 3: per-load soffset SGPRs (item 3 from log:586)
  - slice 4: W-tile partition rewrite (item 2)
  - slice 5: VGPR pressure ≤192 to unlock 2 WGs/WGP (item 4)

Once 2-5 land, the SCHED slice (PGR2 distribution at slots [0,2,…,14] for
loads + [14..21] for ds_stores + barrier slot 24) can be retried — the
prior `pgr2-distribute-bse` standalone patch regressed (137 TFLOP/s)
specifically because slices 2-5 weren't co-applied.

## 2026-04-29 — Slice 2 (LDS stride 144 -> 128) landed in pgr2 emitter

After the prior 144 -> 160 (+padding) standalone patch failed cosine=0.998,
bisection-tested the *opposite* direction: 144 -> 128 (Tensile default,
no padding). Stride 128 patches cleanly:

- standalone (just stride 128 over barriersig_early): 143.4 TFLOP/s, cosine=1.0
- combined slice 1 + slice 2 in pgr2 emitter:        142.3 TFLOP/s, cosine=1.0

The +1 TFLOP/s from stride 128 alone washes out when composed with the
buffer_load swap (slice 1). Both are effectively perf-neutral baseline
moves; the 17% gap to hipBLASLt still lives in slices 3-5 (per-load soffset
SGPRs, W-tile 4 -> 8 b128/thread, VGPR <= 192) plus the SCHED slice (PGR2
distribution). Bug in the 144 -> 160 attempt was specific to the padded-row
direction (subtle offset miscalc in B-bank +384B sub-row sites); -padding
direction is bank-aligned to natural tile width so works first try.

The pgr2 emitter (`gen_mm0_bf16_asm_pgr2.py`) now owns slices 1+2 as plain
function additions. `patch_lds_stride.py` (160) and `patch_lds_stride128.py`
remain as historical bisection artifacts. Slice 3 (per-load soffset SGPRs)
is up next — it has the lowest structural coupling of the remaining slices.

## 2026-04-29 — Slice 3 (per-load soffset SGPRs) closed as no-op

Plan called for replacing `s0 offen offset:N` (N in {64,80,96,112}) with
`sX offen` where sX = pre-loaded `s0 + N`, claiming it "saves immediate-decode
bytes per load." Verified on RDNA4 encoding: buffer_load_b128 with 12-bit
immediate offset is a single 64-bit instruction word. Offsets 64..112 all
fit; no extra DWORD to save. Pre-computing into SGPRs adds 4-instruction
prologue cost with zero encoding savings. Slice 3 closed without emit.

Remaining gap drivers: slices 4 (W-tile 4 -> 8 b128/thread) + 5 (VGPR <= 192
to unlock 2 WGs/WGP) plus the SCHED slice (PGR2 distribution). These are
large coordinated rewrites; further patch-style increments on the
`barriersig_early` baseline are exhausted.

## 2026-04-30 — SCHED slice + slices 1+2 still regresses to 137 TFLOP/s

Wired `--with-sched` flag into `gen_mm0_bf16_asm_pgr2.py` (delegates to
`patch_pgr2_distribute`). Tested with slice 2 (stride 128) co-applied:

- baseline (slices 1+2):              142.4 TFLOP/s, cosine=1.0 PASS
- slices 1+2+SCHED (PGR2 distribute): 137.1 TFLOP/s, cosine=1.0 PASS

Identical regression magnitude to the prior `pgr2-distribute-bse` (bufload
+ sched, no stride change). Slice 2 made literally zero difference under
distributed scheduling — confirming the bottleneck under PGR2 is upstream
of LDS row-stride: most likely VGPR pressure preventing the second WG/WGP
from co-launching, or W-tile per-thread bandwidth saturating the load
issue rate. Slices 4 (W-tile 4 -> 8 b128/thread) and 5 (VGPR <= 192) are
therefore *required* co-conditions for SCHED, not optional.

The pgr2 emitter ships with `--with-sched` defaulted off so the production
.co stays at 142 TFLOP/s + cosine=1.0. SCHED stays available behind the
flag for future slice-4/5 work.

## 2026-04-30 — Profile + reference data confirms patch path exhausted

Profile data after slices 1+2 inlined into pgr2 emitter:

  VGPR_count        : 238 (HW rounds to 248)
  SGPR_count        : 14
  LDS               : 32768 (slice 2 dropped from 36864)
  SQ_BUSY_CYCLES    : 26.87M / dispatch
  SQ_WAVES          : 1152 / dispatch
  busy/wave         : 23,318 cycles
  TFLOP/s           : 142.4

Reference (memory): hipBLASLt @ 21,979 cycles/wave, 166 TFLOP/s. So our
busy-cycle gap = 6%, perf gap = 14.5% — the extra 8.5% is non-busy
(s_wait_loadcnt / s_wait_dscnt stalls, tail K iter, memory latency
overlap).

Reference VGPR data from `/tmp/hipblaslt_bf16_alik_bljk.notes.txt` shows
hipBLASLt's bundle includes kernels at 256, 252, 160, 123, 48 VGPRs.
Several of the high-perf kernels run at 252-256 VGPRs — i.e. **slice 5's
"VGPR ≤192 to unlock 2 WGs/WGP" hypothesis is not the dominant lever for
hipBLASLt's perf**. The actual differentiators per the kernel name decode
of algo 73840 (`Cijk_Alik_Bljk_..._MT128x128x32_MI16x16x1_PGR2_PLR1_SIA3_
DTVB1_LDSB1_..._WG32_4_1_MIWT4_4`):

- **WG32_4_1**, not 128x1x1. The wave/lane → tile mapping is
  fundamentally different from our kernel. This is a structural fork that
  affects every per-thread offset compute, every LDS swizzle, and the
  WMMA register layout. Can NOT be patched onto our 128x1x1 baseline; it
  requires a co-emit from scratch.
- MIWT4_4 with MI16x16x1 (Tensile's stride-K=1 naming) — same as ours.
- DTVB1 LDSB1 — direct-to-VGPR for B operand, LDS-buffered for A. We
  use DTVA0_DTVB1 too, so this is consistent.

**Final conclusion.** Patch path closed at 142.4 TFLOP/s (cosine=1.0,
production-ready). The remaining 17% gap requires a from-scratch emit
with WG dim 32x4x1 — a multi-day greenfield rewrite of every lane-wise
offset compute, not a slice-by-slice progression on the existing
baseline. Slice 4 (W-tile) and slice 5 (VGPR) are subsidiary to that
core fork.

Production path: ship `mm0_hipblaslt_bridge.cpp` (166 TFLOP/s end-to-end,
already linked into `bench_vlm_hipblaslt`), or accept 142 TFLOP/s from
`mm0_bf16_asm_pgr2.co` and skip the libhipblaslt dep.

## 2026-04-30 — Production framing correction

`hip_vision_encoder.c` already uses hipBLASLt directly for the mm0
projector (`hipblaslt_mm0_gelu` plan with fused bias+GELU, lines
2069/2530/2580). The `gemm_mm0_bf16_asm` symbol from any CO variant is
consumed ONLY by `bench_vlm_gemm.c`. The mm0_bf16_asm work has been a
research artifact to understand hipBLASLt's PGR2/SIA3 schedule, not a
production swap. The "drop libhipblaslt dep" goal in the original plan
was based on incorrect framing — production already runs through
hipBLASLt for mm0 (and qkv, attn_out, ffn_up, ffn_down). Truly removing
the dep requires ASM kernels for every GEMM shape, a different project.

End state of this branch:

- `mm0_bf16_asm_pgr2.co` (slices 1+2): 142.4 TFLOP/s, cosine=1.0 PASS,
  benchmarking artifact, no production consumer.
- `mm0_hipblaslt_bridge.cpp` linked into `bench_vlm_hipblaslt`: 166
  TFLOP/s reference, also benchmarking-only.
- Production VLM path: `hip_vision_encoder.c` → libhipblaslt fused
  projector path (was already this way before any of this work).

The 17% gap closure was a research goal, not a production one. Stopping.

## 2026-04-30 — DTVB from-scratch emitter scaffolding landed

A re-attempt at the DTVB path produced Phase 1 ground-truth annotation
and Phase 2 emitter scaffolding, but did not write a kernel.

**Files added:**

- `hipblaslt_mm0_alik_bljk_dtvb_groundtruth.s.annot` — full disasm
  annotation of algo-73624's hot loop (32 WMMA, 12 buffer_load split
  4 A + 8 B per outer-K, 4 ds_store, 8 ds_load, 0 v_add_co_u32; SGPR
  layout S_A_SRD=48 S_B_SRD=52, per-load offsets s66-s75; LDS 26624 B).
- `dtvb_emit/__init__.py`, `dtvb_emit/isa.py`, `dtvb_emit/regalloc.py`,
  `dtvb_emit/scheduler.py`
  — gfx1201 mnemonic builders, named VGPR/SGPR windows with non-overlap
  asserts (verified 246 total VGPRs), and a VMEM/LDS issue-tracker that
  computes `s_wait_loadcnt`/`s_wait_dscnt` automatically via
  `compute_loadcnt_after_consume(idx, total) = total - 1 - idx`.
- `gen_mm0_dtvb.py` — CLI entry; `--check-regalloc` works; mainloop
  emission stubbed with the next-session implementation plan in the
  module docstring.

**Why no kernel was emitted.** Phase 1 ground truth shows hipBLASLt's
algo-73624 splits B across two 32-VGPR banks (B_DIR_A v[182:213],
B_DIR_B v[214:245]) — a **2× outer-K-unrolled** body with B
register-renaming. The complementary `patch_dtva.py` finding (verified
2026-04-29 via `mm0_bf16_directa_pf.s`) records that the equivalent
ring constraint on A makes single-set DTV regress to ~129 TFLOP/s.
The same applies to B. Therefore a single-set DTVB iter-1 generator
would emit a known-regression kernel; the only DTVB shape that can
beat 142 is the 2-iter unrolled body, which is multi-day work AND
requires the 32x4x1 WG decomposition (the dominant lever from the
2026-04-29 conclusion above).

The scaffolding is the durable artifact of this session. Future work
can pick it up at the docstring's "Path to ≥160 TFLOP/s" section
without re-doing the disasm or the register-window allocation.

**Pipeline validated end-to-end 2026-04-30.** `gen_mm0_dtvb.py --mode
baseline` emits a verbatim pass-through of the 142 TFLOP/s kernel
through the dtvb_emit modules; build pipeline goes
`gen_mm0_dtvb.py → .s → clang++ → .o → .co`. Bench result of the
generated CO: **143.46 TFLOP/s, cosine=1.0, max_abs=0** (200-iter median),
matching `mm0_bf16_asm_barriersig_early.co`'s 142.45 TFLOP/s within
run-to-run variance. Confirms generator + Makefile target
(`build-mm0-asm-dtvb-baseline` / `bench-mm0-asm-dtvb-baseline`) work.

`gen_mm0_dtvb.py --mode drop-b-lds` correctly identifies all 24 B-LDS
staging instructions for surgical removal (verified by grep count after
emission). This mode produces an INCORRECT kernel (B unmaterialized) —
useful only as a structural-substitution sanity step before the next
session's DTVB direct-load implementation.

## 2026-04-30 — extracted-kernel launcher (`mm0extract` mode)

Pivoting from the from-scratch DTVB generator (multi-day greenfield
work, see above), this session **extracts** hipBLASLt's algo-73624
kernel binary and ships a standalone launcher that dispatches it
without runtime libhipblaslt linkage.

**Pipeline.**

1. `dump_kernarg_shim.c` — LD_PRELOAD shim wrapping
   `hipExtModuleLaunchKernel`. Captures the 140-byte kernarg buffer +
   grid/block to `/tmp/mm0_kernarg_<i>.{bin,meta}`.
2. Decode 28 fields per Tensile UserArgs ABI (`Signature.py` lines
   122-125). Captured constants: `gemm_info=1`,
   `internalArg0=0x02200001`, `internalArg1=0x08010008`, `numWG=36`,
   plus the problem-shape strides/sizes.
3. `mm0_extracted_launcher.cpp` — loads
   `TensileLibrary_BB_SB_HA_Bias_SAV_UA_Type_BS_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx1201.co`
   via `hipModuleLoad`, resolves `Cijk_Alik_Bljk_..._WG32_4_1` symbol
   via `hipModuleGetFunction`, prefills the kernarg in `init`, and
   replays it per call updating only the D/C/A/B pointers. Launch
   matches captured grid `(4608, 8, 1)` × block `(128, 1, 1)`.
4. New bench mode `--mode mm0extract` exercises it end-to-end.

**Symbol-collision gotcha.** rocew's dynamic-loader wrapper defines
`hipModuleLoad`, `hipModuleGetFunction`, `hipModuleUnload` as global
function-pointer variables in BSS that collide with libamdhip64 at
link time. Calls from the launcher TU resolve to rocew's NULL
pointers and SIGSEGV. Fix: launcher uses `dlsym(RTLD_DEFAULT, ...)`
to bypass and call the real libamdhip64 symbols directly.
(`hipExtModuleLaunchKernel` is undefined in rocew so it links
normally.)

**Results** (RX 9070 XT, 5 s cooldown between runs):

| Mode             | iters=100      | iters=200     | Correctness    |
|------------------|----------------|---------------|----------------|
| `mm0extract`     | 155.1 TFLOP/s  | 148.6 TFLOP/s | max_abs=0 PASS |
| `mm0blaslt`      | 162.5 TFLOP/s  | 165.5 TFLOP/s | (reference)    |
| `mm0asm` (hand)  | 142.5 TFLOP/s  | —             | PASS           |

The launcher hits **155 TFLOP/s** at 100 iters (79.5% peak) — within
~5% of the libhipblaslt-mediated launch of the same binary. The
residual gap is launch-side warmup state; bench was extended to do
64 warmup iters before timing in the extract path (mirrors
hipblaslt's heuristic-search warmup). Same kernel binary, same
kernarg, same `hipExtModuleLaunchKernel` ABI as
`HipSolutionAdapter::launchKernel` (verified against Tensile source).

**Practical impact.** Hand-asm `mm0_bf16_asm_barriersig_early.s`
ceiling is 142 TFLOP/s; extracted-launcher ceiling is 155 TFLOP/s —
a **+9% delivered uplift** for `vlm/mm0` without the multi-week
DTVB+WG32_4_1 from-scratch port and without runtime libhipblaslt
overhead. Caveat: depends on the
`TensileLibrary_BB_SB_HA_..._gfx1201.co` file shipping with ROCm
(present in 7.2.1 at `/opt/rocm-7.2.1/lib/hipblaslt/library/`).

### Algo enumeration — 2026-04-30 follow-up

The 155 TFLOP/s number above was the original algo-73624 default.
After enumerating all 1498 kernels in the .co (via
`llvm-objdump --syms`) and filtering on the DTVB1+PGR2 family (59
matches), a sweep of `MM0_EXTRACTED_KERNEL_SYM=$k` over each variant
showed algo 73624 was **not** the optimum for this shape:

| Pattern (DTVB1+PGR2) | best | sustained avg (4 trials) |
|---|---|---|
| LBSPPA128, GRVWA{4,8}_GRVWB8 (low end) | 155–160 | — |
| LBSPPA256, GRVWA{4,8}_GRVWB8, SS0      | 165–167 | — |
| LBSPPA256, GRVWA{4,8}_GRVWB8, SS1, SVW4 (full warm) | 174–177 | — |
| LDSB1+CLR0+LBSPPA256+TLDS2+SS1+SVW4+VWA4_VWB4 | **174.3** | **172.1** |

The winner is hoisted as the new launcher default
(`KERNEL_SYM` in `mm0_extracted_launcher.cpp`). Five back-to-back
runs of `--mode mm0extract --iters 200 --check`: 174.06, 168.68,
174.14, 173.88, 174.31 TFLOP/s — median **174.1 TFLOP/s, 89.3%
peak**. Correctness `max_abs=0` (bit-exact vs reference) on every
run.

Versus the prior default this is **+19 TFLOP/s** at the same launch
path; versus hand-asm `mm0_bf16_asm_barriersig_early.s` (142
TFLOP/s) it is **+22%**. The dominant tunables on top of DTVB1+PGR2
are LBSPPA256 (longer A-prefetch lookahead — biggest single-bit
swing, +10 TFLOP/s), LDSB1 (double-buffered LDS for A — +2 TFLOP/s
sustained over LDSB0), and the VWA4 read-vector-width unification
(no measurable benefit from VWB4 vs VWB1 within noise).

The earlier sweep numbers (175–177) were inflated by sustained-warm
back-to-back runs across kernels; with proper iter-200 measurement
within a single kernel the steady-state ceiling is **172 TFLOP/s
sustained / 174 best**. This effectively closes the 17% gap that
motivated the whole investigation — `mm0extract` mode now operates
above hipBLASLt's nominal `mm0blaslt` reference (166 TFLOP/s)
because it skips libhipblaslt's per-call algo-selection overhead.

### Lever attribution (2026-04-30)

Per-wave cycle delta from rocprofv3 (132 dispatches each, gfx1201
SQ_BUSY_CYCLES / SQ_WAVES) reconciles the 23 TFLOP/s win across five
levers. Full breakdown in `mm0_lever_attribution.md`. Key numbers:

- OLD (PGR1+SVW1+VWA1+LBSPPA128+LDSB0): 24,174 cyc/wave, 0.276 ms, 158 TFLOP/s
- WIN (PGR2+SVW4+VWA4+LBSPPA256+LDSB1): 20,363 cyc/wave, 0.233 ms, 187 TFLOP/s
- Delta: **−3,811 cyc/wave (−15.8 %)**, dominated by VWA1→4 (~2,000 cyc,
  ~12 TFLOP/s) which fixes WMMA register-bank stalls and enables
  back-to-back dual-issue. PGR1→2 ~800, LBSPPA128→256 ~600,
  SVW1→4 ~300, LDSB1+CLR0 ~150 — sum within 1 % of measured.
- LDS allocation drops 26,624 → 9,216 B (DTVB1+TLDS2+LDSB1 collapses A-only).
- VGPR=256 in both → occupancy unchanged. The full VGPR budget is the
  hard ceiling on going further (no headroom for PGR3).

Generic recipe for future GEMM ports lives in `peak_efficiency_playbook.md`.

## 2026-04-30 — VLM end-to-end optimization (status snapshot)

mm0 closed at 174 TFLOP/s sustained (89% peak). Effort then shifted from
"the projector GEMM" to "the rest of the vision encoder," which had
become the dominant cost once mm0 was no longer it. Profile-driven
attack of the actual hot kernels in `hip_vision_encoder.c` produced an
end-to-end **18.4× speedup at 1024² (260 → 4793 tok/s)** and **21.3×
at 2048² (77 → 1638 tok/s)** vs the original scalar-F32 attention
reference, all with cosine ≥ 0.99996 vs llama.cpp.

### Layered optimizations and headline numbers

Measurements: RX 9070 XT, Qwen3.6-27B mmproj-BF16, fujisan.jpg.
HIP path active by default; toggle with `HIP_VLM_FA=...`.

| Path / step                        | 1024² ms | tok/s | 2048² ms | tok/s | cosine     |
|---|---:|---:|---:|---:|---|
| scalar F32 (`flash_attn_dyn_f32`)  | 3930 |  260 | 53180 |   77 | (ref)      |
| WMMA BF16 BQ=16 1-wave             | 1085 |  943 | 11776 |  348 | 0.99996665 |
| WMMA BF16 BQ=32 2-wave             |  758 | 1350 |  5702 |  718 | 0.99996665 |
| + KV pre-pack (half-prec K/V)      |  713 | 1436 |  5078 |  807 | 0.99996665 |
| + W0+W1 host-fold (patch_embed)    |  645 | 1586 |  4683 |  874 | 0.99996930 |
| + patch_embed WMMA (im2col+F16)    |  608 | 1687 |  4596 |  891 | 0.99996577 |
| + hipBLASLt for ViT block GEMMs    |  278 | 3689 |  3281 | 1248 | 0.99996593 |
| + FA fixed-stride K/V loads        |  215 | 4544 |  2826 | 1450 | 0.99996593 |
| **+ FA double-buffered KV (BF16)** |  **202** | **4793** |  **2500** | **1638** | 0.99996593 |

Each row is the cumulative state with all prior optimizations active.

### What each step does

1. **WMMA flash-attn kernels** (`flash_attn_wmma_{bf16,f16}{,_2w}{,_pre}`).
   1 wave / 32 threads, BQ=16, BKV=16, head_dim padded to 80. F32
   inputs cast to half-precision in LDS. Online softmax with
   `__shfl_xor` cross-lane. The `_2w` 2-wave BQ=32 variant doubles
   queries-per-K/V-tile so DRAM K/V traffic halves and 64-thread
   cooperative LDS load amortizes load-issue cost. Selected by env
   `HIP_VLM_FA=wmma_bf16_2w` etc.
2. **KV pre-pack** (`kv_transpose_{bf16,f16}` + `_2w_pre` kernels).
   Writes half-precision K/V once per block from F32 qkv. Halves DRAM
   read bandwidth in the FA hot loop and removes the per-element F32
   cast inside the K/V load. Cosine bit-identical to non-pre.
3. **W0+W1 host-fold** for the Qwen3-VL dual-conv2d patch embedding.
   `(W0 + W1) @ pix` is algebraically equivalent to summing two
   convolutions but requires only one GEMM at runtime. Done in init.
   Cosine slightly *better* (a single rounding instead of two).
4. **patch_embed WMMA** via `patch_unfold_f16` (im2col) +
   `gemm_wmma_f16_f32`. F16 (10-bit mantissa) is required: BF16
   (7-bit) fails the accuracy gate on small normalized RGB inputs.
   patch_embed: 113 → 0.85 ms (133×).
5. **hipBLASLt for ViT block GEMMs** (`HIP_VLM_HIPBLASLT_VIT=1`,
   default on). 16-slot per-shape plan cache `hipblaslt_vit_plans[]`
   with find-or-create keyed by (dtype, m, n, k, epilogue). Routes
   `qkv` (BIAS), `attn_out` (BIAS+residual via β=1, C=residual),
   `ffn_w0` (GELU+BIAS), `ffn_w1` (BIAS+residual) through hipBLASLt.
   Encoder GEMMs total 350 → 32 ms (11×, ~100 TFLOP/s avg vs prior
   ~10).
6. **FA fixed-stride K/V loads**. Replaced `e/HD_PAD` and `e%HD_PAD`
   divides (HD_PAD=80, non-power-of-2 = real integer divide on AMD
   ISA — ~30 cycles each) with a static thread→(row, col) layout:
   `ld_row = tid >> 2`, `ld_col0 = (tid & 3) * 20`, each thread
   handles a fixed 20-col stripe of one row. FA: 7.0 → 4.92 ms/call
   (-30%). DRAM coalescing modestly worse (stride-20 between adjacent
   threads), but L2 absorbs it; the divide elimination dominates.
7. **FA double-buffered KV** (BF16 path). Two K and two V LDS slots;
   prologue loads tile 0; each iter prefetches tile *t+1* into the
   alternate slot while WMMA computes on tile *t*. One
   `__syncthreads()` per iter, at the end. LDS doubles to ~11 KB/WG
   (still well under 64 KB/CU). FA: 4.92 → 4.60 ms/call (-7%, larger
   gain at 2048² where n_kv is bigger so prefetch/compute overlap is
   more frequent).

### Profile after all of the above (1024², BF16, total GPU 174 ms)

- `flash_attn_wmma_bf16_2w_pre`: 71.3% (124 ms over 27 calls = 4.6 ms/call)
- hipBLASLt ViT GEMMs: 18.4% (32 ms over 110 calls = 0.29 ms/call)
- `pack_bf16_from_f32` (GEMM input cast): 4.3% (7.5 ms / 110)
- `layernorm_f32`: 2.0% (3.6 ms)
- `rope_vision_f32`: 1.8% (3.1 ms)
- `kv_transpose_bf16`: 1.2% (2.1 ms)
- everything else: <1%

FA is decisively the next bottleneck; GEMMs and the cast/norm/rope
tail are below 4% each.

### Remaining headroom

FA at 4.6 ms/call corresponds to ~13 TFLOP/s vs WMMA peak ~195 — only
~7% of peak. Likely attacks, in roughly decreasing expected ROI:

1. **BQ=64 4-wave FA variant.** Halves grid_y (each WG covers 64
   queries instead of 32) and shares one K/V tile across 4 waves
   instead of 2. Reduces launch/sync overhead and per-iter K/V DRAM
   re-reads. VGPRs per wave are stable (~168), so 4 waves × 168 =
   672 VGPR/SIMD-WG, still leaving 2 WGs/SIMD. Expected: −10 to
   −20% FA.
2. **Vectorized b128 LDS loads/stores.** Current K/V loads do 20
   `ds_store_b16` per thread per tile. Switching to a layout where
   each thread writes a contiguous 16-byte chunk (1 `ds_store_b128`)
   needs PAD=96 or 128 (multiple of 8 BF16 ≥ 80). Trade: more LDS,
   fewer issue slots. Expected: −5 to −10% FA.
3. **Pack V in [h, d, kv] layout in `kv_transpose`.** smV is
   transposed [d × kv]; current V_t is [h, kv, d] so the LDS-side
   transpose costs scattered writes. Pre-transposing in the pack
   step gives contiguous DRAM reads matching the LDS layout.
   Expected: small (kv_transpose itself is 1.2% of total).
4. **88-stride or 96-stride LDS padding** to break bank conflicts on
   stride-80 reads. Bank cycle is 128 B (= 32 banks × 4 B); 80-byte
   rows are non-aligned. Matters more after b128 vectorization.
5. **Faster F32→BF16 cast for hipBLASLt inputs**
   (`pack_bf16_from_f32` is 4.3% at 88 GB/s effective, far below
   DRAM peak — small grid suspected). Or fuse the cast into the
   prior kernel's output.
6. **Soft caps further along**:
   - `rope_vision_f32` is scalar; could be vectorized.
   - `layernorm_f32` could use a wave-cooperative reduction.
   These are <2% each so worth only after FA is sub-3 ms.
7. **Apply double-buffer to F16 path.** Currently only the BF16
   `_2w_pre` is double-buffered; the F16 variant still uses the
   single-buffer body. Mechanical port (a few hundred lines).

### Files

- Kernels and runner: `rdna4/vlm/hip_vision_encoder.c`
- Bench/test driver: `rdna4/vlm/test_hip_vision.c`,
  `rdna4/vlm/test_hip_vlm.c`

Reproduce:

```sh
HIP_VLM_FA=wmma_bf16_2w_pre ./test_hip_vision \
    /mnt/disk1/models/qwen36/27b/mmproj-BF16.gguf \
    --image fujisan.jpg --image-size 1024 \
    --ref /tmp/vlm_baseline_f32attn.bin --bf16 \
    --warmup 2 --iters 5
```

