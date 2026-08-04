# hetero/ds4f — DeepSeek-V4-Flash 0731 on Threadripper 1950X + RX 9070 XT

Goal: 10+ tok/s single-stream decode at native quantization quality (FP8-E4M3
dense + MXFP4 experts, no requantization). Baseline to beat is llama.cpp on the
same box: 7.33 tok/s on the full MXFP4 model, 8.57 tok/s on a Q3_K_M requant
(see `../../../llama.cpp/da4f.md`).

Split target: the FP8 dense path (MLA + shared expert, about 5.7 GB) and the
replicated BF16 vocabulary head are GPU-resident on the 9070 XT; routed
MXFP4 experts, router, and embedding stay on the CPU.
The device-bank capacity, full multi-layer model callback, and two-slot
asynchronous shared-expert path are now validated. The existing CPU runner
remains CPU-default; the native x86 staged-session adapter in `server/` can
also attach that HIP bank with the explicit `DS4F_HIP=1` opt-in.

## Hardware

| | |
|---|---|
| CPU | Threadripper 1950X, 16C/32T, AVX2 (Zen1 splits every 256-bit op into 2x128) |
| RAM | 188 GiB, single NUMA node, quad-channel DDR4 |
| GPU | RX 9070 XT, gfx1201, 16 GiB |
| Model | `/mnt/disk1/models/ds4f-0731`, 48 safetensors shards, 156 GB |

Per-expert weights from the safetensors headers: `w1[2048,2048] I8` +
`w1.scale[2048,128] F8_E8M0`, `w3` likewise, `w2[4096,1024] I8` +
`w2.scale[4096,64]` = **13.35 MiB per expert**. With 6 of 256 experts active
over 43 layers that is **3.44 GB of CPU DRAM traffic per decoded token**, so
10 tok/s requires 34.5 GB/s sustained.

## S0 results — roofline (2026-08-03)

`make && ./build/bench_expert_bw --mode <m> [--i8seq|...] --gib 90 --threads 16`

Measured ceilings:

| Reference | GB/s |
|---|---:|
| STREAM triad (2R+1W) | 36.0 |
| Read-only gather over the same random 13.35 MiB expert blocks | **48.1** |

Full MXFP4 SwiGLU expert compute, 16 threads, 90 GiB region:

| Kernel | GB/s | tok/s (expert path only) |
|---|---:|---:|
| f32 activations, 8-row group (v1, direct SVE port) | 23.8 | 6.90 |
| f32 activations, 8-row, ILP-restructured (v2) | 22.5 | 6.53 |
| W4A8 int8 activations, 8-row, 128-bit | 33.4 | 9.69 |
| W4A8, 8-row, 256-bit paired blocks | 32.0 | 9.26 |
| W4A8, 8-row, hoisted scale prologue | 32.9 | 9.53 |
| f32 activations, **sequential single row** | 31.0 | 8.98 |
| **W4A8, sequential single row** (`--i8seq`) | **43.3** | **12.55** |

**Gate PASSED.** 43.3 GB/s is 90% of the 48.1 GB/s read ceiling.

### What actually mattered

Two independent findings, in order of size:

1. **Row grouping was thrashing the prefetchers** (+10 GB/s). The 8-row kernel
   is a direct port of SVE's `matvec_mxfp4_8row`, where grouping amortizes a
   16 KiB f32 activation across eight rows. It makes each thread drive eight
   concurrent memory streams 2 KiB apart; at 16 threads that is 128 weight
   streams plus 128 scale streams, well past what Zen1's L2 prefetchers and
   64-entry miss queue can track. Under W4A8 the activation is int8 and
   permanently L1-resident, so there is nothing to amortize -- walking one row
   start to finish gives each thread a single sequential stream, which is what
   the hardware prefetcher wants.

2. **The f32 dequant chain is the compute wall** (+12 GB/s). Unpacking 32
   nibbles to f32 costs 4x `vpmovsxbd` + 4x `vcvtdq2ps` plus the nibble
   shuffles -- ~19 instructions of overhead for 4 useful FMAs, and Zen1
   double-pumps all of them. Keeping the dot product in integers via
   `vpmaddubsw`/`vpmaddwd` removes every float conversion from the inner loop.
   Sequential f32 reaches only 31.0 GB/s (8.98 tok/s) and misses the gate, so
   **W4A8 is required, not optional**.

Things that did *not* help, and why they are worth not retrying: 256-bit
paired-block processing (Zen1 costs the same uops and the cross-lane scale
build adds more); hoisting the scale conversion into a vectorized prologue (the
scale bookkeeping was never the limiter); restructuring for FMA latency (the
loop is throughput-bound, not latency-bound). SMT hurts -- 32 threads measured
42.7 GB/s read vs 47.6 at 16, matching llama.cpp's own finding that 12 threads
beat 16 and 24.

### The W4A8 trade

`ds4f_matvec_mxfp4_*_i8` quantizes the **activation** to int8 per 32-element
block -- the same granularity the weight's E8M0 scale already uses. Weights
stay exact MXFP4. The identity used is

    sum_k w_k x_k  =  e8m0 * xscale_b * ( sum_k wu_k xq_k  -  12 * sum_k xq_k )

with `wu = value + 12` unsigned so it can be the u8 operand of `vpmaddubsw`.
The debias term depends only on the block, not the row, so it is computed once
per matvec. Overflow is safe: 24 * 127 * 2 = 6096, inside int16.

On the synthetic random-weight smoke test (`--mode verify`) this shows up to
2.4e-2 relative error, but uniformly random nibbles make the true dot a random
walk, which is the worst case for activation quantization -- real trained
weights are far better conditioned. This is the same trade ggml makes for every
K-quant, and the DS4F A64FX path already has an int8-activation dense mode
(`DS4F_Q8_PV`). **The real-token gate now compares exact-f32 and W4A8 model
logits**, and if W4A8 turns out to cost measurable quality, the fallback is to
keep f32 activations on the `w2` down-projection only (it is 1/3 of the traffic,
so the cost would be roughly 12.6 -> 11.4 tok/s).

### Projection

The GPU dense path plus the BF16 head is 6.741 GB resident and overlaps the CPU
expert phase. The final safe EP=8 mechanical harness measures **12.40 tok/s**
short context and **10.53 tok/s** at a warmed 4k position after the cleanup
(10.33 tok/s on the preceding check); a complete EP=1 run remains the
quality-valid end-to-end measurement.

## S1 results — x86 port (2026-08-03)

The DS4F model implementation (`common/ds4f.h`, `common/ds4f_impl.h`) now builds
and runs on x86 without forking it away from the A64FX runners.

### What was done

- **Six AVX2 decode matvec kernels** in `common/ds4f_matvec_avx2.h`, included by
  `ggml_dequant.h` when `__ARM_FEATURE_SVE` is absent: BF16, BF16_PV, Q8_PV
  (`vpmaddubsw`+`vpsignb`, no VNNI on Zen1), FP8-E4M3, and MXFP4 in both
  exact-f32 and W4A8 form. All verified against independent scalar references in
  `test_ds4f_kernels.c` -- BF16, BF16_PV and FP8 are bit-exact, Q8_PV and
  MXFP4-f32 within 1e-6 relative.
- **Portable stand-ins** for the 15 SVE helper primitives
  (`common/ds4f_kernels_x86.h`) and scalar twins for six SVE workers
  (compressor/indexer matvecs and scoring).
- **Arch guards** around the remaining SVE-only code. The mHC SVE paths were
  already opt-in (`DS4F_HC_SVE`, default 0) with scalar twins. On x86,
  `ds4f_gemm` now uses the AVX2 row kernels for the supported BF16/FP8/MXFP4
  prefill types, including two-token BF16/FP8 weight-reuse microkernels and
  multi-GEMM dispatches that share one pool barrier across independent
  projections. Routed prefill now stages all local expert assignments in one
  slab and batches every expert's gate/up, SwiGLU, and down phase. The exact
  sliding-window prefill attention has an AVX2
  widen/dot/AXPY worker. Unsupported SVE-only tile formats still fall back to
  the correct token path.
- **`ds4f_relax` and `ds4f_pin`** given x86 forms (`pause`; one thread per
  physical core). The ARM `yield` was an assembler-only failure that
  `-fsyntax-only` did not catch.
- **`DS4F_STAGE_NOCOPY`** in `a64fx/llm/ds4f_stage.c`: emits a manifest whose
  offsets point into a virtual concatenation of the original safetensors shards,
  which `ds4f_blob_open` builds by reserving one address range and `MAP_FIXED`ing
  each shard into it. Every consumer still sees a flat `blob + off`. Staged the
  real 156 GB model in **0.1 s with zero bytes copied** (67,612 tensors: 1,564
  dense + 66,048 expert = 256 x 43 x 6).

### The A64FX build is intact

`make sve-check` syntax-checks the `__ARM_FEATURE_SVE` branch with
`clang --target=aarch64-linux-gnu -march=armv8.2-a+sve`, no Fujitsu toolchain
needed. Verified it genuinely compiles that branch by confirming a deliberate
error inside an SVE-guarded region is caught. Run it after any change to these
headers. The diff touches 3 existing lines; everything else is additive.

### Measured (LLM_THREADS=16, DS4F_EXACT=1, real weights)

| Model slice | Arena | Resident | ms/tok | experts phase |
|---|---:|---:|---:|---:|
| ep_size=8 (27 GB) | 26.6 GB | yes | 580 | 13.6 ms |
| ep_size=2 (82 GB) | 76.2 GB | yes | 628 | 55.9 ms |
| **ep_size=1 (full, 156 GB)** | 155.4 GB | **no** | **4228** | **2505 ms** |

Thread count matters enormously: the runner defaults to A64FX's
`LLM_THREADS=48`, which oversubscribes this 32-thread part and, with spin-wait
barriers, cost 5.4x (3150 -> 580 ms/tok).

### Open blocker: the full model does not fit the arena

At resident sizes the expert path runs at ~30 GB/s, in line with S0. At
ep_size=1 the experts phase blows up 45x (2505 ms vs the ~112 ms that rate
predicts) with 42.5 M minor faults and max RSS 104 GB against a 144 GB arena.

The cause is that `ds4f_load_real` **copies** every tensor into one anonymous
arena, so the full model needs 155 GB of anonymous pages plus page cache for the
156 GB source -- more than the 188 GB host. Nothing is wrong with the kernels.

The fix is to **reference the mapped shard bytes directly instead of copying**
for the MXFP4 experts, which are 147 GB of the total. See the next section --
note that MXFP4 experts turned out NOT to be a straight byte copy at load, so
this needed kernel work, not just a loader change.

## Zero-copy expert tensors (2026-08-03)

`ds4f_copy_worker` does not merely copy MXFP4 experts into the arena -- it
**repacks** them, in two ways:

1. **nibble permutation** -- on disk, byte j holds elements 2j (low nibble) and
   2j+1 (high); the arena form wants byte j to hold elements j and j+16.
2. **scale adjustment** -- the E8M0 byte e becomes `(e ? e-1 : 0)`, i.e. x0.5,
   because the e2m1 code table `ds4f_kvalues_mxfp4_f32` is the 2x form.

So the experts could not simply be pointed at. Instead the x86 kernels now
consume the **on-disk** bytes directly (`matvec_mxfp4_1row_{f32,i8}_raw`),
absorbing both differences without touching the weights:

- the **activation** is permuted once per matvec into `[evens | odds]` per
  32-element block, which is exactly the order the on-disk nibbles unpack to;
- the scale byte is decremented in-kernel, reproducing the repack's arithmetic
  exactly -- including its flush of e<=1 to +0.0, which `ggml_e8m0_to_fp32_half`
  does *not* do (it keeps a denormal there).

Expert tensors then point straight into the mapped safetensors shards, and the
mapping is handed to the model (`m->blob_map`) to outlive the load. Enabled by
default when the manifest is `DS4F_STAGE_NOCOPY` and the kernels are available;
`DS4F_ZEROCOPY_EXPERTS=0` restores the repack path.

### Result (full model, ep_size=1, LLM_THREADS=16)

| | repack into arena | zero-copy | |
|---|---:|---:|---|
| arena | 155.4 GB | **8.21 GB** | 19x smaller |
| load time | 377 s | **83 s** | 4.5x |
| decode, 8 tok (cold) | 4228 ms/tok | **1005 ms/tok** | 4.2x |
| decode, 64 tok (warm) | -- | **721 ms/tok** | |
| **experts phase (warm)** | **2505 ms** | **143 ms** | **17.5x** |

The experts phase now runs at 24 GB/s, in line with the ~30 GB/s measured at
resident sizes in S1 and no longer memory-thrashing.

### Correctness

- `make test` compares the raw kernels against the repacked kernels on bytes put
  through `ds4f_copy_worker`'s exact repack: **8.1e-7** relative for the f32
  path, within the W4A8 budget for the int8 path.
- End to end at ep_size=8, all four combinations of
  `DS4F_ZEROCOPY_EXPERTS` x `DS4F_MXFP4_W4A8` produce the **same argmax token**,
  at both 8 and 64 generated tokens.

### Remaining gap to S0

S0 measured 43.3 GB/s on the expert gather; the model path gets 24 GB/s. Two
known causes, both S2 work rather than anything wrong here:

- **Dispatch overhead.** `ds4f_matvec` runs one pool dispatch per expert per
  tensor: 6 x 3 x 43 = 774 spin-barrier dispatches per token, against S0's 86.
  At ~30 us each that is ~23 ms of the 143 ms.
- **Page-cache faulting.** The weights are now file-backed, so first touch of
  each expert costs a fault that S0's prefaulted anonymous region did not pay.
  Worth testing `MADV_HUGEPAGE` / `MAP_POPULATE` on the shard mapping.

### Note on CPU-only decode cost

At resident sizes, 93% of CPU decode time is the FP8 dense path (`o_proj` 275 ms,
`qkv_proj` 165 ms, `shared` 104 ms) -- all of which moves to the GPU in S3, so it
is deliberately not optimized here. The FP8 AVX2 kernel uses `vpgatherdd`, which
is slow on Zen1; if a fast CPU dense path is ever wanted, replace the LUT gather
with an arithmetic e4m3 decode.

## S2 results — routed-expert batching and shard mapping (2026-08-03)

`DS4F_EXPERT_BATCH` now defaults to 1. The routed path gathers the locally-owned
top-k experts, runs all w1/w3 projections in one pool dispatch, applies SwiGLU,
then runs all w2 projections in a second dispatch. `DS4F_EXPERT_BATCH=0` keeps
the previous per-expert sequence for A/B comparison. The per-expert matvec
kernels and top-k accumulation order are unchanged.

Full no-copy model, CPU-only, 16 threads, 8 decode tokens:

| Path | experts | total decode |
|---|---:|---:|
| legacy (`DS4F_EXPERT_BATCH=0`) | 134.7 ms/tok | 701.8 ms/tok |
| batched (default) | **107.8 ms/tok** | **675.3 ms/tok** |

Both runs ended at argmax token 128819, with zero major faults. The reduced
real-weight ep_size=8 run also showed the same argmax and 1.677 -> 1.491 ms/tok
expert improvement. The full run peaked at 20.0 GiB RSS; load time was 90.5 s
with batching and 86.5 s in the legacy run, which is normal file-cache noise.

The zero-copy loader now accepts two opt-in mapping experiments:
`DS4F_BLOB_HUGEPAGE=1` applies `MADV_HUGEPAGE`, and
`DS4F_BLOB_POPULATE=1` adds `MAP_POPULATE`. On the reduced real model,
`MADV_HUGEPAGE` made no measurable difference (zero major faults). `MAP_POPULATE`
was not run interactively because the no-copy mapping spans the full ~166 GB
virtual shard range; it remains opt-in for a detached/batch measurement.

## S3 results — HIPRTC dense FP8/E8M0 bring-up and integration (2026-08-03)

The first GPU slice is now buildable without a ROCm SDK. It uses the existing
`rocew` dynamic loader and HIPRTC to compile a one-block-per-row matvec for
row-major FP8 E4M3FN weights with 128x128 E8M0 scales. The host API uploads a
matrix once with `hip_ds4f_dense_load()` and reuses the device weights for
subsequent `hip_ds4f_dense_matvec_loaded()` calls; the convenience API keeps a
single-call upload path for small tests. A one-in-flight asynchronous API
(`hip_ds4f_dense_matvec_loaded_async()` / `hip_ds4f_dense_wait()`) owns a stream
and event so CPU routed-expert work can run while the GPU matvec is pending.
The two-slot `hip_ds4f_dense_matvec_tensors_async()` /
`hip_ds4f_dense_wait_tensors()` API launches independent shared-expert matrices
on separate streams and is used by the model-level overlap gate.

The CPU comparison harness uses a 131x259 matrix, so it exercises partial row
and column blocks, FP8 zero/subnormal/negative/finite-exp15 codes, and multiple
E8M0 scales. On the RX 9070 XT (`gfx1201`, ROCm 7.14), it reports
`max_rel=8.4e-6` and PASS. The loader also accepts `ROCEW_ROCM_LIB` when ROCm
lives outside the built-in search paths.

```bash
make -C hetero/ds4f all
make -C hetero/ds4f hip-test
make -C hetero/ds4f forward-hip-test
```

`forward-hip-test` allocates two one-layer synthetic models, binds the eight
FP8 MLA/shared tensors into the persistent bank, and compares a CPU-default
forward against the callback-enabled forward. It passed with six synchronous
GPU calls plus one two-matrix async batch for shared `w1/w3`, matching argmax,
`x_rel=9.1e-5`, and `logits_rel=3.3e-5`.

The real staged harness now keeps a stable bank ID on each layer-0 tensor and
does the same forward A/B. With `DS4F_EP_SIZE=8` and one layer, all eight real
FP8 matvecs passed (`max_rel <= 1.67e-6`), and the hybrid forward matched the
CPU argmax (`109502`) with `x_rel=6.65e-7` and `logits_rel=5.01e-6`.

The bank holds all eight FP8 dense matrices for all 43 layers (344 matrices,
**5.682 GB**) plus the replicated flat-BF16 vocabulary head (**345 matrices,
6.741 GB** total). The staged real harness now binds the complete bank and
runs a full 43-layer CPU/GPU forward followed by a GPU-attached decode loop.
The BF16 head A/B gate passes at `max_rel=1.60e-5`; the layer-0 FP8 gates remain
below `1.67e-6`. The gfx1201 row kernel now defaults to 128 threads (the
original 256-thread kernel remains selectable with `DS4F_HIP_BLOCK_THREADS`).
The safe EP=8 mechanical case measures **80.7 ms/token (12.40 tok/s)** at the
short-context position after the cleanup; the profiled phase split remains
qkv about 23 ms, attention 0.7 ms, o_proj about 24 ms, shared about 11 ms,
routed experts about 17 ms, and head about 2.5 ms.
The local shard is incomplete, so this is an attachment/timing result, not a
quality result. Across all 43 layers the existing CPU/GPU reduction order
accumulates `x_rel=0.148` and `logits_rel=0.113`, but the output is finite and
the greedy argmax remains locked; the harness reports this as cumulative drift
rather than treating it as a per-kernel failure.

```bash
./build/test_hip_ds4f_real --stage-dir /tmp/ds4f_nocopy_ep8 \
  --ep-size 8 --ep-rank 0 --threads 16 --cmgs 1 --max-pos 12288 \
  --layers 43 --bank-layers 43 --iters 8 --pos0 1 --hip-device 0 --hip-async 1
```

### S3d — batched GPU prefill GEMM

The prefill path now exposes a `ds4f_gpu_dense_gemm_fn` callback and reuses the
RDNA4 16x64 tiled GEMM layout for token-major `X[M,K]` to `Y[M,N]` projections.
FP8/E8M0 weights are dequantized inside the tile; flat BF16 weights use a
BF16-correct sibling of `rdna4/gemm_tiled_f16_f32` (the existing function is
FP16-weight, so it cannot be used directly for DS4F BF16). Eligible projection
pairs/groups are dispatched through the callback, while MXFP4 experts and
unsupported/view tensors retain the CPU fallback.

The FP8 batched GEMM uses a resident 256-entry E4M3FN bit-pattern LUT on the
device, removing repeated decode branches from the tile while preserving the
standalone correctness result.

The standalone gates cover both types: FP8 `M=19` reaches max relative error
`2.31e-5`, and BF16 `M=13` reaches `1.91e-6`. On the staged EP=8 real shard,
one layer with 16 CPU workers measured:

| prefill batch | CPU tok/s | GPU dense tok/s | speedup | argmax mismatches |
|---:|---:|---:|---:|---:|
| 16 | ~86 | 353.38 | ~4.1x | 0 |
| 64 | ~86 | 589.23 | ~6.8x | 0 |
| 128 | ~86 | 608.81 | ~7.1x | 0 |

The current exact default on the complete staged 43-layer shard measures
**3.45 tok/s CPU versus about 22.1 tok/s hybrid GPU at batch 64 (about 6.4x)**
with zero argmax mismatches. The pre-LUT baseline was 21.388 tok/s; profiled
runs vary around 21.3--22.1 tok/s. The earlier batch-16 checkpoint was 16.050 tok/s;
batch 128 reaches 22.401 tok/s but currently has 3/128 mismatches and remains
a tuning result, not the exactness baseline.

The default path uploads 344 FP8 MLA/shared matrices plus the BF16 head;
routed MXFP4 experts and the router remain CPU-owned. Shared-input GEMM pairs
upload once, grouped `wo_a` row slices reuse the resident bank, and x86
attention reuses each BF16 KV row across eight heads. Projection activations
still cross host memory, so a device-resident activation arena and fused GPU
attention/norm/MLP are the remaining route toward a full-model 30-tok/s
prompt-rate result.

#### SM120 CUDA + RDNA4 dual-GPU prefill

The SM120 path is implemented by `cuda/llm/mmq_kernels.cubin` and the
`hetero/ds4f` CUDA bridge. MXFP4 expert matrices are packed once into the
native SM120 MMQ layout, uploaded through persistent pinned host staging, and
executed on the NVIDIA device. FP8/BF16 dense matrices remain resident on
the RDNA4 bank. `dual_ds4f_prefill_gemm_multi()` partitions each prefill group
by type and runs the CUDA MXFP4 and HIP dense groups concurrently.

Build and run the real staged gate with:

```bash
make -B -C cuda/llm mmq_kernels.cubin
make -C hetero/ds4f real-dual-test STAGE_DIR=/tmp/ds4f_nocopy_ep8 \
  EP_SIZE=8 EP_RANK=0 LAYERS=43 BANK_LAYERS=43 THREADS=48 CMGS=4 \
  PREFILL_BATCH=64 PREFILL_CONTEXT=0
```

The bridge gates pass at M=64 and M=128, and the direct small-bucket path
also passes at M=7. Four-layer real dual runs reach approximately 129--133
tok/s with zero argmax mismatches. The full 43-layer EP=8 shard is
operational at approximately 9--11 tok/s; its remaining one-or-few argmax
differences are deterministic cumulative error from SM120 FP4 activation
quantization, not a dispatcher race. `--dual-cuda-mxfp4 0` selects the exact
HIP MXFP4 path for small models, but the full 43-layer expert bank does not fit
on the available 8-GB device, so it is not a full-model fallback.

For the highest-quality CUDA approximation, pass `--dual-cuda-terms 2` (or set
`DUAL_CUDA_TERMS=2` for `real-dual-test`). The bridge decomposes each
activation into two FP4 terms and accumulates two SM120 GEMMs. This reduces
the full-model drift substantially, but repeated 43-layer staged runs have
still shown a small number of argmax differences (typically 3--4/64), so it
is not yet a strict mismatch-free mode. The one-term default is about 10--12
tok/s and can drift more; two-term is the quality setting.

The fast GPU prefill path is numerically approximate over a separately generated
KV history: small dense-GEMM reduction differences can accumulate into a few
CPU-reference argmax changes at long context. For a guaranteed CPU-reference
prompt, set `"hip_exact_prefill": 1` in the JSON or pass
`--hip-exact-prefill 1`. This skips only M>1 GPU GEMMs, so M=1 decode continues
to use the GPU dense bank; the trade-off is CPU prefill throughput.

There is also an explicit approximate-speed experiment for hot shared weights:
set `"hip_shared_bf16": 1` (and optionally
`"hip_shared_bf16_layers": N`) or pass
`--ds4f-hip-shared-bf16 1 --ds4f-hip-shared-bf16-layers N` to the server.
This promotes shared `w1/w3/w2` and the router to BF16 GPU matrices and reached
34.40 tok/s at batch 64, but produced one marginal argmax mismatch. It is not
the exactness baseline and is disabled by default.

`"hip_shared_fp16": 1` is a separate experiment: it expands only the shared
FP8 weights to mathematically exact FP16 values and reuses the existing RDNA4
FP16-weight tile. The one-layer gate passes at 753.55 tok/s with zero
mismatches; the full 43-layer run reaches about 34--35 tok/s but accumulated one
argmax change from the different reduction path. It is therefore also opt-in,
although its weight conversion itself is exact. The layer cap is
`"hip_shared_fp16_layers": N`, or pass
`--ds4f-hip-shared-fp16 1 --ds4f-hip-shared-fp16-layers N` to the server.

`"hip_ordered_wkv_layers": N` enables a CPU-compatible FP8 reduction for the
first `N` WKV projections during batched prefill (M=1 decode is unchanged).
It reduces the first-layer KV rounding error at a small throughput cost, but it
does not make the full recurrent prompt path mismatch-free; use
`hip_exact_prefill` when zero mismatches are required.

`"hip_ordered_fp8_layers": N` extends the CPU-compatible reduction to every
FP8 dense prefill projection in the first `N` layers. Ordered mode compiles the
HIPRTC module with precise math so the compiler cannot reassociate the ordered
sum. The validated RDNA4 path reaches 21.27 tok/s at 4k and 20.90 tok/s at 8k,
with 0/64 argmax mismatches at both contexts. It remains opt-in because the
ordinary fast reduction is the higher-throughput default for non-deterministic
workloads.

`"hip_mxfp4_widen_layers": N` streams local MXFP4 expert weights for the first
`N` layers, widening only the active layer to an exact row-scale FP8
representation. This is intended for RDNA4, which has no native MXFP4
instruction. Streaming keeps expert residency bounded to one layer and is
used for batched prefill; decode retains the existing resident dense path.
For upload-latency experiments, `"hip_mxfp4_stream_raw": 1` selects the
compact raw-MXFP4/LUT kernel instead of FP8 widening; it is exact but slightly
slower in the expert GEMM itself.
On 16-GB-class RDNA4 cards, `"hip_mxfp4_resident_layers": 20` keeps the first
20 local expert layers resident in compact raw form and streams the remainder;
this measured 35.4 tok/s for full 43-layer batch-64 prefill with zero argmax
mismatches.

## Long-context stability and speculative-decode probe

The real HIP harness now accepts `DS4F_MAXPOS` and can warm a synthetic KV
prefix before measuring a later position. This reaches a 4k context directly:

```bash
./build/test_hip_ds4f_real --stage-dir /tmp/ds4f_nocopy_ep8 \
  --ep-size 8 --ep-rank 0 --threads 16 --cmgs 1 --max-pos 12288 \
  --layers 43 --bank-layers 43 --iters 8 --pos0 4096 --warm 4096 \
  --hip-device 0 --hip-async 1
```

`DS4F_EXPERT_RESIDENT=1` queues readahead for the owned file-backed expert
shard. `=2` synchronously touches up to 2 GB by default; increase that limit
with `DS4F_EXPERT_RESIDENT_GB` only when the node has sufficient headroom.
This avoids requiring a full `MAP_POPULATE` of the 166-GB virtual mapping.

For batched prefill at a later context position, use
`--prefill-context N --prefill-batch 64`. The harness warms the final
128-token attention window at `N-128` before measuring the batch at `N`, so
the timing includes long-position RoPE/KV/attention behavior without
recomputing history outside the sliding window. On the staged EP=8 shard:

| mode | context | GPU tok/s | CPU-vs-GPU argmax mismatches |
|---|---:|---:|---:|
| exact default | 4096 | 21.20 | 5/64 |
| exact default | 8192 | 21.01 | 4/64 |
| ordered WKV first 8 layers | 4096 | 21.11 | 3/64 |
| ordered WKV first 8 layers | 8192 | 20.86 | 4/64 |
| exact prefill (`hip_exact_prefill`) | 4096 | 3.33 | 0/64 |
| exact prefill (`hip_exact_prefill`) | 8192 | 3.34 | 0/64 |
| shared BF16 | 4096 | 32.89 | 5/64 |
| shared BF16 | 8192 | 32.05 | 5/64 |
| shared FP16 | 4096 | 32.83 | 1/64 |
| shared FP16 | 8192 | 31.73 | 4/64 |

Prefill throughput is stable within roughly 1% for exact, 2.5% for BF16,
and 3.4% for FP16 from 4k to 8k. The mismatch column compares CPU and GPU
passes after separately warming their KV tails; it exposes accumulated
reduction/history drift and should not be read as a GPU-only determinism test.
The exact BF16 attention window now uses AVX2 widen/dot/AXPY helpers on x86;
the original scalar path remains available with `DS4F_ATTN_SVE=0`. Serial,
The final serial, uncontended run measures **94.95 ms/token (10.53 tok/s)** at
position 4096 after the cleanup; the preceding check measured 96.77 ms/token
(10.33 tok/s), so both remain above the 10 tok/s long-context gate. These
are EP8 mechanical attachment/timing results, not full-model quality results.

The loader now detects a missing `mtp.0.*` block instead of aborting.
`make -C hetero/ds4f mtp-test` runs `build/test_ds4f_mtp`, which drafts K
tokens with the loaded MTP block and checks them against the main model. It
intentionally claims no speedup yet: verification is sequential, and the MTP
KV prefix bootstrap/rollback is not complete. The current EP8 mechanical
manifest has no MTP tensors, so the probe cleanly reports SKIP. No DS4F DFlash
draft checkpoint or implementation is present; DFlash remains a follow-on
draft-model integration after MTP is validated.

## Explicit runtime configuration and ownership

AMD/heterogeneous DS4F tools take configuration from command-line arguments or
the small JSON runtime object. JSON is loaded first and every explicit command-
line option wins, independent of argument order. For example:

```json
{
  "model": "flash",
  "stage_dir": "/tmp/ds4f_nocopy_ep8",
  "ep_size": 8,
  "ep_rank": 0,
  "threads": 16,
  "cmgs": 1,
  "max_pos": 12288,
  "exact": 1,
  "hip": 1,
  "hip_async": 1,
  "hip_device": 0,
  "hip_shared_fp16": 0,
  "hip_shared_fp16_layers": 0,
  "hip_ordered_wkv_layers": 0,
  "hip_exact_prefill": 0
}
```

The real GPU harness is then invoked as:

```bash
./build/test_hip_ds4f_real --config ds4f.json --layers 43 \
  --bank-layers 43 --iters 8 --pos0 4096 --warm 4096
```

`test_ds4f_real_tokens` and `test_ds4f_mtp` use the same interface; their
prompt is supplied with `--prompt-ids`, and MTP uses `--mtp-k N` plus the
opt-in `--batch-verify`. `--debug-env` is the only compatibility mode that
reads the old `DS4F_*`/`LLM_THREADS` settings. Low-level profiler, residency,
and diagnostic switches remain environment-only by design.

Model, session, request, prefill, and harness scratch allocations are owned by
the mmap-backed `ds4f_mem_pool`. Pool statistics are available through the
debug-only `DS4F_MEM_STATS=1` switch; normal teardown releases whole mapped
chunks, so callers do not need raw `malloc/free` ownership rules.

The deterministic prefill gate checks batched versus token-at-a-time argmax
results and reports throughput:

```bash
make -C hetero/ds4f prefill-test
# representative: argmax_mismatch=0, positive batched speedup
```

The real-staged gate uses the same explicit options. After AVX2 two-token
weight reuse, multi-GEMM barrier fusion, and routed-expert slab staging, EP8,
one layer, 16 threads, and batch 16 measure **42.83 tok/s batched versus
12.67 tok/s token-at-a-time (3.38x)** with `argmax_mismatch=0`; batch 128
reaches **51.85 tok/s** with zero mismatches. At `pos0=4096`, the same gate
measures **45.95 tok/s** with zero mismatches. The full 43-layer CPU gate is
**1.63 tok/s** at batch 16, so the 30 tok/s headline applies to the measured
one-layer prefill kernel gate, not complete model latency. The test also
exercises the zero-copy on-disk MXFP4 nibble path.

The explicit resident-BF16 profile (`dense_bf16=1`, `bf16_pv=1` in JSON, or
the legacy `--debug-env` compatibility path) raises the same one-layer gate
to **67.73 tok/s** at batch 16 and **86.69 tok/s** at batch 128, with zero
mismatches; at `pos0=4096` it measures **63.17 tok/s**. It adds about 6 GB of
resident dense weights, so the default remains on-demand FP8.

```bash
./build/test_ds4f_prefill_real --stage-dir /tmp/ds4f_nocopy_ep8 \
  --ep-size 8 --ep-rank 0 --threads 16 --cmgs 1 --max-pos 128 \
  --layers 1 --batch 16 --iters 1 --pos0 0
```

## Files

| | |
|---|---|
| `mxfp4_avx2.h` | AVX2 MXFP4 matvec kernels: f32 and W4A8, several shapes. `ds4f_matvec_mxfp4_1row_i8` is the one that won. Migrates into `common/ggml_dequant.h` as the `__AVX2__` branch in S1. |
| `bench_expert_bw.c` | S0 roofline harness. Modes `stream`/`matvec`/`triad`/`verify`; kernel selection `--i8`, `--i8x2`, `--i8h`, `--i8seq`, `--f32seq`. |
| `test_ds4f_kernels.c` | S1 correctness gate: every AVX2 decode kernel vs an independent scalar reference. `make test`. |
| `../../common/ds4f_matvec_avx2.h` | The AVX2 decode kernels that ship in the model path (S0's winner, productionized). |
| `../../common/ds4f_kernels_x86.h` | Portable stand-ins for ds4f_impl.h's SVE helper primitives. |
| `hip_ds4f_dense.{c,h}` | HIPRTC host runner with persistent FP8/E8M0 device-weight loading. |
| `hip_ds4f_kernels.h` / `test_hip_ds4f_dense.c` | Dense FP8/E8M0 kernel source and CPU comparison harness. |
| `test_hip_ds4f_real.c` | Real staged-tensor A/B, full multi-layer bank attachment, and hybrid decode timing gate. |
| `test_hip_ds4f_forward.c` | Synthetic model-level CPU/GPU callback A/B gate. |
| `test_ds4f_real_tokens.c` | Real-token exact-f32 versus W4A8 teacher-forced logit/argmax gate. |
| `test_ds4f_mtp.c` | Guarded MTP draft/verification acceptance probe. |
| `test_ds4f_prefill_real.c` | Real-staged batched-vs-token prefill throughput/parity gate. |
| `../../server/server_ds4f.{c,h}` | Native x86 staged-session adapter for `server_llm` (seeded sampling, one-slot exact-prefix KV/logit cache). |

## Reproducing

```bash
make && make test && make sve-check

# stage a no-copy manifest (ep_size=1 = whole model; 8 or 2 for a smaller slice)
DS4F_STAGE_NOCOPY=1 DS4F_MODEL_DIR=/mnt/disk1/models/ds4f-0731 \
DS4F_STAGE_DIR=$PWD/stage DS4F_EP_RANK=0 DS4F_EP_SIZE=1 DS4F_NSHARDS=48 \
  ./build/ds4f_stage

# decode (LLM_THREADS=16 -- the A64FX default of 48 oversubscribes this host)
DS4F_REAL=1 DS4F_EXACT=1 DS4F_STAGE_DIR=$PWD/stage DS4F_EP_RANK=0 DS4F_EP_SIZE=1 \
LLM_THREADS=16 DS4F_CMGS=1 DS4F_MAXGEN=8 DS4F_PROF=1 ./build/ds4f_runner

# real-token prompt and greedy generation (tokenizer is stdlib-only)
python3 ../../a64fx/llm/tools/ds4f_tokenizer.py encode \
  --tokenizer /mnt/disk1/models/ds4f-0731/tokenizer.json \
  --prompt 'Write a short Python function that adds two numbers.' \
  --out /tmp/ds4f_prompt_ids.txt
DS4F_REAL=1 DS4F_EXACT=1 DS4F_PROMPT_IDS=/tmp/ds4f_prompt_ids.txt \
DS4F_GEN_OUT=/tmp/ds4f_generated_ids.txt DS4F_MAX_NEW=32 \
DS4F_STAGE_DIR=$PWD/stage DS4F_EP_RANK=0 DS4F_EP_SIZE=1 \
LLM_THREADS=16 DS4F_CMGS=1 ./build/ds4f_runner
python3 ../../a64fx/llm/tools/ds4f_tokenizer.py decode \
  --tokenizer /mnt/disk1/models/ds4f-0731/tokenizer.json \
  --ids-file /tmp/ds4f_generated_ids.txt

# real-token W4A8 quality gate (EP=8 is mechanical; EP=1 is quality-valid)
./build/test_ds4f_real_tokens --stage-dir /tmp/ds4f_nocopy_ep8 \
  --prompt-ids /tmp/ds4f_prompt_ids.txt --ep-size 8 --threads 16 --cmgs 1 \
  --max-pos 8192
```

`DS4F_MXFP4_W4A8=0` selects the exact-f32 expert kernel instead of W4A8.

The native x86 OpenAI server adapter is an opt-in CMake feature. It loads the
same staged no-copy model, invokes the stdlib-only tokenizer as a child
process, serves `/v1/completions` and `/v1/chat/completions` with greedy or
seeded temperature/top-p decoding, and reuses an exact one-slot prompt-prefix
KV plus logit snapshot. It currently requires replicated embedding/head
weights and Tier-B1 bf16 KV state.

```bash
cmake -S ../../server -B /tmp/ds4f-server-build \
  -DDIFFUSION_SERVER_ENABLE_QWEN_IMAGE=OFF \
  -DDIFFUSION_SERVER_ENABLE_SAM3=OFF \
  -DDIFFUSION_SERVER_ENABLE_DS4F_HETERO=ON \
  -DDIFFUSION_SERVER_ENABLE_DS4F_HIP=ON
cmake --build /tmp/ds4f-server-build -j
/tmp/ds4f-server-build/diffusion-server \
  --ds4f-model /tmp/ds4f_nocopy_ep8 --ds4f-threads 16 \
  --ds4f-ep-size 8 --ds4f-ep-rank 0 --ds4f-max-pos 8192 \
  --ds4f-hip 1 --ds4f-hip-async 1 --ds4f-exact 1 --ds4f-mhc 1 \
  --ds4f-tierb2 0 --ds4f-tp-embed 0 --ds4f-tp-head 0 \
  --ds4f-config ds4f.json
```

`--ds4f-hip 1` attaches the persistent RDNA4 dense bank at startup. The server
then uploads the eight FP8 MLA/shared tensors per layer plus the replicated
flat-BF16 vocabulary head (345 matrices, about 6.741 GB for the 43-layer
model), while routed MXFP4 experts, router, and embedding remain on the CPU.
`--ds4f-hip-async 1` enables the existing two-stream shared-expert launch,
overlapping that GPU work with CPU routing and experts. Leaving it at zero
keeps the same native x86 session CPU-only.

## Next

1. **Long-context endurance run**: repeat the 4k-position harness for at least
   256 tokens on a quiet host. The final-binary 8-token serial gate passes
   10.53 tok/s at position 4096; the bounded `DS4F_EXPERT_RESIDENT` policy is
   available for the longer run.
2. **Real accuracy gate for W4A8**: the EP=8 real-token gate now reports
   11/11 argmax matches, worst logit relative error 6.37%, and mean
   cross-entropy delta -0.00031. Repeat against the complete EP=1 model and a
   PyTorch/reference token stream before declaring quality.
3. **S3c follow-up**: repeat the full attachment with the complete EP=1 model
   for a quality-valid end-to-end measurement; the current CPU-only dense cost
   is 587 ms/token, while EP=8 is only a mechanical timing case.
4. **S4a/S4b done**: `DIFFUSION_SERVER_ENABLE_DS4F_HETERO` now adds a native x86
   staged-safetensors session to `server_llm`; it has stdlib tokenizer handoff,
   OpenAI text/chat endpoints with seeded sampling, a one-slot exact-prefix
   KV/logit cache, and an opt-in persistent HIP dense bank. The EP=8 GPU server
   smoke passed twice through the same process with identical text output.
   Remaining S4 work is TP-sharded embedding/head state. The A64FX tree still
   provides the file/socket runner protocol for multi-node deployments.
5. **MTP/DFlash**: obtain or stage a checkpoint containing `mtp.0.*`, then
   bootstrap its KV prefix during prompt processing, add safe snapshot/rollback
   for partial acceptance, and benchmark the existing batched verifier. DFlash
   needs a separate compatible draft checkpoint; no implementation is present
   in this tree yet.
