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
| GPU | RTX 5060 Ti, SM120, 16 GiB (14.4 GiB free) |
| Host link | **PCIe gen3 x8, ~6.5 GB/s practical** |
| Model | `/mnt/disk1/models/ds4f-0731`, 48 safetensors shards, 156 GB |

The link width matters more than it looks. Decode touches ~431 MB of expert
weights per token, so streaming experts to a GPU costs ~66 ms/token against
the CPU's measured 15.4 ms. Any offload plan that moves weights per call is
dead on arrival here; only fully resident weights pay off. Check
`nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current` first.

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
`x_rel=9.1e-5`, and `logits_rel=3.3e-5`. (2026-08-05: this gate used to hang --
`ds4f_runtime_options_debug_env()` clobbered the caller's `cfg` with the full
43-layer default, so the one-layer synthetic model allocated a 155 GB arena.
Fixed: the env config is applied only when `DS4F_MODEL` is explicitly set.)

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
The safe EP=8 mechanical case now measures **53.0--53.2 ms/token
(18.78--18.87 tok/s)** at the short-context position, and **14.13--14.21 tok/s**
at a warmed 4k position. The profiled short-context split is qkv 11.9 ms,
attention 0.9 ms, o_proj 11.9 ms, shared 8.0 ms, router 2.1 ms, routed experts
15.4 ms, head 1.9 ms. The CPU MXFP4 expert phase is now the largest single
term (29.5%).

`--decode-verify N` is a CPU-referenced multi-step decode gate: it runs N
teacher-forced steps as a KV history accumulates, with the CPU pass first and
the GPU pass rerunning the same position so both read identical history and
write the same cache slot. It reports 0 argmax mismatches over 8 steps both at
`--pos0 1` (worst logit rel 0.047) and at `--pos0 4096 --warm 4096` (1.01,
inflated by the synthetic warm). Use this rather than the free-running
`last_argmax`, which is autoregressive over synthetic KV and diverges
chaotically after any single flip.

```bash
./build/test_hip_ds4f_real --stage-dir /tmp/ds4f_nocopy_ep8 --ep-size 8 \
  --ep-rank 0 --threads 16 --cmgs 1 --layers 43 --bank-layers 43 \
  --hip-device 0 --hip-async 1 --max-pos 12288 \
  --decode-verify 8 --pos0 4096 --warm 4096
```

The matvec kernels previously gave each output row a whole 128-thread block.
For `wq_b` (cols=1024) that is 64 chunks of work over 128 lanes, half of them
idle, plus a shared-memory reduction tree and a `__syncthreads` per row. All
four now run **one wave32 per row** with `blockDim.x/32` rows per block, so the
row total is one shuffle reduction with no shared memory and no barrier:
`wq_b` 182 -> 280 GB/s. Block size stopped mattering (64/128/256 all land at
18.78--18.81 tok/s), which is what one expects once a row is wave-local.

Beyond the vectorized weight loads described next, two dispatch fixes took
decode from 15.2 to 17.8 tok/s without touching any arithmetic. `DS4F_MV_FUSE`
defaults to 0, so the independent `wq_a`/`wkv` pair ran as two dispatches; it
is bit-exact by construction and this harness now enables it (an explicit env
setting still wins; the shared default stays 0 pending an A64FX measurement).
And `ds4f_matvec_multi`'s GPU branch took a launch, event sync and blocking
download *per tensor* -- it now puts the whole group in flight through the
existing `gpu_dense_async_multi`/`gpu_dense_wait` pair, with
`HIP_DS4F_ASYNC_MAX` raised from 2 to 8, falling back to the serial loop when
the adapter already holds a batch.

Separately, **every host transfer was pageable** -- there was no
`hipHostMalloc` anywhere in `hip_ds4f_dense.c` -- so the driver bounced each
one through its own staging buffer. `wq_b` alone downloads 131 KB per layer,
5.6 MB/token, and measured 11.96 ms against a 7.9 ms kernel. The matvec upload
and download paths now stage through pinned buffers, and the async group
queues every download before synchronizing so they overlap: `wq_b` 11.96 ->
9.69 ms. `ensure_pinned` returns -1 when the loaded driver has no
`hipHostMalloc`, leaving the original pageable path in place.

The previous figures were 80.7 ms/token (12.40 tok/s) short-context and
10.53 tok/s at 4k, with qkv about 23 ms and o_proj about 24 ms. The dense
matvec kernels were reading **one FP8 byte per lane per iteration**
(`for (col = tid; col < cols; col += blockDim.x)`), so each ~32-byte wave
request touched a 128-byte line to consume a quarter of it; `wq_b` measured
132 GB/s of the roughly 640 GB/s the card offers. `ds4f_dense_fp8_matvec` and
`ds4f_dense_fp8_blockdiag` now take 16 consecutive weights per lane through a
`uint4` load, with a scalar tail for any `cols % 16`. A 16-aligned run cannot
straddle a 128-column E8M0 block, so the block scale factors out of the run
and is applied once; `wq_b` rose to 182.8 GB/s.

That reassociation is the only numerical change, and it moves the result
*toward* the CPU reference rather than away: every layer-0 A/B gate improved
(worst `max_rel` 1.67e-6 -> 1.10e-6), the multi-layer drift fell from
`x_rel=0.0547` to `0.0159`, and the greedy argmax is unchanged across every
repetition. The odd-shaped `hip-test` case (`cols=259`) exercises the scalar
tail and still passes.
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
**3.51 tok/s CPU versus 40.2 tok/s hybrid GPU at batch 64 (11.5x)** with zero
argmax mismatches, using `--hip-ordered-fp8-layers 43`. Batch 128 reaches
41.60 tok/s, also at 0/128 mismatches. At warmed long context the same
configuration measures 37.63 tok/s at 4k and 37.06 tok/s at 8k, both 0/64.

Of that, the last 35.0 -> 40.2 tok/s came from the CPU side. The routed-expert
phase is the only significant CPU work left in prefill, and
`ds4f_gemm_worker_x86`'s raw-MXFP4 branch ran its token loop *outermost*, so a
tile of M tokens called `ds4f_mxfp4_unpack16` M times over the same weights --
about fifteen ops to decode a 32-weight block against six to consume it for
one token. `matvec_mxfp4_1row_f32_raw_4x` decodes once and feeds four tokens,
each with its own accumulator pair in the original order, so every output is
bit-identical to the 1-row kernel. With the tile quantized up front and the
row loop hoisted outside the token loop (token blocks innermost, so a row's
2 KB of weights stays in L1), the expert phase went 13.43 -> 9.78 ms/token.
Note this targets AVX2/Zen1; the SVE path is unchanged.

The previous exact figure was 21.3--22.1 tok/s. The gap was **not** kernel
throughput: `m->gpu_dense_mixed` was being set from
`hip_shared_bf16 || hip_shared_fp16`, so in the exact default it was 0. With
mixed dispatch off, `ds4f_gemm_multi` sends a whole independent group to the
CPU whenever any one member is CPU-owned — and the shared-expert group pairs
`sh_w2` with the CPU-resident router `gate`. Shared-expert prefill was
therefore paying 19.0 ms/token of CPU `sh_w2` work. Mixed dispatch is a
routing policy, not a precision mode: each member keeps exactly the arithmetic
it had before, so enabling it unconditionally moved `shared` from 19.0 to
2.6 ms/token. The one argmax change it introduces on its own is the GPU
reduction order for `sh_w2`, which `hip_ordered_fp8_layers` removes — hence
0 mismatches at 35.28 tok/s.

This also explains the old `hip_shared_bf16` / `hip_shared_fp16` results
(34.40 and ~34--35 tok/s, 1 mismatch each). Those flags were never buying
speed through BF16/FP16 weights; they were buying it by switching mixed
dispatch on as a side effect. They are no longer needed to reach this rate.

Two kernel hypotheses were tested against this and **falsified**: staging the
256-entry E4M3 LUT in LDS to remove the dependent global gather, and
re-indexing the FP8 GEMM's LDS weight staging so lanes read consecutive bytes
instead of 16 rows `n_in` apart. Each is value-identical and each moved the
shared phase by under 0.15 ms/token (19.08 -> 18.96 -> 19.00). Neither was
kept. The FP8 GEMM tile is not the prefill bottleneck at these shapes.

The default path uploads 344 FP8 MLA/shared matrices plus the BF16 head;
routed MXFP4 experts and the router remain CPU-owned. Shared-input GEMM pairs
upload once, grouped `wo_a` row slices reuse the resident bank, and x86
attention reuses each BF16 KV row across eight heads. Projection activations
still cross host memory, so a device-resident activation arena and fused GPU
attention/norm/MLP are the remaining route toward a full-model 30-tok/s
prompt-rate result.

#### GPU prefill GEMM

`--gemm-bench M` times repeated device GEMMs on real bound layer-0 tensors and
nothing else. Use it instead of the whole-phase prefill numbers when the host
is busy: the `gpu=` phase columns still contain CPU rmsnorm/RoPE/SwiGLU and
the routed-expert phase, so they move with load average.

Best-of-5 at M=64, batch-64 prefill shapes:

| tensor | N x K | before | after | |
|---|---|---:|---:|---:|
| `wq_b`  | 32768 x 1024 | 4.637 ms / 926 GFLOP/s | 3.633 ms / 1182 | 1.28x |
| `wo_b`  | 4096 x 8192  | 4.509 ms / 953 GFLOP/s | 2.707 ms / 1587 | 1.67x |
| `sh_w1` | 2048 x 4096  | 1.471 ms / 730 GFLOP/s | 0.792 ms / 1355 | 1.86x |

Two changes:

1. **The FP8 tile staged `smB` four times per k-step**, paying eight
   `__syncthreads` per 16 columns of K for four outputs per thread. It now
   stages one `smB[16][65]` and pays two. The odd row stride matters: the tile
   is indexed `[tx][...]` on the staging store, and a 64-float stride puts
   every lane in LDS bank 0. Worth about 1.26x on its own. The ordered kernel
   keeps its per-output accumulation order, so `--hip-ordered-fp8-layers`
   remains exact; the lane index is now written `i & 7` directly, which is
   provably what `(k - kb + i) & 7` evaluates to since `k - kb` always steps
   by 16.
2. **The GEMM staging buffers were pageable.** A batch-64 `wq_b` GEMM moves
   8.4 MB back per call, so the driver was bouncing it through its own staging
   buffer. `ensure_gemm_host_pack` now prefers `hipHostMalloc` and both
   directions route through it. This is where `wo_b` and `sh_w1` get most of
   their gain.

`wq_b` barely moved because it is **transfer-bound, not compute-bound**: 8.4 MB
over PCIe gen3 x8 is a ~1.3 ms floor against roughly 0.9 ms of arithmetic.
Nothing in the kernel can fix that. The remaining structural win for GPU
prefill is a device-resident activation arena so projection intermediates stop
crossing host memory between chained GEMMs -- which needs GPU rmsnorm / SiLU /
SwiGLU kernels, since today the CPU work between `wq_a -> wq_b`,
`wo_a -> wo_b` and `sh_w1/w3 -> sh_w2` forces every intermediate back to the
host.

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
  EP_SIZE=8 EP_RANK=0 LAYERS=43 BANK_LAYERS=43 CMGS=4 \
  PREFILL_BATCH=64 PREFILL_CONTEXT=0
```

`real-dual-test` defaults `THREADS` to one thread per **physical** core (16 on
this 16-core Threadripper 1950X), the same default the harness uses when
`--threads` is omitted. This is the stable point for the spin-wait pool: SMT
siblings contend and add run-to-run variance (threads=32 measured 33--43
tok/s), and the old hard default 48 oversubscribed the 32 hardware threads and
dropped the prefill to ~17 tok/s. Override with `THREADS=N` if you have a
reason to; 24 measured marginally faster but with more variance.

**2026-08-04 update.** The dual path is no longer a regression and is now
exact: batch 64 measures **39.8--40.1 tok/s at 0/64 mismatches** and batch 128
**40.8--41.5 tok/s at 0/128**, matching the single-GPU numbers on the same
host, and 4k/8k measure 36.5/36.1 tok/s at 0/64. Three defects were fixed:

1. Dense tensors were bound through `dual_ds4f_prefill_bind_tensor()`, which
   always takes the plain FP8 bind and silently bypassed the ordered / fp16 /
   bf16 variant selection. Dual therefore ignored `--hip-ordered-fp8-layers`
   and reported mismatches the single path did not. Dual wraps the *same* HIP
   context, so dense now binds through the normal HIP chain and dual's own
   bind is used only for MXFP4 experts.
2. `attach_prefill_backend()` returned early for dual and never installed
   `gpu_dense_matvec` / `wait` / `blockdiag`, so prefill's M=1 residual
   matvecs fell back to the CPU -- a different reduction order. Because
   `gpu_dense_ctx` is one pointer shared by every callback, dual now supplies
   forwarding wrappers (`dual_matvec`, `dual_wait`, `dual_blockdiag`, ...)
   rather than leaving HIP entry points bound to a dual context.
3. MXFP4 tensors were given a `gpu_id = 0` sentinel regardless of batch, which
   made the dispatcher treat them as device-owned and fail every routed-expert
   group over one task at a time. `dual_ds4f_prefill_set_max_batch()` now
   leaves them CPU-owned unless the run can actually reach the SM120 MMQ
   path's M >= 128. A deliberately CPU-owned tensor returns success, not the
   negative code callers treat as a hard failure, and `--dual-cuda-mxfp4 0`
   likewise leaves experts on the CPU instead of attempting a HIP bind of the
   17.9 GiB expert bank (which cannot fit and previously failed the run).

`real-dual-test` now passes `--hip-ordered-fp8-layers` equal to the layer
count by default (`HIP_ORDERED_FP8_LAYERS`, default `LAYERS`), because that
reduction order is what makes the gate report 0 argmax mismatches. The
non-ordered dense reduction is faster but approximates the CPU reference and
drifts a marginal argmax on this shard (batch 64 token 59), so it is not the
gate default. Set `HIP_ORDERED_FP8_LAYERS=0` to run the fast reduction.

**Fused shared FFN (2026-08-05).** The shared-expert `sh_w1/sh_w3 -> SwiGLU ->
sh_w2` chain can run with the two `[M, shared_inter]` intermediates kept
resident on the RDNA4 card (`--hip-fused-shared-ffn 1`, or
`HIP_FUSED_SHARED_FFN=1` for `real-dual-test`, which now enables it by
default): only `x` is uploaded and only `[M, hidden]` comes back, replacing
the two download / two upload round trips of the unfused chain. Its SiLU is a
bit-exact port of glibc 2.39's `__expf` (the `__exp2f_data` table, `rint`
range reduction, explicit `fma` polynomial) and is compiled into its own
always-precise HIPRTC module, because clang's `-ffast-math` turns the SiLU
division into an approximate reciprocal even through `__fdiv_rn`. Keeping it
separate means the dense module's math mode -- and therefore its GEMM results
-- are untouched, so the fused chain is bit-identical to the unfused one.
Validated at `0` argmax mismatches with `HIP_FUSED_SHARED_FFN=1
HIP_ORDERED_FP8_LAYERS=43` at batch 64, batch 128, and 4k context, on the
staged EP=8 shard.

**CUDA is nonetheless inert at realistic batch sizes.** `--hip-verbose 1`
reports every dispatched task as `cuda=0`: routed-expert buckets top out well
below the M >= 128 the SM120 MMQ path requires. Combined with the PCIe gen3 x8
finding above -- the bridge is PCIe-bound at 3.7 GB/s on its per-call weight
upload -- engaging it would not help even if the buckets were larger. Dual
mode is currently single-GPU parity plus an idle CUDA context; making the
NVIDIA card earn its place needs resident experts, not a better kernel.

**Measured prefill case (2026-08-06).** `--dual-cuda-small-buckets 1` opts the
owned MXFP4 experts into the padded small-bucket SM120 path (verified
numerically correct at M=1--64 for the real N=2048/K=4096 shapes). With the
hybrid split (`--dual-cuda-resident-from 21`: layers 0-20 stay CPU-exact,
21+ route to the SM120 with the 10 GB weight cache; ~22 layers fit -- the
RTX 5060 Ti's real cudaMalloc ceiling is ~11 GB of its 16 GB), the expert
dispatches go through the async batch (`cuda_ds4f_mxfp4_gemm_batch`): weights
are preloaded before the timed prefill, the whole dispatch queues on one
stream with a single sync, and the two-term activation residual is computed
by the `mmqv_mxfp4_residual` device kernel instead of on the host. A stale
`c->y` output pointer (evaluated at the call site, then realloc'd inside
`gemm_once`) used to corrupt the context and forced per-task fallbacks;
fixed, the batch runs every dispatch with zero fallbacks and compute-sanitizer
is clean.

**Settled prefill table (2026-08-06, `bench_summary.cpp`).** Tok/s across the
configs, current tree (DS4F_MAX_MTILE=8192 for batch >512).  "exact" is the
dual dense/shared on ROCm with routed experts CPU-exact for small buckets and
auto-routed to the SM120 once M >= 128 (batch 1024+); the tiled head GEMM and
the async batch roughly doubled the large-batch exact rate.  The accelerated
routes are approximate (~1/64); the exact default stays 0/64 at batch 64.

| batch | exact | rocm-expert | cuda-stream | split |
|---:|---:|---:|---:|---:|
| 256 | 40.6 | 40.3 | 30.6 | **61.7** |
| 512 | 41.3 | 41.2 | 34.0 | **62.4** |
| 1024 | 46.7 | 45.8 | 40.3 | **63.4** |
| 2048 | 52.1 | 54.1 | 49.1 | **65.4** |
| 4096 | 57.1 | 57.9 | 52.6 | **69.1** |

The recommended prefill config is the split with the GPU sliding-window
attention (`--hip-prefill-attn 1`): ~62-65 tok/s flat at every batch.  The
GPU attention frees the CPU attention (was ~23% of the time) but is
approximate like the other accelerated routes (exact default stays 0/64).
`--dual-cuda-terms` is the quality/speed knob for the CUDA tail: terms=2 (the
activation residual) is ~1/64 at 54.6 tok/s (batch 64), terms=1 skips the
residual for ~56 tok/s but ~4/64.

The split now preloads all 29 CUDA layers: the cache is a single contiguous
12.5 GB pool (one `cuMemAlloc`; the driver tops out at ~13 GB for one
allocation vs ~10.8 GB for the old per-tensor 4.46 MB chunks, which
fragmented the heap), so the split is flat ~55-57 tok/s at every batch.
`DS4F_CUDA_MXFP4_CACHE_MB` overrides the pool size (the driver's realistic
single-allocation ceiling, not the 14.4 GB nvidia-smi "free").

`rocm-expert`: `--hip-mxfp4-resident-layers 14 --hip-mxfp4-stream-raw 1`;
`cuda-stream`: `--dual-cuda-resident-from 0 --dual-cuda-preload 0`;
`split`: `--hip-mxfp4-resident-layers 14 --dual-cuda-resident-from 14`.

The route is approximate (~1/64 at batch 64 small buckets; the exact default
stays 0/64). The 1k--2k prefill gain is the target for the resident-weight
server: the CPU-exact head layers 0-20 still dominate once the CUDA tail is
fast, and their routed-expert GEMM degrades at large batch -- the next lever
is covering more layers (or streaming the weights) as the cache allows.

**Weight streaming beats the CPU head layers.** `--dual-cuda-resident-from 0`
routes every layer's experts to the SM120 and lets the 10 GB cache stream
(the LRU keeps the working set; the ~4 ms cold load per tensor amortizes once
the batch is large enough). All tok/s below are approximate (~1/64):

| batch | CUDA-stream | exact |
|---:|---:|---:|
| 256 | 27.8 | 40.7 |
| 512 | 31.8 | 35.6 |
| 1024 | 36.1 | 24.1 |
| 2048 | 36.6 | 23.8 |
| 4096 | 32.4 | 13.7 |

**The ROCm raw-expert path is the real win (2026-08-06).** The RX 9070 XT's
existing `--hip-mxfp4-resident-layers N --hip-mxfp4-stream-raw 1` uploads the
first N layers' raw MXFP4 experts resident (dense bank 6.7 GB + ~0.43 GB/layer;
14 layers leaves headroom for the M=4096 prefill buffers on 16 GB) and streams
the rest one layer at a time. The HIP MXFP4 kernels are exact for the weights
and far faster than the CPU routed-expert GEMM:

| batch | ROCm-expert | exact | delta |
|---:|---:|---:|---:|
| 256 | 47.2 | 40.7 | +16% |
| 512 | 46.3 | 35.6 | +30% |
| 1024 | 51.7 | 24.1 | +115% |
| 2048 | 53.6 | 23.8 | +125% |
| 4096 | 42.2 | 13.7 | +208% |

~1/64 argmax mismatch vs the CPU reference (the raw FP4 kernel's rounding),
batch 64 only ~17 tok/s because the 43-layer streaming cold-load does not
amortize. The widened FP8 variant (default `--hip-mxfp4-stream-raw 0`) is
2x the VRAM per layer, so it fits fewer resident layers and is far slower.
The NVIDIA card is idle under this config; a dual split (ROCm head experts +
CUDA tail experts) is the remaining integration.

**The batch-4096 taper was the head GEMM OOM, not the experts.** With 14
resident layers the M=4096 head gemm ([M, 129280] = 2.1 GB output) failed to
allocate its device scratch and `ds4f_gemm` silently fell back to a ~20x
slower CPU GEMM (42 vs ~56 tok/s). `hip_ds4f_dense_gemm_tensor` now tiles
Ystride==N outputs over 256 MB into a ~256 MB chunk scratch, so the head runs
on the GPU at every batch (batch 4096: 54.8 tok/s with 14 resident layers).
The ROCm raw-expert curve is now smooth:

| batch | ROCm-expert (14 res) | exact |
|---:|---:|---:|
| 512 | 46.2 | 35.6 |
| 1024 | 51.2 | 24.1 |
| 2048 | 52.8 | 23.8 |
| 4096 | 54.8 | 13.7 |

The dual split (ROCm head 14 + CUDA tail) reaches 50.6 tok/s at batch 4096
but is otherwise behind the ROCm-only route, which is the recommended large
batch prefill config.

**Single-token decode is ~17 tok/s; batch it.** The M=1 decode
(`--iters`, `ds4f_forward_token`) reads ~9 GB of weights per token (dense
~5.8 GB + routed experts ~3.4 GB) at ~190 GB/s effective -- the dense
projections already run on the ROCm as M=1 matvecs and the MXFP4 experts now
also have an M=1 ROCm path (`a0214381`), but the per-token serialized
launch+sync leaves both GPUs far from their read bandwidth.  Profile (exact):
qkv 12.0, o_proj 12.0, experts 16.8, shared 8.0, attn 4.3 ms/token.  The
batched path (ds4f_forward_prefill, "M independent forward_token calls with
the dense projections batched so each weight is read from HBM once") is the
2.5-3x speedup: 42 tok/s exact at batch 64, 51-55 tok/s ROCm-expert at
batch 1024-4096.

**Long-context decode stability (2026-08-06).** `DS4F_MAX_POS` raises the
model's `max_pos` (default 4096) so generation can be exercised at long
context.  Decode runs 16 steps at warmed contexts 4096, 8192, and 16384 --
all stable (no OOM or crash; ~16 tok/s, flat, because the sliding-window
attention keeps the per-token cost constant):

| context | decode tok/s |
|---:|---:|
| 4096 | 15.8 |
| 8192 | 15.5 |
| 16384 | 16.5 |

`hetero/ds4f/bench_summary.cpp` re-runs the settled prefill configs
(exact / rocm-expert / cuda-stream / split) across batch 256-4096 and prints
the comparison table.


The historical measurements below predate these fixes.

The bridge gates pass at M=64 and M=128, and the direct small-bucket path
also passes at M=7. Four-layer real dual runs reach approximately 129--133
tok/s with zero argmax mismatches. The full 43-layer EP=8 shard was
operational at approximately 9--11 tok/s; its remaining one-or-few argmax
differences are deterministic cumulative error from SM120 FP4 activation
quantization, not a dispatcher race. `--dual-cuda-mxfp4 0` selects the exact
HIP MXFP4 path for small models, but the full 43-layer expert bank does not fit
in the VRAM left after the dense bank, so it is not a full-model fallback.
(An earlier revision of this line said "the available 8-GB device". That was
wrong on both counts: the NVIDIA device is an RTX 5060 Ti with 16311 MiB total
and about 14.4 GiB free, and the constraint is the dense bank sharing the AMD
card, not an 8-GB part.)

Routed expert buckets below 128 tokens use the exact CPU path; SM120 native MMQ
is reserved for full x128 batches. This avoids unstable partial-tile launches
while retaining CUDA acceleration for large expert groups.

For the highest-quality CUDA approximation, pass `--dual-cuda-terms 2` (or set
`DUAL_CUDA_TERMS=2` for `real-dual-test`). The bridge decomposes each
activation into two FP4 terms and accumulates two SM120 GEMMs. This reduces
the full-model drift substantially, but repeated 43-layer staged runs have
still shown a small number of argmax differences (typically 3--4/64), so it
is not yet a strict mismatch-free mode. The one-term default is about 10--12
tok/s and can drift more; two-term is the quality setting.

For a strict CPU-reference quality gate, `--hip-exact-prefill 1` disables the
approximate HIP dense prefill path: the full context=128 dual-GPU run measured
0/64 mismatches at 3.64 tok/s. The default approximate path is about 10--11
tok/s; its remaining long-context differences originate in AMD dense
approximation rather than the SM120 expert dispatcher.

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

These MXFP4 expert streaming options are single-GPU HIP features. The dual
dispatcher owns expert routing and keeps small buckets on the exact CPU path,
so streaming would only upload weights it never uses; `--dual-gpu 1` leaves
the layer-residency callbacks off (they previously crashed because the dual
wrapper swapped `gpu_dense_ctx` out from under the raw HIP entry points).

Current measurements on this host (2026-08-05, batch 64, 43 layers) put all
three expert routes behind the CPU-expert baseline: `resident-raw` measures
~7.2 tok/s and `widen` slower still, versus ~43 tok/s with the exact CPU
expert fallback plus the GPU dense/shared path (`--hip-ordered-fp8-layers 43
--hip-fused-shared-ffn 1`, threads = `nproc`). The recommended stable-fast
config is the CPU-expert dual (or single) run at 0/64 argmax mismatches.

Why the GPU expert routes lose here, point by point (profiled):

- **Streaming bandwidth, not the GEMM, dominates.** `resident-raw` with 20
  resident layers still streams the other 23 layers' *entire* owned expert
  bank (32 experts x ~12 MB = 384 MB/layer -> ~8.8 GB total). At the PCIe
  gen3 x8 ~3.7 GB/s limit that is ~2.4 s of uploads, which is exactly the
  37 ms/token the `experts` phase shows. No per-layer async prefetch can hide
  it: 8.8 GB > the whole prefill's GPU-compute budget.
- **Routed buckets are tiny at batch 64.** Top-6 of 256 experts over 64 tokens
  leaves ~1--2 tokens per active expert. The raw-MXFP4 tile is 16 tokens wide,
  so ~93% of every tile's M lanes are padding; the raw LUT GEMM measures
  196 GFLOP/s at a full M=16 tile (widened-FP8: 279 GFLOP/s), which collapses
  toward ~12--25 GFLOP/s effective at the real bucket sizes. The CPU
  per-token AVX2 matvec has no such floor.
- **Full residency does not fit.** All 43 layers resident would need ~16.5 GB
  of experts on top of the ~6.7 GB dense bank on a 16 GB card, so non-resident
  layers must stream, which loses anyway.
- **Not a kernel regression.** The grouped and per-expert kernels share the
  same tile arithmetic (196 GFLOP/s); the historical 35.4 tok/s resident
  figure is not reproducible on this host and likely predates the current
  bucket/routing behavior or a different card.

The path forward for GPU experts is larger buckets (bigger batch or
cross-expert packing), native MXFP4 hardware, or a much faster interconnect --
not a kernel fix.

## Long-context stability and speculative-decode probe

### Prefill tuning for 2k--4k context chunks (2026-08-05)

For a 2k--4k token chunk at a long position, the per-token cost breaks down as
(43 layers, batch 64, threads=16, dual + ordered + fused, all 0/64):

| phase | ms/token | where |
|---|---:|---|
| routed experts | ~10 | CPU (raw-MXFP4 AVX2, bandwidth-bound) |
| o_proj | ~5.6 | GPU |
| attention | ~4 | CPU (full 128-token window) |
| qkv_proj | ~3.8 | GPU |
| shared FFN | ~1.8 | GPU (fused) |
| head | ~1.0 | GPU |

Two knobs matter. **(1) Threads = physical cores** (16 here), not `nproc`/48:
the spin-wait pool oversubscription dropped the prefill to ~17 tok/s. **(2)
Batch 128--256 for a long chunk**: the routed expert buckets grow, so the
M>=4 token-block amortizes each expert weight's decode (batch 64 leaves
~1--2 tokens/expert, bucket M=1--2, per-token decode). Measured at 4k
context: batch 64 = 39, batch 128 = 40, batch 256 = 41 tok/s, all 0/64. At
batch 64 context 0 the same phases give ~42 tok/s (no warm tail).

The dense phases (qkv/o_proj/shared/head) are at the measured kernel
throughput (wq_b ~1370 GFLOP/s); the routed experts are the CPU wall and do
not offload to either GPU here (the SM120 path quantizes activations and pays
the 16.5 GB owned-weight PCIe upload; the RDNA4 resident path streams the same
bank). The attention is full-window at long context and is the second CPU
term.

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
