# hetero/ds4f — DeepSeek-V4-Flash 0731 on Threadripper 1950X + RX 9070 XT

Goal: 10+ tok/s single-stream decode at native quantization quality (FP8-E4M3
dense + MXFP4 experts, no requantization). Baseline to beat is llama.cpp on the
same box: 7.33 tok/s on the full MXFP4 model, 8.57 tok/s on a Q3_K_M requant
(see `../../../llama.cpp/da4f.md`).

Split: the FP8 dense path (MLA + shared expert, 5.7 GB) is GPU-resident on the
9070 XT; only the MXFP4 routed experts stay on the CPU.

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
(`DS4F_Q8_PV`). **The real accuracy gate is a model-level logit comparison in
S1**, and if W4A8 turns out to cost measurable quality, the fallback is to keep
f32 activations on the `w2` down-projection only (it is 1/3 of the traffic, so
the cost would be roughly 12.6 -> 11.4 tok/s).

### Projection

The GPU dense path is ~5.7 GB at ~640 GB/s = ~10 ms/token, which overlaps the
80 ms CPU expert time, and the shared expert overlaps too. Adding per-layer
PCIe sync (~1 ms/token over 86 round trips), sampling and the mHC/router work,
end-to-end decode should land around **11-12 tok/s** -- against 8.57 tok/s for
llama.cpp's Q3_K_M, at strictly better quantization quality.

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
  already opt-in (`DS4F_HC_SVE`, default 0) with scalar twins. `ds4f_gemm`'s
  tiled/PV/tile-dequant kernels now fall through to its existing per-token
  matvec branch off A64FX -- correct, just not batched. Batched prefill
  attention has no scalar twin and aborts with a pointer to token-at-a-time.
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

## Files

| | |
|---|---|
| `mxfp4_avx2.h` | AVX2 MXFP4 matvec kernels: f32 and W4A8, several shapes. `ds4f_matvec_mxfp4_1row_i8` is the one that won. Migrates into `common/ggml_dequant.h` as the `__AVX2__` branch in S1. |
| `bench_expert_bw.c` | S0 roofline harness. Modes `stream`/`matvec`/`triad`/`verify`; kernel selection `--i8`, `--i8x2`, `--i8h`, `--i8seq`, `--f32seq`. |
| `test_ds4f_kernels.c` | S1 correctness gate: every AVX2 decode kernel vs an independent scalar reference. `make test`. |
| `../../common/ds4f_matvec_avx2.h` | The AVX2 decode kernels that ship in the model path (S0's winner, productionized). |
| `../../common/ds4f_kernels_x86.h` | Portable stand-ins for ds4f_impl.h's SVE helper primitives. |

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
```

`DS4F_MXFP4_W4A8=0` selects the exact-f32 expert kernel instead of W4A8.

## Next

1. **S2 expert-path tuning**: batch the 774 per-token pool dispatches (S0 used
   86), and test huge pages / `MAP_POPULATE` on the shard mapping. Closing the
   24 -> 43 GB/s gap is worth roughly 143 -> 80 ms/token.
2. **Real accuracy gate for W4A8**: argmax agreement on a synthetic-embedding
   harness is weak evidence. Needs real token input and a logit comparison
   against `ref/`.
3. **S3**: HIP dense offload. 587 of the current 721 ms/token is the FP8 dense
   path, so this is where the remaining end-to-end win is.
