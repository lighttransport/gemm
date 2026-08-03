# hetero/ds4f — resume notes

Snapshot: 2026-08-03. Working tree based on commit `da6b61a8` on branch `ds4f`;
S3/S4 changes below are intentionally uncommitted.

## Resuming prompt

> Continue the hetero/ds4f project: running DeepSeek-V4-Flash 0731 at 10+ tok/s
> single-stream decode on this Threadripper 1950X + RX 9070 XT, at native
> quantization quality (FP8 dense + MXFP4 experts, no requantization).
>
> Read `hetero/ds4f/README.md` first — it is the findings log and has the
> measurements, the repro commands, and the things already tried that did *not*
> work (do not re-try those). `resume-ds4f.md` has the task list.
>
> S0 (roofline) and S1 (x86 port + zero-copy experts) are done and committed.
> S2 is implemented and measured: routed-expert batching reduced the full
> no-copy expert phase from 134.7 to 107.8 ms/token. S3a/S3b are now GPU-
> validated on the RX 9070 XT, including a 43-layer persistent dense bank and
> a one-layer real-weight hybrid forward A/B. S3c is attached through the
> full real multi-layer harness, including asynchronous MLA/shared dense and
> BF16 vocabulary-head offload. The real EP=8 mechanical decode now measures
> 12.40 tok/s (80.7 ms/token) short-context and 10.53 tok/s at a warmed 4k
> context (10.33 tok/s on the preceding check);
> the remaining work is a complete EP=1 quality
> run, sampling parity, and TP support. EP=8 is not a quality-valid full-model
> result because it owns only a routed-expert shard.
>
> Constraints that matter: the model headers `common/ds4f.h` / `ds4f_impl.h` /
> `ggml_dequant.h` are shared with the A64FX/Fugaku runners in `a64fx/llm`, so
> run `make -C hetero/ds4f sve-check` after touching them. Run
> `make -C hetero/ds4f test` after touching any kernel. Do not commit or push
> unless I ask.

## Where things stand

Goal is to beat llama.cpp on this box (7.33 tok/s full MXFP4, 8.57 tok/s Q3_K_M
— see `../../llama.cpp/da4f.md`) without giving up quantization quality.

**Done and measured:**

- Expert-gather roofline: **43.3 GB/s** of a 48.1 GB/s ceiling ⇒ 12.55 tok/s for
  the expert path alone. Target is reachable.
- The DS4F model implementation builds and runs on x86, sharing one codebase
  with the A64FX runners. Six AVX2 decode kernels, all verified.
- `DS4F_STAGE_NOCOPY`: 156 GB staged in 0.1 s, zero bytes copied.
- Zero-copy experts: arena 155.4 → 8.21 GB, experts phase 2505 → 143 ms.

**Current full-model decode, CPU only, ep_size=1, 16 threads: 675 ms/token.**
Of that, ~587 ms is the FP8 dense path (`o_proj` 273, `qkv_proj` 170, `shared`
104) and 108 ms is the routed experts.

**Current HIP hybrid EP=8 mechanical decode, 16 threads: 80.7 ms/token,
12.40 tok/s short-context.** The 43-layer bank has 345 matrices and 6.741 GB
resident: 344 FP8 MLA/shared matrices plus the replicated BF16 vocabulary
head. The gfx1201 row kernel defaults to 128 threads (256 remains selectable
with `DS4F_HIP_BLOCK_THREADS`). The short-context phase profile is qkv 22.5
ms, attention 0.7 ms, o_proj 23.4 ms, shared 10.7 ms, routed experts 14.4 ms,
and head 2.4 ms. The head A/B gate passes at `max_rel=1.60e-5`; all FP8
layer-0 A/B gates remain below `1.67e-6`.

**Long-context harness and residency work:** `test_hip_ds4f_real` now accepts
`DS4F_MAXPOS`, `DS4F_HIP_CTX_WARM`, and `DS4F_HIP_POS0`, allowing a direct
measurement at positions 4096+ with an 8192-position KV allocation. The
no-copy loader has a guarded `DS4F_EXPERT_RESIDENT` mode: `=1` queues
readahead; `=2` synchronously touches a 2-GB default cap, adjustable with
`DS4F_EXPERT_RESIDENT_GB`. The exact BF16 attention window now uses AVX2
widen/dot/AXPY helpers on x86. The final-binary serial EP8 check measures
10.53 tok/s at position 4096 (94.95 ms/token); the preceding cleanup check
measured 10.33 tok/s, so both clear the 10 tok/s long-context gate.
Short-context decode is 12.40 tok/s.

**MTP/DFlash exploration:** `test_ds4f_mtp` is now a guarded draft/verify
probe. The loader cleanly disables MTP when `mtp.0.*` is absent (the current
EP8 manifest has no MTP tensors). When a full checkpoint supplies MTP, the
probe measures exact sequential acceptance but deliberately claims no speedup
until MTP KV-prefix bootstrap, rollback, and batched verification are wired.
No DS4F DFlash draft checkpoint or implementation is present yet.

**Configuration/ownership cleanup:** AMD-facing harnesses and the server now
load a shared `ds4f_runtime_options` object from JSON, with explicit CLI values
overriding JSON. `--debug-env` is the compatibility path for legacy environment
settings. Model, session, request, prefill, and test scratch are backed by the
mmap-based `ds4f_mem_pool`; normal DS4F code no longer has raw allocator
ownership. The x86 prefill path now has AVX2 attention, a batched GEMM worker,
two-token BF16/FP8 weight-reuse microkernels, multi-GEMM barrier fusion, and
routed-expert slab staging (all local gate/up, SwiGLU, and down work shares one
dispatch per phase), with `make -C hetero/ds4f prefill-test` checking argmax
parity. The real-staged one-layer gate now reports 42.83 tok/s versus 12.67
tok/s for token-at-a-time at batch 16 (3.38x, zero argmax mismatches); batch
128 reaches 51.85 tok/s, and `pos0=4096` reaches 45.95 tok/s. The complete
43-layer CPU gate is 1.63 tok/s at batch 16.
The explicit resident-BF16 profile (`dense_bf16=1`, `bf16_pv=1`) reaches
67.73 tok/s at batch 16, 86.69 tok/s at batch 128, and 63.17 tok/s at
`pos0=4096`, all with zero mismatches; it costs roughly +6 GB, so FP8 remains
the default.

## Remaining tasks

### S2 — CPU expert-path tuning — implemented

The model path gets 24 GB/s where S0's standalone benchmark got 43.3. Two known
causes:

1. **Dispatch overhead.** `ds4f_matvec` runs one pool dispatch per expert per
   tensor: 6 × 3 × 43 = **774 spin-barrier dispatches per token**, against S0's
   86. At ~30 µs each that is ~23 ms of the 143 ms. Fix: batch the 6 active
   experts' w1/w3 into one dispatch and their w2 into a second, the way
   `bench_expert_bw.c`'s `job_gateup`/`job_down` do. The MoE call site is around
   `ds4f_forward_token`'s expert loop in `common/ds4f_impl.h` (search
   `DS4F_P_EXPERTS`).
2. **Page-cache faulting.** Expert weights are now file-backed, so first touch
   of each expert pays a fault S0's prefaulted anonymous region did not. Test
   `MADV_HUGEPAGE` and/or `MAP_POPULATE` on the shard mapping in
   `ds4f_blob_open`. Note the model is 147 GB of experts against ~180 GB of
   usable page cache, so steady state should be nearly all resident — verify
   that rather than assuming it.

Measured result: the two-dispatch batch path reaches 107.8 ms/token on the full
model. A bounded file-backed residency policy is now available through
`DS4F_EXPERT_RESIDENT`; the final quiet-host long-context check passes at
10.53 tok/s at position 4096 (10.33 tok/s on the preceding check).

### S3 — HIP dense offload (the big one)

This is where the end-to-end win is: 587 of 721 ms/token. Put the FP8 dense
path (MLA + shared expert, 8.85 GB — fits the 16 GB card) on the 9070 XT and
keep only the MXFP4 experts on the CPU, overlapping the GPU shared expert with
the CPU routed experts.

S3a is implemented as a standalone matvec gate in
`hetero/ds4f/hip_ds4f_dense.{c,h}`, `hip_ds4f_kernels.h`, and
`test_hip_ds4f_dense.c`, reusing
`rdna4/rocew.h`, `rdna4/hip_runner_common.h`, `rdna4/hip_kernels_common.h`
(plain gcc, HIP dlopen'd, kernels via HIPRTC — no ROCm SDK at build time; copy
the `rdna4/dequant/` Makefile pattern).

`make -C hetero/ds4f all` builds the runner. `make -C hetero/ds4f hip-test`
performs the CPU comparison when a HIP device is available and reports a
non-failing SKIP otherwise. `forward-hip-test` performs a synthetic model-level
callback A/B. The real staged harness validates all eight layer-0 FP8 tensors,
the 43-layer bank capacity (344 FP8 matrices, 5.682 GB), and the full multi-layer
attachment/decode path. The runner retains device weights and exposes both a
one-in-flight stream/event API and a two-slot async tensor API. The common model
gate uses the two-slot path for shared `w1/w3`.

Order:
1. **S3a/S3b done:** FP8-E4M3 + 128×128 E8M0-block matvec, persistent bank,
   and synthetic/real CPU A/B gates. The standalone GPU gate reaches
   `max_rel=8.4e-6`; the real one-layer hybrid forward matches argmax with
   `logits_rel=5.01e-6`. Closest references:
   `rdna4/dequant/bench_dequant.c` (mxfp4 GPU dequant), `rdna4/fp8/` (FP8 WMMA).
2. **S3c real attachment:** done in the staged real harness. All eight FP8
   dense tensors for every real layer are bound to the bank; grouped `wo_a`,
   the replicated BF16 vocabulary head, mHC, router, and attention retain
   their intended CPU/GPU split; MXFP4 experts remain CPU-side.
3. **S3c production attachment:** done in the same harness. The multi-matrix
   async launch keeps shared w1/w3 in flight while CPU routing and routed
   experts run, then synchronizes before shared w2/residual combine. Grouped
   `wo_a` is a single GPU block-diagonal launch, and the replicated BF16
   vocabulary head is resident on the GPU. EP=8 now measures **80.7 ms/token /
   12.40 tok/s** short-context and **94.95 ms/token / 10.53 tok/s** at a warmed
   4k position (10.33 tok/s on the preceding check); repeat with EP=1 before
   treating it as a quality/performance result.
4. mHC pre/post (Sinkhorn, 20 iters) and the router; keep them CPU-side first.
5. Lightning indexer / Tier-B2 for sparse layers — biggest single kernel item.
   **Keep it on the CPU initially** (layers 0, 1 and the last are ratio 0
   anyway) and offload only after the dense path is validated.
6. hipGraph-capture the per-layer GPU segment. Read
   `rdna4/llm/decode-graph-capture-audit.md` first.

A/B every offloaded op against its CPU counterpart before enabling it, the same
discipline as `rdna4/llm/test_hip_llm.c --verify-quant-kernels`. The CPU path is
the reference, which is why S1 built it.

Per-layer PCIe traffic is ~16 KB each way; 86 round trips/token ≈ 1 ms, which is
noise against the ~80 ms CPU expert time.

### S4 — OpenAI-compatible server

The native x86 phase is implemented in `server/server_ds4f.{c,h}` and wired
into `server_llm` behind `DIFFUSION_SERVER_ENABLE_DS4F_HETERO`. It loads the
staged safetensors model, invokes the stdlib-only tokenizer without a shell,
serves OpenAI text/chat endpoints with greedy or seeded temperature/top-p
decoding, and maintains one exact prompt-prefix KV plus logit snapshot. A real
EP=8 HTTP smoke passed twice through the same process (same text output, 11
prompt tokens); the chat path also passed with 27 prompt tokens.

The persistent HIP dense bank is now attached behind `DS4F_HIP=1`: the server
uploads 344 FP8 MLA/shared matrices plus the replicated flat-BF16 vocabulary
head (345 matrices, 6.741 GB) and leaves routed MXFP4 experts, router, and
embedding on the CPU. An EP=8 GPU HTTP smoke passed twice through one process
with identical output and prompt-token counts. A repeated seeded request also
returned the same sampled text after a prefix-cache hit. Remaining S4 work is
TP-sharded embedding/head state. The
A64FX tree already has
`a64fx/llm/ds4f_serve.py` plus file/socket runner protocols for multi-node
deployments.

### Batched GPU prefill follow-up

The HIP dense callback now handles prefill GEMMs with the existing RDNA4 16x64
tile geometry. FP8/E8M0 dequantization is fused into the tile, and flat BF16
weights use a BF16-correct sibling kernel rather than the RDNA4 FP16-weight
entry point. `ds4f_gemm_multi()` sends eligible FP8/BF16 projection groups to
the GPU and keeps MXFP4 experts, row-slice views, and failed/unsupported calls
on the CPU fallback.

Validated on the staged EP=8 real shard, one layer, 16 CPU workers:

* batch 16: 353.38 tok/s GPU versus approximately 86 tok/s CPU, 0 mismatches;
* batch 64: 589.23 tok/s GPU versus approximately 86 tok/s CPU, 0 mismatches;
* batch 128: 608.81 tok/s GPU versus approximately 86 tok/s CPU, 0 mismatches.

The current exact default on the complete staged 43-layer shard measures
3.450 tok/s CPU versus 21.388 tok/s hybrid GPU at batch 64 (6.20x), with zero
argmax mismatches. Batch 128 reaches 22.401 tok/s but currently has 3/128
mismatches, so it remains a performance result pending numerical tuning.

The standalone FP8 and BF16 batched GEMM gates pass (`2.31e-5` and `1.91e-6`
max relative error respectively). This is a dense-offload layer gate, not a
full-model prompt benchmark: projection intermediates still cross host/device
memory, while attention, norm, router, and MXFP4 experts remain CPU-side in
the exact default. Shared-input GEMM pairs upload activations once, grouped
`wo_a` row slices reuse the resident bank, and x86 prefill attention reuses
each BF16 KV row across eight heads. A device-resident prefill arena and fused
GPU attention/MLP are the next steps needed to approach the 30-tok/s target.

An opt-in mixed-BF16 experiment (`hip_shared_bf16=1`, optionally capped with
`hip_shared_bf16_layers`) reached 34.40 tok/s at batch 64 by promoting hot
shared weights and the router to BF16 GPU matrices. It currently changes one
greedy output, so it remains an approximate speed mode rather than the exact
default.

The separate `hip_shared_fp16=1` experiment expands only shared FP8 weights to
exact FP16 values and reuses the RDNA4 FP16-weight tile. Its one-layer gate is
753.55 tok/s with zero mismatches; full 43-layer batch-64 prefill reaches
33.64 tok/s but accumulates one argmax change from the alternate reduction
path, so it is also opt-in.

### Correctness debt (do before trusting any output)

- **W4A8 accuracy gate is partially closed.** The real-token EP=8 gate now
  compares exact-f32 and W4A8 on 11 teacher-forced tokens: 11/11 argmax
  matches, worst logit relative error 6.37%, and mean cross-entropy delta
  -0.00031. Repeat against the complete EP=1 model and a PyTorch/reference
  token stream before treating this as a quality result. If W4A8 does cost
  measurable quality, the fallback is f32 activations on the `w2`
  down-projection only (⅓ of the traffic, ~12.6 → 11.4 tok/s).
- **Prefill is functional on x86.** The AVX2 attention worker preserves the
  exact sliding-window/sink/de-rotate behavior, and the supported x86 GEMM
  types use a batched worker with two-token BF16/FP8 weight reuse and fused
  independent-GEMM barriers. Routed expert assignments are staged in one
  mmap-pool slab, removing the per-expert barriers from the gate/up, SwiGLU,
  and down phases. The real one-layer gate reaches 42.83 tok/s at batch 16
  and 51.85 tok/s at batch 128; the complete 43-layer CPU path is 1.63 tok/s
  because layers and the rest of the model remain serial.
- **Real generation is now available in the single-node runner.** The
  stdlib-only tokenizer writes `DS4F_PROMPT_IDS`, and
  `a64fx/llm/ds4f_runner.c` writes greedy output IDs via `DS4F_GEN_OUT`.
  The same tokenizer is now used by the native x86 server session and its
  OpenAI-compatible text/chat request paths.

### MTP/DFlash follow-on

`test_ds4f_mtp` provides a guarded MTP draft/verify acceptance probe and the
loader cleanly skips MTP when `mtp.0.*` is absent. The current EP8 manifest has
no MTP tensors. Before claiming speculative speedup, bootstrap MTP KV state for
the prompt, snapshot/restore partial accepts, attach its dense tensors to the
HIP bank where useful, and measure the existing mHC+Tier-B2 batched verifier.
DFlash has no compatible draft checkpoint or DS4F implementation in this tree.

### Known-good opportunistic wins

- The FP8 AVX2 kernel uses `vpgatherdd`, slow on Zen1. An arithmetic e4m3 decode
  would be ~2×. Deliberately not done because this path moves to the GPU in S3
  — only worth it if a fast CPU-only dense path is ever wanted.
- `ds4f_gemm`'s tiled/PV/tile-dequant kernels are SVE-only; off A64FX the
  supported BF16/FP8/MXFP4 types now use an AVX2 batched worker. Unsupported
  tile formats still fall back to the correct token path.

## Gotchas that cost time

- **`LLM_THREADS` defaults to 48** (an A64FX number). On this 32-thread part
  that oversubscribes and, with spin-wait barriers, costs **5.4×** (3150 → 580
  ms/tok). Always pass `LLM_THREADS=16 DS4F_CMGS=1`.
- **SMT hurts.** 32 threads measured 42.7 GB/s vs 47.6 at 16. Matches
  llama.cpp's own finding that 12 threads beat 16 and 24.
- **`-fsyntax-only` does not catch inline-asm errors.** The ARM `yield` in
  `ds4f_relax` only failed at assembly time. Actually build, don't just
  syntax-check.
- **MXFP4 experts are repacked at load, not copied** (nibble permutation +
  `e8m0 e-1`). I initially assumed a straight copy and was wrong; that
  assumption is what made zero-copy look like a loader-only change.
- **`ggml_e8m0_to_fp32_half` is not equivalent to the repack's `e-1`** — they
  disagree at e≤1 (repack flushes to +0.0, `_half` keeps a denormal). Use the
  explicit decrement.
- The runner prints `arena reservation: 144.78 GB` before loading even under
  zero-copy; it calls `ds4f_arena_size` itself without the flag. Cosmetic, the
  real figure is on the `ds4f_load_real` line.

## Repro

```bash
cd hetero/ds4f
make && make test && make sve-check

# stage a no-copy manifest (ep_size=1 = whole model; 8 or 2 for a smaller slice)
DS4F_STAGE_NOCOPY=1 DS4F_MODEL_DIR=/mnt/disk1/models/ds4f-0731 \
DS4F_STAGE_DIR=$PWD/stage DS4F_EP_RANK=0 DS4F_EP_SIZE=1 DS4F_NSHARDS=48 \
  ./build/ds4f_stage

# full-model decode
DS4F_REAL=1 DS4F_EXACT=1 DS4F_STAGE_DIR=$PWD/stage DS4F_EP_RANK=0 DS4F_EP_SIZE=1 \
LLM_THREADS=16 DS4F_CMGS=1 DS4F_MAXGEN=64 DS4F_PROF=1 ./build/ds4f_runner

# roofline benchmark
./build/bench_expert_bw --mode matvec --i8seq --gib 90 --threads 16 --tokens 6
```

Toggles: `DS4F_ZEROCOPY_EXPERTS=0` (repack into arena),
`DS4F_MXFP4_W4A8=0` (exact-f32 expert kernel).

The 64-token run takes ~83 s to load plus ~46 s to decode. Use `DS4F_EP_SIZE=8`
(27 GB, loads in 2 s) for quick mechanical checks — output is wrong, since only
1/8 of the experts are present, but the timings and code paths are real.
