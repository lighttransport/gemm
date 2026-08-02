# hetero/ds4f — resume notes

Snapshot: 2026-08-03. Commit `da6b61a8` on branch `ds4f`.

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
> The next task is S2: close the 24 → 43 GB/s gap on the CPU expert path.
> After that, S3: the HIP dense offload, which is where the remaining
> end-to-end win is (587 of the current 721 ms/token).
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

**Current full-model decode, CPU only, ep_size=1, 16 threads: 721 ms/token.**
Of that, ~587 ms is the FP8 dense path (`o_proj` 273, `qkv_proj` 170, `shared`
104) and 143 ms is the routed experts.

## Remaining tasks

### S2 — CPU expert-path tuning

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

Worth roughly 143 → 80 ms/token if both land.

### S3 — HIP dense offload (the big one)

This is where the end-to-end win is: 587 of 721 ms/token. Put the FP8 dense
path (MLA + shared expert, 8.85 GB — fits the 16 GB card) on the 9070 XT and
keep only the MXFP4 experts on the CPU, overlapping the GPU shared expert with
the CPU routed experts.

New `hetero/ds4f/hip_ds4f_dense.{c,h}` + `hip_ds4f_kernels.h`, reusing
`rdna4/rocew.h`, `rdna4/hip_runner_common.h`, `rdna4/hip_kernels_common.h`
(plain gcc, HIP dlopen'd, kernels via HIPRTC — no ROCm SDK at build time; copy
the `rdna4/dequant/` Makefile pattern).

Order:
1. FP8-E4M3 + 128×128 E8M0-block matvec. Closest references:
   `rdna4/dequant/bench_dequant.c` (mxfp4 GPU dequant), `rdna4/fp8/` (FP8 WMMA).
2. MLA decode: `wq_a→q_norm→wq_b`, `wkv`, per-head norm + YaRN RoPE, KV-latent
   append, attention over the compressed latent (adapt `attn_decode_f32` and
   `rdna4/fa2/`), block-diagonal `wo_a` (8 groups) + `wo_b`.
3. mHC pre/post (Sinkhorn, 20 iters) and the router; shared expert.
4. Lightning indexer / Tier-B2 for sparse layers — biggest single kernel item.
   **Keep it on the CPU initially** (layers 0, 1 and the last are ratio 0
   anyway) and offload only after the dense path is validated.
5. hipGraph-capture the per-layer GPU segment. Read
   `rdna4/llm/decode-graph-capture-audit.md` first.

A/B every offloaded op against its CPU counterpart before enabling it, the same
discipline as `rdna4/llm/test_hip_llm.c --verify-quant-kernels`. The CPU path is
the reference, which is why S1 built it.

Per-layer PCIe traffic is ~16 KB each way; 86 round trips/token ≈ 1 ms, which is
noise against the ~80 ms CPU expert time.

### S4 — OpenAI-compatible server

Extend `server/server_llm.c` behind a new `DIFFUSION_SERVER_ENABLE_DS4F_HETERO`
CMake option. Needs the DS4F chat template (model's `encoding/` dir) and
single-slot prompt-prefix caching. Target parity with the llama.cpp launcher's
Codex / Claude Code integration.

### Correctness debt (do before trusting any output)

- **W4A8 accuracy gate is weak.** Current evidence is argmax agreement on a
  harness that feeds *random embeddings*. That is suggestive, not a quality
  gate. Needs real token input and a logit comparison against `ref/`. If W4A8
  does cost measurable quality, the fallback is f32 activations on the `w2`
  down-projection only (⅓ of the traffic, ~12.6 → 11.4 tok/s).
- **Batched prefill attention is SVE-only** and aborts off A64FX
  (`ds4f_attn_prefill_worker` has no scalar twin). Token-at-a-time prefill
  works. Either write the scalar/AVX2 twin or let S3 put prefill on the GPU.
- **No real generation yet.** Everything so far uses the synthetic-embedding
  harness `a64fx/llm/ds4f_runner.c`. There is no tokenizer wired up and no
  `ds4f_x86_runner` that produces actual text.

### Known-good opportunistic wins

- The FP8 AVX2 kernel uses `vpgatherdd`, slow on Zen1. An arithmetic e4m3 decode
  would be ~2×. Deliberately not done because this path moves to the GPU in S3
  — only worth it if a fast CPU-only dense path is ever wanted.
- `ds4f_gemm`'s tiled/PV/tile-dequant kernels are SVE-only; off A64FX every
  dtype falls through to per-token matvec. Correct, but prefill is unbatched.

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
