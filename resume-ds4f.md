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

## 2026-08-04: prefill and decode targets met (exact)

Both standing performance targets are now met on the staged EP=8 shard with
**zero argmax mismatches** against the CPU reference:

| | before | now | target |
|---|---:|---:|---:|
| prefill, batch 64, exact | 22.0 tok/s | **39.9--40.3** | 30+ |
| prefill, batch 128 | 22.4 (3/128 mismatch) | **41.0--41.6** (0/128) | — |
| prefill, 4k / 8k context | 21.2 / 21.0 | **36.2 / 36.2** | — |
| decode, short context | 12.28 tok/s | **18.78--18.87** | 15+ |
| decode, warmed 4k | 10.53 tok/s | **14.13--14.21** | — |

Prefill was measured at 39.9--40.3 tok/s on a quiet host and 37.5--38.7 during a
later session with `load average: 21.67` and 7 other users; the CPU reference
dropped in lockstep (3.50 -> 3.34 tok/s) and the GPU/CPU ratio held at
11.2--11.5x, so the spread is host contention rather than a regression. Re-measure
on a quiet host before quoting a single figure.

Note this host is a Threadripper 1950X (Zen1, AVX2), not A64FX/SVE — the CPU
work below targets the AVX2 path only.

Six changes, all small:

1. **`gpu_dense_mixed` was gated on the approximate precision flags**
   (`test_hip_ds4f_real.c`, three sites). Mixed dispatch is a routing policy,
   not a precision mode. With it off, `ds4f_gemm_multi` sends an entire
   independent group to the CPU when any one member is CPU-owned, and the
   shared-expert group pairs `sh_w2` with the CPU-resident router `gate` — so
   exact prefill was paying 19.0 ms/token of CPU `sh_w2`. Enabling it
   unconditionally drops `shared` to 2.6 ms/token. Pair it with
   `--hip-ordered-fp8-layers 43` to keep `sh_w2`'s reduction order identical to
   the AVX2 reference; that is what makes 35.28 tok/s a 0-mismatch number.
   This also retires `hip_shared_bf16` / `hip_shared_fp16` as speed levers —
   they were only ever switching mixed dispatch on as a side effect.
2. **The FP8 decode matvecs read one byte per lane per iteration.** They now
   use `uint4` (16 weights/lane) with a scalar tail. A 16-aligned run cannot
   straddle a 128-column E8M0 block, so the scale applies once per run. `wq_b`
   went 132 -> 182.8 GB/s. The reassociation moves results *toward* the CPU
   reference: worst layer-0 `max_rel` 1.67e-6 -> 1.10e-6, drift `x_rel`
   0.0547 -> 0.0159, argmax unchanged.
3. **The AVX2 MXFP4 expert GEMM decoded every weight once per token.** In
   `ds4f_gemm_worker_x86`'s raw-MXFP4 branch the token loop was *outermost*, so
   a prefill tile of M tokens ran `ds4f_mxfp4_unpack16` M times over the same
   weights. That unpack -- mask, shift, two `pshufb`, four sign-extends, four
   int-to-float converts -- is roughly fifteen ops to produce one 32-weight
   block against six ops to consume it for one token. BF16 and FP8 both had
   two-token microkernels; MXFP4 had none. Added
   `matvec_mxfp4_1row_f32_raw_4x` (and the W4A8 sibling
   `matvec_mxfp4_1row_i8_raw_4x`), quantize/permute the whole tile up front,
   and hoist the row loop outside the token loop with token blocks innermost
   so each row's 2 KB of weights stays in L1 across the tile. Every token keeps
   its own accumulator pair and accumulates in the original order, so each
   output is bit-identical to the 1-row kernel. Experts went 13.43 -> 9.78
   ms/token and prefill 35.0 -> 40.2 tok/s.

   Two gotchas cost time here and are worth knowing. First, **the exact path
   uses the f32 kernel, not W4A8** (`ds4f_mxfp4_w4a8_on` is 0 under
   `--hip-ordered-fp8-layers`), so the W4A8 4x kernel written first was
   unreachable and measured exactly zero change; it is kept because the W4A8
   mode still uses it. Second, my a-priori estimate that per-expert buckets
   average ~1.5 tokens (256 experts, top-6, EP=8, batch 64) was **wrong** --
   measured over 43 layers the histogram is 1:75 2:41 3:42 4:24 5:27 6:19
   7:10 >=8:90, i.e. most routed work sits in buckets of 8 or more. Measure
   the bucket distribution before reasoning about expert batching.
4. **Decode paid a full synchronous round trip per GPU tensor.** Two parts.
   `DS4F_MV_FUSE` defaults to 0, so decode's independent `wq_a`/`wkv` pair ran
   as two separate dispatches; it is bit-exact by construction (same rowsplit,
   kernel and per-row dot order, only the barrier is shared) and the harness
   now sets it with `setenv(..., 0)` so an explicit env still wins. The shared
   default is deliberately left at 0 because it has not been re-measured on
   A64FX. Then `ds4f_matvec_multi`'s GPU branch issued one launch + event sync
   + blocking download *per tensor*; it now puts the whole group in flight via
   the existing `gpu_dense_async_multi`/`gpu_dense_wait` pair (slots raised
   from `HIP_DS4F_ASYNC_MAX` 2 to 8) and falls back to the serial loop if the
   adapter is already holding a batch. Measured separately: fuse alone
   15.09 -> 15.55, plus async group -> 16.24 (that pair went 7.92 -> 5.21 ms).
5. **Every host transfer was pageable** (`grep -c hipHostMalloc` was 0), so the
   driver bounced each one through its own staging buffer. `wq_b` downloads
   131 KB per layer, 5.6 MB/token, and measured 11.96 ms against a 7.9 ms
   kernel. The matvec H2D and D2H paths now stage through pinned buffers and
   the async group queues all downloads before synchronizing, so they overlap.
   This changes no arithmetic -- every A/B `max_rel` is unchanged -- and took
   decode 16.24 -> 17.75 tok/s, `wq_b` 11.96 -> 9.69 ms. `ensure_pinned` returns
   -1 if the loaded driver lacks `hipHostMalloc`, keeping the pageable path.

6. **The matvec gave each row a whole 128-thread block.** For `wq_b`
   (cols=1024) that is 64 chunks of work spread over 128 lanes -- half of them
   idle -- plus a shared-memory reduction tree and a `__syncthreads` per row.
   All four matvec kernels now run **one wave32 per row**, `blockDim.x/32` rows
   per block, so the row total is a single shuffle reduction with no shared
   memory and no barrier, and the block count drops 4x. `wq_b` went 182 -> 280
   GB/s and 9.69 -> 6.80 ms; qkv 14.8 -> 11.9 ms, head 2.5 -> 1.9 ms. Block size
   no longer matters much (64/128/256 threads all measure 18.78--18.81 tok/s),
   which is the expected signature of the row being wave-local.

   This reduces each row over 32 lanes instead of 128, so the summation order
   changes. Layer-0 A/B errors stay in the same band (dense <= 1.5e-6, head
   1.48e-5, all PASS), the full-model CPU-vs-GPU forward argmax is unchanged at
   both pos0=1 and pos0=4096, prefill CPU-parity stays 0/64 at 4k and 8k, and
   the short-context decode rollout still ends on 103035 in every run. What
   *did* change is `last_argmax` of the **8-token 4k rollout**, 10371 -> 16.
   That rollout is autoregressive over a *synthetic* warmed KV context with no
   CPU reference, so one flip diverges the whole tail; treat it as a
   self-consistency marker for a given binary, not a correctness gate.

   **That check now exists and passes.** `--decode-verify N` (new,
   `decode_verify()` in `test_hip_ds4f_real.c`) runs N teacher-forced decode
   steps against the CPU reference as a KV history accumulates: at each
   position the CPU pass runs first, the GPU pass reruns the *same* position so
   both read an identical history and write the same cache slot, and the GPU
   result is what survives into the next step. Results:

   | | steps | argmax_mismatch | worst_logit_rel |
   |---|---:|---:|---:|
   | `--pos0 1` | 8 | **0** | 0.047 |
   | `--pos0 4096 --warm 4096` | 8 | **0** | 1.01 |

   So the GPU decode path tracks the CPU reference at 4k and the `10371 -> 16`
   change really was the chaotic rollout. The `worst_logit_rel` of 1.01 at 4k
   is worth knowing but is against a *synthetic* warmed KV context, which
   produces unrealistic activations -- the same measure is 0.047 at pos0=1 with
   no synthetic warm, so treat the 4k figure as an artifact of the warm, not a
   quality claim. A real-prompt long-context check is still the honest way to
   validate long generations.

The decode phase split is now qkv 11.9, o_proj 11.9, shared 8.0, router 2.1,
experts 15.4, head 1.9 ms. **The CPU MXFP4 expert phase is now the single
largest term at 29.5%.** It is bandwidth-bound, and the PCIe finding below
shows that moving it to either GPU by streaming is strictly worse -- the only
viable GPU route is full expert residency split across both cards.

Falsified along the way, recorded so they are not retried:

- The README's `"hip_mxfp4_resident_layers": 20` -> 35.4 tok/s claim **does not
  reproduce here**. `=20` fails outright ("GPU MXFP4 resident upload failed at
  layer 12") because the 6.741 GB dense bank leaves too little VRAM, and
  `--hip-mxfp4-resident-auto 1` picks 9 layers and measures **0.214 tok/s** —
  a 100x regression, not a win. The widened-MXFP4 resident path is unusable in
  its current form.
- Staging the 256-entry E4M3 LUT in LDS, and re-indexing the FP8 GEMM's LDS
  weight staging for coalesced loads, each moved the shared phase by under
  0.15 ms/token (19.08 -> 18.96 -> 19.00). Both were reverted. The FP8 GEMM
  tile is not the prefill bottleneck at these shapes.
- **The host link is PCIe gen3 x8, and that closes the GPU-expert route.**
  `nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current` reports
  `3, 8` -- about 7.9 GB/s theoretical, ~6.5 GB/s practical. This single fact
  explains a pile of earlier results and should be checked before any future
  offload plan:

  * The HIP raw-LUT MXFP4 GEMM benchmarks at 1.122 ms/call for one
    2048x4096 expert matrix (`--hip-mxfp4-gemm-test`). That matrix is 4.19 MB,
    so the call is running at **3.7 GB/s -- it is PCIe-bound on the per-call
    weight upload, not compute-bound.** The widened FP8 variant is 0.927 ms,
    same story.
  * Decode touches roughly **431 MB of expert weights per token**
    (0.75 owned experts/layer x 43 layers x 13.35 MiB). Streaming that over
    the link costs ~66 ms/token against the CPU's measured 15.4 ms. Sending
    experts to *either* GPU is about **4x worse than just computing on the
    CPU**, and that is before counting the RAM read the host must do anyway to
    supply the upload.
  * It also explains the 0.214 tok/s `hip_mxfp4_resident_auto` result above:
    9 layers resident and 34 layers streaming ~340 MB each is ~1.8 s of pure
    PCIe per prefill batch.

  **So routed experts belong on the CPU unless they are fully resident.** The
  one arrangement that could work is residency *split across both cards*: the
  owned expert bank is 32 experts x 43 layers x 13.35 MiB = **17.9 GiB**,
  against ~14.4 GiB free on the RTX 5060 Ti plus ~9.5 GiB left on the 9070 XT
  after the 6.741 GB dense bank -- about 23.9 GiB combined, so it fits with
  room to spare. Only activations would cross the link (~16 KB per expert
  call, negligible even on gen3 x8). That, not a faster kernel, is the real
  remaining decode lever, and it is a substantial piece of work: routing-aware
  dispatch across two vendors' runtimes with an exact kernel on each.
- **CPU expert bandwidth is not the lever it looks like.** `bench_expert_bw
  --mode matvec --i8seq --threads 16` measures **44.3 GB/s** on this host
  (prefaulted anonymous), and the model path runs the expert stream at roughly
  26 GB/s. Neither `DS4F_EXPERT_RESIDENT=2` with a 30 GB cap (pages were
  already cached; 15.22 vs 15.19 tok/s) nor a thread sweep moved it: 8 threads
  15.21, 12 threads 14.51, 16 threads 15.22, 24 threads 15.17, 32 threads
  **9.64** tok/s. 16 threads remains right and SMT remains catastrophic. The
  gap to the synthetic ceiling is a scattered file-backed access pattern, not
  page-cache residency or thread count. The win came from arithmetic reuse
  (item 3), not from bandwidth.
- This box is a single NUMA node (`numactl --hardware`: 1 node, 193 GB), so
  there is no interleave/affinity lever to pull.
- hipGraph capture remains inapplicable: `rdna4/llm/decode-graph-capture-audit.md`
  rules out MoE models (host-side top-K), and DS4F additionally runs host
  rmsnorm/RoPE/attention between GPU ops.

Not done, in value order, if more decode headroom is wanted: `ds4f_matvec_multi`
still issues one synchronous round trip per tensor (`common/ds4f_impl.h:698`),
so decode qkv pays ~129 launch+sync round trips per token where the async
two-slot API already used for the shared expert would need 43; host staging is
still fully pageable (`grep -c hipHostMalloc hip_ds4f_dense.c` is 0); and the
dual-GPU CUDA prefill regression (9--11 tok/s) is untouched — its root cause is
that `cuda_eligible()` requires `M >= 128`, which no routed-expert bucket
reaches at batch 64, while `dual_ds4f_prefill_bind_tensor` gives MXFP4 tensors
a `gpu_id = 0` sentinel without uploading them to HIP, so experts land on the
CPU on both paths.

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

The FP8 batched GEMM tile now uses a resident 256-entry E4M3FN LUT on the
device, removing repeated decode branches while preserving the standalone
correctness result. The latest exact full-shard runs are about 22.1 tok/s at
batch 64 (profiled runs vary around 21.3--22.1 tok/s), with zero argmax
mismatches.

An opt-in mixed-BF16 experiment (`hip_shared_bf16=1`, optionally capped with
`hip_shared_bf16_layers`) reached 34.40 tok/s at batch 64 by promoting hot
shared weights and the router to BF16 GPU matrices. It currently changes one
greedy output, so it remains an approximate speed mode rather than the exact
default.

The separate `hip_shared_fp16=1` experiment expands only shared FP8 weights to
exact FP16 values and reuses the RDNA4 FP16-weight tile. Its one-layer gate is
753.55 tok/s with zero mismatches; full 43-layer batch-64 prefill reaches
about 34--35 tok/s but accumulates one argmax change from the alternate reduction
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

### Long-context batched-prefill stability

The real HIP prefill harness accepts `--prefill-context N`. It warms the final
128-token sliding attention window and measures a batch of 64 at position N.
On the staged EP=8 shard, GPU throughput at 4k/8k was: exact `21.20/21.01`
tok/s, shared BF16 `32.89/32.05` tok/s, and shared FP16 `32.83/31.73` tok/s.
The corresponding CPU-vs-GPU argmax mismatch counts were exact `5/4`, BF16
`5/5`, and FP16 `1/4` out of 64. Throughput is stable; the mismatch counts
reflect separately warmed CPU/GPU KV histories and accumulated reduction drift,
not a GPU-only determinism check.

To guarantee CPU-reference parity for prompt batches, `hip_exact_prefill` is
now available in JSON or as `--hip-exact-prefill 1`. It disables only M>1 GPU
GEMMs, preserving GPU M=1 decode, and measured 0/64 mismatches at both 4k
(3.33 tok/s) and 8k (3.34 tok/s). The default remains the fast GPU prefill
mode because the exact mode intentionally pays the CPU reference cost.

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
