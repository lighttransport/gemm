# Qwen3.8 bounded MoE staging

The staging path is opt-in. Passing repeatability does not establish scalar F16
parity or the 200 prefill / 30 decode tok/s targets. See `QWEN38_STATUS.md` for
matched runtime measurements and `resume.md` at the repository root for next work.

## Ownership

`--qwen4-prefill-staging` retains the two-bank quantized pool. Its default
512 MiB budget includes both banks' weights, device metadata, and Q8_0 repack
scratch. Host metadata is separately pinned; model weights may be pageable or
registered. Each bank holds an expert map, task expert IDs, and task positions;
none aliases the router's host or device scratch.

For each bank reuse:

1. Finish the previous metadata upload before rewriting its host source.
2. Make the copy stream wait for the bank's last compute/promotion consumer.
3. Upload all wave weights and metadata; repack Q8_0 on that same copy stream.
4. Record upload readiness; make compute wait before launching gate-up/down.
5. Record consumption after down projection and any cache promotions.

`LLM_MOE_COPY_PIPELINE=0` uses one bank and serializes waves. `=1` alternates
banks, allowing the next wave's upload to overlap the current wave's compute.
Resident experts execute first from an immutable cache-map snapshot. Cold waves
use descending request-local score, with ascending expert ID for ties. Each
assignment writes its own existing output slot; final top-K combination remains
in the existing fixed rank order.

Promotions remain off by default (`LLM_QWEN4_STAGE_PROMOTE=0`). When enabled,
promotion copies and map-publication kernels precede the bank's consumption
event. Host cache IDs are updated in the same dispatch order. The next layer
starts only after the current layer's staged computation finishes.

Direct prefill uses separate per-slot fences with the same two dependencies,
including Q8_0 misses and resident-hit readers. Its copy resources are initialized
independently of the overlap flag and do not reuse decode/MTP event arrays.
Prefill has its own Q8_0 scratch; staging banks each own their own scratch.

Reset, decode-mode transitions, offload, and destruction drain prefill copies
and compute before touching owned storage. Execution errors propagate; fallback
is permitted only before staged dispatch begins. Cache-map updates use kernel
arguments instead of asynchronous copies from temporary host integers. The
prefill-to-decode map upload now completes before its host allocation is freed.

## Validation commands

Run from the repository root; use repository-local temporary storage:

```sh
export TMPDIR="$PWD/rdna4/llm/tmp"
make -C rdna4/llm -j8 test_hip_llm tmp/test_hip_qwen4_moe_stage
make -C rdna4/llm moe-stage-test
fuser /dev/kfd
rocm-smi --showmeminfo vram
make -C rdna4/llm moe-stage-gpu-test
make -C rdna4/llm tmp/test_hip_qwen4_moe_stage_large
timeout --foreground 180s ./rdna4/llm/tmp/test_hip_qwen4_moe_stage_large
```

The CPU test queues copies without eagerly reading their host pointers and
executes consumers late. It covers bank reuse, repacking, resets, partial
allocation cleanup, and event failures. Removing either host-lifetime or
slot-reuse synchronization causes its delayed-consumer assertions to fail.
The lifecycle test also defers map DMA and rejects freeing its source early.

The model-free GPU test calls the production staged dispatcher with finite
synthetic Q4_K/Q5_K gate/up and Q5_1/Q8_0/Q6_K down weights. It compares bitwise
against a serialized expert oracle across both host registration modes,
resident-only/cold-only/mixed residency, partial waves, repeated bank reuse,
overlap, promotion, request drains, and an injected mid-upload failure.
The default DIM256 and larger DIM2048 variants both pass on RX 9070 XT.
The latter stresses multi-MiB transfers. This tests staging correctness, not
whole-model scalar F16 parity.

For the real model, run this command for every pair `reg=0/1`, `copy=0/1`;
repeat the successful overlapping configuration in a fresh process:

```sh
LLM_MOE_REGISTER_HOST=1 LLM_MOE_COPY_PIPELINE=1 LLM_BENCH_WARMUP=0 \
QWEN38_TARGET_PROFILE=batch4k-stage QWEN38_TARGET_PREFILL=4096 \
QWEN38_TARGET_DECODE=64 QWEN38_TARGET_CONTEXT=8192 QWEN38_TARGET_REPEATS=8 \
QWEN38_TARGET_REQUIRE_EXCLUSIVE_GPU=1 \
QWEN38_TARGET_PROMPT_FILE=rdna4/llm/tmp/qwen38_target_prompt.txt \
QWEN38_TARGET_LOG=rdna4/llm/tmp/qwen38_staging_r1_p1.log \
./rdna4/llm/bench_qwen38_target.sh
```

Do not enable launch blocking or layer tracing for the repeatability gate.
`LLM_BENCH_WARMUP=0` now actually disables warmup; a nonzero value explicitly
enables it and emits a warmup message. Thus the first measured request also
exercises cold initialization.

## Reference parity

`make -C rdna4/llm moe-stage-quality` runs matched scalar/staged checks on a
4K header-source prompt and four fixed 128-token controls (coding, arithmetic,
English prose, Japanese), with two repeats each. It generates fresh references
and fails on either repeatability or reference disagreement. The exact prompt
texts and settings are in `test_qwen38_staging_quality.sh`; logs stay in
`rdna4/llm/tmp/staging_quality/`.

Capture a fresh scalar F16 reference with the same model, prompt, prefill length,
context, and decode length. Pass both reference values to the candidate run:

```sh
QWEN38_TARGET_EXPECTED_FIRST_TOKEN=30 \
QWEN38_TARGET_EXPECTED_HASH=6d67721190bdaa83 \
QWEN38_TARGET_PROFILE=batch4k-stage \
QWEN38_TARGET_PROMPT_FILE=rdna4/llm/tmp/qwen38_target_prompt.txt \
./rdna4/llm/bench_qwen38_target.sh
```

Those example reference values are historical, not a substitute for a matched
fresh scalar run. Both variables are required together. A repeatable candidate
that differs from either value fails the gate. Missing footers and incomplete
runs fail as before; `qwen38_target_result.sh` contains the shared footer parser.

No serving default, KV format, CPU expert policy, or MTP setting is promoted by
this change. Multi-chunk state carry and long-context quality remain separate
follow-up work.
