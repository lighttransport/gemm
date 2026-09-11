# Qwen4 / Qwen3.8-Flash-Next HIP MTP

The complete NextN sidecar now executes on HIP, with independent KV and HC
state, routed/shared experts, greedy drafting, and verified CLI/HTTP output.
MTP is **opt-in and experimental**. Correctness tests pass on RX 9070 XT;
the current verifiers are not yet a throughput improvement over exact decode.

Current-build caveat: the grouped verifier requires the explicit
`LLM_QWEN4_BATCH=1` gate. Its grouped logits can still diverge
(`pred0=1144` versus scalar `2688`) on rejected windows, so the path remains
diagnostic and does not publish those rows. Forced transaction checks now
explicitly use the scalar oracle whenever grouped mode is requested; the full
reject/EOS suite passes while grouped state/commit parity is repaired.

I8 KV validation is a separate gate from scalar-MTP parity. The target trunk
may use `LLM_QWEN4_KV_QUANT=i8`; the NextN sidecar remains F16 and must never
index target scale arrays. The I8 stores reject unsupported `head_dim > 256`
geometry and guard inactive scale groups. After GPU access is restored, run the
target-only smoke first, then the combined exact-MTP test:

```sh
env LLM_QWEN4_KV_QUANT=i8 rdna4/llm/test_hip_llm "$TARGET" \
  --gpu-only-bench -s 4096 --moe-cache-mb 6000 --decode 4
env LLM_QWEN4_KV_QUANT=i8 rdna4/llm/test_hip_llm "$TARGET" \
  --gpu-only-bench -s 4096 --moe-cache-mb 6000 \
  --qwen4-mtp "$NEXTN" --qwen4-mtp-draft 4 \
  --qwen4-mtp-cache-mb 128 --qwen4-mtp-verify scalar --decode 4
```

The second command is not validated until its output hash and rollback/EOS
checks match the F16 scalar-MTP control.

The elevated RX 9070 XT rerun has now passed that gate at 4K: I8 and F16
produced the identical greedy hash `454146399ff97e88`. I8 measured 6.86
decode tok/s and 5.03 end-to-end tok/s for the 11-token control prompt plus
four generated tokens; F16 measured 6.89 decode and 5.05 end-to-end tok/s.
This validates quality and stability, not a throughput improvement.

A longer 28-token coding prompt with 64 generated tokens remained coherent and
returned `PASS` with I8 (8.34 decode / 7.38 end-to-end tok/s). Its hash differed
from the F16 control after the longer horizon, so I8 remains a capacity-saving
experimental mode rather than a bit-identical replacement for F16.

The explicit scaled-E4M3 FP8 KV path also passed exact-MTP at 4K with the
short control hash `454146399ff97e88`. On the 28-token/64-output coding probe,
FP8 returned `PASS` at 8.61 decode / 7.48 end-to-end tok/s and matched the F16
hash `4b8937cd7db0e7a7`; it remains opt-in pending long-context qualification.

The repeatable regression harness is `test_qwen4_i8_quality.sh`; set
`QWEN38_MODEL` and `QWEN38_MTP`, then use `QWEN4_I8_QUALITY_DECODE=4` for the
strict short-parity gate or `64` for the coherent long-horizon diagnostic.

For a matched target-only 512-token/64-decode control without the sidecar,
F16 and FP8 produced the same hash `e7e9b5ace7b8f98a` (26.06 and 24.25
decode tok/s), while I8 produced `6f231930c119b7e6` (15.86 tok/s). This
isolates the long-horizon drift to INT8 KV quantization/attention error rather
than routing or recurrent state; I8 therefore remains explicit-only.

## Run

From the repository root (set `TARGET` and `NEXTN` to the GGUF paths):

```sh
mkdir -p tmp
export TMPDIR="$PWD/tmp"
make -C rdna4/llm -j2 test_hip_llm
rdna4/llm/test_hip_llm "$TARGET" --gpu-only-bench -s 4096 \
  --moe-cache-mb 2048 --qwen4-mtp "$NEXTN" --qwen4-mtp-draft 4 \
  --qwen4-mtp-cache-mb 128 --qwen4-mtp-verify scalar \
  -t 'Write a C function returning the larger of two integers.' --decode 32
```

HTTP serving uses the same explicit arguments:

```sh
python3 rdna4/llm/codex_server.py "$TARGET" \
  --runner ./rdna4/llm/test_hip_llm --host 127.0.0.1 --port 8080 \
  --context 4096 --moe-cache-mb 2048 \
  --qwen4-mtp "$NEXTN" --qwen4-mtp-draft 4 \
  --qwen4-mtp-cache-mb 128 --qwen4-mtp-verify scalar
```

Only `temperature <= 0` uses speculative acceptance. Sampled requests use the
ordinary target sampler; NextN is still teacher-forced to keep its state valid.
Only verified tokens reach the stream. Cancellation discards cached state.
`--qwen4-exact` without a sidecar selects the comparison baseline (also exposed
by the Python server). MTP disables approximate routing, expert dropping,
reduced-precision CPU miss execution, and the older batched prefill path.

## Execution and memory

- Sidecar block 48 is an HC + routed-MoE layer, not a dense NextN block.
  Trunk embeddings/output head are shared; sidecar expert bytes are owned.
- Q/K norms are shared head-dimension vectors. Fusion consumes all four HC
  streams; recurrent draft hidden state is not replaced with the mixed vector.
- The scalar verifier never executes a rejected target token. The window
  verifier executes layer-major rows and restores the accepted recurrent/PLE
  checkpoint on device. Both rebase NextN from committed target hidden states.
- Scalar verification now keeps target logits on-device and uses a GPU argmax,
  transferring only the selected token. This preserves the exact hash and all
  transaction checks, but current end-to-end timing is noisy (3.2–4.3 tok/s in
  repeated 32-token runs), so it is retained for bandwidth reduction rather
  than advertised as a speedup.
- Exact draft generation also uses stream ordering: the sidecar forward no
  longer waits before the immediately-following GPU argmax. Public logits and
  target verification retain their synchronization points, and the transaction
  checker covers the scoped fast path.
- Exact draft batches now advance through the sidecar HC output while saving
  and restoring the target HC state around drafting. Previously only trusted
  mode copied this recurrent state, so exact drafts repeatedly started from
  the same HC prefix. The corrected chain raised acceptance from about 52% to
  about 68% on the validation prompt; the 64-token exact sequence hash still
  matches ordinary target decode (`c8000d2b39140ae9`).
- Qwen4 UD routed experts use Q4_K gate/up with Q6_K down. The exact decode
  path now fuses the cached Q6_K down projection and weighted accumulation into
  one warp-per-row launch (the old path issued one scalar chain per expert).
  On RX 9070 XT this raised the same 32-token exact-MTP run from 5.22 to
  7.74 tok/s with an unchanged sequence hash. Set LLM_QWEN4_FUSED_Q6K=0
  to restore the pre-fusion path for diagnostics.
- Exact MTP normally keeps all routed misses on the GPU for reproducibility.
  `LLM_QWEN4_EXACT_CPU_MISSES=1` is an explicit hybrid-residency experiment:
  with the CPU GGML library and `LLM_MOE_CPU_DECODE_MISSES=1`, missing routed
  experts may execute on CPU while preserving the exact router/top-k choice.
  It is not enabled by default and must pass the same token-hash checks before
  being used in a production profile.
- The grouped Qwen4 prefill path now includes a Q6_K down kernel and has passed
  exact parity at both 7-token and padded 128-token prompts (same sequence
  hash `b08cf8f036d9b0fa` at the latter). It remains opt-in via
  `--qwen4-batched-prefill` (or `LLM_QWEN4_BATCH=1`); MTP configuration now
  preserves that explicit request instead of clearing it. This improves the
  128-token prefill measurement from 18.07 to 18.41 tok/s, but does not change
  the scalar decode bottleneck (about 7.7--8.0 tok/s exact on this card).
  The explicit CPU-miss experiment also passed a short transaction but slowed
  it to 2.39 tok/s; host registration stalled initialization in this
  environment.
- An experimental grouped verifier is available only in window mode with
  `--qwen4-mtp-verify window --qwen4-batched-prefill` plus
  `LLM_QWEN4_GROUPED_VERIFY=1`. It snapshots HC/SSM state, but it is **not
  exact-parity safe across prompts**: the current sky-blue validation produced
  hash `2b509c770f987ded` versus scalar `a959d312c3c07e17` and was much slower.
  Keep it disabled for exact production runs.
  A newer constrained control (`LLM_QWEN4_GROUPED_BATCH_LAYERS=1`,
  `LLM_BMAX=33`) now actually enters the grouped verifier and matches the
  scalar hash on the coding prompt (`ec2867ab19d1240b`). Partial-rejection
  replay was then reduced to the first grouped mismatch plus one pending row.
  Grouped windows now default to a minimum of three rows because one- and
  two-row attempts pay more snapshot/plan overhead than scalar verification;
  `LLM_QWEN4_GROUPED_MIN_ROWS=1` restores the old diagnostic behavior.
  On the RX 9070 XT 8-token/2-token control, this removed the rejected
  two-row grouped attempt: the first verification fell from about `744 ms`
  to `461 ms`, with the same `Result: PASS` and no grouped commit.
  An opt-in per-row checkpoint prototype is available with
  `LLM_QWEN4_GROUPED_PARTIAL_COMMIT=1` together with
  `LLM_QWEN4_GROUPED_COMMIT=1`. It reduced the accepted `3/5` verification
  from about `1.69 s` to `1.02 s` in the width-8 control, but produced a
  different sequence hash (`5dfe9bee3db263b5` versus the rollback-safe
  `5e003cee3db52848`). It is therefore an experimental diagnostic only and
  remains disabled by default pending target-state parity work.
  For isolation, forcing scalar HC and scalar embedding operations and disabling
  grouped layer bodies still produced `b08cf8f036d9b0fa`, so the divergence is
  in grouped-window state/commit semantics above the individual layer kernels,
  not just BF16 HC projection rounding.
  A fresh matched 16-token MTP control with `LLM_QWEN4_GROUPED_COMMIT=1`
  preserved the scalar hash `34202a88d2a8906a`, but measured only `2.79`
  decode tok/s and `3.05` end-to-end tok/s versus `5.11` and `4.84` for the
  scalar verifier. Grouped commit is therefore parity-safe for this control
  but still a performance regression; it remains opt-in.
  Enabling the parity-gated batched target flags for the same grouped run
  reduced decode to `1.97 tok/s`, changed the sequence hash to
  `17a124730cc56bda`, and increased H2D traffic to `46.83 GiB`; this confirms
  that the current batched target path is not a safe grouped-MTP optimization.

The sidecar checkpoint ring was also corrected so slot 0 remains the
pre-anchor state and draft step `i` is stored at slot `i+1`. The exact scalar
MTP control retained hash `5e003cee3db263b5`, and the forced transaction suite
passed all reject positions (`0/4` through `4/4`) plus EOS/output-limit checks.

The QSA score reduction now selects a 64/128/256-thread block from the
indexer dimension instead of always launching 256 threads; the reduction
bounds are dynamic and retain the same zero-padding semantics. With the real
sidecar and a 4K runner context, the verifier passes
(`n=2055`, `selected=2051`, score relative error `7.17e-7`, selected-attention
relative error `2.16e-6`). The old 256-token diagnostic was invalid because
its synthetic top-k window exceeded the configured context.
The same verifier passes with `-s 262144` and a 128-MiB MTP cache: the runner
allocates the 6-GiB F16/3-GiB FP8 long-context KV profile and retains the same
QSA parity metrics, confirming stable 256K index-cache allocation on the
16-GiB card.

The exact QSA path now has an opt-in device-side selector,
`LLM_QWEN4_QSA_DEVICE_SELECT=1`. It performs the same deterministic
score-tie ordering and causal-token sorting on the HIP stream, avoiding the
per-token score D2H/H2D synchronization. The host selector remains the
default. Device selection uses a two-stage local-sort/global-merge kernel so
large windows do not require a score round-trip or a single-thread O(n²) scan.
On the RX 9070 XT, the 4K verifier passes with device selection
(`scores_rel=7.17e-7`, selected-attention `output_rel=2.16e-6`), and an
an 8-prefill/8-decode smoke run retains sequence hash `5e003cee3db52848`
and the same `8110 MiB` peak as the host-selector control. The full-window
device-selector diagnostic also passes at `n=262144` with
`scores_rel=1.11e-4` and selected-attention `output_rel=2.17e-6`; the
two-stage selector now has exact 256K ID parity. An additional opt-in
`LLM_QWEN4_QSA_WARP_ATTN=1` kernel assigns one warp per head and passes the
256K F16 gate (`output_rel=2.22e-6`). A dedicated 4K scaled-I8 verifier now
also passes the warp kernel
(`output_rel=2.09e-6`), alongside the F16 check (`2.11e-6`). It remains
opt-in pending long-context throughput comparison.
  Rollback now snapshots cache identities and preserves untouched slots,
  invalidating only overwritten payloads. With an 8.2-GiB cache this raised the
  coding control to `9.89 tok/s` (from `8.54 tok/s` before selective rollback)
  with the same hash; an independent clamp control measured `8.75 tok/s` and
  matched the exact non-MTP hash. It is still below scalar MTP because the
  larger verifier scratch reduces residency.
  The earlier `LLM_BMAX=1` measurements were silently scalar and should not be
  used as grouped-performance evidence. Grouped mode now automatically raises
  an inherited BMAX below 33 to the required 33-row scratch size, and emits a
  diagnostic cap message. The sky-blue mismatch and broader
  corpus parity gate still prevent promotion.
- `LLM_QWEN4_EXACT_GPU_TOPK=1` enables the GPU router top-k kernel while keeping
  exact expert weights and routing enabled. It is an opt-in parity experiment:
  default exact mode retains the host top-k path until its tie/weight behavior
  is independently validated.
- `LLM_QWEN4_MTP_GPU_TOPK=1` additionally permits that GPU router during exact
  MTP verification. It preserves the default host-synchronized route and is
  intended for A/B measurements; on the RX 9070 XT short exact run it matched
  the reference hash but did not materially change end-to-end throughput.
- Exact Qwen4 uses LRU expert-cache replacement by default, with same-token slot
  protection.  LFU remains an explicit `LLM_MOE_LFU_CACHE=1` experiment; the
  256K exact control showed lower decode throughput with LFU, so it is not the
  production policy.
- The fixed 9.0 GiB cache budget (9216 MiB requested; 8.56 GiB trunk after
  reserve) is now rebalanced toward under-hit layers
  (notably layers 1, 12, 18, and 47) and away from layers already above 85%
  hit rate. A 32-token run retained the same hash and measured 8.51 tok/s
  decode / 6.93 tok/s end-to-end on the RX 9070 XT. Before the exact
  `LLM_BMAX=1` profile, 9728 MiB was not safe because the 512-row scratch set
  consumed the remaining VRAM.
- Scalar exact verification defaults to draft width 1. Since scalar target
  verification still evaluates every target token independently, width 2 adds
  sidecar work without batching target layers. On the same 32-token prompt,
  width 1 measured 8.72 decode / 7.07 end-to-end tok/s with the same hash;
  pass `--qwen4-mtp-draft 2` (or higher) only for window experiments.
- The RX 9070 XT exact launcher profile uses registered-host staged misses
  (`LLM_QWEN4_MAPPED_MISSES=1`, direct misses default off), sets `LLM_BMAX=1`
  because exact decode does not use batched-prefill scratch, and uses a 9728
  MiB device cache. 9856 MiB fails during sidecar load on the 16 GiB card.
  Direct BAR reads remain diagnostic-only for Q6_K/Q8_0 because they are not
  parity-safe; staged device-cache misses measured 15.1 tok/s on the short
  exact check and retained the scalar hash.
- The stdio/Codex server applies the same staged exact settings automatically
  for `--qwen4-exact --qwen4-mtp` when no explicit MoE cache budget is given.
- Delayed resident-cache promotion is intentionally disabled for direct BAR
  misses: that experimental combination can replace a live cache slot while
  the mapped kernel is reading it. The normal exact profile remains race-free
  and uses staged misses.
- `LLM_QWEN4_MTP_APPROX=1` selects an explicitly approximate MTP profile. It
  leaves the default exact verifier untouched and enables the existing resident
  expert/approximate decode machinery; its output must be treated as a separate
  quality/performance profile, not exact-token parity.
- Approximate MTP plus the existing CPU-miss residency profile measured
  14.9 tok/s on the 32-token run, versus ~4.5 tok/s for approximate MTP with
  GPU-only misses. Sidecar/target routing alignment still produced zero
  accepted drafts on this prompt, so this is not yet a speculative speedup.
  The CLI and HTTP server detect a zero-acceptance first batch in this opt-in
  profile and fall back to ordinary target decoding for the remainder of the
  request, avoiding repeated speculative overhead when alignment is poor.
- `LLM_QWEN4_MTP_TRUST_DRAFT=1` is a separate throughput-ceiling mode for
  approximate experiments. It commits the sidecar draft prefix without target
  verification and advances only the independent NextN state; it measured
  300+ decode tok/s on a four-token RX 9070 XT run, but provides no
  target-token parity and can become incoherent over longer generations. It
  must not be used for exact or quality-sensitive serving.
  A sustained 32-token run with draft width 4 measured **322 tok/s**
  (`drafted=4 accepted=4` on every full batch); width 8 was slower at 279
  tok/s because the sidecar work scales linearly with the wider batch.
  Because the target trunk is not advanced in this mode, the HTTP runner
  invalidates its reusable target prefix after each trusted request.
  The standalone runner exposes the same profile as
  `--qwen4-mtp-trust-draft`; that flag also enables the resident-hit cache
  defaults, so it does not require copying the launcher environment. The
  flash-next launcher drops its default `--coding` flag only when this option
  is present; ordinary coding runs intentionally disable MTP. Trusted draft
  steps reuse the argmax synchronization and measured 323.9 tok/s on the
  RX 9070 XT after removing a redundant stream wait. This is decode-only;
  the 32-token warm-request measurement is 321.6 decode tok/s but only
  29.1 end-to-end tok/s including the 6-token prefill (model load excluded).
  The flash launcher also defaults `LLM_MW_THREADS` to 128 on this GPU
  (override it for other devices); ordinary decode measured 51.27 tok/s and
  trusted MTP 322.69 tok/s with that setting.
  Runtime logs label this path `backend=hip-approx` to distinguish it from
  target-verified MTP.
- `LLM_QWEN4_MOE_GRAPHS=1` experimentally captures the resident-hit Qwen4
  MoE stage together with the per-layer preamble. It is parity-safe on the
  tested prompt, but currently measures about the same 51 tok/s as the normal
  graph profile and is therefore not enabled by default.
- Window mode is explicitly selected by `--qwen4-mtp-verify window`. It still
  uses scalar per-row arithmetic, not batched GEMMs. It avoids replay but adds
  checkpoint traffic and computation for rejected suffixes.
- Scalar verification does not consume the sidecar HC checkpoint chain, so the
  draft loop skips those large device-to-device checkpoint copies in scalar
  mode. Window mode retains the full chain for transactional rollback and
  commit. This is a transfer-only optimization; it does not change token
  selection or the exact output hash.
- The exact draft chain now keeps the sidecar recurrent HC in its dedicated
  buffer across draft positions instead of copying the full HC tensor into and
  out of target storage for every token. Exact mode leaves target HC untouched;
  trusted mode publishes the final sidecar HC once per batch so its cross-
  request chain remains unchanged. The now-unneeded saved-target HC allocation
  is removed as well.
- Draft steps also avoid restoring the prior scalar position between chained
  NextN calls; the next draft/target operation writes its required position
  before consuming it. Non-MTP/public calls retain the restore for safety.
- Draft argmax results can now remain device-resident and feed the next
  embedding lookup directly. The exact/trusted draft loop copies the compact
  token-id chain back once after the batch instead of synchronizing the stream
  after every speculative token; the public forward path remains unchanged.
  Set `LLM_QWEN4_MTP_HOST_CHAIN=1` to force the legacy host-synchronized chain
  for A/B comparison or driver recovery.
- `LLM_QWEN4_MTP_COPY_PIPELINE=1` enables an experimental asynchronous expert
  miss-copy stream during exact MTP. It passes rollback/hash checks, but the
  short RX 9070 XT run was slower than the default because MTP transactions
  leave too little overlap; it remains opt-in.
- `LLM_QWEN4_MTP_CPU_MISSES=1` allows the existing exact CPU-miss fallback to
  run during MTP when combined with `LLM_MOE_CPU_DECODE_MISSES=1` and
  `LLM_QWEN4_EXACT_CPU_MISSES=1`. It preserved the sequence hash but was slower
  on the 9070 XT, so it remains an experiment rather than a default.
- Combining `LLM_QWEN4_DEVICE_CACHE=1`, `LLM_QWEN4_EXACT_GPU_TOPK=1`, and
  `LLM_QWEN4_MTP_GPU_TOPK=1` exercises the device-resident exact cache map. It
  also preserved the hash, but added map-kernel overhead and measured slower
  than the host-slot path; it is not a production setting.
- The non-MTP approximate resident-hit path now guards stale/out-of-range cache
  slots in the Q4_K map kernels, preventing the prior illegal-memory fault.
  `LLM_QWEN4_DEVICE_HITS_1W=1` exposes the one-warp map geometry; it is safe
  but did not improve resident-only throughput on the 9070 XT. The best
  measured approximate direct-map run was 40.27 decode tok/s (with BMAX=1 and
  an 8.7 GiB cache) and produced a
  non-exact hash, so it must not be used for exact output claims.
- With `LLM_QWEN4_BATCH=0`, `LLM_BMAX=1`, an 8.2 GiB cache, host registration,
  direct mapped misses, and `LLM_QWEN4_DEVICE_REFRESH_INTERVAL=128`, the
  approximate non-MTP path measured 54.08 decode tok/s over 128 tokens on the
  RX 9070 XT. End-to-end for the same short prompt was 44.74 tok/s because the
  628 ms prefill is included; these are separate metrics. A text-enabled
  32-token run measured 52.57 tok/s and produced readable sky/scattering text;
  this remains approximate rather than exact-parity decoding.
- Window verification uses the same device token scratch for row predictions,
  replacing one tiny device→host transfer per row with one transfer per
  window; ordering and argmax results are unchanged.
- The experimental grouped MoE allocator now protects every resident expert
  already queued in the same verification window from same-token slot reuse.
  This closes a cache-map race that could corrupt a grouped direct-miss window;
  scalar decode behavior and hashes are unchanged.
- Grouped exact verification now admits Q4_K gate/up with Q6_K down experts;
  the existing grouped Q6_K kernel was previously unreachable for this model.
- The experimental device-cache route is exact with staged misses and measured
  at 15.1 tok/s decode on the short RX 9070 XT check. Direct BAR misses remain
  disabled for Q6_K/Q8_0 because their host-mapped kernels are not yet parity-safe.
- Grouped exact scratch is capped at 33 rows when `LLM_QWEN4_GROUPED_VERIFY=1`.
  This avoids reserving the normal 512-row prefill workspace (including the
  large vocab-logit matrix) and lets the staged exact profile fit in 16 GiB.
- Grouped verification defaults to the scalar Q8_0 lm-head for small windows;
  the tiled BF16 lm-head is retained only with
  `LLM_QWEN4_GROUPED_SCALAR_LMHEAD=0` because its many tiny GEMMs are much
  slower at M≤33.
- The scalar lm-head path now writes all window logits into the grouped buffer
  and uses one deterministic `qwen4_argmax_batch` launch for the window,
  replacing one argmax launch/device copy per row. The forced-reject grouped
  transaction gate still passes with unchanged hash and rollback checks; this
  reduces launch overhead without changing token selection. A matched
  16-token RX 9070 XT control measured `3.30` decode / `3.46` end-to-end
  tok/s with hash `34202a88d2a8906a`, versus the earlier `2.79` / `3.05`
  grouped-commit control.
  Repeating the same window with `LLM_QWEN4_GROUPED_SCALAR_LMHEAD=0` measured
  `3.28` decode / `3.45` end-to-end tok/s with the same hash, so the tiled
  BF16 lm-head remains a diagnostic path rather than a promotion.
  The same grouped-commit transaction suite with the tuned 9-GiB expert-cache
  budget (rather than the earlier 2-GiB stress budget) completed all reject and
  EOS checks with hash `34202a88d2a8906a`. It measured `6.08` decode / `5.57`
  end-to-end tok/s, with a `55.0%` prefill-cache hit rate, `80.3%` decode-cache
  hit rate, and `14.76` GiB peak VRAM used. The larger resident cache therefore
  roughly doubles grouped throughput, but remains far below the scalar-MTP
  baseline and is still experimental.
- Grouped MoE task metadata is now copied asynchronously on the compute stream
  instead of using a host-visible `hipMemcpy` plus an immediate stream fence.
  This removes an avoidable CPU/GPU synchronization from both resident and
  deferred grouped waves. A repeat of the 9-GiB control remained hash-identical
  and measured `6.07` decode / `5.59` end-to-end tok/s (`PASS`); the change is
  therefore a safe scheduling cleanup, not yet a statistically significant
  throughput gain.
- A full-attention-layer A/B (`LLM_QWEN4_BATCH_ATTN_MAX_LAYER=47`) preserved
  the 32-token coding hash `48f9514bc5863ce4` and passed, but was slightly
  slower than the parity-safe three-layer prefix (`5.94` vs `6.09` decode
  tok/s). Enabling the experimental batched SSM projections likewise kept the
  hash and was neutral. The scalar SSM recurrence and MoE/cache traffic, not
  the remaining attention projections, are therefore the dominant grouped
  verifier costs; the conservative attention-layer default stays in place.
- `LLM_QWEN4_GROUPED_GPU_TOPK=1` is an opt-in grouped-window experiment that
  keeps Qwen4 mixed-quant routing on GPU (top-k, offsets, and gather) instead
  of synchronizing router logits to the CPU. It may alter borderline routes
  through GPU softmax rounding, so exact hash parity must be checked before
  enabling it as a production default.
- When grouped verification is explicitly enabled, the loader now pre-warms
  the M=2 HC projection plans. This removes the first-window hipBLASLt setup
  spike (about 1.1 s on the RX 9070 XT); the exact token hash is unchanged.
- Scalar MTP is correctness-focused rather than a throughput ceiling: target
  verification still executes one full target forward per accepted/rejected
  position, so the sidecar cannot remove the target trunk work. Reaching the
  60 tok/s goal requires grouped exact verification with batched target
  projections/expert execution; the current window path is not that backend.
- An opt-in `LLM_Q6_DOWN_KPAR=1` kernel parallelizes the ten selected Q6_K
  down experts per output row. It is hash-identical but currently throughput
  neutral on the RX 9070 XT, so it remains diagnostic rather than default.
- `LLM_QWEN4_MTP_ADAPTIVE=1` enables a quality-preserving fallback for exact
  MTP: after a batch below 50% acceptance, the committed target token is
  replayed and the request continues with ordinary target decode. A 32-token
  low-acceptance run completed with `Result: PASS` at 4.91 tok/s.
  The standalone runner and HTTP server also expose this as
  `--qwen4-mtp-adaptive`.
- Draft width is 1–32. Sidecar cache defaults to 128 MiB; pinned host payloads
  and reusable GPU Q8 packing scratch reduce miss-path transfer/packing cost.
  The runner/server default is width 1 for exact scalar mode; width 4 remains
  available explicitly for window experiments with unusually high acceptance.
- MTP/QSA and window-state estimates are subtracted from the requested expert
  budget before trunk cache sizing. Too-small budgets fail explicitly. Context
  length is never silently reduced. This is a reserve, not a peak-VRAM guarantee.
- QSA stores raw F16 index keys, pools complete compression blocks before
  normalization/RoPE, selects blocks with stable ties, and retains the partial
  causal tail. Beyond dense-equivalent width 2051, selected attention is used.
- The NextN block itself uses dense causal attention with an explicit first
  initialized position. Its initial prompt token has no previous target hidden
  state, so teacher forcing begins at prompt position 1.

## Regressions

```sh
gcc -O2 -Wall -Wextra -Wpedantic common/test_qwen4_qsa.c -lm -o tmp/test_qwen4_qsa
tmp/test_qwen4_qsa
rdna4/llm/test_hip_llm "$TARGET" --gpu-only-bench -s 512 \
  --moe-cache-mb 2048 --verify-qwen4-nextn "$NEXTN"
rdna4/llm/test_hip_llm "$TARGET" --gpu-only-bench -s 4096 \
  --moe-cache-mb 2048 --verify-qwen4-qsa
rdna4/llm/test_hip_llm "$TARGET" --gpu-only-bench -s 512 \
  --moe-cache-mb 2048 --qwen4-mtp "$NEXTN" --qwen4-mtp-check \
  --qwen4-mtp-draft 4 --qwen4-mtp-verify window --decode 32
python3 rdna4/llm/test_qwen4_mtp_http.py --model "$TARGET" --sidecar "$NEXTN"
```

The CPU oracle is `qwen4_nextn_ref.h`, independent of HIP arithmetic. It uses
F16 round-to-nearest-even cache stores, matching HIP (the older truncating
helper is unsuitable). Four recurrent steps produced identical argmax tokens
and hidden/logit relative errors below 6.3e-6. The transaction checker forces
rejection at every slot plus full acceptance and checks subsequent continuation;
it also covers immediate/accepted EOS and a one-token output limit.

HTTP regression compares exact baseline/scalar/window text, repeated prompts,
zero-output prompt caching, one-token output limits, sampled requests, and
cooperative cancellation followed by deterministic recovery. It uses an
ephemeral loopback port and reaps each runner.

The full GPU-backed HTTP suite was rerun after selective grouped-cache rollback;
baseline, scalar, and window modes all passed repeat/cache/output-limit/sampling/
SSE/cancellation checks and retained greedy parity.

Final GPU logs are in `tmp/qwen4_oracle_final.log`, `tmp/qwen4_qsa_final.log`,
`tmp/qwen4_mtp_final_scalar.log`, and `tmp/qwen4_mtp_final_window.log`.
CPU/HIP oracle maximum logit relative error was 6.0e-6; selected-attention
output error was 2.16e-6 after the wave-level dot-reduction optimization.
The final HTTP suite additionally checks actual SSE content and requires an
explicit `cancelled` result, not merely successful post-cancellation recovery.
All three modes passed; see `tmp/qwen4_mtp_http_final.log`.

## Performance status and remaining optimization

Initial 32-token K=4 runs at a 2 GiB expert budget produced identical sequence
hash `6fa2706e4dd81299`: scalar 4.97 tok/s, window 3.27 tok/s. These are
correctness runs, not controlled performance comparisons: their memory reserve
settings differed. Following pinned-transfer/Q8-scratch changes, four-token
drafts measured roughly 15–18 ms on the HTTP coding prompt.

Final K=4 32-token checks measured scalar 5.05 tok/s and window 3.25 tok/s,
again with identical token hashes. Acceptance was 21/40 draft tokens (52.5%),
32 emitted tokens over 11 rounds. The 2048 MiB requested budget leaves
1661.5 MiB for scalar trunk experts versus 1100.4 MiB for window trunk experts;
these timings must not be presented as an equal-cache kernel comparison.

Exact MTP now enables the fused GPU router/top-K plus exact CPU cold-miss path
by default (both remain opt-out with `LLM_QWEN4_EXACT_GPU_TOPK=0` and
`LLM_QWEN4_EXACT_CPU_MISSES=0`). The full forced-rejection transaction guard
passes with this combination and the 32-token run retains hash
`fdab3631dfc6f672`; it measured 4.88 tok/s under the 2 GiB cache configuration,
so this removes router overhead but does not yet reach the 60 tok/s target.

A matched RX 9070 XT control with the tuned 9-GiB cache, registered host misses,
mapped misses, and GPU top-k gives a clearer current baseline: ordinary exact
decode measured `17.89 tok/s` (32 generated tokens, 11-token prompt), while
exact scalar MTP with draft width 1 measured `16.93 tok/s` with the identical
sequence hash. The adaptive fallback measured `16.58 tok/s` after a first
zero-acceptance batch. MTP is therefore still correctness infrastructure, not
a speedup, until target verification is grouped across accepted rows.

Scalar/window commit now retains sidecar hidden-state checkpoints during draft
generation. For normal generated drafts, commit resumes NextN from the accepted
prefix and advances only the final emitted token; forced-draft transaction tests
still use the replay path because their tokens intentionally differ. This removes
redundant sidecar replay without changing the exact sequence hash. A controlled
RX 9070 XT run measured 8.44 tok/s decode and 6.81 tok/s end-to-end (8 prompt +
32 generated tokens, 9.0 GiB requested cache, hash `c7b4ec68b2f23ea`).

GPU-free protocol/template/tool tests (19 tests), sampler, prompt-policy, and
copy-lifecycle regressions pass. The prompt-policy test now matches the runner's
metadata rule: absent `add_bos_token` means no implicit BOS insertion.

Do not advertise a speculative speedup from these results. The next performance
milestone is grouped exact target projections/expert execution across window
rows, with scalar parity gates. Also outstanding: representative long-context
end-to-end comparison against upstream,
multi-prompt K/cache sweeps with repeated measurements, and peak VRAM
measurements across representative profiles. Runtime peak-VRAM
telemetry is now exposed by `hip_llm_get_vram_stats()` and printed by the
benchmark/server; only the multi-prompt/profile sweep remains. Large prefill requests now default to 512-token
streaming in the diagnostic runner; a 65K allocation test is not a 65K
inference-quality test.
Keep scalar mode as the default and window mode experimental until those pass.

The grouped verifier's LM-head buffers are now sized to its actual 33-row
maximum (`draft <= 32` plus the anchor), rather than the ordinary prefill
`BMAX`. At `BMAX=2048` this removes roughly 2 GiB of unnecessary F32 logits
allocation on the 16-GiB card, improving the chance of running the grouped
parity experiment without changing scalar output or acceptance semantics.

The first real grouped run exposed a numerical-parity hazard: grouped attempts
returned `rc=0` but produced a different sequence hash when their state was
committed. Grouped commit is therefore now guarded by the explicit
`LLM_QWEN4_GROUPED_COMMIT=1` knob. With the default unset, grouped rows are
still used diagnostically, then the transaction restores state and scalar
exact replay produces the same hash as the scalar-window control
(`e47725b8c4eb1cc5`, `Result: PASS`).

Exact MTP also has an opt-in adaptive width controller via
`LLM_QWEN4_MTP_ADAPTIVE_SHRINK=1` together with the existing adaptive flag. It
halves the draft width after a low-acceptance batch and grows it by one after a
fully accepted batch, without changing target verification. A 4→2→3→4 run
preserved the scalar hash `e47725b8c4eb1cc5` and returned `Result: PASS`.
A draft-width-8 A/B on the same 16-token control improved decode from `4.91`
to `5.01 tok/s` (~2%) while preserving the same hash; it remains opt-in until
longer coding prompts confirm the benefit.

The dedicated MTP copy stream is now selected automatically by the flash
launcher when `--qwen4-mtp` is present (override with
`LLM_QWEN4_MTP_COPY_PIPELINE=0`). On the 16-token exact control it improved
draft-4 decode from `5.08` to `5.36 tok/s`, reduced H2D from `15.19` to `13.30
GiB`, and preserved hash `e47725b8c4eb1cc5` with `Result: PASS`.
The copy pipeline and adaptive-width controller compose cleanly at draft 8:
the combined run measured `5.36 tok/s`, H2D `13.43 GiB`, and the same exact
hash.
The forced-rejection transaction suite also passes with the pipeline enabled:
reject counts `0/4` through `4/4`, continuation checks, and EOS/output-limit
all returned `PASS`.

A matched 32-token coding control with the tuned 9-GiB trunk cache confirms the
current scalar-MTP defaults: draft width 1 measured `16.71` decode tok/s,
while width 4 measured `16.19` tok/s. Both produced the identical greedy hash
`ec2867ab19d1240b` and `PASS`, so width 1 remains the scalar default. Enabling
the parity-tested MTP copy stream raised the same width-1 run to `17.44`
decode tok/s and `15.80` end-to-end tok/s, with the same hash and `PASS`.
The copy stream is therefore the only validated MTP speedup in this profile;
grouped exact verification remains the next optimization target.

The flash launcher and HTTP backend now automatically enable the parity-tested
QSA device selector plus warp-per-head attention for exact requests with
`max_seq >= 8192`. Approximate requests and shorter exact requests retain their
previous defaults; `LLM_QWEN4_QSA_DEVICE_SELECT` and
`LLM_QWEN4_QSA_WARP_ATTN` explicitly override the automatic choice.
