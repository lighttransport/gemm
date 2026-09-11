# Qwen3.8/Qwen4 RDNA4 status

This file records only matched measurements that are currently reproducible on
the RX 9070 XT. F16 remains the default quality-safe profile; I8 and FP8 are
explicit experiments.

## Verified

- FP8 KV uses real scaled E4M3 encode/decode kernels, not an I8 alias.
- Short exact-MTP parity: I8, FP8, and F16 all returned hash
  `454146399ff97e88` for the 28-token/4-output control.
- FP8 64-token coding control retained the F16 hash
  `4b8937cd7db0e7a7` and returned `PASS`.
- FP8 256K allocation smoke: 3.000 GiB KV, 4.91 prefill, 5.60 decode,
  5.23 end-to-end tok/s, `PASS`.
- FP8 generation under the 256K allocation: 8-token prompt + 64-token greedy
  output, 4.84 prefill, 8.05 decode, 7.50 end-to-end tok/s, hash
  `d6951038d33a3c17`, 15.934 GiB peak used, `PASS`.
- F16 at max 256K allocation requests 6.000 GiB KV and fails finalization on
  the 16-GiB card; no F16 256K result is claimed.
- Profile, lifecycle, protocol, FP8 codec, and shell/static checks pass.
- With temporary `/dev/kfd`/`/dev/dri` access, the 512-token scalar control
  completed end-to-end at 25.71 prefill / 27.24 decode / 25.73 tok/s and
  `PASS` (12.708 GiB peak VRAM).
- The scalar two-chunk 1,024-token control also completed with the same
  sequence hash (`a2d4f49620d5b663`): 23.61 prefill / 23.24 decode / 23.61
  end-to-end tok/s, `PASS` (12.708 GiB peak VRAM).
- The built-in 512-token batched-vs-per-token comparison reported exact
  logits (`rel_l2=0`, `max_abs=0`) and matching argmax token 30.
- The I8 GQA8 selector now correctly recognizes Qwen4's 24/2 (12:1) shape,
  and its reduction bug (only four keys reaching softmax) was fixed. A matched
  512+64 control still produced a different hash with GQA8 enabled
  (`0b3e44e3e7fb5b0c`) versus scalar I8 (`6f231930c119b7e6`), so it remains
  opt-in pending long-horizon quality parity.
- I8 now honors `LLM_QWEN4_DISABLE_QSA=1`, matching the F16 control and
  allowing KV-only drift isolation; the default remains unchanged.
- Matched 512-token/64-decode greedy controls without MTP confirm the format
  boundary: F16 and FP8 both hash `e7e9b5ace7b8f98a` (26.06/24.25 decode
  tok/s respectively), while I8 hashes `6f231930c119b7e6` (15.86 tok/s).
  This isolates the long-horizon drift to INT8 KV quantization/attention, not
  routing or general recurrent state.
- A 16-channel I8/FP8 scale-group experiment was rejected: the I8 control's
  first token changed from 30 to 271. The implementation is restored to the
  validated 32-channel groups; finer grouping needs a separate parity design.
- The explicit batched WMMA path is stable for streamed-shaped requests: 2,048
  tokens at 114.88 prefill / 15.05 decode / 111.99 end-to-end tok/s, and
  4,096 tokens at 112.21 prefill / 14.53 decode / 110.75 end-to-end tok/s;
  both returned `PASS` with 13.448/13.544 GiB peak VRAM respectively.
- Two identical resumed 4,096-token batched runs reproduced throughput
  (111.26/111.49 prefill and 109.39/109.34 end-to-end tok/s, 13.544 GiB
  peak), but produced different hashes (`d973fd2d3e5f6bfc` and
  `42af972770bae7cf`). This is direct evidence that grouped/batched state is
  still nondeterministic on gfx1201 and cannot be promoted to a quality-safe
  default.
- Disabling expert copy pipelining (`LLM_MOE_COPY_PIPELINE=0`) did not remove
  the variance: the 4K run measured 96.67 prefill / 95.17 end-to-end tok/s
  and hash `d9f9ce178c4b4eb1`. The nondeterminism is therefore broader than
  the asynchronous copy overlap; no production switch was changed.
- An opt-in ordered expert-scatter reduction was also tested on the same 2K
  control. Two runs still produced different hashes, so atomic expert
  scatter is not the sole source of the grouped-path variance; the
  experimental kernel was removed rather than adding another unsupported
  production switch.
- The grouped resident/staged task lists previously used asynchronous H2D
  copies from pageable scratch that was overwritten during the subsequent
  cold-expert walk. Those metadata copies now complete synchronously before
  scratch reuse; runtime parity still needs a model-mounted rerun.
- The post-fix 2,048-token grouped controls both passed, at 115.61/109.73
  prefill/end-to-end tok/s and 118.97/113.57 tok/s, but still diverged at the
  first decoded token (17512 vs 220) and sequence hash
  (`70e30279debbe15f` vs `b01463d3c4871377`). The copy-publication race was
  real-risk mitigation, not the complete source of grouped nondeterminism.
- An opt-in scalar-router diagnostic (`LLM_QWEN4_BATCH_ROUTER_SCALAR=1`) also
  failed to stabilize matched 512-token controls: hashes were
  `5c7ed763fe63b0da` and `c6761ef1770a4293` (first tokens 515 and 10586), at
  71.97/66.68 and 71.10/66.07 prefill/end-to-end tok/s. Router reduction
  ordering is therefore not the sole remaining source.
- A stricter 512-token isolation with batched SSM, attention, router,
  projections, grouped MoE, and copy overlap disabled still diverged: hashes
  `ee573f2372785825` and `fbade0a3d53fbd73` (first tokens 17 and 11), at
  62.49/60.34 and 62.49/60.37 prefill/end-to-end tok/s. The remaining bug is
  therefore in broader batched state/stream publication, not one isolated
  grouped kernel; scalar dispatch remains the reference path.
- Per-row fallback and scalar KV diagnostic loops also had an async-copy hazard:
  each queued position transfer referenced a loop-local stack `pos`. Those
  publications are now synchronous. Matched post-fix controls still differed
  (`2aa0ef09c71a181b` vs `fbade0a3d53fbd73`, first tokens 17 vs 11), so this
  was another real race but not the complete source.
- Fresh rebuilt scalar controls remain deterministic: both 512-token runs
  returned first token 30 and hash `a2d4f49620d5b663`, at 25.84/25.86 and
  25.87/25.83 prefill/end-to-end tok/s. The instability is specific to the
  batched dispatcher, not general GPU state or model loading.
- For completeness, forcing `OMP_NUM_THREADS=1` on the scalarized batched
  isolation also failed to stabilize output: hashes `b9ee92610dbbc1cd` and
  `6254e5064f4050f4` (first tokens 17 and 271). Host OpenMP scheduling is not
  the remaining explanation.
- Forcing `LLM_QWEN4_BATCH_HC_SCALAR=1` stabilized the first decoded token
  (`907` in both 512-token controls), but later decode still diverged: hashes
  `3ced02b6184309f7` and `1c91fa80c934253e` at 48.84/48.35 tok/s end-to-end.
  This identifies batched HC/PLE arithmetic as one prefill mismatch source,
  while leaving a separate post-prefill decode-state/KV handoff issue.
- Forcing both `LLM_QWEN4_BATCH_KV_SCALAR=1` and
  `LLM_QWEN4_BATCH_ATTN_SCALAR=1` did not stabilize the batched path either:
  first tokens were 30 and 220, with hashes `525be427d6b320b5` and
  `f5829d6be7fb6728`. KV-store/attention publication is not a standalone fix.
- The HC-scalar result is repeatable as a prefill diagnostic but not a serving
  solution: it agrees on the first token while later decode state diverges;
  scalar KV/attention publication instead changes the first token again. All
  such switches remain opt-in and the scalar dispatcher remains the only
  quality-safe default.
- `HIP_LAUNCH_BLOCKING=1` also failed to stabilize matched 512-token batched
  controls: hashes `7c0b751e708b64fd` and `7ec265c22d7a16f8` (first tokens 47
  and 12920). The variance is not eliminated by global launch serialization.
- Forcing scalar token embedding (`LLM_QWEN4_BATCH_EMBED_SCALAR=1`) likewise
  left the batched path nondeterministic: hashes `fc6397b74e8e9d83` and
  `83452d2dfbc8621b` (first tokens 435 and 5652). Divergence begins after or
  within the first batched layer, not in embedding publication.

## Explicitly unresolved

- Long exact-MTP I8 diverges from F16 after roughly 16 generated tokens,
  although output remains coherent. I8 remains explicit-only.
- Grouped Qwen4 prefill still has numerical/state parity failures and remains
  diagnostic-only.
- Scalar 2K/4K runs exceeded the interactive observation window before a
  footer was captured (the 1K two-chunk control is clean); they must not be
  called failures until rerun with a persistent host/container GPU session.
  The batched results above are therefore still an explicit A/B path, not a
  default-quality claim.
- A 32K+ input / 8K+ output streamed coding run has not completed with a
  trustworthy footer and coherence gate.
- The current ROCm image has `libhipblaslt.so` but no hipBLASLt headers, so the
  200+ tok/s accelerated prefill path cannot be rebuilt here. `make -C rdna4/llm
  hipblaslt-status` reports this directly.
- FP4 KV is not implemented or aliased: the current cache formats are F16,
  symmetric I8, and real E4M3 FP8. A useful FP4 implementation would need a
  specified packed FP4 encoding and scale/error policy first; silently
  reusing the I8 byte path would not be a valid assessment.
- Persistent `/dev/kfd` access requires host/container device passthrough;
  elevated command namespaces are temporary and are not persistent access.

## Remaining tasks

- [ ] Find and fix the remaining batched-dispatch nondeterminism on gfx1201.
      Repeated identical requests must produce the same first token and full
      sequence hash before the WMMA path can be promoted from diagnostic-only.
      Current evidence points to batched state/stream publication or the
      post-prefill KV/decode handoff; no single kernel isolation has fixed it.
- [ ] Add a repeatability gate to the streamed 512/2K/4K benchmark: run at
      least two identical requests, compare first token and sequence hash, and
      report prefill, decode, and end-to-end wall-clock tok/s together.
- [ ] Complete a quality-gated 32K+ prompt / 8K+ streamed coding workload
      using 512--2048-token prefill chunks and 64--128-token decode chunks.
      Record coherence, hash/repeatability, peak VRAM, and end-to-end tok/s.
- [ ] Re-run scalar 2K/4K controls in a persistent GPU session and capture a
      complete footer; do not infer their throughput from timed-out runs.
- [ ] Obtain a ROCm image with hipBLASLt development headers, rebuild the
      accelerated prefill path, and compare it against the current scalar and
      WMMA controls without changing the quality gate.
- [ ] Continue FP8 KV long-context quality testing and measure the practical
      256K prompt path. Keep F16/I8/FP8 comparisons separate; no FP4 result is
      valid until a packed encoding and scale/error policy are specified.
- [ ] Make GPU device passthrough persistent for benchmark sessions
      (`/dev/kfd`, `/dev/dri`, `video`, and `render`) so results are not tied to
      temporary elevated namespaces.

Capacity-only results must not be reported as 256K-prompt throughput.
