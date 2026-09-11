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
- Persistent `/dev/kfd` access requires host/container device passthrough;
  elevated command namespaces are temporary and are not persistent access.

Capacity-only results must not be reported as 256K-prompt throughput.
