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

## Explicitly unresolved

- Long exact-MTP I8 diverges from F16 after roughly 16 generated tokens,
  although output remains coherent. I8 remains explicit-only.
- Grouped Qwen4 prefill still has numerical/state parity failures and remains
  diagnostic-only.
- A 32K+ input / 8K+ output streamed coding run has not completed with a
  trustworthy footer and coherence gate.
- The current ROCm image has `libhipblaslt.so` but no hipBLASLt headers, so the
  200+ tok/s accelerated prefill path cannot be rebuilt here. `make -C rdna4/llm
  hipblaslt-status` reports this directly.
- Persistent `/dev/kfd` access requires host/container device passthrough;
  elevated command namespaces are temporary and are not persistent access.

Capacity-only results must not be reported as 256K-prompt throughput.
