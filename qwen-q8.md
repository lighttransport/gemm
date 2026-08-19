# Qwen3.8-27B Q8 on one A64FX node

## Measured baseline

Measurements used the second node (`a35-1110s`) of job 50639255 so the Codex
process did not consume the model node's HBM. Both model runs used 48 cores,
`max_seq=8`, one prompt token, four measured decode tokens, and no NextN draft.

| Path | Load | Decode | Notes |
| --- | ---: | ---: | --- |
| Q4_K_XL anonymous, NUMA distributed | 609.6 s | 0.164 tok/s | 24.398 s / 4 tokens; 16.7 GB resident weights |
| Q8_0 lazy mmap from `/local` | 0.53 s | 0.020 tok/s | 204.607 s / 4 tokens; repeated file faults/storage traffic |
| Q8_0 full anonymous | OOM | n/a | killed during load even with `max_seq=8` |

The old decode profiler sees only the separate logits projection because the
Qwen trunk executes inside one persistent pool dispatch. It also charges two
bytes per Q8 weight instead of the GGUF Q8_0 size of 34/32 bytes. Its reported
bandwidth and dispatch count are therefore not valid for the trunk. Dividing
the Q4 resident tensor bytes by the measured trunk time gives only about
2.7 GB/s, consistent with the observed 4 GB/s-class behavior.

The Q8 file contains 29.036 GB of tensors: 0.451 GB belongs to block 64/NextN,
and the token embedding is 1.351 GB. Ordinary decode can leave the embedding
file-backed (only one row is read per token) and omit NextN, reducing required
anonymous weight residency to about 27.23 GB.

The existing DS4F cold-pool probe on the same node measured:

| Kernel/layout | Weight bandwidth | Accuracy observation |
| --- | ---: | --- |
| packed BF16 | 730.4 GB/s | hardware/placement control |
| block-scaled Q8 SDOT | 431.4 GB/s | retains per-block scaling |
| row-weight/per-token-activation Q8 | 549.3 GB/s | up to 10.6% relative error on an activation-spike case |
| row-weight/per-64-activation Q8 | 406.3 GB/s | safer spike behavior |

The 800 GB/s HBM roofline is consequently not the Q8 kernel roofline: SDOT,
scale conversion, and scale application are material costs. A 20 tok/s result
requires roughly 540 GB/s over 27 GB before non-matvec work, so the aggressive
row-scaled path must remain gated by end-to-end quality.

## Root cause

The persistent Qwen worker parallelizes attention and dense FFN rows, but for
each of the 48 SSM layers thread 0 disables the pool and calls the complete SSM
forward routine while the other 47 cores wait. This serializes the large QKV,
gate, and output projections as well as convolution and recurrence. Fixing this
execution structure precedes further kernel tuning.

## Implementation design

1. Add persistent-forward timings for QKV, SSM projections/recurrence,
   attention output, FFN gate/up/down, logits, barriers/serial work, and exact
   bytes for each tensor format.
2. Parallelize the SSM stages within the already-running persistent workers:
   row-split QKV/gate projections, channel-split convolution, head-split
   normalization/recurrence, and row-split output projection. Pre-dequantize
   the small depthwise-convolution weights once per layer.
3. Open Q8 lazily and selectively materialize decode tensors. Keep embeddings
   mmap-backed, skip NextN for `spec_k=0`, discard source pages after each
   tensor, and abort loading before `MemAvailable` falls below 2 GB.
4. Provide three Q8 modes:
   - `reference`: resident GGUF Q8_0 with F32 activations.
   - `block64`: eight-row, 64-column panels retaining original Q8 values and
     block scales, with per-64 activation quantization and SVE SDOT.
   - `row`: eight-row int8 panels with one weight and activation scale, full-K
     int32 accumulation, and scale application once per output row.
5. Quantize each source activation once and reuse it for gate/up and Q/K/V.
   Split panels on eight-row boundaries and align the worker mapping with the
   four 12-core CMGs.
6. Expose `--q8-mode reference|block64|row|auto`. `auto` may select a fast mode
   only after its numerical and greedy-token gates pass; reference always
   remains available.

## Gates

- Packing tests cover every Qwen matrix shape, tails, exact source decoding,
  finite output, and int32 overflow bounds.
- Projection comparisons require cosine at least 0.9999 and relative L2 at
  most 1%.
- A fast mode must reproduce 128 greedy tokens across ordinary, multilingual,
  code, and activation-spike prompts, including the known synthetic next token
  3165.
- Performance uses at least 32 warmed tokens with `max_seq=64`, reports peak
  memory and stage timings, and runs only on the non-Codex node.
- Target: at least 600 GB/s for the row kernel and 20 tok/s end to end. If the
  row mode fails quality, `auto` falls back and the quality-preserving ceiling
  is reported instead of weakening the gate.

Q4-specific packing is deferred until Q8 is closed, although Q4 receives the
shared persistent-SSM parallelism and is remeasured for regression coverage.
