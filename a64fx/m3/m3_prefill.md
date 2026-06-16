# M3 prefill optimization — target ≥100 tok/s on A64FX

## Where the time goes now
Prefill currently calls `m3_forward_token` **once per prompt token** (M=1, same path as
decode). Measured ~1.5–5 tok/s. That is a *structural* loss, not a compute limit:
- every GEMM runs at M=1 (matvec) — no reuse of the weight read across tokens;
- one EP/TP all-reduce **per token per layer** — comm dominates;
- the MXFP8 FP8→f32 decode is redone per token.

FLOP budget says the headroom is huge. M3 activates ~23B params/token ⇒ ~46 GFLOP/token.
A64FX sustains ~1.0–1.3 TFLOP/s/node on these (bf16→f32-widened) GEMMs:
- 48 nodes ≈ 50–60 TFLOP/s ÷ 46 GFLOP ≈ **~1000 tok/s** theoretical; even at 20–40%
  efficiency that is **200–400 tok/s**. So **100 tok/s is reachable on the existing
  48–96 nodes** — the lever is batching the sequence, not adding nodes.

## Lever 1 (primary, biggest win): chunked batched prefill
Process the prompt in chunks of **S tokens** (e.g. S=256–512) through the *already-existing*
batched GEMM path (`m3_gemm`, M=N — the multi-stream decode kernels) instead of token-by-token.
Per chunk, per layer:
- **Projections / FFN / experts**: M=S GEMMs. Weight read + (for MXFP8) the FP8→bf16 decode
  amortize over all S tokens. Expert-grouping (already built) buckets the S×top4 (token,expert)
  pairs per owned expert → one big GEMM per expert.
- **ONE all-reduce per layer** for the whole chunk → comm amortized S-fold (the current
  per-token-per-layer reduce is the main killer).
- **NEW kernel — prefill (causal) attention**: a chunk of S query rows attends to K/V at
  positions [0, p] with a causal mask (token i sees ≤ p0+i). Flash-style streaming over the
  KV so memory stays bounded; MSA top-k block selection per query row keeps it ~linear at long
  context (only k·128 keys/query). This replaces the per-token decode attention for prefill.
- KV for all S positions written at once (bf16/fp16/int4, same cache).

Effort: moderate. The batched GEMM, expert grouping, MoE router, and KV codec all exist; the
genuinely new piece is the **batched causal MSA attention** kernel (`m3_forward_prefill_chunk`).
Expected: S=256 → comm/256 and GEMMs at the ~100+ Gflop/s the microbench already showed at
N=8 (rises further at larger M) ⇒ **several hundred tok/s** for single-request prefill.

## Lever 2: prefill-tuned parallelism (compute-bound ⇒ TP-heavy)
Decode is memory/latency-bound (favours EP + multi-stream). Prefill is compute-bound, so the
*opposite* layout is optimal: **TP-heavy** (split every GEMM across nodes for more FLOP/s/token)
with the per-chunk all-reduce amortizing the TP cost. Run the prefill phase with
`M3_TP_ATTN/FFN/SHARED/HEAD=1` + EP for experts; decode keeps EP + multi-stream + int4-KV/CP.
Same binary, different env per phase.

## Lever 3 (the requested layout): disaggregated prefill/decode, 192 = 96 + 96
For **sustained multi-request serving** (not single-request latency), split the allocation:
- **Prefill cluster (96 nodes)**: TP-heavy chunked prefill (Levers 1–2), maximises FLOP/s,
  large chunks. Produces the full KV cache for each prompt.
- **Decode cluster (96 nodes)**: EP + multi-stream (N=8) + int4-KV/CP, low per-token latency.
- **KV handoff over uTofu**: after a prompt's prefill, RDMA-`put` its KV cache (per layer, per
  position) from prefill ranks → decode ranks, then decode continues autoregressively. The repo
  already has the uTofu put/get primitives (`a64fx/utofu-tests`); int4-KV makes the transfer
  3.9× smaller, and CP means both sides already think in block-cyclic shards — the handoff is a
  shard-to-shard remap. Transfer cost per 1k-token prompt @ int4 ≈ 1k·(k/v/idx per layer)·60 ≈
  ~0.5 GB → a few ms over Tofu-D (~6.8 GB/s/link, multi-link) — negligible vs prefill compute.
- **Win**: pipeline — prefill of request B overlaps decode of request A; no head-of-line
  blocking; each phase runs its optimal parallelism. This is the throughput architecture
  (Mooncake/DistServe-style P/D disaggregation) the 192-node split enables.

Cost: the largest lift — a KV transfer protocol (layout matching prefill-TP ↔ decode-EP/CP),
a request scheduler, and double the allocation. Justified only once Levers 1–2 saturate a
single cluster.

## Recommended phasing
1. **Lever 1 — chunked batched prefill** (single cluster). Biggest win, gets ≥100 tok/s
   (likely 200–400). New `m3_forward_prefill_chunk` + batched causal-MSA attention.
2. **Lever 2 — prefill TP-heavy config**. Cheap (env), squeezes the per-node FLOP rate.
3. Measure. If single-request prefill is enough, **stop** — 100+ tok/s is met without 192 nodes.
4. **Lever 3 — disaggregated P/D + uTofu KV handoff** only if the goal is sustained
   *concurrent-request* serving throughput (then the 192-node split pays off).

## Validation
- Microbench the batched GEMM at M=64/128/256 (extend `m3_mxgemm_bench`) → confirm Gflop/s
  saturates → per-token cost.
- Real 48n chunked prefill of a 2–8k prompt → tok/s + ‖x‖/argmax match the token-serial path
  (correctness) + NaN=0.
- Lever 3: a 2-cluster uTofu KV-handoff smoke (prefill 4n → decode 4n, verify decode continues
  coherently from transferred KV).
