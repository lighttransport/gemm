# GLM-5.1 decode on A64FX/Fugaku — long-context roofline

Throughput estimate for decoding **GLM-5.1** (`GlmMoeDsa`: 744B total / 40B active
MoE, **MLA + DeepSeek Sparse Attention**) at long context on Fugaku A64FX nodes.
Calculator: [`tools/decode_estimate.py`](../../tools/decode_estimate.py). Cost
constants come from the `a64fx/utofu-tests` suite and the in-model decode profile.

## Model facts (HF `GlmMoeDsaConfig`)

| | value |
|---|---|
| total / active params | 744B / 40B (256 routed experts, top-8 + 1 shared) |
| layers / hidden | 78 (3 dense + 75 MoE) / 6144 |
| attention | **MLA**: kv_lora_rank 512, qk_rope 64 → **576 latent values/token/layer** |
| sparse attn | **DSA**: `index_topk=2048`, indexer head_dim 128, ~½ layers full-scan |
| footprints | 2-bit 220 GB · 4-bit ~420 GB · 8-bit 805 GB · bf16 1.65 TB |

The architecture is what makes 1M context tractable: MLA stores a *compressed
latent* (≈90 GB total KV at 1M, not multi-TB), and DSA reads only the **top-2048**
selected tokens at decode — so attention traffic is the indexer scan (~5 GB/token),
**not** a full-KV stream. Decode is therefore **weight-streaming + per-layer-comm
bound, not KV-bound**.

## Cost model (per decode step)

```
t_step = max(t_mem, t_comm)              # overlap; t_mem + t_comm if no overlap

t_mem  = (weight_read + attn_read·B) / (BW·N)      # weights amortize over batch B,
                                                    # KV is per-sequence
  weight_read = 40e9 · (bits/8)                     # MoE active params, once/step
  attn_read   = idx_dim·idx_bytes·n_idx_layers·S    # DSA indexer scan over context
              + latent·kv_bytes·min(2048,S)·layers  # main sparse attn (tiny)

tree   = tree_steps(TP) · (1.23µs + hidden·2·B / 6.36GB/s)   # one fused all-reduce (upper bound)
t_comm = layers·0.5·tree                      # attn TP all-reduce, decomposed RS+AG (#13)
       + moe_layers·{1.8|4.5}·tree            # MoE dispatch+combine, uniform | skew (#10)
```

**A64FX/uTofu constants (measured):** realized matvec BW **650 GB/s/node** (not the
900 GB/s stream peak — decode is GEMV); uTofu per-hop **1.23 µs + bytes/6.36 GB/s**;
tree all-reduce **log₂(⌊pow2⌋)(+2 if non-pow2)** steps; comm is **context-independent**.
HBM **32 GiB/node** (≈30 usable).

The per-layer comm term is no longer a flat `2 × tree`; each collective is now the
**measured best-case** cost from `a64fx/utofu-tests`, as a multiple of one fused tree
all-reduce (`--route` selects the MoE case):
- **Attention TP all-reduce** done as decomposed multi-TNI **reduce-scatter +
  all-gather** (`reducescatter_bench`, #13): **0.52×** the fused tree at B=1 decode
  (0.23× at B=256) — the fused tree is pessimistic for TP comm expressible as
  scatter-then-gather. (Caveat: a real RS also does N−1 shard-sized adds the fused
  tree folds in — negligible at decode shards.)
- **MoE dispatch+combine** (top-8-of-256 all-to-all, `moe_dispatch_bench`, #10):
  - *uniform routing + multi-TNI* → **0.88–0.93×** of `2 × tree` ≈ **1.8 trees**.
    (Distinct destinations across the 6 TNIs give ~3×; naive single-TNI is ~2.8× one
    tree and would blow the budget — the roofline assumes multi-rail.)
  - *skewed routing (hot experts)* → **2.2–2.7×** of `2 × tree` ≈ **4.5 trees**:
    traffic concentrates on a few hot links (agg BW halves), and multi-TNI cannot
    rescue a single congested link (#5). `--route uniform` is the load-balanced
    case; keep routing near-uniform (aux load-balance loss + capacity factor, the
    standard EP mitigation) or pay the skew term.

A **pipeline-parallel** alternative (`pp_handoff_bench`, #14) drops the per-layer
collectives entirely: comm is fixed at `S−1` single-link handoffs for the *whole*
model (one TofuD link/hop, torus-distance-free) — ~80× below TP's `2·layers`
all-reduce sum — but decode's one-token-in-flight makes the pipeline bubble 92% at
S=12, so PP is a high-batch *throughput* lever, not a single-stream latency one.

## Results — 1M context, single stream (batch=1)

```
                       TP   agg BW    mem      comm    ->step  (bound)  tok/s   fit
4-bit / 24 nodes       8    15.6 TB/s 1.61ms   1.65ms  1.65ms  (comm)   606     21 GiB OK
4-bit / 48 nodes       8    31.2 TB/s 0.81ms   1.65ms  1.65ms  (comm)   606     11 GiB OK
4-bit / 60 nodes       12   39.0 TB/s 0.65ms   2.75ms  2.75ms  (comm)   364      9 GiB OK
8-bit / 24 nodes       8    15.6 TB/s 2.90ms   1.65ms  2.90ms  (mem)    345     37 GiB OOM ✗
8-bit / 48 nodes       8    31.2 TB/s 1.45ms   1.65ms  1.65ms  (comm)   606     19 GiB OK
8-bit / 60 nodes       12   39.0 TB/s 1.16ms   2.75ms  2.75ms  (comm)   364     15 GiB OK
--- reference (route=uniform; --route skew ~2.2x the comm term) ---
2-bit / 12 nodes       12    7.8 TB/s 1.95ms   2.75ms  2.75ms  (comm)   364     26 GiB OK
bf16  / 96 nodes       8    62.4 TB/s 1.37ms   1.65ms  1.65ms  (comm)   606     18 GiB OK
bf16  / 192 nodes      8   124.8 TB/s 0.68ms   1.65ms  1.65ms  (comm)   606      9 GiB OK
```

## Key findings

1. **Single-stream floors out on per-layer comm at ~606 tok/s (TP=8).** Once you
   have enough nodes, the context-independent per-layer collective latency is the
   floor — adding nodes only helps the memory-bound term. With the measured-collective
   model that floor is 78 attn all-reduces (0.5 tree each, decomposed RS+AG) + 75 MoE
   dispatch+combine (1.8 tree each, uniform) ≈ 174 tree-equivalents/step. bf16/192
   ties bf16/96; 4-bit/48 ≈ 8-bit/48 ≈ 606. This is the ring-attention lesson
   restated: **comm is the fixed cost, compute parallelizes.** (Skewed MoE routing
   drops the ceiling to ~280 tok/s — the MoE term ~2.2×; see `--route skew`.)

2. **Use node counts divisible by 8.** 24, 48, 96, 192 give clean **TP=8** groups
   (3 tree steps → 606 tok/s ceiling). **12 and 60 are awkward** — forced to TP=12
   (5 steps) → comm rises, ceiling drops to **364 tok/s**. The 60-node penalty is
   pure layout, not compute. (Override with `--tp`.)

3. **8-bit does not fit on 24 nodes at 1M** (37 GiB/node > 32). Minimum ~32 nodes;
   use **8-bit/48**. 8-bit is the **A64FX-native** quant (int8 `svdot` W8A8, no
   dequant — proven at decode parity), so its numbers are the most trustworthy.

4. **Memory→comm crossover:** 4-bit past **~27 nodes**, 8-bit past **~44 nodes**.
   Below the crossover, fewer nodes / lower bits help (mem-bound); above it, you're
   paying for nodes that don't speed up one stream.

5. **Batching is KV-memory-limited at 1M.** Each sequence needs its own ~90 GB
   latent cache, so batch>1 at 1M OOMs quickly (batch=32 → 60+ GiB/node). Long
   context trades away batch throughput; short/medium context batches fine and
   approaches the memory ceilings (1000s of tok/s aggregate).

6. **4-bit needs unpack→int8** (no native int4 on A64FX), so its BW saving is softer
   than the table; **8-bit is the safer realized number.**

## Caveats

These are **optimistic roofline ceilings**. The DSA top-k sort over 1M, softmax,
expert routing, and kernel-launch overheads are not modeled — expect real
single-stream **2–3× lower** (~200–300 tok/s). Apply `--derate 0.4` for a
realistic figure.

## Running it

```bash
tools/decode_estimate.py --bits 4,8 --nodes 24,48,60 --ctx 1000000   # this table
tools/decode_estimate.py --sweep                                      # full bits×nodes
tools/decode_estimate.py --bits 8 --nodes 48 --ctx 200000 --batch 16  # batched, 200K
tools/decode_estimate.py --bits 8 --nodes 48 --ctx 1000000 --derate 0.4   # realistic
```
Override any constant: `--bw`, `--tp`, `--kv-bytes`, `--idx-bytes`, `--act-bytes`.

Sources: [Unsloth GLM-5.1](https://unsloth.ai/docs/models/glm-5.1),
[HF GlmMoeDsa](https://huggingface.co/docs/transformers/en/model_doc/glm_moe_dsa).
See also `a64fx/utofu-tests/summary.md` (uTofu comm numbers) and auto-memory
`reference_ring_attn_decode_cost`.
