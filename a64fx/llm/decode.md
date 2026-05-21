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

t_comm = layers · 2 · tree_steps(TP) · (1.23µs + hidden·2·B / 6.36GB/s)
```

**A64FX/uTofu constants (measured):** realized matvec BW **650 GB/s/node** (not the
900 GB/s stream peak — decode is GEMV); uTofu per-hop **1.23 µs + bytes/6.36 GB/s**;
tree all-reduce **log₂(⌊pow2⌋)(+2 if non-pow2)** steps; comm is **context-independent**.
HBM **32 GiB/node** (≈30 usable).

The two-collectives-per-layer MoE term (`COLLECTIVES_PER_LAYER=2`) is now
**measured** by `a64fx/utofu-tests/moe_dispatch_bench` (top-8-of-256 all-to-all
dispatch+combine on the 12-node torus), with two important conditions:
- **uniform routing + multi-TNI dispatch:** real cost = **0.88–0.93×** of
  `2 × tree-all-reduce` — the proxy is sound, slightly conservative. (Distinct
  destinations across the 6 TNIs give ~3×; a naive single-TNI dispatch is ~2.8×
  the proxy and would blow the budget — the roofline assumes multi-rail.)
- **skewed routing (hot experts):** real cost rises to **2.2–2.7× the proxy** —
  traffic concentrates on a few hot destination links (agg BW halves) and the
  proxy under-estimates. So `COLLECTIVES_PER_LAYER=2` is the *load-balanced* case;
  derate the MoE comm term for routing skew, or keep routing near-uniform
  (auxiliary load-balance loss + capacity factor, the standard EP mitigation).

## Results — 1M context, single stream (batch=1)

```
                       TP   agg BW    mem      comm    ->step  (bound)  tok/s   fit
4-bit / 24 nodes       8    15.6 TB/s 1.61ms   1.48ms  1.61ms  (mem)    620     21 GiB OK
4-bit / 48 nodes       8    31.2 TB/s 0.81ms   1.48ms  1.48ms  (comm)   676     11 GiB OK
4-bit / 60 nodes       12   39.0 TB/s 0.65ms   2.47ms  2.47ms  (comm)   405      9 GiB OK
8-bit / 24 nodes       8    15.6 TB/s 2.90ms   1.48ms  2.90ms  (mem)    345     37 GiB OOM ✗
8-bit / 48 nodes       8    31.2 TB/s 1.45ms   1.48ms  1.48ms  (comm)   676     19 GiB OK
8-bit / 60 nodes       12   39.0 TB/s 1.16ms   2.47ms  2.47ms  (comm)   405     15 GiB OK
--- reference ---
2-bit / 12 nodes       12    7.8 TB/s 1.95ms   2.47ms  2.47ms  (comm)   405     26 GiB OK
bf16  / 96 nodes       8    62.4 TB/s 1.37ms   1.48ms  1.48ms  (comm)   676     18 GiB OK
bf16  / 192 nodes      8   124.8 TB/s 0.68ms   1.48ms  1.48ms  (comm)   676      9 GiB OK
```

## Key findings

1. **Single-stream floors out on per-layer comm at ~676 tok/s (TP=8).** Once you
   have enough nodes, the context-independent per-layer all-reduce latency
   (78 layers × ~2 collectives) is the floor — adding nodes only helps the
   memory-bound term. bf16/192 ties bf16/96; 4-bit/48 ≈ 8-bit/48 ≈ 676. This is the
   ring-attention lesson restated: **comm is the fixed cost, compute parallelizes.**

2. **Use node counts divisible by 8.** 24, 48, 96, 192 give clean **TP=8** groups
   (3 tree steps → 676 tok/s ceiling). **12 and 60 are awkward** — forced to TP=12
   (5 steps) → comm doubles, ceiling drops to **405 tok/s**. The 60-node penalty is
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
