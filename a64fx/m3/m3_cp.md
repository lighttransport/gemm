# M3 int4-KV + CP (context-parallel KV) for 1M context

## Why
M3 uses **GQA with full K/V heads** (kv_dim 512), not DeepSeek's MLA latent — so its KV
cache is large. At 1M context, per-rank (replicated) KV ≈ **137 GB**:
- k/v: 60 layers × 2 × 1M × 512 × 2B ≈ **123 GB**
- idx_k (MSA index): 57 layers × 1M × 128 × 2B ≈ **15 GB**

That dwarfs the 32 GB node, so 1M needs BOTH levers (neither alone fits):
- **int4-KV** ~3.9× → ~35 GB (still > a node)
- **CP** shards positions across N ranks → 35/N GB (e.g. 48n → ~0.7 GB/rank)

## int4-KV (`M3_INT4_KV`)
Per (position, head) group of `head_dim`=128: signed **int4** (±7) + a **bf16 scale**
(absmax/7). Storage: `head_dim/2` bytes + 1 bf16 vs `2·head_dim` bytes → ~3.9×.
- Codec: `m3_q4_pack` / `m3_q4_dot` / `m3_q4_axpy` (scalar, sign-extended nibbles), in
  `common/m3_impl.h`. Dequant happens on read in the attention score/PV and MSA-index loops
  (decode is latency-bound; scalar matches the existing bf16 inner loops).
- Caches: `k_q4/v_q4` `[nslot, kv_dim/2]`, `k_qs/v_qs` `[nslot, n_kv_heads]` bf16,
  `idx_q4 [nslot, idx_dim/2]`, `idx_qs [nslot]`. Independent of CP (`M3_INT4_KV=1` alone
  halves+ the KV on existing node counts → longer ctx without more nodes).

## CP (`M3_CP`)
Positions sharded **block-cyclic** across the EP ranks: block `b = pos/128` owned by rank
`b % ep_size` (block size == MSA block 128). Each rank stores only its owned blocks
(`cp_nslot = ceil(nblk/N)·128` slots); `m3_cp_owner/slot/mine` in `common/m3.h`.

Decode (per layer, per token):
1. Every rank computes q/k/v (full heads — **TP_ATTN forced off** under CP) and the MSA
   index q. Only the owner stores this position's k/v/idx.
2. **MSA block selection**: each rank scores only its OWNED blocks; `blk_reduce_cb`
   all-reduce-MAX gives every rank the same global block scores → identical global top-k +
   init/local selection; each rank keeps its owned selected blocks.
3. **Attention**: each rank flash-accumulates over its owned selected positions →
   unnormalized `(out, max, sumexp)` per head.
4. **Combine**: `kv_combine_cb` merges the per-rank partials with the online-softmax rule
   `gmx=max_r mx_r; out=(Σ_r e^{mx_r-gmx} out_r)/(Σ_r e^{mx_r-gmx} se_r)` (one all-reduce-MAX
   of the per-head max + one all-reduce-SUM of the rescaled [sumexp | out]). After combine
   every rank holds the full attention output → o_proj is local (no TP_ATTN reduce).

Comm callbacks live in `a64fx/m3/m3_ep_runner.c` (`ep_blk_reduce` = `tp_allreduce_max`,
`ep_kv_combine` = max + chunked sum over `tp_allreduce_sum`). They compose with the existing
EP all-reduce (MoE) and TP_HEAD argmax — same N ranks, different axis.

## Fit (1M, 48 nodes, MXFP8 weights)
- weights (MXFP8 @48n): ~10.7 GB/rank
- int4-KV @1M, CP over 48: ~35.4/48 ≈ **0.74 GB/rank**
- → ~11.5 GB/rank total ⇒ **1M fits with headroom** (32n also fits: ~14+1.1 ≈ 15 GB).

## Cost / notes
- Decode adds 2 collectives/layer (block-score MAX + flash combine) on top of the MoE
  all-reduce — like ds4p's CP, comm share rises but the cache becomes affordable.
- int4 is lossy (~±7 levels/group); coherence-gated against bf16 (job 49251800).
- mstream (throughput) path stays bf16 + replicated KV — CP/int4 target single-stream 1M.
- TODO: SVE-vectorize `m3_q4_dot/axpy`; overlap the combine with compute; CP for the
  mstream path if long-context batched serving is wanted.
