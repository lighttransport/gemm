# GLM-5.2 FP8 Prefill: Scaling Study and the Path to ~50 tok/s

Date: 2026-06-22. Hardware: Fugaku A64FX (48 compute cores/node = 4 CMGs × 12 cores,
~32 GB HBM2/node, 4 NUMA domains). Model: GLM-5.2 FP8, 78 layers (3 dense + 75 MoE),
hidden=6144, dense FFN inter=12288, expert inter=2048, 256 routed + 1 shared expert, 8 active.

Synthetic chunked prefill, 1024 tokens, short context (CP=0, maxpos=2048), FP8 weights,
`pchunk=256`, 5-token GEMM kernel (`GLM5_FP8_GEMM_TOK=5`), `GLM5_COMM_OVERLAP=1`.

## Measured scaling (96 nodes unless noted)

| Lever | Result | Limiting factor |
| --- | --- | --- |
| Threads 12 / 24 / 48 | 16.5 / **18.5** / 14.1 tok/s | OpenMP **cross-CMG fork-join**: effective compute caps at ~2 CMG |
| th=48 + `numactl --interleave` + `OMP_PROC_BIND=spread` | 16.3 tok/s | NUMA helps (+15% vs plain th=48) but **still < th=24** |
| th=48 + interleave + `bind=close` | 13.2 tok/s | worse — packing 4 CMGs doesn't help |
| th=24 + NUMA interleave | 18.65 tok/s | flat vs no-NUMA (already on 2 CMGs) |
| Nodes 48 / 96 (TP_SHARED=1) | 15.7 / 16.5 tok/s | head-sharding saturates (only 64 attn heads) |
| TP_SHARED 0 → 1 | +10–14%, comm 60%→35% | shards the shared expert rows |
| TP_FFN=1 | no gain | dense FFN is only 3/78 layers |
| pchunk 128 / 256 / 512 | 14.4 / **15.7** / 14.4 (48n) | 256 optimal |
| Pure-EP (all TP off, th=24) | **OOM (59 GB arena > 31 GB)** | replicating dense everywhere doesn't fit |

Best measured: **~18.7 tok/s** (96n, th=24, TP_SHARED=1, pchunk=256, tok=5, comm-overlap).

## Why ~18.7 tok/s is the practical ceiling

Three limits are mutually reinforcing — relaxing one hits another:

1. **OpenMP cross-CMG fork-join.** Each layer issues many parallel regions (qkv/o_proj/router/
   shared/expert GEMMs × pchunk). At 24 threads (2 CMGs) the fork-join barrier is cheap; at 48
   (4 CMGs) the barrier cost per region, multiplied across thousands of regions, exceeds the
   added compute — so th=48 *regresses*. NUMA interleave narrows but does not close the gap. The
   prior persistent pinned-pool attempt (`GLM5_POOL`) to dodge fork-join did not work (hangs).
   ⇒ usable compute caps at ~2 of 4 CMGs.
2. **31 GB/node memory forces TP sharding.** Pure-EP (replicate all dense, shard only routed
   experts) needs a 59 GB arena and OOM-kills. So dense compute *must* be TP-sharded to fit,
   which means an allreduce per shard per layer.
3. **Comm becomes dominant.** With TP sharding and once compute speeds up (th=24), comm is ~48%.
   Even perfectly hiding all comm would only lift th=24 from ~18.5 to ~28–30 tok/s.

50 tok/s ≈ 2.7× over the ceiling. No combination of threads / nodes / pchunk / TP knobs / GEMM
kernel reaches it: compute is fork-join-bound at 2 CMGs and comm is ~half the wall.

## The one architecture that could reach ~50 tok/s: 4 ranks/node + shared-dense mmap

The root waste: each node runs **1 MPI rank using 12 of its 48 cores**; the other 3 CMGs sit
idle because (a) OpenMP can't cross CMGs cheaply and (b) 4 MPI ranks/node would 4×-replicate the
~22 GB dense weights and OOM.

Design — make each CMG its own MPI rank, but share the (identical) dense weights once per node:

- **4 MPI ranks/node, 1 per CMG**, each pinned to its 12 cores (`OMP_NUM_THREADS=12`, bound to
  that CMG's cores). No cross-CMG fork-join — every parallel region stays within one CMG.
- **Dense weights shared per node.** Dense tensors (attention wq/wkv, shared expert, dense FFN,
  embed, head) are byte-identical across the 4 ranks. Stage them **once per node** into a
  POSIX shared-memory / `mmap(MAP_SHARED)` blob; all 4 ranks map it read-only. Memory ≈ 22 GB
  (dense, shared once) + 4 × (KV + scratch + this-rank routed-expert shard, each small) ⇒ fits
  31 GB. The current per-rank `*.blob` staging is split into a shared dense blob + a per-rank
  expert blob.
- **Routed experts** shard over `4 × nodes` ranks (finer EP) — fewer experts/rank, less expert
  compute and memory per rank.
- **Hierarchical allreduce.** The TP/EP reduces become 2-level: the 4 intra-node ranks reduce
  through shared memory (no NIC), then one rank/node does the inter-node uTofu reduce, then
  broadcast back intra-node. This keeps inter-node message count at ~node-count (not 4×) while
  using all cores.

Expected: ~2–3× compute (4 CMGs, no fork-join) with comm held roughly constant by the
hierarchical reduce ⇒ plausibly **~35–50 tok/s**. Not guaranteed — see risks.

### Risks / unknowns
- Staging rework: split dense (shared) vs expert (per-rank) blobs; node-local shared-memory
  segment lifecycle (create by local-rank-0, map by the other 3, barrier, cleanup).
- uTofu at higher rank counts: the EP group grows 4× (96 nodes → 384 ranks). The 384-rank path
  **now runs end to end** — four scaling walls were fixed: `MAX_NODES 256→512` (`d5513d3`),
  `TP_AR_MAXN 256→512` + `TP_AR_NSTEP 9→11` (`7f4bb51`), and a robust barrier for the 383→1
  fan-in incast (`4a19895`). 384n short prefill = 15.83 tok/s, NaNs=0. Hierarchical reduce
  (intra-node shared-mem) would further cut the inter-node uTofu group size back to node-count.
- Memory headroom is tight (~22 GB dense + 4× scratch on a 31 GB node); KV/scratch must stay lean.
- Per-CMG HBM locality: the shared dense blob should be NUMA-interleaved or each CMG's hot rows
  first-touched locally.

### Suggested incremental path
1. [DONE, 4a19895] 384-rank init now works (robust barrier). 384n prefill runs end to end.
2. Prototype node-local shared-dense mmap with **2 ranks/node** (2 CMGs) first — lower memory and
   comm risk — and confirm the fork-join win materializes before going to 4.
3. Add the hierarchical (intra-node shared-mem) allreduce.
4. Scale to 4 ranks/node and re-measure.

## Recommended current best config (until the above is built)

```sh
ssh fugaku 'cd ~/work/gemm/glm5-1 && pjsub --no-check-directory \
  -x GLM5_MAXPOS=2048 -x GLM5_PREFILL_SYNTH=1024 \
  -x GLM5_CP=0 -x GLM5_INT4_KV=0 -x GLM5_MSA=0 \
  -x GLM5_TP_SHARED=1 -x GLM5_PCHUNK_SWEEP=256 -x GLM5_THREAD_SWEEP=24 \
  -x GLM5_COMM_OVERLAP=1 -x TP_AR_BF16=1 \
  a64fx/glm5/pjsub_glm5_prefill_fp8_96n_noncontig.sh'
```

→ ~18.7 tok/s base; **+9% with `TP_AR_BF16=1`** (torus-confirmed 16.09→17.55, comm 43→33%,
argmax-identical → output-equivalent on synth), NaNs=0. (`tok=5` GEMM is the binary default.)
Note `th=24` over `th=12`: the earlier "best" used 12 threads; 24 (2 CMGs) is ~+12%.


## UPDATE (2026-06-22): 4-ranks/node measured — NOT a win (Phase 2 closed)

The multi-rank/node uTofu addressing was fixed (distinct TNI per local rank, commit) so 2- and
4-ranks/node now run. Decisive 40-layer test, same 96 nodes, fp32 AR:

```text
1-rank th=24 (2 CMG):       40.33 tok/s  comm 34%
1-rank th=48 (4 CMG):       27.14 tok/s  comm 45%   (fork-join regression, as expected)
4-ranks/node th=12 (4x1CMG):16.02 tok/s  comm 61%   <- WORST
```

4-ranks/node is the **worst**. The 1-rank-per-CMG idea does avoid cross-CMG fork-join, but using
more *ranks* to get there backfires: (1) the all-reduce becomes 384-way and comm explodes to 61%;
(2) the un-shardable replicated dense (q_a/kv_a/gate/norms) is recomputed per rank, and attention
caps at 64 heads, so more ranks add redundant work, not parallelism. Even a perfect hierarchical
reduce (comm back to node-count) only recovers ~40 tok/s compute-only = **equal to 1-rank th=24**,
not better. So the shared-dense staging + hierarchical-allreduce rework would not pay off and is
**not pursued**. The 2-CMG (th=24) point is the genuine optimum; the ~20 tok/s full-model ceiling
(≈40 tok/s at 40 layers) is fundamental for this model on A64FX. The multi-rank/node capability is
kept (committed) as it may help other configs, but is off by default (GLM5_RANKS_PER_NODE=1).


## UPDATE (2026-06-22): batched CP attention combine (Tier-B / context-parallel comm)

The above is short-context (CP off). For **context-parallel (Tier B) prefill** the dominant comm
was the cross-rank attention flash-combine: `glm5_forward_prefill_chunk` merged the per-rank
partials with a **serial per-token loop**, firing one tiny `allreduce_max` + one `allreduce_sum`
*per token* — `2·S` latency-bound recursive-doubling collectives per layer per chunk. uTofu
allreduce is latency-bound (per-message α over O(log N) rounds), so those per-token launches
dominate.

Batched it into **two collectives per layer**: one MAX over the contiguous `[S·nh]` maxes + one
SUM over the packed `[S·(nh+nh·hd)]` payload (commit `7f6d620`). Bit-identical (allreduce reduces
each element under a fixed rank schedule regardless of fragmentation); the local rescale/normalize
now run OpenMP-parallel. Gated by `GLM5_CP_COMBINE_BATCH` (default 1; `=0` = per-token fallback).

A second fix was required for it to help by default: CP forced `ar_tokens=1`
(`max_count=hidden=6144` floats) — *smaller* than one token's combine payload
(`nh+nh·hd=16448`), so the per-token reduce already fragmented and batching had nothing to merge.
Now the lean CP slot is auto-sized to a chunk's combine payload, bounded by `GLM5_AR_HARD_CAP`
(64 → `max_count=393216`, region ~18 MB/rank — negligible) (commit `9db4b95`).

Measured, 12 nodes, static CP (`GLM5_CP=1 GLM5_CP_THRESHOLD=-1 GLM5_MSA=1`, bf16 KV), 8 layers,
synthetic 1024-tok prefill, th=24, **no `GLM5_AR_TOKENS` override**:

| pchunk | batched tok/s | per-token tok/s | speedup | combine calls (batch→per-tok) | frags |
| --- | --- | --- | --- | --- | --- |
| 64  | **68.3** | 66.8 | +2.3% | 208 → 8272 (40×) | 592 → 16464 (28×) |
| 128 | 54.3 | 52.0 | +4.5% | 104 → 8232 (79×) | 528 → 16464 (31×) |
| 256 | 50.6 | 49.8 | +1.7% | 52 → 8212 (158×) | 464 → 16464 (35×) |

`pchunk=64` is fastest (tok/s falls at larger pchunk — the CP+MSA batched-attention working-set
effect, independent of the combine). The combine wins at every pchunk: collective invocations cut
**40–158×**, fragments **~30×**, NaNs=0. Correctness is bit-exact — verified single-threaded
(deterministic), batch on/off both `argmax=152743` across repeated runs (the synthetic argmax is
otherwise run-to-run nondeterministic under threads, including the unmodified per-token baseline —
a speed signal only, not a correctness signal).

The **wall-clock win is small at 12 nodes** because comm is only ~15% of the prefill wall there;
it scales with node count (this study's 96n attn-CP is comm-bound). A/B harness:
`a64fx/glm5/run_glm5_cp_combine_ab.sh`. Follow-ups not yet done: overlap the combine SUM with the
o-proj GEMM; batch the MSA block-score reduce across the chunk.


## UPDATE (2026-06-24): data-parallel prefill groups + dynamic merge (the real throughput win)

The ~18.7 tok/s ceiling above is for **one sequence on all the ranks**. But at short/mid context
the whole model fits sharded across far fewer than all ranks, so dedicating every rank to one
sequence wastes the per-token collective (the allreduce/argmax over N ranks is ~half the wall). The
fix is **data parallelism**: split the N-rank EP pool into **G independent groups** of `GSize=N/G`
contiguous ranks; each group is a complete EP model (`ep_rank/ep_size` group-local) prefilling its
**own** sequence, with **group-scoped collectives** (a `gbarrier` rooted at the group base over a
disjoint slot region; `tp_comm_init(PeerVcq+GBase, GRank, GSize, gbarrier)`). Only the bootstrap and
final barrier are global, so groups run fully async. The model code is unchanged — it already shards
purely off `m->ep_rank/m->ep_size`. Commits on `glm5-2` (`288de08b`, `9825950c`, `f68c632d`,
`8e852f77`, `e34fd707`, `1e4e8014`).

### Phase 1 — static groups: ~4.6× short-context throughput

384 nodes, 8k synthetic prefill, `TP_SHARED=1`, th=24, pchunk=256, all NaNs=0:

| config | per-group rate | aggregate | speedup |
| --- | --- | --- | --- |
| 1×384 | 13.3 | **13.3 tok/s** | 1× |
| 2×192 | 13.8 / 15.3 | **29.0 tok/s** | 2.2× |
| 4×96 | 16.4 / 15.4 / 13.0 / 15.6 | **60.5 tok/s** | **4.6×** |

Slightly **super-linear** in group count: per-group rate *rises* as groups shrink (13.3 → 14.5 →
15.1) because a 96-rank collective is ~4× cheaper than a 384-rank one, and the extra per-rank expert
compute (4× more experts/rank at ep96) does not dominate. Single ep96 alone = 15.96 tok/s. Beats the
naive `4 × 13.98 = 55.9` target. `pick_groups()` auto-selects the largest G whose Tier-A KV budget
at `ep=N/G` holds the target context (the per-rank weight footprint grows as groups shrink, so
smaller groups have a lower context ceiling); `GLM5_PREFILL_GROUPS=G` overrides. `glm5_stage` shards
group-locally (`rank %= ep_size`); the 384n script sets `GLM5_EP_SIZE=NP/NGRP` (do **not** name the
bash var `GROUPS` — it is a reserved array, silently broke staging once: `GLM5_EP_SIZE=NP/GID=0`).

### Phase 2 — dynamic merge: 4×96 → 2×192 → 1×384 as one sequence grows

As the surviving sequence's KV outgrows the small group's HBM, **merge adjacent groups** (concurrency
4→2→1; survivors get more nodes, fewer experts/rank, more Tier-A headroom). Each merge, per sibling
pair (`GLM5_MERGE_AT=p1:p2`; `:` because `pjsub -x` splits on `,`):
- **Routed experts: local drop** (`glm5_group_expert_drop`) — both sibling subgroups independently
  staged the full model, so the new owner `e%(2g)` already holds each expert; just free the
  redundant copies (no transfer).
- **TP-dense weights: in-place re-slice from the node-local blob** (`glm5_group_tp_reslice`) —
  `glm5_stage` keeps dense **un-sharded**, so each rank re-slices its new-ep shard of
  embed/head/attn/shared/ffn locally (no transfer), freeing each old slice first (no 2× spike).
- **KV + MSA index: uTofu pairwise propagate** even→odd (`glm5_kv_xfer`) with `dc civac` cache
  coherence; register `cp_nslot` (the tiered Tier-A buffer size, **not** `max_pos`).
- **Collectives rebuilt** over the merged group (`tp_comm_free` + re-init, `gbarrier` membership).

**Validated lossless**: 24n real weights, G=2 merge ep12→ep24 at pos=1024 (3.1 s reshard) →
survivor `argmax=60590` == G=1 reference `60590`, NaNs=0. 384n forced-merge ladder (8k): merges
18.2 s / 30.2 s, survivor 8192 tok NaNs=0, per-tier survivor 17.2 / 14.9 / 13.6 tok/s.

### 1M-context source-token ladder (384n)

One job, 1,048,576 tokens **tokenized from this repo's source** (`GLM5_PROMPT_TOKENS`, binary
uint32, cycled), `GLM5_KV_BUDGET_GB=4` → T_cp=39168 so the merges (16384:32768) fire below it and
the within-384 Tier A→B handles the long tail. All events fired, NaNs-free:

```text
group_merge: 4x96  -> 2x192 at pos=16384 (184.4 s)
group_merge: 2x192 -> 1x384 at pos=32768 (189.2 s)
prefill_tier: A->B re-shard at pos=39168 (0.12 s) -> CP int4, 2816 slots/rank
```

| tier | window | cumulative tok/s |
| --- | --- | --- |
| 4×96 (Tier A, **MSA on**) | [0, 16k] | 12.4 → 9.95 |
| 2×192 | [16k, 32k] | → 6.78 |
| 1×384 Tier A | [32k, 39k] | → 6.05 |
| 1×384 CP int4 + sparse-MSA | [39k → 1M] | ~5.1 (cumulative), ~4.8 inst |

Notes: the tiered path keeps **MSA on throughout** (Tier B needs it), so the Tier-A rate here (~8–12
tok/s) is below Phase 1's MSA-off ~16 — the throughput numbers above are the right reference for
peak short-context, this is the 1M-capable path. The merge cost (~185 s) is the 78-layer TP re-slice
(local blob read) + KV/index propagate, one-time, amortized over the run. The CP tail sustains ~4.8
tok/s (faster than the ~2.85 of CP-from-start, thanks to the batched combine + sparse MSA), so a 24 h
window reaches ~360–420k of the 1M tokens (1M needs ~90 h at this rate; not bounded by tok/s but by
wall time). `glm5_model.kv_avail` is stored for a future **auto-merge** (recompute T_cp after each
merge so merges fire by memory, not forced points) — not yet wired; the Tier-A `kv_cache` would also
need resizing per merge.
