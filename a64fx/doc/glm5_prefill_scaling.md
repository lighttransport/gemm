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
  -x GLM5_COMM_OVERLAP=1 \
  a64fx/glm5/pjsub_glm5_prefill_fp8_96n_noncontig.sh'
```

→ ~18.7 tok/s, NaNs=0. (`tok=5` GEMM is the binary default.) Note `th=24` over `th=12`:
the earlier "best" used 12 threads; 24 (2 CMGs) is ~+12%.
