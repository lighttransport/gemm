# GLM53F decode-100 handoff (A64FX, 12 nodes)

Snapshot: **2026-10-04 ~18:00 JST**, allocation PJM52159552 (expires 20:05). Goal: 100+ delivered decode tok/s
(≥80 first). Qualified recipe unchanged: **35.462134 decode / 412.273634 prefill tok/s**. No push; all new
behavior is opt-in and default-off. Plan: `~/.claude/plans/vectorized-napping-pearl.md`.

Scratch: `tmp/decode100-20261004/` (local + remote checkout `$HOME/work/gemm/glm53f-strata-20261001`).
Campaign scripts `campaign-b*.py`, drivers `driver-b*.sh`, builds `build-*.sh`, results `results-b*/`,
rank0 logs `tmp/glm53f-q4-52159552/benchmark-decode100-*`. Single-node benches in `tmp/decode-dummy-20261004/`.

## Commits (local only)
| Commit | Option | Status |
|---|---|---|
| a877f30e | `--sparse-verify-kernel front\|front-onecoll` batched sparse front for MTP verify | model state BIT_EXACT (d2–d4) |
| 06eb0130 | `--moe-verify-router batch` | model state BIT_EXACT |
| 3c5d5495, a7188cc5 | link-time sync profiler (`GLM53F_PROFILE_SYNC=1/2/3`, `--wrap`) | diagnostic |
| e386227e, 3caf76aa, 211a2aec, 0f283fd6 | `--weight-placement cmg\|cmg-experts` | works since v9; state BIT_EXACT |
| 0aa37acb, 61ef4008 | R16 repacked Q4_K/Q5_K expert kernel + test | not wired into runtime |
| 0f283fd6, 211a2aec | `--mhc-kernel local` | IDs equal at 512; not bit-exact by design |
| afa58e10 | `--moe-layer-kernel fused` (router+top-k+experts one team) | 8K state BIT_EXACT (combined) |
| 2bda6f67, 40abd5f9 | `--kda-out-kernel fused`, `--kda-layer-kernel fused` (KDA inside preceding mHC team) | 8K BIT_EXACT; +0.3%/0.0% |
| d364942d | `--sparse-layer-kernel fused` (whole sparse layer in mHC team) | 8K BIT_EXACT (combined) |
| f567a861 | `--act-header-kernel cached`, `--kda-quant-kernel team`, `--mhc-kernel local-gram` | b14 (v15) |
| d081d9bf | setup/publish barrier removal in fused paths | b14 (v15) |

## Measurements (512-token context screens unless noted; not qualification)
- Plain decode 36.8–37.2 tok/s (27.2 ms/token). Best MTP: **45.19** (depth 3, all options, b9b), 44.96 (d2).
- MTP verify width curve (b2, front+router): legacy→batched +4.5% (d1), +8.4% (d2), +5.9% (d3).
- **Sync profile (b4/b8, plain, per token):** 258 team dispatches (17.6 ms incl. work), **1482 OpenMP barriers**
  (4.9 ms master wait; mHC sites ~2.1 ms), 90 allreduces 3.9 ms (skew 1.66 + pure 2.59; ~29 µs each).
  Dispatch time by site: KDA 4.13 ms, experts 3.87, mHC ~5.2 (two sites), sparse front 1.47, MLA 0.96.
- **P0 bytes:** ~1.55 GB weights/rank/token (experts 429 MB, KDA 315, replicated sparse front 272, o_proj 166,
  router 99, shared 94). Expert skew (max/mean rank load) 1.25× by simulation.
- **Placement:** `move_pages` is refused on compute nodes (errno path); v9 re-homes by copy-out + MADV_DONTNEED +
  CMG-local first touch, only pages fully inside weight spans (edge pages crashed the OpenMP runtime in v8).
  All 5,074 dense + 171,336 expert pages placed; decode **+1.0–1.25%**, with mHC local **+1.5%** (b9b pairs).
- **Single-node (b6/b7c/b7d):** expert kernel 78 µs/5 parts (stages: quant 8, gate/up 30, swiglu 5, down 35).
  R16 vs production on interleaved pool: 62.2→56.3 µs (−9.5%). mHC local + prefetch: 63.7→51.6 µs/site.
- **Allreduce (b10):** uTofu decode 16 KiB 23.6 µs, 4 KiB 15.7; MPI 48 µs; barrier 2.0 µs. Env variants did
  not propagate through mpiexec (identical numbers) — rerun with explicit `-x` if needed.

- **b11 (v9):** placement 8K BIT_EXACT; pairs cmg +0.8/+0.7%, both(cmg+mhc local) +1.3/+3.9%; MTP all d2 44.96, d3 44.73.
- **b12 slot (v13, all four fusions):** 8K BIT_EXACT. Pairs: kda-layer +0.3/0.0%, all fusions **−1.3/−1.7%**,
  fusions+cmg+mhc-local +0.2/−0.1%. MTP d2 all-on **45.41** (best so far, within noise).

## Conclusion so far
Placement and kernel work each give ~1–2% in-model, and removing dispatches/serial sections by fusing whole
layers into one team gives **nothing** (slightly negative). The 4.9 ms of barrier wait is therefore not
barrier/dispatch overhead but threads waiting on the slowest thread of each phase (imbalance and memory
latency inside phases) plus 90 allreduces with cross-rank skew. 100 tok/s needs a different execution model,
not more fusion of the current one. Candidates, in order:
1. Measure per-phase thread arrival spread (extend the sync profiler: per-site max-min arrival) to find which
   phases are imbalanced; fix their partitions (e.g. CMG-balanced rows, Q8 row groups vs 47 threads).
2. 4 MPI ranks per node (48 ranks × 12 threads, one CMG each): CMG-local barriers and memory; TP48 needs
   restaging and a hierarchical allreduce. Largest structural lever left on A64FX.
3. Overlap communication: shared expert during the routed allreduce; reduce-scatter + all-gather folded into
   the next mHC; MTP draft during the target allreduce.
4. Deeper MTP only after verify cost becomes sublinear (KDA/MoE verify batches still ~linear).

## Next steps
1. Read b14 (v15: local-gram mHC, header cache, KDA team quant, fused-path barrier removal) results.
2. Continue fusion: attention mHC-pre + KDA/sparse front as the mHC `after_normalize` hook; sparse layer
   front/select/MLA/o_proj into one region; reduce mHC barriers (2 per call minimum with local kernel).
3. Overlap shared expert with the routed allreduce; consider reduce-scatter + all-gather into next mHC.
4. Wire R16 into decode experts with a byte-identical R16 prefill expander (`gmn_expand_sblk` mapping in this
   doc's commit message for 0aa37acb) so the GGUF layout can be dropped.
5. Promotion requires 8K and 32K qualification; none of the above is promoted.
