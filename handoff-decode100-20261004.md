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

- **New job PJM52167472** (12 nodes, 19:00→01:00): staged by `tmp/decode100-20261004/stage-j2.sh`; campaigns via
  `driver-j2.sh WAITPID ARMS TAG BIN` + `campaign-j2.py` (arms JSON). Queue: j2a (v18 kernels), j2b (v19 mHC
  prefetch), j2c (v20: crash bisect chunk16/hdr/kdaq at 8K + act options).
- **b14 (v15) crashed (SIGSEGV) at the 8K exact-sync run** with header cache + KDA team quant + fused paths; suspect
  is one of those options (v13 fusions alone were 8K exact). j2c bisects.
- **Single-node microbenchmarks (old job, 47 threads, FLIB_BARRIER=HARD):** barrier 1.2 µs; persistent dispatch
  1.2 µs (2.3 with a barrier); Q8 matvec 11.8 MB cold 33.6 µs (351 GB/s); tiny 48x4096 matvec 2.7 µs;
  **act_prepare_team(4096) 11.5 µs** → 7.7 (chunk16) → 6.9 (header cache) → **4.6 µs** (both).
- **mHC local kernel:** the residual copy cost ~16 µs/site (store traffic); eliding it (in-place post update,
  ce4d6052) + FP64 SVE post (2547fd42): local-gram **26.5 µs/site warm, 31.2 µs with 90 cold sites, legacy 59.5**.
  In-model (b17, v17 before SVE post) mHC dispatch was still ~62 µs/site; v18+ untested in-model yet.
- **KDA recurrence:** only 5–6 head tasks → 15.7 µs arrival spread; `--kda-decode-kernel columns16` gives 48
  bit-exact tasks (792-case native test PASS at 1/47 threads).
- **b16 (v17):** local-gram 0.997×, +cmg 1.008×; MTP d2 all-on **45.49** (best so far).

- **j2a (new job, v18):** cmg+columns16 8K BIT_EXACT; but **columns16 0.961/0.957×** and kern (cmg+local-gram+
  columns16) 0.963/0.963× — the 16-column KDA split is a regression (strided 64-B state accesses, 8× scalar reloads);
  dropped from later arms. MTP d2 kern 44.57. Verify d2 profile per position: moe 7.52, kda 5.06, mhc 4.05,
  sparse 3.34 ms — MoE ~55 GB/s effective in verify (j2h microbench queued).
- **MTP runs without the persistent executor** (`glm53f_mtp_spec_12n.c` never calls `glm53f_team_run`); converting is
  large (100+ plain parallel regions in verify-path files would nest to 1 thread). j2g measures the executor's
  value on plain decode first.
- Pending on new job: j2b (prefetch, columns-64 arm), j2c (crash bisect + act options), j2d (allreduce 2D/MTNI),
  j2e (KDA verify columns, state-gated MTP), j2f (batch mHC tail), j2g (executor value), j2h (verify MoE micro).

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
2. Fusion is done and measured null; focus on per-phase latency (act prepare, mHC, KDA recurrence spread) and
   allreduce latency/skew (90/token × ~24 µs pure + ~20 µs skew).
3. Overlap shared expert with the routed allreduce; consider reduce-scatter + all-gather into next mHC.
4. Wire R16 into decode experts with a byte-identical R16 prefill expander (`gmn_expand_sblk` mapping in this
   doc's commit message for 0aa37acb) so the GGUF layout can be dropped.
5. Promotion requires 8K and 32K qualification; none of the above is promoted.
