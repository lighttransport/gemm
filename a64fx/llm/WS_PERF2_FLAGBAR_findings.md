# Single-node decode perf revisit: pool flag-barrier (kept) + mHC serial-matvec (refuted)

**Status:** Phase 1 implemented + validated (bit-identical, kept on the worktree). Phase 2 tried and
**refuted by measurement, reverted.** Not committed.

## Motivation
After WS1/WS2/WS6 landed, revisited single-node decode for more headroom. CMG/first-touch locality
is already correct (Q8 repack / FP8 promote / synth fill all first-touch with the same
`ds4f_rowsplit8` the decode matvec reads) and SVE prefetch is already refuted (ds4f-opt.md). The
remaining hypothesis was **pool dispatch/barrier overhead** (~688 `ds4f_pool_run`/token, each a
48-thread barrier on one contended `done` atomic).

## Phase 1 — per-worker flag barrier (`DS4F_FLAGBAR`, default off) — KEPT
Replace the centralized `_Atomic int done` counter (47 workers `fetch_add` one cache line, main
spins on it) with **per-worker completion flags, each on its own 64-byte cache line**
(`ds4f_cacheline donef[]`); each worker writes its own flag = current `seq`, main waits each
distinct line. `common/ds4f_impl.h` pool primitive only (`struct ds4f_pool`, `ds4f_worker`,
`ds4f_pool_run`, `ds4f_pool_start/stop`); all ~688 call sites unchanged. **Bit-identical** (pure
synchronization — zero compute reorder).

**Microbench** (`tools/ws_poolbar_bench.c`, 48 thr): empty dispatch **~8.0µs → ~5.3µs**
(counter→flag), ~work dispatch ~28µs → ~22µs. The contended counter costs ~2.5–6µs extra/dispatch.

**Decode A/B** (Q8 dense, 5-rep median): **123.8 → 122.0 ms/tok (~1.5%)**, byte-identical
3618-line `||.||` trace, NaN=0. Gates: gemm 205/205, exact 5e-8, tierb2 2e-6, race-stress ×3 OK.

**Key finding:** the barrier is ~30–40% cheaper in isolation but only ~1.5% at decode level ⇒
**decode is NOT dispatch-bound.** The ~688 dispatches mostly do real BW/compute work; the barrier
is only ~5% of decode, and the flag removes ~40% of that. Still a clean, bit-identical, zero-risk
infra win that compounds across the whole forward pass — worth keeping.

## Phase 2 — mHC fn-matvec serial-SVE (`DS4F_HC_MV_SERIAL`) — REFUTED, reverted
Hypothesis: the tiny mHC fn-matvec ([24 or 4] rows × 16384) is dispatched to 48 threads but uses
≤24 and is dwarfed by the barrier ⇒ run it serial-SVE on tid0. **Measured the opposite:**
HC_MV_SERIAL=1 made decode **149→153 ms/tok** and the "other" phase **21.3→27.3 ms** (worse).

**Why:** the fn-matvec reads **1.5 MB of fn weights per call** and is **BW-bound**, not
dispatch-bound. A single core has low memory bandwidth, so the pooled version — even at 24/48
threads (2 CMGs) — far outpaces serial-on-one-core. This corroborates the Phase 1 reframe (decode
is BW/compute-bound, not dispatch-bound). Reverted (no production code left from Phase 2).

## Conclusion
- **Keep Phase 1** (`DS4F_FLAGBAR`): bit-identical, +1.5%, broad. Ready to commit if wanted; needs
  no 11n re-validation (byte-identical).
- The decode bottleneck is the **dense matvec BW/compute** (qkv+o_proj+shared+experts ≈ 95 ms),
  which is at the Q8 rate and not meaningfully dispatch- or locality- or prefetch-limited on a
  single node. Further single-node gains would need a different category (lower-precision weights,
  or the structural batched/spec-decode = MTP path), not pool/kernel micro-opts.
- Single-node micro-optimization headroom is **effectively exhausted**: the clean win was Phase 1
  (+1.5%); the coherent mHC lever was refuted.

## Artifacts
- `common/ds4f_impl.h` — Phase 1 flag-barrier (worktree, uncommitted).
- `a64fx/llm/tools/ws_poolbar_bench.c` — dispatch-overhead microbench.
