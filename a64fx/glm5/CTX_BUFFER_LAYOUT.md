# GLM-5.2 Q2 — 256K+ context-aware buffer layout design

## Motivation

Three distinct 256K failures this session, all rooted in buffers whose size does
not track context length correctly:

1. **SIGABRT (heap overflow)** — `glm5_msa_select` wrote `sel[nblk+nsel]` into a
   `maxsel`-wide `ms->psel` slice; `nblk` grows with ctx (2050 @256K) and overran
   the fixed slice. *(fixed: offset only when `sel` aliases the bitmap)*
2. **SIGABRT (heap overflow)** — `glm5_msa_prefill_select` strode `ms->sc` by
   `max_pos` while `ms->sc` is allocated at stride `n_heads*maxsel`. *(fixed:
   stride by `nblkmax`)*
3. **SIGKILL (OOM)** — the auto Tier-A budget grew the *replicated* bf16 KV to
   ~2.85 GB without reserving the prefill scratch / transition / OS overhead, and
   dense prefill exceeded the 31 GB node. *(fixed: explicit scratch reserve)*

Common cause: buffers are sized by whatever quantity was locally convenient, not
by a principled ctx-aware rule. As ctx → 256K+, quantities that were "small"
(`nblk`, `max_pos`) grow and collide with fixed-size neighbors. The point fixes
plug the specific leaks; this design removes the class of bug.

## Buffer taxonomy by ctx-scaling

Every per-token / per-layer buffer, classified by how its *correct* size scales:

| class | correct size scales as | examples | 256K footprint |
|---|---|---|---|
| **A** — O(1) in ctx | fixed model dims | `q,k,v,qlat,kvb,attn,router,shg/shu` | fine |
| **B** — O(maxsel) | `topk·block` (ctx-INDEPENDENT) | `psel` (selected positions), sparse attn scores | 2304, fine |
| **C** — O(nblk) | `ctx/block` | `pbit` (block bitmap), block scores | 2050 @256K, tiny |
| **D** — O(ctx), replicated | `ctx` | Tier-A KV, `rope`, dense attn scores | **huge — must be capped** |
| **E** — O(ctx/ep), sharded | `ctx/ep_size` | Tier-B CP KV | ~1/12, fits |

The three bugs were all **class confusion**: a class-C quantity (`nblk`) leaking
into a class-B allocation (`psel`); a class-D stride (`max_pos`) applied to a
non-D buffer (`sc`); and a class-D pool (replicated Tier-A KV) sized without
budgeting its class-A/B/C scratch neighbors.

## Design rule

**No class-D buffer may be sized by the full logical context.** Split the single
`max_pos` into two explicit caps:

- **`T_ctx`** — total logical context (256K, 1M). Only class-E (sharded KV) and
  class-B/C (intrinsically bounded) buffers may scale with it.
- **`T_dense`** — max positions attended *densely* (the Tier-A window). Chosen so
  BOTH the replicated KV (`T_dense · per_pos`) AND the dense-score storage
  (`n_heads · T_dense · pchunk · 4`) fit the memory budget. Every class-D buffer
  sizes by `T_dense`, never `T_ctx`.

Invariant: `Tier A→B transition fires at T_dense`, so class-D buffers provably
never exceed their cap. Class-B/C are bounded by construction. Only class-E grows
with `T_ctx`, and it is sharded by `ep_size`, so per-node it stays ~`T_ctx/ep`.

Today `ms->sc` (dense-score path), the single-token dense score buffer, and
`rope_cos/sin` are all sized by `max_pos = T_ctx` — the design moves the first two
to `T_dense` and acknowledges rope as a bounded-but-replicated exception.

## Concrete layout changes

1. **Split the overloaded `ms->sc`** into three correctly-strided buffers so no
   consumer reinterprets another's stride (the source of overflow #2):
   - `sc_block[pchunk · nblkmax]`  (class C) — MSA block scores.
   - `sc_attn [pchunk · maxsel]`   (class B) — sparse attention scores.
   - `sc_dense[slots · T_dense · n_heads]` (class D, **capped**) — dense scores,
     allocated only for the Tier-A / batch-decode path, sized by `T_dense`.

2. **`psel`**: either keep the gather's scratch-offset scheme and size the slice
   `nblkmax + maxsel`, or (as done) drop the offset for the non-aliased prefill
   path so the slice stays `maxsel`. Add a check-build bound assert
   `nblk + nsel <= slice_width`.

3. **KV tiering, sized by class:**
   - Tier-A: `T_dense` slots, replicated bf16 (class D, capped).
   - Tier-B: `ceil(nblk/ep)·block` slots, CP-sharded (class E), int4 default or
     bf16 (`--kv-tier-bf16`). This carries the 256K length.

4. **Budget derives `T_dense` from BOTH constraints** (currently only the KV
   term): `T_dense = budget / (per_pos + n_heads·pchunk·4)`, after subtracting
   weights + rope + prefill scratch + OS. *(the scratch reserve is implemented;
   the joint dense-score term is the remaining step)*

5. **rope** (`max_pos · rotary_dim/2 · 8` bytes): 65 MB @256K, 256 MB @1M —
   replicated but tolerable. Shard or compute-on-the-fly only if ≥1M becomes a
   working target.

## Implementation phases

1. **DONE** — point fixes: psel gather offset, `sc` block-stride, bf16 Tier-B,
   budget scratch reserve. 256K prefill completes NaNs=0.
2. **DONE** — `T_dense` is now a first-class field (`glm5.h`), set in
   `glm5_kv_init` in both branches (= `T_cp` when tiered, else `max_pos`), and
   documents the class-D invariant. Bound **guard** added to the live
   `glm5_msa_select` (the actual overflow site): it takes a `sel_cap` argument
   (`max_pos` for the decode `s_blk_sel` alias, `ms->maxsel` for the prefill
   `psel` slice) and `abort()`s with a clear message instead of corrupting the
   heap. *Verified: decode 16.52 tok/s NaNs=0 (no false-abort), 256K prefill
   262144 tok 318 tok/s NaNs=0.* (Discovery along the way: the batched
   `glm5_msa_prefill_select` is dead code — the live prefill path calls
   `glm5_msa_select`, so that is the one that matters.)
3. **DONE (revised)** — The audit found `ms->sc` is NOT three live uses: its only
   live consumer is `glm5_forward_batch_decode` (per-stream multi-serving, dense
   attention scores, class D, stride `max_pos`); the block-score consumer was the
   **dead** `glm5_msa_prefill_select`, and the single-stream long-context path
   never touches it. So the "split" reduces to **allocate `ms->sc` (now documented
   as `sc_dense`) ONLY for `per_stream_kv`** — freeing ~`N·n_heads·maxsel·4`
   (~300 MB at pchunk 512) in exactly the memory-tight single-stream long-context
   path, adding OOM headroom. Batch decode is byte-identical (same `max_pos`
   stride). *Verified: decode 16.45 tok/s NaNs=0, 256K prefill 262144 tok
   313.7 tok/s NaNs=0.*
4. **DONE (revised)** — Post-Phase-3 the single-stream tiered path has NO
   dense-score buffer, so the budget constraint is KV + fixed mstream scratch (not
   a joint dense-score solve). The auto budget now reserves a **computed,
   pchunk-adaptive** scratch (`per_tok · pchunk · 4`, `per_tok` summed from the
   actual dominant mstream buffer dims: logits/kvb/q/attn/k/v/qlat/hidden-pool/
   moe+dense inter/psel/piq/router) plus a 12.5% proportional OS/page-cache margin
   — replacing the ad-hoc 2 GB floor. This drops the full-78-layer auto `T_dense`
   from the OOM'd 26112 (2.85 GB Tier-A) to ~11520 (~1.26 GB), leaving ~3.9 GB for
   OS overhead. *Verified: full-78 AUTO-tier prefill loads + prefills NaNs=0, NO
   SIGKILL (job 49671021); decode regression 16.25 tok/s NaNs=0.*
5. **Optional** — shard/stream rope for ≥1M.

## Status

Phases 1–4 landed and verified. The 256K+ buffer layout is now context-aware:
class-D buffers are bounded by `T_dense` (never `max_pos`), the one live dense
score buffer (`sc_dense`) is allocated only where used (per-stream serving), the
MSA gather is bound-guarded, and the auto Tier-A budget reserves the coexisting
scratch + OS overhead so the full-model auto path no longer OOMs. Remaining work
is the optional rope sharding for ≥1M and, for long-context *coherence* (a
separate axis — see output-code.md), more nodes for a larger exact dense window.

## Verification

- Bound asserts enabled in the check build; 256K + 1M synth prefill → no overflow.
- Auto `codegen` (no forced transition) → no OOM (validates the budget derivation
  of `T_dense`).
- Decode / 2048-prefill benchmark **unchanged**: with `T_dense ≥ 2304` the
  Tier-A-only path is bit-identical to today (verified this session: decode
  16.28–16.38 tok/s, NaNs=0, after all buffer edits).
