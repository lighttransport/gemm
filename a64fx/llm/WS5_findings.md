# WS5 — quick wins / tb2prep micro: analyzed, no clean single-node lever (diminishing returns confirmed)

**Status:** investigated single-node; **no production change** — the doc's "diminishing returns"
call is confirmed by measurement. No code committed.

## WS5a — TP_HEAD in plain decode
Per `ds4f-opt.md`: "already implemented + bit-exact-validated at 11n; just ensure the champion
wrappers can enable it; nothing to R&D." This is **wrapper wiring** (`run_ds4f_gen_11n.sh` /
`run_ds4f_longctx_11n.sh`), a multi-node concern, and it **breaks the MTP draft** (accepts=0).
Not single-node R&D — out of scope here.

## WS5b — tb2prep micro
Single-node decode profile (`DS4F_TIERB2=1 DS4F_PROF=1`), tb2prep ≈ 13–18 ms/tok, sub-timers:

| sub-timer | ms | state |
|-----------|----|-------|
| tb2qproj | ~5.5 | **already POOLED** M=1 bf16 matvec (`ds4f_bf16mv_worker`), per-layer |
| tb2lcmp | ~5.2 | **already POOLED** M=1 bf16 matvec (`ds4f_cmpmv_bf16_worker`, computes kv+gate in one dispatch), per-layer |
| tb2scan | ~0.5 (grows w/ ctx) | **already POOLED** int8 resident scan (`ds4f_idxsc8r4_worker`), BW-bound per `tools/idxscan_bw_probe.c` |
| tb2icmp | ~1.3 | pooled indexer compress matvec |
| tb2rope | ~0.2 | already parallelized (`DS4F_TB2ROPE_PAR`, 2k) |
| tb2wproj | ~0.2 | pooled |
| tb2topk | ~0.01 | already optimized (heap-merge, longctx work) |

**Conclusion:** the two dominant sub-timers (qproj, lcmp ≈ 10.7 ms) are **already pooled M=1
bf16 matvecs** — the same BW-bound + per-layer-dispatch regime WS2 addressed for the main dense
path. They are single `pool_run`s each (lcmp already fuses kv+gate), and the indexer's matvecs are
data-dependent (compress → q-proj → rope → scan), so the WS2 multi-matvec fusion doesn't apply.

The only **serial-on-tid0** remnant is the compress-block softmax (`ds4f_compress_step`,
`ds4f_impl.h:1499–1525`): d=512 independent softmaxes over P=8 values. It runs only on `should`
tokens (1-in-`ratio`=1-in-4) and costs <0.5 ms/tok amortized. Parallelizing it WS1-style would add
**43 pool dispatches per `should` token** (one per layer) — the dispatch overhead would likely
match or exceed the saved work (net-neutral to negative). SVE-vectorizing it needs a polynomial
`expf` (argmax-flip risk). Neither is a clean win.

**No high-value single-node lever remains in tb2prep** — consistent with the doc's "diminishing
returns; only attack after WS1/WS2 land." The remaining tb2 cost is pooled BW-bound M=1 matvec
work; the structural lever for that is fewer tokens-worth of decode (batched/spec decode = the MTP
workstream), not a tb2prep micro-opt.

## Net single-node WS campaign
- WS1 (`DS4F_HC_PAR`): mHC parallelization, +22% decode — **committed** (ef0642e)
- WS2 (`DS4F_MV_FUSE`): matvec dispatch fusion, +7% Q8 decode — **committed** (81575de)
- WS6: Q8 fp16-scale overflow NaN fix — **committed** (b17018b)
- WS3: outer-product GEMM port — **refuted** at scale (2× single-thread → 1.0× at 48 threads), documented
- WS5: **diminishing returns confirmed** (this doc)
- WS4 (tree all-reduce): **needs the 2–4 node alloc** — only remaining workstream, not single-node
