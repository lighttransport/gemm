# WS2 — dense matvec dispatch fusion (`DS4F_MV_FUSE`) — single-node result

**Status:** implemented + single-node validated. **Not committed** (per `ds4f-opt.md` rule 4) —
hand to the main session for the 11n real-weight token-identical A/B.

## What changed
`common/ds4f_impl.h` only. Decode runs ~10 `ds4f_pool_run` dispatches/layer; each pays a thread
wake + a barrier (wait-for-slowest). WS2 fuses **independent** consecutive matvecs into ONE
`pool_run` so that cost is paid once instead of N times. Added behind **`DS4F_MV_FUSE=1`
(default off)**:
- `ds4f_matvec_multi(m, list, n)` + `ds4f_mv_multi_worker` — one `pool_run` that calls the existing
  `ds4f_mv_worker` for each entry with the **same `(tid,nthr)`**, so every sub-matvec uses the
  identical `rowsplit8` + kernel + per-row dot order as a standalone `ds4f_matvec`. Only the barrier
  is shared ⇒ **BIT-EXACT** (no FP reassociation). Same precedent/argument as the block-diagonal
  o-proj worker (`ds4f_mv_bd_worker`).
- flag `ds4f_mv_fuse` / `ds4f_mv_fuse_on()` (lazy `getenv`).

Fused sites in `ds4f_forward_token` (each pair reads a shared input, no inter-dependency):
| Site | Fused | Note |
|------|-------|------|
| qkv | `wq_a` + `wkv` (both read `s_hn`) | 2→1; `wq_b` depends on `wq_a`, stays separate. `s_kvlat` computed early — independent of the q-path, bit-exact. |
| shared expert | `sh_w1` + `sh_w3` (both read `s_h2`) | 2→1 |
| routed experts | `ex_w1[slot]` + `ex_w3[slot]` (both read `s_h2`) | 2→1 **per expert** (~8 experts/layer = the biggest dispatch-count saving) |

`wo_a`/`wo_b` (o-proj) and `wq_b` are left alone — `wo_b` depends on `wo_a`, `wq_b` on `wq_a`
(true data deps, not fuseable).

## Validation (this A64FX node, native `fcc`, no mpiexec)

| Gate | Result |
|------|--------|
| **forward A/B** `DS4F_DEBUG=1` per-layer `\|\|.\|\|` trace (43 layers × all tensors × tokens, 3618 lines), **MV_FUSE 0 vs 1** | **BYTE-IDENTICAL** + last-hidden `\|\|x\|\|` identical, NaNs=0 |
| `ds4f_exact_test` (MV_FUSE=1) | 5e-8 ≤ 5e-8 ✓ (isolated kernels untouched) |
| `ds4f_tierb2_test` (MV_FUSE=1) | 2e-6 ≤ 2e-6 ✓ |
| `ds4f_gemm_test` | ALL OK = 205/205 ✓ |

The 3618-line byte-identical forward trace is the real bit-exact gate (exact/tierb2 don't drive
`forward_token`). Combined with the by-construction argument, fusion is bit-exact.

## Perf — single-node synthetic decode, **Q8 dense** (the production decode config; FP8 dense is
dequant-bound so dispatch overhead is a small share there — measure on Q8)
`OMP_NUM_THREADS=48 taskset -c 12-59 DS4F_Q8_DENSE=1 DS4F_TIERB2=1 DS4F_PROF=1 DS4F_MAXGEN=32 ./build/ds4f_runner`

| | decode ms/tok (5 reps, sorted) | median |
|---|---|---|
| MV_FUSE=0 | 121.1 125.1 128.1 128.9 132.9 | **128.1** |
| MV_FUSE=1 | 116.9 117.2 119.5 120.3 124.9 | **119.5** |

**−6.7% ms/tok (+7.2% tok/s), bit-exact.** Fusion also tightens the distribution (117–125 vs
121–133) — fewer barriers = less exposure to per-dispatch stragglers. Dense-phase breakdown
(`qkv_proj+o_proj+shared+experts`) ~104 → ~96 ms median.

## Honest assessment vs the +15–18% target
Dispatch fusion alone yields ~7% on single-node Q8 because the **two largest matvec phases are
un-fuseable**: `o_proj` (~36 ms, dominated by `wo_b` which depends on `wo_a`) and `wq_b`
(32768×1024, depends on `wq_a`). The remaining roofline gap (247→610 GB/s in isolation) is then
**in-kernel BW**, not dispatch — the doc's next WS2 sub-lever is **SVE software prefetch (`svprfd`)
in the Q8/bf16/FP8 inner kernels** (`ggml_dequant.h`: `matvec_sdot_8row`,
`matvec_bf16_8row_pv`, `matvec_fp8e4m3_8row`), which is semantically neutral (bit-exact) and targets
the dominant un-fuseable phases directly. Not done here.

**Expect a larger win at 11n than this single-node number:** under TP the dense rows are
sharded across nodes, so each matvec has *fewer rows to amortize the wake/barrier over* ⇒ dispatch
overhead is a bigger fraction ⇒ fusion's relative benefit grows (the doc's +15–18% is an 11n
estimate).

## Next step (main session, 11n alloc)
Real-weight gen A/B `DS4F_MV_FUSE` 0 vs 1 must be **TOKEN-IDENTICAL** (bit-exact change), NaN=0,
lockstep. Then optionally add the `svprfd` prefetch sub-lever and re-measure the o_proj/wq_b phases.
Composes cleanly with WS1 (`DS4F_HC_PAR`) — disjoint functions, both default-off flags.
