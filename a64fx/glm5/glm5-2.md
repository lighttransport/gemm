# GLM-5.2 A64FX decode — current state & practical ceiling (2026-07-03)

Summary of the decode-optimization campaign after the Fugaku calibration + real-weight profile
jobs. Sources: `ar_probe_results_20260703.md` (raw), `decode_sim.py` (model), `glm-decode-jobplan.md`
(plan). Node-hours used so far: **~145 / 1000**.

## TL;DR

- **The old "0.25 tok/s @96n" baseline was wrong.** Real int8 full-model single-stream decode is
  **~1.75 tok/s/slot** (~7× higher); batched **~2.9/slot**, aggregate **~11.7 tok/s** (bd=1, 4 slots).
- **Decode is NOT dominated by the all-reduce.** The bare `tp_allreduce` is ~0.14 ms; in-decode comm
  (straggler-sync included) is **~30–37 %** of the token. **Compute is ~65 %, attention-dominated.**
- **32n ≈ 96n throughput → run short-context int8 decode at 32n** (~3× more node-efficient; comm is
  ~flat in N, so it's an efficiency win, not a latency win).
- **Practical ceiling with the compute bottleneck solved (2× compute + compute/comm overlap):
  ~4.5–5 tok/s single-stream, ~35–37 tok/s aggregate (batched+MTP).**

## How the picture changed (belief → measurement)

| claim (pre-job model) | reality (measured) |
|---|---|
| decode is ~99 % all-reduce comm | comm is ~30–37 %; **compute ~65 %** |
| per-AR ≈ 26 ms (robustness tax) | per-AR ≈ **0.14 ms** tight-loop; ~0.66 ms effective in-decode (straggler-sync) |
| lean-AR (robust=2) & 2→1-AR are the big levers | **near-worthless** (comm already cheap; cost is straggler-sync, not completion path) |
| single-stream ≈ 0.25 tok/s | **~1.75 tok/s/slot** (0.25 was stale/contention-bound) |
| more nodes → faster decode | 32n ≈ 96n; **fewer nodes = same speed, better efficiency** |

## Real int8 decode profile (full 78L, short ctx, 8 slots / 4 active)

| N | bd=0 tok/s/slot | bd=1 (batched) | batched win | comm % |
|---|---|---|---|---|
| 32n | 1.72 | 2.94 | 1.71× | 37 % |
| 96n | 1.79 | 2.90 | 1.62× | 32 % |

Compute breakdown (bd=0 ms/tok, 32n → 96n):

| stage | 32n | 96n | note |
|---|---|---|---|
| **attn** | 26.4 | **48.2** | **top target — MLA does NOT scale with N** |
| qkv_proj | 20.2 | 26.8 | |
| shared expert | 20.0 | 20.9 | |
| router | 14.2 | 15.1 | |
| o_proj | 9.7 | 10.7 | |
| experts | 9.1 | 3.8 | scales well with EP |
| dense/head/other | ~5 | ~4 | |

## Practical ceiling — "compute bottleneck solved"

Model: `T = comm + compute`; lever A halves compute (2× efficiency), lever B overlaps comm & compute
(comm thread already exists via `GLM5_COMM_OVERLAP`, was OFF in the profile). `η` = overlap efficiency.

| metric | now | +2× compute | +overlap (η=1) | net |
|---|---|---|---|---|
| per-slot single-stream | 1.75 | ~2.6 | **~4.7–5.3 tok/s** | ~2.8× |
| aggregate bd=0 (4 slots) | ~7 | ~10.5 | **~19–21 tok/s** | ~2.8× |
| aggregate batched bd=1 | ~11.7 | ~18 | **~33–35 tok/s** | ~3× |
| **best stack** (32n, batched + 2× + overlap + MTP×1.19) | 11.8 | 21 | **~41 tok/s** | ~3.5× |

Why overlap is the multiplier: comm (~⅓) and compute (~⅔) run **serial** today. Halving compute makes
them comparable, so overlap nearly hides one behind the other — worth ~2× on top of the ~1.46× from
compute alone. Realistic `η ≈ 0.7–0.8` → ~10–15 % below the ideal, i.e. **~35–37 tok/s aggregate,
~4.5 tok/s single-stream**. Beyond that decode is comm-bound again → batching to higher M / fewer ranks.

## Levers, ranked (data-driven)

> **UPDATE (kernel campaign, `glm5-2-kernel-results.md`):** the "2× compute" lever is OPTIMISTIC for int8.
> The attention dot/axpy were optimized (4-acc dot + unrolled axpy, landed, native-validated) → **~1.24× on
> attention ≈ ~4–5 % end-to-end**. The bigger compute term, the **int8 GEMV, is near-optimal already**
> (convert-throughput-bound; native-confirmed a multi-accumulator variant is *slower*). So kernels give a
> modest e2e win; **batching (1.6–1.7×) is now the ONE dominant working lever** (overlap is broken, below).
> A true 2× compute needs a format change (w8a8 int8·int8 SDOT / bf16 BFDOT), not a kernel tweak.

> **UPDATE (K2 overlap A/B, job 49420218, 48n):** compute/comm **overlap is BROKEN in the decode path.**
> Enabling `GLM5_COMM_OVERLAP=1` + `GLM5_TP_SHARED=0` (required for it to engage) → **0.12 tok/s, ~56×
> SLOWER** (comm 18 s→1321 s): the comm-driver thread + 48 OMP threads oversubscribe 48 cores and the
> spin-waiting AR starves. Replicating the shared expert alone (no overlap) is already 14 % slower (real
> cost, no benefit). **The ceiling's ~2× overlap factor is NOT achievable as implemented** — it would need
> the comm thread pinned to a reserved core (OMP=47 + affinity), an untried fix. Ceiling revised DOWN.

1. **Batched decode** — measured 1.6–1.7×; **raise the M≤4 cap** (8 slots but only 4 batched). The top (only) working throughput lever.
2. **Serve int8 short-ctx at 32n** — free 3× node-efficiency.
3. **MTP** — ×1.19 on top (α≈0.4 assumed; real α still to be measured with the layer-78 loader).
4. **Attention decode kernel** — dot4+axpy4 landed (~1.24× on attn ≈ 4–5% e2e); int8 GEMV has no easy headroom.
5. ~~Compute/comm overlap~~ — **BROKEN in decode** (K2: 56× slower, thread oversubscription); needs a pinned comm core to even try.
6. ~~lean-AR (robust=2) / 2→1 AR~~ — deprioritized (comm cost is straggler-sync, not the AR primitive).

### Revised realistic ceiling (overlap OUT, batching PLATEAUS) — K1–K3 measured
- **K2:** overlap is BROKEN in decode (56× slower, thread oversubscription).
- **K3:** batching = **1.67× at M=8** (bd1 6.97 vs bd0 4.17 agg, comm 26→17%), but it amortizes only the
  ~⅓ **comm** — NOT the int8 **convert-bound compute** (K0b), which scales with M. So the aggregate
  **plateaus around the comm fraction (~1.4–1.7×); M=8 already captures most of it.** (The sim's M=64→15.8
  projection assumed comm-bound and is optimistic.) M=16/32 unmeasured — OOM at 32n from per-slot runtime
  clones; needs 48/96n.
- **Kernels:** attn dot4+axpy4 ≈ 4–5% e2e; int8 GEMV near-optimal.

**Realistic ceiling ≈ ~10–12 tok/s aggregate** (batching plateau × MTP 1.19 × attn 1.05 from ~7 baseline) —
NOT 35, and below the earlier ~14–15 (which over-credited batching scaling). **The binding constraint is the
int8 w8a16 convert-bound compute**; the only lever that breaks it is a **format change** (w8a8 int8·int8 SDOT
= activation quantization, or a bf16 BFDOT path). Comm levers (lean-AR, overlap, batching>M8) are exhausted.

## Job artifacts

- Calibration ladder: `pjsub_glm5_ar_probe_{8,16,32,96}n.sh` (fixed: topo retry + `GLM5_PREFILL_GROUPS=1`).
- Real-weight int8 decode profile: `pjsub_glm5_cbatch_int8_{32,48,96}n.sh` (bd=0 vs bd=1 A/B, short ctx).
- Model: `decode_sim.py` (`REAL_DECODE`, `REAL_DECODE_COMPUTE_MS`, MEASURED CALIBRATION + REAL PROFILE sections).

## Next steps

- **48n profile** (job 49419851, in flight) — completes the node-count curve.
- **bf16 track** — separate ≥96n decode-profile jobs (bf16 doesn't fit <96n); compare vs int8.
- **Attention decode-kernel** optimization + enable `GLM5_COMM_OVERLAP` in the cbatch decode path → re-profile.
- **Batched M cap** fix + batched bit-identity fix, then a higher-M batched A/B.
- **MTP** real-weight α (needs `model.layers.78.*` loader).
