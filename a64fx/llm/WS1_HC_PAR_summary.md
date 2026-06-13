# WS1 — parallelize mHC collapse/expand (`DS4F_HC_PAR`) — single-node result

**Status:** implemented + single-node validated. **Not committed** (per `ds4f-opt.md` rule 4) —
hand to the main session for the 11n real-weight token-identical A/B.

## What changed
`common/ds4f_impl.h` only. The mHC hyper-connection wrap (`ds4f_hc_pre` / `ds4f_hc_post` /
`ds4f_hc_head_p`, called 86×/token) ran its per-`d` collapse/expand loops serially on tid0
between pool dispatches (WS1 in `a64fx/ds4f-opt.md`). Added behind **`DS4F_HC_PAR=1` (default off)**:
- flag `ds4f_hc_par` + `ds4f_hc_par_on()` (lazy `getenv`, mirrors `ds4f_qnr_par`);
- `ds4f_hccol_worker` — collapse `y[d]=Σ_k pre[k]·x4[k*C+d]`, even C-split (used by `hc_pre`, `hc_head_p`);
- `ds4f_hcpost_worker` — expand `x4[k,d]=post[k]·f[d]+Σ_j comb[j,k]·resid[j,d]`, even C-split (used by `hc_post`);
- each function dispatches its loop via `ds4f_pool_run(m->pool, ...)` when the flag is on, else the
  unchanged serial loop.

**Bit-exact by construction:** each output element is an independent reduction over `k` in the
identical fixed order — only *which thread* computes *which d* changes. The RMS reduction (would
reassociate the `double` sum), the already-threaded f32 matvec, sinkhorn, and the 64 KB
`memcpy(s_resid,s_x4)` are deliberately left serial/untouched. No call-site or signature changes;
`ds4f_mhc_test.c` unchanged.

## Validation (this host, native `fcc`, no mpiexec)

| Gate | Result |
|------|--------|
| `ds4f_mhc_test` `mhc_c.txt` **HC_PAR=0 vs =1** | **BYTE-IDENTICAL** (bit-exact) |
| `ds4f_mhc_test` vs `mhc_py.txt` | max-abs 1e-7 (= `%.7e` text floor, unchanged) |
| `ds4f_exact_test` (HC_PAR=1) | max-abs **5e-8** ≤ 5e-8 ✓ |
| `ds4f_tierb2_test` (HC_PAR=1) | max-abs **2e-6** ≤ 2e-6 ✓ |
| `ds4f_gemm_test` | **ALL OK (0 failures) = 205/205** ✓ |

(exact/tierb2/gemm aren't on the mHC path, so HC_PAR can't shift them — they confirm no collateral
breakage and sit at their standard baselines.)

## Perf — single-node synthetic decode (rc=0, NaN-free)
`OMP_NUM_THREADS=48 taskset -c 12-59 DS4F_MHC=1 DS4F_TIERB2=1 DS4F_PROF=1 DS4F_MAXGEN=32 ./build/ds4f_runner`

| | decode ms/tok | tok/s | `other` (hc_pre proxy) |
|---|---|---|---|
| HC_PAR=0 (baseline) | ~167–178 | 5.6–6.0 | ~38–41 ms |
| HC_PAR=1 (parallel) | ~136–149 | 6.7–7.3 | ~21–26 ms |
| **delta** | **−~30 ms (≈ +22% tok/s)** | | **−~18 ms (−45%)** |

Stable across 3 reps each. The total decode drop (~30 ms) exceeds the `other`-phase drop because
`hc_post` (parallelized too) lands in the `o_proj`/`experts`-adjacent phases, not in `other`.

The residual `other` (~22 ms) is now the **serial RMS reduction + the under-utilized 24-row f32
matvec** (rowsplit8 → only 3 of 48 threads active) + the 64 KB memcpy — not the collapse loops.
Pushing it toward the doc's ~5 ms target would need either reassociating the RMS (breaks bit-exact)
or improving that tiny matvec's thread utilization (column-split reassociates) — both are WS2-class,
out of WS1's bit-exact scope.

## Next step (main session, 11n alloc)
Real-weight gen A/B `DS4F_HC_PAR` 0 vs 1 must be **TOKEN-IDENTICAL** (bit-exact change), NaN=0,
lockstep — the standard final rung. The real-gen mHC fraction is larger than synthetic (doc:
`other` 50.2 ms real vs 13.5 ms no-mHC), so the real-gen win should be ≥ the synthetic +22%.
