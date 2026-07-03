# Calibrating the local decode simulators from real A64FX runs

The decode optimization loop runs LOCALLY (decode_sim.py + the qlair A64FX simulator with
`--ranks 8`) to save node-hours. This doc maps the outputs of the calibration job
(`pjsub_glm5_ar_probe_8n.sh`, 8 nodes x ~25 min) onto the simulator knobs, so one cheap job
re-anchors both models.

## Submit + collect

```
ssh fugaku 'cd ~/work/gemm/glm5-1 && pjsub a64fx/glm5/pjsub_glm5_ar_probe_8n.sh'
ssh fugaku 'grep -hE "ARPROBE|KERNBENCH|SELFCHECK|cbatch:|TOKEN DIFF" \
    ~/work/gemm/glm5-1/a64fx/glm5/ar_probe_run_<jobid>/*rank00.txt \
    ~/work/gemm/glm5-1/a64fx/glm5/pjsub_glm5_ar_probe_8n.sh.<jobid>.out'
```

Repeat at other node counts by copying the script and changing `node=/proc=` (16/32 give the
log2(N) round-scaling decode_sim assumes; 96 anchors the production point).

## Phase 1 — ARPROBE → decode_sim.py comm term

`ARPROBE,sum,N=8,robust=1,bf16=0,M=1,bytes=24576,us_per_ar=...,round_us=...`

- `us_per_ar` at robust=1, M=1 is THE production decode AR cost. Feed it:
  `decode_sim.recalibrate(ar_ms=us_per_ar/1000, n_ref=8)` — or, preferred once several N are
  measured, fit `ar = init + rounds*round_ms` across N ∈ {8,16,32,96} and call
  `recalibrate(round_ms=..., init_ms=...)`.
- robust=2 rows quantify the LEAN path win (`TP_AR_ROBUST=2`, the m1_tput(round_ms) lever) —
  this is the number that decides whether bf16 M=1 ≥ 2 tok/s is reachable.
- robust=0 rows bound the tax floor (passive spin; unsafe at scale, reference only).
- The M sweep gives the marginal per-byte cost → check `UTOFU_BW` (1/8e9 s/B assumed).
- `ARPROBE,token78,...,ms_per_token_comm=X` is directly `1000/X` comm-bound tok/s for a
  78-AR decode token — compare with `m1_tput()` before trusting any lever stack.
- `argmax` vs `argmax_n32`: per-stream vs batched head merge — validates cutting M head
  ARs to 1 in the batched head.

## Phase 1 → qlair Tofu model

qlair charges `1400ns + bytes/6.3GB/s` per put and has no multi-node topology. Compare
`ARPROBE` per-AR against the same tp_ar_8rank run under `qlair --ranks 8`: the DIFFERENCE is
the real-cluster term (incast, OS jitter, robust drain) qlair cannot see.

**MEASURED (2026-07-03, `qlair --native --ranks {2,4,8}` × `TP_AR_ROBUST {0,1,2}`, CNT=6144 fp32),
now baked into `decode_sim.py` (`QLAIR_WIRE_*`, `cluster_multiplier()`):**

| N | rounds=⌈log₂N⌉ | robust0 | robust1 | robust2 | r2/r1 | r1/r0 |
|---|---|---|---|---|---|---|
| 2 | 1 | 151 µs | 170 µs | 155 µs | 0.91 | 1.13 |
| 4 | 2 | 209 µs | 245 µs | 214 µs | 0.87 | 1.17 |
| 8 | 3 | (timeout) | 306 µs | 277 µs | 0.91 | — |

- **Wire floor fit (robust=1): `init 0.105 ms + rounds×0.0677 ms`, latency LINEAR in ⌈log₂N⌉** —
  confirms decode_sim's round-scaling assumption *with data*. Extrapolated wire AR: 96n≈0.58 ms.
- **Cluster multiplier @96n = real 26 ms / wire 0.58 ms ≈ 45×** — the qlair-invisible incast +
  OS-jitter + robust-drain-under-contention term. This is the residual the ARPROBE ladder anchors.
- **Because the wire *shape* is now measured locally, the job ladder only needs ~3 N-points
  (8/32/96) to fit the cluster multiplier's N-trend — not the dense 8/16/32/96 sweep.** ← node-hour cut.
- **robust=2 (lean AR) is bit-exact and ≤ robust=1 in the wire model (r2/r1≈0.89).** Its REAL-cluster
  win magnitude is the ONE number qlair cannot give (it amortizes MRQ drain that only exists under
  real contention) → the robust1-vs-robust2 A/B is the top must-run job.
- robust=0 at R=8 times out under qlair load (passive spin pathological) — reference only, matches
  the "unsafe at scale" note.

## Phase 2 — KERNBENCH → decode_sim BW_NODE + qlair kernel check

`KERNBENCH,shape,rows,cols,M,ms_per_call,GBs,GBs_per_stream`

- decode_sim `BW_NODE` (currently 300e9) := the M=1 `GBs` averaged over the per-token shape
  mix (attention + shared + router + owned experts + head). Update the constant, rerun
  `python3 decode_sim.py`, and the bandwidth-bound ceiling + all predictions re-anchor.
- The M sweep is the batched-decode compute model: `sum(shape ms at M)/M` = compute
  ms/token/stream. If GBs saturates before M=32, batching stops amortizing weight reads
  there — cap the M recommendation accordingly.
- qlair cross-check: run the same binary shapes under `qlair -p` (cycle mode) on the dev
  host and compare GBs — target ≤10% error (QLAIR_VERIFICATION.md methodology).

## Phase 3 — SELFCHECK

`BATCH_SELFCHECK ... MATCH` on real SVE hardware with real 8-rank comm = the batched-MLA
kernel is production-safe at M=1 (the qlair TEST2 result already covers M>1 bit-identity;
a real-weight M>1 A/B at 96n stays on the job-validation list).

## Phase 4 — cbatch A/B → end-to-end batched-decode win

- `TOKEN DIFF` lines must be absent (bd=0 and bd=1 emit identical streams).
- `cbatch: service decode ... agg tok/s` ratio bd=1/bd=0 at 8n, 8 slots, 8-layer synthetic
  ≈ the comm-amortization factor. decode_sim comparison: scale `n_allreduce` to the
  synthetic layer count (8 MoE-ish ARs/token) and N=8 — predicted ratio ≈
  `pred_tok_s(8, M=8) / (8 * pred_tok_s(8, 1))` with the ARPROBE-recalibrated constants.

## MEASURED A64FX kernel perf (fapp + wall-clock, 2026-07-03, node c33-7214c)

Actual on-hardware numbers for the decode matvec, prefill GEMM, and comm kernels — the ground truth
the `decode_sim.py` (python) and qlair (cycle) sims should reproduce. Tool: `a64fx/glm5/glm5_kern_prof.c`
(fapp `fapp_start/stop` regions) profiled via `fapp -C -Hevent=statistics` → `fapp -A -ttext` (GFLOPS,
FP-peak%, mem GB/s, mem-peak% per region); comm from the ARPROBE campaign. Run at 48t, correct NUMA
(`XOS_MMM_L_PAGING_POLICY=demand:demand:demand OMP_PROC_BIND=close OMP_PLACES=cores`).

### Compute kernels (rows=8192 cols=6144 gs=128; wall-clock Gop/s is authoritative)
| kernel | wall Gop/s | fapp FP-peak% | fapp mem GB/s | mem-peak% |
|---|---|---|---|---|
| decode w8a16 matvec (M=1) | 449 | 19.5% | 5.5 | 0.54% |
| decode int16 SDOT (M=1)   | 431 | 0.97% | 5.2 | 0.51% |
| prefill w8a16 GEMM (M=64) | 1095 | **26.2%** | 0.16 | 0.015% |
| prefill int16 SDOT (M=64) | 2234 | 4.6% | 0.46 | 0.04% |
| prefill int8-rb SDOT (M=64)| 2005 | 8.4% | 0.41 | 0.04% |

**Two interpretation traps the sims must respect:**
1. **fapp "GFLOPS / FP-peak%" counts FLOATING-POINT ops only.** The int16/int8 SDOT kernels use
   `svdot` on the INTEGER pipe → they show ~1–8% FP-peak despite being 2× the throughput of the FP
   w8a16 path. So the FP-peak% ranks kernels by *FP content*, not speed — use the **wall-clock Gop/s**
   for the SDOT kernels, and the FP-peak% only for the FP (w8a16 / bf16) path. The w8a16 prefill GEMM
   hits **26% of the f32 FMA peak** (the FP-GEMM ceiling on A64FX for this shape); the int SDOT kernels
   move the same work onto the integer `svdot` pipe (int16 svdot_s64 = 2× f32 density, int8 svdot_s32 =
   4×) → ~2× (int16) / ~2–4× (int8-rb) the w8a16 Gop/s. This is why decode's int16 SDOT is a *loss*
   (M=1 memory-bound, no compute advantage to exploit) but prefill's is a 2–4× *win* (M≥8 compute-bound).
2. **This microbench is L2-RESIDENT, not HBM-streaming** (mem-peak% 0.01–0.5%): each thread re-reads its
   own ~1 MB weight slice across reps → cache-resident. So these Gop/s are the **compute ceiling**, NOT
   the real serving rate. The REAL decode streams the whole model (≫ cache) from HBM → it is
   **BW-bound**: measured node read BW = **100 GB/s single-CMG (default prepage) → 739 GB/s NUMA-local**
   (numactl --interleave=all + pin). Prefill reuses weights M× → less BW-bound (int16 GEMM NUMA
   crossover: 1.26× @M=8 → 1.01× @M=64). ⇒ `decode_sim.BW_NODE` should use the **739 GB/s NUMA-local**
   figure (not 300e9); the decode compute term is BW-bound, the prefill term compute-bound.

### Prefill PCHUNK — the hidden lever for the EXPERTS stage (int16)
Experts run a per-expert GEMM over g=tokens-routed; PCHUNK=64 gives g~2-4 (<5) → each falls to the M=1
matvec remainder path (no int16 win). Larger chunk → larger g → the register-blocked 5-token svdot
kernel. Measured 12L int16 prefill (256-tok prompt): experts ms/tok 1.099(PC64)→0.705(PC128)→0.380(PC256);
int16 prefill tok/s 187→218→230; experts int16 speedup 1.50×(PC64)→1.99×(PC128). PC192 DIPS (ragged
192+64 split — avoid non-divisor chunks). ⇒ use PCHUNK≥128 (ideally ~full prompt) with int16; the
committed 1.30× e2e @PC64 under-represents the win (small-chunk artifact starving experts).

### Batched-decode GEMM (M=8) — int16 unlocks the decode compute headroom
Batched decode (bd=1) runs projections through glm5_gemm at M=batch → the compute-bound regime. fapp
(glm5_kern_prof M=8, 48t): M=1 matvec int16 450 < w8a16 515 Gop/s (0.87×, LOSS) BUT M=8 GEMM int16 1324
vs w8a16 428 (3.10×) / int8-rb 1888 (4.41×). REAL e2e batched decode (12L, 12n, cbatch bd=1, 8 slots):
w8a16 63.95 → int16 **98.89 agg tok/s = 1.55×**. ⇒ int16 GEMM is a DECODE-SERVING lever when batched
(stacks on batching), NOT just prefill — the resolution of the "int16 lost at M=1 decode" result.
SLOTS sweep (int16, PCHUNK=128): agg tok/s 104.6(8)→129.4(16)→151.0(32); w8a16 63.95(8)→111.5(32);
int16 ratio SHRINKS with batch 1.55×(8)→1.35×(32) (larger M amortizes w8a16's dequant tile, same as the
kernel M-sweep) → int16 best at MODEST batch. PCHUNK does NOT apply to batched decode (decode M=slots).

### Comm kernel (tp_allreduce, hidden=6144 f32 = 24 KB, from ARPROBE)
- **Bare tp_allreduce (tight loop): ~0.08 ms @12n, ~0.14 ms @96n** (NOT the fictional 26 ms the pre-job
  sim back-solved — ~180× cheaper). Raw ARPROBE @N=2 M=1 bytes=24576: `us_per_ar=51.9`.
- **Effective in-decode: ~0.66 ms @12n** (~8× the tight-loop) — the gap is **straggler-sync** (ranks
  reach the AR at different times after uneven expert compute), NOT the AR primitive. Comm is
  **~30–37% of the decode token**; compute ~65% (attention-dominated). `decode_sim`: UTOFU_ROUND ≈ the
  0.08–0.14 ms bare AR; multiply by a ~5–8× straggler factor for the effective in-decode comm term
  (lean-AR / robust=2 is worthless — the cost is straggler-sync, not the completion path).

### qlair (cycle sim) calibration notes
- qlair cycle-mode is **FP-latency accurate** (the dot4/axpy story) but the fapp data here shows the
  hot kernels are now **integer `svdot`** (decode/prefill SDOT) and **HBM-BW-bound** (real decode) —
  regimes qlair models poorly (memory model misses store-buffer/BW effects; see K0 calibration where
  qlair mis-ranked the memory-bound axpy 1.03× vs native 1.93×). ⇒ **anchor qlair only for the FP
  compute-bound kernels; use these native fapp Gop/s + the 739 GB/s BW for the SDOT/decode paths.**

## After calibration

Update `decode_sim.py` constants (BW_NODE := **739e9** NUMA-local, UTOFU_ROUND := ~0.1e-3 bare AR ×
straggler factor via `recalibrate()` defaults or inline), re-run the report, and refresh the lever stack
in BATCHED_DECODE_PLAN.md / the job-validation list with the re-anchored predictions.
