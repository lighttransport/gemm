# ARPROBE ladder — real Fugaku calibration (2026-07-03, gate G0)

Jobs: 49419107(8n) 49419357(16n) 49419359(32n) 49419361(96n). Synthetic 1-layer dummy,
GLM5_AR_PROBE=1, GLM5_PREFILL_GROUPS=1 (AR spans all N ranks), robust=1, M=1, [1,H=6144] fp32.

| N  | rounds | AR us (r0/r1/r2)      | round us | token78 comm ms (r1) | comm-bound tok/s |
|----|--------|----------------------|----------|----------------------|------------------|
| 8  | 3      | 73.3 / 72.4 / 73.6   | ~24      | 2.83                 | 353              |
| 16 | 4      | 86.4 / 86.2 / 86.7   | ~22      | 3.94                 | 254              |
| 32 | 5      | 107.8 / 139.6 / 170.0| ~28      | 8.14                 | 123              |
| 96 | 6      | 145.2 / 141.1 / 145.6| ~24      | 9.09                 | 110              |

## Findings (fold into decode_sim.py — DONE)
- **Real per-AR @96n = 0.141 ms, NOT the 26 ms the pre-job sim back-solved.** ~180x cheaper.
- Round cost is a flat ~24-26 us/round; robust 0/1/2 within ~30% at every N — **no 3.4 ms robustness tax**.
- Comm @96n (153 AR) ≈ 21-28 ms/token → 35-110 tok/s comm-bound ceiling.
- **The 0.25 tok/s anchor (4000 ms/token) is ~99% NON-comm.** Decode is compute/HBM/overhead/contention-
  bound, NOT comm-bound. The whole comm-lever stack (lean-AR robust=2, 2→1 AR/layer) is near-worthless.
- Batched decode measured **~2.6x @8n synthetic** (bd=1 29.4 vs bd=0 11.1 agg tok/s) — win is per-FORWARD
  overhead amortization, not AR-count. Still the top lever.
- cbatch synthetic A/B shows TOKEN DIFF (bd0 vs bd1): random-weight greedy argmax is knife-edge under
  batched-GEMM FP reduction order — NOT a correctness verdict; real-weight bit-identity is the true test.
- qlair --native over-estimated the wire ~3x (pthread-barrier sim overhead), but got the SHAPE (linear
  in ceil(log2 N)) and robust=2 bit-exactness right.

## Pivot
NEXT job (was Tier 1): a **real-weight decode PROFILE at 96n** (int8, M=1, `cbatch: … comm X%`) to get the
TRUE current decode tok/s and attribute the 99% residual. The 0.25 anchor may be stale/contention-bound —
the synthetic cbatch suggests 10-30 tok/s is achievable. Deprioritize J1(lean-AR)/J4(replicated-attn).

## CORRECTION (real-weight cbatch, job 49419532, bf16 12L 12n) — walks back the "99% non-comm" overclaim

The ARPROBE tight loop measured the bare collective in isolation and UNDER-measures the in-decode AR.
Real-weight decode: **6.22 tok/s/slot (24.9 agg), comm 41.5%**, 3224 AR calls / 5.1 s → **~0.66 ms
EFFECTIVE per-AR at 12n = ~8× the tight-loop probe** (~0.08 ms @12n). The 8× gap is
straggler-synchronization (ranks reach the AR at different times after uneven expert compute) +
cache-cold / MRQ-under-load interleave — NOT the robust completion path.

**Reconciled picture (both anchors):**
- The fictional 26 ms/AR robustness tax is dead (bare AR ~0.14 ms @96n). ✔ (holds)
- BUT decode is **~40% comm at 12n** (rising with N), NOT ~0.5%. The earlier "99% non-comm" was wrong —
  it trusted the tight-loop probe alone.
- Corrected levers: **BATCHING** (amortize AR over M — measured 2.6× @8n synth) and **FEWER RANKS**
  (smaller AR + less straggler spread) are the comm levers; **lean-AR (robust=2) is still ~worthless**
  (the cost is straggler-sync, not completion). The other ~60% is compute/HBM.
- Real-weight bit-identity: CBATCH_IDS show a repetition-collapse (req=2 → 103216×) on the 12-LAYER
  PARTIAL model — expected (partial model = garbage logits), not a decode bug.

decode_sim.py updated: EFFECTIVE_AR_FACTOR=8.0, two-anchor MEASURED CALIBRATION section.

## J0 — REAL-WEIGHT INT8 FULL-MODEL DECODE PROFILE (jobs 49419683/684, 32n/96n, ~53 nh)

Full 78L int8, 8 slots, short ctx (maxpos 2048, max_new 24), bd=0 (per-slot) vs bd=1 (batched, M≤4).

| N   | bd=0 agg / slot | bd=1 agg / slot | batched win | comm% (bd0→bd1) |
|-----|-----------------|-----------------|-------------|-----------------|
| 32n | 6.90 / 1.72     | 11.77 / 2.94    | 1.71×       | 37.0 → 32.5     |
| 96n | 7.18 / 1.79     | 11.62 / 2.90    | 1.62×       | 31.8 → 29.7     |

### Headline findings
1. **The 0.25 tok/s anchor is DEAD.** Real int8 single-stream decode = **~1.75 tok/s/slot** (bd=0),
   **~2.9/slot batched** — ~7–11× the stale anchor. The campaign baseline was wrong.
2. **32n ≈ 96n throughput** → **32n is ~3× more node-efficient** (same tok/s at 1/3 the nodes). For
   short-context int8 decode serving, run at **32n**. (Validates the 32/48n instinct — efficiency, not speed.)
3. **comm ~30–37%** — confirms the corrected read (decode is ~1/3 comm, NOT 0.5% and NOT 99%). 96n comm%
   is slightly LOWER than 32n → "fewer ranks = less comm" is FALSE; comm is ~flat. 32n wins on efficiency only.
4. **Batched decode = 1.6–1.7× real** (M≤4; AR calls 14536→3634). But bd=1 is NOT bit-identical across all
   slots: req=0 matches bd=0 exactly, reqs 1–2 diverge (FP reduction-order argmax-tie flips) → batched path
   needs a bit-identity fix before production, though the per-forward win is real.
5. **Compute breakdown (~65% non-comm), ms/tok bd=0:**
   - attn: 26 ms (32n) → **48 ms (96n)** — attention MLA does NOT scale with node count; TOP compute target.
   - qkv 20→27, shared-expert 20→21, router 14→15, o_proj 10→11, experts 9→4 (experts scale well w/ EP).

### Next levers (data-driven)
- **Compute** now dominates (~65%): attn (biggest, non-scaling) + qkv + shared. Kernel-level optimization.
- **Batching** to higher M (fix the M≤4 cap; slots=8 but only 4 batched — scheduler/active-slot limit).
- **Serve at 32n** for efficiency. bf16 (≥96n) is the other precision track (separate jobs).
