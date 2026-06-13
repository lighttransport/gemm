# DS4F optimization R&D on 1–4 nodes — workstreams + agent resume prompts

Companion to `a64fx/ds4f.md` ("Decode roofline revisit 2026-06-10", lines ~874+). That analysis
decomposed the perf gaps; **this doc turns each gap into an independent workstream that iterates
on 1–4 nodes** (NOT the scarce 11/12-node alloc) with exact repro commands and a resume prompt per
workstream, so multiple coding agents can work in parallel. Final integration (real-weight 11n
token-identical A/B + tok/s) is the ONLY step that needs the big alloc — hand that back to the
main session.

## The numbers being attacked (from the roofline revisit)

DECODE (M=1, 11n, Q8 dense, @ctx10240 = 77.9 ms/tok; real-gen mHC path = 105.8 ms/tok):
- matvecs 29.6 ms at **247 GB/s = 31% of 800 GB/s** (µbench proves 85% per-matvec) → WS2
- **mHC hc_pre/hc_post ~37 ms/tok, 35% of the real-gen wall** (newly found) → WS1
- comm 13.5 ms = 43 all-reduces × ~300 µs (utofu tree bench = 23.5 µs) → WS4 ❌ CLOSED: the reduce
  is already a tree and the 300 µs is straggler-sync (EP barrier waiting on the slowest rank's
  expert compute), NOT comm — unfixable at the comm layer; lever is structural (MTP/spec). See WS4.
- attn 15.0 + tb2prep 17.4 = per-position floor (already 6.6×/19× optimized; low priority)

PREFILL (batched GEMM, compute-bound): 56.3 tok/s @batch32 = **~13% of the 450 tok/s FMA
ceiling**; the 8x3-pv GEMM kernel runs at 7–13% of peak vs the 89% proven in
`a64fx/doc/FP16_GEMM_CEILING.md` → WS3.

## Ground rules (every agent)

1. **Compilers**: native `fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast` (this host IS
   an A64FX node — run single-node binaries directly, no pjsub). Build via
   `make -C a64fx/llm <target> CC=fcc OPENMP=1`.
2. **Every change behind its own env flag, default OFF** (`DS4F_<NAME>=1` to enable). This is how
   parallel streams merge without conflicts and how A/Bs stay honest.
3. **Validation ladder** (cheap → expensive; never skip a rung):
   a. single-node Python-ref tests: `OMP_NUM_THREADS=12 taskset -c 12-23 ./build/ds4f_exact_test`
      then max-abs vs ref: `paste tools/exact_py.txt exact_c.txt | awk '{d=$2-$4;a=d<0?-d:d;if(a>m)m=a}END{print m}'`
      — gate **exact ≤ 5e-8**; same for `ds4f_tierb2_test` gate **≤ 2e-6**; `ds4f_mhc_test` for HC
      paths; `ds4f_gemm_test` must stay **205/205**.
   b. single-node perf bench (per workstream below) — report before/after numbers.
   c. FINAL (main session only): 11n real-weight gen A/B — bit-exact/token-identical for
      disjoint-output changes, "coherent + lockstep + NaN=0" for reassociating changes (the
      Step-2r standard).
4. **Do not commit** — leave changes on the worktree + a summary; the main session integrates.
5. Multi-node (2–4) when needed: synthetic weights (`DS4F_REAL=0`) run at ANY node count
   (experts shard e%N). Real weights at N=4 are *plausible* (est. RSS ≈ 22.5 GB/node: experts
   64×~13 MB vs 24× at /11) but unproven — validate against **MemFree, never RSS**
   (see CLAUDE.md), and don't fight it: synthetic + lockstep is the multi-node R&D vehicle.
6. Profile vocabulary: `DS4F_PROF=1` prints per-phase ms; top-level phases are indices 0..8
   (`qkv_proj attn o_proj shared router experts head other tb2prep`); the `tb2*`/`qkv_*` lines
   are SUB-timers (do not double-count). `o_proj` has an underscore.

---

## WS1 — parallelize mHC hc_pre/hc_post (decode real-gen)  ✅ LANDED (2026-06-11)

**RESULT (11n real-weight gen A/B, MAX_NEW=64, the quicksort prompt):** `DS4F_HC_PAR=1` is
**TOKEN-IDENTICAL** to off (64/64 ids, identical completion, NaNs=0, lockstep) and lifts real-gen
decode **9.46 → 11.80 tok/s (+24.7%)**; the `other`/mHC phase drops **51.1 → 30.5 ms** (−40% of the
phase). Below the optimistic 37→5 ms target — HC_PAR parallelized the collapse/expand fn-matvecs but
`other` is still 30.5 ms. **Landed**: `DS4F_HC_PAR` now defaults ON in the perf wrappers
(`run_ds4f_gen_11n.sh`, `run_ds4f_longctx_11n.sh`) and OFF in the base `run_ds4f_11n.sh` (clean
reference) — mirrors the `Q8_DENSE` pattern. Code was already committed (ef0642e); the impl below
is unchanged.

**WS1b LANDED (2026-06-11): fold the mHC RMS into the mixes-matvec dispatch.** Measured the residual
30.5 ms `other`: RMS sum-of-squares = 89 µs/call = **35% of `other`**, a serial latency-bound double
reduction on tid0 (matvec 111 µs and collapse 61 µs are already-parallel and irreducible). `DS4F_HC_RMSPAR=1`
runs a fused worker that computes the IDENTICAL F32 matvec rows AND a per-thread partial sum-of-squares
of x4 over a disjoint slice; the caller combines partials in fixed tid order. Because `ss` is a **double**
accumulation, the parallel-vs-sequential reassociation (~1e-13) is below the `float rsq` epsilon ⇒
**BIT-IDENTICAL** result (mhc_test on-vs-off = 0.0 even at 48 threads; exact 5e-8 unchanged). 11n real-gen
A/B **TOKEN-IDENTICAL** (64/64, NaNs=0, lockstep): `other` 30.5 → 24.1 ms, decode **11.80 → 12.77 tok/s
(+8.2%)**. Cumulative WS1+WS1b vs pre-mHC-par: 9.46 → **12.77 tok/s (+35%)**, `other` 51.1 → 24.1 ms.
Default ON in the perf wrappers, OFF in base. (A dead-end checked first: serializing the tiny mixes matvec
— `DS4F_HC_MVSER` — was a **10× regression**; the 24-row matvec is latency-bound and *needs* the pool.)
Remaining `other` (24.1 ms) is now the two irreducible parallel dispatches (matvec+collapse) per hc_pre.



**Current**: ~37 ms/tok ("other" 50.2 gen vs 13.5 synthetic) = 86 calls/tok (2 per layer × 43) of
`ds4f_hc_pre` / `ds4f_hc_post` (+`ds4f_hc_head` once): f32 fn-matvec `[hc=4, 16384]`, sinkhorn
weights, collapse/expand loops over `[4×4096]`, plus `memcpy(s_resid, s_x4, 64KB)` — all serial
on the calling thread, BETWEEN pool dispatches. Same pattern as the validated 2j/2k wins
(`ds4f_q_norm_rope_par` 15×, `DS4F_TB2ROPE_PAR` 19× — copy their structure).
**Target**: ~5 ms/tok. **Files**: `common/ds4f_impl.h` — `ds4f_hc_pre`, `ds4f_hc_post`,
`ds4f_hc_head_p` (~line 3930–4000); flag `DS4F_HC_PAR` (default off).

Repro (1 node):
```
make -C a64fx/llm ds4f_mhc_test ds4f_exact_test ds4f_tierb2_test ds4f_runner CC=fcc OPENMP=1
cd a64fx/llm
python3 tools/ds4f_mhc_ref.py                     # writes mhc_py reference
OMP_NUM_THREADS=12 taskset -c 12-23 ./build/ds4f_mhc_test    # correctness gate
# perf: single-node synthetic decode w/ mHC on, profile "other":
OMP_NUM_THREADS=48 DS4F_MHC=1 DS4F_TIERB2=1 DS4F_PROF=1 DS4F_MAXGEN=32 ./build/ds4f_runner
```
Gates: mhc_test bit-exact serial-vs-par (disjoint per-stream/per-dim splits ⇒ must be
BIT-EXACT, not just close); exact/tierb2 unchanged; single-node "other" phase before/after.

**Resume prompt**: "In /vol0006/mdt0/data/hp250467/work/gemm/ds4f, the mHC hyper-connection wrap
(ds4f_hc_pre/ds4f_hc_post in common/ds4f_impl.h, called 86×/token in ds4f_forward_token) costs
~37 ms/tok of serial scalar work between pool dispatches — 35% of real-gen decode. Parallelize it
across the existing thread pool (ds4f_pool_run, see ds4f_q_norm_rope_par for the validated
pattern: split disjoint output ranges, bit-exact). Gate behind DS4F_HC_PAR=1 default off. Validate
bit-exact via build/ds4f_mhc_test + ds4f_exact_test 5e-8 + ds4f_tierb2_test 2e-6, then measure the
'other' phase with DS4F_PROF=1 on the single-node ds4f_runner (DS4F_MHC=1 DS4F_TIERB2=1). Target
37→5 ms. Single A64FX node, no mpiexec. Don't commit; report before/after phase ms."

## WS2 — dense matvec issue efficiency  ⏹ CLOSED / no clean win at production scale (2026-06-11)

**FINISH (2026-06-11):** `DS4F_MV_FUSE` (the landed dispatch-fusion, commit 81575de) was validated
on 11n real-gen for the first time: **TOKEN-IDENTICAL** but **+0.86% only** (12.77 → 12.88 tok/s,
phases move both directions) — i.e. **NEUTRAL within noise**. The premise below ("the in-loop matvec
loss is pool-dispatch/serial-gap overhead") does NOT hold at production scale: real-gen dense is **Q8
(int8 svdot), which is issue/dequant-bound, not BW- or dispatch-bound** (`ds4f_decode_bw` confirms
fp8/Q8 decode is ~10% of the 723 GB/s ceiling — a different regime from the bf16 80%-of-BW the "247
vs 610" framing assumed). The matvec phases (qkv 7 + o_proj 9 + shared 6 + experts 5 + head 2 = 29 ms)
are near the Q8 kernel's throughput, and fusing 2 dispatches/layer barely registers. `matvec_sdot_8row`
already runs 8 independent row-accumulators and is svdot-throughput-bound, so the remaining idea
(SVE prefetch / more accumulators) would help latency, not issue throughput — uncertain micro-opt, not
a clean win. **MV_FUSE left default-OFF** (bit-exact but not worth enabling). WS2 is closed; the dense
decode matvec is issue-bound and the realistic next decode lever is structural (MTP/spec). Original
(refuted-at-scale) framing preserved below.

**Current** (premise refuted at 11n — see above): decode matvecs 29.6 ms for 7.3 GB = 247 GB/s; `ds4f_decode_bw_bench.c` proves ~85%
of the ~720 GB/s node read ceiling per matvec in isolation ⇒ the loss is in-loop: pool dispatch
overhead (~10 `ds4f_pool_run` per layer), thread ramp, inter-matvec serial gaps, CMG locality of
the Q8 528-byte blocks. **Target**: ≥440 GB/s in-loop (matvecs ≤ ~16 ms).
**Files**: `common/ds4f_impl.h` — `ds4f_matvec`, `ds4f_mv_worker`/`ds4f_mv_bd_worker`,
`ds4f_pool_run` (pool at top of file); flag `DS4F_MV_FUSE` (or similar).

Repro (1 node, NO model load needed):
```
make -C a64fx/llm ds4f_decode_bw CC=fcc OPENMP=1
cd a64fx/llm && taskset -c 12-59 ./build/ds4f_decode_bw     # per-shape/per-dtype GB/s table
# in-loop: synthetic single-node decode, watch qkv_proj/o_proj/shared phases:
OMP_NUM_THREADS=48 DS4F_TIERB2=1 DS4F_PROF=1 DS4F_MAXGEN=32 ./build/ds4f_runner
```
Ideas (in expected-value order): chain consecutive matvecs into ONE pool dispatch (qkv triple;
sh_w1+w3 pair — workers already exist, give them a task list); persistent-worker spin instead of
wake-per-dispatch; software prefetch (`svprfd`) in the Q8/bf16 inner loops; verify Q8 block
first-touch is CMG-local to the rowsplit that reads it (the 2i lesson: mismatched first-touch =
cross-CMG reads). Gates: exact/tierb2/gemm tests unchanged + bit-exact A/B of one forward
(fused vs not — same dot order ⇒ bit-exact required).

**Resume prompt**: "In /vol0006/mdt0/data/hp250467/work/gemm/ds4f, M=1 decode dense matvecs
achieve 247 GB/s in-loop vs ~610 GB/s (85% of node ceiling) in the isolated
build/ds4f_decode_bw bench — the loss is dispatch/serial-gap/locality, not the kernel. Reduce
per-layer pool dispatches (fuse the wq_a/wq_b/wkv chain and sh_w1+sh_w3 into single dispatches
with task lists), add SVE prefetch, and check Q8_PV 528B-block first-touch CMG locality (pattern:
a64fx/ds4f.md Step 2i). Flag DS4F_MV_FUSE=1 default off, keep per-tensor dot order identical
(bit-exact gate). Validate: ds4f_exact_test 5e-8, ds4f_tierb2_test 2e-6, then DS4F_PROF=1 phases
(qkv_proj+o_proj+shared+experts+head, currently ~29.6 ms/tok @48T) on single-node ds4f_runner
synthetic. Target ≤16 ms. One A64FX node. Don't commit; report the phase table before/after."

## WS3 — GEMM kernel 12×2 port (prefill: 7–13% → 20–40% of FMA peak, 3–4×)

**Current**: `ds4f_gemm_worker` (8x3-pv register blocking, K-tile 4096) + MXFP4 tile-dequant hit
~84–147 Gmac/s vs 3.07 Tmac/s fp32 peak. The repo has an 89%-of-peak blueprint:
`a64fx/doc/FP16_GEMM_CEILING.md` (12×2 blocking, 4K unroll, no-epilogue-convert).
**Target**: ≥600 Gmac/s (20%) conservatively; stretch 1 Tmac/s.
**Files**: `common/ds4f_impl.h` — `ds4f_gemm`/`ds4f_gemm_worker` (~line 558+) + the MXFP4 tile
path; flag `DS4F_GEMM12X2` (default off, fall back per-shape).

Repro (1 node):
```
make -C a64fx/llm ds4f_gemm_test ds4f_kernels_bench CC=fcc OPENMP=1
cd a64fx/llm
OMP_NUM_THREADS=48 ./build/ds4f_gemm_test          # MUST stay 205/205, prints Gmac/s per shape
taskset -c 12-59 ./build/ds4f_kernels_bench        # kernel-level Gmac/s
```
Gates: gemm_test 205/205 + relL2 thresholds unchanged + argmax-exact columns; exact/tierb2
regression. NOTE: stay fp32-accumulate (bf16 in, f32 acc) — fp16-accumulate flips argmax (the
Step-2o magic/FTZ lesson); it's a later, separately-gated lever.

**Resume prompt**: "In /vol0006/mdt0/data/hp250467/work/gemm/ds4f, the batched-prefill GEMM
(ds4f_gemm_worker in common/ds4f_impl.h, 8x3-pv) runs at 7–13% of the A64FX 3.07 Tmac/s fp32 FMA
peak; prefill is compute-bound so this is the whole prefill lever (56.3 tok/s vs 450 ceiling).
Port the register-blocking/scheduling from a64fx/doc/FP16_GEMM_CEILING.md (89%-of-peak 12×2
study) into the bf16→f32 GEMM path, keeping fp32 accumulate (fp16-acc flips argmax — do NOT).
Flag DS4F_GEMM12X2=1 default off with per-shape fallback to 8x3. Validate: build/ds4f_gemm_test
must stay 205/205 (it also prints Gmac/s per shape — that's the perf metric), plus
ds4f_exact_test 5e-8 / ds4f_tierb2_test 2e-6. Iterate kernel-only via build/ds4f_kernels_bench
pinned taskset -c 12-59. Target ≥600 Gmac/s on the [M=32, 4096×4096-class] shapes. One A64FX
node, no MPI. Don't commit; report Gmac/s per shape before/after."

## WS4 — tree/pipelined EP all-reduce  ❌ CLOSED / NOT-ACTIONABLE (2026-06-11)

**RESOLUTION (do NOT re-attempt the ring→tree port):** the premise below is wrong on two counts.
(1) `tp_allreduce.h` is **already a Rabenseifner tree**, not a ring (floor 7 µs@N=2 / 14 µs@N=4 /
23 µs@N=12). (2) The in-loop ~300 µs/reduce is **straggler-sync-bound, not comm-bound**: the
per-layer MoE-combine all-reduce is a hard EP barrier and `ep_ar_callback`'s timer counts the
spin-wait for the **slowest rank's expert compute** as "comm". Measured (2–4 node subset,
`tp_ar_diag_bench.c`): skew slope **b = 0.999–1.000** (reduce time = floor + max per-rank delay,
1:1); robust-mode overhead **0.3 µs**; cold-cache penalty **~0**. Corroborated by the prior
`TP_AR_BF16` refutation (−2.5%, not the −35% a payload-bound reduce shows) and a synthetic N=4
runner (comm ≈1 ms/reduce = ~70× the floor on balanced load — the barrier absorbs per-rank compute
jitter). ⇒ **No `tp_allreduce.h` header lever (robust cadence / prefetch / topology / bf16 payload)
can recover it; the floor is already optimal and <8% of 300 µs.** The real fix is **STRUCTURAL** and
lives in the **MTP/spec-decode stream** — batched/spec decode amortizes the per-barrier straggler
wait 1/K (exactly why M2b GEMM-decode already cut comm 13.1→2.3 ms/tok). Async overlap (summary.md
#8) is N/A to M=1 decode (layer L+1 data-depends on L's reduced hidden). Full writeup + tables:
`a64fx/ds4f-ws4-findings.md`. **No `DS4F_AR_TREE` flag was added.** The original (now-refuted)
framing is preserved below for the record.

---

**Current** (REFUTED — see resolution above): 43 sequential f32 all-reduces/token (16 KB hidden each)
at ~300 µs each through `tp_allreduce.h` (ring). `a64fx/utofu-tests/summary.md` measured
Rabenseifner-tree ≈ 23.5 µs at 11 nodes. **Target**: ≤50 µs/reduce. **Files**:
`a64fx/utofu-tests/tp_allreduce.h` (the production reduce used by ds4f_ep_runner) + benches
`reducescatter_bench.c`/`allgather_bench.c`. Flag `DS4F_AR_TREE=1` default off.

CORRECTNESS NOTE: a tree changes the f32 summation ORDER vs the ring ⇒ results are COHERENT, not
bit-identical to ring (like the Step-2r contraction shards). What MUST hold: (a) every rank
computes the IDENTICAL value (same deterministic order on all ranks ⇒ lockstep argmax holds);
(b) NaN=0; (c) fixed order across calls (no opportunistic arrival-order reduction). The earlier
`TP_AR_BF16` refutation was the *payload precision*, not the topology — f32 tree is fine.

Repro (2–4 nodes; this is the one WS needing mpiexec — use a small pjsub alloc, see CLAUDE.md):
```
pjsub -g hp250467 -L "freq=2000,eco_state=0,rscgrp=small,node=4,elapse=00:30:00" --no-check-directory <script>
# in-script: cd a64fx/utofu-tests && make CC=fcc && mpiexec -np 4 ./tofu_topo_helper && \
#   mpiexec -np 4 ./<your_tree_allreduce_bench>     # latency vs ring at 16 KB f32
# integration smoke (synthetic weights run at ANY N):
cd a64fx/llm && mpiexec -np 4 ./build/ds4f_ep_runner   # DS4F_AR_TREE=0/1: lockstep argmax must match itself across ranks
```
Gates: bench exactness (tree sum == ring sum within deterministic-order reproducibility; rank0
vs rankN identical bits); 4-node synthetic ep_runner lockstep (`argmax_distinct_across_ranks==1`)
+ NaN=0 with the flag on; latency table 16 KB/64 KB/1 MB payloads ring-vs-tree at N=2,4.

**Resume prompt**: "In /vol0006/mdt0/data/hp250467/work/gemm/ds4f, DS4F decode does 43 sequential
16 KB f32 all-reduces/token via the ring in a64fx/utofu-tests/tp_allreduce.h at ~300 µs each
(13.5 ms/tok = 17% of decode); the repo's own utofu tree benchmark measured 23.5 µs. Implement a
deterministic-order tree (Rabenseifner RS+AG or binomial) in tp_allreduce.h behind DS4F_AR_TREE=1
(default off), keeping f32 payload (bf16 payload was REFUTED — argmax flip) and a FIXED reduction
order so all ranks produce identical bits (lockstep requirement). R&D on a 4-node pjsub alloc
(rscgrp=small node=4, see repo CLAUDE.md for pjsub): microbench ring-vs-tree at 16KB/64KB/1MB,
then smoke ds4f_ep_runner -np 4 with synthetic weights (DS4F_REAL unset) checking
argmax_distinct_across_ranks==1 and NaN=0 with the flag on. Target ≤50 µs at 16 KB. Don't commit;
report the latency table + the lockstep check."

## WS5 — quick wins / wiring (low risk, fold into any stream)

- **TP_HEAD in plain decode** (−1.0 ms): already implemented + bit-exact-validated at 11n; just
  ensure the champion-config wrappers (`run_ds4f_gen_11n.sh`, `run_ds4f_longctx_11n.sh`) can
  enable it; nothing to R&D. (CAUTION: TP_HEAD currently breaks the MTP *draft* — accepts=0 in
  the M4 sweep — debug that in the MTP stream, not here.)
- **tb2prep micro** (17.4 ms: lcmp 4.6 + topk 4.2 + scan 3.6 + qproj 3.2): single-node probes
  exist (`tools/idxscan_bw_probe.c`, `tools/idxscore_probe.c`). Diminishing returns — only attack
  after WS1/WS2 land.

## Parallelization & merge notes

- WS1/WS2/WS3 all edit `common/ds4f_impl.h` but DISJOINT functions (hc_*, matvec/pool, gemm) —
  use separate worktrees/branches off `ds4f`, each behind its own default-off flag; merges are
  textual no-ops. WS4 edits only `a64fx/utofu-tests/tp_allreduce.h`.
- Expected combined effect (from the roofline): real-gen decode 105.8 → ~55 ms (~18 tok/s),
  synthetic 77.9 → ~45 ms (~22 tok/s), prefill 56 → 150–200 tok/s; structural decode ceiling
  stays ~32 tok/s (attn+tb2 per-position floor — only MTP/spec decode goes past it). **NOTE: WS4's
  +12% is removed — diagnosed as unrecoverable straggler-sync; its budget folds into MTP/spec.**
- Integration order (main session, 11n alloc): WS1 (bit-exact) → WS2 (bit-exact) → WS3
  (argmax-exact). ~~WS4~~ CLOSED (no code change). WS1 HC_PAR + WS2 MV_FUSE already confirmed
  lockstep + bit-identical on/off at N=4 synthetic (2026-06-11) ahead of the 11n A/B.

## WS6 — Q8 GEMM small-M remainder NaN (1-node bug, found 2026-06-10)

**Symptom**: `ds4f_forward_verify` / `DS4F_GEMM_DECODE` under `DS4F_Q8_DENSE=1` produces **NaN
nondeterministically** (some 11n runs fine, some all-NaN; |hc_head|=nan at the first head eval ⇒
the layer dense GEMMs NaN'd). `DS4F_Q8_DENSE=0` (bf16 dense) always works. Plain decode (matvec
Q8) is fine. Now GUARDED: the runner `die()`s if SPEC/GEMM_DECODE + Q8_DENSE (ds4f_ep_runner.c).
**Root-cause hypothesis**: the verify calls `ds4f_gemm` with **M=1–2**, so it only hits the Q8
worker's single-token remainder kernel (`matvec_sdot_8row`, `ds4f_gemm_worker` ~line 570), which
`ds4f_gemm_test` (M=32/64) barely exercises — vs the M≥3 `matvec_sdot_8row_3x` path it covers
well. Likely an uninitialized/tail read in the 1–2-token Q8 sdot as driven by the gemm (the
`__thread` xq/xs scratch is correct; NOT a realloc race — that was mis-diagnosed and reverted).

**1-node repro** (NO alloc, fast iterate):
```
make -C a64fx/llm ds4f_gemm_test CC=fcc OPENMP=1
cd a64fx/llm
# EXTEND ds4f_gemm_test.c to add M=1 and M=2 Q8_PV cases (it currently tests M=32/64) -> should
# reproduce the NaN/garbage at M=1,2; then it's a pure single-node kernel debug loop:
OMP_NUM_THREADS=48 ./build/ds4f_gemm_test            # add small-M Q8 rows; watch relL2/argmax/NaN
```
Gate: gemm_test 205/205 + the new small-M Q8 rows pass (relL2 ~1e-6, argmax-exact, NaN=0).
Then the verify works under Q8 and the guard can be relaxed (though bf16 dense stays the
recommended spec config — Q8 batched is dequant-bound, no speed win).

**Resume prompt**: "In /vol0006/mdt0/data/hp250467/work/gemm/ds4f, the Q8_PV path in
ds4f_gemm_worker (common/ds4f_impl.h ~line 550) NaNs nondeterministically for small M (1-2 tokens)
— the verify hits only the single-token remainder kernel matvec_sdot_8row, which ds4f_gemm_test
(M=32/64) doesn't cover. Add M=1 and M=2 Q8_PV cases to a64fx/llm/ds4f_gemm_test.c, reproduce on
ONE node (OMP_NUM_THREADS=48 ./build/ds4f_gemm_test, no mpiexec), and fix the kernel (suspect
uninitialized accumulator / K-tail / scale read in the 1-2-token Q8 sdot). The __thread xq/xs
scratch is correct — do NOT touch its allocation (a realloc-race 'fix' was already tried and
reverted). Gate: gemm_test 205/205 + new small-M rows relL2~1e-6 argmax-exact NaN=0. Then remove
the DS4F_SPEC/GEMM_DECODE+Q8 guard in ds4f_ep_runner.c. Don't commit; report the failing
shape + the fix."
