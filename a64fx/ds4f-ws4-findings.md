# WS4 (EP all-reduce comm) — diagnosis findings, 2026-06-11

R&D on a 2–4 node subset of the live 12-node alloc (`PJM_JOBID=49204629`, shape 2×3×2),
excluding the relative-(0,0,0) login/claude node. **No production code change is warranted** —
the workstream's premise was wrong and the real cost is structural. Left on the worktree for the
main session; **not committed**.

## TL;DR

- `a64fx/utofu-tests/tp_allreduce.h` is **already a recursive-doubling (Rabenseifner) tree**, not a
  ring. WS4's stated task ("replace the ring with a tree") was already done. Standalone floor:
  **7µs @N=2, 14µs @N=4, ~23µs @N=12** (the last from `utofu-tests/summary.md` #3).
- The DS4F decode "~300µs/reduce comm" is **straggler-sync-bound**: the per-layer MoE-combine
  all-reduce is a hard EP barrier, and `ep_ar_callback`'s timer counts the spin-wait for the
  **slowest rank's expert compute** as "comm". Measured skew slope **b = 0.999–1.000** (reduce time
  tracks max per-rank delay 1:1). Robust-mode overhead is **0.3µs**; cold-cache penalty is **~0**.
- Independently corroborated by the prior `TP_AR_BF16` refutation (−2.5%, not the −35% a
  payload-bound reduce would show) **and** by a fresh synthetic N=4 runner: decode comm 6937µs/tok
  over ~6 reduces ≈ **1ms/reduce, ~70× the 14µs floor**, on balanced synthetic load — i.e. the
  barrier absorbs per-rank compute-phase jitter (o_proj alone is 5.5ms/26.8% and varies per rank).
- **Therefore no `tp_allreduce.h` header lever (robust cadence, prefetch, topology, bf16 payload)
  can recover it.** The fix is structural and already in flight in another stream: batched/spec
  decode (MTP) amortizes the per-barrier straggler wait over K tokens.

## Microbench: `a64fx/utofu-tests/tp_ar_diag_bench.c`

New pure-uTofu bench (bootstrap from `pp_handoff_bench.c`, reduces via `tp_allreduce.h`). Decomposes
the in-loop reduce into warm-floor / cold-cache / robust / straggler-skew. Build + run:

```
make -C a64fx/utofu-tests tp_ar_diag_bench CC=fcc
cd a64fx/utofu-tests && ./run_tp_ar_diag.sh            # NP=4, TP_AR_ROBUST=1
NP=2 ./run_tp_ar_diag.sh ; TP_AR_ROBUST=0 ./run_tp_ar_diag.sh
```

Results (16KB f32 payload, 3000 iters):

| term                       | N=4 robust=1 | N=4 robust=0 | N=2 robust=1 |
|----------------------------|-------------:|-------------:|-------------:|
| warm FLOOR (µs/reduce)     |        14.0  |        13.8  |         7.0  |
| cold-cache penalty (µs)    |        −3.9  |        −3.0  |        +0.2  |
| skew slope b (reduce/skew) |       0.999  |       0.999  |       1.000  |

Skew sweep (N=4, rank3 slow by S µs, rank0 measures its reduce):

```
  skew_us   reduce_us   over_floor
        0       13.85      -0.19
       50       63.61     +49.58
      100      113.64     +99.60
      200      213.59    +199.55
      400      413.61    +399.58     <- reduce = floor + skew, exactly
```

Reading: **robust overhead = 14.0 − 13.8 = 0.3µs** (the per-spin `dc civac`+`dsb sy`+MRQ-drain is
free even during a 400µs wait); **cold cache does not hurt** (a 32 MiB evict between reduces leaves
the reduce cost unchanged); **b≈1 ⇒ the entire excess over the floor is waiting for the slow rank.**

## Why the header levers are dead ends here

| candidate header lever          | measured effect            | verdict |
|---------------------------------|----------------------------|---------|
| ring → tree topology            | already a tree (floor ≤23µs)| done    |
| bf16 payload (`TP_AR_BF16`)     | −2.5% in-loop, flips argmax | refuted |
| robust cadence (thin civac/drain)| robust overhead is 0.3µs   | nothing to recover |
| prefetch / warm-keep comm region| cold-cache penalty ~0       | nothing to recover |
| multi-TNI striping / hop order  | dead ends (summary.md #2,#5)| n/a     |

The floor (7–23µs) is already optimal and is <8% of the in-loop 300µs. The other ~92% is the
straggler barrier, which lives in the compute schedule, not the comm primitive.

## The actual lever (structural — out of scope for the comm header)

1. **Batched / speculative decode (MTP).** One reduce/layer carries K tokens-worth of work, so the
   per-barrier straggler wait is amortized 1/K. Already landed in the MTP stream (M2b batched verify,
   8.69→10.49 tok/s; GEMM-decode cut comm 13.1→2.3 ms/tok by exactly this mechanism). **This is the
   WS4 fix; it just isn't a `tp_allreduce.h` change.**
2. **Lower per-rank compute variance** at the EP barrier (so the slowest rank is closer to the mean).
   Indirect; the dense-phase parallelization levers (o_proj fuse, qnr/tb2rope par) already help by
   shrinking and balancing the per-layer phases.
3. Across-step async comm overlap (summary.md #8 / `ring_attn_async.c`) is **not** applicable to M=1
   decode: layer L+1 is data-dependent on layer L's reduced hidden, so a fast rank has nothing to
   overlap the wait with. It only helps when compute≈comm and there is independent next-step work.

## Multi-node smoke (Step 3) — WS1/WS2 confirmed at N=4

Synthetic `ds4f_ep_runner -np 4`, all lockstep (perf=4/4), NaNs=0, A/B **bit-identical**:

| config                          | ‖x‖        | argmax | NaNs |
|---------------------------------|------------|--------|------|
| plain baseline                  | 1.115e+11  | 0      | 0    |
| `DS4F_MV_FUSE=1` (WS2)          | 1.115e+11  | 0      | 0    |
| MHC baseline                    | 3.673e+01  | 1      | 0    |
| `DS4F_HC_PAR=1` (WS1)           | 3.673e+01  | 1      | 0    |
| `DS4F_HC_PAR=1 DS4F_MV_FUSE=1`  | 3.673e+01  | 1      | 0    |

WS1 (`HC_PAR`) and WS2 (`MV_FUSE`), single-node bit-exact in their commits, are now also confirmed
lockstep + bit-identical at multi-node. No comm-path interaction.

## Files (worktree only, not committed)

- `a64fx/utofu-tests/tp_ar_diag_bench.c` — new diagnostic microbench
- `a64fx/utofu-tests/run_tp_ar_diag.sh` — NP-subset launcher (excludes the claude node)
- `a64fx/utofu-tests/Makefile` — added the `tp_ar_diag_bench` target
- `a64fx/ds4f-ws4-findings.md` — this file
- **No change to `tp_allreduce.h`** — by design; the data says there is nothing to fix there.

## Recommendation for the main session

Mark WS4 (as written) **closed / not-actionable at the comm layer**. Reallocate its expected +12%
to the MTP/spec-decode stream, which is the mechanism that actually removes the per-token EP
barriers. Update `a64fx/ds4f-opt.md` WS4 to reflect that the reduce is already a tree and the cost
is straggler-sync, so future readers don't re-attempt the ring→tree port.
