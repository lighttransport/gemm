# GLM-5.2 Q2 decode scaling to 24-48 nodes

Design analysis for single-stream decode on more than 12 A64FX nodes (e.g. 25, 36).
Grounded in the measured 12-node breakdown (GLM52_Q2_12N.md §11, ctx 640,
16.1 tok/s = ~62 ms/tok) and the AR probe (§9: bare bf16 M=1 all-reduce 87.6 us =
~29 us/round × 3 rounds; 12-rank recursive doubling = pof2 8+4).

## The 12-node token, split by how each term scales with node count N

| term | ms/tok @12n | scales with N as | why |
|---|---:|---|---|
| experts (routed) | 9.6 | **÷N** (minus skew) | EP-sharded: rank owns ~256/N experts |
| qa+kva GEMV | 7.6 | **÷N** | TP-sharded dense weights, 1/N streamed/node |
| o-proj GEMV | ~2.6 | ÷N | TP-sharded |
| qb / shared / attn / head | ~12 | ÷N (heads/FFN shard) | TP-sharded |
| **compute subtotal** | **~42** | **÷N** | |
| route all-reduce | 14.6 | **grows**: rounds(N)+skew(N) | reduce routed partial [hidden] over N |
| o-proj all-reduce | ~6 | grows | reduce attn-out [hidden] over N |
| **comm subtotal** | **~20** | **↑ with N** | ~2 AR/layer, ~158 AR/token |

The structure is the classic strong-scaling wall: **compute shrinks ÷N, comm
grows**. Two forces inside comm:
- **rounds(N)**: recursive-doubling all-reduce is `ceil(log2 N)` rounds × ~29 us.
  12→4 rounds, 24→~5, 32→5, 36→~6, 48→~6. Non-power-of-2 adds an awkward remainder
  round (25 is the worst nearby count; 32/48 the cleanest; 36 = 32+4 is tolerable).
- **skew(N)**: ~half of the 14.6 ms route AR is *arrival skew* from expert-count
  imbalance — top-8 of 256 over N ranks leaves some ranks with 0 owned-hit experts,
  and the AR waits for the slowest arrival. Skew **worsens** with N (higher variance
  in hits/rank). The 2-way expert shard already halved this.

## First-order projection (flat recursive-doubling AR)

`t(N) ≈ 42·(12/N) + 20·(rounds(N)/4)·skew(N)` ms

| N | compute | comm (rough) | total | tok/s | note |
|---:|---:|---:|---:|---:|---|
| 12 | 42 | 20 | 62 | 16.1 | measured baseline |
| 24 | 21 | ~26 | ~47 | ~21 | pof2-ish (16+8) |
| 25 | 20 | ~30 | ~50 | ~20 | **worst nearby**: prime, bad AR remainder |
| 32 | 16 | ~27 | ~43 | ~23 | clean pof2 |
| 36 | 14 | ~30 | ~44 | ~23 | 32+4; 6×6 factorable |
| 48 | 10.5 | ~33 | ~43 | ~23 | 32+16; skew high |

**Conclusion:** flat scaling lifts decode ~16→~23 tok/s and then **plateaus** —
comm becomes 65-75% of the token. Node *count* alone cannot push past ~23 tok/s;
the all-reduce is the wall. And **which** count matters: prefer 32/48/36 over 25.

## Levers that break the comm wall (the real work for 25/36+)

1. **Hierarchical / torus-aware all-reduce (biggest lever).** The flat
   recursive-doubling ignores the Tofu 6D torus. Factor N = A×B and do a two-level
   AR (reduce within A along one torus axis, then across B along another): rounds
   drop from `log2 N` to `log2 A + log2 B` with each level on a *contiguous* torus
   ring (lower per-hop latency). This is exactly what rescues non-pof2 counts —
   36 = 6×6 becomes two 6-node ring reduces, far better than a flat 36-rank
   recursive-double. **This is where 25/36 should invest**, and it reuses the
   uTofu AR machinery already in `tp_allreduce`.
2. **Cut the AR skew (~7 ms/tok today).** Push the expert shard from 2-way to
   4-way (or a hit-balanced static placement) so every rank owns a similar number
   of *activated* experts per layer; the route AR then waits less. Measured:
   removing idle ranks cut route-AR attribution 14.6→7.1 ms (§ "8-way slice"), so
   this is a several-ms lever — but must avoid the 256-wide-slice overhead that
   sank the naive 8-way attempt.
3. **Comm/compute overlap becomes profitable at scale.** `--overlap`
   (GLM5_COMM_OVERLAP, dedicated comm-driver thread) was ~neutral at 12n because
   comm was only 32%. At 36-48n comm is 65-75%, so overlapping the route/o AR with
   the next stage's TP-sharded GEMV can hide most of the growth. Re-measure with
   overlap ON as the default above ~24 nodes.
4. **Fewer / cheaper reductions.** Fuse the o-proj AR and route AR where the
   dataflow allows, or drop the o-proj AR by not TP-sharding o (keep o replicated,
   trading a little dense streaming for one fewer AR/layer) — favorable once
   dense is cheap (÷N) but AR is dear.

## Recommended configurations

- **Node count:** prefer **32 or 48** (clean pof2 / nice factors). Of the two the
  user named, **36 (=6×6) over 25** — 25 is prime-adjacent and gets the worst AR.
- **Topology mapping:** EP=TP=N over a factorable torus shape; wire a **2-level
  torus AR** for N≥24 (lever 1). Keep the 2-way expert shard (→4-way at N≥32).
- **Runtime flags:** `--overlap` ON for N≥24; hardware barrier (FLIB_BARRIER=HARD)
  as today; `--active-experts 3` still available for a further ~1.25× on top.
- **Expected:** ~20-23 tok/s at 32-48 nodes with flat AR; **>25 tok/s only with
  the hierarchical torus AR + skew reduction** (levers 1-2), which turn the comm
  term from `O(log N)+skew` into `O(log A + log B)` on contiguous rings.

## Hierarchical AR prototype (implemented)

Lever 1 is prototyped in `a64fx/utofu-tests/tp_allreduce.h` (additive, the flat AR
is untouched and back-compatible):
- `tp_comm_init_ex(..., stag)` — the existing init, now parametrized by steering
  tag so two sub-comms can register disjoint regions (`TP_AR_STAG` / `TP_AR_STAG2`).
  `tp_comm_init(...)` is a thin wrapper (stag = TP_AR_STAG) — no caller changes.
- `tp_comm_init_2d(row, col, vcq, peer_vcq, my_rank, N, A, ...)` — factors N = A×B
  (rank r → group g=r/B, pos b=r%B) and builds two sub-comms: **row** over the B
  contiguous ranks of my group, **col** over my A group-siblings at stride B.
- `tp_allreduce_sum_2d(row, col, buf, count)` = `sum(row)` then `sum(col)`: reduce
  within the group (each rank gets S_g), then across groups (Σ_g S_g = global).
  Reuses the recursive-doubling / bf16 / robust machinery verbatim; bit-reproducible.

Round count is `log2 A + log2 B` (= flat's `log2 N`), so the win is per-hop
locality + smaller per-level remainders + skew localization, not fewer rounds.
Pick A to match a torus axis (36→A=6,B=6; 32→A=4,B=8).

**Benches (two ways to measure):**
- Standalone: `a64fx/utofu-tests/tp_ar_ack_test.c` `TP_AR_2D=A` mode — runs flat,
  then (same allocation, same buffers) the 2-level AR; reports both `us/reduce` +
  speedup + a correctness check (sum == N(N+1)/2 on every rank). `make -C
  a64fx/utofu-tests tp_ar_ack_test`; run under mpiexec, e.g. `TP_AR_2D=6` at N=36.
- **In-runner** (wired into `GLM5_AR_PROBE`): `--ar-probe --ar-2d A` (or env
  `GLM5_AR_PROBE=1 GLM5_AR_2D=A`) emits both `ARPROBE,sum` (flat) and
  `ARPROBE2D,sum` / `ARPROBE2D,token78` (2-level) lines at the decode-relevant
  sizes M=1,2,8 with the production robust=2/bf16=1 config, on the real EP/TP group.
  e.g. `sh run_glm52_q2_12n.sh check --no-enforce --layers 1 --ar-probe --ar-2d 6`
  (A must divide the group; the flat comm's region is freed first so the 2D row can
  reuse the tag). The `token78` line projects the comm-bound tok/s (78 ARs/token) —
  compare flat vs 2D directly.

**First measurement (12n, A=3×B=4, job 49671021, in-runner probe):** already a
measurable win despite the small scale —

| token78 comm-bound projection | flat | 2-level (A=3×B=4) | win |
|---|---:|---:|---:|
| M=1 | 4.81 ms (208 tok/s) | **4.42 ms (226)** | +8% |
| M=8 | 27.19 ms (294 tok/s) | **22.75 ms (352)** | +16% |

The 2-level AR completes deadlock-free (sentinel `glm5_ar_probe_12n=done`) and is
faster even vs. the *fastest* flat config (robust=0), because the row sub-reduce
runs over 4 contiguous ranks with a simpler schedule. The win grows with payload
(M). At 24/36/48 nodes — where flat adds rounds + skew — it should widen further;
pick A to match a Tofu axis. Status: **implemented, correct, and net-positive at
12n**; scale-out measurement pending a larger allocation.

## Validation plan (when a >12-node allocation is available)

1. AR probe at N ∈ {24,32,36,48}: measure rounds × per-round latency, flat vs
   2-level torus, to calibrate `rounds(N)` and the hierarchical win.
2. Full decode A/B: flat vs hierarchical AR; overlap off/on; 2-way vs 4-way expert
   shard. Judge by best-of-3 (run-to-run AR-wait variance is large; §6).
3. Confirm the compute÷N projection holds (experts + qa+kva ms/tok should track
   12/N) and locate the new bottleneck (expected: the AR, per above).
