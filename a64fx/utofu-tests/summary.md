# uTofu ring-attention decode-cost study — summary & findings

A64FX / Fugaku, measured 2026-05-21 on a real **12-node 2×3×2 in-unit torus**
allocation (all nodes share x,y,z; differ in the a,b,c in-unit axes).

This directory holds an **MPI-free uTofu** micro-benchmark suite that models the
two communication/memory costs of *distributed ring-attention DECODE* on Fugaku
**without doing any GEMM or attention math** — it isolates the interconnect and
memory-bandwidth behaviour so the GEMM kernels can be tuned against a realistic
distributed-attention budget.

## What's here

| file | role |
|------|------|
| `tofu_demo.h` | shared constants/structs; the whole design hinges on every node running the identical binary with the same conventional uTofu indices (TNI=0, CMP=0, CQ=0, STAG=1) |
| `tofu_topo_helper.c` | MPI + uTofu helper, run **once** per allocation: writes `tofu_topo.txt` (rank → 6D coords) so the pure-uTofu app needs no runtime VCQ/STADD handshake |
| `tofu_put_demo.c` | minimal MPI-free Put + verify demo (ring exchange, validates the construct-VCQ-from-coords convention) |
| `ring_attn_bench.c` | **the main deliverable**: ring-attention decode cost model + measured uTofu comm paths + scaling projection |
| `tni_stripe_bench.c` | multi-TNI striping microbench: 2-rank ping-pong, stripes one hop's payload across K TNIs to the *same* peer, sweep K via `TS_NTNI` |
| `ring_attn_overlap.c` | **comm/compute OVERLAP validation with REAL attention math** (the one bench that does GEMM-like work): real flash-decode (bf16 KV, SVE softmax) + real online-softmax ring-reduce + a head-group pipeline; tests whether comm actually hides under compute |
| `ring_attn_async.c` | **across-step async comm driver + A64FX HW barrier** (libhwb): overlaps the *full* online-softmax all-reduce of step k−1 (thread 0) with step-k compute (threads 1..47), double-buffered, 2 barriers/step; compares SERIAL / ASYNC-across-step / head-group PIPE under a flat OpenMP barrier vs the EL0 hardware barrier (`TF_HW_BARRIER`). Uses the **production qpacked+ktbl QK kernel** (svtbl K-broadcast, K_DP `[p][d][kv]`, no svaddv) — see finding #9 |
| `moe_dispatch_bench.c` | **MoE expert-dispatch all-to-all** (the *other* half of decode comm): top-8-of-256 routing scattered to expert-owning ranks then gathered back. Naive single-TNI / multi-TNI distinct-destination / tree-all-reduce baseline / dimension-ordered store-and-forward relay (variant `d`, all 6 Tofu axes `x→y→z→a→b→c`, `MOE_DIMORDER`) / Bruck logarithmic-index all-to-all (variant `e`, `MOE_BRUCK`) over a deterministic shared routing matrix — see findings #10 (multi-TNI wins), #11 (dimensional relay dead end in-unit; 6-axis multi-unit harness ready), #15 (Bruck dead end) |
| `allgather_bench.c` | **all-gather** (the BW-optimal *half* of an all-reduce; TP sequence-parallel hidden / KV-shard gather): naive single-TNI all-broadcast / multi-TNI all-broadcast / ring all-gather / recursive-doubling all-gather (Rabenseifner non-pow2), plus the tree all-reduce baseline for the roofline ratio. `AG_BATCH`/`AG_SHARD`/`AG_NTNI` — see finding #12 |
| `reducescatter_bench.c` | **reduce-scatter** (the *other* BW-optimal half; the all-reduce = reduce-scatter + all-gather decomposition): naive single-TNI direct scatter / multi-TNI direct scatter / ring reduce-scatter, plus a multi-TNI all-gather and the fused tree all-reduce so the decomposed pair is compared to the fused tree in one run. Shares `AG_*` knobs with `allgather_bench` — see finding #13 |
| `pp_handoff_bench.c` | **pipeline-parallel handoff** (the point-to-point inter-stage send/recv — the comm pattern that *replaces* per-layer collectives): ping-pong one-hop (near vs far for distance sensitivity) / full forward+backward store-and-forward chain `0→…→S-1→…→0` / tree all-reduce baseline (the TP per-layer cost it trades against). `PP_HID`/`PP_ABYTES`/`PP_BATCH`/`PP_FAR` — see finding #14 |
| `p2p_latbw_bench.c` | **point-to-point latency/bandwidth characterization** of the raw Tofu links (the floor every estimate above builds on): per-peer Put ping-pong / one-sided Get / `armw8` fetch-add latency reported vs computed torus hop-distance, plus a Put-injection & Get-delivered bandwidth size-sweep with a pipelined async window. The only program here that exercises `utofu_get`/`utofu_armw8` (the rest are Put + memory-sentinel). `P2P_MAXBYTES`/`P2P_WINDOW`/`P2P_LAT_BYTES`/`P2P_BW_PEER` — see finding #17 |
| `Makefile`, `run_bench.sh`, `run_demo.sh`, `run_p2p.sh` | build + launch |

The whole approach: `utofu_construct_vcq_id(peer_coords, tni, cq, cmp, &vcq)`
rebuilds a peer's VCQ ID purely from its 6D Tofu coordinates, and
`utofu_query_stadd(peer_vcq, STAG, &base)` resolves the remote buffer address by
the STAG convention — **zero runtime exchange**, no MPI in the app itself
(`mpiexec` only places the identical binary on each node).

## How to run

```sh
# 1. (re)discover topology for the CURRENT allocation — ALWAYS regenerate,
#    a stale tofu_topo.txt from a different allocation aborts the app loudly.
mpiexec -np 12 ./tofu_topo_helper
# 2. run the benchmark (rank 0 prints the full report)
mpiexec -np 12 ./ring_attn_bench
# knobs (env): RA_SEQ=<ctx len>, RA_ITERS=<timed tokens, use 3000+ for stable comm>
RA_SEQ=1048576 RA_ITERS=3000 mpiexec -np 12 ./ring_attn_bench
# overlap validation with real attention math (RA_GROUPS divides RA_KVH=8)
RA_SEQ=16384 RA_GROUPS=4 RA_ITERS=2000 mpiexec -np 12 ./ring_attn_overlap
# across-step async driver + HW barrier (TF_HW_BARRIER=1 uses libhwb; needs -lhwb)
RA_SEQ=16384 RA_GROUPS=4 RA_ITERS=500 TF_HW_BARRIER=1 mpiexec -np 12 ./ring_attn_async
# MoE expert-dispatch all-to-all (B=1 decode vs B=256 batched; MOE_NTNI sweeps TNIs)
MOE_BATCH=1   MOE_NTNI=6 mpiexec -np 12 ./moe_dispatch_bench
MOE_BATCH=256 MOE_NTNI=6 mpiexec -np 12 ./moe_dispatch_bench
# skewed routing (hot experts) + dimensional relay (MOE_DIMORDER, MOE_RELAY=0 skips) +
# Bruck variant (MOE_BRUCK=0 skips). worst-route at B=256 needs MOE_ITERS=300 (see #15).
MOE_BATCH=256 MOE_ROUTE=worst MOE_ITERS=300 MOE_DIMORDER=bac mpiexec -np 12 ./moe_dispatch_bench
# MoE dispatch+combine all-to-all || REAL expert SwiGLU compute OVERLAP (across-step
# async driver, libhwb HW barrier; needs -fopenmp -lhwb). B=1 = comm-trivial/imbalance-
# bound; B=256 = comm-substantial, overlap 67-81% effective (see #16). Per-rank load +
# dispatch/combine split logged for every rank.
MOE_BATCH=1   TF_HW_BARRIER=1 mpiexec -np 12 ./moe_async_bench
MOE_BATCH=256 MOE_ITERS=60 TF_HW_BARRIER=1 mpiexec -np 12 ./moe_async_bench
# all-gather (B=1 decode = latency-bound; B=256 batched = bandwidth-bound; AG_NTNI sweeps TNIs)
AG_BATCH=1   AG_NTNI=6 mpiexec -np 12 ./allgather_bench
AG_BATCH=256 AG_NTNI=6 mpiexec -np 12 ./allgather_bench
```
Rank-0 reports are also written to `ra_log_<coords>.txt` / `rao_log_<coords>.txt`
/ `raa_log_<coords>.txt` / `moe_log_<coords>.txt` / `moea_log_<coords>.txt` /
`ag_log_<coords>.txt` (mpiexec swallows rank-0 stdout on this node, so read the log
file; `moe_async_bench` logs per-rank load to *every* rank's file, not just rank 0).

## The model

KV cache of length **S** is sharded across **N** nodes (S/N positions each). One
decode step costs:

1. **KV read** — each node streams its KV shard from DRAM: `S/N * kv_bytes *
   (q-relevant)` at full node BW. Scales as **1/N**.
2. **Ring-reduce comm** — online-softmax partials (m, l, o) per query head are
   reduced around the ring. Payload = `QH*(HD+2)*4` = **16640 B** for the 9B GQA
   shape (32 q-heads, 8 kv-heads, head_dim 128). **Context-INDEPENDENT** — the
   same 16640 B whether S=16K or 1M.

## Measured findings (12 nodes, S=16384)

```
per-hop latency   = 3.80 us   (payload BW only 4.38 GB/s)
ring-reduce       = 41.83 us  (11 hops = the decode comm cost)
KV read/step      = 6.1 us    @ 915.3 GB/s node BW
```

### 1. uTofu per-hop latency ≈ 3.8 µs, **software-bound**
Put issue + remote landing + poll detection dominate; payload BW is only
~4.4 GB/s. This is the single most important number — comm cost is latency-, not
bandwidth-, limited at attention payload sizes.

### 2. The ring is **topology-insensitive** (A/B measured, commit `0c82f46`)
The default `peer=(rank+1)%N` ring on the natural rank order traverses **20
physical links** (some logical hops span 2–3 physical hops). A greedy
hop-minimized order — `0 1 3 2 4 5 11 7 6 8 9 10` — cuts that to **14 physical
links**, yet measured per-hop latency moves only **−0.5% to −1.0%** (stable over
RA_ITERS=3000; a one-off −3.8% first run was launcher-node contention).

→ ~3.8 µs/hop is base software cost; an extra physical link adds **<0.1 µs**.
**A Hamiltonian / coordinate-aware ring builder is not worth the complexity.**

### 3. Tree all-reduce is **1.78× faster** than the linear ring (MEASURED, `8b3a962`)
Real recursive-doubling (Rabenseifner) all-reduce over uTofu:
```
measured all-reduce = 23.47 us  (5 steps)
vs linear ring      = 41.83 us  (11 hops)  -> 1.78x
```
Two effects the bare `ceil(log2 N)*hop` projection (15.2 µs) missed:
- **non-power-of-2 N** needs `log2(pof2)+2` steps (12 → 5, not 4): a pre-reduce
  fold + a final broadcast.
- the tree's **distant XOR partners** (e.g. rank1↔rank8) cost ~24% above a
  nearest-neighbour hop (4.69 vs 3.80 µs/step). Combined with finding #2 (pure
  physical distance is nearly free), this extra cost is mostly **congestion**
  from the simultaneous all-pairs butterfly traffic, not physical distance.

### 4. Crossover: comm overtakes memory at **N ≈ 5** (for S=16K) — **regime flip confirmed at 1M**
At 16K context, 12 nodes are **comm-bound ~7×** (41.8 µs comm vs 6.1 µs KV read).
Direct 1M run on the same 12-node torus confirms the flip: KV read **390.8 µs**
≫ comm **41.95 µs** (comm ≈ 10%), and comm is **context-independent** (41.95 µs
@1M vs 41.83 µs @16K, 0.3%). The crossover N moves out to **36** at 1M because KV
read scales with S. So **short/medium context = comm-bound, long context =
mem-bound** — the optimization target flips with sequence length. The scaling
table below (rank-0 report, `RA_SEQ=16384`) shows the whole story from one run:

```
   N  pos/node  KV_read(us)  ring_lin(us)  tree(us)  step_max(us)    tok/s
   2      8192        36.66         3.80      3.80         36.66   27279
   4      4096        18.33        11.41      7.61         18.33   54558
   8      2048         9.16        26.62     11.41         26.62   37566
  12      1366         6.11        41.83     19.01         41.83   23906
  32       512         2.29       117.89     19.01        117.89    8482
 128       128         0.57       482.95     26.62        482.95    2070
crossover: linear ring-reduce overtakes KV read at N=5
```
The tree column flattens dramatically at large N (26.6 µs @ N=128 vs 483 µs for
the linear ring) — **once past N≈5, use a tree, not a ring**.

### 5. Per-hop decomposition + multi-TNI striping is a **dead end** (`tni_stripe_bench`)
A payload-size sweep (vary `RA_QH`) shows per-hop latency is **linear in bytes**:
```
payload   2080   4160   8320  16640  33280  66560  B
per-hop   1.56   1.83   2.49   3.81   6.75  11.70  us
```
Fit: **per-hop ≈ 1.23 µs fixed overhead + bytes / 6.36 GB/s**. The 6.36 GB/s
asymptote ≈ one TofuD link (~6.8 GB/s), so at large payload a single Put already
nears single-link saturation; the "low" 4.4 GB/s at our 16640 B payload is just
the unamortized fixed overhead, not a BW ceiling.

A64FX exposes **6 onesided TNIs**, so the obvious lever is striping a hop's
payload across K of them. **Measured: it does nothing.** Per-hop is flat in K
(same destination), then *worse* at K=6:
```
payload 16640 B:  K=1 3.87us  K=2 3.88us  K=3 3.87us  K=6 5.42us
payload 66560 B:  K=1 11.70us K=2 11.85us K=3 11.68us K=6 12.27us
```
The 6 TNIs to the **same peer share one physical link** — they add no
independent bandwidth; K=6 only piles on per-Put issue/poll overhead that can't
be hidden. Since every step of the linear ring and the recursive-doubling tree
targets a single peer, **multi-TNI striping cannot help either collective.**
TNIs are for parallelism across *distinct* destinations (different torus
links/directions), not for fattening one peer-to-peer pipe — a multi-rail /
dimensional collective could exploit them, but single-peer striping is closed.

### 6. bf16 reduce payload — the **one single-peer lever that works** (`RA_RBYTES`)
The per-hop fit says the only knob left for a single-peer hop is fewer bytes.
Sending the online-softmax partials in bf16/fp16 instead of fp32 halves the
payload (16640 → 8320 B) and delivers exactly the predicted cut on the 12-node
torus @16K:
```
                fp32 16640 B   bf16 8320 B   change
per-hop            3.81 us       2.47 us     -35%
ring-reduce        41.92 us      27.14 us    -35%
tree all-reduce    23.59 us      13.61 us    -42%
attn-only u.b.     23.9k tok/s   36.8k tok/s +54%
```
The tree gains *more* (−42%) than the ring (−35%): its distant-partner premium
is partly congestion (bytes-in-flight), so halving the payload cuts both the
base transfer and the congestion term (per-step premium +0.91 → +0.25 µs).
**bf16 + tree = 13.6 µs vs the original fp32 ring 41.9 µs = 3.1× total.**

Precision note: `m` (running max, 1 scalar/head) and ideally `l` (normalizer)
should stay fp32 — only `o` (the HD-vector accumulator, the bulk) needs to ride
in bf16. A mixed `m,l` fp32 + `o` bf16 payload = QH*(2·4 + HD·2) = 8448 B, only
~1.5% above full-bf16, so the comm win is essentially unchanged with no risk to
the normalizer. The benchmark sizes the payload via `RA_RBYTES` (4=fp32,
2=bf16); it does not model the (negligible) accumulation error.

### 7. Overlap validation with **real attention math** — the model's `max()` is doubly wrong (`ring_attn_overlap`)
Findings #1–6 model comm and approximate compute as a pure KV memory stream
(`S/N·kv_bytes` at 915 GB/s). `ring_attn_overlap.c` replaces that proxy with the
**real flash-decode kernel** (q·K dots, vectorised SVE `exp`, A·V over a per-CMG
NUMA-local bf16 KV shard, producing genuine online-softmax m,l,o partials) plus a
**real online-softmax ring-reduce**, and a **head-group pipeline** (thread 0
drives the reduce of group g−1 while threads 1..47 compute group g, per-stage
OpenMP barrier). Measured on the 12-node torus, 47 compute threads + 1 comm
driver:

```
                    16K (S/node 1363)     1M (S/node 87373)
compute-only (47T)     224 us               6368 us
comm-only (per-node)   5.7 us               5.7 us
SERIAL (compute+reduce) 221 us              6372 us
OVERLAP (head-grp G=4)  297 us  (+34%)      6868 us (+8%)
```

Two corrections to the model fall out:

- **Real compute ≫ the KV-read lower bound.** The pure-stream proxy predicts
  ~6 µs @16K / ~390 µs @1M; the *real* kernel costs **224 µs / 6368 µs** — 15–35×
  more. Even our production qpacked+ktbl kernel is only ~5–7× faster, so in
  practice **both regimes are compute-bound**, and comm is a tiny slice (16K:
  2.5%; 1M: 0.09%). The "16K is comm-bound 7×" conclusion of finding #4 holds
  only under the optimistic assumption that compute is a bare memory stream;
  with real attention math the long pole is compute, not the reduce. *(This
  bench's kernel is unoptimised — svaddv per dot, ~29 positions/thread — so its
  absolute compute is a pessimistic upper bound; the relative comm≪compute
  conclusion is robust to kernel speed.)*

- **The head-group pipeline is counterproductive — `max()` is unreachable.**
  OVERLAP is *worse* than SERIAL at both context lengths, and the penalty grows
  monotonically with the number of groups (single-run G-sweep @16K):
  ```
  G        1     2     4     8
  OVERLAP 173   298   439   534  us
  ```
  Each added group (a) makes the reduce pay the ~1.23 µs fixed per-hop overhead G
  times (finding #5's tax) and (b) adds OpenMP barriers — a G-group pipeline runs
  2·(G+1) barriers/step vs SERIAL's 3, and a 48-thread flat barrier is ~5–8 µs.
  The **extra barrier cost (~7 barriers × ~8 µs ≈ 56 µs) alone exceeds even the
  full-ring comm (42 µs)**, so a *staged within-step* pipeline can never win.
  The right way to overlap (if compute and comm were balanced — e.g. with the
  production kernel at 16K, ~35 µs compute vs ~42 µs full-ring comm) is a
  **barrier-free async comm driver**: thread 0 ring-reduces step k−1's partials
  while threads 1..47 compute step k in a *single un-staged* region, ONE barrier
  per step — overlapping *across* decode steps, not within one.

(`comm-only` here is rank-0's per-node participation ≈ one hop, not the full
N−1-hop ring; the full-ring 42 µs from finding #4 is the fair comm figure and is
still ≪ compute in both regimes.)

### 8. Across-step async comm driver **works**; head-group pipeline still loses; HW barrier usable (`ring_attn_async`)
Finding #7 said the right overlap design is *across decode steps*, not within
one. `ring_attn_async.c` builds it and measures it against a fair, **honest full
22-hop all-reduce** (reduce 0→…→11 then broadcast back, every rank ends with the
global). It also retimes everything under the **A64FX EL0 hardware barrier**
(Fujitsu `libhwb`, hierarchical: HW intra-CMG arrival + 4-way SW leader combine +
HW release) toggled by `TF_HW_BARRIER`, to see whether a cheaper barrier rescues
the head-group pipeline.

```
                         16K (S/node 1363)              1M (S/node 87373)
                      flat OMP        HW libhwb       flat OMP      HW libhwb
compute-only (47T)    187–402 us      163–190 us      7109 us       5014 us
comm-only (22-hop)    171–201 us      118–158 us       492 us        117 us
SERIAL                333–606 us      322–381 us      7653 us       6665 us
ASYNC (across-step)   196–253 us      174–269 us      7450 us       6223 us
  ASYNC vs serial    −24% … −62%     −29% … −46%       −2.7%         −6.6%
PIPE (head-grp G=4)   390–462 us      323–414 us      7521 us       7803 us
  PIPE vs serial      +19% … +22%    −5% … +29%        −1.7%        +17.1%
```
(16K is ITERS=500 over several runs — launcher-contention variance is large, so
ranges not point estimates; 1M is ITERS=40, one run/barrier.)

Three results, all robust to the variance:

- **The across-step async driver is the real win.** ASYNC beats SERIAL in *every*
  completed run — −24% to −62% at 16K (where the 22-hop comm is comparable to
  compute) and it lands at ≈ the model's `max(compute,comm)`. At 1M it wins only
  −3% to −7%, because there comm is just ~6% of the 5–7 ms step, so hiding it can
  save at most ~6% — **ASYNC's benefit scales with the comm/compute ratio and it
  never hurts.** This is the design finding #7 pointed to: thread 0 runs the full
  collective for step k−1 while threads 1..47 compute step k, double-buffered by
  iteration parity, only **2 barriers/step** (vs the staged pipeline's 2·(G+1)).

- **The head-group pipeline still loses, and the HW barrier does *not* rescue it.**
  PIPE is +19% to +29% vs serial at 16K under both barriers (the −5% in one HW run
  is within the variance, with inflated compute that dwarfs all barrier effects).
  The staged within-step tax is structural — G× the 1.23 µs fixed per-hop overhead
  plus 2·(G+1) barriers — so a cheaper barrier shrinks the second term but not the
  first; #7's conclusion stands.

- **The EL0 HW barrier is usable from plain OpenMP threads here**, and helps:
  comm-only and SERIAL are consistently lower under `libhwb` than flat OMP, and —
  notably — the HW path **completed every run**, while the flat-barrier path hit
  the intermittent cross-rank exit-race timeout ~3 of 7 runs (the tighter HW
  coupling reduces inter-rank drift). Caveat: compute-only variance is large
  (launcher contention, 163–402 µs), so treat absolute flat-vs-HW deltas as
  directional, not precise. (Matches `[[project_hwbarrier_libhwb_win]]`: same
  libhwb hierarchical barrier, default-ON in the transformer.)

### 9. Production qpacked+ktbl QK kernel in the async driver: ~20–30%, not 5–7× (`ring_attn_async`)
Findings #7–#8 used a deliberately slow QK kernel (one `svaddv` reduction per
query·position dot). To see how the across-step async overlap behaves with a
*realistic* kernel, `ring_attn_async.c` now runs the **production qpacked+ktbl
QK** kernel from `common/transformer.h`: the 16 query heads ride the SVE lanes
(q packed `[d][h]`), K is stored **K_DP `[pos][dim][kv]`**, and for each
(position, dim) the few GQA kv values are loaded once and broadcast to the
head-lanes via **`svtbl`** (`idx[i]=i/gqa`) — **no per-dot `svaddv`**. bf16 KV is
expanded with `svld1uh_u32` + `svlsl_n_u32_x(...,16)`. QH=32 is processed as two
16-lane groups; A·V and the per-thread online-softmax merge are unchanged.

```
                       16K (S/node 1363)            1M
                    flat OMP      HW libhwb       (mem-bound)
compute slow        184–257 us    136–172 us      ~6910 us
compute qpacked     148–237 us    111–144 us      ~6152 us
  qpkd vs slow      ~8–20%        ~20–30%         ~11%
```
(HW-barrier absolute compute drifts ~111–144 µs across sessions with launcher
contention; the qpkd-vs-slow ratio is from interleaved A/B in one session.)

The result: the production kernel is correct here (`chk=3.539e+04` stable across
G and runs) but buys only **~20–30%** compute, **not** the **5–7×** the same
kernel gives in the full transformer. Two reasons:

- **Production's real win is the position-parallel softmax restructure, which the
  async overlap precludes.** In the model, qpacked+ktbl rides on top of
  position-parallel softmax (shared `att[]` arrays, cross-thread max/sum
  reductions). That structure is *incompatible* with across-step overlap: the
  comm-driver thread (tid 0) cannot join compute's cross-thread barriers while it
  drives the ring all-reduce. The async driver therefore keeps the per-thread
  flash partial (each thread owns a position sub-range, merges via online-softmax)
  — so it gets the barrier-free QK trick but not the restructure that carries most
  of the 5–7×.
- **Decode attention here is fixed-cost-bound short, DRAM-bound long.** At 16K
  (~29 positions/thread across 47 threads) the online-softmax merge and the 2–3
  barriers dominate, so a faster inner dot moves the total only ~20–30%. At 1M the
  step just streams the KV shard from DRAM (~915 GB/s), so the kernel is bandwidth-
  bound and the speedup shrinks to ~11%.

This is the useful part: the faster kernel makes **16K compute (~142 µs HW) ≈ the
full-ring comm (~124 µs)** — exactly the balanced regime where across-step
overlap is the right tool. ASYNC still beats SERIAL under the qpacked kernel and
the across-step win was reconfirmed at both contexts under `TF_HW_BARRIER=1`:

```
              ASYNC vs SERIAL (qpacked, HW barrier)        n     basis
16K   −34% … −38%  (compute ~142, comm ~124, serial ~315)  2     clean, low variance
1M    −9% median  (−8% mean; range −43%…+15%)              14    pooled, contention-noisy
```

At 16K the win is clean and large (compute ≈ comm, ~70% overlap efficiency). At 1M
the step is DRAM-bound (compute ~5.6–7.2 ms ≫ comm ~0.12 ms), so the async
headroom is only the comm+sync fraction; the per-run number is buried under
±20–40% launcher-contention noise (several runs even logged SERIAL *below*
compute-only — physically impossible, pure cross-mode drift), so the −9% is the
median over 14 runs that passed a SERIAL ≥ compute-only sanity filter. It is a bit
larger than #8's −2.7%…−6.6% because async also hides the barrier/sync overhead
SERIAL pays sequentially, not just the bare ~120 µs comm. Correctness held
throughout (`chk=3.539e+04` @16K, `2.269e+06` @1M, identical across all runs).
Consistent with #8: async never hurts and its benefit scales with comm/compute.
(G now only sets PIPE granularity and comm grouping; modes A/C/D do a single
full-width 16-lane pass that reads each K cache line once. The old `svaddv` kernel
is preserved at git commit `07c0c67` for the A/B.)

### 10. MoE expert-dispatch all-to-all: multi-TNI across distinct destinations **works** (`moe_dispatch_bench`)
Findings #1–9 model the **all-reduce** half of decode comm (ring-attention's
online-softmax reduce). The *other* half is **MoE expert dispatch**: with
expert-parallelism each token's top-8-of-256 routed-expert hidden states are
scattered to the rank owning each expert (**dispatch**), the FFN runs, and the
results are gathered back (**combine**) — an all-to-all *personalized* exchange.
`tools/decode_estimate.py` modeled this as `COLLECTIVES_PER_LAYER=2 ×`
tree-all-reduce cost; this bench measures the real pattern. Comm-only (no FFN):
every rank reproduces the same N×N routing matrix from a shared seed, so per-pair
byte sizes agree with no runtime exchange; per-pair payload is `cnt·HID·2 B` with
the seq counter trailing at the variable offset `plen(cnt)`. Three variants:
naive single-TNI pairwise, multi-TNI (distinct dsts round-robined over `MOE_NTNI`
TNIs, all Puts issued before any TCQ drain), and the tree-all-reduce baseline.

**The headline: distinct destinations DO parallelize across TNIs** — the exact
case finding #5's single-peer dead-end said *should* work. NTNI sweep, B=256
(2048 selections/rank, every pair ~170 H-vectors ≈ dense all-to-all):

```
NTNI   dispatch us   speedup   agg BW
 1       10464         1.02×    28.9 GB/s   (sanity: multi w/ 1 TNI ≈ naive)
 2        7365         1.45×    41.0 GB/s
 3        4703         2.28×    64.2 GB/s
 4        3720         2.71×    81.2 GB/s
 6        3500         3.00×    86.3 GB/s
```

It is **sub-linear — caps at ~3× for 6 TNIs, not 6×.** With 11 destinations over
a 2×3×2 torus (axis extents 2/3/2) the first few TNIs hit independent axis links;
past ~4 TNIs the distinct-destination Puts start sharing physical links. So
finding #5's "the only payoff from 6 TNIs is sending to *different* peers" is now
**measured and bounded**: real, worth ~3×, but not the naive 6×. At B=1 decode
(only K=8 H-vectors over ≤5 distinct dsts) multi-TNI still helps **1.5–2.4×**.

**Roofline check** — `(dispatch+combine)_best / (2·tree)`:

```
            dispatch+combine (best, multi-TNI)   2×tree   ratio
B=1            34.6 us                            37.3 us   0.93
B=256        6529   us                          7415   us   0.88
```

The ratio < 1 means real multi-TNI all-to-all is **~10% cheaper** than
`decode_estimate.py`'s "2× tree all-reduce" proxy — so `COLLECTIVES_PER_LAYER=2`
is sound (slightly conservative). **But that only holds *because of* multi-TNI:**
the naive single-TNI all-to-all is ~2.8× the tree cost and *would* blow the
budget. The multi-TNI dispatch is precisely what keeps the roofline assumption
valid. Whole-model (×75 MoE layers) best dispatch+combine: **2.6 ms/tok @B=1**,
**490 ms/tok @B=256** (the batched all-to-all is the real long pole at scale).

**…and the proxy holds only under UNIFORM routing.** Real MoE routing is skewed
(hot experts); with `MOE_ROUTE=worst` (selections concentrated on ~N/4 hot ranks)
the picture changes sharply (B=256, multi-TNI):

```
route                max pair   dispatch+combine   ratio vs 2×tree   agg BW
uniform              214        6530 us            0.88              85.6 GB/s
worst                725        16195 us           2.18              36.5 GB/s
worst, capfac 1.25   907        20195 us           2.72              36.7 GB/s
```

Two effects compound: traffic concentrates on a few **hot destination links**
(agg BW *halves*, 85→36 GB/s — multi-TNI can't rescue a congested link any more
than finding #5's single-peer striping could), and the per-pair payload grows
with the skew. So **`COLLECTIVES_PER_LAYER=2` is the optimistic, load-balanced
case**; under realistic imbalance the MoE comm term is **2–3× larger** and the
all-reduce proxy under-estimates it. The multi-TNI speedup itself stays ~3.2×
(the few hot destinations are large transfers spread over independent links), but
the absolute cost is set by the hottest link. Practical implication: the MoE
comm budget should be derated for routing skew (or kept near-uniform with an
auxiliary load-balancing loss + capacity factor, the standard EP mitigation).

### 11. Dimensional (store-and-forward) all-to-all is a **dead end** on the in-unit torus (`moe_dispatch_bench` variant `d`)

The open question from finding #10 was whether an **axis-aware** schedule could
beat the round-robin multi-TNI's ~3× cap and, especially, relieve the hot-link
congestion that wrecks all-to-all under skew. Variant `d` (`MOE_RELAY=1`, default
on) implements the textbook **dimension-ordered store-and-forward** all-to-all
along the 2×3×2 in-unit torus axes a→b→c (`coords[3..5]`): each message hops one
axis per phase, and intermediate nodes **aggregate and forward** (Bruck-style),
so every Put is a single-axis line send rather than a hardware multi-hop route.
Per-phase bundle byte sizes are derived analytically from the shared count matrix
(no real data needed). It **loses in every regime:**

```
regime                 multi-TNI(dc)   relay(dc)    relay/multiTNI   relay wire/useful
B=1   decode           34.6 us         98.4 us      2.84×            1.69×
B=256 uniform          5832 us         14310 us     2.45×            1.66×
B=64  worst (skew)     4009 us         8608 us      2.15×            1.67×
B=256 worst (skew)     16283 us        36018 us     2.21×            1.67×
```

Four reasons, none fixable by scheduling:
1. **Three sequential phases** serialize — pure latency tax, fatal at B=1 (small
   messages), where direct multi-TNI just fires N−1 concurrent Puts.
2. **Forwarding moves 1.66–1.69× the useful bytes** (each byte re-sent at every
   axis it must traverse) — a structural bandwidth penalty.
3. **It does NOT relieve hot-link congestion.** Store-and-forward funnels a hot
   destination's *entire* incoming column through its few final-hop line-neighbors,
   so the hot node's ingest stays bottlenecked exactly as in direct routing —
   relay/multi-TNI ratio is ~2.2× under skew, same as uniform.
4. The **TofuD hardware already routes multi-hop well**, and multi-TNI already
   parallelizes across distinct destinations (finding #10, ~3× agg BW). Software
   dimensional routing only piles overhead on top.

Dimension order has a secondary effect (~13%: `MOE_DIMORDER=bac`, the extent-3
axis first, is best under skew) but never closes the ~2× gap. Relay still beats
*naive single-TNI* (~0.86×) only because naive is so bad — that is not a win.
Also surfaced here: a **single uTofu Put caps at ~16 MiB** (12.6 MB completes,
25.3 MB hangs); the aggregated relay bundles exceed it under skew, so `relay_put`
chunks at 8 MiB — relevant for any large batched payload, not just relay.
**Takeaway: stick with direct multi-TNI all-to-all (finding #10); the single-unit
torus is too small and too well-served by hardware routing for software dimension
ordering to pay.** (Across *multiple* units — inter-unit Tofu axes with real
multi-hop distance — the trade-off could differ; untested here.)

**Multi-unit readiness (code in place, not yet measured):** variant `d` now relays
over **all six Tofu axes** `x→y→z→a→b→c` (`coords[0..5]`), not just the in-unit
`a,b,c`. Axes are modeled by the *set of distinct coordinate values present* on
each axis and ranks by their per-axis index into that set, so the machine-scale
x,y,z coords collapse to small dense ranges. On a single unit x,y,z are degenerate
(extent 1) and the run is byte-identical to the old a,b,c relay (verified: grid
`1x1x1x2x3x2`, RATIO relay/(2·tree)=2.64 unchanged); a multi-unit allocation lights
up the x,y,z phases automatically. The relay enables only when the ranks form a
full Cartesian grid (`N == prod(ext)`, every cell populated); `MOE_DIMORDER` now
takes any subset of `xyzabc` (omitted axes appended in canonical order). This is
the harness for the finding-#11 inter-unit experiment — it needs a multi-unit job
(`pjsub` from a login node, unavailable on this fixed single-unit interactive node).

### 12. All-gather: multi-TNI all-broadcast wins **both** regimes; recursive-doubling is a non-pow2 trap (`allgather_bench`)

All-gather is the third collective the suite now covers — the **bandwidth-optimal
half of an all-reduce** (all-reduce = reduce-scatter + all-gather) and the concrete
cost of reconstructing a sequence-parallel sharded hidden vector (or gathering KV /
sharded weights) across TP ranks. Each rank owns `HID·abytes·B / N` of the vector
and gathers the rest. Four transports over the same gather result, plus the tree
all-reduce baseline:

```
                              B=1 decode (1 KiB shard)   B=256 batched (256 KiB shard)
naive (1-TNI all-broadcast)   14.99 us                   1160 us
multi-TNI all-broadcast       4.93 us  (×3.04 vs naive)  436 us  (×2.66 vs naive)  <- WINS both
ring all-gather               15.68 us (×0.96)           470 us  (×2.47)
recursive-doubling (Rab.)     13.02 us (×1.15)           1532 us (×0.76 — worse than naive!)
tree all-reduce (full vec)    18.39 us                   3707 us
RATIO best_all-gather / tree  0.27                       0.12
```

Findings:
1. **Multi-TNI all-broadcast is the single robust winner.** Just fire the N−1
   independent "send my shard to peer p" Puts round-robined over the 6 TNIs and
   drain per-TNI. No ring/recursive structure needed. NTNI sweep at B=256:
   ×1.00 / 1.13 / 2.22 / 2.63 for NTNI=1/2/3/6 — the **same sub-linear ~3× cap as
   finding #10** (distinct destinations start sharing torus axis links past ~3–4
   TNIs). It wins in the latency-bound decode regime (×3.04, because N−1 concurrent
   small Puts beat any serial-dependency schedule) *and* ties/edges ring in the
   bandwidth-bound regime.
2. **Recursive-doubling all-gather is a trap on 12 (non-pow2) nodes.** It is the
   latency-optimal small-message pattern in theory and the direct analog of the
   tree all-reduce that wins finding #3 — but the Rabenseifner non-pow2 fold makes
   it **over-move bytes**: each paired-odd rank sends 2+4+8 (doubling) + 12
   (fold-out) ≈ 26 shards vs ring's 11, all on one serial TNI. So in the
   bandwidth-bound regime it is the *worst* variant (×0.31 of ring, slower than
   even naive). In the latency-bound regime its fewer steps (log N + fold) only
   modestly beat ring (×1.20) and lose badly to multi-TNI. The tree-beats-ring
   result of finding #3 does **not** carry over to all-gather here.
3. **All-gather is much cheaper than a full all-reduce** — 0.27× (decode) to 0.12×
   (batched) of the tree all-reduce, because all-reduce moves ≈NRounds×full bytes
   while all-gather moves ≈1×full. So `decode_estimate.py`'s all-reduce term is a
   conservative bound for any gather-only TP comm (and reduce-scatter, its mirror,
   is the same cheap half). The 0.12 batched ratio shows the gather is firmly
   bandwidth-bound there; the 0.27 decode ratio is fixed-cost-bound.

**Takeaway: for any fan-out-to-all-peers collective (all-gather, MoE dispatch),
multi-TNI distinct-destination Puts are the right primitive in every regime; ring
is an acceptable BW-optimal fallback; skip recursive-doubling all-gather on
non-power-of-2 node counts.**

### 13. Reduce-scatter mirrors all-gather; decomposing all-reduce beats the fused tree **2–4×** (`reducescatter_bench`)

Reduce-scatter is the other bandwidth-optimal half of an all-reduce
(all-reduce = reduce-scatter + all-gather): every rank starts with the full
N-shard vector and ends owning the *sum* of just its own shard. This bench
measures it directly (comm only — the per-receive add is elided, like every
bench here) and closes the decomposition empirically in **one** allocation: it
also times a multi-TNI all-gather over the reduced shards so the decomposed pair
can be compared to the fused recursive-doubling tree the estimator assumes.

```
                              B=1 decode (1 KiB shard)   B=256 batched (256 KiB shard)
naive (1-TNI direct scatter)  15.18 us                   469 us
multi-TNI direct scatter      4.79 us  (×3.17 vs naive)  429 us  (×1.09 vs naive)
ring reduce-scatter           15.91 us (×0.95)           469 us  (×1.00)
multi-TNI all-gather (the AG half) 4.91 us               422 us
tree all-reduce (fused)       18.74 us                   3708 us
RATIO best_RS / tree          0.26                       0.12
DECOMPOSED (RS + AG) / tree    0.52×                      0.23×   <- decomposing WINS
```

Findings:
1. **Reduce-scatter and all-gather cost the same** — they are exact wire-pattern
   time-reverses. multi-TNI RS (4.79 / 429 µs) ≈ multi-TNI AG (4.91 / 422 µs) and
   `best_RS/tree` = 0.26 / 0.12 reproduces finding #12's all-gather ratios. So the
   two halves of an all-reduce are interchangeable in cost; pick by which buffer
   layout the surrounding kernel wants.
2. **Decomposing an all-reduce into multi-TNI reduce-scatter + multi-TNI
   all-gather is 2× cheaper at decode and ~4× at batched** than the fused
   recursive-doubling tree (0.52× / 0.23×). The fused tree moves ≈2×(1−1/N)×full
   bytes per rank over its `log N + fold` serial steps; two multi-TNI fan-outs each
   move ≈1×full with the distinct-destination links running concurrently. *Caveat:*
   this is a **comm-only** comparison — a real reduce-scatter must also perform the
   N−1 shard-sized vector adds, which the fused tree folds into its sends; that
   reduction compute (memory-bound, ≈ the gathered bytes) eats into the 2–4× when
   the shard is large, but at decode (1 KiB shard) it is negligible. The headline:
   `decode_estimate.py`'s fused-tree all-reduce term is **pessimistic** for TP comm
   that can be expressed as scatter-then-gather.
3. **The multi-TNI ~3× lever is a latency effect, not a bandwidth multiplier.**
   Direct scatter gets ×3.17 at B=1 but only ×1.09 at B=256: naive single-TNI
   scatter already runs at the serial per-link limit there (469 µs ≈ 11 × 256 KiB
   at ~6.4 GB/s), and one process's Tofu injection caps at ≈1 link's bandwidth
   (~6.7 GB/s ingest) regardless of TNI count for large messages. The 6 TNIs to 6
   distinct peers buy concurrency only while messages are small enough to be
   fixed-cost-bound — exactly the decode regime. (This also explains why the big
   multi-TNI speedups in #10/#12 are largest at B=1.)

**Takeaway: model a TP all-reduce as `reduce-scatter + all-gather` of multi-TNI
fan-outs, not as a fused tree — it is 2–4× cheaper in pure comm. Reduce-scatter
and all-gather are cost-symmetric, so the suite's all-gather numbers (finding #12)
carry over directly.**

### Finding #14 — pipeline-parallel handoff: comm-cheap but bubble-bound; one hop = one link, distance-free

`pp_handoff_bench.c` measures the *point-to-point* inter-stage send/recv that
pipeline parallelism uses **instead of** per-layer collectives. The model's layers
split into S=12 stages (one per rank); a microbatch's activation (`HID·ABYTES·B`,
**not** sharded) is Put across each stage boundary, forward `0→…→S-1` then backward
in training. Measured 12-node 2×3×2 in-unit torus:

| | per-hop (isolated) | per-hop (in-chain) | fwd chain (S-1=11 hops) | fwd+bwd RT (22 hops) | one tree all-reduce |
|---|---|---|---|---|---|
| **B=1** (decode, 12 KiB) | 3.17 µs near / 3.35 far | 3.20 µs | 35.3 µs | 70.5 µs | 18.3 µs |
| **B=256** (microbatch, 3 MiB) | 496.9 / 497.1 µs | 497.0 µs | 5467 µs | 10934 µs | 3707 µs |

1. **A single handoff is one TofuD link, and torus distance is free.** Near (rank 0↔1)
   vs far (rank 0↔11, opposite corner) differ by ×1.06 at B=1 and ×1.00 at B=256 —
   confirming the ring's topology-insensitivity (≈1%) for an isolated P2P. At B=256
   the 3 MiB hop runs at 3 MiB / 497 µs ≈ **6.33 GB/s = exactly one TofuD link**
   (cf. the 6.36 GB/s ring-hop and the 6.7 GB/s reduce-scatter ingest cap, #13) —
   one process injecting to one peer cannot exceed one link, distance notwithstanding.
2. **Store-and-forward is free.** The in-chain per-hop (3.20 µs / 497.0 µs) equals the
   isolated ping-pong per-hop — each stage's wake-on-seq + re-issue latency is fully
   hidden behind the next hop's transfer; the chain is just `(S-1) ×` one hop, with no
   relay tax (unlike the MoE dimensional relay #11, which paid 3 serial phases + 1.67×
   forwarding bytes).
3. **PP trades comm volume for the pipeline bubble.** The full S-1-hop forward chain
   costs ×1.93 (B=1) / ×1.47 (B=256) of *one* tree all-reduce — but PP pays this
   **once for the whole model**, whereas TP pays `2·num_layers` all-reduces. For a
   ~75-layer decode that is PP's 35 µs vs TP's ≈150 × 18.3 µs ≈ 2.7 ms of comm — PP
   comm is ~80× smaller. The catch is the bubble: with M microbatches in flight the
   idle fraction is `(S-1)/(M+S-1)` = **0.92 at M=1**, 0.58 at M=8, 0.15 at M=64.
   Autoregressive decode has **one token in flight** (M=1), so the pipeline can never
   fill: the (S-1)-hop serial chain lands *fully on the critical path* (35 µs added
   latency) and 92% of the stages sit idle. **PP only pays off for decode with many
   concurrent sequences (large M) to amortize the bubble; for single-stream
   low-latency decode it adds chain latency without usable comm savings.** TP (cheap
   per-step comm, no bubble, but `2·num_layers` collectives) is the better single-stream
   structure; PP wins only at high decode batch / throughput-oriented serving.

**Takeaway: a PP stage handoff is one link, distance-free, and store-and-forward is
free — so the whole-model PP comm is just `(S-1) ×` one-link transfer, ~80× below TP's
per-layer all-reduce sum. But decode's M=1 makes the bubble 92%, so PP is a
throughput (large-M) lever, not a latency one. `decode_estimate.py`'s collective
terms model TP; a PP variant would replace them with `(S-1)·t_hop / (1-bubble)`.**

### 15. Bruck logarithmic all-to-all is a **dead end** for MoE dispatch (`moe_dispatch_bench` variant `e`)

The last untested all-to-all schedule: the classic **Bruck** index algorithm
(`MOE_BRUCK=1`, default on), which trades **message count for byte volume** —
`⌈log₂N⌉` doubling steps instead of N−1 sends. Step `z` sends the blocks whose
index has bit `z` set to `dst=(i+2^z)%N` and receives the mirror set from
`src=(i−2^z)%N`, with a final inverse rotation mapping each received block back to
its originator (verified: every rank's slot `(i−s)%N` ends holding source `s`'s
block). Unlike the dimensional relay (#11) it needs **no Cartesian grid** — it runs
for any N≥2 — but it requires **uniform blocks**, so every per-pair payload is
padded to the global max pair count. Single TNI (each step hits one peer, so there
is nothing to stripe across rails). 12 nodes → 4 steps. It **loses in every regime:**

```
regime                 multiTNI(dc)  bruck(dc)  bruck/multiTNI  bruck/naive  wire/useful  bruck/(2·tree)
B=1   decode           34.8 us       246 us     7.06×           3.85×        7.50×        6.61
B=256 uniform          6565 us       16657 us   2.54×           0.79×        2.09×        2.25
B=256 worst (skew)     16436 us      56280 us   3.42×           1.36×        7.08×        7.59
```

Two root causes, both structural — the *same* failure mode as #5 (single-peer
striping) and #11 (dimensional relay):
1. **It collapses the all-to-all into a few fat single-link rounds.** Each step
   targets one peer, so the 6 TNIs across distinct destinations — the *only* thing
   that gives direct multi-TNI its ~3× win (#10) — cannot be used. Cutting the
   message count 11→4 doesn't pay because MoE all-to-all on TofuD is
   **bandwidth/padding-bound, not message-count-bound** (the regime where Bruck
   classically wins is tiny dense messages, latency-bound — not 12 KiB+ blocks).
2. **Uniform-block padding inflates the wire 2.1–7.5×.** MoE routing is sparse
   (B=1: 5 of 11 pairs active, counts ≈1) and skewed (worst: 3 of 11 pairs, one at
   725× the rest), so padding *every* block to the global max moves enormous dead
   bytes. Padding is worst exactly where it hurts most — sparse B=1 (7.5×) and
   skewed B=256 (7.1×); only dense-uniform B=256 keeps it to 2.1×.

Bruck beats *naive single-TNI* only in the one dense-uniform case (0.79×), and only
because naive is so bad — same non-win as the relay. It never approaches multi-TNI.
**Takeaway: direct multi-TNI all-to-all (#10) remains the answer; both software
all-to-all reschedulings tested (dimensional relay #11, Bruck #15) lose for the same
reason — they give up the multi-rail, distinct-destination concurrency that is the
torus's actual lever, and add padding/forwarding bytes on top.** This closes the
all-to-all schedule space for the single-unit torus.

(Note: the B=256 *worst*-route run needs `MOE_ITERS≈300` — at the default 2000 the
huge skewed payloads make the four direct phases finish tens of seconds apart, and
the un-barriered **tree** baseline phase, which a fast rank enters while a slow rank
is still draining, trips its 30 s wait. Lower iters keep the cross-phase skew bounded;
the timing is unaffected since each iter already moves ~GB.)

### 16. MoE dispatch+combine async overlap: a **two-regime split** — useless at B=1 (comm is 0.5%, load imbalance dominates), 67–81% effective at B=256 (`moe_async_bench`)

The first bench in the suite to overlap **real expert compute** with the
dispatch+combine all-to-all — the MoE analogue of the across-step async driver (#8).
tid0 is the comm driver (runs `combine(prev)+dispatch(next)` over multi-TNI distinct-
destination all-to-all, #10) while threads 1..47 run the expert SwiGLU FFN for the
current step; a **compute-threads-only barrier** (CBAR, tid0 excluded — the #9
constraint) carries the gate/up→down dependency. Compute is **real bf16 GEMV**,
M=1 per active expert (decode is weight-read-bound: `NActive = min(fan-in, experts/node)`
expert weight matrices, 54 MiB each, streamed from HBM by 47 threads = 4 CMG × 12,
libhwb HW barrier). Measured 12-node 2×3×2, E=256/K=8/HID=6144/INTER=1536, per MoE step (µs):

```
regime          compute  comm   serial  async   overlap-eff  async/serial  async/max()  bound
B=1   decode      5179      39    8406    7090     41%          -16%          +37%       straggler-compute
B=256 dense      13906    6440   22305   16651   67-81%         -25%          +20%       compute (+ real comm)
```

1. **B=1 decode: comm is a rounding error (39 µs = 0.5% of the step) — there is nothing
   to hide.** The measured dispatch+combine (39 µs) *matches* `decode_estimate`'s
   `2·tree` proxy (44–106 µs), so the roofline's comm term is accurate — but it is
   negligible against ~5–7 ms of expert compute. The SERIAL "combine" cost (2.78 ms)
   is **not communication**: it is rank 0 blocking at the combine all-to-all for the
   **straggler rank's compute** to finish. At B=1 expert load is severely imbalanced —
   each rank's fan-in R (the (token,expert) pairs landing on its experts) ranged **3 to
   12 across the 12 ranks** (a balls-in-bins spread: 12 tokens × top-8 over 12 owner-
   ranks), so per-rank compute ranged 1.8–7.1 ms. ASYNC's 16% gain is **not from hiding
   comm** — it comes from dropping 3 of SERIAL's 4 global barriers and decoupling the
   per-step straggler sync. **The B=1 lever is load balancing (capacity factor / expert
   placement), not comm/compute overlap.**
2. **B=256 dense: load balances (every rank saturates its 22 local experts) and comm
   becomes substantial (6.4 ms ≈ 31% of the serial step, 94 GB/s aggregate over 6
   TNIs).** Here the across-step driver delivers: ASYNC (16.7 ms) lands **67–81% of the
   way** from serial (22.3 ms) to the compute roofline (13.9 ms) — −25% vs serial,
   within +20% of `max(compute,comm)`. This is the MoE confirmation of #8: the multi-TNI
   dispatch+combine all-to-all **can be hidden behind expert compute** with a tid0
   driver, exactly when comm is a meaningful fraction of compute.
3. **Caveat — B=256 here is a comm-favorable stress test, not real batched MoE.**
   Compute is M=1 GEMV per expert (the weight-read-bound decode model), but the dispatch
   payload is B-scaled (2 MiB/pair ≈ 170 tokens' worth). A faithful batched MoE would
   compute M≈170 per expert — far more FLOPs — making comm a *smaller* fraction and
   overlap *less* impactful. So this regime over-states comm's weight; it proves the
   overlap *mechanism* works for all-to-all, but real prefill is more compute-dominated.
   The DIAG also shows the in-phase compute inflating ~1.2× at B=256 (the 2 MiB/pair
   dispatch DMA writes contend with the 47 threads' weight reads for HBM bandwidth);
   at B=1 the payload is tiny and there is no such contention (ratio ≈1.0).

**Takeaway: `decode_estimate`'s MoE-comm term (≈2·tree) is accurate in magnitude but the
wrong thing to optimize at B=1 — comm is 0.5% of the MoE step; the straggler/imbalance is
the cost. Async dispatch+combine overlap is a real but regime-specific win (only when comm
is ≳20–30% of compute, i.e. dense / large batch), confirming #8 generalizes from the ring
all-reduce to the MoE all-to-all. Across both regimes the dominant cost is never the comm:
it is expert load imbalance at decode and batched-GEMM compute at prefill.**

### 17. Point-to-point link floor: ~1 µs one-way, ~6.3 GB/s/link, mild distance slope (`p2p_latbw_bench`)

Direct characterization of the raw Tofu links on the 12-node 2×3×2 (a,b,c extents
2,3,2; x,y,z fixed), all driven from rank 0. This is the only bench that exercises
one-sided **Get** (`utofu_get`, completion via local MRQ `UTOFU_MRQ_TYPE_LCL_GET`)
and **`armw8`** fetch-add (MRQ `LCL_ARMW`, pre-op value in `notice.rmt_value`) —
the rest are Put + memory-sentinel polling.

**Latency vs torus hop-distance** (8 B, 2000 iters/op, one-way = ping-pong RTT/2):

| dist | peers | Put 1-way | Get RTT | armw RTT |
|------|-------|-----------|---------|----------|
| 1 | 1,2,4,6 | 0.97–0.99 µs | 1.48–1.51 µs | 1.32–1.35 µs |
| 2 | 3,5,7,8,10 | 1.04–1.08 µs | 1.64–1.66 µs | 1.48–1.50 µs |
| 3 | 9,11 | 1.13 µs | 1.80 µs | 1.63 µs |

The wraparound torus metric is confirmed correct: rank 2 (b=2) is distance **1**,
not 2, via the ring-of-3 b-axis. Distance slope is real but **shallow** — ~+0.08 µs
one-way per hop (~+8%/hop) — consistent with finding #14 ("one hop = one link,
distance is a rounding error") and the ring-attn ~1.23 µs fixed per-hop figure.
Get/armw cost a bit more than a Put one-way because each is a full request→response
round-trip the initiator waits on (no remote CPU involved).

**Bandwidth vs message size** (peer rank 9, dist 3, async window W=8):

| size | Put inject | Get delivered |
|------|-----------|---------------|
| 1 KiB | 3.49 GB/s | 3.48 GB/s |
| 4 KiB | 6.18 | 6.12 |
| ≥16 KiB | 6.30–6.34 | 6.28–6.35 |

Saturates at **~6.34 GB/s** for ≥16 KiB — matches the recorded single-TofuD-link
~6.36 GB/s (and finding #5: one peer = one link, so W>1 fills the pipe but does not
exceed one link). Small messages are latency-bound; the half-BW point is ~4 KiB.
Put-injection (sender TCQ) and Get-delivered (data landed locally, MRQ) track each
other within noise, so the link is symmetric and injection ≈ delivered at this scale.

**Takeaway: the link floor is ~1 µs one-way / ~6.3 GB/s, distance adds only ~8%/hop.
Every higher-level number in this suite is consistent with this floor; the levers
that matter are payload size (finding #6) and parallelism across *distinct* peers
(findings #10/#12), never a single point-to-point link.**

## Practical guidance for distributed decode attention

- **Few/large shards** beat many/small ones while comm-bound: keep N at or below
  the crossover unless KV is too large to fit. The sweet spot here is N≈4 (54k
  tok/s attn-only upper bound) vs N=12 (24k).
- **Use tree all-reduce past N≈5.** The linear ring is fine only for tiny N.
- **Don't bother coordinate-ordering the ring** — software-bound per-hop makes it
  a ~1% effect.
- **Don't stripe a hop across TNIs** — same-peer TNIs share one link, zero gain
  (K=6 is a net loss). The payoff from the 6 TNIs is sending to *different* peers
  at once, which is **real and measured** (finding #10): a multi-TNI all-to-all
  is ~3× the single-TNI one (sub-linear past ~4 TNIs as distinct destinations
  start sharing torus axis links). Reach for it whenever the collective fans out
  to many peers (MoE dispatch **and all-gather**, finding #12), not when it targets
  one peer per step (ring/tree). For all-gather specifically it beats ring,
  recursive-doubling and naive in *both* the latency- and bandwidth-bound regimes;
  **skip recursive-doubling all-gather on non-power-of-2 node counts** (the
  Rabenseifner fold over-moves bytes — finding #12).
- **Send the reduce payload in bf16, not fp32** (`RA_RBYTES=2`) — the one
  single-peer knob that works: ring-reduce −35% (41.9→27.1 µs), tree −42%
  (23.6→13.6 µs), attn-only upper bound +54%. Keep `m` (and ideally `l`) fp32,
  send `o` in bf16 (mixed payload only ~1.5% above full-bf16). With a tree this
  compounds to 3.1× over the fp32 ring.
- Comm being context-independent means **long-context decode is mem-bound** (KV
  read dominates) and **short/medium-context decode is comm-bound** (latency of
  the reduce dominates) — the optimization target flips with S. *But this only
  holds for the bare-memory-stream model; with real attention math (finding #7)
  the long pole is compute in both regimes.*
- **Don't pipeline comm under compute with a per-stage barrier** (head-group or
  similar). The barrier overhead per stage exceeds the comm it hides (finding #7:
  overlap is +34% at 16K, +8% at 1M, monotonically worse with more groups; finding
  #8: still +19–29% even under the cheaper HW barrier). Overlap **across decode
  steps** instead — the async driver of finding #8 (thread 0 runs step k−1's full
  all-reduce while 1..47 compute step k, 2 barriers/step) beats serial in every
  run, by −24% to −62% at 16K and ≈ the comm fraction at 1M. It never hurts.
- **The A64FX EL0 HW barrier (`libhwb`) is usable from OpenMP threads** and worth
  it (finding #8): lower SERIAL/comm cost and, here, far fewer cross-rank
  exit-race hangs than the flat OMP barrier. Link `-lhwb`, gate with
  `TF_HW_BARRIER`; see `[[project_hwbarrier_libhwb_win]]`.
- **A faster QK kernel helps decode attention only ~20–30% on its own** (finding
  #9). The production qpacked+ktbl QK trick (svtbl K-broadcast, no `svaddv`) gives
  5–7× *in the full transformer* mostly via the position-parallel softmax
  restructure — which the async overlap can't use (the comm-driver thread can't
  join compute's cross-thread barriers). With the per-thread flash partial that
  async needs, you keep the barrier-free QK win but not the restructure; short ctx
  is fixed-cost-bound (merge + barriers), long ctx DRAM-bound. The payoff is that
  it rebalances 16K compute to ≈ the full-ring comm, the regime async targets.

## Implementation notes / gotchas

- **Cache-line hazard**: each uTofu recv slot must sit on its own 256 B cache
  line that the CPU only ever *reads*. If a recv buffer shares a line with
  CPU-written data, the RDMA write lands but the CPU never sees it (the dirty
  line clobbers it). See `[[reference_utofu_cacheline_hazard]]`.
- **`TREE_ALLREDUCE` is a macro** that evaluates its argument many times — pass a
  plain variable, never `tok++` (caused a double-eval blow-up; found + fixed).
- **Always regenerate `tofu_topo.txt`** for the current allocation before
  running; a stale file from a different shape aborts with "my coords not in
  topo".
- Node read BW (~915 GB/s/node from one process) requires per-CMG-local buffers +
  `mbind(MPOL_MF_MOVE)` + pinned threads + multi-accumulator SVE sum; see
  `[[reference_a64fx_full_node_bw]]`.

## Commit lineage

```
9f4df6c  MPI-free multi-node uTofu Put exchange demo
33881a2  scale ring demo to up to 12 nodes
1095967  ring-attention DECODE cost benchmark (no GEMM)
765bcce  saturate node memory BW (per-CMG-local SVE stream)
44100a5  N-sweep scaling projection in rank-0 report
8b3a962  measure real tree all-reduce comm path
0c82f46  A/B topology-ordered ring vs default rank+1
8797a77  study summary & findings (summary.md)
5157173  multi-TNI striping microbench + 1M confirmation
7584a0b  RA_RBYTES knob — bf16 reduce payload win
8e2da9e  ring_attn_overlap — comm/compute overlap validation (real math)
07c0c67  ring_attn_async — across-step async driver + EL0 HW barrier
5110468  ring_attn_async — production qpacked+ktbl QK kernel (svtbl, K_DP)
ae8e731  moe_dispatch_bench — MoE expert-dispatch all-to-all (multi-TNI)
57bc9f7  decode roofline: fold MoE dispatch all-to-all result in
0db2fe1  docs: MoE roofline holds only under uniform routing; skew 2.2-2.7×
c07f425  moe_dispatch_bench — dimensional store-and-forward variant (dead end, #11)
14f0a43  allgather_bench — all-gather (multi-TNI wins; recursive-doubling trap, #12)
e317b82  moe_dispatch_bench — relay extended to all 6 Tofu axes (multi-unit ready, #11)
f12ee81  reducescatter_bench — reduce-scatter mirrors all-gather; decompose all-reduce beats fused tree 2-4× (#13)
51e8a4f  pp_handoff_bench — pipeline-parallel handoff: one link/hop, distance-free, bubble-bound (#14)
5814272  decode_estimate — replace flat 2×-tree comm proxy with measured per-collective costs
080a70a  moe_dispatch_bench — Bruck logarithmic-index all-to-all (variant e, dead end, #15)
(this)   moe_async_bench — dispatch+combine all-to-all || real expert SwiGLU overlap (two-regime, #16)
```

Cross-references in auto-memory: `reference_ring_attn_decode_cost`,
`reference_a64fx_full_node_bw`, `reference_utofu_cacheline_hazard`.
