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
| `Makefile`, `run_bench.sh`, `run_demo.sh` | build + launch |

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
```
Rank-0 reports are also written to `ra_log_<coords>.txt` / `rao_log_<coords>.txt`
(mpiexec swallows rank-0 stdout on this node, so read the log file).

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

## Practical guidance for distributed decode attention

- **Few/large shards** beat many/small ones while comm-bound: keep N at or below
  the crossover unless KV is too large to fit. The sweet spot here is N≈4 (54k
  tok/s attn-only upper bound) vs N=12 (24k).
- **Use tree all-reduce past N≈5.** The linear ring is fine only for tiny N.
- **Don't bother coordinate-ordering the ring** — software-bound per-hop makes it
  a ~1% effect.
- **Don't stripe a hop across TNIs** — same-peer TNIs share one link, zero gain
  (K=6 is a net loss). The only payoff from the 6 TNIs is sending to *different*
  peers at once.
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
  overlap is +34% at 16K, +8% at 1M, monotonically worse with more groups). If
  compute and comm are genuinely balanced, overlap with a **single barrier-free
  async comm driver across decode steps** instead — never a staged within-step
  pipeline.

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
         ring_attn_overlap — comm/compute overlap validation (real math)
```

Cross-references in auto-memory: `reference_ring_attn_decode_cost`,
`reference_a64fx_full_node_bw`, `reference_utofu_cacheline_hazard`.
