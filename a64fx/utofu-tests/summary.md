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
```

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
- **To shrink the reduce payload** (the only per-hop knob that helps once
  overhead is amortized): send m,l,o in bf16/fp16 instead of fp32 — halves the
  2.6 µs transfer component at 16640 B. The per-hop fit `1.23 + bytes/6.36GB/s`
  says nothing else moves the single-peer hop.
- Comm being context-independent means **long-context decode is mem-bound** (KV
  read dominates) and **short/medium-context decode is comm-bound** (latency of
  the reduce dominates) — the optimization target flips with S.

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
```

Cross-references in auto-memory: `reference_ring_attn_decode_cost`,
`reference_a64fx_full_node_bw`, `reference_utofu_cacheline_hazard`.
