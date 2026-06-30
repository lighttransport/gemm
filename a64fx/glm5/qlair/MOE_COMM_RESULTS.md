# GLM5 MoE communication cost on qlair (uTofu model), 4–8 ranks

Models the GLM5 MoE **all-to-all** (expert dispatch + combine) and the attention
**all-reduce** on qlair's Fugaku-calibrated uTofu cost model (`put_ns = 1400 +
bytes/6.3`, 6.3 GB/s per TNI, 6 TNIs). GLM5: hidden=6144, **int8** activations
(1 B/elem), 256 routed experts, top-8.

## Method (and why single-process)
qlair's MPI rank == `cmg_id`, and A64FX has only **4 CMGs/node**, so real
multi-threaded MPI caps at 4 ranks and the thread path is fragile. Instead we
drive the **validated uTofu put primitive** directly: a rank's all-to-all
*egress* is `(N-1)` puts of the per-pair size, timed from the guest cycle
counter (`CNTVCT_EL0`). This scales to any N, stays memory-light (two 1 MB
buffers — well under the per-node budget), and charges the **same** calibrated
cost the 2-node test validated against the model exactly. Reported:
- **ser** = serialized over 1 TNI (`(N-1)` puts back-to-back);
- **par** = divided by `min(6, N-1)` TNIs (A64FX has 6 one-sided TNIs → up to 6
  concurrent transfers per rank).

What this is **not**: qlair has no multi-node Tofu *topology* (hop counts, link
contention, recursive-doubling tree), so this is the per-rank **volume × wire
cost**, i.e. the hardware floor — not the full collective schedule.

## Results (run: `qlair --cores 2 -p moe_all2all.elf`)

| pattern (per-pair) | N | steps | serialized | parallel (6 TNI) | eff. BW |
|---|---|---|---|---|---|
| **decode dispatch** 6144 B | 4 | 3 | 9.0 µs | 3.0 µs | 6.1 GB/s |
| **decode dispatch** 6144 B | 8 | 7 | 14.5 µs | 2.4 µs | 17.9 GB/s |
| **decode combine** 6144 B | 4 | 3 | 4.8 µs | 1.6 µs | 11.5 GB/s |
| **decode combine** 6144 B | 8 | 7 | 14.2 µs | 2.4 µs | 18.2 GB/s |
| **prefill dispatch** c64 | 4 | 3 | 505 µs | 168 µs | 14.0 GB/s |
| **prefill dispatch** c64 | 8 | 7 | 510 µs | 85 µs | 32.4 GB/s |
| **attn all-reduce** 2048 B | 4 | 3 | 3.5 µs | 1.2 µs | 5.2 GB/s |
| **attn all-reduce** 2048 B | 8 | 7 | 10.5 µs | 1.8 µs | 8.2 GB/s |

(prefill per-pair = `chunk(64) × top8 / N × 6144 B` = 786 KB at N=4, 393 KB at N=8.)

## Reading the numbers
- **Decode is latency-bound.** A 6 KB int8 dispatch is one ~2.4 µs put; the
  all-to-all floors at the **~1.4 µs base** regardless of N (more ranks = more
  steps but they overlap across the 6 TNIs). Per MoE layer = dispatch + combine
  ≈ **5–6 µs (parallel)** of hardware wire time. Across 75 MoE layers that is a
  few hundred µs/token of MoE-comm floor — same order as the 2 attention
  all-reduces/layer (≈1–2 µs each).
- **Prefill is bandwidth-bound.** A 64-token chunk dispatches hundreds of KB per
  pair; effective BW climbs toward the 6.3 GB/s/TNI × min(6,N-1) aggregate
  (32 GB/s at N=8). This is where larger int8 (vs bf16, 2×) directly halves the
  dispatch time.
- **Caveat — the floor vs the measured decode.** `decode_sim.py`'s all-reduce
  term is ~26 ms, **~4 orders of magnitude** above these µs-scale floors, because
  the measured cost is dominated by the uTofu **robust-completion drain**
  (trailer-seq/civac/MRQ), not the wire. qlair confirms the floor for the MoE
  all-to-all just as it did for the AR; the drain still needs the real cluster.

## Files
- `moe_all2all.c` — the cost harness (parameterized N and per-pair size).
- `moe_stubs.S` — uTofu symbol stubs + `rd_cyc` (CNTVCT via `.inst`, now that
  clair supports it) + `myputn` (5-arg put wrapper around the 8-arg `utofu_put`).

Build/run (local dev host):
```
~/work/clair/clair/build/clair -O -t arm64 -f bin moe_all2all.c moe_stubs.S -o m.elf
~/work/clair/clair/build/qlair --cores 2 -p -n 30000000 m.elf
```
