# 2-node uTofu communication test on the qlair A64FX simulator

qlair **does** model the Tofu interconnect (`~/work/clair/clair/tools/qlair/tofu/`):
a uTofu RDMA simulator (`TofuSimulator`) intercepting the full uTofu API
(`utofu_create_vcq` / `utofu_reg_mem` / `utofu_put` / `utofu_poll_tcq` /
`utofu_poll_mrq` / VBG barrier+reduce / MPI collectives) plus a **latency model
calibrated from real Fugaku 2-node measurements**:

```
put/get latency_ns = 1400 + bytes / 6.3        (6.3 GB/s per TNI)
utofu_put API call  = 460 cycles (~230 ns); transfer charged on poll_tcq success
```

It is intra-node (the two "ranks" are CMGs/threads sharing one address space, so
data moves by memcpy) — there is **no multi-node Tofu *topology*** (hop counts,
link contention). But the per-transfer **latency/bandwidth cost** is modeled and
charged into the cycle counter, which is exactly the term `decode_sim.py` needs.

## Files
- `utofu_2node_put.c` — minimal 2-endpoint test: create two VCQs, register a
  buffer on each, `utofu_put` rank0→rank1, `utofu_poll_tcq` for completion.
- `utofu_2node_latency.c` — sweeps message size and times each `put+poll` via the
  guest cycle counter (`CNTVCT_EL0`), printing latency + effective bandwidth.
- `utofu_stubs.S` — symbol stubs qlair intercepts by name. Needed because:
  1. clair drops empty C stubs, so uTofu/pthread symbols are provided in asm;
  2. clair's C frontend remaps `pthread_create` to a fake `clair_pthread_create`
     that returns EAGAIN (no thread) — the `qlair_spawn`/`qlair_join` shims
     tail-call the **raw** `pthread_*` symbols so qlair's ThreadManager fires;
  3. clair mis-lowers 8-argument calls — `myput`/`myputn` are 5-arg wrappers that
     set `edata`/`flags`/`cbdata` in registers and tail-call `utofu_put`;
  4. clair's assembler rejects `mrs`/`isb`/`.inst`, so `CNTVCT_EL0` is emitted as
     `.word 0xd53be040`.

## Build & run (local dev host, not Fugaku)
```
cd ~/work/clair/clair
./build/clair -O -t arm64 -f bin /path/utofu_2node_latency.c /path/utofu_stubs.S -o u.elf
./build/qlair --cores 2 -P -p -n 6000000 u.elf      # --cores>=2 enables TofuSimulator
```
`--cores 2` is what turns the Tofu simulator on (`num_cores >= 2`).

## Result (qlair, matches the calibrated model)
```
uTofu put+poll latency (qlair, 6 TNIs, model: 1400ns + 6.3GB/s/TNI):
      64 B:   1413 ns/put  (  1.41 us )    0.04 GB/s
    1024 B:   1569 ns/put  (  1.56 us )    0.65 GB/s
    4096 B:   2057 ns/put  (  2.05 us )    1.99 GB/s
   16384 B:   4007 ns/put  (  4.00 us )    4.08 GB/s
   65536 B:  11809 ns/put  ( 11.80 us )    5.54 GB/s
```
Small-message latency floors at the **~1.4 µs base**; bandwidth climbs toward the
**6.3 GB/s/TNI** asymptote — reproducing the documented model from the guest's own
`CNTVCT_EL0`.

## Relevance to decode_sim.py
This is the **hardware floor** for a single put (~1.4 µs). The decode all-reduce
cost in `decode_sim.py` is ~26 ms — **~4 orders of magnitude larger** — because it
is dominated by the *robust-completion drain* (trailer-seq / civac / MRQ), not the
wire transfer. qlair confirms the floor; the drain overhead is the real lever, and
still needs the cluster (qlair has no multi-node topology/contention model).
