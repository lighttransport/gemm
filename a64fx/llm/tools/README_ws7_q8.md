# Single-node analysis tools added 2026-07-10 (session: prefill/decode optimization)

| tool | question it answers | result |
|---|---|---|
| `ws7_tp_q8_test.c` | is WS7 (TP_ATTN×Q8 wrong output) a kernel bug? | NO — fp reassoc class (1.5e-7 == bf16's 2.8e-7); row-shard bit-exact; contraction shards need 64-aligned boundaries (8-aligned = 1.5e-3 quant-block straddle) |
| `mhc_bench.c` | where do hc_pre's ~118 µs/call go? | hcmix scalar dot 96-110 µs (16 GB/s) + serial sinkhorn 24-32 µs; SVE half-row hcmix → ~35 µs, SVE sinkhorn → 5.2 µs bit-exact |
| `q8_mv_bw.c` | is the M=1 Q8 dense matvec BW- or issue-bound? | ISSUE-bound: q8 402 GB/s at the same wall time as bf16-pv's 735 GB/s (≈390 Gmac/s kernel ceiling); v2 svmla_lane restructure 2× slower (refuted) |

Build (all): `fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -D_GNU_SOURCE -I../../common -o build/<name> tools/<name>.c -lm -lpthread -lhwb`, run on cores 12-59.
