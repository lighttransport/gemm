# Qwen3.8-27B A64FX prefill pipeline

The MPI pipeline assigns contiguous mixer/FFN units to 2–128 nodes (one rank
per mixer/FFN unit, the 128-unit maximum). Each
process uses 48 A64FX workers and keeps the full FP4 decode image resident;
only its assigned prefill matrix descriptors are built. A stage sends the
FP32 residual for each prompt chunk to the next stage. Successive prompts
can occupy different stages at the same time. The benchmark repeats the same
tokenized prompt with fresh KV and SSM state for each prompt.

The stage cuts minimize the difference between cumulative relative unit costs
calibrated from an 8-prompt, 1024-token, 12-node run. A mixer and its FFN can
land on different nodes. The 3-node run on job 51917132 exercised cuts at
units 43 and 85 and produced the same final residual hash as the single-node
build.

## Build and run on Fugaku

Stage the FP4 image at `/local/q38/fp4.image` on every allocated node and
make `/local/q38/tmp` on the first node. The 4-node stage hook used for job
51917132 is `tmp/q38-fast/stage_hook_4n.sh`. From the repository root on the
first compute node:

```sh
bash a64fx/llm/q38p/build_pp.sh
bash a64fx/llm/q38p/run_pp.sh 4 8 1024 160 1
```

The runner arguments are `NODES PROMPTS TOKENS CHUNK DECODE`. `DECODE=1`
gathers the final prompt's KV cache and SSM/conv state to rank 0, then runs
256 decode steps there and compares them with `/local/q38/ref-f32.log`.
For another prompt length, set `Q38P_REF` to its F32 reference log. Rank logs
are written under `tmp/q38p/pp_runs/`.

The standard scale sweep uses 8 independent prompts of 1024 tokens and chunk
160. The per-node target is 150 tok/s, giving targets of 900, 1200, and 1800
tok/s for 6, 8, and 12 nodes, and 3600 tok/s for 24 nodes. From an allocation
of at least 12 nodes, run the 6/8/12 configurations with:

```sh
bash a64fx/llm/q38p/run_pp.sh 6 8 1024 160 1
bash a64fx/llm/q38p/run_pp.sh 8 8 1024 160 1
bash a64fx/llm/q38p/run_pp.sh 12 8 1024 160 1
```

Ranks above 12 use the same command, up to 128 ranks. For example, a
24-node allocation runs `bash a64fx/llm/q38p/run_pp.sh 24 8 1024 160 1`.
An allocation must provide at least as many nodes as the first argument.
The partitioner keeps at least one mixer/FFN unit per rank and balances the
estimated MAC cost; throughput above 12 nodes still needs measurement.

Submit the bundled scale jobs from the Fugaku frontend:

```sh
pjsub --no-check-directory a64fx/llm/q38p/pjsub_pp12_v2.sh  # 6, 8, 12
pjsub --no-check-directory a64fx/llm/q38p/pjsub_pp24.sh      # 24
```

The MPI build uses `-O2 -mcpu=a64fx` without `-ffp-contract=fast` to retain
the established decode correctness. An `-O3 -ffp-contract=fast` build on job
51917132 diverged from F32 at generated token 5 even on one node; the
baseline-precision MPI build matched 256/256 on one through four nodes.
Its single-node 1024-token prefill rate was 155.2 tok/s at chunk 480.
An `int16_t` activation panel packing alias violation has since been fixed
with `memcpy`. With that fix, a single-node `-O3` build without fast math
passes 256/256 decode but remains near 154 tok/s; the earlier apparent
162 tok/s `-O3` result came from an incorrect panel. Keep the MPI build at
its validated `-O2` setting until the pipeline is retested with the fix.

## Measurements

FP4 model, 1024-token prompt, eight repeated independent prompts. `steady` is
the seven inter-completion intervals at the final stage. `end-to-end` includes
pipeline fill and drain; `latency` is the first prompt's completion time.
State transfer and decode follow those timings when `DECODE=1`.

| Nodes | Chunk | Steady tok/s | End-to-end tok/s | Latency s | Decode vs F32 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 160 | 275.1 | 272.5 | 4.01 | 256/256 |
| 3 | 160 | 411.3 | 400.3 | 3.04 | 256/256 |
| 4 | 480 | 482.7 | 442.7 | 3.66 | 256/256 at chunk 480, one prompt |
| 4 | 320 | 523.4 | 489.2 | 3.05 | residual hash stable |
| 4 | 240 | 541.5 | 511.8 | 2.77 | residual hash stable |
| 4 | 200 | 526.8 | 502.4 | 2.70 | residual hash stable |
| 4 | 160 | **544.9** | **521.7** | **2.55** | **256/256** |
| 4 | 120 | 535.8 | 515.8 | 2.50 | residual hash stable |

Six-, eight-, and twelve-node results from interactive job 51918651, using
chunk 160 and the calibrated partition, were:

| Nodes | Steady tok/s | End-to-end tok/s | Steady tok/s/node | First prompt s | Decode vs F32 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 802.7 | 743.0 | 133.8 | 2.10 | 256/256 |
| 8 | 1074.8 | 964.2 | 134.3 | 1.83 | 256/256 |
| 12 | 1562.8 | 1316.6 | 130.2 | 1.64 | 256/256 |

The target is 150 tok/s/node (900/1200/1800 steady tok/s at 6/8/12 nodes),
so these measurements are still below target. The initial 12-node partition
measured 1555.1 steady tok/s; fitting relative FFN, SSM-mixer, and attention-
mixer costs to per-rank compute times adjusted the cuts and raised the repeat
to 1562.8 tok/s. This 0.5% gain is small and does not explain the remaining
gap.

The 12-node chunk sweep (original partition, decode disabled) measured steady
rates of 1517.8, 1555.1, 1506.5, 1547.6, 1503.5, and 1388.3 tok/s for chunks
120, 160, 200, 240, 320, and 480. Chunk 160 remains the best setting. A
four-node smoke run after raising the rank limit measured 543.3 steady
(135.8 tok/s/node), 520.1 end-to-end, and matched 256/256 decode tokens.

On four nodes at chunk 480, each stage spent about 12.3 s computing eight
prompts; upstream stages also spent about 4.7 s in blocking sends. Smaller
chunks reduce this pipeline wait but increase GEMM work. The best measured
chunk was 160. MPI nonblocking sends without a progress thread serialized the
single-prompt path and were removed.

The pipeline currently replicates the full model on each node and gathers
state to rank 0 for single-node decode. It does not yet use stage-sharded
weight images, pre-expanded int16 weights, uTofu handoff, or concurrent TP4
decode groups. Throughput above 12 nodes remains unmeasured.
