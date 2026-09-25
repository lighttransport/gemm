# Qwen3.8-27B A64FX prefill pipeline

The MPI pipeline assigns contiguous mixer/FFN units to 2–12 nodes. Each
process uses 48 A64FX workers and keeps the full FP4 decode image resident;
only its assigned prefill matrix descriptors are built. A stage sends the
FP32 residual for each prompt chunk to the next stage. Successive prompts
can occupy different stages at the same time. The benchmark repeats the same
tokenized prompt with fresh KV and SSM state for each prompt.

The stage cuts minimize the difference between cumulative estimated MAC
counts. A mixer and its FFN can land on different nodes. The 3-node run on
job 51917132 exercised cuts at units 43 and 85 and produced the same final
residual hash as the single-node build.

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

The MPI build uses `-O2 -mcpu=a64fx` without `-ffp-contract=fast` to retain
the established decode correctness. An `-O3 -ffp-contract=fast` build on job
51917132 diverged from F32 at generated token 5 even on one node; the
baseline-precision MPI build matched 256/256 on one through four nodes.
Its single-node 1024-token prefill rate was 155.2 tok/s at chunk 480.

## Measurements, job 51917132

FP4 model, 1024-token prompt, eight repeated independent prompts. `steady`
is the seven inter-completion intervals at the final stage. `end-to-end`
includes pipeline fill and drain; `latency` is the first prompt's completion
time. State transfer and decode follow those timings when `DECODE=1`.

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

On four nodes at chunk 480, each stage spent about 12.3 s computing eight
prompts; upstream stages also spent about 4.7 s in blocking sends. Smaller
chunks reduce this pipeline wait but increase GEMM work. The best measured
chunk was 160. MPI nonblocking sends without a progress thread serialized the
single-prompt path and were removed.

The pipeline currently replicates the full model on each node and gathers
state to rank 0 for single-node decode. It does not yet use stage-sharded
weight images, pre-expanded int16 weights, uTofu handoff, or concurrent TP4
decode groups. Six- and twelve-node throughput remain unmeasured.
