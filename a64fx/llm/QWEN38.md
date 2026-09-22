# Qwen3.8-27B on A64FX

The Qwen loader treats `qwen35.block_count=65` as 64 autoregressive trunk
blocks plus one native NextN block. Split GGUF input is accepted by passing any
`-00001-of-00002.gguf` shard; tensor metadata and mappings are merged without
copying the files.

Build:

```sh
mkdir -p tmp/fcc
TMPDIR="$PWD/tmp/fcc" \
  make -C a64fx/llm qwen38_runner qwen38_pp_runner CC=fcc OPENMP=1
```

When `uname -m` reports `aarch64` on an A64FX host, compile and run the A64FX
binary natively there. Do not cross-build with the login host's GCC: its SVE
headers and target runtime may not match. Fujitsu `fcc -Nclang` needs a writable
temporary directory, so point `TMPDIR` at `tmp/fcc` in this repository rather
than `/tmp`. The resulting executable should report `ARM aarch64` from `file`
and can be invoked directly through the launchers below.

Single-node Q4 (anonymous HBM upload, with source-cache eviction):

```sh
a64fx/llm/run_qwen38.sh \
  --model ~/models/qwen38/27b/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  --mode single --nodes 1 -- --prompt 'Hello' --max-gen 16 --spec-k 1
```

Single-node Q8 llama-bench-style benchmark:

```sh
MODEL=/home/u14346/models/qwen38/27b/Qwen3.8-27B-Q8_0.gguf
STAGE=/local/u14346/qwen38
sh a64fx/llm/stage_gguf_shards.sh "$MODEL" "$STAGE"
make -C a64fx/llm qwen38_runner CC=fcc OPENMP=1

OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores \
numactl --interleave=all a64fx/llm/build/qwen38_runner \
  "$STAGE/Qwen3.8-27B-Q8_0.gguf" --threads 48 --max-seq 1024 \
  --q8-mode reference --bench --bench-prompt 8,32,128 \
  --bench-gen 32,128 --bench-warmup 1 --bench-runs 3
```

`--bench-prompt` and `--bench-gen` accept comma-separated token counts. The
runner prints llama-bench-style `pp` (prompt/prefill) and `tg` (generation)
throughput, reuses the resident model between cases, and resets recurrent
runtime state between repetitions. Use `--bench-csv` for a machine-readable
header and rows. Benchmark mode currently requires `--spec-k 0` so speculative
draft work does not mix with the plain prefill/decode rates.

### Four-CMG Q8 decode

Use the dedicated launcher for the packed K-major Q8 path. It stages the GGUF
under `/local`, allocates one 2 MiB-hugepage arena for the 27.2 GB decode
weights, binds it across NUMA nodes 4-7, and pins one twelve-core worker group
to each A64FX CMG:

```sh
a64fx/llm/run_qwen38_q8_cmg4.sh \
  /local/u14346/qwen38/Qwen3.8-27B-Q8_0.gguf \
  --prompt Hello --max-gen 16 --max-seq 128

a64fx/llm/run_qwen38_q8_cmg4.sh \
  /local/u14346/qwen38/Qwen3.8-27B-Q8_0.gguf \
  --max-seq 256 --bench --bench-prompt 8 --bench-gen 32 \
  --bench-warmup 1 --bench-runs 3 --bench-csv
```

The launcher sets hugepage fallback to zero and the runner checks the resulting
mapping, so a run fails instead of silently using base pages or one-CMG
placement. CMG4 mode also rejects GGUF paths outside `/local`: shared-storage
weights are always staged node-locally before conversion into the resident
HBM2 arena. The current 64-output by 128-input SVE kernel preserves the Q8_0
bytes and FP16 scales while changing only FP32 reduction grouping.

CMG4 uses four independent NUMA-aware allocations rather than subranges of one
recycled heap allocation. Allocation 0/1/2/3 is bound to node 4/5/6/7 and is
consumed only by worker IDs 0-11/12-23/24-35/36-47, pinned to CPUs
12-23/24-35/36-47/48-59. Each tensor is split at complete 64-row packed-group
boundaries; segment starts are 256-byte aligned. `NUMA_REPORT=1` enables the
runtime tensor-owner check and must report zero errors.

On the development node, the isolated kernel reached 110.8 GB/s on one CMG and
349.8 GB/s over 48 cores. The first integrated one-token check generated the
same `,` token as the reference Q8 path. Initial end-to-end decode measured
0.67 tok/s versus 0.41 tok/s for reference Q8. Fusing the DeltaNet Q/K
normalization, expansion, and scalar preparation phases removed three pool
barriers per SSM layer and raised the one-token check to 1.42 tok/s. Profiling
still attributes most remaining time to DeltaNet/SSM preparation rather than
weight streaming. Thus the original 20-30 tok/s value remains a bandwidth
roofline target, not an achieved runner result.

Pipeline parallel Q8 on two nodes or Q8/BF16 on twelve nodes:

```sh
a64fx/llm/run_qwen38.sh --model MODEL.gguf --mode pp --nodes 12 -- \
  --prompt 'Hello' --max-gen 16 --spec-k 1
```

Tensor parallel Q8/BF16 uses FFN, SSM V-head, and vocabulary sharding. Attention
is also sharded when KV heads divide the rank count (two-node Q8); for twelve
nodes it is replicated because Qwen3.8 has four KV heads and the query-only
replicated-KV path is not greedy-exact.

```sh
a64fx/llm/run_qwen38.sh --model MODEL.gguf --mode tp --nodes 12 -- \
  --token-id 1 --max-gen 1 --spec-k 1
```

On a fresh allocation, add `--stage-dir /local/qwen38/27b` before the runner
options. The launcher fans the source file (and every split shard) out once per
node using direct I/O for whole-MiB blocks, fsyncs the small tail, and reuses an
existing same-size local file. This avoids accumulating source and dirty
destination pages in the 32 GB HBM page cache.

PP assigns byte-balanced contiguous trunk ranges. Each rank lazily maps the
GGUF and faults only its range; rank 0 owns embedding work and the last rank
owns final normalization, vocabulary projection, and NextN. This avoids ever
materializing the 51 GB split BF16 model on one 32 GB node.

`--spec-k` accepts 0 through 4. The runner reports the first-draft greedy match
rate (`alpha`) while recurrently exercising all requested draft steps.

Current validation:

- Q4, Q8, and split BF16 metadata: 866 tensors, trunk 64, NextN 1.
- Q4 anonymous single-node greedy decode: completed on A64FX.
- Q4 native NextN execution: completed with K=1 and greedy comparison.
- Two-node Q4 PP completed on the live allocation with byte-balanced ranges
  `[0,32)` and `[32,64)` (380.906 s correctness run).
- Q8 two-node TP completed with 12 Q heads, two KV heads, FFN 8704/rank.
- Q8 twelve-node PP completed with ranges beginning `[0,5)` and generated
  `Please` from token ID 1 (549.864 s for two serial pipeline passes).
- Q8 twelve-node TP completed; the correctness-first non-divisible-KV topology
  reproduces token ID 3165 while sharding FFN/SSM/vocab.
- BF16 twelve-node PP completed from split GGUF and generated `Please` from
  synthetic token ID 1 (823.355 s for two serial pipeline passes).
- BF16 twelve-node TP completed with four SSM V heads/rank, 113 reductions per
  token, and the same second token ID (`3165`) as Q8/two-node controls.
- Native TP MTP K=1 was exercised on both Q8 and BF16; the corrected topology
  reported its first-draft greedy miss as `0/1` for this one-token prefix.
- Q8 two-node TP from node-local storage reproduced token ID `3165`; the
  one-token baseline measured 33.177 s prefill and 26.203 s decode. The current
  `transformer_build_panels()` hook is a no-op, so `TF_NO_PANEL=0` is not an
  optimization (the parity trial measured 25.987/26.923 s).

## Module performance measurement

Set `TF_MODULE_PROFILE=1` for a compact bottleneck report from the Qwen runner.
It enables the existing low-level stage timers during both prompt processing
and generation, then prints total prefill/decode throughput plus milliseconds
per token for attention QKV/output, SSM input/prepare/core/output, FFN
gate/up/down, LM head, and aggregate matrix work:

```sh
TF_MODULE_PROFILE=1 TF_KV_DTYPE=f16 \
  a64fx/llm/build/qwen38_runner MODEL.gguf \
  --prompt 'Return only code.' --max-gen 32 --max-seq 65536
```

The prefill line is captured before decode counters are reset. Use the same
prompt, context, thread count, and cache dtype when comparing kernel changes.
