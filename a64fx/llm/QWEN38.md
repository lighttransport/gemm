# Qwen3.8-27B on A64FX

The Qwen loader treats `qwen35.block_count=65` as 64 autoregressive trunk
blocks plus one native NextN block. Split GGUF input is accepted by passing any
`-00001-of-00002.gguf` shard; tensor metadata and mappings are merged without
copying the files.

Build:

```sh
make -C a64fx/llm qwen38_runner qwen38_pp_runner CC=fcc OPENMP=1
```

Single-node Q4 (anonymous HBM upload, with source-cache eviction):

```sh
a64fx/llm/run_qwen38.sh \
  --model ~/models/qwen38/27b/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  --mode single --nodes 1 -- --prompt 'Hello' --max-gen 16 --spec-k 1
```

Pipeline parallel Q8 on two nodes or Q8/BF16 on twelve nodes:

```sh
a64fx/llm/run_qwen38.sh --model MODEL.gguf --mode pp --nodes 12 -- \
  --prompt 'Hello' --max-gen 16 --spec-k 1
```

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
- Q8 and BF16 PP still need live inference runs; their metadata/load paths pass.
- TP remains disabled because the legacy runner references a removed tensor/KV
  slicing API. It must not be presented as working until that path is ported.
