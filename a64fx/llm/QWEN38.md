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

Tensor parallel Q8/BF16 uses FFN, SSM V-head, and vocabulary sharding. Attention
is also sharded when KV heads divide the rank count (two-node Q8); for twelve
nodes it is replicated because Qwen3.8 has four KV heads and the query-only
replicated-KV path is not greedy-exact.

```sh
a64fx/llm/run_qwen38.sh --model MODEL.gguf --mode tp --nodes 12 -- \
  --token-id 1 --max-gen 1 --spec-k 1
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
- Q8 two-node TP completed with 12 Q heads, two KV heads, FFN 8704/rank.
- Q8 twelve-node TP completed; the correctness-first non-divisible-KV topology
  reproduces token ID 3165 while sharding FFN/SSM/vocab.
- BF16 twelve-node PP completed from split GGUF and generated `Please` from
  synthetic token ID 1 (823.355 s for two serial pipeline passes).
- BF16 twelve-node TP completed with four SSM V heads/rank, 113 reductions per
  token, and the same second token ID (`3165`) as Q8/two-node controls.
- Native TP MTP K=1 was exercised on both Q8 and BF16; the corrected topology
  reported its first-draft greedy miss as `0/1` for this one-token prefix.
