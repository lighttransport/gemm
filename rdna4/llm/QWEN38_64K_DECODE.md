# Qwen3.8 sustained decode at a 64K context offset

The runner's depth benchmark now follows `llama-bench -d`: it runs random
tokens through the complete model to create the K/V cache and recurrent state,
then saves and restores that state outside the timed region. Prefix K/V rows
remain device-resident; each repeat overwrites the same measured suffix.

The random stream uses the same `rand() % n_vocab` construction and conditional
BOS insertion as the pinned llama.cpp benchmark. The runner explicitly seeds
libc `rand()` with one so the stream and its hash are reproducible. This is a
model-processed synthetic context, rather than a semantic long-prompt quality
evaluation.

## Reproduce

Build the runner, then run the three-repeat IQ2 gate:

```sh
make -C rdna4/llm test_hip_llm
rdna4/llm/bench_qwen38_gsq_decode_64k.sh
```

The explicit equivalent is one measured prompt token at position 65,536,
followed by 512 generated tokens in a 66,560-token allocation:

```sh
bash rdna4/llm/run_qwen38_gsq_rocm.sh \
  --gpu-only-bench -n 1 -s 66560 --ubatch 512 \
  --kv-cache q8q8 --qwen35-prefill-bf16 --qwen35-decode-graph \
  --qwen35-native-q8-prefill --qwen35-native-mmvq \
  --sampling-profile llama --temp 0 --seed 42 \
  --decode 512 --bench-ignore-eos --bench-depth 65536 --bench-repeat 3
```

The script's default 26.5 tok/s floor is an IQ2 random-depth regression gate,
not the open 40 tok/s performance goal. For IQ3, set `QWEN38_MODEL` to the
IQ3_XXS file and choose a floor after recording a full random-depth baseline.

## RDNA4 result

RX 9070 XT / gfx1201 / ROCm 10, Q8 K and Q8 V, greedy sampling:

| Path | Measured suffix | Decode tok/s | Prefix tok/s | Free VRAM | Sequence hash |
|---|---:|---:|---:|---:|---|
| IQ2_XS ordinary, original exact gate | 3 x 512 tokens | 26.92 / 26.91 / 26.90 | 142.36 | 4284 MiB | `b01a17fae16f806d` |
| IQ2_XS + DFlash2 K=7, optimized | 256 tokens | 44.94 | 443.41 | 1274 MiB | `2ddd068dca63669a` |

Both runs use the same fully processed 65,536-token random prefix with token
hash `90178de69a24a76e`.  The ordinary row records the original exact gate.  Its
prefix took 460.37 seconds before long-context WMMA prefill was enabled.  The
optimized DFlash run processed the prefix in 147.801 seconds, then generated
256 tokens in 5.696 seconds.  It drafted 259 tokens, accepted 217, and spent
667.500/4936.944/56.114 ms in draft/verify/commit.  The earlier 27.94 tok/s
result used zero cache rows and is not comparable.

At 16K and longer, native Q8 attention uses up to 128 splits and submits the
decode grid in split-major order. This keeps blocks that read the same GQA K/V
group close in the launch order. On the 64K operator test, 128 splits took
578.648 microseconds per attention layer; 16, 32, 64, and 256 splits took
718.924, 637.534, 584.820, and 624.078 microseconds. The runner also avoids
allocating F16 Q/K/V packing buffers when native Q8/Q8 decode and prefill are
both selected, recovering about 266 MiB.

The exact DFlash verifier now evaluates up to eight adjacent causal queries in
one four-wave block.  Each old K/V row is loaded once, while each query retains
its own Q8_1 quantization, online-softmax state, packed-F16 accumulation and
split-combine ordering.  At 64K and eight splits, the eight-query attention
operator takes 3.076784 ms per layer, compared with 8.653874 ms for the prior
generic verifier at the same split count.  The DFlash path therefore clears
the 40 tok/s sustained target without approximating authoritative target
output.

Ordinary one-token decode remains about 29.13 tok/s in the latest retained
run, so its 40 tok/s target remains open. A prior trace attributes about
9.8 ms/token to the 16 attention layers at 64K and about 26 ms/token to the
projection, SSM, normalization, and output path. Reducing projection weight
traffic and recurrent-state work is now more useful than attention-only work.
Dense NextN remains opt-in; its previous 64K result used the removed zero-cache
setup and must be remeasured before making a random-depth performance claim.

## Correctness evidence

`make -C rdna4/llm reference-attention-test` compares the runner kernel with
the pinned llama.cpp HIP kernel. The final run passed 46,743,552 bitwise Q8/Q8
output comparisons.  It covers the shared-K/V verifier with 2, 7 and 8 queries
at short context, plus real random-like 64K K/V patterns at 8, 12, 16, 32, 64,
128 and 256 matching split counts. The full benchmark separately exercises
real random K/V and recurrent state. The optimized DFlash run retains the
earlier suffix hash despite changing the verifier scheduling.

Fresh 4096-token C++ coding-task validations cover the normal semantic path.
For IQ2 and IQ3, greedy and temperature-0.6 responses match pinned llama.cpp
token IDs, EOS, and output bytes. Each generated program passes fixed edge
cases and 10,000 randomized cases. Artifacts are in
`tmp/qwen38/depth64-final-iq2-v2/` and `tmp/qwen38/depth64-final-iq3/`.
