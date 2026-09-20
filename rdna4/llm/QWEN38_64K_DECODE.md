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

The script's default 32.0 tok/s floor is an IQ2 random-depth regression gate,
not the open 40 tok/s performance goal. For IQ3, set `QWEN38_MODEL` to the
IQ3_XXS file and choose a floor after recording a full random-depth baseline.

## RDNA4 result

RX 9070 XT / gfx1201 / ROCm 10, Q8 K and Q8 V, greedy sampling:

| Path | Measured suffix | Decode tok/s | Prefix tok/s | Free VRAM | Sequence hash |
|---|---:|---:|---:|---:|---|
| IQ2_XS ordinary, original exact gate | 3 x 512 tokens | 26.92 / 26.91 / 26.90 | 142.36 | 4284 MiB | `b01a17fae16f806d` |
| IQ2_XS ordinary, exact three-head K/V reuse | 512 tokens | 32.56 | 444.31 | 4282 MiB | `051e7338c23a544e` |
| IQ2_XS ordinary, staged IQ codebooks | 512 tokens | 33.31 | 443.44 | 4282 MiB | `051e7338c23a544e` |
| IQ2_XS ordinary, hoisted GQA scales | 256 tokens | 34.08 | 445.67 | 4282 MiB | `f4b35758fb99e6db` |
| IQ2_XS ordinary, reused packed probabilities | 256 tokens | 34.21 | 443.63 | 4282 MiB | `f4b35758fb99e6db` |
| IQ2_XS ordinary, shape-tuned IQ blocks | 256 tokens | 34.98 | 441.44 | 4282 MiB | `f4b35758fb99e6db` |
| IQ2_XS + DFlash2 K=7, optimized | 256 tokens | 47.72 | 445.71 | 1274 MiB | `2ddd068dca63669a` |
| IQ2_XS + DFlash2 K=7, packed Q4_K/Q8_1 draft | 256 tokens | 49.74 | 443.57 | 1274 MiB | `2ddd068dca63669a` |

All runs use the same fully processed 65,536-token random prefix with token
hash `90178de69a24a76e`.  The ordinary row records the original exact gate.  Its
prefix took 460.37 seconds before long-context WMMA prefill was enabled.  The
optimized DFlash run processed the prefix in 147.039 seconds, then generated
256 tokens in 5.365 seconds.  It drafted 259 tokens, accepted 217, and spent
668.717/4602.977/57.440 ms in draft/verify/commit.  The earlier 27.94 tok/s
result used zero cache rows and is not comparable.

The packed draft run processed the prefix in 147.747 seconds and generated the
same suffix in 5.147 seconds. Its 474.117/4585.996/56.181 ms
draft/verify/commit split shows that Q4_K/Q8_1 packed dots removed 194.600 ms
from proposal work without changing acceptance or the authoritative suffix.

At 16K and longer, native Q8 attention uses an exact three-head GQA kernel.
Four waves load each K/V row once and update three independent query heads;
each head retains llama.cpp's dot, online-softmax, packed-F16 accumulation and
split-combine order. The decode graph records both the short and long kernels,
which gate themselves from the device position so graph capture at position
zero cannot freeze the short path. The kernel now also computes each K/Q scale
product once per four adjacent packed dots rather than repeating it for every
dot. It also loads and packs each head's probability once for both
128-dimension value tiles, removing half of those LDS reads without changing
any packed-F16 FMA. The dot and accumulation sequence is unchanged. At 64K
and 128 splits, the complete attention operator fell from about 361 to 348
microseconds per layer after the scale hoist, then to 321.8--323.6
microseconds after probability reuse (and from about 606 microseconds before
GQA reuse). The runner also avoids allocating F16 Q/K/V packing buffers when
native Q8/Q8 decode and prefill are both selected, recovering about 266 MiB.

The exact DFlash verifier now evaluates up to eight adjacent causal queries in
one four-wave block.  Each old K/V row is loaded once, while each query retains
its own Q8_1 quantization, online-softmax state, packed-F16 accumulation and
split-combine ordering.  At 64K and eight splits, the eight-query attention
operator takes 3.076784 ms per layer, compared with 8.653874 ms for the prior
generic verifier at the same split count.  The DFlash path therefore clears
the 40 tok/s sustained target without approximating authoritative target
output.

Ordinary one-token decode now reaches 34.98 tok/s in the latest retained run,
so its 40 tok/s target remains open. The 16 attention layers now cost about
5.7 ms/token at 64K; the remaining projection, SSM, normalization and output
path is about 23 ms/token. Reducing projection weight traffic and recurrent
state work is now more useful than further attention-only work.

The one-row IQ scheduler now uses sixteen waves for IQ2_XS, sixteen for the
17408-row IQ2_S projection, and four for the 5120/17408-row IQ3_S shapes.
Each wave still owns one output row, so the arithmetic is unchanged. Matched
64-row traces predict a 0.23 ms/token reduction compared with eight waves for
all three formats. The tuned and fallback paths are bitwise identical across
all 248,320 final logits (SHA-256 `5b5f2f1a...953c0`), and both retain the
zero-depth and 64K suffix hashes. Set the diagnostic
`LLM_QWEN35_IQ_SHAPE_THREADS=0` to restore eight waves for the A/B.

The scalar IQ2_XXS, IQ2_XS and IQ3_XXS kernels stage their 1--4 KiB decode
tables in LDS once per block. IQ3_XXS uses four waves because eight or more
waves lose occupancy after staging. The change retains the short-context
sequence hash and raises a 512-token zero-depth run from about 40.7 to
41.9--42.0 tok/s. IQ3_S staging was measured and rejected because its 2 KiB
table slowed the important projection shapes.
Dense NextN remains opt-in; its previous 64K result used the removed zero-cache
setup and must be remeasured before making a random-depth performance claim.

## Correctness evidence

`make -C rdna4/llm reference-attention-test` compares the runner kernel with
the pinned llama.cpp HIP kernel. The final run passed 49,188,864 bitwise Q8/Q8
output comparisons. It covers the adaptive ordinary decode graph, the
three-head K/V-reuse kernel, and the shared-K/V verifier with 2, 7 and 8 queries
at short context, plus real random-like 64K K/V patterns at 8, 12, 16, 32, 64,
128 and 256 matching split counts. The full benchmark separately exercises
real random K/V and recurrent state. The optimized DFlash run retains the
earlier suffix hash despite changing the verifier scheduling.

The exact projection differential separately passes 2,948,352 activation
values and 13,191,360 Q2_K/IQ outputs.  It covers the compact fixed-eight
Q2_K, IQ2_XXS, IQ2_XS, IQ2_S and IQ3_S schedules, the retained IQ3_XXS
schedule, and the exact IQ4_XS eight-row reduction.

Fresh 4096-token C++ coding-task validations cover the normal semantic path.
For IQ2 and IQ3, greedy and temperature-0.6 responses match pinned llama.cpp
token IDs, EOS, and output bytes. Each generated program passes fixed edge
cases and 10,000 randomized cases. Artifacts are in
`tmp/qwen38/depth64-final-iq2-v2/` and `tmp/qwen38/depth64-final-iq3/`.
