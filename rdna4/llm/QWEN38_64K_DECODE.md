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

| Model | 512-token repeats | Minimum | Free VRAM | Sequence hash |
|---|---:|---:|---:|---|
| IQ2_XS | 26.92 / 26.91 / 26.90 tok/s | 26.90 | 4284 MiB | `b01a17fae16f806d` |

The 65,536-token random prefix took 460.37 seconds at 142.36 tok/s and has token
hash `90178de69a24a76e`. All three measured repeats produced the same complete
512-token hash. The earlier 27.94 tok/s result used zero cache rows and is not
comparable; it has been superseded by this model-processed depth result.

At 16K and longer, native Q8 attention uses up to 128 splits and submits the
decode grid in split-major order. This keeps blocks that read the same GQA K/V
group close in the launch order. On the 64K operator test, 128 splits took
578.648 microseconds per attention layer; 16, 32, 64, and 256 splits took
718.924, 637.534, 584.820, and 624.078 microseconds. The runner also avoids
allocating F16 Q/K/V packing buffers when native Q8/Q8 decode and prefill are
both selected, recovering about 266 MiB.

The current 40 tok/s sustained target is unmet. A prior kernel trace attributes
about 9.8 ms/token to the 16 attention layers at 64K and about 26 ms/token to
the projection, SSM, normalization, and output path. Removing all attention
work would still only approach the target, so the next useful step is reducing
weight traffic or making verified multi-token decode faster. The dense NextN
path remains opt-in; its previous 64K result used the removed zero-cache setup
and must be remeasured before making a random-depth performance claim.

## Correctness evidence

`make -C rdna4/llm reference-attention-test` compares the runner kernel with
the pinned llama.cpp HIP kernel. The final run passed 39,567,360 bitwise Q8/Q8
output comparisons, including 64K zero-cache cases at 16, 32, 64, 128, and
256 matching split counts. The full benchmark above separately exercises real
random K/V and recurrent state. Short-context split selection and arithmetic
are unchanged.

Fresh 4096-token C++ coding-task validations cover the normal semantic path.
For IQ2 and IQ3, greedy and temperature-0.6 responses match pinned llama.cpp
token IDs, EOS, and output bytes. Each generated program passes fixed edge
cases and 10,000 randomized cases. Artifacts are in
`tmp/qwen38/depth64-final-iq2-v2/` and `tmp/qwen38/depth64-final-iq3/`.
