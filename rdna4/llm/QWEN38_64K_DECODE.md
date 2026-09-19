# Qwen3.8 sustained decode at a 64K context offset

The runner has a synthetic depth benchmark for measuring decode after the
attention cache has grown. It clears the first `depth` quantized K/V rows and
their scale rows on the GPU, resets recurrent state, then runs the measured
prompt and decode at that position. Cache preparation and model loading are
outside the timed region.

This differs slightly from `llama-bench -d`: the pinned llama.cpp benchmark
runs random tokens to create the depth and saves/restores the resulting
context. `--bench-depth` deliberately creates zero K/V rows without running
64K prompt tokens. Qwen3.8 recurrent state therefore starts from reset. The
benchmark isolates the cost of scanning a deep K/V cache; it is not a
long-prompt quality evaluation.

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

For IQ3, set `QWEN38_MODEL` to the IQ3_XXS file and set the current regression
floor with `QWEN38_GSQ_64K_FLOOR_TPS=26.5`. The script's default 27.5 tok/s
floor is an IQ2 regression gate, not the open 40 tok/s performance goal.

## RDNA4 result

RX 9070 XT / gfx1201 / ROCm 10, Q8 K and Q8 V, greedy sampling:

| Model | 512-token repeats | Minimum | Free VRAM | Sequence hash |
|---|---:|---:|---:|---|
| IQ2_XS | 28.04 / 27.98 / 27.94 tok/s | 27.94 | 4434 MiB | `1a74985dead45082` |
| IQ3_XXS | 26.99 / 26.95 / 26.94 tok/s | 26.94 | 770 MiB | `465c934d56046d85` |

All three repeats for each model produced the same complete 512-token hash.
The preceding IQ2 implementation sustained 26.79--26.93 tok/s. The retained
long-context scheduling raises its minimum by 4.3%.

At 16K and longer, native Q8 attention uses up to 128 splits and submits the
decode grid in split-major order. This keeps blocks that read the same GQA K/V
group close in the launch order. On the 64K operator test, 128 splits took
578.648 microseconds per attention layer; 16, 32, 64, and 256 splits took
718.924, 637.534, 584.820, and 624.078 microseconds. The runner also avoids
allocating F16 Q/K/V packing buffers when native Q8/Q8 decode and prefill are
both selected, recovering about 266 MiB.

The current 40 tok/s sustained target is unmet. A kernel trace attributes
about 9.8 ms/token to the 16 attention layers at 64K and about 26 ms/token to
the projection, SSM, normalization, and output path. Removing all attention
work would still only approach the target, so the next useful step is reducing
weight traffic or making verified multi-token decode faster. The existing
dense NextN path reached only 21.21 tok/s on this synthetic prefix with draft
width three (168 accepted of 259 proposed tokens), so it remains opt-in.

## Correctness evidence

`make -C rdna4/llm reference-attention-test` compares the runner kernel with
the pinned llama.cpp HIP kernel. The final run passed 39,567,360 bitwise Q8/Q8
output comparisons, including 64K zero-cache cases at 16, 32, 64, 128, and
256 matching split counts. Short-context split selection and arithmetic are
unchanged.

Fresh 4096-token C++ coding-task validations cover the normal semantic path.
For IQ2 and IQ3, greedy and temperature-0.6 responses match pinned llama.cpp
token IDs, EOS, and output bytes. Each generated program passes fixed edge
cases and 10,000 randomized cases. Artifacts are in
`tmp/qwen38/depth64-final-iq2-v2/` and `tmp/qwen38/depth64-final-iq3/`.
