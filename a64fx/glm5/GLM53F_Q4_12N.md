# GLM-5.3-Flash Q4 routed experts on 12 A64FX nodes

Run in an allocated interactive job with 12 MPI ranks:

```sh
bash a64fx/glm5/run_glm53f_q4_12n.sh
```

The launcher uses `mpifcc -Nclang`, node-local `/local` staging, and the
allocation's freshly generated uTofu topology. `GLM53F_Q4_MODEL` overrides
the first GGUF shard (default: `~/models/glm53f-q4/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf`).
The underlying shared launcher retains historical Q2 labels in its logs.

This is a hybrid: routed experts use the GGUF's Q4_K, Q5_K and Q6_K tensors;
shared experts and the compact attention/dense core use the existing
safetensors-derived images. It is not a fully GGUF-backed model.

The routed image occupies 15,456,534,528 bytes per rank. Each expert has eight
256-column parts distributed over 12 ranks, preserving quantization blocks.
Restaging reads bounded chunks, syncs output, drops file cache, and atomically
publishes complete images. Use a separate Q4 directory; the legacy reuse
check does not identify the source model.

Q4_K and Q5_K use SVE SDOT with affine minimum correction. Q6_K uses signed
six-bit values and per-16-value scales. Activations are quantized to Q8 in
256-value blocks. The fused kernels do not expand complete weight matrices.

Numerical validation:

```sh
export OPAL_PREFIX=/opt/FJSVxtclanga/tcsds-1.2.43
export MPI_HOME=$OPAL_PREFIX TMPDIR=/local
mpifcc -Nclang -O3 -march=armv8.2-a+sve -fopenmp \
  a64fx/glm5/test_glm53f_kquant.c -lm -o /local/test_glm53f_kquant
/local/test_glm53f_kquant
```

The test compares 300 randomized 256/4096-column rows against independently
dequantized weights and the same Q8 activation values. Both conservative and
fast-math builds pass; worst error normalized by the absolute term sum is
3.05e-8 or less. This checks fused arithmetic, not activation quantization loss
or whole-model semantic correctness.

## Measurements

Job 51351748: packed Q4/Q5 SVE baseline measured 19.424 tok/s at short context
and 17.826 tok/s for 64 output tokens after an 8,378-token prompt. The latter
profile was 6.388 ms MHC, 31.043 ms attention, 17.935 ms FFN, 1.383 ms head.

Job 51370956: restored the same images on all 12 nodes. Rank 0's routed
checksum is `80c171ac9f66d851`, matching the previous job. The existing binary
measured 20.585 tok/s over 64 short-context tokens and reproduced all token
IDs and printed logits from the previous packed-kernel run. Cross-job timing
differences must not be attributed to kernel changes.

The candidate accumulates integer dot products and minimum corrections over
each 256-value block before conversion to float. Its 8K-context generation
and profile are under `tmp/glm53f-q4-51370956/candidate8k.*`; output IDs are
retained on shared storage in `candidate8k.ids` when generation finishes.
The prompt requests a revised C11/C++17 reference header. The candidate
completed at EOS after 7,253 output tokens: 18.644 tok/s decode while context
grew from 8,378 to roughly 15,631 tokens. Prompt throughput was 19.081 tok/s.
Decode phase averages: 6.113 ms MHC, 29.972 ms attention, 16.732 ms FFN,
1.512 ms head. These long-run numbers do not isolate kernel speedup against
the short-context baseline; stable 20+ tok/s long-context decode is not met.

The raw response contains Markdown fences despite the source-only instruction.
After removing only those fences, the generated header compiles with both
`mpifcc -Nclang -std=c11` and `mpiFCC -Nclang -std=c++17`, using
`-O2 -Wall -Wextra -Werror -pedantic`. Both ARM executables pass RMSNorm,
BF16 dot-product, and context-partition smoke checks. This demonstrates a
coherent compilable output with a formatting violation; it is not exhaustive
verification of all generated numerical primitives or requested bug fixes.
Raw output, extracted header and harness are in the same shared log directory.
