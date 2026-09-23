# Qwen3.8-27B NVFP4 single-node decode roofline

The metadata-only inventory reads the original GGUF header without mapping or
loading its 16 GB payload:

```sh
make -C a64fx/llm qwen38_nvfp4_roofline
a64fx/llm/build/qwen38_nvfp4_roofline MODEL.gguf
```

The model has 14.754 GB of **active trunk projection and head payload** per
greedy token. The current eight-row packed layout expands this active stream
to 19.283 GB. The 20.627 GB allocation also includes embeddings, NextN, and
other weights that are not all streamed on each ordinary decode step.

Fugaku's [HBM2 peak is 1024 GB/s](https://global.fujitsu/en-global/technology/research/fugaku/specifications);
Fujitsu reports [about 830 GB/s STREAM Triad](https://www.fujitsu.com/global/documents/solutions/business-technology/tc/catalog/20180821hotchips30.pdf).
At 40 tokens/s the compact trunk needs 590 GB/s and the current packed trunk
needs 771 GB/s before activation, state, or dispatch traffic. The bounds below
use 830 GB/s and are optimistic; the measured packed stage times come from the
prior 48-worker boost-eco 128-token `hi` run with vectorized four-value
activation conversion and `TF_DPROF=1`.

| Stage | Compact GB | Current packed GB | Packed floor ms | Measured ms | Effective packed GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Attention QKV | 0.674 | 0.863 | 1.04 | 6.0 | 144 |
| Attention output | 0.283 | 0.377 | 0.45 | 2.2 | 172 |
| SSM input projections | 2.278 | 3.033 | 3.65 | 14.8 | 205 |
| SSM output | 0.849 | 1.132 | 1.36 | 6.5 | 174 |
| FFN gate/up | 6.417 | 8.556 | 10.31 | 29.4 | 291 |
| FFN down | 3.209 | 4.278 | 5.15 | 25.4 | 168 |
| Q6_K vocabulary head | 1.043 | 1.043 | 1.26 | 20.6 | 51 |

These stages sum to 104.9 ms of the 110.7 ms/token profiled decode. SSM
preparation/core and dispatch account for the remainder. The packed path is
not bandwidth-limited. On the same 48-core job a 0.503 GB packed-kernel
microbenchmark reached 408.7 GB/s while a read-only scan of the same buffer
reached 509.0 GB/s. Larger end-to-end gaps therefore include projection
shape, reduction, and dispatch effects as well as the nibble decode itself.

An in-place exact eight-row tile layout preserves the original 36-byte NVFP4
blocks and the serial FP32 FMA order. Together with an exact predecoded Q6_K
head, it produced 128/128 identical `hi` token IDs and zero selected-logit
difference from the original path. Decode rose from 4.694 to 5.126 tok/s;
head time fell from 20.5 to 8.1 ms/token. The Q6 head's separate 1.589 GB
arena replaces 1.043 GB of streamed original blocks, so its 830 GB/s floor is
1.91 ms. The FFN gate/up and down remain at 82.4 and 44.4 ms/token, far above
their compact byte floors (7.73 and 3.87 ms). Batched exact target
verification is the next optimization. The activation-quantized packed path
remains a diagnostic speed baseline and must not be used as the final verifier.

The exact small-N verifier shares each decoded NVFP4 value across up to four
candidate activations. Its four-row kernel reduced a four-token target batch
from 0.652 to 0.423 s (6.14 to 9.45 target tokens/s); all four argmax IDs
matched serial and the largest full-vocabulary logit delta was 1.3e-5. An
eight-row/three-token variant regressed and was removed. Batch attention
currently requires `TF_KV_DTYPE=f32`; the runner rejects F16 KV in this mode.

An exact NextN verifier now uses the batched target logits for every emitted
token, accepts only the matching draft prefix, and restores DeltaNet state
from the corresponding per-token snapshot. The auxiliary NextN tensors stay
in their original GGUF layout: tiling them collapsed all drafts to token zero,
while excluding those three tensors restored the original draft probe's 7/15
first-token matches. On the 64-token `hi` trace, every emitted `(position, ID)`
matched serial exact decode for K=2, 3, and 4. The best depth was K=3:
64 tokens / 12.861 s = **4.976 tok/s**, 35 accepted of 58 proposed draft
positions. Serial exact with F32 KV was 5.132 tok/s. The verifier is correct
but still slower; the 40 tok/s objective remains open. K=3 verification spent
2.826 s in projections, 3.810 s in FFN gate/up, and 1.932 s in FFN down.
Those kernels are the next priority.

On `Explain how matrix multiplication works in one paragraph.` the exact K=3
verifier matched all 32 serial output IDs, accepted 20/26 drafts, and reached
5.491 tok/s versus 5.041 tok/s serial. This demonstrates a small end-to-end
gain for a multi-token prompt while keeping full target verification.

Representative native runs (48 cores, one A64FX node, staged GGUF under
`/local`, `OMP_PROC_BIND=close`, four-CMG anonymous residency):

```sh
export TF_KV_DTYPE=f32 TF_DUMP_TOKENS=1
a64fx/llm/run_qwen38_nvfp4_cmg4.sh MODEL --prompt hi --max-seq 256 \
    --max-gen 64 --spec-k 0 --nvfp4-exact-tiled --q6-exact-head
a64fx/llm/run_qwen38_nvfp4_cmg4.sh MODEL --prompt hi --max-seq 256 \
    --max-gen 64 --spec-k 3 --spec-verify --nvfp4-exact-tiled --q6-exact-head
```
