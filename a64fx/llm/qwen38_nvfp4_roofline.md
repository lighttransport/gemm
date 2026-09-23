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

The next exact kernel packs two output rows into the low and high halves of
each 512-bit SVE register, using all 16 FP32 lanes. Four row pairs consume one
eight-row compact tile across up to four candidates. The four-token `hi`
batch improved from 0.423 to 0.308 s (13.00 verified target tokens/s), with
the same 4/4 argmax IDs and 1.3e-5 maximum logit delta. End-to-end K=3
improved from 4.976 to **6.788 tok/s** on `hi` (64/64 IDs identical), and
from 5.491 to **7.346 tok/s** on the matrix-multiplication prompt (32/32 IDs
identical). The single-request 40 tok/s target remains open.

`--kernel-probe 200` times resident exact three-token projections after load
and in-place tiling. On the same 48-core job, the measured source-stream
rates were 71.8 GB/s for SSM QKV, 69.6 for SSM output, 73.5 for FFN gate,
72.2 for FFN down, and 73.3 for attention Q. The 1.589 GB exact Q6 head
reached 198.1 GB/s. The compact FP4 projections are therefore limited by
decode and FP32 accumulation rather than raw HBM bandwidth. The gate matrix,
for example, streams 0.050 GB in 0.682 ms for three candidates; its 830 GB/s
source-byte floor is 0.060 ms. The head takes 8.021 ms versus a 1.91 ms
source-byte floor. These are per-matrix timings, not end-to-end token rates.

A standalone exact compact-kernel FFN probe reached 76 GB/s with runtime N=3,
matching the loaded model. Compile-time N=3 reached about 89 GB/s, but
specializing the production call by forced inlining or a separate routine
regressed to 68–71 GB/s on the loaded model, so that change was discarded.
Expanding FP4 nibbles to signed bytes with exact FP16 scales consumed about
twice the source storage and still took about 0.64 ms for the FFN gate shape.
A packed FP32-scale variant took about 0.68 ms. Neither is a useful resident
format for this 32 GB node. The original compact exact path remains in use.

Representative native runs (48 cores, one A64FX node, staged GGUF under
`/local`, `OMP_PROC_BIND=close`, four-CMG anonymous residency):

```sh
export TF_KV_DTYPE=f32 TF_DUMP_TOKENS=1
a64fx/llm/run_qwen38_nvfp4_cmg4.sh MODEL --prompt hi --max-seq 256 \
    --max-gen 64 --spec-k 0 --nvfp4-exact-tiled --q6-exact-head
a64fx/llm/run_qwen38_nvfp4_cmg4.sh MODEL --prompt hi --max-seq 256 \
    --max-gen 64 --spec-k 3 --spec-verify --nvfp4-exact-tiled --q6-exact-head
a64fx/llm/run_qwen38_nvfp4_cmg4.sh MODEL --max-seq 256 \
    --nvfp4-exact-tiled --q6-exact-head --kernel-probe 200
```
