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

Further A64FX checks rule out simple placement and dispatch fixes. Copying
the 50 MB resident FFN gate matrix into a freshly first-touched allocation
gave 73.8 GB/s versus 73.5 GB/s for the original matrix. Calling the same
SVE loop directly gave 71.8 GB/s versus 72.2 GB/s through the normal GEMM
entry point. A separately compiled, fixed-N=3 version matched every one of
52,224 FFN output floats bitwise but reached only 71.3 GB/s. Explicit
lookahead prefetch distances of 2–64 compact blocks did not beat the
unprefetched 74.9 GB/s isolated baseline. LLVM 21 and `-mcpu=a64fx` gave no
material improvement over the Fujitsu compiler on the exact isolated shape.

On 500 passes of the corrected isolated FFN kernel, `perf stat` counted
58.1 billion instructions, 36.3 billion retired SVE instructions, 8.36
billion FP FMA instructions, and 101 million L2 refills; 42.3% of cycles
were backend-stalled. That is roughly 116 million instructions and 202,000
L2 refills per 50 MB matrix pass. The L2 refills are consistent with reading
the whole source matrix each pass, while the low throughput and instruction
count point to decode/accumulation work as the limiting factor. Reaching the
40 tok/s end-to-end target will require a substantially different exact
projection kernel, not a prefetch or NUMA adjustment.

The first effective code-generation change compiles a separate exact N=3
kernel for each of the model's six active NVFP4 matrix shapes. Those 368
tensors account for 13.589 GB of active compact weights. Keeping row count,
column count, and token strides constant in each compiled routine raises
resident projection throughput from about 69–74 GB/s to 93–100 GB/s. The
probe's `--kernel-probe-check` option compares every output float against the
generic exact SVE kernel: all six shapes had zero mismatches, including
52,224 outputs for the FFN gate matrix. The dispatch applies only to exact
eight-row tiled weights, N=3, matching strides, and 512-bit SVE; other cases
continue through the generic kernel.

With the specialized N=3 path, the 64-token `hi` verifier reached **7.802
tok/s** (35/58 draft positions accepted), versus 6.834 tok/s immediately
before specialization. The 32-token matrix-multiplication prompt reached
**8.783 tok/s** (20/26 accepted), versus 7.346 tok/s. Every emitted
`(position, ID)` matched the corresponding unapproximated serial target trace
(64/64 and 32/32). These are end-to-end single-request measurements. The
40 tok/s objective remains open; the exact Q6 head still takes about 8 ms
per three-token batch, and these verifier runs emit about 2.2–2.5 tokens
per round.

Representative native runs (48 cores, one A64FX node, staged GGUF under
`/local`, `OMP_PROC_BIND=close`, four-CMG anonymous residency):

```sh
export TF_KV_DTYPE=f32 TF_DUMP_TOKENS=1
a64fx/llm/run_qwen38_nvfp4_cmg4.sh MODEL --prompt hi --max-seq 256 \
    --max-gen 64 --spec-k 0 --nvfp4-exact-tiled --q6-exact-head
a64fx/llm/run_qwen38_nvfp4_cmg4.sh MODEL --prompt hi --max-seq 256 \
    --max-gen 64 --spec-k 3 --spec-verify --nvfp4-exact-tiled --q6-exact-head
a64fx/llm/run_qwen38_nvfp4_cmg4.sh MODEL --max-seq 256 \
    --nvfp4-exact-tiled --q6-exact-head --kernel-probe 200 --kernel-probe-check
```

The NextN draft FFN's three `blk.64` NVFP4 matrices total 0.150 GB per
draft call. They previously stayed in GGUF row layout because a full-model
tile experiment changed the drafts to token zero. A separate probe located
the defect: the fused gate/up matvec decoded tiled weights as GGUF rows.
The fused worker now dispatches tile-aware matvecs for both matrices.
`--nextn-tile-probe` compares the original and tiled NextN path after prompt
prefill; all three matrix outputs match bitwise, the full logits have a
maximum 5.01e-6 difference from accumulation order, and the top ID matches.

`--nextn-exact-tiled` tiles only those three draft FFN matrices. On the
32-token matrix-multiplication prompt, K=3 with this flag reached 9.250
tok/s, accepted 20/26 drafts, and matched all 32 serial target `(position,
ID)` pairs. Draft time fell from 878 to 566 ms; target verification remained
2.79 s. Enabling the existing NextN persistent worker modes cut draft time
further to 421 ms and raised the same trace to **9.743 tok/s**, again with
32/32 exact target IDs. These timings include only decode, not model load.
At 40 tok/s the 32-token trace must finish in 0.8 s, so the current 2.76 s
target-verification stage alone exceeds the full budget by 3.45x. The exact
projection kernels, particularly gate/up and down, remain the limiting work.
The 64-token `hi` trace reached 8.645 tok/s with 35/58 drafts accepted; all
64 `(position, ID)` pairs matched serial target decode.

```sh
export TF_NEXTN_FFN_PERSIST=1 TF_NEXTN_BLOCK_PERSIST=1
export TF_NEXTN_FULL_PERSIST=1 TF_NEXTN_ATTN_BLOCK_PERSIST=1
export TF_NEXTN_QKV_PERSIST=1 TF_NEXTN_INLINE_ARGMAX=1
export TF_KV_DTYPE=f32 TF_DUMP_TOKENS=1
a64fx/llm/run_qwen38_nvfp4_cmg4.sh MODEL \
    --prompt 'Explain how matrix multiplication works in one paragraph.' \
    --max-seq 256 --max-gen 32 --spec-k 3 --spec-verify \
    --nvfp4-exact-tiled --nextn-exact-tiled --q6-exact-head
```

With the same exact NextN layout and persistent workers, changing only the
speculation depth did not help the 32-token prompt. K=2 took 4.243 s (7.541
tok/s, 18 rounds, 15/18 accepted); K=4 took 4.478 s (7.147 tok/s, 12 rounds,
22/36 accepted). Both matched all 32 serial target IDs. K=3 remains the
fastest measured depth at 3.284 s. The extra speculative positions of K=4
cost more target projection and draft work than its saved round recovers.

An isolated 50.135 MB FFN gate benchmark with fixed dimensions and otherwise
the same SVE decode measured 0.395 ms for N=1, 0.499 ms for N=3, and 0.637
ms for N=4 on the 48-core node. Thus even single-candidate compact decode
is far below a read-only HBM scan; candidate FMAs add cost but are not the
sole limit. Two exact representation experiments were slower or too small an
improvement to justify residency: FP16 scales enlarged the stream by 11%
and took 0.669 ms for N=3; paired-row signed-byte FP4 codes doubled it and
took 0.458 ms. These are synthetic resident-kernel timings, not end-to-end
rates or full-logit correctness checks. The latter byte layout would also
increase the active trunk by 13.589 GB on a 32 GB node.
Fully unrolling the four 16-value subblocks also regressed the fixed N=3
50.135 MB kernel from 0.499 to 0.598 ms. Disassembly of the current kernel
showed no vector accumulator spills in its inner loop, so register-spill
removal is not an available shortcut to the required throughput.
