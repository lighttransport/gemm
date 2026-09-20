# Qwen3.8 DFlash2 on RDNA4

The Qwen3.8 runner can load the
[IncoAI Qwen3.8-27B DFlash2 GGUF](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF)
as an opt-in draft sidecar.  The implementation is native HIP and does not
invoke llama.cpp at runtime.

DFlash2 has a five-layer 5120-wide draft transformer with its own 32-head,
8-KV-head, 128-dimensional attention geometry and a 2048-token sliding
window.  It consumes the target inputs to layers 6, 20, 34, 48 and 62.  The
runner captures those features during each 512-token target prefill tile,
fuses them, and injects the resulting K/V rows into the draft cache.  A draft
step evaluates one non-causal block containing the target anchor followed by
mask tokens.  Its rank-256 selector walks the top-16 candidate lattice.

The target remains authoritative.  Draft tokens are evaluated by the Q8/Q8
multi-row target verifier.  Recurrent states and the target hidden state are
committed only through the accepted row.  The corresponding target features
are then injected into the draft cache, replacing the speculative rows.  A
rejection therefore cannot alter later target output.

The multi-row verifier is enabled only for greedy selection.  Its first row
matches ordinary decode, while later rows currently have small logit changes
from batching recurrent and projection work.  Greedy token choices pass the
pinned fixtures; probabilistic and coding samplers use ordinary exact-target
decode so the sidecar cannot change their token stream.  The runner prints
`DFLASH2 sampled fallback=exact-target` when this happens.

## Run

```sh
bash rdna4/llm/run_qwen38_gsq_rocm.sh --gpu-only-bench \
  --prompt-file tmp/qwen38/prefill-4k-512/coding-4096.txt \
  -n 4096 -s 8192 --ubatch 512 --kv-cache q8q8 \
  --qwen35-prefill-bf16 --qwen35-decode-graph \
  --qwen35-native-q8-prefill --qwen35-native-mmvq \
  --sampling-profile llama --temp 0 --seed 42 --decode 64 \
  --qwen35-dflash2 \
    /mnt/nvme02/models/qwen38/27b/dflash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --qwen35-dflash2-draft 7
```

Draft width may be 1 through 7.  The sidecar currently requires benchmark
mode, batched Qwen3.8 prefill, the decode graph, and Q8 K plus Q8 V.  It is
mutually exclusive with dense NextN and Qwen4 MTP.  HTTP/stdio scheduling is
not implemented.

The reference validator accepts the sidecar directly:

```sh
python3 rdna4/llm/validate_qwen38_reference.py \
  --model /mnt/nvme02/models/qwen38/27b/gsq/Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf \
  --out tmp/qwen38/dflash2-quality-k7 \
  --reuse-reference tmp/qwen38/final-iq2-native-q2k \
  --native-q8-prefill --native-mmvq \
  --dflash2 \
    /mnt/nvme02/models/qwen38/27b/dflash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --dflash2-draft 7 --decode 256 --repeats 3
```

## Validation and performance

Measured on RX 9070 XT / gfx1201 / ROCm 10 with the IQ2_XS target, a 4096-token
C coding prompt, 512-token chunks, context 8192, greedy sampling, and Q8 for
both K and V:

| Path | Prefill tok/s | Decode tok/s | Accepted drafts | Sequence hash |
|---|---:|---:|---:|---|
| Ordinary target, recent baseline | 533.19 | 39.55 | — | `15f17d2640c1adfc` |
| Native DFlash2, K=4 | 537.42 | 54.49 | 37/40 | `15f17d2640c1adfc` |
| Native DFlash2, K=7 | 540.43 | 81.05 | 41/42 | `15f17d2640c1adfc` |

All three paths emitted the same 46-token response and EOS.  The response is
valid C and implements the requested inclusive clamp without overflow-prone
arithmetic:

```c
int clamp(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
```

A broader 4096-token C++17 merge-intervals fixture covers greedy and
temperature-0.6 sampling.  Both K=4 and K=7 match the pinned llama.cpp token
IDs, EOS and output bytes.  The generated function passes fixed edge cases,
ASan/UBSan, and 10,000 randomized cases.  Warm results were:

| Draft width / selection | Prefill tok/s | Decode tok/s | Output SHA-256 |
|---|---:|---:|---|
| K=4 / greedy | 606.22–606.73 | 60.83 | `4a0cb461966fae9a9d9da3b73c1b0c686ce8ee9ac3895c228bc6a653bc99a354` |
| K=7 / greedy | 607.36–608.28 | 85.35–85.44 | `4a0cb461966fae9a9d9da3b73c1b0c686ce8ee9ac3895c228bc6a653bc99a354` |
| sampled exact-target fallback | 605.92–607.82 | 40.17–40.23 | `ddd1752b6c2a44251b659516b5937fdaa0e84f464530607e493abf8bbc37c9ac` |

The 4096-token early-context retrieval fixture also returns exactly
`ZEPHYR-7319` with K=7, including the ordinary target's token sequence and
EOS.  These results are under `tmp/qwen38/dflash2-qkprep-k4/`,
`tmp/qwen38/dflash2-qkprep-k7/`, and
`tmp/qwen38/dflash2-retrieval-k7.*`.

The upstream llama.cpp server reference accepted 37/40 drafts at K=4 on the
same prompt, but measured 16.54 tok/s versus its 25.88 tok/s ordinary path.
The native K=4 implementation reproduces that acceptance exactly and is 3.29
times as fast.  K=7 is 105 percent faster than the retained recent ordinary
native baseline on this prompt.  DFlash prefill also clears the 500 tok/s
target.  The feature remains opt-in while serving integration and broader
quality coverage remain incomplete.

The optimized target verifier decodes IQ and Q2_K weights once for up to eight
candidate rows.  Quantization-format-specific kernels remove runtime codebook
branches; exact Q8_1 IQ1_S/IQ1_M kernels reuse each decoded group; IQ4_XS keeps
the reference's eight virtual sums.  The fixed-eight Q2_K kernel also finishes
one query at a time after decoding a weight block, while compact IQ2 and IQ3_S
schedules reduce live accumulators and retain the reference reduction order.
IQ3_XXS keeps its faster original shared schedule.  RMSNorm and
residual-plus-RMSNorm launch one independent block per candidate row. Q4_K
draft projections quantize each candidate activation in 32-value Q8_1 groups,
reuse one decoded weight chunk across all rows, and use gfx1201 packed integer
dots. Adjacent projections with the same activation reuse the quantized input.
This reduces the 4K K=7 draft phase from 109.739 to 77.239 ms. Draft arithmetic
may change rejected proposals; accepted tokens and state still come only from
the exact target verifier. The exact Q8/Q8 verifier attention
now loads each old K/V row once and evaluates up to eight adjacent causal
queries in the same four-wave block.  It preserves each query's quantization,
online softmax, packed-F16 accumulation and split-combine order.  The draft
also reuses Q4_K weights and holds one K/V vector while evaluating four mask
rows.

The exact fixed-eight Q2_K/IQ kernels and IQ4_XS multi-row projection use
eight-wave, 256-thread blocks. This changes only the assignment of eight
output rows to a block; each row retains its existing arithmetic. In matched
K=7 traces it reduced aggregate fixed-eight projection time from 223.706 to
222.336 ms and IQ4_XS multi-row time from 47.068 to 46.624 ms. The projection
differential passes 2,948,352 exact activation values and 13,191,360 bitwise
outputs with the production launch geometry.

These changes keep the target sequence unchanged while reducing the K=7
draft/verify/commit split to 77.239/478.857/10.624 ms for the complete
46-token response.  The eight-query attention operator takes 213.382
microseconds at 4K and 3.076784 milliseconds at 64K with eight splits.  The
pinned llama.cpp differential test passes 46,743,552 bitwise Q8/Q8 output
comparisons.  The expanded projection differential passes 2,948,352 exact
activation values and 13,191,360 bitwise Q2_K/IQ outputs, including all fixed
eight-row Q2_K/IQ kernels.  The emitted source passes
`gcc -std=c17 -Wall -Wextra -Wpedantic -Werror` and boundary tests using
`INT_MIN` and `INT_MAX`.

The fixed-shape GDA recurrence has separate contracts for raw scalar alpha and
prefill's precomputed decay.  Its fixed-bound loop preserves the generic
kernel's operation order under the runner's `-ffast-math` HIPRTC mode.  The
dedicated test compares 19,537,920 state/output values bitwise across both
contracts.  This restores the pinned sampled sequence while retaining the
specialized prefill speed; spelling the four rows as separate accumulators
reassociated operations and changed the sampled output.

Decode attention preparation now uses one exact kernel per attention layer for
Q/gate deinterleave, Q and K RMSNorm, Q and K M-RoPE, and Q8/Q8 K/V storage.
It replaces six launches with one in each of the 16 attention layers, removing
80 launches per decoded target row. The fused kernel retains the separate
operators' reduction order, trigonometric operations, Q8 scale rounding and
integer conversion. The complete K=4 and K=7 greedy and sampled C++ gates
remain byte-identical to the pinned llama.cpp fixtures. At a zero-length
prefix, ordinary throughput is unchanged at 42.53--42.65 tok/s; at the 4K
coding shape, sampled exact-target fallback improves from 39.84--39.93 to
40.17--40.23 tok/s.

The same K=7 path was measured after a fully processed 65,536-token random
prefix. Prefix processing sustained 443.57 tok/s with hash
`90178de69a24a76e`; the following 256 generated tokens sustained **49.74
tok/s** with hash `2ddd068dca63669a`. It drafted 259 tokens and accepted 217.
The draft/verify/commit split was 474.117/4585.996/56.181 ms. The final suffix
hash is unchanged from the earlier exact verifier.  The fixed eight-split
grouping can change rejected draft proposals at synthetic depth through
floating-point grouping, but it does not change authoritative target output.

The tested sidecar is
`Qwen3.8-27B-DFlash2-Q4_K_M.gguf`, SHA-256
`1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd`.

## Remaining optimization opportunities

The short-context K=7 response emits 46 tokens in 566.72 ms, clearing the
60 tok/s target with about 35 percent throughput headroom. DFlash also clears
40 tok/s after a real random-token 64K prefix. Ordinary one-token decode now
reaches 34.08 tok/s at 64K after exact GQA reuse, grouped K/Q scale products
and scalar IQ codebook staging, so work that helps
both ordinary and verifier execution remains useful. The following order
reflects the remaining measured costs.

1. **Ordinary one-row target projections.** Scalar IQ2_XXS, IQ2_XS and
   IQ3_XXS now stage their small codebooks in LDS, lifting zero-depth decode
   from about 40.7 to 41.9--42.0 tok/s and 64K decode from 32.56 to 33.31
   tok/s. Hoisting repeated K/Q scale products in the exact three-head
   attention kernel raises the latest 256-token 64K run to 34.08 tok/s, so the
   remaining gap is dominated by work outside attention. Fixed-eight Q2_K/IQ
   projections already share decoded weights, but
   ordinary decode still streams weights for one row at a time. Reuse the
   quantized input across gate/up projections and investigate cooperative
   weight staging. A WMMA or reordered reduction path needs full
   output-token and logit validation because the current kernels preserve the
   target arithmetic order.
2. **Verifier attention tail.** The shared-K/V kernel dominates the remaining
   long-context verifier time and still writes split partials for a second
   combine launch. An exact in-kernel combine or adaptive split policy may
   reduce the tail if it preserves the pinned arithmetic at the selected split
   count.
3. **Exact sampled multi-row verification.** Audit the first divergent
   verifier row against repeated scalar target decode, beginning with the
   recurrent checkpoints and compact projection inputs.  Enable DFlash for
   probabilistic sampling only after every verifier row produces the ordinary
   target logits bitwise and both pinned sampling fixtures still match.
4. **Hybrid recurrent tail.** Sequential candidate recurrence and rollback
   checkpoints are already batched and device-local. DeltaNet, state
   preparation, checkpoint copies, and F16 matrix-vector work remain visible;
   fuse preparation with the recurrence where exact row rollback is retained.
5. **Kernel and graph count.** Q/gate deinterleave, QK normalization, RoPE
   and Q8/Q8 KV storage are now fused exactly. The remaining small launches
   include SiLU/gating and state preparation. Fuse adjacent operations when
   their intermediate values need no external checkpoint.
6. **Remaining draft cost.** Top-k and selector decisions already run on the
   GPU, and packed Q4_K/Q8_1 projections cut draft work to 77.239 ms at 4K and
   474.117 ms across the 256-token 64K suffix. Position-parallel attention and
   cheaper draft-cache storage are the next candidates, provided K=4/K=7
   acceptance and authoritative output remain stable.
7. **Prompt-cache injection.**  The 4K and random 64K prefill targets are now
   met, but the five feature taps and sidecar K/V injection still consume
   avoidable bandwidth.  Fuse tap capture with target hidden writes, batch the
   five sidecar injections, and overlap independent sidecar work with the next
   target tile.

Each optimization should retain the exact sequence hash and response bytes at
K=4 and K=7, compile the emitted program, and cover non-coding prompts plus
random-token 64K depth.  HTTP/stdio scheduling remains a separate integration
task after the benchmark path has broader quality coverage.
