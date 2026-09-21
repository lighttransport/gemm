# Qwen3.8 DFlash2 on RDNA4

## Packed Q8/Q8 KV scales (2026-09-21)

Q8 K/V cache scales now use the same rounded FP16 contract as the stored
packed-F16 cache values.  The native attention kernels consume the 16-bit
scales directly, halving scale-cache traffic and preserving the existing
Q8/Q8 arithmetic.  The exact differential test passes 49,188,864 bitwise
comparisons, including 64K split counts and the multi-query reuse paths.  A
graph-captured 506-token zero-depth run completed at 43.1--43.2 tok/s with
stable output and `Result: PASS`.  The long-context operator differential is
also clean.  A real random 65,536-token depth run completed with a 443.33
tok/s prefix and 35.52 tok/s ordinary decode suffix, retaining prefix hash
`90178de69a24a76e`; the gate completed with `Result: PASS`.

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

The exact multi-row verifier is enabled for greedy, probabilistic, and coding
samplers. Every candidate row follows ordinary decode's projection, recurrent,
Q8/Q8 attention, and rollback arithmetic. The runner prints
`DFLASH2 sampled verifier=exact-window` for a sampled window. The target remains
authoritative: the sampler sees target logits and only accepted target state is
committed.

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
mode or the resident stdio server, batched Qwen3.8 prefill, the decode graph,
and Q8 K plus Q8 V.  It is mutually exclusive with dense NextN and Qwen4 MTP.
The HTTP shim enables it with `--qwen35-dflash2 SIDECAR`; each request is
serialized through the exact propose/verify/commit window.  Temperature-zero
requests use the exact argmax window, while sampled requests verify full row
logits with the existing sampler.  Bounded target prompt snapshots include the
sidecar's recurrent K/V cache, target Q8 KV rows, captured features and prompt
logits, so repeated short requests restore both target and draft state without
replaying the prompt.  Long prompts fall back to replay to avoid multi-gigabyte
host snapshots.

The opt-in HTTP quality gate covers deterministic greedy and seeded sampled
requests, repeated-request state isolation, disconnect cancellation followed
by recovery, and coherent coding and non-coding responses:

```sh
python3 rdna4/llm/test_qwen35_dflash2_http.py \
  --model /mnt/nvme02/models/qwen38/27b/gsq/Qwen3.8-27B-GSQ-RCO-IQ2_XS.gguf \
  --sidecar /mnt/nvme02/models/qwen38/27b/dflash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
```

The ordinary target's sampled random-64K quality gate also passes: a
temperature-0.6, seed-42 suffix retains prefix hash `90178de69a24a76e`,
produces suffix hash `34e2f6bc082bc49f`, and completes 32 tokens with
`Result: PASS` after a 445.67 tok/s random prefix.

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
| K=4 / greedy exact window | 605.43–605.74 | 59.43–59.50 | `4a0cb461966fae9a9d9da3b73c1b0c686ce8ee9ac3895c228bc6a653bc99a354` |
| K=4 / sampled exact window | 606.85–607.42 | 55.79–55.82 | `ddd1752b6c2a44251b659516b5937fdaa0e84f464530607e493abf8bbc37c9ac` |
| K=7 / greedy exact window | 607.37–608.15 | 81.68–81.82 | `4a0cb461966fae9a9d9da3b73c1b0c686ce8ee9ac3895c228bc6a653bc99a354` |
| K=7 / sampled exact window | 605.37–605.85 | 68.67–68.78 | `ddd1752b6c2a44251b659516b5937fdaa0e84f464530607e493abf8bbc37c9ac` |

The 4096-token early-context retrieval fixture also returns exactly
`ZEPHYR-7319` with K=7, including the ordinary target's token sequence and
EOS.  These results are under `tmp/qwen38/dflash2-sampled-window-k4/`,
`tmp/qwen38/dflash2-sampled-window-k7/`, and
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

Sampled parity exposed three verifier-specific hazards. Non-FFN IQ1_S
projections must use the scalar path's MMQ-scale interpretation, so their
batched launch now uses the same MMQ-scale kernel with the candidate row in
the grid Y dimension. Attention split counts must be selected independently
for each causal query, so the native query grid applies the ordinary adaptive
split policy in one launch. Finally, all verifier projections share one Q8_1
scratch allocation inside the captured graph; each projection now restages
that scratch instead of treating source-pointer identity as proof that its
contents are still live. With these fixes, all 135 sampled target rows match
the former ordinary exact-target trace bit-for-bit, including every stored
248,320-entry logit row.

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
remain byte-identical to the pinned llama.cpp fixtures. Both emitted functions
pass ASan/UBSan, fixed edge cases, and 10,000 randomized cases. At K=7, the
exact sampled window sustains 68.67--68.78 tok/s instead of the former
40.17--40.23 tok/s scalar fallback. K=4 sustains 55.79--55.82 tok/s; its
shorter draft does less useful work per verifier launch and remains below the
60 tok/s speculative target.

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

The ordinary target's recurrent alpha and beta F16 projections now use one
flattened launch while retaining the exact per-row FMA and XOR-reduction
order.  In a matched 65-row trace, this replaces 6,144 launches taking 23.102
ms with 3,072 launches taking 12.492 ms, or about 0.163 ms saved per target
row.  Zero-depth decode measures 42.83--43.04 tok/s and keeps the pinned
256-token hash `3c53b75f283cb9b0`.  The profile is under
`tmp/qwen38/ordinary-decode-profile-f16pair/`.

The dense FFN path also fuses SiLU multiplication with native Q8_1 staging
when the following down projection uses Q2_K, IQ2/3, or IQ4_XS.  This covers
58 of 64 layers in the tested IQ2_XS target and removes 58 launches per
ordinary decoded row.  The protected quotient and FP16 scale conversion match
the split quantizer: all 248,320 final logits are bitwise identical in the
fused/split A/B, and the 256-token hash remains `3c53b75f283cb9b0`.
Three-repeat zero-depth decode rises from a 42.79 tok/s split mean to a 43.11
tok/s fused mean.  The kernel trace is under
`tmp/qwen38/ordinary-decode-profile-siluq81/`; set the diagnostic
`LLM_QWEN35_SPLIT_SILU_Q81=1` to restore the two-launch boundary.

The 48 recurrent output layers now fold native Q8_1 staging into their
per-head gated RMSNorm/SiLU kernel.  The fused kernel retains the original
128-thread reduction, load loop, and stored activation boundary before four
independent waves quantize each head.  Its split-path A/B is bitwise identical
for all 248,320 final logits (SHA-256
`5b5f2f1a334ae644ac5633908e3d447c6d741c764a61dc0e9573addf699553c0`),
and three-repeat decode keeps the pinned `3c53b75f283cb9b0` sequence hash at
42.62--43.36 tok/s (43.07 mean), versus 41.64--43.14 tok/s (42.60 mean) for
the matched split path.  A 64-row trace removes exactly 3,072 launches, 48 per
row, and is under `tmp/qwen38/ordinary-decode-profile-ssmq81/`.  Set
`LLM_QWEN35_SPLIT_SSM_NORM_Q81=1` to restore the standalone quantizer.

Gated-attention output now applies sigmoid gating and stages native Q8_1 in
one wave-per-block kernel for 12 of 16 attention layers; the four IQ1 output
layers retain their dedicated quantizer.  The fused/split final logits are
bitwise identical with the same `5b5f2f1a...953c0` SHA-256, and all three
256-token repeats keep `3c53b75f283cb9b0`.  Matched decode means are 43.33
tok/s fused and 43.21 tok/s split.  A 64-row trace replaces 768 standalone
quantizer launches with 768 `sigmoid_mul_q81_f32` launches, reducing the total
by exactly 12 launches per row; it is under
`tmp/qwen38/ordinary-decode-profile-attngateq81/`.  Set
`LLM_QWEN35_SPLIT_ATTN_GATE_Q81=1` for the split path.

The exact one-row IQ scheduler now chooses its block geometry by projection
shape: IQ2_XS uses sixteen waves, the 17408-row IQ2_S shape uses sixteen, and
the frequent 5120/17408-row IQ3_S shapes use four. A wave still owns one
output row, preserving the reduction order. Matched 64-row traces save about
0.23 ms per decoded token; the tuned and eight-wave fallback paths produce
bitwise-identical 248,320-entry logits with SHA-256
`5b5f2f1a334ae644ac5633908e3d447c6d741c764a61dc0e9573addf699553c0`.
The authoritative 64K run keeps prefix/suffix hashes
`90178de69a24a76e`/`f4b35758fb99e6db` at 441.44 prefill and 34.98 decode
tok/s. `LLM_QWEN35_IQ_SHAPE_THREADS=0` restores the diagnostic fallback.

## Remaining optimization opportunities

The current 4K C++ gate sustains 68.67--68.78 tok/s for sampled K=7 and
81.68--81.82 tok/s for greedy K=7, clearing the 60 tok/s target. DFlash also
clears 40 tok/s after a real random-token 64K prefix. Ordinary one-token decode now
reaches 34.98 tok/s at 64K after exact GQA reuse, grouped K/Q scale products,
packed-probability reuse and scalar IQ codebook staging, so work that helps
both ordinary and verifier execution remains useful. The following order
reflects the remaining measured costs.

1. **Ordinary one-row target projections.** Scalar IQ2_XXS, IQ2_XS and
   IQ3_XXS now stage their small codebooks in LDS, lifting zero-depth decode
   from about 40.7 to 41.9--42.0 tok/s and 64K decode from 32.56 to 33.31
   tok/s. Hoisting repeated K/Q scale products in the exact three-head
   attention kernel raised the 256-token 64K run to 34.08 tok/s. Reusing each
   packed probability across both value tiles lowers the exact 128-split
   operator to 321.8--323.6 microseconds per layer and raises the full run to
   34.98 tok/s with unchanged prefix and suffix hashes, so the
   remaining gap is dominated by work outside attention. SiLU and Q8_1
   staging for 58 dense down projections now share one exact launch; gated
   RMSNorm and Q8_1 staging do the same for all 48 recurrent output
   projections, and sigmoid gating does the same for 12 attention output
   projections.
   Fixed-eight Q2_K/IQ projections already share decoded weights, but
   ordinary decode still streams weights for one row at a time. Reuse the
   quantized input across gate/up projections and investigate cooperative
   weight staging. A WMMA or reordered reduction path needs full
   output-token and logit validation because the current kernels preserve the
   target arithmetic order.
2. **Verifier attention tail.** The query-grid verifier now selects ordinary
   decode's split count independently for every causal row. It still writes
   split partials for a second combine launch, and that shared-K/V pass
   dominates the long-context verifier tail. Fuse the combine only if the
   selected split count and packed-F16 accumulation order remain exact.
3. **Hybrid recurrent tail.** Sequential candidate recurrence and rollback
   checkpoints are already batched and device-local. Alpha/beta F16 work now
   shares one exact launch per recurrent layer. DeltaNet, state preparation,
   checkpoint copies, and the remaining matrix-vector work remain visible;
   fuse preparation with the recurrence where exact row rollback is retained.
4. **Kernel and graph count.** Q/gate deinterleave, QK normalization, RoPE
   and Q8/Q8 KV storage are now fused exactly. The remaining small launches
   include SiLU/gating and state preparation. Fuse adjacent operations when
   their intermediate values need no external checkpoint.
5. **Remaining draft cost.** Top-k and selector decisions already run on the
   GPU, and packed Q4_K/Q8_1 projections cut draft work to 77.239 ms at 4K and
   474.117 ms across the 256-token 64K suffix. Position-parallel attention and
   cheaper draft-cache storage are the next candidates, provided K=4/K=7
   acceptance and authoritative output remain stable.
6. **Prompt-cache injection.**  The 4K and random 64K prefill targets are now
   met, but the five feature taps and sidecar K/V injection still consume
   avoidable bandwidth.  Fuse tap capture with target hidden writes, batch the
   five sidecar injections, and overlap independent sidecar work with the next
   target tile.

Each optimization should retain the exact sequence hash and response bytes at
K=4 and K=7, compile the emitted program, and cover non-coding prompts plus
random-token 64K depth.  HTTP/stdio scheduling remains a separate integration
task after the benchmark path has broader quality coverage.
