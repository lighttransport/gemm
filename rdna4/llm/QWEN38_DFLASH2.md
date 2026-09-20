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

The target remains authoritative.  Draft tokens are evaluated by the existing
exact Q8/Q8 multi-row target verifier.  Recurrent states and the target hidden
state are committed only through the accepted row.  The corresponding target
features are then injected into the draft cache, replacing the speculative
rows.  A rejection therefore cannot alter later target output.

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

## Validation and performance

Measured on RX 9070 XT / gfx1201 / ROCm 10 with the IQ2_XS target, a 4096-token
C coding prompt, 512-token chunks, context 8192, greedy sampling, and Q8 for
both K and V:

| Path | Prefill tok/s | Decode tok/s | Accepted drafts | Sequence hash |
|---|---:|---:|---:|---|
| Ordinary target, recent baseline | 533.19 | 39.55 | — | `15f17d2640c1adfc` |
| Native DFlash2, K=4 | 537.39 | 52.78 | 37/40 | `15f17d2640c1adfc` |
| Native DFlash2, K=7 | 539.54 | 76.16 | 41/42 | `15f17d2640c1adfc` |

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

The upstream llama.cpp server reference accepted 37/40 drafts at K=4 on the
same prompt, but measured 16.54 tok/s versus its 25.88 tok/s ordinary path.
The native K=4 implementation reproduces that acceptance exactly and is 3.19
times as fast.  K=7 is 93 percent faster than the retained recent ordinary
native baseline on this prompt.  DFlash prefill also clears the 500 tok/s
target.  The feature remains opt-in while serving integration and broader
quality coverage are incomplete.

The optimized target verifier decodes IQ and Q2_K weights once for up to eight
candidate rows.  Quantization-format-specific kernels remove runtime codebook
branches; exact Q8_1 IQ1_S/IQ1_M kernels reuse each decoded group; IQ4_XS keeps
the reference's eight virtual sums.  The fixed-eight Q2_K kernel also finishes
one query at a time after decoding a weight block, while compact IQ2 and IQ3_S
schedules reduce live accumulators and retain the reference reduction order.
IQ3_XXS keeps its faster original shared schedule.  RMSNorm and
residual-plus-RMSNorm launch one independent block per candidate row.  The
exact Q8/Q8 verifier attention
now loads each old K/V row once and evaluates up to eight adjacent causal
queries in the same four-wave block.  It preserves each query's quantization,
online softmax, packed-F16 accumulation and split-combine order.  The draft
also reuses Q4_K weights and holds one K/V vector while evaluating four mask
rows.

These changes keep the target sequence unchanged while reducing the K=7
draft/verify/commit split to 108.910/483.679/10.651 ms for the complete
46-token response.  The eight-query attention operator takes 213.382
microseconds at 4K and 3.076784 milliseconds at 64K with eight splits.  The
pinned llama.cpp differential test passes 46,743,552 bitwise Q8/Q8 output
comparisons.  The expanded projection differential passes 2,948,352 exact
activation values and 13,191,360 bitwise Q2_K/IQ outputs, including all fixed
eight-row Q2_K/IQ kernels.  The emitted source passes
`gcc -std=c17 -Wall -Wextra -Wpedantic -Werror` and boundary tests using
`INT_MIN` and `INT_MAX`.

The same K=7 path was measured after a fully processed 65,536-token random
prefix.  Prefix processing sustained 445.71 tok/s with hash
`90178de69a24a76e`; the following 256 generated tokens sustained 47.72 tok/s
with hash `2ddd068dca63669a`.  It drafted 259 tokens and accepted 217.  The
draft/verify/commit split was 668.717/4602.977/57.440 ms.  The final suffix
hash is unchanged from the earlier exact verifier.  The fixed eight-split
grouping can change rejected draft proposals at synthetic depth through
floating-point grouping, but it does not change authoritative target output.

The tested sidecar is
`Qwen3.8-27B-DFlash2-Q4_K_M.gguf`, SHA-256
`1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd`.

## Remaining optimization opportunities

The short-context K=7 response emits 46 tokens in 604.00 ms, clearing the
60 tok/s target with about 21 percent wall-time headroom.  DFlash also clears
40 tok/s after a real random-token 64K prefix.  Ordinary one-token decode is
still about 29.13 tok/s at 64K, so work that helps both ordinary and verifier
execution remains useful.  The following order reflects the remaining
measured costs.

1. **One-row target projections.**  Fixed-eight Q2_K/IQ projections now share
   decoded weights with lower accumulator pressure, but ordinary decode still
   streams the same weights for one row at a time.  Reuse the quantized input
   across gate/up projections, reduce codebook traffic, and investigate
   cooperative staging.  A WMMA or reordered reduction path needs full
   output-token and logit validation because the current kernels preserve the
   target arithmetic order.
2. **Hybrid recurrent state.**  DeltaNet used 65.17 ms, state preparation
   14.94 ms, checkpoint copies 16.45 ms, and F16 matrix-vector work 13.32 ms.
   Processing sequential candidate rows in one state kernel and keeping
   checkpoints device-local could remove launches and memory traffic.  Every
   row must retain a rollback point so a rejected draft cannot affect later
   target state.
3. **Verifier attention tail.**  The shared-K/V kernel halves the 64K
   eight-query operator time, but it still writes split partials for a second
   combine launch.  An exact in-kernel combine or adaptive split policy may
   reduce the tail if it preserves the pinned arithmetic at the selected split
   count.
4. **Kernel and graph count.**  The remaining small launches include QK
   normalization, RoPE, KV storage, SiLU/gating, and state preparation.  Fuse
   adjacent operations when their intermediate values need no external
   checkpoint.
5. **Draft and selector cost.**  Draft work is 108.910 ms at 4K and 668.717 ms
   across the 256-token 64K suffix.  Position-parallel attention, cheaper
   draft-cache storage, and more selector work on the GPU are candidates,
   provided K=4/K=7 acceptance and authoritative output remain stable.
6. **Prompt-cache injection.**  The 4K and random 64K prefill targets are now
   met, but the five feature taps and sidecar K/V injection still consume
   avoidable bandwidth.  Fuse tap capture with target hidden writes, batch the
   five sidecar injections, and overlap independent sidecar work with the next
   target tile.

Each optimization should retain the exact sequence hash and response bytes at
K=4 and K=7, compile the emitted program, and cover non-coding prompts plus
random-token 64K depth.  HTTP/stdio scheduling remains a separate integration
task after the benchmark path has broader quality coverage.
