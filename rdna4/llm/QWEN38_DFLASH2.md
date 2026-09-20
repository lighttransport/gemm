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
| Ordinary target | 489.49 | 37.82 | — | `15f17d2640c1adfc` |
| Native DFlash2, K=4 | 446.24 | 38.91 | 37/40 | `15f17d2640c1adfc` |
| Native DFlash2, K=7 | 446.38 | 43.68 | 41/42 | `15f17d2640c1adfc` |

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
The native K=4 implementation reproduces that acceptance exactly and is 2.35
times as fast.  K=7 is 15 percent faster than ordinary native decode on this
prompt.  DFlash prompt-cache injection still reduces 4K prefill by about nine
percent, and the feature remains opt-in while serving integration and broader
quality coverage are incomplete.

The optimized target verifier decodes IQ and Q2_K weights once for up to eight
candidate rows.  Quantization-format-specific kernels remove runtime codebook
branches; exact Q8_1 IQ1_S/IQ1_M kernels reuse each decoded group; IQ4_XS keeps
the reference's eight virtual sums.  RMSNorm and residual-plus-RMSNorm launch
one independent block per candidate row.  The draft also reuses Q4_K weights
and holds one K/V vector while evaluating four mask rows.  These changes keep
the target sequence unchanged while reducing the K=7 draft/verify/commit split
to 197.28/813.57/15.51 ms for the complete 46-token response.  The emitted
source passes `gcc -std=c17 -Wall -Wextra -Wpedantic -Werror` and boundary
tests using `INT_MIN` and `INT_MAX`.

The tested sidecar is
`Qwen3.8-27B-DFlash2-Q4_K_M.gguf`, SHA-256
`1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd`.

## Remaining optimization opportunities

The K=7 response emits 46 tokens in 1053.09 ms.  Reaching 60 tok/s requires
at most 766.67 ms, a 286.42 ms or 27 percent reduction.  Draft and commit take
197.28 and 15.51 ms, while unassigned host/runtime overhead is about 26.74 ms.
If those costs remain fixed, target verification must fall from 813.57 to
527.15 ms, a 35 percent reduction.  The following order reflects the current
profile; its kernel times were captured before final batched normalization and
are useful for ranking rather than summing into the final wall time.

1. **Target projection kernels.**  Exact Q2_K multi-row projection used 93.96
   ms; IQ2/IQ3 formats used 265.74 ms combined; IQ4_XS used 61.20 ms; and
   IQ1_S/IQ1_M used 52.68 ms.  The next kernel work should reduce register
   pressure and repeated codebook traffic, cooperatively stage decoded weight
   tiles, and fuse gate/up activation quantization where the same input is
   reused.  A WMMA or reordered reduction path needs full output-token and
   logit validation because the current kernels preserve the target arithmetic
   order.
2. **Hybrid recurrent state.**  DeltaNet used 65.17 ms, state preparation
   14.94 ms, checkpoint copies 16.45 ms, and F16 matrix-vector work 13.32 ms.
   Processing sequential candidate rows in one state kernel and keeping
   checkpoints device-local could remove launches and memory traffic.  Every
   row must retain a rollback point so a rejected draft cannot affect later
   target state.
3. **Target Q8/Q8 attention.**  Attention plus combine used 56.78 ms.  Store
   all speculative K/V rows first, then evaluate them in one batched kernel
   with a row-specific causal end while reusing older cache vectors across
   rows.  Preserve Q8 K and Q8 V and each row's reduction order.
4. **Kernel and graph count.**  Batching normalization already removed about
   55 ms from the response.  The remaining small launches include QK
   normalization, RoPE, KV storage, SiLU/gating, and state preparation.  Fuse
   adjacent operations when their intermediate values need no external
   checkpoint.
5. **Draft cost.**  The final DFlash stage takes 197.28 ms.  Position-parallel
   attention and cheaper draft-cache storage are candidates, provided K=7
   acceptance stays at 41/42 and target output remains unchanged.

Prefill also has a separate gap.  DFlash runs at 446.38 tok/s versus 489.49
tok/s for the ordinary target.  Capturing five feature taps and injecting the
sidecar cache costs about 808 ms over 4096 tokens.  Removing that entire cost
would only recover the ordinary 489.49 tok/s rate; 500 tok/s also requires
about 176 ms from the underlying target prefill.  The most direct DFlash work
is to fuse tap capture into the target hidden writes, batch the five sidecar
K/V injections, and overlap independent sidecar work with the next target
tile.

Each optimization should retain the exact sequence hash and response bytes at
K=4 and K=7, compile the emitted program, and cover non-coding prompts plus
random-token 64K depth.  HTTP/stdio scheduling remains a separate integration
task after the benchmark path has broader quality coverage.
