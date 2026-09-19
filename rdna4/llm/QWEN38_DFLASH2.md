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
| Native DFlash2, K=4 | 451.22 | 30.37 | 37/40 | `15f17d2640c1adfc` |
| Native DFlash2, K=7 | 449.07 | 32.04 | 41/42 | `15f17d2640c1adfc` |

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
The native implementation reproduces that acceptance exactly and is about
1.84 times as fast as the upstream DFlash2 result.  It is still slower than
ordinary native decode, so DFlash2 remains opt-in.  The main remaining cost is
target verification: current IQ multi-row kernels preserve scalar arithmetic
but do not reuse enough target weights across eight rows.  DFlash prompt-cache
injection also reduces 4K prefill by about eight percent.

The tested sidecar is
`Qwen3.8-27B-DFlash2-Q4_K_M.gguf`, SHA-256
`1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd`.
