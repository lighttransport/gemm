# Native and PyTorch performance comparison

Measured on 2026-10-03 (JST), NVIDIA RTX 5060 Ti 16 GB, driver 615.71.09,
PyTorch 2.14.0+cu130. Both paths use the same pinned weights, portrait, input
noise, prompt and 480×848, 81-frame I2V recipe. Native uses repository GEMM
with zero vendor/fallback calls. The reference uses the clean official source
at `60783e704160023913bee78f0b47036d393d4dfa`.

## Optimized bounded replays

The isolated candidate implements stream-ordered buffer reuse, pinned weight
prefetch, immutable weight caching, repository v7 GEMM, FP16 FlashAttention2,
fused DiT operations, implicit-GEMM VAE convolution and fused norm/SiLU.
Encoder arithmetic remains IEEE FP32; register tiling accelerates its GEMM,
and raw F16/BF16 weights convert exactly to FP32 on the GPU.

Each new GPU command has a 55-second watchdog, including loading, compilation,
capture writes and reservation overhead. The measurements below are medians of
the two warm forwards after the first forward, with CUDA synchronization.
Setup and complete experiment times are recorded separately. The native
candidate uses **zero cuBLAS/fallback calls**. Desktop graphics remain active.

| Production-shape component | Native | PyTorch reference | Native/reference |
|---|---:|---:|---:|
| Joint attention, 34,138 tokens, 16×128 heads | 0.425 s | 0.482 s | 0.881× |
| Complete first Fast12 DiT block | 0.549 s | 0.635 s | 0.864× |
| VAE interior tile, 81×128×128 output | 1.504 s | 1.714 s | 0.878× |
| VAE edge tile, 81×80×96 output | 0.699 s | 0.787 s | 0.888× |
| Qwen layer, IEEE FP32 | 0.0223 s | 0.5568 s | 0.040× |
| SigLIP layer, IEEE FP32 | 0.0330 s | 0.1596 s | 0.207× |
| ByT5 layer, IEEE FP32 | 0.00117 s | 0.02425 s | 0.048× |
| MLP GEMM, 33,390×8,192×2,048 | 0.0423 s | 0.0369 s | **1.144×** |
| Final quality projections, CFG and Euler update | 0.00994 s | 0.01797 s | 0.553× |

Encoder references use the same FP32 CPU/16-thread policy as the original
benchmark; native encoder layers execute on CUDA. DiT/VAE references use
FP16 CUDA. Native GEMM writes FP32 output and converts FP32 inputs, while the
reference consumes and writes FP16. The isolated MLP GEMM still has a 14.4%
gap; the complete measured DiT block and VAE tiles are faster than PyTorch.
These component measurements do not establish complete-video throughput.

A matching legacy VAE interior tile takes **25.872 s** for its first forward,
versus **3.876 s** for the candidate including initial weight transfer: 6.68×
faster. Its convolution chunks fall from 12,609 to 44 per tile. The historical
whole-video VAE gap below must not be treated as a newly measured candidate
whole-video speedup. The private encoder register tile reduces a warmed Qwen
layer from 0.06154 s to 0.02187 s; exact raw-weight transfer then reduces its
first forward from 1.395 s to 0.357 s. Repository v6 was also measured on the
MLP shape (0.0869 s warm) and rejected because it was slower than v7.

Independent native/reference states are carried through all **54 blocks** of
the first Fast12 I2V step and both branches of the first quality I2V step in
segments of at most eight blocks. Every segment passes cosine ≥0.9999 and
relative L2 ≤0.02. Final projection, CFG-6 prediction and first Euler update
also pass. The quality guided prediction relative L2 is 0.00899; its updated
latent is 5.77e-5. The worst final quality text-state relative L2 is 0.01174.
Four consecutive positive/negative quality T2V blocks also pass, with maximum
relative L2 0.000662. Both VAE tiles pass all 81 per-frame gates, with maximum frame relative L2
0.000298. Qwen/SigLIP/ByT5 layer errors are at most 3.15e-6.

The largest sampled candidate process VRAM is 9,584 MiB, below the approved
14,336 MiB cap. Pinned host staging is bounded to 768 MiB. The owned quality
baseline resumes after every check; its original binary and verifier remain
unchanged. No new full-video acceptance run was started. **Full-video speed
and quality of the optimized build remain provisional**, including later
denoising steps and VAE tile blending.

Receipts, raw fixtures and outputs are under `tmp/hv15-native/opt-results/`.
`performance-v3.json` summarizes accepted replay pairs (SHA256
`95f60b7406405d2685801a820c221b6eb56dfcfd91655aa61488ec2b45de531b`).
The pre-push audit rechecks the same saved tensors and leaves all timing samples
unchanged; `performance-v2.json` is retained as the earlier report. The state-carry
reports are `fast-chain16-54-v1/chain.json` (preceded by `fast-chain0-*` and
`fast-chain8-*`) and `quality-pair8-54-v1/chain.json` (preceded by
`quality-pair0-*`). Each GPU receipt binds its executable/source hashes,
timing budget, memory samples and baseline pause interval. Refreshed parity
reports bind fixture, raw output, metadata, timing and execution receipt hashes;
the reporter rejects changed or unbound artifacts. Audit regression tests cover
descendant cleanup before releasing the GPU and baseline resumption after a
stalled monitor. A strict repository GPU math check passes in 3.42 seconds with
zero vendor/fallback calls, and the frozen baseline resumes unchanged.
Reproduction is documented in the
[bounded replay procedure](../../ref/hunyuan_video15_native/README.md#bounded-optimization-replays).

## Completed Fast12 video

Both runs complete 12 Euler steps, CFG 1, shift 7, decode all 81 frames and
package a 24 fps MP4. The new reference output passes comparisons for every
required component, every step and all 81 frames (cosine ≥0.9999, relative
L2 ≤0.02).

| Measurement | Native repo GEMM | PyTorch reference | Native/reference |
|---|---:|---:|---:|
| Generation and MP4 packaging | 3690.20 s (61.50 min) | 724.79 s (12.08 min) | 5.09× |
| Denoising, including load/captures | 2555.29 s (42.59 min) | 594.21 s (9.90 min) | 4.30× |
| Warm denoising step, 11 samples | 210.77 s | 48.76 s | 4.32× |
| Full VAE decode, including load | 974.62 s (16.24 min) | 77.96 s (1.30 min) | 12.50× |
| Sampled process VRAM peak | 3108 MiB (3.04 GiB) | 9334 MiB (9.12 GiB) | — |
| Sampled process host RSS peak | 16237 MiB (15.86 GiB) | 40148 MiB (39.21 GiB) | — |

PyTorch encoder/image preparation stages: Qwen 28.14 s, empty ByT5 0.018 s,
SigLIP 4.21 s and portrait VAE encode 18.56 s. RGB/MP4 packaging takes 1.26 s.
The first denoising step takes 54.77 s; subsequent steps range 47.11–50.48 s.

The VAE has the largest relative gap. Denoising dominates native wall time
and accounts for the larger absolute time difference. Kernel profiling is
needed to attribute those phase differences to individual operations.

## Quality schedule throughput

The reference measures the first **four steps of the prescribed 50-step
schedule**, CFG 6, shift 5. It executes both positive and negative DiT passes
at each step. Conditioning, the first prediction and all four updated latents
pass independent comparisons; the fourth latent relative L2 is 1.20e-4.
This bounded comparison does not establish full quality video acceptance.

| Measurement | Native repo GEMM | PyTorch reference | Native/reference |
|---|---:|---:|---:|
| Warm step mean | 417.66 s (5 samples) | 90.91 s (3 samples) | 4.59× |
| Warm step range | 399.28–438.28 s | 88.19–92.64 s | — |
| Projected 50 warm denoising steps | 348.05 min (5.80 h) | 75.76 min (1.26 h) | 4.59× |

The first reference step takes 97.50 s and is excluded from the warm mean.
**The 50-step figures are projections**, excluding encoder/model loading,
the first-step overhead and VAE decode. A complete quality reference video
was not measured. Native timing samples use completed capture intervals and
subtract any recorded benchmark suspension that overlaps those intervals.

## Method and limitations

The reference uses FP32 CPU encoders (16 threads), FP16 CUDA DiT/VAE,
official Torch FlexAttention, complete transformer-block offload and
128-pixel VAE tiles. Native has FP32 activations and FP16 weights/projection
inputs. These are measured implementation comparisons with validated
numerical outputs; precision, attention, convolution and memory strategies
differ. CUDA wall timers synchronize, and include model loading, compilation
and capture writes. Pinned receipt verification/imports and numerical
comparison are outside the reference generation interval. Native phase
times are capture-file timestamp intervals; its total comes from its manifest.
The reference VAE phase ends after the CPU tensor copy; its final capture write
is included in the total generation time rather than that phase timer.

Desktop graphics remained active (Xorg sampled at 19% SM before the reference).
The historical native Fast12 run overlapped two short CPU encoder probes, so
the ratios are approximate comparisons rather than isolated hardware limits.
The owned quality process was suspended during the reference and resumed
automatically afterward. Its resident GPU memory is excluded from the
reference process peak. Suspension durations remain recorded separately
without rewriting the native generation manifest.

Reproduction commands and timing tests are in
[the reference README](../../ref/hunyuan_video15_native/README.md#reference-performance).
The measured invocation used `--quality-measure-steps 4 --threads 16` and
`--suspend-native-pid 700860` with the paths shown there, writing to
`tmp/hv15-native/reference-performance-v2/`.

## Evidence

- Native Fast12: `tmp/hv15-integration-review/full-repo-run/manifest.json`,
  SHA256 `e9c9598c86bb4a36770e85867d83b2d564b377c35888beab6ac0ce6ae2ecf864`.
- Complete timing/memory report: `tmp/hv15-native/reference-performance-v2/performance.json`,
  SHA256 `d564c0f4ec03857972b427472bc37290f5083c499588a9f17b0859e0c27e5555`.
- New Fast12 parity report: `tmp/hv15-native/reference-performance-v2/fast12/parity.json`,
  SHA256 `e33ad846a1d50a1a199b0412214216906a00cae3c29438758e46f7d406a6c413`.
- Quality prefix parity: `tmp/hv15-native/reference-performance-v2/quality/prefix_parity.json`,
  SHA256 `0310dfe18a1d264225d0cd5c22ea747db24ec5b1f3a8d5d7d5a8a47fb334371d`.
- Benchmark source SHA256:
  `d11ccc8bef6ffade16053db7073e00aee75f512037b865623d13e4f08f003f48`.
