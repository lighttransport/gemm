# Native and PyTorch performance comparison

Measured on 2026-10-03 (JST), NVIDIA RTX 5060 Ti 16 GB, driver 615.71.09,
PyTorch 2.14.0+cu130. Both paths use the same pinned weights, portrait, input
noise, prompt and 480×848, 81-frame I2V recipe. Native uses repository GEMM
with zero vendor/fallback calls. The reference uses the clean official source
at `60783e704160023913bee78f0b47036d393d4dfa`.

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
