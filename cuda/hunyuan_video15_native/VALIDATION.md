# Native port validation record

Hardware: NVIDIA RTX 5060 Ti, 16 GiB, compute capability 12.0. Reference source:
`60783e704160023913bee78f0b47036d393d4dfa`. Model revisions and SHA256 values
are recorded by `stage_models.py` in its separate model manifest.

The checks below include independent component comparisons and complete fast12
video parity. They do **not** establish acceptance of either quality pipeline
or trained rig quality.
Runtime receipts keep `parity: unverified` and require `--allow-experimental`.

## Verified correctness

| Component | Shape/scope | Cosine | Relative L2 |
|---|---|---:|---:|
| Qwen hidden state 26 | Full encoder, cropped prompt | 0.999999999999 | 1.05e-6 |
| ByT5 glyph encoder | Full encoder, quoted text | 0.999999999999 | 2.30e-7 |
| Google SigLIP | 729 image patches, 27 blocks | 0.999999999945 | 1.05e-5 |
| Causal VAE encoder | 1 frame, 128×128 | 0.999999773 | 6.97e-4 |
| Causal VAE decoder | 5 frames, 128×128; all five pass separately | 0.999999196 | 1.27e-3 |
| Fast12 I2V DiT | All 54 blocks; latent 2×4×3; zero image condition | 0.999974840 | 7.10e-3 |
| Quality I2V DiT | All 54 blocks; latent 2×4×3; zero image condition | 0.999974911 | 7.08e-3 |
| Quality T2V DiT | All 54 blocks; latent 2×4×3 | 0.999962524 | 8.67e-3 |

Acceptance thresholds are cosine ≥0.9999 and relative L2 ≤0.02. Encoders were
compared to independent FP32 Transformers references, DiT/VAE to the pinned
official FP16 graphs. Captures and detailed receipts are local artifacts under
`tmp/hv15-native/*-reference/`. Qwen token IDs also match the reference for the
entire system-prefix/prompt/assistant sequence and a Japanese prompt.

The DiT tensor-lifetime fix is byte-exact against the earlier bounded fast12
capture. It releases packed/separate QKV tensors before attention/MLP stages
and avoids retaining the pre-activation MLP tensor through its second GEMM.

## Commands and checks

```sh
make -C cuda/hunyuan_video15_native -j4
make -C cuda/hunyuan_video15_native test
make -C cuda/hunyuan_video15_native compile-kernels
make -C cuda/hunyuan_video15_native ../../tmp/hv15-native/build/test_gpu
flock -w 30 tmp/pixal3d/device-locks/cuda-0.lock tmp/hv15-native/build/test_gpu
flock -w 30 tmp/pixal3d/device-locks/cuda-0.lock tmp/hv15-native/build/test_gpu repo-only
flock -w 30 tmp/pixal3d/device-locks/cuda-0.lock tmp/hv15-native/build/test_gpu cublas
flock -w 30 tmp/pixal3d/device-locks/cuda-0.lock tmp/hv15-native/build/test_gpu memory
```

The build is warning-clean with `-Wall -Wextra -Wpedantic`. Host tests include
all 65,536 half encodings, schedule/bucket checks, glyph parsing, tokenizer and
request bounds. Fourteen Python tests exercise process cancellation, actual FFmpeg
81-frame packaging, atomic publication through the existing video listing,
crop restoration and fitted expression controls, review/seed/hash gates,
candidate failure state, and fail-closed reference acceptance. The quality
controller tests also verify lock cancellation/release, exact generation
receipt checks, PID-bound adoption cleanup, and preservation of a real Python
virtual environment when its interpreter is a symlink.

GPU tests pass against independent double-precision CPU calculations for
FP16 and IEEE GEMM (including tall/tail matrices), multi-key-tile attention,
causal/frame-causal masking, GQA, T5 relative bias, and three normalization
modes. The default backend reports 94 repo GEMMs and 2 explicit IEEE cuBLAS
fallbacks. `repo-only` reports 96 repo GEMMs and zero vendor calls;
`cublas` reports 96 vendor GEMMs and zero repo calls.

The 33,390×2,048 by 8,192×2,048 memory probe reports about 1,368 MiB of live
tensors with device memory growth below that amount plus the 512 MiB guard.
The original/private GEMM kernels use 72/92 registers and zero local bytes.
This isolates GEMM scratch behavior; it is not a full model memory-fit result.

## Complete native fast12 execution

On 2026-10-02, the native runtime completed all 12 denoising steps and decoded
all 81 frames at 480×848, 24 fps, using the explicit `--gemm cublas` comparison
backend. The seed was 42 and the prompt was
`The same person smiles gently, then relaxes. Fixed frontal camera.`
The MP4 was decoded independently with FFmpeg, which reported exactly 81 frames.

| Measurement | Result |
|---|---:|
| Generation and packaging time | 3206.22 s (53.44 min) |
| Managed allocation peak | 2965.52 MiB |
| Sampled process VRAM peak | 3206 MiB |
| Sampled host RSS peak | 16740.80 MiB |
| Measured memory budget | Pass, 14336 MiB |
| GEMM calls | 6,068,402 cuBLAS; zero repo calls |
| Attention calls | 815 |

Local artifacts are `tmp/hv15-native/full-fast12-v5/{clip.mp4,manifest.json,
metrics.json,codec_check.log}` and `tmp/hv15-native/full-fast12-captures-v5/`.
The captures include matched noise, conditioning, every denoising step, the
final latent, and the full decoded tensor. The manifest binds the executed
binary, source, input and pinned weights to this run.

Independent CPU references for this exact completed run pass:

| Component | Cosine | Relative L2 |
|---|---:|---:|
| Qwen hidden state 26 | 0.999999999999531 | 9.69e-7 |
| SigLIP, including portrait preprocessing | 0.999999999944709 | 1.05e-5 |
| ByT5 empty glyph condition (no quoted text in this prompt) | 1 | 0 |

The receipts are in `tmp/hv15-native/full-fast12-v5-reference/`. The nonempty
ByT5 encoder comparison is recorded in the component table above.
On 2026-10-03, the full portrait VAE comparison passed and all reference tensors
from the other agent's completed integration review were audited against the
exact generation manifest, pinned receipts, shapes and capture hashes. Those
verified independent references were copied into the isolated reference output;
the final `compare` phase then passed for all 20 required captures and 81 frames.

| Full fast12 comparison | Cosine | Relative L2 |
|---|---:|---:|
| Portrait VAE, 848×480 | 0.999999809 | 6.53e-4 |
| First DiT prediction | 0.999998721 | 1.60e-3 |
| Final latent (all 12 steps pass) | 0.999996716 | 2.56e-3 |
| Decoded RGB, all 81 frames | 0.999991876 | 4.03e-3 |
| Worst decoded frame | 0.999967295 | 8.09e-3 |

The integration review also completed a strict repository-GEMM fast12 run with
zero cuBLAS/fallback calls. A read-only audit verified its exact input noise,
prepared images, complete recipe, manifests, reference ledger, all tensor
hashes and comparison report. Final latent relative L2 is 2.15e-3, full RGB
3.14e-3, and worst frame 6.66e-3; all 81 frames pass. Sampled VRAM is 3108 MiB,
host RSS 16237.23 MiB, and wall time 3690.20 s (with concurrent encoder probes,
so this is not an isolated performance benchmark). The report is
`tmp/hv15-integration-review/full-repo-parity.json`, SHA256
`2ad09e60e363120d34434fc1b72b17fd86e2e011092dd3cd9b9f493e93ece448`;
the audit receipt is `tmp/hv15-native/full-repo-fast12-audit.json`.

A same-input repo/cuBLAS first-step cross-check also passes: relative L2 is
8.80e-5 for the full portrait latent, 2.39e-4 for the DiT prediction, and 4.31e-6
for the updated latent; noise and encoder captures are byte-identical.
This first-step backend cross-check remains distinct from the complete
independent comparisons above. Neither complete 50-step quality pipeline has
yet been validated. The resumable controller has started that campaign at
`tmp/hv15-native/quality-campaign-v2/`, adopting the strict-GEMM I2V run at
`tmp/hv15-native/full-quality-i2v-v1/` before queuing T2V. Its receipt remains
`pass_both_quality_pipelines: false` while work is pending.

For complete reference phases and exact capture requirements, see
[the reference instructions](../../ref/hunyuan_video15_native/README.md).
