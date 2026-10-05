# Wan 2.2 TI2V-5B on RX 9070 XT

This backend uses PyTorch ROCm/Diffusers for the Wan graph, UMT5, scheduler,
attention and VAE, and a native HIP adapter for the quantized DiT projections.
It supports text-to-video and first-frame image-to-video. The 14B dual-expert
models are outside this backend's scope.

The pinned [QuantStack Q8_0 checkpoint](https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF)
is 5.4 GB. Q8_0 is signed, block-scaled **weight-only INT8**. Each HIP projection
expands only its own weights into a temporary FP16 buffer using our fused HIP
decoder. The default `--hip-gemm blaslt` uses AMD hipBLASLt FP32 accumulation
with fused bias and FP16 output. It caches host-side plans by matrix shape and
bias presence, rebinds each layer's bias, and shares a 32 MiB PyTorch workspace
per stream. `--hip-gemm wmma` retains the shared Hunyuan WMMA kernels in
`rdna4/video_common/gemm.hip`; FP32 input callers also retain this path.
This is W8A16 inference, not H3's dynamic activation INT8/ConvRot algorithm.
All tensor storage belongs to PyTorch and launches use its current HIP stream. Nonquantized
projections use PyTorch. `--backend pytorch` provides an independent GGUF
dequantization/GEMM comparison path without loading our HIP adapter.

Use the existing repository-local ROCm environment (system Python has CUDA
PyTorch and is unsuitable):

```sh
TMPDIR="$PWD/tmp" uv pip install --python tmp/vhuman-rocm-venv/bin/python \
  gguf ftfy imageio-ffmpeg
make -C rdna4/wan22
LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib HF_HUB_DISABLE_XET=1 \
  TMPDIR="$PWD/tmp" tmp/vhuman-rocm-venv/bin/python rdna4/wan22/download.py
make -C rdna4/wan22 test
sh rdna4/wan22/run.sh \
  --prompt 'A red ball rolling on a wooden table, cinematic lighting.' \
  --out tmp/video-rocm/wan22-video
```

Add `--image IMAGE` for I2V. Defaults: 832×480, 81 frames, 24 fps, 50 updates,
guidance 5, seed 42. The official UniPC flow scheduler and expanded timestep
semantics are loaded from the pinned model configuration. Dimensions must be
multiples of 32; frames must be `4*n+1`. A reduced diagnostic uses
`--width 64 --height 64 --frames 5 --steps 2 --dump-latents`.

For fast optimization iterations, use full spatial resolution with nine frames
(Wan requires `4*n+1`, so ten frames are invalid), five updates and no VAE:

```sh
sh rdna4/wan22/run.sh --prompt 'A red ball rolling on a wooden table.' \
  --width 832 --height 480 --frames 9 --steps 5 --latent-only --dump-latents \
  --out tmp/video-rocm/wan22-fast-hip
```

Add `--backend pytorch` and use a new output directory to obtain the reference.
`--latent-only` writes `final_latent.npy` instead of an MP4. `compare.py` compares
the captured schedule when both runs use this option. The manifest records
`denoise_seconds`, including pipeline preparation, the first DiT upload and
optional capture overhead; `seconds` additionally includes loading and text
encoding, plus VAE/video work when enabled.

Weights live under `/mnt/disk01/models/wan22/{gguf,pipeline}`. Only the Q8_0 DiT,
official BF16 UMT5, tokenizer, FP32 VAE and configurations are downloaded;
the dense DiT and other quantizations are excluded. Allow roughly 20 GB disk
and a 64 GB host. The downloader pins both repositories and writes
`download.json`; completed 64 MiB HTTP ranges permit retries and each large
file is SHA-256 checked. `download.py --workers 16` increases transfer concurrency.

The default CPU text encoding keeps UMT5 out of VRAM; `--text-device gpu`
encodes on GPU before moving it back to CPU. Model CPU offload separates DiT
and VAE residency; spatial VAE tiling is enabled. The 14,336 MiB allocation
limit reserves device headroom, and the shared H3/Hunyuan AMD device lock
serializes generation. GPU access needs `/dev/kfd` and the AMD render node.
Builds target gfx1201 and use HIP 7.14 with hipBLASLt; Python requires a ROCm PyTorch build,
GGUF-capable Diffusers, Transformers, Accelerate, NumPy and video packaging.
The tested environment is PyTorch 2.11.0 ROCm 7.2.2 and Diffusers 0.41.0.dev0.

Output contains `video.mp4` and `manifest.json` with runtime, projection call
counts, configuration and sampled process/allocator VRAM. `--dump-latents`
writes each denoising update for comparison using the same seed, geometry,
prompt and scheduler under both backends. Output is experimental: projection
tests or reduced diagnostic runs do not establish full-video numerical parity
or full-resolution memory fit. The manifest records `parity: unverified`.

## Validation

`make -C rdna4/wan22 test` checks four projection shapes with signed values,
matrix tails, large K and a nondefault stream under both GEMM paths. It tests
no bias, replacement bias pointers in cached plans, FP32 callers and shaped
inputs. hipBLASLt fixtures and a two-block, 21-projection synthetic Wan graph
match the independent PyTorch reference exactly on the tested system. The
WMMA graph passes at relative L2 0.000512.

The following video runs and original DiT probes used the **WMMA** path before
the hipBLASLt optimization; optimized measurements follow in the performance
section.

The downloaded Q8_0 model has 300 quantized projection tensors across 30
blocks. An actual-weight complete DiT comparison on shared 2×4×4 latent noise,
32 text tokens and expanded timesteps passes: cosine **0.9999939**, relative
L2 **0.003477**, peak allocated VRAM **5,437 MiB**. This diagnostic uses
synthetic conditioning and does not establish full-video acceptance.

After GPU recovery, the synchronized **832×480, 81-frame** DiT comparison passes:
relative L2 **0.004740**, cosine **0.9999888**, and peak PyTorch allocation
**7,229 MiB**. Both the independent PyTorch pass and all 300 HIP projections
complete in **31.2 seconds** combined. The initial full-geometry attempt had
failed with AMD SDMA timeouts and `device lost from bus!`; its diagnostics remain
under `tmp/video-rocm/wan22-build/gpu-driver-errors.log`.

Actual end-to-end **64×64, five-frame, two-update** runs pass for both T2V and
I2V. Each executes 1,200 HIP projections and produces a 24 fps MP4. T2V takes
**44.9 seconds** with **5,442 MiB** peak PyTorch allocation; I2V takes
**42.8 seconds** with **8,139 MiB** peak allocation. CPU-encoded prompt embeddings
are explicitly moved to the GPU before denoising; this fixes a device mismatch
in the timestep conditioning path. These results establish reduced end-to-end
execution; full-schedule numerical parity remains unverified.

The default **832×480, 81-frame, 50-update** T2V profile completed after the GPU
reset in **2,592.7 seconds (43.2 minutes)**, including **30,000 HIP projections**,
tiled FP32 decoding and MP4 packaging. All 50 captured latent updates and all
81 decoded frames are finite. `ffprobe` confirms 832×480, 24 fps and 3.375 seconds.
Peak PyTorch allocation is **8,033 MiB (7.84 GiB)**. External physical VRAM
readings were approximately **8.6 GiB** during denoising and reached **10.33 GiB**
during decoding; the in-process VRAM sampler returned no measurement for this
run. The video has nonblank RGB content and temporal variation, and a decoded
frame was inspected. Outputs are under `tmp/video-rocm/wan22-full-reset/`.

A fresh PyTorch ROCm reference passes both updates of the five-frame, two-update
T2V diagnostic: maximum latent relative L2 **0.010934**, minimum cosine
**0.9999434**, and decoded MP4 pixel MAE **1.409 / 255**. This compares the HIP
projections with independent Diffusers GGUF dequantization and PyTorch GEMM on
identical model settings and noise. It establishes that reduced captured
schedule and encoded-video comparison, not raw-frame parity or full 50-update
reference parity. The manifest therefore retains `parity: unverified`.

```sh
sh rdna4/wan22/run.sh --backend pytorch \
  --prompt 'A red ball rolling on a wooden table.' \
  --width 64 --height 64 --frames 5 --steps 2 --dump-latents \
  --out tmp/video-rocm/wan22-pytorch-smoke
python3 rdna4/wan22/compare.py \
  tmp/video-rocm/wan22-smoke tmp/video-rocm/wan22-pytorch-smoke \
  --out tmp/video-rocm/wan22-pipeline-parity.json
```

The downloaded bundle also passes an independent CPU asset check: both T2V
and I2V pipelines load, UMT5 produces finite positive/negative `1×226×4096`
prompt embeddings, and the FP32 VAE decodes a finite `1×3×5×64×64` output.
This check executed on CPU using the system PyTorch 2.11.0 build; ROCm import
was blocked inside the failed AMD driver even with GPU visibility disabled.
It verifies asset loading and CPU execution, not GPU video generation.
The complete bundle is 19.6 GB; all five large checkpoint SHA-256 hashes and
both pinned repository revisions are in `/mnt/disk01/models/wan22/download.json`.

To rerun the GPU tests and reduced video diagnostic:

```sh
make -C rdna4/wan22 test
sh rdna4/wan22/run.sh \
  --prompt 'A red ball rolling on a wooden table.' \
  --width 64 --height 64 --frames 5 --steps 2 --dump-latents \
  --out tmp/video-rocm/wan22-smoke
```

The CPU asset check is available as `check_assets.py --out RESULT.json`.
`probe.py --width 832 --height 480 --frames 81 --out RESULT.json` repeats
the full-geometry DiT comparison; it is separate from full-video validation.

```sh
LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib TMPDIR="$PWD/tmp" \
  tmp/vhuman-rocm-venv/bin/python rdna4/wan22/probe.py \
  --out tmp/video-rocm/wan22-build/actual-dit-parity.json
```

## HIP versus PyTorch performance

On the RX 9070 XT, the full-geometry DiT benchmark uses the same Q8_0 weights,
832×480 / 81-frame geometry, 226 synthetic text tokens and timestep 500.
Loading, text encoding, scheduler and VAE are excluded. Each backend runs once
to warm up, then three synchronized passes alternate order in the same process.
Both retain PyTorch attention; only quantized projection execution differs.
The optimized path keeps our HIP Q8 decoder and replaces the generic WMMA
GEMM with hipBLASLt, including a fused bias/output epilogue. It does not invoke
Diffusers dequantization or PyTorch linear for quantized FP16 projections.

| Measurement | Repository HIP | PyTorch ROCm reference | HIP runtime gap |
| --- | ---: | ---: | ---: |
| Original WMMA, full-size DiT forward, median of 3 | 18.411 s | 10.215 s | +80.2% (1.802×) |
| Optimized hipBLASLt, full-size DiT forward, median of 3 | 10.209 s | 10.218 s | −0.09%, effectively on par |
| Optimized hipBLASLt, 9-frame DiT forward, median of 3 | 1.369 s | 1.374 s | −0.4%, effectively on par |
| Optimized hipBLASLt, 832×480 / 9 frames / 5 updates, preparation + denoising, single run | 17.427 s | 17.598 s | −1.0% |
| Original WMMA, 64×64 / 5 frames / 2 updates, end-to-end, single run | 44.941 s | 40.334 s | +11.4% (1.114×) |
| Original WMMA, 832×480 / 81 frames / 50 updates, end-to-end | 2,592.7 s | Unmeasured | Unmeasured |

Optimized full-size samples are PyTorch `[10.2267, 10.2177, 10.2181]` seconds
and HIP `[10.2071, 10.2107, 10.2087]` seconds: **44.5% less runtime** than the
original HIP median. The small advantage over PyTorch is within timing noise;
the target of on-par full-size DiT execution is met. Full-size and nine-frame
synthetic DiT outputs are exactly equal to PyTorch (relative L2 0, cosine 1).
Peak allocation in the paired full-size probe is **6,963 MiB**, including the
workspace. All five captured nine-frame schedule updates also match exactly.
That short schedule executes 3,000 HIP projections and peaks at 5,710 MiB;
loading, CPU UMT5, preparation and denoising total 47.59 seconds with HIP versus
47.55 seconds with PyTorch. VAE is skipped for this iteration profile.
The small end-to-end timings include CPU UMT5 and VAE, so they do not represent
DiT throughput. A full-schedule PyTorch reference is required to measure the
full-video gap. The optimized full 50-update video has not been benchmarked.
Q8 weights stay compressed between projections; no full-model FP16 cache is
introduced. Single-forward comparisons use synthetic conditioning, while the
five-update comparison uses actual CPU-encoded prompt embeddings.

```sh
LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib TMPDIR="$PWD/tmp" \
  tmp/vhuman-rocm-venv/bin/python rdna4/wan22/probe.py \
  --width 832 --height 480 --frames 81 --text-tokens 226 \
  --hip-gemm blaslt --benchmark-repeats 3 \
  --out tmp/video-rocm/wan22-build/blaslt-full-dit.json
```

Use `--frames 9` for faster iterations and `--hip-gemm wmma` for the old path.
Raw optimized results are `blaslt-full-dit.json`, `blaslt-short-dit.json` and
`blaslt-5step-parity.json` under `tmp/video-rocm/wan22-build/`.
`bench_projection.py --out RESULT.json` measures representative attention/FFN
shapes with CUDA/HIP events, including decode, GEMM and bias, for independent
PyTorch, original WMMA and optimized hipBLASLt on synthetic Q8 weights.

The projection benchmark confirms the main saving at 8,190 latent tokens
(median of five, milliseconds, including Q8 decoding and bias):

| Projection N×K | Original WMMA | Optimized hipBLASLt | PyTorch |
| --- | ---: | ---: | ---: |
| Attention 3,072×3,072 | 11.51 | 3.48 | 3.39 |
| FFN expansion 14,336×3,072 | 187.92 | 27.40 | 27.32 |
| FFN reduction 3,072×14,336 | 98.62 | 31.23 | 31.60 |

These isolated projection timings explain the DiT improvement; they do not
measure attention or scheduler execution. All optimized projection outputs in
this benchmark match PyTorch exactly. Raw samples are in
`tmp/video-rocm/wan22-build/projection-performance.json`.
