# RDNA4 HIP/ROCm Runners

GPU inference runners for VLM, LLM, DA3, and PPD using ROCm/HIP with runtime kernel compilation via HIPRTC. Targets AMD RDNA4 (RX 9070 XT, gfx1201).

No `hipcc` needed at build time - kernels are compiled at runtime via HIPRTC, loaded dynamically through `rocew`.

## Architecture

- **Target**: AMD RDNA4 (gfx1200/gfx1201), 64 CUs, wave size 32
- **WMMA matrix engine**: BF16/FP16 (`v_wmma_f32_16x16x16_bf16/f16`) and FP8 e4m3 (`v_wmma_f32_16x16x16_fp8_fp8`) on gfx1201. Microbench peaks: BF16 195 TF/s, FP8 351 TF/s (8-wave). Tuned BF16 mm0 sustains 174 TF/s (89% peak); standalone FP8 mm0 via extracted hipBLASLt kernel sustains 218 TF/s.
- **Runtime compilation**: HIPRTC compiles HIP C kernel strings at program startup
- **Dynamic loading**: `rocew` (ROCm Extension Wrangler) loads `libamdhip64.so` + `libhiprtc.so` via dlopen. When present, it prefers `/opt/rocm/core-10.0/lib` over an older system `libamdhip64.so.5` in the linker cache.

## Runners

| Runner | Description | Source Lines | Input |
|--------|-------------|--------------|-------|
| **VLM** | Qwen3-VL vision encoder (mmproj) | ~1400 | GGUF mmproj + image |
| **LLM** | Qwen3-style transformer (F16/Q8_0/Q2-Q6_K) | ~5500 | GGUF model |
| **DA3** | Depth Anything 3 (depth + pose + rays + gaussians) | ~2000 | GGUF or safetensors |
| **PPD** | Pixel-Perfect Depth (DA2 encoder + DiT diffusion) | ~2700 | PyTorch .pth |
| **MiniMax H3** | Native Ref2VA INT8 ConvRot text-to-video with INT8 WMMA | [runner](minimax_h3/README.md) | Local H3 safetensors |
| **HunyuanVideo 1.5** | Native quality T2V/I2V and fast12 I2V | [runner](hunyuan_video15_native/README.md) | Existing native model manifest |
| **Qwen Image 2.1** | 32-block BF16 denoiser with RDNA4 WMMA GEMM | native fixture ABI | Qwen 2.1 safetensors |

## Requirements

- GCC (build time only - no hipcc/ROCm SDK needed to compile)
- ROCm 6.x+ runtime: `libamdhip64.so`, `libhiprtc.so` (in `/opt/rocm/lib/`
  or versioned core prefixes such as `/opt/rocm/core-10.0/lib/`)
- AMD GPU with RDNA4 architecture (gfx1200 or gfx1201)

## Build

Each runner has its own Makefile:

```bash
# Vision encoder
cd vlm && make

# LLM transformer
cd llm && make

# Depth Anything 3
cd da3 && make

# Pixel-Perfect Depth
cd ppd && make

# Qwen Image 2.1 denoiser
cd qimg21 && make
```

## Run

```bash
# VLM: multimodal inference (vision + LLM)
cd vlm && ./test_hip_vlm <model.gguf> <mmproj.gguf> <image.jpg> [-n max_tokens]

# LLM: text-only inference (compare GPU vs CPU)
cd llm && ./test_hip_llm <model.gguf> [-t "prompt"] [-n max_tokens]

# DA3: depth estimation
cd da3 && ./test_hip_da3 <da3.gguf> -i image.jpg -o depth.exr [--full]

# PPD: pixel-perfect depth
cd ppd && ./test_hip_ppd <ppd.pth> <da2_vitl.pth> [-i image.ppm] [-o depth.pgm]
```

## Directory Structure

```
rdna4/
├── rocew.h, rocew.c            # ROCm Extension Wrangler (dynamic HIP/HIPRTC loader)
├── hip_runner_common.h         # Shared host utilities (error macros, HIPRTC compile, upload)
├── hip_kernels_common.h        # Shared GPU kernel source strings (GEMM, layernorm, etc.)
├── vlm/
│   ├── hip_vision_encoder.h    # Vision encoder API
│   ├── hip_vision_encoder.c    # Vision encoder implementation
│   ├── test_hip_vlm.c          # VLM test program
│   └── Makefile
├── llm/
│   ├── hip_llm_runner.h        # LLM runner API
│   ├── hip_llm_runner.c        # LLM runner implementation
│   ├── test_hip_llm.c          # LLM test program
│   └── Makefile
├── da3/
│   ├── hip_da3_runner.h        # DA3 runner API
│   ├── hip_da3_runner.c        # DA3 runner implementation
│   ├── test_hip_da3.c          # DA3 test program
│   └── Makefile
├── ppd/
│   ├── hip_ppd_runner.h        # PPD runner API
│   ├── hip_ppd_runner.c        # PPD runner implementation
│   ├── test_hip_ppd.c          # PPD test program
│   └── Makefile
└── README.md
```

## Key Differences from CUDA Version

| CUDA | HIP (RDNA4) |
|------|-------------|
| NVRTC runtime compilation | HIPRTC runtime compilation |
| cuew dynamic loader | rocew dynamic loader |
| MMA tensor core GEMM (`gemm_f16_f32`) | WMMA `v_wmma_f32_16x16x16_bf16/f16` + tiled fallback |
| FP8 E4M3 MMA (`gemm_fp8_f32`) | WMMA `v_wmma_f32_16x16x16_fp8_fp8` (gfx1201 only) — see `rdna4/fp8/` |
| MMA prefill attention | Tiled flash attention (`flash_attn_tiled_f32`) |
| PTX inline ASM (`cvt.f32.f16`) | HIP builtins (`__half2float`) |
| `__shfl_down_sync(mask, val, off)` | `__shfl_down(val, off)` |
| Warp size 32 (NVIDIA) | Wave size 32 (RDNA4) |

## PyTorch video performance comparison

[video_common/bench_pytorch.py](video_common/bench_pytorch.py) measures upstream
ComfyUI generation with ROCm PyTorch, local H3 INT8 ConvRot or HunyuanVideo 1.5
fast12 weights, and PyTorch SDPA. Put a ComfyUI checkout and its dependencies
on `PYTHONPATH`; the measured checkout revision and benchmark source hash are
recorded in each result. The current benchmark setup uses ComfyUI
`2472a20bd291451acc303917059ab14dfc380478`, comfy-kitchen 0.2.37 and
ROCm PyTorch 2.11.0. No server or custom nodes are used.

Serialize these measurements with the native video runners using the same
`tmp/pixal3d/device-locks/rocm-0.lock` lock. The prepared environment can be
invoked through the wrapper below, which sets the isolated dependency paths
and takes that lock:

```sh
sh rdna4/video_common/bench_pytorch_rocm.sh \
  --model hv15 --image tmp/video-rocm/hv-fast12-full/input.png \
  --width 480 --height 848 --frames 81 --steps 12 \
  --out tmp/video-rocm/pytorch-hunyuan-benchmark
sh rdna4/video_common/bench_pytorch_rocm.sh \
  --model h3 --width 864 --height 480 --frames 360 --steps 20 --lowvram \
  --out tmp/video-rocm/pytorch-h3-benchmark
```

H3 rounds 360 requested frames to 362, or 15.08 seconds at 24 fps. `--steps`
counts sampler updates. `--lowvram` selects upstream low-VRAM loading and
disables dynamic VRAM. Completed encoders are released before loading the DiT
to bound host memory. `--smoke` checks a 64×64, five-frame, one-update graph.
Results include per-stage and per-update timings, peak PyTorch allocation,
all-frame finite/shape checks and a preview. Total time includes model loading,
conditioning, sampling and video decoding; Python imports, audio decoding,
MP4 encoding and diagnostic capture writes are excluded. These measurements
do not establish native numerical parity.

An initial RX 9070 XT measurement of Hunyuan fast12 at 480×848, 81 frames and
12 updates completed in **1,188.8 seconds (19.81 minutes)**. Sampling took
**470.8 seconds** and tiled decoding **597.5 seconds**; all 81 decoded frames
passed finite and shape checks. Peak PyTorch allocation was **13,540.6 MiB**.
This run retained encoders on the host; its exact benchmark source is preserved
with the diagnostic artifacts. The earlier native repository-GEMM run took
6,160.9 seconds, a 5.18× wall-time difference. These implementations differ in
precision handling, tiling and capture overhead; this is a pipeline timing
comparison, not a controlled GEMM speedup or a numerical-parity result.

H3 INT8 ConvRot at 864×480, 362 frames and 20 updates completed on the same
RX 9070 XT in **3,848.1 seconds (64.13 minutes)** with low-VRAM loading.
Sampling took **3,255.1 seconds (54.25 minutes)**, video decoding
**192.8 seconds (3.21 minutes)**, and CPU text conditioning **369.5 seconds
(6.16 minutes)**. All 362 decoded frames passed finite and shape checks.
Peak PyTorch allocation was **10,860.7 MiB**, with **13,108 MiB** reserved.
The run uses joint video/audio sampling but does not decode audio.

A subsequent measurement used a shorter H3 workload:
864×480, 73 frames (3.04 seconds), the same prompt and 20 updates. It completed
in **654.0 seconds (10.90 minutes)**, including **167.2 seconds** of CPU
conditioning, **425.3 seconds** of sampling and **41.8 seconds** of decoding.
All 73 decoded frames passed finite and shape checks. CPU conditioning in the
earlier run took 369.5 seconds, but the shortened clip also permits greater
GPU weight residency; total times are not a matched workload comparison.

The subsequent Hunyuan measurement retained the exact earlier benchmark source,
weights, prompt, seed, dimensions and 12 updates. It completed in
**1,127.3 seconds (18.79 minutes)**, **5.2% less total time** than the earlier
1,188.8-second run. Sampling took **468.3 seconds** (previously 470.8), while
decoding took **570.4 seconds** (previously 597.5). All 81 decoded frames passed
finite and shape checks. This is a single rerun; cache warmth and ordinary
run variation may also contribute to the difference.

## License

MIT License - Copyright 2025 Light Transport Entertainment Inc.
