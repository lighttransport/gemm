# RDNA4 HIP/ROCm Runners

GPU inference runners for VLM, LLM, DA3, and PPD using ROCm/HIP with runtime kernel compilation via HIPRTC. Targets AMD RDNA4 (RX 9070 XT, gfx1201).

No `hipcc` needed at build time - kernels are compiled at runtime via HIPRTC, loaded dynamically through `rocew`.

## Architecture

- **Target**: AMD RDNA4 (gfx1200/gfx1201), 64 CUs, wave size 32
- **WMMA matrix engine**: BF16/FP16 (`v_wmma_f32_16x16x16_bf16/f16`) and FP8 e4m3 (`v_wmma_f32_16x16x16_fp8_fp8`) on gfx1201. Microbench peaks: BF16 195 TF/s, FP8 351 TF/s (8-wave). Tuned BF16 mm0 sustains 174 TF/s (89% peak); standalone FP8 mm0 via extracted hipBLASLt kernel sustains 218 TF/s.
- **Runtime compilation**: HIPRTC compiles HIP C kernel strings at program startup
- **Dynamic loading**: `rocew` (ROCm Extension Wrangler) loads `libamdhip64.so` + `libhiprtc.so` via dlopen

## Runners

| Runner | Description | Source Lines | Input |
|--------|-------------|--------------|-------|
| **VLM** | Qwen3-VL vision encoder (mmproj) | ~1400 | GGUF mmproj + image |
| **LLM** | Qwen3-style transformer (F16/Q8_0/Q2-Q6_K) | ~5500 | GGUF model |
| **DA3** | Depth Anything 3 (depth + pose + rays + gaussians) | ~2000 | GGUF or safetensors |
| **PPD** | Pixel-Perfect Depth (DA2 encoder + DiT diffusion) | ~2700 | PyTorch .pth |

## Requirements

- GCC (build time only - no hipcc/ROCm SDK needed to compile)
- ROCm 6.x+ runtime: `libamdhip64.so`, `libhiprtc.so` (in `/opt/rocm/lib/`)
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

## License

MIT License - Copyright 2025 Light Transport Entertainment Inc.
