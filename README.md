# GEMM, Attention, and LLM/VLM Inference

High-performance GEMM, FlashAttention kernels, and full LLM/VLM inference engines for multiple architectures. What started as GEMM optimization experiments has grown into a complete inference stack supporting Qwen-family LLMs, vision-language models, and depth estimation models.

## Key Components

| Component | Location | Description |
|-----------|----------|-------------|
| Transformer Engine | `common/transformer.h` | Header-only LLM inference (Qwen2/3/3.5, MoE, SSM) |
| Vision Encoder | `common/vision_encoder.h` | CPU vision encoder for Qwen3-VL |
| Depth Anything 3 | `common/depth_anything3.h` | CPU monocular depth estimation |
| Pixel-Perfect Depth | `common/pixel_perfect_depth.h` | CPU diffusion-based depth |
| GGML Dequantization | `common/ggml_dequant.h` | 24 quantization types |
| GGUF Loader | `common/gguf_loader.h` | Model loading from GGUF format |
| BPE Tokenizer | `common/bpe_tokenizer.h` | Qwen2/Qwen3 tokenization |
| Distributed Comm | `common/comm.h` | Custom collective communication library |
| CUDA LLM Runner | `cuda/llm/` | NVRTC-based CUDA LLM inference |
| CUDA Vision Encoder | `cuda/vlm/` | CUDA vision encoder with dynamic resolution |
| CUDA DA3 Runner | `cuda/da3/` | CUDA depth estimation (all output modalities) |
| CUDA PPD Runner | `cuda/ppd/` | CUDA diffusion-based depth |
| Vulkan LLM Runner | `vulkan/` | Vulkan LLM with cooperative matrix |
| SAM 2 / SAM 2.1 | `ref/sam2/`, `cpu/sam2/`, `rdna4/sam2/`, `cuda/sam2/` | Point/box-prompted image (+ video) segmentation (Meta). Shared arch for 2.0 and 2.1 weights. |
| SAM 3 | `ref/sam3/`, `cpu/sam3/`, `rdna4/sam3/`, `cuda/sam3/` | Concept-level (text + exemplar) promptable segmentation (Meta, 2025). |

See [VLM_LLM.md](VLM_LLM.md) for detailed VLM/LLM feature matrix, supported models, and TODOs.

## Supported Architectures

### A64FX (Fujitsu Fugaku) - ARM SVE

- **INT8 GEMM (SDOT):** 94% efficiency with 6x4 microkernel, 12-core CMG scaling with NUMA-aware allocation
- **FP32 GEMM:** 92% peak with 8x3 microkernel
- **FP16 GEMM:** 90% peak with optimized broadcast kernels
- **FlashAttention:** 45% FP32 peak for fused exp2+GEMM (FEXPA-based fast exp2)
- **INT8 FFN:** Fused GEMM + activation (SiLU, GELU) with LayerNorm
- **SVE Kernels:** Embedding, Philox RNG, sector cache optimizations

See: `a64fx/int8-cmg/OPTIMIZATION_SUMMARY.md`, `a64fx/exp2-sve/README.md`

### NVIDIA CUDA (sm_70 ~ sm_100+)

- V100 (sm_70), A100 (sm_80), H100 (sm_90), Blackwell (sm_100+)
- NVRTC runtime compilation (no nvcc build dependency)
- INT8/FP8/BF16/FP16 GEMM via cuBLAS
- Full LLM, VLM, DA3, and PPD inference

### Vulkan (Cross-platform GPU)

- Compute shaders with cooperative matrix support
- Tested on AMD RDNA4
- LLM runner with Q8_0/F16 weight support, batched prefill

### x86-64 (Ryzen / Zen2)

- AVX2/FMA vectorized GEMM, RMSNorm, RoPE, attention, batch norms
- Fallback path for all common operations

## Quantization Support (24 types)

K-quant: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K | Legacy: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 | IQ: IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS | TQ: TQ1_0, TQ2_0 | Float: F32, F16, BF16

## A64FX Profiling

Performance report tool for A64FX PMU events: `a64fx/preport/`

## License

See [LICENSE](LICENSE).
