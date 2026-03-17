# VLM/LLM Inference Features

## Supported Models

### LLM (Text Generation)

| Model | Architecture | CPU | CUDA | Vulkan | A64FX |
|-------|-------------|-----|------|--------|-------|
| Qwen2 | Dense transformer | x | x | x | - |
| Qwen3 | Dense transformer | x | x | x | - |
| Qwen3-MoE | Mixture of Experts | x | x | - | - |
| Qwen3.5 | Hybrid SSM+Attention (Delta-Net) | x | x | - | - |
| Qwen3.5-MoE | Hybrid SSM+Attention + MoE (256 experts, top-8) | x | x | - | - |

### VLM (Vision-Language)

| Model | Architecture | CPU | CUDA | Vulkan |
|-------|-------------|-----|------|--------|
| Qwen3-VL | CLIP vision encoder + M-RoPE + DeepStack | x | x | x |
| Qwen3-VLMoE | Vision-Language + MoE | x | x | - |

### Depth Estimation

| Model | Architecture | CPU | CUDA |
|-------|-------------|-----|------|
| Depth Anything 3 (DA3) | DINOv2 ViT encoder + DPT head | x | x |
| Pixel-Perfect Depth (PPD) | DINOv2 ViT-L + DiT diffusion (4-step Euler) | x | x |

DA3 supports multiple output modalities: depth + confidence, pose estimation (CameraDec), ray extraction + sky segmentation, 3D Gaussians, and metric depth. Variants: NESTED, LARGE, GIANT.

## Key Inference Features

### Model Loading
- **GGUF format** with automatic architecture detection (model type, MoE params, SSM layers, M-RoPE sections)
- **SafeTensors format** for DA3 models
- **PyTorch PTH format** with BF16/FP16 support

### Quantization
24 GGML quantization types supported on both CPU and CUDA. See README.md for full list.

### Attention & Decoding
- FlashAttention with two-pass online softmax, tiled computation, K/V transpose optimization
- Grouped Query Attention (GQA)
- M-RoPE (multi-dimensional RoPE) for vision models (spatial + temporal dimensions)

### MoE (Mixture of Experts)
- Top-K expert routing with softmax gating
- Shared expert path (routed + shared)
- Up to 256 experts with top-8 selection (Qwen3.5-MoE)
- Thread pool with persistent workers for expert parallelism

### SSM (State Space Model) - Qwen3.5 Hybrid
- Delta-Net SSM layers alternating with full attention layers
- Conv1d with circular buffer
- Configurable: kernel size 4, state dim 128, inner hidden 6144, combined QKV 10240

### Vision Encoding (Qwen3-VL)
- CLIP-style vision encoder with dynamic resolution
- Bilinear positional embedding interpolation
- DeepStack multi-scale vision token merging
- Vision token injection into LLM via `forward_embd` API

### Distributed Inference
- Custom collective communication library (no MPI dependency)
- Pipeline parallelism, tensor parallelism, data parallelism

### Optimizations by Architecture

**CPU (x86-64 AVX2/FMA):**
- Vectorized RMSNorm, RoPE, batch norms, attention dot products, GEMM fallbacks
- Batch dequantization with prefetch hints
- Multi-threaded vision encoder with batch LayerNorm
- Fused matvec operations

**CUDA:**
- NVRTC runtime kernel compilation (no nvcc needed at build time)
- F16 and Q8_0 weight support with auto-dispatch
- MMA m16n8k16 tensor core operations (verified on Blackwell)
- Flash attention variants with shared memory optimization
- Tiled GEMM, im2col, FP8 K/V cache

**Vulkan:**
- Cooperative matrix GEMM (RDNA4)
- Q8_0/F16 batched prefill and decode
- Command buffer coalescing

**A64FX (SVE):**
- FEXPA-based fast exp2 for attention
- INT8 SDOT microkernels for FFN
- Sector cache pragmas for memory optimization
- NUMA-aware allocation for 12-core CMG scaling

## TODOs

### CUDA DA3
- `cuda/da3/cuda_da3_runner.c:3564` - Inject merger features at level 0 (add to d_dpt_adapted[0])

### CUDA INT8
- `cuda/int8/int8_gemm.c:2081` - tcgen05.mma syntax needs adjustment (Blackwell tensor core instruction)

### Vulkan
- `vulkan/vulkan_llm_runner.cc:1762` - Implement batched deepstack injection for VLM
- `vulkan/deps/vulkan-runner.cc:968,998` - Fix VkPhysicalDevice vs VkDevice handling in device management
- `vulkan/deps/vulkan-runner.cc:1009` - Implement proper Vulkan memory deallocation

### A64FX
- `a64fx/mem-pattern/bench_streaming_s.c:214` - True streaming would use TBL to extract S values
