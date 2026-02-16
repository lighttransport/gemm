# GEMM and Attention experiment

High-performance GEMM and FlashAttention kernels for various architectures.

## A64FX (Fujitsu Fugaku)

### INT8 GEMM (SDOT)
- **94% SDOT efficiency** achieved with 6x4 microkernel
- 12-core CMG scaling with NUMA-aware memory allocation
- See: `a64fx/int8-cmg/OPTIMIZATION_SUMMARY.md`

### FP32/FP16 GEMM
- **92% FP32 peak** with 8x3 microkernel on 12 cores
- **90% FP16 peak** with optimized broadcast kernels
- See: `a64fx/int8-cmg/FP32_CMG12_92PERCENT_ACHIEVED.md`

### FlashAttention exp2+GEMM
- **45% FP32 peak** for fused exp2+GEMM (Stage 2)
- FEXPA-based fast exp2 approximation
- LD1RW vs DUP analysis: DUP is faster (eliminates P buffer overhead)
- See: `a64fx/exp2-sve/README.md`, `a64fx/exp2-sve/EXP2_FMLA_SUMMARY.md`

### INT8 FFN Kernels
- Fused INT8 GEMM + activation (SiLU, GELU)
- LayerNorm optimizations
- See: `a64fx/int8-new/README.md`

## Other Architectures

* [ ] Ryzen (x86-64 AVX-512)
* Vulkan
  * [x] Compute shaders
  * [x] Cooperative matrix (Tested on RDNA4)
* [ ] CUDA (planned)
