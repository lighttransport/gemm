/* Repository FP32 GEMM/GEMV NVRTC sources. Include inside extern "C".
 * GEMM: Y[M,N] = X[M,K] W[N,K]^T + bias[N]; block (16,16).
 * GEMV: Y[N] = W[N,K] X[K]; one 256-thread block per row.
 * No reduced-precision inputs, vendor libraries, or K alignment required. */
#ifndef CUDA_GEMM_F32_KERNELS_H
#define CUDA_GEMM_F32_KERNELS_H

#define CUDA_GEMM_F32_BIAS_SRC \
"__global__ void gemm_f32_bias(float *Y, const float *X, const float *W,\n" \
"                              const float *b,\n" \
"                              int N, int D_in, int D_out) {\n" \
"    int n = blockIdx.x * blockDim.x + threadIdx.x;\n" \
"    int d = blockIdx.y * blockDim.y + threadIdx.y;\n" \
"    if (n >= N || d >= D_out) return;\n" \
"    const float *xr = X + (size_t)n * D_in;\n" \
"    const float *wr = W + (size_t)d * D_in;\n" \
"    float acc = (b ? b[d] : 0.0f);\n" \
"    for (int k = 0; k < D_in; k++) acc += wr[k] * xr[k];\n" \
"    Y[(size_t)n * D_out + d] = acc;\n" \
"}\n"

#define CUDA_GEMV_F32_SRC \
"__global__ void mhr_matvec_f32(float *Y,\n" \
"                               const float *W,\n" \
"                               const float *X,\n" \
"                               int D_in, int D_out) {\n" \
"    int row = blockIdx.x;\n" \
"    if (row >= D_out) return;\n" \
"    int tid = threadIdx.x;\n" \
"    float acc = 0.0f;\n" \
"    const float *wr = W + (size_t)row * D_in;\n" \
"    for (int k = tid; k < D_in; k += blockDim.x) acc += wr[k] * X[k];\n" \
"    __shared__ float red[256];\n" \
"    red[tid] = acc;\n" \
"    __syncthreads();\n" \
"    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {\n" \
"        if (tid < stride) red[tid] += red[tid + stride];\n" \
"        __syncthreads();\n" \
"    }\n" \
"    if (tid == 0) Y[row] = red[0];\n" \
"}\n"

#endif
