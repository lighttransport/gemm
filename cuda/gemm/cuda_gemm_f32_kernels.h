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

/* Device-resident training GEMM, including both transpose forms. Row-major
 * C[M,N]=op(A)[M,K]op(B)[K,N]. Shared tiles avoid rereading K per output. */
#define CUDA_GEMM_F32_TRAIN_SRC \
"__global__ void gemm_f32_train(float *C,const float *A,const float *B,\n" \
"    int M,int N,int K,int ta,int tb) {\n" \
"    __shared__ float a[16][16],b[16][16];\n" \
"    int x=threadIdx.x,y=threadIdx.y,r=blockIdx.y*16+y,c=blockIdx.x*16+x;\n" \
"    float sum=0;\n" \
"    for(int base=0;base<K;base+=16) {\n" \
"        int ka=base+x,kb=base+y;\n" \
"        a[y][x]=(r<M&&ka<K)?A[ta?(size_t)ka*M+r:(size_t)r*K+ka]:0;\n" \
"        b[y][x]=(c<N&&kb<K)?B[tb?(size_t)c*K+kb:(size_t)kb*N+c]:0;\n" \
"        __syncthreads();\n" \
"        for(int k=0;k<16;++k)sum+=a[y][k]*b[k][x];\n" \
"        __syncthreads();\n" \
"    }\n" \
"    if(r<M&&c<N)C[(size_t)r*N+c]=sum;\n" \
"}\n"

#endif
