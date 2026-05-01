/*
 * bench_vlm_gemm.c - RDNA4 WMMA GEMM benchmark for Qwen3.6 vision shapes.
 *
 * Measures packed activation x packed weight GEMM kernels used by the HIP
 * vision encoder. Kernels use gfx12 WMMA F16/BF16 with FP32 accumulation.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

#include "../rocew.h"

#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../hip_runner_common.h"

#ifdef VLM_HIPBLASLT_ENABLED
extern int mm0_hipblaslt_set_algo_index(int idx);
extern int mm0_hipblaslt_get_algo_index(void);
extern int mm0_hipblaslt_init(int M, int N, int K, const void *d_bias);
extern int mm0_hipblaslt_run(void *d_y, const void *d_w, const void *d_x);
extern int mm0_hipblaslt_destroy(void);
#endif

extern int mm0_extracted_init(int M, int N, int K, const void *d_bias);
extern int mm0_extracted_run(void *d_y, const void *d_w, const void *d_x);
extern hipStream_t mm0_extracted_get_stream(void);
extern int mm0_extracted_destroy(void);

typedef struct {
    const char *name;
    int m;
    int n;
    int k;
} gemm_shape;

static const gemm_shape g_shapes[] = {
    {"qkv",      4096, 3456, 1152},
    {"attn_out", 4096, 1152, 1152},
    {"ffn_up",   4096, 4304, 1152},
    {"ffn_down", 4096, 1152, 4304},
    {"mm0",      1024, 4608, 4608},
    {"mm2",      1024, 5120, 4608},
};

static const char *kernel_src =
"typedef unsigned short half_raw;\n"
"typedef unsigned short bf16_raw;\n"
"typedef _Float16 f16x8 __attribute__((ext_vector_type(8)));\n"
"typedef unsigned short bf16x8 __attribute__((ext_vector_type(8)));\n"
"typedef float float8 __attribute__((ext_vector_type(8)));\n"
"__device__ __forceinline__ _Float16 f16_bits_to_f16(half_raw h) { _Float16 v; memcpy(&v, &h, 2); return v; }\n"
"__device__ __forceinline__ void store_acc8_f32(float *Y, const float *bias, float8 acc, int row0, int col, int ld) {\n"
"    float bv = bias ? bias[col] : 0.0f;\n"
"    Y[(size_t)(row0 + 0) * ld + col] = acc[0] + bv;\n"
"    Y[(size_t)(row0 + 1) * ld + col] = acc[1] + bv;\n"
"    Y[(size_t)(row0 + 2) * ld + col] = acc[2] + bv;\n"
"    Y[(size_t)(row0 + 3) * ld + col] = acc[3] + bv;\n"
"    Y[(size_t)(row0 + 4) * ld + col] = acc[4] + bv;\n"
"    Y[(size_t)(row0 + 5) * ld + col] = acc[5] + bv;\n"
"    Y[(size_t)(row0 + 6) * ld + col] = acc[6] + bv;\n"
"    Y[(size_t)(row0 + 7) * ld + col] = acc[7] + bv;\n"
"}\n"
"__device__ __forceinline__ void lds_barrier_signal() { __asm__ __volatile__(\"s_barrier_signal -1\" ::: \"memory\"); }\n"
"__device__ __forceinline__ void lds_barrier_wait() { __asm__ __volatile__(\"s_barrier_wait 0xffff\" ::: \"memory\"); }\n"
"__device__ __forceinline__ void lds_barrier() { lds_barrier_signal(); lds_barrier_wait(); }\n"
"\n"
"extern \"C\" {\n"
"\n"
"__global__ void gemm_wmma_f16_f32(float *Y, const half_raw *W, const half_raw *X,\n"
"                                   const float *bias, int N, int K, int M) {\n"
"    int tid = threadIdx.x;\n"
"    int wave_id = tid >> 5;\n"
"    int lane = tid & 31;\n"
"    int wM = wave_id & 1;\n"
"    int wN = wave_id >> 1;\n"
"    int half = lane >> 4;\n"
"    int idx = lane & 15;\n"
"    int k_off = half * 8;\n"
"    int cta_m0 = blockIdx.y * 128;\n"
"    int cta_n0 = blockIdx.x * 128;\n"
"    __shared__ _Float16 smA[128*32];\n"
"    __shared__ _Float16 smB[128*32];\n"
"    float8 z = {0,0,0,0,0,0,0,0};\n"
"    float8 cv00=z,cv01=z,cv10=z,cv11=z,cv20=z,cv21=z,cv30=z,cv31=z;\n"
"    for (int k = 0; k < K; k += 32) {\n"
"        for (int it = 0; it < 16; it++) {\n"
"            int e = tid * 16 + it;\n"
"            int er = e >> 5, ek = e & 31;\n"
"            int row = cta_m0 + er, kp = k + ek;\n"
"            smA[e] = (row < M && kp < K) ? f16_bits_to_f16(X[(size_t)row * K + kp]) : (_Float16)0.0f;\n"
"        }\n"
"        for (int it = 0; it < 16; it++) {\n"
"            int e = tid * 16 + it;\n"
"            int er = e >> 5, ek = e & 31;\n"
"            int col = cta_n0 + er, kp = k + ek;\n"
"            smB[e] = (col < N && kp < K) ? f16_bits_to_f16(W[(size_t)col * K + kp]) : (_Float16)0.0f;\n"
"        }\n"
"        __syncthreads();\n"
"        int a_base = wM * 64;\n"
"        int b_base = wN * 32;\n"
"        for (int kk0 = 0; kk0 < 32; kk0 += 16) {\n"
"            f16x8 a0,a1,a2,a3,b0,b1;\n"
"            for (int i = 0; i < 8; i++) {\n"
"                a0[i]=smA[(a_base+0 +idx)*32+kk0+k_off+i];\n"
"                a1[i]=smA[(a_base+16+idx)*32+kk0+k_off+i];\n"
"                a2[i]=smA[(a_base+32+idx)*32+kk0+k_off+i];\n"
"                a3[i]=smA[(a_base+48+idx)*32+kk0+k_off+i];\n"
"                b0[i]=smB[(b_base+0 +idx)*32+kk0+k_off+i];\n"
"                b1[i]=smB[(b_base+16+idx)*32+kk0+k_off+i];\n"
"            }\n"
"            cv00=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a0,b0,cv00);\n"
"            cv01=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a0,b1,cv01);\n"
"            cv10=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a1,b0,cv10);\n"
"            cv11=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a1,b1,cv11);\n"
"            cv20=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a2,b0,cv20);\n"
"            cv21=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a2,b1,cv21);\n"
"            cv30=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a3,b0,cv30);\n"
"            cv31=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a3,b1,cv31);\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    int wave_m0 = cta_m0 + wM * 64;\n"
"    int wave_n0 = cta_n0 + wN * 32;\n"
"    float8 *accs[8] = {&cv00,&cv01,&cv10,&cv11,&cv20,&cv21,&cv30,&cv31};\n"
"    int ms[8] = {0,0,16,16,32,32,48,48};\n"
"    int ns[8] = {0,16,0,16,0,16,0,16};\n"
"    for (int t = 0; t < 8; t++) {\n"
"        int col = wave_n0 + ns[t] + idx;\n"
"        if (col >= N) continue;\n"
"        float8 acc = *accs[t];\n"
"        float bv = bias ? bias[col] : 0.0f;\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int row = wave_m0 + ms[t] + half * 8 + i;\n"
"            if (row < M) Y[(size_t)row * N + col] = acc[i] + bv;\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"__global__ void gemm_wmma_bf16_f32(float *Y, const bf16_raw *W, const bf16_raw *X,\n"
"                                    const float *bias, int N, int K, int M) {\n"
"    int tid = threadIdx.x;\n"
"    int wave_id = tid >> 5;\n"
"    int lane = tid & 31;\n"
"    int wM = wave_id & 1;\n"
"    int wN = wave_id >> 1;\n"
"    int half = lane >> 4;\n"
"    int idx = lane & 15;\n"
"    int k_off = half * 8;\n"
"    int cta_m0 = blockIdx.y * 128;\n"
"    int cta_n0 = blockIdx.x * 128;\n"
"    __shared__ unsigned short smA[128*32];\n"
"    __shared__ unsigned short smB[128*32];\n"
"    float8 z = {0,0,0,0,0,0,0,0};\n"
"    float8 cv00=z,cv01=z,cv10=z,cv11=z,cv20=z,cv21=z,cv30=z,cv31=z;\n"
"    for (int k = 0; k < K; k += 32) {\n"
"        for (int it = 0; it < 16; it++) {\n"
"            int e = tid * 16 + it;\n"
"            int er = e >> 5, ek = e & 31;\n"
"            int row = cta_m0 + er, kp = k + ek;\n"
"            smA[e] = (row < M && kp < K) ? X[(size_t)row * K + kp] : 0;\n"
"        }\n"
"        for (int it = 0; it < 16; it++) {\n"
"            int e = tid * 16 + it;\n"
"            int er = e >> 5, ek = e & 31;\n"
"            int col = cta_n0 + er, kp = k + ek;\n"
"            smB[e] = (col < N && kp < K) ? W[(size_t)col * K + kp] : 0;\n"
"        }\n"
"        __syncthreads();\n"
"        int a_base = wM * 64;\n"
"        int b_base = wN * 32;\n"
"        for (int kk0 = 0; kk0 < 32; kk0 += 16) {\n"
"            bf16x8 a0,a1,a2,a3,b0,b1;\n"
"            for (int i = 0; i < 8; i++) {\n"
"                a0[i]=smA[(a_base+0 +idx)*32+kk0+k_off+i];\n"
"                a1[i]=smA[(a_base+16+idx)*32+kk0+k_off+i];\n"
"                a2[i]=smA[(a_base+32+idx)*32+kk0+k_off+i];\n"
"                a3[i]=smA[(a_base+48+idx)*32+kk0+k_off+i];\n"
"                b0[i]=smB[(b_base+0 +idx)*32+kk0+k_off+i];\n"
"                b1[i]=smB[(b_base+16+idx)*32+kk0+k_off+i];\n"
"            }\n"
"            cv00=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b0,cv00);\n"
"            cv01=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b1,cv01);\n"
"            cv10=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b0,cv10);\n"
"            cv11=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b1,cv11);\n"
"            cv20=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b0,cv20);\n"
"            cv21=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b1,cv21);\n"
"            cv30=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b0,cv30);\n"
"            cv31=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b1,cv31);\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    int wave_m0 = cta_m0 + wM * 64;\n"
"    int wave_n0 = cta_n0 + wN * 32;\n"
"    float8 *accs[8] = {&cv00,&cv01,&cv10,&cv11,&cv20,&cv21,&cv30,&cv31};\n"
"    int ms[8] = {0,0,16,16,32,32,48,48};\n"
"    int ns[8] = {0,16,0,16,0,16,0,16};\n"
"    for (int t = 0; t < 8; t++) {\n"
"        int col = wave_n0 + ns[t] + idx;\n"
"        if (col >= N) continue;\n"
"        float8 acc = *accs[t];\n"
"        float bv = bias ? bias[col] : 0.0f;\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int row = wave_m0 + ms[t] + half * 8 + i;\n"
"            if (row < M) Y[(size_t)row * N + col] = acc[i] + bv;\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"__global__ void gemm_mm0_bf16_128x128(float *Y, const bf16_raw *W, const bf16_raw *X,\n"
"                                       const float *bias) {\n"
"    int tid = threadIdx.x;\n"
"    int wave_id = tid >> 5;\n"
"    int lane = tid & 31;\n"
"    int wM = wave_id & 1;\n"
"    int wN = wave_id >> 1;\n"
"    int half = lane >> 4;\n"
"    int idx = lane & 15;\n"
"    int k_off = half * 8;\n"
"    int cta_m0 = blockIdx.y * 128;\n"
"    int cta_n0 = blockIdx.x * 128;\n"
"    __shared__ unsigned short smA[128*32];\n"
"    __shared__ unsigned short smB[128*32];\n"
"    float8 z = {0,0,0,0,0,0,0,0};\n"
"    float8 cv00=z,cv01=z,cv10=z,cv11=z,cv20=z,cv21=z,cv30=z,cv31=z;\n"
"    for (int k = 0; k < 4608; k += 32) {\n"
"        for (int it = 0; it < 16; it++) {\n"
"            int e = tid * 16 + it;\n"
"            int er = e >> 5, ek = e & 31;\n"
"            smA[e] = X[(size_t)(cta_m0 + er) * 4608 + k + ek];\n"
"            smB[e] = W[(size_t)(cta_n0 + er) * 4608 + k + ek];\n"
"        }\n"
"        __syncthreads();\n"
"        int a_base = wM * 64;\n"
"        int b_base = wN * 32;\n"
"        for (int kk0 = 0; kk0 < 32; kk0 += 16) {\n"
"            bf16x8 a0,a1,a2,a3,b0,b1;\n"
"            for (int i = 0; i < 8; i++) {\n"
"                a0[i]=smA[(a_base+0 +idx)*32+kk0+k_off+i];\n"
"                a1[i]=smA[(a_base+16+idx)*32+kk0+k_off+i];\n"
"                a2[i]=smA[(a_base+32+idx)*32+kk0+k_off+i];\n"
"                a3[i]=smA[(a_base+48+idx)*32+kk0+k_off+i];\n"
"                b0[i]=smB[(b_base+0 +idx)*32+kk0+k_off+i];\n"
"                b1[i]=smB[(b_base+16+idx)*32+kk0+k_off+i];\n"
"            }\n"
"            cv00=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b0,cv00);\n"
"            cv01=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b1,cv01);\n"
"            cv10=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b0,cv10);\n"
"            cv11=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b1,cv11);\n"
"            cv20=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b0,cv20);\n"
"            cv21=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b1,cv21);\n"
"            cv30=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b0,cv30);\n"
"            cv31=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b1,cv31);\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    int wave_m0 = cta_m0 + wM * 64;\n"
"    int wave_n0 = cta_n0 + wN * 32;\n"
"    float8 *accs[8] = {&cv00,&cv01,&cv10,&cv11,&cv20,&cv21,&cv30,&cv31};\n"
"    int ms[8] = {0,0,16,16,32,32,48,48};\n"
"    int ns[8] = {0,16,0,16,0,16,0,16};\n"
"    for (int t = 0; t < 8; t++) {\n"
"        int col = wave_n0 + ns[t] + idx;\n"
"        float8 acc = *accs[t];\n"
"        float bv = bias[col];\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int row = wave_m0 + ms[t] + half * 8 + i;\n"
"            Y[(size_t)row * 4608 + col] = acc[i] + bv;\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"__global__ void gemm_mm0_bf16_vec128(float *Y, const bf16_raw *W, const bf16_raw *X,\n"
"                                      const float *bias) {\n"
"    int tid = threadIdx.x;\n"
"    int wave_id = tid >> 5;\n"
"    int lane = tid & 31;\n"
"    int wM = wave_id & 1;\n"
"    int wN = wave_id >> 1;\n"
"    int half = lane >> 4;\n"
"    int idx = lane & 15;\n"
"    int k_off = half * 8;\n"
"    int cta_m0 = blockIdx.y * 128;\n"
"    int cta_n0 = blockIdx.x * 128;\n"
"    __shared__ bf16x8 smA8[128*4];\n"
"    __shared__ bf16x8 smB8[128*4];\n"
"    float8 z = {0,0,0,0,0,0,0,0};\n"
"    float8 cv00=z,cv01=z,cv10=z,cv11=z,cv20=z,cv21=z,cv30=z,cv31=z;\n"
"    for (int k = 0; k < 4608; k += 32) {\n"
"        for (int it = 0; it < 2; it++) {\n"
"            int e = tid * 2 + it;\n"
"            int row = e >> 2;\n"
"            int kc = (e & 3) * 8;\n"
"            smA8[e] = *((const bf16x8 *)(X + (size_t)(cta_m0 + row) * 4608 + k + kc));\n"
"            smB8[e] = *((const bf16x8 *)(W + (size_t)(cta_n0 + row) * 4608 + k + kc));\n"
"        }\n"
"        __syncthreads();\n"
"        int a_base = wM * 64;\n"
"        int b_base = wN * 32;\n"
"        for (int kk0 = 0; kk0 < 32; kk0 += 16) {\n"
"            int kslot = (kk0 + k_off) >> 3;\n"
"            bf16x8 a0 = smA8[(a_base+0 +idx)*4+kslot];\n"
"            bf16x8 a1 = smA8[(a_base+16+idx)*4+kslot];\n"
"            bf16x8 a2 = smA8[(a_base+32+idx)*4+kslot];\n"
"            bf16x8 a3 = smA8[(a_base+48+idx)*4+kslot];\n"
"            bf16x8 b0 = smB8[(b_base+0 +idx)*4+kslot];\n"
"            bf16x8 b1 = smB8[(b_base+16+idx)*4+kslot];\n"
"            cv00=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b0,cv00);\n"
"            cv01=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b1,cv01);\n"
"            cv10=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b0,cv10);\n"
"            cv11=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b1,cv11);\n"
"            cv20=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b0,cv20);\n"
"            cv21=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b1,cv21);\n"
"            cv30=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b0,cv30);\n"
"            cv31=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b1,cv31);\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    int wave_m0 = cta_m0 + wM * 64;\n"
"    int wave_n0 = cta_n0 + wN * 32;\n"
"    float8 *accs[8] = {&cv00,&cv01,&cv10,&cv11,&cv20,&cv21,&cv30,&cv31};\n"
"    int ms[8] = {0,0,16,16,32,32,48,48};\n"
"    int ns[8] = {0,16,0,16,0,16,0,16};\n"
"    for (int t = 0; t < 8; t++) {\n"
"        int col = wave_n0 + ns[t] + idx;\n"
"        float8 acc = *accs[t];\n"
"        float bv = bias[col];\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int row = wave_m0 + ms[t] + half * 8 + i;\n"
"            Y[(size_t)row * 4608 + col] = acc[i] + bv;\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"__global__ void gemm_mm0_f16_vec128(float *Y, const half_raw *W, const half_raw *X,\n"
"                                     const float *bias) {\n"
"    int tid = threadIdx.x;\n"
"    int wave_id = tid >> 5;\n"
"    int lane = tid & 31;\n"
"    int wM = wave_id & 1;\n"
"    int wN = wave_id >> 1;\n"
"    int half = lane >> 4;\n"
"    int idx = lane & 15;\n"
"    int k_off = half * 8;\n"
"    int cta_m0 = blockIdx.y * 128;\n"
"    int cta_n0 = blockIdx.x * 128;\n"
"    __shared__ f16x8 smA8[128*4];\n"
"    __shared__ f16x8 smB8[128*4];\n"
"    float8 z = {0,0,0,0,0,0,0,0};\n"
"    float8 cv00=z,cv01=z,cv10=z,cv11=z,cv20=z,cv21=z,cv30=z,cv31=z;\n"
"    for (int k = 0; k < 4608; k += 32) {\n"
"        for (int it = 0; it < 2; it++) {\n"
"            int e = tid * 2 + it;\n"
"            int row = e >> 2;\n"
"            int kc = (e & 3) * 8;\n"
"            smA8[e] = *((const f16x8 *)(X + (size_t)(cta_m0 + row) * 4608 + k + kc));\n"
"            smB8[e] = *((const f16x8 *)(W + (size_t)(cta_n0 + row) * 4608 + k + kc));\n"
"        }\n"
"        __syncthreads();\n"
"        int a_base = wM * 64;\n"
"        int b_base = wN * 32;\n"
"        for (int kk0 = 0; kk0 < 32; kk0 += 16) {\n"
"            int kslot = (kk0 + k_off) >> 3;\n"
"            f16x8 a0 = smA8[(a_base+0 +idx)*4+kslot];\n"
"            f16x8 a1 = smA8[(a_base+16+idx)*4+kslot];\n"
"            f16x8 a2 = smA8[(a_base+32+idx)*4+kslot];\n"
"            f16x8 a3 = smA8[(a_base+48+idx)*4+kslot];\n"
"            f16x8 b0 = smB8[(b_base+0 +idx)*4+kslot];\n"
"            f16x8 b1 = smB8[(b_base+16+idx)*4+kslot];\n"
"            cv00=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a0,b0,cv00);\n"
"            cv01=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a0,b1,cv01);\n"
"            cv10=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a1,b0,cv10);\n"
"            cv11=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a1,b1,cv11);\n"
"            cv20=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a2,b0,cv20);\n"
"            cv21=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a2,b1,cv21);\n"
"            cv30=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a3,b0,cv30);\n"
"            cv31=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a3,b1,cv31);\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    int wave_m0 = cta_m0 + wM * 64;\n"
"    int wave_n0 = cta_n0 + wN * 32;\n"
"    float8 *accs[8] = {&cv00,&cv01,&cv10,&cv11,&cv20,&cv21,&cv30,&cv31};\n"
"    int ms[8] = {0,0,16,16,32,32,48,48};\n"
"    int ns[8] = {0,16,0,16,0,16,0,16};\n"
"    for (int t = 0; t < 8; t++) {\n"
"        int col = wave_n0 + ns[t] + idx;\n"
"        float8 acc = *accs[t];\n"
"        float bv = bias[col];\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int row = wave_m0 + ms[t] + half * 8 + i;\n"
"            Y[(size_t)row * 4608 + col] = acc[i] + bv;\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"__global__ __launch_bounds__(128, 1) void gemm_mm0_bf16_pipe4w(float *Y, const bf16_raw *W, const bf16_raw *X,\n"
"                                      const float *bias) {\n"
"    int tid = threadIdx.x;\n"
"    int wave_id = tid >> 5;\n"
"    int lane = tid & 31;\n"
"    int wM = wave_id & 1;\n"
"    int wN = wave_id >> 1;\n"
"    int half = lane >> 4;\n"
"    int idx = lane & 15;\n"
"    int k_off = half * 8;\n"
"    int cta_m0 = blockIdx.y * 128;\n"
"    int cta_n0 = blockIdx.x * 128;\n"
"    __shared__ bf16x8 smA8[2*128*4];\n"
"    __shared__ bf16x8 smB8[2*128*4];\n"
"    int e0 = tid * 4 + 0;\n"
"    int e1 = tid * 4 + 1;\n"
"    int e2 = tid * 4 + 2;\n"
"    int e3 = tid * 4 + 3;\n"
"    int r0 = e0 >> 2, r1 = e1 >> 2, r2 = e2 >> 2, r3 = e3 >> 2;\n"
"    int c0 = (e0 & 3) * 8, c1 = (e1 & 3) * 8, c2 = (e2 & 3) * 8, c3 = (e3 & 3) * 8;\n"
"    smA8[(e0&3)*128+r0] = *((const bf16x8 *)(X + (size_t)(cta_m0 + r0) * 4608 + c0));\n"
"    smA8[(e1&3)*128+r1] = *((const bf16x8 *)(X + (size_t)(cta_m0 + r1) * 4608 + c1));\n"
"    smA8[(e2&3)*128+r2] = *((const bf16x8 *)(X + (size_t)(cta_m0 + r2) * 4608 + c2));\n"
"    smA8[(e3&3)*128+r3] = *((const bf16x8 *)(X + (size_t)(cta_m0 + r3) * 4608 + c3));\n"
"    smB8[(e0&3)*128+r0] = *((const bf16x8 *)(W + (size_t)(cta_n0 + r0) * 4608 + c0));\n"
"    smB8[(e1&3)*128+r1] = *((const bf16x8 *)(W + (size_t)(cta_n0 + r1) * 4608 + c1));\n"
"    smB8[(e2&3)*128+r2] = *((const bf16x8 *)(W + (size_t)(cta_n0 + r2) * 4608 + c2));\n"
"    smB8[(e3&3)*128+r3] = *((const bf16x8 *)(W + (size_t)(cta_n0 + r3) * 4608 + c3));\n"
"    lds_barrier();\n"
"    float8 z = {0,0,0,0,0,0,0,0};\n"
"    float8 cv00=z,cv01=z,cv02=z,cv03=z,cv10=z,cv11=z,cv12=z,cv13=z;\n"
"    float8 cv20=z,cv21=z,cv22=z,cv23=z,cv30=z,cv31=z,cv32=z,cv33=z;\n"
"    int buf = 0;\n"
"    for (int k = 0; k < 4608; k += 32) {\n"
"        bf16x8 na0,na1,na2,na3,nb0,nb1,nb2,nb3;\n"
"        int has_next = k + 32 < 4608;\n"
"        if (has_next) {\n"
"            int nk = k + 32;\n"
"            na0 = *((const bf16x8 *)(X + (size_t)(cta_m0 + r0) * 4608 + nk + c0));\n"
"            na1 = *((const bf16x8 *)(X + (size_t)(cta_m0 + r1) * 4608 + nk + c1));\n"
"            na2 = *((const bf16x8 *)(X + (size_t)(cta_m0 + r2) * 4608 + nk + c2));\n"
"            na3 = *((const bf16x8 *)(X + (size_t)(cta_m0 + r3) * 4608 + nk + c3));\n"
"            nb0 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r0) * 4608 + nk + c0));\n"
"            nb1 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r1) * 4608 + nk + c1));\n"
"            nb2 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r2) * 4608 + nk + c2));\n"
"            nb3 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r3) * 4608 + nk + c3));\n"
"        }\n"
"        int base = buf * 512;\n"
"        int a_base = wM * 64;\n"
"        int b_base = wN * 64;\n"
"        for (int kk0 = 0; kk0 < 32; kk0 += 16) {\n"
"            int kslot = (kk0 + k_off) >> 3;\n"
"            bf16x8 a0 = smA8[base+kslot*128+(a_base+0 +idx)];\n"
"            bf16x8 a1 = smA8[base+kslot*128+(a_base+16+idx)];\n"
"            bf16x8 a2 = smA8[base+kslot*128+(a_base+32+idx)];\n"
"            bf16x8 a3 = smA8[base+kslot*128+(a_base+48+idx)];\n"
"            bf16x8 b0 = smB8[base+kslot*128+(b_base+0 +idx)];\n"
"            bf16x8 b1 = smB8[base+kslot*128+(b_base+16+idx)];\n"
"            bf16x8 b2 = smB8[base+kslot*128+(b_base+32+idx)];\n"
"            bf16x8 b3 = smB8[base+kslot*128+(b_base+48+idx)];\n"
"            cv00=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b0,cv00);\n"
"            cv01=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b1,cv01);\n"
"            cv02=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b2,cv02);\n"
"            cv03=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b3,cv03);\n"
"            cv10=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b0,cv10);\n"
"            cv11=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b1,cv11);\n"
"            cv12=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b2,cv12);\n"
"            cv13=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b3,cv13);\n"
"            cv20=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b0,cv20);\n"
"            cv21=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b1,cv21);\n"
"            cv22=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b2,cv22);\n"
"            cv23=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b3,cv23);\n"
"            cv30=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b0,cv30);\n"
"            cv31=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b1,cv31);\n"
"            cv32=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b2,cv32);\n"
"            cv33=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b3,cv33);\n"
"        }\n"
"        if (has_next) {\n"
"            int nb = 1 - buf;\n"
"            int nbase = nb * 512;\n"
"            smA8[nbase+(e0&3)*128+r0] = na0; smA8[nbase+(e1&3)*128+r1] = na1; smA8[nbase+(e2&3)*128+r2] = na2; smA8[nbase+(e3&3)*128+r3] = na3;\n"
"            smB8[nbase+(e0&3)*128+r0] = nb0; smB8[nbase+(e1&3)*128+r1] = nb1; smB8[nbase+(e2&3)*128+r2] = nb2; smB8[nbase+(e3&3)*128+r3] = nb3;\n"
"            lds_barrier_signal();\n"
"            buf = nb;\n"
"            lds_barrier_wait();\n"
"        }\n"
"    }\n"
"    int wave_m0 = cta_m0 + wM * 64;\n"
"    int wave_n0 = cta_n0 + wN * 64;\n"
"    int row = wave_m0 + half * 8;\n"
"    store_acc8_f32(Y, bias, cv00, row +  0, wave_n0 +  0 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv01, row +  0, wave_n0 + 16 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv02, row +  0, wave_n0 + 32 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv03, row +  0, wave_n0 + 48 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv10, row + 16, wave_n0 +  0 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv11, row + 16, wave_n0 + 16 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv12, row + 16, wave_n0 + 32 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv13, row + 16, wave_n0 + 48 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv20, row + 32, wave_n0 +  0 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv21, row + 32, wave_n0 + 16 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv22, row + 32, wave_n0 + 32 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv23, row + 32, wave_n0 + 48 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv30, row + 48, wave_n0 +  0 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv31, row + 48, wave_n0 + 16 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv32, row + 48, wave_n0 + 32 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv33, row + 48, wave_n0 + 48 + idx, 4608);\n"
"}\n"
"\n"
"__global__ __launch_bounds__(128, 1) void gemm_mm0_bf16_pipe4w_glint(float *Y, const bf16_raw *W, const bf16_raw *X,\n"
"                                            const float *bias) {\n"
"    int tid = threadIdx.x;\n"
"    int wave_id = tid >> 5;\n"
"    int lane = tid & 31;\n"
"    int wM = wave_id & 1;\n"
"    int wN = wave_id >> 1;\n"
"    int half = lane >> 4;\n"
"    int idx = lane & 15;\n"
"    int k_off = half * 8;\n"
"    int cta_m0 = blockIdx.y * 128;\n"
"    int cta_n0 = blockIdx.x * 128;\n"
"    __shared__ bf16x8 smA8[2*128*4];\n"
"    __shared__ bf16x8 smB8[2*128*4];\n"
"    int e0 = tid * 4 + 0;\n"
"    int e1 = tid * 4 + 1;\n"
"    int e2 = tid * 4 + 2;\n"
"    int e3 = tid * 4 + 3;\n"
"    int r0 = e0 >> 2, r1 = e1 >> 2, r2 = e2 >> 2, r3 = e3 >> 2;\n"
"    int c0 = (e0 & 3) * 8, c1 = (e1 & 3) * 8, c2 = (e2 & 3) * 8, c3 = (e3 & 3) * 8;\n"
"    smA8[(e0&3)*128+r0] = *((const bf16x8 *)(X + (size_t)(cta_m0 + r0) * 4608 + c0));\n"
"    smA8[(e1&3)*128+r1] = *((const bf16x8 *)(X + (size_t)(cta_m0 + r1) * 4608 + c1));\n"
"    smA8[(e2&3)*128+r2] = *((const bf16x8 *)(X + (size_t)(cta_m0 + r2) * 4608 + c2));\n"
"    smA8[(e3&3)*128+r3] = *((const bf16x8 *)(X + (size_t)(cta_m0 + r3) * 4608 + c3));\n"
"    smB8[(e0&3)*128+r0] = *((const bf16x8 *)(W + (size_t)(cta_n0 + r0) * 4608 + c0));\n"
"    smB8[(e1&3)*128+r1] = *((const bf16x8 *)(W + (size_t)(cta_n0 + r1) * 4608 + c1));\n"
"    smB8[(e2&3)*128+r2] = *((const bf16x8 *)(W + (size_t)(cta_n0 + r2) * 4608 + c2));\n"
"    smB8[(e3&3)*128+r3] = *((const bf16x8 *)(W + (size_t)(cta_n0 + r3) * 4608 + c3));\n"
"    lds_barrier();\n"
"    float8 z = {0,0,0,0,0,0,0,0};\n"
"    float8 cv00=z,cv01=z,cv02=z,cv03=z,cv10=z,cv11=z,cv12=z,cv13=z;\n"
"    float8 cv20=z,cv21=z,cv22=z,cv23=z,cv30=z,cv31=z,cv32=z,cv33=z;\n"
"    int buf = 0;\n"
"    for (int k = 0; k < 4608; k += 32) {\n"
"        bf16x8 na0,na1,na2,na3,nb0,nb1,nb2,nb3;\n"
"        int has_next = k + 32 < 4608;\n"
"        int base = buf * 512;\n"
"        int a_base = wM * 64;\n"
"        int b_base = wN * 64;\n"
"        int kslot = k_off >> 3;\n"
"        bf16x8 a0 = smA8[base+kslot*128+(a_base+0 +idx)];\n"
"        bf16x8 a1 = smA8[base+kslot*128+(a_base+16+idx)];\n"
"        bf16x8 a2 = smA8[base+kslot*128+(a_base+32+idx)];\n"
"        bf16x8 a3 = smA8[base+kslot*128+(a_base+48+idx)];\n"
"        bf16x8 b0 = smB8[base+kslot*128+(b_base+0 +idx)];\n"
"        bf16x8 b1 = smB8[base+kslot*128+(b_base+16+idx)];\n"
"        bf16x8 b2 = smB8[base+kslot*128+(b_base+32+idx)];\n"
"        bf16x8 b3 = smB8[base+kslot*128+(b_base+48+idx)];\n"
"        int nk = k + 32;\n"
"        if (has_next) { na0 = *((const bf16x8 *)(X + (size_t)(cta_m0 + r0) * 4608 + nk + c0)); nb0 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r0) * 4608 + nk + c0)); }\n"
"        cv00=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b0,cv00);\n"
"        cv01=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b1,cv01);\n"
"        cv02=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b2,cv02);\n"
"        cv03=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b3,cv03);\n"
"        if (has_next) { na1 = *((const bf16x8 *)(X + (size_t)(cta_m0 + r1) * 4608 + nk + c1)); nb1 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r1) * 4608 + nk + c1)); }\n"
"        cv10=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b0,cv10);\n"
"        cv11=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b1,cv11);\n"
"        cv12=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b2,cv12);\n"
"        cv13=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b3,cv13);\n"
"        if (has_next) { na2 = *((const bf16x8 *)(X + (size_t)(cta_m0 + r2) * 4608 + nk + c2)); nb2 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r2) * 4608 + nk + c2)); }\n"
"        cv20=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b0,cv20);\n"
"        cv21=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b1,cv21);\n"
"        cv22=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b2,cv22);\n"
"        cv23=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b3,cv23);\n"
"        if (has_next) { na3 = *((const bf16x8 *)(X + (size_t)(cta_m0 + r3) * 4608 + nk + c3)); nb3 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r3) * 4608 + nk + c3)); }\n"
"        cv30=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b0,cv30);\n"
"        cv31=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b1,cv31);\n"
"        cv32=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b2,cv32);\n"
"        cv33=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b3,cv33);\n"
"        kslot = (16 + k_off) >> 3;\n"
"        a0 = smA8[base+kslot*128+(a_base+0 +idx)];\n"
"        a1 = smA8[base+kslot*128+(a_base+16+idx)];\n"
"        a2 = smA8[base+kslot*128+(a_base+32+idx)];\n"
"        a3 = smA8[base+kslot*128+(a_base+48+idx)];\n"
"        b0 = smB8[base+kslot*128+(b_base+0 +idx)];\n"
"        b1 = smB8[base+kslot*128+(b_base+16+idx)];\n"
"        b2 = smB8[base+kslot*128+(b_base+32+idx)];\n"
"        b3 = smB8[base+kslot*128+(b_base+48+idx)];\n"
"        cv00=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b0,cv00);\n"
"        cv01=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b1,cv01);\n"
"        cv02=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b2,cv02);\n"
"        cv03=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b3,cv03);\n"
"        cv10=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b0,cv10);\n"
"        cv11=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b1,cv11);\n"
"        cv12=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b2,cv12);\n"
"        cv13=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b3,cv13);\n"
"        cv20=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b0,cv20);\n"
"        cv21=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b1,cv21);\n"
"        cv22=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b2,cv22);\n"
"        cv23=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b3,cv23);\n"
"        cv30=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b0,cv30);\n"
"        cv31=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b1,cv31);\n"
"        cv32=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b2,cv32);\n"
"        cv33=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b3,cv33);\n"
"        if (has_next) {\n"
"            int nb = 1 - buf;\n"
"            int nbase = nb * 512;\n"
"            smA8[nbase+(e0&3)*128+r0] = na0; smA8[nbase+(e1&3)*128+r1] = na1; smA8[nbase+(e2&3)*128+r2] = na2; smA8[nbase+(e3&3)*128+r3] = na3;\n"
"            smB8[nbase+(e0&3)*128+r0] = nb0; smB8[nbase+(e1&3)*128+r1] = nb1; smB8[nbase+(e2&3)*128+r2] = nb2; smB8[nbase+(e3&3)*128+r3] = nb3;\n"
"            lds_barrier_signal();\n"
"            buf = nb;\n"
"            lds_barrier_wait();\n"
"        }\n"
"    }\n"
"    int wave_m0 = cta_m0 + wM * 64;\n"
"    int wave_n0 = cta_n0 + wN * 64;\n"
"    int row = wave_m0 + half * 8;\n"
"    store_acc8_f32(Y, bias, cv00, row +  0, wave_n0 +  0 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv01, row +  0, wave_n0 + 16 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv02, row +  0, wave_n0 + 32 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv03, row +  0, wave_n0 + 48 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv10, row + 16, wave_n0 +  0 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv11, row + 16, wave_n0 + 16 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv12, row + 16, wave_n0 + 32 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv13, row + 16, wave_n0 + 48 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv20, row + 32, wave_n0 +  0 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv21, row + 32, wave_n0 + 16 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv22, row + 32, wave_n0 + 32 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv23, row + 32, wave_n0 + 48 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv30, row + 48, wave_n0 +  0 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv31, row + 48, wave_n0 + 16 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv32, row + 48, wave_n0 + 32 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv33, row + 48, wave_n0 + 48 + idx, 4608);\n"
"}\n"
"\n"
"__global__ __launch_bounds__(128, 1) void gemm_mm0_bf16_pipe4w_midstore(float *Y, const bf16_raw *W, const bf16_raw *X,\n"
"                                               const float *bias) {\n"
"    int tid = threadIdx.x;\n"
"    int wave_id = tid >> 5;\n"
"    int lane = tid & 31;\n"
"    int wM = wave_id & 1;\n"
"    int wN = wave_id >> 1;\n"
"    int half = lane >> 4;\n"
"    int idx = lane & 15;\n"
"    int k_off = half * 8;\n"
"    int cta_m0 = blockIdx.y * 128;\n"
"    int cta_n0 = blockIdx.x * 128;\n"
"    __shared__ bf16x8 smA8[2*128*4];\n"
"    __shared__ bf16x8 smB8[2*128*4];\n"
"    int e0 = tid * 4 + 0;\n"
"    int e1 = tid * 4 + 1;\n"
"    int e2 = tid * 4 + 2;\n"
"    int e3 = tid * 4 + 3;\n"
"    int r0 = e0 >> 2, r1 = e1 >> 2, r2 = e2 >> 2, r3 = e3 >> 2;\n"
"    int c0 = (e0 & 3) * 8, c1 = (e1 & 3) * 8, c2 = (e2 & 3) * 8, c3 = (e3 & 3) * 8;\n"
"    smA8[(e0&3)*128+r0] = *((const bf16x8 *)(X + (size_t)(cta_m0 + r0) * 4608 + c0));\n"
"    smA8[(e1&3)*128+r1] = *((const bf16x8 *)(X + (size_t)(cta_m0 + r1) * 4608 + c1));\n"
"    smA8[(e2&3)*128+r2] = *((const bf16x8 *)(X + (size_t)(cta_m0 + r2) * 4608 + c2));\n"
"    smA8[(e3&3)*128+r3] = *((const bf16x8 *)(X + (size_t)(cta_m0 + r3) * 4608 + c3));\n"
"    smB8[(e0&3)*128+r0] = *((const bf16x8 *)(W + (size_t)(cta_n0 + r0) * 4608 + c0));\n"
"    smB8[(e1&3)*128+r1] = *((const bf16x8 *)(W + (size_t)(cta_n0 + r1) * 4608 + c1));\n"
"    smB8[(e2&3)*128+r2] = *((const bf16x8 *)(W + (size_t)(cta_n0 + r2) * 4608 + c2));\n"
"    smB8[(e3&3)*128+r3] = *((const bf16x8 *)(W + (size_t)(cta_n0 + r3) * 4608 + c3));\n"
"    lds_barrier();\n"
"    float8 z = {0,0,0,0,0,0,0,0};\n"
"    float8 cv00=z,cv01=z,cv02=z,cv03=z,cv10=z,cv11=z,cv12=z,cv13=z;\n"
"    float8 cv20=z,cv21=z,cv22=z,cv23=z,cv30=z,cv31=z,cv32=z,cv33=z;\n"
"    int buf = 0;\n"
"    for (int k = 0; k < 4608; k += 32) {\n"
"        bf16x8 na0,na1,na2,na3,nb0,nb1,nb2,nb3;\n"
"        int has_next = k + 32 < 4608;\n"
"        if (has_next) {\n"
"            int nk = k + 32;\n"
"            na0 = *((const bf16x8 *)(X + (size_t)(cta_m0 + r0) * 4608 + nk + c0));\n"
"            na1 = *((const bf16x8 *)(X + (size_t)(cta_m0 + r1) * 4608 + nk + c1));\n"
"            na2 = *((const bf16x8 *)(X + (size_t)(cta_m0 + r2) * 4608 + nk + c2));\n"
"            na3 = *((const bf16x8 *)(X + (size_t)(cta_m0 + r3) * 4608 + nk + c3));\n"
"            nb0 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r0) * 4608 + nk + c0));\n"
"            nb1 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r1) * 4608 + nk + c1));\n"
"            nb2 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r2) * 4608 + nk + c2));\n"
"            nb3 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r3) * 4608 + nk + c3));\n"
"        }\n"
"        int base = buf * 512;\n"
"        int a_base = wM * 64;\n"
"        int b_base = wN * 64;\n"
"        int kslot = k_off >> 3;\n"
"        bf16x8 a0 = smA8[base+kslot*128+(a_base+0 +idx)];\n"
"        bf16x8 a1 = smA8[base+kslot*128+(a_base+16+idx)];\n"
"        bf16x8 a2 = smA8[base+kslot*128+(a_base+32+idx)];\n"
"        bf16x8 a3 = smA8[base+kslot*128+(a_base+48+idx)];\n"
"        bf16x8 b0 = smB8[base+kslot*128+(b_base+0 +idx)];\n"
"        bf16x8 b1 = smB8[base+kslot*128+(b_base+16+idx)];\n"
"        bf16x8 b2 = smB8[base+kslot*128+(b_base+32+idx)];\n"
"        bf16x8 b3 = smB8[base+kslot*128+(b_base+48+idx)];\n"
"        cv00=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b0,cv00);\n"
"        cv01=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b1,cv01);\n"
"        cv02=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b2,cv02);\n"
"        cv03=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b3,cv03);\n"
"        cv10=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b0,cv10);\n"
"        cv11=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b1,cv11);\n"
"        cv12=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b2,cv12);\n"
"        cv13=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b3,cv13);\n"
"        cv20=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b0,cv20);\n"
"        cv21=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b1,cv21);\n"
"        cv22=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b2,cv22);\n"
"        cv23=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b3,cv23);\n"
"        cv30=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b0,cv30);\n"
"        cv31=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b1,cv31);\n"
"        cv32=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b2,cv32);\n"
"        cv33=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b3,cv33);\n"
"        if (has_next) {\n"
"            int nb = 1 - buf;\n"
"            int nbase = nb * 512;\n"
"            smA8[nbase+(e0&3)*128+r0] = na0; smA8[nbase+(e1&3)*128+r1] = na1; smA8[nbase+(e2&3)*128+r2] = na2; smA8[nbase+(e3&3)*128+r3] = na3;\n"
"            smB8[nbase+(e0&3)*128+r0] = nb0; smB8[nbase+(e1&3)*128+r1] = nb1; smB8[nbase+(e2&3)*128+r2] = nb2; smB8[nbase+(e3&3)*128+r3] = nb3;\n"
"            lds_barrier_signal();\n"
"        }\n"
"        kslot = (16 + k_off) >> 3;\n"
"        a0 = smA8[base+kslot*128+(a_base+0 +idx)];\n"
"        a1 = smA8[base+kslot*128+(a_base+16+idx)];\n"
"        a2 = smA8[base+kslot*128+(a_base+32+idx)];\n"
"        a3 = smA8[base+kslot*128+(a_base+48+idx)];\n"
"        b0 = smB8[base+kslot*128+(b_base+0 +idx)];\n"
"        b1 = smB8[base+kslot*128+(b_base+16+idx)];\n"
"        b2 = smB8[base+kslot*128+(b_base+32+idx)];\n"
"        b3 = smB8[base+kslot*128+(b_base+48+idx)];\n"
"        cv00=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b0,cv00);\n"
"        cv01=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b1,cv01);\n"
"        cv02=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b2,cv02);\n"
"        cv03=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b3,cv03);\n"
"        cv10=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b0,cv10);\n"
"        cv11=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b1,cv11);\n"
"        cv12=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b2,cv12);\n"
"        cv13=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b3,cv13);\n"
"        cv20=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b0,cv20);\n"
"        cv21=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b1,cv21);\n"
"        cv22=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b2,cv22);\n"
"        cv23=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b3,cv23);\n"
"        cv30=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b0,cv30);\n"
"        cv31=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b1,cv31);\n"
"        cv32=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b2,cv32);\n"
"        cv33=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b3,cv33);\n"
"        if (has_next) {\n"
"            buf = 1 - buf;\n"
"            lds_barrier_wait();\n"
"        }\n"
"    }\n"
"    int wave_m0 = cta_m0 + wM * 64;\n"
"    int wave_n0 = cta_n0 + wN * 64;\n"
"    int row = wave_m0 + half * 8;\n"
"    store_acc8_f32(Y, bias, cv00, row +  0, wave_n0 +  0 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv01, row +  0, wave_n0 + 16 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv02, row +  0, wave_n0 + 32 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv03, row +  0, wave_n0 + 48 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv10, row + 16, wave_n0 +  0 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv11, row + 16, wave_n0 + 16 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv12, row + 16, wave_n0 + 32 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv13, row + 16, wave_n0 + 48 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv20, row + 32, wave_n0 +  0 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv21, row + 32, wave_n0 + 16 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv22, row + 32, wave_n0 + 32 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv23, row + 32, wave_n0 + 48 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv30, row + 48, wave_n0 +  0 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv31, row + 48, wave_n0 + 16 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv32, row + 48, wave_n0 + 32 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv33, row + 48, wave_n0 + 48 + idx, 4608);\n"
"}\n"
"\n"
"__global__ __launch_bounds__(256, 1) void gemm_mm0_bf16_pipe8w(float *Y, const bf16_raw *W, const bf16_raw *X,\n"
"                                      const float *bias) {\n"
"    int tid = threadIdx.x;\n"
"    int wave_id = tid >> 5;\n"
"    int lane = tid & 31;\n"
"    int wM = wave_id & 1;\n"
"    int wN = wave_id >> 1;\n"
"    int half = lane >> 4;\n"
"    int idx = lane & 15;\n"
"    int k_off = half * 8;\n"
"    int cta_m0 = blockIdx.y * 128;\n"
"    int cta_n0 = blockIdx.x * 128;\n"
"    __shared__ bf16x8 smA8[2*128*4];\n"
"    __shared__ bf16x8 smB8[2*128*4];\n"
"    int e0 = tid * 2 + 0;\n"
"    int e1 = tid * 2 + 1;\n"
"    int r0 = e0 >> 2, r1 = e1 >> 2;\n"
"    int c0 = (e0 & 3) * 8, c1 = (e1 & 3) * 8;\n"
"    smA8[(e0&3)*128+r0] = *((const bf16x8 *)(X + (size_t)(cta_m0 + r0) * 4608 + c0));\n"
"    smA8[(e1&3)*128+r1] = *((const bf16x8 *)(X + (size_t)(cta_m0 + r1) * 4608 + c1));\n"
"    smB8[(e0&3)*128+r0] = *((const bf16x8 *)(W + (size_t)(cta_n0 + r0) * 4608 + c0));\n"
"    smB8[(e1&3)*128+r1] = *((const bf16x8 *)(W + (size_t)(cta_n0 + r1) * 4608 + c1));\n"
"    lds_barrier();\n"
"    float8 z = {0,0,0,0,0,0,0,0};\n"
"    float8 cv00=z,cv01=z,cv10=z,cv11=z,cv20=z,cv21=z,cv30=z,cv31=z;\n"
"    int buf = 0;\n"
"    for (int k = 0; k < 4608; k += 32) {\n"
"        bf16x8 na0,na1,nb0,nb1;\n"
"        int has_next = k + 32 < 4608;\n"
"        if (has_next) {\n"
"            int nk = k + 32;\n"
"            na0 = *((const bf16x8 *)(X + (size_t)(cta_m0 + r0) * 4608 + nk + c0));\n"
"            na1 = *((const bf16x8 *)(X + (size_t)(cta_m0 + r1) * 4608 + nk + c1));\n"
"            nb0 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r0) * 4608 + nk + c0));\n"
"            nb1 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r1) * 4608 + nk + c1));\n"
"        }\n"
"        int base = buf * 512;\n"
"        int a_base = wM * 64;\n"
"        int b_base = wN * 32;\n"
"        for (int kk0 = 0; kk0 < 32; kk0 += 16) {\n"
"            int kslot = (kk0 + k_off) >> 3;\n"
"            bf16x8 a0 = smA8[base+kslot*128+(a_base+0 +idx)];\n"
"            bf16x8 a1 = smA8[base+kslot*128+(a_base+16+idx)];\n"
"            bf16x8 a2 = smA8[base+kslot*128+(a_base+32+idx)];\n"
"            bf16x8 a3 = smA8[base+kslot*128+(a_base+48+idx)];\n"
"            bf16x8 b0 = smB8[base+kslot*128+(b_base+0 +idx)];\n"
"            bf16x8 b1 = smB8[base+kslot*128+(b_base+16+idx)];\n"
"            cv00=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b0,cv00);\n"
"            cv01=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b1,cv01);\n"
"            cv10=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b0,cv10);\n"
"            cv11=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b1,cv11);\n"
"            cv20=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b0,cv20);\n"
"            cv21=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b1,cv21);\n"
"            cv30=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b0,cv30);\n"
"            cv31=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b1,cv31);\n"
"        }\n"
"        if (has_next) {\n"
"            int nb = 1 - buf;\n"
"            int nbase = nb * 512;\n"
"            smA8[nbase+(e0&3)*128+r0] = na0; smA8[nbase+(e1&3)*128+r1] = na1;\n"
"            smB8[nbase+(e0&3)*128+r0] = nb0; smB8[nbase+(e1&3)*128+r1] = nb1;\n"
"            lds_barrier_signal();\n"
"            buf = nb;\n"
"            lds_barrier_wait();\n"
"        }\n"
"    }\n"
"    int wave_m0 = cta_m0 + wM * 64;\n"
"    int wave_n0 = cta_n0 + wN * 32;\n"
"    int row = wave_m0 + half * 8;\n"
"    store_acc8_f32(Y, bias, cv00, row +  0, wave_n0 +  0 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv01, row +  0, wave_n0 + 16 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv10, row + 16, wave_n0 +  0 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv11, row + 16, wave_n0 + 16 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv20, row + 32, wave_n0 +  0 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv21, row + 32, wave_n0 + 16 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv30, row + 48, wave_n0 +  0 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv31, row + 48, wave_n0 + 16 + idx, 4608);\n"
"}\n"
"\n"
"__global__ __launch_bounds__(128, 1) void gemm_mm0_bf16_pipe4w_k64(float *Y, const bf16_raw *W, const bf16_raw *X,\n"
"                                          const float *bias) {\n"
"    int tid = threadIdx.x;\n"
"    int wave_id = tid >> 5;\n"
"    int lane = tid & 31;\n"
"    int wM = wave_id & 1;\n"
"    int wN = wave_id >> 1;\n"
"    int half = lane >> 4;\n"
"    int idx = lane & 15;\n"
"    int k_off = half * 8;\n"
"    int cta_m0 = blockIdx.y * 128;\n"
"    int cta_n0 = blockIdx.x * 128;\n"
"    __shared__ bf16x8 smA8[2*128*8];\n"
"    __shared__ bf16x8 smB8[2*128*8];\n"
"    int e0 = tid * 8 + 0;\n"
"    int e1 = tid * 8 + 1;\n"
"    int e2 = tid * 8 + 2;\n"
"    int e3 = tid * 8 + 3;\n"
"    int e4 = tid * 8 + 4;\n"
"    int e5 = tid * 8 + 5;\n"
"    int e6 = tid * 8 + 6;\n"
"    int e7 = tid * 8 + 7;\n"
"    int r0 = e0 >> 3, r1 = e1 >> 3, r2 = e2 >> 3, r3 = e3 >> 3;\n"
"    int r4 = e4 >> 3, r5 = e5 >> 3, r6 = e6 >> 3, r7 = e7 >> 3;\n"
"    int c0 = (e0 & 7) * 8, c1 = (e1 & 7) * 8, c2 = (e2 & 7) * 8, c3 = (e3 & 7) * 8;\n"
"    int c4 = (e4 & 7) * 8, c5 = (e5 & 7) * 8, c6 = (e6 & 7) * 8, c7 = (e7 & 7) * 8;\n"
"    smA8[(e0&7)*128+r0] = *((const bf16x8 *)(X + (size_t)(cta_m0 + r0) * 4608 + c0));\n"
"    smA8[(e1&7)*128+r1] = *((const bf16x8 *)(X + (size_t)(cta_m0 + r1) * 4608 + c1));\n"
"    smA8[(e2&7)*128+r2] = *((const bf16x8 *)(X + (size_t)(cta_m0 + r2) * 4608 + c2));\n"
"    smA8[(e3&7)*128+r3] = *((const bf16x8 *)(X + (size_t)(cta_m0 + r3) * 4608 + c3));\n"
"    smA8[(e4&7)*128+r4] = *((const bf16x8 *)(X + (size_t)(cta_m0 + r4) * 4608 + c4));\n"
"    smA8[(e5&7)*128+r5] = *((const bf16x8 *)(X + (size_t)(cta_m0 + r5) * 4608 + c5));\n"
"    smA8[(e6&7)*128+r6] = *((const bf16x8 *)(X + (size_t)(cta_m0 + r6) * 4608 + c6));\n"
"    smA8[(e7&7)*128+r7] = *((const bf16x8 *)(X + (size_t)(cta_m0 + r7) * 4608 + c7));\n"
"    smB8[(e0&7)*128+r0] = *((const bf16x8 *)(W + (size_t)(cta_n0 + r0) * 4608 + c0));\n"
"    smB8[(e1&7)*128+r1] = *((const bf16x8 *)(W + (size_t)(cta_n0 + r1) * 4608 + c1));\n"
"    smB8[(e2&7)*128+r2] = *((const bf16x8 *)(W + (size_t)(cta_n0 + r2) * 4608 + c2));\n"
"    smB8[(e3&7)*128+r3] = *((const bf16x8 *)(W + (size_t)(cta_n0 + r3) * 4608 + c3));\n"
"    smB8[(e4&7)*128+r4] = *((const bf16x8 *)(W + (size_t)(cta_n0 + r4) * 4608 + c4));\n"
"    smB8[(e5&7)*128+r5] = *((const bf16x8 *)(W + (size_t)(cta_n0 + r5) * 4608 + c5));\n"
"    smB8[(e6&7)*128+r6] = *((const bf16x8 *)(W + (size_t)(cta_n0 + r6) * 4608 + c6));\n"
"    smB8[(e7&7)*128+r7] = *((const bf16x8 *)(W + (size_t)(cta_n0 + r7) * 4608 + c7));\n"
"    lds_barrier();\n"
"    float8 z = {0,0,0,0,0,0,0,0};\n"
"    float8 cv00=z,cv01=z,cv02=z,cv03=z,cv10=z,cv11=z,cv12=z,cv13=z;\n"
"    float8 cv20=z,cv21=z,cv22=z,cv23=z,cv30=z,cv31=z,cv32=z,cv33=z;\n"
"    int buf = 0;\n"
"    for (int k = 0; k < 4608; k += 64) {\n"
"        bf16x8 na0,na1,na2,na3,na4,na5,na6,na7,nb0,nb1,nb2,nb3,nb4,nb5,nb6,nb7;\n"
"        int has_next = k + 64 < 4608;\n"
"        if (has_next) {\n"
"            int nk = k + 64;\n"
"            na0 = *((const bf16x8 *)(X + (size_t)(cta_m0 + r0) * 4608 + nk + c0));\n"
"            na1 = *((const bf16x8 *)(X + (size_t)(cta_m0 + r1) * 4608 + nk + c1));\n"
"            na2 = *((const bf16x8 *)(X + (size_t)(cta_m0 + r2) * 4608 + nk + c2));\n"
"            na3 = *((const bf16x8 *)(X + (size_t)(cta_m0 + r3) * 4608 + nk + c3));\n"
"            na4 = *((const bf16x8 *)(X + (size_t)(cta_m0 + r4) * 4608 + nk + c4));\n"
"            na5 = *((const bf16x8 *)(X + (size_t)(cta_m0 + r5) * 4608 + nk + c5));\n"
"            na6 = *((const bf16x8 *)(X + (size_t)(cta_m0 + r6) * 4608 + nk + c6));\n"
"            na7 = *((const bf16x8 *)(X + (size_t)(cta_m0 + r7) * 4608 + nk + c7));\n"
"            nb0 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r0) * 4608 + nk + c0));\n"
"            nb1 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r1) * 4608 + nk + c1));\n"
"            nb2 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r2) * 4608 + nk + c2));\n"
"            nb3 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r3) * 4608 + nk + c3));\n"
"            nb4 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r4) * 4608 + nk + c4));\n"
"            nb5 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r5) * 4608 + nk + c5));\n"
"            nb6 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r6) * 4608 + nk + c6));\n"
"            nb7 = *((const bf16x8 *)(W + (size_t)(cta_n0 + r7) * 4608 + nk + c7));\n"
"        }\n"
"        int base = buf * 1024;\n"
"        int a_base = wM * 64;\n"
"        int b_base = wN * 64;\n"
"        for (int kk0 = 0; kk0 < 64; kk0 += 16) {\n"
"            int kslot = (kk0 + k_off) >> 3;\n"
"            bf16x8 a0 = smA8[base+kslot*128+(a_base+0 +idx)];\n"
"            bf16x8 a1 = smA8[base+kslot*128+(a_base+16+idx)];\n"
"            bf16x8 a2 = smA8[base+kslot*128+(a_base+32+idx)];\n"
"            bf16x8 a3 = smA8[base+kslot*128+(a_base+48+idx)];\n"
"            bf16x8 b0 = smB8[base+kslot*128+(b_base+0 +idx)];\n"
"            bf16x8 b1 = smB8[base+kslot*128+(b_base+16+idx)];\n"
"            bf16x8 b2 = smB8[base+kslot*128+(b_base+32+idx)];\n"
"            bf16x8 b3 = smB8[base+kslot*128+(b_base+48+idx)];\n"
"            cv00=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b0,cv00);\n"
"            cv01=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b1,cv01);\n"
"            cv02=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b2,cv02);\n"
"            cv03=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b3,cv03);\n"
"            cv10=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b0,cv10);\n"
"            cv11=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b1,cv11);\n"
"            cv12=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b2,cv12);\n"
"            cv13=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b3,cv13);\n"
"            cv20=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b0,cv20);\n"
"            cv21=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b1,cv21);\n"
"            cv22=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b2,cv22);\n"
"            cv23=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b3,cv23);\n"
"            cv30=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b0,cv30);\n"
"            cv31=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b1,cv31);\n"
"            cv32=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b2,cv32);\n"
"            cv33=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b3,cv33);\n"
"        }\n"
"        if (has_next) {\n"
"            int nb = 1 - buf;\n"
"            int nbase = nb * 1024;\n"
"            smA8[nbase+(e0&7)*128+r0] = na0; smA8[nbase+(e1&7)*128+r1] = na1; smA8[nbase+(e2&7)*128+r2] = na2; smA8[nbase+(e3&7)*128+r3] = na3;\n"
"            smA8[nbase+(e4&7)*128+r4] = na4; smA8[nbase+(e5&7)*128+r5] = na5; smA8[nbase+(e6&7)*128+r6] = na6; smA8[nbase+(e7&7)*128+r7] = na7;\n"
"            smB8[nbase+(e0&7)*128+r0] = nb0; smB8[nbase+(e1&7)*128+r1] = nb1; smB8[nbase+(e2&7)*128+r2] = nb2; smB8[nbase+(e3&7)*128+r3] = nb3;\n"
"            smB8[nbase+(e4&7)*128+r4] = nb4; smB8[nbase+(e5&7)*128+r5] = nb5; smB8[nbase+(e6&7)*128+r6] = nb6; smB8[nbase+(e7&7)*128+r7] = nb7;\n"
"            lds_barrier_signal();\n"
"            buf = nb;\n"
"            lds_barrier_wait();\n"
"        }\n"
"    }\n"
"    int wave_m0 = cta_m0 + wM * 64;\n"
"    int wave_n0 = cta_n0 + wN * 64;\n"
"    int row = wave_m0 + half * 8;\n"
"    store_acc8_f32(Y, bias, cv00, row +  0, wave_n0 +  0 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv01, row +  0, wave_n0 + 16 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv02, row +  0, wave_n0 + 32 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv03, row +  0, wave_n0 + 48 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv10, row + 16, wave_n0 +  0 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv11, row + 16, wave_n0 + 16 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv12, row + 16, wave_n0 + 32 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv13, row + 16, wave_n0 + 48 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv20, row + 32, wave_n0 +  0 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv21, row + 32, wave_n0 + 16 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv22, row + 32, wave_n0 + 32 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv23, row + 32, wave_n0 + 48 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv30, row + 48, wave_n0 +  0 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv31, row + 48, wave_n0 + 16 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv32, row + 48, wave_n0 + 32 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv33, row + 48, wave_n0 + 48 + idx, 4608);\n"
"}\n"
"\n"
"__global__ __launch_bounds__(128, 1) void gemm_mm0_bf16_pipe4w_n64(float *Y, const bf16_raw *W, const bf16_raw *X,\n"
"                                          const float *bias) {\n"
"    int tid = threadIdx.x;\n"
"    int wave_id = tid >> 5;\n"
"    int lane = tid & 31;\n"
"    int wM = wave_id & 1;\n"
"    int wN = wave_id >> 1;\n"
"    int half = lane >> 4;\n"
"    int idx = lane & 15;\n"
"    int k_off = half * 8;\n"
"    int cta_m0 = blockIdx.y * 128;\n"
"    int cta_n0 = blockIdx.x * 64;\n"
"    __shared__ bf16x8 smA8[2*128*4];\n"
"    __shared__ bf16x8 smB8[2*64*4];\n"
"    int e0 = tid * 6 + 0;\n"
"    int e1 = tid * 6 + 1;\n"
"    int e2 = tid * 6 + 2;\n"
"    int e3 = tid * 6 + 3;\n"
"    int e4 = tid * 6 + 4;\n"
"    int e5 = tid * 6 + 5;\n"
"    int a0 = e0, a1 = e1, a2 = e2, a3 = e3, a4 = e4, a5 = e5;\n"
"    int r0 = a0 >> 2, r1 = a1 >> 2, r2 = a2 >> 2, r3 = a3 >> 2, r4 = a4 >> 2, r5 = a5 >> 2;\n"
"    int c0 = (a0 & 3) * 8, c1 = (a1 & 3) * 8, c2 = (a2 & 3) * 8;\n"
"    int c3 = (a3 & 3) * 8, c4 = (a4 & 3) * 8, c5 = (a5 & 3) * 8;\n"
"    if (e0 < 512) smA8[(a0&3)*128+r0] = *((const bf16x8 *)(X + (size_t)(cta_m0 + r0) * 4608 + c0));\n"
"    else { int b = e0 - 512; smB8[(b&3)*64+(b>>2)] = *((const bf16x8 *)(W + (size_t)(cta_n0 + (b >> 2)) * 4608 + ((b & 3) * 8))); }\n"
"    if (e1 < 512) smA8[(a1&3)*128+r1] = *((const bf16x8 *)(X + (size_t)(cta_m0 + r1) * 4608 + c1));\n"
"    else { int b = e1 - 512; smB8[(b&3)*64+(b>>2)] = *((const bf16x8 *)(W + (size_t)(cta_n0 + (b >> 2)) * 4608 + ((b & 3) * 8))); }\n"
"    if (e2 < 512) smA8[(a2&3)*128+r2] = *((const bf16x8 *)(X + (size_t)(cta_m0 + r2) * 4608 + c2));\n"
"    else { int b = e2 - 512; smB8[(b&3)*64+(b>>2)] = *((const bf16x8 *)(W + (size_t)(cta_n0 + (b >> 2)) * 4608 + ((b & 3) * 8))); }\n"
"    if (e3 < 512) smA8[(a3&3)*128+r3] = *((const bf16x8 *)(X + (size_t)(cta_m0 + r3) * 4608 + c3));\n"
"    else { int b = e3 - 512; smB8[(b&3)*64+(b>>2)] = *((const bf16x8 *)(W + (size_t)(cta_n0 + (b >> 2)) * 4608 + ((b & 3) * 8))); }\n"
"    if (e4 < 512) smA8[(a4&3)*128+r4] = *((const bf16x8 *)(X + (size_t)(cta_m0 + r4) * 4608 + c4));\n"
"    else { int b = e4 - 512; smB8[(b&3)*64+(b>>2)] = *((const bf16x8 *)(W + (size_t)(cta_n0 + (b >> 2)) * 4608 + ((b & 3) * 8))); }\n"
"    if (e5 < 512) smA8[(a5&3)*128+r5] = *((const bf16x8 *)(X + (size_t)(cta_m0 + r5) * 4608 + c5));\n"
"    else { int b = e5 - 512; smB8[(b&3)*64+(b>>2)] = *((const bf16x8 *)(W + (size_t)(cta_n0 + (b >> 2)) * 4608 + ((b & 3) * 8))); }\n"
"    lds_barrier();\n"
"    float8 z = {0,0,0,0,0,0,0,0};\n"
"    float8 cv00=z,cv01=z,cv10=z,cv11=z,cv20=z,cv21=z,cv30=z,cv31=z;\n"
"    int buf = 0;\n"
"    for (int k = 0; k < 4608; k += 32) {\n"
"        bf16x8 n0,n1,n2,n3,n4,n5;\n"
"        int has_next = k + 32 < 4608;\n"
"        if (has_next) {\n"
"            int nk = k + 32;\n"
"            if (e0 < 512) n0 = *((const bf16x8 *)(X + (size_t)(cta_m0 + r0) * 4608 + nk + c0));\n"
"            else { int b = e0 - 512; n0 = *((const bf16x8 *)(W + (size_t)(cta_n0 + (b >> 2)) * 4608 + nk + ((b & 3) * 8))); }\n"
"            if (e1 < 512) n1 = *((const bf16x8 *)(X + (size_t)(cta_m0 + r1) * 4608 + nk + c1));\n"
"            else { int b = e1 - 512; n1 = *((const bf16x8 *)(W + (size_t)(cta_n0 + (b >> 2)) * 4608 + nk + ((b & 3) * 8))); }\n"
"            if (e2 < 512) n2 = *((const bf16x8 *)(X + (size_t)(cta_m0 + r2) * 4608 + nk + c2));\n"
"            else { int b = e2 - 512; n2 = *((const bf16x8 *)(W + (size_t)(cta_n0 + (b >> 2)) * 4608 + nk + ((b & 3) * 8))); }\n"
"            if (e3 < 512) n3 = *((const bf16x8 *)(X + (size_t)(cta_m0 + r3) * 4608 + nk + c3));\n"
"            else { int b = e3 - 512; n3 = *((const bf16x8 *)(W + (size_t)(cta_n0 + (b >> 2)) * 4608 + nk + ((b & 3) * 8))); }\n"
"            if (e4 < 512) n4 = *((const bf16x8 *)(X + (size_t)(cta_m0 + r4) * 4608 + nk + c4));\n"
"            else { int b = e4 - 512; n4 = *((const bf16x8 *)(W + (size_t)(cta_n0 + (b >> 2)) * 4608 + nk + ((b & 3) * 8))); }\n"
"            if (e5 < 512) n5 = *((const bf16x8 *)(X + (size_t)(cta_m0 + r5) * 4608 + nk + c5));\n"
"            else { int b = e5 - 512; n5 = *((const bf16x8 *)(W + (size_t)(cta_n0 + (b >> 2)) * 4608 + nk + ((b & 3) * 8))); }\n"
"        }\n"
"        int abase = buf * 512;\n"
"        int bbase = buf * 256;\n"
"        int a_base = wM * 64;\n"
"        int b_base = wN * 32;\n"
"        for (int kk0 = 0; kk0 < 32; kk0 += 16) {\n"
"            int kslot = (kk0 + k_off) >> 3;\n"
"            bf16x8 a0v = smA8[abase+kslot*128+(a_base+0 +idx)];\n"
"            bf16x8 a1v = smA8[abase+kslot*128+(a_base+16+idx)];\n"
"            bf16x8 a2v = smA8[abase+kslot*128+(a_base+32+idx)];\n"
"            bf16x8 a3v = smA8[abase+kslot*128+(a_base+48+idx)];\n"
"            bf16x8 b0v = smB8[bbase+kslot*64+(b_base+0 +idx)];\n"
"            bf16x8 b1v = smB8[bbase+kslot*64+(b_base+16+idx)];\n"
"            cv00=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0v,b0v,cv00);\n"
"            cv01=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0v,b1v,cv01);\n"
"            cv10=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1v,b0v,cv10);\n"
"            cv11=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1v,b1v,cv11);\n"
"            cv20=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2v,b0v,cv20);\n"
"            cv21=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2v,b1v,cv21);\n"
"            cv30=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3v,b0v,cv30);\n"
"            cv31=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3v,b1v,cv31);\n"
"        }\n"
"        if (has_next) {\n"
"            int nb = 1 - buf;\n"
"            int nabase = nb * 512;\n"
"            int nbbase = nb * 256;\n"
"            if (e0 < 512) smA8[nabase+(a0&3)*128+r0] = n0; else { int b = e0 - 512; smB8[nbbase+(b&3)*64+(b>>2)] = n0; }\n"
"            if (e1 < 512) smA8[nabase+(a1&3)*128+r1] = n1; else { int b = e1 - 512; smB8[nbbase+(b&3)*64+(b>>2)] = n1; }\n"
"            if (e2 < 512) smA8[nabase+(a2&3)*128+r2] = n2; else { int b = e2 - 512; smB8[nbbase+(b&3)*64+(b>>2)] = n2; }\n"
"            if (e3 < 512) smA8[nabase+(a3&3)*128+r3] = n3; else { int b = e3 - 512; smB8[nbbase+(b&3)*64+(b>>2)] = n3; }\n"
"            if (e4 < 512) smA8[nabase+(a4&3)*128+r4] = n4; else { int b = e4 - 512; smB8[nbbase+(b&3)*64+(b>>2)] = n4; }\n"
"            if (e5 < 512) smA8[nabase+(a5&3)*128+r5] = n5; else { int b = e5 - 512; smB8[nbbase+(b&3)*64+(b>>2)] = n5; }\n"
"            lds_barrier_signal();\n"
"            buf = nb;\n"
"            lds_barrier_wait();\n"
"        }\n"
"    }\n"
"    int wave_m0 = cta_m0 + wM * 64;\n"
"    int wave_n0 = cta_n0 + wN * 32;\n"
"    int row = wave_m0 + half * 8;\n"
"    store_acc8_f32(Y, bias, cv00, row +  0, wave_n0 +  0 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv01, row +  0, wave_n0 + 16 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv10, row + 16, wave_n0 +  0 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv11, row + 16, wave_n0 + 16 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv20, row + 32, wave_n0 +  0 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv21, row + 32, wave_n0 + 16 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv30, row + 48, wave_n0 +  0 + idx, 4608);\n"
"    store_acc8_f32(Y, bias, cv31, row + 48, wave_n0 + 16 + idx, 4608);\n"
"}\n"
"\n"
"__global__ void gemm_wmma_f16_64x64(float *Y, const half_raw *W, const half_raw *X,\n"
"                                     const float *bias, int N, int K, int M) {\n"
"    int tid = threadIdx.x;\n"
"    int wave_id = tid >> 5;\n"
"    int lane = tid & 31;\n"
"    int wM = wave_id >> 1;\n"
"    int wN = wave_id & 1;\n"
"    int half = lane >> 4;\n"
"    int idx = lane & 15;\n"
"    int k_off = half * 8;\n"
"    int cta_m0 = blockIdx.y * 64;\n"
"    int cta_n0 = blockIdx.x * 64;\n"
"    __shared__ _Float16 smA[64*16];\n"
"    __shared__ _Float16 smB[64*16];\n"
"    float8 z = {0,0,0,0,0,0,0,0};\n"
"    float8 cv00=z,cv01=z,cv10=z,cv11=z;\n"
"    for (int k = 0; k < K; k += 16) {\n"
"        for (int it = 0; it < 8; it++) {\n"
"            int e = tid * 8 + it;\n"
"            int er = e >> 4, ek = e & 15;\n"
"            int row = cta_m0 + er, kp = k + ek;\n"
"            smA[e] = (row < M && kp < K) ? f16_bits_to_f16(X[(size_t)row * K + kp]) : (_Float16)0.0f;\n"
"        }\n"
"        for (int it = 0; it < 8; it++) {\n"
"            int e = tid * 8 + it;\n"
"            int er = e >> 4, ek = e & 15;\n"
"            int col = cta_n0 + er, kp = k + ek;\n"
"            smB[e] = (col < N && kp < K) ? f16_bits_to_f16(W[(size_t)col * K + kp]) : (_Float16)0.0f;\n"
"        }\n"
"        __syncthreads();\n"
"        int a_base = wM * 32;\n"
"        int b_base = wN * 32;\n"
"        f16x8 a0,a1,b0,b1;\n"
"        for (int i = 0; i < 8; i++) {\n"
"            a0[i]=smA[(a_base+0 +idx)*16+k_off+i];\n"
"            a1[i]=smA[(a_base+16+idx)*16+k_off+i];\n"
"            b0[i]=smB[(b_base+0 +idx)*16+k_off+i];\n"
"            b1[i]=smB[(b_base+16+idx)*16+k_off+i];\n"
"        }\n"
"        cv00=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a0,b0,cv00);\n"
"        cv01=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a0,b1,cv01);\n"
"        cv10=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a1,b0,cv10);\n"
"        cv11=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a1,b1,cv11);\n"
"        __syncthreads();\n"
"    }\n"
"    int wave_m0 = cta_m0 + wM * 32;\n"
"    int wave_n0 = cta_n0 + wN * 32;\n"
"    float8 *accs[4] = {&cv00,&cv01,&cv10,&cv11};\n"
"    int ms[4] = {0,0,16,16};\n"
"    int ns[4] = {0,16,0,16};\n"
"    for (int t = 0; t < 4; t++) {\n"
"        int col = wave_n0 + ns[t] + idx;\n"
"        if (col >= N) continue;\n"
"        float8 acc = *accs[t];\n"
"        float bv = bias ? bias[col] : 0.0f;\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int row = wave_m0 + ms[t] + half * 8 + i;\n"
"            if (row < M) Y[(size_t)row * N + col] = acc[i] + bv;\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"__global__ void gemm_wmma_bf16_64x64(float *Y, const bf16_raw *W, const bf16_raw *X,\n"
"                                      const float *bias, int N, int K, int M) {\n"
"    int tid = threadIdx.x;\n"
"    int wave_id = tid >> 5;\n"
"    int lane = tid & 31;\n"
"    int wM = wave_id >> 1;\n"
"    int wN = wave_id & 1;\n"
"    int half = lane >> 4;\n"
"    int idx = lane & 15;\n"
"    int k_off = half * 8;\n"
"    int cta_m0 = blockIdx.y * 64;\n"
"    int cta_n0 = blockIdx.x * 64;\n"
"    __shared__ unsigned short smA[64*16];\n"
"    __shared__ unsigned short smB[64*16];\n"
"    float8 z = {0,0,0,0,0,0,0,0};\n"
"    float8 cv00=z,cv01=z,cv10=z,cv11=z;\n"
"    for (int k = 0; k < K; k += 16) {\n"
"        for (int it = 0; it < 8; it++) {\n"
"            int e = tid * 8 + it;\n"
"            int er = e >> 4, ek = e & 15;\n"
"            int row = cta_m0 + er, kp = k + ek;\n"
"            smA[e] = (row < M && kp < K) ? X[(size_t)row * K + kp] : 0;\n"
"        }\n"
"        for (int it = 0; it < 8; it++) {\n"
"            int e = tid * 8 + it;\n"
"            int er = e >> 4, ek = e & 15;\n"
"            int col = cta_n0 + er, kp = k + ek;\n"
"            smB[e] = (col < N && kp < K) ? W[(size_t)col * K + kp] : 0;\n"
"        }\n"
"        __syncthreads();\n"
"        int a_base = wM * 32;\n"
"        int b_base = wN * 32;\n"
"        bf16x8 a0,a1,b0,b1;\n"
"        for (int i = 0; i < 8; i++) {\n"
"            a0[i]=smA[(a_base+0 +idx)*16+k_off+i];\n"
"            a1[i]=smA[(a_base+16+idx)*16+k_off+i];\n"
"            b0[i]=smB[(b_base+0 +idx)*16+k_off+i];\n"
"            b1[i]=smB[(b_base+16+idx)*16+k_off+i];\n"
"        }\n"
"        cv00=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b0,cv00);\n"
"        cv01=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b1,cv01);\n"
"        cv10=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b0,cv10);\n"
"        cv11=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b1,cv11);\n"
"        __syncthreads();\n"
"    }\n"
"    int wave_m0 = cta_m0 + wM * 32;\n"
"    int wave_n0 = cta_n0 + wN * 32;\n"
"    float8 *accs[4] = {&cv00,&cv01,&cv10,&cv11};\n"
"    int ms[4] = {0,0,16,16};\n"
"    int ns[4] = {0,16,0,16};\n"
"    for (int t = 0; t < 4; t++) {\n"
"        int col = wave_n0 + ns[t] + idx;\n"
"        if (col >= N) continue;\n"
"        float8 acc = *accs[t];\n"
"        float bv = bias ? bias[col] : 0.0f;\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int row = wave_m0 + ms[t] + half * 8 + i;\n"
"            if (row < M) Y[(size_t)row * N + col] = acc[i] + bv;\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"__global__ __launch_bounds__(32) void gemm_wmma_f16_direct(float *Y, const half_raw *W, const half_raw *X,\n"
"                                                            const float *bias, int N, int K, int M) {\n"
"    int lane = threadIdx.x;\n"
"    int half = lane >> 4;\n"
"    int idx = lane & 15;\n"
"    int k_off = half * 8;\n"
"    int m0 = blockIdx.y * 64;\n"
"    int n0 = blockIdx.x * 32;\n"
"    float8 z = {0,0,0,0,0,0,0,0};\n"
"    float8 cv00=z,cv01=z,cv10=z,cv11=z,cv20=z,cv21=z,cv30=z,cv31=z;\n"
"    for (int k = 0; k < K; k += 16) {\n"
"        f16x8 a0,a1,a2,a3,b0,b1;\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int kk = k + k_off + i;\n"
"            a0[i] = (m0+0 +idx < M && kk < K) ? f16_bits_to_f16(X[(size_t)(m0+0 +idx)*K+kk]) : (_Float16)0.0f;\n"
"            a1[i] = (m0+16+idx < M && kk < K) ? f16_bits_to_f16(X[(size_t)(m0+16+idx)*K+kk]) : (_Float16)0.0f;\n"
"            a2[i] = (m0+32+idx < M && kk < K) ? f16_bits_to_f16(X[(size_t)(m0+32+idx)*K+kk]) : (_Float16)0.0f;\n"
"            a3[i] = (m0+48+idx < M && kk < K) ? f16_bits_to_f16(X[(size_t)(m0+48+idx)*K+kk]) : (_Float16)0.0f;\n"
"            b0[i] = (n0+0 +idx < N && kk < K) ? f16_bits_to_f16(W[(size_t)(n0+0 +idx)*K+kk]) : (_Float16)0.0f;\n"
"            b1[i] = (n0+16+idx < N && kk < K) ? f16_bits_to_f16(W[(size_t)(n0+16+idx)*K+kk]) : (_Float16)0.0f;\n"
"        }\n"
"        cv00=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a0,b0,cv00);\n"
"        cv01=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a0,b1,cv01);\n"
"        cv10=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a1,b0,cv10);\n"
"        cv11=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a1,b1,cv11);\n"
"        cv20=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a2,b0,cv20);\n"
"        cv21=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a2,b1,cv21);\n"
"        cv30=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a3,b0,cv30);\n"
"        cv31=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a3,b1,cv31);\n"
"    }\n"
"    float8 *accs[8] = {&cv00,&cv01,&cv10,&cv11,&cv20,&cv21,&cv30,&cv31};\n"
"    int ms[8] = {0,0,16,16,32,32,48,48};\n"
"    int ns[8] = {0,16,0,16,0,16,0,16};\n"
"    for (int t = 0; t < 8; t++) {\n"
"        int col = n0 + ns[t] + idx;\n"
"        if (col >= N) continue;\n"
"        float8 acc = *accs[t];\n"
"        float bv = bias ? bias[col] : 0.0f;\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int row = m0 + ms[t] + half * 8 + i;\n"
"            if (row < M) Y[(size_t)row * N + col] = acc[i] + bv;\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"__global__ __launch_bounds__(32) void gemm_wmma_bf16_direct(float *Y, const bf16_raw *W, const bf16_raw *X,\n"
"                                                             const float *bias, int N, int K, int M) {\n"
"    int lane = threadIdx.x;\n"
"    int half = lane >> 4;\n"
"    int idx = lane & 15;\n"
"    int k_off = half * 8;\n"
"    int m0 = blockIdx.y * 64;\n"
"    int n0 = blockIdx.x * 32;\n"
"    float8 z = {0,0,0,0,0,0,0,0};\n"
"    float8 cv00=z,cv01=z,cv10=z,cv11=z,cv20=z,cv21=z,cv30=z,cv31=z;\n"
"    for (int k = 0; k < K; k += 16) {\n"
"        bf16x8 a0,a1,a2,a3,b0,b1;\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int kk = k + k_off + i;\n"
"            a0[i] = (m0+0 +idx < M && kk < K) ? X[(size_t)(m0+0 +idx)*K+kk] : 0;\n"
"            a1[i] = (m0+16+idx < M && kk < K) ? X[(size_t)(m0+16+idx)*K+kk] : 0;\n"
"            a2[i] = (m0+32+idx < M && kk < K) ? X[(size_t)(m0+32+idx)*K+kk] : 0;\n"
"            a3[i] = (m0+48+idx < M && kk < K) ? X[(size_t)(m0+48+idx)*K+kk] : 0;\n"
"            b0[i] = (n0+0 +idx < N && kk < K) ? W[(size_t)(n0+0 +idx)*K+kk] : 0;\n"
"            b1[i] = (n0+16+idx < N && kk < K) ? W[(size_t)(n0+16+idx)*K+kk] : 0;\n"
"        }\n"
"        cv00=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b0,cv00);\n"
"        cv01=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a0,b1,cv01);\n"
"        cv10=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b0,cv10);\n"
"        cv11=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a1,b1,cv11);\n"
"        cv20=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b0,cv20);\n"
"        cv21=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a2,b1,cv21);\n"
"        cv30=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b0,cv30);\n"
"        cv31=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a3,b1,cv31);\n"
"    }\n"
"    float8 *accs[8] = {&cv00,&cv01,&cv10,&cv11,&cv20,&cv21,&cv30,&cv31};\n"
"    int ms[8] = {0,0,16,16,32,32,48,48};\n"
"    int ns[8] = {0,16,0,16,0,16,0,16};\n"
"    for (int t = 0; t < 8; t++) {\n"
"        int col = n0 + ns[t] + idx;\n"
"        if (col >= N) continue;\n"
"        float8 acc = *accs[t];\n"
"        float bv = bias ? bias[col] : 0.0f;\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int row = m0 + ms[t] + half * 8 + i;\n"
"            if (row < M) Y[(size_t)row * N + col] = acc[i] + bv;\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"}\n";

static uint16_t f32_to_bf16(float f) {
    union { float f; uint32_t u; } v;
    v.f = f;
    uint32_t lsb = (v.u >> 16) & 1u;
    return (uint16_t)((v.u + 0x7fffu + lsb) >> 16);
}

static float bf16_to_f32(uint16_t h) {
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)h << 16;
    return v.f;
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    int exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x03ffu;
    union { uint32_t u; float f; } v;
    if (exp == 0) {
        if (!mant) { v.u = sign; return v.f; }
        while ((mant & 0x0400u) == 0) { mant <<= 1; exp--; }
        exp++;
        mant &= 0x03ffu;
    } else if (exp == 31) {
        v.u = sign | 0x7f800000u | (mant << 13);
        return v.f;
    }
    v.u = sign | ((uint32_t)(exp + (127 - 15)) << 23) | (mant << 13);
    return v.f;
}

static float packed_to_f32(uint16_t h, int is_bf16) {
    return is_bf16 ? bf16_to_f32(h) : f16_to_f32(h);
}

static float elapsed_kernel_ms(hipFunction_t fn, unsigned gx, unsigned gy, unsigned bx,
                               void **args, int iters) {
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, NULL));
    for (int i = 0; i < iters; i++) {
        HIP_CHECK(hipModuleLaunchKernel(fn, gx, gy, 1, bx, 1, 1, 0, NULL, args, NULL));
    }
    HIP_CHECK(hipEventRecord(stop, NULL));
    HIP_CHECK(hipEventSynchronize(stop));
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return ms / (float)iters;
}

static float elapsed_extracted_ms(void *d_y, const void *d_w, const void *d_x, int iters) {
    hipEvent_t start, stop;
    hipStream_t s = mm0_extracted_get_stream();  // matches the launcher's stream
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, s));
    for (int i = 0; i < iters; i++) {
        if (mm0_extracted_run(d_y, d_w, d_x) != 0) {
            fprintf(stderr, "mm0_extracted_run failed at iter %d\n", i);
            exit(1);
        }
    }
    HIP_CHECK(hipEventRecord(stop, s));
    HIP_CHECK(hipEventSynchronize(stop));
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return ms / (float)iters;
}

#ifdef VLM_HIPBLASLT_ENABLED
static float elapsed_blaslt_ms(void *d_y, const void *d_w, const void *d_x, int iters) {
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start, NULL));
    for (int i = 0; i < iters; i++) {
        if (mm0_hipblaslt_run(d_y, d_w, d_x) != 0) {
            fprintf(stderr, "mm0_hipblaslt_run failed at iter %d\n", i);
            exit(1);
        }
    }
    HIP_CHECK(hipEventRecord(stop, NULL));
    HIP_CHECK(hipEventSynchronize(stop));
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return ms / (float)iters;
}
#endif

static int matches_shape(const char *want, const char *name) {
    return !want || strcmp(want, "all") == 0 || strcmp(want, name) == 0;
}

static int check_samples(const uint16_t *h_a, const uint16_t *h_w, const float *h_y,
                         int m, int n, int k, int is_bf16) {
    int rows[] = {0, 1, 17, 255, 511, 1023};
    int cols[] = {0, 1, 31, 128, 1024, 4095, 4607};
    double max_abs = 0.0, sum_sq = 0.0, ref_sq = 0.0, got_sq = 0.0, dot = 0.0;
    int count = 0;
    for (size_t ri = 0; ri < sizeof(rows) / sizeof(rows[0]); ri++) {
        if (rows[ri] >= m) continue;
        for (size_t ci = 0; ci < sizeof(cols) / sizeof(cols[0]); ci++) {
            if (cols[ci] >= n) continue;
            double ref = 0.0;
            for (int kk = 0; kk < k; kk++) {
                ref += (double)packed_to_f32(h_a[(size_t)rows[ri] * k + kk], is_bf16) *
                       (double)packed_to_f32(h_w[(size_t)cols[ci] * k + kk], is_bf16);
            }
            double got = (double)h_y[(size_t)rows[ri] * n + cols[ci]];
            double diff = got - ref;
            double abs_diff = diff < 0.0 ? -diff : diff;
            if (abs_diff > max_abs) max_abs = abs_diff;
            sum_sq += diff * diff;
            ref_sq += ref * ref;
            got_sq += got * got;
            dot += ref * got;
            count++;
        }
    }
    double rms = count ? sqrt(sum_sq / (double)count) : 0.0;
    double cosine = (ref_sq > 0.0 && got_sq > 0.0) ? dot / sqrt(ref_sq * got_sq) : 1.0;
    int pass = max_abs < 1.0e-2 && rms < 1.0e-2 && cosine > 0.999999;
    printf("check    samples=%d max_abs=%.6g rms=%.6g cosine=%.9f %s\n",
           count, max_abs, rms, cosine, pass ? "PASS" : "FAIL");
    return pass ? 0 : -1;
}

int main(int argc, char **argv) {
    const char *dtype = "f16";
    const char *shape_name = "all";
    const char *mode = "cta";
    int iters = 50;
    int device = 0;
    int check = 0;
    int use_bias = 1;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--dtype") == 0 && i + 1 < argc) dtype = argv[++i];
        else if (strcmp(argv[i], "--shape") == 0 && i + 1 < argc) shape_name = argv[++i];
        else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) mode = argv[++i];
        else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) device = atoi(argv[++i]);
        else if (strcmp(argv[i], "--check") == 0) check = 1;
        else if (strcmp(argv[i], "--no-bias") == 0) use_bias = 0;
        else {
            fprintf(stderr, "Usage: %s [--dtype f16|bf16] [--shape all|qkv|attn_out|ffn_up|ffn_down|mm0|mm2] [--mode cta|cta64|direct|mm0spec|mm0vec|mm0pipe|mm0asm|mm0pipegl|mm0pipemid|mm0pipe8|mm0pipek64|mm0pipe4n64] [--iters N] [--device N] [--check] [--no-bias]\n", argv[0]);
            return 1;
        }
    }
    if (iters < 1) iters = 1;
    int use_bf16 = strcmp(dtype, "bf16") == 0;
    if (!use_bf16 && strcmp(dtype, "f16") != 0) {
        fprintf(stderr, "dtype must be f16 or bf16\n");
        return 1;
    }
    int use_direct = strcmp(mode, "direct") == 0;
    int use_cta64 = strcmp(mode, "cta64") == 0;
    int use_mm0spec = strcmp(mode, "mm0spec") == 0;
    int use_mm0vec = strcmp(mode, "mm0vec") == 0;
    int use_mm0pipe = strcmp(mode, "mm0pipe") == 0;
    int use_mm0asm = strcmp(mode, "mm0asm") == 0;
    int use_mm0pipegl = strcmp(mode, "mm0pipegl") == 0;
    int use_mm0pipemid = strcmp(mode, "mm0pipemid") == 0;
    int use_mm0pipe8 = strcmp(mode, "mm0pipe8") == 0;
    int use_mm0pipek64 = strcmp(mode, "mm0pipek64") == 0;
    int use_mm0pipe4n64 = strcmp(mode, "mm0pipe4n64") == 0;
    int use_mm0blaslt = strcmp(mode, "mm0blaslt") == 0;
    int use_mm0extract = strcmp(mode, "mm0extract") == 0;
#ifndef VLM_HIPBLASLT_ENABLED
    if (use_mm0blaslt) {
        fprintf(stderr, "mm0blaslt mode requires VLM_HIPBLASLT_ENABLED build (use bench_vlm_gemm_blaslt)\n");
        return 1;
    }
#endif
    if (!use_direct && !use_cta64 && !use_mm0spec && !use_mm0vec && !use_mm0pipe && !use_mm0asm && !use_mm0pipegl && !use_mm0pipemid && !use_mm0pipe8 && !use_mm0pipek64 && !use_mm0pipe4n64 && !use_mm0blaslt && !use_mm0extract && strcmp(mode, "cta") != 0) {
        fprintf(stderr, "mode must be cta, cta64, direct, mm0spec, mm0vec, mm0pipe, mm0asm, mm0pipegl, mm0pipemid, mm0pipe8, mm0pipek64, mm0pipe4n64, mm0blaslt, or mm0extract\n");
        return 1;
    }
    if ((use_mm0spec || use_mm0pipe || use_mm0asm || use_mm0pipegl || use_mm0pipemid || use_mm0pipe8 || use_mm0pipek64 || use_mm0pipe4n64 || use_mm0blaslt || use_mm0extract) && !use_bf16) {
        fprintf(stderr, "%s mode currently supports bf16 only\n", mode);
        return 1;
    }
    if ((use_mm0vec || use_mm0pipe || use_mm0asm || use_mm0pipegl || use_mm0pipemid || use_mm0pipe8 || use_mm0pipek64 || use_mm0pipe4n64 || use_mm0blaslt || use_mm0extract) && strcmp(shape_name, "mm0") != 0) {
        fprintf(stderr, "%s mode is specialized for --shape mm0\n", mode);
        return 1;
    }

    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) {
        fprintf(stderr, "failed to initialize ROCm/HIP runtime\n");
        return 1;
    }
    HIP_CHECK(hipInit(0));
    HIP_CHECK(hipSetDevice(device));

    hipModule_t module = NULL;
    int compile_verbose = getenv("VLM_GEMM_DUMP") ? 3 : 0;
    if (hip_compile_kernels(&module, device, kernel_src, "vlm_gemm_bench", compile_verbose, "vlm_gemm_bench") < 0) {
        return 1;
    }
    hipFunction_t fn_f16 = NULL, fn_bf16 = NULL, fn_f16_64 = NULL, fn_bf16_64 = NULL, fn_f16_direct = NULL, fn_bf16_direct = NULL, fn_mm0_bf16 = NULL, fn_mm0_bf16_vec = NULL, fn_mm0_f16_vec = NULL, fn_mm0_bf16_pipe = NULL, fn_mm0_bf16_pipegl = NULL, fn_mm0_bf16_pipemid = NULL, fn_mm0_bf16_pipe8 = NULL, fn_mm0_bf16_pipek64 = NULL, fn_mm0_bf16_pipe4n64 = NULL;
    HIP_CHECK(hipModuleGetFunction(&fn_f16, module, "gemm_wmma_f16_f32"));
    HIP_CHECK(hipModuleGetFunction(&fn_bf16, module, "gemm_wmma_bf16_f32"));
    HIP_CHECK(hipModuleGetFunction(&fn_f16_64, module, "gemm_wmma_f16_64x64"));
    HIP_CHECK(hipModuleGetFunction(&fn_bf16_64, module, "gemm_wmma_bf16_64x64"));
    HIP_CHECK(hipModuleGetFunction(&fn_f16_direct, module, "gemm_wmma_f16_direct"));
    HIP_CHECK(hipModuleGetFunction(&fn_bf16_direct, module, "gemm_wmma_bf16_direct"));
    HIP_CHECK(hipModuleGetFunction(&fn_mm0_bf16, module, "gemm_mm0_bf16_128x128"));
    HIP_CHECK(hipModuleGetFunction(&fn_mm0_bf16_vec, module, "gemm_mm0_bf16_vec128"));
    HIP_CHECK(hipModuleGetFunction(&fn_mm0_f16_vec, module, "gemm_mm0_f16_vec128"));
    HIP_CHECK(hipModuleGetFunction(&fn_mm0_bf16_pipe, module, "gemm_mm0_bf16_pipe4w"));
    HIP_CHECK(hipModuleGetFunction(&fn_mm0_bf16_pipegl, module, "gemm_mm0_bf16_pipe4w_glint"));
    HIP_CHECK(hipModuleGetFunction(&fn_mm0_bf16_pipemid, module, "gemm_mm0_bf16_pipe4w_midstore"));
    HIP_CHECK(hipModuleGetFunction(&fn_mm0_bf16_pipe8, module, "gemm_mm0_bf16_pipe8w"));
    HIP_CHECK(hipModuleGetFunction(&fn_mm0_bf16_pipek64, module, "gemm_mm0_bf16_pipe4w_k64"));
    HIP_CHECK(hipModuleGetFunction(&fn_mm0_bf16_pipe4n64, module, "gemm_mm0_bf16_pipe4w_n64"));
    hipFunction_t fn = use_bf16 ? (use_direct ? fn_bf16_direct : (use_cta64 ? fn_bf16_64 : fn_bf16))
                                : (use_direct ? fn_f16_direct : (use_cta64 ? fn_f16_64 : fn_f16));
    hipModule_t asm_module = NULL;
    if (use_mm0spec) fn = fn_mm0_bf16;
    if (use_mm0vec) fn = use_bf16 ? fn_mm0_bf16_vec : fn_mm0_f16_vec;
    if (use_mm0pipe) fn = fn_mm0_bf16_pipe;
    if (use_mm0asm) {
        const char *asm_co = getenv("VLM_GEMM_ASM_CO");
        if (!asm_co || !asm_co[0]) {
            fprintf(stderr, "mm0asm requires VLM_GEMM_ASM_CO=/path/to/code-object.co\n");
            return 1;
        }
        HIP_CHECK(hipModuleLoad(&asm_module, asm_co));
        HIP_CHECK(hipModuleGetFunction(&fn, asm_module, "gemm_mm0_bf16_asm"));
    }
    if (use_mm0pipegl) fn = fn_mm0_bf16_pipegl;
    if (use_mm0pipemid) fn = fn_mm0_bf16_pipemid;
    if (use_mm0pipe8) fn = fn_mm0_bf16_pipe8;
    if (use_mm0pipek64) fn = fn_mm0_bf16_pipek64;
    if (use_mm0pipe4n64) fn = fn_mm0_bf16_pipe4n64;

    printf("# RDNA4 VLM GEMM benchmark dtype=%s mode=%s iters=%d bias=%s peak=195.0 TFLOP/s target80=156.0 TFLOP/s\n", dtype, mode, iters, use_bias ? "on" : "off");
    for (size_t si = 0; si < sizeof(g_shapes) / sizeof(g_shapes[0]); si++) {
        const gemm_shape *s = &g_shapes[si];
        if (!matches_shape(shape_name, s->name)) continue;
        size_t a_elems = (size_t)s->m * s->k;
        size_t w_elems = (size_t)s->n * s->k;
        size_t y_elems = (size_t)s->m * s->n;
        uint16_t *h_a = (uint16_t *)malloc(a_elems * sizeof(uint16_t));
        uint16_t *h_w = (uint16_t *)malloc(w_elems * sizeof(uint16_t));
        if (!h_a || !h_w) {
            fprintf(stderr, "host allocation failed for %s\n", s->name);
            return 1;
        }
        for (size_t i = 0; i < a_elems; i++) {
            float v = 0.5f + (float)(i & 15) * 0.001f;
            h_a[i] = use_bf16 ? f32_to_bf16(v) : hip_f32_to_f16(v);
        }
        for (size_t i = 0; i < w_elems; i++) {
            float v = 0.25f + (float)(i & 7) * 0.002f;
            h_w[i] = use_bf16 ? f32_to_bf16(v) : hip_f32_to_f16(v);
        }

        void *d_a = NULL, *d_w = NULL, *d_y = NULL, *d_bias = NULL;
        HIP_CHECK(hipMalloc(&d_a, a_elems * sizeof(uint16_t)));
        HIP_CHECK(hipMalloc(&d_w, w_elems * sizeof(uint16_t)));
        HIP_CHECK(hipMalloc(&d_y, y_elems * sizeof(float)));
        if (use_bias) HIP_CHECK(hipMalloc(&d_bias, (size_t)s->n * sizeof(float)));
        HIP_CHECK(hipMemcpy(d_a, h_a, a_elems * sizeof(uint16_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_w, h_w, w_elems * sizeof(uint16_t), hipMemcpyHostToDevice));
        if (use_bias) HIP_CHECK(hipMemset(d_bias, 0, (size_t)s->n * sizeof(float)));

        int m = s->m, n = s->n, k = s->k;
        void *args[] = { &d_y, &d_w, &d_a, &d_bias, &n, &k, &m };
        void *args_mm0[] = { &d_y, &d_w, &d_a, &d_bias };
        void **launch_args = (use_mm0spec || use_mm0vec || use_mm0pipe || use_mm0asm || use_mm0pipegl || use_mm0pipemid || use_mm0pipe8 || use_mm0pipek64 || use_mm0pipe4n64) ? args_mm0 : args;
        unsigned gx = use_mm0pipe4n64 ? (unsigned)(4608 / 64)
                    : (use_mm0spec || use_mm0vec || use_mm0pipe || use_mm0asm || use_mm0pipegl || use_mm0pipemid || use_mm0pipe8 || use_mm0pipek64) ? (unsigned)(4608 / 128)
                    : use_direct ? (unsigned)((n + 31) / 32)
                                 : (use_cta64 ? (unsigned)((n + 63) / 64) : (unsigned)((n + 127) / 128));
        unsigned gy = (use_mm0spec || use_mm0vec || use_mm0pipe || use_mm0asm || use_mm0pipegl || use_mm0pipemid || use_mm0pipe8 || use_mm0pipek64 || use_mm0pipe4n64) ? (unsigned)(1024 / 128)
                    : use_direct ? (unsigned)((m + 63) / 64)
                                 : (use_cta64 ? (unsigned)((m + 63) / 64) : (unsigned)((m + 127) / 128));
        unsigned bx = (use_mm0pipe || use_mm0asm || use_mm0pipegl || use_mm0pipemid || use_mm0pipek64 || use_mm0pipe4n64) ? 128u : ((use_mm0spec || use_mm0vec || use_mm0pipe8) ? 256u : (use_direct ? 32u : (use_cta64 ? 128u : 256u)));
        float ms = 0.0f;
#ifdef VLM_HIPBLASLT_ENABLED
        if (use_mm0blaslt) {
            const char *algo_env = getenv("VLM_GEMM_BLASLT_ALGO");
            if (algo_env && algo_env[0]) {
                mm0_hipblaslt_set_algo_index(atoi(algo_env));
            }
            if (mm0_hipblaslt_init(m, n, k, d_bias) != 0) {
                fprintf(stderr, "mm0_hipblaslt_init failed\n");
                return 1;
            }
            if (mm0_hipblaslt_run(d_y, d_w, d_a) != 0) return 1;
            HIP_CHECK(hipDeviceSynchronize());
            ms = elapsed_blaslt_ms(d_y, d_w, d_a, iters);
        } else
#endif
        if (use_mm0extract) {
            if (mm0_extracted_init(m, n, k, d_bias) != 0) {
                fprintf(stderr, "mm0_extracted_init failed\n");
                return 1;
            }
            // Warm the GPU clock state. Empirical: 64 iters lifts perf
            // from ~140 to ~155 TFLOP/s (avx clock ramp). MM0_EXTRACTED_WARMUP
            // override (default 64). Optional cooldown after — useful
            // for matching hipblaslt's slow-init signature.
            const char* wu = getenv("MM0_EXTRACTED_WARMUP");
            int n_warmup = (wu && wu[0]) ? atoi(wu) : 32;
            for (int wi = 0; wi < n_warmup; ++wi) {
                if (mm0_extracted_run(d_y, d_w, d_a) != 0) return 1;
            }
            HIP_CHECK(hipDeviceSynchronize());
            const char* cd = getenv("MM0_EXTRACTED_COOLDOWN_MS");
            if (cd && cd[0]) {
                int ms_cd = atoi(cd);
                if (ms_cd > 0) usleep(ms_cd * 1000);
            }
            ms = elapsed_extracted_ms(d_y, d_w, d_a, iters);
        } else {
            HIP_CHECK(hipModuleLaunchKernel(fn, gx, gy, 1, bx, 1, 1, 0, NULL, launch_args, NULL));
            HIP_CHECK(hipDeviceSynchronize());
            ms = elapsed_kernel_ms(fn, gx, gy, bx, launch_args, iters);
        }
        double flops = 2.0 * (double)m * (double)n * (double)k;
        double tflops = flops / ((double)ms * 1.0e9);
        double pct = tflops / 195.0 * 100.0;
        printf("%-8s M=%5d N=%5d K=%5d ms=%8.4f TFLOP/s=%8.3f peak=%5.1f%% %s\n",
               s->name, m, n, k, ms, tflops, pct, tflops >= 156.0 ? "PASS80" : "FAIL80");

        if (check) {
            float *h_y = (float *)malloc(y_elems * sizeof(float));
            if (!h_y) {
                fprintf(stderr, "host output allocation failed for %s\n", s->name);
                free(h_a);
                free(h_w);
                return 1;
            }
            HIP_CHECK(hipMemcpy(h_y, d_y, y_elems * sizeof(float), hipMemcpyDeviceToHost));
            if (check_samples(h_a, h_w, h_y, m, n, k, use_bf16) != 0) {
                free(h_y);
                free(h_a);
                free(h_w);
                return 1;
            }
            free(h_y);
        }

        free(h_a);
        free(h_w);
        HIP_CHECK(hipFree(d_a));
        HIP_CHECK(hipFree(d_w));
        HIP_CHECK(hipFree(d_y));
        if (d_bias) HIP_CHECK(hipFree(d_bias));
#ifdef VLM_HIPBLASLT_ENABLED
        if (use_mm0blaslt) mm0_hipblaslt_destroy();
#endif
        if (use_mm0extract) mm0_extracted_destroy();
    }

    return 0;
}
