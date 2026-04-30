/*
 * hip_vision_encoder.c - HIP/ROCm vision encoder for Qwen3-VL mmproj
 *
 * Compiles with plain gcc (no hipcc). Uses rocew for dynamic HIP/HIPRTC loading.
 * Supports F32 (verification), F16, and BF16 weight modes.
 * Single-stream sequential kernel launches.
 * Targets RDNA4 (gfx1200/gfx1201).
 */

#include "../../common/ggml_dequant.h"

#include "hip_vision_encoder.h"
#include "../rocew.h"
#include "../hip_kernels_common.h"
#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../hip_runner_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dlfcn.h>

extern int  mm0_extracted_init(int M, int N, int K, const void *d_bias);
extern int  mm0_extracted_run(void *d_y, const void *d_w, const void *d_x);
extern int  mm0_extracted_destroy(void);

/* Aliases for shared macros */
#define CHECK_HIP HIP_CHECK
#define CHECK_HIP_NULL HIP_CHECK_NULL

enum {
    VLM_PREC_F32  = 0,
    VLM_PREC_F16  = 1,
    VLM_PREC_BF16 = 2,
};

static const char *vlm_prec_name(int mode) {
    return mode == VLM_PREC_F16 ? "F16" :
           mode == VLM_PREC_BF16 ? "BF16" : "F32";
}

enum {
    VLM_ROCBLAS_STATUS_SUCCESS = 0,
    VLM_ROCBLAS_OP_NONE = 111,
    VLM_ROCBLAS_OP_TRANSPOSE = 112,
    VLM_ROCBLAS_DATATYPE_F16 = 150,
    VLM_ROCBLAS_DATATYPE_F32 = 151,
    VLM_ROCBLAS_DATATYPE_BF16 = 168,
    VLM_ROCBLAS_POINTER_MODE_HOST = 0,
    VLM_ROCBLAS_GEMM_ALGO_STANDARD = 0,
};

typedef void *vlm_rocblas_handle;
typedef int (*vlm_rocblas_create_handle_fn)(vlm_rocblas_handle *handle);
typedef int (*vlm_rocblas_destroy_handle_fn)(vlm_rocblas_handle handle);
typedef int (*vlm_rocblas_set_stream_fn)(vlm_rocblas_handle handle, hipStream_t stream);
typedef int (*vlm_rocblas_set_pointer_mode_fn)(vlm_rocblas_handle handle, int pointer_mode);
typedef int (*vlm_rocblas_gemm_ex_fn)(vlm_rocblas_handle handle,
                                      int transA, int transB,
                                      int m, int n, int k,
                                      const void *alpha,
                                      const void *a, int a_type, int lda,
                                      const void *b, int b_type, int ldb,
                                      const void *beta,
                                      const void *c, int c_type, int ldc,
                                      void *d, int d_type, int ldd,
                                      int compute_type, int algo,
                                      int32_t solution_index, uint32_t flags);

enum {
    VLM_HIPBLASLT_STATUS_SUCCESS = 0,
    VLM_HIPBLASLT_OP_NONE = 111,
    VLM_HIPBLASLT_OP_TRANSPOSE = 112,
    VLM_HIPBLASLT_COMPUTE_32F = 2,
    VLM_HIPBLASLT_DATATYPE_F32 = 0,
    VLM_HIPBLASLT_DATATYPE_F16 = 2,
    VLM_HIPBLASLT_DATATYPE_BF16 = 14,
    VLM_HIPBLASLT_EPILOGUE_BIAS = 4,
    VLM_HIPBLASLT_EPILOGUE_GELU_BIAS = 36,
    VLM_HIPBLASLT_MATMUL_DESC_TRANSA = 0,
    VLM_HIPBLASLT_MATMUL_DESC_TRANSB = 1,
    VLM_HIPBLASLT_MATMUL_DESC_EPILOGUE = 2,
    VLM_HIPBLASLT_MATMUL_DESC_BIAS_POINTER = 3,
    VLM_HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE = 4,
    VLM_HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1,
};

typedef void *vlm_hipblaslt_handle;
typedef void *vlm_hipblaslt_matmul_desc;
typedef void *vlm_hipblaslt_matrix_layout;
typedef void *vlm_hipblaslt_matmul_pref;

typedef struct {
    uint8_t data[16];
    size_t max_workspace_bytes;
} vlm_hipblaslt_matmul_algo;

typedef struct {
    vlm_hipblaslt_matmul_algo algo;
    size_t workspace_size;
    int state;
    float waves_count;
    int reserved[4];
} vlm_hipblaslt_heuristic_result;

typedef int (*vlm_hipblaslt_create_fn)(vlm_hipblaslt_handle *handle);
typedef int (*vlm_hipblaslt_destroy_fn)(vlm_hipblaslt_handle handle);
typedef int (*vlm_hipblaslt_matmul_desc_create_fn)(vlm_hipblaslt_matmul_desc *desc,
                                                   int compute_type, int scale_type);
typedef int (*vlm_hipblaslt_matmul_desc_destroy_fn)(vlm_hipblaslt_matmul_desc desc);
typedef int (*vlm_hipblaslt_matmul_desc_set_attr_fn)(vlm_hipblaslt_matmul_desc desc,
                                                     int attr, const void *buf,
                                                     size_t size);
typedef int (*vlm_hipblaslt_matrix_layout_create_fn)(vlm_hipblaslt_matrix_layout *layout,
                                                     int type, uint64_t rows,
                                                     uint64_t cols, int64_t ld);
typedef int (*vlm_hipblaslt_matrix_layout_destroy_fn)(vlm_hipblaslt_matrix_layout layout);
typedef int (*vlm_hipblaslt_matmul_pref_create_fn)(vlm_hipblaslt_matmul_pref *pref);
typedef int (*vlm_hipblaslt_matmul_pref_destroy_fn)(vlm_hipblaslt_matmul_pref pref);
typedef int (*vlm_hipblaslt_matmul_pref_set_attr_fn)(vlm_hipblaslt_matmul_pref pref,
                                                     int attr, const void *buf,
                                                     size_t size);
typedef int (*vlm_hipblaslt_algo_get_heuristic_fn)(vlm_hipblaslt_handle handle,
                                                   vlm_hipblaslt_matmul_desc desc,
                                                   vlm_hipblaslt_matrix_layout a,
                                                   vlm_hipblaslt_matrix_layout b,
                                                   vlm_hipblaslt_matrix_layout c,
                                                   vlm_hipblaslt_matrix_layout d,
                                                   vlm_hipblaslt_matmul_pref pref,
                                                   int requested_count,
                                                   vlm_hipblaslt_heuristic_result *results,
                                                   int *returned_count);
typedef int (*vlm_hipblaslt_matmul_fn)(vlm_hipblaslt_handle handle,
                                       vlm_hipblaslt_matmul_desc desc,
                                       const void *alpha,
                                       const void *a,
                                       vlm_hipblaslt_matrix_layout a_desc,
                                       const void *b,
                                       vlm_hipblaslt_matrix_layout b_desc,
                                       const void *beta,
                                       const void *c,
                                       vlm_hipblaslt_matrix_layout c_desc,
                                       void *d,
                                       vlm_hipblaslt_matrix_layout d_desc,
                                       const vlm_hipblaslt_matmul_algo *algo,
                                       void *workspace,
                                       size_t workspace_size,
                                       hipStream_t stream);

typedef struct {
    int valid;
    int dtype;
    int m;
    int n;
    int k;
    int epilogue;
    int algo_index;
    size_t workspace_size;
    vlm_hipblaslt_matmul_desc matmul;
    vlm_hipblaslt_matrix_layout a;
    vlm_hipblaslt_matrix_layout b;
    vlm_hipblaslt_matrix_layout c;
    vlm_hipblaslt_matrix_layout d;
    vlm_hipblaslt_matmul_pref pref;
    vlm_hipblaslt_matmul_algo algo;
} vlm_hipblaslt_plan;

/* ======================================================================== */
/* Vision-specific HIP kernels (compiled at runtime via HIPRTC)             */
/* Shared kernels (layernorm, GEMM, gelu, add, etc.) are in                 */
/* hip_kernels_common.h. This string is concatenated after them.            */
/* ======================================================================== */

static const char *hip_vlm_specific_kernels =
"\n"
"/* ---- gemm_f32_f32: Naive tiled F32 GEMM for verification ---- */\n"
"/* Y[tok][i] = sum_j(W[i][j] * X[tok][j]) + bias[i] */\n"
"/* Grid: (ceil(n_out/TILE), ceil(n_tok/TILE)), Block: (TILE, TILE) */\n"
"#define TILE_F32 16\n"
"__global__ void gemm_f32_f32(float *Y, const float *W, const float *X,\n"
"                              const float *bias,\n"
"                              int n_out, int n_in, int n_tok) {\n"
"    __shared__ float sW[TILE_F32][TILE_F32];\n"
"    __shared__ float sX[TILE_F32][TILE_F32];\n"
"    int bx = blockIdx.x * TILE_F32;\n"
"    int by = blockIdx.y * TILE_F32;\n"
"    int tx = threadIdx.x;\n"
"    int ty = threadIdx.y;\n"
"    int row = by + ty;  /* token index */\n"
"    int col = bx + tx;  /* output dim index */\n"
"    float sum = 0.0f;\n"
"    for (int t = 0; t < n_in; t += TILE_F32) {\n"
"        /* Load W tile: W[col][t+ty] — but col is the output row of W */\n"
"        if (col < n_out && (t + ty) < n_in)\n"
"            sW[tx][ty] = W[(size_t)col * n_in + t + ty];\n"
"        else\n"
"            sW[tx][ty] = 0.0f;\n"
"        /* Load X tile: X[row][t+tx] */\n"
"        if (row < n_tok && (t + tx) < n_in)\n"
"            sX[ty][tx] = X[(size_t)row * n_in + t + tx];\n"
"        else\n"
"            sX[ty][tx] = 0.0f;\n"
"        __syncthreads();\n"
"        for (int k = 0; k < TILE_F32; k++)\n"
"            sum += sW[tx][k] * sX[ty][k];\n"
"        __syncthreads();\n"
"    }\n"
"    if (row < n_tok && col < n_out) {\n"
"        float b = (bias) ? bias[col] : 0.0f;\n"
"        Y[(size_t)row * n_out + col] = sum + b;\n"
"    }\n"
"}\n"
"\n"
"/* ---- gemm_opt_f16_f32: Optimized 128x128 tiled F16 GEMM ---- */\n"
"/* Y[M,N] = X[M,K] * W^T[N,K] + bias[N].  Grid: (ceil(N/128), ceil(M/128)), Block: 256 */\n"
"#define G16_BM 128\n"
"#define G16_BN 128\n"
"#define G16_BK 16\n"
"#define G16_TM 8\n"
"#define G16_TN 8\n"
"__global__ void gemm_opt_f16_f32(float *Y, const half_raw *W, const float *X,\n"
"                                  const float *bias, int N, int K, int M) {\n"
"    __shared__ float smA[G16_BK][G16_BM];\n"
"    __shared__ float smB[G16_BK][G16_BN];\n"
"    int tid = threadIdx.x;\n"
"    int bm = blockIdx.y * G16_BM, bn = blockIdx.x * G16_BN;\n"
"    int tr = tid / 16, tc = tid % 16;\n"
"    float acc[G16_TM][G16_TN];\n"
"    for (int i = 0; i < G16_TM; i++)\n"
"        for (int j = 0; j < G16_TN; j++) acc[i][j] = 0.0f;\n"
"    for (int k = 0; k < K; k += G16_BK) {\n"
"        for (int _i = tid; _i < G16_BM*G16_BK; _i += 256) {\n"
"            int _r = _i % G16_BM, _c = _i / G16_BM;\n"
"            int gr = bm+_r, gk = k+_c;\n"
"            smA[_c][_r] = (gr<M&&gk<K) ? X[(size_t)gr*K+gk] : 0.0f;\n"
"        }\n"
"        for (int _i = tid; _i < G16_BN*G16_BK; _i += 256) {\n"
"            int _r = _i % G16_BN, _c = _i / G16_BN;\n"
"            int gn = bn+_r, gk = k+_c;\n"
"            smB[_c][_r] = (gn<N&&gk<K) ? half_to_float(W[(size_t)gn*K+gk]) : 0.0f;\n"
"        }\n"
"        __syncthreads();\n"
"        for (int kk = 0; kk < G16_BK; kk++) {\n"
"            float af[G16_TM], bf[G16_TN];\n"
"            for (int i=0;i<G16_TM;i++) af[i] = smA[kk][tr*G16_TM+i];\n"
"            for (int j=0;j<G16_TN;j++) bf[j] = smB[kk][tc*G16_TN+j];\n"
"            for (int i=0;i<G16_TM;i++)\n"
"                for (int j=0;j<G16_TN;j++) acc[i][j] += af[i]*bf[j];\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    for (int i=0;i<G16_TM;i++) {\n"
"        int gr = bm+tr*G16_TM+i; if(gr>=M) continue;\n"
"        for (int j=0;j<G16_TN;j++) {\n"
"            int gn = bn+tc*G16_TN+j;\n"
"            if(gn<N) Y[(size_t)gr*N+gn] = acc[i][j]+(bias?bias[gn]:0.0f);\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"/* ---- gemm_opt_f16_f32_gelu: 128x128 F16 GEMM with fused GELU ---- */\n"
"__global__ void gemm_opt_f16_f32_gelu(float *Y, const half_raw *W, const float *X,\n"
"                                       const float *bias, int N, int K, int M) {\n"
"    __shared__ float smA[G16_BK][G16_BM];\n"
"    __shared__ float smB[G16_BK][G16_BN];\n"
"    int tid = threadIdx.x;\n"
"    int bm = blockIdx.y * G16_BM, bn = blockIdx.x * G16_BN;\n"
"    int tr = tid / 16, tc = tid % 16;\n"
"    float acc[G16_TM][G16_TN];\n"
"    for (int i = 0; i < G16_TM; i++)\n"
"        for (int j = 0; j < G16_TN; j++) acc[i][j] = 0.0f;\n"
"    for (int k = 0; k < K; k += G16_BK) {\n"
"        for (int _i = tid; _i < G16_BM*G16_BK; _i += 256) {\n"
"            int _r = _i % G16_BM, _c = _i / G16_BM;\n"
"            int gr = bm+_r, gk = k+_c;\n"
"            smA[_c][_r] = (gr<M&&gk<K) ? X[(size_t)gr*K+gk] : 0.0f;\n"
"        }\n"
"        for (int _i = tid; _i < G16_BN*G16_BK; _i += 256) {\n"
"            int _r = _i % G16_BN, _c = _i / G16_BN;\n"
"            int gn = bn+_r, gk = k+_c;\n"
"            smB[_c][_r] = (gn<N&&gk<K) ? half_to_float(W[(size_t)gn*K+gk]) : 0.0f;\n"
"        }\n"
"        __syncthreads();\n"
"        for (int kk = 0; kk < G16_BK; kk++) {\n"
"            float af[G16_TM], bf[G16_TN];\n"
"            for (int i=0;i<G16_TM;i++) af[i] = smA[kk][tr*G16_TM+i];\n"
"            for (int j=0;j<G16_TN;j++) bf[j] = smB[kk][tc*G16_TN+j];\n"
"            for (int i=0;i<G16_TM;i++)\n"
"                for (int j=0;j<G16_TN;j++) acc[i][j] += af[i]*bf[j];\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    for (int i=0;i<G16_TM;i++) {\n"
"        int gr = bm+tr*G16_TM+i; if(gr>=M) continue;\n"
"        for (int j=0;j<G16_TN;j++) {\n"
"            int gn = bn+tc*G16_TN+j;\n"
"            if(gn<N) {\n"
"                float v = acc[i][j]+(bias?bias[gn]:0.0f);\n"
"                v = 0.5f*v*(1.0f+tanhf(0.7978845608f*(v+0.044715f*v*v*v)));\n"
"                Y[(size_t)gr*N+gn] = v;\n"
"            }\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"/* ---- gemm_opt_f16_f32_res: 128x128 F16 GEMM with fused residual add ---- */\n"
"__global__ void gemm_opt_f16_f32_res(float *Y, const half_raw *W, const float *X,\n"
"                                      const float *bias, const float *residual,\n"
"                                      int N, int K, int M) {\n"
"    __shared__ float smA[G16_BK][G16_BM];\n"
"    __shared__ float smB[G16_BK][G16_BN];\n"
"    int tid = threadIdx.x;\n"
"    int bm = blockIdx.y * G16_BM, bn = blockIdx.x * G16_BN;\n"
"    int tr = tid / 16, tc = tid % 16;\n"
"    float acc[G16_TM][G16_TN];\n"
"    for (int i = 0; i < G16_TM; i++)\n"
"        for (int j = 0; j < G16_TN; j++) acc[i][j] = 0.0f;\n"
"    for (int k = 0; k < K; k += G16_BK) {\n"
"        for (int _i = tid; _i < G16_BM*G16_BK; _i += 256) {\n"
"            int _r = _i % G16_BM, _c = _i / G16_BM;\n"
"            int gr = bm+_r, gk = k+_c;\n"
"            smA[_c][_r] = (gr<M&&gk<K) ? X[(size_t)gr*K+gk] : 0.0f;\n"
"        }\n"
"        for (int _i = tid; _i < G16_BN*G16_BK; _i += 256) {\n"
"            int _r = _i % G16_BN, _c = _i / G16_BN;\n"
"            int gn = bn+_r, gk = k+_c;\n"
"            smB[_c][_r] = (gn<N&&gk<K) ? half_to_float(W[(size_t)gn*K+gk]) : 0.0f;\n"
"        }\n"
"        __syncthreads();\n"
"        for (int kk = 0; kk < G16_BK; kk++) {\n"
"            float af[G16_TM], bf[G16_TN];\n"
"            for (int i=0;i<G16_TM;i++) af[i] = smA[kk][tr*G16_TM+i];\n"
"            for (int j=0;j<G16_TN;j++) bf[j] = smB[kk][tc*G16_TN+j];\n"
"            for (int i=0;i<G16_TM;i++)\n"
"                for (int j=0;j<G16_TN;j++) acc[i][j] += af[i]*bf[j];\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    for (int i=0;i<G16_TM;i++) {\n"
"        int gr = bm+tr*G16_TM+i; if(gr>=M) continue;\n"
"        for (int j=0;j<G16_TN;j++) {\n"
"            int gn = bn+tc*G16_TN+j;\n"
"            if(gn<N) Y[(size_t)gr*N+gn] = acc[i][j]+(bias?bias[gn]:0.0f)+residual[(size_t)gr*N+gn];\n"
"        }\n"
"    }\n"
"}\n"
"#undef G16_BM\n"
"#undef G16_BN\n"
"#undef G16_BK\n"
"#undef G16_TM\n"
"#undef G16_TN\n"
"\n"
"typedef unsigned short bf16_raw;\n"
"__device__ __forceinline__ float bf16_to_float(bf16_raw h) {\n"
"    return __uint_as_float(((unsigned int)h) << 16);\n"
"}\n"
"\n"
"/* ---- gemm_opt_bf16_f32: 128x128 tiled BF16-weight GEMM, F32 accum ---- */\n"
"__global__ void gemm_opt_bf16_f32(float *Y, const bf16_raw *W, const float *X,\n"
"                                   const float *bias, const float *residual,\n"
"                                   int epilogue, int N, int K, int M) {\n"
"    __shared__ float smA[128][16];\n"
"    __shared__ float smB[16][128];\n"
"    int tid = threadIdx.x;\n"
"    int bm = blockIdx.y * 128, bn = blockIdx.x * 128;\n"
"    int tr = tid / 16, tc = tid % 16;\n"
"    float acc[8][8];\n"
"    for (int i = 0; i < 8; i++)\n"
"        for (int j = 0; j < 8; j++) acc[i][j] = 0.0f;\n"
"    for (int k = 0; k < K; k += 16) {\n"
"        for (int idx = tid; idx < 128*16; idx += 256) {\n"
"            int mr = idx / 16, kc = idx % 16;\n"
"            int gr = bm + mr, gk = k + kc;\n"
"            smA[mr][kc] = (gr < M && gk < K) ? X[(size_t)gr*K + gk] : 0.0f;\n"
"        }\n"
"        for (int idx = tid; idx < 16*128; idx += 256) {\n"
"            int kc = idx / 128, nr = idx % 128;\n"
"            int gn = bn + nr, gk = k + kc;\n"
"            smB[kc][nr] = (gn < N && gk < K) ? bf16_to_float(W[(size_t)gn*K + gk]) : 0.0f;\n"
"        }\n"
"        __syncthreads();\n"
"        for (int kk = 0; kk < 16; kk++) {\n"
"            float af[8], bf[8];\n"
"            for (int i = 0; i < 8; i++) af[i] = smA[tr*8 + i][kk];\n"
"            for (int j = 0; j < 8; j++) bf[j] = smB[kk][tc*8 + j];\n"
"            for (int i = 0; i < 8; i++)\n"
"                for (int j = 0; j < 8; j++) acc[i][j] += af[i] * bf[j];\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    for (int i = 0; i < 8; i++) {\n"
"        int gr = bm + tr*8 + i;\n"
"        if (gr >= M) continue;\n"
"        for (int j = 0; j < 8; j++) {\n"
"            int gn = bn + tc*8 + j;\n"
"            if (gn < N) {\n"
"                float v = acc[i][j] + (bias ? bias[gn] : 0.0f);\n"
"                if (epilogue == 1) {\n"
"                    v = 0.5f*v*(1.0f+tanhf(0.7978845608f*(v+0.044715f*v*v*v)));\n"
"                } else if (epilogue == 2) {\n"
"                    v += residual[(size_t)gr*N + gn];\n"
"                }\n"
"                Y[(size_t)gr*N + gn] = v;\n"
"            }\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"typedef _Float16 f16x8 __attribute__((ext_vector_type(8)));\n"
"typedef unsigned short bf16x8 __attribute__((ext_vector_type(8)));\n"
"typedef float float8 __attribute__((ext_vector_type(8)));\n"
"__device__ __forceinline__ half_raw f32_to_f16_bits(float v) {\n"
"    __half hv = __float2half(v);\n"
"    return *((half_raw*)&hv);\n"
"}\n"
"__device__ __forceinline__ _Float16 f16_bits_to_f16(half_raw h) {\n"
"    _Float16 v;\n"
"    memcpy(&v, &h, 2);\n"
"    return v;\n"
"}\n"
"__device__ __forceinline__ unsigned short f32_to_bf16_bits(float v) {\n"
"    unsigned int bits;\n"
"    memcpy(&bits, &v, 4);\n"
"    unsigned int lsb = (bits >> 16) & 1u;\n"
"    return (unsigned short)((bits + 0x7fffu + lsb) >> 16);\n"
"}\n"
"\n"
"/* Vectorized: each thread converts 4 contiguous f32 elements -> 4 half-prec.\n"
" * Reads one float4 (b128) and writes one ushort4 (b64). Tail handled scalarly. */\n"
"__global__ void pack_f16_from_f32(half_raw *dst, const float *src, int n) {\n"
"    int tid = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int idx = tid * 4;\n"
"    int n4 = n & ~3;\n"
"    if (idx + 3 < n4) {\n"
"        float4 f = *(const float4 *)(src + idx);\n"
"        ushort4 b;\n"
"        b.x = f32_to_f16_bits(f.x);\n"
"        b.y = f32_to_f16_bits(f.y);\n"
"        b.z = f32_to_f16_bits(f.z);\n"
"        b.w = f32_to_f16_bits(f.w);\n"
"        *(ushort4 *)(dst + idx) = b;\n"
"    } else if (idx < n) {\n"
"        for (int k = 0; k < 4 && idx + k < n; k++)\n"
"            dst[idx + k] = f32_to_f16_bits(src[idx + k]);\n"
"    }\n"
"}\n"
"\n"
"__global__ void pack_bf16_from_f32(bf16_raw *dst, const float *src, int n) {\n"
"    int tid = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int idx = tid * 4;\n"
"    int n4 = n & ~3;\n"
"    if (idx + 3 < n4) {\n"
"        float4 f = *(const float4 *)(src + idx);\n"
"        ushort4 b;\n"
"        b.x = f32_to_bf16_bits(f.x);\n"
"        b.y = f32_to_bf16_bits(f.y);\n"
"        b.z = f32_to_bf16_bits(f.z);\n"
"        b.w = f32_to_bf16_bits(f.w);\n"
"        *(ushort4 *)(dst + idx) = b;\n"
"    } else if (idx < n) {\n"
"        for (int k = 0; k < 4 && idx + k < n; k++)\n"
"            dst[idx + k] = f32_to_bf16_bits(src[idx + k]);\n"
"    }\n"
"}\n"
"\n"
"/* ---- gemm_wmma_f16_f32: gfx12 WMMA F16×F16 -> F32, 128x128 CTA ---- */\n"
"__global__ void gemm_wmma_f16_f32(float *Y, const half_raw *W, const half_raw *X,\n"
"                                   const float *bias, const float *residual,\n"
"                                   int epilogue, int N, int K, int M) {\n"
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
"        for (int i = 0; i < 8; i++) {\n"
"            int row = wave_m0 + ms[t] + half * 8 + i;\n"
"            if (row >= M) continue;\n"
"            float v = acc[i] + (bias ? bias[col] : 0.0f);\n"
"            if (epilogue == 1) v = 0.5f*v*(1.0f+tanhf(0.7978845608f*(v+0.044715f*v*v*v)));\n"
"            else if (epilogue == 2) v += residual[(size_t)row * N + col];\n"
"            Y[(size_t)row * N + col] = v;\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"/* ---- gemm_wmma_bf16_f32: gfx12 WMMA BF16×BF16 -> F32, 128x128 CTA ---- */\n"
"__global__ void gemm_wmma_bf16_f32(float *Y, const bf16_raw *W, const bf16_raw *X,\n"
"                                    const float *bias, const float *residual,\n"
"                                    int epilogue, int N, int K, int M) {\n"
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
"        for (int i = 0; i < 8; i++) {\n"
"            int row = wave_m0 + ms[t] + half * 8 + i;\n"
"            if (row >= M) continue;\n"
"            float v = acc[i] + (bias ? bias[col] : 0.0f);\n"
"            if (epilogue == 1) v = 0.5f*v*(1.0f+tanhf(0.7978845608f*(v+0.044715f*v*v*v)));\n"
"            else if (epilogue == 2) v += residual[(size_t)row * N + col];\n"
"            Y[(size_t)row * N + col] = v;\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"/* ---- flash_attn_dyn_f32: FlashAttention with dynamic head_dim ---- */\n"
"/* Grid: (n_heads, ceil(n_tok/FA_DYN_BQ)), Block: FA_DYN_BQ=64 */\n"
"#define FA_DYN_BQ  64\n"
"#define FA_DYN_BKV 16\n"
"#define FA_DYN_MAX_HD 128\n"
"__global__ void flash_attn_dyn_f32(float *out, const float *qkv,\n"
"    const float *K_t, const float *V_t,\n"
"    int n_tok, int dim, int n_heads, int head_dim, float scale) {\n"
"    int h = blockIdx.x;\n"
"    int qi = blockIdx.y * FA_DYN_BQ + threadIdx.x;\n"
"    if (h >= n_heads) return;\n"
"    int dim3 = 3 * dim;\n"
"    const float *qt = qkv + (qi < n_tok ? qi : 0) * dim3 + h * head_dim;\n"
"    const float *kt_h = K_t + (size_t)h * n_tok * head_dim;\n"
"    const float *vt_h = V_t + (size_t)h * n_tok * head_dim;\n"
"    extern __shared__ float smem[];\n"
"    float *smK = smem;\n"
"    float *smV = smem + FA_DYN_BKV * head_dim;\n"
"    /* Use register arrays sized to max, only use head_dim elements */\n"
"    float q_reg[FA_DYN_MAX_HD];\n"
"    float O_i[FA_DYN_MAX_HD];\n"
"    for (int d = 0; d < head_dim; d++)\n"
"        q_reg[d] = (qi < n_tok) ? qt[d] : 0.0f;\n"
"    float m_i = -1e30f, l_i = 0.0f;\n"
"    for (int d = 0; d < head_dim; d++) O_i[d] = 0.0f;\n"
"    int tid = threadIdx.x;\n"
"    int kv_tiles = (n_tok + FA_DYN_BKV - 1) / FA_DYN_BKV;\n"
"    int smem_per_tile = FA_DYN_BKV * head_dim;\n"
"    for (int tile = 0; tile < kv_tiles; tile++) {\n"
"        int kv = tile * FA_DYN_BKV;\n"
"        for (int idx = tid; idx < smem_per_tile; idx += FA_DYN_BQ) {\n"
"            int kj = idx / head_dim, d = idx % head_dim;\n"
"            int kv_tok = kv + kj;\n"
"            smK[idx] = (kv_tok < n_tok) ? kt_h[(size_t)kv_tok * head_dim + d] : 0.0f;\n"
"            smV[idx] = (kv_tok < n_tok) ? vt_h[(size_t)kv_tok * head_dim + d] : 0.0f;\n"
"        }\n"
"        __syncthreads();\n"
"        float sc[FA_DYN_BKV];\n"
"        for (int kj = 0; kj < FA_DYN_BKV; kj++) sc[kj] = 0.0f;\n"
"        for (int d = 0; d < head_dim; d++) {\n"
"            float qd = q_reg[d];\n"
"            for (int kj = 0; kj < FA_DYN_BKV; kj++)\n"
"                sc[kj] += qd * smK[kj * head_dim + d];\n"
"        }\n"
"        float mx_tile = -1e30f;\n"
"        for (int kj = 0; kj < FA_DYN_BKV; kj++) {\n"
"            sc[kj] = (kv + kj < n_tok) ? sc[kj] * scale : -1e30f;\n"
"            if (sc[kj] > mx_tile) mx_tile = sc[kj];\n"
"        }\n"
"        float mn_i = fmaxf(m_i, mx_tile);\n"
"        float alpha = expf(m_i - mn_i);\n"
"        l_i *= alpha;\n"
"        for (int d = 0; d < head_dim; d++) O_i[d] *= alpha;\n"
"        m_i = mn_i;\n"
"        float ej[FA_DYN_BKV];\n"
"        for (int kj = 0; kj < FA_DYN_BKV; kj++) {\n"
"            ej[kj] = (kv + kj < n_tok) ? expf(sc[kj] - m_i) : 0.0f;\n"
"            l_i += ej[kj];\n"
"        }\n"
"        for (int kj = 0; kj < FA_DYN_BKV; kj++) {\n"
"            float e = ej[kj];\n"
"            for (int d = 0; d < head_dim; d++) O_i[d] += e * smV[kj * head_dim + d];\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    if (qi < n_tok) {\n"
"        float inv_l = (l_i > 0.0f) ? 1.0f / l_i : 0.0f;\n"
"        float *out_qi = out + (size_t)qi * dim + h * head_dim;\n"
"        for (int d = 0; d < head_dim; d++)\n"
"            out_qi[d] = O_i[d] * inv_l;\n"
"    }\n"
"}\n"
"#undef FA_DYN_BQ\n"
"#undef FA_DYN_BKV\n"
"#undef FA_DYN_MAX_HD\n"
"\n"
"/* ==== flash_attn_wmma_*: WMMA-accelerated FlashAttention ====              */\n"
"/* Each WG = 1 wave (32 threads) processing 1 head x BQ=16 queries.           */\n"
"/* Tiles: BQ=16, BKV=16. WMMA 16x16x16; head_dim padded to FA_WM_HD_PAD=80    */\n"
"/* (covers head_dim in {16,32,...,80}; suits Qwen3.6 head_dim=72).             */\n"
"#define FA_WM_HD_PAD 80\n"
"#define FA_WM_K_NB 5      /* HD_PAD / 16 */\n"
"#define FA_WM_BQ 16\n"
"#define FA_WM_BKV 16\n"
"\n"
"__global__ void flash_attn_wmma_bf16(float *out, const float *qkv,\n"
"    const float *K_t, const float *V_t,\n"
"    int n_tok, int dim, int n_heads, int head_dim, float scale) {\n"
"    int h = blockIdx.x;\n"
"    int qb = blockIdx.y;\n"
"    int q0 = qb * FA_WM_BQ;\n"
"    int tid = threadIdx.x;\n"
"    int half = tid >> 4;\n"
"    int idx  = tid & 15;\n"
"    int dim3 = 3 * dim;\n"
"    extern __shared__ unsigned short smem_b[];\n"
"    unsigned short *smK = smem_b;                      /* [16 x HD_PAD]      */\n"
"    unsigned short *smV = smK + 16 * FA_WM_HD_PAD;     /* [HD_PAD x 16] V^T  */\n"
"    unsigned short *smP = smV + FA_WM_HD_PAD * 16;     /* [16 x 16] BF16 P   */\n"
"\n"
"    /* Q tile [16 x HD_PAD] -> registers, pre-scaled, BF16. */\n"
"    bf16x8 q_reg[FA_WM_K_NB];\n"
"    for (int kb = 0; kb < FA_WM_K_NB; kb++) {\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int d = kb * 16 + half * 8 + i;\n"
"            int row = q0 + idx;\n"
"            float v = 0.0f;\n"
"            if (row < n_tok && d < head_dim) {\n"
"                v = qkv[(size_t)row * dim3 + h * head_dim + d];\n"
"            }\n"
"            q_reg[kb][i] = f32_to_bf16_bits(v * scale);\n"
"        }\n"
"    }\n"
"\n"
"    float8 O_acc[FA_WM_K_NB];\n"
"    for (int kb = 0; kb < FA_WM_K_NB; kb++)\n"
"        for (int i = 0; i < 8; i++) O_acc[kb][i] = 0.0f;\n"
"    float m_i[8], l_i[8];\n"
"    for (int i = 0; i < 8; i++) { m_i[i] = -1e30f; l_i[i] = 0.0f; }\n"
"\n"
"    int n_kv = (n_tok + FA_WM_BKV - 1) / FA_WM_BKV;\n"
"    for (int t = 0; t < n_kv; t++) {\n"
"        int kv0 = t * FA_WM_BKV;\n"
"        for (int e = tid; e < 16 * FA_WM_HD_PAD; e += 32) {\n"
"            int kv_row = e / FA_WM_HD_PAD;\n"
"            int d = e % FA_WM_HD_PAD;\n"
"            float v = 0.0f;\n"
"            int kv = kv0 + kv_row;\n"
"            if (kv < n_tok && d < head_dim)\n"
"                v = K_t[((size_t)h * n_tok + kv) * head_dim + d];\n"
"            smK[kv_row * FA_WM_HD_PAD + d] = f32_to_bf16_bits(v);\n"
"        }\n"
"        for (int e = tid; e < FA_WM_HD_PAD * 16; e += 32) {\n"
"            int d = e / 16;\n"
"            int kv_row = e & 15;\n"
"            float v = 0.0f;\n"
"            int kv = kv0 + kv_row;\n"
"            if (kv < n_tok && d < head_dim)\n"
"                v = V_t[((size_t)h * n_tok + kv) * head_dim + d];\n"
"            smV[d * 16 + kv_row] = f32_to_bf16_bits(v);\n"
"        }\n"
"        __syncthreads();\n"
"\n"
"        /* QK^T: D[m,n=kv_row] = sum_d Q[m,d]*K[kv_row,d] */\n"
"        float8 score = {0,0,0,0,0,0,0,0};\n"
"        for (int kb = 0; kb < FA_WM_K_NB; kb++) {\n"
"            bf16x8 b_K;\n"
"            for (int i = 0; i < 8; i++) {\n"
"                int dpos = kb * 16 + half * 8 + i;\n"
"                b_K[i] = smK[idx * FA_WM_HD_PAD + dpos];\n"
"            }\n"
"            score = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(q_reg[kb], b_K, score);\n"
"        }\n"
"        bool col_valid = (kv0 + idx) < n_tok;\n"
"        for (int i = 0; i < 8; i++) {\n"
"            score[i] = col_valid ? score[i] : -1e30f;\n"
"        }\n"
"        /* Online softmax: reduce max + sum across 16 lanes within same half. */\n"
"        float row_max[8];\n"
"        for (int i = 0; i < 8; i++) {\n"
"            float v = score[i];\n"
"            v = fmaxf(v, __shfl_xor(v, 1, 32));\n"
"            v = fmaxf(v, __shfl_xor(v, 2, 32));\n"
"            v = fmaxf(v, __shfl_xor(v, 4, 32));\n"
"            v = fmaxf(v, __shfl_xor(v, 8, 32));\n"
"            row_max[i] = v;\n"
"        }\n"
"        float alpha[8];\n"
"        for (int i = 0; i < 8; i++) {\n"
"            float new_max = fmaxf(m_i[i], row_max[i]);\n"
"            alpha[i] = __expf(m_i[i] - new_max);\n"
"            float ej = __expf(score[i] - new_max);\n"
"            float s = ej;\n"
"            s += __shfl_xor(s, 1, 32);\n"
"            s += __shfl_xor(s, 2, 32);\n"
"            s += __shfl_xor(s, 4, 32);\n"
"            s += __shfl_xor(s, 8, 32);\n"
"            l_i[i] = l_i[i] * alpha[i] + s;\n"
"            m_i[i] = new_max;\n"
"            score[i] = ej;\n"
"        }\n"
"        for (int kb = 0; kb < FA_WM_K_NB; kb++)\n"
"            for (int i = 0; i < 8; i++) O_acc[kb][i] *= alpha[i];\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int m_row = half * 8 + i;\n"
"            int n_col = idx;\n"
"            smP[m_row * 16 + n_col] = f32_to_bf16_bits(score[i]);\n"
"        }\n"
"        __syncthreads();\n"
"\n"
"        /* P @ V: O[m,d] += sum_kv P[m,kv]*V_T[d,kv] */\n"
"        bf16x8 ap;\n"
"        for (int i = 0; i < 8; i++) ap[i] = smP[idx * 16 + half * 8 + i];\n"
"        for (int kb = 0; kb < FA_WM_K_NB; kb++) {\n"
"            bf16x8 bv;\n"
"            for (int i = 0; i < 8; i++) {\n"
"                int d_col = kb * 16 + idx;\n"
"                int kv_k  = half * 8 + i;\n"
"                bv[i] = smV[d_col * 16 + kv_k];\n"
"            }\n"
"            O_acc[kb] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(ap, bv, O_acc[kb]);\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"\n"
"    for (int kb = 0; kb < FA_WM_K_NB; kb++) {\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int m_row = half * 8 + i;\n"
"            int d = kb * 16 + idx;\n"
"            int row = q0 + m_row;\n"
"            if (row < n_tok && d < head_dim) {\n"
"                float v = O_acc[kb][i] / (l_i[i] > 0.0f ? l_i[i] : 1.0f);\n"
"                out[(size_t)row * dim + h * head_dim + d] = v;\n"
"            }\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"__global__ void flash_attn_wmma_f16(float *out, const float *qkv,\n"
"    const float *K_t, const float *V_t,\n"
"    int n_tok, int dim, int n_heads, int head_dim, float scale) {\n"
"    int h = blockIdx.x;\n"
"    int qb = blockIdx.y;\n"
"    int q0 = qb * FA_WM_BQ;\n"
"    int tid = threadIdx.x;\n"
"    int half = tid >> 4;\n"
"    int idx  = tid & 15;\n"
"    int dim3 = 3 * dim;\n"
"    extern __shared__ _Float16 smem_h[];\n"
"    _Float16 *smK = smem_h;\n"
"    _Float16 *smV = smK + 16 * FA_WM_HD_PAD;\n"
"    _Float16 *smP = smV + FA_WM_HD_PAD * 16;\n"
"\n"
"    f16x8 q_reg[FA_WM_K_NB];\n"
"    for (int kb = 0; kb < FA_WM_K_NB; kb++) {\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int d = kb * 16 + half * 8 + i;\n"
"            int row = q0 + idx;\n"
"            float v = 0.0f;\n"
"            if (row < n_tok && d < head_dim) {\n"
"                v = qkv[(size_t)row * dim3 + h * head_dim + d];\n"
"            }\n"
"            q_reg[kb][i] = (_Float16)(v * scale);\n"
"        }\n"
"    }\n"
"    float8 O_acc[FA_WM_K_NB];\n"
"    for (int kb = 0; kb < FA_WM_K_NB; kb++)\n"
"        for (int i = 0; i < 8; i++) O_acc[kb][i] = 0.0f;\n"
"    float m_i[8], l_i[8];\n"
"    for (int i = 0; i < 8; i++) { m_i[i] = -1e30f; l_i[i] = 0.0f; }\n"
"\n"
"    int n_kv = (n_tok + FA_WM_BKV - 1) / FA_WM_BKV;\n"
"    for (int t = 0; t < n_kv; t++) {\n"
"        int kv0 = t * FA_WM_BKV;\n"
"        for (int e = tid; e < 16 * FA_WM_HD_PAD; e += 32) {\n"
"            int kv_row = e / FA_WM_HD_PAD;\n"
"            int d = e % FA_WM_HD_PAD;\n"
"            float v = 0.0f;\n"
"            int kv = kv0 + kv_row;\n"
"            if (kv < n_tok && d < head_dim)\n"
"                v = K_t[((size_t)h * n_tok + kv) * head_dim + d];\n"
"            smK[kv_row * FA_WM_HD_PAD + d] = (_Float16)v;\n"
"        }\n"
"        for (int e = tid; e < FA_WM_HD_PAD * 16; e += 32) {\n"
"            int d = e / 16;\n"
"            int kv_row = e & 15;\n"
"            float v = 0.0f;\n"
"            int kv = kv0 + kv_row;\n"
"            if (kv < n_tok && d < head_dim)\n"
"                v = V_t[((size_t)h * n_tok + kv) * head_dim + d];\n"
"            smV[d * 16 + kv_row] = (_Float16)v;\n"
"        }\n"
"        __syncthreads();\n"
"\n"
"        float8 score = {0,0,0,0,0,0,0,0};\n"
"        for (int kb = 0; kb < FA_WM_K_NB; kb++) {\n"
"            f16x8 b_K;\n"
"            for (int i = 0; i < 8; i++) {\n"
"                int dpos = kb * 16 + half * 8 + i;\n"
"                b_K[i] = smK[idx * FA_WM_HD_PAD + dpos];\n"
"            }\n"
"            score = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(q_reg[kb], b_K, score);\n"
"        }\n"
"        bool col_valid = (kv0 + idx) < n_tok;\n"
"        for (int i = 0; i < 8; i++) {\n"
"            score[i] = col_valid ? score[i] : -1e30f;\n"
"        }\n"
"        float row_max[8];\n"
"        for (int i = 0; i < 8; i++) {\n"
"            float v = score[i];\n"
"            v = fmaxf(v, __shfl_xor(v, 1, 32));\n"
"            v = fmaxf(v, __shfl_xor(v, 2, 32));\n"
"            v = fmaxf(v, __shfl_xor(v, 4, 32));\n"
"            v = fmaxf(v, __shfl_xor(v, 8, 32));\n"
"            row_max[i] = v;\n"
"        }\n"
"        float alpha[8];\n"
"        for (int i = 0; i < 8; i++) {\n"
"            float new_max = fmaxf(m_i[i], row_max[i]);\n"
"            alpha[i] = __expf(m_i[i] - new_max);\n"
"            float ej = __expf(score[i] - new_max);\n"
"            float s = ej;\n"
"            s += __shfl_xor(s, 1, 32);\n"
"            s += __shfl_xor(s, 2, 32);\n"
"            s += __shfl_xor(s, 4, 32);\n"
"            s += __shfl_xor(s, 8, 32);\n"
"            l_i[i] = l_i[i] * alpha[i] + s;\n"
"            m_i[i] = new_max;\n"
"            score[i] = ej;\n"
"        }\n"
"        for (int kb = 0; kb < FA_WM_K_NB; kb++)\n"
"            for (int i = 0; i < 8; i++) O_acc[kb][i] *= alpha[i];\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int m_row = half * 8 + i;\n"
"            int n_col = idx;\n"
"            smP[m_row * 16 + n_col] = (_Float16)score[i];\n"
"        }\n"
"        __syncthreads();\n"
"\n"
"        f16x8 ap;\n"
"        for (int i = 0; i < 8; i++) ap[i] = smP[idx * 16 + half * 8 + i];\n"
"        for (int kb = 0; kb < FA_WM_K_NB; kb++) {\n"
"            f16x8 bv;\n"
"            for (int i = 0; i < 8; i++) {\n"
"                int d_col = kb * 16 + idx;\n"
"                int kv_k  = half * 8 + i;\n"
"                bv[i] = smV[d_col * 16 + kv_k];\n"
"            }\n"
"            O_acc[kb] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ap, bv, O_acc[kb]);\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    for (int kb = 0; kb < FA_WM_K_NB; kb++) {\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int m_row = half * 8 + i;\n"
"            int d = kb * 16 + idx;\n"
"            int row = q0 + m_row;\n"
"            if (row < n_tok && d < head_dim) {\n"
"                float v = O_acc[kb][i] / (l_i[i] > 0.0f ? l_i[i] : 1.0f);\n"
"                out[(size_t)row * dim + h * head_dim + d] = v;\n"
"            }\n"
"        }\n"
"    }\n"
"}\n"
"#undef FA_WM_HD_PAD\n"
"#undef FA_WM_K_NB\n"
"#undef FA_WM_BQ\n"
"#undef FA_WM_BKV\n"
"\n"
"/* ==== flash_attn_wmma_*_2w: BQ=32, 2 waves per WG ====                     */\n"
"/* K/V loads amortize across 32 queries; cooperative 64-thread LDS load.      */\n"
"#define FA_WM2_HD_PAD 80\n"
"#define FA_WM2_K_NB 5\n"
"#define FA_WM2_BQ 32\n"
"#define FA_WM2_BKV 16\n"
"\n"
"__global__ void flash_attn_wmma_bf16_2w(float *out, const float *qkv,\n"
"    const float *K_t, const float *V_t,\n"
"    int n_tok, int dim, int n_heads, int head_dim, float scale) {\n"
"    int h = blockIdx.x;\n"
"    int qb = blockIdx.y;\n"
"    int q0 = qb * FA_WM2_BQ;\n"
"    int tid = threadIdx.x;     /* 0..63 */\n"
"    int wid = tid >> 5;\n"
"    int lid = tid & 31;\n"
"    int half = lid >> 4;\n"
"    int idx  = lid & 15;\n"
"    int dim3 = 3 * dim;\n"
"    extern __shared__ unsigned short smem_b[];\n"
"    unsigned short *smK = smem_b;                                /* [16xHD]  */\n"
"    unsigned short *smV = smK + 16 * FA_WM2_HD_PAD;              /* [HDx16]  */\n"
"    unsigned short *smP = smV + FA_WM2_HD_PAD * 16;              /* 2 x 16x16*/\n"
"    unsigned short *smP_w = smP + wid * 16 * 16;\n"
"\n"
"    int q_base = q0 + wid * 16;\n"
"    bf16x8 q_reg[FA_WM2_K_NB];\n"
"    for (int kb = 0; kb < FA_WM2_K_NB; kb++) {\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int d = kb * 16 + half * 8 + i;\n"
"            int row = q_base + idx;\n"
"            float v = 0.0f;\n"
"            if (row < n_tok && d < head_dim)\n"
"                v = qkv[(size_t)row * dim3 + h * head_dim + d];\n"
"            q_reg[kb][i] = f32_to_bf16_bits(v * scale);\n"
"        }\n"
"    }\n"
"    float8 O_acc[FA_WM2_K_NB];\n"
"    for (int kb = 0; kb < FA_WM2_K_NB; kb++)\n"
"        for (int i = 0; i < 8; i++) O_acc[kb][i] = 0.0f;\n"
"    float m_i[8], l_i[8];\n"
"    for (int i = 0; i < 8; i++) { m_i[i] = -1e30f; l_i[i] = 0.0f; }\n"
"\n"
"    int n_kv = (n_tok + FA_WM2_BKV - 1) / FA_WM2_BKV;\n"
"    for (int t = 0; t < n_kv; t++) {\n"
"        int kv0 = t * FA_WM2_BKV;\n"
"        for (int e = tid; e < 16 * FA_WM2_HD_PAD; e += 64) {\n"
"            int kv_row = e / FA_WM2_HD_PAD;\n"
"            int d = e % FA_WM2_HD_PAD;\n"
"            float v = 0.0f;\n"
"            int kv = kv0 + kv_row;\n"
"            if (kv < n_tok && d < head_dim)\n"
"                v = K_t[((size_t)h * n_tok + kv) * head_dim + d];\n"
"            smK[kv_row * FA_WM2_HD_PAD + d] = f32_to_bf16_bits(v);\n"
"        }\n"
"        for (int e = tid; e < FA_WM2_HD_PAD * 16; e += 64) {\n"
"            int d = e / 16;\n"
"            int kv_row = e & 15;\n"
"            float v = 0.0f;\n"
"            int kv = kv0 + kv_row;\n"
"            if (kv < n_tok && d < head_dim)\n"
"                v = V_t[((size_t)h * n_tok + kv) * head_dim + d];\n"
"            smV[d * 16 + kv_row] = f32_to_bf16_bits(v);\n"
"        }\n"
"        __syncthreads();\n"
"\n"
"        float8 score = {0,0,0,0,0,0,0,0};\n"
"        for (int kb = 0; kb < FA_WM2_K_NB; kb++) {\n"
"            bf16x8 b_K;\n"
"            for (int i = 0; i < 8; i++) {\n"
"                int dpos = kb * 16 + half * 8 + i;\n"
"                b_K[i] = smK[idx * FA_WM2_HD_PAD + dpos];\n"
"            }\n"
"            score = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(q_reg[kb], b_K, score);\n"
"        }\n"
"        bool col_valid = (kv0 + idx) < n_tok;\n"
"        for (int i = 0; i < 8; i++) score[i] = col_valid ? score[i] : -1e30f;\n"
"        float row_max[8];\n"
"        for (int i = 0; i < 8; i++) {\n"
"            float v = score[i];\n"
"            v = fmaxf(v, __shfl_xor(v, 1, 32));\n"
"            v = fmaxf(v, __shfl_xor(v, 2, 32));\n"
"            v = fmaxf(v, __shfl_xor(v, 4, 32));\n"
"            v = fmaxf(v, __shfl_xor(v, 8, 32));\n"
"            row_max[i] = v;\n"
"        }\n"
"        float alpha[8];\n"
"        for (int i = 0; i < 8; i++) {\n"
"            float new_max = fmaxf(m_i[i], row_max[i]);\n"
"            alpha[i] = __expf(m_i[i] - new_max);\n"
"            float ej = __expf(score[i] - new_max);\n"
"            float s = ej;\n"
"            s += __shfl_xor(s, 1, 32);\n"
"            s += __shfl_xor(s, 2, 32);\n"
"            s += __shfl_xor(s, 4, 32);\n"
"            s += __shfl_xor(s, 8, 32);\n"
"            l_i[i] = l_i[i] * alpha[i] + s;\n"
"            m_i[i] = new_max;\n"
"            score[i] = ej;\n"
"        }\n"
"        for (int kb = 0; kb < FA_WM2_K_NB; kb++)\n"
"            for (int i = 0; i < 8; i++) O_acc[kb][i] *= alpha[i];\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int m_row = half * 8 + i;\n"
"            int n_col = idx;\n"
"            smP_w[m_row * 16 + n_col] = f32_to_bf16_bits(score[i]);\n"
"        }\n"
"        bf16x8 ap;\n"
"        for (int i = 0; i < 8; i++) ap[i] = smP_w[idx * 16 + half * 8 + i];\n"
"        for (int kb = 0; kb < FA_WM2_K_NB; kb++) {\n"
"            bf16x8 bv;\n"
"            for (int i = 0; i < 8; i++) {\n"
"                int d_col = kb * 16 + idx;\n"
"                int kv_k  = half * 8 + i;\n"
"                bv[i] = smV[d_col * 16 + kv_k];\n"
"            }\n"
"            O_acc[kb] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(ap, bv, O_acc[kb]);\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    for (int kb = 0; kb < FA_WM2_K_NB; kb++) {\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int m_row = half * 8 + i;\n"
"            int d = kb * 16 + idx;\n"
"            int row = q_base + m_row;\n"
"            if (row < n_tok && d < head_dim) {\n"
"                float v = O_acc[kb][i] / (l_i[i] > 0.0f ? l_i[i] : 1.0f);\n"
"                out[(size_t)row * dim + h * head_dim + d] = v;\n"
"            }\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"__global__ void flash_attn_wmma_f16_2w(float *out, const float *qkv,\n"
"    const float *K_t, const float *V_t,\n"
"    int n_tok, int dim, int n_heads, int head_dim, float scale) {\n"
"    int h = blockIdx.x;\n"
"    int qb = blockIdx.y;\n"
"    int q0 = qb * FA_WM2_BQ;\n"
"    int tid = threadIdx.x;\n"
"    int wid = tid >> 5;\n"
"    int lid = tid & 31;\n"
"    int half = lid >> 4;\n"
"    int idx  = lid & 15;\n"
"    int dim3 = 3 * dim;\n"
"    extern __shared__ _Float16 smem_h[];\n"
"    _Float16 *smK = smem_h;\n"
"    _Float16 *smV = smK + 16 * FA_WM2_HD_PAD;\n"
"    _Float16 *smP = smV + FA_WM2_HD_PAD * 16;\n"
"    _Float16 *smP_w = smP + wid * 16 * 16;\n"
"    int q_base = q0 + wid * 16;\n"
"    f16x8 q_reg[FA_WM2_K_NB];\n"
"    for (int kb = 0; kb < FA_WM2_K_NB; kb++) {\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int d = kb * 16 + half * 8 + i;\n"
"            int row = q_base + idx;\n"
"            float v = 0.0f;\n"
"            if (row < n_tok && d < head_dim)\n"
"                v = qkv[(size_t)row * dim3 + h * head_dim + d];\n"
"            q_reg[kb][i] = (_Float16)(v * scale);\n"
"        }\n"
"    }\n"
"    float8 O_acc[FA_WM2_K_NB];\n"
"    for (int kb = 0; kb < FA_WM2_K_NB; kb++)\n"
"        for (int i = 0; i < 8; i++) O_acc[kb][i] = 0.0f;\n"
"    float m_i[8], l_i[8];\n"
"    for (int i = 0; i < 8; i++) { m_i[i] = -1e30f; l_i[i] = 0.0f; }\n"
"    int n_kv = (n_tok + FA_WM2_BKV - 1) / FA_WM2_BKV;\n"
"    for (int t = 0; t < n_kv; t++) {\n"
"        int kv0 = t * FA_WM2_BKV;\n"
"        for (int e = tid; e < 16 * FA_WM2_HD_PAD; e += 64) {\n"
"            int kv_row = e / FA_WM2_HD_PAD;\n"
"            int d = e % FA_WM2_HD_PAD;\n"
"            float v = 0.0f;\n"
"            int kv = kv0 + kv_row;\n"
"            if (kv < n_tok && d < head_dim)\n"
"                v = K_t[((size_t)h * n_tok + kv) * head_dim + d];\n"
"            smK[kv_row * FA_WM2_HD_PAD + d] = (_Float16)v;\n"
"        }\n"
"        for (int e = tid; e < FA_WM2_HD_PAD * 16; e += 64) {\n"
"            int d = e / 16;\n"
"            int kv_row = e & 15;\n"
"            float v = 0.0f;\n"
"            int kv = kv0 + kv_row;\n"
"            if (kv < n_tok && d < head_dim)\n"
"                v = V_t[((size_t)h * n_tok + kv) * head_dim + d];\n"
"            smV[d * 16 + kv_row] = (_Float16)v;\n"
"        }\n"
"        __syncthreads();\n"
"        float8 score = {0,0,0,0,0,0,0,0};\n"
"        for (int kb = 0; kb < FA_WM2_K_NB; kb++) {\n"
"            f16x8 b_K;\n"
"            for (int i = 0; i < 8; i++) {\n"
"                int dpos = kb * 16 + half * 8 + i;\n"
"                b_K[i] = smK[idx * FA_WM2_HD_PAD + dpos];\n"
"            }\n"
"            score = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(q_reg[kb], b_K, score);\n"
"        }\n"
"        bool col_valid = (kv0 + idx) < n_tok;\n"
"        for (int i = 0; i < 8; i++) score[i] = col_valid ? score[i] : -1e30f;\n"
"        float row_max[8];\n"
"        for (int i = 0; i < 8; i++) {\n"
"            float v = score[i];\n"
"            v = fmaxf(v, __shfl_xor(v, 1, 32));\n"
"            v = fmaxf(v, __shfl_xor(v, 2, 32));\n"
"            v = fmaxf(v, __shfl_xor(v, 4, 32));\n"
"            v = fmaxf(v, __shfl_xor(v, 8, 32));\n"
"            row_max[i] = v;\n"
"        }\n"
"        float alpha[8];\n"
"        for (int i = 0; i < 8; i++) {\n"
"            float new_max = fmaxf(m_i[i], row_max[i]);\n"
"            alpha[i] = __expf(m_i[i] - new_max);\n"
"            float ej = __expf(score[i] - new_max);\n"
"            float s = ej;\n"
"            s += __shfl_xor(s, 1, 32);\n"
"            s += __shfl_xor(s, 2, 32);\n"
"            s += __shfl_xor(s, 4, 32);\n"
"            s += __shfl_xor(s, 8, 32);\n"
"            l_i[i] = l_i[i] * alpha[i] + s;\n"
"            m_i[i] = new_max;\n"
"            score[i] = ej;\n"
"        }\n"
"        for (int kb = 0; kb < FA_WM2_K_NB; kb++)\n"
"            for (int i = 0; i < 8; i++) O_acc[kb][i] *= alpha[i];\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int m_row = half * 8 + i;\n"
"            int n_col = idx;\n"
"            smP_w[m_row * 16 + n_col] = (_Float16)score[i];\n"
"        }\n"
"        f16x8 ap;\n"
"        for (int i = 0; i < 8; i++) ap[i] = smP_w[idx * 16 + half * 8 + i];\n"
"        for (int kb = 0; kb < FA_WM2_K_NB; kb++) {\n"
"            f16x8 bv;\n"
"            for (int i = 0; i < 8; i++) {\n"
"                int d_col = kb * 16 + idx;\n"
"                int kv_k  = half * 8 + i;\n"
"                bv[i] = smV[d_col * 16 + kv_k];\n"
"            }\n"
"            O_acc[kb] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ap, bv, O_acc[kb]);\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    for (int kb = 0; kb < FA_WM2_K_NB; kb++) {\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int m_row = half * 8 + i;\n"
"            int d = kb * 16 + idx;\n"
"            int row = q_base + m_row;\n"
"            if (row < n_tok && d < head_dim) {\n"
"                float v = O_acc[kb][i] / (l_i[i] > 0.0f ? l_i[i] : 1.0f);\n"
"                out[(size_t)row * dim + h * head_dim + d] = v;\n"
"            }\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"/* ---- kv_transpose_bf16/f16: pre-pack K,V into half-precision per-head layout ---- */\n"
"/* Writes K_t/V_t as [n_heads, n_tok, head_dim] in BF16 (or F16). One thread/elt. */\n"
"__global__ void kv_transpose_bf16(unsigned short *K_t, unsigned short *V_t,\n"
"                                   const float *qkv,\n"
"                                   int n_tok, int dim, int n_heads, int head_dim) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = n_tok * dim;\n"
"    if (idx >= total) return;\n"
"    int tok = idx / dim;\n"
"    int hd_idx = idx % dim;\n"
"    int h = hd_idx / head_dim;\n"
"    int d = hd_idx % head_dim;\n"
"    int dim3 = 3 * dim;\n"
"    int dst_idx = h * n_tok * head_dim + tok * head_dim + d;\n"
"    K_t[dst_idx] = f32_to_bf16_bits(qkv[tok * dim3 + dim     + hd_idx]);\n"
"    V_t[dst_idx] = f32_to_bf16_bits(qkv[tok * dim3 + 2 * dim + hd_idx]);\n"
"}\n"
"\n"
"__global__ void kv_transpose_f16(_Float16 *K_t, _Float16 *V_t,\n"
"                                  const float *qkv,\n"
"                                  int n_tok, int dim, int n_heads, int head_dim) {\n"
"    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    int total = n_tok * dim;\n"
"    if (idx >= total) return;\n"
"    int tok = idx / dim;\n"
"    int hd_idx = idx % dim;\n"
"    int h = hd_idx / head_dim;\n"
"    int d = hd_idx % head_dim;\n"
"    int dim3 = 3 * dim;\n"
"    int dst_idx = h * n_tok * head_dim + tok * head_dim + d;\n"
"    K_t[dst_idx] = (_Float16)qkv[tok * dim3 + dim     + hd_idx];\n"
"    V_t[dst_idx] = (_Float16)qkv[tok * dim3 + 2 * dim + hd_idx];\n"
"}\n"
"\n"
"/* ==== flash_attn_wmma_*_2w_pre: BQ=32 2-wave with pre-packed BF16/F16 K_t,V_t ==== */\n"
"/* Same algorithm as _2w but reads K_t/V_t directly (no F32 read + cast in hot loop). */\n"
"__global__ void flash_attn_wmma_bf16_2w_pre(float *out, const float *qkv,\n"
"    const unsigned short *K_t, const unsigned short *V_t,\n"
"    int n_tok, int dim, int n_heads, int head_dim, float scale) {\n"
"    int h = blockIdx.x;\n"
"    int qb = blockIdx.y;\n"
"    int q0 = qb * FA_WM2_BQ;\n"
"    int tid = threadIdx.x;\n"
"    int wid = tid >> 5;\n"
"    int lid = tid & 31;\n"
"    int half = lid >> 4;\n"
"    int idx  = lid & 15;\n"
"    int dim3 = 3 * dim;\n"
"    extern __shared__ unsigned short smem_b[];\n"
"    unsigned short *smK0 = smem_b;\n"
"    unsigned short *smK1 = smK0 + 16 * FA_WM2_HD_PAD;\n"
"    unsigned short *smV0 = smK1 + 16 * FA_WM2_HD_PAD;\n"
"    unsigned short *smV1 = smV0 + FA_WM2_HD_PAD * 16;\n"
"    unsigned short *smP = smV1 + FA_WM2_HD_PAD * 16;\n"
"    unsigned short *smP_w = smP + wid * 16 * 16;\n"
"\n"
"    int q_base = q0 + wid * 16;\n"
"    bf16x8 q_reg[FA_WM2_K_NB];\n"
"    for (int kb = 0; kb < FA_WM2_K_NB; kb++) {\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int d = kb * 16 + half * 8 + i;\n"
"            int row = q_base + idx;\n"
"            float v = 0.0f;\n"
"            if (row < n_tok && d < head_dim)\n"
"                v = qkv[(size_t)row * dim3 + h * head_dim + d];\n"
"            q_reg[kb][i] = f32_to_bf16_bits(v * scale);\n"
"        }\n"
"    }\n"
"    float8 O_acc[FA_WM2_K_NB];\n"
"    for (int kb = 0; kb < FA_WM2_K_NB; kb++)\n"
"        for (int i = 0; i < 8; i++) O_acc[kb][i] = 0.0f;\n"
"    float m_i[8], l_i[8];\n"
"    for (int i = 0; i < 8; i++) { m_i[i] = -1e30f; l_i[i] = 0.0f; }\n"
"\n"
"    int n_kv = (n_tok + FA_WM2_BKV - 1) / FA_WM2_BKV;\n"
"    int ld_row = tid >> 2;\n"
"    int ld_col0 = (tid & 3) * 20;\n"
"\n"
"    /* Prologue: load tile 0 into buffer 0 */\n"
"    {\n"
"        int kv = ld_row;\n"
"        size_t k_base = ((size_t)h * n_tok + kv) * head_dim;\n"
"        bool kv_ok = (kv < n_tok);\n"
"        for (int j = 0; j < 20; j++) {\n"
"            int d = ld_col0 + j;\n"
"            unsigned short k_v = 0, v_v = 0;\n"
"            if (kv_ok && d < head_dim) {\n"
"                k_v = K_t[k_base + d];\n"
"                v_v = V_t[k_base + d];\n"
"            }\n"
"            smK0[ld_row * FA_WM2_HD_PAD + d] = k_v;\n"
"            smV0[d * 16 + ld_row] = v_v;\n"
"        }\n"
"    }\n"
"    __syncthreads();\n"
"\n"
"    for (int t = 0; t < n_kv; t++) {\n"
"        unsigned short *smK_cur = (t & 1) ? smK1 : smK0;\n"
"        unsigned short *smV_cur = (t & 1) ? smV1 : smV0;\n"
"        unsigned short *smK_pre = (t & 1) ? smK0 : smK1;\n"
"        unsigned short *smV_pre = (t & 1) ? smV0 : smV1;\n"
"        int t_next = t + 1;\n"
"        bool prefetch = t_next < n_kv;\n"
"\n"
"        if (prefetch) {\n"
"            int kv_pre = t_next * FA_WM2_BKV + ld_row;\n"
"            size_t k_base_pre = ((size_t)h * n_tok + kv_pre) * head_dim;\n"
"            bool pre_ok = (kv_pre < n_tok);\n"
"            for (int j = 0; j < 20; j++) {\n"
"                int d = ld_col0 + j;\n"
"                unsigned short k_v = 0, v_v = 0;\n"
"                if (pre_ok && d < head_dim) {\n"
"                    k_v = K_t[k_base_pre + d];\n"
"                    v_v = V_t[k_base_pre + d];\n"
"                }\n"
"                smK_pre[ld_row * FA_WM2_HD_PAD + d] = k_v;\n"
"                smV_pre[d * 16 + ld_row] = v_v;\n"
"            }\n"
"        }\n"
"\n"
"        int kv0 = t * FA_WM2_BKV;\n"
"        float8 score = {0,0,0,0,0,0,0,0};\n"
"        for (int kb = 0; kb < FA_WM2_K_NB; kb++) {\n"
"            bf16x8 b_K;\n"
"            for (int i = 0; i < 8; i++) {\n"
"                int dpos = kb * 16 + half * 8 + i;\n"
"                b_K[i] = smK_cur[idx * FA_WM2_HD_PAD + dpos];\n"
"            }\n"
"            score = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(q_reg[kb], b_K, score);\n"
"        }\n"
"        bool col_valid = (kv0 + idx) < n_tok;\n"
"        for (int i = 0; i < 8; i++) score[i] = col_valid ? score[i] : -1e30f;\n"
"        float row_max[8];\n"
"        for (int i = 0; i < 8; i++) {\n"
"            float v = score[i];\n"
"            v = fmaxf(v, __shfl_xor(v, 1, 32));\n"
"            v = fmaxf(v, __shfl_xor(v, 2, 32));\n"
"            v = fmaxf(v, __shfl_xor(v, 4, 32));\n"
"            v = fmaxf(v, __shfl_xor(v, 8, 32));\n"
"            row_max[i] = v;\n"
"        }\n"
"        float alpha[8];\n"
"        for (int i = 0; i < 8; i++) {\n"
"            float new_max = fmaxf(m_i[i], row_max[i]);\n"
"            alpha[i] = __expf(m_i[i] - new_max);\n"
"            float ej = __expf(score[i] - new_max);\n"
"            float s = ej;\n"
"            s += __shfl_xor(s, 1, 32);\n"
"            s += __shfl_xor(s, 2, 32);\n"
"            s += __shfl_xor(s, 4, 32);\n"
"            s += __shfl_xor(s, 8, 32);\n"
"            l_i[i] = l_i[i] * alpha[i] + s;\n"
"            m_i[i] = new_max;\n"
"            score[i] = ej;\n"
"        }\n"
"        for (int kb = 0; kb < FA_WM2_K_NB; kb++)\n"
"            for (int i = 0; i < 8; i++) O_acc[kb][i] *= alpha[i];\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int m_row = half * 8 + i;\n"
"            int n_col = idx;\n"
"            smP_w[m_row * 16 + n_col] = f32_to_bf16_bits(score[i]);\n"
"        }\n"
"        bf16x8 ap;\n"
"        for (int i = 0; i < 8; i++) ap[i] = smP_w[idx * 16 + half * 8 + i];\n"
"        for (int kb = 0; kb < FA_WM2_K_NB; kb++) {\n"
"            bf16x8 bv;\n"
"            for (int i = 0; i < 8; i++) {\n"
"                int d_col = kb * 16 + idx;\n"
"                int kv_k  = half * 8 + i;\n"
"                bv[i] = smV_cur[d_col * 16 + kv_k];\n"
"            }\n"
"            O_acc[kb] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(ap, bv, O_acc[kb]);\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    for (int kb = 0; kb < FA_WM2_K_NB; kb++) {\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int m_row = half * 8 + i;\n"
"            int d = kb * 16 + idx;\n"
"            int row = q_base + m_row;\n"
"            if (row < n_tok && d < head_dim) {\n"
"                float v = O_acc[kb][i] / (l_i[i] > 0.0f ? l_i[i] : 1.0f);\n"
"                out[(size_t)row * dim + h * head_dim + d] = v;\n"
"            }\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"/* ==== flash_attn_wmma_bf16_4w_pre: BQ=64 4-wave BF16 with double-buffered K/V ====\n"
" * 4 waves (128 threads), each owning 16 queries. One K/V tile is loaded\n"
" * cooperatively by all 128 threads (8 threads/row x 10 ushorts each), then\n"
" * shared across all 4 waves' WMMAs. Double-buffered like _2w_pre. */\n"
"__global__ void flash_attn_wmma_bf16_4w_pre(float *out, const float *qkv,\n"
"    const unsigned short *K_t, const unsigned short *V_t,\n"
"    int n_tok, int dim, int n_heads, int head_dim, float scale) {\n"
"    int h = blockIdx.x;\n"
"    int qb = blockIdx.y;\n"
"    int q0 = qb * 64;  /* BQ=64 */\n"
"    int tid = threadIdx.x;\n"
"    int wid = tid >> 5;        /* 0..3 */\n"
"    int lid = tid & 31;\n"
"    int half = lid >> 4;\n"
"    int idx  = lid & 15;\n"
"    int dim3 = 3 * dim;\n"
"    extern __shared__ unsigned short smem_b[];\n"
"    unsigned short *smK0 = smem_b;\n"
"    unsigned short *smK1 = smK0 + 16 * FA_WM2_HD_PAD;\n"
"    unsigned short *smV0 = smK1 + 16 * FA_WM2_HD_PAD;\n"
"    unsigned short *smV1 = smV0 + FA_WM2_HD_PAD * 16;\n"
"    unsigned short *smP  = smV1 + FA_WM2_HD_PAD * 16;\n"
"    unsigned short *smP_w = smP + wid * 16 * 16;\n"
"\n"
"    int q_base = q0 + wid * 16;\n"
"    bf16x8 q_reg[FA_WM2_K_NB];\n"
"    for (int kb = 0; kb < FA_WM2_K_NB; kb++) {\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int d = kb * 16 + half * 8 + i;\n"
"            int row = q_base + idx;\n"
"            float v = 0.0f;\n"
"            if (row < n_tok && d < head_dim)\n"
"                v = qkv[(size_t)row * dim3 + h * head_dim + d];\n"
"            q_reg[kb][i] = f32_to_bf16_bits(v * scale);\n"
"        }\n"
"    }\n"
"    float8 O_acc[FA_WM2_K_NB];\n"
"    for (int kb = 0; kb < FA_WM2_K_NB; kb++)\n"
"        for (int i = 0; i < 8; i++) O_acc[kb][i] = 0.0f;\n"
"    float m_i[8], l_i[8];\n"
"    for (int i = 0; i < 8; i++) { m_i[i] = -1e30f; l_i[i] = 0.0f; }\n"
"\n"
"    int n_kv = (n_tok + FA_WM2_BKV - 1) / FA_WM2_BKV;\n"
"    /* 128 threads -> 16 rows x 8 threads/row, 10 ushorts each */\n"
"    int ld_row  = tid >> 3;       /* 0..15 */\n"
"    int ld_col0 = (tid & 7) * 10; /* 0,10,20,...,70 */\n"
"\n"
"    /* Prologue: load tile 0 into buffer 0 */\n"
"    {\n"
"        int kv = ld_row;\n"
"        size_t k_base = ((size_t)h * n_tok + kv) * head_dim;\n"
"        bool kv_ok = (kv < n_tok);\n"
"        for (int j = 0; j < 10; j++) {\n"
"            int d = ld_col0 + j;\n"
"            unsigned short k_v = 0, v_v = 0;\n"
"            if (kv_ok && d < head_dim) {\n"
"                k_v = K_t[k_base + d];\n"
"                v_v = V_t[k_base + d];\n"
"            }\n"
"            smK0[ld_row * FA_WM2_HD_PAD + d] = k_v;\n"
"            smV0[d * 16 + ld_row] = v_v;\n"
"        }\n"
"    }\n"
"    __syncthreads();\n"
"\n"
"    for (int t = 0; t < n_kv; t++) {\n"
"        unsigned short *smK_cur = (t & 1) ? smK1 : smK0;\n"
"        unsigned short *smV_cur = (t & 1) ? smV1 : smV0;\n"
"        unsigned short *smK_pre = (t & 1) ? smK0 : smK1;\n"
"        unsigned short *smV_pre = (t & 1) ? smV0 : smV1;\n"
"        int t_next = t + 1;\n"
"        bool prefetch = t_next < n_kv;\n"
"\n"
"        if (prefetch) {\n"
"            int kv_pre = t_next * FA_WM2_BKV + ld_row;\n"
"            size_t k_base_pre = ((size_t)h * n_tok + kv_pre) * head_dim;\n"
"            bool pre_ok = (kv_pre < n_tok);\n"
"            for (int j = 0; j < 10; j++) {\n"
"                int d = ld_col0 + j;\n"
"                unsigned short k_v = 0, v_v = 0;\n"
"                if (pre_ok && d < head_dim) {\n"
"                    k_v = K_t[k_base_pre + d];\n"
"                    v_v = V_t[k_base_pre + d];\n"
"                }\n"
"                smK_pre[ld_row * FA_WM2_HD_PAD + d] = k_v;\n"
"                smV_pre[d * 16 + ld_row] = v_v;\n"
"            }\n"
"        }\n"
"\n"
"        int kv0 = t * FA_WM2_BKV;\n"
"        float8 score = {0,0,0,0,0,0,0,0};\n"
"        for (int kb = 0; kb < FA_WM2_K_NB; kb++) {\n"
"            bf16x8 b_K;\n"
"            for (int i = 0; i < 8; i++) {\n"
"                int dpos = kb * 16 + half * 8 + i;\n"
"                b_K[i] = smK_cur[idx * FA_WM2_HD_PAD + dpos];\n"
"            }\n"
"            score = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(q_reg[kb], b_K, score);\n"
"        }\n"
"        bool col_valid = (kv0 + idx) < n_tok;\n"
"        for (int i = 0; i < 8; i++) score[i] = col_valid ? score[i] : -1e30f;\n"
"        float row_max[8];\n"
"        for (int i = 0; i < 8; i++) {\n"
"            float v = score[i];\n"
"            v = fmaxf(v, __shfl_xor(v, 1, 32));\n"
"            v = fmaxf(v, __shfl_xor(v, 2, 32));\n"
"            v = fmaxf(v, __shfl_xor(v, 4, 32));\n"
"            v = fmaxf(v, __shfl_xor(v, 8, 32));\n"
"            row_max[i] = v;\n"
"        }\n"
"        float alpha[8];\n"
"        for (int i = 0; i < 8; i++) {\n"
"            float new_max = fmaxf(m_i[i], row_max[i]);\n"
"            alpha[i] = __expf(m_i[i] - new_max);\n"
"            float ej = __expf(score[i] - new_max);\n"
"            float s = ej;\n"
"            s += __shfl_xor(s, 1, 32);\n"
"            s += __shfl_xor(s, 2, 32);\n"
"            s += __shfl_xor(s, 4, 32);\n"
"            s += __shfl_xor(s, 8, 32);\n"
"            l_i[i] = l_i[i] * alpha[i] + s;\n"
"            m_i[i] = new_max;\n"
"            score[i] = ej;\n"
"        }\n"
"        for (int kb = 0; kb < FA_WM2_K_NB; kb++)\n"
"            for (int i = 0; i < 8; i++) O_acc[kb][i] *= alpha[i];\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int m_row = half * 8 + i;\n"
"            int n_col = idx;\n"
"            smP_w[m_row * 16 + n_col] = f32_to_bf16_bits(score[i]);\n"
"        }\n"
"        bf16x8 ap;\n"
"        for (int i = 0; i < 8; i++) ap[i] = smP_w[idx * 16 + half * 8 + i];\n"
"        for (int kb = 0; kb < FA_WM2_K_NB; kb++) {\n"
"            bf16x8 bv;\n"
"            for (int i = 0; i < 8; i++) {\n"
"                int d_col = kb * 16 + idx;\n"
"                int kv_k  = half * 8 + i;\n"
"                bv[i] = smV_cur[d_col * 16 + kv_k];\n"
"            }\n"
"            O_acc[kb] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(ap, bv, O_acc[kb]);\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    for (int kb = 0; kb < FA_WM2_K_NB; kb++) {\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int m_row = half * 8 + i;\n"
"            int d = kb * 16 + idx;\n"
"            int row = q_base + m_row;\n"
"            if (row < n_tok && d < head_dim) {\n"
"                float v = O_acc[kb][i] / (l_i[i] > 0.0f ? l_i[i] : 1.0f);\n"
"                out[(size_t)row * dim + h * head_dim + d] = v;\n"
"            }\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"__global__ void flash_attn_wmma_f16_2w_pre(float *out, const float *qkv,\n"
"    const _Float16 *K_t, const _Float16 *V_t,\n"
"    int n_tok, int dim, int n_heads, int head_dim, float scale) {\n"
"    int h = blockIdx.x;\n"
"    int qb = blockIdx.y;\n"
"    int q0 = qb * FA_WM2_BQ;\n"
"    int tid = threadIdx.x;\n"
"    int wid = tid >> 5;\n"
"    int lid = tid & 31;\n"
"    int half = lid >> 4;\n"
"    int idx  = lid & 15;\n"
"    int dim3 = 3 * dim;\n"
"    extern __shared__ _Float16 smem_h[];\n"
"    _Float16 *smK0 = smem_h;\n"
"    _Float16 *smK1 = smK0 + 16 * FA_WM2_HD_PAD;\n"
"    _Float16 *smV0 = smK1 + 16 * FA_WM2_HD_PAD;\n"
"    _Float16 *smV1 = smV0 + FA_WM2_HD_PAD * 16;\n"
"    _Float16 *smP  = smV1 + FA_WM2_HD_PAD * 16;\n"
"    _Float16 *smP_w = smP + wid * 16 * 16;\n"
"    int q_base = q0 + wid * 16;\n"
"    f16x8 q_reg[FA_WM2_K_NB];\n"
"    for (int kb = 0; kb < FA_WM2_K_NB; kb++) {\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int d = kb * 16 + half * 8 + i;\n"
"            int row = q_base + idx;\n"
"            float v = 0.0f;\n"
"            if (row < n_tok && d < head_dim)\n"
"                v = qkv[(size_t)row * dim3 + h * head_dim + d];\n"
"            q_reg[kb][i] = (_Float16)(v * scale);\n"
"        }\n"
"    }\n"
"    float8 O_acc[FA_WM2_K_NB];\n"
"    for (int kb = 0; kb < FA_WM2_K_NB; kb++)\n"
"        for (int i = 0; i < 8; i++) O_acc[kb][i] = 0.0f;\n"
"    float m_i[8], l_i[8];\n"
"    for (int i = 0; i < 8; i++) { m_i[i] = -1e30f; l_i[i] = 0.0f; }\n"
"    int n_kv = (n_tok + FA_WM2_BKV - 1) / FA_WM2_BKV;\n"
"    int ld_row = tid >> 2;\n"
"    int ld_col0 = (tid & 3) * 20;\n"
"\n"
"    /* Prologue: load tile 0 into buffer 0 */\n"
"    {\n"
"        int kv = ld_row;\n"
"        size_t k_base = ((size_t)h * n_tok + kv) * head_dim;\n"
"        bool kv_ok = (kv < n_tok);\n"
"        for (int j = 0; j < 20; j++) {\n"
"            int d = ld_col0 + j;\n"
"            _Float16 k_v = (_Float16)0.0f, v_v = (_Float16)0.0f;\n"
"            if (kv_ok && d < head_dim) {\n"
"                k_v = K_t[k_base + d];\n"
"                v_v = V_t[k_base + d];\n"
"            }\n"
"            smK0[ld_row * FA_WM2_HD_PAD + d] = k_v;\n"
"            smV0[d * 16 + ld_row] = v_v;\n"
"        }\n"
"    }\n"
"    __syncthreads();\n"
"\n"
"    for (int t = 0; t < n_kv; t++) {\n"
"        _Float16 *smK_cur = (t & 1) ? smK1 : smK0;\n"
"        _Float16 *smV_cur = (t & 1) ? smV1 : smV0;\n"
"        _Float16 *smK_pre = (t & 1) ? smK0 : smK1;\n"
"        _Float16 *smV_pre = (t & 1) ? smV0 : smV1;\n"
"        int t_next = t + 1;\n"
"        bool prefetch = t_next < n_kv;\n"
"\n"
"        if (prefetch) {\n"
"            int kv_pre = t_next * FA_WM2_BKV + ld_row;\n"
"            size_t k_base_pre = ((size_t)h * n_tok + kv_pre) * head_dim;\n"
"            bool pre_ok = (kv_pre < n_tok);\n"
"            for (int j = 0; j < 20; j++) {\n"
"                int d = ld_col0 + j;\n"
"                _Float16 k_v = (_Float16)0.0f, v_v = (_Float16)0.0f;\n"
"                if (pre_ok && d < head_dim) {\n"
"                    k_v = K_t[k_base_pre + d];\n"
"                    v_v = V_t[k_base_pre + d];\n"
"                }\n"
"                smK_pre[ld_row * FA_WM2_HD_PAD + d] = k_v;\n"
"                smV_pre[d * 16 + ld_row] = v_v;\n"
"            }\n"
"        }\n"
"\n"
"        int kv0 = t * FA_WM2_BKV;\n"
"        float8 score = {0,0,0,0,0,0,0,0};\n"
"        for (int kb = 0; kb < FA_WM2_K_NB; kb++) {\n"
"            f16x8 b_K;\n"
"            for (int i = 0; i < 8; i++) {\n"
"                int dpos = kb * 16 + half * 8 + i;\n"
"                b_K[i] = smK_cur[idx * FA_WM2_HD_PAD + dpos];\n"
"            }\n"
"            score = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(q_reg[kb], b_K, score);\n"
"        }\n"
"        bool col_valid = (kv0 + idx) < n_tok;\n"
"        for (int i = 0; i < 8; i++) score[i] = col_valid ? score[i] : -1e30f;\n"
"        float row_max[8];\n"
"        for (int i = 0; i < 8; i++) {\n"
"            float v = score[i];\n"
"            v = fmaxf(v, __shfl_xor(v, 1, 32));\n"
"            v = fmaxf(v, __shfl_xor(v, 2, 32));\n"
"            v = fmaxf(v, __shfl_xor(v, 4, 32));\n"
"            v = fmaxf(v, __shfl_xor(v, 8, 32));\n"
"            row_max[i] = v;\n"
"        }\n"
"        float alpha[8];\n"
"        for (int i = 0; i < 8; i++) {\n"
"            float new_max = fmaxf(m_i[i], row_max[i]);\n"
"            alpha[i] = __expf(m_i[i] - new_max);\n"
"            float ej = __expf(score[i] - new_max);\n"
"            float s = ej;\n"
"            s += __shfl_xor(s, 1, 32);\n"
"            s += __shfl_xor(s, 2, 32);\n"
"            s += __shfl_xor(s, 4, 32);\n"
"            s += __shfl_xor(s, 8, 32);\n"
"            l_i[i] = l_i[i] * alpha[i] + s;\n"
"            m_i[i] = new_max;\n"
"            score[i] = ej;\n"
"        }\n"
"        for (int kb = 0; kb < FA_WM2_K_NB; kb++)\n"
"            for (int i = 0; i < 8; i++) O_acc[kb][i] *= alpha[i];\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int m_row = half * 8 + i;\n"
"            int n_col = idx;\n"
"            smP_w[m_row * 16 + n_col] = (_Float16)score[i];\n"
"        }\n"
"        f16x8 ap;\n"
"        for (int i = 0; i < 8; i++) ap[i] = smP_w[idx * 16 + half * 8 + i];\n"
"        for (int kb = 0; kb < FA_WM2_K_NB; kb++) {\n"
"            f16x8 bv;\n"
"            for (int i = 0; i < 8; i++) {\n"
"                int d_col = kb * 16 + idx;\n"
"                int kv_k  = half * 8 + i;\n"
"                bv[i] = smV_cur[d_col * 16 + kv_k];\n"
"            }\n"
"            O_acc[kb] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ap, bv, O_acc[kb]);\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    for (int kb = 0; kb < FA_WM2_K_NB; kb++) {\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int m_row = half * 8 + i;\n"
"            int d = kb * 16 + idx;\n"
"            int row = q_base + m_row;\n"
"            if (row < n_tok && d < head_dim) {\n"
"                float v = O_acc[kb][i] / (l_i[i] > 0.0f ? l_i[i] : 1.0f);\n"
"                out[(size_t)row * dim + h * head_dim + d] = v;\n"
"            }\n"
"        }\n"
"    }\n"
"}\n"
"#undef FA_WM2_HD_PAD\n"
"#undef FA_WM2_K_NB\n"
"#undef FA_WM2_BQ\n"
"#undef FA_WM2_BKV\n"
"\n"
"/* ---- patch_unfold_{bf16,f16}: im2col for patch embedding into BF16/F16 ---- */\n"
"/* Grid: (n_patches), Block: (256). Output [n_patches, ks=ps*ps*3].         */\n"
"__global__ void patch_unfold_bf16(unsigned short *out, const float *rgb,\n"
"                                   int gw, int ps, int ks, int img_w) {\n"
"    int patch = blockIdx.x;\n"
"    int tid = threadIdx.x;\n"
"    int py = patch / gw, px = patch % gw;\n"
"    int ps2 = ps * ps;\n"
"    for (int k = tid; k < ks; k += blockDim.x) {\n"
"        int c = k / ps2;\n"
"        int rest = k - c * ps2;\n"
"        int ky = rest / ps;\n"
"        int kx = rest - ky * ps;\n"
"        int iy = py * ps + ky;\n"
"        int ix = px * ps + kx;\n"
"        float v = rgb[(iy * img_w + ix) * 3 + c];\n"
"        out[(size_t)patch * ks + k] = f32_to_bf16_bits(v);\n"
"    }\n"
"}\n"
"\n"
"__global__ void patch_unfold_f16(half_raw *out, const float *rgb,\n"
"                                  int gw, int ps, int ks, int img_w) {\n"
"    int patch = blockIdx.x;\n"
"    int tid = threadIdx.x;\n"
"    int py = patch / gw, px = patch % gw;\n"
"    int ps2 = ps * ps;\n"
"    for (int k = tid; k < ks; k += blockDim.x) {\n"
"        int c = k / ps2;\n"
"        int rest = k - c * ps2;\n"
"        int ky = rest / ps;\n"
"        int kx = rest - ky * ps;\n"
"        int iy = py * ps + ky;\n"
"        int ix = px * ps + kx;\n"
"        float v = rgb[(iy * img_w + ix) * 3 + c];\n"
"        out[(size_t)patch * ks + k] = f32_to_f16_bits(v);\n"
"    }\n"
"}\n"
"\n"
"/* ---- patch_embed_dual_f32: Dual Conv2D patch extraction ---- */\n"
"/* Grid: (n_patches), Block: (256) */\n"
"__global__ void patch_embed_dual_f32(float *out, const float *rgb,\n"
"                                      const float *w0, const float *w1,\n"
"                                      const float *bias,\n"
"                                      int gw, int dim, int ps, int img_w) {\n"
"    int patch = blockIdx.x;\n"
"    int tid = threadIdx.x;\n"
"    int py = patch / gw, px = patch % gw;\n"
"    int ks = ps * ps * 3;\n"
"    for (int d = tid; d < dim; d += blockDim.x) {\n"
"        float sum = bias ? bias[d] : 0.0f;\n"
"        for (int c = 0; c < 3; c++) {\n"
"            for (int ky = 0; ky < ps; ky++) {\n"
"                for (int kx = 0; kx < ps; kx++) {\n"
"                    int iy = py * ps + ky;\n"
"                    int ix = px * ps + kx;\n"
"                    float pix = rgb[(iy * img_w + ix) * 3 + c];\n"
"                    int ki = c * ps * ps + ky * ps + kx;\n"
"                    sum += w0[d * ks + ki] * pix;\n"
"                    if (w1) sum += w1[d * ks + ki] * pix;\n"
"                }\n"
"            }\n"
"        }\n"
"        out[patch * dim + d] = sum;\n"
"    }\n"
"}\n"
"\n"
"/* ---- add_pos_embd: add position embeddings via indirection map ---- */\n"
"/* Grid: (n_patches), Block: (256) */\n"
"__global__ void add_pos_embd(float *hidden, const float *pos_emb,\n"
"                               const int *pos_map, int dim) {\n"
"    int p = blockIdx.x;\n"
"    int orig_p = pos_map[p];\n"
"    int tid = threadIdx.x;\n"
"    for (int d = tid; d < dim; d += blockDim.x)\n"
"        hidden[p * dim + d] += pos_emb[orig_p * dim + d];\n"
"}\n"
"\n"
"/* ---- add_pos_embd_direct: add pre-interpolated position embeddings ---- */\n"
"/* Grid: (n_patches), Block: (256) */\n"
"__global__ void add_pos_embd_direct(float *hidden, const float *pos_emb,\n"
"                                      int dim, int n) {\n"
"    int p = blockIdx.x;\n"
"    if (p >= n) return;\n"
"    for (int d = threadIdx.x; d < dim; d += blockDim.x)\n"
"        hidden[p * dim + d] += pos_emb[p * dim + d];\n"
"}\n"
"\n"
"/* ---- rope_vision_f32: M-RoPE on Q and K ---- */\n"
"/* Grid: (n_patches * n_heads), Block: (half_dim) */\n"
"__global__ void rope_vision_f32(float *qkv, const float *rope_cos,\n"
"                                  const float *rope_sin,\n"
"                                  int n_patches, int n_heads,\n"
"                                  int dim, int head_dim, int half) {\n"
"    int idx = blockIdx.x;\n"
"    int p = idx / n_heads;\n"
"    int h = idx % n_heads;\n"
"    int i = threadIdx.x;\n"
"    if (i >= half) return;\n"
"    float cos_t = rope_cos[p * head_dim + 2 * i];\n"
"    float sin_t = rope_sin[p * head_dim + 2 * i];\n"
"    /* Q */\n"
"    float *q = qkv + p * 3 * dim + h * head_dim;\n"
"    float q0 = q[i], q1 = q[i + half];\n"
"    q[i]        = q0 * cos_t - q1 * sin_t;\n"
"    q[i + half] = q0 * sin_t + q1 * cos_t;\n"
"    /* K */\n"
"    float *k = qkv + p * 3 * dim + dim + h * head_dim;\n"
"    float k0 = k[i], k1 = k[i + half];\n"
"    k[i]        = k0 * cos_t - k1 * sin_t;\n"
"    k[i + half] = k0 * sin_t + k1 * cos_t;\n"
"}\n"
"\n"
"/* ---- attn_full_f32: Full NxN self-attention per head ---- */\n"
"/* Grid: (n_heads), Block: (256) */\n"
"/* Uses shared memory for softmax reduction. */\n"
"__global__ void attn_full_f32(float *out, const float *qkv,\n"
"                                int n_patches, int dim, int n_heads,\n"
"                                int head_dim, float scale) {\n"
"    extern __shared__ float smem[];\n"
"    int h = blockIdx.x;\n"
"    if (h >= n_heads) return;\n"
"    int tid = threadIdx.x;\n"
"    int nt = blockDim.x;\n"
"    int dim3 = 3 * dim;\n"
"    /* Process queries sequentially */\n"
"    for (int qi = 0; qi < n_patches; qi++) {\n"
"        const float *q_h = qkv + qi * dim3 + h * head_dim;\n"
"        /* Compute scores QK^T for this query */\n"
"        for (int ki = tid; ki < n_patches; ki += nt) {\n"
"            const float *k_h = qkv + ki * dim3 + dim + h * head_dim;\n"
"            float score = 0.0f;\n"
"            for (int d = 0; d < head_dim; d++)\n"
"                score += q_h[d] * k_h[d];\n"
"            smem[ki] = score * scale;\n"
"        }\n"
"        __syncthreads();\n"
"        /* Softmax: find max */\n"
"        float local_max = -1e30f;\n"
"        for (int ki = tid; ki < n_patches; ki += nt)\n"
"            if (smem[ki] > local_max) local_max = smem[ki];\n"
"        /* Reduce max in shared memory */\n"
"        smem[n_patches + tid] = local_max;\n"
"        __syncthreads();\n"
"        for (int r = nt/2; r > 0; r >>= 1) {\n"
"            if (tid < r && smem[n_patches + tid + r] > smem[n_patches + tid])\n"
"                smem[n_patches + tid] = smem[n_patches + tid + r];\n"
"            __syncthreads();\n"
"        }\n"
"        float max_val = smem[n_patches];\n"
"        /* Exp and sum */\n"
"        float local_sum = 0.0f;\n"
"        for (int ki = tid; ki < n_patches; ki += nt) {\n"
"            smem[ki] = expf(smem[ki] - max_val);\n"
"            local_sum += smem[ki];\n"
"        }\n"
"        smem[n_patches + tid] = local_sum;\n"
"        __syncthreads();\n"
"        for (int r = nt/2; r > 0; r >>= 1) {\n"
"            if (tid < r) smem[n_patches + tid] += smem[n_patches + tid + r];\n"
"            __syncthreads();\n"
"        }\n"
"        float inv_sum = 1.0f / smem[n_patches];\n"
"        /* Normalize */\n"
"        for (int ki = tid; ki < n_patches; ki += nt)\n"
"            smem[ki] *= inv_sum;\n"
"        __syncthreads();\n"
"        /* Weighted sum of V */\n"
"        for (int d = tid; d < head_dim; d += nt) {\n"
"            float sum = 0.0f;\n"
"            for (int vi = 0; vi < n_patches; vi++) {\n"
"                sum += smem[vi] * qkv[vi * dim3 + 2 * dim + h * head_dim + d];\n"
"            }\n"
"            out[qi * dim + h * head_dim + d] = sum;\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"}\n"
"\n"
"/* ---- spatial_merge_f32: Gather 2x2 patches into merged tokens ---- */\n"
"/* Grid: (n_merged), Block: (256) */\n"
"__global__ void spatial_merge_f32(float *dst, const float *src,\n"
"                                    int gw, int sm, int dim) {\n"
"    int m = blockIdx.x;\n"
"    int mgw = gw / sm;\n"
"    int my = m / mgw, mx = m % mgw;\n"
"    int merged_dim = dim * sm * sm;\n"
"    int tid = threadIdx.x;\n"
"    for (int di = tid; di < merged_dim; di += blockDim.x) {\n"
"        int sub = di / dim;\n"
"        int d = di % dim;\n"
"        int sy = sub / sm, sx = sub % sm;\n"
"        int py = my * sm + sy;\n"
"        int px = mx * sm + sx;\n"
"        dst[m * merged_dim + di] = src[(py * gw + px) * dim + d];\n"
"    }\n"
"}\n"
"\n"
"/* ---- spatial_merge_contig_f32: Merge already grouped Qwen3VL tokens ---- */\n"
"/* Grid: (n_merged), Block: (256) */\n"
"__global__ void spatial_merge_contig_f32(float *dst, const float *src,\n"
"                                          int sm, int dim) {\n"
"    int m = blockIdx.x;\n"
"    int merged_dim = dim * sm * sm;\n"
"    int tid = threadIdx.x;\n"
"    for (int di = tid; di < merged_dim; di += blockDim.x) {\n"
"        int sub = di / dim;\n"
"        int d = di % dim;\n"
"        dst[(size_t)m * merged_dim + di] = src[((size_t)m * sm * sm + sub) * dim + d];\n"
"    }\n"
"}\n"
"\n"
"/* ---- qwen3vl_repack_f32: match llama.cpp Qwen3VL patch/pos layout ---- */\n"
"/* Grid: (n_patches), Block: (256), requires spatial_merge=2. */\n"
"__global__ void qwen3vl_repack_f32(float *dst, const float *src,\n"
"                                    int gw, int dim) {\n"
"    int p = blockIdx.x;\n"
"    int tid = threadIdx.x;\n"
"    int half_gw = gw / 2;\n"
"    for (int d = tid; d < dim; d += blockDim.x) {\n"
"        int f = d + (p & 1) * dim;\n"
"        int r = p >> 1;\n"
"        int dy = r & 1;\n"
"        int x2 = (r >> 1) % half_gw;\n"
"        int y2 = r / gw;\n"
"        int c = f % dim;\n"
"        int sx = f / dim;\n"
"        int x = 2 * x2 + sx;\n"
"        int y = 2 * y2 + dy;\n"
"        dst[(size_t)p * dim + d] = src[(size_t)(y * gw + x) * dim + c];\n"
"    }\n"
"}\n"
"\n"
"} /* extern C */\n"
;

/* ======================================================================== */
/* Runner struct                                                            */
/* ======================================================================== */

typedef struct {
    void *w_f32;   /* F32 weight [n_out, n_in] */
    void *w_f16;   /* F16 weight [n_out, n_in] */
    void *w_bf16;  /* BF16 weight [n_out, n_in] */
    void *bias;    /* F32 bias [n_out] (always F32) */
} gpu_weight;

typedef struct {
    gpu_weight attn_qkv;    /* [3*dim, dim] */
    gpu_weight attn_out;    /* [dim, dim] */
    gpu_weight ffn_up;      /* [ffn_dim, dim] */
    gpu_weight ffn_down;    /* [dim, ffn_dim] */
    void *ln1_w, *ln1_b;  /* F32 [dim] */
    void *ln2_w, *ln2_b;  /* F32 [dim] */
} gpu_vit_block;

typedef struct {
    gpu_weight fc1;   /* [merged_dim, merged_dim] */
    gpu_weight fc2;   /* [proj_dim, merged_dim] */
    void *norm_w, *norm_b;  /* F32 [merged_dim] */
} gpu_deepstack;

struct hip_vision_runner {
    hipDevice_t device;
    hipCtx_t context;
    hipStream_t stream;
    int verbose;
    int use_f16;

    hipModule_t module;
    /* Shared kernels */
    hipFunction_t fn_layernorm_f32;
    hipFunction_t fn_gemm_tiled_f16_f32;
    hipFunction_t fn_gelu_f32;
    hipFunction_t fn_add_f32;
    hipFunction_t fn_add_bias_f32;
    /* Vision-specific kernels */
    hipFunction_t fn_gemm_f32_f32;
    hipFunction_t fn_patch_embed_dual_f32;
    hipFunction_t fn_patch_unfold_bf16;
    hipFunction_t fn_patch_unfold_f16;
    hipFunction_t fn_add_pos_embd;
    hipFunction_t fn_add_pos_embd_direct;
    hipFunction_t fn_rope_vision_f32;
    hipFunction_t fn_attn_full_f32;
    hipFunction_t fn_flash_attn_tiled_f32;
    hipFunction_t fn_flash_attn_dyn_f32;
    hipFunction_t fn_flash_attn_wmma_bf16;
    hipFunction_t fn_flash_attn_wmma_f16;
    hipFunction_t fn_flash_attn_wmma_bf16_2w;
    hipFunction_t fn_flash_attn_wmma_f16_2w;
    hipFunction_t fn_flash_attn_wmma_bf16_2w_pre;
    hipFunction_t fn_flash_attn_wmma_f16_2w_pre;
    hipFunction_t fn_flash_attn_wmma_bf16_4w_pre;
    hipFunction_t fn_kv_transpose;
    hipFunction_t fn_kv_transpose_bf16;
    hipFunction_t fn_kv_transpose_f16;
    hipFunction_t fn_gemm_opt_f16_f32;
    hipFunction_t fn_gemm_opt_f16_f32_gelu;
    hipFunction_t fn_gemm_opt_f16_f32_res;
    hipFunction_t fn_gemm_opt_bf16_f32;
    hipFunction_t fn_pack_f16_from_f32;
    hipFunction_t fn_pack_bf16_from_f32;
    hipFunction_t fn_gemm_wmma_f16_f32;
    hipFunction_t fn_gemm_wmma_bf16_f32;
    hipFunction_t fn_spatial_merge_f32;
    hipFunction_t fn_spatial_merge_contig_f32;
    hipFunction_t fn_qwen3vl_repack_f32;

    /* Optional rocBLAS acceleration for large packed F16/BF16 GEMMs */
    void *rocblas_lib;
    vlm_rocblas_handle rocblas;
    vlm_rocblas_destroy_handle_fn rocblas_destroy_handle;
    vlm_rocblas_gemm_ex_fn rocblas_gemm_ex;
    int rocblas_enabled;

    /* Optional hipBLASLt fused epilogue path for projector GEMMs */
    void *hipblaslt_lib;
    vlm_hipblaslt_handle hipblaslt;
    vlm_hipblaslt_destroy_fn hipblaslt_destroy;
    vlm_hipblaslt_matmul_desc_create_fn hipblaslt_matmul_desc_create;
    vlm_hipblaslt_matmul_desc_destroy_fn hipblaslt_matmul_desc_destroy;
    vlm_hipblaslt_matmul_desc_set_attr_fn hipblaslt_matmul_desc_set_attr;
    vlm_hipblaslt_matrix_layout_create_fn hipblaslt_matrix_layout_create;
    vlm_hipblaslt_matrix_layout_destroy_fn hipblaslt_matrix_layout_destroy;
    vlm_hipblaslt_matmul_pref_create_fn hipblaslt_matmul_pref_create;
    vlm_hipblaslt_matmul_pref_destroy_fn hipblaslt_matmul_pref_destroy;
    vlm_hipblaslt_matmul_pref_set_attr_fn hipblaslt_matmul_pref_set_attr;
    vlm_hipblaslt_algo_get_heuristic_fn hipblaslt_algo_get_heuristic;
    vlm_hipblaslt_matmul_fn hipblaslt_matmul;
    int hipblaslt_enabled;
    vlm_hipblaslt_plan hipblaslt_mm0_gelu;
    /* Per-shape plan cache for ViT block GEMMs (qkv/attn_out/ffn_w0/ffn_w1).
     * Slots are find-or-create keyed by (dtype, m, n, k, epilogue). */
#define VLM_HIPBLASLT_VIT_MAX_PLANS 16
    vlm_hipblaslt_plan hipblaslt_vit_plans[VLM_HIPBLASLT_VIT_MAX_PLANS];
    int hipblaslt_vit_n_plans;

    /* Model hyperparams */
    int n_blocks;
    int dim;
    int n_heads;
    int head_dim;
    int ffn_dim;
    int patch_size;
    int image_size;
    int n_patches;
    int proj_dim;
    int spatial_merge;
    int n_merged;
    int use_qwen3vl_layout;
    float ln_eps;
    float image_mean[3];
    float image_std[3];

    /* Dynamic resolution support */
    int max_patches;           /* max patches for buffer allocation (0 = use n_patches) */
    int max_merged;            /* max merged tokens */
    int max_pixels;            /* max pixel count for RGB buffer */
    float *h_pos_embd;         /* CPU copy of original pos embedding [n_patches * dim] */
    void *d_pos_interp;        /* GPU buffer for interpolated pos embedding [max_patches * dim] */

    /* GPU weights: patch embeddings */
    void *d_patch_w0;          /* F32 [dim, ps*ps*3] */
    void *d_patch_w0_bf16;     /* BF16 [dim, ps*ps*3] (unused, kept for compat) */
    void *d_patch_w0_f16;      /* F16  [dim, ps*ps*3] for WMMA patch embed */
    void *d_patch_unfold_bf16; /* (unused) */
    void *d_patch_unfold_f16;  /* F16  [max_patches, ks] im2col scratch */
    void *d_patch_w1;     /* F32 [dim, ps*ps*3] (second conv, may be NULL) */
    void *d_patch_bias;   /* F32 [dim] */

    /* Position embedding */
    void *d_pos_embd;     /* F32 [n_patches, dim] */

    /* Blocks */
    gpu_vit_block *blocks;

    /* DeepStack */
    int n_deepstack;
    int *deepstack_indices;
    gpu_deepstack *deepstack;

    /* Post LN */
    void *d_post_ln_w, *d_post_ln_b;  /* F32 [dim] */

    /* MM projection */
    gpu_weight mm0;   /* [merged_dim, merged_dim] */
    gpu_weight mm2;   /* [proj_dim, merged_dim] */

    /* Scratch buffers (allocated on load) */
    void *d_hidden;     /* [max_patches * dim] */
    void *d_hidden2;    /* [max_patches * dim] */
    void *d_qkv;        /* [max_patches * 3 * dim] */
    void *d_kt;         /* [max_patches * dim] - transposed K for flash attn (F32) */
    void *d_vt;         /* [max_patches * dim] - transposed V for flash attn (F32) */
    void *d_kt_h;       /* [max_patches * dim] - transposed K BF16/F16 (half-size) */
    void *d_vt_h;       /* [max_patches * dim] - transposed V BF16/F16 (half-size) */
    void *d_attn_out;   /* [max_patches * dim] */
    void *d_ffn_buf;    /* [max_patches * ffn_dim] */
    void *d_ln_buf;     /* [max_patches * dim] */
    void *d_gemm_pack;  /* packed F16/BF16 GEMM activation scratch */
    void *d_merge_buf;  /* [n_merged * merged_dim] */
    void *d_mm_buf;     /* [n_merged * merged_dim] */
    void *d_mm_out;     /* [n_merged * proj_dim] */
    void *d_rgb;        /* [max_pixels * 3] */
    void *d_rope_cos;   /* [max_patches * head_dim] */
    void *d_rope_sin;   /* [max_patches * head_dim] */
    void *d_pos_map;    /* [max_patches] int */
    void *d_ds_feats;   /* deepstack feature accumulation */

    /* Host output */
    float *h_output;
    int loaded;
};

/* ======================================================================== */
/* HIPRTC compilation                                                       */
/* ======================================================================== */

static int vlm_compile_kernels(hip_vision_runner *r) {
    size_t len1 = strlen(hip_kernels_common_src);
    size_t len2 = strlen(hip_vlm_specific_kernels);
    char *full_src = (char *)malloc(len1 + len2 + 1);
    memcpy(full_src, hip_kernels_common_src, len1);
    memcpy(full_src + len1, hip_vlm_specific_kernels, len2 + 1);

    int ret = hip_compile_kernels(&r->module, r->device,
                                   full_src, "vlm_kernels",
                                   r->verbose, "hip_vlm");
    free(full_src);
    if (ret < 0) return -1;

    hipError_t err;
#define GET_FN(name) do { \
    err = hipModuleGetFunction(&r->fn_##name, r->module, #name); \
    if (err != hipSuccess) { fprintf(stderr, "hip_vlm: kernel '%s' not found\n", #name); return -1; } \
} while(0)

    /* Shared kernels */
    GET_FN(layernorm_f32);
    GET_FN(gemm_tiled_f16_f32);
    GET_FN(gelu_f32);
    GET_FN(add_f32);
    GET_FN(add_bias_f32);

    /* Vision-specific kernels */
    GET_FN(gemm_f32_f32);
    GET_FN(patch_embed_dual_f32);
    GET_FN(patch_unfold_bf16);
    GET_FN(patch_unfold_f16);
    GET_FN(add_pos_embd);
    GET_FN(add_pos_embd_direct);
    GET_FN(rope_vision_f32);
    GET_FN(attn_full_f32);
    GET_FN(flash_attn_tiled_f32);
    GET_FN(flash_attn_dyn_f32);
    GET_FN(flash_attn_wmma_bf16);
    GET_FN(flash_attn_wmma_f16);
    GET_FN(flash_attn_wmma_bf16_2w);
    GET_FN(flash_attn_wmma_f16_2w);
    GET_FN(flash_attn_wmma_bf16_2w_pre);
    GET_FN(flash_attn_wmma_f16_2w_pre);
    GET_FN(flash_attn_wmma_bf16_4w_pre);
    GET_FN(kv_transpose);
    GET_FN(kv_transpose_bf16);
    GET_FN(kv_transpose_f16);
    GET_FN(gemm_opt_f16_f32);
    GET_FN(gemm_opt_f16_f32_gelu);
    GET_FN(gemm_opt_f16_f32_res);
    GET_FN(gemm_opt_bf16_f32);
    GET_FN(pack_f16_from_f32);
    GET_FN(pack_bf16_from_f32);
    GET_FN(gemm_wmma_f16_f32);
    GET_FN(gemm_wmma_bf16_f32);
    GET_FN(spatial_merge_f32);
    GET_FN(spatial_merge_contig_f32);
    GET_FN(qwen3vl_repack_f32);

#undef GET_FN

    if (r->verbose >= 1)
        fprintf(stderr, "hip_vlm: %d kernels compiled\n", 24);
    return 0;
}

static int vlm_build_qwen3vl_token_perm(int gw, int gh, int sm, int *token_perm) {
    if (gw <= 0 || gh <= 0 || sm <= 0 || (gw % sm) != 0 || (gh % sm) != 0) return -1;

    int mgw = gw / sm;
    int mgh = gh / sm;
    for (int gy = 0; gy < mgh; gy++) {
        for (int gx = 0; gx < mgw; gx++) {
            int group = gy * mgw + gx;
            for (int sy = 0; sy < sm; sy++) {
                for (int sx = 0; sx < sm; sx++) {
                    int sub = sy * sm + sx;
                    int dst = group * sm * sm + sub;
                    int src = (gy * sm + sy) * gw + (gx * sm + sx);
                    token_perm[dst] = src;
                }
            }
        }
    }

    return 0;
}

static void vlm_try_init_rocblas(hip_vision_runner *r) {
    if (!r) return;

    void *lib = dlopen("librocblas.so", RTLD_NOW | RTLD_LOCAL);
    if (!lib) lib = dlopen("librocblas.so.5", RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        if (r->verbose >= 1) fprintf(stderr, "hip_vlm: rocBLAS not available, using HIPRTC WMMA GEMM\n");
        return;
    }

    vlm_rocblas_create_handle_fn create_handle =
        (vlm_rocblas_create_handle_fn)dlsym(lib, "rocblas_create_handle");
    vlm_rocblas_destroy_handle_fn destroy_handle =
        (vlm_rocblas_destroy_handle_fn)dlsym(lib, "rocblas_destroy_handle");
    vlm_rocblas_set_stream_fn set_stream =
        (vlm_rocblas_set_stream_fn)dlsym(lib, "rocblas_set_stream");
    vlm_rocblas_set_pointer_mode_fn set_pointer_mode =
        (vlm_rocblas_set_pointer_mode_fn)dlsym(lib, "rocblas_set_pointer_mode");
    vlm_rocblas_gemm_ex_fn gemm_ex =
        (vlm_rocblas_gemm_ex_fn)dlsym(lib, "rocblas_gemm_ex");

    if (!create_handle || !destroy_handle || !set_stream || !set_pointer_mode || !gemm_ex) {
        if (r->verbose >= 1) fprintf(stderr, "hip_vlm: rocBLAS symbols missing, using HIPRTC WMMA GEMM\n");
        dlclose(lib);
        return;
    }

    vlm_rocblas_handle handle = NULL;
    if (create_handle(&handle) != VLM_ROCBLAS_STATUS_SUCCESS || !handle) {
        if (r->verbose >= 1) fprintf(stderr, "hip_vlm: rocBLAS handle creation failed, using HIPRTC WMMA GEMM\n");
        dlclose(lib);
        return;
    }

    if (set_stream(handle, r->stream) != VLM_ROCBLAS_STATUS_SUCCESS ||
        set_pointer_mode(handle, VLM_ROCBLAS_POINTER_MODE_HOST) != VLM_ROCBLAS_STATUS_SUCCESS) {
        if (r->verbose >= 1) fprintf(stderr, "hip_vlm: rocBLAS setup failed, using HIPRTC WMMA GEMM\n");
        destroy_handle(handle);
        dlclose(lib);
        return;
    }

    r->rocblas_lib = lib;
    r->rocblas = handle;
    r->rocblas_destroy_handle = destroy_handle;
    r->rocblas_gemm_ex = gemm_ex;
    r->rocblas_enabled = 1;
    if (r->verbose >= 1) fprintf(stderr, "hip_vlm: rocBLAS large-GEMM path enabled\n");
}

static int vlm_hipblaslt_algo_index(const vlm_hipblaslt_matmul_algo *algo) {
    int v = 0;
    if (algo) memcpy(&v, algo->data, sizeof(v));
    return v;
}

static void vlm_hipblaslt_plan_destroy(hip_vision_runner *r, vlm_hipblaslt_plan *p) {
    if (!r || !p) return;
    if (p->pref && r->hipblaslt_matmul_pref_destroy) r->hipblaslt_matmul_pref_destroy(p->pref);
    if (p->d && r->hipblaslt_matrix_layout_destroy) r->hipblaslt_matrix_layout_destroy(p->d);
    if (p->c && r->hipblaslt_matrix_layout_destroy) r->hipblaslt_matrix_layout_destroy(p->c);
    if (p->b && r->hipblaslt_matrix_layout_destroy) r->hipblaslt_matrix_layout_destroy(p->b);
    if (p->a && r->hipblaslt_matrix_layout_destroy) r->hipblaslt_matrix_layout_destroy(p->a);
    if (p->matmul && r->hipblaslt_matmul_desc_destroy) r->hipblaslt_matmul_desc_destroy(p->matmul);
    memset(p, 0, sizeof(*p));
}

static void vlm_try_init_hipblaslt(hip_vision_runner *r) {
    if (!r) return;

    void *lib = dlopen("libhipblaslt.so", RTLD_NOW | RTLD_LOCAL);
    if (!lib) lib = dlopen("libhipblaslt.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        if (r->verbose >= 1) fprintf(stderr, "hip_vlm: hipBLASLt not available, using rocBLAS/HIPRTC GEMM\n");
        return;
    }

    vlm_hipblaslt_create_fn create =
        (vlm_hipblaslt_create_fn)dlsym(lib, "hipblasLtCreate");
    vlm_hipblaslt_destroy_fn destroy =
        (vlm_hipblaslt_destroy_fn)dlsym(lib, "hipblasLtDestroy");
    vlm_hipblaslt_matmul_desc_create_fn matmul_desc_create =
        (vlm_hipblaslt_matmul_desc_create_fn)dlsym(lib, "hipblasLtMatmulDescCreate");
    vlm_hipblaslt_matmul_desc_destroy_fn matmul_desc_destroy =
        (vlm_hipblaslt_matmul_desc_destroy_fn)dlsym(lib, "hipblasLtMatmulDescDestroy");
    vlm_hipblaslt_matmul_desc_set_attr_fn matmul_desc_set_attr =
        (vlm_hipblaslt_matmul_desc_set_attr_fn)dlsym(lib, "hipblasLtMatmulDescSetAttribute");
    vlm_hipblaslt_matrix_layout_create_fn matrix_layout_create =
        (vlm_hipblaslt_matrix_layout_create_fn)dlsym(lib, "hipblasLtMatrixLayoutCreate");
    vlm_hipblaslt_matrix_layout_destroy_fn matrix_layout_destroy =
        (vlm_hipblaslt_matrix_layout_destroy_fn)dlsym(lib, "hipblasLtMatrixLayoutDestroy");
    vlm_hipblaslt_matmul_pref_create_fn matmul_pref_create =
        (vlm_hipblaslt_matmul_pref_create_fn)dlsym(lib, "hipblasLtMatmulPreferenceCreate");
    vlm_hipblaslt_matmul_pref_destroy_fn matmul_pref_destroy =
        (vlm_hipblaslt_matmul_pref_destroy_fn)dlsym(lib, "hipblasLtMatmulPreferenceDestroy");
    vlm_hipblaslt_matmul_pref_set_attr_fn matmul_pref_set_attr =
        (vlm_hipblaslt_matmul_pref_set_attr_fn)dlsym(lib, "hipblasLtMatmulPreferenceSetAttribute");
    vlm_hipblaslt_algo_get_heuristic_fn algo_get_heuristic =
        (vlm_hipblaslt_algo_get_heuristic_fn)dlsym(lib, "hipblasLtMatmulAlgoGetHeuristic");
    vlm_hipblaslt_matmul_fn matmul =
        (vlm_hipblaslt_matmul_fn)dlsym(lib, "hipblasLtMatmul");

    if (!create || !destroy || !matmul_desc_create || !matmul_desc_destroy ||
        !matmul_desc_set_attr || !matrix_layout_create || !matrix_layout_destroy ||
        !matmul_pref_create || !matmul_pref_destroy || !matmul_pref_set_attr ||
        !algo_get_heuristic || !matmul) {
        if (r->verbose >= 1) fprintf(stderr, "hip_vlm: hipBLASLt symbols missing, using rocBLAS/HIPRTC GEMM\n");
        dlclose(lib);
        return;
    }

    vlm_hipblaslt_handle handle = NULL;
    if (create(&handle) != VLM_HIPBLASLT_STATUS_SUCCESS || !handle) {
        if (r->verbose >= 1) fprintf(stderr, "hip_vlm: hipBLASLt handle creation failed, using rocBLAS/HIPRTC GEMM\n");
        dlclose(lib);
        return;
    }

    r->hipblaslt_lib = lib;
    r->hipblaslt = handle;
    r->hipblaslt_destroy = destroy;
    r->hipblaslt_matmul_desc_create = matmul_desc_create;
    r->hipblaslt_matmul_desc_destroy = matmul_desc_destroy;
    r->hipblaslt_matmul_desc_set_attr = matmul_desc_set_attr;
    r->hipblaslt_matrix_layout_create = matrix_layout_create;
    r->hipblaslt_matrix_layout_destroy = matrix_layout_destroy;
    r->hipblaslt_matmul_pref_create = matmul_pref_create;
    r->hipblaslt_matmul_pref_destroy = matmul_pref_destroy;
    r->hipblaslt_matmul_pref_set_attr = matmul_pref_set_attr;
    r->hipblaslt_algo_get_heuristic = algo_get_heuristic;
    r->hipblaslt_matmul = matmul;
    r->hipblaslt_enabled = 1;
    if (r->verbose >= 1) fprintf(stderr, "hip_vlm: hipBLASLt fused projector path enabled\n");
}

static int vlm_hipblaslt_plan_init(hip_vision_runner *r, vlm_hipblaslt_plan *p,
                                   int dtype, int m, int n, int k, int epilogue,
                                   const void *bias) {
    if (!r || !p || !r->hipblaslt_enabled) return 0;
    if (p->valid && p->dtype == dtype && p->m == m && p->n == n &&
        p->k == k && p->epilogue == epilogue) {
        return 1;
    }

    vlm_hipblaslt_plan_destroy(r, p);

    int st = r->hipblaslt_matmul_desc_create(&p->matmul,
                                             VLM_HIPBLASLT_COMPUTE_32F,
                                             VLM_HIPBLASLT_DATATYPE_F32);
    if (st != VLM_HIPBLASLT_STATUS_SUCCESS) goto fail;

    int trans_a = VLM_HIPBLASLT_OP_TRANSPOSE;
    int trans_b = VLM_HIPBLASLT_OP_NONE;
    int bias_type = VLM_HIPBLASLT_DATATYPE_F32;
    if (r->hipblaslt_matmul_desc_set_attr(p->matmul, VLM_HIPBLASLT_MATMUL_DESC_TRANSA,
                                          &trans_a, sizeof(trans_a)) != VLM_HIPBLASLT_STATUS_SUCCESS) goto fail;
    if (r->hipblaslt_matmul_desc_set_attr(p->matmul, VLM_HIPBLASLT_MATMUL_DESC_TRANSB,
                                          &trans_b, sizeof(trans_b)) != VLM_HIPBLASLT_STATUS_SUCCESS) goto fail;
    if (r->hipblaslt_matmul_desc_set_attr(p->matmul, VLM_HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                          &epilogue, sizeof(epilogue)) != VLM_HIPBLASLT_STATUS_SUCCESS) goto fail;
    if (bias) {
        if (r->hipblaslt_matmul_desc_set_attr(p->matmul, VLM_HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                              &bias, sizeof(bias)) != VLM_HIPBLASLT_STATUS_SUCCESS) goto fail;
        if (r->hipblaslt_matmul_desc_set_attr(p->matmul, VLM_HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                              &bias_type, sizeof(bias_type)) != VLM_HIPBLASLT_STATUS_SUCCESS) goto fail;
    }

    if (r->hipblaslt_matrix_layout_create(&p->a, dtype, (uint64_t)k, (uint64_t)n, k) != VLM_HIPBLASLT_STATUS_SUCCESS) goto fail;
    if (r->hipblaslt_matrix_layout_create(&p->b, dtype, (uint64_t)k, (uint64_t)m, k) != VLM_HIPBLASLT_STATUS_SUCCESS) goto fail;
    if (r->hipblaslt_matrix_layout_create(&p->c, VLM_HIPBLASLT_DATATYPE_F32, (uint64_t)n, (uint64_t)m, n) != VLM_HIPBLASLT_STATUS_SUCCESS) goto fail;
    if (r->hipblaslt_matrix_layout_create(&p->d, VLM_HIPBLASLT_DATATYPE_F32, (uint64_t)n, (uint64_t)m, n) != VLM_HIPBLASLT_STATUS_SUCCESS) goto fail;
    if (r->hipblaslt_matmul_pref_create(&p->pref) != VLM_HIPBLASLT_STATUS_SUCCESS) goto fail;

    {
        uint64_t max_workspace = 0;
        if (r->hipblaslt_matmul_pref_set_attr(p->pref, VLM_HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &max_workspace, sizeof(max_workspace)) != VLM_HIPBLASLT_STATUS_SUCCESS) goto fail;
    }

    {
        const int requested = 64;
        vlm_hipblaslt_heuristic_result results[64];
        int returned = 0;
        memset(results, 0, sizeof(results));
        st = r->hipblaslt_algo_get_heuristic(r->hipblaslt, p->matmul, p->a, p->b, p->c, p->d,
                                             p->pref, requested, results, &returned);
        if (st != VLM_HIPBLASLT_STATUS_SUCCESS || returned <= 0) goto fail;

        int target_algo = (dtype == VLM_HIPBLASLT_DATATYPE_BF16) ? 73698 : 88308;
        int chosen = -1;
        for (int i = 0; i < returned; i++) {
            if (results[i].state != VLM_HIPBLASLT_STATUS_SUCCESS || results[i].workspace_size != 0) continue;
            if (vlm_hipblaslt_algo_index(&results[i].algo) == target_algo) {
                chosen = i;
                break;
            }
        }
        if (chosen < 0) {
            for (int i = 0; i < returned; i++) {
                if (results[i].state == VLM_HIPBLASLT_STATUS_SUCCESS && results[i].workspace_size == 0) {
                    chosen = i;
                    break;
                }
            }
        }
        if (chosen < 0) goto fail;

        p->algo = results[chosen].algo;
        p->workspace_size = results[chosen].workspace_size;
        p->algo_index = vlm_hipblaslt_algo_index(&p->algo);
    }

    p->valid = 1;
    p->dtype = dtype;
    p->m = m;
    p->n = n;
    p->k = k;
    p->epilogue = epilogue;
    if (r->verbose >= 1) {
        fprintf(stderr, "hip_vlm: hipBLASLt plan M=%d N=%d K=%d dtype=%d epilogue=%d algo=%d ws=%zu\n",
                m, n, k, dtype, epilogue, p->algo_index, p->workspace_size);
    }
    return 1;

fail:
    if (r->verbose >= 1) {
        fprintf(stderr, "hip_vlm: hipBLASLt plan failed for M=%d N=%d K=%d dtype=%d epilogue=%d, using rocBLAS/HIPRTC GEMM\n",
                m, n, k, dtype, epilogue);
    }
    vlm_hipblaslt_plan_destroy(r, p);
    return 0;
}

/* ======================================================================== */
/* Weight upload helpers                                                    */
/* ======================================================================== */

/* Helper to find a tensor in GGUF by name */
static int vlm_find_tensor(const gguf_context *g, const char *name) {
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        if (strcmp(g->tensors[i].name.str, name) == 0) return (int)i;
    }
    return -1;
}

int hip_vision_infer_precision(const gguf_context *g) {
    if (!g) return VLM_PREC_F32;

    uint64_t f16_elems = 0;
    uint64_t bf16_elems = 0;
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        const gguf_tensor_info *ti = &g->tensors[i];
        const char *name = ti->name.str ? ti->name.str : "";
        if (!strstr(name, ".weight") && strcmp(name, "mm.0.weight") != 0 && strcmp(name, "mm.2.weight") != 0) {
            continue;
        }
        uint64_t n_elem = 1;
        for (uint32_t d = 0; d < ti->n_dims; d++) n_elem *= ti->dims[d];
        if (ti->type == GGML_TYPE_F16) {
            f16_elems += n_elem;
        } else if (ti->type == GGML_TYPE_BF16) {
            bf16_elems += n_elem;
        }
    }

    if (bf16_elems > f16_elems && bf16_elems > 0) return VLM_PREC_BF16;
    if (f16_elems > 0) return VLM_PREC_F16;
    return VLM_PREC_F32;
}

/* Helper struct for tensor info */
typedef struct {
    const void *data;
    int type;
    int n_cols;
    int n_rows;
    int n_elem;
} vlm_tensor_info;

static vlm_tensor_info vlm_get_tensor(const gguf_context *g, const char *name, int req) {
    vlm_tensor_info t = {0};
    int idx = vlm_find_tensor(g, name);
    if (idx < 0) {
        if (req) fprintf(stderr, "hip_vlm: missing tensor '%s'\n", name);
        return t;
    }
    t.data = gguf_tensor_data(g, idx);
    t.type = (int)g->tensors[idx].type;
    t.n_cols = (int)g->tensors[idx].dims[0];
    t.n_rows = (g->tensors[idx].n_dims >= 2) ? (int)g->tensors[idx].dims[1] : 1;
    /* Compute total elements as product of all dimensions */
    t.n_elem = 1;
    for (int d = 0; d < (int)g->tensors[idx].n_dims; d++)
        t.n_elem *= (int)g->tensors[idx].dims[d];
    return t;
}

static int vlm_get_int(const gguf_context *g, const char *key, int def) {
    int idx = gguf_find_key(g, key);
    if (idx < 0) return def;
    if (g->kv[idx].type == GGUF_TYPE_UINT32) return (int)g->kv[idx].value.u32;
    if (g->kv[idx].type == GGUF_TYPE_INT32) return g->kv[idx].value.i32;
    return def;
}

static const char *vlm_get_string(const gguf_context *g, const char *key) {
    int idx = gguf_find_key(g, key);
    if (idx < 0) return NULL;
    if (g->kv[idx].type != GGUF_TYPE_STRING) return NULL;
    return g->kv[idx].value.str.str;
}

static float vlm_get_float(const gguf_context *g, const char *key, float def) {
    int idx = gguf_find_key(g, key);
    if (idx < 0) return def;
    if (g->kv[idx].type == GGUF_TYPE_FLOAT32) return g->kv[idx].value.f32;
    return def;
}

/* Dequantize full tensor to F32 and upload */
static void *vlm_upload_f32(const vlm_tensor_info *t) {
    if (!t->data) return NULL;
    int n = t->n_elem;
    float *buf = (float *)malloc((size_t)n * sizeof(float));
    if (t->type == GGML_TYPE_F32) {
        memcpy(buf, t->data, (size_t)n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const uint16_t *src = (const uint16_t *)t->data;
        for (int i = 0; i < n; i++) buf[i] = ggml_fp16_to_fp32(src[i]);
    } else {
        /* Dequantize row by row */
        size_t rb = dequant_row_size(t->type, t->n_cols);
        for (int row = 0; row < t->n_rows; row++) {
            const void *row_data = (const uint8_t *)t->data + row * rb;
            dequant_row(t->type, row_data, buf + row * t->n_cols, t->n_cols);
        }
    }
    void *d = NULL;
    if (hipMalloc(&d, (size_t)n * sizeof(float)) != hipSuccess) { free(buf); return NULL; }
    hipMemcpy(d, buf, (size_t)n * sizeof(float), hipMemcpyHostToDevice);
    free(buf);
    return d;
}

/* Upload tensor as F16 (converting from F32 if needed) */
static void *vlm_upload_f16(const vlm_tensor_info *t) {
    if (!t->data) return NULL;
    int n = t->n_elem;
    if (t->type == GGML_TYPE_F16) {
        /* Direct copy */
        return hip_upload_raw(t->data, (size_t)n * 2);
    }
    /* Convert F32 -> F16 */
    float *f32_buf = NULL;
    if (t->type == GGML_TYPE_F32) {
        f32_buf = (float *)t->data;
    } else {
        /* Dequant to F32 first */
        f32_buf = (float *)malloc((size_t)n * sizeof(float));
        size_t rb = dequant_row_size(t->type, t->n_cols);
        for (int row = 0; row < t->n_rows; row++) {
            const void *row_data = (const uint8_t *)t->data + row * rb;
            dequant_row(t->type, row_data, f32_buf + row * t->n_cols, t->n_cols);
        }
    }
    uint16_t *h16 = (uint16_t *)malloc((size_t)n * 2);
    for (int i = 0; i < n; i++) h16[i] = hip_f32_to_f16(f32_buf[i]);
    if (f32_buf != (float *)t->data) free(f32_buf);
    void *d = NULL;
    if (hipMalloc(&d, (size_t)n * 2) != hipSuccess) { free(h16); return NULL; }
    hipMemcpy(d, h16, (size_t)n * 2, hipMemcpyHostToDevice);
    free(h16);
    return d;
}

static uint16_t vlm_f32_to_bf16(float f) {
    union { float f; uint32_t u; } v;
    v.f = f;
    uint32_t lsb = (v.u >> 16) & 1u;
    return (uint16_t)((v.u + 0x7fffu + lsb) >> 16);
}

/* Upload tensor as BF16 (converting from F32/dequantized F32 if needed) */
static void *vlm_upload_bf16(const vlm_tensor_info *t) {
    if (!t->data) return NULL;
    int n = t->n_elem;
    if (t->type == GGML_TYPE_BF16) {
        return hip_upload_raw(t->data, (size_t)n * 2);
    }
    float *f32_buf = NULL;
    if (t->type == GGML_TYPE_F32) {
        f32_buf = (float *)t->data;
    } else {
        f32_buf = (float *)malloc((size_t)n * sizeof(float));
        size_t rb = dequant_row_size(t->type, t->n_cols);
        for (int row = 0; row < t->n_rows; row++) {
            const void *row_data = (const uint8_t *)t->data + row * rb;
            dequant_row(t->type, row_data, f32_buf + row * t->n_cols, t->n_cols);
        }
    }
    uint16_t *h16 = (uint16_t *)malloc((size_t)n * 2);
    for (int i = 0; i < n; i++) h16[i] = vlm_f32_to_bf16(f32_buf[i]);
    if (f32_buf != (float *)t->data) free(f32_buf);
    void *d = NULL;
    if (hipMalloc(&d, (size_t)n * 2) != hipSuccess) { free(h16); return NULL; }
    hipMemcpy(d, h16, (size_t)n * 2, hipMemcpyHostToDevice);
    free(h16);
    return d;
}

/* Upload a weight matrix in the selected precision mode. */
static gpu_weight vlm_upload_weight(const vlm_tensor_info *w, const vlm_tensor_info *b, int precision) {
    gpu_weight gw = {0};
    if (precision == VLM_PREC_F16) {
        gw.w_f16 = vlm_upload_f16(w);
    } else if (precision == VLM_PREC_BF16) {
        gw.w_bf16 = vlm_upload_bf16(w);
    } else {
        gw.w_f32 = vlm_upload_f32(w);
    }
    if (b && b->data) {
        gw.bias = vlm_upload_f32(b);
    }
    return gw;
}

/* ======================================================================== */
/* Public API: init                                                         */
/* ======================================================================== */

hip_vision_runner *hip_vision_init(int device_id, int verbose, int use_f16) {
    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) {
        fprintf(stderr, "hip_vlm: rocew init failed (no HIP/HIPRTC?)\n");
        return NULL;
    }
    hipError_t err = hipInit(0);
    if (err != hipSuccess) {
        fprintf(stderr, "hip_vlm: hipInit failed\n");
        return NULL;
    }

    hip_vision_runner *r = (hip_vision_runner *)calloc(1, sizeof(hip_vision_runner));
    r->verbose = verbose;
    if (use_f16 != VLM_PREC_F16 && use_f16 != VLM_PREC_BF16) use_f16 = VLM_PREC_F32;
    r->use_f16 = use_f16;
    r->device = device_id;

    HIP_CHECK_NULL(hipSetDevice(device_id));
    HIP_CHECK_NULL(hipCtxCreate(&r->context, 0, device_id));
    HIP_CHECK_NULL(hipStreamCreate(&r->stream));
    vlm_try_init_rocblas(r);
    vlm_try_init_hipblaslt(r);

    if (vlm_compile_kernels(r) != 0) {
        fprintf(stderr, "hip_vlm: kernel compilation failed\n");
        free(r);
        return NULL;
    }

    return r;
}

void hip_vision_set_max_pixels(hip_vision_runner *r, int max_pixels) {
    if (r) r->max_pixels = max_pixels;
}

/* ======================================================================== */
/* Public API: load_weights                                                 */
/* ======================================================================== */

int hip_vision_load_weights(hip_vision_runner *r, gguf_context *g) {
    if (!r || !g) return -1;
    const char *proj_type = vlm_get_string(g, "clip.projector_type");
    if (!proj_type) proj_type = vlm_get_string(g, "clip.vision.projector_type");

    /* Read hyperparameters */
    r->n_blocks    = vlm_get_int(g, "clip.vision.block_count", 24);
    r->dim         = vlm_get_int(g, "clip.vision.embedding_length", 1024);
    r->n_heads     = vlm_get_int(g, "clip.vision.attention.head_count", 16);
    r->ffn_dim     = vlm_get_int(g, "clip.vision.feed_forward_length", 4096);
    r->patch_size  = vlm_get_int(g, "clip.vision.patch_size", 16);
    r->image_size  = vlm_get_int(g, "clip.vision.image_size", 768);
    r->proj_dim    = vlm_get_int(g, "clip.vision.projection_dim", 2048);
    r->spatial_merge = vlm_get_int(g, "clip.vision.spatial_merge_size", 2);
    r->use_qwen3vl_layout = (proj_type && strstr(proj_type, "qwen3vl") != NULL) ? 1 : 0;
    r->ln_eps      = vlm_get_float(g, "clip.vision.attention.layer_norm_epsilon", 1e-6f);
    r->head_dim    = r->dim / r->n_heads;

    int ps = r->patch_size;
    int gs = r->image_size / ps;
    r->n_patches = gs * gs;
    r->n_merged  = r->n_patches / (r->spatial_merge * r->spatial_merge);

    /* Compute max buffer sizes for dynamic resolution */
    {
        int mp = r->max_pixels > 0 ? r->max_pixels / (ps * ps) : r->n_patches;
        if (mp < r->n_patches) mp = r->n_patches;
        r->max_patches = mp;
        r->max_merged = mp / (r->spatial_merge * r->spatial_merge);
    }

    /* Image mean/std */
    int idx = gguf_find_key(g, "clip.vision.image_mean");
    if (idx >= 0) {
        float *d = (float *)g->kv[idx].value.arr.data;
        r->image_mean[0] = d[0]; r->image_mean[1] = d[1]; r->image_mean[2] = d[2];
    }
    idx = gguf_find_key(g, "clip.vision.image_std");
    if (idx >= 0) {
        float *d = (float *)g->kv[idx].value.arr.data;
        r->image_std[0] = d[0]; r->image_std[1] = d[1]; r->image_std[2] = d[2];
    }

    fprintf(stderr, "hip_vlm: dim=%d heads=%d blocks=%d ffn=%d patch=%d image=%d patches=%d merged=%d proj=%d precision=%s max_patches=%d proj_type=%s qwen3vl_layout=%d\n",
            r->dim, r->n_heads, r->n_blocks, r->ffn_dim,
            r->patch_size, r->image_size, r->n_patches, r->n_merged, r->proj_dim, vlm_prec_name(r->use_f16),
            r->max_patches, proj_type ? proj_type : "(unknown)", r->use_qwen3vl_layout);

    int dim = r->dim;
    int mp = r->max_patches;
    int max_merged = r->max_merged;
    int sm = r->spatial_merge;
    int merged_dim = dim * sm * sm;

    /* Patch embeddings (always F32 -- small, applied once).
     * Dual conv2d: out = (W0 + W1) @ pix. Fold W0+=W1 once at init so the
     * runtime kernel issues 1 mul/load instead of 2 (halves inner-loop work). */
    vlm_tensor_info t_pw0 = vlm_get_tensor(g, "v.patch_embd.weight", 1);
    vlm_tensor_info t_pw1 = vlm_get_tensor(g, "v.patch_embd.weight.1", 0);
    vlm_tensor_info t_pb  = vlm_get_tensor(g, "v.patch_embd.bias", 0);
    if (t_pw1.data) {
        /* Dequant both, sum on host, upload combined. */
        int n = t_pw0.n_elem;
        float *w0 = (float *)malloc((size_t)n * sizeof(float));
        float *w1 = (float *)malloc((size_t)n * sizeof(float));
        if (t_pw0.type == GGML_TYPE_F32) memcpy(w0, t_pw0.data, (size_t)n * sizeof(float));
        else if (t_pw0.type == GGML_TYPE_F16) {
            const uint16_t *src = (const uint16_t *)t_pw0.data;
            for (int i = 0; i < n; i++) w0[i] = ggml_fp16_to_fp32(src[i]);
        } else {
            size_t rb = dequant_row_size(t_pw0.type, t_pw0.n_cols);
            for (int r0 = 0; r0 < t_pw0.n_rows; r0++)
                dequant_row(t_pw0.type, (const uint8_t *)t_pw0.data + r0 * rb,
                            w0 + r0 * t_pw0.n_cols, t_pw0.n_cols);
        }
        if (t_pw1.type == GGML_TYPE_F32) memcpy(w1, t_pw1.data, (size_t)n * sizeof(float));
        else if (t_pw1.type == GGML_TYPE_F16) {
            const uint16_t *src = (const uint16_t *)t_pw1.data;
            for (int i = 0; i < n; i++) w1[i] = ggml_fp16_to_fp32(src[i]);
        } else {
            size_t rb = dequant_row_size(t_pw1.type, t_pw1.n_cols);
            for (int r1 = 0; r1 < t_pw1.n_rows; r1++)
                dequant_row(t_pw1.type, (const uint8_t *)t_pw1.data + r1 * rb,
                            w1 + r1 * t_pw1.n_cols, t_pw1.n_cols);
        }
        for (int i = 0; i < n; i++) w0[i] += w1[i];
        if (hipMalloc(&r->d_patch_w0, (size_t)n * sizeof(float)) != hipSuccess) {
            free(w0); free(w1); return NULL;
        }
        hipMemcpy(r->d_patch_w0, w0, (size_t)n * sizeof(float), hipMemcpyHostToDevice);
        free(w0); free(w1);
        r->d_patch_w1 = NULL;
        fprintf(stderr, "hip_vlm: loaded dual conv2d patch embeddings (W0+W1 folded)\n");
    } else {
        r->d_patch_w0 = vlm_upload_f32(&t_pw0);
        r->d_patch_w1 = NULL;
    }
    r->d_patch_bias = vlm_upload_f32(&t_pb);

    /* Pre-pack patch weight to F16 for WMMA-GEMM patch embedding.
     * F16 (10-bit mantissa) is needed here for precision: BF16 (7-bit) loses
     * too much on the small normalized RGB inputs and fails the accuracy gate.
     * Layout: [dim, ks=ps*ps*3] (matches gemm_wmma_f16_f32 W). */
    {
        int n = t_pw0.n_elem;
        if (hipMalloc(&r->d_patch_w0_f16, (size_t)n * sizeof(unsigned short)) == hipSuccess) {
            int n_int = (int)n;
            void *args[] = { &r->d_patch_w0_f16, &r->d_patch_w0, &n_int };
            int n4 = (n + 3) / 4;
            int grid = (n4 + 255) / 256;
            hipModuleLaunchKernel(r->fn_pack_f16_from_f32,
                           grid, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
            hipStreamSynchronize(r->stream);
        }
        int ks = r->patch_size * r->patch_size * 3;
        size_t unfold_sz = (size_t)r->max_patches * ks * sizeof(unsigned short);
        if (hipMalloc(&r->d_patch_unfold_f16, unfold_sz) != hipSuccess) {
            r->d_patch_unfold_f16 = NULL;
        }
    }

    /* Position embedding (always F32) -- keep CPU copy for interpolation */
    vlm_tensor_info t_pos = vlm_get_tensor(g, "v.position_embd.weight", 1);
    r->d_pos_embd = vlm_upload_f32(&t_pos);
    r->h_pos_embd = (float *)malloc(t_pos.n_elem * sizeof(float));
    if (t_pos.type == GGML_TYPE_F32) {
        memcpy(r->h_pos_embd, t_pos.data, t_pos.n_elem * sizeof(float));
    } else {
        dequant_row(t_pos.type, t_pos.data, r->h_pos_embd, t_pos.n_elem);
    }

    /* Blocks */
    r->blocks = (gpu_vit_block *)calloc(r->n_blocks, sizeof(gpu_vit_block));
    for (int l = 0; l < r->n_blocks; l++) {
        char name[128];
        gpu_vit_block *blk = &r->blocks[l];

        /* QKV */
        snprintf(name, sizeof(name), "v.blk.%d.attn_qkv.weight", l);
        vlm_tensor_info tw = vlm_get_tensor(g, name, 1);
        snprintf(name, sizeof(name), "v.blk.%d.attn_qkv.bias", l);
        vlm_tensor_info tb = vlm_get_tensor(g, name, 1);
        blk->attn_qkv = vlm_upload_weight(&tw, &tb, r->use_f16);

        /* Attn out */
        snprintf(name, sizeof(name), "v.blk.%d.attn_out.weight", l);
        tw = vlm_get_tensor(g, name, 1);
        snprintf(name, sizeof(name), "v.blk.%d.attn_out.bias", l);
        tb = vlm_get_tensor(g, name, 1);
        blk->attn_out = vlm_upload_weight(&tw, &tb, r->use_f16);

        /* FFN up */
        snprintf(name, sizeof(name), "v.blk.%d.ffn_up.weight", l);
        tw = vlm_get_tensor(g, name, 1);
        snprintf(name, sizeof(name), "v.blk.%d.ffn_up.bias", l);
        tb = vlm_get_tensor(g, name, 1);
        blk->ffn_up = vlm_upload_weight(&tw, &tb, r->use_f16);

        /* FFN down */
        snprintf(name, sizeof(name), "v.blk.%d.ffn_down.weight", l);
        tw = vlm_get_tensor(g, name, 1);
        snprintf(name, sizeof(name), "v.blk.%d.ffn_down.bias", l);
        tb = vlm_get_tensor(g, name, 1);
        blk->ffn_down = vlm_upload_weight(&tw, &tb, r->use_f16);

        /* LayerNorms (always F32) */
        snprintf(name, sizeof(name), "v.blk.%d.ln1.weight", l);
        vlm_tensor_info tln = vlm_get_tensor(g, name, 1);
        blk->ln1_w = vlm_upload_f32(&tln);
        snprintf(name, sizeof(name), "v.blk.%d.ln1.bias", l);
        tln = vlm_get_tensor(g, name, 1);
        blk->ln1_b = vlm_upload_f32(&tln);

        snprintf(name, sizeof(name), "v.blk.%d.ln2.weight", l);
        tln = vlm_get_tensor(g, name, 1);
        blk->ln2_w = vlm_upload_f32(&tln);
        snprintf(name, sizeof(name), "v.blk.%d.ln2.bias", l);
        tln = vlm_get_tensor(g, name, 1);
        blk->ln2_b = vlm_upload_f32(&tln);
    }

    /* DeepStack */
    idx = gguf_find_key(g, "clip.vision.is_deepstack_layers");
    if (idx >= 0) {
        uint8_t *flags = (uint8_t *)g->kv[idx].value.arr.data;
        int n = (int)g->kv[idx].value.arr.n;
        int ns = 0;
        for (int i = 0; i < n; i++) if (flags[i]) ns++;
        r->n_deepstack = ns;
        r->deepstack_indices = (int *)malloc(ns * sizeof(int));
        r->deepstack = (gpu_deepstack *)calloc(ns, sizeof(gpu_deepstack));
        int si = 0;
        for (int i = 0; i < n; i++) {
            if (!flags[i]) continue;
            r->deepstack_indices[si] = i;
            char name[128];

            snprintf(name, sizeof(name), "v.deepstack.%d.fc1.weight", i);
            vlm_tensor_info tw = vlm_get_tensor(g, name, 1);
            snprintf(name, sizeof(name), "v.deepstack.%d.fc1.bias", i);
            vlm_tensor_info tb = vlm_get_tensor(g, name, 1);
            r->deepstack[si].fc1 = vlm_upload_weight(&tw, &tb, r->use_f16);

            snprintf(name, sizeof(name), "v.deepstack.%d.fc2.weight", i);
            tw = vlm_get_tensor(g, name, 1);
            snprintf(name, sizeof(name), "v.deepstack.%d.fc2.bias", i);
            tb = vlm_get_tensor(g, name, 1);
            r->deepstack[si].fc2 = vlm_upload_weight(&tw, &tb, r->use_f16);

            snprintf(name, sizeof(name), "v.deepstack.%d.norm.weight", i);
            vlm_tensor_info tln = vlm_get_tensor(g, name, 1);
            r->deepstack[si].norm_w = vlm_upload_f32(&tln);
            snprintf(name, sizeof(name), "v.deepstack.%d.norm.bias", i);
            tln = vlm_get_tensor(g, name, 1);
            r->deepstack[si].norm_b = vlm_upload_f32(&tln);

            si++;
        }
        fprintf(stderr, "hip_vlm: %d deepstack layers at:", ns);
        for (int i = 0; i < ns; i++) fprintf(stderr, " %d", r->deepstack_indices[i]);
        fprintf(stderr, "\n");
    }

    /* Post LN */
    vlm_tensor_info tln = vlm_get_tensor(g, "v.post_ln.weight", 1);
    r->d_post_ln_w = vlm_upload_f32(&tln);
    tln = vlm_get_tensor(g, "v.post_ln.bias", 1);
    r->d_post_ln_b = vlm_upload_f32(&tln);

    /* MM projection */
    vlm_tensor_info tw, tb;
    tw = vlm_get_tensor(g, "mm.0.weight", 1);
    tb = vlm_get_tensor(g, "mm.0.bias", 1);
    r->mm0 = vlm_upload_weight(&tw, &tb, r->use_f16);

    tw = vlm_get_tensor(g, "mm.2.weight", 1);
    tb = vlm_get_tensor(g, "mm.2.bias", 1);
    r->mm2 = vlm_upload_weight(&tw, &tb, r->use_f16);

    /* Allocate scratch buffers (sized for max_patches, not n_patches) */
    {
        size_t rgb_pixels = r->max_pixels > 0 ? (size_t)r->max_pixels : (size_t)r->image_size * r->image_size;
        size_t max_pack_elems = (size_t)mp * r->ffn_dim;
        size_t merge_pack_elems = (size_t)max_merged * merged_dim;
        if (merge_pack_elems > max_pack_elems) max_pack_elems = merge_pack_elems;
        CHECK_HIP(hipMalloc(&r->d_hidden,    (size_t)mp * dim * sizeof(float)));
        CHECK_HIP(hipMalloc(&r->d_hidden2,   (size_t)mp * dim * sizeof(float)));
        CHECK_HIP(hipMalloc(&r->d_qkv,       (size_t)mp * 3 * dim * sizeof(float)));
        CHECK_HIP(hipMalloc(&r->d_kt,        (size_t)mp * dim * sizeof(float)));
        CHECK_HIP(hipMalloc(&r->d_vt,        (size_t)mp * dim * sizeof(float)));
        CHECK_HIP(hipMalloc(&r->d_kt_h,      (size_t)mp * dim * sizeof(unsigned short)));
        CHECK_HIP(hipMalloc(&r->d_vt_h,      (size_t)mp * dim * sizeof(unsigned short)));
        CHECK_HIP(hipMalloc(&r->d_attn_out,  (size_t)mp * dim * sizeof(float)));
        CHECK_HIP(hipMalloc(&r->d_ffn_buf,   (size_t)mp * r->ffn_dim * sizeof(float)));
        CHECK_HIP(hipMalloc(&r->d_ln_buf,    (size_t)mp * dim * sizeof(float)));
        CHECK_HIP(hipMalloc(&r->d_gemm_pack, max_pack_elems * sizeof(uint16_t)));
        CHECK_HIP(hipMalloc(&r->d_merge_buf, (size_t)max_merged * merged_dim * sizeof(float)));
        CHECK_HIP(hipMalloc(&r->d_mm_buf,    (size_t)max_merged * merged_dim * sizeof(float)));
        CHECK_HIP(hipMalloc(&r->d_mm_out,    (size_t)max_merged * r->proj_dim * sizeof(float)));
        CHECK_HIP(hipMalloc(&r->d_rgb,       rgb_pixels * 3 * sizeof(float)));
        CHECK_HIP(hipMalloc(&r->d_rope_cos,  (size_t)mp * r->head_dim * sizeof(float)));
        CHECK_HIP(hipMalloc(&r->d_rope_sin,  (size_t)mp * r->head_dim * sizeof(float)));
        CHECK_HIP(hipMalloc(&r->d_pos_map,   (size_t)mp * sizeof(int)));
        CHECK_HIP(hipMalloc(&r->d_pos_interp,(size_t)mp * dim * sizeof(float)));
    }

    /* DeepStack feature buffer */
    if (r->n_deepstack > 0) {
        CHECK_HIP(hipMalloc(&r->d_ds_feats,
            (size_t)max_merged * r->n_deepstack * r->proj_dim * sizeof(float)));
    }

    int total_embd = r->proj_dim * (1 + r->n_deepstack);
    r->h_output = (float *)malloc((size_t)max_merged * total_embd * sizeof(float));

    r->loaded = 1;
    fprintf(stderr, "hip_vlm: weights loaded, VRAM for weights ~%.1f MB\n",
            (r->use_f16 ? 0.5f : 1.0f) * (float)(
                (size_t)r->n_blocks * (3*dim*dim + dim*dim + r->ffn_dim*dim + dim*r->ffn_dim) +
                merged_dim*merged_dim + r->proj_dim*merged_dim +
                r->n_deepstack * (merged_dim*merged_dim + r->proj_dim*merged_dim)
            ) * sizeof(float) / (1024.0f * 1024.0f));

    return 0;
}

/* ======================================================================== */
/* GEMM dispatch: F32, F16, or BF16 weights with F32 accumulation           */
/* ======================================================================== */

static void *vlm_pack_gemm_input(hip_vision_runner *r, void *d_X, int n_elem) {
    if (!r->d_gemm_pack || n_elem <= 0) return NULL;
    /* Vectorized pack: each thread handles 4 elements; round n up. */
    int n4 = (n_elem + 3) / 4;
    int grid = (n4 + 255) / 256;
    if (r->use_f16 == VLM_PREC_F16) {
        void *args[] = { &r->d_gemm_pack, &d_X, &n_elem };
        hipModuleLaunchKernel(r->fn_pack_f16_from_f32,
                       grid, 1, 1,
                       256, 1, 1,
                       0, r->stream,
                       args, NULL);
        return r->d_gemm_pack;
    }
    if (r->use_f16 == VLM_PREC_BF16) {
        void *args[] = { &r->d_gemm_pack, &d_X, &n_elem };
        hipModuleLaunchKernel(r->fn_pack_bf16_from_f32,
                       grid, 1, 1,
                       256, 1, 1,
                       0, r->stream,
                       args, NULL);
        return r->d_gemm_pack;
    }
    return NULL;
}

static void vlm_apply_gemm_epilogue(hip_vision_runner *r, void *d_Y, const gpu_weight *w,
                                    void *d_residual, int epilogue, int n_tok, int n_out) {
    if (w->bias) {
        void *d_bias = w->bias;
        void *args[] = { &d_Y, &d_bias, &n_out, &n_tok };
        hipModuleLaunchKernel(r->fn_add_bias_f32,
                       (n_tok * n_out + 255) / 256, 1, 1,
                       256, 1, 1,
                       0, r->stream,
                       args, NULL);
    }
    if (epilogue == 1) {
        int n = n_tok * n_out;
        void *args[] = { &d_Y, &n };
        hipModuleLaunchKernel(r->fn_gelu_f32,
                       (n + 255) / 256, 1, 1,
                       256, 1, 1,
                       0, r->stream,
                       args, NULL);
    } else if (epilogue == 2 && d_residual) {
        int n = n_tok * n_out;
        void *args[] = { &d_Y, &d_residual, &n };
        hipModuleLaunchKernel(r->fn_add_f32,
                       (n + 255) / 256, 1, 1,
                       256, 1, 1,
                       0, r->stream,
                       args, NULL);
    }
}

static int vlm_gemm_hipblaslt(hip_vision_runner *r, void *d_Y, const gpu_weight *w,
                              void *d_X, void *d_residual, int epilogue,
                              int n_tok, int n_out, int n_in) {
    if (!r->hipblaslt_enabled || !r->hipblaslt_matmul) return 0;
    if (r->use_f16 != VLM_PREC_F16 && r->use_f16 != VLM_PREC_BF16) return 0;
    if (r->use_f16 == VLM_PREC_F16 && !w->w_f16) return 0;
    if (r->use_f16 == VLM_PREC_BF16 && !w->w_bf16) return 0;

    /* Tuned win: projector-like square GEMM with fused bias+GELU.  ViT FFN-up
     * has smaller K and tuned slower with hipBLASLt, so leave it on WMMA. */
    if (epilogue != 1 || d_residual || !w->bias) return 0;
    if (!(n_tok >= 512 && n_tok <= 2048 && n_in >= 4096 && n_out >= 4096 && n_in == n_out)) return 0;

    void *d_X_pack = vlm_pack_gemm_input(r, d_X, n_tok * n_in);
    if (!d_X_pack) return 0;

    const int dtype = (r->use_f16 == VLM_PREC_BF16) ? VLM_HIPBLASLT_DATATYPE_BF16 : VLM_HIPBLASLT_DATATYPE_F16;
    const void *d_W = (r->use_f16 == VLM_PREC_BF16) ? w->w_bf16 : w->w_f16;
    const void *d_bias = w->bias;

    if (!vlm_hipblaslt_plan_init(r, &r->hipblaslt_mm0_gelu, dtype,
                                 n_tok, n_out, n_in,
                                 VLM_HIPBLASLT_EPILOGUE_GELU_BIAS, d_bias)) {
        return 0;
    }

    if (r->hipblaslt_matmul_desc_set_attr(r->hipblaslt_mm0_gelu.matmul,
                                          VLM_HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                          &d_bias, sizeof(d_bias)) != VLM_HIPBLASLT_STATUS_SUCCESS) {
        return 0;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    int st = r->hipblaslt_matmul(r->hipblaslt,
                                 r->hipblaslt_mm0_gelu.matmul,
                                 &alpha,
                                 d_W, r->hipblaslt_mm0_gelu.a,
                                 d_X_pack, r->hipblaslt_mm0_gelu.b,
                                 &beta,
                                 d_Y, r->hipblaslt_mm0_gelu.c,
                                 d_Y, r->hipblaslt_mm0_gelu.d,
                                 &r->hipblaslt_mm0_gelu.algo,
                                 NULL, 0,
                                 r->stream);
    if (st != VLM_HIPBLASLT_STATUS_SUCCESS) {
        if (r->verbose >= 1) fprintf(stderr, "hip_vlm: hipBLASLt GEMM failed (%d), falling back to rocBLAS/HIPRTC WMMA\n", st);
        return 0;
    }

    return 1;
}

/* Find-or-create cached hipBLASLt plan for a given (dtype, m, n, k, epilogue).
 * Returns NULL if cache full or plan creation failed. */
static vlm_hipblaslt_plan *vlm_hipblaslt_vit_get_plan(hip_vision_runner *r,
        int dtype, int m, int n, int k, int epilogue, const void *bias) {
    for (int i = 0; i < r->hipblaslt_vit_n_plans; i++) {
        vlm_hipblaslt_plan *p = &r->hipblaslt_vit_plans[i];
        if (p->valid && p->dtype == dtype && p->m == m && p->n == n &&
            p->k == k && p->epilogue == epilogue) {
            return p;
        }
    }
    if (r->hipblaslt_vit_n_plans >= VLM_HIPBLASLT_VIT_MAX_PLANS) return NULL;
    vlm_hipblaslt_plan *p = &r->hipblaslt_vit_plans[r->hipblaslt_vit_n_plans];
    if (!vlm_hipblaslt_plan_init(r, p, dtype, m, n, k, epilogue, bias)) return NULL;
    r->hipblaslt_vit_n_plans++;
    return p;
}

/* Try hipBLASLt for ViT block GEMMs (qkv/attn_out/ffn_w0/ffn_w1).
 * Handles epilogue ∈ {bias, GELU+bias, residual+bias} with bias != NULL.
 * Residual is implemented via beta=1.0 with C=residual, D=Y. */
static int vlm_gemm_hipblaslt_vit(hip_vision_runner *r, void *d_Y, const gpu_weight *w,
                                   void *d_X, void *d_residual, int epilogue,
                                   int n_tok, int n_out, int n_in) {
    static int env_checked = 0;
    static int env_enabled = 1;
    if (!env_checked) {
        const char *e = getenv("HIP_VLM_HIPBLASLT_VIT");
        env_enabled = (e == NULL) ? 1 : (atoi(e) != 0);
        env_checked = 1;
    }
    if (!env_enabled) return 0;
    if (!r->hipblaslt_enabled || !r->hipblaslt_matmul) return 0;
    if (r->use_f16 != VLM_PREC_F16 && r->use_f16 != VLM_PREC_BF16) return 0;
    if (r->use_f16 == VLM_PREC_F16 && !w->w_f16) return 0;
    if (r->use_f16 == VLM_PREC_BF16 && !w->w_bf16) return 0;
    if (!w->bias) return 0;

    /* Determine hipBLASLt epilogue code from caller's epilogue+residual.
     *   epilogue==0, no residual -> BIAS
     *   epilogue==1, no residual -> GELU_BIAS
     *   epilogue==2 + residual   -> BIAS with beta=1, C=residual */
    int hbl_epi;
    if (epilogue == 0 && !d_residual) hbl_epi = VLM_HIPBLASLT_EPILOGUE_BIAS;
    else if (epilogue == 1 && !d_residual) hbl_epi = VLM_HIPBLASLT_EPILOGUE_GELU_BIAS;
    else if (epilogue == 2 && d_residual) hbl_epi = VLM_HIPBLASLT_EPILOGUE_BIAS;
    else return 0;

    void *d_X_pack = vlm_pack_gemm_input(r, d_X, n_tok * n_in);
    if (!d_X_pack) return 0;

    const int dtype = (r->use_f16 == VLM_PREC_BF16) ? VLM_HIPBLASLT_DATATYPE_BF16
                                                     : VLM_HIPBLASLT_DATATYPE_F16;
    const void *d_W    = (r->use_f16 == VLM_PREC_BF16) ? w->w_bf16 : w->w_f16;
    const void *d_bias = w->bias;

    vlm_hipblaslt_plan *p = vlm_hipblaslt_vit_get_plan(r, dtype,
            n_tok, n_out, n_in, hbl_epi, d_bias);
    if (!p) return 0;

    if (r->hipblaslt_matmul_desc_set_attr(p->matmul,
                                          VLM_HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                          &d_bias, sizeof(d_bias)) != VLM_HIPBLASLT_STATUS_SUCCESS) {
        return 0;
    }

    const float alpha = 1.0f;
    const float beta = (epilogue == 2 && d_residual) ? 1.0f : 0.0f;
    const void *d_C = (epilogue == 2 && d_residual) ? d_residual : d_Y;
    int st = r->hipblaslt_matmul(r->hipblaslt, p->matmul,
                                 &alpha,
                                 d_W, p->a,
                                 d_X_pack, p->b,
                                 &beta,
                                 d_C, p->c,
                                 d_Y, p->d,
                                 &p->algo,
                                 NULL, 0,
                                 r->stream);
    if (st != VLM_HIPBLASLT_STATUS_SUCCESS) {
        if (r->verbose >= 1)
            fprintf(stderr, "hip_vlm: hipBLASLt VIT GEMM failed (%d) M=%d N=%d K=%d epi=%d, falling back\n",
                    st, n_tok, n_out, n_in, hbl_epi);
        return 0;
    }
    return 1;
}

static int vlm_gemm_rocblas(hip_vision_runner *r, void *d_Y, const gpu_weight *w,
                            void *d_X, void *d_residual, int epilogue,
                            int n_tok, int n_out, int n_in) {
    if (!r->rocblas_enabled || !r->rocblas_gemm_ex) return 0;
    /* rocBLAS wins on the large projector GEMMs but loses the fused ViT block
     * GEMMs because it needs separate epilogue launches. */
    if (!(n_tok <= 2048 && n_in >= 4096 && n_out >= 4096)) return 0;
    if (r->use_f16 != VLM_PREC_F16 && r->use_f16 != VLM_PREC_BF16) return 0;
    if (r->use_f16 == VLM_PREC_F16 && !w->w_f16) return 0;
    if (r->use_f16 == VLM_PREC_BF16 && !w->w_bf16) return 0;

    void *d_X_pack = vlm_pack_gemm_input(r, d_X, n_tok * n_in);
    if (!d_X_pack) return 0;

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int dtype = (r->use_f16 == VLM_PREC_BF16) ? VLM_ROCBLAS_DATATYPE_BF16 : VLM_ROCBLAS_DATATYPE_F16;
    const void *d_W = (r->use_f16 == VLM_PREC_BF16) ? w->w_bf16 : w->w_f16;

    int st = r->rocblas_gemm_ex(r->rocblas,
                                VLM_ROCBLAS_OP_TRANSPOSE,
                                VLM_ROCBLAS_OP_NONE,
                                n_out, n_tok, n_in,
                                &alpha,
                                d_W, dtype, n_in,
                                d_X_pack, dtype, n_in,
                                &beta,
                                d_Y, VLM_ROCBLAS_DATATYPE_F32, n_out,
                                d_Y, VLM_ROCBLAS_DATATYPE_F32, n_out,
                                VLM_ROCBLAS_DATATYPE_F32,
                                VLM_ROCBLAS_GEMM_ALGO_STANDARD,
                                0, 0);
    if (st != VLM_ROCBLAS_STATUS_SUCCESS) {
        if (r->verbose >= 1) fprintf(stderr, "hip_vlm: rocBLAS GEMM failed (%d), falling back to HIPRTC WMMA\n", st);
        return 0;
    }

    vlm_apply_gemm_epilogue(r, d_Y, w, d_residual, epilogue, n_tok, n_out);
    return 1;
}

static int vlm_gemm_wmma(hip_vision_runner *r, void *d_Y, const gpu_weight *w,
                         void *d_X, void *d_residual, int epilogue,
                         int n_tok, int n_out, int n_in) {
    if (n_tok < 16) return 0;

    if (vlm_gemm_hipblaslt(r, d_Y, w, d_X, d_residual, epilogue, n_tok, n_out, n_in)) {
        return 1;
    }

    if (vlm_gemm_hipblaslt_vit(r, d_Y, w, d_X, d_residual, epilogue, n_tok, n_out, n_in)) {
        return 1;
    }

    if (vlm_gemm_rocblas(r, d_Y, w, d_X, d_residual, epilogue, n_tok, n_out, n_in)) {
        return 1;
    }

    void *d_X_pack = vlm_pack_gemm_input(r, d_X, n_tok * n_in);
    if (!d_X_pack) return 0;

    int grid_x = (n_out + 127) / 128;
    int grid_y = (n_tok + 127) / 128;
    void *d_bias = w->bias;

    if (r->use_f16 == VLM_PREC_F16 && w->w_f16) {
        void *d_W = w->w_f16;
        void *args[] = { &d_Y, &d_W, &d_X_pack, &d_bias, &d_residual, &epilogue, &n_out, &n_in, &n_tok };
        hipModuleLaunchKernel(r->fn_gemm_wmma_f16_f32,
                       grid_x, grid_y, 1,
                       256, 1, 1,
                       0, r->stream,
                       args, NULL);
        return 1;
    }

    if (r->use_f16 == VLM_PREC_BF16 && w->w_bf16) {
        void *d_W = w->w_bf16;
        void *args[] = { &d_Y, &d_W, &d_X_pack, &d_bias, &d_residual, &epilogue, &n_out, &n_in, &n_tok };
        hipModuleLaunchKernel(r->fn_gemm_wmma_bf16_f32,
                       grid_x, grid_y, 1,
                       256, 1, 1,
                       0, r->stream,
                       args, NULL);
        return 1;
    }

    return 0;
}

/* Launch a GEMM: Y[n_tok, n_out] = X[n_tok, n_in] * W^T[n_out, n_in] + bias */
static void vlm_gemm(hip_vision_runner *r, void *d_Y, const gpu_weight *w,
                       void *d_X, int n_tok, int n_out, int n_in) {
    void *d_W, *d_bias;
    d_bias = w->bias;
    if (vlm_gemm_wmma(r, d_Y, w, d_X, NULL, 0, n_tok, n_out, n_in)) {
        return;
    }

    if (r->use_f16 == VLM_PREC_F16 && w->w_f16) {
        /* Optimized 128x128 tiled F16 GEMM */
        d_W = w->w_f16;
        int grid_x = (n_out + 127) / 128;
        int grid_y = (n_tok + 127) / 128;
        void *args[] = { &d_Y, &d_W, &d_X, &d_bias, &n_out, &n_in, &n_tok };
        hipModuleLaunchKernel(r->fn_gemm_opt_f16_f32,
                       grid_x, grid_y, 1,
                       256, 1, 1,
                       0, r->stream,
                       args, NULL);
    } else if (r->use_f16 == VLM_PREC_BF16 && w->w_bf16) {
        /* Optimized 128x128 tiled BF16-weight GEMM */
        d_W = w->w_bf16;
        void *d_residual = NULL;
        int epilogue = 0;
        int grid_x = (n_out + 127) / 128;
        int grid_y = (n_tok + 127) / 128;
        void *args[] = { &d_Y, &d_W, &d_X, &d_bias, &d_residual, &epilogue, &n_out, &n_in, &n_tok };
        hipModuleLaunchKernel(r->fn_gemm_opt_bf16_f32,
                       grid_x, grid_y, 1,
                       256, 1, 1,
                       0, r->stream,
                       args, NULL);
    } else {
        /* F32 tiled path */
        d_W = w->w_f32;
        int grid_x = (n_out + 15) / 16;
        int grid_y = (n_tok + 15) / 16;
        void *args[] = { &d_Y, &d_W, &d_X, &d_bias, &n_out, &n_in, &n_tok };
        hipModuleLaunchKernel(r->fn_gemm_f32_f32,
                       grid_x, grid_y, 1,
                       16, 16, 1,
                       0, r->stream,
                       args, NULL);
    }
}

/* GEMM + fused GELU (for FFN-up) */
static void vlm_gemm_gelu(hip_vision_runner *r, void *d_Y, const gpu_weight *w,
                            void *d_X, int n_tok, int n_out, int n_in) {
    if (vlm_gemm_wmma(r, d_Y, w, d_X, NULL, 1, n_tok, n_out, n_in)) {
        return;
    }

    if (r->use_f16 == VLM_PREC_F16 && w->w_f16) {
        void *d_W = w->w_f16, *d_bias = w->bias;
        int grid_x = (n_out + 127) / 128;
        int grid_y = (n_tok + 127) / 128;
        void *args[] = { &d_Y, &d_W, &d_X, &d_bias, &n_out, &n_in, &n_tok };
        hipModuleLaunchKernel(r->fn_gemm_opt_f16_f32_gelu,
                       grid_x, grid_y, 1, 256, 1, 1, 0, r->stream, args, NULL);
    } else if (r->use_f16 == VLM_PREC_BF16 && w->w_bf16) {
        void *d_W = w->w_bf16, *d_bias = w->bias, *d_residual = NULL;
        int epilogue = 1;
        int grid_x = (n_out + 127) / 128;
        int grid_y = (n_tok + 127) / 128;
        void *args[] = { &d_Y, &d_W, &d_X, &d_bias, &d_residual, &epilogue, &n_out, &n_in, &n_tok };
        hipModuleLaunchKernel(r->fn_gemm_opt_bf16_f32,
                       grid_x, grid_y, 1, 256, 1, 1, 0, r->stream, args, NULL);
    } else {
        vlm_gemm(r, d_Y, w, d_X, n_tok, n_out, n_in);
        /* Separate GELU for F32 path */
        int n = n_tok * n_out;
        int grid = (n + 255) / 256;
        void *args[] = { &d_Y, &n };
        hipModuleLaunchKernel(r->fn_gelu_f32, grid, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
    }
}

/* Extracted-launcher path: hand-tuned hipBLASLt kernel sustaining 174 TFLOP/s
 * (89% peak) on M=1024 N=4608 K=4608 BF16. See rdna4/vlm/mm0_lever_attribution.md.
 * Skips libhipblaslt's per-call algo selection. Bias is fused; GELU runs as
 * a separate kernel after. */
static int vlm_projector_mm0_extracted(hip_vision_runner *r, void *d_Y,
                                       const gpu_weight *w, void *d_X,
                                       int n_tok, int n_out, int n_in) {
    static int s_inited = 0;
    static int s_init_failed = 0;
    static const void *s_bias_at_init = NULL;
    if (s_init_failed) return 0;
    if (r->use_f16 != VLM_PREC_BF16 || !w->w_bf16 || !w->bias) return 0;
    if (n_tok != 1024 || n_out != 4608 || n_in != 4608) return 0;
    if (!getenv("HIP_VLM_MM0_EXTRACTED") || getenv("HIP_VLM_MM0_EXTRACTED")[0] == '0') return 0;

    if (!s_inited) {
        if (mm0_extracted_init(n_tok, n_out, n_in, w->bias) != 0) {
            s_init_failed = 1;
            return 0;
        }
        s_inited = 1;
        s_bias_at_init = w->bias;
    }
    if (s_bias_at_init != w->bias) return 0; /* bias pointer changed; bail */

    void *d_X_pack = vlm_pack_gemm_input(r, d_X, n_tok * n_in);
    if (!d_X_pack) return 0;

    if (mm0_extracted_run(d_Y, w->w_bf16, d_X_pack) != 0) return 0;

    int n = n_tok * n_out;
    int grid = (n + 255) / 256;
    void *args[] = { &d_Y, &n };
    hipModuleLaunchKernel(r->fn_gelu_f32, grid, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
    return 1;
}

static void vlm_projector_gemm_gelu(hip_vision_runner *r, void *d_Y, const gpu_weight *w,
                                    void *d_X, int n_tok, int n_out, int n_in) {
    if (vlm_projector_mm0_extracted(r, d_Y, w, d_X, n_tok, n_out, n_in)) {
        return;
    }
    if (n_tok >= 512) {
        vlm_gemm_gelu(r, d_Y, w, d_X, n_tok, n_out, n_in);
        return;
    }

    vlm_gemm(r, d_Y, w, d_X, n_tok, n_out, n_in);
    int n = n_tok * n_out;
    int grid = (n + 255) / 256;
    void *args[] = { &d_Y, &n };
    hipModuleLaunchKernel(r->fn_gelu_f32,
                   grid, 1, 1,
                   256, 1, 1,
                   0, r->stream,
                   args, NULL);
}

/* GEMM + fused residual add (for attn-out and FFN-down).
 * d_Y and d_residual may alias (in-place residual). F16 fused kernel handles
 * this atomically; F32 fallback uses d_hidden2 as scratch to avoid aliasing. */
static void vlm_gemm_res(hip_vision_runner *r, void *d_Y, const gpu_weight *w,
                           void *d_X, void *d_residual, int n_tok, int n_out, int n_in) {
    if (vlm_gemm_wmma(r, d_Y, w, d_X, d_residual, 2, n_tok, n_out, n_in)) {
        return;
    }

    if (r->use_f16 == VLM_PREC_F16 && w->w_f16) {
        void *d_W = w->w_f16, *d_bias = w->bias;
        int grid_x = (n_out + 127) / 128;
        int grid_y = (n_tok + 127) / 128;
        void *args[] = { &d_Y, &d_W, &d_X, &d_bias, &d_residual, &n_out, &n_in, &n_tok };
        hipModuleLaunchKernel(r->fn_gemm_opt_f16_f32_res,
                       grid_x, grid_y, 1, 256, 1, 1, 0, r->stream, args, NULL);
    } else if (r->use_f16 == VLM_PREC_BF16 && w->w_bf16) {
        void *d_W = w->w_bf16, *d_bias = w->bias;
        int epilogue = 2;
        int grid_x = (n_out + 127) / 128;
        int grid_y = (n_tok + 127) / 128;
        void *args[] = { &d_Y, &d_W, &d_X, &d_bias, &d_residual, &epilogue, &n_out, &n_in, &n_tok };
        hipModuleLaunchKernel(r->fn_gemm_opt_bf16_f32,
                       grid_x, grid_y, 1, 256, 1, 1, 0, r->stream, args, NULL);
    } else {
        /* Write GEMM to scratch, then add residual */
        vlm_gemm(r, r->d_hidden2, w, d_X, n_tok, n_out, n_in);
        /* Copy scratch to output: d_Y = d_hidden2 + d_residual */
        int n = n_tok * n_out;
        int grid = (n + 255) / 256;
        /* d_Y = d_residual (copy first if not aliased, then add) */
        if (d_Y != d_residual) {
            hipMemcpyAsync(d_Y, d_residual, (size_t)n * sizeof(float),
                          hipMemcpyDeviceToDevice, r->stream);
        }
        void *args[] = { &d_Y, &r->d_hidden2, &n };
        hipModuleLaunchKernel(r->fn_add_f32, grid, 1, 1, 256, 1, 1, 0, r->stream, args, NULL);
    }
}

/* ======================================================================== */
/* Public API: encode                                                       */
/* ======================================================================== */

float *hip_vision_encode(hip_vision_runner *r, const float *rgb_norm, int width, int height) {
    if (!r || !r->loaded) return NULL;

    int ps = r->patch_size;
    int dim = r->dim;
    int n_heads = r->n_heads;
    int head_dim = r->head_dim;
    int ffn_dim = r->ffn_dim;
    int gw = width / ps;
    int gh = height / ps;
    int n_patches = gw * gh;
    int sm = r->spatial_merge;
    int merged_dim = dim * sm * sm;
    int n_merged = n_patches / (sm * sm);
    int *token_perm = NULL;

    if ((width % (ps * sm)) != 0 || (height % (ps * sm)) != 0) {
        fprintf(stderr, "hip_vlm: image %dx%d must be divisible by patch_size*spatial_merge (%d)\n",
                width, height, ps * sm);
        return NULL;
    }
    if (r->use_qwen3vl_layout && sm != 2) {
        fprintf(stderr, "hip_vlm: Qwen3VL layout currently requires spatial_merge=2 (got %d)\n", sm);
        return NULL;
    }

    if (n_patches > r->max_patches) {
        fprintf(stderr, "hip_vlm: too many patches %d (max %d)\n", n_patches, r->max_patches);
        return NULL;
    }

    fprintf(stderr, "hip_vlm: encoding %dx%d image (%d patches, %d merged tokens)\n",
            width, height, n_patches, n_merged);

    /* 1. Upload RGB to GPU */
    hipMemcpy(r->d_rgb, rgb_norm, (size_t)width * height * 3 * sizeof(float), hipMemcpyHostToDevice);

    /* 2. Patch embedding.
     *  Fast path: im2col (rgb → BF16 unfold) + WMMA GEMM (W folded W0+W1).
     *  Fallback: scalar dual-conv kernel. */
    fprintf(stderr, "  patch embedding (dual conv2d)...\n");
    if (r->d_patch_w0_f16 && r->d_patch_unfold_f16) {
        int img_w = width;
        int ks = ps * ps * 3;
        /* im2col into F16 [n_patches, ks] */
        void *uf_args[] = { &r->d_patch_unfold_f16, &r->d_rgb,
                            &gw, &ps, &ks, &img_w };
        hipModuleLaunchKernel(r->fn_patch_unfold_f16,
                       n_patches, 1, 1, 256, 1, 1,
                       0, r->stream, uf_args, NULL);
        /* GEMM: out = unfold @ W^T -> Y[n_patches, dim].
         * gemm_wmma_f16_f32(Y, W, X, bias, residual, epilogue, N=dim, K=ks, M=n_patches)
         * Bias path: qwen3vl drops the patch bias (use_qwen3vl_layout). */
        void *patch_bias = r->use_qwen3vl_layout ? NULL : r->d_patch_bias;
        void *residual = NULL;
        int epilogue = 0;
        int N = dim, K = ks, M = n_patches;
        void *gemm_args[] = { &r->d_hidden, &r->d_patch_w0_f16, &r->d_patch_unfold_f16,
                              &patch_bias, &residual, &epilogue, &N, &K, &M };
        int grid_x = (N + 127) / 128;
        int grid_y = (M + 127) / 128;
        hipModuleLaunchKernel(r->fn_gemm_wmma_f16_f32,
                       grid_x, grid_y, 1,
                       256, 1, 1,
                       0, r->stream, gemm_args, NULL);
    } else {
        int img_w = width;
        void *patch_bias = r->use_qwen3vl_layout ? NULL : r->d_patch_bias;
        void *args[] = {
            &r->d_hidden, &r->d_rgb,
            &r->d_patch_w0, &r->d_patch_w1, &patch_bias,
            &gw, &dim, &ps, &img_w
        };
        hipModuleLaunchKernel(r->fn_patch_embed_dual_f32,
                       n_patches, 1, 1,
                       256, 1, 1,
                       0, r->stream,
                       args, NULL);
    }

    /* Debug: check patch embedding output */
    if (r->verbose >= 2) {
        hipDeviceSynchronize();
        float dbg[8];
        hipMemcpy(dbg, r->d_hidden, 8 * sizeof(float), hipMemcpyDeviceToHost);
        fprintf(stderr, "  [DBG] hidden after patch_embed: %.6f %.6f %.6f %.6f\n",
                dbg[0], dbg[1], dbg[2], dbg[3]);
    }

    /* 3. Position embeddings (bilinear interpolation for dynamic resolution) */
    fprintf(stderr, "  position embeddings...\n");
    {
        int orig_gw = r->image_size / ps;
        int orig_gh = orig_gw;  /* original grid is square */

        if (gw == orig_gw && gh == orig_gh) {
            /* Exact match: use direct indirection (no interpolation needed) */
            int *pos_map = (int *)malloc(n_patches * sizeof(int));
            for (int py = 0; py < gh; py++)
                for (int px = 0; px < gw; px++)
                    pos_map[py * gw + px] = py * orig_gw + px;
            hipMemcpy(r->d_pos_map, pos_map, n_patches * sizeof(int), hipMemcpyHostToDevice);
            free(pos_map);

            void *args[] = { &r->d_hidden, &r->d_pos_embd, &r->d_pos_map, &dim };
            hipModuleLaunchKernel(r->fn_add_pos_embd,
                           n_patches, 1, 1,
                           256, 1, 1,
                           0, r->stream,
                           args, NULL);
        } else {
            /* Bilinear interpolation on CPU, upload to d_pos_interp */
            fprintf(stderr, "  interpolating pos embedding: %dx%d -> %dx%d\n",
                    orig_gw, orig_gh, gw, gh);
            float *interp = (float *)calloc((size_t)n_patches * dim, sizeof(float));
            float sf_x = (float)gw / (float)orig_gw;
            float sf_y = (float)gh / (float)orig_gh;
            float support_x = fmaxf(1.0f, 1.0f / sf_x);
            float support_y = fmaxf(1.0f, 1.0f / sf_y);
            float invscale_x = 1.0f / support_x;
            float invscale_y = 1.0f / support_y;
            const float pixel_offset = 0.5f;
            for (int py = 0; py < gh; py++) {
                float y = ((float)py + pixel_offset) / sf_y;
                int y_min = (int)fmaxf(y - support_y + pixel_offset, 0.0f);
                int y_max = (int)fminf(y + support_y + pixel_offset, (float)orig_gh);
                for (int px = 0; px < gw; px++) {
                    float x = ((float)px + pixel_offset) / sf_x;
                    int x_min = (int)fmaxf(x - support_x + pixel_offset, 0.0f);
                    int x_max = (int)fminf(x + support_x + pixel_offset, (float)orig_gw);
                    int dst_idx = (py * gw + px) * dim;
                    float total_weight = 0.0f;
                    for (int sy = y_min; sy < y_max; sy++) {
                        float wy = fmaxf(1.0f - fabsf((sy - y + pixel_offset) * invscale_y), 0.0f);
                        for (int sx = x_min; sx < x_max; sx++) {
                            float wx = fmaxf(1.0f - fabsf((sx - x + pixel_offset) * invscale_x), 0.0f);
                            float w = wx * wy;
                            if (w <= 0.0f) continue;
                            int src_idx = (sy * orig_gw + sx) * dim;
                            for (int d = 0; d < dim; d++) {
                                interp[dst_idx + d] += r->h_pos_embd[src_idx + d] * w;
                            }
                            total_weight += w;
                        }
                    }
                    if (total_weight > 0.0f) {
                        float inv_w = 1.0f / total_weight;
                        for (int d = 0; d < dim; d++) {
                            interp[dst_idx + d] *= inv_w;
                        }
                    }
                }
            }
            hipMemcpy(r->d_pos_interp, interp, (size_t)n_patches * dim * sizeof(float), hipMemcpyHostToDevice);
            free(interp);

            /* Add interpolated pos embedding directly (no pos_map needed) */
            void *args[] = { &r->d_hidden, &r->d_pos_interp, &dim, &n_patches };
            hipModuleLaunchKernel(r->fn_add_pos_embd_direct,
                           n_patches, 1, 1,
                           256, 1, 1,
                           0, r->stream,
                           args, NULL);
        }
    }

    if (r->use_qwen3vl_layout) {
        token_perm = (int *)malloc((size_t)n_patches * sizeof(int));
        if (!token_perm || vlm_build_qwen3vl_token_perm(gw, gh, sm, token_perm) != 0) {
            fprintf(stderr, "hip_vlm: failed to build Qwen3VL token permutation\n");
            free(token_perm);
            return NULL;
        }

        void *args[] = { &r->d_hidden2, &r->d_hidden, &gw, &dim };
        hipModuleLaunchKernel(r->fn_qwen3vl_repack_f32,
                       n_patches, 1, 1,
                       256, 1, 1,
                       0, r->stream,
                       args, NULL);
        {
            void *tmp = r->d_hidden;
            r->d_hidden = r->d_hidden2;
            r->d_hidden2 = tmp;
        }
        if (r->d_patch_bias) {
            int n_out = dim;
            int n_tok = n_patches;
            int n = n_out * n_tok;
            int grid = (n + 255) / 256;
            void *args[] = { &r->d_hidden, &r->d_patch_bias, &n_out, &n_tok };
            hipModuleLaunchKernel(r->fn_add_bias_f32,
                           grid, 1, 1,
                           256, 1, 1,
                           0, r->stream,
                           args, NULL);
        }
    }

    /* 4. Precompute M-RoPE cos/sin on host, upload to GPU */
    {
        int half = head_dim / 2;
        int sect_size = head_dim / 4;
        float freq_base = 10000.0f;
        float theta_scale = powf(freq_base, -2.0f / (float)half);
        float *rope_cos = (float *)malloc(n_patches * head_dim * sizeof(float));
        float *rope_sin = (float *)malloc(n_patches * head_dim * sizeof(float));

        for (int p = 0; p < n_patches; p++) {
            int src = token_perm ? token_perm[p] : p;
            int py = src / gw;
            int px = src % gw;
            float p_t = (float)py, p_h = (float)px, p_w = (float)py, p_e = (float)px;
            float cur_t = p_t, cur_h = p_h, cur_w = p_w, cur_e = p_e;

            for (int i0 = 0; i0 < head_dim; i0 += 2) {
                int sector = i0 / 2;
                if (sector == 0) cur_t = p_t;
                if (sector == sect_size) cur_h = p_h;
                if (sector == 2 * sect_size) cur_w = p_w;
                if (sector == 3 * sect_size) cur_e = p_e;

                float theta;
                if (sector < sect_size) theta = cur_t;
                else if (sector < 2 * sect_size) theta = cur_h;
                else if (sector < 3 * sect_size) theta = cur_w;
                else theta = cur_e;

                rope_cos[p * head_dim + i0] = cosf(theta);
                rope_sin[p * head_dim + i0] = sinf(theta);
                rope_cos[p * head_dim + i0 + 1] = cosf(theta);
                rope_sin[p * head_dim + i0 + 1] = sinf(theta);

                cur_t *= theta_scale;
                cur_h *= theta_scale;
                cur_w *= theta_scale;
                cur_e *= theta_scale;
            }
        }

        hipMemcpy(r->d_rope_cos, rope_cos, n_patches * head_dim * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(r->d_rope_sin, rope_sin, n_patches * head_dim * sizeof(float), hipMemcpyHostToDevice);
        free(token_perm);
        free(rope_cos);
        free(rope_sin);
    }

    /* 5. ViT blocks */
    int half = head_dim / 2;
    int ds_count = 0;

    for (int l = 0; l < r->n_blocks; l++) {
        if (l == 0 || l == r->n_blocks - 1 || (l + 1) % 6 == 0)
            fprintf(stderr, "  vit block %d/%d\n", l, r->n_blocks);

        gpu_vit_block *blk = &r->blocks[l];

        /* LayerNorm1 */
        {
            float eps = r->ln_eps;
            size_t smem = 256 * sizeof(float);
            void *args[] = { &r->d_ln_buf, &r->d_hidden, &blk->ln1_w, &blk->ln1_b, &dim, &eps };
            hipModuleLaunchKernel(r->fn_layernorm_f32,
                           n_patches, 1, 1,
                           256, 1, 1,
                           smem, r->stream,
                           args, NULL);
        }

        /* QKV projection */
        {
            int n_out = 3 * dim;
            vlm_gemm(r, r->d_qkv, &blk->attn_qkv, r->d_ln_buf, n_patches, n_out, dim);
        }

        /* Debug: check QKV right after GEMM (before RoPE) */
        if (l == 0 && r->verbose >= 2) {
            hipDeviceSynchronize();
            float dbg[8];
            hipMemcpy(dbg, r->d_qkv, 8 * sizeof(float), hipMemcpyDeviceToHost);
            fprintf(stderr, "  [DBG] qkv after GEMM (pre-RoPE), block 0: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                    dbg[0], dbg[1], dbg[2], dbg[3], dbg[4], dbg[5], dbg[6], dbg[7]);
            hipMemcpy(dbg, r->d_ln_buf, 8 * sizeof(float), hipMemcpyDeviceToHost);
            fprintf(stderr, "  [DBG] ln_buf (QKV input), block 0: %.6f %.6f %.6f %.6f\n",
                    dbg[0], dbg[1], dbg[2], dbg[3]);
            /* Check weight values */
            if (r->use_f16 == VLM_PREC_F16 && blk->attn_qkv.w_f16) {
                uint16_t wdbg[8];
                hipMemcpy(wdbg, blk->attn_qkv.w_f16, 8 * sizeof(uint16_t), hipMemcpyDeviceToHost);
                fprintf(stderr, "  [DBG] QKV weight F16[0..3]: %04x %04x %04x %04x (%.6f %.6f %.6f %.6f)\n",
                        wdbg[0], wdbg[1], wdbg[2], wdbg[3],
                        ggml_fp16_to_fp32(wdbg[0]), ggml_fp16_to_fp32(wdbg[1]),
                        ggml_fp16_to_fp32(wdbg[2]), ggml_fp16_to_fp32(wdbg[3]));
            } else if (r->use_f16 == VLM_PREC_BF16 && blk->attn_qkv.w_bf16) {
                uint16_t wdbg[8];
                hipMemcpy(wdbg, blk->attn_qkv.w_bf16, 8 * sizeof(uint16_t), hipMemcpyDeviceToHost);
                fprintf(stderr, "  [DBG] QKV weight BF16[0..3]: %04x %04x %04x %04x\n",
                        wdbg[0], wdbg[1], wdbg[2], wdbg[3]);
            } else if (blk->attn_qkv.w_f32) {
                float wdbg[4];
                hipMemcpy(wdbg, blk->attn_qkv.w_f32, 4 * sizeof(float), hipMemcpyDeviceToHost);
                fprintf(stderr, "  [DBG] QKV weight F32[0..3]: %.6f %.6f %.6f %.6f\n",
                        wdbg[0], wdbg[1], wdbg[2], wdbg[3]);
            }
        }

        /* M-RoPE on Q and K */
        {
            void *args[] = {
                &r->d_qkv, &r->d_rope_cos, &r->d_rope_sin,
                &n_patches, &n_heads, &dim, &head_dim, &half
            };
            hipModuleLaunchKernel(r->fn_rope_vision_f32,
                           n_patches * n_heads, 1, 1,
                           half, 1, 1,
                           0, r->stream,
                           args, NULL);
        }

        /* Multi-head self-attention (flash attention) */
        {
            float scale = 1.0f / sqrtf((float)head_dim);

            /* Select attention path:
             *   HIP_VLM_FA=wmma_bf16        -> WMMA BF16 (BQ=16, 32-thread WG)
             *   HIP_VLM_FA=wmma_f16         -> WMMA F16
             *   HIP_VLM_FA=wmma_bf16_2w     -> WMMA BF16 BQ=32 2-wave (F32 K/V read+cast)
             *   HIP_VLM_FA=wmma_f16_2w      -> WMMA F16 BQ=32 2-wave
             *   HIP_VLM_FA=wmma_bf16_2w_pre -> WMMA BF16 BQ=32 2-wave w/ pre-packed BF16 K/V
             *   HIP_VLM_FA=wmma_f16_2w_pre  -> WMMA F16 BQ=32 2-wave w/ pre-packed F16 K/V
             *   HIP_VLM_FA=wmma_bf16_4w_pre -> WMMA BF16 BQ=64 4-wave w/ pre-packed BF16 K/V
             *   HIP_VLM_FA=tiled or unset (head_dim==64) -> scalar tiled
             *   default                                  -> scalar dynamic */
            const char *fa_mode = getenv("HIP_VLM_FA");
            int use_wmma_bf16        = (fa_mode && strcmp(fa_mode, "wmma_bf16") == 0);
            int use_wmma_f16         = (fa_mode && strcmp(fa_mode, "wmma_f16")  == 0);
            int use_wmma_bf16_2w     = (fa_mode && strcmp(fa_mode, "wmma_bf16_2w") == 0);
            int use_wmma_f16_2w      = (fa_mode && strcmp(fa_mode, "wmma_f16_2w")  == 0);
            int use_wmma_bf16_2w_pre = (fa_mode && strcmp(fa_mode, "wmma_bf16_2w_pre") == 0);
            int use_wmma_f16_2w_pre  = (fa_mode && strcmp(fa_mode, "wmma_f16_2w_pre")  == 0);
            int use_wmma_bf16_4w_pre = (fa_mode && strcmp(fa_mode, "wmma_bf16_4w_pre") == 0);

            /* Transpose K,V — choose precision matching the FA path. */
            int total = n_patches * dim;
            int grid_kv = (total + 255) / 256;
            if ((use_wmma_bf16_2w_pre || use_wmma_f16_2w_pre || use_wmma_bf16_4w_pre) && head_dim <= 80) {
                hipFunction_t kv_fn = (use_wmma_bf16_2w_pre || use_wmma_bf16_4w_pre)
                                          ? r->fn_kv_transpose_bf16
                                          : r->fn_kv_transpose_f16;
                void *kv_args[] = { &r->d_kt_h, &r->d_vt_h, &r->d_qkv,
                                    &n_patches, &dim, &n_heads, &head_dim };
                hipModuleLaunchKernel(kv_fn,
                               grid_kv, 1, 1, 256, 1, 1,
                               0, r->stream, kv_args, NULL);
            } else {
                void *kv_args[] = { &r->d_kt, &r->d_vt, &r->d_qkv,
                                    &n_patches, &dim, &n_heads, &head_dim };
                hipModuleLaunchKernel(r->fn_kv_transpose,
                               grid_kv, 1, 1, 256, 1, 1,
                               0, r->stream, kv_args, NULL);
            }

            void *args[] = {
                &r->d_attn_out, &r->d_qkv, &r->d_kt, &r->d_vt,
                &n_patches, &dim, &n_heads, &head_dim, &scale
            };
            void *args_pre[] = {
                &r->d_attn_out, &r->d_qkv, &r->d_kt_h, &r->d_vt_h,
                &n_patches, &dim, &n_heads, &head_dim, &scale
            };

            if ((use_wmma_bf16 || use_wmma_f16) && head_dim <= 80) {
                int n_q_blocks = (n_patches + 16 - 1) / 16;
                size_t smem_bytes = (16 * 80 + 80 * 16 + 16 * 16) * sizeof(unsigned short);
                hipFunction_t fn = use_wmma_bf16 ? r->fn_flash_attn_wmma_bf16
                                                 : r->fn_flash_attn_wmma_f16;
                hipModuleLaunchKernel(fn,
                               n_heads, n_q_blocks, 1,
                               32, 1, 1, smem_bytes, r->stream, args, NULL);
            } else if ((use_wmma_bf16_2w || use_wmma_f16_2w) && head_dim <= 80) {
                int n_q_blocks = (n_patches + 32 - 1) / 32;
                size_t smem_bytes = (16 * 80 + 80 * 16 + 2 * 16 * 16) * sizeof(unsigned short);
                hipFunction_t fn = use_wmma_bf16_2w ? r->fn_flash_attn_wmma_bf16_2w
                                                    : r->fn_flash_attn_wmma_f16_2w;
                hipModuleLaunchKernel(fn,
                               n_heads, n_q_blocks, 1,
                               64, 1, 1, smem_bytes, r->stream, args, NULL);
            } else if ((use_wmma_bf16_2w_pre || use_wmma_f16_2w_pre) && head_dim <= 80) {
                int n_q_blocks = (n_patches + 32 - 1) / 32;
                /* Double-buffered K/V (both paths): 2x16x80 K + 2x80x16 V + 2x16x16 P */
                size_t smem_bytes = (2 * 16 * 80 + 2 * 80 * 16 + 2 * 16 * 16) * sizeof(unsigned short);
                hipFunction_t fn = use_wmma_bf16_2w_pre ? r->fn_flash_attn_wmma_bf16_2w_pre
                                                        : r->fn_flash_attn_wmma_f16_2w_pre;
                hipModuleLaunchKernel(fn,
                               n_heads, n_q_blocks, 1,
                               64, 1, 1, smem_bytes, r->stream, args_pre, NULL);
            } else if (use_wmma_bf16_4w_pre && head_dim <= 80) {
                int n_q_blocks = (n_patches + 64 - 1) / 64;
                /* Double-buffered K/V + 4-wave smP: 2x16x80 K + 2x80x16 V + 4x16x16 P */
                size_t smem_bytes = (2 * 16 * 80 + 2 * 80 * 16 + 4 * 16 * 16) * sizeof(unsigned short);
                hipModuleLaunchKernel(r->fn_flash_attn_wmma_bf16_4w_pre,
                               n_heads, n_q_blocks, 1,
                               128, 1, 1, smem_bytes, r->stream, args_pre, NULL);
            } else {
                int bq = 64;  /* queries per block */
                int n_q_blocks = (n_patches + bq - 1) / bq;
                if (head_dim == 64) {
                    size_t smem = 2 * 16 * 64 * sizeof(float);
                    hipModuleLaunchKernel(r->fn_flash_attn_tiled_f32,
                                   n_heads, n_q_blocks, 1,
                                   bq, 1, 1, smem, r->stream, args, NULL);
                } else {
                    size_t smem = 2 * 16 * head_dim * sizeof(float);
                    hipModuleLaunchKernel(r->fn_flash_attn_dyn_f32,
                                   n_heads, n_q_blocks, 1,
                                   bq, 1, 1, smem, r->stream, args, NULL);
                }
            }
        }

        /* Attn output projection + fused residual add */
        vlm_gemm_res(r, r->d_hidden, &blk->attn_out, r->d_attn_out, r->d_hidden,
                     n_patches, dim, dim);

        /* LayerNorm2 */
        {
            float eps = r->ln_eps;
            size_t smem = 256 * sizeof(float);
            void *args[] = { &r->d_ln_buf, &r->d_hidden, &blk->ln2_w, &blk->ln2_b, &dim, &eps };
            hipModuleLaunchKernel(r->fn_layernorm_f32,
                           n_patches, 1, 1,
                           256, 1, 1,
                           smem, r->stream,
                           args, NULL);
        }

        /* FFN: up+GELU fused -> down+residual fused */
        vlm_gemm_gelu(r, r->d_ffn_buf, &blk->ffn_up, r->d_ln_buf, n_patches, ffn_dim, dim);
        vlm_gemm_res(r, r->d_hidden, &blk->ffn_down, r->d_ffn_buf, r->d_hidden,
                     n_patches, dim, ffn_dim);

        /* Debug: check hidden and qkv after first block */
        if (l == 0 && r->verbose >= 2) {
            hipDeviceSynchronize();
            float dbg[8];
            hipMemcpy(dbg, r->d_hidden, 8 * sizeof(float), hipMemcpyDeviceToHost);
            fprintf(stderr, "  [DBG] hidden after block 0: %.6f %.6f %.6f %.6f\n",
                    dbg[0], dbg[1], dbg[2], dbg[3]);
            hipMemcpy(dbg, r->d_qkv, 8 * sizeof(float), hipMemcpyDeviceToHost);
            fprintf(stderr, "  [DBG] qkv[0..3] after block 0: %.6f %.6f %.6f %.6f\n",
                    dbg[0], dbg[1], dbg[2], dbg[3]);
        }

        /* DeepStack extraction */
        for (int ds = 0; ds < r->n_deepstack; ds++) {
            if (r->deepstack_indices[ds] != l) continue;

            fprintf(stderr, "  deepstack at layer %d\n", l);
            gpu_deepstack *dsl = &r->deepstack[ds];

            /* Spatial merge current hidden -> merge_buf */
            {
                if (r->use_qwen3vl_layout) {
                    void *args[] = { &r->d_merge_buf, &r->d_hidden, &sm, &dim };
                    hipModuleLaunchKernel(r->fn_spatial_merge_contig_f32,
                                   n_merged, 1, 1,
                                   256, 1, 1,
                                   0, r->stream,
                                   args, NULL);
                } else {
                    void *args[] = { &r->d_merge_buf, &r->d_hidden, &gw, &sm, &dim };
                    hipModuleLaunchKernel(r->fn_spatial_merge_f32,
                                   n_merged, 1, 1,
                                   256, 1, 1,
                                   0, r->stream,
                                   args, NULL);
                }
            }

            /* LayerNorm on merge_buf */
            {
                float eps = r->ln_eps;
                size_t smem = 256 * sizeof(float);
                void *args[] = { &r->d_merge_buf, &r->d_merge_buf,
                                 &dsl->norm_w, &dsl->norm_b, &merged_dim, &eps };
                hipModuleLaunchKernel(r->fn_layernorm_f32,
                               n_merged, 1, 1,
                               256, 1, 1,
                               smem, r->stream,
                               args, NULL);
            }

            /* fc1 + GELU: [merged_dim -> merged_dim] */
            vlm_projector_gemm_gelu(r, r->d_mm_buf, &dsl->fc1, r->d_merge_buf, n_merged, merged_dim, merged_dim);

            /* fc2: [merged_dim -> proj_dim] */
            vlm_gemm(r, r->d_mm_out, &dsl->fc2, r->d_mm_buf, n_merged, r->proj_dim, merged_dim);

            /* Copy to deepstack feature buffer at offset ds_count */
            {
                size_t nbytes = (size_t)n_merged * r->proj_dim * sizeof(float);
                void *dst = (char *)r->d_ds_feats + (size_t)ds_count * nbytes;
                hipMemcpyAsync(dst, r->d_mm_out, nbytes, hipMemcpyDeviceToDevice, r->stream);
            }
            ds_count++;
        }
    }

    /* 6. Post LayerNorm */
    fprintf(stderr, "  post layernorm...\n");
    {
        float eps = r->ln_eps;
        size_t smem = 256 * sizeof(float);
        void *args[] = { &r->d_hidden, &r->d_hidden, &r->d_post_ln_w, &r->d_post_ln_b, &dim, &eps };
        hipModuleLaunchKernel(r->fn_layernorm_f32,
                       n_patches, 1, 1,
                       256, 1, 1,
                       smem, r->stream,
                       args, NULL);
    }

    /* 7. Final spatial merge */
    fprintf(stderr, "  spatial merge...\n");
    {
        if (r->use_qwen3vl_layout) {
            void *args[] = { &r->d_merge_buf, &r->d_hidden, &sm, &dim };
            hipModuleLaunchKernel(r->fn_spatial_merge_contig_f32,
                           n_merged, 1, 1,
                           256, 1, 1,
                           0, r->stream,
                           args, NULL);
        } else {
            void *args[] = { &r->d_merge_buf, &r->d_hidden, &gw, &sm, &dim };
            hipModuleLaunchKernel(r->fn_spatial_merge_f32,
                           n_merged, 1, 1,
                           256, 1, 1,
                           0, r->stream,
                           args, NULL);
        }
    }

    /* 8. MM projection: mm.0 -> GELU -> mm.2 */
    fprintf(stderr, "  mm projection...\n");
    vlm_projector_gemm_gelu(r, r->d_mm_buf, &r->mm0, r->d_merge_buf, n_merged, merged_dim, merged_dim);

    vlm_gemm(r, r->d_mm_out, &r->mm2, r->d_mm_buf, n_merged, r->proj_dim, merged_dim);

    /* Debug: check mm_out */
    if (r->verbose >= 2) {
        hipDeviceSynchronize();
        float dbg[8];
        hipMemcpy(dbg, r->d_mm_out, 8 * sizeof(float), hipMemcpyDeviceToHost);
        fprintf(stderr, "  [DBG] mm_out: %.6f %.6f %.6f %.6f\n",
                dbg[0], dbg[1], dbg[2], dbg[3]);
        hipMemcpy(dbg, r->d_merge_buf, 8 * sizeof(float), hipMemcpyDeviceToHost);
        fprintf(stderr, "  [DBG] merge_buf: %.6f %.6f %.6f %.6f\n",
                dbg[0], dbg[1], dbg[2], dbg[3]);
        hipMemcpy(dbg, r->d_mm_buf, 8 * sizeof(float), hipMemcpyDeviceToHost);
        fprintf(stderr, "  [DBG] mm_buf (after gelu): %.6f %.6f %.6f %.6f\n",
                dbg[0], dbg[1], dbg[2], dbg[3]);
    }

    /* 9. Synchronize and copy results to host */
    hipDeviceSynchronize();

    int total_embd = r->proj_dim * (1 + r->n_deepstack);
    float *result = (float *)calloc(n_merged * total_embd, sizeof(float));

    /* Copy main embeddings */
    float *mm_host = (float *)malloc((size_t)n_merged * r->proj_dim * sizeof(float));
    hipMemcpy(mm_host, r->d_mm_out, (size_t)n_merged * r->proj_dim * sizeof(float), hipMemcpyDeviceToHost);

    /* Copy deepstack features */
    float *ds_host = NULL;
    if (ds_count > 0) {
        ds_host = (float *)malloc((size_t)ds_count * n_merged * r->proj_dim * sizeof(float));
        hipMemcpy(ds_host, r->d_ds_feats,
                      (size_t)ds_count * n_merged * r->proj_dim * sizeof(float), hipMemcpyDeviceToHost);
    }

    /* Interleave: [main, ds0, ds1, ...] per token */
    for (int t = 0; t < n_merged; t++) {
        float *dst = result + t * total_embd;
        memcpy(dst, mm_host + t * r->proj_dim, r->proj_dim * sizeof(float));
        for (int d = 0; d < ds_count; d++) {
            memcpy(dst + (1 + d) * r->proj_dim,
                   ds_host + d * n_merged * r->proj_dim + t * r->proj_dim,
                   r->proj_dim * sizeof(float));
        }
    }

    free(mm_host);
    free(ds_host);

    fprintf(stderr, "  vision encoding done: %d tokens of dim %d (main %d + %d deepstack)\n",
            n_merged, total_embd, r->proj_dim, ds_count);

    return result;
}

/* ======================================================================== */
/* Public API: accessors                                                    */
/* ======================================================================== */

int hip_vision_n_merged(const hip_vision_runner *r) {
    return r ? r->n_merged : 0;
}

int hip_vision_proj_dim(const hip_vision_runner *r) {
    return r ? r->proj_dim : 0;
}

int hip_vision_total_embd(const hip_vision_runner *r) {
    return r ? r->proj_dim * (1 + r->n_deepstack) : 0;
}

/* ======================================================================== */
/* Public API: free                                                         */
/* ======================================================================== */

static void vlm_free_weight(gpu_weight *w) {
    if (w->w_f32) hipFree(w->w_f32);
    if (w->w_f16) hipFree(w->w_f16);
    if (w->w_bf16) hipFree(w->w_bf16);
    if (w->bias) hipFree(w->bias);
    memset(w, 0, sizeof(*w));
}

void hip_vision_free(hip_vision_runner *r) {
    if (!r) return;

    vlm_hipblaslt_plan_destroy(r, &r->hipblaslt_mm0_gelu);
    for (int i = 0; i < r->hipblaslt_vit_n_plans; i++) {
        vlm_hipblaslt_plan_destroy(r, &r->hipblaslt_vit_plans[i]);
    }
    r->hipblaslt_vit_n_plans = 0;
    mm0_extracted_destroy();
    if (r->hipblaslt && r->hipblaslt_destroy) r->hipblaslt_destroy(r->hipblaslt);
    if (r->hipblaslt_lib) dlclose(r->hipblaslt_lib);
    if (r->rocblas && r->rocblas_destroy_handle) r->rocblas_destroy_handle(r->rocblas);
    if (r->rocblas_lib) dlclose(r->rocblas_lib);

    /* Free weights */
    if (r->d_patch_w0) hipFree(r->d_patch_w0);
    if (r->d_patch_w0_bf16) hipFree(r->d_patch_w0_bf16);
    if (r->d_patch_w0_f16) hipFree(r->d_patch_w0_f16);
    if (r->d_patch_unfold_bf16) hipFree(r->d_patch_unfold_bf16);
    if (r->d_patch_unfold_f16) hipFree(r->d_patch_unfold_f16);
    if (r->d_patch_w1) hipFree(r->d_patch_w1);
    if (r->d_patch_bias) hipFree(r->d_patch_bias);
    if (r->d_pos_embd) hipFree(r->d_pos_embd);

    if (r->blocks) {
        for (int l = 0; l < r->n_blocks; l++) {
            gpu_vit_block *blk = &r->blocks[l];
            vlm_free_weight(&blk->attn_qkv);
            vlm_free_weight(&blk->attn_out);
            vlm_free_weight(&blk->ffn_up);
            vlm_free_weight(&blk->ffn_down);
            if (blk->ln1_w) hipFree(blk->ln1_w);
            if (blk->ln1_b) hipFree(blk->ln1_b);
            if (blk->ln2_w) hipFree(blk->ln2_w);
            if (blk->ln2_b) hipFree(blk->ln2_b);
        }
        free(r->blocks);
    }

    if (r->deepstack) {
        for (int i = 0; i < r->n_deepstack; i++) {
            vlm_free_weight(&r->deepstack[i].fc1);
            vlm_free_weight(&r->deepstack[i].fc2);
            if (r->deepstack[i].norm_w) hipFree(r->deepstack[i].norm_w);
            if (r->deepstack[i].norm_b) hipFree(r->deepstack[i].norm_b);
        }
        free(r->deepstack);
    }
    free(r->deepstack_indices);

    if (r->d_post_ln_w) hipFree(r->d_post_ln_w);
    if (r->d_post_ln_b) hipFree(r->d_post_ln_b);
    vlm_free_weight(&r->mm0);
    vlm_free_weight(&r->mm2);

    /* Free scratch buffers */
    if (r->d_hidden) hipFree(r->d_hidden);
    if (r->d_hidden2) hipFree(r->d_hidden2);
    if (r->d_qkv) hipFree(r->d_qkv);
    if (r->d_kt) hipFree(r->d_kt);
    if (r->d_vt) hipFree(r->d_vt);
    if (r->d_kt_h) hipFree(r->d_kt_h);
    if (r->d_vt_h) hipFree(r->d_vt_h);
    if (r->d_attn_out) hipFree(r->d_attn_out);
    if (r->d_ffn_buf) hipFree(r->d_ffn_buf);
    if (r->d_ln_buf) hipFree(r->d_ln_buf);
    if (r->d_gemm_pack) hipFree(r->d_gemm_pack);
    if (r->d_merge_buf) hipFree(r->d_merge_buf);
    if (r->d_mm_buf) hipFree(r->d_mm_buf);
    if (r->d_mm_out) hipFree(r->d_mm_out);
    if (r->d_rgb) hipFree(r->d_rgb);
    if (r->d_rope_cos) hipFree(r->d_rope_cos);
    if (r->d_rope_sin) hipFree(r->d_rope_sin);
    if (r->d_pos_map) hipFree(r->d_pos_map);
    if (r->d_pos_interp) hipFree(r->d_pos_interp);
    if (r->d_ds_feats) hipFree(r->d_ds_feats);

    free(r->h_pos_embd);
    free(r->h_output);

    /* Destroy HIP objects */
    if (r->module) hipModuleUnload(r->module);
    if (r->stream) hipStreamDestroy(r->stream);
    if (r->context) hipCtxDestroy(r->context);

    free(r);
}
