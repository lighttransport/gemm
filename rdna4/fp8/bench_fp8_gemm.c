/*
 * bench_fp8_gemm.c - RDNA4 FP8 WMMA GEMM benchmark (gfx1201 / RX 9070 XT).
 *
 * Targets the v_wmma_f32_16x16x16_fp8_fp8 path. Inputs are e4m3 (HIP_R_8F_E4M3
 * representation), accumulator is FP32. Mirrors rdna4/vlm/bench_vlm_gemm
 * structure: HIPRTC-compiled kernels, FP32 reference computed from the
 * dequantized inputs so cosine is bounded by reduction-order noise, not by
 * fp8 quantization.
 *
 * Goal: ~90% of FP8 WMMA peak (~350 TFLOP/s) on the mm0 shape
 * (M=1024, N=4608, K=4608). See ../vlm/optimized-gemm-guide.md for the
 * BF16 lever attribution; the same playbook applies here.
 */

#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

#include "../rocew.h"

#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../hip_runner_common.h"

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

static const int g_num_shapes = (int)(sizeof(g_shapes) / sizeof(g_shapes[0]));

/* ------------------------------------------------------------------------ */
/* Host-side fp8 e4m3 dequant (must match hip_f32_to_fp8_e4m3 in
 * hip_runner_common.h). Used to build the FP32 reference. */

static float fp8_e4m3_to_f32(uint8_t v) {
    if (v == 0x00) return 0.0f;
    if (v == 0x80) return -0.0f;
    uint32_t sign = (v >> 7) & 1;
    uint32_t exp  = (v >> 3) & 0xF;
    uint32_t mant = v & 0x7;
    if (exp == 0xF && mant == 0x7) {
        /* E4M3 reserves S.1111.111 as NaN (no inf). */
        return NAN;
    }
    if (exp == 0) {
        /* subnormal: 2^(1-7) * (mant/8) */
        float f = ldexpf((float)mant / 8.0f, -6);
        return sign ? -f : f;
    }
    /* normal: 2^(exp-7) * (1 + mant/8) */
    float f = ldexpf(1.0f + (float)mant / 8.0f, (int)exp - 7);
    return sign ? -f : f;
}

/* ------------------------------------------------------------------------ */
/* Kernel source compiled by HIPRTC. Baseline FP8 WMMA: MT=128x128, K_step=32,
 * 4 waves per WG, MIWT 4x4 (each wave does 64M x 64N). LDS double-buffered
 * for next-K prefetch. */

static const char *kernel_src =
"typedef unsigned char u8;\n"
"typedef unsigned int  u32;\n"
"typedef u32 u32x2 __attribute__((ext_vector_type(2)));\n"
"typedef u8  u8x8 __attribute__((ext_vector_type(8)));\n"
"typedef float float8 __attribute__((ext_vector_type(8)));\n"
"\n"
"__device__ __forceinline__ u32x2 pack_a8(u8x8 v) {\n"
"    union { u8x8 b; u32x2 w; } u; u.b = v; return u.w;\n"
"}\n"
"\n"
"__device__ __forceinline__ void store_acc8_f32(float *Y, const float *bias,\n"
"                                                float8 acc, int row0, int col, int ld) {\n"
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
"\n"
"__device__ __forceinline__ void lds_barrier_signal() {\n"
"    __asm__ __volatile__(\"s_barrier_signal -1\" ::: \"memory\");\n"
"}\n"
"__device__ __forceinline__ void lds_barrier_wait() {\n"
"    __asm__ __volatile__(\"s_barrier_wait 0xffff\" ::: \"memory\");\n"
"}\n"
"__device__ __forceinline__ void lds_barrier() { lds_barrier_signal(); lds_barrier_wait(); }\n"
"\n"
"#define WMMA_FP8(A,B,C) C = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(A, B, C)\n"
"\n"
"extern \"C\" {\n"
"\n"
"/* Baseline kernel. NT layout: A is M x K row-major (X[m*K+k]),\n"
" * B is N x K row-major (W[n*K+k] — i.e. weight transposed once at load).\n"
" * Output Y is M x N row-major. K must be divisible by 32, M and N by 128. */\n"
"__global__ __launch_bounds__(128, 1)\n"
"void gemm_fp8_baseline(float *Y, const u8 *W, const u8 *X,\n"
"                       const float *bias, int N, int K, int M) {\n"
"    int tid = threadIdx.x;\n"
"    int wave_id = tid >> 5;\n"
"    int lane = tid & 31;\n"
"    int wM = wave_id & 1;            /* 0 or 1 */\n"
"    int wN = wave_id >> 1;           /* 0 or 1 */\n"
"    int half = lane >> 4;            /* 0 or 1 */\n"
"    int idx = lane & 15;\n"
"    int k_off = half * 8;            /* this lane carries K[k_off..k_off+7] */\n"
"    int cta_m0 = blockIdx.y * 128;\n"
"    int cta_n0 = blockIdx.x * 128;\n"
"\n"
"    /* LDS: 128 rows of K=32, packed as fp8. 4 KB each. */\n"
"    __shared__ u8 smA[128 * 32];\n"
"    __shared__ u8 smB[128 * 32];\n"
"\n"
"    float8 z = {0,0,0,0,0,0,0,0};\n"
"    float8 cv00=z,cv01=z,cv02=z,cv03=z;\n"
"    float8 cv10=z,cv11=z,cv12=z,cv13=z;\n"
"    float8 cv20=z,cv21=z,cv22=z,cv23=z;\n"
"    float8 cv30=z,cv31=z,cv32=z,cv33=z;\n"
"\n"
"    for (int k = 0; k < K; k += 32) {\n"
"        /* Cooperative load: 128*32 = 4096 fp8 per operand, 128 threads,\n"
"         * so each thread loads 32 fp8 (= 8 elts of 4 rows). */\n"
"        for (int it = 0; it < 32; it++) {\n"
"            int e = tid * 32 + it;       /* 0 .. 4095 */\n"
"            int er = e >> 5;             /* row 0..127 */\n"
"            int ek = e & 31;             /* k offset 0..31 */\n"
"            int row = cta_m0 + er, kp = k + ek;\n"
"            smA[e] = (row < M && kp < K) ? X[(size_t)row * K + kp] : (u8)0;\n"
"        }\n"
"        for (int it = 0; it < 32; it++) {\n"
"            int e = tid * 32 + it;\n"
"            int er = e >> 5;\n"
"            int ek = e & 31;\n"
"            int col = cta_n0 + er, kp = k + ek;\n"
"            smB[e] = (col < N && kp < K) ? W[(size_t)col * K + kp] : (u8)0;\n"
"        }\n"
"        __syncthreads();\n"
"\n"
"        int a_base = wM * 64;\n"
"        int b_base = wN * 64;\n"
"        for (int kk0 = 0; kk0 < 32; kk0 += 16) {\n"
"            u8x8 ra0,ra1,ra2,ra3,rb0,rb1,rb2,rb3;\n"
"            for (int i = 0; i < 8; i++) {\n"
"                ra0[i] = smA[(a_base+0 +idx)*32 + kk0 + k_off + i];\n"
"                ra1[i] = smA[(a_base+16+idx)*32 + kk0 + k_off + i];\n"
"                ra2[i] = smA[(a_base+32+idx)*32 + kk0 + k_off + i];\n"
"                ra3[i] = smA[(a_base+48+idx)*32 + kk0 + k_off + i];\n"
"                rb0[i] = smB[(b_base+0 +idx)*32 + kk0 + k_off + i];\n"
"                rb1[i] = smB[(b_base+16+idx)*32 + kk0 + k_off + i];\n"
"                rb2[i] = smB[(b_base+32+idx)*32 + kk0 + k_off + i];\n"
"                rb3[i] = smB[(b_base+48+idx)*32 + kk0 + k_off + i];\n"
"            }\n"
"            u32x2 a0 = pack_a8(ra0), a1 = pack_a8(ra1);\n"
"            u32x2 a2 = pack_a8(ra2), a3 = pack_a8(ra3);\n"
"            u32x2 b0 = pack_a8(rb0), b1 = pack_a8(rb1);\n"
"            u32x2 b2 = pack_a8(rb2), b3 = pack_a8(rb3);\n"
"            WMMA_FP8(a0, b0, cv00); WMMA_FP8(a0, b1, cv01);\n"
"            WMMA_FP8(a0, b2, cv02); WMMA_FP8(a0, b3, cv03);\n"
"            WMMA_FP8(a1, b0, cv10); WMMA_FP8(a1, b1, cv11);\n"
"            WMMA_FP8(a1, b2, cv12); WMMA_FP8(a1, b3, cv13);\n"
"            WMMA_FP8(a2, b0, cv20); WMMA_FP8(a2, b1, cv21);\n"
"            WMMA_FP8(a2, b2, cv22); WMMA_FP8(a2, b3, cv23);\n"
"            WMMA_FP8(a3, b0, cv30); WMMA_FP8(a3, b1, cv31);\n"
"            WMMA_FP8(a3, b2, cv32); WMMA_FP8(a3, b3, cv33);\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"\n"
"    int wave_m0 = cta_m0 + wM * 64;\n"
"    int wave_n0 = cta_n0 + wN * 64;\n"
"    int row = wave_m0 + half * 8;\n"
"    /* Each WMMA writes 16 M-rows (per half) x 16 N-cols.\n"
"     * Half=0 owns rows [0..7], half=1 owns rows [8..15] within each 16-row\n"
"     * tile (8 fp32 per lane in float8 acc). */\n"
"    if (row < M) {\n"
"        store_acc8_f32(Y, bias, cv00, row +  0, wave_n0 +  0 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv01, row +  0, wave_n0 + 16 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv02, row +  0, wave_n0 + 32 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv03, row +  0, wave_n0 + 48 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv10, row + 16, wave_n0 +  0 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv11, row + 16, wave_n0 + 16 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv12, row + 16, wave_n0 + 32 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv13, row + 16, wave_n0 + 48 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv20, row + 32, wave_n0 +  0 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv21, row + 32, wave_n0 + 16 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv22, row + 32, wave_n0 + 32 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv23, row + 32, wave_n0 + 48 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv30, row + 48, wave_n0 +  0 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv31, row + 48, wave_n0 + 16 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv32, row + 48, wave_n0 + 32 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv33, row + 48, wave_n0 + 48 + idx, N);\n"
"    }\n"
"}\n"
"\n"
"\n"
"/* Pipelined kernel: MT=128x128, K_step=64, MIWT 4x4, double-buffered LDS,\n"
" * vectorized 16-byte global loads, next-K prefetch held in registers across\n"
" * the WMMA body (PGR=1 effective). 4 waves per WG (128 threads).\n"
" *\n"
" * LDS layout (per buffer, per operand): u8x8 sm[8 K-chunks][128 rows]. Each\n"
" * u8x8 = 8 fp8 covering one K=8 fragment of one row. K_step=64 ⇒ 8 chunks. */\n"
"\n"
"typedef u8 u8x16 __attribute__((ext_vector_type(16)));\n"
"\n"
"__global__ __launch_bounds__(128, 1)\n"
"void gemm_fp8_pipe(float *Y, const u8 *W, const u8 *X,\n"
"                   const float *bias, int N, int K, int M) {\n"
"    int tid = threadIdx.x;\n"
"    int wave_id = tid >> 5;\n"
"    int lane = tid & 31;\n"
"    int wM = wave_id & 1;\n"
"    int wN = wave_id >> 1;\n"
"    int half = lane >> 4;\n"
"    int idx = lane & 15;\n"
"    int k_off = half * 8;             /* 0 or 8 */\n"
"    int cta_m0 = blockIdx.y * 128;\n"
"    int cta_n0 = blockIdx.x * 128;\n"
"\n"
"    /* Each thread cooperatively loads 4 u8x16 = 64 fp8 per operand per K-step.\n"
"     * Layout: 128 threads × 4 = 512 u8x16 = 8192 fp8 = 128 rows × 64 K. */\n"
"    /* e ∈ [0,512) covers (row=e>>2, k_off_in_step=(e&3)*16). */\n"
"    int e0 = tid * 4 + 0;\n"
"    int e1 = tid * 4 + 1;\n"
"    int e2 = tid * 4 + 2;\n"
"    int e3 = tid * 4 + 3;\n"
"    int r0 = e0 >> 2, r1 = e1 >> 2, r2 = e2 >> 2, r3 = e3 >> 2;\n"
"    int c0 = (e0 & 3) * 16, c1 = (e1 & 3) * 16;\n"
"    int c2 = (e2 & 3) * 16, c3 = (e3 & 3) * 16;\n"
"\n"
"    /* LDS double-buffer. Stored as u8x8 (8 fp8 per WMMA frag). */\n"
"    __shared__ u8x8 smA8[2 * 128 * 8];\n"
"    __shared__ u8x8 smB8[2 * 128 * 8];\n"
"\n"
"    /* Stage 0 load (k=0). u8x16 split into two u8x8 halves at LDS write. */\n"
"    u8x16 a0v = *((const u8x16 *)(X + (size_t)(cta_m0 + r0) * K + c0));\n"
"    u8x16 a1v = *((const u8x16 *)(X + (size_t)(cta_m0 + r1) * K + c1));\n"
"    u8x16 a2v = *((const u8x16 *)(X + (size_t)(cta_m0 + r2) * K + c2));\n"
"    u8x16 a3v = *((const u8x16 *)(X + (size_t)(cta_m0 + r3) * K + c3));\n"
"    u8x16 b0v = *((const u8x16 *)(W + (size_t)(cta_n0 + r0) * K + c0));\n"
"    u8x16 b1v = *((const u8x16 *)(W + (size_t)(cta_n0 + r1) * K + c1));\n"
"    u8x16 b2v = *((const u8x16 *)(W + (size_t)(cta_n0 + r2) * K + c2));\n"
"    u8x16 b3v = *((const u8x16 *)(W + (size_t)(cta_n0 + r3) * K + c3));\n"
"    /* Split u8x16 into two u8x8 (low half, high half) and write both K-chunks. */\n"
"    #define SPLIT_HI(v) ((u8x8){v[8],v[9],v[10],v[11],v[12],v[13],v[14],v[15]})\n"
"    #define SPLIT_LO(v) ((u8x8){v[0],v[1],v[2], v[3], v[4], v[5], v[6], v[7]})\n"
"    /* Each thread writes 4 entries × 2 halves = 8 u8x8 per operand for stage 0.\n"
"     * Layout: smA8[chunk * 128 + row]. chunk derived from c (low half = c/8,\n"
"     * high half = c/8 + 1). */\n"
"    smA8[(c0>>3) * 128 + r0]    = SPLIT_LO(a0v);\n"
"    smA8[((c0>>3)+1)*128 + r0]  = SPLIT_HI(a0v);\n"
"    smA8[(c1>>3) * 128 + r1]    = SPLIT_LO(a1v);\n"
"    smA8[((c1>>3)+1)*128 + r1]  = SPLIT_HI(a1v);\n"
"    smA8[(c2>>3) * 128 + r2]    = SPLIT_LO(a2v);\n"
"    smA8[((c2>>3)+1)*128 + r2]  = SPLIT_HI(a2v);\n"
"    smA8[(c3>>3) * 128 + r3]    = SPLIT_LO(a3v);\n"
"    smA8[((c3>>3)+1)*128 + r3]  = SPLIT_HI(a3v);\n"
"    smB8[(c0>>3) * 128 + r0]    = SPLIT_LO(b0v);\n"
"    smB8[((c0>>3)+1)*128 + r0]  = SPLIT_HI(b0v);\n"
"    smB8[(c1>>3) * 128 + r1]    = SPLIT_LO(b1v);\n"
"    smB8[((c1>>3)+1)*128 + r1]  = SPLIT_HI(b1v);\n"
"    smB8[(c2>>3) * 128 + r2]    = SPLIT_LO(b2v);\n"
"    smB8[((c2>>3)+1)*128 + r2]  = SPLIT_HI(b2v);\n"
"    smB8[(c3>>3) * 128 + r3]    = SPLIT_LO(b3v);\n"
"    smB8[((c3>>3)+1)*128 + r3]  = SPLIT_HI(b3v);\n"
"    lds_barrier();\n"
"\n"
"    float8 z = {0,0,0,0,0,0,0,0};\n"
"    float8 cv00=z,cv01=z,cv02=z,cv03=z;\n"
"    float8 cv10=z,cv11=z,cv12=z,cv13=z;\n"
"    float8 cv20=z,cv21=z,cv22=z,cv23=z;\n"
"    float8 cv30=z,cv31=z,cv32=z,cv33=z;\n"
"\n"
"    int buf = 0;\n"
"    for (int k = 0; k < K; k += 64) {\n"
"        u8x16 na0,na1,na2,na3,nb0,nb1,nb2,nb3;\n"
"        int has_next = (k + 64 < K);\n"
"        if (has_next) {\n"
"            int nk = k + 64;\n"
"            na0 = *((const u8x16 *)(X + (size_t)(cta_m0 + r0) * K + nk + c0));\n"
"            na1 = *((const u8x16 *)(X + (size_t)(cta_m0 + r1) * K + nk + c1));\n"
"            na2 = *((const u8x16 *)(X + (size_t)(cta_m0 + r2) * K + nk + c2));\n"
"            na3 = *((const u8x16 *)(X + (size_t)(cta_m0 + r3) * K + nk + c3));\n"
"            nb0 = *((const u8x16 *)(W + (size_t)(cta_n0 + r0) * K + nk + c0));\n"
"            nb1 = *((const u8x16 *)(W + (size_t)(cta_n0 + r1) * K + nk + c1));\n"
"            nb2 = *((const u8x16 *)(W + (size_t)(cta_n0 + r2) * K + nk + c2));\n"
"            nb3 = *((const u8x16 *)(W + (size_t)(cta_n0 + r3) * K + nk + c3));\n"
"        }\n"
"        int base = buf * (128 * 8);\n"
"        int a_base = wM * 64;\n"
"        int b_base = wN * 64;\n"
"        for (int kk0 = 0; kk0 < 64; kk0 += 16) {\n"
"            int kslot = (kk0 + k_off) >> 3;\n"
"            u8x8 a0 = smA8[base + kslot * 128 + (a_base + 0  + idx)];\n"
"            u8x8 a1 = smA8[base + kslot * 128 + (a_base + 16 + idx)];\n"
"            u8x8 a2 = smA8[base + kslot * 128 + (a_base + 32 + idx)];\n"
"            u8x8 a3 = smA8[base + kslot * 128 + (a_base + 48 + idx)];\n"
"            u8x8 b0 = smB8[base + kslot * 128 + (b_base + 0  + idx)];\n"
"            u8x8 b1 = smB8[base + kslot * 128 + (b_base + 16 + idx)];\n"
"            u8x8 b2 = smB8[base + kslot * 128 + (b_base + 32 + idx)];\n"
"            u8x8 b3 = smB8[base + kslot * 128 + (b_base + 48 + idx)];\n"
"            u32x2 ra0 = pack_a8(a0), ra1 = pack_a8(a1);\n"
"            u32x2 ra2 = pack_a8(a2), ra3 = pack_a8(a3);\n"
"            u32x2 rb0 = pack_a8(b0), rb1 = pack_a8(b1);\n"
"            u32x2 rb2 = pack_a8(b2), rb3 = pack_a8(b3);\n"
"            WMMA_FP8(ra0, rb0, cv00); WMMA_FP8(ra0, rb1, cv01);\n"
"            WMMA_FP8(ra0, rb2, cv02); WMMA_FP8(ra0, rb3, cv03);\n"
"            WMMA_FP8(ra1, rb0, cv10); WMMA_FP8(ra1, rb1, cv11);\n"
"            WMMA_FP8(ra1, rb2, cv12); WMMA_FP8(ra1, rb3, cv13);\n"
"            WMMA_FP8(ra2, rb0, cv20); WMMA_FP8(ra2, rb1, cv21);\n"
"            WMMA_FP8(ra2, rb2, cv22); WMMA_FP8(ra2, rb3, cv23);\n"
"            WMMA_FP8(ra3, rb0, cv30); WMMA_FP8(ra3, rb1, cv31);\n"
"            WMMA_FP8(ra3, rb2, cv32); WMMA_FP8(ra3, rb3, cv33);\n"
"        }\n"
"        if (has_next) {\n"
"            int nb = 1 - buf;\n"
"            int nbase = nb * (128 * 8);\n"
"            smA8[nbase + (c0>>3)*128 + r0]     = SPLIT_LO(na0);\n"
"            smA8[nbase + ((c0>>3)+1)*128 + r0] = SPLIT_HI(na0);\n"
"            smA8[nbase + (c1>>3)*128 + r1]     = SPLIT_LO(na1);\n"
"            smA8[nbase + ((c1>>3)+1)*128 + r1] = SPLIT_HI(na1);\n"
"            smA8[nbase + (c2>>3)*128 + r2]     = SPLIT_LO(na2);\n"
"            smA8[nbase + ((c2>>3)+1)*128 + r2] = SPLIT_HI(na2);\n"
"            smA8[nbase + (c3>>3)*128 + r3]     = SPLIT_LO(na3);\n"
"            smA8[nbase + ((c3>>3)+1)*128 + r3] = SPLIT_HI(na3);\n"
"            smB8[nbase + (c0>>3)*128 + r0]     = SPLIT_LO(nb0);\n"
"            smB8[nbase + ((c0>>3)+1)*128 + r0] = SPLIT_HI(nb0);\n"
"            smB8[nbase + (c1>>3)*128 + r1]     = SPLIT_LO(nb1);\n"
"            smB8[nbase + ((c1>>3)+1)*128 + r1] = SPLIT_HI(nb1);\n"
"            smB8[nbase + (c2>>3)*128 + r2]     = SPLIT_LO(nb2);\n"
"            smB8[nbase + ((c2>>3)+1)*128 + r2] = SPLIT_HI(nb2);\n"
"            smB8[nbase + (c3>>3)*128 + r3]     = SPLIT_LO(nb3);\n"
"            smB8[nbase + ((c3>>3)+1)*128 + r3] = SPLIT_HI(nb3);\n"
"            lds_barrier_signal();\n"
"            buf = nb;\n"
"            lds_barrier_wait();\n"
"        }\n"
"    }\n"
"    int wave_m0 = cta_m0 + wM * 64;\n"
"    int wave_n0 = cta_n0 + wN * 64;\n"
"    int row = wave_m0 + half * 8;\n"
"    if (row < M) {\n"
"        store_acc8_f32(Y, bias, cv00, row +  0, wave_n0 +  0 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv01, row +  0, wave_n0 + 16 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv02, row +  0, wave_n0 + 32 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv03, row +  0, wave_n0 + 48 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv10, row + 16, wave_n0 +  0 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv11, row + 16, wave_n0 + 16 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv12, row + 16, wave_n0 + 32 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv13, row + 16, wave_n0 + 48 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv20, row + 32, wave_n0 +  0 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv21, row + 32, wave_n0 + 16 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv22, row + 32, wave_n0 + 32 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv23, row + 32, wave_n0 + 48 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv30, row + 48, wave_n0 +  0 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv31, row + 48, wave_n0 + 16 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv32, row + 48, wave_n0 + 32 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv33, row + 48, wave_n0 + 48 + idx, N);\n"
"    }\n"
"}\n"
"\n"
"\n"
"/* K_step=32 pipelined kernel — mirror of BF16 winning structure (mm0pipe4w).\n"
" * MT=128x128, MIWT 4x4, double-buffered LDS, 4 K-chunks/buffer (K=8 each).\n"
" * Each thread loads 2 u8x16 = 32 fp8 per operand per K-step (4 b128 total). */\n"
"__global__ __launch_bounds__(128, 1)\n"
"void gemm_fp8_pipe32(float *Y, const u8 *W, const u8 *X,\n"
"                     const float *bias, int N, int K, int M) {\n"
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
"\n"
"    /* 128 threads × 2 b128 = 256 b128 = 4096 fp8 = 128 rows × 32 K. */\n"
"    int e0 = tid * 2 + 0;\n"
"    int e1 = tid * 2 + 1;\n"
"    int r0 = e0 >> 1, r1 = e1 >> 1;\n"
"    int c0 = (e0 & 1) * 16, c1 = (e1 & 1) * 16;\n"
"\n"
"    /* LDS: 4 chunks × 128 rows × u8x8 = 4 KB per buffer per operand. */\n"
"    __shared__ u8x8 smA8[2 * 128 * 4];\n"
"    __shared__ u8x8 smB8[2 * 128 * 4];\n"
"\n"
"    #define SPLIT_HI(v) ((u8x8){v[8],v[9],v[10],v[11],v[12],v[13],v[14],v[15]})\n"
"    #define SPLIT_LO(v) ((u8x8){v[0],v[1],v[2], v[3], v[4], v[5], v[6], v[7]})\n"
"\n"
"    /* Stage 0 */\n"
"    {\n"
"        u8x16 a0v = *((const u8x16 *)(X + (size_t)(cta_m0 + r0) * K + c0));\n"
"        u8x16 a1v = *((const u8x16 *)(X + (size_t)(cta_m0 + r1) * K + c1));\n"
"        u8x16 b0v = *((const u8x16 *)(W + (size_t)(cta_n0 + r0) * K + c0));\n"
"        u8x16 b1v = *((const u8x16 *)(W + (size_t)(cta_n0 + r1) * K + c1));\n"
"        smA8[(c0>>3) * 128 + r0]    = SPLIT_LO(a0v);\n"
"        smA8[((c0>>3)+1)*128 + r0]  = SPLIT_HI(a0v);\n"
"        smA8[(c1>>3) * 128 + r1]    = SPLIT_LO(a1v);\n"
"        smA8[((c1>>3)+1)*128 + r1]  = SPLIT_HI(a1v);\n"
"        smB8[(c0>>3) * 128 + r0]    = SPLIT_LO(b0v);\n"
"        smB8[((c0>>3)+1)*128 + r0]  = SPLIT_HI(b0v);\n"
"        smB8[(c1>>3) * 128 + r1]    = SPLIT_LO(b1v);\n"
"        smB8[((c1>>3)+1)*128 + r1]  = SPLIT_HI(b1v);\n"
"    }\n"
"    lds_barrier();\n"
"\n"
"    float8 z = {0,0,0,0,0,0,0,0};\n"
"    float8 cv00=z,cv01=z,cv02=z,cv03=z;\n"
"    float8 cv10=z,cv11=z,cv12=z,cv13=z;\n"
"    float8 cv20=z,cv21=z,cv22=z,cv23=z;\n"
"    float8 cv30=z,cv31=z,cv32=z,cv33=z;\n"
"\n"
"    int buf = 0;\n"
"    for (int k = 0; k < K; k += 32) {\n"
"        u8x16 na0,na1,nb0,nb1;\n"
"        int has_next = (k + 32 < K);\n"
"        if (has_next) {\n"
"            int nk = k + 32;\n"
"            na0 = *((const u8x16 *)(X + (size_t)(cta_m0 + r0) * K + nk + c0));\n"
"            na1 = *((const u8x16 *)(X + (size_t)(cta_m0 + r1) * K + nk + c1));\n"
"            nb0 = *((const u8x16 *)(W + (size_t)(cta_n0 + r0) * K + nk + c0));\n"
"            nb1 = *((const u8x16 *)(W + (size_t)(cta_n0 + r1) * K + nk + c1));\n"
"        }\n"
"        int base = buf * (128 * 4);\n"
"        int a_base = wM * 64;\n"
"        int b_base = wN * 64;\n"
"        for (int kk0 = 0; kk0 < 32; kk0 += 16) {\n"
"            int kslot = (kk0 + k_off) >> 3;\n"
"            u8x8 a0 = smA8[base + kslot * 128 + (a_base + 0  + idx)];\n"
"            u8x8 a1 = smA8[base + kslot * 128 + (a_base + 16 + idx)];\n"
"            u8x8 a2 = smA8[base + kslot * 128 + (a_base + 32 + idx)];\n"
"            u8x8 a3 = smA8[base + kslot * 128 + (a_base + 48 + idx)];\n"
"            u8x8 b0 = smB8[base + kslot * 128 + (b_base + 0  + idx)];\n"
"            u8x8 b1 = smB8[base + kslot * 128 + (b_base + 16 + idx)];\n"
"            u8x8 b2 = smB8[base + kslot * 128 + (b_base + 32 + idx)];\n"
"            u8x8 b3 = smB8[base + kslot * 128 + (b_base + 48 + idx)];\n"
"            u32x2 ra0 = pack_a8(a0), ra1 = pack_a8(a1);\n"
"            u32x2 ra2 = pack_a8(a2), ra3 = pack_a8(a3);\n"
"            u32x2 rb0 = pack_a8(b0), rb1 = pack_a8(b1);\n"
"            u32x2 rb2 = pack_a8(b2), rb3 = pack_a8(b3);\n"
"            WMMA_FP8(ra0, rb0, cv00); WMMA_FP8(ra0, rb1, cv01);\n"
"            WMMA_FP8(ra0, rb2, cv02); WMMA_FP8(ra0, rb3, cv03);\n"
"            WMMA_FP8(ra1, rb0, cv10); WMMA_FP8(ra1, rb1, cv11);\n"
"            WMMA_FP8(ra1, rb2, cv12); WMMA_FP8(ra1, rb3, cv13);\n"
"            WMMA_FP8(ra2, rb0, cv20); WMMA_FP8(ra2, rb1, cv21);\n"
"            WMMA_FP8(ra2, rb2, cv22); WMMA_FP8(ra2, rb3, cv23);\n"
"            WMMA_FP8(ra3, rb0, cv30); WMMA_FP8(ra3, rb1, cv31);\n"
"            WMMA_FP8(ra3, rb2, cv32); WMMA_FP8(ra3, rb3, cv33);\n"
"        }\n"
"        if (has_next) {\n"
"            int nb = 1 - buf;\n"
"            int nbase = nb * (128 * 4);\n"
"            smA8[nbase + (c0>>3)*128 + r0]     = SPLIT_LO(na0);\n"
"            smA8[nbase + ((c0>>3)+1)*128 + r0] = SPLIT_HI(na0);\n"
"            smA8[nbase + (c1>>3)*128 + r1]     = SPLIT_LO(na1);\n"
"            smA8[nbase + ((c1>>3)+1)*128 + r1] = SPLIT_HI(na1);\n"
"            smB8[nbase + (c0>>3)*128 + r0]     = SPLIT_LO(nb0);\n"
"            smB8[nbase + ((c0>>3)+1)*128 + r0] = SPLIT_HI(nb0);\n"
"            smB8[nbase + (c1>>3)*128 + r1]     = SPLIT_LO(nb1);\n"
"            smB8[nbase + ((c1>>3)+1)*128 + r1] = SPLIT_HI(nb1);\n"
"            lds_barrier_signal();\n"
"            buf = nb;\n"
"            lds_barrier_wait();\n"
"        }\n"
"    }\n"
"    int wave_m0 = cta_m0 + wM * 64;\n"
"    int wave_n0 = cta_n0 + wN * 64;\n"
"    int row = wave_m0 + half * 8;\n"
"    if (row < M) {\n"
"        store_acc8_f32(Y, bias, cv00, row +  0, wave_n0 +  0 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv01, row +  0, wave_n0 + 16 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv02, row +  0, wave_n0 + 32 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv03, row +  0, wave_n0 + 48 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv10, row + 16, wave_n0 +  0 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv11, row + 16, wave_n0 + 16 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv12, row + 16, wave_n0 + 32 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv13, row + 16, wave_n0 + 48 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv20, row + 32, wave_n0 +  0 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv21, row + 32, wave_n0 + 16 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv22, row + 32, wave_n0 + 32 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv23, row + 32, wave_n0 + 48 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv30, row + 48, wave_n0 +  0 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv31, row + 48, wave_n0 + 16 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv32, row + 48, wave_n0 + 32 + idx, N);\n"
"        store_acc8_f32(Y, bias, cv33, row + 48, wave_n0 + 48 + idx, N);\n"
"    }\n"
"}\n"
"\n"
"} /* extern C */\n";

/* ------------------------------------------------------------------------ */
/* CPU reference: Y = X (M x K) * W^T (K x N) + bias[N], using fp8-dequant inputs.
 * Slow but only used at --check time on a single dispatch. */

static void gemm_ref_fp32(float *Y, const uint8_t *W, const uint8_t *X,
                          const float *bias, int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                float xv = fp8_e4m3_to_f32(X[(size_t)m * K + k]);
                float wv = fp8_e4m3_to_f32(W[(size_t)n * K + k]);
                acc += xv * wv;
            }
            float bv = bias ? bias[n] : 0.0f;
            Y[(size_t)m * N + n] = acc + bv;
        }
    }
}

/* ------------------------------------------------------------------------ */

static double timer_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static double cosine_sim(const float *a, const float *b, size_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        na  += (double)a[i] * (double)a[i];
        nb  += (double)b[i] * (double)b[i];
    }
    if (na == 0.0 || nb == 0.0) return 0.0;
    return dot / sqrt(na * nb);
}

static float max_abs_diff(const float *a, const float *b, size_t n) {
    float mx = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

/* ------------------------------------------------------------------------ */

typedef struct {
    hipFunction_t baseline;
    hipFunction_t pipe;
    hipFunction_t pipe32;
} fp8_kernels;

static int load_kernels(int dev, fp8_kernels *out, int verbose) {
    hipModule_t mod;
    if (hip_compile_kernels(&mod, dev, kernel_src, "fp8_gemm", verbose, "fp8") < 0)
        return -1;
    HIP_CHECK(hipModuleGetFunction(&out->baseline, mod, "gemm_fp8_baseline"));
    HIP_CHECK(hipModuleGetFunction(&out->pipe,     mod, "gemm_fp8_pipe"));
    HIP_CHECK(hipModuleGetFunction(&out->pipe32,   mod, "gemm_fp8_pipe32"));
    return 0;
}

/* Quantize a [-A,A] uniform random fp32 buffer to fp8 e4m3. We pre-clamp to
 * the e4m3 representable range to avoid NaN results for stress-test ranges. */
static void fill_fp8_random(uint8_t *dst, size_t n, float abs_max, unsigned *rng) {
    for (size_t i = 0; i < n; i++) {
        unsigned r = *rng = *rng * 1103515245u + 12345u;
        float u = ((int)(r & 0xFFFFFF) - 0x800000) / (float)0x800000; /* [-1,1) */
        float v = u * abs_max;
        if (v >  448.0f) v =  448.0f;
        if (v < -448.0f) v = -448.0f;
        dst[i] = hip_f32_to_fp8_e4m3(v);
    }
}

static int run_shape(const fp8_kernels *kn, const gemm_shape *s,
                     int iters, int do_check, int use_bias, float abs_max,
                     const char *mode) {
    int M = s->m, N = s->n, K = s->k;
    int needK = (!strcmp(mode, "pipe")) ? 64 : 32;
    if (M % 128 || N % 128 || K % needK) {
        fprintf(stderr, "skip %s: M/N must be %%128, K %%%d (got %d,%d,%d)\n",
                s->name, needK, M, N, K);
        return 0;
    }
    hipFunction_t fn;
    if      (!strcmp(mode, "pipe"))   fn = kn->pipe;
    else if (!strcmp(mode, "pipe32")) fn = kn->pipe32;
    else                              fn = kn->baseline;

    size_t bytes_X = (size_t)M * K;
    size_t bytes_W = (size_t)N * K;
    size_t bytes_Y = (size_t)M * N * sizeof(float);

    uint8_t *hX = (uint8_t *)malloc(bytes_X);
    uint8_t *hW = (uint8_t *)malloc(bytes_W);
    float   *hB = use_bias ? (float *)malloc((size_t)N * sizeof(float)) : NULL;

    unsigned rng = 0x1234abcd;
    fill_fp8_random(hX, bytes_X, abs_max, &rng);
    fill_fp8_random(hW, bytes_W, abs_max, &rng);
    if (hB) {
        for (int i = 0; i < N; i++) {
            unsigned r = rng = rng * 1103515245u + 12345u;
            float u = ((int)(r & 0xFFFFFF) - 0x800000) / (float)0x800000;
            hB[i] = u * 0.1f;
        }
    }

    void *dX = hip_upload_raw(hX, bytes_X);
    void *dW = hip_upload_raw(hW, bytes_W);
    void *dB = hB ? hip_upload_raw(hB, (size_t)N * sizeof(float)) : NULL;
    void *dY = NULL; HIP_CHECK(hipMalloc(&dY, bytes_Y));

    /* Launch params: 128 threads/WG, grid (N/128, M/128). */
    dim3 block = {128, 1, 1};
    dim3 grid  = {(unsigned)(N / 128), (unsigned)(M / 128), 1};

    void *args[] = { &dY, &dW, &dX, &dB, &N, &K, &M };

    /* Warmup + check */
    HIP_CHECK(hipMemset(dY, 0, bytes_Y));
    HIP_CHECK(hipModuleLaunchKernel(fn,
                                    grid.x, grid.y, grid.z,
                                    block.x, block.y, block.z,
                                    0, NULL, args, NULL));
    HIP_CHECK(hipDeviceSynchronize());

    double cos = -2.0;
    float maxd = -1.0f;
    if (do_check) {
        float *hY = (float *)malloc(bytes_Y);
        float *hRef = (float *)malloc(bytes_Y);
        HIP_CHECK(hipMemcpy(hY, dY, bytes_Y, hipMemcpyDeviceToHost));
        gemm_ref_fp32(hRef, hW, hX, hB, M, N, K);
        cos  = cosine_sim(hY, hRef, (size_t)M * N);
        maxd = max_abs_diff(hY, hRef, (size_t)M * N);
        free(hY); free(hRef);
    }

    /* Time */
    double t0 = timer_ms();
    for (int i = 0; i < iters; i++) {
        HIP_CHECK(hipModuleLaunchKernel(fn,
                                        grid.x, grid.y, grid.z,
                                        block.x, block.y, block.z,
                                        0, NULL, args, NULL));
    }
    HIP_CHECK(hipDeviceSynchronize());
    double t1 = timer_ms();
    double ms = (t1 - t0) / (double)iters;
    double tflops = (2.0 * (double)M * (double)N * (double)K) / (ms * 1.0e-3) * 1.0e-12;

    printf("  [%-8s] %-9s M=%4d N=%4d K=%4d  %7.4f ms  %6.1f TFLOP/s",
           mode, s->name, M, N, K, ms, tflops);
    if (do_check) printf("  cos=%.6f  maxd=%.4f", cos, maxd);
    printf("\n");

    free(hX); free(hW); free(hB);
    hipFree(dX); hipFree(dW); if (dB) hipFree(dB); hipFree(dY);
    return 0;
}

/* ------------------------------------------------------------------------ */

static void usage(const char *prog) {
    fprintf(stderr,
        "usage: %s [--shape NAME] [--iters N] [--check] [--no-bias]\n"
        "          [--abs-max V] [--verbose]\n"
        "  --shape  qkv|attn_out|ffn_up|ffn_down|mm0|mm2|all  (default: mm0)\n"
        "  --iters  iterations to time (default: 200)\n"
        "  --check  validate vs FP32 reference (slow; only first dispatch)\n"
        "  --no-bias   omit bias add\n"
        "  --abs-max V  clamp random fp32 inputs to ±V before fp8 quant (default 1.0)\n",
        prog);
}

int main(int argc, char **argv) {
    const char *want = "mm0";
    const char *mode = "all";
    int iters = 200;
    int do_check = 0;
    int use_bias = 1;
    float abs_max = 1.0f;
    int verbose = 1;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--shape") && i + 1 < argc) want = argv[++i];
        else if (!strcmp(argv[i], "--mode") && i + 1 < argc) mode = argv[++i];
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--check")) do_check = 1;
        else if (!strcmp(argv[i], "--no-bias")) use_bias = 0;
        else if (!strcmp(argv[i], "--abs-max") && i + 1 < argc) abs_max = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--verbose")) verbose = 2;
        else { usage(argv[0]); return 1; }
    }

    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) {
        fprintf(stderr, "failed to load HIP runtime\n");
        return 1;
    }
    int dev = 0;
    HIP_CHECK(hipSetDevice(dev));

    fp8_kernels kn;
    if (load_kernels(dev, &kn, verbose) < 0) return 1;

    printf("rdna4/fp8 WMMA GEMM bench  (iters=%d, abs_max=%.2f, bias=%s%s)\n",
           iters, abs_max, use_bias ? "on" : "off",
           do_check ? ", check=on" : "");

    const char *modes_all[] = {"baseline", "pipe", "pipe32"};
    int n_modes = !strcmp(mode, "all") ? 3 : 1;
    const char *modes[3];
    if (n_modes == 1) modes[0] = mode;
    else { modes[0] = modes_all[0]; modes[1] = modes_all[1]; modes[2] = modes_all[2]; }

    int matched = 0;
    for (int i = 0; i < g_num_shapes; i++) {
        if (strcmp(want, "all") && strcmp(want, g_shapes[i].name)) continue;
        for (int mi = 0; mi < n_modes; mi++)
            run_shape(&kn, &g_shapes[i], iters, do_check, use_bias, abs_max, modes[mi]);
        matched++;
    }
    if (!matched) {
        fprintf(stderr, "no shape matches '%s'\n", want);
        return 1;
    }
    return 0;
}
