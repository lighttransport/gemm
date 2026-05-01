/*
 * bench_fp8_fa.c - RDNA4 FP8 WMMA flash-attention benchmark (gfx1201).
 *
 * Mirrors flash_attn_wmma_bf16_4w_pre from rdna4/vlm: BQ=64 4-wave layout,
 * MFMA replaced by v_wmma_f32_16x16x16_fp8_fp8. Q/K/V are FP8 e4m3, output
 * FP32. Softmax accumulates in FP32; P is rebiased to FP8 for the second
 * WMMA. K_t and V_t are pre-packed as [n_heads, n_tok, head_dim].
 *
 * Reference: rdna4/vlm/hip_vision_encoder.c flash_attn_wmma_bf16_4w_pre.
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

/* ------------------------------------------------------------------------ */
/* Host-side fp8 e4m3 dequant (matches hip_f32_to_fp8_e4m3). */

static float fp8_e4m3_to_f32(uint8_t v) {
    if (v == 0x00) return 0.0f;
    if (v == 0x80) return -0.0f;
    uint32_t sign = (v >> 7) & 1;
    uint32_t exp  = (v >> 3) & 0xF;
    uint32_t mant = v & 0x7;
    if (exp == 0xF && mant == 0x7) return NAN;
    if (exp == 0) {
        float f = ldexpf((float)mant / 8.0f, -6);
        return sign ? -f : f;
    }
    float f = ldexpf(1.0f + (float)mant / 8.0f, (int)exp - 7);
    return sign ? -f : f;
}

/* ------------------------------------------------------------------------ */
/* Kernel source — HIPRTC-compiled. BQ=64 4-wave FP8 FA. */

static const char *kernel_src =
"typedef unsigned char u8;\n"
"typedef unsigned int  u32;\n"
"typedef u32 u32x2 __attribute__((ext_vector_type(2)));\n"
"typedef u8  u8x8  __attribute__((ext_vector_type(8)));\n"
"typedef float float8 __attribute__((ext_vector_type(8)));\n"
"\n"
"__device__ __forceinline__ u32x2 pack_u8x8(u8x8 v) {\n"
"    union { u8x8 b; u32x2 w; } u; u.b = v; return u.w;\n"
"}\n"
"\n"
"/* FP32 -> FP8 e4m3 (round-to-nearest-even on top 3 mantissa bits, no NaN). */\n"
"__device__ __forceinline__ u8 f32_to_fp8_e4m3(float f) {\n"
"    if (f == 0.0f) return 0;\n"
"    union { float f; u32 i; } u; u.f = f;\n"
"    u32 bits = u.i;\n"
"    u32 sign = (bits >> 31) & 1u;\n"
"    int  e   = (int)((bits >> 23) & 0xFFu) - 127 + 7;\n"
"    u32 mant = (bits >> 20) & 0x7u;\n"
"    if (e >= 15) { e = 15; mant = 6u; } /* clamp to max finite */\n"
"    if (e <= 0)  return (u8)(sign << 7);\n"
"    return (u8)((sign << 7) | ((u32)(e & 0xF) << 3) | (mant & 0x7u));\n"
"}\n"
"\n"
"#define WMMA_FP8(A,B,C) C = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(A, B, C)\n"
"\n"
"#ifndef HD\n"
"#define HD 64\n"
"#endif\n"
"#define K_NB (HD / 16)   /* number of 16-wide K blocks for the QK / O paths */\n"
"#define BQ   64\n"
"#define BKV  16\n"
"\n"
"extern \"C\" {\n"
"\n"
"/* FP8 flash-attention.\n"
" *  Q     : [n_tok, n_heads, head_dim] FP8 e4m3 (already scaled).\n"
" *  K_t   : [n_heads, n_tok, head_dim] FP8 e4m3 pre-packed per head.\n"
" *  V_t   : [n_heads, n_tok, head_dim] FP8 e4m3 pre-packed per head.\n"
" *  out   : [n_tok, n_heads, head_dim] FP32.\n"
" *\n"
" * Layout: BQ=64 queries × BKV=16 keys per inner WMMA tile. 4 waves per WG,\n"
" * each wave owns 16 query rows. 128 threads cooperatively load one BKV tile\n"
" * of K and V. */\n"
"__global__ __launch_bounds__(128, 1)\n"
"void flash_attn_fp8_4w(float *out, const u8 *Q,\n"
"                       const u8 *K_t, const u8 *V_t,\n"
"                       int n_tok, int n_heads, float scale) {\n"
"    int h    = blockIdx.x;\n"
"    int qb   = blockIdx.y;\n"
"    int q0   = qb * BQ;\n"
"    int tid  = threadIdx.x;\n"
"    int wid  = tid >> 5;     /* 0..3 */\n"
"    int lid  = tid & 31;\n"
"    int half = lid >> 4;     /* 0,1 */\n"
"    int idx  = lid & 15;\n"
"    int dim  = n_heads * HD;\n"
"\n"
"    /* Shared memory: K-tile (16 rows × HD), V-tile (HD × 16). 1 byte/elt. */\n"
"    __shared__ u8 smK[16 * HD];     /* row-major: smK[kv*HD + d] */\n"
"    __shared__ u8 smV[HD * 16];     /* col-major: smV[d*16 + kv]  (transpose) */\n"
"    __shared__ u8 smP[4 * 16 * 16]; /* 4 waves × 16x16 P tile */\n"
"    u8 *smP_w = smP + wid * 16 * 16;\n"
"\n"
"    int q_row = q0 + wid * 16 + idx;\n"
"\n"
"    /* Load 8 fp8 of Q per K-block (head_dim sliced into K_NB blocks of 16).\n"
"     * Each lane carries 8 fp8 = u8x8 = u32x2 for one 16x16 fragment. */\n"
"    u32x2 q_reg[K_NB];\n"
"    {\n"
"        u8x8 tmp;\n"
"        for (int kb = 0; kb < K_NB; kb++) {\n"
"            for (int i = 0; i < 8; i++) {\n"
"                int d = kb * 16 + half * 8 + i;\n"
"                u8 qv = 0;\n"
"                if (q_row < n_tok && d < HD)\n"
"                    qv = Q[(size_t)q_row * dim + h * HD + d];\n"
"                tmp[i] = qv;\n"
"            }\n"
"            q_reg[kb] = pack_u8x8(tmp);\n"
"        }\n"
"    }\n"
"\n"
"    float8 O_acc[K_NB];\n"
"    for (int kb = 0; kb < K_NB; kb++)\n"
"        for (int i = 0; i < 8; i++) O_acc[kb][i] = 0.0f;\n"
"    float m_i[8], l_i[8];\n"
"    for (int i = 0; i < 8; i++) { m_i[i] = -1e30f; l_i[i] = 0.0f; }\n"
"\n"
"    int n_kv_tiles = (n_tok + BKV - 1) / BKV;\n"
"\n"
"    /* 128 threads cooperatively load 16 rows × HD bytes per tile.\n"
"     *   ld_row = tid >> 3 (0..15), ld_dchunk = tid & 7 (covers HD/8 bytes). */\n"
"    int ld_row = tid >> 3;\n"
"    int ld_d0  = (tid & 7) * (HD / 8);\n"
"\n"
"    for (int t = 0; t < n_kv_tiles; t++) {\n"
"        /* Load this tile's K and V into LDS. */\n"
"        int kv = t * BKV + ld_row;\n"
"        size_t kv_base = ((size_t)h * n_tok + kv) * HD;\n"
"        bool kv_ok = (kv < n_tok);\n"
"        for (int j = 0; j < HD / 8; j++) {\n"
"            int d = ld_d0 + j;\n"
"            u8 kv_K = 0, kv_V = 0;\n"
"            if (kv_ok && d < HD) {\n"
"                kv_K = K_t[kv_base + d];\n"
"                kv_V = V_t[kv_base + d];\n"
"            }\n"
"            smK[ld_row * HD + d] = kv_K;\n"
"            smV[d * 16 + ld_row] = kv_V;\n"
"        }\n"
"        __syncthreads();\n"
"\n"
"        /* QK^T: float8 score per lane. The 16x16 output of one WMMA tiles\n"
"         * over rows owned by this wave (wid * 16 .. wid * 16 + 15) and the\n"
"         * 16 keys in this BKV tile. */\n"
"        int kv0 = t * BKV;\n"
"        float8 score = {0,0,0,0,0,0,0,0};\n"
"        for (int kb = 0; kb < K_NB; kb++) {\n"
"            u8x8 b_K;\n"
"            for (int i = 0; i < 8; i++) {\n"
"                int dpos = kb * 16 + half * 8 + i;\n"
"                /* Each lane (idx 0..15) reads one key row of HD bytes. */\n"
"                b_K[i] = smK[idx * HD + dpos];\n"
"            }\n"
"            u32x2 b_K_p = pack_u8x8(b_K);\n"
"            WMMA_FP8(q_reg[kb], b_K_p, score);\n"
"        }\n"
"        bool col_valid = (kv0 + idx) < n_tok;\n"
"        for (int i = 0; i < 8; i++) score[i] = col_valid ? score[i] * scale : -1e30f;\n"
"\n"
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
"        for (int kb = 0; kb < K_NB; kb++)\n"
"            for (int i = 0; i < 8; i++) O_acc[kb][i] *= alpha[i];\n"
"\n"
"        /* Quantize P (softmax row in [0,1]) to FP8 and stage in LDS to swap\n"
"         * the WMMA operand layout (need P arranged so that lane idx provides\n"
"         * one column-of-16 K rows for the second WMMA). */\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int m_row = half * 8 + i;\n"
"            int n_col = idx;\n"
"            smP_w[m_row * 16 + n_col] = f32_to_fp8_e4m3(score[i]);\n"
"        }\n"
"        u8x8 ap_b;\n"
"        for (int i = 0; i < 8; i++) ap_b[i] = smP_w[idx * 16 + half * 8 + i];\n"
"        u32x2 ap = pack_u8x8(ap_b);\n"
"\n"
"        /* P × V: each kb is one HD/16 block of d. */\n"
"        for (int kb = 0; kb < K_NB; kb++) {\n"
"            u8x8 bv;\n"
"            for (int i = 0; i < 8; i++) {\n"
"                int d_col = kb * 16 + idx;\n"
"                int kv_k  = half * 8 + i;\n"
"                bv[i] = smV[d_col * 16 + kv_k];\n"
"            }\n"
"            u32x2 bv_p = pack_u8x8(bv);\n"
"            WMMA_FP8(ap, bv_p, O_acc[kb]);\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"\n"
"    /* Epilogue: divide by l, write FP32 output. */\n"
"    int q_base = q0 + wid * 16;\n"
"    for (int kb = 0; kb < K_NB; kb++) {\n"
"        for (int i = 0; i < 8; i++) {\n"
"            int m_row = half * 8 + i;\n"
"            int d = kb * 16 + idx;\n"
"            int row = q_base + m_row;\n"
"            if (row < n_tok && d < HD) {\n"
"                float v = O_acc[kb][i] / (l_i[i] > 0.0f ? l_i[i] : 1.0f);\n"
"                out[(size_t)row * dim + h * HD + d] = v;\n"
"            }\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"} /* extern C */\n";

/* ------------------------------------------------------------------------ */
/* CPU FP32 reference: dequantize fp8 inputs, run standard attention.
 * Used only with --check on the warmup launch. Slow O(n_tok^2 * head_dim). */

static void fa_ref_fp32(float *out, const uint8_t *Q,
                        const uint8_t *K_t, const uint8_t *V_t,
                        int n_tok, int n_heads, int head_dim, float scale) {
    int dim = n_heads * head_dim;
    float *scores = (float *)malloc((size_t)n_tok * sizeof(float));
    for (int h = 0; h < n_heads; h++) {
        for (int q = 0; q < n_tok; q++) {
            float row_max = -1e30f;
            for (int kv = 0; kv < n_tok; kv++) {
                float s = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    float qv = fp8_e4m3_to_f32(Q[(size_t)q * dim + h * head_dim + d]);
                    float kv_v = fp8_e4m3_to_f32(K_t[((size_t)h * n_tok + kv) * head_dim + d]);
                    s += qv * kv_v;
                }
                s *= scale;
                scores[kv] = s;
                if (s > row_max) row_max = s;
            }
            float l = 0.0f;
            for (int kv = 0; kv < n_tok; kv++) {
                scores[kv] = expf(scores[kv] - row_max);
                l += scores[kv];
            }
            float inv_l = (l > 0.0f) ? 1.0f / l : 0.0f;
            for (int d = 0; d < head_dim; d++) {
                float acc = 0.0f;
                for (int kv = 0; kv < n_tok; kv++) {
                    float p = scores[kv] * inv_l;
                    /* Quantize P to fp8 to mirror kernel's online quantization. */
                    uint8_t pq = hip_f32_to_fp8_e4m3(p * l);  /* multiply by l: kernel divides at end */
                    /* Simpler: skip the round-trip, the kernel approximates P
                     * post-renormalization but pre-divide. The reference
                     * computes the ideal softmax. The maxd will reflect FP8
                     * P quantization error, which we accept. */
                    (void)pq;
                    float vv = fp8_e4m3_to_f32(V_t[((size_t)h * n_tok + kv) * head_dim + d]);
                    acc += p * vv;
                }
                out[(size_t)q * dim + h * head_dim + d] = acc;
            }
        }
    }
    free(scores);
}

/* ------------------------------------------------------------------------ */

static double timer_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

static double cosine_sim(const float *a, const float *b, size_t n) {
    double dot=0, na=0, nb=0;
    for (size_t i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        na  += (double)a[i] * (double)a[i];
        nb  += (double)b[i] * (double)b[i];
    }
    if (na == 0 || nb == 0) return 0.0;
    return dot / (sqrt(na) * sqrt(nb));
}

static float max_abs_diff(const float *a, const float *b, size_t n) {
    float mx = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

static void fill_fp8_random(uint8_t *dst, size_t n, float abs_max, unsigned *rng) {
    for (size_t i = 0; i < n; i++) {
        unsigned r = *rng = *rng * 1103515245u + 12345u;
        float u = ((int)(r & 0xFFFFFF) - 0x800000) / (float)0x800000;
        float v = u * abs_max;
        if (v >  448.0f) v =  448.0f;
        if (v < -448.0f) v = -448.0f;
        dst[i] = hip_f32_to_fp8_e4m3(v);
    }
}

/* ------------------------------------------------------------------------ */

static int run_shape(hipFunction_t fn, int n_tok, int n_heads, int head_dim,
                     int iters, int do_check, float abs_max) {
    if (n_tok % 64) {
        fprintf(stderr, "skip: n_tok must be %%64 (got %d)\n", n_tok);
        return 0;
    }
    int dim = n_heads * head_dim;
    size_t bytes_QKV = (size_t)n_tok * dim;
    size_t bytes_out = (size_t)n_tok * dim * sizeof(float);

    uint8_t *hQ  = (uint8_t *)malloc(bytes_QKV);
    uint8_t *hKt = (uint8_t *)malloc(bytes_QKV);
    uint8_t *hVt = (uint8_t *)malloc(bytes_QKV);
    unsigned rng = 0x1234abcd;
    fill_fp8_random(hQ,  bytes_QKV, abs_max, &rng);
    fill_fp8_random(hKt, bytes_QKV, abs_max, &rng);
    fill_fp8_random(hVt, bytes_QKV, abs_max, &rng);

    void *dQ  = hip_upload_raw(hQ,  bytes_QKV);
    void *dKt = hip_upload_raw(hKt, bytes_QKV);
    void *dVt = hip_upload_raw(hVt, bytes_QKV);
    void *dO = NULL; HIP_CHECK(hipMalloc(&dO, bytes_out));
    HIP_CHECK(hipMemset(dO, 0, bytes_out));

    float scale = 1.0f / sqrtf((float)head_dim);
    void *args[] = { &dO, &dQ, &dKt, &dVt, &n_tok, &n_heads, &scale };

    /* Grid: (n_heads, n_tok/64). 4 waves * 32 threads = 128 threads per WG. */
    dim3 block = {128, 1, 1};
    dim3 grid  = {(unsigned)n_heads, (unsigned)(n_tok / 64), 1};

    /* Warmup */
    HIP_CHECK(hipModuleLaunchKernel(fn, grid.x, grid.y, grid.z,
                                    block.x, block.y, block.z,
                                    0, NULL, args, NULL));
    HIP_CHECK(hipDeviceSynchronize());

    double cos = -2.0; float maxd = -1.0f;
    if (do_check) {
        float *hO   = (float *)malloc(bytes_out);
        float *hRef = (float *)malloc(bytes_out);
        HIP_CHECK(hipMemcpy(hO, dO, bytes_out, hipMemcpyDeviceToHost));
        fa_ref_fp32(hRef, hQ, hKt, hVt, n_tok, n_heads, head_dim, scale);
        cos  = cosine_sim(hO, hRef, (size_t)n_tok * dim);
        maxd = max_abs_diff(hO, hRef, (size_t)n_tok * dim);
        free(hO); free(hRef);
    }

    double t0 = timer_ms();
    for (int i = 0; i < iters; i++) {
        HIP_CHECK(hipModuleLaunchKernel(fn, grid.x, grid.y, grid.z,
                                        block.x, block.y, block.z,
                                        0, NULL, args, NULL));
    }
    HIP_CHECK(hipDeviceSynchronize());
    double t1 = timer_ms();
    double ms = (t1 - t0) / (double)iters;
    /* FLOPs: 2 * n_heads * n_tok * n_tok * head_dim (QK^T) +
     *        2 * n_heads * n_tok * n_tok * head_dim (PV) */
    double flops = 4.0 * (double)n_heads * (double)n_tok * (double)n_tok * (double)head_dim;
    double tflops = flops / (ms * 1.0e-3) * 1.0e-12;

    printf("  fp8 4w  n_tok=%5d heads=%2d hd=%3d  %7.4f ms  %6.1f TFLOP/s",
           n_tok, n_heads, head_dim, ms, tflops);
    if (do_check) printf("  cos=%.6f  maxd=%.4f", cos, maxd);
    printf("\n");

    free(hQ); free(hKt); free(hVt);
    hipFree(dQ); hipFree(dKt); hipFree(dVt); hipFree(dO);
    return 0;
}

/* ------------------------------------------------------------------------ */

static void usage(const char *prog) {
    fprintf(stderr,
        "usage: %s [--n-tok N] [--heads H] [--head-dim D] [--iters N]\n"
        "          [--check] [--abs-max V] [--verbose]\n"
        "  --n-tok    number of query tokens, must be %%64 (default 1024)\n"
        "  --heads    number of attention heads (default 16)\n"
        "  --head-dim head dim (must match HD compile macro; default 64)\n"
        "  --iters    timing iterations (default 100)\n"
        "  --check    cosine vs FP32 reference (slow)\n"
        "  --abs-max  clamp random fp32 inputs before fp8 quant (default 1.0)\n",
        prog);
}

int main(int argc, char **argv) {
    int n_tok = 1024;
    int n_heads = 16;
    int head_dim = 64;
    int iters = 100;
    int do_check = 0;
    float abs_max = 1.0f;
    int verbose = 1;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--n-tok") && i + 1 < argc)    n_tok = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--heads") && i + 1 < argc) n_heads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--head-dim") && i + 1 < argc) head_dim = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--check")) do_check = 1;
        else if (!strcmp(argv[i], "--abs-max") && i + 1 < argc) abs_max = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--verbose")) verbose = 2;
        else { usage(argv[0]); return 1; }
    }

    if (head_dim != 64) {
        fprintf(stderr, "this build is fixed at HD=64 (recompile the kernel string with -DHD=...)\n");
        return 1;
    }

    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) {
        fprintf(stderr, "failed to load HIP runtime\n");
        return 1;
    }
    int dev = 0;
    HIP_CHECK(hipSetDevice(dev));

    hipModule_t mod;
    if (hip_compile_kernels(&mod, dev, kernel_src, "fp8_fa", verbose, "fp8_fa") < 0) return 1;
    hipFunction_t fn;
    HIP_CHECK(hipModuleGetFunction(&fn, mod, "flash_attn_fp8_4w"));

    printf("rdna4/fp8 WMMA flash-attention bench (iters=%d, abs_max=%.2f%s)\n",
           iters, abs_max, do_check ? ", check=on" : "");
    run_shape(fn, n_tok, n_heads, head_dim, iters, do_check, abs_max);
    return 0;
}
