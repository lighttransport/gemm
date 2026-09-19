#define _GNU_SOURCE
#include "swfp4fp8.h"

#include <arm_sve.h>
#include <errno.h>
#include <math.h>
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#define PANEL_N 16

struct swfp4fp8_context {
    int nthreads;
    int *cpu_ids;
};

struct swfp4fp8_matrix {
    swfp4fp8_format format;
    size_t n;
    size_t n_pad;
    size_t k;
    size_t group_k;
    size_t groups;
    uint8_t *codes;
    uint8_t *row_codes;
    uint8_t *sdot_codes;
    uint8_t *scales8;
    float *scales32;
    size_t code_bytes;
    size_t row_code_bytes;
    size_t scale_bytes;
    float global_scale;
    int heap_backed;
};

extern void swfp4_mxfp4_sdot_panel16(const uint8_t *, const uint8_t *,
                                     const int8_t *, const float *, size_t,
                                     float *);

static uint32_t fp8_lut[256];
static uint32_t e8m0_lut[256];
static int lut_ready;
static const float fp4_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
   -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
};

static uint32_t f32_bits(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    return u;
}

float swfp4fp8_decode_e2m1(uint8_t code) {
    static const float values[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
       -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
    };
    return values[code & 15];
}

float swfp4fp8_decode_e4m3(uint8_t code) {
    int sign = code >> 7;
    int exp = (code >> 3) & 15;
    int mant = code & 7;
    float v;
    if (exp == 0) {
        v = ldexpf((float)mant, -9);
    } else if (exp == 15 && mant == 7) {
        return copysignf(NAN, sign ? -1.0f : 1.0f);
    } else {
        v = ldexpf(1.0f + (float)mant * 0.125f, exp - 7);
    }
    return sign ? -v : v;
}

float swfp4fp8_decode_e8m0(uint8_t code) {
    return code == 255 ? NAN : ldexpf(1.0f, (int)code - 127);
}

static void init_luts(void) {
    if (lut_ready) return;
    #pragma omp critical(swfp4fp8_lut_init)
    {
        if (!lut_ready) {
            for (int i = 0; i < 256; ++i) {
                fp8_lut[i] = f32_bits(swfp4fp8_decode_e4m3((uint8_t)i));
                e8m0_lut[i] = f32_bits(swfp4fp8_decode_e8m0((uint8_t)i));
            }
            lut_ready = 1;
        }
    }
}

static void pin_worker(const swfp4fp8_context *ctx) {
    int tid = omp_get_thread_num();
    if (!ctx || tid >= ctx->nthreads || ctx->cpu_ids[tid] < 0) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(ctx->cpu_ids[tid], &set);
    (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

int swfp4fp8_context_create(swfp4fp8_context **out, int nthreads,
                            const int *cpu_ids) {
    if (!out || nthreads < 1 || nthreads > 256 || (!cpu_ids && nthreads > 48))
        return EINVAL;
    swfp4fp8_context *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) return ENOMEM;
    ctx->cpu_ids = malloc((size_t)nthreads * sizeof(*ctx->cpu_ids));
    if (!ctx->cpu_ids) {
        free(ctx);
        return ENOMEM;
    }
    ctx->nthreads = nthreads;
    for (int i = 0; i < nthreads; ++i)
        ctx->cpu_ids[i] = cpu_ids ? cpu_ids[i] : 12 + i;
    init_luts();
    *out = ctx;
    return 0;
}

void swfp4fp8_context_destroy(swfp4fp8_context *ctx) {
    if (!ctx) return;
    free(ctx->cpu_ids);
    free(ctx);
}

static void *aligned_zero(size_t bytes) {
    if (!bytes) return NULL;
    void *p = NULL;
    if (posix_memalign(&p, 256, (bytes + 255) & ~(size_t)255)) return NULL;
    memset(p, 0, bytes);
    return p;
}

static void *mapped_zero(size_t bytes) {
    if (!bytes) return NULL;
    if (getenv("SWFP4FP8_XOS_ALLOC")) return aligned_zero(bytes);
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? NULL : p;
}

static int matrix_alloc(swfp4fp8_matrix **out, swfp4fp8_format format,
                        size_t n, size_t k, size_t group_k) {
    swfp4fp8_matrix *w = calloc(1, sizeof(*w));
    if (!w) return ENOMEM;
    w->format = format;
    w->n = n;
    w->n_pad = (n + PANEL_N - 1) & ~(size_t)(PANEL_N - 1);
    w->k = k;
    w->group_k = group_k;
    w->groups = k / group_k;
    w->code_bytes = w->n_pad * k / (format == SWFP4FP8_NVFP4_G16 ||
                                    format == SWFP4FP8_MXFP4_G32 ? 2 : 1);
    w->row_code_bytes = n * k / (format == SWFP4FP8_NVFP4_G16 ||
                                 format == SWFP4FP8_MXFP4_G32 ? 2 : 1);
    w->codes = mapped_zero(w->code_bytes);
    w->row_codes = mapped_zero(w->row_code_bytes);
    w->heap_backed = getenv("SWFP4FP8_XOS_ALLOC") != NULL;
    if (!w->codes || !w->row_codes) {
        if (w->heap_backed) { free(w->codes); free(w->row_codes); }
        else {
            if (w->codes) munmap(w->codes, w->code_bytes);
            if (w->row_codes) munmap(w->row_codes, w->row_code_bytes);
        }
        free(w);
        return ENOMEM;
    }
    *out = w;
    return 0;
}

static const uint8_t korder[16] = {
    0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15
};
static const uint8_t lane_to_col[32] = {
     0, 1, 2, 3,  8, 9,10,11, 16,17,18,19, 24,25,26,27,
     4, 5, 6, 7, 12,13,14,15, 20,21,22,23, 28,29,30,31
};

static int qpn_lane_for_col(int col) {
    for (int lane = 0; lane < 32; ++lane)
        if (lane_to_col[lane] == col) return lane;
    return -1;
}

static int qpn_pos_for_k(int k) {
    for (int pos = 0; pos < 16; ++pos)
        if (korder[pos] == k) return pos;
    return -1;
}

static uint8_t source_nvfp4_code(const uint8_t *src, size_t n, size_t k,
                                 size_t row, size_t col,
                                 swfp4fp8_source_layout layout) {
    (void)n;
    if (layout == SWFP4FP8_LAYOUT_CANONICAL) {
        uint8_t b = src[row * (k / 2) + col / 2];
        return (b >> (4 * (col & 1))) & 15;
    }
    size_t groups = k / 16;
    size_t tile = row / 32;
    int lane = qpn_lane_for_col((int)(row & 31));
    size_t g = col / 16;
    int pos = qpn_pos_for_k((int)(col & 15));
    size_t off = (((tile * groups + g) * 32 + (size_t)lane) * 8 +
                  (size_t)pos / 2);
    uint8_t b = src[off];
    return (b >> (4 * (pos & 1))) & 15;
}

static uint8_t source_qpn8_code(const uint8_t *src, size_t n, size_t k,
                                size_t row, size_t col,
                                swfp4fp8_source_layout layout) {
    (void)n;
    if (layout == SWFP4FP8_LAYOUT_CANONICAL) return src[row * k + col];
    size_t groups = k / 16;
    size_t tile = row / 32;
    int lane = qpn_lane_for_col((int)(row & 31));
    size_t g = col / 16;
    int pos = qpn_pos_for_k((int)(col & 15));
    return src[(((tile * groups + g) * 32 + (size_t)lane) * 16 +
                (size_t)pos)];
}

int swfp4fp8_pack_nvfp4(swfp4fp8_context *ctx, swfp4fp8_matrix **out,
                        size_t n, size_t k, const uint8_t *codes,
                        const uint8_t *scales, float global_scale,
                        swfp4fp8_source_layout layout) {
    if (!ctx || !out || !codes || !scales || !n || !k || k % 16 ||
        (layout != SWFP4FP8_LAYOUT_CANONICAL && layout != SWFP4FP8_LAYOUT_V100_QPN) ||
        (layout == SWFP4FP8_LAYOUT_V100_QPN && (n % 32 || k % 64)))
        return EINVAL;
    swfp4fp8_matrix *w;
    int rc = matrix_alloc(&w, SWFP4FP8_NVFP4_G16, n, k, 16);
    if (rc) return rc;
    w->global_scale = global_scale;
    w->scale_bytes = (w->n_pad / 16) * w->groups * 16;
    w->scales8 = mapped_zero(w->scale_bytes);
    if (!w->scales8) { swfp4fp8_matrix_destroy(w); return ENOMEM; }
    #pragma omp parallel for num_threads(ctx->nthreads) schedule(static)
    for (size_t p = 0; p < w->n_pad / 16; ++p) {
        pin_worker(ctx);
        for (size_t g = 0; g < w->groups; ++g) {
            for (size_t lane = 0; lane < 16; ++lane) {
                size_t row = p * 16 + lane;
                if (row >= n) continue;
                size_t si;
                if (layout == SWFP4FP8_LAYOUT_CANONICAL) {
                    si = row * w->groups + g;
                } else {
                    int qlane = qpn_lane_for_col((int)(row & 31));
                    si = (((row / 32) * w->groups + g) * 32 + (size_t)qlane);
                }
                w->scales8[(p * w->groups + g) * 16 + lane] = scales[si];
            }
            for (size_t pair = 0; pair < 8; ++pair) {
                for (size_t lane = 0; lane < 16; ++lane) {
                    size_t row = p * 16 + lane;
                    if (row >= n) continue;
                    size_t k0 = g * 16 + pair * 2;
                    uint8_t lo = source_nvfp4_code(codes, n, k, row, k0, layout);
                    uint8_t hi = source_nvfp4_code(codes, n, k, row, k0 + 1, layout);
                    w->codes[((p * w->groups + g) * 8 + pair) * 16 + lane] =
                        lo | (uint8_t)(hi << 4);
                    w->row_codes[row * (k / 2) + g * 8 + pair] =
                        lo | (uint8_t)(hi << 4);
                }
            }
        }
    }
    *out = w;
    return 0;
}

int swfp4fp8_pack_qpn8(swfp4fp8_context *ctx, swfp4fp8_matrix **out,
                       size_t n, size_t k, const uint8_t *codes,
                       const float *tile_scales,
                       swfp4fp8_source_layout layout) {
    if (!ctx || !out || !codes || !tile_scales || !n || !k || k % 16 ||
        (layout != SWFP4FP8_LAYOUT_CANONICAL && layout != SWFP4FP8_LAYOUT_V100_QPN) ||
        n % 32) return EINVAL;
    swfp4fp8_matrix *w;
    int rc = matrix_alloc(&w, SWFP4FP8_QPN8_TILE32, n, k, 16);
    if (rc) return rc;
    w->scale_bytes = (w->n_pad / 16) * sizeof(float);
    w->scales32 = mapped_zero(w->scale_bytes);
    if (!w->scales32) { swfp4fp8_matrix_destroy(w); return ENOMEM; }
    #pragma omp parallel for num_threads(ctx->nthreads) schedule(static)
    for (size_t p = 0; p < w->n_pad / 16; ++p) {
        pin_worker(ctx);
        w->scales32[p] = tile_scales[p / 2] * (1.0f / 256.0f);
        for (size_t kk = 0; kk < k; ++kk)
            for (size_t lane = 0; lane < 16; ++lane) {
                size_t row = p * 16 + lane;
                if (row < n)
                    w->codes[(p * k + kk) * 16 + lane] =
                        w->row_codes[row * k + kk] =
                            source_qpn8_code(codes, n, k, row, kk, layout);
            }
    }
    *out = w;
    return 0;
}

int swfp4fp8_pack_mxfp4(swfp4fp8_context *ctx, swfp4fp8_matrix **out,
                        size_t n, size_t k, const uint8_t *codes,
                        const uint8_t *scales) {
    if (!ctx || !out || !codes || !scales || !n || !k || k % 32) return EINVAL;
    swfp4fp8_matrix *w;
    int rc = matrix_alloc(&w, SWFP4FP8_MXFP4_G32, n, k, 32);
    if (rc) return rc;
    w->scale_bytes = (w->n_pad / 16) * w->groups * 16;
    w->scales8 = mapped_zero(w->scale_bytes);
    w->sdot_codes = mapped_zero(w->code_bytes);
    if (!w->scales8 || !w->sdot_codes) { swfp4fp8_matrix_destroy(w); return ENOMEM; }
    #pragma omp parallel for num_threads(ctx->nthreads) schedule(static)
    for (size_t p = 0; p < w->n_pad / 16; ++p) {
        pin_worker(ctx);
        for (size_t g = 0; g < w->groups; ++g) {
            for (size_t lane = 0; lane < 16; ++lane) {
                size_t row = p * 16 + lane;
                if (row >= n) continue;
                w->scales8[(p * w->groups + g) * 16 + lane] =
                    scales[row * w->groups + g];
                const uint8_t *sb = codes + row * (k / 2) + g * 16;
                memcpy(w->row_codes + row * (k / 2) + g * 16, sb, 16);
                for (size_t pair = 0; pair < 16; ++pair) {
                    size_t k0 = pair * 2, k1 = k0 + 1;
                    uint8_t c0 = k0 < 16 ? sb[k0] & 15 : sb[k0 - 16] >> 4;
                    uint8_t c1 = k1 < 16 ? sb[k1] & 15 : sb[k1 - 16] >> 4;
                    w->codes[((p * w->groups + g) * 16 + pair) * 16 + lane] =
                        c0 | (uint8_t)(c1 << 4);
                }
                for (size_t q = 0; q < 8; ++q) {
                    size_t k0 = q * 4;
                    uint8_t c0 = k0 < 16 ? sb[k0] & 15 : sb[k0 - 16] >> 4;
                    uint8_t c1 = k0 + 1 < 16 ? sb[k0 + 1] & 15 : sb[k0 - 15] >> 4;
                    uint8_t c2 = k0 + 2 < 16 ? sb[k0 + 2] & 15 : sb[k0 - 14] >> 4;
                    uint8_t c3 = k0 + 3 < 16 ? sb[k0 + 3] & 15 : sb[k0 - 13] >> 4;
                    uint8_t *dp = w->sdot_codes +
                        ((p * w->groups + g) * 8 + q) * 32 + lane * 2;
                    dp[0] = c0 | (uint8_t)(c1 << 4);
                    dp[1] = c2 | (uint8_t)(c3 << 4);
                }
            }
        }
    }
    *out = w;
    return 0;
}

int swfp4fp8_pack_fp8_block128(swfp4fp8_context *ctx,
                               swfp4fp8_matrix **out, size_t n, size_t k,
                               const uint8_t *codes,
                               const uint8_t *block_scales) {
    if (!ctx || !out || !codes || !block_scales || !n || !k || k % 128)
        return EINVAL;
    swfp4fp8_matrix *w;
    int rc = matrix_alloc(&w, SWFP4FP8_FP8_BLOCK128, n, k, 128);
    if (rc) return rc;
    w->scale_bytes = (w->n_pad / 16) * w->groups;
    w->scales8 = mapped_zero(w->scale_bytes);
    if (!w->scales8) { swfp4fp8_matrix_destroy(w); return ENOMEM; }
    #pragma omp parallel for num_threads(ctx->nthreads) schedule(static)
    for (size_t p = 0; p < w->n_pad / 16; ++p) {
        pin_worker(ctx);
        for (size_t g = 0; g < w->groups; ++g)
            w->scales8[p * w->groups + g] =
                block_scales[(p / 8) * w->groups + g];
        for (size_t kk = 0; kk < k; ++kk)
            for (size_t lane = 0; lane < 16; ++lane) {
                size_t row = p * 16 + lane;
                if (row < n)
                    w->codes[(p * k + kk) * 16 + lane] =
                        w->row_codes[row * k + kk] = codes[row * k + kk];
            }
    }
    *out = w;
    return 0;
}

void swfp4fp8_matrix_destroy(swfp4fp8_matrix *w) {
    if (!w) return;
    if (w->heap_backed) {
        free(w->codes); free(w->row_codes); free(w->sdot_codes);
        free(w->scales8); free(w->scales32);
    } else {
        if (w->codes) munmap(w->codes, w->code_bytes);
        if (w->row_codes) munmap(w->row_codes, w->row_code_bytes);
        if (w->sdot_codes) munmap(w->sdot_codes, w->code_bytes);
        if (w->scales8) munmap(w->scales8, w->scale_bytes);
        if (w->scales32) munmap(w->scales32, w->scale_bytes);
    }
    free(w);
}

size_t swfp4fp8_matrix_n(const swfp4fp8_matrix *w) { return w ? w->n : 0; }
size_t swfp4fp8_matrix_k(const swfp4fp8_matrix *w) { return w ? w->k : 0; }
size_t swfp4fp8_matrix_bytes(const swfp4fp8_matrix *w) {
    return w ? w->code_bytes + w->scale_bytes : 0;
}
swfp4fp8_format swfp4fp8_matrix_format(const swfp4fp8_matrix *w) {
    return w ? w->format : SWFP4FP8_NVFP4_G16;
}

static inline void set_ftz(void) {
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    fpcr |= 1ull << 24;
    __asm__ volatile("msr fpcr, %0" :: "r"(fpcr));
}

static inline svfloat32_t fp8_magic(svbool_t pg, svuint32_t b) {
    svuint32_t sign = svlsl_n_u32_x(pg, svand_n_u32_x(pg, b, 0x80u), 24);
    svuint32_t mag = svlsl_n_u32_x(pg, svand_n_u32_x(pg, b, 0x7fu), 20);
    svfloat32_t f = svmul_n_f32_x(pg, svreinterpret_f32_u32(mag), 0x1.0p+120f);
    return svreinterpret_f32_u32(svorr_u32_x(pg, sign,
                                             svreinterpret_u32_f32(f)));
}

#define DECL_1 svfloat32_t z0 = svdup_f32(0.0f)
#define DECL_2 DECL_1; svfloat32_t z1 = svdup_f32(0.0f)
#define DECL_4 DECL_2; svfloat32_t z2 = svdup_f32(0.0f); svfloat32_t z3 = svdup_f32(0.0f)
#define DECL_8 DECL_4; svfloat32_t z4 = svdup_f32(0.0f); svfloat32_t z5 = svdup_f32(0.0f); svfloat32_t z6 = svdup_f32(0.0f); svfloat32_t z7 = svdup_f32(0.0f)
#define DECL_16 DECL_8; svfloat32_t z8 = svdup_f32(0.0f); svfloat32_t z9 = svdup_f32(0.0f); svfloat32_t z10 = svdup_f32(0.0f); svfloat32_t z11 = svdup_f32(0.0f); svfloat32_t z12 = svdup_f32(0.0f); svfloat32_t z13 = svdup_f32(0.0f); svfloat32_t z14 = svdup_f32(0.0f); svfloat32_t z15 = svdup_f32(0.0f)

#define FMA1(Z,I,W,KK) Z = svmla_n_f32_x(pg, Z, W, a[(I) * lda + (KK)])
#define FMA_1(W,KK) FMA1(z0,0,W,KK)
#define FMA_2(W,KK) FMA_1(W,KK); FMA1(z1,1,W,KK)
#define FMA_4(W,KK) FMA_2(W,KK); FMA1(z2,2,W,KK); FMA1(z3,3,W,KK)
#define FMA_8(W,KK) FMA_4(W,KK); FMA1(z4,4,W,KK); FMA1(z5,5,W,KK); FMA1(z6,6,W,KK); FMA1(z7,7,W,KK)
#define FMA_16(W,KK) FMA_8(W,KK); FMA1(z8,8,W,KK); FMA1(z9,9,W,KK); FMA1(z10,10,W,KK); FMA1(z11,11,W,KK); FMA1(z12,12,W,KK); FMA1(z13,13,W,KK); FMA1(z14,14,W,KK); FMA1(z15,15,W,KK)

#define ST1(Z,I) svst1(pg, c + (I) * ldc + p * 16, Z)
#define STORE_1 ST1(z0,0)
#define STORE_2 STORE_1; ST1(z1,1)
#define STORE_4 STORE_2; ST1(z2,2); ST1(z3,3)
#define STORE_8 STORE_4; ST1(z4,4); ST1(z5,5); ST1(z6,6); ST1(z7,7)
#define STORE_16 STORE_8; ST1(z8,8); ST1(z9,9); ST1(z10,10); ST1(z11,11); ST1(z12,12); ST1(z13,13); ST1(z14,14); ST1(z15,15)

#define DEFINE_PANEL(MB) \
static void panel_##MB(const swfp4fp8_matrix *w, const float *a, size_t lda, \
                       float *c, size_t ldc, size_t p, int lossy_ftz) { \
    svbool_t pg = svwhilelt_b32((uint64_t)0, (uint64_t)(w->n - p * 16 < 16 ? w->n - p * 16 : 16)); \
    svbool_t all = svptrue_b32(); \
    svfloat32_t fp4tab = svld1(all, fp4_lut); \
    DECL_##MB; \
    if (w->format == SWFP4FP8_NVFP4_G16 || w->format == SWFP4FP8_MXFP4_G32) { \
        size_t pairs = w->group_k / 2; \
        for (size_t g = 0; g < w->groups; ++g) { \
            const uint8_t *sp = w->scales8 + (p * w->groups + g) * 16; \
            svuint32_t si = svld1ub_u32(all, sp); \
            svuint32_t sbits = svld1_gather_u32index_u32(all, \
                w->format == SWFP4FP8_NVFP4_G16 ? fp8_lut : e8m0_lut, si); \
            svfloat32_t scale = svreinterpret_f32_u32(sbits); \
            if (w->format == SWFP4FP8_NVFP4_G16) \
                scale = svmul_n_f32_x(all, scale, w->global_scale); \
            for (size_t q = 0; q < pairs; ++q) { \
                const uint8_t *cp = w->codes + ((p * w->groups + g) * pairs + q) * 16; \
                svuint32_t raw = svld1ub_u32(all, cp); \
                svfloat32_t w0 = svmul_x(all, svtbl_f32(fp4tab, svand_n_u32_x(all, raw, 15)), scale); \
                svfloat32_t w1 = svmul_x(all, svtbl_f32(fp4tab, svlsr_n_u32_x(all, raw, 4)), scale); \
                size_t kk = g * w->group_k + q * 2; \
                FMA_##MB(w0,kk); FMA_##MB(w1,kk+1); \
            } \
        } \
    } else { \
        for (size_t kk = 0; kk < w->k; ++kk) { \
            svuint32_t raw = svld1ub_u32(all, w->codes + (p * w->k + kk) * 16); \
            svfloat32_t vw = lossy_ftz ? fp8_magic(all, raw) : \
                svreinterpret_f32_u32(svld1_gather_u32index_u32(all, fp8_lut, raw)); \
            float scale = w->format == SWFP4FP8_QPN8_TILE32 ? w->scales32[p] : \
                swfp4fp8_decode_e8m0(w->scales8[p * w->groups + kk / 128]); \
            vw = svmul_n_f32_x(all, vw, scale); \
            FMA_##MB(vw,kk); \
        } \
    } \
    STORE_##MB; \
}

DEFINE_PANEL(1)
DEFINE_PANEL(2)
DEFINE_PANEL(4)
DEFINE_PANEL(8)
DEFINE_PANEL(16)

#undef DEFINE_PANEL

static void run_panel_block(const swfp4fp8_matrix *w, const float *a,
                            size_t lda, float *c, size_t ldc, size_t m,
                            size_t p, int ftz) {
    switch (m) {
    case 16: panel_16(w,a,lda,c,ldc,p,ftz); break;
    case 8:  panel_8(w,a,lda,c,ldc,p,ftz); break;
    case 4:  panel_4(w,a,lda,c,ldc,p,ftz); break;
    case 2:  panel_2(w,a,lda,c,ldc,p,ftz); break;
    default: panel_1(w,a,lda,c,ldc,p,ftz); break;
    }
}

static void row_fp8_8(const swfp4fp8_matrix *w, const float *x, float *y,
                      size_t n0, size_t nr) {
    svbool_t pg=svptrue_b32();
    DECL_8;
    for(size_t kk=0;kk<w->k;kk+=16){
        float sc=w->format==SWFP4FP8_QPN8_TILE32?w->scales32[n0/16]:
            swfp4fp8_decode_e8m0(w->scales8[(n0/16)*w->groups+kk/128]);
        svfloat32_t vx=svmul_n_f32_x(pg,svld1(pg,x+kk),sc);
        #define FP8R(R,Z) do{if(nr>(R)){ \
            svuint32_t q=svld1ub_u32(pg,w->row_codes+(n0+(R))*w->k+kk); \
            svfloat32_t vw=svreinterpret_f32_u32(svld1_gather_u32index_u32(pg,fp8_lut,q)); \
            Z=svmla_x(pg,Z,vw,vx);}}while(0)
        FP8R(0,z0);FP8R(1,z1);FP8R(2,z2);FP8R(3,z3);
        FP8R(4,z4);FP8R(5,z5);FP8R(6,z6);FP8R(7,z7);
        #undef FP8R
    }
    #define FP8S(R,Z) do{if(nr>(R))y[n0+(R)]=svaddv(pg,Z);}while(0)
    FP8S(0,z0);FP8S(1,z1);FP8S(2,z2);FP8S(3,z3);
    FP8S(4,z4);FP8S(5,z5);FP8S(6,z6);FP8S(7,z7);
    #undef FP8S
}

static void row_fp4_8(const swfp4fp8_matrix *w, const float *x, float *y,
                      size_t n0, size_t nr) {
    svbool_t pg=svptrue_b32(),pg8=svwhilelt_b32((uint64_t)0,(uint64_t)8);
    svfloat32_t tab=svld1(pg,fp4_lut);
    svuint32_t even=svindex_u32(0,2),odd=svindex_u32(1,2);
    DECL_8;
    for(size_t g=0;g<w->groups;++g){
        svfloat32_t x0,x1;
        size_t bytes;
        svbool_t active;
        if(w->format==SWFP4FP8_MXFP4_G32){
            x0=svld1(pg,x+g*32);x1=svld1(pg,x+g*32+16);bytes=16;active=pg;
        }else{
            x0=svld1_gather_u32index_f32(pg8,x+g*16,even);
            x1=svld1_gather_u32index_f32(pg8,x+g*16,odd);
            bytes=8;active=pg8;
        }
        #define FP4R(R,Z) do{if(nr>(R)){ \
            size_t rr=n0+(R);const uint8_t*qp=w->row_codes+rr*(w->k/2)+g*bytes; \
            svuint32_t q=svld1ub_u32(active,qp); \
            svfloat32_t lo=svtbl_f32(tab,svand_n_u32_x(active,q,15)); \
            svfloat32_t hi=svtbl_f32(tab,svlsr_n_u32_x(active,q,4)); \
            uint8_t sb=w->scales8[((rr/16)*w->groups+g)*16+(rr&15)]; \
            float sc=w->format==SWFP4FP8_NVFP4_G16?swfp4fp8_decode_e4m3(sb)*w->global_scale:swfp4fp8_decode_e8m0(sb); \
            svfloat32_t vx0=svmul_n_f32_x(active,x0,sc),vx1=svmul_n_f32_x(active,x1,sc); \
            Z=svmla_x(active,Z,lo,vx0);Z=svmla_x(active,Z,hi,vx1);}}while(0)
        FP4R(0,z0);FP4R(1,z1);FP4R(2,z2);FP4R(3,z3);
        FP4R(4,z4);FP4R(5,z5);FP4R(6,z6);FP4R(7,z7);
        #undef FP4R
    }
    svbool_t sum_pg=w->format==SWFP4FP8_MXFP4_G32?pg:pg8;
    #define FP4S(R,Z) do{if(nr>(R))y[n0+(R)]=svaddv(sum_pg,Z);}while(0)
    FP4S(0,z0);FP4S(1,z1);FP4S(2,z2);FP4S(3,z3);
    FP4S(4,z4);FP4S(5,z5);FP4S(6,z6);FP4S(7,z7);
    #undef FP4S
}

static void run_row_gemm(swfp4fp8_context *ctx,const swfp4fp8_matrix *w,
                         const float *a,size_t lda,float *c,size_t ldc,size_t m){
    size_t ng=(w->n+7)/8,jobs=m*ng;
    #pragma omp parallel for num_threads(ctx->nthreads) schedule(static)
    for(size_t job=0;job<jobs;++job){
        pin_worker(ctx);size_t im=job/ng,n0=(job%ng)*8,nr=w->n-n0<8?w->n-n0:8;
        if(w->format==SWFP4FP8_NVFP4_G16||w->format==SWFP4FP8_MXFP4_G32)
            row_fp4_8(w,a+im*lda,c+im*ldc,n0,nr);
        else row_fp8_8(w,a+im*lda,c+im*ldc,n0,nr);
    }
}

static int run_fp4_sdot(swfp4fp8_context *ctx,const swfp4fp8_matrix *w,
                        const float *a,size_t lda,float *c,size_t ldc,size_t m){
    size_t ng=w->groups,gk=w->group_k;
    int8_t *xq=aligned_zero(m*ng*64);
    float *xs=aligned_zero(m*ng*sizeof(*xs));
    if(!xq||!xs){free(xq);free(xs);return ENOMEM;}
    #pragma omp parallel for num_threads(ctx->nthreads) schedule(static)
    for(size_t job=0;job<m*ng;++job){
        size_t im=job/ng,g=job%ng;float ma=0;
        for(size_t j=0;j<gk;++j)ma=fmaxf(ma,fabsf(a[im*lda+g*gk+j]));
        float sc=ma>0?ma/127.f:1.f;xs[job]=sc;
        for(size_t j=0;j<gk;++j)xq[job*64+j]=(int8_t)lrintf(a[im*lda+g*gk+j]/sc);
    }
    size_t jobs=m*w->n;
    #pragma omp parallel for num_threads(ctx->nthreads) schedule(static)
    for(size_t job=0;job<jobs;++job){
        pin_worker(ctx);size_t im=job/w->n,row=job%w->n;float sum=0;int8_t wb[64]={0};
        for(size_t g=0;g<ng;++g){
            const uint8_t *q=w->row_codes+row*(w->k/2)+g*(gk/2);
            if(w->format==SWFP4FP8_NVFP4_G16){
                for(size_t j=0;j<8;++j){wb[2*j]=(int8_t)(2*swfp4fp8_decode_e2m1(q[j]&15));wb[2*j+1]=(int8_t)(2*swfp4fp8_decode_e2m1(q[j]>>4));}
            }else{
                for(size_t j=0;j<16;++j){wb[j]=(int8_t)(2*swfp4fp8_decode_e2m1(q[j]&15));wb[j+16]=(int8_t)(2*swfp4fp8_decode_e2m1(q[j]>>4));}
            }
            svint32_t d=svdot_s32(svdup_s32(0),svld1_s8(svptrue_b8(),wb),
                                  svld1_s8(svptrue_b8(),xq+(im*ng+g)*64));
            int dot=svaddv(svptrue_b32(),d);
            uint8_t sb=w->scales8[((row/16)*ng+g)*16+(row&15)];
            float ws=w->format==SWFP4FP8_NVFP4_G16?swfp4fp8_decode_e4m3(sb)*w->global_scale:swfp4fp8_decode_e8m0(sb);
            sum+=(float)dot*(0.5f*ws*xs[im*ng+g]);
        }
        c[im*ldc+row]=sum;
    }
    free(xq);free(xs);return 0;
}

static int run_mxfp4_fused_sdot(swfp4fp8_context *ctx,
                                const swfp4fp8_matrix *w,
                                const float *x, float *y) {
    if (w->format != SWFP4FP8_MXFP4_G32 || !w->sdot_codes || w->n % 16)
        return EINVAL;
    int8_t *xq = aligned_zero(w->k);
    float *xs = aligned_zero(w->groups * sizeof(*xs));
    if (!xq || !xs) { free(xq); free(xs); return ENOMEM; }
    for (size_t g = 0; g < w->groups; ++g) {
        float ma = 0.0f;
        for (size_t j = 0; j < 32; ++j) ma = fmaxf(ma, fabsf(x[g * 32 + j]));
        float scale = ma > 0.0f ? ma / 127.0f : 1.0f;
        xs[g] = 0.5f * scale;
        for (size_t j = 0; j < 32; ++j)
            xq[g * 32 + j] = (int8_t)lrintf(x[g * 32 + j] / scale);
    }
    size_t panels = w->n / 16;
    #pragma omp parallel for num_threads(ctx->nthreads) schedule(static)
    for (size_t p = 0; p < panels; ++p) {
        pin_worker(ctx);
        swfp4_mxfp4_sdot_panel16(w->sdot_codes + p * w->groups * 256,
                                 w->scales8 + p * w->groups * 16,
                                 xq, xs, w->groups, y + p * 16);
    }
    free(xq); free(xs);
    return 0;
}

int swfp4fp8_gemm_f32(swfp4fp8_context *ctx,
                      const swfp4fp8_matrix *w,
                      const float *a, size_t lda, float *c, size_t ldc,
                      size_t m, swfp4fp8_kernel kernel) {
    if (!ctx || !w || !a || !c || !m || lda < w->k || ldc < w->n)
        return EINVAL;
    if(kernel==SWFP4FP8_KERNEL_FP4_SDOT){
        if(w->format!=SWFP4FP8_NVFP4_G16&&w->format!=SWFP4FP8_MXFP4_G32)return EINVAL;
        return run_fp4_sdot(ctx,w,a,lda,c,ldc,m);
    }
    if (kernel == SWFP4FP8_KERNEL_MXFP4_FUSED_SDOT) {
        if (m != 1) return EINVAL;
        return run_mxfp4_fused_sdot(ctx, w, a, c);
    }
    int ftz = kernel == SWFP4FP8_KERNEL_FP8_FTZ;
    if (ftz && w->format != SWFP4FP8_QPN8_TILE32 &&
        w->format != SWFP4FP8_FP8_BLOCK128) return EINVAL;
    if (kernel == SWFP4FP8_KERNEL_ROW) {
        run_row_gemm(ctx,w,a,lda,c,ldc,m);
        return 0;
    }
    size_t np = w->n_pad / 16;
    #pragma omp parallel for num_threads(ctx->nthreads) schedule(static)
    for (size_t p = 0; p < np; ++p) {
        pin_worker(ctx);
        if (ftz) set_ftz();
        size_t mb = 0;
        while (mb < m) {
            size_t rem = m - mb;
            /* The 16-row form exceeds FCC's spill-free register budget on
             * large K.  Two 8-row passes are decisively faster on A64FX. */
            size_t blk = rem >= 8 ? 8 : rem >= 4 ? 4 : rem >= 2 ? 2 : 1;
            run_panel_block(w, a + mb * lda, lda, c + mb * ldc, ldc, blk, p, ftz);
            mb += blk;
        }
    }
    return 0;
}

int swfp4fp8_ffn_mxfp4_sdot(swfp4fp8_context *ctx,
                            const swfp4fp8_matrix *gate,
                            const swfp4fp8_matrix *up,
                            const swfp4fp8_matrix *down,
                            const float *x, float *y, float *scratch) {
    if (!ctx || !gate || !up || !down || !x || !y || !scratch ||
        gate->format != SWFP4FP8_MXFP4_G32 ||
        up->format != SWFP4FP8_MXFP4_G32 ||
        down->format != SWFP4FP8_MXFP4_G32 || gate->k != up->k ||
        gate->n != up->n || down->k != gate->n)
        return EINVAL;
    float *gate_out = scratch;
    float *up_out = scratch + gate->n;
    int rc = run_mxfp4_fused_sdot(ctx, gate, x, gate_out);
    if (!rc) rc = run_mxfp4_fused_sdot(ctx, up, x, up_out);
    if (rc) return rc;
    #pragma omp parallel for num_threads(ctx->nthreads) schedule(static)
    for (size_t i = 0; i < gate->n; ++i) {
        float v = gate_out[i];
        gate_out[i] = (v / (1.0f + expf(-v))) * up_out[i];
    }
    return run_mxfp4_fused_sdot(ctx, down, gate_out, y);
}

static float half_to_float(uint16_t h) {
    _Float16 x;
    memcpy(&x, &h, sizeof(h));
    return (float)x;
}

static uint16_t float_to_half(float x) {
    _Float16 h = (_Float16)x;
    uint16_t bits;
    memcpy(&bits, &h, sizeof(bits));
    return bits;
}

int swfp4fp8_gemm_f16(swfp4fp8_context *ctx,
                      const swfp4fp8_matrix *w,
                      const uint16_t *a, size_t lda, uint16_t *c, size_t ldc,
                      size_t m, swfp4fp8_kernel kernel) {
    if (!ctx || !w || !a || !c || !m || lda < w->k || ldc < w->n)
        return EINVAL;
    float *af = aligned_zero(m * w->k * sizeof(float));
    float *cf = aligned_zero(m * w->n * sizeof(float));
    if (!af || !cf) { free(af); free(cf); return ENOMEM; }
    #pragma omp parallel for num_threads(ctx->nthreads) schedule(static)
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < w->k; ++j) af[i * w->k + j] = half_to_float(a[i * lda + j]);
    int rc = swfp4fp8_gemm_f32(ctx, w, af, w->k, cf, w->n, m, kernel);
    if (!rc) {
        #pragma omp parallel for num_threads(ctx->nthreads) schedule(static)
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < w->n; ++j) c[i * ldc + j] = float_to_half(cf[i * w->n + j]);
    }
    free(af);
    free(cf);
    return rc;
}

int swfp4fp8_unpack_codes(const swfp4fp8_matrix *w, uint8_t *dst) {
    if (!w || !dst) return EINVAL;
    if (w->format == SWFP4FP8_NVFP4_G16 || w->format == SWFP4FP8_MXFP4_G32) {
        memset(dst, 0, w->n * w->k / 2);
        size_t pairs = w->group_k / 2;
        for (size_t p = 0; p < w->n_pad / 16; ++p)
            for (size_t g = 0; g < w->groups; ++g)
                for (size_t q = 0; q < pairs; ++q)
                    for (size_t lane = 0; lane < 16; ++lane) {
                        size_t row = p * 16 + lane;
                        if (row >= w->n) continue;
                        uint8_t b = w->codes[((p * w->groups + g) * pairs + q) * 16 + lane];
                        if (w->format == SWFP4FP8_NVFP4_G16) {
                            dst[row * (w->k / 2) + (g * pairs + q)] = b;
                        } else {
                            uint8_t *db = dst + row * (w->k / 2) + g * 16;
                            size_t k0 = q * 2, k1 = k0 + 1;
                            uint8_t c0 = b & 15, c1 = b >> 4;
                            if (k0 < 16) db[k0] = (db[k0] & 0xf0) | c0;
                            else db[k0 - 16] = (db[k0 - 16] & 0x0f) | (uint8_t)(c0 << 4);
                            if (k1 < 16) db[k1] = (db[k1] & 0xf0) | c1;
                            else db[k1 - 16] = (db[k1 - 16] & 0x0f) | (uint8_t)(c1 << 4);
                        }
                    }
    } else {
        for (size_t p = 0; p < w->n_pad / 16; ++p)
            for (size_t kk = 0; kk < w->k; ++kk)
                for (size_t lane = 0; lane < 16; ++lane) {
                    size_t row = p * 16 + lane;
                    if (row < w->n) dst[row * w->k + kk] = w->codes[(p * w->k + kk) * 16 + lane];
                }
    }
    return 0;
}

const char *swfp4fp8_format_name(swfp4fp8_format f) {
    static const char *names[] = {"nvfp4-g16", "qpn8-tile32", "mxfp4-g32", "fp8-block128"};
    return (unsigned)f < 4 ? names[f] : "unknown";
}

const char *swfp4fp8_kernel_name(swfp4fp8_kernel k) {
    static const char *names[] = {"auto", "panel", "row", "fp8-ftz", "fp4-sdot", "mxfp4-fused-sdot"};
    return (unsigned)k < 6 ? names[k] : "unknown";
}
