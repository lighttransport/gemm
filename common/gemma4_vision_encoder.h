/*
 * gemma4_vision_encoder.h - Gemma4 vision encoder (mmproj)
 *
 * Usage:
 *   #define GEMMA4_VISION_IMPLEMENTATION
 *   #include "gemma4_vision_encoder.h"
 *
 * Dependencies: gguf_loader.h, ggml_dequant.h
 *
 * Architecture: ViT with RMSNorm, 2D RoPE, SwiGLU (GELU gate), avg pooling
 *   16 blocks, dim=768, heads=12, head_dim=64, ffn=3072
 *   patch_size=16, image=224 -> 196 patches -> 49 merged tokens -> proj to 2560
 */
#ifndef GEMMA4_VISION_ENCODER_H
#define GEMMA4_VISION_ENCODER_H

#include <stdint.h>
#include <stddef.h>
#include "gguf_loader.h"
#include "ggml_dequant.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    /* Attention: separate Q/K/V */
    qtensor attn_q_w, attn_k_w, attn_v_w, attn_out_w;
    qtensor attn_q_norm, attn_k_norm;   /* [head_dim] per-head RMSNorm */
    qtensor attn_post_norm;             /* [dim] */
    /* FFN: gate/up/down */
    qtensor ffn_gate_w, ffn_up_w, ffn_down_w;
    qtensor ffn_post_norm;              /* [dim] */
    /* Pre-norms (RMSNorm) */
    qtensor ln1_w;  /* pre-attention */
    qtensor ln2_w;  /* pre-FFN */
} g4v_block;

typedef struct {
    /* Hyperparameters */
    int n_blocks;
    int dim;          /* 768 */
    int n_heads;      /* 12 */
    int head_dim;     /* 64 */
    int ffn_dim;      /* 3072 */
    int patch_size;   /* 16 */
    int image_size;   /* 224 */
    int n_patches;    /* 196 */
    int proj_dim;     /* 2560 (LLM embedding dim) */
    int spatial_merge; /* 2 */
    int n_merged;     /* 49 */
    float ln_eps;
    int is_gemma4uv;  /* Gemma4 "unified vision" projector: no transformer blocks, 48x48 patches, custom 3-norm + linear projection */

    /* Patch embedding (conv2d) */
    qtensor patch_embd_w;  /* ViT: [16, 16, 3, dim]; Gemma4UV: [patch_size*patch_size*3, dim] (matmul) */
    qtensor patch_embd_b;  /* Gemma4UV: [dim] bias added after patch matmul (optional for ViT) */

    /* Gemma4UV: three LayerNorms + linear projection */
    qtensor patch_norm_1_w, patch_norm_1_b;  /* LayerNorm over 6912-dim (Gemma4UV) */
    qtensor patch_norm_2_w, patch_norm_2_b;  /* LayerNorm over dim (after matmul, Gemma4UV) */
    qtensor patch_norm_3_w, patch_norm_3_b;  /* LayerNorm over dim (after pos_emb, Gemma4UV) */

    /* Position embedding: 2D lookup tables */
    qtensor position_embd; /* [dim, n_pos, 2] — two tables for X/Y */
    int n_pos;             /* number of positions per axis */

    /* Transformer blocks (empty for Gemma4UV) */
    g4v_block *blocks;

    /* MM projection */
    qtensor mm_proj_w;   /* ViT: [dim, proj_dim]; Gemma4UV: [proj_dim, dim] */
} g4v_model;

g4v_model *g4v_load(gguf_context *mmproj_gguf);
void g4v_free(g4v_model *vm);

/* Encode an image. rgb is raw uint8 [height * width * 3] in RGB order.
 * Returns malloc'd float array of [n_merged * proj_dim] = [49 * 2560].
 * Caller must free the result. */
float *g4v_encode(g4v_model *vm, const uint8_t *rgb, int width, int height);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef GEMMA4_VISION_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

/* ---- AVX2 helpers ---- */

static inline float g4v_dot(const float *a, const float *b, int n) {
#ifdef __AVX2__
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 15 < n; i += 16) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),     _mm256_loadu_ps(b + i),     sum0);
        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), sum1);
    }
    sum0 = _mm256_add_ps(sum0, sum1);
    for (; i + 7 < n; i += 8)
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum0);
    /* horizontal sum */
    __m128 lo = _mm256_castps256_ps128(sum0);
    __m128 hi = _mm256_extractf128_ps(sum0, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float result = _mm_cvtss_f32(lo);
    for (; i < n; i++) result += a[i] * b[i];
    return result;
#else
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
#endif
}

static inline float g4v_sum_sq(const float *a, int n) {
    return g4v_dot(a, a, n);
}

static inline void g4v_vec_mul(float *out, const float *a, const float *b, int n) {
#ifdef __AVX2__
    int i = 0;
    for (; i + 7 < n; i += 8)
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    for (; i < n; i++) out[i] = a[i] * b[i];
#else
    for (int i = 0; i < n; i++) out[i] = a[i] * b[i];
#endif
}

static inline void g4v_vec_scale(float *x, float s, int n) {
#ifdef __AVX2__
    __m256 vs = _mm256_set1_ps(s);
    int i = 0;
    for (; i + 7 < n; i += 8)
        _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), vs));
    for (; i < n; i++) x[i] *= s;
#else
    for (int i = 0; i < n; i++) x[i] *= s;
#endif
}

static inline void g4v_vec_add(float *dst, const float *src, int n) {
#ifdef __AVX2__
    int i = 0;
    for (; i + 7 < n; i += 8)
        _mm256_storeu_ps(dst + i, _mm256_add_ps(_mm256_loadu_ps(dst + i), _mm256_loadu_ps(src + i)));
    for (; i < n; i++) dst[i] += src[i];
#else
    for (int i = 0; i < n; i++) dst[i] += src[i];
#endif
}

static inline void g4v_vec_fmadd(float *dst, float s, const float *src, int n) {
#ifdef __AVX2__
    __m256 vs = _mm256_set1_ps(s);
    int i = 0;
    for (; i + 7 < n; i += 8)
        _mm256_storeu_ps(dst + i, _mm256_fmadd_ps(vs, _mm256_loadu_ps(src + i), _mm256_loadu_ps(dst + i)));
    for (; i < n; i++) dst[i] += s * src[i];
#else
    for (int i = 0; i < n; i++) dst[i] += s * src[i];
#endif
}

/* ---- Threading helpers ---- */

static int g4v_n_threads(void) {
    int n = 4;
#ifdef _SC_NPROCESSORS_ONLN
    int hw = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (hw > 0) n = hw > 16 ? 16 : hw;
#endif
    return n;
}

typedef struct {
    const float *mat_f32;       /* used when mat_type == GGML_TYPE_F32 */
    const void *mat_quant;      /* used when mat_type == F16/BF16 (raw bytes) */
    int mat_type;               /* GGML_TYPE_F32 / F16 / BF16 */
    int elem_size;              /* bytes per element (4/2/2) */
    const float *inp;
    float *out;
    int n_cols, n_rows, N;
    int row_start, row_end;
} g4v_matmul_task;

#ifdef __AVX2__
/* Fused F16 dequant + dot: load 8 F16 elements, convert to F32, FMA with input. */
static inline float g4v_dot_f16(const uint16_t *w, const float *x, int n) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 15 < n; i += 16) {
        __m256 w0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + i)));
        __m256 w1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + i + 8)));
        sum0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x + i),     sum0);
        sum1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x + i + 8), sum1);
    }
    sum0 = _mm256_add_ps(sum0, sum1);
    for (; i + 7 < n; i += 8) {
        __m256 w0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + i)));
        sum0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x + i), sum0);
    }
    __m128 lo = _mm256_castps256_ps128(sum0);
    __m128 hi = _mm256_extractf128_ps(sum0, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float result = _mm_cvtss_f32(lo);
    for (; i < n; i++) result += ggml_fp16_to_fp32(w[i]) * x[i];
    return result;
}

/* Fused BF16 dequant + dot: BF16 is just F32 with low 16 bits zero. */
static inline float g4v_dot_bf16(const uint16_t *w, const float *x, int n) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 15 < n; i += 16) {
        __m256i wi0 = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(w + i)));
        __m256i wi1 = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(w + i + 8)));
        __m256 w0 = _mm256_castsi256_ps(_mm256_slli_epi32(wi0, 16));
        __m256 w1 = _mm256_castsi256_ps(_mm256_slli_epi32(wi1, 16));
        sum0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x + i),     sum0);
        sum1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x + i + 8), sum1);
    }
    sum0 = _mm256_add_ps(sum0, sum1);
    for (; i + 7 < n; i += 8) {
        __m256i wi = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(w + i)));
        __m256 w0 = _mm256_castsi256_ps(_mm256_slli_epi32(wi, 16));
        sum0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x + i), sum0);
    }
    __m128 lo = _mm256_castps256_ps128(sum0);
    __m128 hi = _mm256_extractf128_ps(sum0, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float result = _mm_cvtss_f32(lo);
    for (; i < n; i++) {
        uint32_t bits = (uint32_t)w[i] << 16;
        float wf;
        __builtin_memcpy(&wf, &bits, 4);
        result += wf * x[i];
    }
    return result;
}
#endif /* __AVX2__ */

static void *g4v_matmul_worker(void *arg) {
    g4v_matmul_task *t = (g4v_matmul_task *)arg;
    if (t->mat_type == GGML_TYPE_F32 || t->mat_f32) {
        for (int r = t->row_start; r < t->row_end; r++) {
            const float *row = t->mat_f32 + (size_t)r * t->n_cols;
            for (int tok = 0; tok < t->N; tok++) {
                const float *x = t->inp + tok * t->n_cols;
                t->out[tok * t->n_rows + r] = g4v_dot(row, x, t->n_cols);
            }
        }
    } else {
        /* Fused dequant + dot: read raw F16/BF16 weight bytes directly. */
#ifdef __AVX2__
        int is_bf16 = (t->mat_type == GGML_TYPE_BF16);
        const uint16_t *mat_q = (const uint16_t *)t->mat_quant;
        for (int r = t->row_start; r < t->row_end; r++) {
            const uint16_t *row = mat_q + (size_t)r * t->n_cols;
            for (int tok = 0; tok < t->N; tok++) {
                const float *x = t->inp + tok * t->n_cols;
                t->out[tok * t->n_rows + r] = is_bf16
                    ? g4v_dot_bf16(row, x, t->n_cols)
                    : g4v_dot_f16(row, x, t->n_cols);
            }
        }
#else
        /* Scalar fallback: dequant on the fly per element. */
        for (int r = t->row_start; r < t->row_end; r++) {
            const uint16_t *row = (const uint16_t *)t->mat_quant + (size_t)r * t->n_cols;
            for (int tok = 0; tok < t->N; tok++) {
                const float *x = t->inp + tok * t->n_cols;
                float acc = 0;
                for (int k = 0; k < t->n_cols; k++) {
                    float wf;
                    uint32_t bits = (uint32_t)row[k] << 16;  /* BF16 path; F16 is similar */
                    __builtin_memcpy(&wf, &bits, 4);
                    acc += wf * x[k];
                }
                t->out[tok * t->n_rows + r] = acc;
            }
        }
#endif
    }
    return NULL;
}

/* ---- Helpers ---- */

static void g4v_rmsnorm(float *out, const float *x, const qtensor *w, int n, float eps, float *tmp) {
    float ss = g4v_sum_sq(x, n);
    ss = 1.0f / sqrtf(ss / n + eps);
    float *wf = tmp;
    dequant_row(w->type, w->data, wf, n);
    /* out[i] = x[i] * ss * wf[i] */
#ifdef __AVX2__
    __m256 vss = _mm256_set1_ps(ss);
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vw = _mm256_loadu_ps(wf + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(vx, vss), vw));
    }
    for (; i < n; i++) out[i] = x[i] * ss * wf[i];
#else
    for (int i = 0; i < n; i++) out[i] = x[i] * ss * wf[i];
#endif
}

static void g4v_rmsnorm_inplace(float *x, const qtensor *w, int n, float eps, float *tmp) {
    g4v_rmsnorm(x, x, w, n, eps, tmp);
}

/* LayerNorm: out = (x - mean) / sqrt(var + eps) * weight + bias
 * Mean/var computed over the last dim (n). PyTorch LayerNorm. */
static void g4v_layernorm(float *out, const float *x, const qtensor *w, const qtensor *b,
                          int n, float eps, float *tmp) {
    float mean = 0.0f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    float var = 0.0f;
    for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d * d; }
    var /= n;
    float scale = 1.0f / sqrtf(var + eps);
    float *wf = tmp;
    float *bf = tmp + n;
    dequant_row(w->type, w->data, wf, n);
    dequant_row(b->type, b->data, bf, n);
    for (int i = 0; i < n; i++) out[i] = (x[i] - mean) * scale * wf[i] + bf[i];
}

/* AVX2 LayerNorm in-place: x = (x - mean) / sqrt(var + eps) * w + b.
 * Single-pass Welford-style mean+var, then in-place normalize+scale+shift.
 * n should be a multiple of 8 for best performance. */
static void g4v_layernorm_avx2_inplace(float *x, const float *w, const float *b, int n, float eps) {
#ifdef __AVX2__
    __m256 vsum = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < n; i += 8) vsum = _mm256_add_ps(vsum, _mm256_loadu_ps(x + i));
    float tail_sum = 0.0f;
    for (; i < n; i++) tail_sum += x[i];
    if (tail_sum != 0.0f) vsum = _mm256_add_ps(vsum, _mm256_set1_ps(tail_sum));
    /* horizontal sum */
    __m128 lo = _mm256_castps256_ps128(vsum);
    __m128 hi = _mm256_extractf128_ps(vsum, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    __m128 sums = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    float mean = _mm_cvtss_f32(sums) / (float)n;

    __m256 vmean = _mm256_set1_ps(mean);
    __m256 vvar = _mm256_setzero_ps();
    i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vd = _mm256_sub_ps(vx, vmean);
        vvar = _mm256_fmadd_ps(vd, vd, vvar);
    }
    float tail_var = 0.0f;
    for (; i < n; i++) { float d = x[i] - mean; tail_var += d * d; }
    if (tail_var != 0.0f) vvar = _mm256_add_ps(vvar, _mm256_set1_ps(tail_var));
    lo = _mm256_castps256_ps128(vvar);
    hi = _mm256_extractf128_ps(vvar, 1);
    lo = _mm_add_ps(lo, hi);
    shuf = _mm_movehdup_ps(lo);
    __m128 vars = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, vars);
    vars = _mm_add_ss(vars, shuf);
    float var = _mm_cvtss_f32(vars) / (float)n;
    float scale = 1.0f / sqrtf(var + eps);

    __m256 vscale = _mm256_set1_ps(scale);
    i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vw = _mm256_loadu_ps(w + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vd = _mm256_sub_ps(vx, vmean);
        __m256 vout = _mm256_fmadd_ps(_mm256_mul_ps(vd, vscale), vw, vb);
        _mm256_storeu_ps(x + i, vout);
    }
    for (; i < n; i++) x[i] = (x[i] - mean) * scale * w[i] + b[i];
#else
    float mean = 0.0f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    float var = 0.0f;
    for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d * d; }
    var /= n;
    float scale = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < n; i++) x[i] = (x[i] - mean) * scale * w[i] + b[i];
#endif
}

/* Simple matmul: out[n_rows] = mat[n_rows, n_cols] @ vec[n_cols] */
static void g4v_matvec(float *out, const qtensor *mat, const float *vec, int n_rows, float *tmp) {
    int n_cols = mat->n_cols;
    size_t row_bytes = 0;
    {
        int bs = 1, ts = 4;
        switch (mat->type) {
            case GGML_TYPE_F32:  bs = 1; ts = 4; break;
            case GGML_TYPE_F16:  bs = 1; ts = 2; break;
            case GGML_TYPE_BF16: bs = 1; ts = 2; break;
            default: bs = 1; ts = 4; break;
        }
        row_bytes = (size_t)((n_cols + bs - 1) / bs) * ts;
    }
    const uint8_t *base = (const uint8_t *)mat->data;
    for (int r = 0; r < n_rows; r++) {
        dequant_row(mat->type, base + r * row_bytes, tmp, n_cols);
        float dot = 0.0f;
        for (int c = 0; c < n_cols; c++) dot += tmp[c] * vec[c];
        out[r] = dot;
    }
}

/* Batched matmul: out[N, n_rows] = mat[n_rows, n_cols] @ inp[N, n_cols].
 * Dequantizes the full matrix once, then dot-products all N tokens.
 * Uses AVX2 dot products and pthreads for row parallelism. */
static void g4v_matmul_batch(float *out, const qtensor *mat, const float *inp,
                              int N, int n_rows, float *tmp) {
    int n_cols = mat->n_cols;
    size_t row_bytes = 0;
    {
        int bs = 1, ts = 4;
        switch (mat->type) {
            case GGML_TYPE_F32:  bs = 1; ts = 4; break;
            case GGML_TYPE_F16:  bs = 1; ts = 2; break;
            case GGML_TYPE_BF16: bs = 1; ts = 2; break;
            default: bs = 1; ts = 4; break;
        }
        row_bytes = (size_t)((n_cols + bs - 1) / bs) * ts;
    }
    const uint8_t *base = (const uint8_t *)mat->data;

    /* Set up the mat pointer.
     * Fast paths:
     *   - F32: use the data pointer directly, no copy, no dequant.
     *   - F16 / BF16: keep the raw bytes pointer. The worker will dequant
     *     on the fly inside the AVX2 dot product (fused dequant + matmul),
     *     avoiding a 100+MB intermediate F32 buffer + the dequant pass. */
    const float *mat_f32 = NULL;
    const void  *mat_quant = NULL;
    int owns_mat = 0;
    if (mat->type == GGML_TYPE_F32) {
        mat_f32 = (float *)mat->data;
    } else {
        mat_quant = mat->data;  /* raw F16/BF16; no copy */
    }
    (void)base; (void)row_bytes;  /* unused on the fused path */

    /* Threaded + AVX2 dot products (with on-the-fly F16/BF16 → F32 conversion
     * when the input weight is not F32). */
    struct timespec _mts0;
    clock_gettime(CLOCK_MONOTONIC, &_mts0);
    int nt = g4v_n_threads();
    if (nt > n_rows) nt = n_rows;
    if (nt <= 1) {
        /* Single-threaded fast path */
        g4v_matmul_task t = {0};
        t.mat_f32 = mat_f32;
        t.mat_quant = mat_quant;
        t.mat_type = mat->type;
        t.inp = inp;
        t.out = out;
        t.n_cols = n_cols;
        t.n_rows = n_rows;
        t.N = N;
        t.row_start = 0;
        t.row_end = n_rows;
        g4v_matmul_worker(&t);
    } else {
        pthread_t *threads = (pthread_t *)alloca(nt * sizeof(pthread_t));
        g4v_matmul_task *tasks = (g4v_matmul_task *)alloca(nt * sizeof(g4v_matmul_task));
        int rows_per = (n_rows + nt - 1) / nt;
        for (int i = 0; i < nt; i++) {
            tasks[i].mat_f32 = mat_f32;
            tasks[i].mat_quant = mat_quant;
            tasks[i].mat_type = mat->type;
            tasks[i].inp = inp;
            tasks[i].out = out;
            tasks[i].n_cols = n_cols;
            tasks[i].n_rows = n_rows;
            tasks[i].N = N;
            tasks[i].row_start = i * rows_per;
            tasks[i].row_end = (i + 1) * rows_per;
            if (tasks[i].row_end > n_rows) tasks[i].row_end = n_rows;
            pthread_create(&threads[i], NULL, g4v_matmul_worker, &tasks[i]);
        }
        for (int i = 0; i < nt; i++) pthread_join(threads[i], NULL);
    }
    struct timespec _mts1; clock_gettime(CLOCK_MONOTONIC, &_mts1);
#ifdef GEMMA4_PROFILE_MS
    {
        double _mat_t = (_mts1.tv_sec*1000.0+_mts1.tv_nsec/1e6) - (_mts0.tv_sec*1000.0+_mts0.tv_nsec/1e6);
        double _mat_gflops = (double)(2LL * n_rows * n_cols * N) / (_mat_t * 1e6);
        const char *tn = (mat->type == GGML_TYPE_F32 ? "F32" :
                          mat->type == GGML_TYPE_F16 ? "F16" :
                          mat->type == GGML_TYPE_BF16 ? "BF16" : "??");
        fprintf(stderr, "    g4v_matmul_batch matmul(%s): n_rows=%d n_cols=%d N=%d threads=%d -> %.2fms (%.1f GFLOPS)\n",
                tn, n_rows, n_cols, N, nt, _mat_t, _mat_gflops);
    }
#endif
    (void)owns_mat;  /* always 0 with fused path; left for future fallback */
}

/* ---- Loading ---- */

static qtensor g4v_load_tensor(gguf_context *g, const char *name, int required) {
    qtensor t = {0};
    int idx = -1;
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        if (strcmp(g->tensors[i].name.str, name) == 0) { idx = (int)i; break; }
    }
    if (idx < 0) {
        if (required) fprintf(stderr, "g4v: missing tensor %s\n", name);
        return t;
    }
    t.data = gguf_tensor_data(g, idx);
    t.type = g->tensors[idx].type;
    t.n_dims = g->tensors[idx].n_dims;
    for (int d = 0; d < 4; d++) t.dims[d] = g->tensors[idx].dims[d];
    /* For matrix: n_rows = product of dims[1..], n_cols = dims[0] */
    t.n_cols = (int)t.dims[0];
    t.n_rows = 1;
    for (int d = 1; d < t.n_dims; d++) t.n_rows *= (int)t.dims[d];
    return t;
}

g4v_model *g4v_load(gguf_context *g) {
    if (!g) return NULL;

    g4v_model *vm = (g4v_model *)calloc(1, sizeof(g4v_model));

    /* Read hyperparameters */
    vm->dim = 768;
    vm->n_heads = 12;
    vm->ffn_dim = 3072;
    vm->patch_size = 16;
    vm->image_size = 224;
    vm->proj_dim = 2560;
    vm->spatial_merge = 3;  /* Gemma4 uses kernel_size=3 avg pooling */
    vm->ln_eps = 1e-6f;

    /* Override from GGUF if available */
    {
        int idx;
        idx = gguf_find_key(g, "clip.vision.embedding_length");
        if (idx >= 0) vm->dim = g->kv[idx].value.u32;
        idx = gguf_find_key(g, "clip.vision.attention.head_count");
        if (idx >= 0) vm->n_heads = g->kv[idx].value.u32;
        idx = gguf_find_key(g, "clip.vision.feed_forward_length");
        if (idx >= 0) vm->ffn_dim = g->kv[idx].value.u32;
        idx = gguf_find_key(g, "clip.vision.patch_size");
        if (idx >= 0) vm->patch_size = g->kv[idx].value.u32;
        idx = gguf_find_key(g, "clip.vision.image_size");
        if (idx >= 0) vm->image_size = g->kv[idx].value.u32;
        idx = gguf_find_key(g, "clip.vision.projection_dim");
        if (idx >= 0) vm->proj_dim = g->kv[idx].value.u32;
        idx = gguf_find_key(g, "clip.vision.block_count");
        if (idx >= 0) vm->n_blocks = g->kv[idx].value.u32;
        idx = gguf_find_key(g, "clip.vision.attention.layer_norm_epsilon");
        if (idx >= 0) vm->ln_eps = g->kv[idx].value.f32;
        /* Detect Gemma4 "unified vision" projector */
        idx = gguf_find_key(g, "clip.vision.projector_type");
        if (idx >= 0) {
            const char *pt = g->kv[idx].value.str.str;
            if (pt && strcmp(pt, "gemma4uv") == 0) {
                vm->is_gemma4uv = 1;
                /* llama.cpp: patch_size *= spatial_merge (3), n_merge = 1 */
                vm->patch_size = vm->patch_size * vm->spatial_merge;
            }
        }
    }

    vm->head_dim = vm->dim / vm->n_heads;
    vm->n_patches = (vm->image_size / vm->patch_size) * (vm->image_size / vm->patch_size);
    {
        int ps_grid = vm->image_size / vm->patch_size;  /* 14 for ViT, 4 for Gemma4UV */
        if (vm->is_gemma4uv) {
            /* Gemma4UV: patch_size already includes spatial merge, so n_merged = n_patches */
            vm->n_merged = vm->n_patches;
        } else {
            vm->n_merged = (ps_grid / vm->spatial_merge) * (ps_grid / vm->spatial_merge);
        }
    }

    fprintf(stderr, "g4v: dim=%d heads=%d head_dim=%d ffn=%d blocks=%d\n",
            vm->dim, vm->n_heads, vm->head_dim, vm->ffn_dim, vm->n_blocks);
    fprintf(stderr, "g4v: patch=%d image=%d n_patches=%d merge=%d n_merged=%d proj=%d\n",
            vm->patch_size, vm->image_size, vm->n_patches, vm->spatial_merge, vm->n_merged, vm->proj_dim);

    /* Load global tensors */
    vm->patch_embd_w = g4v_load_tensor(g, "v.patch_embd.weight", 1);
    vm->patch_embd_b = g4v_load_tensor(g, "v.patch_embd.bias", 0);
    vm->position_embd = g4v_load_tensor(g, "v.position_embd.weight", 1);
    if (vm->position_embd.data) {
        vm->n_pos = (int)vm->position_embd.dims[1];
        fprintf(stderr, "g4v: position_embd: [%d, %d, %d]\n",
                (int)vm->position_embd.dims[0], vm->n_pos, (int)vm->position_embd.dims[2]);
    }

    vm->mm_proj_w = g4v_load_tensor(g, "mm.input_projection.weight", 1);

    if (vm->is_gemma4uv) {
        vm->patch_norm_1_w = g4v_load_tensor(g, "v.patch_norm.1.weight", 1);
        vm->patch_norm_1_b = g4v_load_tensor(g, "v.patch_norm.1.bias",   1);
        vm->patch_norm_2_w = g4v_load_tensor(g, "v.patch_norm.2.weight", 1);
        vm->patch_norm_2_b = g4v_load_tensor(g, "v.patch_norm.2.bias",   1);
        vm->patch_norm_3_w = g4v_load_tensor(g, "v.patch_norm.3.weight", 1);
        vm->patch_norm_3_b = g4v_load_tensor(g, "v.patch_norm.3.bias",   1);
        fprintf(stderr, "g4v: gemma4uv projector detected (effective patch_size=%d, n_merged=%d)\n",
                vm->patch_size, vm->n_merged);
    }

    /* Load blocks (zero for Gemma4UV) */
    vm->blocks = (g4v_block *)calloc(vm->n_blocks > 0 ? vm->n_blocks : 1, sizeof(g4v_block));
    for (int b = 0; b < vm->n_blocks; b++) {
        char name[128];
        #define G4V_LOAD(field, suffix) \
            snprintf(name, sizeof(name), "v.blk.%d." suffix ".weight", b); \
            vm->blocks[b].field = g4v_load_tensor(g, name, 1);

        G4V_LOAD(attn_q_w, "attn_q")
        G4V_LOAD(attn_k_w, "attn_k")
        G4V_LOAD(attn_v_w, "attn_v")
        G4V_LOAD(attn_out_w, "attn_out")
        G4V_LOAD(attn_q_norm, "attn_q_norm")
        G4V_LOAD(attn_k_norm, "attn_k_norm")
        G4V_LOAD(attn_post_norm, "attn_post_norm")
        G4V_LOAD(ffn_gate_w, "ffn_gate")
        G4V_LOAD(ffn_up_w, "ffn_up")
        G4V_LOAD(ffn_down_w, "ffn_down")
        G4V_LOAD(ffn_post_norm, "ffn_post_norm")
        G4V_LOAD(ln1_w, "ln1")
        G4V_LOAD(ln2_w, "ln2")
        #undef G4V_LOAD
    }

    return vm;
}

void g4v_free(g4v_model *vm) {
    if (!vm) return;
    free(vm->blocks); /* qtensor fields point into GGUF mmap, not owned */
    free(vm);
}

/* ---- Forward pass ---- */

/* Conv2D patch embedding: [H, W, 3] -> [n_patches, dim] */
static void g4v_patch_embed(g4v_model *vm, float *out, const float *inp,
                             int img_h, int img_w) {
    int ps = vm->patch_size;
    int dim = vm->dim;
    int ph = img_h / ps, pw = img_w / ps;
    int n_patches = ph * pw;

    /* Dequant all filters: [16, 16, 3, dim] -> [dim][3*16*16] */
    int filter_size = ps * ps * 3;
    float *filters = (float *)malloc((size_t)dim * filter_size * sizeof(float));
    dequant_row(vm->patch_embd_w.type, vm->patch_embd_w.data, filters, dim * filter_size);

    /* For each patch, extract pixels into contiguous buffer then dot with filters */
    float *patch_buf = (float *)malloc(filter_size * sizeof(float));
    for (int py = 0; py < ph; py++) {
        for (int px = 0; px < pw; px++) {
            int patch_idx = py * pw + px;
            float *patch_out = out + patch_idx * dim;

            /* Extract patch pixels: [3, ps, ps] in filter order */
            for (int c = 0; c < 3; c++) {
                for (int fy = 0; fy < ps; fy++) {
                    int iy = py * ps + fy;
                    for (int fx = 0; fx < ps; fx++) {
                        int ix = px * ps + fx;
                        patch_buf[c * ps * ps + fy * ps + fx] = inp[(iy * img_w + ix) * 3 + c];
                    }
                }
            }
            /* Dot product with each filter using AVX2 */
            for (int d = 0; d < dim; d++)
                patch_out[d] = g4v_dot(filters + d * filter_size, patch_buf, filter_size);
        }
    }
    free(patch_buf);
    free(filters);
}

/* Add 2D positional embeddings via lookup tables */
static void g4v_add_pos_embd(g4v_model *vm, float *patches, int ph, int pw) {
    int dim = vm->dim;
    int n_pos = vm->n_pos;

    /* position_embd: [dim, n_pos, 2] — table 0 for X, table 1 for Y */
    float *pos_data = (float *)malloc((size_t)dim * n_pos * 2 * sizeof(float));
    dequant_row(vm->position_embd.type, vm->position_embd.data, pos_data, dim * n_pos * 2);

    float *tbl_x = pos_data;                          /* [dim, n_pos] */
    float *tbl_y = pos_data + (size_t)dim * n_pos;    /* [dim, n_pos] */

    for (int py = 0; py < ph; py++) {
        for (int px = 0; px < pw; px++) {
            int patch_idx = py * pw + px;
            float *p = patches + patch_idx * dim;
            g4v_vec_add(p, tbl_x + px * dim, dim);
            g4v_vec_add(p, tbl_y + py * dim, dim);
        }
    }
    free(pos_data);
}

/* Per-head RMSNorm on [n_heads, head_dim] packed as [n_heads * head_dim] */
static void g4v_head_norm(float *x, int n_heads, int head_dim, const qtensor *w, float eps, float *tmp) {
    float norm_w[128]; /* head_dim up to 128 */
    dequant_row(w->type, w->data, norm_w, head_dim);
    for (int h = 0; h < n_heads; h++) {
        float *xh = x + h * head_dim;
        float ss = g4v_sum_sq(xh, head_dim);
        ss = 1.0f / sqrtf(ss / head_dim + eps);
        /* xh[i] = xh[i] * ss * norm_w[i] */
#ifdef __AVX2__
        __m256 vss = _mm256_set1_ps(ss);
        int i = 0;
        for (; i + 7 < head_dim; i += 8) {
            __m256 vx = _mm256_loadu_ps(xh + i);
            __m256 vw = _mm256_loadu_ps(norm_w + i);
            _mm256_storeu_ps(xh + i, _mm256_mul_ps(_mm256_mul_ps(vx, vss), vw));
        }
        for (; i < head_dim; i++) xh[i] = xh[i] * ss * norm_w[i];
#else
        for (int i = 0; i < head_dim; i++) xh[i] = xh[i] * ss * norm_w[i];
#endif
    }
}

/* Apply 2D RoPE (NeoX ordering) to Q or K: first half uses X coords, second half uses Y coords */
static void g4v_rope_2d(float *qk, int n_heads, int head_dim, int n_tokens,
                         const int *pos_x, const int *pos_y, float theta) {
    int half = head_dim / 2;
    for (int t = 0; t < n_tokens; t++) {
        for (int h = 0; h < n_heads; h++) {
            float *v = qk + (t * n_heads + h) * head_dim;
            /* First half: X-axis RoPE */
            for (int j = 0; j < half / 2; j++) {
                float freq = (float)pos_x[t] / powf(theta, (float)(2 * j) / half);
                float cos_v = cosf(freq), sin_v = sinf(freq);
                float r0 = v[j], r1 = v[j + half / 2];
                v[j]            = r0 * cos_v - r1 * sin_v;
                v[j + half / 2] = r0 * sin_v + r1 * cos_v;
            }
            /* Second half: Y-axis RoPE */
            for (int j = 0; j < half / 2; j++) {
                float freq = (float)pos_y[t] / powf(theta, (float)(2 * j) / half);
                float cos_v = cosf(freq), sin_v = sinf(freq);
                int idx0 = half + j;
                int idx1 = half + j + half / 2;
                float r0 = v[idx0], r1 = v[idx1];
                v[idx0] = r0 * cos_v - r1 * sin_v;
                v[idx1] = r0 * sin_v + r1 * cos_v;
            }
        }
    }
}

/* ViT block forward for all N tokens at once.
 * attn_scratch: caller-allocated [N*dim*6 + N*N] floats
 * ffn_scratch:  caller-allocated [N*ff*3] floats */
static void g4v_block_forward(g4v_model *vm, g4v_block *blk, float *x, int N,
                               const int *pos_x, const int *pos_y, float *tmp,
                               float *attn_scratch, float *ffn_scratch) {
    int dim = vm->dim;
    int n_heads = vm->n_heads;
    int hd = vm->head_dim;
    float eps = vm->ln_eps;

    float *xn = attn_scratch;
    float *q  = attn_scratch + N * dim;
    float *k  = attn_scratch + N * dim * 2;
    float *v  = attn_scratch + N * dim * 3;
    float *att_out = attn_scratch + N * dim * 4;
    float *proj_out = attn_scratch + N * dim * 5;
    float *scores = attn_scratch + N * dim * 6; /* [N * N] */

    /* Pre-attention RMSNorm */
    for (int t = 0; t < N; t++)
        g4v_rmsnorm(xn + t * dim, x + t * dim, &blk->ln1_w, dim, eps, tmp);

    /* Q/K/V projections */
    g4v_matmul_batch(q, &blk->attn_q_w, xn, N, dim, tmp);
    g4v_matmul_batch(k, &blk->attn_k_w, xn, N, dim, tmp);
    g4v_matmul_batch(v, &blk->attn_v_w, xn, N, dim, tmp);

    /* Q/K per-head RMSNorm */
    for (int t = 0; t < N; t++) {
        g4v_head_norm(q + t * dim, n_heads, hd, &blk->attn_q_norm, eps, tmp);
        g4v_head_norm(k + t * dim, n_heads, hd, &blk->attn_k_norm, eps, tmp);
    }

    /* 2D RoPE */
    g4v_rope_2d(q, n_heads, hd, N, pos_x, pos_y, 100.0f); /* Gemma4 vision: theta=100 */
    g4v_rope_2d(k, n_heads, hd, N, pos_x, pos_y, 100.0f);

    /* Multi-head attention (all N tokens attend to all N tokens) */
    {
        memset(att_out, 0, N * dim * sizeof(float));
        for (int h = 0; h < n_heads; h++) {
            /* Compute attention scores: scores[i][j] = dot(q_i, k_j) */
            for (int i = 0; i < N; i++) {
                const float *qi = q + i * dim + h * hd;
                for (int j = 0; j < N; j++)
                    scores[i * N + j] = g4v_dot(qi, k + j * dim + h * hd, hd);
            }
            /* Softmax per row */
            for (int i = 0; i < N; i++) {
                float *s = scores + i * N;
                float max_s = s[0];
                for (int j = 1; j < N; j++) if (s[j] > max_s) max_s = s[j];
                float sum_e = 0.0f;
                for (int j = 0; j < N; j++) { s[j] = expf(s[j] - max_s); sum_e += s[j]; }
                g4v_vec_scale(s, 1.0f / sum_e, N);
            }
            /* Weighted sum of values: att_out[i] += sum_j(s[j] * v[j]) */
            for (int i = 0; i < N; i++) {
                float *out_i = att_out + i * dim + h * hd;
                const float *s = scores + i * N;
                for (int j = 0; j < N; j++)
                    g4v_vec_fmadd(out_i, s[j], v + j * dim + h * hd, hd);
            }
        }
    }

    /* Output projection (proj_out already allocated in attn_scratch) */
    g4v_matmul_batch(proj_out, &blk->attn_out_w, att_out, N, dim, tmp);

    /* Post-attention norm */
    for (int t = 0; t < N; t++)
        g4v_rmsnorm_inplace(proj_out + t * dim, &blk->attn_post_norm, dim, eps, tmp);

    /* Residual add */
    g4v_vec_add(x, proj_out, N * dim);

    /* --- FFN --- */
    /* Pre-FFN RMSNorm */
    for (int t = 0; t < N; t++)
        g4v_rmsnorm(xn + t * dim, x + t * dim, &blk->ln2_w, dim, eps, tmp);

    /* Gate/Up projections */
    int ff = vm->ffn_dim;
    float *gate = ffn_scratch;
    float *up   = ffn_scratch + N * ff;
    float *ffn_out = ffn_scratch + N * ff * 2;

    g4v_matmul_batch(gate, &blk->ffn_gate_w, xn, N, ff, tmp);
    g4v_matmul_batch(up,   &blk->ffn_up_w,   xn, N, ff, tmp);

    /* GELU(gate) * up */
    {
        int total = N * ff;
#ifdef __AVX2__
        /* GELU approx: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
         * But exact GELU uses erf. We keep erff for accuracy, vectorize the mul. */
        for (int i = 0; i < total; i++) {
            float g = gate[i];
            gate[i] = g * 0.5f * (1.0f + erff(g * 0.7071067811865476f));
        }
        g4v_vec_mul(ffn_out, gate, up, total);
#else
        for (int i = 0; i < total; i++) {
            float g = gate[i];
            float gelu_g = g * 0.5f * (1.0f + erff(g * 0.7071067811865476f));
            ffn_out[i] = gelu_g * up[i];
        }
#endif
    }

    /* Down projection — reuse proj_out (attention is done, safe to overwrite) */
    float *ffn_proj = proj_out;
    g4v_matmul_batch(ffn_proj, &blk->ffn_down_w, ffn_out, N, dim, tmp);

    /* Post-FFN norm */
    for (int t = 0; t < N; t++)
        g4v_rmsnorm_inplace(ffn_proj + t * dim, &blk->ffn_post_norm, dim, eps, tmp);

    /* Residual add */
    g4v_vec_add(x, ffn_proj, N * dim);

    /* scratch owned by caller */
}

/* Average pooling: [ph, pw, dim] -> [ph/k, pw/k, dim] */
static void g4v_avg_pool(float *out, const float *in, int ph, int pw, int dim, int kernel) {
    int oh = ph / kernel, ow = pw / kernel;
    float inv_area = 1.0f / (kernel * kernel);
    for (int y = 0; y < oh; y++) {
        for (int x = 0; x < ow; x++) {
            float *o = out + (y * ow + x) * dim;
            memset(o, 0, dim * sizeof(float));
            for (int ky = 0; ky < kernel; ky++) {
                for (int kx = 0; kx < kernel; kx++) {
                    int iy = y * kernel + ky;
                    int ix = x * kernel + kx;
                    const float *p = in + (iy * pw + ix) * dim;
                    g4v_vec_add(o, p, dim);
                }
            }
            g4v_vec_scale(o, inv_area, dim);
        }
    }
}

/* Gemma4 "unified vision" (Gemma4UV) encode pipeline.
 * Mirrors llama.cpp tools/mtmd/models/gemma4uv.cpp.
 *   im2col: [ps*ps*3, n_patches] (ps = patch_size * spatial_merge = 48 for 12B)
 *   patch_norm_1 (LayerNorm over 6912-dim, per patch)
 *   matmul: patch_embd_w [dim, 6912] @ [6912, n_patches] -> [dim, n_patches]
 *   add patch_embd_b [dim]
 *   patch_norm_2 (LayerNorm over dim)
 *   add 2D position embedding (X, Y lookups)
 *   patch_norm_3 (LayerNorm over dim)
 *   RMSNorm over dim (raw, no weight)
 *   matmul: mm.input_projection [proj_dim, dim] @ [dim, n_patches] -> [proj_dim, n_patches]
 * Output: n_merged tokens of proj_dim each.
 *
 * Note: GGUF stores weight matrices in a transposed layout (first dim = contraction,
 * second dim = output). For math: out = mat_math @ inp, where mat_math is [out, in],
 * the GGUF stores it as [in, out] = mat.n_rows=in, mat.n_cols=out. The matmul we need
 * is out[t, r] = sum_c mat_gguf[c, r] * inp[t, c] = (inp @ mat_gguf.T)[t, r].
 */
/* Helper: return F32 pointer to qtensor data without copy. Returns NULL if not F32. */
static const float *g4v_f32_ptr(const qtensor *t) {
    if (t->type == GGML_TYPE_F32) return (const float *)t->data;
    return NULL;
}

/* Helper: like g4v_f32_ptr but returns a writable pointer; dequantizes into a new
 * malloc'd buffer if the tensor is not F32. Caller must free the returned pointer
 * if it is non-NULL and the tensor was not F32 (use g4v_f32_owned_free to check). */
static float *g4v_f32_get_or_dequant(const qtensor *t, int n) {
    if (t->type == GGML_TYPE_F32) return (float *)t->data;  /* borrowed; do not free */
    float *out = (float *)malloc((size_t)n * sizeof(float));
    dequant_row(t->type, t->data, out, n);
    return out;
}

static float *g4v_encode_gemma4uv(g4v_model *vm, const float *img_norm, int width, int height) {
    int dim = vm->dim;
    int ps = vm->patch_size;            /* 48 for 12B */
    int ph = height / ps, pw = width / ps;
    int N = ph * pw;                    /* 16 for 12B */
    int proj_dim = vm->proj_dim;
    float eps = vm->ln_eps;
    int patch_dim = ps * ps * 3;        /* 6912 for 12B */

    fprintf(stderr, "g4v_encode_gemma4uv: %dx%d ps=%d -> %d patches\n", width, height, ps, N);
#ifdef GEMMA4_PROFILE_MS
    /* STEP_BEGIN(i) at start of section i, STEP_END(i) at end of section i.
     * Records the wall time between BEGIN and END into _pt[i]. */
    double _pt[8] = {0}; const char *_pn[8] = {"im2col","LN1","MM1","LN2","pos","LN3","RMS","MM2"};
    double _pt_t[8] = {0}; int _pt_active[8] = {0};
#define STEP_BEGIN(i) do { struct timespec _now; clock_gettime(CLOCK_MONOTONIC, &_now); _pt_t[i] = _now.tv_sec*1000.0 + _now.tv_nsec/1e6; _pt_active[i] = 1; } while (0)
#define STEP_END(i) do { if (_pt_active[i]) { struct timespec _now; clock_gettime(CLOCK_MONOTONIC, &_now); double _ms = _now.tv_sec*1000.0 + _now.tv_nsec/1e6; _pt[i] += _ms - _pt_t[i]; _pt_active[i] = 0; } } while (0)
#else
#define STEP_BEGIN(i) (void)0
#define STEP_END(i) (void)0
#endif

    float *patches_im2col = (float *)malloc((size_t)N * patch_dim * sizeof(float));
    float *patches        = (float *)malloc((size_t)N * dim      * sizeof(float));
    float *proj_patches   = (float *)malloc((size_t)N * proj_dim * sizeof(float));
    float *tmp            = (float *)malloc((size_t)patch_dim * 4 * sizeof(float));
    if (!patches_im2col || !patches || !proj_patches || !tmp) {
        fprintf(stderr, "g4v_encode_gemma4uv: alloc failed\n");
        free(patches_im2col); free(patches); free(proj_patches); free(tmp);
        return NULL;
    }

    /* 1. im2col: [patch_dim, N] layout — patches_im2col[patch * N + i] = pixel value */
    STEP_BEGIN(0);
    for (int py = 0; py < ph; py++) {
        for (int px = 0; px < pw; px++) {
            int patch_idx = py * pw + px;
            for (int c = 0; c < 3; c++) {
                for (int fy = 0; fy < ps; fy++) {
                    for (int fx = 0; fx < ps; fx++) {
                        int iy = py * ps + fy;
                        int ix = px * ps + fx;
                        float v = img_norm[(iy * width + ix) * 3 + c];
                        int off = c * ps * ps + fy * ps + fx;
                        patches_im2col[patch_idx * patch_dim + off] = v;
                    }
                }
            }
        }
    }
    STEP_END(0);

    /* 2. patch_norm_1 (LayerNorm over patch_dim = 6912) */
    STEP_BEGIN(1);
    {
        float *w = g4v_f32_get_or_dequant(&vm->patch_norm_1_w, patch_dim);
        float *b = g4v_f32_get_or_dequant(&vm->patch_norm_1_b, patch_dim);
        int w_owned = (w != (float *)vm->patch_norm_1_w.data);
        int b_owned = (b != (float *)vm->patch_norm_1_b.data);
        for (int p = 0; p < N; p++)
            g4v_layernorm_avx2_inplace(patches_im2col + p * patch_dim, w, b, patch_dim, eps);
        if (w_owned) free(w);
        if (b_owned) free(b);
    }

    /* 3. matmul: g4v_matmul_batch computes out = inp @ mat.T.
     *    qtensor: n_rows=dim, n_cols=patch_dim (loaded reversed from GGUF dims=[patch_dim, dim]).
     *    patches[N, dim] = im2col[N, patch_dim] @ mat[dim, patch_dim].T = im2col @ mat_view.T
     *    g4v_matmul_batch handles this directly with n_rows=dim. */
    STEP_BEGIN(2);
    {
        int mat_rows = vm->patch_embd_w.n_rows;     /* dim */
        int mat_cols = vm->patch_embd_w.n_cols;     /* patch_dim */
        if (mat_rows != dim || mat_cols != patch_dim) {
            fprintf(stderr, "g4v_encode_gemma4uv: patch_embd_w shape mismatch (%dx%d vs %dx%d)\n",
                    mat_rows, mat_cols, dim, patch_dim);
            free(patches_im2col); free(patches); free(proj_patches); free(tmp);
            return NULL;
        }
        g4v_matmul_batch(patches, &vm->patch_embd_w, patches_im2col, N, dim, tmp);
    }
    STEP_END(2);
    
    /* 4. add patch_embd_b [dim] */
    if (vm->patch_embd_b.data) {
        float *b = g4v_f32_get_or_dequant(&vm->patch_embd_b, dim);
        int b_owned = (b != (float *)vm->patch_embd_b.data);
        for (int p = 0; p < N; p++)
            for (int d = 0; d < dim; d++) patches[p * dim + d] += b[d];
        if (b_owned) free(b);
    }

    /* 5. patch_norm_2 (LayerNorm over dim) */
    STEP_BEGIN(3);
    {
        float *w = g4v_f32_get_or_dequant(&vm->patch_norm_2_w, dim);
        float *b = g4v_f32_get_or_dequant(&vm->patch_norm_2_b, dim);
        int w_owned = (w != (float *)vm->patch_norm_2_w.data);
        int b_owned = (b != (float *)vm->patch_norm_2_b.data);
        for (int p = 0; p < N; p++)
            g4v_layernorm_avx2_inplace(patches + p * dim, w, b, dim, eps);
        if (w_owned) free(w);
        if (b_owned) free(b);
    }
    STEP_END(3);

    free(patches_im2col);
    patches_im2col = NULL;

    /* 6. add 2D position embedding (X, Y lookups from [dim, n_pos, 2]) */
    STEP_BEGIN(4);
    {
        int n_pos = vm->n_pos;
        int owns_pos = 0;
        float *pos_data = NULL;
        if (vm->position_embd.type == GGML_TYPE_F32) {
            pos_data = (float *)vm->position_embd.data;  /* direct, no copy */
        } else {
            pos_data = (float *)malloc((size_t)dim * n_pos * 2 * sizeof(float));
            owns_pos = 1;
            dequant_row(vm->position_embd.type, vm->position_embd.data, pos_data, dim * n_pos * 2);
        }
        float *tbl_x = pos_data;
        float *tbl_y = pos_data + (size_t)dim * n_pos;
        for (int py = 0; py < ph; py++) {
            for (int px = 0; px < pw; px++) {
                int patch_idx = py * pw + px;
                float *p = patches + patch_idx * dim;
                const float *src_x = tbl_x + (size_t)px * dim;
                const float *src_y = tbl_y + (size_t)py * dim;
                for (int d = 0; d < dim; d++) p[d] += src_x[d] + src_y[d];
            }
        }
        if (owns_pos) free(pos_data);
    }
    STEP_END(4);

    /* 7. patch_norm_3 (LayerNorm over dim) */
    STEP_BEGIN(5);
    {
        float *w = g4v_f32_get_or_dequant(&vm->patch_norm_3_w, dim);
        float *b = g4v_f32_get_or_dequant(&vm->patch_norm_3_b, dim);
        int w_owned = (w != (float *)vm->patch_norm_3_w.data);
        int b_owned = (b != (float *)vm->patch_norm_3_b.data);
        for (int p = 0; p < N; p++)
            g4v_layernorm_avx2_inplace(patches + p * dim, w, b, dim, eps);
        if (w_owned) free(w);
        if (b_owned) free(b);
    }
    STEP_END(5);

    /* 8. RMSNorm over dim (raw, no learned weight) */
    STEP_BEGIN(6);
    for (int p = 0; p < N; p++) {
        float *x = patches + p * dim;
        float ss = g4v_sum_sq(x, dim);
        float scale = 1.0f / sqrtf(ss / dim + eps);
#ifdef __AVX2__
        __m256 vsc = _mm256_set1_ps(scale);
        int i = 0;
        for (; i + 7 < dim; i += 8)
            _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), vsc));
        for (; i < dim; i++) x[i] *= scale;
#else
        g4v_vec_scale(x, scale, dim);
#endif
    }
    STEP_END(6);

    /* 9. mm.input_projection: out = patches @ mat_view.T
     *    qtensor: n_rows=proj_dim, n_cols=dim (GGUF dims=[dim, proj_dim]). */
    STEP_BEGIN(7);
    {
        int mat_rows = vm->mm_proj_w.n_rows;     /* proj_dim */
        int mat_cols = vm->mm_proj_w.n_cols;     /* dim */
        if (mat_rows != proj_dim || mat_cols != dim) {
            fprintf(stderr, "g4v_encode_gemma4uv: mm_proj_w shape mismatch (%dx%d vs %dx%d)\n",
                    mat_rows, mat_cols, proj_dim, dim);
            free(patches); free(proj_patches); free(tmp);
            return NULL;
        }
        g4v_matmul_batch(proj_patches, &vm->mm_proj_w, patches, N, proj_dim, tmp);
    }
    STEP_END(7);

    free(patches);
    free(tmp);
#ifdef GEMMA4_PROFILE_MS
    {
        fprintf(stderr, "  gemma4uv step timings: ");
        double _total = 0;
        for (int _i=0;_i<8;_i++) { fprintf(stderr, "%s=%.1fms ", _pn[_i], _pt[_i]); _total += _pt[_i]; }
        fprintf(stderr, "total=%.1fms\n", _total);
    }
#endif
    return proj_patches;
}

float *g4v_encode(g4v_model *vm, const uint8_t *rgb, int width, int height) {
    if (!vm || !rgb) return NULL;

    int dim = vm->dim;
    int ps = vm->patch_size;
    int merge = vm->spatial_merge;
    int ph = height / ps, pw = width / ps;
    int N = ph * pw;
    int n_merged = (ph / merge) * (pw / merge);
    float *img_norm = NULL;
    float *patches = NULL;
    int *pos_x = NULL;
    int *pos_y = NULL;
    float *tmp = NULL;
    float *attn_scratch = NULL;
    float *ffn_scratch = NULL;
    float *pooled = NULL;
    float *projected = NULL;

    fprintf(stderr, "g4v_encode: %dx%d -> %d patches -> %d tokens\n",
            width, height, N, n_merged);

    /* 1. Normalize image: pixels * 2 - 1 (Gemma4 specific) */
    img_norm = (float *)malloc(height * width * 3 * sizeof(float));
    if (!img_norm) {
        fprintf(stderr, "g4v_encode: img_norm alloc failed (%d x %d)\n", width, height);
        goto fail;
    }
    for (int i = 0; i < height * width * 3; i++)
        img_norm[i] = ((float)rgb[i] / 255.0f) * 2.0f - 1.0f;

    /* Gemma4UV (12B) uses a different pipeline: im2col -> 3 LayerNorms -> matmul */
    if (vm->is_gemma4uv) {
        projected = g4v_encode_gemma4uv(vm, img_norm, width, height);
        free(img_norm);
        return projected;
    }

    /* 2. Patch embedding (conv2d stride=patch_size) */
    patches = (float *)calloc(N * dim, sizeof(float));
    if (!patches) {
        fprintf(stderr, "g4v_encode: patches alloc failed (N=%d dim=%d)\n", N, dim);
        goto fail;
    }
    g4v_patch_embed(vm, patches, img_norm, height, width);
    free(img_norm);
    img_norm = NULL;

    /* patch embedding done */

    /* 3. Add 2D positional embeddings */
    g4v_add_pos_embd(vm, patches, ph, pw);

    /* 4. Build position arrays for 2D RoPE */
    pos_x = (int *)malloc(N * sizeof(int));
    pos_y = (int *)malloc(N * sizeof(int));
    if (!pos_x || !pos_y) {
        fprintf(stderr, "g4v_encode: position alloc failed (N=%d)\n", N);
        goto fail;
    }
    for (int y = 0; y < ph; y++)
        for (int x = 0; x < pw; x++) {
            pos_x[y * pw + x] = x;
            pos_y[y * pw + x] = y;
        }

    /* 5. ViT transformer blocks — allocate scratch once for all blocks */
    tmp = (float *)malloc(dim * 16 * sizeof(float));
    attn_scratch = (float *)malloc(((size_t)N * dim * 6 + (size_t)N * N) * sizeof(float));
    ffn_scratch  = (float *)malloc((size_t)N * vm->ffn_dim * 3 * sizeof(float));
    if (!tmp || !attn_scratch || !ffn_scratch) {
        fprintf(stderr, "g4v_encode: block scratch alloc failed (N=%d dim=%d ffn=%d)\n",
                N, dim, vm->ffn_dim);
        goto fail;
    }
    for (int b = 0; b < vm->n_blocks; b++) {
        g4v_block_forward(vm, &vm->blocks[b], patches, N, pos_x, pos_y, tmp,
                          attn_scratch, ffn_scratch);
    }
    free(attn_scratch); free(ffn_scratch);
    attn_scratch = NULL; ffn_scratch = NULL;
    free(pos_x); free(pos_y); free(tmp);
    pos_x = NULL; pos_y = NULL; tmp = NULL;

    /* 6. Average pooling: [ph, pw, dim] -> [ph/merge, pw/merge, dim] */
    pooled = (float *)malloc(n_merged * dim * sizeof(float));
    if (!pooled) {
        fprintf(stderr, "g4v_encode: pooled alloc failed (tokens=%d dim=%d)\n", n_merged, dim);
        goto fail;
    }
    g4v_avg_pool(pooled, patches, ph, pw, dim, merge);
    free(patches);
    patches = NULL;

    /* 7. Scale by sqrt(dim) */
    {
        float scale = sqrtf((float)dim);
        g4v_vec_scale(pooled, scale, n_merged * dim);
    }

    /* pooled and scaled */

    /* 8. MM projection: [dim] -> [proj_dim] per token */
    int proj_dim = vm->proj_dim;
    projected = (float *)malloc(n_merged * proj_dim * sizeof(float));
    if (!projected) {
        fprintf(stderr, "g4v_encode: projected alloc failed (tokens=%d dim=%d)\n", n_merged, proj_dim);
        goto fail;
    }
    tmp = (float *)malloc(dim * sizeof(float));
    if (!tmp) {
        fprintf(stderr, "g4v_encode: projection tmp alloc failed (dim=%d)\n", dim);
        goto fail;
    }
    g4v_matmul_batch(projected, &vm->mm_proj_w, pooled, n_merged, proj_dim, tmp);
    free(pooled); free(tmp);
    pooled = NULL; tmp = NULL;

    /* 9. Final RMSNorm on projected embeddings (embedding_post_projection_norm)
     * This is a raw RMSNorm without learned weights */
    {
        float eps = vm->ln_eps;
        for (int t = 0; t < n_merged; t++) {
            float *p = projected + t * proj_dim;
            float ss = g4v_sum_sq(p, proj_dim);
            ss = 1.0f / sqrtf(ss / proj_dim + eps);
            g4v_vec_scale(p, ss, proj_dim);
        }
    }

    /* output: n_merged tokens of proj_dim dims */
    return projected;

fail:
    free(projected);
    free(pooled);
    free(tmp);
    free(ffn_scratch);
    free(attn_scratch);
    free(pos_y);
    free(pos_x);
    free(patches);
    free(img_norm);
    return NULL;
}

#endif /* GEMMA4_VISION_IMPLEMENTATION */
#endif /* GEMMA4_VISION_ENCODER_H */
