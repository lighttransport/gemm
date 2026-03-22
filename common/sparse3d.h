/*
 * sparse3d.h - Sparse 3D tensor infrastructure for TRELLIS.2 inference
 *
 * Usage:
 *   #define SPARSE3D_IMPLEMENTATION
 *   #include "sparse3d.h"
 *
 * Dependencies: ggml_dequant.h (for qtensor, dequant)
 *
 * API:
 *   sp3d_tensor *sp3d_create(coords, feats, N, C, batch_size);
 *   void         sp3d_free(sp3d_tensor *t);
 *   sp3d_tensor *sp3d_replace_feats(t, new_feats, new_C);
 *   sp3d_tensor *sp3d_clone(t);
 *   sp3d_tensor *sp3d_cat_feats(a, b);
 *
 *   void sp3d_linear(dst, t, W, bias, out_C, n_threads);
 *   void sp3d_layernorm(dst, src, w, b, N, C, eps);
 *   void sp3d_gelu(x, count);
 *   void sp3d_silu(x, count);
 *
 *   void sp3d_conv3d_forward(dst, t, weight, bias, in_C, out_C, kernel_size, n_threads);
 *   void sp3d_attention(out, qkv, t, n_heads, head_dim, n_threads);
 *   void sp3d_rope_3d(qk, t, n_heads, head_dim, dim_stride, rope_freqs, n_freqs);
 *
 *   sp3d_tensor *sp3d_downsample(t, factor, pool_mode);
 *   sp3d_tensor *sp3d_upsample(t, factor, target_coords, target_N);
 *
 * The SparseTensor stores N occupied voxels with:
 *   coords[N][4] = (batch_idx, z, y, x)  int32
 *   feats[N][C]  = feature vectors         float32
 *   batch_starts[B+1] = CSR-style batch boundaries
 *
 * All operations are CPU. Compute-intensive ops (conv, attention) have
 * clear CUDA extension points marked in comments.
 */
#ifndef SPARSE3D_H
#define SPARSE3D_H

#include <stdint.h>
#include <stddef.h>
#include "ggml_dequant.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Reuse qtensor if not already defined */
#ifndef TRANSFORMER_H
#ifndef DEPTH_ANYTHING3_H
#ifndef DINOV3_H
typedef struct {
    void    *data;
    uint32_t type;
    int      n_rows;
    int      n_cols;
    int      n_dims;
    uint64_t dims[4];
} qtensor;
#endif
#endif
#endif

/* ---- Spatial hash table (opaque) ---- */
typedef struct sp3d_hash sp3d_hash;

/* ---- Sparse 3D Tensor ---- */
typedef struct {
    int32_t *coords;        /* [N, 4] row-major: (batch, z, y, x) per voxel */
    float   *feats;         /* [N, C] row-major: feature vector per voxel */
    int      N;             /* total occupied voxels across all batches */
    int      C;             /* feature channels */
    int      batch_size;    /* B */
    int     *batch_starts;  /* [B+1] CSR offsets: batch b owns [batch_starts[b], batch_starts[b+1]) */
    sp3d_hash *hash;        /* lazy-built spatial hash (NULL until needed) */
} sp3d_tensor;

/* ---- Lifecycle ---- */
sp3d_tensor *sp3d_create(const int32_t *coords, const float *feats,
                          int N, int C, int batch_size);
void         sp3d_free(sp3d_tensor *t);
sp3d_tensor *sp3d_replace_feats(const sp3d_tensor *t, const float *new_feats, int new_C);
sp3d_tensor *sp3d_clone(const sp3d_tensor *t);
sp3d_tensor *sp3d_cat_feats(const sp3d_tensor *a, const sp3d_tensor *b);

/* ---- Elementwise ops on features (operate on raw float arrays) ---- */
void sp3d_linear(float *dst, const float *src, int N,
                  const qtensor *W, const qtensor *bias,
                  int out_C, int in_C, int n_threads);
void sp3d_layernorm(float *dst, const float *src,
                     const qtensor *w, const qtensor *b,
                     int N, int C, float eps);
void sp3d_gelu(float *x, int count);
void sp3d_silu(float *x, int count);

/* ---- Sparse 3D convolution (submanifold, stride=1) ---- */
void sp3d_conv3d_forward(float *dst, const sp3d_tensor *t,
                          const float *weight, const float *bias_f,
                          int in_C, int out_C, int kernel_size,
                          int n_threads);

/* ---- Variable-length sparse attention ---- */
void sp3d_attention(float *out, const float *qkv, const sp3d_tensor *t,
                     int n_heads, int head_dim, int n_threads);

/* ---- 3D RoPE for sparse coordinates ---- */
void sp3d_rope_3d(float *qk, const sp3d_tensor *t,
                   int n_heads, int head_dim, int dim_stride,
                   const float *rope_freqs, int n_freqs);

/* ---- Spatial operations ---- */
/* pool_mode: 0=mean, 1=max */
sp3d_tensor *sp3d_downsample(const sp3d_tensor *t, int factor, int pool_mode);
/* Upsample: nearest-neighbor into target_coords positions */
sp3d_tensor *sp3d_upsample(const sp3d_tensor *t, int factor,
                             const int32_t *target_coords, int target_N);

/* ---- Hash table (public for advanced use) ---- */
sp3d_hash *sp3d_hash_build(const int32_t *coords, int N);
void       sp3d_hash_free(sp3d_hash *h);
int        sp3d_hash_lookup(const sp3d_hash *h, int32_t batch,
                             int32_t z, int32_t y, int32_t x);
void       sp3d_ensure_hash(sp3d_tensor *t);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef SPARSE3D_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <float.h>

#ifndef CPU_COMPUTE_H
#define CPU_COMPUTE_IMPLEMENTATION
#include "cpu_compute.h"
#endif

/* ==================================================================== */
/* Spatial Hash Table                                                     */
/* ==================================================================== */

#define SP3D_HASH_EMPTY (~(uint64_t)0)

struct sp3d_hash {
    uint64_t *keys;
    int32_t  *vals;
    int       capacity;
    int       count;
};

static uint64_t sp3d_pack_coord(int32_t batch, int32_t z, int32_t y, int32_t x) {
    return ((uint64_t)(uint16_t)batch << 48) |
           ((uint64_t)(uint16_t)z << 32) |
           ((uint64_t)(uint16_t)y << 16) |
           (uint64_t)(uint16_t)x;
}

sp3d_hash *sp3d_hash_build(const int32_t *coords, int N) {
    /* Capacity: next power of 2 >= 2*N, minimum 16 */
    int cap = 16;
    while (cap < 2 * N) cap *= 2;

    sp3d_hash *h = (sp3d_hash *)malloc(sizeof(sp3d_hash));
    h->capacity = cap;
    h->count = N;
    h->keys = (uint64_t *)malloc((size_t)cap * sizeof(uint64_t));
    h->vals = (int32_t *)malloc((size_t)cap * sizeof(int32_t));
    for (int i = 0; i < cap; i++) {
        h->keys[i] = SP3D_HASH_EMPTY;
        h->vals[i] = -1;
    }

    uint64_t mask = (uint64_t)(cap - 1);
    for (int i = 0; i < N; i++) {
        uint64_t key = sp3d_pack_coord(coords[i*4], coords[i*4+1],
                                        coords[i*4+2], coords[i*4+3]);
        /* Fibonacci hash for slot */
        uint64_t slot = (key * 0x9E3779B97F4A7C15ULL) & mask;
        while (h->keys[slot] != SP3D_HASH_EMPTY) {
            slot = (slot + 1) & mask;
        }
        h->keys[slot] = key;
        h->vals[slot] = i;
    }
    return h;
}

void sp3d_hash_free(sp3d_hash *h) {
    if (!h) return;
    free(h->keys);
    free(h->vals);
    free(h);
}

int sp3d_hash_lookup(const sp3d_hash *h, int32_t batch,
                      int32_t z, int32_t y, int32_t x) {
    uint64_t key = sp3d_pack_coord(batch, z, y, x);
    uint64_t mask = (uint64_t)(h->capacity - 1);
    uint64_t slot = (key * 0x9E3779B97F4A7C15ULL) & mask;
    while (1) {
        if (h->keys[slot] == SP3D_HASH_EMPTY) return -1;
        if (h->keys[slot] == key) return h->vals[slot];
        slot = (slot + 1) & mask;
    }
}

void sp3d_ensure_hash(sp3d_tensor *t) {
    if (!t->hash) {
        t->hash = sp3d_hash_build(t->coords, t->N);
    }
}

/* ==================================================================== */
/* Lifecycle                                                              */
/* ==================================================================== */

sp3d_tensor *sp3d_create(const int32_t *coords, const float *feats,
                          int N, int C, int batch_size) {
    sp3d_tensor *t = (sp3d_tensor *)calloc(1, sizeof(sp3d_tensor));
    t->N = N;
    t->C = C;
    t->batch_size = batch_size;
    t->hash = NULL;

    /* Copy coords */
    t->coords = (int32_t *)malloc((size_t)N * 4 * sizeof(int32_t));
    memcpy(t->coords, coords, (size_t)N * 4 * sizeof(int32_t));

    /* Copy features */
    t->feats = (float *)malloc((size_t)N * C * sizeof(float));
    if (feats) {
        memcpy(t->feats, feats, (size_t)N * C * sizeof(float));
    } else {
        memset(t->feats, 0, (size_t)N * C * sizeof(float));
    }

    /* Compute batch_starts by scanning batch indices in coords */
    t->batch_starts = (int *)calloc((size_t)(batch_size + 1), sizeof(int));
    /* Count voxels per batch */
    int *counts = (int *)calloc((size_t)batch_size, sizeof(int));
    for (int i = 0; i < N; i++) {
        int b = coords[i * 4];
        if (b >= 0 && b < batch_size) counts[b]++;
    }
    /* Cumulative sum */
    t->batch_starts[0] = 0;
    for (int b = 0; b < batch_size; b++) {
        t->batch_starts[b + 1] = t->batch_starts[b] + counts[b];
    }
    free(counts);

    return t;
}

void sp3d_free(sp3d_tensor *t) {
    if (!t) return;
    free(t->coords);
    free(t->feats);
    free(t->batch_starts);
    sp3d_hash_free(t->hash);
    free(t);
}

sp3d_tensor *sp3d_replace_feats(const sp3d_tensor *t, const float *new_feats, int new_C) {
    sp3d_tensor *out = (sp3d_tensor *)calloc(1, sizeof(sp3d_tensor));
    out->N = t->N;
    out->C = new_C;
    out->batch_size = t->batch_size;
    out->hash = NULL;

    /* Share coords (copy) */
    out->coords = (int32_t *)malloc((size_t)t->N * 4 * sizeof(int32_t));
    memcpy(out->coords, t->coords, (size_t)t->N * 4 * sizeof(int32_t));

    /* Copy new features */
    out->feats = (float *)malloc((size_t)t->N * new_C * sizeof(float));
    memcpy(out->feats, new_feats, (size_t)t->N * new_C * sizeof(float));

    /* Copy batch_starts */
    out->batch_starts = (int *)malloc((size_t)(t->batch_size + 1) * sizeof(int));
    memcpy(out->batch_starts, t->batch_starts, (size_t)(t->batch_size + 1) * sizeof(int));

    return out;
}

sp3d_tensor *sp3d_clone(const sp3d_tensor *t) {
    return sp3d_replace_feats(t, t->feats, t->C);
}

sp3d_tensor *sp3d_cat_feats(const sp3d_tensor *a, const sp3d_tensor *b) {
    if (a->N != b->N) {
        fprintf(stderr, "sp3d_cat_feats: N mismatch (%d vs %d)\n", a->N, b->N);
        return NULL;
    }
    int new_C = a->C + b->C;
    float *cat = (float *)malloc((size_t)a->N * new_C * sizeof(float));
    for (int i = 0; i < a->N; i++) {
        memcpy(cat + i * new_C, a->feats + i * a->C, (size_t)a->C * sizeof(float));
        memcpy(cat + i * new_C + a->C, b->feats + i * b->C, (size_t)b->C * sizeof(float));
    }
    sp3d_tensor *out = sp3d_replace_feats(a, cat, new_C);
    free(cat);
    return out;
}

/* ==================================================================== */
/* Tensor helpers (dequant, GEMM)                                        */
/* ==================================================================== */

static void sp3d_dequant_row(const qtensor *t, int row, float *dst) {
    int n = t->n_cols;
    if (row < 0 || row >= t->n_rows) {
        memset(dst, 0, (size_t)n * sizeof(float));
        return;
    }
    if (t->type == GGML_TYPE_F16) {
        const uint16_t *src = (const uint16_t *)t->data + (size_t)row * n;
        for (int i = 0; i < n; i++) dst[i] = ggml_fp16_to_fp32(src[i]);
    } else if (t->type == GGML_TYPE_F32) {
        memcpy(dst, (const float *)t->data + (size_t)row * n, (size_t)n * sizeof(float));
    } else {
        int bs, ts;
        switch (t->type) {
            case GGML_TYPE_Q8_0: bs = 32; ts = 34; break;
            case GGML_TYPE_Q4_K: bs = 256; ts = 144; break;
            case GGML_TYPE_Q6_K: bs = 256; ts = 210; break;
            default: memset(dst, 0, (size_t)n * sizeof(float)); return;
        }
        size_t row_bytes = (size_t)((n + bs - 1) / bs) * ts;
        dequant_row(t->type, (const uint8_t *)t->data + row * row_bytes, dst, n);
    }
}

/* ==================================================================== */
/* SIMD helpers                                                           */
/* ==================================================================== */

/* AVX2 dot product: sum(a[0..n-1] * b[0..n-1]) */
static inline float sp3d_dot(const float *a, const float *b, int n) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int ic = 0;
    for (; ic + 15 < n; ic += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + ic),
                               _mm256_loadu_ps(b + ic), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + ic + 8),
                               _mm256_loadu_ps(b + ic + 8), acc1);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    for (; ic + 7 < n; ic += 8) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + ic),
                               _mm256_loadu_ps(b + ic), acc0);
    }
    float s = cpu_hsum_avx(acc0);
    for (; ic < n; ic++) s += a[ic] * b[ic];
    return s;
#else
    float s = 0.0f;
    for (int ic = 0; ic < n; ic++) s += a[ic] * b[ic];
    return s;
#endif
}

/* ==================================================================== */
/* Elementwise operations                                                */
/* ==================================================================== */

/* F32 GEMM kernel: dst[i_start..i_end][0..out_C] = src @ Wf^T + bf */
static void sp3d_gemm_f32_range(float *dst, const float *src,
                                 const float *Wf, const float *bf,
                                 int i_start, int i_end, int out_C, int in_C) {
    for (int i = i_start; i < i_end; i++) {
        const float *s_row = src + (size_t)i * in_C;
        float *d_row = dst + (size_t)i * out_C;
        int j = 0;
        for (; j + 3 < out_C; j += 4) {
            const float *w0 = Wf + (size_t)(j)     * in_C;
            const float *w1 = Wf + (size_t)(j + 1) * in_C;
            const float *w2 = Wf + (size_t)(j + 2) * in_C;
            const float *w3 = Wf + (size_t)(j + 3) * in_C;
#if defined(__AVX2__) && defined(__FMA__)
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
            int k = 0;
            for (; k + 7 < in_C; k += 8) {
                __m256 sv = _mm256_loadu_ps(s_row + k);
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w0 + k), sv, a0);
                a1 = _mm256_fmadd_ps(_mm256_loadu_ps(w1 + k), sv, a1);
                a2 = _mm256_fmadd_ps(_mm256_loadu_ps(w2 + k), sv, a2);
                a3 = _mm256_fmadd_ps(_mm256_loadu_ps(w3 + k), sv, a3);
            }
            float s0 = cpu_hsum_avx(a0), s1 = cpu_hsum_avx(a1);
            float s2 = cpu_hsum_avx(a2), s3 = cpu_hsum_avx(a3);
            for (; k < in_C; k++) {
                float sv = s_row[k];
                s0 += w0[k] * sv; s1 += w1[k] * sv;
                s2 += w2[k] * sv; s3 += w3[k] * sv;
            }
#else
            float s0=0,s1=0,s2=0,s3=0;
            for (int k = 0; k < in_C; k++) {
                float sv = s_row[k];
                s0 += w0[k]*sv; s1 += w1[k]*sv;
                s2 += w2[k]*sv; s3 += w3[k]*sv;
            }
#endif
            d_row[j]   = bf ? s0 + bf[j]   : s0;
            d_row[j+1] = bf ? s1 + bf[j+1] : s1;
            d_row[j+2] = bf ? s2 + bf[j+2] : s2;
            d_row[j+3] = bf ? s3 + bf[j+3] : s3;
        }
        for (; j < out_C; j++) {
            float s = sp3d_dot(Wf + (size_t)j * in_C, s_row, in_C);
            d_row[j] = bf ? s + bf[j] : s;
        }
    }
}

typedef struct {
    float *dst;
    const float *src;
    const float *Wf;
    const float *bf;
    int i_start, i_end, out_C, in_C;
} sp3d_gemm_task;

static void *sp3d_gemm_worker(void *arg) {
    sp3d_gemm_task *t = (sp3d_gemm_task *)arg;
    sp3d_gemm_f32_range(t->dst, t->src, t->Wf, t->bf,
                         t->i_start, t->i_end, t->out_C, t->in_C);
    return NULL;
}

void sp3d_linear(float *dst, const float *src, int N,
                  const qtensor *W, const qtensor *bias,
                  int out_C, int in_C, int n_threads) {
    if (!W->data) {
        memset(dst, 0, (size_t)N * out_C * sizeof(float));
        return;
    }
    if (W->type == GGML_TYPE_F16) {
        float *b = NULL;
        if (bias && bias->data) {
            b = (float *)malloc((size_t)out_C * sizeof(float));
            sp3d_dequant_row(bias, 0, b);
        }
        cpu_gemm_f16(dst, (const uint16_t *)W->data, b, src,
                     N, out_C, in_C, n_threads);
        free(b);
    } else if (W->type == GGML_TYPE_F32) {
        /* F32 GEMM: dst[i][j] = sum_k src[i][k] * W[j][k] + bias[j] */
        const float *Wf = (const float *)W->data;
        const float *bf = (bias && bias->data) ? (const float *)bias->data : NULL;
        if (n_threads <= 1 || N < n_threads * 4) {
            sp3d_gemm_f32_range(dst, src, Wf, bf, 0, N, out_C, in_C);
        } else {
            sp3d_gemm_task *tasks = (sp3d_gemm_task *)calloc((size_t)n_threads, sizeof(sp3d_gemm_task));
            pthread_t *threads = (pthread_t *)malloc((size_t)n_threads * sizeof(pthread_t));
            int per = N / n_threads, rem = N % n_threads, v = 0;
            for (int ti = 0; ti < n_threads; ti++) {
                int cnt = per + (ti < rem ? 1 : 0);
                tasks[ti] = (sp3d_gemm_task){dst, src, Wf, bf, v, v + cnt, out_C, in_C};
                v += cnt;
                pthread_create(&threads[ti], NULL, sp3d_gemm_worker, &tasks[ti]);
            }
            for (int ti = 0; ti < n_threads; ti++) pthread_join(threads[ti], NULL);
            free(tasks); free(threads);
        }
    } else {
        /* Quantized: dequant row by row */
        float *tmp = (float *)malloc((size_t)in_C * sizeof(float));
        for (int i = 0; i < N; i++) {
            for (int r = 0; r < out_C; r++) {
                sp3d_dequant_row(W, r, tmp);
                float s = 0.0f;
                for (int k = 0; k < in_C; k++) s += tmp[k] * src[i * in_C + k];
                dst[i * out_C + r] = s;
            }
        }
        free(tmp);
        if (bias && bias->data) {
            float *b = (float *)malloc((size_t)out_C * sizeof(float));
            sp3d_dequant_row(bias, 0, b);
            for (int i = 0; i < N; i++)
                for (int j = 0; j < out_C; j++)
                    dst[i * out_C + j] += b[j];
            free(b);
        }
    }
}

void sp3d_layernorm(float *dst, const float *src,
                     const qtensor *w, const qtensor *b,
                     int N, int C, float eps) {
    float *wf = (float *)malloc((size_t)C * sizeof(float));
    float *bf = (float *)malloc((size_t)C * sizeof(float));
    sp3d_dequant_row(w, 0, wf);
    sp3d_dequant_row(b, 0, bf);
#if defined(__AVX2__) && defined(__FMA__)
    __m256 v_eps = _mm256_set1_ps(eps);
    for (int t = 0; t < N; t++) {
        const float *x = src + (size_t)t * C;
        float *y = dst + (size_t)t * C;
        /* Pass 1: mean */
        __m256 sum = _mm256_setzero_ps();
        int i = 0;
        for (; i + 7 < C; i += 8)
            sum = _mm256_add_ps(sum, _mm256_loadu_ps(x + i));
        float mean = cpu_hsum_avx(sum);
        for (; i < C; i++) mean += x[i];
        mean /= C;
        /* Pass 2: variance */
        __m256 vmean = _mm256_set1_ps(mean);
        __m256 vvar = _mm256_setzero_ps();
        i = 0;
        for (; i + 7 < C; i += 8) {
            __m256 d = _mm256_sub_ps(_mm256_loadu_ps(x + i), vmean);
            vvar = _mm256_fmadd_ps(d, d, vvar);
        }
        float var = cpu_hsum_avx(vvar);
        for (; i < C; i++) { float d = x[i] - mean; var += d * d; }
        var /= C;
        /* Pass 3: normalize */
        float inv = 1.0f / sqrtf(var + eps);
        __m256 vinv = _mm256_set1_ps(inv);
        i = 0;
        for (; i + 7 < C; i += 8) {
            __m256 xv = _mm256_loadu_ps(x + i);
            __m256 norm = _mm256_mul_ps(_mm256_sub_ps(xv, vmean), vinv);
            __m256 out = _mm256_fmadd_ps(norm, _mm256_loadu_ps(wf + i),
                                          _mm256_loadu_ps(bf + i));
            _mm256_storeu_ps(y + i, out);
        }
        for (; i < C; i++)
            y[i] = (x[i] - mean) * inv * wf[i] + bf[i];
    }
    (void)v_eps;
#else
    cpu_layernorm(dst, src, wf, bf, N, C, eps);
#endif
    free(wf); free(bf);
}

#if defined(__AVX2__) && defined(__FMA__)
/* Fast exp approximation for AVX2: max relative error ~1.5e-7 in [-88, 88]
 * Uses Cephes-style polynomial minimax on reduced range */
static inline __m256 sp3d_exp_avx(__m256 x) {
    /* Clamp to avoid overflow/underflow */
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.0f));
    x = _mm256_min_ps(x, _mm256_set1_ps(88.0f));
    /* exp(x) = 2^(x/ln2) = 2^n * 2^f where n=floor(x/ln2), f=frac */
    __m256 log2e = _mm256_set1_ps(1.44269504089f);
    __m256 t = _mm256_mul_ps(x, log2e);
    __m256 n = _mm256_floor_ps(t);
    __m256 f = _mm256_sub_ps(t, n);  /* fractional part in [0,1) */
    /* 2^f approx via polynomial: p(f) ≈ 2^f for f in [0,1) */
    __m256 p = _mm256_set1_ps(1.3534167e-2f);
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(5.2011464e-2f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(2.4114209e-1f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(6.9315836e-1f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(9.9999994e-1f));
    /* Multiply by 2^n: add n to float exponent */
    __m256i ni = _mm256_cvtps_epi32(n);
    ni = _mm256_slli_epi32(ni, 23);
    __m256 pow2n = _mm256_castsi256_ps(_mm256_add_epi32(ni,
                    _mm256_set1_epi32(0x3f800000)));
    return _mm256_mul_ps(p, pow2n);
}

/* Fast tanh approximation: tanh(x) = 1 - 2/(exp(2x)+1) */
static inline __m256 sp3d_tanh_avx(__m256 x) {
    __m256 two = _mm256_set1_ps(2.0f);
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 e2x = sp3d_exp_avx(_mm256_mul_ps(two, x));
    return _mm256_sub_ps(one, _mm256_div_ps(two, _mm256_add_ps(e2x, one)));
}
#endif

void sp3d_gelu(float *x, int count) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 c = _mm256_set1_ps(0.7978845608f);
    __m256 c2 = _mm256_set1_ps(0.044715f);
    int i = 0;
    for (; i + 7 < count; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        __m256 v3 = _mm256_mul_ps(_mm256_mul_ps(v, v), v);
        __m256 inner = _mm256_mul_ps(c, _mm256_fmadd_ps(c2, v3, v));
        __m256 t = sp3d_tanh_avx(inner);
        _mm256_storeu_ps(x + i, _mm256_mul_ps(half, _mm256_mul_ps(v, _mm256_add_ps(one, t))));
    }
    for (; i < count; i++) {
        float v = x[i];
        float t = tanhf(0.7978845608f * (v + 0.044715f * v * v * v));
        x[i] = 0.5f * v * (1.0f + t);
    }
#else
    cpu_gelu(x, count);
#endif
}

void sp3d_silu(float *x, int count) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256 one = _mm256_set1_ps(1.0f);
    int i = 0;
    for (; i + 7 < count; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        __m256 neg_v = _mm256_sub_ps(_mm256_setzero_ps(), v);
        __m256 sigmoid = _mm256_div_ps(one, _mm256_add_ps(one, sp3d_exp_avx(neg_v)));
        _mm256_storeu_ps(x + i, _mm256_mul_ps(v, sigmoid));
    }
    for (; i < count; i++) {
        float v = x[i];
        x[i] = v / (1.0f + expf(-v));
    }
#else
    cpu_silu(x, count);
#endif
}

/* ==================================================================== */
/* Sparse 3D Convolution (submanifold, stride=1)                         */
/* ==================================================================== */

/* 3x3x3 kernel offsets: 27 neighbors */
static const int sp3d_conv_offsets_3x3x3[27][3] = {
    {-1,-1,-1},{-1,-1,0},{-1,-1,1},{-1,0,-1},{-1,0,0},{-1,0,1},{-1,1,-1},{-1,1,0},{-1,1,1},
    { 0,-1,-1},{ 0,-1,0},{ 0,-1,1},{ 0,0,-1},{ 0,0,0},{ 0,0,1},{ 0,1,-1},{ 0,1,0},{ 0,1,1},
    { 1,-1,-1},{ 1,-1,0},{ 1,-1,1},{ 1,0,-1},{ 1,0,0},{ 1,0,1},{ 1,1,-1},{ 1,1,0},{ 1,1,1},
};

typedef struct {
    float *dst;
    const sp3d_tensor *t;
    const float *weight;
    int in_C, out_C, kernel_size, n_offsets;
    int v_start, v_end;
} sp3d_conv_task;

static void *sp3d_conv_worker(void *arg) {
    sp3d_conv_task *task = (sp3d_conv_task *)arg;
    const sp3d_tensor *t = task->t;
    const float *W = task->weight;
    int in_C = task->in_C, out_C = task->out_C;
    int K3 = task->n_offsets;

    for (int i = task->v_start; i < task->v_end; i++) {
        int32_t batch = t->coords[i*4];
        int32_t z = t->coords[i*4+1];
        int32_t y = t->coords[i*4+2];
        int32_t x = t->coords[i*4+3];
        float *out = task->dst + (size_t)i * out_C;

        for (int k = 0; k < K3; k++) {
            int32_t nz = z + sp3d_conv_offsets_3x3x3[k][0];
            int32_t ny = y + sp3d_conv_offsets_3x3x3[k][1];
            int32_t nx = x + sp3d_conv_offsets_3x3x3[k][2];

            int j = sp3d_hash_lookup(t->hash, batch, nz, ny, nx);
            if (j < 0) continue;

            const float *feat_j = t->feats + (size_t)j * in_C;
            /* W layout: [out_C, K3, in_C] */
            const float *w_k_base = W + (size_t)k * in_C;
            size_t w_stride = (size_t)K3 * in_C; /* stride between oc rows */

            /* Process 4 output channels at a time to hide FMA latency */
            int oc = 0;
            for (; oc + 3 < out_C; oc += 4) {
                const float *w0 = w_k_base + (size_t)(oc)     * w_stride;
                const float *w1 = w_k_base + (size_t)(oc + 1) * w_stride;
                const float *w2 = w_k_base + (size_t)(oc + 2) * w_stride;
                const float *w3 = w_k_base + (size_t)(oc + 3) * w_stride;
#if defined(__AVX2__) && defined(__FMA__)
                __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
                __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
                int ic = 0;
                for (; ic + 7 < in_C; ic += 8) {
                    __m256 f = _mm256_loadu_ps(feat_j + ic);
                    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w0 + ic), f, a0);
                    a1 = _mm256_fmadd_ps(_mm256_loadu_ps(w1 + ic), f, a1);
                    a2 = _mm256_fmadd_ps(_mm256_loadu_ps(w2 + ic), f, a2);
                    a3 = _mm256_fmadd_ps(_mm256_loadu_ps(w3 + ic), f, a3);
                }
                float s0 = cpu_hsum_avx(a0), s1 = cpu_hsum_avx(a1);
                float s2 = cpu_hsum_avx(a2), s3 = cpu_hsum_avx(a3);
                for (; ic < in_C; ic++) {
                    float fv = feat_j[ic];
                    s0 += w0[ic] * fv; s1 += w1[ic] * fv;
                    s2 += w2[ic] * fv; s3 += w3[ic] * fv;
                }
                out[oc]   += s0; out[oc+1] += s1;
                out[oc+2] += s2; out[oc+3] += s3;
#else
                float s0=0, s1=0, s2=0, s3=0;
                for (int ic = 0; ic < in_C; ic++) {
                    float fv = feat_j[ic];
                    s0 += w0[ic] * fv; s1 += w1[ic] * fv;
                    s2 += w2[ic] * fv; s3 += w3[ic] * fv;
                }
                out[oc]   += s0; out[oc+1] += s1;
                out[oc+2] += s2; out[oc+3] += s3;
#endif
            }
            for (; oc < out_C; oc++) {
                const float *w_ock = w_k_base + (size_t)oc * w_stride;
                out[oc] += sp3d_dot(w_ock, feat_j, in_C);
            }
        }
    }
    return NULL;
}

void sp3d_conv3d_forward(float *dst, const sp3d_tensor *t,
                          const float *weight, const float *bias_f,
                          int in_C, int out_C, int kernel_size,
                          int n_threads) {
    if (kernel_size != 3) {
        fprintf(stderr, "sp3d_conv3d: only kernel_size=3 supported, got %d\n", kernel_size);
        return;
    }
    int K3 = 27; /* 3*3*3 */

    /* Ensure hash table is built */
    sp3d_ensure_hash((sp3d_tensor *)t);

    /* Zero output */
    memset(dst, 0, (size_t)t->N * out_C * sizeof(float));

    /* Initialize with bias */
    if (bias_f) {
        for (int i = 0; i < t->N; i++)
            for (int oc = 0; oc < out_C; oc++)
                dst[i * out_C + oc] = bias_f[oc];
    }

    if (n_threads <= 1 || t->N < n_threads * 4) {
        sp3d_conv_task task = {dst, t, weight, in_C, out_C, kernel_size, K3, 0, t->N};
        sp3d_conv_worker(&task);
    } else {
        sp3d_conv_task *tasks = (sp3d_conv_task *)calloc((size_t)n_threads, sizeof(sp3d_conv_task));
        pthread_t *threads = (pthread_t *)malloc((size_t)n_threads * sizeof(pthread_t));
        int per = t->N / n_threads;
        int rem = t->N % n_threads;
        int v = 0;
        for (int i = 0; i < n_threads; i++) {
            int count = per + (i < rem ? 1 : 0);
            tasks[i] = (sp3d_conv_task){dst, t, weight, in_C, out_C, kernel_size, K3, v, v + count};
            v += count;
            pthread_create(&threads[i], NULL, sp3d_conv_worker, &tasks[i]);
        }
        for (int i = 0; i < n_threads; i++) pthread_join(threads[i], NULL);
        free(tasks); free(threads);
    }
}

/* ==================================================================== */
/* Variable-length sparse attention                                      */
/* ==================================================================== */

void sp3d_attention(float *out, const float *qkv, const sp3d_tensor *t,
                     int n_heads, int head_dim, int n_threads) {
    int dim = n_heads * head_dim;

    /* Process each batch element independently.
     * CUDA extension point: one CTA per (batch, head). */
    for (int b = 0; b < t->batch_size; b++) {
        int start = t->batch_starts[b];
        int end = t->batch_starts[b + 1];
        int seq_len = end - start;
        if (seq_len <= 0) continue;

        /* Delegate to cpu_attention for this batch's subsequence.
         * qkv layout: [N, 3*dim], batch b spans rows [start, end). */
        const float *batch_qkv = qkv + (size_t)start * 3 * dim;
        float *batch_out = out + (size_t)start * dim;
        cpu_attention(batch_out, batch_qkv, seq_len, dim, n_heads, head_dim, n_threads);
    }
}

/* ==================================================================== */
/* 3D RoPE for sparse coordinates                                        */
/* ==================================================================== */

void sp3d_rope_3d(float *qk, const sp3d_tensor *t,
                   int n_heads, int head_dim, int dim_stride,
                   const float *rope_freqs, int n_freqs) {
    int axis_dim = 2 * n_freqs;
    if (3 * axis_dim > head_dim) {
        axis_dim = head_dim / 3;
        n_freqs = axis_dim / 2;
    }

    /* Precompute sin/cos tables for all 3 axes × n_freqs.
     * These are reused across all heads for a given voxel. */
    float *cs_tab = (float *)malloc((size_t)3 * n_freqs * sizeof(float));
    float *sn_tab = (float *)malloc((size_t)3 * n_freqs * sizeof(float));

    for (int i = 0; i < t->N; i++) {
        float coord_vals[3] = {
            (float)t->coords[i*4+1],
            (float)t->coords[i*4+2],
            (float)t->coords[i*4+3]
        };

        /* Precompute sin/cos for this voxel (3 axes × n_freqs) */
        for (int axis = 0; axis < 3; axis++) {
            float coord = coord_vals[axis];
            for (int j = 0; j < n_freqs; j++) {
                float theta = coord * rope_freqs[j];
                cs_tab[axis * n_freqs + j] = cosf(theta);
                sn_tab[axis * n_freqs + j] = sinf(theta);
            }
        }

        /* Apply rotation to all heads using precomputed tables */
        for (int h = 0; h < n_heads; h++) {
            float *v = qk + (size_t)i * dim_stride + h * head_dim;

            for (int axis = 0; axis < 3; axis++) {
                int base = axis * axis_dim;
                const float *cs_a = cs_tab + axis * n_freqs;
                const float *sn_a = sn_tab + axis * n_freqs;
                int j = 0;
#if defined(__AVX2__) && defined(__FMA__)
                for (; j + 7 < n_freqs; j += 8) {
                    int idx0 = base + j;
                    int idx1 = base + j + n_freqs;
                    if (idx1 + 7 >= head_dim) break;
                    __m256 v0 = _mm256_loadu_ps(v + idx0);
                    __m256 v1 = _mm256_loadu_ps(v + idx1);
                    __m256 c  = _mm256_loadu_ps(cs_a + j);
                    __m256 s  = _mm256_loadu_ps(sn_a + j);
                    /* v0' = v0*cos - v1*sin, v1' = v0*sin + v1*cos */
                    __m256 new0 = _mm256_fmsub_ps(v0, c, _mm256_mul_ps(v1, s));
                    __m256 new1 = _mm256_fmadd_ps(v0, s, _mm256_mul_ps(v1, c));
                    _mm256_storeu_ps(v + idx0, new0);
                    _mm256_storeu_ps(v + idx1, new1);
                }
#endif
                for (; j < n_freqs; j++) {
                    int idx0 = base + j;
                    int idx1 = base + j + n_freqs;
                    if (idx1 >= head_dim) break;
                    float v0 = v[idx0], v1 = v[idx1];
                    v[idx0] = v0 * cs_a[j] - v1 * sn_a[j];
                    v[idx1] = v0 * sn_a[j] + v1 * cs_a[j];
                }
            }
        }
    }

    free(cs_tab);
    free(sn_tab);
}

/* ==================================================================== */
/* Spatial operations: downsample / upsample                             */
/* ==================================================================== */

sp3d_tensor *sp3d_downsample(const sp3d_tensor *t, int factor, int pool_mode) {
    /* Compute downsampled coords: floor(coord / factor) */
    int32_t *new_coords = (int32_t *)malloc((size_t)t->N * 4 * sizeof(int32_t));
    for (int i = 0; i < t->N; i++) {
        new_coords[i*4]   = t->coords[i*4];           /* batch unchanged */
        new_coords[i*4+1] = t->coords[i*4+1] / factor;
        new_coords[i*4+2] = t->coords[i*4+2] / factor;
        new_coords[i*4+3] = t->coords[i*4+3] / factor;
    }

    /* Hash to map unique downsampled coords to output indices */
    int out_N = 0;
    int *mapping = (int *)malloc((size_t)t->N * sizeof(int));
    int32_t *unique_coords = (int32_t *)malloc((size_t)t->N * 4 * sizeof(int32_t));

    sp3d_hash *h = (sp3d_hash *)malloc(sizeof(sp3d_hash));
    int cap = 16;
    while (cap < 2 * t->N) cap *= 2;
    h->capacity = cap;
    h->count = 0;
    h->keys = (uint64_t *)malloc((size_t)cap * sizeof(uint64_t));
    h->vals = (int32_t *)malloc((size_t)cap * sizeof(int32_t));
    for (int i = 0; i < cap; i++) { h->keys[i] = SP3D_HASH_EMPTY; h->vals[i] = -1; }

    uint64_t hmask = (uint64_t)(cap - 1);
    for (int i = 0; i < t->N; i++) {
        uint64_t key = sp3d_pack_coord(new_coords[i*4], new_coords[i*4+1],
                                        new_coords[i*4+2], new_coords[i*4+3]);
        uint64_t slot = (key * 0x9E3779B97F4A7C15ULL) & hmask;
        while (h->keys[slot] != SP3D_HASH_EMPTY && h->keys[slot] != key)
            slot = (slot + 1) & hmask;

        if (h->keys[slot] == SP3D_HASH_EMPTY) {
            /* New unique coord */
            h->keys[slot] = key;
            h->vals[slot] = out_N;
            memcpy(unique_coords + out_N * 4, new_coords + i * 4, 4 * sizeof(int32_t));
            mapping[i] = out_N;
            out_N++;
        } else {
            /* Existing coord */
            mapping[i] = h->vals[slot];
        }
    }
    sp3d_hash_free(h);

    /* Aggregate features */
    int C = t->C;
    float *out_feats = (float *)calloc((size_t)out_N * C, sizeof(float));
    int *counts = (int *)calloc((size_t)out_N, sizeof(int));

    if (pool_mode == 0) {
        /* Mean pooling */
        for (int i = 0; i < t->N; i++) {
            int oi = mapping[i];
            for (int c = 0; c < C; c++)
                out_feats[oi * C + c] += t->feats[i * C + c];
            counts[oi]++;
        }
        for (int i = 0; i < out_N; i++) {
            if (counts[i] > 1) {
                float inv = 1.0f / (float)counts[i];
                for (int c = 0; c < C; c++)
                    out_feats[i * C + c] *= inv;
            }
        }
    } else {
        /* Max pooling */
        for (int i = 0; i < out_N * C; i++) out_feats[i] = -FLT_MAX;
        for (int i = 0; i < t->N; i++) {
            int oi = mapping[i];
            for (int c = 0; c < C; c++) {
                float v = t->feats[i * C + c];
                if (v > out_feats[oi * C + c])
                    out_feats[oi * C + c] = v;
            }
        }
    }

    sp3d_tensor *result = sp3d_create(unique_coords, out_feats, out_N, C, t->batch_size);

    free(new_coords); free(unique_coords); free(mapping);
    free(out_feats); free(counts);
    return result;
}

sp3d_tensor *sp3d_upsample(const sp3d_tensor *t, int factor,
                             const int32_t *target_coords, int target_N) {
    /* Nearest-neighbor upsample: for each target coord, find source via floor(coord/factor) */
    sp3d_ensure_hash((sp3d_tensor *)t);

    int C = t->C;
    float *out_feats = (float *)calloc((size_t)target_N * C, sizeof(float));

    for (int i = 0; i < target_N; i++) {
        int32_t b = target_coords[i*4];
        int32_t sz = target_coords[i*4+1] / factor;
        int32_t sy = target_coords[i*4+2] / factor;
        int32_t sx = target_coords[i*4+3] / factor;

        int j = sp3d_hash_lookup(t->hash, b, sz, sy, sx);
        if (j >= 0) {
            memcpy(out_feats + i * C, t->feats + j * C, (size_t)C * sizeof(float));
        }
    }

    sp3d_tensor *result = sp3d_create(target_coords, out_feats, target_N, C, t->batch_size);
    free(out_feats);
    return result;
}

#endif /* SPARSE3D_IMPLEMENTATION */
#endif /* SPARSE3D_H */
