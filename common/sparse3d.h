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
/* Elementwise operations                                                */
/* ==================================================================== */

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
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < out_C; j++) {
                float s = 0.0f;
                const float *w_row = Wf + (size_t)j * in_C;
                const float *s_row = src + (size_t)i * in_C;
                for (int k = 0; k < in_C; k++) s += w_row[k] * s_row[k];
                dst[i * out_C + j] = s;
            }
        }
        if (bias && bias->data) {
            const float *bf = (const float *)bias->data;
            for (int i = 0; i < N; i++)
                for (int j = 0; j < out_C; j++)
                    dst[i * out_C + j] += bf[j];
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
    cpu_layernorm(dst, src, wf, bf, N, C, eps);
    free(wf); free(bf);
}

void sp3d_gelu(float *x, int count) {
    cpu_gelu(x, count);
}

void sp3d_silu(float *x, int count) {
    cpu_silu(x, count);
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

    /* CUDA extension point: launch one thread per output voxel */
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

            /* Accumulate: out[oc] += sum_ic W[oc, k, ic] * feats[j, ic]
             * W layout: [out_C, K3, in_C] */
            const float *feat_j = t->feats + (size_t)j * in_C;
            for (int oc = 0; oc < out_C; oc++) {
                const float *w_ock = W + ((size_t)oc * K3 + k) * in_C;
                float s = 0.0f;
                for (int ic = 0; ic < in_C; ic++) {
                    s += w_ock[ic] * feat_j[ic];
                }
                out[oc] += s;
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
    /* Split head_dim into 3 axes: z, y, x.
     * TRELLIS.2 convention: head_dim is split into 3 equal groups of pairs.
     * n_freqs = head_dim / 6 (number of rotation pairs per axis).
     * Layout within head: [z_pairs, y_pairs, x_pairs] each of size n_freqs*2.
     * Uses rotate_half: out = x * cos + rotate_half(x) * sin
     *
     * rope_freqs[j] = 1.0 / (freq_base ^ (2*j / (2*n_freqs)))
     *   typically 3 * n_freqs * 2 = head_dim
     */
    int axis_dim = 2 * n_freqs;  /* dims per axis */
    if (3 * axis_dim > head_dim) {
        /* Fallback: use as many dims as fit */
        axis_dim = head_dim / 3;
        n_freqs = axis_dim / 2;
    }

    for (int i = 0; i < t->N; i++) {
        float cz = (float)t->coords[i*4+1];
        float cy = (float)t->coords[i*4+2];
        float cx = (float)t->coords[i*4+3];
        float coords[3] = {cz, cy, cx};

        for (int h = 0; h < n_heads; h++) {
            float *v = qk + (size_t)i * dim_stride + h * head_dim;

            for (int axis = 0; axis < 3; axis++) {
                float coord = coords[axis];
                int base = axis * axis_dim;

                for (int j = 0; j < n_freqs; j++) {
                    float theta = coord * rope_freqs[j];
                    float cs = cosf(theta);
                    float sn = sinf(theta);

                    /* rotate_half: pairs are (v[j], v[j+n_freqs]) */
                    int idx0 = base + j;
                    int idx1 = base + j + n_freqs;
                    if (idx1 >= head_dim) break;

                    float v0 = v[idx0];
                    float v1 = v[idx1];
                    v[idx0] = v0 * cs - v1 * sn;
                    v[idx1] = v0 * sn + v1 * cs;
                }
            }
        }
    }
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
