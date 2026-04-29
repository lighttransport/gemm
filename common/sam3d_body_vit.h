/*
 * sam3d_body_vit.h — vanilla ViT-H/16 backbone for sam-3d-body
 * (vit_hmr_512_384 variant). Distinct enough from common/dinov3.h
 * that retrofitting was not worth the complexity:
 *
 *   - Patch_embed conv has padding=2 (NOT 0)
 *   - Absolute LEARNED pos_embed (NOT RoPE)
 *   - No register/storage tokens, no CLS injected for output
 *     (pos_embed slot 0 is added uniformly to every patch as a bias)
 *   - GELU MLP (NOT SwiGLU)
 *   - No QK norm, no LayerScale
 *   - Output: (B, dim, Hp, Wp) feature map (no token prefix)
 *
 * Architecture (from sam_3d_body/models/backbones/vit.py::vit512_384):
 *   img_size=(512, 384), patch=16, embed_dim=1280, depth=32,
 *   num_heads=16  (head_dim=80), mlp_ratio=4 (ffn=5120), pre-LN.
 *
 * Usage:
 *   #define SAM3D_BODY_VIT_IMPLEMENTATION
 *   #include "sam3d_body_vit.h"
 *
 * Dependencies: ggml_dequant.h, qtensor_utils.h, safetensors.h,
 *               cpu_compute.h.
 */
#ifndef SAM3D_BODY_VIT_H
#define SAM3D_BODY_VIT_H

#include <stdint.h>
#include <stddef.h>
#include "ggml_dequant.h"
#include "qtensor_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    qtensor ln1_w, ln1_b;
    qtensor attn_qkv_w, attn_qkv_b;
    qtensor attn_out_w, attn_out_b;
    qtensor ln2_w, ln2_b;
    qtensor ffn_fc1_w, ffn_fc1_b;
    qtensor ffn_fc2_w, ffn_fc2_b;
} sam3d_body_vit_block;

typedef struct {
    int n_blocks, dim, n_heads, head_dim, ffn_hidden;
    int patch_size;             /* 16 */
    int patch_pad;              /* 2 — kernel/stride differ from padding */
    int img_h, img_w;           /* 512, 384 */
    int grid_h, grid_w;         /* 32, 24 */
    int n_patches;              /* 768 = 32*24 */
    float ln_eps;
    float image_mean[3], image_std[3];

    qtensor patch_embed_w, patch_embed_b;
    qtensor pos_embed;          /* (1, n_patches+1, dim) — slot 0 is added uniformly */
    qtensor norm_w, norm_b;     /* last_norm */
    sam3d_body_vit_block *blocks;

    void *st_ctx;
} sam3d_body_vit_model;

typedef struct {
    /* Channels-last token grid: (n_patches, dim) row-major. The caller
     * is responsible for reshaping to (1, dim, grid_h, grid_w) when
     * comparing against the upstream feature map (which is permuted
     * back to channels-first inside forward_features). */
    float *tokens;
    int n_patches;
    int dim;
    int grid_h, grid_w;
} sam3d_body_vit_result;

sam3d_body_vit_model *sam3d_body_vit_load_safetensors(const char *st_path);
void                  sam3d_body_vit_free(sam3d_body_vit_model *m);

/* `chw` is (3, img_h, img_w) f32, ImageNet-normalized AND already
 * W-axis cropped from 512×512 to 512×384 (the upstream
 * data_preprocess slice `[:, :, :, 64:-64]`). */
sam3d_body_vit_result sam3d_body_vit_encode_from_normalized(
        sam3d_body_vit_model *m, const float *chw,
        int img_w, int img_h, int n_threads);
void sam3d_body_vit_result_free(sam3d_body_vit_result *r);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef SAM3D_BODY_VIT_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <float.h>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

#define CPU_COMPUTE_IMPLEMENTATION
#include "cpu_compute.h"

static double sam3d_body_vit_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ---- Local helpers (mirrors of dinov3's batched ops, kept private) ---- */

static void s3dvit_batch_gemm(float *dst, const qtensor *W, const qtensor *bias,
                              const float *src, int n_tok, int n_out, int n_in,
                              int n_threads) {
    if (!W->data) {
        memset(dst, 0, (size_t)n_tok * n_out * sizeof(float));
        return;
    }
    float *bf_alloc = NULL;
    const float *bf = NULL;
    if (bias && bias->data) {
        if (bias->type == GGML_TYPE_F32) {
            bf = (const float *)bias->data;
        } else {
            bf_alloc = (float *)malloc((size_t)n_out * sizeof(float));
            qt_dequant_row(bias, 0, bf_alloc);
            bf = bf_alloc;
        }
    }
    if (W->type == GGML_TYPE_F16) {
        cpu_gemm_f16(dst, (const uint16_t *)W->data, (float *)bf, src,
                     n_tok, n_out, n_in, n_threads);
    } else if (W->type == GGML_TYPE_F32) {
        cpu_gemm_f32(dst, (const float *)W->data, bf, src,
                     n_tok, n_out, n_in, n_threads);
    } else {
        float *Wf = qt_dequant(W);
        cpu_gemm_f32(dst, Wf, bf, src, n_tok, n_out, n_in, n_threads);
        free(Wf);
    }
    free(bf_alloc);
}

static void s3dvit_layernorm_batch(float *dst, const float *src,
                                   const qtensor *w, const qtensor *b,
                                   int n_tok, int dim, float eps) {
    float *wf = (float *)malloc((size_t)dim * sizeof(float));
    float *bf = (float *)malloc((size_t)dim * sizeof(float));
    qt_dequant_row(w, 0, wf);
    qt_dequant_row(b, 0, bf);
    cpu_layernorm(dst, src, wf, bf, n_tok, dim, eps);
    free(wf); free(bf);
}

/* ---- Generic-head_dim self-attention ----
 *
 * cpu_compute.h's cpu_attn_worker hard-codes head_dim=64 in its AVX2
 * register allocation. ViT-H has head_dim=80 (1280/16), so we ship a
 * dim-generic variant inline here. AVX2 8-lane vector loops; tile size
 * 64 along K to amortize Q reloads. Online softmax is a straight port
 * of the dinov3 path. */

#define S3DVIT_ATTN_TILE 64

typedef struct {
    const float *qkv;
    float       *attn_out;
    int n_tok, dim, head_dim, n_heads;
    int h_start, h_end;
    float scale;
} s3dvit_attn_task;

#if defined(__AVX2__) && defined(__FMA__)
static inline float s3dvit_hsum256(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}
#endif

static void *s3dvit_attn_worker(void *arg) {
    s3dvit_attn_task *t = (s3dvit_attn_task *)arg;
    int N = t->n_tok, hd = t->head_dim, dim3 = 3 * t->dim;
    float scale = t->scale;

    float *K_buf = (float *)malloc((size_t)N * (size_t)hd * sizeof(float));
    float *V_buf = (float *)malloc((size_t)N * (size_t)hd * sizeof(float));
    float *acc   = (float *)malloc((size_t)hd * sizeof(float));
    float scores[S3DVIT_ATTN_TILE];

    for (int h = t->h_start; h < t->h_end; h++) {
        for (int ki = 0; ki < N; ki++) {
            memcpy(K_buf + (size_t)ki * hd,
                   t->qkv + (size_t)ki * dim3 + t->dim + h * hd,
                   (size_t)hd * sizeof(float));
            memcpy(V_buf + (size_t)ki * hd,
                   t->qkv + (size_t)ki * dim3 + 2 * t->dim + h * hd,
                   (size_t)hd * sizeof(float));
        }

        for (int qi = 0; qi < N; qi++) {
            const float *q_h = t->qkv + (size_t)qi * dim3 + h * hd;
            memset(acc, 0, (size_t)hd * sizeof(float));
            float running_max = -FLT_MAX;
            float running_sum = 0.0f;

            for (int ki_base = 0; ki_base < N; ki_base += S3DVIT_ATTN_TILE) {
                int tile_end = ki_base + S3DVIT_ATTN_TILE;
                if (tile_end > N) tile_end = N;
                int tile_len = tile_end - ki_base;

                /* QKᵀ for this tile */
                float tile_max = -FLT_MAX;
                for (int j = 0; j < tile_len; j++) {
                    const float *k_j = K_buf + (size_t)(ki_base + j) * hd;
                    float dot = 0.0f;
#if defined(__AVX2__) && defined(__FMA__)
                    int d = 0;
                    __m256 vsum = _mm256_setzero_ps();
                    for (; d + 7 < hd; d += 8) {
                        vsum = _mm256_fmadd_ps(_mm256_loadu_ps(q_h + d),
                                               _mm256_loadu_ps(k_j + d), vsum);
                    }
                    dot = s3dvit_hsum256(vsum);
                    for (; d < hd; d++) dot += q_h[d] * k_j[d];
#else
                    for (int d = 0; d < hd; d++) dot += q_h[d] * k_j[d];
#endif
                    float s = dot * scale;
                    scores[j] = s;
                    if (s > tile_max) tile_max = s;
                }

                /* Online softmax: rescale running accumulators */
                float new_max = running_max > tile_max ? running_max : tile_max;
                float correction = expf(running_max - new_max);
                running_sum *= correction;
#if defined(__AVX2__) && defined(__FMA__)
                {
                    __m256 vc = _mm256_set1_ps(correction);
                    int d = 0;
                    for (; d + 7 < hd; d += 8) {
                        _mm256_storeu_ps(acc + d,
                            _mm256_mul_ps(_mm256_loadu_ps(acc + d), vc));
                    }
                    for (; d < hd; d++) acc[d] *= correction;
                }
#else
                for (int d = 0; d < hd; d++) acc[d] *= correction;
#endif

                /* Accumulate weighted V */
                for (int j = 0; j < tile_len; j++) {
                    float w = expf(scores[j] - new_max);
                    running_sum += w;
                    const float *v_j = V_buf + (size_t)(ki_base + j) * hd;
#if defined(__AVX2__) && defined(__FMA__)
                    {
                        __m256 vw = _mm256_set1_ps(w);
                        int d = 0;
                        for (; d + 7 < hd; d += 8) {
                            __m256 va = _mm256_loadu_ps(acc + d);
                            __m256 vv = _mm256_loadu_ps(v_j + d);
                            _mm256_storeu_ps(acc + d,
                                _mm256_fmadd_ps(vw, vv, va));
                        }
                        for (; d < hd; d++) acc[d] += w * v_j[d];
                    }
#else
                    for (int d = 0; d < hd; d++) acc[d] += w * v_j[d];
#endif
                }
                running_max = new_max;
            }

            float inv_sum = 1.0f / running_sum;
            float *out_h = t->attn_out + (size_t)qi * t->dim + h * hd;
            for (int d = 0; d < hd; d++) out_h[d] = acc[d] * inv_sum;
        }
    }
    free(K_buf); free(V_buf); free(acc);
    return NULL;
}

static void s3dvit_attention(float *out, const float *qkv, int n_tok, int dim,
                             int n_heads, int head_dim, int n_threads) {
    float scale = 1.0f / sqrtf((float)head_dim);
    if (n_threads <= 1) {
        s3dvit_attn_task task = {qkv, out, n_tok, dim, head_dim, n_heads,
                                 0, n_heads, scale};
        s3dvit_attn_worker(&task);
        return;
    }
    int nt = n_threads < n_heads ? n_threads : n_heads;
#if defined(_OPENMP)
    int heads_per = n_heads / nt;
    int extra = n_heads % nt;
    #pragma omp parallel for schedule(static) num_threads(nt)
    for (int i = 0; i < nt; i++) {
        int h_start = i * heads_per + (i < extra ? i : extra);
        int count   = heads_per + (i < extra ? 1 : 0);
        s3dvit_attn_task task = {qkv, out, n_tok, dim, head_dim, n_heads,
                                 h_start, h_start + count, scale};
        s3dvit_attn_worker(&task);
    }
#else
    pthread_t *threads = (pthread_t *)malloc((size_t)nt * sizeof(pthread_t));
    s3dvit_attn_task *tasks = (s3dvit_attn_task *)calloc((size_t)nt, sizeof(s3dvit_attn_task));
    int heads_per = n_heads / nt;
    int extra = n_heads % nt;
    int h = 0;
    for (int i = 0; i < nt; i++) {
        int count = heads_per + (i < extra ? 1 : 0);
        tasks[i] = (s3dvit_attn_task){qkv, out, n_tok, dim, head_dim, n_heads,
                                      h, h + count, scale};
        h += count;
        pthread_create(&threads[i], NULL, s3dvit_attn_worker, &tasks[i]);
    }
    for (int i = 0; i < nt; i++) pthread_join(threads[i], NULL);
    free(threads); free(tasks);
#endif
}

/* ==================================================================== */
/* SafeTensors loading                                                   */
/* ==================================================================== */

#ifdef SAFETENSORS_H

sam3d_body_vit_model *sam3d_body_vit_load_safetensors(const char *st_path) {
    st_context *st = safetensors_open(st_path);
    if (!st) return NULL;

    fprintf(stderr, "sam3d_body_vit: opened %s, %d tensors\n",
            st_path, st->n_tensors);

    /* Detect prefix by scanning tensor names — same trick as dinov3.h. */
    char prefix[256] = "";
    for (int i = 0; i < st->n_tensors; i++) {
        const char *nm = safetensors_name(st, i);
        const char *p = strstr(nm, "patch_embed");
        if (p) {
            size_t pl = (size_t)(p - nm);
            if (pl < sizeof(prefix)) {
                memcpy(prefix, nm, pl);
                prefix[pl] = '\0';
            }
            break;
        }
    }
    fprintf(stderr, "sam3d_body_vit: detected prefix: '%s'\n", prefix);

    /* Auto-detect parameters from first relevant tensor shapes. */
    int embed_dim = 1280, n_blocks = 0, ffn_hidden = 5120;
    int patch_size = 16, n_patches_plus_one = 0;

    for (int i = 0; i < st->n_tensors; i++) {
        const char *nm = safetensors_name(st, i);
        const uint64_t *sh = safetensors_shape(st, i);
        int nd = safetensors_ndims(st, i);

        if (strstr(nm, "patch_embed.proj.weight") && nd == 4) {
            embed_dim  = (int)sh[0];
            patch_size = (int)sh[2];
        }
        if (strstr(nm, "pos_embed") && nd == 3) {
            n_patches_plus_one = (int)sh[1];
        }
        if (strstr(nm, "blocks.0.mlp.fc1.weight") && nd == 2) {
            ffn_hidden = (int)sh[0];
        }
        const char *bp = strstr(nm, "blocks.");
        if (bp) {
            bp += 7;
            int blk = 0;
            while (*bp >= '0' && *bp <= '9') { blk = blk * 10 + (*bp - '0'); bp++; }
            if (blk + 1 > n_blocks) n_blocks = blk + 1;
        }
    }

    int n_heads  = 16;
    int head_dim = embed_dim / n_heads;

    /* The vit_hmr_512_384 factory hard-codes 32×24 patches (img 512×384,
     * patch 16, padding 2 → ((512+4-16)/16+1)×((384+4-16)/16+1)
     * = 32×24 = 768). pos_embed has +1 slot for the legacy CLS bias. */
    int img_h = 512, img_w = 384;
    int grid_h = 32, grid_w = 24;
    int n_patches = grid_h * grid_w;
    if (n_patches_plus_one > 0 && n_patches_plus_one - 1 != n_patches) {
        fprintf(stderr, "sam3d_body_vit: WARN pos_embed has %d slots, "
                        "expected %d; assuming default 32×24 layout\n",
                n_patches_plus_one, n_patches + 1);
    }

    fprintf(stderr,
            "sam3d_body_vit: dim=%d heads=%d hd=%d blocks=%d ffn=%d\n",
            embed_dim, n_heads, head_dim, n_blocks, ffn_hidden);
    fprintf(stderr,
            "sam3d_body_vit: img=%dx%d patch=%d pad=2 grid=%dx%d patches=%d\n",
            img_h, img_w, patch_size, grid_h, grid_w, n_patches);

    sam3d_body_vit_model *m = (sam3d_body_vit_model *)
            calloc(1, sizeof(sam3d_body_vit_model));
    m->dim        = embed_dim;
    m->n_heads    = n_heads;
    m->head_dim   = head_dim;
    m->n_blocks   = n_blocks;
    m->ffn_hidden = ffn_hidden;
    m->patch_size = patch_size;
    m->patch_pad  = 2;
    m->img_h      = img_h;
    m->img_w      = img_w;
    m->grid_h     = grid_h;
    m->grid_w     = grid_w;
    m->n_patches  = n_patches;
    m->ln_eps     = 1e-6f;
    m->image_mean[0] = 0.485f; m->image_mean[1] = 0.456f; m->image_mean[2] = 0.406f;
    m->image_std[0]  = 0.229f; m->image_std[1]  = 0.224f; m->image_std[2]  = 0.225f;
    m->st_ctx     = st;

    char buf[512];
    #define S3DVIT_FIND(suffix)  ({ \
        snprintf(buf, sizeof(buf), "%s%s", prefix, suffix); \
        int _idx = safetensors_find(st, buf); \
        (_idx >= 0) ? qt_make_tensor(st, _idx) : (qtensor){0}; \
    })

    m->patch_embed_w = S3DVIT_FIND("patch_embed.proj.weight");
    m->patch_embed_b = S3DVIT_FIND("patch_embed.proj.bias");
    m->pos_embed     = S3DVIT_FIND("pos_embed");
    m->norm_w        = S3DVIT_FIND("last_norm.weight");
    m->norm_b        = S3DVIT_FIND("last_norm.bias");

    fprintf(stderr, "sam3d_body_vit: patch_embed_w: %s, pos_embed: %s, last_norm: %s\n",
            m->patch_embed_w.data ? "loaded" : "MISSING",
            m->pos_embed.data     ? "loaded" : "MISSING",
            m->norm_w.data        ? "loaded" : "MISSING");

    m->blocks = (sam3d_body_vit_block *)
            calloc((size_t)n_blocks, sizeof(sam3d_body_vit_block));
    for (int L = 0; L < n_blocks; L++) {
        sam3d_body_vit_block *b = &m->blocks[L];
        char namebuf[160];

        #define S3DVIT_FIND_BLK(slot, suffix) do { \
            snprintf(namebuf, sizeof(namebuf), "%sblocks.%d.%s", prefix, L, suffix); \
            int _idx = safetensors_find(st, namebuf); \
            (slot) = (_idx >= 0) ? qt_make_tensor(st, _idx) : (qtensor){0}; \
        } while (0)

        S3DVIT_FIND_BLK(b->ln1_w,      "norm1.weight");
        S3DVIT_FIND_BLK(b->ln1_b,      "norm1.bias");
        S3DVIT_FIND_BLK(b->attn_qkv_w, "attn.qkv.weight");
        S3DVIT_FIND_BLK(b->attn_qkv_b, "attn.qkv.bias");
        S3DVIT_FIND_BLK(b->attn_out_w, "attn.proj.weight");
        S3DVIT_FIND_BLK(b->attn_out_b, "attn.proj.bias");
        S3DVIT_FIND_BLK(b->ln2_w,      "norm2.weight");
        S3DVIT_FIND_BLK(b->ln2_b,      "norm2.bias");
        S3DVIT_FIND_BLK(b->ffn_fc1_w,  "mlp.fc1.weight");
        S3DVIT_FIND_BLK(b->ffn_fc1_b,  "mlp.fc1.bias");
        S3DVIT_FIND_BLK(b->ffn_fc2_w,  "mlp.fc2.weight");
        S3DVIT_FIND_BLK(b->ffn_fc2_b,  "mlp.fc2.bias");

        #undef S3DVIT_FIND_BLK
    }

    #undef S3DVIT_FIND
    return m;
}

void sam3d_body_vit_free(sam3d_body_vit_model *m) {
    if (!m) return;
    /* qtensor.data lives inside the safetensors mmap (or a malloc'd
     * BF16→F32 buffer owned by qt_make_tensor; releasing the
     * safetensors context is left to the caller via st_ctx if they
     * want to free those buffers explicitly). */
    if (m->st_ctx) {
        safetensors_close((st_context *)m->st_ctx);
    }
    free(m->blocks);
    free(m);
}

#endif /* SAFETENSORS_H */

/* ==================================================================== */
/* Patch embedding (Conv2d, kernel=patch, stride=patch, padding=2)
 * Output token (py, px) = sum over (ci, kh, kw) of
 *   weight[co, ci, kh, kw] * input[ci, py*stride - pad + kh, px*stride - pad + kw]
 * with zero-pad boundary. With patch=stride=16 and pad=2, neighbouring
 * windows OVERLAP by 4 px (16+2*2=20 > 16). We do not bother with
 * im2col; the convolution is small (ci=3, k=16), so a direct loop is
 * already memory-bound. n_patches × dim × 768 ≈ 30M MACs is trivial.
 * ==================================================================== */

static void s3dvit_patch_embed(float *tokens, const float *img,
                               const qtensor *Wq, const qtensor *Bq,
                               int img_h, int img_w,
                               int grid_h, int grid_w,
                               int patch_size, int patch_pad, int dim)
{
    float *W = qt_dequant(Wq);
    float *B = qt_dequant(Bq);
    int Co = dim, Ci = 3;
    int kH = patch_size, kW = patch_size;

    #if defined(_OPENMP)
    #pragma omp parallel for collapse(2) schedule(static)
    #endif
    for (int py = 0; py < grid_h; py++) {
        for (int px = 0; px < grid_w; px++) {
            float *out = tokens + ((size_t)py * grid_w + px) * dim;
            if (B) memcpy(out, B, (size_t)dim * sizeof(float));
            else   memset(out, 0, (size_t)dim * sizeof(float));

            int ih_base = py * patch_size - patch_pad;
            int iw_base = px * patch_size - patch_pad;

            for (int co = 0; co < Co; co++) {
                float sum = 0.0f;
                for (int ci = 0; ci < Ci; ci++) {
                    for (int kh = 0; kh < kH; kh++) {
                        int ih = ih_base + kh;
                        if (ih < 0 || ih >= img_h) continue;
                        for (int kw = 0; kw < kW; kw++) {
                            int iw = iw_base + kw;
                            if (iw < 0 || iw >= img_w) continue;
                            sum += W[((co * Ci + ci) * kH + kh) * kW + kw]
                                 * img[(ci * img_h + ih) * img_w + iw];
                        }
                    }
                }
                out[co] += sum;
            }
        }
    }
    free(W); free(B);
}

/* ==================================================================== */
/* Forward                                                               */
/* ==================================================================== */

sam3d_body_vit_result sam3d_body_vit_encode_from_normalized(
        sam3d_body_vit_model *m, const float *chw,
        int img_w, int img_h, int n_threads)
{
    sam3d_body_vit_result result = {0};
    if (n_threads < 1) n_threads = 1;
    if (img_w != m->img_w || img_h != m->img_h) {
        fprintf(stderr, "sam3d_body_vit: input %dx%d != expected %dx%d\n",
                img_w, img_h, m->img_w, m->img_h);
        return result;
    }

    int dim = m->dim, np = m->n_patches;
    int gh = m->grid_h, gw = m->grid_w;
    int ffn = m->ffn_hidden;

    double t0 = sam3d_body_vit_time_ms();

    /* ─── Patch embed: (3, H, W) → (np, dim) ─── */
    float *hidden = (float *)calloc((size_t)np * dim, sizeof(float));
    s3dvit_patch_embed(hidden, chw,
                       &m->patch_embed_w, &m->patch_embed_b,
                       img_h, img_w, gh, gw,
                       m->patch_size, m->patch_pad, dim);

    /* ─── Add pos_embed[1:] + pos_embed[0] (broadcast slot 0) ───
     * Upstream: x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]
     * Slot 0 is the legacy CLS bias and is added uniformly to all
     * patches. */
    {
        int n_pe = m->pos_embed.dims[1];   /* 769 */
        float *pe = qt_dequant(&m->pos_embed);
        if (pe && n_pe == np + 1) {
            const float *cls_bias = pe;                 /* slot 0 */
            const float *patches  = pe + (size_t)dim;   /* slots 1..np */
            for (int p = 0; p < np; p++) {
                float *h = hidden + (size_t)p * dim;
                const float *q = patches + (size_t)p * dim;
                for (int d = 0; d < dim; d++)
                    h[d] += q[d] + cls_bias[d];
            }
        } else {
            fprintf(stderr,
                    "sam3d_body_vit: WARN pos_embed missing or wrong slots (%d)\n",
                    n_pe);
        }
        free(pe);
    }

    double t_embed = sam3d_body_vit_time_ms();

    /* ─── Transformer blocks ─── */
    float *ln_buf   = (float *)malloc((size_t)np * dim * sizeof(float));
    float *qkv      = (float *)malloc((size_t)np * 3 * dim * sizeof(float));
    float *attn_out = (float *)malloc((size_t)np * dim * sizeof(float));
    float *proj_out = (float *)malloc((size_t)np * dim * sizeof(float));
    float *ffn_buf  = (float *)malloc((size_t)np * ffn * sizeof(float));
    float *ffn_out  = (float *)malloc((size_t)np * dim * sizeof(float));

    for (int L = 0; L < m->n_blocks; L++) {
        sam3d_body_vit_block *b = &m->blocks[L];

        /* x = x + attn(norm1(x)) */
        s3dvit_layernorm_batch(ln_buf, hidden, &b->ln1_w, &b->ln1_b,
                               np, dim, m->ln_eps);
        s3dvit_batch_gemm(qkv, &b->attn_qkv_w, &b->attn_qkv_b,
                          ln_buf, np, 3 * dim, dim, n_threads);
        s3dvit_attention(attn_out, qkv, np, dim, m->n_heads, m->head_dim, n_threads);
        s3dvit_batch_gemm(proj_out, &b->attn_out_w, &b->attn_out_b,
                          attn_out, np, dim, dim, n_threads);
        for (size_t i = 0; i < (size_t)np * dim; i++) hidden[i] += proj_out[i];

        /* x = x + mlp(norm2(x)), GELU activation. Upstream uses
         * `nn.GELU()` with the default `approximate='none'`, which is
         * the exact erf form — tanh-approx accumulates a measurable
         * drift over 32 blocks (max-abs +0.2 vs erf on the verify
         * dump). erfBased GELU = 0.5 * x * (1 + erf(x / sqrt(2))). */
        s3dvit_layernorm_batch(ln_buf, hidden, &b->ln2_w, &b->ln2_b,
                               np, dim, m->ln_eps);
        s3dvit_batch_gemm(ffn_buf, &b->ffn_fc1_w, &b->ffn_fc1_b,
                          ln_buf, np, ffn, dim, n_threads);
        {
            const float inv_sqrt2 = 0.70710678118654752440f;
            for (size_t i = 0; i < (size_t)np * ffn; i++) {
                float v = ffn_buf[i];
                ffn_buf[i] = 0.5f * v * (1.0f + erff(v * inv_sqrt2));
            }
        }
        s3dvit_batch_gemm(ffn_out, &b->ffn_fc2_w, &b->ffn_fc2_b,
                          ffn_buf, np, dim, ffn, n_threads);
        for (size_t i = 0; i < (size_t)np * dim; i++) hidden[i] += ffn_out[i];
    }

    free(ln_buf); free(qkv); free(attn_out);
    free(proj_out); free(ffn_buf); free(ffn_out);

    double t_blocks = sam3d_body_vit_time_ms();

    /* ─── last_norm (LayerNorm32 in upstream — we already run fp32) ─── */
    float *output = (float *)malloc((size_t)np * dim * sizeof(float));
    s3dvit_layernorm_batch(output, hidden, &m->norm_w, &m->norm_b,
                           np, dim, m->ln_eps);
    free(hidden);

    double t1 = sam3d_body_vit_time_ms();
    fprintf(stderr,
            "sam3d_body_vit: patch+pe %.1f ms, %d blocks %.1f ms, "
            "last_norm %.1f ms, total %.1f ms (%d threads)\n",
            t_embed - t0, m->n_blocks, t_blocks - t_embed,
            t1 - t_blocks, t1 - t0, n_threads);

    result.tokens    = output;
    result.n_patches = np;
    result.dim       = dim;
    result.grid_h    = gh;
    result.grid_w    = gw;
    return result;
}

void sam3d_body_vit_result_free(sam3d_body_vit_result *r) {
    if (!r) return;
    free(r->tokens);
    r->tokens = NULL;
}

#endif /* SAM3D_BODY_VIT_IMPLEMENTATION */
#endif /* SAM3D_BODY_VIT_H */
