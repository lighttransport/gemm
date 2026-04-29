/*
 * rt_detr.h — RT-DETR-S (R18-VD backbone) person/object detector.
 *
 * Single-header CPU port matching HF `PekingU/rtdetr_r18vd_coco_o365`.
 * Used by sam3d_body --auto-bbox to crop the largest person before
 * mesh recovery; standalone API also exposes general detection.
 *
 * Pipeline:
 *   img → preprocess (640x640, /255, RGB) → R18-VD backbone (3 fmaps) →
 *   HybridEncoder (FPN+PAN, 1× transformer enc) → 8400 anchors →
 *   top-300 queries → 3-layer decoder with deformable msattn →
 *   (300, 80) logits + (300, 4) cxcywh boxes
 *
 * Loader strategy: tensors are kept in the safetensors mmap; conv
 * weights are BN-folded lazily on first use and cached. Only ~120
 * conv layers in this model, so the cache stays small.
 *
 * Usage:
 *   #define RT_DETR_IMPLEMENTATION
 *   #include "rt_detr.h"
 */
#ifndef GEMM_COMMON_RT_DETR_H
#define GEMM_COMMON_RT_DETR_H

#include <stdint.h>
#include <stddef.h>

#include "safetensors.h"

#ifdef __cplusplus
extern "C" {
#endif

#define RT_DETR_INPUT_SIZE 640
#define RT_DETR_NUM_QUERIES 300
#define RT_DETR_NUM_CLASSES 80
#define RT_DETR_D_MODEL 256
#define RT_DETR_FFN_DIM 1024
#define RT_DETR_NUM_HEADS 8
#define RT_DETR_NUM_DECODER_LAYERS 3
#define RT_DETR_NUM_FEATURE_LEVELS 3
#define RT_DETR_NUM_SAMPLING_POINTS 4
#define RT_DETR_PERSON_CLASS_ID 0  /* COCO "person" */

/* BN-folded conv weight cache entry. Keyed by `base` (e.g. the path up
 * to and not including ".convolution" / ".normalization"). */
typedef struct rt_detr_bn_entry {
    char *base;        /* malloc'd */
    int co, ci, kh, kw;
    float *weight;     /* (co, ci, kh, kw), malloc'd, BN-folded */
    float *bias;       /* (co), malloc'd, BN-folded (bias absorbed) */
} rt_detr_bn_entry_t;

typedef struct {
    st_context *st;            /* safetensors mmap context */
    rt_detr_bn_entry_t *cache; /* lazy BN-fold cache */
    int n_cache, cap_cache;
} rt_detr_t;

/* Detection result: cxcywh box (image px) + score + class id */
typedef struct {
    float score;
    int   class_id;
    float x0, y0, x1, y1;  /* image pixel coordinates */
} rt_detr_box_t;

/* ---- Public API ---- */

/* Load model from a single safetensors file (HF format). Tensors stay
 * mmap'd; this just opens the index. Returns NULL on failure. */
rt_detr_t *rt_detr_load(const char *safetensors_path);

void rt_detr_free(rt_detr_t *m);

/* Preprocess uint8 RGB (HWC, w*h*3) → float32 (3, 640, 640):
 *   bilinear resize to 640×640, then /255. No mean/std (do_normalize=false).
 * Caller frees the returned 3*640*640 buffer with free(). */
float *rt_detr_preprocess_image(const uint8_t *rgb, int w, int h);

/* Stage hook: run only the backbone (R18-VD), write 3 feature maps:
 *   out_s3: (128, 80, 80), stride 8
 *   out_s4: (256, 40, 40), stride 16
 *   out_s5: (512, 20, 20), stride 32
 * Caller-allocated. Returns 0 on success, negative on error. */
int rt_detr_forward_backbone(rt_detr_t *m, const float *input,
                             float *out_s3, float *out_s4, float *out_s5);

/* Run encoder stage (HybridEncoder) over the 3 backbone fmaps,
 * producing 3 fused 256-channel fmaps (same spatial sizes). */
int rt_detr_forward_encoder(rt_detr_t *m,
                            const float *bb_s3, const float *bb_s4,
                            const float *bb_s5,
                            float *out_s3, float *out_s4, float *out_s5);

/* Run the decoder over the 3 fused encoder fmaps:
 *   enc_s3 (256, 80, 80), enc_s4 (256, 40, 40), enc_s5 (256, 20, 20)
 *   out_logits: (300, 80) class logits from final decoder layer
 *   out_boxes:  (300, 4)  cxcywh in [0,1] (post-sigmoid, post-refinement) */
int rt_detr_forward_decoder(rt_detr_t *m,
                            const float *enc_s3, const float *enc_s4,
                            const float *enc_s5,
                            float *out_logits, float *out_boxes);

/* Full forward: image → 300 logits + 300 boxes (cxcywh, normalized). */
int rt_detr_forward(rt_detr_t *m, const float *input,
                    float *out_logits,    /* (300, 80) */
                    float *out_boxes);    /* (300, 4)  cxcywh in [0,1] */

/* Post-process logits+boxes (HF use_focal_loss=True style):
 *   sigmoid → flat-topk(K=300) → label = idx % 80, query = idx // 80
 *   filter score > threshold, optionally filter by class_id (>=0 to keep
 *   only one class; pass -1 to keep all classes), scale cxcywh→xyxy in
 *   image pixels.
 * out capacity is `max_out`; returns # detections written. */
int rt_detr_postprocess(const float *logits, const float *boxes_norm,
                        int orig_w, int orig_h,
                        int class_id, float score_thresh,
                        rt_detr_box_t *out, int max_out);

/* Convenience: full pipeline image (uint8 RGB HWC) → largest person bbox.
 * Returns 0 if a detection above score_thresh is found, -1 otherwise. */
int rt_detr_detect_largest_person(rt_detr_t *m,
                                  const uint8_t *rgb, int w, int h,
                                  float score_thresh,
                                  rt_detr_box_t *out);

/* ---- Internal helpers (exposed for unit tests) ---- */

/* BN-fold a single conv. Caller-allocated dst_weight/dst_bias.
 *   y = conv(x, W) + b              ← raw conv (b may be 0)
 *   y = (y - mean) / sqrt(var+eps) * gamma + beta   ← BN
 *
 *   W' = W * (gamma / sqrt(var+eps))   per-out-channel
 *   b' = (b - mean) * (gamma / sqrt(var+eps)) + beta
 */
void rt_detr_bn_fold(int co, int ci, int kh, int kw,
                     const float *src_weight, const float *src_bias,
                     const float *gamma, const float *beta,
                     const float *mean,  const float *var, float eps,
                     float *dst_weight, float *dst_bias);

/* Look up (and fold + cache on first use) a BN-folded conv at `base`.
 * Expects safetensors entries:
 *   {base}.convolution.weight   (co, ci, kh, kw)  f32
 *   {base}.normalization.weight (co)              f32  gamma
 *   {base}.normalization.bias   (co)              f32  beta
 *   {base}.normalization.running_mean
 *   {base}.normalization.running_var
 * (RT-DETR R18-VD has no conv bias; BN absorbs everything.)
 *
 * Returns 0 on success and writes pointers into the cache (do not
 * free). Negative on missing tensor. */
int rt_detr_lookup_bnfolded(rt_detr_t *m, const char *base,
                            const float **out_weight, const float **out_bias,
                            int *out_co, int *out_ci, int *out_kh, int *out_kw);

#ifdef __cplusplus
}
#endif

/* ============================== IMPLEMENTATION ============================ */
#ifdef RT_DETR_IMPLEMENTATION

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double rt_detr__time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec * 1.0e-6;
}

static int rt_detr__timing_enabled(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("RT_DETR_TIMING");
        cached = (e && e[0] && strcmp(e, "0") != 0) ? 1 : 0;
    }
    return cached;
}

void rt_detr_bn_fold(int co, int ci, int kh, int kw,
                     const float *src_weight, const float *src_bias,
                     const float *gamma, const float *beta,
                     const float *mean,  const float *var, float eps,
                     float *dst_weight, float *dst_bias)
{
    const size_t per_oc = (size_t)ci * kh * kw;
    for (int o = 0; o < co; o++) {
        float scale = gamma[o] / sqrtf(var[o] + eps);
        const float *Wsrc = src_weight + (size_t)o * per_oc;
        float       *Wdst = dst_weight + (size_t)o * per_oc;
        for (size_t k = 0; k < per_oc; k++) Wdst[k] = Wsrc[k] * scale;
        float b_in = src_bias ? src_bias[o] : 0.0f;
        dst_bias[o] = (b_in - mean[o]) * scale + beta[o];
    }
}

static int rt_detr__cache_find(rt_detr_t *m, const char *base) {
    for (int i = 0; i < m->n_cache; i++)
        if (strcmp(m->cache[i].base, base) == 0) return i;
    return -1;
}

static rt_detr_bn_entry_t *rt_detr__cache_grow(rt_detr_t *m) {
    if (m->n_cache == m->cap_cache) {
        int nc = m->cap_cache ? m->cap_cache * 2 : 16;
        rt_detr_bn_entry_t *nb = (rt_detr_bn_entry_t *)realloc(
            m->cache, (size_t)nc * sizeof(*nb));
        if (!nb) return NULL;
        m->cache = nb;
        m->cap_cache = nc;
    }
    rt_detr_bn_entry_t *e = &m->cache[m->n_cache++];
    memset(e, 0, sizeof(*e));
    return e;
}

int rt_detr_lookup_bnfolded(rt_detr_t *m, const char *base,
                            const float **out_weight, const float **out_bias,
                            int *out_co, int *out_ci, int *out_kh, int *out_kw)
{
    int hit = rt_detr__cache_find(m, base);
    rt_detr_bn_entry_t *e;
    if (hit >= 0) {
        e = &m->cache[hit];
    } else {
        char nm[512];
        /* HF uses two naming conventions:
         *   backbone:    {base}.convolution.weight  + .normalization.{...}
         *   encoder:     {base}.conv.weight         + .norm.{...}
         *   sequential:  {base}.0.weight            + .1.{...}
         * Try them in order. */
        const char *conv_suffixes[3] = {"convolution", "conv", "0"};
        const char *norm_prefixes[3] = {"normalization", "norm", "1"};
        int iw = -1, j = 0;
        for (j = 0; j < 3; j++) {
            snprintf(nm, sizeof(nm), "%s.%s.weight", base, conv_suffixes[j]);
            iw = safetensors_find(m->st, nm);
            if (iw >= 0) break;
        }
        if (iw < 0) {
            fprintf(stderr, "rt_detr: missing conv weight for base=%s\n", base);
            return -1;
        }
        const char *npfx = norm_prefixes[j];

        const uint64_t *sh = safetensors_shape(m->st, iw);
        int co = (int)sh[0], ci = (int)sh[1], kh = (int)sh[2], kw = (int)sh[3];
        const float *W = (const float *)safetensors_data(m->st, iw);

        snprintf(nm, sizeof(nm), "%s.%s.weight", base, npfx);
        int igamma = safetensors_find(m->st, nm);
        snprintf(nm, sizeof(nm), "%s.%s.bias", base, npfx);
        int ibeta  = safetensors_find(m->st, nm);
        snprintf(nm, sizeof(nm), "%s.%s.running_mean", base, npfx);
        int imean  = safetensors_find(m->st, nm);
        snprintf(nm, sizeof(nm), "%s.%s.running_var", base, npfx);
        int ivar   = safetensors_find(m->st, nm);
        if (igamma < 0 || ibeta < 0 || imean < 0 || ivar < 0) {
            fprintf(stderr, "rt_detr: missing BN params for %s (npfx=%s)\n",
                    base, npfx);
            return -1;
        }
        const float *gamma = (const float *)safetensors_data(m->st, igamma);
        const float *beta  = (const float *)safetensors_data(m->st, ibeta);
        const float *mean  = (const float *)safetensors_data(m->st, imean);
        const float *var_  = (const float *)safetensors_data(m->st, ivar);

        e = rt_detr__cache_grow(m);
        if (!e) return -1;
        e->base = strdup(base);
        e->co = co; e->ci = ci; e->kh = kh; e->kw = kw;
        e->weight = (float *)malloc((size_t)co * ci * kh * kw * sizeof(float));
        e->bias   = (float *)malloc((size_t)co * sizeof(float));
        if (!e->base || !e->weight || !e->bias) return -1;
        rt_detr_bn_fold(co, ci, kh, kw, W, NULL, gamma, beta, mean, var_,
                        1e-5f, e->weight, e->bias);
    }
    if (out_weight) *out_weight = e->weight;
    if (out_bias)   *out_bias   = e->bias;
    if (out_co) *out_co = e->co;
    if (out_ci) *out_ci = e->ci;
    if (out_kh) *out_kh = e->kh;
    if (out_kw) *out_kw = e->kw;
    return 0;
}

rt_detr_t *rt_detr_load(const char *path) {
    rt_detr_t *m = (rt_detr_t *)calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->st = safetensors_open(path);
    if (!m->st) { free(m); return NULL; }
    return m;
}

void rt_detr_free(rt_detr_t *m) {
    if (!m) return;
    if (m->st) safetensors_close(m->st);
    for (int i = 0; i < m->n_cache; i++) {
        free(m->cache[i].base);
        free(m->cache[i].weight);
        free(m->cache[i].bias);
    }
    free(m->cache);
    free(m);
}

float *rt_detr_preprocess_image(const uint8_t *rgb, int w, int h) {
    /* HF RTDetrImageProcessor: PIL bilinear resize to (640, 640), /255.
     * PIL bilinear is align_corners=False with NO antialias by default at
     * resample=2 (BILINEAR). Implement that explicitly here so we match. */
    const int S = RT_DETR_INPUT_SIZE;
    float *out = (float *)malloc((size_t)3 * S * S * sizeof(float));
    if (!out) return NULL;

    /* Inverse-mapped bilinear (align_corners=False). For each output pixel
     * (oy, ox), source coord = (oy + 0.5) * h / S - 0.5. Clamp at boundary.
     * RGB is HWC uint8; output is CHW float32 in [0,1]. */
    const float sh = (float)h / (float)S;
    const float sw = (float)w / (float)S;
    for (int oy = 0; oy < S; oy++) {
        float fy = (oy + 0.5f) * sh - 0.5f;
        int   y0 = (int)floorf(fy);
        float dy = fy - y0;
        if (y0 < 0)        { y0 = 0; dy = 0.0f; }
        if (y0 >= h - 1)   { y0 = h - 1; dy = 0.0f; }
        int   y1 = y0 + 1; if (y1 >= h) y1 = h - 1;

        for (int ox = 0; ox < S; ox++) {
            float fx = (ox + 0.5f) * sw - 0.5f;
            int   x0 = (int)floorf(fx);
            float dx = fx - x0;
            if (x0 < 0)        { x0 = 0; dx = 0.0f; }
            if (x0 >= w - 1)   { x0 = w - 1; dx = 0.0f; }
            int   x1 = x0 + 1; if (x1 >= w) x1 = w - 1;

            const uint8_t *p00 = rgb + ((size_t)y0 * w + x0) * 3;
            const uint8_t *p01 = rgb + ((size_t)y0 * w + x1) * 3;
            const uint8_t *p10 = rgb + ((size_t)y1 * w + x0) * 3;
            const uint8_t *p11 = rgb + ((size_t)y1 * w + x1) * 3;
            float w00 = (1 - dx) * (1 - dy);
            float w01 = dx       * (1 - dy);
            float w10 = (1 - dx) * dy;
            float w11 = dx       * dy;
            for (int c = 0; c < 3; c++) {
                float v = w00 * p00[c] + w01 * p01[c]
                        + w10 * p10[c] + w11 * p11[c];
                out[(size_t)c * S * S + (size_t)oy * S + ox] = v / 255.0f;
            }
        }
    }
    return out;
}

/* ---- Generic NCHW conv2d (single-image, no batching) ----
 * out[co, oh, ow] = sum_ci sum_kh sum_kw  W[co, ci, kh, kw] * in[ci, iy, ix]
 *                 + bias[co]
 * Padding is zero. iy = oy*stride + kh - pad. */
static void rt_detr__conv2d_nchw(const float *in, int ci, int ih, int iw,
                                 const float *W, const float *b,
                                 int co, int kh, int kw,
                                 int stride, int pad, int do_relu,
                                 float *out)
{
    const int oh = (ih + 2 * pad - kh) / stride + 1;
    const int ow = (iw + 2 * pad - kw) / stride + 1;
    const size_t per_oc = (size_t)ci * kh * kw;
    const size_t spat_in = (size_t)ih * iw;
    if (kh == 1 && kw == 1 && pad == 0 && stride == 1) {
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
#endif
        for (int o = 0; o < co; o++) {
            const float *Wo = W + (size_t)o * ci;
            float *Oo = out + (size_t)o * oh * ow;
            const float bias = b ? b[o] : 0.0f;
            const size_t spat = (size_t)oh * ow;
            for (size_t i = 0; i < spat; i++) Oo[i] = bias;
            for (int c = 0; c < ci; c++) {
                const float wc = Wo[c];
                const float *Ic = in + (size_t)c * spat_in;
#if defined(__GNUC__) || defined(__clang__)
                #pragma GCC ivdep
#endif
                for (size_t i = 0; i < spat; i++) Oo[i] += wc * Ic[i];
            }
            if (do_relu) {
                for (size_t i = 0; i < spat; i++)
                    if (Oo[i] < 0.0f) Oo[i] = 0.0f;
            }
        }
        return;
    }
    if (kh == 1 && kw == 1 && pad == 0) {
#if defined(_OPENMP)
        #pragma omp parallel for collapse(2) schedule(static)
#endif
        for (int o = 0; o < co; o++) {
            for (int oy = 0; oy < oh; oy++) {
                const float *Wo = W + (size_t)o * ci;
                float *Oo = out + (size_t)o * oh * ow + (size_t)oy * ow;
                const float bias = b ? b[o] : 0.0f;
                const int iy = oy * stride;
                for (int ox = 0; ox < ow; ox++) {
                    const int ix = ox * stride;
                    float acc = bias;
#if defined(__GNUC__) || defined(__clang__)
                    #pragma GCC ivdep
#endif
                    for (int c = 0; c < ci; c++) {
                        acc += Wo[c] * in[(size_t)c * spat_in + (size_t)iy * iw + ix];
                    }
                    if (do_relu && acc < 0.0f) acc = 0.0f;
                    Oo[ox] = acc;
                }
            }
        }
        return;
    }
    if (kh == 3 && kw == 3 && stride == 1 && pad == 1) {
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
#endif
        for (int o = 0; o < co; o++) {
            const float *Wo = W + (size_t)o * per_oc;
            float *Oo = out + (size_t)o * oh * ow;
            const float bias = b ? b[o] : 0.0f;

            {
                const size_t spat = (size_t)oh * ow;
                for (size_t i = 0; i < spat; i++) Oo[i] = bias;
            }

            for (int c = 0; c < ci; c++) {
                const float *Wc = Wo + (size_t)c * 9;
                const float *Ic0 = in + (size_t)c * spat_in;
                for (int fy = 0; fy < 3; fy++) {
                    const int row_shift = fy - 1;
                    for (int fx = 0; fx < 3; fx++) {
                        const float wc = Wc[fy * 3 + fx];
                        const int col_shift = fx - 1;
                        for (int oy = 1; oy < oh - 1; oy++) {
                            float *Orow = Oo + (size_t)oy * ow;
                            const float *Irow = Ic0 + (size_t)(oy + row_shift) * iw
                                                + 1 + col_shift;
#if defined(__GNUC__) || defined(__clang__)
                            #pragma GCC ivdep
#endif
                            for (int ox = 1; ox < ow - 1; ox++) {
                                Orow[ox] += wc * Irow[ox - 1];
                            }
                        }
                    }
                }
            }

            for (int oy = 0; oy < oh; oy++) {
                for (int ox = 0; ox < ow; ox++) {
                    if ((unsigned)(oy - 1) < (unsigned)(oh - 2) &&
                        (unsigned)(ox - 1) < (unsigned)(ow - 2))
                        continue;
                    float acc = bias;
                    int iy0 = oy - 1;
                    int ix0 = ox - 1;
                    for (int c = 0; c < ci; c++) {
                        const float *Wc = Wo + (size_t)c * 9;
                        const float *Ic = in + (size_t)c * spat_in;
                        for (int fy = 0; fy < 3; fy++) {
                            int iy = iy0 + fy;
                            if ((unsigned)iy >= (unsigned)ih) continue;
                            for (int fx = 0; fx < 3; fx++) {
                                int ix = ix0 + fx;
                                if ((unsigned)ix >= (unsigned)iw) continue;
                                acc += Wc[fy * 3 + fx] * Ic[iy * iw + ix];
                            }
                        }
                    }
                    Oo[oy * ow + ox] = acc;
                }
            }
            if (do_relu) {
                const size_t spat = (size_t)oh * ow;
                for (size_t i = 0; i < spat; i++)
                    if (Oo[i] < 0.0f) Oo[i] = 0.0f;
            }
        }
        return;
    }
    if (kh == 3 && kw == 3 && stride == 2 && pad == 1) {
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
#endif
        for (int o = 0; o < co; o++) {
            const float *Wo = W + (size_t)o * per_oc;
            float *Oo = out + (size_t)o * oh * ow;
            const float bias = b ? b[o] : 0.0f;

            for (int oy = 1; oy < oh - 1; oy++) {
                int iy0 = oy * 2 - 1;
                for (int ox = 1; ox < ow - 1; ox++) {
                    int ix0 = ox * 2 - 1;
                    float acc = bias;
                    for (int c = 0; c < ci; c++) {
                        const float *Wc = Wo + (size_t)c * 9;
                        const float *Ic = in + (size_t)c * spat_in
                                        + (size_t)iy0 * iw + ix0;
                        acc += Wc[0] * Ic[0]      + Wc[1] * Ic[1]      + Wc[2] * Ic[2]
                             + Wc[3] * Ic[iw]     + Wc[4] * Ic[iw + 1] + Wc[5] * Ic[iw + 2]
                             + Wc[6] * Ic[2 * iw] + Wc[7] * Ic[2 * iw + 1]
                             + Wc[8] * Ic[2 * iw + 2];
                    }
                    if (do_relu && acc < 0.0f) acc = 0.0f;
                    Oo[oy * ow + ox] = acc;
                }
            }

            for (int oy = 0; oy < oh; oy++) {
                for (int ox = 0; ox < ow; ox++) {
                    if ((unsigned)(oy - 1) < (unsigned)(oh - 2) &&
                        (unsigned)(ox - 1) < (unsigned)(ow - 2))
                        continue;
                    float acc = bias;
                    int iy0 = oy * 2 - 1;
                    int ix0 = ox * 2 - 1;
                    for (int c = 0; c < ci; c++) {
                        const float *Wc = Wo + (size_t)c * 9;
                        const float *Ic = in + (size_t)c * spat_in;
                        for (int fy = 0; fy < 3; fy++) {
                            int iy = iy0 + fy;
                            if ((unsigned)iy >= (unsigned)ih) continue;
                            for (int fx = 0; fx < 3; fx++) {
                                int ix = ix0 + fx;
                                if ((unsigned)ix >= (unsigned)iw) continue;
                                acc += Wc[fy * 3 + fx] * Ic[iy * iw + ix];
                            }
                        }
                    }
                    if (do_relu && acc < 0.0f) acc = 0.0f;
                    Oo[oy * ow + ox] = acc;
                }
            }
        }
        return;
    }
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int o = 0; o < co; o++) {
        const float *Wo = W + (size_t)o * per_oc;
        float *Oo = out + (size_t)o * oh * ow;
        const float bias = b ? b[o] : 0.0f;
        for (int oy = 0; oy < oh; oy++) {
            for (int ox = 0; ox < ow; ox++) {
                float acc = bias;
                int iy0 = oy * stride - pad;
                int ix0 = ox * stride - pad;
                for (int c = 0; c < ci; c++) {
                    const float *Wc = Wo + (size_t)c * kh * kw;
                    const float *Ic = in + (size_t)c * spat_in;
                    for (int fy = 0; fy < kh; fy++) {
                        int iy = iy0 + fy;
                        if ((unsigned)iy >= (unsigned)ih) continue;
                        for (int fx = 0; fx < kw; fx++) {
                            int ix = ix0 + fx;
                            if ((unsigned)ix >= (unsigned)iw) continue;
                            acc += Wc[fy * kw + fx] * Ic[iy * iw + ix];
                        }
                    }
                }
                if (do_relu && acc < 0.0f) acc = 0.0f;
                Oo[oy * ow + ox] = acc;
            }
        }
    }
}

/* MaxPool 2D with kernel=3, stride=2, pad=1 (matches HF R18 max-pool). */
static void rt_detr__maxpool_3x3_s2(const float *in, int c, int ih, int iw,
                                    float *out)
{
    const int oh = (ih + 2 - 3) / 2 + 1;
    const int ow = (iw + 2 - 3) / 2 + 1;
    const size_t spat_in = (size_t)ih * iw;
    for (int ch = 0; ch < c; ch++) {
        const float *Ic = in + (size_t)ch * spat_in;
        float *Oc = out + (size_t)ch * oh * ow;
        for (int oy = 0; oy < oh; oy++) {
            int iy0 = oy * 2 - 1;
            for (int ox = 0; ox < ow; ox++) {
                int ix0 = ox * 2 - 1;
                float mx = -INFINITY;
                for (int fy = 0; fy < 3; fy++) {
                    int iy = iy0 + fy;
                    if ((unsigned)iy >= (unsigned)ih) continue;
                    for (int fx = 0; fx < 3; fx++) {
                        int ix = ix0 + fx;
                        if ((unsigned)ix >= (unsigned)iw) continue;
                        float v = Ic[iy * iw + ix];
                        if (v > mx) mx = v;
                    }
                }
                Oc[oy * ow + ox] = mx;
            }
        }
    }
}

/* AvgPool 2D, kernel=2, stride=2, no pad. */
static void rt_detr__avgpool_2x2_s2(const float *in, int c, int ih, int iw,
                                    float *out)
{
    const int oh = ih / 2;
    const int ow = iw / 2;
    for (int ch = 0; ch < c; ch++) {
        const float *Ic = in + (size_t)ch * ih * iw;
        float *Oc = out + (size_t)ch * oh * ow;
        for (int oy = 0; oy < oh; oy++) {
            for (int ox = 0; ox < ow; ox++) {
                int iy0 = oy * 2, ix0 = ox * 2;
                float s = Ic[iy0 * iw + ix0]
                        + Ic[iy0 * iw + (ix0 + 1)]
                        + Ic[(iy0 + 1) * iw + ix0]
                        + Ic[(iy0 + 1) * iw + (ix0 + 1)];
                Oc[oy * ow + ox] = s * 0.25f;
            }
        }
    }
}

static void rt_detr__add_relu_inplace(float *a, const float *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float v = a[i] + b[i];
        a[i] = v < 0.0f ? 0.0f : v;
    }
}

/* Run a BN-folded conv from cache by base name; allocate and return the
 * output. `do_relu`: apply ReLU after add. */
static float *rt_detr__run_conv(rt_detr_t *m, const char *base,
                                const float *in, int ih, int iw,
                                int stride, int pad, int do_relu,
                                int *out_co, int *out_oh, int *out_ow)
{
    const float *W; const float *b;
    int co, ci, kh, kw;
    if (rt_detr_lookup_bnfolded(m, base, &W, &b, &co, &ci, &kh, &kw) != 0)
        return NULL;
    int oh = (ih + 2 * pad - kh) / stride + 1;
    int ow = (iw + 2 * pad - kw) / stride + 1;
    float *out = (float *)malloc((size_t)co * oh * ow * sizeof(float));
    if (!out) return NULL;
    rt_detr__conv2d_nchw(in, ci, ih, iw, W, b, co, kh, kw,
                         stride, pad, do_relu, out);
    if (out_co) *out_co = co;
    if (out_oh) *out_oh = oh;
    if (out_ow) *out_ow = ow;
    return out;
}

/* R18-VD BasicBlock forward with residual.
 *   y = layer.1(layer.0(x))  ← BN-folded, ReLU after layer.0 only
 *   shortcut: identity OR shortcut.* (1×1 conv) OR shortcut.1.* (avgpool + 1×1)
 *   out = ReLU(y + shortcut(x))
 */
static float *rt_detr__basic_block(rt_detr_t *m, const char *stage_base,
                                   int layer_idx, int has_avgpool_shortcut,
                                   const float *in, int ih, int iw,
                                   int *out_co, int *out_oh, int *out_ow)
{
    char buf[512];
    int oh, ow, co, mid_h, mid_w, mid_c;
    /* layer.0: 3×3, optionally stride 2 (when has_avgpool_shortcut: stride 2) */
    int s = has_avgpool_shortcut ? 2 : 1;
    snprintf(buf, sizeof(buf), "%s.layers.%d.layer.0", stage_base, layer_idx);
    float *t1 = rt_detr__run_conv(m, buf, in, ih, iw, s, 1, /*relu=*/1,
                                  &mid_c, &mid_h, &mid_w);
    if (!t1) return NULL;

    snprintf(buf, sizeof(buf), "%s.layers.%d.layer.1", stage_base, layer_idx);
    float *t2 = rt_detr__run_conv(m, buf, t1, mid_h, mid_w, 1, 1, /*relu=*/0,
                                  &co, &oh, &ow);
    free(t1);
    if (!t2) return NULL;

    /* Shortcut */
    int sc_co, sc_oh, sc_ow;
    float *sc;
    if (layer_idx == 0) {
        if (has_avgpool_shortcut) {
            /* avgpool 2×2 s2 → 1×1 conv (BN-folded) */
            float *pooled = (float *)malloc(
                (size_t)mid_c /* dummy: we don't know in_c yet — use ih·iw scale */
                * sizeof(float));
            (void)pooled; free(pooled);
            /* Allocate properly: input has same channels as `in`, ih,iw → oh',ow' = oh,ow */
            int in_c;  /* need to know ci of shortcut.1 to size pool buffer */
            /* lookup shortcut.1 to get ci */
            const float *Ws; const float *bs;
            int sc_kh, sc_kw, sc_ci;
            snprintf(buf, sizeof(buf), "%s.layers.%d.shortcut.1", stage_base, layer_idx);
            if (rt_detr_lookup_bnfolded(m, buf, &Ws, &bs, &sc_co, &sc_ci, &sc_kh, &sc_kw) != 0) {
                free(t2); return NULL;
            }
            in_c = sc_ci;
            int p_h = ih / 2, p_w = iw / 2;
            float *pool_buf = (float *)malloc((size_t)in_c * p_h * p_w * sizeof(float));
            if (!pool_buf) { free(t2); return NULL; }
            rt_detr__avgpool_2x2_s2(in, in_c, ih, iw, pool_buf);
            sc = (float *)malloc((size_t)sc_co * p_h * p_w * sizeof(float));
            if (!sc) { free(pool_buf); free(t2); return NULL; }
            rt_detr__conv2d_nchw(pool_buf, in_c, p_h, p_w, Ws, bs,
                                 sc_co, 1, 1, 1, 0, /*relu=*/0, sc);
            sc_oh = p_h; sc_ow = p_w;
            free(pool_buf);
        } else {
            /* Stage 0: shortcut.convolution (1×1, no avgpool) */
            snprintf(buf, sizeof(buf), "%s.layers.%d.shortcut", stage_base, layer_idx);
            sc = rt_detr__run_conv(m, buf, in, ih, iw, 1, 0, /*relu=*/0,
                                   &sc_co, &sc_oh, &sc_ow);
            if (!sc) { free(t2); return NULL; }
        }
    } else {
        /* No shortcut conv — input must already match (co, oh, ow). */
        sc = (float *)malloc((size_t)co * oh * ow * sizeof(float));
        if (!sc) { free(t2); return NULL; }
        memcpy(sc, in, (size_t)co * oh * ow * sizeof(float));
        sc_co = co; sc_oh = oh; sc_ow = ow;
    }

    /* sanity */
    if (sc_co != co || sc_oh != oh || sc_ow != ow) {
        fprintf(stderr, "rt_detr: basic_block shape mismatch sc=(%d,%d,%d) main=(%d,%d,%d)\n",
                sc_co, sc_oh, sc_ow, co, oh, ow);
        free(t2); free(sc); return NULL;
    }
    rt_detr__add_relu_inplace(t2, sc, (size_t)co * oh * ow);
    free(sc);
    *out_co = co; *out_oh = oh; *out_ow = ow;
    return t2;
}

int rt_detr_forward_backbone(rt_detr_t *m, const float *input,
                             float *out_s3, float *out_s4, float *out_s5)
{
    /* Stem: 3 BN-folded conv 3×3 with ReLU.
     * 0: 3 → 32 stride 2 → 320×320
     * 1: 32 → 32 stride 1 → 320×320
     * 2: 32 → 64 stride 1 → 320×320 */
    int co, oh, ow;
    float *x = rt_detr__run_conv(m, "model.backbone.model.embedder.embedder.0",
                                 input, 640, 640, 2, 1, 1, &co, &oh, &ow);
    if (!x) return -1;
    {
        float *y = rt_detr__run_conv(m, "model.backbone.model.embedder.embedder.1",
                                     x, oh, ow, 1, 1, 1, &co, &oh, &ow);
        free(x); x = y; if (!x) return -1;
    }
    {
        float *y = rt_detr__run_conv(m, "model.backbone.model.embedder.embedder.2",
                                     x, oh, ow, 1, 1, 1, &co, &oh, &ow);
        free(x); x = y; if (!x) return -1;
    }

    /* MaxPool 3×3 stride 2 pad 1 → 160×160 */
    int p_h = oh / 2, p_w = ow / 2;
    float *pool = (float *)malloc((size_t)co * p_h * p_w * sizeof(float));
    if (!pool) { free(x); return -1; }
    rt_detr__maxpool_3x3_s2(x, co, oh, ow, pool);
    free(x); x = pool;
    int h = p_h, w = p_w;

    /* Stage 0: 2 BasicBlocks at 64ch, NO downsample (stage_0 layers.0 has
     *          a 1×1 shortcut.convolution, no avgpool) */
    for (int li = 0; li < 2; li++) {
        float *y = rt_detr__basic_block(m, "model.backbone.model.encoder.stages.0",
                                        li, /*avgpool=*/0, x, h, w, &co, &h, &w);
        free(x); x = y; if (!x) return -1;
    }

    /* Stage 1: 2 BasicBlocks at 128ch, first has avgpool+1×1 shortcut */
    for (int li = 0; li < 2; li++) {
        float *y = rt_detr__basic_block(m, "model.backbone.model.encoder.stages.1",
                                        li, /*avgpool=*/(li == 0),
                                        x, h, w, &co, &h, &w);
        free(x); x = y; if (!x) return -1;
    }
    /* x is now (128, 80, 80) — bb_s3 */
    memcpy(out_s3, x, (size_t)co * h * w * sizeof(float));

    /* Stage 2 → 256ch */
    for (int li = 0; li < 2; li++) {
        float *y = rt_detr__basic_block(m, "model.backbone.model.encoder.stages.2",
                                        li, /*avgpool=*/(li == 0),
                                        x, h, w, &co, &h, &w);
        free(x); x = y; if (!x) return -1;
    }
    memcpy(out_s4, x, (size_t)co * h * w * sizeof(float));

    /* Stage 3 → 512ch */
    for (int li = 0; li < 2; li++) {
        float *y = rt_detr__basic_block(m, "model.backbone.model.encoder.stages.3",
                                        li, /*avgpool=*/(li == 0),
                                        x, h, w, &co, &h, &w);
        free(x); x = y; if (!x) return -1;
    }
    memcpy(out_s5, x, (size_t)co * h * w * sizeof(float));
    free(x);
    return 0;
}

/* ========================== HybridEncoder helpers ========================= */

static inline void rt_detr__silu_inplace(float *x, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float v = x[i];
        x[i] = v / (1.0f + expf(-v));
    }
}

/* tanh-approx GELU (matches torch.nn.functional.gelu with approximate='none'
 * to ~1e-6; HF uses the exact erf form by default but here we mirror the
 * approximation already used elsewhere in this codebase). For accuracy we
 * switch to the erf-form to match torch's default. */
static inline void rt_detr__gelu_inplace(float *x, size_t n) {
    /* erf-based GELU: 0.5 * x * (1 + erf(x / sqrt(2))) */
    const float inv_sqrt2 = 0.7071067811865475f;
    for (size_t i = 0; i < n; i++) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + erff(v * inv_sqrt2));
    }
}

/* Nearest-neighbour 2× upsample on NCHW (single image). */
static void rt_detr__nearest_upsample_2x(const float *in, int c, int ih, int iw,
                                         float *out)
{
    int oh = ih * 2, ow = iw * 2;
    for (int ch = 0; ch < c; ch++) {
        const float *Ic = in  + (size_t)ch * ih * iw;
        float       *Oc = out + (size_t)ch * oh * ow;
        for (int oy = 0; oy < oh; oy++) {
            int iy = oy / 2;
            for (int ox = 0; ox < ow; ox++) {
                int ix = ox / 2;
                Oc[oy * ow + ox] = Ic[iy * iw + ix];
            }
        }
    }
}

/* Concatenate two CHW tensors along the channel axis. */
static void rt_detr__concat_channels(const float *a, int ca,
                                     const float *b, int cb,
                                     int h, int w, float *out)
{
    size_t spat = (size_t)h * w;
    memcpy(out,                       a, (size_t)ca * spat * sizeof(float));
    memcpy(out + (size_t)ca * spat,   b, (size_t)cb * spat * sizeof(float));
}

/* LayerNorm over last dim. */
static void rt_detr__layernorm(float *dst, const float *src,
                               const float *w, const float *b,
                               int n_tok, int dim, float eps)
{
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int t = 0; t < n_tok; t++) {
        const float *x = src + (size_t)t * dim;
        float       *y = dst + (size_t)t * dim;
        float mean = 0.0f;
        for (int i = 0; i < dim; i++) mean += x[i];
        mean /= (float)dim;
        float var = 0.0f;
        for (int i = 0; i < dim; i++) { float d = x[i] - mean; var += d * d; }
        var /= (float)dim;
        float inv = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < dim; i++)
            y[i] = (x[i] - mean) * inv * w[i] + b[i];
    }
}

/* dst[t, r] = sum_j W[r, j] * src[t, j] + (bias ? bias[r] : 0)
 *   W:   (n_out, n_in) row-major
 *   src: (n_tok, n_in) row-major
 *   dst: (n_tok, n_out) row-major  */
static void rt_detr__gemm_f32(float *dst, const float *W, const float *bias,
                              const float *src, int n_tok, int n_out, int n_in)
{
#if defined(_OPENMP)
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int t = 0; t < n_tok; t++) {
        for (int r = 0; r < n_out; r++) {
            const float *a = src + (size_t)t * n_in;
            const float *w = W   + (size_t)r * n_in;
            float s = bias ? bias[r] : 0.0f;
            for (int j = 0; j < n_in; j++) s += a[j] * w[j];
            dst[(size_t)t * n_out + r] = s;
        }
    }
}

/* Build 2D sin-cos positional embedding (1, H*W, embed_dim).
 * Channel layout: [sin(y*omega) cos(y*omega) sin(x*omega) cos(x*omega)],
 * each block of width embed_dim/4. omega[k] = 1 / temp^(k/(embed_dim/4)).
 * Matches RTDetrSinePositionEmbedding.forward(). */
static void rt_detr__sincos_pos_embed_2d(int h, int w, int embed_dim,
                                         float temperature, float *out)
{
    int pos_dim = embed_dim / 4;
    float *omega = (float *)malloc((size_t)pos_dim * sizeof(float));
    for (int k = 0; k < pos_dim; k++) {
        omega[k] = 1.0f / powf(temperature, (float)k / (float)pos_dim);
    }
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float *o = out + (size_t)(y * w + x) * embed_dim;
            for (int k = 0; k < pos_dim; k++) {
                float wy = (float)y * omega[k];
                float wx = (float)x * omega[k];
                o[k]                 = sinf(wy);
                o[pos_dim + k]       = cosf(wy);
                o[2 * pos_dim + k]   = sinf(wx);
                o[3 * pos_dim + k]   = cosf(wx);
            }
        }
    }
    free(omega);
}

/* Look up a Linear (weight + bias) by base name. Returns 0 on success.
 * Expects safetensors entries `{base}.weight` (n_out, n_in) and
 * `{base}.bias` (n_out). */
static int rt_detr__linear(rt_detr_t *m, const char *base,
                           const float **W, const float **b,
                           int *n_out, int *n_in)
{
    char nm[512];
    snprintf(nm, sizeof(nm), "%s.weight", base);
    int iw = safetensors_find(m->st, nm);
    if (iw < 0) { fprintf(stderr, "rt_detr: missing %s\n", nm); return -1; }
    const uint64_t *sh = safetensors_shape(m->st, iw);
    *n_out = (int)sh[0]; *n_in = (int)sh[1];
    *W = (const float *)safetensors_data(m->st, iw);
    snprintf(nm, sizeof(nm), "%s.bias", base);
    int ib = safetensors_find(m->st, nm);
    if (ib < 0) { fprintf(stderr, "rt_detr: missing %s\n", nm); return -1; }
    *b = (const float *)safetensors_data(m->st, ib);
    return 0;
}

/* RepVgg block: silu(BN(conv1_3x3(x)) + BN(conv2_1x1(x))).
 * Both convs are BN-folded (no internal activation). Pure post-add silu.
 *   in/out shape: (c, h, w)  (channels unchanged) */
static int rt_detr__rep_vgg_block(rt_detr_t *m, const char *base,
                                  const float *in, int c, int h, int w,
                                  float *out)
{
    char buf[512];
    snprintf(buf, sizeof(buf), "%s.conv1", base);
    const float *W1; const float *b1;
    int co1, ci1, kh1, kw1;
    if (rt_detr_lookup_bnfolded(m, buf, &W1, &b1, &co1, &ci1, &kh1, &kw1) != 0)
        return -1;
    snprintf(buf, sizeof(buf), "%s.conv2", base);
    const float *W2; const float *b2;
    int co2, ci2, kh2, kw2;
    if (rt_detr_lookup_bnfolded(m, buf, &W2, &b2, &co2, &ci2, &kh2, &kw2) != 0)
        return -1;

    /* conv1 (3×3 pad 1) into out, conv2 (1×1 pad 0) into tmp, then add. */
    rt_detr__conv2d_nchw(in, ci1, h, w, W1, b1, co1, kh1, kw1, 1, 1, 0, out);
    float *tmp = (float *)malloc((size_t)co2 * h * w * sizeof(float));
    if (!tmp) return -1;
    rt_detr__conv2d_nchw(in, ci2, h, w, W2, b2, co2, kh2, kw2, 1, 0, 0, tmp);
    size_t n = (size_t)c * h * w;
    for (size_t i = 0; i < n; i++) out[i] += tmp[i];
    free(tmp);
    rt_detr__silu_inplace(out, n);
    (void)co1; (void)co2;
    return 0;
}

/* CSPRepLayer:
 *   t1 = silu(BN(conv1_1x1(x)))     (in_c → 128)
 *   t1 = bottlenecks[0..2](t1)      (128 → 128 each)
 *   t2 = silu(BN(conv2_1x1(x)))     (in_c → 128)
 *   y  = silu(BN(conv3_1x1(t1 + t2)))  (128 → out_c=256)
 * For RT-DETR-S: in_c=512, out_c=256, hidden=128. */
static int rt_detr__csp_rep_layer(rt_detr_t *m, const char *base,
                                  const float *in, int in_c, int h, int w,
                                  int *out_c_p, float *out)
{
    (void)in_c;
    /* conv1 (in_c → 128, 1×1, silu) */
    int t1_co = 128;
    float *t1 = (float *)malloc((size_t)t1_co * h * w * sizeof(float));
    if (!t1) return -1;
    char buf[512];
    snprintf(buf, sizeof(buf), "%s.conv1", base);
    const float *W; const float *b;
    int co, ci, kh, kw;
    if (rt_detr_lookup_bnfolded(m, buf, &W, &b, &co, &ci, &kh, &kw) != 0) {
        free(t1); return -1;
    }
    rt_detr__conv2d_nchw(in, ci, h, w, W, b, co, kh, kw, 1, 0, 0, t1);
    rt_detr__silu_inplace(t1, (size_t)co * h * w);
    t1_co = co;

    /* 3 RepVgg bottlenecks on t1 path, each preserves channels */
    float *bn_buf = (float *)malloc((size_t)t1_co * h * w * sizeof(float));
    if (!bn_buf) { free(t1); return -1; }
    for (int i = 0; i < 3; i++) {
        snprintf(buf, sizeof(buf), "%s.bottlenecks.%d", base, i);
        if (rt_detr__rep_vgg_block(m, buf, t1, t1_co, h, w, bn_buf) != 0) {
            free(t1); free(bn_buf); return -1;
        }
        memcpy(t1, bn_buf, (size_t)t1_co * h * w * sizeof(float));
    }
    free(bn_buf);

    /* conv2 (in_c → 128, 1×1, silu) on the original input */
    snprintf(buf, sizeof(buf), "%s.conv2", base);
    if (rt_detr_lookup_bnfolded(m, buf, &W, &b, &co, &ci, &kh, &kw) != 0) {
        free(t1); return -1;
    }
    float *t2 = (float *)malloc((size_t)co * h * w * sizeof(float));
    if (!t2) { free(t1); return -1; }
    rt_detr__conv2d_nchw(in, ci, h, w, W, b, co, kh, kw, 1, 0, 0, t2);
    rt_detr__silu_inplace(t2, (size_t)co * h * w);
    /* Add t2 onto t1 */
    {
        size_t n = (size_t)t1_co * h * w;
        for (size_t i = 0; i < n; i++) t1[i] += t2[i];
    }
    free(t2);

    /* conv3 (128 → out_c, 1×1, silu) */
    snprintf(buf, sizeof(buf), "%s.conv3", base);
    if (rt_detr_lookup_bnfolded(m, buf, &W, &b, &co, &ci, &kh, &kw) != 0) {
        free(t1); return -1;
    }
    rt_detr__conv2d_nchw(t1, ci, h, w, W, b, co, kh, kw, 1, 0, 0, out);
    rt_detr__silu_inplace(out, (size_t)co * h * w);
    free(t1);
    if (out_c_p) *out_c_p = co;
    return 0;
}

/* Run a single transformer encoder layer (RT-DETR-S AIFI):
 *   q = k = x + pos_embed,  v = x
 *   attn = MHA(q, k, v)  →  out_proj  →  add+LN1
 *   ffn  = LN2(prev + fc2(GELU(fc1(prev))))
 *
 * x: (n_tok, dim) — modified in place (post-LN result returned in x).
 * pos_embed: (n_tok, dim). */
static int rt_detr__encoder_layer(rt_detr_t *m, const char *base,
                                  float *x, const float *pos_embed,
                                  int n_tok, int dim, int n_heads)
{
    const int head_dim = dim / n_heads;
    const float scale = 1.0f / sqrtf((float)head_dim);

    /* qk_input = x + pos_embed  (n_tok, dim) */
    float *qk_in = (float *)malloc((size_t)n_tok * dim * sizeof(float));
    if (!qk_in) return -1;
    {
        size_t n = (size_t)n_tok * dim;
        for (size_t i = 0; i < n; i++) qk_in[i] = x[i] + pos_embed[i];
    }

    /* Q, K, V projections */
    char buf[512];
    const float *Wq, *bq, *Wk, *bk, *Wv, *bv, *Wo, *bo;
    int n_out, n_in;
    snprintf(buf, sizeof(buf), "%s.self_attn.q_proj", base);
    if (rt_detr__linear(m, buf, &Wq, &bq, &n_out, &n_in) != 0) { free(qk_in); return -1; }
    snprintf(buf, sizeof(buf), "%s.self_attn.k_proj", base);
    if (rt_detr__linear(m, buf, &Wk, &bk, &n_out, &n_in) != 0) { free(qk_in); return -1; }
    snprintf(buf, sizeof(buf), "%s.self_attn.v_proj", base);
    if (rt_detr__linear(m, buf, &Wv, &bv, &n_out, &n_in) != 0) { free(qk_in); return -1; }
    snprintf(buf, sizeof(buf), "%s.self_attn.out_proj", base);
    if (rt_detr__linear(m, buf, &Wo, &bo, &n_out, &n_in) != 0) { free(qk_in); return -1; }

    float *Q = (float *)malloc((size_t)n_tok * dim * sizeof(float));
    float *K = (float *)malloc((size_t)n_tok * dim * sizeof(float));
    float *V = (float *)malloc((size_t)n_tok * dim * sizeof(float));
    float *attn_out = (float *)malloc((size_t)n_tok * dim * sizeof(float));
    if (!Q || !K || !V || !attn_out) {
        free(qk_in); free(Q); free(K); free(V); free(attn_out); return -1;
    }
    rt_detr__gemm_f32(Q, Wq, bq, qk_in, n_tok, dim, dim);
    rt_detr__gemm_f32(K, Wk, bk, qk_in, n_tok, dim, dim);
    rt_detr__gemm_f32(V, Wv, bv, x,     n_tok, dim, dim);
    free(qk_in);

    /* MHA over (n_heads, n_tok, head_dim). For each head independently:
     *   scores[t1, t2] = sum_d Q[t1, h*hd + d] * K[t2, h*hd + d] * scale
     *   p[t1, :] = softmax(scores[t1, :])
     *   y[t1, d] = sum_t2 p[t1, t2] * V[t2, h*hd + d]  */
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int h = 0; h < n_heads; h++) {
        float *scores = (float *)malloc((size_t)n_tok * sizeof(float));
        for (int t1 = 0; t1 < n_tok; t1++) {
            const float *q = Q + (size_t)t1 * dim + h * head_dim;
            for (int t2 = 0; t2 < n_tok; t2++) {
                const float *k = K + (size_t)t2 * dim + h * head_dim;
                float s = 0.0f;
                for (int d = 0; d < head_dim; d++) s += q[d] * k[d];
                scores[t2] = s * scale;
            }
            /* softmax over scores */
            float mx = scores[0];
            for (int j = 1; j < n_tok; j++) if (scores[j] > mx) mx = scores[j];
            float sum = 0.0f;
            for (int j = 0; j < n_tok; j++) { scores[j] = expf(scores[j] - mx); sum += scores[j]; }
            float inv = 1.0f / sum;
            for (int j = 0; j < n_tok; j++) scores[j] *= inv;
            /* y = scores @ V[:, h*hd:(h+1)*hd] */
            float *y = attn_out + (size_t)t1 * dim + h * head_dim;
            for (int d = 0; d < head_dim; d++) y[d] = 0.0f;
            for (int t2 = 0; t2 < n_tok; t2++) {
                const float *v = V + (size_t)t2 * dim + h * head_dim;
                float p = scores[t2];
                for (int d = 0; d < head_dim; d++) y[d] += p * v[d];
            }
        }
        free(scores);
    }
    free(Q); free(K); free(V);

    /* out_proj */
    float *proj = (float *)malloc((size_t)n_tok * dim * sizeof(float));
    if (!proj) { free(attn_out); return -1; }
    rt_detr__gemm_f32(proj, Wo, bo, attn_out, n_tok, dim, dim);
    free(attn_out);

    /* x = LN(x + proj)  (post-LN) */
    {
        size_t n = (size_t)n_tok * dim;
        for (size_t i = 0; i < n; i++) x[i] += proj[i];
    }
    free(proj);

    snprintf(buf, sizeof(buf), "%s.self_attn_layer_norm.weight", base);
    int iln1w = safetensors_find(m->st, buf);
    snprintf(buf, sizeof(buf), "%s.self_attn_layer_norm.bias",   base);
    int iln1b = safetensors_find(m->st, buf);
    if (iln1w < 0 || iln1b < 0) { return -1; }
    const float *ln1w = (const float *)safetensors_data(m->st, iln1w);
    const float *ln1b = (const float *)safetensors_data(m->st, iln1b);
    rt_detr__layernorm(x, x, ln1w, ln1b, n_tok, dim, 1e-5f);

    /* FFN: fc1 (dim → 1024) GELU fc2 (1024 → dim) */
    const float *Wfc1, *bfc1, *Wfc2, *bfc2;
    int ff_out, ff_in;
    snprintf(buf, sizeof(buf), "%s.fc1", base);
    if (rt_detr__linear(m, buf, &Wfc1, &bfc1, &ff_out, &ff_in) != 0) return -1;
    snprintf(buf, sizeof(buf), "%s.fc2", base);
    if (rt_detr__linear(m, buf, &Wfc2, &bfc2, &ff_out, &ff_in) != 0) return -1;
    int ffn_dim = RT_DETR_FFN_DIM;

    float *ff = (float *)malloc((size_t)n_tok * ffn_dim * sizeof(float));
    float *ff2 = (float *)malloc((size_t)n_tok * dim * sizeof(float));
    if (!ff || !ff2) { free(ff); free(ff2); return -1; }
    rt_detr__gemm_f32(ff, Wfc1, bfc1, x, n_tok, ffn_dim, dim);
    rt_detr__gelu_inplace(ff, (size_t)n_tok * ffn_dim);
    rt_detr__gemm_f32(ff2, Wfc2, bfc2, ff, n_tok, dim, ffn_dim);
    free(ff);

    /* x = LN(x + ff2) */
    {
        size_t n = (size_t)n_tok * dim;
        for (size_t i = 0; i < n; i++) x[i] += ff2[i];
    }
    free(ff2);
    snprintf(buf, sizeof(buf), "%s.final_layer_norm.weight", base);
    int iln2w = safetensors_find(m->st, buf);
    snprintf(buf, sizeof(buf), "%s.final_layer_norm.bias",   base);
    int iln2b = safetensors_find(m->st, buf);
    if (iln2w < 0 || iln2b < 0) return -1;
    const float *ln2w = (const float *)safetensors_data(m->st, iln2w);
    const float *ln2b = (const float *)safetensors_data(m->st, iln2b);
    rt_detr__layernorm(x, x, ln2w, ln2b, n_tok, dim, 1e-5f);
    return 0;
}

/* Project a backbone fmap through encoder_input_proj.{idx} (1×1 BN-folded
 * conv, NO activation — see RTDetrConvNormLayer in HF: activation=None for
 * encoder_input_proj per source). */
static float *rt_detr__input_proj(rt_detr_t *m, int idx,
                                  const float *in, int ih, int iw,
                                  int *out_c_p)
{
    char buf[64];
    snprintf(buf, sizeof(buf), "model.encoder_input_proj.%d", idx);
    const float *W; const float *b;
    int co, ci, kh, kw;
    if (rt_detr_lookup_bnfolded(m, buf, &W, &b, &co, &ci, &kh, &kw) != 0)
        return NULL;
    float *out = (float *)malloc((size_t)co * ih * iw * sizeof(float));
    if (!out) return NULL;
    rt_detr__conv2d_nchw(in, ci, ih, iw, W, b, co, kh, kw, 1, 0, 0, out);
    if (out_c_p) *out_c_p = co;
    return out;
}

/* Lateral / downsample / generic 1×1 or 3×3 BN-folded conv with silu. */
static float *rt_detr__bn_silu_conv(rt_detr_t *m, const char *base,
                                    const float *in, int ih, int iw,
                                    int stride, int pad, int *out_co_p,
                                    int *out_oh_p, int *out_ow_p)
{
    const float *W; const float *b;
    int co, ci, kh, kw;
    if (rt_detr_lookup_bnfolded(m, base, &W, &b, &co, &ci, &kh, &kw) != 0)
        return NULL;
    int oh = (ih + 2 * pad - kh) / stride + 1;
    int ow = (iw + 2 * pad - kw) / stride + 1;
    float *out = (float *)malloc((size_t)co * oh * ow * sizeof(float));
    if (!out) return NULL;
    rt_detr__conv2d_nchw(in, ci, ih, iw, W, b, co, kh, kw, stride, pad, 0, out);
    rt_detr__silu_inplace(out, (size_t)co * oh * ow);
    if (out_co_p) *out_co_p = co;
    if (out_oh_p) *out_oh_p = oh;
    if (out_ow_p) *out_ow_p = ow;
    return out;
}

int rt_detr_forward_encoder(rt_detr_t *m,
                            const float *bb_s3, const float *bb_s4,
                            const float *bb_s5,
                            float *out_s3, float *out_s4, float *out_s5)
{
    const int dim = RT_DETR_D_MODEL;
    const int n_heads = RT_DETR_NUM_HEADS;

    /* 1) input projections (1×1 BN-folded conv, no activation) */
    int s3_c, s4_c, s5_c;
    float *s3_proj = rt_detr__input_proj(m, 0, bb_s3, 80, 80, &s3_c);
    float *s4_proj = rt_detr__input_proj(m, 1, bb_s4, 40, 40, &s4_c);
    float *s5_proj = rt_detr__input_proj(m, 2, bb_s5, 20, 20, &s5_c);
    if (!s3_proj || !s4_proj || !s5_proj) {
        free(s3_proj); free(s4_proj); free(s5_proj); return -1;
    }

    /* 2) AIFI on S5: flatten (256,20,20) → (400,256), add pos, encoder, reshape back */
    {
        const int H = 20, W = 20, n_tok = H * W;
        /* Flatten s5_proj from CHW to row-major (n_tok, dim).
         * (c, y, x) → tok = y*W + x, ch = c → flat (tok, ch). */
        float *toks = (float *)malloc((size_t)n_tok * dim * sizeof(float));
        if (!toks) { free(s3_proj); free(s4_proj); free(s5_proj); return -1; }
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                int tok = y * W + x;
                for (int c = 0; c < dim; c++)
                    toks[(size_t)tok * dim + c] = s5_proj[(size_t)c * H * W + y * W + x];
            }
        }
        float *pos = (float *)malloc((size_t)n_tok * dim * sizeof(float));
        if (!pos) { free(toks); free(s3_proj); free(s4_proj); free(s5_proj); return -1; }
        rt_detr__sincos_pos_embed_2d(H, W, dim, 10000.0f, pos);

        if (rt_detr__encoder_layer(m, "model.encoder.encoder.0.layers.0",
                                   toks, pos, n_tok, dim, n_heads) != 0) {
            free(toks); free(pos); free(s3_proj); free(s4_proj); free(s5_proj);
            return -1;
        }
        free(pos);

        /* Reshape back (n_tok, dim) → (dim, H, W). */
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                int tok = y * W + x;
                for (int c = 0; c < dim; c++)
                    s5_proj[(size_t)c * H * W + y * W + x] = toks[(size_t)tok * dim + c];
            }
        }
        free(toks);
    }

    /* feature_maps[2] is now AIFI'd; let's call it s5_aifi (still in s5_proj). */

    /* 3) Top-down FPN. fpn_feature_maps starts as [s5_aifi]; iterate twice.
     *    idx=0: lateral_convs.0 on s5_aifi → upsample 2× → concat with s4_proj
     *           → fpn_blocks.0 → s4_fpn
     *    idx=1: lateral_convs.1 on s4_fpn → upsample 2× → concat with s3_proj
     *           → fpn_blocks.1 → s3_fpn
     *    After loop, the first (replaced) entry holds s5_lateral. */
    float *fpn_top = s5_proj;  /* current 'top' of the fpn stack */
    int top_h = 20, top_w = 20;

    /* idx=0 */
    int lat0_c, lat0_h, lat0_w;
    float *s5_lateral = rt_detr__bn_silu_conv(m, "model.encoder.lateral_convs.0",
                                              fpn_top, top_h, top_w, 1, 0,
                                              &lat0_c, &lat0_h, &lat0_w);
    if (!s5_lateral) { free(s3_proj); free(s4_proj); free(s5_proj); return -1; }
    /* upsample 2× → (256, 40, 40) */
    float *up = (float *)malloc((size_t)lat0_c * lat0_h * 2 * lat0_w * 2 * sizeof(float));
    if (!up) { free(s5_lateral); free(s3_proj); free(s4_proj); free(s5_proj); return -1; }
    rt_detr__nearest_upsample_2x(s5_lateral, lat0_c, lat0_h, lat0_w, up);
    /* concat with s4_proj (256, 40, 40) → (512, 40, 40) */
    int fused_c = lat0_c + s4_c;
    float *fused = (float *)malloc((size_t)fused_c * 40 * 40 * sizeof(float));
    if (!fused) { free(up); free(s5_lateral); free(s3_proj); free(s4_proj); free(s5_proj); return -1; }
    rt_detr__concat_channels(up, lat0_c, s4_proj, s4_c, 40, 40, fused);
    free(up);
    /* fpn_block 0: 512 → 256 */
    float *s4_fpn = (float *)malloc((size_t)dim * 40 * 40 * sizeof(float));
    if (!s4_fpn) { free(fused); free(s5_lateral); free(s3_proj); free(s4_proj); free(s5_proj); return -1; }
    int s4_fpn_c;
    if (rt_detr__csp_rep_layer(m, "model.encoder.fpn_blocks.0", fused, fused_c, 40, 40,
                               &s4_fpn_c, s4_fpn) != 0) {
        free(s4_fpn); free(fused); free(s5_lateral); free(s3_proj); free(s4_proj); free(s5_proj); return -1;
    }
    free(fused);

    /* idx=1: lateral_convs.1 on s4_fpn → upsample → concat s3_proj → fpn_blocks.1.
     * NOTE: in HF, fpn_feature_maps[-1] is OVERWRITTEN by lateral_conv(top),
     * so after the FPN loop the middle entry holds s4_lateral (not s4_fpn).
     * The PAN bottom-up consumes that s4_lateral as its idx=0 fpn slot. */
    int lat1_c, lat1_h, lat1_w;
    float *s4_lateral = rt_detr__bn_silu_conv(m, "model.encoder.lateral_convs.1",
                                              s4_fpn, 40, 40, 1, 0,
                                              &lat1_c, &lat1_h, &lat1_w);
    if (!s4_lateral) { free(s4_fpn); free(s5_lateral); free(s3_proj); free(s4_proj); free(s5_proj); return -1; }
    float *up2 = (float *)malloc((size_t)lat1_c * 80 * 80 * sizeof(float));
    if (!up2) { free(s4_lateral); free(s4_fpn); free(s5_lateral); free(s3_proj); free(s4_proj); free(s5_proj); return -1; }
    rt_detr__nearest_upsample_2x(s4_lateral, lat1_c, lat1_h, lat1_w, up2);
    int fused2_c = lat1_c + s3_c;
    float *fused2 = (float *)malloc((size_t)fused2_c * 80 * 80 * sizeof(float));
    if (!fused2) { free(up2); free(s4_lateral); free(s4_fpn); free(s5_lateral); free(s3_proj); free(s4_proj); free(s5_proj); return -1; }
    rt_detr__concat_channels(up2, lat1_c, s3_proj, s3_c, 80, 80, fused2);
    free(up2);
    float *s3_fpn = (float *)malloc((size_t)dim * 80 * 80 * sizeof(float));
    if (!s3_fpn) { free(fused2); free(s4_lateral); free(s4_fpn); free(s5_lateral); free(s3_proj); free(s4_proj); free(s5_proj); return -1; }
    int s3_fpn_c;
    if (rt_detr__csp_rep_layer(m, "model.encoder.fpn_blocks.1", fused2, fused2_c, 80, 80,
                               &s3_fpn_c, s3_fpn) != 0) {
        free(s3_fpn); free(fused2); free(s4_lateral); free(s4_fpn); free(s5_lateral); free(s3_proj); free(s4_proj); free(s5_proj); return -1;
    }
    free(fused2);
    free(s4_fpn);  /* superseded by s4_lateral in fpn_feature_maps */

    /* After top-down: fpn_feature_maps = [s5_lateral, s4_lateral, s3_fpn].
     * Reverse → [s3_fpn, s4_lateral, s5_lateral]. We don't need the originals
     * (s4_proj/s5_proj) anymore. */
    free(s4_proj);
    free(s5_proj);
    free(s3_proj);

    /* 4) Bottom-up PAN.
     *    pan_feature_maps starts as [s3_fpn].
     *    idx=0: downsample_convs.0 on s3_fpn (3×3 s2 silu) → concat with s4_lateral
     *           → pan_blocks.0 → s4_pan
     *    idx=1: downsample_convs.1 on s4_pan → concat with s5_lateral
     *           → pan_blocks.1 → s5_pan */
    int dn0_c, dn0_h, dn0_w;
    float *dn0 = rt_detr__bn_silu_conv(m, "model.encoder.downsample_convs.0",
                                       s3_fpn, 80, 80, 2, 1,
                                       &dn0_c, &dn0_h, &dn0_w);
    if (!dn0) { free(s3_fpn); free(s4_lateral); free(s5_lateral); return -1; }
    int fused3_c = dn0_c + lat1_c;
    float *fused3 = (float *)malloc((size_t)fused3_c * 40 * 40 * sizeof(float));
    if (!fused3) { free(dn0); free(s3_fpn); free(s4_lateral); free(s5_lateral); return -1; }
    rt_detr__concat_channels(dn0, dn0_c, s4_lateral, lat1_c, 40, 40, fused3);
    free(dn0);
    free(s4_lateral);
    float *s4_pan = (float *)malloc((size_t)dim * 40 * 40 * sizeof(float));
    if (!s4_pan) { free(fused3); free(s3_fpn); free(s5_lateral); return -1; }
    int s4_pan_c;
    if (rt_detr__csp_rep_layer(m, "model.encoder.pan_blocks.0", fused3, fused3_c, 40, 40,
                               &s4_pan_c, s4_pan) != 0) {
        free(s4_pan); free(fused3); free(s3_fpn); free(s5_lateral); return -1;
    }
    free(fused3);

    int dn1_c, dn1_h, dn1_w;
    float *dn1 = rt_detr__bn_silu_conv(m, "model.encoder.downsample_convs.1",
                                       s4_pan, 40, 40, 2, 1,
                                       &dn1_c, &dn1_h, &dn1_w);
    if (!dn1) { free(s4_pan); free(s3_fpn); free(s5_lateral); return -1; }
    int fused4_c = dn1_c + lat0_c;  /* s5_lateral has lat0_c (256) channels */
    float *fused4 = (float *)malloc((size_t)fused4_c * 20 * 20 * sizeof(float));
    if (!fused4) { free(dn1); free(s4_pan); free(s3_fpn); free(s5_lateral); return -1; }
    rt_detr__concat_channels(dn1, dn1_c, s5_lateral, lat0_c, 20, 20, fused4);
    free(dn1);
    float *s5_pan = (float *)malloc((size_t)dim * 20 * 20 * sizeof(float));
    if (!s5_pan) { free(fused4); free(s4_pan); free(s3_fpn); free(s5_lateral); return -1; }
    int s5_pan_c;
    if (rt_detr__csp_rep_layer(m, "model.encoder.pan_blocks.1", fused4, fused4_c, 20, 20,
                               &s5_pan_c, s5_pan) != 0) {
        free(s5_pan); free(fused4); free(s4_pan); free(s3_fpn); free(s5_lateral); return -1;
    }
    free(fused4);
    free(s5_lateral);

    /* Copy outputs */
    memcpy(out_s3, s3_fpn, (size_t)dim * 80 * 80 * sizeof(float));
    memcpy(out_s4, s4_pan, (size_t)dim * 40 * 40 * sizeof(float));
    memcpy(out_s5, s5_pan, (size_t)dim * 20 * 20 * sizeof(float));
    free(s3_fpn); free(s4_pan); free(s5_pan);
    (void)s4_fpn_c; (void)s3_fpn_c; (void)s4_pan_c; (void)s5_pan_c;
    (void)lat0_h; (void)lat0_w; (void)lat1_h; (void)lat1_w;
    (void)dn0_h; (void)dn0_w; (void)dn1_h; (void)dn1_w;
    return 0;
}

/* ============================== Decoder (A2.1.3) ========================== */

/* Generate the 8400 anchor points used by RT-DETR-S as initial reference
 * coordinates. Layout matches HF's RTDetrModel.generate_anchors:
 *   level 0: 80x80 grid, grid_size=0.05*1
 *   level 1: 40x40 grid, grid_size=0.05*2
 *   level 2: 20x20 grid, grid_size=0.05*4
 * For each grid point (y, x) at level L of size H×W:
 *   cx = (x + 0.5) / W
 *   cy = (y + 0.5) / H
 *   wh = 0.05 * 2^L
 *   raw = (cx, cy, wh, wh)  — in [0, 1]
 *   valid = all(raw > 1e-2 & raw < 1-1e-2)
 *   anchor = log(raw / (1-raw))  if valid, else FLT_MAX
 *
 * Outputs:
 *   anchors (8400, 4)  — log-odds form
 *   valid  (8400,)     — float 1.0 / 0.0 mask  */
static void rt_detr__generate_anchors_8400(float *anchors, float *valid_mask) {
    const int sizes[3] = {80, 40, 20};
    const float grid_step[3] = {0.05f, 0.10f, 0.20f};
    const float eps = 1e-2f;
    int off = 0;
    for (int L = 0; L < 3; L++) {
        int S = sizes[L];
        float wh = grid_step[L];
        for (int y = 0; y < S; y++) {
            for (int x = 0; x < S; x++) {
                float cx = ((float)x + 0.5f) / (float)S;
                float cy = ((float)y + 0.5f) / (float)S;
                float raw[4] = {cx, cy, wh, wh};
                int v = 1;
                for (int k = 0; k < 4; k++) {
                    if (!(raw[k] > eps && raw[k] < 1.0f - eps)) { v = 0; break; }
                }
                valid_mask[off] = v ? 1.0f : 0.0f;
                for (int k = 0; k < 4; k++) {
                    if (v) {
                        anchors[off * 4 + k] = logf(raw[k] / (1.0f - raw[k]));
                    } else {
                        anchors[off * 4 + k] = 3.4e38f; /* FLT_MAX-ish */
                    }
                }
                off++;
            }
        }
    }
    /* off should be 8400 */
}

static inline float rt_detr__sigmoidf(float x) {
    if (x >= 0.0f) {
        float e = expf(-x);
        return 1.0f / (1.0f + e);
    } else {
        float e = expf(x);
        return e / (1.0f + e);
    }
}

static inline float rt_detr__inverse_sigmoidf(float x) {
    /* HF clamps x in [0, 1] then x1=clamp(x, eps, 1), x2=clamp(1-x, eps, 1) */
    const float eps = 1e-5f;
    if (x < 0.0f) x = 0.0f;
    if (x > 1.0f) x = 1.0f;
    float x1 = x < eps ? eps : x;
    float x2 = (1.0f - x) < eps ? eps : (1.0f - x);
    return logf(x1 / x2);
}

/* Apply BN-folded 1×1 decoder_input_proj.{idx} (256→256 for RT-DETR-S).
 * Input/output: (256, H, W). Allocates and returns output. */
static float *rt_detr__decoder_input_proj_apply(rt_detr_t *m, int idx,
                                                const float *in, int h, int w,
                                                int *out_c_p)
{
    char buf[64];
    snprintf(buf, sizeof(buf), "model.decoder_input_proj.%d", idx);
    const float *W; const float *b;
    int co, ci, kh, kw;
    if (rt_detr_lookup_bnfolded(m, buf, &W, &b, &co, &ci, &kh, &kw) != 0)
        return NULL;
    float *out = (float *)malloc((size_t)co * h * w * sizeof(float));
    if (!out) return NULL;
    rt_detr__conv2d_nchw(in, ci, h, w, W, b, co, kh, kw, 1, 0, 0, out);
    if (out_c_p) *out_c_p = co;
    return out;
}

/* enc_output: Linear (256→256) + LayerNorm (256). Applied to memory of shape
 * (n_tok, 256). dst may alias src. */
static int rt_detr__enc_output(rt_detr_t *m, const float *src, float *dst,
                               int n_tok, int dim)
{
    const float *W, *b;
    int n_out, n_in;
    if (rt_detr__linear(m, "model.enc_output.0", &W, &b, &n_out, &n_in) != 0) return -1;

    float *tmp = (float *)malloc((size_t)n_tok * dim * sizeof(float));
    if (!tmp) return -1;
    rt_detr__gemm_f32(tmp, W, b, src, n_tok, dim, dim);

    /* LayerNorm: model.enc_output.1 (weight, bias) */
    int iln_w = safetensors_find(m->st, "model.enc_output.1.weight");
    int iln_b = safetensors_find(m->st, "model.enc_output.1.bias");
    if (iln_w < 0 || iln_b < 0) { free(tmp); return -1; }
    const float *lw = (const float *)safetensors_data(m->st, iln_w);
    const float *lb = (const float *)safetensors_data(m->st, iln_b);
    rt_detr__layernorm(dst, tmp, lw, lb, n_tok, dim, 1e-5f);
    free(tmp);
    return 0;
}

/* enc_score_head: Linear 256 → 80, applied to (n_tok, 256). */
static int rt_detr__enc_score_head(rt_detr_t *m, const float *src, float *dst,
                                   int n_tok, int dim, int n_classes)
{
    const float *W, *b;
    int n_out, n_in;
    if (rt_detr__linear(m, "model.enc_score_head", &W, &b, &n_out, &n_in) != 0)
        return -1;
    rt_detr__gemm_f32(dst, W, b, src, n_tok, n_classes, dim);
    return 0;
}

/* enc_bbox_head: 3-layer MLP 256→256→256→4 with ReLU between layers.
 * Applied to (n_tok, 256), writes (n_tok, 4) to dst. */
static int rt_detr__enc_bbox_head(rt_detr_t *m, const float *src, float *dst,
                                  int n_tok, int dim)
{
    const float *W0, *b0, *W1, *b1, *W2, *b2;
    int n_out, n_in;
    if (rt_detr__linear(m, "model.enc_bbox_head.layers.0", &W0, &b0, &n_out, &n_in) != 0) return -1;
    if (rt_detr__linear(m, "model.enc_bbox_head.layers.1", &W1, &b1, &n_out, &n_in) != 0) return -1;
    if (rt_detr__linear(m, "model.enc_bbox_head.layers.2", &W2, &b2, &n_out, &n_in) != 0) return -1;

    float *t0 = (float *)malloc((size_t)n_tok * dim * sizeof(float));
    float *t1 = (float *)malloc((size_t)n_tok * dim * sizeof(float));
    if (!t0 || !t1) { free(t0); free(t1); return -1; }
    rt_detr__gemm_f32(t0, W0, b0, src, n_tok, dim, dim);
    for (size_t i = 0; i < (size_t)n_tok * dim; i++) if (t0[i] < 0.0f) t0[i] = 0.0f;
    rt_detr__gemm_f32(t1, W1, b1, t0, n_tok, dim, dim);
    for (size_t i = 0; i < (size_t)n_tok * dim; i++) if (t1[i] < 0.0f) t1[i] = 0.0f;
    rt_detr__gemm_f32(dst, W2, b2, t1, n_tok, 4, dim);
    free(t0); free(t1);
    return 0;
}

/* class_embed[idx]: Linear 256 → 80. */
static int rt_detr__class_embed(rt_detr_t *m, int idx, const float *src,
                                float *dst, int n_tok, int dim, int n_classes)
{
    char buf[64];
    snprintf(buf, sizeof(buf), "model.decoder.class_embed.%d", idx);
    const float *W, *b;
    int n_out, n_in;
    if (rt_detr__linear(m, buf, &W, &b, &n_out, &n_in) != 0) return -1;
    rt_detr__gemm_f32(dst, W, b, src, n_tok, n_classes, dim);
    return 0;
}

/* bbox_embed[idx]: 3-layer MLP 256→256→256→4 with ReLU. */
static int rt_detr__bbox_embed(rt_detr_t *m, int idx, const float *src,
                               float *dst, int n_tok, int dim)
{
    char buf[64];
    const float *W0, *b0, *W1, *b1, *W2, *b2;
    int n_out, n_in;
    snprintf(buf, sizeof(buf), "model.decoder.bbox_embed.%d.layers.0", idx);
    if (rt_detr__linear(m, buf, &W0, &b0, &n_out, &n_in) != 0) return -1;
    snprintf(buf, sizeof(buf), "model.decoder.bbox_embed.%d.layers.1", idx);
    if (rt_detr__linear(m, buf, &W1, &b1, &n_out, &n_in) != 0) return -1;
    snprintf(buf, sizeof(buf), "model.decoder.bbox_embed.%d.layers.2", idx);
    if (rt_detr__linear(m, buf, &W2, &b2, &n_out, &n_in) != 0) return -1;

    float *t0 = (float *)malloc((size_t)n_tok * dim * sizeof(float));
    float *t1 = (float *)malloc((size_t)n_tok * dim * sizeof(float));
    if (!t0 || !t1) { free(t0); free(t1); return -1; }
    rt_detr__gemm_f32(t0, W0, b0, src, n_tok, dim, dim);
    for (size_t i = 0; i < (size_t)n_tok * dim; i++) if (t0[i] < 0.0f) t0[i] = 0.0f;
    rt_detr__gemm_f32(t1, W1, b1, t0, n_tok, dim, dim);
    for (size_t i = 0; i < (size_t)n_tok * dim; i++) if (t1[i] < 0.0f) t1[i] = 0.0f;
    rt_detr__gemm_f32(dst, W2, b2, t1, n_tok, 4, dim);
    free(t0); free(t1);
    return 0;
}

/* query_pos_head: 2-layer MLP 4→512→256, with ReLU between layers (no relu
 * after the last layer per RTDetrMLPPredictionHead). Applied to (n_tok, 4). */
static int rt_detr__query_pos_head(rt_detr_t *m, const float *src, float *dst,
                                   int n_tok, int in_dim, int hidden, int out_dim)
{
    const float *W0, *b0, *W1, *b1;
    int n_out, n_in;
    if (rt_detr__linear(m, "model.decoder.query_pos_head.layers.0",
                        &W0, &b0, &n_out, &n_in) != 0) return -1;
    if (rt_detr__linear(m, "model.decoder.query_pos_head.layers.1",
                        &W1, &b1, &n_out, &n_in) != 0) return -1;

    float *t0 = (float *)malloc((size_t)n_tok * hidden * sizeof(float));
    if (!t0) return -1;
    rt_detr__gemm_f32(t0, W0, b0, src, n_tok, hidden, in_dim);
    for (size_t i = 0; i < (size_t)n_tok * hidden; i++) if (t0[i] < 0.0f) t0[i] = 0.0f;
    rt_detr__gemm_f32(dst, W1, b1, t0, n_tok, out_dim, hidden);
    free(t0);
    return 0;
}

/* Decoder self-attention: 300 queries × 256 dim, 8 heads × 32 d.
 * Position embeddings are added to Q and K only (not V).
 * x is the pre-attention hidden state (300, 256).
 * pos is the query positional embedding (300, 256).
 * The output (300, 256) is `out_proj(MHA(...))`, written to attn_out. */
static int rt_detr__decoder_self_attn(rt_detr_t *m, int layer_idx,
                                      const float *x, const float *pos,
                                      int n_q, int dim, int n_heads,
                                      float *attn_out)
{
    const int head_dim = dim / n_heads;
    const float scale = 1.0f / sqrtf((float)head_dim);
    char buf[256];

    /* qk_in = x + pos */
    float *qk_in = (float *)malloc((size_t)n_q * dim * sizeof(float));
    if (!qk_in) return -1;
    for (size_t i = 0; i < (size_t)n_q * dim; i++) qk_in[i] = x[i] + pos[i];

    const float *Wq, *bq, *Wk, *bk, *Wv, *bv, *Wo, *bo;
    int n_out, n_in;
    snprintf(buf, sizeof(buf), "model.decoder.layers.%d.self_attn.q_proj", layer_idx);
    if (rt_detr__linear(m, buf, &Wq, &bq, &n_out, &n_in) != 0) { free(qk_in); return -1; }
    snprintf(buf, sizeof(buf), "model.decoder.layers.%d.self_attn.k_proj", layer_idx);
    if (rt_detr__linear(m, buf, &Wk, &bk, &n_out, &n_in) != 0) { free(qk_in); return -1; }
    snprintf(buf, sizeof(buf), "model.decoder.layers.%d.self_attn.v_proj", layer_idx);
    if (rt_detr__linear(m, buf, &Wv, &bv, &n_out, &n_in) != 0) { free(qk_in); return -1; }
    snprintf(buf, sizeof(buf), "model.decoder.layers.%d.self_attn.out_proj", layer_idx);
    if (rt_detr__linear(m, buf, &Wo, &bo, &n_out, &n_in) != 0) { free(qk_in); return -1; }

    float *Q = (float *)malloc((size_t)n_q * dim * sizeof(float));
    float *K = (float *)malloc((size_t)n_q * dim * sizeof(float));
    float *V = (float *)malloc((size_t)n_q * dim * sizeof(float));
    float *mha = (float *)malloc((size_t)n_q * dim * sizeof(float));
    if (!Q || !K || !V || !mha) {
        free(qk_in); free(Q); free(K); free(V); free(mha); return -1;
    }
    rt_detr__gemm_f32(Q, Wq, bq, qk_in, n_q, dim, dim);
    rt_detr__gemm_f32(K, Wk, bk, qk_in, n_q, dim, dim);
    rt_detr__gemm_f32(V, Wv, bv, x,     n_q, dim, dim);
    free(qk_in);

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int h = 0; h < n_heads; h++) {
        float *scores = (float *)malloc((size_t)n_q * sizeof(float));
        for (int t1 = 0; t1 < n_q; t1++) {
            const float *q = Q + (size_t)t1 * dim + h * head_dim;
            for (int t2 = 0; t2 < n_q; t2++) {
                const float *k = K + (size_t)t2 * dim + h * head_dim;
                float s = 0.0f;
                for (int d = 0; d < head_dim; d++) s += q[d] * k[d];
                scores[t2] = s * scale;
            }
            float mx = scores[0];
            for (int j = 1; j < n_q; j++) if (scores[j] > mx) mx = scores[j];
            float sum = 0.0f;
            for (int j = 0; j < n_q; j++) { scores[j] = expf(scores[j] - mx); sum += scores[j]; }
            float inv = 1.0f / sum;
            for (int j = 0; j < n_q; j++) scores[j] *= inv;
            float *y = mha + (size_t)t1 * dim + h * head_dim;
            for (int d = 0; d < head_dim; d++) y[d] = 0.0f;
            for (int t2 = 0; t2 < n_q; t2++) {
                const float *v = V + (size_t)t2 * dim + h * head_dim;
                float p = scores[t2];
                for (int d = 0; d < head_dim; d++) y[d] += p * v[d];
            }
        }
        free(scores);
    }
    free(Q); free(K); free(V);

    rt_detr__gemm_f32(attn_out, Wo, bo, mha, n_q, dim, dim);
    free(mha);
    return 0;
}

/* PyTorch grid_sample (mode=bilinear, padding=zeros, align_corners=False)
 * sampling on a (C, H, W) feature map at normalized location (gx, gy) ∈ [-1, 1].
 *
 * align_corners=False: pixel coord = (g + 1) / 2 * size - 0.5
 * Returns 0 if location maps to fully outside (zero-padded).
 *
 * out: float[C], accumulates into out (caller initializes to 0). */
static inline void rt_detr__grid_sample_bilinear(const float *fm, int C, int H, int W,
                                                 float gx, float gy, float *out)
{
    /* Map normalized [-1, 1] → pixel coords with align_corners=False */
    float ix = (gx + 1.0f) * 0.5f * (float)W - 0.5f;
    float iy = (gy + 1.0f) * 0.5f * (float)H - 0.5f;

    int ix0 = (int)floorf(ix);
    int iy0 = (int)floorf(iy);
    int ix1 = ix0 + 1;
    int iy1 = iy0 + 1;

    float dx = ix - (float)ix0;
    float dy = iy - (float)iy0;
    float w00 = (1.0f - dx) * (1.0f - dy);
    float w01 = dx          * (1.0f - dy);
    float w10 = (1.0f - dx) * dy;
    float w11 = dx          * dy;

    int v00 = (iy0 >= 0 && iy0 < H && ix0 >= 0 && ix0 < W);
    int v01 = (iy0 >= 0 && iy0 < H && ix1 >= 0 && ix1 < W);
    int v10 = (iy1 >= 0 && iy1 < H && ix0 >= 0 && ix0 < W);
    int v11 = (iy1 >= 0 && iy1 < H && ix1 >= 0 && ix1 < W);

    size_t HW = (size_t)H * W;
    for (int c = 0; c < C; c++) {
        float acc = 0.0f;
        const float *Cc = fm + (size_t)c * HW;
        if (v00) acc += w00 * Cc[iy0 * W + ix0];
        if (v01) acc += w01 * Cc[iy0 * W + ix1];
        if (v10) acc += w10 * Cc[iy1 * W + ix0];
        if (v11) acc += w11 * Cc[iy1 * W + ix1];
        out[c] += acc;
    }
}

/* Multi-scale deformable attention encoder_attn for one decoder layer.
 *
 *   hidden:                 (n_q, dim)  — pre-attention queries
 *   pos:                    (n_q, dim)  — query positional embedding
 *   ref_points:             (n_q, 4)    — refined points (cx, cy, w, h) in [0,1]
 *   value_levels[L]:        per-level encoder feature pointer (C=dim, H, W)
 *   level_shapes[L*2..L*2+1]: H, W per level
 *
 * Output (n_q, dim) written to dst. */
static int rt_detr__deformable_attn(rt_detr_t *m, int layer_idx,
                                    const float *hidden, const float *pos,
                                    const float *ref_points,
                                    const float * const *value_levels,
                                    const int *level_shapes,
                                    int n_q, int dim, int n_heads,
                                    int n_levels, int n_points,
                                    int total_seq_len,
                                    float *dst)
{
    const int head_dim = dim / n_heads;
    const int total_pts = n_levels * n_points;     /* 12 */
    const int s_offsets = n_heads * n_levels * n_points * 2;  /* 192 */
    const int s_weights = n_heads * n_levels * n_points;      /* 96 */
    char buf[256];

    /* hidden + pos as input to sampling_offsets / attention_weights */
    float *hp = (float *)malloc((size_t)n_q * dim * sizeof(float));
    if (!hp) return -1;
    for (size_t i = 0; i < (size_t)n_q * dim; i++) hp[i] = hidden[i] + pos[i];

    const float *Ws, *bs, *Ww, *bw, *Wv, *bv, *Wo, *bo;
    int n_out, n_in;
    snprintf(buf, sizeof(buf), "model.decoder.layers.%d.encoder_attn.sampling_offsets", layer_idx);
    if (rt_detr__linear(m, buf, &Ws, &bs, &n_out, &n_in) != 0) { free(hp); return -1; }
    snprintf(buf, sizeof(buf), "model.decoder.layers.%d.encoder_attn.attention_weights", layer_idx);
    if (rt_detr__linear(m, buf, &Ww, &bw, &n_out, &n_in) != 0) { free(hp); return -1; }
    snprintf(buf, sizeof(buf), "model.decoder.layers.%d.encoder_attn.value_proj", layer_idx);
    if (rt_detr__linear(m, buf, &Wv, &bv, &n_out, &n_in) != 0) { free(hp); return -1; }
    snprintf(buf, sizeof(buf), "model.decoder.layers.%d.encoder_attn.output_proj", layer_idx);
    if (rt_detr__linear(m, buf, &Wo, &bo, &n_out, &n_in) != 0) { free(hp); return -1; }

    /* sampling_offsets(hp): (n_q, n_heads*n_levels*n_points*2) */
    float *off = (float *)malloc((size_t)n_q * s_offsets * sizeof(float));
    /* attention_weights(hp): (n_q, n_heads*n_levels*n_points) */
    float *w_raw = (float *)malloc((size_t)n_q * s_weights * sizeof(float));
    /* value_proj over the full encoder sequence: (total_seq_len, dim) */
    float *value_flat = (float *)malloc((size_t)total_seq_len * dim * sizeof(float));
    if (!off || !w_raw || !value_flat) {
        free(off); free(w_raw); free(value_flat); free(hp); return -1;
    }
    rt_detr__gemm_f32(off,   Ws, bs, hp, n_q, s_offsets, dim);
    rt_detr__gemm_f32(w_raw, Ww, bw, hp, n_q, s_weights, dim);
    free(hp);

    /* Build value_proj on the encoder sequence. value_levels[L] is (dim, H, W);
     * we need to assemble a (total_seq_len, dim) tensor first (HF concats over
     * the spatial dim, level-by-level), then apply value_proj. */
    float *enc_seq = (float *)malloc((size_t)total_seq_len * dim * sizeof(float));
    if (!enc_seq) { free(off); free(w_raw); free(value_flat); return -1; }
    {
        size_t cursor = 0;
        for (int L = 0; L < n_levels; L++) {
            int H = level_shapes[L * 2 + 0];
            int W = level_shapes[L * 2 + 1];
            const float *fm = value_levels[L];
            /* fm is (dim, H, W). Convert to (H*W, dim). */
            for (int y = 0; y < H; y++) {
                for (int x = 0; x < W; x++) {
                    int tok = y * W + x;
                    for (int c = 0; c < dim; c++) {
                        enc_seq[(cursor + tok) * (size_t)dim + c]
                            = fm[(size_t)c * H * W + y * W + x];
                    }
                }
            }
            cursor += (size_t)H * W;
        }
    }
    rt_detr__gemm_f32(value_flat, Wv, bv, enc_seq, total_seq_len, dim, dim);
    free(enc_seq);

    /* Re-pack value_flat into per-level (n_heads, head_dim, H, W) so that
     * grid_sample sees CHW per (level, head). For each level L, allocate
     * (n_heads, head_dim, H, W) where channel index = h*head_dim + d. */
    float **value_per_level = (float **)malloc((size_t)n_levels * sizeof(float *));
    if (!value_per_level) {
        free(off); free(w_raw); free(value_flat); return -1;
    }
    {
        size_t cursor = 0;
        for (int L = 0; L < n_levels; L++) {
            int H = level_shapes[L * 2 + 0];
            int W = level_shapes[L * 2 + 1];
            float *vL = (float *)malloc((size_t)dim * H * W * sizeof(float));
            if (!vL) {
                for (int Lp = 0; Lp < L; Lp++) free(value_per_level[Lp]);
                free(value_per_level);
                free(off); free(w_raw); free(value_flat); return -1;
            }
            /* From (H*W, dim) → (dim, H, W). Channel-major: c*HW + y*W+x. */
            for (int y = 0; y < H; y++) {
                for (int x = 0; x < W; x++) {
                    int tok = y * W + x;
                    for (int c = 0; c < dim; c++) {
                        vL[(size_t)c * H * W + y * W + x]
                            = value_flat[(cursor + tok) * (size_t)dim + c];
                    }
                }
            }
            value_per_level[L] = vL;
            cursor += (size_t)H * W;
        }
    }
    free(value_flat);

    /* For each query: compute attention_weights softmax (over n_levels*n_points
     * for each head independently), compute sampling_locations from refpts +
     * offsets, sample values, weight, sum → (dim,). */
    float *mha_out = (float *)calloc((size_t)n_q * dim, sizeof(float));
    if (!mha_out) {
        for (int L = 0; L < n_levels; L++) free(value_per_level[L]);
        free(value_per_level);
        free(off); free(w_raw); return -1;
    }

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int q = 0; q < n_q; q++) {
        const float *roff = off  + (size_t)q * s_offsets;
        const float *rw   = w_raw + (size_t)q * s_weights;
        const float *rp   = ref_points + (size_t)q * 4;
        float ref_cx = rp[0], ref_cy = rp[1], ref_w = rp[2], ref_h = rp[3];

        /* Softmax attention_weights per head over (n_levels*n_points). */
        float *aw = (float *)malloc((size_t)n_heads * total_pts * sizeof(float));
        for (int h = 0; h < n_heads; h++) {
            const float *src = rw + h * total_pts;
            float *dstw = aw + h * total_pts;
            float mx = src[0];
            for (int j = 1; j < total_pts; j++) if (src[j] > mx) mx = src[j];
            float sum = 0.0f;
            for (int j = 0; j < total_pts; j++) {
                dstw[j] = expf(src[j] - mx);
                sum += dstw[j];
            }
            float inv = 1.0f / sum;
            for (int j = 0; j < total_pts; j++) dstw[j] *= inv;
        }

        float *qout = mha_out + (size_t)q * dim;

        for (int h = 0; h < n_heads; h++) {
            for (int L = 0; L < n_levels; L++) {
                int H = level_shapes[L * 2 + 0];
                int W = level_shapes[L * 2 + 1];
                int hd = head_dim;
                /* value_per_level[L]: (n_heads*head_dim, H, W) — sample at the
                 * head_dim slice corresponding to this head. */
                const float *vL_h = value_per_level[L]
                                  + (size_t)(h * hd) * H * W;

                for (int p = 0; p < n_points; p++) {
                    /* offset[(h, L, p, 0..1)] */
                    const float *o2 = roff +
                        (((size_t)h * n_levels + L) * n_points + p) * 2;
                    /* HF for 4D refpts:
                     * sampling_locations = ref[:2] + offsets / n_points * ref[2:] * 0.5 */
                    float sx = ref_cx + o2[0] / (float)n_points * ref_w * 0.5f;
                    float sy = ref_cy + o2[1] / (float)n_points * ref_h * 0.5f;
                    /* Convert [0,1] → [-1,1] */
                    float gx = 2.0f * sx - 1.0f;
                    float gy = 2.0f * sy - 1.0f;

                    float weight = aw[h * total_pts + L * n_points + p];
                    float *ohead = qout + h * hd;
                    /* Sample (head_dim,) from vL_h, accumulate weight*sample into ohead. */
                    /* Inline grid_sample but multiply by weight. */
                    float ix = (gx + 1.0f) * 0.5f * (float)W - 0.5f;
                    float iy = (gy + 1.0f) * 0.5f * (float)H - 0.5f;
                    int ix0 = (int)floorf(ix);
                    int iy0 = (int)floorf(iy);
                    int ix1 = ix0 + 1;
                    int iy1 = iy0 + 1;
                    float dx = ix - (float)ix0;
                    float dy = iy - (float)iy0;
                    float w00 = (1.0f - dx) * (1.0f - dy);
                    float w01 = dx          * (1.0f - dy);
                    float w10 = (1.0f - dx) * dy;
                    float w11 = dx          * dy;
                    int v00 = (iy0 >= 0 && iy0 < H && ix0 >= 0 && ix0 < W);
                    int v01 = (iy0 >= 0 && iy0 < H && ix1 >= 0 && ix1 < W);
                    int v10 = (iy1 >= 0 && iy1 < H && ix0 >= 0 && ix0 < W);
                    int v11 = (iy1 >= 0 && iy1 < H && ix1 >= 0 && ix1 < W);
                    size_t HW = (size_t)H * W;

                    for (int d = 0; d < hd; d++) {
                        const float *Cc = vL_h + (size_t)d * HW;
                        float acc = 0.0f;
                        if (v00) acc += w00 * Cc[iy0 * W + ix0];
                        if (v01) acc += w01 * Cc[iy0 * W + ix1];
                        if (v10) acc += w10 * Cc[iy1 * W + ix0];
                        if (v11) acc += w11 * Cc[iy1 * W + ix1];
                        ohead[d] += weight * acc;
                    }
                }
            }
        }
        free(aw);
    }

    for (int L = 0; L < n_levels; L++) free(value_per_level[L]);
    free(value_per_level);
    free(off); free(w_raw);

    /* output_proj */
    rt_detr__gemm_f32(dst, Wo, bo, mha_out, n_q, dim, dim);
    free(mha_out);
    return 0;
}

/* Decoder layer forward (post-LN, RT-DETR style):
 *   residual = x
 *   x        = self_attn(x, pos)
 *   x        = LN1(residual + x)
 *   residual = x
 *   x        = encoder_attn(x, pos, ref_pts, encoder_features)
 *   x        = LN2(residual + x)
 *   residual = x
 *   x        = mlp(x)
 *   x        = LN3(residual + x)
 *
 * x is updated in place. */
static int rt_detr__decoder_layer(rt_detr_t *m, int layer_idx,
                                  float *x, const float *pos,
                                  const float *ref_points,
                                  const float * const *value_levels,
                                  const int *level_shapes,
                                  int n_q, int dim, int n_heads,
                                  int n_levels, int n_points,
                                  int total_seq_len)
{
    char buf[256];
    /* === Self-attention === */
    float *sa_out = (float *)malloc((size_t)n_q * dim * sizeof(float));
    if (!sa_out) return -1;
    if (rt_detr__decoder_self_attn(m, layer_idx, x, pos, n_q, dim, n_heads,
                                   sa_out) != 0) { free(sa_out); return -1; }
    for (size_t i = 0; i < (size_t)n_q * dim; i++) x[i] += sa_out[i];
    free(sa_out);

    snprintf(buf, sizeof(buf), "model.decoder.layers.%d.self_attn_layer_norm.weight", layer_idx);
    int iln1w = safetensors_find(m->st, buf);
    snprintf(buf, sizeof(buf), "model.decoder.layers.%d.self_attn_layer_norm.bias",   layer_idx);
    int iln1b = safetensors_find(m->st, buf);
    if (iln1w < 0 || iln1b < 0) return -1;
    rt_detr__layernorm(x, x,
                       (const float *)safetensors_data(m->st, iln1w),
                       (const float *)safetensors_data(m->st, iln1b),
                       n_q, dim, 1e-5f);

    /* === Cross-attention (deformable) === */
    float *ca_out = (float *)malloc((size_t)n_q * dim * sizeof(float));
    if (!ca_out) return -1;
    if (rt_detr__deformable_attn(m, layer_idx, x, pos, ref_points,
                                 value_levels, level_shapes,
                                 n_q, dim, n_heads, n_levels, n_points,
                                 total_seq_len, ca_out) != 0) {
        free(ca_out); return -1;
    }
    for (size_t i = 0; i < (size_t)n_q * dim; i++) x[i] += ca_out[i];
    free(ca_out);

    snprintf(buf, sizeof(buf), "model.decoder.layers.%d.encoder_attn_layer_norm.weight", layer_idx);
    int iln2w = safetensors_find(m->st, buf);
    snprintf(buf, sizeof(buf), "model.decoder.layers.%d.encoder_attn_layer_norm.bias",   layer_idx);
    int iln2b = safetensors_find(m->st, buf);
    if (iln2w < 0 || iln2b < 0) return -1;
    rt_detr__layernorm(x, x,
                       (const float *)safetensors_data(m->st, iln2w),
                       (const float *)safetensors_data(m->st, iln2b),
                       n_q, dim, 1e-5f);

    /* === FFN: fc1 (256→1024) ReLU fc2 (1024→256)  (decoder_activation_function="relu") === */
    const float *Wfc1, *bfc1, *Wfc2, *bfc2;
    int n_out, n_in;
    snprintf(buf, sizeof(buf), "model.decoder.layers.%d.fc1", layer_idx);
    if (rt_detr__linear(m, buf, &Wfc1, &bfc1, &n_out, &n_in) != 0) return -1;
    snprintf(buf, sizeof(buf), "model.decoder.layers.%d.fc2", layer_idx);
    if (rt_detr__linear(m, buf, &Wfc2, &bfc2, &n_out, &n_in) != 0) return -1;

    float *ff = (float *)malloc((size_t)n_q * RT_DETR_FFN_DIM * sizeof(float));
    float *ff2 = (float *)malloc((size_t)n_q * dim * sizeof(float));
    if (!ff || !ff2) { free(ff); free(ff2); return -1; }
    rt_detr__gemm_f32(ff, Wfc1, bfc1, x, n_q, RT_DETR_FFN_DIM, dim);
    {
        size_t ffn_total = (size_t)n_q * RT_DETR_FFN_DIM;
        for (size_t i = 0; i < ffn_total; i++) if (ff[i] < 0.0f) ff[i] = 0.0f;
    }
    rt_detr__gemm_f32(ff2, Wfc2, bfc2, ff, n_q, dim, RT_DETR_FFN_DIM);
    free(ff);
    for (size_t i = 0; i < (size_t)n_q * dim; i++) x[i] += ff2[i];
    free(ff2);

    snprintf(buf, sizeof(buf), "model.decoder.layers.%d.final_layer_norm.weight", layer_idx);
    int iln3w = safetensors_find(m->st, buf);
    snprintf(buf, sizeof(buf), "model.decoder.layers.%d.final_layer_norm.bias",   layer_idx);
    int iln3b = safetensors_find(m->st, buf);
    if (iln3w < 0 || iln3b < 0) return -1;
    rt_detr__layernorm(x, x,
                       (const float *)safetensors_data(m->st, iln3w),
                       (const float *)safetensors_data(m->st, iln3b),
                       n_q, dim, 1e-5f);
    return 0;
}

/* Run the full decoder pipeline given encoder outputs (3 fmaps).
 *   enc_s3: (256, 80, 80)
 *   enc_s4: (256, 40, 40)
 *   enc_s5: (256, 20, 20)
 *   out_logits: (300, 80)
 *   out_boxes:  (300, 4)  cxcywh in [0,1] (post-sigmoid, post-refinement) */
int rt_detr_forward_decoder(rt_detr_t *m,
                            const float *enc_s3, const float *enc_s4,
                            const float *enc_s5,
                            float *out_logits, float *out_boxes)
{
    const int dim = RT_DETR_D_MODEL;
    const int n_heads = RT_DETR_NUM_HEADS;
    const int n_q = RT_DETR_NUM_QUERIES;
    const int n_classes = RT_DETR_NUM_CLASSES;
    const int n_levels = RT_DETR_NUM_FEATURE_LEVELS;
    const int n_points = RT_DETR_NUM_SAMPLING_POINTS;
    const int level_shapes[6] = {80, 80, 40, 40, 20, 20};
    const int total_seq = 80 * 80 + 40 * 40 + 20 * 20;  /* 8400 */

    /* 1) decoder_input_proj — 1×1 BN-folded conv, 256→256 per level. */
    int proj_c;
    float *p3 = rt_detr__decoder_input_proj_apply(m, 0, enc_s3, 80, 80, &proj_c);
    float *p4 = rt_detr__decoder_input_proj_apply(m, 1, enc_s4, 40, 40, &proj_c);
    float *p5 = rt_detr__decoder_input_proj_apply(m, 2, enc_s5, 20, 20, &proj_c);
    if (!p3 || !p4 || !p5) {
        free(p3); free(p4); free(p5); return -1;
    }

    /* 2) Build flat encoder sequence (n_tok, dim) by concatenating levels. */
    float *source_flatten = (float *)malloc((size_t)total_seq * dim * sizeof(float));
    if (!source_flatten) { free(p3); free(p4); free(p5); return -1; }
    {
        const float *fms[3] = {p3, p4, p5};
        size_t cursor = 0;
        for (int L = 0; L < n_levels; L++) {
            int H = level_shapes[L * 2 + 0];
            int W = level_shapes[L * 2 + 1];
            for (int y = 0; y < H; y++) {
                for (int x = 0; x < W; x++) {
                    int tok = y * W + x;
                    for (int c = 0; c < dim; c++) {
                        source_flatten[(cursor + tok) * (size_t)dim + c]
                            = fms[L][(size_t)c * H * W + y * W + x];
                    }
                }
            }
            cursor += (size_t)H * W;
        }
    }

    /* 3) Generate anchors + valid_mask. */
    float *anchors = (float *)malloc((size_t)total_seq * 4 * sizeof(float));
    float *valid   = (float *)malloc((size_t)total_seq * sizeof(float));
    if (!anchors || !valid) { free(anchors); free(valid); free(source_flatten);
        free(p3); free(p4); free(p5); return -1; }
    rt_detr__generate_anchors_8400(anchors, valid);

    /* 4) memory = source_flatten * valid_mask  (broadcast over dim) */
    float *memory = (float *)malloc((size_t)total_seq * dim * sizeof(float));
    if (!memory) { free(anchors); free(valid); free(source_flatten);
        free(p3); free(p4); free(p5); return -1; }
    for (int t = 0; t < total_seq; t++) {
        float v = valid[t];
        const float *src = source_flatten + (size_t)t * dim;
        float       *dst = memory         + (size_t)t * dim;
        for (int c = 0; c < dim; c++) dst[c] = src[c] * v;
    }

    /* 5) output_memory = enc_output(memory) (Linear+LN) */
    float *output_memory = (float *)malloc((size_t)total_seq * dim * sizeof(float));
    if (!output_memory) { free(memory); free(anchors); free(valid);
        free(source_flatten); free(p3); free(p4); free(p5); return -1; }
    if (rt_detr__enc_output(m, memory, output_memory, total_seq, dim) != 0) {
        free(output_memory); free(memory); free(anchors); free(valid);
        free(source_flatten); free(p3); free(p4); free(p5); return -1;
    }
    free(memory);

    /* 6) enc_outputs_class = enc_score_head(output_memory)  (8400, 80)
     *    enc_outputs_coord_logits = enc_bbox_head(output_memory) + anchors (8400, 4) */
    float *enc_class = (float *)malloc((size_t)total_seq * n_classes * sizeof(float));
    float *enc_coord = (float *)malloc((size_t)total_seq * 4 * sizeof(float));
    if (!enc_class || !enc_coord) { free(enc_class); free(enc_coord); free(output_memory);
        free(anchors); free(valid); free(source_flatten); free(p3); free(p4); free(p5); return -1; }
    if (rt_detr__enc_score_head(m, output_memory, enc_class, total_seq, dim, n_classes) != 0 ||
        rt_detr__enc_bbox_head(m, output_memory, enc_coord, total_seq, dim) != 0) {
        free(enc_class); free(enc_coord); free(output_memory); free(anchors); free(valid);
        free(source_flatten); free(p3); free(p4); free(p5); return -1;
    }
    for (int t = 0; t < total_seq; t++) {
        for (int k = 0; k < 4; k++) enc_coord[t * 4 + k] += anchors[t * 4 + k];
    }
    free(anchors); free(valid);

    /* 7) topk_ind = top-300 by max class score per token. */
    int *topk = (int *)malloc((size_t)n_q * sizeof(int));
    if (!topk) { free(enc_class); free(enc_coord); free(output_memory);
        free(source_flatten); free(p3); free(p4); free(p5); return -1; }
    {
        /* Build (max_score, idx) array, then partial-sort. Use a simple
         * O(N*K) selection — N=8400, K=300, = 2.5M comparisons, trivial. */
        float *maxs = (float *)malloc((size_t)total_seq * sizeof(float));
        for (int t = 0; t < total_seq; t++) {
            const float *r = enc_class + (size_t)t * n_classes;
            float mx = r[0];
            for (int c = 1; c < n_classes; c++) if (r[c] > mx) mx = r[c];
            maxs[t] = mx;
        }
        /* Max-K selection via repeated linear scan with masking. */
        for (int k = 0; k < n_q; k++) {
            float mx = -3.4e38f;
            int   mi = -1;
            for (int t = 0; t < total_seq; t++) {
                if (maxs[t] > mx) { mx = maxs[t]; mi = t; }
            }
            topk[k] = mi;
            maxs[mi] = -3.4e38f;
        }
        free(maxs);
    }

    /* 8) reference_points_unact = enc_coord[topk]  (300, 4)
     *    target = output_memory[topk]              (300, 256) */
    float *ref_unact = (float *)malloc((size_t)n_q * 4 * sizeof(float));
    float *target    = (float *)malloc((size_t)n_q * dim * sizeof(float));
    if (!ref_unact || !target) {
        free(ref_unact); free(target); free(topk); free(enc_class); free(enc_coord);
        free(output_memory); free(source_flatten); free(p3); free(p4); free(p5); return -1;
    }
    for (int q = 0; q < n_q; q++) {
        int idx = topk[q];
        for (int k = 0; k < 4; k++) ref_unact[q * 4 + k] = enc_coord[idx * 4 + k];
        memcpy(target + (size_t)q * dim, output_memory + (size_t)idx * dim,
               sizeof(float) * dim);
    }
    free(topk); free(enc_class); free(enc_coord); free(output_memory);

    /* 9) Initialize ref_points = sigmoid(ref_unact). */
    float *ref_points = (float *)malloc((size_t)n_q * 4 * sizeof(float));
    if (!ref_points) {
        free(ref_unact); free(target); free(source_flatten);
        free(p3); free(p4); free(p5); return -1;
    }
    for (int i = 0; i < n_q * 4; i++) ref_points[i] = rt_detr__sigmoidf(ref_unact[i]);
    free(ref_unact);

    /* 10) Decoder loop. Encoder values per level come from the input projections
     * p3/p4/p5 (256, H, W). Rebuild source_flatten was used only for memory; the
     * deformable attention re-projects per-level features through value_proj. */
    const float *value_levels[3] = {p3, p4, p5};
    float *hidden = target;  /* (n_q, dim), reuse buffer in-place */
    float *q_pos  = (float *)malloc((size_t)n_q * dim * sizeof(float));
    float *predicted_corners = (float *)malloc((size_t)n_q * 4 * sizeof(float));
    float *new_ref           = (float *)malloc((size_t)n_q * 4 * sizeof(float));
    float *layer_logits      = (float *)malloc((size_t)n_q * n_classes * sizeof(float));
    if (!q_pos || !predicted_corners || !new_ref || !layer_logits) {
        free(q_pos); free(predicted_corners); free(new_ref); free(layer_logits);
        free(hidden); free(ref_points); free(source_flatten);
        free(p3); free(p4); free(p5); return -1;
    }

    for (int li = 0; li < RT_DETR_NUM_DECODER_LAYERS; li++) {
        /* q_pos = query_pos_head(ref_points)  4 → 512 → 256 */
        if (rt_detr__query_pos_head(m, ref_points, q_pos, n_q, 4, 512, dim) != 0)
            goto fail;

        /* Decoder layer in place: hidden = layer(hidden, q_pos, ref_points). */
        if (rt_detr__decoder_layer(m, li, hidden, q_pos, ref_points,
                                   value_levels, level_shapes,
                                   n_q, dim, n_heads, n_levels, n_points,
                                   total_seq) != 0) goto fail;

        /* predicted_corners = bbox_embed[li](hidden) */
        if (rt_detr__bbox_embed(m, li, hidden, predicted_corners, n_q, dim) != 0) goto fail;
        /* new_ref = sigmoid(predicted_corners + inverse_sigmoid(ref_points)) */
        for (int i = 0; i < n_q * 4; i++) {
            new_ref[i] = rt_detr__sigmoidf(predicted_corners[i]
                                          + rt_detr__inverse_sigmoidf(ref_points[i]));
        }
        memcpy(ref_points, new_ref, sizeof(float) * n_q * 4);

        /* If this is the last layer, emit logits + final boxes. */
        if (li == RT_DETR_NUM_DECODER_LAYERS - 1) {
            if (rt_detr__class_embed(m, li, hidden, layer_logits, n_q, dim, n_classes) != 0)
                goto fail;
            memcpy(out_logits, layer_logits, sizeof(float) * n_q * n_classes);
            memcpy(out_boxes,  new_ref,      sizeof(float) * n_q * 4);
        }
    }

    free(q_pos); free(predicted_corners); free(new_ref); free(layer_logits);
    free(hidden); free(ref_points); free(source_flatten);
    free(p3); free(p4); free(p5);
    return 0;

fail:
    free(q_pos); free(predicted_corners); free(new_ref); free(layer_logits);
    free(hidden); free(ref_points); free(source_flatten);
    free(p3); free(p4); free(p5);
    return -1;
}

int rt_detr_forward(rt_detr_t *m, const float *input,
                    float *out_logits, float *out_boxes)
{
    const double t0 = rt_detr__time_ms();
    double t_backbone = 0.0, t_encoder = 0.0, t_decoder = 0.0;
    /* backbone */
    float *bb_s3 = (float *)malloc((size_t)128 * 80 * 80 * sizeof(float));
    float *bb_s4 = (float *)malloc((size_t)256 * 40 * 40 * sizeof(float));
    float *bb_s5 = (float *)malloc((size_t)512 * 20 * 20 * sizeof(float));
    if (!bb_s3 || !bb_s4 || !bb_s5) {
        free(bb_s3); free(bb_s4); free(bb_s5); return -1;
    }
    double ts = rt_detr__time_ms();
    if (rt_detr_forward_backbone(m, input, bb_s3, bb_s4, bb_s5) != 0) {
        free(bb_s3); free(bb_s4); free(bb_s5); return -1;
    }
    t_backbone = rt_detr__time_ms() - ts;
    /* encoder */
    float *enc_s3 = (float *)malloc((size_t)256 * 80 * 80 * sizeof(float));
    float *enc_s4 = (float *)malloc((size_t)256 * 40 * 40 * sizeof(float));
    float *enc_s5 = (float *)malloc((size_t)256 * 20 * 20 * sizeof(float));
    if (!enc_s3 || !enc_s4 || !enc_s5) {
        free(bb_s3); free(bb_s4); free(bb_s5);
        free(enc_s3); free(enc_s4); free(enc_s5); return -1;
    }
    ts = rt_detr__time_ms();
    if (rt_detr_forward_encoder(m, bb_s3, bb_s4, bb_s5, enc_s3, enc_s4, enc_s5) != 0) {
        free(bb_s3); free(bb_s4); free(bb_s5);
        free(enc_s3); free(enc_s4); free(enc_s5); return -1;
    }
    t_encoder = rt_detr__time_ms() - ts;
    free(bb_s3); free(bb_s4); free(bb_s5);

    /* decoder */
    ts = rt_detr__time_ms();
    int rc = rt_detr_forward_decoder(m, enc_s3, enc_s4, enc_s5, out_logits, out_boxes);
    t_decoder = rt_detr__time_ms() - ts;
    if (rt_detr__timing_enabled()) {
        fprintf(stderr,
                "[rt_detr][timing] forward total %.3f ms "
                "(backbone %.3f, encoder %.3f, decoder %.3f)\n",
                rt_detr__time_ms() - t0, t_backbone, t_encoder, t_decoder);
    }
    free(enc_s3); free(enc_s4); free(enc_s5);
    return rc;
}

typedef struct { float score; int q; int c; } rt_detr__cand_t;

static int rt_detr__cand_cmp_desc(const void *a, const void *b)
{
    const rt_detr__cand_t *aa = (const rt_detr__cand_t *)a;
    const rt_detr__cand_t *bb = (const rt_detr__cand_t *)b;
    if (aa->score < bb->score) return  1;
    if (aa->score > bb->score) return -1;
    return 0;
}

/* HF post_process_object_detection (use_focal_loss=True path):
 *   scores = sigmoid(logits)               (300, 80)
 *   flat   = scores.flatten()              (24000,)
 *   topk(flat, 300) → score, idx           idx in [0, 24000)
 *   label  = idx % 80; query = idx // 80
 *   box    = boxes[query] (cxcywh-norm) → xyxy image px
 * Then filter by score > thresh, optionally by class_id (>=0). */
int rt_detr_postprocess(const float *logits, const float *boxes_norm,
                        int orig_w, int orig_h,
                        int class_id, float score_thresh,
                        rt_detr_box_t *out, int max_out)
{
    if (!logits || !boxes_norm || !out || max_out <= 0) return -1;
    const int N = RT_DETR_NUM_QUERIES;     /* 300 */
    const int C = RT_DETR_NUM_CLASSES;     /* 80  */
    const int total = N * C;

    rt_detr__cand_t *cand = (rt_detr__cand_t *)malloc(sizeof(*cand) * (size_t)total);
    if (!cand) return -1;
    for (int q = 0; q < N; q++) {
        const float *r = logits + (size_t)q * C;
        for (int c = 0; c < C; c++) {
            float s = 1.0f / (1.0f + expf(-r[c]));
            cand[q * C + c].score = s;
            cand[q * C + c].q     = q;
            cand[q * C + c].c     = c;
        }
    }
    qsort(cand, (size_t)total, sizeof(*cand), rt_detr__cand_cmp_desc);

    int n_out = 0;
    for (int i = 0; i < N && n_out < max_out; i++) {
        if (cand[i].score <= score_thresh) break;
        if (class_id >= 0 && cand[i].c != class_id) continue;
        int q = cand[i].q;
        const float *bx = boxes_norm + (size_t)q * 4;
        float cx = bx[0], cy = bx[1], bw = bx[2], bh = bx[3];
        out[n_out].score    = cand[i].score;
        out[n_out].class_id = cand[i].c;
        out[n_out].x0 = (cx - bw * 0.5f) * (float)orig_w;
        out[n_out].y0 = (cy - bh * 0.5f) * (float)orig_h;
        out[n_out].x1 = (cx + bw * 0.5f) * (float)orig_w;
        out[n_out].y1 = (cy + bh * 0.5f) * (float)orig_h;
        n_out++;
    }
    free(cand);
    return n_out;
}

int rt_detr_detect_largest_person(rt_detr_t *m,
                                  const uint8_t *rgb, int w, int h,
                                  float score_thresh,
                                  rt_detr_box_t *out)
{
    if (!m || !rgb || !out) return -1;

    const double t0 = rt_detr__time_ms();
    double ts = rt_detr__time_ms();
    float *input = rt_detr_preprocess_image(rgb, w, h);
    if (!input) return -1;
    double t_pre = rt_detr__time_ms() - ts;

    float *logits = (float *)malloc((size_t)RT_DETR_NUM_QUERIES * RT_DETR_NUM_CLASSES * sizeof(float));
    float *boxes  = (float *)malloc((size_t)RT_DETR_NUM_QUERIES * 4 * sizeof(float));
    if (!logits || !boxes) { free(input); free(logits); free(boxes); return -1; }

    ts = rt_detr__time_ms();
    if (rt_detr_forward(m, input, logits, boxes) != 0) {
        free(input); free(logits); free(boxes); return -1;
    }
    double t_forward = rt_detr__time_ms() - ts;
    free(input);

    /* Largest-person fast path: no need to sort all 300*80 logits. */
    ts = rt_detr__time_ms();
    rt_detr_box_t best_box;
    int n = 0;
    float best_area = -1.0f;
    for (int q = 0; q < RT_DETR_NUM_QUERIES; q++) {
        float s = rt_detr__sigmoidf(logits[(size_t)q * RT_DETR_NUM_CLASSES +
                                           RT_DETR_PERSON_CLASS_ID]);
        if (s <= score_thresh) continue;
        const float *bx = boxes + (size_t)q * 4;
        float cx = bx[0], cy = bx[1], bw = bx[2], bh = bx[3];
        rt_detr_box_t cand;
        cand.score = s;
        cand.class_id = RT_DETR_PERSON_CLASS_ID;
        cand.x0 = (cx - bw * 0.5f) * (float)w;
        cand.y0 = (cy - bh * 0.5f) * (float)h;
        cand.x1 = (cx + bw * 0.5f) * (float)w;
        cand.y1 = (cy + bh * 0.5f) * (float)h;
        float area = (cand.x1 - cand.x0) * (cand.y1 - cand.y0);
        if (area > best_area) {
            best_area = area;
            best_box = cand;
        }
        n++;
    }
    double t_post = rt_detr__time_ms() - ts;
    free(logits); free(boxes);
    if (n <= 0) return -1;
    *out = best_box;
    if (rt_detr__timing_enabled()) {
        fprintf(stderr,
                "[rt_detr][timing] detect total %.3f ms "
                "(preprocess %.3f, forward %.3f, postprocess %.3f, candidates %d)\n",
                rt_detr__time_ms() - t0, t_pre, t_forward, t_post, n);
    }
    return 0;
}

#endif /* RT_DETR_IMPLEMENTATION */

#endif  /* GEMM_COMMON_RT_DETR_H */
