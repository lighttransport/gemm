/* Laguna S 2.1 INT4 shared scalar/SVE-safe primitives. */
#ifndef LAGUNA_S21_H
#define LAGUNA_S21_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>

enum {
    LAGUNA_LAYERS = 48, LAGUNA_HIDDEN = 3072, LAGUNA_VOCAB = 100352,
    LAGUNA_KV_HEADS = 8, LAGUNA_HEAD_DIM = 128, LAGUNA_EXPERTS = 256,
    LAGUNA_ACTIVE = 10, LAGUNA_EXPERT_INTER = 1024, LAGUNA_GROUP = 32,
    LAGUNA_SLIDING_WINDOW = 512,
    LAGUNA_DENSE_INTER = 12288, LAGUNA_SHARED_INTER = 1024,
    LAGUNA_FULL_HEADS = 48, LAGUNA_SLIDING_HEADS = 72,
    LAGUNA_ROPE_FULL_DIM = 64, LAGUNA_ROPE_SLIDING_DIM = 128,
    LAGUNA_MAX_HEADS = 72
};
#define LAGUNA_RMS_EPS      1e-6f
#define LAGUNA_ROUTED_SCALE 2.5f
/* YaRN (full_attention) rope parameters, from config.json rope_parameters.full_attention */
#define LAGUNA_YARN_THETA   500000.0
#define LAGUNA_YARN_FACTOR  32.0
#define LAGUNA_YARN_BETA_FAST 32.0
#define LAGUNA_YARN_BETA_SLOW 1.0
#define LAGUNA_YARN_ORIG_MAX  8192.0
#define LAGUNA_YARN_ATTN_FACTOR 1.3465735902799727
/* default rope (sliding_attention) */
#define LAGUNA_SWA_THETA    10000.0

static inline float laguna_bf16_to_f32(uint16_t x) {
    uint32_t u = (uint32_t)x << 16; float f; memcpy(&f, &u, sizeof f); return f;
}
static inline uint16_t laguna_f32_to_bf16(float f) {
    uint32_t u; memcpy(&u, &f, sizeof u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

/* compressed-tensors pack-quantized symmetric INT4: eight low-to-high nibbles
 * per little-endian I32.  Each output row has one BF16 scale per 32 input values. */
static inline int laguna_i4_at(const uint32_t *packed, int col) {
    uint32_t w = packed[(unsigned)col >> 3];
    return (int)((w >> (4 * (col & 7))) & 15u) - 8;
}
static inline float laguna_i4g32_dot(const uint32_t *packed, const uint16_t *scales,
                                     const float *x, int cols) {
    float sum = 0.0f;
    for (int g = 0; g < cols / LAGUNA_GROUP; ++g) {
        float s = laguna_bf16_to_f32(scales[g]);
        float part = 0.0f;
        for (int j = 0; j < LAGUNA_GROUP; ++j)
            part += (float)laguna_i4_at(packed, g * LAGUNA_GROUP + j) * x[g * LAGUNA_GROUP + j];
        sum += s * part;
    }
    return sum;
}

/* Normalized Walsh-Hadamard transform over a 128-wide block. The checkpoint's
 * transform config declares head_dim=128, so dimensions are transformed blockwise. */
static inline void laguna_fht128(float *v) {
    for (int step = 1; step < 128; step <<= 1)
        for (int i = 0; i < 128; i += 2 * step)
            for (int j = 0; j < step; ++j) {
                float a = v[i + j], b = v[i + j + step];
                v[i + j] = a + b; v[i + j + step] = a - b;
            }
    const float scale = 0.08838834764831843f; /* 1/sqrt(128) */
    for (int i = 0; i < 128; ++i) v[i] *= scale;
}
static inline void laguna_fht_blocks(float *v, int n) {
    for (int i = 0; i < n; i += 128) laguna_fht128(v + i);
}

typedef struct { float score; int id; } laguna_route;
static inline void laguna_top10(const float *logits, const float *bias, int out_id[LAGUNA_ACTIVE], float out_w[LAGUNA_ACTIVE]) {
    laguna_route best[LAGUNA_ACTIVE];
    for (int i = 0; i < LAGUNA_ACTIVE; ++i) best[i] = (laguna_route){-INFINITY, -1};
    for (int e = 0; e < LAGUNA_EXPERTS; ++e) {
        float p = 1.0f / (1.0f + expf(-logits[e]));
        float s = p + bias[e];
        for (int k = 0; k < LAGUNA_ACTIVE; ++k) if (s > best[k].score || (s == best[k].score && e < best[k].id)) {
            for (int q = LAGUNA_ACTIVE - 1; q > k; --q) best[q] = best[q - 1];
            best[k] = (laguna_route){s, e}; break;
        }
    }
    float z = 0.0f;
    for (int k = 0; k < LAGUNA_ACTIVE; ++k) { out_id[k] = best[k].id; out_w[k] = 1.0f / (1.0f + expf(-logits[best[k].id])); z += out_w[k]; }
    for (int k = 0; k < LAGUNA_ACTIVE; ++k) out_w[k] /= z;
}

static inline int laguna_expert_owner(int expert, int ep_size) { return expert % ep_size; }

/* ===================================================================== *
 *  Forward-pass primitives (correctness-first; f32 accumulation).       *
 * ===================================================================== */

/* y[rows] = W[rows,cols] (bf16, row-major) * x[cols] (f32). */
static inline void laguna_matvec_bf16(float *restrict y, const uint16_t *restrict W,
                                      const float *restrict x, int rows, int cols) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < rows; ++r) {
        const uint16_t *w = W + (size_t)r * cols;
        float acc = 0.0f;
        for (int c = 0; c < cols; ++c) acc += laguna_bf16_to_f32(w[c]) * x[c];
        y[r] = acc;
    }
}

/* RMSNorm in-place style: out[n] = (x/rms(x)) * w (bf16 weight). */
static inline void laguna_rmsnorm(float *restrict out, const float *restrict x,
                                  const uint16_t *restrict w, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; ++i) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; ++i) out[i] = (x[i] * inv) * laguna_bf16_to_f32(w[i]);
}

static inline float laguna_silu(float x) { return x / (1.0f + expf(-x)); }
static inline float laguna_softplus(float x) {
    /* numerically stable log(1+e^x) */
    return x > 20.0f ? x : log1pf(expf(x));
}

/* GPT-NeoX ("half") rotary applied to the first rotary_dim entries of a head
 * vector.  cos/sin hold rotary_dim/2 entries each (freqs, not duplicated). */
static inline void laguna_rope_half(float *restrict v, const float *restrict cosp,
                                    const float *restrict sinp, int rotary_dim) {
    int half = rotary_dim / 2;
    for (int i = 0; i < half; ++i) {
        float a = v[i], b = v[i + half];
        v[i]        = a * cosp[i] - b * sinp[i];
        v[i + half] = b * cosp[i] + a * sinp[i];
    }
}

/* Build YaRN inv_freq[dim/2] for the full-attention rope (matches transformers
 * _compute_yarn_parameters with truncate=True). */
static inline void laguna_yarn_inv_freq(float *inv_freq, int dim) {
    int half = dim / 2;
    double lo_d = (dim * log(LAGUNA_YARN_ORIG_MAX / (LAGUNA_YARN_BETA_FAST * 2.0 * M_PI)))
                  / (2.0 * log(LAGUNA_YARN_THETA));
    double hi_d = (dim * log(LAGUNA_YARN_ORIG_MAX / (LAGUNA_YARN_BETA_SLOW * 2.0 * M_PI)))
                  / (2.0 * log(LAGUNA_YARN_THETA));
    double low = floor(lo_d); if (low < 0) low = 0;
    double high = ceil(hi_d); if (high > dim - 1) high = dim - 1;
    if (low == high) high += 0.001;
    for (int i = 0; i < half; ++i) {
        double pos_freq = pow(LAGUNA_YARN_THETA, (double)(2 * i) / (double)dim);
        double extrap = 1.0 / pos_freq;                       /* extrapolation inv_freq */
        double interp = 1.0 / (LAGUNA_YARN_FACTOR * pos_freq);/* interpolation inv_freq */
        double ramp = ((double)i - low) / (high - low);       /* linear ramp over dim/2 */
        if (ramp < 0) ramp = 0; if (ramp > 1) ramp = 1;
        double extrap_factor = 1.0 - ramp;                    /* 1 - linear_ramp */
        inv_freq[i] = (float)(interp * (1.0 - extrap_factor) + extrap * extrap_factor);
    }
}

/* Default rope inv_freq[dim/2] = 1 / theta^(2i/dim). */
static inline void laguna_default_inv_freq(float *inv_freq, int dim, double theta) {
    int half = dim / 2;
    for (int i = 0; i < half; ++i)
        inv_freq[i] = (float)(1.0 / pow(theta, (double)(2 * i) / (double)dim));
}

/* INT4 group-32 symmetric matvec: y[rows] = sum_c (nibble(packed)-8)*scale * x[c].
 * packed is row-major uint32 (cols/8 per row); scales bf16 (cols/32 per row). */
static inline void laguna_matvec_i4g32(float *restrict y, const uint32_t *restrict packed,
                                       const uint16_t *restrict scales,
                                       const float *restrict x, int rows, int cols) {
    int ppr = cols / 8, spr = cols / LAGUNA_GROUP;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int r = 0; r < rows; ++r)
        y[r] = laguna_i4g32_dot(packed + (size_t)r * ppr, scales + (size_t)r * spr, x, cols);
}

/* ===================== per-rank model ===================== */
typedef struct {
    const uint32_t *gp, *up, *dp;   /* weight_packed (uint32) */
    const uint16_t *gs, *us, *ds;   /* weight_scale  (bf16)   */
    int present;                    /* owned by this rank      */
} laguna_expert;

typedef struct {
    int num_heads;         /* per-layer query heads (48 full / 72 sliding) */
    int is_sliding;        /* 1 => sliding_attention, 0 => full_attention  */
    int is_moe;            /* 1 => MoE mlp, 0 => dense mlp (layer 0)        */
    /* attention (bf16) */
    const uint16_t *q_proj, *k_proj, *v_proj, *o_proj, *g_proj;
    const uint16_t *q_norm, *k_norm, *in_ln, *post_ln;
    /* dense mlp (layer 0) */
    const uint16_t *dense_gate, *dense_up, *dense_down;
    /* moe: shared expert (bf16), router weight (bf16 [256,hidden]), bias (f32 [256]) */
    const uint16_t *shared_gate, *shared_up, *shared_down;
    const uint16_t *router_w;
    const float    *router_bias;
    laguna_expert   experts[LAGUNA_EXPERTS];
} laguna_layer;

typedef struct {
    int n_layers, max_pos, ep_rank, ep_size;
    const uint16_t *embed;      /* bf16 [vocab,hidden] */
    const uint16_t *lm_head;    /* bf16 [vocab,hidden] */
    const uint16_t *final_norm; /* bf16 [hidden]       */
    laguna_layer layers[LAGUNA_LAYERS];
    /* rope tables: cos/sin[pos * half] for full (half=32) and sliding (half=64) */
    float *full_cos, *full_sin;       /* [max_pos * 32] */
    float *swa_cos, *swa_sin;         /* [max_pos * 64] */
    /* KV cache (bf16), replicated per rank: [layer][pos][KV_HEADS*HEAD_DIM] */
    uint16_t *kcache, *vcache;
    size_t kv_layer_stride;           /* per-layer element stride = max_pos*KV_HEADS*HEAD_DIM */
} laguna_model;

static inline void laguna_build_rope_tables(laguna_model *m) {
    float inv_full[LAGUNA_ROPE_FULL_DIM / 2];
    float inv_swa[LAGUNA_ROPE_SLIDING_DIM / 2];
    laguna_yarn_inv_freq(inv_full, LAGUNA_ROPE_FULL_DIM);
    laguna_default_inv_freq(inv_swa, LAGUNA_ROPE_SLIDING_DIM, LAGUNA_SWA_THETA);
    int hf = LAGUNA_ROPE_FULL_DIM / 2, hs = LAGUNA_ROPE_SLIDING_DIM / 2;
    for (int p = 0; p < m->max_pos; ++p) {
        for (int i = 0; i < hf; ++i) {
            double ang = (double)p * inv_full[i];
            m->full_cos[(size_t)p * hf + i] = (float)(cos(ang) * LAGUNA_YARN_ATTN_FACTOR);
            m->full_sin[(size_t)p * hf + i] = (float)(sin(ang) * LAGUNA_YARN_ATTN_FACTOR);
        }
        for (int i = 0; i < hs; ++i) {
            double ang = (double)p * inv_swa[i];
            m->swa_cos[(size_t)p * hs + i] = (float)cos(ang);
            m->swa_sin[(size_t)p * hs + i] = (float)sin(ang);
        }
    }
}
#endif
