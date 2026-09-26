#ifndef Q38D_KV_I8_H
#define Q38D_KV_I8_H

/*
 * Standalone INT8 K/V cache for Qwen3.8-27B (Q38D).
 *
 * The engine has 16 attention layers, four KV heads per attention layer, and
 * a 256 element head.  This file deliberately does not include the engine or
 * change its FP32 cache.  It is a small layout/codec/attention reference for
 * evaluating an INT8 cache before wiring one into the decode path.
 *
 * Source and output tensors use token-major layout:
 *   [token][attention-layer][kv-head][head-dimension]
 * K and V are separate arrays.  The cache itself is token-major and stores
 * K/V as adjacent streams:
 *   [token][layer][head][K|V][dimension]  (int8)
 *
 * Scales are per K/V stream.  TOKEN mode stores one scale for every
 * [token, layer, head, K|V].  BLOCK mode stores one scale for every
 * [token-block, layer, head, K|V]; a block must be packed as a whole so its
 * scale sees all values in the block.  Float scales are intentional here: the
 * header is a correctness and capacity primitive, and a later A64FX path can
 * choose a smaller scale type after measuring the error budget.
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    Q38D_KV_I8_PER_TOKEN = 0,
    Q38D_KV_I8_PER_BLOCK = 1
} q38d_kv_i8_scale_mode;

typedef struct {
    size_t attention_layers;
    size_t kv_heads;
    size_t head_dim;
} q38d_kv_i8_spec;

typedef struct {
    q38d_kv_i8_spec spec;
    size_t max_tokens;
    size_t block_tokens;
    q38d_kv_i8_scale_mode scale_mode;
    int8_t *data;
    float *scales;
    size_t data_bytes;
    size_t scale_bytes;
} q38d_kv_i8_cache;

static inline q38d_kv_i8_spec q38d_kv_i8_qwen38_spec(void) {
    q38d_kv_i8_spec s = {16, 4, 256};
    return s;
}

static inline int q38d_kv_i8_spec_valid(q38d_kv_i8_spec s) {
    return s.attention_layers > 0 && s.kv_heads > 0 && s.head_dim > 0;
}

static inline size_t q38d_kv_i8_streams(const q38d_kv_i8_spec *s) {
    return s->attention_layers * s->kv_heads * 2u;
}

static inline size_t q38d_kv_i8_plane_elems(const q38d_kv_i8_spec *s) {
    return s->attention_layers * s->kv_heads * s->head_dim;
}

static inline size_t q38d_kv_i8_token_data_bytes(const q38d_kv_i8_spec *s) {
    return q38d_kv_i8_streams(s) * s->head_dim;
}

static inline size_t q38d_kv_i8_scale_slots(const q38d_kv_i8_spec *s,
                                             size_t tokens,
                                             q38d_kv_i8_scale_mode mode,
                                             size_t block_tokens) {
    if (!q38d_kv_i8_spec_valid(*s) || block_tokens == 0 ||
        (mode != Q38D_KV_I8_PER_TOKEN && mode != Q38D_KV_I8_PER_BLOCK)) return 0;
    if (mode == Q38D_KV_I8_PER_TOKEN) return tokens;
    if (tokens > SIZE_MAX - (block_tokens - 1u)) return 0;
    return (tokens + block_tokens - 1u) / block_tokens;
}

static inline size_t q38d_kv_i8_data_bytes(const q38d_kv_i8_spec *s,
                                           size_t tokens) {
    size_t per = q38d_kv_i8_token_data_bytes(s);
    if (!q38d_kv_i8_spec_valid(*s) || per == 0 || tokens > SIZE_MAX / per) return 0;
    return tokens * per;
}

static inline size_t q38d_kv_i8_scale_bytes(const q38d_kv_i8_spec *s,
                                             size_t tokens,
                                             q38d_kv_i8_scale_mode mode,
                                             size_t block_tokens) {
    size_t slots = q38d_kv_i8_scale_slots(s, tokens, mode, block_tokens);
    size_t streams = q38d_kv_i8_streams(s);
    if (streams == 0 || (slots && slots > SIZE_MAX / streams)) return 0;
    size_t n = slots * streams;
    if (n > SIZE_MAX / sizeof(float)) return 0;
    return n * sizeof(float);
}

static inline size_t q38d_kv_i8_cache_bytes(const q38d_kv_i8_spec *s,
                                             size_t tokens,
                                             q38d_kv_i8_scale_mode mode,
                                             size_t block_tokens) {
    size_t data = q38d_kv_i8_data_bytes(s, tokens);
    size_t scale = q38d_kv_i8_scale_bytes(s, tokens, mode, block_tokens);
    if (!data || !scale || data > SIZE_MAX - scale) return 0;
    return data + scale;
}

static inline size_t q38d_kv_i8_data_offset(const q38d_kv_i8_cache *c,
                                            size_t token, size_t layer,
                                            size_t head, size_t stream) {
    size_t unit = c->spec.head_dim;
    return ((((token * c->spec.attention_layers + layer) * c->spec.kv_heads + head) * 2u + stream) * unit);
}

static inline size_t q38d_kv_i8_scale_offset(const q38d_kv_i8_cache *c,
                                             size_t token, size_t layer,
                                             size_t head, size_t stream) {
    size_t slot = c->scale_mode == Q38D_KV_I8_PER_TOKEN
                    ? token : token / c->block_tokens;
    return (((slot * c->spec.attention_layers + layer) * c->spec.kv_heads + head) * 2u + stream);
}

static inline int q38d_kv_i8_init(q38d_kv_i8_cache *c,
                                  q38d_kv_i8_spec spec,
                                  size_t max_tokens,
                                  q38d_kv_i8_scale_mode mode,
                                  size_t block_tokens) {
    if (!c || !q38d_kv_i8_spec_valid(spec) || max_tokens == 0 ||
        (mode != Q38D_KV_I8_PER_TOKEN && mode != Q38D_KV_I8_PER_BLOCK) ||
        (mode == Q38D_KV_I8_PER_BLOCK && block_tokens == 0)) return -1;
    size_t db = q38d_kv_i8_data_bytes(&spec, max_tokens);
    size_t sb = q38d_kv_i8_scale_bytes(&spec, max_tokens, mode, block_tokens);
    if (!db || !sb) return -1;
    int8_t *data = (int8_t *)malloc(db);
    float *scales = (float *)malloc(sb);
    if (!data || !scales) {
        free(data);
        free(scales);
        return -1;
    }
    memset(c, 0, sizeof(*c));
    c->spec = spec;
    c->max_tokens = max_tokens;
    c->block_tokens = mode == Q38D_KV_I8_PER_TOKEN ? 1 : block_tokens;
    c->scale_mode = mode;
    c->data = data;
    c->scales = scales;
    c->data_bytes = db;
    c->scale_bytes = sb;
    return 0;
}

static inline void q38d_kv_i8_destroy(q38d_kv_i8_cache *c) {
    if (!c) return;
    free(c->data);
    free(c->scales);
    memset(c, 0, sizeof(*c));
}

static inline int q38d_kv_i8_valid_range(const q38d_kv_i8_cache *c,
                                         size_t first_token, size_t count) {
    return c && c->data && c->scales && first_token <= c->max_tokens &&
           count <= c->max_tokens - first_token;
}

static inline int q38d_kv_i8_quantize_row(const float *src, int8_t *dst,
                                          size_t n, float *scale_out) {
    float amax = 0.0f;
    for (size_t d = 0; d < n; d++) {
        float a = fabsf(src[d]);
        if (!isfinite(a)) return -1;
        if (a > amax) amax = a;
    }
    float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    float inv = 1.0f / scale;
    for (size_t d = 0; d < n; d++) {
        int q = (int)lrintf(src[d] * inv);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        dst[d] = (int8_t)q;
    }
    *scale_out = scale;
    return 0;
}

/* Pack token-major K/V rows. In BLOCK mode, first_token must start on a block
 * boundary and count must end on a block boundary unless it reaches max_tokens.
 * This prevents a later partial call from silently changing a scale already
 * used by earlier rows. */
static inline int q38d_kv_i8_pack_tokens(q38d_kv_i8_cache *c,
                                         size_t first_token, size_t count,
                                         const float *k, const float *v) {
    if (!q38d_kv_i8_valid_range(c, first_token, count) || !k || !v || count == 0) return -1;
    size_t end = first_token + count;
    if (c->scale_mode == Q38D_KV_I8_PER_BLOCK &&
        (first_token % c->block_tokens != 0 ||
         (end < c->max_tokens && end % c->block_tokens != 0))) return -2;
    size_t plane = q38d_kv_i8_plane_elems(&c->spec);
    size_t hd = c->spec.head_dim;
    size_t block = c->scale_mode == Q38D_KV_I8_PER_TOKEN ? 1 : c->block_tokens;
    for (size_t b = first_token; b < end; b += block) {
        size_t bn = end - b < block ? end - b : block;
        for (size_t layer = 0; layer < c->spec.attention_layers; layer++) {
            for (size_t head = 0; head < c->spec.kv_heads; head++) {
                for (size_t stream = 0; stream < 2; stream++) {
                    float amax = 0.0f;
                    for (size_t t = 0; t < bn; t++) {
                        const float *src = (stream == 0 ? k : v) +
                            (b + t) * plane + (layer * c->spec.kv_heads + head) * hd;
                        for (size_t d = 0; d < hd; d++) {
                            float a = fabsf(src[d]);
                            if (!isfinite(a)) return -3;
                            if (a > amax) amax = a;
                        }
                    }
                    float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
                    size_t so = q38d_kv_i8_scale_offset(c, b, layer, head, stream);
                    c->scales[so] = scale;
                    float inv = 1.0f / scale;
                    for (size_t t = 0; t < bn; t++) {
                        const float *src = (stream == 0 ? k : v) +
                            (b + t) * plane + (layer * c->spec.kv_heads + head) * hd;
                        int8_t *dst = c->data + q38d_kv_i8_data_offset(c, b + t, layer, head, stream);
                        for (size_t d = 0; d < hd; d++) {
                            int q = (int)lrintf(src[d] * inv);
                            if (q > 127) q = 127;
                            if (q < -127) q = -127;
                            dst[d] = (int8_t)q;
                        }
                    }
                }
            }
        }
    }
    return 0;
}

static inline int q38d_kv_i8_pack_token(q38d_kv_i8_cache *c, size_t token,
                                        const float *k, const float *v) {
    if (!c || c->scale_mode != Q38D_KV_I8_PER_TOKEN) return -1;
    return q38d_kv_i8_pack_tokens(c, token, 1, k, v);
}

static inline int q38d_kv_i8_unpack_token(const q38d_kv_i8_cache *c,
                                          size_t token, float *k, float *v) {
    if (!c || !c->data || !c->scales || token >= c->max_tokens || !k || !v) return -1;
    size_t hd = c->spec.head_dim;
    for (size_t layer = 0; layer < c->spec.attention_layers; layer++) {
        for (size_t head = 0; head < c->spec.kv_heads; head++) {
            float sk = c->scales[q38d_kv_i8_scale_offset(c, token, layer, head, 0)];
            float sv = c->scales[q38d_kv_i8_scale_offset(c, token, layer, head, 1)];
            const int8_t *qk = c->data + q38d_kv_i8_data_offset(c, token, layer, head, 0);
            const int8_t *qv = c->data + q38d_kv_i8_data_offset(c, token, layer, head, 1);
            float *dk = k + (layer * c->spec.kv_heads + head) * hd;
            float *dv = v + (layer * c->spec.kv_heads + head) * hd;
            for (size_t d = 0; d < hd; d++) {
                dk[d] = (float)qk[d] * sk;
                dv[d] = (float)qv[d] * sv;
            }
        }
    }
    return 0;
}

/* Indexed single-head attention. `scores` is caller-owned scratch of nidx
 * floats and receives the unnormalized logits. The output is the weighted V
 * vector. K/V are dequantized on demand, so this is also a direct bandwidth
 * model for a sparse indexed-attention read. */
static inline int q38d_kv_i8_attention_indexed(const q38d_kv_i8_cache *c,
                                               size_t layer, size_t head,
                                               const float *q,
                                               const int32_t *indices,
                                               size_t nidx,
                                               float attn_scale,
                                               float *scores,
                                               float *out) {
    if (!c || !c->data || !c->scales || !q || !indices || !scores || !out ||
        layer >= c->spec.attention_layers || head >= c->spec.kv_heads || nidx == 0) return -1;
    size_t hd = c->spec.head_dim;
    float mx = -INFINITY;
    for (size_t i = 0; i < nidx; i++) {
        int32_t ti = indices[i];
        if (ti < 0 || (size_t)ti >= c->max_tokens) return -2;
        size_t token = (size_t)ti;
        size_t so = q38d_kv_i8_scale_offset(c, token, layer, head, 0);
        const int8_t *qk = c->data + q38d_kv_i8_data_offset(c, token, layer, head, 0);
        float dot = 0.0f;
        for (size_t d = 0; d < hd; d++) dot += q[d] * ((float)qk[d] * c->scales[so]);
        scores[i] = dot * attn_scale;
        if (scores[i] > mx) mx = scores[i];
    }
    for (size_t d = 0; d < hd; d++) out[d] = 0.0f;
    float den = 0.0f;
    for (size_t i = 0; i < nidx; i++) den += expf(scores[i] - mx);
    if (!(den > 0.0f) || !isfinite(den)) return -3;
    size_t plane = q38d_kv_i8_plane_elems(&c->spec);
    (void)plane;
    for (size_t i = 0; i < nidx; i++) {
        size_t token = (size_t)indices[i];
        size_t so = q38d_kv_i8_scale_offset(c, token, layer, head, 1);
        const int8_t *qv = c->data + q38d_kv_i8_data_offset(c, token, layer, head, 1);
        float w = expf(scores[i] - mx) / den;
        for (size_t d = 0; d < hd; d++) out[d] += w * ((float)qv[d] * c->scales[so]);
    }
    return 0;
}

#endif /* Q38D_KV_I8_H */
