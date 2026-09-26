#define _POSIX_C_SOURCE 200809L

#include "q38d_kv_i8.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

enum { TEST_TOKENS = 97, TEST_BLOCK = 16, TP_NODES = 12 };

static float test_value(size_t i, int lane) {
    /* Bounded, nonuniform values with a few deliberately different channels.
     * This is closer to normalized K/V latents than a unit impulse and keeps
     * the rowwise int8 error visible in the attention comparison. */
    double x = (double)(i * 17u + (size_t)lane * 13u);
    float v = (float)(0.75 * sin(x * 0.017) + 0.25 * cos(x * 0.0031));
    if ((i + (size_t)lane) % 113u == 0) v *= 1.75f;
    return v;
}

static void fill_kv(float *k, float *v, size_t tokens,
                    q38d_kv_i8_spec spec) {
    size_t plane = q38d_kv_i8_plane_elems(&spec);
    for (size_t t = 0; t < tokens; t++) {
        for (size_t i = 0; i < plane; i++) {
            k[t * plane + i] = test_value(t * plane + i, 0);
            v[t * plane + i] = test_value(t * plane + i + 7u, 1) * 0.8f;
        }
    }
}

static float max_unpack_error(const q38d_kv_i8_cache *c,
                              const float *k, const float *v,
                              size_t tokens, float *uk, float *uv) {
    size_t plane = q38d_kv_i8_plane_elems(&c->spec);
    float worst = 0.0f;
    for (size_t t = 0; t < tokens; t++) {
        assert(q38d_kv_i8_unpack_token(c, t, uk, uv) == 0);
        for (size_t i = 0; i < plane; i++) {
            float ek = fabsf(uk[i] - k[t * plane + i]);
            float ev = fabsf(uv[i] - v[t * plane + i]);
            if (ek > worst) worst = ek;
            if (ev > worst) worst = ev;
        }
    }
    return worst;
}

static int ref_attention(const float *k, const float *v,
                         q38d_kv_i8_spec spec, size_t layer, size_t head,
                         const float *q, const int32_t *indices, size_t nidx,
                         float attn_scale, float *scores, float *out) {
    size_t plane = q38d_kv_i8_plane_elems(&spec), hd = spec.head_dim;
    float mx = -INFINITY;
    for (size_t i = 0; i < nidx; i++) {
        if (indices[i] < 0 || (size_t)indices[i] >= TEST_TOKENS) return -1;
        const float *kr = k + (size_t)indices[i] * plane +
                            (layer * spec.kv_heads + head) * hd;
        float dot = 0.0f;
        for (size_t d = 0; d < hd; d++) dot += q[d] * kr[d];
        scores[i] = dot * attn_scale;
        if (scores[i] > mx) mx = scores[i];
    }
    float den = 0.0f;
    for (size_t i = 0; i < nidx; i++) den += expf(scores[i] - mx);
    for (size_t d = 0; d < hd; d++) out[d] = 0.0f;
    for (size_t i = 0; i < nidx; i++) {
        const float *vr = v + (size_t)indices[i] * plane +
                            (layer * spec.kv_heads + head) * hd;
        float w = expf(scores[i] - mx) / den;
        for (size_t d = 0; d < hd; d++) out[d] += w * vr[d];
    }
    return 0;
}

static float max_diff(const float *a, const float *b, size_t n) {
    float worst = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float e = fabsf(a[i] - b[i]);
        if (e > worst) worst = e;
    }
    return worst;
}

static void test_cache_mode(q38d_kv_i8_scale_mode mode, size_t block_tokens,
                            const float *k, const float *v,
                            q38d_kv_i8_spec spec) {
    q38d_kv_i8_cache c = {0};
    assert(q38d_kv_i8_init(&c, spec, TEST_TOKENS, mode, block_tokens) == 0);
    assert(q38d_kv_i8_pack_tokens(&c, 0, TEST_TOKENS, k, v) == 0);
    size_t plane = q38d_kv_i8_plane_elems(&spec);
    float *uk = malloc(plane * sizeof(float));
    float *uv = malloc(plane * sizeof(float));
    assert(uk && uv);
    float unpack_err = max_unpack_error(&c, k, v, TEST_TOKENS, uk, uv);
    float unpack_bound = mode == Q38D_KV_I8_PER_TOKEN ? 0.025f : 0.08f;
    assert(unpack_err < unpack_bound);

    const int32_t indices[] = {0, 3, 7, 16, 17, 31, 32, 47, 63, 95, 96};
    const size_t nidx = sizeof(indices) / sizeof(indices[0]);
    float q[256], ref_scores[sizeof(indices) / sizeof(indices[0])];
    float got_scores[sizeof(indices) / sizeof(indices[0])];
    float ref[256], got[256];
    for (size_t d = 0; d < spec.head_dim; d++) q[d] = test_value(d + 991, 3);
    float attn_scale = 1.0f / sqrtf((float)spec.head_dim);
    assert(ref_attention(k, v, spec, 9, 2, q, indices, nidx,
                         attn_scale, ref_scores, ref) == 0);
    assert(q38d_kv_i8_attention_indexed(&c, 9, 2, q, indices, nidx,
                                        attn_scale, got_scores, got) == 0);
    float attn_err = max_diff(ref, got, spec.head_dim);
    float attn_bound = mode == Q38D_KV_I8_PER_TOKEN ? 0.012f : 0.04f;
    assert(attn_err < attn_bound);
    printf("mode=%s block=%zu data=%zu scale=%zu unpack_max=%.6g attn_max=%.6g\n",
           mode == Q38D_KV_I8_PER_TOKEN ? "token" : "block",
           mode == Q38D_KV_I8_PER_TOKEN ? 1u : block_tokens,
           c.data_bytes, c.scale_bytes, unpack_err, attn_err);
    free(uk);
    free(uv);
    q38d_kv_i8_destroy(&c);
}

static void test_capacity(q38d_kv_i8_spec spec) {
    const size_t aggregate = 8u * 1024u * 1024u;
    const size_t contexts[] = {8, 16, 32};
    const size_t lengths[] = {1024u * 1024u, 512u * 1024u, 256u * 1024u};
    size_t token_total = q38d_kv_i8_cache_bytes(&spec, aggregate,
                                                 Q38D_KV_I8_PER_TOKEN, 1);
    size_t block_total = q38d_kv_i8_cache_bytes(&spec, aggregate,
                                                 Q38D_KV_I8_PER_BLOCK, 32);
    assert(token_total && block_total);
    size_t token_scale_per = q38d_kv_i8_scale_bytes(&spec, 1,
                                                     Q38D_KV_I8_PER_TOKEN, 1);
    size_t block_scale_per = q38d_kv_i8_scale_bytes(&spec, 32,
                                                     Q38D_KV_I8_PER_BLOCK, 32) / 32;
    assert(token_scale_per == 512 && block_scale_per == 16);
    printf("Q38D INT8 KV: data/token=%zu, token-scale=%zu, block32-scale=%zu bytes/token\n",
           q38d_kv_i8_token_data_bytes(&spec),
           token_scale_per, block_scale_per);
    printf("aggregate=%zu tokens token-scale=%zu bytes (%.3f GiB), block32=%zu bytes (%.3f GiB), ideal/12=%.3f GiB\n",
           aggregate, token_total, (double)token_total / (double)(1ull << 30),
           block_total, (double)block_total / (double)(1ull << 30),
           (double)block_total / (double)TP_NODES / (double)(1ull << 30));
    for (size_t i = 0; i < 3; i++) {
        size_t per = q38d_kv_i8_cache_bytes(&spec, lengths[i],
                                             Q38D_KV_I8_PER_BLOCK, 32);
        assert(per && per * contexts[i] == block_total);
        printf("%zux%zu: %zu bytes/context (%.3f GiB), aggregate=%zu bytes, ideal/12=%.3f GiB\n",
               contexts[i], lengths[i], per, (double)per / (double)(1ull << 30),
               per * contexts[i], (double)per * (double)contexts[i] /
                   (double)TP_NODES / (double)(1ull << 30));
    }
    assert(q38d_kv_i8_cache_bytes(&spec, aggregate,
                                  Q38D_KV_I8_PER_TOKEN, 1) == 279172874240ull);
    assert(q38d_kv_i8_cache_bytes(&spec, aggregate,
                                  Q38D_KV_I8_PER_BLOCK, 32) == 275012124672ull);
}

int main(void) {
    q38d_kv_i8_spec spec = q38d_kv_i8_qwen38_spec();
    assert(spec.attention_layers == 16 && spec.kv_heads == 4 && spec.head_dim == 256);
    size_t plane = q38d_kv_i8_plane_elems(&spec);
    float *k = malloc((size_t)TEST_TOKENS * plane * sizeof(float));
    float *v = malloc((size_t)TEST_TOKENS * plane * sizeof(float));
    assert(k && v);
    fill_kv(k, v, TEST_TOKENS, spec);
    test_cache_mode(Q38D_KV_I8_PER_TOKEN, 1, k, v, spec);
    test_cache_mode(Q38D_KV_I8_PER_BLOCK, TEST_BLOCK, k, v, spec);

    q38d_kv_i8_cache block = {0};
    assert(q38d_kv_i8_init(&block, spec, TEST_TOKENS, Q38D_KV_I8_PER_BLOCK, TEST_BLOCK) == 0);
    assert(q38d_kv_i8_pack_tokens(&block, 1, 1, k + plane, v + plane) == -2);
    q38d_kv_i8_destroy(&block);
    test_capacity(spec);
    free(k);
    free(v);
    puts("PASS: Q38D INT8 KV pack/unpack, indexed attention, and 8M capacity arithmetic");
    return 0;
}
