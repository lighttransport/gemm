/* CPU bring-up for one GLM5Next KDA layer.  This intentionally stops at the
 * recurrent-attention sublayer; it is used to validate real GGUF tensor
 * layouts before integrating the complete mHC + MoE block. */
#ifndef GLM5NEXT_CPU_KDA_H
#define GLM5NEXT_CPU_KDA_H

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "glm5next_ref.h"
#include "ggml_dequant.h"

static inline int glm5next_cpu_matvec(float *out, const glm5next_tensor_view *w,
                                      const float *x) {
    int rows, cols;
    size_t row_bytes;
    if (!out || !w || !x || w->n_dims < 2 || !w->data) return -1;
    cols = (int)w->dims[0];
    rows = (int)w->dims[1];
    row_bytes = dequant_row_size(w->type, cols);
    for (int r = 0; r < rows; ++r) {
        const uint8_t *row = (const uint8_t *)w->data + (size_t)r * row_bytes;
        float *tmp = (float *)malloc((size_t)cols * sizeof(float));
        double sum = 0.0;
        if (!tmp || dequant_row(w->type, row, tmp, cols) != 0) {
            free(tmp);
            return -1;
        }
        for (int j = 0; j < cols; ++j) sum += (double)tmp[j] * x[j];
        out[r] = (float)sum;
        free(tmp);
    }
    return 0;
}

static inline int glm5next_cpu_vector(const glm5next_tensor_view *w, float *out, int n) {
    if (!w || !out || !w->data || w->n_dims != 1 || (int)w->dims[0] != n) return -1;
    return dequant_row(w->type, w->data, out, n);
}

static inline void glm5next_cpu_rmsnorm(float *out, const float *x,
                                        const float *weight, int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; ++i) ss += (double)x[i] * x[i];
    float inv = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; ++i) out[i] = x[i] * inv * weight[i];
}

/* Execute one recurrent KDA layer from the real GGUF views.  State layouts
 * are contiguous by layer and owned by the caller.  This is deliberately a
 * reference implementation: each quantized matrix row is dequantized before
 * its dot product, which makes it slow but useful for graph bring-up. */
static inline int glm5next_cpu_kda_forward(const gguf_shards *model,
        int layer, const glm5next_config *c, const float *hidden, float *out,
        float *recurrent, float *conv_state) {
    char name[128];
    glm5next_tensor_view t;
    float *x = NULL, *q = NULL, *k = NULL, *v = NULL, *fa = NULL;
    float *gate = NULL, *beta = NULL, *ga = NULL, *gg = NULL, *a = NULL;
    float *decay = NULL, *core = NULL, *norm = NULL, *work = NULL;
    int h, d, z, qdim;
    int rc = -1;
#define GET(s, required) do { \
        snprintf(name, sizeof(name), "blk.%d.%s", layer, (s)); \
        if (glm5next_tensor_view_get(model, name, (required), &t) != 0) goto done; \
    } while (0)
#define MAT(dst, s) do { GET((s), 1); if (glm5next_cpu_matvec((dst), &t, x) != 0) goto done; } while (0)

    if (!model || !c || !hidden || !out || !recurrent || !conv_state ||
        layer < 0 || layer >= c->n_layers ||
        glm5next_layer_type(c, layer) != GLM5NEXT_LAYER_KDA) goto done;
    x = (float *)malloc((size_t)c->hidden_size * sizeof(float));
    qdim = c->attention_heads * c->linear_head_dim;
    q = (float *)malloc((size_t)qdim * sizeof(float));
    k = (float *)malloc((size_t)qdim * sizeof(float));
    v = (float *)malloc((size_t)qdim * sizeof(float));
    fa = (float *)malloc(128 * sizeof(float));
    gate = (float *)malloc((size_t)qdim * sizeof(float));
    beta = (float *)malloc((size_t)c->attention_heads * sizeof(float));
    ga = (float *)malloc(128 * sizeof(float));
    gg = (float *)malloc((size_t)qdim * sizeof(float));
    a = (float *)malloc((size_t)c->attention_heads * sizeof(float));
    decay = (float *)malloc((size_t)qdim * sizeof(float));
    core = (float *)malloc((size_t)qdim * sizeof(float));
    norm = (float *)malloc((size_t)c->hidden_size * sizeof(float));
    work = (float *)malloc((size_t)c->linear_head_dim * sizeof(float));
    if (!x || !q || !k || !v || !fa || !gate || !beta || !ga || !gg || !a ||
        !decay || !core || !norm || !work) goto done;

    GET("attn_norm.weight", 1);
    if (glm5next_cpu_vector(&t, norm, c->hidden_size) != 0) goto done;
    glm5next_cpu_rmsnorm(x, hidden, norm, c->hidden_size, c->norm_epsilon);
    MAT(q, "attn_q.weight");
    MAT(k, "attn_k.weight");
    MAT(v, "attn_v.weight");
    GET("ssm_conv1d_q.weight", 1);
    if (t.type != GGML_TYPE_F32 || t.n_dims != 3) goto done;
    {
        const float *w = (const float *)t.data;
        for (z = 0; z < 3; ++z) {
            float *y = z == 0 ? q : (z == 1 ? k : v);
            float *s = conv_state + (size_t)z * qdim * (c->short_conv_kernel - 1);
            for (d = 0; d < qdim; ++d) {
                float current = y[d];
                float value = 0.0f;
                for (h = 0; h < c->short_conv_kernel - 1; ++h)
                    value += s[(size_t)d * (c->short_conv_kernel - 1) + h] *
                             w[(size_t)d * c->short_conv_kernel + h];
                value += current * w[(size_t)d * c->short_conv_kernel +
                                      c->short_conv_kernel - 1];
                memmove(s + (size_t)d * (c->short_conv_kernel - 1),
                        s + (size_t)d * (c->short_conv_kernel - 1) + 1,
                        (size_t)(c->short_conv_kernel - 2) * sizeof(float));
                s[(size_t)d * (c->short_conv_kernel - 1) + c->short_conv_kernel - 2] = current;
                y[d] = value / (1.0f + expf(-value));
            }
        }
    }
    glm53f_l2norm(q, qdim, 1e-6f);
    glm53f_l2norm(k, qdim, 1e-6f);
    MAT(fa, "ssm_f_a.weight");
    GET("ssm_f_b.weight", 1);
    if (glm5next_cpu_matvec(gate, &t, fa) != 0) goto done;
    GET("ssm_dt.bias", 1);
    float *dt = (float *)malloc((size_t)qdim * sizeof(float));
    if (!dt || glm5next_cpu_vector(&t, dt, qdim) != 0) { free(dt); goto done; }
    MAT(beta, "ssm_beta.weight");
    MAT(ga, "ssm_g_a.weight");
    GET("ssm_g_b.weight", 1);
    if (glm5next_cpu_matvec(gg, &t, ga) != 0) { free(dt); goto done; }
    GET("ssm_a", 1);
    if (glm5next_cpu_vector(&t, a, c->attention_heads) != 0) { free(dt); goto done; }
    for (h = 0; h < c->attention_heads; ++h) {
        for (d = 0; d < c->linear_head_dim; ++d) {
            int j = h * c->linear_head_dim + d;
            decay[j] = c->kda_gate_lower_bound *
                (1.0f / (1.0f + expf(-expf(a[h]) * (gate[j] + dt[j]))));
        }
        beta[h] = 1.0f / (1.0f + expf(-beta[h]));
        glm5next_kda_step(recurrent + (size_t)h * c->linear_head_dim * c->linear_head_dim,
                          q + (size_t)h * c->linear_head_dim,
                          k + (size_t)h * c->linear_head_dim,
                          v + (size_t)h * c->linear_head_dim,
                          decay + (size_t)h * c->linear_head_dim, beta[h],
                          c->linear_head_dim, c->linear_head_dim,
                          core + (size_t)h * c->linear_head_dim, work);
    }
    GET("ssm_norm.weight", 1);
    if (glm5next_cpu_vector(&t, norm, c->linear_head_dim) != 0) { free(dt); goto done; }
    for (h = 0; h < c->attention_heads; ++h) {
        float ss = 0.0f;
        for (d = 0; d < c->linear_head_dim; ++d) {
            float value = core[h * c->linear_head_dim + d]; ss += value * value;
        }
        float inv = 1.0f / sqrtf(ss / c->linear_head_dim + 1e-6f);
        for (d = 0; d < c->linear_head_dim; ++d) {
            int j = h * c->linear_head_dim + d;
            float sig = 1.0f / (1.0f + expf(-gg[j]));
            core[j] *= inv * norm[d] * sig;
        }
    }
    MAT(out, "attn_output.weight");
    free(dt);
    rc = 0;
done:
    free(x); free(q); free(k); free(v); free(fa); free(gate); free(beta);
    free(ga); free(gg); free(a); free(decay); free(core); free(norm); free(work);
    return rc;
#undef MAT
#undef GET
}

#endif /* GLM5NEXT_CPU_KDA_H */
