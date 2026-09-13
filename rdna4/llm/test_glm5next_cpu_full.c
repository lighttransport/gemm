/* Full single-token CPU GLM5Next smoke path.  DSA history is intentionally
 * empty here; this test validates embedding, every block family, recurrent
 * state transitions, final norm, and the quantized output head. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GGUF_LOADER_IMPLEMENTATION
#include "../../common/gguf_loader.h"
#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"
#include "../../common/glm5next_cpu_kda.h"
#include "../../common/glm5next_cpu_runtime.h"

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]); return 2; }
    gguf_shards *model = gguf_open_shards(argv[1], 2);
    if (!model) return 1;
    glm5next_config c; char error[160];
    if (glm5next_config_load(model->metadata, &c, error, sizeof(error)) != 0 ||
        glm5next_validate_tensors(model, &c, error, sizeof(error)) != 0) {
        fprintf(stderr, "contract failure: %s\n", error); gguf_close_shards(model); return 1;
    }
    int h = c.hidden_size, hc = c.hc_count, d = c.linear_head_dim;
    size_t recurrent_n = (size_t)c.n_layers * c.attention_heads * d * d;
    size_t conv_n = (size_t)c.n_layers * 3 * c.attention_heads * d * (c.short_conv_kernel - 1);
    float *streams = (float *)malloc((size_t)hc * h * sizeof(float));
    float *recurrent = (float *)calloc(recurrent_n, sizeof(float));
    float *conv = (float *)calloc(conv_n, sizeof(float));
    float *hidden = (float *)malloc((size_t)h * sizeof(float));
    float *normed = (float *)malloc((size_t)h * sizeof(float));
    float *logits = (float *)malloc((size_t)c.vocab_size * sizeof(float));
    if (!streams || !recurrent || !conv || !hidden || !normed || !logits) return 1;
    glm5next_tensor_view t;
    if (glm5next_tensor_view_get(model, "token_embd.weight", 1, &t) != 0 ||
        t.n_dims != 2 || t.dims[0] != (uint64_t)h) return 1;
    size_t row_bytes = dequant_row_size(t.type, h);
    if (dequant_row(t.type, (const unsigned char *)t.data + row_bytes * 1, hidden, h) != 0) return 1;
    for (int s = 0; s < hc; ++s) memcpy(streams + (size_t)s * h, hidden, (size_t)h * sizeof(float));
    for (int l = 0; l < c.n_layers; ++l) {
        int rc;
        float *rs = recurrent + (size_t)l * c.attention_heads * d * d;
        float *cs = conv + (size_t)l * 3 * c.attention_heads * d * (c.short_conv_kernel - 1);
        if (glm5next_layer_type(&c, l) == GLM5NEXT_LAYER_KDA)
            rc = l < c.first_k_dense_replace
                ? glm5next_cpu_kda_dense_block(model, l, &c, streams, rs, cs)
                : glm5next_cpu_kda_moe_block(model, l, &c, streams, rs, cs);
        else
            rc = glm5next_cpu_dsa_moe_block(model, l, &c, streams);
        if (rc != 0) { fprintf(stderr, "layer %d failed\n", l); return 1; }
        fprintf(stderr, "layer %d/%d %s ok\n", l + 1, c.n_layers,
                glm5next_layer_type(&c, l) == GLM5NEXT_LAYER_KDA ? "KDA" : "DSA");
    }
    for (int i = 0; i < h; ++i) {
        double sum = 0.0;
        for (int s = 0; s < hc; ++s) sum += streams[(size_t)s * h + i];
        hidden[i] = (float)(sum / hc);
    }
    if (glm5next_tensor_view_get(model, "output_norm.weight", 1, &t) != 0 ||
        glm5next_cpu_vector(&t, normed, h) != 0) return 1;
    glm5next_cpu_rmsnorm(hidden, hidden, normed, h, c.norm_epsilon);
    if (glm5next_tensor_view_get(model, "output.weight", 1, &t) != 0 ||
        glm5next_cpu_matvec(logits, &t, hidden) != 0) return 1;
    int best = 0; for (int i = 1; i < c.vocab_size; ++i) if (logits[i] > logits[best]) best = i;
    double ss = 0.0; for (int i = 0; i < h; ++i) ss += (double)hidden[i] * hidden[i];
    printf("GLM5NEXT_CPU_FULL PASS token=1 argmax=%d hidden_rms=%.7g logit=%.7g\n",
           best, sqrt(ss / h), logits[best]);
    free(streams); free(recurrent); free(conv); free(hidden); free(normed); free(logits);
    glm5next_config_free(&c); gguf_close_shards(model); return 0;
}
