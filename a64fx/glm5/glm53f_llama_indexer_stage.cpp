#include "ggml.h"
#include "ggml-cpu.h"
#include "../../common/glm53f_stage_artifact.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

enum { DIM = 16, HEADS = 3, TOKENS = 4, KV = 11 };

static void emit(glm53f_stage_artifact * a, const char * name,
                 const float * data, size_t count, const size_t * shape, size_t rank) {
    if (a->manifest && glm53f_stage_write_f32(a, "dsa_indexer", name, data, count, shape, rank)) {
        std::fprintf(stderr, "INDEXER_STAGE artifact failure: %s\n", name);
        std::exit(1);
    }
}

int main() {
    ggml_init_params ip = { 64ull * 1024ull * 1024ull, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return 1;
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DIM, HEADS, TOKENS, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DIM, 1, KV, 1);
    ggml_tensor * w = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HEADS, TOKENS, 1, 1);
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, KV, TOKENS, 1, 1);
    for (int t = 0; t < TOKENS; ++t) {
        for (int h = 0; h < HEADS; ++h)
            ((float *) w->data)[(size_t)t * HEADS + h] = 0.25f + 0.07f * (float)((t + h) % 5);
        for (int d = 0; d < DIM; ++d)
            ((float *) q->data)[((size_t)t * HEADS) * DIM + d] = 0.01f * (float)((d * 3 + t * 5) % 17 - 8);
        for (int h = 1; h < HEADS; ++h)
            for (int d = 0; d < DIM; ++d)
                ((float *) q->data)[((size_t)t * HEADS + h) * DIM + d] =
                    0.01f * (float)((d * (h + 2) + t * 7 + h) % 19 - 9);
        for (int ik = 0; ik < KV; ++ik) {
            for (int d = 0; d < DIM; ++d)
                ((float *) k->data)[(size_t)ik * DIM + d] =
                    0.015f * (float)((d * 5 + ik * 3) % 23 - 11);
            ((ggml_fp16_t *) m->data)[(size_t)t * KV + ik] =
                ggml_fp16_to_fp32(ggml_fp32_to_fp16((ik == KV - 1 && t == 0) ? -0.5f : 0.01f * (float)(ik - t)));
        }
    }
    ggml_tensor * out = ggml_lightning_indexer(ctx, q, k, w, m);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    if (ggml_graph_compute_with_ctx(ctx, graph, 48) != GGML_STATUS_SUCCESS) return 1;

    std::vector<float> custom((size_t)TOKENS * KV);
    for (int t = 0; t < TOKENS; ++t) for (int ik = 0; ik < KV; ++ik) {
        float score = 0.0f;
        for (int h = 0; h < HEADS; ++h) {
            float dot = 0.0f;
            for (int d = 0; d < DIM; ++d)
                dot += ((float *) q->data)[((size_t)t * HEADS + h) * DIM + d] *
                       ((float *) k->data)[(size_t)ik * DIM + d];
            score += (dot > 0.0f ? dot : 0.0f) * ((float *) w->data)[(size_t)t * HEADS + h];
        }
        custom[(size_t)t * KV + ik] = score + ggml_fp16_to_fp32(((ggml_fp16_t *)m->data)[(size_t)t * KV + ik]);
    }
    const float * llama = (const float *) out->data;
    double err = 0.0, norm = 0.0, max_abs = 0.0;
    for (size_t i = 0; i < custom.size(); ++i) {
        double d = (double) llama[i] - custom[i];
        err += d*d; norm += (double)llama[i]*llama[i];
        if (std::fabs(d) > max_abs) max_abs = std::fabs(d);
    }
    double rel = std::sqrt(err / (norm + 1e-30));
    std::printf("INDEXER_STAGE rel_l2=%.9g max_abs=%.9g %s\n", rel, max_abs,
                rel <= 1e-5 ? "PASS" : "FAIL");
    glm53f_stage_artifact la = {}, ca = {};
    const char * lr = std::getenv("GLM53F_INDEXER_LLAMA_OUT");
    const char * cr = std::getenv("GLM53F_INDEXER_CUSTOM_OUT");
    if (lr && glm53f_stage_artifact_open(&la, lr, "llama_cpp", "glm53f-indexer-primitive", "indexer-stage", 3, 0, TOKENS)) return 1;
    if (cr && glm53f_stage_artifact_open(&ca, cr, "custom_adapter", "glm53f-indexer-primitive", "indexer-stage", 3, 0, TOKENS)) return 1;
    const size_t shape[] = { TOKENS, KV };
    if (la.manifest) emit(&la, "scores", llama, custom.size(), shape, 2);
    if (ca.manifest) emit(&ca, "scores", custom.data(), custom.size(), shape, 2);
    if (la.manifest && glm53f_stage_artifact_close(&la)) return 1;
    if (ca.manifest && glm53f_stage_artifact_close(&ca)) return 1;
    ggml_free(ctx);
    return rel <= 1e-5 ? 0 : 1;
}
