#include "ggml.h"
#include "ggml-cpu.h"
#include "../../common/glm53f_ref.h"
#include "../../common/glm53f_stage_artifact.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

enum { S = 128, HEADS = 2, TOKENS = 3 };

static void put(glm53f_stage_artifact * a, const char * name,
                const float * x, size_t n, const size_t * shape, size_t rank) {
    if (a->manifest && glm53f_stage_write_f32(a, "kda_layer3", name, x, n, shape, rank)) {
        std::fprintf(stderr, "KDA_STAGE FAIL artifact=%s\n", name);
        std::exit(1);
    }
}

int main() {
    ggml_init_params ip = { 256ull * 1024ull * 1024ull, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return 1;
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, HEADS, TOKENS, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, HEADS, TOKENS, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, HEADS, TOKENS, 1);
    ggml_tensor * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, HEADS, TOKENS, 1);
    ggml_tensor * beta = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, HEADS, TOKENS, 1);
    ggml_tensor * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, HEADS, 1);
    for (int h = 0; h < HEADS; ++h) {
        for (int t = 0; t < TOKENS; ++t) {
            float * qv = (float *) q->data + ((size_t)t * HEADS + h) * S;
            float * kv = (float *) k->data + ((size_t)t * HEADS + h) * S;
            float * vv = (float *) v->data + ((size_t)t * HEADS + h) * S;
            float * gv = (float *) g->data + ((size_t)t * HEADS + h) * S;
            double qn = 0.0, kn = 0.0;
            for (int d = 0; d < S; ++d) {
                qv[d] = 0.01f * (float)((d * 7 + t * 11 + h * 13) % 31 - 15);
                kv[d] = 0.01f * (float)((d * 5 + t * 3 + h * 17) % 29 - 14);
                vv[d] = 0.02f * (float)((d * 3 + t * 19 + h * 23) % 23 - 11);
                gv[d] = -0.03f * (float)(1 + ((d + 2*t + h) % 7));
                qn += qv[d] * qv[d]; kn += kv[d] * kv[d];
            }
            for (int d = 0; d < S; ++d) { qv[d] /= std::sqrt(qn); kv[d] /= std::sqrt(kn); }
            ((float *) beta->data)[(size_t)t * HEADS + h] = 0.35f + 0.05f * (float)((t+h)%4);
        }
    }
    std::vector<float> initial((size_t)HEADS * S * S);
    for (int h = 0; h < HEADS; ++h) {
        for (int i = 0; i < S; ++i) for (int j = 0; j < S; ++j) {
            const float z = 0.0007f * (float)((i * 3 + j * 5 + h * 7) % 17 - 8);
            initial[(size_t)h*S*S + (size_t)i*S + j] = z;
            /* GGML stores the recurrent matrix transposed in its flat state. */
            ((float *) state->data)[(size_t)h*S*S + (size_t)j*S + i] = z;
        }
    }
    ggml_tensor * out = ggml_gated_delta_net(ctx, q, k, v, g, beta, state, 1);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    if (ggml_graph_compute_with_ctx(ctx, graph, 48) != GGML_STATUS_SUCCESS) return 1;

    std::vector<float> custom_state = initial;
    std::vector<float> custom_attn((size_t)TOKENS * HEADS * S);
    std::vector<float> work(S), qv(S), kv(S), vv(S), gv(S);
    for (int t = 0; t < TOKENS; ++t) for (int h = 0; h < HEADS; ++h) {
        for (int d = 0; d < S; ++d) {
            qv[d] = ((float *)q->data)[((size_t)t*HEADS+h)*S+d];
            kv[d] = ((float *)k->data)[((size_t)t*HEADS+h)*S+d];
            vv[d] = ((float *)v->data)[((size_t)t*HEADS+h)*S+d];
            gv[d] = ((float *)g->data)[((size_t)t*HEADS+h)*S+d];
        }
        glm53f_kda_step_vec_streamed(custom_state.data() + (size_t)h*S*S,
            qv.data(), kv.data(), vv.data(), gv.data(),
            ((float *)beta->data)[(size_t)t*HEADS+h], S, S,
            custom_attn.data() + ((size_t)t*HEADS+h)*S, work.data());
    }
    const float * llama_attn = (const float *) out->data;
    const float * llama_state = llama_attn + (size_t)TOKENS * HEADS * S;
    double sd = 0.0, sr = 0.0, max_abs = 0.0;
    for (size_t i = 0; i < custom_attn.size(); ++i) {
        const double d = (double)llama_attn[i] - custom_attn[i];
        sd += d*d; sr += (double)llama_attn[i]*llama_attn[i];
        if (std::fabs(d) > max_abs) max_abs = std::fabs(d);
    }
    std::vector<float> llama_state_canon(custom_state.size());
    for (int h = 0; h < HEADS; ++h) for (int i = 0; i < S; ++i) for (int j = 0; j < S; ++j)
        llama_state_canon[(size_t)h*S*S + (size_t)i*S+j] = llama_state[(size_t)h*S*S + (size_t)j*S+i];
    double ss = 0.0, sr_state = 0.0, max_state = 0.0;
    for (size_t i = 0; i < custom_state.size(); ++i) {
        const double d = (double)llama_state_canon[i] - custom_state[i];
        ss += d*d; sr_state += (double)llama_state_canon[i]*llama_state_canon[i];
        if (std::fabs(d) > max_state) max_state = std::fabs(d);
    }
    const double rel = std::sqrt(sd / (sr + 1e-30));
    const double rel_state = std::sqrt(ss / (sr_state + 1e-30));
    std::printf("KDA_STAGE attn_rel_l2=%.9g state_rel_l2=%.9g max_abs=%.9g max_state_abs=%.9g %s\n",
        rel, rel_state, max_abs, max_state, (rel <= 2e-5 && rel_state <= 2e-5) ? "PASS" : "FAIL");

    glm53f_stage_artifact la = {}, ca = {};
    const char * lr = std::getenv("GLM53F_KDA_LLAMA_OUT");
    const char * cr = std::getenv("GLM53F_KDA_CUSTOM_OUT");
    if (lr && glm53f_stage_artifact_open(&la, lr, "llama_cpp", "glm53f-kda-primitive", "kda-stage", 3, 0, TOKENS)) return 1;
    if (cr && glm53f_stage_artifact_open(&ca, cr, "custom_adapter", "glm53f-kda-primitive", "kda-stage", 3, 0, TOKENS)) return 1;
    const size_t ashape[] = { TOKENS, HEADS, S }, sshape[] = { HEADS, S, S };
    if (la.manifest) { put(&la, "attention", llama_attn, custom_attn.size(), ashape, 3); put(&la, "state", llama_state_canon.data(), llama_state_canon.size(), sshape, 3); }
    if (ca.manifest) { put(&ca, "attention", custom_attn.data(), custom_attn.size(), ashape, 3); put(&ca, "state", custom_state.data(), custom_state.size(), sshape, 3); }
    if (la.manifest && glm53f_stage_artifact_close(&la)) return 1;
    if (ca.manifest && glm53f_stage_artifact_close(&ca)) return 1;
    ggml_free(ctx);
    return (rel <= 2e-5 && rel_state <= 2e-5) ? 0 : 1;
}
