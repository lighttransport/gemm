#include "llama-model-loader.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "../../common/glm53f_stage_artifact.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

enum { HIDDEN = 4096, INTER = 12288, LIMIT = 10 };

static void write_f32(glm53f_stage_artifact * a, const char * stage,
                      const char * name, const float * x, size_t n,
                      const size_t * shape, size_t rank) {
    if (a->manifest && glm53f_stage_write_f32(a, stage, name, x, n, shape, rank)) {
        std::fprintf(stderr, "DENSE_STAGE FAIL artifact stage=%s name=%s\n", stage, name);
        std::exit(1);
    }
}

static void load_tensor(llama_model_loader & ml,
                        const llama_model_loader::llama_tensor_weight & w,
                        ggml_tensor * dst) {
    const size_t bytes = ggml_nbytes(dst);
    if (bytes != ggml_nbytes(w.tensor) || !ml.load_data_range(w, 0, bytes, dst->data)) {
        std::fprintf(stderr, "DENSE_STAGE FAIL tensor_load\n");
        std::exit(1);
    }
}

static void matvec_quantized(const ggml_tensor * w, const float * x,
                             int rows, int cols, std::vector<float> & out) {
    const auto * traits = ggml_get_type_traits_cpu(w->type);
    const auto * input_traits = traits ? ggml_get_type_traits_cpu(traits->vec_dot_type) : nullptr;
    if (!traits || !input_traits || !input_traits->from_float || !traits->vec_dot) {
        std::fprintf(stderr, "DENSE_STAGE FAIL no_cpu_quant_path type=%d\n", (int) w->type);
        std::exit(1);
    }
    const size_t qbytes = ggml_row_size(traits->vec_dot_type, cols);
    const size_t row_bytes = ggml_row_size(w->type, cols);
    std::vector<unsigned char> qx(qbytes);
    input_traits->from_float(x, qx.data(), cols);
    out.resize((size_t) rows);
    for (int r = 0; r < rows; ++r) {
        traits->vec_dot(cols, out.data() + r, 0,
                        (const char *) w->data + (size_t) r * row_bytes, 0,
                        qx.data(), 0, 1);
    }
}

static void custom_ffn(const ggml_tensor * gate_w,
                       const ggml_tensor * up_w,
                       const ggml_tensor * down_w,
                       const float * x, float * gate, float * up,
                       float * act, float * out) {
    std::vector<float> gate_v, up_v, down_v;
    matvec_quantized(gate_w, x, INTER, HIDDEN, gate_v);
    matvec_quantized(up_w, x, INTER, HIDDEN, up_v);
    for (int r = 0; r < INTER; ++r) {
        gate[r] = gate_v[r]; up[r] = up_v[r];
        const float g = std::min(gate[r], (float) LIMIT);
        const float u = std::max(-((float) LIMIT), std::min(up[r], (float) LIMIT));
        act[r] = g / (1.0f + std::exp(-g)) * u;
    }
    matvec_quantized(down_w, act, HIDDEN, INTER, down_v);
    std::memcpy(out, down_v.data(), HIDDEN * sizeof(float));
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s FIRST_GGUF_SHARD [TOKEN_ID ...]\n", argv[0]);
        return 2;
    }
    try {
        std::vector<std::string> splits;
        llama_model_loader ml(nullptr, nullptr, nullptr, argv[1], splits, nullptr,
                              LLAMA_LOAD_MODE_NONE, true, true, false, nullptr, nullptr);
        const auto * ew = ml.get_weight("token_embd.weight");
        const auto * gw = ml.get_weight("blk.0.ffn_gate.weight");
        const auto * uw = ml.get_weight("blk.0.ffn_up.weight");
        const auto * dw = ml.get_weight("blk.0.ffn_down.weight");
        if (!ew || !gw || !uw || !dw) {
            std::fprintf(stderr, "DENSE_STAGE FAIL missing layer-0 tensors\n");
            return 1;
        }
        const int ids_default[] = { 0, 42, 1234 };
        std::vector<int> ids;
        if (argc == 2) ids.assign(ids_default, ids_default + 3);
        else for (int i = 2; i < argc; ++i) ids.push_back(std::atoi(argv[i]));

        const char * llama_root = std::getenv("GLM53F_DENSE_LLAMA_OUT");
        const char * custom_root = std::getenv("GLM53F_DENSE_CUSTOM_OUT");
        glm53f_stage_artifact llama_art = {}, custom_art = {};
        if (llama_root && glm53f_stage_artifact_open(&llama_art, llama_root,
                "llama_cpp", "glm53f-gguf-q2", "layer0-dense-stage", 0, 0, ids.size())) return 1;
        if (custom_root && glm53f_stage_artifact_open(&custom_art, custom_root,
                "custom_adapter", "glm53f-gguf-q2", "layer0-dense-stage", 0, 0, ids.size())) return 1;

        ggml_init_params ip = { 1024ull * 1024ull * 1024ull, nullptr, false };
        ggml_context * ctx = ggml_init(ip);
        if (!ctx) return 1;
        ggml_tensor * qg = ggml_new_tensor_2d(ctx, gw->tensor->type, HIDDEN, INTER);
        ggml_tensor * qu = ggml_new_tensor_2d(ctx, uw->tensor->type, HIDDEN, INTER);
        ggml_tensor * qd = ggml_new_tensor_2d(ctx, dw->tensor->type, INTER, HIDDEN);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HIDDEN, 1);
        load_tensor(ml, *gw, qg); load_tensor(ml, *uw, qu); load_tensor(ml, *dw, qd);
        const size_t embed_bytes = ggml_row_size(ew->tensor->type, HIDDEN);
        const auto * embed_traits = ggml_get_type_traits(ew->tensor->type);
        std::vector<unsigned char> embed_raw(embed_bytes), embed(HIDDEN * sizeof(float));
        std::vector<float> cgate(INTER), cup(INTER), cact(INTER), cout(HIDDEN);

        ggml_tensor * g = ggml_mul_mat(ctx, qg, x);
        ggml_tensor * u = ggml_mul_mat(ctx, qu, x);
        ggml_tensor * a = ggml_swiglu_clamp(ctx, g, u, LIMIT);
        ggml_tensor * y = ggml_mul_mat(ctx, qd, a);
        /* Keep the intermediate alive: the graph planner may reuse the
         * original activation buffer once the down projection consumes it. */
        ggml_tensor * a_save = ggml_dup(ctx, a);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, y);
        ggml_build_forward_expand(graph, a_save);

        float max_rel = 0.0f, max_abs = 0.0f;
        for (int token : ids) {
            ml.load_data_range(*ew, (size_t) token * embed_bytes, embed_raw.size(), embed_raw.data());
            embed_traits->to_float(embed_raw.data(), (float *) embed.data(), HIDDEN);
            std::memcpy(x->data, embed.data(), HIDDEN * sizeof(float));
            ggml_graph_compute_with_ctx(ctx, graph, 48);
            const float * llama_out = (const float *) y->data;
            custom_ffn(qg, qu, qd, (const float *) embed.data(),
                       cgate.data(), cup.data(), cact.data(), cout.data());
            double sd = 0.0, sr = 0.0;
            for (int i = 0; i < HIDDEN; ++i) {
                const double d = (double) llama_out[i] - cout[i];
                sd += d * d; sr += (double) llama_out[i] * llama_out[i];
                max_abs = std::max(max_abs, (float) std::fabs(d));
            }
            const float rel = (float) std::sqrt(sd / (sr + 1e-30));
            max_rel = std::max(max_rel, rel);
            double llama_ss = 0.0, custom_ss = 0.0;
            for (int i = 0; i < HIDDEN; ++i) {
                llama_ss += (double) llama_out[i] * llama_out[i];
                custom_ss += (double) cout[i] * cout[i];
            }
            std::printf("DENSE_STAGE_TOKEN id=%d rel_l2=%.9g max_abs=%.9g llama_rms=%.9g custom_rms=%.9g result=%s\n",
                        token, rel, max_abs, std::sqrt(llama_ss / HIDDEN),
                        std::sqrt(custom_ss / HIDDEN), rel <= 1e-3f ? "PASS" : "FAIL");
            char name[96];
            const size_t hshape[] = { 1, HIDDEN }, ishape[] = { 1, INTER };
            std::snprintf(name, sizeof(name), "token_%d_input", token);
            if (llama_art.manifest) write_f32(&llama_art, "ffn_layer0", name, (float *) embed.data(), HIDDEN, hshape, 2);
            if (custom_art.manifest) write_f32(&custom_art, "ffn_layer0", name, (float *) embed.data(), HIDDEN, hshape, 2);
            std::snprintf(name, sizeof(name), "token_%d_output", token);
            if (llama_art.manifest) write_f32(&llama_art, "ffn_layer0", name, llama_out, HIDDEN, hshape, 2);
            if (custom_art.manifest) write_f32(&custom_art, "ffn_layer0", name, cout.data(), HIDDEN, hshape, 2);
            std::snprintf(name, sizeof(name), "token_%d_activation", token);
            if (llama_art.manifest) write_f32(&llama_art, "ffn_layer0", name, (const float *) a_save->data, INTER, ishape, 2);
            if (custom_art.manifest) write_f32(&custom_art, "ffn_layer0", name, cact.data(), INTER, ishape, 2);
        }
        if (llama_art.manifest && glm53f_stage_artifact_close(&llama_art)) return 1;
        if (custom_art.manifest && glm53f_stage_artifact_close(&custom_art)) return 1;
        std::printf("DENSE_STAGE %s tokens=%zu max_rel_l2=%.9g max_abs=%.9g\n",
                    max_rel <= 1e-3f ? "PASS" : "FAIL", ids.size(), max_rel, max_abs);
        ggml_free(ctx);
        return max_rel <= 1e-3f ? 0 : 1;
    } catch (const std::exception & ex) {
        std::fprintf(stderr, "DENSE_STAGE FAIL exception=%s\n", ex.what());
        return 1;
    }
}
