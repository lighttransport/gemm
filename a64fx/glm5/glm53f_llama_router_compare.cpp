#include "llama-model-loader.h"
#include "ggml.h"
#include "../../common/glm53f_ref.h"
#include "../../common/glm53f_stage_artifact.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

enum { HIDDEN = 4096, EXPERTS = 288, TOPK = 8 };

static float sigmoid(float x) {
    return x >= 0.0f ? 1.0f/(1.0f + std::exp(-x)) : std::exp(x)/(1.0f + std::exp(x));
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
        const auto * gw = ml.get_weight("blk.3.ffn_gate_inp.weight");
        const auto * bw = ml.get_weight("blk.3.exp_probs_b.bias");
        if (!ew || !gw || !bw) {
            std::fprintf(stderr, "LLAMA_GLM53F_ROUTER FAIL missing GGUF tensors\n");
            return 1;
        }
        const size_t embed_bytes = ggml_row_size(ew->tensor->type, HIDDEN);
        const auto * embed_traits = ggml_get_type_traits(ew->tensor->type);
        if (!embed_traits->to_float) {
            std::fprintf(stderr, "LLAMA_GLM53F_ROUTER FAIL no embedding dequantizer\n");
            return 1;
        }
        std::vector<unsigned char> embed_raw(embed_bytes);
        std::vector<float> embed(HIDDEN);
        std::vector<float> gate((size_t) HIDDEN * EXPERTS);
        std::vector<float> bias(EXPERTS), logits(EXPERTS);
        ml.load_data_range(*gw, 0, gate.size()*sizeof(gate[0]), gate.data());
        ml.load_data_range(*bw, 0, bias.size()*sizeof(bias[0]), bias.data());

        std::vector<int> token_ids;
        if (argc == 2) {
            const int defaults[] = { 0, 1, 2, 42, 1234, 154820, 154822, 154827 };
            token_ids.assign(defaults, defaults + sizeof(defaults)/sizeof(defaults[0]));
        } else {
            for (int i = 2; i < argc; ++i) {
                char * end = nullptr;
                long id = std::strtol(argv[i], &end, 10);
                if (!end || *end != '\0' || id < 0 || id >= 154880) {
                    std::fprintf(stderr, "invalid TOKEN_ID: %s\n", argv[i]);
                    return 2;
                }
                token_ids.push_back((int) id);
            }
        }

        glm53f_stage_artifact artifact = {};
        const char * artifact_root = std::getenv("GLM53F_STAGE_OUT");
        const bool write_artifacts = artifact_root && *artifact_root;
        if (write_artifacts && glm53f_stage_artifact_open(
                &artifact, artifact_root, "llama_cpp", "glm53f-gguf-q2",
                std::getenv("GLM53F_STAGE_PROMPT") ? std::getenv("GLM53F_STAGE_PROMPT") : "token-id-suite",
                3, 0, (unsigned) token_ids.size())) {
            std::fprintf(stderr, "LLAMA_GLM53F_STAGE FAIL artifact_open=%s\n", artifact_root);
            return 1;
        }
        glm53f_stage_artifact custom_artifact = {};
        const char * custom_artifact_root = std::getenv("GLM53F_CUSTOM_STAGE_OUT");
        const bool write_custom_artifacts = custom_artifact_root && *custom_artifact_root;
        if (write_custom_artifacts && glm53f_stage_artifact_open(
                &custom_artifact, custom_artifact_root, "custom_adapter", "glm53f-gguf-q2",
                std::getenv("GLM53F_STAGE_PROMPT") ? std::getenv("GLM53F_STAGE_PROMPT") : "token-id-suite",
                3, 0, (unsigned) token_ids.size())) {
            std::fprintf(stderr, "LLAMA_GLM53F_STAGE FAIL custom_artifact_open=%s\n", custom_artifact_root);
            return 1;
        }

        bool all_pass = true;
        float global_max_weight_diff = 0.0f;
        for (int token_id : token_ids) {
            std::vector<float> weights(TOPK);
            std::vector<int> ids(TOPK, -1);
            ml.load_data_range(*ew, (size_t) token_id * embed_bytes,
                               embed_raw.size(), embed_raw.data());
            embed_traits->to_float(embed_raw.data(), embed.data(), HIDDEN);

            if (write_artifacts || write_custom_artifacts) {
                char name[96];
                const size_t shape[] = { 1, HIDDEN };
                std::snprintf(name, sizeof(name), "token_%d", token_id);
                if (write_artifacts && glm53f_stage_write_f32(&artifact, "embedding", name,
                        embed.data(), HIDDEN, shape, 2)) return 1;
                if (write_custom_artifacts && glm53f_stage_write_f32(&custom_artifact, "embedding", name,
                        embed.data(), HIDDEN, shape, 2)) return 1;
            }

            for (int e = 0; e < EXPERTS; ++e) {
                double z = 0.0;
                for (int j = 0; j < HIDDEN; ++j) {
                    z += (double) gate[(size_t)e*HIDDEN + j] * embed[j];
                }
                logits[e] = (float) z;
            }

            for (int e = 0; e < EXPERTS; ++e) {
                float choice = sigmoid(logits[e]) + bias[e];
                for (int j = 0; j < TOPK; ++j) {
                    if (ids[j] < 0 || choice > weights[j]) {
                        for (int k = TOPK - 1; k > j; --k) {
                            weights[k] = weights[k-1]; ids[k] = ids[k-1];
                        }
                        weights[j] = choice; ids[j] = e; break;
                    }
                }
            }
            float sum = 0.0f;
            for (int j = 0; j < TOPK; ++j) { weights[j] = sigmoid(logits[ids[j]]); sum += weights[j]; }
            for (int j = 0; j < TOPK; ++j) weights[j] = weights[j] / sum * 2.5f;

            int ref_ids[TOPK];
            float ref_weights[TOPK];
            glm53f_router_topk(logits.data(), bias.data(), EXPERTS, TOPK, 2.5f,
                               ref_ids, ref_weights);

            bool same = true;
            float max_weight_diff = 0.0f, routed_sum = 0.0f;
            for (int j = 0; j < TOPK; ++j) same &= ids[j] == ref_ids[j];
            for (int j = 0; j < TOPK; ++j) {
                max_weight_diff = std::fmax(max_weight_diff, std::fabs(weights[j] - ref_weights[j]));
                routed_sum += weights[j];
            }
            const bool pass = same && std::fabs(routed_sum - 2.5f) < 1e-5f && max_weight_diff <= 2e-6f;
            all_pass &= pass;
            global_max_weight_diff = std::fmax(global_max_weight_diff, max_weight_diff);
            std::printf("LLAMA_GLM53F_ROUTER_TOKEN id=%d llama=", token_id);
            for (int j = 0; j < TOPK; ++j) std::printf(" %d:%.9g", ids[j], (double)weights[j]);
            std::printf(" ref=");
            for (int j = 0; j < TOPK; ++j) std::printf(" %d:%.9g", ref_ids[j], (double)ref_weights[j]);
            std::printf(" result=%s routed_weight_sum=%.9g max_weight_diff=%.9g\n",
                        pass ? "PASS" : "FAIL", (double)routed_sum, (double)max_weight_diff);
            if (write_artifacts || write_custom_artifacts) {
                char name[96];
                const size_t logits_shape[] = { 1, EXPERTS };
                const size_t route_shape[] = { 1, TOPK };
                int32_t ids32[TOPK];
                for (int j = 0; j < TOPK; ++j) ids32[j] = ref_ids[j];
                std::snprintf(name, sizeof(name), "token_%d_logits", token_id);
                if (write_artifacts && glm53f_stage_write_f32(&artifact, "router", name, logits.data(),
                        EXPERTS, logits_shape, 2)) return 1;
                if (write_custom_artifacts && glm53f_stage_write_f32(&custom_artifact, "router", name, logits.data(),
                        EXPERTS, logits_shape, 2)) return 1;
                std::snprintf(name, sizeof(name), "token_%d_ids", token_id);
                if (write_artifacts && glm53f_stage_write_i32(&artifact, "router", name, ids32,
                        TOPK, route_shape, 2)) return 1;
                if (write_custom_artifacts && glm53f_stage_write_i32(&custom_artifact, "router", name, ids32,
                        TOPK, route_shape, 2)) return 1;
                std::snprintf(name, sizeof(name), "token_%d_weights", token_id);
                if (write_artifacts && glm53f_stage_write_f32(&artifact, "router", name, ref_weights,
                        TOPK, route_shape, 2)) return 1;
                if (write_custom_artifacts && glm53f_stage_write_f32(&custom_artifact, "router", name, ref_weights,
                        TOPK, route_shape, 2)) return 1;
            }
        }
        if (write_artifacts && glm53f_stage_artifact_close(&artifact)) return 1;
        if (write_custom_artifacts && glm53f_stage_artifact_close(&custom_artifact)) return 1;
        std::printf("LLAMA_GLM53F_ROUTER_PROMPT_SUITE %s tokens=%zu max_weight_diff=%.9g\n",
                    all_pass ? "PASS" : "FAIL", token_ids.size(), (double)global_max_weight_diff);
        return all_pass ? 0 : 1;
    } catch (const std::exception & ex) {
        std::fprintf(stderr, "LLAMA_GLM53F_ROUTER FAIL exception=%s\n", ex.what());
        return 1;
    }
}
