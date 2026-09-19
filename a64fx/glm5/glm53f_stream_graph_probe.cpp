#include "llama-context.h"
#include "llama-model.h"
#include "llama-model-loader.h"
#include "llama-batch.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <vector>

struct sublayer_trace {
    const char * dir = nullptr;
    int layer = -1;
};

static bool trace_name(const char * name, int layer) {
    static const char * prefixes[] = {
        "hc_attn_pre", "attn_norm", "kda_qkv", "kda_conv",
        "kda_q_norm", "kda_k_norm", "kda_gate", "kda_beta",
        "kda_scan_out", "kda_normed", "kda_out", "hc_attn_post",
        "dsa_q_a_norm", "dsa_q_b", "dsa_kv_a_norm", "dsa_q_absorbed",
        "dsa_kv_latent", "dsa_out", "hc_ffn_pre", "ffn_norm", "ffn_out",
        "l_last",
    };
    if (std::strcmp(name, "hc_init") == 0) return true;
    char wanted[128];
    for (const char * prefix : prefixes) {
        std::snprintf(wanted, sizeof(wanted), "%s-%d", prefix, layer);
        if (std::strcmp(name, wanted) == 0) return true;
    }
    return false;
}

static bool trace_sublayer_cb(ggml_tensor * tensor, bool ask, void * opaque) {
    auto * trace = static_cast<sublayer_trace *>(opaque);
    if (!trace || !trace->dir || !*trace->dir ||
        !trace_name(tensor->name, trace->layer)) return false;
    if (ask) return true;
    if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor)) {
        std::fprintf(stderr, "GLM53F_STREAM_SUBLAYER_FAIL name=%s type=%s contiguous=%d\n",
                     tensor->name, ggml_type_name(tensor->type),
                     ggml_is_contiguous(tensor));
        return false;
    }
    if (mkdir(trace->dir, 0755) != 0 && errno != EEXIST) return false;
    char path[4096];
    if (std::snprintf(path, sizeof(path), "%s/%s.f32", trace->dir,
                      tensor->name) >= (int) sizeof(path)) return false;
    const size_t bytes = ggml_nbytes(tensor);
    std::vector<unsigned char> data(bytes);
    ggml_backend_tensor_get(tensor, data.data(), 0, bytes);
    FILE * out = std::fopen(path, "wb");
    if (!out) return false;
    const bool ok = std::fwrite(data.data(), 1, bytes, out) == bytes &&
                    std::fclose(out) == 0;
    if (!ok) {
        std::fprintf(stderr, "GLM53F_STREAM_SUBLAYER_FAIL name=%s path=%s\n",
                     tensor->name, path);
        return false;
    }
    std::fprintf(stderr, "GLM53F_STREAM_SUBLAYER name=%s count=%zu PASS\n",
                 tensor->name, bytes / sizeof(float));
    return true;
}

struct loaded_tensor {
    ggml_tensor * tensor = nullptr;
    void * data = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
};

static void load_layer(llama_model_loader & loader, llama_model & model,
                       int layer, std::vector<loaded_tensor> & loaded) {
    for (const auto & item : model.tensors_by_name) {
        int got = -1;
        if (std::sscanf(item.first.c_str(), "blk.%d.", &got) != 1 || got != layer)
            continue;
        const auto * weight = loader.get_weight(item.first.c_str());
        if (!weight) throw std::runtime_error("missing " + item.first);
        void * data = nullptr;
        const size_t nbytes = ggml_nbytes(weight->tensor);
        if (posix_memalign(&data, 64, nbytes ? nbytes : 64) != 0 || !data)
            throw std::runtime_error("allocation failed for " + item.first);
        loader.load_data_range(*weight, 0, nbytes, data);
        item.second->data = data;
        ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(data, nbytes ? nbytes : 64);
        if (!buffer) throw std::runtime_error("buffer binding failed for " + item.first);
        item.second->buffer = buffer;
        loaded.push_back({item.second, data, buffer});
    }
}

static void load_final(llama_model_loader & loader, llama_model & model,
                       std::vector<loaded_tensor> & loaded) {
    for (const char * name : {"output_norm.weight", "output.weight"}) {
        auto it = std::find_if(model.tensors_by_name.begin(), model.tensors_by_name.end(),
                               [name](const auto & p) { return p.first == name; });
        if (it == model.tensors_by_name.end()) throw std::runtime_error(std::string("missing ") + name);
        const auto * weight = loader.get_weight(name);
        if (!weight) throw std::runtime_error(std::string("missing ") + name);
        const size_t nbytes = ggml_nbytes(weight->tensor);
        void * data = nullptr;
        if (posix_memalign(&data, 64, nbytes ? nbytes : 64) != 0 || !data)
            throw std::runtime_error(std::string("allocation failed for ") + name);
        loader.load_data_range(*weight, 0, nbytes, data);
        it->second->data = data;
        ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(data, nbytes ? nbytes : 64);
        if (!buffer) throw std::runtime_error(std::string("buffer binding failed for ") + name);
        it->second->buffer = buffer;
        loaded.push_back({it->second, data, buffer});
    }
}

static void unload_layer(std::vector<loaded_tensor> & loaded) {
    for (const loaded_tensor & x : loaded) {
        x.tensor->data = nullptr;
        ggml_backend_buffer_free(x.buffer);
        x.tensor->buffer = nullptr;
        std::free(x.data);
    }
    loaded.clear();
}

static void check_layer_buffers(const std::vector<loaded_tensor> & loaded, const char * where) {
    size_t missing = 0;
    for (const loaded_tensor & x : loaded) {
        if (!x.tensor->buffer || !x.tensor->data) {
            std::fprintf(stderr, "GLM53F_STREAM_GRAPH missing_buffer where=%s name=%s data=%p buffer=%p\n",
                         where, x.tensor->name, x.tensor->data, (void *) x.tensor->buffer);
            ++missing;
        }
    }
    if (missing) throw std::runtime_error("staged layer has missing tensor buffers");
}

static uint64_t fnv1a_f32(const std::vector<float> & values) {
    uint64_t h = UINT64_C(1469598103934665603);
    for (float x : values) {
        const unsigned char * p = reinterpret_cast<const unsigned char *>(&x);
        for (size_t i = 0; i < sizeof(x); ++i) {
            h ^= p[i];
            h *= UINT64_C(1099511628211);
        }
    }
    return h;
}

static int custom_terminal_logits(const llama_model & model,
                                  const std::vector<float> & wide,
                                  std::vector<float> & logits) {
    constexpr int H = 4096;
    constexpr int HC = 4;
    if (!model.output || !model.output_norm || wide.size() != (size_t) H * HC)
        return -1;
    const auto * nt = ggml_get_type_traits(model.output_norm->type);
    const auto * wtc = ggml_get_type_traits_cpu(model.output->type);
    const auto * xtc = wtc ? ggml_get_type_traits_cpu(wtc->vec_dot_type) : nullptr;
    if (!nt || !wtc || !xtc || !wtc->vec_dot || !xtc->from_float) return -1;
    std::vector<float> norm((size_t) H), x((size_t) H);
    if (model.output_norm->type == GGML_TYPE_F32)
        std::memcpy(norm.data(), model.output_norm->data, H * sizeof(float));
    else if (nt->to_float)
        nt->to_float(model.output_norm->data, norm.data(), H);
    else return -1;
    double ss = 0.0;
    for (int d = 0; d < H; ++d) {
        float v = 0.0f;
        for (int s = 0; s < HC; ++s) v += wide[(size_t) s * H + d] * 0.25f;
        x[d] = v;
        ss += (double) v * v;
    }
    const float inv = 1.0f / std::sqrt((float) (ss / H) + 1e-5f);
    for (int d = 0; d < H; ++d) x[d] *= inv * norm[d];
    const int vocab = (int) model.vocab.n_tokens();
    const size_t row_bytes = ggml_row_size(model.output->type, H);
    const size_t xbytes = ggml_row_size(wtc->vec_dot_type, H);
    std::vector<unsigned char> qx(xbytes);
    xtc->from_float(x.data(), qx.data(), H);
    logits.resize((size_t) vocab);
    for (int r = 0; r < vocab; ++r) {
        const void * src = (const char *) model.output->data + (size_t) r * row_bytes;
        wtc->vec_dot(H, logits.data() + r, 0, (const char *) src, 0,
                     qx.data(), 0, 1);
    }
    return 0;
}

static void dump_hidden(const char * dir, int layer, const std::vector<float> & hidden) {
    if (!dir || !*dir) return;
    if (mkdir(dir, 0755) != 0 && errno != EEXIST)
        throw std::runtime_error("cannot create stream artifact directory");
    char path[4096];
    if (std::snprintf(path, sizeof(path), "%s/layer-%02d.f32", dir, layer) >= (int) sizeof(path))
        throw std::runtime_error("stream artifact path too long");
    FILE * f = std::fopen(path, "wb");
    if (!f) throw std::runtime_error("cannot create stream artifact");
    if (std::fwrite(hidden.data(), sizeof(float), hidden.size(), f) != hidden.size()) {
        std::fclose(f);
        throw std::runtime_error("cannot write stream artifact");
    }
    std::fclose(f);
}

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 5) {
        std::fprintf(stderr, "usage: %s FIRST_GGUF_SHARD [ARTIFACT_DIR] [LAYER] [INPUT_F32]\n", argv[0]);
        return 2;
    }
    try {
        std::vector<std::string> splits;
        llama_model_loader loader(nullptr, nullptr, nullptr, argv[1], splits, nullptr,
            LLAMA_LOAD_MODE_NONE, true, true, false, nullptr, nullptr);
        llama_model_params mp = llama_model_default_params();
        mp.no_alloc = true;
        mp.load_mode = LLAMA_LOAD_MODE_NONE;
        mp.n_gpu_layers = 0;
        std::unique_ptr<llama_model> model(llama_model_create(loader, mp));
        auto * base = dynamic_cast<llama_model_base *>(model.get());
        if (!base) throw std::runtime_error("not a base model");
        model->hparams.no_alloc = true;
        base->load_hparams(loader);
        base->load_vocab(loader);
        base->load_stats(loader);
        if (!base->load_tensors(loader)) throw std::runtime_error("load_tensors failed");

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 1;
        cp.n_batch = 1;
        cp.n_ubatch = 1;
        cp.n_seq_max = 1;
        cp.n_threads = 48;
        cp.n_threads_batch = 48;
        const size_t stream_n = (size_t) model->hparams.n_embd * model->hparams.dsv4_hc_mult;
        // GLM5NEXT exposes 46 GGUF blocks: 45 trunk layers plus one NextN
        // block. The streamed trunk/final sentinel follows llama.cpp's n_layer.
        const int n_layers = 45;
        const int layer = argc >= 4 ? std::atoi(argv[3]) : 0;
        const bool final = layer == n_layers;
        if (layer < 0 || layer > n_layers) throw std::runtime_error("invalid stream layer");
        sublayer_trace trace = {std::getenv("GLM53F_STREAM_SUBLAYER_DIR"), layer};
        if (trace.dir && *trace.dir) {
            cp.cb_eval = trace_sublayer_cb;
            cp.cb_eval_user_data = &trace;
        }
        std::vector<float> hidden(stream_n);
        for (size_t i = 0; i < hidden.size(); ++i)
            hidden[i] = 0.001f * (float)((int)(i % 97) - 48);
        if (!final && layer == 0 && argc < 5) {
        const char * token_env = std::getenv("GLM53F_STREAM_TOKEN");
        const int token = std::atoi(token_env != nullptr ? token_env : "1");
            const auto * ew = loader.get_weight("token_embd.weight");
            const auto * et = ew ? ggml_get_type_traits(ew->tensor->type) : nullptr;
            if (!ew || !et || (!et->to_float && ew->tensor->type != GGML_TYPE_F32))
                throw std::runtime_error("token embedding dequantization unavailable");
            const size_t row_bytes = ggml_row_size(ew->tensor->type, model->hparams.n_embd);
            std::vector<unsigned char> raw(row_bytes);
            std::vector<float> row(model->hparams.n_embd);
            loader.load_data_range(*ew, (size_t) token * row_bytes, row_bytes, raw.data());
            if (ew->tensor->type == GGML_TYPE_F32)
                std::memcpy(row.data(), raw.data(), row.size() * sizeof(float));
            else et->to_float(raw.data(), row.data(), row.size());
            for (int s = 0; s < 4; ++s)
                std::memcpy(hidden.data() + (size_t) s * row.size(), row.data(), row.size() * sizeof(float));
            std::fprintf(stderr, "GLM53F_STREAM_INPUT token=%d source=token_embd.weight PASS\n", token);
        }
        if (argc >= 5) {
            FILE * f = std::fopen(argv[4], "rb");
            if (!f || std::fread(hidden.data(), sizeof(float), hidden.size(), f) != hidden.size()) {
                if (f) std::fclose(f);
                throw std::runtime_error("cannot read stream input artifact");
            }
            std::fclose(f);
        }
        std::vector<loaded_tensor> loaded;
        if (final) load_final(loader, *model, loaded);
        else load_layer(loader, *model, layer, loaded);
        if (loaded.empty()) throw std::runtime_error("stream layer has no tensors");
        check_layer_buffers(loaded, "before_context");
        std::unique_ptr<llama_context> ctx(new llama_context(*model, cp));
        check_layer_buffers(loaded, "after_context");
        if (!ctx->set_stream_layer(layer, hidden.data(), hidden.size()))
            throw std::runtime_error("set_stream_layer failed");
        const std::vector<float> input_hidden = hidden;

        llama_batch batch = llama_batch_init(1, 0, 1);
        batch.n_tokens = 1;
        batch.token[0] = 1;
        batch.pos[0] = 0;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 0;
        const int rc = ctx->decode(batch);
        llama_batch_free(batch);
        if (rc != 0) throw std::runtime_error("stream decode failed");

        const float * out = ctx->get_stream_hidden();
        if (!out || (!final && ctx->get_stream_hidden_size() != stream_n) ||
            (final && ctx->get_stream_hidden_size() != model->vocab.n_tokens()))
            throw std::runtime_error("stream output size mismatch");
        hidden.assign(out, out + ctx->get_stream_hidden_size());
        double ss = 0.0;
        size_t finite = 0;
        for (float x : hidden) {
            ss += (double) x * x;
            finite += std::isfinite(x) ? 1u : 0u;
        }
        dump_hidden(argc >= 3 ? argv[2] : nullptr, layer, hidden);
        int best = -1;
        if (final) {
            best = 0;
            for (size_t i = 1; i < hidden.size(); ++i)
                if (hidden[i] > hidden[(size_t) best]) best = (int) i;
            std::vector<float> custom;
            if (custom_terminal_logits(*model, input_hidden, custom) != 0)
                throw std::runtime_error("custom terminal projection failed");
            int custom_best = 0;
            double sd = 0.0, sr = 0.0;
            for (size_t i = 0; i < custom.size(); ++i) {
                if (custom[i] > custom[(size_t) custom_best]) custom_best = (int) i;
                const double d = (double) hidden[i] - custom[i];
                sd += d * d; sr += (double) hidden[i] * hidden[i];
            }
            const double rel = std::sqrt(sd / (sr + 1e-30));
            std::printf("GLM53F_STREAM_CUSTOM_FINAL token=%d llama_token=%d rel_l2=%.9g exact=%s PASS\n",
                        custom_best, best, rel, custom_best == best ? "YES" : "NO");
            if (custom_best != best) throw std::runtime_error("custom terminal token mismatch");
        }
        std::printf("GLM53F_STREAM_GRAPH layer=%d hidden=%zu finite=%zu rms=%.9g fnv=%016llx%s PASS\n",
                    layer, hidden.size(), finite, std::sqrt(ss / hidden.size()),
                    (unsigned long long) fnv1a_f32(hidden),
                    final ? (std::string(" token=") + std::to_string(best)).c_str() : "");
        unload_layer(loaded);
        return 0;
    } catch (const std::exception & ex) {
        std::fprintf(stderr, "GLM53F_STREAM_GRAPH FAIL exception=%s\n", ex.what());
        return 1;
    }
}
