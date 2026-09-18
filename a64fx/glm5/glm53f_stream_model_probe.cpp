#include "llama-model.h"
#include "llama-model-loader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <cmath>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 3) {
        std::fprintf(stderr, "usage: %s FIRST_GGUF_SHARD [LAYER]\n", argv[0]);
        return 2;
    }
    try {
        const int target_layer = argc == 3 ? std::atoi(argv[2]) : 11;
        std::vector<std::string> splits;
        llama_model_loader loader(nullptr, nullptr, nullptr, argv[1], splits, nullptr,
            LLAMA_LOAD_MODE_NONE, true, true, false, nullptr, nullptr);

        llama_model_params params = llama_model_default_params();
        params.no_alloc = true;
        params.load_mode = LLAMA_LOAD_MODE_NONE;
        params.n_gpu_layers = 0;

        std::unique_ptr<llama_model> model(llama_model_create(loader, params));
        auto * base = dynamic_cast<llama_model_base *>(model.get());
        if (!base) throw std::runtime_error("GLM53F_STREAM_PROBE no llama_model_base");
        model->hparams.no_alloc = true;
        base->load_hparams(loader);
        base->load_vocab(loader);
        base->load_stats(loader);
        if (!base->load_tensors(loader))
            throw std::runtime_error("GLM53F_STREAM_PROBE load_tensors failed");

        size_t layer_bytes = 0;
        size_t layer_tensors = 0;
        for (size_t il = 0; il < model->layers.size(); ++il) {
            size_t bytes = 0, tensors = 0;
            for (const auto & item : model->tensors_by_name) {
                const char * name = item.first.c_str();
                int got = -1;
                if (std::sscanf(name, "blk.%d.", &got) == 1 && got == (int)il) {
                    if (const auto * w = loader.get_weight(name)) {
                        bytes += ggml_nbytes(w->tensor);
                        ++tensors;
                    }
                }
            }
            if (bytes > layer_bytes) { layer_bytes = bytes; layer_tensors = tensors; }
            std::printf("GLM53F_STREAM_LAYER il=%zu tensors=%zu bytes=%zu\n", il, tensors, bytes);
        }
        std::printf("GLM53F_STREAM_MODEL arch=%s layers=%zu n_embd=%u vocab=%u\n",
            model->arch_name().c_str(), model->layers.size(), model->hparams.n_embd,
            model->vocab.n_tokens());
        std::printf("GLM53F_STREAM_MODEL PASS max_layer_tensors=%zu max_layer_bytes=%zu\n",
            layer_tensors, layer_bytes);

        size_t loaded = 0;
        uint64_t fnv = 1469598103934665603ull;
        size_t finite_values = 0;
        std::vector<void *> payloads;
        for (const auto & item : model->tensors_by_name) {
            int got = -1;
            if (std::sscanf(item.first.c_str(), "blk.%d.", &got) != 1 || got != target_layer)
                continue;
            const auto * w = loader.get_weight(item.first.c_str());
            if (!w) throw std::runtime_error("missing stream weight " + item.first);
            const size_t nbytes = ggml_nbytes(w->tensor);
            void * buf = nullptr;
            if (posix_memalign(&buf, 64, nbytes ? nbytes : 64) != 0 || !buf)
                throw std::runtime_error("stream layer allocation failed");
            loader.load_data_range(*w, 0, nbytes, buf);
            const unsigned char * p = static_cast<const unsigned char *>(buf);
            for (size_t i = 0; i < nbytes; ++i) {
                fnv ^= p[i];
                fnv *= 1099511628211ull;
            }
            if (w->tensor->type == GGML_TYPE_F32) {
                const float * f = static_cast<const float *>(buf);
                for (size_t i = 0; i < nbytes / sizeof(float); ++i)
                    finite_values += std::isfinite(f[i]) ? 1u : 0u;
            }
            loaded += nbytes;
            payloads.push_back(buf);
        }
        if (target_layer < 0 || target_layer >= (int)model->layers.size() || loaded == 0)
            throw std::runtime_error("invalid or empty target stream layer");
        for (void * p : payloads) std::free(p);
        std::printf("GLM53F_STREAM_PAYLOAD layer=%d tensors=%zu bytes=%zu f32_finite=%zu fnv=%016llx PASS\n",
            target_layer, payloads.size(), loaded, finite_values,
            (unsigned long long)fnv);
        return 0;
    } catch (const std::exception & ex) {
        std::fprintf(stderr, "GLM53F_STREAM_MODEL FAIL exception=%s\n", ex.what());
        return 1;
    }
}
