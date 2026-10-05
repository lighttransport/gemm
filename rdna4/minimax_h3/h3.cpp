// SPDX-License-Identifier: MIT
#include "runtime.hpp"
#include <chrono>
#ifdef HV15N_ROCM
#define H3_DEFAULT_MODEL_DIR "/mnt/disk01/models/h3/weights"
#define H3_BACKEND "minimax_h3_rocm_experimental"
#define H3_DEFAULT_CONVROT_BLAS 1
#else
#define H3_DEFAULT_MODEL_DIR "/mnt/nvme01/models/h3/weights"
#define H3_BACKEND "minimax_h3_cuda_experimental"
// CUDA defaults to the fused factorized ConvRot; 1 selects dense cuBLAS BF16 rotation.
#define H3_DEFAULT_CONVROT_BLAS 0
#endif
struct h3_context {
    std::unique_ptr<h3::Engine> engine;
    std::string metrics;
    std::atomic<bool> busy{false};
};
static int fail(char *error, size_t capacity, const std::string &text) {
    if (error && capacity)
        std::snprintf(error, capacity, "%s", text.c_str());
    return -1;
}
extern "C" {
void h3_config_defaults(h3_config *c) {
    if (c)
        *c = {H3_DEFAULT_MODEL_DIR, 0, 12288, 1, H3_DEFAULT_CONVROT_BLAS, nullptr, 0};
}
void h3_request_defaults(h3_request *r) {
    if (r)
        *r = {"", nullptr, nullptr, nullptr, 1344, 768, 124, 40, 42};
}
int h3_validate(const h3_request *r, char *error, size_t capacity) {
    if (!r || !r->prompt || !*r->prompt || std::strlen(r->prompt) > 4096 || r->seed < 0)
        return fail(error, capacity,
                    "prompt must be nonempty, <=4096 bytes; seed must be nonnegative");
    if (r->width < 64 || r->height < 64 || r->width % 32 || r->height % 32 || r->width > 2048 ||
        r->height > 2048 || int64_t(r->width) * r->height > 1344 * 768)
        return fail(error, capacity,
                    "dimensions must be multiples of 32, >=64, with area <=1344*768");
    if (r->frames < 5 || r->frames > 362 || (r->frames - 5) % 17)
        return fail(error, capacity, "frames must be 17*n+5, in 5..362");
    if (r->steps < 2 || r->steps > 100)
        return fail(error, capacity, "steps is the number of sigma-grid points, in 2..100");
    return 0;
}
h3_context *h3_load(const h3_config *c, char *error, size_t capacity) {
    try {
        h3::require(c && c->model_dir, "missing H3 configuration");
        h3::require(c->convrot_hipblas == 0 || c->convrot_hipblas == 1,
                    "ConvRot backend must be 0 or 1");
        h3::require(c->bf16_hipblas == 0 || c->bf16_hipblas == 1, "BF16 backend must be 0 or 1");
        h3::require(c->vae_hipblas == 0 || c->vae_hipblas == 1, "VAE backend must be 0 or 1");
        h3::require(!c->aotriton_bridge || h3::fs::is_regular_file(c->aotriton_bridge),
                    "AOTriton bridge must be an existing shared library");
        auto ctx = std::make_unique<h3_context>();
        ctx->engine = std::make_unique<h3::Engine>(*c);
        return ctx.release();
    } catch (const std::exception &e) {
        fail(error, capacity, e.what());
        return nullptr;
    }
}
int h3_set_fp32_hipblas(h3_context *ctx, int enabled, char *error, size_t capacity) {
    if (!ctx || !ctx->engine)
        return fail(error, capacity, "missing H3 context");
    if (enabled != 0 && enabled != 1)
        return fail(error, capacity, "FP32 backend must be 0 or 1");
    bool expected = false;
    if (!ctx->busy.compare_exchange_strong(expected, true))
        return fail(error, capacity, "H3 context is already generating");
    ctx->engine->fp32_hipblas = enabled != 0;
    ctx->busy = false;
    return 0;
}
int h3_set_conditioning(h3_context *ctx, const char *variant, const char *directory,
                        char *error, size_t capacity) {
    if (!ctx || !ctx->engine)
        return fail(error, capacity, "missing H3 context");
    bool expected = false;
    if (!ctx->busy.compare_exchange_strong(expected, true))
        return fail(error, capacity, "H3 context is already generating");
    struct Busy { h3_context *ctx; ~Busy() { ctx->busy = false; } } busy{ctx};
    try {
        using namespace h3;
        std::string mode = variant ? variant : "ref2va";
        require(mode == "ref2va" || mode == "fl2va", "variant must be ref2va or fl2va");
        fs::path folder = directory ? directory : "";
        int rows = 0, text = 0;
        if (!folder.empty()) {
            Json manifest(folder / "manifest.json");
            auto obj = manifest.value.get();
            require(string(obj, "schema") == "h3.image_conditioning.v1" &&
                        string(obj, "variant") == mode, "conditioning schema/variant mismatch");
            auto integer = [&](const char *key, int maximum) {
                auto value = field(obj, key);
                require(value && value->type == JSON_NUMBER && std::isfinite(value->num) &&
                            value->num >= 1 && value->num <= maximum && std::floor(value->num) == value->num,
                        std::string("invalid conditioning ") + key);
                return int(value->num);
            };
            text = integer("text_rows", 2048);
            rows = integer("condition_rows", 32768);
            for (const char *name : {"qwen_inputs.f32", "qwen_rotation.f32", "deepstack_0.f32",
                                     "deepstack_1.f32", "deepstack_2.f32", "text_tags.f32",
                                     "dit_phases.f32", "condition_patches.f32"})
                relative_file(folder, name);
        }
        auto &e = *ctx->engine;
#ifdef HV15N_ROCM
        if (!folder.empty())
            require(e.aot_attention && (!e.convrot_hipblas || e.lt_forward),
                    "conditioned H3 requires the hipBLASLt-enabled AOTriton bridge");
#endif
        e.variant = mode;
        e.conditioning = folder;
        e.condition_rows = rows;
        e.condition_text_rows = text;
        return 0;
    } catch (const std::exception &e) {
        return fail(error, capacity, e.what());
    }
}
int h3_set_cudnn_attention(h3_context *ctx, const char *mode, const char *cudnn_library,
                           char *error, size_t capacity) {
    if (!ctx || !ctx->engine)
        return fail(error, capacity, "missing H3 context");
    bool expected = false;
    if (!ctx->busy.compare_exchange_strong(expected, true))
        return fail(error, capacity, "H3 context is already generating");
    struct Busy {
        h3_context *ctx;
        ~Busy() { ctx->busy = false; }
    } busy{ctx};
    std::string m = mode ? mode : "off";
#ifdef HV15N_ROCM
    (void)cudnn_library;
    if (m == "off")
        return 0;
    return fail(error, capacity, "cuDNN attention is CUDA-only");
#else
    auto &e = *ctx->engine;
    if (m == "off") {
        e.disable_cudnn();
        return 0;
    }
    std::string bridge = m;
    if (m == "auto") {
        Dl_info self{};
        dladdr(reinterpret_cast<void *>(&h3_set_cudnn_attention), &self);
        h3::fs::path dir = self.dli_fname ? h3::fs::path(self.dli_fname).parent_path() : "";
        if (dir.empty() || !h3::fs::exists(dir / "libh3_cudnn.so"))
            dir = h3::fs::read_symlink("/proc/self/exe").parent_path();
        bridge = (dir / "libh3_cudnn.so").string();
    }
    try {
        e.enable_cudnn(bridge, cudnn_library && *cudnn_library ? cudnn_library : nullptr);
        std::cerr << "H3 cuDNN attention: " << e.cudnn_info << "\n";
        return 0;
    } catch (const std::exception &x) {
        e.disable_cudnn();
        if (m != "auto")
            return fail(error, capacity, x.what());
        std::cerr << "H3 cuDNN attention unavailable, using FlashAttention-2: " << x.what() << "\n";
        return 0;
    }
#endif
}
int h3_generate(h3_context *ctx, const h3_request *r, const h3_callbacks *callbacks, char *error,
                size_t capacity) {
    if (h3_validate(r, error, capacity) != 0)
        return -1;
    if (!ctx || !ctx->engine)
        return fail(error, capacity, "missing H3 context");
    bool expected = false;
    if (!ctx->busy.compare_exchange_strong(expected, true))
        return fail(error, capacity, "H3 context is already generating");
    struct Busy {
        h3_context *ctx;
        ~Busy() { ctx->busy = false; }
    } busy{ctx};
    try {
        using namespace h3;
        auto &e = *ctx->engine;
        auto &g = e.g;
        e.dit_mod_indices = {};
        e.condition_patches = {};
        h3_callbacks cb = callbacks ? *callbacks : h3_callbacks{};
        g.check(cuCtxSetCurrent(g.context), "activate H3 context");
        g.cancelled = false;
        g.cancel_check = [cb]() { return cb.cancelled && cb.cancelled(cb.user); };
        fs::path dump = r->dump_dir ? r->dump_dir : "";
        if (!dump.empty())
            fs::create_directories(dump);
        int t = (r->frames - 5) / 17 * 5 + 2, h = r->height / 16, w = r->width / 16,
            at = int(std::nearbyint(r->frames / 24. * 40));
        auto noise = [&](const char *file, int rows, int channels, uint64_t seed) {
            std::vector<float> x(size_t(rows) * channels);
            if (file && *file) {
                auto raw = read_f32(file, x.size());
                for (int row = 0; row < rows; row++)
                    for (int c = 0; c < channels; c++)
                        x[size_t(row) * channels + c] = raw[size_t(c) * rows + row];
            } else {
                std::mt19937_64 rng(seed);
                for (size_t i = 0; i < x.size(); i += 2) {
                    double u = (double(rng() >> 11) + .5) / 9007199254740992.,
                           v = (double(rng() >> 11) + .5) / 9007199254740992.;
                    double radius = std::sqrt(-2 * std::log(u)), angle = 6.283185307179586 * v;
                    x[i] = float(radius * std::cos(angle));
                    if (i + 1 < x.size())
                        x[i + 1] = float(radius * std::sin(angle));
                }
            }
            return g.upload(x, {rows, channels});
        };
        auto video = noise(r->noise_file, t * h * w, 24, r->seed);
        video.shape = {t, h, w, 24};
        auto audio = noise(r->audio_noise_file, at * 2, 32, uint64_t(r->seed) + 1);
        g.dump(video, dump, "noise_video");
        g.dump(audio, dump, "noise_audio");
        using clock = std::chrono::steady_clock;
        auto seconds = [&](clock::time_point start) {
            g.check(cuStreamSynchronize(g.stream), "stage timing");
            return std::chrono::duration<double>(clock::now() - start).count();
        };
        auto stage = clock::now();
        if (!e.conditioning.empty()) {
            Json manifest(e.conditioning / "manifest.json");
            auto obj = manifest.value.get();
            for (auto pair : {std::pair<const char *, int>{"width", r->width}, {"height", r->height},
                               {"frames", r->frames}}) {
                auto v = field(obj, pair.first);
                require(v && v->type == JSON_NUMBER && v->num == pair.second,
                        "conditioning target geometry mismatch");
            }
            require(string(obj, "prompt") == r->prompt, "conditioning prompt mismatch");
            auto condition_seed = field(obj, "seed");
            require(condition_seed && condition_seed->type == JSON_NUMBER &&
                        condition_seed->num == double(r->seed), "conditioning seed mismatch");
            e.condition_patches = g.upload(read_f32(e.conditioning / "condition_patches.f32",
                                                   size_t(e.condition_rows)*96), {e.condition_rows, 96});
            g.dump(e.condition_patches, dump, "condition_patches");
        }
        auto text = e.encode(r->prompt, dump);
        double qwen_seconds = seconds(stage), refine_seconds = 0, dit_seconds = 0;
        stage = clock::now();
        {
            Weights weights(e.checkpoint());
            text = e.refine(weights, text, dump);
            refine_seconds = seconds(stage);
            stage = clock::now();
            auto rotation = e.dit_rope(weights, text.rows(), at, t, h, w);
            if (!e.conditioning.empty())
                g.dump(rotation, dump, "condition_rotation");
            auto sigma = [&](int i, float shift) {
                float base = 1.f - float(i) / float(r->steps - 1);
                return shift * base / (1.f + (shift - 1.f) * base);
            };
            for (int step = 0; step < r->steps - 1; step++) {
                g.poll();
                float sv = sigma(step, 12), sa = sigma(step, 3);
                auto velocity = e.denoise(weights, text, video, audio, rotation, t, h, w, 1 - sv,
                                          1 - sa, step + 1);
                video = g.op(video, 1, &velocity[0], nullptr, sv - sigma(step + 1, 12));
                audio = g.op(audio, 1, &velocity[1], nullptr, sa - sigma(step + 1, 3));
                char name[64];
                std::snprintf(name, sizeof(name), "latent_video_%03d", step);
                g.dump(video, dump, name);
                std::snprintf(name, sizeof(name), "latent_audio_%03d", step);
                g.dump(audio, dump, name);
                if (cb.progress)
                    cb.progress(step + 1, r->steps - 1, cb.user);
            }
        }
        dit_seconds = seconds(stage);
        text = {};
        audio = {};
        e.condition_patches = {};
        e.dit_mod_indices = {};
        g.clear_weights();
        g.trim_pool();
        stage = clock::now();
        e.decode(video, t, h, w, r->frames, cb, dump);
        g.check(cuStreamSynchronize(g.stream), "finish H3");
        double vae_seconds = seconds(stage);
        ctx->metrics =
            "{\"backend\":\"" H3_BACKEND "\",\"int8_wmma_calls\":" +
            std::to_string(e.int8_calls) + ",\"bf16_wmma_calls\":" + std::to_string(e.bf16_calls) +
            ",\"bf16_hipblas_calls\":" + std::to_string(e.bf16_blas_calls) +
            ",\"fp32_hipblas_calls\":" + std::to_string(e.fp32_blas_calls) +
            ",\"convrot_hipblas_calls\":" + std::to_string(e.convrot_blas_calls) +
            ",\"aotriton_attention_calls\":" + std::to_string(e.aot_calls) +
#ifndef HV15N_ROCM
            ",\"cudnn_attention_calls\":" + std::to_string(e.cudnn_calls) +
            ",\"cudnn_library\":\"" + e.cudnn_info + "\"" +
#endif
            ",\"qwen_seconds\":" + std::to_string(qwen_seconds) +
            ",\"refine_seconds\":" + std::to_string(refine_seconds) +
            ",\"dit_seconds\":" + std::to_string(dit_seconds) +
            ",\"vae_seconds\":" + std::to_string(vae_seconds) +
            ",\"variant\":" + escape(e.variant) + ",\"condition_rows\":" + std::to_string(e.condition_rows) +
            ",\"convrot_hipblaslt_calls\":" + std::to_string(e.convrot_lt_calls) +
            ",\"sigma_grid_points\":" + std::to_string(r->steps) +
            ",\"euler_updates\":" + std::to_string(r->steps - 1) + ",\"gpu\":" + g.metrics() + "}";
        g.cancel_check = {};
        return 0;
    } catch (const std::exception &e) {
        ctx->engine->g.cancel_check = {};
        return fail(error, capacity, e.what());
    }
}
const char *h3_metrics(const h3_context *c) { return c ? c->metrics.c_str() : "{}"; }
void h3_cancel(h3_context *c) {
    if (c)
        c->engine->g.cancelled = true;
}
void h3_free(h3_context *c) { delete c; }
}
