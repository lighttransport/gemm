#include "hv15_native.h"
#include "models.hpp"
#include <random>
#include <sstream>
#define STB_IMAGE_IMPLEMENTATION
#include "../../common/stb_image.h"
using namespace hv15n;
struct hv15n_context {
    fs::path root;
    std::unique_ptr<Gpu> gpu;
    std::string metrics = "{}";
    std::atomic<bool> busy{false};
};
static int failure(char *error, size_t capacity, const std::string &message) {
    if (error && capacity)
        std::snprintf(error, capacity, "%s", message.c_str());
    return -1;
}
extern "C" {
void hv15n_config_defaults(hv15n_config *c) {
    if (c)
        *c = {HV15N_DEFAULT_MODEL_DIR, 0, 14336, "repo", "cublas"};
}
void hv15n_request_defaults(hv15n_request *r) {
    if (r)
        *r = {"i2v", "quality", "", "", nullptr, nullptr, nullptr, nullptr, 480, 848, 81, 42};
}
int hv15n_validate(const hv15n_request *r, char *error, size_t capacity) {
    if (!r || !r->task || !r->preset || !r->prompt || !*r->prompt)
        return failure(error, capacity, "task, preset and nonempty prompt are required");
    std::string task = r->task, preset = r->preset;
    if (task != "i2v" && task != "t2v")
        return failure(error, capacity, "task must be i2v or t2v");
    if (preset != "quality" && preset != "fast12")
        return failure(error, capacity, "preset must be quality or fast12");
    if (preset == "fast12" && task != "i2v")
        return failure(error, capacity, "fast12 requires I2V");
    bool image = r->image && *r->image, vision = r->vision_pixels && *r->vision_pixels;
    if ((task == "i2v") != image || (task == "i2v") != vision)
        return failure(error, capacity, "I2V requires prepared image and vision pixels; T2V omits both");
    if (r->width != 480 || r->height != 848 || r->frames != 81)
        return failure(error, capacity, "initial native profile is 480x848, 81 frames");
    if (r->seed < 0 || std::strlen(r->prompt) > 4096 ||
        (r->negative_prompt && std::strlen(r->negative_prompt) > 4096))
        return failure(error, capacity, "invalid seed or prompt length");
    return 0;
}
hv15n_context *hv15n_load(const hv15n_config *c, char *error, size_t capacity) {
    try {
        require(c && c->model_dir, "model directory is required");
        require(c->gemm && (std::string(c->gemm) == "repo" || std::string(c->gemm) == "cublas"),
                "gemm must be repo or cublas");
        require(c->gemm_fallback &&
                    (std::string(c->gemm_fallback) == "error" || std::string(c->gemm_fallback) == "cublas"),
                "fallback must be error or cublas");
        require(c->device >= 0, "device must be nonnegative");
        auto ctx = std::make_unique<hv15n_context>();
        ctx->root = fs::canonical(c->model_dir);
        Json manifest(ctx->root / "model.json");
        require(string(manifest.value.get(), "schema") == "hunyuan_video15.model.v1",
                "unsupported model manifest");
        ctx->gpu = std::make_unique<Gpu>(c->device, c->vram_budget_mib, std::string(c->gemm) == "cublas",
                                         std::string(c->gemm_fallback) == "cublas");
        return ctx.release();
    } catch (const std::exception &e) {
        failure(error, capacity, e.what());
        return nullptr;
    }
}
int hv15n_generate(hv15n_context *ctx, const hv15n_request *r, const hv15n_callbacks *cb, char *error,
                   size_t capacity) {
    if (!ctx || hv15n_validate(r, error, capacity))
        return -1;
    if (ctx->busy.exchange(true))
        return failure(error, capacity, "context is already generating");
    struct Release {
        hv15n_context *ctx;
        ~Release() {
            ctx->gpu->cancel_check = {};
            ctx->busy = false;
        }
    } release{ctx};
    auto started = std::chrono::steady_clock::now();
    try {
        auto &g = *ctx->gpu;
        g.check(cuCtxSetCurrent(g.context), "activate context");
        g.repo_gemm = g.blas_gemm = g.fallback_gemm = g.attention_calls = 0;
        g.peak = g.allocated;
        g.cancelled = false;
        g.cancel_check = [cb]() { return cb && cb->cancelled && cb->cancelled(cb->user); };
        Json manifest(ctx->root / "model.json");
        auto root = manifest.value.get();
        auto components = field(root, "components");
        auto checkpoint = std::string(r->preset) + "_" + r->task;
        auto denoiser = relative_file(ctx->root, string(field(root, "checkpoints"), checkpoint.c_str()));
        auto component = [&](const char *key) { return relative_file(ctx->root, string(components, key)); };
        fs::path dump = r->dump_dir ? r->dump_dir : "";
        auto notify = [&](int step, int total) {
            if (cb && cb->cancelled && cb->cancelled(cb->user))
                g.cancelled = true;
            g.poll();
            if (cb && cb->progress)
                cb->progress(step, total, cb->user);
        };
        notify(0, 0);
        Tensor text, negative, glyph, vision, portrait;
        {
            Tokenizer tokenizer(component("tokenizer"));
            Weights weights(component("qwen"));
            text = qwen(g, weights, tokenizer, r->prompt);
            if (std::string(r->preset) == "quality")
                negative = qwen(g, weights, tokenizer, r->negative_prompt ? r->negative_prompt : "");
        }
        g.dump(text, dump, "qwen_hidden");
        if (negative.pointer)
            g.dump(negative, dump, "qwen_negative_hidden");
        notify(0, 0);
        {
            Weights weights(component("byt5"));
            glyph = byt5(g, weights, r->prompt);
        }
        if (glyph.pointer)
            g.dump(glyph, dump, "byt5_hidden");
        else if (!dump.empty())
            g.dump(g.upload(std::vector<float>(1472, 0.f), {1, 1472}), dump, "byt5_hidden");
        if (std::string(r->task) == "i2v") {
            require(string(root, "vision_profile") == "google_siglip_so400m_14_384",
                    "unsupported vision profile");
            {
                Weights weights(component("vision"));
                vision = siglip(g, weights, r->vision_pixels);
            }
            g.dump(vision, dump, "siglip_hidden");
            int width, height, channels;
            std::unique_ptr<unsigned char, decltype(&stbi_image_free)> image(
                stbi_load(r->image, &width, &height, &channels, 3), stbi_image_free);
            require(bool(image) && width == r->width && height == r->height,
                    "prepared portrait size mismatch");
            std::vector<float> pixels(size_t(width) * height * 3);
            for (size_t i = 0; i < pixels.size(); i++)
                pixels[i] = image.get()[i] / 127.5f - 1.f;
            {
                Weights weights(component("vae"));
                portrait = vae_tiled(g, weights, g.upload(pixels, {1, height, width, 3}), true);
            }
            g.dump(portrait, dump, "vae_encoded", true);
        }
        const std::vector<int> shape = {21, 53, 30, 32};
        size_t count = product(shape);
        std::vector<float> noise(count);
        if (r->noise_file && *r->noise_file) {
            auto canonical = read_f32(r->noise_file, count);
            for (int row = 0; row < 21 * 53 * 30; row++)
                for (int c = 0; c < 32; c++)
                    noise[size_t(row) * 32 + c] = canonical[size_t(c) * 21 * 53 * 30 + row];
        } else {
            std::mt19937_64 rng(uint64_t(r->seed));
            for (size_t i = 0; i < count; i += 2) {
                double u = (double(rng() >> 11) + .5) / 9007199254740992.0,
                       v = (double(rng() >> 11) + .5) / 9007199254740992.0;
                double radius = std::sqrt(-2 * std::log(u)), angle = 6.283185307179586 * v;
                noise[i] = float(radius * std::cos(angle));
                if (i + 1 < count)
                    noise[i + 1] = float(radius * std::sin(angle));
            }
        }
        auto latent = g.upload(noise, shape);
        g.dump(latent, dump, "noise_input", true);
        std::vector<float> conditioning(size_t(21 * 53 * 30) * 33, 0.f);
        if (portrait.pointer) {
            auto first = g.download(portrait);
            require(portrait.shape == std::vector<int>({1, 53, 30, 32}), "portrait latent shape mismatch");
            for (int row = 0; row < 53 * 30; row++) {
                std::copy_n(first.data() + size_t(row) * 32, 32, conditioning.data() + size_t(row) * 33);
                conditioning[size_t(row) * 33 + 32] = 1.f;
            }
        }
        auto condition = g.upload(conditioning, {21, 53, 30, 33});
        int steps = std::string(r->preset) == "fast12" ? 12 : 50;
        auto sigmas = schedule(steps, steps == 12 ? 7.f : 5.f);
        {
            Weights weights(denoiser);
            for (int step = 0; step < steps; step++) {
                notify(step, steps);
                Tensor positive,uncond;
                if(steps==50) {
                    auto predictions=dit_pair(g,weights,latent,condition,text,negative,glyph,vision,
                                              sigmas[step]*1000.f,sigmas[step+1]*1000.f);
                    positive=std::move(predictions.first);uncond=std::move(predictions.second);
                } else positive=dit(g,weights,latent,condition,text,glyph,vision,
                                    sigmas[step]*1000.f,sigmas[step+1]*1000.f);
                if(step==0)g.dump(positive,dump,"dit_first",true);
                if(steps==50) {
                    auto difference = g.op(positive, 1, &uncond, nullptr, -1.f);
                    positive = g.op(uncond, 1, &difference, nullptr, 6.f);
                }
                latent = g.op(latent, 1, &positive, nullptr, sigmas[step + 1] - sigmas[step]);
                if (!dump.empty())
                    g.dump(latent, dump, "latent_step_" + std::to_string(step), true);
            }
        }
        g.dump(latent, dump, "latent_final", true);
        notify(steps, steps);
        Tensor decoded;
        {
            Weights weights(component("vae"));
            decoded = vae_tiled(g, weights, latent, false);
        }
        g.dump(decoded, dump, "vae_decoded", true);
        require(decoded.shape == std::vector<int>({81, 848, 480, 3}), "decoded video shape mismatch");
        auto pixels = g.download(decoded);
        std::vector<unsigned char> frame(size_t(848) * 480 * 3);
        for (int f = 0; f < 81; f++) {
            notify(steps, steps);
            for (size_t i = 0; i < frame.size(); i++) {
                float value = (pixels[size_t(f) * frame.size() + i] + 1.f) * 127.5f;
                frame[i] = static_cast<unsigned char>(std::nearbyint(std::clamp(value, 0.f, 255.f)));
            }
            require(!cb || !cb->frame || cb->frame(f, 480, 848, frame.data(), cb->user) == 0,
                    "frame callback failed");
        }
        ctx->metrics = g.metrics();
        auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        ctx->metrics.pop_back();
        ctx->metrics +=
            ",\"wall_seconds\":" + std::to_string(elapsed) + ",\"rng\":\"mt19937_64_box_muller_v1\"}";
        return 0;
    } catch (const std::exception &e) {
        ctx->metrics = ctx->gpu->metrics();
        return failure(error, capacity, e.what());
    }
}
const char *hv15n_metrics(const hv15n_context *ctx) {
    return ctx ? ctx->metrics.c_str() : "{}";
}
void hv15n_cancel(hv15n_context *ctx) {
    if (ctx)
        ctx->gpu->cancelled = true;
}
void hv15n_free(hv15n_context *ctx) {
    delete ctx;
}
}
