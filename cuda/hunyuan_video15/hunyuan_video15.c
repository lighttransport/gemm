#include "hunyuan_video15.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "stable-diffusion.h"
#include "progress.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../../common/stb_image.h"

extern bool hv15_set_siglip_pixels(const char *path);
extern uint64_t hv15_cuda_attention_calls(void);

struct hv15_context { sd_ctx_t *native; const hv15_callbacks *callbacks; int sample_steps; };
static hv15_context *active;
static int fail(char *error, unsigned cap, const char *message) {
    if (cap) snprintf(error, cap, "%s", message);
    return -1;
}
void hv15_defaults(hv15_request *r) {
    *r = (hv15_request){"i2v", "quality", "", "", NULL, NULL, 480, 848, 81, 42};
}
int hv15_validate(const hv15_request *r, char *error, unsigned cap) {
    if (!r || !r->task || !r->preset || !r->prompt || !*r->prompt)
        return fail(error, cap, "task, preset and a nonempty prompt are required");
    if (strcmp(r->task, "i2v") && strcmp(r->task, "t2v"))
        return fail(error, cap, "task must be i2v or t2v");
    if (strcmp(r->preset, "quality") && strcmp(r->preset, "fast12"))
        return fail(error, cap, "preset must be quality or fast12");
    if (!strcmp(r->preset, "fast12") && strcmp(r->task, "i2v"))
        return fail(error, cap, "fast12 requires the distilled I2V checkpoint");
    if ((!strcmp(r->task, "i2v")) != (r->image != NULL && *r->image != 0))
        return fail(error, cap, "I2V requires an image; T2V must omit it");
    if (r->frames != 81 && r->frames != 121)
        return fail(error, cap, "supported frame counts are 81 and 121");
    if (!((r->width == 480 && r->height == 848) ||
          (r->width == 848 && r->height == 480) ||
          (r->width == 640 && r->height == 640)))
        return fail(error, cap, "use a 480p bucket: 480x848, 848x480 or 640x640");
    if (r->seed < 0) return fail(error, cap, "seed must be non-negative");
    if (strlen(r->prompt) > 4096) return fail(error, cap, "prompt exceeds 4096 bytes");
    return 0;
}
static void progress(int step, int total, float seconds, void *user) {
    (void)seconds;
    hv15_context *ctx = user;
    const hv15_callbacks *cb = ctx->callbacks;
    if (!cb) return;
    if (cb->cancelled && cb->cancelled(cb->user)) sd_cancel_generation(ctx->native, SD_CANCEL_ALL);
    fprintf(stderr, "NATIVE_PROGRESS %d %d\n", step, total);
}
static void sample_progress(int step, int total, float seconds, void *user) {
    (void)seconds;
    hv15_context *ctx = user;
    const hv15_callbacks *cb = ctx->callbacks;
    if (!cb) return;
    if (cb->cancelled && cb->cancelled(cb->user)) sd_cancel_generation(ctx->native, SD_CANCEL_ALL);
    if (cb->progress && total == ctx->sample_steps && step >= 0 && step <= total)
        cb->progress(step, total, cb->user);
}
static void log_message(enum sd_log_level_t level, const char *message, void *user) {
    (void)user;
    if (level >= SD_LOG_INFO) fprintf(stderr, "%s", message);
}
hv15_context *hv15_load(const hv15_model *m, char *error, unsigned cap) {
    if (active) { fail(error, cap, "only one native context may be active per process"); return NULL; }
    if (!m || !m->allow_experimental) {
        fail(error, cap, "full pipeline parity is unverified; explicit experimental opt-in required");
        return NULL;
    }
    if (m->device < 0 || m->vram_budget_mib < 4096 || m->vram_budget_mib > 14336) {
        fail(error, cap, "device must be non-negative; VRAM budget must be 4096..14336 MiB"); return NULL;
    }
    const char *paths[] = {m->diffusion, m->vae, m->qwen, m->byt5, m->vision, m->tokenizer};
    for (unsigned i = 0; i < sizeof(paths)/sizeof(paths[0]); ++i) {
        if (!paths[i] || access(paths[i], R_OK)) {
            fail(error, cap, "all six component files must exist and be readable"); return NULL;
        }
    }
    hv15_context *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) { fail(error, cap, "context allocation failed"); return NULL; }
    char backend[32], budget[32];
    snprintf(backend, sizeof(backend), "cuda%d", m->device);
    /* Managed allocations exclude driver contexts: leave 1 GiB inside the requested ceiling. */
    snprintf(budget, sizeof(budget), "%.6f", (m->vram_budget_mib - 3072) / 1024.0);
    sd_ctx_params_t p;
    sd_ctx_params_init(&p);
    p.diffusion_model_path = m->diffusion; p.vae_path = m->vae;
    p.llm_path = m->qwen; p.t5xxl_path = m->byt5;
    p.clip_vision_path = m->vision; p.tokenizer = m->tokenizer;
    p.backend = backend; p.params_backend = "cpu"; p.max_vram = budget;
    p.enable_mmap = true; p.auto_fit = true;
    p.flash_attn = true; p.diffusion_flash_attn = true;
    p.conditioning_cache_size = 0; p.n_threads = m->threads > 0 ? m->threads : 8;
    p.rng_type = CUDA_RNG; p.sampler_rng_type = CUDA_RNG;
    sd_set_log_callback(log_message, NULL);
    sd_set_progress_callback(progress, ctx);
    hv15_set_sample_progress_callback(sample_progress, ctx);
    ctx->native = new_sd_ctx(&p);
    if (!ctx->native || !sd_ctx_supports_video_generation(ctx->native) ||
        !strstr(sd_get_model_version_name(ctx->native), "Hunyuan")) {
        hv15_free(ctx); fail(error, cap, "could not load a Hunyuan video model"); return NULL;
    }
    active = ctx;
    return ctx;
}
int hv15_generate(hv15_context *ctx, const hv15_request *r, const hv15_callbacks *cb,
                  char *error, unsigned cap) {
    if (!ctx || hv15_validate(r, error, cap)) return -1;
    if (cb && cb->cancelled && cb->cancelled(cb->user)) return fail(error, cap, "cancelled");
    sd_vid_gen_params_t p;
    sd_vid_gen_params_init(&p);
    p.prompt = r->prompt; p.negative_prompt = r->negative_prompt ? r->negative_prompt : "";
    p.width = r->width; p.height = r->height; p.video_frames = r->frames;
    p.fps = 24; p.seed = r->seed; p.clip_skip = 2; p.strength = 1.0f;
    p.sample_params.sample_method = EULER_SAMPLE_METHOD;
    p.sample_params.sample_steps = !strcmp(r->preset, "fast12") ? 12 : 50;
    p.sample_params.guidance.txt_cfg = !strcmp(r->preset, "fast12") ? 1.0f : 6.0f;
    p.sample_params.flow_shift = !strcmp(r->preset, "fast12") ? 7.0f : 5.0f;
    /* Explicit continuous flow schedule, including terminal zero. */
    float sigmas[51];
    int steps = p.sample_params.sample_steps;
    for (int i = 0; i <= steps; ++i) {
        float t = 1.0f - (float)i / steps, shift = p.sample_params.flow_shift;
        sigmas[i] = shift * t / (1.0f + (shift - 1.0f) * t);
    }
    p.sample_params.custom_sigmas = sigmas; p.sample_params.custom_sigmas_count = steps + 1;
    p.vae_tiling_params.enabled = true; p.vae_tiling_params.temporal_tiling = false;
    p.vae_tiling_params.tile_size_w = 128; p.vae_tiling_params.tile_size_h = 128;
    p.vae_tiling_params.target_overlap = 0.25f;
    p.cache.mode = SD_CACHE_DISABLED;
    if (r->image) {
        int w, h, channels;
        unsigned char *pixels = stbi_load(r->image, &w, &h, &channels, 3);
        if (!pixels) return fail(error, cap, "could not decode input portrait");
        p.init_image = (sd_image_t){(uint32_t)w, (uint32_t)h, 3, pixels};
    }
    if (!strcmp(r->task, "i2v") && (!r->vision_pixels || !hv15_set_siglip_pixels(r->vision_pixels))) {
        stbi_image_free(p.init_image.data);
        return fail(error, cap, "I2V requires prepared SigLIP pixels; use native_generate.py");
    }
    ctx->sample_steps = steps;
    ctx->callbacks = cb;
    sd_cancel_generation(ctx->native, SD_CANCEL_RESET);
    sd_image_t *frames = NULL; int count = 0, fps = 0;
    sd_audio_t *audio = NULL;
    bool ok = generate_video(ctx->native, &p, &frames, &count, &audio, &fps);
    fprintf(stderr, "HV15 precise attention calls: %llu\n",
            (unsigned long long)hv15_cuda_attention_calls());
    stbi_image_free(p.init_image.data);
    int cancelled = cb && cb->cancelled && cb->cancelled(cb->user);
    int result = ok && !cancelled && count == r->frames && fps == 24 ? 0 : -1;
    if (!result) for (int i = 0; i < count; ++i) {
        if (frames[i].channel != 3 || frames[i].width != (unsigned)r->width ||
            frames[i].height != (unsigned)r->height ||
            (cb && cb->frame && cb->frame(i, r->width, r->height, frames[i].data, cb->user))) {
            result = -1; break;
        }
    }
    for (int i = 0; i < count; ++i) free(frames[i].data);
    free(frames); if (audio) free_sd_audio(audio);
    ctx->callbacks = NULL;
    hv15_set_siglip_pixels(NULL);
    if (result) return fail(error, cap, cancelled ? "cancelled" : "generation or frame callback failed");
    return 0;
}
void hv15_cancel(hv15_context *ctx) { if (ctx && ctx->native) sd_cancel_generation(ctx->native, SD_CANCEL_ALL); }
void hv15_free(hv15_context *ctx) {
    if (!ctx) return;
    if (ctx->native) free_sd_ctx(ctx->native);
    sd_set_progress_callback(NULL, NULL);
    hv15_set_sample_progress_callback(NULL, NULL);
    if (active == ctx) active = NULL;
    free(ctx);
}
