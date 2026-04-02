/*
 * test_cuda_flux2.c - CUDA Flux.2 Klein end-to-end test
 *
 * Modes:
 *   --test-init    : Initialize CUDA, compile kernels
 *   --test-load    : Load DiT + VAE weights
 *   --test-dit     : Run single DiT step
 *   --test-vae     : Run VAE decoder
 *   --generate     : Full text-to-image pipeline
 *
 * Build:
 *   make test_cuda_flux2
 *   (or: cc -O2 -mavx2 -mfma -I../../common -I.. -o test_cuda_flux2 \
 *        test_cuda_flux2.c ../llm/cuda_llm_runner.c ../cuew.c -lm -ldl -lpthread)
 */

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#define SAFETENSORS_IMPLEMENTATION
#define GGUF_LOADER_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION
#define BPE_TOKENIZER_IMPLEMENTATION
#define TRANSFORMER_IMPLEMENTATION
#define QIMG_SCHEDULER_IMPLEMENTATION
#define FLUX2_DIT_IMPLEMENTATION
#define FLUX2_VAE_IMPLEMENTATION
#define FLUX2_TEXT_ENCODER_IMPLEMENTATION
#define CUDA_FLUX2_RUNNER_IMPLEMENTATION

#include "../../common/gguf_loader.h"
#include "../../common/ggml_dequant.h"
#include "../../common/bpe_tokenizer.h"
#include "../../common/transformer.h"
#include "../../common/qwen_image_scheduler.h"
#include "../../common/flux2_klein_dit.h"
#include "../../common/flux2_klein_vae.h"
#include "../llm/cuda_llm_runner.h"
#include "../../common/flux2_klein_text_encoder.h"
#include "cuda_flux2_runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ---- Default weight paths ---- */
static const char *DEFAULT_DIT = "/mnt/disk01/models/klein2-4b/diffusion_models/flux-2-klein-4b-fp8.safetensors";
static const char *DEFAULT_VAE = "/mnt/disk01/models/klein2-4b/vae/flux2-vae.safetensors";
static const char *DEFAULT_ENC = "/mnt/disk01/models/klein2-4b/text_encoder";
static const char *DEFAULT_TOK = "/mnt/disk01/models/Qwen3-VL-4B-Instruct-GGUF/Qwen3VL-4B-Instruct-Q8_0.gguf";

/* ---- PRNG: Box-Muller with pair caching ---- */
static uint64_t rng_state = 42;
static int rng_cached_valid = 0;
static float rng_cached = 0.0f;

static float randn(void) {
    if (rng_cached_valid) { rng_cached_valid = 0; return rng_cached; }
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u1 = (double)(rng_state >> 11) / (double)(1ULL << 53);
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u2 = (double)(rng_state >> 11) / (double)(1ULL << 53);
    if (u1 < 1e-10) u1 = 1e-10;
    double rv = sqrt(-2.0 * log(u1));
    double theta = 2.0 * 3.14159265358979323846 * u2;
    rng_cached = (float)(rv * sin(theta));
    rng_cached_valid = 1;
    return (float)(rv * cos(theta));
}

static void save_ppm(const char *path, const float *rgb, int h, int w) {
    FILE *fp = fopen(path, "wb");
    if (!fp) return;
    fprintf(fp, "P6\n%d %d\n255\n", w, h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            uint8_t px[3];
            for (int c = 0; c < 3; c++) {
                float v = rgb[(size_t)c * h * w + y * w + x] * 0.5f + 0.5f;
                if (v < 0.0f) { v = 0.0f; } if (v > 1.0f) { v = 1.0f; }
                px[c] = (uint8_t)(v * 255.0f + 0.5f);
            }
            fwrite(px, 1, 3, fp);
        }
    fclose(fp);
    fprintf(stderr, "Saved %s (%dx%d)\n", path, w, h);
}

/* ---- Patchify / unpatchify (same as cpu test) ---- */

static void flux2_patchify(float *out, const float *latent,
                            int lc, int lat_h, int lat_w, int ps) {
    int ph = lat_h / ps, pw = lat_w / ps, pin = lc * ps * ps;
    for (int r = 0; r < ph; r++)
        for (int c = 0; c < pw; c++) {
            float *tok = out + ((size_t)r * pw + c) * pin;
            for (int ch = 0; ch < lc; ch++)
                for (int pr = 0; pr < ps; pr++)
                    for (int pc = 0; pc < ps; pc++)
                        tok[ch*ps*ps + pr*ps + pc] =
                            latent[(size_t)ch*lat_h*lat_w + (r*ps+pr)*lat_w + (c*ps+pc)];
        }
}

static void flux2_unpatchify(float *latent, const float *tok,
                              int lc, int lat_h, int lat_w, int ps) {
    int ph = lat_h/ps, pw = lat_w/ps, pin = lc*ps*ps;
    for (int r = 0; r < ph; r++)
        for (int c = 0; c < pw; c++) {
            const float *t = tok + ((size_t)r*pw+c)*pin;
            for (int ch = 0; ch < lc; ch++)
                for (int pr = 0; pr < ps; pr++)
                    for (int pc = 0; pc < ps; pc++)
                        latent[(size_t)ch*lat_h*lat_w+(r*ps+pr)*lat_w+(c*ps+pc)] =
                            t[ch*ps*ps+pr*ps+pc];
        }
}

/* ---- Flux.2 Klein scheduler helpers ---- */

static void flux2_sched_distilled(qimg_scheduler *s, int n_steps) {
    qimg_sched_init(s);
    s->base_shift = 0.5f; s->max_shift = 1.15f;
    s->base_image_seq_len = 256; s->max_image_seq_len = 4096;
    qimg_sched_set_timesteps_comfyui(s, n_steps, 1.0f, 1.0f);
}

static void flux2_sched_base(qimg_scheduler *s, int n_steps, int n_img) {
    qimg_sched_init(s);
    s->base_shift = 0.5f; s->max_shift = 1.15f;
    s->base_image_seq_len = 256; s->max_image_seq_len = 4096;
    qimg_sched_set_timesteps(s, n_steps, n_img);
}

/* ---- Test modes ---- */

static int test_init(void) {
    fprintf(stderr, "=== CUDA Init Test ===\n");
    cuda_flux2_runner *r = cuda_flux2_init(0, 2);
    if (!r) return 1;
    fprintf(stderr, "CUDA init OK\n");
    cuda_flux2_free(r);
    return 0;
}

static int test_load(const char *dit_path, const char *vae_path) {
    fprintf(stderr, "=== CUDA Load Test ===\n");
    cuda_flux2_runner *r = cuda_flux2_init(0, 1);
    if (!r) return 1;
    if (cuda_flux2_load_dit(r, dit_path) != 0) { cuda_flux2_free(r); return 1; }
    if (cuda_flux2_load_vae(r, vae_path) != 0) { cuda_flux2_free(r); return 1; }
    fprintf(stderr, "Load OK\n");
    cuda_flux2_free(r);
    return 0;
}

static int test_dit(const char *dit_path, int lat_h, int lat_w) {
    fprintf(stderr, "=== CUDA DiT Step Test ===\n");
    cuda_flux2_runner *r = cuda_flux2_init(0, 1);
    if (!r) return 1;
    if (cuda_flux2_load_dit(r, dit_path) != 0) { cuda_flux2_free(r); return 1; }

    int ps = 2;
    int n_img = (lat_h/ps) * (lat_w/ps);
    int pin = r->pin;
    int n_txt = 8, txt_dim = r->txt_dim;

    float *img_tok = (float *)calloc((size_t)n_img * pin, sizeof(float));
    float *txt_tok = (float *)calloc((size_t)n_txt * txt_dim, sizeof(float));
    float *vel_out = (float *)malloc((size_t)n_img * pin * sizeof(float));

    rng_state = 12345;
    for (int i = 0; i < n_img*pin; i++) img_tok[i] = randn() * 0.1f;
    for (int i = 0; i < n_txt*txt_dim; i++) txt_tok[i] = randn() * 0.1f;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    cuda_flux2_dit_step(r, img_tok, n_img, txt_tok, n_txt, 750.0f, 0.0f, vel_out);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec-t0.tv_sec) + (t1.tv_nsec-t0.tv_nsec)*1e-9;
    fprintf(stderr, "DiT step: %.2f s (n_img=%d, pin=%d)\n", dt, n_img, pin);

    float mn=vel_out[0], mx=vel_out[0];
    for (int i=0;i<n_img*pin;i++) { if(vel_out[i]<mn) mn=vel_out[i]; if(vel_out[i]>mx) mx=vel_out[i]; }
    fprintf(stderr, "GPU velocity: min=%.4f max=%.4f\n", mn, mx);

    /* CPU comparison (also skip single blocks to match GPU) */
    int saved_sgl = r->dit->n_single_blocks;
    int saved_dbl = r->dit->n_double_blocks;
    float *cpu_out = (float *)malloc((size_t)n_img * pin * sizeof(float));
    flux2_dit_forward(cpu_out, img_tok, n_img, txt_tok, n_txt, 750.0f, r->dit, 1);
    r->dit->n_single_blocks = saved_sgl;
    r->dit->n_double_blocks = saved_dbl;
    mn=cpu_out[0]; mx=cpu_out[0];
    for (int i=0;i<n_img*pin;i++) { if(cpu_out[i]<mn) mn=cpu_out[i]; if(cpu_out[i]>mx) mx=cpu_out[i]; }
    fprintf(stderr, "CPU velocity: min=%.4f max=%.4f\n", mn, mx);
    float max_diff = 0, sum_diff = 0;
    for (int i=0;i<n_img*pin;i++) { float d=fabsf(vel_out[i]-cpu_out[i]); if(d>max_diff) max_diff=d; sum_diff+=d; }
    fprintf(stderr, "GPU vs CPU: max_diff=%.6f mean_diff=%.6f\n", max_diff, sum_diff/(n_img*pin));
    for (int i=0;i<5;i++)
        fprintf(stderr, "  [%d] GPU=%.6f CPU=%.6f\n", i, vel_out[i], cpu_out[i]);
    free(cpu_out);

    free(img_tok); free(txt_tok); free(vel_out);
    cuda_flux2_free(r); return 0;
}

static int run_generate(const char *dit_path, const char *vae_path,
                         const char *enc_path, const char *tok_path,
                         const char *prompt,
                         int out_h, int out_w, int n_steps,
                         uint64_t seed, int is_distilled, float cfg_scale,
                         int use_gpu_enc, int device_id) {
    fprintf(stderr, "\n=== Flux.2 Klein GPU Pipeline ===\n");
    fprintf(stderr, "Prompt: '%s'\n", prompt);
    fprintf(stderr, "Output: %dx%d, %d steps, seed=%llu, distilled=%d, cfg=%.1f\n",
            out_w, out_h, n_steps, (unsigned long long)seed, is_distilled, cfg_scale);

    struct timespec t_start, t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    /* Text encoding */
    fprintf(stderr, "\n[1/4] Text encoding (%s)...\n", use_gpu_enc ? "GPU" : "CPU");
    clock_gettime(CLOCK_MONOTONIC, &t0);

    flux2_text_enc *enc = use_gpu_enc
        ? flux2_text_enc_load_gpu(enc_path, device_id)
        : flux2_text_enc_load_safetensors(enc_path, tok_path);
    if (!enc) return 1;

    int n_txt = 0;
    int enc_embd = enc->n_embd;
    float *txt_hidden = flux2_text_enc_encode(enc, prompt, &n_txt);
    flux2_text_enc_free(enc);
    if (!txt_hidden) return 1;

    clock_gettime(CLOCK_MONOTONIC, &t1);
    fprintf(stderr, "Text enc: %.1f s (%d tokens × %d)\n",
            (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9, n_txt, enc_embd);

    /* Init CUDA runner */
    fprintf(stderr, "\n[2/4] Init CUDA + load DiT + VAE...\n");
    clock_gettime(CLOCK_MONOTONIC, &t0);

    cuda_flux2_runner *r = cuda_flux2_init(device_id, 1);
    if (!r) { free(txt_hidden); return 1; }
    if (cuda_flux2_load_dit(r, dit_path) != 0) { free(txt_hidden); cuda_flux2_free(r); return 1; }
    if (cuda_flux2_load_vae(r, vae_path) != 0) { free(txt_hidden); cuda_flux2_free(r); return 1; }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    fprintf(stderr, "Init+load: %.1f s\n",
            (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9);

    /* Latent setup */
    int lat_h = out_h / 8, lat_w = out_w / 8;
    int lc = FLUX2_VAE_LATENT_CHANNELS;
    int ps = (r->pin == lc * 4) ? 2 : 1;
    int n_img = (lat_h / ps) * (lat_w / ps);
    int pin = r->pin;

    fprintf(stderr, "Latent: [%d, %d, %d], ps=%d, n_img=%d, pin=%d, n_txt=%d, txt_dim=%d\n",
            lc, lat_h, lat_w, ps, n_img, pin, n_txt, r->txt_dim);

    rng_state = seed;
    size_t lat_sz = (size_t)lc * lat_h * lat_w;
    float *latent = (float *)malloc(lat_sz * sizeof(float));
    for (size_t i = 0; i < lat_sz; i++) latent[i] = randn();

    float *img_tok = (float *)malloc((size_t)n_img * pin * sizeof(float));
    float *vel_out = (float *)malloc((size_t)n_img * pin * sizeof(float));

    qimg_scheduler sched;
    if (is_distilled) flux2_sched_distilled(&sched, n_steps);
    else              flux2_sched_base(&sched, n_steps, n_img);

    /* Denoising loop */
    fprintf(stderr, "\n[3/4] Denoising (%d steps)...\n", n_steps);
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int step = 0; step < n_steps; step++) {
        float t_sigma = sched.timesteps[step];
        struct timespec ts0, ts1;
        clock_gettime(CLOCK_MONOTONIC, &ts0);

        flux2_patchify(img_tok, latent, lc, lat_h, lat_w, ps);

        if (is_distilled || cfg_scale <= 1.0f) {
            cuda_flux2_dit_step(r, img_tok, n_img, txt_hidden, n_txt,
                                t_sigma, 0.0f, vel_out);
            /* Velocity stats */
            float vmn=vel_out[0], vmx=vel_out[0]; double vsum=0;
            for(int i=0;i<n_img*pin;i++){if(vel_out[i]<vmn)vmn=vel_out[i];if(vel_out[i]>vmx)vmx=vel_out[i];vsum+=vel_out[i];}
            fprintf(stderr, "    vel: min=%.4f max=%.4f mean=%.6f\n", vmn, vmx, vsum/(n_img*pin));
        } else {
            float *txt_uncond = (float *)calloc((size_t)n_txt * r->txt_dim, sizeof(float));
            float *vel_uncond = (float *)malloc((size_t)n_img * pin * sizeof(float));
            cuda_flux2_dit_step(r, img_tok, n_img, txt_uncond, n_txt,
                                t_sigma, 0.0f, vel_uncond);
            cuda_flux2_dit_step(r, img_tok, n_img, txt_hidden, n_txt,
                                t_sigma, 0.0f, vel_out);
            for (int i = 0; i < n_img * pin; i++)
                vel_out[i] = vel_uncond[i] + cfg_scale * (vel_out[i] - vel_uncond[i]);
            free(txt_uncond); free(vel_uncond);
        }

        float *vel_lat = (float *)calloc(lat_sz, sizeof(float));
        flux2_unpatchify(vel_lat, vel_out, lc, lat_h, lat_w, ps);
        qimg_sched_step(latent, vel_lat, (int)lat_sz, step, &sched);
        free(vel_lat);

        clock_gettime(CLOCK_MONOTONIC, &ts1);
        double sdt = (ts1.tv_sec-ts0.tv_sec)+(ts1.tv_nsec-ts0.tv_nsec)*1e-9;
        fprintf(stderr, "  step %d/%d  sigma=%.4f  %.1f s\n", step+1, n_steps, t_sigma, sdt);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    fprintf(stderr, "Denoising: %.1f s\n",
            (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9);

    free(img_tok); free(vel_out);

    /* VAE decode */
    fprintf(stderr, "\n[4/4] VAE decoding...\n");
    clock_gettime(CLOCK_MONOTONIC, &t0);

    float *rgb = (float *)malloc((size_t)3 * out_h * out_w * sizeof(float));
    cuda_flux2_vae_decode(r, latent, lat_h, lat_w, rgb);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    fprintf(stderr, "VAE: %.1f s\n",
            (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9);

    save_ppm("cuda_flux2_output.ppm", rgb, out_h, out_w);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double total = (t1.tv_sec-t_start.tv_sec)+(t1.tv_nsec-t_start.tv_nsec)*1e-9;
    fprintf(stderr, "\nTotal: %.1f s\n", total);

    free(rgb); free(latent); free(txt_hidden);
    cuda_flux2_free(r);
    return 0;
}

int main(int argc, char **argv) {
    const char *dit_path = DEFAULT_DIT;
    const char *vae_path = DEFAULT_VAE;
    const char *enc_path = DEFAULT_ENC;
    const char *tok_path = DEFAULT_TOK;
    const char *prompt = "a red apple on a white table";
    const char *mode = NULL;

    int out_h = 256, out_w = 256, n_steps = 4;
    int is_distilled = 1, use_gpu_enc = 0, device_id = 0;
    float cfg_scale = 1.0f;
    uint64_t seed = 42;
    int verbose = 1;
    (void)verbose;

    for (int i = 1; i < argc; i++) {
        if      (strcmp(argv[i], "--test-init") == 0) mode = "init";
        else if (strcmp(argv[i], "--test-load") == 0) mode = "load";
        else if (strcmp(argv[i], "--test-dit")  == 0) mode = "dit";
        else if (strcmp(argv[i], "--test-vae")  == 0) mode = "vae";
        else if (strcmp(argv[i], "--generate")  == 0) mode = "gen";
        else if (strcmp(argv[i], "--base")       == 0) { is_distilled = 0; n_steps = 20; }
        else if (strcmp(argv[i], "--distilled")  == 0) { is_distilled = 1; n_steps = 4; }
        else if (strcmp(argv[i], "--gpu-enc")    == 0) use_gpu_enc = 1;
        else if (strcmp(argv[i], "--dit")    == 0 && i+1<argc) dit_path = argv[++i];
        else if (strcmp(argv[i], "--vae")    == 0 && i+1<argc) vae_path = argv[++i];
        else if (strcmp(argv[i], "--enc")    == 0 && i+1<argc) enc_path = argv[++i];
        else if (strcmp(argv[i], "--tok")    == 0 && i+1<argc) tok_path = argv[++i];
        else if (strcmp(argv[i], "--prompt") == 0 && i+1<argc) prompt   = argv[++i];
        else if (strcmp(argv[i], "--height") == 0 && i+1<argc) out_h    = atoi(argv[++i]);
        else if (strcmp(argv[i], "--width")  == 0 && i+1<argc) out_w    = atoi(argv[++i]);
        else if (strcmp(argv[i], "--steps")  == 0 && i+1<argc) n_steps  = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed")   == 0 && i+1<argc) seed     = (uint64_t)atoll(argv[++i]);
        else if (strcmp(argv[i], "--cfg")    == 0 && i+1<argc) cfg_scale= (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--device") == 0 && i+1<argc) device_id= atoi(argv[++i]);
        else if (strcmp(argv[i], "--verbose")== 0 && i+1<argc) verbose  = atoi(argv[++i]);
    }

    if (!mode) {
        fprintf(stderr,
            "Usage: %s [--test-init|--test-load|--test-dit|--test-vae|--generate]\n"
            "          [--dit PATH] [--vae PATH] [--enc PATH]\n"
            "          [--prompt TEXT] [--height H] [--width W]\n"
            "          [--steps N] [--seed S] [--cfg SCALE]\n"
            "          [--base|--distilled] [--gpu-enc] [--device N]\n",
            argv[0]);
        return 1;
    }

    if (strcmp(mode, "init") == 0) return test_init();
    if (strcmp(mode, "load") == 0) return test_load(dit_path, vae_path);
    if (strcmp(mode, "dit")  == 0) return test_dit(dit_path, out_h/16, out_w/16);
    if (strcmp(mode, "gen")  == 0)
        return run_generate(dit_path, vae_path, enc_path, tok_path, prompt,
                            out_h, out_w, n_steps, seed, is_distilled,
                            cfg_scale, use_gpu_enc, device_id);

    fprintf(stderr, "Unknown mode: %s\n", mode);
    return 1;
}
