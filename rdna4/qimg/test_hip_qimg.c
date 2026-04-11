/*
 * test_hip_qimg.c - Qwen-Image end-to-end text-to-image test (HIP/RDNA4)
 *
 * Modes:
 *   --test-init    : Initialize HIP, compile kernels
 *   --test-dit     : Run single DiT step on GPU
 *   --test-vae     : VAE decode test
 *   --generate     : Full text-to-image pipeline (text encoder on CPU)
 *
 * Build:
 *   make
 */

#define SAFETENSORS_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION
#define GGUF_LOADER_IMPLEMENTATION
#define BPE_TOKENIZER_IMPLEMENTATION
#define TRANSFORMER_IMPLEMENTATION
#define QIMG_SCHEDULER_IMPLEMENTATION
#define QIMG_TEXT_ENCODER_IMPLEMENTATION

#include "../../common/safetensors.h"
#include "../../common/ggml_dequant.h"
#include "../../common/gguf_loader.h"
#include "../../common/bpe_tokenizer.h"
#include "../../common/transformer.h"
#include "../../common/qwen_image_scheduler.h"
#include "../llm/hip_llm_runner.h"  /* must be before text_encoder.h for GPU path */
#include "../../common/qwen_image_text_encoder.h"
#include "hip_qimg_runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/wait.h>

/* ---- PRNG: Box-Muller ---- */
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
    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * 3.14159265358979323846 * u2;
    rng_cached = (float)(r * sin(theta));
    rng_cached_valid = 1;
    return (float)(r * cos(theta));
}

static void save_ppm(const char *path, const float *rgb, int h, int w) {
    FILE *fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "Cannot write %s\n", path); return; }
    fprintf(fp, "P6\n%d %d\n255\n", w, h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            unsigned char px[3];
            for (int c = 0; c < 3; c++) {
                float v = rgb[c * h * w + y * w + x];
                v = v * 0.5f + 0.5f;  /* [-1,1] → [0,1] */
                v = v < 0 ? 0 : (v > 1 ? 1 : v);
                px[c] = (unsigned char)(v * 255.0f + 0.5f);
            }
            fwrite(px, 1, 3, fp);
        }
    fclose(fp);
    fprintf(stderr, "Saved %s (%dx%d)\n", path, w, h);
}

int main(int argc, char **argv) {
    const char *dit_path = NULL;
    const char *vae_path = NULL;
    const char *enc_path = NULL;
    const char *prompt = "a red apple on a white table";
    const char *out_path = "hip_qimg_out.ppm";
    const char *mode = NULL;
    int device_id = 0;
    int verbose = 1;
    int out_h = 256, out_w = 256, n_steps = 20;
    uint64_t seed = 42;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--test-init")) mode = "init";
        else if (!strcmp(argv[i], "--test-dit")) mode = "dit";
        else if (!strcmp(argv[i], "--test-vae")) mode = "vae";
        else if (!strcmp(argv[i], "--generate")) mode = "gen";
        else if (!strcmp(argv[i], "--dit") && i+1 < argc) dit_path = argv[++i];
        else if (!strcmp(argv[i], "--vae") && i+1 < argc) vae_path = argv[++i];
        else if (!strcmp(argv[i], "--enc") && i+1 < argc) enc_path = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i+1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--height") && i+1 < argc) out_h = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--width") && i+1 < argc) out_w = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps") && i+1 < argc) n_steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i+1 < argc) seed = (uint64_t)atoll(argv[++i]);
        else if (!strcmp(argv[i], "-o") && i+1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "-d") && i+1 < argc) device_id = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) verbose = 2;
        else {
            fprintf(stderr, "Usage: %s --test-init|--test-dit|--test-vae|--generate\n"
                    "  --dit <st>  --vae <st>  --enc <gguf>  --prompt <text>\n"
                    "  --height <h>  --width <w>  --steps <n>  --seed <s>\n"
                    "  [-o out.ppm] [-d dev] [-v]\n", argv[0]);
            return 1;
        }
    }

    if (!mode) {
        fprintf(stderr, "Error: specify a mode (--test-init, --test-dit, --test-vae, --generate)\n");
        return 1;
    }

    int lat_h = out_h / 8, lat_w = out_w / 8;

    fprintf(stderr, "=== HIP Qwen-Image: %s ===\n", mode);

    /* GPU text encoder runs in a forked child process to avoid ROCm HIPRTC
     * context corruption (hipSetDevice fails after hipModuleUnload in same process).
     * Child writes hidden states to a temp file, parent reads them back. */
    int n_txt_precomputed = 0;
    float *txt_precomputed = NULL;
    if (!strcmp(mode, "gen") && enc_path) {
        const char *tmp_path = "/tmp/qimg_text_hidden.bin";
        fprintf(stderr, "\n[1/3] Text conditioning (GPU, forked)...\n");
        clock_t enc_t0 = clock();
        pid_t pid = fork();
        if (pid == 0) {
            /* Child: run GPU text encoder, write result to file, exit */
            qimg_text_enc *enc = qimg_text_enc_load_gpu(enc_path, NULL, device_id);
            if (!enc) _exit(1);
            int n_out = 0;
            float *hidden = qimg_text_enc_encode(enc, prompt, &n_out);
            qimg_text_enc_free(enc);
            if (!hidden || n_out <= 0) _exit(2);
            FILE *fp = fopen(tmp_path, "wb");
            if (!fp) _exit(3);
            fwrite(&n_out, sizeof(int), 1, fp);
            fwrite(hidden, sizeof(float), (size_t)n_out * 3584, fp);
            fclose(fp);
            free(hidden);
            _exit(0);
        } else if (pid > 0) {
            /* Parent: wait for child, read result */
            int status;
            waitpid(pid, &status, 0);
            if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
                FILE *fp = fopen(tmp_path, "rb");
                if (fp) {
                    int n_out;
                    if (fread(&n_out, sizeof(int), 1, fp) == 1 && n_out > 0) {
                        txt_precomputed = (float *)malloc((size_t)n_out * 3584 * sizeof(float));
                        if (fread(txt_precomputed, sizeof(float), (size_t)n_out * 3584, fp) == (size_t)n_out * 3584) {
                            n_txt_precomputed = n_out;
                            fprintf(stderr, "  Text hidden: [%d, 3584] (%.1fs, GPU forked)\n",
                                    n_out, (double)(clock()-enc_t0)/CLOCKS_PER_SEC);
                        } else {
                            free(txt_precomputed); txt_precomputed = NULL;
                        }
                    }
                    fclose(fp);
                    unlink(tmp_path);
                }
            }
            if (!txt_precomputed)
                fprintf(stderr, "  GPU text encoder (forked) failed, will use CPU fallback\n");
        }
    }

    /* Init HIP + compile qimg kernels */
    hip_qimg_runner *r = hip_qimg_init(device_id, verbose);
    if (!r) { fprintf(stderr, "Init failed\n"); return 1; }

    if (!strcmp(mode, "init")) { hip_qimg_free(r); return 0; }

    /* Load DiT */
    if (!dit_path) {
        fprintf(stderr, "Error: --dit <path> is required\n");
        hip_qimg_free(r); return 1;
    }
    clock_t t0 = clock();
    if (hip_qimg_load_dit(r, dit_path) != 0) {
        hip_qimg_free(r); return 1;
    }
    fprintf(stderr, "DiT loaded in %.1fs\n", (double)(clock()-t0)/CLOCKS_PER_SEC);

    /* ---- test-vae ---- */
    if (!strcmp(mode, "vae")) {
        if (!vae_path) {
            fprintf(stderr, "Error: --vae <path> is required for VAE test\n");
            hip_qimg_free(r); return 1;
        }
        hip_qimg_load_vae(r, vae_path);

        /* Generate random latent for testing */
        rng_state = seed;
        float *latent = (float *)malloc((size_t)16 * lat_h * lat_w * sizeof(float));
        for (int i = 0; i < 16 * lat_h * lat_w; i++) latent[i] = randn() * 0.5f;

        float *rgb = (float *)malloc((size_t)3 * out_h * out_w * sizeof(float));
        t0 = clock();
        hip_qimg_vae_decode(r, latent, lat_h, lat_w, rgb);
        fprintf(stderr, "VAE decode: %.1fs\n", (double)(clock()-t0)/CLOCKS_PER_SEC);

        save_ppm("hip_qimg_vae_test.ppm", rgb, out_h, out_w);
        free(latent); free(rgb);
        hip_qimg_free(r); return 0;
    }

    /* ---- test-dit: single step ---- */
    if (!strcmp(mode, "dit")) {
        int ps = 2, hp = lat_h/ps, wp = lat_w/ps, n_img = hp*wp, n_txt = 7;
        int in_ch = 64, txt_dim = 3584;
        rng_state = seed;
        float *img = (float *)malloc((size_t)n_img * in_ch * sizeof(float));
        float *txt = (float *)malloc((size_t)n_txt * txt_dim * sizeof(float));
        float *out = (float *)malloc((size_t)n_img * in_ch * sizeof(float));
        for (int i = 0; i < n_img*in_ch; i++) img[i] = randn() * 0.1f;
        for (int i = 0; i < n_txt*txt_dim; i++) txt[i] = randn() * 0.1f;

        t0 = clock();
        hip_qimg_dit_step(r, img, n_img, txt, n_txt, 500.0f, out);
        fprintf(stderr, "DiT step: %.2fs\n", (double)(clock()-t0)/CLOCKS_PER_SEC);

        float mn=out[0], mx=out[0], sm=0;
        int nn = n_img*in_ch, nc=0;
        for (int i = 0; i < nn; i++) {
            if (out[i] != out[i]) nc++;
            else { if (out[i]<mn) mn=out[i]; if (out[i]>mx) mx=out[i]; sm+=out[i]; }
        }
        fprintf(stderr, "Output: min=%.4f max=%.4f mean=%.4f nan=%d/%d\n",
                mn, mx, sm/(nn-nc), nc, nn);
        free(img); free(txt); free(out);
        hip_qimg_free(r); return 0;
    }

    /* ---- generate: full pipeline ---- */
    if (!strcmp(mode, "gen")) {
        int ps = 2;
        int hp = lat_h / ps, wp = lat_w / ps;
        int n_img = hp * wp;
        int in_ch = 64, lat_ch = 16;
        int txt_dim = 3584;

        /* 1. Text encoder — use GPU (forked) result, or fall back to CPU */
        int n_txt = 0;
        float *txt_tokens = NULL;

        if (txt_precomputed) {
            txt_tokens = txt_precomputed;
            n_txt = n_txt_precomputed;
            txt_precomputed = NULL;
        } else if (enc_path) {
            fprintf(stderr, "\n[1/3] Text conditioning (CPU fallback)...\n");
            fprintf(stderr, "  Loading text encoder: %s\n", enc_path);
            clock_t enc_t0 = clock();
            qimg_text_enc *enc = qimg_text_enc_load(enc_path);
            if (enc) {
                fprintf(stderr, "  Encoding: \"%s\"\n", prompt);
                txt_tokens = qimg_text_enc_encode(enc, prompt, &n_txt);
                if (txt_tokens)
                    fprintf(stderr, "  Text hidden: [%d, %d] (%.1fs)\n",
                            n_txt, txt_dim,
                            (double)(clock()-enc_t0)/CLOCKS_PER_SEC);
                qimg_text_enc_free(enc);
            }
        }
        if (!txt_tokens) {
            fprintf(stderr, "  No text encoder -- using random conditioning\n");
            n_txt = 7;
            rng_state = seed + 1;
            txt_tokens = (float *)calloc((size_t)n_txt * txt_dim, sizeof(float));
            for (int i = 0; i < n_txt * txt_dim; i++) txt_tokens[i] = randn() * 0.1f;
        }

        /* 2. Generate initial noise */
        fprintf(stderr, "\n[2/3] Denoising (%dx%d, %d steps)...\n", out_w, out_h, n_steps);
        rng_state = seed;
        float *latent = (float *)malloc((size_t)lat_ch * lat_h * lat_w * sizeof(float));
        for (int i = 0; i < lat_ch * lat_h * lat_w; i++) latent[i] = randn();

        /* Patchify latent -> img_tokens */
        float *img_tokens = (float *)malloc((size_t)n_img * in_ch * sizeof(float));
        for (int tok = 0; tok < n_img; tok++) {
            int py = tok / wp, px = tok % wp;
            int idx = 0;
            for (int c = 0; c < lat_ch; c++)
                for (int dy = 0; dy < ps; dy++)
                    for (int dx = 0; dx < ps; dx++)
                        img_tokens[tok * in_ch + idx++] =
                            latent[c * lat_h * lat_w + (py*ps+dy) * lat_w + (px*ps+dx)];
        }

        /* FlowMatch schedule */
        qimg_scheduler sched;
        int seq_len = n_img + n_txt;
        qimg_sched_init(&sched);
        qimg_sched_set_timesteps(&sched, n_steps, seq_len);

        float *vel = (float *)malloc((size_t)n_img * in_ch * sizeof(float));

        fprintf(stderr, "Generating %dx%d image with %d steps...\n", out_w, out_h, n_steps);
        t0 = clock();

        for (int step = 0; step < n_steps; step++) {
            float t = sched.sigmas[step] * 1000.0f;
            float dt = sched.dt[step];

            hip_qimg_dit_step(r, img_tokens, n_img, txt_tokens, n_txt, t, vel);

            /* Euler step: img_tokens += dt * vel */
            for (int i = 0; i < n_img * in_ch; i++)
                img_tokens[i] += dt * vel[i];

            fprintf(stderr, "  step %d/%d: t=%.1f dt=%.4f\n", step+1, n_steps, t, dt);
        }

        fprintf(stderr, "Denoising done in %.1fs (%.2fs/step)\n",
                (double)(clock()-t0)/CLOCKS_PER_SEC,
                (double)(clock()-t0)/CLOCKS_PER_SEC/n_steps);

        /* Unpatchify → latent */
        for (int tok = 0; tok < n_img; tok++) {
            int py = tok / wp, px = tok % wp;
            int idx = 0;
            for (int c = 0; c < lat_ch; c++)
                for (int dy = 0; dy < ps; dy++)
                    for (int dx = 0; dx < ps; dx++)
                        latent[c * lat_h * lat_w + (py*ps+dy) * lat_w + (px*ps+dx)] =
                            img_tokens[tok * in_ch + idx++];
        }

        /* VAE decode */
        /* 3. VAE decode */
        fprintf(stderr, "\n[3/3] VAE decode...\n");
        if (vae_path) {
            hip_qimg_load_vae(r, vae_path);
            float *rgb = (float *)malloc((size_t)3 * out_h * out_w * sizeof(float));
            t0 = clock();
            hip_qimg_vae_decode(r, latent, lat_h, lat_w, rgb);
            fprintf(stderr, "VAE decode: %.1fs\n", (double)(clock()-t0)/CLOCKS_PER_SEC);
            save_ppm(out_path, rgb, out_h, out_w);
            free(rgb);
        } else {
            fprintf(stderr, "No VAE specified, skipping decode\n");
        }

        free(latent); free(img_tokens); free(txt_tokens); free(vel);
        hip_qimg_free(r); return 0;
    }

    hip_qimg_free(r);
    return 0;
}
