/*
 * test_hip_flux2.c - Flux.2 Klein 4B end-to-end text-to-image test (HIP/RDNA4)
 *
 * Loads DiT weights, runs one or more denoising steps on GPU,
 * optionally decodes with VAE (CPU fallback), saves output image.
 *
 * Build:
 *   make
 *
 * Usage:
 *   ./test_hip_flux2 --dit <dit.safetensors> [--vae <vae.safetensors>] \
 *                    [--prompt "text"] [--steps 4] [--size 512] \
 *                    [-o output.ppm] [-d device_id] [-v]
 */

#define SAFETENSORS_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION
#define QIMG_SCHEDULER_IMPLEMENTATION

#include "../../common/safetensors.h"
#include "../../common/ggml_dequant.h"
#include "../../common/qwen_image_scheduler.h"
#include "hip_flux2_runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

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

/* ---- Utility ---- */

static void save_ppm(const char *path, const float *rgb, int h, int w) {
    FILE *fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "Cannot write %s\n", path); return; }
    fprintf(fp, "P6\n%d %d\n255\n", w, h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            unsigned char px[3];
            for (int c = 0; c < 3; c++) {
                float v = rgb[c * h * w + y * w + x];
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
    const char *out_path = "hip_flux2_out.ppm";
    int device_id = 0;
    int verbose = 1;
    int n_steps = 4;
    int img_size = 512;
    int test_dit_only = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--dit") && i+1 < argc) dit_path = argv[++i];
        else if (!strcmp(argv[i], "--vae") && i+1 < argc) vae_path = argv[++i];
        else if (!strcmp(argv[i], "-o") && i+1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "-d") && i+1 < argc) device_id = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) verbose = 2;
        else if (!strcmp(argv[i], "--steps") && i+1 < argc) n_steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--size") && i+1 < argc) img_size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--test-dit")) test_dit_only = 1;
        else {
            fprintf(stderr, "Usage: %s --dit <dit.safetensors> [--vae <vae.safetensors>] "
                    "[--steps N] [--size S] [-o out.ppm] [-d dev] [-v] [--test-dit]\n", argv[0]);
            return 1;
        }
    }

    if (!dit_path) {
        fprintf(stderr, "Error: --dit <path> is required\n");
        return 1;
    }

    /* Initialize HIP runner */
    hip_flux2_runner *r = hip_flux2_init(device_id, verbose);
    if (!r) { fprintf(stderr, "hip_flux2_init failed\n"); return 1; }

    /* Load DiT */
    if (hip_flux2_load_dit(r, dit_path) < 0) {
        fprintf(stderr, "Failed to load DiT\n");
        hip_flux2_free(r);
        return 1;
    }

    /* Load VAE (optional) */
    if (vae_path) {
        if (hip_flux2_load_vae(r, vae_path) < 0)
            fprintf(stderr, "Warning: Failed to load VAE, will skip decode\n");
    }

    /* Setup: patchified latent (random noise) */
    int patch_size = 2;
    int lat_h = img_size / 8;
    int lat_w = img_size / 8;
    int n_img = (lat_h / patch_size) * (lat_w / patch_size);
    int patch_in_ch = 32 * patch_size * patch_size;  /* 128 */
    int n_txt = 128;  /* dummy text tokens */
    int txt_dim = 7680;  /* Qwen3-4B hidden dim * 2 layers */

    fprintf(stderr, "Image: %dx%d -> lat %dx%d -> %d patches (patch_in=%d)\n",
            img_size, img_size, lat_h, lat_w, n_img, patch_in_ch);
    fprintf(stderr, "Text: %d tokens x %d dim\n", n_txt, txt_dim);

    float *img_tok = (float *)calloc((size_t)n_img * patch_in_ch, sizeof(float));
    float *txt_tok = (float *)calloc((size_t)n_txt * txt_dim, sizeof(float));
    float *velocity = (float *)calloc((size_t)n_img * patch_in_ch, sizeof(float));

    /* Initialize with random noise */
    for (int i = 0; i < n_img * patch_in_ch; i++) img_tok[i] = randn();
    for (int i = 0; i < n_txt * txt_dim; i++) txt_tok[i] = randn() * 0.01f;

    if (test_dit_only) {
        /* Single step test */
        fprintf(stderr, "\n--- Single DiT step test ---\n");
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        int ret = hip_flux2_dit_step(r, img_tok, n_img, txt_tok, n_txt, 1.0f, 0.0f, velocity);

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        fprintf(stderr, "DiT step: %s (%.3f s)\n", ret == 0 ? "OK" : "FAIL", dt);

        /* Print some output values */
        float sum = 0, mx = 0;
        for (int i = 0; i < n_img * patch_in_ch; i++) {
            sum += velocity[i];
            float a = fabsf(velocity[i]);
            if (a > mx) mx = a;
        }
        fprintf(stderr, "velocity: mean=%.6f max_abs=%.6f\n",
                sum / (n_img * patch_in_ch), mx);
    } else {
        /* Full denoising loop */
        fprintf(stderr, "\n--- %d-step denoising ---\n", n_steps);
        float *sigmas = (float *)malloc((n_steps + 1) * sizeof(float));
        for (int i = 0; i <= n_steps; i++)
            sigmas[i] = 1.0f - (float)i / (float)n_steps;

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        for (int step = 0; step < n_steps; step++) {
            float sigma = sigmas[step];
            float dt = sigmas[step + 1] - sigmas[step];

            int ret = hip_flux2_dit_step(r, img_tok, n_img, txt_tok, n_txt,
                                         sigma, 0.0f, velocity);
            if (ret != 0) {
                fprintf(stderr, "DiT step %d failed\n", step);
                break;
            }

            /* Euler step: x += dt * v */
            for (int i = 0; i < n_img * patch_in_ch; i++)
                img_tok[i] += dt * velocity[i];

            fprintf(stderr, "  step %d/%d (sigma=%.4f)\n", step+1, n_steps, sigma);
        }

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double total = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        fprintf(stderr, "Denoising: %.3f s total (%.3f s/step)\n", total, total / n_steps);

        free(sigmas);

        /* VAE decode if available */
        if (vae_path) {
            /* Unpatchify: [n_img, pin] -> [32, lat_h, lat_w] */
            float *latent = (float *)calloc((size_t)32 * lat_h * lat_w, sizeof(float));
            int ph = lat_h / patch_size, pw = lat_w / patch_size;
            for (int py = 0; py < ph; py++)
                for (int px = 0; px < pw; px++) {
                    int pidx = py * pw + px;
                    for (int c = 0; c < 32; c++)
                        for (int dy = 0; dy < patch_size; dy++)
                            for (int dx = 0; dx < patch_size; dx++) {
                                int src = pidx * patch_in_ch + c * patch_size * patch_size + dy * patch_size + dx;
                                int y = py * patch_size + dy;
                                int x = px * patch_size + dx;
                                latent[c * lat_h * lat_w + y * lat_w + x] = img_tok[src];
                            }
                }

            float *rgb = (float *)calloc((size_t)3 * img_size * img_size, sizeof(float));
            fprintf(stderr, "VAE decode (CPU)...\n");
            clock_gettime(CLOCK_MONOTONIC, &t0);
            hip_flux2_vae_decode(r, latent, lat_h, lat_w, rgb);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double vae_t = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
            fprintf(stderr, "VAE decode: %.3f s\n", vae_t);

            save_ppm(out_path, rgb, img_size, img_size);
            free(latent);
            free(rgb);
        }
    }

    free(img_tok);
    free(txt_tok);
    free(velocity);
    hip_flux2_free(r);

    return 0;
}
