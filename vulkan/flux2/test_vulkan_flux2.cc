// SPDX-License-Identifier: MIT
// Copyright 2025 - Present, Light Transport Entertainment Inc.
//
// test_vulkan_flux2.cc - Flux.2 Klein 4B end-to-end test (Vulkan)

#include "vulkan_flux2_runner.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <cstdint>

/* ---- PRNG: Box-Muller ---- */
static uint64_t rng_state = 42;
static int rng_cached_valid = 0;
static float rng_cached = 0.0f;

static float randn() {
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
                v = v < 0 ? 0 : (v > 1 ? 1 : v);
                px[c] = (unsigned char)(v * 255.0f + 0.5f);
            }
            fwrite(px, 1, 3, fp);
        }
    fclose(fp);
    fprintf(stderr, "Saved %s (%dx%d)\n", path, w, h);
}

int main(int argc, char **argv) {
    const char *dit_path = nullptr;
    const char *vae_path = nullptr;
    const char *shader_dir = ".";
    const char *out_path = "vk_flux2_out.ppm";
    int device_id = 0;
    int verbose = 1;
    int n_steps = 4;
    int img_size = 512;
    bool test_dit_only = false;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--dit") && i+1 < argc) dit_path = argv[++i];
        else if (!strcmp(argv[i], "--vae") && i+1 < argc) vae_path = argv[++i];
        else if (!strcmp(argv[i], "--shaders") && i+1 < argc) shader_dir = argv[++i];
        else if (!strcmp(argv[i], "-o") && i+1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "-d") && i+1 < argc) device_id = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) verbose = 2;
        else if (!strcmp(argv[i], "--steps") && i+1 < argc) n_steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--size") && i+1 < argc) img_size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--test-dit")) test_dit_only = true;
        else {
            fprintf(stderr, "Usage: %s --dit <dit.safetensors> [--vae <vae.safetensors>] "
                    "[--shaders dir] [--steps N] [--size S] [-o out.ppm] [-d dev] [-v] [--test-dit]\n",
                    argv[0]);
            return 1;
        }
    }

    if (!dit_path) {
        fprintf(stderr, "Error: --dit <path> is required\n");
        return 1;
    }

    vulkan_flux2_runner *r = vulkan_flux2_init(device_id, verbose, shader_dir);
    if (!r) { fprintf(stderr, "vulkan_flux2_init failed\n"); return 1; }

    if (vulkan_flux2_load_dit(r, dit_path) < 0) {
        fprintf(stderr, "Failed to load DiT\n");
        vulkan_flux2_free(r);
        return 1;
    }

    if (vae_path) {
        if (vulkan_flux2_load_vae(r, vae_path) < 0)
            fprintf(stderr, "Warning: Failed to load VAE\n");
    }

    int patch_size = 2;
    int lat_h = img_size / 8;
    int lat_w = img_size / 8;
    int n_img = (lat_h / patch_size) * (lat_w / patch_size);
    int patch_in_ch = 32 * patch_size * patch_size;
    int n_txt = 128;
    int txt_dim = 7680;

    fprintf(stderr, "Image: %dx%d -> %d patches, Text: %d tokens x %d\n",
            img_size, img_size, n_img, n_txt, txt_dim);

    float *img_tok = (float *)calloc((size_t)n_img * patch_in_ch, sizeof(float));
    float *txt_tok = (float *)calloc((size_t)n_txt * txt_dim, sizeof(float));
    float *velocity = (float *)calloc((size_t)n_img * patch_in_ch, sizeof(float));

    for (int i = 0; i < n_img * patch_in_ch; i++) img_tok[i] = randn();
    for (int i = 0; i < n_txt * txt_dim; i++) txt_tok[i] = randn() * 0.01f;

    if (test_dit_only) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        int ret = vulkan_flux2_dit_step(r, img_tok, n_img, txt_tok, n_txt, 1.0f, 0.0f, velocity);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        fprintf(stderr, "DiT step: %s (%.3f s)\n", ret == 0 ? "OK" : "FAIL", dt);

        float sum = 0, mx = 0;
        for (int i = 0; i < n_img * patch_in_ch; i++) {
            sum += velocity[i];
            float a = fabsf(velocity[i]);
            if (a > mx) mx = a;
        }
        fprintf(stderr, "velocity: mean=%.6f max_abs=%.6f\n",
                sum / (n_img * patch_in_ch), mx);
    } else {
        fprintf(stderr, "\n--- %d-step denoising ---\n", n_steps);
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        for (int step = 0; step < n_steps; step++) {
            float sigma = 1.0f - (float)step / (float)n_steps;
            float dt = -1.0f / (float)n_steps;

            int ret = vulkan_flux2_dit_step(r, img_tok, n_img, txt_tok, n_txt,
                                            sigma, 0.0f, velocity);
            if (ret != 0) { fprintf(stderr, "Step %d failed\n", step); break; }

            for (int i = 0; i < n_img * patch_in_ch; i++)
                img_tok[i] += dt * velocity[i];

            fprintf(stderr, "  step %d/%d (sigma=%.4f)\n", step+1, n_steps, sigma);
        }

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double total = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        fprintf(stderr, "Denoising: %.3f s (%.3f s/step)\n", total, total / n_steps);

        if (vae_path) {
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
            vulkan_flux2_vae_decode(r, latent, lat_h, lat_w, rgb);
            save_ppm(out_path, rgb, img_size, img_size);
            free(latent);
            free(rgb);
        }
    }

    free(img_tok); free(txt_tok); free(velocity);
    vulkan_flux2_free(r);
    return 0;
}
