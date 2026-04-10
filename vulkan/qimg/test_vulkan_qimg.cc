/*
 * test_vulkan_qimg.cc - Qwen-Image end-to-end text-to-image test (Vulkan)
 *
 * Modes:
 *   --test-init    : Initialize Vulkan, compile shaders
 *   --test-dit     : Run single DiT step on GPU
 *   --test-vae     : VAE decode test
 *   --generate     : Full text-to-image pipeline
 *
 * Build:
 *   cmake .. && make test_vulkan_qimg
 */

#include "../../common/ggml_dequant.h"
#include "../../common/qwen_image_scheduler.h"
#include "vulkan_qimg_runner.h"

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
                v = v * 0.5f + 0.5f;
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
    const char *shader_dir = "build/shaders";
    const char *out_path = "vulkan_qimg_out.ppm";
    const char *mode = nullptr;
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
        else if (!strcmp(argv[i], "--shaders") && i+1 < argc) shader_dir = argv[++i];
        else if (!strcmp(argv[i], "--height") && i+1 < argc) out_h = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--width") && i+1 < argc) out_w = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps") && i+1 < argc) n_steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i+1 < argc) seed = (uint64_t)atoll(argv[++i]);
        else if (!strcmp(argv[i], "-o") && i+1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "-d") && i+1 < argc) device_id = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) verbose = 2;
        else {
            fprintf(stderr, "Usage: %s --test-init|--test-dit|--test-vae|--generate\n"
                    "  --dit <st>  --vae <st>  --shaders <dir>\n"
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

    fprintf(stderr, "=== Vulkan Qwen-Image: %s ===\n", mode);

    vulkan_qimg_runner *r = vulkan_qimg_init(device_id, verbose, shader_dir);
    if (!r) { fprintf(stderr, "Init failed\n"); return 1; }

    if (!strcmp(mode, "init")) { vulkan_qimg_free(r); return 0; }

    if (!dit_path) {
        fprintf(stderr, "Error: --dit <path> is required\n");
        vulkan_qimg_free(r); return 1;
    }

    clock_t t0 = clock();
    if (vulkan_qimg_load_dit(r, dit_path) != 0) {
        vulkan_qimg_free(r); return 1;
    }
    fprintf(stderr, "DiT loaded in %.1fs\n", (double)(clock()-t0)/CLOCKS_PER_SEC);

    /* ---- test-vae ---- */
    if (!strcmp(mode, "vae")) {
        if (!vae_path) {
            fprintf(stderr, "Error: --vae <path> is required\n");
            vulkan_qimg_free(r); return 1;
        }
        vulkan_qimg_load_vae(r, vae_path);
        rng_state = seed;
        float *latent = (float *)malloc((size_t)16 * lat_h * lat_w * sizeof(float));
        for (int i = 0; i < 16 * lat_h * lat_w; i++) latent[i] = randn() * 0.5f;
        float *rgb = (float *)malloc((size_t)3 * out_h * out_w * sizeof(float));
        t0 = clock();
        vulkan_qimg_vae_decode(r, latent, lat_h, lat_w, rgb);
        fprintf(stderr, "VAE decode: %.1fs\n", (double)(clock()-t0)/CLOCKS_PER_SEC);
        save_ppm("vulkan_qimg_vae_test.ppm", rgb, out_h, out_w);
        free(latent); free(rgb);
        vulkan_qimg_free(r); return 0;
    }

    /* ---- test-dit ---- */
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
        vulkan_qimg_dit_step(r, img, n_img, txt, n_txt, 500.0f, out);
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
        vulkan_qimg_free(r); return 0;
    }

    /* ---- generate ---- */
    if (!strcmp(mode, "gen")) {
        int ps = 2;
        int hp = lat_h / ps, wp = lat_w / ps;
        int n_img = hp * wp;
        int in_ch = 64, lat_ch = 16;

        rng_state = seed;
        float *latent = (float *)malloc((size_t)lat_ch * lat_h * lat_w * sizeof(float));
        for (int i = 0; i < lat_ch * lat_h * lat_w; i++) latent[i] = randn();

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

        int n_txt = 7, txt_dim = 3584;
        float *txt_tokens = (float *)calloc((size_t)n_txt * txt_dim, sizeof(float));
        for (int i = 0; i < n_txt * txt_dim; i++) txt_tokens[i] = randn() * 0.1f;

        qimg_scheduler sched;
        int seq_len = n_img + n_txt;
        qimg_scheduler_init(&sched, n_steps, seq_len);

        float *vel = (float *)malloc((size_t)n_img * in_ch * sizeof(float));

        fprintf(stderr, "Generating %dx%d image with %d steps...\n", out_w, out_h, n_steps);
        t0 = clock();

        for (int step = 0; step < n_steps; step++) {
            float t = sched.sigmas[step] * 1000.0f;
            float dt = sched.dt[step];
            vulkan_qimg_dit_step(r, img_tokens, n_img, txt_tokens, n_txt, t, vel);
            for (int i = 0; i < n_img * in_ch; i++)
                img_tokens[i] += dt * vel[i];
            fprintf(stderr, "  step %d/%d: t=%.1f dt=%.4f\n", step+1, n_steps, t, dt);
        }

        fprintf(stderr, "Denoising done in %.1fs\n", (double)(clock()-t0)/CLOCKS_PER_SEC);

        for (int tok = 0; tok < n_img; tok++) {
            int py = tok / wp, px = tok % wp;
            int idx = 0;
            for (int c = 0; c < lat_ch; c++)
                for (int dy = 0; dy < ps; dy++)
                    for (int dx = 0; dx < ps; dx++)
                        latent[c * lat_h * lat_w + (py*ps+dy) * lat_w + (px*ps+dx)] =
                            img_tokens[tok * in_ch + idx++];
        }

        if (vae_path) {
            vulkan_qimg_load_vae(r, vae_path);
            float *rgb = (float *)malloc((size_t)3 * out_h * out_w * sizeof(float));
            t0 = clock();
            vulkan_qimg_vae_decode(r, latent, lat_h, lat_w, rgb);
            fprintf(stderr, "VAE decode: %.1fs\n", (double)(clock()-t0)/CLOCKS_PER_SEC);
            save_ppm(out_path, rgb, out_h, out_w);
            free(rgb);
        }

        free(latent); free(img_tokens); free(txt_tokens); free(vel);
        vulkan_qimg_free(r); return 0;
    }

    vulkan_qimg_free(r);
    return 0;
}
