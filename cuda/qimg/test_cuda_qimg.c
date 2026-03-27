/*
 * test_cuda_qimg.c - CUDA Qwen-Image test
 *
 * Tests:
 *   --test-init    : Initialize CUDA, compile kernels
 *   --test-load    : Load DiT + VAE weights to GPU
 *   --test-dit     : Run single DiT step on GPU
 *
 * Build:
 *   cc -O2 -I../../common -I.. -o test_cuda_qimg test_cuda_qimg.c ../cuew.c -lm -ldl -lpthread
 */

#define SAFETENSORS_IMPLEMENTATION
#define CUDA_QIMG_RUNNER_IMPLEMENTATION

#include "cuda_qimg_runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Simple PRNG */
static uint64_t rng_state = 42;
static float randn(void) {
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u1 = (double)(rng_state >> 11) / (double)(1ULL << 53);
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u2 = (double)(rng_state >> 11) / (double)(1ULL << 53);
    if (u1 < 1e-10) u1 = 1e-10;
    return (float)(sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2));
}

int main(int argc, char **argv) {
    const char *dit_path = "/mnt/disk01/models/qwen-image-st/diffusion_models/qwen_image_fp8_e4m3fn.safetensors";
    const char *vae_path = "/mnt/disk01/models/qwen-image-st/vae/qwen_image_vae.safetensors";
    const char *mode = NULL;
    int lat_h = 8, lat_w = 8;
    int force_f16 = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--test-init") == 0) mode = "init";
        else if (strcmp(argv[i], "--test-load") == 0) mode = "load";
        else if (strcmp(argv[i], "--test-dit") == 0) mode = "dit";
        else if (strcmp(argv[i], "--no-fp8") == 0) force_f16 = 1;
        else if (strcmp(argv[i], "--dit") == 0 && i+1 < argc) dit_path = argv[++i];
        else if (strcmp(argv[i], "--vae") == 0 && i+1 < argc) vae_path = argv[++i];
        else if (strcmp(argv[i], "--height") == 0 && i+1 < argc) lat_h = atoi(argv[++i]) / 8;
        else if (strcmp(argv[i], "--width") == 0 && i+1 < argc) lat_w = atoi(argv[++i]) / 8;
    }

    if (!mode) {
        fprintf(stderr, "Usage: %s --test-init|--test-load|--test-dit\n"
                "  --dit <path>  --vae <path>  --height <h>  --width <w>\n", argv[0]);
        return 1;
    }

    fprintf(stderr, "=== CUDA Qwen-Image Test: %s ===\n", mode);

    cuda_qimg_runner *r = cuda_qimg_init(0, 1);
    if (r && force_f16) {
        r->use_fp8_gemm = 0;
        fprintf(stderr, "Forced F16 GEMM path (--no-fp8)\n");
    }
    if (!r) { fprintf(stderr, "Init failed\n"); return 1; }

    if (strcmp(mode, "init") == 0) { cuda_qimg_free(r); return 0; }

    clock_t t0 = clock();
    if (cuda_qimg_load_dit(r, dit_path) != 0) {
        fprintf(stderr, "DiT load failed\n"); cuda_qimg_free(r); return 1;
    }
    fprintf(stderr, "DiT loaded in %.1fs\n", (double)(clock()-t0)/CLOCKS_PER_SEC);

    if (strcmp(mode, "load") == 0) {
        cuda_qimg_load_vae(r, vae_path);
        fprintf(stderr, "Load test passed.\n");
        cuda_qimg_free(r);
        return 0;
    }

    /* --test-dit: run single DiT step */
    if (strcmp(mode, "dit") == 0) {
        int ps = 2;
        int hp = lat_h / ps, wp = lat_w / ps;
        int n_img = hp * wp;
        int n_txt = 7;  /* "a red apple on a white table" */
        int in_ch = 64, txt_dim = 3584;

        fprintf(stderr, "n_img=%d, n_txt=%d\n", n_img, n_txt);

        /* Generate test inputs */
        rng_state = 42;
        float *img_tokens = (float *)malloc((size_t)n_img * in_ch * sizeof(float));
        for (int i = 0; i < n_img * in_ch; i++) img_tokens[i] = randn() * 0.1f;

        float *txt_tokens = (float *)malloc((size_t)n_txt * txt_dim * sizeof(float));
        for (int i = 0; i < n_txt * txt_dim; i++) txt_tokens[i] = randn() * 0.1f;

        float *out = (float *)malloc((size_t)n_img * in_ch * sizeof(float));

        t0 = clock();
        int rc = cuda_qimg_dit_step(r, img_tokens, n_img, txt_tokens, n_txt,
                                     500.0f, out);
        double elapsed = (double)(clock()-t0)/CLOCKS_PER_SEC;

        if (rc != 0) {
            fprintf(stderr, "DiT step failed\n");
        } else {
            float min_v = out[0], max_v = out[0], sum = 0;
            for (int i = 0; i < n_img * in_ch; i++) {
                if (out[i] < min_v) min_v = out[i];
                if (out[i] > max_v) max_v = out[i];
                sum += out[i];
            }
            fprintf(stderr, "DiT step: %.2fs\n", elapsed);
            fprintf(stderr, "Output: min=%.6f max=%.6f mean=%.6f\n",
                    min_v, max_v, sum / (n_img * in_ch));
        }

        free(img_tokens); free(txt_tokens); free(out);
    }

    cuda_qimg_free(r);
    return 0;
}
