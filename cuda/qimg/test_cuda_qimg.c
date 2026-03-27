/*
 * test_cuda_qimg.c - CUDA Qwen-Image test
 *
 * Tests:
 *   --test-init    : Initialize CUDA, compile kernels
 *   --test-load    : Load DiT + VAE weights to GPU
 *   --test-dit     : Run single DiT step on GPU, compare with CPU
 *
 * Build:
 *   cc -O2 -I../../common -I.. -o test_cuda_qimg test_cuda_qimg.c ../cuew.c -lm -ldl -lpthread
 */

#define SAFETENSORS_IMPLEMENTATION
#define GGUF_LOADER_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION
#define CUDA_QIMG_RUNNER_IMPLEMENTATION

#include "cuda_qimg_runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(int argc, char **argv) {
    const char *dit_path = NULL, *vae_path = NULL;
    const char *mode = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--test-init") == 0) mode = "init";
        else if (strcmp(argv[i], "--test-load") == 0) mode = "load";
        else if (strcmp(argv[i], "--test-dit") == 0) mode = "dit";
        else if (strcmp(argv[i], "--dit") == 0 && i+1 < argc) dit_path = argv[++i];
        else if (strcmp(argv[i], "--vae") == 0 && i+1 < argc) vae_path = argv[++i];
    }

    if (!mode) {
        fprintf(stderr, "Usage: %s --test-init|--test-load|--test-dit\n"
                "  --dit <path.gguf>  --vae <path.safetensors>\n", argv[0]);
        return 1;
    }

    fprintf(stderr, "=== CUDA Qwen-Image Test: %s ===\n", mode);

    cuda_qimg_runner *r = cuda_qimg_init(0, 1);
    if (!r) { fprintf(stderr, "Init failed\n"); return 1; }
    fprintf(stderr, "CUDA init OK\n");

    if (strcmp(mode, "init") == 0) {
        cuda_qimg_free(r);
        return 0;
    }

    if (!dit_path) dit_path = "/mnt/disk01/models/qwen-image/diffusion-models/qwen-image-Q4_0.gguf";
    if (!vae_path) vae_path = "/mnt/disk01/models/qwen-image/vae/qwen_image_vae.safetensors";

    clock_t t0 = clock();
    if (cuda_qimg_load_dit(r, dit_path) != 0) {
        fprintf(stderr, "DiT load failed\n");
        cuda_qimg_free(r); return 1;
    }
    clock_t t1 = clock();
    fprintf(stderr, "DiT loaded in %.1fs\n", (double)(t1-t0)/CLOCKS_PER_SEC);

    if (cuda_qimg_load_vae(r, vae_path) != 0) {
        fprintf(stderr, "VAE load failed\n");
        cuda_qimg_free(r); return 1;
    }
    fprintf(stderr, "VAE loaded\n");

    if (strcmp(mode, "load") == 0) {
        fprintf(stderr, "Load test passed.\n");
        cuda_qimg_free(r);
        return 0;
    }

    /* --test-dit: TODO - run a single forward step and compare */
    fprintf(stderr, "DiT step test: placeholder\n");

    cuda_qimg_free(r);
    return 0;
}
