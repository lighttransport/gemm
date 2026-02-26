/*
 * test_cuda_vision.c - Test CUDA vision encoder against CPU reference
 *
 * Usage: ./test_cuda_vision <mmproj.gguf> [--f16] [--image-size N]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* CPU vision encoder (reference) — needs qtensor from transformer.h */
#define GGUF_LOADER_IMPLEMENTATION
#include "../../common/gguf_loader.h"

#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"

#include "../../common/transformer.h"

#define VISION_ENCODER_IMPLEMENTATION
#include "../../common/vision_encoder.h"

/* CUDA vision encoder */
#include "cuda_vision_encoder.h"

/* Timer helper */
static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* Generate synthetic checkerboard image */
static uint8_t *generate_checkerboard(int width, int height, int block_size) {
    uint8_t *rgb = (uint8_t *)malloc(width * height * 3);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int bx = x / block_size;
            int by = y / block_size;
            int white = (bx + by) % 2;
            uint8_t val = white ? 220 : 35;
            /* Add some color variation */
            rgb[(y * width + x) * 3 + 0] = val;
            rgb[(y * width + x) * 3 + 1] = (uint8_t)(val * 0.8f);
            rgb[(y * width + x) * 3 + 2] = (uint8_t)(val * 0.6f);
        }
    }
    return rgb;
}

/* Compare two float arrays */
static void compare_outputs(const float *cpu, const float *gpu, int n,
                             const char *label, float threshold) {
    float sum_diff2 = 0.0f, sum_ref2 = 0.0f;
    float max_abs_diff = 0.0f;
    int max_diff_idx = 0;
    int nan_count = 0, inf_count = 0;

    for (int i = 0; i < n; i++) {
        if (isnan(gpu[i])) { nan_count++; continue; }
        if (isinf(gpu[i])) { inf_count++; continue; }
        float diff = gpu[i] - cpu[i];
        sum_diff2 += diff * diff;
        sum_ref2 += cpu[i] * cpu[i];
        if (fabsf(diff) > max_abs_diff) {
            max_abs_diff = fabsf(diff);
            max_diff_idx = i;
        }
    }

    float rel_l2 = (sum_ref2 > 0.0f) ? sqrtf(sum_diff2 / sum_ref2) : sqrtf(sum_diff2);

    printf("\n=== %s ===\n", label);
    printf("  Elements:      %d\n", n);
    printf("  Relative L2:   %.6e", rel_l2);
    if (rel_l2 < threshold)
        printf("  [PASS < %.0e]\n", threshold);
    else
        printf("  [FAIL >= %.0e]\n", threshold);
    printf("  Max abs diff:  %.6e (at index %d)\n", max_abs_diff, max_diff_idx);
    if (max_diff_idx < n && max_diff_idx >= 0) {
        printf("    CPU[%d] = %.8f, GPU[%d] = %.8f\n",
               max_diff_idx, cpu[max_diff_idx], max_diff_idx, gpu[max_diff_idx]);
    }
    if (nan_count > 0) printf("  NaN count:     %d [FAIL]\n", nan_count);
    if (inf_count > 0) printf("  Inf count:     %d [FAIL]\n", inf_count);

    /* Print first few values */
    printf("  First 8 values (CPU vs GPU):\n");
    for (int i = 0; i < 8 && i < n; i++) {
        printf("    [%d] CPU=%.6f GPU=%.6f diff=%.6e\n", i, cpu[i], gpu[i], gpu[i] - cpu[i]);
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <mmproj.gguf> [--f16] [--image-size N]\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    int use_f16 = 0;
    int image_size = 0;  /* 0 = use model default */
    int verbose = 1;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--f16") == 0) use_f16 = 1;
        else if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0) verbose = 2;
        else if (strcmp(argv[i], "--image-size") == 0 && i + 1 < argc) {
            image_size = atoi(argv[++i]);
        }
    }

    printf("=== CUDA Vision Encoder Test ===\n");
    printf("Model: %s\n", model_path);
    printf("Mode:  %s\n", use_f16 ? "F16 (performance)" : "F32 (verification)");

    /* Load GGUF */
    printf("\nLoading GGUF model...\n");
    gguf_context *gguf = gguf_open(model_path, 1);
    if (!gguf) {
        fprintf(stderr, "Failed to load GGUF: %s\n", model_path);
        return 1;
    }

    /* Load CPU vision model */
    printf("Loading CPU vision model...\n");
    vision_model *vm = vision_load(gguf);
    if (!vm) {
        fprintf(stderr, "Failed to load CPU vision model\n");
        gguf_close(gguf);
        return 1;
    }

    /* Determine image size */
    if (image_size <= 0) image_size = vm->image_size;
    /* Must be multiple of patch_size * spatial_merge */
    int align = vm->patch_size * vm->spatial_merge;
    image_size = (image_size / align) * align;
    if (image_size < align) image_size = align;

    printf("Image size: %dx%d\n", image_size, image_size);

    /* Init CUDA vision encoder */
    printf("\nInitializing CUDA vision encoder...\n");
    cuda_vision_runner *cuda_r = cuda_vision_init(0, verbose, use_f16);
    if (!cuda_r) {
        fprintf(stderr, "Failed to init CUDA vision encoder\n");
        vision_free(vm);
        gguf_close(gguf);
        return 1;
    }

    printf("Loading CUDA weights...\n");
    if (cuda_vision_load_weights(cuda_r, gguf) != 0) {
        fprintf(stderr, "Failed to load CUDA weights\n");
        cuda_vision_free(cuda_r);
        vision_free(vm);
        gguf_close(gguf);
        return 1;
    }

    /* Generate test image */
    printf("\nGenerating %dx%d checkerboard test image...\n", image_size, image_size);
    uint8_t *test_img = generate_checkerboard(image_size, image_size, 32);

    /* Normalize image */
    float *rgb_norm = vision_normalize_image(vm, test_img, image_size, image_size);
    free(test_img);

    /* CPU encoding */
    printf("\n--- CPU Encoding ---\n");
    double t0 = get_time_ms();
    float *cpu_result = vision_encode(vm, rgb_norm, image_size, image_size, 1);
    double t1 = get_time_ms();
    printf("CPU time: %.1f ms\n", t1 - t0);

    if (!cpu_result) {
        fprintf(stderr, "CPU encoding failed!\n");
        free(rgb_norm);
        cuda_vision_free(cuda_r);
        vision_free(vm);
        gguf_close(gguf);
        return 1;
    }

    /* GPU encoding */
    printf("\n--- GPU Encoding ---\n");
    double t2 = get_time_ms();
    float *gpu_result = cuda_vision_encode(cuda_r, rgb_norm, image_size, image_size);
    double t3 = get_time_ms();
    printf("GPU time: %.1f ms\n", t3 - t2);

    if (!gpu_result) {
        fprintf(stderr, "GPU encoding failed!\n");
        free(cpu_result);
        free(rgb_norm);
        cuda_vision_free(cuda_r);
        vision_free(vm);
        gguf_close(gguf);
        return 1;
    }

    /* Compare — compute actual n_merged for the given image size */
    int ps = vm->patch_size;
    int sp = vm->spatial_merge;
    int actual_gw = image_size / ps;
    int actual_n_patches = actual_gw * actual_gw;
    int actual_n_merged = actual_n_patches / (sp * sp);
    int total_embd = cuda_vision_total_embd(cuda_r);
    int total_floats = actual_n_merged * total_embd;
    printf("\nComparing %d merged tokens x %d embd = %d floats\n",
           actual_n_merged, total_embd, total_floats);
    float threshold = use_f16 ? 1e-2f : 1e-4f;

    compare_outputs(cpu_result, gpu_result, total_floats,
                     use_f16 ? "F16 GPU vs CPU" : "F32 GPU vs CPU", threshold);

    /* Timing summary */
    printf("\n=== Timing Summary ===\n");
    printf("  CPU (1 thread): %.1f ms\n", t1 - t0);
    printf("  GPU:            %.1f ms\n", t3 - t2);
    printf("  Speedup:        %.1fx\n", (t1 - t0) / (t3 - t2));

    /* Cleanup */
    free(cpu_result);
    free(gpu_result);
    free(rgb_norm);
    cuda_vision_free(cuda_r);
    vision_free(vm);
    gguf_close(gguf);

    printf("\nDone.\n");
    return 0;
}
