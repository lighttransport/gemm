// SPDX-License-Identifier: MIT
// Copyright 2025 - Present, Light Transport Entertainment Inc.
//
// Test: Vulkan Vision Encoder vs CPU reference
//

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>

// C headers (implementation compiled in vision_cpu_impl.c)
extern "C" {
#include "../common/gguf_loader.h"
#include "../common/ggml_dequant.h"
#include "../common/transformer.h"
#include "../common/vision_encoder.h"
}

#include "vulkan_vision_encoder.hh"

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s --mmproj <path> [--image-size <N>] [--device <id>] [--attn <mode>] [--verbose]\n", prog);
    fprintf(stderr, "  --mmproj <path>     Path to mmproj GGUF file\n");
    fprintf(stderr, "  --image-size <N>    Image size (default: model default, must be multiple of patch_size)\n");
    fprintf(stderr, "  --device <id>       Vulkan device ID (default: 0)\n");
    fprintf(stderr, "  --attn <mode>       Attention: cpu, naive, flash (default: flash)\n");
    fprintf(stderr, "  --verbose           Verbose output\n");
}

int main(int argc, char **argv) {
    std::string mmproj_path;
    int image_size = 0;  // 0 = use model default
    int device_id = 0;
    bool verbose = false;
    VulkanVisionEncoder::AttentionMode attn_mode = VulkanVisionEncoder::ATTN_FLASH_GPU;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--mmproj") == 0 && i + 1 < argc) {
            mmproj_path = argv[++i];
        } else if (strcmp(argv[i], "--image-size") == 0 && i + 1 < argc) {
            image_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            device_id = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--attn") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "cpu") == 0) attn_mode = VulkanVisionEncoder::ATTN_CPU;
            else if (strcmp(argv[i], "naive") == 0) attn_mode = VulkanVisionEncoder::ATTN_NAIVE_GPU;
            else if (strcmp(argv[i], "flash") == 0) attn_mode = VulkanVisionEncoder::ATTN_FLASH_GPU;
            else { fprintf(stderr, "Unknown attn mode: %s\n", argv[i]); return 1; }
        } else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (mmproj_path.empty()) {
        fprintf(stderr, "Error: --mmproj is required\n");
        print_usage(argv[0]);
        return 1;
    }

    // Load GGUF
    fprintf(stderr, "Loading mmproj from %s...\n", mmproj_path.c_str());
    gguf_context *g = gguf_open(mmproj_path.c_str(), 1);
    if (!g) {
        fprintf(stderr, "Failed to load GGUF file\n");
        return 1;
    }

    // Load CPU reference model
    fprintf(stderr, "Loading CPU vision model...\n");
    vision_model *vm = vision_load(g);
    if (!vm) {
        fprintf(stderr, "Failed to load vision model\n");
        gguf_close(g);
        return 1;
    }

    if (image_size == 0) image_size = vm->image_size;

    int width = image_size;
    int height = image_size;
    int ps = vm->patch_size;
    int gw = width / ps;
    int gh = height / ps;
    int n_patches = gw * gh;
    int sm = vm->spatial_merge;
    int n_merged = n_patches / (sm * sm);

    fprintf(stderr, "Image: %dx%d, patches: %d, merged: %d\n",
            width, height, n_patches, n_merged);

    // Create synthetic test image (gradient pattern)
    std::vector<float> rgb_norm(width * height * 3);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            rgb_norm[idx + 0] = (float)x / width * 2.0f - 1.0f;
            rgb_norm[idx + 1] = (float)y / height * 2.0f - 1.0f;
            rgb_norm[idx + 2] = sinf((float)(x + y) * 0.1f);
        }
    }

    // CPU encode
    fprintf(stderr, "\n=== CPU Encode ===\n");
    auto cpu_start = std::chrono::high_resolution_clock::now();
    float *cpu_result = vision_encode(vm, rgb_norm.data(), width, height, 1);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    if (!cpu_result) {
        fprintf(stderr, "CPU encode failed\n");
        vision_free(vm);
        gguf_close(g);
        return 1;
    }
    fprintf(stderr, "CPU encode time: %.1f ms\n", cpu_ms);

    // GPU encode
    fprintf(stderr, "\n=== GPU Encode ===\n");
    VulkanVisionEncoder gpu_encoder;

    if (!gpu_encoder.initialize(device_id, verbose)) {
        fprintf(stderr, "GPU init failed: %s\n", gpu_encoder.getLastError().c_str());
        free(cpu_result);
        vision_free(vm);
        gguf_close(g);
        return 1;
    }

    gpu_encoder.setAttentionMode(attn_mode);
    fprintf(stderr, "Attention mode: %s\n",
            attn_mode == VulkanVisionEncoder::ATTN_CPU ? "cpu" :
            attn_mode == VulkanVisionEncoder::ATTN_NAIVE_GPU ? "naive (GPU)" : "flash (GPU)");

    if (!gpu_encoder.loadWeights(g)) {
        fprintf(stderr, "GPU weight load failed: %s\n", gpu_encoder.getLastError().c_str());
        free(cpu_result);
        vision_free(vm);
        gguf_close(g);
        return 1;
    }

    auto gpu_start = std::chrono::high_resolution_clock::now();
    float *gpu_result = gpu_encoder.encode(rgb_norm.data(), width, height);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

    if (!gpu_result) {
        fprintf(stderr, "GPU encode failed: %s\n", gpu_encoder.getLastError().c_str());
        free(cpu_result);
        vision_free(vm);
        gguf_close(g);
        return 1;
    }
    fprintf(stderr, "GPU encode time: %.1f ms\n", gpu_ms);

    // Compare results
    fprintf(stderr, "\n=== Comparison ===\n");
    int total = n_merged * vm->proj_dim;
    float max_abs_err = 0.0f;
    double sum_abs_err = 0.0;
    int error_count = 0;

    for (int i = 0; i < total; i++) {
        float diff = fabsf(gpu_result[i] - cpu_result[i]);
        if (diff > max_abs_err) max_abs_err = diff;
        sum_abs_err += diff;
        if (diff > 1e-2f) error_count++;
    }

    float mean_abs_err = (float)(sum_abs_err / total);
    fprintf(stderr, "Output size: %d elements\n", total);
    fprintf(stderr, "Max abs error:  %e\n", max_abs_err);
    fprintf(stderr, "Mean abs error: %e\n", mean_abs_err);
    fprintf(stderr, "Errors > 1e-2:  %d / %d\n", error_count, total);

    // Print first few values for inspection
    if (verbose) {
        fprintf(stderr, "\nFirst 10 values:\n");
        for (int i = 0; i < std::min(10, total); i++) {
            fprintf(stderr, "  [%d] CPU=%.6f GPU=%.6f diff=%.6f\n",
                    i, cpu_result[i], gpu_result[i],
                    fabsf(gpu_result[i] - cpu_result[i]));
        }
    }

    bool passed = (max_abs_err < 1e-1f);  // Allow some tolerance for floating point differences
    fprintf(stderr, "\nResult: %s\n", passed ? "PASSED" : "FAILED");
    fprintf(stderr, "Speedup: %.1fx\n", cpu_ms / gpu_ms);

    delete[] gpu_result;
    free(cpu_result);
    vision_free(vm);
    gguf_close(g);

    return passed ? 0 : 1;
}
