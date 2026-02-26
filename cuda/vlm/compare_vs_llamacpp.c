/*
 * compare_vs_llamacpp.c - Compare CUDA vision encoder against llama.cpp reference
 *
 * Usage: ./compare_vs_llamacpp <mmproj.gguf> <image.jpg> <llamacpp_embd.bin> [--f16]
 *
 * Reads the reference embeddings from dump_clip_embd, runs the same image through
 * our CUDA vision encoder, and compares the outputs.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

/* stb_image for JPEG loading */
#define STB_IMAGE_IMPLEMENTATION
#include "../../common/stb_image.h"

/* GGUF loader for our encoder */
#define GGUF_LOADER_IMPLEMENTATION
#include "../../common/gguf_loader.h"

#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"

#include "../../common/transformer.h"

#define VISION_ENCODER_IMPLEMENTATION
#include "../../common/vision_encoder.h"

/* CUDA vision encoder */
#include "cuda_vision_encoder.h"

/* Simple bilinear resize */
static unsigned char *bilinear_resize(const unsigned char *src, int sw, int sh,
                                       int dw, int dh) {
    unsigned char *dst = (unsigned char *)malloc(dw * dh * 3);
    for (int dy = 0; dy < dh; dy++) {
        float fy = (sh > 1) ? (float)dy * (sh - 1) / (dh - 1) : 0;
        int y0 = (int)fy, y1 = (y0 + 1 < sh) ? y0 + 1 : y0;
        float wy = fy - y0;
        for (int dx = 0; dx < dw; dx++) {
            float fx = (sw > 1) ? (float)dx * (sw - 1) / (dw - 1) : 0;
            int x0 = (int)fx, x1 = (x0 + 1 < sw) ? x0 + 1 : x0;
            float wx = fx - x0;
            for (int c = 0; c < 3; c++) {
                float v = src[(y0*sw+x0)*3+c] * (1-wy)*(1-wx)
                        + src[(y0*sw+x1)*3+c] * (1-wy)*wx
                        + src[(y1*sw+x0)*3+c] * wy*(1-wx)
                        + src[(y1*sw+x1)*3+c] * wy*wx;
                dst[(dy*dw+dx)*3+c] = (unsigned char)(v + 0.5f);
            }
        }
    }
    return dst;
}

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <mmproj.gguf> <image.jpg> <llamacpp_embd.bin> [--f16]\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *image_path = argv[2];
    const char *embd_path = argv[3];
    int use_f16 = 0;
    for (int i = 4; i < argc; i++)
        if (strcmp(argv[i], "--f16") == 0) use_f16 = 1;

    /* Load reference embeddings */
    printf("=== Compare CUDA VLM vs llama.cpp ===\n");
    printf("Loading reference embeddings: %s\n", embd_path);
    FILE *fref = fopen(embd_path, "rb");
    if (!fref) { fprintf(stderr, "Cannot open %s\n", embd_path); return 1; }

    int32_t hdr[4];
    if (fread(hdr, sizeof(int32_t), 4, fref) != 4) {
        fprintf(stderr, "Bad header in %s\n", embd_path);
        fclose(fref);
        return 1;
    }
    int ref_n_tokens = hdr[0];
    int ref_embd_dim = hdr[1];
    int ref_img_w = hdr[2];
    int ref_img_h = hdr[3];
    printf("Reference: %d tokens x %d dim (image %dx%d)\n",
           ref_n_tokens, ref_embd_dim, ref_img_w, ref_img_h);

    int ref_total = ref_n_tokens * ref_embd_dim;
    float *ref_embd = (float *)malloc(ref_total * sizeof(float));
    if ((int)fread(ref_embd, sizeof(float), ref_total, fref) != ref_total) {
        fprintf(stderr, "Incomplete data in %s\n", embd_path);
        free(ref_embd);
        fclose(fref);
        return 1;
    }
    fclose(fref);

    /* Load image */
    printf("Loading image: %s\n", image_path);
    int img_w, img_h, img_c;
    unsigned char *img_data = stbi_load(image_path, &img_w, &img_h, &img_c, 3);
    if (!img_data) { fprintf(stderr, "Failed to load: %s\n", image_path); return 1; }
    printf("Image: %dx%d\n", img_w, img_h);

    /* Resize to match llama.cpp's preprocessed size */
    unsigned char *resized = NULL;
    if (img_w != ref_img_w || img_h != ref_img_h) {
        printf("Resizing %dx%d -> %dx%d (matching llama.cpp preprocessing)\n",
               img_w, img_h, ref_img_w, ref_img_h);
        resized = bilinear_resize(img_data, img_w, img_h, ref_img_w, ref_img_h);
        stbi_image_free(img_data);
        img_data = resized;
        img_w = ref_img_w;
        img_h = ref_img_h;
    }

    /* Load model */
    printf("Loading model: %s\n", model_path);
    gguf_context *gguf = gguf_open(model_path, 0);
    if (!gguf) { fprintf(stderr, "Failed to load GGUF\n"); return 1; }

    vision_model *vm = vision_load(gguf);
    if (!vm) { fprintf(stderr, "Failed to load vision model\n"); return 1; }

    /* Normalize image (HWC format, same as llama.cpp) */
    float *rgb_norm = vision_normalize_image(vm, img_data, img_w, img_h);
    if (resized) free(resized); else stbi_image_free(img_data);
    img_data = NULL;

    /* Init CUDA encoder */
    printf("Initializing CUDA encoder (f16=%d)...\n", use_f16);
    cuda_vision_runner *cuda_r = cuda_vision_init(0, 1, use_f16);
    if (!cuda_r) { fprintf(stderr, "CUDA init failed\n"); return 1; }
    if (cuda_vision_load_weights(cuda_r, gguf) != 0) {
        fprintf(stderr, "CUDA weight load failed\n");
        return 1;
    }

    /* Encode with CUDA */
    printf("\n--- CUDA Encoding ---\n");
    double t0 = get_time_ms();
    float *cuda_embd = cuda_vision_encode(cuda_r, rgb_norm, img_w, img_h);
    double t1 = get_time_ms();
    printf("CUDA time: %.1f ms\n", t1 - t0);

    if (!cuda_embd) {
        fprintf(stderr, "CUDA encoding failed\n");
        return 1;
    }

    /* Compute actual token count from image dimensions */
    int ps = vm->patch_size;
    int sp = vm->spatial_merge;
    int actual_gw = img_w / ps;
    int actual_gh = img_h / ps;
    int actual_patches = actual_gw * actual_gh;
    int cuda_n_merged = actual_patches / (sp * sp);
    int cuda_embd_dim = cuda_vision_total_embd(cuda_r);
    printf("CUDA output: %d tokens x %d dim (from %dx%d image)\n",
           cuda_n_merged, cuda_embd_dim, img_w, img_h);

    if (cuda_n_merged != ref_n_tokens) {
        fprintf(stderr, "WARNING: token count mismatch: CUDA=%d llama.cpp=%d\n",
                cuda_n_merged, ref_n_tokens);
    }
    if (cuda_embd_dim != ref_embd_dim) {
        fprintf(stderr, "WARNING: embd_dim mismatch: CUDA=%d llama.cpp=%d\n",
                cuda_embd_dim, ref_embd_dim);
    }

    /* Compare */
    int n_compare = (cuda_n_merged < ref_n_tokens ? cuda_n_merged : ref_n_tokens) *
                    (cuda_embd_dim < ref_embd_dim ? cuda_embd_dim : ref_embd_dim);
    float sum_diff2 = 0, sum_ref2 = 0;
    float max_diff = 0;
    int max_diff_idx = 0;
    int nan_count = 0;

    for (int i = 0; i < n_compare; i++) {
        if (isnan(cuda_embd[i]) || isnan(ref_embd[i])) { nan_count++; continue; }
        float diff = cuda_embd[i] - ref_embd[i];
        sum_diff2 += diff * diff;
        sum_ref2 += ref_embd[i] * ref_embd[i];
        if (fabsf(diff) > max_diff) {
            max_diff = fabsf(diff);
            max_diff_idx = i;
        }
    }

    float rel_l2 = (sum_ref2 > 0) ? sqrtf(sum_diff2 / sum_ref2) : sqrtf(sum_diff2);
    float threshold = use_f16 ? 0.05f : 0.02f;

    printf("\n=== CUDA vs llama.cpp (%s mode) ===\n", use_f16 ? "F16" : "F32");
    printf("  Elements:    %d\n", n_compare);
    printf("  Relative L2: %.6e", rel_l2);
    if (rel_l2 < threshold)
        printf("  [PASS < %.2f]\n", threshold);
    else
        printf("  [FAIL >= %.2f]\n", threshold);
    printf("  Max abs diff: %.6e (at index %d)\n", max_diff, max_diff_idx);
    if (max_diff_idx < n_compare) {
        int tok = max_diff_idx / ref_embd_dim;
        int dim = max_diff_idx % ref_embd_dim;
        printf("    token=%d dim=%d: CUDA=%.8f llama.cpp=%.8f\n",
               tok, dim, cuda_embd[max_diff_idx], ref_embd[max_diff_idx]);
    }
    if (nan_count > 0) printf("  NaN count: %d\n", nan_count);

    /* First 8 values */
    printf("\n  First 8 values:\n");
    for (int i = 0; i < 8 && i < n_compare; i++) {
        printf("    [%d] CUDA=%.8f  llama.cpp=%.8f  diff=%.6e\n",
               i, cuda_embd[i], ref_embd[i], cuda_embd[i] - ref_embd[i]);
    }

    /* Cosine similarity of first few tokens */
    printf("\n  Per-token cosine similarity (first 8 tokens):\n");
    for (int t = 0; t < 8 && t < ref_n_tokens && t < cuda_n_merged; t++) {
        float dot = 0, na = 0, nb = 0;
        int d = (ref_embd_dim < cuda_embd_dim) ? ref_embd_dim : cuda_embd_dim;
        for (int j = 0; j < d; j++) {
            float a = cuda_embd[t * cuda_embd_dim + j];
            float b = ref_embd[t * ref_embd_dim + j];
            dot += a * b;
            na += a * a;
            nb += b * b;
        }
        float cos_sim = dot / (sqrtf(na) * sqrtf(nb) + 1e-10f);
        printf("    token %d: cos_sim = %.8f\n", t, cos_sim);
    }

    /* Cleanup */
    free(ref_embd);
    free(cuda_embd);
    free(rgb_norm);
    cuda_vision_free(cuda_r);
    vision_free(vm);
    gguf_close(gguf);

    printf("\nDone.\n");
    return 0;
}
