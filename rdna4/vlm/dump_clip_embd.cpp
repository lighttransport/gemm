/*
 * dump_clip_embd.cpp - Dump llama.cpp CLIP embeddings for RDNA4 vision checks.
 *
 * Usage:
 *   ./dump_clip_embd <mmproj.gguf> <image.jpg> [output.bin] [--image-size N]
 */

#define STB_IMAGE_IMPLEMENTATION
#include "../../common/stb_image.h"

#include "clip.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

static unsigned char *bilinear_resize(const unsigned char *src, int sw, int sh, int dw, int dh) {
    unsigned char *dst = (unsigned char *)malloc((size_t)dw * dh * 3);
    if (!dst) return nullptr;
    for (int y = 0; y < dh; y++) {
        float fy = (dh > 1) ? (float)y * (sh - 1) / (dh - 1) : 0.0f;
        int y0 = (int)fy;
        int y1 = (y0 + 1 < sh) ? y0 + 1 : y0;
        float wy = fy - y0;
        for (int x = 0; x < dw; x++) {
            float fx = (dw > 1) ? (float)x * (sw - 1) / (dw - 1) : 0.0f;
            int x0 = (int)fx;
            int x1 = (x0 + 1 < sw) ? x0 + 1 : x0;
            float wx = fx - x0;
            for (int c = 0; c < 3; c++) {
                float v = src[(y0 * sw + x0) * 3 + c] * (1.0f - wy) * (1.0f - wx)
                        + src[(y0 * sw + x1) * 3 + c] * (1.0f - wy) * wx
                        + src[(y1 * sw + x0) * 3 + c] * wy * (1.0f - wx)
                        + src[(y1 * sw + x1) * 3 + c] * wy * wx;
                dst[(y * dw + x) * 3 + c] = (unsigned char)(v + 0.5f);
            }
        }
    }
    return dst;
}

static int get_u32(const gguf_context *g, const char *key, int def) {
    int64_t idx = gguf_find_key(g, key);
    if (idx < 0) return def;
    enum gguf_type type = gguf_get_kv_type(g, idx);
    if (type == GGUF_TYPE_UINT32) return (int)gguf_get_val_u32(g, idx);
    if (type == GGUF_TYPE_INT32) return gguf_get_val_i32(g, idx);
    return def;
}

static void get_f32x3(const gguf_context *g, const char *key, float v[3], float def) {
    v[0] = v[1] = v[2] = def;
    int64_t idx = gguf_find_key(g, key);
    if (idx < 0 || gguf_get_kv_type(g, idx) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(g, idx) != GGUF_TYPE_FLOAT32 || gguf_get_arr_n(g, idx) < 3) {
        return;
    }
    const float *d = (const float *)gguf_get_arr_data(g, idx);
    v[0] = d[0];
    v[1] = d[1];
    v[2] = d[2];
}

static void calc_size_preserved_ratio(int orig_w, int orig_h, int align_size,
                                      int min_pixels, int max_pixels,
                                      int *out_w, int *out_h) {
    int w = (int)((orig_w + align_size / 2) / align_size) * align_size;
    int h = (int)((orig_h + align_size / 2) / align_size) * align_size;
    if (w < align_size) w = align_size;
    if (h < align_size) h = align_size;

    if ((long long)w * h > max_pixels) {
        float beta = sqrtf((float)orig_w * orig_h / (float)max_pixels);
        w = (int)(orig_w / beta / align_size) * align_size;
        h = (int)(orig_h / beta / align_size) * align_size;
        if (w < align_size) w = align_size;
        if (h < align_size) h = align_size;
    } else if ((long long)w * h < min_pixels) {
        float beta = sqrtf((float)min_pixels / ((float)orig_w * orig_h));
        w = ((int)ceilf((float)orig_w * beta / align_size)) * align_size;
        h = ((int)ceilf((float)orig_h * beta / align_size)) * align_size;
    }

    *out_w = w;
    *out_h = h;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <mmproj.gguf> <image.jpg> [output.bin] [--image-size N]\n", argv[0]);
        return 1;
    }

    const char *mmproj_path = argv[1];
    const char *image_path = argv[2];
    const char *output_path = "llamacpp_embd.bin";
    int image_size = 0;

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--image-size") == 0 && i + 1 < argc) image_size = atoi(argv[++i]);
        else output_path = argv[i];
    }

    gguf_init_params gguf_params = {};
    gguf_params.no_alloc = true;
    gguf_params.ctx = nullptr;
    gguf_context *gguf = gguf_init_from_file(mmproj_path, gguf_params);
    if (!gguf) {
        fprintf(stderr, "Failed to open GGUF: %s\n", mmproj_path);
        return 1;
    }

    int patch_size = get_u32(gguf, "clip.vision.patch_size", 16);
    int spatial_merge = get_u32(gguf, "clip.vision.spatial_merge_size", 2);
    int align = patch_size * spatial_merge;
    int min_pixels = get_u32(gguf, "clip.vision.image_min_pixels", 8192);
    int max_pixels = get_u32(gguf, "clip.vision.image_max_pixels", 4194304);
    float mean[3], std[3];
    get_f32x3(gguf, "clip.vision.image_mean", mean, 0.5f);
    get_f32x3(gguf, "clip.vision.image_std", std, 0.5f);

    int img_w, img_h, img_c;
    unsigned char *img = stbi_load(image_path, &img_w, &img_h, &img_c, 3);
    if (!img) {
        fprintf(stderr, "Failed to load image: %s\n", image_path);
        gguf_free(gguf);
        return 1;
    }

    int target_w, target_h;
    if (image_size > 0) {
        target_w = (image_size / align) * align;
        if (target_w < align) target_w = align;
        target_h = target_w;
    } else {
        calc_size_preserved_ratio(img_w, img_h, align, min_pixels, max_pixels, &target_w, &target_h);
    }

    if (target_w != img_w || target_h != img_h) {
        unsigned char *resized = bilinear_resize(img, img_w, img_h, target_w, target_h);
        stbi_image_free(img);
        img = resized;
        if (!img) {
            fprintf(stderr, "Resize failed\n");
            gguf_free(gguf);
            return 1;
        }
    }

    float *norm = (float *)malloc((size_t)target_w * target_h * 3 * sizeof(float));
    if (!norm) {
        stbi_image_free(img);
        gguf_free(gguf);
        return 1;
    }
    for (int i = 0; i < target_w * target_h; i++) {
        norm[i * 3 + 0] = ((float)img[i * 3 + 0] / 255.0f - mean[0]) / std[0];
        norm[i * 3 + 1] = ((float)img[i * 3 + 1] / 255.0f - mean[1]) / std[1];
        norm[i * 3 + 2] = ((float)img[i * 3 + 2] / 255.0f - mean[2]) / std[2];
    }
    stbi_image_free(img);
    gguf_free(gguf);

    clip_context_params params = {};
    params.use_gpu = true;
    params.flash_attn_type = CLIP_FLASH_ATTN_TYPE_DISABLED;
    params.warmup = false;
    clip_init_result init_res = clip_init(mmproj_path, params);
    clip_ctx *ctx = init_res.ctx_v;
    if (!ctx) {
        fprintf(stderr, "clip_init failed: %s\n", mmproj_path);
        free(norm);
        return 1;
    }

    int embd_dim = clip_n_mmproj_embd(ctx);
    size_t embd_bytes = clip_embd_nbytes_by_img(ctx, target_w, target_h);
    int n_tokens = (int)(embd_bytes / ((size_t)embd_dim * sizeof(float)));
    float *embd = (float *)malloc(embd_bytes);
    if (!embd) {
        clip_free(ctx);
        free(norm);
        return 1;
    }

    if (!clip_encode_float_image(ctx, 4, norm, target_h, target_w, embd)) {
        fprintf(stderr, "clip_encode_float_image failed\n");
        free(embd);
        clip_free(ctx);
        free(norm);
        return 1;
    }
    free(norm);

    FILE *f = fopen(output_path, "wb");
    if (!f) {
        fprintf(stderr, "Cannot write: %s\n", output_path);
        free(embd);
        clip_free(ctx);
        return 1;
    }
    int32_t hdr[4] = { n_tokens, embd_dim, target_w, target_h };
    fwrite(hdr, sizeof(int32_t), 4, f);
    fwrite(embd, sizeof(float), (size_t)n_tokens * embd_dim, f);
    fclose(f);

    printf("Written: %s (%d tokens x %d dim, image %dx%d)\n",
           output_path, n_tokens, embd_dim, target_w, target_h);

    free(embd);
    clip_free(ctx);
    return 0;
}
