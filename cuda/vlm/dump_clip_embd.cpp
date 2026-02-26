/*
 * dump_clip_embd.cpp - Dump llama.cpp CLIP vision embeddings to binary file
 *
 * Links against libmtmd.so from llama.cpp build.
 * Uses clip_image_preprocess + clip_image_encode to produce reference embeddings.
 *
 * Usage: ./dump_clip_embd <mmproj.gguf> <image.jpg> [output.bin]
 *
 * Output binary format:
 *   int32: n_tokens
 *   int32: embd_dim
 *   int32: image_w (preprocessed)
 *   int32: image_h (preprocessed)
 *   float[n_tokens * embd_dim]: embedding data
 */

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "clip.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <mmproj.gguf> <image.jpg> [output.bin]\n", argv[0]);
        return 1;
    }

    const char *mmproj_path = argv[1];
    const char *image_path = argv[2];
    const char *output_path = (argc > 3) ? argv[3] : "llamacpp_embd.bin";

    /* Load image */
    int img_w, img_h, img_c;
    unsigned char *img_data = stbi_load(image_path, &img_w, &img_h, &img_c, 3);
    if (!img_data) {
        fprintf(stderr, "Failed to load image: %s\n", image_path);
        return 1;
    }
    printf("Loaded image: %s (%dx%d)\n", image_path, img_w, img_h);

    /* Init clip context */
    clip_context_params ctx_params = {};
    ctx_params.use_gpu = true;
    ctx_params.flash_attn_type = CLIP_FLASH_ATTN_TYPE_DISABLED;
    ctx_params.warmup = false;

    clip_init_result init_res = clip_init(mmproj_path, ctx_params);
    clip_ctx *ctx = init_res.ctx_v;
    if (!ctx) {
        fprintf(stderr, "Failed to load clip model: %s\n", mmproj_path);
        stbi_image_free(img_data);
        return 1;
    }

    int image_size = clip_get_image_size(ctx);
    int patch_size = clip_get_patch_size(ctx);
    int hidden_size = clip_get_hidden_size(ctx);
    int embd_dim = clip_n_mmproj_embd(ctx);
    printf("Model: image_size=%d patch_size=%d hidden_size=%d embd_dim=%d\n",
           image_size, patch_size, hidden_size, embd_dim);

    /* Create clip_image_u8 from raw pixels */
    clip_image_u8 *img_u8 = clip_image_u8_init();
    clip_build_img_from_pixels(img_data, img_w, img_h, img_u8);
    stbi_image_free(img_data);

    /* Preprocess: resize + normalize (Qwen preserves aspect ratio, aligns to patch_size*2) */
    clip_image_f32_batch *batch = clip_image_f32_batch_init();
    if (!clip_image_preprocess(ctx, img_u8, batch)) {
        fprintf(stderr, "clip_image_preprocess failed\n");
        clip_image_u8_free(img_u8);
        clip_free(ctx);
        return 1;
    }
    clip_image_u8_free(img_u8);

    size_t n_images = clip_image_f32_batch_n_images(batch);
    if (n_images == 0) {
        fprintf(stderr, "No preprocessed images\n");
        clip_image_f32_batch_free(batch);
        clip_free(ctx);
        return 1;
    }

    int pp_w = (int)clip_image_f32_batch_nx(batch, 0);
    int pp_h = (int)clip_image_f32_batch_ny(batch, 0);
    printf("Preprocessed image: %dx%d\n", pp_w, pp_h);

    clip_image_f32 *img_f32 = clip_image_f32_get_img(batch, 0);

    int n_tokens = clip_n_output_tokens(ctx, img_f32);
    printf("Output: %d tokens x %d embd_dim\n", n_tokens, embd_dim);

    /* Encode */
    printf("Encoding with llama.cpp CLIP...\n");
    size_t embd_bytes = clip_embd_nbytes_by_img(ctx, pp_w, pp_h);
    float *embd = (float *)malloc(embd_bytes);
    if (!clip_image_encode(ctx, 4, img_f32, embd)) {
        fprintf(stderr, "clip_image_encode failed\n");
        free(embd);
        clip_image_f32_batch_free(batch);
        clip_free(ctx);
        return 1;
    }
    printf("Encoding done.\n");

    /* Write binary output */
    {
        FILE *f = fopen(output_path, "wb");
        if (!f) { fprintf(stderr, "Cannot write %s\n", output_path); return 1; }
        int32_t hdr[4] = { (int32_t)n_tokens, (int32_t)embd_dim, (int32_t)pp_w, (int32_t)pp_h };
        fwrite(hdr, sizeof(int32_t), 4, f);
        fwrite(embd, sizeof(float), (size_t)n_tokens * embd_dim, f);
        fclose(f);
        printf("Written: %s (%d tokens x %d dim, image %dx%d)\n",
               output_path, n_tokens, embd_dim, pp_w, pp_h);
    }

    /* Print first values and stats */
    printf("\nFirst 16 values:\n");
    for (int i = 0; i < 16 && i < n_tokens * embd_dim; i++)
        printf("  [%d] = %.8f\n", i, embd[i]);

    float sum = 0, sum2 = 0, mn = embd[0], mx = embd[0];
    int total = n_tokens * embd_dim;
    for (int i = 0; i < total; i++) {
        sum += embd[i]; sum2 += embd[i] * embd[i];
        if (embd[i] < mn) mn = embd[i];
        if (embd[i] > mx) mx = embd[i];
    }
    printf("\nStats: mean=%.6f std=%.6f min=%.6f max=%.6f\n",
           sum/total, sqrtf(sum2/total - (sum/total)*(sum/total)), mn, mx);

    free(embd);
    clip_image_f32_batch_free(batch);
    clip_free(ctx);
    return 0;
}
