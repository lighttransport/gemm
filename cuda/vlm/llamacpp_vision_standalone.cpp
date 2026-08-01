/* Standalone vision encoder using llama.cpp: 
 * Called as a subprocess by test_cuda_vlm to avoid symbol conflicts.
 * Usage: llamacpp_vision_standalone <mmproj.gguf> <image.jpg> <w> <h> <out_embd.bin>
 * Writes vision embeddings to out_embd.bin: [n_tokens, proj_dim] float32.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "llama.h"
#include "clip.h"
#include "clip-impl.h"
#include "stb_image.h"

/* From stb_image */
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

static float *load_and_normalize_image(const char *path, int *w, int *h) {
    int c;
    unsigned char *img = stbi_load(path, w, h, &c, 3);
    if (!img) { fprintf(stderr, "Failed to load image: %s\n", path); return NULL; }
    float *norm = (float *)malloc((size_t)(*w) * (*h) * 3 * sizeof(float));
    for (int i = 0; i < (*w) * (*h); i++) {
        norm[i*3+0] = ((float)img[i*3+0] / 255.0f - 0.5f) / 0.5f;
        norm[i*3+1] = ((float)img[i*3+1] / 255.0f - 0.5f) / 0.5f;
        norm[i*3+2] = ((float)img[i*3+2] / 255.0f - 0.5f) / 0.5f;
    }
    stbi_image_free(img);
    return norm;
}

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "Usage: %s <mmproj.gguf> <image.jpg> <out_w> <out_h> <output.bin>\n", argv[0]);
        return 1;
    }
    const char *mmproj_path = argv[1];
    const char *image_path = argv[2];
    int target_w = atoi(argv[3]);
    int target_h = atoi(argv[4]);
    const char *out_path = argv[5];

    // Load image and pre-resize to target dimensions (matching test_cuda_vlm's resize)
    int orig_w, orig_h;
    float *rgb_norm = load_and_normalize_image(image_path, &orig_w, &orig_h);
    if (!rgb_norm) return 1;
    fprintf(stderr, "llamacpp_vision: loaded image %dx%d\n", orig_w, orig_h);

    // Pre-resize to match test_cuda_vlm's dynamic resize (bilinear, align_corners=True)
    float *resized = (float *)malloc((size_t)target_w * target_h * 3 * sizeof(float));
    for (int dy = 0; dy < target_h; dy++) {
        float fy = (orig_h > 1) ? (float)dy * (orig_h - 1) / (target_h - 1) : 0;
        int y0 = (int)fy, y1 = (y0 + 1 < orig_h) ? y0 + 1 : y0;
        float wy = fy - y0;
        for (int dx = 0; dx < target_w; dx++) {
            float fx = (orig_w > 1) ? (float)dx * (orig_w - 1) / (target_w - 1) : 0;
            int x0 = (int)fx, x1 = (x0 + 1 < orig_w) ? x0 + 1 : x0;
            float wx = fx - x0;
            for (int c = 0; c < 3; c++) {
                float v = rgb_norm[(y0*orig_w+x0)*3+c] * (1-wy)*(1-wx)
                        + rgb_norm[(y0*orig_w+x1)*3+c] * (1-wy)*wx
                        + rgb_norm[(y1*orig_w+x0)*3+c] * wy*(1-wx)
                        + rgb_norm[(y1*orig_w+x1)*3+c] * wy*wx;
                resized[(dy*target_w+dx)*3+c] = v;
            }
        }
    }
    free(rgb_norm);
    fprintf(stderr, "llamacpp_vision: resized to %dx%d\n", target_w, target_h);

    // Initialize llama backend (use CUDA unified memory to avoid OOM with large vision graphs)
    setenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY", "1", 0);
    llama_backend_init();

    struct clip_context_params params;
    std::memset(&params, 0, sizeof(params));
    params.use_gpu = true;
    params.flash_attn_type = CLIP_FLASH_ATTN_TYPE_AUTO;
    params.warmup = true;

    auto result = clip_init(mmproj_path, params);
    if (!result.ctx_v) {
        fprintf(stderr, "llamacpp_vision: clip_init failed\n");
        free(resized);
        llama_backend_free();
        return 1;
    }

    struct clip_ctx *clip = result.ctx_v;
    int proj_dim = clip_n_mmproj_embd(clip);
    fprintf(stderr, "llamacpp_vision: proj_dim=%d\n", proj_dim);

    // Get output token count (now using pre-resized dimensions)
    size_t nbytes = clip_embd_nbytes_by_img(clip, target_w, target_h);
    int n_tokens = (int)(nbytes / (proj_dim * sizeof(float)));
    fprintf(stderr, "llamacpp_vision: n_tokens=%d (from %dx%d)\n", n_tokens, target_w, target_h);

    // Allocate output buffer and encode the pre-resized image
    float *embd = (float *)malloc(nbytes);
    if (!embd) { fprintf(stderr, "Out of memory\n"); return 1; }

    bool ok = clip_encode_float_image(clip, 1, resized, target_h, target_w, embd);
    free(resized);

    if (!ok) {
        fprintf(stderr, "llamacpp_vision: encode failed\n");
        free(embd);
        clip_free(clip);
        llama_backend_free();
        return 1;
    }

    // Write output file
    FILE *fout = fopen(out_path, "wb");
    if (!fout) { fprintf(stderr, "Cannot write %s\n", out_path); return 1; }
    int32_t hdr[4] = { n_tokens, proj_dim, target_w, target_h };
    fwrite(hdr, sizeof(int32_t), 4, fout);
    fwrite(embd, sizeof(float), (size_t)n_tokens * proj_dim, fout);
    fclose(fout);
    fprintf(stderr, "llamacpp_vision: wrote %d tokens x %d dim to %s\n", n_tokens, proj_dim, out_path);

    free(embd);
    clip_free(clip);
    llama_backend_free();
    return 0;
}
