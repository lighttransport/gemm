/*
 * test_cuda_vlm.c - End-to-end VLM inference: CUDA vision encoder + CUDA LLM
 *
 * Usage: ./test_cuda_vlm <model.gguf> <mmproj.gguf> <image.jpg> [-n max_tokens] [--resize dynamic|fit]
 *
 * Encodes an image through the CUDA vision encoder (mmproj), injects the
 * resulting embeddings into the CUDA LLM runner, and generates text.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* stb_image for JPEG loading */
#define STB_IMAGE_IMPLEMENTATION
#include "../../common/stb_image.h"

/* GGUF loader */
#define GGUF_LOADER_IMPLEMENTATION
#include "../../common/gguf_loader.h"

/* Dequantization */
#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"

/* BPE tokenizer */
#define BPE_TOKENIZER_IMPLEMENTATION
#include "../../common/bpe_tokenizer.h"

/* transformer.h for qtensor type */
#include "../../common/transformer.h"

/* CPU vision model for normalize_image + metadata */
#define VISION_ENCODER_IMPLEMENTATION
#include "../../common/vision_encoder.h"

/* CUDA runners */
#include "cuda_vision_encoder.h"
#include "../llm/cuda_llm_runner.h"

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

/* llama.cpp-style dynamic resolution: preserve aspect ratio within pixel budget */
static void calc_size_preserved_ratio(int orig_w, int orig_h, int align_size,
                                       int min_pixels, int max_pixels,
                                       int *out_w, int *out_h) {
    /* Round to nearest align_size */
    int w = (int)((orig_w + align_size / 2) / align_size) * align_size;
    int h = (int)((orig_h + align_size / 2) / align_size) * align_size;
    if (w < align_size) w = align_size;
    if (h < align_size) h = align_size;

    if ((long long)w * h > max_pixels) {
        /* Scale down: use floor rounding to stay under budget */
        float beta = sqrtf((float)orig_w * orig_h / max_pixels);
        w = (int)(orig_w / beta / align_size) * align_size;  /* floor */
        h = (int)(orig_h / beta / align_size) * align_size;  /* floor */
        if (w < align_size) w = align_size;
        if (h < align_size) h = align_size;
    } else if ((long long)w * h < min_pixels) {
        /* Scale up: use ceil rounding to meet minimum */
        float beta = sqrtf((float)min_pixels / (orig_w * orig_h));
        w = ((int)ceilf((float)(orig_w * beta) / align_size)) * align_size;
        h = ((int)ceilf((float)(orig_h * beta) / align_size)) * align_size;
    }

    *out_w = w;
    *out_h = h;
}

static int32_t argmax_i32(const float *x, int n) {
    int32_t best = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > x[best]) best = i;
    return best;
}

static const char *gguf_get_kv_string(const gguf_context *g, const char *key) {
    int idx = gguf_find_key(g, key);
    if (idx < 0) return NULL;
    if (g->kv[idx].type != GGUF_TYPE_STRING) return NULL;
    return g->kv[idx].value.str.str;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <model.gguf> <mmproj.gguf> <image.jpg> [-n max_tokens] [--resize dynamic|fit]\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *mmproj_path = argv[2];
    const char *image_path = argv[3];
    int max_gen = 100;
    int resize_mode = 0;  /* 0 = dynamic (default, matches llama.cpp), 1 = fit (simple) */
    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) max_gen = atoi(argv[++i]);
        else if (strcmp(argv[i], "--resize") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "fit") == 0) resize_mode = 1;
            else if (strcmp(argv[i], "dynamic") == 0) resize_mode = 0;
            else { fprintf(stderr, "Unknown resize mode: %s (use 'dynamic' or 'fit')\n", argv[i]); return 1; }
        }
    }

    fprintf(stderr, "=== CUDA VLM Pipeline ===\n");
    fprintf(stderr, "LLM model:   %s\n", model_path);
    fprintf(stderr, "Vision model: %s\n", mmproj_path);
    fprintf(stderr, "Image:       %s\n", image_path);
    fprintf(stderr, "Max gen:     %d\n\n", max_gen);

    /* ---- 1. Load image ---- */
    fprintf(stderr, "Loading image...\n");
    int img_w, img_h, img_c;
    unsigned char *img_data = stbi_load(image_path, &img_w, &img_h, &img_c, 3);
    if (!img_data) {
        fprintf(stderr, "Failed to load image: %s\n", image_path);
        return 1;
    }
    fprintf(stderr, "Image: %dx%d (%d channels)\n", img_w, img_h, img_c);

    /* ---- 2. Load mmproj GGUF ---- */
    fprintf(stderr, "Loading mmproj: %s\n", mmproj_path);
    gguf_context *gguf_mmproj = gguf_open(mmproj_path, 0);
    if (!gguf_mmproj) { fprintf(stderr, "Failed to open mmproj GGUF\n"); return 1; }

    /* Load CPU vision model for metadata (mean/std, patch_size, spatial_merge) */
    vision_model *vm = vision_load(gguf_mmproj);
    if (!vm) { fprintf(stderr, "Failed to load vision model metadata\n"); return 1; }

    /* ---- 3. Resize image ---- */
    int ps = vm->patch_size;
    int tile = ps * vm->spatial_merge;  /* typically 32 for Qwen3.5-VL */
    int new_w, new_h;

    int dyn_max_pixels = 4194304;  /* also passed to encoder */
    if (resize_mode == 0) {
        /* Dynamic mode: llama.cpp-style preserved ratio */
        int min_pixels = 8192;
        /* Try to read from GGUF metadata */
        int idx;
        idx = gguf_find_key(gguf_mmproj, "clip.vision.image_min_pixels");
        if (idx >= 0) min_pixels = (int)gguf_mmproj->kv[idx].value.u32;
        idx = gguf_find_key(gguf_mmproj, "clip.vision.image_max_pixels");
        if (idx >= 0) dyn_max_pixels = (int)gguf_mmproj->kv[idx].value.u32;

        calc_size_preserved_ratio(img_w, img_h, tile, min_pixels, dyn_max_pixels, &new_w, &new_h);
        fprintf(stderr, "Dynamic resize: %dx%d -> %dx%d (min_px=%d, max_px=%d)\n",
                img_w, img_h, new_w, new_h, min_pixels, dyn_max_pixels);
    } else {
        /* Fit mode: align to tile, scale down if exceeds max patches */
        int max_patches = vm->n_patches;
        new_w = (img_w / tile) * tile;
        new_h = (img_h / tile) * tile;
        if (new_w < tile) new_w = tile;
        if (new_h < tile) new_h = tile;

        int n_patches = (new_w / ps) * (new_h / ps);
        if (n_patches > max_patches) {
            float scale = sqrtf((float)max_patches / n_patches);
            new_w = ((int)(new_w * scale) / tile) * tile;
            new_h = ((int)(new_h * scale) / tile) * tile;
            if (new_w < tile) new_w = tile;
            if (new_h < tile) new_h = tile;
        }
        fprintf(stderr, "Fit resize: %dx%d -> %dx%d\n", img_w, img_h, new_w, new_h);
    }

    /* Bilinear resize */
    unsigned char *resized = NULL;
    if (new_w != img_w || new_h != img_h) {
        resized = bilinear_resize(img_data, img_w, img_h, new_w, new_h);
        stbi_image_free(img_data);
        img_data = resized;
        img_w = new_w;
        img_h = new_h;
    }

    /* ---- 4. Normalize image ---- */
    fprintf(stderr, "Normalizing image...\n");
    float *rgb_norm = vision_normalize_image(vm, img_data, img_w, img_h);
    if (resized) free(resized); else stbi_image_free(img_data);
    img_data = NULL;

    /* ---- 5. CUDA vision encode ---- */
    fprintf(stderr, "Initializing CUDA vision encoder...\n");
    cuda_vision_runner *cuda_vis = cuda_vision_init(0, 1, 0);  /* F32 mode */
    if (!cuda_vis) { fprintf(stderr, "CUDA vision init failed\n"); return 1; }
    cuda_vision_set_max_pixels(cuda_vis, dyn_max_pixels);
    if (cuda_vision_load_weights(cuda_vis, gguf_mmproj) != 0) {
        fprintf(stderr, "CUDA vision weight load failed\n");
        return 1;
    }

    fprintf(stderr, "Encoding image (%dx%d)...\n", img_w, img_h);
    double t0 = get_time_ms();
    float *vision_embd = cuda_vision_encode(cuda_vis, rgb_norm, img_w, img_h);
    double t1 = get_time_ms();
    free(rgb_norm);

    if (!vision_embd) {
        fprintf(stderr, "Vision encoding failed\n");
        return 1;
    }

    /* Compute actual merged token count from image dimensions */
    int patches_w = img_w / vm->patch_size;
    int patches_h = img_h / vm->patch_size;
    int n_vision_tokens = (patches_w / vm->spatial_merge) * (patches_h / vm->spatial_merge);
    int total_embd = cuda_vision_total_embd(cuda_vis);
    int proj_dim = cuda_vision_proj_dim(cuda_vis);
    fprintf(stderr, "Vision encoding: %d tokens x %d dim (proj=%d, %.1f ms)\n",
            n_vision_tokens, total_embd, proj_dim, t1 - t0);

    /* Print vision embedding stats */
    {
        float vmin = vision_embd[0], vmax = vision_embd[0], vnorm = 0;
        for (int i = 0; i < n_vision_tokens * total_embd; i++) {
            if (vision_embd[i] < vmin) vmin = vision_embd[i];
            if (vision_embd[i] > vmax) vmax = vision_embd[i];
            vnorm += vision_embd[i] * vision_embd[i];
        }
        fprintf(stderr, "Vision embeddings: min=%.4f max=%.4f L2=%.4f\n",
                vmin, vmax, sqrtf(vnorm));
    }

    /* Free vision encoder GPU weights — no longer needed after encoding */
    fprintf(stderr, "Freeing vision encoder to reclaim VRAM...\n");
    cuda_vision_free(cuda_vis);
    cuda_vis = NULL;
    gguf_close(gguf_mmproj);
    gguf_mmproj = NULL;

    /* ---- 6. Load LLM model ---- */
    fprintf(stderr, "\nLoading LLM model: %s\n", model_path);
    gguf_context *gguf_llm = gguf_open(model_path, 1);
    if (!gguf_llm) { fprintf(stderr, "Failed to open LLM GGUF\n"); return 1; }

    bpe_vocab *vocab = bpe_vocab_load(gguf_llm);
    if (!vocab) { fprintf(stderr, "Failed to load vocab\n"); return 1; }
    fprintf(stderr, "Vocab: %d tokens\n", vocab->n_tokens);

    /* Need enough seq_len for prompt + vision tokens + generation */
    int max_seq_len = n_vision_tokens + 256 + max_gen;
    if (max_seq_len < 1024) max_seq_len = 1024;

    fprintf(stderr, "Initializing CUDA LLM runner (max_seq_len=%d)...\n", max_seq_len);
    cuda_llm_runner *llm = cuda_llm_init(0, 1);
    if (!llm) { fprintf(stderr, "CUDA LLM init failed\n"); return 1; }
    if (cuda_llm_load_weights(llm, gguf_llm, max_seq_len) != 0) {
        fprintf(stderr, "CUDA LLM weight load failed\n");
        return 1;
    }

    int n_embd = cuda_llm_n_embd(llm);
    int n_vocab = cuda_llm_n_vocab(llm);
    fprintf(stderr, "LLM: n_embd=%d n_vocab=%d n_layers=%d\n",
            n_embd, n_vocab, cuda_llm_n_layers(llm));

    /* Verify vision proj_dim matches LLM n_embd */
    if (proj_dim != n_embd) {
        fprintf(stderr, "ERROR: vision proj_dim=%d != LLM n_embd=%d\n", proj_dim, n_embd);
        return 1;
    }

    /* ---- 7. Build multimodal prompt ---- */
    char text_before[256];
    char text_after[512];
    {
        const char *tmpl = gguf_get_kv_string(gguf_llm, "tokenizer.chat_template");
        (void)tmpl;  /* always use chatml+vision format */
        snprintf(text_before, sizeof(text_before), "<|im_start|>user\n<|vision_start|>");
        snprintf(text_after, sizeof(text_after),
                 "<|vision_end|>Explain the image<|im_end|>\n<|im_start|>assistant\n");
    }

    int32_t tokens_before[64], tokens_after[64];
    int n_before = bpe_tokenize(vocab, text_before, -1, tokens_before, 64);
    int n_after  = bpe_tokenize(vocab, text_after, -1, tokens_after, 64);

    fprintf(stderr, "\nTokens before vision (%d):", n_before);
    for (int i = 0; i < n_before; i++)
        fprintf(stderr, " %d", tokens_before[i]);
    fprintf(stderr, "\nTokens after vision (%d):", n_after);
    for (int i = 0; i < n_after; i++)
        fprintf(stderr, " %d", tokens_after[i]);
    fprintf(stderr, "\n");

    int total_prompt_len = n_before + n_vision_tokens + n_after;
    fprintf(stderr, "\nTotal prompt: %d tokens (%d text + %d vision + %d text)\n",
            total_prompt_len, n_before, n_vision_tokens, n_after);

    /* ---- 8. Prefill ---- */
    fprintf(stderr, "\n=== Prefill ===\n");
    int pos = 0;

    /* Text tokens before vision */
    t0 = get_time_ms();
    for (int i = 0; i < n_before; i++) {
        cuda_llm_forward(llm, tokens_before[i], pos);
        pos++;
    }
    t1 = get_time_ms();
    fprintf(stderr, "  Text before: %d tokens, %.1f ms\n", n_before, t1 - t0);

    /* Vision tokens */
    t0 = get_time_ms();
    for (int i = 0; i < n_vision_tokens; i++) {
        float *embd_i = vision_embd + i * total_embd;
        cuda_llm_forward_embd(llm, embd_i, total_embd, pos);
        pos++;
        if (i == 0 || i == n_vision_tokens - 1 || (i + 1) % 100 == 0) {
            double tc = get_time_ms();
            fprintf(stderr, "  Vision token %d/%d, pos=%d (%.1f ms cumulative)\n",
                    i + 1, n_vision_tokens, pos - 1, tc - t0);
        }
    }
    t1 = get_time_ms();
    fprintf(stderr, "  Vision prefill: %d tokens, %.1f ms (%.2f ms/token)\n",
            n_vision_tokens, t1 - t0, (t1 - t0) / n_vision_tokens);

    /* Text tokens after vision (last one returns logits) */
    float *logits = NULL;
    t0 = get_time_ms();
    for (int i = 0; i < n_after; i++) {
        if (i == n_after - 1) {
            logits = cuda_llm_forward_logits(llm, tokens_after[i], pos);
        } else {
            cuda_llm_forward(llm, tokens_after[i], pos);
        }
        pos++;
    }
    t1 = get_time_ms();
    fprintf(stderr, "  Text after: %d tokens, %.1f ms\n", n_after, t1 - t0);

    if (!logits) {
        fprintf(stderr, "ERROR: forward_logits returned NULL\n");
        return 1;
    }

    /* ---- 9. Generate ---- */
    fprintf(stderr, "\n=== Generation ===\n");
    int32_t eos_id = bpe_eos_id(vocab);
    int32_t eot_id = bpe_eot_id(vocab);

    int32_t next_token = argmax_i32(logits, n_vocab);

    for (int g = 0; g < max_gen; g++) {
        if (next_token == eos_id || next_token == eot_id) {
            fprintf(stderr, "\n  [EOS/EOT token %d at step %d]\n", next_token, g);
            break;
        }

        /* Print token */
        const char *tok_str = bpe_token_to_str(vocab, next_token);
        if (tok_str) {
            int dec_len;
            char *decoded = bpe_byte_decode(tok_str, (int)strlen(tok_str), &dec_len);
            fwrite(decoded, 1, dec_len, stdout);
            fflush(stdout);
            free(decoded);
        }

        /* Next step */
        logits = cuda_llm_forward_logits(llm, next_token, pos);
        pos++;
        if (!logits) break;
        next_token = argmax_i32(logits, n_vocab);
    }
    printf("\n");

    /* ---- 10. Cleanup ---- */
    fprintf(stderr, "\nCleaning up...\n");
    free(vision_embd);
    if (cuda_vis) cuda_vision_free(cuda_vis);
    vision_free(vm);
    cuda_llm_free(llm);
    bpe_vocab_free(vocab);
    if (gguf_mmproj) gguf_close(gguf_mmproj);
    gguf_close(gguf_llm);

    fprintf(stderr, "Done.\n");
    return 0;
}
