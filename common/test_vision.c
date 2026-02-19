/*
 * test_vision.c - End-to-end vision inference test for Qwen3-VL
 *
 * Usage:
 *   ./test_vision <model.gguf> <mmproj.gguf> [max_gen_tokens]
 *
 * Generates a synthetic checkerboard image, encodes it through the vision
 * encoder (mmproj), builds a multimodal prompt, and runs LLM generation.
 */

#define GGUF_LOADER_IMPLEMENTATION
#include "gguf_loader.h"

#define GGML_DEQUANT_IMPLEMENTATION
#include "ggml_dequant.h"

#define BPE_TOKENIZER_IMPLEMENTATION
#include "bpe_tokenizer.h"

#define TRANSFORMER_IMPLEMENTATION
#include "transformer.h"

#define VISION_ENCODER_IMPLEMENTATION
#include "vision_encoder.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static const char *gguf_get_kv_string(const gguf_context *g, const char *key) {
    int idx = gguf_find_key(g, key);
    if (idx < 0) return NULL;
    if (g->kv[idx].type != GGUF_TYPE_STRING) return NULL;
    return g->kv[idx].value.str.str;
}

static void build_multimodal_chat_prompt(const gguf_context *gguf_main,
                                         const char *user_prompt,
                                         char *text_before, size_t text_before_cap,
                                         char *text_after, size_t text_after_cap) {
    const char *tmpl = gguf_get_kv_string(gguf_main, "tokenizer.chat_template");
    int has_chatml = tmpl &&
                     strstr(tmpl, "<|im_start|>") &&
                     strstr(tmpl, "<|im_end|>");
    int has_vision = tmpl &&
                     strstr(tmpl, "<|vision_start|>") &&
                     strstr(tmpl, "<|vision_end|>");

    if (has_chatml && has_vision) {
        snprintf(text_before, text_before_cap, "<|im_start|>user\n<|vision_start|>");
        snprintf(text_after, text_after_cap, "<|vision_end|>%s<|im_end|>\n<|im_start|>assistant\n", user_prompt);
        fprintf(stderr, "Chat template: tokenizer.chat_template detected (chatml+vision)\n");
        return;
    }

    snprintf(text_before, text_before_cap, "<|im_start|>user\n<|vision_start|>");
    snprintf(text_after, text_after_cap, "<|vision_end|>%s<|im_end|>\n<|im_start|>assistant\n", user_prompt);
    if (tmpl) {
        fprintf(stderr, "Chat template: unsupported template format, using chatml fallback\n");
    } else {
        fprintf(stderr, "Chat template: missing tokenizer.chat_template, using chatml fallback\n");
    }
}

/* Generate a checkerboard pattern as uint8 RGB [h][w][3] */
static uint8_t *generate_checkerboard(int width, int height, int cell_size) {
    uint8_t *img = (uint8_t *)malloc(width * height * 3);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int checker = ((x / cell_size) + (y / cell_size)) & 1;
            uint8_t val = checker ? 255 : 0;
            int idx = (y * width + x) * 3;
            img[idx + 0] = val;
            img[idx + 1] = val;
            img[idx + 2] = val;
        }
    }
    return img;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <mmproj.gguf> [max_gen] [image_size]\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *mmproj_path = argv[2];
    int max_gen = (argc >= 4) ? atoi(argv[3]) : 30;
    int img_size_override = (argc >= 5) ? atoi(argv[4]) : 192; /* default small for speed */

    srand((unsigned)time(NULL));

    /* ---- Load main model ---- */
    fprintf(stderr, "Loading main model: %s\n", model_path);
    gguf_context *gguf_main = gguf_open(model_path, 1);
    if (!gguf_main) { fprintf(stderr, "Failed to open main model\n"); return 1; }

    bpe_vocab *vocab = bpe_vocab_load(gguf_main);
    if (!vocab) { fprintf(stderr, "Failed to load vocab\n"); return 1; }
    fprintf(stderr, "Vocab: %d tokens\n", vocab->n_tokens);

    /* We need enough seq len for: chat template tokens + 576 vision tokens + generation */
    int max_seq_len = 1024;
    transformer_model *model = transformer_load(gguf_main, max_seq_len);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    /* ---- Load vision encoder ---- */
    fprintf(stderr, "Loading mmproj: %s\n", mmproj_path);
    gguf_context *gguf_mmproj = gguf_open(mmproj_path, 1);
    if (!gguf_mmproj) { fprintf(stderr, "Failed to open mmproj\n"); return 1; }

    vision_model *vm = vision_load(gguf_mmproj);
    if (!vm) { fprintf(stderr, "Failed to load vision model\n"); return 1; }

    /* ---- Generate and encode image ---- */
    int img_w = img_size_override, img_h = img_size_override;
    if (img_w % vm->patch_size != 0) {
        fprintf(stderr, "Image size %d must be multiple of patch size %d\n", img_w, vm->patch_size);
        return 1;
    }
    fprintf(stderr, "Generating %dx%d checkerboard image...\n", img_w, img_h);
    uint8_t *img_rgb = generate_checkerboard(img_w, img_h, 64);

    fprintf(stderr, "Normalizing image...\n");
    float *img_norm = vision_normalize_image(vm, img_rgb, img_w, img_h);
    free(img_rgb);

    fprintf(stderr, "Encoding image through vision encoder...\n");
    clock_t t0 = clock();
    float *vision_embd = vision_encode(vm, img_norm, img_w, img_h);
    clock_t t1 = clock();
    free(img_norm);

    if (!vision_embd) { fprintf(stderr, "Vision encoding failed\n"); return 1; }

    int patches_w = img_w / vm->patch_size;
    int patches_h = img_h / vm->patch_size;
    int n_vision_tokens = (patches_w / vm->spatial_merge) * (patches_h / vm->spatial_merge);
    int proj_dim = vm->proj_dim;
    fprintf(stderr, "Vision encoding done: %d tokens x %d dim (%.1f s)\n",
            n_vision_tokens, proj_dim, (double)(t1 - t0) / CLOCKS_PER_SEC);

    /* Print vision embedding stats */
    {
        float vmin = vision_embd[0], vmax = vision_embd[0], vnorm = 0;
        for (int i = 0; i < n_vision_tokens * proj_dim; i++) {
            if (vision_embd[i] < vmin) vmin = vision_embd[i];
            if (vision_embd[i] > vmax) vmax = vision_embd[i];
            vnorm += vision_embd[i] * vision_embd[i];
        }
        fprintf(stderr, "Vision embeddings: min=%.4f max=%.4f norm=%.4f\n",
                vmin, vmax, sqrtf(vnorm));
    }

    /* ---- Build token sequence ---- */
    char text_before[256];
    char text_after[512];
    build_multimodal_chat_prompt(gguf_main, "Explain the image",
                                 text_before, sizeof(text_before),
                                 text_after, sizeof(text_after));

    int32_t tokens_before[64], tokens_after[64];
    int n_before = bpe_tokenize(vocab, text_before, -1, tokens_before, 64);
    int n_after  = bpe_tokenize(vocab, text_after, -1, tokens_after, 64);

    fprintf(stderr, "\nTokens before vision (%d):", n_before);
    for (int i = 0; i < n_before; i++)
        fprintf(stderr, " %d(\"%s\")", tokens_before[i], bpe_token_to_str(vocab, tokens_before[i]));
    fprintf(stderr, "\nTokens after vision (%d):", n_after);
    for (int i = 0; i < n_after; i++)
        fprintf(stderr, " %d(\"%s\")", tokens_after[i], bpe_token_to_str(vocab, tokens_after[i]));
    fprintf(stderr, "\n");

    int total_prompt_len = n_before + n_vision_tokens + n_after;
    fprintf(stderr, "\nTotal prompt: %d tokens (%d text + %d vision + %d text)\n",
            total_prompt_len, n_before, n_vision_tokens, n_after);

    /* ---- Prefill ---- */
    fprintf(stderr, "\n=== Prefill ===\n");
    int pos = 0;

    /* Process text tokens before vision */
    for (int i = 0; i < n_before; i++) {
        fprintf(stderr, "  [prefill %d/%d] token=%d pos=%d\n", pos + 1, total_prompt_len, tokens_before[i], pos);
        t0 = clock();
        transformer_forward(model, tokens_before[i], pos);
        t1 = clock();
        fprintf(stderr, "    %.3f s\n", (double)(t1 - t0) / CLOCKS_PER_SEC);
        pos++;
    }

    /* Process vision embedding tokens */
    fprintf(stderr, "  [prefill vision %d tokens, pos %d-%d]\n",
            n_vision_tokens, pos, pos + n_vision_tokens - 1);
    t0 = clock();
    for (int i = 0; i < n_vision_tokens; i++) {
        float *embd_i = vision_embd + i * proj_dim;
        transformer_forward_embd(model, embd_i, pos);
        pos++;
        if (i == 0 || i == n_vision_tokens - 1 || (i + 1) % 100 == 0) {
            clock_t tc = clock();
            fprintf(stderr, "    vision token %d/%d, pos=%d (%.1f s cumulative)\n",
                    i + 1, n_vision_tokens, pos - 1, (double)(tc - t0) / CLOCKS_PER_SEC);
        }
    }
    t1 = clock();
    fprintf(stderr, "  Vision prefill: %.1f s (%.2f s/token)\n",
            (double)(t1 - t0) / CLOCKS_PER_SEC,
            (double)(t1 - t0) / CLOCKS_PER_SEC / n_vision_tokens);

    /* Process text tokens after vision */
    for (int i = 0; i < n_after; i++) {
        fprintf(stderr, "  [prefill %d/%d] token=%d pos=%d\n", pos + 1, total_prompt_len, tokens_after[i], pos);
        t0 = clock();
        float *logits = NULL;
        if (i == n_after - 1) {
            logits = transformer_forward_logits(model, tokens_after[i], pos);
        } else {
            transformer_forward(model, tokens_after[i], pos);
        }
        t1 = clock();
        fprintf(stderr, "    %.3f s\n", (double)(t1 - t0) / CLOCKS_PER_SEC);

        if (logits) {
            int32_t argmax = 0;
            for (int j = 1; j < model->n_vocab; j++) {
                if (logits[j] > logits[argmax]) argmax = j;
            }
            fprintf(stderr, "  Greedy next: %d \"%s\" (logit=%.4f)\n",
                    argmax, bpe_token_to_str(vocab, argmax), logits[argmax]);
        }
        pos++;
    }

    free(vision_embd);

    /* ---- Generation ---- */
    fprintf(stderr, "\n--- Generation ---\n");

    int32_t next_token = -1;
    for (int g = 0; g < max_gen; g++) {
        float *logits;
        if (g == 0) {
            logits = model->logits;
        } else {
            t0 = clock();
            logits = transformer_forward_logits(model, next_token, pos);
            t1 = clock();
            fprintf(stderr, "  [gen %d] pos=%d %.3f s\n", g, pos, (double)(t1 - t0) / CLOCKS_PER_SEC);
            pos++;
        }

        if (!logits) break;

        /* Greedy */
        next_token = 0;
        for (int j = 1; j < model->n_vocab; j++) {
            if (logits[j] > logits[next_token]) next_token = j;
        }

        if (next_token == vocab->eos_id || next_token == vocab->eot_id) {
            fprintf(stderr, "  [EOS token %d]\n", next_token);
            break;
        }

        const char *tok_str = bpe_token_to_str(vocab, next_token);
        if (tok_str) {
            int dec_len;
            char *decoded = bpe_byte_decode(tok_str, (int)strlen(tok_str), &dec_len);
            fwrite(decoded, 1, dec_len, stdout);
            fflush(stdout);
            free(decoded);
        }

        if (g == 0) pos++;
    }
    printf("\n");

    /* Cleanup */
    vision_free(vm);
    transformer_free(model);
    bpe_vocab_free(vocab);
    gguf_close(gguf_mmproj);
    gguf_close(gguf_main);

    fprintf(stderr, "\nDone.\n");
    return 0;
}
