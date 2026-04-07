/*
 * test_gemma4_vision.c - End-to-end vision inference test for Gemma4
 *
 * Usage:
 *   ./test_gemma4_vision <model.gguf> <mmproj.gguf> [image] [prompt] [max_gen]
 *
 * If no image is provided, generates a synthetic checkerboard test pattern.
 * Supports JPEG, PNG, BMP, PPM, and other formats via stb_image.
 */

#define GGUF_LOADER_IMPLEMENTATION
#include "../../common/gguf_loader.h"

#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"

#define BPE_TOKENIZER_IMPLEMENTATION
#include "../../common/bpe_tokenizer.h"

#define TRANSFORMER_IMPLEMENTATION
#include "../../common/transformer.h"

#define GEMMA4_VISION_IMPLEMENTATION
#include "../../common/gemma4_vision_encoder.h"

#define IMAGE_UTILS_IMPLEMENTATION
#include "../../common/image_utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Generate a synthetic checkerboard test image (224x224 RGB) */
static uint8_t *generate_test_image(int size) {
    uint8_t *img = (uint8_t *)malloc(size * size * 3);
    int block = size / 8;
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            int checker = ((y / block) + (x / block)) % 2;
            uint8_t val = checker ? 220 : 35;
            int idx = (y * size + x) * 3;
            img[idx + 0] = val;       /* R */
            img[idx + 1] = val / 2;   /* G */
            img[idx + 2] = 255 - val; /* B */
        }
    }
    return img;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <mmproj.gguf> [image] [prompt] [max_gen]\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *mmproj_path = argv[2];
    const char *image_path = NULL;
    const char *user_prompt = "explain the image";
    int max_gen = 100;

    /* Parse optional args: image path (has . extension), prompt (text), max_gen (number) */
    for (int i = 3; i < argc; i++) {
        if (strchr(argv[i], '.') && (strstr(argv[i], ".jpg") || strstr(argv[i], ".jpeg") ||
            strstr(argv[i], ".png") || strstr(argv[i], ".bmp") || strstr(argv[i], ".ppm"))) {
            image_path = argv[i];
        } else if (argv[i][0] >= '0' && argv[i][0] <= '9') {
            max_gen = atoi(argv[i]);
        } else {
            user_prompt = argv[i];
        }
    }

    /* 1. Load main LLM model */
    fprintf(stderr, "Loading LLM: %s\n", model_path);
    gguf_context *gguf_main = gguf_open(model_path, 1);
    if (!gguf_main) { fprintf(stderr, "failed to open LLM GGUF\n"); return 1; }

    bpe_vocab *vocab = bpe_vocab_load(gguf_main);
    if (!vocab) { fprintf(stderr, "failed to load vocab\n"); return 1; }

    transformer_model *model = transformer_load(gguf_main, 512);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    /* 2. Load vision encoder */
    fprintf(stderr, "Loading mmproj: %s\n", mmproj_path);
    gguf_context *gguf_mm = gguf_open(mmproj_path, 1);
    if (!gguf_mm) { fprintf(stderr, "failed to open mmproj GGUF\n"); return 1; }

    g4v_model *vision = g4v_load(gguf_mm);
    if (!vision) { fprintf(stderr, "failed to load vision model\n"); return 1; }

    /* 3. Load or generate image */
    int img_w, img_h;
    uint8_t *image;
    if (image_path) {
        image = img_load(image_path, &img_w, &img_h);
        if (!image) { fprintf(stderr, "failed to load image: %s\n", image_path); return 1; }
        fprintf(stderr, "Loaded image: %s (%dx%d)\n", image_path, img_w, img_h);
        /* Resize to vision encoder's expected size (224x224) */
        int target = vision->image_size;
        if (img_w != target || img_h != target) {
            uint8_t *resized = img_resize_ac(image, img_w, img_h, target, target);
            img_free(image);
            image = resized;
            img_w = img_h = target;
            fprintf(stderr, "Resized to %dx%d\n", target, target);
        }
    } else {
        img_w = img_h = vision->image_size;
        image = generate_test_image(img_w);
        fprintf(stderr, "Using synthetic checkerboard %dx%d\n", img_w, img_h);
    }

    fprintf(stderr, "Prompt: \"%s\"\n", user_prompt);

    /* 4. Encode image through vision encoder */
    fprintf(stderr, "\n=== Vision Encoding ===\n");
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    float *vision_embd = g4v_encode(vision, image, img_w, img_h);
    free(image);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double vis_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    fprintf(stderr, "Vision encoding: %.2f s, %d tokens x %d dim\n",
            vis_time, vision->n_merged, vision->proj_dim);

    if (!vision_embd) { fprintf(stderr, "vision encoding failed\n"); return 1; }

    /* Print some embedding stats */
    {
        float min_v = vision_embd[0], max_v = vision_embd[0], sum = 0;
        int total = vision->n_merged * vision->proj_dim;
        for (int i = 0; i < total; i++) {
            if (vision_embd[i] < min_v) min_v = vision_embd[i];
            if (vision_embd[i] > max_v) max_v = vision_embd[i];
            sum += vision_embd[i];
        }
        fprintf(stderr, "Vision embedding: min=%.4f max=%.4f mean=%.6f\n",
                min_v, max_v, sum / total);
    }

    /* 5. Build multimodal prompt */
    /* Gemma4 chat format:
     *   <|turn>user\n<|image>...<image|>{prompt}<turn|>\n<|turn>model\n
     * <|image> = token 255999, <image|> = token 258882
     */
    char pre_image_buf[256];
    snprintf(pre_image_buf, sizeof(pre_image_buf), "<|turn>user\n<|image>");
    const char *pre_image = pre_image_buf;

    char post_image_buf[512];
    snprintf(post_image_buf, sizeof(post_image_buf), "<image|>%s<turn|>\n<|turn>model\n", user_prompt);
    const char *post_image = post_image_buf;

    /* Tokenize pre-image text */
    int32_t pre_tokens[64];
    int n_pre = bpe_tokenize(vocab, pre_image, -1, pre_tokens, 64);
    fprintf(stderr, "Pre-image tokens (%d):", n_pre);
    for (int i = 0; i < n_pre; i++)
        fprintf(stderr, " %d(\"%s\")", pre_tokens[i], bpe_token_to_str(vocab, pre_tokens[i]));
    fprintf(stderr, "\n");

    /* Tokenize post-image text */
    int32_t post_tokens[64];
    int n_post = bpe_tokenize(vocab, post_image, -1, post_tokens, 64);
    fprintf(stderr, "Post-image tokens (%d):", n_post);
    for (int i = 0; i < n_post; i++)
        fprintf(stderr, " %d(\"%s\")", post_tokens[i], bpe_token_to_str(vocab, post_tokens[i]));
    fprintf(stderr, "\n");

    /* 6. Prefill: text tokens + vision embeddings + text tokens */
    fprintf(stderr, "\n=== LLM Prefill ===\n");
    int pos = 0;
    int n_vision = vision->n_merged;
    int proj_dim = vision->proj_dim;

    /* Prefill pre-image text tokens */
    for (int i = 0; i < n_pre; i++) {
        transformer_forward(model, pre_tokens[i], pos++);
    }
    fprintf(stderr, "Prefilled %d pre-image tokens\n", n_pre);

    /* Inject vision embeddings (using transformer_forward_embd) */
    for (int i = 0; i < n_vision; i++) {
        float *embd_i = vision_embd + i * proj_dim;
        transformer_forward_embd(model, embd_i, pos++);
    }
    fprintf(stderr, "Injected %d vision tokens\n", n_vision);

    /* Prefill post-image text tokens */
    for (int i = 0; i < n_post; i++) {
        if (i == n_post - 1) {
            /* Last token: get logits */
            float *logits = transformer_forward_logits(model, post_tokens[i], pos++);
            if (logits) {
                /* Find argmax */
                int32_t argmax = 0;
                for (int j = 1; j < model->n_vocab; j++)
                    if (logits[j] > logits[argmax]) argmax = j;
                fprintf(stderr, "First generated token: %d \"%s\" (logit=%.4f)\n",
                        argmax, bpe_token_to_str(vocab, argmax), logits[argmax]);

                /* 7. Generate tokens with top-k sampling (temperature=0.7, top_k=40) */
                fprintf(stderr, "\n=== Generation ===\n");
                int32_t next_token = argmax;
                int eos_id = bpe_eos_id(vocab);
                int eot_id = bpe_eot_id(vocab);
                float temperature = 0.7f;
                int top_k = 40;
                srand((unsigned)time(NULL));

                for (int g = 0; g < max_gen; g++) {
                    const char *tok_str = bpe_token_to_str(vocab, next_token);
                    if (tok_str) {
                        int dec_len;
                        char *decoded = bpe_byte_decode(tok_str, (int)strlen(tok_str), &dec_len);
                        fwrite(decoded, 1, dec_len, stdout);
                        fflush(stdout);
                        free(decoded);
                    }

                    if (next_token == eos_id || next_token == eot_id) break;
                    /* Also stop on <turn|> (end of turn) */
                    if (next_token == 106) break;

                    logits = transformer_forward_logits(model, next_token, pos++);
                    if (!logits) break;

                    /* Top-k sampling with temperature */
                    int n_vocab = model->n_vocab;
                    /* Apply temperature */
                    float inv_temp = 1.0f / temperature;
                    for (int j = 0; j < n_vocab; j++) logits[j] *= inv_temp;

                    /* Find top-k */
                    int *topk_idx = (int *)malloc(top_k * sizeof(int));
                    float *topk_val = (float *)malloc(top_k * sizeof(float));
                    for (int k = 0; k < top_k; k++) { topk_idx[k] = 0; topk_val[k] = -1e30f; }
                    for (int j = 0; j < n_vocab; j++) {
                        if (logits[j] > topk_val[top_k-1]) {
                            topk_idx[top_k-1] = j; topk_val[top_k-1] = logits[j];
                            for (int k = top_k-2; k >= 0; k--) {
                                if (topk_val[k+1] > topk_val[k]) {
                                    int ti = topk_idx[k]; topk_idx[k] = topk_idx[k+1]; topk_idx[k+1] = ti;
                                    float tv = topk_val[k]; topk_val[k] = topk_val[k+1]; topk_val[k+1] = tv;
                                } else break;
                            }
                        }
                    }

                    /* Softmax over top-k */
                    float max_logit = topk_val[0];
                    float sum_exp = 0;
                    for (int k = 0; k < top_k; k++) {
                        topk_val[k] = expf(topk_val[k] - max_logit);
                        sum_exp += topk_val[k];
                    }

                    /* Sample */
                    float r = (float)rand() / RAND_MAX * sum_exp;
                    float cumsum = 0;
                    next_token = topk_idx[0];
                    for (int k = 0; k < top_k; k++) {
                        cumsum += topk_val[k];
                        if (cumsum >= r) { next_token = topk_idx[k]; break; }
                    }
                    free(topk_idx);
                    free(topk_val);
                }
                printf("\n");
            }
        } else {
            transformer_forward(model, post_tokens[i], pos++);
        }
    }

    fprintf(stderr, "Done.\n");

    free(vision_embd);
    g4v_free(vision);
    transformer_free(model);
    bpe_vocab_free(vocab);
    gguf_close(gguf_mm);
    gguf_close(gguf_main);
    return 0;
}
