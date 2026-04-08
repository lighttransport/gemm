/*
 * test_cuda_gemma4_vlm.c - Gemma4 VLM: CPU vision encoder + CUDA LLM
 *
 * Usage:
 *   ./test_cuda_gemma4_vlm <model.gguf> <mmproj.gguf> [image] [prompt] [max_gen] [--budget N]
 *
 * --budget N: reasoning token budget (default 200). The model can think for up
 *             to N tokens inside <|channel>thought...<channel|> before being
 *             forced to exit and produce visible output.
 *
 * Vision encoding runs on CPU (gemma4_vision_encoder.h).
 * LLM inference runs on CUDA (cuda_llm_runner).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* GGUF loader */
#define GGUF_LOADER_IMPLEMENTATION
#include "../../common/gguf_loader.h"

/* Dequantization */
#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"

/* BPE tokenizer */
#define BPE_TOKENIZER_IMPLEMENTATION
#include "../../common/bpe_tokenizer.h"

/* transformer.h for qtensor type (needed by gemma4_vision_encoder.h) */
#include "../../common/transformer.h"

/* Gemma4 vision encoder (CPU) */
#define GEMMA4_VISION_IMPLEMENTATION
#include "../../common/gemma4_vision_encoder.h"

/* Image loading/resizing */
#define IMAGE_UTILS_IMPLEMENTATION
#include "../../common/image_utils.h"

/* CUDA LLM runner */
#include "../llm/cuda_llm_runner.h"

/* ---- Prompt helpers ---- */

typedef struct {
    char pre_image[256];
    char post_image[512];
} gemma4_prompt_parts;

typedef struct {
    int32_t eos_id;
    int32_t eot_id;
    int32_t turn_end_id;
    int32_t channel_start_id;
    int32_t channel_end_id;
    int in_thought;
} gemma4_decode_state;

/* ---- Reasoning budget state machine ---- */

typedef enum {
    RB_IDLE,
    RB_COUNTING,
    RB_FORCING,
    RB_DONE
} rb_state;

typedef struct {
    rb_state state;
    int      budget;
    int      count;
    int32_t  start_tokens[8];
    int      n_start;
    int      start_matched;
    int32_t  end_tokens[8];
    int      n_end;
    int      end_matched;
    int      force_idx;
} reasoning_budget;

static reasoning_budget rb_init(const bpe_vocab *vocab, int budget) {
    reasoning_budget rb;
    memset(&rb, 0, sizeof(rb));
    rb.budget = budget;
    rb.state = RB_IDLE;
    rb.n_start = bpe_tokenize(vocab, "<|channel>thought\n", -1, rb.start_tokens, 8);
    if (rb.n_start < 0) rb.n_start = 0;
    rb.n_end = bpe_tokenize(vocab, "<channel|>", -1, rb.end_tokens, 8);
    if (rb.n_end < 0) rb.n_end = 0;
    return rb;
}

static void rb_observe(reasoning_budget *rb, int32_t token) {
    switch (rb->state) {
    case RB_IDLE:
        if (rb->n_start > 0 && token == rb->start_tokens[rb->start_matched]) {
            rb->start_matched++;
            if (rb->start_matched >= rb->n_start) {
                rb->state = RB_COUNTING;
                rb->count = 0;
                rb->start_matched = 0;
            }
        } else {
            rb->start_matched = 0;
            if (rb->n_start > 0 && token == rb->start_tokens[0])
                rb->start_matched = 1;
        }
        break;
    case RB_COUNTING:
        rb->count++;
        if (rb->n_end > 0 && token == rb->end_tokens[rb->end_matched]) {
            rb->end_matched++;
            if (rb->end_matched >= rb->n_end) {
                rb->state = RB_DONE;
                rb->end_matched = 0;
            }
        } else {
            rb->end_matched = 0;
            if (rb->n_end > 0 && token == rb->end_tokens[0])
                rb->end_matched = 1;
        }
        if (rb->state == RB_COUNTING && rb->count >= rb->budget) {
            rb->state = RB_FORCING;
            rb->force_idx = 0;
        }
        break;
    case RB_FORCING:
    case RB_DONE:
        break;
    }
}

static int rb_needs_forcing(const reasoning_budget *rb) {
    return rb->state == RB_FORCING;
}

static int32_t rb_force_next(reasoning_budget *rb) {
    if (rb->state != RB_FORCING) return -1;
    if (rb->force_idx >= rb->n_end) { rb->state = RB_DONE; return -1; }
    int32_t tok = rb->end_tokens[rb->force_idx++];
    if (rb->force_idx >= rb->n_end) rb->state = RB_DONE;
    return tok;
}

/* ---- Prompt / decode helpers ---- */

static gemma4_prompt_parts build_gemma4_prompt(const char *user_prompt) {
    gemma4_prompt_parts parts;
    snprintf(parts.pre_image, sizeof(parts.pre_image),
             "<|turn>system\n<|think|><turn|>\n<|turn>user\n<|image>");
    snprintf(parts.post_image, sizeof(parts.post_image),
             "<image|>%s<turn|>\n<|turn>model\n", user_prompt);
    return parts;
}

static gemma4_decode_state init_decode_state(const bpe_vocab *vocab) {
    gemma4_decode_state st;
    st.eos_id = bpe_eos_id(vocab);
    st.eot_id = bpe_eot_id(vocab);
    st.turn_end_id = -1;
    st.channel_start_id = -1;
    st.channel_end_id = -1;
    st.in_thought = 0;

    int32_t tmp[4];
    int n = bpe_tokenize(vocab, "<|channel>", -1, tmp, 4);
    if (n == 1) st.channel_start_id = tmp[0];
    n = bpe_tokenize(vocab, "<channel|>", -1, tmp, 4);
    if (n == 1) st.channel_end_id = tmp[0];
    n = bpe_tokenize(vocab, "<turn|>", -1, tmp, 4);
    if (n == 1) st.turn_end_id = tmp[0];
    return st;
}

static void emit_visible_token(const bpe_vocab *vocab, gemma4_decode_state *st, int32_t token) {
    if (token == st->channel_start_id) { st->in_thought = 1; }
    if (token == st->channel_end_id)   { st->in_thought = 0; }

    const char *tok_str = bpe_token_to_str(vocab, token);
    if (!tok_str) return;
    int dec_len = 0;
    char *decoded = bpe_byte_decode(tok_str, (int)strlen(tok_str), &dec_len);
    fwrite(decoded, 1, dec_len, stdout);
    fflush(stdout);
    free(decoded);
}

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ---- Main ---- */

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <mmproj.gguf> [image] [prompt] [max_gen] [--budget N]\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *mmproj_path = argv[2];
    const char *image_path = NULL;
    const char *user_prompt = "explain the image";
    int max_gen = 100;
    int reasoning_budget_tokens = 200;

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--budget") == 0 && i + 1 < argc) {
            reasoning_budget_tokens = atoi(argv[++i]);
        } else if (strchr(argv[i], '.') && (strstr(argv[i], ".jpg") || strstr(argv[i], ".jpeg") ||
            strstr(argv[i], ".png") || strstr(argv[i], ".bmp") || strstr(argv[i], ".ppm"))) {
            image_path = argv[i];
        } else if (argv[i][0] >= '0' && argv[i][0] <= '9') {
            max_gen = atoi(argv[i]);
        } else {
            user_prompt = argv[i];
        }
    }

    fprintf(stderr, "=== CUDA Gemma4 VLM ===\n");
    fprintf(stderr, "LLM:    %s\n", model_path);
    fprintf(stderr, "mmproj: %s\n", mmproj_path);
    if (image_path) fprintf(stderr, "Image:  %s\n", image_path);
    fprintf(stderr, "Prompt: \"%s\" (budget=%d, max_gen=%d)\n\n",
            user_prompt, reasoning_budget_tokens, max_gen);

    /* 1. Load vision encoder (CPU) */
    fprintf(stderr, "Loading mmproj: %s\n", mmproj_path);
    gguf_context *gguf_mm = gguf_open(mmproj_path, 1);
    if (!gguf_mm) { fprintf(stderr, "failed to open mmproj GGUF\n"); return 1; }

    g4v_model *vision = g4v_load(gguf_mm);
    if (!vision) { fprintf(stderr, "failed to load vision model\n"); return 1; }

    /* 2. Load or generate image */
    int img_w, img_h;
    uint8_t *image;
    if (image_path) {
        image = img_load(image_path, &img_w, &img_h);
        if (!image) { fprintf(stderr, "failed to load image: %s\n", image_path); return 1; }
        fprintf(stderr, "Loaded image: %s (%dx%d)\n", image_path, img_w, img_h);
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
        image = (uint8_t *)malloc(img_w * img_h * 3);
        int block = img_w / 8;
        for (int y = 0; y < img_h; y++)
            for (int x = 0; x < img_w; x++) {
                int idx = (y * img_w + x) * 3;
                uint8_t val = (((y / block) + (x / block)) % 2) ? 220 : 35;
                image[idx] = val; image[idx+1] = val/2; image[idx+2] = 255-val;
            }
        fprintf(stderr, "Using synthetic checkerboard %dx%d\n", img_w, img_h);
    }

    /* 3. Encode image (CPU) */
    fprintf(stderr, "\n=== Vision Encoding (CPU) ===\n");
    double t0 = get_time_ms();
    float *vision_embd = g4v_encode(vision, image, img_w, img_h);
    double t1 = get_time_ms();
    free(image);
    if (!vision_embd) { fprintf(stderr, "vision encoding failed\n"); return 1; }

    int n_vision = vision->n_merged;
    int proj_dim = vision->proj_dim;
    fprintf(stderr, "Vision: %d tokens x %d dim (%.1f ms)\n", n_vision, proj_dim, t1 - t0);

    {
        float vmin = vision_embd[0], vmax = vision_embd[0], sum = 0;
        int total = n_vision * proj_dim;
        for (int i = 0; i < total; i++) {
            if (vision_embd[i] < vmin) vmin = vision_embd[i];
            if (vision_embd[i] > vmax) vmax = vision_embd[i];
            sum += vision_embd[i];
        }
        fprintf(stderr, "Vision embedding: min=%.4f max=%.4f mean=%.6f\n",
                vmin, vmax, sum / total);
    }

    /* Done with vision model */
    g4v_free(vision);
    gguf_close(gguf_mm);

    /* 4. Load LLM on CUDA */
    fprintf(stderr, "\nLoading LLM: %s\n", model_path);
    gguf_context *gguf_main = gguf_open(model_path, 1);
    if (!gguf_main) { fprintf(stderr, "failed to open LLM GGUF\n"); return 1; }

    bpe_vocab *vocab = bpe_vocab_load(gguf_main);
    if (!vocab) { fprintf(stderr, "failed to load vocab\n"); return 1; }

    int max_seq_len = n_vision + 256 + max_gen;
    if (max_seq_len < 1024) max_seq_len = 1024;

    fprintf(stderr, "Initializing CUDA LLM (max_seq_len=%d)...\n", max_seq_len);
    cuda_llm_runner *llm = cuda_llm_init(0, 1);
    if (!llm) { fprintf(stderr, "CUDA LLM init failed\n"); return 1; }
    if (cuda_llm_load_weights(llm, gguf_main, max_seq_len) != 0) {
        fprintf(stderr, "CUDA LLM weight load failed\n"); return 1;
    }

    /* cuda_llm_set_debug(llm, 3); */
    int n_embd = cuda_llm_n_embd(llm);
    int n_vocab = cuda_llm_n_vocab(llm);
    fprintf(stderr, "LLM: n_embd=%d n_vocab=%d n_layers=%d\n",
            n_embd, n_vocab, cuda_llm_n_layers(llm));

    if (proj_dim != n_embd) {
        fprintf(stderr, "ERROR: vision proj_dim=%d != LLM n_embd=%d\n", proj_dim, n_embd);
        return 1;
    }

    /* 5. Build prompt */
    gemma4_prompt_parts prompt_parts = build_gemma4_prompt(user_prompt);

    int32_t bos_id = vocab->bos_id;
    if (bos_id >= 0)
        fprintf(stderr, "BOS: %d(\"%s\")\n", bos_id, bpe_token_to_str(vocab, bos_id));

    int32_t pre_tokens[64];
    int n_pre = bpe_tokenize(vocab, prompt_parts.pre_image, -1, pre_tokens, 64);
    fprintf(stderr, "Pre-image tokens (%d):", n_pre);
    for (int i = 0; i < n_pre; i++)
        fprintf(stderr, " %d(\"%s\")", pre_tokens[i], bpe_token_to_str(vocab, pre_tokens[i]));
    fprintf(stderr, "\n");

    int32_t post_tokens[64];
    int n_post = bpe_tokenize(vocab, prompt_parts.post_image, -1, post_tokens, 64);
    fprintf(stderr, "Post-image tokens (%d):", n_post);
    for (int i = 0; i < n_post; i++)
        fprintf(stderr, " %d(\"%s\")", post_tokens[i], bpe_token_to_str(vocab, post_tokens[i]));
    fprintf(stderr, "\n");

    /* 6. Prefill */
    fprintf(stderr, "\n=== LLM Prefill (CUDA) ===\n");
    int pos = 0;

    if (bos_id >= 0) {
        cuda_llm_forward(llm, bos_id, pos++);
        fprintf(stderr, "Prefilled BOS\n");
    }

    t0 = get_time_ms();
    for (int i = 0; i < n_pre; i++)
        cuda_llm_forward(llm, pre_tokens[i], pos++);
    t1 = get_time_ms();
    fprintf(stderr, "Pre-image: %d tokens (%.1f ms)\n", n_pre, t1 - t0);

    t0 = get_time_ms();
    for (int i = 0; i < n_vision; i++) {
        float *embd_i = vision_embd + i * proj_dim;
        cuda_llm_forward_embd(llm, embd_i, proj_dim, pos++);
    }
    t1 = get_time_ms();
    fprintf(stderr, "Vision: %d tokens (%.1f ms, %.2f ms/tok)\n",
            n_vision, t1 - t0, (t1 - t0) / n_vision);

    float *logits = NULL;
    t0 = get_time_ms();
    for (int i = 0; i < n_post; i++) {
        if (i == n_post - 1)
            logits = cuda_llm_forward_logits(llm, post_tokens[i], pos++);
        else
            cuda_llm_forward(llm, post_tokens[i], pos++);
    }
    t1 = get_time_ms();
    fprintf(stderr, "Post-image: %d tokens (%.1f ms)\n", n_post, t1 - t0);

    if (!logits) { fprintf(stderr, "ERROR: forward_logits returned NULL\n"); return 1; }

    /* Find argmax for first generated token */
    int32_t argmax = 0;
    for (int j = 1; j < n_vocab; j++)
        if (logits[j] > logits[argmax]) argmax = j;
    fprintf(stderr, "First token: %d \"%s\" (logit=%.4f)\n",
            argmax, bpe_token_to_str(vocab, argmax), logits[argmax]);

    /* 7. Generate with reasoning budget */
    fprintf(stderr, "\n=== Generation (budget=%d) ===\n", reasoning_budget_tokens);
    int32_t next_token = argmax;
    gemma4_decode_state ds = init_decode_state(vocab);
    reasoning_budget rb = rb_init(vocab, reasoning_budget_tokens);
    float temperature = 1.0f;
    int top_k = 64;
    srand((unsigned)time(NULL));

    t0 = get_time_ms();
    int gen_count = 0;

    for (int g = 0; g < max_gen; g++) {
        emit_visible_token(vocab, &ds, next_token);
        rb_observe(&rb, next_token);

        if (next_token == ds.eos_id || next_token == ds.eot_id ||
            next_token == ds.turn_end_id) break;

        logits = cuda_llm_forward_logits(llm, next_token, pos++);
        if (!logits) break;
        gen_count++;

        if (rb_needs_forcing(&rb)) {
            int32_t forced = rb_force_next(&rb);
            if (forced >= 0) { next_token = forced; continue; }
        }

        /* Top-k sampling with temperature */
        float inv_temp = 1.0f / temperature;
        for (int j = 0; j < n_vocab; j++) logits[j] *= inv_temp;

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

        float max_logit = topk_val[0], sum_exp = 0;
        for (int k = 0; k < top_k; k++) {
            topk_val[k] = expf(topk_val[k] - max_logit);
            sum_exp += topk_val[k];
        }

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
    t1 = get_time_ms();
    fprintf(stderr, "[thought: %d/%d tokens used]\n", rb.count, rb.budget);
    fprintf(stderr, "[generation: %d tokens, %.1f ms, %.2f ms/tok]\n",
            gen_count, t1 - t0, gen_count > 0 ? (t1 - t0) / gen_count : 0);

    /* Cleanup */
    free(vision_embd);
    cuda_llm_free(llm);
    bpe_vocab_free(vocab);
    gguf_close(gguf_main);

    fprintf(stderr, "Done.\n");
    return 0;
}
