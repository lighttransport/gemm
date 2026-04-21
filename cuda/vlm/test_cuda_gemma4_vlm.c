/*
 * test_cuda_gemma4_vlm.c - Gemma4 VLM: CPU vision encoder + CUDA LLM
 *
 * Usage:
 *   ./test_cuda_gemma4_vlm <model.gguf> <mmproj.gguf> [image] [prompt] [max_gen] [--budget N]
 *
 * --budget N: reasoning token budget (default 32). If the model emits hidden
 *             thought-channel tokens, cap them to N tokens before forcing an
 *             exit back to visible output.
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

/* SafeTensors loader needed by cuda_llm_runner.c helper paths */
#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"

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
    char *pre_image;
    char *post_image;
} gemma4_prompt_parts;

typedef struct {
    int32_t eos_id;
    int32_t eot_id;
    int32_t turn_end_id;
    int32_t channel_start_id;
    int32_t channel_end_id;
    int in_thought;
} gemma4_decode_state;

typedef struct {
    char *data;
    size_t len;
    size_t cap;
} visible_text;

typedef struct {
    int32_t *ids;
    int count;
} token_buffer;

typedef struct {
    float temperature;
    int top_k;
    float top_p;
    float min_p;
} gemma4_sampling_params;

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
    gemma4_prompt_parts parts = {0};
    const char *pre =
        "<|turn>system\n"
        "<|think|><turn|>\n"
        "<|turn>user\n<|image>";
    const char *post_fmt = "<image|>%s<turn|>\n<|turn>model\n";
    size_t pre_len = strlen(pre);
    parts.pre_image = (char *)malloc(pre_len + 1);
    if (parts.pre_image) memcpy(parts.pre_image, pre, pre_len + 1);
    int post_len = snprintf(NULL, 0, post_fmt, user_prompt ? user_prompt : "");
    if (post_len >= 0) {
        parts.post_image = (char *)malloc((size_t)post_len + 1);
        if (parts.post_image)
            snprintf(parts.post_image, (size_t)post_len + 1, post_fmt, user_prompt ? user_prompt : "");
    }
    return parts;
}

static void free_gemma4_prompt(gemma4_prompt_parts *parts) {
    if (!parts) return;
    free(parts->pre_image);
    free(parts->post_image);
    parts->pre_image = NULL;
    parts->post_image = NULL;
}

static void visible_text_free(visible_text *out) {
    if (!out) return;
    free(out->data);
    out->data = NULL;
    out->len = 0;
    out->cap = 0;
}

static int visible_text_append(visible_text *out, const char *buf, size_t n) {
    if (!out || !buf || n == 0) return 0;
    size_t need = out->len + n + 1;
    if (need > out->cap) {
        size_t new_cap = out->cap ? out->cap * 2 : 128;
        while (new_cap < need) new_cap *= 2;
        char *p = (char *)realloc(out->data, new_cap);
        if (!p) return -1;
        out->data = p;
        out->cap = new_cap;
    }
    memcpy(out->data + out->len, buf, n);
    out->len += n;
    out->data[out->len] = '\0';
    return 0;
}

static void token_buffer_free(token_buffer *buf) {
    if (!buf) return;
    free(buf->ids);
    buf->ids = NULL;
    buf->count = 0;
}

static token_buffer tokenize_text_alloc(const bpe_vocab *vocab, const char *text) {
    token_buffer buf = {0};
    int count = bpe_tokenize(vocab, text, -1, NULL, 0);
    if (count <= 0) return buf;
    buf.ids = (int32_t *)malloc((size_t)count * sizeof(int32_t));
    if (!buf.ids) return buf;
    buf.count = bpe_tokenize(vocab, text, -1, buf.ids, count);
    if (buf.count != count) {
        token_buffer_free(&buf);
    }
    return buf;
}

static int32_t find_exact_token_id(const bpe_vocab *vocab, const char *text) {
    if (!vocab || !text) return -1;
    return bpe_hm_get(&vocab->token_to_id, text, (int)strlen(text));
}

static int32_t sample_gemma4_token(const float *logits, int n_vocab,
                                   const gemma4_sampling_params *params) {
    if (!logits || !params || n_vocab <= 0) return -1;

    const int top_k = params->top_k > 0 && params->top_k < n_vocab ? params->top_k : n_vocab;
    int *topk_idx = (int *)malloc((size_t)top_k * sizeof(int));
    float *topk_val = (float *)malloc((size_t)top_k * sizeof(float));
    if (!topk_idx || !topk_val) {
        free(topk_idx);
        free(topk_val);
        return 0;
    }

    for (int k = 0; k < top_k; k++) {
        topk_idx[k] = 0;
        topk_val[k] = -1e30f;
    }

    const float inv_temp = params->temperature > 0.0f ? (1.0f / params->temperature) : 1.0f;
    for (int j = 0; j < n_vocab; j++) {
        const float scaled = logits[j] * inv_temp;
        if (scaled > topk_val[top_k - 1]) {
            topk_idx[top_k - 1] = j;
            topk_val[top_k - 1] = scaled;
            for (int k = top_k - 2; k >= 0; k--) {
                if (topk_val[k + 1] > topk_val[k]) {
                    int ti = topk_idx[k];
                    float tv = topk_val[k];
                    topk_idx[k] = topk_idx[k + 1];
                    topk_val[k] = topk_val[k + 1];
                    topk_idx[k + 1] = ti;
                    topk_val[k + 1] = tv;
                } else {
                    break;
                }
            }
        }
    }

    const float max_logit = topk_val[0];
    float sum_exp = 0.0f;
    for (int k = 0; k < top_k; k++) {
        topk_val[k] = expf(topk_val[k] - max_logit);
        sum_exp += topk_val[k];
    }

    const int32_t fallback = topk_idx[0];
    if (sum_exp <= 0.0f) {
        free(topk_idx);
        free(topk_val);
        return fallback;
    }

    for (int k = 0; k < top_k; k++) {
        topk_val[k] /= sum_exp;
    }

    int kept = top_k;
    if (params->top_p > 0.0f && params->top_p < 1.0f) {
        float cum = 0.0f;
        kept = 0;
        for (int k = 0; k < top_k; k++) {
            cum += topk_val[k];
            kept = k + 1;
            if (cum >= params->top_p) break;
        }
    }

    if (params->min_p > 0.0f) {
        const float threshold = topk_val[0] * params->min_p;
        int minp_kept = 0;
        while (minp_kept < kept && topk_val[minp_kept] >= threshold) minp_kept++;
        if (minp_kept > 0) kept = minp_kept;
    }

    float kept_sum = 0.0f;
    for (int k = 0; k < kept; k++) kept_sum += topk_val[k];
    if (kept_sum <= 0.0f) {
        free(topk_idx);
        free(topk_val);
        return fallback;
    }

    float r = (float)rand() / (float)RAND_MAX * kept_sum;
    float cumsum = 0.0f;
    int32_t next_token = fallback;
    for (int k = 0; k < kept; k++) {
        cumsum += topk_val[k];
        if (cumsum >= r) {
            next_token = topk_idx[k];
            break;
        }
    }

    free(topk_idx);
    free(topk_val);
    return next_token;
}

static gemma4_decode_state init_decode_state(const bpe_vocab *vocab) {
    gemma4_decode_state st;
    st.eos_id = bpe_eos_id(vocab);
    st.eot_id = bpe_eot_id(vocab);
    st.turn_end_id = find_exact_token_id(vocab, "<turn|>");
    st.channel_start_id = find_exact_token_id(vocab, "<|channel>");
    st.channel_end_id = find_exact_token_id(vocab, "<channel|>");
    st.in_thought = 0;
    return st;
}

static void emit_visible_token(const bpe_vocab *vocab, gemma4_decode_state *st,
                               int32_t token, visible_text *out) {
    if (token == st->eos_id || token == st->eot_id || token == st->turn_end_id) return;
    if (token == st->channel_start_id) {
        st->in_thought = 1;
        return;
    }
    if (token == st->channel_end_id) {
        st->in_thought = 0;
        return;
    }
    if (st->in_thought) return;

    const char *tok_str = bpe_token_to_str(vocab, token);
    if (!tok_str) return;
    int dec_len = 0;
    char *decoded = bpe_byte_decode(tok_str, (int)strlen(tok_str), &dec_len);
    if (decoded && dec_len > 0) (void)visible_text_append(out, decoded, (size_t)dec_len);
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
    int reasoning_budget_tokens = 32;

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

    /* Done with CPU vision model (not needed for GPU path) */
    int n_vision = vision->n_merged;
    (void)vision->proj_dim; /* GPU path computes proj_dim from mmproj GGUF */
    g4v_free(vision);

    /* 3. Load LLM on CUDA first (need CUDA context for GPU vision) */
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

    /* 4. Encode image on GPU */
    fprintf(stderr, "\n=== Vision Encoding (GPU) ===\n");
    int proj_dim = 0;
    double t0 = get_time_ms();
    float *vision_embd = cuda_llm_vision_encode(llm, gguf_mm, image, img_w, img_h, &n_vision, &proj_dim);
    double t1 = get_time_ms();
    free(image);
    gguf_close(gguf_mm);

    if (!vision_embd) { fprintf(stderr, "GPU vision encoding failed\n"); return 1; }
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

    if (proj_dim != n_embd) {
        fprintf(stderr, "ERROR: vision proj_dim=%d != LLM n_embd=%d\n", proj_dim, n_embd);
        return 1;
    }

    /* 5. Build prompt */
    gemma4_prompt_parts prompt_parts = build_gemma4_prompt(user_prompt);
    if (!prompt_parts.pre_image || !prompt_parts.post_image) {
        fprintf(stderr, "ERROR: prompt allocation failed\n");
        free_gemma4_prompt(&prompt_parts);
        return 1;
    }

    int32_t bos_id = vocab->bos_id;
    if (bos_id >= 0)
        fprintf(stderr, "BOS: %d(\"%s\")\n", bos_id, bpe_token_to_str(vocab, bos_id));

    token_buffer pre_tokens = tokenize_text_alloc(vocab, prompt_parts.pre_image);
    fprintf(stderr, "Pre-image tokens (%d):", pre_tokens.count);
    for (int i = 0; i < pre_tokens.count; i++)
        fprintf(stderr, " %d(\"%s\")", pre_tokens.ids[i], bpe_token_to_str(vocab, pre_tokens.ids[i]));
    fprintf(stderr, "\n");

    token_buffer post_tokens = tokenize_text_alloc(vocab, prompt_parts.post_image);
    fprintf(stderr, "Post-image tokens (%d):", post_tokens.count);
    for (int i = 0; i < post_tokens.count; i++)
        fprintf(stderr, " %d(\"%s\")", post_tokens.ids[i], bpe_token_to_str(vocab, post_tokens.ids[i]));
    fprintf(stderr, "\n");
    free_gemma4_prompt(&prompt_parts);

    /* 6. Batched Prefill */
    fprintf(stderr, "\n=== LLM Prefill (CUDA, batched) ===\n");
    int pos = 0;

    /* BOS token */
    if (bos_id >= 0) {
        cuda_llm_forward(llm, bos_id, pos++);
        fprintf(stderr, "Prefilled BOS\n");
    }

    /* Batch: pre-image tokens */
    t0 = get_time_ms();
    cuda_llm_prefill(llm, pre_tokens.ids, NULL, 0, pre_tokens.count, pos);
    pos += pre_tokens.count;
    t1 = get_time_ms();
    fprintf(stderr, "Pre-image: %d tokens (%.1f ms, %.2f ms/tok)\n",
            pre_tokens.count, t1 - t0, pre_tokens.count > 0 ? (t1 - t0) / pre_tokens.count : 0);

    /* Batch: vision embeddings */
    t0 = get_time_ms();
    cuda_llm_prefill(llm, NULL, vision_embd, proj_dim, n_vision, pos);
    pos += n_vision;
    t1 = get_time_ms();
    fprintf(stderr, "Vision: %d tokens (%.1f ms, %.2f ms/tok)\n",
            n_vision, t1 - t0, (t1 - t0) / n_vision);

    /* Batch: post-image tokens (last one returns logits) */
    float *logits = NULL;
    t0 = get_time_ms();
    logits = cuda_llm_prefill_logits(llm, post_tokens.ids, NULL, 0, post_tokens.count, pos);
    pos += post_tokens.count;
    t1 = get_time_ms();
    fprintf(stderr, "Post-image: %d tokens (%.1f ms, %.2f ms/tok)\n",
            post_tokens.count, t1 - t0, post_tokens.count > 0 ? (t1 - t0) / post_tokens.count : 0);

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
    visible_text out = {0};
    gemma4_sampling_params sampling = {
        .temperature = 0.8f,
        .top_k = 40,
        .top_p = 0.95f,
        .min_p = 0.05f,
    };
    srand((unsigned)time(NULL));

    t0 = get_time_ms();
    int gen_count = 0;

    for (int g = 0; g < max_gen; g++) {
        emit_visible_token(vocab, &ds, next_token, &out);
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

        next_token = sample_gemma4_token(logits, n_vocab, &sampling);
    }
    if (out.len > 0) fwrite(out.data, 1, out.len, stdout);
    printf("\n");
    fflush(stdout);
    visible_text_free(&out);
    t1 = get_time_ms();
    fprintf(stderr, "[thought: %d/%d tokens used]\n", rb.count, rb.budget);
    fprintf(stderr, "[generation: %d tokens, %.1f ms, %.2f ms/tok]\n",
            gen_count, t1 - t0, gen_count > 0 ? (t1 - t0) / gen_count : 0);

    /* Cleanup */
    token_buffer_free(&pre_tokens);
    token_buffer_free(&post_tokens);
    free(vision_embd);
    cuda_llm_free(llm);
    bpe_vocab_free(vocab);
    gguf_close(gguf_main);

    fprintf(stderr, "Done.\n");
    return 0;
}
