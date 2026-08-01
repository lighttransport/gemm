#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/socket.h>
#include <stdarg.h>
#include <pthread.h>

#include "../common/gguf_loader.h"
#include "../common/ggml_dequant.h"
#include "../common/bpe_tokenizer.h"
#include "../common/transformer.h"
#include "../common/safetensors.h"

#define VISION_ENCODER_IMPLEMENTATION
#include "../common/vision_encoder.h"

#include "image_decode.h"

extern uint8_t *base64_decode_buf(const char *s, size_t in_len, size_t *out_len);

#include "server_llm.h"

/* ---- String builder (local copy of server.c's sbuf) ---- */
typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} sbuf;

static void sbuf_init(sbuf *b) {
    b->cap = 4096;
    b->len = 0;
    b->ptr = (char *)malloc(b->cap);
    if (b->ptr) b->ptr[0] = 0;
}

static void sbuf_free(sbuf *b) {
    free(b->ptr);
    memset(b, 0, sizeof(*b));
}

static int sbuf_reserve(sbuf *b, size_t need) {
    if (need <= b->cap) return 0;
    size_t nc = b->cap ? b->cap : 4096;
    while (nc < need) nc *= 2;
    char *p = (char *)realloc(b->ptr, nc);
    if (!p) return -1;
    b->ptr = p;
    b->cap = nc;
    return 0;
}

static int sbuf_appendn(sbuf *b, const char *s, size_t n) {
    if (sbuf_reserve(b, b->len + n + 1) != 0) return -1;
    memcpy(b->ptr + b->len, s, n);
    b->len += n;
    b->ptr[b->len] = 0;
    return 0;
}

static int sbuf_append(sbuf *b, const char *s) {
    return sbuf_appendn(b, s, strlen(s));
}

static int sbuf_printf(sbuf *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0) { va_end(ap2); return -1; }
    if (sbuf_reserve(b, b->len + (size_t)n + 1) != 0) { va_end(ap2); return -1; }
    vsnprintf(b->ptr + b->len, (size_t)n + 1, fmt, ap2);
    va_end(ap2);
    b->len += (size_t)n;
    return 0;
}

/* ---- JSON helpers ---- */
static const char *j_str(const json_val *obj, const char *key, const char *def) {
    json_val *v = json_obj_get(obj, key);
    if (!v || v->type != JSON_STRING) return def;
    return v->str.ptr ? v->str.ptr : def;
}

static int j_int(const json_val *obj, const char *key, int def) {
    json_val *v = json_obj_get(obj, key);
    if (!v || v->type != JSON_NUMBER) return def;
    return (int)v->num;
}

static double j_f64(const json_val *obj, const char *key, double def) {
    json_val *v = json_obj_get(obj, key);
    if (!v || v->type != JSON_NUMBER) return def;
    return v->num;
}

static char *json_escape_dup(const char *s) {
    if (!s) return strdup("");
    sbuf b;
    sbuf_init(&b);
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        switch (*p) {
        case '\\': sbuf_append(&b, "\\\\"); break;
        case '"': sbuf_append(&b, "\\\""); break;
        case '\n': sbuf_append(&b, "\\n"); break;
        case '\r': sbuf_append(&b, "\\r"); break;
        case '\t': sbuf_append(&b, "\\t"); break;
        default:
            if (*p < 32) sbuf_printf(&b, "\\u%04x", (unsigned)*p);
            else sbuf_appendn(&b, (const char *)p, 1);
            break;
        }
    }
    return b.ptr;
}

/* ---- Bilinear resize for uint8 RGB (align_corners=True) ---- */
static uint8_t *img_resize_bilinear(const uint8_t *src, int sw, int sh, int dw, int dh) {
    uint8_t *dst = (uint8_t *)malloc((size_t)dw * dh * 3);
    if (!dst) return NULL;
    float sx = (sw > 1) ? (float)(sw - 1) / (float)(dw - 1) : 0.0f;
    float sy = (sh > 1) ? (float)(sh - 1) / (float)(dh - 1) : 0.0f;
    for (int dy = 0; dy < dh; dy++) {
        float fy = (sh > 1) ? (float)dy * sy : 0.0f;
        int iy = (int)fy;
        if (iy >= sh - 1) iy = sh - 2;
        if (iy < 0) iy = 0;
        float ty = fy - (float)iy;
        for (int dx = 0; dx < dw; dx++) {
            float fx = (sw > 1) ? (float)dx * sx : 0.0f;
            int ix = (int)fx;
            if (ix >= sw - 1) ix = sw - 2;
            if (ix < 0) ix = 0;
            float tx = fx - (float)ix;
            for (int c = 0; c < 3; c++) {
                float v = (1.0f - ty) * ((1.0f - tx) * src[(iy * sw + ix) * 3 + c] + tx * src[(iy * sw + ix + 1) * 3 + c])
                        + ty       * ((1.0f - tx) * src[((iy + 1) * sw + ix) * 3 + c] + tx * src[((iy + 1) * sw + ix + 1) * 3 + c]);
                if (v < 0) v = 0;
                if (v > 255) v = 255;
                dst[(dy * dw + dx) * 3 + c] = (uint8_t)(v + 0.5f);
            }
        }
    }
    return dst;
}

/* ---- Parse data URI: "data:image/{png,jpg};base64,..." ---- */
/* Returns 0 on success, -1 on error. The decoded data is written to *out (caller must free). */
static int parse_data_uri(const char *uri, size_t uri_len,
                           uint8_t **out_data, size_t *out_len,
                           char *mime, size_t mime_cap) {
    if (!uri || uri_len < 20) return -1;
    /* Check for data: URI prefix */
    if (strncmp(uri, "data:", 5) != 0) return -1;
    const char *semi = memchr(uri + 5, ';', uri_len - 5);
    if (!semi) return -1;
    /* Extract MIME type */
    size_t mime_len = (size_t)(semi - (uri + 5));
    if (mime_len >= mime_cap) mime_len = mime_cap - 1;
    memcpy(mime, uri + 5, mime_len);
    mime[mime_len] = 0;
    /* Find base64 data after ";base64," */
    const char *comma = memchr(semi, ',', uri_len - (size_t)(semi - uri));
    if (!comma) return -1;
    const char *b64 = comma + 1;
    size_t b64_len = uri_len - (size_t)(b64 - uri);
    if (b64_len == 0) return -1;
    /* Check it's base64 (after ";base64,") */
    if ((size_t)(comma - semi) < 8) return -1;
    if (strncasecmp(semi + 1, "base64", 6) != 0) return -1;

    *out_data = base64_decode_buf(b64, b64_len, out_len);
    if (!*out_data) return -1;
    return 0;
}

/* ---- Check if generated text ends with any stop sequence ---- */
static int matches_stop_sequence(const char *text, const json_val *stop_arr) {
    if (!text || !*text || !stop_arr || stop_arr->type != JSON_ARRAY) return 0;
    size_t text_len = strlen(text);
    for (int i = 0; i < stop_arr->arr.count; i++) {
        json_val *item = &stop_arr->arr.items[i];
        if (item->type != JSON_STRING || !item->str.ptr) continue;
        size_t slen = strlen(item->str.ptr);
        if (slen > 0 && text_len >= slen &&
            strcmp(text + text_len - slen, item->str.ptr) == 0) {
            return 1;
        }
    }
    return 0;
}

/* ---- Sampling ---- */
static uint64_t llm_rng_state = 42;

static float randn_local(void) {
    static int cached_valid = 0;
    static float cached = 0.0f;
    if (cached_valid) { cached_valid = 0; return cached; }
    llm_rng_state = llm_rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u1 = (double)(llm_rng_state >> 11) / (double)(1ULL << 53);
    llm_rng_state = llm_rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u2 = (double)(llm_rng_state >> 11) / (double)(1ULL << 53);
    if (u1 < 1e-10) u1 = 1e-10;
    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * 3.14159265358979323846 * u2;
    cached = (float)(r * sin(theta));
    cached_valid = 1;
    return (float)(r * cos(theta));
}

static int sample_token(const float *logits, int n_vocab, float temperature, float top_p) {
    if (!logits || n_vocab <= 0) return -1;

    if (temperature < 0.001f) {
        int argmax = 0;
        for (int j = 1; j < n_vocab; j++) {
            if (logits[j] > logits[argmax]) argmax = j;
        }
        return argmax;
    }

    float *probs = (float *)malloc((size_t)n_vocab * sizeof(float));
    if (!probs) return 0;

    float max_logit = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < n_vocab; i++) {
        probs[i] = expf((logits[i] - max_logit) / temperature);
        sum += probs[i];
    }
    if (sum < 1e-10f) { free(probs); return 0; }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n_vocab; i++) probs[i] *= inv_sum;

    /* Top-p (nucleus) filtering */
    if (top_p < 1.0f && top_p > 0.0f) {
        int *indices = (int *)malloc((size_t)n_vocab * sizeof(int));
        if (indices) {
            for (int i = 0; i < n_vocab; i++) indices[i] = i;
            for (int i = 0; i < n_vocab - 1; i++) {
                for (int j = i + 1; j < n_vocab; j++) {
                    if (probs[indices[j]] > probs[indices[i]]) {
                        int tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
                    }
                }
            }
            float cum = 0.0f;
            for (int i = 0; i < n_vocab; i++) {
                if (cum >= top_p) probs[indices[i]] = 0.0f;
                cum += probs[indices[i]];
            }
            free(indices);
            sum = 0.0f;
            for (int i = 0; i < n_vocab; i++) sum += probs[i];
            if (sum > 0) {
                inv_sum = 1.0f / sum;
                for (int i = 0; i < n_vocab; i++) probs[i] *= inv_sum;
            }
        }
    }

    /* Sample from distribution */
    double r = (double)rand() / (double)RAND_MAX;
    double cum = 0.0;
    for (int i = 0; i < n_vocab; i++) {
        cum += probs[i];
        if (r < cum) { free(probs); return i; }
    }
    free(probs);
    return n_vocab - 1;
}

/* ---- GGUF KV string helper ---- */
static const char *gguf_kv_str(const gguf_context *g, const char *key) {
    int idx = gguf_find_key(g, key);
    if (idx < 0) return NULL;
    if (g->kv[idx].type != GGUF_TYPE_STRING) return NULL;
    return g->kv[idx].value.str.str;
}

/* ---- Build chat prompt from OpenAI messages ---- */
/* For VLM with image: builds text_before (including <|vision_start|>) and text_after (including <|vision_end|>...<|im_start|>assistant) */
/* For text-only: builds full prompt in text_before, text_after is empty */
/* Returns number of images found (0 or 1 for now) */
static int build_chat_prompt(llm_state *s, const json_val *messages,
                              sbuf *full_prompt,
                              uint8_t **image_data, size_t *image_len,
                              int *image_w, int *image_h,
                              char *err, size_t err_cap) {
    if (!messages || messages->type != JSON_ARRAY) {
        snprintf(err, err_cap, "messages must be a non-empty array");
        return -1;
    }

    const gguf_context *gguf_main = (const gguf_context *)s->gguf;
    int n_msgs = messages->arr.count;
    int has_vision = 0;

    /* Check for vision template in GGUF metadata */
    const char *tmpl = gguf_kv_str(gguf_main, "tokenizer.chat_template");
    int use_vision = tmpl &&
                     strstr(tmpl, "<|vision_start|>") &&
                     strstr(tmpl, "<|vision_end|>");

    sbuf_init(full_prompt);
    *image_data = NULL;
    *image_len = 0;
    *image_w = 0;
    *image_h = 0;

    for (int i = 0; i < n_msgs; i++) {
        json_val *msg = &messages->arr.items[i];
        if (msg->type != JSON_OBJECT) continue;

        const char *role = j_str(msg, "role", "user");
        json_val *content = json_obj_get(msg, "content");
        if (!content) continue;

        /* Append role header */
        if (use_vision) {
            if (i > 0 && strcmp(role, "assistant") == 0) {
                sbuf_printf(full_prompt, "<|im_start|>assistant\n");
            } else if (i == 0 && strcmp(role, "system") == 0) {
                sbuf_printf(full_prompt, "<|im_start|>system\n");
            } else {
                sbuf_printf(full_prompt, "<|im_start|>user\n");
            }
        } else {
            sbuf_printf(full_prompt, "<|im_start|>%s\n", role);
        }

        if (content->type == JSON_STRING) {
            /* Plain text content */
            if (content->str.ptr) {
                sbuf_append(full_prompt, content->str.ptr);
            }
        } else if (content->type == JSON_ARRAY) {
            /* Array of content parts (text + image_url) */
            for (int j = 0; j < content->arr.count; j++) {
                json_val *part = &content->arr.items[j];
                if (part->type != JSON_OBJECT) continue;
                const char *type = j_str(part, "type", "");

                if (strcmp(type, "text") == 0) {
                    const char *text = j_str(part, "text", "");
                    if (*text) sbuf_append(full_prompt, text);
                } else if (strcmp(type, "image_url") == 0 && s->is_vlm && !has_vision) {
                    json_val *image_url = json_obj_get(part, "image_url");
                    if (image_url && image_url->type == JSON_OBJECT) {
                        const char *url = j_str(image_url, "url", "");
                        if (*url) {
                            size_t url_len = strlen(url);
                            char mime[64];
                            uint8_t *img_data = NULL;
                            size_t img_len = 0;
                            if (parse_data_uri(url, url_len, &img_data, &img_len, mime, sizeof(mime)) == 0 && img_data) {
                                int W = 0, H = 0;
                                uint8_t *rgb = server_decode_image_rgb(img_data, img_len, &W, &H);
                                free(img_data);
                                if (rgb) {
                                    *image_data = rgb;
                                    *image_w = W;
                                    *image_h = H;
                                    *image_len = (size_t)W * H * 3;
                                    has_vision = 1;

                                    /* Insert vision start/end markers */
                                    if (use_vision) {
                                        sbuf_append(full_prompt, "<|vision_start|>");
                                        sbuf_append(full_prompt, "<|vision_end|>");
                                    }
                                }
                            }
                        }
                    }
                } else if (strcmp(type, "image_url") == 0 && !s->is_vlm) {
                    snprintf(err, err_cap, "model is not a VLM (no --mmproj), but image input was provided");
                    sbuf_free(full_prompt);
                    return -1;
                }
            }
        }

        if (use_vision) {
            sbuf_append(full_prompt, "<|im_end|>\n");
        }
    }

    /* If we have image data but no vision template markers, append them to the last user message */
    if (has_vision && !use_vision) {
        /* Basic ChatML insertion */
        sbuf tmp;
        sbuf_init(&tmp);
        sbuf_printf(&tmp, "<|im_start|>user\n<|vision_start|><|vision_end|>");
        sbuf_append(&tmp, full_prompt->ptr);
        if (tmp.ptr[tmp.len - 1] == '\n') tmp.len--;
        sbuf_append(&tmp, "<|im_end|>\n<|im_start|>assistant\n");
        sbuf_free(full_prompt);
        *full_prompt = tmp;
    } else {
        sbuf_append(full_prompt, "<|im_start|>assistant\n");
    }

    return has_vision;
}

/* ---- Core generation (non-streaming) ---- */
/* Returns generated text as malloc'd string, or NULL on error. */
static char *generate_text(llm_state *s, const char *full_prompt_text,
                            uint8_t *image_rgb, int img_w, int img_h,
                            int max_tokens, float temperature, float top_p, int seed,
                            const json_val *stop_arr,
                            int *out_prompt_tokens, int *out_completion_tokens,
                            char *err, size_t err_cap) {
    bpe_vocab *vocab = (bpe_vocab *)s->vocab;
    transformer_model *model = (transformer_model *)s->model;
    vision_model *vm = (vision_model *)s->vm;

    if (seed) srand((unsigned)seed);
    llm_rng_state = seed ? (uint64_t)seed : 42;

    /* Tokenize prompt */
    int32_t tokens[8192];
    int n_tokens = bpe_tokenize(vocab, full_prompt_text, -1, tokens, 8192);
    if (n_tokens <= 0) {
        snprintf(err, err_cap, "failed to tokenize prompt");
        return NULL;
    }

    int max_seq_len = s->max_seq_len;
    int pos = 0;

    /* Prefill */
    int image_processed = 0;

    if (image_rgb && s->is_vlm && vm) {
        /* VLM: find vision markers in tokenized prompt and replace with vision embeddings */
        int32_t vision_end_tok = -1;
        int n_tok_before_markers = 0, n_tok_after_markers = 0;
        int found_start = 0;

        /* Find the <|vision_start|> and <|vision_end|> tokens in the tokenized prompt */
        /* First, tokenize the markers */
        int32_t start_tokens[4], end_tokens[4];
        int n_start = bpe_tokenize(vocab, "<|vision_start|>", -1, start_tokens, 4);
        int n_end = bpe_tokenize(vocab, "<|vision_end|>", -1, end_tokens, 4);

        if (n_start > 0 && n_end > 0) {
            int search_pos = 0;
            while (search_pos < n_tokens - n_start) {
                int match = 1;
                for (int k = 0; k < n_start; k++) {
                    if (tokens[search_pos + k] != start_tokens[k]) { match = 0; break; }
                }
                if (match) {
                    n_tok_before_markers = search_pos;
                    search_pos += n_start;
                    found_start = 1;
                    break;
                }
                search_pos++;
            }
            if (found_start) {
                while (search_pos < n_tokens - n_end) {
                    int match = 1;
                    for (int k = 0; k < n_end; k++) {
                        if (tokens[search_pos + k] != end_tokens[k]) { match = 0; break; }
                    }
                    if (match) {
                        vision_end_tok = search_pos + n_end;
                        n_tok_after_markers = n_tokens - vision_end_tok;
                        break;
                    }
                    search_pos++;
                }
            }

            if (found_start && vision_end_tok > 0) {
                /* Prefill text tokens before vision markers */
                for (int i = 0; i < n_tok_before_markers; i++) {
                    transformer_forward(model, tokens[i], pos);
                    pos++;
                    if (pos >= max_seq_len) {
                        snprintf(err, err_cap, "context length exceeded during vision prefill");
                        return NULL;
                    }
                }
            }

            /* Encode image */
            int grid = vm->patch_size * vm->spatial_merge;
            int long_side = (img_w > img_h) ? img_w : img_h;
            float scale = 384.0f / (float)long_side;
            int dw = (int)(img_w * scale); if (dw < grid) dw = grid;
            int dh = (int)(img_h * scale); if (dh < grid) dh = grid;
            dw = (dw / grid) * grid;
            dh = (dh / grid) * grid;

            uint8_t *resized = img_resize_bilinear(image_rgb, img_w, img_h, dw, dh);

            float *img_norm = vision_normalize_image(vm, resized, dw, dh);
            free(resized);

            float *vision_embd = vision_encode(vm, img_norm, dw, dh, 1);
            free(img_norm);

            if (!vision_embd) {
                snprintf(err, err_cap, "vision encoding failed");
                return NULL;
            }

            int patches_w = dw / vm->patch_size;
            int patches_h = dh / vm->patch_size;
            int n_vision_tokens = (patches_w / vm->spatial_merge) * (patches_h / vm->spatial_merge);
            int proj_dim = vm->proj_dim;

            /* Prefill vision embeddings */
            for (int i = 0; i < n_vision_tokens; i++) {
                float *embd_i = vision_embd + i * proj_dim;
                transformer_forward_embd(model, embd_i, pos);
                pos++;
                if (pos >= max_seq_len) break;
            }
            free(vision_embd);

            /* Prefill text tokens after vision markers */
            for (int i = 0; i < n_tok_after_markers; i++) {
                int idx = vision_end_tok + i;
                float *logits = NULL;
                if (i == n_tok_after_markers - 1) {
                    logits = transformer_forward_logits(model, tokens[idx], pos);
                } else {
                    transformer_forward(model, tokens[idx], pos);
                }
                pos++;
                if (pos >= max_seq_len) break;
                (void)logits;
            }

            image_processed = 1;
        }
    }

    if (!image_processed) {
        /* Text-only prefill */
        for (int i = 0; i < n_tokens; i++) {
            float *logits = NULL;
            if (i == n_tokens - 1) {
                logits = transformer_forward_logits(model, tokens[i], pos);
            } else {
                transformer_forward(model, tokens[i], pos);
            }
            pos++;
            if (pos >= max_seq_len) break;
            (void)logits;
        }
    }

    if (out_prompt_tokens) *out_prompt_tokens = pos;

    /* Generation */
    sbuf output;
    sbuf_init(&output);

    int32_t next_token = -1;
    int gen_tokens = 0;

    for (int g = 0; g < max_tokens; g++) {
        float *logits;
        if (g == 0) {
            logits = model->logits;
        } else {
            logits = transformer_forward_logits(model, next_token, pos);
            pos++;
            if (pos >= max_seq_len) break;
        }

        if (!logits) break;

        next_token = sample_token(logits, model->n_vocab, temperature, top_p);
        if (next_token < 0) break;

        /* Check for EOS */
        if (next_token == s->eos_id || next_token == s->eot_id) {
            gen_tokens++;
            break;
        }

        /* Decode token */
        const char *tok_str = bpe_token_to_str(vocab, next_token);
        if (tok_str) {
            int dec_len;
            char *decoded = bpe_byte_decode(tok_str, (int)strlen(tok_str), &dec_len);
            if (decoded) {
                /* Check stop sequences */
                size_t old_len = output.len;
                sbuf_appendn(&output, decoded, (size_t)dec_len);
                if (matches_stop_sequence(output.ptr, stop_arr)) {
                    /* Truncate at stop sequence */
                    output.len = old_len;
                    output.ptr[output.len] = 0;
                    free(decoded);
                    gen_tokens++;
                    break;
                }
                free(decoded);
            }
        }

        gen_tokens++;
        if (g == 0) pos++;
    }

    if (out_completion_tokens) *out_completion_tokens = gen_tokens;
    return output.ptr;
}

/* ---- Core generation for streaming ---- */
static int generate_text_stream(llm_state *s, int fd, const char *full_prompt_text,
                                 uint8_t *image_rgb, int img_w, int img_h,
                                 int max_tokens, float temperature, float top_p, int seed,
                                 const json_val *stop_arr,
                                 const char *chat_id, int is_chat,
                                 int *out_prompt_tokens, int *out_completion_tokens,
                                 char *err, size_t err_cap) {
    bpe_vocab *vocab = (bpe_vocab *)s->vocab;
    transformer_model *model = (transformer_model *)s->model;
    vision_model *vm = (vision_model *)s->vm;

    if (seed) srand((unsigned)seed);
    llm_rng_state = seed ? (uint64_t)seed : 42;

    /* Tokenize prompt */
    int32_t tokens[8192];
    int n_tokens = bpe_tokenize(vocab, full_prompt_text, -1, tokens, 8192);
    if (n_tokens <= 0) {
        snprintf(err, err_cap, "failed to tokenize prompt");
        return -1;
    }

    /* Prefill (same as non-streaming) */
    int max_seq_len = s->max_seq_len;
    int pos = 0;
    int image_processed = 0;

    if (image_rgb && s->is_vlm && vm) {
        int32_t vision_end_tok = -1;
        int n_tok_before_markers = 0, n_tok_after_markers = 0;
        int found_start = 0;

        int32_t start_tokens[4], end_tokens[4];
        int n_start = bpe_tokenize(vocab, "<|vision_start|>", -1, start_tokens, 4);
        int n_end = bpe_tokenize(vocab, "<|vision_end|>", -1, end_tokens, 4);

        if (n_start > 0 && n_end > 0) {
            int search_pos = 0;
            while (search_pos < n_tokens - n_start) {
                int match = 1;
                for (int k = 0; k < n_start; k++) {
                    if (tokens[search_pos + k] != start_tokens[k]) { match = 0; break; }
                }
                if (match) {
                    n_tok_before_markers = search_pos;
                    search_pos += n_start;
                    found_start = 1;
                    break;
                }
                search_pos++;
            }
            if (found_start) {
                while (search_pos < n_tokens - n_end) {
                    int match = 1;
                    for (int k = 0; k < n_end; k++) {
                        if (tokens[search_pos + k] != end_tokens[k]) { match = 0; break; }
                    }
                    if (match) {
                        vision_end_tok = search_pos + n_end;
                        n_tok_after_markers = n_tokens - vision_end_tok;
                        break;
                    }
                    search_pos++;
                }
            }

            if (found_start && vision_end_tok > 0) {
                /* Prefill text tokens before vision markers */
                for (int i = 0; i < n_tok_before_markers; i++) {
                    transformer_forward(model, tokens[i], pos); pos++;
                    if (pos >= max_seq_len) return -1;
                }
            }

            int grid = vm->patch_size * vm->spatial_merge;
            int long_side = (img_w > img_h) ? img_w : img_h;
            float scale = 384.0f / (float)long_side;
            int dw = (int)(img_w * scale); if (dw < grid) dw = grid;
            int dh = (int)(img_h * scale); if (dh < grid) dh = grid;
            dw = (dw / grid) * grid;
            dh = (dh / grid) * grid;

            uint8_t *resized = img_resize_bilinear(image_rgb, img_w, img_h, dw, dh);
            float *img_norm = vision_normalize_image(vm, resized, dw, dh);
            free(resized);
            float *vision_embd = vision_encode(vm, img_norm, dw, dh, 1);
            free(img_norm);
            if (!vision_embd) { snprintf(err, err_cap, "vision encoding failed"); return -1; }

            int patches_w = dw / vm->patch_size;
            int patches_h = dh / vm->patch_size;
            int n_vision_tokens = (patches_w / vm->spatial_merge) * (patches_h / vm->spatial_merge);
            int proj_dim = vm->proj_dim;

            for (int i = 0; i < n_vision_tokens; i++) {
                transformer_forward_embd(model, vision_embd + i * proj_dim, pos); pos++;
                if (pos >= max_seq_len) break;
            }
            free(vision_embd);

            for (int i = 0; i < n_tok_after_markers; i++) {
                int idx = vision_end_tok + i;
                if (i == n_tok_after_markers - 1)
                    transformer_forward_logits(model, tokens[idx], pos);
                else
                    transformer_forward(model, tokens[idx], pos);
                pos++;
                if (pos >= max_seq_len) break;
            }
            image_processed = 1;
        }
    }

    if (!image_processed) {
        for (int i = 0; i < n_tokens; i++) {
            if (i == n_tokens - 1)
                transformer_forward_logits(model, tokens[i], pos);
            else
                transformer_forward(model, tokens[i], pos);
            pos++;
            if (pos >= max_seq_len) break;
        }
    }

    if (out_prompt_tokens) *out_prompt_tokens = pos;

    /* Send SSE header with role (for chat) */
    if (is_chat) {
        char role_sse[1024];
        snprintf(role_sse, sizeof(role_sse),
            "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,"
            "\"model\":\"%s\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
            chat_id, (long)time(NULL), s->model_path);
        send(fd, role_sse, strlen(role_sse), 0);
    }

    /* Generation */
    int32_t next_token = -1;
    int gen_tokens = 0;
    const char *finish_reason = NULL;

    for (int g = 0; g < max_tokens; g++) {
        float *logits;
        if (g == 0) {
            logits = model->logits;
        } else {
            logits = transformer_forward_logits(model, next_token, pos);
            pos++;
            if (pos >= max_seq_len) break;
        }

        if (!logits) break;

        next_token = sample_token(logits, model->n_vocab, temperature, top_p);
        if (next_token < 0) break;

        if (next_token == s->eos_id || next_token == s->eot_id) {
            finish_reason = "stop";
            gen_tokens++;
            break;
        }

        const char *tok_str = bpe_token_to_str(vocab, next_token);
        if (tok_str) {
            int dec_len;
            char *decoded = bpe_byte_decode(tok_str, (int)strlen(tok_str), &dec_len);
            if (decoded) {
                /* Check stop sequence */
                sbuf tmp;
                sbuf_init(&tmp);
                sbuf_appendn(&tmp, decoded, (size_t)dec_len);
                if (stop_arr && stop_arr->type == JSON_ARRAY) {
                    int stopped = 0;
                    for (int si = 0; si < stop_arr->arr.count; si++) {
                        json_val *sv = &stop_arr->arr.items[si];
                        if (sv->type == JSON_STRING && sv->str.ptr) {
                            size_t slen = strlen(sv->str.ptr);
                            if ((size_t)dec_len >= slen && strncmp(decoded + dec_len - slen, sv->str.ptr, slen) == 0) {
                                stopped = 1;
                                break;
                            }
                        }
                    }
                    if (stopped) {
                        free(decoded);
                        sbuf_free(&tmp);
                        finish_reason = "stop";
                        gen_tokens++;
                        break;
                    }
                }

                /* Escape and send as SSE */
                char *escaped = json_escape_dup(decoded);
                char sse_buf[4096];
                if (is_chat) {
                    snprintf(sse_buf, sizeof(sse_buf),
                        "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,"
                        "\"model\":\"%s\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}\n\n",
                        chat_id, (long)time(NULL), s->model_path, escaped);
                } else {
                    snprintf(sse_buf, sizeof(sse_buf),
                        "data: {\"id\":\"%s\",\"object\":\"text_completion\",\"created\":%ld,"
                        "\"model\":\"%s\",\"choices\":[{\"index\":0,\"text\":\"%s\",\"finish_reason\":null}]}\n\n",
                        chat_id, (long)time(NULL), s->model_path, escaped);
                }
                if (send(fd, sse_buf, strlen(sse_buf), 0) < 0) {
                    free(escaped);
                    free(decoded);
                    sbuf_free(&tmp);
                    gen_tokens++;
                    break;
                }
                free(escaped);
                free(decoded);
                sbuf_free(&tmp);
            }
        }

        gen_tokens++;
        if (g == 0) pos++;
    }

    if (!finish_reason) finish_reason = (pos >= max_seq_len) ? "length" : "stop";

    /* Send final SSE with finish_reason */
    char final_sse[1024];
    if (is_chat) {
        snprintf(final_sse, sizeof(final_sse),
            "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,"
            "\"model\":\"%s\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"%s\"}]}\n\n",
            chat_id, (long)time(NULL), s->model_path, finish_reason);
    } else {
        snprintf(final_sse, sizeof(final_sse),
            "data: {\"id\":\"%s\",\"object\":\"text_completion\",\"created\":%ld,"
            "\"model\":\"%s\",\"choices\":[{\"index\":0,\"text\":\"\",\"finish_reason\":\"%s\"}]}\n\n",
            chat_id, (long)time(NULL), s->model_path, finish_reason);
    }
    send(fd, final_sse, strlen(final_sse), 0);
    send(fd, "data: [DONE]\n\n", 14, 0);

    if (out_completion_tokens) *out_completion_tokens = gen_tokens;
    return 0;
}

/* ---- Public API ---- */

int llm_init(llm_state *s, const char *model_path, const char *mmproj_path) {
    memset(s, 0, sizeof(*s));

    if (!model_path || !*model_path) return -1;
    snprintf(s->model_path, sizeof(s->model_path), "%s", model_path);

    /* Load main GGUF model */
    int use_mmap = 1;
    gguf_context *gguf = gguf_open(model_path, use_mmap);
    if (!gguf) {
        fprintf(stderr, "[llm] failed to open GGUF: %s\n", model_path);
        return -1;
    }
    s->gguf = gguf;

    /* Load vocab */
    bpe_vocab *vocab = bpe_vocab_load(gguf);
    if (!vocab) {
        fprintf(stderr, "[llm] failed to load vocab from %s\n", model_path);
        gguf_close(gguf);
        return -1;
    }
    s->vocab = vocab;
    s->eos_id = bpe_eos_id(vocab);
    s->eot_id = bpe_eot_id(vocab);

    /* Load transformer model */
    int max_seq_len = 8192;
    transformer_model *model = transformer_load(gguf, max_seq_len);
    if (!model) {
        fprintf(stderr, "[llm] failed to load model from %s\n", model_path);
        bpe_vocab_free(vocab);
        gguf_close(gguf);
        return -1;
    }
    s->model = model;
    s->n_embd = model->n_embd;
    s->max_seq_len = model->max_seq_len;

    fprintf(stderr, "[llm] loaded: layers=%d n_embd=%d n_vocab=%d max_seq=%d\n",
            model->n_layers, model->n_embd, model->n_vocab, model->max_seq_len);

    /* Optionally load vision encoder */
    if (mmproj_path && *mmproj_path) {
        snprintf(s->mmproj_path, sizeof(s->mmproj_path), "%s", mmproj_path);
        gguf_context *gguf_mmproj = gguf_open(mmproj_path, 1);
        if (!gguf_mmproj) {
            fprintf(stderr, "[llm] failed to open mmproj: %s (proceeding as text-only LLM)\n", mmproj_path);
        } else {
            s->gguf_mmproj = gguf_mmproj;
            vision_model *vm = vision_load(gguf_mmproj);
            if (!vm) {
                fprintf(stderr, "[llm] failed to load vision encoder from %s (proceeding as text-only LLM)\n", mmproj_path);
                gguf_close(gguf_mmproj);
                s->gguf_mmproj = NULL;
            } else {
                s->vm = vm;
                s->is_vlm = 1;
                s->patch_size = vm->patch_size;
                s->spatial_merge = vm->spatial_merge;
                s->proj_dim = vm->proj_dim;
                fprintf(stderr, "[llm] vision encoder loaded: %d blocks, patch=%d proj=%d\n",
                        vm->n_blocks, vm->patch_size, vm->proj_dim);
            }
        }
    }

    s->loaded = 1;
    return 0;
}

void llm_free(llm_state *s) {
    if (!s || !s->loaded) return;
    if (s->vm) vision_free((vision_model *)s->vm);
    if (s->gguf_mmproj) gguf_close((gguf_context *)s->gguf_mmproj);
    if (s->model) transformer_free((transformer_model *)s->model);
    if (s->vocab) bpe_vocab_free((bpe_vocab *)s->vocab);
    if (s->gguf) gguf_close((gguf_context *)s->gguf);
    memset(s, 0, sizeof(*s));
}

char *llm_chat_completion(llm_state *s, const json_val *messages,
                           int max_tokens, float temperature, float top_p, int seed,
                           const json_val *stop_arr,
                           int *status, char *err, size_t err_cap) {
    if (!s || !s->loaded) {
        snprintf(err, err_cap, "model not loaded");
        if (status) *status = 500;
        return NULL;
    }

    sbuf full_prompt;
    uint8_t *image_rgb = NULL;
    size_t image_len = 0;
    int img_w = 0, img_h = 0;

    int has_image = build_chat_prompt(s, messages, &full_prompt,
                                       &image_rgb, &image_len, &img_w, &img_h,
                                       err, err_cap);
    if (has_image < 0) {
        if (status) *status = 400;
        return NULL;
    }

    if (full_prompt.len == 0) {
        sbuf_free(&full_prompt);
        snprintf(err, err_cap, "empty prompt after processing messages");
        if (status) *status = 400;
        return NULL;
    }

    int prompt_tokens = 0, completion_tokens = 0;
    char *generated = generate_text(s, full_prompt.ptr,
                                     image_rgb, img_w, img_h,
                                     max_tokens, temperature, top_p, seed,
                                     stop_arr, &prompt_tokens, &completion_tokens,
                                     err, err_cap);
    sbuf_free(&full_prompt);
    free(image_rgb);

    if (!generated) {
        if (status) *status = 500;
        return NULL;
    }

    /* Build OpenAI-format response */
    char id_buf[64];
    snprintf(id_buf, sizeof(id_buf), "chatcmpl-%08x", (unsigned)(rand() & 0x7FFFFFFF));

    char *content_escaped = json_escape_dup(generated);
    sbuf resp;
    sbuf_init(&resp);
    sbuf_printf(&resp,
        "{\"id\":\"%s\",\"object\":\"chat.completion\",\"created\":%ld,"
        "\"model\":\"%s\",\"choices\":[{\"index\":0,\"message\":{"
        "\"role\":\"assistant\",\"content\":\"%s\""
        "},\"finish_reason\":\"stop\"}],"
        "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",
        id_buf, (long)time(NULL), s->model_path,
        content_escaped,
        prompt_tokens, completion_tokens, prompt_tokens + completion_tokens);
    free(content_escaped);
    free(generated);

    if (status) *status = 200;
    return resp.ptr;
}

int llm_chat_completion_stream(llm_state *s, int fd, const json_val *messages,
                                int max_tokens, float temperature, float top_p, int seed,
                                const json_val *stop_arr,
                                char *err, size_t err_cap) {
    if (!s || !s->loaded) {
        snprintf(err, err_cap, "model not loaded");
        return -1;
    }

    sbuf full_prompt;
    uint8_t *image_rgb = NULL;
    size_t image_len = 0;
    int img_w = 0, img_h = 0;

    int has_image = build_chat_prompt(s, messages, &full_prompt,
                                       &image_rgb, &image_len, &img_w, &img_h,
                                       err, err_cap);
    if (has_image < 0) return -1;

    char id_buf[64];
    snprintf(id_buf, sizeof(id_buf), "chatcmpl-%08x", (unsigned)(rand() & 0x7FFFFFFF));

    int prompt_tokens = 0, completion_tokens = 0;
    int rc = generate_text_stream(s, fd, full_prompt.ptr,
                                   image_rgb, img_w, img_h,
                                   max_tokens, temperature, top_p, seed,
                                   stop_arr, id_buf, 1,
                                   &prompt_tokens, &completion_tokens,
                                   err, err_cap);

    sbuf_free(&full_prompt);
    free(image_rgb);
    return rc;
}

char *llm_text_completion(llm_state *s, const char *prompt,
                           int max_tokens, float temperature, float top_p, int seed,
                           const json_val *stop_arr,
                           int *status, char *err, size_t err_cap) {
    if (!s || !s->loaded) {
        snprintf(err, err_cap, "model not loaded");
        if (status) *status = 500;
        return NULL;
    }
    if (!prompt || !*prompt) {
        snprintf(err, err_cap, "empty prompt");
        if (status) *status = 400;
        return NULL;
    }

    int prompt_tokens = 0, completion_tokens = 0;
    char *generated = generate_text(s, prompt, NULL, 0, 0,
                                     max_tokens, temperature, top_p, seed,
                                     stop_arr, &prompt_tokens, &completion_tokens,
                                     err, err_cap);
    if (!generated) {
        if (status) *status = 500;
        return NULL;
    }

    char id_buf[64];
    snprintf(id_buf, sizeof(id_buf), "cmpl-%08x", (unsigned)(rand() & 0x7FFFFFFF));

    char *content_escaped = json_escape_dup(generated);
    sbuf resp;
    sbuf_init(&resp);
    sbuf_printf(&resp,
        "{\"id\":\"%s\",\"object\":\"text_completion\",\"created\":%ld,"
        "\"model\":\"%s\",\"choices\":[{\"index\":0,\"text\":\"%s\","
        "\"finish_reason\":\"stop\"}],"
        "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",
        id_buf, (long)time(NULL), s->model_path,
        content_escaped,
        prompt_tokens, completion_tokens, prompt_tokens + completion_tokens);
    free(content_escaped);
    free(generated);

    if (status) *status = 200;
    return resp.ptr;
}

int llm_text_completion_stream(llm_state *s, int fd, const char *prompt,
                                int max_tokens, float temperature, float top_p, int seed,
                                const json_val *stop_arr,
                                char *err, size_t err_cap) {
    if (!s || !s->loaded) {
        snprintf(err, err_cap, "model not loaded");
        return -1;
    }
    if (!prompt || !*prompt) {
        snprintf(err, err_cap, "empty prompt");
        return -1;
    }

    char id_buf[64];
    snprintf(id_buf, sizeof(id_buf), "cmpl-%08x", (unsigned)(rand() & 0x7FFFFFFF));

    int prompt_tokens = 0, completion_tokens = 0;
    return generate_text_stream(s, fd, prompt, NULL, 0, 0,
                                 max_tokens, temperature, top_p, seed,
                                 stop_arr, id_buf, 0,
                                 &prompt_tokens, &completion_tokens,
                                 err, err_cap);
}
