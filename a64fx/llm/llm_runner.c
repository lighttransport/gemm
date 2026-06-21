/*
 * a64fx/llm/llm_runner.c
 *
 * End-to-end VLM→LLM runner for Qwen3-VL on A64FX.
 *
 * Usage:
 *   ./llm_runner <model.gguf> [<mmproj.gguf>] [image] [options]
 *
 * Examples:
 *   # text-only
 *   ./llm_runner model.gguf --prompt "Hello" --max-gen 32
 *
 *   # vision+text (synthetic checkerboard if no image given)
 *   ./llm_runner model.gguf mmproj.gguf --image-size 384 \
 *                --prompt "Describe the image" --max-gen 64 \
 *                --vit-dtype fp16 --vit-threads 12 --llm-threads 12
 *
 * The vision path reuses a64fx/vlm's vit_a64fx_encode (FP16/BF16 packed-B
 * SVE GEMM). The LLM path runs common/transformer.h (scalar on A64FX —
 * no SVE kernels there yet). DeepStack injection is activated by setting
 * model->ds_embd_stride = proj_dim * (1 + n_deepstack); the encoder
 * output is laid out as [n_merged, proj_dim*(1+n_ds)] which matches.
 */

#define GGUF_LOADER_IMPLEMENTATION
#include "gguf_loader.h"

#define GGML_DEQUANT_IMPLEMENTATION
#include "ggml_dequant.h"

#define BPE_TOKENIZER_IMPLEMENTATION
#include "bpe_tokenizer.h"

#define PROFILER_IMPLEMENTATION
#include "profiler.h"

#define TRANSFORMER_IMPLEMENTATION
#include "transformer.h"

#define VISION_ENCODER_IMPLEMENTATION
#include "vision_encoder.h"

#define IMAGE_UTILS_IMPLEMENTATION
#include "image_utils.h"

#include "vit_a64fx.h"
#include "vlm_parallel.h"
#include "tensor_dump.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>

/* ── helpers ──────────────────────────────────────────────────────── */

static double mono_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int parse_dtype(const char *s) {
    if (!strcasecmp(s, "fp32")) return VIT_DTYPE_FP32;
    if (!strcasecmp(s, "bf16")) return VIT_DTYPE_BF16;
    if (!strcasecmp(s, "fp16")) return VIT_DTYPE_FP16;
    return -1;
}

static const char *dtype_name(int dt) {
    switch (dt) {
        case VIT_DTYPE_FP32: return "fp32";
        case VIT_DTYPE_BF16: return "bf16";
        case VIT_DTYPE_FP16: return "fp16";
        default:             return "?";
    }
}

static int arg_is_image(const char *s) {
    if (!s) return 0;
    int n = (int)strlen(s);
    if (n < 4) return 0;
    const char *e4 = s + n - 4;
    const char *e5 = (n >= 5) ? s + n - 5 : NULL;
    if (!strcasecmp(e4, ".png")) return 1;
    if (!strcasecmp(e4, ".jpg")) return 1;
    if (!strcasecmp(e4, ".bmp")) return 1;
    if (!strcasecmp(e4, ".ppm")) return 1;
    if (e5 && !strcasecmp(e5, ".jpeg")) return 1;
    return 0;
}

static int arg_is_gguf(const char *s) {
    if (!s) return 0;
    int n = (int)strlen(s);
    return (n >= 5) && !strcasecmp(s + n - 5, ".gguf");
}

static uint8_t *make_checkerboard(int w, int h, int cell) {
    uint8_t *img = (uint8_t *)malloc((size_t)w * h * 3);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int c = ((x / cell) + (y / cell)) & 1;
            uint8_t v = c ? 255 : 0;
            size_t i = ((size_t)y * w + x) * 3;
            img[i + 0] = v; img[i + 1] = v; img[i + 2] = v;
        }
    }
    return img;
}

static const char *gguf_get_kv_string(const gguf_context *g, const char *key) {
    int idx = gguf_find_key(g, key);
    if (idx < 0) return NULL;
    if (g->kv[idx].type != GGUF_TYPE_STRING) return NULL;
    return g->kv[idx].value.str.str;
}

static void build_chat_prompt(const gguf_context *gguf_main,
                              const char *user_prompt,
                              int have_vision,
                              char *text_before, size_t cap_before,
                              char *text_after,  size_t cap_after) {
    const char *tmpl = gguf_get_kv_string(gguf_main, "tokenizer.chat_template");
    int has_chatml = tmpl &&
                     strstr(tmpl, "<|im_start|>") &&
                     strstr(tmpl, "<|im_end|>");
    int has_vision_tok = tmpl &&
                         strstr(tmpl, "<|vision_start|>") &&
                         strstr(tmpl, "<|vision_end|>");
    (void)has_chatml; (void)has_vision_tok;

    if (have_vision) {
        snprintf(text_before, cap_before, "<|im_start|>user\n<|vision_start|>");
        snprintf(text_after,  cap_after,  "<|vision_end|>%s<|im_end|>\n<|im_start|>assistant\n", user_prompt);
    } else {
        snprintf(text_before, cap_before, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", user_prompt);
        text_after[0] = '\0';
    }
}

static char *xstrdup_local(const char *s) {
    if (!s) return NULL;
    size_t n = strlen(s) + 1;
    char *p = (char *)malloc(n);
    if (p) memcpy(p, s, n);
    return p;
}

static int sb_reserve(char **buf, size_t *cap, size_t need) {
    if (need <= *cap) return 0;
    size_t nc = *cap ? *cap : 256;
    while (nc < need) nc *= 2;
    char *p = (char *)realloc(*buf, nc);
    if (!p) return -1;
    *buf = p;
    *cap = nc;
    return 0;
}

static int sb_appendn(char **buf, size_t *len, size_t *cap, const char *s, size_t n) {
    if (sb_reserve(buf, cap, *len + n + 1) != 0) return -1;
    memcpy(*buf + *len, s, n);
    *len += n;
    (*buf)[*len] = 0;
    return 0;
}

static int sb_append(char **buf, size_t *len, size_t *cap, const char *s) {
    return sb_appendn(buf, len, cap, s, strlen(s));
}

static char *json_escape_dup(const char *s) {
    if (!s) s = "";
    size_t cap = strlen(s) + 16, len = 0;
    char *out = (char *)malloc(cap);
    if (!out) return NULL;
    out[0] = 0;
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        char tmp[8];
        const char *add = NULL;
        size_t n = 0;
        switch (*p) {
            case '\\': add = "\\\\"; n = 2; break;
            case '"':  add = "\\\""; n = 2; break;
            case '\n': add = "\\n";  n = 2; break;
            case '\r': add = "\\r";  n = 2; break;
            case '\t': add = "\\t";  n = 2; break;
            default:
                if (*p < 0x20) {
                    snprintf(tmp, sizeof(tmp), "\\u%04x", *p);
                    add = tmp; n = 6;
                } else {
                    tmp[0] = (char)*p; tmp[1] = 0;
                    add = tmp; n = 1;
                }
                break;
        }
        if (sb_appendn(&out, &len, &cap, add, n) != 0) {
            free(out); return NULL;
        }
    }
    return out;
}

static const char *json_find_key(const char *json, const char *key) {
    char pat[128];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = json;
    while ((p = strstr(p, pat)) != NULL) {
        p += strlen(pat);
        while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
        if (*p == ':') return p + 1;
    }
    return NULL;
}

static char *json_get_string_field(const char *json, const char *key, const char *def) {
    const char *p = json_find_key(json, key);
    if (!p) return def ? xstrdup_local(def) : NULL;
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    if (*p != '"') return def ? xstrdup_local(def) : NULL;
    p++;
    size_t cap = 128, len = 0;
    char *out = (char *)malloc(cap);
    if (!out) return NULL;
    out[0] = 0;
    while (*p && *p != '"') {
        char c = *p++;
        if (c == '\\') {
            c = *p++;
            if (!c) break;
            switch (c) {
                case 'n': c = '\n'; break;
                case 'r': c = '\r'; break;
                case 't': c = '\t'; break;
                case '"': case '\\': case '/': break;
                case 'u':
                    /* Requests from the Python wrapper are UTF-8 JSON. Keep
                     * non-ASCII escapes as '?' instead of implementing full
                     * UTF-16 decoding in this tiny control parser. */
                    for (int i = 0; i < 4 && *p; i++) p++;
                    c = '?';
                    break;
                default: break;
            }
        }
        if (sb_appendn(&out, &len, &cap, &c, 1) != 0) {
            free(out); return NULL;
        }
    }
    return out;
}

static int json_get_int_field(const char *json, const char *key, int def) {
    const char *p = json_find_key(json, key);
    if (!p) return def;
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    char *end = NULL;
    long v = strtol(p, &end, 10);
    return (end && end != p) ? (int)v : def;
}

static int json_get_bool_field(const char *json, const char *key, int def) {
    const char *p = json_find_key(json, key);
    if (!p) return def;
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    if (!strncmp(p, "true", 4)) return 1;
    if (!strncmp(p, "false", 5)) return 0;
    return def;
}

static char *decode_token_text(const bpe_vocab *vocab, int32_t token) {
    const char *tok_str = bpe_token_to_str(vocab, token);
    if (!tok_str) return xstrdup_local("");
    int dec_len = 0;
    char *decoded = bpe_byte_decode(tok_str, (int)strlen(tok_str), &dec_len);
    if (decoded) {
        char *out = (char *)malloc((size_t)dec_len + 1);
        if (out) {
            memcpy(out, decoded, (size_t)dec_len);
            out[dec_len] = 0;
        }
        free(decoded);
        return out;
    }
    return xstrdup_local(tok_str);
}

typedef void (*llm_token_cb)(const char *text, int32_t token, void *user);

/* Debug: when TF_DUMP_LOGITS is set, print final-logit health each gen step.
 * Discriminates the "!!!!" degenerate-output failure mode: NaN/inf => numerical
 * overflow in the forward; all-equal finite => uniform logits (structural);
 * all-zero => hidden state never written into lm_head. step<0 = prefill row. */
static void dump_logit_stats(const float *lg, int n, int step, int chosen) {
    if (!getenv("TF_DUMP_LOGITS") || !lg || n <= 0) return;
    float mn = lg[0], mx = lg[0];
    double sum = 0;
    long nnan = 0, ninf = 0, nz = 0, distinct1 = 0;
    int amax = 0;
    for (int j = 0; j < n; j++) {
        float v = lg[j];
        if (v != v) { nnan++; continue; }
        if (v > 1e30f || v < -1e30f) { ninf++; }
        if (v == 0.0f) nz++;
        if (v != lg[0]) distinct1 = 1;
        sum += v;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        if (v > lg[amax]) amax = j;
    }
    long finite = (long)n - nnan;
    fprintf(stderr,
        "[TF_DUMP_LOGITS] step=%d n=%d argmax=%d chosen=%d min=%.6g max=%.6g "
        "mean=%.6g range=%.6g nan=%ld inf=%ld zeros=%ld all_equal=%s lg[0]=%.6g lg[argmax]=%.6g\n",
        step, n, amax, chosen, mn, mx, finite ? sum / (double)finite : 0.0,
        (double)mx - (double)mn, nnan, ninf, nz, distinct1 ? "no" : "YES",
        lg[0], lg[amax]);
    fflush(stderr);
}

static char *run_text_generation(gguf_context *gguf_main,
                                 bpe_vocab *vocab,
                                 transformer_model *model,
                                 const char *user_prompt,
                                 int max_gen,
                                 int *out_prompt_tokens,
                                 int *out_completion_tokens,
                                 const char **out_finish_reason,
                                 llm_token_cb on_token,
                                 void *on_token_user,
                                 char **err_msg) {
    transformer_reset_runtime_state(model);

    size_t prompt_cap = strlen(user_prompt) + 256;
    char *text_before = (char *)malloc(prompt_cap);
    char *text_after  = (char *)malloc(prompt_cap);
    if (!text_before || !text_after) {
        free(text_before); free(text_after);
        if (err_msg) *err_msg = xstrdup_local("out of memory");
        return NULL;
    }
    build_chat_prompt(gguf_main, user_prompt, 0,
                      text_before, prompt_cap,
                      text_after,  prompt_cap);

    int tok_cap = model->max_seq_len > 256 ? model->max_seq_len : 256;
    int32_t *tokens_before = (int32_t *)malloc((size_t)tok_cap * sizeof(int32_t));
    int n_before = bpe_tokenize(vocab, text_before, -1, tokens_before, tok_cap);
    free(text_before); free(text_after);
    if (!tokens_before || n_before <= 0) {
        free(tokens_before);
        if (err_msg) *err_msg = xstrdup_local("tokenization failed");
        return NULL;
    }
    if (n_before + max_gen >= model->max_seq_len) {
        free(tokens_before);
        if (err_msg) {
            char buf[160];
            snprintf(buf, sizeof(buf), "prompt plus max_tokens exceeds max_seq_len (%d)", model->max_seq_len);
            *err_msg = xstrdup_local(buf);
        }
        return NULL;
    }

    float *logits = NULL;
    int pos = 0;
    if (getenv("TF_PREFILL_GEMM") && n_before > 0) {
        logits = transformer_prefill_gemm(model, tokens_before, n_before, 0);
        if (logits) pos = n_before;
    }
    if (!logits) {
        for (int i = 0; i < n_before; i++) {
            if (i == n_before - 1) logits = transformer_forward_logits(model, tokens_before[i], pos);
            else transformer_forward(model, tokens_before[i], pos);
            pos++;
        }
    }
    free(tokens_before);
    if (!logits) {
        if (err_msg) *err_msg = xstrdup_local("model did not produce logits");
        return NULL;
    }

    char *content = NULL;
    size_t content_len = 0, content_cap = 0;
    const char *finish = "length";
    int n_gen = 0;
    int32_t next = -1;
    for (int g = 0; g < max_gen; g++) {
        if (g > 0) {
            logits = transformer_forward_logits(model, next, pos);
            pos++;
        }
        if (!logits) {
            finish = "error";
            break;
        }
        next = 0;
        for (int j = 1; j < model->n_vocab; j++) {
            if (logits[j] > logits[next]) next = j;
        }
        dump_logit_stats(logits, model->n_vocab, g, next);
        if (next == vocab->eos_id || next == vocab->eot_id) {
            finish = "stop";
            break;
        }
        char *piece = decode_token_text(vocab, next);
        if (!piece) {
            finish = "error";
            break;
        }
        if (sb_append(&content, &content_len, &content_cap, piece) != 0) {
            free(piece);
            finish = "error";
            break;
        }
        if (on_token) on_token(piece, next, on_token_user);
        free(piece);
        n_gen++;
    }
    if (!content) content = xstrdup_local("");
    if (out_prompt_tokens) *out_prompt_tokens = n_before;
    if (out_completion_tokens) *out_completion_tokens = n_gen;
    if (out_finish_reason) *out_finish_reason = finish;
    return content;
}

typedef struct {
    const char *id;
} stdio_token_ctx;

static void stdio_token_event(const char *text, int32_t token, void *user) {
    stdio_token_ctx *ctx = (stdio_token_ctx *)user;
    char *eid = json_escape_dup(ctx->id ? ctx->id : "");
    char *et = json_escape_dup(text);
    printf("{\"id\":\"%s\",\"type\":\"token\",\"token\":%d,\"text\":\"%s\"}\n",
           eid ? eid : "", token, et ? et : "");
    fflush(stdout);
    free(eid); free(et);
}

static int llm_serve_stdio(gguf_context *gguf_main,
                           bpe_vocab *vocab,
                           transformer_model *model,
                           int default_max_gen) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    printf("{\"type\":\"ready\",\"max_seq_len\":%d,\"n_vocab\":%d}\n",
           model->max_seq_len, model->n_vocab);
    fflush(stdout);

    char *line = NULL;
    size_t cap = 0;
    while (getline(&line, &cap, stdin) >= 0) {
        char *id = json_get_string_field(line, "id", "");
        char *prompt = json_get_string_field(line, "prompt", NULL);
        int max_gen = json_get_int_field(line, "max_tokens",
                         json_get_int_field(line, "max_gen", default_max_gen));
        int stream = json_get_bool_field(line, "stream", 0);
        if (max_gen <= 0) max_gen = default_max_gen > 0 ? default_max_gen : 1;
        if (max_gen > model->max_seq_len - 1) max_gen = model->max_seq_len - 1;

        if (!prompt) {
            char *eid = json_escape_dup(id ? id : "");
            printf("{\"id\":\"%s\",\"type\":\"error\",\"error\":\"missing prompt\"}\n", eid ? eid : "");
            fflush(stdout);
            free(eid); free(id);
            continue;
        }

        int prompt_tokens = 0, completion_tokens = 0;
        const char *finish = "stop";
        char *err = NULL;
        stdio_token_ctx cbctx = { id };
        char *content = run_text_generation(gguf_main, vocab, model, prompt, max_gen,
                                            &prompt_tokens, &completion_tokens, &finish,
                                            stream ? stdio_token_event : NULL, &cbctx, &err);
        char *eid = json_escape_dup(id ? id : "");
        if (!content) {
            char *ee = json_escape_dup(err ? err : "generation failed");
            printf("{\"id\":\"%s\",\"type\":\"error\",\"error\":\"%s\"}\n",
                   eid ? eid : "", ee ? ee : "");
            free(ee);
        } else {
            char *ec = json_escape_dup(content);
            char *ef = json_escape_dup(finish);
            printf("{\"id\":\"%s\",\"type\":\"final\",\"text\":\"%s\",\"finish_reason\":\"%s\","
                   "\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}\n",
                   eid ? eid : "", ec ? ec : "", ef ? ef : "stop",
                   prompt_tokens, completion_tokens, prompt_tokens + completion_tokens);
            free(ec); free(ef);
        }
        fflush(stdout);
        free(eid); free(content); free(err); free(prompt); free(id);
    }
    free(line);
    return 0;
}

/* ── usage ────────────────────────────────────────────────────────── */

static void usage(const char *p) {
    fprintf(stderr,
        "Usage: %s <model.gguf> [<mmproj.gguf>] [image] [options]\n"
        "Options:\n"
        "  --prompt \"...\"           (default: \"Explain the image\" / \"Hello\")\n"
        "  --max-gen N               (default 32)\n"
        "  --max-seq N               (KV cache size, default 1024)\n"
        "  --image-size S            (longer-side target, default 384)\n"
        "  --vit-dtype fp32|bf16|fp16 (default fp16)\n"
        "  --vit-threads N           (vision encoder threads)\n"
        "  --llm-threads N           (LLM forward threads; default 1)\n"
        "  --no-deepstack            (disable deepstack injection)\n"
        "  --mmap                    (file-backed weights; slower per matvec)\n"
        "  --kv-dtype f32|f16|q8     (KV cache element format; default f16)\n"
        "  --seed N                  (rng seed; default time-based)\n"
        "  --serve-stdio             (persistent JSONL control mode; text-only)\n",
        p);
}

/* ── main ─────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    /* NOTE: do NOT setenv OMP_WAIT_POLICY=active here. This runner mixes the
     * OMP-parallel VIT encoder with transformer.h's own pthread pool for the
     * LLM. With active wait + OMP_NUM_THREADS=48, the 48 OMP workers spin
     * forever after the encode finishes and starve the LLM pthread pool
     * (measured: LLM gen 55 → 0.66 tok/s). vlm_runner (VIT only) sets it
     * safely; here we leave the OMP wait policy at the env default. */

    if (argc < 2) { usage(argv[0]); return 1; }

    const char *model_path  = NULL;
    const char *mmproj_path = NULL;
    const char *image_path  = NULL;
    const char *user_prompt = NULL;
    int max_gen     = 32;
    int max_seq_len = 1024;
    int img_size    = 384;
    int vit_dtype   = VIT_DTYPE_FP16;
    int vit_threads = 0;
    int llm_threads = 1;
    int use_deepstack = 1;
    int use_mmap_main = 0;
    int serve_stdio = 0;
    unsigned seed   = (unsigned)time(NULL);

    /* Positionals: first .gguf = model, second .gguf = mmproj, first image = image */
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--prompt") && i + 1 < argc) {
            user_prompt = argv[++i];
        } else if (!strcmp(a, "--max-gen") && i + 1 < argc) {
            max_gen = atoi(argv[++i]);
        } else if (!strcmp(a, "--max-seq") && i + 1 < argc) {
            max_seq_len = atoi(argv[++i]);
        } else if (!strcmp(a, "--image-size") && i + 1 < argc) {
            img_size = atoi(argv[++i]);
        } else if (!strcmp(a, "--vit-dtype") && i + 1 < argc) {
            int d = parse_dtype(argv[++i]);
            if (d < 0) { fprintf(stderr, "unknown vit-dtype '%s'\n", argv[i]); return 1; }
            vit_dtype = d;
        } else if (!strcmp(a, "--vit-threads") && i + 1 < argc) {
            vit_threads = atoi(argv[++i]);
        } else if (!strcmp(a, "--llm-threads") && i + 1 < argc) {
            llm_threads = atoi(argv[++i]);
        } else if (!strcmp(a, "--no-deepstack")) {
            use_deepstack = 0;
        } else if (!strcmp(a, "--mmap")) {
            use_mmap_main = 1;
        } else if (!strcmp(a, "--serve-stdio")) {
            serve_stdio = 1;
        } else if (!strcmp(a, "--kv-dtype") && i + 1 < argc) {
            const char *kv = argv[++i];
            if (strcmp(kv, "f32") && strcmp(kv, "fp32") &&
                strcmp(kv, "f16") && strcmp(kv, "fp16") &&
                strcmp(kv, "q8")  && strcmp(kv, "int8")) {
                fprintf(stderr, "unknown kv-dtype '%s' (expected f32|f16|q8)\n", kv);
                return 1;
            }
            setenv("TF_KV_DTYPE", kv, 1);
        } else if (!strcmp(a, "--seed") && i + 1 < argc) {
            seed = (unsigned)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            usage(argv[0]); return 0;
        } else if (arg_is_image(a)) {
            image_path = a;
        } else if (arg_is_gguf(a)) {
            if (!model_path)        model_path = a;
            else if (!mmproj_path)  mmproj_path = a;
            else { fprintf(stderr, "too many .gguf positional args\n"); return 1; }
        } else {
            fprintf(stderr, "unknown arg: %s\n", a);
            usage(argv[0]); return 1;
        }
    }

    if (!model_path) {
        fprintf(stderr, "error: <model.gguf> required\n");
        usage(argv[0]); return 1;
    }
    if (!user_prompt) {
        user_prompt = mmproj_path ? "Explain the image" : "Hello";
    }
    srand(seed);

    fprintf(stderr, "llm_runner: model=%s\n", model_path);
    if (mmproj_path) {
        fprintf(stderr, "             mmproj=%s vit_dtype=%s vit_threads=%d image_size=%d\n",
                mmproj_path, dtype_name(vit_dtype), vit_threads, img_size);
    } else {
        fprintf(stderr, "             text-only mode\n");
    }
    fprintf(stderr, "             llm_threads=%d max_seq=%d max_gen=%d seed=%u\n",
            llm_threads, max_seq_len, max_gen, seed);

    /* ── load main model ── */
    /* use_mmap=0: read weights into 2MB-aligned anonymous memory (hugepage-
     * backed on Fugaku). mmap'd file-backed weights use 4KB pages and cause
     * catastrophic dTLB thrashing on A64FX — measured ~10x slower matvec and
     * no thread scaling. See a64fx/doc / M9 investigation. */
    double t = mono_sec();
    fprintf(stderr, "[1/6] load main GGUF: %s\n", model_path);
    /* --mmap forces use_mmap=1 (file-backed pages). Slower per matvec
     * (4KB pages → dTLB thrashing on A64FX) but the only option when
     * the weights exceed available anonymous RAM (e.g. 17GB BF16 9B
     * model on 31GB node). */
    /* gguf_open_multi loads llama.cpp split GGUFs (NNNNN-of-MMMMM) when given any
     * one shard; falls back to gguf_open() for single-file paths. Lets this
     * single-node runner load the 2-file 27B for TP-free isolation. */
    gguf_context *gguf_main = gguf_open_multi(model_path, use_mmap_main ? 1 : 0);
    if (!gguf_main) { fprintf(stderr, "failed to open main GGUF\n"); return 1; }

    bpe_vocab *vocab = bpe_vocab_load(gguf_main);
    if (!vocab) { fprintf(stderr, "failed to load vocab\n"); return 1; }
    fprintf(stderr, "      vocab: %d tokens\n", vocab->n_tokens);

    transformer_model *model = transformer_load(gguf_main, max_seq_len);
    if (!model) { fprintf(stderr, "transformer_load failed\n"); return 1; }
    if (llm_threads > 1) transformer_set_threads(model, llm_threads);
    /* Build A64FX panel layout after the (pinned) thread pool exists so each
     * panel's row blocks are first-touched onto the consuming core's CMG. */
    transformer_build_panels(model);
    fprintf(stderr, "      n_embd=%d n_layers=%d n_heads=%d n_vocab=%d (%.3f s)\n",
            model->n_embd, model->n_layers, model->n_heads, model->n_vocab,
            mono_sec() - t);

    if (serve_stdio) {
        if (mmproj_path || image_path) {
            fprintf(stderr, "--serve-stdio is text-only; ignoring mmproj/image arguments\n");
        }
        int rc = llm_serve_stdio(gguf_main, vocab, model, max_gen);
        transformer_free(model);
        bpe_vocab_free(vocab);
        gguf_close(gguf_main);
        return rc;
    }

    /* ── vision setup (optional) ── */
    vision_model    *vm    = NULL;
    gguf_context    *gguf_mm = NULL;
    vlm_pool        *pool  = NULL;
    struct vit_a64fx_cache *cache = NULL;
    float           *vision_embd = NULL;
    int              n_vision_tokens = 0;
    int              vit_embd_dim    = 0;  /* per-token stride (proj_dim*(1+n_ds)) */

    if (mmproj_path) {
        t = mono_sec();
        fprintf(stderr, "[2/6] load mmproj: %s\n", mmproj_path);
        gguf_mm = gguf_open(mmproj_path, 1);
        if (!gguf_mm) { fprintf(stderr, "failed to open mmproj\n"); return 1; }
        vm = vision_load(gguf_mm);
        if (!vm) { fprintf(stderr, "vision_load failed\n"); return 1; }
        fprintf(stderr, "      patch=%d merge=%d proj_dim=%d n_deepstack=%d (%.3f s)\n",
                vm->patch_size, vm->spatial_merge, vm->proj_dim, vm->n_deepstack,
                mono_sec() - t);

        /* ── image (load or synth) ── */
        int img_w = img_size, img_h = img_size;
        int grid  = vm->patch_size * vm->spatial_merge;
        uint8_t *img_rgb = NULL;
        if (image_path) {
            int sw = 0, sh = 0;
            uint8_t *src = img_load(image_path, &sw, &sh);
            if (!src) { fprintf(stderr, "failed to load image '%s'\n", image_path); return 1; }
            int long_side = (sw > sh) ? sw : sh;
            float s = (float)img_size / (float)long_side;
            int dw = (int)(sw * s); if (dw < grid) dw = grid;
            int dh = (int)(sh * s); if (dh < grid) dh = grid;
            dw = (dw / grid) * grid;
            dh = (dh / grid) * grid;
            fprintf(stderr, "      image %dx%d → %dx%d\n", sw, sh, dw, dh);
            img_rgb = img_resize_ac(src, sw, sh, dw, dh);
            free(src);
            img_w = dw; img_h = dh;
        } else {
            if (img_w % grid != 0) img_w = (img_w / grid) * grid;
            if (img_h % grid != 0) img_h = (img_h / grid) * grid;
            if (img_w < grid) img_w = grid;
            if (img_h < grid) img_h = grid;
            fprintf(stderr, "      synth checkerboard %dx%d\n", img_w, img_h);
            img_rgb = make_checkerboard(img_w, img_h, 64);
        }

        float *img_norm = vision_normalize_image(vm, img_rgb, img_w, img_h);
        free(img_rgb);

        /* ── vit pool + cache ── */
        pool = vlm_pool_init(vit_threads);
        if (!pool) { fprintf(stderr, "vlm_pool_init failed\n"); return 1; }
        fprintf(stderr, "[3/6] vit thread pool = %d\n", vlm_pool_size(pool));

        t = mono_sec();
        fprintf(stderr, "[4/6] build vit cache (dtype=%s)\n", dtype_name(vit_dtype));
        cache = vit_a64fx_cache_build(vm, vit_dtype);
        if (!cache) { fprintf(stderr, "vit_a64fx_cache_build failed\n"); return 1; }
        fprintf(stderr, "      cache built in %.3f s\n", mono_sec() - t);

        /* ── encode ── */
        vit_a64fx_opts opts = {
            .dtype       = vit_dtype,
            .pool        = pool,
            .dump        = NULL,
            .enable_prof = 0,
            .cache       = cache,
        };
        t = mono_sec();
        fprintf(stderr, "[5/6] encode image...\n");
        vision_embd = vit_a64fx_encode(vm, img_norm, img_w, img_h, &opts,
                                       &n_vision_tokens, &vit_embd_dim);
        double t_enc = mono_sec() - t;
        free(img_norm);
        if (!vision_embd) { fprintf(stderr, "vit_a64fx_encode failed\n"); return 1; }
        fprintf(stderr, "      tokens=%d embd_dim=%d (%.3f s, %.1f tok/s)\n",
                n_vision_tokens, vit_embd_dim, t_enc, n_vision_tokens / t_enc);

        /* Sanity: embd_dim should be proj_dim*(1+n_ds) */
        int expected_full = vm->proj_dim * (1 + vm->n_deepstack);
        if (vit_embd_dim != expected_full) {
            fprintf(stderr, "      warning: vit embd_dim=%d != proj_dim*(1+n_ds)=%d\n",
                    vit_embd_dim, expected_full);
        }

        /* DeepStack: tell the LLM the per-token stride so it picks up
         * ds_slice = embd + (1 + l)*n_embd for layers 0..n_deepstack-1. */
        if (use_deepstack && model->n_deepstack > 0 && vit_embd_dim > model->n_embd) {
            model->ds_embd_stride = vit_embd_dim;
            fprintf(stderr, "      deepstack: ds_embd_stride=%d (layers=%d)\n",
                    model->ds_embd_stride, model->n_deepstack);
        } else {
            model->ds_embd_stride = 0;
            fprintf(stderr, "      deepstack: disabled\n");
        }
    }

    /* ── build chat prompt ── */
    size_t prompt_cap = strlen(user_prompt) + 256;
    char *text_before = (char *)malloc(prompt_cap);
    char *text_after  = (char *)malloc(prompt_cap);
    build_chat_prompt(gguf_main, user_prompt, mmproj_path != NULL,
                      text_before, prompt_cap,
                      text_after,  prompt_cap);

    int tok_cap = max_seq_len > 256 ? max_seq_len : 256;
    int32_t *tokens_before = (int32_t *)malloc((size_t)tok_cap * sizeof(int32_t));
    int32_t *tokens_after  = (int32_t *)malloc((size_t)tok_cap * sizeof(int32_t));
    int n_before = bpe_tokenize(vocab, text_before, -1, tokens_before, tok_cap);
    int n_after  = text_after[0] ? bpe_tokenize(vocab, text_after, -1, tokens_after, tok_cap) : 0;

    int total_prompt = n_before + n_vision_tokens + n_after;
    fprintf(stderr, "[6/6] prompt: %d tokens (%d text + %d vision + %d text)\n",
            total_prompt, n_before, n_vision_tokens, n_after);

    if (total_prompt >= max_seq_len) {
        fprintf(stderr, "error: prompt %d >= max_seq_len %d (raise --max-seq)\n",
                total_prompt, max_seq_len);
        return 1;
    }

    /* ── prefill ── */
    fprintf(stderr, "\n=== Prefill ===\n");
    int pos = 0;
    double t_pre = mono_sec();
    int trace_per_tok = getenv("LLM_TRACE_PER_TOK") ? 1 : 0;
    float *logits = NULL;

    /* Stage-1 batched-GEMM prefill (TF_PREFILL_GEMM): batch the dense FFN as a GEMM
     * across the whole text prompt. Text-only (no interleaved vision) only. Writes the
     * KV cache identically to the per-token path; returns last-token logits. NULL =>
     * unsupported (no KV written yet) => fall through to the per-token loops below. */
    int gemm_prefill_done = 0;
    if (getenv("TF_PREFILL_GEMM") && n_vision_tokens == 0 && (n_before + n_after) > 0) {
        int M = n_before + n_after;
        int32_t *toks = (int32_t *)malloc((size_t)M * sizeof(int32_t));
        if (toks) {
            memcpy(toks, tokens_before, (size_t)n_before * sizeof(int32_t));
            memcpy(toks + n_before, tokens_after, (size_t)n_after * sizeof(int32_t));
            logits = transformer_prefill_gemm(model, toks, M, 0);
            free(toks);
            if (logits) { pos = M; gemm_prefill_done = 1; }
        }
        if (!gemm_prefill_done)
            fprintf(stderr, "  TF_PREFILL_GEMM: unsupported config, using per-token prefill\n");
    }

    /* text before */
    for (int i = 0; !gemm_prefill_done && i < n_before; i++) {
        double tt0 = trace_per_tok ? mono_sec() : 0.0;
        transformer_forward(model, tokens_before[i], pos);
        if (trace_per_tok) {
            double tt1 = mono_sec();
            fprintf(stderr, "  pre[%d] tok=%d  %.3f s\n", pos, tokens_before[i], tt1 - tt0);
        }
        pos++;
    }

    /* vision */
    if (n_vision_tokens > 0) {
        double tv0 = mono_sec();
        for (int i = 0; i < n_vision_tokens; i++) {
            const float *embd_i = vision_embd + (size_t)i * vit_embd_dim;
            transformer_forward_embd(model, embd_i, pos);
            pos++;
        }
        double tv1 = mono_sec();
        fprintf(stderr, "  vision prefill: %.2f s (%.2f tok/s)\n",
                tv1 - tv0, n_vision_tokens / (tv1 - tv0));
    }
    /* Make sure ds_embd doesn't leak into the post-vision text tokens.
     * transformer_forward_embd_pos already nulls ds_embd before returning,
     * but transformer_forward goes through transformer_forward_pos which
     * doesn't touch ds_embd. We rely on the *_embd variant having reset
     * it; nothing to do here. */

    /* text after (last token returns logits) */
    for (int i = 0; !gemm_prefill_done && i < n_after; i++) {
        double tt0 = trace_per_tok ? mono_sec() : 0.0;
        if (i == n_after - 1) {
            logits = transformer_forward_logits(model, tokens_after[i], pos);
        } else {
            transformer_forward(model, tokens_after[i], pos);
        }
        if (trace_per_tok) {
            double tt1 = mono_sec();
            fprintf(stderr, "  pre[%d] tok=%d  %.3f s%s\n", pos, tokens_after[i], tt1 - tt0,
                    (i == n_after - 1) ? " (logits)" : "");
        }
        pos++;
    }
    /* If text-only and there was no after-text (shouldn't happen), still need logits */
    if (!gemm_prefill_done && !logits && n_before > 0 && n_after == 0) {
        /* re-run last text token through logits */
        logits = transformer_forward_logits(model, tokens_before[n_before - 1], pos - 1);
    }
    double t_pre_end = mono_sec();
    fprintf(stderr, "prefill total: %.2f s (%d tokens, %.1f tok/s)\n",
            t_pre_end - t_pre, pos, pos / (t_pre_end - t_pre));

    /* ── generation ── */
    /* After prefill: pos == total_prompt; `logits` is the prediction for
     * position `total_prompt`. The first generated token therefore lives
     * at position `total_prompt`, the next at `total_prompt+1`, etc. */
    fprintf(stderr, "\n=== Generation ===\n");
    fflush(stderr);
    transformer_pool_profile_reset(); /* steady-state decode profile only */
    double t_gen0 = mono_sec();
    int n_gen = 0;
    int32_t next = -1;

    for (int g = 0; g < max_gen; g++) {
        if (g > 0) {
            /* forward(next, pos) consumes the previous gen token at slot pos */
            logits = transformer_forward_logits(model, next, pos);
            pos++;
        }
        if (!logits) { fprintf(stderr, "\n[no logits]\n"); break; }

        /* greedy argmax */
        next = 0;
        for (int j = 1; j < model->n_vocab; j++) {
            if (logits[j] > logits[next]) next = j;
        }

        dump_logit_stats(logits, model->n_vocab, g, next);

        if (next == vocab->eos_id || next == vocab->eot_id) {
            fprintf(stderr, "\n[eos %d]\n", next);
            if (!getenv("TF_NO_EOS_STOP")) break;
        }

        const char *tok_str = bpe_token_to_str(vocab, next);
        if (tok_str) {
            int dec_len = 0;
            char *decoded = bpe_byte_decode(tok_str, (int)strlen(tok_str), &dec_len);
            if (decoded) {
                fwrite(decoded, 1, dec_len, stdout);
                fflush(stdout);
                free(decoded);
            }
        }
        n_gen++;
    }
    printf("\n");
    double t_gen1 = mono_sec();
    if (n_gen > 0) {
        fprintf(stderr, "gen: %d tokens in %.2f s (%.2f tok/s)\n",
                n_gen, t_gen1 - t_gen0, n_gen / (t_gen1 - t_gen0));
    }

    prof_summary();

    /* ── cleanup ── */
    if (vision_embd) free(vision_embd);
    if (cache)       vit_a64fx_cache_free(cache);
    if (pool)        vlm_pool_free(pool);
    if (vm)          vision_free(vm);
    if (gguf_mm)     gguf_close(gguf_mm);
    transformer_free(model);
    bpe_vocab_free(vocab);
    gguf_close(gguf_main);
    return 0;
}
