/*
 * test_hip_llm.c - Test harness for HIP LLM runner
 *
 * Loads a GGUF model, runs both CPU reference and HIP side-by-side,
 * compares hidden states per token.
 *
 * Usage: ./test_hip_llm [model.gguf] [-t "prompt text"] [-n max_tokens]
 *
 * Compile with gcc (no hipcc needed).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>
#include <signal.h>
#include <unistd.h>

/* GGUF loader */
#define GGUF_LOADER_IMPLEMENTATION
#include "../../common/gguf_loader.h"

/* Dequant (needed by transformer.h) */
#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"

/* CPU reference transformer */
#define TRANSFORMER_IMPLEMENTATION
#include "../../common/transformer.h"

/* BPE tokenizer */
#define BPE_TOKENIZER_IMPLEMENTATION
#include "../../common/bpe_tokenizer.h"

/* HIP LLM runner */
#include "hip_llm_runner.h"
#include "reference_sampler.h"
#include "generation_trace.h"

/* ---- Comparison helpers ---- */

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static float vec_norm(const float *v, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += v[i] * v[i];
    return sqrtf(s);
}

static float rel_l2_error(const float *a, const float *b, int n) {
    float diff_sq = 0.0f, ref_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = a[i] - b[i];
        diff_sq += d * d;
        ref_sq += b[i] * b[i];
    }
    if (ref_sq < 1e-12f) return sqrtf(diff_sq);
    return sqrtf(diff_sq / ref_sq);
}

/* Qwen3.8-Next non-thinking/Instruct defaults from the model card.  Keeping
 * only top-k candidates makes sampling O(vocab*k), with no full-vocab sort. */
static int sample_top_k_p(const float *logits, int n, int top_k, float top_p,
                          float temperature, float presence_penalty,
                          float repetition_penalty, float min_p,
                          const unsigned char *seen,
                          unsigned *rng) {
    if (top_k < 1) top_k = 1;
    if (top_k > 64) top_k = 64;
    int ids[64];
    float vals[64];
    for (int j = 0; j < top_k; ++j) { ids[j] = -1; vals[j] = -INFINITY; }
    for (int i = 0; i < n; ++i) {
        /* Match the API sampling contract exactly. A neutral repetition
         * penalty must not turn into a hidden frequency penalty or n-gram
         * ban, both of which distort ordinary code identifiers. */
        float v = logits[i] - ((seen && seen[i]) ? presence_penalty : 0.0f);
        if (seen && seen[i] && repetition_penalty != 1.0f)
            v = v < 0.0f ? v * repetition_penalty : v / repetition_penalty;
        if (!isfinite(v)) continue;
        if (v <= vals[top_k - 1]) continue;
        int j = top_k - 1;
        while (j > 0 && v > vals[j - 1]) {
            vals[j] = vals[j - 1]; ids[j] = ids[j - 1]; --j;
        }
        vals[j] = v; ids[j] = i;
    }
    if (ids[0] < 0) return 0;
    while (top_k > 1 && ids[top_k - 1] < 0) --top_k;
    if (temperature <= 0.0f || top_k == 1) return ids[0];
    if (min_p > 0.0f) {
        float cutoff = vals[0] + logf(min_p);
        while (top_k > 1 && vals[top_k - 1] < cutoff) --top_k;
    }
    float sum = 0.0f;
    const float max_logit = vals[0];
    for (int j = 0; j < top_k; ++j) {
        vals[j] = expf((vals[j] - max_logit) / temperature);
        sum += vals[j];
    }
    float keep = 0.0f;
    int nkeep = 0;
    do { keep += vals[nkeep++]; } while (nkeep < top_k && keep < top_p * sum);
    *rng ^= *rng << 13; *rng ^= *rng >> 17; *rng ^= *rng << 5;
    float pick = ((float)(*rng & 0x00ffffffu) / 16777216.0f) * keep;
    for (int j = 0; j < nkeep; ++j) {
        pick -= vals[j];
        if (pick <= 0.0f) return ids[j];
    }
    return ids[nkeep - 1];
}

static int argmax_logits(const float *logits, int n);

/* Vocabulary strings use GPT-2/Qwen's byte-to-Unicode display alphabet.
 * Decode every piece before it reaches a user-visible stream so spaces,
 * newlines, arbitrary UTF-8, and byte-fallback tokens are emitted as bytes
 * rather than tokenizer markers such as U+0120/U+010A (Ġ/Ċ). */
static void print_decoded_token(FILE *stream, const bpe_vocab *vocab, int token_id) {
    const char *piece = bpe_token_to_str(vocab, token_id);
    if (!stream || !piece) return;
    int decoded_len = 0;
    char *decoded = bpe_byte_decode(piece, (int)strlen(piece), &decoded_len);
    if (!decoded) return;
    if (decoded_len > 0)
        fwrite(decoded, 1, (size_t)decoded_len, stream);
    free(decoded);
}

static int is_generation_stop(const bpe_vocab *vocab, int token_id,
                              int eos, int eot) {
    if (token_id == eos || token_id == eot) return 1;
    /* Some converted vocabularies contain duplicate control-token strings.
     * Match the piece as well as the metadata ID so benchmark text never
     * exposes a duplicate ChatML turn terminator. */
    const char *piece = bpe_token_to_str(vocab, token_id);
    return piece && (strcmp(piece, "<|im_end|>") == 0 ||
                     strcmp(piece, "<|endoftext|>") == 0);
}

/* Coding-only sampler variant. Keep the production sampler above with its
 * small standalone signature because test_sampler.py extracts it directly;
 * delimiter filtering belongs in this opt-in wrapper instead. */
static int sample_top_k_p_coding(const float *logits, int n, int top_k, float top_p,
                                 float temperature, float presence_penalty,
                                 float repetition_penalty, float min_p,
                                 const unsigned char *seen, unsigned *rng,
                                 const bpe_vocab *vocab) {
    if (top_k < 1) top_k = 1;
    if (top_k > 64) top_k = 64;
    int ids[64]; float vals[64];
    for (int j = 0; j < top_k; ++j) { ids[j] = -1; vals[j] = -INFINITY; }
    for (int i = 0; i < n; ++i) {
        const char *piece = vocab ? bpe_token_to_str(vocab, i) : NULL;
        if (piece && piece[0] == '<' && piece[1] == '|') continue;
        float v = logits[i] - ((seen && seen[i]) ? presence_penalty : 0.0f);
        if (seen && seen[i] && repetition_penalty != 1.0f)
            v = v < 0.0f ? v * repetition_penalty : v / repetition_penalty;
        if (!isfinite(v) || v <= vals[top_k - 1]) continue;
        int j = top_k - 1;
        while (j > 0 && v > vals[j - 1]) {
            vals[j] = vals[j - 1]; ids[j] = ids[j - 1]; --j;
        }
        vals[j] = v; ids[j] = i;
    }
    if (ids[0] < 0) return 0;
    while (top_k > 1 && ids[top_k - 1] < 0) --top_k;
    if (temperature <= 0.0f || top_k == 1) return ids[0];
    if (min_p > 0.0f) {
        float cutoff = vals[0] + logf(min_p);
        while (top_k > 1 && vals[top_k - 1] < cutoff) --top_k;
    }
    float sum = 0.0f, max_logit = vals[0];
    for (int j = 0; j < top_k; ++j) { vals[j] = expf((vals[j] - max_logit) / temperature); sum += vals[j]; }
    float keep = 0.0f; int nkeep = 0;
    do { keep += vals[nkeep++]; } while (nkeep < top_k && keep < top_p * sum);
    *rng ^= *rng << 13; *rng ^= *rng >> 17; *rng ^= *rng << 5;
    float pick = ((float)(*rng & 0x00ffffffu) / 16777216.0f) * keep;
    for (int j = 0; j < nkeep; ++j) { pick -= vals[j]; if (pick <= 0.0f) return ids[j]; }
    return ids[nkeep - 1];
}

static volatile sig_atomic_t g_stdio_cancel;

static void stdio_cancel_handler(int signo) {
    (void)signo;
    g_stdio_cancel = 1;
}

/* Small line protocol used by codex_server.py.  Keeping HTTP/JSON out of the
 * GPU process makes the runner easy to embed and, more importantly, keeps one
 * HIP context alive for the lifetime of the API server. */
static const char b64_chars[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static char *b64_encode(const unsigned char *src, size_t n, size_t *out_n) {
    size_t cap = ((n + 2) / 3) * 4 + 1;
    char *out = (char *)malloc(cap);
    if (!out) return NULL;
    size_t p = 0;
    for (size_t i = 0; i < n; i += 3) {
        unsigned v = (unsigned)src[i] << 16;
        if (i + 1 < n) v |= (unsigned)src[i + 1] << 8;
        if (i + 2 < n) v |= src[i + 2];
        out[p++] = b64_chars[(v >> 18) & 63];
        out[p++] = b64_chars[(v >> 12) & 63];
        out[p++] = i + 1 < n ? b64_chars[(v >> 6) & 63] : '=';
        out[p++] = i + 2 < n ? b64_chars[v & 63] : '=';
    }
    out[p] = '\0';
    if (out_n) *out_n = p;
    return out;
}

static int b64_value(int c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

static unsigned char *b64_decode(const char *src, size_t *out_n) {
    size_t n = strlen(src), cap, p = 0;
    if (n % 4 != 0) return NULL;
    cap = (n / 4) * 3 + 1;
    unsigned char *out = (unsigned char *)malloc(cap);
    if (!out) return NULL;
    for (size_t i = 0; i < n; i += 4) {
        int a = b64_value((unsigned char)src[i]);
        int b = b64_value((unsigned char)src[i + 1]);
        int c = src[i + 2] == '=' ? 0 : b64_value((unsigned char)src[i + 2]);
        int d = src[i + 3] == '=' ? 0 : b64_value((unsigned char)src[i + 3]);
        int last = i + 4 == n;
        if (a < 0 || b < 0 || c < 0 || d < 0 ||
            (!last && (src[i + 2] == '=' || src[i + 3] == '=')) ||
            (src[i + 2] == '=' && src[i + 3] != '=')) {
            free(out); return NULL;
        }
        unsigned v = ((unsigned)a << 18) | ((unsigned)b << 12) |
                     ((unsigned)c << 6) | (unsigned)d;
        out[p++] = (unsigned char)(v >> 16);
        if (src[i + 2] != '=') out[p++] = (unsigned char)(v >> 8);
        if (src[i + 3] != '=') out[p++] = (unsigned char)v;
    }
    out[p] = 0;
    if (out_n) *out_n = p;
    return out;
}

/* A BOS id alone does not mean it should be inserted (Qwen uses the same
 * id for padding). Preserve the explicit diagnostic override. */
static int prompt_bos_id(const gguf_context *gguf) {
    const char *override = getenv("LLM_ADD_BOS");
    if (override) return atoi(override);
    int add = gguf_find_key(gguf, "tokenizer.ggml.add_bos_token");
    /* Match llama.cpp: absent add_bos_token defaults to false.  A BOS ID by
     * itself is metadata, not an instruction to prepend it. */
    if (add < 0 || gguf->kv[add].type != GGUF_TYPE_BOOL || !gguf->kv[add].value.b)
        return -1;
    int id = gguf_find_key(gguf, "tokenizer.ggml.bos_token_id");
    return id >= 0 && gguf->kv[id].type == GGUF_TYPE_UINT32 ?
        (int)gguf->kv[id].value.u32 : -1;
}

typedef struct {
    int32_t *tokens;
    int n_tokens;
    char *identity;
    hip_llm_state_snapshot *snapshot;
    size_t bytes;
    uint64_t age;
    int portable;
} stdio_snapshot_entry;

typedef struct {
    stdio_snapshot_entry *entries;
    int capacity;
    size_t byte_limit;
    size_t bytes;
    uint64_t clock;
} stdio_snapshot_cache;

static void stdio_snapshot_entry_clear(stdio_snapshot_cache *cache, int i) {
    stdio_snapshot_entry *entry = &cache->entries[i];
    if (entry->snapshot) {
        if (cache->bytes >= entry->bytes) cache->bytes -= entry->bytes;
        else cache->bytes = 0;
    }
    free(entry->tokens);
    free(entry->identity);
    hip_llm_free_state_snapshot(entry->snapshot);
    memset(entry, 0, sizeof(*entry));
}

static int stdio_snapshot_cache_init(stdio_snapshot_cache *cache, int capacity,
                                     size_t byte_limit) {
    memset(cache, 0, sizeof(*cache));
    if (capacity < 0) capacity = 0;
    cache->capacity = capacity;
    cache->byte_limit = byte_limit;
    if (capacity == 0 || byte_limit == 0) return 0;
    cache->entries = calloc((size_t)capacity, sizeof(*cache->entries));
    return cache->entries ? 0 : -1;
}

static void stdio_snapshot_cache_free(stdio_snapshot_cache *cache) {
    if (!cache) return;
    for (int i = 0; i < cache->capacity; ++i)
        stdio_snapshot_entry_clear(cache, i);
    free(cache->entries);
    memset(cache, 0, sizeof(*cache));
}

static void stdio_snapshot_cache_drop_resident(stdio_snapshot_cache *cache) {
    if (!cache || !cache->entries) return;
    for (int i = 0; i < cache->capacity; ++i)
        if (cache->entries[i].snapshot && !cache->entries[i].portable)
            stdio_snapshot_entry_clear(cache, i);
}

static int stdio_snapshot_cache_find(const stdio_snapshot_cache *cache,
                                     const char *identity,
                                     const int32_t *tokens, int n_tokens,
                                     int allow_resident) {
    int best = -1;
    for (int i = 0; i < cache->capacity; ++i) {
        const stdio_snapshot_entry *entry = &cache->entries[i];
        if (!entry->snapshot || entry->n_tokens > n_tokens ||
            (!entry->portable && !allow_resident) ||
            strcmp(entry->identity, identity) != 0) continue;
        if (memcmp(entry->tokens, tokens,
                   (size_t)entry->n_tokens * sizeof(*tokens)) != 0) continue;
        if (best < 0 || entry->n_tokens > cache->entries[best].n_tokens)
            best = i;
    }
    return best;
}

/* Takes ownership of snapshot whether it is retained or rejected. */
static int stdio_snapshot_cache_publish(stdio_snapshot_cache *cache,
                                        const char *identity,
                                        const int32_t *tokens, int n_tokens,
                                        hip_llm_state_snapshot *snapshot) {
    if (!snapshot) return 0;
    size_t state_bytes = hip_llm_state_snapshot_bytes(snapshot);
    size_t bytes = state_bytes + (size_t)n_tokens * sizeof(*tokens) +
                   strlen(identity) + 1;
    int snapshot_tokens = hip_llm_state_snapshot_token_count(snapshot);
    int portable = hip_llm_state_snapshot_is_portable(snapshot);
    if (snapshot_tokens != n_tokens || !cache->entries ||
        n_tokens <= 0 || bytes > cache->byte_limit) {
        fprintf(stderr,
                "llm_server: context snapshot rejected tokens=%d state_tokens=%d portable=%d "
                "bytes=%.1f MiB limit=%.1f MiB\n",
                n_tokens, snapshot_tokens,
                hip_llm_state_snapshot_is_portable(snapshot),
                bytes / (double)(1ULL << 20),
                cache->byte_limit / (double)(1ULL << 20));
        hip_llm_free_state_snapshot(snapshot);
        return 0;
    }
    int32_t *token_copy = malloc((size_t)n_tokens * sizeof(*tokens));
    char *identity_copy = strdup(identity);
    if (!token_copy || !identity_copy) {
        free(token_copy); free(identity_copy);
        hip_llm_free_state_snapshot(snapshot);
        return -1;
    }
    memcpy(token_copy, tokens, (size_t)n_tokens * sizeof(*tokens));
    int slot = -1;
    for (int i = 0; i < cache->capacity; ++i) {
        stdio_snapshot_entry *entry = &cache->entries[i];
        if (entry->snapshot && entry->n_tokens == n_tokens &&
            strcmp(entry->identity, identity) == 0 &&
            memcmp(entry->tokens, tokens,
                   (size_t)n_tokens * sizeof(*tokens)) == 0) {
            slot = i;
            break;
        }
        if (!entry->snapshot && slot < 0) slot = i;
    }
    size_t replaced = slot >= 0 && cache->entries[slot].snapshot ?
                      cache->entries[slot].bytes : 0;
    while (cache->bytes - replaced + bytes > cache->byte_limit || slot < 0) {
        int oldest = -1;
        for (int i = 0; i < cache->capacity; ++i)
            if (i != slot && cache->entries[i].snapshot &&
                (oldest < 0 || cache->entries[i].age < cache->entries[oldest].age))
                oldest = i;
        if (oldest < 0) break;
        stdio_snapshot_entry_clear(cache, oldest);
        if (slot < 0) slot = oldest;
    }
    replaced = slot >= 0 && cache->entries[slot].snapshot ?
               cache->entries[slot].bytes : 0;
    if (slot < 0 || cache->bytes - replaced + bytes > cache->byte_limit) {
        free(token_copy); free(identity_copy);
        hip_llm_free_state_snapshot(snapshot);
        return 0;
    }
    stdio_snapshot_entry_clear(cache, slot);
    cache->entries[slot] = (stdio_snapshot_entry) {
        .tokens = token_copy, .n_tokens = n_tokens,
        .identity = identity_copy, .snapshot = snapshot,
        .bytes = bytes, .age = ++cache->clock,
        .portable = portable,
    };
    cache->bytes += bytes;
    fprintf(stderr,
            "llm_server: context snapshot committed slot=%d kind=%s tokens=%d "
            "bytes=%.1f MiB cache=%.1f MiB\n",
            slot, portable ? "portable" : "resident", n_tokens,
            bytes / (double)(1ULL << 20),
            cache->bytes / (double)(1ULL << 20));
    return 1;
}

static int run_stdio_server(hip_llm_runner *gpu, bpe_vocab *vocab,
                            int n_vocab, int max_seq_len, int bos_id, int mtp_draft,
                            int dense_mtp_draft, int dense_mtp_window,
                            int dflash_draft,
                            int coding_mode, const hllm_sampler_config *sampling_defaults,
                            int context_cache_entries, size_t context_cache_bytes) {
    char line[4 * 1024 * 1024];
    int32_t *cache = (int32_t *)malloc((size_t)max_seq_len * sizeof(int32_t));
    int cache_n = 0;
    char active_identity[513] = "";
    stdio_snapshot_cache snapshot_cache;
    unsigned rng = 0x51f15e5du;
    if (!cache || stdio_snapshot_cache_init(&snapshot_cache,
            context_cache_entries, context_cache_bytes) != 0) {
        free(cache); return 1;
    }
    signal(SIGUSR1, stdio_cancel_handler);
    fprintf(stderr, "JSONL backend ready (max_seq_len=%d)\n", max_seq_len);
    fflush(stderr);
    puts("READY");
    fflush(stdout);
    while (fgets(line, sizeof(line), stdin)) {
        g_stdio_cancel = 0;
        size_t line_len = strlen(line);
        if (line_len == sizeof(line) - 1 && line[line_len - 1] != '\n') {
            int ch;
            while ((ch = fgetc(stdin)) != '\n' && ch != EOF) {}
            puts("ERR request too large");
            fflush(stdout);
            continue;
        }
        int max_tokens = 16, top_k = 20;
        float temperature = 0.2f, top_p = 0.95f, presence = 0.0f;
        float repetition = 1.0f, min_p = 0.0f;
        char *b64 = NULL, *prefix_b64 = NULL;
        const char *cache_identity = "shared";
        hllm_sampler_config request_sampling;
        hllm_sampler_defaults(&request_sampling);
        if (sampling_defaults) request_sampling = *sampling_defaults;
        int use_reference = sampling_defaults != NULL;
        int version3 = !strncmp(line, "REQ3 ", 5);
        int version2 = !strncmp(line, "REQ2 ", 5);
        static char b64buf[sizeof(line)];
        static char prefix_b64buf[sizeof(line)];
        static char cache_identity_buf[513];
        int fields = 0;
        if (version3) {
            char seed_text[32], extra, *end = NULL;
            int n = sscanf(line + 5, "%512s %31s %d %f %f %d %f %f %f %f %d %4194303s %4194303s %c",
                cache_identity_buf, seed_text, &max_tokens, &temperature, &top_p,
                &top_k, &presence, &repetition, &min_p,
                &request_sampling.frequency, &request_sampling.penalty_last_n,
                prefix_b64buf, b64buf, &extra);
            if (n != 13 || strlen(cache_identity_buf) > 512) {
                puts("ERR invalid REQ3"); fflush(stdout); continue;
            }
            cache_identity = cache_identity_buf;
            if (strcmp(seed_text, "-") != 0) {
                unsigned long seed = strtoul(seed_text, &end, 10);
                if (seed_text[0] == '-' || !end || *end || seed >= UINT32_MAX) {
                    puts("ERR invalid REQ3"); fflush(stdout); continue;
                }
                request_sampling.seed = seed;
                use_reference = 1;
            }
            fields = 9;
        } else if (version2) {
            char seed_text[32], extra, *end = NULL;
            int n = sscanf(line + 5, "%31s %d %f %f %d %f %f %f %f %d %4194303s %4194303s %c",
                seed_text, &max_tokens, &temperature, &top_p, &top_k, &presence,
                &repetition, &min_p, &request_sampling.frequency,
                &request_sampling.penalty_last_n, prefix_b64buf, b64buf, &extra);
            unsigned long seed = n > 0 ? strtoul(seed_text, &end, 10) : UINT32_MAX;
            if (n != 12 || seed_text[0] == '-' || !end || *end || seed >= UINT32_MAX) {
                puts("ERR invalid REQ2"); fflush(stdout); continue;
            }
            request_sampling.seed = seed;
            fields = 9;
            use_reference = 1;
        } else {
        if (strncmp(line, "REQ ", 4) != 0 ||
            sscanf(line + 4, "%d %f %f %d %f ", &max_tokens, &temperature,
                   &top_p, &top_k, &presence) != 5) {
            puts("ERR invalid request"); fflush(stdout); continue;
        }
        fields = sscanf(line + 4, "%d %f %f %d %f %f %f %4194303s %4194303s", &max_tokens,
                            &temperature, &top_p, &top_k, &presence,
                            &repetition, &min_p, prefix_b64buf, b64buf);
        if (fields != 9) {
            min_p = 0.0f;
            repetition = 1.0f;
            fields = sscanf(line + 4, "%d %f %f %d %f %4194303s %4194303s", &max_tokens,
                            &temperature, &top_p, &top_k, &presence,
                            prefix_b64buf, b64buf);
        }
        if (fields != 9 && fields != 7) {
            fields = sscanf(line + 4, "%d %f %f %d %f %4194303s", &max_tokens,
                            &temperature, &top_p, &top_k, &presence, b64buf);
        }
        if (fields != 9 && fields != 7 && fields != 6) {
            puts("ERR missing prompt"); fflush(stdout); continue;
        }
        }
        if (use_reference && (coding_mode || mtp_draft > 0)) {
            puts("ERR reference sampling incompatible with coding/Qwen4 MTP"); fflush(stdout); continue;
        }
        if (max_tokens < 0 || (!use_reference && top_k < 1) ||
            !isfinite(temperature) || temperature < 0.0f ||
            !isfinite(top_p) || top_p < 0.0f || top_p > 1.0f ||
            !isfinite(presence) || !isfinite(repetition) || repetition <= 0.0f ||
            !isfinite(min_p) ||
            min_p < 0.0f || min_p > 1.0f) {
            puts("ERR invalid sampling"); fflush(stdout); continue;
        }
        request_sampling.top_k = top_k;
        request_sampling.top_p = top_p;
        request_sampling.min_p = min_p;
        request_sampling.temperature = temperature;
        request_sampling.repetition = repetition;
        request_sampling.presence = presence;
        if (!isfinite(request_sampling.frequency) || request_sampling.penalty_last_n < 0) {
            puts("ERR invalid sampling"); fflush(stdout); continue;
        }
        prefix_b64 = fields >= 7 ? prefix_b64buf : NULL;
        b64 = b64buf;
        size_t prompt_n = 0;
        unsigned char *prompt = b64_decode(b64, &prompt_n);
        if (!prompt) { puts("ERR bad base64"); fflush(stdout); continue; }
        size_t prefix_n_bytes = 0;
        unsigned char *prefix = (prefix_b64 && strcmp(prefix_b64, "-") != 0) ?
            b64_decode(prefix_b64, &prefix_n_bytes) : NULL;
        if (prefix_b64 && strcmp(prefix_b64, "-") != 0 && !prefix) {
            free(prompt); puts("ERR bad prefix base64"); fflush(stdout); continue;
        }

        int cap = max_seq_len > 0 ? max_seq_len : 512;
        int32_t *tokens = (int32_t *)malloc((size_t)cap * sizeof(int32_t));
        int n_tokens = tokens ? bpe_tokenize(vocab, (const char *)prompt,
                                              (int)prompt_n, tokens, cap) : -1;
        free(prompt);
        if (!tokens || n_tokens <= 0) {
            free(prefix); free(tokens); puts("ERR tokenization"); fflush(stdout); continue;
        }
        if (n_tokens > cap || (bos_id > 0 && tokens[0] != bos_id && n_tokens == cap)) {
            free(prefix); free(tokens);
            puts("ERR prompt exceeds context capacity"); fflush(stdout); continue;
        }
        if (bos_id > 0 && n_tokens < cap && (n_tokens == 0 || tokens[0] != bos_id)) {
            memmove(tokens + 1, tokens, (size_t)n_tokens * sizeof(int32_t));
            tokens[0] = bos_id;
            n_tokens++;
        }
        int requested_prefix = 0;
        int32_t *prefix_tokens = prefix ? (int32_t *)malloc((size_t)cap * sizeof(int32_t)) : NULL;
        if (prefix && prefix_n_bytes > 0 && prefix_tokens) {
            requested_prefix = bpe_tokenize(vocab, (const char *)prefix,
                                             (int)prefix_n_bytes, prefix_tokens, cap);
            if (requested_prefix > cap || requested_prefix < 0)
                requested_prefix = 0;
            if (requested_prefix > 0 && bos_id > 0 && requested_prefix < cap && prefix_tokens[0] != bos_id) {
                memmove(prefix_tokens + 1, prefix_tokens, (size_t)requested_prefix * sizeof(int32_t));
                prefix_tokens[0] = bos_id;
                requested_prefix++;
            }
            for (int i = 0; i < requested_prefix && i < n_tokens; ++i)
                if (prefix_tokens[i] != tokens[i]) { requested_prefix = 0; break; }
            if (requested_prefix > n_tokens) requested_prefix = 0;
        }
        free(prefix_tokens); free(prefix);
        int common = 0;
        if (strcmp(active_identity, cache_identity) == 0)
            while (common < cache_n && common < n_tokens &&
                   cache[common] == tokens[common]) common++;
        int have_state = cache_n > 0 && common == cache_n;
        int prompt_snapshot_present = 0;
        if (have_state && common == n_tokens) {
            int exact = stdio_snapshot_cache_find(&snapshot_cache,
                cache_identity, tokens, n_tokens, 1);
            if (exact >= 0 && snapshot_cache.entries[exact].n_tokens == n_tokens) {
                snapshot_cache.entries[exact].age = ++snapshot_cache.clock;
                prompt_snapshot_present = 1;
            }
        }
        if (!have_state) {
            int allow_resident = strcmp(active_identity, cache_identity) == 0;
            int hit = stdio_snapshot_cache_find(&snapshot_cache, cache_identity,
                                                 tokens, n_tokens,
                                                 allow_resident);
            if (hit >= 0 && snapshot_cache.entries[hit].portable)
                stdio_snapshot_cache_drop_resident(&snapshot_cache);
            int restore_rc = hit >= 0 ? hip_llm_restore_state(
                gpu, snapshot_cache.entries[hit].snapshot) : -1;
            if (hit >= 0 && restore_rc == 0) {
                stdio_snapshot_entry *entry = &snapshot_cache.entries[hit];
                memcpy(cache, entry->tokens,
                       (size_t)entry->n_tokens * sizeof(*cache));
                cache_n = entry->n_tokens;
                common = entry->n_tokens;
                entry->age = ++snapshot_cache.clock;
                snprintf(active_identity, sizeof(active_identity), "%s",
                         cache_identity);
                fprintf(stderr,
                        "llm_server: context snapshot restored slot=%d tokens=%d\n",
                        hit, common);
                prompt_snapshot_present = common == n_tokens;
                have_state = 1;
            } else if (hit >= 0) {
                fprintf(stderr,
                        "llm_server: context snapshot restore failed slot=%d rc=%d\n",
                        hit, restore_rc);
                /* A failed restore may have partially written device state.
                 * Reset below and evict the unusable snapshot so future
                 * requests do not loop on the same corrupt entry. */
                stdio_snapshot_entry_clear(&snapshot_cache, hit);
            } else {
                int entries = 0;
                for (int i = 0; i < snapshot_cache.capacity; ++i)
                    entries += snapshot_cache.entries[i].snapshot != NULL;
                fprintf(stderr,
                        "llm_server: context snapshot miss tokens=%d entries=%d "
                        "identity=%.8s\n",
                        n_tokens, entries, cache_identity);
            }
        }
        if (!have_state) {
            stdio_snapshot_cache_drop_resident(&snapshot_cache);
            hip_llm_reset_state(gpu);
            cache_n = 0;
            common = 0;
            active_identity[0] = '\0';
        }
        /* Dense NextN owns request-local draft KV. It does not consume prompt
         * tokens, so even an append to a live target prefix must begin a new
         * draft sequence from the completed prompt boundary. */
        if (dense_mtp_draft > 0)
            hip_llm_qwen35_mtp_reset(gpu);
        if (n_tokens > max_seq_len) n_tokens = max_seq_len;
        hip_llm_set_qwen4_batch_request_tokens(gpu, n_tokens);
        int batch_size = 128;
        const char *batch_env = getenv("LLM_BMAX");
        if (batch_env) batch_size = atoi(batch_env);
        if (batch_size < 1) batch_size = 1;
        const char *publish_chunk_env = getenv("LLM_QWEN4_PREFILL_COPY_PIPELINE_PUBLISH_CHUNK");
        int publish_chunk = publish_chunk_env && atoi(publish_chunk_env) != 0;
        int prompt_added = n_tokens - common;
        int batches = prompt_added > 0 ? (prompt_added + batch_size - 1) / batch_size : 0;
        if (requested_prefix > common && requested_prefix < n_tokens) {
            int a = requested_prefix - common;
            int b = n_tokens - requested_prefix;
            batches = (a + batch_size - 1) / batch_size +
                      (b + batch_size - 1) / batch_size;
        }
        double t_prefill0 = get_time_ms();
        hip_llm_set_decode_mode(gpu, 0);
        float *logits = prompt_added == 0 && cache_n > 0 ?
                        hip_llm_current_logits(gpu) : NULL;
        hip_llm_state_snapshot *pending_prefix_snapshot = NULL;
        hip_llm_state_snapshot *pending_prompt_snapshot = NULL;
        int cancelled = 0;
        int batch_index = 0;
        for (int off = 0; off < prompt_added; ) {
            if (g_stdio_cancel) { cancelled = 1; break; }
            int cc = prompt_added - off;
            if (cc > batch_size) cc = batch_size;
            if (publish_chunk)
                hip_llm_set_qwen4_batch_request_tokens(gpu, cc);
            if (requested_prefix > 0 && common + off < requested_prefix &&
                common + off + cc > requested_prefix)
                cc = requested_prefix - common - off;
            logits = hip_llm_forward_batch_logits(gpu, tokens + common + off, cc,
                                                   common + off);
            double batch_now = get_time_ms();
            double batch_ms = batch_now - t_prefill0;
            fprintf(stderr,
                    "llm_server: prefill batch=%d/%d tokens=%d..%d count=%d "
                    "elapsed=%.2f ms (%.2f tok/s)\n",
                    ++batch_index, batches, common + off,
                    common + off + cc - 1, cc, batch_ms,
                    batch_ms > 0.0 ? 1000.0 * (off + cc) / batch_ms : 0.0);
            fflush(stderr);
            if (!logits) break;
            if (snapshot_cache.entries && requested_prefix > common &&
                common + off + cc == requested_prefix)
                pending_prefix_snapshot = hip_llm_snapshot_state(gpu);
            off += cc;
        }
        double t_prefill1 = get_time_ms();
        if (g_stdio_cancel) cancelled = 1;
        if (cancelled) {
            hip_llm_free_state_snapshot(pending_prefix_snapshot);
            stdio_snapshot_cache_drop_resident(&snapshot_cache);
            hip_llm_reset_state(gpu);
            cache_n = 0;
            active_identity[0] = '\0';
            free(tokens);
            puts("OK 0 0 0 cancelled");
            fflush(stdout);
            continue;
        }
        if (!logits && prompt_added > 0) {
            hip_llm_free_state_snapshot(pending_prefix_snapshot);
            stdio_snapshot_cache_drop_resident(&snapshot_cache);
            hip_llm_reset_state(gpu);
            cache_n = 0;
            active_identity[0] = '\0';
            free(tokens); puts("ERR prefill"); fflush(stdout); continue;
        }
        /* Save the state at the complete prompt boundary. Generated text is
         * decoded to UTF-8 and may not re-tokenize to the original BPE pieces
         * on the next turn. If that happens, restore this boundary and replay
         * only the appended conversation suffix instead of resetting. */
        if (snapshot_cache.entries && !prompt_snapshot_present)
            pending_prompt_snapshot = hip_llm_snapshot_state(gpu);
        hllm_sampler *sampler = use_reference ? hllm_sampler_create(&request_sampling, n_vocab) : NULL;
        if (use_reference && !sampler) {
            hip_llm_free_state_snapshot(pending_prefix_snapshot);
            hip_llm_free_state_snapshot(pending_prompt_snapshot);
            stdio_snapshot_cache_drop_resident(&snapshot_cache);
            hip_llm_reset_state(gpu);
            cache_n = 0;
            active_identity[0] = '\0';
            free(tokens); puts("ERR sampler"); fflush(stdout); continue;
        }
        for (int i = 0; sampler && i < n_tokens; ++i) hllm_sampler_accept(sampler, tokens[i]);
        unsigned char *seen = (unsigned char *)calloc((size_t)n_vocab, 1);
        for (int i = 0; i < n_tokens; i++) {
            cache[i] = tokens[i];
        }
        cache_n = n_tokens;
        snprintf(active_identity, sizeof(active_identity), "%s", cache_identity);
        free(tokens);
        if (max_tokens < 0) max_tokens = 0;
        if (max_tokens > max_seq_len - cache_n) max_tokens = max_seq_len - cache_n;
        size_t text_cap = (size_t)max_tokens * 16 + 1, text_n = 0;
        char *text = (char *)calloc(text_cap ? text_cap : 1, 1);
        int generated = 0, finish_eos = 0;
        int eos = bpe_eos_id(vocab), eot = bpe_eot_id(vocab), im_end = -1;
        /* Qwen ChatML terminates an assistant turn with this control token;
         * GGUF's eot_token_id is a different token for this checkpoint. */
        for (int i = 0; i < n_vocab; ++i) {
            const char *s = bpe_token_to_str(vocab, i);
            if (s && strcmp(s, "<|im_end|>") == 0) { im_end = i; break; }
        }
        double t_decode0 = get_time_ms();
        hip_llm_reset_moe_stats(gpu);
        hip_llm_set_decode_mode(gpu, 1);
        hip_llm_qwen4_mtp_result mtp = {0};
        int mtp_index = 0, mtp_pending = -1, mtp_error = 0;
        int mtp_approx_fallback = 0;
        int mtp_adaptive_fallback = 0;
        int32_t q35_drafts[16], q35_argmax[16];
        int q35_count = 0, q35_index = 0, q35_rows = 0;
        int q35_dflash_fallback = 0;
        int q35_position = -1;
        float *q35_logits = NULL;
        const int q35_is_dflash = dflash_draft > 0;
        const int q35_draft = q35_is_dflash ? dflash_draft :
            (dense_mtp_draft > 0 && dense_mtp_window && temperature <= 0.0f ?
             dense_mtp_draft : 0);
        int q35_proposed = 0, q35_accepted = 0;
        double q35_draft_ms = 0.0, q35_verify_ms = 0.0,
               q35_commit_ms = 0.0;
        int32_t stops[] = { eos, eot, im_end };
        for (int k = 0; logits && k < max_tokens; k++) {
            if (g_stdio_cancel) { cancelled = 1; break; }
            int use_mtp = mtp_draft > 0 && temperature <= 0.0f &&
                          !mtp_approx_fallback && !mtp_adaptive_fallback;
            if (use_mtp && mtp_index == mtp.emitted) {
                int anchor = mtp_pending >= 0 ? mtp_pending : argmax_logits(logits, n_vocab);
                if (hip_llm_qwen4_mtp_step(gpu, anchor, cache_n, mtp_draft,
                        max_tokens-k, stops, 3, &mtp)) { mtp_error = 1; break; }
                mtp_index = 0; mtp_pending = mtp.pending;
                fprintf(stderr, "llm_server: MTP backend=%s drafted=%d accepted=%d emitted=%d draft_ms=%.3f verify_ms=%.3f\n",
                        getenv("LLM_QWEN4_MTP_TRUST_DRAFT") ? "hip-approx" : "hip",
                        mtp.drafted, mtp.accepted, mtp.emitted, mtp.draft_ms, mtp.verify_ms);
                if (getenv("LLM_QWEN4_MTP_APPROX") && mtp.drafted > 0 && mtp.accepted == 0) {
                    mtp_approx_fallback = 1;
                    fprintf(stderr, "llm_server: MTP approximate acceptance=0; falling back to target decode\n");
                }
                if (getenv("LLM_QWEN4_MTP_ADAPTIVE") && mtp.drafted > 0 &&
                    mtp.accepted * 2 < mtp.drafted) {
                    mtp_adaptive_fallback = 1;
                    fprintf(stderr, "llm_server: MTP acceptance=%d/%d; adaptive target fallback\n",
                            mtp.accepted, mtp.drafted);
                }
            }
            int q35_window = q35_draft > 0 && q35_rows > 0;
            int next = q35_window ?
                (q35_logits ?
                    (sampler ? hllm_sampler_sample(sampler,
                        q35_logits + (size_t)q35_index * n_vocab) :
                     (temperature <= 0.0f ? argmax_logits(
                        q35_logits + (size_t)q35_index * n_vocab, n_vocab) :
                      sample_top_k_p(q35_logits + (size_t)q35_index * n_vocab,
                        n_vocab, top_k, top_p, temperature, presence,
                        repetition, min_p, seen, &rng))) :
                 q35_argmax[q35_index]) :
                use_mtp ? mtp.tokens[mtp_index++] : sampler ? hllm_sampler_sample(sampler, logits) : (temperature <= 0.0f) ? argmax_logits(logits, n_vocab) :
                (coding_mode ? sample_top_k_p_coding(logits, n_vocab, top_k, top_p,
                               temperature, presence, repetition, min_p, seen, &rng, vocab) :
                 sample_top_k_p(logits, n_vocab, top_k, top_p, temperature, presence,
                               repetition, min_p, seen, &rng));
            if (next < 0 || next >= n_vocab) { mtp_error = 1; break; }
            if (sampler) hllm_sampler_accept(sampler, next);
            int is_stop = is_generation_stop(vocab, next, eos, eot) ||
                          next == im_end;
            const char *piece = bpe_token_to_str(vocab, next);
            if (!is_stop && piece && text) {
                int raw_n = (int)strlen(piece), dec_n = 0;
                char *decoded = bpe_byte_decode(piece, raw_n, &dec_n);
                if (decoded) {
                    if (text_n + (size_t)dec_n + 1 > text_cap) {
                        text_cap = (text_n + (size_t)dec_n + 1) * 2;
                        text = (char *)realloc(text, text_cap);
                    }
                    if (text) { memcpy(text + text_n, decoded, (size_t)dec_n); text_n += (size_t)dec_n; text[text_n] = 0; }
                    if (dec_n > 0) {
                        size_t tok_enc_n = 0;
                        char *tok_enc = b64_encode((const unsigned char *)decoded,
                                                   (size_t)dec_n, &tok_enc_n);
                        if (tok_enc) {
                            printf("TOK %s\n", tok_enc);
                            fflush(stdout);
                            free(tok_enc);
                        }
                    }
                    free(decoded);
                }
            }
            if (cache_n < max_seq_len) cache[cache_n++] = next;
            if (seen && next >= 0 && next < n_vocab) seen[next] = 1;
            generated++;
            if (is_stop) {
                /* Stop was sampled but never forwarded. Only processed
                 * tokens belong in the reusable KV/recurrent prefix.  A
                 * speculative row predicts this stop from an already
                 * processed input, so publish inputs through that row before
                 * closing the transaction. */
                cache_n--;
                if (q35_window) {
                    int processed = q35_index + 1;
                    double tc = get_time_ms();
                    int commit_rc = q35_is_dflash ?
                        hip_llm_qwen35_dflash2_commit(gpu, q35_position,
                                                      processed) :
                        hip_llm_qwen35_mtp_commit(gpu, processed);
                    q35_commit_ms += get_time_ms() - tc;
                    if (commit_rc) {
                        fprintf(stderr,
                                "llm_server: Qwen3.8 speculative stop commit failed "
                                "pos=%d rows=%d\n", q35_position, processed);
                        fflush(stderr);
                        mtp_error = 1;
                    }
                    q35_rows = q35_index = q35_count = 0;
                    q35_position = -1;
                    q35_logits = NULL;
                }
                finish_eos = 1;
                break;
            }
            if (q35_window) {
                int matched = q35_index < q35_count &&
                              next == q35_drafts[q35_index];
                q35_index++;
                if (matched) q35_accepted++;
                if (!matched || q35_index == q35_rows) {
                    double tc = get_time_ms();
                    int commit_rc = q35_is_dflash ?
                        hip_llm_qwen35_dflash2_commit(gpu, q35_position,
                                                      q35_index) :
                        hip_llm_qwen35_mtp_commit(gpu, q35_index);
                    q35_commit_ms += get_time_ms() - tc;
                    if (commit_rc) {
                        fprintf(stderr,
                                "llm_server: Qwen3.8 speculative commit failed "
                                "pos=%d rows=%d\n", q35_position, q35_index);
                        fflush(stderr);
                        mtp_error = 1; break;
                    }
                    q35_rows = q35_index = q35_count = 0;
                    q35_position = -1;
                    q35_logits = NULL;
                }
            }
            if (q35_draft > 0 && !q35_dflash_fallback &&
                (!q35_is_dflash || !coding_mode) &&
                q35_rows == 0 && k + 1 < max_tokens) {
                if (q35_is_dflash && cache_n - 1 >= 32768) {
                    q35_dflash_fallback = 1;
                    fprintf(stderr,
                            "llm_server: DFlash2 disabled at position %d; "
                            "using target decode for long context\n",
                            cache_n - 1);
                }
                if (!q35_dflash_fallback) {
                    int count = max_tokens - k - 1;
                    if (count > q35_draft) count = q35_draft;
                    double td = get_time_ms();
                    int propose_rc = count > 0 ? (q35_is_dflash ?
                        hip_llm_qwen35_dflash2_propose(gpu, next, cache_n - 1,
                                                       count, q35_drafts) :
                        hip_llm_qwen35_mtp_propose(gpu, next, cache_n - 1,
                                                   count, q35_drafts)) : 0;
                    q35_draft_ms += get_time_ms() - td;
                    if (propose_rc) {
                        fprintf(stderr,
                                "llm_server: Qwen3.8 speculative propose failed "
                                "pos=%d count=%d\n",
                                cache_n - 1, count);
                        fflush(stderr);
                        mtp_error = 1; break;
                    }
                    if (count > 0) {
                        int32_t inputs[16]; inputs[0] = next;
                        memcpy(inputs + 1, q35_drafts,
                               (size_t)count * sizeof(int32_t));
                        double tv = get_time_ms();
                        if (sampler || temperature > 0.0f)
                            q35_logits = hip_llm_qwen35_mtp_verify(
                                gpu, inputs, count + 1, cache_n - 1);
                        else
                            q35_logits = NULL;
                        if (q35_logits == NULL &&
                            hip_llm_qwen35_mtp_verify_argmax(gpu, inputs, count + 1,
                                                             cache_n - 1, q35_argmax)) {
                            fprintf(stderr,
                                    "llm_server: Qwen3.8 speculative verify failed "
                                    "pos=%d rows=%d\n",
                                    cache_n - 1, count + 1);
                            fflush(stderr);
                            mtp_error = 1; break;
                        }
                        q35_verify_ms += get_time_ms() - tv;
                        q35_proposed += count;
                        q35_count = count;
                        q35_rows = count + 1;
                        q35_index = 0;
                        q35_position = cache_n - 1;
                    }
                }
            }
            /* If approximate MTP just fell back after a zero-accept batch,
             * replay the emitted anchor through the target so the ordinary
             * decode path resumes with fresh logits and state. */
            if (!use_mtp && !q35_window && !q35_rows)
                logits = hip_llm_forward_logits(gpu, next, cache_n - 1);
            double token_now = get_time_ms();
            double token_ms = token_now - t_decode0;
            fprintf(stderr,
                    "llm_server: decode token=%d id=%d elapsed=%.2f ms "
                    "(%.2f tok/s)\n",
                    generated, next, token_ms,
                    token_ms > 0.0 ? 1000.0 * generated / token_ms : 0.0);
            fflush(stderr);
        }
        if (g_stdio_cancel) cancelled = 1;
        /* A length limit may cut through an accepted speculative window.
         * Publish exactly the rows whose tokens were emitted; later verifier
         * rows remain transaction-local and must not leak into the next turn. */
        if (!cancelled && !mtp_error && q35_rows > 0 && q35_index > 0) {
            double tc = get_time_ms();
            int commit_rc = q35_is_dflash ?
                hip_llm_qwen35_dflash2_commit(gpu, q35_position, q35_index) :
                hip_llm_qwen35_mtp_commit(gpu, q35_index);
            q35_commit_ms += get_time_ms() - tc;
            if (commit_rc) {
                fprintf(stderr,
                        "llm_server: Qwen3.8 speculative length commit failed "
                        "pos=%d rows=%d\n", q35_position, q35_index);
                fflush(stderr);
                mtp_error = 1;
            }
            q35_rows = q35_index = q35_count = 0;
            q35_position = -1;
            q35_logits = NULL;
        }
        if (mtp_error) {
            hip_llm_free_state_snapshot(pending_prefix_snapshot);
            hip_llm_free_state_snapshot(pending_prompt_snapshot);
            stdio_snapshot_cache_drop_resident(&snapshot_cache);
            cache_n = 0;
            active_identity[0] = '\0';
            hip_llm_reset_state(gpu);
            hip_llm_set_decode_mode(gpu, 0);
            hllm_sampler_free(sampler);
            free(text); free(seen);
            puts("ERR generation"); fflush(stdout); continue;
        }
        double t_decode1 = get_time_ms();
        hip_llm_set_decode_mode(gpu, 0);
        double prefill_ms = t_prefill1 - t_prefill0;
        double decode_ms = t_decode1 - t_decode0;
        double end_to_end_ms = t_decode1 - t_prefill0;
        int end_to_end_tokens = prompt_added + generated;
        fprintf(stderr,
                "llm_server: prompt=%d cached=%d added=%d batches=%d batch=%d "
                "prefill=%.2f ms (%.2f tok/s) decode=%d in %.2f ms (%.2f tok/s)\n",
                n_tokens, common, prompt_added, batches, batch_size,
                prefill_ms, prefill_ms > 0.0 ? 1000.0 * prompt_added / prefill_ms : 0.0,
                generated, decode_ms, decode_ms > 0.0 ? 1000.0 * generated / decode_ms : 0.0);
        fprintf(stderr,
                "llm_server: end-to-end=%d prompt-added + %d generated in %.2f ms (%.2f tok/s)\n",
                prompt_added, generated, end_to_end_ms,
                end_to_end_ms > 0.0 ? 1000.0 * end_to_end_tokens / end_to_end_ms : 0.0);
        if (q35_draft > 0) {
            fprintf(stderr,
                    "llm_server: %s drafted=%d accepted=%d draft_ms=%.3f "
                    "verify_ms=%.3f commit_ms=%.3f\n",
                    q35_is_dflash ? "DFlash2" : "Dense NextN",
                    q35_proposed, q35_accepted, q35_draft_ms,
                    q35_verify_ms, q35_commit_ms);
        }
        {
            hip_llm_moe_stats ms;
            if (hip_llm_get_moe_stats(gpu, &ms) == 0 &&
                ms.cache_hits + ms.cache_misses > 0) {
                double hit = 100.0 * (double)ms.cache_hits /
                             (double)(ms.cache_hits + ms.cache_misses);
                fprintf(stderr,
                        "llm_server: MoE cache=%.1f%% (%llu/%llu) H2D=%.2f GiB CPU=%llu GPU=%llu skipped=%llu CPU-time=%.2f ms\n",
                        hit, (unsigned long long)ms.cache_hits,
                        (unsigned long long)(ms.cache_hits + ms.cache_misses),
                        ms.h2d_bytes / (double)(1ULL << 30),
                        (unsigned long long)ms.cpu_assignments,
                        (unsigned long long)ms.gpu_assignments,
                        (unsigned long long)ms.skipped_assignments, ms.cpu_ms);
            }
        }
        {
            hip_llm_vram_stats vs = { .struct_size = sizeof(vs) };
            if (hip_llm_get_vram_stats(gpu, &vs) == 0) {
                fprintf(stderr,
                        "llm_server: VRAM free=%.1f MiB total=%.1f MiB peak-used=%.1f MiB\n",
                        vs.free_bytes / (double)(1ULL << 20),
                        vs.total_bytes / (double)(1ULL << 20),
                        vs.peak_used_bytes / (double)(1ULL << 20));
            }
        }
        size_t enc_n = 0; char *enc = b64_encode((const unsigned char *)(text ? text : ""), text_n, &enc_n);
        /* Trusted sidecar MTP advances only the independent NextN state.  The
         * target recurrent/KV state is therefore not a valid reusable prefix;
         * drop it before the next HTTP request rather than serving from stale
         * target state. */
        if (getenv("LLM_QWEN4_MTP_TRUST_DRAFT")) {
            hip_llm_free_state_snapshot(pending_prefix_snapshot);
            hip_llm_free_state_snapshot(pending_prompt_snapshot);
            pending_prefix_snapshot = pending_prompt_snapshot = NULL;
            stdio_snapshot_cache_drop_resident(&snapshot_cache);
            hip_llm_reset_state(gpu);
            cache_n = 0;
            active_identity[0] = '\0';
        }
        if (cancelled) {
            hip_llm_free_state_snapshot(pending_prefix_snapshot);
            hip_llm_free_state_snapshot(pending_prompt_snapshot);
            pending_prefix_snapshot = pending_prompt_snapshot = NULL;
            stdio_snapshot_cache_drop_resident(&snapshot_cache);
            hip_llm_reset_state(gpu);
            cache_n = 0;
            active_identity[0] = '\0';
        } else if (!getenv("LLM_QWEN4_MTP_TRUST_DRAFT")) {
            if (pending_prefix_snapshot) {
                stdio_snapshot_cache_publish(&snapshot_cache, cache_identity,
                    cache, requested_prefix, pending_prefix_snapshot);
                pending_prefix_snapshot = NULL;
            }
            if (pending_prompt_snapshot) {
                stdio_snapshot_cache_publish(&snapshot_cache, cache_identity,
                    cache, n_tokens, pending_prompt_snapshot);
                pending_prompt_snapshot = NULL;
            }
        }
        printf("OK %d %d %d %s %s %.3f %.3f\n", cancelled ? 0 : common,
               cancelled ? 0 : n_tokens, generated,
               cancelled ? "cancelled" : (finish_eos ? "stop" : "length"),
               enc ? enc : "", prefill_ms, decode_ms);
        fflush(stdout);
        hllm_sampler_free(sampler);
        free(enc); free(text); free(seen);
    }
    free(cache);
    stdio_snapshot_cache_free(&snapshot_cache);
    return 0;
}

static void print_first_n(const char *label, const float *v, int n, int show) {
    if (show > n) show = n;
    fprintf(stderr, "  %s [", label);
    for (int i = 0; i < show; i++) {
        fprintf(stderr, "%s%.6f", i > 0 ? ", " : "", v[i]);
    }
    fprintf(stderr, " ...] norm=%.4f\n", vec_norm(v, n));
}

/* ---- Quant-matvec A/B verifier (--verify-quant-kernels) ----
 *
 * For each ported IQ/TQ matvec kernel, run the runner-internal verifier
 * that compares the GPU launch_matvec_<type> against dequantize_row_<type>
 * + scalar matvec on identical random raw bytes. */
typedef void (*deq_row_fn)(const void *src, float *dst, int n);

static void dequantize_row_q8_0_padded(const void *src, float *dst, int n) {
    const unsigned char *p = (const unsigned char *)src;
    int nb = n / 32;
    for (int b = 0; b < nb; b++) {
        const unsigned char *bp = p + (size_t)b * 36;
        uint16_t dh;
        memcpy(&dh, bp, sizeof(dh));
        float d = ggml_fp16_to_fp32(dh);
        const signed char *qs = (const signed char *)(bp + 4);
        for (int i = 0; i < 32; i++) dst[b * 32 + i] = d * (float)qs[i];
    }
}

static int quant_type_from_name(const char *s) {
    if (!s) return -1;
    if (strcmp(s, "Q2_K") == 0)    return GGML_TYPE_Q2_K;
    if (strcmp(s, "Q3_K") == 0)    return GGML_TYPE_Q3_K;
    if (strcmp(s, "Q4_K") == 0)    return GGML_TYPE_Q4_K;
    if (strcmp(s, "Q5_K") == 0)    return GGML_TYPE_Q5_K;
    if (strcmp(s, "Q6_K") == 0)    return GGML_TYPE_Q6_K;
    if (strcmp(s, "Q8_0") == 0)    return GGML_TYPE_Q8_0;
    if (strcmp(s, "Q4_0") == 0)    return GGML_TYPE_Q4_0;
    if (strcmp(s, "Q4_1") == 0)    return GGML_TYPE_Q4_1;
    if (strcmp(s, "Q5_0") == 0)    return GGML_TYPE_Q5_0;
    if (strcmp(s, "Q5_1") == 0)    return GGML_TYPE_Q5_1;
    if (strcmp(s, "IQ2_XXS") == 0) return GGML_TYPE_IQ2_XXS;
    if (strcmp(s, "IQ2_XS") == 0)  return GGML_TYPE_IQ2_XS;
    if (strcmp(s, "IQ2_S") == 0)   return GGML_TYPE_IQ2_S;
    if (strcmp(s, "IQ3_XXS") == 0) return GGML_TYPE_IQ3_XXS;
    if (strcmp(s, "IQ3_S") == 0)   return GGML_TYPE_IQ3_S;
    if (strcmp(s, "IQ1_S") == 0)   return GGML_TYPE_IQ1_S;
    if (strcmp(s, "IQ1_M") == 0)   return GGML_TYPE_IQ1_M;
    if (strcmp(s, "IQ4_NL") == 0)  return GGML_TYPE_IQ4_NL;
    if (strcmp(s, "IQ4_XS") == 0)  return GGML_TYPE_IQ4_XS;
    if (strcmp(s, "TQ1_0") == 0)   return GGML_TYPE_TQ1_0;
    if (strcmp(s, "TQ2_0") == 0)   return GGML_TYPE_TQ2_0;
    return -1;
}

static int run_verify_quant_kernels(void) {
    fprintf(stderr, "=== --verify-quant-kernels: A/B HIP matvec vs CPU dequant+matvec ===\n");
    fprintf(stderr, "Shape: n_rows=64, n_cols=512.  Pass threshold: rel_l2 < 1e-4.\n\n");

    hip_llm_runner *r = hip_llm_init(0, 0);
    if (!r) { fprintf(stderr, "hip_llm_init failed\n"); return 1; }

    struct { int type; const char *name; deq_row_fn fn; } cases[] = {
        { GGML_TYPE_Q2_K,    "Q2_K",    dequantize_row_q2_K    },
        { GGML_TYPE_Q3_K,    "Q3_K",    dequantize_row_q3_K    },
        { GGML_TYPE_Q4_K,    "Q4_K",    dequantize_row_q4_K    },
        { GGML_TYPE_Q5_K,    "Q5_K",    dequantize_row_q5_K    },
        { GGML_TYPE_Q6_K,    "Q6_K",    dequantize_row_q6_K    },
        { GGML_TYPE_Q8_0,    "Q8_0",    dequantize_row_q8_0_padded },
        { GGML_TYPE_Q4_0,    "Q4_0",    dequantize_row_q4_0    },
        { GGML_TYPE_Q4_1,    "Q4_1",    dequantize_row_q4_1    },
        { GGML_TYPE_Q5_0,    "Q5_0",    dequantize_row_q5_0    },
        { GGML_TYPE_Q5_1,    "Q5_1",    dequantize_row_q5_1    },
        { GGML_TYPE_IQ2_XXS, "IQ2_XXS", dequantize_row_iq2_xxs },
        { GGML_TYPE_IQ2_XS,  "IQ2_XS",  dequantize_row_iq2_xs  },
        { GGML_TYPE_IQ2_S,   "IQ2_S",   dequantize_row_iq2_s   },
        { GGML_TYPE_IQ3_XXS, "IQ3_XXS", dequantize_row_iq3_xxs },
        { GGML_TYPE_IQ3_S,   "IQ3_S",   dequantize_row_iq3_s   },
        { GGML_TYPE_IQ1_S,   "IQ1_S",   dequantize_row_iq1_s   },
        { GGML_TYPE_IQ1_M,   "IQ1_M",   dequantize_row_iq1_m   },
        { GGML_TYPE_IQ4_NL,  "IQ4_NL",  dequantize_row_iq4_nl  },
        { GGML_TYPE_IQ4_XS,  "IQ4_XS",  dequantize_row_iq4_xs  },
        { GGML_TYPE_TQ1_0,   "TQ1_0",   dequantize_row_tq1_0   },
        { GGML_TYPE_TQ2_0,   "TQ2_0",   dequantize_row_tq2_0   },
    };
    int n_cases = (int)(sizeof(cases) / sizeof(cases[0]));
    int n_pass = 0, n_fail = 0, n_skip = 0;
    const double thresh = 1e-4;

    fprintf(stderr, "%-8s  %12s  %12s   %s\n", "type", "rel_l2", "max_abs", "result");
    fprintf(stderr, "%-8s  %12s  %12s   %s\n", "----", "------", "-------", "------");
    for (int i = 0; i < n_cases; i++) {
        double rel_l2 = 0.0, max_abs = 0.0;
        int rc = hip_llm_verify_quant_matvec(r, cases[i].type, cases[i].fn,
                                             64, 512, &rel_l2, &max_abs);
        if (rc != 0) {
            fprintf(stderr, "%-8s  %12s  %12s   SKIP (rc=%d)\n",
                    cases[i].name, "-", "-", rc);
            n_skip++;
            continue;
        }
        int pass = (rel_l2 < thresh);
        fprintf(stderr, "%-8s  %12.3e  %12.3e   %s\n",
                cases[i].name, rel_l2, max_abs, pass ? "PASS" : "FAIL");
        if (pass) n_pass++; else n_fail++;
    }
    fprintf(stderr, "\n%d PASS, %d FAIL, %d SKIP (threshold rel_l2 < %.0e).\n",
            n_pass, n_fail, n_skip, thresh);

    hip_llm_free(r);
    return n_fail == 0 ? 0 : 2;
}

static int run_verify_moe_routing(void) {
    fprintf(stderr, "=== --verify-moe-routing: batched 512-expert top-k ===\n");
    hip_llm_runner *r = hip_llm_init(0, 0);
    if (!r) { fprintf(stderr, "hip_llm_init failed\n"); return 1; }
    int rc = hip_llm_verify_moe_routing(r, 512, 10);
    hip_llm_free(r);
    fprintf(stderr, "routing: %s\n", rc == 0 ? "PASS" : "FAIL");
    return rc == 0 ? 0 : 2;
}

static int cmp_float_asc(const void *a, const void *b) {
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

static int run_bench_quant_matvec(const char *type_name, int n_rows, int n_cols, int iters, int repeats) {
    int type = quant_type_from_name(type_name);
    if (type < 0 || n_rows <= 0 || n_cols <= 0 || iters <= 0 || repeats <= 0) {
        fprintf(stderr, "Invalid --bench-quant-matvec args. Usage: --bench-quant-matvec TYPE ROWS COLS ITERS [REPEATS]\n");
        return 1;
    }

    hip_llm_runner *r = hip_llm_init(0, 0);
    if (!r) { fprintf(stderr, "hip_llm_init failed\n"); return 1; }

    float *samples = (float *)malloc((size_t)repeats * sizeof(float));
    if (!samples) {
        hip_llm_free(r);
        return 1;
    }
    int rc = 0;
    for (int rep = 0; rep < repeats; rep++) {
        rc = hip_llm_bench_quant_matvec(r, type, n_rows, n_cols, 20, iters, &samples[rep]);
        if (rc != 0) break;
    }
    hip_llm_free(r);
    if (rc != 0) {
        free(samples);
        fprintf(stderr, "--bench-quant-matvec failed (type=%s rows=%d cols=%d iters=%d repeats=%d rc=%d)\n",
                type_name, n_rows, n_cols, iters, repeats, rc);
        return 1;
    }

    qsort(samples, (size_t)repeats, sizeof(float), cmp_float_asc);
    float ms = samples[repeats / 2];
    float min_ms = samples[0];
    float max_ms = samples[repeats - 1];
    double dot_ops = 2.0 * (double)n_rows * (double)n_cols;
    double gops = dot_ops / (double)ms / 1.0e6;
    fprintf(stderr,
            "quant_matvec %s rows=%d cols=%d iters=%d repeats=%d: median %.6f ms/launch  %.2f GOP/s  range [%.6f, %.6f]\n",
            type_name, n_rows, n_cols, iters, repeats, ms, gops, min_ms, max_ms);
    free(samples);
    return 0;
}

/* ---- Main ---- */

static int argmax_logits(const float *logits, int n) {
    int best = 0;
    float best_v = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > best_v) { best_v = logits[i]; best = i; }
    }
    return best;
}

/* LLM_DECODE_LOGITS_HASH=1: route decode through full logits and hash them. */
static int decode_logits_hash_on;
static unsigned long long decode_logits_hash = 1469598103934665603ULL;
static void dump_top_logits(const float *logits, int n) {
    const char *dump_path = getenv("LLM_DUMP_LOGITS_BIN");
    if (dump_path) {
        FILE *f = fopen(dump_path, "wb");
        if (f) { fwrite(logits, sizeof(float), (size_t)n, f); fclose(f); }
    }
    if (!getenv("LLM_DUMP_LOGITS")) return;
    int ids[20];
    float vals[20];
    int count = 0;
    for (int id = 0; id < n; id++) {
        float v = logits[id];
        int p = count < 20 ? count++ : 19;
        if (count == 20 && v <= vals[19]) continue;
        while (p > 0 && v > vals[p - 1]) {
            if (p < 20) { vals[p] = vals[p - 1]; ids[p] = ids[p - 1]; }
            p--;
        }
        vals[p] = v; ids[p] = id;
    }
    fprintf(stderr, "top logits:\n");
    for (int i = 0; i < count; i++) fprintf(stderr, "%d %.9g\n", ids[i], vals[i]);
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *prompt = "Hello, how are you?";
    char *prompt_owned = NULL;
    int max_tokens = 8;
    int max_seq_len = 256;
    int bench_mode = 0;       /* --bench: split prefill/decode tps; skip CPU compare */
    int gpu_only_bench = 0;   /* --gpu-only-bench: also skip CPU model load */
    int gpu_only = 0;         /* --gpu-only: generate normally without CPU shadow */
    int decode_n = 0;         /* --decode N: greedy-sample N tokens after prefill */
    int prefill_pad = 0;      /* --prefill-len M: pad prompt up to M tokens with last token (for bench) */
    int bench_repeat = 1;     /* --bench-repeat N: rerun the same request N times in-process */
    int bench_depth = 0;      /* --bench-depth N: random-token prefix before the measured request */
    int compare_paths = 0;    /* --compare-paths: report rel-L2 between batched and per-token logits */
    int coding_mode = 0;      /* Qwen3.8 non-thinking coding sampling profile */
    int reference_sampling = 0;
    hllm_sampler_config sampling;
    hllm_sampler_defaults(&sampling);
    sampling.temperature = 0.0f;
    const char *trace_prefix = NULL;
    int bench_ignore_eos = 0;
    int qwen4_coding_profile = 0;
    int qwen4_batched_prefill = 0;
    int qwen4_prefill_staging = 0;
    int qwen4_prefill_stage_mb = 0;
    int moe_cache_mb = 0;
    int moe_cpu_only = 0;
    hip_llm_kv_cache_type kv_cache_type = HIP_LLM_KV_AUTO;
    hip_llm_decode_kernel_mode decode_kernel_mode = HIP_LLM_DECODE_KERNEL_DEFAULT;
    hip_llm_decode_layout_mode decode_layout_mode = HIP_LLM_DECODE_LAYOUT_NATIVE;
    const char *decode_layout_cache_path = NULL;
    int decode_layout_budget_mib = 0;
    int prefill_batch_tokens = 0;
    int qwen35_batched_prefill = 0;
    int qwen35_prefill_bf16 = 0;
    int qwen35_decode_graph = 0;
    int qwen35_reference_math = 0;
    int qwen35_native_q8_attention = 0;
    int qwen35_native_q8_prefill = 0;
    int qwen35_native_q2k = 0;
    int qwen35_native_mmvq = 0;
    int max_layers = 0;
    int verify_hc_batch = 0, verify_ple_split = 0, verify_ssm_projections = 0, verify_moe_native = 0;
    int verify_glm5next_kda = 0;
    int glm5next_kda_dim = 128;
    int verify_glm5next_dsa = 0;
    int verify_glm5next_model = 0;
    int verify_glm5next_projections = 0;
    int verify_glm5next_kda_layer = 0;
    int verify_glm5next_dsa_layer = 0;
    const char *inspect_qwen4_nextn = NULL;
    const char *qwen35_mtp_path = NULL;
    int qwen35_mtp_draft = 3;
    int qwen35_mtp_window = 0;
    const char *qwen35_dflash2_path = NULL;
    int qwen35_dflash2_draft = 4;
    int qwen35_snapshot_max_tokens = 0;
    int context_cache_entries = 4;
    int context_cache_max_mib = 2048;
    const char *load_qwen4_nextn_fusion = NULL;
    /* Q4_K/Q6_K exact verification is host-synchronization bound; recurrent
     * width-2 drafts minimize rejected-suffix work on the RX 9070 XT. */
    /* Scalar exact verification still evaluates the target one token at a
     * time, so extra sidecar steps only add overhead. Width 1 is the default;
     * wider windows remain available for the experimental batched verifier. */
    int qwen4_mtp_draft = 1;
    int verify_qwen4_nextn = 0;
    int qwen4_mtp = 0;
    int qwen4_mtp_trust = 0;
    int qwen4_mtp_adaptive = 0;
    int qwen4_exact = 0;
    int qwen4_mtp_check = 0;
    int verify_qwen4_qsa = 0;
    int qwen4_mtp_cache_mb = 128;
    int qwen4_mtp_window = 0;
    int glm5next_verify_layer = 0;
    int verify_quant_kernels = 0; /* --verify-quant-kernels: A/B HIP vs CPU per quant type, then exit */
    int verify_moe_routing = 0;
    int stdio_server = 0;
    const char *bench_qmv_type = NULL; /* --bench-quant-matvec TYPE ROWS COLS ITERS [REPEATS] */
    int bench_qmv_rows = 0, bench_qmv_cols = 0, bench_qmv_iters = 0, bench_qmv_repeats = 1;

    /* --st <qwen3.safetensors>: sanity-check the safetensors loader + hidden
     * snapshots (text-encoder path). Loads, runs a few forwards, prints snapshot
     * norms (finite + reasonable = loader OK), then exits. */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--st") == 0 && i + 1 < argc) {
            const char *st_path = argv[i + 1];
            hip_llm_runner *r = hip_llm_init(0, 1);
            if (!r || hip_llm_load_weights_qwen3_safetensors(r, st_path, 512) != 0) {
                fprintf(stderr, "--st: load failed\n"); return 1;
            }
            int hs[3] = {8, 17, 26};
            if (hip_llm_set_hidden_snapshot_layers(r, hs, 3) != 0) { fprintf(stderr, "--st: set snapshots failed\n"); return 1; }
            int n = hip_llm_n_embd(r);
            float *snap = (float *)malloc((size_t)3 * n * sizeof(float));
            hip_llm_reset_state(r);
            for (int pos = 0; pos < 5; pos++) {
                int32_t tok = (int32_t)(100 + pos);
                if (!hip_llm_forward(r, tok, pos)) { fprintf(stderr, "--st: forward failed at pos %d\n", pos); return 1; }
                if (hip_llm_read_hidden_snapshots(r, snap, 3, n) != 0) { fprintf(stderr, "--st: read snapshots failed\n"); return 1; }
                for (int s = 0; s < 3; s++) {
                    double nn = 0; int nan = 0;
                    for (int j = 0; j < n; j++) { float v = snap[s*n+j]; if (v != v) nan = 1; nn += (double)v*v; }
                    printf("pos %d  layer[%d]  norm=%.4f  first=%.5f%s\n", pos, hs[s], sqrt(nn), snap[s*n], nan ? "  NaN!" : "");
                }
            }
            free(snap); hip_llm_free(r);
            return 0;
        }
    }

    /* Parse args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--verify-quant-kernels") == 0) {
            verify_quant_kernels = 1;
        } else if (strcmp(argv[i], "--verify-moe-routing") == 0) {
            verify_moe_routing = 1;
        } else if (strcmp(argv[i], "--stdio-server") == 0) {
            stdio_server = 1;
        } else if (strcmp(argv[i], "--bench-quant-matvec") == 0 && i + 4 < argc) {
            bench_qmv_type = argv[++i];
            bench_qmv_rows = atoi(argv[++i]);
            bench_qmv_cols = atoi(argv[++i]);
            bench_qmv_iters = atoi(argv[++i]);
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                bench_qmv_repeats = atoi(argv[++i]);
            }
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "--prompt-file") == 0 && i + 1 < argc) {
            const char *path = argv[++i];
            FILE *pf = fopen(path, "rb");
            if (!pf) { perror("--prompt-file"); return 2; }
            if (fseek(pf, 0, SEEK_END) != 0) { fclose(pf); return 2; }
            long n = ftell(pf);
            if (n < 0 || fseek(pf, 0, SEEK_SET) != 0) { fclose(pf); return 2; }
            prompt_owned = (char *)malloc((size_t)n + 1);
            if (!prompt_owned || fread(prompt_owned, 1, (size_t)n, pf) != (size_t)n) {
                fclose(pf); free(prompt_owned); prompt_owned = NULL;
                fprintf(stderr, "--prompt-file: read failed\n"); return 2;
            }
            fclose(pf); prompt_owned[n] = '\0'; prompt = prompt_owned;
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            max_seq_len = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--bench") == 0) {
            bench_mode = 1;
        } else if (strcmp(argv[i], "--gpu-only-bench") == 0) {
            bench_mode = 1;
            gpu_only_bench = 1;
        } else if (strcmp(argv[i], "--gpu-only") == 0) {
            gpu_only = 1;
        } else if (strcmp(argv[i], "--decode") == 0 && i + 1 < argc) {
            decode_n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--bench-repeat") == 0 && i + 1 < argc) {
            bench_repeat = atoi(argv[++i]);
            if (bench_repeat < 1) bench_repeat = 1;
            if (bench_repeat > 64) bench_repeat = 64;
        } else if (strcmp(argv[i], "--bench-depth") == 0 && i + 1 < argc) {
            bench_depth = atoi(argv[++i]);
            if (bench_depth < 0) { fprintf(stderr, "Invalid benchmark depth\n"); return 2; }
        } else if (strcmp(argv[i], "--prefill-len") == 0 && i + 1 < argc) {
            prefill_pad = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--compare-paths") == 0) {
            compare_paths = 1;
        } else if (strcmp(argv[i], "--sampling-profile") == 0 && i + 1 < argc) {
            if (strcmp(argv[++i], "llama")) { fprintf(stderr, "--sampling-profile requires llama\n"); return 2; }
            reference_sampling = 1;
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            char *end = NULL;
            unsigned long value = strtoul(argv[++i], &end, 10);
            if (!*argv[i] || *end || value >= UINT32_MAX) { fprintf(stderr, "Invalid explicit seed\n"); return 2; }
            sampling.seed = (uint32_t)value; reference_sampling = 1;
        } else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
            sampling.temperature = strtof(argv[++i], NULL); reference_sampling = 1;
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            sampling.top_k = atoi(argv[++i]); reference_sampling = 1;
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            sampling.top_p = strtof(argv[++i], NULL); reference_sampling = 1;
        } else if (strcmp(argv[i], "--min-p") == 0 && i + 1 < argc) {
            sampling.min_p = strtof(argv[++i], NULL); reference_sampling = 1;
        } else if (strcmp(argv[i], "--repeat-penalty") == 0 && i + 1 < argc) {
            sampling.repetition = strtof(argv[++i], NULL); reference_sampling = 1;
        } else if (strcmp(argv[i], "--presence-penalty") == 0 && i + 1 < argc) {
            sampling.presence = strtof(argv[++i], NULL); reference_sampling = 1;
        } else if (strcmp(argv[i], "--frequency-penalty") == 0 && i + 1 < argc) {
            sampling.frequency = strtof(argv[++i], NULL); reference_sampling = 1;
        } else if (strcmp(argv[i], "--penalty-last-n") == 0 && i + 1 < argc) {
            sampling.penalty_last_n = atoi(argv[++i]); reference_sampling = 1;
        } else if (strcmp(argv[i], "--trace-prefix") == 0 && i + 1 < argc) {
            trace_prefix = argv[++i];
        } else if (strcmp(argv[i], "--bench-ignore-eos") == 0) {
            bench_ignore_eos = 1;
        } else if (strcmp(argv[i], "--coding") == 0) {
            coding_mode = 1;
        } else if (strcmp(argv[i], "--qwen4-coding-profile") == 0) {
            qwen4_coding_profile = 1;
        } else if (strcmp(argv[i], "--qwen4-batched-prefill") == 0) {
            qwen4_batched_prefill = 1;
        } else if (strcmp(argv[i], "--qwen4-prefill-staging") == 0) {
            qwen4_prefill_staging = 1;
        } else if (strcmp(argv[i], "--qwen4-prefill-stage-mb") == 0 && i + 1 < argc) {
            qwen4_prefill_staging = 1;
            qwen4_prefill_stage_mb = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--moe-cache-mb") == 0 && i + 1 < argc) {
            moe_cache_mb = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--moe-cpu") == 0) {
            moe_cpu_only = 1;
        } else if (strcmp(argv[i], "--kv-cache") == 0 && i + 1 < argc) {
            const char *mode = argv[++i];
            if (!strcmp(mode, "auto")) kv_cache_type = HIP_LLM_KV_AUTO;
            else if (!strcmp(mode, "f32")) kv_cache_type = HIP_LLM_KV_F32;
            else if (!strcmp(mode, "f16")) kv_cache_type = HIP_LLM_KV_F16;
            else if (!strcmp(mode, "q8q4")) kv_cache_type = HIP_LLM_KV_Q8_0_Q4_0;
            else if (!strcmp(mode, "q8q8")) kv_cache_type = HIP_LLM_KV_Q8_0_Q8_0;
            else { fprintf(stderr, "--kv-cache must be auto, f32, f16, q8q4, or q8q8\n"); return 2; }
        } else if (strcmp(argv[i], "--ubatch") == 0 && i + 1 < argc) {
            prefill_batch_tokens = atoi(argv[++i]);
            if (prefill_batch_tokens < 1 || prefill_batch_tokens > 8192) {
                fprintf(stderr, "--ubatch must be in 1..8192\n"); return 2;
            }
        } else if (strcmp(argv[i], "--qwen35-batched-prefill") == 0) {
            qwen35_batched_prefill = 1;
        } else if (strcmp(argv[i], "--qwen35-decode-graph") == 0) {
            qwen35_decode_graph = 1;
        } else if (strcmp(argv[i], "--qwen35-reference-math") == 0) {
            qwen35_reference_math = 1;
        } else if (strcmp(argv[i], "--qwen35-native-q8-attn") == 0) {
            qwen35_native_q8_attention = 1;
        } else if (strcmp(argv[i], "--qwen35-native-q8-prefill") == 0) {
            qwen35_native_q8_prefill = 1;
        } else if (strcmp(argv[i], "--qwen35-native-q2k") == 0) {
            qwen35_native_q2k = 1;
        } else if (strcmp(argv[i], "--qwen35-native-mmvq") == 0) {
            qwen35_native_mmvq = 1;
        } else if (strcmp(argv[i], "--qwen35-prefill-bf16") == 0) {
            qwen35_batched_prefill = 1;
            qwen35_prefill_bf16 = 1;
        } else if (strcmp(argv[i], "--decode-kernels") == 0 && i + 1 < argc) {
            const char *mode = argv[++i];
            if (!strcmp(mode, "native")) decode_kernel_mode = HIP_LLM_DECODE_KERNEL_NATIVE;
            else if (!strcmp(mode, "dp4a2")) decode_kernel_mode = HIP_LLM_DECODE_KERNEL_DP4A2;
            else if (!strcmp(mode, "auto")) decode_kernel_mode = HIP_LLM_DECODE_KERNEL_AUTO;
            else { fprintf(stderr, "--decode-kernels must be native, dp4a2, or auto\n"); return 2; }
        } else if (strcmp(argv[i], "--decode-layout") == 0 && i + 1 < argc) {
            const char *mode = argv[++i];
            if (!strcmp(mode, "native")) decode_layout_mode = HIP_LLM_DECODE_LAYOUT_NATIVE;
            else if (!strcmp(mode, "auto")) decode_layout_mode = HIP_LLM_DECODE_LAYOUT_AUTO_REPACK;
            else { fprintf(stderr, "--decode-layout must be native or auto\n"); return 2; }
        } else if (strcmp(argv[i], "--decode-layout-cache") == 0 && i + 1 < argc) {
            decode_layout_cache_path = argv[++i];
        } else if (strcmp(argv[i], "--decode-layout-budget-mib") == 0 && i + 1 < argc) {
            decode_layout_budget_mib = atoi(argv[++i]);
            if (decode_layout_budget_mib < 0 || decode_layout_budget_mib > 4096) {
                fprintf(stderr, "--decode-layout-budget-mib must be 0..4096\n"); return 2;
            }
        } else if (strcmp(argv[i], "--max-layers") == 0 && i + 1 < argc) {
            max_layers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--verify-ple-split") == 0) {
            verify_ple_split = 1;
        } else if (strcmp(argv[i], "--verify-moe-native") == 0) {
            verify_moe_native = 1;
        } else if (strcmp(argv[i], "--verify-ssm-projections") == 0) {
            verify_ssm_projections = 1;
        } else if (strcmp(argv[i], "--verify-hc-batch") == 0) {
            verify_hc_batch = 1;
        } else if (strcmp(argv[i], "--verify-glm5next-kda") == 0) {
            verify_glm5next_kda = 1;
            if (i + 1 < argc && argv[i + 1][0] != '-') glm5next_kda_dim = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--verify-glm5next-dsa") == 0) {
            verify_glm5next_dsa = 1;
        } else if (strcmp(argv[i], "--verify-glm5next-model") == 0) {
            verify_glm5next_model = 1;
        } else if (strcmp(argv[i], "--verify-glm5next-projections") == 0) {
            verify_glm5next_projections = 1;
            if (i + 1 < argc && argv[i + 1][0] != '-') glm5next_verify_layer = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--verify-glm5next-kda-layer") == 0) {
            verify_glm5next_kda_layer = 1;
            if (i + 1 < argc && argv[i + 1][0] != '-') glm5next_verify_layer = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--verify-glm5next-dsa-layer") == 0) {
            verify_glm5next_dsa_layer = 1;
            if (i + 1 < argc && argv[i + 1][0] != '-') glm5next_verify_layer = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--qwen35-mtp") == 0 && i + 1 < argc) {
            qwen35_mtp_path = argv[++i];
        } else if (strcmp(argv[i], "--qwen35-mtp-draft") == 0 && i + 1 < argc) {
            qwen35_mtp_draft = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--qwen35-mtp-window") == 0) {
            qwen35_mtp_window = 1;
        } else if (strcmp(argv[i], "--qwen35-dflash2") == 0 && i + 1 < argc) {
            qwen35_dflash2_path = argv[++i];
        } else if (strcmp(argv[i], "--qwen35-dflash2-draft") == 0 && i + 1 < argc) {
            qwen35_dflash2_draft = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--qwen35-snapshot-max-tokens") == 0 && i + 1 < argc) {
            qwen35_snapshot_max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--context-cache-entries") == 0 && i + 1 < argc) {
            context_cache_entries = atoi(argv[++i]);
            if (context_cache_entries < 0 || context_cache_entries > 64) {
                fprintf(stderr, "--context-cache-entries must be 0..64\n"); return 1;
            }
        } else if (strcmp(argv[i], "--context-cache-max-mib") == 0 && i + 1 < argc) {
            context_cache_max_mib = atoi(argv[++i]);
            if (context_cache_max_mib < 0 || context_cache_max_mib > 65536) {
                fprintf(stderr, "--context-cache-max-mib must be 0..65536\n"); return 1;
            }
        } else if (strcmp(argv[i], "--inspect-qwen4-nextn") == 0 && i + 1 < argc) {
            inspect_qwen4_nextn = argv[++i];
        } else if (strcmp(argv[i], "--verify-qwen4-nextn") == 0 && i + 1 < argc) {
            verify_qwen4_nextn = 1;
            load_qwen4_nextn_fusion = argv[++i];
        } else if (strcmp(argv[i], "--load-qwen4-nextn-fusion") == 0 && i + 1 < argc) {
            load_qwen4_nextn_fusion = argv[++i];
        } else if (strcmp(argv[i], "--qwen4-mtp") == 0 && i + 1 < argc) {
            /* MTP and the legacy fusion loader share the same sidecar. */
            load_qwen4_nextn_fusion = argv[++i];
            qwen4_mtp = 1;
        } else if (strcmp(argv[i], "--qwen4-exact") == 0) {
            qwen4_exact = 1;
        } else if (strcmp(argv[i], "--qwen4-mtp-check") == 0) {
            qwen4_mtp_check = 1;
        } else if (strcmp(argv[i], "--verify-qwen4-qsa") == 0) {
            verify_qwen4_qsa = 1;
        } else if (strcmp(argv[i], "--qwen4-mtp-cache-mb") == 0 && i+1<argc) {
            qwen4_mtp_cache_mb=atoi(argv[++i]);
            if(qwen4_mtp_cache_mb<32 || qwen4_mtp_cache_mb>4096) {
                fprintf(stderr,"--qwen4-mtp-cache-mb must be 32..4096\n");return 1;
            }
        } else if (strcmp(argv[i], "--qwen4-mtp-verify") == 0 && i+1<argc) {
            const char *mode=argv[++i];
            if(strcmp(mode,"scalar") && strcmp(mode,"window")) {
                fprintf(stderr,"--qwen4-mtp-verify must be scalar or window\n");return 1;
            }
            qwen4_mtp_window=strcmp(mode,"window")==0;
        } else if (strcmp(argv[i], "--qwen4-mtp-draft") == 0 && i + 1 < argc) {
            qwen4_mtp_draft = atoi(argv[++i]);
            if (qwen4_mtp_draft < 1) qwen4_mtp_draft = 1;
            if (qwen4_mtp_draft > 32) qwen4_mtp_draft = 32;
        } else if (strcmp(argv[i], "--qwen4-mtp-trust-draft") == 0) {
            qwen4_mtp_trust = 1;
        } else if (strcmp(argv[i], "--qwen4-mtp-adaptive") == 0) {
            qwen4_mtp_adaptive = 1;
        } else if (argv[i][0] != '-') {
            model_path = argv[i];
        } else {
            fprintf(stderr, "Usage: %s [model.gguf] [-t \"prompt\"] [-n max_tokens] [-s max_seq_len]\n", argv[0]);
            fprintf(stderr, "       [--bench] [--gpu-only-bench] [--gpu-only] [--decode N] [--prefill-len M] [--bench-depth N] [--coding]\n");
            fprintf(stderr, "       [--moe-cache-mb MiB] [--moe-cpu]\n");
            fprintf(stderr, "       [--kv-cache auto|f32|f16|q8q4|q8q8] [--ubatch N] [--qwen35-batched-prefill]\n");
            fprintf(stderr, "       [--qwen35-prefill-bf16] (experimental BF16 prefill projections; unchanged decode kernels)\n");
            fprintf(stderr, "       [--qwen35-decode-graph] [--qwen35-native-q8-attn] (gfx1201 Q8/Q8 decode)\n");
            fprintf(stderr, "       [--qwen35-native-q8-prefill] (also use native Q8/Q8 for prefill)\n");
            fprintf(stderr, "       [--qwen35-native-q2k] (native Q2_K x Q8_1 decode on gfx1201)\n");
            fprintf(stderr, "       [--qwen35-native-mmvq] (native Q2_K, IQ2_XXS/XS/S, IQ3_XXS/S x Q8_1 decode)\n");
            fprintf(stderr, "       [--qwen35-mtp SIDECAR --qwen35-mtp-draft 1..16] (verified dense NextN, benchmark mode)\n");
            fprintf(stderr, "       [--qwen35-mtp-window] (exact Q8/Q8 target windows, draft <=15; requires decode graph)\n");
            fprintf(stderr, "       [--qwen35-dflash2 SIDECAR --qwen35-dflash2-draft 1..7] (exact DFlash2 windows)\n");
            fprintf(stderr, "       [--qwen35-snapshot-max-tokens N] (bound Qwen3.8 Q8 prompt snapshots)\n");
            fprintf(stderr, "       [--context-cache-entries N] [--context-cache-max-mib MiB] (bounded prompt snapshots; portable snapshots may cross contexts)\n");
            fprintf(stderr, "       [--qwen35-reference-math] (diagnostic; incomplete whole-model parity)\n");
            fprintf(stderr, "       [--sampling-profile llama] [--seed N] [--temp T] [--top-k K] [--top-p P] [--min-p P]\n");
            fprintf(stderr, "       [--repeat-penalty R] [--presence-penalty P] [--frequency-penalty F] [--penalty-last-n N]\n");
            fprintf(stderr, "       [--trace-prefix PATH] [--bench-ignore-eos] (synthetic benchmark only)\n");
            fprintf(stderr, "       [--decode-kernels native|dp4a2|auto]\n");
            fprintf(stderr, "       [--decode-layout native|auto] [--decode-layout-cache auto|off|PATH]\n");
            fprintf(stderr, "       [--decode-layout-budget-mib MiB]\n");
            fprintf(stderr, "       [--qwen4-mtp SIDECAR.gguf] [--qwen4-mtp-draft 1..32]\n");
            fprintf(stderr, "       [--qwen4-mtp-cache-mb MiB] [--qwen4-mtp-verify scalar|window]\n");
            fprintf(stderr, "       [--qwen4-mtp-check] [--qwen4-exact]\n");
            fprintf(stderr, "       [--qwen4-mtp-trust-draft] (approximate sidecar-only mode)\n");
            fprintf(stderr, "       [--qwen4-mtp-adaptive] (exact low-acceptance fallback)\n");
            fprintf(stderr, "       [--verify-qwen4-nextn SIDECAR.gguf] [--verify-qwen4-qsa]\n");
            fprintf(stderr, "       [--qwen4-batched-prefill] [--qwen4-prefill-staging]\n");
            fprintf(stderr, "       [--qwen4-prefill-stage-mb MiB]\n");
            fprintf(stderr, "       [--verify-quant-kernels] [--bench-quant-matvec TYPE ROWS COLS ITERS [REPEATS]]\n");
            fprintf(stderr, "       [--verify-moe-routing]\n");
            fprintf(stderr, "       [--verify-glm5next-kda [HEAD_DIM]]\n");
            fprintf(stderr, "       [--verify-glm5next-dsa]\n");
            fprintf(stderr, "       [--verify-glm5next-model]\n");
            fprintf(stderr, "       [--verify-glm5next-projections [LAYER]]\n");
            fprintf(stderr, "       [--verify-glm5next-kda-layer [LAYER]]\n");
            fprintf(stderr, "       [--verify-glm5next-dsa-layer [LAYER]]\n");
            fprintf(stderr, "       [--inspect-qwen4-nextn SIDEcar.gguf]\n");
            fprintf(stderr, "       [--load-qwen4-nextn-fusion SIDEcar.gguf]\n");
            return 1;
        }
    }

    /* --verify-quant-kernels has no model dependency; run it and exit. */
    if (verify_quant_kernels) {
        return run_verify_quant_kernels();
    }
    if (verify_moe_routing) {
        return run_verify_moe_routing();
    }
    if (inspect_qwen4_nextn) {
        gguf_shards *sidecar = gguf_open_shards(inspect_qwen4_nextn, 2);
        hip_llm_qwen4_nextn_info info;
        char error[192];
        int rc = sidecar ? hip_llm_qwen4_nextn_inspect(sidecar, &info,
            error, sizeof(error)) : -1;
        if (rc == 0) {
            fprintf(stderr, "Qwen4 NextN sidecar: blk=%d hidden=%d heads=%d/%d "
                    "head_dim=%d experts=%d/%d expert_ff=%d hc=%d@%d PASS\n",
                    info.layer_index, info.n_embd, info.n_heads, info.n_kv_heads,
                    info.head_dim, info.n_experts, info.n_experts_used,
                    info.expert_ff, info.hc_count, info.hc_low_rank);
        } else {
            fprintf(stderr, "Qwen4 NextN sidecar: FAIL (%s)\n",
                    sidecar ? error : "could not open GGUF");
        }
        if (sidecar) gguf_close_shards(sidecar);
        return rc == 0 ? 0 : 1;
    }
    if (verify_glm5next_kda) {
        hip_llm_runner *r = hip_llm_init(0, 1);
        if (!r) return 1;
        double rel = 0.0, max_abs = 0.0;
        int rc = hip_llm_verify_glm5next_kda_heads(r, 4, glm5next_kda_dim, &rel, &max_abs);
        fprintf(stderr, "GLM5Next KDA verify: heads=4 head_dim=%d rel_l2=%.6e max_abs=%.6e %s\n",
                glm5next_kda_dim, rel, max_abs,
                rc == 0 && rel < 1e-6 ? "PASS" : "FAIL");
        hip_llm_free(r);
        return rc == 0 && rel < 1e-6 ? 0 : 1;
    }
    if (verify_glm5next_dsa) {
        hip_llm_runner *r = hip_llm_init(0, 1);
        if (!r) return 1;
        double rel = 0.0, max_abs = 0.0;
        int rc = hip_llm_verify_glm5next_dsa_attention(r, 4, 512, 256, 5, &rel, &max_abs);
        fprintf(stderr, "GLM5Next DSA verify: heads=4 kv=512 value=256 tokens=5 rel_l2=%.6e max_abs=%.6e %s\n",
                rel, max_abs, rc == 0 && rel < 1e-6 ? "PASS" : "FAIL");
        hip_llm_free(r);
        return rc == 0 && rel < 1e-6 ? 0 : 1;
    }
    if (verify_glm5next_projections) {
        if (!model_path) {
            fprintf(stderr, "--verify-glm5next-projections requires a model path\n");
            return 2;
        }
        gguf_shards *model = gguf_open_shards(model_path, 2);
        hip_llm_runner *r = hip_llm_init(0, 1);
        double rel = 0.0, max_abs = 0.0, ms = 0.0;
        int rc = model && r ? hip_llm_verify_glm5next_model_kda_projections(
            r, model, glm5next_verify_layer, &rel, &max_abs, &ms) : -1;
        fprintf(stderr, "GLM5Next KDA projections: layer=%d rel_l2=%.6e max_abs=%.6e "
                        "launch_ms=%.3f %s\n", glm5next_verify_layer, rel, max_abs,
                        ms, rc == 0 && rel < 1e-6 ? "PASS" : "FAIL");
        if (r) hip_llm_free(r);
        if (model) gguf_close_shards(model);
        return rc == 0 && rel < 1e-6 ? 0 : 1;
    }
    if (verify_glm5next_kda_layer) {
        if (!model_path) {
            fprintf(stderr, "--verify-glm5next-kda-layer requires a model path\n");
            return 2;
        }
        gguf_shards *model = gguf_open_shards(model_path, 2);
        hip_llm_runner *r = hip_llm_init(0, 1);
        double rel = 0.0, max_abs = 0.0, ms = 0.0;
        int rc = model && r ? hip_llm_verify_glm5next_model_kda_layer(
            r, model, glm5next_verify_layer, &rel, &max_abs, &ms) : -1;
        fprintf(stderr, "GLM5Next KDA layer: layer=%d rel_l2=%.6e max_abs=%.6e "
                        "launch_ms=%.3f %s\n", glm5next_verify_layer, rel, max_abs,
                        ms, rc == 0 && rel < 1e-5 ? "PASS" : "FAIL");
        if (r) hip_llm_free(r);
        if (model) gguf_close_shards(model);
        return rc == 0 && rel < 1e-5 ? 0 : 1;
    }
    if (verify_glm5next_dsa_layer) {
        if (!model_path) {
            fprintf(stderr, "--verify-glm5next-dsa-layer requires a model path\n");
            return 2;
        }
        gguf_shards *model = gguf_open_shards(model_path, 2);
        hip_llm_runner *r = hip_llm_init(0, 1);
        double rel = 0.0, max_abs = 0.0, ms = 0.0;
        int rc = model && r ? hip_llm_verify_glm5next_model_dsa_layer(
            r, model, glm5next_verify_layer, &rel, &max_abs, &ms) : -1;
        fprintf(stderr, "GLM5Next DSA layer: layer=%d rel_l2=%.6e max_abs=%.6e "
                        "launch_ms=%.3f %s\n", glm5next_verify_layer, rel, max_abs,
                        ms, rc == 0 && rel < 1e-5 ? "PASS" : "FAIL");
        if (r) hip_llm_free(r);
        if (model) gguf_close_shards(model);
        return rc == 0 && rel < 1e-5 ? 0 : 1;
    }
    if (bench_qmv_type) {
        return run_bench_quant_matvec(bench_qmv_type, bench_qmv_rows, bench_qmv_cols,
                                      bench_qmv_iters, bench_qmv_repeats);
    }

    if (!model_path) {
        fprintf(stderr, "Usage: %s <model.gguf> [-t \"prompt\"] [-n max_tokens] [-s max_seq_len]\n", argv[0]);
        fprintf(stderr, "       [--bench] [--gpu-only-bench] [--decode N] [--prefill-len M] [--prompt-file PATH]\n");
        fprintf(stderr, "       [--verify-quant-kernels]   (standalone; no model needed)\n");
        fprintf(stderr, "       [--verify-moe-routing]     (standalone; no model needed)\n");
        fprintf(stderr, "       [--verify-ple-split]       (real-model PLE/SSM phase-order check)\n");
        fprintf(stderr, "       [--verify-ssm-projections] (real-model native/scalar projection check)\n");
        fprintf(stderr, "       [--verify-moe-native]      (real-model router/shared-expert check)\n");
        fprintf(stderr, "       [--verify-glm5next-kda [HEAD_DIM]] (standalone)\n");
        fprintf(stderr, "       [--verify-glm5next-dsa]            (standalone)\n");
        fprintf(stderr, "       [--bench-quant-matvec TYPE ROWS COLS ITERS [REPEATS]]   (standalone)\n");
        return 1;
    }

    if (qwen4_mtp_trust) {
        if (!qwen4_mtp || qwen4_exact) {
            fprintf(stderr, "--qwen4-mtp-trust-draft requires approximate --qwen4-mtp\n");
            return 2;
        }
        setenv("LLM_QWEN4_MTP_APPROX", "1", 1);
        setenv("LLM_QWEN4_MTP_TRUST_DRAFT", "1", 1);
        fprintf(stderr, "WARNING: trusted MTP is approximate; target verification is skipped and output quality is not guaranteed\n");
        /* Keep the standalone trusted profile consistent with the tuned ROCm
         * launcher, while preserving any caller-provided override. */
        if (!getenv("LLM_QWEN4_APPROX_DECODE")) setenv("LLM_QWEN4_APPROX_DECODE", "1", 0);
        if (!getenv("LLM_QWEN4_DEVICE_HITS_ONLY")) setenv("LLM_QWEN4_DEVICE_HITS_ONLY", "1", 0);
        if (!getenv("LLM_QWEN4_DEVICE_REFRESH_INTERVAL")) setenv("LLM_QWEN4_DEVICE_REFRESH_INTERVAL", "32", 0);
        if (!getenv("LLM_MOE_LFU_CACHE")) setenv("LLM_MOE_LFU_CACHE", "1", 0);
        if (!getenv("LLM_MOE_COPY_PIPELINE")) setenv("LLM_MOE_COPY_PIPELINE", "1", 0);
        if (!getenv("LLM_MOE_STREAM_SLOTS")) setenv("LLM_MOE_STREAM_SLOTS", "4", 0);
    }
    if (qwen4_mtp_adaptive) setenv("LLM_QWEN4_MTP_ADAPTIVE", "1", 1);

    /* Load GGUF */
    fprintf(stderr, "Loading GGUF: %s\n", model_path);
    gguf_shards *gguf_model = gguf_open_shards(model_path, 1);
    if (!gguf_model) {
        fprintf(stderr, "Failed to open GGUF file\n");
        return 1;
    }
    gguf_context *gguf = gguf_model->metadata;

    int arch_idx = gguf_find_key(gguf, "general.architecture");
    if (arch_idx >= 0 && gguf->kv[arch_idx].type == GGUF_TYPE_STRING &&
        strcmp(gguf->kv[arch_idx].value.str.str, "glm5next") == 0) {
        glm5next_config config;
        glm5next_state_layout layout;
        char error[160];
        int kda = 0, dsa = 0;
        memset(&config, 0, sizeof(config));
        if (glm5next_config_load(gguf, &config, error, sizeof(error)) != 0) {
            fprintf(stderr, "GLM5Next GGUF rejected: %s\n", error);
            gguf_close_shards(gguf_model);
            return 2;
        }
        if (glm5next_state_layout_compute(&config, max_seq_len, &layout) != 0) {
            fprintf(stderr, "GLM5Next state layout rejected\n");
            glm5next_config_free(&config);
            gguf_close_shards(gguf_model);
            return 2;
        }
        for (int l = 0; l < config.n_layers; ++l) {
            if (glm5next_layer_type(&config, l) == GLM5NEXT_LAYER_KDA) ++kda;
            else ++dsa;
        }
        fprintf(stderr,
                "GLM5Next detected: layers=%d (+%d NextN), hidden=%d, "
                "KDA=%d, DSA=%d, experts=%d/%d, indexer=%d/kpool%d, "
                "state@%d=%.1f MiB\n",
                config.n_layers, config.n_nextn_layers, config.hidden_size,
                kda, dsa, config.expert_count, config.expert_used_count,
                config.indexer_top_k, config.indexer_kpool, max_seq_len,
                (double)(layout.conv_bytes + layout.recurrent_bytes +
                         layout.latent_kv_bytes + layout.indexer_bytes +
                         layout.mhc_bytes) / (1024.0 * 1024.0));
        fprintf(stderr, "GLM5Next contract accepted; using the runner's CPU reference path\n");
        glm5next_config_free(&config);
    }
    if (arch_idx >= 0 && gguf->kv[arch_idx].type == GGUF_TYPE_STRING &&
        strcmp(gguf->kv[arch_idx].value.str.str, "deepseek4") == 0) {
        fprintf(stderr, "deepseek4 GGUF detected (%u split shards, %llu tensors), "
                "but the generic Qwen/Gemma HIP runner does not implement the "
                "DeepSeek4 graph; use the DS4F backend or add its tensor adapter.\n",
                gguf_model->n_shards, (unsigned long long)gguf->n_tensors);
        gguf_close_shards(gguf_model);
        return 2;
    }

    /* Load tokenizer */
    bpe_vocab *vocab = bpe_vocab_load(gguf);
    if (!vocab) {
        fprintf(stderr, "Failed to load vocab\n");
        gguf_close_shards(gguf_model);
        return 1;
    }
    fprintf(stderr, "Vocab: %d tokens\n", vocab->n_tokens);

    /* Tokenize prompt. Buffer sized to hold a large --prefill-len pad. */
    /* Long-context quality/stability runs must not silently truncate at the
     * historical 4096-token scratch capacity.  Reserve enough room for the
     * requested context and a small BOS/padding margin. */
    int tok_cap = max_seq_len > 4096 ? max_seq_len + 16 : 4096;
    if (prefill_pad > tok_cap - 16) tok_cap = prefill_pad + 16;
    int32_t *tokens = (int32_t *)malloc((size_t)tok_cap * sizeof(int32_t));
    if (!tokens) { fprintf(stderr, "tokens alloc failed\n"); return 1; }
    int n_tokens = bpe_tokenize(vocab, prompt, -1, tokens, tok_cap);
    if (n_tokens <= 0) {
        fprintf(stderr, "Tokenization failed\n");
        bpe_vocab_free(vocab);
        gguf_close_shards(gguf_model);
        return 1;
    }
    /* bpe_tokenize does not insert BOS; honor the model's insertion policy. */
    {
        int bos = prompt_bos_id(gguf);
        if (bos > 0 && (n_tokens == 0 || tokens[0] != bos)) {
            for (int i = n_tokens; i > 0; i--) tokens[i] = tokens[i-1];
            tokens[0] = bos; n_tokens++;
        }
    }
    fprintf(stderr, "Prompt: \"%s\" -> %d tokens:", prompt, n_tokens);
    for (int i = 0; i < n_tokens && i < 32; i++) {
        fprintf(stderr, " %d", tokens[i]);
    }
    if (n_tokens > 32) fprintf(stderr, " ...");
    fprintf(stderr, "\n");

    /* --prefill-len M: pad prompt by repeating last token until length == M */
    if (prefill_pad > n_tokens && prefill_pad <= tok_cap) {
        int32_t pad = tokens[n_tokens - 1];
        for (int i = n_tokens; i < prefill_pad; i++) tokens[i] = pad;
        n_tokens = prefill_pad;
        fprintf(stderr, "Prompt padded to %d tokens for bench\n", n_tokens);
    }

    /* In bench mode, default to using the full (possibly padded) prompt for prefill. */
    if (bench_mode && max_tokens == 8) max_tokens = n_tokens;
    if (max_tokens > n_tokens) max_tokens = n_tokens;

    /* Avoid loading a model-sized CPU shadow before reporting a missing AMD
     * device.  The launchers perform the same guard, but this binary is also
     * invoked directly by benchmarks and the HTTP backend. */
    if (access("/dev/kfd", R_OK | W_OK) != 0) {
        fprintf(stderr, "ROCm device unavailable: /dev/kfd is missing or inaccessible\n");
        fprintf(stderr, "Expose AMD KFD/render nodes before running test_hip_llm\n");
        return 1;
    }

    /* Load CPU reference model (may fail for MoE -- run GPU-only in that case) */
    transformer_model *cpu_model = NULL;
    if (gpu_only_bench || gpu_only) {
        gpu_only = 1;
        fprintf(stderr, "\n=== Skipping CPU reference (--%s) ===\n",
                gpu_only_bench ? "gpu-only-bench" : "gpu-only");
    } else {
        fprintf(stderr, "\n=== Loading CPU reference model ===\n");
        cpu_model = transformer_load(gguf, max_seq_len);
        if (!cpu_model) {
            fprintf(stderr, "CPU model load failed (MoE?), running GPU-only mode\n");
            gpu_only = 1;
        } else {
            const char *tf_threads = getenv("TF_THREADS");
            if (tf_threads && atoi(tf_threads) > 1) {
                int n = atoi(tf_threads);
                transformer_set_threads(cpu_model, n);
                fprintf(stderr, "CPU reference threads: %d (TF_THREADS)\n", n);
            }
        }
    }

    /* The Qwen4 NextN weights live in a standalone GGUF sidecar.  Attach and
     * materialize them before closing the sidecar so the CPU shadow context
     * can safely share the immutable tensor descriptors with the trunk. */
    if (cpu_model && load_qwen4_nextn_fusion && !qwen4_mtp && !verify_qwen4_nextn) {
        gguf_shards *nextn_sidecar = gguf_open_shards(load_qwen4_nextn_fusion, 2);
        char nextn_error[192];
        int nextn_rc = nextn_sidecar ? transformer_load_nextn_sidecar(
            cpu_model, nextn_sidecar, nextn_error, sizeof(nextn_error)) : -1;
        size_t nextn_bytes = nextn_rc == 0 ? transformer_materialize_nextn(cpu_model) : 0;
        fprintf(stderr, "CPU Qwen4 NextN sidecar: %s%s%s%s (%.3f GB resident)\n",
                nextn_rc == 0 && nextn_bytes != 0 ? "PASS" : "FAIL",
                nextn_rc == 0 ? "" : " (",
                nextn_rc == 0 ? "" : (nextn_sidecar ? nextn_error : "could not open sidecar"),
                nextn_rc == 0 ? "" : ")",
                (double)nextn_bytes / 1e9);
        if (nextn_sidecar) gguf_close_shards(nextn_sidecar);
        if (nextn_rc != 0 || nextn_bytes == 0) {
            transformer_free(cpu_model);
            cpu_model = NULL;
            gpu_only = 1;
        }
    }

    /* Initialize HIP runner */
    fprintf(stderr, "\n=== Initializing HIP runner ===\n");
    hip_llm_runner *gpu = hip_llm_init(0, 1);
    if (!gpu) {
        fprintf(stderr, "Failed to init HIP runner\n");
        if (cpu_model) transformer_free(cpu_model);
        bpe_vocab_free(vocab);
        gguf_close_shards(gguf_model);
        return 1;
    }
    if (getenv("LLM_DEBUG_LAYERS")) hip_llm_set_debug(gpu, 1);
    if (max_layers > 0) hip_llm_set_max_layers(gpu, max_layers);

    /* Load weights to GPU */
    fprintf(stderr, "\n=== Loading weights to GPU ===\n");
    hip_llm_load_options load_options;
    hip_llm_load_options_default(&load_options);
    load_options.max_seq_len = max_seq_len;
    if (moe_cache_mb > 0) load_options.moe_cache_bytes = (uint64_t)moe_cache_mb << 20;
    if (moe_cpu_only) load_options.moe_mode = HIP_LLM_MOE_CPU;
    if (qwen4_prefill_staging) load_options.qwen4_prefill_staging = 1;
    if (qwen4_prefill_stage_mb > 0)
        load_options.qwen4_prefill_stage_bytes = (uint64_t)qwen4_prefill_stage_mb << 20;
    load_options.kv_cache_type = kv_cache_type;
    load_options.prefill_batch_tokens = prefill_batch_tokens;
    load_options.qwen35_batched_prefill = qwen35_batched_prefill;
    load_options.qwen35_prefill_bf16 = qwen35_prefill_bf16;
    load_options.qwen35_decode_graph = qwen35_decode_graph;
    load_options.qwen35_reference_math = qwen35_reference_math;
    load_options.qwen35_native_q8_attention = qwen35_native_q8_attention;
    load_options.qwen35_native_q8_prefill = qwen35_native_q8_prefill;
    load_options.qwen35_native_q2k = qwen35_native_q2k;
    load_options.qwen35_native_mmvq = qwen35_native_mmvq;
    load_options.decode_kernel_mode = decode_kernel_mode;
    load_options.decode_layout_mode = decode_layout_mode;
    load_options.decode_layout_cache_path = decode_layout_cache_path;
    if (decode_layout_budget_mib > 0)
        load_options.decode_layout_budget_bytes = (uint64_t)decode_layout_budget_mib << 20;
    if (qwen4_batched_prefill) hip_llm_set_qwen4_batched_prefill(gpu, 1);
    if(qwen4_mtp) {
        hip_llm_qwen4_mtp_set_verify(gpu,qwen4_mtp_window);
        hip_llm_qwen4_mtp_configure(gpu,(size_t)qwen4_mtp_cache_mb<<20,qwen4_mtp_draft);
    }
    if (hip_llm_load_weights_sharded(gpu, gguf_model, &load_options) != 0) {
        fprintf(stderr, "Failed to load weights to GPU\n");
        hip_llm_free(gpu);
        if (cpu_model) transformer_free(cpu_model);
        bpe_vocab_free(vocab);
        gguf_close_shards(gguf_model);
        return 1;
    }
    if (qwen4_coding_profile) hip_llm_set_qwen4_coding_profile(gpu);
    if (qwen4_exact && hip_llm_qwen4_exact_enable(gpu)) {
        fprintf(stderr,"Qwen4 exact mode initialization failed\n");
        hip_llm_free(gpu);
        if(cpu_model)transformer_free(cpu_model);
        bpe_vocab_free(vocab);gguf_close_shards(gguf_model);return 1;
    }
    if (qwen35_mtp_path) {
        char error[192] = "invalid benchmark mode or draft width";
        if ((!bench_mode && !stdio_server) || qwen4_mtp ||
            qwen35_mtp_draft < 1 || qwen35_mtp_draft > 16 ||
            (qwen35_mtp_window && qwen35_mtp_draft > 15) ||
            (stdio_server && (!qwen35_mtp_window || !qwen35_batched_prefill ||
                              !qwen35_decode_graph ||
                              kv_cache_type != HIP_LLM_KV_Q8_0_Q8_0)) ||
            hip_llm_qwen35_mtp_load(gpu, qwen35_mtp_path, error, sizeof(error))) {
            fprintf(stderr, "Dense NextN load/configuration failed: %s\n", error);
            hip_llm_free(gpu);
            if (cpu_model) transformer_free(cpu_model);
            bpe_vocab_free(vocab); gguf_close_shards(gguf_model); return 1;
        }
        fprintf(stderr, "Dense NextN loaded: draft=%d, target verification enabled\n", qwen35_mtp_draft);
    }
    if (qwen35_dflash2_path) {
        char error[192] = "invalid benchmark mode or draft width";
        if ((!bench_mode && !stdio_server) || qwen4_mtp || qwen35_mtp_path ||
            qwen35_dflash2_draft < 1 || qwen35_dflash2_draft > 7 ||
            !qwen35_batched_prefill || !qwen35_decode_graph ||
            kv_cache_type != HIP_LLM_KV_Q8_0_Q8_0 ||
            hip_llm_qwen35_dflash2_load(gpu, qwen35_dflash2_path,
                                        error, sizeof(error))) {
            fprintf(stderr, "DFlash2 load/configuration failed: %s\n", error);
            hip_llm_free(gpu);
            if (cpu_model) transformer_free(cpu_model);
            bpe_vocab_free(vocab); gguf_close_shards(gguf_model); return 1;
        }
        fprintf(stderr, "DFlash2 loaded: draft=%d, exact target windows enabled "
                        "for greedy and sampled generation\n",
                        qwen35_dflash2_draft);
    }
    if (load_qwen4_nextn_fusion) {
        gguf_shards *sidecar = gguf_open_shards(load_qwen4_nextn_fusion, 2);
        char error[192];
        int rc = sidecar ? hip_llm_load_qwen4_nextn_fusion(gpu, sidecar,
            error, sizeof(error)) : -1;
        fprintf(stderr, "Qwen4 NextN fusion load: %s%s%s\n", rc == 0 ? "PASS" : "FAIL",
                rc == 0 ? "" : " (", rc == 0 ? "" : (sidecar ? error : "could not open sidecar"));
        if (rc != 0) fprintf(stderr, ")\n");
        if (rc == 0 && verify_qwen4_nextn)
            rc = hip_llm_verify_qwen4_nextn(gpu, sidecar, gguf_model, qwen4_mtp_draft);
        if (sidecar) gguf_close_shards(sidecar);
        if (rc == 0 && qwen4_mtp) rc = hip_llm_qwen4_mtp_enable(gpu);
        if (rc == 0 && qwen4_mtp) rc = hip_llm_qwen4_mtp_set_verify(gpu,qwen4_mtp_window);
        if (rc != 0 || !qwen4_mtp) {
            hip_llm_free(gpu);
            if (cpu_model) transformer_free(cpu_model);
            bpe_vocab_free(vocab); gguf_close_shards(gguf_model);
            return rc == 0 ? 0 : 1;
        }
    }
    if (verify_qwen4_qsa) {
        int rc=hip_llm_verify_qwen4_qsa(gpu);
        fprintf(stderr, "Qwen4 QSA verifier: %s (rc=%d)\n",
                rc ? "FAIL" : "PASS", rc);
        hip_llm_free(gpu);if(cpu_model)transformer_free(cpu_model);
        bpe_vocab_free(vocab);gguf_close_shards(gguf_model);return rc?1:0;
    }
    if (verify_glm5next_model) {
        double rel = 0.0, max_abs = 0.0;
        int rc = hip_llm_verify_glm5next_model_matvec(gpu, gguf_model, 0, &rel, &max_abs);
        fprintf(stderr, "GLM5Next real KDA matvec verify: layer=0 rel_l2=%.6e max_abs=%.6e %s\n",
                rel, max_abs, rc == 0 && rel < 5e-4 ? "PASS" : "FAIL");
        hip_llm_free(gpu);
        if (cpu_model) transformer_free(cpu_model);
        bpe_vocab_free(vocab); gguf_close_shards(gguf_model);
        return rc == 0 && rel < 5e-4 ? 0 : 1;
    }
    if (verify_ple_split) {
        int rc = hip_llm_verify_qwen4_ple_split(gpu, 8);
        fprintf(stderr, "PLE phase-order: HC outputs + PLE/SSM state bitwise %s\n", rc ? "FAIL" : "PASS");
        hip_llm_free(gpu);
        if (cpu_model) transformer_free(cpu_model);
        bpe_vocab_free(vocab); gguf_close_shards(gguf_model);
        return rc ? 1 : 0;
    }
    if (verify_moe_native) {
        int rc = hip_llm_verify_moe_native(gpu, 8);
        hip_llm_free(gpu);
        if (cpu_model) transformer_free(cpu_model);
        bpe_vocab_free(vocab); gguf_close_shards(gguf_model);
        return rc ? 1 : 0;
    }
    if (verify_ssm_projections) {
        int rc = hip_llm_verify_ssm_projections(gpu, 8);
        hip_llm_free(gpu);
        if (cpu_model) transformer_free(cpu_model);
        bpe_vocab_free(vocab); gguf_close_shards(gguf_model);
        return rc ? 1 : 0;
    }
    if (verify_hc_batch) {
        double rel = 0.0, max_abs = 0.0;
        int rc = hip_llm_verify_hc_batch(gpu, 8, &rel, &max_abs);
        fprintf(stderr, "HC batch verify: rel_l2=%.6e max_abs=%.6e %s\n",
                rel, max_abs, rc == 0 && rel < 2e-2 ? "PASS" : "FAIL");
        hip_llm_free(gpu);
        if (cpu_model) transformer_free(cpu_model);
        bpe_vocab_free(vocab);
        gguf_close_shards(gguf_model);
        return rc == 0 && rel < 2e-2 ? 0 : 1;
    }

    int n_embd = hip_llm_n_embd(gpu);
    int n_vocab = hip_llm_n_vocab(gpu);
    int n_max_seq = hip_llm_max_seq_len(gpu);
    if (qwen35_snapshot_max_tokens > 0)
        hip_llm_set_qwen35_snapshot_max_tokens(gpu, qwen35_snapshot_max_tokens);
    int pass = 1;

    if (stdio_server) {
        int bos = prompt_bos_id(gguf);
        pass = run_stdio_server(gpu, vocab, n_vocab, n_max_seq, bos,
                                qwen4_mtp ? qwen4_mtp_draft : 0,
                                qwen35_mtp_path ? qwen35_mtp_draft : 0,
                                qwen35_mtp_window,
                                qwen35_dflash2_path ? qwen35_dflash2_draft : 0,
                                coding_mode, reference_sampling ? &sampling : NULL,
                                context_cache_entries,
                                (size_t)context_cache_max_mib << 20) == 0;
        hip_llm_free(gpu);
        if (cpu_model) transformer_free(cpu_model);
        bpe_vocab_free(vocab);
        gguf_close_shards(gguf_model);
        return pass ? 0 : 1;
    }

    if (bench_mode) {
        unsigned char *seen = NULL;
        hllm_sampler *sampler = NULL;
        hllm_generation_trace trace = {0};
        int32_t *depth_tokens = NULL;
        hip_llm_state_snapshot *depth_snapshot = NULL;
        if ((reference_sampling && (coding_mode || qwen4_mtp)) || (trace_prefix && qwen4_mtp)) {
            fprintf(stderr, "Reference sampler cannot use coding filters or Qwen4 MTP\n");
            pass = 0; goto bench_done;
        }
        if (bench_repeat < 1) bench_repeat = 1;
        {
            int n_prefill = max_tokens > 0 ? max_tokens : 1;
            if (bench_depth + n_prefill > n_max_seq) {
                fprintf(stderr, "Benchmark depth and prompt exceed max_seq_len=%d\n", n_max_seq);
                pass = 0; goto bench_done;
            }
            if (bench_depth + n_prefill + decode_n > n_max_seq) {
                decode_n = n_max_seq - bench_depth - n_prefill;
                if (decode_n < 0) decode_n = 0;
                fprintf(stderr, "Clamped decode to %d (max_seq_len=%d)\n", decode_n, n_max_seq);
            }
        }
        /* Optional throwaway prefill + reset before the measured repeats.  Some
         * paths (Qwen4 grouped staging, VRAM-tight profiles) differ on their
         * first invocation only; warming them makes the measured repeats
         * describe steady-state execution. */
        const char *warmup_env = getenv("LLM_BENCH_WARMUP");
        if (bench_repeat > 1 && warmup_env && atoi(warmup_env) != 0) {
            fprintf(stderr, "hip_llm: explicit benchmark warmup\n");
            int wn = max_tokens > 0 ? max_tokens : 1;
            if (wn + 1 > n_max_seq) wn = n_max_seq > 1 ? n_max_seq - 1 : 1;
            hip_llm_forward_batch_logits(gpu, tokens, wn, 0);
            hip_llm_reset_state(gpu);
        }
        if (bench_depth > 0) {
            const unsigned depth_seed = 1;
            int depth_chunk = prefill_batch_tokens > 0 ? prefill_batch_tokens : 512;
            if (depth_chunk > bench_depth) depth_chunk = bench_depth;
            depth_tokens = (int32_t *)malloc((size_t)bench_depth * sizeof(*depth_tokens));
            if (!depth_tokens) {
                fprintf(stderr, "Random depth token allocation failed\n");
                pass = 0; goto bench_done;
            }
            srand(depth_seed);
            int bos = prompt_bos_id(gguf);
            for (int i = 0; i < bench_depth; ++i)
                depth_tokens[i] = i == 0 && bos >= 0 ? bos : rand() % n_vocab;
            uint64_t depth_hash = 1469598103934665603ULL;
            for (int i = 0; i < bench_depth; ++i) {
                depth_hash ^= (uint32_t)depth_tokens[i];
                depth_hash *= 1099511628211ULL;
            }
            hip_llm_reset_state(gpu);
            hip_llm_set_decode_mode(gpu, 0);
            hip_llm_set_qwen4_batch_request_tokens(gpu, bench_depth);
            double depth_start = get_time_ms();
            for (int off = 0; off < bench_depth; off += depth_chunk) {
                int count = bench_depth - off;
                if (count > depth_chunk) count = depth_chunk;
                if (!hip_llm_forward_batch(gpu, depth_tokens + off, count, off)) {
                    fprintf(stderr, "Random depth prefill failed at %d/%d\n", off, bench_depth);
                    pass = 0; goto bench_done;
                }
                if ((off + count) % 8192 == 0 || off + count == bench_depth)
                    fprintf(stderr, "Depth prefill: %d/%d tokens\n", off + count, bench_depth);
            }
            depth_snapshot = hip_llm_snapshot_state(gpu);
            if (!depth_snapshot) {
                fprintf(stderr, "Failed to snapshot random depth state\n");
                pass = 0; goto bench_done;
            }
            double depth_ms = get_time_ms() - depth_start;
            fprintf(stderr,
                "Depth prefill: %d random tokens in %.2f ms -> %.2f tok/s "
                "(seed=%u hash=%016llx)\n",
                bench_depth, depth_ms,
                depth_ms > 0.0 ? 1000.0 * bench_depth / depth_ms : 0.0,
                depth_seed, (unsigned long long)depth_hash);
        }
        for (int bench_rep = 0; bench_rep < bench_repeat; bench_rep++) {
        if (bench_repeat > 1)
            fprintf(stderr, "\n=== Bench repeat %d/%d ===\n", bench_rep + 1, bench_repeat);
        /* Restore the random depth's recurrent state before every repeat.
         * Prefix KV rows remain resident and measured rows are overwritten at
         * identical positions, matching llama-bench's cached-depth restore. */
        if (depth_snapshot) {
            if (hip_llm_restore_state(gpu, depth_snapshot)) {
                fprintf(stderr, "Failed to restore random depth state\n");
                pass = 0; goto bench_done;
            }
        } else {
            hip_llm_reset_state(gpu);
        }
        free(seen); seen = NULL;
        hllm_sampler_free(sampler); sampler = NULL;
        hllm_trace_close(&trace);
        if (hllm_trace_open(&trace, trace_prefix, bench_rep)) { pass = 0; goto bench_done; }
        if (reference_sampling) {
            sampler = hllm_sampler_create(&sampling, n_vocab);
            if (!sampler) { fprintf(stderr, "Invalid sampler configuration\n"); pass = 0; goto bench_done; }
        }
        /* ---- Bench mode: split prefill and decode tokens/sec ---- */
        int n_prefill = max_tokens;
        if (n_prefill < 1) n_prefill = 1;
        if (decode_n < 0) decode_n = 0;
        if (bench_depth && !bench_mode) {
            fprintf(stderr, "--bench-depth requires benchmark mode\n"); pass = 0; goto bench_done;
        }

        if (trace_prefix && bench_rep == 0) {
            char path[4096];
            int n = snprintf(path, sizeof(path), "%s.prompt.tokens", trace_prefix);
            FILE *f = n > 0 && n < (int)sizeof(path) ? fopen(path, "wb") : NULL;
            if (!f) { pass = 0; goto bench_done; }
            for (int i = 0; i < n_prefill; ++i) fprintf(f, "%d\n", tokens[i]);
            if (fclose(f)) { pass = 0; goto bench_done; }
        }
        /* Keep the benchmark harness aligned with the server's Qwen4
         * stateful-batch contract.  The runner uses the published request
         * length to distinguish a single tile (safe to batch) from a
         * multi-chunk request (scalar fallback unless explicitly forced).
         * Without this setter every benchmark looked like an unknown long
         * request and silently measured the scalar path. */
        hip_llm_set_qwen4_batch_request_tokens(gpu, n_prefill);

        fprintf(stderr, "\n=== Bench: depth=%d, prefill=%d tokens, decode=%d tokens, n_embd=%d, n_vocab=%d ===\n",
                bench_depth, n_prefill, decode_n, n_embd, n_vocab);

        if (compare_paths && hip_llm_batched_path_available(gpu)) {
            /* Run prefill via per-token path */
            hip_llm_set_batched_path(gpu, 0);
            float *log_p = hip_llm_forward_batch_logits(gpu, tokens, n_prefill, bench_depth);
            if (!log_p) { fprintf(stderr, "compare: per-token path failed\n"); pass = 0; goto bench_done; }
            float *buf_p = (float *)malloc((size_t)n_vocab * sizeof(float));
            memcpy(buf_p, log_p, (size_t)n_vocab * sizeof(float));

            /* Run prefill via batched path */
            if (depth_snapshot) {
                if (hip_llm_restore_state(gpu, depth_snapshot)) {
                    fprintf(stderr, "compare: failed to restore depth state\n");
                    free(buf_p); pass = 0; goto bench_done;
                }
            } else {
                hip_llm_reset_state(gpu);
            }
            hip_llm_set_batched_path(gpu, 1);
            float *log_b = hip_llm_forward_batch_logits(gpu, tokens, n_prefill, bench_depth);
            if (!log_b) { fprintf(stderr, "compare: batched path failed\n"); free(buf_p); pass = 0; goto bench_done; }

            double diff_sq = 0.0, ref_sq = 0.0, batch_sq = 0.0;
            float max_abs = 0.0f;
            int finite_p = 0, finite_b = 0;
            for (int i = 0; i < n_vocab; i++) {
                float d = log_b[i] - buf_p[i];
                diff_sq += d * d;
                ref_sq += buf_p[i] * buf_p[i];
                batch_sq += log_b[i] * log_b[i];
                finite_p += isfinite(buf_p[i]) != 0;
                finite_b += isfinite(log_b[i]) != 0;
                if (fabsf(d) > max_abs) max_abs = fabsf(d);
            }
            double rl2 = (ref_sq > 1e-12) ? sqrt(diff_sq / ref_sq) : sqrt(diff_sq);
            int top_p = argmax_logits(buf_p, n_vocab);
            int top_b = argmax_logits(log_b, n_vocab);
            fprintf(stderr,
                "[--compare-paths] rel_l2=%.4e max_abs=%.4e l2=(%.4e,%.4e) "
                "finite=(%d,%d) argmax: per-token=%d batched=%d %s\n",
                rl2, max_abs, sqrt(ref_sq), sqrt(batch_sq), finite_p, finite_b,
                top_p, top_b, (top_p == top_b) ? "(match)" : "(DIFFER)");
            free(buf_p);
        }

        /* Warm-up prefill (first call builds hipBLASLt plans; not counted). */
        const char *warmup_env = getenv("LLM_PREFILL_WARMUP");
        int n_warmup = warmup_env ? atoi(warmup_env) : 0;
        for (int w = 0; w < n_warmup; w++) {
            float *lg = hip_llm_forward_batch_logits(gpu, tokens, n_prefill, bench_depth);
            if (!lg) { fprintf(stderr, "GPU prefill warmup %d failed\n", w); pass = 0; goto bench_done; }
        }
        if (compare_paths || n_warmup) {
            if (depth_snapshot) {
                if (hip_llm_restore_state(gpu, depth_snapshot)) {
                    fprintf(stderr, "Failed to restore depth state after warmup\n");
                    pass = 0; goto bench_done;
                }
            } else {
                hip_llm_reset_state(gpu);
            }
        }

        /* Prefill: normally one forward_batch_logits call.  The optional
         * LLM_BENCH_STREAM_CHUNK mode models llama-server-style streamed
         * prefill: each bounded chunk is submitted as its own validated batch
         * call, while recurrent/KV state carries across calls.  This avoids
         * asking the Qwen4 batched dispatcher to split one oversized request
         * internally (that multi-chunk path is not safe on gfx1201). */
        hip_llm_reset_moe_stats(gpu);
        hip_llm_set_decode_mode(gpu, 0);
        double t_pf0 = get_time_ms();
        float *last_logits = NULL;
        int stream_chunk = 0;
        const char *stream_chunk_env = getenv("LLM_BENCH_STREAM_CHUNK");
        if (stream_chunk_env) stream_chunk = atoi(stream_chunk_env);
        if (!stream_chunk_env && n_prefill >= 2048) {
            /* Keep direct test_hip_llm invocations on the same bounded
             * large-request path as the serving launcher.  A single 4K+
             * dispatch can exhaust gfx1201 scratch before quality or timing
             * is even observable. */
            stream_chunk = 512;
            fprintf(stderr, "Large prefill: defaulting to streamed chunks (%d tokens)\n",
                    stream_chunk);
        }
        if (stream_chunk < 1 || stream_chunk >= n_prefill) stream_chunk = 0;
        if (stream_chunk > 0) {
            const char *batch_env = getenv("LLM_QWEN4_BATCH");
            const char *publish_chunk_env = getenv("LLM_BENCH_STREAM_PUBLISH_CHUNK");
            int publish_chunk = publish_chunk_env && atoi(publish_chunk_env) != 0;
            /* No batch knob means the runner's parity-safe scalar default. */
            int scalar_stream = (!batch_env || atoi(batch_env) == 0) &&
                                !qwen35_batched_prefill;
            if (scalar_stream)
                fprintf(stderr, "Large prefill: scalar streamed forward (dispatcher bypass)\n");
            for (int off = 0; off < n_prefill; off += stream_chunk) {
                int cc = n_prefill - off;
                if (cc > stream_chunk) cc = stream_chunk;
                /* The overlap path is guarded by the published request
                 * length.  Serving-shaped chunk publication keeps a long
                 * request on the validated 1K window instead of enabling a
                 * single unsafe 4K pipeline reservation. */
                if (publish_chunk)
                    hip_llm_set_qwen4_batch_request_tokens(gpu, cc);
                if (scalar_stream) {
                    for (int i = 0; i < cc; ++i) {
                        /* Match hip_llm_forward_batch_logits fallback:
                         * intermediate prefill tokens only advance the hidden
                         * state; materialize the vocab head once at the final
                         * token.  Computing 248K logits for every prompt
                         * token dominated the supposedly safe path. */
                        int is_last = (off + i + 1 == n_prefill);
                        last_logits = is_last ?
                            hip_llm_forward_logits(gpu, tokens[off + i], bench_depth + off + i) :
                            hip_llm_forward(gpu, tokens[off + i], bench_depth + off + i);
                        if (!last_logits) break;
                    }
                } else {
                    last_logits = hip_llm_forward_batch_logits(gpu, tokens + off, cc, bench_depth + off);
                }
                if (!last_logits) break;
            }
        } else {
            last_logits = hip_llm_forward_batch_logits(gpu, tokens, n_prefill, bench_depth);
        }
        if (!last_logits) { fprintf(stderr, "GPU forward_batch_logits failed\n"); pass = 0; goto bench_done; }
        double t_pf1 = get_time_ms();
        dump_top_logits(last_logits, n_vocab);
        const char *logits_path = getenv("LLM_LOGITS_PATH");
        if (logits_path && *logits_path) {
            FILE *lf = fopen(logits_path, "wb");
            if (lf) {
                fwrite(last_logits, sizeof(float), (size_t)n_vocab, lf);
                fclose(lf);
                fprintf(stderr, "Wrote logits: %s\n", logits_path);
            } else {
                fprintf(stderr, "Cannot write logits: %s\n", logits_path);
            }
        }
        double first_sample_start = get_time_ms();
        seen = coding_mode ? (unsigned char *)calloc((size_t)n_vocab, 1) : NULL;
        unsigned sample_rng = 0x51f15e5du;
        for (int i = 0; sampler && i < n_prefill; ++i) hllm_sampler_accept(sampler, tokens[i]);
        int next_tok = sampler ? hllm_sampler_sample(sampler, last_logits) : coding_mode ? sample_top_k_p_coding(last_logits, n_vocab, 20, 0.80f,
                                                    0.70f, 1.50f, 1.0f, 0.0f, seen,
                                                    &sample_rng, vocab)
                                   : argmax_logits(last_logits, n_vocab);
        double first_sample_ms = get_time_ms() - first_sample_start;
        int sampler_argmax = sampler && sampling.temperature == 0.0f &&
            sampling.repetition == 1.0f && sampling.presence == 0.0f && sampling.frequency == 0.0f;
        double prefill_ms = t_pf1 - t_pf0;
        double prefill_tps = (prefill_ms > 0.0) ? (1000.0 * n_prefill / prefill_ms) : 0.0;
        hip_llm_moe_stats prefill_moe = {0};
        hip_llm_get_moe_stats(gpu, &prefill_moe);

        /* Decode: greedy-sample decode_n tokens. */
        double decode_ms = 0.0, decode_tps = 0.0;
        int decoded = 0;
        int first_decode_tok = next_tok;
        if(qwen4_mtp_check && (!qwen4_mtp || coding_mode ||
            hip_llm_verify_qwen4_mtp(gpu,next_tok,bench_depth+n_prefill,qwen4_mtp_draft))) {
            fprintf(stderr,"Qwen4 MTP transaction check failed\n");pass=0;goto bench_done;
        }
        uint64_t decode_hash = 1469598103934665603ULL;
        if (decode_n > 0) {
            hip_llm_reset_moe_stats(gpu);
            hip_llm_set_decode_mode(gpu, 1);
            int gen_text = (getenv("LLM_GEN_TEXT") != NULL);
            int text_finished = 0;
            const int text_eos = bpe_eos_id(vocab);
            const int text_eot = bpe_eot_id(vocab);
            if (qwen35_dflash2_path && (coding_mode ||
                (sampler && !sampler_argmax)))
                fprintf(stderr, "DFLASH2 sampled verifier=exact-window\n");
            if (gen_text) fprintf(stderr, "\n=== Generated text ===\n");
            const char *finish_reason = "length";
            int selected = 0;
            int32_t dense_drafts[16];
            int dense_count = 0, dense_index = 0, dense_proposed = 0, dense_accepted = 0;
            int dense_dflash_longctx_notified = 0;
            double dense_draft_ms = 0, dense_verify_ms = 0, dense_commit_ms = 0;
            float *dense_logits = NULL;
            int32_t dense_argmax[16];
            int dense_window_rows = 0;
            const int dense_dflash2_configured = qwen35_dflash2_path != NULL;
            /* The verifier preserves scalar target logits for greedy,
             * probabilistic, and coding samplers. */
            const int dense_dflash2 = dense_dflash2_configured;
            int dense_dflash2_enabled = dense_dflash2;
            const int dense_path = qwen35_mtp_path != NULL || dense_dflash2;
            const int dense_window = qwen35_mtp_window || dense_dflash2;
            const int dense_draft_width = dense_dflash2 ? qwen35_dflash2_draft : qwen35_mtp_draft;
            const int dense_device_argmax = !coding_mode &&
                (!sampler || sampler_argmax) && !trace.logits;
            float *selection_logits = last_logits;
            int32_t stop_ids[3] = {text_eos, text_eot, -1};
            for (int id = 0; id < n_vocab; ++id) {
                const char *piece = bpe_token_to_str(vocab, id);
                if (piece && !strcmp(piece, "<|im_end|>")) { stop_ids[2] = id; break; }
            }
            {
                const char *lh = getenv("LLM_DECODE_LOGITS_HASH");
                decode_logits_hash_on = lh && atoi(lh) != 0;
                decode_logits_hash = 1469598103934665603ULL;
            }
            double t_dec0 = get_time_ms();
            int mtp_approx_fallback = 0;
            int mtp_adaptive_fallback = 0;
            int mtp_runtime_draft = qwen4_mtp_draft;
            const int mtp_adaptive_shrink =
                getenv("LLM_QWEN4_MTP_ADAPTIVE_SHRINK") &&
                atoi(getenv("LLM_QWEN4_MTP_ADAPTIVE_SHRINK")) != 0;
            for (int k = 0; k < decode_n; k++) {
                if (qwen4_mtp && !coding_mode && !mtp_approx_fallback && !mtp_adaptive_fallback) {
                    hip_llm_qwen4_mtp_result mtp;
                    if (hip_llm_qwen4_mtp_step(gpu, next_tok, bench_depth+n_prefill+k,
                            mtp_runtime_draft, decode_n-k,
                            bench_ignore_eos ? NULL : stop_ids, bench_ignore_eos ? 0 : 3, &mtp)) {
                        fprintf(stderr, "GPU MTP failed at decode k=%d\n", k); pass=0; break;
                    }
                    for (int i=0;i<mtp.emitted;++i) {
                        selected++;
                        int stop = is_generation_stop(vocab, mtp.tokens[i], text_eos, text_eot);
                        if (stop && !bench_ignore_eos) break;
                        decode_hash ^= (uint32_t)mtp.tokens[i]; decode_hash *= 1099511628211ULL;
                        decoded++;
                        if (gen_text && !text_finished) {
                            text_finished = is_generation_stop(
                                vocab, mtp.tokens[i], text_eos, text_eot);
                            if (!text_finished)
                                print_decoded_token(stderr, vocab, mtp.tokens[i]);
                        }
                    }
                    k += mtp.emitted-1; next_tok=mtp.pending;
                    if (mtp.stopped && !bench_ignore_eos) { finish_reason = "eos"; break; }
                    if (!mtp.emitted) { pass = 0; finish_reason = "error"; break; }
                    fprintf(stderr,"MTP backend=%s drafted=%d accepted=%d emitted=%d draft_ms=%.3f verify_ms=%.3f\n",
                            getenv("LLM_QWEN4_MTP_TRUST_DRAFT") ? "hip-approx" : "hip",
                            mtp.drafted,mtp.accepted,mtp.emitted,mtp.draft_ms,mtp.verify_ms);
                    if (getenv("LLM_QWEN4_MTP_APPROX") && mtp.drafted > 0 && mtp.accepted == 0) {
                        mtp_approx_fallback = 1;
                        fprintf(stderr, "MTP approximate acceptance=0; falling back to target decode\n");
                    }
                    if (getenv("LLM_QWEN4_MTP_ADAPTIVE") && mtp.drafted > 0 &&
                        mtp.accepted * 2 < mtp.drafted) {
                        if (mtp_adaptive_shrink && mtp_runtime_draft > 1) {
                            int old_width = mtp_runtime_draft;
                            mtp_runtime_draft = (mtp_runtime_draft + 1) / 2;
                            fprintf(stderr, "MTP acceptance=%d/%d; adaptive draft width %d->%d\n",
                                    mtp.accepted, mtp.drafted, old_width, mtp_runtime_draft);
                        } else {
                            mtp_adaptive_fallback = 1;
                            fprintf(stderr, "MTP acceptance=%d/%d; adaptive target fallback\n",
                                    mtp.accepted, mtp.drafted);
                        }
                    } else if (mtp_adaptive_shrink && mtp.drafted > 0 &&
                               mtp.accepted == mtp.drafted &&
                               mtp_runtime_draft < qwen4_mtp_draft) {
                        mtp_runtime_draft++;
                        fprintf(stderr, "MTP full acceptance; adaptive draft width ->%d\n",
                                mtp_runtime_draft);
                    }
                    continue;
                }
                if (next_tok < 0 || next_tok >= n_vocab ||
                    hllm_trace_token(&trace, next_tok, selection_logits, n_vocab)) {
                    pass = 0; finish_reason = "error"; break;
                }
                selected++;
                int stop = is_generation_stop(vocab, next_tok, text_eos, text_eot);
                if (stop && !bench_ignore_eos) { finish_reason = "eos"; break; }
                decode_hash ^= (uint32_t)next_tok;
                decode_hash *= 1099511628211ULL;
                if (!text_finished && !stop) {
                    if (gen_text) print_decoded_token(stderr, vocab, next_tok);
                    if (trace.bytes) print_decoded_token(trace.bytes, vocab, next_tok);
                }
                if (stop) text_finished = 1;
                decoded++;
                if (sampler) hllm_sampler_accept(sampler, next_tok);
                if (k + 1 == decode_n) break;
                int pos = bench_depth + n_prefill + k;
                if (dense_dflash2 && dense_dflash2_enabled && pos >= 32768) {
                    dense_dflash2_enabled = 0;
                    if (!dense_dflash_longctx_notified) {
                        dense_dflash_longctx_notified = 1;
                        fprintf(stderr,
                                "DFlash2 disabled at position %d; using "
                                "target decode for long context\n", pos);
                    }
                }
                if (dense_path && (!dense_dflash2 || dense_dflash2_enabled) &&
                    (dense_window ? !dense_window_rows : dense_index == dense_count)) {
                    dense_count = dense_draft_width;
                    int available = decode_n-k-1-dense_window;
                    if (dense_count > available) dense_count = available;
                    dense_index = 0;
                    double td = get_time_ms();
                    int propose_rc = 0;
                    if (dense_count > 0) propose_rc = dense_dflash2 ?
                        hip_llm_qwen35_dflash2_propose(gpu, next_tok, pos,
                                                       dense_count, dense_drafts) :
                        hip_llm_qwen35_mtp_propose(gpu, next_tok, pos,
                                                   dense_count, dense_drafts);
                    if (propose_rc) {
                        pass = 0; finish_reason = "error"; break;
                    }
                    dense_draft_ms += get_time_ms()-td;
                    dense_proposed += dense_count;
                    if (dense_window && dense_count > 0) {
                        int32_t inputs[16]; inputs[0] = next_tok;
                        memcpy(inputs+1, dense_drafts, (size_t)dense_count*sizeof(int32_t));
                        dense_window_rows = dense_count+1;
                        double tv = get_time_ms();
                        int verify_rc = 0;
                        if (dense_device_argmax)
                            verify_rc = hip_llm_qwen35_mtp_verify_argmax(gpu, inputs,
                                dense_window_rows, pos, dense_argmax);
                        else {
                            dense_logits = hip_llm_qwen35_mtp_verify(gpu, inputs,
                                dense_window_rows, pos);
                            verify_rc = dense_logits ? 0 : -1;
                        }
                        dense_verify_ms += get_time_ms()-tv;
                        if (verify_rc) { pass = 0; finish_reason = "error"; break; }
                    }
                }
                if (dense_window_rows) {
                    if (dense_device_argmax) next_tok = dense_argmax[dense_index];
                    else {
                        selection_logits = dense_logits + (size_t)dense_index*n_vocab;
                        if (seen) seen[next_tok] = 1;
                        next_tok = sampler ? hllm_sampler_sample(sampler, selection_logits) : coding_mode ?
                            sample_top_k_p_coding(selection_logits, n_vocab, 20, 0.80f,
                                0.70f, 1.50f, 1.0f, 0.0f, seen, &sample_rng, vocab) :
                            argmax_logits(selection_logits, n_vocab);
                    }
                } else if (!coding_mode && (!sampler || sampler_argmax) && !trace.logits &&
                           !decode_logits_hash_on) {
                    next_tok = hip_llm_forward_argmax(gpu, next_tok, pos);
                    if (next_tok < 0) { pass = 0; finish_reason = "error"; break; }
                } else {
                    selection_logits = hip_llm_forward_logits(gpu, next_tok, pos);
                    if (!selection_logits) { pass = 0; finish_reason = "error"; break; }
                    if (decode_logits_hash_on) {
                        /* Bitwise decode parity probe: FNV-1a over every
                         * step's full logits vector. */
                        const unsigned char *lb = (const unsigned char *)selection_logits;
                        for (size_t bi = 0; bi < (size_t)n_vocab * sizeof(float); ++bi)
                            decode_logits_hash = (decode_logits_hash ^ lb[bi]) *
                                                 1099511628211ULL;
                    }
                    if (seen) seen[next_tok] = 1;
                    next_tok = sampler ? hllm_sampler_sample(sampler, selection_logits) : coding_mode ?
                        sample_top_k_p_coding(selection_logits, n_vocab, 20, 0.80f,
                            0.70f, 1.50f, 1.0f, 0.0f, seen, &sample_rng, vocab) :
                        argmax_logits(selection_logits, n_vocab);
                }
                if (dense_window_rows) {
                    int matched = dense_index < dense_count && next_tok == dense_drafts[dense_index];
                    dense_index++;
                    if (matched) dense_accepted++;
                    if (!matched || dense_index == dense_window_rows) {
                        double tc = get_time_ms();
                        int commit_rc = dense_dflash2 ?
                            hip_llm_qwen35_dflash2_commit(gpu, pos-dense_index+1,
                                                          dense_index) :
                            hip_llm_qwen35_mtp_commit(gpu, dense_index);
                        dense_commit_ms += get_time_ms()-tc;
                        if (commit_rc) { pass=0; finish_reason="error"; break; }
                        dense_window_rows = dense_index = dense_count = 0;
                    }
                } else if (dense_path && dense_count > 0) {
                    if (next_tok == dense_drafts[dense_index++]) dense_accepted++;
                    else dense_count = dense_index = 0;
                }
            }
            if (dense_window_rows && dense_index > 0) {
                double tc = get_time_ms();
                int commit_rc = dense_dflash2 ?
                    hip_llm_qwen35_dflash2_commit(gpu,
                        bench_depth+n_prefill+decoded-dense_index,dense_index) :
                    hip_llm_qwen35_mtp_commit(gpu,dense_index);
                dense_commit_ms += get_time_ms()-tc;
                if (commit_rc) pass=0;
            }
            if (gen_text) fprintf(stderr, "\n=== end ===\n");
            if (dense_path)
                fprintf(stderr, "%s drafted=%d accepted=%d draft_ms=%.3f "
                        "verify_ms=%.3f commit_ms=%.3f verify=%s\n",
                        dense_dflash2 ? "DFLASH2" : "DENSE_MTP",
                        dense_proposed, dense_accepted, dense_draft_ms,
                        dense_verify_ms, dense_commit_ms,
                        dense_window ? "window" : "sequential");
            double t_dec1 = get_time_ms();
            fprintf(stderr, "GENERATION finish=%s selected=%d emitted=%d synthetic=%d\n",
                    finish_reason, selected, decoded, bench_ignore_eos);
            hip_llm_set_decode_mode(gpu, 0);
            decode_ms = t_dec1 - t_dec0 + first_sample_ms;
            decode_tps = (decode_ms > 0.0) ? (1000.0 * decoded / decode_ms) : 0.0;
        }

        /* CPU reference MTP smoke/transaction path.  This intentionally runs
         * beside the GPU benchmark until the GPU NextN kernels are available;
         * it proves sidecar loading, chaining, target rollback, and exact
         * greedy acceptance on the real model. */
        if (cpu_model && cpu_model->nextn.loaded && decode_n > 0) {
            transformer_model *nextn_ctx = transformer_nextn_context_create(cpu_model, 1);
            int mtp_ok = nextn_ctx != NULL;
            int32_t mtp_token = -1;
            if (mtp_ok) {
                for (int i = 0; i < n_prefill; i++) {
                    float *cpu_logits = transformer_forward_logits(cpu_model, tokens[i], i);
                    if (!cpu_logits) {
                        mtp_ok = 0; break;
                    }
                    if (i == n_prefill - 1)
                        mtp_token = argmax_logits(cpu_logits, n_vocab);
                }
            }
            int mtp_accepted = 0, mtp_emitted = 0;
            double mtp_ms = get_time_ms();
            if (mtp_ok) {
                int pos = n_prefill;
                for (int k = 0; k < decode_n; ) {
                    int accepted = 0;
                    int32_t replacement = -1;
                    int rc = transformer_nextn_speculate_greedy(
                        cpu_model, nextn_ctx, mtp_token, pos, qwen4_mtp_draft,
                        &accepted, &replacement);
                    if (rc < 0 || replacement < 0) { mtp_ok = 0; break; }
                    mtp_accepted += accepted;
                    mtp_emitted += accepted + (accepted < 4 ? 1 : 0);
                    mtp_token = replacement;
                    pos += accepted + (accepted < 4 ? 1 : 0);
                    k = pos - n_prefill;
                }
            }
            mtp_ms = get_time_ms() - mtp_ms;
            fprintf(stderr, "CPU MTP: %s accepted=%d emitted=%d draft=%d time=%.1f ms\n",
                    mtp_ok ? "PASS" : "FAIL", mtp_accepted, mtp_emitted,
                    qwen4_mtp_draft, mtp_ms);
            transformer_nextn_context_free(nextn_ctx);
            if (!mtp_ok) pass = 0;
        }

        fprintf(stderr, "\n=== Bench results ===\n");
        fprintf(stderr, "Prefill: %d tokens in %.2f ms  -> %.2f tok/s  (%.3f ms/tok)\n",
                n_prefill, prefill_ms, prefill_tps,
                n_prefill > 0 ? prefill_ms / n_prefill : 0.0);
        if (decode_n > 0) {
            fprintf(stderr, "Decode:  %d tokens in %.2f ms  -> %.2f tok/s  (%.3f ms/tok)\n",
                    decoded, decode_ms, decode_tps,
                    decoded > 0 ? decode_ms / decoded : 0.0);
            if (decode_logits_hash_on)
                fprintf(stderr, "Decode logits hash=%016llx\n",
                        (unsigned long long)decode_logits_hash);
            fprintf(stderr, "First decoded token id=%d, last id=%d, sequence hash=%016llx\n",
                    first_decode_tok, next_tok, (unsigned long long)decode_hash);
            double request_ms = prefill_ms + decode_ms;
            int request_tokens = n_prefill + decoded;
            fprintf(stderr, "End-to-end: %d prompt + %d generated tokens in %.2f ms  -> %.2f tok/s (prefill + decode, warm model)\n",
                    n_prefill, decoded, request_ms,
                    request_ms > 0.0 ? 1000.0 * request_tokens / request_ms : 0.0);
        }
        {
            hip_llm_moe_stats ms;
            int live_ok = hip_llm_get_moe_stats(gpu, &ms) == 0;
            int have_pf = prefill_moe.cache_hits + prefill_moe.cache_misses > 0;
            int have_live = live_ok && ms.cache_hits + ms.cache_misses > 0;
            /* Decode resets the live counters at its start. Keep the two
             * phases separate so a prefill miss burst cannot hide steady
             * decode residency (and vice versa). */
            if (have_pf || have_live) {
                if (have_pf) {
                    hip_llm_moe_stats pf = prefill_moe;
                    double hit = 100.0 * (double)pf.cache_hits /
                                 (double)(pf.cache_hits + pf.cache_misses);
                    fprintf(stderr, "MoE cache: %.1f%% hit (%llu/%llu), H2D %.2f GiB, stage %.2f GiB / %llu waves / %llu promotions / %llu fallbacks, CPU-time %.2f ms\n", hit,
                            (unsigned long long)pf.cache_hits,
                            (unsigned long long)(pf.cache_hits + pf.cache_misses),
                            pf.h2d_bytes / (double)(1ULL << 30),
                            pf.stage_h2d_bytes / (double)(1ULL << 30),
                            (unsigned long long)pf.stage_waves,
                            (unsigned long long)pf.stage_promotions,
                            (unsigned long long)pf.stage_fallbacks, pf.cpu_ms);
                }
                if (have_live && decode_n > 0) {
                    double hit = 100.0 * (double)ms.cache_hits /
                                 (double)(ms.cache_hits + ms.cache_misses);
                    fprintf(stderr, "MoE decode cache: %.1f%% hit (%llu/%llu), H2D %.2f GiB, stage %.2f GiB / %llu waves / %llu promotions / %llu fallbacks, CPU-time %.2f ms\n", hit,
                            (unsigned long long)ms.cache_hits,
                            (unsigned long long)(ms.cache_hits + ms.cache_misses),
                            ms.h2d_bytes / (double)(1ULL << 30),
                            ms.stage_h2d_bytes / (double)(1ULL << 30),
                            (unsigned long long)ms.stage_waves,
                            (unsigned long long)ms.stage_promotions,
                            (unsigned long long)ms.stage_fallbacks, ms.cpu_ms);
                }
                /* Preserve the historical single-line label for callers that
                 * only run a prefill or inspect the benchmark footer. */
                if (!have_pf && have_live) {
                double hit = 100.0 * (double)ms.cache_hits /
                             (double)(ms.cache_hits + ms.cache_misses);
                fprintf(stderr, "MoE cache: %.1f%% hit (%llu/%llu), H2D %.2f GiB, stage %.2f GiB / %llu waves / %llu promotions / %llu fallbacks, CPU-time %.2f ms\n", hit,
                        (unsigned long long)ms.cache_hits,
                        (unsigned long long)(ms.cache_hits + ms.cache_misses),
                        ms.h2d_bytes / (double)(1ULL << 30),
                        ms.stage_h2d_bytes / (double)(1ULL << 30),
                        (unsigned long long)ms.stage_waves,
                        (unsigned long long)ms.stage_promotions,
                        (unsigned long long)ms.stage_fallbacks, ms.cpu_ms);
                }
            }
        }
        {
            hip_llm_vram_stats vs = { .struct_size = sizeof(vs) };
            if (hip_llm_get_vram_stats(gpu, &vs) == 0)
                fprintf(stderr, "VRAM: free %.1f / %.1f MiB, peak used %.1f MiB\n",
                        vs.free_bytes / (double)(1ULL << 20),
                        vs.total_bytes / (double)(1ULL << 20),
                        vs.peak_used_bytes / (double)(1ULL << 20));
        }
        if (hllm_trace_close(&trace)) pass = 0;
        fprintf(stderr, "Result: %s\n", pass ? "PASS" : "FAIL");
        } /* bench_rep */
bench_done:
        hllm_trace_close(&trace);
        hllm_sampler_free(sampler);
        hip_llm_free_state_snapshot(depth_snapshot);
        free(depth_tokens);
        free(seen);
    } else {
        /* ---- Correctness mode: per-token CPU vs GPU compare (legacy) ---- */
        fprintf(stderr, "\n=== Running %d tokens (n_embd=%d)%s ===\n",
                max_tokens, n_embd, gpu_only ? " [GPU-only]" : "");

        double total_cpu_ms = 0.0, total_gpu_ms = 0.0;

        for (int i = 0; i < max_tokens; i++) {
            int32_t token = tokens[i];

            /* CPU forward (skip if GPU-only) */
            float *cpu_out = NULL;
            double cpu_ms = 0.0;
            if (!gpu_only) {
                double t0 = get_time_ms();
                cpu_out = transformer_forward(cpu_model, token, i);
                cpu_ms = get_time_ms() - t0;
                total_cpu_ms += cpu_ms;
            }

            /* GPU forward */
            double t0 = get_time_ms();
            float *gpu_out = hip_llm_forward(gpu, token, i);
            double gpu_ms = get_time_ms() - t0;
            total_gpu_ms += gpu_ms;

            if (!gpu_out) {
                fprintf(stderr, "Token %d: GPU forward failed\n", i);
                pass = 0;
                continue;
            }

            if (gpu_only) {
                fprintf(stderr, "\nToken %d (id=%d): GPU=%.1fms\n", i, token, gpu_ms);
                print_first_n("GPU", gpu_out, n_embd, 8);
            } else if (!cpu_out) {
                fprintf(stderr, "Token %d: CPU forward failed\n", i);
                pass = 0;
            } else {
                float err = rel_l2_error(gpu_out, cpu_out, n_embd);
                const char *status = (err < 1e-2f) ? "OK" : "MISMATCH";
                if (err >= 1e-2f) pass = 0;

                fprintf(stderr, "\nToken %d (id=%d): rel_L2=%.6f [%s]  CPU=%.1fms  GPU=%.1fms  (%.1fx)\n",
                        i, token, err, status, cpu_ms, gpu_ms,
                        gpu_ms > 0 ? cpu_ms / gpu_ms : 0.0);
                print_first_n("CPU", cpu_out, n_embd, 8);
                print_first_n("GPU", gpu_out, n_embd, 8);
            }
        }

        fprintf(stderr, "\n=== Summary ===\n");
        fprintf(stderr, "Tokens processed: %d\n", max_tokens);
        if (!gpu_only) {
            fprintf(stderr, "Total CPU time: %.1f ms (%.1f ms/token)\n",
                    total_cpu_ms, total_cpu_ms / max_tokens);
        }
        fprintf(stderr, "Total GPU time: %.1f ms (%.1f ms/token)\n",
                total_gpu_ms, total_gpu_ms / max_tokens);
        if (!gpu_only && total_gpu_ms > 0) {
            fprintf(stderr, "Speedup: %.1fx\n", total_cpu_ms / total_gpu_ms);
        }
        fprintf(stderr, "Result: %s\n", pass ? "PASS" : "FAIL");
    }

    /* Cleanup */
    hip_llm_free(gpu);
    if (cpu_model) transformer_free(cpu_model);
    bpe_vocab_free(vocab);
    gguf_close_shards(gguf_model);
    free(prompt_owned);

    return pass ? 0 : 1;
}
