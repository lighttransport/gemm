/* Binary-safe generation evidence; no output normalization or EOS hiding. */
#ifndef HLLM_GENERATION_TRACE_H
#define HLLM_GENERATION_TRACE_H
#include <stdio.h>
#include <string.h>
typedef struct {
    FILE *tokens, *bytes, *logits;
} hllm_generation_trace;

static int hllm_trace_close(hllm_generation_trace *t) {
    int failed = 0;
    FILE *files[] = {t->tokens, t->bytes, t->logits};
    for (int i = 0; i < 3; ++i) if (files[i]) {
        failed |= ferror(files[i]) != 0;
        failed |= fclose(files[i]) != 0;
    }
    memset(t, 0, sizeof(*t));
    return failed ? -1 : 0;
}

static int hllm_trace_open(hllm_generation_trace *t, const char *prefix, int repeat) {
    memset(t, 0, sizeof(*t));
    if (!prefix) return 0;
    char path[4096];
    FILE **files[] = {&t->tokens, &t->bytes, &t->logits};
    const char *suffix[] = {"tokens", "output", "logits.f32"};
    for (int i = 0; i < 3; ++i) {
        int n = snprintf(path, sizeof(path), "%s.%d.%s", prefix, repeat, suffix[i]);
        if (n < 0 || n >= (int)sizeof(path) || !(*files[i] = fopen(path, "wb"))) {
            hllm_trace_close(t); return -1;
        }
    }
    return 0;
}

static int hllm_trace_token(hllm_generation_trace *t, int token,
                           const float *logits, int n_vocab) {
    if (t->tokens && fprintf(t->tokens, "%d\n", token) < 0) return -1;
    if (t->logits && (!logits || fwrite(logits, sizeof(float), (size_t)n_vocab,
                                       t->logits) != (size_t)n_vocab)) return -1;
    return 0;
}
#endif
