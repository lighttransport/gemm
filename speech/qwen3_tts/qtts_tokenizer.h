/* SPDX-License-Identifier: MIT
 * Copyright 2026 - Present, Light Transport Entertainment Inc.
 *
 * qtts_tokenizer.h - Qwen2 byte-level BPE from vocab.json + merges.txt
 * (+ tokenizer_config.json added tokens), feeding common/bpe_tokenizer.h's
 * Unicode-aware Qwen2 pre-tokenizer/merge engine.
 *
 * The caller must include gguf_loader.h and bpe_tokenizer.h with
 * BPE_TOKENIZER_IMPLEMENTATION in the same TU (the bpe_vocab struct is used).
 * Note: HF Qwen2Tokenizer applies NFC normalization; inputs are expected to be NFC.
 */
#ifndef QTTS_TOKENIZER_H
#define QTTS_TOKENIZER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *qtok__read(const char *path, long *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = (char *)malloc((size_t)n + 1);
    if (fread(b, 1, (size_t)n, f) != (size_t)n) { free(b); fclose(f); return NULL; }
    b[n] = 0;
    fclose(f);
    if (len) *len = n;
    return b;
}

static int qtok__hex(int c) {
    return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
}

/* Parse a JSON string starting at *pp (pointing at the opening quote) into out (UTF-8). */
static int qtok__jstr(const char **pp, char *out, int cap) {
    const char *p = *pp;
    int n = 0;
    if (*p++ != '"') return -1;
    while (*p && *p != '"') {
        unsigned cp;
        if (*p == '\\') {
            p++;
            switch (*p) {
            case 'n': cp = '\n'; p++; break;
            case 't': cp = '\t'; p++; break;
            case 'r': cp = '\r'; p++; break;
            case 'b': cp = '\b'; p++; break;
            case 'f': cp = '\f'; p++; break;
            case 'u': {
                cp = 0;
                for (int i = 1; i <= 4; i++) cp = cp * 16 + (unsigned)qtok__hex(p[i]);
                p += 5;
                if (cp >= 0xD800 && cp < 0xDC00 && p[0] == '\\' && p[1] == 'u') {
                    unsigned lo = 0;
                    for (int i = 2; i <= 5; i++) lo = lo * 16 + (unsigned)qtok__hex(p[i]);
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    p += 6;
                }
                break;
            }
            default: cp = (unsigned char)*p++; break;
            }
            char tmp[4];
            int l = bpe_cpt_to_utf8(cp, tmp);
            if (n + l >= cap) return -1;
            memcpy(out + n, tmp, (size_t)l);
            n += l;
        } else {
            if (n + 1 >= cap) return -1;
            out[n++] = *p++;
        }
    }
    if (*p == '"') p++;
    out[n] = 0;
    *pp = p;
    return n;
}

/* dir: model directory containing vocab.json, merges.txt, tokenizer_config.json */
static bpe_vocab *qtts_tokenizer_load(const char *dir) {
    char path[1024], key[1024];
    long len = 0;
    snprintf(path, sizeof(path), "%s/vocab.json", dir);
    char *vj = qtok__read(path, &len);
    if (!vj) { fprintf(stderr, "qtts_tokenizer: cannot read %s\n", path); return NULL; }
    snprintf(path, sizeof(path), "%s/tokenizer_config.json", dir);
    char *tc = qtok__read(path, NULL);

    /* pass 1: max id among vocab + added tokens */
    int max_id = -1, n_vocab = 0;
    for (const char *p = vj; (p = strchr(p, '"')); ) {
        if (qtok__jstr(&p, key, sizeof(key)) < 0) break;
        while (*p == ' ' || *p == ':') p++;
        int id = (int)strtol(p, (char **)&p, 10);
        if (id > max_id) max_id = id;
        n_vocab++;
    }
    int n_added = 0;
    const char *added = tc ? strstr(tc, "\"added_tokens_decoder\"") : NULL;
    if (added) {
        for (const char *p = added; (p = strstr(p, "\"content\"")); p++) {
            n_added++;
        }
    }
    if (max_id < 151642 + n_added) max_id = 151642 + n_added;

    bpe_vocab *v = (bpe_vocab *)calloc(1, sizeof(*v));
    v->n_tokens = max_id + 1;
    v->token_strs = (char **)calloc((size_t)v->n_tokens, sizeof(char *));
    v->token_str_lens = (int *)calloc((size_t)v->n_tokens, sizeof(int));
    bpe_hm_init(&v->token_to_id, n_vocab * 2 + 64);
    for (const char *p = vj; (p = strchr(p, '"')); ) {
        int kl = qtok__jstr(&p, key, sizeof(key));
        if (kl < 0) break;
        while (*p == ' ' || *p == ':') p++;
        int id = (int)strtol(p, (char **)&p, 10);
        if (id < 0 || id >= v->n_tokens || v->token_strs[id]) continue;
        v->token_strs[id] = (char *)malloc((size_t)kl + 1);
        memcpy(v->token_strs[id], key, (size_t)kl + 1);
        v->token_str_lens[id] = kl;
        bpe_hm_set(&v->token_to_id, key, kl, id);
    }
    free(vj);

    snprintf(path, sizeof(path), "%s/merges.txt", dir);
    char *mt = qtok__read(path, &len);
    if (!mt) { fprintf(stderr, "qtts_tokenizer: cannot read %s\n", path); bpe_vocab_free(v); free(tc); return NULL; }
    int n_merges = 0;
    for (const char *p = mt; *p; p++) n_merges += *p == '\n';
    bpe_hm_init(&v->merge_ranks, n_merges * 2 + 64);
    int rank = 0;
    for (char *line = mt; line && *line; ) {
        char *nl = strchr(line, '\n');
        if (nl) *nl = 0;
        size_t ll = strlen(line);
        if (ll && line[ll - 1] == '\r') line[--ll] = 0;
        if (ll && strncmp(line, "#version", 8)) {
            char *sp = strchr(line, ' ');
            if (sp) {
                *sp = 0;  /* key = "left\0right" */
                bpe_hm_set(&v->merge_ranks, line, (int)ll, rank++);
            }
        }
        line = nl ? nl + 1 : NULL;
    }
    free(mt);

    /* added (special) tokens: "<id>": {"content": "..."} */
    v->special_ids = (int32_t *)calloc((size_t)n_added + 1, sizeof(int32_t));
    v->special_strs = (char **)calloc((size_t)n_added + 1, sizeof(char *));
    v->special_lens = (int *)calloc((size_t)n_added + 1, sizeof(int));
    if (added) {
        const char *p = strchr(added + 22, '{');
        while (p && (p = strchr(p, '"'))) {
            if (qtok__jstr(&p, key, sizeof(key)) < 0) break;
            int id = atoi(key);
            const char *c = strstr(p, "\"content\"");
            if (!c) break;
            c = strchr(c + 9, '"');
            int kl = qtok__jstr(&c, key, sizeof(key));
            if (kl < 0 || id < 0 || id >= v->n_tokens) break;
            if (!v->token_strs[id]) {
                v->token_strs[id] = (char *)malloc((size_t)kl + 1);
                memcpy(v->token_strs[id], key, (size_t)kl + 1);
                v->token_str_lens[id] = kl;
                bpe_hm_set(&v->token_to_id, key, kl, id);
            }
            v->special_ids[v->n_special] = id;
            v->special_strs[v->n_special] = v->token_strs[id];
            v->special_lens[v->n_special] = v->token_str_lens[id];
            v->n_special++;
            p = strchr(c, '}');
            if (!p) break;
            p++;
            while (*p == ' ' || *p == '\n' || *p == ',' || *p == '\r' || *p == '\t') p++;
            if (*p == '}') break;  /* end of added_tokens_decoder */
        }
    }
    free(tc);
    v->pre_type = BPE_PRE_TYPE_QWEN2;
    v->eos_id = v->eot_id = 151645;
    v->bos_id = v->pad_id = v->unk_id = -1;
    return v;
}

#endif /* QTTS_TOKENIZER_H */
