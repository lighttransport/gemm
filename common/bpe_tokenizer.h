/*
 * bpe_tokenizer.h - Single-file BPE tokenizer for Qwen2 models (loads from GGUF)
 *
 * Usage:
 *   #define BPE_TOKENIZER_IMPLEMENTATION
 *   #include "bpe_tokenizer.h"
 *
 * Dependencies: gguf_loader.h (must be included before, with implementation defined separately)
 *
 * API:
 *   bpe_vocab *bpe_vocab_load(gguf_context *gguf);
 *   void bpe_vocab_free(bpe_vocab *vocab);
 *   int bpe_tokenize(const bpe_vocab *vocab, const char *text, int text_len,
 *                    int32_t *tokens, int max_tokens);
 *   const char *bpe_token_to_str(const bpe_vocab *vocab, int32_t token_id);
 */
#ifndef BPE_TOKENIZER_H
#define BPE_TOKENIZER_H

#include <stdint.h>
#include <stddef.h>
#include "gguf_loader.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bpe_vocab bpe_vocab;

bpe_vocab *bpe_vocab_load(gguf_context *gguf);
void bpe_vocab_free(bpe_vocab *vocab);

/* Tokenize text. Returns number of tokens written, or -1 on error.
 * If tokens is NULL, returns the number of tokens that would be produced. */
int bpe_tokenize(const bpe_vocab *vocab, const char *text, int text_len,
                 int32_t *tokens, int max_tokens);

/* Get the string for a token ID. Returns NULL if invalid. */
const char *bpe_token_to_str(const bpe_vocab *vocab, int32_t token_id);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef BPE_TOKENIZER_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Unicode codepoint flags (same bitfield as llama.cpp) ---- */
#define BPE_FLAG_UNDEFINED   0x0001
#define BPE_FLAG_NUMBER      0x0002  /* \p{N} */
#define BPE_FLAG_LETTER      0x0004  /* \p{L} */
#define BPE_FLAG_SEPARATOR   0x0008  /* \p{Z} */
#define BPE_FLAG_ACCENT_MARK 0x0010  /* \p{M} */
#define BPE_FLAG_PUNCTUATION 0x0020  /* \p{P} */
#define BPE_FLAG_SYMBOL      0x0040  /* \p{S} */
#define BPE_FLAG_CONTROL     0x0080  /* \p{C} */
#define BPE_FLAG_WHITESPACE  0x0100

/* ---- Unicode data tables ---- */
#include "unicode_data_tables.inc"

/* ---- UTF-8 utilities ---- */

static int bpe_utf8_len(uint8_t c) {
    if (c < 0x80) return 1;
    if (c < 0xC0) return 1; /* continuation byte, treat as 1 */
    if (c < 0xE0) return 2;
    if (c < 0xF0) return 3;
    return 4;
}

static uint32_t bpe_utf8_to_cpt(const char *s, int len, int *out_len) {
    uint8_t c = (uint8_t)s[0];
    if (c < 0x80) { *out_len = 1; return c; }
    if (c < 0xE0 && len >= 2) {
        *out_len = 2;
        return ((uint32_t)(c & 0x1F) << 6) | (s[1] & 0x3F);
    }
    if (c < 0xF0 && len >= 3) {
        *out_len = 3;
        return ((uint32_t)(c & 0x0F) << 12) | ((uint32_t)(s[1] & 0x3F) << 6) | (s[2] & 0x3F);
    }
    if (len >= 4) {
        *out_len = 4;
        return ((uint32_t)(c & 0x07) << 18) | ((uint32_t)(s[1] & 0x3F) << 12) |
               ((uint32_t)(s[2] & 0x3F) << 6) | (s[3] & 0x3F);
    }
    *out_len = 1;
    return c;
}

static int bpe_cpt_to_utf8(uint32_t cpt, char *out) {
    if (cpt < 0x80) { out[0] = (char)cpt; return 1; }
    if (cpt < 0x800) { out[0] = 0xC0 | (cpt >> 6); out[1] = 0x80 | (cpt & 0x3F); return 2; }
    if (cpt < 0x10000) { out[0] = 0xE0 | (cpt >> 12); out[1] = 0x80 | ((cpt >> 6) & 0x3F); out[2] = 0x80 | (cpt & 0x3F); return 3; }
    out[0] = 0xF0 | (cpt >> 18); out[1] = 0x80 | ((cpt >> 12) & 0x3F); out[2] = 0x80 | ((cpt >> 6) & 0x3F); out[3] = 0x80 | (cpt & 0x3F);
    return 4;
}

/* ---- GPT-2 byte-level encoding ---- */
/* Maps each byte (0-255) to a Unicode codepoint, then to its UTF-8 encoding.
 * Printable ASCII (0x21-0x7E), 0xA1-0xAC, 0xAE-0xFF map to themselves.
 * All other bytes (0x00-0x20, 0x7F-0xA0, 0xAD) map to codepoints 256+ in order. */

static int bpe_byte_encoding_inited = 0;
static char bpe_byte_to_utf8[256][4]; /* UTF-8 encoding of mapped codepoint */
static int  bpe_byte_to_utf8_len[256];
static uint8_t bpe_utf8_to_byte_map[512]; /* codepoint -> original byte (for codepoints < 512) */
static int bpe_utf8_to_byte_valid[512];

static void bpe_init_byte_encoding(void) {
    if (bpe_byte_encoding_inited) return;
    bpe_byte_encoding_inited = 1;

    memset(bpe_utf8_to_byte_valid, 0, sizeof(bpe_utf8_to_byte_valid));

    int n = 0;
    for (int ch = 0; ch < 256; ch++) {
        uint32_t cpt;
        if ((ch >= 0x21 && ch <= 0x7E) || (ch >= 0xA1 && ch <= 0xAC) || (ch >= 0xAE && ch <= 0xFF)) {
            cpt = (uint32_t)ch;
        } else {
            cpt = 256 + n;
            n++;
        }
        bpe_byte_to_utf8_len[ch] = bpe_cpt_to_utf8(cpt, bpe_byte_to_utf8[ch]);
        if (cpt < 512) {
            bpe_utf8_to_byte_map[cpt] = (uint8_t)ch;
            bpe_utf8_to_byte_valid[cpt] = 1;
        }
    }
}

/* Encode raw bytes to GPT-2 byte-level UTF-8 string.
 * Returns malloc'd string and sets *out_len. */
static char *bpe_byte_encode(const char *data, int data_len, int *out_len) {
    bpe_init_byte_encoding();
    /* Worst case: each byte -> 2 UTF-8 bytes (codepoints 128-511) */
    char *out = (char *)malloc(data_len * 4 + 1);
    int pos = 0;
    for (int i = 0; i < data_len; i++) {
        uint8_t b = (uint8_t)data[i];
        memcpy(out + pos, bpe_byte_to_utf8[b], bpe_byte_to_utf8_len[b]);
        pos += bpe_byte_to_utf8_len[b];
    }
    out[pos] = '\0';
    *out_len = pos;
    return out;
}

/* Decode GPT-2 byte-level UTF-8 string back to raw bytes.
 * Returns malloc'd string and sets *out_len. */
static char *bpe_byte_decode(const char *data, int data_len, int *out_len) {
    bpe_init_byte_encoding();
    char *out = (char *)malloc(data_len + 1);
    int opos = 0;
    int pos = 0;
    while (pos < data_len) {
        int clen;
        uint32_t cpt = bpe_utf8_to_cpt(data + pos, data_len - pos, &clen);
        if (cpt < 512 && bpe_utf8_to_byte_valid[cpt]) {
            out[opos++] = (char)bpe_utf8_to_byte_map[cpt];
        } else {
            /* Pass through unknown codepoints as UTF-8 */
            memcpy(out + opos, data + pos, clen);
            opos += clen;
        }
        pos += clen;
    }
    out[opos] = '\0';
    *out_len = opos;
    return out;
}

/* ---- Unicode flag lookup (binary search on range table) ---- */

static uint16_t bpe_cpt_flags(uint32_t cpt) {
    /* Binary search: find the last entry with start <= cpt */
    int lo = 0, hi = bpe_unicode_ranges_flags_len - 1;
    uint16_t flags = BPE_FLAG_UNDEFINED;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (bpe_unicode_ranges_flags[mid].start <= cpt) {
            flags = bpe_unicode_ranges_flags[mid].flags;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    /* Check whitespace set */
    for (int i = 0; i < bpe_unicode_whitespace_len; i++) {
        if (bpe_unicode_whitespace[i] == cpt) {
            flags |= BPE_FLAG_WHITESPACE;
            break;
        }
    }
    return flags;
}

static uint32_t bpe_tolower(uint32_t cpt) {
    /* Binary search in lowercase map */
    int lo = 0, hi = bpe_unicode_lowercase_len - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (bpe_unicode_lowercase[mid].from == cpt) return bpe_unicode_lowercase[mid].to;
        if (bpe_unicode_lowercase[mid].from < cpt) lo = mid + 1;
        else hi = mid - 1;
    }
    return cpt;
}

/* ---- Hash map (open addressing, FNV-1a) ---- */

#define BPE_HM_EMPTY -1

typedef struct {
    char *key;
    int key_len;
    int32_t value;
} bpe_hm_entry;

typedef struct {
    bpe_hm_entry *entries;
    int capacity;
    int size;
} bpe_hashmap;

static uint32_t bpe_fnv1a(const char *data, int len) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < len; i++) {
        h ^= (uint8_t)data[i];
        h *= 16777619u;
    }
    return h;
}

static void bpe_hm_init(bpe_hashmap *hm, int capacity) {
    hm->capacity = capacity;
    hm->size = 0;
    hm->entries = (bpe_hm_entry *)calloc(capacity, sizeof(bpe_hm_entry));
    for (int i = 0; i < capacity; i++) hm->entries[i].value = BPE_HM_EMPTY;
}

static void bpe_hm_free(bpe_hashmap *hm) {
    if (!hm->entries) return;
    for (int i = 0; i < hm->capacity; i++) {
        if (hm->entries[i].value != BPE_HM_EMPTY) free(hm->entries[i].key);
    }
    free(hm->entries);
    hm->entries = NULL;
}

static void bpe_hm_set(bpe_hashmap *hm, const char *key, int key_len, int32_t value) {
    /* Grow if > 70% full */
    if (hm->size * 10 > hm->capacity * 7) {
        int old_cap = hm->capacity;
        bpe_hm_entry *old = hm->entries;
        int new_cap = old_cap * 2;
        hm->entries = (bpe_hm_entry *)calloc(new_cap, sizeof(bpe_hm_entry));
        hm->capacity = new_cap;
        hm->size = 0;
        for (int i = 0; i < new_cap; i++) hm->entries[i].value = BPE_HM_EMPTY;
        for (int i = 0; i < old_cap; i++) {
            if (old[i].value != BPE_HM_EMPTY) {
                bpe_hm_set(hm, old[i].key, old[i].key_len, old[i].value);
                free(old[i].key);
            }
        }
        free(old);
    }
    uint32_t h = bpe_fnv1a(key, key_len) % (uint32_t)hm->capacity;
    while (hm->entries[h].value != BPE_HM_EMPTY) {
        if (hm->entries[h].key_len == key_len && memcmp(hm->entries[h].key, key, key_len) == 0) {
            hm->entries[h].value = value;
            return;
        }
        h = (h + 1) % (uint32_t)hm->capacity;
    }
    hm->entries[h].key = (char *)malloc(key_len);
    memcpy(hm->entries[h].key, key, key_len);
    hm->entries[h].key_len = key_len;
    hm->entries[h].value = value;
    hm->size++;
}

static int32_t bpe_hm_get(const bpe_hashmap *hm, const char *key, int key_len) {
    if (!hm->entries || hm->capacity == 0) return BPE_HM_EMPTY;
    uint32_t h = bpe_fnv1a(key, key_len) % (uint32_t)hm->capacity;
    int probes = 0;
    while (hm->entries[h].value != BPE_HM_EMPTY && probes < hm->capacity) {
        if (hm->entries[h].key_len == key_len && memcmp(hm->entries[h].key, key, key_len) == 0) {
            return hm->entries[h].value;
        }
        h = (h + 1) % (uint32_t)hm->capacity;
        probes++;
    }
    return BPE_HM_EMPTY;
}

/* ---- Vocab structure ---- */

struct bpe_vocab {
    int n_tokens;
    char **token_strs;   /* token_strs[id] = string */
    int *token_str_lens;
    bpe_hashmap token_to_id;  /* token_str -> token_id */
    bpe_hashmap merge_ranks;  /* "left\0right" -> rank */
    int32_t eos_id, bos_id, eot_id, pad_id, unk_id;

    /* Special tokens (type=3 control tokens like <|im_start|>) */
    int n_special;
    int32_t *special_ids;
    char **special_strs;
    int *special_lens;
};

/* ---- GGUF vocab loading ---- */

bpe_vocab *bpe_vocab_load(gguf_context *gguf) {
    if (!gguf) return NULL;

    /* Find token list */
    int tokens_idx = gguf_find_key(gguf, "tokenizer.ggml.tokens");
    if (tokens_idx < 0) {
        fprintf(stderr, "bpe: cannot find tokenizer.ggml.tokens\n");
        return NULL;
    }
    gguf_kv *tokens_kv = &gguf->kv[tokens_idx];
    if (tokens_kv->type != GGUF_TYPE_ARRAY || tokens_kv->value.arr.type != GGUF_TYPE_STRING) {
        fprintf(stderr, "bpe: tokens is not a string array\n");
        return NULL;
    }

    int n_tokens = (int)tokens_kv->value.arr.n;
    gguf_str *token_strs = (gguf_str *)tokens_kv->value.arr.data;

    bpe_vocab *vocab = (bpe_vocab *)calloc(1, sizeof(bpe_vocab));
    vocab->n_tokens = n_tokens;
    vocab->token_strs = (char **)calloc(n_tokens, sizeof(char *));
    vocab->token_str_lens = (int *)calloc(n_tokens, sizeof(int));
    vocab->eos_id = -1;
    vocab->bos_id = -1;
    vocab->eot_id = -1;
    vocab->pad_id = -1;
    vocab->unk_id = -1;

    /* Build token_to_id map */
    bpe_hm_init(&vocab->token_to_id, n_tokens * 2 + 1);
    for (int i = 0; i < n_tokens; i++) {
        int slen = (int)token_strs[i].len;
        vocab->token_strs[i] = (char *)malloc(slen + 1);
        memcpy(vocab->token_strs[i], token_strs[i].str, slen);
        vocab->token_strs[i][slen] = '\0';
        vocab->token_str_lens[i] = slen;
        bpe_hm_set(&vocab->token_to_id, token_strs[i].str, slen, i);
    }

    /* Load merges */
    int merges_idx = gguf_find_key(gguf, "tokenizer.ggml.merges");
    if (merges_idx >= 0) {
        gguf_kv *merges_kv = &gguf->kv[merges_idx];
        if (merges_kv->type == GGUF_TYPE_ARRAY && merges_kv->value.arr.type == GGUF_TYPE_STRING) {
            int n_merges = (int)merges_kv->value.arr.n;
            gguf_str *merge_strs = (gguf_str *)merges_kv->value.arr.data;
            bpe_hm_init(&vocab->merge_ranks, n_merges * 2 + 1);
            for (int i = 0; i < n_merges; i++) {
                /* Each merge is "left right" - split on first space after pos 0 */
                const char *s = merge_strs[i].str;
                int slen = (int)merge_strs[i].len;
                const char *space = NULL;
                for (int j = 1; j < slen; j++) {
                    if (s[j] == ' ') { space = s + j; break; }
                }
                if (!space) continue;
                int left_len = (int)(space - s);
                int right_len = slen - left_len - 1;
                /* Key is "left\0right" */
                int key_len = left_len + 1 + right_len;
                char *key = (char *)malloc(key_len);
                memcpy(key, s, left_len);
                key[left_len] = '\0';
                memcpy(key + left_len + 1, space + 1, right_len);
                bpe_hm_set(&vocab->merge_ranks, key, key_len, i);
                free(key);
            }
        }
    }

    /* Special token IDs from GGUF KV */
    int idx;
    idx = gguf_find_key(gguf, "tokenizer.ggml.eos_token_id");
    if (idx >= 0 && gguf->kv[idx].type == GGUF_TYPE_UINT32) vocab->eos_id = (int32_t)gguf->kv[idx].value.u32;
    idx = gguf_find_key(gguf, "tokenizer.ggml.bos_token_id");
    if (idx >= 0 && gguf->kv[idx].type == GGUF_TYPE_UINT32) vocab->bos_id = (int32_t)gguf->kv[idx].value.u32;
    idx = gguf_find_key(gguf, "tokenizer.ggml.eot_token_id");
    if (idx >= 0 && gguf->kv[idx].type == GGUF_TYPE_UINT32) vocab->eot_id = (int32_t)gguf->kv[idx].value.u32;
    idx = gguf_find_key(gguf, "tokenizer.ggml.padding_token_id");
    if (idx >= 0 && gguf->kv[idx].type == GGUF_TYPE_UINT32) vocab->pad_id = (int32_t)gguf->kv[idx].value.u32;
    idx = gguf_find_key(gguf, "tokenizer.ggml.unknown_token_id");
    if (idx >= 0 && gguf->kv[idx].type == GGUF_TYPE_UINT32) vocab->unk_id = (int32_t)gguf->kv[idx].value.u32;

    /* Load special/control tokens (type 3 or 4) for literal matching */
    int type_idx = gguf_find_key(gguf, "tokenizer.ggml.token_type");
    if (type_idx >= 0 && gguf->kv[type_idx].type == GGUF_TYPE_ARRAY) {
        int32_t *types = (int32_t *)gguf->kv[type_idx].value.arr.data;
        /* Count special tokens */
        int ns = 0;
        for (int i = 0; i < n_tokens; i++) {
            if (types[i] == 3 || types[i] == 4) ns++;
        }
        vocab->n_special = ns;
        vocab->special_ids = (int32_t *)malloc(ns * sizeof(int32_t));
        vocab->special_strs = (char **)malloc(ns * sizeof(char *));
        vocab->special_lens = (int *)malloc(ns * sizeof(int));
        int si = 0;
        for (int i = 0; i < n_tokens; i++) {
            if (types[i] == 3 || types[i] == 4) {
                vocab->special_ids[si] = i;
                vocab->special_strs[si] = vocab->token_strs[i];
                vocab->special_lens[si] = vocab->token_str_lens[i];
                si++;
            }
        }
        /* Sort by length descending for greedy matching */
        for (int i = 0; i < ns - 1; i++) {
            for (int j = i + 1; j < ns; j++) {
                if (vocab->special_lens[j] > vocab->special_lens[i]) {
                    int32_t ti = vocab->special_ids[i]; vocab->special_ids[i] = vocab->special_ids[j]; vocab->special_ids[j] = ti;
                    char *ts = vocab->special_strs[i]; vocab->special_strs[i] = vocab->special_strs[j]; vocab->special_strs[j] = ts;
                    int tl = vocab->special_lens[i]; vocab->special_lens[i] = vocab->special_lens[j]; vocab->special_lens[j] = tl;
                }
            }
        }
        fprintf(stderr, "bpe: loaded %d special tokens\n", ns);
    }

    return vocab;
}

void bpe_vocab_free(bpe_vocab *vocab) {
    if (!vocab) return;
    for (int i = 0; i < vocab->n_tokens; i++) free(vocab->token_strs[i]);
    free(vocab->token_strs);
    free(vocab->token_str_lens);
    bpe_hm_free(&vocab->token_to_id);
    bpe_hm_free(&vocab->merge_ranks);
    free(vocab->special_ids);
    free(vocab->special_strs);  /* strings owned by token_strs, don't free contents */
    free(vocab->special_lens);
    free(vocab);
}

const char *bpe_token_to_str(const bpe_vocab *vocab, int32_t token_id) {
    if (!vocab || token_id < 0 || token_id >= vocab->n_tokens) return NULL;
    return vocab->token_strs[token_id];
}

/* ---- Pre-tokenizer: Qwen2 regex splitter ---- */
/*
 * Regex (Qwen2):
 *   (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
 *   |[^\r\n\p{L}\p{N}]?\p{L}+
 *   |\p{N}
 *   | ?[^\s\p{L}\p{N}]+[\r\n]*
 *   |\s*[\r\n]+
 *   |\s+(?!\S)
 *   |\s+
 *
 * We operate on codepoints. The input text is first converted to an array of
 * codepoints, then we emit "word" boundaries as byte offsets.
 */

/* Internal: convert text to codepoints, storing byte offsets */
typedef struct {
    uint32_t *cpts;
    int *byte_offsets; /* byte_offsets[i] = byte position of cpt[i] in original text */
    int n;
    int text_len;
} bpe_cpt_buf;

static bpe_cpt_buf bpe_text_to_cpts(const char *text, int text_len) {
    bpe_cpt_buf buf;
    buf.cpts = (uint32_t *)malloc(text_len * sizeof(uint32_t));
    buf.byte_offsets = (int *)malloc((text_len + 1) * sizeof(int));
    buf.n = 0;
    buf.text_len = text_len;
    int pos = 0;
    while (pos < text_len) {
        int clen;
        buf.byte_offsets[buf.n] = pos;
        buf.cpts[buf.n] = bpe_utf8_to_cpt(text + pos, text_len - pos, &clen);
        buf.n++;
        pos += clen;
    }
    buf.byte_offsets[buf.n] = text_len; /* sentinel */
    return buf;
}

static void bpe_cpt_buf_free(bpe_cpt_buf *buf) {
    free(buf->cpts);
    free(buf->byte_offsets);
}

/* Word list: array of (byte_start, byte_len) pairs */
typedef struct {
    int *starts;
    int *lens;
    int n;
    int cap;
} bpe_word_list;

static void bpe_wl_init(bpe_word_list *wl) {
    wl->cap = 64;
    wl->n = 0;
    wl->starts = (int *)malloc(wl->cap * sizeof(int));
    wl->lens = (int *)malloc(wl->cap * sizeof(int));
}

static void bpe_wl_push(bpe_word_list *wl, int start, int len) {
    if (len <= 0) return;
    if (wl->n >= wl->cap) {
        wl->cap *= 2;
        wl->starts = (int *)realloc(wl->starts, wl->cap * sizeof(int));
        wl->lens = (int *)realloc(wl->lens, wl->cap * sizeof(int));
    }
    wl->starts[wl->n] = start;
    wl->lens[wl->n] = len;
    wl->n++;
}

static void bpe_wl_free(bpe_word_list *wl) {
    free(wl->starts);
    free(wl->lens);
}

static bpe_word_list bpe_pretokenize_qwen2(const char *text, int text_len) {
    bpe_word_list wl;
    bpe_wl_init(&wl);

    bpe_cpt_buf cb = bpe_text_to_cpts(text, text_len);
    uint32_t *cpts = cb.cpts;
    int *boff = cb.byte_offsets;
    int n = cb.n;

    #define CPT(p) ((p) >= 0 && (p) < n ? cpts[p] : 0xFFFFFFFFu)
    #define FLAGS(p) ((p) >= 0 && (p) < n ? bpe_cpt_flags(cpts[p]) : 0)
    #define IS_LETTER(f) ((f) & BPE_FLAG_LETTER)
    #define IS_NUMBER(f) ((f) & BPE_FLAG_NUMBER)
    #define IS_WS(f) ((f) & BPE_FLAG_WHITESPACE)

    int prev_end = 0; /* codepoint index of previous token end */

    #define ADD_TOKEN(end_cpt) do { \
        int _s = prev_end, _e = (end_cpt); \
        if (_e > _s) { \
            bpe_wl_push(&wl, boff[_s], boff[_e] - boff[_s]); \
        } \
        prev_end = _e; \
    } while(0)

    for (int pos = 0; pos < n; ) {
        uint32_t cpt = CPT(pos);
        uint16_t flags = FLAGS(pos);

        /* (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD]) */
        if (cpt == '\'' && pos + 1 < n) {
            uint32_t c1 = bpe_tolower(CPT(pos + 1));
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
                ADD_TOKEN(pos + 2);
                pos += 2;
                continue;
            }
            if (pos + 2 < n) {
                uint32_t c2 = bpe_tolower(CPT(pos + 2));
                if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
                    ADD_TOKEN(pos + 3);
                    pos += 3;
                    continue;
                }
            }
        }

        /* [^\r\n\p{L}\p{N}]?\p{L}+ */
        if (!(cpt == '\r' || cpt == '\n' || IS_NUMBER(flags))) {
            if (IS_LETTER(flags) || IS_LETTER(FLAGS(pos + 1))) {
                int p = pos + 1;
                while (IS_LETTER(FLAGS(p))) p++;
                ADD_TOKEN(p);
                pos = p;
                continue;
            }
        }

        /* \p{N} (single digit for Qwen2) */
        if (IS_NUMBER(flags)) {
            ADD_TOKEN(pos + 1);
            pos++;
            continue;
        }

        /* <space>?[^\s\p{L}\p{N}]+[\r\n]* */
        {
            uint16_t f2 = (cpt == ' ') ? FLAGS(pos + 1) : flags;
            if (!(IS_WS(f2) | IS_LETTER(f2) | IS_NUMBER(f2)) && f2) {
                int p = pos + (cpt == ' ');
                while (p < n) {
                    uint16_t ff = FLAGS(p);
                    if (IS_WS(ff) | IS_LETTER(ff) | IS_NUMBER(ff) || !ff) break;
                    p++;
                }
                /* consume trailing \r\n */
                while (p < n && (CPT(p) == '\r' || CPT(p) == '\n')) p++;
                ADD_TOKEN(p);
                pos = p;
                continue;
            }
        }

        /* Count whitespace run */
        {
            int num_ws = 0;
            int last_rn = 0;
            while (IS_WS(FLAGS(pos + num_ws))) {
                uint32_t c2 = CPT(pos + num_ws);
                if (c2 == '\r' || c2 == '\n') last_rn = pos + num_ws + 1;
                num_ws++;
            }

            /* \s*[\r\n]+ */
            if (last_rn > 0) {
                int p = last_rn;
                ADD_TOKEN(p);
                pos = p;
                continue;
            }

            /* \s+(?!\S) */
            if (num_ws > 1 && CPT(pos + num_ws) != 0xFFFFFFFFu) {
                int p = pos + num_ws - 1;
                ADD_TOKEN(p);
                pos = p;
                continue;
            }

            /* \s+ */
            if (num_ws > 0) {
                int p = pos + num_ws;
                ADD_TOKEN(p);
                pos = p;
                continue;
            }
        }

        /* No match - advance one codepoint */
        ADD_TOKEN(pos + 1);
        pos++;
    }

    #undef CPT
    #undef FLAGS
    #undef IS_LETTER
    #undef IS_NUMBER
    #undef IS_WS
    #undef ADD_TOKEN

    bpe_cpt_buf_free(&cb);
    return wl;
}

/* ---- BPE merge algorithm ---- */

typedef struct {
    int prev, next; /* linked list indices, -1 = end */
    const char *text;
    int n; /* byte length */
} bpe_symbol;

typedef struct {
    int left, right;
    int rank;
    int size; /* combined byte length */
} bpe_bigram;

/* Min-heap for bigrams ordered by rank */
typedef struct {
    bpe_bigram *data;
    int n, cap;
} bpe_heap;

static void bpe_heap_init(bpe_heap *h) {
    h->cap = 64;
    h->n = 0;
    h->data = (bpe_bigram *)malloc(h->cap * sizeof(bpe_bigram));
}

static void bpe_heap_push(bpe_heap *h, bpe_bigram bg) {
    if (h->n >= h->cap) {
        h->cap *= 2;
        h->data = (bpe_bigram *)realloc(h->data, h->cap * sizeof(bpe_bigram));
    }
    int i = h->n++;
    h->data[i] = bg;
    /* sift up */
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (h->data[parent].rank <= h->data[i].rank) break;
        bpe_bigram tmp = h->data[parent];
        h->data[parent] = h->data[i];
        h->data[i] = tmp;
        i = parent;
    }
}

static bpe_bigram bpe_heap_pop(bpe_heap *h) {
    bpe_bigram top = h->data[0];
    h->data[0] = h->data[--h->n];
    /* sift down */
    int i = 0;
    for (;;) {
        int smallest = i;
        int l = 2 * i + 1, r = 2 * i + 2;
        if (l < h->n && h->data[l].rank < h->data[smallest].rank) smallest = l;
        if (r < h->n && h->data[r].rank < h->data[smallest].rank) smallest = r;
        if (smallest == i) break;
        bpe_bigram tmp = h->data[smallest];
        h->data[smallest] = h->data[i];
        h->data[i] = tmp;
        i = smallest;
    }
    return top;
}

static void bpe_heap_free(bpe_heap *h) { free(h->data); }

static int bpe_find_merge_rank(const bpe_vocab *vocab, const char *left, int left_len,
                               const char *right, int right_len) {
    int key_len = left_len + 1 + right_len;
    char key_buf[256];
    char *key = key_len <= (int)sizeof(key_buf) ? key_buf : (char *)malloc(key_len);
    memcpy(key, left, left_len);
    key[left_len] = '\0';
    memcpy(key + left_len + 1, right, right_len);
    int32_t rank = bpe_hm_get(&vocab->merge_ranks, key, key_len);
    if (key != key_buf) free(key);
    return rank;
}

static void bpe_add_bigram(const bpe_vocab *vocab, bpe_symbol *syms, bpe_heap *heap, int left, int right) {
    if (left < 0 || right < 0) return;
    int rank = bpe_find_merge_rank(vocab, syms[left].text, syms[left].n, syms[right].text, syms[right].n);
    if (rank < 0) return;
    bpe_bigram bg;
    bg.left = left;
    bg.right = right;
    bg.rank = rank;
    bg.size = syms[left].n + syms[right].n;
    bpe_heap_push(heap, bg);
}

/* Tokenize a single word (pre-tokenized segment) using BPE */
static int bpe_tokenize_word(const bpe_vocab *vocab, const char *word, int word_len,
                             int32_t *out_tokens, int max_tokens) {
    if (word_len <= 0) return 0;

    /* Byte-encode the word for GPT-2 byte-level BPE */
    int enc_len;
    char *encoded = bpe_byte_encode(word, word_len, &enc_len);

    /* Split encoded word into UTF-8 characters → symbols */
    int n_syms = 0;
    /* Count chars first */
    {
        int p = 0;
        while (p < enc_len) { p += bpe_utf8_len((uint8_t)encoded[p]); n_syms++; }
    }

    bpe_symbol *syms = (bpe_symbol *)calloc(n_syms, sizeof(bpe_symbol));
    {
        int idx = 0, p = 0;
        while (p < enc_len) {
            int clen = bpe_utf8_len((uint8_t)encoded[p]);
            if (p + clen > enc_len) clen = enc_len - p;
            syms[idx].text = encoded + p;
            syms[idx].n = clen;
            syms[idx].prev = idx - 1;
            syms[idx].next = (p + clen >= enc_len) ? -1 : idx + 1;
            p += clen;
            idx++;
        }
    }

    /* Build initial bigrams */
    bpe_heap heap;
    bpe_heap_init(&heap);
    for (int i = 1; i < n_syms; i++) {
        bpe_add_bigram(vocab, syms, &heap, i - 1, i);
    }

    /* Greedily merge */
    while (heap.n > 0) {
        bpe_bigram bg = bpe_heap_pop(&heap);
        bpe_symbol *left_sym = &syms[bg.left];
        bpe_symbol *right_sym = &syms[bg.right];

        /* Skip if either symbol was already merged away */
        if (left_sym->n == 0 || right_sym->n == 0) continue;

        /* Skip if symbols have changed (outdated bigram) */
        if (left_sym->n + right_sym->n != bg.size) continue;

        /* Verify text still matches (contiguous) */
        if (left_sym->text + left_sym->n != right_sym->text) continue;

        /* Merge right into left */
        left_sym->n += right_sym->n;
        right_sym->n = 0;

        /* Update linked list */
        left_sym->next = right_sym->next;
        if (right_sym->next >= 0) {
            syms[right_sym->next].prev = bg.left;
        }

        /* Add new bigrams */
        bpe_add_bigram(vocab, syms, &heap, left_sym->prev, bg.left);
        bpe_add_bigram(vocab, syms, &heap, bg.left, left_sym->next);
    }

    /* Collect final tokens */
    int n_out = 0;
    for (int i = 0; i < n_syms && (out_tokens == NULL || n_out < max_tokens); i++) {
        if (syms[i].n == 0) continue;

        int32_t token_id = bpe_hm_get(&vocab->token_to_id, syms[i].text, syms[i].n);
        if (token_id >= 0) {
            if (out_tokens) out_tokens[n_out] = token_id;
            n_out++;
        } else {
            /* Byte fallback: look up each byte as a token */
            for (int b = 0; b < syms[i].n; b++) {
                char hex_buf[8];
                snprintf(hex_buf, sizeof(hex_buf), "<0x%02X>", (uint8_t)syms[i].text[b]);
                int32_t byte_token = bpe_hm_get(&vocab->token_to_id, hex_buf, (int)strlen(hex_buf));
                if (byte_token >= 0) {
                    if (out_tokens) {
                        if (n_out < max_tokens) out_tokens[n_out] = byte_token;
                    }
                    n_out++;
                }
            }
        }
    }

    bpe_heap_free(&heap);
    free(syms);
    free(encoded);
    return n_out;
}

/* ---- Public API ---- */

/* Tokenize a segment (between special tokens) using BPE */
static int bpe_tokenize_segment(const bpe_vocab *vocab, const char *text, int text_len,
                                int32_t *tokens, int max_tokens) {
    if (text_len <= 0) return 0;
    bpe_word_list wl = bpe_pretokenize_qwen2(text, text_len);
    int total = 0;
    for (int w = 0; w < wl.n; w++) {
        int32_t *dst = tokens ? tokens + total : NULL;
        int remaining = tokens ? max_tokens - total : 0;
        int n = bpe_tokenize_word(vocab, text + wl.starts[w], wl.lens[w], dst, remaining);
        total += n;
    }
    bpe_wl_free(&wl);
    return total;
}

int bpe_tokenize(const bpe_vocab *vocab, const char *text, int text_len,
                 int32_t *tokens, int max_tokens) {
    if (!vocab || !text) return -1;
    if (text_len < 0) text_len = (int)strlen(text);

    int total = 0;
    int pos = 0;

    while (pos < text_len) {
        /* Try to match a special token at current position */
        int matched = 0;
        for (int s = 0; s < vocab->n_special; s++) {
            int slen = vocab->special_lens[s];
            if (pos + slen <= text_len &&
                memcmp(text + pos, vocab->special_strs[s], slen) == 0) {
                /* Tokenize any text before the special token */
                if (pos > 0) {
                    /* Find start of unprocessed text */
                }
                /* Emit the special token */
                if (tokens && total < max_tokens) tokens[total] = vocab->special_ids[s];
                total++;
                pos += slen;
                matched = 1;
                break;
            }
        }
        if (matched) continue;

        /* Find the next special token occurrence */
        int next_special = text_len;
        for (int s = 0; s < vocab->n_special; s++) {
            int slen = vocab->special_lens[s];
            for (int p = pos + 1; p + slen <= text_len; p++) {
                if (memcmp(text + p, vocab->special_strs[s], slen) == 0) {
                    if (p < next_special) next_special = p;
                    break;
                }
            }
        }

        /* Tokenize the segment [pos, next_special) with BPE */
        int seg_len = next_special - pos;
        int32_t *dst = tokens ? tokens + total : NULL;
        int remaining = tokens ? max_tokens - total : 0;
        int n = bpe_tokenize_segment(vocab, text + pos, seg_len, dst, remaining);
        total += n;
        pos = next_special;
    }

    return total;
}

#endif /* BPE_TOKENIZER_IMPLEMENTATION */
#endif /* BPE_TOKENIZER_H */
