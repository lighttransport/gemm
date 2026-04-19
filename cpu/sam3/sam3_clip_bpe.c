/* CLIP BPE tokenizer — HF-compatible for simple phrases.
 *
 * Matches transformers.CLIPTokenizer output for ASCII / basic UTF-8 input:
 *   "cat"  -> [49406, 2368, 49407, 49407, ...]
 *   "a dog" -> [49406, 320, 1929, 49407, ...]
 *
 * Algorithm: lowercase → regex-lite word split → GPT-2 byte_to_unicode per
 * byte → pairwise BPE merge using ranked merges → vocab lookup.
 */
#include "sam3_clip_bpe.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --- tiny hash table (string -> int32) ------------------------------ */

typedef struct {
    char   **keys;
    int32_t *vals;
    uint32_t cap;
    uint32_t n;
} str_map;

static uint32_t fnv1a(const char *s, size_t len) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < len; i++) { h ^= (unsigned char)s[i]; h *= 16777619u; }
    return h;
}

static void smap_init(str_map *m, uint32_t cap_pow2) {
    m->cap = cap_pow2; m->n = 0;
    m->keys = (char **)calloc(cap_pow2, sizeof(char *));
    m->vals = (int32_t *)calloc(cap_pow2, sizeof(int32_t));
}

static void smap_free(str_map *m) {
    for (uint32_t i = 0; i < m->cap; i++) if (m->keys[i]) free(m->keys[i]);
    free(m->keys); free(m->vals);
}

static void smap_put(str_map *m, const char *key, size_t klen, int32_t val) {
    uint32_t h = fnv1a(key, klen) & (m->cap - 1);
    while (m->keys[h]) {
        if (strlen(m->keys[h]) == klen && memcmp(m->keys[h], key, klen) == 0) {
            m->vals[h] = val; return;
        }
        h = (h + 1) & (m->cap - 1);
    }
    m->keys[h] = (char *)malloc(klen + 1);
    memcpy(m->keys[h], key, klen); m->keys[h][klen] = '\0';
    m->vals[h] = val; m->n++;
}

static int smap_get(const str_map *m, const char *key, size_t klen, int32_t *val) {
    uint32_t h = fnv1a(key, klen) & (m->cap - 1);
    while (m->keys[h]) {
        if (strlen(m->keys[h]) == klen && memcmp(m->keys[h], key, klen) == 0) {
            *val = m->vals[h]; return 1;
        }
        h = (h + 1) & (m->cap - 1);
    }
    return 0;
}

/* --- byte_to_unicode (GPT-2 / CLIP) ---------------------------------- */
/* Each byte maps to a unicode codepoint; we precompute the UTF-8 bytes
 * for that codepoint into b2u[b] (up to 3 bytes + nul). */

static char g_b2u[256][5];
static int  g_b2u_len[256];

static void encode_utf8(unsigned cp, char *out, int *len) {
    if (cp < 0x80)      { out[0] = (char)cp;                                              *len = 1; }
    else if (cp < 0x800){ out[0] = (char)(0xC0 | (cp >> 6));
                          out[1] = (char)(0x80 | (cp & 0x3F));                            *len = 2; }
    else                { out[0] = (char)(0xE0 | (cp >> 12));
                          out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
                          out[2] = (char)(0x80 | (cp & 0x3F));                            *len = 3; }
    out[*len] = '\0';
}

static void init_b2u(void) {
    /* "printable" ranges mapped identity; remaining bytes shifted to 256+n. */
    int used[256]; memset(used, 0, sizeof(used));
    int bs[256]; int nb = 0;
    for (int c = '!'; c <= '~'; c++) { bs[nb++] = c; used[c] = 1; }
    for (int c = 0xA1; c <= 0xAC; c++) { bs[nb++] = c; used[c] = 1; }
    for (int c = 0xAE; c <= 0xFF; c++) { bs[nb++] = c; used[c] = 1; }
    /* identity map */
    for (int i = 0; i < nb; i++) encode_utf8((unsigned)bs[i], g_b2u[bs[i]], &g_b2u_len[bs[i]]);
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (!used[b]) { encode_utf8((unsigned)(256 + n), g_b2u[b], &g_b2u_len[b]); n++; }
    }
}

/* --- vocab/merges loader --------------------------------------------- */

struct sam3_clip_bpe {
    str_map vocab;   /* token string -> id */
    str_map merges;  /* "a b" -> rank */
    int     loaded;
};

/* Parse a JSON string literal starting at *p (points at opening quote).
 * Writes unescaped bytes into out (must be large enough). Advances *p to
 * byte after closing quote. Returns length written. */
static size_t parse_json_string(const char **p, char *out) {
    const char *s = *p; if (*s != '"') return 0; s++;
    size_t n = 0;
    while (*s && *s != '"') {
        if (*s == '\\') {
            s++;
            if (*s == 'u') {
                unsigned cp = 0;
                s++;
                for (int i = 0; i < 4; i++) {
                    char c = s[i]; unsigned d;
                    if (c >= '0' && c <= '9') d = c - '0';
                    else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
                    else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
                    else d = 0;
                    cp = cp * 16 + d;
                }
                s += 4;
                int len; char buf[5];
                encode_utf8(cp, buf, &len);
                memcpy(out + n, buf, len); n += len;
            } else {
                char c = *s++;
                switch (c) {
                    case 'n': out[n++] = '\n'; break;
                    case 't': out[n++] = '\t'; break;
                    case 'r': out[n++] = '\r'; break;
                    case 'b': out[n++] = '\b'; break;
                    case 'f': out[n++] = '\f'; break;
                    default:  out[n++] = c;    break;
                }
            }
        } else {
            out[n++] = *s++;
        }
    }
    if (*s == '"') s++;
    *p = s;
    return n;
}

static int load_vocab(str_map *m, const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) return -1;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(sz + 1);
    if (fread(buf, 1, sz, f) != (size_t)sz) { free(buf); fclose(f); return -1; }
    buf[sz] = '\0'; fclose(f);

    smap_init(m, 131072);
    const char *p = buf;
    /* skip to first '{' */
    while (*p && *p != '{') p++;
    if (*p == '{') p++;
    char key[512];
    while (*p) {
        while (*p && *p != '"' && *p != '}') p++;
        if (*p != '"') break;
        size_t klen = parse_json_string(&p, key);
        while (*p && *p != ':') p++;
        if (*p == ':') p++;
        while (*p == ' ') p++;
        long val = strtol(p, (char **)&p, 10);
        smap_put(m, key, klen, (int32_t)val);
        while (*p && *p != ',' && *p != '}') p++;
        if (*p == ',') p++;
        else if (*p == '}') break;
    }
    free(buf);
    return 0;
}

static int load_merges(str_map *m, const char *path) {
    FILE *f = fopen(path, "r"); if (!f) return -1;
    smap_init(m, 131072);
    char line[1024];
    int rank = 0;
    int first = 1;
    while (fgets(line, sizeof(line), f)) {
        size_t ll = strlen(line);
        while (ll && (line[ll-1] == '\n' || line[ll-1] == '\r')) line[--ll] = '\0';
        if (ll == 0) continue;
        if (first) { first = 0; if (line[0] == '#') continue; }
        /* line is "a b" */
        smap_put(m, line, ll, rank);
        rank++;
    }
    fclose(f);
    return 0;
}

/* --- public API ------------------------------------------------------ */

sam3_clip_bpe *sam3_clip_bpe_load(const char *vocab_path, const char *merges_path) {
    sam3_clip_bpe *t = (sam3_clip_bpe *)calloc(1, sizeof(*t));
    init_b2u();
    if (load_vocab(&t->vocab, vocab_path) < 0) { free(t); return NULL; }
    if (load_merges(&t->merges, merges_path) < 0) {
        smap_free(&t->vocab); free(t); return NULL;
    }
    t->loaded = 1;
    return t;
}

void sam3_clip_bpe_free(sam3_clip_bpe *t) {
    if (!t) return;
    smap_free(&t->vocab); smap_free(&t->merges);
    free(t);
}

/* BPE on a single word: `parts` holds N token-strings; merge in place.
 * Returns final N. */
static int bpe_merge(const str_map *merges, char parts[][16], int n) {
    char pair[40];
    while (n > 1) {
        int best_rank = -1; int best_i = -1;
        for (int i = 0; i < n - 1; i++) {
            int la = (int)strlen(parts[i]);
            int lb = (int)strlen(parts[i+1]);
            if (la + 1 + lb >= (int)sizeof(pair)) continue;
            memcpy(pair, parts[i], la);
            pair[la] = ' ';
            memcpy(pair + la + 1, parts[i+1], lb);
            int plen = la + 1 + lb;
            int32_t r;
            if (smap_get(merges, pair, plen, &r)) {
                if (best_rank < 0 || r < best_rank) { best_rank = r; best_i = i; }
            }
        }
        if (best_i < 0) break;
        /* merge parts[best_i] + parts[best_i+1] */
        int la = (int)strlen(parts[best_i]);
        int lb = (int)strlen(parts[best_i+1]);
        if (la + lb >= (int)sizeof(parts[0])) break;
        memcpy(parts[best_i] + la, parts[best_i+1], lb + 1);
        for (int j = best_i + 1; j < n - 1; j++)
            memcpy(parts[j], parts[j+1], sizeof(parts[0]));
        n--;
    }
    return n;
}

int sam3_clip_bpe_encode(const sam3_clip_bpe *t, const char *text,
                          int max_len, int32_t *out_ids, int32_t *out_mask)
{
    if (!t || !t->loaded) return -1;
    const int BOS = 49406, EOS = 49407;
    int32_t ids_buf[512]; int nid = 0;
    ids_buf[nid++] = BOS;

    /* Lowercase + simple whitespace split. */
    const char *p = text;
    while (*p) {
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;
        /* collect a word = contiguous non-space bytes, lowercased */
        char word[256]; int wl = 0;
        while (*p && !isspace((unsigned char)*p) && wl < (int)sizeof(word) - 1) {
            unsigned char c = (unsigned char)*p;
            if (c >= 'A' && c <= 'Z') c = c - 'A' + 'a';
            word[wl++] = (char)c; p++;
        }
        word[wl] = '\0';

        /* byte -> unicode per byte; initial parts list */
        char parts[256][16]; int np = 0;
        for (int i = 0; i < wl && np < 256; i++) {
            unsigned char c = (unsigned char)word[i];
            int ul = g_b2u_len[c];
            memcpy(parts[np], g_b2u[c], ul);
            parts[np][ul] = '\0';
            np++;
        }
        /* append </w> to last part */
        if (np > 0) {
            int ll = (int)strlen(parts[np-1]);
            if (ll + 4 < (int)sizeof(parts[0])) {
                memcpy(parts[np-1] + ll, "</w>", 5);
            }
        }
        np = bpe_merge(&t->merges, parts, np);

        for (int i = 0; i < np; i++) {
            int32_t id;
            if (!smap_get(&t->vocab, parts[i], strlen(parts[i]), &id)) {
                /* Unknown token — skip (shouldn't happen for ASCII). */
                continue;
            }
            if (nid < (int)(sizeof(ids_buf)/sizeof(ids_buf[0]))) ids_buf[nid++] = id;
        }
    }
    if (nid < (int)(sizeof(ids_buf)/sizeof(ids_buf[0]))) ids_buf[nid++] = EOS;

    /* Truncate / pad to max_len. */
    int n_valid = nid < max_len ? nid : max_len;
    for (int i = 0; i < max_len; i++) {
        if (i < n_valid) { out_ids[i] = ids_buf[i]; out_mask[i] = 1; }
        else             { out_ids[i] = EOS;       out_mask[i] = 0; }
    }
    return n_valid;
}
