/* SPDX-License-Identifier: MIT
 * Copyright 2026 - Present, Light Transport Entertainment Inc.
 *
 * en_phoneme.h - English symbol tables for en_align.
 *
 *   - the model vocabulary (label strings of the Apache-2.0
 *     facebook/wav2vec2-lv-60-espeak-cv-ft model: IPA phone tokens, <pad> = CTC blank),
 *     loaded from the "<model>.vocab.txt" written by convert_ckpt.py
 *   - IPA token -> viseme class (the common 15-class set shared with ja_align)
 *   - grapheme-to-phoneme: the CMU Pronouncing Dictionary (BSD) when available,
 *     with an original letter-to-sound fallback for words it does not contain;
 *     ARPAbet -> the model's IPA tokens
 */
#ifndef EN_PHONEME_H
#define EN_PHONEME_H

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../ja_align/ja_phoneme.h"   /* the viseme enum and names */

#define EN_MAX_VOCAB 512
#define EN_TOK 24

typedef struct {
    int n;
    char tok[EN_MAX_VOCAB][EN_TOK];
} en_vocab;

/* "<model>.safetensors" -> "<model>.vocab.txt" */
static int en_vocab_load(en_vocab *v, const char *model_path) {
    char path[1024];
    size_t n = strlen(model_path);
    const char *dot = strrchr(model_path, '.');
    size_t stem = dot && !strchr(dot, '/') ? (size_t)(dot - model_path) : n;
    snprintf(path, sizeof(path), "%.*s.vocab.txt", (int)stem, model_path);
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "en_align: cannot open the vocabulary %s\n", path); return -1; }
    v->n = 0;
    char line[256];
    while (v->n < EN_MAX_VOCAB && fgets(line, sizeof(line), f)) {
        size_t l = strlen(line);
        while (l && (line[l - 1] == '\n' || line[l - 1] == '\r')) line[--l] = 0;
        if (l >= EN_TOK) { fprintf(stderr, "en_align: vocabulary label too long: %s\n", line); fclose(f); return -1; }
        memcpy(v->tok[v->n++], line, l + 1);
    }
    fclose(f);
    return v->n > 1 && !strcmp(v->tok[0], "<pad>") ? 0 : -1;
}

static int en_vocab_index(const en_vocab *v, const char *tok) {
    for (int i = 1; i < v->n; i++) if (!strcmp(v->tok[i], tok)) return i;
    return -1;
}

/* Viseme of an IPA token: classify by its leading symbol (UTF-8 aware for the IPA letters). */
static int en_viseme(const char *t) {
    if (!t[0] || t[0] == '<') return JA_VIS_SIL;
    unsigned char c0 = (unsigned char)t[0];
    if (c0 < 0x80) {
        switch (c0) {
        case 'p': case 'b': case 'm': return JA_VIS_PP;
        case 'f': case 'v': return JA_VIS_FF;
        /* affricates only: ts, tʃ, tɕ / dz, dʒ, dʑ; aspirated or palatalised stops (tʰ, tʲ, dʲ) stay DD */
        case 't': return (t[1] == 's' || !strncmp(t + 1, "ʃ", 2) || !strncmp(t + 1, "ɕ", 2)) ? JA_VIS_CH : JA_VIS_DD;
        case 'd': return (t[1] == 'z' || !strncmp(t + 1, "ʒ", 2) || !strncmp(t + 1, "ʑ", 2)) ? JA_VIS_CH : JA_VIS_DD;
        case 'n': case 'l': return JA_VIS_NN;
        case 'k': case 'g': case 'x': case 'h': case 'c': case 'q': case 'X': case 'N': return JA_VIS_KK;
        case 's': case 'z': case 'S': return JA_VIS_SS;
        case 'r': return JA_VIS_RR;
        case 'w': return JA_VIS_OU;
        case 'j': case 'y': return JA_VIS_IH;
        case 'a': return (unsigned char)t[1] == 0xCA ? JA_VIS_OU : JA_VIS_AA;                   /* aʊ: rounded end */
        case 'e': return JA_VIS_E;
        case 'i': return JA_VIS_IH;
        case 'o': return JA_VIS_OH;
        case 'u': return JA_VIS_OU;
        default: return JA_VIS_SIL;
        }
    }
    /* two- and three-byte IPA letters */
    if (!strncmp(t, "ɑ", 2) || !strncmp(t, "æ", 2) || !strncmp(t, "ʌ", 2) || !strncmp(t, "ɐ", 2) || !strncmp(t, "ä", 2)) return JA_VIS_AA;
    if (!strncmp(t, "ɛ", 2) || !strncmp(t, "ə", 2) || !strncmp(t, "ɚ", 2) || !strncmp(t, "ɜ", 2) || !strncmp(t, "œ", 2) || !strncmp(t, "ø", 2) || !strncmp(t, "ᵻ", 3)) return JA_VIS_E;
    if (!strncmp(t, "ɪ", 2) || !strncmp(t, "ɨ", 2) || !strncmp(t, "ʲ", 2)) return JA_VIS_IH;
    if (!strncmp(t, "ɔ", 2) || !strncmp(t, "ɒ", 2)) return JA_VIS_OH;
    if (!strncmp(t, "ʊ", 2) || !strncmp(t, "ɯ", 2) || !strncmp(t, "ʉ", 2) || !strncmp(t, "ɵ", 2)) return JA_VIS_OU;
    if (!strncmp(t, "θ", 2) || !strncmp(t, "ð", 2)) return JA_VIS_TH;
    if (!strncmp(t, "ʃ", 2) || !strncmp(t, "ʒ", 2) || !strncmp(t, "ɕ", 2) || !strncmp(t, "ʑ", 2) || !strncmp(t, "ʂ", 2) || !strncmp(t, "ʐ", 2)) return JA_VIS_CH;
    if (!strncmp(t, "ɹ", 2) || !strncmp(t, "ɾ", 2) || !strncmp(t, "ɻ", 2) || !strncmp(t, "ʁ", 2) || !strncmp(t, "ɽ", 2)) return JA_VIS_RR;
    if (!strncmp(t, "ŋ", 2) || !strncmp(t, "ɲ", 2) || !strncmp(t, "ɳ", 2) || !strncmp(t, "ɫ", 2) || !strncmp(t, "ɬ", 2) || !strncmp(t, "ʎ", 2) || !strncmp(t, "ɭ", 2)) return JA_VIS_NN;
    if (!strncmp(t, "ɡ", 2) || !strncmp(t, "ɣ", 2) || !strncmp(t, "χ", 2) || !strncmp(t, "ç", 2) || !strncmp(t, "ħ", 2) || !strncmp(t, "ʔ", 2)) return JA_VIS_KK;
    if (!strncmp(t, "β", 2) || !strncmp(t, "ɸ", 2) || !strncmp(t, "ʋ", 2)) return JA_VIS_FF;
    return JA_VIS_SIL;
}

/* ARPAbet (stress digit stripped; AH0 -> schwa) -> the model's IPA token. */
static const char *en_arpa_to_ipa(const char *p) {
    static const char *const table[][2] = {
        {"AA", "ɑː"}, {"AE", "æ"}, {"AH", "ʌ"}, {"AO", "ɔː"}, {"AW", "aʊ"}, {"AY", "aɪ"}, {"EH", "ɛ"}, {"ER", "ɚ"},
        {"EY", "eɪ"}, {"IH", "ɪ"}, {"IY", "iː"}, {"OW", "oʊ"}, {"OY", "ɔɪ"}, {"UH", "ʊ"}, {"UW", "uː"},
        {"B", "b"}, {"CH", "tʃ"}, {"D", "d"}, {"DH", "ð"}, {"F", "f"}, {"G", "ɡ"}, {"HH", "h"}, {"JH", "dʒ"}, {"K", "k"},
        {"L", "l"}, {"M", "m"}, {"N", "n"}, {"NG", "ŋ"}, {"P", "p"}, {"R", "ɹ"}, {"S", "s"}, {"SH", "ʃ"}, {"T", "t"},
        {"TH", "θ"}, {"V", "v"}, {"W", "w"}, {"Y", "j"}, {"Z", "z"}, {"ZH", "ʒ"}};
    char base[8];
    size_t n = 0;
    for (; p[n] && n < 7 && !isdigit((unsigned char)p[n]); n++) base[n] = (char)toupper((unsigned char)p[n]);
    base[n] = 0;
    if (!strcmp(base, "AH") && p[n] == '0') return "ə";
    if (!strcmp(base, "ER") && p[n] == '1') return "ɜː";
    for (size_t i = 0; i < sizeof(table) / sizeof(table[0]); i++) if (!strcmp(table[i][0], base)) return table[i][1];
    return NULL;
}

/* ---- CMU Pronouncing Dictionary (BSD): word -> ARPAbet, open-addressing hash ---- */
typedef struct { char *word, *phones; } en_dict_entry;
typedef struct { en_dict_entry *e; size_t cap, n; } en_dict;

static unsigned en__hash(const char *s) { unsigned h = 2166136261u; for (; *s; s++) h = (h ^ (unsigned char)*s) * 16777619u; return h; }

static int en_dict_load(en_dict *d, const char *path) {
    memset(d, 0, sizeof(*d));
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    d->cap = 1u << 19;
    d->e = (en_dict_entry *)calloc(d->cap, sizeof(en_dict_entry));
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == ';' || line[0] == '#') continue;
        char *sp = strchr(line, ' ');
        if (!sp) continue;
        *sp = 0;
        if (strchr(line, '(')) continue;                   /* alternate pronunciations: keep the first */
        char *ph = sp + 1, *end = ph + strlen(ph);
        while (end > ph && (end[-1] == '\n' || end[-1] == '\r' || end[-1] == ' ')) *--end = 0;
        char *cm = strstr(ph, " #");                       /* trailing comments */
        if (cm) *cm = 0;
        for (char *c = line; *c; c++) *c = (char)tolower((unsigned char)*c);
        size_t i = en__hash(line) & (d->cap - 1);
        while (d->e[i].word) { if (!strcmp(d->e[i].word, line)) break; i = (i + 1) & (d->cap - 1); }
        if (d->e[i].word) continue;
        d->e[i].word = strdup(line); d->e[i].phones = strdup(ph); d->n++;
        if (d->n * 2 > d->cap) break;                      /* the dictionary is ~135k words; never reached */
    }
    fclose(f);
    return d->n ? 0 : -1;
}

static const char *en_dict_lookup(const en_dict *d, const char *word) {
    if (!d->e) return NULL;
    size_t i = en__hash(word) & (d->cap - 1);
    while (d->e[i].word) { if (!strcmp(d->e[i].word, word)) return d->e[i].phones; i = (i + 1) & (d->cap - 1); }
    return NULL;
}

static void en_dict_free(en_dict *d) {
    if (!d->e) return;
    for (size_t i = 0; i < d->cap; i++) { free(d->e[i].word); free(d->e[i].phones); }
    free(d->e);
    memset(d, 0, sizeof(*d));
}

/* Original letter-to-sound fallback for out-of-vocabulary words: greedy longest-match grapheme rules
 * (English spelling regularities), emitting ARPAbet. Approximate; the aligner tolerates it. */
static int en_letters_to_arpa(const char *w, char *out, size_t cap) {
    static const char *const rules[][2] = {
        {"tion", "SH AH0 N"}, {"sion", "ZH AH0 N"}, {"ough", "AH1 F"}, {"ight", "AY1 T"}, {"tch", "CH"}, {"dge", "JH"},
        {"igh", "AY1"}, {"ing", "IH0 NG"}, {"ous", "AH0 S"}, {"ch", "CH"}, {"sh", "SH"}, {"th", "TH"}, {"ph", "F"}, {"wh", "W"},
        {"ck", "K"}, {"qu", "K W"}, {"ng", "NG"}, {"ee", "IY1"}, {"ea", "IY1"}, {"oo", "UW1"}, {"ai", "EY1"}, {"ay", "EY1"},
        {"ou", "AW1"}, {"ow", "OW1"}, {"oi", "OY1"}, {"oy", "OY1"}, {"ie", "IY1"}, {"ey", "IY1"}, {"ar", "AA1 R"}, {"er", "ER0"},
        {"ir", "ER1"}, {"or", "AO1 R"}, {"ur", "ER1"}, {"ce", "S"}, {"ci", "S IH0"}, {"ge", "JH"}, {"gi", "JH IH0"},
        {"a", "AE1"}, {"b", "B"}, {"c", "K"}, {"d", "D"}, {"e", "EH1"}, {"f", "F"}, {"g", "G"}, {"h", "HH"}, {"i", "IH1"},
        {"j", "JH"}, {"k", "K"}, {"l", "L"}, {"m", "M"}, {"n", "N"}, {"o", "AA1"}, {"p", "P"}, {"r", "R"}, {"s", "S"},
        {"t", "T"}, {"u", "AH1"}, {"v", "V"}, {"w", "W"}, {"x", "K S"}, {"y", "IY0"}, {"z", "Z"}};
    size_t n = 0, len = strlen(w);
    out[0] = 0;
    for (size_t i = 0; i < len;) {
        int matched = 0;
        for (size_t r = 0; r < sizeof(rules) / sizeof(rules[0]); r++) {
            size_t rl = strlen(rules[r][0]);
            if (strncmp(w + i, rules[r][0], rl)) continue;
            if (rl == 1 && w[i] == 'e' && i + 1 == len && n) { i++; matched = 1; break; }   /* silent final e */
            n += (size_t)snprintf(out + n, cap > n ? cap - n : 0, "%s%s", n ? " " : "", rules[r][1]);
            i += rl; matched = 1; break;
        }
        if (!matched) i++;
        if (n >= cap) return -1;
    }
    return n ? 0 : -1;
}

/* Text -> model token ids (and the word spans). Lowercases, strips punctuation, splits on spaces.
 * Returns the token count, or -1. word_of[k] = word index of token k; nwords receives the count. */
static int en_text_to_ids(const en_vocab *v, const en_dict *d, const char *text, int *ids, int *word_of, int cap,
                          int *nwords, int *unknown_words) {
    int n = 0, w = 0, unk = 0;
    char word[128];
    size_t wl = 0;
    const char *p = text;
    for (;; p++) {
        unsigned char c = (unsigned char)*p;
        int alpha = c >= 0x80 || isalpha(c) || c == '\'';
        if (alpha && wl < sizeof(word) - 1) { word[wl++] = (char)tolower(c); continue; }
        if (wl) {
            word[wl] = 0;
            char arpa[256];
            const char *ph = en_dict_lookup(d, word);
            if (!ph) { unk++; if (en_letters_to_arpa(word, arpa, sizeof(arpa))) ph = NULL; else ph = arpa; }
            if (ph) {
                char buf[256];
                snprintf(buf, sizeof(buf), "%s", ph);
                for (char *t = strtok(buf, " "); t; t = strtok(NULL, " ")) {
                    const char *ipa = en_arpa_to_ipa(t);
                    int id = ipa ? en_vocab_index(v, ipa) : -1;
                    if (id < 0) continue;
                    if (n >= cap) return -1;
                    ids[n] = id; word_of[n] = w; n++;
                }
                w++;
            }
            wl = 0;
        }
        if (!c) break;
    }
    if (nwords) *nwords = w;
    if (unknown_words) *unknown_words = unk;
    return n;
}

#endif /* EN_PHONEME_H */
