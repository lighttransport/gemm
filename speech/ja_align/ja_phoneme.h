/* SPDX-License-Identifier: MIT
 * Copyright 2026 - Present, Light Transport Entertainment Inc.
 *
 * ja_phoneme.h - Japanese symbol tables for ja_align.
 *
 *   - model vocabularies (label strings of the Apache-2.0 hiragana-ctc model:
 *     42 OpenJTalk-style phonemes, 82 kana; index 0 = CTC blank)
 *   - kana (hiragana or katakana) reading -> phoneme sequence, written from
 *     standard Japanese phonology (mora table, digraphs, sokuon `cl`, moraic nasal
 *     `N`, long vowel mark repeats the previous vowel)
 *   - phoneme -> viseme class, using the common 15-class viseme naming
 *     (sil PP FF TH DD kk CH SS nn RR aa E ih oh ou)
 */
#ifndef JA_PHONEME_H
#define JA_PHONEME_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define JA_N_PHON 43 /* incl. blank */
#define JA_N_KANA 83 /* incl. blank */

static const char *const ja_phoneme_names[JA_N_PHON] = {
    "<blank>", "A", "E", "I", "N", "O", "U", "a", "b", "by", "ch", "cl", "d", "dy", "e", "f", "g", "gy",
    "h", "hy", "i", "j", "k", "ky", "m", "my", "n", "ny", "o", "p", "py", "r", "ry", "s", "sh", "t",
    "ts", "ty", "u", "v", "w", "y", "z"
};

static const char *const ja_kana_names[JA_N_KANA] = {
    "<blank>",
    "あ", "い", "う", "え", "お", "か", "き", "く", "け", "こ", "さ", "し", "す", "せ", "そ",
    "た", "ち", "つ", "て", "と", "な", "に", "ぬ", "ね", "の", "は", "ひ", "ふ", "へ", "ほ",
    "ま", "み", "む", "め", "も", "や", "ゆ", "よ", "ら", "り", "る", "れ", "ろ", "わ", "を", "ん",
    "が", "ぎ", "ぐ", "げ", "ご", "ざ", "じ", "ず", "ぜ", "ぞ", "だ", "ぢ", "づ", "で", "ど",
    "ば", "び", "ぶ", "べ", "ぼ", "ぱ", "ぴ", "ぷ", "ぺ", "ぽ",
    "ぁ", "ぃ", "ぅ", "ぇ", "ぉ", "っ", "ゃ", "ゅ", "ょ", "ゎ", "ー"
};

enum {
    JA_VIS_SIL, JA_VIS_PP, JA_VIS_FF, JA_VIS_TH, JA_VIS_DD, JA_VIS_KK, JA_VIS_CH, JA_VIS_SS,
    JA_VIS_NN, JA_VIS_RR, JA_VIS_AA, JA_VIS_E, JA_VIS_IH, JA_VIS_OH, JA_VIS_OU, JA_N_VIS
};
static const char *const ja_viseme_names[JA_N_VIS] = {
    "sil", "PP", "FF", "TH", "DD", "kk", "CH", "SS", "nn", "RR", "aa", "E", "ih", "oh", "ou"
};

static inline int ja_phoneme_index(const char *p) {
    for (int i = 1; i < JA_N_PHON; i++) if (!strcmp(ja_phoneme_names[i], p)) return i;
    return -1;
}

/* Viseme of a phoneme symbol. Uppercase (devoiced) vowels keep the vowel mouth shape.
 * `cl` (geminate hold) returns -1: the caller extends the following consonant closure. */
static inline int ja_phoneme_viseme(const char *p) {
    if (!strcmp(p, "cl")) return -1;
    switch (p[0]) {
    case 'a': case 'A': return JA_VIS_AA;
    case 'i': case 'I': return JA_VIS_IH;
    case 'u': case 'U': return JA_VIS_OU;
    case 'e': case 'E': return JA_VIS_E;
    case 'o': case 'O': return JA_VIS_OH;
    case 'N': return JA_VIS_NN;
    case 'm': case 'b': case 'p': return JA_VIS_PP;
    case 'f': case 'v': return JA_VIS_FF;
    case 't': return p[1] == 's' ? JA_VIS_SS : JA_VIS_DD;
    case 'd': return JA_VIS_DD;
    case 'n': return JA_VIS_NN;
    case 'k': case 'g': case 'h': return JA_VIS_KK;
    case 'c': case 'j': return JA_VIS_CH;
    case 's': return p[1] == 'h' ? JA_VIS_CH : JA_VIS_SS;
    case 'z': return JA_VIS_SS;
    case 'r': return JA_VIS_RR;
    case 'w': return JA_VIS_OU;
    case 'y': return JA_VIS_IH;
    }
    return JA_VIS_SIL;
}

/* ---- kana reading -> phonemes ---- */

typedef struct { const char *kana; const char *ph; } ja_kana_rule;

/* Hiragana rules; katakana is folded to hiragana first. Two-character entries first. */
static const ja_kana_rule ja_kana_rules[] = {
    /* palatalized digraphs */
    {"きゃ","ky a"},{"きゅ","ky u"},{"きょ","ky o"},{"ぎゃ","gy a"},{"ぎゅ","gy u"},{"ぎょ","gy o"},
    {"しゃ","sh a"},{"しゅ","sh u"},{"しょ","sh o"},{"しぇ","sh e"},{"じゃ","j a"},{"じゅ","j u"},{"じょ","j o"},{"じぇ","j e"},
    {"ちゃ","ch a"},{"ちゅ","ch u"},{"ちょ","ch o"},{"ちぇ","ch e"},{"ぢゃ","j a"},{"ぢゅ","j u"},{"ぢょ","j o"},
    {"にゃ","ny a"},{"にゅ","ny u"},{"にょ","ny o"},{"ひゃ","hy a"},{"ひゅ","hy u"},{"ひょ","hy o"},
    {"びゃ","by a"},{"びゅ","by u"},{"びょ","by o"},{"ぴゃ","py a"},{"ぴゅ","py u"},{"ぴょ","py o"},
    {"みゃ","my a"},{"みゅ","my u"},{"みょ","my o"},{"りゃ","ry a"},{"りゅ","ry u"},{"りょ","ry o"},
    /* loanword combinations */
    {"てぃ","t i"},{"でぃ","d i"},{"とぅ","t u"},{"どぅ","d u"},{"てゅ","ty u"},{"でゅ","dy u"},
    {"ふぁ","f a"},{"ふぃ","f i"},{"ふぇ","f e"},{"ふぉ","f o"},{"ふゅ","hy u"},
    {"うぃ","w i"},{"うぇ","w e"},{"うぉ","w o"},{"ゔぁ","v a"},{"ゔぃ","v i"},{"ゔぇ","v e"},{"ゔぉ","v o"},
    {"つぁ","ts a"},{"つぃ","ts i"},{"つぇ","ts e"},{"つぉ","ts o"},{"いぇ","y e"},{"くぁ","k w a"},{"ぐぁ","g w a"},
    /* single morae */
    {"あ","a"},{"い","i"},{"う","u"},{"え","e"},{"お","o"},
    {"か","k a"},{"き","k i"},{"く","k u"},{"け","k e"},{"こ","k o"},
    {"が","g a"},{"ぎ","g i"},{"ぐ","g u"},{"げ","g e"},{"ご","g o"},
    {"さ","s a"},{"し","sh i"},{"す","s u"},{"せ","s e"},{"そ","s o"},
    {"ざ","z a"},{"じ","j i"},{"ず","z u"},{"ぜ","z e"},{"ぞ","z o"},
    {"た","t a"},{"ち","ch i"},{"つ","ts u"},{"て","t e"},{"と","t o"},
    {"だ","d a"},{"ぢ","j i"},{"づ","z u"},{"で","d e"},{"ど","d o"},
    {"な","n a"},{"に","n i"},{"ぬ","n u"},{"ね","n e"},{"の","n o"},
    {"は","h a"},{"ひ","h i"},{"ふ","f u"},{"へ","h e"},{"ほ","h o"},
    {"ば","b a"},{"び","b i"},{"ぶ","b u"},{"べ","b e"},{"ぼ","b o"},
    {"ぱ","p a"},{"ぴ","p i"},{"ぷ","p u"},{"ぺ","p e"},{"ぽ","p o"},
    {"ま","m a"},{"み","m i"},{"む","m u"},{"め","m e"},{"も","m o"},
    {"や","y a"},{"ゆ","y u"},{"よ","y o"},
    {"ら","r a"},{"り","r i"},{"る","r u"},{"れ","r e"},{"ろ","r o"},
    {"わ","w a"},{"ゐ","i"},{"ゑ","e"},{"を","o"},{"ん","N"},{"っ","cl"},{"ゔ","v u"},
    {"ぁ","a"},{"ぃ","i"},{"ぅ","u"},{"ぇ","e"},{"ぉ","o"},{"ゃ","y a"},{"ゅ","y u"},{"ょ","y o"},{"ゎ","w a"},
};

static inline uint32_t ja__utf8_next(const char **s) {
    const unsigned char *p = (const unsigned char *)*s;
    uint32_t c;
    int n;
    if (p[0] < 0x80) { c = p[0]; n = 1; }
    else if ((p[0] & 0xE0) == 0xC0) { c = p[0] & 0x1F; n = 2; }
    else if ((p[0] & 0xF0) == 0xE0) { c = p[0] & 0x0F; n = 3; }
    else { c = p[0] & 0x07; n = 4; }
    for (int i = 1; i < n && p[i]; i++) c = (c << 6) | (p[i] & 0x3F);
    *s += n;
    return c;
}

static inline int ja__utf8_put(uint32_t c, char *o) {
    if (c < 0x80) { o[0] = (char)c; return 1; }
    if (c < 0x800) { o[0] = (char)(0xC0 | (c >> 6)); o[1] = (char)(0x80 | (c & 0x3F)); return 2; }
    o[0] = (char)(0xE0 | (c >> 12)); o[1] = (char)(0x80 | ((c >> 6) & 0x3F)); o[2] = (char)(0x80 | (c & 0x3F));
    return 3;
}

/* Fold katakana (U+30A1..U+30F6) to hiragana; drop spaces/punctuation. Returns length. */
static inline int ja_kana_normalize(const char *in, char *out, int cap) {
    int n = 0;
    while (*in) {
        uint32_t c = ja__utf8_next(&in);
        if (c >= 0x30A1 && c <= 0x30F6) c -= 0x60;
        int keep = (c >= 0x3041 && c <= 0x3096) || c == 0x30FC;
        if (!keep) continue;
        if (n + 4 >= cap) break;
        n += ja__utf8_put(c, out + n);
    }
    out[n] = 0;
    return n;
}

/* Convert a kana reading to space-separated phonemes (OpenJTalk-style symbols, voiced vowels).
 * Returns the number of phonemes written into ids (model indices), or -1 on an unknown symbol. */
static inline int ja_kana_to_phonemes(const char *kana, int *ids, int cap, char *text, int text_cap) {
    char norm[4096];
    ja_kana_normalize(kana, norm, sizeof(norm));
    int n = 0, tl = 0;
    char last_vowel = 0;
    if (text && text_cap) text[0] = 0;
    for (const char *p = norm; *p; ) {
        if (!strncmp(p, "ー", strlen("ー"))) {
            p += strlen("ー");
            if (!last_vowel) continue;
            char v[2] = { last_vowel, 0 };
            if (n < cap) ids[n++] = ja_phoneme_index(v);
            if (text) tl += snprintf(text + tl, (size_t)(text_cap - tl > 0 ? text_cap - tl : 0), "%s%s", tl ? " " : "", v);
            continue;
        }
        const ja_kana_rule *hit = NULL;
        size_t hl = 0;
        for (size_t r = 0; r < sizeof(ja_kana_rules) / sizeof(ja_kana_rules[0]); r++) {
            size_t kl = strlen(ja_kana_rules[r].kana);
            if (kl > hl && !strncmp(p, ja_kana_rules[r].kana, kl)) { hit = &ja_kana_rules[r]; hl = kl; }
        }
        if (!hit) return -1;
        p += hl;
        char buf[32];
        snprintf(buf, sizeof(buf), "%s", hit->ph);
        for (char *tok = strtok(buf, " "); tok; tok = strtok(NULL, " ")) {
            int id = ja_phoneme_index(tok);
            if (id < 0) return -1;
            if (n < cap) ids[n++] = id;
            if (text) tl += snprintf(text + tl, (size_t)(text_cap - tl > 0 ? text_cap - tl : 0), "%s%s", tl ? " " : "", tok);
            if (strchr("aiueo", tok[0]) && !tok[1]) last_vowel = tok[0];
        }
    }
    return n;
}

/* Kana reading -> kana-head label ids (katakana folded). -1 if a symbol is not in the vocabulary. */
static inline int ja_kana_to_ids(const char *kana, int *ids, int cap) {
    char norm[4096];
    ja_kana_normalize(kana, norm, sizeof(norm));
    int n = 0;
    for (const char *p = norm; *p; ) {
        const char *s = p;
        ja__utf8_next(&p);
        char one[8] = {0};
        memcpy(one, s, (size_t)(p - s));
        int id = -1;
        for (int i = 1; i < JA_N_KANA; i++) if (!strcmp(ja_kana_names[i], one)) { id = i; break; }
        if (id < 0) {
            if (!strcmp(one, "ゐ")) id = 2; else if (!strcmp(one, "ゑ")) id = 4; else if (!strcmp(one, "ゔ")) id = 3;
            else return -1;
        }
        if (n < cap) ids[n++] = id;
    }
    return n;
}

#endif /* JA_PHONEME_H */
