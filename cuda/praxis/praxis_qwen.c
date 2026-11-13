/*
 * praxis_qwen.c - in-process Qwen3.5-2B text conditioning for praxis_e2e (separate translation unit: it carries the transformer.h /
 * gguf_loader.h implementations and links cuda/llm/cuda_llm_runner.c; praxis_e2e.c only sees the tiny API in praxis_qwen.h).
 * cond = hidden_states[-3] (output of layer 22 of 24, pre final norm), ids right-padded with <|endoftext|> to a multiple of 32, padding rows dropped.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#define GGUF_LOADER_IMPLEMENTATION
#include "../../common/gguf_loader.h"
#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"
#define TRANSFORMER_IMPLEMENTATION
#include "../../common/transformer.h"
#include "../../common/safetensors.h"   /* json_* declarations (implementation lives in praxis_e2e.c) */
#define BPE_TOKENIZER_IMPLEMENTATION
#include "../../common/bpe_tokenizer.h"
#include "../llm/cuda_llm_runner.h"
#include "praxis_qwen.h"

struct praxis_qwen { cuda_llm_runner *r; int max_seq; };

praxis_qwen *praxis_qwen_open(const char *dir, int max_seq) {
    praxis_qwen *q = (praxis_qwen *)calloc(1, sizeof *q);
    q->max_seq = max_seq;
    q->r = cuda_llm_init(0, 0);
    if (!q->r) { free(q); return NULL; }
    if (cuda_llm_load_weights_qwen35_safetensors(q->r, dir, max_seq)) { cuda_llm_free(q->r); free(q); return NULL; }
    return q;
}

/* ids[n] -> malloc'd float [n][2048] (real rows only); NULL on error */
float *praxis_qwen_cond(praxis_qwen *q, const int32_t *ids, int n) {
    const int pad_to = 32, n_embd = 2048, cond_slot = 22;
    int np = (n + pad_to - 1) / pad_to * pad_to;
    if (np > q->max_seq) { fprintf(stderr, "praxis_qwen: %d tokens exceed max_seq %d\n", np, q->max_seq); return NULL; }
    int32_t *p = (int32_t *)malloc((size_t)np * 4);
    memcpy(p, ids, (size_t)n * 4);
    for (int i = n; i < np; i++) p[i] = 151643;
    float *all = (float *)malloc((size_t)np * n_embd * 4);
    int rc = cuda_llm_prefill_hidden_ex(q->r, p, np, cond_slot, 0, cond_slot, all, NULL);
    free(p);
    if (rc) { free(all); return NULL; }
    return all;   /* caller keeps the first n rows (the buffer is np rows; extra rows are harmless) */
}

void praxis_qwen_close(praxis_qwen *q) {
    if (!q) return;
    cuda_llm_free(q->r);
    free(q);
}

/* ---------------- tokenizer.json (HF byte-level BPE, Qwen3.5 pre-tokenizer) ----------------
 * Builds the bpe_vocab of common/bpe_tokenizer.h straight from tokenizer.json: model.vocab + added_tokens (all matched literally, like HF),
 * model.merges ("a b" strings or [a,b] pairs), pre-tokenizer = Qwen3.5 regex (marks grouped with letters, as common/bpe_tokenizer.h implements it).
 * Not implemented: the NFC normalizer (non-NFC input differs from HF), the Split regex is NOT parsed (it is assumed to be the Qwen3.5 one).
 * No BOS/EOS is added (the ByteLevel post-processor adds none). */
struct praxis_tok { bpe_vocab *v; };

praxis_tok *praxis_tok_load(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) { fprintf(stderr, "praxis_tok: cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(n + 1);
    if (fread(buf, 1, n, f) != (size_t)n) { fclose(f); free(buf); return NULL; }
    fclose(f); buf[n] = 0;
    json_val *root = json_parse(buf, (int)n); free(buf);
    if (!root) return NULL;
    json_val *model = json_obj_get(root, "model"), *added = json_obj_get(root, "added_tokens");
    json_val *vocab = model ? json_obj_get(model, "vocab") : NULL, *merges = model ? json_obj_get(model, "merges") : NULL;
    json_val *ty = model ? json_obj_get(model, "type") : NULL;
    if (!vocab || vocab->type != JSON_OBJECT || !merges || merges->type != JSON_ARRAY || !ty || ty->type != JSON_STRING || strcmp(ty->str.ptr, "BPE")) {
        fprintf(stderr, "praxis_tok: %s is not a BPE tokenizer.json\n", path); json_free(root); return NULL; }
    int maxid = -1;
    for (int i = 0; i < vocab->obj.count; i++) if ((int)vocab->obj.vals[i].num > maxid) maxid = (int)vocab->obj.vals[i].num;
    if (added && added->type == JSON_ARRAY)
        for (int i = 0; i < added->arr.count; i++) { json_val *id = json_obj_get(&added->arr.items[i], "id"); if (id && (int)id->num > maxid) maxid = (int)id->num; }
    int nt = maxid + 1;
    bpe_vocab *v = (bpe_vocab *)calloc(1, sizeof *v);
    v->n_tokens = nt; v->token_strs = (char **)calloc(nt, sizeof(char *)); v->token_str_lens = (int *)calloc(nt, sizeof(int));
    v->eos_id = v->bos_id = v->eot_id = v->pad_id = v->unk_id = -1;
    v->pre_type = BPE_PRE_TYPE_QWEN35;
    bpe_hm_init(&v->token_to_id, nt * 2 + 1);
    for (int i = 0; i < vocab->obj.count; i++) {
        int id = (int)vocab->obj.vals[i].num; const char *k = vocab->obj.keys[i]; int kl = (int)strlen(k);
        v->token_strs[id] = (char *)malloc(kl + 1); memcpy(v->token_strs[id], k, kl + 1); v->token_str_lens[id] = kl;
        bpe_hm_set(&v->token_to_id, k, kl, id);
    }
    int ns = 0;
    if (added && added->type == JSON_ARRAY) ns = added->arr.count;
    v->n_special = ns; v->special_ids = (int32_t *)malloc((ns + 1) * 4); v->special_strs = (char **)malloc((ns + 1) * sizeof(char *)); v->special_lens = (int *)malloc((ns + 1) * sizeof(int));
    for (int i = 0; i < ns; i++) {
        json_val *a = &added->arr.items[i]; json_val *id = json_obj_get(a, "id"), *c = json_obj_get(a, "content");
        int ii = (int)id->num, cl = (int)strlen(c->str.ptr);
        if (!v->token_strs[ii]) { v->token_strs[ii] = (char *)malloc(cl + 1); memcpy(v->token_strs[ii], c->str.ptr, cl + 1); v->token_str_lens[ii] = cl; bpe_hm_set(&v->token_to_id, c->str.ptr, cl, ii); }
        v->special_ids[i] = ii; v->special_strs[i] = v->token_strs[ii]; v->special_lens[i] = v->token_str_lens[ii];
    }
    for (int i = 0; i < ns - 1; i++) for (int j = i + 1; j < ns; j++) if (v->special_lens[j] > v->special_lens[i]) {   /* longest first */
        int32_t ti = v->special_ids[i]; v->special_ids[i] = v->special_ids[j]; v->special_ids[j] = ti;
        char *ts = v->special_strs[i]; v->special_strs[i] = v->special_strs[j]; v->special_strs[j] = ts;
        int tl = v->special_lens[i]; v->special_lens[i] = v->special_lens[j]; v->special_lens[j] = tl; }
    bpe_hm_init(&v->merge_ranks, merges->arr.count * 2 + 1);
    for (int i = 0; i < merges->arr.count; i++) {
        json_val *m = &merges->arr.items[i]; const char *l, *r; int ll, rl; char *tmp = NULL;
        if (m->type == JSON_ARRAY && m->arr.count == 2) { l = m->arr.items[0].str.ptr; r = m->arr.items[1].str.ptr; ll = (int)strlen(l); rl = (int)strlen(r); }
        else if (m->type == JSON_STRING) {
            const char *s = m->str.ptr; int sl = (int)strlen(s), j; for (j = 1; j < sl && s[j] != ' '; j++) ;
            if (j >= sl) continue; l = s; ll = j; r = s + j + 1; rl = sl - j - 1;
        } else continue;
        int kl = ll + 1 + rl; tmp = (char *)malloc(kl); memcpy(tmp, l, ll); tmp[ll] = 0; memcpy(tmp + ll + 1, r, rl);
        bpe_hm_set(&v->merge_ranks, tmp, kl, i); free(tmp);
    }
    json_free(root);
    praxis_tok *t = (praxis_tok *)calloc(1, sizeof *t); t->v = v; return t;
}
int praxis_tok_encode(praxis_tok *t, const char *text, int32_t *ids, int max_ids) {
    /* the NFC normalizer is not implemented: warn when the text contains code points that NFC would compose (combining marks, Hangul jamo, kana voicing marks) */
    for (const unsigned char *p = (const unsigned char *)text; *p; p++)
        if ((p[0] == 0xCC || (p[0] == 0xCD && p[1] < 0xB0)) || (p[0] == 0xE1 && p[1] >= 0x84 && p[1] <= 0x87) || (p[0] == 0xE3 && p[1] == 0x82 && (p[2] == 0x99 || p[2] == 0x9A))) {
            fprintf(stderr, "praxis_tok: warning: input contains decomposed/combining characters; HF applies NFC first, ids may differ (NFC not implemented)\n"); break; }
    int n = bpe_tokenize(t->v, text, (int)strlen(text), ids, max_ids);
    return n > max_ids ? -1 : n;
}
void praxis_tok_free(praxis_tok *t) { if (t) { bpe_vocab_free(t->v); free(t); } }
