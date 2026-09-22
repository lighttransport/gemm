#ifndef QIMG21_QWEN_TOKENIZER_JSON_H
#define QIMG21_QWEN_TOKENIZER_JSON_H

/* Adapter from Hugging Face tokenizer.json to common/bpe_tokenizer.h's
 * Unicode-aware Qwen2 byte-BPE engine.  Qwen-Image 2.1 fixes the vocabulary
 * size and ChatML token IDs, so reject incompatible snapshots explicitly. */
#include "../../common/glm5_bpe.h"

static bpe_vocab *q21_bpe_load_json(const char *path) {
    glm5_bpe source;
    if (glm5_bpe_load(path, &source) != 0) return NULL;
    int max_id = 151935;
    for (size_t i = 0; i < source.n_vocab; i++)
        if (source.vocab[i].id > max_id) max_id = source.vocab[i].id;
    if (max_id >= 200000) { glm5_bpe_free(&source); return NULL; }
    bpe_vocab *v = (bpe_vocab *)calloc(1, sizeof(*v));
    if (!v) { glm5_bpe_free(&source); return NULL; }
    v->n_tokens = max_id + 1;
    v->token_strs = (char **)calloc((size_t)v->n_tokens, sizeof(char *));
    v->token_str_lens = (int *)calloc((size_t)v->n_tokens, sizeof(int));
    if (!v->token_strs || !v->token_str_lens) goto fail;
    bpe_hm_init(&v->token_to_id, (int)source.n_vocab * 2 + 64);
    bpe_hm_init(&v->merge_ranks, (int)source.n_merges * 2 + 64);
    for (size_t i = 0; i < source.n_vocab; i++) {
        int id = source.vocab[i].id;
        if (id < 0 || id >= v->n_tokens || v->token_strs[id]) goto fail;
        v->token_strs[id] = glm5_bpe_strdup(source.vocab[i].text);
        if (!v->token_strs[id]) goto fail;
        v->token_str_lens[id] = (int)strlen(v->token_strs[id]);
        bpe_hm_set(&v->token_to_id, v->token_strs[id], v->token_str_lens[id], id);
    }
    for (size_t i = 0; i < source.n_merges; i++) {
        int a = (int)strlen(source.merges[i].left);
        int b = (int)strlen(source.merges[i].right);
        char *key = (char *)malloc((size_t)a + b + 1);
        if (!key) goto fail;
        memcpy(key, source.merges[i].left, (size_t)a);
        key[a] = '\0';
        memcpy(key + a + 1, source.merges[i].right, (size_t)b);
        bpe_hm_set(&v->merge_ranks, key, a + 1 + b, source.merges[i].rank);
        free(key);
    }
    const char *special[] = {"<|im_start|>", "<|im_end|>"};
    const int ids[] = {151644, 151645};
    v->n_special = 2;
    v->special_ids = (int32_t *)malloc(2 * sizeof(*v->special_ids));
    v->special_strs = (char **)malloc(2 * sizeof(*v->special_strs));
    v->special_lens = (int *)malloc(2 * sizeof(*v->special_lens));
    if (!v->special_ids || !v->special_strs || !v->special_lens) goto fail;
    for (int i = 0; i < 2; i++) {
        if (ids[i] >= v->n_tokens) goto fail;
        if (!v->token_strs[ids[i]]) v->token_strs[ids[i]] = glm5_bpe_strdup(special[i]);
        if (!v->token_strs[ids[i]]) goto fail;
        v->token_str_lens[ids[i]] = (int)strlen(special[i]);
        v->special_ids[i] = ids[i];
        v->special_strs[i] = v->token_strs[ids[i]];
        v->special_lens[i] = v->token_str_lens[ids[i]];
    }
    v->pre_type = BPE_PRE_TYPE_QWEN2;
    v->eos_id = v->eot_id = 151645;
    v->bos_id = v->pad_id = v->unk_id = -1;
    glm5_bpe_free(&source);
    return v;
fail:
    glm5_bpe_free(&source);
    bpe_vocab_free(v);
    return NULL;
}

static int q21_build_prompt_tokens(const char *tokenizer_json, const char *prompt,
                                   int32_t *tokens, int capacity, int *drop_prefix) {
    static const char system[] = "Comprehend and analyze the provided prompt.";
    bpe_vocab *v = q21_bpe_load_json(tokenizer_json);
    if (!v) return -1;
    size_t needed = strlen(system) + strlen(prompt) + 160;
    char *text = (char *)malloc(needed);
    if (!text) { bpe_vocab_free(v); return -1; }
    snprintf(text, needed,
             "<|im_start|>system\n%s<|im_end|>\n"
             "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
             system, *prompt ? prompt : " ");
    int n = bpe_tokenize(v, text, -1, tokens, capacity);
    snprintf(text, needed, "<|im_start|>system\n%s<|im_end|>\n", system);
    int drop = bpe_tokenize(v, text, -1, NULL, 0);
    free(text);
    bpe_vocab_free(v);
    if (n < 0 || n > capacity || drop < 0 || drop > n) return -1;
    *drop_prefix = drop;
    return n;
}

static int q21_build_multimodal_prompt_tokens(const char *tokenizer_json,
                                               const char *prompt, int image_tokens,
                                               int32_t *tokens, int capacity,
                                               int *drop_prefix, int *image_start) {
    static const char system[] = "Comprehend and analyze the provided prompt.";
    if (image_tokens <= 0 || capacity <= image_tokens + 8) return -1;
    bpe_vocab *v = q21_bpe_load_json(tokenizer_json);
    if (!v) return -1;
    size_t needed = strlen(system) + strlen(prompt) + 192;
    char *text = (char *)malloc(needed);
    if (!text) { bpe_vocab_free(v); return -1; }
    snprintf(text, needed,
             "<|im_start|>system\n%s<|im_end|>\n"
             "<|im_start|>user\n<image1>", system);
    int n = bpe_tokenize(v, text, -1, tokens, capacity);
    snprintf(text, needed, "<|im_start|>system\n%s<|im_end|>\n", system);
    int drop = bpe_tokenize(v, text, -1, NULL, 0);
    if (n < 0 || drop < 0 || n + image_tokens + 2 > capacity) goto fail;
    tokens[n++] = 151652; /* <|vision_start|> */
    *image_start = n;
    for (int i = 0; i < image_tokens; i++) tokens[n++] = 151655; /* <|image_pad|> */
    tokens[n++] = 151653; /* <|vision_end|> */
    snprintf(text, needed, "%s<|im_end|>\n<|im_start|>assistant\n",
             *prompt ? prompt : " ");
    int suffix = bpe_tokenize(v, text, -1, tokens + n, capacity - n);
    if (suffix < 0 || n + suffix > capacity) goto fail;
    n += suffix;
    free(text);
    bpe_vocab_free(v);
    *drop_prefix = drop;
    return n;
fail:
    free(text);
    bpe_vocab_free(v);
    return -1;
}
#endif
