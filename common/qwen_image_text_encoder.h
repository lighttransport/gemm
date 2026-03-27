/*
 * qwen_image_text_encoder.h - Text encoder for Qwen-Image pipeline
 *
 * Wraps transformer.h (Qwen2.5-VL LLM) to extract per-token hidden states
 * as conditioning for the diffusion model.
 *
 * Usage:
 *   #define QIMG_TEXT_ENCODER_IMPLEMENTATION
 *   #include "qwen_image_text_encoder.h"
 *
 * Dependencies: transformer.h, bpe_tokenizer.h, gguf_loader.h
 *
 * API:
 *   qimg_text_enc *qimg_text_enc_load(const char *gguf_path);
 *   void           qimg_text_enc_free(qimg_text_enc *enc);
 *   float         *qimg_text_enc_encode(qimg_text_enc *enc, const char *text,
 *                                       int *out_n_tokens);
 */
#ifndef QIMG_TEXT_ENCODER_H
#define QIMG_TEXT_ENCODER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void *model;     /* transformer_model* */
    void *vocab;     /* bpe_vocab* */
    void *gguf;      /* gguf_context* (kept for mmap lifetime) */
    int   n_embd;    /* 3584 for Qwen2.5-VL-7B */
    int   n_vocab;
} qimg_text_enc;

qimg_text_enc *qimg_text_enc_load(const char *gguf_path);
void           qimg_text_enc_free(qimg_text_enc *enc);

/* Encode text prompt to hidden states.
 * Returns [n_tokens, n_embd] float array (caller must free).
 * Sets *out_n_tokens to the number of tokens. */
float *qimg_text_enc_encode(qimg_text_enc *enc, const char *text,
                            int *out_n_tokens);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef QIMG_TEXT_ENCODER_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "transformer.h"
#include "bpe_tokenizer.h"

qimg_text_enc *qimg_text_enc_load(const char *gguf_path) {
    fprintf(stderr, "qimg_text_enc: loading %s\n", gguf_path);
    gguf_context *gguf = gguf_open(gguf_path, 1);
    if (!gguf) {
        fprintf(stderr, "qimg_text_enc: failed to open GGUF\n");
        return NULL;
    }

    /* Load tokenizer from GGUF metadata */
    bpe_vocab *vocab = bpe_vocab_load(gguf);
    if (!vocab) {
        fprintf(stderr, "qimg_text_enc: failed to load tokenizer\n");
        gguf_close(gguf);
        return NULL;
    }

    /* Load transformer model */
    transformer_model *model = transformer_load(gguf, 2048);
    if (!model) {
        fprintf(stderr, "qimg_text_enc: failed to load model\n");
        bpe_vocab_free(vocab);
        gguf_close(gguf);
        return NULL;
    }

    transformer_set_threads(model, 1);

    qimg_text_enc *enc = (qimg_text_enc *)calloc(1, sizeof(qimg_text_enc));
    enc->model = model;
    enc->vocab = vocab;
    enc->gguf = gguf;
    enc->n_embd = model->n_embd;
    enc->n_vocab = model->n_vocab;

    fprintf(stderr, "qimg_text_enc: loaded (n_embd=%d, n_vocab=%d, n_layers=%d)\n",
            model->n_embd, model->n_vocab, model->n_layers);
    return enc;
}

void qimg_text_enc_free(qimg_text_enc *enc) {
    if (!enc) return;
    if (enc->model) transformer_free((transformer_model *)enc->model);
    if (enc->vocab) bpe_vocab_free((bpe_vocab *)enc->vocab);
    if (enc->gguf) gguf_close((gguf_context *)enc->gguf);
    free(enc);
}

float *qimg_text_enc_encode(qimg_text_enc *enc, const char *text,
                            int *out_n_tokens) {
    transformer_model *model = (transformer_model *)enc->model;
    bpe_vocab *vocab = (bpe_vocab *)enc->vocab;
    int n_embd = enc->n_embd;

    /* Tokenize */
    int text_len = (int)strlen(text);
    int n_tokens = bpe_tokenize(vocab, text, text_len, NULL, 0);
    if (n_tokens <= 0) {
        fprintf(stderr, "qimg_text_enc: tokenization failed\n");
        *out_n_tokens = 0;
        return NULL;
    }

    int32_t *tokens = (int32_t *)malloc((size_t)n_tokens * sizeof(int32_t));
    bpe_tokenize(vocab, text, text_len, tokens, n_tokens);

    fprintf(stderr, "qimg_text_enc: tokenized \"%s\" -> %d tokens: [",
            text, n_tokens);
    for (int i = 0; i < n_tokens && i < 10; i++)
        fprintf(stderr, "%d%s", tokens[i], i < n_tokens - 1 ? ", " : "");
    if (n_tokens > 10) fprintf(stderr, "...");
    fprintf(stderr, "]\n");

    /* Run each token through transformer and collect hidden states */
    float *hidden_states = (float *)malloc((size_t)n_tokens * n_embd * sizeof(float));

    for (int i = 0; i < n_tokens; i++) {
        /* Forward pass: returns hidden state after final RMSNorm, before lm_head */
        float *h = transformer_forward(model, tokens[i], i);
        if (!h) {
            fprintf(stderr, "qimg_text_enc: forward failed at token %d\n", i);
            free(tokens);
            free(hidden_states);
            *out_n_tokens = 0;
            return NULL;
        }
        memcpy(hidden_states + (size_t)i * n_embd, h, (size_t)n_embd * sizeof(float));

        if (i == 0 || (i + 1) % 10 == 0 || i == n_tokens - 1)
            fprintf(stderr, "\r  qimg_text_enc: token %d/%d", i + 1, n_tokens);
    }
    fprintf(stderr, "\n");

    free(tokens);
    *out_n_tokens = n_tokens;
    return hidden_states;
}

#endif /* QIMG_TEXT_ENCODER_IMPLEMENTATION */
#endif /* QIMG_TEXT_ENCODER_H */
