/*
 * test_transformer.c - Test transformer inference on a GGUF model
 *
 * Usage: ./test_transformer <model.gguf>
 */

#define GGUF_LOADER_IMPLEMENTATION
#include "gguf_loader.h"

#define GGML_DEQUANT_IMPLEMENTATION
#include "ggml_dequant.h"

#define BPE_TOKENIZER_IMPLEMENTATION
#include "bpe_tokenizer.h"

#define TRANSFORMER_IMPLEMENTATION
#include "transformer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [prompt]\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *prompt = (argc >= 3) ? argv[2] : "Hello world!";

    /* Load GGUF */
    fprintf(stderr, "Loading GGUF: %s\n", model_path);
    gguf_context *gguf = gguf_open(model_path, 1);
    if (!gguf) {
        fprintf(stderr, "Failed to open GGUF\n");
        return 1;
    }
    fprintf(stderr, "GGUF: %lu tensors, %lu KV pairs\n",
            (unsigned long)gguf->n_tensors, (unsigned long)gguf->n_kv);

    /* Load vocab */
    bpe_vocab *vocab = bpe_vocab_load(gguf);
    if (!vocab) {
        fprintf(stderr, "Failed to load vocab\n");
        gguf_close(gguf);
        return 1;
    }
    fprintf(stderr, "Vocab: %d tokens\n", vocab->n_tokens);

    /* Load model */
    int max_seq_len = 128;
    transformer_model *model = transformer_load(gguf, max_seq_len);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        bpe_vocab_free(vocab);
        gguf_close(gguf);
        return 1;
    }

    /* Tokenize */
    fprintf(stderr, "Prompt: \"%s\"\n", prompt);
    int32_t tokens[256];
    int n_tokens = bpe_tokenize(vocab, prompt, -1, tokens, 256);
    fprintf(stderr, "Tokens (%d):", n_tokens);
    for (int i = 0; i < n_tokens; i++) {
        fprintf(stderr, " %d", tokens[i]);
    }
    fprintf(stderr, "\n");

    /* Forward pass for each token */
    for (int i = 0; i < n_tokens; i++) {
        fprintf(stderr, "\n--- Token %d/%d: id=%d \"%s\" pos=%d ---\n",
                i + 1, n_tokens, tokens[i],
                bpe_token_to_str(vocab, tokens[i]), i);

        clock_t t0 = clock();
        float *hidden = transformer_forward(model, tokens[i], i);
        clock_t t1 = clock();
        double elapsed = (double)(t1 - t0) / CLOCKS_PER_SEC;

        /* Compute stats */
        float mean = 0.0f, norm = 0.0f;
        float min_val = hidden[0], max_val = hidden[0];
        for (int j = 0; j < model->n_embd; j++) {
            mean += hidden[j];
            norm += hidden[j] * hidden[j];
            if (hidden[j] < min_val) min_val = hidden[j];
            if (hidden[j] > max_val) max_val = hidden[j];
        }
        mean /= model->n_embd;
        norm = sqrtf(norm);

        fprintf(stderr, "  Hidden state: mean=%.6f norm=%.4f min=%.4f max=%.4f\n",
                mean, norm, min_val, max_val);
        fprintf(stderr, "  First 8 values:");
        for (int j = 0; j < 8 && j < model->n_embd; j++) {
            fprintf(stderr, " %.4f", hidden[j]);
        }
        fprintf(stderr, "\n");
        fprintf(stderr, "  Time: %.3f s\n", elapsed);
    }

    /* Cleanup */
    transformer_free(model);
    bpe_vocab_free(vocab);
    gguf_close(gguf);

    fprintf(stderr, "\nDone.\n");
    return 0;
}
