/*
 * test_transformer.c - Test transformer inference on a GGUF model
 *
 * Usage:
 *   ./test_transformer <model.gguf> [prompt] [max_gen_tokens]
 *
 * For embedding models: prints hidden state stats per token
 * For generative models: runs autoregressive generation
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

static void print_hidden_stats(const float *hidden, int n_embd) {
    float mean = 0.0f, norm = 0.0f;
    float min_val = hidden[0], max_val = hidden[0];
    for (int j = 0; j < n_embd; j++) {
        mean += hidden[j];
        norm += hidden[j] * hidden[j];
        if (hidden[j] < min_val) min_val = hidden[j];
        if (hidden[j] > max_val) max_val = hidden[j];
    }
    mean /= n_embd;
    norm = sqrtf(norm);
    fprintf(stderr, "  Hidden: mean=%.6f norm=%.4f min=%.4f max=%.4f\n",
            mean, norm, min_val, max_val);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [prompt] [max_gen_tokens]\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *prompt = (argc >= 3) ? argv[2] : "Hello";
    int max_gen = (argc >= 4) ? atoi(argv[3]) : 32;

    srand((unsigned)time(NULL));

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
    int max_seq_len = 256;
    transformer_model *model = transformer_load(gguf, max_seq_len);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        bpe_vocab_free(vocab);
        gguf_close(gguf);
        return 1;
    }

    /* Tokenize prompt */
    fprintf(stderr, "Prompt: \"%s\"\n", prompt);
    int32_t tokens[256];
    int n_tokens = bpe_tokenize(vocab, prompt, -1, tokens, 256);
    fprintf(stderr, "Tokens (%d):", n_tokens);
    for (int i = 0; i < n_tokens; i++) {
        fprintf(stderr, " %d(\"%s\")", tokens[i], bpe_token_to_str(vocab, tokens[i]));
    }
    fprintf(stderr, "\n");

    if (model->has_lm_head) {
        /* --- Generative mode --- */
        fprintf(stderr, "\n=== Generative mode (max_gen=%d) ===\n", max_gen);

        /* Process prompt tokens */
        int pos = 0;
        for (int i = 0; i < n_tokens; i++) {
            fprintf(stderr, "  [prefill %d/%d] token=%d pos=%d\n", i+1, n_tokens, tokens[i], pos);
            clock_t t0 = clock();

            float *logits;
            if (i == n_tokens - 1) {
                logits = transformer_forward_logits(model, tokens[i], pos);
            } else {
                transformer_forward(model, tokens[i], pos);
                logits = NULL;
            }
            clock_t t1 = clock();
            fprintf(stderr, "    %.3f s\n", (double)(t1 - t0) / CLOCKS_PER_SEC);

            if (logits) {
                /* Find argmax for greedy check */
                int32_t argmax = 0;
                for (int j = 1; j < model->n_vocab; j++) {
                    if (logits[j] > logits[argmax]) argmax = j;
                }
                fprintf(stderr, "  Greedy next: %d \"%s\" (logit=%.4f)\n",
                        argmax, bpe_token_to_str(vocab, argmax), logits[argmax]);
            }
            pos++;
        }

        /* Autoregressive generation */
        fprintf(stderr, "\n--- Generation ---\n");
        printf("%s", prompt);
        fflush(stdout);

        int32_t next_token = -1;
        for (int g = 0; g < max_gen; g++) {
            float *logits;
            if (g == 0) {
                /* Logits already computed for last prompt token; sample from them */
                logits = model->logits;
            } else {
                clock_t t0 = clock();
                logits = transformer_forward_logits(model, next_token, pos);
                clock_t t1 = clock();
                fprintf(stderr, "  [gen %d] pos=%d %.3f s\n", g, pos, (double)(t1 - t0) / CLOCKS_PER_SEC);
                pos++;
            }

            if (!logits) break;

            /* Greedy sampling (temperature=0) for deterministic output */
            next_token = 0;
            for (int j = 1; j < model->n_vocab; j++) {
                if (logits[j] > logits[next_token]) next_token = j;
            }

            /* Check for EOS */
            if (next_token == vocab->eos_id || next_token == vocab->eot_id) {
                fprintf(stderr, "  [EOS token %d]\n", next_token);
                break;
            }

            const char *tok_str = bpe_token_to_str(vocab, next_token);
            if (tok_str) {
                /* Decode GPT-2 byte encoding back to raw text */
                int dec_len;
                char *decoded = bpe_byte_decode(tok_str, (int)strlen(tok_str), &dec_len);
                fwrite(decoded, 1, dec_len, stdout);
                fflush(stdout);
                free(decoded);
            }

            if (g == 0) pos++; /* account for the first generated token's position */
        }
        printf("\n");

    } else {
        /* --- Embedding mode --- */
        fprintf(stderr, "\n=== Embedding mode ===\n");
        for (int i = 0; i < n_tokens; i++) {
            fprintf(stderr, "\n--- Token %d/%d: id=%d \"%s\" pos=%d ---\n",
                    i + 1, n_tokens, tokens[i],
                    bpe_token_to_str(vocab, tokens[i]), i);

            clock_t t0 = clock();
            float *hidden = transformer_forward(model, tokens[i], i);
            clock_t t1 = clock();

            print_hidden_stats(hidden, model->n_embd);
            fprintf(stderr, "  First 8:");
            for (int j = 0; j < 8 && j < model->n_embd; j++)
                fprintf(stderr, " %.4f", hidden[j]);
            fprintf(stderr, "\n");
            fprintf(stderr, "  Time: %.3f s\n", (double)(t1 - t0) / CLOCKS_PER_SEC);
        }
    }

    /* Cleanup */
    transformer_free(model);
    bpe_vocab_free(vocab);
    gguf_close(gguf);

    fprintf(stderr, "\nDone.\n");
    return 0;
}
