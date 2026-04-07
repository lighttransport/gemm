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
#include "../../common/gguf_loader.h"

#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"

#define BPE_TOKENIZER_IMPLEMENTATION
#include "../../common/bpe_tokenizer.h"

#define TRANSFORMER_IMPLEMENTATION
#include "../../common/transformer.h"

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

/* Synthetic benchmark: allocate a fake model with zero weights, run forward passes.
 * No file I/O, pure compute+memory measurement. */
static int run_bench(int n_embd, int n_heads, int n_kv_heads, int head_dim,
                      int n_ff, int n_layers, int n_tokens, int n_threads) {
    /* Allocate a fake GGUF data block sized for BF16 weights */
    size_t per_layer = (size_t)(
        (size_t)n_embd * n_heads * head_dim +    /* attn_q */
        (size_t)n_embd * n_kv_heads * head_dim + /* attn_k */
        (size_t)n_embd * n_kv_heads * head_dim + /* attn_v */
        (size_t)n_embd * n_heads * head_dim +    /* attn_output */
        (size_t)n_embd * n_ff +                  /* ffn_gate */
        (size_t)n_embd * n_ff +                  /* ffn_up */
        (size_t)n_ff * n_embd +                  /* ffn_down */
        (size_t)n_embd * 2                       /* norms */
    ) * 2;  /* BF16 = 2 bytes per element */
    size_t total = per_layer * n_layers + (size_t)n_embd * 248320 * 2; /* + token_embd */
    fprintf(stderr, "bench: synthetic model %dx%d, %d layers, %.1f GB BF16\n",
            n_embd, n_ff, n_layers, (double)total / (1024.0*1024.0*1024.0));

    /* Allocate zeroed weight buffer */
    void *weights = NULL;
    size_t align = 2 * 1024 * 1024;
    size_t alloc_size = (total + align - 1) & ~(align - 1);
    if (posix_memalign(&weights, align, alloc_size) != 0) {
        fprintf(stderr, "bench: failed to allocate %.1f GB\n", (double)alloc_size/(1024.0*1024.0*1024.0));
        return 1;
    }

    /* NUMA distribute the weight buffer if multi-threaded */
    if (n_threads > 1 && getenv("NUMA_DISTRIBUTE")) {
        fprintf(stderr, "bench: NUMA distributing weights...\n");
        size_t chunk = alloc_size / n_threads;
        /* Touch each thread's partition from the corresponding thread
         * (simplified: main thread touches all for now) */
        memset(weights, 0, alloc_size);
    } else {
        memset(weights, 0, alloc_size);
    }

    /* Build a minimal transformer_model manually */
    int max_seq_len = 256;
    int kv_dim = n_kv_heads * head_dim;
    int q_dim = n_heads * head_dim;
    int max_dim = n_embd > n_ff ? n_embd : n_ff;

    transformer_model *m = (transformer_model *)calloc(1, sizeof(transformer_model));
    m->n_layers = n_layers; m->n_embd = n_embd; m->n_heads = n_heads;
    m->n_kv_heads = n_kv_heads; m->head_dim = head_dim; m->n_ff = n_ff;
    m->n_vocab = 248320; m->max_seq_len = max_seq_len; m->has_lm_head = 0;
    m->rms_norm_eps = 1e-6f;

    /* Point all weight tensors into the zeroed buffer */
    uint8_t *wp = (uint8_t *)weights;
    #define MAKE_TENSOR(rows, cols) \
        (qtensor){wp, GGML_TYPE_BF16, (rows), (cols), 2, {(cols), (rows), 0, 0}}; \
        wp += (size_t)(rows) * (cols) * 2

    m->token_embd = MAKE_TENSOR(m->n_vocab, n_embd);
    m->output_norm = MAKE_TENSOR(1, n_embd);
    m->layers = (transformer_layer *)calloc(n_layers, sizeof(transformer_layer));

    m->key_cache = (float **)calloc(n_layers, sizeof(float *));
    m->value_cache = (float **)calloc(n_layers, sizeof(float *));
    for (int l = 0; l < n_layers; l++) {
        transformer_layer *layer = &m->layers[l];
        layer->attn_norm = MAKE_TENSOR(1, n_embd);
        layer->attn_q = MAKE_TENSOR(q_dim, n_embd);
        layer->attn_k = MAKE_TENSOR(kv_dim, n_embd);
        layer->attn_v = MAKE_TENSOR(kv_dim, n_embd);
        layer->attn_output = MAKE_TENSOR(n_embd, q_dim);
        layer->ffn_norm = MAKE_TENSOR(1, n_embd);
        layer->ffn_gate = MAKE_TENSOR(n_ff, n_embd);
        layer->ffn_up = MAKE_TENSOR(n_ff, n_embd);
        layer->ffn_down = MAKE_TENSOR(n_embd, n_ff);
        m->key_cache[l] = (float *)calloc(max_seq_len * kv_dim, sizeof(float));
        m->value_cache[l] = (float *)calloc(max_seq_len * kv_dim, sizeof(float));
    }
    #undef MAKE_TENSOR

    /* Scratch buffers */
    m->x = (float *)calloc(n_embd, sizeof(float));
    m->xb = (float *)calloc(n_embd, sizeof(float));
    m->xb2 = (float *)calloc(q_dim > n_ff ? q_dim : n_ff, sizeof(float));
    m->q = (float *)calloc(q_dim, sizeof(float));
    m->k = (float *)calloc(kv_dim, sizeof(float));
    m->v = (float *)calloc(kv_dim, sizeof(float));
    m->att = (float *)calloc(n_heads * max_seq_len, sizeof(float));
    m->ffn_buf1 = (float *)calloc(n_ff, sizeof(float));
    m->ffn_buf2 = (float *)calloc(n_ff, sizeof(float));
    m->ffn_buf3 = (float *)calloc(n_ff, sizeof(float));
    m->matvec_tmp = (float *)calloc(max_dim, sizeof(float));
    m->rope_inv_freq = (float *)calloc(head_dim / 2, sizeof(float));
    m->n_threads = 1;
    m->thread_tmp = (float **)calloc(1, sizeof(float *));
    m->thread_tmp[0] = m->matvec_tmp;

    if (n_threads > 1) transformer_set_threads(m, n_threads);

    /* Warmup: 2 forward passes */
    for (int w = 0; w < 2; w++) {
        for (int i = 0; i < n_embd; i++) m->x[i] = 0.01f * (i % 100);
        transformer_forward(m, 0, w);
    }

    /* Benchmark */
    fprintf(stderr, "bench: running %d tokens with %d threads...\n", n_tokens, n_threads);
    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);

    for (int t = 0; t < n_tokens; t++) {
        for (int i = 0; i < n_embd; i++) m->x[i] = 0.01f * ((t + i) % 100);
        transformer_forward(m, 0, (t + 2) % max_seq_len);
    }

    clock_gettime(CLOCK_MONOTONIC, &ts1);
    double elapsed = (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
    double tok_per_sec = n_tokens / elapsed;
    double gb_per_sec = (double)total * n_tokens / elapsed / (1024.0*1024.0*1024.0);
    double ms_per_tok = elapsed / n_tokens * 1000.0;

    fprintf(stderr, "\nbench: %d tokens in %.3f s\n", n_tokens, elapsed);
    fprintf(stderr, "bench: %.1f ms/tok, %.1f tok/s\n", ms_per_tok, tok_per_sec);
    fprintf(stderr, "bench: %.1f GB/s effective bandwidth\n", gb_per_sec);

    /* Cleanup */
    for (int l = 0; l < n_layers; l++) {
        free(m->key_cache[l]); free(m->value_cache[l]);
    }
    free(m->key_cache); free(m->value_cache); free(m->layers);
    free(m->x); free(m->xb); free(m->xb2); free(m->q); free(m->k); free(m->v);
    free(m->att); free(m->ffn_buf1); free(m->ffn_buf2); free(m->ffn_buf3);
    free(m->matvec_tmp); free(m->rope_inv_freq);
    if (m->n_threads > 1) {
        for (int t = 1; t < m->n_threads; t++) free(m->thread_tmp[t]);
    }
    free(m->thread_tmp); free(m); free(weights);
    return 0;
}

int main(int argc, char **argv) {
    /* Synthetic benchmark mode: --bench [n_threads] [n_tokens] */
    if (argc >= 2 && strcmp(argv[1], "--bench") == 0) {
        int nt = (argc >= 3) ? atoi(argv[2]) : 1;
        int ntok = (argc >= 4) ? atoi(argv[3]) : 20;
        /* Qwen3.5-2B dimensions */
        return run_bench(2048, 8, 2, 256, 6144, 24, ntok, nt);
    }

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [prompt] [max_gen_tokens] [n_threads]\n", argv[0]);
        fprintf(stderr, "       %s --bench [n_threads] [n_tokens]\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *prompt = (argc >= 3) ? argv[2] : "Hello";
    int max_gen = (argc >= 4) ? atoi(argv[3]) : 32;
    int n_threads = (argc >= 5) ? atoi(argv[4]) : 1;

    srand((unsigned)time(NULL));

    /* Load GGUF */
    int use_mmap = (getenv("NO_MMAP") == NULL) ? 1 : 0;
    fprintf(stderr, "Loading GGUF: %s (mmap=%d)\n", model_path, use_mmap);
    gguf_context *gguf = gguf_open(model_path, use_mmap);
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

    if (n_threads > 1) transformer_set_threads(model, n_threads);
    if (n_threads > 1) transformer_numa_setup(model, gguf);

    /* Tokenize prompt */
    fprintf(stderr, "Prompt: \"%s\"\n", prompt);
    int32_t tokens[2048];
    int n_tokens = bpe_tokenize(vocab, prompt, -1, tokens, 2048);
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
