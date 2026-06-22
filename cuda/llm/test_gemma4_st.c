/* test_gemma4_st.c — drive the DIRECT gemma-4 safetensors loader.
 *
 * Loads a gemma-4 "qat-mobile-transformers" checkpoint straight from
 * safetensors (no GGUF conversion), tokenizes with the BPE vocab from a
 * companion GGUF (same base model → identical vocab), and greedily generates.
 *
 * Usage:
 *   ./test_gemma4_st <model_dir_or.safetensors> <tokenizer.gguf> \
 *                    [-t "prompt"] [-n n_tokens] [-s max_seq_len]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* single-header implementations (this TU provides them; runner.c only declares) */
#define GGUF_LOADER_IMPLEMENTATION
#include "../../common/gguf_loader.h"
#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"
#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"
#define TRANSFORMER_IMPLEMENTATION
#include "../../common/transformer.h"
#define BPE_TOKENIZER_IMPLEMENTATION
#include "../../common/bpe_tokenizer.h"

#include "cuda_llm_runner.h"

int main(int argc, char **argv) {
    const char *model_path = NULL, *tok_gguf = NULL;
    const char *prompt = "The capital of France is";
    int n_gen = 32, max_seq_len = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n_gen = atoi(argv[++i]);
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) max_seq_len = atoi(argv[++i]);
        else if (argv[i][0] != '-') { if (!model_path) model_path = argv[i]; else tok_gguf = argv[i]; }
    }
    if (!model_path || !tok_gguf) {
        fprintf(stderr, "Usage: %s <model_dir_or.safetensors> <tokenizer.gguf> [-t prompt] [-n n] [-s seq]\n", argv[0]);
        return 1;
    }

    fprintf(stderr, "Tokenizer GGUF: %s\n", tok_gguf);
    gguf_context *gguf = gguf_open(tok_gguf, 1);
    if (!gguf) { fprintf(stderr, "failed to open tokenizer gguf\n"); return 1; }
    bpe_vocab *vocab = bpe_vocab_load(gguf);
    if (!vocab) { fprintf(stderr, "failed to load vocab\n"); return 1; }
    fprintf(stderr, "Vocab: %d tokens\n", vocab->n_tokens);

    int32_t tokens[1024];
    int n_tok = 0;
    tokens[n_tok++] = 2; /* Gemma BOS */
    int nt = bpe_tokenize(vocab, prompt, -1, tokens + n_tok, 1024 - n_tok);
    if (nt <= 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
    n_tok += nt;
    if (max_seq_len <= 0) max_seq_len = n_tok + n_gen + 16;

    fprintf(stderr, "Prompt: \"%s\" -> %d tokens:", prompt, n_tok);
    for (int i = 0; i < n_tok; i++) fprintf(stderr, " %d", tokens[i]);
    fprintf(stderr, "\n\n=== Init + load (DIRECT safetensors) ===\n");

    cuda_llm_runner *gpu = cuda_llm_init(0, 1);
    if (!gpu) { fprintf(stderr, "init failed\n"); return 1; }
    if (cuda_llm_load_weights_gemma4_safetensors(gpu, model_path, max_seq_len) != 0) {
        fprintf(stderr, "load_weights_gemma4_safetensors FAILED\n");
        cuda_llm_free(gpu);
        return 1;
    }
    cuda_llm_set_debug(gpu, getenv("CUDA_LLM_DEBUG") ? atoi(getenv("CUDA_LLM_DEBUG")) : 0);
    if (cuda_llm_reset_state(gpu) != 0) { fprintf(stderr, "reset failed\n"); return 1; }

    int n_vocab = cuda_llm_n_vocab(gpu);
    fprintf(stderr, "\n=== Generate (greedy, n_vocab=%d) ===\n", n_vocab);

    /* Feed the prompt sequentially, then greedily continue. */
    float *logits = NULL;
    int pos = 0;
    for (int i = 0; i < n_tok; i++) {
        logits = cuda_llm_forward_logits(gpu, tokens[i], pos++);
        if (!logits) { fprintf(stderr, "forward failed at prompt tok %d\n", i); return 1; }
    }

    printf("%s", prompt);
    fflush(stdout);
    for (int g = 0; g < n_gen; g++) {
        int best = 0; float bv = logits[0];
        for (int v = 1; v < n_vocab; v++) if (logits[v] > bv) { bv = logits[v]; best = v; }
        if (best == bpe_eos_id(vocab) || best == bpe_eot_id(vocab) || best == 1 || best == 106) break;
        const char *s = bpe_token_to_str(vocab, best);
        if (s) { /* convert SentencePiece ▁ (U+2581, E2 96 81) to space */
            for (const unsigned char *p = (const unsigned char *)s; *p; ) {
                if (p[0] == 0xE2 && p[1] == 0x96 && p[2] == 0x81) { putchar(' '); p += 3; }
                else putchar(*p++);
            }
        }
        fflush(stdout);
        logits = cuda_llm_forward_logits(gpu, best, pos++);
        if (!logits) { fprintf(stderr, "\nforward failed at gen %d\n", g); break; }
    }
    printf("\n");

    cuda_llm_free(gpu);
    bpe_vocab_free(vocab);
    gguf_close(gguf);
    return 0;
}
