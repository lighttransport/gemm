/*
 * test_cuda_llm.c - Test harness for CUDA LLM runner
 *
 * Loads a GGUF model, runs both CPU reference and CUDA side-by-side,
 * compares hidden states per token.
 *
 * Usage: ./test_cuda_llm [model.gguf] [-t "prompt text"] [-n max_tokens]
 *
 * Compile with gcc (no nvcc needed).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* GGUF loader */
#define GGUF_LOADER_IMPLEMENTATION
#include "../../common/gguf_loader.h"

/* Dequant (needed by transformer.h) */
#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"

/* CPU reference transformer */
#define TRANSFORMER_IMPLEMENTATION
#include "../../common/transformer.h"

/* BPE tokenizer */
#define BPE_TOKENIZER_IMPLEMENTATION
#include "../../common/bpe_tokenizer.h"

/* CUDA LLM runner */
#include "cuda_llm_runner.h"

/* ---- Comparison helpers ---- */

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static float vec_norm(const float *v, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += v[i] * v[i];
    return sqrtf(s);
}

static float rel_l2_error(const float *a, const float *b, int n) {
    float diff_sq = 0.0f, ref_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = a[i] - b[i];
        diff_sq += d * d;
        ref_sq += b[i] * b[i];
    }
    if (ref_sq < 1e-12f) return sqrtf(diff_sq);
    return sqrtf(diff_sq / ref_sq);
}

static void print_first_n(const char *label, const float *v, int n, int show) {
    if (show > n) show = n;
    fprintf(stderr, "  %s [", label);
    for (int i = 0; i < show; i++) {
        fprintf(stderr, "%s%.6f", i > 0 ? ", " : "", v[i]);
    }
    fprintf(stderr, " ...] norm=%.4f\n", vec_norm(v, n));
}

/* ---- Main ---- */

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *prompt = "Hello, how are you?";
    int max_tokens = 8;
    int max_seq_len = 256;

    /* Parse args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            max_seq_len = atoi(argv[++i]);
        } else if (argv[i][0] != '-') {
            model_path = argv[i];
        } else {
            fprintf(stderr, "Usage: %s [model.gguf] [-t \"prompt\"] [-n max_tokens] [-s max_seq_len]\n", argv[0]);
            return 1;
        }
    }

    if (!model_path) {
        fprintf(stderr, "Usage: %s <model.gguf> [-t \"prompt\"] [-n max_tokens] [-s max_seq_len]\n", argv[0]);
        return 1;
    }

    /* Load GGUF */
    fprintf(stderr, "Loading GGUF: %s\n", model_path);
    gguf_context *gguf = gguf_open(model_path, 1);
    if (!gguf) {
        fprintf(stderr, "Failed to open GGUF file\n");
        return 1;
    }

    /* Load tokenizer */
    bpe_vocab *vocab = bpe_vocab_load(gguf);
    if (!vocab) {
        fprintf(stderr, "Failed to load vocab\n");
        gguf_close(gguf);
        return 1;
    }
    fprintf(stderr, "Vocab: %d tokens\n", vocab->n_tokens);

    /* Tokenize prompt */
    int32_t tokens[512];
    int n_tokens = bpe_tokenize(vocab, prompt, -1, tokens, 512);
    if (n_tokens <= 0) {
        fprintf(stderr, "Tokenization failed\n");
        bpe_vocab_free(vocab);
        gguf_close(gguf);
        return 1;
    }
    fprintf(stderr, "Prompt: \"%s\" -> %d tokens:", prompt, n_tokens);
    for (int i = 0; i < n_tokens; i++) {
        fprintf(stderr, " %d", tokens[i]);
    }
    fprintf(stderr, "\n");

    if (max_tokens > n_tokens) max_tokens = n_tokens;

    /* Load CPU reference model */
    fprintf(stderr, "\n=== Loading CPU reference model ===\n");
    transformer_model *cpu_model = transformer_load(gguf, max_seq_len);
    if (!cpu_model) {
        fprintf(stderr, "Failed to load CPU model\n");
        bpe_vocab_free(vocab);
        gguf_close(gguf);
        return 1;
    }

    /* Initialize CUDA runner */
    fprintf(stderr, "\n=== Initializing CUDA runner ===\n");
    cuda_llm_runner *gpu = cuda_llm_init(0, 1);
    if (!gpu) {
        fprintf(stderr, "Failed to init CUDA runner\n");
        transformer_free(cpu_model);
        bpe_vocab_free(vocab);
        gguf_close(gguf);
        return 1;
    }

    /* Load weights to GPU */
    fprintf(stderr, "\n=== Loading weights to GPU ===\n");
    if (cuda_llm_load_weights(gpu, gguf, max_seq_len) != 0) {
        fprintf(stderr, "Failed to load weights to GPU\n");
        cuda_llm_free(gpu);
        transformer_free(cpu_model);
        bpe_vocab_free(vocab);
        gguf_close(gguf);
        return 1;
    }

    int n_embd = cuda_llm_n_embd(gpu);
    fprintf(stderr, "\n=== Running %d tokens (n_embd=%d) ===\n", max_tokens, n_embd);

    /* Run tokens through both */
    double total_cpu_ms = 0.0, total_gpu_ms = 0.0;
    int pass = 1;

    for (int i = 0; i < max_tokens; i++) {
        int32_t token = tokens[i];

        /* CPU forward */
        double t0 = get_time_ms();
        float *cpu_out = transformer_forward(cpu_model, token, i);
        double cpu_ms = get_time_ms() - t0;
        total_cpu_ms += cpu_ms;

        /* GPU forward */
        t0 = get_time_ms();
        float *gpu_out = cuda_llm_forward(gpu, token, i);
        double gpu_ms = get_time_ms() - t0;
        total_gpu_ms += gpu_ms;

        if (!cpu_out || !gpu_out) {
            fprintf(stderr, "Token %d: forward failed (cpu=%p gpu=%p)\n", i, (void*)cpu_out, (void*)gpu_out);
            pass = 0;
            continue;
        }

        /* Compare */
        float err = rel_l2_error(gpu_out, cpu_out, n_embd);
        const char *status = (err < 1e-2f) ? "OK" : "MISMATCH";
        if (err >= 1e-2f) pass = 0;

        fprintf(stderr, "\nToken %d (id=%d): rel_L2=%.6f [%s]  CPU=%.1fms  GPU=%.1fms  (%.1fx)\n",
                i, token, err, status, cpu_ms, gpu_ms,
                gpu_ms > 0 ? cpu_ms / gpu_ms : 0.0);
        print_first_n("CPU", cpu_out, n_embd, 8);
        print_first_n("GPU", gpu_out, n_embd, 8);
    }

    fprintf(stderr, "\n=== Summary ===\n");
    fprintf(stderr, "Tokens processed: %d\n", max_tokens);
    fprintf(stderr, "Total CPU time: %.1f ms (%.1f ms/token)\n",
            total_cpu_ms, total_cpu_ms / max_tokens);
    fprintf(stderr, "Total GPU time: %.1f ms (%.1f ms/token)\n",
            total_gpu_ms, total_gpu_ms / max_tokens);
    if (total_gpu_ms > 0) {
        fprintf(stderr, "Speedup: %.1fx\n", total_cpu_ms / total_gpu_ms);
    }
    fprintf(stderr, "Result: %s\n", pass ? "PASS" : "FAIL");

    /* Cleanup */
    cuda_llm_free(gpu);
    transformer_free(cpu_model);
    bpe_vocab_free(vocab);
    gguf_close(gguf);

    return pass ? 0 : 1;
}
