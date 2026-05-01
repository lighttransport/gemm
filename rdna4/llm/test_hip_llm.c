/*
 * test_hip_llm.c - Test harness for HIP LLM runner
 *
 * Loads a GGUF model, runs both CPU reference and HIP side-by-side,
 * compares hidden states per token.
 *
 * Usage: ./test_hip_llm [model.gguf] [-t "prompt text"] [-n max_tokens]
 *
 * Compile with gcc (no hipcc needed).
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

/* HIP LLM runner */
#include "hip_llm_runner.h"

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

static int argmax_logits(const float *logits, int n) {
    int best = 0;
    float best_v = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > best_v) { best_v = logits[i]; best = i; }
    }
    return best;
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *prompt = "Hello, how are you?";
    int max_tokens = 8;
    int max_seq_len = 256;
    int bench_mode = 0;       /* --bench: split prefill/decode tps; skip CPU compare */
    int gpu_only_bench = 0;   /* --gpu-only-bench: also skip CPU model load */
    int decode_n = 0;         /* --decode N: greedy-sample N tokens after prefill */
    int prefill_pad = 0;      /* --prefill-len M: pad prompt up to M tokens with last token (for bench) */
    int compare_paths = 0;    /* --compare-paths: report rel-L2 between batched and per-token logits */

    /* Parse args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            max_seq_len = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--bench") == 0) {
            bench_mode = 1;
        } else if (strcmp(argv[i], "--gpu-only-bench") == 0) {
            bench_mode = 1;
            gpu_only_bench = 1;
        } else if (strcmp(argv[i], "--decode") == 0 && i + 1 < argc) {
            decode_n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--prefill-len") == 0 && i + 1 < argc) {
            prefill_pad = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--compare-paths") == 0) {
            compare_paths = 1;
        } else if (argv[i][0] != '-') {
            model_path = argv[i];
        } else {
            fprintf(stderr, "Usage: %s [model.gguf] [-t \"prompt\"] [-n max_tokens] [-s max_seq_len]\n", argv[0]);
            fprintf(stderr, "       [--bench] [--gpu-only-bench] [--decode N] [--prefill-len M]\n");
            return 1;
        }
    }

    if (!model_path) {
        fprintf(stderr, "Usage: %s <model.gguf> [-t \"prompt\"] [-n max_tokens] [-s max_seq_len]\n", argv[0]);
        fprintf(stderr, "       [--bench] [--gpu-only-bench] [--decode N] [--prefill-len M]\n");
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
    int32_t tokens[4096];
    int n_tokens = bpe_tokenize(vocab, prompt, -1, tokens, 4096);
    if (n_tokens <= 0) {
        fprintf(stderr, "Tokenization failed\n");
        bpe_vocab_free(vocab);
        gguf_close(gguf);
        return 1;
    }
    fprintf(stderr, "Prompt: \"%s\" -> %d tokens:", prompt, n_tokens);
    for (int i = 0; i < n_tokens && i < 32; i++) {
        fprintf(stderr, " %d", tokens[i]);
    }
    if (n_tokens > 32) fprintf(stderr, " ...");
    fprintf(stderr, "\n");

    /* --prefill-len M: pad prompt by repeating last token until length == M */
    if (prefill_pad > n_tokens && prefill_pad <= 4096) {
        int32_t pad = tokens[n_tokens - 1];
        for (int i = n_tokens; i < prefill_pad; i++) tokens[i] = pad;
        n_tokens = prefill_pad;
        fprintf(stderr, "Prompt padded to %d tokens for bench\n", n_tokens);
    }

    /* In bench mode, default to using the full (possibly padded) prompt for prefill. */
    if (bench_mode && max_tokens == 8) max_tokens = n_tokens;
    if (max_tokens > n_tokens) max_tokens = n_tokens;

    /* Load CPU reference model (may fail for MoE -- run GPU-only in that case) */
    int gpu_only = 0;
    transformer_model *cpu_model = NULL;
    if (gpu_only_bench) {
        gpu_only = 1;
        fprintf(stderr, "\n=== Skipping CPU reference (--gpu-only-bench) ===\n");
    } else {
        fprintf(stderr, "\n=== Loading CPU reference model ===\n");
        cpu_model = transformer_load(gguf, max_seq_len);
        if (!cpu_model) {
            fprintf(stderr, "CPU model load failed (MoE?), running GPU-only mode\n");
            gpu_only = 1;
        }
    }

    /* Initialize HIP runner */
    fprintf(stderr, "\n=== Initializing HIP runner ===\n");
    hip_llm_runner *gpu = hip_llm_init(0, 1);
    if (!gpu) {
        fprintf(stderr, "Failed to init HIP runner\n");
        if (cpu_model) transformer_free(cpu_model);
        bpe_vocab_free(vocab);
        gguf_close(gguf);
        return 1;
    }

    /* Load weights to GPU */
    fprintf(stderr, "\n=== Loading weights to GPU ===\n");
    if (hip_llm_load_weights(gpu, gguf, max_seq_len) != 0) {
        fprintf(stderr, "Failed to load weights to GPU\n");
        hip_llm_free(gpu);
        if (cpu_model) transformer_free(cpu_model);
        bpe_vocab_free(vocab);
        gguf_close(gguf);
        return 1;
    }

    int n_embd = hip_llm_n_embd(gpu);
    int n_vocab = hip_llm_n_vocab(gpu);
    int n_max_seq = hip_llm_max_seq_len(gpu);
    int pass = 1;

    if (bench_mode) {
        /* ---- Bench mode: split prefill and decode tokens/sec ---- */
        int n_prefill = max_tokens;
        if (n_prefill < 1) n_prefill = 1;
        if (decode_n < 0) decode_n = 0;
        if (n_prefill + decode_n > n_max_seq) {
            decode_n = n_max_seq - n_prefill;
            if (decode_n < 0) decode_n = 0;
            fprintf(stderr, "Clamped decode to %d (max_seq_len=%d)\n", decode_n, n_max_seq);
        }

        fprintf(stderr, "\n=== Bench: prefill=%d tokens, decode=%d tokens, n_embd=%d, n_vocab=%d ===\n",
                n_prefill, decode_n, n_embd, n_vocab);

        if (compare_paths && hip_llm_batched_path_available(gpu)) {
            /* Run prefill via per-token path */
            hip_llm_set_batched_path(gpu, 0);
            float *log_p = hip_llm_forward_batch_logits(gpu, tokens, n_prefill, 0);
            if (!log_p) { fprintf(stderr, "compare: per-token path failed\n"); pass = 0; goto bench_done; }
            float *buf_p = (float *)malloc((size_t)n_vocab * sizeof(float));
            memcpy(buf_p, log_p, (size_t)n_vocab * sizeof(float));

            /* Run prefill via batched path */
            hip_llm_set_batched_path(gpu, 1);
            float *log_b = hip_llm_forward_batch_logits(gpu, tokens, n_prefill, 0);
            if (!log_b) { fprintf(stderr, "compare: batched path failed\n"); free(buf_p); pass = 0; goto bench_done; }

            float diff_sq = 0.0f, ref_sq = 0.0f, max_abs = 0.0f;
            for (int i = 0; i < n_vocab; i++) {
                float d = log_b[i] - buf_p[i];
                diff_sq += d * d;
                ref_sq += buf_p[i] * buf_p[i];
                if (fabsf(d) > max_abs) max_abs = fabsf(d);
            }
            float rl2 = (ref_sq > 1e-12f) ? sqrtf(diff_sq / ref_sq) : sqrtf(diff_sq);
            int top_p = argmax_logits(buf_p, n_vocab);
            int top_b = argmax_logits(log_b, n_vocab);
            fprintf(stderr,
                "[--compare-paths] rel_l2=%.4e  max_abs=%.4e  argmax: per-token=%d batched=%d %s\n",
                rl2, max_abs, top_p, top_b, (top_p == top_b) ? "(match)" : "(DIFFER)");
            free(buf_p);
        }

        /* Prefill: a single forward_batch_logits call. Phase 1 implementation is a
         * per-token loop; Phase 2 will swap in a true batched WMMA path. */
        double t_pf0 = get_time_ms();
        float *last_logits = hip_llm_forward_batch_logits(gpu, tokens, n_prefill, 0);
        if (!last_logits) { fprintf(stderr, "GPU forward_batch_logits failed\n"); pass = 0; goto bench_done; }
        int next_tok = argmax_logits(last_logits, n_vocab);
        double t_pf1 = get_time_ms();
        double prefill_ms = t_pf1 - t_pf0;
        double prefill_tps = (prefill_ms > 0.0) ? (1000.0 * n_prefill / prefill_ms) : 0.0;

        /* Decode: greedy-sample decode_n tokens. */
        double decode_ms = 0.0, decode_tps = 0.0;
        int first_decode_tok = next_tok;
        if (decode_n > 0) {
            double t_dec0 = get_time_ms();
            for (int k = 0; k < decode_n; k++) {
                int pos = n_prefill + k;
                float *lg = hip_llm_forward_logits(gpu, next_tok, pos);
                if (!lg) { fprintf(stderr, "GPU forward_logits failed at decode k=%d\n", k); pass = 0; break; }
                next_tok = argmax_logits(lg, n_vocab);
            }
            double t_dec1 = get_time_ms();
            decode_ms = t_dec1 - t_dec0;
            decode_tps = (decode_ms > 0.0) ? (1000.0 * decode_n / decode_ms) : 0.0;
        }

        fprintf(stderr, "\n=== Bench results ===\n");
        fprintf(stderr, "Prefill: %d tokens in %.2f ms  -> %.2f tok/s  (%.3f ms/tok)\n",
                n_prefill, prefill_ms, prefill_tps,
                n_prefill > 0 ? prefill_ms / n_prefill : 0.0);
        if (decode_n > 0) {
            fprintf(stderr, "Decode:  %d tokens in %.2f ms  -> %.2f tok/s  (%.3f ms/tok)\n",
                    decode_n, decode_ms, decode_tps,
                    decode_ms / decode_n);
            fprintf(stderr, "First decoded token id=%d, last id=%d\n", first_decode_tok, next_tok);
        }
        fprintf(stderr, "Result: %s\n", pass ? "PASS" : "FAIL");
bench_done: ;
    } else {
        /* ---- Correctness mode: per-token CPU vs GPU compare (legacy) ---- */
        fprintf(stderr, "\n=== Running %d tokens (n_embd=%d)%s ===\n",
                max_tokens, n_embd, gpu_only ? " [GPU-only]" : "");

        double total_cpu_ms = 0.0, total_gpu_ms = 0.0;

        for (int i = 0; i < max_tokens; i++) {
            int32_t token = tokens[i];

            /* CPU forward (skip if GPU-only) */
            float *cpu_out = NULL;
            double cpu_ms = 0.0;
            if (!gpu_only) {
                double t0 = get_time_ms();
                cpu_out = transformer_forward(cpu_model, token, i);
                cpu_ms = get_time_ms() - t0;
                total_cpu_ms += cpu_ms;
            }

            /* GPU forward */
            double t0 = get_time_ms();
            float *gpu_out = hip_llm_forward(gpu, token, i);
            double gpu_ms = get_time_ms() - t0;
            total_gpu_ms += gpu_ms;

            if (!gpu_out) {
                fprintf(stderr, "Token %d: GPU forward failed\n", i);
                pass = 0;
                continue;
            }

            if (gpu_only) {
                fprintf(stderr, "\nToken %d (id=%d): GPU=%.1fms\n", i, token, gpu_ms);
                print_first_n("GPU", gpu_out, n_embd, 8);
            } else if (!cpu_out) {
                fprintf(stderr, "Token %d: CPU forward failed\n", i);
                pass = 0;
            } else {
                float err = rel_l2_error(gpu_out, cpu_out, n_embd);
                const char *status = (err < 1e-2f) ? "OK" : "MISMATCH";
                if (err >= 1e-2f) pass = 0;

                fprintf(stderr, "\nToken %d (id=%d): rel_L2=%.6f [%s]  CPU=%.1fms  GPU=%.1fms  (%.1fx)\n",
                        i, token, err, status, cpu_ms, gpu_ms,
                        gpu_ms > 0 ? cpu_ms / gpu_ms : 0.0);
                print_first_n("CPU", cpu_out, n_embd, 8);
                print_first_n("GPU", gpu_out, n_embd, 8);
            }
        }

        fprintf(stderr, "\n=== Summary ===\n");
        fprintf(stderr, "Tokens processed: %d\n", max_tokens);
        if (!gpu_only) {
            fprintf(stderr, "Total CPU time: %.1f ms (%.1f ms/token)\n",
                    total_cpu_ms, total_cpu_ms / max_tokens);
        }
        fprintf(stderr, "Total GPU time: %.1f ms (%.1f ms/token)\n",
                total_gpu_ms, total_gpu_ms / max_tokens);
        if (!gpu_only && total_gpu_ms > 0) {
            fprintf(stderr, "Speedup: %.1fx\n", total_cpu_ms / total_gpu_ms);
        }
        fprintf(stderr, "Result: %s\n", pass ? "PASS" : "FAIL");
    }

    /* Cleanup */
    hip_llm_free(gpu);
    if (cpu_model) transformer_free(cpu_model);
    bpe_vocab_free(vocab);
    gguf_close(gguf);

    return pass ? 0 : 1;
}
