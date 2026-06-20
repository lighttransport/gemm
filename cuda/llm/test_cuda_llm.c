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
#include <unistd.h>

/* GGUF loader */
#define GGUF_LOADER_IMPLEMENTATION
#include "../../common/gguf_loader.h"
#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"

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

static uint32_t bench_lcg_next(uint32_t *state) {
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

#define BATCHED_PREFILL_MIN_TOKENS 33

/* ---- Main ---- */

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *prompt = "Hello, how are you?";
    int max_tokens = 8;
    int max_seq_len = 256;
    int large_bench = 0;
    int large_bench_random = 0;
    uint32_t large_bench_seed = 1;

    /* Parse args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            max_seq_len = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--perf") == 0) {
            max_tokens = 10;  /* short prompt, focus on prefill + decode perf */
        } else if (strcmp(argv[i], "--bench") == 0) {
            max_tokens = 100;  /* 100-token prefill benchmark */
        } else if (strcmp(argv[i], "--large-bench") == 0 && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
            large_bench = max_tokens;
        } else if (strcmp(argv[i], "--large-bench-random") == 0) {
            large_bench_random = 1;
        } else if (strcmp(argv[i], "--large-bench-seed") == 0 && i + 1 < argc) {
            large_bench_seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (argv[i][0] != '-') {
            model_path = argv[i];
        } else {
            fprintf(stderr, "Usage: %s [model.gguf] [-t \"prompt\"] [-n max_tokens] [-s max_seq_len] [--large-bench N] [--large-bench-random] [--large-bench-seed N]\n", argv[0]);
            return 1;
        }
    }

    if (large_bench > 0 && max_seq_len < max_tokens) {
        max_seq_len = max_tokens + 64;
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

    /* CUDA_LLM_PREFILL_REPEAT=N forces an N-token prefill (repeating the prompt)
     * so the batched-vs-sequential verify covers chunked prefill (N > chunk). */
    int prefill_cap = 512;
    { const char *e = getenv("CUDA_LLM_PREFILL_REPEAT");
      if (e) { int v = atoi(e); if (v > prefill_cap) prefill_cap = v; } }
    int32_t *prefill_tokens = (int32_t *)malloc((size_t)prefill_cap * sizeof(int32_t));
    int prefill_token_count = max_tokens;
    int repeated_prefill_tokens = 0;
    for (int i = 0; i < prefill_token_count; i++) {
        prefill_tokens[i] = tokens[i];
    }
    { const char *e = getenv("CUDA_LLM_PREFILL_REPEAT");
      int want = e ? atoi(e) : 0;
      if (want > prefill_token_count) {
          if (want > prefill_cap) want = prefill_cap;
          prefill_token_count = want;
          for (int i = 0; i < prefill_token_count; i++) prefill_tokens[i] = tokens[i % max_tokens];
          repeated_prefill_tokens = 1;
      }
    }
    if (prefill_token_count > 0 && prefill_token_count < BATCHED_PREFILL_MIN_TOKENS) {
        prefill_token_count = BATCHED_PREFILL_MIN_TOKENS;
        if (prefill_token_count > prefill_cap) {
            prefill_token_count = prefill_cap;
        }
        for (int i = 0; i < prefill_token_count; i++) {
            prefill_tokens[i] = tokens[i % max_tokens];
        }
        repeated_prefill_tokens = 1;
    }
    if (max_seq_len < prefill_token_count) {
        max_seq_len = prefill_token_count;
    }

    /* Load CPU reference model (may fail for MoE — run GPU-only in that case) */
    fprintf(stderr, "\n=== Loading CPU reference model ===\n");
    transformer_model *cpu_model = transformer_load(gguf, max_seq_len);
    int gpu_only = 0;
    if (!cpu_model) {
        fprintf(stderr, "CPU model load failed (MoE?), running GPU-only mode\n");
        gpu_only = 1;
    }

    /* Initialize CUDA runner */
    fprintf(stderr, "\n=== Initializing CUDA runner ===\n");
    cuda_llm_runner *gpu = cuda_llm_init(0, 1);
    if (!gpu) {
        fprintf(stderr, "Failed to init CUDA runner\n");
        if (cpu_model) transformer_free(cpu_model);
        bpe_vocab_free(vocab);
        gguf_close(gguf);
        return 1;
    }

    /* Load weights to GPU */
    fprintf(stderr, "\n=== Loading weights to GPU ===\n");
    if (cuda_llm_load_weights(gpu, gguf, max_seq_len) != 0) {
        fprintf(stderr, "Failed to load weights to GPU\n");
        cuda_llm_free(gpu);
        if (cpu_model) transformer_free(cpu_model);
        bpe_vocab_free(vocab);
        gguf_close(gguf);
        return 1;
    }

    {
        const char *debug_env = getenv("CUDA_LLM_DEBUG");
        const char *max_layers_env = getenv("CUDA_LLM_MAX_LAYERS");
        cuda_llm_set_debug(gpu, debug_env ? atoi(debug_env) : 0);
        if (max_layers_env) cuda_llm_set_max_layers(gpu, atoi(max_layers_env));
    }
    if (cuda_llm_reset_state(gpu) != 0) {
        fprintf(stderr, "cuda_llm_reset_state failed before initial run\n");
        cuda_llm_free(gpu);
        if (cpu_model) transformer_free(cpu_model);
        bpe_vocab_free(vocab);
        gguf_close(gguf);
        return 1;
    }
    int n_embd = cuda_llm_n_embd(gpu);
    int n_vocab_size = cuda_llm_n_vocab(gpu);
    fprintf(stderr, "\n=== Running %d tokens (n_embd=%d)%s ===\n",
            max_tokens, n_embd, gpu_only ? " [GPU-only]" : "");

    /* Run tokens through both */
    double total_cpu_ms = 0.0, total_gpu_ms = 0.0;
    int pass = 1;
    int have_seq_logits = 0;
    int verify_decode = getenv("CUDA_LLM_VERIFY_DECODE") != NULL; /* compare 1st decode step (validates KV cache) */
    int have_seq_decode = 0;
    float *last_gpu_hidden = (float *)malloc((size_t)n_embd * sizeof(float));
    float *last_seq_logits = n_vocab_size > 0 ? (float *)malloc((size_t)n_vocab_size * sizeof(float)) : NULL;
    float *seq_decode_logits = (verify_decode && n_vocab_size > 0) ? (float *)malloc((size_t)n_vocab_size * sizeof(float)) : NULL;
    /* CPU F32 oracle (opt-in CUDA_LLM_CPU_ORACLE=1, slow): a true F32 reference to see
     * which GPU path (batched F16 vs sequential F32) is actually closer to truth. */
    int cpu_oracle = (!gpu_only) && cpu_model && getenv("CUDA_LLM_CPU_ORACLE");
    float *cpu_pf_hidden = NULL, *cpu_pf_logits = NULL;
    int have_cpu_pf_hidden = 0, have_cpu_pf_logits = 0;
    if (cpu_oracle) {
        cpu_pf_hidden = (float *)malloc((size_t)n_embd * sizeof(float));
        if (n_vocab_size > 0) cpu_pf_logits = (float *)malloc((size_t)n_vocab_size * sizeof(float));
    }
    if (!last_gpu_hidden) {
        fprintf(stderr, "Failed to allocate last_gpu_hidden\n");
        cuda_llm_free(gpu);
        if (cpu_model) transformer_free(cpu_model);
        bpe_vocab_free(vocab);
        gguf_close(gguf);
        return 1;
    }
    if (n_vocab_size > 0 && !last_seq_logits) {
        fprintf(stderr, "Failed to allocate last_seq_logits\n");
        free(last_gpu_hidden);
        cuda_llm_free(gpu);
        if (cpu_model) transformer_free(cpu_model);
        bpe_vocab_free(vocab);
        gguf_close(gguf);
        return 1;
    }

    if (!large_bench) {
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
        float *gpu_out = cuda_llm_forward(gpu, token, i);
        double gpu_ms = get_time_ms() - t0;
        total_gpu_ms += gpu_ms;

        if (!gpu_out) {
            fprintf(stderr, "Token %d: GPU forward failed\n", i);
            pass = 0;
            continue;
        }
        memcpy(last_gpu_hidden, gpu_out, (size_t)n_embd * sizeof(float));

        if (gpu_only) {
            /* GPU-only: just print hidden state */
            fprintf(stderr, "\nToken %d (id=%d): GPU=%.1fms\n", i, token, gpu_ms);
            print_first_n("GPU", gpu_out, n_embd, 8);
        } else if (!cpu_out) {
            fprintf(stderr, "Token %d: CPU forward failed\n", i);
            pass = 0;
        } else {
            /* Compare (dp4a INT8 quantization widens expected error) */
            float err = rel_l2_error(gpu_out, cpu_out, n_embd);
            float tol = cuda_llm_uses_dp4a(gpu) ? 0.5f : 1e-2f;
            const char *status = (err < tol) ? "OK" : "MISMATCH";
            if (err >= tol) pass = 0;

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
    } else {
        fprintf(stderr, "\n=== Skipping decode path for large-batch prefill benchmark ===\n");
    }

    /* Take the sequential reference as a true F32 oracle: turn OFF the INT8 dp4a
     * matvec path and dense MMQ. The batched prefill is F16, so vs an F32 oracle
     * it is ~3e-3 (pure F16 precision) and a tight 1e-2 tolerance catches real
     * batched-dataflow bugs (rope/attention/norm/kv/dequant-GEMM). Comparing
     * against the dp4a/MMQ int8 path instead inflated rel_L2 to ~0.2-0.7 (int8
     * activation-quant divergence — argmax-preserving, NOT a batched bug; it was
     * largest for Q8_0-weight models like the 12B Q6_K, ~0.70). The MMQ int8
     * kernels are validated separately (cuda/llm/mmq/*_test + the top-5 check). */
    int saved_dp4a = 0;
    if (!large_bench) {
        /* DBG: CUDA_LLM_REF_DP4A keeps dp4a ON for the reference so rel_L2 reports
         * dp4a error (vs F16 batched); combine with NO_*_DP4A to attribute. */
        saved_dp4a = cuda_llm_set_dp4a(gpu, getenv("CUDA_LLM_REF_DP4A") ? 1 : 0);
        setenv("CUDA_LLM_NO_MMQ_DENSE", "1", 1);
    }

    /* Get logits from sequential forward_logits (skip for large bench) */
    if (!large_bench) {
    if (cuda_llm_reset_state(gpu) != 0) {
        fprintf(stderr, "cuda_llm_reset_state failed before sequential logits\n");
        pass = 0;
        goto cleanup;
    }
    {
        /* Run all tokens except last to build up state */
        for (int t = 0; t + 1 < prefill_token_count; t++) {
            cuda_llm_forward(gpu, prefill_tokens[t], t);
        }
        /* Get logits for the last token */
        float *seq_logits = cuda_llm_forward_logits(gpu, prefill_tokens[prefill_token_count - 1], prefill_token_count - 1);
        if (seq_logits) {
            if (n_vocab_size > 0) {
                memcpy(last_seq_logits, seq_logits, (size_t)n_vocab_size * sizeof(float));
                have_seq_logits = 1;
                int top5_ids[5] = {0};
                float top5_vals[5] = {-1e30f, -1e30f, -1e30f, -1e30f, -1e30f};
                for (int v = 0; v < n_vocab_size; v++) {
                    float val = seq_logits[v];
                    for (int k = 0; k < 5; k++) {
                        if (val > top5_vals[k]) {
                            for (int j = 4; j > k; j--) { top5_ids[j] = top5_ids[j-1]; top5_vals[j] = top5_vals[j-1]; }
                            top5_ids[k] = v; top5_vals[k] = val;
                            break;
                        }
                    }
                }
                fprintf(stderr, "Sequential top-5%s:", repeated_prefill_tokens ? " (batched prefill check)" : "");
                for (int k = 0; k < 5; k++) {
                    const char *s = bpe_token_to_str(vocab, top5_ids[k]);
                    fprintf(stderr, " [%d]=%.4f('%s')", top5_ids[k], top5_vals[k], s ? s : "?");
                }
                fprintf(stderr, "\n");
            }
        }
        /* One decode step after the per-token prefill -> reference for the KV cache.
         * Cache is at position prefill_token_count-1; forward a fixed token at the
         * next position and capture its logits. */
        if (verify_decode && seq_decode_logits && n_vocab_size > 0) {
            float *dl = cuda_llm_forward_logits(gpu, prefill_tokens[0], prefill_token_count);
            if (dl) { memcpy(seq_decode_logits, dl, (size_t)n_vocab_size * sizeof(float)); have_seq_decode = 1; }
        }
    }
    } /* end !large_bench */

    if (!large_bench) {
    /* CPU F32 oracle: replay prefill_tokens on the CPU transformer (true F32, full-
     * precision dequant + F32 KV) to get a reference both GPU paths can be measured
     * against. Opt-in (slow: prefill_token_count CPU forwards). The CPU KV cache is
     * position-keyed, so the in-order replay overwrites all needed slots (no reset). */
    if (cpu_oracle) {
        const char *ot = getenv("CUDA_LLM_CPU_ORACLE_THREADS");
        if (ot) { int nt = atoi(ot); if (nt > 0) transformer_set_threads(cpu_model, nt); }
        double ct0 = get_time_ms();
        float *ch = NULL;
        for (int t = 0; t < prefill_token_count; t++)
            ch = transformer_forward(cpu_model, prefill_tokens[t], t);
        if (ch && cpu_pf_hidden) {
            memcpy(cpu_pf_hidden, ch, (size_t)n_embd * sizeof(float));
            have_cpu_pf_hidden = 1;
            fprintf(stderr, "CPU oracle: hidden norm=%.4f [0]=%.6f [1]=%.6f\n",
                    vec_norm(cpu_pf_hidden, n_embd), cpu_pf_hidden[0], cpu_pf_hidden[1]);
        }
        if (cpu_pf_logits) {
            float *cl = transformer_forward_logits(cpu_model, prefill_tokens[prefill_token_count - 1],
                                                   prefill_token_count - 1);
            if (cl) {
                memcpy(cpu_pf_logits, cl, (size_t)n_vocab_size * sizeof(float));
                have_cpu_pf_logits = 1;
            }
        }
        fprintf(stderr, "CPU oracle: replayed %d tokens in %.1f ms\n",
                prefill_token_count, get_time_ms() - ct0);
    }
    if (cuda_llm_reset_state(gpu) != 0) {
        fprintf(stderr, "cuda_llm_reset_state failed before sequential hidden replay\n");
        pass = 0;
        goto cleanup;
    }
    {
        if (repeated_prefill_tokens) {
            fprintf(stderr, "Prefill check uses %d tokens by repeating the prompt to force batched prefill\n",
                    prefill_token_count);
        }
        float *seq_hidden = NULL;
        for (int t = 0; t < prefill_token_count; t++) {
            seq_hidden = cuda_llm_forward(gpu, prefill_tokens[t], t);
            if (!seq_hidden) break;
        }
        if (seq_hidden) {
            memcpy(last_gpu_hidden, seq_hidden, (size_t)n_embd * sizeof(float));
        } else {
            fprintf(stderr, "Sequential hidden replay failed\n");
            pass = 0;
            goto cleanup;
        }
    }

    if (cuda_llm_reset_state(gpu) != 0) {
        fprintf(stderr, "cuda_llm_reset_state failed before prefill hidden\n");
        pass = 0;
        goto cleanup;
    }
    double t0 = get_time_ms();
    float *prefill_hidden = cuda_llm_prefill(gpu, prefill_tokens, NULL, 0, prefill_token_count, 0);
    double prefill_ms = get_time_ms() - t0;
    if (prefill_hidden) {
        float err = rel_l2_error(prefill_hidden, last_gpu_hidden, n_embd);
        float ptol = 1e-2f;  /* reference is F32 oracle (dp4a+MMQ off); batched is F16 */
        fprintf(stderr, "Prefill hidden: %.1f ms (%.1f tok/s) rel_L2_vs_seq=%.6f\n",
                prefill_ms, prefill_ms > 0 ? (1000.0 * prefill_token_count / prefill_ms) : 0.0, err);
        if (have_cpu_pf_hidden) {
            float e_bc = rel_l2_error(prefill_hidden, cpu_pf_hidden, n_embd);
            float e_sc = rel_l2_error(last_gpu_hidden, cpu_pf_hidden, n_embd);
            fprintf(stderr, "Prefill hidden vs CPU-F32 oracle: batched_vs_cpu=%.6f seq_vs_cpu=%.6f (batched_vs_seq=%.6f)\n",
                    e_bc, e_sc, err);
        }
        if (!isfinite(err) || err >= ptol) pass = 0;
    } else {
        fprintf(stderr, "Prefill hidden failed\n");
        pass = 0;
    }

    if (cuda_llm_reset_state(gpu) != 0) {
        fprintf(stderr, "cuda_llm_reset_state failed before prefill logits\n");
        pass = 0;
        goto cleanup;
    }
    t0 = get_time_ms();
    float *prefill_logits = cuda_llm_prefill_logits(gpu, prefill_tokens, NULL, 0, prefill_token_count, 0);
    double prefill_logits_ms = get_time_ms() - t0;
    if (prefill_logits) {
        if (have_seq_logits && last_seq_logits && n_vocab_size > 0) {
            float logit_err = rel_l2_error(prefill_logits, last_seq_logits, n_vocab_size);
            float ptol = 1e-2f;  /* reference is F32 oracle (dp4a+MMQ off); batched is F16 */
            fprintf(stderr, "Prefill logits rel_L2_vs_seq=%.6f\n", logit_err);
            if (have_cpu_pf_logits) {
                float le_bc = rel_l2_error(prefill_logits, cpu_pf_logits, n_vocab_size);
                float le_sc = rel_l2_error(last_seq_logits, cpu_pf_logits, n_vocab_size);
                fprintf(stderr, "Prefill logits vs CPU-F32 oracle: batched_vs_cpu=%.6f seq_vs_cpu=%.6f\n",
                        le_bc, le_sc);
            }
            if (!isfinite(logit_err) || logit_err >= ptol) pass = 0;
        }
        fprintf(stderr, "Prefill logits: %.1f ms (%.1f tok/s)\n",
                prefill_logits_ms,
                prefill_logits_ms > 0 ? (1000.0 * prefill_token_count / prefill_logits_ms) : 0.0);
        if (n_vocab_size > 0) {
            int top5_ids[5] = {0};
            float top5_vals[5] = {-1e30f, -1e30f, -1e30f, -1e30f, -1e30f};
            for (int v = 0; v < n_vocab_size; v++) {
                float val = prefill_logits[v];
                for (int k = 0; k < 5; k++) {
                    if (val > top5_vals[k]) {
                        for (int j = 4; j > k; j--) { top5_ids[j] = top5_ids[j-1]; top5_vals[j] = top5_vals[j-1]; }
                        top5_ids[k] = v; top5_vals[k] = val;
                        break;
                    }
                }
            }
            fprintf(stderr, "Prefill top-5:");
            for (int k = 0; k < 5; k++) {
                const char *s = bpe_token_to_str(vocab, top5_ids[k]);
                fprintf(stderr, " [%d]=%.4f('%s')", top5_ids[k], top5_vals[k], s ? s : "?");
            }
            fprintf(stderr, "\n");
        }
        /* Decode-step check: after the BATCHED prefill, forward the same fixed
         * token at the next position and compare logits to the per-token reference.
         * This validates the KV cache the batched prefill leaves behind (e.g. the
         * windowed-SWA circular-cache population). */
        if (verify_decode && have_seq_decode && seq_decode_logits && n_vocab_size > 0) {
            float *dl = cuda_llm_forward_logits(gpu, prefill_tokens[0], prefill_token_count);
            if (dl) {
                float derr = rel_l2_error(dl, seq_decode_logits, n_vocab_size);
                fprintf(stderr, "Decode-step logits rel_L2_vs_seq=%.6f\n", derr);
                if (!isfinite(derr) || derr >= 0.05f) pass = 0;
            }
        }
    } else {
        fprintf(stderr, "Prefill logits failed\n");
        pass = 0;
    }
    /* Restore the production paths for the decode/perf benchmark below. */
    unsetenv("CUDA_LLM_NO_MMQ_DENSE");
    cuda_llm_set_dp4a(gpu, saved_dp4a);
    } /* end !large_bench */

    /* Large batch prefill benchmark (synthetic tokens, no correctness) */
    if (large_bench > 0) {
        int bench_tokens = large_bench;
        fprintf(stderr, "\n=== Large-batch prefill: %d tokens ===\n", bench_tokens);
        if (cuda_llm_reset_state(gpu) != 0) {
            fprintf(stderr, "cuda_llm_reset_state failed before large bench\n");
            pass = 0; goto cleanup;
        }
        /* Create synthetic token IDs. Random mode is closer to llama-bench prompt processing. */
        int32_t *big_tokens = (int32_t *)calloc((size_t)bench_tokens, sizeof(int32_t));
        if (!big_tokens) { fprintf(stderr,"malloc failed\n"); pass=0; goto cleanup; }
        if (large_bench_random) {
            uint32_t rng = large_bench_seed ? large_bench_seed : 1u;
            int vocab_n = vocab->n_tokens > 0 ? vocab->n_tokens : 1;
            for (int i = 0; i < bench_tokens; i++) {
                big_tokens[i] = (int32_t)(bench_lcg_next(&rng) % (uint32_t)vocab_n);
            }
            fprintf(stderr, "Large bench tokens: deterministic random seed=%u vocab=%d\n",
                    large_bench_seed, vocab_n);
        }
        /* Warm-up run */
        if (!cuda_llm_prefill(gpu, big_tokens, NULL, 0, bench_tokens, 0)) {
            fprintf(stderr, "Large bench warm-up prefill failed\n");
            free(big_tokens); pass = 0; goto cleanup;
        }
        /* Timed run (reset state first for fair measurement) */
        if (cuda_llm_reset_state(gpu) != 0) {
            fprintf(stderr, "reset failed before timed run\n");
            free(big_tokens); pass = 0; goto cleanup;
        }
        double t0 = get_time_ms();
        float *result = cuda_llm_prefill(gpu, big_tokens, NULL, 0, bench_tokens, 0);
        double ms = get_time_ms() - t0;
        if (result) {
            fprintf(stderr, "Large bench: %.1f ms (%.1f tok/s)\n",
                    ms, ms > 0 ? (1000.0 * bench_tokens / ms) : 0.0);
            /* Checksum of the last-token hidden state — lets two batched runs
             * (e.g. chunked vs single-batch) be compared without the sequential
             * oracle. Deterministic random tokens (--large-bench-random) make it
             * reproducible across runs. */
            double cs = 0.0; double cs2 = 0.0;
            for (int i = 0; i < n_embd; i++) { cs += result[i]; cs2 += (double)result[i] * result[i]; }
            fprintf(stderr, "Large bench hidden: sum=%.6f l2=%.6f h[0..4]=%.5f,%.5f,%.5f,%.5f\n",
                    cs, cs2, result[0], result[1], result[2], result[3]);
        } else {
            fprintf(stderr, "Large bench prefill failed\n");
            pass = 0;
        }
        free(big_tokens);
        goto cleanup; /* skip decode bench after large bench */
    }

    /* Decode benchmark: N steps via prefill-then-decode */
    {
        int bench_decode_steps = 50;  /* generate this many tokens */
        if (cuda_llm_reset_state(gpu) != 0) {
            fprintf(stderr, "cuda_llm_reset_state failed before decode bench\n");
            pass = 0;
            goto cleanup;
        }
        /* Prefill the prompt first */
        if (!cuda_llm_prefill(gpu, tokens, NULL, 0, max_tokens, 0)) {
            fprintf(stderr, "Decode bench: prefill failed\n");
            pass = 0;
            goto cleanup;
        }
        /* Warm-up: run 5 decode steps */
        int32_t token = tokens[max_tokens - 1];
        for (int i = 0; i < 5; i++) {
            float *logits = cuda_llm_forward_logits(gpu, token, max_tokens + i);
            if (!logits) { fprintf(stderr, "Decode bench warm-up failed at step %d\n", i); break; }
            /* Pick top token */
            int n_v = cuda_llm_n_vocab(gpu);
            token = 0; float best = -1e30f;
            for (int j = 0; j < n_v; j++) if (logits[j] > best) { best = logits[j]; token = j; }
        }
        /* Timed decode */
        double t_dec_start = get_time_ms();
        for (int i = 0; i < bench_decode_steps; i++) {
            float *logits = cuda_llm_forward_logits(gpu, token, max_tokens + 5 + i);
            if (!logits) { fprintf(stderr, "Decode bench failed at step %d\n", i); pass = 0; break; }
            int n_v = cuda_llm_n_vocab(gpu);
            token = 0; float best = -1e30f;
            for (int j = 0; j < n_v; j++) if (logits[j] > best) { best = logits[j]; token = j; }
        }
        double t_dec_end = get_time_ms();
        double dec_ms = t_dec_end - t_dec_start;
        fprintf(stderr, "\n=== Decode benchmark ===\n");
        fprintf(stderr, "Decode: %d tokens in %.1f ms (%.1f tok/s, %.1f ms/tok)\n",
                bench_decode_steps, dec_ms,
                bench_decode_steps / (dec_ms / 1000.0),
                dec_ms / bench_decode_steps);
    }

cleanup:
    /* Cleanup */
    free(last_seq_logits);
    free(last_gpu_hidden);
    free(cpu_pf_hidden);
    free(cpu_pf_logits);
    cuda_llm_free(gpu);
    if (cpu_model) transformer_free(cpu_model);
    bpe_vocab_free(vocab);
    gguf_close(gguf);

    return pass ? 0 : 1;
}
