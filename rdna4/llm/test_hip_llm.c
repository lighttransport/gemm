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

/* ---- Quant-matvec A/B verifier (--verify-quant-kernels) ----
 *
 * For each ported IQ/TQ matvec kernel, run the runner-internal verifier
 * that compares the GPU launch_matvec_<type> against dequantize_row_<type>
 * + scalar matvec on identical random raw bytes. */
typedef void (*deq_row_fn)(const void *src, float *dst, int n);

static void dequantize_row_q8_0_padded(const void *src, float *dst, int n) {
    const unsigned char *p = (const unsigned char *)src;
    int nb = n / 32;
    for (int b = 0; b < nb; b++) {
        const unsigned char *bp = p + (size_t)b * 36;
        uint16_t dh;
        memcpy(&dh, bp, sizeof(dh));
        float d = ggml_fp16_to_fp32(dh);
        const signed char *qs = (const signed char *)(bp + 4);
        for (int i = 0; i < 32; i++) dst[b * 32 + i] = d * (float)qs[i];
    }
}

static int quant_type_from_name(const char *s) {
    if (!s) return -1;
    if (strcmp(s, "Q2_K") == 0)    return GGML_TYPE_Q2_K;
    if (strcmp(s, "Q3_K") == 0)    return GGML_TYPE_Q3_K;
    if (strcmp(s, "Q4_K") == 0)    return GGML_TYPE_Q4_K;
    if (strcmp(s, "Q5_K") == 0)    return GGML_TYPE_Q5_K;
    if (strcmp(s, "Q6_K") == 0)    return GGML_TYPE_Q6_K;
    if (strcmp(s, "Q8_0") == 0)    return GGML_TYPE_Q8_0;
    if (strcmp(s, "Q4_0") == 0)    return GGML_TYPE_Q4_0;
    if (strcmp(s, "Q4_1") == 0)    return GGML_TYPE_Q4_1;
    if (strcmp(s, "Q5_0") == 0)    return GGML_TYPE_Q5_0;
    if (strcmp(s, "Q5_1") == 0)    return GGML_TYPE_Q5_1;
    if (strcmp(s, "IQ2_XXS") == 0) return GGML_TYPE_IQ2_XXS;
    if (strcmp(s, "IQ2_XS") == 0)  return GGML_TYPE_IQ2_XS;
    if (strcmp(s, "IQ2_S") == 0)   return GGML_TYPE_IQ2_S;
    if (strcmp(s, "IQ3_XXS") == 0) return GGML_TYPE_IQ3_XXS;
    if (strcmp(s, "IQ3_S") == 0)   return GGML_TYPE_IQ3_S;
    if (strcmp(s, "IQ1_S") == 0)   return GGML_TYPE_IQ1_S;
    if (strcmp(s, "IQ1_M") == 0)   return GGML_TYPE_IQ1_M;
    if (strcmp(s, "IQ4_NL") == 0)  return GGML_TYPE_IQ4_NL;
    if (strcmp(s, "IQ4_XS") == 0)  return GGML_TYPE_IQ4_XS;
    if (strcmp(s, "TQ1_0") == 0)   return GGML_TYPE_TQ1_0;
    if (strcmp(s, "TQ2_0") == 0)   return GGML_TYPE_TQ2_0;
    return -1;
}

static int run_verify_quant_kernels(void) {
    fprintf(stderr, "=== --verify-quant-kernels: A/B HIP matvec vs CPU dequant+matvec ===\n");
    fprintf(stderr, "Shape: n_rows=64, n_cols=512.  Pass threshold: rel_l2 < 1e-4.\n\n");

    hip_llm_runner *r = hip_llm_init(0, 0);
    if (!r) { fprintf(stderr, "hip_llm_init failed\n"); return 1; }

    struct { int type; const char *name; deq_row_fn fn; } cases[] = {
        { GGML_TYPE_Q2_K,    "Q2_K",    dequantize_row_q2_K    },
        { GGML_TYPE_Q3_K,    "Q3_K",    dequantize_row_q3_K    },
        { GGML_TYPE_Q4_K,    "Q4_K",    dequantize_row_q4_K    },
        { GGML_TYPE_Q5_K,    "Q5_K",    dequantize_row_q5_K    },
        { GGML_TYPE_Q6_K,    "Q6_K",    dequantize_row_q6_K    },
        { GGML_TYPE_Q8_0,    "Q8_0",    dequantize_row_q8_0_padded },
        { GGML_TYPE_Q4_0,    "Q4_0",    dequantize_row_q4_0    },
        { GGML_TYPE_Q4_1,    "Q4_1",    dequantize_row_q4_1    },
        { GGML_TYPE_Q5_0,    "Q5_0",    dequantize_row_q5_0    },
        { GGML_TYPE_Q5_1,    "Q5_1",    dequantize_row_q5_1    },
        { GGML_TYPE_IQ2_XXS, "IQ2_XXS", dequantize_row_iq2_xxs },
        { GGML_TYPE_IQ2_XS,  "IQ2_XS",  dequantize_row_iq2_xs  },
        { GGML_TYPE_IQ2_S,   "IQ2_S",   dequantize_row_iq2_s   },
        { GGML_TYPE_IQ3_XXS, "IQ3_XXS", dequantize_row_iq3_xxs },
        { GGML_TYPE_IQ3_S,   "IQ3_S",   dequantize_row_iq3_s   },
        { GGML_TYPE_IQ1_S,   "IQ1_S",   dequantize_row_iq1_s   },
        { GGML_TYPE_IQ1_M,   "IQ1_M",   dequantize_row_iq1_m   },
        { GGML_TYPE_IQ4_NL,  "IQ4_NL",  dequantize_row_iq4_nl  },
        { GGML_TYPE_IQ4_XS,  "IQ4_XS",  dequantize_row_iq4_xs  },
        { GGML_TYPE_TQ1_0,   "TQ1_0",   dequantize_row_tq1_0   },
        { GGML_TYPE_TQ2_0,   "TQ2_0",   dequantize_row_tq2_0   },
    };
    int n_cases = (int)(sizeof(cases) / sizeof(cases[0]));
    int n_pass = 0, n_fail = 0, n_skip = 0;
    const double thresh = 1e-4;

    fprintf(stderr, "%-8s  %12s  %12s   %s\n", "type", "rel_l2", "max_abs", "result");
    fprintf(stderr, "%-8s  %12s  %12s   %s\n", "----", "------", "-------", "------");
    for (int i = 0; i < n_cases; i++) {
        double rel_l2 = 0.0, max_abs = 0.0;
        int rc = hip_llm_verify_quant_matvec(r, cases[i].type, cases[i].fn,
                                             64, 512, &rel_l2, &max_abs);
        if (rc != 0) {
            fprintf(stderr, "%-8s  %12s  %12s   SKIP (rc=%d)\n",
                    cases[i].name, "-", "-", rc);
            n_skip++;
            continue;
        }
        int pass = (rel_l2 < thresh);
        fprintf(stderr, "%-8s  %12.3e  %12.3e   %s\n",
                cases[i].name, rel_l2, max_abs, pass ? "PASS" : "FAIL");
        if (pass) n_pass++; else n_fail++;
    }
    fprintf(stderr, "\n%d PASS, %d FAIL, %d SKIP (threshold rel_l2 < %.0e).\n",
            n_pass, n_fail, n_skip, thresh);

    hip_llm_free(r);
    return n_fail == 0 ? 0 : 2;
}

static int cmp_float_asc(const void *a, const void *b) {
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

static int run_bench_quant_matvec(const char *type_name, int n_rows, int n_cols, int iters, int repeats) {
    int type = quant_type_from_name(type_name);
    if (type < 0 || n_rows <= 0 || n_cols <= 0 || iters <= 0 || repeats <= 0) {
        fprintf(stderr, "Invalid --bench-quant-matvec args. Usage: --bench-quant-matvec TYPE ROWS COLS ITERS [REPEATS]\n");
        return 1;
    }

    hip_llm_runner *r = hip_llm_init(0, 0);
    if (!r) { fprintf(stderr, "hip_llm_init failed\n"); return 1; }

    float *samples = (float *)malloc((size_t)repeats * sizeof(float));
    if (!samples) {
        hip_llm_free(r);
        return 1;
    }
    int rc = 0;
    for (int rep = 0; rep < repeats; rep++) {
        rc = hip_llm_bench_quant_matvec(r, type, n_rows, n_cols, 20, iters, &samples[rep]);
        if (rc != 0) break;
    }
    hip_llm_free(r);
    if (rc != 0) {
        free(samples);
        fprintf(stderr, "--bench-quant-matvec failed (type=%s rows=%d cols=%d iters=%d repeats=%d rc=%d)\n",
                type_name, n_rows, n_cols, iters, repeats, rc);
        return 1;
    }

    qsort(samples, (size_t)repeats, sizeof(float), cmp_float_asc);
    float ms = samples[repeats / 2];
    float min_ms = samples[0];
    float max_ms = samples[repeats - 1];
    double dot_ops = 2.0 * (double)n_rows * (double)n_cols;
    double gops = dot_ops / (double)ms / 1.0e6;
    fprintf(stderr,
            "quant_matvec %s rows=%d cols=%d iters=%d repeats=%d: median %.6f ms/launch  %.2f GOP/s  range [%.6f, %.6f]\n",
            type_name, n_rows, n_cols, iters, repeats, ms, gops, min_ms, max_ms);
    free(samples);
    return 0;
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
    int verify_quant_kernels = 0; /* --verify-quant-kernels: A/B HIP vs CPU per quant type, then exit */
    const char *bench_qmv_type = NULL; /* --bench-quant-matvec TYPE ROWS COLS ITERS [REPEATS] */
    int bench_qmv_rows = 0, bench_qmv_cols = 0, bench_qmv_iters = 0, bench_qmv_repeats = 1;

    /* --st <qwen3.safetensors>: sanity-check the safetensors loader + hidden
     * snapshots (text-encoder path). Loads, runs a few forwards, prints snapshot
     * norms (finite + reasonable = loader OK), then exits. */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--st") == 0 && i + 1 < argc) {
            const char *st_path = argv[i + 1];
            hip_llm_runner *r = hip_llm_init(0, 1);
            if (!r || hip_llm_load_weights_qwen3_safetensors(r, st_path, 512) != 0) {
                fprintf(stderr, "--st: load failed\n"); return 1;
            }
            int hs[3] = {8, 17, 26};
            if (hip_llm_set_hidden_snapshot_layers(r, hs, 3) != 0) { fprintf(stderr, "--st: set snapshots failed\n"); return 1; }
            int n = hip_llm_n_embd(r);
            float *snap = (float *)malloc((size_t)3 * n * sizeof(float));
            hip_llm_reset_state(r);
            for (int pos = 0; pos < 5; pos++) {
                int32_t tok = (int32_t)(100 + pos);
                if (!hip_llm_forward(r, tok, pos)) { fprintf(stderr, "--st: forward failed at pos %d\n", pos); return 1; }
                if (hip_llm_read_hidden_snapshots(r, snap, 3, n) != 0) { fprintf(stderr, "--st: read snapshots failed\n"); return 1; }
                for (int s = 0; s < 3; s++) {
                    double nn = 0; int nan = 0;
                    for (int j = 0; j < n; j++) { float v = snap[s*n+j]; if (v != v) nan = 1; nn += (double)v*v; }
                    printf("pos %d  layer[%d]  norm=%.4f  first=%.5f%s\n", pos, hs[s], sqrt(nn), snap[s*n], nan ? "  NaN!" : "");
                }
            }
            free(snap); hip_llm_free(r);
            return 0;
        }
    }

    /* Parse args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--verify-quant-kernels") == 0) {
            verify_quant_kernels = 1;
        } else if (strcmp(argv[i], "--bench-quant-matvec") == 0 && i + 4 < argc) {
            bench_qmv_type = argv[++i];
            bench_qmv_rows = atoi(argv[++i]);
            bench_qmv_cols = atoi(argv[++i]);
            bench_qmv_iters = atoi(argv[++i]);
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                bench_qmv_repeats = atoi(argv[++i]);
            }
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
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
            fprintf(stderr, "       [--verify-quant-kernels] [--bench-quant-matvec TYPE ROWS COLS ITERS [REPEATS]]\n");
            return 1;
        }
    }

    /* --verify-quant-kernels has no model dependency; run it and exit. */
    if (verify_quant_kernels) {
        return run_verify_quant_kernels();
    }
    if (bench_qmv_type) {
        return run_bench_quant_matvec(bench_qmv_type, bench_qmv_rows, bench_qmv_cols,
                                      bench_qmv_iters, bench_qmv_repeats);
    }

    if (!model_path) {
        fprintf(stderr, "Usage: %s <model.gguf> [-t \"prompt\"] [-n max_tokens] [-s max_seq_len]\n", argv[0]);
        fprintf(stderr, "       [--bench] [--gpu-only-bench] [--decode N] [--prefill-len M]\n");
        fprintf(stderr, "       [--verify-quant-kernels]   (standalone; no model needed)\n");
        fprintf(stderr, "       [--bench-quant-matvec TYPE ROWS COLS ITERS [REPEATS]]   (standalone)\n");
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

    /* Tokenize prompt. Buffer sized to hold a large --prefill-len pad. */
    int tok_cap = (prefill_pad > 4096) ? (prefill_pad + 16) : 4096;
    int32_t *tokens = (int32_t *)malloc((size_t)tok_cap * sizeof(int32_t));
    if (!tokens) { fprintf(stderr, "tokens alloc failed\n"); return 1; }
    int n_tokens = bpe_tokenize(vocab, prompt, -1, tokens, tok_cap);
    if (n_tokens <= 0) {
        fprintf(stderr, "Tokenization failed\n");
        bpe_vocab_free(vocab);
        gguf_close(gguf);
        return 1;
    }
    /* Prepend BOS (Gemma expects it; bpe_tokenize here does not add it). Default to
     * the GGUF's tokenizer.ggml.bos_token_id; LLM_ADD_BOS overrides (0 disables). */
    {
        int bos = -1;
        const char *e = getenv("LLM_ADD_BOS");
        if (e) bos = atoi(e);
        else { int bi = gguf_find_key(gguf, "tokenizer.ggml.bos_token_id");
               if (bi >= 0) bos = (int)gguf->kv[bi].value.u32; }
        if (bos > 0 && (n_tokens == 0 || tokens[0] != bos)) {
            for (int i = n_tokens; i > 0; i--) tokens[i] = tokens[i-1];
            tokens[0] = bos; n_tokens++;
        }
    }
    fprintf(stderr, "Prompt: \"%s\" -> %d tokens:", prompt, n_tokens);
    for (int i = 0; i < n_tokens && i < 32; i++) {
        fprintf(stderr, " %d", tokens[i]);
    }
    if (n_tokens > 32) fprintf(stderr, " ...");
    fprintf(stderr, "\n");

    /* --prefill-len M: pad prompt by repeating last token until length == M */
    if (prefill_pad > n_tokens && prefill_pad <= tok_cap) {
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
    if (getenv("LLM_DEBUG_LAYERS")) hip_llm_set_debug(gpu, 1);

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

        /* Warm-up prefill (first call builds hipBLASLt plans; not counted). */
        const char *warmup_env = getenv("LLM_PREFILL_WARMUP");
        int n_warmup = warmup_env ? atoi(warmup_env) : 0;
        for (int w = 0; w < n_warmup; w++) {
            float *lg = hip_llm_forward_batch_logits(gpu, tokens, n_prefill, 0);
            if (!lg) { fprintf(stderr, "GPU prefill warmup %d failed\n", w); pass = 0; goto bench_done; }
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
            int gen_text = (getenv("LLM_GEN_TEXT") != NULL);
            if (gen_text) fprintf(stderr, "\n=== Generated text ===\n%s", bpe_token_to_str(vocab, next_tok));
            double t_dec0 = get_time_ms();
            for (int k = 0; k < decode_n; k++) {
                int pos = n_prefill + k;
                float *lg = hip_llm_forward_logits(gpu, next_tok, pos);
                if (!lg) { fprintf(stderr, "GPU forward_logits failed at decode k=%d\n", k); pass = 0; break; }
                next_tok = argmax_logits(lg, n_vocab);
                if (gen_text) { const char *s = bpe_token_to_str(vocab, next_tok); if (s) fprintf(stderr, "%s", s); }
            }
            if (gen_text) fprintf(stderr, "\n=== end ===\n");
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
