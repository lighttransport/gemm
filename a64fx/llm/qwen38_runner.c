/* Correctness-first Qwen3.8-27B text runner for A64FX. */
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
#include <time.h>
#include <limits.h>
#include <math.h>

#ifdef QWEN38_FAPP
extern void fapp_start(const char *, int, int);
extern void fapp_stop(const char *, int, int);
#else
#define fapp_start(...) ((void)0)
#define fapp_stop(...)  ((void)0)
#endif

extern double tf_decode_matvec_ms;
extern double tf_decode_matvec_bytes;
extern double tf_decode_null_bytes;
extern long tf_decode_matvec_cnt;
extern double tf_decode_attn_qkv_ms, tf_decode_attn_out_ms;
extern double tf_decode_ssm_in_ms, tf_decode_ssm_prepare_ms, tf_decode_ssm_core_ms, tf_decode_ssm_out_ms;
extern double tf_decode_ffn_gateup_ms, tf_decode_ffn_down_ms;
extern double tf_decode_lm_head_ms;

static double now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * t.tv_nsec;
}

static int argmax(const float *x, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (x[i] > x[best]) best = i;
    return best;
}

#define QWEN38_BENCH_MAX_CASES 16

static int parse_int_list(const char *s, int *out, int cap) {
    const char *p = s;
    int n = 0;
    while (*p) {
        char *end = NULL;
        long value;
        while (*p == ' ' || *p == '\t' || *p == ',') p++;
        if (!*p) break;
        value = strtol(p, &end, 10);
        if (end == p || value < 0 || value > INT_MAX || n >= cap)
            return -1;
        out[n++] = (int)value;
        p = end;
        while (*p == ' ' || *p == '\t') p++;
        if (*p && *p != ',') return -1;
    }
    return n > 0 ? n : -1;
}

static void bench_profile_reset(void) {
    transformer_pool_profile_reset();
}

static int run_benchmark(transformer_model *m, bpe_vocab *v, int32_t *tok,
                         const char *prompt, int max_seq, int threads,
                         const int *prompt_sizes, int n_prompt_sizes,
                         const int *gen_sizes, int n_gen_sizes,
                         int warmup, int runs, int csv, size_t weight_bytes) {
    int32_t *base = NULL;
    int base_cap = max_seq;
    int base_nt;
    (void)threads;

    base = malloc((size_t)base_cap * sizeof(*base));
    if (!base) {
        fprintf(stderr, "qwen38: benchmark token buffer allocation failed\n");
        return 1;
    }
    base_nt = bpe_tokenize(v, prompt, -1, base, base_cap);
    if (base_nt <= 0) {
        fprintf(stderr, "qwen38: benchmark seed prompt did not tokenize\n");
        free(base);
        return 1;
    }

    if (csv) {
        printf("mode,prompt_tokens,generated_tokens,warmup,runs,prefill_ms,"
               "decode_ms,prefill_tok_s,decode_tok_s,decode_stddev_tok_s,effective_gb_s\n");
    } else {
        printf("mode       pp   tg   warmup runs   pp ms      tg ms      pp tok/s   tg tok/s\n");
    }

    for (int pi = 0; pi < n_prompt_sizes; pi++) {
        for (int gi = 0; gi < n_gen_sizes; gi++) {
            int pn = prompt_sizes[pi];
            int gn = gen_sizes[gi];
            double pf_sum = 0.0, dec_sum = 0.0, dec_tps_sum = 0.0, dec_tps_sq = 0.0;
            double pf_min = 1e300, dec_min = 1e300;
            double pf_max = 0.0, dec_max = 0.0;
            int total_trials = warmup + runs;

            if (pn < 1 || gn < 1 || pn + gn >= max_seq) {
                fprintf(stderr, "qwen38: benchmark case pp=%d tg=%d exceeds max-seq=%d\n",
                        pn, gn, max_seq);
                free(base);
                return 1;
            }
            for (int i = 0; i < pn; i++) tok[i] = base[i % base_nt];

            for (int trial = 0; trial < total_trials; trial++) {
                transformer_reset_runtime_state(m);
                bench_profile_reset();
                double pf0 = now_sec();
                for (int pos = 0; pos < pn; pos++)
                    transformer_forward_logits(m, tok[pos], pos);
                double pf = now_sec() - pf0;

                int32_t cur = transformer_last_argmax(m);
                double dec0 = now_sec();
                for (int n = 0; n < gn; n++) {
                    transformer_forward_logits(m, cur, pn + n);
                    cur = transformer_last_argmax(m);
                }
                double dec = now_sec() - dec0;

                if (trial >= warmup) {
                    pf_sum += pf;
                    dec_sum += dec;
                    double trial_tps = gn / dec;
                    dec_tps_sum += trial_tps;
                    dec_tps_sq += trial_tps * trial_tps;
                    if (pf < pf_min) pf_min = pf;
                    if (dec < dec_min) dec_min = dec;
                    if (pf > pf_max) pf_max = pf;
                    if (dec > dec_max) dec_max = dec;
                }
            }

            double pf_avg = pf_sum / runs;
            double dec_avg = dec_sum / runs;
            double pf_tps = pn / pf_avg;
            double dec_tps = gn / dec_avg;
            double mean_tps = dec_tps_sum / runs;
            double variance = dec_tps_sq / runs - mean_tps * mean_tps;
            double stddev_tps = sqrt(variance > 0.0 ? variance : 0.0);
            double effective_gbs = (double)weight_bytes * dec_tps / 1e9;
            if (csv) {
                printf("pp+tg,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                       pn, gn, warmup, runs, pf_avg * 1000.0, dec_avg * 1000.0,
                       pf_tps, dec_tps, stddev_tps, effective_gbs);
            } else {
                printf("pp+tg      %4d %4d %7d %4d %10.3f %10.3f %10.3f %10.3f\n",
                       pn, gn, warmup, runs, pf_avg * 1000.0, dec_avg * 1000.0,
                       pf_tps, dec_tps);
                fprintf(stderr, "qwen38: bench pp=%d tg=%d min/max ms %.3f/%.3f %.3f/%.3f\n",
                        pn, gn, pf_min * 1000.0, pf_max * 1000.0,
                        dec_min * 1000.0, dec_max * 1000.0);
                fprintf(stderr, "qwen38: bench decode mean=%.3f stddev=%.3f tok/s effective=%.1f GB/s\n",
                        mean_tps, stddev_tps, effective_gbs);
            }
        }
    }
    free(base);
    return 0;
}

static void usage(const char *p) {
    fprintf(stderr, "usage: %s MODEL --prompt TEXT [--max-gen N] [--max-seq N] "
                    "[--threads N] [--spec-k 0..4] [--mmap] "
                    "[--q8-mode auto|reference|cmg4|block64|block64-ffn|block64-exact|row] "
                    "[--bench --bench-prompt N[,N...] --bench-gen N[,N...] "
                    "--bench-runs N --bench-warmup N --bench-csv]\n", p);
}

int main(int argc, char **argv) {
    const char *path = NULL, *prompt = "Hello";
    const char *q8_mode = "auto";
    const char *bench_prompt_arg = "512";
    const char *bench_gen_arg = "128";
    int max_gen = 16, max_seq = 512, threads = 48, spec_k = 0, mmap_weights = 0;
    int bench = 0, bench_runs = 3, bench_warmup = 1, bench_csv = 0;
    int bench_prompt_sizes[QWEN38_BENCH_MAX_CASES];
    int bench_gen_sizes[QWEN38_BENCH_MAX_CASES];
    int n_bench_prompt = 0, n_bench_gen = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--prompt") && ++i < argc) prompt = argv[i];
        else if (!strcmp(argv[i], "--max-gen") && ++i < argc) max_gen = atoi(argv[i]);
        else if (!strcmp(argv[i], "--max-seq") && ++i < argc) max_seq = atoi(argv[i]);
        else if (!strcmp(argv[i], "--threads") && ++i < argc) threads = atoi(argv[i]);
        else if (!strcmp(argv[i], "--spec-k") && ++i < argc) spec_k = atoi(argv[i]);
        else if (!strcmp(argv[i], "--q8-mode") && ++i < argc) q8_mode = argv[i];
        else if (!strcmp(argv[i], "--mmap")) mmap_weights = 1;
        else if (!strcmp(argv[i], "--bench")) bench = 1;
        else if (!strcmp(argv[i], "--bench-prompt") && ++i < argc) bench_prompt_arg = argv[i];
        else if (!strcmp(argv[i], "--bench-gen") && ++i < argc) bench_gen_arg = argv[i];
        else if (!strcmp(argv[i], "--bench-runs") && ++i < argc) bench_runs = atoi(argv[i]);
        else if (!strcmp(argv[i], "--bench-warmup") && ++i < argc) bench_warmup = atoi(argv[i]);
        else if (!strcmp(argv[i], "--bench-csv")) bench_csv = 1;
        else if (argv[i][0] != '-' && !path) path = argv[i];
        else { usage(argv[0]); return 2; }
    }
    if (!path || spec_k < 0 || spec_k > 4 || max_seq < 2 || max_gen < 0 ||
        threads < 1 || bench_runs < 1 || bench_warmup < 0) {
        usage(argv[0]); return 2;
    }

    if (strcmp(q8_mode, "auto") && strcmp(q8_mode, "reference") &&
        strcmp(q8_mode, "block64") && strcmp(q8_mode, "block64-ffn") &&
        strcmp(q8_mode, "block64-exact") && strcmp(q8_mode, "cmg4") &&
        strcmp(q8_mode, "row")) {
        usage(argv[0]); return 2;
    }
    if (bench) {
        if (spec_k != 0 || parse_int_list(bench_prompt_arg, bench_prompt_sizes,
                                          QWEN38_BENCH_MAX_CASES) < 0 ||
            parse_int_list(bench_gen_arg, bench_gen_sizes,
                           QWEN38_BENCH_MAX_CASES) < 0) {
            fprintf(stderr, "qwen38: benchmark requires --spec-k 0 and valid size lists\n");
            return 2;
        }
        n_bench_prompt = parse_int_list(bench_prompt_arg, bench_prompt_sizes,
                                        QWEN38_BENCH_MAX_CASES);
        n_bench_gen = parse_int_list(bench_gen_arg, bench_gen_sizes,
                                     QWEN38_BENCH_MAX_CASES);
    }
    if (!strcmp(q8_mode, "cmg4") && threads != 48) {
        fprintf(stderr, "qwen38: --q8-mode cmg4 requires --threads 48\n");
        return 2;
    }
    if (!strcmp(q8_mode, "cmg4")) {
        char *resolved = realpath(path, NULL);
        if (!resolved || strncmp(resolved, "/local/", 7) != 0) {
            fprintf(stderr, "qwen38: cmg4 weights must first be staged under /local\n");
            free(resolved);
            return 2;
        }
        fprintf(stderr, "qwen38: verified node-local GGUF %s; staging into HBM2 next\n",
                resolved);
        free(resolved);
    }
    double load0 = now_sec();
    /* Inspect through a lazy mapping first. Q8 cannot afford a full anonymous
     * GGUF allocation, while smaller formats are reopened through the normal
     * anonymous/NUMA loader. */
    setenv("GGUF_LAZY_MMAP", "1", 0);
    gguf_context *g = gguf_open_multi(path, 1);
    if (!g) return 1;
    int q8_tensors = 0;
    for (uint64_t i = 0; i < g->n_tensors; i++)
        if (g->tensors[i].type == GGML_TYPE_Q8_0 && g->tensors[i].n_dims >= 2)
            q8_tensors++;
    int q8_model = q8_tensors > 100;
    if (!q8_model && !mmap_weights) {
        gguf_close(g);
        g = gguf_open_multi(path, 0);
        if (!g) return 1;
    }
    bpe_vocab *v = bpe_vocab_load(g);
    transformer_model *m = transformer_load(g, max_seq);
    if (!v || !m) return 1;
    if (threads > 1) transformer_set_threads(m, threads);
    size_t q8_resident = 0;
    if (q8_model && !mmap_weights) {
        if (spec_k) {
            fprintf(stderr, "qwen38: selective Q8 residency currently requires --spec-k 0\n");
            return 1;
        }
        int resident_mode = !strcmp(q8_mode, "row") ? 1 :
                            !strcmp(q8_mode, "block64") ? 2 :
                            !strcmp(q8_mode, "block64-ffn") ? 3 :
                            !strcmp(q8_mode, "block64-exact") ? 4 :
                            !strcmp(q8_mode, "cmg4") ? 5 : 0;
        q8_resident = transformer_materialize_q8_decode(m, g, resident_mode);
        if (!q8_resident) return 1;
    } else if (!mmap_weights) {
        transformer_numa_setup(m, g);
    }
    if (!getenv("TF_NO_PANEL")) transformer_build_panels(m);
    fprintf(stderr, "qwen38: load=%.3fs trunk=%d nextn=%d format=%s\n",
            now_sec() - load0, m->n_layers, m->n_nextn_layers,
            mmap_weights ? "mmap" : (q8_resident ? "selective-q8" : "anonymous"));
    if (getenv("TF_NULL_GEMM") && atoi(getenv("TF_NULL_GEMM")) == 2) {
        transformer_null_stream_bench(m, 3);
        transformer_free(m);
        bpe_vocab_free(v);
        gguf_close(g);
        return 0;
    }
    if (spec_k && !m->nextn.loaded) {
        fprintf(stderr, "qwen38: --spec-k requires native NextN tensors\n");
        return 1;
    }

    int32_t *tok = malloc((size_t)max_seq * sizeof(*tok));
    if (!tok) {
        fprintf(stderr, "qwen38: token buffer allocation failed\n");
        transformer_free(m);
        bpe_vocab_free(v);
        gguf_close(g);
        return 1;
    }
    if (bench) {
        int rc = run_benchmark(m, v, tok, prompt, max_seq, threads,
                               bench_prompt_sizes, n_bench_prompt,
                               bench_gen_sizes, n_bench_gen,
                               bench_warmup, bench_runs, bench_csv, q8_resident);
        free(tok);
        transformer_free(m);
        bpe_vocab_free(v);
        gguf_close(g);
        return rc;
    }

    int nt = bpe_tokenize(v, prompt, -1, tok, max_seq);
    if (nt <= 0 || nt + max_gen + spec_k >= max_seq) {
        fprintf(stderr, "qwen38: invalid/too-long prompt (%d tokens; reserve %d drafts)\n",
                nt, spec_k);
        return 1;
    }
    int module_profile = getenv("TF_MODULE_PROFILE") != NULL;
    if (module_profile) setenv("TF_DPROF", "1", 1);
    tf_decode_matvec_ms = tf_decode_matvec_bytes = 0.0;
    tf_decode_matvec_cnt = 0;
    tf_decode_attn_qkv_ms = tf_decode_attn_out_ms = 0.0;
    tf_decode_ssm_in_ms = tf_decode_ssm_prepare_ms = tf_decode_ssm_core_ms = tf_decode_ssm_out_ms = 0.0;
    tf_decode_ffn_gateup_ms = tf_decode_ffn_down_ms = tf_decode_lm_head_ms = 0.0;
    double prefill_t0 = now_sec();
    float *logits = NULL;
    int pos = 0;
    for (; pos < nt; pos++) {
        logits = transformer_forward_logits(m, tok[pos], pos);
        if (spec_k) transformer_nextn_logits(m, tok[pos], transformer_get_hidden(m), pos);
    }
    double prefill_dt = now_sec() - prefill_t0;
    double pf_matvec = tf_decode_matvec_ms, pf_qkv = tf_decode_attn_qkv_ms;
    double pf_attn_out = tf_decode_attn_out_ms;
    double pf_ssm_in = tf_decode_ssm_in_ms, pf_ssm_prepare = tf_decode_ssm_prepare_ms;
    double pf_ssm_core = tf_decode_ssm_core_ms, pf_ssm_out = tf_decode_ssm_out_ms;
    double pf_ffn_gateup = tf_decode_ffn_gateup_ms, pf_ffn_down = tf_decode_ffn_down_ms;
    double pf_lm_head = tf_decode_lm_head_ms;
    int32_t cur = transformer_last_argmax(m);
    int dump_tokens = getenv("TF_DUMP_TOKENS") && atoi(getenv("TF_DUMP_TOKENS")) != 0;
    tf_decode_matvec_ms = 0.0;
    tf_decode_matvec_bytes = 0.0;
    tf_decode_null_bytes = 0.0;
    tf_decode_matvec_cnt = 0;
    tf_decode_attn_qkv_ms = tf_decode_attn_out_ms = 0.0;
    tf_decode_ssm_in_ms = tf_decode_ssm_prepare_ms = tf_decode_ssm_core_ms = tf_decode_ssm_out_ms = 0.0;
    tf_decode_ffn_gateup_ms = tf_decode_ffn_down_ms = 0.0;
    tf_decode_lm_head_ms = 0.0;
    double dec0 = now_sec();
    fapp_start("qwen38_decode", 1, 0);
    long mtp_match = 0, mtp_total = 0;
    int pending_draft = -1;
    for (int n = 0; n < max_gen; n++, pos++) {
        if (dump_tokens)
            fprintf(stderr, "qwen38: token n=%d pos=%d id=%d logit=%a\n",
                    n, pos, cur, logits ? logits[cur] : 0.0f);
        if (pending_draft >= 0) {
            mtp_match += pending_draft == cur;
            mtp_total++;
        }
        const char *piece = bpe_token_to_str(v, cur);
        if (piece) fputs(piece, stdout);
        fflush(stdout);
        logits = transformer_forward_logits(m, cur, pos);
        int32_t next = transformer_last_argmax(m);
        pending_draft = -1;
        if (spec_k) {
            const float *h = transformer_get_hidden(m);
            int32_t prev = cur;
            for (int k = 0; k < spec_k; k++) {
                float *dl = transformer_nextn_logits(m, prev, h, pos + k);
                int32_t draft = argmax(dl, m->n_vocab);
                if (k == 0) pending_draft = draft;
                prev = draft;
                h = transformer_nextn_hidden(m);
            }
        }
        cur = next;
        if (cur == v->eos_id || cur == v->eot_id) break;
    }
    fputc('\n', stdout);
    fapp_stop("qwen38_decode", 1, 0);
    double dt = now_sec() - dec0;
    fprintf(stderr, "qwen38: decode=%d tokens %.3fs %.3f tok/s",
            pos - nt, dt, dt > 0 ? (pos - nt) / dt : 0.0);
    if (mtp_total) fprintf(stderr, " mtp_greedy_match=%ld/%ld alpha=%.4f",
                           mtp_match, mtp_total, (double)mtp_match / mtp_total);
    fputc('\n', stderr);
    if (getenv("TF_DPROF")) {
        double mat_ms = tf_decode_matvec_ms;
        double mat_bw = mat_ms > 0.0 ? tf_decode_matvec_bytes / (mat_ms * 1e6) : 0.0;
        fprintf(stderr, "qwen38: dprof matvec=%.1f ms/tok serial=%.1f ms/tok matvec=%.1f%% BW=%.1f GB/s dispatches=%ld/tok\n",
                mat_ms / (pos - nt), (dt * 1000.0 - mat_ms) / (pos - nt),
                100.0 * mat_ms / (dt * 1000.0), mat_bw,
                tf_decode_matvec_cnt / (pos - nt));
        fprintf(stderr, "qwen38: stages ms/tok attn_qkv=%.1f attn_out=%.1f "
                        "ssm_in=%.1f ssm_prepare=%.1f ssm_core=%.1f ssm_out=%.1f "
                        "ffn_gateup=%.1f ffn_down=%.1f lm_head=%.1f\n",
                tf_decode_attn_qkv_ms / (pos - nt), tf_decode_attn_out_ms / (pos - nt),
                tf_decode_ssm_in_ms / (pos - nt), tf_decode_ssm_prepare_ms / (pos - nt),
                tf_decode_ssm_core_ms / (pos - nt),
                tf_decode_ssm_out_ms / (pos - nt),
                tf_decode_ffn_gateup_ms / (pos - nt), tf_decode_ffn_down_ms / (pos - nt),
                tf_decode_lm_head_ms / (pos - nt));
    }
    if (module_profile) {
        double p = nt > 0 ? (double)nt : 1.0;
        fprintf(stderr, "qwen38: module prefill=%d tokens %.3fs %.3f tok/s "
                        "matvec=%.1f ms/tok attn_qkv=%.1f attn_out=%.1f "
                        "ssm_in=%.1f ssm_prepare=%.1f ssm_core=%.1f ssm_out=%.1f "
                        "ffn_gateup=%.1f ffn_down=%.1f lm_head=%.1f\n",
                nt, prefill_dt, nt / prefill_dt, pf_matvec / p, pf_qkv / p,
                pf_attn_out / p, pf_ssm_in / p, pf_ssm_prepare / p,
                pf_ssm_core / p, pf_ssm_out / p, pf_ffn_gateup / p,
                pf_ffn_down / p, pf_lm_head / p);
    }
    if (getenv("TF_NULL_GEMM") && pos > nt) {
        double stream_gbs = tf_decode_null_bytes / (dt * 1e9);
        double bytes_per_tok = tf_decode_null_bytes / (double)(pos - nt);
        fprintf(stderr, "qwen38: null-stream=%.3f GB/tok %.1f GB/s; "
                "20 tok/s requires %.1f GB/s (equiv %.2f tok/s)\n",
                bytes_per_tok / 1e9, stream_gbs,
                20.0 * bytes_per_tok / 1e9,
                stream_gbs / (bytes_per_tok / 1e9));
    }

    free(tok);
    transformer_free(m);
    bpe_vocab_free(v);
    gguf_close(g);
    return 0;
}
