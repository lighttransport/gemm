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

static void usage(const char *p) {
    fprintf(stderr, "usage: %s MODEL --prompt TEXT [--max-gen N] [--max-seq N] "
                    "[--threads N] [--spec-k 0..4] [--mmap] "
                    "[--q8-mode auto|reference|block64|block64-ffn|block64-exact|row]\n", p);
}

int main(int argc, char **argv) {
    const char *path = NULL, *prompt = "Hello";
    const char *q8_mode = "auto";
    int max_gen = 16, max_seq = 512, threads = 48, spec_k = 0, mmap_weights = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--prompt") && ++i < argc) prompt = argv[i];
        else if (!strcmp(argv[i], "--max-gen") && ++i < argc) max_gen = atoi(argv[i]);
        else if (!strcmp(argv[i], "--max-seq") && ++i < argc) max_seq = atoi(argv[i]);
        else if (!strcmp(argv[i], "--threads") && ++i < argc) threads = atoi(argv[i]);
        else if (!strcmp(argv[i], "--spec-k") && ++i < argc) spec_k = atoi(argv[i]);
        else if (!strcmp(argv[i], "--q8-mode") && ++i < argc) q8_mode = argv[i];
        else if (!strcmp(argv[i], "--mmap")) mmap_weights = 1;
        else if (argv[i][0] != '-' && !path) path = argv[i];
        else { usage(argv[0]); return 2; }
    }
    if (!path || spec_k < 0 || spec_k > 4 || max_seq < 2 || max_gen < 0) {
        usage(argv[0]); return 2;
    }

    if (strcmp(q8_mode, "auto") && strcmp(q8_mode, "reference") &&
        strcmp(q8_mode, "block64") && strcmp(q8_mode, "block64-ffn") &&
        strcmp(q8_mode, "block64-exact") && strcmp(q8_mode, "row")) {
        usage(argv[0]); return 2;
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
                            !strcmp(q8_mode, "block64-exact") ? 4 : 0;
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
    int nt = bpe_tokenize(v, prompt, -1, tok, max_seq);
    if (nt <= 0 || nt + max_gen + spec_k >= max_seq) {
        fprintf(stderr, "qwen38: invalid/too-long prompt (%d tokens; reserve %d drafts)\n",
                nt, spec_k);
        return 1;
    }
    float *logits = NULL;
    int pos = 0;
    for (; pos < nt; pos++) {
        logits = transformer_forward_logits(m, tok[pos], pos);
        if (spec_k) transformer_nextn_logits(m, tok[pos], transformer_get_hidden(m), pos);
    }
    int32_t cur = transformer_last_argmax(m);
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
