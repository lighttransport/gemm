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
extern long tf_decode_matvec_cnt;

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
                    "[--threads N] [--spec-k 0..4] [--mmap]\n", p);
}

int main(int argc, char **argv) {
    const char *path = NULL, *prompt = "Hello";
    int max_gen = 16, max_seq = 512, threads = 48, spec_k = 0, mmap_weights = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--prompt") && ++i < argc) prompt = argv[i];
        else if (!strcmp(argv[i], "--max-gen") && ++i < argc) max_gen = atoi(argv[i]);
        else if (!strcmp(argv[i], "--max-seq") && ++i < argc) max_seq = atoi(argv[i]);
        else if (!strcmp(argv[i], "--threads") && ++i < argc) threads = atoi(argv[i]);
        else if (!strcmp(argv[i], "--spec-k") && ++i < argc) spec_k = atoi(argv[i]);
        else if (!strcmp(argv[i], "--mmap")) mmap_weights = 1;
        else if (argv[i][0] != '-' && !path) path = argv[i];
        else { usage(argv[0]); return 2; }
    }
    if (!path || spec_k < 0 || spec_k > 4 || max_seq < 2 || max_gen < 0) {
        usage(argv[0]); return 2;
    }

    double load0 = now_sec();
    gguf_context *g = gguf_open_multi(path, mmap_weights ? 1 : 0);
    if (!g) return 1;
    bpe_vocab *v = bpe_vocab_load(g);
    transformer_model *m = transformer_load(g, max_seq);
    if (!v || !m) return 1;
    if (threads > 1) transformer_set_threads(m, threads);
    transformer_numa_setup(m, g);
    if (!getenv("TF_NO_PANEL")) transformer_build_panels(m);
    fprintf(stderr, "qwen38: load=%.3fs trunk=%d nextn=%d format=%s\n",
            now_sec() - load0, m->n_layers, m->n_nextn_layers,
            mmap_weights ? "mmap" : "anonymous");
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
    int32_t cur = argmax(logits, m->n_vocab);
    tf_decode_matvec_ms = 0.0;
    tf_decode_matvec_bytes = 0.0;
    tf_decode_matvec_cnt = 0;
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
        int32_t next = argmax(logits, m->n_vocab);
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
    }

    free(tok);
    transformer_free(m);
    bpe_vocab_free(v);
    gguf_close(g);
    return 0;
}
