/*
 * ds4f_serve_lib.c -- single-node DS4F model serving library.
 *
 * Thin ctypes-friendly wrapper around the exact/tierb2/mHC forward path so a
 * Python persistent runner (ds4f_serve_runner.py) can drive the model without
 * reimplementing the MLA/MoE/compressor math in Python.
 *
 *   open(stage_dir, use_hip, hip_device, threads, cmgs, max_pos)
 *   prefill(ids, n)          -- prompt tokens -> KV + last logits
 *   decode(token, pos)       -- one generation step
 *   sample(sampling)         -- pick the next token from the current logits
 *   logits()                 -- full-vocab logits of the last step
 *   kv_save(path) / kv_restore(path)   -- prefix-cache KV + compressor state
 *   reset()                  -- fresh context at position 0
 *
 * The model math (ds4f_forward_token / the compressor) stays in C; the Python
 * runner owns the request protocol, sampling penalties, and the chat loop.
 *
 * Build:  see the Makefile target `libds4f_serve.so`.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "common/ds4f.h"
#if defined(DS4F_SERVE_HIP)
#include "hetero/ds4f/hip_ds4f_dense.h"
#endif

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    double temperature, top_p;
    int top_k;
    double presence_penalty, repeat_penalty;
    long long seed;
} ds4f_serve_sampling;

typedef struct ds4f_serve {
    ds4f_model *m;
    ds4f_mem_pool *pool;
    float *x;          /* embedding scratch [hidden] */
    float *logits;     /* last-step full-vocab logits */
    size_t logits_cap;
    int vocab, hidden;
    int pos;           /* next position to decode into */
    uint64_t rng;
    int eos;
    int *hist;         /* recent generated tokens (for the penalties) */
    int n_hist, hist_cap;
#if defined(DS4F_SERVE_HIP)
    hip_ds4f_dense *hip;
#endif
} ds4f_serve;

void ds4f_serve_close(ds4f_serve *s);

static int env_i(const char *k, int d) { const char *e = getenv(k); return e && *e ? atoi(e) : d; }

typedef struct { float l; int id; } ds4f_pair;
static int cmp_pair(const void *a, const void *b) {
    const ds4f_pair *pa = (const ds4f_pair *)a, *pb = (const ds4f_pair *)b;
    if (pa->l < pb->l) return -1;
    if (pa->l > pb->l) return 1;
    return pa->id - pb->id;
}

static int embed_lookup(const ds4f_model *m, int tok, float *x) {
    if (tok < 0 || tok >= m->cfg.vocab || m->emb_rows != m->cfg.vocab || !m->embed)
        return -1;
    const uint16_t *row = m->embed + (size_t)tok * (size_t)m->cfg.hidden;
    for (int i = 0; i < m->cfg.hidden; i++) {
        uint32_t u = (uint32_t)row[i] << 16;
        memcpy(x + i, &u, sizeof(u));
    }
    return 0;
}

#if defined(DS4F_SERVE_HIP)
/* Bind the dense bank (FP8 ordered, matching the benchmark's decode config) and
 * attach the decode hooks so ds4f_forward_token uses the ROCm for the dense
 * projections + the streamed MXFP4 experts; the routed experts stay CPU. */
static int serve_attach_hip(ds4f_serve *s, int hip_device, int verbose,
                            int ordered_layers) {
    ds4f_model *m = s->m;
    s->hip = hip_ds4f_dense_create_ex(hip_device, verbose, 1);
    if (!s->hip) return -1;
    for (int L = 0; L < m->cfg.n_layers; ++L) {
        ds4f_layer *z = &m->layers[L];
        ds4f_tensor *ts[9] = { &z->wq_a, &z->wq_b, &z->wkv, &z->wo_a, &z->wo_b,
                               &z->sh_w1, &z->sh_w3, &z->sh_w2, &z->gate };
        for (int j = 0; j < 9; ++j) {
            int ordered = L < ordered_layers && ts[j]->type == DS4F_FP8;
            int id = ts[j]->type == DS4F_BF16
                ? hip_ds4f_dense_bind_bf16_tensor(s->hip, ts[j])
                : ordered
                    ? hip_ds4f_dense_bind_fp8_ordered_tensor(s->hip, ts[j])
                    : hip_ds4f_dense_bind_tensor(s->hip, ts[j]);
            if (id < 0) return -1;
        }
    }
    if (m->head.type == DS4F_BF16)
        hip_ds4f_dense_bind_bf16_tensor(s->hip, &m->head);
    m->gpu_dense_ctx = s->hip;
    m->gpu_dense_matvec = hip_ds4f_dense_matvec_tensor;
    m->gpu_dense_async_multi = hip_ds4f_dense_matvec_tensors_async;
    m->gpu_dense_wait = hip_ds4f_dense_wait_tensors;
    m->gpu_dense_blockdiag = hip_ds4f_dense_matvec_blockdiag;
    m->gpu_dense_gemm = hip_ds4f_dense_gemm_tensor;
    m->gpu_dense_gemm_multi = hip_ds4f_dense_gemm_tensors;
    m->gpu_dense_mixed = 1;
    return 0;
}
#endif

ds4f_serve *ds4f_serve_open(const char *stage_dir, int use_hip, int hip_device,
                            int threads, int cmgs, long long max_pos,
                            char *err, size_t err_cap) {
    if (!stage_dir || !*stage_dir) {
        if (err && err_cap) snprintf(err, err_cap, "ds4f_serve_open: no stage dir");
        return NULL;
    }
    ds4f_runtime_options o = ds4f_runtime_options_debug_env(
        ds4f_default_config(), stage_dir, 0, 1, threads > 0 ? threads : 16,
        cmgs > 0 ? cmgs : 1);
    o.ep_rank = 0; o.ep_size = 1;                 /* single-node full load */
    o.mhc = env_i("DS4F_SERVE_MHC", 1);
    o.tierb2 = env_i("DS4F_SERVE_TIERB2", 1);
    o.exact = 1;                                  /* serve bundle */
    o.use_hip = use_hip ? 1 : 0; o.hip_device = hip_device;
    if (max_pos > 0) o.cfg.max_pos = (int)max_pos;
    ds4f_model *m = ds4f_load_real_opts(&o);
    if (!m) {
        if (err && err_cap) snprintf(err, err_cap, "ds4f_serve_open: ds4f_load_real_opts failed");
        return NULL;
    }
    if (m->emb_rows != m->cfg.vocab || m->head.rows != m->cfg.vocab) {
        if (err && err_cap) snprintf(err, err_cap,
            "ds4f_serve_open: single-node serve needs replicated embed/head");
        ds4f_free(m);
        return NULL;
    }
    ds4f_serve *s = (ds4f_serve *)calloc(1, sizeof(*s));
    if (!s) { ds4f_free(m); return NULL; }
    s->m = m;
    s->pool = m->mem;
    s->vocab = m->cfg.vocab;
    s->hidden = m->cfg.hidden;
    s->x = (float *)ds4f_mem_alloc(s->pool, (size_t)s->hidden * 4, 256, 0);
    if (!s->x) { ds4f_serve_close(s); return NULL; }
#if defined(DS4F_SERVE_HIP)
    if (use_hip) {
        if (serve_attach_hip(s, hip_device, env_i("DS4F_HIP_VERBOSE", 0),
                             env_i("DS4F_SERVE_ORDERED_LAYERS", m->cfg.n_layers)) != 0) {
            if (err && err_cap) snprintf(err, err_cap, "ds4f_serve_open: HIP attach failed");
            ds4f_serve_close(s);
            return NULL;
        }
    }
#endif
    m->want_full_logits = 1;
    s->eos = 1;
    s->rng = 0x9e3779b97f4a7c15ULL;
    if (err && err_cap) snprintf(err, err_cap, "ok");
    return s;
}

void ds4f_serve_close(ds4f_serve *s) {
    if (!s) return;
    if (s->m && getenv("DS4F_PROF") && atoi(getenv("DS4F_PROF")) != 0) {
        double acc = 0.0;
        for (int i = 0; i <= DS4F_P_COMM; ++i) acc += s->m->prof[i];
        for (int i = 0; i <= DS4F_P_COMM; ++i) {
            if (s->m->prof[i] > 1e-3)
                fprintf(stderr, "  %-9s %8.3f s %5.1f%%\n", ds4f_prof_names[i],
                        s->m->prof[i], acc > 0 ? 100.0 * s->m->prof[i] / acc : 0.0);
        }
        fprintf(stderr, "  %-9s %8.3f s (profiled)\n", "TOTAL", acc);
    }
#if defined(DS4F_SERVE_HIP)
    if (s->hip) hip_ds4f_dense_destroy(s->hip);
#endif
    if (s->m) ds4f_free(s->m);
    free(s->hist);
    free(s->logits);
    free(s);
}

int ds4f_serve_vocab(ds4f_serve *s) { return s ? s->vocab : 0; }
int ds4f_serve_maxpos(ds4f_serve *s) { return s && s->m ? s->m->cfg.max_pos : 0; }
int ds4f_serve_pos(ds4f_serve *s) { return s ? s->pos : 0; }
int ds4f_serve_eos(ds4f_serve *s) { return s ? s->eos : 1; }

/* Prefill `n` prompt tokens starting at absolute position pos0.  The tokens
 * must already lie within max_pos (the caller enforces truncation).  On
 * return, logits() holds the logits for the token AFTER the last prompt
 * token. */
int ds4f_serve_prefill(ds4f_serve *s, const int *ids, int n, int pos0) {
    if (!s || !ids || n < 1) return -1;
    int p = pos0;
    for (int i = 0; i < n; ++i, ++p) {
        if (ids[i] < 0 || ids[i] >= s->vocab) return -1;
        if (embed_lookup(s->m, ids[i], s->x) != 0) return -1;
        if (ds4f_forward_token(s->m, s->x, p) < 0) return -1;
    }
    s->pos = p;
    s->n_hist = 0;
    if (s->m->cfg.vocab > (int)s->logits_cap) {
        s->logits = (float *)realloc(s->logits, (size_t)s->m->cfg.vocab * sizeof(float));
        s->logits_cap = (size_t)s->m->cfg.vocab;
    }
    memcpy(s->logits, s->m->s_logits, (size_t)s->vocab * sizeof(float));
    return 0;
}

/* Decode one token (its id is `token`) at position `pos`.  Returns the argmax
 * of the resulting logits; logits() holds the full distribution. */
static double dc_wall(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

int ds4f_serve_decode(ds4f_serve *s, int token, int pos) {
    double tw0 = dc_wall();
    if (!s || token < 0 || token >= s->vocab) return -1;
    if (embed_lookup(s->m, token, s->x) != 0) return -1;
    int ar = ds4f_forward_token(s->m, s->x, pos);
    double tw1 = dc_wall();
    if (getenv("DS4F_SERVE_TIME") && (pos % 4) == 0)
        fprintf(stderr, "serve decode pos=%d forward=%.1f ms\n", pos, (tw1 - tw0) * 1e3);
    if (ar < 0) return -1;
    s->pos = pos + 1;
    if (s->n_hist >= s->hist_cap) {
        int cap = s->hist_cap ? s->hist_cap * 2 : 64;
        int *p = (int *)realloc(s->hist, (size_t)cap * sizeof(int));
        if (!p) return ar;
        s->hist = p; s->hist_cap = cap;
    }
    s->hist[s->n_hist++] = token;
    if (s->m->cfg.vocab > (int)s->logits_cap) {
        s->logits = (float *)realloc(s->logits, (size_t)s->m->cfg.vocab * sizeof(float));
        s->logits_cap = (size_t)s->m->cfg.vocab;
    }
    memcpy(s->logits, s->m->s_logits, (size_t)s->vocab * sizeof(float));
    return ar;
}

const float *ds4f_serve_logits(ds4f_serve *s, int *n) {
    if (!s) { if (n) *n = 0; return NULL; }
    if (n) *n = s->vocab;
    return s->logits;
}

/* Greedy or temperature/top-p/top-k sample over logits(). */
int ds4f_serve_sample(ds4f_serve *s, const ds4f_serve_sampling *sp) {
    double tw0 = dc_wall();
    if (!s || !s->logits || !sp) return -1;
    const float *lg = s->logits;
    int n = s->vocab;
    /* greedy */
    int greedy = 0; float bv = lg[0];
    for (int i = 1; i < n; ++i) if (lg[i] > bv) { bv = lg[i]; greedy = i; }
    double temp = sp->temperature > 0.001 ? sp->temperature : 0.0;
    if (temp <= 0.0) return greedy;
    double top_p = (sp->top_p > 0.0 && sp->top_p <= 1.0) ? sp->top_p : 1.0;
    int top_k = sp->top_k > 0 ? sp->top_k : n;
    double inv = 1.0 / temp;

    /* candidate list (id, logit) over the finite logits.  The penalties
     * (presence/repeat) adjust a copy of the const logits first. */
    float *adj = NULL;
    if (sp->presence_penalty > 0.0 || sp->repeat_penalty > 1.0) {
        adj = (float *)malloc((size_t)n * sizeof(float));
        if (!adj) return greedy;
        memcpy(adj, lg, (size_t)n * sizeof(float));
        for (int h = 0; h < s->n_hist; ++h) {
            int t = s->hist[h];
            if (t < 0 || t >= n) continue;
            if (sp->presence_penalty > 0.0) adj[t] -= (float)sp->presence_penalty;
            if (sp->repeat_penalty > 1.0 && adj[t] > 0.0) adj[t] /= (float)sp->repeat_penalty;
            else if (sp->repeat_penalty > 1.0) adj[t] *= (float)sp->repeat_penalty;
        }
        lg = adj;
    }
    int nc = 0;
    for (int i = 0; i < n; ++i) if (isfinite(lg[i])) nc++;
    if (nc < 1) { free(adj); return greedy; }
    float *sc = (float *)malloc((size_t)nc * 2 * sizeof(float));
    int *ids = (int *)malloc((size_t)nc * sizeof(int));
    if (!sc || !ids) { free(sc); free(ids); free(adj); return greedy; }
    nc = 0;
    for (int i = 0; i < n; ++i) if (isfinite(lg[i])) { sc[nc] = (float)(lg[i] * inv); ids[nc] = i; nc++; }
    if (adj) { free(adj); }

    /* sort by descending logit (qsort, O(n log n) -- the vocab is 129280, so an
     * insertion sort here was ~30 s/step) */
    for (int i = 0; i < nc; ++i) { float t = -sc[i]; sc[i] = t; }
    /* sort (logit,id) pairs ascending by negated logit == descending by logit */
    { ds4f_pair *pr = (ds4f_pair *)malloc((size_t)nc * sizeof(*pr));
      if (!pr) { free(sc); free(ids); return greedy; }
      for (int i = 0; i < nc; ++i) { pr[i].l = sc[i]; pr[i].id = ids[i]; }
      qsort(pr, (size_t)nc, sizeof(*pr), cmp_pair);
      for (int i = 0; i < nc; ++i) { sc[i] = pr[i].l; ids[i] = pr[i].id; }
      free(pr); }
    int keep = nc < top_k ? nc : top_k;
    float mx = sc[0];
    float total = 0.0f;
    for (int i = 0; i < keep; ++i) total += expf(sc[i] - mx);
    if (top_p < 1.0f) {
        float cum = 0.0f;
        for (int i = 0; i < keep; ++i) {
            cum += expf(sc[i] - mx);
            if (i > 0 && cum / (total + 1e-30f) >= top_p) { keep = i + 1; break; }
        }
        total = 0.0f;
        for (int i = 0; i < keep; ++i) total += expf(sc[i] - mx);
    }
    s->rng = s->rng * 6364136223846793005ULL + 1442695040888963407ULL;
    double draw = ((double)(s->rng >> 11) / (double)(1ULL << 53)) * total;
    int chosen = ids[keep - 1];
    for (int i = 0; i < keep; ++i) {
        draw -= expf(sc[i] - mx);
        if (draw <= 0.0) { chosen = ids[i]; break; }
    }
    free(sc); free(ids);
    if (getenv("DS4F_SERVE_TIME")) {
        double tw1 = dc_wall();
        fprintf(stderr, "serve sample %.1f ms\n", (tw1 - tw0) * 1e3);
    }
    return chosen;
}

int ds4f_serve_reset(ds4f_serve *s) {
    if (!s) return -1;
    s->pos = 0;
    s->n_hist = 0;
    return 0;
}

/* Prefix-cache snapshot: copy every layer's kv_cache rows [0,pos) plus the
 * compressor ring state to `path`.  restore() puts them back so a request that
 * extends the same prefix skips re-prefilling.  The file format is the raw
 * kv_cache bytes followed by the tierb2 compressor snapshot. */
int ds4f_serve_kv_save(ds4f_serve *s, const char *path) {
    if (!s || !s->m || !path) return -1;
    ds4f_model *m = s->m;
    size_t pos = (size_t)s->pos;
    size_t snap = 0;
    if (m->tierb2) snap = ds4f_tb2_snap_bytes(m);
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    size_t hdr[2] = { pos, snap };
    if (fwrite(hdr, sizeof(hdr), 1, f) != 1) { fclose(f); return -1; }
    for (int L = 0; L < m->cfg.n_layers; ++L) {
        ds4f_layer *ly = &m->layers[L];
        size_t rows = (size_t)ly->kv_slots;
        size_t n = rows * (size_t)m->cfg.kv_lora;
        if (fwrite(ly->kv_cache, 2, n, f) != n) { fclose(f); return -1; }
    }
    if (snap > 0) {
        char *buf = (char *)malloc(snap);
        if (buf) {
            ds4f_tb2_snap(m, buf, 0);
            if (fwrite(buf, 1, snap, f) != snap) { free(buf); fclose(f); return -1; }
        }
        free(buf);
    }
    fclose(f);
    return 0;
}

int ds4f_serve_kv_restore(ds4f_serve *s, const char *path) {
    if (!s || !s->m || !path) return -1;
    ds4f_model *m = s->m;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    size_t hdr[2];
    if (fread(hdr, sizeof(hdr), 1, f) != 1) { fclose(f); return -1; }
    size_t pos = hdr[0], snap = hdr[1];
    for (int L = 0; L < m->cfg.n_layers; ++L) {
        ds4f_layer *ly = &m->layers[L];
        size_t rows = (size_t)ly->kv_slots;
        size_t n = rows * (size_t)m->cfg.kv_lora;
        if (fread(ly->kv_cache, 2, n, f) != n) { fclose(f); return -1; }
    }
    if (snap > 0) {
        char *buf = (char *)malloc(snap);
        if (buf && fread(buf, 1, snap, f) == snap) ds4f_tb2_snap(m, buf, 1);
        free(buf);
    }
    fclose(f);
    s->pos = (int)pos;
    return 0;
}
