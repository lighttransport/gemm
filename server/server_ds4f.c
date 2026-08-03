#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "server_ds4f.h"

#if defined(DIFFUSION_SERVER_ENABLE_DS4F_HIP)
#include "../hetero/ds4f/hip_ds4f_dense.h"
#endif

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "../common/safetensors.h"
#include "../common/ds4f.h"

struct ds4f_session {
    ds4f_model *m;
    ds4f_mem_pool *mem;
    ds4f_mem_pool *cache_mem;
    char stage_dir[1024];
    char tokenizer[1024];
    char tokenizer_py[1024];
    int max_pos;
    ds4f_runtime_options options;

    /* One exact prompt snapshot. It stores only KV rows up to cache_len, not
     * the 156 GB model. Tier-B2 has additional compressor/indexer state and is
     * rejected by open() until those buffers are included here. */
    int *cache_ids;
    int cache_len;
    int cache_next;
    uint16_t **cache_kv;
    float *cache_x4;
    float *cache_logits;
    int cache_hc;
#if defined(DIFFUSION_SERVER_ENABLE_DS4F_HIP)
    hip_ds4f_dense *gpu;
#endif
};

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
    ds4f_mem_pool *pool;
} ds4f_sbuf;

static void sb_init_pool(ds4f_sbuf *b, ds4f_mem_pool *pool) {
    memset(b, 0, sizeof(*b));
    b->pool = pool;
    b->cap = 4096;
    b->ptr = pool ? (char *)ds4f_mem_alloc(pool, b->cap, 64, 0)
                  : (char *)ds4f_map_alloc(b->cap, 64, 0);
    if (b->ptr) b->ptr[0] = 0;
}

static void sb_init(ds4f_sbuf *b) { sb_init_pool(b, NULL); }

static void sb_free(ds4f_sbuf *b) {
    if (b->ptr && !b->pool) ds4f_map_free(b->ptr);
    memset(b, 0, sizeof(*b));
}

static int sb_reserve(ds4f_sbuf *b, size_t need) {
    if (need <= b->cap) return 0;
    size_t nc = b->cap ? b->cap : 4096;
    while (nc < need) nc *= 2;
    char *p;
    if (b->pool) {
        p = (char *)ds4f_mem_alloc(b->pool, nc, 64, 0);
        if (p && b->ptr) memcpy(p, b->ptr, b->len + 1);
    } else {
        p = (char *)ds4f_map_realloc(b->ptr, b->cap, nc, 64);
    }
    if (!p) return -1;
    b->ptr = p;
    b->cap = nc;
    return 0;
}

static int sb_appendn(ds4f_sbuf *b, const char *s, size_t n) {
    if (sb_reserve(b, b->len + n + 1) != 0) return -1;
    memcpy(b->ptr + b->len, s, n);
    b->len += n;
    b->ptr[b->len] = 0;
    return 0;
}

static int sb_append(ds4f_sbuf *b, const char *s) {
    return sb_appendn(b, s, strlen(s));
}

static int sb_printf(ds4f_sbuf *b, const char *fmt, ...) {
    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0 || sb_reserve(b, b->len + (size_t)n + 1) != 0) {
        va_end(ap2);
        return -1;
    }
    vsnprintf(b->ptr + b->len, (size_t)n + 1, fmt, ap2);
    va_end(ap2);
    b->len += (size_t)n;
    return 0;
}

static const char *j_str(const json_val *obj, const char *key, const char *def) {
    json_val *v = json_obj_get(obj, key);
    if (!v || v->type != JSON_STRING || !v->str.ptr) return def;
    return v->str.ptr;
}

static int env_i(const char *name, int def) {
    const char *v = getenv(name);
    return v && *v ? atoi(v) : def;
}

static void set_err(char *err, size_t cap, const char *msg) {
    if (err && cap) snprintf(err, cap, "%s", msg ? msg : "DS4F error");
}

static int write_all_fd(int fd, const char *p, size_t n) {
    while (n) {
        ssize_t w = write(fd, p, n);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) return -1;
        p += w;
        n -= (size_t)w;
    }
    return 0;
}

/* Execute the stdlib-only tokenizer without a shell. Prompt text is passed via
 * a pipe, so request content never becomes part of a command line. */
static int tokenizer_run(const ds4f_session *s, const char *cmd,
                         const char *input, char **out, size_t *out_len) {
    int inpipe[2], outpipe[2];
    if (pipe(inpipe) != 0 || pipe(outpipe) != 0) return -1;
    pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        char tokarg[1100];
        snprintf(tokarg, sizeof(tokarg), "%s", s->tokenizer);
        dup2(inpipe[0], STDIN_FILENO);
        dup2(outpipe[1], STDOUT_FILENO);
        close(inpipe[0]); close(inpipe[1]);
        close(outpipe[0]); close(outpipe[1]);
        execlp("python3", "python3", s->tokenizer_py, cmd,
               "--tokenizer", tokarg, NULL);
        _exit(127);
    }
    close(inpipe[0]);
    close(outpipe[1]);
    if (input && write_all_fd(inpipe[1], input, strlen(input)) != 0) {
        close(inpipe[1]); close(outpipe[0]); kill(pid, SIGTERM); waitpid(pid, NULL, 0); return -1;
    }
    close(inpipe[1]);
    size_t cap = 4096, len = 0;
    char *buf = (char *)ds4f_map_alloc(cap, 64, 0);
    if (!buf) { close(outpipe[0]); waitpid(pid, NULL, 0); return -1; }
    for (;;) {
        char tmp[4096];
        ssize_t n = read(outpipe[0], tmp, sizeof(tmp));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break;
        if (len + (size_t)n + 1 > cap) {
            while (len + (size_t)n + 1 > cap) cap *= 2;
            char *p = (char *)ds4f_map_realloc(buf, len + 1, cap, 64);
            if (!p) { ds4f_map_free(buf); close(outpipe[0]); waitpid(pid, NULL, 0); return -1; }
            buf = p;
        }
        memcpy(buf + len, tmp, (size_t)n);
        len += (size_t)n;
    }
    close(outpipe[0]);
    int status = 0;
    if (waitpid(pid, &status, 0) < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        ds4f_map_free(buf);
        return -1;
    }
    buf[len] = 0;
    *out = buf;
    if (out_len) *out_len = len;
    return 0;
}

static int parse_ids(ds4f_mem_pool *pool, char *text, int **ids_out, int *n_out) {
    int cap = 256, n = 0;
    int *ids = (int *)ds4f_mem_alloc(pool, (size_t)cap * sizeof(*ids), 64, 0);
    if (!ids) return -1;
    char *p = text;
    while (*p) {
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p) { p++; continue; }
        if (n == cap) {
            cap *= 2;
            int *q = (int *)ds4f_mem_alloc(pool, (size_t)cap * sizeof(*ids), 64, 0);
            if (!q) return -1;
            memcpy(q, ids, (size_t)n * sizeof(*ids));
            ids = q;
        }
        ids[n++] = (int)v;
        p = end;
    }
    if (!n) return -1;
    *ids_out = ids;
    *n_out = n;
    return 0;
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

static void cache_free(ds4f_session *s) {
    if (!s) return;
    ds4f_mem_pool_destroy(s->cache_mem);
    s->cache_mem = NULL;
    s->cache_kv = NULL; s->cache_ids = NULL; s->cache_x4 = NULL;
    s->cache_logits = NULL; s->cache_len = 0;
}

static int cache_alloc(ds4f_session *s, int n) {
    cache_free(s);
    s->cache_mem = ds4f_mem_pool_create();
    if (!s->cache_mem) return -1;
    int Ls = s->m->cfg.n_layers, KV = s->m->cfg.kv_lora;
    s->cache_ids = (int *)ds4f_mem_alloc(s->cache_mem, (size_t)n * sizeof(int), 64, 0);
    s->cache_kv = (uint16_t **)ds4f_mem_calloc(s->cache_mem, (size_t)Ls,
                                               sizeof(*s->cache_kv), 64);
    if (!s->cache_ids || !s->cache_kv) return -1;
    for (int L = 0; L < Ls; L++) {
        s->cache_kv[L] = (uint16_t *)ds4f_mem_alloc(s->cache_mem,
                                                     (size_t)n * KV * sizeof(uint16_t),
                                                     256, 0);
        if (!s->cache_kv[L]) return -1;
    }
    if (s->m->mhc) {
        s->cache_x4 = (float *)ds4f_mem_alloc(s->cache_mem,
                                               (size_t)s->m->cfg.hc_mult * s->m->cfg.hidden * sizeof(float),
                                               256, 0);
        if (!s->cache_x4) return -1;
    }
    s->cache_logits = (float *)ds4f_mem_alloc(s->cache_mem,
                                               (size_t)s->m->cfg.vocab * sizeof(float),
                                               256, 0);
    if (!s->cache_logits) return -1;
    return 0;
}

static int cache_save(ds4f_session *s, const int *ids, int n, int next) {
    if (n <= 0 || n > s->max_pos || cache_alloc(s, n) != 0) return -1;
    int KV = s->m->cfg.kv_lora;
    memcpy(s->cache_ids, ids, (size_t)n * sizeof(int));
    for (int L = 0; L < s->m->cfg.n_layers; L++) {
        ds4f_layer *ly = &s->m->layers[L];
        for (int p = 0; p < n; p++)
            memcpy(s->cache_kv[L] + (size_t)p * KV,
                   ly->kv_cache + (size_t)p * KV, (size_t)KV * sizeof(uint16_t));
    }
    if (s->m->mhc)
        memcpy(s->cache_x4, s->m->s_x4,
               (size_t)s->m->cfg.hc_mult * s->m->cfg.hidden * sizeof(float));
    memcpy(s->cache_logits, s->m->s_logits,
           (size_t)s->m->cfg.vocab * sizeof(float));
    s->cache_len = n;
    s->cache_next = next;
    return 0;
}

static int cache_restore(ds4f_session *s, const int *ids, int n, int *pos, int *next) {
    if (!s->cache_len || !s->cache_logits || n < s->cache_len) return 0;
    for (int i = 0; i < s->cache_len; i++) if (ids[i] != s->cache_ids[i]) return 0;
    int KV = s->m->cfg.kv_lora;
    for (int L = 0; L < s->m->cfg.n_layers; L++)
        memcpy(s->m->layers[L].kv_cache, s->cache_kv[L],
               (size_t)s->cache_len * KV * sizeof(uint16_t));
    if (s->m->mhc)
        memcpy(s->m->s_x4, s->cache_x4,
               (size_t)s->m->cfg.hc_mult * s->m->cfg.hidden * sizeof(float));
    memcpy(s->m->s_logits, s->cache_logits,
           (size_t)s->m->cfg.vocab * sizeof(float));
    *pos = s->cache_len;
    *next = s->cache_next;
    return 1;
}

typedef struct {
    int id;
    float logit;
} ds4f_candidate;

static int candidate_cmp(const void *a, const void *b) {
    const ds4f_candidate *x = (const ds4f_candidate *)a;
    const ds4f_candidate *y = (const ds4f_candidate *)b;
    if (x->logit > y->logit) return -1;
    if (x->logit < y->logit) return 1;
    return x->id - y->id;
}

static uint64_t rng_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static float rng_unit(uint64_t *state) {
    return (float)((rng_next(state) >> 40) * (1.0 / 16777216.0));
}

static int logits_argmax(const float *logits, int n) {
    int best = -1;
    float bv = -INFINITY;
    for (int i = 0; i < n; i++) {
        if (isfinite(logits[i]) && (best < 0 || logits[i] > bv)) {
            best = i;
            bv = logits[i];
        }
    }
    return best;
}

static int sample_logits(const float *logits, int n, float temperature,
                         float top_p, uint64_t *rng, ds4f_mem_pool *pool) {
    int greedy = logits_argmax(logits, n);
    if (greedy < 0 || temperature <= 0.001f) return greedy;
    if (top_p <= 0.0f || top_p > 1.0f) top_p = 1.0f;

    ds4f_candidate *c = (ds4f_candidate *)ds4f_mem_alloc(pool,
                                                           (size_t)n * sizeof(*c), 64, 0);
    if (!c) return greedy;
    int nc = 0;
    float inv_temp = 1.0f / temperature;
    for (int i = 0; i < n; i++) {
        if (isfinite(logits[i])) {
            c[nc++] = (ds4f_candidate){ i, logits[i] * inv_temp };
        }
    }
    if (!nc) return greedy;
    qsort(c, (size_t)nc, sizeof(*c), candidate_cmp);

    float max_logit = c[0].logit;
    int keep = nc;
    float total = 0.0f;
    for (int i = 0; i < nc; i++) total += expf(c[i].logit - max_logit);
    if (top_p < 1.0f) {
        float cumulative = 0.0f;
        for (int i = 0; i < nc; i++) {
            cumulative += expf(c[i].logit - max_logit);
            if (i > 0 && cumulative / (total + 1e-30f) >= top_p) {
                keep = i + 1;
                break;
            }
        }
    }
    float z = 0.0f;
    for (int i = 0; i < keep; i++) z += expf(c[i].logit - max_logit);
    float draw = rng_unit(rng) * z;
    int chosen = c[keep - 1].id;
    for (int i = 0; i < keep; i++) {
        draw -= expf(c[i].logit - max_logit);
        if (draw <= 0.0f) { chosen = c[i].id; break; }
    }
    return chosen;
}

#if defined(DIFFUSION_SERVER_ENABLE_DS4F_HIP)
static void gpu_detach(ds4f_session *s) {
    if (!s) return;
    if (s->m) {
        s->m->gpu_dense_ctx = NULL;
        s->m->gpu_dense_matvec = NULL;
        s->m->gpu_dense_async_multi = NULL;
        s->m->gpu_dense_wait = NULL;
        s->m->gpu_dense_blockdiag = NULL;
        s->m->gpu_dense_gemm = NULL;
        s->m->gpu_dense_gemm_multi = NULL;
        s->m->gpu_dense_layer_begin = NULL;
        s->m->gpu_dense_stream_prefill_only = 0;
        s->m->gpu_dense_mixed = 0;
    }
    if (s->gpu) hip_ds4f_dense_destroy(s->gpu);
    s->gpu = NULL;
}

static int gpu_bind_layer(ds4f_session *s, ds4f_layer *ly, int layer,
                          size_t *bytes, int *count) {
    int hot_fp16_layer = s->options.hip_shared_fp16 &&
        (s->options.hip_shared_fp16_layers <= 0 ||
         layer < s->options.hip_shared_fp16_layers);
    int hot_bf16_layer = s->options.hip_shared_bf16 &&
        (s->options.hip_shared_bf16_layers <= 0 ||
         layer < s->options.hip_shared_bf16_layers);
    int ordered_wkv = s->options.hip_ordered_wkv_layers > 0 &&
        layer < s->options.hip_ordered_wkv_layers;
    int ordered_fp8 = s->options.hip_ordered_fp8_layers > 0 &&
        layer < s->options.hip_ordered_fp8_layers;
    ds4f_tensor *t[] = {
        &ly->wq_a, &ly->wq_b, &ly->wkv, &ly->wo_a, &ly->wo_b,
        &ly->sh_w1, &ly->sh_w3, &ly->sh_w2, &ly->gate
    };
    for (size_t i = 0; i < sizeof(t) / sizeof(t[0]); i++) {
        if (i == 8 && !hot_bf16_layer)
            continue;
        if (!t[i]->w || (t[i]->type != DS4F_FP8 && t[i]->type != DS4F_BF16)) return -1;
        int shared_fp16 = hot_fp16_layer && i >= 5 &&
                          t[i]->type == DS4F_FP8;
        int shared_bf16 = !shared_fp16 && hot_bf16_layer && i >= 5 &&
                          t[i]->type == DS4F_FP8;
        int id = t[i]->type == DS4F_FP8
            ? ((ordered_wkv && i == 2) || (ordered_fp8 && !shared_fp16 && !shared_bf16)
                ? hip_ds4f_dense_bind_fp8_ordered_tensor(s->gpu, t[i])
                : shared_fp16
                ? hip_ds4f_dense_bind_fp8_fp16_tensor(s->gpu, t[i])
                : shared_bf16
                ? hip_ds4f_dense_bind_fp8_bf16_tensor(s->gpu, t[i])
                : hip_ds4f_dense_bind_tensor(s->gpu, t[i]))
            : hip_ds4f_dense_bind_bf16_tensor(s->gpu, t[i]);
        if (id < 0) return -1;
        *bytes += (shared_fp16 || shared_bf16)
            ? (size_t)t[i]->rows * (size_t)t[i]->cols * sizeof(uint16_t)
            : ds4f_wbytes(t[i]->type, t[i]->rows, t[i]->cols)
                + ds4f_sbytes(t[i]->type, t[i]->rows, t[i]->cols);
        (*count)++;
    }
    return 0;
}

static int gpu_attach(ds4f_session *s, char *err, size_t err_cap) {
    if (!s->options.use_hip) return 0;
    if (s->m->dense_qt != DS4F_FP8 && s->m->dense_qt != DS4F_BF16) {
        set_err(err, err_cap, "DS4F_HIP requires flat FP8 or BF16 dense weights (disable BF16_PV/Q8_DENSE)");
        return -1;
    }
    int device = s->options.hip_device;
    int verbose = s->options.hip_verbose;
    s->gpu = hip_ds4f_dense_create_ex(device, verbose,
        s->options.hip_ordered_fp8_layers > 0 || s->options.hip_ordered_wkv_layers > 0);
    if (!s->gpu) {
        set_err(err, err_cap, "DS4F_HIP requested but HIPRTC dense-bank initialization failed");
        return -1;
    }
    size_t bytes = 0;
    int count = 0;
    for (int L = 0; L < s->m->cfg.n_layers; L++) {
        if (gpu_bind_layer(s, &s->m->layers[L], L, &bytes, &count) != 0) {
            set_err(err, err_cap, "DS4F_HIP bank requires every MLA/shared tensor to be FP8 with E8M0 scales");
            gpu_detach(s);
            return -1;
        }
    }
    if (s->m->head.type != DS4F_BF16 || !s->m->head.w ||
        hip_ds4f_dense_bind_bf16_tensor(s->gpu, &s->m->head) < 0) {
        set_err(err, err_cap, "DS4F_HIP requires the replicated vocabulary head in flat BF16 layout");
        gpu_detach(s);
        return -1;
    }
    bytes += ds4f_wbytes(s->m->head.type, s->m->head.rows, s->m->head.cols);
    count++;
    if (s->options.hip_mxfp4_resident_layers > 0) {
        int nr = s->options.hip_mxfp4_resident_layers;
        if (nr > s->m->cfg.n_layers) nr = s->m->cfg.n_layers;
        for (int L = 0; L < nr; ++L)
            if (hip_ds4f_dense_resident_mxfp4_layer(
                    s->gpu, &s->m->layers[L], s->options.hip_mxfp4_stream_raw) != 0) {
                set_err(err, err_cap, "DS4F_HIP resident MXFP4 upload exceeded GPU memory");
                gpu_detach(s);
                return -1;
            }
    }
    s->m->gpu_dense_ctx = s->gpu;
    s->m->gpu_dense_matvec = hip_ds4f_dense_matvec_tensor;
    s->m->gpu_dense_async_multi = s->options.hip_async
        ? hip_ds4f_dense_matvec_tensors_async : NULL;
    s->m->gpu_dense_wait = hip_ds4f_dense_wait_tensors;
    s->m->gpu_dense_blockdiag = hip_ds4f_dense_matvec_blockdiag;
    s->m->gpu_dense_gemm = hip_ds4f_dense_gemm_tensor;
    s->m->gpu_dense_gemm_multi = hip_ds4f_dense_gemm_tensors;
    if (s->options.hip_mxfp4_widen_layers > 0 ||
        s->options.hip_mxfp4_resident_layers > 0)
        s->m->mxfp4_w4a8 = 0;
    s->m->gpu_dense_layer_begin = (s->options.hip_mxfp4_widen_layers > 0 ||
                                   s->options.hip_mxfp4_resident_layers > 0)
        ? (s->options.hip_mxfp4_stream_raw
            ? hip_ds4f_dense_stream_layer_raw : hip_ds4f_dense_stream_layer) : NULL;
    s->m->gpu_dense_stream_prefill_only = s->options.hip_mxfp4_widen_layers > 0 ||
                                          s->options.hip_mxfp4_resident_layers > 0;
    s->m->gpu_dense_mixed = s->options.hip_shared_bf16 || s->options.hip_shared_fp16;
    fprintf(stderr, "[llm/ds4f] HIP dense bank attached: %d matrices, %.3f GB, device=%d\n",
            count, (double)bytes / 1e9, device);
    return 0;
}
#endif

ds4f_session *ds4f_session_open_opts(const char *stage_dir,
                                     const ds4f_runtime_options *options,
                                     char *err, size_t err_cap) {
    if (!stage_dir || !*stage_dir) {
        set_err(err, err_cap, "DS4F needs a staged safetensors directory (DS4F_STAGE_DIR)");
        return NULL;
    }
    ds4f_mem_pool *session_mem = ds4f_mem_pool_create();
    ds4f_session *s = session_mem
        ? (ds4f_session *)ds4f_mem_calloc(session_mem, 1, sizeof(*s), 64) : NULL;
    if (!s) {
        ds4f_mem_pool_destroy(session_mem);
        set_err(err, err_cap, "out of memory");
        return NULL;
    }
    s->mem = session_mem;
    snprintf(s->stage_dir, sizeof(s->stage_dir), "%s", stage_dir);
    if (options) s->options = *options;
    else ds4f_runtime_options_init(&s->options);
    snprintf(s->options.stage_dir, sizeof(s->options.stage_dir), "%s", stage_dir);
    if (s->options.tokenizer[0]) snprintf(s->tokenizer, sizeof(s->tokenizer), "%s", s->options.tokenizer);
    else snprintf(s->tokenizer, sizeof(s->tokenizer), "%s/tokenizer.json", stage_dir);
    snprintf(s->tokenizer_py, sizeof(s->tokenizer_py), "%s",
             s->options.tokenizer_py[0] ? s->options.tokenizer_py : "a64fx/llm/tools/ds4f_tokenizer.py");
    /* Leave room for a normal prompt plus a 4k-token completion by default.
     * DS4F_MAXPOS still overrides this when a smaller footprint is desired. */
    s->max_pos = s->options.cfg.max_pos;
    if (s->max_pos < 1) s->max_pos = 1;

    ds4f_config cfg = s->options.cfg;
    cfg.max_pos = s->max_pos;
    if (cfg.max_pos < 1) cfg.max_pos = 1;
    if (s->options.tierb2 || s->options.int8_kv || s->options.int8_cmp) {
        set_err(err, err_cap, "native DS4F server cache currently requires Tier-B1 bf16 KV (disable DS4F_TIERB2/INT8_KV/INT8_CMP)");
        ds4f_mem_pool_destroy(s->mem);
        return NULL;
    }
    s->m = ds4f_load_real_opts(&s->options);
    if (!s->m) {
        set_err(err, err_cap, "ds4f_load_real failed");
        ds4f_mem_pool_destroy(s->mem);
        return NULL;
    }
    if (s->m->emb_rows != cfg.vocab || s->m->head.rows != cfg.vocab) {
        set_err(err, err_cap, "native DS4F server currently requires replicated embedding and head (DS4F_TP_EMBED=0, DS4F_TP_HEAD=0)");
        ds4f_free(s->m); ds4f_mem_pool_destroy(s->mem); return NULL;
    }
#if defined(DIFFUSION_SERVER_ENABLE_DS4F_HIP)
    if (gpu_attach(s, err, err_cap) != 0) {
        ds4f_free(s->m); ds4f_mem_pool_destroy(s->mem); return NULL;
    }
#endif
    return s;
}

ds4f_session *ds4f_session_open(const char *stage_dir, char *err, size_t err_cap) {
    ds4f_runtime_options o = ds4f_runtime_options_debug_env(ds4f_default_config(), stage_dir,
        env_i("DS4F_EP_RANK", 0), env_i("DS4F_EP_SIZE", 1),
        env_i("LLM_THREADS", 16), env_i("DS4F_CMGS", 1));
    o.cfg.max_pos = env_i("DS4F_MAXPOS", 8192);
    o.use_hip = env_i("DS4F_HIP", 0); o.hip_device = env_i("DS4F_HIP_DEVICE", 0);
    o.hip_verbose = env_i("DS4F_HIP_VERBOSE", 0); o.hip_async = env_i("DS4F_HIP_ASYNC", 1);
    return ds4f_session_open_opts(stage_dir, &o, err, err_cap);
}

void ds4f_session_close(ds4f_session *s) {
    if (!s) return;
#if defined(DIFFUSION_SERVER_ENABLE_DS4F_HIP)
    gpu_detach(s);
#endif
    cache_free(s);
    ds4f_free(s->m);
    ds4f_mem_pool_destroy(s->mem);
}

void ds4f_owned_free(void *ptr) { ds4f_map_free(ptr); }

char *ds4f_chat_prompt(const json_val *messages, char *err, size_t err_cap) {
    if (!messages || messages->type != JSON_ARRAY || messages->arr.count == 0) {
        set_err(err, err_cap, "messages must be a non-empty array");
        return NULL;
    }
    ds4f_sbuf b; sb_init(&b);
    if (!b.ptr) { set_err(err, err_cap, "out of memory"); return NULL; }
    sb_append(&b, "<｜begin▁of▁sentence｜>");
    for (int i = 0; i < messages->arr.count; i++) {
        json_val *msg = &messages->arr.items[i];
        if (msg->type != JSON_OBJECT) continue;
        const char *role = j_str(msg, "role", "user");
        const char *label = strcmp(role, "system") == 0 ? "System" :
                            strcmp(role, "assistant") == 0 ? "Assistant" :
                            strcmp(role, "tool") == 0 ? "Tool" : "User";
        sb_printf(&b, "%s: ", label);
        json_val *content = json_obj_get(msg, "content");
        if (content && content->type == JSON_STRING && content->str.ptr) {
            sb_append(&b, content->str.ptr);
        } else if (content && content->type == JSON_ARRAY) {
            for (int j = 0; j < content->arr.count; j++) {
                json_val *part = &content->arr.items[j];
                if (part->type == JSON_OBJECT && strcmp(j_str(part, "type", ""), "text") == 0)
                    sb_append(&b, j_str(part, "text", ""));
            }
        }
        sb_append(&b, "\n");
    }
    sb_append(&b, "Assistant:");
    return b.ptr;
}

char *ds4f_session_generate(ds4f_session *s, const char *prompt, int max_tokens,
                            float temperature, float top_p, int seed,
                            int *prompt_tokens, int *completion_tokens,
                            char *err, size_t err_cap) {
    if (!s || !s->m) { set_err(err, err_cap, "DS4F session is not loaded"); return NULL; }
    if (!prompt || !*prompt) { set_err(err, err_cap, "empty prompt"); return NULL; }
    if (!isfinite(temperature) || temperature < 0.0f ||
        !isfinite(top_p) || top_p < 0.0f || top_p > 1.0f) {
        set_err(err, err_cap, "invalid temperature/top_p");
        return NULL;
    }
    if (max_tokens < 0) max_tokens = 0;
    if (max_tokens > s->max_pos) max_tokens = s->max_pos;
    ds4f_mem_pool *req = ds4f_mem_pool_create();
    if (!req) { set_err(err, err_cap, "out of memory"); return NULL; }
    char *encoded = NULL; size_t encoded_len = 0;
    if (tokenizer_run(s, "encode", prompt, &encoded, &encoded_len) != 0 || !encoded) {
        ds4f_mem_pool_destroy(req);
        set_err(err, err_cap, "DS4F tokenizer encode failed"); return NULL;
    }
    int *ids = NULL, nids = 0;
    if (parse_ids(req, encoded, &ids, &nids) != 0) {
        ds4f_map_free(encoded); ds4f_mem_pool_destroy(req);
        set_err(err, err_cap, "DS4F tokenizer returned no token ids"); return NULL;
    }
    ds4f_map_free(encoded); (void)encoded_len;
    if (nids > s->max_pos) {
        ds4f_mem_pool_destroy(req);
        set_err(err, err_cap, "prompt exceeds DS4F_MAXPOS"); return NULL;
    }
    int room = s->max_pos - nids;
    if (max_tokens > room) max_tokens = room;

    float *x = (float *)ds4f_mem_alloc(req, (size_t)s->m->cfg.hidden * sizeof(float), 256, 0);
    if (!x) { ds4f_mem_pool_destroy(req); set_err(err, err_cap, "out of memory"); return NULL; }
    int pos = 0, next = -1;
    int hit = cache_restore(s, ids, nids, &pos, &next);
    for (int p = hit ? pos : 0; p < nids; p++) {
        if (embed_lookup(s->m, ids[p], x) != 0) {
            ds4f_mem_pool_destroy(req);
            set_err(err, err_cap, "prompt token has no local embedding"); return NULL;
        }
        next = ds4f_forward_token(s->m, x, p);
        pos = p + 1;
    }
    if (!hit && nids > 0) pos = nids;
    if (nids > 0 && cache_save(s, ids, nids, next) != 0) {
        /* Cache is an optimization; generation remains valid without it. */
        cache_free(s);
    }

    int *gen = (int *)ds4f_mem_alloc(req,
                                     (size_t)(max_tokens ? max_tokens : 1) * sizeof(int),
                                     64, 0);
    if (!gen) { ds4f_mem_pool_destroy(req); set_err(err, err_cap, "out of memory"); return NULL; }
    uint64_t rng = (uint64_t)(uint32_t)seed;
    if (!rng) rng = 0x9e3779b97f4a7c15ULL;
    next = sample_logits(s->m->s_logits, s->m->cfg.vocab, temperature, top_p, &rng, req);
    int ng = 0;
    while (ng < max_tokens && next >= 0) {
        gen[ng++] = next;
        if (next == 1) break;
        if (pos >= s->max_pos) break;
        if (embed_lookup(s->m, next, x) != 0) break;
        next = ds4f_forward_token(s->m, x, pos++);
    }

    ds4f_sbuf in; sb_init_pool(&in, req);
    for (int i = 0; i < ng; i++) sb_printf(&in, "%d%s", gen[i], i + 1 < ng ? " " : "\n");
    char *decoded = NULL; size_t decoded_len = 0;
    if (!in.ptr || tokenizer_run(s, "decode", in.ptr, &decoded, &decoded_len) != 0) {
        sb_free(&in); ds4f_mem_pool_destroy(req);
        set_err(err, err_cap, "DS4F tokenizer decode failed"); return NULL;
    }
    (void)decoded_len;
    sb_free(&in); ds4f_mem_pool_destroy(req);
    if (prompt_tokens) *prompt_tokens = nids;
    if (completion_tokens) *completion_tokens = ng;
    return decoded;
}
