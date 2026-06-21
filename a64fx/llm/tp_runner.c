/*
 * tp_runner - MPI-free uTofu TENSOR-PARALLEL LLM decode on A64FX/Fugaku.
 *
 * Megatron-style tensor parallelism: every weight matrix is sliced 1/N across N
 * nodes (transformer_tp_slice_weights). Each rank holds only its 1/N shard of
 * attn_q/k/v/output + ffn_gate/up/down (SSM left replicated in Stage A), so the
 * aggregate weight memory is spread across nodes AND the per-token matvec FLOPs
 * are divided by N -- the path to the HBM-bandwidth decode roofline that a single
 * node (or the serial pipeline) can't reach.
 *
 * Dataflow per token: ALL ranks run the SAME full forward in lockstep. m->x stays
 * fully REPLICATED on every rank; the two row-parallel projections (attn_output,
 * ffn_down) produce partial sums that an in-layer uTofu sum-all-reduce recombines
 * (tp_allreduce.h, recursive-doubling). Because every rank ends each layer with
 * the identical reduced x, every rank computes identical logits and independently
 * argmaxes the SAME next token -- no token broadcast, no stop broadcast needed;
 * the all-reduces themselves keep the ranks in lockstep. rank 0 just prints.
 *
 * Comm is pure uTofu, ZERO MPI (mpiexec only places ranks). Peer VCQ IDs are
 * reconstructed from tofu_topo.txt coordinates (tofu_topo_helper writes it once),
 * the convention shared with a64fx/utofu-tests and pp_runner.
 *
 * NOTE: decode forward goes through transformer_forward_partial. For the full
 * layer range (0..n_layers) with a live pool it now routes to the PERSISTENT
 * worker (tf_forward_persistent): ONE dispatch/token + parallel SSM, carrying
 * the same tid-0 all-reduce hooks (mixer-out, ffn-down) as the per-op block
 * loop -- byte-identical, lockstep-argmax safe. Set TP_DECODE_PERSIST=0 to fall
 * back to the per-op path (tf_forward_blocks_range) for A/B. PP partial ranges
 * still use the per-op loop (only path supporting partial layer spans).
 *
 * Build:  make -C a64fx/llm CC=fcc OPENMP=1 tp_runner
 * Run (after tofu_topo_helper writes tofu_topo.txt, 1 proc/node):
 *   GGUF_LAZY_MMAP=1 LLM_THREADS=48 TP_PROMPT="Hello" TP_MAXGEN=64 \
 *     mpiexec -np 2 ./build/tp_runner ~/models/qwen35/9b/Qwen3.5-9B-BF16.gguf
 *   stdout is swallowed by mpiexec -> rank 0 also writes tp_run_<coords>.txt.
 *
 * Env:  TP_PROMPT / TP_MAXGEN / TP_MAXSEQ / LLM_THREADS (see pp_runner).
 *       TF_KEEP_BF16_SRC is force-set here (required: the bf16_pv reclaim assumes
 *       a contiguous source range, wrong for row-parallel strided slices).
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <strings.h>
#include <errno.h>
#include <limits.h>
#include <sys/stat.h>
#include <utofu.h>

#define GGUF_LOADER_IMPLEMENTATION
#include "../../common/gguf_loader.h"
#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"
#define BPE_TOKENIZER_IMPLEMENTATION
#include "../../common/bpe_tokenizer.h"
#define TRANSFORMER_IMPLEMENTATION
#include "../../common/transformer.h"

#include "../utofu-tests/tofu_demo.h"
#include "../utofu-tests/tp_allreduce.h"

#define MAX_NODES 32
#define RUN_STAG  DEMO_STAG
#define WAIT_TIMEOUT_SEC 300.0   /* tolerate cold-load skew across ranks at startup */

/* ---- logging / time ---- */
static FILE *g_log = NULL;
static FILE *g_curve = NULL;   /* rank0 per-token decode-cost-vs-context curve */
static FILE *g_tokdump = NULL; /* rank0 generated-token-id log (TP_DUMP_TOKENS=1) for A/B parity */
static void logmsg(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    if (g_log) { va_list ap2; va_copy(ap2, ap); vfprintf(g_log, fmt, ap2); fflush(g_log); va_end(ap2); }
    vfprintf(stdout, fmt, ap); fflush(stdout);
    va_end(ap);
}
static void die(const char *what, int rc) { logmsg("FATAL: %s (rc=%d)\n", what, rc); exit(1); }
static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
/* Internal helper declarations from transformer.h (not part of public API, but
 * available in this TU because IMPLEMENTATION is enabled above). */
static size_t tf_runtime_kv_len_for_layer(const transformer_model *model, int layer);
static int tf_runtime_kv_dim_for_layer(const transformer_model *model, int layer);
static const char *envs(const char *n, const char *d) { const char *v = getenv(n); return (v && *v) ? v : d; }
static long envl(const char *n, long d) { const char *v = getenv(n); return (v && *v) ? strtol(v, NULL, 0) : d; }
static int envb(const char *n, int d) {
    const char *v = getenv(n);
    if (!v || !*v) return d;
    if (!strcasecmp(v, "1") || !strcasecmp(v, "true") || !strcasecmp(v, "yes") || !strcasecmp(v, "on"))
        return 1;
    if (!strcasecmp(v, "0") || !strcasecmp(v, "false") || !strcasecmp(v, "no") || !strcasecmp(v, "off"))
        return 0;
    long x = strtol(v, NULL, 0);
    return x > 0 ? 1 : 0;
}

static char *read_text_file(const char *path, size_t *n_out) {
    if (!path || !*path) return NULL;
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long n = ftell(f);
    if (n < 0) { fclose(f); return NULL; }
    rewind(f);
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    if (got != (size_t)n) { free(buf); return NULL; }
    buf[got] = 0;
    if (n_out) *n_out = got;
    return buf;
}

static const char *envs_opt(const char *n, const char *d) {
    const char *v = getenv(n);
    return (v && *v) ? v : d;
}

static long envl_opt(const char *n, long d) {
    const char *v = getenv(n);
    return (v && *v) ? strtol(v, NULL, 0) : d;
}

static int envb_opt(const char *n, int d) {
    const char *v = getenv(n);
    if (!v || !*v) return d;
    if (!strcasecmp(v, "1") || !strcasecmp(v, "true") || !strcasecmp(v, "yes") || !strcasecmp(v, "on"))
        return 1;
    if (!strcasecmp(v, "0") || !strcasecmp(v, "false") || !strcasecmp(v, "no") || !strcasecmp(v, "off"))
        return 0;
    long x = strtol(v, NULL, 0);
    return x > 0 ? 1 : 0;
}

static int env_toggle_auto(const char *v, int auto_value) {
    if (!v || !*v) return auto_value;
    if (!strcasecmp(v, "auto")) return auto_value;
    if (!strcasecmp(v, "1") || !strcasecmp(v, "true") || !strcasecmp(v, "yes") || !strcasecmp(v, "on"))
        return 1;
    if (!strcasecmp(v, "0") || !strcasecmp(v, "false") || !strcasecmp(v, "no") || !strcasecmp(v, "off"))
        return 0;
    long x = strtol(v, NULL, 0);
    return x > 0 ? 1 : 0;
}

/* ---- token cache checkpoint helpers ---- */

#define TP_CACHE_MAGIC   0x544B4350u   /* 'TPCP' */
#define TP_CACHE_VERSION 1

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint16_t header_bytes;
    uint16_t reserved;
    int32_t  n_layers;
    int32_t  n_embd;
    int32_t  n_heads;
    int32_t  n_kv_heads;
    int32_t  n_head_dim;
    int32_t  n_ff;
    int32_t  n_vocab;
    int32_t  tp_size;
    int32_t  tp_rank;
    int32_t  tp_kv_head_count;
    int32_t  tp_kv_head_base;
    int32_t  max_seq_len;
    int32_t  kv_dtype;
    int32_t  kv_elem_bytes;
    int32_t  kv_k_transposed;
    int32_t  kv_k_dp;
    int32_t  is_hybrid;
    int32_t  is_gemma4;
    int32_t  ssm_dt_rank;
    int32_t  ssm_conv_kernel;
    int32_t  ssm_d_state;
    int32_t  ssm_d_inner;
    int32_t  ssm_qkv_dim;
    int32_t  has_lm_head;
    int32_t  tp_vocab_loc;
    int32_t  tp_vocab_lo;
    int32_t  cache_pos;
    int32_t  next_token;
    int64_t  tok_cache_len;
    int64_t  model_name_len;
    int64_t  conv_bytes;
    int64_t  rec_bytes;
    int64_t  scale_bytes;
} tp_cache_header;

typedef struct {
    int64_t key_bytes;
    int64_t value_bytes;
    int64_t key_scale_bytes;
    int64_t value_scale_bytes;
    int64_t conv_state_bytes;
    int64_t recurrent_state_bytes;
} tp_layer_cache;

static int write_full(FILE *f, const void *buf, size_t n) {
    const uint8_t *p = (const uint8_t *)buf;
    while (n > 0) {
        size_t got = fwrite(p, 1, n, f);
        if (got == 0) return -1;
        p += got; n -= got;
    }
    return 0;
}

static int read_full(FILE *f, void *buf, size_t n) {
    uint8_t *p = (uint8_t *)buf;
    while (n > 0) {
        size_t got = fread(p, 1, n, f);
        if (got == 0) {
            if (feof(f) || ferror(f)) return -1;
        }
        p += got; n -= got;
    }
    return 0;
}

static int write_u32(FILE *f, uint32_t x) { return write_full(f, &x, sizeof(x)); }
static int write_i32(FILE *f, int32_t x)   { return write_full(f, &x, sizeof(x)); }
static int write_u64(FILE *f, uint64_t x)  { return write_full(f, &x, sizeof(x)); }
static int write_i64(FILE *f, int64_t x)   { return write_full(f, &x, sizeof(x)); }

static int read_u32(FILE *f, uint32_t *x) { return read_full(f, x, sizeof(*x)); }
static int read_i32(FILE *f, int32_t *x)  { return read_full(f, x, sizeof(*x)); }
static int read_u64(FILE *f, uint64_t *x) { return read_full(f, x, sizeof(*x)); }
static int read_i64(FILE *f, int64_t *x)  { return read_full(f, x, sizeof(*x)); }

static const char *tp_cache_basename(const char *path) {
    const char *p = path ? strrchr(path, '/') : NULL;
    return p ? p + 1 : (path ? path : "");
}

static void tp_cache_safe_name(const char *src, char *dst, size_t n) {
    size_t j = 0;
    if (!dst || n == 0) return;
    if (!src) src = "";
    for (size_t i = 0; src[i] && j + 1 < n; i++) {
        unsigned char c = (unsigned char)src[i];
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-') {
            dst[j++] = (char)c;
        } else {
            dst[j++] = '_';
        }
    }
    dst[j] = '\0';
    if (j == 0) { const char *def = "model"; snprintf(dst, n, "%s", def); }
}

static int mkdir_parent(const char *path) {
    char dir[PATH_MAX];
    if (!path || !*path) return 0;
    snprintf(dir, sizeof(dir), "%s", path);
    size_t len = strlen(dir);
    if (len == 0) return 0;
    if (dir[len - 1] == '/') dir[len - 1] = '\0';
    for (size_t i = 1; i < len; i++) {
        if (dir[i] != '/') continue;
        char old = dir[i];
        dir[i] = '\0';
        if (mkdir(dir, 0777) != 0 && errno != EEXIST) return -1;
        dir[i] = old;
    }
    if (mkdir(dir, 0777) != 0 && errno != EEXIST) return -1;
    return 0;
}

static int tp_build_cache_path(char *out, size_t out_sz,
                              const transformer_model *m,
                              const char *model_path, const char *rank_pattern,
                              const char *root, const char *tag, int rank,
                              int shared_mode) {
    if (!out || out_sz == 0) return -1;
    out[0] = '\0';
    int n;
    if (rank_pattern && *rank_pattern) {
        if (strchr(rank_pattern, '%')) {
            n = snprintf(out, out_sz, rank_pattern, rank);
        } else {
            n = snprintf(out, out_sz, "%s/rank%02d.cache", rank_pattern, rank);
        }
        return (n > 0 && (size_t)n < out_sz) ? 0 : -1;
    }

    if (!root || !*root) return -1;

    char safe_tag[64], safe_model[64], safe_root[PATH_MAX], fname[PATH_MAX];
    tp_cache_safe_name(tag && *tag ? tag : "default", safe_tag, sizeof(safe_tag));
    tp_cache_safe_name(m ? tp_cache_basename(model_path) : "model", safe_model, sizeof(safe_model));
    snprintf(safe_root, sizeof(safe_root), "%s", root);

    if (shared_mode) {
        n = snprintf(fname, sizeof(fname), "%s_%s.shared.tpck", safe_model, safe_tag);
    } else {
        n = snprintf(fname, sizeof(fname), "%s_%s.r%02d.tpck", safe_model, safe_tag, rank);
    }
    if (n < 0 || (size_t)n >= sizeof(fname)) return -1;
    if (mkdir_parent(safe_root) != 0) return -1;
    n = snprintf(out, out_sz, "%s/%s", safe_root, fname);
    return (n > 0 && (size_t)n < out_sz) ? 0 : -1;
}

static int64_t file_exists_size(const char *p) {
    if (!p) return -1;
    struct stat st;
    if (stat(p, &st) != 0) return -1;
    return (int64_t)st.st_size;
}

static size_t tf_cache_seq_len_for_layer(const transformer_model *m, int l) {
    return tf_runtime_kv_len_for_layer(m, l);
}

static int tf_cache_kv_dim_for_layer(const transformer_model *m, int l) {
    return tf_runtime_kv_dim_for_layer(m, l);
}

static tp_layer_cache tp_layer_cache_bytes(const transformer_model *m, int l) {
    tp_layer_cache x = {0,0,0,0,0,0};
    if (!m || l < 0 || l >= m->n_layers) return x;
    if (!m->key_cache || !m->value_cache) return x;
    if (!m->key_cache[l] || !m->value_cache[l]) return x;
    if (m->is_hybrid && m->layers && m->layers[l].is_ssm) return x;
    if (m->is_gemma4 && m->layers && m->layers[l].shared_kv_source >= 0) return x;

    size_t seq = tf_cache_seq_len_for_layer(m, l);
    int kv_dim = tf_cache_kv_dim_for_layer(m, l);
    x.key_bytes = (int64_t)seq * kv_dim * (int64_t)m->kv_elem_bytes;
    x.value_bytes = x.key_bytes;
    if (m->key_scales && m->key_scales[l])
        x.key_scale_bytes = (int64_t)seq * (int64_t)m->n_kv_heads * (int64_t)sizeof(float);
    if (m->value_scales && m->value_scales[l])
        x.value_scale_bytes = x.key_scale_bytes;
    if (m->conv_state && m->conv_state[l] && m->ssm_conv_kernel > 1 && m->ssm_qkv_dim > 0) {
        x.conv_state_bytes = (int64_t)(m->ssm_conv_kernel - 1) * (int64_t)m->ssm_qkv_dim * (int64_t)sizeof(float);
    }
    if (m->recurrent_state && m->recurrent_state[l] && m->ssm_dt_rank > 0 && m->ssm_d_state > 0) {
        x.recurrent_state_bytes = (int64_t)m->ssm_dt_rank * (int64_t)m->ssm_d_state * (int64_t)m->ssm_d_state * (int64_t)sizeof(float);
    }
    return x;
}

static int tp_cache_matches_model(const tp_cache_header *h, const transformer_model *m,
                                 int n, int tp_size, int rank, int shared_mode) {
    if (!h || !m) return 0;
    if (h->magic != TP_CACHE_MAGIC || h->version != TP_CACHE_VERSION) return 0;
    if (h->n_layers != m->n_layers) return 0;
    if (h->n_embd != m->n_embd) return 0;
    if (h->n_heads != m->n_heads) return 0;
    if (h->n_kv_heads != m->n_kv_heads) return 0;
    if (h->n_head_dim != m->head_dim) return 0;
    if (h->n_ff != m->n_ff) return 0;
    if (h->n_vocab != m->n_vocab) return 0;
    if (h->tp_size != tp_size) return 0;
    if (!shared_mode && h->tp_rank != rank) return 0;
    if (h->tp_kv_head_count != m->tp_kv_head_count) return 0;
    if (h->tp_kv_head_base != m->tp_kv_head_base) return 0;
    if (h->kv_dtype != m->kv_dtype) return 0;
    if (h->kv_elem_bytes != m->kv_elem_bytes) return 0;
    if (h->kv_k_transposed != m->kv_k_transposed) return 0;
    if (h->kv_k_dp != m->kv_k_dp) return 0;
    if (h->is_hybrid != m->is_hybrid) return 0;
    if (h->is_gemma4 != m->is_gemma4) return 0;
    if (h->ssm_dt_rank != m->ssm_dt_rank) return 0;
    if (h->ssm_conv_kernel != m->ssm_conv_kernel) return 0;
    if (h->ssm_d_state != m->ssm_d_state) return 0;
    if (h->ssm_d_inner != m->ssm_d_inner) return 0;
    if (h->ssm_qkv_dim != m->ssm_qkv_dim) return 0;
    if (h->has_lm_head != m->has_lm_head) return 0;
    if (h->tp_vocab_loc != m->tp_vocab_loc || h->tp_vocab_lo != m->tp_vocab_lo) return 0;
    if (h->cache_pos < 0) return 0;
    if (h->max_seq_len > 0 && n >= 0 && h->max_seq_len < n) return 0;
    return (n >= 0) ? (h->cache_pos <= n) : 1;
}

static int tp_checkpoint_write(transformer_model *m, const char *path,
                              int cache_pos, int32_t next_token,
                              int shared_mode) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    tp_cache_header h = {
        .magic = TP_CACHE_MAGIC,
        .version = TP_CACHE_VERSION,
        .header_bytes = (uint16_t)sizeof(h),
        .n_layers = m->n_layers,
        .n_embd = m->n_embd,
        .n_heads = m->n_heads,
        .n_kv_heads = m->n_kv_heads,
        .n_head_dim = m->head_dim,
        .n_ff = m->n_ff,
        .n_vocab = m->n_vocab,
        .tp_size = m->tp_size,
        .tp_rank = shared_mode ? -1 : m->tp_rank,
        .tp_kv_head_count = m->tp_kv_head_count,
        .tp_kv_head_base = m->tp_kv_head_base,
        .max_seq_len = m->max_seq_len,
        .kv_dtype = m->kv_dtype,
        .kv_elem_bytes = m->kv_elem_bytes,
        .kv_k_transposed = m->kv_k_transposed,
        .kv_k_dp = m->kv_k_dp,
        .is_hybrid = m->is_hybrid,
        .is_gemma4 = m->is_gemma4,
        .ssm_dt_rank = m->ssm_dt_rank,
        .ssm_conv_kernel = m->ssm_conv_kernel,
        .ssm_d_state = m->ssm_d_state,
        .ssm_d_inner = m->ssm_d_inner,
        .ssm_qkv_dim = m->ssm_qkv_dim,
        .has_lm_head = m->has_lm_head,
        .tp_vocab_loc = m->tp_vocab_loc,
        .tp_vocab_lo = m->tp_vocab_lo,
        .cache_pos = cache_pos,
        .next_token = next_token,
        .tok_cache_len = (int64_t)m->max_seq_len,
        .model_name_len = 0,
        .conv_bytes = 0,
        .rec_bytes = 0,
        .scale_bytes = 0,
    };

    int ok = 0;
    for (int l = 0; l < m->n_layers; l++) {
        tp_layer_cache lc = tp_layer_cache_bytes(m, l);
        h.conv_bytes += lc.conv_state_bytes;
        h.rec_bytes += lc.recurrent_state_bytes;
        h.scale_bytes += lc.key_scale_bytes + lc.value_scale_bytes;
    }

    if (write_full(f, &h, sizeof(h)) != 0) goto fail;
    for (int l = 0; l < m->n_layers; l++) {
        tp_layer_cache lc = tp_layer_cache_bytes(m, l);
        if (write_i64(f, lc.key_bytes) != 0) goto fail;
        if (write_i64(f, lc.value_bytes) != 0) goto fail;
        if (write_i64(f, lc.key_scale_bytes) != 0) goto fail;
        if (write_i64(f, lc.value_scale_bytes) != 0) goto fail;
        if (write_i64(f, lc.conv_state_bytes) != 0) goto fail;
        if (write_i64(f, lc.recurrent_state_bytes) != 0) goto fail;

        if (lc.key_bytes)   if (write_full(f, m->key_cache[l],   (size_t)lc.key_bytes)  != 0) goto fail;
        if (lc.value_bytes) if (write_full(f, m->value_cache[l], (size_t)lc.value_bytes) != 0) goto fail;
        if (lc.key_scale_bytes && m->key_scales && m->key_scales[l])
            if (write_full(f, m->key_scales[l], (size_t)lc.key_scale_bytes) != 0) goto fail;
        if (lc.value_scale_bytes && m->value_scales && m->value_scales[l])
            if (write_full(f, m->value_scales[l], (size_t)lc.value_scale_bytes) != 0) goto fail;
        if (lc.conv_state_bytes && m->conv_state && m->conv_state[l])
            if (write_full(f, m->conv_state[l], (size_t)lc.conv_state_bytes) != 0) goto fail;
        if (lc.recurrent_state_bytes && m->recurrent_state && m->recurrent_state[l])
            if (write_full(f, m->recurrent_state[l], (size_t)lc.recurrent_state_bytes) != 0) goto fail;
    }

    if (m->conv_state_pos) {
        for (int l = 0; l < m->n_layers; l++) {
            int32_t p = m->conv_state_pos[l];
            if (write_i32(f, p) != 0) goto fail;
        }
    }
    ok = 1;

fail:
    if (fclose(f) != 0) ok = 0;
    return ok ? 0 : -1;
}

static int tp_checkpoint_load(transformer_model *m, const char *path,
                             int64_t *cache_pos, int32_t *next_token,
                             int rank, int tp_size, int shared_mode) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    tp_cache_header h;
    if (read_full(f, &h, sizeof(h)) != 0) { fclose(f); return -1; }
    if (!tp_cache_matches_model(&h, m, (int)h.max_seq_len, tp_size, rank, shared_mode)) {
        fclose(f);
        return -1;
    }
    if (h.cache_pos < 0 || h.cache_pos >= h.max_seq_len) { fclose(f); return -1; }
    if (cache_pos) *cache_pos = h.cache_pos;
    if (next_token) *next_token = h.next_token;

    for (int l = 0; l < h.n_layers; l++) {
        tp_layer_cache lc = {0,0,0,0,0,0};
        if (read_i64(f, &lc.key_bytes) != 0 || read_i64(f, &lc.value_bytes) != 0 ||
            read_i64(f, &lc.key_scale_bytes) != 0 || read_i64(f, &lc.value_scale_bytes) != 0 ||
            read_i64(f, &lc.conv_state_bytes) != 0 || read_i64(f, &lc.recurrent_state_bytes) != 0) {
            fclose(f); return -1;
        }

        if (lc.key_bytes > 0 && m->key_cache && m->key_cache[l]) {
            if (read_full(f, m->key_cache[l], (size_t)lc.key_bytes) != 0) { fclose(f); return -1; }
        } else {
            if (lc.key_bytes > 0 && fseek(f, (long long)lc.key_bytes, SEEK_CUR) != 0) { fclose(f); return -1; }
        }
        if (lc.value_bytes > 0 && m->value_cache && m->value_cache[l]) {
            if (read_full(f, m->value_cache[l], (size_t)lc.value_bytes) != 0) { fclose(f); return -1; }
        } else {
            if (lc.value_bytes > 0 && fseek(f, (long long)lc.value_bytes, SEEK_CUR) != 0) { fclose(f); return -1; }
        }
        if (lc.key_scale_bytes > 0 && m->key_scales && m->key_scales[l]) {
            if (read_full(f, m->key_scales[l], (size_t)lc.key_scale_bytes) != 0) { fclose(f); return -1; }
        } else if (lc.key_scale_bytes > 0) {
            if (fseek(f, (long long)lc.key_scale_bytes, SEEK_CUR) != 0) { fclose(f); return -1; }
        }
        if (lc.value_scale_bytes > 0 && m->value_scales && m->value_scales[l]) {
            if (read_full(f, m->value_scales[l], (size_t)lc.value_scale_bytes) != 0) { fclose(f); return -1; }
        } else if (lc.value_scale_bytes > 0) {
            if (fseek(f, (long long)lc.value_scale_bytes, SEEK_CUR) != 0) { fclose(f); return -1; }
        }
        if (lc.conv_state_bytes > 0 && m->conv_state && m->conv_state[l]) {
            if (read_full(f, m->conv_state[l], (size_t)lc.conv_state_bytes) != 0) { fclose(f); return -1; }
        } else if (lc.conv_state_bytes > 0) {
            if (fseek(f, (long long)lc.conv_state_bytes, SEEK_CUR) != 0) { fclose(f); return -1; }
        }
        if (lc.recurrent_state_bytes > 0 && m->recurrent_state && m->recurrent_state[l]) {
            if (read_full(f, m->recurrent_state[l], (size_t)lc.recurrent_state_bytes) != 0) { fclose(f); return -1; }
        } else if (lc.recurrent_state_bytes > 0) {
            if (fseek(f, (long long)lc.recurrent_state_bytes, SEEK_CUR) != 0) { fclose(f); return -1; }
        }
    }

    if (m->conv_state_pos) {
        for (int l = 0; l < m->n_layers; l++) {
            int32_t p = 0;
            if (read_i32(f, &p) != 0) { fclose(f); return -1; }
            m->conv_state_pos[l] = p;
        }
    }

    fclose(f);
    return 0;
}

static int tp_checkpoint_probe(const transformer_model *m, const char *path,
                              int64_t *cache_pos, int32_t *next_token,
                              int tp_size, int rank, int n, int shared_mode) {
    if (!m || !path || !*path) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    tp_cache_header h;
    int rc = read_full(f, &h, sizeof(h));
    fclose(f);
    if (rc != 0) return -1;

    if (!tp_cache_matches_model(&h, m, n, tp_size, rank, shared_mode)) return -1;
    if (h.cache_pos < 0 || h.cache_pos > h.max_seq_len) return -1;

    if (cache_pos) *cache_pos = h.cache_pos;
    if (next_token) *next_token = h.next_token;
    return 0;
}

static char *build_chat_text(const char *prompt, size_t *out_len) {
    const char *pre = "<|im_start|>user\n";
    const char *post = "<|im_end|>\n<|im_start|>assistant\n";
    size_t p_len = prompt ? (size_t)strlen(prompt) : 0;
    size_t cap = strlen(pre) + p_len + strlen(post) + 1;
    char *text = (char *)malloc(cap);
    if (!text) return NULL;
    memcpy(text, pre, strlen(pre));
    if (p_len > 0) memcpy(text + strlen(pre), prompt, p_len);
    memcpy(text + strlen(pre) + p_len, post, strlen(post));
    text[cap - 1] = 0;
    if (out_len) *out_len = cap - 1;
    return text;
}

static int32_t sample_argmax(transformer_model *m, float *lg, tp_comm *c,
                            double *ar_secs_out, long *ar_calls_out) {
    int32_t nt = 0;
    float best = -1e30f;
    int nloc = m->tp_vocab_sharded ? m->tp_vocab_loc : m->n_vocab;
    for (int v = 0; v < nloc; v++) {
        if (lg[v] > best) { best = lg[v]; nt = v; }
    }
    nt += m->tp_vocab_lo;
    if (m->tp_vocab_sharded) {
        double t0 = now_sec();
        tp_allreduce_argmax(c, &best, &nt);
        if (ar_secs_out) *ar_secs_out += now_sec() - t0;
        if (ar_calls_out) (*ar_calls_out)++;
    }
    return nt;
}

static void print_token(const bpe_vocab *vocab, int32_t nt) {
    const char *s = bpe_token_to_str(vocab, nt);
    if (!s) return;
    int dec_len = 0;
    char *dec = bpe_byte_decode(s, (int)strlen(s), &dec_len);
    const char *out = dec ? dec : s;
    int len = dec ? dec_len : (int)strlen(s);
    if (g_log) { fwrite(out, 1, len, g_log); fflush(g_log); }
    fwrite(out, 1, len, stdout); fflush(stdout);
    free(dec);
}

static int token_in_vocab(const bpe_vocab *vocab, int32_t tok) {
    return (vocab && tok >= 0 && tok < vocab->n_tokens) ? 1 : 0;
}

/* ---- topology (tofu_topo.txt) ---- */
static int read_topo(uint8_t coords[][TOFU_NCOORDS]) {
    FILE *f = fopen(TOPO_PATH, "r");
    if (!f) { perror("cannot open " TOPO_PATH); fprintf(stderr, "  (run tofu_topo_helper first)\n"); exit(1); }
    int n = 0; char line[256];
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        if (n >= MAX_NODES) { fprintf(stderr, "too many nodes\n"); exit(1); }
        unsigned r, c[TOFU_NCOORDS];
        if (sscanf(line, "%u %u %u %u %u %u %u", &r, &c[0], &c[1], &c[2], &c[3], &c[4], &c[5]) != 7)
            { fprintf(stderr, "malformed line: %s", line); exit(1); }
        if ((int)r != n) { fprintf(stderr, "%s ranks out of order\n", TOPO_PATH); exit(1); }
        for (int k = 0; k < TOFU_NCOORDS; k++) coords[n][k] = (uint8_t)c[k];
        n++;
    }
    fclose(f);
    if (n < 1) { fprintf(stderr, "%s lists %d node(s)\n", TOPO_PATH, n); exit(1); }
    return n;
}

/* ---- uTofu state (barrier region only; the all-reduce keeps its own region) ---- */
static int             N, MyRank;
static char           *Region;
static size_t          SEND_OFF, BAR_BASE, SlotSend, SlotB;
static utofu_vcq_hdl_t Vcq;
static utofu_stadd_t   Base;
static utofu_vcq_id_t  PeerVcq[MAX_NODES];
static utofu_stadd_t   PeerBase[MAX_NODES];
static const unsigned long FLAGS = UTOFU_ONESIDED_FLAG_TCQ_NOTICE;
static uint64_t        Bt = 1;

static inline size_t bar_recv_off(int s) { return BAR_BASE + (size_t)s * SlotB; }
static inline size_t bar_go_off(void)    { return BAR_BASE + (size_t)N * SlotB; }

static void put_issue(utofu_vcq_id_t pv, utofu_stadd_t s, utofu_stadd_t d, size_t len, int drain) {
    int rc; void *cb;
    for (;;) { rc = utofu_put(Vcq, pv, s, d, len, 0, FLAGS, NULL);
               if (rc != UTOFU_ERR_BUSY) break; utofu_poll_tcq(Vcq, 0, &cb); }
    if (rc != UTOFU_SUCCESS) die("utofu_put", rc);
    if (drain) { do { rc = utofu_poll_tcq(Vcq, 0, &cb); } while (rc == UTOFU_ERR_NOT_FOUND);
                 if (rc != UTOFU_SUCCESS) die("utofu_poll_tcq", rc); }
}
static void wait_ge(volatile uint64_t *q, uint64_t v, const char *what) {
    double ts = now_sec();
    while (*q < v) if (now_sec() - ts > WAIT_TIMEOUT_SEC) die(what, -1);
}

/* fan-in to rank 0, fan-out release (pp_runner idiom). robust=1 retries (startup). */
static void barrier_robust(int robust) {
    uint64_t t = ++Bt;
    char *sb = Region + SEND_OFF;
    if (MyRank == 0) {
        for (int s = 1; s < N; s++)
            wait_ge((volatile uint64_t *)(Region + bar_recv_off(s)), t, "barrier fan-in");
        for (int s = 1; s < N; s++) {
            *(volatile uint64_t *)sb = t;
            put_issue(PeerVcq[s], Base + SEND_OFF, PeerBase[s] + bar_go_off(), 8, 1);
        }
    } else {
        volatile uint64_t *go = (volatile uint64_t *)(Region + bar_go_off());
        double ts = now_sec();
        do {
            *(volatile uint64_t *)sb = t;
            put_issue(PeerVcq[0], Base + SEND_OFF, PeerBase[0] + bar_recv_off(MyRank), 8, 1);
            if (!robust) { wait_ge(go, t, "barrier release"); break; }
            for (int a = 0; a < 50 && *go < t; a++) usleep(2000);
            if (now_sec() - ts > WAIT_TIMEOUT_SEC) die("bootstrap barrier timeout", -1);
        } while (*go < t);
    }
}
static void barrier(void) { barrier_robust(0); }   /* tp_comm_init's barrier_fn */

/* Per-token inter-node all-reduce timing. g_ar_secs/g_ar_calls are reset at the
 * top of each decode step and read back after the forward to split per-token
 * time into compute vs comm (both the in-layer sum all-reduces, via the callback,
 * and the vocab argmax-reduce, timed inline in the loop). */
static double g_ar_secs  = 0.0;
static long   g_ar_calls = 0;

/* transformer_set_tp callback shim -> tp_allreduce.h.
 * The comm region is sized for c->max_count floats (= n_embd). Decode reduces
 * exactly n_embd, but the BATCHED-prefill path reduces the whole [M,n_embd] tile
 * at once (count = M*n_embd >> max_count) — reducing that in one tp_allreduce_sum
 * overruns the registered send slot → UTOFU_ERR_TCQ_MEMORY (-130). Chunk by
 * max_count here so no caller can overrun the region. Lockstep-safe: every TP
 * rank passes the identical count (M and n_embd are replicated), so all ranks
 * issue the same number of sub-reduces in the same order → the per-comm seq
 * counter stays aligned with the argmax reduces. */
static void tp_ar_callback(float *buf, int count, void *ctx) {
    tp_comm *c = (tp_comm *)ctx;
    int mc = c->max_count > 0 ? c->max_count : count;
    double t0 = now_sec();
    for (int off = 0; off < count; ) {
        int n = count - off;
        if (n > mc) n = mc;
        tp_allreduce_sum(c, buf + off, n);
        off += n;
    }
    g_ar_secs += now_sec() - t0;
    g_ar_calls++;
}

static void tp_dry_token_walk(transformer_model *m, tp_comm *c, int32_t tok,
                             int work_reps, int ar_reps) {
    if (!m || !m->x || m->n_embd <= 0) return;
    int n = m->n_embd;
    float seed = (float)(tok & 0x7fffffff) * 0.001f;
    for (int i = 0; i < n; i++) {
        m->x[i] = seed + 0.0001f * (float)i;
    }
    for (int r = 0; r < work_reps; r++) {
        float acc = 0.0f;
        for (int i = 0; i < n; i++) {
            acc += m->x[i] * 0.00001f;
            m->x[i] = acc;
        }
    }
    for (int a = 0; a < ar_reps; a++) tp_allreduce_sum(c, m->x, n);
}

static int32_t tp_next_token_synth(int32_t tok, int32_t step, int n_vocab) {
    if (step <= 0) step = 1;
    if (n_vocab <= 1) return 0;
    int64_t t = (int64_t)tok + (int64_t)step;
    t = ((t % n_vocab) + n_vocab) % n_vocab;
    if (t < 0) t += n_vocab;
    return (int32_t)t;
}

/* Local per-token weight bytes actually streamed by the matvecs on this rank
 * (post-slice dims). Rough BF16 estimate (2 B/elem) for an effective GB/s; SSM
 * recurrence state and tiny norm/bias vectors are negligible and omitted. */
static double tp_local_weight_bytes(const transformer_model *m) {
    const double EB = 2.0;   /* BF16 weight element */
    double b = 0.0;
    #define QB(t) (b += (double)(t).n_rows * (double)(t).n_cols * EB)
    for (int l = 0; l < m->n_layers; l++) {
        const transformer_layer *L = &m->layers[l];
        if (!(m->is_hybrid && L->is_ssm)) {
            QB(L->attn_q); QB(L->attn_k); QB(L->attn_v); QB(L->attn_output);
        } else {
            QB(L->ssm_qkv); QB(L->ssm_gate); QB(L->ssm_out);
        }
        QB(L->ffn_gate); QB(L->ffn_up); QB(L->ffn_down);
    }
    QB(m->output);   /* LM head (local vocab shard) */
    #undef QB
    return b;
}

int main(int argc, char **argv) {
    int rc;
    if (argc < 2) { fprintf(stderr, "usage: %s <model-shard1.gguf>\n", argv[0]); return 1; }
    const char *model_path = argv[1];
    const char *prompt_env = envs("TP_PROMPT", "Hello, who are you?");
    const char *prompt_file = envs("TP_PROMPT_FILE", "");
    char *prompt_file_text = NULL;
    const char *prompt = prompt_env;
    int  max_gen           = (int)envl("TP_MAXGEN", 64);
    int  llm_threads       = (int)envl("LLM_THREADS", 48);
    int  ignore_eos        = (int)envl("TP_IGNORE_EOS", 0);  /* long-ctx perf sweep: don't stop at EOS */
    int  prefill_only      = envb("TP_PREFILL_ONLY", 0);
    int  do_prefill_gemm   = envb("TP_PREFILL_GEMM", 1);      /* 1=batched prefill, fallback to token loop */
    int  synth_tokens = (int)envl_opt("TP_SYNTH_TOKENS", 0);  /* >0: skip prompt tokenize, use synthetic IDs */
    int  synth_token_id = (int)envl_opt("TP_SYNTH_TOKEN_ID", 1);
    int  dry_mode         = envb("TP_DRY", 0);
    int  dry_prefill      = envb_opt("TP_DRY_PREFILL", dry_mode);
    int  dry_decode       = envb_opt("TP_DRY_DECODE", dry_mode);
    int  dry_work_reps    = (int)envl_opt("TP_DRY_WORK_REPS", 0);
    int  dry_ar_steps     = (int)envl_opt("TP_DRY_AR_STEPS", 0);
    int  dry_token_step   = (int)envl_opt("TP_DRY_TOKEN_STEP", 1);
    int  cache_load         = envb_opt("TP_CACHE_LOAD", 0);
    int  cache_save         = envb_opt("TP_CACHE_SAVE", 0);
    const char *cache_shared_s = envs_opt("TP_CACHE_SHARED", "auto");
    const char *cache_dir = envs_opt("TP_CACHE_DIR", "");
    const char *cache_path_env = envs_opt("TP_CACHE_PATH", "");
    const char *cache_tag = envs_opt("TP_CACHE_TAG", "tp");
    int cache_shared = -1;

    /* required for correctness with row-parallel strided slices (see header) */
    setenv("TF_KEEP_BF16_SRC", "1", 1);

    /* ---- uTofu bootstrap (single TNI) ---- */
    utofu_tni_id_t *tni_ids = NULL; size_t num_tnis = 0;
    rc = utofu_get_onesided_tnis(&tni_ids, &num_tnis);
    if (rc != UTOFU_SUCCESS) die("utofu_get_onesided_tnis", rc);
    if (num_tnis < 1) die("no onesided TNIs", -1);

    uint8_t my_coords[TOFU_NCOORDS] = {0};
    rc = utofu_query_my_coords(my_coords);
    if (rc != UTOFU_SUCCESS) die("utofu_query_my_coords", rc);

    static uint8_t topo[MAX_NODES][TOFU_NCOORDS];
    N = read_topo(topo);
    MyRank = -1;
    for (int r = 0; r < N; r++) if (memcmp(topo[r], my_coords, TOFU_NCOORDS) == 0) MyRank = r;
    if (MyRank == -1) { fprintf(stderr, "my coords not in %s\n", TOPO_PATH); exit(1); }

    /* FJ mpiexec drops per-rank stderr, so numa:/tp_slice:/transformer: diagnostics
     * and the -DTF_POOL_PROFILE shutdown dump are otherwise lost. Capture this rank's
     * stderr to a file so they survive. */
    {   char en[64]; snprintf(en, sizeof en, "tp_stderr_rank%02d.txt", MyRank);
        if (!freopen(en, "w", stderr)) { /* keep going on the original stderr */ }
        setvbuf(stderr, NULL, _IOLBF, 0);
    }

    if (MyRank == 0) {
        char name[64];
        snprintf(name, sizeof name, "tp_run_%u_%u_%u_%u_%u_%u.txt",
                 my_coords[0], my_coords[1], my_coords[2], my_coords[3], my_coords[4], my_coords[5]);
        g_log = fopen(name, "w");
        g_curve = fopen("tp_curve_rank00.txt", "w");
        if (g_curve) fprintf(g_curve, "# pos ctx_len fwd_ms comm_ms\n");
        if (envb_opt("TP_DUMP_TOKENS", 0)) g_tokdump = fopen("tp_tokens_rank00.txt", "w");
    }

    /* ---- prompt + tokenize (every rank: deterministic, no comm) ---- */
    if (prompt_file[0]) {
        size_t read_len = 0;
        prompt_file_text = read_text_file(prompt_file, &read_len);
        if (prompt_file_text) {
            if (read_len > 0 && prompt_file_text[read_len - 1] == '\n')
                prompt_file_text[read_len - 1] = 0;
            prompt = prompt_file_text;
        }
    }

    gguf_context *gguf = gguf_open_multi(model_path, 1);   /* lazy mmap */
    if (!gguf) die("gguf_open_multi", -1);
    bpe_vocab *vocab = bpe_vocab_load(gguf);
    if (!vocab) die("bpe_vocab_load", -1);

    int P = 0, tok_cap = 0;
    int32_t *ptoks = NULL;

    if (synth_tokens > 0) {
        if (synth_tokens < 1) die("TP_SYNTH_TOKENS must be > 0", -1);
        P = synth_tokens;
        tok_cap = P + 64;
        if (tok_cap < 256) tok_cap = 256;
        ptoks = (int32_t *)malloc((size_t)tok_cap * sizeof(int32_t));
        if (!ptoks) die("malloc(ptoks)", -1);
        if (synth_token_id <= 0) synth_token_id = 1;
        for (int i = 0; i < P; i++) ptoks[i] = (int32_t)synth_token_id;
    } else {
        char *text = build_chat_text(prompt, NULL);
        if (!text) die("build chat text", -1);

        P = bpe_tokenize(vocab, text, -1, NULL, 0);
        if (P <= 0) die("tokenize produced 0 tokens", -1);
        tok_cap = P + 64;
        if (tok_cap < 256) tok_cap = 256;

        ptoks = (int32_t *)malloc((size_t)tok_cap * sizeof(int32_t));
        if (!ptoks) die("malloc(ptoks)", -1);
        P = bpe_tokenize(vocab, text, -1, ptoks, tok_cap);
        if (P > tok_cap) die("tokenization budget overflow", -1);
        free(text);
    }

    int cfg_max_seq = (int)envl("TP_MAXSEQ", 0);
    int need_seq = P + max_gen + 16;
    if (prefill_only) need_seq = P + 16;
    int max_seq = cfg_max_seq > 0 ? cfg_max_seq : need_seq;
    if (max_seq < need_seq) {
        fprintf(stderr, "TP_MAXSEQ=%d too small for P=%d max_gen=%d; clamping to %d\n", max_seq, P, max_gen, need_seq);
        max_seq = need_seq;
    }

    /* ---- load FULL model, slice to this rank's shard, build panels ---- */
    double tl0 = now_sec();
    transformer_model *m = transformer_load(gguf, max_seq);
    if (!m) die("transformer_load", -1);
    double tl1 = now_sec();
    int n_layers = m->n_layers;
    int n_embd   = m->n_embd;

    if (llm_threads > 1) transformer_set_threads(m, llm_threads);
    /* Hybrid models (Qwen3.5/3.6) need Stage-B SSM V-head sharding to fit; pure
     * transformer 9B only has attn+FFN to shard. TP_NO_SSM_SHARD forces Stage A. */
    int ssm_shard = m->is_hybrid && !getenv("TP_NO_SSM_SHARD");
    if (transformer_tp_slice_weights(m, MyRank, N, ssm_shard) != 0)
        die("transformer_tp_slice_weights (check n_heads/n_kv/n_ff/ssm_dt % N)", -1);
    double tp0 = now_sec();
    transformer_build_panels(m);                  /* repack the local shard */
    double tp1 = now_sec();
    if (dry_mode) {
        if (dry_ar_steps <= 0) dry_ar_steps = 2;
        if (dry_work_reps < 0) dry_work_reps = 0;
        if (dry_token_step <= 0) dry_token_step = 1;
    }
    {   char tname[64]; snprintf(tname, sizeof tname, "tp_load_rank%02d.txt", MyRank);
        FILE *tf = fopen(tname, "w");
        if (tf) { fprintf(tf, "rank %d: transformer_load=%.1fs build_panels=%.1fs "
                          "(n_heads=%d n_kv=%d n_ff=%d hybrid=%d ssm_shard=%d "
                          "ssm_dt=%d ssm_qkv=%d ssm_d_inner=%d head_off=%d "
                          "vocab_shard=%d vocab_loc=%d vocab_lo=%d)\n",
                          MyRank, tl1 - tl0, tp1 - tp0, m->n_heads, m->n_kv_heads, m->n_ff,
                          m->is_hybrid, m->tp_ssm_sharded, m->ssm_dt_rank, m->ssm_qkv_dim,
                          m->ssm_d_inner, m->ssm_head_offset,
                          m->tp_vocab_sharded, m->tp_vocab_loc, m->tp_vocab_lo); fclose(tf); }
    }

    /* ---- barrier region (own cache line per remote-written slot) ---- */
    SlotSend = DEMO_CACHE_LINE;
    SlotB    = DEMO_CACHE_LINE;
    SEND_OFF = 0;
    BAR_BASE = SlotSend;
    size_t region_sz = BAR_BASE + (size_t)(N + 1) * SlotB;
    if (posix_memalign((void **)&Region, DEMO_CACHE_LINE, region_sz) != 0) die("posix_memalign", -1);
    memset(Region, 0, region_sz);

    /* ---- VCQ + region registration; reconstruct peers by convention ---- */
    utofu_tni_id_t tni = tni_ids[0];
    rc = utofu_create_vcq_with_cmp_id(tni, DEMO_CMP_ID, 0, &Vcq);
    if (rc != UTOFU_SUCCESS) die("utofu_create_vcq_with_cmp_id", rc);
    utofu_vcq_id_t my_real;
    rc = utofu_query_vcq_id(Vcq, &my_real);
    if (rc != UTOFU_SUCCESS) die("utofu_query_vcq_id", rc);
    {   utofu_vcq_id_t conv;
        rc = utofu_construct_vcq_id(my_coords, tni, DEMO_CQ_ID, DEMO_CMP_ID, &conv);
        if (rc != UTOFU_SUCCESS) die("utofu_construct_vcq_id(self)", rc);
        utofu_vcq_id_t a = my_real, b = conv;
        utofu_set_vcq_id_path(&a, NULL); utofu_set_vcq_id_path(&b, NULL);
        if (a != b) die("VCQ self-check", -1);
    }
    rc = utofu_reg_mem_with_stag(Vcq, Region, region_sz, RUN_STAG, 0, &Base);
    if (rc != UTOFU_SUCCESS) die("utofu_reg_mem_with_stag", rc);
    for (int r = 0; r < N; r++) {
        if (r == MyRank) { PeerVcq[r] = my_real; PeerBase[r] = Base; continue; }
        rc = utofu_construct_vcq_id(topo[r], tni, DEMO_CQ_ID, DEMO_CMP_ID, &PeerVcq[r]);
        if (rc != UTOFU_SUCCESS) die("utofu_construct_vcq_id(peer)", rc);
        utofu_set_vcq_id_path(&PeerVcq[r], NULL);
        rc = utofu_query_stadd(PeerVcq[r], RUN_STAG, &PeerBase[r]);
        if (rc != UTOFU_SUCCESS) die("utofu_query_stadd(peer)", rc);
    }
    free(tni_ids);

    int is_first = (MyRank == 0);

    if (is_first)
        logmsg("=== tensor-parallel decode: %d ranks, %d layers, n_embd=%d ===\n"
               "model=%s\nprompt=\"%s\"  P=%d tokens  max_gen=%d  max_seq=%d  threads=%d\n",
               N, n_layers, n_embd, model_path, prompt, P, max_gen, max_seq, llm_threads);
    if (is_first) {
        logmsg("mode: dry=%d prefill=%d decode=%d work_reps=%d ar_steps=%d token_step=%d\n",
               dry_mode, dry_prefill, dry_decode, dry_work_reps, dry_ar_steps, dry_token_step);
    }
    logmsg("[rank %d] sharded n_heads=%d n_kv=%d n_ff=%d kv_heads=%d kv_head_base=%d  region=%.1f KiB\n",
           MyRank, m->n_heads, m->n_kv_heads, m->n_ff, m->tp_kv_head_count, m->tp_kv_head_base, region_sz / 1024.0);

    if (m->tp_attn_sharded && m->tp_kv_head_count > 0 && m->tp_kv_head_count < m->n_kv_heads) {
        transformer_resize_kv_for_tp(m, 0, n_layers, m->tp_kv_head_count * m->head_dim);
    }

    barrier_robust(1);   /* robust startup bootstrap (all ranks registered + running) */

    /* ---- init all-reduce comm + wire into the model ----
     * Region holds max_count floats. Decode reduces exactly n_embd, but BATCHED
     * prefill reduces the whole [M,n_embd] tile; tp_ar_callback chunks it by
     * max_count, so a max_count of n_embd makes prefill do M (=512) tiny reduces
     * per all-reduce point — pure Put-latency tax. Size the region for TP_AR_BATCH
     * tokens so prefill reduces in ceil(M/chunk) chunks. The default 512 saturates
     * the 2097152-float (8MiB fp32 / 4MiB bf16) clamp below at n_embd=5120 (region
     * ~72MB); measured M=1000 prefill 116.8->118.5 tok/s vs the old 128 default, no
     * regression at smaller M (Workstream B sweep; AR is bandwidth- not latency-bound
     * so the win saturates at the clamp). Safe to raise only post-Workstream-A (the
     * MRQ-drain fix). The argmax send is 16 B regardless, so decode is unaffected. */
    int ar_batch = (int)envl("TP_AR_BATCH", 512);
    if (ar_batch < 1) ar_batch = 1;
    long ar_max = (long)n_embd * ar_batch;
    if (ar_max > 2L * 1024 * 1024) ar_max = 2L * 1024 * 1024;  /* cap bf16 Put < 16MiB, region < ~75MB */
    if (ar_max < n_embd) ar_max = n_embd;
    tp_comm c;
    if (tp_comm_init(&c, Vcq, PeerVcq, MyRank, N, (int)ar_max, barrier) != 0) die("tp_comm_init", -1);
    if (is_first) logmsg("tp_ar region: max_count=%ld (TP_AR_BATCH=%d tokens, ~%.1f MB region)\n",
                         ar_max, ar_batch, (double)(9L * ar_max * 4) / (1024*1024));
    transformer_set_tp(m, MyRank, N, tp_ar_callback, &c);
    barrier();

    int cache_loaded = 0;
    int cache_prefill_used = 0;
    int cache_prefill_skipped = 0;
    int64_t ck_pos = -1;
    int32_t ck_next = -1;
    char cache_path[PATH_MAX];
    int have_cache_path = 0;

    if (cache_shared < 0) {
        cache_shared = (m->tp_kv_head_count == m->n_kv_heads) ? 1 : 0;
    }
    cache_shared = env_toggle_auto(cache_shared_s, cache_shared);
    if (cache_shared && (m->tp_kv_head_count != m->n_kv_heads)) {
        if (is_first)
            logmsg("TP cache shared requested but non-replicated KV cache: disabling shared cache\n");
        cache_shared = 0;
    }

    if (!cache_shared && cache_path_env[0] && N > 1) {
        if (is_first)
            logmsg("TP_CACHE_PATH is rank-agnostic with N=%d: forcing shared cache mode\n", N);
        cache_shared = 1;
    }

    if (is_first)
        logmsg("cache: load=%d save=%d shared=%d kv_heads=%d/%d mode=%s\n",
               cache_load, cache_save, cache_shared, m->tp_kv_head_count, m->n_kv_heads,
               cache_shared ? "shared" : "per-rank");

    if (cache_path_env[0]) {
        snprintf(cache_path, sizeof(cache_path), "%s", cache_path_env);
        have_cache_path = 1;
    } else if (cache_dir[0]) {
        if (tp_build_cache_path(cache_path, sizeof(cache_path), m, model_path, NULL, cache_dir, cache_tag, MyRank, cache_shared) == 0)
            have_cache_path = 1;
    }

    if (cache_path_env[0] || cache_dir[0]) {
        if (!cache_load)  cache_load = 1;
        if (!cache_save)  cache_save = 1;
    }

    if (cache_load && have_cache_path) {
        int64_t probe_pos = -1; int32_t probe_next = -1;
        int cache_rank = cache_shared ? -1 : MyRank;
        int probe_ok = (tp_checkpoint_probe(m, cache_path, &probe_pos, &probe_next,
                                          N, cache_rank, max_seq, cache_shared) == 0);
        float all_probe = probe_ok ? 1.0f : 0.0f;
        tp_allreduce_sum(&c, &all_probe, 1);
        int all_ranks_probe = ((int)(all_probe + 0.5f) == N);
        if (all_ranks_probe) {
            cache_loaded = (tp_checkpoint_load(m, cache_path, &ck_pos, &ck_next,
                                             cache_rank, N, cache_shared) == 0);
            if (cache_loaded) cache_prefill_used = 1;
            if (!cache_loaded && is_first)
                logmsg("TP cache load failed on rank0 after pre-check: %s\n", cache_path);
        } else if (is_first) {
            logmsg("TP cache disabled for this run: checkpoint mismatch %s\n", cache_path);
            logmsg("  TP cache pre-check votes: this=%d all=%g\n", probe_ok ? 1 : 0, all_probe);
        }
    } else if (is_first && (cache_load || cache_save)) {
        logmsg("TP cache requested but no path available (TP_CACHE_PATH / TP_CACHE_DIR)\n");
    }

    double t0_all = now_sec();
    int n_gen = 0;
    double t_prefill = 0.0;
    double t_fwd = 0.0;
    double t_comm = 0.0;
    long pcnt = 0, ar_calls = 0;
    int32_t in_tok = (P > 0) ? ptoks[0] : 0;
    int prefill_gemm_used = 0;
    int prefill_tokens = 0;
    int prefill_from = 0;

    if (P <= 0) die("prompt token count must be positive", -1);

    int64_t final_cache_pos = 0;

    if (cache_loaded && ck_pos > P && !token_in_vocab(vocab, ck_next)) {
        if (is_first) {
            logmsg("TP cache next-token invalid (%d), disabling cache usage\n", ck_next);
        }
        cache_loaded = 0;
        cache_prefill_used = 0;
        ck_pos = -1;
        ck_next = -1;
    }

    if (cache_loaded && ck_pos > 0) {
        prefill_from = (int)(ck_pos < P ? ck_pos : P);
        cache_prefill_used = 1;
    } else {
        cache_prefill_used = 0;
        prefill_from = 0;
    }
    if (cache_loaded && prefill_from == P && !token_in_vocab(vocab, ck_next)) {
        cache_loaded = 0;
        cache_prefill_used = 0;
        prefill_from = 0;
        ck_pos = -1;
        ck_next = -1;
    }

    if (cache_loaded && prefill_from > 0) {
        cache_prefill_skipped = prefill_from;
        if (is_first) {
            logmsg("[cache] resume at pos=%lld/%d, next=%d\n", (long long)ck_pos, P, ck_next);
        }
    }

    if (prefill_from < P) {
        prefill_tokens = P - prefill_from;
        if (dry_prefill) {
            double pf0 = now_sec();
            g_ar_secs = 0.0; g_ar_calls = 0;
            for (int p = prefill_from; p < P; p++) {
                tp_dry_token_walk(m, &c, ptoks[p], dry_work_reps, dry_ar_steps);
                if (p == P - 1) in_tok = ptoks[p];
            }
            t_prefill = now_sec() - pf0;
            t_comm += g_ar_secs;
            ar_calls += g_ar_calls;
            prefill_gemm_used = 1;
            in_tok = (P > 0) ? tp_next_token_synth(in_tok, 1, m->n_vocab) : in_tok;
        } else if (do_prefill_gemm) {
            double pf0 = now_sec();
            float *lg = transformer_prefill_gemm(m, ptoks + prefill_from, prefill_tokens, prefill_from);
            if (lg) {
                double ar_step = 0.0; long ar_calls_step = 0;
                in_tok = sample_argmax(m, lg, &c, &ar_step, &ar_calls_step);
                t_prefill = now_sec() - pf0;
                t_comm += ar_step; ar_calls += ar_calls_step;
                prefill_gemm_used = 1;
            } else if (is_first) {
                fprintf(stderr, "warning: prefill_gemm unavailable; fallback to per-token prefill\n");
            }
        }
        if (!prefill_gemm_used) {
            double pf0 = now_sec();
            for (int p = prefill_from; p < P; p++) {
                transformer_embed_token(m, ptoks[p]);
                double _ta = now_sec();
                transformer_forward_partial(m, p, 0, n_layers);
                t_fwd += now_sec() - _ta;
            }
            float *lg = transformer_compute_logits(m);  /* only final prompt token matters */
            if (!lg) die("transformer_compute_logits after prefill", -1);
            double ar_step = 0.0; long ar_calls_step = 0;
            in_tok = sample_argmax(m, lg, &c, &ar_step, &ar_calls_step);
            t_prefill = now_sec() - pf0;
            t_comm += ar_step; ar_calls += ar_calls_step;
        }
    } else {
        if (cache_loaded && token_in_vocab(vocab, ck_next)) {
            in_tok = ck_next;
            if (is_first) logmsg("[cache] prompt cache fully covered, using ck_next=%d\n", in_tok);
        } else if (cache_loaded && P > 0) {
            double pf0 = now_sec();
            float *lg = transformer_compute_logits(m);
            if (lg) {
                double ar_step = 0.0; long ar_calls_step = 0;
                in_tok = sample_argmax(m, lg, &c, &ar_step, &ar_calls_step);
                t_prefill = now_sec() - pf0;
                t_comm += ar_step;
                ar_calls += ar_calls_step;
            } else if (is_first) {
                fprintf(stderr, "warning: cannot refresh cache-only next token (no logits)\n");
            }
            if (is_first) logmsg("[cache] no valid cached next token, defaulting to sampled prompt tail\n");
        } else if (P > 0) {
            in_tok = ptoks[P - 1];
            if (is_first) logmsg("[cache] no valid cached next token, defaulting to prompt tail\n");
        }
    }

    if (prefill_only) {
        if (is_first) {
            double prefill_tps = (prefill_tokens > 0 && t_prefill > 0.0) ? (double)prefill_tokens / t_prefill : 0.0;
            logmsg("prefill-only: P=%d prefill=%.3f s  %.2f tok/s  prefill_gemm=%d\n",
                   prefill_tokens, t_prefill, prefill_tps, prefill_gemm_used);
            if (cache_prefill_used) {
                logmsg("prefill resumed from cache: skip=%d tokens, start=%d\n", cache_prefill_skipped, prefill_from);
            } else {
                logmsg("prefill fully computed (%d tokens)\n", prefill_tokens);
            }
        }
        final_cache_pos = prefill_from + prefill_tokens;
        goto done;
    }

    t_fwd = 0.0; t_comm = 0.0; ar_calls = 0; pcnt = 0;  /* decode-only stats */
    int64_t decode_start = (cache_loaded && ck_pos > P) ? (int64_t)ck_pos : (int64_t)P;
    if (decode_start >= max_seq) die("checkpoint state exceeds max_seq", -1);
    int decode_started = 0;
    transformer_pool_profile_reset();
    final_cache_pos = decode_start;
    for (int p = (int)decode_start; ; p++) {
        double _ta = now_sec();
        double ar_step = 0.0; long ar_calls_step = 0;
        int32_t nt = 0;

        if (dry_decode) {
            g_ar_secs = 0.0; g_ar_calls = 0;
            tp_dry_token_walk(m, &c, in_tok, dry_work_reps, dry_ar_steps);
            ar_step = 0.0;
            ar_calls_step = g_ar_calls;
            nt = tp_next_token_synth(in_tok, dry_token_step, m->n_vocab);
        } else {
            transformer_embed_token(m, in_tok);
            g_ar_secs = 0.0; g_ar_calls = 0;
            transformer_forward_partial(m, p, 0, n_layers);
            float *lg = transformer_compute_logits(m);
            nt = sample_argmax(m, lg, &c, &ar_step, &ar_calls_step);
        }

        double _tb = now_sec();

        if (!decode_started) { transformer_pool_profile_reset(); decode_started = 1; }

        int eos = (dry_decode ? 0 : (nt == bpe_eos_id(vocab) || nt == bpe_eot_id(vocab)));
        int stop_eos = eos && !ignore_eos;
        if (g_tokdump) { fprintf(g_tokdump, "%d\n", (int)nt); fflush(g_tokdump); }
        if (!eos && is_first && !dry_decode) print_token(vocab, nt);
        n_gen++;

        t_fwd += _tb - _ta;
        t_comm += g_ar_secs + ar_step;
        ar_calls += g_ar_calls + ar_calls_step;
        pcnt++;
        final_cache_pos = (int64_t)p + 1;
        if (g_curve) { fprintf(g_curve, "%d %d %.3f %.3f\n",
                               p, p + 1, 1000.0 * (_tb - _ta), 1000.0 * (g_ar_secs + ar_step));
                         fflush(g_curve); }
        if (stop_eos || n_gen >= max_gen) break;
        in_tok = nt;
    }

done:
    barrier();
    {   char pn[64]; snprintf(pn, sizeof pn, "tp_perf_rank%02d.txt", MyRank);
        FILE *pf2 = fopen(pn, "w");
        if (pf2) {
            double per_tok = pcnt ? 1000.0 * t_fwd / pcnt : 0.0;   /* ms/tok total */
            double comm_ms = pcnt ? 1000.0 * t_comm / pcnt : 0.0;  /* ms/tok all-reduce */
            double comp_ms = per_tok - comm_ms;                     /* ms/tok compute */
            double wb      = tp_local_weight_bytes(m);              /* bytes/tok, this rank */
            double gbps    = comp_ms > 0.0 ? (wb / 1e9) / (comp_ms / 1000.0) : 0.0;
            fprintf(pf2, "rank %d: prefill=%.4fs (%d tok, %.2f tok/s) prefill_gemm=%d\n",
                    MyRank, t_prefill, prefill_tokens, prefill_tokens > 0 ? (double)prefill_tokens / t_prefill : 0.0, prefill_gemm_used);
            fprintf(pf2, "rank %d decode: fwd=%.4fs cnt=%ld  per_tok=%.2fms "
                        "compute=%.2fms comm=%.2fms (%.0f%%) ar_calls/tok=%.0f "
                        "wbytes=%.2fGB eff=%.0fGB/s\n",
                    MyRank, t_fwd, pcnt, per_tok, comp_ms, comm_ms,
                    per_tok > 0 ? 100.0 * comm_ms / per_tok : 0.0,
                    pcnt ? (double)ar_calls / pcnt : 0.0, wb / 1e9, gbps);
            fclose(pf2);
        }
    }
    double t_total = now_sec() - t0_all;
    if (is_first) {
        double t_dec = t_total - t_prefill;
        logmsg("\n\n=== done: %d tokens generated ===\n", n_gen);
        logmsg("prefill(%d tok)=%.3f s (%.2f tok/s)%s\n", prefill_tokens, t_prefill,
               prefill_tokens > 0 ? (double)prefill_tokens / t_prefill : 0.0,
               cache_prefill_used ? " [from cache]" : "");
        if (cache_prefill_used && cache_prefill_skipped > 0)
            logmsg("cache-skipped=%d prompt toks, ck_pos=%lld\n", cache_prefill_skipped, (long long)ck_pos);
        logmsg("decode(%d tok)=%.3f s = %.2f tok/s\n", n_gen, t_dec, n_gen > 0 ? n_gen / t_dec : 0.0);
    }

    if (cache_save && have_cache_path) {
        int64_t save_pos = final_cache_pos;
        if (prefill_only) save_pos = prefill_from + prefill_tokens;
        int32_t save_next = in_tok;
        if (!token_in_vocab(vocab, save_next)) {
            if (is_first) {
                logmsg("[cache] skip save: generated next token invalid (%d)\n", save_next);
            }
        } else {
            if (cache_shared && MyRank != 0) {
                if (is_first) {
                    logmsg("TP cache shared skip save on rank %d (rank0 writes)\n", MyRank);
                }
            } else if (tp_checkpoint_write(m, cache_path, save_pos, save_next, cache_shared) == 0) {
                if (is_first) logmsg("TP cache saved: %s pos=%lld next=%d\n", cache_path, (long long)save_pos, save_next);
            } else if (is_first) {
                logmsg("TP cache save failed: %s\n", cache_path);
            }
        }
    } else if (is_first && cache_save && !have_cache_path) {
        logmsg("TP cache save requested but no path available\n");
    }

    /* transformer_free joins+shuts down the worker pool, which under
     * -DTF_POOL_PROFILE emits the per-dispatch work/wait + per-tid
     * matvec/barrier/serial/attn decode breakdown to stderr. */
    transformer_free(m);
    tp_comm_free(&c);
    utofu_dereg_mem(Vcq, Base, 0);
    utofu_free_vcq(Vcq);
    free(ptoks); free(Region);
    if (g_log) fclose(g_log);
    if (g_curve) fclose(g_curve);
    if (g_tokdump) fclose(g_tokdump);
    return 0;
}
