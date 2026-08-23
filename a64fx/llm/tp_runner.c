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
#ifdef _OPENMP
#include <omp.h>
#endif

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
    int is_ssm = m->is_hybrid && m->layers && m->layers[l].is_ssm;
    if (!is_ssm && m->key_cache && m->value_cache &&
        m->key_cache[l] && m->value_cache[l] &&
        !(m->is_gemma4 && m->layers && m->layers[l].shared_kv_source >= 0)) {
        size_t seq = tf_cache_seq_len_for_layer(m, l);
        int kv_dim = tf_cache_kv_dim_for_layer(m, l);
        x.key_bytes = (int64_t)seq * kv_dim * (int64_t)m->kv_elem_bytes;
        x.value_bytes = x.key_bytes;
        if (m->key_scales && m->key_scales[l])
            x.key_scale_bytes = (int64_t)seq * (int64_t)m->n_kv_heads * (int64_t)sizeof(float);
        if (m->value_scales && m->value_scales[l])
            x.value_scale_bytes = x.key_scale_bytes;
    }
    if (m->conv_state && m->conv_state[l] && m->ssm_conv_kernel > 1 && m->ssm_qkv_dim > 0) {
        x.conv_state_bytes = (int64_t)(m->ssm_conv_kernel - 1) * (int64_t)m->ssm_qkv_dim * (int64_t)sizeof(float);
    }
    if (m->recurrent_state && m->recurrent_state[l] && m->ssm_dt_rank > 0 && m->ssm_d_state > 0) {
        x.recurrent_state_bytes = (int64_t)m->ssm_dt_rank * (int64_t)m->ssm_d_state * (int64_t)m->ssm_d_state * (int64_t)sizeof(float);
    }
    return x;
}

/* In-memory checkpoint for speculative Qwen hybrid verification.  Attention
 * KV at rejected future positions is overwritten on replay, but the SSM
 * convolution ring and recurrent matrices are destructive and must be
 * restored exactly.  This buffer is rank-local anonymous memory in HBM2. */
typedef struct {
    unsigned char *data;
    size_t *conv_off;
    size_t *rec_off;
    int *conv_pos;
    size_t bytes;
    int n_layers;
} tp_spec_state;

static int tp_spec_state_init(tp_spec_state *s, const transformer_model *m) {
    memset(s, 0, sizeof(*s));
    if (!m || !m->is_hybrid || m->n_layers <= 0) return -1;
    s->n_layers = m->n_layers;
    s->conv_off = (size_t *)malloc((size_t)m->n_layers * sizeof(size_t));
    s->rec_off = (size_t *)malloc((size_t)m->n_layers * sizeof(size_t));
    s->conv_pos = (int *)malloc((size_t)m->n_layers * sizeof(int));
    if (!s->conv_off || !s->rec_off || !s->conv_pos) return -1;
    for (int l = 0; l < m->n_layers; l++) {
        tp_layer_cache lc = tp_layer_cache_bytes(m, l);
        s->conv_off[l] = s->bytes;
        s->bytes += (size_t)lc.conv_state_bytes;
        s->rec_off[l] = s->bytes;
        s->bytes += (size_t)lc.recurrent_state_bytes;
    }
    if (posix_memalign((void **)&s->data, 256, s->bytes ? s->bytes : 256) != 0)
        return -1;
    return 0;
}

static void tp_spec_state_save(tp_spec_state *s, const transformer_model *m) {
    for (int l = 0; l < s->n_layers; l++) {
        tp_layer_cache lc = tp_layer_cache_bytes(m, l);
        if (lc.conv_state_bytes)
            memcpy(s->data + s->conv_off[l], m->conv_state[l],
                   (size_t)lc.conv_state_bytes);
        if (lc.recurrent_state_bytes)
            memcpy(s->data + s->rec_off[l], m->recurrent_state[l],
                   (size_t)lc.recurrent_state_bytes);
        s->conv_pos[l] = m->conv_state_pos ? m->conv_state_pos[l] : 0;
    }
}

static void tp_spec_state_restore(const tp_spec_state *s, transformer_model *m) {
    for (int l = 0; l < s->n_layers; l++) {
        tp_layer_cache lc = tp_layer_cache_bytes(m, l);
        if (lc.conv_state_bytes)
            memcpy(m->conv_state[l], s->data + s->conv_off[l],
                   (size_t)lc.conv_state_bytes);
        if (lc.recurrent_state_bytes)
            memcpy(m->recurrent_state[l], s->data + s->rec_off[l],
                   (size_t)lc.recurrent_state_bytes);
        if (m->conv_state_pos) m->conv_state_pos[l] = s->conv_pos[l];
    }
}

static void tp_spec_state_free(tp_spec_state *s) {
    if (!s) return;
    free(s->data); free(s->conv_off); free(s->rec_off); free(s->conv_pos);
    memset(s, 0, sizeof(*s));
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

/* Load a TP12 prefill checkpoint into a TP4 decode rank.  TP12 keeps full KV
 * for the four attention heads and owns four SSM V heads; TP4 owns one KV head
 * and twelve SSM V heads.  The checkpoint files are intentionally per-source
 * rank so the repartitioner can combine them without an MPI dependency. */
static int tp_checkpoint_load_repartition(transformer_model *m,
                                          const char *root,
                                          const char *model_path,
                                          const char *tag,
                                          int src_size, int target_rank,
                                          int64_t *cache_pos,
                                          int32_t *next_token) {
    if (!m || !root || !*root || src_size != 12 || m->tp_size != 4) return -1;
    FILE *fs[12] = {0};
    tp_cache_header hs[12];
    for (int r = 0; r < src_size; r++) {
        char p[PATH_MAX];
        if (tp_build_cache_path(p, sizeof(p), m, model_path, NULL, root, tag, r, 0) != 0)
            goto fail;
        fs[r] = fopen(p, "rb");
        if (!fs[r] || read_full(fs[r], &hs[r], sizeof(hs[r])) != 0) {
            fprintf(stderr, "tp repartition open/read failed rank=%d path=%s errno=%d\n", r, p, errno);
            goto fail;
        }
        if (hs[r].magic != TP_CACHE_MAGIC || hs[r].version != TP_CACHE_VERSION ||
            hs[r].tp_size != src_size || hs[r].n_layers != m->n_layers ||
            hs[r].n_embd != m->n_embd || hs[r].n_heads != m->n_heads * m->tp_size ||
            hs[r].n_kv_heads != m->n_kv_heads * m->tp_size || hs[r].n_head_dim != m->head_dim ||
            !hs[r].is_hybrid || hs[r].cache_pos < 0 || hs[r].cache_pos > m->max_seq_len) {
            fprintf(stderr, "tp repartition header mismatch rank=%d magic=%x ver=%u tp=%d layers=%d hybrid=%d pos=%d\n",
                    r, hs[r].magic, hs[r].version, hs[r].tp_size, hs[r].n_layers,
                    hs[r].is_hybrid, hs[r].cache_pos);
            goto fail;
        }
        if (r > 0 && (hs[r].cache_pos != hs[0].cache_pos || hs[r].next_token != hs[0].next_token))
            goto fail;
    }
    if (cache_pos) *cache_pos = hs[0].cache_pos;
    if (next_token) *next_token = hs[0].next_token;

    int src_dt = 48 / src_size;
    int dst_dt = m->ssm_dt_rank;
    int ds = m->ssm_d_state;
    int qk = 2 * m->ssm_n_group * ds;
    int src_qkv = qk + src_dt * ds;
    int dst_qkv = qk + dst_dt * ds;
    int dst_kv_heads = m->tp_kv_head_count;
    int dst_kv_dim = dst_kv_heads * m->head_dim;
    size_t src_seq = (size_t)hs[0].max_seq_len;
    for (int l = 0; l < m->n_layers; l++) {
        tp_layer_cache lc[12];
        for (int r = 0; r < src_size; r++) {
            if (read_i64(fs[r], &lc[r].key_bytes) != 0 ||
                read_i64(fs[r], &lc[r].value_bytes) != 0 ||
                read_i64(fs[r], &lc[r].key_scale_bytes) != 0 ||
                read_i64(fs[r], &lc[r].value_scale_bytes) != 0 ||
                read_i64(fs[r], &lc[r].conv_state_bytes) != 0 ||
                read_i64(fs[r], &lc[r].recurrent_state_bytes) != 0) goto fail;
        }
        int is_attn = !(m->layers[l].is_ssm);
        if (is_attn && m->key_cache && m->key_cache[l] && lc[0].key_bytes > 0) {
            size_t src_row = (size_t)m->n_kv_heads * m->head_dim * m->kv_elem_bytes;
            size_t dst_row = (size_t)dst_kv_dim * m->kv_elem_bytes;
            uint8_t *all = (uint8_t *)malloc((size_t)lc[0].key_bytes);
            if (!all || read_full(fs[0], all, (size_t)lc[0].key_bytes) != 0) { free(all); goto fail; }
            int head = m->tp_kv_head_base;
            size_t rows = (size_t)lc[0].key_bytes / src_row;
            if (rows > src_seq) rows = src_seq;
            for (size_t t = 0; t < rows; t++) {
                memcpy((uint8_t *)m->key_cache[l] + t * dst_row,
                       all + t * src_row + (size_t)head * m->head_dim * m->kv_elem_bytes, dst_row);
            }
            free(all);
            for (int r = 1; r < src_size; r++)
                if (lc[r].key_bytes && fseek(fs[r], (long)lc[r].key_bytes, SEEK_CUR) != 0) goto fail;
        } else {
            for (int r = 0; r < src_size; r++)
                if (lc[r].key_bytes && fseek(fs[r], (long)lc[r].key_bytes, SEEK_CUR) != 0) goto fail;
        }
        for (int r = 0; r < src_size; r++) {
            if (lc[r].value_bytes && (is_attn && r == 0 && m->value_cache && m->value_cache[l])) {
                size_t src_row = (size_t)m->n_kv_heads * m->head_dim * m->kv_elem_bytes;
                size_t dst_row = (size_t)dst_kv_dim * m->kv_elem_bytes;
                uint8_t *all = (uint8_t *)malloc((size_t)lc[r].value_bytes);
                if (!all || read_full(fs[r], all, (size_t)lc[r].value_bytes) != 0) { free(all); goto fail; }
                int head = m->tp_kv_head_base;
                size_t rows = (size_t)lc[r].value_bytes / src_row;
                if (rows > src_seq) rows = src_seq;
                for (size_t t = 0; t < rows; t++) {
                    memcpy((uint8_t *)m->value_cache[l] + t * dst_row,
                           all + t * src_row + (size_t)head * m->head_dim * m->kv_elem_bytes, dst_row);
                }
                free(all);
            } else if (lc[r].value_bytes && fseek(fs[r], (long)lc[r].value_bytes, SEEK_CUR) != 0) goto fail;
            if (lc[r].key_scale_bytes && fseek(fs[r], (long)lc[r].key_scale_bytes, SEEK_CUR) != 0) goto fail;
            if (lc[r].value_scale_bytes && fseek(fs[r], (long)lc[r].value_scale_bytes, SEEK_CUR) != 0) goto fail;
        }
        if (!is_attn) {
            float *conv[12] = {0}, *rec[12] = {0};
            for (int r = 0; r < src_size; r++) {
                if (lc[r].conv_state_bytes != (int64_t)(m->ssm_conv_kernel - 1) * src_qkv * sizeof(float) ||
                    lc[r].recurrent_state_bytes != (int64_t)src_dt * ds * ds * sizeof(float)) {
                    for (int j = 0; j < src_size; j++) { free(conv[j]); free(rec[j]); }
                    goto fail;
                }
                conv[r] = (float *)malloc((size_t)lc[r].conv_state_bytes);
                rec[r] = (float *)malloc((size_t)lc[r].recurrent_state_bytes);
                if (!conv[r] || !rec[r] ||
                    read_full(fs[r], conv[r], (size_t)lc[r].conv_state_bytes) != 0 ||
                    read_full(fs[r], rec[r], (size_t)lc[r].recurrent_state_bytes) != 0) {
                    for (int j = 0; j < src_size; j++) { free(conv[j]); free(rec[j]); }
                    goto fail;
                }
            }
            if (m->conv_state && m->conv_state[l]) {
                int nh = m->ssm_conv_kernel - 1;
                for (int h = 0; h < nh; h++) {
                    memcpy(m->conv_state[l] + (size_t)h * dst_qkv,
                           conv[0] + (size_t)h * src_qkv, (size_t)qk * sizeof(float));
                    float *d = m->conv_state[l] + (size_t)h * dst_qkv + qk;
                    for (int j = 0; j < 3; j++) {
                        int sr = target_rank * 3 + j;
                        memcpy(d + (size_t)j * src_dt * ds,
                               conv[sr] + (size_t)h * src_qkv + qk,
                               (size_t)src_dt * ds * sizeof(float));
                    }
                }
            }
            if (m->recurrent_state && m->recurrent_state[l]) {
                for (int j = 0; j < 3; j++) {
                    int sr = target_rank * 3 + j;
                    memcpy(m->recurrent_state[l] + (size_t)j * src_dt * ds * ds,
                           rec[sr], (size_t)src_dt * ds * ds * sizeof(float));
                }
            }
            for (int r = 0; r < src_size; r++) { free(conv[r]); free(rec[r]); }
        } else {
            for (int r = 0; r < src_size; r++) {
                if (lc[r].conv_state_bytes && fseek(fs[r], (long)lc[r].conv_state_bytes, SEEK_CUR) != 0) goto fail;
                if (lc[r].recurrent_state_bytes && fseek(fs[r], (long)lc[r].recurrent_state_bytes, SEEK_CUR) != 0) goto fail;
            }
        }
    }
    if (m->conv_state_pos) {
        for (int r = 0; r < src_size; r++) {
            for (int l = 0; l < m->n_layers; l++) {
                int32_t p;
                if (read_i32(fs[r], &p) != 0) goto fail;
                if (r == 0) m->conv_state_pos[l] = p;
            }
        }
    }
    for (int r = 0; r < src_size; r++) fclose(fs[r]);
    if (target_rank == 0)
        fprintf(stderr, "tp cache repartition complete: source TP%d -> target TP%d pos=%d\n",
                src_size, m->tp_size, hs[0].cache_pos);
    return 0;

fail:
    fprintf(stderr, "tp repartition failed rank=%d\n", target_rank);
    for (int r = 0; r < src_size; r++) if (fs[r]) fclose(fs[r]);
    return -1;
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

static void sample_argmax_n(transformer_model *m, float *logits, int stride,
                            int n, int32_t *tokens, tp_comm *c,
                            double *ar_secs_out, long *ar_calls_out) {
    float vi[10];
    int nloc = m->tp_vocab_sharded ? m->tp_vocab_loc : m->n_vocab;
    for (int k = 0; k < n; k++) {
        float best = -1e30f;
        int32_t nt = 0;
        float *lg = logits + (size_t)k * stride;
        for (int v = 0; v < nloc; v++) {
            if (lg[v] > best) { best = lg[v]; nt = v; }
        }
        nt += m->tp_vocab_lo;
        vi[2*k] = best;
        memcpy(&vi[2*k+1], &nt, sizeof(nt));
    }
    if (m->tp_vocab_sharded) {
        double t0 = now_sec();
        tp_allreduce_argmax_n(c, vi, n);
        if (ar_secs_out) *ar_secs_out += now_sec() - t0;
        if (ar_calls_out) (*ar_calls_out)++;
    }
    for (int k = 0; k < n; k++)
        memcpy(&tokens[k], &vi[2*k+1], sizeof(tokens[k]));
}

static void print_token(const bpe_vocab *vocab, int32_t nt) {
    const char *s = bpe_token_to_str(vocab, nt);
    if (!s) return;
    int dec_len = 0;
    char *dec = bpe_byte_decode(s, (int)strlen(s), &dec_len);
    const char *out = dec ? dec : s;
    int len = dec ? dec_len : (int)strlen(s);
    int buffered = getenv("TP_BUFFER_OUTPUT") != NULL;
    if (g_log) { fwrite(out, 1, len, g_log); if (!buffered) fflush(g_log); }
    fwrite(out, 1, len, stdout); if (!buffered) fflush(stdout);
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
    int  perf_warmup       = (int)envl("TP_PERF_WARMUP", 0);
    int  spec_k            = (int)envl("TP_SPEC_K", 0);
    /* Opt in until the batched-vs-token greedy gate is exact for long runs. */
    int  mtp_batch         = envb("TP_MTP_BATCH", 0);
    int  llm_threads       = (int)envl("LLM_THREADS", 48);
    if (spec_k < 0 || spec_k > 5) die("TP_SPEC_K must be in [0,5]", -1);
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
    int  cache_autosave     = envb_opt("TP_CACHE_AUTOSAVE", 1);
    int  cache_repartition_from = (int)envl_opt("TP_CACHE_REPARTITION_FROM", 0);
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
    /* A complete TP stage owns every tensor used by decode.  Parse only
     * source metadata in that mode; never mmap or allocate the 54 GB GGUF. */
    gguf_context *gguf = gguf_open_multi(model_path,
                                         envs_opt("TP_STAGE_DIR", "")[0] ? 3 : 0);
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
        char *text = envb_opt("TP_RAW_PROMPT", 0) ? strdup(prompt)
                                                   : build_chat_text(prompt, NULL);
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
    int prompt_token_limit = (int)envl_opt("TP_PROMPT_TOKEN_LIMIT", 0);
    if (prompt_token_limit > 0 && P > prompt_token_limit) P = prompt_token_limit;

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
    const char *tp_stage_dir = envs_opt("TP_STAGE_DIR", "");
    if ((spec_k > 0 || envs_opt("TP_STAGE_DIR", "")[0]) && m->nextn.loaded &&
        transformer_tp_slice_nextn(m, MyRank, N) != 0)
        die("transformer_tp_slice_nextn", -1);
    size_t tp_stage_bytes = 0;
    if (tp_stage_dir[0]) {
        tp_stage_bytes = transformer_tp_load_stage(m, tp_stage_dir, MyRank, N);
        if (!tp_stage_bytes) die("transformer_tp_load_stage", -1);
    }
    /* Q8 TP shards can be converted once to row-major W8A8.  This selects the
     * native SVE SDOT matvec path; keep it opt-in because it is lossy and only
     * applies to Q8_0 models. */
    const char *q8_mode_env = getenv("TP_Q8_MODE");
    int is_q8_model = m->layers &&
        (m->layers[0].attn_q.type == GGML_TYPE_Q8_0 ||
         m->layers[0].ssm_qkv.type == GGML_TYPE_Q8_0 ||
         m->layers[0].ffn_gate.type == GGML_TYPE_Q8_0);
    if (q8_mode_env && *q8_mode_env && is_q8_model) {
        int q8_mode = !strcmp(q8_mode_env, "row") ? 1 :
                      !strcmp(q8_mode_env, "block64") ? 2 :
                      !strcmp(q8_mode_env, "block64-ffn") ? 3 :
                      !strcmp(q8_mode_env, "block64-exact") ? 4 : 0;
        size_t q8_bytes = transformer_materialize_q8_decode(m, gguf, q8_mode);
        if (!q8_bytes) die("transformer_materialize_q8_decode", -1);
        if (MyRank == 0) logmsg("TP_Q8_MODE=%s resident=%.3fGB\n", q8_mode_env,
                                (double)q8_bytes / 1e9);
    }
    const char *q8_expand = getenv("TP_Q8_EXPAND_BF16");
    if (q8_expand && *q8_expand && atoi(q8_expand) != 0 && is_q8_model) {
        if (q8_mode_env && *q8_mode_env)
            die("TP_Q8_EXPAND_BF16 conflicts with TP_Q8_MODE", -1);
        if (getenv("TP_Q8_VERIFY") && *getenv("TP_Q8_VERIFY"))
            die("TP_Q8_EXPAND_BF16 conflicts with TP_Q8_VERIFY", -1);
        int nextn_mask = spec_k > 0 ? (int)envl("TP_Q8_EXPAND_NEXTN_MASK", 0) : 0;
        if (nextn_mask & ~0x1ff)
            die("TP_Q8_EXPAND_NEXTN_MASK contains unsupported bits", -1);
        if (!tp_stage_bytes)
            die("TP_Q8_EXPAND_BF16 requires a complete anonymous TP stage", -1);
        size_t expanded = transformer_expand_q8_bf16_pv_range(
            m, 0, m->n_layers, 1, nextn_mask);
        if (!expanded) die("Q8 to BF16 PV expansion", -1);
        if (MyRank == 0) logmsg("TP_Q8_EXPAND_BF16=1 expanded=%.3fGB\n",
                                (double)expanded / 1e9);
    }
    const char *q8_verify = getenv("TP_Q8_VERIFY");
    if (q8_verify && *q8_verify && is_q8_model) {
        size_t verify_bytes = 0;
        if (!strcmp(q8_verify, "q8v2"))
            verify_bytes = transformer_prepack_q8v2_range(m, 0, m->n_layers);
        else
            die("TP_Q8_VERIFY must be q8v2", -1);
        if (!verify_bytes) die("Q8 verifier prepack", -1);
        if (MyRank == 0) logmsg("TP_Q8_VERIFY=%s packed=%.3fGB\n",
                                q8_verify, (double)verify_bytes / 1e9);
    }
    if (getenv("TP_INT8_MODE") && *getenv("TP_INT8_MODE")) {
        if (m->layers[0].ffn_gate.type != GGML_TYPE_BF16)
            die("TP_INT8_MODE requires BF16 TP shards", -1);
        if (!strcmp(getenv("TP_INT8_MODE"), "block64-ffn"))
            transformer_prepack_int8_block64_ffn(m);
        else if (!strcmp(getenv("TP_INT8_MODE"), "block64"))
            transformer_prepack_int8_block64(m);
        else
            transformer_prepack_int8(m);
        if (MyRank == 0) logmsg("TP_INT8_MODE=%s enabled for resident BF16 projections\n",
                                getenv("TP_INT8_MODE"));
    }
    if (!tp_stage_bytes && spec_k > 0 && m->nextn.loaded &&
        envb("TP_MATERIALIZE_NEXTN", 1)) {
        size_t nextn_bytes = transformer_materialize_nextn(m);
        if (!nextn_bytes) die("transformer_materialize_nextn", -1);
    }
    double tp0 = now_sec();
    if (!envb("TF_NO_PANEL", 0))
        transformer_build_panels(m);              /* repack the local shard */
    double tp1 = now_sec();
    if (dry_mode) {
        if (dry_ar_steps <= 0) dry_ar_steps = 2;
        if (dry_work_reps < 0) dry_work_reps = 0;
        if (dry_token_step <= 0) dry_token_step = 1;
    }
    {   char tname[64]; snprintf(tname, sizeof tname, "tp_load_rank%02d.txt", MyRank);
        FILE *tf = fopen(tname, "w");
        if (tf) { fprintf(tf, "rank %d: transformer_load=%.1fs build_panels=%.1fs stage=%.3fGB "
                          "(n_heads=%d n_kv=%d n_ff=%d hybrid=%d ssm_shard=%d "
                          "ssm_dt=%d ssm_qkv=%d ssm_d_inner=%d head_off=%d "
                          "vocab_shard=%d vocab_loc=%d vocab_lo=%d)\n",
                          MyRank, tl1 - tl0, tp1 - tp0, (double)tp_stage_bytes/1e9,
                          m->n_heads, m->n_kv_heads, m->n_ff,
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

    int null_stream_passes = (int)envl("TP_NULL_STREAM_PASSES", 0);
    if (null_stream_passes > 0) {
        double bw = transformer_null_stream_bench(m, null_stream_passes);
        barrier();
        char pn[64];
        snprintf(pn, sizeof pn, "tp_null_stream_rank%02d.txt", MyRank);
        FILE *pf = fopen(pn, "w");
        if (pf) {
            fprintf(pf, "rank %d: passes=%d stage=%.3fGB bandwidth=%.1fGB/s\n",
                    MyRank, null_stream_passes, (double)tp_stage_bytes / 1e9, bw);
            fclose(pf);
        }
        if (is_first)
            logmsg("TP null-batched: passes=%d local_stage=%.3f GB bandwidth=%.1f GB/s/rank\n",
                   null_stream_passes, (double)tp_stage_bytes / 1e9, bw);
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

    if (cache_autosave && (cache_path_env[0] || cache_dir[0])) {
        if (!cache_load)  cache_load = 1;
        if (!cache_save)  cache_save = 1;
    }

    if (cache_load && have_cache_path) {
        int64_t probe_pos = -1; int32_t probe_next = -1;
        int cache_rank = cache_shared ? -1 : MyRank;
        int probe_ok;
        if (cache_repartition_from > 0) {
            if (is_first) fprintf(stderr, "tp cache repartition source=%d dir=%s target=%d\n",
                                  cache_repartition_from, cache_dir, N);
            probe_ok = cache_dir[0] && cache_repartition_from != N &&
                tp_checkpoint_load_repartition(m, cache_dir, model_path, cache_tag,
                                               cache_repartition_from, MyRank,
                                               &probe_pos, &probe_next) == 0;
        } else {
            probe_ok = (tp_checkpoint_probe(m, cache_path, &probe_pos, &probe_next,
                                             N, cache_rank, max_seq, cache_shared) == 0);
        }
        float all_probe = probe_ok ? 1.0f : 0.0f;
        tp_allreduce_sum(&c, &all_probe, 1);
        int all_ranks_probe = ((int)(all_probe + 0.5f) == N);
        if (all_ranks_probe) {
            if (cache_repartition_from > 0) {
                ck_pos = probe_pos; ck_next = probe_next; cache_loaded = 1;
            } else {
                cache_loaded = (tp_checkpoint_load(m, cache_path, &ck_pos, &ck_next,
                                                 cache_rank, N, cache_shared) == 0);
            }
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
    long mtp_match = 0, mtp_total = 0;
    long mtp_horizon_match[5] = {0}, mtp_horizon_total[5] = {0};
    long mtp_teacher_match = 0, mtp_teacher_total = 0;
    long mtp_teacher_offset_match[7] = {0};  /* expected token index p + [-1..5] */
    int32_t *mtp_token_counts = spec_k > 0
        ? (int32_t *)calloc((size_t)m->n_vocab, sizeof(int32_t)) : NULL;
    int mtp_unique_targets = 0, mtp_max_target_count = 0;
    int32_t mtp_pending[5] = {-1, -1, -1, -1, -1};
    int mtp_pending_n = 0;
    float *mtp_seed_hidden = spec_k > 0
        ? (float *)malloc((size_t)n_embd * sizeof(float)) : NULL;
    if (spec_k > 0 && !mtp_seed_hidden) die("MTP hidden allocation", -1);
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
            float *pf_hidden = NULL;
            if (spec_k && m->nextn.loaded) tf_batch_hidden_out = &pf_hidden;
            float *lg = transformer_prefill_gemm(m, ptoks + prefill_from, prefill_tokens, prefill_from);
            tf_batch_hidden_out = NULL;
            if (lg) {
                double ar_step = 0.0; long ar_calls_step = 0;
                in_tok = sample_argmax(m, lg, &c, &ar_step, &ar_calls_step);
                t_prefill = now_sec() - pf0;
                t_comm += ar_step; ar_calls += ar_calls_step;
                prefill_gemm_used = 1;
                if (spec_k && m->nextn.loaded && pf_hidden && prefill_from == 0) {
                    float *th = (float *)alloca((size_t)n_embd * sizeof(float));
                    for (int p = 0; p < P; p++) {
                        const float *raw = pf_hidden + (size_t)p * n_embd;
                        if (envb_opt("TP_MTP_RAW_HIDDEN", 0)) memcpy(th, raw, (size_t)n_embd*sizeof(float));
                        else tf_rmsnorm(th, raw, &m->output_norm, n_embd,
                                        m->rms_norm_eps, m->matvec_tmp);
                        memcpy(mtp_seed_hidden, th, (size_t)n_embd * sizeof(float));
                        if (p + 1 < P) {
                            float *dlg = transformer_nextn_logits(m, ptoks[p + 1], th, p);
                            if (p + 2 < P) {
                                int32_t d = sample_argmax(m, dlg, &c, &ar_step, &ar_calls_step);
                                mtp_teacher_match += d == ptoks[p + 2]; mtp_teacher_total++;
                            }
                        }
                    }
                    int seed_drafts = mtp_batch ? spec_k - 1 : spec_k;
                    int prev = in_tok;
                    const float *dh = mtp_seed_hidden;
                    for (int k = 0; k < seed_drafts; k++) {
                        float *dlg = transformer_nextn_logits(m, prev, dh, P - 1 + k);
                        mtp_pending[k] = sample_argmax(m, dlg, &c, &ar_step, &ar_calls_step);
                        prev = mtp_pending[k]; dh = transformer_nextn_hidden(m);
                    }
                    mtp_pending_n = seed_drafts;
                    t_comm += ar_step; ar_calls += ar_calls_step;
                }
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
                if (spec_k && m->nextn.loaded) {
                    float *target_h = (float *)alloca((size_t)n_embd * sizeof(float));
                    memcpy(target_h, transformer_get_hidden(m), (size_t)n_embd * sizeof(float));
                    const float *mtp_target = envb_opt("TP_MTP_RAW_HIDDEN", 0)
                        ? transformer_nextn_target_hidden(m) : transformer_get_hidden(m);
                    memcpy(mtp_seed_hidden, mtp_target, (size_t)n_embd * sizeof(float));
                    /* Teacher-force the known next prompt token.  The draft
                     * head consumes (h[p], token[p+1]) and predicts token[p+2]. */
                    if (p + 1 < P) {
                        float *catchup_logits = transformer_nextn_logits(
                            m, ptoks[p + 1], mtp_seed_hidden, p);
                        if (p + 2 < P) {
                            double catchup_ar = 0.0; long catchup_calls = 0;
                            int32_t catchup = sample_argmax(
                                m, catchup_logits, &c, &catchup_ar, &catchup_calls);
                            mtp_teacher_match += catchup == ptoks[p + 2];
                            mtp_teacher_total++;
                            for (int oi = 0; oi < 7; oi++) {
                                int expected_pos = p + oi - 1;
                                if (expected_pos >= 0 && expected_pos < P)
                                    mtp_teacher_offset_match[oi] += catchup == ptoks[expected_pos];
                            }
                            if (is_first && envb_opt("TP_MTP_TRACE", 0))
                                logmsg("MTP catchup pos=%d input=%d draft=%d expected=%d\n",
                                       p, ptoks[p + 1], catchup, ptoks[p + 2]);
                        }
                    }
                    transformer_set_hidden(m, target_h);
                }
                t_fwd += now_sec() - _ta;
            }
            float *lg = transformer_compute_logits(m);  /* only final prompt token matters */
            if (!lg) die("transformer_compute_logits after prefill", -1);
            double ar_step = 0.0; long ar_calls_step = 0;
            in_tok = sample_argmax(m, lg, &c, &ar_step, &ar_calls_step);
            t_prefill = now_sec() - pf0;
            t_comm += ar_step; ar_calls += ar_calls_step;

            /* Seed predictions beyond the first target token.  The trunk
             * logits above already provide in_tok; NextN predicts the token
             * after it, so agreement is checked on the following trunk step. */
            if (spec_k && m->nextn.loaded && mtp_seed_hidden) {
                int seed_drafts = mtp_batch ? spec_k - 1 : spec_k;
                const float *dh = mtp_seed_hidden;
                int32_t prev = in_tok;
                for (int k = 0; k < seed_drafts; k++) {
                    float *dlg = transformer_nextn_logits(m, prev, dh, P - 1 + k);
                    double da = 0.0; long dc = 0;
                    mtp_pending[k] = sample_argmax(m, dlg, &c, &da, &dc);
                    t_comm += da; ar_calls += dc;
                    prev = mtp_pending[k];
                    dh = transformer_nextn_hidden(m);
                }
                mtp_pending_n = seed_drafts;
            }
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
    double t_decode_wall = 0.0;

    if (!dry_decode && mtp_batch && spec_k > 0 && m->nextn.loaded) {
        tp_spec_state ss;
        tf_batch_keep_pool = 1;
        tf_batch_quiet = 1;
        transformer_prefill_profile_reset();
        if (tp_spec_state_init(&ss, m) != 0) die("MTP recurrent snapshot alloc", -1);
        int vlogits = m->output.n_rows;
        float *all_logits = (float *)malloc((size_t)spec_k * vlogits * sizeof(float));
        float *batch_hidden = NULL;
        size_t snap_conv = (size_t)(m->ssm_conv_kernel - 1) * m->ssm_qkv_dim;
        size_t snap_rec = (size_t)m->ssm_dt_rank * m->ssm_d_state * m->ssm_d_state;
        size_t snap_layer = snap_conv + snap_rec;
        size_t snap_slot = (size_t)n_layers * snap_layer;
        float *batch_ssm = NULL;
        float **batch_ssm_slots = (float **)alloca((size_t)spec_k * sizeof(*batch_ssm_slots));
        float **orig_conv = (float **)alloca((size_t)n_layers * sizeof(*orig_conv));
        float **orig_rec = (float **)alloca((size_t)n_layers * sizeof(*orig_rec));
        int verify_drafts = spec_k - 1;
        if (posix_memalign((void **)&batch_ssm, 256,
                           (size_t)spec_k * snap_slot * sizeof(float)) != 0)
            batch_ssm = NULL;
        if (!all_logits || !batch_ssm) die("MTP batch scratch alloc", -1);
        for (int k = 0; k < spec_k; k++)
            batch_ssm_slots[k] = k < verify_drafts
                ? batch_ssm + (size_t)k * snap_slot : NULL;
        float *batch_ssm_current = batch_ssm + (size_t)verify_drafts * snap_slot;
        for (int l = 0; l < n_layers; l++) {
            orig_conv[l] = m->conv_state ? m->conv_state[l] : NULL;
            orig_rec[l] = m->recurrent_state ? m->recurrent_state[l] : NULL;
            if (!m->layers[l].is_ssm) continue;
            float *ls = batch_ssm_current + (size_t)l * snap_layer;
            memcpy(ls, orig_conv[l], snap_conv * sizeof(float));
            memcpy(ls + snap_conv, orig_rec[l], snap_rec * sizeof(float));
            m->conv_state[l] = ls;
            m->recurrent_state[l] = ls + snap_conv;
        }
        /* The old pre-round copy is no longer needed; retain its byte count for
         * reporting and its conv_pos array for the round boundary. */
        free(ss.data); ss.data = NULL;
        tf_batch_ssm_snapshots = batch_ssm;
        tf_batch_ssm_snapshot_slots = batch_ssm_slots;
        tf_batch_ssm_layer_stride = snap_layer;
        tf_batch_ssm_slot_stride = snap_slot;
        int mtp_shadow_threads = (int)envl_opt("TP_MTP_SHADOW_THREADS", 0);
        transformer_model *mtp_draft_model = m;
        if (mtp_shadow_threads > 0) {
            mtp_draft_model = transformer_nextn_context_create(m, mtp_shadow_threads);
            if (!mtp_draft_model) die("MTP shadow context alloc", -1);
            transformer_nextn_context_copy_state(mtp_draft_model, m);
            /* Tensor-parallel fields and callback are copied from m; only the
             * mutable NextN runtime and worker pool are private. */
            if (is_first)
                logmsg("MTP shadow context: threads=%d independent scratch/KV/pool\n",
                       mtp_shadow_threads);
        }
        if (is_first)
            logmsg("MTP batched verify: K=%d vocab/rank=%d recurrent_snapshot=%.1fMB\n",
                   spec_k, vlogits, (double)ss.bytes / (1024.0 * 1024.0));

        int p = (int)decode_start;
        int measured = perf_warmup == 0;
        int mtp_detail = envb_opt("TP_MTP_PROFILE_DETAIL", 0);
        int mtp_omp_park = envb_opt("TP_MTP_OMP_PARK", 0);
#ifndef _OPENMP
        mtp_omp_park = 0;
#endif
        pthread_mutex_t mtp_park_mu = PTHREAD_MUTEX_INITIALIZER;
        pthread_cond_t mtp_park_cv = PTHREAD_COND_INITIALIZER;
        int mtp_park_done = 0;
        double mtp_verify_sec = 0.0, mtp_restore_sec = 0.0, mtp_draft_sec = 0.0;
        long mtp_detail_rounds = 0;
        double decode_wall_start = now_sec();
        while (n_gen < max_gen + perf_warmup) {
            if (mtp_pending_n < verify_drafts) die("MTP draft queue not full", -1);
            int32_t batch[5], target[5];
            batch[0] = in_tok;
            for (int j = 1; j < spec_k; j++) batch[j] = mtp_pending[j - 1];

            double ta = now_sec();
            for (int l = 0; l < n_layers; l++)
                ss.conv_pos[l] = m->conv_state_pos ? m->conv_state_pos[l] : 0;
            g_ar_secs = 0.0; g_ar_calls = 0;
            tf_batch_all_logits = all_logits;
            tf_batch_hidden_out = &batch_hidden;
            float *batch_ok = transformer_prefill_gemm(m, batch, spec_k, p);
            tf_batch_all_logits = NULL;
            tf_batch_hidden_out = NULL;
            if (!batch_ok || !batch_hidden) die("Qwen MTP batched verify", -1);

            double argmax_ar = 0.0; long argmax_calls = 0;
            sample_argmax_n(m, all_logits, vlogits, spec_k, target, &c,
                            &argmax_ar, &argmax_calls);
            int accepted = 0;
            while (accepted < verify_drafts && mtp_pending[accepted] == target[accepted])
                accepted++;
            double td_verify = mtp_detail ? now_sec() : 0.0;
            if (envb_opt("TP_MTP_FORCE_ACCEPT", 0)) accepted = verify_drafts;
            for (int j = 0; j < verify_drafts; j++) {
                int hit = mtp_pending[j] == target[j];
                mtp_match += hit; mtp_total++;
                mtp_horizon_match[j] += hit; mtp_horizon_total[j]++;
                if (mtp_token_counts && target[j] >= 0 && target[j] < m->n_vocab) {
                    int count = ++mtp_token_counts[target[j]];
                    if (count == 1) mtp_unique_targets++;
                    if (count > mtp_max_target_count) mtp_max_target_count = count;
                }
            }

            int emitted = accepted + 1;
            const float *draft_h = NULL;
            if (accepted < verify_drafts) {
                /* Select the state captured immediately after the last
                 * committed input. No trunk replay is needed. */
                int selected = emitted - 1;
                float *old_current = batch_ssm_current;
                batch_ssm_current = batch_ssm_slots[selected];
                batch_ssm_slots[selected] = old_current;
                for (int l = 0; l < n_layers; l++) {
                    if (!m->layers[l].is_ssm) continue;
                    float *ls = batch_ssm_current + (size_t)l * snap_layer;
                    m->conv_state[l] = ls;
                    m->recurrent_state[l] = ls + snap_conv;
                    m->conv_state_pos[l] =
                        (ss.conv_pos[l] + emitted) % (m->ssm_conv_kernel - 1);
                }
                const float *last_hidden = batch_hidden + (size_t)(emitted - 1) * n_embd;
                if (envb_opt("TP_MTP_RAW_HIDDEN", 0)) draft_h = last_hidden;
                else {
                    tf_rmsnorm(mtp_seed_hidden, last_hidden, &m->output_norm,
                               n_embd, m->rms_norm_eps, m->matvec_tmp);
                    draft_h = mtp_seed_hidden;
                }
            } else {
                const float *last_hidden = batch_hidden + (size_t)(spec_k - 1) * n_embd;
                if (envb_opt("TP_MTP_RAW_HIDDEN", 0)) {
                    draft_h = last_hidden;
                } else {
                    tf_rmsnorm(mtp_seed_hidden, last_hidden, &m->output_norm,
                               n_embd, m->rms_norm_eps, m->matvec_tmp);
                    draft_h = mtp_seed_hidden;
                }
            }
            double td_restore = mtp_detail ? now_sec() : 0.0;

            int stop = 0, committed = 0;
            for (int j = 0; j < emitted && n_gen < max_gen + perf_warmup; j++) {
                int32_t out_tok = target[j];
                int eos = out_tok == bpe_eos_id(vocab) || out_tok == bpe_eot_id(vocab);
                if (g_tokdump) fprintf(g_tokdump, "%d\n", (int)out_tok);
                if (!eos && is_first) print_token(vocab, out_tok);
                n_gen++; committed++;
                if (eos && !ignore_eos) { stop = 1; break; }
            }
            if (g_tokdump) fflush(g_tokdump);
            if (committed == 0) break;
            in_tok = target[committed - 1];
            p += committed;
            final_cache_pos = p;

            /* Build a fresh K-token queue from the target hidden at the last
             * committed input.  The auxiliary head is TP-sharded and its
             * argmax reduction is included in the communication ledger. */
            int prev = in_tok;
            mtp_park_done = 0;
            #ifdef _OPENMP
            #pragma omp parallel num_threads(48) if(mtp_omp_park) \
                shared(mtp_park_done, prev, draft_h, argmax_ar, argmax_calls)
            #endif
            {
                int park_tid = 0;
                #ifdef _OPENMP
                park_tid = omp_get_thread_num();
                #endif
                if (park_tid == 0) {
                    if (envb_opt("TP_MTP_SKIP_DRAFT", 0)) {
                        for (int k = 0; k < verify_drafts; k++) mtp_pending[k] = prev;
                    } else {
                        for (int k = 0; k < verify_drafts; k++) {
                            float *dlg = transformer_nextn_logits(
                                mtp_draft_model, prev, draft_h, p - 1 + k);
                            double da = 0.0; long dc = 0;
                            mtp_pending[k] = sample_argmax(
                                mtp_draft_model, dlg, &c, &da, &dc);
                            argmax_ar += da; argmax_calls += dc;
                            prev = mtp_pending[k];
                            draft_h = transformer_nextn_hidden(mtp_draft_model);
                        }
                    }
                    if (mtp_omp_park) {
                        pthread_mutex_lock(&mtp_park_mu);
                        mtp_park_done = 1;
                        pthread_cond_broadcast(&mtp_park_cv);
                        pthread_mutex_unlock(&mtp_park_mu);
                    }
                } else {
                    pthread_mutex_lock(&mtp_park_mu);
                    while (!mtp_park_done)
                        pthread_cond_wait(&mtp_park_cv, &mtp_park_mu);
                    pthread_mutex_unlock(&mtp_park_mu);
                }
            }
            mtp_pending_n = verify_drafts;
            double td_draft = mtp_detail ? now_sec() : 0.0;

            double tb = now_sec();
            if (measured) {
                t_fwd += tb - ta;
                t_comm += g_ar_secs + argmax_ar;
                ar_calls += g_ar_calls + argmax_calls;
                pcnt += committed;
                if (mtp_detail) {
                    mtp_verify_sec += td_verify - ta;
                    mtp_restore_sec += td_restore - td_verify;
                    mtp_draft_sec += td_draft - td_restore;
                    mtp_detail_rounds++;
                }
            } else if (n_gen >= perf_warmup) {
                measured = 1;
                t_fwd = t_comm = 0.0; ar_calls = 0; pcnt = 0;
                transformer_pool_profile_reset();
            }
            if (g_curve) {
                fprintf(g_curve, "%d %d %.3f %.3f\n", p - committed, p,
                        1000.0 * (tb - ta), 1000.0 * (g_ar_secs + argmax_ar));
                fflush(g_curve);
            }
            if (is_first && envb_opt("TP_MTP_TRACE", 0))
                logmsg("MTP batch pos=%d accepted=%d/%d committed=%d next=%d\n",
                       p - committed, accepted, spec_k, committed, in_tok);
            if (stop) break;
        }
        t_decode_wall = now_sec() - decode_wall_start;
        if (is_first) {
            transformer_prefill_profile bp;
            transformer_prefill_profile_get(&bp);
            logmsg("MTP batch profile: calls=%d norm=%.1f proj=%.1f ssm_prep=%.1f "
                   "ssm_scan=%.1f attn_prep=%.1f attn=%.1f out=%.1f "
                   "ffn_proj=%.1f act=%.1f down=%.1f collective=%.1f ms\n",
                   bp.calls, bp.norm_ms, bp.proj_ms, bp.ssm_prepare_ms,
                   bp.ssm_scan_ms, bp.attn_prepare_ms, bp.attn_kernel_ms,
                   bp.out_proj_ms, bp.ffn_proj_ms, bp.ffn_act_ms,
                   bp.ffn_down_ms, bp.collective_ms);
            if (mtp_detail && mtp_detail_rounds)
                logmsg("MTP round profile: rounds=%ld verify=%.2f restore=%.2f draft=%.2f ms/round\n",
                       mtp_detail_rounds, 1000.0 * mtp_verify_sec / mtp_detail_rounds,
                       1000.0 * mtp_restore_sec / mtp_detail_rounds,
                       1000.0 * mtp_draft_sec / mtp_detail_rounds);
        }
        tf_batch_ssm_snapshots = NULL;
        tf_batch_ssm_snapshot_slots = NULL;
        tf_batch_ssm_layer_stride = tf_batch_ssm_slot_stride = 0;
        pthread_cond_destroy(&mtp_park_cv);
        pthread_mutex_destroy(&mtp_park_mu);
        for (int l = 0; l < n_layers; l++) {
            if (!m->layers[l].is_ssm) continue;
            float *ls = batch_ssm_current + (size_t)l * snap_layer;
            memcpy(orig_conv[l], ls, snap_conv * sizeof(float));
            memcpy(orig_rec[l], ls + snap_conv, snap_rec * sizeof(float));
            m->conv_state[l] = orig_conv[l];
            m->recurrent_state[l] = orig_rec[l];
        }
        free(batch_ssm);
        free(all_logits);
        if (mtp_draft_model != m)
            transformer_nextn_context_free(mtp_draft_model);
        tp_spec_state_free(&ss);
        goto done;
    }

    double decode_wall_start = now_sec();
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
            if (spec_k && m->nextn.loaded) {
                int regenerate_drafts = 1;
                if (is_first && envb_opt("TP_MTP_TRACE", 0))
                    logmsg("MTP trunk pos=%d input=%d target=%d\n", p, in_tok, nt);
                if (mtp_pending_n > 0) {
                    if (is_first && envb_opt("TP_MTP_TRACE", 0))
                        logmsg("MTP compare pos=%d draft=%d target=%d\n",
                               p, mtp_pending[0], nt);
                    int accepted = mtp_pending[0] == nt;
                    int horizon = spec_k - mtp_pending_n;
                    mtp_match += accepted;
                    mtp_total++;
                    if (horizon >= 0 && horizon < 4) {
                        mtp_horizon_match[horizon] += accepted;
                        mtp_horizon_total[horizon]++;
                    }
                    if (mtp_token_counts && nt >= 0 && nt < m->n_vocab) {
                        int count = ++mtp_token_counts[nt];
                        if (count == 1) mtp_unique_targets++;
                        if (count > mtp_max_target_count) mtp_max_target_count = count;
                    }
                    if (accepted && mtp_pending_n > 1) {
                        memmove(mtp_pending, mtp_pending + 1,
                                (size_t)(mtp_pending_n - 1) * sizeof(mtp_pending[0]));
                        mtp_pending_n--;
                        regenerate_drafts = 0;
                    } else {
                        mtp_pending_n = 0;
                    }
                }
                if (regenerate_drafts) {
                    const float *draft_h = envb_opt("TP_MTP_RAW_HIDDEN", 0)
                        ? transformer_nextn_target_hidden(m) : transformer_get_hidden(m);
                    int prev = nt;
                    for (int k = 0; k < spec_k; k++) {
                        float *dlg = transformer_nextn_logits(m, prev, draft_h, p + k);
                        double draft_ar = 0.0; long draft_calls = 0;
                        int32_t draft = sample_argmax(m, dlg, &c, &draft_ar, &draft_calls);
                        ar_step += draft_ar;
                        ar_calls_step += draft_calls;
                        mtp_pending[k] = draft;
                        prev = draft;
                        draft_h = transformer_nextn_hidden(m);
                    }
                    mtp_pending_n = spec_k;
                }
            }
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
        if (perf_warmup > 0 && n_gen == perf_warmup) {
            t_fwd = 0.0; t_comm = 0.0; ar_calls = 0; pcnt = 0;
            transformer_pool_profile_reset();
        }
        if (stop_eos || n_gen >= max_gen + perf_warmup) break;
        in_tok = nt;
    }
    t_decode_wall = now_sec() - decode_wall_start;

done:
    free(mtp_seed_hidden);
    free(mtp_token_counts);
    if (getenv("TP_BUFFER_OUTPUT")) {
        if (g_log) fflush(g_log);
        fflush(stdout);
    }
    barrier();
    if (is_first && mtp_total)
        logmsg("MTP greedy match=%ld/%ld alpha=%.4f K=%d diversity=%d max_share=%.4f gate=%s\n",
               mtp_match, mtp_total, (double)mtp_match / mtp_total, spec_k,
               mtp_unique_targets, (double)mtp_max_target_count / mtp_total,
               mtp_unique_targets >= 8 && mtp_max_target_count * 10 < mtp_total * 9
                   ? "nondegenerate" : "REJECT-DEGENERATE");
    if (is_first && mtp_total) {
        logmsg("MTP horizons:");
        for (int k = 0; k < spec_k; k++)
            logmsg(" h%d=%ld/%ld", k + 1, mtp_horizon_match[k], mtp_horizon_total[k]);
        logmsg("\n");
    }
    if (is_first && mtp_teacher_total)
        logmsg("MTP teacher match=%ld/%ld alpha=%.4f offsets[p-1..p+5]="
               "%ld,%ld,%ld,%ld,%ld,%ld,%ld\n", mtp_teacher_match,
               mtp_teacher_total, (double)mtp_teacher_match / mtp_teacher_total,
               mtp_teacher_offset_match[0], mtp_teacher_offset_match[1],
               mtp_teacher_offset_match[2], mtp_teacher_offset_match[3],
               mtp_teacher_offset_match[4], mtp_teacher_offset_match[5],
               mtp_teacher_offset_match[6]);
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
        double t_dec = t_decode_wall > 0.0 ? t_decode_wall : t_total - t_prefill;
        logmsg("\n\n=== done: %d tokens generated ===\n", n_gen);
        logmsg("prefill(%d tok)=%.3f s (%.2f tok/s)%s\n", prefill_tokens, t_prefill,
               prefill_tokens > 0 ? (double)prefill_tokens / t_prefill : 0.0,
               cache_prefill_used ? " [from cache]" : "");
        if (cache_prefill_used && cache_prefill_skipped > 0)
            logmsg("cache-skipped=%d prompt toks, ck_pos=%lld\n", cache_prefill_skipped, (long long)ck_pos);
        logmsg("decode(%d tok)=%.3f s = %.2f tok/s\n", n_gen, t_dec, n_gen > 0 ? n_gen / t_dec : 0.0);
    }
    if (getenv("TF_DPROF") && pcnt > 0) {
        double mat_bw = tf_decode_matvec_ms > 0.0
                      ? tf_decode_matvec_bytes / (tf_decode_matvec_ms * 1e6) : 0.0;
        fprintf(stderr, "rank %d dprof: matvec=%.2fms/tok BW=%.1fGB/s dispatches=%.1f/tok "
                        "attn_qkv=%.2f attn_out=%.2f ssm_in=%.2f ssm_prepare=%.2f "
                        "ssm_core=%.2f ssm_out=%.2f ffn_gateup=%.2f ffn_down=%.2f lm_head=%.2f\n",
                MyRank, tf_decode_matvec_ms / pcnt, mat_bw,
                (double)tf_decode_matvec_cnt / pcnt,
                tf_decode_attn_qkv_ms / pcnt, tf_decode_attn_out_ms / pcnt,
                tf_decode_ssm_in_ms / pcnt, tf_decode_ssm_prepare_ms / pcnt,
                tf_decode_ssm_core_ms / pcnt, tf_decode_ssm_out_ms / pcnt,
                tf_decode_ffn_gateup_ms / pcnt, tf_decode_ffn_down_ms / pcnt,
                tf_decode_lm_head_ms / pcnt);
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
    free(prompt_file_text);
    if (g_log) fclose(g_log);
    if (g_curve) fclose(g_curve);
    if (g_tokdump) fclose(g_tokdump);
    return 0;
}
