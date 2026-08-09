/* Full Kimi K3 C11 inference runner.
 *
 * The checkpoint image is produced by k3_full_stage.py.  Every rank loads one
 * bounded anonymous copy of its rank blob, owns one attention head, one slice
 * of tensor-parallel projections, and experts e % world_size == rank.  The
 * lockstep token path intentionally keeps the first implementation simple:
 * all ranks see the same token and use sum/max collectives for TP/EP joins.
 *
 * This file is C11.  It uses the existing C headers for the A64FX kernels and
 * uTofu transport; there is no C++ runtime or generated C++ dependency.
 */
#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <utofu.h>

#define GGML_DEQUANT_IMPLEMENTATION
#include "k3_dense.h"
#include "k3_kernels.h"
#include "k3_moe.h"
#include "k3_quant.h"
#include "k3_runtime.h"
#include "../utofu-tests/tofu_demo.h"
#include "../utofu-tests/tp_allreduce.h"

#define K3_FULL_MAX_NODES 96
#define K3_FULL_MAX_ENTRIES 600000
#define K3_FULL_MAX_NAME 512
#define K3_FULL_WAIT_SECONDS 120.0
#define K3_FULL_BARRIER_RETRY_USEC 2000
#define K3_FULL_BARRIER_ITERS_DEFAULT 128
#define K3_FULL_STAG DEMO_STAG
#define K3_FULL_REDUCE_COUNT (K3_HIDDEN + K3_LATENT)
#define K3_FULL_EPS 1.0e-5f
#define K3_FULL_VOCAB 163840
#define K3_FULL_MLA_QK 192
#define K3_FULL_MLA_VALUE 128
#define K3_FULL_CONV_KERNEL 4
#define K3_FULL_MAX_DEBUG_TOKENS 4096
#define K3_FULL_DTYPE_Q8P16 4
#define K3_FULL_DTYPE_Q8P8 5
#define K3_FULL_DTYPE_Q8_0 6
#define K3_FULL_DTYPE_IQ1_S 7
#define K3_FULL_DTYPE_IQ2_XS 8
#define K3_FULL_DTYPE_IQ2_XXS 9
#define K3_FULL_DTYPE_IQ3_XXS 10
#define K3_FULL_PROFILE_PHASES 26

enum {
    K3_FULL_PHASE_LAYER = 0,
    K3_FULL_PHASE_ATTENTION = 1,
    K3_FULL_PHASE_MOE = 2,
    K3_FULL_PHASE_REDUCE = 3,
    K3_FULL_PHASE_RESIDUAL = 4,
    K3_FULL_PHASE_MOE_DISPATCH = 5,
    K3_FULL_PHASE_MOE_EXPERT = 6,
    K3_FULL_PHASE_MOE_SHARED = 7,
    K3_FULL_PHASE_MOE_COLLECTIVE = 8,
    K3_FULL_PHASE_MOE_FINISH = 9,
    K3_FULL_PHASE_DISPATCH_PROJ = 10,
    K3_FULL_PHASE_ROUTER_REDUCE = 11,
    K3_FULL_PHASE_LATENT_REDUCE = 12,
    /* KDA attention breakdown */
    K3_FULL_PHASE_KDA_QKV = 13,
    K3_FULL_PHASE_KDA_CONV = 14,
    K3_FULL_PHASE_KDA_DECAY_PROJ = 15,
    K3_FULL_PHASE_KDA_SERIAL = 16,
    K3_FULL_PHASE_KDA_STEP = 17,
    K3_FULL_PHASE_KDA_OUT = 18,
    K3_FULL_PHASE_SHARED_GATEUP = 19,
    K3_FULL_PHASE_SHARED_SITU = 20,
    K3_FULL_PHASE_SHARED_DOWN = 21,
    K3_FULL_PHASE_KDA_GRMSNORM = 22,
    K3_FULL_PHASE_KDA_OPROJ = 23,
    K3_FULL_PHASE_FINISH_MATVEC = 24,
    K3_FULL_PHASE_FINISH_REDUCE = 25,
};

typedef struct {
    int enabled;
    int current_layer;
    int current_phase;
    uint64_t layer_count;
    uint64_t phase_count[K3_FULL_PROFILE_PHASES];
    uint64_t collective_count;
    double layer_sum[K3_LAYERS];
    double layer_max[K3_LAYERS];
    double phase_sum[K3_FULL_PROFILE_PHASES];
    double phase_max[K3_FULL_PROFILE_PHASES];
    double collective_sum;
    double collective_max;
} k3_full_profile;

typedef enum {
    K3_FULL_MODE_FULL96 = 0,
    K3_FULL_MODE_LAYER12 = 1,
    K3_FULL_MODE_SYNTHETIC12 = 2,
    K3_FULL_MODE_BARRIER = 3,
} k3_full_mode;

#define K3_CMG_COUNT 4
typedef struct {
    const uint8_t *data;
    size_t nbytes;
    int dtype;              /* 1=BF16, 2=F32, 3=U8, 4=Q8P16, 5=Q8P8 */
    int ndims;
    size_t shape[3];
    int pv;                 /* BF16 rows repacked pair-interleaved for _pv */
    unsigned char *cmg_map; /* CMG index per 2 MB large page, NULL if unmapped */
    unsigned char *cmg_copy[K3_CMG_COUNT];  /* per-CMG replica, NULL if not replicated */
    k3_quant_packed quant_packed; /* optional persistent IQ row16 cache */
    char name[K3_FULL_MAX_NAME];
} k3_full_tensor;

typedef struct {
    k3_full_tensor w1;
    k3_full_tensor w2;
    k3_full_tensor w3;
    k3_mxfp4_matrix mw1;
    k3_mxfp4_matrix mw2;
    k3_mxfp4_matrix mw3;
    int expert_id;
} k3_full_expert;

typedef struct {
    int is_mla;
    int state_slot;
    int cache_slot;
    k3_full_tensor input_norm;
    k3_full_tensor post_norm;
    k3_full_tensor attn_res_norm;
    k3_full_tensor attn_res_proj;
    k3_full_tensor mlp_res_norm;
    k3_full_tensor mlp_res_proj;

    /* KDA */
    k3_full_tensor q_proj, k_proj, v_proj, g_proj;
    k3_full_tensor f_a_proj, f_b_proj, b_proj;
    k3_full_tensor q_conv, k_conv, v_conv;
    k3_full_tensor a_log, dt_bias, o_norm, o_proj;

    /* MLA */
    k3_full_tensor q_a_proj, q_a_norm, q_b_proj;
    k3_full_tensor kv_a_proj, kv_a_norm, kv_b_proj;
    k3_full_tensor mla_g_proj, mla_o_proj;

    /* Dense layer 0 */
    k3_full_tensor dense_gate, dense_up, dense_down;

    /* Sparse MoE layers */
    k3_full_tensor router, router_bias;
    k3_full_tensor routed_down, routed_norm, routed_up;
    k3_full_tensor shared_gate, shared_up, shared_down;
    k3_full_expert *experts;
    int expert_count;
    int moe_cmg_replicated;
} k3_full_layer;

typedef struct {
    int version;
    int rank;
    int nodes;
    int layer_index;
    size_t blob_bytes;
    char mode[64];
} k3_full_manifest;

typedef struct {
    int rank;
    int nodes;
    int threads;
    int max_seq;
    int local_heads;
    int first_head;
    int latent_start;
    int latent_rows;
    int embed_start;
    int embed_rows;
    int head_start;
    int head_rows;
    const uint8_t *blob;
    size_t blob_bytes;
    k3_pool *pool;
    k3_full_tensor embed;
    k3_full_tensor lm_head;
    k3_full_tensor final_norm;
    k3_full_layer layers[K3_LAYERS];
    int kda_count;
    int mla_count;
    tp_comm *comm;
    tp_comm *comm_col;
    int debug_layer_index;
    k3_full_layer debug_layer;
    int expert_tp;
    int q8_mode;
    int prefetch_mib;
    k3_full_profile profile;
    uint64_t synthetic_route_hash;
    uint64_t synthetic_collectives;

    float *kda_state;
    float *conv_state;
    float *mla_keys;
    float *mla_values;
    float *block_residual;
    int block_count;

    float *hidden;
    float *normed;
    float *tmp;
    float *tmp2;
    float *attn;
    float *q;
    float *k;
    float *v;
    float *gate;
    float *up;
    float *decay;
    float *local_latent;
    float *routed_latent;
    float *moe_hidden;
    float *logits;
    float *expert_gathered;
    float *expert_gate;
    float *expert_up;
    float *expert_out;
    float *shared_hidden;
    float *reduce;
    int8_t *q_scratch;
    float *mla_scratch, *mla_stats;
    int *expert_counts;
    int *expert_tokens;
    float *expert_weights;
} k3_full_model;

typedef struct {
    k3_full_mode mode;
    int nodes;
    int threads;
    int max_seq;
    int prefill_tokens;
    int new_tokens;
    int prefill_only;
    int real_layer_index;
    uint64_t input_seed;
    int prefill_chunk;
    int barrier_iters;
    int comm_deterministic;
    int ar_groups;
    int comm_use_bf16;
    int comm_robust;
    int comm_poll_spins;
    int comm_a2a;
    int comm_a2a_max;
    int prefetch_mib;
    int profile;
    const char *profile_output;
    const char *stage_dir;
    const char *topo_path;
    const char *prompt_ids;
    const char *output_path;
} k3_full_options;

typedef struct {
    uint64_t offset;
    uint64_t nbytes;
    int dtype;
    int ndims;
    size_t shape[3];
    char name[K3_FULL_MAX_NAME];
} k3_full_entry;

static int g_rank = -1;
static int g_nodes = 0;
static int g_async_latent = 0;
static int g_half_col = 0;
static int g_half_hidden = 0;
static int g_sparse_row = 0;
static char *g_region;
static size_t g_send_off;
static size_t g_bar_base;

static int full_env_int(const char *name, int fallback) {
    const char *value = getenv(name);
    return value ? atoi(value) : fallback;
}
static size_t g_slot_bar;
static utofu_vcq_hdl_t g_vcq;
static utofu_stadd_t g_base;
static utofu_vcq_id_t g_peer_vcq[K3_FULL_MAX_NODES];
static utofu_stadd_t g_peer_base[K3_FULL_MAX_NODES];
static uint64_t g_bar_token = 1;
static const unsigned long g_put_flags = UTOFU_ONESIDED_FLAG_TCQ_NOTICE;

static double full_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static void full_profile_phase_begin(k3_full_model *m, int phase) {
    if (!m->profile.enabled || phase < 0 || phase >= K3_FULL_PROFILE_PHASES)
        return;
    m->profile.current_phase = phase;
}

static void full_profile_phase_add(k3_full_model *m, int phase, double seconds) {
    if (!m->profile.enabled || phase < 0 || phase >= K3_FULL_PROFILE_PHASES)
        return;
    m->profile.phase_sum[phase] += seconds;
    if (seconds > m->profile.phase_max[phase])
        m->profile.phase_max[phase] = seconds;
    ++m->profile.phase_count[phase];
}

static void full_profile_layer_begin(k3_full_model *m, int layer) {
    if (!m->profile.enabled) return;
    m->profile.current_layer = layer;
    m->profile.current_phase = K3_FULL_PHASE_LAYER;
}

static void full_profile_layer_add(k3_full_model *m, int layer, double seconds) {
    if (!m->profile.enabled || layer < 0 || layer >= K3_LAYERS) return;
    m->profile.layer_sum[layer] += seconds;
    if (seconds > m->profile.layer_max[layer])
        m->profile.layer_max[layer] = seconds;
    ++m->profile.layer_count;
}

static void full_profile_layer_end(k3_full_model *m, int layer, double start) {
    if (m->profile.enabled)
        full_profile_layer_add(m, layer, full_now() - start);
}

static void full_profile_collective_add(k3_full_model *m, double seconds) {
    if (!m->profile.enabled) return;
    m->profile.collective_sum += seconds;
    if (seconds > m->profile.collective_max)
        m->profile.collective_max = seconds;
    ++m->profile.collective_count;
}

static void full_profile_reset(k3_full_model *m) {
    int enabled = m->profile.enabled;
    memset(&m->profile, 0, sizeof m->profile);
    m->profile.enabled = enabled;
}

static const char *full_profile_phase_name(int phase) {
    static const char *names[K3_FULL_PROFILE_PHASES] = {
        "layer", "attention", "moe", "reduce", "residual",
        "moe_dispatch", "moe_expert", "moe_shared",
        "moe_collective", "moe_finish", "dispatch_proj",
        "router_reduce", "latent_reduce",
        "kda_qkv", "kda_conv", "kda_decay_proj",
        "kda_serial", "kda_step", "kda_out",
        "shared_gateup", "shared_situ", "shared_down",
        "kda_grmsnorm", "kda_oproj", "finish_matvec", "finish_reduce"
    };
    return phase >= 0 && phase < K3_FULL_PROFILE_PHASES ? names[phase] : "unknown";
}

typedef struct {
    const uint8_t *data;
    size_t bytes;
} k3_full_prefetch_job;

static void *full_prefetch_worker(void *opaque) {
    const k3_full_prefetch_job *job = (const k3_full_prefetch_job *)opaque;
    for (size_t off = 0; off < job->bytes; off += 64)
        __builtin_prefetch(job->data + off, 0, 2);
    return NULL;
}

static int full_prefetch_start(const k3_full_model *m,
                               const k3_full_tensor *weights,
                               pthread_t *thread,
                               k3_full_prefetch_job *job) {
    if (!m->prefetch_mib || !weights->data || !weights->nbytes) return 0;
    size_t limit = (size_t)m->prefetch_mib * 1024 * 1024;
    if (limit > weights->nbytes) limit = weights->nbytes;
    *job = (k3_full_prefetch_job){weights->data, limit};
    return pthread_create(thread, NULL, full_prefetch_worker, job) == 0;
}

static int full_is_mla(int layer);
static int full_max(k3_full_model *m, float *data, int count);

static int full_profile_report(k3_full_model *m, const k3_full_options *o) {
    if (!m->profile.enabled) return 0;
    float layer_mean[K3_LAYERS], layer_peak[K3_LAYERS];
    float phase_mean[K3_FULL_PROFILE_PHASES], phase_peak[K3_FULL_PROFILE_PHASES];
    float collective_mean[1], collective_peak[1];
    uint64_t layer_samples = m->debug_layer_index >= 0 ?
        m->profile.layer_count : m->profile.layer_count / K3_LAYERS;
    if (!layer_samples) return 0;
    for (int i = 0; i < K3_LAYERS; ++i) {
        layer_mean[i] = (float)(m->profile.layer_sum[i] / layer_samples);
        layer_peak[i] = (float)m->profile.layer_max[i];
    }
    for (int i = 0; i < K3_FULL_PROFILE_PHASES; ++i) {
        uint64_t n = m->profile.phase_count[i];
        phase_mean[i] = n ? (float)(m->profile.phase_sum[i] / n) : 0.0f;
        phase_peak[i] = (float)m->profile.phase_max[i];
    }
    collective_mean[0] = m->profile.collective_count ?
        (float)(m->profile.collective_sum / m->profile.collective_count) : 0.0f;
    collective_peak[0] = (float)m->profile.collective_max;
    if (full_max(m, layer_mean, K3_LAYERS) ||
        full_max(m, layer_peak, K3_LAYERS) ||
        full_max(m, phase_mean, K3_FULL_PROFILE_PHASES) ||
        full_max(m, phase_peak, K3_FULL_PROFILE_PHASES) ||
        full_max(m, collective_mean, 1) ||
        full_max(m, collective_peak, 1)) return EIO;
    if (g_rank != 0) return 0;

    char default_path[1024];
    const char *path = o->profile_output;
    if (!path) {
        snprintf(default_path, sizeof default_path, "%s.profile",
                 o->output_path ? o->output_path : "k3_full_runner.out");
        path = default_path;
    }
    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "k3_full_runner: cannot write profile %s: %s\n",
                path, strerror(errno));
        return EIO;
    }
    double layer_sum = 0.0;
    int active_layers = m->debug_layer_index >= 0 ? 1 : K3_LAYERS;
    int over_budget = 0;
    fprintf(f, "K3FULL_PROFILE version=1 nodes=%d samples=%llu\n",
            g_nodes, (unsigned long long)layer_samples);
    for (int i = 0; i < K3_LAYERS; ++i) {
        double ms = layer_mean[i] * 1000.0;
        if (m->debug_layer_index < 0 || i == m->debug_layer_index) {
            layer_sum += ms;
            if (ms > 1.08) ++over_budget;
        }
        fprintf(f, "layer=%d type=%s rank_max_mean_ms=%.6f rank_max_sample_ms=%.6f\n",
                i, full_is_mla(i) ? "mla" : "kda", ms,
                (double)layer_peak[i] * 1000.0);
    }
    fprintf(f, "layer_ms_rank_max_mean=%.6f layer_ms_rank_max_avg=%.6f "
               "layers_over_1p08=%d\n",
            layer_sum / active_layers, layer_sum / active_layers, over_budget);
    for (int i = 1; i < K3_FULL_PROFILE_PHASES; ++i)
        fprintf(f, "phase=%s rank_max_mean_ms=%.6f rank_max_sample_ms=%.6f\n",
                full_profile_phase_name(i), (double)phase_mean[i] * 1000.0,
                (double)phase_peak[i] * 1000.0);
    fprintf(f, "collective_rank_max_mean_ms=%.6f collective_rank_max_sample_ms=%.6f\n",
            (double)collective_mean[0] * 1000.0,
            (double)collective_peak[0] * 1000.0);
    fclose(f);
    printf("K3FULL_PROFILE path=%s rank_max_layer_ms=%.6f over_1p08=%d\n",
           path, layer_sum / active_layers, over_budget);
    return 0;
}

static void full_drain_mrq(void) {
    struct utofu_mrq_notice notice;
    while (utofu_poll_mrq(g_vcq, 0, &notice) == UTOFU_SUCCESS) {
    }
}

static int full_topology(const char *path,
                         uint8_t coords[][TOFU_NCOORDS]) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "k3_full_runner: cannot open topology %s: %s\n",
                path, strerror(errno));
        return -1;
    }
    char line[256];
    int count = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        unsigned rank, c[TOFU_NCOORDS];
        if (count >= K3_FULL_MAX_NODES ||
            sscanf(line, "%u %u %u %u %u %u %u", &rank, &c[0], &c[1],
                   &c[2], &c[3], &c[4], &c[5]) != 7 ||
            (int)rank != count) {
            fprintf(stderr, "k3_full_runner: malformed topology line: %s", line);
            fclose(f);
            return -1;
        }
        for (int i = 0; i < TOFU_NCOORDS; ++i) {
            if (c[i] > UINT8_MAX) {
                fclose(f);
                return -1;
            }
            coords[count][i] = (uint8_t)c[i];
        }
        for (int p = 0; p < count; ++p) {
            if (!memcmp(coords[p], coords[count], TOFU_NCOORDS)) {
                fprintf(stderr, "k3_full_runner: duplicate topology rank %d\n", count);
                fclose(f);
                return -1;
            }
        }
        ++count;
    }
    fclose(f);
    return count;
}

static size_t full_bar_recv_off(int rank) {
    return g_bar_base + (size_t)rank * g_slot_bar;
}

static size_t full_bar_go_off(void) {
    return g_bar_base + (size_t)g_nodes * g_slot_bar;
}

static int full_put(utofu_vcq_id_t peer, utofu_stadd_t src,
                    utofu_stadd_t dst, size_t bytes) {
    int rc;
    void *cb;
    do {
        rc = utofu_put(g_vcq, peer, src, dst, bytes, 0, g_put_flags, NULL);
        if (rc == UTOFU_ERR_BUSY) (void)utofu_poll_tcq(g_vcq, 0, &cb);
    } while (rc == UTOFU_ERR_BUSY);
    if (rc != UTOFU_SUCCESS) return rc;
    do {
        rc = utofu_poll_tcq(g_vcq, 0, &cb);
    } while (rc == UTOFU_ERR_NOT_FOUND);
    full_drain_mrq();
    return rc;
}

/* The barrier source line is rewritten for every generation and then read by
 * uTofu as an RDMA source.  Clean it to the point of coherence before issuing
 * Put; otherwise a dirty A64FX cache line can leave the transport reading the
 * previous token.  Keep the source line resident: the transport may consume it
 * again for the next peer Put. */
static inline void full_barrier_publish(volatile uint64_t *p, uint64_t token) {
    *p = token;
#if defined(__aarch64__)
    __asm__ __volatile__("dc cvac, %0" :: "r"(p) : "memory");
    __asm__ __volatile__("dsb sy" ::: "memory");
#else
    __sync_synchronize();
#endif
}

/* A remote Put lands in DRAM while the receiver may still hold an old clean
 * copy of the receive-only slot.  A64FX exposes clean+invalidate at EL0 (but
 * not invalidate-only), so establish a clean baseline before registration and
 * use the supported operation while polling.  These slots are never CPU-written
 * after that baseline. */
static inline void full_barrier_invalidate(const volatile uint64_t *p) {
#if defined(__aarch64__)
    __asm__ __volatile__("dc civac, %0" :: "r"(p) : "memory");
    __asm__ __volatile__("dsb sy" ::: "memory");
#else
    (void)p;
    __sync_synchronize();
#endif
}

static int full_wait_ge(volatile uint64_t *value, uint64_t want) {
    double start = full_now();
    for (;;) {
        full_drain_mrq();
        full_barrier_invalidate(value);
        if (*value >= want) return 0;
        if (full_now() - start > K3_FULL_WAIT_SECONDS) return -1;
    }
}

static void full_barrier(void) {
    uint64_t token = ++g_bar_token;
    char *send = g_region + g_send_off;
    if (g_rank == 0) {
        for (int r = 1; r < g_nodes; ++r) {
            volatile uint64_t *recv =
                (volatile uint64_t *)(g_region + full_bar_recv_off(r));
            if (full_wait_ge(recv, token)) {
                full_barrier_invalidate(recv);
                fprintf(stderr, "k3_full_runner: barrier timeout from rank %d "
                        "token=%llu got=%llu\n", r,
                        (unsigned long long)token,
                        (unsigned long long)*recv);
                exit(3);
            }
        }
        for (int r = 1; r < g_nodes; ++r) {
            full_barrier_publish((volatile uint64_t *)send, token);
            if (full_put(g_peer_vcq[r], g_base + g_send_off,
                         g_peer_base[r] + full_bar_go_off(), sizeof token) != UTOFU_SUCCESS) {
                fprintf(stderr, "k3_full_runner: barrier release failed for rank %d\n", r);
                exit(3);
            }
        }
    } else {
        volatile uint64_t *go = (volatile uint64_t *)(g_region + full_bar_go_off());
        double start = full_now();
        do {
            full_barrier_publish((volatile uint64_t *)send, token);
            if (full_put(g_peer_vcq[0], g_base + g_send_off,
                         g_peer_base[0] + full_bar_recv_off(g_rank), sizeof token) != UTOFU_SUCCESS) {
                fprintf(stderr, "k3_full_runner: barrier fan-in failed\n");
                exit(3);
            }
            for (int i = 0; i < 50; ++i) {
                full_drain_mrq();
                full_barrier_invalidate(go);
                if (*go >= token)
                    break;
                usleep(K3_FULL_BARRIER_RETRY_USEC);
            }
            if (*go >= token)
                break;
            if (full_now() - start > K3_FULL_WAIT_SECONDS) {
                full_barrier_invalidate(go);
                fprintf(stderr, "k3_full_runner: barrier release timeout rank=%d token=%llu got=%llu\n",
                        g_rank, (unsigned long long)token, (unsigned long long)*go);
                exit(3);
            }
        } while (1);
    }
}

static int full_dtype(const char *s) {
    if (!strcmp(s, "BF16")) return 1;
    if (!strcmp(s, "F32")) return 2;
    if (!strcmp(s, "U8")) return 3;
    if (!strcmp(s, "Q8P16")) return K3_FULL_DTYPE_Q8P16;
    if (!strcmp(s, "Q8P8")) return K3_FULL_DTYPE_Q8P8;
    if (!strcmp(s, "Q8_0")) return K3_FULL_DTYPE_Q8_0;
    if (!strcmp(s, "IQ1_S")) return K3_FULL_DTYPE_IQ1_S;
    if (!strcmp(s, "IQ2_XS")) return K3_FULL_DTYPE_IQ2_XS;
    if (!strcmp(s, "IQ2_XXS")) return K3_FULL_DTYPE_IQ2_XXS;
    if (!strcmp(s, "IQ3_XXS")) return K3_FULL_DTYPE_IQ3_XXS;
    return 0;
}

static int full_parse_entry(char *line, k3_full_entry *out) {
    char *save = NULL;
    char *tok = strtok_r(line, " \t\r\n", &save);
    if (!tok) return 0;
    char *end = NULL;
    out->offset = strtoull(tok, &end, 10);
    if (!end || *end) return -1;
    tok = strtok_r(NULL, " \t\r\n", &save);
    if (!tok) return -1;
    out->nbytes = strtoull(tok, &end, 10);
    if (!end || *end) return -1;
    tok = strtok_r(NULL, " \t\r\n", &save);
    if (!tok) return -1;
    out->dtype = full_dtype(tok);
    tok = strtok_r(NULL, " \t\r\n", &save);
    if (!tok) return -1;
    out->ndims = (int)strtol(tok, &end, 10);
    if (!end || *end || out->ndims < 0 || out->ndims > 3) return -1;
    for (int i = 0; i < out->ndims; ++i) {
        tok = strtok_r(NULL, " \t\r\n", &save);
        if (!tok) return -1;
        out->shape[i] = (size_t)strtoull(tok, &end, 10);
        if (!end || *end) return -1;
    }
    tok = strtok_r(NULL, " \t\r\n", &save);
    if (!tok || !out->dtype) return -1;
    if (strlen(tok) >= sizeof out->name) return -1;
    strcpy(out->name, tok);
    return 1;
}

static int full_load_manifest(const char *path, k3_full_entry *entries,
                              int cap, int *count_out,
                              k3_full_manifest *meta) {
    FILE *f = fopen(path, "r");
    if (!f) return errno ? errno : EIO;
    char line[2048];
    int count = 0;
    memset(meta, 0, sizeof *meta);
    meta->version = 1;
    meta->layer_index = -1;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#') {
            unsigned long long bytes;
            int rank, nodes, layer;
            if (sscanf(line, "# K3FULLV3 mode=%63s rank=%d nodes=%d layer_index=%d "
                       "tensors=%*d blob_bytes=%llu", meta->mode, &rank, &nodes,
                       &layer, &bytes) == 5) {
                meta->version = 3;
                meta->rank = rank;
                meta->nodes = nodes;
                meta->layer_index = layer;
                meta->blob_bytes = (size_t)bytes;
            } else if (sscanf(line, "# K3FULLV2 mode=%63s rank=%d nodes=%d layer_index=%d "
                       "tensors=%*d blob_bytes=%llu", meta->mode, &rank, &nodes,
                       &layer, &bytes) == 5) {
                meta->version = 2;
                meta->rank = rank;
                meta->nodes = nodes;
                meta->layer_index = layer;
                meta->blob_bytes = (size_t)bytes;
            } else if (sscanf(line, "# K3FULLV1 rank=%d nodes=%d tensors=%*d "
                                "blob_bytes=%llu", &rank, &nodes, &bytes) == 3) {
                meta->version = 1;
                meta->rank = rank;
                meta->nodes = nodes;
                meta->layer_index = -1;
                snprintf(meta->mode, sizeof meta->mode, "full96");
                meta->blob_bytes = (size_t)bytes;
            }
            continue;
        }
        if (count >= cap) {
            fclose(f);
            return E2BIG;
        }
        int rc = full_parse_entry(line, &entries[count]);
        if (rc < 0) {
            fclose(f);
            return EINVAL;
        }
        if (rc > 0) ++count;
    }
    if (ferror(f)) {
        fclose(f);
        return EIO;
    }
    fclose(f);
    *count_out = count;
    return 0;
}

static int full_check_ranges(const k3_full_entry *entries, int count,
                             size_t blob_bytes, size_t actual_bytes) {
    if (!blob_bytes || blob_bytes > actual_bytes) return EINVAL;
    for (int i = 0; i < count; ++i) {
        size_t end = (size_t)entries[i].offset + (size_t)entries[i].nbytes;
        if (entries[i].offset > blob_bytes || end > blob_bytes ||
            entries[i].offset % 256 != 0) return EINVAL;
        for (int j = 0; j < i; ++j) {
            size_t other_end = (size_t)entries[j].offset + (size_t)entries[j].nbytes;
            if (!((end <= entries[j].offset) || (other_end <= entries[i].offset)))
                return EINVAL;
        }
    }
    return 0;
}

static const k3_full_entry *full_find(const k3_full_entry *entries, int count,
                                      const char *name) {
    int lo = 0, hi = count;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        int cmp = strcmp(entries[mid].name, name);
        if (cmp == 0) return &entries[mid];
        if (cmp < 0) lo = mid + 1;
        else hi = mid;
    }
    return NULL;
}

static int full_entry_name_cmp(const void *a, const void *b) {
    const k3_full_entry *ea = (const k3_full_entry *)a;
    const k3_full_entry *eb = (const k3_full_entry *)b;
    return strcmp(ea->name, eb->name);
}

static k3_full_tensor full_tensor(const k3_full_entry *entries, int count,
                                  const uint8_t *blob, const char *name) {
    const k3_full_entry *e = full_find(entries, count, name);
    k3_full_tensor t;
    memset(&t, 0, sizeof t);
    if (!e) {
        fprintf(stderr, "k3_full_runner: missing staged tensor %s\n", name);
        return t;
    }
    t.data = blob + e->offset;
    t.nbytes = (size_t)e->nbytes;
    t.dtype = e->dtype;
    t.ndims = e->ndims;
    memcpy(t.shape, e->shape, sizeof t.shape);
    snprintf(t.name, sizeof t.name, "%s", name);
    return t;
}

static int full_tensor_valid(const k3_full_tensor *t) {
    return t && t->data && t->ndims > 0 && t->nbytes > 0;
}

static int full_split(int total, int rank, int size, int *first, int *count) {
    int base = total / size;
    int rem = total % size;
    *first = rank * base + (rank < rem ? rank : rem);
    *count = base + (rank < rem ? 1 : 0);
    return 0;
}

static int full_split_groups(int total, int rank, int size, int group,
                             int *first, int *count) {
    if (group <= 0 || total % group) return EINVAL;
    int first_group, group_count;
    full_split(total / group, rank, size, &first_group, &group_count);
    *first = first_group * group;
    *count = group_count * group;
    return 0;
}

typedef struct {
    float *out;
    const uint16_t *weights;
    const float *f32_weights;
    const uint8_t *packed;
    const float *input;
    const uint8_t *quant;
    const k3_quant_workspace *quant_ws;
    const k3_quant_packed *quant_packed;
    const int8_t *quant_packed_data;
    const int8_t *quant_packed_scales_data;
    const uint16_t *quant_packed_ds_data;
    size_t quant_row_bytes;
    int quant_mode;
    int rows;
    int cols;
    int dtype;
    int pv;
    int cmg;                /* CMG owning these rows, -1 if not CMG-bound */
    unsigned char *const *cmg_copy;   /* per-CMG replicas, NULL if not replicated */
    size_t woff;                      /* byte offset of this task's rows */
} full_bf16_task;

/* K3_CMG_LOCAL=1 routes each 8-row BF16 task to a thread on the CMG that
 * already owns its weight pages.
 *
 * Why routing and not migration.  The heap is backed by 2 MB large pages, so
 * MPOL_INTERLEAVE round-robins whole 2 MB blocks -- a 7168-column bf16 row is
 * 14 KB and an 8-row task 114 KB, so a task's weights are almost always inside
 * one large page, i.e. on ONE CMG already.  What is uncontrolled is *which*:
 * with an arbitrary schedule ~3/4 of tasks run on a thread whose CMG does not
 * own them, and inter-CMG bandwidth is ~119 GB/s against 226 GB/s CMG-local
 * (k3_cmg_bw_bench.c).  Querying the existing placement and matching the thread
 * to it needs no page migration, works for tensors of any size, and inherits
 * the interleave's natural 1/4-per-CMG balance.
 *
 * mbind was tried first and is the wrong tool here: it requires 2 MB-aligned
 * ranges (sysconf(_SC_PAGESIZE) reports 64 KB and every 64 KB-aligned call
 * returns EINVAL), and at 12 nodes a per-rank projection is only 7-15 MB, i.e.
 * 3-7 large pages -- too few to cut four ways.
 *
 * Thread t sits on core 12+t under OMP_PROC_BIND=close/OMP_PLACES=cores, so its
 * CMG is t/12. */
#define K3_CMG_NODE0 4
#define K3_CMG_CORES 12
#define K3_CMG_PAGE (2UL << 20)
#ifndef MPOL_F_NODE
#define MPOL_F_NODE (1 << 0)
#endif
#ifndef MPOL_F_ADDR
#define MPOL_F_ADDR (1 << 1)
#endif
static int full_cmg_local = 0;
static int full_cmg_verify = 0;
/* K3_CMG_FORCE=<0..3> pins every mapped projection entirely onto one CMG.  This
 * exists to simulate the 96-node case: per-rank projections there are ~1.2 MB,
 * i.e. a SINGLE 2 MB large page, so each tensor lands wholly on one CMG and 36
 * of 47 threads read it across the ~119 GB/s interconnect.  At 12 nodes a
 * tensor is 3-7 pages and the interleave spreads it, so the effect is invisible
 * -- this knob makes it measurable here. */
static int full_cmg_force = -1;
static int full_cmg_replicate_on = 0;
static int full_moe_cmg_replicate_on = 0;
static int full_moe_cmg_replicate_max = 16;
/* K3_SITU_FAST=1 routes the shared-expert and dense SiTU through the FEXPA
 * kernel the routed experts already use (k3_moe.h, 5 call sites).  k3_situ_sve
 * is k3_situ_ref: 3 libm transcendentals per element (2x tanhf + sigmoidf), so
 * 512 shared lanes cost 1536 libm calls = 0.089 ms/layer, half of moe_shared.
 * This is an accuracy change -- it moves the layer hash -- but it makes the two
 * expert paths consistent rather than approximating one and not the other, and
 * `make test`'s [situ-fexpa] case bounds the error at 2e-3. */
static int full_situ_fast = 0;
static int full_shared_fused_team = 0;

/* K3_FAST_EXP=1 vectorises the two scalar-libm loops in the KDA path.
 * kda_serial measured 0.047 ms/layer and is essentially all libm: the 1024-lane
 * expf below plus k3_kda_log_decay's 128 expf + 1024 sigmoidf, ~2176 calls on
 * one thread while 46 idle.  Same FEXPA primitives the routed experts already
 * use.  Separate from K3_SITU_FAST because the error lands somewhere riskier:
 * these feed the KDA recurrent state, so it persists across tokens rather than
 * being consumed within one. */
static int full_fast_exp = 0;
static int full_kda_fused_team = 0;
/* K3_MLA_SERIAL_ATTN=1 restores the pre-optimization serial-over-heads KV scan.
 * Bisect for the MLA run-to-run non-determinism: if this is reproducible while
 * the parallel kernel is not, the kernel is implicated. */
static int full_mla_serial_attn = 0;
static int full_mla_split_proj = 0;
static int full_mla_fast_gate = 0;
/* Delay the shared-expert reduction until routed_up.  The routed-up row shard
 * is zero outside this rank's rows, so adding the local shared partial before
 * a normal final sum is algebraically equivalent to reducing shared_hidden
 * in the middle and adding it after the row allgather.  This shrinks the
 * latency-sensitive middle reduction from 512+7168 floats to 512. */
static int full_moe_late_shared_reduce = 0;
static int full_comm_rabenseifner = 0;
/* K3_MLA_TRACE=1 accumulates an FNV hash of each MLA intermediate so the first
 * point of run-to-run divergence can be located. */
static int full_mla_trace = 0;
static int full_iq_trace = 0;
static uint64_t full_trace_h[6];
static const char *full_trace_name[6] = {
    "latent(tmp,q_a)", "latent(tmp2,kv_a)", "q_b", "kv_b", "attn_out", "layer_out"
};
static void full_trace_mix(int slot, const float *v, int n) {
    if (!full_mla_trace) return;
    uint64_t h = full_trace_h[slot] ? full_trace_h[slot] : 1469598103934665603ULL;
    for (int i = 0; i < n; ++i) {
        uint32_t b; memcpy(&b, &v[i], 4);
        h = (h ^ b) * 1099511628211ULL;
    }
    full_trace_h[slot] = h;
}

static void full_exp_vec(float *out, const float *in, int n) {
#if defined(__ARM_FEATURE_SVE)
    int vl = (int)svcntw();
    for (int i = 0; i < n; i += vl) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t t = svmul_n_f32_x(pg, svld1(pg, in + i), 1.4426950408889634f);
        t = svmax_n_f32_x(pg, svmin_n_f32_x(pg, t, 126.0f), -126.0f);
        svst1(pg, out + i, k3_exp2_fexpa_sve(pg, t));
    }
#else
    for (int i = 0; i < n; ++i) out[i] = expf(in[i]);
#endif
}

static void full_kda_log_decay_fast(float *out, const float *g_raw,
                                    const float *a_log, const float *dt_bias,
                                    int heads, int key_dim) {
#if defined(__ARM_FEATURE_SVE)
    int vl = (int)svcntw();
    float a_scale[key_dim];
    full_exp_vec(a_scale, a_log, key_dim);
    for (int h = 0; h < heads; ++h)
        for (int d = 0; d < key_dim; d += vl) {
            svbool_t pg = svwhilelt_b32(d, key_dim);
            int i = h * key_dim + d;
            svfloat32_t t = svadd_f32_x(pg, svld1(pg, g_raw + i),
                                        svld1(pg, dt_bias + i));
            t = svmul_f32_x(pg, svld1(pg, a_scale + d), t);
            svst1(pg, out + i,
                  svmul_n_f32_x(pg, k3_sigmoid_fast_sve(pg, t), -5.0f));
        }
#else
    k3_kda_log_decay(out, g_raw, a_log, dt_bias, heads, key_dim);
#endif
}

/* Some GGUF conversions preserve the constructor layout A_log[num_heads]
 * instead of the release-safetensor layout A_log[key_dim].  The gate equation
 * is identical; only the axis over which exp(A_log) is broadcast changes. */
static void full_kda_log_decay_heads(float *out, const float *g_raw,
                                     const float *a_log, const float *dt_bias,
                                     int heads, int key_dim) {
    for (int h = 0; h < heads; ++h) {
        float a_scale = expf(a_log[h]);
        for (int d = 0; d < key_dim; ++d) {
            int i = h * key_dim + d;
            out[i] = -5.0f * k3_sigmoidf(a_scale *
                                          (g_raw[i] + dt_bias[i]));
        }
    }
}

static void full_situ(float *out, const float *gate, const float *up, int n) {
    if (full_situ_fast) k3_situ_fast_sve(out, gate, up, n);
    else k3_situ_sve(out, gate, up, n);
}
#ifndef MPOL_BIND
#define MPOL_BIND 2
#endif
#ifndef MPOL_MF_MOVE
#define MPOL_MF_MOVE (1 << 1)
#endif

static int full_cmg_threads(int cmg, int workers) {
    int base = cmg * K3_CMG_CORES;
    if (workers <= base) return 0;
    int n = workers - base;
    return n < K3_CMG_CORES ? n : K3_CMG_CORES;
}

/* Simulate the 96-node placement at 12 nodes: pin a whole projection to one CMG.
 * At 96 nodes a per-rank projection is ~1.2 MB, i.e. under one 2 MB large page,
 * so it lands wholly on one CMG regardless of the interleave policy. */
static void full_cmg_force_tensor(k3_full_tensor *t) {
    if (!t || !t->data || t->dtype != 1 || t->ndims != 2 || full_cmg_force < 0) return;
    uintptr_t b = (uintptr_t)t->data, e = b + t->nbytes;
    uintptr_t lo = (b + K3_CMG_PAGE - 1) & ~(uintptr_t)(K3_CMG_PAGE - 1);
    uintptr_t hi = e & ~(uintptr_t)(K3_CMG_PAGE - 1);
    if (hi <= lo) return;
    unsigned long mask = 1UL << (K3_CMG_NODE0 + full_cmg_force);
    if (syscall(SYS_mbind, (void *)lo, (size_t)(hi - lo), MPOL_BIND,
                &mask, 8 * sizeof mask, MPOL_MF_MOVE) != 0)
        fprintf(stderr, "k3: cmg-force mbind(%s): %s\n", t->name, strerror(errno));
}

/* Give every CMG its own copy of a projection, each bound to that CMG, so all
 * 47 threads read locally AND any thread can take any task -- unlike CMG routing,
 * which buys locality at the cost of a load imbalance set by the 2 MB page
 * granularity (7 pages over 4 CMGs at 12 nodes) and measured as a net loss.
 * Costs 3x the bytes of the replicated tensors. */
static int full_cmg_replicate(k3_full_tensor *t) {
    if (!t || !t->data || t->dtype != 1 || t->ndims != 2 || t->cmg_copy[0]) return 0;
    size_t n = t->nbytes, alloc = (n + K3_CMG_PAGE - 1) & ~(size_t)(K3_CMG_PAGE - 1);
    for (int c = 0; c < K3_CMG_COUNT; ++c) {
        void *p = NULL;
        if (posix_memalign(&p, K3_CMG_PAGE, alloc) != 0) return ENOMEM;
        unsigned long mask = 1UL << (K3_CMG_NODE0 + c);
        if (syscall(SYS_mbind, p, alloc, MPOL_BIND, &mask,
                    8 * sizeof mask, MPOL_MF_MOVE) != 0) {
            fprintf(stderr, "k3: replicate mbind(%s, cmg=%d): %s\n",
                    t->name, c, strerror(errno));
            free(p);
            for (int k = 0; k < c; ++k) { free(t->cmg_copy[k]); t->cmg_copy[k] = NULL; }
            return 0;
        }
        memcpy(p, t->data, n);          /* fault in under the binding */
        t->cmg_copy[c] = p;
    }
    if (full_cmg_verify)
        fprintf(stderr, "k3: cmg-replicate %s %.2f MB x4\n", t->name, n / 1048576.0);
    return 0;
}

/* The TP expert slices are small enough to fit in one CMG page at this
 * 12-node probe.  Replicate only selected MXFP4 matrices, on first use, so
 * the decode kernel can read every expert row locally without copying the
 * complete 896-expert layer. */
static int full_mxfp4_replicate(k3_mxfp4_matrix *m) {
    if (!m || !m->packed || m->packed_cmg[0]) return 0;
    size_t packed_n = (size_t)m->rows * (size_t)m->cols / 2;
    size_t scale_n = (size_t)m->rows * (size_t)m->cols / 32;
    size_t packed_alloc = (packed_n + K3_CMG_PAGE - 1) & ~(size_t)(K3_CMG_PAGE - 1);
    size_t scale_alloc = (scale_n + K3_CMG_PAGE - 1) & ~(size_t)(K3_CMG_PAGE - 1);
    for (int c = 0; c < K3_CMG_COUNT; ++c) {
        void *packed = NULL, *scale = NULL;
        unsigned long mask = 1UL << (K3_CMG_NODE0 + c);
        if (posix_memalign(&packed, K3_CMG_PAGE, packed_alloc) != 0 ||
            posix_memalign(&scale, K3_CMG_PAGE, scale_alloc) != 0 ||
            syscall(SYS_mbind, packed, packed_alloc, MPOL_BIND, &mask,
                    8 * sizeof mask, MPOL_MF_MOVE) != 0 ||
            syscall(SYS_mbind, scale, scale_alloc, MPOL_BIND, &mask,
                    8 * sizeof mask, MPOL_MF_MOVE) != 0) {
            free(packed); free(scale);
            for (int k = 0; k < c; ++k) {
                free((void *)m->packed_cmg[k]);
                free((void *)m->scale_cmg[k]);
                m->packed_cmg[k] = m->scale_cmg[k] = NULL;
            }
            return ENOMEM;
        }
        memcpy(packed, m->packed, packed_n);
        memcpy(scale, m->scale, scale_n);
        m->packed_cmg[c] = (const uint8_t *)packed;
        m->scale_cmg[c] = (const uint8_t *)scale;
    }
    return 0;
}

static int full_mxfp4_replicate_expert(k3_full_expert *expert) {
    int rc = full_mxfp4_replicate(&expert->mw2);
    rc |= full_mxfp4_replicate(&expert->mw1);
    rc |= full_mxfp4_replicate(&expert->mw3);
    return rc;
}


static int full_cmg_map_tensor(k3_full_tensor *t) {
    if (!t || !t->data || t->dtype != 1 || t->ndims != 2 || t->cmg_map) return 0;
    size_t npage = (t->nbytes + K3_CMG_PAGE - 1) / K3_CMG_PAGE;
    if (npage < 2) return 0;                 /* one page: nothing to route */
    unsigned char *map = malloc(npage);
    if (!map) return 0;
    int hist[K3_CMG_COUNT] = {0};
    for (size_t p = 0; p < npage; ++p) {
        int node = -1;
        void *a = (char *)(uintptr_t)t->data + p * K3_CMG_PAGE;
        if (syscall(SYS_get_mempolicy, &node, NULL, 0UL, a,
                    MPOL_F_NODE | MPOL_F_ADDR) != 0 ||
            node < K3_CMG_NODE0 || node >= K3_CMG_NODE0 + K3_CMG_COUNT) {
            free(map);
            return 0;                        /* unknown placement: stay generic */
        }
        map[p] = (unsigned char)(node - K3_CMG_NODE0);
        ++hist[map[p]];
    }
    t->cmg_map = map;
    if (full_cmg_verify)
        fprintf(stderr, "k3: cmg-map %s pages=%zu per-cmg=%d/%d/%d/%d\n",
                t->name, npage, hist[0], hist[1], hist[2], hist[3]);
    return 1;
}

/* K3_BF16_PV=1 repacks BF16 projection weights in place, once at load, into the
 * pair-interleaved layout consumed by matvec_bf16_8row_pv.  That kernel replaces
 * the SVE_BF16_ZIP widen with a p_odd predicated load (2 instructions instead of
 * 4) and measures 1.57x at cols=7168 -- see ROOFLINE.md.  Repacking at load
 * rather than at stage time keeps k3_full_stage.py and the K3FULLV1/V2 blob
 * format untouched, and the layout is recorded per tensor so a tensor that is
 * not repacked simply keeps using the row-major kernel.
 *
 * Layout, derived from the kernel: for an 8-row group the block holds four
 * pair-planes of 2*cols halfwords each, and within a plane
 * plane[2*c] = rowEven[c], plane[2*c+1] = rowOdd[c].  The group base stays
 * w + r*cols, so full_bf16_many's task construction is unchanged.  Only whole
 * 8-row groups are repacked; a short tail stays row-major for the scalar path. */
static int full_bf16_pv = 0;

static int full_bf16_pv_repack(k3_full_tensor *t) {
    if (!t || t->dtype != 1 || t->ndims != 2) return 0;
    size_t rows = t->shape[0], cols = t->shape[1];
    if (rows < 8 || (cols % 16u) != 0) return 0;   /* kernel needs n % vl == 0 */
    uint16_t *w = (uint16_t *)(uintptr_t)t->data;
    uint16_t *tmp = malloc(8 * cols * sizeof *tmp);
    if (!tmp) return ENOMEM;
    for (size_t g = 0; g + 8 <= rows; g += 8) {
        uint16_t *base = w + g * cols;
        memcpy(tmp, base, 8 * cols * sizeof *tmp);
        for (int pair = 0; pair < 4; ++pair) {
            uint16_t *plane = base + (size_t)pair * 2 * cols;
            const uint16_t *e = tmp + (size_t)(2 * pair) * cols;
            const uint16_t *o = tmp + (size_t)(2 * pair + 1) * cols;
            for (size_t c = 0; c < cols; ++c) {
                plane[2 * c] = e[c];
                plane[2 * c + 1] = o[c];
            }
        }
    }
    free(tmp);
    t->pv = 1;
    return 0;
}

/* K3_BF16_ROWS=4 splits each 8-row bf16 task into two 4-row kernel calls. */
static int full_bf16_rows = 8;
/* K3_BF16_PREFETCH=<elements> enables the software-prefetched 8-row variant
 * below.  matvec_bf16_8row streams eight weight rows with no SW prefetch and
 * measures 12.7 GB/s single-threaded against a 42 GB/s single-thread node
 * bandwidth; this is the A/B for whether that gap is prefetch. */
static int full_bf16_prefetch = 0;
static int full_quant_packed = 1;
static int full_quant_packed_nibble = 1;
static int full_serial_vector_ops = 0;

static void full_bf16_rows_init(void) {
    const char *env = getenv("K3_BF16_ROWS");
    if (env && env[0] == '4') full_bf16_rows = 4;
    env = getenv("K3_CMG_LOCAL");
    if (env && env[0] == '1') full_cmg_local = 1;
    env = getenv("K3_MLA_TRACE");
    if (env && env[0] == '1') full_mla_trace = 1;
    env = getenv("K3_IQ_FULL_TRACE");
    if (env && env[0] == '1') full_iq_trace = 1;
    env = getenv("K3_MLA_SERIAL_ATTN");
    if (env && env[0] == '1') full_mla_serial_attn = 1;
    env = getenv("K3_MLA_SPLIT_PROJ");
    if (env && env[0] == '1') full_mla_split_proj = 1;
    env = getenv("K3_MLA_FAST_GATE");
    if (env && env[0] == '1') full_mla_fast_gate = 1;
    env = getenv("K3_MOE_LATE_SHARED_REDUCE");
    if (env && env[0] == '1') full_moe_late_shared_reduce = 1;
    env = getenv("K3_COMM_RABENSEIFNER");
    if (env && *env) full_comm_rabenseifner = atoi(env);
    env = getenv("K3_MOE_SCALE_ACTIVATION");
    if (env && *env) k3_tp_scale_activation = atoi(env) != 0;
    env = getenv("K3_FAST_EXP");
    if (env && env[0] == '1') full_fast_exp = 1;
    env = getenv("K3_SITU_FAST");
    if (env && env[0] == '1') full_situ_fast = 1;
    env = getenv("K3_SHARED_FUSED_TEAM");
    if (env && env[0] == '1') full_shared_fused_team = 1;
    env = getenv("K3_CMG_REPLICATE");
    if (env && env[0] == '1') { full_cmg_replicate_on = 1; full_cmg_local = 1; }
    env = getenv("K3_MOE_CMG_REPLICATE");
    if (env && env[0] == '1') full_moe_cmg_replicate_on = 1;
    env = getenv("K3_MOE_CMG_REPLICATE_MAX");
    if (env && *env) full_moe_cmg_replicate_max = atoi(env);
    env = getenv("K3_CMG_FORCE");
    if (env && *env) { full_cmg_force = atoi(env); full_cmg_local = 1; }
    env = getenv("K3_CMG_VERIFY");
    if (env && env[0] == '1') { full_cmg_local = 1; full_cmg_verify = 1; }
    env = getenv("K3_BF16_PV");
    if (env && env[0] == '1') full_bf16_pv = 1;
    env = getenv("K3_BF16_PREFETCH");
    if (env && *env) {
        int v = atoi(env);
        if (v > 0) full_bf16_prefetch = v;
    }
    env = getenv("K3_QUANT_PACKED");
    if (env && *env) full_quant_packed = atoi(env) != 0;
    env = getenv("K3_QUANT_PACKED_NIBBLE");
    if (env && *env) full_quant_packed_nibble = atoi(env) != 0;
    env = getenv("K3_SERIAL_VECTOR_OPS");
    if (env && *env) full_serial_vector_ops = atoi(env) != 0;
}

#if defined(__ARM_FEATURE_SVE)
/* Byte-identical to matvec_bf16_8row: same column order, same lo/hi split, same
 * per-row svaddv.  The only difference is the __builtin_prefetch pair per row. */
static void full_matvec_bf16_8row_pf(float *dst,
        const uint16_t *w0, const uint16_t *w1, const uint16_t *w2, const uint16_t *w3,
        const uint16_t *w4, const uint16_t *w5, const uint16_t *w6, const uint16_t *w7,
        const float *x, int n, int pd) {
    int i = 0;
    svfloat32_t a0l=svdup_f32(0),a1l=svdup_f32(0),a2l=svdup_f32(0),a3l=svdup_f32(0);
    svfloat32_t a4l=svdup_f32(0),a5l=svdup_f32(0),a6l=svdup_f32(0),a7l=svdup_f32(0);
    svfloat32_t a0h=svdup_f32(0),a1h=svdup_f32(0),a2h=svdup_f32(0),a3h=svdup_f32(0);
    svfloat32_t a4h=svdup_f32(0),a5h=svdup_f32(0),a6h=svdup_f32(0),a7h=svdup_f32(0);
    int vl = (int)svcntw(), vlh = (int)svcnth();
    svbool_t pg = svptrue_b32(), pgh = svptrue_b16();
    svuint16_t zero = svdup_u16(0);
    for (; i + vlh - 1 < n; i += vlh) {
        if (i + pd < n) {
            __builtin_prefetch(&w0[i+pd],0,3); __builtin_prefetch(&w1[i+pd],0,3);
            __builtin_prefetch(&w2[i+pd],0,3); __builtin_prefetch(&w3[i+pd],0,3);
            __builtin_prefetch(&w4[i+pd],0,3); __builtin_prefetch(&w5[i+pd],0,3);
            __builtin_prefetch(&w6[i+pd],0,3); __builtin_prefetch(&w7[i+pd],0,3);
        }
        svfloat32_t vxl = svld1(pg, &x[i]), vxh = svld1(pg, &x[i + vl]);
        svfloat32_t wl, wh;
        SVE_BF16_ZIP(svld1_u16(pgh,&w0[i]),zero,wl,wh); a0l=svmla_x(pg,a0l,wl,vxl); a0h=svmla_x(pg,a0h,wh,vxh);
        SVE_BF16_ZIP(svld1_u16(pgh,&w1[i]),zero,wl,wh); a1l=svmla_x(pg,a1l,wl,vxl); a1h=svmla_x(pg,a1h,wh,vxh);
        SVE_BF16_ZIP(svld1_u16(pgh,&w2[i]),zero,wl,wh); a2l=svmla_x(pg,a2l,wl,vxl); a2h=svmla_x(pg,a2h,wh,vxh);
        SVE_BF16_ZIP(svld1_u16(pgh,&w3[i]),zero,wl,wh); a3l=svmla_x(pg,a3l,wl,vxl); a3h=svmla_x(pg,a3h,wh,vxh);
        SVE_BF16_ZIP(svld1_u16(pgh,&w4[i]),zero,wl,wh); a4l=svmla_x(pg,a4l,wl,vxl); a4h=svmla_x(pg,a4h,wh,vxh);
        SVE_BF16_ZIP(svld1_u16(pgh,&w5[i]),zero,wl,wh); a5l=svmla_x(pg,a5l,wl,vxl); a5h=svmla_x(pg,a5h,wh,vxh);
        SVE_BF16_ZIP(svld1_u16(pgh,&w6[i]),zero,wl,wh); a6l=svmla_x(pg,a6l,wl,vxl); a6h=svmla_x(pg,a6h,wh,vxh);
        SVE_BF16_ZIP(svld1_u16(pgh,&w7[i]),zero,wl,wh); a7l=svmla_x(pg,a7l,wl,vxl); a7h=svmla_x(pg,a7h,wh,vxh);
    }
    a0l=svadd_x(pg,a0l,a0h);a1l=svadd_x(pg,a1l,a1h);a2l=svadd_x(pg,a2l,a2h);a3l=svadd_x(pg,a3l,a3h);
    a4l=svadd_x(pg,a4l,a4h);a5l=svadd_x(pg,a5l,a5h);a6l=svadd_x(pg,a6l,a6h);a7l=svadd_x(pg,a7l,a7h);
    dst[0]=svaddv(pg,a0l); dst[1]=svaddv(pg,a1l); dst[2]=svaddv(pg,a2l); dst[3]=svaddv(pg,a3l);
    dst[4]=svaddv(pg,a4l); dst[5]=svaddv(pg,a5l); dst[6]=svaddv(pg,a6l); dst[7]=svaddv(pg,a7l);
    /* Scalar tail.  Every K3 bf16 matvec has cols a multiple of 32, so this is
     * dead in practice; it exists so the kernel is correct for any n, and it is
     * the one place this variant may not be bit-identical to the 8-row form. */
    if (i < n) {
        const uint16_t *w[8] = {w0,w1,w2,w3,w4,w5,w6,w7};
        for (int r = 0; r < 8; ++r) {
            float s = 0.0f;
            for (int c = i; c < n; ++c) s += bf16_to_f32_scalar(w[r][c]) * x[c];
            dst[r] += s;
        }
    }
}
#endif

/* Resolve a replicated task onto the running thread's own CMG.  Thread t is on
 * core 12+t under OMP_PROC_BIND=close, so its CMG is t/12. */
static void full_bf16_run_task(const full_bf16_task *task);
static void full_bf16_run_task_local(const full_bf16_task *task) {
#if defined(_OPENMP)
    if (task->cmg_copy) {
        int c = omp_get_thread_num() / K3_CMG_CORES;
        if (c >= K3_CMG_COUNT) c = K3_CMG_COUNT - 1;
        if (task->cmg_copy[c]) {
            full_bf16_task k = *task;
            k.weights = (const uint16_t *)(task->cmg_copy[c] + task->woff);
            full_bf16_run_task(&k);
            return;
        }
    }
#endif
    full_bf16_run_task(task);
}

static void full_bf16_run_task(const full_bf16_task *task) {
    if (task->dtype >= K3_FULL_DTYPE_Q8_0 && task->dtype <= K3_FULL_DTYPE_IQ3_XXS) {
        int qt = K3_Q_Q8_0 + (task->dtype - K3_FULL_DTYPE_Q8_0);
        k3_quant_matrix qm = {task->quant, qt, task->rows, task->cols,
                              task->quant_row_bytes};
        k3_quant_packed qp;
        const k3_quant_packed *qpp = NULL;
        if (task->quant_packed && task->quant_packed_data) {
            qp = *task->quant_packed;
            qp.data = (int8_t *)task->quant_packed_data;
            qp.scales = (int8_t *)task->quant_packed_scales_data;
            qp.ds = (uint16_t *)task->quant_packed_ds_data;
            qp.rows = task->rows;
            qpp = &qp;
        }
        int qrc = qpp && task->quant_ws ?
            k3_quant_matvec_packed_ws(task->out, &qm, qpp,
                                      task->quant_ws) : task->quant_ws ?
            k3_quant_matvec_ws(task->out, &qm, task->input, task->quant_ws,
                               0, task->quant_mode) :
            k3_quant_matvec_mode(task->out, &qm, task->input, 0,
                                 task->quant_mode);
        if (qrc) {
            for (int r = 0; r < task->rows; ++r)
                task->out[r] = k3_quant_dot_row(task->quant +
                    (size_t)r * task->quant_row_bytes, qt, task->input, task->cols);
        }
    } else if (task->dtype == K3_FULL_DTYPE_Q8P16 && task->rows == 8) {
        k3_matvec_q8pv16_f32_group(task->out, task->packed,
                                   task->input, task->cols);
    } else if (task->dtype == K3_FULL_DTYPE_Q8P8 && task->rows == 8) {
        k3_matvec_q8pv8_f32_group(task->out, task->packed,
                                  task->input, task->cols);
    } else if (task->dtype == 2) {
        for (int r = 0; r < task->rows; ++r)
            task->out[r] = k3_dot_sve(
                task->f32_weights + (size_t)r * task->cols,
                task->input, task->cols);
#if defined(__ARM_FEATURE_SVE)
    } else if (task->pv) {
        /* Four pair-planes of 2*cols halfwords; see full_bf16_pv_repack. */
        const uint16_t *p = task->weights;
        matvec_bf16_8row_pv(task->out, p,
                            p + (size_t)2 * task->cols,
                            p + (size_t)4 * task->cols,
                            p + (size_t)6 * task->cols,
                            task->input, task->cols);
    } else if (task->dtype == 1 && task->rows == 8 && full_bf16_prefetch > 0) {
        const uint16_t *p = task->weights;
        full_matvec_bf16_8row_pf(task->out, p, p + task->cols,
                                 p + (size_t)2 * task->cols,
                                 p + (size_t)3 * task->cols,
                                 p + (size_t)4 * task->cols,
                                 p + (size_t)5 * task->cols,
                                 p + (size_t)6 * task->cols,
                                 p + (size_t)7 * task->cols,
                                 task->input, task->cols, full_bf16_prefetch);
#endif
    } else if (task->dtype == 1 && task->rows == 8 && full_bf16_rows == 4) {
        /* Same rows, same column order, same per-row svaddv reduction as the
         * 8-row form, so the result is byte-identical -- but 5 concurrent
         * streams and 8 accumulators instead of 9 and 16.  A64FX is
         * prefetch-stream limited and has 32 SVE registers, so the 8-row form
         * may be spilling; this is the A/B for that. */
        const uint16_t *p = task->weights;
        matvec_bf16_4row(task->out, p, p + task->cols,
                         p + (size_t)2 * task->cols,
                         p + (size_t)3 * task->cols,
                         task->input, task->cols);
        matvec_bf16_4row(task->out + 4,
                         p + (size_t)4 * task->cols,
                         p + (size_t)5 * task->cols,
                         p + (size_t)6 * task->cols,
                         p + (size_t)7 * task->cols,
                         task->input, task->cols);
    } else if (task->dtype == 1 && task->rows == 8) {
        const uint16_t *p = task->weights;
        matvec_bf16_8row(task->out, p, p + task->cols,
                         p + (size_t)2 * task->cols,
                         p + (size_t)3 * task->cols,
                         p + (size_t)4 * task->cols,
                         p + (size_t)5 * task->cols,
                         p + (size_t)6 * task->cols,
                         p + (size_t)7 * task->cols,
                         task->input, task->cols);
    } else {
        for (int r = 0; r < task->rows; ++r) {
            const uint16_t *p = task->weights + (size_t)r * task->cols;
            float sum = 0.0f;
            for (int c = 0; c < task->cols; ++c)
                sum += bf16_to_f32_scalar(p[c]) * task->input[c];
            task->out[r] = sum;
        }
    }
}

static void full_quant_workspaces_free(k3_quant_workspace *ws,
                                       const int *owner, int count) {
    for (int i = 0; i < count; ++i)
        if (owner[i] == i) k3_quant_workspace_free(&ws[i]);
}

static int full_quant_pack_tensor(k3_full_tensor *tensor) {
    int dtype = tensor->dtype;
    if ((dtype != K3_FULL_DTYPE_IQ1_S &&
         dtype != K3_FULL_DTYPE_IQ2_XS) || tensor->ndims != 2 ||
        (tensor->shape[0] & 15) || (tensor->shape[1] & 255)) return 0;
    if (tensor->quant_packed.data) return 0;
    int qt = K3_Q_Q8_0 + (dtype - K3_FULL_DTYPE_Q8_0);
    k3_quant_matrix matrix = {
        tensor->data, qt, (int)tensor->shape[0], (int)tensor->shape[1],
        k3_quant_row_bytes(qt, (int)tensor->shape[1])};
    return !full_quant_packed_nibble || dtype != K3_FULL_DTYPE_IQ1_S ?
        k3_quant_pack_iq_rows16(&tensor->quant_packed, &matrix) :
        k3_quant_pack_iq_rows16_nibble(&tensor->quant_packed, &matrix);
}

/*
 * Keep all independent projections in one OpenMP team.  The old path opened
 * a fresh team for every BF16 matvec, even for one-head projections.  On the
 * full model that turns the small projections and repeated decode launches
 * into a large synchronization tax.  IQ1/IQ2 tasks use sixteen-row blocks
 * when their shape permits it; the other paths retain eight-row blocks so
 * the existing A64FX kernels remain the inner loop.  Short tails use the
 * scalar fallback.
 */
static void full_bf16_many(float *const *outs,
                           const k3_full_tensor *const *tensors,
                           const float *const *inputs,
                           const int *rows, const int *cols, int count,
                           int threads) {
    int task_count = 0;
    for (int i = 0; i < count; ++i) {
        int step = (tensors[i]->dtype >= K3_FULL_DTYPE_IQ1_S &&
                    tensors[i]->dtype <= K3_FULL_DTYPE_IQ2_XXS &&
                    (rows[i] & 15) == 0) ? 16 : 8;
        task_count += (rows[i] + step - 1) / step;
    }
    if (task_count == 0) return;

    for (int i = 0; i < count; ++i) {
        int dtype = tensors[i]->dtype;
        if (dtype != 1 && dtype != 2 && dtype != K3_FULL_DTYPE_Q8P16 &&
            dtype != K3_FULL_DTYPE_Q8P8 &&
            (dtype < K3_FULL_DTYPE_Q8_0 || dtype > K3_FULL_DTYPE_IQ3_XXS)) {
            fprintf(stderr, "k3_full_runner: unsupported projection dtype=%d tensor=%s\n",
                    dtype, tensors[i]->name);
            return;
        }
        if (dtype >= K3_FULL_DTYPE_Q8P16 && dtype <= K3_FULL_DTYPE_Q8P8 &&
            ((rows[i] & 7) || (cols[i] & 15))) {
            fprintf(stderr, "k3_full_runner: Q8 projection shape is not packed-compatible "
                            "tensor=%s shape=%dx%d\n",
                    tensors[i]->name, rows[i], cols[i]);
            return;
        }
    }

    k3_quant_workspace qws[count];
    int qowner[count], qmode[count];
    memset(qws, 0, sizeof qws);
    for (int i = 0; i < count; ++i) qowner[i] = -1;
    for (int i = 0; i < count; ++i) {
        int dtype = tensors[i]->dtype;
        qmode[i] = k3_quant_kernel_mode_env();
        if (dtype < K3_FULL_DTYPE_Q8_0 ||
            dtype > K3_FULL_DTYPE_IQ3_XXS ||
            qmode[i] == K3_QUANT_REFERENCE)
            continue;
        for (int j = 0; j < i; ++j)
            if (qowner[j] >= 0 && qmode[j] == qmode[i] &&
                tensors[j]->dtype == dtype && cols[j] == cols[i] &&
                inputs[j] == inputs[i]) {
                qowner[i] = qowner[j];
                break;
            }
        if (qowner[i] >= 0) continue;
        if (k3_quant_workspace_prepare(&qws[i], cols[i], qmode[i])) continue;
        k3_quant_ensure_luts();
        if (qmode[i] == K3_QUANT_SVE_Q8 && dtype != K3_FULL_DTYPE_Q8_0) {
            k3_quant_prepare_q8(&qws[i], inputs[i], cols[i]);
        } else {
            k3_quant_prepare_a16(&qws[i], inputs[i], cols[i]);
            qws[i].scale_a16 = qws[i].scale;
        }
        qws[i].a16_ready = qmode[i] != K3_QUANT_SVE_Q8 ||
                           dtype == K3_FULL_DTYPE_Q8_0;
        qowner[i] = i;
    }

    /* IQ unpacking is done once per tensor when explicitly enabled.  The
     * resulting row16 tiles are reused by every decode token; keep this opt-in
     * because the representation expands the compressed weights by ~4x. */
    if (full_quant_packed) {
        for (int i = 0; i < count; ++i) {
            int pack_rc = full_quant_pack_tensor((k3_full_tensor *)tensors[i]);
            if (pack_rc)
                fprintf(stderr, "k3_full_runner: IQ packed allocation failed for tensor %d\n", i);
        }
    }

    full_bf16_task tasks[task_count];
    int n = 0;
    for (int i = 0; i < count; ++i) {
        int dtype = tensors[i]->dtype;
        const uint16_t *w = dtype == 1 ? (const uint16_t *)tensors[i]->data : NULL;
        const float *fw = dtype == 2 ? (const float *)tensors[i]->data : NULL;
        const unsigned char *cmap = dtype == 1 ? tensors[i]->cmg_map : NULL;
        size_t rowb = (size_t)cols[i] * 2;
        size_t qrb = (dtype >= K3_FULL_DTYPE_Q8_0 &&
                      dtype <= K3_FULL_DTYPE_IQ3_XXS) ?
            k3_quant_row_bytes(K3_Q_Q8_0 + (dtype - K3_FULL_DTYPE_Q8_0), cols[i]) : 0;
        int step = (dtype >= K3_FULL_DTYPE_IQ1_S &&
                    dtype <= K3_FULL_DTYPE_IQ2_XXS &&
                    (rows[i] & 15) == 0) ? 16 : 8;
        for (int r = 0; r < rows[i]; r += step) {
            int nr = rows[i] - r;
            if (nr > step) nr = step;
            tasks[n++] = (full_bf16_task){
                .out = (float *)outs[i] + r,
                .weights = w ? w + (size_t)r * cols[i] : NULL,
                .f32_weights = fw ? fw + (size_t)r * cols[i] : NULL,
                .packed = dtype == 1 ? NULL : tensors[i]->data +
                    (size_t)(r / 8) * (size_t)(cols[i] / 16) *
                    (dtype == K3_FULL_DTYPE_Q8P8 ? 192 : 160),
                .input = inputs[i],
                .quant = (dtype >= K3_FULL_DTYPE_Q8_0 &&
                          dtype <= K3_FULL_DTYPE_IQ3_XXS) ?
                    tensors[i]->data + (size_t)r * qrb : NULL,
                .quant_ws = qowner[i] >= 0 ? &qws[qowner[i]] : NULL,
                .quant_packed = tensors[i]->quant_packed.data && nr == 16 ?
                    &tensors[i]->quant_packed : NULL,
                .quant_packed_data = tensors[i]->quant_packed.data && nr == 16 ?
                    tensors[i]->quant_packed.data + (size_t)(r / 16) *
                    tensors[i]->quant_packed.tile_bytes : NULL,
                .quant_packed_scales_data = tensors[i]->quant_packed.scales && nr == 16 ?
                    tensors[i]->quant_packed.scales + (size_t)(r / 16) *
                    tensors[i]->quant_packed.scale_tile_bytes : NULL,
                .quant_packed_ds_data = tensors[i]->quant_packed.ds && nr == 16 ?
                    tensors[i]->quant_packed.ds + (size_t)(r / 16) *
                    tensors[i]->quant_packed.d_tile_bytes / sizeof(uint16_t) : NULL,
                .quant_row_bytes = qrb,
                .quant_mode = qmode[i],
                .rows = nr,
                .cols = cols[i],
                .dtype = dtype,
                .pv = tensors[i]->pv && nr == 8,
                .cmg = -1,
                .cmg_copy = tensors[i]->cmg_copy[0] ? tensors[i]->cmg_copy : NULL,
                .woff = (size_t)r * rowb,
            };
            if (cmap) {
                /* CMG of the page holding this task's first row; an 8-row task
                 * is 114 KB and a page 2 MB, so it rarely straddles. */
                tasks[n - 1].cmg = (int)cmap[((size_t)r * rowb) / K3_CMG_PAGE];
            }
        }
    }

    int workers = threads > 0 && threads < n ? threads : n;
    if (n == 1) {
        /* TP96 leaves several projections at one eight-row task.  Avoid
         * opening an OpenMP team for those latency-bound tails. */
#if defined(_OPENMP)
        omp_set_num_threads(threads > 0 ? threads : 1);
#endif
        full_bf16_run_task(&tasks[0]);
        full_quant_workspaces_free(qws, qowner, count);
        return;
    }
#if defined(_OPENMP)
    omp_set_num_threads(workers > 0 ? workers : 1);
    /* CMG-local dispatch: every thread runs only the tasks whose weight rows are
     * bound to its own CMG.  Replaces the schedule below rather than tuning it --
     * inter-CMG bandwidth (~119 GB/s) is the cap, not the work distribution.
     * Requires all tasks in the batch to be bound, and enough threads that at
     * least two CMGs are populated; otherwise it falls through unchanged. */
    int cmg_ok = full_cmg_local && workers >= 2 * K3_CMG_CORES;
    if (cmg_ok)
        for (int i = 0; i < n; ++i)
            if (tasks[i].cmg < 0) { cmg_ok = 0; break; }
    if (cmg_ok) {
        int order[n];
        int head[K3_CMG_COUNT + 1] = {0}, fill[K3_CMG_COUNT];
        for (int i = 0; i < n; ++i) ++head[tasks[i].cmg + 1];
        for (int c = 0; c < K3_CMG_COUNT; ++c) head[c + 1] += head[c];
        for (int c = 0; c < K3_CMG_COUNT; ++c) fill[c] = head[c];
        for (int i = 0; i < n; ++i) order[fill[tasks[i].cmg]++] = i;
#pragma omp parallel num_threads(workers)
        {
            int t = omp_get_thread_num();
            int c = t / K3_CMG_CORES;
            if (c >= K3_CMG_COUNT) c = K3_CMG_COUNT - 1;
            int nl = full_cmg_threads(c, workers);
            int lid = t - c * K3_CMG_CORES;
            if (nl > 0)
                for (int k = head[c] + lid; k < head[c + 1]; k += nl)
                    full_bf16_run_task_local(&tasks[order[k]]);
        }
        omp_set_num_threads(threads > 0 ? threads : 1);
        full_quant_workspaces_free(qws, qowner, count);
        return;
    }
    /* Pick the schedule from the batch's shape, because the two cases want
     * opposite answers and the difference is large in both directions.
     *
     * Heterogeneous batches (a whole layer's projections, columns spanning
     * 512..12288) must not use static: it hands some threads several times the
     * work of others.  861 eight-row tasks at 47 threads: static 1.170 ms,
     * static,1 0.830, dynamic 0.485, guided 0.453.
     *
     * Homogeneous batches want static.  The shared expert is two 512x7168
     * projections then one 7168x512 -- all equal-cost tasks -- and at 48
     * threads guided costs 0.390 ms against static's 0.052, a 7.5x loss,
     * because guided's chunk arithmetic and shared counter dominate when every
     * task is 8 KB.  An earlier version of this code used guided for both and
     * regressed the moe_shared phase accordingly.
     *
     * Each task writes its own output rows, so scheduling cannot change
     * results either way. */
    int homogeneous = 1;
    for (int i = 1; i < count; ++i)
        if (cols[i] != cols[0]) { homogeneous = 0; break; }
#if defined(_OPENMP)
    omp_set_schedule(homogeneous ? omp_sched_static : omp_sched_guided, 0);
#endif
#pragma omp parallel for schedule(runtime)
#endif
    for (int i = 0; i < n; ++i) {
        full_bf16_run_task_local(&tasks[i]);
    }
#if defined(_OPENMP)
    omp_set_num_threads(threads > 0 ? threads : 1);
#endif
    full_quant_workspaces_free(qws, qowner, count);
}

static void full_bf16_matvec(float *out, const k3_full_tensor *t,
                             int rows, int cols, const float *x, int threads) {
    float *outs[1] = {out};
    const k3_full_tensor *tensors[1] = {t};
    const float *inputs[1] = {x};
    full_bf16_many(outs, tensors, inputs, &rows, &cols, 1, threads);
}

/* Native shared expert in one OpenMP team.  The three stages remain separated
 * by implicit workshare barriers, but reuse the team and the same per-row PV
 * kernel as full_bf16_many. */
static int full_shared_forward_fused(float *hidden, float *gate, float *up,
                                     const k3_full_tensor *gate_w,
                                     const k3_full_tensor *up_w,
                                     const k3_full_tensor *down_w,
                                     const float *x, int threads) {
#if defined(_OPENMP) && defined(__ARM_FEATURE_SVE)
    int local = (int)gate_w->shape[0];
    if (!full_shared_fused_team || !full_situ_fast ||
        gate_w->dtype != 1 || up_w->dtype != 1 || down_w->dtype != 1 ||
        !gate_w->pv || !up_w->pv || !down_w->pv ||
        !gate_w->cmg_copy[0] || !up_w->cmg_copy[0] || !down_w->cmg_copy[0] ||
        (local & 7) || down_w->shape[0] != K3_HIDDEN ||
        down_w->shape[1] != (size_t)local)
        return 0;
    omp_set_num_threads(threads);
#pragma omp parallel num_threads(threads)
    {
        int cmg = omp_get_thread_num() / K3_CMG_CORES;
        if (cmg >= K3_CMG_COUNT) cmg = K3_CMG_COUNT - 1;
        const uint16_t *gw = (const uint16_t *)gate_w->cmg_copy[cmg];
        const uint16_t *uw = (const uint16_t *)up_w->cmg_copy[cmg];
        const uint16_t *dw = (const uint16_t *)down_w->cmg_copy[cmg];
        int groups = local / 8;
#pragma omp for schedule(static)
        for (int task = 0; task < 2 * groups; ++task) {
            int which = task >= groups;
            int row = (task - which * groups) * 8;
            const uint16_t *p = (which ? uw : gw) + (size_t)row * K3_HIDDEN;
            matvec_bf16_8row_pv((which ? up : gate) + row,
                                p, p + (size_t)2 * K3_HIDDEN,
                                p + (size_t)4 * K3_HIDDEN,
                                p + (size_t)6 * K3_HIDDEN,
                                x, K3_HIDDEN);
        }
        int vl = (int)svcntw();
#pragma omp for schedule(static)
        for (int i = 0; i < local; i += vl)
            k3_situ_fast_sve(gate + i, gate + i, up + i,
                             local - i < vl ? local - i : vl);
        int down_groups = K3_HIDDEN / 8;
#pragma omp for schedule(static)
        for (int group = 0; group < down_groups; ++group) {
            int row = group * 8;
            const uint16_t *p = dw + (size_t)row * local;
            matvec_bf16_8row_pv(hidden + row,
                                p, p + (size_t)2 * local,
                                p + (size_t)4 * local,
                                p + (size_t)6 * local,
                                gate, local);
        }
    }
    return 1;
#else
    (void)hidden; (void)gate; (void)up; (void)gate_w; (void)up_w;
    (void)down_w; (void)x; (void)threads;
    return 0;
#endif
}

static int full_kda_front_fused(k3_full_model *m, k3_full_layer *l,
                                const float *x, float *qstate,
                                float *kstate, float *vstate) {
#if defined(_OPENMP) && defined(__ARM_FEATURE_SVE)
    if (!full_kda_fused_team) return 0;
    k3_full_tensor *weight[] = {
        &l->q_proj, &l->k_proj, &l->v_proj,
        &l->f_a_proj, &l->g_proj, &l->b_proj};
    float *output[] = {m->q, m->k, m->v, m->tmp, m->expert_out, m->tmp2};
    int rows[] = {m->local_heads * K3_HEAD_DIM,
                  m->local_heads * K3_HEAD_DIM,
                  m->local_heads * K3_HEAD_DIM,
                  K3_HEAD_DIM, m->local_heads * K3_HEAD_DIM,
                  m->local_heads};
    int total_groups = 0;
    for (int i = 0; i < 6; ++i) {
        if (weight[i]->dtype != 1 || !weight[i]->pv ||
            !weight[i]->cmg_copy[0] || (rows[i] & 7) ||
            weight[i]->shape[1] != K3_HIDDEN)
            return 0;
        total_groups += rows[i] / 8;
    }
    int channels = m->local_heads * K3_HEAD_DIM;
    if (l->f_b_proj.dtype != 1 || !l->f_b_proj.pv ||
        !l->f_b_proj.cmg_copy[0] || l->f_b_proj.shape[0] != (size_t)channels ||
        l->f_b_proj.shape[1] != K3_HEAD_DIM)
        return 0;
    const float *qw = (const float *)l->q_conv.data;
    const float *kw = (const float *)l->k_conv.data;
    const float *vw = (const float *)l->v_conv.data;
    int threads = m->threads;
    omp_set_num_threads(threads);
#pragma omp parallel num_threads(threads)
    {
        int cmg = omp_get_thread_num() / K3_CMG_CORES;
        if (cmg >= K3_CMG_COUNT) cmg = K3_CMG_COUNT - 1;
#pragma omp for schedule(static)
        for (int task = 0; task < total_groups; ++task) {
            int which = 0, local_task = task;
            while (local_task >= rows[which] / 8)
                local_task -= rows[which++] / 8;
            int row = local_task * 8;
            const uint16_t *base =
                (const uint16_t *)weight[which]->cmg_copy[cmg];
            const uint16_t *p = base + (size_t)row * K3_HIDDEN;
            matvec_bf16_8row_pv(output[which] + row,
                                p, p + (size_t)2 * K3_HIDDEN,
                                p + (size_t)4 * K3_HIDDEN,
                                p + (size_t)6 * K3_HIDDEN,
                                x, K3_HIDDEN);
        }
#pragma omp for schedule(static)
        for (int c = 0; c < channels; ++c) {
            float *qs = qstate + (size_t)c * (K3_FULL_CONV_KERNEL - 1);
            float *ks = kstate + (size_t)c * (K3_FULL_CONV_KERNEL - 1);
            float *vs = vstate + (size_t)c * (K3_FULL_CONV_KERNEL - 1);
            float qx = m->q[c], kx = m->k[c], vx = m->v[c];
            float qy = 0.0f, ky = 0.0f, vy = 0.0f;
            for (int j = 0; j < K3_FULL_CONV_KERNEL - 1; ++j) {
                qy += qw[(size_t)c * K3_FULL_CONV_KERNEL + j] * qs[j];
                ky += kw[(size_t)c * K3_FULL_CONV_KERNEL + j] * ks[j];
                vy += vw[(size_t)c * K3_FULL_CONV_KERNEL + j] * vs[j];
            }
            qy += qw[(size_t)c * K3_FULL_CONV_KERNEL + K3_FULL_CONV_KERNEL - 1] * qx;
            ky += kw[(size_t)c * K3_FULL_CONV_KERNEL + K3_FULL_CONV_KERNEL - 1] * kx;
            vy += vw[(size_t)c * K3_FULL_CONV_KERNEL + K3_FULL_CONV_KERNEL - 1] * vx;
            for (int j = 0; j < K3_FULL_CONV_KERNEL - 2; ++j) {
                qs[j] = qs[j + 1]; ks[j] = ks[j + 1]; vs[j] = vs[j + 1];
            }
            qs[K3_FULL_CONV_KERNEL - 2] = qx;
            ks[K3_FULL_CONV_KERNEL - 2] = kx;
            vs[K3_FULL_CONV_KERNEL - 2] = vx;
            m->q[c] = qy; m->k[c] = ky; m->v[c] = vy;
        }
        int decay_groups = channels / 8;
#pragma omp for schedule(static)
        for (int group = 0; group < decay_groups; ++group) {
            int row = group * 8;
            const uint16_t *base =
                (const uint16_t *)l->f_b_proj.cmg_copy[cmg];
            const uint16_t *p = base + (size_t)row * K3_HEAD_DIM;
            matvec_bf16_8row_pv(m->gate + row,
                                p, p + (size_t)2 * K3_HEAD_DIM,
                                p + (size_t)4 * K3_HEAD_DIM,
                                p + (size_t)6 * K3_HEAD_DIM,
                                m->tmp, K3_HEAD_DIM);
        }
    }
    return 1;
#else
    (void)m; (void)l; (void)x; (void)qstate; (void)kstate; (void)vstate;
    return 0;
#endif
}

/* Finish an already-started latent collective on one member of the existing
 * OpenMP team while the remaining workers execute the independent shared-down
 * projection.  This keeps progress on an application worker and avoids a
 * per-token pthread or an extra core. */
static void full_f32_copy(float *dst, const k3_full_tensor *t, int n) {
    memcpy(dst, t->data, (size_t)n * sizeof(float));
}

static void full_add(float *dst, const float *src, int n) {
#if defined(__ARM_FEATURE_SVE)
    if (full_serial_vector_ops) {
        int vl = (int)svcntw();
        for (int i = 0; i < n; i += vl) {
            svbool_t pg = svwhilelt_b32(i, n);
            svst1(pg, dst + i, svadd_f32_x(pg, svld1(pg, dst + i),
                                           svld1(pg, src + i)));
        }
        return;
    }
#endif
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n; ++i) dst[i] += src[i];
}

static void full_add_copy(float *dst, float *accum, const float *src, int n) {
#if defined(__ARM_FEATURE_SVE)
    if (full_serial_vector_ops) {
        int vl = (int)svcntw();
        for (int i = 0; i < n; i += vl) {
            svbool_t pg = svwhilelt_b32(i, n);
            svfloat32_t y = svadd_f32_x(pg, svld1(pg, accum + i),
                                        svld1(pg, src + i));
            svst1(pg, accum + i, y);
            svst1(pg, dst + i, y);
        }
        return;
    }
#endif
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n; ++i)
        dst[i] = (accum[i] += src[i]);
}

static void full_copy(float *dst, const float *src, int n) {
    memcpy(dst, src, (size_t)n * sizeof(float));
}

static void full_iq_trace_finite(const k3_full_model *m, const char *label,
                                 const float *x, int n) {
    if (!full_iq_trace || m->rank != 0) return;
    int finite = 0;
    float lo = INFINITY, hi = -INFINITY;
    for (int i = 0; i < n; ++i) {
        if (!isfinite(x[i])) continue;
        ++finite;
        if (x[i] < lo) lo = x[i];
        if (x[i] > hi) hi = x[i];
    }
    printf("K3_IQ_FULL_TRACE %s finite=%d/%d x0=%+.9e range=[%+.9e,%+.9e]\n",
           label, finite, n, x[0], lo, hi);
}

static float full_weight_at(const k3_full_tensor *t, int index) {
    if (t->dtype == 1)
        return bf16_to_f32_scalar(((const uint16_t *)t->data)[index]);
    if (t->dtype == 2)
        return ((const float *)t->data)[index];
    return 0.0f;
}

static void full_rmsnorm_tensor(float *out, const float *x,
                                const k3_full_tensor *weight, int n) {
    float inv = 1.0f / sqrtf(k3_dot_sve(x, x, n) / n + K3_FULL_EPS);
#if defined(__ARM_FEATURE_SVE)
    int vl=(int)svcntw();
    if(weight->dtype==1){const uint16_t*w=(const uint16_t*)weight->data;
        for(int i=0;i<n;i+=vl){svbool_t pg=svwhilelt_b32(i,n);
            svuint32_t bits=svlsl_n_u32_x(pg,svld1uh_u32(pg,w+i),16);
            svfloat32_t y=svmul_n_f32_x(pg,svld1(pg,x+i),inv);
            svst1(pg,out+i,svmul_f32_x(pg,y,svreinterpret_f32_u32(bits)));}}
    else{const float*w=(const float*)weight->data;
        for(int i=0;i<n;i+=vl){svbool_t pg=svwhilelt_b32(i,n);
            svfloat32_t y=svmul_n_f32_x(pg,svld1(pg,x+i),inv);
            svst1(pg,out+i,svmul_f32_x(pg,y,svld1(pg,w+i)));}}
#else
    for(int i=0;i<n;++i)out[i]=x[i]*inv*full_weight_at(weight,i);
#endif
}

static void full_gated_rmsnorm_tensor(float *out, const float *x,
                                      const float *gate,
                                      const k3_full_tensor *weight, int n) {
    float inv = 1.0f / sqrtf(k3_dot_sve(x, x, n) / n + K3_FULL_EPS);
    int weight_n = weight->ndims == 1 ? (int)weight->shape[0] : n;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n; ++i) {
        int wi = weight_n > 0 && weight_n < n ? i % weight_n : i;
        out[i] = x[i] * inv * full_weight_at(weight, wi) * k3_sigmoidf(gate[i]);
    }
}

static void full_attn_res(float *out, const float *prefix,
                          const float *blocks, int block_count,
                          const k3_full_tensor *proj,
                          const k3_full_tensor *norm) {
    if (block_count == 0) {
        if (out != prefix)
            memcpy(out, prefix, K3_HIDDEN * sizeof(float));
        return;
    }
    int count = block_count + 1;
    float scores[K3_LAYERS / 12 + 2];
    float max_score = -INFINITY;
    for (int c = 0; c < count; ++c) {
        const float *v = c == block_count ? prefix : blocks + (size_t)c * K3_HIDDEN;
        double ss = 0.0;
        for (int j = 0; j < K3_HIDDEN; ++j) ss += (double)v[j] * v[j];
        float inv = 1.0f / sqrtf((float)(ss / K3_HIDDEN) + K3_FULL_EPS);
        double score = 0.0;
        for (int j = 0; j < K3_HIDDEN; ++j)
            score += (double)v[j] * inv * full_weight_at(norm, j) *
                     full_weight_at(proj, j);
        scores[c] = (float)score;
        if (scores[c] > max_score) max_score = scores[c];
    }
    float denom = 0.0f;
    for (int c = 0; c < count; ++c) denom += expf(scores[c] - max_score);
    float coefficient[K3_LAYERS / 12 + 2];
    for (int c = 0; c < count; ++c)
        coefficient[c] = expf(scores[c] - max_score) / denom;
    for (int j = 0; j < K3_HIDDEN; ++j) {
        float value = 0.0f;
        for (int c = 0; c < count; ++c) {
            const float *v = c == block_count ? prefix : blocks + (size_t)c * K3_HIDDEN;
            value += v[j] * coefficient[c];
        }
        out[j] = value;
    }
}

static int full_is_mla(int layer) {
    static const int mla[] = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47,
                              51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 92};
    for (size_t i = 0; i < sizeof mla / sizeof mla[0]; ++i)
        if (mla[i] == layer) return 1;
    return 0;
}

static int full_load_layer(k3_full_model *m, k3_full_layer *l,
                           const k3_full_entry *entries, int n,
                           int layer) {
    char prefix[128];
    snprintf(prefix, sizeof prefix, "language_model.model.layers.%d.", layer);
#define FT(field, suffix) do { \
        char name_[K3_FULL_MAX_NAME]; \
        snprintf(name_, sizeof name_, "%s%s", prefix, suffix); \
        (field) = full_tensor(entries, n, m->blob, name_); \
        if (!full_tensor_valid(&(field))) return EINVAL; \
    } while (0)
    FT(l->input_norm, "input_layernorm.weight");
    FT(l->post_norm, "post_attention_layernorm.weight");
    FT(l->attn_res_norm, "self_attention_res_norm.weight");
    FT(l->attn_res_proj, "self_attention_res_proj.weight");
    FT(l->mlp_res_norm, "mlp_res_norm.weight");
    FT(l->mlp_res_proj, "mlp_res_proj.weight");
    l->is_mla = full_is_mla(layer);
    l->state_slot = -1;
    l->cache_slot = -1;
    if (l->is_mla) {
        FT(l->q_a_proj, "self_attn.q_a_proj.weight");
        FT(l->q_a_norm, "self_attn.q_a_layernorm.weight");
        FT(l->q_b_proj, "self_attn.q_b_proj.weight");
        FT(l->kv_a_proj, "self_attn.kv_a_proj_with_mqa.weight");
        FT(l->kv_a_norm, "self_attn.kv_a_layernorm.weight");
        FT(l->kv_b_proj, "self_attn.kv_b_proj.weight");
        FT(l->mla_g_proj, "self_attn.g_proj.weight");
        FT(l->mla_o_proj, "self_attn.o_proj.weight");
        l->cache_slot = m->mla_count++;
    } else {
        FT(l->q_proj, "self_attn.q_proj.weight");
        FT(l->k_proj, "self_attn.k_proj.weight");
        FT(l->v_proj, "self_attn.v_proj.weight");
        FT(l->g_proj, "self_attn.g_proj.weight");
        FT(l->f_a_proj, "self_attn.f_a_proj.weight");
        FT(l->f_b_proj, "self_attn.f_b_proj.weight");
        FT(l->b_proj, "self_attn.b_proj.weight");
        FT(l->q_conv, "self_attn.q_conv1d.weight");
        FT(l->k_conv, "self_attn.k_conv1d.weight");
        FT(l->v_conv, "self_attn.v_conv1d.weight");
        FT(l->a_log, "self_attn.A_log");
        FT(l->dt_bias, "self_attn.dt_bias");
        FT(l->o_norm, "self_attn.o_norm.weight");
        FT(l->o_proj, "self_attn.o_proj.weight");
        l->state_slot = m->kda_count++;
    }
    if (layer == 0) {
        FT(l->dense_gate, "mlp.gate_proj.weight");
        FT(l->dense_up, "mlp.up_proj.weight");
        FT(l->dense_down, "mlp.down_proj.weight");
        l->expert_count = 0;
    } else {
        FT(l->router, "block_sparse_moe.gate.weight");
        FT(l->router_bias, "block_sparse_moe.gate.e_score_correction_bias");
        FT(l->routed_down, "block_sparse_moe.routed_expert_down_proj.weight");
        FT(l->routed_norm, "block_sparse_moe.routed_expert_norm.weight");
        FT(l->routed_up, "block_sparse_moe.routed_expert_up_proj.weight");
        FT(l->shared_gate, "block_sparse_moe.shared_experts.gate_proj.weight");
        FT(l->shared_up, "block_sparse_moe.shared_experts.up_proj.weight");
        FT(l->shared_down, "block_sparse_moe.shared_experts.down_proj.weight");
        int count = m->expert_tp ? K3_EXPERTS : 0;
        if (!m->expert_tp)
            for (int e = m->rank; e < K3_EXPERTS; e += m->nodes) ++count;
        l->experts = (k3_full_expert *)k3_pool_calloc(m->pool, (size_t)count,
                                                       sizeof *l->experts);
        if (!l->experts) return ENOMEM;
        l->expert_count = count;
        int slot = 0;
        for (int e = 0; e < K3_EXPERTS; ++e) {
            if (!m->expert_tp && e % m->nodes != m->rank) continue;
            char name[K3_FULL_MAX_NAME];
            l->experts[slot].expert_id = e;
#define EX(field, suffix) do { \
                snprintf(name, sizeof name, "%sblock_sparse_moe.experts.%d.%s", \
                         prefix, e, suffix); \
                if (!full_find(entries, n, name)) { \
                    const char *dot_ = strstr(suffix, ".weight_packed"); \
                    if (!dot_) return EINVAL; \
                    snprintf(name, sizeof name, \
                        "%sblock_sparse_moe.experts.%d.%.*s.weight_quant", \
                        prefix, e, (int)(dot_ - suffix), suffix); \
                } \
                (l->experts[slot].field) = full_tensor(entries, n, m->blob, name); \
                if (!full_tensor_valid(&(l->experts[slot].field))) return EINVAL; \
            } while (0)
            EX(w1, "w1.weight_packed");
            EX(w2, "w2.weight_packed");
            EX(w3, "w3.weight_packed");
            if (l->experts[slot].w1.dtype >= K3_FULL_DTYPE_IQ1_S &&
                l->experts[slot].w1.dtype <= K3_FULL_DTYPE_IQ3_XXS) {
                int local_inter = m->expert_tp ? K3_EXPERT_INTER / m->nodes :
                                                 K3_EXPERT_INTER;
                if (l->experts[slot].w2.dtype != l->experts[slot].w1.dtype ||
                    l->experts[slot].w3.dtype != l->experts[slot].w1.dtype ||
                    l->experts[slot].w1.shape[0] != (size_t)local_inter ||
                    l->experts[slot].w1.shape[1] != K3_LATENT ||
                    l->experts[slot].w2.shape[0] != K3_LATENT ||
                    l->experts[slot].w2.shape[1] != (size_t)local_inter ||
                    l->experts[slot].w3.shape[0] != (size_t)local_inter ||
                    l->experts[slot].w3.shape[1] != K3_LATENT) return EINVAL;
                ++slot;
                continue;
            }
            k3_full_tensor s1, s2, s3;
            snprintf(name, sizeof name, "%sblock_sparse_moe.experts.%d.w1.weight_scale", prefix, e);
            s1 = full_tensor(entries, n, m->blob, name);
            snprintf(name, sizeof name, "%sblock_sparse_moe.experts.%d.w2.weight_scale", prefix, e);
            s2 = full_tensor(entries, n, m->blob, name);
            snprintf(name, sizeof name, "%sblock_sparse_moe.experts.%d.w3.weight_scale", prefix, e);
            s3 = full_tensor(entries, n, m->blob, name);
            if (!full_tensor_valid(&s1) || !full_tensor_valid(&s2) || !full_tensor_valid(&s3)) return EINVAL;
            l->experts[slot].mw1 = (k3_mxfp4_matrix){
                .packed = l->experts[slot].w1.data, .scale = s1.data,
                .rows = (int)l->experts[slot].w1.shape[0],
                .cols = (int)l->experts[slot].w1.shape[1] * 2};
            l->experts[slot].mw2 = (k3_mxfp4_matrix){
                .packed = l->experts[slot].w2.data, .scale = s2.data,
                .rows = (int)l->experts[slot].w2.shape[0],
                .cols = (int)l->experts[slot].w2.shape[1] * 2};
            l->experts[slot].mw3 = (k3_mxfp4_matrix){
                .packed = l->experts[slot].w3.data, .scale = s3.data,
                .rows = (int)l->experts[slot].w3.shape[0],
                .cols = (int)l->experts[slot].w3.shape[1] * 2};
            ++slot;
#undef EX
        }
    }
#undef FT
    if (full_bf16_pv) {
        /* Explicit list of the BF16 tensors consumed as matvec weights.  Listed
         * rather than swept so a flat-read tensor can never be repacked; the
         * ndims==2 guard in full_bf16_pv_repack is the second line of defence,
         * and anything omitted keeps the row-major kernel and stays correct. */
        k3_full_tensor *pv[] = {
            &l->q_proj, &l->k_proj, &l->v_proj, &l->g_proj,
            &l->f_a_proj, &l->f_b_proj, &l->b_proj, &l->o_proj,
            &l->q_a_proj, &l->q_b_proj, &l->kv_a_proj, &l->kv_b_proj,
            &l->mla_g_proj, &l->mla_o_proj,
            &l->dense_gate, &l->dense_up, &l->dense_down,
            &l->router, &l->routed_down, &l->routed_up,
            &l->shared_gate, &l->shared_up, &l->shared_down,
        };
        for (size_t i = 0; i < sizeof pv / sizeof *pv; ++i) {
            if (!pv[i]->data) continue;
            int rc = full_bf16_pv_repack(pv[i]);
            if (rc) return rc;
        }
    }
    if (full_cmg_local) {
        k3_full_tensor *cg[] = {
            &l->q_proj, &l->k_proj, &l->v_proj, &l->g_proj,
            &l->f_a_proj, &l->f_b_proj, &l->b_proj, &l->o_proj,
            &l->q_a_proj, &l->q_b_proj, &l->kv_a_proj, &l->kv_b_proj,
            &l->mla_g_proj, &l->mla_o_proj,
            &l->dense_gate, &l->dense_up, &l->dense_down,
            &l->router, &l->routed_down, &l->routed_up,
            &l->shared_gate, &l->shared_up, &l->shared_down,
        };
        for (size_t i = 0; i < sizeof cg / sizeof *cg; ++i) {
            full_cmg_force_tensor(cg[i]);
            if (full_cmg_replicate_on) {
                /* Replicas are local to every CMG, so any thread may take any
                 * task: skip the map so full_bf16_many keeps its normal
                 * schedule instead of the imbalanced CMG-routed one. */
                (void)full_cmg_replicate(cg[i]);
                continue;
            }
            (void)full_cmg_map_tensor(cg[i]);
        }
    }
    return 0;
}

static int full_load_model(k3_full_model *m, const k3_full_options *o,
                           k3_pool *pool) {
    char blob_path[1024], manifest_path[1024];
    int nb = snprintf(blob_path, sizeof blob_path, "%s/rank%03d.blob",
                      o->stage_dir, m->rank);
    int nm = snprintf(manifest_path, sizeof manifest_path, "%s/rank%03d.manifest",
                      o->stage_dir, m->rank);
    if (nb < 0 || nm < 0 || (size_t)nb >= sizeof blob_path || (size_t)nm >= sizeof manifest_path)
        return ENAMETOOLONG;
    k3_full_entry *entries = (k3_full_entry *)calloc(K3_FULL_MAX_ENTRIES, sizeof *entries);
    if (!entries) return ENOMEM;
    int n = 0;
    k3_full_manifest meta;
    int rc = full_load_manifest(manifest_path, entries, K3_FULL_MAX_ENTRIES, &n, &meta);
    if (rc) {
        fprintf(stderr, "k3_full_runner rank %d: manifest load rc=%d\n", m->rank, rc);
        free(entries);
        return rc;
    }
    if (meta.version >= 2 && (meta.nodes != m->nodes || meta.rank != m->rank)) {
        fprintf(stderr, "k3_full_runner rank %d: manifest ownership mismatch "
                "mode=%s rank=%d/%d nodes=%d/%d\n", m->rank, meta.mode,
                meta.rank, m->rank, meta.nodes, m->nodes);
        free(entries);
        return EINVAL;
    }
    m->expert_tp = strstr(meta.mode, "expert-tp") != NULL ||
                   strstr(meta.mode, "tp96") != NULL;
    m->q8_mode = strstr(meta.mode, "q8") != NULL;
    size_t blob_bytes = 0;
    void *blob = k3_pool_load_blob(pool, blob_path, &blob_bytes);
    if (!blob) {
        fprintf(stderr, "k3_full_runner rank %d: %s\n", m->rank, k3_pool_error(pool));
        free(entries);
        return ENOMEM;
    }
    rc = full_check_ranges(entries, n, meta.blob_bytes, blob_bytes);
    if (rc) {
        fprintf(stderr, "k3_full_runner rank %d: invalid blob/manifest layout\n", m->rank);
        free(entries);
        return rc;
    }
    qsort(entries, (size_t)n, sizeof *entries, full_entry_name_cmp);
    m->blob = (const uint8_t *)blob;
    m->blob_bytes = blob_bytes;
    m->pool = pool;
    m->local_heads = 0;
    full_split(K3_HEADS, m->rank, m->nodes, &m->first_head, &m->local_heads);
    full_split(K3_LATENT, m->rank, m->nodes, &m->latent_start, &m->latent_rows);
    full_split(K3_FULL_VOCAB, m->rank, m->nodes, &m->embed_start, &m->embed_rows);
    m->head_start = m->embed_start;
    m->head_rows = m->embed_rows;
    m->embed = full_tensor(entries, n, m->blob, "language_model.model.embed_tokens.weight");
    m->lm_head = full_tensor(entries, n, m->blob, "language_model.lm_head.weight");
    m->final_norm = full_tensor(entries, n, m->blob, "language_model.model.norm.weight");
    if (!full_tensor_valid(&m->embed) || !full_tensor_valid(&m->lm_head) ||
        !full_tensor_valid(&m->final_norm)) {
        free(entries);
        return EINVAL;
    }
    for (int layer = 0; layer < K3_LAYERS; ++layer) {
        rc = full_load_layer(m, &m->layers[layer], entries, n, layer);
        if (rc) {
            fprintf(stderr, "k3_full_runner rank %d: layer %d load rc=%d\n", m->rank, layer, rc);
            free(entries);
            return rc;
        }
    }
    free(entries);
    return 0;
}

static int full_load_debug_model(k3_full_model *m,
                                 const k3_full_options *o,
                                 k3_pool *pool) {
    char blob_path[1024], manifest_path[1024];
    int nb = snprintf(blob_path, sizeof blob_path, "%s/rank%03d.blob",
                      o->stage_dir, m->rank);
    int nm = snprintf(manifest_path, sizeof manifest_path, "%s/rank%03d.manifest",
                      o->stage_dir, m->rank);
    if (nb < 0 || nm < 0 || (size_t)nb >= sizeof blob_path ||
        (size_t)nm >= sizeof manifest_path)
        return ENAMETOOLONG;
    k3_full_entry *entries = (k3_full_entry *)calloc(K3_FULL_MAX_ENTRIES,
                                                       sizeof *entries);
    if (!entries) return ENOMEM;
    int n = 0;
    k3_full_manifest meta;
    int rc = full_load_manifest(manifest_path, entries, K3_FULL_MAX_ENTRIES,
                                &n, &meta);
    if (rc) {
        fprintf(stderr, "k3_full_runner rank %d: debug manifest load rc=%d\n",
                m->rank, rc);
        free(entries);
        return rc;
    }
    int expert_tp = strstr(meta.mode, "expert-tp") != NULL ||
                    strstr(meta.mode, "tp96") != NULL;
    const char *want_mode = "layer12";
    if (o->mode == K3_FULL_MODE_SYNTHETIC12) want_mode = "layer12";
    if (meta.version < 2 ||
        (strcmp(meta.mode, want_mode) && strcmp(meta.mode, "layer12-iq") &&
         !(expert_tp && !strcmp(want_mode, "layer12"))) ||
        meta.nodes != m->nodes || meta.rank != m->rank ||
        meta.layer_index != o->real_layer_index) {
        fprintf(stderr, "k3_full_runner rank %d: debug manifest mismatch "
                "mode=%s layer=%d rank=%d nodes=%d\n", m->rank, meta.mode,
                meta.layer_index, meta.rank, meta.nodes);
        free(entries);
        return EINVAL;
    }
    m->expert_tp = expert_tp;
    m->q8_mode = strstr(meta.mode, "q8") != NULL;
    size_t blob_bytes = 0;
    void *blob = k3_pool_load_blob(pool, blob_path, &blob_bytes);
    if (!blob) {
        fprintf(stderr, "k3_full_runner rank %d: %s\n", m->rank,
                k3_pool_error(pool));
        free(entries);
        return ENOMEM;
    }
    rc = full_check_ranges(entries, n, meta.blob_bytes, blob_bytes);
    if (rc) {
        fprintf(stderr, "k3_full_runner rank %d: invalid debug blob/manifest\n",
                m->rank);
        free(entries);
        return rc;
    }
    qsort(entries, (size_t)n, sizeof *entries, full_entry_name_cmp);
    m->blob = (const uint8_t *)blob;
    m->blob_bytes = blob_bytes;
    m->pool = pool;
    full_split(K3_HEADS, m->rank, m->nodes, &m->first_head, &m->local_heads);
    full_split(K3_LATENT, m->rank, m->nodes, &m->latent_start, &m->latent_rows);
    m->debug_layer_index = o->real_layer_index;
    rc = full_load_layer(m, &m->debug_layer, entries, n,
                         o->real_layer_index);
    if (!rc && !strncmp(meta.mode, "layer12-iq", strlen("layer12-iq")))
        rc = full_split_groups(K3_LATENT, m->rank, m->nodes, 32,
                               &m->latent_start, &m->latent_rows);
    free(entries);
    return rc;
}

/* Bandwidth form of the row allreduce.  After the usual non-power-of-two
 * fold, recursively halve the live range while reducing, then gather the
 * completed ranges in reverse.  For the four surviving ranks of the 6-rank
 * row communicator this moves 1.5 vectors instead of 2 vectors. */
static void full_tp_sum_rabenseifner(tp_comm *c, float *buf, int count) {
    if (c->nprocs == 1) return;
    /* Keep the stock path for shapes that cannot be split exactly or do not
     * fit distinct reduce/gather receive slots. */
    if (c->deterministic || c->a2a || c->pof2 < 2 ||
        count % c->pof2 || 2 * c->nrounds + 1 >= TP_AR_NSTEP) {
        tp_allreduce_sum(c, buf, count);
        return;
    }
    uint64_t tok = ++c->seq;
    int mr = c->my_rank, rem = c->rem;
    if (mr < 2 * rem) {
        if ((mr & 1) == 0) {
            tp_ar_send(c, mr + 1, 0, buf, count, tok);
            tp_ar_confirm(c);
        } else {
            tp_ar_recv_add(c, 0, mr - 1, buf, count, tok);
        }
    }
    int lo = 0, hi = count;
    if (c->newrank != -1) {
        int mask = c->pof2 >> 1;
        for (int k = 0; k < c->nrounds; ++k, mask >>= 1) {
            int mid = lo + (hi - lo) / 2;
            int keep_lo, keep_hi, send_lo, send_hi;
            if (c->newrank & mask) {
                keep_lo = mid; keep_hi = hi; send_lo = lo; send_hi = mid;
            } else {
                keep_lo = lo; keep_hi = mid; send_lo = mid; send_hi = hi;
            }
            int pnr = c->newrank ^ mask;
            int pr = pnr < rem ? pnr * 2 + 1 : pnr + rem;
            tp_ar_send(c, pr, k + 1, buf + send_lo, send_hi - send_lo, tok);
            tp_ar_recv_add(c, k + 1, pr, buf + keep_lo,
                           keep_hi - keep_lo, tok);
            tp_ar_confirm(c);
            lo = keep_lo; hi = keep_hi;
        }
        mask = 1;
        for (int k = 0; k < c->nrounds; ++k, mask <<= 1) {
            int span = hi - lo;
            int recv_lo = (c->newrank & mask) ? lo - span : hi;
            int pnr = c->newrank ^ mask;
            int pr = pnr < rem ? pnr * 2 + 1 : pnr + rem;
            int sid = c->nrounds + k + 1;
            tp_ar_send(c, pr, sid, buf + lo, span, tok);
            tp_ar_recv_copy(c, sid, pr, buf + recv_lo, span, tok);
            tp_ar_confirm(c);
            if (recv_lo < lo) lo = recv_lo;
            else hi += span;
        }
    }
    int bcast_sid = 2 * c->nrounds + 1;
    if (mr < 2 * rem) {
        if ((mr & 1) == 0)
            tp_ar_recv_copy(c, bcast_sid, mr + 1, buf, count, tok);
        else {
            tp_ar_send(c, mr - 1, bcast_sid, buf, count, tok);
            tp_ar_confirm(c);
        }
    }
    if (c->defer_tcq) tp_ar_complete_sends(c);
    if (c->defer_mrq) tp_ar_drain_mrq(c);
}

static int full_tp_sum_rabenseifner_checked(tp_comm *c, float *buf, int count) {
    return tp_allreduce_checked(c, buf, count, full_tp_sum_rabenseifner);
}

static int full_sum_mode(k3_full_model *m, float *data, int count,
                         int sharded_col, int rab_kind) {
    double start = m->profile.enabled ? full_now() : 0.0;
    int half_for_count = count == K3_LATENT ? g_half_col :
                         count == K3_HIDDEN ? g_half_hidden : 0;
    int use_half_col = half_for_count && sharded_col && m->comm_col;
    int use_sparse_row = use_half_col && g_sparse_row;
    int use_rabenseifner = (full_comm_rabenseifner & rab_kind) &&
                           count >= K3_HIDDEN &&
                           !sharded_col && m->comm_col;
    int rc;
    if (use_rabenseifner) {
        rc = full_tp_sum_rabenseifner_checked(m->comm, data, count);
        if (!rc) rc = tp_allreduce_sum_checked(m->comm_col, data, count);
    } else rc = m->comm_col ? (use_sparse_row ?
                            tp_allreduce_sum_2d_sharded_checked(m->comm, m->comm_col,
                                                                 data, count, g_rank, g_nodes) :
                            use_half_col ?
                            tp_allreduce_sum_2d_halves_checked(m->comm, m->comm_col,
                                                               data, count) :
                            tp_allreduce_sum_2d_checked(m->comm, m->comm_col,
                                                        data, count)) :
                           tp_allreduce_sum_checked(m->comm, data, count);
    if (m->profile.enabled) {
        double elapsed = full_now() - start;
        full_profile_collective_add(m, elapsed);
        full_profile_phase_add(m, K3_FULL_PHASE_REDUCE, elapsed);
    }
    return rc;
}

static int full_sum(k3_full_model *m, float *data, int count) {
    return full_sum_mode(m, data, count, 0, 0);
}

static int full_sum_attention(k3_full_model *m, float *data, int count) {
    return full_sum_mode(m, data, count, 0, 1);
}

static int full_sum_final(k3_full_model *m, float *data, int count,
                          int is_mla) {
    return full_sum_mode(m, data, count, 0, is_mla ? 4 : 2);
}

static int full_sum_sharded(k3_full_model *m, float *data, int count) {
    return full_sum_mode(m, data, count, 1, 0);
}

static int full_max(k3_full_model *m, float *data, int count) {
    return m->comm_col ? tp_allreduce_max_2d_checked(m->comm, m->comm_col,
                                                      data, count) :
                         tp_allreduce_max_checked(m->comm, data, count);
}

static void full_kda_forward(k3_full_model *m, k3_full_layer *l,
                             const float *x, float *out) {
    int channels = m->local_heads * K3_HEAD_DIM;
    int state_stride = channels * (K3_FULL_CONV_KERNEL - 1);
    float *state = m->conv_state + (size_t)l->state_slot * (size_t)state_stride * 3;
    float *qstate = state;
    float *kstate = qstate + state_stride;
    float *vstate = kstate + state_stride;
    /* b_proj reads only x, so it belongs in the first batch rather than with
     * f_b.  That leaves both batches homogeneous in columns, which matters more
     * than the ordering: a batch mixing f_b's 128 columns with b_proj's 7168
     * takes the guided schedule, and f_b is 128 eight-row tasks of 128 columns
     * each -- exactly the tiny-homogeneous-task case where guided measured 7.5x
     * slower than static. */
    float *qkv_outs[] = {m->q, m->k, m->v, m->tmp, m->expert_out, m->tmp2};
    const k3_full_tensor *qkv_weights[] = {
        &l->q_proj, &l->k_proj, &l->v_proj, &l->f_a_proj, &l->g_proj, &l->b_proj,
    };
    const float *qkv_inputs[] = {x, x, x, x, x, x};
    int qkv_rows[] = {channels, channels, channels, K3_HEAD_DIM, channels,
                      m->local_heads};
    int qkv_cols[] = {K3_HIDDEN, K3_HIDDEN, K3_HIDDEN, K3_HIDDEN, K3_HIDDEN,
                      K3_HIDDEN};
    double kt = m->profile.enabled ? full_now() : 0.0;
#define KDA_MARK(ph) do { if (m->profile.enabled) { \
        full_profile_phase_add(m, (ph), full_now() - kt); kt = full_now(); } } while (0)
    full_bf16_many(qkv_outs, qkv_weights, qkv_inputs,
                   qkv_rows, qkv_cols, 6, m->threads);
    full_iq_trace_finite(m, "kda_q_proj", m->q, channels);
    full_iq_trace_finite(m, "kda_k_proj", m->k, channels);
    full_iq_trace_finite(m, "kda_v_proj", m->v, channels);
    full_iq_trace_finite(m, "kda_f_a_proj", m->tmp, K3_HEAD_DIM);
    full_iq_trace_finite(m, "kda_g_proj", m->expert_out, channels);
    full_iq_trace_finite(m, "kda_beta_proj", m->tmp2, m->local_heads);
    KDA_MARK(K3_FULL_PHASE_KDA_QKV);
    memcpy(m->up, m->q, (size_t)channels * sizeof(float));
    k3_conv_step_sve(m->q, m->up, qstate, (const float *)l->q_conv.data,
                     NULL, channels, K3_FULL_CONV_KERNEL);
    memcpy(m->up, m->k, (size_t)channels * sizeof(float));
    k3_conv_step_sve(m->k, m->up, kstate, (const float *)l->k_conv.data,
                     NULL, channels, K3_FULL_CONV_KERNEL);
    memcpy(m->up, m->v, (size_t)channels * sizeof(float));
    k3_conv_step_sve(m->v, m->up, vstate, (const float *)l->v_conv.data,
                     NULL, channels, K3_FULL_CONV_KERNEL);
    full_iq_trace_finite(m, "kda_q_conv", m->q, channels);
    KDA_MARK(K3_FULL_PHASE_KDA_CONV);
    float *decay_outs[] = {m->gate};
    const k3_full_tensor *decay_weights[] = {&l->f_b_proj};
    const float *decay_inputs[] = {m->tmp};
    int decay_rows[] = {channels};
    int decay_cols[] = {K3_HEAD_DIM};
    full_bf16_many(decay_outs, decay_weights, decay_inputs,
                   decay_rows, decay_cols, 1, m->threads);
    full_iq_trace_finite(m, "kda_decay_proj", m->gate, channels);
    KDA_MARK(K3_FULL_PHASE_KDA_DECAY_PROJ);
    for (int h = 0; h < m->local_heads; ++h) {
        k3_l2_normalize_sve(m->q + (size_t)h * K3_HEAD_DIM, K3_HEAD_DIM, 1.0e-6f);
        k3_l2_normalize_sve(m->k + (size_t)h * K3_HEAD_DIM, K3_HEAD_DIM, 1.0e-6f);
        m->tmp2[h] = k3_sigmoidf(m->tmp2[h]);
    }
    const float *a_log = (const float *)l->a_log.data;
    if (l->a_log.ndims == 1 && l->a_log.shape[0] == K3_HEADS) {
        full_kda_log_decay_heads(m->decay, m->gate, a_log + m->first_head,
                                 (const float *)l->dt_bias.data,
                                 m->local_heads, K3_HEAD_DIM);
    } else if (full_fast_exp) {
        full_kda_log_decay_fast(m->decay, m->gate, a_log,
                                (const float *)l->dt_bias.data,
                                m->local_heads, K3_HEAD_DIM);
    } else {
        k3_kda_log_decay(m->decay, m->gate, a_log,
                         (const float *)l->dt_bias.data,
                         m->local_heads, K3_HEAD_DIM);
    }
    float *recurrent = m->kda_state +
        (size_t)l->state_slot * m->local_heads * K3_HEAD_DIM * K3_HEAD_DIM;
    float decay[(size_t)m->local_heads * K3_HEAD_DIM];
    if (full_fast_exp)
        full_exp_vec(decay, m->decay, m->local_heads * K3_HEAD_DIM);
    else
        for (int i = 0; i < m->local_heads * K3_HEAD_DIM; ++i)
            decay[i] = expf(m->decay[i]);
    KDA_MARK(K3_FULL_PHASE_KDA_SERIAL);
    k3_kda_step_decay_parallel_sve(m->attn, m->q, m->k, m->v, decay,
                    m->tmp2, recurrent, m->local_heads, K3_HEAD_DIM,
                    K3_HEAD_DIM, m->threads);
    full_iq_trace_finite(m, "kda_step", m->attn, channels);
    KDA_MARK(K3_FULL_PHASE_KDA_STEP);
    full_gated_rmsnorm_tensor(m->tmp, m->attn, m->expert_out,
                              &l->o_norm, channels);
    full_iq_trace_finite(m, "kda_gated_norm", m->tmp, channels);
    KDA_MARK(K3_FULL_PHASE_KDA_GRMSNORM);
    full_bf16_matvec(out, &l->o_proj, K3_HIDDEN, channels, m->tmp, m->threads);
    full_iq_trace_finite(m, "kda_o_proj", out, K3_HIDDEN);
    KDA_MARK(K3_FULL_PHASE_KDA_OPROJ);
#undef KDA_MARK
}

static void full_mla_forward(k3_full_model *m, k3_full_layer *l,
                             const float *x, int position, float *out) {
    double mt = m->profile.enabled ? full_now() : 0.0;
#define MLA_MARK(ph) do { if (m->profile.enabled) { \
        full_profile_phase_add(m, (ph), full_now() - mt); mt = full_now(); } } while (0)
    int channels = m->local_heads * K3_FULL_MLA_QK;
    float *latent_outs[] = {m->tmp, m->tmp2};
    const k3_full_tensor *latent_weights[] = {&l->q_a_proj, &l->kv_a_proj};
    const float *latent_inputs[] = {x, x};
    int latent_rows[] = {1536, 576};
    int latent_cols[] = {K3_HIDDEN, K3_HIDDEN};
    full_bf16_many(latent_outs, latent_weights, latent_inputs,
                   latent_rows, latent_cols, 2, m->threads);
    MLA_MARK(K3_FULL_PHASE_KDA_QKV);
    full_rmsnorm_tensor(m->tmp, m->tmp, &l->q_a_norm, 1536);
    full_rmsnorm_tensor(m->tmp2, m->tmp2, &l->kv_a_norm, 512);
    MLA_MARK(K3_FULL_PHASE_KDA_CONV);
    full_trace_mix(0, m->tmp, 1536);
    full_trace_mix(1, m->tmp2, 576);
    float *attn_proj_outs[] = {m->q, m->k, m->gate};
    const k3_full_tensor *attn_proj_weights[] = {
        &l->q_b_proj, &l->kv_b_proj, &l->mla_g_proj,
    };
    const float *attn_proj_inputs[] = {m->tmp, m->tmp2, x};
    int attn_proj_rows[] = {channels, m->local_heads * 256,
                            channels / 192 * 128};
    int attn_proj_cols[] = {1536, 512, K3_HIDDEN};
    if (full_mla_split_proj) {
        for (int i = 0; i < 3; ++i)
            full_bf16_matvec(attn_proj_outs[i], attn_proj_weights[i],
                             attn_proj_rows[i], attn_proj_cols[i],
                             attn_proj_inputs[i], m->threads);
    } else {
        full_bf16_many(attn_proj_outs, attn_proj_weights, attn_proj_inputs,
                       attn_proj_rows, attn_proj_cols, 3, m->threads);
    }
    MLA_MARK(K3_FULL_PHASE_KDA_DECAY_PROJ);
    full_trace_mix(2, m->q, channels);
    full_trace_mix(3, m->k, m->local_heads * 256);
    /* kv_b consumes only the 512-dimensional compressed part. */
    int key_stride = m->max_seq * K3_FULL_MLA_QK;
    int value_stride = m->max_seq * K3_FULL_MLA_VALUE;
    float *layer_keys = m->mla_keys + (size_t)l->cache_slot * m->local_heads * key_stride;
    float *layer_values = m->mla_values + (size_t)l->cache_slot * m->local_heads * value_stride;
    for (int h = 0; h < m->local_heads; ++h) {
        float *kh = layer_keys + (size_t)h * key_stride + (size_t)position * K3_FULL_MLA_QK;
        float *vh = layer_values + (size_t)h * value_stride + (size_t)position * K3_FULL_MLA_VALUE;
        memcpy(kh, m->k + (size_t)h * 256 + 0, 128 * sizeof(float));
        memcpy(kh + 128, m->tmp2 + 512, 64 * sizeof(float));
        memcpy(vh, m->k + (size_t)h * 256 + 128, 128 * sizeof(float));
    }
    /* The scan over the KV cache was a serial loop over the local heads, which
     * is what made MLA attention cost roughly twice KDA's for the same
     * projection volume.  k3_attention_heads_parallel_sve splits each head's
     * token range across the team and merges with log-sum-exp; k3_ep_runner
     * already uses it (k3_ep_runner.c:715). */
    if (full_mla_serial_attn) {
        for (int h = 0; h < m->local_heads; ++h)
            k3_attention_sve(m->attn + (size_t)h * K3_FULL_MLA_VALUE,
                             m->q + (size_t)h * K3_FULL_MLA_QK,
                             layer_keys + (size_t)h * key_stride,
                             layer_values + (size_t)h * value_stride,
                             position + 1, K3_FULL_MLA_QK, K3_FULL_MLA_VALUE);
    } else {
        k3_attention_heads_parallel_sve(m->attn, m->q, layer_keys, layer_values,
                                        m->local_heads, position + 1, m->max_seq,
                                        K3_FULL_MLA_QK, K3_FULL_MLA_VALUE,
                                        m->threads, m->mla_scratch, m->mla_stats);
    }
    MLA_MARK(K3_FULL_PHASE_KDA_STEP);
    full_trace_mix(4, m->attn, channels / 192 * 128);
    /* The caller passes out == m->attn (full_forward_token), so feeding o_proj
     * from m->attn made this matvec read its own output buffer: 47 threads write
     * out[0..7167] while others still read attn[0..1023].  A real data race, and
     * the reason MLA layers were not reproducible run to run while KDA -- which
     * feeds o_proj from m->tmp -- always was.  Stage the gated attention in
     * m->tmp (dead here: it held q_a, already consumed by the attn_proj batch)
     * exactly as full_kda_forward does. */
    int gate_channels = channels / 192 * 128;
#if defined(__ARM_FEATURE_SVE)
    if (full_mla_fast_gate) {
        int vl = (int)svcntw();
        for (int i = 0; i < gate_channels; i += vl) {
            svbool_t pg = svwhilelt_b32(i, gate_channels);
            svfloat32_t a = svld1(pg, m->attn + i);
            svfloat32_t g = k3_sigmoid_fast_sve(pg, svld1(pg, m->gate + i));
            svst1(pg, m->tmp + i, svmul_f32_x(pg, a, g));
        }
    } else
#endif
    for (int i = 0; i < gate_channels; ++i)
        m->tmp[i] = m->attn[i] * k3_sigmoidf(m->gate[i]);
    MLA_MARK(K3_FULL_PHASE_KDA_GRMSNORM);
    full_bf16_matvec(out, &l->mla_o_proj, K3_HIDDEN,
                     m->local_heads * K3_HEAD_DIM, m->tmp, m->threads);
    MLA_MARK(K3_FULL_PHASE_KDA_OPROJ);
    full_trace_mix(5, out, K3_HIDDEN);
#undef MLA_MARK
}

static void full_dense_forward(k3_full_model *m, k3_full_layer *l,
                               const float *x, float *out) {
    int local_inter = (int)l->dense_gate.shape[0];
    float *gate_up_outs[] = {m->expert_gate, m->expert_up};
    const k3_full_tensor *gate_up_weights[] = {&l->dense_gate, &l->dense_up};
    const float *gate_up_inputs[] = {x, x};
    int gate_up_rows[] = {local_inter, local_inter};
    int gate_up_cols[] = {K3_HIDDEN, K3_HIDDEN};
    full_bf16_many(gate_up_outs, gate_up_weights, gate_up_inputs,
                   gate_up_rows, gate_up_cols, 2, m->threads);
    full_situ(m->expert_gate, m->expert_gate, m->expert_up, local_inter);
    full_bf16_matvec(out, &l->dense_down, K3_HIDDEN, local_inter,
                     m->expert_gate, m->threads);
}

/* Full-model expert-TP path.  Every rank owns the same selected expert IDs
 * but only one intermediate-channel slice, so W1/W3/W2 produce a partial
 * routed latent locally.  Shared-expert hidden partials are packed with that
 * latent before the single MoE reduction. */
static int full_moe_forward_expert_tp(k3_full_model *m, k3_full_layer *l,
                                      const float *x, float *out) {
    double subphase_start = m->profile.enabled ? full_now() : 0.0;
    int local_latent = (int)l->routed_down.shape[0];
    int local_shared = (int)l->shared_gate.shape[0];
    int latent_start = 0;
    if (local_latent != K3_LATENT) {
        int shard_group = l->routed_down.dtype == 1 ? 8 : 32;
        if (l->routed_down.shape[1] != K3_HIDDEN ||
            full_split_groups(K3_LATENT, m->rank, m->nodes, shard_group,
                              &latent_start, &local_latent) ||
            (int)l->routed_down.shape[0] != local_latent) {
            fprintf(stderr, "k3_full_runner rank %d: invalid routed-down shard shape\n",
                    m->rank);
            return EINVAL;
        }
    }
    float router_logits[K3_EXPERTS];
    int route[K3_TOP_K];
    float route_weight[K3_TOP_K];
    int iq_experts = l->experts[0].w1.dtype >= K3_FULL_DTYPE_IQ1_S &&
                     l->experts[0].w1.dtype <= K3_FULL_DTYPE_IQ3_XXS;
    memset(m->local_latent, 0, K3_LATENT * sizeof(float));
    float *route_outs[] = {router_logits, m->local_latent + latent_start,
                           m->expert_gate, m->expert_up};
    const k3_full_tensor *route_weights[] = {
        &l->router, &l->routed_down, &l->shared_gate, &l->shared_up};
    const float *route_inputs[] = {x, x, x, x};
    int route_rows[] = {K3_EXPERTS, local_latent, local_shared, local_shared};
    int route_cols[] = {K3_HIDDEN, K3_HIDDEN, K3_HIDDEN, K3_HIDDEN};

    double dispatch_part = m->profile.enabled ? full_now() : 0.0;
    full_bf16_many(route_outs, route_weights, route_inputs,
                   route_rows, route_cols, iq_experts ? 4 : 2, m->threads);
    for (int e = 0; e < K3_EXPERTS; ++e) {
        if (!isfinite(router_logits[e])) {
            if (m->rank == 0)
                fprintf(stderr, "k3_full_runner: non-finite router logit e=%d value=%g\n",
                        e, router_logits[e]);
            return EDOM;
        }
    }
    if (m->profile.enabled) {
        full_profile_phase_add(m, K3_FULL_PHASE_DISPATCH_PROJ,
                               full_now() - dispatch_part);
        dispatch_part = full_now();
    }
    int latent_async = 0;
    double latent_async_start = 0.0;
    tp_comm *latent_comm = m->comm;
    tp_comm *latent_col = m->comm_col;
    /* Router selection only reads router_logits, so launch the independent
     * latent exchange before its scalar top-k work. */
    if (local_latent != K3_LATENT) {
        if (g_async_latent && latent_comm && latent_comm->pipeline &&
            !latent_comm->deterministic) {
            latent_async_start = full_now();
            if (g_sparse_row &&
                tp_allreduce_sum_2d_sharded_start(latent_comm, latent_col,
                                                  m->local_latent, K3_LATENT,
                                                  g_rank, g_nodes)) {
                latent_async = 2;
            } else {
                tp_allreduce_sum_2d_start(latent_comm, latent_col,
                                          m->local_latent, K3_LATENT);
                latent_async = 1;
            }
        } else {
            if (full_sum_sharded(m, m->local_latent, K3_LATENT)) return EIO;
            if (m->profile.enabled)
                full_profile_phase_add(m, K3_FULL_PHASE_LATENT_REDUCE,
                                       full_now() - dispatch_part);
        }
    }
    k3_router_topk(router_logits, (const float *)l->router_bias.data,
                   K3_EXPERTS, K3_TOP_K, route, route_weight);
    if (full_iq_trace && m->rank == 0) {
        printf("K3_IQ_FULL_TRACE expert_tp_route=");
        for (int k = 0; k < K3_TOP_K; ++k)
            printf("%s%d", k ? "," : "", route[k]);
        fputc('\n', stdout);
    }
    if (m->profile.enabled) {
        full_profile_phase_add(m, K3_FULL_PHASE_MOE_DISPATCH,
                               full_now() - subphase_start);
        subphase_start = full_now();
    }

    /* Shared expert consumes x only.  When the latent collective is started
     * above, run this independent work while its first row exchange is in
     * flight, then finish the collective before routed experts consume it. */
    double st = m->profile.enabled ? full_now() : 0.0;
#define SH_MARK(ph) do { if (m->profile.enabled) { \
        full_profile_phase_add(m, (ph), full_now() - st); st = full_now(); } } while (0)
    int shared_fused = 0;
    if (!iq_experts) {
        shared_fused = full_shared_forward_fused(
            m->shared_hidden, m->expert_gate, m->expert_up,
            &l->shared_gate, &l->shared_up, &l->shared_down, x, m->threads);
        if (!shared_fused) {
            float *shared_outs[] = {m->expert_gate, m->expert_up};
            const k3_full_tensor *shared_weights[] = {&l->shared_gate, &l->shared_up};
            const float *shared_inputs[] = {x, x};
            int shared_rows[] = {local_shared, local_shared};
            int shared_cols[] = {K3_HIDDEN, K3_HIDDEN};
            full_bf16_many(shared_outs, shared_weights, shared_inputs,
                           shared_rows, shared_cols, 2, m->threads);
        }
    }
    SH_MARK(K3_FULL_PHASE_SHARED_GATEUP);
    if (!shared_fused)
        full_situ(m->expert_gate, m->expert_gate, m->expert_up, local_shared);
    SH_MARK(K3_FULL_PHASE_SHARED_SITU);
    if (!shared_fused)
        full_bf16_matvec(m->shared_hidden, &l->shared_down, K3_HIDDEN,
                         local_shared, m->expert_gate, m->threads);
    SH_MARK(K3_FULL_PHASE_SHARED_DOWN);
#undef SH_MARK
    if (m->profile.enabled) {
        full_profile_phase_add(m, K3_FULL_PHASE_MOE_SHARED,
                               full_now() - subphase_start);
        subphase_start = full_now();
    }
    if (latent_async) {
        int latent_rc = latent_async == 2 ?
            tp_allreduce_sum_2d_sharded_finish(latent_comm, latent_col,
                                               m->local_latent, K3_LATENT) :
            g_half_col ?
            tp_allreduce_sum_2d_halves_finish(latent_comm, latent_col,
                                              m->local_latent, K3_LATENT) :
            tp_allreduce_sum_2d_finish(latent_comm, latent_col,
                                       m->local_latent, K3_LATENT);
        if (latent_rc) return EIO;
        if (m->profile.enabled)
            full_profile_phase_add(m, K3_FULL_PHASE_LATENT_REDUCE,
                                   full_now() - latent_async_start);
        subphase_start = m->profile.enabled ? full_now() : subphase_start;
    }

    k3_mxfp4_matrix w1[K3_TOP_K], w2[K3_TOP_K], w3[K3_TOP_K];
    for (int k = 0; k < K3_TOP_K; ++k) {
        if (route[k] < 0 || route[k] >= l->expert_count ||
            l->experts[route[k]].expert_id != route[k]) return EINVAL;
        if (!iq_experts && full_moe_cmg_replicate_on &&
            full_moe_cmg_replicate_max > 0 &&
            !l->experts[route[k]].mw1.packed_cmg[0] &&
            l->moe_cmg_replicated < full_moe_cmg_replicate_max) {
            if (!full_mxfp4_replicate_expert(&l->experts[route[k]]))
                ++l->moe_cmg_replicated;
        }
        w1[k] = l->experts[route[k]].mw1;
        w2[k] = l->experts[route[k]].mw2;
        w3[k] = l->experts[route[k]].mw3;
    }
    pthread_t prefetch_thread;
    k3_full_prefetch_job prefetch_job;
    int prefetch_active = full_prefetch_start(m, &l->routed_up,
                                              &prefetch_thread, &prefetch_job);
    if (iq_experts) {
        int local_inter = (int)l->experts[route[0]].w1.shape[0];
        float *gate_up_outs[2 * K3_TOP_K];
        const k3_full_tensor *gate_up_weights[2 * K3_TOP_K];
        const float *gate_up_inputs[2 * K3_TOP_K];
        int gate_up_rows[2 * K3_TOP_K], gate_up_cols[2 * K3_TOP_K];
        for (int k = 0; k < K3_TOP_K; ++k) {
            k3_full_expert *expert = &l->experts[route[k]];
            gate_up_outs[2 * k] = m->expert_gate + (size_t)k * local_inter;
            gate_up_outs[2 * k + 1] = m->expert_up + (size_t)k * local_inter;
            gate_up_weights[2 * k] = &expert->w1;
            gate_up_weights[2 * k + 1] = &expert->w3;
            gate_up_inputs[2 * k] = m->local_latent;
            gate_up_inputs[2 * k + 1] = m->local_latent;
            gate_up_rows[2 * k] = gate_up_rows[2 * k + 1] = local_inter;
            gate_up_cols[2 * k] = gate_up_cols[2 * k + 1] = K3_LATENT;
        }
        full_bf16_many(gate_up_outs, gate_up_weights, gate_up_inputs,
                       gate_up_rows, gate_up_cols, 2 * K3_TOP_K, m->threads);
        float *down_outs[K3_TOP_K];
        const k3_full_tensor *down_weights[K3_TOP_K];
        const float *down_inputs[K3_TOP_K];
        int down_rows[K3_TOP_K], down_cols[K3_TOP_K];
        for (int k = 0; k < K3_TOP_K; ++k) {
            float *gate = m->expert_gate + (size_t)k * local_inter;
            float *up = m->expert_up + (size_t)k * local_inter;
            full_situ(gate, gate, up, local_inter);
            down_outs[k] = m->expert_out + (size_t)k * K3_LATENT;
            down_weights[k] = &l->experts[route[k]].w2;
            down_inputs[k] = gate;
            down_rows[k] = K3_LATENT;
            down_cols[k] = local_inter;
        }
        full_bf16_many(down_outs, down_weights, down_inputs,
                       down_rows, down_cols, K3_TOP_K, m->threads);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < K3_LATENT; ++i) {
            float sum = 0.0f;
            for (int k = 0; k < K3_TOP_K; ++k)
                sum += route_weight[k] *
                       m->expert_out[(size_t)k * K3_LATENT + i];
            m->routed_latent[i] = sum;
        }
        full_iq_trace_finite(m, "expert_tp_routed_partial", m->routed_latent,
                             K3_LATENT);
    } else {
        /* The TP-selected expert kernel assigns every latent lane in its second
         * workshare; no seed memset is needed before the call. */
        if (k3_expert_tp_forward_selected_mxfp4(
                m->routed_latent, w1, w2, w3, route_weight, K3_TOP_K,
                m->local_latent, m->expert_gate, m->expert_up,
                m->expert_out, m->threads)) {
            if (prefetch_active) pthread_join(prefetch_thread, NULL);
            return EIO;
        }
    }
    if (m->profile.enabled) {
        full_profile_phase_add(m, K3_FULL_PHASE_MOE_EXPERT,
                               full_now() - subphase_start);
        subphase_start = full_now();
    }

    if (prefetch_active) pthread_join(prefetch_thread, NULL);

    if (full_moe_late_shared_reduce)
        memcpy(m->reduce, m->routed_latent, K3_LATENT * sizeof(float));
    else
        k3_moe_pack_reduce(m->reduce, m->routed_latent, m->shared_hidden);
    int middle_reduce_count = full_moe_late_shared_reduce
        ? K3_LATENT : K3_FULL_REDUCE_COUNT;
    if (full_sum(m, m->reduce, middle_reduce_count)) return EIO;
    full_iq_trace_finite(m, "expert_tp_routed_reduced", m->reduce, K3_LATENT);
    if (m->profile.enabled) {
        full_profile_phase_add(m, K3_FULL_PHASE_MOE_COLLECTIVE,
                               full_now() - subphase_start);
        subphase_start = full_now();
    }

    int local_hidden = (int)l->routed_up.shape[0];
    int local_up_cols = (int)l->routed_up.shape[1];
    int hidden_start = 0;
    int late_shared_seeded = 0;
    int up_col_sharded = local_hidden == K3_HIDDEN &&
                         local_up_cols != K3_LATENT;
    if (!up_col_sharded && local_hidden != K3_HIDDEN) {
        int shard_group = l->routed_up.dtype == 1 ? 8 : 32;
        if (l->routed_up.shape[1] != K3_LATENT ||
            full_split_groups(K3_HIDDEN, m->rank, m->nodes, shard_group,
                              &hidden_start, &local_hidden) ||
            (int)l->routed_up.shape[0] != local_hidden) {
            fprintf(stderr, "k3_full_runner rank %d: invalid routed-up shard shape\n",
                    m->rank);
            return EINVAL;
        }
    }
    double ft = m->profile.enabled ? full_now() : 0.0;
    full_rmsnorm_tensor(m->tmp, m->reduce, &l->routed_norm, K3_LATENT);
    if (up_col_sharded) {
        if (local_up_cols != m->latent_rows) return EINVAL;
        full_bf16_matvec(out, &l->routed_up, K3_HIDDEN, local_up_cols,
                         m->tmp + m->latent_start, m->threads);
    } else {
        if (local_hidden != K3_HIDDEN) {
            if (full_moe_late_shared_reduce) {
                memcpy(out, m->shared_hidden, K3_HIDDEN * sizeof(float));
                late_shared_seeded = 1;
            } else {
                memset(out, 0, K3_HIDDEN * sizeof(float));
            }
        }
        full_bf16_matvec(out + hidden_start, &l->routed_up, local_hidden,
                         K3_LATENT, m->tmp, m->threads);
    }
    if (m->profile.enabled) {
        full_profile_phase_add(m, K3_FULL_PHASE_FINISH_MATVEC, full_now() - ft);
        ft = full_now();
    }
    if (full_moe_late_shared_reduce) {
        if (late_shared_seeded)
            full_add(out + hidden_start, m->shared_hidden + hidden_start,
                     local_hidden);
        else
            full_add(out, m->shared_hidden, K3_HIDDEN);
        if (full_sum_final(m, out, K3_HIDDEN, l->is_mla)) return EIO;
    } else if ((up_col_sharded || local_hidden != K3_HIDDEN) &&
               full_sum_sharded(m, out, K3_HIDDEN)) return EIO;
    if (m->profile.enabled)
        full_profile_phase_add(m, K3_FULL_PHASE_FINISH_REDUCE, full_now() - ft);
    if (!full_moe_late_shared_reduce)
        full_add(out, m->reduce + K3_LATENT, K3_HIDDEN);
    if (m->profile.enabled)
        full_profile_phase_add(m, K3_FULL_PHASE_MOE_FINISH,
                               full_now() - subphase_start);
    return 0;
}

static uint64_t full_hash_f32(const float *values, int n);

static int full_moe_forward(k3_full_model *m, k3_full_layer *l,
                            const float *x, float *out) {
    if (m->expert_tp) return full_moe_forward_expert_tp(m, l, x, out);
    double subphase_start = m->profile.enabled ? full_now() : 0.0;
    int local_latent = (int)l->routed_down.shape[0];
    int local_shared = (int)l->shared_gate.shape[0];
    float router_logits[K3_EXPERTS];
    int route[K3_TOP_K];
    float route_weight[K3_TOP_K];
    memset(m->local_latent, 0, K3_LATENT * sizeof(float));
    float *route_outs[] = {router_logits, m->local_latent + m->latent_start,
                           m->gate, m->up};
    const k3_full_tensor *route_weights[] = {&l->router, &l->routed_down,
                                             &l->shared_gate, &l->shared_up};
    const float *route_inputs[] = {x, x, x, x};
    int route_rows[] = {K3_EXPERTS, local_latent, local_shared, local_shared};
    int route_cols[] = {K3_HIDDEN, K3_HIDDEN, K3_HIDDEN, K3_HIDDEN};
    full_bf16_many(route_outs, route_weights, route_inputs,
                   route_rows, route_cols, 4, m->threads);
    if (full_iq_trace && m->rank == 0) {
        int x_finite = 0, logits_finite = 0, latent_finite = 0;
        float x_lo = INFINITY, x_hi = -INFINITY;
        float logit_lo = INFINITY, logit_hi = -INFINITY;
        for (int i = 0; i < K3_HIDDEN; ++i) {
            if (!isfinite(x[i])) continue;
            ++x_finite;
            if (x[i] < x_lo) x_lo = x[i];
            if (x[i] > x_hi) x_hi = x[i];
        }
        for (int i = 0; i < K3_EXPERTS; ++i) {
            if (!isfinite(router_logits[i])) continue;
            ++logits_finite;
            if (router_logits[i] < logit_lo) logit_lo = router_logits[i];
            if (router_logits[i] > logit_hi) logit_hi = router_logits[i];
        }
        for (int i = 0; i < local_latent; ++i)
            latent_finite += isfinite(m->local_latent[m->latent_start + i]);
        printf("K3_IQ_FULL_TRACE projection x_finite=%d/%d x0=%+.9e x_range=[%+.9e,%+.9e] "
               "logits_finite=%d/%d logit0=%+.9e logit_range=[%+.9e,%+.9e] "
               "bias0=%+.9e latent_finite=%d/%d latent0=%+.9e\n",
               x_finite, K3_HIDDEN, x[0], x_lo, x_hi,
               logits_finite, K3_EXPERTS, router_logits[0], logit_lo, logit_hi,
               ((const float *)l->router_bias.data)[0], latent_finite, local_latent,
               m->local_latent[m->latent_start]);
    }
    for (int e = 0; e < K3_EXPERTS; ++e) {
        if (!isfinite(router_logits[e])) {
            if (m->rank == 0)
                fprintf(stderr, "k3_full_runner: non-finite router logit e=%d value=%g\n",
                        e, router_logits[e]);
            return EDOM;
        }
    }
    if (m->profile.enabled) {
        full_profile_phase_add(m, K3_FULL_PHASE_DISPATCH_PROJ,
                               full_now() - subphase_start);
        subphase_start = full_now();
    }
    k3_router_topk(router_logits, (const float *)l->router_bias.data,
                   K3_EXPERTS, K3_TOP_K, route, route_weight);
    if (full_sum(m, m->local_latent, K3_LATENT)) return EIO;
    if (m->profile.enabled) {
        full_profile_phase_add(m, K3_FULL_PHASE_MOE_DISPATCH,
                               full_now() - subphase_start);
        subphase_start = full_now();
    }

    full_situ(m->gate, m->gate, m->up, local_shared);
    full_bf16_matvec(m->tmp2, &l->shared_down, K3_HIDDEN,
                     local_shared, m->gate, m->threads);
    if (m->profile.enabled) {
        full_profile_phase_add(m, K3_FULL_PHASE_MOE_SHARED,
                               full_now() - subphase_start);
        subphase_start = full_now();
    }

    memset(m->expert_counts, 0, (size_t)l->expert_count * sizeof(int));
    memset(m->expert_tokens, 0, (size_t)l->expert_count * sizeof(int));
    memset(m->expert_weights, 0, (size_t)l->expert_count * sizeof(float));
    memset(m->expert_gathered, 0,
           (size_t)l->expert_count * K3_LATENT * sizeof(float));
    for (int k = 0; k < K3_TOP_K; ++k) {
        for (int e = 0; e < l->expert_count; ++e) {
            if (l->experts[e].expert_id == route[k]) {
                m->expert_counts[e] = 1;
                m->expert_tokens[e] = 0;
                m->expert_weights[e] = route_weight[k];
                memcpy(m->expert_gathered + (size_t)e * K3_LATENT,
                       m->local_latent, K3_LATENT * sizeof(float));
                break;
            }
        }
    }
    memset(m->routed_latent, 0, K3_LATENT * sizeof(float));
    int iq_experts = l->expert_count > 0 &&
        l->experts[0].w1.dtype >= K3_FULL_DTYPE_IQ1_S &&
        l->experts[0].w1.dtype <= K3_FULL_DTYPE_IQ3_XXS;
    if (iq_experts) {
        int selected[K3_TOP_K], selected_count = 0;
        for (int e = 0; e < l->expert_count; ++e) {
            if (m->expert_counts[e]) selected[selected_count++] = e;
        }
        float *gate_up_outs[2 * K3_TOP_K];
        const k3_full_tensor *gate_up_weights[2 * K3_TOP_K];
        const float *gate_up_inputs[2 * K3_TOP_K];
        int gate_up_rows[2 * K3_TOP_K], gate_up_cols[2 * K3_TOP_K];
        for (int s = 0; s < selected_count; ++s) {
            int e = selected[s];
            gate_up_outs[2 * s] = m->expert_gate + (size_t)e * K3_EXPERT_INTER;
            gate_up_outs[2 * s + 1] = m->expert_up + (size_t)e * K3_EXPERT_INTER;
            gate_up_weights[2 * s] = &l->experts[e].w1;
            gate_up_weights[2 * s + 1] = &l->experts[e].w3;
            gate_up_inputs[2 * s] = m->local_latent;
            gate_up_inputs[2 * s + 1] = m->local_latent;
            gate_up_rows[2 * s] = gate_up_rows[2 * s + 1] = K3_EXPERT_INTER;
            gate_up_cols[2 * s] = gate_up_cols[2 * s + 1] = K3_LATENT;
        }
        full_bf16_many(gate_up_outs, gate_up_weights, gate_up_inputs,
                       gate_up_rows, gate_up_cols, 2 * selected_count,
                       m->threads);
        float *down_outs[K3_TOP_K];
        const k3_full_tensor *down_weights[K3_TOP_K];
        const float *down_inputs[K3_TOP_K];
        int down_rows[K3_TOP_K], down_cols[K3_TOP_K];
        for (int s = 0; s < selected_count; ++s) {
            int e = selected[s];
            float *gate = m->expert_gate + (size_t)e * K3_EXPERT_INTER;
            float *up = m->expert_up + (size_t)e * K3_EXPERT_INTER;
            full_situ(gate, gate, up, K3_EXPERT_INTER);
            down_outs[s] = m->expert_out + (size_t)e * K3_LATENT;
            down_weights[s] = &l->experts[e].w2;
            down_inputs[s] = gate;
            down_rows[s] = K3_LATENT;
            down_cols[s] = K3_EXPERT_INTER;
        }
        full_bf16_many(down_outs, down_weights, down_inputs,
                       down_rows, down_cols, selected_count, m->threads);
#if defined(_OPENMP)
#pragma omp parallel
        {
            for (int s = 0; s < selected_count; ++s) {
                int e = selected[s];
#pragma omp for schedule(static)
                for (int i = 0; i < K3_LATENT; ++i)
                    m->routed_latent[i] += m->expert_weights[e] *
                        m->expert_out[(size_t)e * K3_LATENT + i];
            }
        }
#else
        for (int s = 0; s < selected_count; ++s) {
            int e = selected[s];
            for (int i = 0; i < K3_LATENT; ++i)
                m->routed_latent[i] += m->expert_weights[e] *
                    m->expert_out[(size_t)e * K3_LATENT + i];
        }
#endif
    } else {
        k3_mxfp4_matrix w1[l->expert_count], w2[l->expert_count],
                         w3[l->expert_count];
        for (int e = 0; e < l->expert_count; ++e) {
            w1[e] = l->experts[e].mw1;
            w2[e] = l->experts[e].mw2;
            w3[e] = l->experts[e].mw3;
        }
        k3_moe_forward_local_mxfp4(m->routed_latent, w1, w2, w3,
                                    l->expert_count, m->expert_counts,
                                    m->expert_tokens, m->expert_weights,
                                    m->expert_gathered, 1, m->expert_gate,
                                    m->expert_up, m->expert_out, m->threads, 8);
    }
    if (m->profile.enabled) {
        full_profile_phase_add(m, K3_FULL_PHASE_MOE_EXPERT,
                               full_now() - subphase_start);
        subphase_start = full_now();
    }
    if (full_sum(m, m->routed_latent, K3_LATENT)) return EIO;
    if (full_iq_trace && m->rank == 0) {
        printf("K3_IQ_FULL_TRACE route=");
        for (int k = 0; k < K3_TOP_K; ++k)
            printf("%s%d", k ? "," : "", route[k]);
        printf(" routed_hash=%016llx routed0=%+.9e\n",
               (unsigned long long)full_hash_f32(m->routed_latent, K3_LATENT),
               m->routed_latent[0]);
    }
    if (m->profile.enabled) {
        full_profile_phase_add(m, K3_FULL_PHASE_MOE_COLLECTIVE,
                               full_now() - subphase_start);
        subphase_start = full_now();
    }
    full_rmsnorm_tensor(m->tmp, m->routed_latent, &l->routed_norm, K3_LATENT);
    full_bf16_matvec(m->moe_hidden, &l->routed_up, K3_HIDDEN,
                     local_latent, m->tmp + m->latent_start, m->threads);
    full_add(m->moe_hidden, m->tmp2, K3_HIDDEN);
    if (m->profile.enabled) {
        full_profile_phase_add(m, K3_FULL_PHASE_MOE_FINISH,
                               full_now() - subphase_start);
        subphase_start = full_now();
    }
    if (full_sum(m, m->moe_hidden, K3_HIDDEN)) return EIO;
    if (m->profile.enabled)
        full_profile_phase_add(m, K3_FULL_PHASE_MOE_COLLECTIVE,
                               full_now() - subphase_start);
    full_copy(out, m->moe_hidden, K3_HIDDEN);
    return 0;
}

static int full_forward_token(k3_full_model *m, int token, int position) {
    m->block_count = 0;
    memset(m->hidden, 0, K3_HIDDEN * sizeof(float));
    if (token >= m->embed_start && token < m->embed_start + m->embed_rows) {
        int row = token - m->embed_start;
        if (m->embed.dtype == 1) {
            const uint16_t *w = (const uint16_t *)m->embed.data +
                (size_t)row * K3_HIDDEN;
            for (int i = 0; i < K3_HIDDEN; ++i)
                m->hidden[i] = bf16_to_f32_scalar(w[i]);
        } else if (m->embed.dtype == 2) {
            memcpy(m->hidden, m->embed.data +
                   (size_t)row * K3_HIDDEN * sizeof(float),
                   K3_HIDDEN * sizeof(float));
        } else {
            int qt = K3_Q_Q8_0 + (m->embed.dtype - K3_FULL_DTYPE_Q8_0);
            size_t row_bytes = k3_quant_row_bytes(qt, K3_HIDDEN);
            if (m->embed.dtype < K3_FULL_DTYPE_Q8_0 ||
                m->embed.dtype > K3_FULL_DTYPE_IQ3_XXS ||
                k3_quant_dequant_row(m->hidden, m->embed.data +
                    (size_t)row * row_bytes, qt, K3_HIDDEN)) return EINVAL;
        }
    }
    if (full_sum(m, m->hidden, K3_HIDDEN)) return EIO;
    for (int layer = 0; layer < K3_LAYERS; ++layer) {
        double layer_start = m->profile.enabled ? full_now() : 0.0;
        double phase_start = m->profile.enabled ? full_now() : 0.0;
        full_profile_layer_begin(m, layer);
        k3_full_layer *l = &m->layers[layer];
        full_copy(m->tmp, m->hidden, K3_HIDDEN); /* prefix_sum */
        if (m->block_count > 0)
            full_attn_res(m->tmp2, m->tmp, m->block_residual,
                          m->block_count, &l->attn_res_proj,
                          &l->attn_res_norm);
        const float *attn_input = m->block_count > 0 ? m->tmp2 : m->tmp;
        int new_block = (layer % 12) == 0;
        if (new_block) {
            memcpy(m->block_residual + (size_t)m->block_count * K3_HIDDEN,
                   m->tmp, K3_HIDDEN * sizeof(float));
            ++m->block_count;
        }
        full_rmsnorm_tensor(m->normed, attn_input, &l->input_norm, K3_HIDDEN);
        full_profile_phase_begin(m, K3_FULL_PHASE_ATTENTION);
        if (l->is_mla)
            full_mla_forward(m, l, m->normed, position, m->attn);
        else
            full_kda_forward(m, l, m->normed, m->attn);
        if (full_sum_attention(m, m->attn, K3_HIDDEN)) {
            full_profile_layer_end(m, layer, layer_start);
            return EIO;
        }
        if (m->profile.enabled)
            full_profile_phase_add(m, K3_FULL_PHASE_ATTENTION,
                                   full_now() - phase_start);
        if (new_block) full_copy(m->tmp, m->attn, K3_HIDDEN);
        else full_add(m->tmp, m->attn, K3_HIDDEN);
        full_attn_res(m->tmp2, m->tmp, m->block_residual,
                      m->block_count, &l->mlp_res_proj, &l->mlp_res_norm);
        full_rmsnorm_tensor(m->normed, m->tmp2, &l->post_norm, K3_HIDDEN);
        phase_start = m->profile.enabled ? full_now() : 0.0;
        full_profile_phase_begin(m, K3_FULL_PHASE_MOE);
        if (layer == 0)
            full_dense_forward(m, l, m->normed, m->moe_hidden);
        else if (full_moe_forward(m, l, m->normed, m->moe_hidden))
        {
            full_profile_layer_end(m, layer, layer_start);
            return EIO;
        }
        if (layer == 0 && full_sum(m, m->moe_hidden, K3_HIDDEN)) {
            full_profile_layer_end(m, layer, layer_start);
            return EIO;
        }
        if (m->profile.enabled)
            full_profile_phase_add(m, K3_FULL_PHASE_MOE,
                                   full_now() - phase_start);
        phase_start = m->profile.enabled ? full_now() : 0.0;
        full_profile_phase_begin(m, K3_FULL_PHASE_RESIDUAL);
        full_add_copy(m->hidden, m->tmp, m->moe_hidden, K3_HIDDEN);
        if (m->profile.enabled)
            full_profile_phase_add(m, K3_FULL_PHASE_RESIDUAL,
                                   full_now() - phase_start);
        full_profile_layer_end(m, layer, layer_start);
    }
    full_rmsnorm_tensor(m->hidden, m->hidden, &m->final_norm, K3_HIDDEN);
    return 0;
}

static int full_forward_prefill_chunk_real(k3_full_model *m,
                                           const int *tokens, int count,
                                           int position) {
    for (int i = 0; i < count; ++i)
        if (full_forward_token(m, tokens[i], position + i)) return EIO;
    return 0;
}

static int full_argmax(k3_full_model *m, float *max_value_out) {
    full_bf16_matvec(m->logits, &m->lm_head, m->head_rows,
                     K3_HIDDEN, m->hidden, m->threads);
    int best = 0;
    for (int i = 1; i < m->head_rows; ++i)
        if (m->logits[i] > m->logits[best]) best = i;
    float value = m->logits[best];
    if (full_max(m, &value, 1)) return -1;
    float encoded = -1.0e30f;
    for (int i = 0; i < m->head_rows; ++i) {
        if (m->logits[i] == value) {
            encoded = -(float)(m->head_start + i);
            break;
        }
    }
    if (full_max(m, &encoded, 1)) return -1;
    if (max_value_out) *max_value_out = value;
    return (int)lrintf(-encoded);
}

static int full_read_ids(const char *path, int *ids, int wanted) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "k3_full_runner: cannot open prompt IDs %s: %s\n",
                path, strerror(errno));
        return -1;
    }
    int n = 0;
    char line[4096];
    while (fgets(line, sizeof line, f) && n < wanted) {
        char *p = line;
        while (*p) {
            while (*p == ' ' || *p == '\t' || *p == ',' || *p == '\n' || *p == '\r') ++p;
            if (!*p || *p == '#') break;
            char *end = NULL;
            long value = strtol(p, &end, 10);
            if (end == p || value < 0 || value >= K3_FULL_VOCAB) {
                fclose(f);
                return -1;
            }
            ids[n++] = (int)value;
            p = end;
        }
    }
    fclose(f);
    return n == wanted ? n : -1;
}

static uint64_t full_hash_ids(const int *ids, int n) {
    uint64_t h = UINT64_C(1469598103934665603);
    for (int i = 0; i < n; ++i) {
        uint32_t x = (uint32_t)ids[i];
        for (int b = 0; b < 4; ++b) {
            h ^= (uint8_t)(x >> (8 * b));
            h *= UINT64_C(1099511628211);
        }
    }
    return h;
}

static uint64_t full_hash_f32(const float *values, int n) {
    uint64_t h = UINT64_C(1469598103934665603);
    for (int i = 0; i < n; ++i) {
        uint32_t bits = 0;
        memcpy(&bits, values + i, sizeof bits);
        for (int b = 0; b < 4; ++b) {
            h ^= (uint8_t)(bits >> (8 * b));
            h *= UINT64_C(1099511628211);
        }
    }
    return h;
}

static int full_parse_int(const char *flag, const char *s, int lo, int hi, int *out) {
    char *end = NULL;
    long value = strtol(s, &end, 10);
    if (!s[0] || !end || *end || value < lo || value > hi) {
        fprintf(stderr, "k3_full_runner: %s expects [%d,%d], got %s\n",
                flag, lo, hi, s);
        return -1;
    }
    *out = (int)value;
    return 0;
}

static int full_parse_u64(const char *flag, const char *s, uint64_t *out) {
    char *end = NULL;
    unsigned long long value = strtoull(s, &end, 0);
    if (!s[0] || !end || *end) {
        fprintf(stderr, "k3_full_runner: %s expects an integer, got %s\n",
                flag, s);
        return -1;
    }
    *out = (uint64_t)value;
    return 0;
}

static const char *full_mode_name(k3_full_mode mode) {
    if (mode == K3_FULL_MODE_LAYER12) return "layer12";
    if (mode == K3_FULL_MODE_SYNTHETIC12) return "synthetic12";
    if (mode == K3_FULL_MODE_BARRIER) return "barrier";
    return "full96";
}

static int full_parse_mode(const char *s, k3_full_mode *out) {
    if (!strcmp(s, "full96")) *out = K3_FULL_MODE_FULL96;
    else if (!strcmp(s, "layer12")) *out = K3_FULL_MODE_LAYER12;
    else if (!strcmp(s, "synthetic12")) *out = K3_FULL_MODE_SYNTHETIC12;
    else if (!strcmp(s, "barrier")) *out = K3_FULL_MODE_BARRIER;
    else return -1;
    return 0;
}

static void full_usage(const char *program) {
    fprintf(stderr,
            "usage: %s --stage-dir DIR --output FILE\n"
            "       [--mode full96|layer12|synthetic12] [--topo FILE]\n"
            "       [--nodes 96|12] [--threads N] [--real-layer-index N]\n"
            "       [--prompt-ids IDS] [--input-seed N] [--prefill-tokens N]\n"
            "       [--new-tokens N] [--prefill-chunk N] [--max-seq N]\n"
            "       [--barrier-iters N] [--comm-deterministic 0|1] [--comm-bf16 0|1]\n"
            "       [--comm-robust N] [--comm-poll-spins N] [--comm-a2a 0|1]\n"
            "       [--comm-a2a-max N] [--prefetch-mib N]\n"
            "       [--ar-groups N] [--profile [FILE]]\n"
            "       [--prefill-only]\n"
            "       %s --mode barrier --topo FILE --nodes N [--barrier-iters N]\n",
            program, program);
}

static int full_options(int argc, char **argv, k3_full_options *o) {
    *o = (k3_full_options){
        .mode = K3_FULL_MODE_FULL96, .nodes = 96, .threads = 48,
        .max_seq = 12288, .real_layer_index = 1,
        .input_seed = UINT64_C(0x4b33444542554701), .prefill_chunk = 1,
        .barrier_iters = K3_FULL_BARRIER_ITERS_DEFAULT,
        /* Fixed-root reductions are required for rank-identical full decode. */
        .comm_deterministic = 1,
        .ar_groups = 0,
        .comm_use_bf16 = 0,
        .comm_robust = 2,
        .comm_poll_spins = 4,
        .comm_a2a = 0,
        .comm_a2a_max = 8192,
        /* Off by default: the prefetch is a per-layer pthread_create whose
         * worker issues one __builtin_prefetch per 64 bytes over 16 MiB on a
         * single core, and it costs more than it saves.  Measured on layer 3 at
         * 12 nodes, 512 samples: 4.14 ms/layer with it off against 4.89-5.41
         * with it on, and the off case is reproducible (4.146/4.138) where the
         * on case is not -- the rogue thread adds jitter as well as time.  The
         * production 96-node script already passed 0. */
        .prefetch_mib = 0,
        .profile = 0,
        .profile_output = NULL,
        .prefill_tokens = 8192, .new_tokens = 4096,
        .stage_dir = NULL, .topo_path = "tofu_topo.txt",
        .prompt_ids = NULL, .output_path = NULL,
    };
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
#define VALUE() do { if (++i >= argc) { full_usage(argv[0]); return -1; } } while (0)
        if (!strcmp(a, "--mode")) { VALUE(); if (full_parse_mode(argv[i], &o->mode)) { full_usage(argv[0]); return -1; } }
        else if (!strcmp(a, "--stage-dir")) { VALUE(); o->stage_dir = argv[i]; }
        else if (!strcmp(a, "--topo")) { VALUE(); o->topo_path = argv[i]; }
        else if (!strcmp(a, "--prompt-ids")) { VALUE(); o->prompt_ids = argv[i]; }
        else if (!strcmp(a, "--output")) { VALUE(); o->output_path = argv[i]; }
        else if (!strcmp(a, "--nodes")) { VALUE(); if (full_parse_int(a, argv[i], 12, 96, &o->nodes)) return -1; }
        else if (!strcmp(a, "--threads")) { VALUE(); if (full_parse_int(a, argv[i], 1, 48, &o->threads)) return -1; }
        else if (!strcmp(a, "--real-layer-index")) { VALUE(); if (full_parse_int(a, argv[i], 0, K3_LAYERS - 1, &o->real_layer_index)) return -1; }
        else if (!strcmp(a, "--input-seed")) { VALUE(); if (full_parse_u64(a, argv[i], &o->input_seed)) return -1; }
        else if (!strcmp(a, "--prefill-tokens")) { VALUE(); if (full_parse_int(a, argv[i], 1, 8192, &o->prefill_tokens)) return -1; }
        else if (!strcmp(a, "--new-tokens")) { VALUE(); if (full_parse_int(a, argv[i], 0, 4096, &o->new_tokens)) return -1; }
        else if (!strcmp(a, "--prefill-chunk")) { VALUE(); if (full_parse_int(a, argv[i], 1, 1024, &o->prefill_chunk)) return -1; }
        else if (!strcmp(a, "--max-seq")) { VALUE(); if (full_parse_int(a, argv[i], 1, 16384, &o->max_seq)) return -1; }
        else if (!strcmp(a, "--barrier-iters")) { VALUE(); if (full_parse_int(a, argv[i], 1, 100000, &o->barrier_iters)) return -1; }
        else if (!strcmp(a, "--comm-deterministic")) { VALUE(); if (full_parse_int(a, argv[i], 0, 1, &o->comm_deterministic)) return -1; }
        else if (!strcmp(a, "--comm-bf16")) { VALUE(); if (full_parse_int(a, argv[i], 0, 1, &o->comm_use_bf16)) return -1; }
        else if (!strcmp(a, "--comm-robust")) { VALUE(); if (full_parse_int(a, argv[i], 0, 2, &o->comm_robust)) return -1; }
        else if (!strcmp(a, "--comm-poll-spins")) { VALUE(); if (full_parse_int(a, argv[i], 1, 1024, &o->comm_poll_spins)) return -1; }
        else if (!strcmp(a, "--comm-a2a")) { VALUE(); if (full_parse_int(a, argv[i], 0, 1, &o->comm_a2a)) return -1; }
        else if (!strcmp(a, "--comm-a2a-max")) { VALUE(); if (full_parse_int(a, argv[i], 1, K3_FULL_REDUCE_COUNT, &o->comm_a2a_max)) return -1; }
        else if (!strcmp(a, "--prefetch-mib")) { VALUE(); if (full_parse_int(a, argv[i], 0, 16, &o->prefetch_mib)) return -1; }
        else if (!strcmp(a, "--ar-groups")) { VALUE(); if (full_parse_int(a, argv[i], 0, 96, &o->ar_groups)) return -1; }
        else if (!strcmp(a, "--profile")) {
            o->profile = 1;
            if (i + 1 < argc && argv[i + 1][0] != '-') o->profile_output = argv[++i];
        }
        else if (!strcmp(a, "--prefill-only")) o->prefill_only = 1;
        else if (!strcmp(a, "--help") || !strcmp(a, "-h")) { full_usage(argv[0]); return 1; }
        else { fprintf(stderr, "k3_full_runner: unknown option %s\n", a); full_usage(argv[0]); return -1; }
#undef VALUE
    }
    if (o->mode == K3_FULL_MODE_BARRIER) {
        if (o->nodes < 2) {
            fprintf(stderr, "k3_full_runner: barrier mode requires at least two nodes\n");
            return -1;
        }
        return 0;
    }
    if (!o->stage_dir || !o->output_path) {
        full_usage(argv[0]);
        return -1;
    }
    if (o->mode == K3_FULL_MODE_FULL96) {
        if (o->nodes != 96 || !o->prompt_ids) {
            fprintf(stderr, "k3_full_runner: full96 requires nodes=96 and --prompt-ids\n");
            return -1;
        }
    } else if (o->nodes != 12) {
        fprintf(stderr, "k3_full_runner: %s requires nodes=12\n",
                full_mode_name(o->mode));
        return -1;
    }
    if (o->prefill_tokens + o->new_tokens > o->max_seq) {
        fprintf(stderr, "k3_full_runner: prompt plus generation exceeds --max-seq\n");
        return -1;
    }
    if (o->ar_groups && o->nodes % o->ar_groups) {
        fprintf(stderr, "k3_full_runner: --ar-groups must divide nodes\n");
        return -1;
    }
    if (o->comm_poll_spins & (o->comm_poll_spins - 1)) {
        fprintf(stderr, "k3_full_runner: --comm-poll-spins must be a power of two\n");
        return -1;
    }
    return 0;
}

static int full_barrier_stress(const k3_full_options *o) {
    for (int i = 0; i < o->barrier_iters; ++i) {
        /* Deliberately skew a changing rank before fan-in.  This exercises
         * both the root's ordered receive polling and non-root retransmit
         * path without making the batch preflight depend on one fixed rank. */
        int delayed_rank = (i * 17 + 46) % g_nodes;
        if (g_rank == delayed_rank)
            usleep((useconds_t)(250 + (i % 7) * 250));
        if ((i & 15) == 7)
            sched_yield();
        full_barrier();
        if ((i & 31) == 0 && g_rank != 0)
            usleep((useconds_t)((g_rank * 13 + i) % 300));
    }
    return 0;
}

static int full_allocate_scratch(k3_full_model *m, const k3_full_options *o) {
    int max_experts = 0;
    if (m->debug_layer_index >= 0) {
        max_experts = m->expert_tp ? K3_TOP_K : m->debug_layer.expert_count;
    } else {
        if (m->expert_tp) {
            max_experts = K3_TOP_K;
        } else {
            for (int layer = 1; layer < K3_LAYERS; ++layer)
                if (m->layers[layer].expert_count > max_experts)
                    max_experts = m->layers[layer].expert_count;
        }
    }
    size_t kda_state_elems = (size_t)m->kda_count * m->local_heads *
        K3_HEAD_DIM * K3_HEAD_DIM;
    size_t conv_state_elems = (size_t)m->kda_count * m->local_heads *
        K3_HEAD_DIM * (K3_FULL_CONV_KERNEL - 1) * 3;
    size_t mla_key_elems = (size_t)m->mla_count * m->local_heads * o->max_seq * K3_FULL_MLA_QK;
    size_t mla_value_elems = (size_t)m->mla_count * m->local_heads * o->max_seq * K3_FULL_MLA_VALUE;
    if (!kda_state_elems) kda_state_elems = 1;
    if (!conv_state_elems) conv_state_elems = 1;
    if (!mla_key_elems) mla_key_elems = 1;
    if (!mla_value_elems) mla_value_elems = 1;
    m->kda_state = (float *)k3_pool_calloc(m->pool, kda_state_elems, sizeof(float));
    m->conv_state = (float *)k3_pool_calloc(m->pool, conv_state_elems, sizeof(float));
    m->mla_keys = (float *)k3_pool_calloc(m->pool, mla_key_elems, sizeof(float));
    m->mla_values = (float *)k3_pool_calloc(m->pool, mla_value_elems, sizeof(float));
    m->block_residual = (float *)k3_pool_alloc(m->pool,
                                               (size_t)((K3_LAYERS + 11) / 12) * K3_HIDDEN * sizeof(float));
    m->hidden = (float *)k3_pool_alloc(m->pool, K3_HIDDEN * sizeof(float));
    m->normed = (float *)k3_pool_alloc(m->pool, K3_HIDDEN * sizeof(float));
    m->tmp = (float *)k3_pool_alloc(m->pool, K3_HIDDEN * sizeof(float));
    m->tmp2 = (float *)k3_pool_alloc(m->pool, K3_HIDDEN * sizeof(float));
    m->attn = (float *)k3_pool_alloc(m->pool, K3_HIDDEN * sizeof(float));
    int head_channels = m->local_heads * K3_HEAD_DIM;
    int mla_q_channels = m->local_heads * K3_FULL_MLA_QK;
    int mla_k_channels = m->local_heads * 256;
    int max_q_channels = head_channels > mla_q_channels ? head_channels : mla_q_channels;
    int max_k_channels = head_channels > mla_k_channels ? head_channels : mla_k_channels;
    m->q = (float *)k3_pool_alloc(m->pool, (size_t)max_q_channels * sizeof(float));
    m->k = (float *)k3_pool_alloc(m->pool, (size_t)max_k_channels * sizeof(float));
    m->v = (float *)k3_pool_alloc(m->pool, (size_t)head_channels * sizeof(float));
    m->gate = (float *)k3_pool_alloc(m->pool, (size_t)head_channels * sizeof(float));
    m->up = (float *)k3_pool_alloc(m->pool, (size_t)head_channels * sizeof(float));
    m->decay = (float *)k3_pool_alloc(m->pool, (size_t)head_channels * sizeof(float));
    m->local_latent = (float *)k3_pool_alloc(m->pool, K3_LATENT * sizeof(float));
    m->routed_latent = (float *)k3_pool_alloc(m->pool, K3_LATENT * sizeof(float));
    m->moe_hidden = (float *)k3_pool_alloc(m->pool, K3_HIDDEN * sizeof(float));
    size_t logits_rows = m->head_rows > 0 ? (size_t)m->head_rows : 1;
    m->logits = (float *)k3_pool_alloc(m->pool, logits_rows * sizeof(float));
    m->expert_gathered = (float *)k3_pool_alloc(m->pool,
                                                (size_t)max_experts * K3_LATENT * sizeof(float));
    m->expert_gate = (float *)k3_pool_alloc(m->pool,
                                            (size_t)max_experts * K3_EXPERT_INTER * sizeof(float));
    m->expert_up = (float *)k3_pool_alloc(m->pool,
                                          (size_t)max_experts * K3_EXPERT_INTER * sizeof(float));
    m->expert_out = (float *)k3_pool_alloc(m->pool,
                                           (size_t)max_experts * K3_LATENT * sizeof(float));
    m->shared_hidden = (float *)k3_pool_alloc(m->pool, K3_HIDDEN * sizeof(float));
    m->reduce = (float *)k3_pool_alloc(m->pool,
                                       K3_FULL_REDUCE_COUNT * sizeof(float));
    m->q_scratch = (int8_t *)k3_pool_alloc(m->pool, K3_LATENT * sizeof(int8_t));
    /* Sized as in k3_ep_runner.c:869-870 for k3_attention_heads_parallel_sve. */
    m->mla_scratch = (float *)k3_pool_alloc(m->pool,
        (size_t)m->local_heads * m->threads * K3_HEAD_DIM * sizeof(float));
    m->mla_stats = (float *)k3_pool_alloc(m->pool,
        ((size_t)m->local_heads * m->threads * 2 +
         (size_t)m->local_heads * 2) * sizeof(float));
    m->expert_counts = (int *)k3_pool_alloc(m->pool, (size_t)max_experts * sizeof(int));
    m->expert_tokens = (int *)k3_pool_alloc(m->pool, (size_t)max_experts * sizeof(int));
    m->expert_weights = (float *)k3_pool_alloc(m->pool, (size_t)max_experts * sizeof(float));
    if (!m->kda_state || !m->conv_state || !m->mla_keys || !m->mla_values ||
        !m->block_residual || !m->hidden || !m->normed || !m->tmp || !m->tmp2 ||
        !m->attn || !m->q || !m->k || !m->v || !m->gate || !m->up || !m->decay ||
        !m->mla_scratch || !m->mla_stats ||
        !m->local_latent || !m->routed_latent || !m->moe_hidden || !m->logits ||
        !m->expert_gathered || !m->expert_gate || !m->expert_up || !m->expert_out ||
        !m->shared_hidden || !m->reduce || !m->q_scratch ||
        !m->expert_counts || !m->expert_tokens || !m->expert_weights) return ENOMEM;
    return 0;
}

static uint64_t full_mix64(uint64_t x) {
    x ^= x >> 30;
    x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x *= UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

static int full_debug_token_id(uint64_t seed, int position) {
    return (int)(full_mix64(seed + (uint64_t)position * UINT64_C(0x9e3779b97f4a7c15)) % K3_FULL_VOCAB);
}

static void full_debug_seed_hidden(float *hidden, uint64_t seed, int token,
                                   int position) {
    uint64_t base = seed ^ ((uint64_t)(unsigned)token << 32) ^
                    (uint64_t)(unsigned)position;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < K3_HIDDEN; ++i) {
        uint64_t h = full_mix64(base + (uint64_t)i * UINT64_C(0x9e3779b97f4a7c15));
        float unit = (float)((h >> 40) & UINT64_C(0xffffff)) / 16777216.0f;
        hidden[i] = (unit * 2.0f - 1.0f) * 0.02f;
    }
}

static int full_debug_layer_forward(k3_full_model *m, k3_full_layer *l,
                                    int position, int input_in_tmp) {
    double layer_start = m->profile.enabled ? full_now() : 0.0;
    full_profile_layer_begin(m, m->debug_layer_index);
    m->block_count = 0;
    if (!input_in_tmp)
        full_copy(m->tmp, m->hidden, K3_HIDDEN);
    full_iq_trace_finite(m, "hidden", m->tmp, K3_HIDDEN);
    full_rmsnorm_tensor(m->normed, m->tmp, &l->input_norm, K3_HIDDEN);
    full_iq_trace_finite(m, "attn_input_norm", m->normed, K3_HIDDEN);
    double phase_start = m->profile.enabled ? full_now() : 0.0;
    full_profile_phase_begin(m, K3_FULL_PHASE_ATTENTION);
    if (l->is_mla)
        full_mla_forward(m, l, m->normed, position, m->attn);
    else
        full_kda_forward(m, l, m->normed, m->attn);
    if (full_sum_attention(m, m->attn, K3_HIDDEN)) {
        if (m->profile.enabled)
            full_profile_phase_add(m, K3_FULL_PHASE_ATTENTION,
                                   full_now() - phase_start);
        full_profile_layer_end(m, m->debug_layer_index, layer_start);
        return EIO;
    }
    full_iq_trace_finite(m, "attn_reduced", m->attn, K3_HIDDEN);
    if (m->profile.enabled)
        full_profile_phase_add(m, K3_FULL_PHASE_ATTENTION,
                               full_now() - phase_start);
    full_add(m->tmp, m->attn, K3_HIDDEN);
    full_iq_trace_finite(m, "attn_residual", m->tmp, K3_HIDDEN);
    /* A standalone layer has no earlier block residuals, so attn_res is the
     * identity.  Consume tmp directly and avoid a redundant 28 KiB copy. */
    full_iq_trace_finite(m, "mlp_res_mix", m->tmp, K3_HIDDEN);
    full_rmsnorm_tensor(m->normed, m->tmp, &l->post_norm, K3_HIDDEN);
    full_iq_trace_finite(m, "moe_input_norm", m->normed, K3_HIDDEN);
    phase_start = m->profile.enabled ? full_now() : 0.0;
    full_profile_phase_begin(m, K3_FULL_PHASE_MOE);
    int rc;
    if (m->debug_layer_index == 0) {
        full_dense_forward(m, l, m->normed, m->moe_hidden);
        rc = full_sum(m, m->moe_hidden, K3_HIDDEN);
    } else {
        rc = full_moe_forward(m, l, m->normed, m->moe_hidden);
    }
    if (rc) {
        if (full_iq_trace && m->rank == 0)
            fprintf(stderr, "K3_IQ_FULL_TRACE moe_forward_rc=%d\n", rc);
        if (m->profile.enabled)
            full_profile_phase_add(m, K3_FULL_PHASE_MOE,
                                   full_now() - phase_start);
        full_profile_layer_end(m, m->debug_layer_index, layer_start);
        return rc;
    }
    full_iq_trace_finite(m, "moe_output", m->moe_hidden, K3_HIDDEN);
    if (m->profile.enabled)
        full_profile_phase_add(m, K3_FULL_PHASE_MOE,
                               full_now() - phase_start);
    phase_start = m->profile.enabled ? full_now() : 0.0;
    full_profile_phase_begin(m, K3_FULL_PHASE_RESIDUAL);
    full_add_copy(m->hidden, m->tmp, m->moe_hidden, K3_HIDDEN);
    full_iq_trace_finite(m, "layer_output", m->hidden, K3_HIDDEN);
    if (m->profile.enabled)
        full_profile_phase_add(m, K3_FULL_PHASE_RESIDUAL,
                               full_now() - phase_start);
    full_profile_layer_end(m, m->debug_layer_index, layer_start);
    return 0;
}

static int full_synthetic_layer(k3_full_model *m, int layer, int position) {
    double ss = 0.0;
    for (int i = 0; i < K3_HIDDEN; ++i)
        ss += (double)m->hidden[i] * m->hidden[i];
    float inv = 1.0f / sqrtf((float)(ss / K3_HIDDEN) + K3_FULL_EPS);
    uint64_t salt = full_mix64(UINT64_C(0x4b33594e4e544845) +
                               (uint64_t)(unsigned)layer * UINT64_C(0x100000001b3) +
                               (uint64_t)(unsigned)position);

    /* The attention contribution is partitioned by hidden index so the
     * allreduce exercises the real TP join without multiplying the vector by
     * the number of physical ranks. */
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < K3_HIDDEN; ++i) {
        float x = m->hidden[i] * inv;
        m->tmp2[i] = (i % m->nodes == m->rank) ?
            (0.87f * x + 0.00001f * (float)((salt + (uint64_t)i) & 31)) : 0.0f;
    }
    if (full_sum(m, m->tmp2, K3_HIDDEN)) return EIO;
    ++m->synthetic_collectives;

    float logits[K3_EXPERTS];
    int route[K3_TOP_K];
    float weights[K3_TOP_K];
    float mean = 0.0f;
    for (int i = 0; i < K3_HIDDEN; ++i) mean += m->hidden[i];
    mean /= K3_HIDDEN;
    for (int e = 0; e < K3_EXPERTS; ++e) {
        uint64_t h = full_mix64(salt + (uint64_t)(unsigned)e * UINT64_C(0x9e3779b9));
        logits[e] = mean + (float)((h >> 40) & UINT64_C(0xffff)) / 65536.0f * 0.01f;
    }
    k3_router_topk(logits, NULL, K3_EXPERTS, K3_TOP_K, route, weights);
    memset(m->local_latent, 0, K3_LATENT * sizeof(float));
    for (int k = 0; k < K3_TOP_K; ++k) {
        if (route[k] % m->nodes != m->rank) continue;
        for (int i = 0; i < K3_LATENT; ++i)
            m->local_latent[i] += weights[k] *
                (0.001f + 0.00001f * (float)(route[k] % 17)) *
                m->hidden[i % K3_HIDDEN];
    }
    if (full_sum(m, m->local_latent, K3_LATENT)) return EIO;
    ++m->synthetic_collectives;
    for (int k = 0; k < K3_TOP_K; ++k)
        m->synthetic_route_hash ^= full_mix64((uint64_t)(unsigned)route[k] +
                                              (uint64_t)(unsigned)layer * 97U);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < K3_HIDDEN; ++i) {
        float routed = m->local_latent[i % K3_LATENT];
        m->hidden[i] = 0.96f * m->tmp2[i] + routed +
                       0.0001f * (float)((layer + i + position) % 19);
    }
    return 0;
}

static int full_synthetic_argmax(const k3_full_model *m, uint64_t seed,
                                 int position) {
    double sum = 0.0;
    for (int i = 0; i < K3_HIDDEN; ++i) sum += m->hidden[i] * (i + 1);
    uint64_t bits = 0;
    memcpy(&bits, &sum, sizeof sum);
    return (int)(full_mix64(seed ^ bits ^ (uint64_t)(unsigned)position) % K3_FULL_VOCAB);
}

static int full_debug_hidden_finite(const k3_full_model *m) {
    for (int i = 0; i < K3_HIDDEN; ++i)
        if (!isfinite(m->hidden[i])) return 0;
    return 1;
}

static int full_debug_forward_token(k3_full_model *m, const k3_full_options *o,
                                    int token, int position) {
    if (o->mode == K3_FULL_MODE_LAYER12) {
        full_debug_seed_hidden(m->tmp, o->input_seed, token, position);
        return full_debug_layer_forward(m, &m->debug_layer, position, 1);
    }
    full_debug_seed_hidden(m->hidden, o->input_seed, token, position);
    for (int layer = 0; layer < K3_LAYERS; ++layer) {
        if (layer == o->real_layer_index) {
            if (full_debug_layer_forward(m, &m->debug_layer, position, 0)) return EIO;
        } else if (full_synthetic_layer(m, layer, position)) {
            return EIO;
        }
    }
    double ss = 0.0;
    for (int i = 0; i < K3_HIDDEN; ++i) ss += (double)m->hidden[i] * m->hidden[i];
    float inv = 1.0f / sqrtf((float)(ss / K3_HIDDEN) + K3_FULL_EPS);
    for (int i = 0; i < K3_HIDDEN; ++i) m->hidden[i] *= inv;
    return 0;
}

/* Packed IQ tiles are persistent decode state, like repacked BF16 panels.
 * Materialize every locally owned expert before benchmark timing so routing a
 * previously unseen expert cannot turn one decode token into an allocation and
 * multi-megabyte conversion benchmark (and make peers wait in the reduction). */
static int full_debug_prepack_experts(k3_full_model *m) {
    if (!full_quant_packed || m->debug_layer_index == 0) return 0;
    k3_full_layer *l = &m->debug_layer;
    for (int e = 0; e < l->expert_count; ++e) {
        if (full_quant_pack_tensor(&l->experts[e].w1) ||
            full_quant_pack_tensor(&l->experts[e].w2) ||
            full_quant_pack_tensor(&l->experts[e].w3)) return ENOMEM;
    }
    return 0;
}

static int full_debug_prefill_chunk(k3_full_model *m,
                                    const k3_full_options *o,
                                    const int *tokens, int count,
                                    int position) {
    for (int i = 0; i < count; ++i)
        if (full_debug_forward_token(m, o, tokens[i], position + i)) return EIO;
    return 0;
}

static int full_debug_run(k3_full_options *o, tp_comm *comm, tp_comm *comm_col,
                          k3_pool *pool) {
    k3_full_model model;
    memset(&model, 0, sizeof model);
    model.rank = g_rank;
    model.nodes = g_nodes;
    model.threads = o->threads;
    model.max_seq = o->max_seq;
    model.comm = comm;
    model.comm_col = comm_col;
    model.debug_layer_index = o->real_layer_index;
    model.prefetch_mib = o->prefetch_mib;
    model.profile.enabled = o->profile;

    int rc = full_load_debug_model(&model, o, pool);
    int ready = rc == 0;
    float ready_sum = (float)ready;
    if (full_sum(&model, &ready_sum, 1)) return 3;
    if ((int)lrintf(ready_sum) != g_nodes) {
        if (g_rank == 0) fprintf(stderr, "k3_full_runner: debug load failed (%d/%d)\n",
                                  (int)lrintf(ready_sum), g_nodes);
        full_barrier();
        return 4;
    }
    rc = full_allocate_scratch(&model, o);
    ready = rc == 0;
    ready_sum = (float)ready;
    if (full_sum(&model, &ready_sum, 1)) return 3;
    if ((int)lrintf(ready_sum) != g_nodes) {
        if (g_rank == 0) fprintf(stderr, "k3_full_runner: debug scratch failed (%d/%d)\n",
                                  (int)lrintf(ready_sum), g_nodes);
        full_barrier();
        return 4;
    }

    double prepack_start = full_now();
    rc = full_debug_prepack_experts(&model);
    ready = rc == 0;
    ready_sum = (float)ready;
    if (full_sum(&model, &ready_sum, 1)) return 3;
    if ((int)lrintf(ready_sum) != g_nodes) {
        if (g_rank == 0)
            fprintf(stderr, "k3_full_runner: debug IQ prepack failed (%d/%d)\n",
                    (int)lrintf(ready_sum), g_nodes);
        full_barrier();
        return 4;
    }
    float prepack_seconds = (float)(full_now() - prepack_start);
    if (full_max(&model, &prepack_seconds, 1)) return 3;
    if (g_rank == 0 && prepack_seconds > 0.01f)
        printf("K3FULL_IQ_PREPACK seconds=%.6f local_experts=%d\n",
               prepack_seconds, model.debug_layer.expert_count);

    int generated_count = o->prefill_only ? 0 : o->new_tokens;
    int *prompt = (int *)malloc((size_t)o->prefill_tokens * sizeof *prompt);
    int *generated = generated_count ?
        (int *)malloc((size_t)generated_count * sizeof *generated) : NULL;
    ready = prompt && (!generated_count || generated);
    ready_sum = (float)ready;
    if (full_sum(&model, &ready_sum, 1)) return 3;
    if ((int)lrintf(ready_sum) != g_nodes) {
        if (g_rank == 0) fprintf(stderr, "k3_full_runner: debug buffers failed\n");
        free(prompt); free(generated); full_barrier(); return 4;
    }
    if (o->prompt_ids) {
        if (full_read_ids(o->prompt_ids, prompt, o->prefill_tokens) != o->prefill_tokens) {
            fprintf(stderr, "k3_full_runner rank %d: prompt-ID load failed\n", g_rank);
            free(prompt); free(generated); full_barrier(); return 4;
        }
    } else {
        for (int i = 0; i < o->prefill_tokens; ++i)
            prompt[i] = full_debug_token_id(o->input_seed, i);
    }

    full_profile_reset(&model);
    full_barrier();
    if (g_rank == 0 && o->prefill_chunk > 1)
        printf("K3FULL_PREFILL_PATH scalar-token-loop requested_chunk=%d effective_chunk=1\n",
               o->prefill_chunk);
    double prefill_start = full_now();
    int failed = 0;
    for (int begin = 0; begin < o->prefill_tokens && !failed; begin += o->prefill_chunk) {
        int end = begin + o->prefill_chunk;
        if (end > o->prefill_tokens) end = o->prefill_tokens;
        if (full_debug_prefill_chunk(&model, o, prompt + begin, end - begin,
                                     begin)) failed = 1;
    }
    float prefill_seconds = (float)(full_now() - prefill_start);
    if (full_max(&model, &prefill_seconds, 1)) failed = 1;
    int next = generated_count ?
        full_synthetic_argmax(&model, o->input_seed, o->prefill_tokens) : -1;
    double decode_start = full_now();
    for (int i = 0; i < generated_count && !failed; ++i) {
        generated[i] = next;
        if (full_debug_forward_token(&model, o, next, o->prefill_tokens + i)) {
            failed = 1;
            break;
        }
        next = full_synthetic_argmax(&model, o->input_seed,
                                     o->prefill_tokens + i + 1);
    }
    float decode_seconds = (float)(full_now() - decode_start);
    if (full_max(&model, &decode_seconds, 1)) failed = 1;
    float failure = failed ? 1.0f : 0.0f;
    if (!full_debug_hidden_finite(&model)) failure = 1.0f;
    if (full_max(&model, &failure, 1)) return 3;
    if (failure != 0.0f) {
        free(prompt); free(generated); full_barrier(); return 5;
    }
    if (full_profile_report(&model, o)) {
        free(prompt); free(generated); full_barrier(); return 6;
    }

    uint64_t output_hash = full_hash_ids(generated, generated_count);
    uint64_t hidden_hash = full_hash_f32(model.hidden, K3_HIDDEN);
    char rank_path[1200];
    snprintf(rank_path, sizeof rank_path, "%s.rank%03d", o->output_path, g_rank);
    FILE *rank_file = fopen(rank_path, "w");
    if (rank_file) {
        fprintf(rank_file, "rank=%d generated=%d hash=%016llx final=%d "
                "route_hash=%016llx collectives=%llu hidden_hash=%016llx\n",
                g_rank, generated_count,
                (unsigned long long)output_hash,
                generated_count ? generated[generated_count - 1] : -1,
                (unsigned long long)model.synthetic_route_hash,
                (unsigned long long)model.synthetic_collectives,
                (unsigned long long)hidden_hash);
        fclose(rank_file);
    }
    if (g_rank == 0) {
        FILE *out = fopen(o->output_path, "w");
        if (!out) {
            fprintf(stderr, "k3_full_runner: cannot write %s: %s\n",
                    o->output_path, strerror(errno));
            free(prompt); free(generated); full_barrier(); return 6;
        }
        fprintf(out, "K3FULLV2 status=PASS mode=%s nodes=%d real_layer_index=%d "
                "prefill_tokens=%d generated_tokens=%d prefill_chunk=%d\n",
                full_mode_name(o->mode), g_nodes, o->real_layer_index,
                o->prefill_tokens, generated_count, o->prefill_chunk);
        fprintf(out, "prefill_seconds=%.9f prefill_tok_s=%.6f decode_seconds=%.9f "
                "decode_tok_s=%.6f input_hash=%016llx output_hash=%016llx "
                "route_hash=%016llx collectives=%llu hidden_hash=%016llx\n",
                (double)prefill_seconds,
                prefill_seconds > 0.0f ? (double)o->prefill_tokens / prefill_seconds : 0.0,
                (double)decode_seconds,
                decode_seconds > 0.0f && generated_count ?
                    (double)generated_count / decode_seconds : 0.0,
                (unsigned long long)full_hash_ids(prompt, o->prefill_tokens),
                (unsigned long long)output_hash,
                (unsigned long long)model.synthetic_route_hash,
                (unsigned long long)model.synthetic_collectives,
                (unsigned long long)hidden_hash);
        fprintf(out, "generated_ids:");
        for (int i = 0; i < generated_count; ++i) fprintf(out, " %d", generated[i]);
        fputc('\n', out);
        fclose(out);
        printf("K3FULLV2 PASS mode=%s nodes=%d prefill=%d %.6f tok/s "
               "decode=%d %.6f tok/s output=%s hash=%016llx\n",
               full_mode_name(o->mode), g_nodes, o->prefill_tokens,
               prefill_seconds > 0.0f ? (double)o->prefill_tokens / prefill_seconds : 0.0,
               generated_count,
               decode_seconds > 0.0f && generated_count ?
                   (double)generated_count / decode_seconds : 0.0,
               o->output_path, (unsigned long long)output_hash);
    }
    if (full_mla_trace)
        for (int i = 0; i < 6; ++i)
            fprintf(stderr, "k3: mla-trace %-18s %016llx\n",
                    full_trace_name[i], (unsigned long long)full_trace_h[i]);
    full_barrier();
    free(prompt);
    free(generated);
    return 0;
}

int main(int argc, char **argv) {
    k3_full_options opt;
    int parsed = full_options(argc, argv, &opt);
    if (parsed) return parsed > 0 ? 0 : 2;
    full_bf16_rows_init();
    omp_set_dynamic(0);
    omp_set_num_threads(opt.threads);

    static uint8_t topology[K3_FULL_MAX_NODES][TOFU_NCOORDS];
    int topo_count = full_topology(opt.topo_path, topology);
    if (topo_count != opt.nodes) {
        fprintf(stderr, "k3_full_runner: topology has %d ranks, expected %d\n",
                topo_count, opt.nodes);
        return 2;
    }
    uint8_t mine[TOFU_NCOORDS] = {0};
    int rc = utofu_query_my_coords(mine);
    if (rc != UTOFU_SUCCESS) {
        fprintf(stderr, "k3_full_runner: query coordinates rc=%d\n", rc);
        return 3;
    }
    g_nodes = topo_count;
    for (int r = 0; r < g_nodes; ++r)
        if (!memcmp(topology[r], mine, TOFU_NCOORDS)) g_rank = r;
    if (g_rank < 0) {
        fprintf(stderr, "k3_full_runner: local coordinates are absent from topology\n");
        return 3;
    }

    k3_pool pool;
    k3_pool_init(&pool, "k3-full-runner");
    g_send_off = 0;
    g_slot_bar = DEMO_CACHE_LINE;
    g_bar_base = DEMO_CACHE_LINE;
    size_t region_bytes = g_bar_base + (size_t)(g_nodes + 1) * g_slot_bar;
    g_region = (char *)k3_pool_calloc(&pool, 1, region_bytes);
    if (!g_region) {
        fprintf(stderr, "%s\n", k3_pool_error(&pool));
        k3_pool_destroy(&pool);
        return 3;
    }
#if defined(__aarch64__)
    for (size_t off = 0; off < region_bytes; off += DEMO_CACHE_LINE)
        __asm__ __volatile__("dc civac, %0" :: "r"(g_region + off) : "memory");
    __asm__ __volatile__("dsb sy" ::: "memory");
#endif
    utofu_tni_id_t *tnis = NULL;
    size_t ntni = 0;
    rc = utofu_get_onesided_tnis(&tnis, &ntni);
    if (rc != UTOFU_SUCCESS || ntni < 1) {
        fprintf(stderr, "k3_full_runner rank %d: no one-sided TNI rc=%d count=%zu\n",
                g_rank, rc, ntni);
        k3_pool_destroy(&pool);
        return 3;
    }
    rc = utofu_create_vcq_with_cmp_id(tnis[0], DEMO_CMP_ID, 0, &g_vcq);
    if (rc != UTOFU_SUCCESS) {
        free(tnis);
        k3_pool_destroy(&pool);
        return 3;
    }
    utofu_vcq_id_t self;
    rc = utofu_query_vcq_id(g_vcq, &self);
    if (rc == UTOFU_SUCCESS)
        rc = utofu_reg_mem_with_stag(g_vcq, g_region, region_bytes,
                                     K3_FULL_STAG, 0, &g_base);
    if (rc != UTOFU_SUCCESS) {
        fprintf(stderr, "k3_full_runner rank %d: VCQ/barrier registration rc=%d\n",
                g_rank, rc);
        free(tnis);
        k3_pool_destroy(&pool);
        return 3;
    }
    for (int r = 0; r < g_nodes; ++r) {
        if (r == g_rank) {
            g_peer_vcq[r] = self;
            g_peer_base[r] = g_base;
            continue;
        }
        rc = utofu_construct_vcq_id(topology[r], tnis[0], DEMO_CQ_ID,
                                    DEMO_CMP_ID, &g_peer_vcq[r]);
        if (rc == UTOFU_SUCCESS) utofu_set_vcq_id_path(&g_peer_vcq[r], NULL);
        if (rc == UTOFU_SUCCESS)
            rc = utofu_query_stadd(g_peer_vcq[r], K3_FULL_STAG, &g_peer_base[r]);
        if (rc != UTOFU_SUCCESS) {
            fprintf(stderr, "k3_full_runner rank %d: peer %d bootstrap rc=%d\n",
                    g_rank, r, rc);
            free(tnis);
            k3_pool_destroy(&pool);
            return 3;
        }
    }
    free(tnis);
    full_barrier();

    if (opt.mode == K3_FULL_MODE_BARRIER) {
        full_barrier_stress(&opt);
        full_barrier();
        if (g_rank == 0) {
            printf("K3FULL_BARRIER PASS nodes=%d iterations=%d total=%d\n",
                   g_nodes, opt.barrier_iters, opt.barrier_iters + 2);
            fflush(stdout);
        }
        utofu_dereg_mem(g_vcq, g_base, 0);
        utofu_free_vcq(g_vcq);
        k3_pool_destroy(&pool);
        return 0;
    }

    tp_comm_config comm_config = {
        .use_bf16 = opt.comm_use_bf16,
        .robust = opt.comm_robust, .poll_spins = opt.comm_poll_spins,
        .a2a = opt.comm_a2a, .a2a_max = opt.comm_a2a_max, .ack = 0,
        .pipeline = full_env_int("K3_COMM_PIPELINE",
                                 full_env_int("K3_COMM_ASYNC_LATENT", 0)),
        .compact_bf16 = full_env_int("K3_COMM_COMPACT_BF16", 0),
        .defer_tcq = full_env_int("K3_COMM_DEFER_TCQ", 0),
        .defer_mrq = full_env_int("K3_COMM_DEFER_MRQ", 0),
        .deterministic = opt.comm_deterministic,
        .ack_retx = 64, .ack_rtt = 0.001, .timeout = 120.0,
    };
    g_async_latent = full_env_int("K3_COMM_ASYNC_LATENT", 0);
    g_half_col = full_env_int("K3_COMM_HALF_COL", 0);
    g_half_hidden = full_env_int("K3_COMM_HALF_HIDDEN", g_half_col);
    g_sparse_row = full_env_int("K3_COMM_SPARSE_ROW", 0);
    /* Sparse-row transport already requires sharded inputs.  Make that public
     * knob sufficient by itself instead of silently selecting the full-vector
     * path unless the older half-column implementation detail is also set. */
    if (g_sparse_row) {
        g_half_col = 1;
        g_half_hidden = 1;
    }
    int ar_groups = opt.ar_groups;
    if (!ar_groups)
        ar_groups = g_nodes == 96 ? 16 : (g_nodes == 12 ? 2 : 0);
    int hierarchical = ar_groups > 1 && ar_groups < g_nodes;
    int row_nodes = hierarchical ? g_nodes / ar_groups : g_nodes;
    int col_nodes = hierarchical ? ar_groups : 0;
    size_t row_comm_bytes = tp_comm_region_size(row_nodes, K3_FULL_REDUCE_COUNT,
                                                &comm_config);
    size_t col_comm_bytes = hierarchical ?
        tp_comm_region_size(col_nodes, K3_FULL_REDUCE_COUNT, &comm_config) : 0;
    void *row_comm_region = k3_pool_alloc(&pool, row_comm_bytes);
    void *col_comm_region = hierarchical ? k3_pool_alloc(&pool, col_comm_bytes) : NULL;
    tp_comm comm;
    tp_comm comm_col;
    memset(&comm, 0, sizeof comm);
    memset(&comm_col, 0, sizeof comm_col);
    if (!row_comm_region || (hierarchical && !col_comm_region)) {
        fprintf(stderr, "k3_full_runner rank %d: comm region allocation failed\n", g_rank);
        return 3;
    }
    if (hierarchical) {
        rc = tp_comm_init_2d_external(
            &comm, &comm_col, g_vcq, g_peer_vcq, g_rank, g_nodes, ar_groups,
            K3_FULL_REDUCE_COUNT, full_barrier, &comm_config,
            row_comm_region, row_comm_bytes, col_comm_region, col_comm_bytes);
    } else {
        rc = tp_comm_init_external(&comm, g_vcq, g_peer_vcq, g_rank, g_nodes,
                                   K3_FULL_REDUCE_COUNT, full_barrier,
                                   &comm_config, row_comm_region, row_comm_bytes);
    }
    if (rc) {
        fprintf(stderr, "k3_full_runner rank %d: comm init rc=%d\n", g_rank, rc);
        return 3;
    }
    full_barrier();

    if (opt.mode != K3_FULL_MODE_FULL96) {
        rc = full_debug_run(&opt, &comm, hierarchical ? &comm_col : NULL, &pool);
        if (hierarchical) tp_comm_free_2d(&comm, &comm_col);
        else tp_comm_free(&comm);
        utofu_dereg_mem(g_vcq, g_base, 0);
        utofu_free_vcq(g_vcq);
        k3_pool_destroy(&pool);
        return rc;
    }

    k3_full_model model;
    memset(&model, 0, sizeof model);
    model.rank = g_rank;
    model.nodes = g_nodes;
    model.threads = opt.threads;
    model.max_seq = opt.max_seq;
    model.comm = &comm;
    model.comm_col = hierarchical ? &comm_col : NULL;
    model.debug_layer_index = -1;
    model.prefetch_mib = opt.prefetch_mib;
    model.profile.enabled = opt.profile;
    rc = full_load_model(&model, &opt, &pool);
    int ready = rc == 0;
    float ready_sum = (float)ready;
    if (full_sum(&model, &ready_sum, 1)) {
        fprintf(stderr, "k3_full_runner rank %d: weight-readiness collective failed\n", g_rank);
        return 3;
    }
    if ((int)lrintf(ready_sum) != g_nodes) {
        if (g_rank == 0)
            fprintf(stderr, "k3_full_runner: staged weight load failed (%d/%d ranks)\n",
                    (int)lrintf(ready_sum), g_nodes);
        full_barrier();
        return 4;
    }
    rc = full_allocate_scratch(&model, &opt);
    ready = rc == 0;
    ready_sum = (float)ready;
    if (full_sum(&model, &ready_sum, 1)) return 3;
    if ((int)lrintf(ready_sum) != g_nodes) {
        if (g_rank == 0)
            fprintf(stderr, "k3_full_runner: scratch allocation failed (%d/%d ranks)\n",
                    (int)lrintf(ready_sum), g_nodes);
        full_barrier();
        return 4;
    }

    int *prompt = (int *)malloc((size_t)opt.prefill_tokens * sizeof(int));
    ready = prompt && full_read_ids(opt.prompt_ids, prompt, opt.prefill_tokens) == opt.prefill_tokens;
    ready_sum = (float)ready;
    if (full_sum(&model, &ready_sum, 1)) return 3;
    if ((int)lrintf(ready_sum) != g_nodes) {
        if (g_rank == 0) fprintf(stderr, "k3_full_runner: prompt-ID load failed\n");
        free(prompt);
        full_barrier();
        return 4;
    }

    int *generated = opt.new_tokens ?
        (int *)malloc((size_t)opt.new_tokens * sizeof(int)) : NULL;
    ready = !opt.new_tokens || generated;
    ready_sum = (float)ready;
    if (full_sum(&model, &ready_sum, 1)) return 3;
    if ((int)lrintf(ready_sum) != g_nodes) {
        if (g_rank == 0) fprintf(stderr, "k3_full_runner: output-ID allocation failed\n");
        free(prompt);
        free(generated);
        full_barrier();
        return 4;
    }
    int warm_threads = 1;
#pragma omp parallel
    {
#pragma omp master
        warm_threads = omp_get_num_threads();
    }
    if (warm_threads != opt.threads && g_rank == 0)
        fprintf(stderr, "k3_full_runner: requested %d OpenMP workers, got %d\n",
                opt.threads, warm_threads);

    full_profile_reset(&model);
    full_barrier();
    if (g_rank == 0 && opt.prefill_chunk > 1)
        printf("K3FULL_PREFILL_PATH scalar-token-loop requested_chunk=%d effective_chunk=1\n",
               opt.prefill_chunk);
    double prefill_start = full_now();
    int failed = 0;
    for (int begin = 0; begin < opt.prefill_tokens && !failed;
         begin += opt.prefill_chunk) {
        int end = begin + opt.prefill_chunk;
        if (end > opt.prefill_tokens) end = opt.prefill_tokens;
        if (full_forward_prefill_chunk_real(&model, prompt + begin,
                                            end - begin, begin)) failed = 1;
    }
    double prefill_seconds = full_now() - prefill_start;
    float prefill_max = (float)prefill_seconds;
    if (full_max(&model, &prefill_max, 1)) failed = 1;
    if (failed) {
        fprintf(stderr, "k3_full_runner rank %d: prefill failed\n", g_rank);
        free(prompt); free(generated); return 5;
    }
    float next_logit = 0.0f;
    int next = -1;
    double decode_seconds = 0.0;
    if (!opt.prefill_only && opt.new_tokens > 0) {
        double decode_start = full_now();
        next = full_argmax(&model, &next_logit);
        if (next < 0) failed = 1;
        for (int i = 0; i < opt.new_tokens && !failed; ++i) {
            generated[i] = next;
            if (full_forward_token(&model, next, opt.prefill_tokens + i)) {
                failed = 1;
                break;
            }
            if (i + 1 < opt.new_tokens) {
                next = full_argmax(&model, &next_logit);
                if (next < 0) failed = 1;
            }
        }
        decode_seconds = full_now() - decode_start;
    }
    float decode_max = (float)decode_seconds;
    if (full_max(&model, &decode_max, 1)) failed = 1;
    float failure = failed ? 1.0f : 0.0f;
    if (full_max(&model, &failure, 1)) return 3;
    if (failure != 0.0f) {
        if (g_rank == 0) fprintf(stderr, "k3_full_runner: distributed inference failed\n");
        free(prompt); free(generated); return 5;
    }
    if (full_profile_report(&model, &opt)) {
        free(prompt); free(generated); return 6;
    }

    uint64_t output_hash = full_hash_ids(generated, opt.new_tokens);
    char rank_path[1200];
    snprintf(rank_path, sizeof rank_path, "%s.rank%03d", opt.output_path, g_rank);
    FILE *rank_file = fopen(rank_path, "w");
    if (rank_file) {
        fprintf(rank_file, "rank=%d generated=%d hash=%016llx final=%d hidden_hash=%016llx\n",
                g_rank, opt.new_tokens, (unsigned long long)output_hash,
                opt.new_tokens ? generated[opt.new_tokens - 1] : -1,
                (unsigned long long)full_hash_f32(model.hidden, K3_HIDDEN));
        fclose(rank_file);
    }
    if (g_rank == 0) {
        FILE *out = fopen(opt.output_path, "w");
        if (!out) {
            fprintf(stderr, "k3_full_runner: cannot write %s: %s\n",
                    opt.output_path, strerror(errno));
            free(prompt); free(generated); return 6;
        }
        fprintf(out,
                "K3FULLV2 status=PASS mode=%s nodes=%d real_layer_index=-1 "
                "prefill_tokens=%d generated_tokens=%d prefill_chunk=%d "
                "comm_deterministic=%d comm_bf16=%d comm_robust=%d "
                "comm_poll_spins=%d comm_a2a=%d comm_a2a_max=%d "
                "comm_pipeline=%d compact_bf16=%d defer_tcq=%d defer_mrq=%d "
                "half_col=%d half_hidden=%d sparse_row=%d async_latent=%d "
                "prefetch_mib=%d ar_groups=%d profile=%d\n",
                full_mode_name(opt.mode), g_nodes, opt.prefill_tokens,
                opt.prefill_only ? 0 : opt.new_tokens,
                opt.prefill_chunk, opt.comm_deterministic, opt.comm_use_bf16,
                opt.comm_robust, opt.comm_poll_spins, opt.comm_a2a,
                opt.comm_a2a_max, comm_config.pipeline, comm_config.compact_bf16,
                comm_config.defer_tcq, comm_config.defer_mrq, g_half_col, g_half_hidden,
                g_sparse_row, g_async_latent,
                opt.prefetch_mib, ar_groups, opt.profile);
        fprintf(out, "prefill_seconds=%.9f prefill_tok_s=%.6f decode_seconds=%.9f decode_tok_s=%.6f\n",
                (double)prefill_max,
                prefill_max > 0.0f ? (double)opt.prefill_tokens / prefill_max : 0.0,
                (double)decode_max,
                decode_max > 0.0f && !opt.prefill_only ? (double)opt.new_tokens / decode_max : 0.0);
        fprintf(out, "prompt_hash=%016llx output_hash=%016llx next_logit=%+.9e\n",
                (unsigned long long)full_hash_ids(prompt, opt.prefill_tokens),
                (unsigned long long)output_hash, (double)next_logit);
        fprintf(out, "generated_ids:");
        for (int i = 0; i < opt.new_tokens && !opt.prefill_only; ++i)
            fprintf(out, " %d", generated[i]);
        fputc('\n', out);
        fclose(out);
        printf("K3FULLV2 PASS mode=%s nodes=%d prefill=%d %.6f tok/s "
               "decode=%d %.6f tok/s output=%s hash=%016llx\n",
               full_mode_name(opt.mode), g_nodes, opt.prefill_tokens,
               prefill_max > 0.0f ? (double)opt.prefill_tokens / prefill_max : 0.0,
               opt.prefill_only ? 0 : opt.new_tokens,
               decode_max > 0.0f && !opt.prefill_only ? (double)opt.new_tokens / decode_max : 0.0,
               opt.output_path, (unsigned long long)output_hash);
    }
    full_barrier();
    free(prompt);
    free(generated);
    if (hierarchical) tp_comm_free_2d(&comm, &comm_col);
    else tp_comm_free(&comm);
    utofu_dereg_mem(g_vcq, g_base, 0);
    utofu_free_vcq(g_vcq);
    k3_pool_destroy(&pool);
    return 0;
}
