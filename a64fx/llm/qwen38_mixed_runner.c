/* Persistent Qwen3.8 service experiment: PP3xTP4 prefill of three requests,
 * in-memory state transpose, then three concurrent TP4 decode replicas.
 * The complete TP4 rank stage is loaded once and remains anonymous/resident. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <mpi.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <utofu.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define GGUF_LOADER_IMPLEMENTATION
#include "../../common/gguf_loader.h"
#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"
#define TRANSFORMER_IMPLEMENTATION
#include "../../common/transformer.h"
#include "../utofu-tests/tofu_demo.h"
#include "../utofu-tests/tp_allreduce.h"

typedef struct {
    MPI_Comm comm;
    tp_comm *tofu;
    double reduce_s;
    long reduce_calls;
} reduce_ctx;
static MPI_Comm g_init_comm = MPI_COMM_NULL;
static void init_barrier(void) { MPI_Barrier(g_init_comm); }

static double wall(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static int env_i(const char *name, int def) {
    const char *v = getenv(name);
    return v && *v ? atoi(v) : def;
}
static const char *env_s(const char *name, const char *def) {
    const char *v = getenv(name);
    return v && *v ? v : def;
}
static void fail(int rank, const char *msg) {
    fprintf(stderr, "qwen38-mixed rank=%d FATAL %s\n", rank, msg);
    MPI_Abort(MPI_COMM_WORLD, 2);
}
static void reduce_sum(float *buf, int count, void *opaque) {
    reduce_ctx *c = (reduce_ctx *)opaque;
    double t0 = wall();
    if (c->tofu) tp_allreduce_sum(c->tofu, buf, count);
    else MPI_Allreduce(MPI_IN_PLACE, buf, count, MPI_FLOAT, MPI_SUM, c->comm);
    c->reduce_s += wall() - t0;
    c->reduce_calls++;
}
static int read_topology(uint8_t coords[][TOFU_NCOORDS], int cap) {
    FILE *f = fopen(TOPO_PATH, "r");
    if (!f) return -1;
    int n = 0;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        unsigned rank, c[TOFU_NCOORDS];
        if (line[0] == '#' || line[0] == '\n') continue;
        if (n >= cap || sscanf(line, "%u %u %u %u %u %u %u", &rank,
                &c[0], &c[1], &c[2], &c[3], &c[4], &c[5]) != 7 ||
            rank != (unsigned)n) { fclose(f); return -1; }
        for (int k = 0; k < TOFU_NCOORDS; k++) coords[n][k] = (uint8_t)c[k];
        n++;
    }
    fclose(f);
    return n;
}
static void init_decode_utofu(int wrank, int group, int tp_rank,
                              int compact, MPI_Comm group_comm, tp_comm *out,
                              utofu_vcq_hdl_t *vcq) {
    uint8_t topo[12][TOFU_NCOORDS];
    if (read_topology(topo, 12) != 12) fail(wrank, "read tofu_topo.txt");
    utofu_tni_id_t *tnis = NULL;
    size_t ntnis = 0;
    if (utofu_get_onesided_tnis(&tnis, &ntnis) != UTOFU_SUCCESS || ntnis < 1)
        fail(wrank, "get uTofu TNI");
    utofu_tni_id_t tni = tnis[0];
    free(tnis);
    if (utofu_create_vcq_with_cmp_id(tni, DEMO_CMP_ID, 0, vcq) != UTOFU_SUCCESS)
        fail(wrank, "create decode VCQ");
    utofu_vcq_id_t peers[4], mine;
    if (utofu_query_vcq_id(*vcq, &mine) != UTOFU_SUCCESS)
        fail(wrank, "query decode VCQ");
    for (int r = 0; r < 4; r++) {
        int peer_pp = compact && r >= 2 ? (group + 1) % 3 : group;
        int peer_world = peer_pp * 4 + r;
        if (r == tp_rank) peers[r] = mine;
        else if (utofu_construct_vcq_id(topo[peer_world], tni, DEMO_CQ_ID,
                                        DEMO_CMP_ID, &peers[r]) != UTOFU_SUCCESS)
            fail(wrank, "construct decode peer VCQ");
        utofu_set_vcq_id_path(&peers[r], NULL);
    }
    g_init_comm = group_comm;
    if (tp_comm_init(out, *vcq, peers, tp_rank, 4, 5120, init_barrier) != 0)
        fail(wrank, "initialize decode uTofu TP4");
}
static int local_argmax(transformer_model *m, const float *logits, MPI_Comm comm) {
    struct { float value; int index; } local = {-INFINITY, 0}, global;
    for (int i = 0; i < m->output.n_rows; i++) {
        if (logits[i] > local.value) {
            local.value = logits[i];
            local.index = m->tp_vocab_lo + i;
        }
    }
    MPI_Allreduce(&local, &global, 1, MPI_FLOAT_INT, MPI_MAXLOC, comm);
    return global.index;
}
static uint64_t hash_token(uint64_t h, int32_t token) {
    h ^= (uint32_t)token;
    h *= UINT64_C(1099511628211);
    return h;
}
static size_t flush_ssm_subnormals(transformer_model *m) {
    size_t flushed = 0;
    for (int l = 0; l < m->n_layers; l++) {
        if (m->conv_state && m->conv_state[l]) {
            size_t n = (size_t)(m->ssm_conv_kernel - 1) * m->ssm_qkv_dim;
            for (size_t i = 0; i < n; i++)
                if (m->conv_state[l][i] != 0.0f &&
                    fabsf(m->conv_state[l][i]) < FLT_MIN) {
                    m->conv_state[l][i] = 0.0f;
                    flushed++;
                }
        }
        if (m->recurrent_state && m->recurrent_state[l]) {
            size_t n = (size_t)m->ssm_dt_rank * m->ssm_d_state * m->ssm_d_state;
            for (size_t i = 0; i < n; i++)
                if (m->recurrent_state[l][i] != 0.0f &&
                    fabsf(m->recurrent_state[l][i]) < FLT_MIN) {
                    m->recurrent_state[l][i] = 0.0f;
                    flushed++;
                }
        }
    }
    return flushed;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int wrank = 0, world = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world != 12 || argc != 2) {
        if (!wrank) fprintf(stderr, "usage: mpiexec -np 12 %s MODEL.gguf\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    const int tp_size = 4, pp_size = 3, requests = 3;
    const int tp_rank = wrank % tp_size, pp_rank = wrank / tp_size;
    const int compact_groups = env_i("Q38_MIXED_COMPACT_GROUPS", 0) != 0;
    /* Compact a/c-plane decode rectangles while keeping the resident TP lane:
     * g0={0,1,6,7}, g1={4,5,10,11}, g2={8,9,2,3}. */
    const int decode_group = compact_groups
        ? (pp_rank + (tp_rank >= 2 ? 2 : 0)) % 3 : pp_rank;
    char log_path[64];
    snprintf(log_path, sizeof(log_path), "q38_mixed_rank%02d.txt", wrank);
    FILE *rank_log = fopen(log_path, "w");
    if (!rank_log) fail(wrank, "open rank log");
    setvbuf(rank_log, NULL, _IOLBF, 0);
    int cuts[4] = {0, env_i("Q38_MIXED_CUT1", 21),
                   env_i("Q38_MIXED_CUT2", 43), 64};
    int l0 = cuts[pp_rank], l1 = cuts[pp_rank + 1];
    int ntok = env_i("Q38_MIXED_PREFILL_TOKENS", 4096);
    int chunk = env_i("Q38_MIXED_CHUNK", 256);
    int maxgen = env_i("Q38_MIXED_MAXGEN", 64);
    int active_replicas = env_i("Q38_MIXED_ACTIVE_REPLICAS", 3);
    int threads = env_i("LLM_THREADS", 48);
    int maxseq = env_i("Q38_MIXED_MAXSEQ", ntok + maxgen + 16);
    if (ntok < 1 || chunk < 1 || maxgen < 1 || active_replicas < 1 ||
        active_replicas > 3 || maxseq < ntok + maxgen)
        fail(wrank, "invalid token/chunk/sequence settings");

    MPI_Comm prefill_tp, pipeline, decode_tp;
    MPI_Comm_split(MPI_COMM_WORLD, pp_rank, tp_rank, &prefill_tp);
    MPI_Comm_split(MPI_COMM_WORLD, tp_rank, pp_rank, &pipeline);
    MPI_Comm_split(MPI_COMM_WORLD, decode_group, tp_rank, &decode_tp);
    reduce_ctx rc = {prefill_tp, NULL, 0.0, 0};

    gguf_context *g = gguf_open_multi(argv[1], 3);
    if (!g) fail(wrank, "open model metadata");
    transformer_model *m = transformer_load(g, maxseq);
    if (!m) fail(wrank, "load model metadata");
    transformer_set_threads(m, threads);
    const char *stage = env_s("TP_STAGE_DIR", "/local/u14346/qwen38-q8-tp4");
    setenv("TP_STAGE_DIR", stage, 1);
    if (transformer_tp_slice_weights(m, tp_rank, tp_size, 1)) fail(wrank, "TP slice");
    size_t stage_bytes = transformer_tp_load_stage(m, stage, tp_rank, tp_size);
    if (!stage_bytes) fail(wrank, "load resident TP4 stage");
    transformer_resize_kv_for_tp(m, 0, m->n_layers,
                                 m->tp_kv_head_count * m->head_dim);
    transformer_set_tp(m, tp_rank, tp_size, reduce_sum, &rc);
    size_t prefill_bytes = transformer_prepack_q8v2_range(m, l0, l1);
    if (!prefill_bytes) fail(wrank, "build transient Q8v2 prefill panels");
    fprintf(rank_log, "load stage=%.3fGB prefill_pack=%.3fGB layers=%d:%d\n",
            stage_bytes / 1e9, prefill_bytes / 1e9, l0, l1);

    int32_t *tokens = (int32_t *)malloc((size_t)ntok * sizeof(*tokens));
    float *hidden = NULL;
    if (!tokens || posix_memalign((void **)&hidden, 256,
            (size_t)chunk * m->n_embd * sizeof(float))) fail(wrank, "prefill buffers");
    void *states[requests];
    size_t state_sizes[requests];
    int next[requests];
    memset(states, 0, sizeof(states));

    MPI_Barrier(MPI_COMM_WORLD);
    double prefill_begin = wall();
    for (int req = 0; req < requests; req++) {
        transformer_reset_runtime_state(m);
        int token_id = env_i(req == 0 ? "Q38_MIXED_TOKEN0" :
                             req == 1 ? "Q38_MIXED_TOKEN1" : "Q38_MIXED_TOKEN2",
                             req + 1);
        for (int i = 0; i < ntok; i++) tokens[i] = token_id;
        float *last_logits = NULL;
        int nchunk = (ntok + chunk - 1) / chunk;
        for (int tick = 0; tick < nchunk + pp_size - 1; tick++) {
            int ci = tick - pp_rank;
            if (ci < 0 || ci >= nchunk) continue;
            int p0 = ci * chunk, n = ntok - p0;
            if (n > chunk) n = chunk;
            int nf = n * m->n_embd;
            if (pp_rank > 0)
                MPI_Recv(hidden, nf, MPI_FLOAT, pp_rank - 1, ci, pipeline,
                         MPI_STATUS_IGNORE);
            unsigned flags = pp_rank == 0 ? TF_PREFILL_EMBED : 0;
            if (pp_rank == pp_size - 1 && ci == nchunk - 1)
                flags |= TF_PREFILL_LOGITS;
            last_logits = transformer_prefill_range(m, tokens + p0, hidden, n, p0,
                                                     l0, l1, flags);
            if (!last_logits) fail(wrank, "range prefill");
            if (pp_rank < pp_size - 1)
                MPI_Send(hidden, nf, MPI_FLOAT, pp_rank + 1, ci, pipeline);
        }
        int nnext = -1;
        if (pp_rank == pp_size - 1)
            nnext = local_argmax(m, last_logits, prefill_tp);
        MPI_Bcast(&nnext, 1, MPI_INT, 8, MPI_COMM_WORLD);
        next[req] = nnext;
        state_sizes[req] = transformer_runtime_state_size(m, l0, l1, ntok);
        states[req] = malloc(state_sizes[req]);
        if (!states[req] || transformer_runtime_state_pack(
                m, l0, l1, ntok, states[req], state_sizes[req]))
            fail(wrank, "pack in-memory prefill state");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double prefill_s = wall() - prefill_begin;
    fprintf(rank_log, "prefill requests=3 tokens=%d time=%.6f\n", ntok, prefill_s);

    /* On each same-TP-lane pipeline communicator, source PP stage s sends its
     * layer-range state for request d to destination decode group d. */
    int sendcounts[3], recvcounts[3], sdisp[3], rdisp[3];
    int send_total = 0, recv_total = 0;
    for (int d = 0; d < requests; d++) {
        int request_for_dest = compact_groups
            ? (d + (tp_rank >= 2 ? 2 : 0)) % 3 : d;
        if (state_sizes[request_for_dest] > INT32_MAX)
            fail(wrank, "state blob exceeds MPI int count");
        sendcounts[d] = (int)state_sizes[request_for_dest];
        sdisp[d] = send_total;
        send_total += sendcounts[d];
    }
    MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, pipeline);
    for (int s = 0; s < pp_size; s++) { rdisp[s] = recv_total; recv_total += recvcounts[s]; }
    uint8_t *sendbuf = (uint8_t *)malloc((size_t)send_total);
    uint8_t *recvbuf = (uint8_t *)malloc((size_t)recv_total);
    if (!sendbuf || !recvbuf) fail(wrank, "state transpose buffers");
    for (int d = 0; d < requests; d++) {
        int request_for_dest = compact_groups
            ? (d + (tp_rank >= 2 ? 2 : 0)) % 3 : d;
        memcpy(sendbuf + sdisp[d], states[request_for_dest],
               state_sizes[request_for_dest]);
    }
    transformer_reset_runtime_state(m);
    double transpose_begin = wall();
    MPI_Alltoallv(sendbuf, sendcounts, sdisp, MPI_BYTE,
                  recvbuf, recvcounts, rdisp, MPI_BYTE, pipeline);
    for (int s = 0; s < pp_size; s++) {
        int restored_pos = -1;
        if (transformer_runtime_state_unpack(m, recvbuf + rdisp[s],
                                             (size_t)recvcounts[s], &restored_pos) ||
            restored_pos != ntok) fail(wrank, "unpack transposed state");
    }
    size_t state_subnormals = flush_ssm_subnormals(m);
    if (env_i("Q38_MIXED_DROP_STATE", 0)) transformer_reset_runtime_state(m);
    MPI_Barrier(MPI_COMM_WORLD);
    double transpose_s = wall() - transpose_begin;
    fprintf(rank_log, "transpose recv=%.3fMB time=%.6f ssm_subnormals=%zu\n",
            recv_total / 1e6, transpose_s, state_subnormals);
    for (int r = 0; r < requests; r++) free(states[r]);
    free(sendbuf); free(recvbuf); free(hidden); free(tokens);

    size_t released = transformer_release_q8v2_range(m, l0, l1);
    /* Batched prefill temporarily owns the OpenMP team and cycles the decode
     * pthread pool.  Recreate it at the service topology boundary so decode
     * starts from the same pinned-worker state as the standalone TP runner. */
    transformer_set_threads(m, 1);
    transformer_set_threads(m, threads);
    rc.comm = decode_tp;
    tp_comm decode_utofu;
    utofu_vcq_hdl_t decode_vcq;
    init_decode_utofu(wrank, decode_group, tp_rank, compact_groups, decode_tp,
                      &decode_utofu, &decode_vcq);
    rc.tofu = &decode_utofu;
    rc.reduce_s = 0.0;
    rc.reduce_calls = 0;
    transformer_set_tp(m, tp_rank, tp_size, reduce_sum, &rc);
    MPI_Barrier(MPI_COMM_WORLD);
    double expand_begin = wall();
    size_t expanded = transformer_expand_q8_bf16_pv_range(m, 0, m->n_layers, 1, 0);
    if (!expanded) fail(wrank, "resident Q8 to BF16-PV decode expansion");
    MPI_Barrier(MPI_COMM_WORLD);
    double expand_s = wall() - expand_begin;
    fprintf(rank_log, "transition released=%.3fGB expanded=%.3fGB time=%.6f\n",
            released / 1e9, expanded / 1e9, expand_s);

    int in_tok = next[decode_group];
    uint64_t token_hash = UINT64_C(1469598103934665603);
    MPI_Barrier(MPI_COMM_WORLD);
    double decode_begin = wall();
    double forward_s = 0.0, logits_s = 0.0;
    int replica_gen = decode_group < active_replicas ? maxgen : 0;
    for (int i = 0; i < replica_gen; i++) {
        token_hash = hash_token(token_hash, in_tok);
        transformer_embed_token(m, in_tok);
        double phase = wall();
        if (!transformer_forward_partial(m, ntok + i, 0, m->n_layers))
            fail(wrank, "decode forward");
        forward_s += wall() - phase;
        phase = wall();
        float *logits = transformer_compute_logits(m);
        if (!logits) fail(wrank, "decode logits");
        in_tok = local_argmax(m, logits, decode_tp);
        logits_s += wall() - phase;
    }
    MPI_Barrier(decode_tp);
    double decode_s = wall() - decode_begin;
    MPI_Barrier(MPI_COMM_WORLD);
    fprintf(rank_log, "decode replica=%d tokens=%d time=%.6f tok/s=%.2f "
            "forward=%.6f logits=%.6f reduce=%.6f calls=%ld hash=%016llx\n",
            decode_group, replica_gen, decode_s,
            replica_gen > 0 ? replica_gen / decode_s : 0.0,
            forward_s, logits_s,
            rc.reduce_s, rc.reduce_calls,
            (unsigned long long)token_hash);
    double decode_max = 0.0;
    MPI_Reduce(&decode_s, &decode_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (tp_rank == 0) {
        printf("qwen38-mixed replica=%d prefill_next=%d decode=%d time=%.6f tok/s=%.2f hash=%016llx\n",
               decode_group, next[decode_group], maxgen, decode_s, maxgen / decode_s,
               (unsigned long long)token_hash);
        fflush(stdout);
    }
    if (!wrank) {
        printf("qwen38-mixed PP3xTP4->3xTP4 prefill=%d/request time=%.6f aggregate=%.2f tok/s "
               "transpose=%.6f expand=%.6f decode_aggregate=%.2f tok/s "
               "stage=%.3fGB prefill_pack=%.3fGB released=%.3fGB expanded=%.3fGB\n",
               ntok, prefill_s, (double)(requests * ntok) / prefill_s,
               transpose_s, expand_s, (double)(requests * maxgen) / decode_max,
               stage_bytes / 1e9, prefill_bytes / 1e9, released / 1e9, expanded / 1e9);
        fflush(stdout);
    }

    transformer_free(m);
    gguf_close(g);
    tp_comm_free(&decode_utofu);
    utofu_free_vcq(decode_vcq);
    fclose(rank_log);
    MPI_Comm_free(&decode_tp);
    MPI_Comm_free(&pipeline);
    MPI_Comm_free(&prefill_tp);
    MPI_Finalize();
    return 0;
}
