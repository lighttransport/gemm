/* Qwen3.8-27B single-sequence prefill.  The primary configurations are
 * eight-node PP2xTP4 and twelve-node PP3xTP4. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <mpi.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
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
#include "../utofu-tests/tp_rsag4.h"

typedef enum { TP_COMM_MPI, TP_COMM_UTOFU_TREE, TP_COMM_UTOFU_RSAG4 } tp_comm_kind;
typedef struct {
    tp_comm_kind kind;
    MPI_Comm mpi;
    tp_comm tofu;
    tp_rsag4 rsag4;
    utofu_vcq_hdl_t vcq[TP_RSAG4_MAX_TNI];
    int nvcq;
} tp_reduce_ctx;
static MPI_Comm g_tp_init_barrier = MPI_COMM_NULL;
static void tp_init_barrier(void) { MPI_Barrier(g_tp_init_barrier); }
static void tp_sum(float *buf, int count, void *opaque) {
    tp_reduce_ctx *c = (tp_reduce_ctx *)opaque;
    if (c->kind == TP_COMM_UTOFU_TREE) tp_allreduce_sum(&c->tofu, buf, count);
    else if (c->kind == TP_COMM_UTOFU_RSAG4) {
        if (tp_rsag4_sum(&c->rsag4, buf, count)) {
            fprintf(stderr, "qwen38-prefill FATAL uTofu RSAG4 failure\n");
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
    }
    else MPI_Allreduce(MPI_IN_PLACE, buf, count, MPI_FLOAT, MPI_SUM, c->mpi);
}
static double wall(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static int env_i(const char *name, int def) {
    const char *v = getenv(name); return v && *v ? atoi(v) : def;
}
static const char *env_s(const char *name, const char *def) {
    const char *v = getenv(name); return v && *v ? v : def;
}
static void f32_to_bf16(uint16_t *dst, const float *src, size_t n, int threads) {
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(static)
#endif
    for (size_t i = 0; i < n; i++) {
        uint32_t u;
        memcpy(&u, src + i, sizeof(u));
        dst[i] = (uint16_t)(u >> 16);
    }
}
static void bf16_to_f32(float *dst, const uint16_t *src, size_t n, int threads) {
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(static)
#endif
    for (size_t i = 0; i < n; i++) {
        uint32_t u = (uint32_t)src[i] << 16;
        memcpy(dst + i, &u, sizeof(u));
    }
}
static void fail(int rank, const char *what) {
    fprintf(stderr, "qwen38-prefill rank=%d FATAL %s\n", rank, what);
    char path[64]; snprintf(path, sizeof(path), "q38_prefill_error_rank%02d.txt", rank);
    FILE *f = fopen(path, "w");
    if (f) { fprintf(f, "rank=%d FATAL %s\n", rank, what); fclose(f); }
    MPI_Abort(MPI_COMM_WORLD, 2);
}

static int read_topology(uint8_t coords[][TOFU_NCOORDS], int cap) {
    const char *path = getenv("TOFU_TOPO_PATH");
    if (!path || !*path) path = TOPO_PATH;
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    int n = 0;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        unsigned rank, c[TOFU_NCOORDS];
        if (n >= cap || sscanf(line, "%u %u %u %u %u %u %u", &rank,
                &c[0], &c[1], &c[2], &c[3], &c[4], &c[5]) != 7 ||
                rank != (unsigned)n) {
            fclose(f); return -1;
        }
        for (int k = 0; k < TOFU_NCOORDS; k++) coords[n][k] = (uint8_t)c[k];
        n++;
    }
    fclose(f);
    return n;
}

static void init_utofu_tp(tp_reduce_ctx *ctx, int wrank, int world,
        int pp_rank, int tp_size, int max_count, MPI_Comm tp_mpi, int rsag4) {
    uint8_t topo[32][TOFU_NCOORDS];
    if (world > 32 || read_topology(topo, 32) != world)
        fail(wrank, "read tofu_topo.txt (run tofu_topo_helper first)");
    utofu_tni_id_t *tnis = NULL;
    size_t ntnis = 0;
    int rc = utofu_get_onesided_tnis(&tnis, &ntnis);
    if (rc != UTOFU_SUCCESS || ntnis < 1) fail(wrank, "get uTofu TNI");
    utofu_tni_id_t tni = tnis[0];
    int use_tni = rsag4 ? (ntnis < 3 ? (int)ntnis : 3) : 1;
    utofu_vcq_hdl_t vcqs[TP_RSAG4_MAX_TNI];
    utofu_vcq_id_t peers[TP_RSAG4_MAX_TNI][4];
    for (int k = 0; k < use_tni; k++) {
        tni = tnis[k];
        rc = utofu_create_vcq_with_cmp_id(tni, DEMO_CMP_ID, 0, &vcqs[k]);
        if (rc != UTOFU_SUCCESS) fail(wrank, "create uTofu VCQ");
        utofu_vcq_id_t mine;
        if (utofu_query_vcq_id(vcqs[k], &mine) != UTOFU_SUCCESS)
            fail(wrank, "query uTofu VCQ");
        for (int r = 0; r < tp_size; r++) {
            int peer_world = pp_rank * tp_size + r;
            if (peer_world == wrank) peers[k][r] = mine;
            else {
                rc = utofu_construct_vcq_id(topo[peer_world], tni,
                                             DEMO_CQ_ID, DEMO_CMP_ID, &peers[k][r]);
                if (rc != UTOFU_SUCCESS) fail(wrank, "construct peer uTofu VCQ");
                utofu_set_vcq_id_path(&peers[k][r], NULL);
            }
        }
    }
    free(tnis);
    g_tp_init_barrier = tp_mpi;
    if (rsag4) {
        if (tp_size != 4 || tp_rsag4_init(&ctx->rsag4, vcqs, peers, use_tni,
                wrank % tp_size, max_count, tp_init_barrier) != 0)
            fail(wrank, "initialize uTofu TP4 RSAG allreduce");
        ctx->kind = TP_COMM_UTOFU_RSAG4;
    } else {
        if (tp_comm_init(&ctx->tofu, vcqs[0], peers[0], wrank % tp_size, tp_size,
                         max_count, tp_init_barrier) != 0)
            fail(wrank, "initialize uTofu TP allreduce");
        ctx->kind = TP_COMM_UTOFU_TREE;
    }
    ctx->nvcq = use_tni;
    for (int k = 0; k < use_tni; k++) ctx->vcq[k] = vcqs[k];
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int wrank, world;
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if ((world != 8 && world != 12) || argc != 2) {
        if (!wrank) fprintf(stderr, "usage: mpiexec -np {8|12} %s MODEL.gguf\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    const int tp_size = env_i("Q38_PREFILL_TP_SIZE", 4);
    if (tp_size != 1 && tp_size != 4 && tp_size != 6 && tp_size != 12)
        fail(wrank, "Q38_PREFILL_TP_SIZE must be 1, 4, 6, or 12");
    if (world % tp_size || world / tp_size > 12)
        fail(wrank, "world size must be divisible by TP size with PP<=12");
    const int pp_size = world / tp_size;
    const int tp_rank = wrank % tp_size, pp_rank = wrank / tp_size;
    int cuts[13] = {0};
    cuts[pp_size] = 64;
    for (int p = 1; p < pp_size; p++) cuts[p] = 64 * p / pp_size;
    if (pp_size == 1) cuts[1] = env_i("Q38_PREFILL_LAYER_END", 64);
    if (pp_size == 2) cuts[1] = env_i("Q38_PREFILL_CUT1", 31);
    if (pp_size == 3) {
        cuts[1] = env_i("Q38_PREFILL_CUT1", 21);
        cuts[2] = env_i("Q38_PREFILL_CUT2", 43);
    }
    for (int p = 0; p < pp_size; p++)
        if (cuts[p] < 0 || cuts[p + 1] <= cuts[p] || cuts[p + 1] > 64)
            fail(wrank, "invalid pipeline layer cuts");
    const int l0 = cuts[pp_rank], l1 = cuts[pp_rank + 1];
    MPI_Comm tp_comm, pp_comm;
    MPI_Comm_split(MPI_COMM_WORLD, pp_rank, tp_rank, &tp_comm);
    MPI_Comm_split(MPI_COMM_WORLD, tp_rank, pp_rank, &pp_comm);
    tp_reduce_ctx tc;
    memset(&tc, 0, sizeof(tc));
    tc.kind = TP_COMM_MPI;
    tc.mpi = tp_comm;

    int ntok = env_i("Q38_PREFILL_TOKENS", 1024);
    int chunk = env_i("Q38_PREFILL_CHUNK", ntok <= 128 ? 64 : ntok <= 512 ? 128 :
                      ntok >= 4096 ? 1024 : 256);
    int threads = env_i("LLM_THREADS", 48);
    if (ntok < 1 || chunk < 1 || threads < 1 || threads > 48) fail(wrank, "invalid tokens/chunk/threads");
    int max_seq = env_i("Q38_PREFILL_MAXSEQ", ntok + 16);
    if (max_seq < ntok + 1) max_seq = ntok + 1;

    /* Pure pipeline mode loads only this rank's layers from the source GGUF.
     * Their prefill panels become anonymous resident copies, so the shared
     * filesystem is not touched by the timed GEMMs. */
    if (tp_size == 1) {
        char lo[16], hi[16];
        snprintf(lo, sizeof(lo), "%d", l0);
        snprintf(hi, sizeof(hi), "%d", l1);
        setenv("TF_PP_L0", lo, 1);
        setenv("TF_PP_L1", hi, 1);
        unsetenv("TP_STAGE_DIR");
    }

    gguf_context *g = gguf_open_multi(argv[1], 2);
    if (!g) fail(wrank, "open model metadata");
    bpe_vocab *vocab = bpe_vocab_load(g);
    if (!vocab) fail(wrank, "load tokenizer");
    transformer_model *m = transformer_load(g, max_seq);
    if (!m) fail(wrank, "load model metadata");
    transformer_set_threads(m, threads);
    const char *stage = env_s("Q38_PREFILL_STAGE", "/local/u14346/qwen38-bf16-tp4");
    if (tp_size > 1) {
        /* Tell metadata slicing that final dense column shards already exist in
         * the stage file; otherwise it needlessly repacks them from the GGUF. */
        setenv("TP_STAGE_DIR", stage, 1);
        if (transformer_tp_slice_weights(m, tp_rank, tp_size, 1)) fail(wrank, "TP slice");
        if (!transformer_tp_load_stage(m, stage, tp_rank, tp_size)) fail(wrank, "load TP stage");
    }
    transformer_free_unused_kv(m, l0, l1);
    const char *bf16_mode=env_s("Q38_PREFILL_BF16","exact");
    if(!strcmp(bf16_mode,"bf16-act")){
        setenv("TF_PODD","1",1);
        if(!transformer_prepack_podd_range(m,l0,l1))fail(wrank,"pack stage BF16 p_odd");
    } else if(strcmp(bf16_mode,"exact"))
        fail(wrank,"Q38_PREFILL_BF16 must be exact or bf16-act");
    if (!strcmp(bf16_mode,"exact") && env_i("Q38_PREFILL_PV48", 1) &&
        !transformer_prepack_prefill_pv48_range(m, l0, l1))
        fail(wrank, "pack stage BF16 PV48");
    const char *q8_prefill = env_s("Q38_PREFILL_Q8", "native");
    if (!strcmp(q8_prefill, "q8blk6")) {
        if (!transformer_prepack_q8b6_range(m, l0, l1))
            fail(wrank, "pack owned Q8 stage to Q8 block6");
    } else if (!strcmp(q8_prefill, "q8v2") || !strcmp(q8_prefill, "q8v2_arow")) {
        if (!strcmp(q8_prefill, "q8v2_arow")) setenv("TF_Q8V2_AROW", "1", 1);
        if (!transformer_prepack_q8v2_range(m, l0, l1))
            fail(wrank, "pack owned Q8 stage to Q8v2");
    } else if (!strcmp(q8_prefill, "w8a8")) {
        if (!transformer_prepack_q8_w8a8_range(m, l0, l1, pp_rank == pp_size - 1))
            fail(wrank, "pack owned Q8 stage to W8A8");
    } else if (strcmp(q8_prefill, "native")) {
        fail(wrank, "Q38_PREFILL_Q8 must be native, w8a8, q8v2, q8v2_arow, or q8blk6");
    }
    const char *quant = env_s("Q38_PREFILL_QUANT", "none");
    if (strcmp(quant, "none")) {
        if (strcmp(quant, "int8") && strcmp(quant, "int16"))
            fail(wrank, "Q38_PREFILL_QUANT must be none, int8, or int16");
        if (!transformer_prepack_int8_range(m, l0, l1, pp_rank == pp_size - 1))
            fail(wrank, "quantize owned stage");
    }
    if (tp_size > 1) transformer_set_tp(m, tp_rank, tp_size, tp_sum, &tc);

    int32_t *tokens = malloc((size_t)ntok * sizeof(*tokens));
    if (!tokens) fail(wrank, "token allocation");
    int synth = env_i("Q38_PREFILL_TOKEN_ID", 1);
    const char *prompt = getenv("Q38_PREFILL_PROMPT");
    if (prompt && *prompt) {
        int n = 0;
        if (!wrank) n = bpe_tokenize(vocab, prompt, -1, tokens, ntok);
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (n < 1 || n > ntok) fail(wrank, "prompt tokenization");
        ntok = n;
        MPI_Bcast(tokens, ntok, MPI_INT32_T, 0, MPI_COMM_WORLD);
    } else {
        for (int i = 0; i < ntok; i++) tokens[i] = synth;
    }
    float *hidden = NULL;
    uint16_t *hidden_bf16 = NULL;
    if (posix_memalign((void **)&hidden, 256, (size_t)chunk * m->n_embd * sizeof(float)))
        fail(wrank, "hidden allocation");
    int pipe_bf16 = env_i("Q38_PREFILL_PIPE_BF16", 0);
    if (pipe_bf16 && posix_memalign((void **)&hidden_bf16, 256,
            (size_t)chunk * m->n_embd * sizeof(uint16_t)))
        fail(wrank, "BF16 pipeline buffer allocation");

    const char *comm_name = env_s("Q38_PREFILL_COMM", "mpi");
    if (!strcmp(comm_name, "utofu") || !strcmp(comm_name, "utofu-rsag"))
        init_utofu_tp(&tc, wrank, world, pp_rank, tp_size,
                      chunk * m->n_embd, tp_comm, 1);
    else if (!strcmp(comm_name, "utofu-tree"))
        init_utofu_tp(&tc, wrank, world, pp_rank, tp_size,
                      chunk * m->n_embd, tp_comm, 0);
    else if (strcmp(comm_name, "mpi"))
        fail(wrank, "Q38_PREFILL_COMM must be mpi, utofu, or utofu-tree");

    MPI_Barrier(MPI_COMM_WORLD);
    transformer_prefill_profile_reset();
    double begin = wall();
    double compute_s = 0.0, recv_s = 0.0, send_s = 0.0;
    float *last_logits = NULL;
    int nchunk = (ntok + chunk - 1) / chunk;
    for (int tick = 0; tick < nchunk + pp_size - 1; tick++) {
        int ci = tick - pp_rank;
        if (ci < 0 || ci >= nchunk) continue;
        int p0 = ci * chunk, n = ntok - p0; if (n > chunk) n = chunk;
        size_t nf = (size_t)n * m->n_embd;
        double phase = wall();
        if (pp_rank > 0) {
            if (pipe_bf16) {
                MPI_Recv(hidden_bf16, (int)nf, MPI_UINT16_T, pp_rank - 1,
                         ci, pp_comm, MPI_STATUS_IGNORE);
                bf16_to_f32(hidden, hidden_bf16, nf, threads);
            } else {
                MPI_Recv(hidden, (int)nf, MPI_FLOAT, pp_rank - 1, ci,
                         pp_comm, MPI_STATUS_IGNORE);
            }
        }
        recv_s += wall() - phase;
        unsigned flags = pp_rank == 0 ? TF_PREFILL_EMBED : 0;
        if (pp_rank == pp_size - 1 && ci == nchunk - 1) flags |= TF_PREFILL_LOGITS;
        phase = wall();
        last_logits = transformer_prefill_range(m, tokens + p0, hidden, n, p0,
                                                l0, l1, flags);
        compute_s += wall() - phase;
        if (!last_logits) fail(wrank, "range prefill");
        phase = wall();
        if (pp_rank < pp_size - 1) {
            if (pipe_bf16) {
                f32_to_bf16(hidden_bf16, hidden, nf, threads);
                MPI_Send(hidden_bf16, (int)nf, MPI_UINT16_T, pp_rank + 1,
                         ci, pp_comm);
            } else {
                MPI_Send(hidden, (int)nf, MPI_FLOAT, pp_rank + 1, ci, pp_comm);
            }
        }
        send_s += wall() - phase;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = wall() - begin;

    int next = -1;
    if (pp_rank == pp_size - 1) {
        struct { float value; int index; } local = {-INFINITY, 0}, global;
        for (int i = 0; i < m->output.n_rows; i++) if (last_logits[i] > local.value) {
            local.value = last_logits[i]; local.index = m->tp_vocab_lo + i;
        }
        MPI_Allreduce(&local, &global, 1, MPI_FLOAT_INT, MPI_MAXLOC, tp_comm);
        next = global.index;
    }
    MPI_Bcast(&next, 1, MPI_INT, (pp_size - 1) * tp_size, MPI_COMM_WORLD);
    transformer_prefill_profile pf;
    transformer_prefill_profile_get(&pf);
    {
        char path[64]; snprintf(path, sizeof(path), "q38_prefill_rank%02d.txt", wrank);
        FILE *f = fopen(path, "w");
        if (f) {
            fprintf(f, "rank=%d pp=%d tp=%d layers=%d:%d quant=%s q8=%s bf16=%s comm=%s tokens=%d chunk=%d "
                    "time=%.6f compute=%.6f recv=%.6f send=%.6f tok/s=%.2f next=%d\n",
                    wrank, pp_rank, tp_rank, l0, l1, quant, q8_prefill, bf16_mode, comm_name, ntok, chunk, elapsed,
                    compute_s, recv_s, send_s, ntok / elapsed, next);
            fprintf(f, "profile_ms calls=%d layers=%d norm=%.3f proj=%.3f "
                    "ssm_prepare=%.3f ssm_scan=%.3f attn_prepare=%.3f "
                    "attn_kernel=%.3f out_proj=%.3f ffn_proj=%.3f "
                    "ffn_act=%.3f ffn_down=%.3f collective=%.3f\n",
                    pf.calls, pf.layers, pf.norm_ms, pf.proj_ms,
                    pf.ssm_prepare_ms, pf.ssm_scan_ms, pf.attn_prepare_ms,
                    pf.attn_kernel_ms, pf.out_proj_ms, pf.ffn_proj_ms,
                    pf.ffn_act_ms, pf.ffn_down_ms, pf.collective_ms);
            fclose(f);
        }
    }
    if (!wrank) {
        printf("qwen38-prefill topology=PP%dxTP%d quant=%s bf16=%s comm=%s tokens=%d chunk=%d "
               "time=%.6f tok/s=%.2f next=%d\n", pp_size, tp_size, quant,
               bf16_mode, comm_name, ntok, chunk, elapsed, ntok / elapsed, next);
        fflush(stdout);
    }

    free(hidden_bf16); free(hidden); free(tokens);
    transformer_free(m); bpe_vocab_free(vocab); gguf_close(g);
    if (tc.kind == TP_COMM_UTOFU_RSAG4) tp_rsag4_free(&tc.rsag4);
    if (tc.kind == TP_COMM_UTOFU_TREE) tp_comm_free(&tc.tofu);
    if (tc.kind != TP_COMM_MPI) {
        for (int k = 0; k < tc.nvcq; k++) utofu_free_vcq(tc.vcq[k]);
    }
    MPI_Comm_free(&pp_comm); MPI_Comm_free(&tp_comm);
    MPI_Finalize();
    return 0;
}
