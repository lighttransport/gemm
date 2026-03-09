/*
 * parallel.h - Parallelism configuration for distributed transformer inference
 *
 * Supports three orthogonal parallelism dimensions:
 *   TP (Tensor Parallel): split attention heads & FFN neurons across ranks
 *   PP (Pipeline Parallel): split layers across ranks
 *   DP (Data Parallel): independent replicas processing different sequences
 *
 * Total ranks = tp_size * pp_size * dp_size
 *
 * Two backends:
 *   - Custom comm (default): uses comm.h (TCP + shared memory, no MPI)
 *   - MPI: define PARALLEL_USE_MPI before including this file
 *
 * Usage:
 *   #define PARALLEL_IMPLEMENTATION
 *   #include "parallel.h"
 */
#ifndef PARALLEL_H
#define PARALLEL_H

#ifdef PARALLEL_USE_MPI
#include <mpi.h>
#else
#include "comm.h"
#endif

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    /* Sizes of each parallelism dimension */
    int tp_size;    /* tensor parallel group size */
    int pp_size;    /* pipeline parallel group size */
    int dp_size;    /* data parallel group size */

    /* This rank's position in each dimension */
    int tp_rank;
    int pp_rank;
    int dp_rank;

    /* Global rank and size */
    int world_rank;
    int world_size;

    /* Pipeline parallel layer range for this rank */
    int pp_layer_start;
    int pp_layer_end;

#ifdef PARALLEL_USE_MPI
    /* MPI communicators */
    MPI_Comm tp_comm;
    MPI_Comm pp_comm;
    MPI_Comm dp_comm;
#else
    /* Custom comm contexts */
    comm_context *world_ctx;
    comm_context *tp_ctx;
    comm_context *pp_ctx;
    comm_context *dp_ctx;
#endif
} parallel_config;

#ifdef PARALLEL_USE_MPI
/* MPI backend: rank/size obtained from MPI_COMM_WORLD */
int parallel_init(parallel_config *cfg, int tp_size, int pp_size, int dp_size);
#else
/* Custom comm backend: caller provides rank/size and bootstrap info */
int parallel_init(parallel_config *cfg, int tp_size, int pp_size, int dp_size,
                   int rank, int world_size, const char *master_addr, int master_port);
#endif

/* Compute pipeline-parallel layer partition for n_layers total layers. */
void parallel_compute_pp_layers(parallel_config *cfg, int n_layers);

/* Allreduce (sum) float buffer in-place over the TP communicator. */
void parallel_tp_allreduce(parallel_config *cfg, float *buf, int count);

/* Allreduce callback compatible with transformer_set_tp(). ctx = parallel_config*. */
void parallel_tp_allreduce_cb(float *buf, int count, void *ctx);

/* Pipeline parallel: send/recv hidden state to/from adjacent PP ranks. */
void parallel_pp_send_hidden(parallel_config *cfg, const float *hidden, int n_embd, int tag);
void parallel_pp_recv_hidden(parallel_config *cfg, float *hidden, int n_embd, int tag);

/* Broadcast int32 from last PP rank to all PP ranks. */
void parallel_pp_bcast_from_last(parallel_config *cfg, int32_t *val);

/* Broadcast int from PP rank 0. */
void parallel_pp_bcast_from_first(parallel_config *cfg, int *val);

/* Broadcast n_tokens int32s from world rank 0. */
void parallel_bcast_tokens(parallel_config *cfg, int32_t *tokens, int *n_tokens);

/* Barrier across all ranks. */
void parallel_barrier(parallel_config *cfg);

/* Cleanup communicators. */
void parallel_finalize(parallel_config *cfg);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef PARALLEL_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Shared logic (backend-independent) ---- */

static void parallel_decompose_rank(parallel_config *cfg) {
    int r = cfg->world_rank;
    cfg->tp_rank = r % cfg->tp_size;
    r /= cfg->tp_size;
    cfg->pp_rank = r % cfg->pp_size;
    cfg->dp_rank = r / cfg->pp_size;
}

void parallel_compute_pp_layers(parallel_config *cfg, int n_layers) {
    int layers_per = n_layers / cfg->pp_size;
    int extra = n_layers % cfg->pp_size;
    int start = 0;
    for (int r = 0; r < cfg->pp_rank; r++)
        start += layers_per + (r < extra ? 1 : 0);
    cfg->pp_layer_start = start;
    cfg->pp_layer_end = start + layers_per + (cfg->pp_rank < extra ? 1 : 0);
}

void parallel_tp_allreduce_cb(float *buf, int count, void *ctx) {
    parallel_config *cfg = (parallel_config *)ctx;
    parallel_tp_allreduce(cfg, buf, count);
}

/* ================================================================ */
#ifdef PARALLEL_USE_MPI
/* ---- MPI backend ---- */

int parallel_init(parallel_config *cfg, int tp_size, int pp_size, int dp_size) {
    MPI_Comm_rank(MPI_COMM_WORLD, &cfg->world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &cfg->world_size);
    cfg->tp_size = tp_size;
    cfg->pp_size = pp_size;
    cfg->dp_size = dp_size;

    int required = tp_size * pp_size * dp_size;
    if (cfg->world_size != required) {
        if (cfg->world_rank == 0)
            fprintf(stderr, "parallel_init: need %d ranks (tp=%d * pp=%d * dp=%d), got %d\n",
                    required, tp_size, pp_size, dp_size, cfg->world_size);
        return -1;
    }

    parallel_decompose_rank(cfg);

    int tp_color = cfg->dp_rank * pp_size + cfg->pp_rank;
    MPI_Comm_split(MPI_COMM_WORLD, tp_color, cfg->tp_rank, &cfg->tp_comm);
    int pp_color = cfg->dp_rank * tp_size + cfg->tp_rank;
    MPI_Comm_split(MPI_COMM_WORLD, pp_color, cfg->pp_rank, &cfg->pp_comm);
    int dp_color = cfg->pp_rank * tp_size + cfg->tp_rank;
    MPI_Comm_split(MPI_COMM_WORLD, dp_color, cfg->dp_rank, &cfg->dp_comm);

    cfg->pp_layer_start = 0;
    cfg->pp_layer_end = 0;
    return 0;
}

void parallel_tp_allreduce(parallel_config *cfg, float *buf, int count) {
    if (cfg->tp_size <= 1) return;
    MPI_Allreduce(MPI_IN_PLACE, buf, count, MPI_FLOAT, MPI_SUM, cfg->tp_comm);
}

void parallel_pp_send_hidden(parallel_config *cfg, const float *hidden, int n_embd, int tag) {
    MPI_Send(hidden, n_embd, MPI_FLOAT, cfg->pp_rank + 1, tag, cfg->pp_comm);
}

void parallel_pp_recv_hidden(parallel_config *cfg, float *hidden, int n_embd, int tag) {
    MPI_Recv(hidden, n_embd, MPI_FLOAT, cfg->pp_rank - 1, tag, cfg->pp_comm, MPI_STATUS_IGNORE);
}

void parallel_pp_bcast_from_last(parallel_config *cfg, int32_t *val) {
    MPI_Bcast(val, 1, MPI_INT32_T, cfg->pp_size - 1, cfg->pp_comm);
}

void parallel_pp_bcast_from_first(parallel_config *cfg, int *val) {
    MPI_Bcast(val, 1, MPI_INT, 0, cfg->pp_comm);
}

void parallel_bcast_tokens(parallel_config *cfg, int32_t *tokens, int *n_tokens) {
    MPI_Bcast(n_tokens, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(tokens, *n_tokens, MPI_INT32_T, 0, MPI_COMM_WORLD);
}

void parallel_barrier(parallel_config *cfg) {
    MPI_Barrier(MPI_COMM_WORLD);
}

void parallel_finalize(parallel_config *cfg) {
    if (cfg->tp_comm != MPI_COMM_NULL) MPI_Comm_free(&cfg->tp_comm);
    if (cfg->pp_comm != MPI_COMM_NULL) MPI_Comm_free(&cfg->pp_comm);
    if (cfg->dp_comm != MPI_COMM_NULL) MPI_Comm_free(&cfg->dp_comm);
}

#else
/* ---- Custom comm backend ---- */

int parallel_init(parallel_config *cfg, int tp_size, int pp_size, int dp_size,
                   int rank, int world_size, const char *master_addr, int master_port) {
    cfg->tp_size = tp_size;
    cfg->pp_size = pp_size;
    cfg->dp_size = dp_size;
    cfg->world_rank = rank;
    cfg->world_size = world_size;

    int required = tp_size * pp_size * dp_size;
    if (world_size != required) {
        if (rank == 0)
            fprintf(stderr, "parallel_init: need %d ranks (tp=%d * pp=%d * dp=%d), got %d\n",
                    required, tp_size, pp_size, dp_size, world_size);
        return -1;
    }

    parallel_decompose_rank(cfg);

    /* Create world communicator */
    cfg->world_ctx = comm_create(rank, world_size, master_addr, master_port);
    if (!cfg->world_ctx) return -1;

    /* Split into TP/PP/DP sub-communicators */
    int tp_color = cfg->dp_rank * pp_size + cfg->pp_rank;
    cfg->tp_ctx = comm_split(cfg->world_ctx, tp_color, cfg->tp_rank);

    int pp_color = cfg->dp_rank * tp_size + cfg->tp_rank;
    cfg->pp_ctx = comm_split(cfg->world_ctx, pp_color, cfg->pp_rank);

    int dp_color = cfg->pp_rank * tp_size + cfg->tp_rank;
    cfg->dp_ctx = comm_split(cfg->world_ctx, dp_color, cfg->dp_rank);

    cfg->pp_layer_start = 0;
    cfg->pp_layer_end = 0;
    return 0;
}

void parallel_tp_allreduce(parallel_config *cfg, float *buf, int count) {
    if (cfg->tp_size <= 1) return;
    comm_allreduce_sum_f32(cfg->tp_ctx, buf, count);
}

void parallel_pp_send_hidden(parallel_config *cfg, const float *hidden, int n_embd, int tag) {
    (void)tag;
    comm_send(cfg->pp_ctx, hidden, n_embd * sizeof(float), cfg->pp_rank + 1);
}

void parallel_pp_recv_hidden(parallel_config *cfg, float *hidden, int n_embd, int tag) {
    (void)tag;
    comm_recv(cfg->pp_ctx, hidden, n_embd * sizeof(float), cfg->pp_rank - 1);
}

void parallel_pp_bcast_from_last(parallel_config *cfg, int32_t *val) {
    comm_broadcast(cfg->pp_ctx, val, sizeof(int32_t), cfg->pp_size - 1);
}

void parallel_pp_bcast_from_first(parallel_config *cfg, int *val) {
    comm_broadcast(cfg->pp_ctx, val, sizeof(int), 0);
}

void parallel_bcast_tokens(parallel_config *cfg, int32_t *tokens, int *n_tokens) {
    comm_broadcast(cfg->world_ctx, n_tokens, sizeof(int), 0);
    comm_broadcast(cfg->world_ctx, tokens, *n_tokens * sizeof(int32_t), 0);
}

void parallel_barrier(parallel_config *cfg) {
    comm_barrier(cfg->world_ctx);
}

void parallel_finalize(parallel_config *cfg) {
    comm_destroy(cfg->dp_ctx);
    comm_destroy(cfg->pp_ctx);
    comm_destroy(cfg->tp_ctx);
    comm_destroy(cfg->world_ctx);
}

#endif /* PARALLEL_USE_MPI */

#endif /* PARALLEL_IMPLEMENTATION */
#endif /* PARALLEL_H */
