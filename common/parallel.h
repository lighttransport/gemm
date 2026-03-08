/*
 * parallel.h - MPI-based parallelism configuration for transformer inference
 *
 * Supports three orthogonal parallelism dimensions:
 *   TP (Tensor Parallel): split attention heads & FFN neurons across ranks
 *   PP (Pipeline Parallel): split layers across ranks
 *   DP (Data Parallel): independent replicas processing different sequences
 *
 * Total MPI ranks = tp_size * pp_size * dp_size
 *
 * Usage:
 *   #define PARALLEL_IMPLEMENTATION
 *   #include "parallel.h"
 *
 * Dependencies: MPI (mpi.h)
 */
#ifndef PARALLEL_H
#define PARALLEL_H

#include <mpi.h>
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

    /* MPI communicators for each dimension.
     * tp_comm: ranks that share the same PP stage and DP replica.
     * pp_comm: ranks that form a pipeline (same TP position and DP replica).
     * dp_comm: ranks that are DP replicas (same TP position and PP stage). */
    MPI_Comm tp_comm;
    MPI_Comm pp_comm;
    MPI_Comm dp_comm;

    /* Global rank and size */
    int world_rank;
    int world_size;

    /* Pipeline parallel layer range for this rank */
    int pp_layer_start;
    int pp_layer_end;
} parallel_config;

/* Initialize parallel config. Total ranks must equal tp * pp * dp.
 * Rank ordering: fastest-varying = TP, then PP, then DP.
 *   global_rank = dp_rank * (tp_size * pp_size) + pp_rank * tp_size + tp_rank
 * Returns 0 on success, -1 on error. */
int parallel_init(parallel_config *cfg, int tp_size, int pp_size, int dp_size);

/* Compute pipeline-parallel layer partition for n_layers total layers.
 * Sets cfg->pp_layer_start and cfg->pp_layer_end. */
void parallel_compute_pp_layers(parallel_config *cfg, int n_layers);

/* Allreduce (sum) a float buffer in-place over the TP communicator. */
void parallel_tp_allreduce(parallel_config *cfg, float *buf, int count);

/* Allreduce callback compatible with transformer_set_tp().
 * ctx should be a parallel_config pointer. */
void parallel_tp_allreduce_cb(float *buf, int count, void *ctx);

/* Send/recv hidden state for pipeline parallel. */
void parallel_pp_send_hidden(parallel_config *cfg, const float *hidden, int n_embd, int tag);
void parallel_pp_recv_hidden(parallel_config *cfg, float *hidden, int n_embd, int tag);

/* Broadcast int32 from last PP rank to all PP ranks. */
void parallel_pp_bcast_from_last(parallel_config *cfg, int32_t *val);

/* Broadcast int from PP rank 0. */
void parallel_pp_bcast_from_first(parallel_config *cfg, int *val);

/* Cleanup communicators. */
void parallel_finalize(parallel_config *cfg);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef PARALLEL_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>

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

    /* Decompose global rank.
     * Ordering: dp_rank * (tp_size * pp_size) + pp_rank * tp_size + tp_rank */
    int r = cfg->world_rank;
    cfg->tp_rank = r % tp_size;
    r /= tp_size;
    cfg->pp_rank = r % pp_size;
    cfg->dp_rank = r / pp_size;

    /* Create TP communicator: ranks with same (pp_rank, dp_rank) */
    int tp_color = cfg->dp_rank * pp_size + cfg->pp_rank;
    MPI_Comm_split(MPI_COMM_WORLD, tp_color, cfg->tp_rank, &cfg->tp_comm);

    /* Create PP communicator: ranks with same (tp_rank, dp_rank) */
    int pp_color = cfg->dp_rank * tp_size + cfg->tp_rank;
    MPI_Comm_split(MPI_COMM_WORLD, pp_color, cfg->pp_rank, &cfg->pp_comm);

    /* Create DP communicator: ranks with same (tp_rank, pp_rank) */
    int dp_color = cfg->pp_rank * tp_size + cfg->tp_rank;
    MPI_Comm_split(MPI_COMM_WORLD, dp_color, cfg->dp_rank, &cfg->dp_comm);

    cfg->pp_layer_start = 0;
    cfg->pp_layer_end = 0;

    return 0;
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

void parallel_tp_allreduce(parallel_config *cfg, float *buf, int count) {
    if (cfg->tp_size <= 1) return;
    MPI_Allreduce(MPI_IN_PLACE, buf, count, MPI_FLOAT, MPI_SUM, cfg->tp_comm);
}

void parallel_tp_allreduce_cb(float *buf, int count, void *ctx) {
    parallel_config *cfg = (parallel_config *)ctx;
    parallel_tp_allreduce(cfg, buf, count);
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

void parallel_finalize(parallel_config *cfg) {
    if (cfg->tp_comm != MPI_COMM_NULL) MPI_Comm_free(&cfg->tp_comm);
    if (cfg->pp_comm != MPI_COMM_NULL) MPI_Comm_free(&cfg->pp_comm);
    if (cfg->dp_comm != MPI_COMM_NULL) MPI_Comm_free(&cfg->dp_comm);
}

#endif /* PARALLEL_IMPLEMENTATION */
#endif /* PARALLEL_H */
