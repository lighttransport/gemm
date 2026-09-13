/* SPDX-License-Identifier: MIT */
#ifndef GEMM_GN_INTERNAL_H
#define GEMM_GN_INTERNAL_H
#include "gn.h"
#define GN_PARAMS 1024
#define GN_NODES 2048
typedef struct {
    char name[96];
    size_t r, c;
    float *x, *g, *m, *v;
    int learned;
} Param;
typedef enum { INPUT, LINEAR, CONV, ADD, MUL, RELU, SILU, BN, LN, ATTENTION } Kind;
typedef struct {
    Kind kind;
    size_t r, c;
    int a, b, k;
    Param *w, *bias, *mean, *variance;
    float *x, *g, *aux;
} Node;
struct gn_model {
    gn_config cfg;
    Param p[GN_PARAMS];
    Node n[GN_NODES];
    size_t np, nn, bytes, parameters, batch, accumulated;
    uint64_t rng, step;
    int policy, value, training;
    void *gpu;
    void *cpu;
};
int gn_fail(const char *);
void *gn_gpu_open(const char *, int, size_t);
void gn_gpu_close(void *);
int gn_gpu_forward(gn_model *, const float *);
int gn_gpu_backward(gn_model *, const float *, const uint32_t *, gn_metrics *);
int gn_gpu_update(gn_model *, float, float, float, gn_metrics *);
int gn_gpu_sync(gn_model *);
/* Diagnostic only: snapshot resident graph values/gradients after backward. */
int gn_gpu_debug_nodes(gn_model *);
/* Diagnostic only: graph readiness and cumulative successful captures/launches. */
int gn_gpu_graph_stats(const gn_model *, uint64_t *captures, uint64_t *launches);
void gn_gpu_zero(gn_model *);
#endif
