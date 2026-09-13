/* SPDX-License-Identifier: MIT */
#ifndef GEMM_GN_CPU_H
#define GEMM_GN_CPU_H
#include "gn_internal.h"
int gn_cpu_projection(gn_model *, Node *);
void gn_cpu_accumulate(float *dx, float *dw, const float *x, const float *w, float dy, size_t n);
void gn_cpu_close(void *);
#endif
