/* SPDX-License-Identifier: MIT */
#ifndef GN_HIPBLASLT_H
#define GN_HIPBLASLT_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Optional SDK bridge. One context/cache per model; no global mutable plans.
 * Device buffers/workspace belong to the caller and its memory budget.
 * Inputs are BF16 row-major A[M,Kpad32], B[N,Kpad32]. Output is FP32[M,N]. */
void *gn_lt_open(void);
void gn_lt_close(void *context);
int gn_lt_run(void *context, void *y, const void *a, const void *b, int M, int N, int K, float beta,
              void *workspace, size_t workspace_bytes);
#ifdef __cplusplus
}
#endif
#endif
