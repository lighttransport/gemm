/* SPDX-License-Identifier: MIT */
#ifndef GN_HIPBLASLT_H
#define GN_HIPBLASLT_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Optional SDK bridge. One context/cache per model; no global mutable plans.
 * Device buffers/workspace belong to the caller and its memory budget.
 * Inputs are BF16 row-major A[M,Kpad32], B[N,Kpad32]. Output is FP32[M,N].
 * Workspace reserves align_up(M*N*4,256) bytes for a tuning destination,
 * plus the library budget. First-use tuning never modifies the caller's C. */
/* tune=1 is a benchmark-only timed heuristic search; model execution uses
 * deterministic first-supported selection for exact checkpoint reloads. */
void *gn_lt_open(int tune);
void gn_lt_close(void *context);
int gn_lt_run(void *context, void *y, const void *a, const void *b, int M, int N, int K, float beta,
              void *workspace, size_t workspace_bytes, int fp16);
#ifdef __cplusplus
}
#endif
#endif
