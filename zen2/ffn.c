/*
 * FFN layer drivers for transformer inference — Zen2 AVX2.
 *
 * SwiGLU:   output = (SiLU(X @ W_gate^T) * (X @ W_up^T)) @ W_down^T
 * Standard: output = Activation(X @ W1^T) @ W2^T
 *
 * Uses gemm_fp32() for all matrix multiplications and AVX2 activation
 * functions from activation.h. Intermediate buffers are persistent
 * (lazy-allocated with posix_memalign + MADV_HUGEPAGE).
 */

#include "ffn.h"
#include "gemm.h"
#include "activation.h"
#include <stdlib.h>
#include <sys/mman.h>

/* ------------------------------------------------------------------ */
/*  Persistent intermediate buffers                                    */
/* ------------------------------------------------------------------ */

static float *g_gate_buf = NULL;  /* [M, D_ff] — gate projection / activation result */
static float *g_up_buf   = NULL;  /* [M, D_ff] — up projection (SwiGLU only) */
static size_t g_gate_cap = 0;    /* capacity in floats */
static size_t g_up_cap   = 0;

static float *ensure_buf(float **buf, size_t *cap, size_t need_floats)
{
    if (need_floats <= *cap)
        return *buf;
    free(*buf);
    size_t bytes = need_floats * sizeof(float);
    posix_memalign((void **)buf, 64, bytes);
    madvise(*buf, bytes, MADV_HUGEPAGE);
    *cap = need_floats;
    return *buf;
}

void ffn_cleanup(void)
{
    free(g_gate_buf); g_gate_buf = NULL; g_gate_cap = 0;
    free(g_up_buf);   g_up_buf   = NULL; g_up_cap   = 0;
}

/* ------------------------------------------------------------------ */
/*  SwiGLU FFN                                                         */
/* ------------------------------------------------------------------ */

void ffn_swiglu_fp32(
    const float *X,
    const float *W_gate,
    const float *W_up,
    const float *W_down,
    float *output,
    int M, int D, int D_ff)
{
    size_t inter_size = (size_t)M * D_ff;
    float *gate_buf = ensure_buf(&g_gate_buf, &g_gate_cap, inter_size);
    float *up_buf   = ensure_buf(&g_up_buf,   &g_up_cap,   inter_size);

    /* Step 1: gate_buf[M, D_ff] = X[M, D] @ W_gate[D_ff, D]^T */
    gemm_fp32(X, D, W_gate, D, gate_buf, D_ff, M, D_ff, D);

    /* Step 2: up_buf[M, D_ff] = X[M, D] @ W_up[D_ff, D]^T */
    gemm_fp32(X, D, W_up, D, up_buf, D_ff, M, D_ff, D);

    /* Step 3: gate_buf[i] = SiLU(gate_buf[i]) * up_buf[i] (fused SwiGLU) */
    swiglu_avx2(gate_buf, up_buf, gate_buf, (int)inter_size);

    /* Step 4: output[M, D] = gate_buf[M, D_ff] @ W_down[D, D_ff]^T */
    gemm_fp32(gate_buf, D_ff, W_down, D_ff, output, D, M, D, D_ff);
}

/* ------------------------------------------------------------------ */
/*  Standard FFN                                                       */
/* ------------------------------------------------------------------ */

void ffn_standard_fp32(
    const float *X,
    const float *W1,
    const float *W2,
    float *output,
    int M, int D, int D_ff,
    ffn_activation_t activation)
{
    size_t inter_size = (size_t)M * D_ff;
    float *buf = ensure_buf(&g_gate_buf, &g_gate_cap, inter_size);

    /* Step 1: buf[M, D_ff] = X[M, D] @ W1[D_ff, D]^T */
    gemm_fp32(X, D, W1, D, buf, D_ff, M, D_ff, D);

    /* Step 2: in-place activation */
    switch (activation) {
    case FFN_ACT_RELU:
        relu_avx2(buf, buf, (int)inter_size);
        break;
    case FFN_ACT_GELU:
        gelu_avx2(buf, buf, (int)inter_size);
        break;
    case FFN_ACT_SILU:
        silu_avx2(buf, buf, (int)inter_size);
        break;
    }

    /* Step 3: output[M, D] = buf[M, D_ff] @ W2[D, D_ff]^T */
    gemm_fp32(buf, D_ff, W2, D_ff, output, D, M, D, D_ff);
}
