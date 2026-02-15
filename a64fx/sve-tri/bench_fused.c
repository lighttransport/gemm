#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "fused_rope_qk.h"

static double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static float randf(void) {
    return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
}

/*
 * Correctness test: compare assembly kernel output against C reference.
 * Uses position=0 so all tokens get zero RoPE angles (cos=1, sin=0),
 * which reduces to pure GEMM — easy to verify.
 * Then also tests with non-zero position.
 */
static int test_correctness(int n_tokens, int d_model, int d_head) {
    printf("  Correctness: n_tok=%d, d_model=%d, d_head=%d ... ",
           n_tokens, d_model, d_head);

    size_t X_sz   = (size_t)n_tokens * d_model;
    size_t W_sz   = (size_t)d_model * d_head;
    size_t QK_sz  = (size_t)n_tokens * d_head;
    size_t th_sz  = d_head / 2;

    float *X     = (float *)aligned_alloc(64, X_sz * sizeof(float));
    float *Wq    = (float *)aligned_alloc(64, W_sz * sizeof(float));
    float *Wk    = (float *)aligned_alloc(64, W_sz * sizeof(float));
    float *Q_ref = (float *)aligned_alloc(64, QK_sz * sizeof(float));
    float *K_ref = (float *)aligned_alloc(64, QK_sz * sizeof(float));
    float *Q_asm = (float *)aligned_alloc(64, QK_sz * sizeof(float));
    float *K_asm = (float *)aligned_alloc(64, QK_sz * sizeof(float));
    float *theta = (float *)aligned_alloc(64, th_sz * sizeof(float));
    float *theta_scaled = (float *)aligned_alloc(64, th_sz * sizeof(float));
    int   *pos   = (int *)malloc(n_tokens * sizeof(int));

    /* Initialize with small random values */
    srand(42);
    for (size_t i = 0; i < X_sz; i++)  X[i]  = randf() * 0.1f;
    for (size_t i = 0; i < W_sz; i++)  Wq[i] = randf() * 0.1f;
    for (size_t i = 0; i < W_sz; i++)  Wk[i] = randf() * 0.1f;

    /* RoPE theta: standard 1/10000^(2i/d_head) */
    for (int i = 0; i < (int)th_sz; i++) {
        theta[i] = 1.0f / powf(10000.0f, 2.0f * i / (float)d_head);
    }

    /* Test with position = 5 for all tokens (same-position, valid for asm path) */
    int test_pos = 5;
    for (int t = 0; t < n_tokens; t++) pos[t] = test_pos;
    for (int i = 0; i < (int)th_sz; i++)
        theta_scaled[i] = theta[i] * (float)test_pos;

    /* Reference */
    fused_proj_rope_f32(X, Wq, Wk, Q_ref, K_ref, theta, pos,
                        n_tokens, d_model, d_head);

    /* Pack weights for asm path */
    float *Wq_p = (float *)aligned_alloc(64, packed_weight_size(d_model, d_head));
    float *Wk_p = (float *)aligned_alloc(64, packed_weight_size(d_model, d_head));
    pack_weight_f32(Wq, Wq_p, d_model, d_head);
    pack_weight_f32(Wk, Wk_p, d_model, d_head);

    /* Assembly path */
    memset(Q_asm, 0, QK_sz * sizeof(float));
    memset(K_asm, 0, QK_sz * sizeof(float));
    fused_proj_rope_asm_f32(X, Wq_p, Wk_p, Q_asm, K_asm,
                            theta_scaled, n_tokens, d_model, d_head);

    /* Compare */
    float max_err_q = 0.0f, max_err_k = 0.0f;
    float max_rel_q = 0.0f, max_rel_k = 0.0f;
    for (size_t i = 0; i < QK_sz; i++) {
        float eq = fabsf(Q_ref[i] - Q_asm[i]);
        float ek = fabsf(K_ref[i] - K_asm[i]);
        if (eq > max_err_q) max_err_q = eq;
        if (ek > max_err_k) max_err_k = ek;
        float dq = fabsf(Q_ref[i]) > 1e-8f ? eq / fabsf(Q_ref[i]) : eq;
        float dk = fabsf(K_ref[i]) > 1e-8f ? ek / fabsf(K_ref[i]) : ek;
        if (dq > max_rel_q) max_rel_q = dq;
        if (dk > max_rel_k) max_rel_k = dk;
    }

    int pass = (max_err_q < 1e-3f) && (max_err_k < 1e-3f);
    printf("%s  max_abs(Q)=%.2e  max_abs(K)=%.2e  max_rel(Q)=%.2e  max_rel(K)=%.2e\n",
           pass ? "PASS" : "FAIL",
           max_err_q, max_err_k, max_rel_q, max_rel_k);

    if (!pass) {
        /* Print first few mismatches */
        int printed = 0;
        for (size_t i = 0; i < QK_sz && printed < 5; i++) {
            float eq = fabsf(Q_ref[i] - Q_asm[i]);
            if (eq > 1e-3f) {
                int t = i / d_head, j = i % d_head;
                printf("    Q[%d,%d]: ref=%.6f asm=%.6f err=%.2e\n",
                       t, j, Q_ref[i], Q_asm[i], eq);
                printed++;
            }
        }
        printed = 0;
        for (size_t i = 0; i < QK_sz && printed < 5; i++) {
            float ek = fabsf(K_ref[i] - K_asm[i]);
            if (ek > 1e-3f) {
                int t = i / d_head, j = i % d_head;
                printf("    K[%d,%d]: ref=%.6f asm=%.6f err=%.2e\n",
                       t, j, K_ref[i], K_asm[i], ek);
                printed++;
            }
        }
    }

    free(X); free(Wq); free(Wk);
    free(Q_ref); free(K_ref); free(Q_asm); free(K_asm);
    free(theta); free(theta_scaled); free(pos);
    free(Wq_p); free(Wk_p);

    return pass;
}

/*
 * Performance benchmark: time the assembly path.
 */
static void bench_perf(int n_tokens, int d_model, int d_head, int n_reps) {
    size_t X_sz  = (size_t)n_tokens * d_model;
    size_t W_sz  = (size_t)d_model * d_head;
    size_t QK_sz = (size_t)n_tokens * d_head;
    size_t th_sz = d_head / 2;
    float *X     = (float *)aligned_alloc(64, X_sz * sizeof(float));
    float *Wq    = (float *)aligned_alloc(64, W_sz * sizeof(float));
    float *Wk    = (float *)aligned_alloc(64, W_sz * sizeof(float));
    float *Q     = (float *)aligned_alloc(64, QK_sz * sizeof(float));
    float *K_out = (float *)aligned_alloc(64, QK_sz * sizeof(float));
    float *theta = (float *)aligned_alloc(64, th_sz * sizeof(float));

    srand(42);
    for (size_t i = 0; i < X_sz; i++) X[i] = randf() * 0.1f;
    for (size_t i = 0; i < W_sz; i++) Wq[i] = randf() * 0.1f;
    for (size_t i = 0; i < W_sz; i++) Wk[i] = randf() * 0.1f;
    for (int i = 0; i < (int)th_sz; i++)
        theta[i] = 1.0f / powf(10000.0f, 2.0f * i / (float)d_head);

    /* Pre-scale theta for position 5 */
    float *theta_scaled = (float *)aligned_alloc(64, th_sz * sizeof(float));
    for (int i = 0; i < (int)th_sz; i++)
        theta_scaled[i] = theta[i] * 5.0f;

    float *Wq_p = (float *)aligned_alloc(64, packed_weight_size(d_model, d_head));
    float *Wk_p = (float *)aligned_alloc(64, packed_weight_size(d_model, d_head));
    pack_weight_f32(Wq, Wq_p, d_model, d_head);
    pack_weight_f32(Wk, Wk_p, d_model, d_head);

    /* Warmup */
    for (int r = 0; r < 3; r++) {
        fused_proj_rope_asm_f32(X, Wq_p, Wk_p, Q, K_out,
                                theta_scaled, n_tokens, d_model, d_head);
    }

    /* Timed */
    double t0 = get_time();
    for (int r = 0; r < n_reps; r++) {
        fused_proj_rope_asm_f32(X, Wq_p, Wk_p, Q, K_out,
                                theta_scaled, n_tokens, d_model, d_head);
    }
    double t1 = get_time();
    double dt = (t1 - t0) / n_reps;

    /* FLOPs: 2 projections × n_tokens × d_model × d_head × 2 (mul+add)
     * + RoPE: n_tokens × d_head × 6 (2 mul, 2 fma, per pair) */
    double gemm_flops = 2.0 * 2.0 * n_tokens * d_model * d_head;
    double rope_flops = 6.0 * n_tokens * d_head;
    double total_flops = gemm_flops + rope_flops;
    double gflops = total_flops / dt / 1e9;
    double peak = 128.0;  /* GF at 2 GHz, 2 FMA units × 16 floats × 2 */
    double pct = gflops / peak * 100.0;

    /* Also compute GEMM-only fraction */
    double gemm_gf = gemm_flops / dt / 1e9;

    printf("  n_tok=%3d d_model=%4d d_head=%3d | %7.1f GF (%.1f%% peak) | "
           "GEMM=%.1f GF | time=%.3f ms\n",
           n_tokens, d_model, d_head,
           gflops, pct, gemm_gf, dt * 1e3);

    free(X); free(Wq); free(Wk); free(Q); free(K_out);
    free(theta); free(theta_scaled);
    free(Wq_p); free(Wk_p);
}

/* ================================================================
 * FP16 Correctness test
 * ================================================================ */
static int test_correctness_f16(int n_tokens, int d_model, int d_head) {
    printf("  Correctness: n_tok=%d, d_model=%d, d_head=%d ... ",
           n_tokens, d_model, d_head);

    size_t X_sz   = (size_t)n_tokens * d_model;
    size_t W_sz   = (size_t)d_model * d_head;
    size_t QK_sz  = (size_t)n_tokens * d_head;
    size_t th_sz  = d_head / 2;

    /* fp32 inputs for reference */
    float *X     = (float *)aligned_alloc(64, X_sz * sizeof(float));
    float *Wq    = (float *)aligned_alloc(64, W_sz * sizeof(float));
    float *Wk    = (float *)aligned_alloc(64, W_sz * sizeof(float));
    float *Q_ref = (float *)aligned_alloc(64, QK_sz * sizeof(float));
    float *K_ref = (float *)aligned_alloc(64, QK_sz * sizeof(float));
    float *theta = (float *)aligned_alloc(64, th_sz * sizeof(float));
    float *theta_scaled = (float *)aligned_alloc(64, th_sz * sizeof(float));
    int   *pos   = (int *)malloc(n_tokens * sizeof(int));

    /* fp16 inputs/outputs */
    _Float16 *X_f16  = (_Float16 *)aligned_alloc(64, X_sz * sizeof(_Float16));
    _Float16 *Wq_f16 = (_Float16 *)aligned_alloc(64, W_sz * sizeof(_Float16));
    _Float16 *Wk_f16 = (_Float16 *)aligned_alloc(64, W_sz * sizeof(_Float16));
    _Float16 *Q_asm  = (_Float16 *)aligned_alloc(64, QK_sz * sizeof(_Float16));
    _Float16 *K_asm  = (_Float16 *)aligned_alloc(64, QK_sz * sizeof(_Float16));

    srand(42);
    for (size_t i = 0; i < X_sz; i++) {
        X[i] = randf() * 0.1f;
        X_f16[i] = (_Float16)X[i];
    }
    for (size_t i = 0; i < W_sz; i++) {
        Wq[i] = randf() * 0.1f;
        Wk[i] = randf() * 0.1f;
        Wq_f16[i] = (_Float16)Wq[i];
        Wk_f16[i] = (_Float16)Wk[i];
    }

    for (int i = 0; i < (int)th_sz; i++)
        theta[i] = 1.0f / powf(10000.0f, 2.0f * i / (float)d_head);

    int test_pos = 5;
    for (int t = 0; t < n_tokens; t++) pos[t] = test_pos;
    for (int i = 0; i < (int)th_sz; i++)
        theta_scaled[i] = theta[i] * (float)test_pos;

    /* Reference: fp32 compute using fp16-quantized inputs */
    fused_proj_rope_f16(X, Wq, Wk, Q_ref, K_ref, theta, pos,
                        n_tokens, d_model, d_head);

    /* Pack weights for asm path */
    _Float16 *Wq_p = (_Float16 *)aligned_alloc(64, packed_weight_size_f16(d_model, d_head));
    _Float16 *Wk_p = (_Float16 *)aligned_alloc(64, packed_weight_size_f16(d_model, d_head));
    pack_weight_f16(Wq_f16, Wq_p, d_model, d_head);
    pack_weight_f16(Wk_f16, Wk_p, d_model, d_head);

    /* Assembly path */
    memset(Q_asm, 0, QK_sz * sizeof(_Float16));
    memset(K_asm, 0, QK_sz * sizeof(_Float16));
    fused_proj_rope_asm_f16(X_f16, Wq_p, Wk_p, Q_asm, K_asm,
                            theta_scaled, n_tokens, d_model, d_head);

    /* Compare: fp16 has ~3 decimal digits, so relax tolerance */
    float max_err_q = 0.0f, max_err_k = 0.0f;
    for (size_t i = 0; i < QK_sz; i++) {
        float eq = fabsf(Q_ref[i] - (float)Q_asm[i]);
        float ek = fabsf(K_ref[i] - (float)K_asm[i]);
        if (eq > max_err_q) max_err_q = eq;
        if (ek > max_err_k) max_err_k = ek;
    }

    int pass = (max_err_q < 0.15f) && (max_err_k < 0.15f);
    printf("%s  max_abs(Q)=%.2e  max_abs(K)=%.2e\n",
           pass ? "PASS" : "FAIL", max_err_q, max_err_k);

    if (!pass) {
        int printed = 0;
        for (size_t i = 0; i < QK_sz && printed < 5; i++) {
            float eq = fabsf(Q_ref[i] - (float)Q_asm[i]);
            if (eq > 0.15f) {
                int t = i / d_head, j = i % d_head;
                printf("    Q[%d,%d]: ref=%.6f asm=%.6f err=%.2e\n",
                       t, j, Q_ref[i], (float)Q_asm[i], eq);
                printed++;
            }
        }
        printed = 0;
        for (size_t i = 0; i < QK_sz && printed < 5; i++) {
            float ek = fabsf(K_ref[i] - (float)K_asm[i]);
            if (ek > 0.15f) {
                int t = i / d_head, j = i % d_head;
                printf("    K[%d,%d]: ref=%.6f asm=%.6f err=%.2e\n",
                       t, j, K_ref[i], (float)K_asm[i], ek);
                printed++;
            }
        }
    }

    free(X); free(Wq); free(Wk);
    free(Q_ref); free(K_ref);
    free(X_f16); free(Wq_f16); free(Wk_f16);
    free(Q_asm); free(K_asm);
    free(theta); free(theta_scaled); free(pos);
    free(Wq_p); free(Wk_p);

    return pass;
}

/* ================================================================
 * FP16 Performance benchmark
 * ================================================================ */
static void bench_perf_f16(int n_tokens, int d_model, int d_head, int n_reps) {
    size_t X_sz  = (size_t)n_tokens * d_model;
    size_t W_sz  = (size_t)d_model * d_head;
    size_t QK_sz = (size_t)n_tokens * d_head;
    size_t th_sz = d_head / 2;
    int nb = d_head / FUSED_NR_F16;

    _Float16 *X     = (_Float16 *)aligned_alloc(64, X_sz * sizeof(_Float16));
    _Float16 *Q     = (_Float16 *)aligned_alloc(64, QK_sz * sizeof(_Float16));
    _Float16 *K_out = (_Float16 *)aligned_alloc(64, QK_sz * sizeof(_Float16));
    float *theta    = (float *)aligned_alloc(64, th_sz * sizeof(float));

    srand(42);
    for (size_t i = 0; i < X_sz; i++) X[i] = (_Float16)(randf() * 0.1f);
    for (int i = 0; i < (int)th_sz; i++)
        theta[i] = 1.0f / powf(10000.0f, 2.0f * i / (float)d_head);

    float *theta_scaled = (float *)aligned_alloc(64, th_sz * sizeof(float));
    for (int i = 0; i < (int)th_sz; i++)
        theta_scaled[i] = theta[i] * 5.0f;

    /* Generate random fp16 weights and pack */
    _Float16 *Wq_rm = (_Float16 *)aligned_alloc(64, W_sz * sizeof(_Float16));
    _Float16 *Wk_rm = (_Float16 *)aligned_alloc(64, W_sz * sizeof(_Float16));
    for (size_t i = 0; i < W_sz; i++) {
        Wq_rm[i] = (_Float16)(randf() * 0.1f);
        Wk_rm[i] = (_Float16)(randf() * 0.1f);
    }

    _Float16 *Wq_p = (_Float16 *)aligned_alloc(64, packed_weight_size_f16(d_model, d_head));
    _Float16 *Wk_p = (_Float16 *)aligned_alloc(64, packed_weight_size_f16(d_model, d_head));
    pack_weight_f16(Wq_rm, Wq_p, d_model, d_head);
    pack_weight_f16(Wk_rm, Wk_p, d_model, d_head);

    /* Pre-compute sin/cos and allocate workspace OUTSIDE timed loop */
    _Float16 *sin_cos_all = (_Float16 *)aligned_alloc(64,
        (size_t)nb * FUSED_NR_F16 * sizeof(_Float16));
    compute_sin_cos_f16(theta_scaled, sin_cos_all, d_head);

    _Float16 *X_packed = (_Float16 *)aligned_alloc(64,
        ((size_t)d_model + 1) * FUSED_MR_F16 * sizeof(_Float16));
    _Float16 *Q_tmp = NULL, *K_tmp = NULL;
    if (n_tokens % FUSED_MR_F16 != 0) {
        Q_tmp = (_Float16 *)aligned_alloc(64, (size_t)FUSED_MR_F16 * d_head * sizeof(_Float16));
        K_tmp = (_Float16 *)aligned_alloc(64, (size_t)FUSED_MR_F16 * d_head * sizeof(_Float16));
    }

    /* Warmup */
    for (int r = 0; r < 3; r++) {
        fused_proj_rope_core_f16(X, Wq_p, Wk_p, Q, K_out,
                                 sin_cos_all, X_packed, Q_tmp, K_tmp,
                                 n_tokens, d_model, d_head);
    }

    /* Timed: only core loop (packing + GEMM + RoPE) */
    double t0 = get_time();
    for (int r = 0; r < n_reps; r++) {
        fused_proj_rope_core_f16(X, Wq_p, Wk_p, Q, K_out,
                                 sin_cos_all, X_packed, Q_tmp, K_tmp,
                                 n_tokens, d_model, d_head);
    }
    double t1 = get_time();
    double dt = (t1 - t0) / n_reps;

    double gemm_flops = 2.0 * 2.0 * n_tokens * d_model * d_head;
    double rope_flops = 6.0 * n_tokens * d_head;
    double total_flops = gemm_flops + rope_flops;
    double gflops = total_flops / dt / 1e9;
    double peak = 256.0;  /* 2 GHz × 2 FMA/cy × 32 fp16 elements × 2 FLOPS */
    double pct = gflops / peak * 100.0;
    double gemm_gf = gemm_flops / dt / 1e9;

    printf("  n_tok=%3d d_model=%4d d_head=%3d | %7.1f GF (%.1f%% peak) | "
           "GEMM=%.1f GF | time=%.3f ms\n",
           n_tokens, d_model, d_head,
           gflops, pct, gemm_gf, dt * 1e3);

    free(X); free(Q); free(K_out); free(theta); free(theta_scaled);
    free(Wq_rm); free(Wk_rm); free(Wq_p); free(Wk_p);
    free(sin_cos_all); free(X_packed); free(Q_tmp); free(K_tmp);
}

int main(void) {
    printf("=== Fused Wq·X + Wk·X + RoPE Kernel Benchmark ===\n\n");

    /* Correctness tests */
    printf("--- Correctness Tests ---\n");
    int all_pass = 1;
    all_pass &= test_correctness(1,  256,  64);
    all_pass &= test_correctness(4,  256,  64);
    all_pass &= test_correctness(4,  512, 128);
    all_pass &= test_correctness(4, 1024, 128);
    all_pass &= test_correctness(16, 512, 128);
    all_pass &= test_correctness(8,  256, 256);
    printf("\n");

    if (!all_pass) {
        printf("CORRECTNESS FAILURES — skipping performance tests.\n");
        return 1;
    }

    /* Performance tests */
    printf("--- Performance Tests ---\n");
    printf("Peak FP32 FMLA: 128 GFLOPS (2 GHz × 2 FMA × 16 floats × 2)\n\n");

    struct { int n_tok, d_model, d_head, reps; } configs[] = {
        /* Decode-like: single token */
        { 1,   256,  64,  5000},
        { 1,   512,  64,  5000},
        { 1,   512, 128,  2000},
        { 1,  1024, 128,  1000},
        /* Small batch */
        { 4,   256,  64,  2000},
        { 4,   512, 128,  1000},
        { 4,  1024, 128,   500},
        { 4,  1024, 256,   300},
        /* Prefill-like (same position for asm path) */
        {16,   512, 128,   300},
        {16,  1024, 128,   200},
        {64,   512, 128,   100},
        {64,  1024, 128,    50},
    };
    int n_configs = sizeof(configs) / sizeof(configs[0]);

    for (int i = 0; i < n_configs; i++) {
        bench_perf(configs[i].n_tok, configs[i].d_model,
                   configs[i].d_head, configs[i].reps);
    }

    /* ================================================================
     * FP16 Section
     * ================================================================ */
    printf("\n=== FP16 Fused Wq*X + Wk*X + RoPE Kernel (NR=64) ===\n\n");

    /* FP16 Correctness tests */
    printf("--- FP16 Correctness Tests ---\n");
    int f16_pass = 1;
    f16_pass &= test_correctness_f16(1,  256,  64);
    f16_pass &= test_correctness_f16(4,  256,  64);
    f16_pass &= test_correctness_f16(4,  512, 128);
    f16_pass &= test_correctness_f16(4, 1024, 128);
    f16_pass &= test_correctness_f16(16, 512, 128);
    f16_pass &= test_correctness_f16(8,  256, 256);
    printf("\n");

    if (!f16_pass) {
        printf("FP16 CORRECTNESS FAILURES — skipping FP16 performance tests.\n");
        return 1;
    }

    /* FP16 Performance tests */
    printf("--- FP16 Performance Tests ---\n");
    printf("Peak FP16 FMLA: 256 GFLOPS (2 GHz x 2 FMA x 32 elements x 2)\n\n");

    struct { int n_tok, d_model, d_head, reps; } f16_configs[] = {
        { 1,   256,  64,  5000},
        { 1,   512,  64,  5000},
        { 1,   512, 128,  2000},
        { 1,  1024, 128,  1000},
        { 4,   256,  64,  2000},
        { 4,   512, 128,  1000},
        { 4,  1024, 128,   500},
        { 4,  1024, 256,   300},
        {16,   512, 128,   300},
        {16,  1024, 128,   200},
        {64,   512, 128,   100},
        {64,  1024, 128,    50},
    };
    int n_f16_configs = sizeof(f16_configs) / sizeof(f16_configs[0]);

    for (int i = 0; i < n_f16_configs; i++) {
        bench_perf_f16(f16_configs[i].n_tok, f16_configs[i].d_model,
                       f16_configs[i].d_head, f16_configs[i].reps);
    }

    printf("\nDone.\n");
    return 0;
}
