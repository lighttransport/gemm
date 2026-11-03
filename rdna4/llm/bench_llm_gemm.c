/*
 * bench_llm_gemm.c - correctness + throughput of the self-owned RDNA4 WMMA
 * GEMM (gemm_wmma.hip) on Qwen3.8-27B prefill projection shapes, optionally
 * head-to-head against hipBLASLt (build with HIPBLASLT=1).
 *
 *   Y[M,N] f32 = X[M,K] x W[N,K]^T   (BF16 or F16 inputs)
 *
 * Usage: bench_llm_gemm [--m M[,M...]] [--shape name|all] [--variant i|auto|all]
 *                       [--splitk n] [--iters n] [--rounds n] [--f16] [--no-check]
 *                       [--blaslt]
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../rocew.h"
#define HIP_RUNNER_COMMON_IMPLEMENTATION
#include "../hip_runner_common.h"
#include "gemm_wmma.inc"
#include "gemm_wmma_dispatch.h"

#ifdef LLM_HIPBLASLT_ENABLED
#include "mm_blaslt_bridge.h"
#endif

typedef struct { const char *name; int n, k; } shape_nk;

/* Qwen3.8-27B: n_embd 5120, n_ff 17408, SSM inner 6144, gated Q 12288. */
static const shape_nk g_shapes[] = {
    { "ssm_qkv",   10240,  5120 },
    { "ssm_gate",   6144,  5120 },
    { "ssm_ab",       48,  5120 },
    { "ssm_out",    5120,  6144 },
    { "attn_q",    12288,  5120 },
    { "attn_kv",    1024,  5120 },
    { "attn_o",     5120,  6144 },
    { "ffn_up",    17408,  5120 },
    { "ffn_down",   5120, 17408 },
    { "dflash_fc",  5120, 25600 },
};
#define NSHAPES ((int)(sizeof(g_shapes) / sizeof(g_shapes[0])))

static uint16_t f32_to_bf16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    u += 0x7fff + ((u >> 16) & 1);
    return (uint16_t)(u >> 16);
}
static float bf16_to_f32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f;
}
static float f16_to_f32(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023, u;
    if (e == 0) { float f = ldexpf((float)m, -24); return s ? -f : f; }
    if (e == 31) u = (s << 31) | 0x7f800000 | (m << 13);
    else u = (s << 31) | ((e + 112) << 23) | (m << 13);
    float f; memcpy(&f, &u, 4); return f;
}

static uint64_t rng_state = 0x9e3779b97f4a7c15ULL;
static float frand(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
    return (float)((rng_state >> 40) & 0xffffff) / 16777216.0f * 2.0f - 1.0f;
}

static void fill(uint16_t *p, size_t n, int f16, float scale) {
    for (size_t i = 0; i < n; i++) {
        float v = frand() * scale;
        p[i] = f16 ? hip_f32_to_f16(v) : f32_to_bf16(v);
    }
}

/* Sampled check vs double-precision reference over the rounded inputs. */
static int check(const float *Y, const uint16_t *X, const uint16_t *W, int M, int N,
                 int K, int f16, double *out_cos, double *out_rel) {
    double dot = 0, ny = 0, nr = 0, max_rel = 0;
    int samples = 0;
    long total = (long)M * N;
    long step = total <= 65536 ? 1 : total / 65536;
    for (long t = 0; t < total; t += step) {
        long idx = step == 1 ? t : (t + (t * 7919) % step) % total;
        int m = (int)(idx / N), n = (int)(idx % N);
        double ref = 0, mag = 0;
        for (int k = 0; k < K; k++) {
            double a = f16 ? f16_to_f32(X[(size_t)m * K + k]) : bf16_to_f32(X[(size_t)m * K + k]);
            double b = f16 ? f16_to_f32(W[(size_t)n * K + k]) : bf16_to_f32(W[(size_t)n * K + k]);
            ref += a * b; mag += fabs(a * b);
        }
        double y = Y[idx];
        if (!isfinite(y)) { *out_cos = 0; *out_rel = INFINITY; return 0; }
        dot += y * ref; ny += y * y; nr += ref * ref;
        double rel = fabs(y - ref) / (mag + 1e-30);
        if (rel > max_rel) max_rel = rel;
        samples++;
    }
    *out_cos = dot / (sqrt(ny * nr) + 1e-300);
    *out_rel = max_rel;
    /* |err| relative to sum|a*b| stays ~1e-6 for F32 accumulation. */
    return samples > 0 && *out_cos > 0.999999 && max_rel < 1e-4;
}

static double time_ms(int iters, hipStream_t st, hipEvent_t e0, hipEvent_t e1,
                      int (*fn)(void *), void *ctx) {
    for (int i = 0; i < 10; i++) if (fn(ctx)) return -1;
    hipStreamSynchronize(st);
    hipEventRecord(e0, st);
    for (int i = 0; i < iters; i++) if (fn(ctx)) return -1;
    hipEventRecord(e1, st);
    hipEventSynchronize(e1);
    float ms = 0;
    hipEventElapsedTime(&ms, e0, e1);
    return ms / iters;
}

typedef struct {
    gemm_wmma_ctx *c;
    int dtype, variant, splitk, M, N, K;
    void *y, *w, *x;
    hipStream_t st;
} run_ctx;

static int run_own(void *p) {
    run_ctx *r = p;
    return gemm_wmma_run_ex(r->c, r->dtype, r->y, r->w, r->x, r->M, r->N, r->K,
                            r->st, r->variant, r->splitk);
}
#ifdef LLM_HIPBLASLT_ENABLED
static int run_blaslt(void *p) {
    run_ctx *r = p;
    return r->dtype == GEMM_WMMA_F16
        ? mm_blaslt_run_f16(r->y, r->w, r->x, r->M, r->N, r->K, r->st)
        : mm_blaslt_run_bf16(r->y, r->w, r->x, r->M, r->N, r->K, r->st);
}
#endif

static const int q_block_bytes[GEMM_WMMA_Q_NFMT] = { 66, 74, 82, 98, 110, 50, 56, 136, 84 };

/* Fused-dequant mode: random quant blocks (timing + cross-variant bit
 * equality; exactness vs the runner's dequant kernels is checked in-runner by
 * LLM_GEMM_FUSED_CHECK=1). Every non-split tile variant reduces each output in
 * the same K order, so all variants must agree bit for bit. */
static int bench_quant(gemm_wmma_ctx *ctx, int fmt, const int *ms_list, int n_ms,
                       const char *shape_sel, int iters, int rounds,
                       hipStream_t st, hipEvent_t e0, hipEvent_t e1) {
    int fails = 0;
    double sum_best[GEMM_WMMA_Q_NTILE] = {0};
    for (int mi = 0; mi < n_ms; mi++) {
        int M = ms_list[mi];
        for (int si = 0; si < NSHAPES; si++) {
            const shape_nk *s = &g_shapes[si];
            if (strcmp(shape_sel, "all") && strcmp(shape_sel, s->name)) continue;
            int N = s->n, K = s->k;
            size_t wbytes = (size_t)N * (K / 256) * q_block_bytes[fmt];
            size_t nx = (size_t)M * K, ny = (size_t)M * N;
            unsigned char *hw = malloc(wbytes);
            uint16_t *hx = malloc(nx * 2);
            for (size_t i = 0; i < wbytes; i++) hw[i] = (unsigned char)(frand() * 127.0f + 128.0f);
            /* keep the f16 scale fields finite: clear exponent MSBs of every byte pair */
            for (size_t i = 1; i < wbytes; i += 2) hw[i] &= 0x3b;
            fill(hx, nx, 0, 1.0f);
            void *dx, *dw, *dy;
            HIP_CHECK(hipMalloc(&dx, nx * 2));
            HIP_CHECK(hipMalloc(&dw, wbytes));
            HIP_CHECK(hipMalloc(&dy, ny * 4));
            HIP_CHECK(hipMemcpy(dx, hx, nx * 2, hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(dw, hw, wbytes, hipMemcpyHostToDevice));
            float *ref = malloc(ny * 4), *out = malloc(ny * 4);
            double best[GEMM_WMMA_Q_NTILE];
            for (int t = 0; t < GEMM_WMMA_Q_NTILE; t++) best[t] = 1e30;
            for (int rd = 0; rd < rounds; rd++) {
                for (int t = 0; t < GEMM_WMMA_Q_NTILE; t++) {
                    for (int i = 0; i < 5; i++)
                        gemm_wmma_run_q_ex(ctx, fmt, dy, dw, dx, M, N, K, st, t, 1);
                    hipStreamSynchronize(st);
                    hipEventRecord(e0, st);
                    for (int i = 0; i < iters; i++)
                        gemm_wmma_run_q_ex(ctx, fmt, dy, dw, dx, M, N, K, st, t, 1);
                    hipEventRecord(e1, st);
                    hipEventSynchronize(e1);
                    float ms = 0;
                    hipEventElapsedTime(&ms, e0, e1);
                    if (ms / iters < best[t]) best[t] = ms / iters;
                }
            }
            double flop = 2.0 * M * N * K;
            for (int t = 0; t < GEMM_WMMA_Q_NTILE; t++) {
                HIP_CHECK(hipMemset(dy, 0xff, ny * 4));
                if (gemm_wmma_run_q_ex(ctx, fmt, dy, dw, dx, M, N, K, st, t, 1)) fails++;
                hipStreamSynchronize(st);
                HIP_CHECK(hipMemcpy(t == 0 ? ref : out, dy, ny * 4, hipMemcpyDeviceToHost));
                const char *ok = t == 0 ? "ref" : (memcmp(ref, out, ny * 4) ? "DIFF" : "same");
                if (t > 0 && memcmp(ref, out, ny * 4)) fails++;
                sum_best[t] += best[t];
                printf("q-%-7s %-9s M=%5d N=%5d K=%5d %-10s ms=%8.4f TF/s=%7.2f %s\n",
                       gemm_wmma_q_names[fmt], s->name, M, N, K, gemm_wmma_q_tiles[t].suffix,
                       best[t], flop / best[t] * 1e-9, ok);
            }
            hipFree(dx); hipFree(dw); hipFree(dy);
            free(hw); free(hx); free(ref); free(out);
        }
    }
    printf("# sum");
    for (int t = 0; t < GEMM_WMMA_Q_NTILE; t++)
        printf(" %s=%.4f", gemm_wmma_q_tiles[t].suffix, sum_best[t]);
    printf("\n# %s\n", fails ? "FAIL" : "ALL SAME");
    return fails;
}

int main(int argc, char **argv) {
    int ms_list[16] = { 512 }, n_ms = 1;
    const char *shape_sel = "all", *variant_sel = "auto";
    int splitk = 0, iters = 50, rounds = 3, f16 = 0, do_check = 1, blaslt = 0, qfmt = -1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--m") && i + 1 < argc) {
            n_ms = 0;
            for (char *t = strtok(argv[++i], ","); t && n_ms < 16; t = strtok(NULL, ","))
                ms_list[n_ms++] = atoi(t);
        } else if (!strcmp(argv[i], "--shape") && i + 1 < argc) shape_sel = argv[++i];
        else if (!strcmp(argv[i], "--variant") && i + 1 < argc) variant_sel = argv[++i];
        else if (!strcmp(argv[i], "--splitk") && i + 1 < argc) splitk = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rounds") && i + 1 < argc) rounds = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--f16")) f16 = 1;
        else if (!strcmp(argv[i], "--no-check")) do_check = 0;
        else if (!strcmp(argv[i], "--blaslt")) blaslt = 1;
        else if (!strcmp(argv[i], "--quant") && i + 1 < argc) {
            const char *f = argv[++i];
            for (int q = 0; q < GEMM_WMMA_Q_NFMT; q++)
                if (!strcmp(f, gemm_wmma_q_names[q])) qfmt = q;
            if (qfmt < 0) { fprintf(stderr, "unknown --quant %s\n", f); return 2; }
        }
        else {
            fprintf(stderr, "usage: %s [--m M,..] [--shape name|all] [--variant i|auto|all] "
                    "[--splitk n] [--iters n] [--rounds n] [--f16] [--no-check] [--blaslt] [--quant fmt]\n", argv[0]);
            return 2;
        }
    }
#ifndef LLM_HIPBLASLT_ENABLED
    if (blaslt) { fprintf(stderr, "--blaslt needs a HIPBLASLT=1 build\n"); return 2; }
#endif
    if (rocewInit(ROCEW_INIT_HIP | ROCEW_INIT_HIPRTC) != ROCEW_SUCCESS) {
        fprintf(stderr, "failed to initialize ROCm/HIP runtime\n");
        return 1;
    }
    HIP_CHECK(hipInit(0));
    HIP_CHECK(hipSetDevice(0));
    gemm_wmma_ctx ctx;
    if (gemm_wmma_init(&ctx, 0, 0) != 0) { fprintf(stderr, "gemm_wmma_init failed\n"); return 1; }
#ifdef LLM_HIPBLASLT_ENABLED
    if (blaslt && mm_blaslt_init() != 0) { fprintf(stderr, "mm_blaslt_init failed\n"); return 1; }
#endif
    int dtype = f16 ? GEMM_WMMA_F16 : GEMM_WMMA_BF16;
    hipStream_t st; hipEvent_t e0, e1;
    HIP_CHECK(hipStreamCreate(&st));
    HIP_CHECK(hipEventCreate(&e0));
    HIP_CHECK(hipEventCreate(&e1));
    printf("# gemm_wmma bench dtype=%s cu=%d iters=%d (peak ~195 TFLOP/s dense bf16)\n",
           f16 ? "f16" : "bf16", ctx.n_cu, iters);
    if (qfmt >= 0) {
        int r = bench_quant(&ctx, qfmt, ms_list, n_ms, shape_sel, iters, rounds, st, e0, e1);
        gemm_wmma_destroy(&ctx);
        return r ? 1 : 0;
    }
    int fails = 0;
    double sum_own = 0, sum_bl = 0;
    for (int mi = 0; mi < n_ms; mi++) {
        int M = ms_list[mi];
        for (int si = 0; si < NSHAPES; si++) {
            const shape_nk *s = &g_shapes[si];
            if (strcmp(shape_sel, "all") && strcmp(shape_sel, s->name)) continue;
            int N = s->n, K = s->k;
            size_t nx = (size_t)M * K, nw = (size_t)N * K, ny = (size_t)M * N;
            uint16_t *hx = malloc(nx * 2), *hw = malloc(nw * 2);
            float *hy = malloc(ny * 4);
            fill(hx, nx, f16, 1.0f);
            fill(hw, nw, f16, 0.05f);
            void *dx, *dw, *dy;
            HIP_CHECK(hipMalloc(&dx, nx * 2));
            HIP_CHECK(hipMalloc(&dw, nw * 2));
            HIP_CHECK(hipMalloc(&dy, ny * 4));
            HIP_CHECK(hipMemcpy(dx, hx, nx * 2, hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(dw, hw, nw * 2, hipMemcpyHostToDevice));
            double flop = 2.0 * M * N * K;
            run_ctx rc = { &ctx, dtype, -1, splitk, M, N, K, dy, dw, dx, st };
            /* Candidates: own variants (or the auto choice) + hipBLASLt (-2).
             * Timed interleaved over `rounds`, keeping each one's minimum, so
             * clock/power drift at the 212 W cap hits all of them alike. */
            int cand_v[GEMM_WMMA_NVARIANTS + 1], cand_s[GEMM_WMMA_NVARIANTS + 1], nc = 0;
            for (int v = 0; v < GEMM_WMMA_NVARIANTS; v++) {
                int want = !strcmp(variant_sel, "all") ||
                           (strcmp(variant_sel, "auto") && atoi(variant_sel) == v);
                if (want && gemm_wmma_variants[v].dtype == dtype) {
                    cand_v[nc] = v; cand_s[nc] = splitk > 0 ? splitk : 1; nc++;
                }
            }
            if (!strcmp(variant_sel, "auto")) {
                gemm_wmma_select(&ctx, dtype, M, N, K, &cand_v[nc], &cand_s[nc]);
                if (splitk > 0) cand_s[nc] = splitk;
                nc++;
            }
            if (blaslt) { cand_v[nc] = -2; cand_s[nc] = 0; nc++; }
            double best[GEMM_WMMA_NVARIANTS + 1];
            for (int c = 0; c < nc; c++) best[c] = 1e30;
            for (int rd = 0; rd < rounds; rd++) {
                for (int c = 0; c < nc; c++) {
                    rc.variant = cand_v[c]; rc.splitk = cand_s[c];
#ifdef LLM_HIPBLASLT_ENABLED
                    double ms = cand_v[c] == -2 ? time_ms(iters, st, e0, e1, run_blaslt, &rc)
                                                : time_ms(iters, st, e0, e1, run_own, &rc);
#else
                    double ms = time_ms(iters, st, e0, e1, run_own, &rc);
#endif
                    if (ms < 0) { best[c] = -1; continue; }
                    if (best[c] > 0 && ms < best[c]) best[c] = ms;
                }
            }
            double best_own = 1e30, bl_ms = 0;
            for (int c = 0; c < nc; c++) {
                rc.variant = cand_v[c]; rc.splitk = cand_s[c];
                const char *ok = "-";
                double cs = 0, rel = 0, ms = best[c];
                if (ms < 0) { ok = "LAUNCH-FAIL"; fails++; }
                else if (do_check) {
                    HIP_CHECK(hipMemset(dy, 0xff, ny * 4));
#ifdef LLM_HIPBLASLT_ENABLED
                    if (cand_v[c] == -2) run_blaslt(&rc); else
#endif
                    run_own(&rc);
                    HIP_CHECK(hipMemcpy(hy, dy, ny * 4, hipMemcpyDeviceToHost));
                    int pass = check(hy, hx, hw, M, N, K, f16, &cs, &rel);
                    ok = pass ? "PASS" : "FAIL";
                    if (!pass && cand_v[c] != -2) fails++;
                }
                if (cand_v[c] == -2) bl_ms = ms;
                else if (ms > 0 && ms < best_own) best_own = ms;
                printf("%-6s %-9s M=%5d N=%5d K=%5d %-28s sk=%d ms=%8.4f TF/s=%7.2f cos=%.9f rel=%.2e %s\n",
                       cand_v[c] == -2 ? "blaslt" : "own", s->name, M, N, K,
                       cand_v[c] == -2 ? "hipBLASLt" : gemm_wmma_variants[cand_v[c]].name,
                       cand_s[c], ms, flop / ms * 1e-9, cs, rel, ok);
            }
            sum_own += best_own;
            sum_bl += bl_ms;
            hipFree(dx); hipFree(dw); hipFree(dy);
            free(hx); free(hw); free(hy);
        }
    }
    printf("# sum best own ms=%.4f", sum_own);
    if (blaslt) printf(" blaslt ms=%.4f own/blaslt=%.3f", sum_bl, sum_own / sum_bl);
    printf("\n# %s\n", fails ? "FAIL" : "ALL PASS");
    gemm_wmma_destroy(&ctx);
    return fails ? 1 : 0;
}
