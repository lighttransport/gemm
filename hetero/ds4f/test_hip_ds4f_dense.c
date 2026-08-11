/* CPU-vs-HIPRTC correctness check for the S3 FP8/E8M0 matvec slice. */

#include "hip_ds4f_dense.h"
#include "../../common/ds4f.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double wall_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static float bits_to_f32(uint32_t bits) {
    float v;
    memcpy(&v, &bits, sizeof(v));
    return v;
}

static float fp8_e4m3fn_to_f32(uint8_t v) {
    uint32_t sign = ((uint32_t)v >> 7) & 1u;
    uint32_t exp = ((uint32_t)v >> 3) & 15u;
    uint32_t mant = (uint32_t)v & 7u;
    if (exp == 0) {
        if (mant == 0) return bits_to_f32(sign << 31);
        int sh = 0;
        while ((mant & 4u) == 0) { mant <<= 1; ++sh; }
        mant &= 3u;
        return bits_to_f32((sign << 31) | ((uint32_t)(120 - sh) << 23) | (mant << 20));
    }
    if (exp == 15 && mant == 7)
        return bits_to_f32((sign << 31) | (255u << 23) | (1u << 22));
    return bits_to_f32((sign << 31) | ((exp + 120u) << 23) | (mant << 20));
}

static float e8m0_to_f32(uint8_t v) {
    return bits_to_f32((uint32_t)v << 23);
}

static void cpu_matvec(float *y, const uint8_t *w, const uint8_t *s,
                       const float *x, int rows, int cols) {
    int scale_cols = (cols + 127) / 128;
    for (int r = 0; r < rows; ++r) {
        const uint8_t *wr = w + (size_t)r * (size_t)cols;
        const uint8_t *sr = s + (size_t)(r >> 7) * (size_t)scale_cols;
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c)
            sum += (fp8_e4m3fn_to_f32(wr[c]) * e8m0_to_f32(sr[c >> 7])) * x[c];
        y[r] = sum;
    }
}

int main(void) {
    const int rows = 131;
    const int cols = 259;
    const int scale_cols = (cols + 127) / 128;
    const int scale_rows = (rows + 127) / 128;
    const size_t w_bytes = (size_t)rows * (size_t)cols;
    const size_t s_bytes = (size_t)scale_rows * (size_t)scale_cols;

    ds4f_mem_pool *mem = ds4f_mem_pool_create();
    if (!mem) {
        fprintf(stderr, "test_hip_ds4f_dense: memory pool creation failed\n");
        return 1;
    }
    uint8_t *w = (uint8_t *)ds4f_mem_alloc(mem, w_bytes, 64, 0);
    uint8_t *s = (uint8_t *)ds4f_mem_alloc(mem, s_bytes, 64, 0);
    float *x = (float *)ds4f_mem_alloc(mem, (size_t)cols * sizeof(float), 64, 0);
    float *ref = (float *)ds4f_mem_alloc(mem, (size_t)rows * sizeof(float), 64, 0);
    float *got = (float *)ds4f_mem_alloc(mem, (size_t)rows * sizeof(float), 64, 0);
    if (!w || !s || !x || !ref || !got) {
        fprintf(stderr, "test_hip_ds4f_dense: host allocation failed\n");
        ds4f_mem_pool_destroy(mem);
        return 1;
    }

    /* Exercise zero, subnormal, negative, normal, and finite exp=15 codes.
     * 0xff (the one E4M3FN NaN code) is intentionally excluded. */
    static const uint8_t codes[] = {
        0x00, 0x01, 0x02, 0x07, 0x08, 0x11, 0x28, 0x55,
        0x80, 0x81, 0x9a, 0xc0, 0xe3, 0xf0, 0xf6
    };
    for (size_t i = 0; i < w_bytes; ++i)
        w[i] = codes[(i * 13u + i / 17u) % (sizeof(codes) / sizeof(codes[0]))];
    for (int r = 0; r < scale_rows; ++r)
        for (int c = 0; c < scale_cols; ++c)
            s[(size_t)r * (size_t)scale_cols + c] = (uint8_t)(124 + ((r * 3 + c) % 7));
    for (int c = 0; c < cols; ++c)
        x[c] = ((float)((c * 29) % 101) - 50.0f) / 37.0f;

    cpu_matvec(ref, w, s, x, rows, cols);
    hip_ds4f_dense *ctx = hip_ds4f_dense_create(0, 1);
    if (!ctx) {
        printf("SKIP: HIP/ROCm device or hipRTC unavailable\n");
        ds4f_mem_pool_destroy(mem);
        return 0;
    }
    int rc = hip_ds4f_dense_load(ctx, w, s, rows, cols);
    if (rc == 0) rc = hip_ds4f_dense_matvec_loaded(ctx, x, got);
    if (rc == 0) {
        enum { ITERS = 50 };
        double t0 = wall_seconds();
        for (int i = 0; i < ITERS && rc == 0; ++i)
            rc = hip_ds4f_dense_matvec_loaded(ctx, x, got);
        double t1 = wall_seconds();
        if (rc == 0)
            printf("persistent matvec: %.3f ms/call\n",
                   (t1 - t0) * 1000.0 / ITERS);

        t0 = wall_seconds();
        for (int i = 0; i < ITERS && rc == 0; ++i) {
            rc = hip_ds4f_dense_matvec_loaded_async(ctx, x);
            if (rc == 0) rc = hip_ds4f_dense_wait(ctx, got);
        }
        t1 = wall_seconds();
        if (rc == 0)
            printf("async matvec + wait: %.3f ms/call\n",
                   (t1 - t0) * 1000.0 / ITERS);
    }
    if (rc == 0) {
        enum { M = 19, XSTRIDE = cols + 5, YSTRIDE = rows + 7 };
        float *bx = (float *)ds4f_mem_alloc(mem, (size_t)M * XSTRIDE * sizeof(float), 64, 0);
        float *by = (float *)ds4f_mem_alloc(mem, (size_t)M * YSTRIDE * sizeof(float), 64, 0);
        float *br = (float *)ds4f_mem_alloc(mem, (size_t)M * rows * sizeof(float), 64, 0);
        ds4f_tensor wt = { w, s, DS4F_FP8, rows, cols, rc };
        if (!bx || !by || !br) rc = -1;
        if (rc == 0) {
            for (int mm = 0; mm < M; mm++) {
                for (int c = 0; c < XSTRIDE; c++)
                    bx[(size_t)mm * XSTRIDE + c] = c < cols
                        ? ((float)((c * 17 + mm * 11) % 97) - 48.0f) / 31.0f : -7.0f;
                cpu_matvec(br + (size_t)mm * rows, w, s,
                           bx + (size_t)mm * XSTRIDE, rows, cols);
            }
            rc = hip_ds4f_dense_gemm_tensor(ctx, by, &wt, bx, M, YSTRIDE, XSTRIDE);
            float max_gemm_abs = 0.0f, max_gemm_rel = 0.0f;
            if (rc == 0) for (int mm = 0; mm < M; mm++) for (int r = 0; r < rows; r++) {
                float e = fabsf(by[(size_t)mm * YSTRIDE + r] - br[(size_t)mm * rows + r]);
                if (e > max_gemm_abs) max_gemm_abs = e;
                float q = e / fmaxf(1.0f, fabsf(br[(size_t)mm * rows + r]));
                if (q > max_gemm_rel) max_gemm_rel = q;
            }
            printf("HIP FP8 batched GEMM: M=%d rows=%d cols=%d max_abs=%.8g max_rel=%.8g %s\n",
                   M, rows, cols, max_gemm_abs, max_gemm_rel,
                   rc == 0 && max_gemm_rel <= 3.0e-4f && max_gemm_abs <= 3.0e-2f ? "PASS" : "FAIL");
            if (rc == 0 && (max_gemm_rel > 3.0e-4f || max_gemm_abs > 3.0e-2f)) rc = -1;
        }
        if (rc == 0) {
            enum { BM = 13, BROWS = 73, BCOLS = 37,
                   BXSTRIDE = BCOLS + 3, BYSTRIDE = BROWS + 5 };
            uint16_t *bw = (uint16_t *)ds4f_mem_alloc(
                mem, (size_t)BROWS * BCOLS * sizeof(*bw), 64, 0);
            float *bx16 = (float *)ds4f_mem_alloc(
                mem, (size_t)BM * BXSTRIDE * sizeof(*bx16), 64, 0);
            float *by16 = (float *)ds4f_mem_alloc(
                mem, (size_t)BM * BYSTRIDE * sizeof(*by16), 64, 0);
            float *br16 = (float *)ds4f_mem_alloc(
                mem, (size_t)BM * BROWS * sizeof(*br16), 64, 0);
            ds4f_tensor bwt = { bw, NULL, DS4F_BF16, BROWS, BCOLS, -1 };
            if (!bw || !bx16 || !by16 || !br16) {
                rc = -1;
            } else {
                for (int r = 0; r < BROWS; r++) for (int c = 0; c < BCOLS; c++)
                    bw[(size_t)r * BCOLS + c] = ds4f_f32bf(
                        ((float)((r * 19 + c * 7) % 83) - 41.0f) / 23.0f);
                rc = hip_ds4f_dense_bind_bf16_tensor(ctx, &bwt) < 0 ? -1 : 0;
            }
            if (rc == 0) {
                for (int mm = 0; mm < BM; mm++) {
                    for (int c = 0; c < BXSTRIDE; c++)
                        bx16[(size_t)mm * BXSTRIDE + c] = c < BCOLS
                            ? ((float)((c * 23 + mm * 5) % 71) - 35.0f) / 17.0f
                            : -3.0f;
                    for (int r = 0; r < BROWS; r++) {
                        float sum = 0.0f;
                        for (int c = 0; c < BCOLS; c++)
                            sum += ds4f_bf16f(bw[(size_t)r * BCOLS + c]) *
                                   bx16[(size_t)mm * BXSTRIDE + c];
                        br16[(size_t)mm * BROWS + r] = sum;
                    }
                }
                rc = hip_ds4f_dense_gemm_tensor(ctx, by16, &bwt, bx16,
                                                 BM, BYSTRIDE, BXSTRIDE);
                float max_bf_abs = 0.0f, max_bf_rel = 0.0f;
                if (rc == 0) for (int mm = 0; mm < BM; mm++) for (int r = 0; r < BROWS; r++) {
                    float e = fabsf(by16[(size_t)mm * BYSTRIDE + r] -
                                    br16[(size_t)mm * BROWS + r]);
                    if (e > max_bf_abs) max_bf_abs = e;
                    float q = e / fmaxf(1.0f, fabsf(br16[(size_t)mm * BROWS + r]));
                    if (q > max_bf_rel) max_bf_rel = q;
                }
                printf("HIP BF16 batched GEMM: M=%d rows=%d cols=%d max_abs=%.8g max_rel=%.8g %s\n",
                       BM, BROWS, BCOLS, max_bf_abs, max_bf_rel,
                       rc == 0 && max_bf_rel <= 3.0e-5f && max_bf_abs <= 3.0e-3f ? "PASS" : "FAIL");
                if (rc == 0 && (max_bf_rel > 3.0e-5f || max_bf_abs > 3.0e-3f)) rc = -1;
            }
        }
    }
    if (rc == 0) {
        /* ds4f_dense_prefill_attn_partial must agree with the existing,
         * already-shipped ds4f_dense_prefill_attn once its unnormalized
         * output is divided by the sum it reports -- same math, only the
         * final normalize step is deferred so a caller can merge in another
         * term first. Pure window case (no second term), so this isolates
         * the new kernel/wrapper from the merge logic that will use it. */
        const int aM = 1, aHeads = 8, aHD = 64, aKVD = 64, aWin = 16, aSlots = 32, aPos0 = 10;
        float *aq = (float *)malloc((size_t)aM * aHeads * aHD * sizeof(float));
        uint16_t *akv = (uint16_t *)malloc((size_t)aSlots * aKVD * sizeof(uint16_t));
        float *asink = (float *)malloc((size_t)aHeads * sizeof(float));
        float *ay_old = (float *)malloc((size_t)aM * aHeads * aHD * sizeof(float));
        float *ay_new = (float *)malloc((size_t)aM * aHeads * aHD * sizeof(float));
        float *ay_max = (float *)malloc((size_t)aM * aHeads * sizeof(float));
        float *ay_sum = (float *)malloc((size_t)aM * aHeads * sizeof(float));
        if (aq && akv && asink && ay_old && ay_new && ay_max && ay_sum) {
            for (int i = 0; i < aM * aHeads * aHD; i++)
                aq[i] = ((float)((i * 37 + 11) % 97) - 48.0f) / 19.0f;
            for (int i = 0; i < aSlots * aKVD; i++)
                akv[i] = ds4f_f32bf(((float)((i * 53 + 7) % 89) - 44.0f) / 21.0f);
            for (int i = 0; i < aHeads; i++) asink[i] = ((float)((i * 13) % 7) - 3.0f) / 5.0f;
            int rc_old = hip_ds4f_dense_prefill_attention(ctx, ay_old, aq, akv, asink,
                NULL, NULL, 0, 0, aM, aPos0, aHeads, aHD, aKVD, aSlots, aWin, 0.125f);
            int rc_new = hip_ds4f_dense_prefill_attention_partial(ctx, ay_new, ay_max, ay_sum,
                aq, akv, asink, aM, aPos0, aHeads, aHD, aKVD, aSlots, aWin, 0.125f);
            if (rc_old != 0 || rc_new != 0) {
                printf("HIP attn partial-vs-full: rc_old=%d rc_new=%d FAIL (call failed)\n", rc_old, rc_new);
                rc = -1;
            } else {
                float amax_abs = 0.0f, amax_rel = 0.0f;
                for (int h = 0; h < aHeads; h++) {
                    float inv = 1.0f / ay_sum[h];
                    for (int d = 0; d < aHD; d++) {
                        float got = ay_new[h * aHD + d] * inv;
                        float ref = ay_old[h * aHD + d];
                        float e = fabsf(got - ref);
                        if (e > amax_abs) amax_abs = e;
                        float r = e / fmaxf(1.0f, fabsf(ref));
                        if (r > amax_rel) amax_rel = r;
                    }
                }
                int apass = amax_abs <= 1.0e-5f && amax_rel <= 1.0e-5f;
                printf("HIP attn partial-vs-full: max_abs=%.8g max_rel=%.8g %s\n",
                       amax_abs, amax_rel, apass ? "PASS" : "FAIL");
                if (!apass) rc = -1;

                /* Same check at pos0 >= kv_slots, forcing ring-buffer
                 * wraparound in the partial kernel's modular slot indexing --
                 * exactly the scenario a naive linear base/end window (rather
                 * than always copying the full ring for M==1) would silently
                 * get wrong for any tier-B2 decode position past the first
                 * window_size tokens. The unmodified reference function
                 * doesn't support this case in its simple (non-resident_kv)
                 * mode, so compare against a direct CPU computation instead. */
                int aPos1 = aPos0 + aSlots + 3;
                int rc_new2 = hip_ds4f_dense_prefill_attention_partial(ctx, ay_new, ay_max, ay_sum,
                    aq, akv, asink, aM, aPos1, aHeads, aHD, aKVD, aSlots, aWin, 0.125f);
                if (rc_new2 != 0) {
                    printf("HIP attn partial-vs-full (wrap): rc_new=%d FAIL (call failed)\n", rc_new2);
                    rc = -1;
                } else {
                    int wlo = aPos1 - aWin + 1; if (wlo < 0) wlo = 0;
                    int wnp = aPos1 - wlo + 1;
                    float *wref = (float *)malloc((size_t)aHeads * aHD * sizeof(float));
                    float *wsc = (float *)malloc((size_t)wnp * sizeof(float));
                    for (int h = 0; h < aHeads && wref && wsc; h++) {
                        float mx = -1e30f;
                        for (int j = 0; j < wnp; j++) {
                            int slot = (wlo + j) % aSlots;
                            float dot = 0.0f;
                            for (int d = 0; d < aKVD; d++) {
                                uint32_t bits = (uint32_t)akv[slot * aKVD + d] << 16;
                                float kv_f; memcpy(&kv_f, &bits, sizeof(kv_f));
                                dot += aq[h * aHD + d] * kv_f;
                            }
                            dot *= 0.125f; wsc[j] = dot; if (dot > mx) mx = dot;
                        }
                        float den = expf(asink[h] - mx);
                        for (int j = 0; j < wnp; j++) { wsc[j] = expf(wsc[j] - mx); den += wsc[j]; }
                        float inv = 1.0f / den;
                        for (int d = 0; d < aHD; d++) {
                            float out = 0.0f;
                            if (d < aKVD) for (int j = 0; j < wnp; j++) {
                                int slot = (wlo + j) % aSlots;
                                uint32_t bits = (uint32_t)akv[slot * aKVD + d] << 16;
                                float kv_f; memcpy(&kv_f, &bits, sizeof(kv_f));
                                out += wsc[j] * inv * kv_f;
                            }
                            wref[h * aHD + d] = out;
                        }
                    }
                    float wmax_abs = 0.0f, wmax_rel = 0.0f;
                    for (int h = 0; wref && h < aHeads; h++) {
                        float inv = 1.0f / ay_sum[h];
                        for (int d = 0; d < aHD; d++) {
                            float got = ay_new[h * aHD + d] * inv;
                            float ref = wref[h * aHD + d];
                            float e = fabsf(got - ref);
                            if (e > wmax_abs) wmax_abs = e;
                            float r = e / fmaxf(1.0f, fabsf(ref));
                            if (r > wmax_rel) wmax_rel = r;
                        }
                    }
                    int wpass = wref && wmax_abs <= 1.0e-5f && wmax_rel <= 1.0e-5f;
                    printf("HIP attn partial-vs-full (wrap): pos0=%d max_abs=%.8g max_rel=%.8g %s\n",
                           aPos1, wmax_abs, wmax_rel, wpass ? "PASS" : "FAIL");
                    if (!wpass) rc = -1;
                    free(wref); free(wsc);
                }
            }
        } else {
            printf("HIP attn partial-vs-full: allocation failed FAIL\n");
            rc = -1;
        }
        free(aq); free(akv); free(asink); free(ay_old); free(ay_new); free(ay_max); free(ay_sum);
    }
    hip_ds4f_dense_destroy(ctx);
    if (rc != 0) {
        ds4f_mem_pool_destroy(mem);
        return 1;
    }

    float max_abs = 0.0f, max_rel = 0.0f;
    int worst = 0;
    for (int r = 0; r < rows; ++r) {
        float abs_err = fabsf(got[r] - ref[r]);
        float rel_err = abs_err / fmaxf(1.0f, fabsf(ref[r]));
        if (abs_err > max_abs) { max_abs = abs_err; worst = r; }
        if (rel_err > max_rel) max_rel = rel_err;
    }
    printf("HIP FP8/E8M0 matvec: rows=%d cols=%d max_abs=%.8g max_rel=%.8g worst_row=%d\n",
           rows, cols, max_abs, max_rel, worst);

    /* The device reduction is intentionally tree-ordered, while the CPU
     * reference is serial.  Keep the bound tight enough to catch wrong byte
     * layout, scale indexing, or FP8 semantics without requiring bit identity. */
    int pass = max_rel <= 3.0e-5f && max_abs <= 3.0e-3f;
    printf("%s\n", pass ? "PASS" : "FAIL");
    ds4f_mem_pool_destroy(mem);
    return pass ? 0 : 1;
}
