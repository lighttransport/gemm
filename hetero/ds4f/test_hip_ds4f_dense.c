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
