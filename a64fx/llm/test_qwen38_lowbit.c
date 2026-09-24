#include "qwen38_lowbit.h"
#include <float.h>
#include <fenv.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL line=%d: %s\n",__LINE__,#x); return 0; } } while (0)

static int encodings(void) {
    CHECK(sizeof(q38_fp4_tile) == 288 && sizeof(q38_fp6_tile) == 400);
    for (int c = 0; c < 64; c++) {
        uint8_t got = 255;
        float value = q38_fp6_decode((uint8_t)c);
        CHECK(q38_fp6_encode(value, &got) && got == c);
        CHECK(fabsf(value) <= 7.5f && value * 8 == truncf(value * 8));
    }
    for (int c = 0; c < 31; c++) {
        float midpoint = (q38_fp6_decode((uint8_t)c) + q38_fp6_decode((uint8_t)(c+1))) * .5f;
        uint8_t got;
        CHECK(q38_fp6_encode(midpoint, &got) && got == (c & 1 ? c + 1 : c));
        CHECK(q38_fp6_encode(-midpoint, &got) && got == ((c & 1 ? c + 1 : c) | 32));
    }
    uint8_t code = 99;
    CHECK(!q38_fp6_encode(NAN, &code) && code == 99);
    CHECK(!q38_fp6_encode(INFINITY, &code));
    CHECK(q38_fp6_encode(FLT_MAX, &code) && code == 31);
    CHECK(q38_fp6_encode(-FLT_MAX, &code) && code == 63);
    CHECK(q38_fp6_encode(FLT_TRUE_MIN, &code) && code == 0);
    CHECK(!q38_lowbit_bytes(0,8,64) && !q38_lowbit_bytes(1,0,64));
    CHECK(q38_lowbit_bytes(2,9,65) == 1600);
    return 1;
}

static int matrix(int format, int rows, int cols) {
    size_t bytes = q38_lowbit_bytes(format, rows, cols);
    uint8_t *weights = malloc(bytes + 16);
    float *src = malloc((size_t)rows * cols * sizeof(float));
    float *decoded = malloc((size_t)cols * sizeof(float));
    float *x = malloc((size_t)cols * sizeof(float));
    float *gold = malloc((size_t)rows * sizeof(float));
    float *got = malloc((size_t)(rows + 1) * sizeof(float));
    size_t na = ((size_t)cols + 15) / 16;
    q38_lowbit_act *act = malloc(na * sizeof(*act));
    uint8_t *raw = NULL;
    CHECK(weights && src && decoded && x && gold && got && act);
    memset(weights + bytes, 0x5a, 16);
    for (int r = 0; r < rows; r++)
        for (int k = 0; k < cols; k++)
            src[(size_t)r * cols + k] = ldexpf(q38_fp6_decode((uint8_t)(k % 64)), r % 7 - 9);
    if (format == Q38_LB_FP6_E2M3) {
        CHECK(q38_lowbit_pack_fp6(weights, bytes, src, cols, rows, cols));
        CHECK(!q38_lowbit_pack_fp6(weights, bytes-1, src, cols, rows, cols));
        src[0] = NAN;
        CHECK(!q38_lowbit_pack_fp6(weights, bytes, src, cols, rows, cols));
        src[0] = 0;
    } else {
        size_t row_bytes = (size_t)(cols / 64) * 36;
        raw = malloc((size_t)rows * row_bytes);
        CHECK(raw);
        const int coeff[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
        for (int r = 0; r < rows; r++)
            for (int b = 0; b < cols / 64; b++) {
                uint8_t *q = raw + (size_t)r * row_bytes + b * 36;
                for (int s = 0; s < 4; s++) {
                    q[s] = (uint8_t)((r * 13 + b * 4 + s) % 256);
                    int e = (q[s] >> 3) & 15, m = q[s] & 7;
                    double scale = (e ? (1.0 + m / 8.0) * pow(2, e - 7) : m * pow(2, -9)) * .5;
                    if (q[s] == 127) scale = 0;
                    for (int j = 0; j < 8; j++) {
                        uint8_t z = (uint8_t)(r * 31 + b * 17 + s * 8 + j);
                        q[4 + s * 8 + j] = z;
                        src[(size_t)r * cols + b * 64 + s * 16 + j] = (float)(coeff[z & 15] * scale);
                        src[(size_t)r * cols + b * 64 + s * 16 + j + 8] = (float)(coeff[z >> 4] * scale);
                    }
                }
            }
        CHECK(q38_lowbit_pack_nvfp4(weights, bytes, raw, row_bytes, rows, cols));
        CHECK(!q38_lowbit_pack_nvfp4(weights, bytes, raw, row_bytes-1, rows, cols));
    }
    CHECK(q38_lowbit_validate(weights, bytes, format, rows, cols));
    CHECK(!q38_lowbit_validate(weights, bytes-1, format, rows, cols));
    for (int r = 0; r < rows; r++) {
        CHECK(q38_lowbit_dequant_row(decoded, weights, format, rows, cols, r));
        for (int k = 0; k < cols; k++) CHECK(decoded[k] == src[(size_t)r * cols + k]);
    }
    for (int k = 0; k < cols; k++) x[k] = (float)((k * 137) % 259 - 129) * 0.015625f;
    CHECK(q38_lowbit_reference(gold, weights, format, x, rows, cols));
    for (int r = 0; r < rows; r++) {
        double expected = 0;
        for (int k = 0; k < cols; k++) expected += (double)src[(size_t)r * cols + k] * x[k];
        CHECK(gold[r] == (float)expected);
    }
#ifdef __ARM_FEATURE_SVE
    CHECK(q38_lowbit_sve_f32(got, weights, format, x, rows, cols));
    for (int r = 0; r < rows; r++) {
        double bound = 0;
        for (int k = 0; k < cols; k++) bound += fabs((double)src[(size_t)r * cols + k] * x[k]);
        CHECK(isfinite(got[r]) && fabs(got[r] - gold[r]) <= 2e-6 * (1 + bound));
    }
#endif
    for (int mode = 8; mode <= 16; mode += 8) {
        CHECK(q38_lowbit_prepare(act, na, x, cols, mode));
        CHECK(!q38_lowbit_prepare(act, na-1, x, cols, mode));
#ifdef __ARM_FEATURE_SVE
        q38_lowbit_act *vector_act = malloc(na * sizeof(*vector_act));
        CHECK(vector_act && q38_lowbit_prepare_sve(vector_act, na, x, cols, mode));
        CHECK(!memcmp(act, vector_act, na * sizeof(*act)));
        CHECK(!q38_lowbit_prepare_sve(vector_act, na-1, x, cols, mode));
        free(vector_act);
#endif
        CHECK(q38_lowbit_dot(gold, weights, format, act, mode, rows, cols));
        got[rows] = 12345;
#ifdef __ARM_FEATURE_SVE
        CHECK(q38_lowbit_sve(got, weights, format, act, mode, rows, cols));
#else
        CHECK(q38_lowbit_dot(got, weights, format, act, mode, rows, cols));
#endif
        for (int r = 0; r < rows; r++) {
            double bound = 0;
            for (int k = 0; k < cols; k++) bound += fabs((double)src[(size_t)r * cols + k] * x[k]);
            CHECK(isfinite(got[r]) && fabs(got[r]-gold[r]) <= 2e-6 * (1 + bound));
        }
        CHECK(got[rows] == 12345);
    }
    for (int i = 0; i < 16; i++) CHECK(weights[bytes+i] == 0x5a);
    if (format == Q38_LB_FP6_E2M3) {
        ((q38_fp6_tile *)weights)->scale[0][0] = 255;
        CHECK(!q38_lowbit_validate(weights, bytes, format, rows, cols));
    }
    free(weights); free(src); free(decoded); free(x); free(gold); free(got); free(act); free(raw);
    printf("lowbit format=%d rows=%d cols=%d PASS\n", format, rows, cols);
    return 1;
}

static int activations(void) {
    float x[64] = {0};
    q38_lowbit_act a[4], b[4];
    x[0] = FLT_MAX; x[1] = -FLT_MAX; x[32] = FLT_TRUE_MIN; x[33] = -FLT_TRUE_MIN;
    for (int mode = 8; mode <= 16; mode += 8) {
        CHECK(q38_lowbit_prepare(a, 4, x, 64, mode));
#ifdef __ARM_FEATURE_SVE
        CHECK(q38_lowbit_prepare_sve(b, 4, x, 64, mode));
        CHECK(!memcmp(a, b, sizeof(a)));
#endif
        for (int i = 0; i < 4; i++) CHECK(isfinite(a[i].scale) && a[i].scale > 0);
        x[0] = NAN;
        CHECK(!q38_lowbit_prepare(a, 4, x, 64, mode));
#ifdef __ARM_FEATURE_SVE
        CHECK(!q38_lowbit_prepare_sve(b, 4, x, 64, mode));
#endif
        x[0] = FLT_MAX;
    }
    for (int i = 0; i < 64; i++) x[i] = (i-32) * 0.03125f;
    CHECK(q38_lowbit_prepare(a, 4, x, 64, 16));
    CHECK(fesetround(FE_DOWNWARD) == 0);
    CHECK(q38_lowbit_prepare(b, 4, x, 64, 16));
    CHECK(fesetround(FE_TONEAREST) == 0);
    /* Digit rounding is defined, but the stored FP32 scale follows FP mode;
     * do not require identical digits when that input scale changes. */
    for (int k = 0; k < 64; k++) {
        const q38_lowbit_act *p = &a[k/16];
        int j = k%8, h = k%16/8;
        int q = p->lo[h][j] + 256*p->hi[h][j];
        CHECK(abs(q) <= 32639);
        CHECK(fabs(q * (double)p->scale - x[k]) <= p->scale * .501);
    }
    return 1;
}

#ifdef __ARM_FEATURE_SVE
static int vector_activation_rounding(void) {
    float x[129];
    q38_lowbit_act scalar[9], vector[9];
    uint32_t random = 0x83419ac7;
    for (int mode = 8; mode <= 16; mode += 8) {
        for (int trial = 0; trial < 512; trial++) {
            int cols = 1 + trial % 129;
            size_t count = ((size_t)cols + 15) / 16;
            for (int k = 0; k < cols; k++) {
                random = random * 1664525u + 1013904223u;
                uint32_t bits = random & UINT32_C(0xfeffffff); /* finite */
                memcpy(x + k, &bits, 4);
            }
            CHECK(q38_lowbit_prepare(scalar, count, x, cols, mode));
            CHECK(q38_lowbit_prepare_sve(vector, count, x, cols, mode));
            CHECK(!memcmp(scalar, vector, count * sizeof(*scalar)));
        }
        int limit = mode == 8 ? 127 : 32639;
        for (int base = -limit; base < limit; base += 31) {
            x[0] = (float)limit; /* exact unit scale */
            for (int k = 1; k < 32; k++) {
                float v = (float)(base + k - 1) + .5f;
                x[k] = v < limit ? v : (float)limit;
            }
            CHECK(q38_lowbit_prepare(scalar, 2, x, 32, mode));
            CHECK(q38_lowbit_prepare_sve(vector, 2, x, 32, mode));
            CHECK(!memcmp(scalar, vector, 2 * sizeof(*scalar)));
        }
    }
    puts("SVE activation tails, exponents and half ties PASS");
    return 1;
}
#endif

int main(void) {
    if (!encodings() || !activations()) return 1;
#ifdef __ARM_FEATURE_SVE
    if (!vector_activation_rounding()) return 1;
#endif
    const int shapes[][2] = {{8,64},{17,128},{64,5120},{9,65},{3,17},{1,1}};
    for (size_t i = 0; i < sizeof(shapes)/sizeof(shapes[0]); i++) {
        if (!matrix(2,shapes[i][0],shapes[i][1])) return 1;
        if (!(shapes[i][1]%64) && !matrix(1,shapes[i][0],shapes[i][1])) return 1;
    }
    puts("true FP6 / compact FP4 PASS");
    return 0;
}
