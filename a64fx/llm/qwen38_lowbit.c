#include "qwen38_lowbit.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(sizeof(q38_fp4_tile) == 288, "compact NVFP4 tile");
_Static_assert(sizeof(q38_fp6_tile) == 400, "true FP6 tile");
static const int8_t fp4[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};

size_t q38_lowbit_bytes(int format, int rows, int cols) {
    size_t size = format == Q38_LB_NVFP4 ? sizeof(q38_fp4_tile) :
                  format == Q38_LB_FP6_E2M3 ? sizeof(q38_fp6_tile) : 0;
    if (!size || rows <= 0 || cols <= 0) return 0;
    size_t nr = ((size_t)rows + 7) / 8, nk = ((size_t)cols + 63) / 64;
    if (nr > SIZE_MAX / nk || nr * nk > SIZE_MAX / size) return 0;
    return nr * nk * size;
}

float q38_fp6_decode(uint8_t code) {
    unsigned e = (code >> 3) & 3, m = code & 7;
    float value = e ? ldexpf((float)(8 + m), (int)e - 4) : (float)m * 0.125f;
    return code & 32 ? -value : value;
}

/* Piecewise uniform E2M3 intervals. Explicit ties-to-even is independent of
 * the process rounding mode and avoids 32 candidate evaluations per weight. */
int q38_fp6_encode(float value, uint8_t *code) {
    if (!code || !isfinite(value)) return 0;
    float a = fabsf(value);
    unsigned selected = 31;
    if (a < 7.5f) {
        float scaled = a < 2 ? a * 8 : a < 4 ? a * 4 : a * 2;
        unsigned q = (unsigned)scaled;
        float remainder = scaled - (float)q;
        q += remainder > .5f || (remainder == .5f && (q & 1));
        selected = q + (a < 2 ? 0 : a < 4 ? 8 : 16);
    }
    *code = (uint8_t)(selected | (signbit(value) ? 32 : 0));
    return 1;
}

float q38_fp4_scale(uint8_t code) {
    /* Match the source GGUF UE4M3 decoder, including its 0x7f sentinel. */
    if (!code || code == 127) return 0.0f;
    int e = (code >> 3) & 15, m = code & 7;
    return e ? ldexpf((float)(8 + m), e - 11) : ldexpf((float)m, -10);
}

static int fp6_exponent(float a) {
    if (!a) return 0;
    int e;
    (void)frexpf(a, &e);
    e -= 3;
    if ((double)a > ldexp(7.5, e)) e++;
    if (e < -127) e = -127;
    if (e > 127) e = 127;
    return e;
}

static void set_fp6(q38_fp6_tile *w, int r, int k, uint8_t code) {
    int s = k / 16, j = k % 8, lane = r * 8 + j;
    int half = (k % 16) / 8;
    w->low[s][lane] |= (uint8_t)((code & 15) << (half * 4));
    w->high[s][lane / 2] |= (uint8_t)((code >> 4) << ((lane & 1) * 4 + half * 2));
}

static uint8_t get_fp6(const q38_fp6_tile *w, int r, int k) {
    int s = k / 16, lane = r * 8 + k % 8, half = (k % 16) / 8;
    unsigned lo = (w->low[s][lane] >> (half * 4)) & 15;
    unsigned hi = (w->high[s][lane / 2] >> ((lane & 1) * 4 + half * 2)) & 3;
    return (uint8_t)(lo | hi << 4);
}

int q38_lowbit_pack_fp6(void *out, size_t bytes, const float *src,
                       size_t stride, int rows, int cols) {
    size_t need = q38_lowbit_bytes(Q38_LB_FP6_E2M3, rows, cols);
    if (!need || !out || !src || bytes < need || stride < (size_t)cols ||
        (size_t)rows > SIZE_MAX / stride / sizeof(float)) return 0;
    for (int r = 0; r < rows; r++)
        for (int k = 0; k < cols; k++)
            if (!isfinite(src[(size_t)r * stride + k])) return 0;
    memset(out, 0, need);
    size_t nb = ((size_t)cols + 63) / 64;
    q38_fp6_tile *tiles = out;
    for (size_t b = 0; b < need / sizeof(*tiles); b++)
        memset(tiles[b].scale, 127, sizeof(tiles[b].scale));
    for (int r = 0; r < rows; r++) {
        for (int k = 0; k < cols; k += 32) {
            float amax = 0;
            int count = cols - k < 32 ? cols - k : 32;
            for (int j = 0; j < count; j++)
                amax = fmaxf(amax, fabsf(src[(size_t)r * stride + k + j]));
            int e = fp6_exponent(amax);
            /* Scaling by a power of two is exact in double for every finite
             * FP32 input and this exponent range. Avoid one libm call per
             * weight when converting a multi-billion-weight BF16 model. */
            double inverse = ldexp(1.0, -e);
            q38_fp6_tile *w = &tiles[(size_t)(r / 8) * nb + k / 64];
            w->scale[(k % 64) / 32][r % 8] = (uint8_t)(e + 127);
            for (int j = 0; j < count; j++) {
                uint8_t code;
                float v = (float)((double)src[(size_t)r * stride + k + j] * inverse);
                if (!q38_fp6_encode(v, &code)) return 0;
                set_fp6(w, r % 8, k % 64 + j, code);
            }
        }
    }
    return 1;
}

int q38_lowbit_pack_nvfp4(void *out, size_t bytes, const void *src,
                         size_t row_bytes, int rows, int cols) {
    size_t need = q38_lowbit_bytes(Q38_LB_NVFP4, rows, cols);
    if (!need || !out || !src || bytes < need || cols % 64 ||
        row_bytes < (size_t)(cols / 64) * 36 ||
        (size_t)rows > SIZE_MAX / row_bytes) return 0;
    memset(out, 0, need);
    q38_fp4_tile *tiles = out;
    const uint8_t *input = src;
    int nb = cols / 64;
    for (int r = 0; r < rows; r++)
        for (int b = 0; b < nb; b++) {
            const uint8_t *q = input + (size_t)r * row_bytes + b * 36;
            q38_fp4_tile *w = &tiles[(size_t)(r / 8) * nb + b];
            for (int s = 0; s < 4; s++) {
                w->scale[s][r % 8] = q[s];
                memcpy(w->codes[s] + (r % 8) * 8, q + 4 + s * 8, 8);
            }
        }
    return 1;
}

int q38_lowbit_validate(const void *weights, size_t bytes,
                       int format, int rows, int cols) {
    size_t need = q38_lowbit_bytes(format, rows, cols);
    if (!need || !weights || bytes != need) return 0;
    if (format == Q38_LB_FP6_E2M3) {
        const q38_fp6_tile *w = weights;
        for (size_t b = 0; b < need / sizeof(*w); b++)
            for (int s = 0; s < 2; s++)
                for (int r = 0; r < 8; r++)
                    if (w[b].scale[s][r] == 255) return 0;
    }
    return 1;
}

static float weight_at(const void *weights, int format, size_t nb, int r, int k) {
    size_t b = (size_t)(r / 8) * nb + k / 64;
    int lane = (r % 8) * 8 + k % 8, s = (k % 64) / 16;
    if (format == Q38_LB_NVFP4) {
        const q38_fp4_tile *w = (const q38_fp4_tile *)weights + b;
        unsigned c = (w->codes[s][lane] >> ((k % 16) / 8 * 4)) & 15;
        return fp4[c] * q38_fp4_scale(w->scale[s][r % 8]);
    }
    const q38_fp6_tile *w = (const q38_fp6_tile *)weights + b;
    return ldexpf(q38_fp6_decode(get_fp6(w, r % 8, k % 64)),
                  (int)w->scale[(k % 64) / 32][r % 8] - 127);
}

int q38_lowbit_dequant_row(float *out, const void *weights,
                          int format, int rows, int cols, int row) {
    if (!out || !weights || !q38_lowbit_bytes(format, rows, cols) ||
        row < 0 || row >= rows) return 0;
    size_t nb = ((size_t)cols + 63) / 64;
    for (int k = 0; k < cols; k++) out[k] = weight_at(weights, format, nb, row, k);
    return 1;
}

int q38_lowbit_prepare(q38_lowbit_act *out, size_t count,
                      const float *x, int cols, int arithmetic) {
    if (!out || !x || cols <= 0 || count < ((size_t)cols + 15) / 16 ||
        (arithmetic != Q38_LB_A8 && arithmetic != Q38_LB_A16)) return 0;
    for (int k = 0; k < cols; k++) if (!isfinite(x[k])) return 0;
    /* Centered radix-256 digits avoid a third weight-sum SDOT. 32639 is the
     * largest symmetric bound whose signed high and low bytes both fit. */
    const int limit = arithmetic == Q38_LB_A8 ? 127 : 32639;
    for (int k = 0; k < cols; k += 32) {
        int n = cols - k < 32 ? cols - k : 32;
        float amax = 0;
        for (int j = 0; j < n; j++) amax = fmaxf(amax, fabsf(x[k + j]));
        double scale = amax ? (double)amax / limit : 1.0;
        float stored_scale = (float)scale;
        if (stored_scale == 0) stored_scale = FLT_TRUE_MIN;
        scale = stored_scale;
        for (int s = 0; s < (n + 15) / 16; s++) {
            q38_lowbit_act *a = out + k / 16 + s;
            memset(a, 0, sizeof(*a));
            a->scale = stored_scale;
            for (int j = 0; j < 16; j++) {
                double f = s * 16 + j < n ? x[k + s * 16 + j] / scale : 0;
                /* round-to-nearest-even independent of the process FP mode */
                double floor_q = floor(f), frac = f - floor_q;
                int q = (int)(floor_q + (frac > .5 || (frac == .5 && fmod(floor_q, 2) != 0)));
                if (q > limit) q = limit;
                if (q < -limit) q = -limit;
                int lo = arithmetic == Q38_LB_A8 ? q : (q + 128) % 256;
                if (arithmetic == Q38_LB_A16) {
                    if (lo < 0) lo += 256;
                    lo -= 128;
                }
                int hi = (q - lo) / 256;
                for (int r = 0; r < 8; r++) {
                    a->lo[j / 8][r * 8 + j % 8] = (int8_t)lo;
                    a->hi[j / 8][r * 8 + j % 8] = (int8_t)hi;
                }
            }
        }
    }
    return 1;
}

int q38_lowbit_reference(float *out, const void *weights, int format,
                        const float *x, int rows, int cols) {
    if (!out || !weights || !x || !q38_lowbit_bytes(format, rows, cols)) return 0;
    size_t nb = ((size_t)cols + 63) / 64;
    for (int r = 0; r < rows; r++) {
        double sum = 0;
        for (int k = 0; k < cols; k++) sum += (double)weight_at(weights, format, nb, r, k) * x[k];
        out[r] = (float)sum;
    }
    return 1;
}

int q38_lowbit_dot(float *out, const void *weights, int format,
                   const q38_lowbit_act *act, int arithmetic, int rows, int cols) {
    if (!out || !weights || !act || !q38_lowbit_bytes(format, rows, cols) ||
        (arithmetic != Q38_LB_A8 && arithmetic != Q38_LB_A16)) return 0;
    size_t nb = ((size_t)cols + 63) / 64;
    for (int r = 0; r < rows; r++) {
        double sum = 0;
        for (int k = 0; k < cols; k++) {
            const q38_lowbit_act *a = act + k / 16;
            int half = (k % 16) / 8, j = k % 8;
            int q = a->lo[half][j];
            if (arithmetic == Q38_LB_A16) q += 256 * a->hi[half][j];
            sum += (double)weight_at(weights, format, nb, r, k) * q * a->scale;
        }
        out[r] = (float)sum;
    }
    return 1;
}
