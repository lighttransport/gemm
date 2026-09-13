#define _POSIX_C_SOURCE 200809L
#include "glm53f_int8.h"
#include "glm53f_cache_bf16.h"
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static double seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static int check_matrix(uint8_t *source, const float *scales, int rows, int cols) {
    size_t bytes = (size_t)rows * cols;
    uint8_t *packed = malloc(bytes);
    float *rs = malloc((size_t)rows * sizeof(float));
    float *x = malloc((size_t)cols * sizeof(float)), xs;
    float *y = malloc((size_t)rows * sizeof(float));
    int8_t *qx = malloc((size_t)cols), *work = malloc((size_t)64 * cols);
    if (!packed || !rs || !x || !y || !qx || !work) return 1;
    memcpy(packed, source, bytes);
    for (int r = 0; r < rows; r += 64)
        if (glm53f_i8_pack_fp8_tile(packed + (size_t)r * cols, rs + r,
                                   scales, r, cols, work)) return 1;
    int fail = 0;
    double err = 0, ref2 = 0, max_arithmetic = 0, max_quant = 0;
    for (int trial = 0; trial < 5; ++trial) {
        for (int c = 0; c < cols; ++c)
            x[c] = trial == 0 ? 0.0f : (float)(((c * 97 + trial * 53) % 251) - 125) / 125;
        if (glm53f_i8_quantize_x(qx, &xs, x, cols)) return 1;
        for (int r = 0; r < rows; r += 64)
            glm53f_i8_dot64(y + r, (int8_t *)packed + (size_t)r * cols,
                           rs + r, qx, xs, cols);
        for (int r = 0; r < rows; r += 16) {
            float fine[16];
            glm53f_i8_dot16(fine, (int8_t *)packed + (size_t)(r / 64) * 64 * cols + (r % 64) * 4,
                rs + r, qx, xs, cols);
            if (memcmp(fine, y + r, sizeof(fine))) fail = 1;
        }
        for (int r = 0; r < rows; ++r) {
            double original = 0, dequantized = 0, magnitude = 0;
            const int8_t *p = (int8_t *)packed + (size_t)(r / 64) * 64 * cols;
            for (int c = 0; c < cols; ++c) {
                int q = p[(size_t)(c / 4) * 256 + (r % 64) * 4 + c % 4];
                double w = glm53f_i8_fp8(source[(size_t)r * cols + c]) *
                           scales[(size_t)(r / 128) * (cols / 128) + c / 128];
                double quant_error = fabs(w - q * (double)rs[r]);
                if (quant_error > max_quant) max_quant = quant_error;
                if (quant_error > rs[r] * 0.501 + 1e-6 * fabs(w)) fail = 1;
                original += w * x[c];
                double product = q * (double)qx[c] * rs[r] * xs;
                dequantized += product;
                magnitude += fabs(product);
            }
            double relative = fabs(y[r] - dequantized) / (magnitude + 1e-30);
            if (relative > max_arithmetic) max_arithmetic = relative;
            if (!isfinite(y[r]) || relative > 2e-6) fail = 1;
            err += (y[r] - original) * (y[r] - original);
            ref2 += original * original;
        }
    }
    printf("INT8_CHECK rows=%d cols=%d arithmetic_error=%.9g quant_max=%.9g output_rel_l2=%.9g %s\n",
           rows, cols, max_arithmetic, max_quant, sqrt(err / (ref2 + 1e-30)), fail ? "FAIL" : "PASS");
    free(work); free(qx); free(y); free(x); free(rs); free(packed);
    return fail;
}

static int real_matrix(const char *blob, const char *manifest, int layer, int expert) {
    char line[2048], want[512], want_scale[544], name[512], dt[24];
    snprintf(want, sizeof(want), "model.language_model.layers.%d.mlp.experts.%d.gate_up_fused.weight", layer, expert);
    snprintf(want_scale, sizeof(want_scale), "%s_scale_inv", want);
    unsigned long long off = 0, so = 0, parsed;
    int rows = 0, cols = 0, sr = 0, sc = 0, nd, r, c, found = 0;
    FILE *f = fopen(manifest, "r");
    if (!f) return 1;
    while (fgets(line, sizeof(line), f)) {
        char *last = strrchr(line, ' ');
        if (line[0] == '#' || !last || sscanf(line, "%llu %23s %d %d %d", &parsed, dt, &nd, &r, &c) != 5) continue;
        snprintf(name, sizeof(name), "%s", last + 1);
        name[strcspn(name, "\r\n")] = 0;
        if (!strcmp(name, want)) { off = parsed; rows = r; cols = c; found |= 1; }
        if (!strcmp(name, want_scale)) { so = parsed; sr = r; sc = c; found |= 2; }
    }
    fclose(f);
    if (found != 3 || rows <= 0 || rows % 64 || cols <= 0 || cols % 128 || sr != (rows + 127) / 128 || sc != cols / 128) return 1;
    size_t bytes = (size_t)rows * cols, scale_bytes = (size_t)sr * sc * 4;
    uint8_t *w = malloc(bytes); float *s = malloc(scale_bytes);
    int fd = open(blob, O_RDONLY);
    if (!w || !s || fd < 0) return 1;
    if (pread(fd, w, bytes, (off_t)off) != (ssize_t)bytes ||
        pread(fd, s, scale_bytes, (off_t)so) != (ssize_t)scale_bytes) return 1;
    posix_fadvise(fd, (off_t)off, (off_t)bytes, POSIX_FADV_DONTNEED);
    close(fd);
    int rc = check_matrix(w, s, rows, cols);
    free(s); free(w);
    return rc;
}

static int check_bf16(int cols) {
    uint16_t *source = malloc((size_t)64 * cols * 2);
    int8_t *packed = malloc((size_t)64 * cols), *qx = malloc((size_t)cols);
    float scale[64], y[64], *x = malloc((size_t)cols * 4), xs;
    if (!source || !packed || !qx || !x) return 1;
    for (int r = 0; r < 64; ++r)
        for (int c = 0; c < cols; ++c) {
            float v = r == 0 ? 0 : (float)((r * 79 + c * 17) % 1009 - 504) * 0.001f;
            uint32_t bits; memcpy(&bits, &v, sizeof(bits));
            source[(size_t)r * cols + c] = (uint16_t)(bits >> 16);
        }
    for (int c = 0; c < cols; ++c) x[c] = (c % 31 - 15) * 0.01f;
    if (glm53f_i8_pack_bf16_tile(packed, scale, source, cols) ||
        glm53f_i8_quantize_x(qx, &xs, x, cols)) return 1;
    glm53f_i8_dot64(y, packed, scale, qx, xs, cols);
    int failed = 0; double err = 0, ref2 = 0, max_arithmetic = 0;
    for (int r = 0; r < 64; ++r) {
        double ref = 0, dequantized = 0, magnitude = 0;
        for (int c = 0; c < cols; ++c) {
            uint32_t bits = (uint32_t)source[(size_t)r * cols + c] << 16;
            float v; memcpy(&v, &bits, sizeof(v));
            int q = packed[(size_t)(c / 4) * 256 + r * 4 + c % 4];
            if (fabs(v - q * (double)scale[r]) > scale[r] * 0.501 + 1e-6 * fabs(v)) failed = 1;
            ref += (double)v * x[c];
            double term = q * (double)qx[c] * scale[r] * xs;
            dequantized += term; magnitude += fabs(term);
        }
        double arithmetic = fabs(y[r] - dequantized) / (magnitude + 1e-30);
        if (arithmetic > max_arithmetic) max_arithmetic = arithmetic;
        if (arithmetic > 2e-6 || !isfinite(y[r])) failed = 1;
        err += (y[r] - ref) * (y[r] - ref); ref2 += ref * ref;
    }
    printf("INT8_BF16_CHECK cols=%d arithmetic_error=%.9g output_rel_l2=%.9g %s\n",
        cols, max_arithmetic, sqrt(err / (ref2 + 1e-30)), failed ? "FAIL" : "PASS");
    free(x); free(qx); free(packed); free(source);
    return failed;
}

static int benchmark(int reps, int tile) {
    enum { ROWS = 16384, COLS = 4096 };
    int8_t *w = malloc((size_t)ROWS * COLS), *qx = malloc(COLS);
    float *sc = malloc(ROWS * 4), *y = malloc(ROWS * 4), x[COLS], xs;
    if (!w || !qx || !sc || !y) return 1;
#pragma omp parallel for schedule(static)
    for (int r = 0; r < ROWS; ++r) {
        sc[r] = 0.001f;
        for (int c = 0; c < COLS; ++c) w[(size_t)r * COLS + c] = (int8_t)((r * 17 + c * 3) % 255 - 127);
    }
    for (int c = 0; c < COLS; ++c) x[c] = (c % 31 - 15) * 0.01f;
    glm53f_i8_quantize_x(qx, &xs, x, COLS);
    double best = 1e30, total = 0, checksum = 0;
    for (int rep = -2; rep < reps; ++rep) {
        double t = seconds();
#pragma omp parallel for schedule(static)
        for (int r = 0; r < ROWS; r += tile) {
            if (tile == 8) glm53f_i8_dot8_rows(y + r, w + (size_t)r * COLS, sc + r, qx, xs, COLS);
            else if (tile == 16) glm53f_i8_dot16(y + r, w + (size_t)(r / 64) * 64 * COLS + (r % 64) * 4, sc + r, qx, xs, COLS);
            else glm53f_i8_dot64(y + r, w + (size_t)r * COLS, sc + r, qx, xs, COLS);
        }
        t = seconds() - t;
        if (rep >= 0) { total += t; if (t < best) best = t; }
        /* Consume every output after every iteration. */
        for (int r = 0; r < ROWS; ++r) checksum += y[r];
    }
    printf("INT8_BENCH tile=%d bytes=%zu reps=%d best_ms=%.6f mean_ms=%.6f best_GB_s=%.3f checksum=%.9g\n",
           tile, (size_t)ROWS * COLS, reps, best * 1e3, total * 1e3 / reps,
           (double)ROWS * COLS / best / 1e9, checksum);
    free(y); free(sc); free(qx); free(w);
    return !isfinite(checksum);
}

int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "--bench")) {
        int reps = argc > 2 ? atoi(argv[2]) : 20, tile = argc > 3 ? atoi(argv[3]) : 64;
        if (reps < 1 || (tile != 8 && tile != 16 && tile != 64)) return 2;
        return benchmark(reps, tile);
    }
    if (argc == 5) return real_matrix(argv[1], argv[2], atoi(argv[3]), atoi(argv[4]));
    int failed = 0;
    const int columns[] = {128, 256, 512, 4096};
    for (unsigned k = 0; k < sizeof(columns) / sizeof(columns[0]); ++k) {
        int cols = columns[k], rows = 256;
        uint8_t *w = malloc((size_t)rows * cols);
        float *s = malloc((size_t)(rows / 128) * (cols / 128) * 4);
        if (!w || !s) return 1;
        for (int i = 0; i < rows * cols; ++i) w[i] = (uint8_t)((i * 71 + i / cols * 19) % 127) | ((i % 3) ? 128 : 0);
        for (int i = 0; i < (rows / 128) * (cols / 128); ++i) s[i] = 0.0001f * (1 + i % 7);
        failed |= check_matrix(w, s, rows, cols);
        free(s); free(w);
    }
    float bad[4] = {0, NAN, 1, 2}, scale;
    int8_t q[4];
    failed |= glm53f_i8_quantize_x(q, &scale, bad, 4) == 0;
    float tiny[4] = {0, 0x1p-140f, -0x1p-140f, 0x1p-141f};
    failed |= glm53f_i8_quantize_x(q, &scale, tiny, 4) != 0;
    failed |= q[0] != 0 || q[1] != 127 || q[2] != -127 || q[3] != 64;
    failed |= check_bf16(640);
    failed |= check_bf16(768);
    failed |= check_bf16(4096);
    int cache_failed = 0;
    cache_failed |= glm53f_cache_bf16_round(1.00390625f) != 0x3f80;
    cache_failed |= glm53f_cache_bf16_round(1.01171875f) != 0x3f82;
    cache_failed |= glm53f_cache_bf16_round(-0.0f) != 0x8000;
    cache_failed |= !isnan(glm53f_cache_bf16_expand(glm53f_cache_bf16_round(NAN)));
    cache_failed |= !isinf(glm53f_cache_bf16_expand(glm53f_cache_bf16_round(INFINITY)));
    for (int i = -100000; i < 100000; ++i) {
        float x = i * 0.0001234567f;
        float y = glm53f_cache_bf16_expand(glm53f_cache_bf16_round(x));
        if (fabsf(x - y) > fabsf(x) / 256.0f + 1e-30f) cache_failed = 1;
    }
    printf("BF16_CACHE_CODEC %s\n", cache_failed ? "FAIL" : "PASS");
    failed |= cache_failed;
    return failed;
}
