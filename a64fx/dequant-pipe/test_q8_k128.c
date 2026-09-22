#define _GNU_SOURCE
#include "q8_k128.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint16_t f32_to_f16(float f)
{
    _Float16 h = (_Float16)f;
    uint16_t u;
    memcpy(&u, &h, sizeof(u));
    return u;
}

static float f16_to_f32(uint16_t u)
{
    _Float16 h;
    memcpy(&h, &u, sizeof(h));
    return (float)h;
}

static int test_cmg_partition(void)
{
    static const int rows[] = {5120, 6144, 10240, 17408, 248320};
    for (size_t s = 0; s < sizeof(rows) / sizeof(rows[0]); s++) {
        if (rows[s] % Q8_K128_N) return 1;
        int groups = rows[s] / Q8_K128_N;
        int previous = 0;
        for (int tid = 0; tid < 48; tid++) {
            int cmg = tid / 12;
            int g0 = groups * tid / 48;
            int g1 = groups * (tid + 1) / 48;
            int cmg_g0 = groups * cmg / 4;
            int cmg_g1 = groups * (cmg + 1) / 4;
            int expected_cpu = 12 + 12 * cmg + tid % 12;
            int expected_node = 4 + cmg;
            if (g0 != previous || g0 < cmg_g0 || g1 > cmg_g1 ||
                expected_cpu != 12 + tid || expected_node != 4 + tid / 12) {
                fprintf(stderr, "cmg partition mismatch rows=%d tid=%d "
                        "groups=[%d,%d) cmg%d=[%d,%d) cpu=%d node=%d\n",
                        rows[s], tid, g0, g1, cmg, cmg_g0, cmg_g1,
                        expected_cpu, expected_node);
                return 1;
            }
            previous = g1;
        }
        if (previous != groups) return 1;
    }
    return 0;
}

int main(void)
{
    if (test_cmg_partition()) return 1;
    uint8_t *record = NULL;
    float x[Q8_K128_K], got[Q8_K128_N], ref[Q8_K128_N];
    uint32_t rng = 0x38a64f27u;
    if (posix_memalign((void **)&record, 256, Q8_K128_RECORD_BYTES)) return 1;
    uint16_t *scales = (uint16_t *)record;
    int8_t *q = (int8_t *)(record + Q8_K128_SCALE_BYTES);
    for (int b = 0; b < 4; b++) for (int n = 0; n < Q8_K128_N; n++) {
        float s = (float)(1 + ((b * Q8_K128_N + n) % 31)) / 256.0f;
        scales[b * Q8_K128_N + n] = f32_to_f16(s);
    }
    for (int k = 0; k < 128; k++) {
        rng = rng * 1664525u + 1013904223u;
        x[k] = (float)((int)(rng >> 24) - 128) / 97.0f;
        for (int n = 0; n < Q8_K128_N; n++) {
            rng = rng * 1664525u + 1013904223u;
            q[k * Q8_K128_N + n] = (int8_t)(rng >> 24);
        }
    }
    memset(got, 0, sizeof(got));
    memset(ref, 0, sizeof(ref));
    q8_k128_f32(record, x, got);
    for (int b = 0; b < 4; b++) for (int n = 0; n < Q8_K128_N; n++) {
        float block = 0.0f;
        for (int j = 0; j < 32; j++) {
            int k = b * 32 + j;
            block = fmaf((float)q[k * Q8_K128_N + n], x[k], block);
        }
        ref[n] = fmaf(block, f16_to_f32(scales[b * Q8_K128_N + n]), ref[n]);
    }
    for (int n = 0; n < Q8_K128_N; n++) {
        if (memcmp(got + n, ref + n, sizeof(float))) {
            fprintf(stderr, "q8_k128 mismatch n=%d got=%a ref=%a\n", n, got[n], ref[n]);
            free(record);
            return 1;
        }
    }
    /* Exercise the production row-major Q8_0 -> tiled K-major pack mapping
     * across multiple K records. */
    struct __attribute__((packed)) block_q8 { uint16_t d; int8_t qs[32]; };
    enum { K2 = 256, NB2 = K2 / 32, TILES2 = K2 / Q8_K128_K };
    struct block_q8 *rows = calloc((size_t)Q8_K128_N * NB2, sizeof(*rows));
    uint8_t *packed = calloc(TILES2, Q8_K128_RECORD_BYTES);
    float x2[K2], got2[Q8_K128_N] = {0}, ref2[Q8_K128_N] = {0};
    if (!rows || !packed) return 1;
    for (int k = 0; k < K2; k++) x2[k] = (float)(k % 17 - 8) / 19.0f;
    for (int r = 0; r < Q8_K128_N; r++) for (int b = 0; b < NB2; b++) {
        struct block_q8 *src = rows + (size_t)r * NB2 + b;
        src->d = f32_to_f16((float)(1 + ((r + b) % 23)) / 128.0f);
        for (int j = 0; j < 32; j++) src->qs[j] = (int8_t)((r * 7 + b * 11 + j * 3) - 127);
    }
    for (int kt = 0; kt < TILES2; kt++) {
        uint8_t *rec = packed + (size_t)kt * Q8_K128_RECORD_BYTES;
        uint16_t *ds = (uint16_t *)rec;
        int8_t *qq = (int8_t *)(rec + Q8_K128_SCALE_BYTES);
        for (int b = 0; b < 4; b++) for (int r = 0; r < Q8_K128_N; r++) {
            struct block_q8 *src = rows + (size_t)r * NB2 + kt * 4 + b;
            ds[b * Q8_K128_N + r] = src->d;
            for (int j = 0; j < 32; j++)
                qq[(b * 32 + j) * Q8_K128_N + r] = src->qs[j];
        }
        q8_k128_f32(rec, x2 + kt * Q8_K128_K, got2);
    }
    for (int kt = 0; kt < TILES2; kt++) for (int b = 0; b < 4; b++)
        for (int r = 0; r < Q8_K128_N; r++) {
            struct block_q8 *src = rows + (size_t)r * NB2 + kt * 4 + b;
            float block = 0;
            for (int j = 0; j < 32; j++)
                block = fmaf((float)src->qs[j], x2[kt * 128 + b * 32 + j], block);
            ref2[r] = fmaf(block, f16_to_f32(src->d), ref2[r]);
        }
    for (int r = 0; r < Q8_K128_N; r++) if (memcmp(got2 + r, ref2 + r, sizeof(float))) {
        fprintf(stderr, "q8_k128 packed mismatch r=%d got=%a ref=%a\n", r, got2[r], ref2[r]);
        return 1;
    }
    free(rows); free(packed);
    puts("q8_k128 correctness: PASS (64x128, FP16 scales, block-reassociated FP32 FMA; "
         "48-worker CMG partition PASS)");
    free(record);
    return 0;
}
