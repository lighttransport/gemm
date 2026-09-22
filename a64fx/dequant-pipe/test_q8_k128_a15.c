#include "q8_k128.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

enum { SHARED_TILES = 8, SHARED_EPOCHS = 64, SHARED_THREADS = 48 };
static float shared_inputs[SHARED_EPOCHS][SHARED_TILES * 128];
static q8_k128_a15_activation shared_ref[SHARED_EPOCHS][SHARED_TILES];
static q8_k128_a15_shared *shared_cache;
static int shared_errors;

static void *check_shared(void *arg)
{
    int tid = (int)(intptr_t)arg;
    for (int e = 0; e < SHARED_EPOCHS; e++) {
        const q8_k128_a15_activation *a = q8_k128_quantize_a15_shared(
            shared_cache, shared_inputs[e], SHARED_TILES, tid, SHARED_THREADS);
        /* Uneven consumers allow fast workers to start the next generation. */
        if (tid == e % SHARED_THREADS)
            for (int i = 0; i < 10000; i++) __asm__ __volatile__("yield");
        if (memcmp(a, shared_ref[e], sizeof(shared_ref[e])))
            __atomic_add_fetch(&shared_errors, 1, __ATOMIC_RELAXED);
    }
    return NULL;
}

static int test_shared(void)
{
    size_t bytes = q8_k128_a15_shared_bytes(SHARED_TILES);
    if (posix_memalign((void **)&shared_cache, 256, bytes)) return 1;
    memset(shared_cache, 0, bytes);
    for (int e = 0; e < SHARED_EPOCHS; e++) {
        for (int k = 0; k < SHARED_TILES * 128; k++)
            shared_inputs[e][k] = (float)((e * 17 + k * 13) % 997 - 498) * 0.0017f;
        for (int k = 0; k < SHARED_TILES; k++)
            q8_k128_quantize_a15(shared_inputs[e] + k * 128, &shared_ref[e][k]);
    }
    pthread_t threads[SHARED_THREADS];
    for (int t = 0; t < SHARED_THREADS; t++)
        if (pthread_create(&threads[t], NULL, check_shared, (void *)(intptr_t)t)) abort();
    for (int t = 0; t < SHARED_THREADS; t++) pthread_join(threads[t], NULL);
    free(shared_cache);
    if (shared_errors) fprintf(stderr, "shared A15 generations failed: %d\n", shared_errors);
    else puts("A15 shared PASS: 48 workers, 64 uneven generations, 8 tiles");
    return shared_errors != 0;
}

int main(void)
{
    uint8_t record[Q8_K128_RECORD_BYTES];
    int8_t hi[128], lo[128];
    float scales[4] = {0.001f, 0.002f, 0.0003f, 0.01f};
    float got[128], ref[128], x[128];
    uint32_t rng = 123;
    for (int zero = 0; zero < 2; zero++) {
        const float cases[] = {16256.0f, -16256.0f, 127.5f, -127.5f,
                               63.5f, -63.5f, 64.5f, -64.5f, 0.0f, -0.0f};
        for (int k = 0; k < 128; k++)
            x[k] = zero ? 0.0f : cases[(k % 32) % 10];
        q8_k128_a15_activation a;
        q8_k128_quantize_a15(x, &a);
        for (int k = 0; k < 128; k++) {
            int expected = (int)(x[k] + copysignf(0.5f, x[k]));
            if (a.hi[k] * 128 + a.lo[k] != expected ||
                a.scales[k / 32] != (zero ? 0.0f : 1.0f)) {
                fprintf(stderr, "A15 zero/tie/endpoint mismatch at %d\n", k);
                return 1;
            }
        }
    }
    for (int trial = 0; trial < 64; trial++) {
        for (int k = 0; k < 128; k++) {
            rng = rng * 1664525u + 1013904223u;
            int q = (int)(rng % 32513u) - 16256;
            int h = (q + 64) >> 7;
            hi[k] = h; lo[k] = q - h * 128;
            x[k] = (float)q * 0.00017f;
        }
        for (int r = 0; r < 128; r++) {
            got[r] = ref[r] = (float)(r - 63) * 0.07f;
            for (int b = 0; b < 4; b++) {
                ((uint16_t *)record)[b * 128 + r] = 0x3c00 - b * 0x400;
                int sum = 0;
                for (int j = 0; j < 32; j++) {
                    rng = rng * 1664525u + 1013904223u;
                    int8_t w = (int8_t)(rng >> 24);
                    record[1024 + ((b * 8 + j / 4) * 128 + r) * 4 + j % 4] = w;
                    sum += (int)w * (128 * hi[b * 32 + j] + lo[b * 32 + j]);
                }
                ref[r] = fmaf((float)sum, scales[b] * ldexpf(1.0f, -b), ref[r]);
            }
        }
        q8_k128_a15_dot(record, hi, lo, scales, got);
        if (memcmp(got, ref, sizeof(got))) {
            for (int r = 0; r < 128; r++) if (got[r] != ref[r])
                fprintf(stderr, "integer oracle mismatch trial=%d row=%d got=%a ref=%a\n", trial, r, got[r], ref[r]);
            return 1;
        }
        memset(got, 0, sizeof(got));
        q8_k128_apply(2, record, x, got);
        for (int r = 0; r < 128; r++) {
            double exact = 0, bound = 0;
            for (int b = 0; b < 4; b++) {
                float amax = 0;
                for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(x[b * 32 + j]));
                for (int j = 0; j < 32; j++) {
                    int w = (int8_t)record[1024 + ((b * 8 + j / 4) * 128 + r) * 4 + j % 4];
                    double scale = ldexp(1.0, -b);
                    exact += w * scale * x[b * 32 + j];
                    bound += abs(w) * scale * amax / 32512.0;
                }
            }
            if (fabs(got[r] - exact) > bound + fabs(exact) * 2e-6 + 1e-4) {
                fprintf(stderr, "activation quantization bound failed trial=%d row=%d\n", trial, r);
                return 1;
            }
        }
    }
    puts("A15 PASS: zero/ties/endpoints, 64 random integer-dot oracles and activation-error bounds");
    return test_shared();
}
