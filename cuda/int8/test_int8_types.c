/*
 * test_int8_types.c - Tests for INT8 type utilities
 *
 * Tests xoroshiro128+ PRNG, stochastic rounding, clamping,
 * scale computation, and random generators from int8_types.h.
 *
 * CPU-only, no GPU required.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "int8_types.h"

static int g_tests_run = 0;
static int g_tests_passed = 0;

#define TEST(name) do { \
    g_tests_run++; \
    printf("  %-50s ", name); \
} while(0)

#define PASS() do { g_tests_passed++; printf("PASS\n"); } while(0)
#define FAIL(msg, ...) do { printf("FAIL: " msg "\n", ##__VA_ARGS__); } while(0)

/* ---- Xoroshiro128+ PRNG Tests ---- */

static void test_xoro_deterministic(void) {
    TEST("xoro: same seed -> same sequence");
    xoroshiro128plus_t rng1, rng2;
    xoro_seed(&rng1, 12345);
    xoro_seed(&rng2, 12345);
    int ok = 1;
    for (int i = 0; i < 100; i++) {
        if (xoro_next(&rng1) != xoro_next(&rng2)) { ok = 0; break; }
    }
    if (ok) PASS();
    else FAIL("sequences diverged");
}

static void test_xoro_different_seeds(void) {
    TEST("xoro: different seeds -> different sequences");
    xoroshiro128plus_t rng1, rng2;
    xoro_seed(&rng1, 1);
    xoro_seed(&rng2, 2);
    /* First values should differ */
    uint64_t v1 = xoro_next(&rng1);
    uint64_t v2 = xoro_next(&rng2);
    if (v1 != v2) PASS();
    else FAIL("same output from different seeds");
}

static void test_xoro_nonzero_state(void) {
    TEST("xoro: seed 0 doesn't produce all-zero state");
    xoroshiro128plus_t rng;
    xoro_seed(&rng, 0);
    if (rng.s[0] != 0 || rng.s[1] != 0) PASS();
    else FAIL("all-zero state from seed 0");
}

static void test_xoro_uniform_range(void) {
    TEST("xoro: uniform() in [0, 1)");
    xoroshiro128plus_t rng;
    xoro_seed(&rng, 42);
    int ok = 1;
    for (int i = 0; i < 10000; i++) {
        float u = xoro_uniform(&rng);
        if (u < 0.0f || u >= 1.0f) { ok = 0; break; }
    }
    if (ok) PASS();
    else FAIL("value out of range");
}

static void test_xoro_uniform_double_range(void) {
    TEST("xoro: uniform_double() in [0, 1)");
    xoroshiro128plus_t rng;
    xoro_seed(&rng, 42);
    int ok = 1;
    for (int i = 0; i < 10000; i++) {
        double u = xoro_uniform_double(&rng);
        if (u < 0.0 || u >= 1.0) { ok = 0; break; }
    }
    if (ok) PASS();
    else FAIL("value out of range");
}

static void test_xoro_uniform_distribution(void) {
    TEST("xoro: uniform() roughly uniform (chi-squared)");
    xoroshiro128plus_t rng;
    xoro_seed(&rng, 42);
    int bins[10] = {0};
    int n = 100000;
    for (int i = 0; i < n; i++) {
        float u = xoro_uniform(&rng);
        int bin = (int)(u * 10);
        if (bin >= 10) bin = 9;
        bins[bin]++;
    }
    /* Each bin should have ~10000 values. Allow 20% deviation. */
    int ok = 1;
    int expected = n / 10;
    for (int i = 0; i < 10; i++) {
        if (abs(bins[i] - expected) > expected / 5) { ok = 0; break; }
    }
    if (ok) PASS();
    else FAIL("distribution not uniform");
}

static void test_xoro_jump(void) {
    TEST("xoro: jump() advances state");
    xoroshiro128plus_t rng1, rng2;
    xoro_seed(&rng1, 42);
    xoro_seed(&rng2, 42);
    xoro_jump(&rng1);
    /* After jump, sequences should differ */
    uint64_t v1 = xoro_next(&rng1);
    uint64_t v2 = xoro_next(&rng2);
    if (v1 != v2) PASS();
    else FAIL("jump didn't change sequence");
}

/* ---- Stochastic Rounding Tests ---- */

static void test_sr_unbiased(void) {
    TEST("stochastic_round: unbiased (E[SR(x)] ≈ x)");
    xoroshiro128plus_t rng;
    xoro_seed(&rng, 42);
    float x = 2.3f;
    double sum = 0.0;
    int n = 100000;
    for (int i = 0; i < n; i++) {
        sum += stochastic_round_f32(x, &rng);
    }
    double mean = sum / n;
    /* Should be close to 2.3 */
    if (fabs(mean - x) < 0.02) PASS();
    else FAIL("mean=%.4f expected=%.4f", mean, x);
}

static void test_sr_integer_passthrough(void) {
    TEST("stochastic_round: exact integers pass through");
    xoroshiro128plus_t rng;
    xoro_seed(&rng, 42);
    int ok = 1;
    for (int i = -5; i <= 5; i++) {
        for (int j = 0; j < 100; j++) {
            int32_t r = stochastic_round_f32((float)i, &rng);
            if (r != i) { ok = 0; break; }
        }
        if (!ok) break;
    }
    if (ok) PASS();
    else FAIL("integer value was rounded");
}

static void test_sr_negative(void) {
    TEST("stochastic_round: negative values unbiased");
    xoroshiro128plus_t rng;
    xoro_seed(&rng, 42);
    float x = -1.7f;
    double sum = 0.0;
    int n = 100000;
    for (int i = 0; i < n; i++) {
        sum += stochastic_round_f32(x, &rng);
    }
    double mean = sum / n;
    if (fabs(mean - x) < 0.02) PASS();
    else FAIL("mean=%.4f expected=%.4f", mean, x);
}

static void test_int32_to_s8_sr_clamp(void) {
    TEST("int32_to_s8_sr: clamps to [-128, 127]");
    xoroshiro128plus_t rng;
    xoro_seed(&rng, 42);
    int8_t lo = int32_to_s8_sr(1000, 1.0f, &rng);
    int8_t hi = int32_to_s8_sr(-1000, 1.0f, &rng);
    if (lo == S8_MAX && hi == S8_MIN) PASS();
    else FAIL("lo=%d hi=%d", lo, hi);
}

static void test_int32_to_u8_sr_clamp(void) {
    TEST("int32_to_u8_sr: clamps to [0, 255]");
    xoroshiro128plus_t rng;
    xoro_seed(&rng, 42);
    uint8_t hi = int32_to_u8_sr(1000, 1.0f, &rng);
    uint8_t lo = int32_to_u8_sr(-1000, 1.0f, &rng);
    if (hi == U8_MAX && lo == U8_MIN) PASS();
    else FAIL("hi=%u lo=%u", hi, lo);
}

static void test_batch_sr_s8(void) {
    TEST("batch_sr_s8: batch conversion");
    xoroshiro128plus_t rng;
    xoro_seed(&rng, 42);
    int32_t src[] = {100, 200, -100, 0, 50};
    int8_t dst[5];
    float scale = 127.0f / 200.0f;
    batch_sr_s8(src, dst, 5, scale, &rng);
    /* All values should be in valid range and non-trivial */
    int ok = 1;
    int has_nonzero = 0;
    for (int i = 0; i < 5; i++) {
        if (dst[i] != 0) has_nonzero = 1;
    }
    ok = has_nonzero;
    if (ok) PASS();
    else FAIL("out of range values");
}

/* ---- Clamp Tests ---- */

static void test_clamp_s8(void) {
    TEST("clamp_s8: boundary values");
    int ok = 1;
    ok &= (clamp_s8(0) == 0);
    ok &= (clamp_s8(127) == 127);
    ok &= (clamp_s8(-128) == -128);
    ok &= (clamp_s8(200) == 127);
    ok &= (clamp_s8(-200) == -128);
    ok &= (clamp_s8(1000000) == 127);
    ok &= (clamp_s8(-1000000) == -128);
    if (ok) PASS();
    else FAIL("clamp_s8 boundary error");
}

static void test_clamp_u8(void) {
    TEST("clamp_u8: boundary values");
    int ok = 1;
    ok &= (clamp_u8(0) == 0);
    ok &= (clamp_u8(255) == 255);
    ok &= (clamp_u8(-1) == 0);
    ok &= (clamp_u8(300) == 255);
    ok &= (clamp_u8(-100) == 0);
    ok &= (clamp_u8(1000000) == 255);
    if (ok) PASS();
    else FAIL("clamp_u8 boundary error");
}

/* ---- Scale Computation Tests ---- */

static void test_compute_scale_s8(void) {
    TEST("compute_scale_s8: known values");
    int32_t data[] = {0, 100, -200, 50, -50};
    float s = compute_scale_s8(data, 5);
    /* max_abs = 200, scale = 127/200 = 0.635 */
    if (fabsf(s - 0.635f) < 0.001f) PASS();
    else FAIL("scale=%f expected ~0.635", s);
}

static void test_compute_scale_s8_zero(void) {
    TEST("compute_scale_s8: all zeros -> 1.0");
    int32_t data[] = {0, 0, 0};
    float s = compute_scale_s8(data, 3);
    if (fabsf(s - 1.0f) < 1e-6f) PASS();
    else FAIL("scale=%f expected 1.0", s);
}

static void test_compute_scale_u8(void) {
    TEST("compute_scale_u8: known values");
    int32_t data[] = {0, 100, 200, 50, 150};
    float s = compute_scale_u8(data, 5);
    /* max = 200, scale = 255/200 = 1.275 */
    if (fabsf(s - 1.275f) < 0.001f) PASS();
    else FAIL("scale=%f expected ~1.275", s);
}

/* ---- Random Generator Range Tests ---- */

static void test_random_s8_range(void) {
    TEST("random_s8_range: values in [-range, range]");
    srand(42);
    int range = 10;
    int ok = 1;
    for (int i = 0; i < 1000; i++) {
        int8_t v = random_s8_range(range);
        if (v < -range || v > range) { ok = 0; break; }
    }
    if (ok) PASS();
    else FAIL("value out of range");
}

static void test_random_u8_range(void) {
    TEST("random_u8_range: values in [0, range]");
    srand(42);
    int range = 50;
    int ok = 1;
    for (int i = 0; i < 1000; i++) {
        uint8_t v = random_u8_range(range);
        if (v > range) { ok = 0; break; }
    }
    if (ok) PASS();
    else FAIL("value out of range");
}

static void test_xoro_random_s8_range(void) {
    TEST("xoro_random_s8_range: values in [-range, range]");
    xoroshiro128plus_t rng;
    xoro_seed(&rng, 42);
    int range = 10;
    int ok = 1;
    for (int i = 0; i < 10000; i++) {
        int8_t v = xoro_random_s8_range(&rng, range);
        if (v < -range || v > range) { ok = 0; break; }
    }
    if (ok) PASS();
    else FAIL("value out of range");
}

static void test_xoro_random_u8_range(void) {
    TEST("xoro_random_u8_range: values in [0, range]");
    xoroshiro128plus_t rng;
    xoro_seed(&rng, 42);
    int range = 50;
    int ok = 1;
    for (int i = 0; i < 10000; i++) {
        uint8_t v = xoro_random_u8_range(&rng, range);
        if (v > range) { ok = 0; break; }
    }
    if (ok) PASS();
    else FAIL("value out of range");
}

static void test_random_s8_range_zero(void) {
    TEST("random_s8_range: range=0 -> always 0");
    int ok = 1;
    for (int i = 0; i < 100; i++) {
        if (random_s8_range(0) != 0) { ok = 0; break; }
    }
    if (ok) PASS();
    else FAIL("nonzero for range=0");
}

static void test_random_u8_range_zero(void) {
    TEST("random_u8_range: range=0 -> always 0");
    int ok = 1;
    for (int i = 0; i < 100; i++) {
        if (random_u8_range(0) != 0) { ok = 0; break; }
    }
    if (ok) PASS();
    else FAIL("nonzero for range=0");
}

int main(void) {
    printf("=== INT8 Type Utility Tests ===\n\n");

    printf("Xoroshiro128+ PRNG:\n");
    test_xoro_deterministic();
    test_xoro_different_seeds();
    test_xoro_nonzero_state();
    test_xoro_uniform_range();
    test_xoro_uniform_double_range();
    test_xoro_uniform_distribution();
    test_xoro_jump();

    printf("\nStochastic Rounding:\n");
    test_sr_unbiased();
    test_sr_integer_passthrough();
    test_sr_negative();
    test_int32_to_s8_sr_clamp();
    test_int32_to_u8_sr_clamp();
    test_batch_sr_s8();

    printf("\nClamp Functions:\n");
    test_clamp_s8();
    test_clamp_u8();

    printf("\nScale Computation:\n");
    test_compute_scale_s8();
    test_compute_scale_s8_zero();
    test_compute_scale_u8();

    printf("\nRandom Generators:\n");
    test_random_s8_range();
    test_random_u8_range();
    test_xoro_random_s8_range();
    test_xoro_random_u8_range();
    test_random_s8_range_zero();
    test_random_u8_range_zero();

    printf("\n=== Results: %d/%d passed ===\n", g_tests_passed, g_tests_run);
    return (g_tests_passed == g_tests_run) ? 0 : 1;
}
