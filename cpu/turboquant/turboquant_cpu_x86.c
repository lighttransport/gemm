#include "turboquant_cpu_internal.h"

#include <stdint.h>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>

int tq3_x86_has_sse2(void) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("sse2") != 0;
#else
    return 1;
#endif
}

int tq3_x86_has_avx2(void) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") != 0;
#else
    return 0;
#endif
}

float tq3_dot_block_sse2(const tq3_block *blk, const float *x,
                         uint64_t seed, uint64_t block_index) {
    uint8_t idx[TQ3_BLOCK_SIZE];
    float xrot[TQ3_BLOCK_SIZE];
    float scale = tq3_f16_to_f32(blk->scale_f16);
    tq3_unpack_indices(idx, blk);
    tq3_forward_rotate(xrot, x, seed, block_index);

    __m128 acc = _mm_setzero_ps();
    __m128 sc = _mm_set1_ps(scale);
    for (int i = 0; i < TQ3_BLOCK_SIZE; i += 4) {
        float cbuf[4] = {
            tq3_centroids[idx[i + 0]], tq3_centroids[idx[i + 1]],
            tq3_centroids[idx[i + 2]], tq3_centroids[idx[i + 3]],
        };
        __m128 c = _mm_loadu_ps(cbuf);
        __m128 xv = _mm_loadu_ps(xrot + i);
        acc = _mm_add_ps(acc, _mm_mul_ps(_mm_mul_ps(c, sc), xv));
    }
    float tmp[4];
    _mm_storeu_ps(tmp, acc);
    return tmp[0] + tmp[1] + tmp[2] + tmp[3];
}

__attribute__((target("avx2,fma")))
float tq3_dot_block_avx2(const tq3_block *blk, const float *x,
                         uint64_t seed, uint64_t block_index) {
    uint8_t idx[TQ3_BLOCK_SIZE];
    float xrot[TQ3_BLOCK_SIZE];
    float scale = tq3_f16_to_f32(blk->scale_f16);
    tq3_unpack_indices(idx, blk);
    tq3_forward_rotate(xrot, x, seed, block_index);

    __m256 acc = _mm256_setzero_ps();
    __m256 sc = _mm256_set1_ps(scale);
    for (int i = 0; i < TQ3_BLOCK_SIZE; i += 8) {
        float cbuf[8] = {
            tq3_centroids[idx[i + 0]], tq3_centroids[idx[i + 1]],
            tq3_centroids[idx[i + 2]], tq3_centroids[idx[i + 3]],
            tq3_centroids[idx[i + 4]], tq3_centroids[idx[i + 5]],
            tq3_centroids[idx[i + 6]], tq3_centroids[idx[i + 7]],
        };
        __m256 c = _mm256_loadu_ps(cbuf);
        __m256 xv = _mm256_loadu_ps(xrot + i);
        acc = _mm256_fmadd_ps(_mm256_mul_ps(c, sc), xv, acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

#else

int tq3_x86_has_sse2(void) { return 0; }
int tq3_x86_has_avx2(void) { return 0; }
float tq3_dot_block_sse2(const tq3_block *blk, const float *x,
                         uint64_t seed, uint64_t block_index) {
    return tq3_dot_block_scalar(blk, x, seed, block_index);
}
float tq3_dot_block_avx2(const tq3_block *blk, const float *x,
                         uint64_t seed, uint64_t block_index) {
    return tq3_dot_block_scalar(blk, x, seed, block_index);
}

#endif
