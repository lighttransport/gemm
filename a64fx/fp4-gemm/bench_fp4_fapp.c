#define _POSIX_C_SOURCE 200112L
#include "fp4_gemm.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef USE_FAPP
#include <fj_tool/fapp.h>
#endif

static void *aligned_alloc_256(size_t bytes)
{
    void *p = NULL;
    return posix_memalign(&p, 256, (bytes + 255) & ~(size_t)255) ? NULL : p;
}

static unsigned rng_state = 7;

static unsigned next_random(void)
{
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}

int main(int argc, char **argv)
{
    int n = argc > 1 ? atoi(argv[1]) : 32768;
    int k = argc > 2 ? atoi(argv[2]) : 4096;
    int promotion_k = argc > 3 ? atoi(argv[3]) : 256;
    int iterations = argc > 4 ? atoi(argv[4]) : 200;
    int threads = argc > 5 ? atoi(argv[5]) : 12;
    fp4_matrix w;
    _Float16 *a;
    float *c;

    if (n <= 0 || k <= 0 || iterations <= 0 || threads <= 0 || n % 256 ||
        k % 32 || promotion_k <= 0 || promotion_k % 32 || promotion_k > k)
        return 2;
    if (fp4_matrix_alloc(&w, FP4_MX, n, k))
        return 1;
    for (size_t i = 0; i < w.code_bytes; ++i)
        w.codes[i] = (uint8_t)next_random();
    for (size_t i = 0; i < w.scale_bytes; ++i)
        w.scales[i] = 124;
    if (fp4_matrix_prepare_n32(&w))
        return 1;

    a = aligned_alloc_256((size_t)k * sizeof(*a));
    c = aligned_alloc_256((size_t)n * sizeof(*c));
    if (!a || !c)
        return 1;
    for (int i = 0; i < k; ++i)
        a[i] = (_Float16)((int)(next_random() & 255) - 128) / 512;

    if (fp4_gemm_f16_n32_omp(c, a, &w, 1, promotion_k, threads))
        return 1;
#ifdef USE_FAPP
    fapp_start("fp4_m1_t8", 1, 0);
#endif
    for (int i = 0; i < iterations; ++i)
        if (fp4_gemm_f16_n32_omp(c, a, &w, 1, promotion_k, threads))
            return 1;
#ifdef USE_FAPP
    fapp_stop("fp4_m1_t8", 1, 0);
#endif

    printf("N=%d K=%d promotion_k=%d iterations=%d threads=%d checksum=%.9g\n",
           n, k, promotion_k, iterations, threads, c[n / 3]);
    free(a);
    free(c);
    fp4_matrix_free(&w);
    return 0;
}
