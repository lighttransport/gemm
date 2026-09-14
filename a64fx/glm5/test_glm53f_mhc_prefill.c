#define _POSIX_C_SOURCE 200809L
#include "glm53f_mhc_sve.h"
#include <stdio.h>

int main(void) {
    const size_t weights = (size_t)GLM53F_MHC_MIX * GLM53F_MHC_FLAT;
    uint16_t *w = malloc(weights * sizeof(*w));
    uint16_t norm[GLM53F_MHC_WIDTH];
    float base[GLM53F_MHC_MIX], scale[3] = {0.25f, 0.5f, 0.125f};
    const int sizes[] = {1, 3, 4, 5, 31, 32, 64, 128, 256};
    int failed = 0;
    if (!w) return 2;
    for (size_t i = 0; i < weights; ++i) {
        float v = ((int)(i * 17 % 1009) - 504) * 0.000013f;
        uint32_t bits; memcpy(&bits, &v, 4); w[i] = (uint16_t)(bits >> 16);
    }
    for (int i = 0; i < GLM53F_MHC_WIDTH; ++i) norm[i] = 0x3f80;
    for (int i = 0; i < GLM53F_MHC_MIX; ++i) base[i] = (i % 7 - 3) * 0.01f;
    glm53f_mhc_site site = {w, base, scale};
    unsetenv("GLM53F_MHC_PREFILL");
    for (unsigned k = 0; k < sizeof(sizes)/sizeof(sizes[0]); ++k) {
        int n = sizes[k];
        float *x = malloc((size_t)n * GLM53F_MHC_FLAT * sizeof(float));
        float *a = malloc((size_t)n * GLM53F_MHC_WIDTH * sizeof(float));
        float *b = malloc((size_t)n * GLM53F_MHC_WIDTH * sizeof(float));
        glm53f_mhc_scratch *sa = calloc(n, sizeof(*sa)), *sb = calloc(n, sizeof(*sb));
        if (!x || !a || !b || !sa || !sb) return 2;
        for (int t = 0; t < n; ++t)
            for (int i = 0; i < GLM53F_MHC_FLAT; ++i)
                x[(size_t)t * GLM53F_MHC_FLAT + i] = t == 0 ? 0 :
                    ((i * 71 + t * 137) % 1009 - 504) * 0.0071f;
        double old = 0, next = 0;
        for (int rep = -1; rep < 5; ++rep) {
            double begin = omp_get_wtime();
            glm53f_mhc_pre_batch_sve(sa, x, &site, norm, n, sizeof(*sa), a);
            if (rep >= 0) old += omp_get_wtime() - begin;
            begin = omp_get_wtime();
            glm53f_mhc_pre_prefill_sve(sb, x, &site, norm, n, sizeof(*sb), b);
            if (rep >= 0) next += omp_get_wtime() - begin;
            failed |= memcmp(sa, sb, (size_t)n * sizeof(*sa)) != 0;
            failed |= memcmp(a, b, (size_t)n * GLM53F_MHC_WIDTH * sizeof(float)) != 0;
        }
        printf("MHC_PREFILL tokens=%d scratch_exact=%d normalized_exact=%d "
               "old_ms=%.6f new_ms=%.6f speedup=%.3f\n", n,
               !memcmp(sa, sb, (size_t)n * sizeof(*sa)),
               !memcmp(a, b, (size_t)n * GLM53F_MHC_WIDTH * sizeof(float)),
               old * 200, next * 200, old / next);
        free(sb); free(sa); free(b); free(a); free(x);
    }
    free(w);
    return failed;
}
