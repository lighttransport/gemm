#define _GNU_SOURCE
#include "glm53f_mhc_sve.h"

enum { SITES = 90, ROUNDS = 4, MAX_T = 5 };
static uint64_t hash_bytes(const void *p, size_t bytes) {
    const unsigned char *s = p;
    uint64_t h = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < bytes; ++i) h = (h ^ s[i]) * UINT64_C(1099511628211);
    return h;
}
int main(void) {
    const size_t wb = (size_t)GLM53F_MHC_MIX * GLM53F_MHC_FLAT * 2;
    uint16_t *weights = aligned_alloc(256, SITES * wb);
    uint16_t *norm = aligned_alloc(256, GLM53F_MHC_WIDTH * 2);
    float *streams = aligned_alloc(256, MAX_T * GLM53F_MHC_FLAT * sizeof(float));
    float *out = aligned_alloc(256, MAX_T * GLM53F_MHC_WIDTH * sizeof(float));
    glm53f_mhc_scratch *scratch = malloc(MAX_T * sizeof(*scratch));
    glm53f_mhc_site site[SITES];
    float base[SITES][GLM53F_MHC_MIX], scale[SITES][3];
    if (!weights || !norm || !streams || !out || !scratch) return 2;
    if (setenv("GLM53F_MHC_PREFILL", "0", 1)) return 2;
    for (int s = 0; s < SITES; ++s) {
        uint16_t *fn = weights + (size_t)s * wb / 2;
        for (size_t i = 0; i < wb / 2; ++i)
            fn[i] = (uint16_t)(0x3a00 + (i * 37 + s * 19) % 768 + ((i & 1) << 15));
        for (int k = 0; k < GLM53F_MHC_MIX; ++k) base[s][k] = (k % 7 - 3) * 0.04f;
        for (int k = 0; k < 3; ++k) scale[s][k] = 0.1f;
        site[s] = (glm53f_mhc_site){fn, base[s], scale[s]};
    }
    for (int i = 0; i < GLM53F_MHC_WIDTH; ++i) norm[i] = (uint16_t)(0x3f00 + i % 128);
    for (int i = 0; i < MAX_T * GLM53F_MHC_FLAT; ++i) streams[i] = sinf(0.013f * i) * 0.3f;
    for (int n = 2; n <= MAX_T; ++n) {
        if (n == 3) continue;
        uint64_t expected = 0;
        for (int rep = -1; rep < 7; ++rep) {
            for (int order = 0; order < 2; ++order) {
                int mode = order ^ ((rep + 1) & 1);
                if (setenv("GLM53F_MHC_BATCH_TEAM", mode ? "1" : "0", 1)) return 2;
                memset(scratch, 0, MAX_T * sizeof(*scratch));
                memset(out, 0, MAX_T * GLM53F_MHC_WIDTH * sizeof(float));
                double begin = omp_get_wtime();
                for (int r = 0; r < ROUNDS; ++r) {
                    for (int s = 0; s < SITES; ++s) {
                        glm53f_mhc_pre_batch_sve(scratch, streams, &site[s], norm, n,
                            sizeof(*scratch), out);
                    }
                }
                double dt = omp_get_wtime() - begin;
                uint64_t h = hash_bytes(out, MAX_T * GLM53F_MHC_WIDTH * sizeof(float)) ^
                    hash_bytes(scratch, MAX_T * sizeof(*scratch));
                if (rep == -1 && order == 0) expected = h;
                if (h != expected) {
                    fprintf(stderr, "GLM53F_MHC_BATCH_BENCH_HASH_FAIL tokens=%d mode=%d rep=%d\n", n, mode, rep);
                    return 1;
                }
                if (rep >= 0) printf("GLM53F_MHC_BATCH_BENCH threads=%d tokens=%d mode=%d repetition=%d us_call=%.6f hash=%016llx\n",
                    omp_get_max_threads(), n, mode, rep, dt * 1e6 / (ROUNDS * SITES),
                    (unsigned long long)h);
            }
        }
    }
    free(scratch); free(out); free(streams); free(norm); free(weights);
    return 0;
}
