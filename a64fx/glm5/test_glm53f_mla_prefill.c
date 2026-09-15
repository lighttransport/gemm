/* Compare independent register blocking with both established MLA orders. */
#define main glm53f_unused_sparse_main
#include "glm53f_sparse_core_12n.c"
#undef main
#include "glm53f_mla_prefill.h"

int main(void) {
    enum { ROWS = 8192, TASKS = 192 };
    float *cache = a256((size_t)ROWS * LAT * sizeof(float));
    uint16_t *w = a256((size_t)6 * (KD + VD) * LAT * sizeof(uint16_t));
    float *q = a256((size_t)TASKS * KD * sizeof(float));
    float *ref = a256((size_t)TASKS * VD * sizeof(float));
    float *out = a256((size_t)TASKS * VD * sizeof(float));
    int selected[2052], failed = 0;
    const int counts[] = {1,4,5,31,32,33,255,256,257,511,512,513,2047,2048,2049,2050,2051};
    for (int i = 0; i < ROWS * LAT; ++i)
        cache[i] = ((i * 17 % 1009) - 504) * 0.0013f;
    for (int i = 0; i < 6 * (KD + VD) * LAT; ++i) {
        float v = ((i * 71 % 1009) - 504) * 0.00011f;
        uint32_t bits; memcpy(&bits, &v, sizeof(bits)); w[i] = bits >> 16;
    }
    for (int i = 0; i < TASKS * KD; ++i)
        q[i] = i < KD ? 0 : ((i * 31 % 1009) - 504) * 0.0031f;
    for (unsigned n = 0; n < sizeof(counts)/sizeof(counts[0]); ++n) {
        int count = counts[n];
        for (int t = 0; t < count; ++t) selected[t] = (t * 104729) % ROWS;
        for (int shards = 1; shards <= 8; shards += 7) {
            double elapsed[2] = {0};
            for (int rep = -1; rep < 3; ++rep)
                for (int mode = 0; mode < 2; ++mode) {
                    double begin = omp_get_wtime();
#pragma omp parallel for schedule(static) reduction(|:failed)
                    for (int t = 0; t < TASKS; ++t) {
                        const uint16_t *wh = w + (size_t)(t % 6) * (KD + VD) * LAT;
                        if (mode)
                            failed |= glm53f_mla_prefill_one(out + (size_t)t * VD,
                                q + (size_t)t * KD, cache, wh, selected, count, shards) != 0;
                        else if (shards == 1)
                            failed |= mla_one(ref + (size_t)t * VD,
                                q + (size_t)t * KD, cache, wh, selected, count) != 0;
                        else
                            failed |= mla_one_sharded_indexed(ref + (size_t)t * VD,
                                q + (size_t)t * KD, cache, wh, selected, count) != 0;
                    }
                    if (rep >= 0) elapsed[mode] += omp_get_wtime() - begin;
                    if (mode) failed |= memcmp(ref, out, (size_t)TASKS * VD * sizeof(float)) != 0;
                }
            printf("MLA_PREFILL selected=%d shards=%d exact=%d old_ms=%.3f new_ms=%.3f speedup=%.3f\n",
                count, shards, !failed, elapsed[0]*1e3/3, elapsed[1]*1e3/3, elapsed[0]/elapsed[1]);
        }
    }
    failed |= glm53f_mla_prefill_one(out,q,cache,w,selected,0,1) == 0;
    failed |= glm53f_mla_prefill_one(out,q,cache,w,selected,2052,1) == 0;
    failed |= glm53f_mla_prefill_one(out,q,cache,w,selected,32,2) == 0;
    printf("MLA_PREFILL_RESULT %s\n", failed ? "FAIL" : "PASS");
    free(out); free(ref); free(q); free(w); free(cache);
    return failed;
}
