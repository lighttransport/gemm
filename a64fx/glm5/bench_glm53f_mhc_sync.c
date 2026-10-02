/* Production-shaped 90-site mHC sweep. Compare identical arithmetic with
 * both ordinary OpenMP and the persistent decode controller. */
#define _GNU_SOURCE
#include "glm53f_mhc_sve.h"

enum { SYNC_SITES = 90, SYNC_ROUNDS = 8 };
typedef struct {
    float *streams, *sub;
    glm53f_mhc_scratch *scratch;
    glm53f_mhc_site *site;
    uint16_t *norm;
    double seconds;
} sync_bench;
static void sequence(void *context) {
    sync_bench *a = context;
    glm53f_mhc_fast(a->streams, NULL, a->scratch, &a->site[0], a->norm, 0);
    double begin = omp_get_wtime();
    for (int r = 0; r < SYNC_ROUNDS; ++r)
        for (int s = 0; s < SYNC_SITES; ++s)
            glm53f_mhc_fast(a->streams, a->sub, a->scratch,
                           &a->site[s], a->norm, 1);
    a->seconds = omp_get_wtime() - begin;
}
int main(void) {
    glm53f_mhc_site site[SYNC_SITES];
    float base[SYNC_SITES][GLM53F_MHC_MIX], scale[SYNC_SITES][3];
    const size_t wb = (size_t)GLM53F_MHC_MIX * GLM53F_MHC_FLAT * 2;
    uint16_t *weights = aligned_alloc(256, SYNC_SITES * wb);
    uint16_t *norm = aligned_alloc(256, GLM53F_MHC_WIDTH * 2);
    float *streams = aligned_alloc(256, GLM53F_MHC_FLAT * sizeof(float));
    float *sub = aligned_alloc(256, GLM53F_MHC_WIDTH * sizeof(float));
    glm53f_mhc_scratch *scratch = malloc(sizeof(*scratch));
    if (!weights || !norm || !streams || !sub || !scratch) return 2;
    for (int s = 0; s < SYNC_SITES; ++s) {
        uint16_t *fn = weights + (size_t)s * wb / 2;
        for (size_t i = 0; i < wb / 2; ++i)
            fn[i] = (uint16_t)(0x3a00 + ((i * 37 + s * 19) % 768) + ((i & 1) << 15));
        for (int k = 0; k < GLM53F_MHC_MIX; ++k) base[s][k] = (k % 7 - 3) * 0.01f;
        for (int k = 0; k < 3; ++k) scale[s][k] = 0.1f;
        site[s] = (glm53f_mhc_site){fn, base[s], scale[s]};
    }
    for (int i = 0; i < GLM53F_MHC_WIDTH; ++i) norm[i] = 0x3f80;
    sync_bench a = {streams, sub, scratch, site, norm, 0};
    for (int persistent = 0; persistent < 2; ++persistent)
        for (int rep = -1; rep < 6; ++rep)
            for (int order = 0; order < 2; ++order) {
                int fused = order ^ (rep & 1);
                if (setenv("GLM53F_MHC_FUSED_SYNC", fused ? "1" : "0", 1)) return 2;
                for (int i = 0; i < GLM53F_MHC_FLAT; ++i) streams[i] = sinf(0.013f * i) * 0.3f;
                for (int i = 0; i < GLM53F_MHC_WIDTH; ++i) sub[i] = cosf(0.03f * i) * 0.1f;
                memset(scratch, 0, sizeof(*scratch));
                if (persistent) glm53f_team_run(sequence, &a);
                else sequence(&a);
                if (rep >= 0) {
                    double checksum = 0;
                    for (int i = 0; i < GLM53F_MHC_WIDTH; ++i) checksum += scratch->normalized[i];
                    printf("GLM53F_MHC_SYNC_BENCH threads=%d persistent=%d fused_sync=%d repetition=%d us_call=%.6f checksum=%.9f\n",
                        omp_get_max_threads(), persistent, fused, rep,
                        a.seconds * 1e6 / (SYNC_ROUNDS * SYNC_SITES), checksum);
                }
            }
    free(scratch); free(sub); free(streams); free(norm); free(weights);
    return 0;
}
