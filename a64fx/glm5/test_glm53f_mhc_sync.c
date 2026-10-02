/* Compare every endpoint of chained pre/post calls, including normalized
 * input publication, with OpenMP teams and the persistent decode executor. */
#define _GNU_SOURCE
#include "glm53f_mhc_sve.h"

enum { SYNC_REPS = 32 };
typedef struct {
    float streams[GLM53F_MHC_FLAT];
    glm53f_mhc_scratch scratch;
} sync_endpoint;
typedef struct {
    float *streams, *sub, *published;
    glm53f_mhc_scratch *scratch;
    const glm53f_mhc_site *site;
    const uint16_t *norm;
    sync_endpoint *reference;
    int record, failed, calls;
} sync_test;

static int finite_values(const float *v, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        uint32_t bits;
        memcpy(&bits, &v[i], sizeof(bits));
        if ((bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000)) return 0;
    }
    return 1;
}
static void publish(void *context, const float *normalized) {
    float *out = context;
#pragma omp for schedule(static)
    for (int i = 0; i < GLM53F_MHC_WIDTH; ++i) out[i] = normalized[i];
}
static void sequence(void *context) {
    sync_test *a = context;
    for (int r = 0; r < SYNC_REPS; ++r) {
        /* Repeated pre-only calls also cover reuse of the completion counter
         * without depending on a preceding residual update. */
        glm53f_mhc_fast_route(a->streams, a->sub, a->scratch,
            &a->site[r % 4], a->norm, r % 5 != 0, publish, a->published);
        a->failed |= !finite_values(a->streams, GLM53F_MHC_FLAT) ||
                     !finite_values(a->scratch->normalized, GLM53F_MHC_WIDTH);
        sync_endpoint *e = &a->reference[r];
        if (a->record) {
            memcpy(e->streams, a->streams, sizeof(e->streams));
            memcpy(&e->scratch, a->scratch, sizeof(e->scratch));
        } else {
            a->failed |= memcmp(e->streams, a->streams, sizeof(e->streams)) != 0;
            a->failed |= memcmp(&e->scratch, a->scratch, sizeof(e->scratch)) != 0;
        }
        a->failed |= memcmp(a->published, a->scratch->normalized,
                           GLM53F_MHC_WIDTH * sizeof(float)) != 0;
        ++a->calls;
    }
}
int main(void) {
    const size_t wb = (size_t)GLM53F_MHC_MIX * GLM53F_MHC_FLAT * 2;
    uint16_t *fn = aligned_alloc(256, wb);
    uint16_t *norm = aligned_alloc(256, GLM53F_MHC_WIDTH * 2);
    float *streams = aligned_alloc(256, GLM53F_MHC_FLAT * sizeof(float));
    float *sub = aligned_alloc(256, GLM53F_MHC_WIDTH * sizeof(float));
    float *published = aligned_alloc(256, GLM53F_MHC_WIDTH * sizeof(float));
    glm53f_mhc_scratch *scratch = malloc(sizeof(*scratch));
    sync_endpoint *reference = malloc(SYNC_REPS * sizeof(*reference));
    if (!fn || !norm || !streams || !sub || !published || !scratch || !reference) return 2;
    for (size_t i = 0; i < wb / 2; ++i)
        fn[i] = (uint16_t)(0x3a00 + (i * 37 % 768) + ((i & 1) << 15));
    float base[4][GLM53F_MHC_MIX], scale[4][3];
    glm53f_mhc_site site[4];
    for (int s = 0; s < 4; ++s) {
        for (int k = 0; k < GLM53F_MHC_MIX; ++k)
            base[s][k] = (k % 7 - 3) * (s + 1) * 0.08f;
        for (int k = 0; k < 3; ++k) scale[s][k] = (s + 1) * 0.06f;
        site[s] = (glm53f_mhc_site){fn, base[s], scale[s]};
    }
    for (int i = 0; i < GLM53F_MHC_WIDTH; ++i) norm[i] = (uint16_t)(0x3f00 + i % 128);
    sync_test a = {streams, sub, published, scratch, site, norm, reference, 0, 0, 0};
    for (int input = 0; input < 4; ++input)
        for (int path = 0; path < 3; ++path) {
            for (int i = 0; i < GLM53F_MHC_FLAT; ++i)
                streams[i] = input == 0 ? 0.0f : sinf(0.013f * i) * (input == 3 ? 20.0f : 0.3f * input);
            for (int i = 0; i < GLM53F_MHC_WIDTH; ++i)
                sub[i] = input == 0 ? 0.0f : cosf(0.03f * i) * input;
            memset(scratch, 0, sizeof(*scratch));
            memset(published, 0, GLM53F_MHC_WIDTH * sizeof(float));
            if (setenv("GLM53F_MHC_FUSED_SYNC", path ? "1" : "0", 1)) return 2;
            a.record = path == 0;
            if (path == 2) glm53f_team_run(sequence, &a);
            else sequence(&a);
        }
    printf("GLM53F_MHC_SYNC threads=%d calls=%d streams=BIT_%s scratch=BIT_%s publication=%s %s\n",
        omp_get_max_threads(), a.calls, a.failed ? "FAIL" : "EXACT", a.failed ? "FAIL" : "EXACT",
        a.failed ? "FAIL" : "EXACT", a.failed ? "FAIL" : "PASS");
    free(reference); free(scratch); free(published); free(sub); free(streams); free(norm); free(fn);
    return a.failed ? 1 : 0;
}
