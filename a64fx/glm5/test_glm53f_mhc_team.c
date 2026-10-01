/* Test the orphaned mHC team and normalized-input publication contract. */
#define _GNU_SOURCE
#include "glm53f_mhc_sve.h"

static void publish(void *context, const float *normalized) {
    float *out = context;
#pragma omp for schedule(static)
    for (int i = 0; i < GLM53F_MHC_WIDTH; ++i) out[i] = normalized[i];
}
int main(void) {
    enum { REPS = 8 };
    const size_t wb = (size_t)GLM53F_MHC_MIX * GLM53F_MHC_FLAT * 2;
    uint16_t *fn = aligned_alloc(256, wb), *norm = aligned_alloc(256, GLM53F_MHC_WIDTH * 2);
    float *a = aligned_alloc(256, GLM53F_MHC_FLAT * 4), *b = aligned_alloc(256, GLM53F_MHC_FLAT * 4);
    float *sub = aligned_alloc(256, GLM53F_MHC_WIDTH * 4), *published = aligned_alloc(256, GLM53F_MHC_WIDTH * 4);
    glm53f_mhc_scratch *sa = malloc(sizeof(*sa)), *sb = malloc(sizeof(*sb));
    float base[GLM53F_MHC_MIX], scale[3] = {0.1f, 0.1f, 0.1f}, logits[GLM53F_MHC_MIX];
    if (!fn || !norm || !a || !b || !sub || !published || !sa || !sb) return 2;
    for (size_t i = 0; i < wb / 2; ++i) fn[i] = (uint16_t)(0x3b00 + (i % 128) + ((i & 1) << 15));
    for (int i = 0; i < GLM53F_MHC_MIX; ++i) base[i] = 0.01f * i;
    for (int i = 0; i < GLM53F_MHC_FLAT; ++i) a[i] = b[i] = sinf(0.013f * i);
    for (int i = 0; i < GLM53F_MHC_WIDTH; ++i) { sub[i] = cosf(0.03f * i); norm[i] = 0x3f80; }
    glm53f_mhc_site site = {fn, base, scale};
    memset(sa, 0, sizeof(*sa)); memset(sb, 0, sizeof(*sb));
    glm53f_mhc_fast(a, NULL, sa, &site, norm, 0);
    for (int r = 0; r < REPS; ++r) glm53f_mhc_fast(a, sub, sa, &site, norm, 1);
#pragma omp parallel shared(logits)
    {
        glm53f_mhc_fast_team(b, NULL, sb, &site, norm, 0, logits, publish, published);
#pragma omp barrier
        for (int r = 0; r < REPS; ++r) {
            glm53f_mhc_fast_team(b, sub, sb, &site, norm, 1, logits, publish, published);
#pragma omp barrier
        }
    }
    int ok = !memcmp(a, b, GLM53F_MHC_FLAT * 4) && !memcmp(sa, sb, sizeof(*sa)) &&
             !memcmp(published, sb->normalized, GLM53F_MHC_WIDTH * 4);
    printf("GLM53F_MHC_TEAM threads=%d calls=%d state=%s publication=%s %s\n", omp_get_max_threads(), REPS + 1,
           !memcmp(sa, sb, sizeof(*sa)) ? "BIT_EXACT" : "FAIL",
           !memcmp(published, sb->normalized, GLM53F_MHC_WIDTH * 4) ? "BIT_EXACT" : "FAIL", ok ? "PASS" : "FAIL");
    free(fn); free(norm); free(a); free(b); free(sub); free(published); free(sa); free(sb);
    return ok ? 0 : 1;
}
