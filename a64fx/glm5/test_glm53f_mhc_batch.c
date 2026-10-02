#define _GNU_SOURCE
#include "glm53f_mhc_sve.h"

enum { MAX_T = 7, GUARD = 128 };
int main(void) {
    const size_t wb = (size_t)GLM53F_MHC_MIX * GLM53F_MHC_FLAT * 2;
    const size_t stride = sizeof(glm53f_mhc_scratch) + GUARD;
    const size_t sb = MAX_T * stride + GUARD;
    uint16_t *weights = aligned_alloc(256, wb);
    uint16_t *norm = aligned_alloc(256, GLM53F_MHC_WIDTH * 2);
    float *streams = aligned_alloc(256, MAX_T * GLM53F_MHC_FLAT * sizeof(float));
    unsigned char *reference = malloc(sb), *candidate = malloc(sb);
    float *a = malloc((MAX_T * GLM53F_MHC_WIDTH + GUARD) * sizeof(float));
    float *b = malloc((MAX_T * GLM53F_MHC_WIDTH + GUARD) * sizeof(float));
    if (!weights || !norm || !streams || !reference || !candidate || !a || !b) return 2;
    if (setenv("GLM53F_MHC_PREFILL", "0", 1)) return 2;
    for (size_t i = 0; i < wb / 2; ++i)
        weights[i] = (uint16_t)(0x3a00 + i * 37 % 768 + ((i & 1) << 15));
    for (int i = 0; i < GLM53F_MHC_WIDTH; ++i) norm[i] = (uint16_t)(0x3f00 + i % 128);
    float base[GLM53F_MHC_MIX], scale[3];
    glm53f_mhc_site site = {weights, base, scale};
    int failed = 0, cases = 0;
    for (int prefill = 0; prefill < 2; ++prefill) {
        if (setenv("GLM53F_MHC_PREFILL", prefill ? "1" : "0", 1)) return 2;
        for (int input = 0; input < 12; ++input) {
            for (int i = 0; i < MAX_T * GLM53F_MHC_FLAT; ++i) {
                float x = sinf(0.013f * i + input * 0.7f);
                streams[i] = input == 0 ? 0.0f : ldexpf(x, input * 2 - 12);
            }
            for (int k = 0; k < GLM53F_MHC_MIX; ++k) base[k] = (k % 7 - 3) * 0.04f;
            for (int k = 0; k < 3; ++k) scale[k] = (input + 1) * 0.02f;
            for (int n = 1; n <= MAX_T; ++n) {
                for (int rep = 0; rep < 4; ++rep) {
                    memset(reference, 0xa5, sb); memset(candidate, 0xa5, sb);
                    memset(a, 0x5a, (MAX_T * GLM53F_MHC_WIDTH + GUARD) * sizeof(float));
                    memset(b, 0x5a, (MAX_T * GLM53F_MHC_WIDTH + GUARD) * sizeof(float));
                    if (setenv("GLM53F_MHC_BATCH_TEAM", "0", 1)) return 2;
                    glm53f_mhc_pre_batch_sve((glm53f_mhc_scratch *)reference, streams,
                        &site, norm, n, stride, a);
                    if (setenv("GLM53F_MHC_BATCH_TEAM", "1", 1)) return 2;
                    glm53f_mhc_pre_batch_sve((glm53f_mhc_scratch *)candidate, streams,
                        &site, norm, n, stride, b);
                    int bad = memcmp(reference, candidate, sb) ||
                        memcmp(a, b, (MAX_T * GLM53F_MHC_WIDTH + GUARD) * sizeof(float));
                    for (int i = 0; i < n * GLM53F_MHC_WIDTH; ++i) {
                        uint32_t bits;
                        memcpy(&bits, a + i, sizeof(bits));
                        bad |= (bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000);
                    }
                    ++cases;
                    if (bad) {
                        fprintf(stderr, "MHC_BATCH_MISMATCH input=%d tokens=%d rep=%d\n", input, n, rep);
                        failed = 1;
                    }
                }
            }
        }
    }
    printf("GLM53F_MHC_BATCH threads=%d cases=%d scratch_and_normalized=%s guards=%s %s\n",
        omp_get_max_threads(), cases, failed ? "FAIL" : "BIT_EXACT",
        failed ? "FAIL" : "BIT_EXACT", failed ? "FAIL" : "PASS");
    free(b); free(a); free(candidate); free(reference); free(streams); free(norm); free(weights);
    return failed ? 1 : 0;
}
