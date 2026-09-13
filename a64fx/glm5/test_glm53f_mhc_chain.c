/* Native A64FX arithmetic check for the scalar mHC post/pre chaining. */
#include "glm53f_mhc_sve.h"
#include <stdio.h>
#include <stdlib.h>

static uint32_t rng = 19;
static float next_float(void) {
    rng = rng * 1664525u + 1013904223u;
    return ((int)(rng >> 8) % 20001 - 10000) * 0.0001f;
}

int main(void) {
    size_t weights = (size_t)GLM53F_MHC_MIX * GLM53F_MHC_FLAT;
    uint16_t *first_weight = malloc(weights * 2), *next_weight = malloc(weights * 2);
    uint16_t *first_norm = malloc(GLM53F_MHC_WIDTH * 2), *next_norm = malloc(GLM53F_MHC_WIDTH * 2);
    float *first_base = malloc(GLM53F_MHC_MIX * 4), *next_base = malloc(GLM53F_MHC_MIX * 4);
    float *a = malloc(GLM53F_MHC_FLAT * 4), *b = malloc(GLM53F_MHC_FLAT * 4);
    float *sublayer = malloc(GLM53F_MHC_WIDTH * 4);
    glm53f_mhc_scratch *sa = malloc(sizeof(*sa)), *sb = malloc(sizeof(*sb));
    float scale[3] = {0.25f, 0.5f, 0.125f};
    if (!first_weight || !next_weight || !first_norm || !next_norm ||
        !first_base || !next_base || !a || !b || !sublayer || !sa || !sb) return 2;
    for (size_t i = 0; i < weights; ++i) {
        float x = next_float() * 0.002f; uint32_t bits;
        memcpy(&bits, &x, 4); first_weight[i] = (uint16_t)(bits >> 16);
        x = next_float() * 0.002f; memcpy(&bits, &x, 4); next_weight[i] = (uint16_t)(bits >> 16);
    }
    for (int i = 0; i < GLM53F_MHC_WIDTH; ++i) {
        uint32_t bits = 0x3f800000u;
        first_norm[i] = next_norm[i] = (uint16_t)(bits >> 16);
        sublayer[i] = next_float();
    }
    for (int i = 0; i < GLM53F_MHC_MIX; ++i)
        first_base[i] = next_base[i] = next_float() * 0.01f;
    for (int i = 0; i < GLM53F_MHC_FLAT; ++i) a[i] = b[i] = next_float();
    glm53f_mhc_site first = {first_weight, first_base, scale};
    glm53f_mhc_site next = {next_weight, next_base, scale};
    glm53f_mhc_pre_sve(sa, a, &first, first_norm);
    memcpy(sb, sa, sizeof(*sa));
    glm53f_mhc_post_sve(a, sublayer, sa);
    glm53f_mhc_pre_sve(sa, a, &next, next_norm);
    glm53f_mhc_post_pre_sve(b, sublayer, sb, &next, next_norm);
    int mismatch = memcmp(a, b, GLM53F_MHC_FLAT * 4) || memcmp(sa, sb, sizeof(*sa));
    printf("MHC_CHAIN streams_match=%d scratch_match=%d %s\n",
        !memcmp(a, b, GLM53F_MHC_FLAT * 4), !memcmp(sa, sb, sizeof(*sa)),
        mismatch ? "FAIL" : "PASS");
    free(sb); free(sa); free(sublayer); free(b); free(a); free(next_base); free(first_base);
    free(next_norm); free(first_norm); free(next_weight); free(first_weight);
    return mismatch != 0;
}
