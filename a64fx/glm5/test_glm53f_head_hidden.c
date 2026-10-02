/* Exercise the actual output-norm helper and normalized-hidden export without
 * loading weights or launching MPI. Unused model functions are link-discarded. */
#define GLM53F_TARGET_HEAD_NO_MAIN
#define GLM53F_EXTERNAL_ST_IMPLEMENTATION
#include "glm53f_target_head_12n.c"

int main(void) {
    const int sizes[] = {1, 4, 5, 47, 512, 4096};
    uint16_t norm[H];
    glm53f_target_head_context_12n c = {.norm = norm};
    int failed = 0, cases = 0;
    for (int i = 0; i < H; ++i) {
        float weight = (float)((i * 17) % 127 + 1) / 64;
        uint32_t bits;
        memcpy(&bits, &weight, sizeof(bits)); norm[i] = (uint16_t)(bits >> 16);
    }
    for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); ++s)
        for (int pattern = 0; pattern < 3; ++pattern) {
            int tokens = sizes[s];
            size_t count = (size_t)tokens * H;
            float *actual = malloc((count + 2) * sizeof(float));
            float *expected = malloc(count * sizeof(float));
            if (!actual || !expected) return 2;
            actual[0] = 123; actual[count + 1] = 456;
            for (size_t i = 0; i < count; ++i)
                actual[i + 1] = expected[i] = pattern == 0 ? 0 : pattern == 1 ?
                    (i % H == 17 ? -0.125f : 0) : (float)((i * 31 + 17) % 257) / 128 - 1;
            for (int t = 0; t < tokens; ++t) {
                float *h = expected + (size_t)t * H;
                double sum = 0;
                for (int i = 0; i < H; ++i) sum += (double)h[i] * h[i];
                float inv = 1 / sqrtf((float)(sum / H) + 1e-5f);
                for (int i = 0; i < H; ++i) {
                    uint32_t bits = (uint32_t)norm[i] << 16;
                    float weight; memcpy(&weight, &bits, sizeof(weight));
                    h[i] = h[i] * inv * weight;
                }
            }
            int bad = glm53f_target_head_normalize_12n(&c, actual + 1, tokens);
            for (size_t i = 0; i < count; ++i)
                bad |= !isfinite(actual[i + 1]) ||
                    fabsf(actual[i + 1] - expected[i]) > 1e-6f * (1 + fabsf(expected[i]));
            bad |= actual[0] != 123 || actual[count + 1] != 456;
            if (tokens <= 5) {
                c.x = actual + 1; c.hidden_tokens = tokens;
                bad |= glm53f_target_head_hidden_12n(&c, expected, tokens) ||
                    memcmp(expected, actual + 1, count * sizeof(float));
                bad |= glm53f_target_head_hidden_12n(&c, expected, tokens + 1) != -1;
            }
            if (bad) fprintf(stderr, "HEAD_HIDDEN_FAIL tokens=%d pattern=%d\n", tokens, pattern);
            failed |= !!bad; ++cases;
            free(actual); free(expected);
        }
    float output[H];
    failed |= glm53f_target_head_normalize_12n(NULL, output, 1) != -1 ||
        glm53f_target_head_normalize_12n(&c, NULL, 1) != -1 ||
        glm53f_target_head_normalize_12n(&c, output, 0) != -1 ||
        glm53f_target_head_normalize_12n(&c, output, 4097) != -1;
    c.hidden_tokens = 0;
    failed |= glm53f_target_head_hidden_12n(&c, output, 1) != -1;
    printf("GLM53F_HEAD_HIDDEN %s cases=%d\n", failed ? "FAIL" : "PASS", cases);
    return failed ? 1 : 0;
}
