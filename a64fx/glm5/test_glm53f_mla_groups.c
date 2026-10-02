#include "glm53f_mla_attention.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
int main(void) {
    enum { HEADS = 16, STRIDE = 768, SLOTS = GLM53F_MLA_ATTENTION_SLOTS, LAT = 512 };
    float *cache = malloc((size_t)SLOTS * LAT * sizeof(float));
    uint16_t *half = malloc((size_t)SLOTS * LAT * sizeof(uint16_t));
    float query[HEADS * LAT], expected[HEADS * STRIDE], actual[HEADS * STRIDE];
    float old_log[HEADS * SLOTS], scratch[6 * SLOTS + 16];
    int selected[SLOTS], bad = 0, cases = 0;
    if (!cache || !half || svcntw() != 16) return 2;
    uint32_t seed = 173;
    for (int i = 0; i < SLOTS * LAT; ++i) {
        seed = seed * 1664525u + 1013904223u;
        cache[i] = ((int)(seed >> 8) % 20001 - 10000) * .0001f;
    }
    for (int i = 0; i < HEADS * LAT; ++i) {
        seed = seed * 1664525u + 1013904223u;
        query[i] = ((int)(seed >> 8) % 20001 - 10000) * .0001f;
    }
    glm53f_mla_cache_f16_store(half, cache, SLOTS * LAT);
    const int counts[] = {1, 3, 7, 15, 16, 17, 128, 2048, 2051, 2052};
    for (int view = 0; view < 2; ++view) for (int order = 0; order < 2; ++order) {
        for (int i = 0; i < SLOTS; ++i) selected[i] = order ? SLOTS - 1 - i : i;
        for (int heads = 1; heads <= HEADS; ++heads)
            for (size_t ci = 0; ci < sizeof(counts) / sizeof(counts[0]); ++ci) {
                memset(expected, 0xa5, sizeof(expected)); memset(actual, 0xa5, sizeof(actual));
                memset(old_log, 0xa5, sizeof(old_log)); memset(scratch, 0xa5, sizeof(scratch));
                const uint16_t *derived = view ? half : NULL;
                for (int h = 0; h < heads; ++h)
                    mlb_token(expected + (size_t)h * STRIDE, STRIDE,
                        old_log + (size_t)h * SLOTS, query + (size_t)h * LAT,
                        cache, derived, selected, counts[ci], 1);
                bad |= mlb_token_grouped(actual, STRIDE, scratch, query,
                    cache, derived, selected, counts[ci], heads) != 0;
                bad |= memcmp(expected, actual, sizeof(actual)) != 0;
                int base = heads <= 6 ? 0 : ((heads - 1) / 4) * 4;
                for (int h = 0; h < heads - base; ++h)
                    bad |= memcmp(scratch + h * SLOTS, old_log + (base + h) * SLOTS,
                        (size_t)counts[ci] * sizeof(float)) != 0;
                for (int j = (heads <= 6 ? heads : 4) * SLOTS; j < 6 * SLOTS + 16; ++j) {
                    uint32_t bits; memcpy(&bits, scratch + j, sizeof(bits));
                    bad |= bits != UINT32_C(0xa5a5a5a5);
                }
                ++cases;
            }
    }
    bad |= mlb_token_grouped(actual, STRIDE, scratch, query, cache, half, selected, 0, 16) != -1;
    bad |= mlb_token_grouped(actual, STRIDE, scratch, query, cache, half, selected, 1, 17) != -1;
    printf("GLM53F_MLA_GROUPS heads=1:16 cases=%d values_logits_canaries=%s %s\n",
        cases, bad ? "FAIL" : "BIT_EXACT", bad ? "FAIL" : "PASS");
    free(cache); free(half); return bad;
}
