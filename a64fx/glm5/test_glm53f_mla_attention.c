#include "glm53f_mla_attention.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

int main(void) {
    enum { HEADS = 6, STRIDE = 768, SLOTS = GLM53F_MLA_ATTENTION_SLOTS, LATENT = 512 };
    float *cache = malloc((size_t)SLOTS * LATENT * sizeof(float));
    uint16_t *half = malloc((size_t)SLOTS * LATENT * sizeof(uint16_t));
    float query[HEADS * LATENT], expected[HEADS * STRIDE], actual[HEADS * STRIDE];
    float old_log[HEADS * SLOTS], new_log[HEADS * SLOTS];
    int selected[SLOTS], bad = 0, cases = 0;
    if (!cache || !half || svcntw() != 16) return 2;
    uint32_t seed = 173;
    for (int i = 0; i < SLOTS * LATENT; ++i) {
        seed = seed * 1664525u + 1013904223u;
        cache[i] = ((int)(seed >> 8) % 20001 - 10000) * .0001f;
    }
    for (int i = 0; i < HEADS * LATENT; ++i) {
        seed = seed * 1664525u + 1013904223u;
        query[i] = ((int)(seed >> 8) % 20001 - 10000) * .0001f;
    }
    glm53f_mla_cache_f16_store(half, cache, SLOTS * LATENT);
    const int counts[] = {1, 7, 128, 2048, 2051};
    for (int order = 0; order < 2; ++order) {
        for (int i = 0; i < SLOTS; ++i) selected[i] = order ? SLOTS - 1 - i : i;
        for (int heads = 1; heads <= HEADS; ++heads) {
            for (unsigned ci = 0; ci < sizeof(counts) / sizeof(counts[0]); ++ci) {
                memset(expected, 0xa5, sizeof(expected)); memset(actual, 0xa5, sizeof(actual));
                memset(old_log, 0xa5, sizeof(old_log)); memset(new_log, 0xa5, sizeof(new_log));
                mlb_token(expected, STRIDE, old_log, query, cache, NULL, selected, counts[ci], heads);
                mlb_token(actual, STRIDE, new_log, query, cache, half, selected, counts[ci], heads);
                bad |= memcmp(expected, actual, sizeof(actual)) != 0;
                bad |= memcmp(old_log, new_log, sizeof(new_log)) != 0;
                ++cases;
            }
        }
    }
    printf("MLA_ATTENTION heads=1..6 cases=%d logits_values_and_canaries=%s %s\n",
        cases, bad ? "FAIL" : "BIT_EXACT", bad ? "FAIL" : "PASS");
    free(cache); free(half);
    return bad;
}
