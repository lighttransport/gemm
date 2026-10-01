#include "glm53f_moe_combine.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
int main(void) {
    enum { H = 4096 };
    float rows[8][H], shared[H], expected[H], actual[H], weight[8];
    const float *route[8];
    for (int k = 0; k < 8; ++k) {
        weight[k] = (k + 1) * .03125f;
        for (int i = 0; i < H; ++i) rows[k][i] = (float)((i * 31 + k * 17) % 997 - 498) / 127;
    }
    for (int i = 0; i < H; ++i) shared[i] = (float)(i % 37 - 18) * .017f;
    int failed = 0;
    for (int mask = 0; mask < 256; ++mask) {
        for (int k = 0; k < 8; ++k) route[k] = mask & (1 << k) ? rows[k] : NULL;
        for (int i = 0; i < H; ++i) {
            float sum = 0;
            for (int k = 0; k < 8; ++k) if (route[k]) sum = fmaf(weight[k], route[k][i], sum);
            expected[i] = shared[i] + sum;
        }
        memcpy(actual, shared, sizeof(actual));
#pragma omp parallel for schedule(static)
        for (int i = 0; i < H; i += 63) glm53f_moe_combine_rows(actual, route, weight, i, i + 63 < H ? i + 63 : H);
        failed |= memcmp(actual, expected, sizeof(actual)) != 0;
    }
    printf("GLM53F_MOE_COMBINE %s masks=256 ragged_rows=63\n", failed ? "FAIL" : "PASS");
    return failed;
}
