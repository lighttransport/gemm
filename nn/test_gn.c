/* SPDX-License-Identifier: MIT */
#include "gn.h"
#undef NDEBUG /* Numerical/accounting gates must also run in Release tests. */
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void ok(int rc) {
    if (rc) {
        fprintf(stderr, "%s\n", gn_error());
        exit(1);
    }
}
int main(int argc, char **argv) {
    assert(argc == 2);
    gn_config c = gn_default_config();
    c.side = 3;
    c.inputs = 4;
    c.actions = 7;
    c.channels = 4;
    c.blocks = 2;
    c.attention_every = 2;
    c.head_dim = 2;
    c.value_channels = 2;
    c.value_hidden = 4;
    gn_model *m = gn_create(&c, "cpu", 0);
    assert(m);
    assert(!gn_create(&c, "unknown", 0));
    float input[2 * 9 * 4], target[2 * 9 * 7], policy[2 * 9 * 7], wdl[6];
    uint32_t label[2] = {0, 2};
    for (size_t i = 0; i < 72; i++)
        input[i] = sinf((float)i * 0.31f);
    for (size_t i = 0; i < 126; i++)
        target[i] = -1;
    for (int b = 0; b < 2; b++) {
        target[b * 63 + 1] = 0.3f;
        target[b * 63 + 17] = 0.7f;
        target[b * 63 + 3] = 0;
    }
    gn_metrics metrics;
    ok(gn_backward(m, 2, input, target, label, &metrics));
    assert(gn_matrix_flops(m) == 104256); /* Explicit small architecture: 3 x 34752. */
    assert(gn_matrix_flops(NULL) == 0);
    assert(gn_gemm_flops(NULL) == 0);
    assert(gn_gemm_flops(m) == 96480); /* Attention: 12*18*4*9 = 7776. */
    float first = metrics.policy + metrics.value;
    size_t checked = 0;
    /* Central differences across every learned tensor, including Q/K/V,
     * relative bias, BN/LN parameters, both residual branches and both heads. */
    for (size_t t = 0; t < gn_tensor_count(m); t++) {
        float *x, *g;
        size_t r, col;
        const char *name = gn_tensor(m, t, &r, &col, &x, &g);
        if (!g)
            continue;
        size_t j = (r * col) / 2;
        gn_zero_grad(m);
        ok(gn_backward(m, 2, input, target, label, &metrics));
        float analytic = g[j] / 2, old = x[j], eps = 0.002f;
        x[j] = old + eps;
        gn_zero_grad(m);
        ok(gn_backward(m, 2, input, target, label, &metrics));
        float plus = metrics.policy + metrics.value;
        x[j] = old - eps;
        gn_zero_grad(m);
        ok(gn_backward(m, 2, input, target, label, &metrics));
        float minus = metrics.policy + metrics.value;
        x[j] = old;
        float numeric = (plus - minus) / (2 * eps), tol = 0.015f + 0.035f * fabsf(numeric);
        if (fabsf(numeric - analytic) > tol) {
            fprintf(stderr, "gradient %s[%zu]: analytic=%g numeric=%g\n", name, j, analytic,
                    numeric);
            return 1;
        }
        checked++;
    }
    gn_zero_grad(m);
    for (int i = 0; i < 80; i++) {
        ok(gn_backward(m, 2, input, target, label, &metrics));
        ok(gn_update(m, 0.01f, 0.0001f, 1, &metrics));
    }
    assert(metrics.policy + metrics.value < first * 0.7f);
    ok(gn_infer(m, 2, input, policy, wdl));
    assert(gn_matrix_flops(m) == 34752);
    assert(gn_gemm_flops(m) == 32160);
    for (int b = 0; b < 2; b++)
        assert(fabsf(wdl[b * 3] + wdl[b * 3 + 1] + wdl[b * 3 + 2] - 1) < 1e-6f);
    ok(gn_save(m, argv[1]));
    gn_model *copy = gn_load(argv[1], "cpu", 0);
    assert(copy);
    float p2[126], v2[6];
    ok(gn_infer(copy, 2, input, p2, v2));
    assert(!memcmp(policy, p2, sizeof(policy)));
    assert(!memcmp(wdl, v2, sizeof(wdl)));
    /* Resume must reproduce the next optimizer update, not only inference. */
    ok(gn_backward(m, 2, input, target, label, &metrics));
    ok(gn_update(m, 0.001f, 0.0001f, 1, &metrics));
    ok(gn_backward(copy, 2, input, target, label, &metrics));
    ok(gn_update(copy, 0.001f, 0.0001f, 1, &metrics));
    ok(gn_infer(m, 2, input, policy, wdl));
    ok(gn_infer(copy, 2, input, p2, v2));
    assert(!memcmp(policy, p2, sizeof(policy)));
    target[0] = NAN;
    assert(gn_backward(m, 2, input, target, label, &metrics) < 0);
    input[0] = NAN;
    assert(gn_infer(m, 2, input, policy, wdl) < 0);
    c.memory_limit = 1024;
    assert(!gn_create(&c, "cpu", 0));
    printf(
        "PASS: %zu tensor gradients, loss %.6f -> %.6f, exact checkpoint/resume, invalid inputs\n",
        checked, first, metrics.policy + metrics.value);
    gn_destroy(copy);
    gn_destroy(m);
    return 0;
}
