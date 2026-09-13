/* SPDX-License-Identifier: MIT */
#include "gn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int check(int status) {
    if (status)
        fprintf(stderr, "%s\n", gn_error());
    return status;
}
static double relative(const float *a, const float *b, size_t n) {
    double delta = 0, base = 0;
    for (size_t i = 0; i < n; i++) {
        double d = a[i] - b[i];
        delta += d * d;
        base += (double)a[i] * a[i];
    }
    return sqrt(delta / fmax(base, 1e-12));
}
int main(int argc, char **argv) {
    if (argc != 2 && argc != 3)
        return 2;
    gn_config c = gn_default_config();
    c.side = 3;
    c.inputs = 5;
    c.channels = 8;
    c.actions = 7;
    c.blocks = 2;
    c.attention_every = 2;
    c.head_dim = 4;
    c.value_channels = 3;
    c.value_hidden = 8;
    gn_model *gpu = gn_create(&c, argv[1], 0);
    if (!gpu) {
        fprintf(stderr, "UNAVAILABLE %s: %s\n", argv[1], gn_error());
        return 77;
    }
    gn_model *cpu = gn_create(&c, "cpu", 0);
    if (!cpu) {
        gn_destroy(gpu);
        return 1;
    }
    float x[90], target[126], p[126], q[126], v[6], w[6];
    uint32_t labels[2] = {0, 2};
    for (int i = 0; i < 90; i++)
        x[i] = sinf((float)i * .19f);
    for (int i = 0; i < 126; i++)
        target[i] = -1;
    for (int b = 0; b < 2; b++) {
        target[b * 63 + 1] = .3f;
        target[b * 63 + 31] = .7f;
        target[b * 63 + 8] = 0;
    }
    int rc = 1;
    gn_metrics a, b;
    double tol = strstr(argv[1], "fp32") ? 0.002 : 0.08;
    if (check(gn_infer(cpu, 2, x, p, v)) || check(gn_infer(gpu, 2, x, q, w)))
        goto done;
    double inference = relative(p, q, 126);
    if (!isfinite(inference) || inference > tol || relative(v, w, 6) > tol) {
        fprintf(stderr, "GPU forward mismatch %.9g\n", inference);
        goto done;
    }
    if (check(gn_backward(cpu, 2, x, target, labels, &a)) ||
        check(gn_backward(gpu, 2, x, target, labels, &b)))
        goto done;
    double total_delta = 0, total_base = 0;
    for (size_t i = 0; i < gn_tensor_count(cpu); i++) {
        size_t r, n;
        float *cg, *gg;
        gn_tensor(cpu, i, &r, &n, NULL, &cg);
        if (!cg)
            continue;
        if (!gn_tensor(gpu, i, NULL, NULL, NULL, &gg))
            goto done;
        for (size_t j = 0; j < r * n; j++) {
            double d = cg[j] - gg[j];
            total_delta += d * d;
            total_base += (double)cg[j] * cg[j];
        }
        /* After independent gradient comparison, use identical gradients to
         * isolate AdamW correctness from BF16 operator roundoff. */
        memcpy(cg, gg, r * n * sizeof(float));
    }
    double grad = sqrt(total_delta / fmax(total_base, 1e-12));
    if (!isfinite(grad) || grad > tol ||
        fabs(a.policy - b.policy) + fabs(a.value - b.value) > tol) {
        fprintf(stderr, "GPU backward mismatch %.9g\n", grad);
        goto done;
    }
    if (check(gn_update(cpu, .001f, .0001f, 1, &a)) ||
        check(gn_update(gpu, .001f, .0001f, 1, &b)) || check(gn_infer(gpu, 2, x, q, w)))
        goto done;
    if (b.step != 1)
        goto done;
    for (size_t i = 0; i < gn_tensor_count(cpu); i++) {
        size_t r, n;
        float *cx, *gx, *cg;
        gn_tensor(cpu, i, &r, &n, &cx, &cg);
        if (!cg)
            continue;
        if (!gn_tensor(gpu, i, NULL, NULL, &gx, NULL))
            goto done;
        for (size_t j = 0; j < r * n; j++)
            if (!isfinite(gx[j]) || fabsf(cx[j] - gx[j]) > 2e-6f + 2e-5f * fabsf(cx[j])) {
                fprintf(stderr, "GPU AdamW parameter mismatch tensor %zu index %zu\n", i, j);
                goto done;
            }
    }
    if (argc == 3) {
        if (check(gn_save(gpu, argv[2])))
            goto done;
        gn_model *resumed = gn_load(argv[2], argv[1], 0);
        if (!resumed)
            goto done;
        int status = gn_infer(resumed, 2, x, p, v);
        gn_destroy(resumed);
        if (check(status) || memcmp(p, q, sizeof(p)) || memcmp(v, w, sizeof(v))) {
            fprintf(stderr, "GPU checkpoint inference mismatch\n");
            goto done;
        }
    }
    printf("PASS %s native forward/backward/update: relative output %.8g gradient %.8g\n", argv[1],
           inference, grad);
    rc = 0;
done:
    gn_destroy(cpu);
    gn_destroy(gpu);
    return rc;
}
