/* SPDX-License-Identifier: MIT */
#include "gn_internal.h"
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
    if (argc < 2 || argc > 6 || (argc == 6 && strcmp(argv[5], "report")) ||
        (argc >= 4 && strcmp(argv[3], "wide") && strcmp(argv[3], "full") &&
         strcmp(argv[3], "stress") && strcmp(argv[3], "silu") && strcmp(argv[3], "attention")))
        return 2;
    int B = 2;
    int report = argc == 6, numerical_failure = 0;
    if (argc >= 5) {
        char *end;
        long batch = strtol(argv[4], &end, 10);
        if (*end || batch < 1 || batch > 64)
            return 2;
        B = (int)batch;
    }
    gn_config c = gn_default_config();
    c.side = argc >= 4 ? 9 : 3;
    c.inputs = 5;
    c.channels = argc >= 4 ? 32 : 8;
    c.actions = 7;
    c.blocks = 2;
    c.attention_every = 2;
    c.head_dim = argc >= 4 ? 8 : 4;
    c.value_channels = 3;
    c.value_hidden = 8;
    if (argc >= 4 &&
        (!strcmp(argv[3], "full") || !strcmp(argv[3], "stress") || !strcmp(argv[3], "silu")))
        c = gn_default_config();
    if (argc >= 4 && !strcmp(argv[3], "silu"))
        c.version = 2;
    /* Small full-width heads exercise CUDA's 81-token grouped attention and
     * fused SiLU/BN under sanitizers without allocating the 20-block model. */
    if (argc >= 4 && !strcmp(argv[3], "attention")) {
        c.version = 2;
        c.channels = 64;
        c.head_dim = 32;
    }
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
    int X = (int)c.side * c.side * c.inputs, A = (int)c.side * c.side * c.actions;
    float *x = malloc(B * X * sizeof(float)), *target = malloc(B * A * sizeof(float));
    float *p = malloc(B * A * sizeof(float)), *q = malloc(B * A * sizeof(float));
    float *v = malloc(B * 3 * sizeof(float)), *w = malloc(B * 3 * sizeof(float));
    uint32_t *labels = malloc(B * sizeof(*labels));
    int rc = 1;
    if (!x || !target || !p || !q || !v || !w || !labels)
        goto done;
    for (int b = 0; b < B; b++)
        labels[b] = b % 2 ? 2 : 0;
    uint32_t random = 12345;
    for (int i = 0; i < B * X; i++) {
        random = random * 1664525U + 1013904223U;
        x[i] = c.blocks == 20 && strcmp(argv[3], "stress")
                   ? (float)(random >> 8) * (2.0f / 16777216.0f) - 1
                   : sinf((float)i * .19f);
    }
    for (int i = 0; i < B * A; i++)
        target[i] = -1;
    for (int b = 0; b < B; b++) {
        target[b * A + 1] = .3f;
        target[b * A + 31] = .7f;
        target[b * A + 8] = 0;
    }
    gn_metrics a, b;
    double tol = strstr(argv[1], "fp32") ? 0.0001 : 0.01;
    if (check(gn_infer(cpu, B, x, p, v)) || check(gn_infer(gpu, B, x, q, w)))
        goto done;
    double inference = relative(p, q, B * A);
    double value_inference = relative(v, w, B * 3);
    if (!isfinite(inference) || !isfinite(value_inference) || inference > tol ||
        value_inference > tol) {
        fprintf(stderr, "GPU forward mismatch %.9g\n", inference);
        if (!report)
            goto done;
        numerical_failure = 1;
    }
    if (check(gn_backward(cpu, B, x, target, labels, &a)) ||
        check(gn_backward(gpu, B, x, target, labels, &b)))
        goto done;
    /* Three passes cover warmup, capture+launch, and cached replay. Equal
     * accumulation on both devices preserves the averaged optimizer update. */
    if (report && !strncmp(argv[1], "hip", 3)) {
        for (int repeat = 0; repeat < 2; repeat++)
            if (check(gn_backward(cpu, B, x, target, labels, &a)) ||
                check(gn_backward(gpu, B, x, target, labels, &b)))
                goto done;
        if (strstr(argv[1], "fp16back")) {
            uint64_t captures, launches;
            if (gn_gpu_graph_stats(gpu, &captures, &launches) != 1 ||
                captures != 1 || launches != 2) {
                fprintf(stderr, "HIP qualification did not exercise capture and cached replay\n");
                goto done;
            }
            fprintf(stderr, "HIP backward graph: captures=%llu launches=%llu\n",
                    (unsigned long long)captures, (unsigned long long)launches);
        }
    }
    if (c.blocks == 20)
        fprintf(stderr, "training losses CPU %.9g %.9g GPU %.9g %.9g\n", a.policy, a.value,
                b.policy, b.value);
    if (report) {
        if (check(gn_gpu_debug_nodes(gpu)))
            goto done;
        const char *kind[] = {"input", "linear", "conv", "add", "mul",
                              "relu",  "silu",   "bn",   "ln",  "attention"};
        for (size_t i = 0; i < cpu->nn; i++) {
            Node *cn = &cpu->n[i], *gn = &gpu->n[i];
            size_t count = cn->r * cn->c, flips = 0;
            float largest_flip = 0;
            if (cn->kind == RELU) {
                Node *ca = &cpu->n[cn->a], *ga = &gpu->n[gn->a];
                for (size_t j = 0; j < count; j++) {
                    if ((ca->x[j] > 0) != (ga->x[j] > 0)) {
                        flips++;
                        largest_flip = fmaxf(largest_flip, fmaxf(fabsf(ca->x[j]), fabsf(ga->x[j])));
                        if (flips <= 4)
                            fprintf(stderr,
                                    "relu flip node=%zu index=%zu cpu=%.9g gpu=%.9g "
                                    "upstream_grad_cpu=%.9g "
                                    "upstream_grad_gpu=%.9g\n",
                                    i, j, ca->x[j], ga->x[j], cn->g[j], gn->g[j]);
                    }
                }
            }
            fprintf(stderr,
                    "node %zu %s %zux%zu: value_rel=%.8g grad_rel=%.8g relu_flips=%zu "
                    "largest_flip=%.9g\n",
                    i, kind[cn->kind], cn->r, cn->c, relative(cn->x, gn->x, count),
                    relative(cn->g, gn->g, count), flips, largest_flip);
        }
    }
    double total_delta = 0, total_base = 0, total_gpu = 0, total_dot = 0;
    for (size_t i = 0; i < gn_tensor_count(cpu); i++) {
        size_t r, n;
        float *cg, *gg;
        const char *name = gn_tensor(cpu, i, &r, &n, NULL, &cg);
        if (!cg)
            continue;
        if (!gn_tensor(gpu, i, NULL, NULL, NULL, &gg))
            goto done;
        double local_delta = 0, local_base = 0;
        for (size_t j = 0; j < r * n; j++) {
            double d = cg[j] - gg[j];
            total_delta += d * d;
            total_base += (double)cg[j] * cg[j];
            total_gpu += (double)gg[j] * gg[j];
            total_dot += (double)cg[j] * gg[j];
            local_delta += d * d;
            local_base += (double)cg[j] * cg[j];
        }
        if (c.blocks == 20 && local_base > 1 && sqrt(local_delta / local_base) > .001)
            fprintf(stderr, "gradient %s: relative %.7g norm %.7g\n", name,
                    sqrt(local_delta / local_base), sqrt(local_base));
        /* After independent gradient comparison, use identical gradients to
         * isolate AdamW correctness from BF16 operator roundoff. */
        memcpy(cg, gg, r * n * sizeof(float));
    }
    double grad = sqrt(total_delta / fmax(total_base, 1e-12));
    if (report) {
        double scale = total_dot / fmax(total_gpu, 1e-30);
        double aligned = sqrt(fmax(0, total_base - total_dot * total_dot / fmax(total_gpu, 1e-30)) /
                              fmax(total_base, 1e-30));
        fprintf(stderr, "gradient alignment: optimal_gpu_scale=%.9g residual_relative=%.9g\n",
                scale, aligned);
    }
    if (!isfinite(grad) || !isfinite(a.policy) || !isfinite(b.policy) || !isfinite(a.value) ||
        !isfinite(b.value) || grad > 0.001 ||
        fabs(a.policy - b.policy) + fabs(a.value - b.value) > tol) {
        fprintf(stderr, "GPU backward mismatch %.9g\n", grad);
        if (!report)
            goto done;
        numerical_failure = 1;
    }
    if (check(gn_update(cpu, .001f, .0001f, 1, &a)) ||
        check(gn_update(gpu, .001f, .0001f, 1, &b)) || check(gn_infer(gpu, B, x, q, w)))
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
    if (argc >= 3) {
        if (check(gn_save(gpu, argv[2])))
            goto done;
        gn_model *resumed = gn_load(argv[2], argv[1], 0);
        if (!resumed)
            goto done;
        int status = gn_infer(resumed, B, x, p, v);
        gn_destroy(resumed);
        if (check(status) || memcmp(p, q, B * A * sizeof(float)) ||
            memcmp(v, w, B * 3 * sizeof(float))) {
            fprintf(stderr, "GPU checkpoint inference mismatch\n");
            goto done;
        }
    }
    printf("%s %s batch=%d native forward/backward/update: relative output %.8g gradient %.8g\n",
           numerical_failure ? "UNQUALIFIED" : "PASS", argv[1], B, inference, grad);
    /* Report mode continues diagnostics after a numerical failure, but must
     * still return failure. It cannot make a campaign's strict gate pass. */
    rc = numerical_failure ? 1 : 0;
done:
    free(x);
    free(target);
    free(p);
    free(q);
    free(v);
    free(w);
    free(labels);
    gn_destroy(cpu);
    gn_destroy(gpu);
    return rc;
}
