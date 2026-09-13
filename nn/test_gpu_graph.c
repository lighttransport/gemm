/* SPDX-License-Identifier: MIT
 * Hardware regression: requires an SDK-enabled build and RX 9070 XT. */
#include "gn_internal.h"
#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void ok(int rc) {
    if (rc) {
        fprintf(stderr, "%s\n", gn_error());
        exit(1);
    }
}
static double gradients(gn_model *cpu, gn_model *gpu) {
    double delta = 0, base = 0;
    for (size_t i = 0; i < gn_tensor_count(cpu); i++) {
        size_t r, c;
        float *a, *b;
        assert(gn_tensor(cpu, i, &r, &c, NULL, &a));
        assert(gn_tensor(gpu, i, NULL, NULL, NULL, &b));
        if (!a)
            continue;
        for (size_t j = 0; j < r * c; j++) {
            assert(isfinite(a[j]) && isfinite(b[j]));
            delta += (double)(a[j] - b[j]) * (a[j] - b[j]);
            base += (double)a[j] * a[j];
        }
    }
    return sqrt(delta / fmax(base, 1e-12));
}
int main(void) {
    gn_config c = gn_default_config();
    c.version = 2;
    c.side = 9;
    c.inputs = 5;
    c.actions = 7;
    c.channels = 64;
    c.blocks = 2;
    c.attention_every = 2;
    c.head_dim = 32;
    c.value_channels = 3;
    c.value_hidden = 8;
    gn_model *gpu = gn_create(&c, "hip-bf16x3-fp16back-blaslt-fast", 0);
    if (!gpu) {
        fprintf(stderr, "UNAVAILABLE: %s\n", gn_error());
        return 77;
    }
    gn_model *cpu = gn_create(&c, "cpu", 0);
    assert(cpu);
    enum { MAX_BATCH = 8, X = 81 * 5, A = 81 * 7 };
    float x[MAX_BATCH * X], target[MAX_BATCH * A], p[MAX_BATCH * A], v[MAX_BATCH * 3];
    uint32_t label[MAX_BATCH];
    uint64_t captures = 0, launches = 0;
    const int batches[] = {2, 2, 1, 3};
    double worst = 0;
    for (int phase = 0; phase < 4; phase++) {
        uint64_t old_captures = captures, old_launches = launches;
        for (int i = 0; i < MAX_BATCH * X; i++)
            x[i] = sinf((float)i * .19f + phase * .1f);
        if (phase == 1) {
            /* Grow inference buffers, then return to the SAME training batch:
             * a batch-only cache key would replay freed device pointers. */
            ok(gn_infer(gpu, MAX_BATCH, x, p, v));
            assert(gn_gpu_graph_stats(gpu, &captures, &launches) == 0);
        }
        for (int step = 0; step < 4; step++) {
            int B = batches[phase];
            for (int i = 0; i < MAX_BATCH * A; i++)
                target[i] = -1;
            for (int b = 0; b < MAX_BATCH; b++) {
                label[b] = (b + step) % 3;
                target[b * A + 1 + step] = .3f;
                target[b * A + 31 + step] = .7f;
            }
            gn_zero_grad(cpu);
            gn_zero_grad(gpu);
            gn_metrics a, b;
            ok(gn_backward(cpu, B, x, target, label, &a));
            ok(gn_backward(gpu, B, x, target, label, &b));
            if (step == 0)
                assert(gn_gpu_graph_stats(gpu, &captures, &launches) == 0);
            double error = gradients(cpu, gpu);
            if (error > worst)
                worst = error;
            if (!isfinite(error) || error > .01) {
                fprintf(stderr, "graph gradient phase=%d step=%d relative=%.9g\n",
                        phase, step, error);
                return 1;
            }
            assert(fabsf(a.policy - b.policy) < .01f && fabsf(a.value - b.value) < .01f);
            ok(gn_update(cpu, .0001f, .0001f, 1, &a));
            ok(gn_update(gpu, .0001f, .0001f, 1, &b));
        }
        assert(gn_gpu_graph_stats(gpu, &captures, &launches) == 1);
        assert(captures == old_captures + 1 && launches >= old_launches + 2);
    }
    printf("PASS HIP graph: captures=%llu launches=%llu worst_gradient_relative=%.9g; "
           "updated targets/weights, inference growth, shrinking/growing training batches\n",
           (unsigned long long)captures, (unsigned long long)launches, worst);
    gn_destroy(cpu);
    gn_destroy(gpu);
    return 0;
}
