/*
 * verify_flow_solver — numerics for the shortcut ODE sampler (pure scalar).
 *
 * No pytorch dump needed. Time-schedule and single-step reference values
 * are derived from the same formulas upstream uses
 * (_prepare_t_and_d in shortcut/model.py + linear_approximation_step
 * in flow_matching/solver.py). This is a guard against regressions in the
 * scalar helpers themselves.
 */
#include "sam3d_shortcut_solver.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int eq_f(float a, float b, float tol, const char *label) {
    float d = fabsf(a - b);
    if (d > tol) {
        fprintf(stderr, "[verify_flow_solver] FAIL %s  got=%.8f expected=%.8f diff=%.2e\n",
                label, (double)a, (double)b, (double)d);
        return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    int fail = 0;

    /* --- Schedule, no rescale, not reversed (plain linspace). --- */
    {
        float t[11];
        sam3d_shortcut_make_times(t, 10, 1.0f, 0);
        for (int i = 0; i <= 10; i++) {
            char lbl[32]; snprintf(lbl, sizeof(lbl), "linspace[%d]", i);
            if (!eq_f(t[i], (float)i / 10.0f, 1e-6f, lbl)) fail = 1;
        }
    }

    /* --- Schedule with reversed=1 (denoise direction). --- */
    {
        float t[6];
        sam3d_shortcut_make_times(t, 5, 1.0f, 1);
        const float expected[6] = {1.0f, 0.8f, 0.6f, 0.4f, 0.2f, 0.0f};
        for (int i = 0; i <= 5; i++) {
            char lbl[32]; snprintf(lbl, sizeof(lbl), "reversed[%d]", i);
            if (!eq_f(t[i], expected[i], 1e-6f, lbl)) fail = 1;
        }
    }

    /* --- Schedule with rescale_t=3.0, reversed=0.
     *     upstream formula: t = i/N; t' = t / (1 + (rescale - 1) * (1 - t))
     *     For N=5, rescale=3:
     *       i=0  t=0.0   → 0.0
     *       i=1  t=0.2   → 0.2/2.6 ≈ 0.07692308
     *       i=2  t=0.4   → 0.4/2.2 ≈ 0.18181818
     *       i=3  t=0.6   → 0.6/1.8 ≈ 0.33333333
     *       i=4  t=0.8   → 0.8/1.4 ≈ 0.57142857
     *       i=5  t=1.0   → 1.0
     * --- */
    {
        float t[6];
        sam3d_shortcut_make_times(t, 5, 3.0f, 0);
        const float expected[6] = {
            0.0f, 0.07692308f, 0.18181818f, 0.33333333f, 0.57142857f, 1.0f
        };
        for (int i = 0; i <= 5; i++) {
            char lbl[32]; snprintf(lbl, sizeof(lbl), "rescale3[%d]", i);
            if (!eq_f(t[i], expected[i], 1e-6f, lbl)) fail = 1;
        }
    }

    /* --- Shortcut d. --- */
    if (!eq_f(sam3d_shortcut_d(25, 0), 1.0f / 25.0f, 1e-7f, "d(25)")) fail = 1;
    if (!eq_f(sam3d_shortcut_d(25, 1), 0.0f,          1e-7f, "d(25,noshortcut)")) fail = 1;

    /* --- Euler step in place. --- */
    {
        float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        float v[4] = {0.5f, -1.0f, 2.0f, 0.0f};
        sam3d_shortcut_euler_step(x, v, -0.1f, 4);
        const float expected[4] = {0.95f, 2.1f, 2.8f, 4.0f};
        for (int i = 0; i < 4; i++) {
            char lbl[32]; snprintf(lbl, sizeof(lbl), "euler[%d]", i);
            if (!eq_f(x[i], expected[i], 1e-6f, lbl)) fail = 1;
        }
    }

    /* --- CFG combine. --- */
    {
        float y_c[3] = {1.0f, 2.0f, 3.0f};
        float y_u[3] = {0.5f, 1.0f, 2.0f};
        float out[3];

        sam3d_shortcut_cfg_combine(out, y_c, y_u, 0.0f, 3);
        if (!eq_f(out[0], 1.0f, 1e-7f, "cfg0[0]")) fail = 1;
        if (!eq_f(out[1], 2.0f, 1e-7f, "cfg0[1]")) fail = 1;
        if (!eq_f(out[2], 3.0f, 1e-7f, "cfg0[2]")) fail = 1;

        sam3d_shortcut_cfg_combine(out, y_c, y_u, 2.0f, 3);
        if (!eq_f(out[0], 3.0f * 1.0f - 2.0f * 0.5f, 1e-6f, "cfg2[0]")) fail = 1;
        if (!eq_f(out[1], 3.0f * 2.0f - 2.0f * 1.0f, 1e-6f, "cfg2[1]")) fail = 1;
        if (!eq_f(out[2], 3.0f * 3.0f - 2.0f * 2.0f, 1e-6f, "cfg2[2]")) fail = 1;
    }

    /* --- CFG interval gating. --- */
    if (sam3d_shortcut_cfg_active(0.5f, 0.0f, 0.0f) != 1) {
        fprintf(stderr, "[verify_flow_solver] FAIL cfg_active disabled-interval\n"); fail = 1;
    }
    if (sam3d_shortcut_cfg_active(0.5f, 0.3f, 0.8f) != 1) {
        fprintf(stderr, "[verify_flow_solver] FAIL cfg_active inside\n"); fail = 1;
    }
    if (sam3d_shortcut_cfg_active(0.9f, 0.3f, 0.8f) != 0) {
        fprintf(stderr, "[verify_flow_solver] FAIL cfg_active above\n"); fail = 1;
    }
    if (sam3d_shortcut_cfg_active(0.1f, 0.3f, 0.8f) != 0) {
        fprintf(stderr, "[verify_flow_solver] FAIL cfg_active below\n"); fail = 1;
    }

    if (fail) {
        fprintf(stderr, "[verify_flow_solver] FAIL\n");
        return 1;
    }
    fprintf(stderr, "[verify_flow_solver] OK\n");
    return 0;
}
