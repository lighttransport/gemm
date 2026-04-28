/*
 * sam3d_shortcut_solver.h — flow-matching shortcut ODE sampler (pure scalar).
 *
 * This header provides only the *sampler math* for SAM-3D-Objects' shortcut
 * ODE sampler. The model forward is not linked here — callers pass a velocity
 * callback.
 *
 * Upstream (facebookresearch/sam-3d-objects):
 *   model/backbone/generator/shortcut/model.py::ShortCut.generate_iter
 *     _prepare_t_and_d     → time schedule + uniform d
 *     _generate_dynamics   → multiplies t,d by `time_scale` before calling model
 *   model/backbone/generator/flow_matching/solver.py::linear_approximation_step
 *     x_{i+1} = x_i + v * dt
 *   model/backbone/generator/classifier_free_guidance.py
 *     y = (1 + w) * y_cond - w * y_uncond
 *
 * Three concerns, three helpers:
 *   1. Time schedule       — sam3d_shortcut_make_times
 *   2. Euler update        — sam3d_shortcut_euler_step  (x += v * dt)
 *   3. CFG combine         — sam3d_shortcut_cfg_combine (mix cond/uncond)
 *
 * Callers build their own outer loop; this keeps the solver model-agnostic.
 */
#ifndef SAM3D_SHORTCUT_SOLVER_H
#define SAM3D_SHORTCUT_SOLVER_H

#include <math.h>
#include <stddef.h>

/* Fill `times[0..steps]` with the shortcut time schedule.
 *   base:              linspace(0, 1, steps+1)
 *   rescale_t (>0):    t → t / (1 + (rescale_t - 1) * (1 - t))
 *                      rescale_t == 1.0f disables rescale (identity mapping).
 *                      Upstream default for SS stage: 3.0.
 *   reversed:          after rescale, reverse direction t → 1 - t
 *                      Upstream SS inference sets this true (denoise 1 → 0).
 *
 * The caller-visible `t` and `dt` remain in [0,1] — the `time_scale` knob is
 * applied inside the velocity callback (see _generate_dynamics upstream). */
static void sam3d_shortcut_make_times(float *times, int steps,
                                      float rescale_t, int reversed) {
    for (int i = 0; i <= steps; i++) {
        float t = (float)i / (float)steps;
        if (rescale_t > 0.0f && rescale_t != 1.0f)
            t = t / (1.0f + (rescale_t - 1.0f) * (1.0f - t));
        if (reversed) t = 1.0f - t;
        times[i] = t;
    }
}

/* Uniform shortcut jump size: d = 1 / steps (with time_scale applied by the
 * velocity callback). Upstream no_shortcut=True forces d=0 (pure flow match). */
static float sam3d_shortcut_d(int steps, int no_shortcut) {
    return no_shortcut ? 0.0f : (1.0f / (float)steps);
}

/* One Euler step: x[i] += v[i] * dt (in place). */
static void sam3d_shortcut_euler_step(float *x, const float *v, float dt, int n) {
    for (int i = 0; i < n; i++) x[i] += v[i] * dt;
}

/* CFG combine: out[i] = (1 + strength) * y_cond[i] - strength * y_uncond[i].
 * strength == 0 degenerates to out = y_cond. Safe if out aliases y_cond. */
static inline void sam3d_shortcut_cfg_combine(float *out,
                                       const float *y_cond, const float *y_uncond,
                                       float strength, int n) {
    if (strength == 0.0f) {
        if (out != y_cond)
            for (int i = 0; i < n; i++) out[i] = y_cond[i];
        return;
    }
    float a = 1.0f + strength;
    for (int i = 0; i < n; i++)
        out[i] = a * y_cond[i] - strength * y_uncond[i];
}

/* Decide whether CFG is active at this timestep. Upstream uses an interval
 * [lo, hi]; strength=0 outside. Pass lo=hi=0 to disable gating (always on).
 * `t` is the pre-time_scale value in [0,1]. */
static inline int sam3d_shortcut_cfg_active(float t, float lo, float hi) {
    if (lo == 0.0f && hi == 0.0f) return 1;
    return (t >= lo) && (t <= hi);
}

#endif  /* SAM3D_SHORTCUT_SOLVER_H */
