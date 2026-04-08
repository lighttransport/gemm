/*
 * qwen_image_scheduler.h - FlowMatchEulerDiscreteScheduler for Qwen-Image
 *
 * Usage:
 *   #define QIMG_SCHEDULER_IMPLEMENTATION
 *   #include "qwen_image_scheduler.h"
 *
 * API:
 *   void qimg_sched_set_timesteps(qimg_scheduler *s, int n_steps, int img_seq_len);
 *   void qimg_sched_step(float *x, const float *v, int n, int step, qimg_scheduler *s);
 */
#ifndef QIMG_SCHEDULER_H
#define QIMG_SCHEDULER_H

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QIMG_SCHED_MAX_STEPS 200

typedef struct {
    int    n_steps;
    float  sigmas[QIMG_SCHED_MAX_STEPS + 1];   /* n_steps+1 values */
    float  timesteps[QIMG_SCHED_MAX_STEPS];     /* n_steps values */
    float  dt[QIMG_SCHED_MAX_STEPS];            /* dt = sigma[i+1] - sigma[i] */

    /* Config */
    int    num_train_timesteps;   /* 1000 */
    float  base_shift;            /* 0.5 */
    float  max_shift;             /* 0.9 */
    float  shift_terminal;        /* 0.02 */
    int    base_image_seq_len;    /* 256 */
    int    max_image_seq_len;     /* 8192 */
} qimg_scheduler;

void qimg_sched_init(qimg_scheduler *s);
void qimg_sched_set_timesteps(qimg_scheduler *s, int n_steps, int img_seq_len);
/* ComfyUI-compatible: fixed shift, multiplier=1.0 (timestep = sigma, not sigma×1000) */
void qimg_sched_set_timesteps_comfyui(qimg_scheduler *s, int n_steps, float shift, float multiplier);
void qimg_sched_step(float *x, const float *v, int n, int step,
                     const qimg_scheduler *s);

#ifdef __cplusplus
}
#endif

/* ---- Implementation ---- */

#ifdef QIMG_SCHEDULER_IMPLEMENTATION

#include <string.h>

void qimg_sched_init(qimg_scheduler *s) {
    memset(s, 0, sizeof(*s));
    s->num_train_timesteps = 1000;
    s->base_shift     = 0.5f;
    s->max_shift      = 0.9f;
    s->shift_terminal = 0.02f;
    s->base_image_seq_len = 256;
    s->max_image_seq_len  = 8192;
}

void qimg_sched_set_timesteps(qimg_scheduler *s, int n_steps, int img_seq_len) {
    if (n_steps > QIMG_SCHED_MAX_STEPS)
        n_steps = QIMG_SCHED_MAX_STEPS;
    s->n_steps = n_steps;

    /* Compute dynamic shift based on image sequence length */
    float log_seq   = logf((float)img_seq_len);
    float log_base  = logf((float)s->base_image_seq_len);
    float log_max   = logf((float)s->max_image_seq_len);
    float mu = (log_seq - log_base) / (log_max - log_base);
    if (mu < 0.0f) mu = 0.0f;
    if (mu > 1.0f) mu = 1.0f;
    float shift = s->base_shift + (s->max_shift - s->base_shift) * mu;

    /* Generate linearly spaced sigmas in [1, shift_terminal] */
    for (int i = 0; i <= n_steps; i++) {
        float t = (float)i / (float)n_steps;
        float sigma = 1.0f - t * (1.0f - s->shift_terminal);

        /* Apply exponential time shift */
        float es = expf(shift);
        sigma = es * sigma / (1.0f + (es - 1.0f) * sigma);

        s->sigmas[i] = sigma;
    }

    /* Timesteps = sigmas * num_train_timesteps */
    for (int i = 0; i < n_steps; i++) {
        s->timesteps[i] = s->sigmas[i] * (float)s->num_train_timesteps;
        s->dt[i] = s->sigmas[i + 1] - s->sigmas[i];
    }
}

/* ComfyUI-compatible scheduler: matches "simple" scheduler with AuraFlow shift.
 *
 * ComfyUI's approach:
 * 1. Pre-compute 1000 shifted sigmas: sigma(i/1000) for i=1..1000
 *    where sigma(t) is the AuraFlow shift function
 * 2. "simple" scheduler picks n_steps evenly-spaced samples from this array
 *
 * This produces DIFFERENT results from directly computing sigma at n_steps
 * equally-spaced points, because the shift function is nonlinear. */
void qimg_sched_set_timesteps_comfyui(qimg_scheduler *s, int n_steps, float shift, float multiplier) {
    if (n_steps > QIMG_SCHED_MAX_STEPS) n_steps = QIMG_SCHED_MAX_STEPS;
    s->n_steps = n_steps;

    /* Step 1: Pre-compute 1000 shifted sigmas (matching ModelSamplingAdvanced) */
    int n_table = 1000;
    float table[1001];  /* 1-indexed: table[1..1000] */
    /* time_snr_shift(alpha, t) = alpha * t / (1 + (alpha-1) * t)
     * NOTE: shift parameter IS alpha directly (NOT exp(shift)) */
    float alpha = shift;
    for (int i = 1; i <= n_table; i++) {
        float t = (float)i / (float)n_table;  /* t in (0, 1] */
        table[i] = alpha * t / (1.0f + (alpha - 1.0f) * t);
    }

    /* Step 2: "simple" scheduler — pick evenly-spaced from reversed table */
    float ss = (float)n_table / (float)n_steps;
    for (int x = 0; x < n_steps; x++) {
        int idx = n_table - (int)(x * ss);  /* index from end */
        if (idx < 1) idx = 1;
        if (idx > n_table) idx = n_table;
        s->sigmas[x] = table[idx];
    }
    s->sigmas[n_steps] = 0.0f;

    for (int i = 0; i < n_steps; i++) {
        s->timesteps[i] = s->sigmas[i] * multiplier;
        s->dt[i] = s->sigmas[i + 1] - s->sigmas[i];
    }
}

/*
 * Euler step: x = x + dt * v
 * (Flow matching convention: v predicts velocity from noisy to clean)
 */
void qimg_sched_step(float *x, const float *v, int n, int step,
                     const qimg_scheduler *s) {
    float dt = s->dt[step];
    for (int i = 0; i < n; i++)
        x[i] += dt * v[i];
}

#endif /* QIMG_SCHEDULER_IMPLEMENTATION */
#endif /* QIMG_SCHEDULER_H */
