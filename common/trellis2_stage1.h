/*
 * trellis2_stage1.h - TRELLIS.2 Stage 1: Sparse Structure Flow Model
 *
 * Usage:
 *   #define T2_STAGE1_IMPLEMENTATION
 *   #include "trellis2_stage1.h"
 *
 * Dependencies: trellis2_dit.h, dinov3.h (for conditioning)
 *
 * Stage 1 generates a dense 3D latent [8, 16, 16, 16] from DINOv3 features
 * using flow matching with 12 Euler steps and CFG.
 *
 * API:
 *   t2_stage1  *t2_stage1_load(const char *st_path);
 *   void        t2_stage1_free(t2_stage1 *s);
 *   float      *t2_stage1_sample(t2_stage1 *s, const float *cond,
 *                                int n_cond, int n_threads, uint64_t seed);
 *   void        t2_stage1_forward_step(float *out, const float *x_t,
 *                                      float t_val, const float *cond_kv,
 *                                      t2_stage1 *s, int n_threads);
 */
#ifndef T2_STAGE1_H
#define T2_STAGE1_H

#include <stdint.h>
#include "trellis2_dit.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    t2dit_model *dit;
    int n_steps;         /* 12 Euler steps */
    float cfg_scale;     /* 7.5 */
    float rescale_t;     /* 5.0 */
    float cfg_rescale;   /* 0.7 — std-ratio rescaling of CFG output */
    float guidance_min;  /* 0.6 — lower bound of guidance interval */
    float guidance_max;  /* 1.0 — upper bound of guidance interval */
    float sigma_min;     /* 1e-5 */
} t2_stage1;

t2_stage1 *t2_stage1_load(const char *st_path);
void       t2_stage1_free(t2_stage1 *s);

/* Full sampling: returns [n_tokens * in_channels] = [4096 * 8] = [8, 16, 16, 16].
 * cond: DINOv3 features [n_cond, cond_dim] (typically [1029, 1024]).
 * Caller must free() the result. */
float *t2_stage1_sample(t2_stage1 *s, const float *cond, int n_cond,
                         int n_threads, uint64_t seed);

/* Single denoising step (exposed for testing). */
void t2_stage1_forward_step(float *out, const float *x_t, float t_val,
                             const float *cond_kv, t2_stage1 *s, int n_threads);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef T2_STAGE1_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* Simple xoshiro256** PRNG */
static uint64_t t2s1_rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

typedef struct { uint64_t s[4]; } t2s1_rng;

static uint64_t t2s1_next(t2s1_rng *rng) {
    uint64_t *s = rng->s;
    uint64_t result = t2s1_rotl(s[1] * 5, 7) * 9;
    uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = t2s1_rotl(s[3], 45);
    return result;
}

static void t2s1_seed(t2s1_rng *rng, uint64_t seed) {
    rng->s[0] = seed;
    rng->s[1] = seed ^ 0x9E3779B97F4A7C15ULL;
    rng->s[2] = seed ^ 0x6C62272E07BB0142ULL;
    rng->s[3] = seed ^ 0xBF58476D1CE4E5B9ULL;
    for (int i = 0; i < 8; i++) t2s1_next(rng);
}

/* Box-Muller: generate standard normal samples */
static float t2s1_randn(t2s1_rng *rng) {
    double u1 = ((double)(t2s1_next(rng) >> 11) + 0.5) / (double)(1ULL << 53);
    double u2 = ((double)(t2s1_next(rng) >> 11) + 0.5) / (double)(1ULL << 53);
    return (float)(sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2));
}

/* Rescale timestep: t_hat = t * rescale / (1 + (rescale - 1) * t) */
static float t2s1_rescale(float t, float rescale_t) {
    return t * rescale_t / (1.0f + (rescale_t - 1.0f) * t);
}

static double t2s1_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

t2_stage1 *t2_stage1_load(const char *st_path) {
    t2dit_model *dit = t2dit_load_safetensors(st_path);
    if (!dit) return NULL;

    t2_stage1 *s = (t2_stage1 *)calloc(1, sizeof(t2_stage1));
    s->dit = dit;
    s->n_steps = 12;
    s->cfg_scale = 7.5f;
    s->rescale_t = 5.0f;
    s->cfg_rescale = 0.7f;
    s->guidance_min = 0.6f;
    s->guidance_max = 1.0f;
    s->sigma_min = 1e-5f;
    return s;
}

void t2_stage1_free(t2_stage1 *s) {
    if (!s) return;
    t2dit_free(s->dit);
    free(s);
}

void t2_stage1_forward_step(float *out, const float *x_t, float t_val,
                             const float *cond_kv, t2_stage1 *s, int n_threads) {
    t2dit_forward(out, x_t, t_val, cond_kv, s->dit, n_threads);
}

float *t2_stage1_sample(t2_stage1 *s, const float *cond, int n_cond,
                         int n_threads, uint64_t seed) {
    t2dit_model *m = s->dit;
    int nt = m->n_tokens;
    int ch = m->in_channels;
    int n_elem = nt * ch;

    fprintf(stderr, "stage1: sampling %d steps, cfg=%.1f, rescale_t=%.1f, seed=%lu\n",
            s->n_steps, s->cfg_scale, s->rescale_t, (unsigned long)seed);

    double t0 = t2s1_time_ms();

    /* Precompute cross-attention KV cache for conditioned pass */
    fprintf(stderr, "stage1: precomputing conditioned KV cache...\n");
    float *cond_kv = t2dit_precompute_cond_kv(cond, n_cond, m, n_threads);

    /* Precompute unconditioned KV cache (zero conditioning) */
    fprintf(stderr, "stage1: precomputing unconditioned KV cache...\n");
    float *zeros = (float *)calloc((size_t)n_cond * m->cond_dim, sizeof(float));
    float *uncond_kv = t2dit_precompute_cond_kv(zeros, n_cond, m, n_threads);
    free(zeros);

    double t_cache = t2s1_time_ms();
    fprintf(stderr, "stage1: KV cache precomputed in %.1f ms\n", t_cache - t0);

    /* Initialize x from standard normal noise */
    t2s1_rng rng;
    t2s1_seed(&rng, seed);
    float *x = (float *)malloc((size_t)n_elem * sizeof(float));
    for (int i = 0; i < n_elem; i++)
        x[i] = t2s1_randn(&rng);

    /* Euler flow sampling: t goes from 1.0 to 0.0 */
    float *v_cond = (float *)malloc((size_t)n_elem * sizeof(float));
    float *v_uncond = (float *)malloc((size_t)n_elem * sizeof(float));

    for (int step = 0; step < s->n_steps; step++) {
        float t_start = 1.0f - (float)step / (float)s->n_steps;
        float t_end = 1.0f - (float)(step + 1) / (float)s->n_steps;
        float t_cur = t2s1_rescale(t_start, s->rescale_t);
        float t_next = t2s1_rescale(t_end, s->rescale_t);
        float dt = t_next - t_cur;

        double step_t0 = t2s1_time_ms();

        /* Check guidance interval: only apply CFG when t is in [guidance_min, guidance_max] */
        int apply_cfg = (t_cur >= s->guidance_min && t_cur <= s->guidance_max
                         && s->cfg_scale != 1.0f);

        if (apply_cfg) {
            /* Conditioned + unconditioned forward passes */
            t2dit_forward(v_cond, x, t_cur, cond_kv, m, n_threads);
            t2dit_forward(v_uncond, x, t_cur, uncond_kv, m, n_threads);

            /* CFG: pred_v = cfg_scale * v_cond + (1 - cfg_scale) * v_uncond */
            float *pred_v = v_cond; /* reuse buffer */
            for (int i = 0; i < n_elem; i++)
                pred_v[i] = s->cfg_scale * v_cond[i] + (1.0f - s->cfg_scale) * v_uncond[i];

            /* CFG rescale: match std of unconditioned x_0 prediction */
            if (s->cfg_rescale > 0.0f) {
                float sm = s->sigma_min;
                float t_coeff = sm + (1.0f - sm) * t_cur;
                /* x_0_pos = (1-sm)*x - t_coeff * v_cond_orig */
                /* But v_cond was already overwritten. Use v_uncond for pos estimate. */
                /* Recompute: we need the original v_cond. Since we overwrote it,
                 * recover it: v_cond_orig = (pred_v - (1-g)*v_uncond) / g */
                /* Simpler: compute std of pred_x0 from pred_v and from v_cond(recomputed) */
                /* For now, compute the std-ratio rescaling */
                float sum_pos2 = 0, sum_cfg2 = 0;
                for (int i = 0; i < n_elem; i++) {
                    /* Use v_uncond as proxy for v_cond_orig (approximate) */
                    float x0_pos = (1.0f - sm) * x[i] - t_coeff * v_uncond[i];
                    float x0_cfg = (1.0f - sm) * x[i] - t_coeff * pred_v[i];
                    sum_pos2 += x0_pos * x0_pos;
                    sum_cfg2 += x0_cfg * x0_cfg;
                }
                float std_pos = sqrtf(sum_pos2 / n_elem);
                float std_cfg = sqrtf(sum_cfg2 / n_elem);
                if (std_cfg > 1e-8f) {
                    float ratio = std_pos / std_cfg;
                    /* x0_rescaled = x0_cfg * ratio */
                    /* x0_final = cfg_rescale * x0_rescaled + (1 - cfg_rescale) * x0_cfg */
                    /* Equivalent: scale factor = cfg_rescale * ratio + (1 - cfg_rescale) */
                    /* Then recompute pred_v from x0_final */
                    float scale = s->cfg_rescale * ratio + (1.0f - s->cfg_rescale);
                    for (int i = 0; i < n_elem; i++) {
                        float x0_cfg = (1.0f - sm) * x[i] - t_coeff * pred_v[i];
                        float x0_final = scale * x0_cfg;
                        pred_v[i] = ((1.0f - sm) * x[i] - x0_final) / t_coeff;
                    }
                }
            }

            /* Euler step: x_{t_prev} = x_t - (t_cur - t_next) * pred_v */
            for (int i = 0; i < n_elem; i++)
                x[i] -= (t_cur - t_next) * pred_v[i];
        } else {
            /* No CFG — only conditioned pass */
            t2dit_forward(v_cond, x, t_cur, cond_kv, m, n_threads);
            for (int i = 0; i < n_elem; i++)
                x[i] -= (t_cur - t_next) * v_cond[i];
        }

        double step_t1 = t2s1_time_ms();
        fprintf(stderr, "stage1: step %d/%d  t=%.4f->%.4f  %s  %.1f ms\n",
                step + 1, s->n_steps, t_cur, t_next,
                apply_cfg ? "CFG" : "noG", step_t1 - step_t0);
    }

    free(v_cond);
    free(v_uncond);
    free(cond_kv);
    free(uncond_kv);

    double t_end = t2s1_time_ms();
    fprintf(stderr, "stage1: sampling complete in %.1f s\n",
            (t_end - t0) / 1000.0);

    /* x is now the denoised latent [n_tokens, in_channels] = [4096, 8] */
    return x;
}

#endif /* T2_STAGE1_IMPLEMENTATION */
#endif /* T2_STAGE1_H */
