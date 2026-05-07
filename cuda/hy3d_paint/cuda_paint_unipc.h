/*
 * cuda_paint_unipc.h - UniPCMultistepScheduler hot-path port for the
 * Hunyuan3D-2.1 paint pipeline.
 *
 * Locked-in config (matches scheduler_config.json + pipeline override
 * `timestep_spacing="trailing"` + UniPC defaults — see
 * ref/hy3d/dump_paint_unipc.py for the validation oracle):
 *   beta_schedule        = scaled_linear (0.00085 -> 0.012, 1000 steps)
 *   prediction_type      = v_prediction
 *   timestep_spacing     = trailing
 *   rescale_betas_zero_snr = true   (alphas_cumprod[-1] -> 2^-24)
 *   solver_order         = 2
 *   solver_type          = bh2
 *   predict_x0           = true
 *   lower_order_final    = true
 *   final_sigmas_type    = zero
 *
 * Only the v_prediction + predict_x0 + bh2 + order<=2 branch is
 * implemented (it's all hy3d_paint exercises). Other paths intentionally
 * trip a fprintf+abort so silent regressions are loud.
 *
 * Sample tensor lives in host memory and is updated in-place. UNet
 * model output is also expected in host memory each step. The caller is
 * responsible for the DtoH/HtoD bounce around the UNet forward pass.
 *
 * Allocation: the scheduler keeps a circular buffer of `solver_order`
 * past converted model outputs (each = `numel` floats), `last_sample`
 * (numel floats), and the precomputed timesteps[N]/sigmas[N+1] tables.
 *
 * No external deps beyond <math.h>/<stdlib.h>/<string.h>.
 */
#ifndef CUDA_PAINT_UNIPC_H
#define CUDA_PAINT_UNIPC_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PUNIPC_MAX_ORDER 4   /* defensive cap; actual order=2 */

typedef struct {
    int num_train;          /* 1000 */
    int num_inference;      /* 15 in pipeline */
    int solver_order;       /* 2 */
    long long *timesteps;   /* [num_inference]    int64 (matches diffusers) */
    float     *sigmas;      /* [num_inference+1]  trailing 0 */

    /* per-step state */
    int   step_index;
    int   lower_order_nums;
    int   this_order;
    /* model_outputs[k] is a [numel] f32 buffer. We keep `solver_order` of
     * them in a small static array (PUNIPC_MAX_ORDER = upper bound). The
     * "newest" lives at index solver_order-1 (matches diffusers index
     * `model_outputs[-1]`). Slot is NULL until populated. */
    float *model_outputs[PUNIPC_MAX_ORDER];
    long long timestep_list[PUNIPC_MAX_ORDER];

    float *last_sample;     /* [numel] or NULL until step>=1 */
    int    has_last_sample; /* 0/1 */

    size_t numel;
} pu_unipc;

/* ---------- private helpers ----------------------------------------- */

/* scaled_linear betas + cumprod, with rescale_zero_terminal_snr applied. */
static void pu_unipc_build_alphas_cumprod(float *out, int N) {
    /* betas_sqrt = linspace(sqrt(0.00085), sqrt(0.012), N); betas = sq */
    const float bs = sqrtf(0.00085f), be = sqrtf(0.012f);
    /* compute alphas_cumprod, then rescale_zero_terminal_snr */
    /* Step 1: alphas_cumprod */
    double cp = 1.0;
    for (int i = 0; i < N; i++) {
        float t = (N == 1) ? bs : (bs + (be - bs) * (float)i / (float)(N - 1));
        float beta = t * t;
        float alpha = 1.f - beta;
        cp *= (double)alpha;
        out[i] = (float)cp;
    }
    /* Step 2: rescale_zero_terminal_snr (mirrors diffusers helper).
     *   alphas_bar_sqrt = sqrt(alphas_cumprod)
     *   s0 = alphas_bar_sqrt[0]; sT = alphas_bar_sqrt[-1]
     *   alphas_bar_sqrt -= sT
     *   alphas_bar_sqrt *= s0 / (s0 - sT)
     *   alphas_cumprod = alphas_bar_sqrt^2
     *   then alphas_cumprod[-1] = 2^-24 (avoid inf log).
     */
    float s0 = sqrtf(out[0]);
    float sT = sqrtf(out[N - 1]);
    float scale = s0 / (s0 - sT);
    for (int i = 0; i < N; i++) {
        float v = sqrtf(out[i]) - sT;
        v *= scale;
        out[i] = v * v;
    }
    out[N - 1] = ldexpf(1.f, -24);  /* 2^-24 */
}

static void pu_unipc_sigma_to_alpha_sigma(float sigma, float *alpha_t, float *sigma_t) {
    /* VP-type: alpha_t = 1/sqrt(1+sigma^2), sigma_t = sigma * alpha_t */
    float a = 1.f / sqrtf(1.f + sigma * sigma);
    *alpha_t = a;
    *sigma_t = sigma * a;
}

/* Linear interp on the alphas_cumprod-derived sigma table at integer-rounded
 * timesteps — mirrors np.interp(timesteps, arange(N), sigmas). */
static float pu_unipc_interp_sigma(const float *sigmas_full, int N, double tq) {
    if (tq <= 0) return sigmas_full[0];
    if (tq >= N - 1) return sigmas_full[N - 1];
    int i0 = (int)tq;
    int i1 = i0 + 1;
    double f = tq - (double)i0;
    return (float)((1.0 - f) * sigmas_full[i0] + f * sigmas_full[i1]);
}

/* ---------- public API ---------------------------------------------- */

static void pu_unipc_init(pu_unipc *s, int num_inference, size_t numel) {
    memset(s, 0, sizeof(*s));
    s->num_train     = 1000;
    s->num_inference = num_inference;
    s->solver_order  = 2;
    s->numel         = numel;

    s->timesteps = (long long*)calloc((size_t)num_inference, sizeof(long long));
    s->sigmas    = (float*)    calloc((size_t)num_inference + 1, sizeof(float));

    /* Build trailing timesteps:
     *   step_ratio = num_train / num_inference
     *   t = arange(num_train, 0, -step_ratio).round() - 1
     */
    double step_ratio = (double)s->num_train / (double)num_inference;
    for (int i = 0; i < num_inference; i++) {
        double tv = (double)s->num_train - step_ratio * (double)i;  /* descending */
        long long ti = (long long)llround(tv) - 1;
        s->timesteps[i] = ti;
    }

    /* Build the full sigmas table on alphas_cumprod, then linear interp at
     * timesteps. Final entry = 0 (final_sigmas_type=zero). */
    float *acp = (float*)malloc(sizeof(float) * (size_t)s->num_train);
    pu_unipc_build_alphas_cumprod(acp, s->num_train);
    float *sig_full = (float*)malloc(sizeof(float) * (size_t)s->num_train);
    for (int i = 0; i < s->num_train; i++) {
        sig_full[i] = sqrtf((1.f - acp[i]) / acp[i]);
    }
    for (int i = 0; i < num_inference; i++) {
        s->sigmas[i] = pu_unipc_interp_sigma(sig_full, s->num_train,
                                             (double)s->timesteps[i]);
    }
    s->sigmas[num_inference] = 0.f;
    free(acp); free(sig_full);

    for (int k = 0; k < PUNIPC_MAX_ORDER; k++) s->model_outputs[k] = NULL;
    s->step_index       = 0;
    s->lower_order_nums = 0;
    s->this_order       = 0;
    s->has_last_sample  = 0;
    s->last_sample      = (float*)calloc(numel, sizeof(float));
    /* preallocate solver_order model_outputs slots */
    for (int k = 0; k < s->solver_order; k++) {
        s->model_outputs[k] = (float*)calloc(numel, sizeof(float));
    }
}

static void pu_unipc_free(pu_unipc *s) {
    free(s->timesteps);
    free(s->sigmas);
    for (int k = 0; k < PUNIPC_MAX_ORDER; k++) free(s->model_outputs[k]);
    free(s->last_sample);
    memset(s, 0, sizeof(*s));
}

/* convert v_prediction model output to x0_pred (predict_x0=true).
 *   x0 = alpha_t * sample - sigma_t * model_out
 */
static void pu_unipc_convert_v(const float *m, const float *x, float *out,
                               size_t n, float alpha_t, float sigma_t) {
    for (size_t i = 0; i < n; i++)
        out[i] = alpha_t * x[i] - sigma_t * m[i];
}

/* UniP/UniC update for order in {1, 2} with predict_x0=True, bh2.
 * `is_corrector`: 0=predictor (uses sigmas[step+1] vs sigmas[step]),
 *                  1=corrector (uses sigmas[step] vs sigmas[step-1]).
 * For order==2 the `D1s` term uses `(m_prev - m0) / rk` and rhos = [0.5]
 * (predictor); corrector solves a 2x2 system but for the bh2 + order==2
 * predict_x0 branch the closed form is rhos_c = [0.5, 0.5] /
 * actually solved from R*rhos=b (see below).
 *
 * Inputs:
 *   m0   = newest converted model output (= model_outputs[-1])
 *   m1   = second-newest (only used if order>=2; pass NULL if order==1)
 *   sigma_a = "from" sigma (predictor: sigmas[step]; corrector: sigmas[step-1])
 *   sigma_b = "to"   sigma (predictor: sigmas[step+1]; corrector: sigmas[step])
 *   sigma_p = order>=2: previous sigma to compute rk (predictor:
 *              sigmas[step-1]; corrector: sigmas[step-2]); ignored if order==1.
 *   model_t = current step's converted output (only corrector uses it).
 */
static void pu_unipc_update_predict_x0_bh2(
    const float *x_in, float *x_out, size_t n,
    const float *m0, const float *m1, const float *model_t,
    float sigma_a, float sigma_b, float sigma_p,
    int order, int is_corrector) {

    float alpha_a, sa, alpha_b, sb;
    pu_unipc_sigma_to_alpha_sigma(sigma_a, &alpha_a, &sa);
    pu_unipc_sigma_to_alpha_sigma(sigma_b, &alpha_b, &sb);
    float lam_a = logf(alpha_a) - logf(sa);
    float lam_b = logf(alpha_b) - logf(sb);
    float h     = lam_b - lam_a;
    float hh    = -h;                  /* predict_x0 -> hh = -h */
    float h_phi_1 = expm1f(hh);        /* e^hh - 1 */
    float B_h     = expm1f(hh);        /* bh2 */

    /* x_t_ = (sb / sa) * x - alpha_b * h_phi_1 * m0
     *      = ratio * x + coefM0 * m0
     */
    float ratio  = sb / sa;
    float coefM0 = -alpha_b * h_phi_1;

    /* D1 contribution. order==1 -> none. order==2 -> single D1.
     *   predictor (rhos_p, order==2): rhos_p = [0.5]
     *     pred_res = 0.5 * D1, where D1 = (m_prev - m0) / rk
     *   corrector (rhos_c, order==2): solve R*rhos = b (R=[[1,1],[1,rk]],
     *     b=[h_phi_1/B_h, (h_phi_1/hh - 1) * 2 / B_h]).
     *     corr_res = rhos_c[0] * D1; D1_t = model_t - m0; final adds
     *     rhos_c[1] * D1_t.
     */
    float coefD1   = 0.f;       /* multiplies (m1 - m0) (already /rk) */
    float coefD1t  = 0.f;       /* multiplies (model_t - m0) (corrector only) */

    if (order >= 2) {
        float alpha_p, sp;
        pu_unipc_sigma_to_alpha_sigma(sigma_p, &alpha_p, &sp);
        float lam_p = logf(alpha_p) - logf(sp);
        float rk    = (lam_p - lam_a) / h;

        if (!is_corrector) {
            /* predictor, order==2: rhos_p = [0.5]
             * pred_res = 0.5 * (m1 - m0) / rk
             * x_t = x_t_ - alpha_b * B_h * pred_res */
            coefD1 = -alpha_b * B_h * 0.5f / rk;
        } else {
            /* corrector, order==2:
             *   rks = [rk, 1.0];  R = stack([rks^0, rks^1]) = [[1,1],[rk,1]]
             *   h_phi_2 = h_phi_1/hh - 1
             *   b[0] = h_phi_2 / B_h                     (factorial_i=1, k=1)
             *   b[1] = 2 * (h_phi_2/hh - 1/2) / B_h      (factorial_i=2, k=2)
             *        = (2*h_phi_2/hh - 1) / B_h
             *   solve [[1,1],[rk,1]] * rho = b:
             *     rho[0] = (b1 - b0) / (rk - 1)         (coef on D1 column)
             *     rho[1] = (b0*rk - b1) / (rk - 1)      (coef on D1_t)
             *   D1 column already pre-divided by rk in einsum factorization.
             */
            float h_phi_2 = h_phi_1 / hh - 1.f;
            float b0  = h_phi_2 / B_h;
            float bk  = (2.f * h_phi_2 / hh - 1.f) / B_h;
            float det = rk - 1.f;
            float rho0 = (bk - b0) / det;            /* multiplies (m1 - m0)/rk */
            float rho1 = (rk * b0 - bk) / det;       /* multiplies (model_t - m0) */
            coefD1  = -alpha_b * B_h * rho0 / rk;
            coefD1t = -alpha_b * B_h * rho1;
        }
    } else {
        /* order==1: no D1 term in predictor. Corrector at order==1
         * uses rhos_c = [0.5] -> coefD1t = -alpha_b * B_h * 0.5. */
        if (is_corrector) {
            coefD1t = -alpha_b * B_h * 0.5f;
        }
    }

    if (order >= 2 && m1) {
        if (is_corrector && model_t) {
            for (size_t i = 0; i < n; i++) {
                x_out[i] = ratio * x_in[i]
                         + coefM0  * m0[i]
                         + coefD1  * (m1[i]      - m0[i])
                         + coefD1t * (model_t[i] - m0[i]);
            }
        } else {
            for (size_t i = 0; i < n; i++) {
                x_out[i] = ratio * x_in[i]
                         + coefM0 * m0[i]
                         + coefD1 * (m1[i] - m0[i]);
            }
        }
    } else {
        if (is_corrector && model_t) {
            for (size_t i = 0; i < n; i++) {
                x_out[i] = ratio * x_in[i]
                         + coefM0  * m0[i]
                         + coefD1t * (model_t[i] - m0[i]);
            }
        } else {
            for (size_t i = 0; i < n; i++) {
                x_out[i] = ratio * x_in[i] + coefM0 * m0[i];
            }
        }
    }
}

/* One denoising step. `x` is in/out [numel] f32 host buffer; `model_out` is
 * the raw v_prediction UNet output for the current timestep. */
static void pu_unipc_step(pu_unipc *s, const float *model_out, float *x) {
    if (s->step_index >= s->num_inference) {
        fprintf(stderr, "pu_unipc_step: step_index %d past num_inference %d\n",
                s->step_index, s->num_inference);
        abort();
    }
    size_t n = s->numel;
    float sigma_now = s->sigmas[s->step_index];
    float alpha_t, sigma_t;
    pu_unipc_sigma_to_alpha_sigma(sigma_now, &alpha_t, &sigma_t);

    /* convert v -> x0_pred (predict_x0 path) */
    float *converted = (float*)malloc(sizeof(float) * n);
    pu_unipc_convert_v(model_out, x, converted, n, alpha_t, sigma_t);

    int use_corrector = (s->step_index > 0 && s->has_last_sample);

    if (use_corrector) {
        /* this_order set on the *previous* step iteration, since corrector
         * uses last step's order. last_sample = x BEFORE predictor at step
         * (step_index-1), this_sample = current x. */
        const float *m0 = s->model_outputs[s->solver_order - 1];     /* newest after rotate from last step */
        const float *m1 = (s->this_order >= 2) ? s->model_outputs[s->solver_order - 2] : NULL;
        float sigma_a = s->sigmas[s->step_index - 1];
        float sigma_b = s->sigmas[s->step_index];
        float sigma_p = (s->this_order >= 2 && s->step_index >= 2)
                        ? s->sigmas[s->step_index - 2] : 0.f;

        float *corrected = (float*)malloc(sizeof(float) * n);
        pu_unipc_update_predict_x0_bh2(s->last_sample, corrected, n,
                                       m0, m1, converted,
                                       sigma_a, sigma_b, sigma_p,
                                       s->this_order, /*is_corrector=*/1);
        memcpy(x, corrected, sizeof(float) * n);
        free(corrected);
    }

    /* rotate model_outputs left by 1, store newest at slot solver_order-1 */
    float *oldest = s->model_outputs[0];
    for (int k = 0; k < s->solver_order - 1; k++) {
        s->model_outputs[k] = s->model_outputs[k + 1];
    }
    s->model_outputs[s->solver_order - 1] = oldest;
    memcpy(s->model_outputs[s->solver_order - 1], converted, sizeof(float) * n);
    /* also rotate timestep_list */
    for (int k = 0; k < s->solver_order - 1; k++) {
        s->timestep_list[k] = s->timestep_list[k + 1];
    }
    s->timestep_list[s->solver_order - 1] = s->timesteps[s->step_index];

    /* compute this_order for predictor — lower_order_final */
    int this_order = s->solver_order;
    int remaining  = s->num_inference - s->step_index;  /* matches `len(timesteps) - step_index` */
    if (this_order > remaining) this_order = remaining;
    if (this_order > s->lower_order_nums + 1) this_order = s->lower_order_nums + 1;
    s->this_order = this_order;

    /* save x as last_sample (predictor input) */
    memcpy(s->last_sample, x, sizeof(float) * n);
    s->has_last_sample = 1;

    /* predictor */
    const float *m0 = s->model_outputs[s->solver_order - 1];
    const float *m1 = (this_order >= 2) ? s->model_outputs[s->solver_order - 2] : NULL;
    float sigma_a = s->sigmas[s->step_index];
    float sigma_b = s->sigmas[s->step_index + 1];
    float sigma_p = (this_order >= 2 && s->step_index >= 1)
                    ? s->sigmas[s->step_index - 1] : 0.f;

    float *predicted = (float*)malloc(sizeof(float) * n);
    pu_unipc_update_predict_x0_bh2(x, predicted, n,
                                   m0, m1, NULL,
                                   sigma_a, sigma_b, sigma_p,
                                   this_order, /*is_corrector=*/0);
    memcpy(x, predicted, sizeof(float) * n);
    free(predicted);
    free(converted);

    if (s->lower_order_nums < s->solver_order) s->lower_order_nums++;
    s->step_index++;
}

#endif /* CUDA_PAINT_UNIPC_H */
