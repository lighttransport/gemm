/* Hybrid DS4F forward gate: CPU reference versus the S3 persistent FP8 bank. */

#include "../../common/ds4f.h"
#include "hip_ds4f_dense.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_gpu_calls;
static int g_gpu_async_calls;

static int count_gpu_matvec(void *ctx, float *dst,
                            const ds4f_tensor *t, const float *x) {
    g_gpu_calls++;
    return hip_ds4f_dense_matvec_tensor(ctx, dst, t, x);
}

static int count_gpu_async(void *ctx, float *const *dst,
                           const ds4f_tensor *const *t,
                           const float *const *x, int n) {
    g_gpu_async_calls++;
    return hip_ds4f_dense_matvec_tensors_async(ctx, dst, t, x, n);
}

static int count_gpu_wait(void *ctx) {
    return hip_ds4f_dense_wait_tensors(ctx);
}

static float compare_vec(const float *a, const float *b, int n,
                         float *max_abs, float *scale) {
    float absmax = 0.0f, refmax = 0.0f, rel = 0.0f;
    for (int i = 0; i < n; ++i) {
        float e = fabsf(a[i] - b[i]);
        if (e > absmax) absmax = e;
        if (fabsf(a[i]) > refmax) refmax = fabsf(a[i]);
        float r = e / fmaxf(1.0f, fabsf(a[i]));
        if (r > rel) rel = r;
    }
    *max_abs = absmax;
    *scale = fmaxf(1.0f, refmax);
    return rel;
}

static int add_tensor(hip_ds4f_dense *hip, ds4f_tensor *t, const char *name) {
    if (t->type != DS4F_FP8 || !t->w || !t->scale) {
        fprintf(stderr, "hybrid forward: %s is not FP8 with scales\n", name);
        return -1;
    }
    int id = hip_ds4f_dense_bind_tensor(hip, t);
    if (id < 0) return -1;
    t->gpu_id = id;
    return id;
}

int main(void) {
    /* Keep this gate self-contained: it validates FP8 offload, not the other
     * optional synthetic accuracy/performance modes. */
    setenv("DS4F_FP8_BF16", "0", 1);
    setenv("DS4F_DENSE_MXFP4", "0", 1);
    setenv("DS4F_Q8_DENSE", "0", 1);
    setenv("DS4F_EXACT", "0", 1);
    setenv("DS4F_MHC", "0", 1);
    setenv("DS4F_TIERB2", "0", 1);

    hip_ds4f_dense *hip = hip_ds4f_dense_create(0, 1);
    if (!hip) {
        printf("SKIP: HIP/ROCm device or hipRTC unavailable\n");
        return 0;
    }

    ds4f_config cfg = ds4f_config_from_env();
    cfg.n_layers = 1;
    cfg.max_pos = 128;
    ds4f_model *cpu = ds4f_alloc_synth(cfg, 0, 1, 16, 1);
    ds4f_model *gpu = ds4f_alloc_synth(cfg, 0, 1, 16, 1);
    if (!cpu || !gpu) {
        fprintf(stderr, "hybrid forward: synthetic allocation failed\n");
        ds4f_free(cpu);
        ds4f_free(gpu);
        hip_ds4f_dense_destroy(hip);
        return 1;
    }

    ds4f_layer *ly = &gpu->layers[0];
    int pass = 1;
    pass &= add_tensor(hip, &ly->wq_a, "wq_a") >= 0;
    pass &= add_tensor(hip, &ly->wq_b, "wq_b") >= 0;
    pass &= add_tensor(hip, &ly->wkv,  "wkv") >= 0;
    pass &= add_tensor(hip, &ly->wo_a, "wo_a") >= 0;
    pass &= add_tensor(hip, &ly->wo_b, "wo_b") >= 0;
    pass &= add_tensor(hip, &ly->sh_w1, "sh_w1") >= 0;
    pass &= add_tensor(hip, &ly->sh_w3, "sh_w3") >= 0;
    pass &= add_tensor(hip, &ly->sh_w2, "sh_w2") >= 0;
    if (!pass) {
        ds4f_free(cpu);
        ds4f_free(gpu);
        hip_ds4f_dense_destroy(hip);
        return 1;
    }
    gpu->gpu_dense_ctx = hip;
    gpu->gpu_dense_matvec = count_gpu_matvec;
    gpu->gpu_dense_async_multi = count_gpu_async;
    gpu->gpu_dense_wait = count_gpu_wait;

    const int C = cfg.hidden;
    float *xcpu = (float *)ds4f_mem_alloc(cpu->mem, (size_t)C * sizeof(float), 64, 0);
    float *xgpu = (float *)ds4f_mem_alloc(gpu->mem, (size_t)C * sizeof(float), 64, 0);
    if (!xcpu || !xgpu) {
        fprintf(stderr, "hybrid forward: activation allocation failed\n");
        ds4f_free(cpu); ds4f_free(gpu); hip_ds4f_dense_destroy(hip);
        return 1;
    }
    for (int i = 0; i < C; ++i)
        xcpu[i] = xgpu[i] = ((float)((i * 29) % 101) - 50.0f) / 37.0f;

    int cpu_best = ds4f_forward_token(cpu, xcpu, 0);
    int gpu_best = ds4f_forward_token(gpu, xgpu, 0);
    float x_abs, x_scale, logit_abs, logit_scale;
    float x_rel = compare_vec(xcpu, xgpu, C, &x_abs, &x_scale);
    float logit_rel = compare_vec(cpu->s_logits, gpu->s_logits, cfg.vocab,
                                  &logit_abs, &logit_scale);
    printf("hybrid FP8 forward: gpu_calls=%d async_batches=%d x_max_abs=%.8g x_rel=%.8g "
           "logits_max_abs=%.8g logits_rel=%.8g cpu_argmax=%d gpu_argmax=%d\n",
           g_gpu_calls, g_gpu_async_calls, x_abs, x_rel, logit_abs, logit_rel,
           cpu_best, gpu_best);

    /* The GPU reduction is tree-ordered, so a forward pass is not expected to
     * be bit-identical.  The gate is deliberately tighter than a quality gate:
     * it should catch a wrong bank id, tensor shape, or CPU/GPU layout mismatch. */
    if (g_gpu_calls < 5 || g_gpu_async_calls != 1 || cpu_best != gpu_best ||
        x_rel > 3.0e-4f || x_abs > 3.0e-3f * x_scale ||
        logit_rel > 3.0e-4f || logit_abs > 3.0e-3f * logit_scale)
        pass = 0;
    printf("%s\n", pass ? "PASS" : "FAIL");

    ds4f_free(cpu);
    ds4f_free(gpu);
    hip_ds4f_dense_destroy(hip);
    return pass ? 0 : 1;
}
