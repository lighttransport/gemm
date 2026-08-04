/* Real-staged DS4F FP8 tensor A/B gate: CPU matvec versus HIPRTC matvec. */

#include "../../common/ds4f.h"
#include "hip_ds4f_dense.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double wall_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s [--config file.json] [--stage-dir dir] [--model flash|ds4p|ds4fbase] "
                    "[--ep-size n --ep-rank n --threads n --cmgs n --max-pos n] "
                    "[--layers n --bank-layers n --iters n --pos0 n --warm n "
                    "--prefill-batch n --prefill-context n] "
                    "[--hip-device n --hip-verbose 0|1 --hip-async 0|1 "
                    "--hip-shared-bf16 0|1 --hip-shared-bf16-layers n "
                    "--hip-shared-fp16 0|1 --hip-shared-fp16-layers n "
                    "--hip-ordered-wkv-layers n "
                    "--hip-ordered-fp8-layers n "
                    "--hip-mxfp4-widen-layers n "
                    "--hip-mxfp4-resident-layers n "
                    "--hip-mxfp4-resident-auto 0|1 --hip-vram-reserve-mb n "
                    "--hip-mxfp4-stream-raw 0|1 "
                    "--hip-prefill-attn 0|1 "
                    "[--hip-mxfp4-gemm-test] [--hip-mxfp4-widened-gemm-test] "
                    "--hip-exact-prefill 0|1] [--debug-env]\n", prog);
}

static int hip_async_enabled(const ds4f_runtime_options *opt) {
    return opt->hip_async != 0;
}

static int hip_shared_bf16_layer(const ds4f_runtime_options *opt, int layer) {
    return opt->hip_shared_bf16 &&
           (opt->hip_shared_bf16_layers <= 0 || layer < opt->hip_shared_bf16_layers);
}

static int hip_shared_fp16_layer(const ds4f_runtime_options *opt, int layer) {
    return opt->hip_shared_fp16 &&
           (opt->hip_shared_fp16_layers <= 0 || layer < opt->hip_shared_fp16_layers);
}

static int hip_ordered_wkv_layer(const ds4f_runtime_options *opt, int layer) {
    return opt->hip_ordered_wkv_layers > 0 &&
           layer < opt->hip_ordered_wkv_layers;
}

static int hip_ordered_fp8_layer(const ds4f_runtime_options *opt, int layer) {
    return opt->hip_ordered_fp8_layers > 0 && layer < opt->hip_ordered_fp8_layers;
}

static int hip_mxfp4_streaming(const ds4f_runtime_options *opt) {
    return opt->hip_mxfp4_widen_layers > 0 ||
           opt->hip_mxfp4_resident_layers > 0 ||
           opt->hip_mxfp4_resident_auto;
}

static float max_rel_error(const float *a, const float *b, int n, float *max_abs) {
    float rel = 0.0f, absmax = 0.0f;
    for (int i = 0; i < n; ++i) {
        float e = fabsf(a[i] - b[i]);
        if (e > absmax) absmax = e;
        float r = e / fmaxf(1.0f, fabsf(a[i]));
        if (r > rel) rel = r;
    }
    *max_abs = absmax;
    return rel;
}

static int check_mxfp4_gemm(ds4f_model *m, hip_ds4f_dense *hip, int widened) {
    if (!m || !m->mxfp4_raw || !m->layers[0].ex_w1 ||
        m->layers[0].ex_w1[0].type != DS4F_MXFP4) {
        printf("real MXFP4 GEMM: SKIP (raw MXFP4 expert bank unavailable)\n");
        return 1;
    }
    ds4f_tensor *t = &m->layers[0].ex_w1[0];
    const int M = 16, K = t->cols, N = t->rows;
    float *x = (float *)ds4f_mem_alloc(m->mem, (size_t)M*K*4, 256, 0);
    float *ref = (float *)ds4f_mem_alloc(m->mem, (size_t)M*N*4, 256, 0);
    float *got = (float *)ds4f_mem_alloc(m->mem, (size_t)M*N*4, 256, 0);
    if (!x || !ref || !got) return 0;
    for (int i = 0; i < M*K; i++) x[i] = (float)((i * 17) % 101 - 50) / 31.0f;
    int old_w4a8 = m->mxfp4_w4a8;
    m->mxfp4_w4a8 = 0; /* compare the widened F32 LUT path first */
    t->gpu_id = -1;
    for (int mm = 0; mm < M; mm++)
        ds4f_matvec(m, ref + (size_t)mm*N, t, x + (size_t)mm*K);
    int id = widened
        ? hip_ds4f_dense_bind_mxfp4_widened_tensor(hip, t)
        : hip_ds4f_dense_bind_mxfp4_tensor(hip, t);
    int rc = id >= 0 ? hip_ds4f_dense_gemm_tensor(hip, got, t, x, M, N, K) : -1;
    float absmax = 0.0f, rel = rc == 0 ? max_rel_error(ref, got, M*N, &absmax) : 1.0f;
    double t0 = wall_seconds();
    int bench_iters = 8;
    for (int it = 0; rc == 0 && it < bench_iters; ++it)
        rc = hip_ds4f_dense_gemm_tensor(hip, got, t, x, M, N, K);
    double elapsed = wall_seconds() - t0;
    printf("real MXFP4 %sGEMM: M=%d N=%d K=%d max_abs=%.8g max_rel=%.8g %s\n",
           widened ? "widened FP8 " : "raw LUT ", M, N, K, absmax, rel,
           rc == 0 && rel <= 2.0e-5f ? "PASS" : "FAIL");
    if (rc == 0)
        printf("real MXFP4 %sGEMM: %.3f ms/call %.1f batch-token/s\n",
               widened ? "widened FP8 " : "raw LUT ",
               elapsed * 1000.0 / bench_iters,
               (double)(M * bench_iters) / elapsed);
    m->mxfp4_w4a8 = old_w4a8;
    return rc == 0 && rel <= 2.0e-5f;
}

static int forward_ab(ds4f_model *m, hip_ds4f_dense *hip,
                      const ds4f_runtime_options *opt) {
    const int C = m->cfg.hidden, V = m->cfg.vocab;
    float *x_cpu = (float *)ds4f_mem_alloc(m->mem, (size_t)C * sizeof(float), 256, 0);
    float *x_gpu = (float *)ds4f_mem_alloc(m->mem, (size_t)C * sizeof(float), 256, 0);
    float *logits_cpu = (float *)ds4f_mem_alloc(m->mem, (size_t)V * sizeof(float), 256, 0);
    if (!x_cpu || !x_gpu || !logits_cpu) {
        fprintf(stderr, "real hybrid forward: allocation failed\n");
        return 0;
    }
    for (int i = 0; i < C; ++i)
        x_cpu[i] = x_gpu[i] = ((float)((i * 29) % 101) - 50.0f) / 37.0f;

    /* Run both passes on the same model. Position zero overwrites the cache
     * slot used by this check; all layer scratch is recomputed. */
    m->gpu_dense_ctx = NULL;
    m->gpu_dense_matvec = NULL;
    m->gpu_dense_async_multi = NULL;
    m->gpu_dense_wait = NULL;
    int cpu_best = ds4f_forward_token(m, x_cpu, 0);
    memcpy(logits_cpu, m->s_logits, (size_t)V * sizeof(float));
    m->gpu_dense_ctx = hip;
    m->gpu_dense_matvec = hip_ds4f_dense_matvec_tensor;
    m->gpu_dense_async_multi = hip_async_enabled(opt)
        ? hip_ds4f_dense_matvec_tensors_async : NULL;
    m->gpu_dense_wait = hip_ds4f_dense_wait_tensors;
    m->gpu_dense_blockdiag = hip_ds4f_dense_matvec_blockdiag;
    m->gpu_dense_gemm = hip_ds4f_dense_gemm_tensor;
    m->gpu_dense_layer_prefetch = NULL;
    m->gpu_dense_layer_begin = hip_mxfp4_streaming(opt)
        ? (opt->hip_mxfp4_stream_raw ? hip_ds4f_dense_stream_layer_raw
                                     : hip_ds4f_dense_stream_layer) : NULL;
    m->gpu_dense_stream_prefill_only = hip_mxfp4_streaming(opt);
    if (hip_mxfp4_streaming(opt))
        m->mxfp4_w4a8 = 0;
    int gpu_best = ds4f_forward_token(m, x_gpu, 0);

    float x_abs = 0.0f, logits_abs = 0.0f;
    float x_rel = max_rel_error(x_cpu, x_gpu, C, &x_abs);
    float logits_rel = max_rel_error(logits_cpu, m->s_logits, V, &logits_abs);
    int finite = 1;
    for (int i = 0; i < C; ++i) if (!isfinite(x_gpu[i])) { finite = 0; break; }
    for (int i = 0; i < V; ++i) if (!isfinite(m->s_logits[i])) { finite = 0; break; }
    printf("real hybrid forward: x_max_abs=%.8g x_rel=%.8g "
           "logits_max_abs=%.8g logits_rel=%.8g cpu_argmax=%d gpu_argmax=%d\n",
           x_abs, x_rel, logits_abs, logits_rel, cpu_best, gpu_best);
    m->gpu_dense_ctx = NULL;
    m->gpu_dense_matvec = NULL;
    m->gpu_dense_async_multi = NULL;
    m->gpu_dense_wait = NULL;
    m->gpu_dense_blockdiag = NULL;
    m->gpu_dense_gemm = NULL;
    m->gpu_dense_layer_begin = NULL;
    m->gpu_dense_layer_prefetch = NULL;
    m->gpu_dense_stream_prefill_only = 0;
    m->gpu_dense_gemm_multi = NULL;
    m->gpu_prefill_attn = NULL;
    m->gpu_dense_mixed = 0;
    int strict = x_rel <= 3.0e-4f && logits_rel <= 3.0e-4f;
    if (!strict && m->cfg.n_layers > 1 && cpu_best == gpu_best && finite)
        printf("real hybrid forward: cumulative multi-layer drift is above the "
               "single-layer gate; argmax remains locked\n");
    return cpu_best == gpu_best && finite &&
           (strict || m->cfg.n_layers > 1);
}

static int benchmark_forward(ds4f_model *m, hip_ds4f_dense *hip, int iters,
                             int pos0, int warm,
                             const ds4f_runtime_options *opt) {
    const int C = m->cfg.hidden;
    float *x = (float *)ds4f_mem_alloc(m->mem, (size_t)C * sizeof(float), 256, 0);
    if (!x) return 0;
    memset(m->prof, 0, sizeof(m->prof));
    m->gpu_dense_ctx = hip;
    m->gpu_dense_matvec = hip_ds4f_dense_matvec_tensor;
    m->gpu_dense_async_multi = hip_async_enabled(opt)
        ? hip_ds4f_dense_matvec_tensors_async : NULL;
    m->gpu_dense_wait = hip_ds4f_dense_wait_tensors;
    m->gpu_dense_blockdiag = hip_ds4f_dense_matvec_blockdiag;
    m->gpu_dense_gemm = hip_ds4f_dense_gemm_tensor;
    m->gpu_dense_gemm_multi = hip_ds4f_dense_gemm_tensors;
    m->gpu_dense_layer_prefetch = NULL;
    m->gpu_prefill_attn = opt->hip_prefill_attn ? hip_ds4f_dense_prefill_attention : NULL;
    m->gpu_dense_mixed = opt->hip_shared_bf16 || opt->hip_shared_fp16;
    m->gpu_dense_layer_begin = hip_mxfp4_streaming(opt)
        ? (opt->hip_mxfp4_stream_raw ? hip_ds4f_dense_stream_layer_raw
                                     : hip_ds4f_dense_stream_layer) : NULL;
    m->gpu_dense_stream_prefill_only = hip_mxfp4_streaming(opt);
    if (hip_mxfp4_streaming(opt))
        m->mxfp4_w4a8 = 0;

    if (pos0 < 0) pos0 = 0;
    if (pos0 + iters > m->cfg.max_pos) iters = m->cfg.max_pos - pos0;
    if (iters < 1) {
        fprintf(stderr, "real hybrid decode: no positions available (pos0=%d max_pos=%d)\n",
                pos0, m->cfg.max_pos);
        return 0;
    }
    if (warm > 0) {
        if (warm > pos0) warm = pos0;
        ds4f_warm_kv(m, warm);
        ds4f_warm_tb2(m, warm);
        fprintf(stderr, "real hybrid decode: synthetic KV context warmed to %d, measuring pos %d..%d\n",
                warm, pos0, pos0 + iters - 1);
    }

    double t0 = wall_seconds();
    int last = -1;
    for (int k = 0; k < iters; ++k) {
        for (int i = 0; i < C; ++i)
            x[i] = ((float)((i * 29 + k * 17) % 101) - 50.0f) / 37.0f;
        last = ds4f_forward_token(m, x, pos0 + k);
    }
    double elapsed = wall_seconds() - t0;
    printf("real hybrid decode: layers=%d tokens=%d %.3f ms/token %.3f tok/s last_argmax=%d\n",
           m->cfg.n_layers, iters, elapsed * 1000.0 / iters,
           iters / elapsed, last);
    if (getenv("DS4F_PROF") && atoi(getenv("DS4F_PROF")) != 0) {
        double accounted = 0.0;
        for (int i = 0; i <= DS4F_P_TB2PREP; i++) accounted += m->prof[i];
        printf("real hybrid phase profile (ms/token):\n");
        for (int i = 0; i <= DS4F_P_TB2PREP; i++) {
            double ms = m->prof[i] * 1000.0 / iters;
            if (ms > 0.001)
                printf("  %-9s %8.3f ms %5.1f%%\n", ds4f_prof_names[i], ms,
                       accounted > 0.0 ? 100.0 * m->prof[i] / accounted : 0.0);
        }
    }

    m->gpu_dense_ctx = NULL;
    m->gpu_dense_matvec = NULL;
    m->gpu_dense_async_multi = NULL;
    m->gpu_dense_wait = NULL;
    m->gpu_dense_blockdiag = NULL;
    m->gpu_dense_gemm = NULL;
    m->gpu_dense_gemm_multi = NULL;
    m->gpu_prefill_attn = NULL;
    m->gpu_dense_mixed = 0;
    m->gpu_dense_layer_begin = NULL;
    m->gpu_dense_layer_prefetch = NULL;
    m->gpu_dense_stream_prefill_only = 0;
    return 1;
}

static void fill_prefill_inputs(float *x, int batch, int C, int pos0) {
    for (int mm = 0; mm < batch; mm++) for (int i = 0; i < C; i++)
        x[(size_t)mm*C+i] = ((float)((i * 29 + (mm + pos0) * 17) % 101) - 50.0f) / 37.0f;
}

static int benchmark_prefill(ds4f_model *m, hip_ds4f_dense *hip, int batch,
                             int context, const ds4f_runtime_options *opt) {
    if (!m->exact || m->mhc || m->tierb2 || m->int8_kv) {
        fprintf(stderr, "real hybrid prefill: requires exact && !mhc && !tierb2 && !int8_kv\n");
        return 0;
    }
    int C = m->cfg.hidden;
    float *x = (float *)ds4f_mem_alloc(m->mem, (size_t)batch * C * sizeof(float), 256, 0);
    int warm_batch = context > 0 ? m->cfg.window_size : 0;
    if (warm_batch > context) warm_batch = context;
    float *warm_x = warm_batch > 0
        ? (float *)ds4f_mem_alloc(m->mem, (size_t)warm_batch * C * sizeof(float), 256, 0)
        : NULL;
    int *warm_tok = warm_batch > 0
        ? (int *)ds4f_mem_alloc(m->mem, (size_t)warm_batch * sizeof(int), 64, 0)
        : NULL;
    int *cpu_tok = (int *)ds4f_mem_alloc(m->mem, (size_t)batch * sizeof(int), 64, 0);
    int *gpu_tok = (int *)ds4f_mem_alloc(m->mem, (size_t)batch * sizeof(int), 64, 0);
    const char *diag_env = getenv("DS4F_PREFILL_DIAG");
    int diag = diag_env && atoi(diag_env) != 0;
    int hrows = m->head.rows;
    float *cpu_logits = diag
        ? (float *)ds4f_mem_alloc(m->mem, (size_t)batch * hrows * sizeof(float), 256, 0)
        : NULL;
    double cpu_prof[DS4F_NPHASE], gpu_prof[DS4F_NPHASE];
    if (!x || !cpu_tok || !gpu_tok || (warm_batch > 0 && (!warm_x || !warm_tok)) ||
        (diag && !cpu_logits)) return 0;
    if (context < 0 || context + batch > m->cfg.max_pos) {
        fprintf(stderr, "real hybrid prefill: context=%d batch=%d exceeds max_pos=%d\n",
                context, batch, m->cfg.max_pos);
        return 0;
    }
    ds4f_alloc_prefill_batch(m, batch > warm_batch ? batch : warm_batch);
    fill_prefill_inputs(x, batch, C, context);
    if (warm_batch > 0) fill_prefill_inputs(warm_x, warm_batch, C, context - warm_batch);

    /* First run the same batched forward with all device hooks detached. */
    m->gpu_dense_ctx = NULL;
    m->gpu_dense_matvec = NULL;
    m->gpu_dense_async_multi = NULL;
    m->gpu_dense_wait = NULL;
    m->gpu_dense_blockdiag = NULL;
    m->gpu_dense_gemm = NULL;
    m->gpu_dense_gemm_multi = NULL;
    m->gpu_dense_mixed = 0;
    memset(m->prof, 0, sizeof(m->prof));
    double warm_cpu_s = 0.0, warm_gpu_s = 0.0;
    if (warm_batch > 0) {
        double tw = wall_seconds();
        ds4f_forward_prefill(m, warm_x, warm_batch, context - warm_batch, warm_tok);
        warm_cpu_s = wall_seconds() - tw;
    }
    double t0 = wall_seconds();
    ds4f_forward_prefill(m, x, batch, context, cpu_tok);
    double cpu_s = wall_seconds() - t0;
    memcpy(cpu_prof, m->prof, sizeof(cpu_prof));
    if (diag) memcpy(cpu_logits, m->p_logits,
                     (size_t)batch * hrows * sizeof(float));

    m->gpu_dense_ctx = hip;
    m->gpu_dense_matvec = hip_ds4f_dense_matvec_tensor;
    m->gpu_dense_async_multi = hip_async_enabled(opt)
        ? hip_ds4f_dense_matvec_tensors_async : NULL;
    m->gpu_dense_wait = hip_ds4f_dense_wait_tensors;
    m->gpu_dense_blockdiag = hip_ds4f_dense_matvec_blockdiag;
    m->gpu_dense_gemm = hip_ds4f_dense_gemm_tensor;
    m->gpu_dense_gemm_multi = hip_ds4f_dense_gemm_tensors;
    m->gpu_prefill_attn = opt->hip_prefill_attn ? hip_ds4f_dense_prefill_attention : NULL;
    m->gpu_dense_layer_prefetch = opt->hip_mxfp4_stream_raw &&
        hip_mxfp4_streaming(opt)
        ? hip_ds4f_dense_prefetch_layer_raw : NULL;
    m->gpu_dense_layer_begin = hip_mxfp4_streaming(opt)
        ? (opt->hip_mxfp4_stream_raw ? hip_ds4f_dense_begin_layer
                                     : hip_ds4f_dense_stream_layer) : NULL;
    m->gpu_dense_stream_prefill_only = hip_mxfp4_streaming(opt);
    m->gpu_dense_mixed = opt->hip_shared_bf16 || opt->hip_shared_fp16;
    memset(m->prof, 0, sizeof(m->prof));
    if (warm_batch > 0) {
        double tw = wall_seconds();
        ds4f_forward_prefill(m, warm_x, warm_batch, context - warm_batch, warm_tok);
        warm_gpu_s = wall_seconds() - tw;
    }
    t0 = wall_seconds();
    ds4f_forward_prefill(m, x, batch, context, gpu_tok);
    double gpu_s = wall_seconds() - t0;
    memcpy(gpu_prof, m->prof, sizeof(gpu_prof));

    int mismatches = 0;
    for (int i = 0; i < batch; i++) if (cpu_tok[i] != gpu_tok[i]) mismatches++;
    printf("real hybrid prefill: layers=%d context=%d batch=%d cpu=%.3f tok/s gpu=%.3f tok/s "
           "speedup=%.3fx argmax_mismatch=%d", m->cfg.n_layers, context, batch,
           batch / cpu_s, batch / gpu_s, cpu_s / gpu_s, mismatches);
    if (warm_batch > 0)
        printf(" warm_tail=%d cpu=%.3fs gpu=%.3fs", warm_batch, warm_cpu_s, warm_gpu_s);
    putchar('\n');
    if (diag) {
        float max_abs = 0.0f, max_rel = 0.0f;
        for (int mm = 0; mm < batch; mm++) {
            const float *a = cpu_logits + (size_t)mm * hrows;
            const float *b = m->p_logits + (size_t)mm * hrows;
            for (int v = 0; v < hrows; v++) {
                float e = fabsf(a[v] - b[v]);
                if (e > max_abs) max_abs = e;
                float r = e / fmaxf(1.0f, fabsf(a[v]));
                if (r > max_rel) max_rel = r;
            }
            if (cpu_tok[mm] != gpu_tok[mm]) {
                int cb = cpu_tok[mm] - m->head_r0;
                int gb = gpu_tok[mm] - m->head_r0;
                printf("  prefill mismatch token=%d cpu=%d gpu=%d "
                       "cpu_logit=%.8g gpu_at_cpu=%.8g "
                       "gpu_logit=%.8g cpu_at_gpu=%.8g\n", mm,
                       cpu_tok[mm], gpu_tok[mm], a[cb], b[cb], b[gb], a[gb]);
            }
        }
        printf("real hybrid prefill logits: max_abs=%.8g max_rel=%.8g\n",
               max_abs, max_rel);
    }
    const char *prof = getenv("DS4F_PROF");
    if (prof && atoi(prof) != 0) {
        double cpu_total = 0.0, gpu_total = 0.0;
        for (int i = 0; i <= DS4F_P_TB2PREP; i++) {
            cpu_total += cpu_prof[i];
            gpu_total += gpu_prof[i];
        }
        printf("real hybrid prefill profile (ms/token):\n");
        for (int i = 0; i <= DS4F_P_TB2PREP; i++) {
            double cms = cpu_prof[i] * 1000.0 / batch;
            double gms = gpu_prof[i] * 1000.0 / batch;
            if (cms > 0.001 || gms > 0.001)
                printf("  %-9s cpu=%8.3f (%5.1f%%) gpu=%8.3f (%5.1f%%)\n",
                       ds4f_prof_names[i], cms,
                       cpu_total > 0.0 ? 100.0 * cpu_prof[i] / cpu_total : 0.0,
                       gms, gpu_total > 0.0 ? 100.0 * gpu_prof[i] / gpu_total : 0.0);
        }
    }

    m->gpu_dense_ctx = NULL;
    m->gpu_dense_matvec = NULL;
    m->gpu_dense_async_multi = NULL;
    m->gpu_dense_wait = NULL;
    m->gpu_dense_blockdiag = NULL;
    m->gpu_dense_gemm = NULL;
    m->gpu_dense_gemm_multi = NULL;
    m->gpu_prefill_attn = NULL;
    m->gpu_dense_layer_prefetch = NULL;
    m->gpu_dense_layer_begin = NULL;
    m->gpu_dense_stream_prefill_only = 0;
    m->gpu_dense_mixed = 0;
    return mismatches == 0;
}

static int check_tensor(ds4f_model *m, hip_ds4f_dense *hip,
                        const char *name, const ds4f_tensor *t, int id,
                        int benchmark) {
    if ((t->type != DS4F_FP8 && t->type != DS4F_BF16) || !t->w ||
        (t->type == DS4F_FP8 && !t->scale)) {
        fprintf(stderr, "real dense A/B: %s has an unsupported tensor layout\n", name);
        return 0;
    }
    float *x = (float *)ds4f_mem_alloc(m->mem, (size_t)t->cols * sizeof(float), 256, 0);
    float *cpu = (float *)ds4f_mem_alloc(m->mem, (size_t)t->rows * sizeof(float), 256, 0);
    float *gpu = (float *)ds4f_mem_alloc(m->mem, (size_t)t->rows * sizeof(float), 256, 0);
    if (!x || !cpu || !gpu) {
        fprintf(stderr, "real DS4F A/B allocation failed for %s\n", name);
        return 0;
    }
    for (int i = 0; i < t->cols; ++i)
        x[i] = ((float)((i * 31 + t->rows) % 127) - 63.0f) / 41.0f;

    ds4f_matvec(m, cpu, t, x);
    int rc = hip_ds4f_dense_matvec_id(hip, id, x, gpu);
    if (rc != 0) {
        return 0;
    }

    float max_abs = 0.0f;
    float max_rel = max_rel_error(cpu, gpu, t->rows, &max_abs);
    int pass = max_rel <= 3.0e-5f && max_abs <= 3.0e-2f;
    printf("real dense A/B: layer=0 tensor=%s rows=%d cols=%d max_abs=%.8g max_rel=%.8g %s\n",
           name, t->rows, t->cols, max_abs, max_rel, pass ? "PASS" : "FAIL");

    if (benchmark) {
        enum { ITERS = 20 };
        double t0 = wall_seconds();
        for (int i = 0; i < ITERS; ++i)
            if (hip_ds4f_dense_matvec_id(hip, id, x, gpu) != 0) { pass = 0; break; }
        double elapsed = wall_seconds() - t0;
        if (pass) {
            size_t bytes = (size_t)t->rows * (size_t)t->cols + ds4f_sbytes(t->type, t->rows, t->cols);
            printf("real %s persistent: %.3f ms/call %.2f GB/s (weight+scale read)\n",
                   name, elapsed * 1000.0 / ITERS,
                   (double)bytes * ITERS / (elapsed * 1e9));
        }
    }
    return pass;
}

int main(int argc, char **argv) {
    ds4f_runtime_options opt;
    ds4f_runtime_options_init(&opt);
    char config_path[1024] = {0};
    int debug_env = 0, bank_layers = 1, layers = 0;
    int mxfp4_test = 0, mxfp4_widened_test = 0;
    int iters = 0, pos0 = 1, warm = 0, prefill_batch = 0, prefill_context = 0;
    /* Load JSON first so explicit command-line values have the conventional
     * higher precedence regardless of where --config appears in argv. */
    for (int i = 1; i + 1 < argc; i++)
        if (strcmp(argv[i], "--config") == 0)
            snprintf(config_path, sizeof(config_path), "%s", argv[i + 1]);
    if (config_path[0] && ds4f_runtime_options_load_json(&opt, config_path) != 0) {
        fprintf(stderr, "cannot load DS4F config: %s\n", config_path); return 2;
    }
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (strcmp(a, "--config") == 0 && i + 1 < argc) { i++; }
        else if (strcmp(a, "--stage-dir") == 0 && i + 1 < argc) snprintf(opt.stage_dir, sizeof(opt.stage_dir), "%s", argv[++i]);
        else if (strcmp(a, "--model") == 0 && i + 1 < argc) {
            const char *v = argv[++i];
            opt.cfg = strcmp(v, "ds4p") == 0 ? ds4f_pro_config() :
                      strcmp(v, "ds4fbase") == 0 ? ds4f_base_config() : ds4f_default_config();
        } else if (strcmp(a, "--ep-size") == 0 && i + 1 < argc) opt.ep_size = atoi(argv[++i]);
        else if (strcmp(a, "--ep-rank") == 0 && i + 1 < argc) opt.ep_rank = atoi(argv[++i]);
        else if (strcmp(a, "--threads") == 0 && i + 1 < argc) opt.n_threads = atoi(argv[++i]);
        else if (strcmp(a, "--cmgs") == 0 && i + 1 < argc) opt.n_cmgs = atoi(argv[++i]);
        else if (strcmp(a, "--max-pos") == 0 && i + 1 < argc) opt.cfg.max_pos = atoi(argv[++i]);
        else if (strcmp(a, "--layers") == 0 && i + 1 < argc) layers = atoi(argv[++i]);
        else if (strcmp(a, "--bank-layers") == 0 && i + 1 < argc) bank_layers = atoi(argv[++i]);
        else if (strcmp(a, "--iters") == 0 && i + 1 < argc) iters = atoi(argv[++i]);
        else if (strcmp(a, "--pos0") == 0 && i + 1 < argc) pos0 = atoi(argv[++i]);
        else if (strcmp(a, "--warm") == 0 && i + 1 < argc) warm = atoi(argv[++i]);
        else if (strcmp(a, "--prefill-batch") == 0 && i + 1 < argc) prefill_batch = atoi(argv[++i]);
        else if (strcmp(a, "--prefill-context") == 0 && i + 1 < argc) prefill_context = atoi(argv[++i]);
        else if (strcmp(a, "--hip-device") == 0 && i + 1 < argc) opt.hip_device = atoi(argv[++i]);
        else if (strcmp(a, "--hip-verbose") == 0 && i + 1 < argc) opt.hip_verbose = atoi(argv[++i]);
        else if (strcmp(a, "--hip-async") == 0 && i + 1 < argc) opt.hip_async = atoi(argv[++i]);
        else if (strcmp(a, "--hip-shared-bf16") == 0 && i + 1 < argc) opt.hip_shared_bf16 = atoi(argv[++i]);
        else if (strcmp(a, "--hip-shared-bf16-layers") == 0 && i + 1 < argc) opt.hip_shared_bf16_layers = atoi(argv[++i]);
        else if (strcmp(a, "--hip-shared-fp16") == 0 && i + 1 < argc) opt.hip_shared_fp16 = atoi(argv[++i]);
        else if (strcmp(a, "--hip-shared-fp16-layers") == 0 && i + 1 < argc) opt.hip_shared_fp16_layers = atoi(argv[++i]);
        else if (strcmp(a, "--hip-ordered-wkv-layers") == 0 && i + 1 < argc) opt.hip_ordered_wkv_layers = atoi(argv[++i]);
        else if (strcmp(a, "--hip-ordered-fp8-layers") == 0 && i + 1 < argc) opt.hip_ordered_fp8_layers = atoi(argv[++i]);
        else if (strcmp(a, "--hip-mxfp4-widen-layers") == 0 && i + 1 < argc) opt.hip_mxfp4_widen_layers = atoi(argv[++i]);
        else if (strcmp(a, "--hip-mxfp4-resident-layers") == 0 && i + 1 < argc) opt.hip_mxfp4_resident_layers = atoi(argv[++i]);
        else if (strcmp(a, "--hip-mxfp4-resident-auto") == 0 && i + 1 < argc) opt.hip_mxfp4_resident_auto = atoi(argv[++i]);
        else if (strcmp(a, "--hip-vram-reserve-mb") == 0 && i + 1 < argc) opt.hip_vram_reserve_mb = atoi(argv[++i]);
        else if (strcmp(a, "--hip-mxfp4-stream-raw") == 0 && i + 1 < argc) opt.hip_mxfp4_stream_raw = atoi(argv[++i]);
        else if (strcmp(a, "--hip-prefill-attn") == 0 && i + 1 < argc) opt.hip_prefill_attn = atoi(argv[++i]);
        else if (strcmp(a, "--hip-mxfp4-gemm-test") == 0) mxfp4_test = 1;
        else if (strcmp(a, "--hip-mxfp4-widened-gemm-test") == 0) mxfp4_widened_test = 1;
        else if (strcmp(a, "--hip-exact-prefill") == 0 && i + 1 < argc) opt.hip_exact_prefill = atoi(argv[++i]);
        else if (strcmp(a, "--debug-env") == 0) debug_env = 1;
        else { usage(argv[0]); return 2; }
    }
    if (getenv("DS4F_DEBUG_ENV") && atoi(getenv("DS4F_DEBUG_ENV"))) debug_env = 1;
    if (debug_env) {
        ds4f_runtime_options envopt = ds4f_runtime_options_debug_env(opt.cfg,
            opt.stage_dir[0] ? opt.stage_dir : NULL, opt.ep_rank, opt.ep_size,
            opt.n_threads, opt.n_cmgs);
        envopt.cfg.max_pos = opt.cfg.max_pos;
        envopt.hip_device = opt.hip_device; envopt.hip_async = opt.hip_async;
        envopt.hip_verbose = opt.hip_verbose; opt = envopt;
    }
    if (!opt.stage_dir[0]) {
        printf("SKIP: pass --stage-dir/--config (or --debug-env for DS4F_STAGE_DIR)\n");
        return 0;
    }
    ds4f_config cfg = opt.cfg;
    if (layers > 0) cfg.n_layers = layers;
    if (cfg.max_pos < 2) cfg.max_pos = 2;
    if (cfg.max_pos > 16384) cfg.max_pos = 16384;
    opt.cfg = cfg;
    if (opt.hip_mxfp4_widen_layers > 0)
        mxfp4_test = mxfp4_widened_test = 0;
    if (bank_layers < 1) bank_layers = 1;
    if (bank_layers > cfg.n_layers) bank_layers = cfg.n_layers;
    opt.cfg.n_layers = bank_layers;
    ds4f_model *m = ds4f_load_real_opts(&opt);
    if (!m) {
        fprintf(stderr, "real DS4F load failed\n");
        return 1;
    }
    hip_ds4f_dense *hip = hip_ds4f_dense_create_ex(opt.hip_device, opt.hip_verbose,
        opt.hip_ordered_fp8_layers > 0 || opt.hip_ordered_wkv_layers > 0);
    if (!hip) {
        printf("SKIP: HIP/hipRTC unavailable\n");
        ds4f_free(m);
        return 0;
    }

    ds4f_layer *ly = &m->layers[0];
    struct { const char *name; ds4f_tensor *t; int bench; } cases[] = {
        { "wq_a", &ly->wq_a, 0 }, { "wq_b", &ly->wq_b, 1 },
        { "wkv",  &ly->wkv,  0 }, { "wo_a", &ly->wo_a, 0 },
        { "wo_b", &ly->wo_b, 0 }, { "sh_w1", &ly->sh_w1, 0 },
        { "sh_w3", &ly->sh_w3, 0 }, { "sh_w2", &ly->sh_w2, 0 },
        { "gate",  &ly->gate,  0 },
    };
    int pass = 1;
    int ids[sizeof(cases) / sizeof(cases[0])];
    size_t bank_bytes = 0;
    int bank_matrices = 0;
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        if (i == 8 && !hip_shared_bf16_layer(&opt, 0)) {
            ids[i] = -1;
            continue;
        }
        int shared_fp16 = hip_shared_fp16_layer(&opt, 0) && i >= 5 &&
                          cases[i].t->type == DS4F_FP8;
        int shared_bf16 = !shared_fp16 && hip_shared_bf16_layer(&opt, 0) && i >= 5 &&
                          cases[i].t->type == DS4F_FP8;
        int ordered_wkv = (i == 2 && hip_ordered_wkv_layer(&opt, 0)) ||
                          (hip_ordered_fp8_layer(&opt, 0) && !shared_fp16 && !shared_bf16);
        ids[i] = cases[i].t->type == DS4F_BF16
            ? hip_ds4f_dense_bind_bf16_tensor(hip, cases[i].t)
            : ordered_wkv
                ? hip_ds4f_dense_bind_fp8_ordered_tensor(hip, cases[i].t)
            : shared_fp16
                ? hip_ds4f_dense_bind_fp8_fp16_tensor(hip, cases[i].t)
                : shared_bf16
                ? hip_ds4f_dense_bind_fp8_bf16_tensor(hip, cases[i].t)
                : hip_ds4f_dense_bind_tensor(hip, cases[i].t);
        if (ids[i] < 0) pass = 0;
        else {
            bank_bytes += (shared_fp16 || shared_bf16)
                ? (size_t)cases[i].t->rows * (size_t)cases[i].t->cols * sizeof(uint16_t)
                : ds4f_wbytes(cases[i].t->type, cases[i].t->rows, cases[i].t->cols)
                    + ds4f_sbytes(cases[i].t->type, cases[i].t->rows, cases[i].t->cols);
            bank_matrices++;
        }
    }
    for (int L = 1; L < cfg.n_layers; ++L) {
        ds4f_layer *z = &m->layers[L];
        ds4f_tensor *ts[] = {
            &z->wq_a, &z->wq_b, &z->wkv, &z->wo_a, &z->wo_b,
            &z->sh_w1, &z->sh_w3, &z->sh_w2, &z->gate,
        };
        for (size_t j = 0; j < sizeof(ts) / sizeof(ts[0]); ++j) {
            if (j == 8 && !hip_shared_bf16_layer(&opt, L)) continue;
            int shared_fp16 = hip_shared_fp16_layer(&opt, L) && j >= 5 &&
                              ts[j]->type == DS4F_FP8;
            int shared_bf16 = !shared_fp16 && hip_shared_bf16_layer(&opt, L) && j >= 5 &&
                              ts[j]->type == DS4F_FP8;
            int ordered_wkv = (j == 2 && hip_ordered_wkv_layer(&opt, L)) ||
                              (hip_ordered_fp8_layer(&opt, L) && !shared_fp16 && !shared_bf16);
            int id = ts[j]->type == DS4F_BF16
                ? hip_ds4f_dense_bind_bf16_tensor(hip, ts[j])
                : ordered_wkv
                    ? hip_ds4f_dense_bind_fp8_ordered_tensor(hip, ts[j])
                : shared_fp16
                    ? hip_ds4f_dense_bind_fp8_fp16_tensor(hip, ts[j])
                    : shared_bf16
                    ? hip_ds4f_dense_bind_fp8_bf16_tensor(hip, ts[j])
                    : hip_ds4f_dense_bind_tensor(hip, ts[j]);
            if (id < 0) pass = 0;
            else {
                bank_bytes += (shared_fp16 || shared_bf16)
                    ? (size_t)ts[j]->rows * (size_t)ts[j]->cols * sizeof(uint16_t)
                    : ds4f_wbytes(ts[j]->type, ts[j]->rows, ts[j]->cols)
                        + ds4f_sbytes(ts[j]->type, ts[j]->rows, ts[j]->cols);
                bank_matrices++;
            }
        }
    }
    int head_id = -1;
    if (m->head.type == DS4F_BF16)
        head_id = hip_ds4f_dense_bind_bf16_tensor(hip, &m->head);
    if (head_id < 0) pass = 0;
    else {
        bank_bytes += ds4f_wbytes(m->head.type, m->head.rows, m->head.cols);
        bank_matrices++;
    }
    if (pass && (opt.hip_mxfp4_resident_layers > 0 || opt.hip_mxfp4_resident_auto)) {
        int nr = opt.hip_mxfp4_resident_layers;
        if (opt.hip_mxfp4_resident_auto)
            nr = hip_ds4f_dense_recommend_mxfp4_resident_layers(
                hip, m->layers, cfg.n_layers, opt.hip_mxfp4_stream_raw,
                opt.hip_vram_reserve_mb > 0 ? opt.hip_vram_reserve_mb : 512);
        if (nr > cfg.n_layers) nr = cfg.n_layers;
        for (int L = 0; L < nr; ++L)
            if (hip_ds4f_dense_resident_mxfp4_layer(
                    hip, &m->layers[L], opt.hip_mxfp4_stream_raw) != 0) {
                fprintf(stderr, "GPU MXFP4 resident upload failed at layer %d\n", L);
                pass = 0;
                break;
            }
        if (pass)
            fprintf(stderr, "GPU MXFP4 resident experts: layers=%d%s mode=%s\n",
                    nr, opt.hip_mxfp4_resident_auto ? " (auto)" : "",
                    opt.hip_mxfp4_stream_raw ? "raw" : "widened");
    }
    printf("GPU dense bank: layers=%d matrices=%d resident=%.3f GB\n",
           cfg.n_layers, bank_matrices, bank_bytes / 1e9);
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i)
        if (ids[i] >= 0)
            pass &= check_tensor(m, hip, cases[i].name, cases[i].t,
                                 ids[i], cases[i].bench);
    if (head_id >= 0)
        pass &= check_tensor(m, hip, "head_bf16", &m->head, head_id, 0);
    if (pass) {
        if (mxfp4_test) pass &= check_mxfp4_gemm(m, hip, 0);
        if (mxfp4_widened_test) pass &= check_mxfp4_gemm(m, hip, 1);
        /* With one layer this is the small A/B gate; with the full bank it is
         * the production multi-layer attachment check.  EP>1 intentionally
         * remains a mechanical path check because the local shard is partial. */
        pass &= forward_ab(m, hip, &opt);
        if (iters > 0) pass &= benchmark_forward(m, hip, iters, pos0, warm, &opt);
        if (prefill_batch > 1)
            pass &= benchmark_prefill(m, hip, prefill_batch, prefill_context, &opt);
    }
    printf("%s\n", pass ? "PASS" : "FAIL");

    hip_ds4f_dense_destroy(hip);
    ds4f_free(m);
    return pass ? 0 : 1;
}
