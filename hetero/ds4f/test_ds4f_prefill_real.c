/* Real-staged x86 prefill gate.
 *
 * Loads one staged rank twice, compares ds4f_forward_prefill against the
 * exact token-at-a-time path on the same synthetic hidden inputs, and reports
 * throughput plus greedy parity. This is CPU-only by design: it isolates the
 * prefill scheduler/GEMM/attention work from the HIP decode bank. */

#include "../../common/ds4f.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s --stage-dir dir [--config file.json] "
                    "[--model flash|ds4p|ds4fbase] [--ep-size n --ep-rank n "
                    "--threads n --cmgs n --max-pos n] [--layers n] "
                    "[--batch n --iters n --pos0 n] [--debug-env]\n", prog);
}

static void fill_inputs(float *x, int batch, int C, int pos0) {
    for (int k = 0; k < batch; k++) {
        int pos = pos0 + k;
        for (int d = 0; d < C; d++)
            x[(size_t)k * C + d] = (float)(((pos + 3) * (d + 11)) % 101 - 50) / 37.0f;
    }
}

int main(int argc, char **argv) {
    ds4f_runtime_options opt;
    ds4f_runtime_options_init(&opt);
    char config_path[1024] = {0};
    int debug_env = 0, layers = 0, batch = 16, iters = 2, pos0 = 0;

    /* JSON first; explicit arguments below always override it. */
    for (int i = 1; i + 1 < argc; i++)
        if (strcmp(argv[i], "--config") == 0)
            snprintf(config_path, sizeof(config_path), "%s", argv[i + 1]);
    if (config_path[0] && ds4f_runtime_options_load_json(&opt, config_path) != 0) {
        fprintf(stderr, "cannot load DS4F config: %s\n", config_path);
        return 2;
    }
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (strcmp(a, "--config") == 0 && i + 1 < argc) i++;
        else if (strcmp(a, "--stage-dir") == 0 && i + 1 < argc)
            snprintf(opt.stage_dir, sizeof(opt.stage_dir), "%s", argv[++i]);
        else if (strcmp(a, "--model") == 0 && i + 1 < argc) {
            const char *v = argv[++i];
            opt.cfg = strcmp(v, "ds4p") == 0 ? ds4f_pro_config() :
                      strcmp(v, "ds4fbase") == 0 ? ds4f_base_config() :
                      ds4f_default_config();
        } else if (strcmp(a, "--ep-size") == 0 && i + 1 < argc) opt.ep_size = atoi(argv[++i]);
        else if (strcmp(a, "--ep-rank") == 0 && i + 1 < argc) opt.ep_rank = atoi(argv[++i]);
        else if (strcmp(a, "--threads") == 0 && i + 1 < argc) opt.n_threads = atoi(argv[++i]);
        else if (strcmp(a, "--cmgs") == 0 && i + 1 < argc) opt.n_cmgs = atoi(argv[++i]);
        else if (strcmp(a, "--max-pos") == 0 && i + 1 < argc) opt.cfg.max_pos = atoi(argv[++i]);
        else if (strcmp(a, "--layers") == 0 && i + 1 < argc) layers = atoi(argv[++i]);
        else if (strcmp(a, "--batch") == 0 && i + 1 < argc) batch = atoi(argv[++i]);
        else if (strcmp(a, "--iters") == 0 && i + 1 < argc) iters = atoi(argv[++i]);
        else if (strcmp(a, "--pos0") == 0 && i + 1 < argc) pos0 = atoi(argv[++i]);
        else if (strcmp(a, "--debug-env") == 0) debug_env = 1;
        else { usage(argv[0]); return 2; }
    }
    if (getenv("DS4F_DEBUG_ENV") && atoi(getenv("DS4F_DEBUG_ENV"))) debug_env = 1;
    if (debug_env) {
        ds4f_runtime_options envopt = ds4f_runtime_options_debug_env(opt.cfg,
            opt.stage_dir[0] ? opt.stage_dir : NULL, opt.ep_rank, opt.ep_size,
            opt.n_threads, opt.n_cmgs);
        envopt.cfg.max_pos = opt.cfg.max_pos;
        opt = envopt;
        if (!opt.stage_dir[0]) {
            const char *e = getenv("DS4F_STAGE_DIR");
            if (e) snprintf(opt.stage_dir, sizeof(opt.stage_dir), "%s", e);
        }
        if (getenv("DS4F_PREFILL_BATCH")) batch = atoi(getenv("DS4F_PREFILL_BATCH"));
        if (getenv("DS4F_PREFILL_ITERS")) iters = atoi(getenv("DS4F_PREFILL_ITERS"));
    }
    if (!opt.stage_dir[0]) {
        printf("SKIP: pass --stage-dir/--config (or --debug-env for DS4F_STAGE_DIR)\n");
        return 0;
    }
    if (batch < 2 || batch > 128 || iters < 1 || pos0 < 0) {
        usage(argv[0]); return 2;
    }
    if (layers > 0) opt.cfg.n_layers = layers;
    if (opt.cfg.max_pos < pos0 + batch * iters + 1)
        opt.cfg.max_pos = pos0 + batch * iters + 1;
    opt.expert_resident = 0;
    opt.use_hip = 0;

    ds4f_model *batched = ds4f_load_real_opts(&opt);
    if (!batched) { fprintf(stderr, "real prefill: batched model load failed\n"); return 1; }
    ds4f_model *serial = ds4f_load_real_opts(&opt);
    if (!serial) {
        fprintf(stderr, "real prefill: serial model load failed\n");
        ds4f_free(batched); return 1;
    }
    if (!batched->exact || batched->mhc || batched->tierb2 || batched->int8_kv) {
        printf("SKIP: real prefill gate requires exact && !mhc && !tierb2 && !int8_kv\n");
        ds4f_free(serial); ds4f_free(batched); return 0;
    }

    int C = batched->cfg.hidden;
    ds4f_alloc_prefill_batch(batched, batch);
    float *xb = (float *)ds4f_mem_alloc(batched->mem, (size_t)batch * C * sizeof(float), 256, 0);
    float *xs = (float *)ds4f_mem_alloc(serial->mem, (size_t)C * sizeof(float), 256, 0);
    int *out_b = (int *)ds4f_mem_alloc(batched->mem, (size_t)batch * iters * sizeof(int), 64, 0);
    int *out_s = (int *)ds4f_mem_alloc(serial->mem, (size_t)batch * iters * sizeof(int), 64, 0);
    if (!xb || !xs || !out_b || !out_s) {
        fprintf(stderr, "real prefill: scratch allocation failed\n");
        ds4f_free(serial); ds4f_free(batched); return 1;
    }

    double tb0 = now_seconds();
    for (int it = 0; it < iters; it++) {
        int p = pos0 + it * batch;
        fill_inputs(xb, batch, C, p);
        ds4f_forward_prefill(batched, xb, batch, p, out_b + it * batch);
    }
    double tb = now_seconds() - tb0;
    if (getenv("DS4F_PROF") && atoi(getenv("DS4F_PROF")) != 0) {
        double accounted = 0.0;
        for (int i = 0; i <= DS4F_P_TB2PREP; i++) accounted += batched->prof[i];
        printf("real prefill profile (average per token):\n");
        for (int i = 0; i <= DS4F_P_TB2PREP; i++) {
            double ms = batched->prof[i] * 1000.0 / (batch * iters);
            printf("  %-9s %8.3f ms %5.1f%%\n", ds4f_prof_names[i], ms,
                   accounted > 0.0 ? 100.0 * batched->prof[i] / accounted : 0.0);
        }
    }

    double ts0 = now_seconds();
    for (int it = 0; it < iters; it++) {
        int p = pos0 + it * batch;
        for (int k = 0; k < batch; k++) {
            fill_inputs(xs, 1, C, p + k);
            out_s[it * batch + k] = ds4f_forward_token(serial, xs, p + k);
        }
    }
    double ts = now_seconds() - ts0;

    int mismatches = 0;
    for (int i = 0; i < batch * iters; i++) if (out_b[i] != out_s[i]) mismatches++;
    printf("real prefill gate: layers=%d batch=%d tokens=%d threads=%d "
           "batched=%.3f tok/s token=%.3f tok/s speedup=%.3fx argmax_mismatch=%d\n",
           batched->cfg.n_layers, batch, batch * iters, opt.n_threads,
           (batch * iters) / tb, (batch * iters) / ts, ts / tb, mismatches);

    ds4f_free(serial);
    ds4f_free(batched);
    return mismatches == 0 ? 0 : 1;
}
