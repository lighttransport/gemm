/* Real-token W4A8 quality gate: exact-f32 experts versus W4A8 experts. */

#include "../../common/ds4f.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_ids(const char *path, ds4f_mem_pool *mem, int **out, int *nout) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    int cap = 128, n = 0, v;
    int *ids = (int *)ds4f_mem_alloc(mem, (size_t)cap * sizeof(*ids), 64, 0);
    if (!ids) { fclose(f); return -1; }
    while (fscanf(f, "%d", &v) == 1) {
        if (n == cap) {
            cap *= 2;
            int *p = (int *)ds4f_mem_realloc(mem, ids,
                                             (size_t)(cap / 2) * sizeof(*ids),
                                             (size_t)cap * sizeof(*ids), 64);
            if (!p) { fclose(f); return -1; }
            ids = p;
        }
        ids[n++] = v;
    }
    fclose(f);
    if (n < 1) return -1;
    *out = ids; *nout = n;
    return 0;
}

static float bf16_to_f32(uint16_t h) {
    union { uint32_t u; float f; } x;
    x.u = (uint32_t)h << 16;
    return x.f;
}

static int embed_lookup(const ds4f_model *m, int tok, float *dst) {
    if (m->emb_rows != m->cfg.vocab || tok < 0 || tok >= m->cfg.vocab)
        return -1;
    const uint16_t *row = m->embed + (size_t)tok * (size_t)m->cfg.hidden;
    for (int i = 0; i < m->cfg.hidden; ++i) dst[i] = bf16_to_f32(row[i]);
    return 0;
}

static float cross_entropy(const float *logits, int n, int target) {
    if (target < 0 || target >= n) return 0.0f;
    float mx = -FLT_MAX;
    for (int i = 0; i < n; ++i) if (logits[i] > mx) mx = logits[i];
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += exp((double)logits[i] - mx);
    return (float)(log(sum) + mx - logits[target]);
}

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s --stage-dir dir --prompt-ids file [--config file.json] "
                    "[--model flash|ds4p|ds4fbase] [--ep-size n --ep-rank n "
                    "--threads n --cmgs n --max-pos n] [--debug-env]\n", prog);
}

int main(int argc, char **argv) {
    ds4f_runtime_options opt;
    ds4f_runtime_options_init(&opt);
    char config_path[1024] = {0}, prompt_path[1024] = {0};
    int debug_env = 0;
    for (int i = 1; i + 1 < argc; i++)
        if (strcmp(argv[i], "--config") == 0)
            snprintf(config_path, sizeof(config_path), "%s", argv[i + 1]);
    if (config_path[0] && ds4f_runtime_options_load_json(&opt, config_path) != 0) {
        fprintf(stderr, "cannot load DS4F config: %s\n", config_path); return 2;
    }
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (strcmp(a, "--config") == 0 && i + 1 < argc) i++;
        else if (strcmp(a, "--stage-dir") == 0 && i + 1 < argc) snprintf(opt.stage_dir, sizeof(opt.stage_dir), "%s", argv[++i]);
        else if (strcmp(a, "--prompt-ids") == 0 && i + 1 < argc) snprintf(prompt_path, sizeof(prompt_path), "%s", argv[++i]);
        else if (strcmp(a, "--model") == 0 && i + 1 < argc) {
            const char *v = argv[++i];
            opt.cfg = strcmp(v, "ds4p") == 0 ? ds4f_pro_config() :
                      strcmp(v, "ds4fbase") == 0 ? ds4f_base_config() : ds4f_default_config();
        } else if (strcmp(a, "--ep-size") == 0 && i + 1 < argc) opt.ep_size = atoi(argv[++i]);
        else if (strcmp(a, "--ep-rank") == 0 && i + 1 < argc) opt.ep_rank = atoi(argv[++i]);
        else if (strcmp(a, "--threads") == 0 && i + 1 < argc) opt.n_threads = atoi(argv[++i]);
        else if (strcmp(a, "--cmgs") == 0 && i + 1 < argc) opt.n_cmgs = atoi(argv[++i]);
        else if (strcmp(a, "--max-pos") == 0 && i + 1 < argc) opt.cfg.max_pos = atoi(argv[++i]);
        else if (strcmp(a, "--debug-env") == 0) debug_env = 1;
        else { usage(argv[0]); return 2; }
    }
    if (getenv("DS4F_DEBUG_ENV") && atoi(getenv("DS4F_DEBUG_ENV"))) debug_env = 1;
    if (debug_env) {
        const char *legacy_stage = opt.stage_dir[0] ? opt.stage_dir : getenv("DS4F_STAGE_DIR");
        ds4f_runtime_options envopt = ds4f_runtime_options_debug_env(opt.cfg,
            legacy_stage, opt.ep_rank, opt.ep_size,
            opt.n_threads, opt.n_cmgs);
        envopt.cfg.max_pos = opt.cfg.max_pos;
        opt = envopt;
        if (!prompt_path[0]) {
            const char *legacy_prompt = getenv("DS4F_PROMPT_IDS");
            if (legacy_prompt) snprintf(prompt_path, sizeof(prompt_path), "%s", legacy_prompt);
        }
    }
    if (!opt.stage_dir[0] || !prompt_path[0]) {
        printf("SKIP: pass --stage-dir and --prompt-ids (or --debug-env for legacy paths)\n");
        return 0;
    }

    ds4f_mem_pool *work_mem = ds4f_mem_pool_create();
    if (!work_mem) { fprintf(stderr, "real-token gate: allocation pool failed\n"); return 1; }
    int *ids = NULL, n_ids = 0;
    if (read_ids(prompt_path, work_mem, &ids, &n_ids) != 0) {
        fprintf(stderr, "cannot read prompt ids: %s\n", prompt_path);
        ds4f_mem_pool_destroy(work_mem);
        return 1;
    }
    ds4f_config cfg = opt.cfg;
    if (cfg.max_pos < n_ids + 2) cfg.max_pos = n_ids + 2;
    opt.cfg = cfg;

    /* Both models share the read-only zero-copy expert mappings, but own their
     * arenas, KV state, and worker pools. This makes the comparison stateful
     * and token-by-token without changing the process environment mid-run. */
    ds4f_model *ref = ds4f_load_real_opts(&opt);
    if (!ref) { ds4f_mem_pool_destroy(work_mem); return 1; }
    if (ref->emb_rows != cfg.vocab || ref->head.rows != cfg.vocab) {
        printf("SKIP: use DS4F_TP_EMBED=0 and DS4F_TP_HEAD=0 for a local-token gate\n");
        ds4f_free(ref); ds4f_mem_pool_destroy(work_mem); return 0;
    }
    ds4f_model *fast = ds4f_load_real_opts(&opt);
    if (!fast) { ds4f_free(ref); ds4f_mem_pool_destroy(work_mem); return 1; }
    ref->mxfp4_w4a8 = 0;
    fast->mxfp4_w4a8 = 1;

    int C = cfg.hidden, V = cfg.vocab;
    float *xr = (float *)ds4f_mem_alloc(work_mem, (size_t)C * sizeof(float), 256, 0);
    float *xf = (float *)ds4f_mem_alloc(work_mem, (size_t)C * sizeof(float), 256, 0);
    if (!xr || !xf) {
        fprintf(stderr, "real-token gate: activation allocation failed\n");
        ds4f_free(fast); ds4f_free(ref); ds4f_mem_pool_destroy(work_mem); return 1;
    }

    int arg_mismatch = 0, nonfinite = 0;
    float worst_abs = 0.0f, worst_rel = 0.0f;
    double ce_ref = 0.0, ce_fast = 0.0;
    int ce_n = 0;
    for (int pos = 0; pos < n_ids; ++pos) {
        if (embed_lookup(ref, ids[pos], xr) != 0 ||
            embed_lookup(fast, ids[pos], xf) != 0) {
            fprintf(stderr, "real-token gate: invalid embedding token %d at pos %d\n", ids[pos], pos);
            nonfinite = 1; break;
        }
        int ar = ds4f_forward_token(ref, xr, pos);
        int af = ds4f_forward_token(fast, xf, pos);
        if (ar != af) arg_mismatch++;
        float pos_abs = 0.0f, pos_rel = 0.0f;
        for (int i = 0; i < V; ++i) {
            float a = ref->s_logits[i], b = fast->s_logits[i];
            if (!isfinite(a) || !isfinite(b)) { nonfinite = 1; continue; }
            float d = fabsf(a - b);
            if (d > pos_abs) pos_abs = d;
            float r = d / fmaxf(1.0f, fabsf(a));
            if (r > pos_rel) pos_rel = r;
        }
        if (pos_abs > worst_abs) worst_abs = pos_abs;
        if (pos_rel > worst_rel) worst_rel = pos_rel;
        if (pos + 1 < n_ids) {
            ce_ref += cross_entropy(ref->s_logits, V, ids[pos + 1]);
            ce_fast += cross_entropy(fast->s_logits, V, ids[pos + 1]);
            ce_n++;
        }
        printf("real-token[%d] input=%d next_ref=%d next_w4a8=%d "
               "logit_abs=%.6g logit_rel=%.6g %s\n",
               pos, ids[pos], ar, af, pos_abs, pos_rel,
               ar == af ? "argmax=PASS" : "argmax=FAIL");
    }
    if (ce_n) { ce_ref /= ce_n; ce_fast /= ce_n; }
    printf("real W4A8 gate: tokens=%d argmax_mismatch=%d finite=%s "
           "worst_logit_abs=%.8g worst_logit_rel=%.8g "
           "mean_ce_exact=%.8g mean_ce_w4a8=%.8g delta=%.8g\n",
           n_ids, arg_mismatch, nonfinite ? "FAIL" : "PASS",
           worst_abs, worst_rel, ce_ref, ce_fast, ce_fast - ce_ref);

    ds4f_free(fast); ds4f_free(ref); ds4f_mem_pool_destroy(work_mem);
    return (!nonfinite && arg_mismatch == 0) ? 0 : 1;
}
