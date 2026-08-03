/* MTP draft/verify probe.
 *
 * This intentionally measures acceptance before attempting an optimization:
 * the main model remains the exact verifier and every proposed token is
 * checked with ds4f_forward_token. The batched verifier can be enabled for
 * the mHC+Tier-B2 configuration once the checkpoint's MTP KV prefix is
 * bootstrapped. A missing mtp.0 block is a clean SKIP. */

#include "../../common/ds4f.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float bf16_to_f32(uint16_t h) {
    union { uint32_t u; float f; } x;
    x.u = (uint32_t)h << 16;
    return x.f;
}

static int read_ids(const char *path, ds4f_mem_pool *mem, int **out, int *nout) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    int cap = 128, n = 0, v;
    int *ids = (int *)ds4f_mem_alloc(mem, (size_t)cap * sizeof(*ids), 64, 0);
    if (!ids) { fclose(f); return -1; }
    while (fscanf(f, "%d", &v) == 1) {
        if (n == cap) {
            int old_cap = cap;
            cap *= 2;
            int *p = (int *)ds4f_mem_realloc(mem, ids,
                                             (size_t)old_cap * sizeof(*ids),
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

static int embed_lookup(const ds4f_model *m, int tok, float *dst) {
    if (!m->embed || m->emb_rows != m->cfg.vocab || tok < 0 || tok >= m->cfg.vocab)
        return -1;
    const uint16_t *row = m->embed + (size_t)tok * (size_t)m->cfg.hidden;
    for (int i = 0; i < m->cfg.hidden; i++) dst[i] = bf16_to_f32(row[i]);
    return 0;
}

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s --stage-dir dir --prompt-ids file [--config file.json] "
                    "[--model flash|ds4p|ds4fbase] [--ep-size n --ep-rank n "
                    "--threads n --cmgs n --max-pos n] [--mtp-k n] "
                    "[--batch-verify] [--debug-env]\n", prog);
}

int main(int argc, char **argv) {
    ds4f_runtime_options opt;
    ds4f_runtime_options_init(&opt);
    opt.mtp = 1;
    char config_path[1024] = {0}, prompt_path[1024] = {0};
    int K = 4, batch_verify = 0, debug_env = 0;

    /* JSON is loaded first; explicit command-line values always win. */
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
        else if (strcmp(a, "--prompt-ids") == 0 && i + 1 < argc)
            snprintf(prompt_path, sizeof(prompt_path), "%s", argv[++i]);
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
        else if (strcmp(a, "--mtp-k") == 0 && i + 1 < argc) K = atoi(argv[++i]);
        else if (strcmp(a, "--batch-verify") == 0) batch_verify = 1;
        else if (strcmp(a, "--debug-env") == 0) debug_env = 1;
        else { usage(argv[0]); return 2; }
    }
    if (getenv("DS4F_DEBUG_ENV") && atoi(getenv("DS4F_DEBUG_ENV"))) debug_env = 1;
    if (debug_env) {
        ds4f_runtime_options envopt = ds4f_runtime_options_debug_env(opt.cfg,
            opt.stage_dir[0] ? opt.stage_dir : NULL, opt.ep_rank, opt.ep_size,
            opt.n_threads, opt.n_cmgs);
        envopt.mtp = opt.mtp ? opt.mtp : 1;
        envopt.cfg.max_pos = opt.cfg.max_pos;
        opt = envopt;
        if (!opt.stage_dir[0]) {
            const char *e = getenv("DS4F_STAGE_DIR");
            if (e) snprintf(opt.stage_dir, sizeof(opt.stage_dir), "%s", e);
        }
        if (!prompt_path[0]) {
            const char *e = getenv("DS4F_PROMPT_IDS");
            if (e) snprintf(prompt_path, sizeof(prompt_path), "%s", e);
        }
        if (!batch_verify) {
            const char *e = getenv("DS4F_MTP_BATCH_VERIFY");
            batch_verify = e && atoi(e);
        }
        if (K == 4) {
            const char *e = getenv("DS4F_MTP_K");
            if (e && *e) K = atoi(e);
        }
    }
    if (!opt.stage_dir[0] || !prompt_path[0]) {
        printf("SKIP: pass --stage-dir and --prompt-ids (or --debug-env for legacy paths)\n");
        return 0;
    }
    if (K < 1) K = 1;
    if (K > 8) K = 8;

    ds4f_mem_pool *work_mem = ds4f_mem_pool_create();
    if (!work_mem) { fprintf(stderr, "MTP probe allocation pool failed\n"); return 1; }
    int *ids = NULL, nids = 0;
    if (read_ids(prompt_path, work_mem, &ids, &nids) != 0) {
        fprintf(stderr, "cannot read prompt ids: %s\n", prompt_path);
        ds4f_mem_pool_destroy(work_mem);
        return 1;
    }
    ds4f_config cfg = opt.cfg;
    if (cfg.max_pos < nids + K + 2) cfg.max_pos = nids + K + 2;
    opt.cfg = cfg;

    ds4f_model *m = ds4f_load_real_opts(&opt);
    if (!m) { ds4f_mem_pool_destroy(work_mem); return 1; }
    if (!m->has_mtp) {
        printf("SKIP: checkpoint has no loaded MTP block (--mtp is enabled)\n");
        ds4f_free(m); ds4f_mem_pool_destroy(work_mem); return 0;
    }
    if (m->emb_rows != cfg.vocab || m->head.rows != cfg.vocab) {
        printf("SKIP: MTP probe needs replicated embedding/head\n");
        ds4f_free(m); ds4f_mem_pool_destroy(work_mem); return 0;
    }

    int C = cfg.hidden, hc = cfg.hc_mult;
    size_t hcC = (size_t)hc * (size_t)C;
    float *x = (float *)ds4f_mem_alloc(work_mem, (size_t)C * sizeof(float), 256, 1);
    float *xe = (float *)ds4f_mem_alloc(work_mem, (size_t)C * sizeof(float), 256, 1);
    float *hc_state = (float *)ds4f_mem_alloc(work_mem, hcC * sizeof(float), 256, 1);
    float *verify_inputs = (float *)ds4f_mem_alloc(work_mem, (size_t)K * (size_t)C * sizeof(float), 256, 1);
    int *draft = (int *)ds4f_mem_calloc(work_mem, (size_t)K, sizeof(int), 64);
    int *verify = (int *)ds4f_mem_calloc(work_mem, (size_t)K, sizeof(int), 64);
    if (!x || !xe || !hc_state || !verify_inputs || !draft || !verify) {
        fprintf(stderr, "MTP probe allocation failed\n");
        ds4f_free(m); ds4f_mem_pool_destroy(work_mem);
        return 1;
    }

    int next = -1;
    for (int p = 0; p < nids; p++) {
        if (embed_lookup(m, ids[p], x) != 0) {
            fprintf(stderr, "MTP probe: invalid prompt token %d\n", ids[p]);
            ds4f_free(m); ds4f_mem_pool_destroy(work_mem);
            return 1;
        }
        next = ds4f_forward_token(m, x, p);
    }
    int main_next = next;
    memcpy(hc_state, m->s_x4, hcC * sizeof(float));
    int mtp_pos = nids;
    for (int k = 0; k < K; k++) {
        if (embed_lookup(m, next, xe) != 0) { draft[k] = -1; break; }
        draft[k] = ds4f_mtp_predict(m, hc_state, xe, mtp_pos + k, NULL, 0);
        memcpy(hc_state, m->s_x4, hcC * sizeof(float));
        next = draft[k];
    }

    int exact_batched = 0, valid = 1;
    if (batch_verify && m->exact && m->mhc && m->tierb2 && !m->int8_kv) {
        for (int k = 0; k < K; k++) {
            int tok_in = k == 0 ? main_next : draft[k - 1];
            if (embed_lookup(m, tok_in, verify_inputs + (size_t)k * C) != 0) {
                valid = 0; break;
            }
        }
        if (valid) {
            ds4f_alloc_prefill_batch(m, K);
            ds4f_forward_verify(m, verify_inputs, K, nids, verify, NULL, NULL);
            exact_batched = 1;
        }
    }

    /* Exact sequential verification remains the portable reference. */
    next = -1;
    int accepted = 0;
    if (!exact_batched) for (int k = 0; k < K; k++) {
        int tok_in = k == 0 ? main_next : draft[k - 1];
        if (embed_lookup(m, tok_in, x) != 0) { valid = 0; break; }
        verify[k] = ds4f_forward_token(m, x, nids + k);
    }
    for (int k = 0; k < K; k++) {
        if (verify[k] == draft[k]) accepted++;
        next = verify[k];
    }
    printf("MTP probe: K=%d accepted=%d/%d rate=%.3f verifier=%s\n",
           K, accepted, K, (double)accepted / (double)K,
           exact_batched ? "batched" : "sequential");
    for (int k = 0; k < K; k++)
        printf("  k=%d draft=%d verifier=%d %s\n", k, draft[k], verify[k],
               draft[k] == verify[k] ? "MATCH" : "REJECT");
    printf("MTP probe note: no speedup claimed; MTP KV prefix bootstrap/rollback is not yet complete.\n");

    ds4f_free(m);
    ds4f_mem_pool_destroy(work_mem);
    return valid ? 0 : 1;
}
