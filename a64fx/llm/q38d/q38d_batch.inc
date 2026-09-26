/* Independent context slots for bounded, round-robin decode batches. */
#define Q38D_MAX_CONTEXTS 32
typedef struct {
    float *ssm[NLAYER][NVH], *conv[NLAYER][NVH];
    float *kcp[NATTN][NKV][NCMG], *vcp[NATTN][NKV][NCMG];
    q38d_i8_heads qi;
    int32_t *tok;
    int next_token;
    float next_logit;
} q38d_context_slot;
static int context_count = 1;
static q38d_context_slot *context_slots;
static int32_t *context_prompt_tokens;
static char **context_state_dirs;
static int prune_model_parts;
static int warm_kv_depth;

static void context_load_state_list(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) state_fail("context state list open");
    context_state_dirs = calloc((size_t)context_count, sizeof *context_state_dirs);
    if (!context_state_dirs) state_fail("context state list allocation");
    char line[4096];
    for (int b = 0; b < context_count; b++) {
        if (!fgets(line, sizeof(line), f)) state_fail("context state list has too few paths");
        size_t n = strlen(line);
        if (n && line[n - 1] != '\n' && !feof(f)) state_fail("context state path too long");
        while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = 0;
        if (!n) state_fail("empty context state path");
        context_state_dirs[b] = malloc(n + 1);
        if (!context_state_dirs[b]) state_fail("context state path allocation");
        memcpy(context_state_dirs[b], line, n + 1);
    }
    if (fgets(line, sizeof(line), f)) state_fail("context state list has too many paths");
    fclose(f);
}

static size_t context_rss_kib(void) {
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256];
    size_t rss = 0;
    while (fgets(line, sizeof(line), f))
        if (sscanf(line, "VmRSS: %zu kB", &rss) == 1) break;
    fclose(f);
    return rss;
}
static size_t context_memavailable_kib(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    size_t available = 0;
    while (fgets(line, sizeof(line), f))
        if (sscanf(line, "MemAvailable: %zu kB", &available) == 1) break;
    fclose(f);
    return available;
}

static void context_keep_mat(const q38d_mat *m, const void **keep, size_t *count) {
    for (int c = 0; c < 4; c++) if (m->part[c]) keep[(*count)++] = m->part[c];
}

static void context_prune_model(void) {
    size_t before = context_rss_kib();
    const void *keep[(NLAYER + 1) * 12 * 4 + 8];
    size_t count = 0;
    for (int l = 0; l < NLAYER; l++) {
        const q38d_layer *p = &E.L[l];
        const q38d_mat *m[] = {&p->qkv, &p->z, &p->alpha, &p->beta, &p->out,
                              &p->q, &p->k, &p->v, &p->o, &p->gate, &p->up, &p->down};
        for (size_t j = 0; j < sizeof(m) / sizeof(m[0]); j++)
            context_keep_mat(m[j], keep, &count);
    }
    context_keep_mat(&E.head, keep, &count);
    if (E.embed_lb)
        for (int c = 0; c < 4; c++) if (E.embed_lb->part[c]) keep[count++] = E.embed_lb->part[c];
    size_t bytes = q38_lowbit_model_prune_matrix_parts(LB, keep, count);
    fprintf(stderr, "q38d: TP%d group=%d pruned source matrix mappings %.3f GiB rss_before=%.3f GiB rss_after=%.3f GiB\n",
            tp_n, tp_group, (double)bytes / (double)(1ull << 30),
            (double)before / (double)(1ull << 20),
            (double)context_rss_kib() / (double)(1ull << 20));
}

static void context_alloc_owned(int tid) {
    if (context_count == 1) return;
    int hh = ssm_head_of(tid);
    for (int layer = 0; layer < NLAYER; layer++) {
        if (!E.L[layer].ssm || hh < 0) continue;
        context_slots[0].ssm[layer][hh] = ssm_buf[layer][hh];
        context_slots[0].conv[layer][hh] = E.conv_hist[layer][hh];
        for (int b = 1; b < context_count; b++) {
            float *s = aligned_alloc(256, (DS * DS + 2 * DS) * sizeof(float));
            float *v = aligned_alloc(256, 8 * 384 * sizeof(float));
            if (!s || !v) state_fail("context SSM allocation");
            memset(s, 0, (DS * DS + 2 * DS) * sizeof(float));
            memset(v, 0, 8 * 384 * sizeof(float));
            context_slots[b].ssm[layer][hh] = s;
            context_slots[b].conv[layer][hh] = v;
        }
    }
    int c = tid / PER;
    if (tid % PER == 0) {
        int cpk = kv_cpk(), kv = c / cpk, k = c % cpk;
        size_t np = ((size_t)E.max_seq / (KVB * cpk) + 1) * KVB;
        for (int ai = 0; ai < NATTN; ai++) {
            if (kv_i8_mode || kv_i6_mode) {
                context_slots[0].qi.k[ai][kv][k] = KI.k[ai][kv][k];
                context_slots[0].qi.v[ai][kv][k] = KI.v[ai][kv][k];
                context_slots[0].qi.ks[ai][kv][k] = KI.ks[ai][kv][k];
                context_slots[0].qi.vs[ai][kv][k] = KI.vs[ai][kv][k];
                for (int b = 1; b < context_count; b++) {
                    q38d_i8_heads *q = &context_slots[b].qi;
                    q->k[ai][kv][k] = cmg_alloc(np * kv_row_bytes(), c);
                    q->v[ai][kv][k] = cmg_alloc(np * kv_row_bytes(), c);
                    q->ks[ai][kv][k] = cmg_alloc(np * sizeof(float), c);
                    q->vs[ai][kv][k] = cmg_alloc(np * sizeof(float), c);
                    if (!q->k[ai][kv][k] || !q->v[ai][kv][k] || !q->ks[ai][kv][k] || !q->vs[ai][kv][k])
                        state_fail("context compressed KV allocation");
                }
            } else {
                context_slots[0].kcp[ai][kv][k] = E.kcp[ai][kv][k];
                context_slots[0].vcp[ai][kv][k] = E.vcp[ai][kv][k];
                for (int b = 1; b < context_count; b++) {
                    float *kp = cmg_alloc(np * HD * sizeof(float), c);
                    float *vp = cmg_alloc(np * HD * sizeof(float), c);
                    if (!kp || !vp) state_fail("context KV allocation");
                    memset(kp, 0, np * HD * sizeof(float));
                    memset(vp, 0, np * HD * sizeof(float));
                    context_slots[b].kcp[ai][kv][k] = kp;
                    context_slots[b].vcp[ai][kv][k] = vp;
                }
            }
        }
    }
}

/* Called by worker 0 only while all other workers are held at a barrier. */
static void context_select(int b) {
    q38d_context_slot *slot = &context_slots[b];
    for (int layer = 0; layer < NLAYER; layer++) {
        if (!E.L[layer].ssm) continue;
        for (int h = 0; h < NVH; h++) {
            ssm_buf[layer][h] = slot->ssm[layer][h];
            E.conv_hist[layer][h] = slot->conv[layer][h];
        }
    }
    if (kv_i8_mode || kv_i6_mode) KI = slot->qi;
    else for (int ai = 0; ai < NATTN; ai++)
        for (int h = 0; h < NKV; h++) {
            for (int k = 0; k < NCMG; k++) {
                E.kcp[ai][h][k] = slot->kcp[ai][h][k];
                E.vcp[ai][h][k] = slot->vcp[ai][h][k];
            }
            E.kc[ai][h] = slot->kcp[ai][h][0];
            E.vc[ai][h] = slot->vcp[ai][h][0];
        }
    E.next_token = slot->next_token;
    E.next_logit = slot->next_logit;
}

static void context_warm_owned(int tid) {
    if (!warm_kv_depth || tid % PER != 0) return;
    int c = tid / PER, cpk = kv_cpk(), h = c / cpk, part = c % cpk;
    size_t rows = (size_t)kv_count(warm_kv_depth, cpk, part);
    for (int b = 0; b < context_count; b++) {
        q38d_i8_heads *q = &context_slots[b].qi;
        for (int ai = 0; ai < NATTN; ai++) {
            memset(q->k[ai][h][part], 0, rows * kv_row_bytes());
            memset(q->v[ai][h][part], 0, rows * kv_row_bytes());
            float *ks = q->ks[ai][h][part], *vs = q->vs[ai][h][part];
            for (size_t i = 0; i < rows; i++) { ks[i] = 1.0f; vs[i] = 1.0f; }
        }
        if (context_memavailable_kib() < (size_t)2 * 1024 * 1024)
            state_fail("synthetic KV warm crossed 2 GiB MemAvailable guard");
    }
}

static int context_clone_source(int b, int pn) {
    for (int a = 0; a < b; a++)
        if (!strcmp(context_state_dirs[a], context_state_dirs[b]) &&
            !memcmp(context_slots[a].tok, context_slots[b].tok, (size_t)pn * sizeof(int32_t)))
            return a;
    return -1;
}

/* A repeated state directory with identical prompt tokens has already passed
 * the model/prompt header and checksum checks. Copy its resident state into a
 * separate slot so subsequent decoding may diverge independently. */
static void context_clone_loaded_state(int dst_index, int src_index, int pn) {
    q38d_context_slot *dst = &context_slots[dst_index], *src = &context_slots[src_index];
    for (int layer = 0; layer < NLAYER; layer++) {
        if (!E.L[layer].ssm) continue;
        for (int h = 0; h < NVH; h++) {
            if (!src->ssm[layer][h]) continue;
            memcpy(dst->ssm[layer][h], src->ssm[layer][h], (DS * DS + 2 * DS) * sizeof(float));
            memcpy(dst->conv[layer][h], src->conv[layer][h], 8 * 384 * sizeof(float));
        }
    }
    int cpk = kv_cpk();
    for (int ai = 0; ai < NATTN; ai++)
        for (int h = 0; h < NKV; h++)
            for (int part = 0; part < cpk; part++) {
                size_t rows = (size_t)kv_count(pn, cpk, part);
                if (kv_i6_mode) {
                    if (!src->qi.k[ai][h][part]) continue;
                    memcpy(dst->qi.k[ai][h][part], src->qi.k[ai][h][part], rows * kv_row_bytes());
                    memcpy(dst->qi.v[ai][h][part], src->qi.v[ai][h][part], rows * kv_row_bytes());
                    memcpy(dst->qi.ks[ai][h][part], src->qi.ks[ai][h][part], rows * sizeof(float));
                    memcpy(dst->qi.vs[ai][h][part], src->qi.vs[ai][h][part], rows * sizeof(float));
                } else {
                    if (!src->kcp[ai][h][part]) continue;
                    memcpy(dst->kcp[ai][h][part], src->kcp[ai][h][part], rows * HD * sizeof(float));
                    memcpy(dst->vcp[ai][h][part], src->vcp[ai][h][part], rows * HD * sizeof(float));
                }
            }
    dst->next_token = src->next_token;
    dst->next_logit = src->next_logit;
}

static int context_copy_source_index;

static void context_run(int tid, int *gs, int *cs) {
    const int pn = JOB.n_prompt, gn = JOB.n_gen;
    double start = 0, decode_start = 0;
    if (tid == 0) start = now_sec();
    context_warm_owned(tid);
    gbarrier(tid, gs);
    if (tid == 0 && warm_kv_depth) {
        double sec = now_sec() - start;
        fprintf(stderr, "q38d_batch: group=%d synthetic_warm_depth=%d contexts=%d seconds=%.6f rss=%.3f GiB MemAvailable=%.3f GiB\n",
                tp_group, warm_kv_depth, context_count, sec,
                (double)context_rss_kib() / (double)(1ull << 20),
                (double)context_memavailable_kib() / (double)(1ull << 20));
        if (context_memavailable_kib() < (size_t)2 * 1024 * 1024)
            state_fail("synthetic KV warm left less than 2 GiB MemAvailable");
    }
    if (tid == 0) {
        for (int b = 0; b < context_count; b++) {
            q38d_context_slot *slot = &context_slots[b];
            slot->tok = malloc((size_t)(pn + gn) * sizeof(int32_t));
            if (!slot->tok) state_fail("context tokens allocation");
            if (context_prompt_tokens)
                memcpy(slot->tok, context_prompt_tokens + (size_t)b * pn, (size_t)pn * sizeof(int32_t));
            else
                for (int i = 0; i < pn; i++) slot->tok[i] = JOB.tok[(i + b) % pn];
        }
        start = now_sec();
    }
    gbarrier(tid, gs);
    if (context_state_dirs) {
        for (int b = 0; b < context_count; b++) {
            if (tid == 0) {
                context_copy_source_index = context_clone_source(b, pn);
                if (context_copy_source_index >= 0) {
                    double a = now_sec();
                    context_clone_loaded_state(b, context_copy_source_index, pn);
                    fprintf(stderr, "q38d_batch: group=%d context=%d cloned_context=%d copy=%.6f s\n",
                            tp_group, b, context_copy_source_index, now_sec() - a);
                } else {
                    context_select(b);
                    state_in = context_state_dirs[b];
                    state_prompt_id = state_hash(context_slots[b].tok, (size_t)pn * sizeof(int32_t));
                    state_import();
                }
            }
            gbarrier(tid, gs);
            if (context_copy_source_index < 0) {
                uint64_t tt = ticks();
                head_argmax(tid, E.out_norm, gs, &tt);
                if (tid == 0) {
                    if (E.next_token != state_first_token) state_fail("batched handoff first-token mismatch");
                    context_slots[b].next_token = E.next_token;
                    context_slots[b].next_logit = E.next_logit;
                    fprintf(stderr, "q38d_batch: group=%d context=%d state_import=%.6f s first_token=%d rss=%.3f GiB MemAvailable=%.3f GiB\n",
                            tp_group, b, state_read_sec, E.next_token,
                            (double)context_rss_kib() / (double)(1ull << 20),
                            (double)context_memavailable_kib() / (double)(1ull << 20));
                }
            }
            if (tid == 0 && context_memavailable_kib() < (size_t)2 * 1024 * 1024)
                state_fail("batched state import/copy crossed 2 GiB MemAvailable guard");
            gbarrier(tid, gs);
        }
        if (tid == 0) {
            state_in = NULL;
            decode_start = now_sec();
            fprintf(stderr, "q38d_batch: TP%d group=%d contexts=%d state_import_total=%.6f s\n",
                    tp_n, tp_group, context_count, decode_start - start);
        }
    } else {
        for (int b = 0; b < context_count; b++) {
            if (tid == 0) context_select(b);
            gbarrier(tid, gs);
            for (int pos = 0; pos < pn; pos++)
                step(tid, context_slots[b].tok[pos], warm_kv_depth + pos, pos == pn - 1, gs, cs);
            gbarrier(tid, gs);
            if (tid == 0) {
                context_slots[b].next_token = E.next_token;
                context_slots[b].next_logit = E.next_logit;
            }
            gbarrier(tid, gs);
        }
        if (tid == 0) {
            decode_start = now_sec();
            fprintf(stderr, "q38d_batch: TP%d group=%d contexts=%d prompt_each=%d prefill_tokens=%d seconds=%.6f tok_s=%.3f\n",
                    tp_n, tp_group, context_count, pn, context_count * pn,
                    decode_start - start, context_count * pn / (decode_start - start));
        }
    }
    gbarrier(tid, gs);
    for (int n = 0; n < gn; n++)
        for (int b = 0; b < context_count; b++) {
            if (tid == 0) context_select(b);
            gbarrier(tid, gs);
            int cur = E.next_token;
            if (tid == 0) context_slots[b].tok[pn + n] = cur;
            step(tid, cur, warm_kv_depth + pn + n, 1, gs, cs);
            gbarrier(tid, gs);
            if (tid == 0) {
                context_slots[b].next_token = E.next_token;
                context_slots[b].next_logit = E.next_logit;
            }
            gbarrier(tid, gs);
        }
    if (tid == 0) {
        double sec = now_sec() - decode_start;
        fprintf(stderr, "q38d_batch: TP%d group=%d contexts=%d decode_tokens=%d seconds=%.6f aggregate_tok_s=%.3f\n",
                tp_n, tp_group, context_count, context_count * gn, sec,
                context_count * gn / sec);
        for (int b = 0; b < context_count; b++) {
            uint64_t hash = UINT64_C(14695981039346656037);
            for (int n = 0; n < gn; n++) {
                int id = context_slots[b].tok[pn + n];
                hash ^= (uint32_t)id;
                hash *= UINT64_C(1099511628211);
            }
            fprintf(stderr, "q38d_batch: group=%d context=%d output_hash=%016llx first=%d\n",
                    tp_group, b, (unsigned long long)hash, context_slots[b].tok[pn]);
        }
    }
}
