/* Context-sensitive target oracle, deliberately approximate draft hiddens,
 * every rejection prefix and ragged delivery bound. This checks the actual
 * controller's target rollback, MTP teacher forcing and token accounting. */
#define GLM53F_CLOCK_H
static double mock_time;
static double glm53f_clock(void) { return mock_time += 0.001; }
#include "glm53f_mtp_spec_12n.c"
#include <stdint.h>
#include <stdio.h>

enum { MAX_POSITION = 512, STEPS = 128 };
struct glm53f_target_model_12n { int position, bulk_calls; unsigned hash; float head[5][HIDDEN]; };
struct glm53f_target_snapshot_12n { int position; unsigned hash; };
struct glm53f_mtp_context_12n {
    int length, draft_index, reject_at;
    int input[MAX_POSITION];
    float parent[MAX_POSITION][3], head[HIDDEN];
};
static int expected_token[MAX_POSITION];
static unsigned expected_hash[MAX_POSITION];
static unsigned append(unsigned hash, int token) {
    return (hash * 313u + (unsigned)token + 1) & 65535u;
}
static int prediction(int position, unsigned hash) {
    return 100 + (int)((hash * 17u + (unsigned)position * 31u) % 150000u);
}
static void hidden(float *out, int position, unsigned hash) {
    memset(out, 0, HIDDEN * sizeof(float));
    out[0] = (float)position; out[1] = (float)hash;
    out[2] = 11;
}
static int step(glm53f_target_model_12n *m, int input, float *h) {
    m->hash = append(m->hash, input); ++m->position;
    if (h) hidden(h, m->position, m->hash);
    return prediction(m->position, m->hash);
}
glm53f_target_snapshot_12n *glm53f_target_snapshot_create_12n(const glm53f_target_model_12n *m) {
    return m ? calloc(1, sizeof(glm53f_target_snapshot_12n)) : NULL;
}
void glm53f_target_snapshot_free_12n(glm53f_target_snapshot_12n *s) { free(s); }
int glm53f_target_snapshot_restore_12n(glm53f_target_model_12n *m, const glm53f_target_snapshot_12n *s) {
    m->position = s->position; m->hash = s->hash; return 0;
}
int glm53f_target_model_step_12n(glm53f_target_model_12n *m, int input,
        int *next, float *logit, float *h) {
    *next = step(m, input, h); *logit = 1;
    hidden(m->head[0], m->position, m->hash);
    if (h) h[2] = 0; /* raw hidden is deliberately distinct from the head */
    return 0;
}
int glm53f_target_model_step_batch_12n(glm53f_target_model_12n *m,
        const int *input, int n, int *next, float *logit, float *h,
        glm53f_target_snapshot_12n **after) {
    for (int j = 0; j < n; ++j) {
        float *raw = h ? h + (size_t)j * HIDDEN : NULL;
        next[j] = step(m, input[j], raw); logit[j] = 1;
        hidden(m->head[j], m->position, m->hash);
        if (raw) raw[2] = 0;
        after[j]->position = m->position; after[j]->hash = m->hash;
    }
    return 0;
}
int glm53f_target_model_head_hidden_12n(const glm53f_target_model_12n *m,
        float *h, int tokens) {
    memcpy(h, m->head, (size_t)tokens * HIDDEN * sizeof(float)); return 0;
}
int glm53f_target_decode_sequence_12n(glm53f_target_model_12n *m,
        int first, int n, int *ids, void (*observer)(void *, int), void *context) {
    ids[0] = first; ++m->bulk_calls;
    for (int j = 0; j < n; ++j) {
        ids[j + 1] = step(m, ids[j], NULL);
        if (observer) observer(context, j + 1);
    }
    return 0;
}
int glm53f_mtp_length_12n(const glm53f_mtp_context_12n *c) { return c->length; }
int glm53f_mtp_restore_length_12n(glm53f_mtp_context_12n *c, int length) {
    if (length < 0 || length > c->length) return -1;
    c->length = length; c->draft_index = 0; return 0;
}
int glm53f_mtp_cache_append_12n(glm53f_mtp_context_12n *c, int input, const float *h) {
    if (c->length >= MAX_POSITION || h[2] != 11) return -1;
    c->input[c->length] = input;
    c->parent[c->length][0] = h[0]; c->parent[c->length][1] = h[1];
    c->parent[c->length][2] = h[2];
    ++c->length; return 0;
}
int glm53f_mtp_forward_12n(glm53f_mtp_context_12n *c, int input,
        const float *h, int *draft, float *logit, float *draft_hidden) {
    if (glm53f_mtp_cache_append_12n(c, input, h)) return -1;
    *draft = expected_token[c->length + 1];
    if (c->draft_index++ == c->reject_at) ++*draft;
    *logit = 1;
    /* Even correctly predicted tokens carry approximate draft hiddens. */
    hidden(c->head, c->length + 1,
        (expected_hash[c->length + 1] + 1337u) & 65535u);
    if (draft_hidden) { memcpy(draft_hidden, c->head, sizeof(c->head)); draft_hidden[2] = 0; }
    return 0;
}
int glm53f_mtp_head_hidden_12n(const glm53f_mtp_context_12n *c, float *h) {
    memcpy(h, c->head, sizeof(c->head)); return 0;
}
struct observed { int last, limit, failed; };
static void observe(void *context, int done) {
    struct observed *o = context;
    o->failed |= done < o->last || done > o->limit;
    o->last = done;
}
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int failed = 0, cases = 0;
    const int prompts[] = {1, 4, 47, 128}, lengths[] = {1, 2, 3, 4, 5, 7, 31, 128};
    for (size_t p = 0; p < sizeof(prompts) / sizeof(prompts[0]); ++p)
        for (int depth = 1; depth <= 4; ++depth)
            for (int reject = 0; reject <= depth; ++reject)
                for (int adaptive = 0; adaptive <= 1; ++adaptive)
                    for (size_t n = 0; n < sizeof(lengths) / sizeof(lengths[0]); ++n) {
                        int prompt = prompts[p], transitions = lengths[n];
                        expected_hash[0] = 0;
                        for (int t = 0; t < MAX_POSITION - 1; ++t) {
                            expected_token[t] = t < prompt ? 17 + t * 13 :
                                prediction(t, expected_hash[t]);
                            expected_hash[t + 1] = append(expected_hash[t], expected_token[t]);
                        }
                        glm53f_target_model_12n m = {.position = prompt, .hash = expected_hash[prompt]};
                        glm53f_mtp_context_12n mtp = {.reject_at = reject};
                        float parent[HIDDEN];
                        for (int t = 1; t < prompt; ++t) {
                            hidden(parent, t, expected_hash[t]);
                            glm53f_mtp_cache_append_12n(&mtp, expected_token[t], parent);
                        }
                        hidden(parent, prompt, expected_hash[prompt]);
                        glm53f_mtp_spec_workspace_12n *w = glm53f_mtp_spec_workspace_create_12n(&m);
                        int ids[STEPS + 1];
                        glm53f_mtp_spec_stats_12n stats;
                        struct observed o = {0, transitions, 0};
                        mock_time = 0;
                        int rc = w ? glm53f_mtp_spec_decode_12n(&m, &mtp, w, prompt,
                            parent, expected_token[prompt], transitions, depth,
                            adaptive ? 1e-6 : 0, ids, &stats, observe, &o) : -1;
                        int bad = rc || memcmp(ids, expected_token + prompt,
                            (size_t)(transitions + 1) * sizeof(int)) ||
                            m.position != prompt + transitions ||
                            m.hash != expected_hash[prompt + transitions] || o.failed || o.last != transitions;
                        if (!rc) {
                            bad |= stats.accepted > stats.proposed ||
                                stats.accepted + stats.cycles + stats.fallback_tokens != transitions;
                            if (stats.cache_synchronized) {
                                bad |= mtp.length != m.position - 1;
                                for (int t = 0; t < mtp.length; ++t)
                                    bad |= mtp.input[t] != expected_token[t + 1] ||
                                        mtp.parent[t][0] != (float)(t + 1) ||
                                        mtp.parent[t][1] != (float)expected_hash[t + 1] || mtp.parent[t][2] != 11;
                            } else bad |= mtp.length != 0 || !m.bulk_calls;
                        }
                        if (bad) fprintf(stderr, "MTP_SPEC_FAIL prompt=%d depth=%d reject=%d adaptive=%d transitions=%d rc=%d\n",
                            prompt, depth, reject, adaptive, transitions, rc);
                        failed |= bad; ++cases;
                        glm53f_mtp_spec_workspace_free_12n(w);
                    }
    printf("GLM53F_MTP_SPEC %s cases=%d\n", failed ? "FAIL" : "PASS", cases);
    MPI_Finalize(); return failed ? 1 : 0;
}
