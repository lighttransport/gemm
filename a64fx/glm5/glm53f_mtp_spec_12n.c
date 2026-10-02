#include "glm53f_clock.h"
#include "glm53f_mtp_spec_12n.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

enum { HIDDEN = 4096, MAX_DRAFT = 4 };
struct glm53f_mtp_spec_workspace_12n {
    glm53f_target_snapshot_12n *after[MAX_DRAFT + 1];
    float parent[HIDDEN], verified[MAX_DRAFT + 1][HIDDEN];
    float draft[2][HIDDEN];
};
struct mtp_spec_observer { void (*fn)(void *, int); void *context; int prefix; };
static void observe_plain(void *context, int completed) {
    struct mtp_spec_observer *a = context;
    if (a->fn) a->fn(a->context, a->prefix + completed);
}

glm53f_mtp_spec_workspace_12n *glm53f_mtp_spec_workspace_create_12n(
        const glm53f_target_model_12n *m) {
    if (!m) return NULL;
    glm53f_mtp_spec_workspace_12n *w = calloc(1, sizeof(*w));
    if (!w) return NULL;
    for (int t = 0; t <= MAX_DRAFT; ++t) {
        w->after[t] = glm53f_target_snapshot_create_12n(m);
        if (!w->after[t]) {
            glm53f_mtp_spec_workspace_free_12n(w);
            return NULL;
        }
    }
    return w;
}
void glm53f_mtp_spec_workspace_free_12n(glm53f_mtp_spec_workspace_12n *w) {
    if (!w) return;
    for (int t = 0; t <= MAX_DRAFT; ++t)
        glm53f_target_snapshot_free_12n(w->after[t]);
    free(w);
}

int glm53f_mtp_spec_decode_12n(glm53f_target_model_12n *m,
        glm53f_mtp_context_12n *mtp, glm53f_mtp_spec_workspace_12n *w,
        int prompt_count, const float *parent_hidden, int first, int transitions,
        int depth, double plain_seconds_per_token, int *ids,
        glm53f_mtp_spec_stats_12n *stats,
        void (*observer)(void *, int), void *observer_context) {
    if (!m || !mtp || !w || !parent_hidden || !ids || !stats || prompt_count < 1 ||
        first < 0 || first >= 154880 || transitions < 1 || transitions > 32768 ||
        depth < 1 || depth > MAX_DRAFT ||
        glm53f_mtp_length_12n(mtp) != prompt_count - 1) return -1;
    memset(stats, 0, sizeof(*stats));
    stats->cache_synchronized = 1;
    memcpy(w->parent, parent_hidden, sizeof(w->parent));
    ids[0] = first;
    int completed = 0, window_completed = 0, window_cycles = 0;
    double begin = glm53f_clock();
    while (completed < transitions) {
        if (stats->cycles - window_cycles >= 16 && plain_seconds_per_token > 0) {
            double local = glm53f_clock() - begin, maximum;
            if (MPI_Allreduce(&local, &maximum, 1, MPI_DOUBLE, MPI_MAX,
                    MPI_COMM_WORLD) != MPI_SUCCESS) return -1;
            if (maximum / (completed - window_completed) > plain_seconds_per_token * 1.05) {
                double start = glm53f_clock();
                int remaining = transitions - completed;
                struct mtp_spec_observer guard = {observer, observer_context, completed};
                if (glm53f_target_decode_sequence_12n(m, ids[completed], remaining,
                        ids + completed, observe_plain, &guard)) return -1;
                stats->plain_seconds += glm53f_clock() - start;
                stats->fallback_tokens += remaining;
                /* The persistent suffix deliberately avoids MTP replay work.
                 * Clear stale draft state rather than presenting it as primed. */
                if (glm53f_mtp_restore_length_12n(mtp, 0)) return -1;
                stats->cache_synchronized = 0;
                completed = transitions;
                break;
            }
            begin = glm53f_clock();
            window_completed = completed;
            window_cycles = stats->cycles;
        }
        int n = transitions - completed - 1;
        if (n > depth) n = depth;
        if (!n) {
            double start = glm53f_clock();
            if (glm53f_mtp_cache_append_12n(mtp, ids[completed], w->parent)) return -1;
            stats->replay_seconds += glm53f_clock() - start;
            float logit;
            start = glm53f_clock();
            if (glm53f_target_model_step_12n(m, ids[completed], ids + completed + 1,
                    &logit, NULL) || glm53f_target_model_head_hidden_12n(m, w->parent, 1)) return -1;
            stats->plain_seconds += glm53f_clock() - start;
            ++stats->fallback_tokens;
            ++completed;
        } else {
            int draft[MAX_DRAFT], input[MAX_DRAFT + 1], prediction[MAX_DRAFT + 1];
            float logits[MAX_DRAFT + 1], ignored_logit;
            int base = glm53f_mtp_length_12n(mtp), token = ids[completed];
            const float *hidden = w->parent;
            double start = glm53f_clock();
            for (int j = 0; j < n; ++j) {
                float *next_hidden = w->draft[j & 1];
                if (glm53f_mtp_forward_12n(mtp, token, hidden, draft + j,
                        &ignored_logit, NULL)) return -1;
                if (j + 1 < n && glm53f_mtp_head_hidden_12n(mtp, next_hidden)) return -1;
                token = draft[j]; hidden = next_hidden;
            }
            stats->draft_seconds += glm53f_clock() - start;
            input[0] = ids[completed];
            memcpy(input + 1, draft, (size_t)n * sizeof(int));
            start = glm53f_clock();
            if (glm53f_target_model_step_batch_12n(m, input, n + 1, prediction,
                    logits, NULL, w->after) ||
                glm53f_target_model_head_hidden_12n(m, w->verified[0], n + 1)) return -1;
            int accepted = 0;
            while (accepted < n && draft[accepted] == prediction[accepted]) ++accepted;
            if (glm53f_target_snapshot_restore_12n(m, w->after[accepted])) return -1;
            stats->verify_seconds += glm53f_clock() - start;
            start = glm53f_clock();
            /* The first draft pair used the exact parent hidden. Retain it,
             * replay only committed draft inputs with actual target hiddens,
             * and leave the emitted bonus input for the next window. */
            if (glm53f_mtp_restore_length_12n(mtp, base + 1)) return -1;
            for (int j = 0; j < accepted; ++j)
                if (glm53f_mtp_cache_append_12n(mtp, draft[j], w->verified[j])) return -1;
            memcpy(w->parent, w->verified[accepted], sizeof(w->parent));
            stats->replay_seconds += glm53f_clock() - start;
            stats->accepted += accepted; stats->proposed += n; ++stats->cycles;
            for (int j = 0; j <= accepted; ++j)
                ids[++completed] = j < accepted ? draft[j] : prediction[accepted];
        }
        if (observer) observer(observer_context, completed);
    }
    return 0;
}
