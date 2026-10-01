#include "glm53f_clock.h"
#include "glm53f_lookup_spec_12n.h"
#include "glm53f_lookup.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
struct lookup_observer { void (*fn)(void *, int); void *context; int prefix; };
static void observe_fallback(void *context, int completed) {
    struct lookup_observer *a = context;
    if (a->fn) a->fn(a->context, a->prefix + completed);
}

struct glm53f_lookup_workspace_12n {
    int *history, capacity;
    glm53f_target_snapshot_12n *after[5];
};
glm53f_lookup_workspace_12n *glm53f_lookup_workspace_create_12n(
        const glm53f_target_model_12n *model, int capacity) {
    if (!model || capacity < 2 || capacity > 262144 + 32768 + 1) return NULL;
    glm53f_lookup_workspace_12n *w = calloc(1, sizeof(*w));
    if (!w) return NULL;
    w->capacity = capacity;
    w->history = malloc((size_t)capacity * sizeof(int));
    if (!w->history) goto fail;
    for (int t = 0; t < 5; ++t) {
        w->after[t] = glm53f_target_snapshot_create_12n(model);
        if (!w->after[t]) goto fail;
    }
    return w;
fail:
    glm53f_lookup_workspace_free_12n(w);
    return NULL;
}
void glm53f_lookup_workspace_free_12n(glm53f_lookup_workspace_12n *w) {
    if (!w) return;
    for (int t = 0; t < 5; ++t) glm53f_target_snapshot_free_12n(w->after[t]);
    free(w->history); free(w);
}
int glm53f_lookup_decode_12n(glm53f_target_model_12n *m,
        glm53f_lookup_workspace_12n *w, const int *prompt, int prompt_count,
        int first, int transitions, int depth, double plain_seconds_per_token,
        int *ids, glm53f_lookup_stats_12n *stats,
        void (*observer)(void *, int), void *observer_context) {
    if (!m || !w || !prompt || !ids || !stats || prompt_count < 1 || transitions < 1 ||
        depth < 1 || depth > 4 || first < 0 || first >= 154880 ||
        transitions > w->capacity - prompt_count - 1) return -1;
    memset(stats, 0, sizeof(*stats));
    memcpy(w->history, prompt, (size_t)prompt_count * sizeof(int));
    w->history[prompt_count] = ids[0] = first;
    int completed = 0;
    double begin = glm53f_clock();
    int window_completed = 0, window_cycles = 0;
    while (completed < transitions) {
        /* All ranks choose the same policy from maximum elapsed time. A local
         * timing decision could diverge collective order and deadlock. */
        if (stats->cycles - window_cycles >= 16 && plain_seconds_per_token > 0) {
            double local = glm53f_clock() - begin, maximum;
            MPI_Allreduce(&local, &maximum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (maximum / (completed - window_completed) > plain_seconds_per_token * 1.05) {
                double start = glm53f_clock();
                int remaining = transitions - completed;
                /* A single resident team handles the complete fallback suffix.
                 * Observer below receives the global completed count. */
                struct lookup_observer guard = {observer, observer_context, completed};
                if (glm53f_target_decode_sequence_12n(m, ids[completed], remaining,
                        ids + completed, observe_fallback, &guard)) return -1;
                stats->plain_seconds += glm53f_clock() - start;
                stats->fallback_tokens += remaining;
                completed = transitions;
                if (observer) observer(observer_context, completed);
                break;
            }
            /* Recheck later windows: an initially cheap no-hit prefix does
             * not predict the cost once verification starts finding drafts. */
            begin = glm53f_clock();
            window_completed = completed;
            window_cycles = stats->cycles;
        }
        int draft[4], input[5], prediction[5];
        float logits[5];
        double start = glm53f_clock();
        int n = glm53f_lookup_draft(w->history, prompt_count + completed + 1, depth, draft);
        if (n >= transitions - completed) n = transitions - completed - 1;
        stats->lookup_seconds += glm53f_clock() - start;
        ++stats->cycles;
        if (n < 1) {
            start = glm53f_clock();
            if (glm53f_target_decode_sequence_12n(m, ids[completed], 1,
                    ids + completed, NULL, NULL)) return -1;
            stats->plain_seconds += glm53f_clock() - start;
            ++stats->fallback_tokens;
            ++completed;
            w->history[prompt_count + completed] = ids[completed];
        } else {
            input[0] = ids[completed];
            memcpy(input + 1, draft, (size_t)n * sizeof(int));
            start = glm53f_clock();
            if (glm53f_target_model_step_batch_12n(m, input, n + 1,
                    prediction, logits, NULL, w->after)) return -1;
            stats->verify_seconds += glm53f_clock() - start;
            int accepted = 0;
            while (accepted < n && draft[accepted] == prediction[accepted]) ++accepted;
            start = glm53f_clock();
            if (glm53f_target_snapshot_restore_12n(m, w->after[accepted])) return -1;
            stats->rollback_seconds += glm53f_clock() - start;
            stats->accepted += accepted; stats->proposed += n;
            /* Accepted draft positions followed by the exact bonus/correction.
             * State contains the old prediction plus the accepted inputs;
             * the final emitted prediction is consumed in the next cycle. */
            for (int j = 0; j <= accepted; ++j) {
                ++completed;
                ids[completed] = j < accepted ? draft[j] : prediction[accepted];
                w->history[prompt_count + completed] = ids[completed];
            }
        }
        if (observer) observer(observer_context, completed);
    }
    return 0;
}
