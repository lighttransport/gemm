/* Exercise the real lookup controller with a context-sensitive toy target.
 * Wrong draft inputs poison subsequent predictions until a snapshot restore.
 * This tests accounting/rollback, not the numerical full-model verifier. */
#include "glm53f_lookup_spec_12n.h"
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { STEPS = 128 };
struct glm53f_target_model_12n { int position, bulk_calls; uint64_t hash; };
struct glm53f_target_snapshot_12n { int position; uint64_t hash; };
static int reference[STEPS + 1];
static uint64_t expected_hash[STEPS + 1];
static uint64_t append(uint64_t hash, int token) { return hash * UINT64_C(1000003) + (unsigned)token + 1; }
glm53f_target_snapshot_12n *glm53f_target_snapshot_create_12n(const glm53f_target_model_12n *m) {
    return m ? calloc(1, sizeof(glm53f_target_snapshot_12n)) : NULL;
}
void glm53f_target_snapshot_free_12n(glm53f_target_snapshot_12n *s) { free(s); }
int glm53f_target_snapshot_restore_12n(glm53f_target_model_12n *m, const glm53f_target_snapshot_12n *s) {
    m->position = s->position; m->hash = s->hash; return 0;
}
static int step(glm53f_target_model_12n *m, int input) {
    m->hash = append(m->hash, input);
    ++m->position;
    return m->position <= STEPS && m->hash == expected_hash[m->position] ?
        reference[m->position] : 2000 + m->position;
}
int glm53f_target_model_step_batch_12n(glm53f_target_model_12n *m,
        const int *input, int n, int *tokens, float *logits, float *hidden,
        glm53f_target_snapshot_12n **after) {
    (void)hidden;
    for (int j = 0; j < n; ++j) {
        tokens[j] = step(m, input[j]); logits[j] = 1;
        after[j]->position = m->position; after[j]->hash = m->hash;
    }
    return 0;
}
int glm53f_target_decode_sequence_12n(glm53f_target_model_12n *m,
        int first, int n, int *ids, void (*observer)(void *, int), void *context) {
    ids[0] = first;
    m->bulk_calls += n > 1;
    for (int j = 0; j < n; ++j) {
        ids[j + 1] = step(m, ids[j]);
        if (observer) observer(context, j + 1);
    }
    return 0;
}
struct observer_state { int last, failed; };
static void observe(void *context, int done) {
    struct observer_state *s = context;
    s->failed |= done < s->last || done > STEPS;
    s->last = done;
}
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    const int prompt[] = {10, 11, 12, 13, 14, 15, 16, 99, 10, 11};
    int failed = 0, cases = 0;
    for (int depth = 1; depth <= 4; ++depth) {
        for (int reject = 0; reject <= depth; ++reject) {
            for (int adaptive = 0; adaptive <= 1; ++adaptive) {
                reference[0] = 12;
                for (int j = 1; j <= STEPS; ++j) reference[j] = j <= reject ? 12 + j : 100 + j;
                expected_hash[0] = 0;
                for (int j = 1; j <= STEPS; ++j)
                    expected_hash[j] = append(expected_hash[j - 1], reference[j - 1]);
                glm53f_target_model_12n m = {0};
                glm53f_lookup_workspace_12n *w = glm53f_lookup_workspace_create_12n(&m, STEPS + 11);
                int ids[STEPS + 1];
                glm53f_lookup_stats_12n stats;
                struct observer_state observer = {0};
                int rc = w ? glm53f_lookup_decode_12n(&m, w, prompt, 10, 12,
                    STEPS, depth, adaptive ? 1e-12 : 0, ids, &stats, observe, &observer) : -1;
                int bad = rc || memcmp(ids, reference, sizeof(ids)) || m.position != STEPS ||
                    m.hash != expected_hash[STEPS] || observer.failed || observer.last != STEPS;
                if (!rc) bad |= stats.accepted != reject || stats.proposed != depth ||
                    (adaptive && !m.bulk_calls) || (!adaptive && m.bulk_calls);
                if (bad) fprintf(stderr, "LOOKUP_SPEC_FAIL depth=%d reject=%d adaptive=%d rc=%d position=%d\n",
                    depth, reject, adaptive, rc, m.position);
                failed |= bad; ++cases;
                glm53f_lookup_workspace_free_12n(w);
            }
        }
    }
    printf("GLM53F_LOOKUP_SPEC %s cases=%d\n", failed ? "FAIL" : "PASS", cases);
    MPI_Finalize();
    return failed ? 1 : 0;
}
