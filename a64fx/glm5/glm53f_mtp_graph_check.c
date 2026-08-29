/* Real-checkpoint GLM-5.3F MTP fusion and row-sharded vocabulary-head check. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "../../common/glm53f_safetensors.h"
#include "../../common/glm53f_ref.h"

#include <mpi.h>
#include <omp.h>
#include <arm_sve.h>
#include "glm53f_expert_kern.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

enum { HIDDEN = 4096, VOCAB = 154880 };

static void *xmalloc(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 256, n)) p = NULL;
    if (!p) { fprintf(stderr, "allocation failed: %zu bytes\n", n); MPI_Abort(MPI_COMM_WORLD, 2); }
    return p;
}

static void rmsnorm(float *out, const float *x, const uint16_t *w) {
    double ss = 0.0;
#pragma omp parallel for reduction(+:ss)
    for (int i = 0; i < HIDDEN; ++i) ss += (double)x[i] * x[i];
    float inv = 1.0f / sqrtf((float)(ss / HIDDEN) + 1e-5f);
#pragma omp parallel for
    for (int i = 0; i < HIDDEN; ++i)
        out[i] = x[i] * inv * glm53f_bf16_to_f32(w[i]);
}

static void matvec(float *out, const uint16_t *w, const float *x, int rows, int cols) {
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r)
        out[r] = glm53f_dot_bf16_sve(w + (size_t)r * cols, x, cols);
}

int main(int argc, char **argv) {
    int rank, ranks, token = argc > 2 ? atoi(argv[2]) : 1;
    int r0, r1, rows, e0, e1, erows, local_id[2], global_id[2];
    float local_logit[2], elapsed[2], phase[2][3];
    struct { float value; int index; } in, best;
    glm53f_st_context *st;
    uint16_t *embedding_b, *enorm, *hnorm, *head_norm, *eh, *head;
    float *embedding, *hidden, *fusion, *scratch, *normalized, *logits;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (argc < 2 || token < 0 || token >= VOCAB) {
        if (!rank) fprintf(stderr, "usage: %s MODEL_DIR [token]\n", argv[0]);
        MPI_Finalize(); return 2;
    }
    r0 = (int)((long long)VOCAB * rank / ranks);
    r1 = (int)((long long)VOCAB * (rank + 1) / ranks);
    rows = r1 - r0;
    e0 = (int)((long long)HIDDEN * rank / ranks);
    e1 = (int)((long long)HIDDEN * (rank + 1) / ranks);
    erows = e1 - e0;
    st = glm53f_st_open(argv[1]);
    if (!st) MPI_Abort(MPI_COMM_WORLD, 2);
    embedding_b = xmalloc(HIDDEN * sizeof(*embedding_b));
    enorm = xmalloc(HIDDEN * sizeof(*enorm));
    hnorm = xmalloc(HIDDEN * sizeof(*hnorm));
    head_norm = xmalloc(HIDDEN * sizeof(*head_norm));
    eh = xmalloc((size_t)erows * 2 * HIDDEN * sizeof(*eh));
    head = xmalloc((size_t)rows * HIDDEN * sizeof(*head));
    embedding = xmalloc(HIDDEN * sizeof(*embedding));
    hidden = xmalloc(HIDDEN * sizeof(*hidden));
    fusion = xmalloc(HIDDEN * sizeof(*fusion));
    scratch = xmalloc(2 * HIDDEN * sizeof(*scratch));
    normalized = xmalloc(HIDDEN * sizeof(*normalized));
    logits = xmalloc((size_t)rows * sizeof(*logits));
#define READ(N,O,P,Z) do { if (glm53f_st_read(st, (N), (O), (P), (Z))) { \
    fprintf(stderr, "rank=%d read failed: %s\n", rank, (N)); MPI_Abort(MPI_COMM_WORLD, 2); } } while (0)
    READ("model.language_model.embed_tokens.weight", (size_t)token * HIDDEN * 2,
         embedding_b, HIDDEN * 2);
    READ("model.language_model.layers.45.enorm.weight", 0, enorm, HIDDEN * 2);
    READ("model.language_model.layers.45.hnorm.weight", 0, hnorm, HIDDEN * 2);
    READ("model.language_model.layers.45.shared_head.norm.weight", 0, head_norm, HIDDEN * 2);
    READ("model.language_model.layers.45.eh_proj.weight",
         (size_t)e0 * 2 * HIDDEN * 2, eh,
         (size_t)erows * 2 * HIDDEN * 2);
    READ("lm_head.weight", (size_t)r0 * HIDDEN * 2, head,
         (size_t)rows * HIDDEN * 2);
#undef READ
    glm53f_st_close(st);
    for (int i = 0; i < HIDDEN; ++i) {
        embedding[i] = glm53f_bf16_to_f32(embedding_b[i]);
        hidden[i] = (float)((i * 17 + 3) % 251 - 125) / 125.0f;
    }
    for (int pass = 0; pass < 2; ++pass) {
        int counts[ranks], displs[ranks];
        for (int r = 0; r < ranks; ++r) {
            displs[r] = (int)((long long)HIDDEN * r / ranks);
            counts[r] = (int)((long long)HIDDEN * (r + 1) / ranks) - displs[r];
        }
        double t0 = MPI_Wtime();
        rmsnorm(scratch, embedding, enorm);
        rmsnorm(scratch + HIDDEN, hidden, hnorm);
        matvec(fusion + e0, eh, scratch, erows, 2 * HIDDEN);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, fusion,
                       counts, displs, MPI_FLOAT, MPI_COMM_WORLD);
        double t1 = MPI_Wtime();
        rmsnorm(normalized, fusion, head_norm);
        matvec(logits, head, normalized, rows, HIDDEN);
        in.value = -INFINITY; in.index = -1;
        for (int r = 0; r < rows; ++r) {
            int id = r0 + r;
            if (logits[r] > in.value || (logits[r] == in.value && id < in.index)) {
                in.value = logits[r]; in.index = id;
            }
        }
        double t2 = MPI_Wtime();
        MPI_Allreduce(&in, &best, 1, MPI_FLOAT_INT, MPI_MAXLOC, MPI_COMM_WORLD);
        double t3 = MPI_Wtime();
        local_id[pass] = in.index; local_logit[pass] = in.value;
        global_id[pass] = best.index;
        elapsed[pass] = (float)(t3 - t0);
        phase[pass][0] = (float)(t1 - t0);
        phase[pass][1] = (float)(t2 - t1);
        phase[pass][2] = (float)(t3 - t2);
    }
    int stable = global_id[0] == global_id[1] &&
                 local_id[0] == local_id[1] && local_logit[0] == local_logit[1];
    int all_stable = 0;
    float max_sec, max_phase[3];
    MPI_Allreduce(&stable, &all_stable, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&elapsed[1], &max_sec, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(phase[1], max_phase, 3, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    if (!rank) {
        printf("GLM53F_MTP_GRAPH token=%d global_argmax=%d repeat=%s max_sec=%.6f fusion_ms=%.3f head_ms=%.3f reduce_ms=%.3f ranks=%d %s\n",
               token, global_id[0], all_stable ? "BIT_EXACT" : "FAIL",
               max_sec, max_phase[0]*1e3f, max_phase[1]*1e3f,
               max_phase[2]*1e3f, ranks, all_stable ? "PASS" : "FAIL");
        const char *status_path = getenv("GLM53F_MTP_GRAPH_STATUS");
        if (status_path && *status_path) {
            FILE *status = fopen(status_path, "w");
            if (status) {
                fprintf(status, "GLM53F_MTP_GRAPH token=%d global_argmax=%d repeat=%s "
                        "max_sec=%.6f fusion_ms=%.3f head_ms=%.3f reduce_ms=%.3f ranks=%d %s\n", token, global_id[0],
                        all_stable ? "BIT_EXACT" : "FAIL", max_sec,
                        max_phase[0]*1e3f, max_phase[1]*1e3f,
                        max_phase[2]*1e3f, ranks,
                        all_stable ? "PASS" : "FAIL");
                fclose(status);
            }
        }
    }
    free(logits); free(normalized); free(scratch); free(fusion); free(hidden);
    free(embedding); free(head); free(eh); free(head_norm); free(hnorm); free(enorm);
    free(embedding_b);
    MPI_Finalize();
    return all_stable ? 0 : 1;
}
