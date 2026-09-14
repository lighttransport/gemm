/* Real-weight causal/cache test; two sparse layers fit well below model HBM. */
#define GLM53F_SPARSE_NO_MAIN
#include "glm53f_sparse_layer_12n.c"

int main(int argc, char **argv) {
    int rank, ranks, positions = argc > 2 ? atoi(argv[2]) : 2084;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (argc < 2 || ranks != 12 || positions < 1 || positions > 8192) MPI_Abort(MPI_COMM_WORLD, 2);
    setenv("GLM53F_SPARSE_CP", "0", 1);
    if (getenv("GLM53F_UTOFU") && glm53f_collective_init_12n(getenv("TOFU_TOPO_PATH"), 4 * H))
        MPI_Abort(MPI_COMM_WORLD, 2);
    glm53f_sparse_context_12n *a = glm53f_sparse_create_12n(argv[1], 3, positions);
    glm53f_sparse_context_12n *b = glm53f_sparse_create_12n(argv[1], 3, positions);
    glm53f_sparse_prefill_workspace_12n *work = glm53f_sparse_prefill_workspace_create_12n();
    float *x = a256((size_t)32 * H * sizeof(float));
    float *ref = a256((size_t)32 * H * sizeof(float)), *out = a256((size_t)32 * H * sizeof(float));
    int selected[32][TOPK + KPOOL], failed = !a || !b || !work;
    if (failed) MPI_Abort(MPI_COMM_WORLD, 2);
    double seq_time = 0, batch_time = 0;
    for (int base = 0; base < positions && !failed; base += 32) {
        int n = positions - base;
        if (n > 32) n = 32;
        for (int t = 0; t < n; ++t)
            for (int d = 0; d < H; ++d)
                x[(size_t)t * H + d] = ((d * 17 + (base + t) * 31 + 5) % 251 - 125) / 125.0f;
        double begin = MPI_Wtime();
        for (int t = 0; t < n; ++t) {
            failed |= glm53f_sparse_sublayer_12n(a, ref + (size_t)t * H, x + (size_t)t * H) != 0;
            int length = base + t + 1, np = length / KPOOL;
            if (np > TOPK / KPOOL) np = TOPK / KPOOL;
            memcpy(selected[t], a->selected, (size_t)(np * KPOOL + length % KPOOL) * sizeof(int));
        }
        seq_time += MPI_Wtime() - begin;
        begin = MPI_Wtime();
        failed |= glm53f_sparse_prefill_12n(b, work, out, x, n) != 0;
        batch_time += MPI_Wtime() - begin;
        failed |= memcmp(ref, out, (size_t)n * H * sizeof(float)) != 0;
        for (int t = 0; t < n; ++t)
            failed |= memcmp(selected[t], work->selected[t], (size_t)work->count[t] * sizeof(int)) != 0;
        failed |= a->length != b->length;
        failed |= memcmp(a->latent, b->latent, (size_t)a->length * LAT * sizeof(float)) != 0;
        failed |= memcmp(a->key, b->key, (size_t)a->length * ID * sizeof(float)) != 0;
        failed |= memcmp(a->gcache, b->gcache, (size_t)a->length * ID * sizeof(float)) != 0;
        failed |= memcmp(a->pool, b->pool, (size_t)(a->length / KPOOL) * ID * sizeof(float)) != 0;
        int any;
        MPI_Allreduce(&failed, &any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (any) {
            fprintf(stderr, "SPARSE_PREFILL_FAIL rank=%d end=%d local=%d\n", rank, base + n, failed);
            failed = 1; break;
        }
    }
    if (!failed && positions > 3) {
        int cut = positions - 3;
        failed |= glm53f_sparse_restore_length_12n(a, cut) != 0;
        failed |= glm53f_sparse_restore_length_12n(b, cut) != 0;
        for (int t = 0; t < 3; ++t)
            for (int d = 0; d < H; ++d)
                x[(size_t)t * H + d] = ((d * 17 + (cut + t) * 31 + 23) % 251 - 125) / 125.0f;
        for (int t = 0; t < 3; ++t)
            failed |= glm53f_sparse_sublayer_12n(a, ref + (size_t)t * H, x + (size_t)t * H) != 0;
        failed |= glm53f_sparse_prefill_12n(b, work, out, x, 3) != 0;
        failed |= memcmp(ref, out, (size_t)3 * H * sizeof(float)) != 0;
        failed |= memcmp(a->latent, b->latent, (size_t)positions * LAT * sizeof(float)) != 0;
        failed |= memcmp(a->key, b->key, (size_t)positions * ID * sizeof(float)) != 0;
        failed |= memcmp(a->gcache, b->gcache, (size_t)positions * ID * sizeof(float)) != 0;
        failed |= memcmp(a->pool, b->pool, (size_t)(positions / KPOOL) * ID * sizeof(float)) != 0;
        failed |= memcmp(a->selected, b->selected, (size_t)work->count[2] * sizeof(int)) != 0;
    }
    int rollback_failed;
    MPI_Allreduce(&failed, &rollback_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    failed = rollback_failed;
    if (!failed && positions >= 7) {
        /* The optional converted-attention path must fall back without
         * dereferencing its freed FP8 matrices. It is not the tuned FP8
         * attention / INT8-expert configuration measured by prefill. */
        failed |= glm53f_sparse_convert_int8_12n(a) != 0;
        failed |= glm53f_sparse_convert_int8_12n(b) != 0;
        failed |= glm53f_sparse_restore_length_12n(a, 0) != 0;
        failed |= glm53f_sparse_restore_length_12n(b, 0) != 0;
        for (int t = 0; t < 7; ++t) {
            for (int d = 0; d < H; ++d)
                x[(size_t)t * H + d] = ((d * 17 + t * 31 + 23) % 251 - 125) / 125.0f;
            failed |= glm53f_sparse_sublayer_12n(a, ref + (size_t)t * H, x + (size_t)t * H) != 0;
        }
        failed |= glm53f_sparse_prefill_12n(b, work, out, x, 7) != 0;
        failed |= memcmp(ref, out, (size_t)7 * H * sizeof(float)) != 0;
        failed |= memcmp(a->latent, b->latent, (size_t)7 * LAT * sizeof(float)) != 0;
        failed |= memcmp(a->key, b->key, (size_t)7 * ID * sizeof(float)) != 0;
        failed |= memcmp(a->gcache, b->gcache, (size_t)7 * ID * sizeof(float)) != 0;
        failed |= memcmp(a->pool, b->pool, (size_t)ID * sizeof(float)) != 0;
        failed |= memcmp(a->selected, b->selected, (size_t)7 * sizeof(int)) != 0;
    }
    int global_failed;
    MPI_Allreduce(&failed, &global_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    failed = global_failed;
    if (!rank) printf("SPARSE_PREFILL positions=%d outputs_cache_selection_rollback_int8_fallback_exact=%d "
        "scalar_ms=%.3f batch_ms=%.3f speedup=%.3f\n", positions, !failed,
        seq_time * 1e3, batch_time * 1e3, seq_time / batch_time);
    free(out); free(ref); free(x);
    glm53f_sparse_prefill_workspace_free_12n(work);
    glm53f_sparse_free_12n(b); glm53f_sparse_free_12n(a);
    glm53f_collective_free_12n(); MPI_Finalize();
    return failed;
}
