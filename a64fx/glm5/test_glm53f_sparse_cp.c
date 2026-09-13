/* Populated-cache component benchmark, not a model prefill or quality test.
 * Include the implementation to inspect selection and restore only the append
 * cursor between repeated measurements against the same synthetic prefix.
 */
#include <mpi.h>
static double exchange_seconds, candidate_seconds, pool_seconds;
static int timed_allgatherv(const void *send, int count, MPI_Datatype type,
        void *recv, const int *counts, const int *offsets, MPI_Datatype recv_type, MPI_Comm comm) {
    double begin = MPI_Wtime();
    int rc = MPI_Allgatherv(send, count, type, recv, counts, offsets, recv_type, comm);
    exchange_seconds += MPI_Wtime() - begin;
    return rc;
}
static int timed_allgather(const void *send, int count, MPI_Datatype type,
        void *recv, int recv_count, MPI_Datatype recv_type, MPI_Comm comm) {
    double begin = MPI_Wtime();
    int rc = MPI_Allgather(send, count, type, recv, recv_count, recv_type, comm);
    candidate_seconds += MPI_Wtime() - begin;
    return rc;
}
static int timed_allreduce(const void *send, void *recv, int count,
        MPI_Datatype type, MPI_Op op, MPI_Comm comm) {
    double begin = MPI_Wtime();
    int rc = MPI_Allreduce(send, recv, count, type, op, comm);
    if (count > 1024) exchange_seconds += MPI_Wtime() - begin;
    else pool_seconds += MPI_Wtime() - begin;
    return rc;
}
#define MPI_Allgatherv timed_allgatherv
#define MPI_Allgather timed_allgather
#define MPI_Allreduce timed_allreduce
#define GLM53F_SPARSE_NO_MAIN
#include "glm53f_sparse_layer_12n.c"
#undef MPI_Allgatherv
#undef MPI_Allgather
#undef MPI_Allreduce

static unsigned long long hash_bytes(const void *data, size_t bytes) {
    const unsigned char *p = data;
    unsigned long long hash = 14695981039346656037ULL;
    for (size_t i = 0; i < bytes; ++i) { hash ^= p[i]; hash *= 1099511628211ULL; }
    return hash;
}

int main(int argc, char **argv) {
    int rank, ranks, positions = argc > 2 ? atoi(argv[2]) : 524288;
    int repeats = argc > 3 ? atoi(argv[3]) : 8;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (argc < 2 || ranks != 12 || positions < 65536 || positions > 1048576 || repeats < 1)
        MPI_Abort(MPI_COMM_WORLD, 2);
    int bf16 = argc > 4 && !strcmp(argv[4], "bf16");
#ifdef GLM53F_CP_BF16_LATENT
    glm53f_sparse_context_12n *c = glm53f_sparse_create_format_12n(argv[1], 43, positions, bf16);
#else
    if (bf16) MPI_Abort(MPI_COMM_WORLD, 2);
    glm53f_sparse_context_12n *c = glm53f_sparse_create_12n(argv[1], 43, positions);
#endif
    if (!c || !c->cp) MPI_Abort(MPI_COMM_WORLD, 2);
    float x[H], out[H], reference[H];
    int selected[TOPK + KPOOL];
    for (int d = 0; d < H; ++d) x[d] = (d % 29 - 14) * 0.01f;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < c->local_capacity; ++i) {
        int p = i * ranks + rank;
        float latent[LAT];
        for (int d = 0; d < LAT; ++d) latent[d] = ((p * 17LL + d * 31) % 251 - 125) * 0.002f;
#ifdef GLM53F_CP_BF16_LATENT
        cp_store_latent(c, i, latent);
#else
        memcpy(c->cp_latent + (size_t)i * LAT, latent, sizeof(latent));
#endif
        for (int d = 0; d < ID; ++d) {
            c->cp_key[(size_t)i * ID + d] = ((p * 7LL + d * 13) % 127 - 63) * 0.01f;
            c->cp_gate[(size_t)i * ID + d] = ((p * 11LL + d * 5) % 59 - 29) * 0.01f;
        }
    }
#pragma omp parallel for schedule(static)
    for (int i = 0; i <= c->pool_capacity; ++i)
        for (int d = 0; d < ID; ++d)
            c->cp_pool[(size_t)i * ID + d] = (((i * ranks + rank) * 43LL + d * 19) % 257 - 128) * 0.01f;
    double best = 1e30, total = 0, checksum = 0;
    for (int rep = -1; rep < repeats; ++rep) {
        if (!rep) exchange_seconds = candidate_seconds = pool_seconds = 0;
        c->length = positions - 1;
        MPI_Barrier(MPI_COMM_WORLD);
        double begin = MPI_Wtime();
        if (glm53f_sparse_sublayer_12n(c, out, x)) MPI_Abort(MPI_COMM_WORLD, 3);
        double elapsed = MPI_Wtime() - begin, maximum;
        MPI_Allreduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (rep < 0) {
            memcpy(reference, out, sizeof(out));
            memcpy(selected, c->selected, TOPK * sizeof(int));
        } else {
            if (memcmp(reference, out, sizeof(out)) || memcmp(selected, c->selected, TOPK * sizeof(int)))
                MPI_Abort(MPI_COMM_WORLD, 4);
            total += maximum;
            if (maximum < best) best = maximum;
        }
        for (int d = 0; d < H; ++d) {
            if (!isfinite(out[d])) MPI_Abort(MPI_COMM_WORLD, 5);
            checksum += out[d];
        }
    }
    double phases[3] = {exchange_seconds, candidate_seconds, pool_seconds}, maximum_phases[3];
    MPI_Reduce(phases, maximum_phases, 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) printf("CP_COLLECTIVE_MEAN selected_exchange_ms=%.6f candidate_gather_ms=%.6f pool_exchange_ms=%.6f\n",
        maximum_phases[0] * 1e3 / repeats, maximum_phases[1] * 1e3 / repeats, maximum_phases[2] * 1e3 / repeats);
    if (!rank) printf("GLM53F_SYNTHETIC_CP positions=%d repeats=%d latent_bf16=%d cache_GiB_rank=%.6f layer_best_ms=%.6f layer_mean_ms=%.6f checksum=%.12g first_selected=%d output_hash=%016llx selected_hash=%016llx PASS\n",
        positions, repeats, bf16, glm53f_sparse_cache_bytes_12n(c) / 1073741824.0,
        best * 1e3, total * 1e3 / repeats, checksum, selected[0],
        hash_bytes(reference, sizeof(reference)), hash_bytes(selected, TOPK * sizeof(int)));
    if (!rank && argc > 5) {
        FILE *f = fopen(argv[5], "wb");
        if (!f || fwrite(reference, 1, sizeof(reference), f) != sizeof(reference) || fclose(f))
            MPI_Abort(MPI_COMM_WORLD, 6);
    }
    glm53f_sparse_free_12n(c);
    MPI_Finalize();
    return 0;
}
