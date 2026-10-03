/* Exact scalar/batched MTP cache state, pool rollback, ragged tiles and timings. */
#include "glm53f_mtp_12n.c"
#include "glm53f_state_io.h"
#include "glm53f_collective_12n.h"

static void prime_input(int position, float *hidden) {
    for (int i = 0; i < MTP_H; ++i)
        hidden[i] = (float)((i * 31 + position * 17) % 257 - 128) / 128.0f;
}
static int append(glm53f_mtp_context_12n *m, int count, int tile, int shift) {
    float hidden[64 * MTP_H]; int ids[64];
    for (int base = 0; base < count; base += tile) {
        int n = count - base < tile ? count - base : tile;
        for (int t = 0; t < n; ++t) {
            ids[t] = (base + t + shift) * 7919 % 154880;
            prime_input(base + t + shift, hidden + (size_t)t * MTP_H);
        }
        if (tile == 1) {
            if (glm53f_mtp_cache_append_12n(m, ids[0], hidden)) return -1;
        } else if (glm53f_mtp_cache_append_batch_12n(m, ids, hidden, n)) return -1;
    }
    return 0;
}
int main(int argc, char **argv) {
    int rank, ranks, provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (argc != 5 || ranks != 12 || provided < MPI_THREAD_SERIALIZED) MPI_Abort(MPI_COMM_WORLD, 2);
    if (getenv("GLM53F_UTOFU") && glm53f_collective_init_12n(getenv("TOFU_TOPO_PATH"), 5 * MTP_H)) MPI_Abort(MPI_COMM_WORLD, 2);
    glm53f_mtp_context_12n *m = glm53f_mtp_create_12n(argv[1], argv[2], argv[3], 2064);
    if (!m) MPI_Abort(MPI_COMM_WORLD, 2);
    char path[4096];
    if (snprintf(path, sizeof(path), "%s.rank%02d.state", argv[4], rank) >= (int)sizeof(path)) MPI_Abort(MPI_COMM_WORLD, 2);
    FILE *state = fopen(path, "wbx+"); if (!state) MPI_Abort(MPI_COMM_WORLD, 2);
    const int counts[] = {1, 3, 4, 63, 64, 65, 127, 129, 2051};
    const int tiles[] = {1, 4, 16, 64};
    int ok = 1, cases = 0, expected_id = 0; float expected_logit = 0, expected_hidden[MTP_H];
    for (size_t k = 0; k < sizeof(counts) / sizeof(counts[0]); ++k) {
        for (size_t mode = 0; mode < sizeof(tiles) / sizeof(tiles[0]); ++mode) {
            if (glm53f_mtp_restore_length_12n(m, 0) || append(m, counts[k], tiles[mode], 0)) MPI_Abort(MPI_COMM_WORLD, 2);
            rewind(state);
            glm53f_state_io io = {.file = state, .compare = mode != 0};
            ok &= glm53f_mtp_length_12n(m) == counts[k];
            float first_hidden[MTP_H], first_logit; int first_id;
            prime_input(222, first_hidden);
            if (glm53f_mtp_forward_12n(m, 23, first_hidden, &first_id, &first_logit, NULL) ||
                glm53f_sparse_state_io_12n(m->attention, &io) || fflush(state)) MPI_Abort(MPI_COMM_WORLD, 2);
            if (counts[k] >= 4) {
                if (glm53f_mtp_restore_length_12n(m, counts[k] - 3) || append(m, 6, tiles[mode], 99)) MPI_Abort(MPI_COMM_WORLD, 2);
            }
            float hidden[MTP_H], actual[MTP_H], logit; int id;
            prime_input(333, hidden);
            if (glm53f_mtp_forward_12n(m, 17, hidden, &id, &logit, actual)) MPI_Abort(MPI_COMM_WORLD, 2);
            if (glm53f_sparse_state_io_12n(m->attention, &io) || fflush(state)) MPI_Abort(MPI_COMM_WORLD, 2);
            if (!mode) { expected_id = id; expected_logit = logit; memcpy(expected_hidden, actual, sizeof(actual)); }
            else ok &= id == expected_id && logit == expected_logit && !memcmp(expected_hidden, actual, sizeof(actual));
            ++cases;
        }
    }
    float hidden[MTP_H] = {0}; int id = 1;
    ok &= glm53f_mtp_cache_append_batch_12n(m, &id, hidden, 0) == -1;
    ok &= glm53f_mtp_cache_append_batch_12n(m, &id, hidden, 65) == -1;
    ok &= glm53f_mtp_cache_append_batch_12n(NULL, &id, hidden, 1) == -1;
    if (append(m, 2064 - glm53f_mtp_length_12n(m), 64, 0)) MPI_Abort(MPI_COMM_WORLD, 2);
    ok &= glm53f_mtp_cache_append_batch_12n(m, &id, hidden, 1) == -1;
    int all_ok; MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!all_ok) MPI_Abort(MPI_COMM_WORLD, 2);
    for (int pair = 0; pair < 5; ++pair) {
        double times[2] = {0, 0}, maximum[2];
        for (int turn = 0; turn < 2; ++turn) {
            int mode = (turn + pair) % 2;
            if (glm53f_mtp_restore_length_12n(m, 0)) MPI_Abort(MPI_COMM_WORLD, 2);
            MPI_Barrier(MPI_COMM_WORLD); double start = MPI_Wtime();
            if (append(m, 2051, mode ? 64 : 1, 0)) MPI_Abort(MPI_COMM_WORLD, 2);
            times[mode] = MPI_Wtime() - start;
        }
        MPI_Reduce(times, maximum, 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (!rank) printf("GLM53F_MTP_PRIME_TIMING pair=%d positions=2051 scalar_s=%.9f batch64_s=%.9f speedup=%.9f\n", pair, maximum[0], maximum[1], maximum[0] / maximum[1]);
    }
    fclose(state); unlink(path);
    if (!rank) printf("GLM53F_MTP_PRIME_PASS cases=%d state=BIT_EXACT rollback=BIT_EXACT\n", cases);
    glm53f_mtp_free_12n(m); glm53f_collective_free_12n(); MPI_Finalize(); return 0;
}
