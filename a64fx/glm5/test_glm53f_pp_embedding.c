/* Packed-owner broadcasts must preserve every FP32 bit and token order. */
#include "glm53f_embedding_12n.c"
#include <assert.h>
int main(int argc, char **argv) {
    int provided, rank;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    assert(provided >= MPI_THREAD_SERIALIZED);
    glm53f_parallel_config config = glm53f_parallel_default(); config.layout = GLM53F_PP3_TP4;
    glm53f_dist d; assert(!glm53f_dist_init(&d, MPI_COMM_WORLD, &config));
    int cases = 0;
    if (!d.map.stage) {
        glm53f_embedding_context_12n c = {0};
        c.dist = &d; c.rank = d.map.tp_rank; c.ranks = 4;
        c.row0 = c.rank * 38720; c.rows = 38720;
        c.q2_weight = calloc((size_t)c.rows * 4096, sizeof(float)); assert(c.q2_weight);
        const int lengths[] = {1, 4, 17, 127, 512, 1024, 2048, 2049};
        for (int pass = 0; pass < 8; ++pass) {
            int n = lengths[pass], *ids = malloc((size_t)n * sizeof(int));
            float *scalar = malloc((size_t)n * 16384 * sizeof(float));
            float *packed = malloc((size_t)n * 16384 * sizeof(float));
            assert(ids && scalar && packed);
            for (int t = 0; t < n; ++t) {
                int owner = (t * 3 + pass) % 4;
                ids[t] = owner * 38720 + (t % 3 == 0 ? 38719 : t % 31);
                if (owner == c.rank) for (int j = 0; j < 4096; ++j) {
                    uint32_t bits = j % 5 == 0 ? UINT32_C(0x80000000) :
                        UINT32_C(0x3f000000) + (uint32_t)((ids[t] * 4096u + j) & 0x7fffff);
                    memcpy(c.q2_weight + (size_t)(ids[t] - c.row0) * 4096 + j, &bits, 4);
                }
            }
            for (int t = 0; t < n; ++t) assert(!glm53f_embedding_streams_12n(&c, ids[t], scalar + (size_t)t * 16384));
            assert(!glm53f_embedding_streams_batch_12n(&c, ids, n, packed));
            assert(!memcmp(scalar, packed, (size_t)n * 16384 * sizeof(float)));
            free(ids); free(scalar); free(packed); ++cases;
        }
        int bad = 154880; float scratch[16384];
        assert(glm53f_embedding_streams_batch_12n(&c, &bad, 1, scratch) == -1);
        free(c.q2_weight);
    }
    MPI_Barrier(d.world);
    if (!rank) printf("GLM53F_PP_EMBEDDING_PASS cases=%d owners=4 signed_zero=1 token_order=exact invalid_rejected=1\n", cases);
    glm53f_dist_free(&d); MPI_Finalize(); return 0;
}
