/* Real-weight diagnostic: identical inputs through TP12 and owned TP4 dense
 * projections. Include the implementation to inspect intermediate buffers;
 * this is an independent fixture, not an inference tuning path. */
#define GLM53F_DENSE_NO_MAIN
#include "glm53f_dense_ffn_12n.c"
#include <inttypes.h>

static int report(const float *reference, const float *actual, int count,
        int layer, int tokens, const char *field) {
    double norm = 0, error = 0;
    uint64_t bits = 0;
    int finite = 1;
    for (int i = 0; i < count; ++i) {
        uint32_t a, b;
        memcpy(&a, reference + i, 4); memcpy(&b, actual + i, 4);
        finite &= (a & 0x7f800000u) != 0x7f800000u &&
                  (b & 0x7f800000u) != 0x7f800000u;
        bits += a != b;
        double delta = (double)actual[i] - reference[i];
        norm += (double)reference[i] * reference[i]; error += delta * delta;
    }
    double relative = norm ? sqrt(error / norm) : 0;
    int pass = finite && (norm ? relative <= 1e-3 : !bits);
    printf("GLM53F_DENSE_CROSS layer=%d tokens=%d field=%s rel_l2=%.17g bit_mismatches=%" PRIu64 " finite=%d %s\n",
        layer, tokens, field, relative, bits, finite, pass ? "PASS" : "FAIL");
    fflush(stdout);
    return !pass;
}

int main(int argc, char **argv) {
    int rank, ranks, provided, failed = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (argc != 3 || ranks != 12 || provided < MPI_THREAD_SERIALIZED)
        MPI_Abort(MPI_COMM_WORLD, 2);
    glm53f_parallel_config config = glm53f_parallel_default();
    config.layout = GLM53F_PP3_TP4;
    glm53f_dist dist;
    if (glm53f_dist_init(&dist, MPI_COMM_WORLD, &config)) MPI_Abort(MPI_COMM_WORLD, 2);
    float *x = a256(4 * H * sizeof(float));
    float *reference = a256(4 * H * sizeof(float));
    float *actual = a256(4 * H * sizeof(float));
    float *wide_reference = a256(4 * I * sizeof(float));
    float *wide_actual = a256(4 * I * sizeof(float));
    for (int i = 0; i < 4 * H; ++i)
        x[i] = (float)(((i * 17 + 3) % 251) - 125) / 125.0f;
    if (setenv("GLM53F_Q2_DENSE_STAGE", argv[1], 1)) MPI_Abort(MPI_COMM_WORLD, 2);
    for (int layer = 0; layer < 3; ++layer) {
        glm53f_dense_ffn_context_12n *a = glm53f_dense_ffn_create_12n(NULL, layer);
        glm53f_dense_ffn_context_12n *b = dist.map.stage == 0 ?
            glm53f_dense_ffn_create_dist(&dist, NULL, argv[2], layer) : NULL;
        if (!a || (dist.map.stage == 0 && !b)) MPI_Abort(MPI_COMM_WORLD, 2);
        if (!rank) printf("GLM53F_DENSE_CROSS_TYPES layer=%d TP12=%d,%d,%d TP4=%d,%d,%d\n",
            layer, a->gtype, a->utype, a->dtype, b->gtype, b->utype, b->dtype);
        for (int test = 0; test < 3; ++test) {
            int tokens = test == 2 ? 4 : 1;
            int rc = test == 0 ? glm53f_dense_ffn_sublayer_12n(a, reference, x) :
                glm53f_dense_ffn_sublayer_batch_12n(a, reference, x, tokens);
            if (rc) MPI_Abort(MPI_COMM_WORLD, 2);
            if (b) {
                rc = test == 0 ? glm53f_dense_ffn_sublayer_12n(b, actual, x) :
                    glm53f_dense_ffn_sublayer_batch_12n(b, actual, x, tokens);
                if (rc) MPI_Abort(MPI_COMM_WORLD, 2);
            }
            const float *av[] = {test ? a->bgv : a->gv, test ? a->buv : a->uv, test ? a->bact : a->act};
            const float *bv[3] = {NULL, NULL, NULL};
            if (b) { bv[0] = test ? b->bgv : b->gv; bv[1] = test ? b->buv : b->uv; bv[2] = test ? b->bact : b->act; }
            const char *names[] = {"gate", "up", "activation"};
            for (int field = 0; field < 3; ++field) {
                for (int t = 0; t < tokens; ++t) {
                    MPI_Gather(av[field] + t * a->in, a->in, MPI_FLOAT,
                        wide_reference + t * I, a->in, MPI_FLOAT, 0, MPI_COMM_WORLD);
                    if (b) MPI_Gather(bv[field] + t * b->in, b->in, MPI_FLOAT,
                        wide_actual + t * I, b->in, MPI_FLOAT, 0, dist.tp);
                }
                if (!rank) failed |= report(wide_reference, wide_actual, tokens * I,
                    layer, test == 0 ? 0 : tokens, names[field]);
            }
            if (!rank) failed |= report(reference, actual, tokens * H,
                layer, test == 0 ? 0 : tokens, "output");
            MPI_Barrier(MPI_COMM_WORLD);
        }
        glm53f_dense_ffn_free_12n(a); glm53f_dense_ffn_free_12n(b);
    }
    MPI_Bcast(&failed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!rank) printf("GLM53F_DENSE_CROSS_LAYOUT_%s\n", failed ? "FAIL" : "PASS");
    free(wide_actual); free(wide_reference); free(actual); free(reference); free(x);
    glm53f_dist_free(&dist); MPI_Finalize(); return failed ? 1 : 0;
}
