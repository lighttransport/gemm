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

/* Diagnostic only: preserve the original TP12-sized down dot products.
 * The temporary compact rows copy bytes, without requantization. Scatter the
 * twelve virtual partials to the real twelve ranks, then use the same world
 * collective as the reference. Optional MTNI mode also checks a four-rank
 * gather and ordered virtual-rank sum against production TP12 MTNI. This is
 * a bounded diagnostic, not a whole-PP implementation or qualification. */
static int virtual_tp12_down(const glm53f_dist *dist,
        glm53f_dense_ffn_context_12n *a, glm53f_dense_ffn_context_12n *b,
        const float *reference, int tokens, int scalar, int layer, int mtni) {
    const int width = I / 12, count = tokens * H;
    float *parts = a256((size_t)3 * count * sizeof(float));
    float *gathered = a256((size_t)12 * count * sizeof(float));
    float *partial = a256((size_t)count * sizeof(float));
    float *output = a256((size_t)count * sizeof(float));
    uint64_t stage_bits = 0, stage_max;
    if (b) {
        if (b->dtype != GLM53F_NATIVE_Q8_0R || b->in != 3 * width)
            MPI_Abort(MPI_COMM_WORLD, 2);
        size_t rb = glm53f_native_row_size(b->dtype, b->in);
        size_t sub_rb = glm53f_native_row_size(b->dtype, width);
        uint8_t *weight = a256((size_t)H * sub_rb);
        float *input = a256((size_t)tokens * width * sizeof(float));
        const float *activation = scalar ? b->act : b->bact;
        for (int slice = 0; slice < 3; ++slice) {
            /* Q8_0R stores a contiguous byte plane followed by FP32 scales,
             * rather than interleaved36-byte blocks. Slice both planes. */
            for (int row = 0; row < H; ++row) {
                uint8_t *dst = weight + (size_t)row * sub_rb;
                const uint8_t *src = b->d + (size_t)row * rb;
                memcpy(dst, src + slice * width, width);
                memcpy(dst + width, src + b->in +
                    (size_t)slice * (width / 32) * sizeof(float),
                    (size_t)(width / 32) * sizeof(float));
            }
            for (int t = 0; t < tokens; ++t)
                memcpy(input + (size_t)t * width,
                    activation + (size_t)t * b->in + slice * width,
                    (size_t)width * sizeof(float));
            glm53f_native_matrix m = {
                parts + (size_t)slice * count, weight, b->dtype, H, width};
            int rc = scalar ? glm53f_native_matvec_n(&m, 1, input) :
                glm53f_native_matvec_batch(&m, 1, input, tokens);
            if (rc) MPI_Abort(MPI_COMM_WORLD, 2);
        }
        free(input); free(weight);
        MPI_Allgather(parts, 3 * count, MPI_FLOAT, gathered, 3 * count,
            MPI_FLOAT, dist->tp);
        if (mtni) {
            /* Same rank0..11 FP32 additions as production MTNI. */
            for (int i = 0; i < count; i += 16) {
                svbool_t pg = svwhilelt_b32(i, count);
                svfloat32_t acc = svdup_f32(0);
                for (int r = 0; r < 12; ++r) {
                    acc = svadd_f32_x(pg, acc,
                        svld1_f32(pg, gathered + (size_t)r * count + i));
                    /* Retain rank order even under fast-math reassociation. */
                    __asm__ __volatile__("" : "+w"(acc));
                }
                svst1_f32(pg, output + i, acc);
            }
            for (int i = 0; i < count; ++i)
                stage_bits += memcmp(output + i, reference + i, 4) != 0;
        }
    }
    MPI_Allreduce(&stage_bits, &stage_max, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
    MPI_Scatter(gathered, count, MPI_FLOAT, partial, count, MPI_FLOAT,
        0, MPI_COMM_WORLD);
    const float *expected = scalar ? a->part : a->bpart;
    uint64_t bits = 0, maximum;
    for (int i = 0; i < count; ++i) bits += memcmp(partial + i, expected + i, 4) != 0;
    MPI_Allreduce(&bits, &maximum, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
    if (glm53f_sum_allreduce_12n(partial, output, count)) MPI_Abort(MPI_COMM_WORLD, 2);
    uint64_t output_bits = 0;
    for (int i = 0; i < count; ++i) output_bits += memcmp(output + i, reference + i, 4) != 0;
    uint64_t output_max;
    MPI_Allreduce(&output_bits, &output_max, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
    if (!dist->map.world_rank) {
        printf("GLM53F_DENSE_VIRTUAL_TP12 layer=%d tokens=%d scalar=%d partial_bit_mismatches=%" PRIu64 " output_bit_mismatches=%" PRIu64 " %s\n",
            layer, tokens, scalar, maximum, output_max,
            maximum || output_max ? "FAIL" : "PASS");
        if (mtni) printf("GLM53F_DENSE_STAGE4_MTNI layer=%d tokens=%d scalar=%d output_bit_mismatches=%" PRIu64 " %s\n",
            layer, tokens, scalar, stage_max, stage_max ? "FAIL" : "PASS");
        fflush(stdout);
    }
    free(output); free(partial); free(gathered); free(parts);
    return maximum != 0 || output_max != 0 || stage_max != 0;
}

int main(int argc, char **argv) {
    int rank, ranks, provided, failed = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int mtni = argc == 4 && !strcmp(argv[3], "--virtual-tp12-down-mtni");
    int virtual_down = mtni || (argc == 4 && !strcmp(argv[3], "--virtual-tp12-down"));
    if ((argc != 3 && !virtual_down) || ranks != 12 || provided < MPI_THREAD_SERIALIZED)
        MPI_Abort(MPI_COMM_WORLD, 2);
    if (mtni) {
        if (setenv("GLM53F_UTOFU", "1", 1) || setenv("GLM53F_MTNI_DECODE", "1", 1) ||
            glm53f_collective_init_12n(getenv("TOFU_TOPO_PATH"), 4 * H) ||
            !glm53f_collective_is_utofu_12n() || glm53f_collective_prefill_algorithm_12n(5))
            MPI_Abort(MPI_COMM_WORLD, 2);
    }
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
            if (virtual_down) failed |= virtual_tp12_down(&dist, a, b, reference, tokens, test == 0, layer, mtni);
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
    glm53f_dist_free(&dist);
    if (mtni) glm53f_collective_free_12n();
    MPI_Finalize(); return failed ? 1 : 0;
}
