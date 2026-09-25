/* Compare one production sparse layer with retained llama.cpp artifacts while
 * feeding both implementations the exact same normalized hidden vector. */
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "../../common/glm53f_safetensors.h"
#include "glm53f_sparse_12n.h"

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

enum { HIDDEN = 4096 };

static int read_f32(const char *path, float *values, size_t count) {
    FILE *in = fopen(path, "rb");
    if (!in) return -1;
    size_t got = fread(values, sizeof(*values), count, in);
    int extra = fgetc(in);
    int rc = ferror(in) || got != count || extra != EOF;
    fclose(in);
    return rc ? -1 : 0;
}

int main(int argc, char **argv) {
    int rank, ranks, local_failed = 0, failed = 0;
    int layer = argc > 4 ? atoi(argv[4]) : 3;
    double threshold = argc > 5 ? strtod(argv[5], NULL) : 1e-3;
    float *input = NULL, *expected = NULL, *actual = NULL, *attention = NULL;
    float *latent = NULL, *local_heads = NULL;
    glm53f_sparse_context_12n *context = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (argc < 4 || argc > 8 || ranks != 12 || layer < 0 || layer >= 45 ||
        layer % 4 != 3 || threshold <= 0.0) {
        if (!rank)
            fprintf(stderr, "usage: mpiexec -n 12 %s MODEL INPUT.f32 EXPECTED.f32 [layer=3] [threshold=1e-3] [ATTENTION.f32] [KV_LATENT.f32]\n",
                    argv[0]);
        local_failed = 1;
        goto finish;
    }
    input = malloc(HIDDEN * sizeof(*input));
    expected = malloc(HIDDEN * sizeof(*expected));
    actual = malloc(HIDDEN * sizeof(*actual));
    if (!input || !expected || !actual ||
        read_f32(argv[2], input, HIDDEN) ||
        read_f32(argv[3], expected, HIDDEN)) {
        fprintf(stderr, "rank=%d controlled sparse input read failed\n", rank);
        local_failed = 1;
        goto finish;
    }
    context = glm53f_sparse_create_12n(argv[1], layer, 1);
    if (!context || glm53f_sparse_sublayer_12n(context, actual, input)) {
        fprintf(stderr, "rank=%d controlled sparse execution failed layer=%d\n",
                rank, layer);
        local_failed = 1;
        goto finish;
    }

    double err2 = 0.0, ref2 = 0.0;
    float max_abs = 0.0f;
    int finite = 1;
    for (int i = 0; i < HIDDEN; ++i) {
        float d = actual[i] - expected[i];
        if (!isfinite(actual[i]) || !isfinite(expected[i])) finite = 0;
        err2 += (double)d * d;
        ref2 += (double)expected[i] * expected[i];
        if (fabsf(d) > max_abs) max_abs = fabsf(d);
    }
    double rel = ref2 != 0.0 ? sqrt(err2 / ref2) : sqrt(err2);
    int ok = finite && rel <= threshold;
    if (!rank)
        printf("GLM53F_SPARSE_REFERENCE layer=%d count=%d rel_l2=%.9g max_abs=%.9g finite=%s threshold=%.9g %s\n",
               layer, HIDDEN, rel, max_abs, finite ? "YES" : "NO", threshold,
               ok ? "PASS" : "FAIL");
    local_failed |= !ok;

    if (argc >= 7) {
        attention = malloc(16384 * sizeof(*attention));
        if (!attention || read_f32(argv[6], attention, 16384) ||
            glm53f_sparse_output_reference_12n(context, actual, attention)) {
            fprintf(stderr, "rank=%d controlled output projection failed\n", rank);
            local_failed = 1;
        } else {
            err2 = ref2 = 0.0; max_abs = 0.0f; finite = 1;
            for (int i = 0; i < HIDDEN; ++i) {
                float d = actual[i] - expected[i];
                if (!isfinite(actual[i]) || !isfinite(expected[i])) finite = 0;
                err2 += (double)d * d; ref2 += (double)expected[i] * expected[i];
                if (fabsf(d) > max_abs) max_abs = fabsf(d);
            }
            rel = ref2 != 0.0 ? sqrt(err2 / ref2) : sqrt(err2);
            ok = finite && rel <= threshold;
            if (!rank)
                printf("GLM53F_SPARSE_OUTPUT_REFERENCE count=%d rel_l2=%.9g max_abs=%.9g finite=%s threshold=%.9g %s\n",
                       HIDDEN, rel, max_abs, finite ? "YES" : "NO", threshold,
                       ok ? "PASS" : "FAIL");
            local_failed |= !ok;
        }
    }

    if (argc == 8) {
        int h0 = 64 * rank / 12, h1 = 64 * (rank + 1) / 12;
        int local_count = (h1 - h0) * 256;
        latent = malloc(512 * sizeof(*latent));
        local_heads = malloc((size_t)local_count * sizeof(*local_heads));
        if (!latent || !local_heads || read_f32(argv[7], latent, 512) ||
            glm53f_sparse_value_reference_12n(context, local_heads, latent)) {
            fprintf(stderr, "rank=%d controlled value projection failed\n", rank);
            local_failed = 1;
        } else {
            double local_err2 = 0.0, local_ref2 = 0.0;
            float local_max = 0.0f;
            for (int i = 0; i < local_count; ++i) {
                float d = local_heads[i] - attention[h0 * 256 + i];
                local_err2 += (double)d * d;
                local_ref2 += (double)attention[h0 * 256 + i] * attention[h0 * 256 + i];
                if (fabsf(d) > local_max) local_max = fabsf(d);
            }
            double global_err2 = 0.0, global_ref2 = 0.0;
            float global_max = 0.0f;
            MPI_Reduce(&local_err2, &global_err2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&local_ref2, &global_ref2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&local_max, &global_max, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
            if (!rank) {
                rel = sqrt(global_err2 / global_ref2);
                ok = rel <= threshold;
                printf("GLM53F_SPARSE_VALUE_REFERENCE count=16384 rel_l2=%.9g max_abs=%.9g threshold=%.9g %s\n",
                       rel, global_max, threshold, ok ? "PASS" : "FAIL");
                local_failed |= !ok;
            }
        }
    }

finish:
    glm53f_sparse_free_12n(context);
    free(actual);
    free(expected);
    free(input);
    free(attention);
    free(local_heads);
    free(latent);
    MPI_Allreduce(&local_failed, &failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (!rank)
        printf("SENTINEL glm53f_sparse_reference_12n=%s\n",
               failed ? "FAIL" : "PASS");
    MPI_Finalize();
    return failed;
}
