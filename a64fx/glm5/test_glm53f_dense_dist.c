#define _POSIX_C_SOURCE 200809L
#include "glm53f_dense_ffn_12n.h"
#include "glm53f_pp_manifest.h"
#include <assert.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <inttypes.h>
static uint64_t zeros(int fd, size_t n) {
    unsigned char buffer[65536] = {0};
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t offset = 0; offset < n;) {
        size_t count = n - offset; if (count > sizeof(buffer)) count = sizeof(buffer);
        assert(write(fd, buffer, count) == (ssize_t)count);
        for (size_t i = 0; i < count; ++i) hash *= UINT64_C(1099511628211);
        offset += count;
    }
    return hash;
}
int main(int argc, char **argv) {
    int provided, rank;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); assert(argc == 2 && provided >= MPI_THREAD_SERIALIZED);
    const int cuts[3][2] = {{15, 30}, {1, 3}, {1, 2}};
    for (int test = 0; test < 3; ++test) {
    glm53f_parallel_config config = glm53f_parallel_default(); config.layout = GLM53F_PP3_TP4;
    memcpy(config.cuts, cuts[test], sizeof(config.cuts));
    glm53f_dist d; assert(!glm53f_dist_init(&d, MPI_COMM_WORLD, &config));
    assert(!glm53f_dense_ffn_create_dist(&d, NULL, NULL, 0));
    const int layer = d.map.first_layer;
    if (layer < 3) {
        assert(!mkdir(argv[1], 0700) || access(argv[1], W_OK) == 0);
        char manifest[4096], blob[4096];
        snprintf(manifest, sizeof(manifest), "%s/rank%02d.manifest", argv[1], rank);
        snprintf(blob, sizeof(blob), "%s/rank%02d.blob", argv[1], rank);
        FILE *f = fopen(manifest, "wx"); assert(f);
        fputs("# GLM53F_Q2_DENSE_V2 rank=0 ranks=12 layers=0:3\n", f); assert(!fclose(f));
        assert(glm53f_pp_manifest_check(manifest, "DENSE", &d, d.map.first_layer, d.map.end_layer) < 0);
        assert(!glm53f_dense_ffn_create_dist(&d, NULL, argv[1], layer));
        assert(!unlink(manifest));
        int fd = open(blob, O_CREAT | O_EXCL | O_RDWR, 0600); assert(fd >= 0);
        f = fopen(manifest, "wx"); assert(f);
        fprintf(f, "# GLM53F_PP_DENSE_V1 layout=pp3-tp4 world_rank=%d stage=%d "
            "tp_rank=%d tp_size=4 cuts=%d,%d layers=%d:%d source_metadata_fnv1a=0\n",
            rank, d.map.stage, d.map.tp_rank, config.cuts[0], config.cuts[1],
            d.map.first_layer, d.map.end_layer);
        const char *names[] = {"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"};
        uint64_t offset = 0;
        for (int i = 0; i < 3; ++i) {
            int rows = i == 2 ? 4096 : 3072, columns = i == 2 ? 3072 : 4096;
            size_t bytes = (size_t)rows * (columns / 256) * 144; /* native Q4_K */
            uint64_t hash = zeros(fd, bytes);
            fprintf(f, "%" PRIu64 " 12 Q4_K %d %d blk.%d.%s\n", offset, rows, columns, layer, names[i]);
            fprintf(f, "# PAYLOAD offset=%" PRIu64 " bytes=%zu fnv1a=%016" PRIx64 "\n", offset, bytes, hash);
            offset += bytes;
        }
        assert(!fclose(f));
        assert(!glm53f_pp_manifest_check(manifest, "DENSE", &d, d.map.first_layer, d.map.end_layer));
        assert(glm53f_pp_manifest_check(manifest, "DENSE", &d, d.map.first_layer, d.map.end_layer - 1) < 0);
        glm53f_dense_ffn_context_12n *dense = glm53f_dense_ffn_create_dist(&d, NULL, argv[1], layer);
        assert(dense);
        float *input = calloc(4 * 4096, sizeof(float)), *output = malloc(4 * 4096 * sizeof(float));
        assert(input && output);
        for (int n = 1; n <= 4; ++n) {
            assert(!glm53f_dense_ffn_sublayer_batch_12n(dense, output, input, n));
            for (int j = 0; j < n * 4096; ++j) assert(output[j] == 0.0f);
        }
        assert(!glm53f_dense_ffn_sublayer_12n(dense, output, input));
        for (int j = 0; j < 4096; ++j) assert(output[j] == 0.0f);
        glm53f_dense_ffn_free_12n(dense); free(input); free(output);
        unsigned char corrupt = 1; assert(pwrite(fd, &corrupt, 1, 0) == 1);
        assert(!glm53f_dense_ffn_create_dist(&d, NULL, argv[1], layer));
        assert(!close(fd)); assert(!unlink(blob)); assert(!unlink(manifest));
    } else assert(!glm53f_dense_ffn_create_dist(&d, NULL, argv[1], layer));
    MPI_Barrier(d.world);
    glm53f_dist_free(&d);
    }
    if (!rank) puts("GLM53F_DENSE_DIST_PASS cuts=3 stage_tp=4 batches=1:4 legacy_rejected=1 corruption_rejected=1");
    MPI_Finalize(); return 0;
}
