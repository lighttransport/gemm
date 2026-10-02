#define _GNU_SOURCE
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "glm53f_pp_core.h"
#include <assert.h>
#include <inttypes.h>
static uint64_t digest(const unsigned char *p, size_t bytes) {
    uint64_t h = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < bytes; ++i) { h ^= p[i]; h *= UINT64_C(1099511628211); }
    return h;
}
int main(int argc, char **argv) {
    int provided, rank;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    assert(argc == 2 && provided >= MPI_THREAD_SERIALIZED);
    glm53f_parallel_config config = glm53f_parallel_default(); config.layout = GLM53F_PP3_TP4;
    glm53f_dist d; assert(!glm53f_dist_init(&d, MPI_COMM_WORLD, &config));
    char dir[4096], path[4096], name[512], header[1024];
    assert(snprintf(dir, sizeof(dir), "%s/core-rank%02d-XXXXXX", argv[1], rank) < (int)sizeof(dir));
    assert(mkdtemp(dir)); d.core_stage = dir;
    snprintf(name, sizeof(name), "model.language_model.layers.%d.hc_attn_fn", d.map.first_layer);
    assert(snprintf(path, sizeof(path), "%s/model.safetensors.index.json", dir) < (int)sizeof(path));
    FILE *f = fopen(path, "w"); assert(f);
    assert(fprintf(f, "{\"weight_map\":{\"%s\":\"part.safetensors\"}}", name) > 0 && !fclose(f));
    assert(snprintf(path, sizeof(path), "%s/part.safetensors", dir) < (int)sizeof(path));
    int hlen = snprintf(header, sizeof(header), "{\"%s\":{\"dtype\":\"U8\",\"shape\":[3,8],\"data_offsets\":[0,24]}}", name);
    assert(hlen > 0 && hlen < (int)sizeof(header)); uint64_t length = hlen;
    unsigned char data[24], result[24], columns[9];
    for (int i = 0; i < 24; ++i) data[i] = (unsigned char)(i + rank);
    for (int i = 0; i < 9; ++i) columns[i] = data[(i / 3) * 8 + 2 + i % 3];
    f = fopen(path, "wb"); assert(f);
    assert(fwrite(&length, 8, 1, f) == 1 && fwrite(header, 1, length, f) == length && fwrite(data, 1, 24, f) == 24 && !fclose(f));
    assert(snprintf(path, sizeof(path), "%s/rank%02d.core.blob", dir, rank) < (int)sizeof(path));
    int fd = open(path, O_CREAT | O_EXCL | O_RDWR, 0644); assert(fd >= 0);
    assert(pwrite(fd, data + 4, 12, 0) == 12 && pwrite(fd, columns, 9, 256) == 9);
    assert(snprintf(path, sizeof(path), "%s/rank%02d.core.manifest", dir, rank) < (int)sizeof(path));
    f = fopen(path, "w"); assert(f);
    assert(fprintf(f, "# GLM53F_PP_CORE_V1 layout=pp3-tp4 world_rank=%d stage=%d tp_rank=%d tp_size=4 cuts=15,30 layers=%d:%d source_metadata_fnv1a=0000000000000001\n", rank, d.map.stage, d.map.tp_rank, d.map.first_layer, d.map.end_layer) > 0);
    assert(fprintf(f, "R %s 4 12 0\n# PAYLOAD offset=0 bytes=12 fnv1a=%016" PRIx64 "\nC %s 8 2 3 256\n# PAYLOAD offset=256 bytes=9 fnv1a=%016" PRIx64 "\n# COMPLETE bytes=265\n", name, digest(data + 4, 12), name, digest(columns, 9)) > 0 && !fclose(f));
    glm53f_st_context *st = glm53f_pp_core_open(&d, dir); assert(st);
    assert(!glm53f_st_read(st, name, 4, result, 12) && !memcmp(result, data + 4, 12));
    assert(!glm53f_st_read_columns(st, name, 8, 2, 3, result) && !memcmp(result, columns, 9));
    assert(glm53f_st_read(st, name, 0, result, 12) == -1);
    unsigned char bad = data[4] ^ 1; assert(pwrite(fd, &bad, 1, 0) == 1);
    assert(glm53f_st_read(st, name, 4, result, 12) == -1);
    assert(!glm53f_st_read_columns(st, name, 8, 2, 3, result));
    glm53f_st_close(st); assert(!close(fd));
    d.config.cuts[0] = 14; assert(!glm53f_pp_core_open(&d, dir)); d.config.cuts[0] = 15;
    assert(!unlink(path));
    assert(snprintf(path, sizeof(path), "%s/rank%02d.core.blob", dir, rank) < (int)sizeof(path)); assert(!unlink(path));
    assert(snprintf(path, sizeof(path), "%s/part.safetensors", dir) < (int)sizeof(path)); assert(!unlink(path));
    assert(snprintf(path, sizeof(path), "%s/model.safetensors.index.json", dir) < (int)sizeof(path)); assert(!unlink(path) && !rmdir(dir));
    glm53f_dist_free(&d);
    if (!rank) puts("GLM53F_PP_CORE_PASS rows=1 columns=1 missing_slice_rejected=1 corruption_rejected=1 cuts_rejected=1 ranks=12");
    MPI_Finalize(); return 0;
}
