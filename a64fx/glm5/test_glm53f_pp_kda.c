#define main pp_kda_stage_program_main
#include "glm53f_pp_kda_stage.c"
#undef main
#include <assert.h>
static unsigned char pattern(uint64_t i) { return (unsigned char)((i * 17 + i / 253) % 256); }
static void fill_file(int fd, size_t bytes) {
    unsigned char buffer[65536];
    for (size_t offset = 0; offset < bytes; offset += sizeof(buffer)) {
        size_t count = bytes - offset; if (count > sizeof(buffer)) count = sizeof(buffer);
        for (size_t i = 0; i < count; ++i) buffer[i] = pattern(offset + i);
        assert(!write_all(fd, buffer, count));
    }
}
int main(int argc, char **argv) {
    int rank; MPI_Init(&argc, &argv); MPI_Comm_rank(MPI_COMM_WORLD, &rank); assert(argc == 2);
    if (!rank) {
        assert(!mkdir(argv[1], 0700) || access(argv[1], W_OK) == 0);
        pp_config = glm53f_parallel_default(); pp_config.layout = GLM53F_PP3_TP4;
        assert(!glm53f_parallel_map_rank(&pp_config, 0, 12, &pp_map));
        char source[4096], destination[4096], manifest[4096];
        snprintf(source, sizeof(source), "%s/kda-source-%ld", argv[1], (long)getpid());
        snprintf(destination, sizeof(destination), "%s/kda-output-%ld", argv[1], (long)getpid());
        snprintf(manifest, sizeof(manifest), "%s/kda-manifest-%ld", argv[1], (long)getpid());
        void *buffer = malloc(64 * row_bytes(GGML_TYPE_Q8_0, HIDDEN));
        void *full = malloc(64 * row_bytes(GGML_TYPE_Q8_0, QKV));
        void *local = malloc(64 * row_bytes(GGML_TYPE_Q8_0, QKV / 4));
        assert(buffer && full && local);
        const int types[] = {GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K};
        int cases = 0;
        for (int ti = 0; ti < 4; ++ti) {
            int type = types[ti]; size_t rb = row_bytes(type, HIDDEN), bytes = QKV * rb;
            int fd = open(source, O_CREAT | O_EXCL | O_RDWR, 0600); assert(fd >= 0);
            fill_file(fd, 2 * bytes);
            gguf_tensor_info tensors[2]; memset(tensors, 0, sizeof(tensors));
            tensors[0].name.str = "q"; tensors[1].name.str = "output";
            for (int i = 0; i < 2; ++i) { tensors[i].type = type; tensors[i].n_dims = 2; }
            tensors[0].dims[0] = HIDDEN; tensors[0].dims[1] = QKV;
            tensors[1].dims[0] = QKV; tensors[1].dims[1] = HIDDEN; tensors[1].offset = bytes;
            gguf_context g; memset(&g, 0, sizeof(g)); g.fd = fd; g.n_tensors = 2; g.tensors = tensors;
            for (int part = 0; part < 4; ++part) {
                pp_hash = UINT64_C(1469598103934665603);
                int out = open(destination, O_CREAT | O_EXCL | O_RDWR, 0600); assert(out >= 0);
                FILE *m = fopen(manifest, "wx"); assert(m);
                char header[512]; stage_header(header, sizeof(header), 0, "source_metadata_fnv1a=0"); fputs(header, m);
                uint64_t offset = 0;
                stage_rows(0, out, m, &offset, &g, "q", QKV, HIDDEN, part * (QKV / 4), QKV / 4, buffer);
                stage_output(0, out, m, &offset, &g, "output", part * (QKV / 4), QKV / 4, full, local);
                fprintf(m, "# COMPLETE bytes=%" PRIu64 " fnv1a=%016" PRIx64 "\n", offset, pp_hash);
                assert(!fclose(m));
                assert(complete(manifest, destination, 0, "source_metadata_fnv1a=0"));
                assert(!complete(manifest, destination, 1, "source_metadata_fnv1a=0"));
                assert(!complete(manifest, destination, 0, "source_metadata_fnv1a=1"));
                size_t fr = row_bytes(type, QKV), lr = row_bytes(type, QKV / 4);
                uint64_t hash = UINT64_C(1469598103934665603); unsigned char check[65536];
                for (uint64_t start = 0; start < offset; start += sizeof(check)) {
                    size_t count = offset - start; if (count > sizeof(check)) count = sizeof(check);
                    assert(pread(out, check, count, (off_t)start) == (ssize_t)count);
                    for (size_t j = 0; j < count; ++j) {
                        uint64_t k = start + j, expected;
                        if (k < bytes / 4) expected = (uint64_t)part * bytes / 4 + k;
                        else { k -= bytes / 4; expected = bytes + (k / lr) * fr + (uint64_t)part * lr + k % lr; }
                        assert(check[j] == pattern(expected)); hash ^= check[j]; hash *= UINT64_C(1099511628211);
                    }
                }
                assert(offset == bytes / 2 && hash == pp_hash);
                unsigned char original = pattern((uint64_t)part * bytes / 4), corrupt = original ^ 1;
                assert(pwrite(out, &corrupt, 1, 0) == 1);
                assert(!complete(manifest, destination, 0, "source_metadata_fnv1a=0"));
                assert(pwrite(out, &original, 1, 0) == 1);
                assert(complete(manifest, destination, 0, "source_metadata_fnv1a=0"));
                assert(!close(out)); assert(!unlink(destination)); assert(!unlink(manifest)); ++cases;
            }
            assert(!close(fd)); assert(!unlink(source));
        }
        free(buffer); free(full); free(local);
        printf("GLM53F_PP_KDA_SHARDS_PASS native_formats=4 parts=4 cases=%d raw_bytes_and_hashes=EXACT\n", cases);
    }
    MPI_Barrier(MPI_COMM_WORLD); MPI_Finalize(); return 0;
}
