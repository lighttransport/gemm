#define main pp_dense_stage_program_main
#ifdef GLM53F_PP_SHARED_FIXTURE
#include "glm53f_pp_shared_stage.c"
#else
#include "glm53f_pp_dense_stage.c"
#endif
#undef main
#include <assert.h>
static unsigned char pattern(uint64_t i) { return (unsigned char)((i * 13 + i / 251) % 256); }
static void fill_file(int fd, size_t bytes) {
    unsigned char buffer[65536];
    for (size_t offset = 0; offset < bytes; offset += sizeof(buffer)) {
        size_t n = bytes - offset; if (n > sizeof(buffer)) n = sizeof(buffer);
        for (size_t j = 0; j < n; ++j) buffer[j] = pattern(offset + j);
        assert(!write_all(fd, buffer, n));
    }
}
int main(int argc, char **argv) {
    int rank; MPI_Init(&argc, &argv); MPI_Comm_rank(MPI_COMM_WORLD, &rank); assert(argc == 2);
    if (!rank) {
        assert(!mkdir(argv[1], 0700) || access(argv[1], W_OK) == 0);
        pp_config = glm53f_parallel_default(); pp_config.layout = GLM53F_PP3_TP4;
        assert(!glm53f_parallel_map_rank(&pp_config, 0, 12, &pp_map));
        char source[4096], destination[4096], manifest_path[4096];
        snprintf(source, sizeof(source), "%s/dense-source-%ld", argv[1], (long)getpid());
        snprintf(destination, sizeof(destination), "%s/dense-output-%ld", argv[1], (long)getpid());
        snprintf(manifest_path, sizeof(manifest_path), "%s/dense-manifest-%ld", argv[1], (long)getpid());
        const int types[] = {GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K,
            GGML_TYPE_Q6_K, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_XS};
        unsigned char *buffer = malloc(64u * row_bytes(GGML_TYPE_Q8_0, HIDDEN));
        unsigned char *input = malloc(64u * row_bytes(GGML_TYPE_Q8_0, INTER));
        unsigned char *output = malloc(64u * row_bytes(GGML_TYPE_Q8_0, INTER / 4));
        assert(buffer && input && output);
        int cases = 0;
        for (int type_index = 0; type_index < 7; ++type_index) {
            int type = types[type_index]; size_t rb = row_bytes(type, HIDDEN), full = INTER * rb;
            int fd = open(source, O_CREAT | O_EXCL | O_RDWR, 0600); assert(fd >= 0);
            fill_file(fd, 3 * full);
            gguf_tensor_info gu, down;
            memset(&gu, 0, sizeof(gu)); memset(&down, 0, sizeof(down));
            gu.type = down.type = (uint32_t)type; gu.n_dims = down.n_dims = 2;
            gu.name.str = down.name.str = "fixture";
            gu.dims[0] = HIDDEN; gu.dims[1] = INTER;
            down.dims[0] = INTER; down.dims[1] = HIDDEN;
            tensor_ref gate = {fd, 0, &gu}, up = {fd, full, &gu}, d = {fd, 2 * full, &down};
            size_t down_rb = row_bytes(type, INTER), part_rb = row_bytes(type, INTER / 4);
            for (int part = 0; part < 4; ++part) {
                pp_hash = UINT64_C(1469598103934665603);
                int out = open(destination, O_CREAT | O_EXCL | O_RDWR, 0600); assert(out >= 0);
                FILE *m = fopen(manifest_path, "wx"); assert(m);
                char header[512]; stage_header(header, sizeof(header), 0, "source_metadata_fnv1a=0"); fputs(header, m);
                uint64_t offset = 0;
                assert(!stage_rows(out, m, &offset, &gate, part * (INTER / 4), INTER / 4, 0, "ffn_gate.weight", buffer));
                assert(!stage_rows(out, m, &offset, &up, part * (INTER / 4), INTER / 4, 0, "ffn_up.weight", buffer));
                assert(!stage_down_columns(out, m, &offset, &d, part * (INTER / 4), INTER / 4, 0, input, output));
                fprintf(m, "# COMPLETE bytes=%" PRIu64 " fnv1a=%016" PRIx64 "\n", offset, pp_hash);
                assert(!fclose(m));
                assert(stage_complete(manifest_path, destination, 0, "source_metadata_fnv1a=0"));
                assert(!stage_complete(manifest_path, destination, 1, "source_metadata_fnv1a=0"));
                assert(!stage_complete(manifest_path, destination, 0, "source_metadata_fnv1a=1"));
                uint64_t actual_hash = UINT64_C(1469598103934665603);
                unsigned char check[65536]; size_t channels = INTER / 4;
                for (uint64_t start = 0; start < offset; start += sizeof(check)) {
                    size_t n = offset - start; if (n > sizeof(check)) n = sizeof(check);
                    assert(pread(out, check, n, (off_t)start) == (ssize_t)n);
                    for (size_t j = 0; j < n; ++j) {
                        uint64_t k = start + j, expected;
                        if (k < channels * rb) expected = (uint64_t)part * channels * rb + k;
                        else if (k < 2 * channels * rb) expected = full + (uint64_t)part * channels * rb + k - channels * rb;
                        else {
                            k -= 2 * channels * rb;
                            expected = 2 * full + (k / part_rb) * down_rb + (uint64_t)part * part_rb + k % part_rb;
                        }
                        assert(check[j] == pattern(expected));
                        actual_hash ^= check[j]; actual_hash *= UINT64_C(1099511628211);
                    }
                }
                assert(actual_hash == pp_hash && offset == 3 * full / 4);
                unsigned char original = pattern((uint64_t)part * (INTER / 4) * rb), corrupt = original ^ 1;
                assert(pwrite(out, &corrupt, 1, 0) == 1);
                assert(!stage_complete(manifest_path, destination, 0, "source_metadata_fnv1a=0"));
                assert(pwrite(out, &original, 1, 0) == 1);
                assert(stage_complete(manifest_path, destination, 0, "source_metadata_fnv1a=0"));
                assert(!close(out)); assert(!unlink(destination)); assert(!unlink(manifest_path)); ++cases;
            }
            assert(!close(fd)); assert(!unlink(source));
        }
        free(buffer); free(input); free(output);
        printf("GLM53F_PP_%s_SHARDS_PASS native_formats=7 parts=4 cases=%d raw_bytes_and_hashes=EXACT\n", PP_COMPONENT, cases);
    }
    MPI_Barrier(MPI_COMM_WORLD); MPI_Finalize(); return 0;
}
