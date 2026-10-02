#define main routed_stage_program_main
#include "glm53f_pp_routed_stage.c"
#undef main
#include <assert.h>
static unsigned char pattern(uint64_t i) { return (unsigned char)((i * 13 + i / 251) % 256); }
static void fill_file(int fd, size_t bytes) {
    unsigned char buf[65536];
    for (size_t start = 0; start < bytes; start += sizeof(buf)) {
        size_t n = bytes - start; if (n > sizeof(buf)) n = sizeof(buf);
        for (size_t j = 0; j < n; ++j) buf[j] = pattern(start + j);
        assert(!write_all(fd, buf, n, NULL));
    }
}
int main(int argc, char **argv) {
    int rank; MPI_Init(&argc, &argv); MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    assert(argc == 2);
    if (!rank) {
        pp_config = glm53f_parallel_default(); pp_config.layout = GLM53F_PP3_TP4;
        assert(!glm53f_parallel_map_rank(&pp_config, 0, 12, &pp_map));
        for (int expert = 0; expert < NEXPERTS; ++expert) {
            int coverage[4] = {0};
            for (int r = 0; r < 4; ++r) {
                int part = owned_part(expert, r); assert(part >= 0 && part < 4);
                assert(owned_part(expert, r + 4) == part && owned_part(expert, r + 8) == part);
                ++coverage[part];
            }
            for (int p = 0; p < 4; ++p) assert(coverage[p] == 1);
        }
        char source[4096], destination[4096], manifest_path[4096];
        snprintf(source, sizeof(source), "%s/fixture-source-%ld", argv[1], (long)getpid());
        snprintf(destination, sizeof(destination), "%s/fixture-output-%ld", argv[1], (long)getpid());
        snprintf(manifest_path, sizeof(manifest_path), "%s/fixture-manifest-%ld", argv[1], (long)getpid());
        const int types[] = {GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
            GGML_TYPE_IQ2_XS, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_XS};
        unsigned char *buf = malloc(2u << 20), *in = malloc(1u << 20), *packed = malloc(1u << 20);
        assert(buf && in && packed);
        int cases = 0;
        for (int type_index = 0; type_index < 6; ++type_index) {
            int type = types[type_index];
            size_t rb = row_bytes(type, HIDDEN), full = INTER * rb;
            int fd = open(source, O_CREAT | O_EXCL | O_RDWR, 0600); assert(fd >= 0);
            fill_file(fd, 3 * full);
            gguf_tensor_info ti; memset(&ti, 0, sizeof(ti)); ti.type = (uint32_t)type; ti.name.str = "fixture";
            tensor_ref gate = {NULL, fd, 0, &ti}, up = {NULL, fd, full, &ti}, down = {NULL, fd, 2 * full, &ti};
            size_t down_rb = row_bytes(type, INTER), part_rb = row_bytes(type, PART_INTER);
            for (int part = 0; part < 4; ++part) {
                int out = open(destination, O_CREAT | O_EXCL | O_RDWR, 0600); assert(out >= 0);
                FILE *m = fopen(manifest_path, "wx"); assert(m);
                char header[512]; stage_header(header, sizeof(header), 3, 12, 0); fputs(header, m);
                uint64_t offset = 0, hash = UINT64_C(1469598103934665603);
                assert(!put_gate_up(out, m, &offset, &hash, &gate, &up, 3, 0, part, buf, 0));
                assert(!put_down(out, m, &offset, &hash, &down, 3, 0, part, in, packed, 0));
                fprintf(m, "# COMPLETE bytes=%" PRIu64 " fnv1a=%016" PRIx64 "\n", offset, hash);
                assert(!fclose(m));
                assert(stage_complete(manifest_path, destination, 3, 12, 0));
                assert(!stage_complete(manifest_path, destination, 3, 11, 0));
                assert(!stage_complete(manifest_path, destination, 3, 12, 1));
                ++pp_source_identity;
                assert(!stage_complete(manifest_path, destination, 3, 12, 0));
                --pp_source_identity;
                uint64_t actual_hash = UINT64_C(1469598103934665603);
                unsigned char check[65536];
                for (uint64_t start = 0; start < offset; start += sizeof(check)) {
                    size_t n = offset - start; if (n > sizeof(check)) n = sizeof(check);
                    assert(pread(out, check, n, (off_t)start) == (ssize_t)n);
                    for (size_t j = 0; j < n; ++j) {
                        uint64_t k = start + j, expected;
                        if (k < (uint64_t)PART_INTER * rb) expected = (uint64_t)part * PART_INTER * rb + k;
                        else if (k < (uint64_t)2 * PART_INTER * rb)
                            expected = full + (uint64_t)part * PART_INTER * rb + k - (uint64_t)PART_INTER * rb;
                        else {
                            k -= (uint64_t)2 * PART_INTER * rb;
                            expected = 2 * full + (k / part_rb) * down_rb + (uint64_t)part * part_rb + k % part_rb;
                        }
                        assert(check[j] == pattern(expected));
                        actual_hash ^= check[j]; actual_hash *= UINT64_C(1099511628211);
                    }
                }
                assert(actual_hash == hash);
                assert(!close(out)); assert(!unlink(destination)); assert(!unlink(manifest_path)); ++cases;
            }
            assert(!close(fd)); assert(!unlink(source));
        }
        free(buf); free(in); free(packed);
        printf("GLM53F_PP_ROUTED_PASS native_formats=6 parts=4 cases=%d experts=288\n", cases);
    }
    MPI_Barrier(MPI_COMM_WORLD); MPI_Finalize(); return 0;
}
