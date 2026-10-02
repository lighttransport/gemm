#define main pp_sparse_stage_program_main
#include "glm53f_pp_sparse_stage.c"
#undef main
#include <assert.h>
static unsigned char pattern(uint64_t i) { return (unsigned char)((i * 19 + i / 251) % 256); }
int main(int argc, char **argv) {
    int rank; MPI_Init(&argc, &argv); MPI_Comm_rank(MPI_COMM_WORLD, &rank); assert(argc == 2);
    if (!rank) {
        char source[4096], output[4096], manifest[4096];
        assert(snprintf(source, sizeof(source), "%s/sparse-source-%ld", argv[1], (long)getpid()) < (int)sizeof(source));
        assert(snprintf(output, sizeof(output), "%s/sparse-output-%ld", argv[1], (long)getpid()) < (int)sizeof(output));
        assert(snprintf(manifest, sizeof(manifest), "%s/sparse-manifest-%ld", argv[1], (long)getpid()) < (int)sizeof(manifest));
        const int types[] = {GGML_TYPE_Q8_0, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K};
        int cases = 0;
        for (int type_index = 0; type_index < 3; ++type_index) {
            int type = types[type_index];
            gguf_tensor_info q, op; memset(&q, 0, sizeof(q)); memset(&op, 0, sizeof(op));
            q.type = op.type = type; q.n_dims = op.n_dims = 2;
            q.dims[0] = QA; q.dims[1] = QKV; op.dims[0] = QKV; op.dims[1] = HIDDEN;
            int fd = open(source, O_CREAT | O_EXCL | O_RDWR, 0600); assert(fd >= 0);
            tensor_ref qr = {fd, 0, &q}, opr = {fd, 0, &op};
            size_t qb = QKV * row_bytes(&qr, QA), ob = HIDDEN * row_bytes(&opr, QKV); opr.base = qb;
            unsigned char scratch[65536];
            for (uint64_t offset = 0; offset < qb + ob; offset += sizeof(scratch)) {
                size_t count = qb + ob - offset; if (count > sizeof(scratch)) count = sizeof(scratch);
                for (size_t i = 0; i < count; ++i) scratch[i] = pattern(offset + i);
                assert(write(fd, scratch, count) == (ssize_t)count);
            }
            void *rows = malloc(64 * row_bytes(&qr, QA));
            void *full = malloc(64 * row_bytes(&opr, QKV));
            void *local = malloc(64 * row_bytes(&opr, QKV / 4)); assert(rows && full && local);
            for (int part = 0; part < 4; ++part) {
                int out = open(output, O_CREAT | O_EXCL | O_RDWR, 0600); assert(out >= 0);
                FILE *m = fopen(manifest, "wx"); assert(m);
                pp_hash = UINT64_C(1469598103934665603); uint64_t offset = 0;
                assert(!stage_rows(out, m, &offset, &qr, type, QKV, QA, part * (QKV / 4), QKV / 4, "query", rows));
                assert(!stage_columns(out, m, &offset, &opr, type, HIDDEN, QKV, part * (QKV / 4), QKV / 4, "output", full, local));
                assert(!fclose(m));
                size_t fr = row_bytes(&opr, QKV), lr = row_bytes(&opr, QKV / 4);
                uint64_t hash = UINT64_C(1469598103934665603);
                for (uint64_t start = 0; start < offset; start += sizeof(scratch)) {
                    size_t count = offset - start; if (count > sizeof(scratch)) count = sizeof(scratch);
                    assert(pread(out, scratch, count, (off_t)start) == (ssize_t)count);
                    for (size_t i = 0; i < count; ++i) {
                        uint64_t k = start + i, expected;
                        if (k < qb / 4) expected = (uint64_t)part * qb / 4 + k;
                        else { k -= qb / 4; expected = qb + (k / lr) * fr + (uint64_t)part * lr + k % lr; }
                        assert(scratch[i] == pattern(expected)); hash ^= scratch[i]; hash *= UINT64_C(1099511628211);
                    }
                }
                assert(offset == (qb + ob) / 4 && hash == pp_hash);
                assert(!glm53f_pp_blob_verify(output, offset, hash));
                unsigned char bad = pattern((uint64_t)part * qb / 4) ^ 1;
                assert(pwrite(out, &bad, 1, 0) == 1 && glm53f_pp_blob_verify(output, offset, hash) == -1);
                assert(!close(out) && !unlink(output) && !unlink(manifest)); ++cases;
            }
            free(rows); free(full); free(local); assert(!close(fd) && !unlink(source));
        }
        printf("GLM53F_PP_SPARSE_SHARDS_PASS native_formats=3 parts=4 cases=%d raw_bytes_and_hashes=EXACT corruption_rejected=1\n", cases);
    }
    MPI_Barrier(MPI_COMM_WORLD); MPI_Finalize(); return 0;
}
