/* Dequantize a Q2 GGUF vocabulary matrix into rank-local F32 row shards. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define GGUF_LOADER_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/gguf_loader.h"
#include "../../common/ggml_dequant.h"
#include <mpi.h>
#include <errno.h>
#include <inttypes.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

enum { VOCAB = 154880, HIDDEN = 4096, CHUNK_ROWS = 128 };

#ifndef GLM53F_Q2_MATRIX_NAME
#define GLM53F_Q2_MATRIX_NAME "token_embd.weight"
#endif
#ifndef GLM53F_Q2_STAGE_LABEL
#define GLM53F_Q2_STAGE_LABEL "glm53f_q2_embed_stage"
#endif
#ifndef GLM53F_Q2_STAGE_MANIFEST
#define GLM53F_Q2_STAGE_MANIFEST "GLM53F_Q2_EMBED_V1"
#endif
static void die(int rank, const char *what) {
    fprintf(stderr, "rank=%d %s: %s: %s\n", rank, GLM53F_Q2_STAGE_LABEL, what,
            errno ? strerror(errno) : "contract failure");
    MPI_Abort(MPI_COMM_WORLD, 2);
}

static int read_exact(int fd, uint64_t off, void *dst, size_t bytes) {
    unsigned char *p = dst;
    while (bytes) {
        ssize_t n = pread(fd, p, bytes, (off_t) off);
        if (n < 0) { if (errno == EINTR) continue; return -1; }
        if (!n) { errno = EIO; return -1; }
        p += n; off += (uint64_t)n; bytes -= (size_t)n;
    }
    return 0;
}

static uint64_t fnv1a(const float *p, size_t n) {
    uint64_t h = UINT64_C(1469598103934665603);
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n * sizeof(*p); ++i) { h ^= b[i]; h *= UINT64_C(1099511628211); }
    return h;
}

int main(int argc, char **argv) {
    int rank, ranks;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (argc != 3 || ranks != 12) die(rank, "usage: MODEL-00001-of-00004.gguf STAGE_DIR");

    const int row0 = (int)((long long)VOCAB * rank / ranks);
    const int rows = (int)((long long)VOCAB * (rank + 1) / ranks) - row0;
    char path[4096], tmp[4096], manifest[4096];
    snprintf(path, sizeof(path), "%s/rank%02d.f32", argv[2], rank);
    snprintf(tmp, sizeof(tmp), "%s/.rank%02d.f32.tmp.%ld", argv[2], rank, (long)getpid());
    snprintf(manifest, sizeof(manifest), "%s/rank%02d.manifest", argv[2], rank);
    if (mkdir(argv[2], 0755) && errno != EEXIST) die(rank, "mkdir stage");

    struct stat st;
    if (!stat(path, &st) && st.st_size == (off_t)((size_t)rows * HIDDEN * sizeof(float))) {
        printf("SENTINEL %s=REUSE rank=%d rows=%d\n",
               GLM53F_Q2_STAGE_LABEL, rank, rows);
        MPI_Finalize(); return 0;
    }

    gguf_context *g = gguf_open_multi(argv[1], 3);
    if (!g) die(rank, "open GGUF");
    const gguf_tensor_info *ti = NULL;
    int fd = -1;
    uint64_t base = 0;
    for (uint64_t i = 0; i < g->n_tensors; ++i) {
        if (g->tensors[i].name.str &&
            !strcmp(g->tensors[i].name.str, GLM53F_Q2_MATRIX_NAME)) {
            ti = &g->tensors[i];
            fd = g->tensor_fds ? g->tensor_fds[i] : g->fd;
            base = g->tensor_file_offsets ? g->tensor_file_offsets[i] :
                   g->data_offset + g->tensors[i].offset;
            break;
        }
    }
    if (!ti || ti->n_dims != 2 || ti->dims[0] != HIDDEN || ti->dims[1] != VOCAB)
        die(rank, GLM53F_Q2_MATRIX_NAME " contract");
    if (ti->type >= GGML_TYPE_COUNT || ggml_type_info[ti->type].block_size <= 0 ||
        HIDDEN % ggml_type_info[ti->type].block_size)
        die(rank, "unsupported vocabulary matrix type");
    const size_t row_bytes = (size_t)(HIDDEN / ggml_type_info[ti->type].block_size) *
                             ggml_type_info[ti->type].type_size;
    unsigned char *raw = malloc((size_t)CHUNK_ROWS * row_bytes);
    float *out = malloc((size_t)CHUNK_ROWS * HIDDEN * sizeof(float));
    if (!raw || !out) die(rank, "scratch allocation");
    int ofd = open(tmp, O_CREAT | O_EXCL | O_WRONLY, 0644);
    if (ofd < 0) die(rank, "create staged vocabulary matrix");
    uint64_t hash = UINT64_C(1469598103934665603);
    for (int r0 = 0; r0 < rows; r0 += CHUNK_ROWS) {
        int nr = rows - r0 < CHUNK_ROWS ? rows - r0 : CHUNK_ROWS;
        if (read_exact(fd, base + (uint64_t)(row0 + r0) * row_bytes, raw,
                       (size_t)nr * row_bytes)) die(rank, "read vocabulary matrix rows");
        for (int r = 0; r < nr; ++r)
            if (dequant_row(ti->type, raw + (size_t)r * row_bytes,
                            out + (size_t)r * HIDDEN, HIDDEN))
                die(rank, "dequantize vocabulary matrix row");
        size_t bytes = (size_t)nr * HIDDEN * sizeof(float);
        if (write(ofd, out, bytes) != (ssize_t)bytes)
            die(rank, "write staged vocabulary matrix");
        hash ^= fnv1a(out, (size_t)nr * HIDDEN);
        hash *= UINT64_C(1099511628211);
    }
    if (fdatasync(ofd) || close(ofd) || rename(tmp, path))
        die(rank, "commit staged vocabulary matrix");
    FILE *mf = fopen(manifest, "w");
    if (!mf) die(rank, "write vocabulary matrix manifest");
    fprintf(mf, "# %s tensor=%s rank=%d ranks=12 row0=%d rows=%d hidden=%d type=%s bytes=%zu fnv1a=%016" PRIx64 "\n",
            GLM53F_Q2_STAGE_MANIFEST, GLM53F_Q2_MATRIX_NAME,
            rank, row0, rows, HIDDEN, ggml_type_name(ti->type),
            (size_t)rows * HIDDEN * sizeof(float), hash);
    fclose(mf);
    printf("SENTINEL %s=OK tensor=%s rank=%d row0=%d rows=%d type=%s bytes=%zu hash=%016" PRIx64 "\n",
           GLM53F_Q2_STAGE_LABEL, GLM53F_Q2_MATRIX_NAME,
           rank, row0, rows, ggml_type_name(ti->type),
           (size_t)rows * HIDDEN * sizeof(float), hash);
    free(out); free(raw); gguf_close(g); MPI_Finalize(); return 0;
}
