/* Stage native GGUF shared-expert blocks for routed layers 3--44 into one
 * rank-local image.  Gate/up rows and down columns use one 64-value aligned
 * partition of the 2048-wide intermediate dimension, so the rank-local
 * activation quantization equals llama.cpp's full-vector Q8_0 blocks and the
 * repacked Q8_0 kernel applies.  Quantized blocks are preserved byte-for-byte. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define GGUF_LOADER_IMPLEMENTATION
#include "../../common/gguf_loader.h"

#include <mpi.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

enum { RANKS = 12, FIRST = 3, LAST = 45, HIDDEN = 4096, INTER = 2048, UNIT = 64 };

typedef struct { int fd; uint64_t base; const gguf_tensor_info *info; } tensor_ref;

static void die(int rank, const char *message, const char *tensor) {
    fprintf(stderr, "rank=%d glm53f_q2_shexp_stage: %s%s%s: %s\n", rank, message,
            tensor ? " " : "", tensor ? tensor : "",
            errno ? strerror(errno) : "contract failure");
    MPI_Abort(MPI_COMM_WORLD, 2);
}

static tensor_ref find_tensor(const gguf_context *g, const char *name) {
    tensor_ref t = {-1, 0, NULL};
    for (uint64_t i = 0; i < g->n_tensors; ++i)
        if (g->tensors[i].name.str && !strcmp(g->tensors[i].name.str, name)) {
            t.info = &g->tensors[i];
            t.fd = g->tensor_fds ? g->tensor_fds[i] : g->fd;
            t.base = g->tensor_file_offsets ? g->tensor_file_offsets[i] :
                     g->data_offset + g->tensors[i].offset;
            break;
        }
    return t;
}

static size_t row_bytes(uint32_t type, int columns) {
    if (type >= GGML_TYPE_COUNT || ggml_type_info[type].block_size <= 0 ||
        columns % ggml_type_info[type].block_size) return 0;
    return (size_t)(columns / ggml_type_info[type].block_size) *
           ggml_type_info[type].type_size;
}

static int read_exact(const tensor_ref *t, uint64_t rel, void *dst, size_t n) {
    unsigned char *p = dst;
    while (n) {
        ssize_t z = pread(t->fd, p, n, (off_t)(t->base + rel));
        if (z < 0) { if (errno == EINTR) continue; return -1; }
        if (!z) { errno = EIO; return -1; }
        p += z; rel += (uint64_t)z; n -= (size_t)z;
    }
    return 0;
}

static int write_all(int fd, const void *src, size_t n) {
    const unsigned char *p = src;
    while (n) {
        ssize_t z = write(fd, p, n);
        if (z < 0) { if (errno == EINTR) continue; return -1; }
        if (!z) { errno = EIO; return -1; }
        p += z; n -= (size_t)z;
    }
    return 0;
}

static int complete(const char *manifest, const char *blob, const char *want) {
    FILE *f = fopen(manifest, "r");
    struct stat st;
    char line[8192];
    int header = 0;
    uint64_t bytes = UINT64_MAX;
    if (!f || stat(blob, &st)) { if (f) fclose(f); return 0; }
    while (fgets(line, sizeof line, f)) {
        unsigned long long z;
        if (!strcmp(line, want)) header = 1;
        if (sscanf(line, "# COMPLETE bytes=%llu", &z) == 1) bytes = z;
    }
    fclose(f);
    return header && bytes == (uint64_t)st.st_size;
}

int main(int argc, char **argv) {
    int rank, nr, fd = -1;
    uint64_t off = 0;
    char blob[4096], manifest[4096], bt[4096], mt[4096], header[8192], name[128];
    gguf_context *g = NULL;
    FILE *m = NULL;
    struct stat ms;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nr);
    if (argc != 3 || nr != RANKS || strncmp(argv[2], "/local/", 7))
        die(rank, "usage: GGUF /local/STAGE", NULL);
    const int blocks = INTER / UNIT;
    const int i0 = blocks * rank / RANKS * UNIT;
    const int in = blocks * (rank + 1) / RANKS * UNIT - i0;
    if (stat(argv[1], &ms)) ms.st_size = 0;
    snprintf(header, sizeof header,
             "# GLM53F_Q2_SHEXP_V1 rank=%d ranks=12 slice=%d+%d model_bytes=%lld model=%s\n",
             rank, i0, in, (long long)ms.st_size, argv[1]);
    if (mkdir(argv[2], 0755) && errno != EEXIST) die(rank, "mkdir", argv[2]);
    snprintf(blob, sizeof blob, "%s/rank%02d.blob", argv[2], rank);
    snprintf(manifest, sizeof manifest, "%s/rank%02d.manifest", argv[2], rank);
    if (complete(manifest, blob, header)) {
        printf("SENTINEL glm53f_q2_shexp_stage=REUSE rank=%d\n", rank);
        MPI_Finalize();
        return 0;
    }
    snprintf(bt, sizeof bt, "%s/.rank%02d.blob.%ld", argv[2], rank, (long)getpid());
    snprintf(mt, sizeof mt, "%s/.rank%02d.manifest.%ld", argv[2], rank, (long)getpid());
    if (!(g = gguf_open_multi(argv[1], 3)) || g->n_tensors != 1412)
        die(rank, "GGUF metadata", argv[1]);
    if ((fd = open(bt, O_CREAT | O_EXCL | O_WRONLY, 0644)) < 0 || !(m = fopen(mt, "wx")))
        die(rank, "create", bt);
    fputs(header, m);
    const size_t q8_hidden = row_bytes(GGML_TYPE_Q8_0, HIDDEN);
    const size_t q8_inter = row_bytes(GGML_TYPE_Q8_0, INTER);
    void *rows = malloc((size_t)in * q8_hidden);
    void *full = malloc(64 * q8_inter), *local = malloc(64 * q8_inter);
    if (!rows || !full || !local) die(rank, "scratch", NULL);
    for (int layer = FIRST; layer < LAST; ++layer) {
        static const char *const gate_up[2] = {"ffn_gate_shexp", "ffn_up_shexp"};
        for (int i = 0; i < 2; ++i) {
            snprintf(name, sizeof name, "blk.%d.%s.weight", layer, gate_up[i]);
            tensor_ref t = find_tensor(g, name);
            if (!t.info || t.info->type != GGML_TYPE_Q8_0 || t.info->n_dims != 2 ||
                t.info->dims[0] != HIDDEN || t.info->dims[1] != INTER)
                die(rank, "gate/up contract (Q8_0 required)", name);
            size_t rb = row_bytes(t.info->type, HIDDEN), bytes = (size_t)in * rb;
            if (read_exact(&t, (uint64_t)i0 * rb, rows, bytes) ||
                write_all(fd, rows, bytes)) die(rank, "gate/up stage", name);
            fprintf(m, "%" PRIu64 " %u %s %d %d %s\n", off, t.info->type,
                    ggml_type_name(t.info->type), in, HIDDEN, name);
            off += bytes;
        }
        snprintf(name, sizeof name, "blk.%d.ffn_down_shexp.weight", layer);
        tensor_ref t = find_tensor(g, name);
        if (!t.info || t.info->type != GGML_TYPE_Q8_0 || t.info->n_dims != 2 ||
            t.info->dims[0] != INTER || t.info->dims[1] != HIDDEN)
            die(rank, "down contract (Q8_0 required)", name);
        const int bs = ggml_type_info[t.info->type].block_size;
        size_t fr = row_bytes(t.info->type, INTER), lr = row_bytes(t.info->type, in);
        size_t byte0 = (size_t)(i0 / bs) * ggml_type_info[t.info->type].type_size;
        for (int r0 = 0; r0 < HIDDEN; r0 += 64) {
            if (read_exact(&t, (uint64_t)r0 * fr, full, 64 * fr))
                die(rank, "down read", name);
            for (int r = 0; r < 64; ++r)
                memcpy((unsigned char *)local + (size_t)r * lr,
                       (unsigned char *)full + (size_t)r * fr + byte0, lr);
            if (write_all(fd, local, 64 * lr)) die(rank, "down write", name);
        }
        fprintf(m, "%" PRIu64 " %u %s %d %d %s\n", off, t.info->type,
                ggml_type_name(t.info->type), HIDDEN, in, name);
        off += (uint64_t)HIDDEN * lr;
    }
    for (uint64_t i = 0; i < g->n_tensors; ++i)
        (void)posix_fadvise(g->tensor_fds ? g->tensor_fds[i] : g->fd, 0, 0,
                            POSIX_FADV_DONTNEED);
    fprintf(m, "# COMPLETE bytes=%" PRIu64 "\n", off);
    if (fflush(m) || fsync(fileno(m)) || fsync(fd) || fclose(m) || close(fd) ||
        rename(bt, blob) || rename(mt, manifest)) die(rank, "publish", blob);
    printf("SENTINEL glm53f_q2_shexp_stage=OK rank=%d slice=%d+%d bytes=%" PRIu64 "\n",
           rank, i0, in, off);
    free(local); free(full); free(rows);
    gguf_close(g);
    MPI_Finalize();
    return 0;
}
