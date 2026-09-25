/* Stage native GGUF dense-FFN blocks for layers 0--2 to one rank-local image.
 * Gate/up rows and down columns use the production 12-way tensor partition;
 * all slices start on a 256-value GGML quantization-block boundary. */
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

enum { RANKS = 12, LAYERS = 3, HIDDEN = 4096, INTER = 12288 };

typedef struct {
    int fd;
    uint64_t base;
    const gguf_tensor_info *info;
} tensor_ref;

static void die(int rank, const char *message) {
    fprintf(stderr, "rank=%d glm53f_q2_dense_stage: %s: %s\n", rank,
            message, errno ? strerror(errno) : "contract failure");
    MPI_Abort(MPI_COMM_WORLD, 2);
}

static tensor_ref find_tensor(const gguf_context *g, const char *name) {
    tensor_ref out = {-1, 0, NULL};
    for (uint64_t i = 0; i < g->n_tensors; ++i) {
        if (!g->tensors[i].name.str || strcmp(g->tensors[i].name.str, name))
            continue;
        out.info = &g->tensors[i];
        out.fd = g->tensor_fds ? g->tensor_fds[i] : g->fd;
        out.base = g->tensor_file_offsets ? g->tensor_file_offsets[i] :
                   g->data_offset + g->tensors[i].offset;
        break;
    }
    return out;
}

static size_t row_bytes(uint32_t type, uint64_t columns) {
    if (type >= GGML_TYPE_COUNT || ggml_type_info[type].block_size <= 0 ||
        columns % (uint64_t)ggml_type_info[type].block_size) return 0;
    return (size_t)(columns / (uint64_t)ggml_type_info[type].block_size) *
           (size_t)ggml_type_info[type].type_size;
}

static int supported(uint32_t type) {
    return type == GGML_TYPE_Q8_0 ||
           type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K ||
           type == GGML_TYPE_Q6_K || type == GGML_TYPE_IQ2_XS ||
           type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ4_XS;
}

static int read_exact(const tensor_ref *tensor, uint64_t relative,
                      void *buffer, size_t bytes) {
    unsigned char *p = buffer;
    while (bytes) {
        ssize_t got = pread(tensor->fd, p, bytes,
                            (off_t)(tensor->base + relative));
        if (got < 0) { if (errno == EINTR) continue; return -1; }
        if (!got) { errno = EIO; return -1; }
        p += got;
        relative += (uint64_t)got;
        bytes -= (size_t)got;
    }
    return 0;
}

static int write_all(int fd, const void *buffer, size_t bytes) {
    const unsigned char *p = buffer;
    while (bytes) {
        ssize_t put = write(fd, p, bytes);
        if (put < 0) { if (errno == EINTR) continue; return -1; }
        if (!put) { errno = EIO; return -1; }
        p += put;
        bytes -= (size_t)put;
    }
    return 0;
}

static int stage_rows(int out, FILE *manifest, uint64_t *offset,
                      const tensor_ref *tensor, int row0, int rows,
                      int layer, const char *kind, void *buffer) {
    size_t bytes_per_row = row_bytes(tensor->info->type, HIDDEN);
    size_t bytes = (size_t)rows * bytes_per_row;
    if (!bytes_per_row || tensor->info->n_dims != 2 ||
        tensor->info->dims[0] != HIDDEN ||
        tensor->info->dims[1] != INTER ||
        read_exact(tensor, (uint64_t)row0 * bytes_per_row, buffer, bytes) ||
        write_all(out, buffer, bytes)) return -1;
    fprintf(manifest, "%" PRIu64 " %u %s %d %d blk.%d.%s\n",
            *offset, tensor->info->type, ggml_type_name(tensor->info->type),
            rows, HIDDEN, layer, kind);
    *offset += bytes;
    return 0;
}

static int stage_down_columns(int out, FILE *manifest, uint64_t *offset,
                              const tensor_ref *tensor, int column0,
                              int columns, int layer, void *input,
                              void *output) {
    size_t full_row = row_bytes(tensor->info->type, INTER);
    size_t local_row = row_bytes(tensor->info->type, columns);
    const int chunk_rows = 64;
    if (!full_row || !local_row || tensor->info->n_dims != 2 ||
        tensor->info->dims[0] != INTER ||
        tensor->info->dims[1] != HIDDEN ||
        column0 % ggml_type_info[tensor->info->type].block_size ||
        (size_t)column0 / ggml_type_info[tensor->info->type].block_size *
            ggml_type_info[tensor->info->type].type_size + local_row > full_row)
        return -1;
    uint64_t begin = *offset;
    size_t byte0 = (size_t)column0 /
        ggml_type_info[tensor->info->type].block_size *
        ggml_type_info[tensor->info->type].type_size;
    for (int row0 = 0; row0 < HIDDEN; row0 += chunk_rows) {
        int rows = HIDDEN - row0 < chunk_rows ? HIDDEN - row0 : chunk_rows;
        if (read_exact(tensor, (uint64_t)row0 * full_row, input,
                       (size_t)rows * full_row)) return -1;
        for (int row = 0; row < rows; ++row)
            memcpy((unsigned char *)output + (size_t)row * local_row,
                   (unsigned char *)input + (size_t)row * full_row + byte0,
                   local_row);
        if (write_all(out, output, (size_t)rows * local_row)) return -1;
    }
    fprintf(manifest, "%" PRIu64 " %u %s %d %d blk.%d.ffn_down.weight\n",
            begin, tensor->info->type, ggml_type_name(tensor->info->type),
            HIDDEN, columns, layer);
    *offset += (uint64_t)HIDDEN * local_row;
    return 0;
}

static void model_identity(const char *path, char *out, size_t n) {
    struct stat st;
    if (stat(path, &st)) st.st_size = 0;
    snprintf(out, n, "model_bytes=%lld model=%s", (long long)st.st_size, path);
}

static int stage_complete(const char *manifest_path, const char *blob_path,
                          int rank, const char *identity) {
    FILE *manifest = fopen(manifest_path, "r");
    struct stat st;
    char line[8192], want[8192];
    int header = 0;
    uint64_t expected = UINT64_MAX;
    if (!manifest || stat(blob_path, &st)) {
        if (manifest) fclose(manifest);
        return 0;
    }
    snprintf(want, sizeof(want),
             "# GLM53F_Q2_DENSE_V2 rank=%d ranks=12 layers=0:3 %s\n",
             rank, identity);
    while (fgets(line, sizeof(line), manifest)) {
        unsigned long long bytes;
        if (!strcmp(line, want)) header = 1;
        if (sscanf(line, "# COMPLETE bytes=%llu", &bytes) == 1)
            expected = (uint64_t)bytes;
    }
    fclose(manifest);
    return header && expected == (uint64_t)st.st_size;
}

int main(int argc, char **argv) {
    int rank, ranks, out = -1, rc = 1;
    char blob[4096], manifest_path[4096], blob_tmp[4096], manifest_tmp[4096];
    char identity[4096];
    gguf_context *g = NULL;
    FILE *manifest = NULL;
    void *row_buffer = NULL, *input = NULL, *output = NULL;
    uint64_t offset = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (argc != 3 || ranks != RANKS || strncmp(argv[2], "/local/", 7))
        die(rank, "usage: MODEL-00001-of-00004.gguf /local/STAGE_DIR");
    if (mkdir(argv[2], 0755) && errno != EEXIST) die(rank, "mkdir");
    snprintf(blob, sizeof(blob), "%s/rank%02d.blob", argv[2], rank);
    snprintf(manifest_path, sizeof(manifest_path), "%s/rank%02d.manifest",
             argv[2], rank);
    model_identity(argv[1], identity, sizeof(identity));
    if (stage_complete(manifest_path, blob, rank, identity)) {
        printf("SENTINEL glm53f_q2_dense_stage=REUSE rank=%d\n", rank);
        MPI_Finalize();
        return 0;
    }
    snprintf(blob_tmp, sizeof(blob_tmp), "%s/.rank%02d.blob.tmp.%ld",
             argv[2], rank, (long)getpid());
    snprintf(manifest_tmp, sizeof(manifest_tmp), "%s/.rank%02d.manifest.tmp.%ld",
             argv[2], rank, (long)getpid());
    if (!(g = gguf_open_multi(argv[1], 3)) || g->n_tensors != 1412)
        die(rank, "GGUF metadata");
    if ((out = open(blob_tmp, O_CREAT | O_EXCL | O_WRONLY, 0644)) < 0 ||
        !(manifest = fopen(manifest_tmp, "wx"))) die(rank, "create output");
    fprintf(manifest, "# GLM53F_Q2_DENSE_V2 rank=%d ranks=12 layers=0:3 %s\n",
            rank, identity);
    int row0 = rank * (INTER / RANKS), rows = INTER / RANKS;
    /* Q8_0 (8.5 bits/value) is the widest supported type. */
    size_t max_rows = (size_t)rows * row_bytes(GGML_TYPE_Q8_0, HIDDEN);
    size_t max_full = 64u * row_bytes(GGML_TYPE_Q8_0, INTER);
    size_t max_local = 64u * row_bytes(GGML_TYPE_Q8_0, rows);
    row_buffer = malloc(max_rows);
    input = malloc(max_full);
    output = malloc(max_local);
    if (!row_buffer || !input || !output) die(rank, "scratch allocation");
    for (int layer = 0; layer < LAYERS; ++layer) {
        char name[128];
        snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", layer);
        tensor_ref gate = find_tensor(g, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", layer);
        tensor_ref up = find_tensor(g, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", layer);
        tensor_ref down = find_tensor(g, name);
        if (!gate.info || !up.info || !down.info ||
            !supported(gate.info->type) || !supported(up.info->type) ||
            !supported(down.info->type) ||
            stage_rows(out, manifest, &offset, &gate, row0, rows, layer,
                       "ffn_gate.weight", row_buffer) ||
            stage_rows(out, manifest, &offset, &up, row0, rows, layer,
                       "ffn_up.weight", row_buffer) ||
            stage_down_columns(out, manifest, &offset, &down, row0, rows,
                               layer, input, output))
            die(rank, "tensor stage");
        (void)posix_fadvise(gate.fd, (off_t)gate.base, 0,
                            POSIX_FADV_DONTNEED);
        (void)posix_fadvise(up.fd, (off_t)up.base, 0,
                            POSIX_FADV_DONTNEED);
        (void)posix_fadvise(down.fd, (off_t)down.base, 0,
                            POSIX_FADV_DONTNEED);
    }
    if (fprintf(manifest, "# COMPLETE bytes=%" PRIu64 "\n", offset) < 0 ||
        fflush(manifest) || fsync(fileno(manifest)) || fsync(out) ||
        fclose(manifest) || close(out) || rename(blob_tmp, blob) ||
        rename(manifest_tmp, manifest_path)) die(rank, "publish");
    manifest = NULL;
    out = -1;
    printf("SENTINEL glm53f_q2_dense_stage=OK rank=%d bytes=%" PRIu64 "\n",
           rank, offset);
    rc = 0;
    free(output);
    free(input);
    free(row_buffer);
    gguf_close(g);
    MPI_Finalize();
    return rc;
}
