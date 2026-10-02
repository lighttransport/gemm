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

#ifdef GLM53F_PP_DENSE_STAGE
#include "glm53f_parallel.h"
#include "glm53f_pp_source.h"
#include "glm53f_pp_blob.h"
static glm53f_parallel_config pp_config;
static glm53f_parallel_map pp_map;
static uint64_t pp_hash = UINT64_C(1469598103934665603), pp_entry_hash;
#ifdef GLM53F_PP_SHARED_STAGE
#define PP_COMPONENT "SHARED"
#define PP_GATE "ffn_gate_shexp.weight"
#define PP_UP "ffn_up_shexp.weight"
#define PP_DOWN "ffn_down_shexp.weight"
enum { RANKS = 4, FIRST = 3, LAYERS = 45, HIDDEN = 4096, INTER = 2048 };
#else
#define PP_COMPONENT "DENSE"
#define PP_GATE "ffn_gate.weight"
#define PP_UP "ffn_up.weight"
#define PP_DOWN "ffn_down.weight"
enum { RANKS = 4, FIRST = 0, LAYERS = 3, HIDDEN = 4096, INTER = 12288 };
#endif
#else
#define PP_GATE "ffn_gate.weight"
#define PP_UP "ffn_up.weight"
#define PP_DOWN "ffn_down.weight"
enum { RANKS = 12, FIRST = 0, LAYERS = 3, HIDDEN = 4096, INTER = 12288 };
#endif

typedef struct {
    int fd;
    uint64_t base;
    const gguf_tensor_info *info;
} tensor_ref;

#ifdef GLM53F_PP_DENSE_STAGE
#define DENSE_STAGE_LABEL "glm53f_pp_" PP_COMPONENT "_stage"
#else
#define DENSE_STAGE_LABEL "glm53f_q2_dense_stage"
#endif
static void die(int rank, const char *message) {
    fprintf(stderr, "rank=%d " DENSE_STAGE_LABEL ": %s: %s\n", rank,
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
#ifdef GLM53F_PP_DENSE_STAGE
        for (ssize_t i = 0; i < put; ++i) {
            pp_hash ^= p[i]; pp_hash *= UINT64_C(1099511628211);
            pp_entry_hash ^= p[i]; pp_entry_hash *= UINT64_C(1099511628211);
        }
#endif
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
        tensor->info->dims[1] != INTER) return -1;
#ifdef GLM53F_PP_DENSE_STAGE
    pp_entry_hash = UINT64_C(1469598103934665603);
    for (int base = 0; base < rows; base += 64) {
        int n = rows - base < 64 ? rows - base : 64;
        uint64_t position = (uint64_t)(row0 + base) * bytes_per_row;
        size_t count = (size_t)n * bytes_per_row;
        if (read_exact(tensor, position, buffer, count) || write_all(out, buffer, count)) return -1;
        (void)posix_fadvise(tensor->fd, (off_t)(tensor->base + position), count, POSIX_FADV_DONTNEED);
    }
#else
    if (read_exact(tensor, (uint64_t)row0 * bytes_per_row, buffer, bytes) ||
        write_all(out, buffer, bytes)) return -1;
#endif
    fprintf(manifest, "%" PRIu64 " %u %s %d %d blk.%d.%s\n",
            *offset, tensor->info->type, ggml_type_name(tensor->info->type),
            rows, HIDDEN, layer, kind);
#ifdef GLM53F_PP_DENSE_STAGE
    fprintf(manifest, "# PAYLOAD offset=%" PRIu64 " bytes=%zu fnv1a=%016" PRIx64
        " source=%s rows=%d:%d columns=0:%d\n", *offset, bytes, pp_entry_hash,
        tensor->info->name.str, row0, row0 + rows, HIDDEN);
#endif
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
#ifdef GLM53F_PP_DENSE_STAGE
    pp_entry_hash = UINT64_C(1469598103934665603);
#endif
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
    fprintf(manifest, "%" PRIu64 " %u %s %d %d blk.%d.%s\n",
            begin, tensor->info->type, ggml_type_name(tensor->info->type),
            HIDDEN, columns, layer, PP_DOWN);
#ifdef GLM53F_PP_DENSE_STAGE
    fprintf(manifest, "# PAYLOAD offset=%" PRIu64 " bytes=%" PRIu64 " fnv1a=%016" PRIx64
        " source=%s rows=0:%d columns=%d:%d\n", begin, (uint64_t)HIDDEN * local_row,
        pp_entry_hash, tensor->info->name.str, HIDDEN, column0, column0 + columns);
#endif
    *offset += (uint64_t)HIDDEN * local_row;
    return 0;
}

static void model_identity(const char *path, char *out, size_t n) {
    struct stat st;
    if (stat(path, &st)) st.st_size = 0;
    snprintf(out, n, "model_bytes=%lld model=%s", (long long)st.st_size, path);
}

static void stage_header(char *out, size_t cap, int rank, const char *identity) {
#ifdef GLM53F_PP_DENSE_STAGE
    snprintf(out, cap, "# GLM53F_PP_" PP_COMPONENT "_V1 layout=pp3-tp4 world_rank=%d "
        "stage=%d tp_rank=%d tp_size=4 cuts=%d,%d layers=%d:%d %s\n",
        rank, pp_map.stage, pp_map.tp_rank, pp_config.cuts[0], pp_config.cuts[1],
        pp_map.first_layer, pp_map.end_layer, identity);
#else
    snprintf(out, cap, "# GLM53F_Q2_DENSE_V2 rank=%d ranks=12 layers=0:3 %s\n", rank, identity);
#endif
}

static int stage_complete(const char *manifest_path, const char *blob_path,
                          int rank, const char *identity) {
    FILE *manifest = fopen(manifest_path, "r");
    struct stat st;
    char line[8192], want[8192];
    int header = 0;
    uint64_t expected = UINT64_MAX;
#ifdef GLM53F_PP_DENSE_STAGE
    uint64_t expected_hash = 0; int has_hash = 0;
#endif
    if (!manifest || stat(blob_path, &st)) {
        if (manifest) fclose(manifest);
        return 0;
    }
    stage_header(want, sizeof(want), rank, identity);
    while (fgets(line, sizeof(line), manifest)) {
        unsigned long long bytes;
        if (!strcmp(line, want)) header = 1;
        if (sscanf(line, "# COMPLETE bytes=%llu", &bytes) == 1)
            expected = (uint64_t)bytes;
#ifdef GLM53F_PP_DENSE_STAGE
        unsigned long long hash;
        if (sscanf(line, "# COMPLETE bytes=%llu fnv1a=%llx", &bytes, &hash) == 2) {
            expected_hash = (uint64_t)hash; has_hash = 1;
        }
#endif
    }
    fclose(manifest);
#ifdef GLM53F_PP_DENSE_STAGE
    return header && expected == (uint64_t)st.st_size && has_hash &&
        !glm53f_pp_blob_verify(blob_path, expected, expected_hash);
#else
    return header && expected == (uint64_t)st.st_size;
#endif
}

int main(int argc, char **argv) {
    (void)gguf_type_name;
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
    int first = FIRST, end = LAYERS;
#ifdef GLM53F_PP_DENSE_STAGE
    pp_config = glm53f_parallel_default(); pp_config.layout = GLM53F_PP3_TP4;
    for (int i = 3; i < argc; ++i)
        if (glm53f_parallel_option(&pp_config, argc, argv, &i) != 1) die(rank, "pipeline option");
    if (pp_config.layout != GLM53F_PP3_TP4 ||
        glm53f_parallel_map_rank(&pp_config, rank, ranks, &pp_map)) die(rank, "PP layout");
    int low[2], high[2];
    MPI_Allreduce(pp_config.cuts, low, 2, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(pp_config.cuts, high, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (memcmp(low, high, sizeof(low))) die(rank, "inconsistent cuts");
    first = pp_map.first_layer > FIRST ? pp_map.first_layer : FIRST; end = pp_map.end_layer < LAYERS ? pp_map.end_layer : LAYERS;
    if (argc < 3 || ranks != 12 || strncmp(argv[2], "/local/", 7)) die(rank, "PP usage");
    if (first >= end) {
        printf("SENTINEL glm53f_pp_dense_stage=SKIP rank=%d\n", rank);
        MPI_Finalize(); return 0;
    }
#else
    if (argc != 3 || ranks != RANKS || strncmp(argv[2], "/local/", 7))
        die(rank, "usage: MODEL-00001-of-00004.gguf /local/STAGE_DIR");
#endif
    if (mkdir(argv[2], 0755) && errno != EEXIST) die(rank, "mkdir");
    snprintf(blob, sizeof(blob), "%s/rank%02d.blob", argv[2], rank);
    snprintf(manifest_path, sizeof(manifest_path), "%s/rank%02d.manifest",
             argv[2], rank);
    model_identity(argv[1], identity, sizeof(identity));
#ifdef GLM53F_PP_DENSE_STAGE
    uint64_t stamp = 0;
    if (!(g = gguf_open_multi(argv[1], 3)) || g->n_tensors != 1412 ||
        glm53f_pp_source_stamp(g, argv[1], &stamp)) die(rank, "GGUF source identity");
    snprintf(identity, sizeof(identity), "source_metadata_fnv1a=%016" PRIx64, stamp);
#endif
    if (stage_complete(manifest_path, blob, rank, identity)) {
        printf("SENTINEL " DENSE_STAGE_LABEL "=REUSE rank=%d\n", rank);
        gguf_close(g); MPI_Finalize();
        return 0;
    }
    snprintf(blob_tmp, sizeof(blob_tmp), "%s/.rank%02d.blob.tmp.%ld",
             argv[2], rank, (long)getpid());
    snprintf(manifest_tmp, sizeof(manifest_tmp), "%s/.rank%02d.manifest.tmp.%ld",
             argv[2], rank, (long)getpid());
#ifndef GLM53F_PP_DENSE_STAGE
    if (!(g = gguf_open_multi(argv[1], 3)) || g->n_tensors != 1412)
        die(rank, "GGUF metadata");
#endif
    if ((out = open(blob_tmp, O_CREAT | O_EXCL | O_WRONLY, 0644)) < 0 ||
        !(manifest = fopen(manifest_tmp, "wx"))) die(rank, "create output");
    char header[8192]; stage_header(header, sizeof(header), rank, identity); fputs(header, manifest);
#ifdef GLM53F_PP_DENSE_STAGE
    int row0 = pp_map.tp_rank * (INTER / RANKS), rows = INTER / RANKS;
#else
    int row0 = rank * (INTER / RANKS), rows = INTER / RANKS;
#endif
    /* Q8_0 (8.5 bits/value) is the widest supported type. */
#ifdef GLM53F_PP_DENSE_STAGE
    size_t max_rows = 64u * row_bytes(GGML_TYPE_Q8_0, HIDDEN);
#else
    size_t max_rows = (size_t)rows * row_bytes(GGML_TYPE_Q8_0, HIDDEN);
#endif
    size_t max_full = 64u * row_bytes(GGML_TYPE_Q8_0, INTER);
    size_t max_local = 64u * row_bytes(GGML_TYPE_Q8_0, rows);
    row_buffer = malloc(max_rows);
    input = malloc(max_full);
    output = malloc(max_local);
    if (!row_buffer || !input || !output) die(rank, "scratch allocation");
    for (int layer = first; layer < end; ++layer) {
        char name[128];
        snprintf(name, sizeof(name), "blk.%d." PP_GATE, layer);
        tensor_ref gate = find_tensor(g, name);
        snprintf(name, sizeof(name), "blk.%d." PP_UP, layer);
        tensor_ref up = find_tensor(g, name);
        snprintf(name, sizeof(name), "blk.%d." PP_DOWN, layer);
        tensor_ref down = find_tensor(g, name);
        if (!gate.info || !up.info || !down.info ||
            !supported(gate.info->type) || !supported(up.info->type) ||
            !supported(down.info->type) ||
            stage_rows(out, manifest, &offset, &gate, row0, rows, layer,
                       PP_GATE, row_buffer) ||
            stage_rows(out, manifest, &offset, &up, row0, rows, layer,
                       PP_UP, row_buffer) ||
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
#ifdef GLM53F_PP_DENSE_STAGE
    if (fprintf(manifest, "# COMPLETE bytes=%" PRIu64 " fnv1a=%016" PRIx64 "\n", offset, pp_hash) < 0 ||
#else
    if (fprintf(manifest, "# COMPLETE bytes=%" PRIu64 "\n", offset) < 0 ||
#endif
        fflush(manifest) || fsync(fileno(manifest)) || fsync(out) ||
        fclose(manifest) || close(out) || rename(blob_tmp, blob) ||
        rename(manifest_tmp, manifest_path)) die(rank, "publish");
    manifest = NULL;
    out = -1;
    printf("SENTINEL " DENSE_STAGE_LABEL "=OK rank=%d bytes=%" PRIu64 "\n",
           rank, offset);
    rc = 0;
    free(output);
    free(input);
    free(row_buffer);
    gguf_close(g);
    MPI_Finalize();
    return rc;
}
