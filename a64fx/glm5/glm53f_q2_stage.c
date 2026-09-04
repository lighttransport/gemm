/* Stream the GLM-5.3-Flash mixed-IQ GGUF routed experts into one rank-local
 * image.  GGUF payloads are never mmap'ed or copied wholesale. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define GGUF_LOADER_IMPLEMENTATION
#include "../../common/gguf_loader.h"
#include "../../common/ggml_dequant.h"
#include <mpi.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

enum { NLAYERS = 45, NEXPERTS = 288, HIDDEN = 4096, INTER = 2048,
       PARTS = 8, PART_INTER = 256 };
static const int owner_offset[PARTS] = {0,1,3,4,6,7,9,10};

typedef struct {
    gguf_context *g;
    int fd;
    uint64_t base;
    const gguf_tensor_info *ti;
} tensor_ref;

static void die(int rank, const char *what) {
    fprintf(stderr, "rank=%d glm53f_q2_stage: %s: %s\n", rank, what,
            errno ? strerror(errno) : "contract failure");
    MPI_Abort(MPI_COMM_WORLD, 2);
}

static int write_all(int fd, const void *ptr, size_t bytes, uint64_t *hash) {
    const uint8_t *p = ptr;
    while (bytes) {
        ssize_t n = write(fd, p, bytes);
        if (n < 0) { if (errno == EINTR) continue; return -1; }
        if (hash) for (ssize_t i = 0; i < n; ++i) {
            *hash ^= p[i]; *hash *= UINT64_C(1099511628211);
        }
        p += n; bytes -= (size_t)n;
    }
    return 0;
}

static size_t row_bytes(uint32_t type, int columns) {
    if (type >= GGML_TYPE_COUNT || ggml_type_info[type].block_size <= 0 ||
        columns % ggml_type_info[type].block_size) return 0;
    return (size_t)(columns / ggml_type_info[type].block_size) *
           ggml_type_info[type].type_size;
}

static tensor_ref find_tensor(gguf_context *g, const char *name) {
    tensor_ref r = {g, -1, 0, NULL};
    for (uint64_t i = 0; i < g->n_tensors; ++i) if
        (g->tensors[i].name.str && !strcmp(g->tensors[i].name.str, name)) {
        r.ti = &g->tensors[i];
        r.fd = g->tensor_fds ? g->tensor_fds[i] : g->fd;
        r.base = g->tensor_file_offsets ? g->tensor_file_offsets[i]
                                        : g->data_offset + g->tensors[i].offset;
        break;
    }
    return r;
}

static int supported_iq(uint32_t t) {
    return t == GGML_TYPE_IQ2_XS || t == GGML_TYPE_IQ3_XXS ||
           t == GGML_TYPE_IQ4_XS;
}

static int exact_read(const tensor_ref *r, uint64_t relative,
                      void *dst, size_t bytes) {
    uint8_t *p = dst;
    size_t left = bytes;
    while (left) {
        ssize_t n = pread(r->fd, p, left, (off_t)(r->base + relative));
        if (n < 0) { if (errno == EINTR) continue; return -1; }
        if (!n) { errno = EIO; return -1; }
        p += n; left -= (size_t)n; relative += (uint64_t)n;
    }
    return 0;
}

static int owned_part(int expert, int rank) {
    for (int p = 0; p < PARTS; ++p)
        if ((expert + owner_offset[p]) % 12 == rank) return p;
    return -1;
}

static int stage_complete(const char *manifest_path, const char *blob_path,
                          int first_layer, int layer_count, int rank) {
    FILE *f = fopen(manifest_path, "r");
    struct stat st;
    char line[2048];
    uint64_t bytes = UINT64_MAX;
    int header = 0;
    if (!f || stat(blob_path, &st)) { if (f) fclose(f); return 0; }
    while (fgets(line, sizeof(line), f)) {
        int got_rank, begin, end;
        unsigned long long got_bytes;
        if (sscanf(line, "# GLM53F_Q2_DECODE_V1 rank=%d ranks=12 expert_parts=8 "
                   "offsets=0,1,3,4,6,7,9,10 layers=%d:%d",
                   &got_rank, &begin, &end) == 3)
            header = got_rank == rank && begin == first_layer &&
                     end == first_layer + layer_count;
        if (sscanf(line, "# COMPLETE bytes=%llu", &got_bytes) == 1)
            bytes = (uint64_t)got_bytes;
    }
    fclose(f);
    return header && bytes == (uint64_t)st.st_size;
}

static int put_gate_up(int out, FILE *manifest, uint64_t *offset,
                       uint64_t *hash, const tensor_ref *gate,
                       const tensor_ref *up, int layer, int expert, int part,
                       void *buffer, int dry) {
    size_t rb = row_bytes(gate->ti->type, HIDDEN);
    size_t bytes = PART_INTER * rb;
    uint64_t expert_base = (uint64_t)expert * INTER * rb;
    uint64_t begin = *offset;
    if (!rb || gate->ti->type != up->ti->type) return -1;
    if (!dry) {
        if (exact_read(gate, expert_base + (uint64_t)part * PART_INTER * rb,
                       buffer, bytes) || write_all(out, buffer, bytes, hash) ||
            exact_read(up, expert_base + (uint64_t)part * PART_INTER * rb,
                       buffer, bytes) || write_all(out, buffer, bytes, hash)) return -1;
        posix_fadvise(gate->fd, (off_t)(gate->base + expert_base),
                      (off_t)(INTER * rb), POSIX_FADV_DONTNEED);
        posix_fadvise(up->fd, (off_t)(up->base + expert_base),
                      (off_t)(INTER * rb), POSIX_FADV_DONTNEED);
        fprintf(manifest, "%" PRIu64 " %s 2 %d %d part=%d source_expert=%d "
                "model.language_model.layers.%d.mlp.experts.%d.gate_up_fused.weight\n",
                begin, ggml_type_name(gate->ti->type), 2 * PART_INTER, HIDDEN,
                part, expert, layer, expert);
    }
    *offset += 2 * bytes;
    return 0;
}

static int put_down(int out, FILE *manifest, uint64_t *offset, uint64_t *hash,
                    const tensor_ref *down, int layer, int expert, int part,
                    uint8_t *input, uint8_t *output, int dry) {
    size_t full_rb = row_bytes(down->ti->type, INTER);
    size_t part_rb = row_bytes(down->ti->type, PART_INTER);
    const int chunk_rows = 128;
    uint64_t expert_base = (uint64_t)expert * HIDDEN * full_rb;
    uint64_t begin = *offset;
    if (!full_rb || !part_rb || full_rb != PARTS * part_rb) return -1;
    if (!dry) {
        for (int r0 = 0; r0 < HIDDEN; r0 += chunk_rows) {
            int nr = HIDDEN - r0 < chunk_rows ? HIDDEN - r0 : chunk_rows;
            size_t in_bytes = (size_t)nr * full_rb;
            if (exact_read(down, expert_base + (uint64_t)r0 * full_rb,
                           input, in_bytes)) return -1;
            for (int r = 0; r < nr; ++r)
                memcpy(output + (size_t)r * part_rb,
                       input + (size_t)r * full_rb + (size_t)part * part_rb,
                       part_rb);
            if (write_all(out, output, (size_t)nr * part_rb, hash)) return -1;
        }
        posix_fadvise(down->fd, (off_t)(down->base + expert_base),
                      (off_t)(HIDDEN * full_rb), POSIX_FADV_DONTNEED);
        fprintf(manifest, "%" PRIu64 " %s 2 %d %d part=%d source_expert=%d "
                "model.language_model.layers.%d.mlp.experts.%d.down_proj.weight\n",
                begin, ggml_type_name(down->ti->type), HIDDEN, PART_INTER,
                part, expert, layer, expert);
    }
    *offset += (uint64_t)HIDDEN * part_rb;
    return 0;
}

int main(int argc, char **argv) {
    int rank, ranks, dry = 0, out = -1, first_layer = 3, layer_count = NLAYERS - 3;
    char blob[4096], manifest_path[4096], blob_tmp[4096], manifest_tmp[4096];
    gguf_context *g = NULL;
    FILE *manifest = NULL;
    uint8_t *buf = NULL, *in = NULL, *packed = NULL;
    uint64_t offset = 0, hash = UINT64_C(1469598103934665603);
    MPI_Init(&argc, &argv); MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    for (int a = 3; a < argc; ++a) {
        if (!strcmp(argv[a], "--dry-run")) dry = 1;
        else if (!strcmp(argv[a], "--first-layer") && a + 1 < argc)
            first_layer = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--layers") && a + 1 < argc)
            layer_count = atoi(argv[++a]);
        else die(rank, "unknown option");
    }
    if (argc < 3 || ranks != 12 || first_layer < 3 || layer_count < 1 ||
        first_layer + layer_count > NLAYERS)
        die(rank, "usage: MODEL-00001-of-00004.gguf STAGE_DIR [--dry-run] [--first-layer N] [--layers N]");
    if (mkdir(argv[2], 0755) && errno != EEXIST) die(rank, "mkdir stage");
    snprintf(blob, sizeof(blob), "%s/rank%02d.blob", argv[2], rank);
    snprintf(manifest_path, sizeof(manifest_path), "%s/rank%02d.manifest", argv[2], rank);
    snprintf(blob_tmp, sizeof(blob_tmp), "%s/.rank%02d.blob.tmp.%ld", argv[2], rank, (long)getpid());
    snprintf(manifest_tmp, sizeof(manifest_tmp), "%s/.rank%02d.manifest.tmp.%ld", argv[2], rank, (long)getpid());
    if (!dry && stage_complete(manifest_path, blob, first_layer, layer_count, rank)) {
        printf("SENTINEL glm53f_q2_stage=REUSE rank=%d\n", rank);
        MPI_Finalize(); return 0;
    }
    g = gguf_open_multi(argv[1], 3);
    if (!g || g->n_tensors != 1412) die(rank, "GGUF metadata contract");
    if (!dry) {
        out = open(blob_tmp, O_CREAT | O_EXCL | O_WRONLY, 0644);
        manifest = fopen(manifest_tmp, "wx");
        if (out < 0 || !manifest) die(rank, "create stage output");
        fprintf(manifest, "# GLM53F_Q2_DECODE_V1 rank=%d ranks=12 expert_parts=8 "
                "offsets=0,1,3,4,6,7,9,10 layers=%d:%d\n", rank,
                first_layer, first_layer + layer_count);
    }
    buf = malloc(1u << 20); in = malloc(1u << 20); packed = malloc(1u << 20);
    if (!buf || !in || !packed) die(rank, "scratch allocation");
    for (int layer = first_layer; layer < first_layer + layer_count; ++layer) {
        char name[128];
        snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps.weight", layer);
        tensor_ref gate = find_tensor(g, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_up_exps.weight", layer);
        tensor_ref up = find_tensor(g, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_down_exps.weight", layer);
        tensor_ref down = find_tensor(g, name);
        if (!gate.ti || !up.ti || !down.ti || !supported_iq(gate.ti->type) ||
            !supported_iq(up.ti->type) || !supported_iq(down.ti->type) ||
            gate.ti->n_dims != 3 || down.ti->n_dims != 3 ||
            gate.ti->dims[0] != HIDDEN || gate.ti->dims[1] != INTER ||
            gate.ti->dims[2] != NEXPERTS || down.ti->dims[0] != INTER ||
            down.ti->dims[1] != HIDDEN || down.ti->dims[2] != NEXPERTS)
            die(rank, "expert tensor contract");
        for (int expert = 0; expert < NEXPERTS; ++expert) {
            int part = owned_part(expert, rank);
            if (part < 0) continue;
            if (put_gate_up(out, manifest, &offset, &hash, &gate, &up,
                            layer, expert, part, buf, dry) ||
                put_down(out, manifest, &offset, &hash, &down,
                         layer, expert, part, in, packed, dry))
                die(rank, "expert payload");
        }
        if (!dry && (layer & 3) == 3) {
            fdatasync(out); posix_fadvise(out, 0, 0, POSIX_FADV_DONTNEED);
        }
        fprintf(stdout, "GLM53F_Q2_STAGE rank=%d layer=%d bytes=%" PRIu64 "\n",
                rank, layer, offset); fflush(stdout);
    }
    if (!dry) {
        fprintf(manifest, "# COMPLETE bytes=%" PRIu64 " fnv1a=%016" PRIx64 "\n", offset, hash);
        if (fflush(manifest) || fsync(fileno(manifest)) || fdatasync(out))
            die(rank, "sync stage");
        fclose(manifest); manifest = NULL; close(out); out = -1;
        if (rename(blob_tmp, blob) || rename(manifest_tmp, manifest_path))
            die(rank, "commit stage");
    }
    printf("SENTINEL glm53f_q2_stage=%s rank=%d bytes=%" PRIu64 " hash=%016" PRIx64 "\n",
           dry ? "DRY_RUN" : "OK", rank, offset, hash);
    free(packed); free(in); free(buf); gguf_close(g); MPI_Finalize(); return 0;
}
