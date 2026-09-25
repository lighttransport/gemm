/* Build one rank-owned Q8_0 GGUF image without faulting the model payload.
 *
 * The input is the first shard of a split GGUF.  gguf_open_multi(..., 3)
 * reads headers and keeps source file descriptors, while every payload byte
 * below is transferred with bounded pread/write calls.  Matrix rows are
 * tensor-parallel; the third dimension of GLM expert tensors is expert
 * parallel.  The resulting image is suitable for copying to /local and
 * loading into anonymous HBM memory by glm53f_q8_resident_load().
 */
#define _GNU_SOURCE
#define GGUF_LOADER_IMPLEMENTATION
#include "../../common/gguf_loader.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

enum { COPY_CHUNK = 64 * 1024 * 1024, DEFAULT_RANKS = 12 };

typedef struct {
    int mode;                 /* 0=replicated, 1=row TP, 2=expert EP */
    int expert;
    uint64_t source;
    uint64_t bytes;
    uint64_t rows0;
    uint64_t rows;
} slice;

static void usage(const char *name) {
    fprintf(stderr, "usage: %s FIRST-GGUF-SHARD OUTPUT-DIR RANK [RANKS=12] [--dry-run]\n", name);
}

static int parse_expert(const char *name) {
    const char *p = strstr(name, "ffn_gate_exps.weight");
    if (!p) p = strstr(name, "ffn_up_exps.weight");
    if (!p) p = strstr(name, "ffn_down_exps.weight");
    if (!p) return 0;
    return strstr(name, "blk.") == name;
}

static int copy_span(int in, int out, uint64_t source, uint64_t bytes,
                     uint64_t *hash) {
    unsigned char *buf = malloc(COPY_CHUNK);
    if (!buf) return -1;
    uint64_t done = 0;
    while (done < bytes) {
        size_t want = (size_t)((bytes - done) < COPY_CHUNK ? bytes - done : COPY_CHUNK);
        ssize_t n;
        do { n = pread(in, buf, want, (off_t)(source + done)); } while (n < 0 && errno == EINTR);
        if (n <= 0) { free(buf); errno = n == 0 ? EIO : errno; return -1; }
        size_t written = 0;
        while (written < (size_t)n) {
            ssize_t w = write(out, buf + written, (size_t)n - written);
            if (w < 0) { if (errno == EINTR) continue; free(buf); return -1; }
            if (w == 0) { free(buf); errno = EIO; return -1; }
            for (ssize_t i = 0; i < w; ++i) {
                *hash ^= buf[written + (size_t)i];
                *hash *= UINT64_C(1099511628211);
            }
            written += (size_t)w;
        }
        (void)posix_fadvise(in, (off_t)(source + done), (size_t)n, POSIX_FADV_DONTNEED);
        if (fdatasync(out)) { free(buf); return -1; }
        (void)posix_fadvise(out, 0, 0, POSIX_FADV_DONTNEED);
        done += (uint64_t)n;
    }
    free(buf);
    return 0;
}

static int tensor_slice(const gguf_context *g, uint64_t i, int rank, int ranks,
                        slice *s) {
    const gguf_tensor_info *t = &g->tensors[i];
    const uint64_t total = gguf_tensor_size(g, (int)i);
    memset(s, 0, sizeof(*s));
    s->source = g->tensor_file_offsets[i];
    s->bytes = total;
    if (parse_expert(t->name.str) && t->n_dims == 3 && t->dims[2] > 0) {
        const int experts = (int)t->dims[2];
        const int first = experts * rank / ranks;
        const int end = experts * (rank + 1) / ranks;
        const int count = end - first;
        if (!count) { s->bytes = 0; return 0; }
        /* Expert tensors are contiguous in GGML's dim-0-fastest layout. */
        const uint64_t expert_bytes = total / t->dims[2];
        s->mode = 2;
        s->expert = first;
        s->source += (uint64_t)first * expert_bytes;
        s->bytes = (uint64_t)count * expert_bytes;
        return 0;
    }
    if (t->n_dims == 2 && total >= 1024 * 1024) {
        const uint64_t rows = t->dims[1];
        const uint64_t r0 = rows * (uint64_t)rank / (uint64_t)ranks;
        const uint64_t r1 = rows * (uint64_t)(rank + 1) / (uint64_t)ranks;
        const uint64_t row_bytes = rows ? total / rows : 0;
        s->mode = 1;
        s->rows0 = r0;
        s->rows = r1 - r0;
        s->source += r0 * row_bytes;
        s->bytes = (r1 - r0) * row_bytes;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 4) { usage(argv[0]); return 2; }
    const char *model = argv[1], *out_dir = argv[2];
    const int rank = atoi(argv[3]);
    const int ranks = argc > 4 && argv[4][0] != '-' ? atoi(argv[4]) : DEFAULT_RANKS;
    int dry = 0;
    for (int i = 4; i < argc; ++i) if (!strcmp(argv[i], "--dry-run")) dry = 1;
    if (rank < 0 || rank >= ranks || ranks < 1 || ranks > 384) {
        fprintf(stderr, "invalid rank/ranks: %d/%d\n", rank, ranks); return 2;
    }

    gguf_context *g = gguf_open_multi(model, 3);
    if (!g) { perror("gguf_open_multi"); return 1; }
    char blob[4096], manifest[4096], blob_tmp[4096], manifest_tmp[4096];
    snprintf(blob, sizeof(blob), "%s/rank%02d.blob", out_dir, rank);
    snprintf(manifest, sizeof(manifest), "%s/rank%02d.manifest", out_dir, rank);
    snprintf(blob_tmp, sizeof(blob_tmp), "%s/.rank%02d.blob.tmp.%ld", out_dir, rank, (long)getpid());
    snprintf(manifest_tmp, sizeof(manifest_tmp), "%s/.rank%02d.manifest.tmp.%ld", out_dir, rank, (long)getpid());
    uint64_t image_bytes = 0, image_tensors = 0, mode_bytes[3] = {0, 0, 0};
    uint64_t mode_tensors[3] = {0, 0, 0};
    uint64_t hash = UINT64_C(1469598103934665603);
    int out = -1;
    FILE *mf = NULL;
    if (!dry) {
        if (mkdir(out_dir, 0755) && errno != EEXIST) { perror("mkdir output"); goto fail; }
        out = open(blob_tmp, O_CREAT | O_EXCL | O_WRONLY, 0644);
        mf = fopen(manifest_tmp, "wx");
        if (out < 0 || !mf) { perror("create image"); goto fail; }
        fprintf(mf, "# GLM53F_Q8_IMAGE_V1 rank=%d ranks=%d tensors=%" PRIu64 "\n",
                rank, ranks, g->n_tensors);
    }
    for (uint64_t i = 0; i < g->n_tensors; ++i) {
        slice s;
        if (tensor_slice(g, i, rank, ranks, &s)) goto fail;
        if (!s.bytes) continue;
        const gguf_tensor_info *t = &g->tensors[i];
        const int mode = s.mode;
        if (!dry && copy_span(g->tensor_fds[i], out, s.source, s.bytes, &hash)) {
            fprintf(stderr, "rank=%d tensor=%s read failed: %s\n", rank, t->name.str, strerror(errno));
            goto fail;
        }
        if (!dry) {
            fprintf(mf, "T %" PRIu64 " %s %u %u %" PRIu64 " %" PRIu64 " %" PRIu64
                    " %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %d %d\n",
                    i, t->name.str, t->type, t->n_dims, t->dims[0], t->dims[1],
                    t->dims[2], t->dims[3], s.source, s.bytes, image_bytes, mode, s.expert);
        }
        image_bytes += s.bytes;
        ++image_tensors;
        mode_bytes[mode] += s.bytes;
        ++mode_tensors[mode];
    }
    if (!dry) {
        fprintf(mf, "# COMPLETE bytes=%" PRIu64 " tensors=%" PRIu64 " fnv1a=%016" PRIx64 "\n",
                image_bytes, image_tensors, hash);
        if (fflush(mf) || fsync(fileno(mf)) || fdatasync(out) || fclose(mf) || close(out)) {
            mf = NULL; out = -1; perror("sync image"); goto fail;
        }
        mf = NULL; out = -1;
        if (rename(blob_tmp, blob) || rename(manifest_tmp, manifest)) { perror("publish image"); goto fail; }
    }
    printf("SENTINEL glm53f_q8_stage=%s rank=%d ranks=%d tensors=%" PRIu64
           " bytes=%" PRIu64 " replicated=%" PRIu64 "/%" PRIu64
           " row_tp=%" PRIu64 "/%" PRIu64 " expert_ep=%" PRIu64 "/%" PRIu64
           " hash=%016" PRIx64 "\n", dry ? "DRY_RUN" : "OK", rank, ranks,
           image_tensors, image_bytes, mode_bytes[0], mode_tensors[0],
           mode_bytes[1], mode_tensors[1], mode_bytes[2], mode_tensors[2], hash);
    gguf_close(g);
    return 0;
fail:
    if (mf) fclose(mf);
    if (out >= 0) close(out);
    unlink(blob_tmp); unlink(manifest_tmp);
    gguf_close(g);
    return 1;
}
