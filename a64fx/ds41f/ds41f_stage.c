/* Stage the two DeepSeek-V4.1 Engram row shards into node-local storage.
 * The source checkpoint is read through safetensors headers and bounded
 * pread/pwrite copies; no source tensor is materialized in memory. */
#define _GNU_SOURCE
#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define DS41F_LAYERS 2
#define DS41F_ROWS0 UINT64_C(384006168)
#define DS41F_ROWS1 UINT64_C(384016682)
#define DS41F_DIM 256u
#define DS41F_SCALE_BLOCK 32u
#define DS41F_CHUNK (8u * 1024u * 1024u)

typedef struct {
    int layer;
    uint64_t rows;
    const char *source;
    const char *weight_name;
    const char *scale_name;
} layer_desc;

static const layer_desc layers[DS41F_LAYERS] = {
    { 1, DS41F_ROWS0, "model-00047-of-00048.safetensors",
      "layers.1.engram.embed.weight", "layers.1.engram.embed.scale" },
    { 14, DS41F_ROWS1, "model-00048-of-00048.safetensors",
      "layers.14.engram.embed.weight", "layers.14.engram.embed.scale" }
};

static void usage(const char *p)
{
    fprintf(stderr, "usage: %s --model-dir DIR --stage-dir DIR --rank R --ranks N [--dry-run]\n", p);
}

static int mkdir_p(const char *path)
{
    char tmp[2048]; size_t n = strlen(path);
    if (!n || n >= sizeof tmp) return ENAMETOOLONG;
    memcpy(tmp, path, n + 1);
    for (char *p = tmp + 1; *p; ++p) {
        if (*p != '/') continue;
        *p = '\0';
        if (mkdir(tmp, 0755) && errno != EEXIST) return errno;
        *p = '/';
    }
    return mkdir(tmp, 0755) && errno != EEXIST ? errno : 0;
}

static int copy_range(int in, uint64_t in_off, int out, uint64_t out_off,
                      uint64_t bytes)
{
    unsigned char *buf = malloc(DS41F_CHUNK);
    if (!buf) return ENOMEM;
    uint64_t done = 0;
    while (done < bytes) {
        size_t want = (size_t)((bytes - done) < DS41F_CHUNK ?
                               (bytes - done) : DS41F_CHUNK);
        ssize_t nr;
        do { nr = pread(in, buf, want, (off_t)(in_off + done)); }
        while (nr < 0 && errno == EINTR);
        if (nr != (ssize_t)want) { int e = nr < 0 ? errno : EIO; free(buf); return e; }
        size_t written = 0;
        while (written < want) {
            ssize_t nw = pwrite(out, buf + written, want - written,
                                (off_t)(out_off + done + written));
            if (nw < 0 && errno == EINTR) continue;
            if (nw <= 0) { int e = nw < 0 ? errno : EIO; free(buf); return e; }
            written += (size_t)nw;
        }
        done += want;
    }
    free(buf);
    return 0;
}

static int validate_tensor(const st_context *st, const char *name,
                           uint64_t rows, uint64_t cols, const char *dtype,
                           int *index)
{
    int i = safetensors_find(st, name);
    if (i < 0) return ENOENT;
    const uint64_t *shape = safetensors_shape(st, i);
    if (safetensors_ndims(st, i) != 2 || shape[0] != rows || shape[1] != cols ||
        strcmp(safetensors_dtype(st, i), dtype) != 0)
        return EINVAL;
    *index = i;
    return 0;
}

static int stage_layer(const char *model, const char *stage, int rank, int nranks,
                       const layer_desc *d, FILE *manifest, int dry_run)
{
    char source[2048];
    int n = snprintf(source, sizeof source, "%s/%s", model, d->source);
    if (n < 0 || (size_t)n >= sizeof source) return ENAMETOOLONG;
    st_context *st = safetensors_open_header(source);
    if (!st) return errno ? errno : EIO;
    int wi = -1, si = -1;
    int rc = validate_tensor(st, d->weight_name, d->rows, DS41F_DIM,
                              "F8_E4M3", &wi);
    if (!rc) rc = validate_tensor(st, d->scale_name, d->rows,
                                  DS41F_DIM / DS41F_SCALE_BLOCK, "F8_E8M0", &si);
    if (rc) { safetensors_close(st); return rc; }

    uint64_t per = (d->rows + (uint64_t)nranks - 1) / (uint64_t)nranks;
    uint64_t first = per * (uint64_t)rank;
    uint64_t count = first < d->rows ? d->rows - first : 0;
    if (count > per) count = per;
    if (dry_run) {
        printf("DS41F_DRYRUN layer=%d source=%s rows=%" PRIu64 " first=%" PRIu64
               " weight_bytes=%" PRIu64 " scale_bytes=%" PRIu64 "\n", d->layer,
               source, count, first, count * DS41F_DIM,
               count * (DS41F_DIM / DS41F_SCALE_BLOCK));
        safetensors_close(st);
        return 0;
    }
    char wf[2048], sf[2048];
    snprintf(wf, sizeof wf, "%s/layer%d.weight", stage, d->layer);
    snprintf(sf, sizeof sf, "%s/layer%d.scale", stage, d->layer);
    int wfd = open(wf, O_CREAT | O_RDWR | O_TRUNC, 0644);
    int sfd = open(sf, O_CREAT | O_RDWR | O_TRUNC, 0644);
    if (wfd < 0 || sfd < 0) { rc = errno; if (wfd >= 0) close(wfd); if (sfd >= 0) close(sfd); safetensors_close(st); return rc; }
    uint64_t wb = count * DS41F_DIM, sb = count * (DS41F_DIM / DS41F_SCALE_BLOCK);
    if (ftruncate(wfd, (off_t)wb) || ftruncate(sfd, (off_t)sb)) rc = errno;
    if (!rc) {
        int in = open(source, O_RDONLY);
        if (in < 0) rc = errno;
        else { rc = copy_range(in, st->data_offset + st->tensors[wi].offset + first * DS41F_DIM, wfd, 0, wb); close(in); }
    }
    if (!rc) {
        int in = open(source, O_RDONLY);
        if (in < 0) rc = errno;
        else { rc = copy_range(in, st->data_offset + st->tensors[si].offset + first * (DS41F_DIM / DS41F_SCALE_BLOCK), sfd, 0, sb); close(in); }
    }
    if (!rc && (fsync(wfd) || fsync(sfd))) rc = errno;
    close(wfd); close(sfd); safetensors_close(st);
    if (!rc) fprintf(manifest, "layer=%d first=%" PRIu64 " rows=%" PRIu64
                     " weight_bytes=%" PRIu64 " scale_bytes=%" PRIu64
                     " weight=%s scale=%s source=%s\n", d->layer, first, count,
                     wb, sb, wf, sf, source);
    return rc;
}

int main(int argc, char **argv)
{
    const char *model = NULL, *stage = NULL; int rank = -1, nranks = -1, dry_run = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--model-dir") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--stage-dir") && i + 1 < argc) stage = argv[++i];
        else if (!strcmp(argv[i], "--rank") && i + 1 < argc) rank = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ranks") && i + 1 < argc) nranks = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dry-run")) dry_run = 1;
        else { usage(argv[0]); return 2; }
    }
    if (!model || !stage || rank < 0 || rank >= nranks || nranks < 1 || nranks > 12) {
        usage(argv[0]); return 2;
    }
    int rc = dry_run ? 0 : mkdir_p(stage);
    if (rc) { fprintf(stderr, "mkdir %s: %s\n", stage, strerror(rc)); return 1; }
    char mp[2048]; snprintf(mp, sizeof mp, "%s/manifest.rank%02d", stage, rank);
    FILE *manifest = dry_run ? stdout : fopen(mp, "w");
    if (!manifest) { perror(mp); return 1; }
    fprintf(manifest, "format=ds41f-engram-v1 rank=%d ranks=%d dim=%u\n", rank, nranks, DS41F_DIM);
    for (int i = 0; i < DS41F_LAYERS && !rc; ++i)
        rc = stage_layer(model, stage, rank, nranks, &layers[i], manifest, dry_run);
    if (!dry_run && (fclose(manifest) || rc)) { fprintf(stderr, "stage failed rc=%d\n", rc); return 1; }
    if (dry_run && rc) { fprintf(stderr, "stage dry-run failed rc=%d\n", rc); return 1; }
    printf("DS41F_STAGE rank=%d/%d stage=%s rows_per_layer~%" PRIu64 "\n", rank, nranks, stage,
           (DS41F_ROWS0 + (uint64_t)nranks - 1) / (uint64_t)nranks);
    return 0;
}
