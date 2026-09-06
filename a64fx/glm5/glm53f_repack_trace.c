/* Offline compact core repacker for a traced GLM-5.3F 12-way target rank.
 *
 * Usage: glm53f_repack_trace MODEL_DIR rankNN.trace OUT_DIR RANK
 *
 * The trace is produced by GLM53F_REPACK_TRACE_DIR together with
 * GLM53F_REPACK_TRACE_ONLY=1.  It contains exactly the bounded reads made by
 * one rank while constructing the target graph.  The resulting blob preserves
 * source slices, including non-contiguous column reads, without duplicating
 * the 306 GiB checkpoint.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "../../common/glm53f_safetensors.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
    char kind;
    char name[512];
    size_t a, b, c;
} request;

static int seen(const request *v, size_t n, const request *r) {
    for (size_t i = 0; i < n; ++i)
        if (v[i].kind == r->kind && v[i].a == r->a && v[i].b == r->b &&
            v[i].c == r->c && !strcmp(v[i].name, r->name)) return 1;
    return 0;
}

static int copy_out(int fd, uint64_t *offset, const void *src, size_t n) {
    uint64_t aligned = (*offset + 255u) & ~UINT64_C(255);
    size_t remaining = n;
    if (aligned != *offset && lseek(fd, (off_t)aligned, SEEK_SET) < 0) return -1;
    const unsigned char *p = src;
    while (remaining) {
        ssize_t w = write(fd, p, remaining);
        if (w < 0) { if (errno == EINTR) continue; return -1; }
        p += w;
        remaining -= (size_t)w;
    }
    *offset = aligned + n;
    return 0;
}

int main(int argc, char **argv) {
    FILE *tf = NULL, *mf = NULL;
    request *requests = NULL, r;
    size_t nreq = 0, cap = 0;
    glm53f_st_context *st = NULL;
    char blob[4096], manifest[4096], btmp[4096], mtmp[4096];
    int fd = -1, rank;
    uint64_t offset = 0;

    if (argc != 5 || (rank = atoi(argv[4])) < 0 || rank >= 12) {
        fprintf(stderr, "usage: %s MODEL_DIR rankNN.trace OUT_DIR RANK\n", argv[0]);
        return 2;
    }
    if (mkdir(argv[3], 0755) && errno != EEXIST) { perror("mkdir"); return 2; }
    if (snprintf(blob, sizeof(blob), "%s/rank%02d.core.blob", argv[3], rank) >= (int)sizeof(blob) ||
        snprintf(manifest, sizeof(manifest), "%s/rank%02d.core.manifest", argv[3], rank) >= (int)sizeof(manifest) ||
        snprintf(btmp, sizeof(btmp), "%s/.rank%02d.core.blob.tmp.%ld", argv[3], rank, (long)getpid()) >= (int)sizeof(btmp) ||
        snprintf(mtmp, sizeof(mtmp), "%s/.rank%02d.core.manifest.tmp.%ld", argv[3], rank, (long)getpid()) >= (int)sizeof(mtmp)) return 2;
    if (!access(blob, F_OK) && !access(manifest, F_OK)) {
        fprintf(stderr, "rank %d repack already complete: %s\n", rank, blob);
        return 0;
    }
    if (!(tf = fopen(argv[2], "r"))) { perror(argv[2]); return 2; }
    while (fscanf(tf, " %c %511s %zu %zu %zu", &r.kind, r.name, &r.a, &r.b, &r.c) == 5) {
        if (r.kind != 'R' && r.kind != 'C') { fprintf(stderr, "bad trace record\n"); goto fail; }
        if (seen(requests, nreq, &r)) continue;
        if (nreq == cap) {
            size_t next = cap ? cap * 2 : 1024;
            request *p = realloc(requests, next * sizeof(*p));
            if (!p) goto fail;
            requests = p; cap = next;
        }
        requests[nreq++] = r;
    }
    fclose(tf); tf = NULL;
    if (!(st = glm53f_st_open(argv[1]))) { fprintf(stderr, "open model failed\n"); goto fail; }
    if ((fd = open(btmp, O_CREAT | O_EXCL | O_WRONLY, 0644)) < 0 || !(mf = fopen(mtmp, "w"))) {
        perror("output"); goto fail;
    }
    fprintf(mf, "# GLM53F_CORE_REPACK rank=%d requests=%zu\n", rank, nreq);
    for (size_t i = 0; i < nreq; ++i) {
        size_t bytes;
        void *buf;
        uint64_t start;
        const st_tensor_info *t = glm53f_st_find(st, requests[i].name, NULL);
        if (!t) { fprintf(stderr, "missing %s\n", requests[i].name); goto fail; }
        if (requests[i].kind == 'R') {
            bytes = requests[i].b;
            buf = malloc(bytes);
            if (!buf || glm53f_st_read(st, requests[i].name, requests[i].a, buf, bytes)) {
                free(buf); fprintf(stderr, "read failed %s\n", requests[i].name); goto fail;
            }
            if (copy_out(fd, &offset, buf, bytes)) { free(buf); goto fail; }
            start = offset - bytes;
            fprintf(mf, "R %s %zu %zu %" PRIu64 "\n", requests[i].name, requests[i].a, bytes, start);
            free(buf);
        } else {
            if (!requests[i].a || t->nbytes % requests[i].a) { fprintf(stderr, "bad columns %s\n", requests[i].name); goto fail; }
            bytes = (t->nbytes / requests[i].a) * requests[i].c;
            buf = malloc(bytes);
            if (!buf || glm53f_st_read_columns(st, requests[i].name, requests[i].a,
                                                 requests[i].b, requests[i].c, buf)) {
                free(buf); fprintf(stderr, "column read failed %s\n", requests[i].name); goto fail;
            }
            if (copy_out(fd, &offset, buf, bytes)) { free(buf); goto fail; }
            start = offset - bytes;
            fprintf(mf, "C %s %zu %zu %zu %" PRIu64 "\n", requests[i].name,
                    requests[i].a, requests[i].b, requests[i].c, start);
            free(buf);
        }
        if ((i % 32) == 31) { fsync(fd); fflush(mf); }
    }
    fprintf(mf, "# blob_bytes %" PRIu64 "\n", offset);
    if (fsync(fd) || fclose(mf) || close(fd) || rename(btmp, blob) || rename(mtmp, manifest)) goto fail;
    glm53f_st_close(st); free(requests);
    fprintf(stderr, "SENTINEL glm53f_repack_trace=OK rank=%d requests=%zu bytes=%" PRIu64 "\n", rank, nreq, offset);
    return 0;
fail:
    if (tf) fclose(tf);
    if (mf) fclose(mf);
    if (fd >= 0) close(fd);
    if (st) glm53f_st_close(st);
    free(requests);
    unlink(btmp); unlink(mtmp);
    return 1;
}
