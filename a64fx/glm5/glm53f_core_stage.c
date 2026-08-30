/* Copy one compact GLM-5.3F core blob to the rank's node-local /local store.
 * Uses bounded I/O and drops source/destination cache pages as it progresses.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int rank_id(void) {
    const char *s = getenv("PMIX_RANK");
    if (!s || !*s) s = getenv("PJM_MPI_RANK");
    if (!s || !*s) s = getenv("OMPI_COMM_WORLD_RANK");
    return s && *s ? atoi(s) : 0;
}

static int copy_file(const char *src, const char *dst) {
    const size_t chunk = 32u * 1024u * 1024u;
    char *buf = NULL;
    int in = -1, out = -1, rc = -1;
    off_t off = 0;
    if (!(buf = malloc(chunk)) || (in = open(src, O_RDONLY)) < 0 ||
        (out = open(dst, O_CREAT | O_EXCL | O_WRONLY, 0644)) < 0) goto done;
    for (;;) {
        ssize_t n = read(in, buf, chunk);
        if (n == 0) break;
        if (n < 0) { if (errno == EINTR) continue; goto done; }
        for (ssize_t done_bytes = 0; done_bytes < n;) {
            ssize_t w = write(out, buf + done_bytes, (size_t)(n - done_bytes));
            if (w < 0) { if (errno == EINTR) continue; goto done; }
            done_bytes += w;
        }
        (void)posix_fadvise(in, off, n, POSIX_FADV_DONTNEED);
        (void)posix_fadvise(out, off, n, POSIX_FADV_DONTNEED);
        off += n;
    }
    if (fsync(out)) goto done;
    rc = 0;
done:
    if (in >= 0) close(in);
    if (out >= 0) close(out);
    free(buf);
    return rc;
}

int main(int argc, char **argv) {
    char src_blob[4096], src_manifest[4096], dst_blob[4096], dst_manifest[4096];
    char tmp_blob[4096], tmp_manifest[4096];
    int rank = argc > 3 ? atoi(argv[3]) : rank_id();
    struct stat sb, sm;
    if (argc < 3 || rank < 0 || rank >= 12) {
        fprintf(stderr, "usage: %s CORE_DIR LOCAL_DIR [RANK]\n", argv[0]);
        return 2;
    }
    if (mkdir(argv[2], 0755) && errno != EEXIST) { perror("mkdir"); return 2; }
    snprintf(src_blob, sizeof(src_blob), "%s/rank%02d.core.blob", argv[1], rank);
    snprintf(src_manifest, sizeof(src_manifest), "%s/rank%02d.core.manifest", argv[1], rank);
    snprintf(dst_blob, sizeof(dst_blob), "%s/rank%02d.core.blob", argv[2], rank);
    snprintf(dst_manifest, sizeof(dst_manifest), "%s/rank%02d.core.manifest", argv[2], rank);
    snprintf(tmp_blob, sizeof(tmp_blob), "%s/.rank%02d.core.blob.tmp.%ld", argv[2], rank, (long)getpid());
    snprintf(tmp_manifest, sizeof(tmp_manifest), "%s/.rank%02d.core.manifest.tmp.%ld", argv[2], rank, (long)getpid());
    if (stat(src_blob, &sb) || stat(src_manifest, &sm) || sb.st_size <= 0 || sm.st_size <= 0) {
        fprintf(stderr, "rank=%d incomplete source core stage\n", rank); return 2;
    }
    if (!stat(dst_blob, &sb) && !stat(dst_manifest, &sm) && sb.st_size > 0 && sm.st_size > 0) {
        printf("SENTINEL glm53f_core_stage=REUSE rank=%d bytes=%lld\n", rank, (long long)sb.st_size);
        return 0;
    }
    unlink(tmp_blob); unlink(tmp_manifest);
    if (copy_file(src_blob, tmp_blob) || copy_file(src_manifest, tmp_manifest) ||
        rename(tmp_blob, dst_blob) || rename(tmp_manifest, dst_manifest)) {
        perror("stage"); unlink(tmp_blob); unlink(tmp_manifest); return 1;
    }
    printf("SENTINEL glm53f_core_stage=OK rank=%d bytes=%lld\n", rank, (long long)sb.st_size);
    return 0;
}
