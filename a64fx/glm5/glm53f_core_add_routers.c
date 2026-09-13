/* Add missing replicated target routers to an existing NODE-LOCAL core image.
 * Old construction traces did not include MoE router reads. Append only missing
 * records, sync each bounded tensor before publishing its manifest entry, and
 * leave the original checkpoint/core image on shared storage unchanged.
 * Usage: glm53f_core_add_routers MODEL LOCAL_CORE_DIR RANK
 */
#define _GNU_SOURCE
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "../../common/glm53f_safetensors.h"
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv) {
    int rank, fd = -1, added = 0, rc = 1;
    FILE *manifest = NULL;
    glm53f_st_context *st = NULL;
    void *buffer = NULL;
    char path[4096], line[2048];
    unsigned char present[42][2] = {{0}};
    if (argc != 4 || strncmp(argv[2], "/local/", 7) ||
        (rank = atoi(argv[3])) < 0 || rank >= 12) {
        fprintf(stderr, "usage: %s MODEL /local/CORE_DIR RANK\n", argv[0]);
        return 2;
    }
    snprintf(path, sizeof(path), "%s/rank%02d.core.manifest", argv[2], rank);
    if (!(manifest = fopen(path, "a+"))) goto done;
    rewind(manifest);
    while (fgets(line, sizeof(line), manifest)) {
        int layer; char suffix[128]; size_t a, b; unsigned long long off;
        if (sscanf(line, "R model.language_model.layers.%d.mlp.gate.%127s %zu %zu %llu",
            &layer, suffix, &a, &b, &off) == 5 && layer >= 3 && layer < 45 && !a) {
            if (!strcmp(suffix, "weight") && b == 288u * 4096u * 2u) present[layer - 3][0] = 1;
            if (!strcmp(suffix, "e_score_correction_bias") && b == 288u * 4u) present[layer - 3][1] = 1;
        }
    }
    snprintf(path, sizeof(path), "%s/rank%02d.core.blob", argv[2], rank);
    if ((fd = open(path, O_WRONLY)) < 0 || !(buffer = malloc(288u * 4096u * 2u))) goto done;
    /* This utility reads the source checkpoint, never the incomplete repack. */
    unsetenv("GLM53F_REPACK_DIR"); unsetenv("GLM53F_REPACK_TRACE_ONLY");
    if (!(st = glm53f_st_open(argv[1]))) goto done;
    for (int layer = 3; layer < 45; ++layer) for (int which = 0; which < 2; ++which) {
        if (present[layer - 3][which]) continue;
        char name[256];
        size_t bytes = which ? 288u * 4u : 288u * 4096u * 2u;
        snprintf(name, sizeof(name), "model.language_model.layers.%d.mlp.gate.%s", layer,
            which ? "e_score_correction_bias" : "weight");
        const st_tensor_info *t = glm53f_st_find(st, name, NULL);
        if (!t || t->nbytes != bytes || glm53f_st_read(st, name, 0, buffer, bytes)) goto done;
        off_t end = lseek(fd, 0, SEEK_END);
        if (end < 0) goto done;
        off_t offset = (end + 255) & ~(off_t)255;
        if (lseek(fd, offset, SEEK_SET) < 0) goto done;
        for (size_t n = 0; n < bytes;) {
            ssize_t w = write(fd, (char *)buffer + n, bytes - n);
            if (w < 0 && errno == EINTR) continue;
            if (w <= 0) goto done;
            n += (size_t)w;
        }
        if (fdatasync(fd)) goto done;
        (void)posix_fadvise(fd, offset, bytes, POSIX_FADV_DONTNEED);
        if (fprintf(manifest, "R %s 0 %zu %lld\n", name, bytes, (long long)offset) < 0 ||
            fflush(manifest) || fsync(fileno(manifest))) goto done;
        ++added;
    }
    printf("GLM53F_CORE_ROUTERS rank=%d added=%d PASS\n", rank, added);
    rc = 0;
done:
    if (rc) perror("add routers");
    if (fd >= 0) close(fd);
    if (manifest) fclose(manifest);
    if (st) glm53f_st_close(st);
    free(buffer);
    return rc;
}
