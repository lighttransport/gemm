#ifndef GLM53F_PP_F32_H
#define GLM53F_PP_F32_H
#include "glm53f_pp_manifest.h"
#include <stdlib.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>
/* FP32 vocabulary slices have their own PP namespace and raw-byte digest. */
static inline int glm53f_pp_f32_load(const glm53f_dist *d, const char *stage,
        const char *component, const char *tensor, int row0, int rows,
        int columns, float **output) {
    if (!d || !d->initialized || !stage || !component || !tensor || !output || rows < 1 || row0 < 0 || columns < 1 ||
        (size_t)rows > SIZE_MAX / sizeof(float) / (size_t)columns) return -1;
    char manifest[4096], blob[4096];
    int n = snprintf(manifest, sizeof(manifest), "%s/rank%02d.manifest", stage, d->map.world_rank);
    int b = snprintf(blob, sizeof(blob), "%s/rank%02d.f32", stage, d->map.world_rank);
    if (n < 0 || n >= (int)sizeof(manifest) || b < 0 || b >= (int)sizeof(blob) ||
        glm53f_pp_manifest_check(manifest, component, d, d->map.first_layer, d->map.end_layer)) return -1;
    FILE *f = fopen(manifest, "r"); if (!f) return -1;
    char line[2048], name[256]; int first, end, cols, found = 0;
    unsigned long long bytes = 0, hash = 0;
    while (fgets(line, sizeof(line), f))
        if (sscanf(line, "# F32 tensor=%255s rows=%d:%d columns=%d bytes=%llu fnv1a=%llx",
                name, &first, &end, &cols, &bytes, &hash) == 6) {
            found = !strcmp(name, tensor) && first == row0 && end == row0 + rows && cols == columns;
            break;
        }
    if (fclose(f) || !found || bytes != (size_t)rows * columns * sizeof(float)) return -1;
    int fd = open(blob, O_RDONLY); struct stat st; float *data = NULL;
    if (fd < 0) return -1;
    if (fstat(fd, &st) || st.st_size < 0 || bytes != (uint64_t)st.st_size ||
        posix_memalign((void **)&data, 256, (size_t)bytes)) { close(fd); return -1; }
    uint64_t actual = UINT64_C(1469598103934665603); size_t offset = 0;
    while (offset < bytes) {
        size_t count = (size_t)bytes - offset; if (count > (1u << 20)) count = 1u << 20;
        unsigned char *p = (unsigned char *)data + offset;
        ssize_t got = pread(fd, p, count, (off_t)offset);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) { free(data); close(fd); return -1; }
        for (ssize_t i = 0; i < got; ++i) { actual ^= p[i]; actual *= UINT64_C(1099511628211); }
        (void)posix_fadvise(fd, (off_t)offset, got, POSIX_FADV_DONTNEED); offset += (size_t)got;
    }
    if (close(fd) || actual != hash) { free(data); return -1; }
    *output = data; return 0;
}
#endif
