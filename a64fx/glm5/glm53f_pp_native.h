#ifndef GLM53F_PP_NATIVE_H
#define GLM53F_PP_NATIVE_H
#include "glm53f_iq_bridge.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
/* Load a PP tensor only after shape, byte count, bounds and payload digest
 * validation. Repacking follows hashing of the original native bytes. */
static inline int glm53f_pp_native_load(int fd, const char *manifest,
        const char *wanted, int expected_type, int rows, int columns,
        int allow_panel, uint8_t **output, int *output_type) {
    if (fd < 0 || !manifest || !wanted || !output || !output_type || rows < 1 || columns < 1) return -1;
    FILE *f = fopen(manifest, "r"); if (!f) return -1;
    char line[2048], name[256], type_name[32];
    unsigned type = 0; int r = 0, c = 0, found = 0, have_hash = 0;
    unsigned long long offset = 0, bytes = 0, hash = 0;
    while (fgets(line, sizeof(line), f)) {
        unsigned long long o, b, h;
        if (!found && sscanf(line, "%llu %u %31s %d %d %255s", &o, &type, type_name, &r, &c, name) == 6 && !strcmp(name, wanted)) {
            offset = o; found = 1;
        } else if (found && sscanf(line, "# PAYLOAD offset=%llu bytes=%llu fnv1a=%llx", &o, &b, &h) == 3 && o == offset) {
            bytes = b; hash = h; have_hash = 1; break;
        }
    }
    if (fclose(f) || !found || !have_hash || r != rows || c != columns ||
        (expected_type >= 0 && type != (unsigned)expected_type) || !glm53f_native_type_supported((int)type)) return -1;
    size_t rb = glm53f_native_row_size((int)type, columns);
    struct stat st;
    if (!rb || (size_t)rows > SIZE_MAX / rb || bytes != (size_t)rows * rb ||
        fstat(fd, &st) || st.st_size < 0 || offset > (uint64_t)st.st_size || bytes > (uint64_t)st.st_size - offset) return -1;
    uint8_t *p = NULL;
    if (posix_memalign((void **)&p, 256, (size_t)bytes)) return -1;
    size_t done = 0; uint64_t actual = UINT64_C(1469598103934665603);
    while (done < bytes) {
        size_t count = (size_t)bytes - done; if (count > (1u << 20)) count = 1u << 20;
        ssize_t n = pread(fd, p + done, count, (off_t)(offset + done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { free(p); return -1; }
        for (ssize_t i = 0; i < n; ++i) { actual ^= p[done + i]; actual *= UINT64_C(1099511628211); }
        (void)posix_fadvise(fd, (off_t)(offset + done), n, POSIX_FADV_DONTNEED);
        done += (size_t)n;
    }
    if (actual != hash) { free(p); return -1; }
    uint8_t *packed = NULL; int packed_type = (int)type;
    int rc = allow_panel ? glm53f_native_repack((int)type, p, rows, columns, &packed, &packed_type) :
        glm53f_native_repack_rowwise((int)type, p, rows, columns, &packed, &packed_type);
    if (rc) { free(p); return -1; }
    if (packed) { free(p); p = packed; }
    *output = p; *output_type = packed_type; return 0;
}
#endif
