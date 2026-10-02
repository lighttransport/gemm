#ifndef GLM53F_PP_BLOB_H
#define GLM53F_PP_BLOB_H
#include <stdint.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
/* Bounded validation before reusing a PP image. Does not keep source pages
 * resident or allocate an image-sized temporary copy. */
static inline int glm53f_pp_blob_verify(const char *path, uint64_t bytes, uint64_t expected) {
    const size_t chunk = 1u << 20;
    unsigned char *buffer = malloc(chunk);
    int fd = path ? open(path, O_RDONLY) : -1;
    if (!buffer || fd < 0) { free(buffer); if (fd >= 0) close(fd); return -1; }
    uint64_t offset = 0, hash = UINT64_C(1469598103934665603);
    int failed = 0;
    for (;;) {
        ssize_t n = read(fd, buffer, chunk);
        if (n < 0) { if (errno == EINTR) continue; failed = 1; break; }
        if (!n) break;
        if ((uint64_t)n > bytes - offset) { failed = 1; break; }
        for (ssize_t i = 0; i < n; ++i) { hash ^= buffer[i]; hash *= UINT64_C(1099511628211); }
        (void)posix_fadvise(fd, (off_t)offset, n, POSIX_FADV_DONTNEED);
        offset += (uint64_t)n;
    }
    failed |= offset != bytes || hash != expected;
    failed |= close(fd) != 0;
    free(buffer); return failed ? -1 : 0;
}
#endif
