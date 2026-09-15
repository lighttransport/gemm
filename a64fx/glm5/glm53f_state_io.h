#ifndef GLM53F_STATE_IO_H
#define GLM53F_STATE_IO_H
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <math.h>

/* Diagnostic byte-for-byte stream comparison. Bounded I/O and explicit
 * writeback avoid accumulating a second large cache in HBM during checks. */
typedef struct glm53f_state_io {
    FILE *file;
    int compare, failed;
    uint64_t offset, flushed;
    unsigned char buffer[65536];
} glm53f_state_io;

/* compare=2: typed floating-point diagnostics. Integers/metadata still use
 * byte-exact comparison; this does not weaken the original compare=1 gate. */
static inline int glm53f_state_io_floats(glm53f_state_io *io, const void *data,
                                        size_t bytes, const char *name);

static inline int glm53f_state_io_bytes(glm53f_state_io *io, const void *data,
                                       size_t bytes, const char *name) {
    if (!io || io->failed) return -1;
    const unsigned char *p = data;
    while (bytes) {
        size_t n = bytes < sizeof(io->buffer) ? bytes : sizeof(io->buffer);
        if (io->compare) {
            if (fread(io->buffer, 1, n, io->file) != n || memcmp(io->buffer, p, n)) {
                fprintf(stderr, "GLM53F_STATE_MISMATCH field=%s offset=%llu\n",
                        name, (unsigned long long)io->offset);
                io->failed = 1; return -1;
            }
        } else if (fwrite(p, 1, n, io->file) != n) {
            io->failed = 1; return -1;
        }
        io->offset += n; p += n; bytes -= n;
        if (io->offset - io->flushed >= 1048576) {
            if (!io->compare && (fflush(io->file) || fsync(fileno(io->file)))) {
                io->failed = 1; return -1;
            }
            posix_fadvise(fileno(io->file), (off_t)io->flushed,
                          (off_t)(io->offset - io->flushed), POSIX_FADV_DONTNEED);
            io->flushed = io->offset;
        }
    }
    return 0;
}

static inline int glm53f_state_io_floats(glm53f_state_io *io, const void *data,
                                        size_t bytes, const char *name) {
    if (!io || io->failed || bytes % sizeof(float)) return -1;
    if (io->compare != 2) return glm53f_state_io_bytes(io, data, bytes, name);
    const unsigned char *p = data;
    double error = 0, norm = 0, maximum = 0;
    while (bytes) {
        size_t n = bytes < sizeof(io->buffer) ? bytes : sizeof(io->buffer);
        if (fread(io->buffer, 1, n, io->file) != n) { io->failed = 1; return -1; }
        for (size_t i = 0; i < n; i += sizeof(float)) {
            float a, b;
            memcpy(&a, p + i, sizeof(a)); memcpy(&b, io->buffer + i, sizeof(b));
            if (!isfinite(a) || !isfinite(b)) { io->failed = 1; return -1; }
            double d = (double)a - b;
            error += d * d; norm += (double)b * b;
            if (fabs(d) > maximum) maximum = fabs(d);
        }
        io->offset += n; p += n; bytes -= n;
        if (io->offset - io->flushed >= 1048576) {
            posix_fadvise(fileno(io->file), (off_t)io->flushed,
                (off_t)(io->offset - io->flushed), POSIX_FADV_DONTNEED);
            io->flushed = io->offset;
        }
    }
    double relative = norm > 0 ? sqrt(error / norm) : error == 0 ? 0 : INFINITY;
    fprintf(stderr, "GLM53F_STATE_NUMERIC field=%s offset=%llu rel_l2=%.9g max_abs=%.9g %s\n",
        name, (unsigned long long)io->offset, relative, maximum, relative <= 1e-3 ? "PASS" : "FAIL");
    if (relative > 1e-3) io->failed = 1;
    return io->failed ? -1 : 0;
}

static inline int glm53f_state_io_indices(glm53f_state_io *io, const int *data,
                                         size_t count, const char *name) {
    if (!io || io->failed) return -1;
    if (io->compare != 2) return glm53f_state_io_bytes(io, data, count*sizeof(int), name);
    size_t different = 0;
    for (size_t i = 0; i < count; ++i) {
        int ref;
        if (fread(&ref,sizeof(ref),1,io->file) != 1) { io->failed=1; return -1; }
        different += ref != data[i]; io->offset += sizeof(ref);
    }
    fprintf(stderr,"GLM53F_STATE_INDICES field=%s changed=%zu count=%zu\n",name,different,count);
    return 0;
}
#endif
