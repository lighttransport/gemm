#ifndef GLM53F_STATE_IO_H
#define GLM53F_STATE_IO_H
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>

/* Diagnostic byte-for-byte stream comparison. Bounded I/O and explicit
 * writeback avoid accumulating a second large cache in HBM during checks. */
typedef struct glm53f_state_io {
    FILE *file;
    int compare, failed;
    uint64_t offset, flushed;
    unsigned char buffer[65536];
} glm53f_state_io;

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
#endif
