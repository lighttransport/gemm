#define _POSIX_C_SOURCE 200809L
#include "ds41f_tensor.h"
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

int ds41f_tensor_load(const char *directory, const char *name, size_t bytes,
                      void **data)
{
    if (!directory || !name || !data || !bytes || strchr(name,'/')) return EINVAL;
    *data = NULL;
    char path[4096];
    int len = snprintf(path,sizeof path,"%s/%s.bin",directory,name);
    if (len < 0 || (size_t)len >= sizeof path) return ENAMETOOLONG;
    int fd = open(path,O_RDONLY);
    if (fd < 0) return errno;
    struct stat st;
    int rc = 0;
    if (fstat(fd,&st)) { rc = errno; close(fd); return rc; }
    if (!S_ISREG(st.st_mode) || st.st_size < 0 || (size_t)st.st_size != bytes) {
        close(fd); return EINVAL;
    }
    void *buffer = NULL;
    rc = posix_memalign(&buffer,256,bytes);
    if (rc) { close(fd); return rc; }
    /* Parallel first touch places each row range near its consuming OpenMP
     * workers instead of filling a single CMG's memory first. */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    for (size_t page=0;page<(bytes+4095)/4096;++page)
        ((char *)buffer)[page*4096]=0;
    #endif
    size_t offset = 0;
    while (offset < bytes) {
        size_t count = bytes-offset;
        if (count > 8*1024*1024) count = 8*1024*1024;
        ssize_t got = pread(fd,(char *)buffer+offset,count,(off_t)offset);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) { rc = got < 0 ? errno : EIO; break; }
        posix_fadvise(fd,(off_t)offset,got,POSIX_FADV_DONTNEED);
        offset += (size_t)got;
    }
    close(fd);
    if (rc) free(buffer); else *data = buffer;
    return rc;
}
