#ifndef K3_RUNTIME_H
#define K3_RUNTIME_H

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#ifndef MPOL_INTERLEAVE
#define MPOL_INTERLEAVE 3
#endif

/* Interleave future anonymous allocations over the NUMA nodes containing CPUs
 * in this rank's affinity mask.  This excludes A64FX assistant-core nodes. */
static inline unsigned long k3_apply_numa_interleave(void) {
    unsigned long nodemask = 0;
    cpu_set_t affinity;
    if (sched_getaffinity(0, sizeof affinity, &affinity) == 0) {
        for (int node = 0; node < (int)(8 * sizeof nodemask); ++node) {
            char path[128], line[256] = {0};
            snprintf(path, sizeof path,
                     "/sys/devices/system/node/node%d/cpulist", node);
            FILE *f = fopen(path, "r");
            if (!f) continue;
            if (fgets(line, sizeof line, f)) {
                char *p = line;
                while (*p && *p != '\n') {
                    char *end;
                    long first = strtol(p, &end, 10), last = first;
                    if (end == p) break;
                    p = end;
                    if (*p == '-') {
                        last = strtol(p + 1, &end, 10);
                        p = end;
                    }
                    for (long cpu = first; cpu <= last; ++cpu) {
                        if (cpu >= 0 && cpu < CPU_SETSIZE &&
                            CPU_ISSET((int)cpu, &affinity)) {
                            nodemask |= 1UL << node;
                            break;
                        }
                    }
                    if (*p == ',') ++p;
                    else break;
                }
            }
            fclose(f);
        }
    }
    if (!nodemask) nodemask = ~0UL;
    if (syscall(SYS_set_mempolicy, MPOL_INTERLEAVE, &nodemask,
                (unsigned long)(8 * sizeof nodemask)) != 0) {
        fprintf(stderr, "k3: set_mempolicy(mask=0x%lx): %s\n",
                nodemask, strerror(errno));
        return 0;
    }
    fprintf(stderr, "k3: anonymous allocation NUMA mask=0x%lx\n", nodemask);
    return nodemask;
}

/* Read a bounded partial-weight blob into anonymous memory.  Dropping source
 * pages as we progress prevents staged files from competing with HBM weights. */
static inline void *k3_load_blob_anon(const char *path, size_t *size_out) {
    int fd = open(path, O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st) != 0 || st.st_size <= 0) {
        if (fd >= 0) close(fd);
        return NULL;
    }
    size_t size = (size_t)st.st_size;
    void *base = NULL;
    if (posix_memalign(&base, 2UL * 1024 * 1024, size) != 0) {
        close(fd);
        return NULL;
    }
    const size_t chunk = 16UL * 1024 * 1024;
    size_t off = 0;
    while (off < size) {
        size_t want = size - off < chunk ? size - off : chunk;
        size_t done = 0;
        while (done < want) {
            ssize_t n = pread(fd, (char *)base + off + done,
                              want - done, (off_t)(off + done));
            if (n <= 0) {
                free(base);
                close(fd);
                return NULL;
            }
            done += (size_t)n;
        }
#ifdef POSIX_FADV_DONTNEED
        (void)posix_fadvise(fd, (off_t)off, (off_t)want,
                            POSIX_FADV_DONTNEED);
#endif
        off += want;
    }
    close(fd);
    *size_out = size;
    return base;
}

#endif
