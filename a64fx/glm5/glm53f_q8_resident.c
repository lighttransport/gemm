#define _GNU_SOURCE
#include "glm53f_q8_resident.h"
#include "../../common/glm5_mem.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

enum { LOAD_CHUNK = 64 * 1024 * 1024 };
struct glm53f_q8_resident {
    void *data;
    size_t size;
    uint64_t hash;
    glm53f_q8_resident_entry *entries;
    size_t entry_count;
};

static int available_ok(size_t extra) {
    FILE *f = fopen("/proc/meminfo", "r");
    char line[256]; unsigned long long kb = 0;
    if (!f) return 1;
    while (fgets(line, sizeof(line), f))
        if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1) break;
    fclose(f);
    return !kb || kb * 1024ULL >= (unsigned long long)extra + 2ULL * 1024 * 1024 * 1024;
}

static uint64_t fnv_update(uint64_t h, const unsigned char *p, size_t n) {
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= UINT64_C(1099511628211); }
    return h;
}

static void apply_interleave_policy(void) {
#if defined(__linux__) && defined(SYS_set_mempolicy)
    if (getenv("NUMA_INTERLEAVE") && atoi(getenv("NUMA_INTERLEAVE"))) {
        unsigned long mask = 0xFFUL; /* A64FX memory nodes 0..7 */
        long rc = syscall(SYS_set_mempolicy, 3 /* MPOL_INTERLEAVE */, &mask, 8UL);
        fprintf(stderr, "glm53f_q8_resident: MPOL_INTERLEAVE%s\n",
                rc == 0 ? " enabled" : " unavailable");
    }
#endif
}

glm53f_q8_resident *glm53f_q8_resident_load(const char *dir, int rank) {
    char path[4096], manifest[4096], line[4096];
    struct stat st;
    snprintf(path, sizeof(path), "%s/rank%02d.blob", dir, rank);
    snprintf(manifest, sizeof(manifest), "%s/rank%02d.manifest", dir, rank);
    int fd = open(path, O_RDONLY), mf = open(manifest, O_RDONLY);
    if (fd < 0 || mf < 0 || fstat(fd, &st) || st.st_size <= 0) goto fail;
    size_t bytes = (size_t)st.st_size;
    /* The preceding /local copy intentionally leaves no useful file cache,
     * but make that contract explicit before measuring HBM headroom. */
    (void)posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    if (!available_ok(bytes < LOAD_CHUNK ? bytes : LOAD_CHUNK)) { errno = ENOMEM; goto fail; }
    apply_interleave_policy();
    void *data = glm5_amalloc(bytes);
    if (!data) goto fail;
    unsigned char *dst = data; size_t done = 0; uint64_t hash = UINT64_C(1469598103934665603);
    while (done < bytes) {
        size_t want = bytes - done < LOAD_CHUNK ? bytes - done : LOAD_CHUNK;
        ssize_t n;
        do { n = pread(fd, dst + done, want, (off_t)done); } while (n < 0 && errno == EINTR);
        if (n <= 0) { glm5_afree(data); goto fail; }
        hash = fnv_update(hash, dst + done, (size_t)n);
        done += (size_t)n;
        size_t next = bytes - done;
        if (next > LOAD_CHUNK) next = LOAD_CHUNK;
        if (!available_ok(next)) { glm5_afree(data); errno = ENOMEM; goto fail; }
        (void)posix_fadvise(fd, (off_t)(done - (size_t)n), (size_t)n, POSIX_FADV_DONTNEED);
    }
    FILE *stream = fdopen(dup(mf), "r");
    if (!stream) { glm5_afree(data); goto fail; }
    char complete[128] = {0};
    glm53f_q8_resident_entry *entries = NULL;
    size_t entry_count = 0, entry_cap = 0, expected_offset = 0;
    while (fgets(line, sizeof(line), stream)) {
        if (!strncmp(line, "T ", 2)) {
            glm53f_q8_resident_entry e;
            memset(&e, 0, sizeof(e));
            unsigned long long idx, d0, d1, d2, d3, source, nbytes, offset;
            unsigned type, ndims;
            if (sscanf(line, "T %llu %191s %u %u %llu %llu %llu %llu %llu %llu %llu %d %d",
                       &idx, e.name, &type, &ndims, &d0, &d1, &d2, &d3,
                       &source, &nbytes, &offset, &e.mode, &e.expert) != 13 || ndims > 4) {
                fclose(stream); free(entries); glm5_afree(data); errno = EILSEQ; goto fail;
            }
            e.tensor_index = idx; e.type = type; e.n_dims = ndims;
            e.dims[0] = d0; e.dims[1] = d1; e.dims[2] = d2; e.dims[3] = d3;
            e.source_offset = source; e.bytes = nbytes; e.data_offset = offset;
            if (offset != expected_offset || offset > bytes || nbytes > bytes - offset) {
                fclose(stream); free(entries); glm5_afree(data); errno = EILSEQ; goto fail;
            }
            expected_offset += (size_t)nbytes;
            if (entry_count == entry_cap) {
                size_t next = entry_cap ? entry_cap * 2 : 256;
                glm53f_q8_resident_entry *p = realloc(entries, next * sizeof(*p));
                if (!p) { fclose(stream); free(entries); glm5_afree(data); goto fail; }
                entries = p; entry_cap = next;
            }
            entries[entry_count++] = e;
        } else if (strstr(line, "# COMPLETE")) {
            snprintf(complete, sizeof(complete), "%s", strstr(line, "# COMPLETE"));
        }
    }
    fclose(stream);
    unsigned long long declared = 0, declared_tensors = 0; char declared_hash[32] = {0};
    if (sscanf(complete, "# COMPLETE bytes=%llu tensors=%llu fnv1a=%31s",
               &declared, &declared_tensors, declared_hash) != 3 ||
        declared != bytes || expected_offset != bytes || declared_tensors != entry_count || entry_count == 0 ||
        strtoull(declared_hash, NULL, 16) != hash) {
        free(entries); glm5_afree(data); errno = EILSEQ; goto fail;
    }
    close(fd); close(mf);
    glm53f_q8_resident *r = calloc(1, sizeof(*r));
    if (!r) { glm5_afree(data); return NULL; }
    r->data = data; r->size = bytes; r->hash = hash;
    r->entries = entries; r->entry_count = entry_count;
    return r;
fail:
    if (fd >= 0) close(fd);
    if (mf >= 0) close(mf);
    return NULL;
}

void glm53f_q8_resident_free(glm53f_q8_resident *r) {
    if (!r) return;
    if (r->data) glm5_afree(r->data);
    free(r->entries);
    free(r);
}
const void *glm53f_q8_resident_data(const glm53f_q8_resident *r) { return r ? r->data : NULL; }
size_t glm53f_q8_resident_size(const glm53f_q8_resident *r) { return r ? r->size : 0; }
uint64_t glm53f_q8_resident_hash(const glm53f_q8_resident *r) { return r ? r->hash : 0; }
size_t glm53f_q8_resident_entry_count(const glm53f_q8_resident *r) { return r ? r->entry_count : 0; }
const glm53f_q8_resident_entry *glm53f_q8_resident_entry_at(const glm53f_q8_resident *r, size_t i) {
    return r && i < r->entry_count ? &r->entries[i] : NULL;
}
const glm53f_q8_resident_entry *glm53f_q8_resident_find(const glm53f_q8_resident *r, const char *name) {
    if (!r || !name) return NULL;
    for (size_t i = 0; i < r->entry_count; ++i)
        if (!strcmp(r->entries[i].name, name)) return &r->entries[i];
    return NULL;
}
