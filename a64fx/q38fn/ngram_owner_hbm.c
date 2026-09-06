#define _GNU_SOURCE
#include "ngram_owner_hbm.h"
#include <errno.h>
#include <fcntl.h>
#include <omp.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

/* A uTofu response is normally a short contiguous span.  Pull a few future
 * rows into the CMG cache while the current span is being copied so the
 * service thread does not expose every HBM line miss to the requester.  The
 * low-locality hint is deliberately small: larger distances pollute the cache for the
 * random-lookup case. */
#define Q38FN_HBM_PREFETCH_ROWS 4

static inline void copy_hbm_rows(void *restrict dst, const void *restrict src,
                                 uint32_t rows, uint32_t prefetch_rows)
{
#if defined(__ARM_FEATURE_SVE)
    uint16_t *d = (uint16_t *)dst;
    const uint16_t *s = (const uint16_t *)src;
    const uint64_t vl = svcnth();
    if (vl * 5 == Q38FN_NGRAM_HEAD_DIM) {
        const svbool_t all = svptrue_b16();
        for (uint32_t row = 0; row < rows; ++row) {
            if (prefetch_rows && row + prefetch_rows < rows)
                __builtin_prefetch(s + (size_t)(row + prefetch_rows) *
                                   Q38FN_NGRAM_HEAD_DIM, 0, 0);
            const size_t base = (size_t)row * Q38FN_NGRAM_HEAD_DIM;
            for (size_t v = 0; v < 5; ++v) {
                const size_t offset = base + v * vl;
                svst1_u16(all, d + offset, svld1_u16(all, s + offset));
            }
        }
        return;
    }
    /* Interleave lookahead with copying. Issuing every hint before the copy
     * floods the miss queue for short uTofu spans; one future row per
     * completed row preserves memory-level parallelism. */
    for (uint32_t row = 0; row < rows; ++row) {
        if (prefetch_rows && row + prefetch_rows < rows)
            __builtin_prefetch(s + (size_t)(row + prefetch_rows) *
                               Q38FN_NGRAM_HEAD_DIM, 0, 0);
        uint64_t i = (uint64_t)row * Q38FN_NGRAM_HEAD_DIM;
        const uint64_t end = i + Q38FN_NGRAM_HEAD_DIM;
        for (; i < end; i += vl) {
            svbool_t pg = svwhilelt_b16(i, end);
            svst1_u16(pg, d + i, svld1_u16(pg, s + i));
        }
    }
#else
    (void)prefetch_rows;
    memcpy(dst, src, (size_t)rows * Q38FN_NGRAM_ROW_BYTES);
#endif
}

struct q38fn_ngram_owner_hbm {
    glm53f_st_context *ctx;
    uint32_t rank, nranks;
    uint32_t shard_limit;
    uint32_t prefetch_rows;
    uint32_t load_threads;
    char *stage_dir;
    uint16_t *data[Q38FN_NGRAM_SHARDS];
};

static int load_file_parallel(int fd, void *dst, size_t bytes, size_t chunk,
                              uint32_t threads)
{
    size_t nchunks = (bytes + chunk - 1) / chunk;
    int error = 0;
#pragma omp parallel for num_threads(threads) schedule(static)
    for (long index = 0; index < (long)nchunks; ++index) {
        size_t off = (size_t)index * chunk;
        size_t want = bytes - off < chunk ? bytes - off : chunk;
        size_t done = 0;
        while (done < want) {
            errno = 0;
            ssize_t got = pread(fd, (char *)dst + off + done,
                                want - done, (off_t)(off + done));
            if (got <= 0) {
                int expected = 0;
                __atomic_compare_exchange_n(&error, &expected,
                                            errno ? errno : EIO, 0,
                                            __ATOMIC_RELAXED, __ATOMIC_RELAXED);
                break;
            }
            done += (size_t)got;
        }
        if (done != want) {
            int expected = 0;
            __atomic_compare_exchange_n(&error, &expected,
                                        errno ? errno : EIO, 0,
                                        __ATOMIC_RELAXED, __ATOMIC_RELAXED);
        }
    }
    return error;
}

static int mkdir_p(const char *path)
{
    char buf[4096];
    size_t n = strlen(path);
    if (!n || n >= sizeof buf) return ENAMETOOLONG;
    memcpy(buf, path, n + 1);
    for (char *p = buf + 1; *p; ++p) {
        if (*p != '/') continue;
        *p = '\0';
        if (mkdir(buf, 0700) != 0 && errno != EEXIST) return errno;
        *p = '/';
    }
    if (mkdir(buf, 0700) != 0 && errno != EEXIST) return errno;
    return 0;
}

static void *resident_alloc(size_t bytes)
{
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return NULL;
#ifdef MADV_HUGEPAGE
    /* The resident image is scanned repeatedly by the uTofu service.  Ask
     * the kernel for transparent huge pages to reduce translation pressure;
     * deployments with strict page placement can disable the hint. */
    if (!getenv("Q38FN_HBM_NO_HUGEPAGE"))
        (void)madvise(p, bytes, MADV_HUGEPAGE);
#endif
    cpu_set_t allowed;
    int cores[CPU_SETSIZE], ncores = 0;
    if (sched_getaffinity(0, sizeof allowed, &allowed) == 0)
        for (int cpu = 0; cpu < CPU_SETSIZE && ncores < CPU_SETSIZE; ++cpu)
            if (CPU_ISSET(cpu, &allowed)) cores[ncores++] = cpu;
    int first_touch_threads = ncores >= 48 ? 48 : ncores;
    if (first_touch_threads < 1) first_touch_threads = 1;
    /* Establish NUMA placement before file I/O.  Touch complete 2 MiB
     * regions, with threads explicitly spread across the four 12-core CMGs;
     * a single-byte probe is insufficient when 4 KiB pages are selected. */
#pragma omp parallel num_threads(first_touch_threads)
    {
        int t = omp_get_thread_num();
        if (ncores > 0) {
            cpu_set_t set;
            CPU_ZERO(&set); CPU_SET(cores[t % ncores], &set);
            (void)sched_setaffinity(0, sizeof set, &set);
        }
#pragma omp for schedule(static)
        for (size_t block = 0; block < bytes; block += 2 * 1024 * 1024) {
            size_t n = bytes - block < 2 * 1024 * 1024 ? bytes - block : 2 * 1024 * 1024;
            memset((char *)p + block, 0, n);
        }
    }
    return p;
}

static int load_shard(q38fn_ngram_owner_hbm *h, uint32_t shard)
{
    char name[192], path[4096];
    const st_context *owner = NULL;
    int n = snprintf(name, sizeof name,
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_%u.weight",
        shard);
    if (n < 0 || (size_t)n >= sizeof name) return EINVAL;
    const st_tensor_info *t = glm53f_st_find(h->ctx, name, &owner);
    if (!t || !owner || t->nbytes != (uint64_t)Q38FN_NGRAM_ROWS_PER_SHARD * Q38FN_NGRAM_ROW_BYTES)
        return EINVAL;
    int sid = -1;
    for (int i = 0; i < h->ctx->n_shards; ++i)
        if (h->ctx->shards[i].st == owner) { sid = i; break; }
    if (sid < 0 || snprintf(path, sizeof path, "%s/%s", h->ctx->model_dir,
                            h->ctx->shards[sid].name) >= (int)sizeof path) return EINVAL;
    double t0;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    t0 = (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
    size_t bytes = (size_t)t->nbytes;
    uint16_t *p = resident_alloc(bytes);
    if (!p) return ENOMEM;
    const size_t chunk = 16 * 1024 * 1024;
    char stage_path[4096];
    int stage_fd = -1, staged = 0;
    if (h->stage_dir) {
        if (snprintf(stage_path, sizeof stage_path, "%s/shard-%03u.bin",
                     h->stage_dir, shard) >= (int)sizeof stage_path) {
            munmap(p, bytes); return ENAMETOOLONG;
        }
        struct stat st;
        if (stat(stage_path, &st) == 0 && (uint64_t)st.st_size == (uint64_t)bytes) {
            stage_fd = open(stage_path, O_RDONLY);
            staged = stage_fd >= 0;
        }
    }
    if (stage_fd >= 0) {
        int e = load_file_parallel(stage_fd, p, bytes, chunk, h->load_threads);
        if (e) { close(stage_fd); munmap(p, bytes); return e; }
        (void)posix_fadvise(stage_fd, 0, 0, POSIX_FADV_DONTNEED);
        close(stage_fd);
    } else {
        int fd = open(path, O_RDONLY);
        if (fd < 0) { munmap(p, bytes); return errno; }
        (void)posix_fadvise(fd, (off_t)(owner->data_offset + t->offset),
                            (off_t)bytes, POSIX_FADV_SEQUENTIAL);
        int cache_fd = -1;
        if (h->stage_dir) {
            cache_fd = open(stage_path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
            if (cache_fd < 0) { close(fd); munmap(p, bytes); return errno; }
        }
        for (size_t off = 0; off < bytes; off += chunk) {
            size_t nbytes = bytes - off < chunk ? bytes - off : chunk;
            ssize_t got = pread(fd, (char *)p + off, nbytes,
                                (off_t)(owner->data_offset + t->offset + off));
            if (got != (ssize_t)nbytes) {
                int e = errno ? errno : EIO;
                if (cache_fd >= 0) { close(cache_fd); unlink(stage_path); }
                close(fd); munmap(p, bytes); return e;
            }
            (void)posix_fadvise(fd, (off_t)(owner->data_offset + t->offset + off),
                                (off_t)nbytes, POSIX_FADV_DONTNEED);
            if (cache_fd >= 0) {
                size_t done = 0;
                while (done < nbytes) {
                    ssize_t put = write(cache_fd, (char *)p + off + done, nbytes - done);
                    if (put <= 0) {
                        int e = errno ? errno : EIO;
                        close(cache_fd); unlink(stage_path);
                        close(fd); munmap(p, bytes); return e;
                    }
                    done += (size_t)put;
                }
            }
        }
        if (cache_fd >= 0) { fsync(cache_fd); close(cache_fd); staged = 1; }
        (void)posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        close(fd);
    }
    h->data[shard] = p;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    double dt = (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9 - t0;
    fprintf(stderr, "q38fn_hbm rank=%u loaded shard=%u bytes=%zu seconds=%.3f GB_s=%.3f prefetch_rows=%u source=%s\n",
            h->rank, shard, bytes, dt, dt > 0.0 ? (double)bytes / dt / 1e9 : 0.0,
            h->prefetch_rows, staged ? "local-stage" : "shared-fs");
    return 0;
}

int q38fn_ngram_owner_hbm_open(q38fn_ngram_owner_hbm **out,
                               const char *model_dir, uint32_t rank,
                               uint32_t nranks)
{
    if (!out || !model_dir || !nranks || rank >= nranks) return EINVAL;
    q38fn_ngram_owner_hbm *h = calloc(1, sizeof *h);
    if (!h) return ENOMEM;
    h->rank = rank; h->nranks = nranks; h->ctx = glm53f_st_open(model_dir);
    if (!h->ctx) { free(h); return ENOENT; }
    h->prefetch_rows = Q38FN_HBM_PREFETCH_ROWS;
    h->load_threads = 4;
    const char *load_threads = getenv("Q38FN_HBM_LOAD_THREADS");
    if (load_threads && *load_threads) {
        char *end = NULL;
        unsigned long value = strtoul(load_threads, &end, 10);
        if (end != load_threads && *end == '\0' && value >= 1 && value <= 48)
            h->load_threads = (uint32_t)value;
    }
    h->shard_limit = Q38FN_NGRAM_SHARDS;
    const char *limit_env = getenv("Q38FN_NGRAM_SHARD_LIMIT");
    if (limit_env && *limit_env) {
        char *end = NULL;
        unsigned long value = strtoul(limit_env, &end, 10);
        if (end != limit_env && *end == '\0' && value > 0 &&
            value <= Q38FN_NGRAM_SHARDS)
            h->shard_limit = (uint32_t)value;
    }
    const char *prefetch = getenv("Q38FN_HBM_PREFETCH_ROWS");
    if (prefetch && *prefetch) {
        char *end = NULL;
        unsigned long value = strtoul(prefetch, &end, 10);
        if (end != prefetch && *end == '\0' && value <= 64)
            h->prefetch_rows = (uint32_t)value;
    }
    const char *stage = getenv("Q38FN_NGRAM_STAGE_DIR");
    char stage_base[4096];
    const char *stage_root = getenv("Q38FN_NGRAM_STAGE_BASE");
    if ((!stage || !*stage) && stage_root && *stage_root) {
        if (snprintf(stage_base, sizeof stage_base, "%s-%u", stage_root, rank) >=
            (int)sizeof stage_base) { q38fn_ngram_owner_hbm_close(h); return ENAMETOOLONG; }
        stage = stage_base;
    }
    if (stage && *stage) {
        h->stage_dir = strdup(stage);
        if (!h->stage_dir) { q38fn_ngram_owner_hbm_close(h); return ENOMEM; }
        int dir_rc = mkdir_p(h->stage_dir);
        if (dir_rc) { q38fn_ngram_owner_hbm_close(h); return dir_rc; }
    }
    for (uint32_t shard = rank; shard < h->shard_limit; shard += nranks) {
        int rc = load_shard(h, shard);
        if (rc) { q38fn_ngram_owner_hbm_close(h); return rc; }
    }
    *out = h; return 0;
}

void q38fn_ngram_owner_hbm_close(q38fn_ngram_owner_hbm *h)
{
    if (!h) return;
    for (uint32_t shard = h->rank; shard < h->shard_limit; shard += h->nranks)
        if (h->data[shard]) munmap(h->data[shard],
                                   (size_t)Q38FN_NGRAM_ROWS_PER_SHARD * Q38FN_NGRAM_ROW_BYTES);
    glm53f_st_close(h->ctx); free(h->stage_dir); free(h);
}

int q38fn_ngram_owner_hbm_read(void *opaque, uint32_t shard,
                               uint64_t first_row, uint32_t rows, void *dst)
{
    q38fn_ngram_owner_hbm *h = (q38fn_ngram_owner_hbm *)opaque;
    if (!h || shard >= Q38FN_NGRAM_SHARDS || shard % h->nranks != h->rank ||
        !h->data[shard] || first_row >= Q38FN_NGRAM_ROWS_PER_SHARD ||
        rows > Q38FN_NGRAM_ROWS_PER_SHARD - first_row || !dst) {
        if (getenv("Q38FN_UTOFU_DEBUG"))
            fprintf(stderr, "q38fn_hbm invalid read h=%p rank=%u nranks=%u shard=%u first=%llu rows=%u data=%p\n",
                    (void *)h, h ? h->rank : 0, h ? h->nranks : 0, shard,
                    (unsigned long long)first_row, rows,
                    h && shard < Q38FN_NGRAM_SHARDS ? (void *)h->data[shard] : NULL);
        return EINVAL;
    }
    const size_t row_bytes = Q38FN_NGRAM_ROW_BYTES;
    const char *src = (const char *)h->data[shard] +
                      first_row * (size_t)row_bytes;
    /* Short spans benefit from an explicit lookahead, but a full 32-row
     * uTofu response is already a sequential stream and the extra hints
     * compete with its hardware prefetcher. */
    uint32_t prefetch_rows = rows <= 16 ? h->prefetch_rows : 0;
    copy_hbm_rows(dst, src, rows, prefetch_rows);
    return 0;
}

int q38fn_ngram_owner_hbm_validate(const q38fn_ngram_owner_hbm *h,
                                   uint64_t *checksum, uint32_t *samples)
{
    if (!h || !checksum || !samples) return EINVAL;
    uint16_t row[Q38FN_NGRAM_HEAD_DIM];
    uint64_t sum = UINT64_C(1469598103934665603);
    uint32_t count = 0;
    for (uint32_t shard = h->rank; shard < h->shard_limit;
         shard += h->nranks) {
        if (!h->data[shard]) return EINVAL;
        const uint64_t last = Q38FN_NGRAM_ROWS_PER_SHARD - 1;
        if (q38fn_ngram_owner_hbm_read((void *)h, shard, 0, 1, row) != 0)
            return EIO;
        for (size_t i = 0; i < Q38FN_NGRAM_HEAD_DIM; ++i)
            sum = (sum ^ row[i]) * UINT64_C(1099511628211);
        if (q38fn_ngram_owner_hbm_read((void *)h, shard, last, 1, row) != 0)
            return EIO;
        for (size_t i = 0; i < Q38FN_NGRAM_HEAD_DIM; ++i)
            sum = (sum ^ row[i]) * UINT64_C(1099511628211);
        count += 2;
    }
    *checksum = sum;
    *samples = count;
    return 0;
}
