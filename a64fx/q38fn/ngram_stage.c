/* Resumable shared-FS -> /local staging utility for owned n-gram shards. */
#define _GNU_SOURCE
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "../../common/glm53f_safetensors.h"
#include "../../common/q38fn_arch.h"
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#if defined(Q38FN_STAGE_PARALLEL)
#include <omp.h>
#endif
#if defined(Q38FN_USE_MPI)
#include <mpi.h>
#endif

#if defined(Q38FN_USE_MPI)
static int q38fn_mpi_active;
static void q38fn_mpi_finalize(void)
{
    if (q38fn_mpi_active) MPI_Finalize();
}
#endif

static int write_all(int fd, const void *buf, size_t n)
{
    const char *p = (const char *)buf;
    while (n) {
        ssize_t got = write(fd, p, n);
        if (got <= 0) return errno ? errno : EIO;
        p += got; n -= (size_t)got;
    }
    return 0;
}

static int mkdir_p(const char *path)
{
    char buf[4096]; size_t n = strlen(path);
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

static int stage_one(const glm53f_st_context *ctx, uint32_t shard, const char *dir)
{
    char name[192], src_path[4096], dst_path[4096], tmp_path[4096];
    const st_context *owner = NULL;
    snprintf(name, sizeof name,
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_%u.weight", shard);
    const st_tensor_info *t = glm53f_st_find(ctx, name, &owner);
    if (!t || !owner) return EINVAL;
    int sid = -1;
    for (int i = 0; i < ctx->n_shards; ++i)
        if (ctx->shards[i].st == owner) { sid = i; break; }
    if (sid < 0 || snprintf(src_path, sizeof src_path, "%s/%s", ctx->model_dir,
                            ctx->shards[sid].name) >= (int)sizeof src_path ||
        snprintf(dst_path, sizeof dst_path, "%s/shard-%03u.bin", dir, shard) >=
            (int)sizeof dst_path ||
        snprintf(tmp_path, sizeof tmp_path, "%s.tmp.%ld", dst_path,
                 (long)getpid()) >= (int)sizeof tmp_path) return ENAMETOOLONG;
    struct stat st;
    if (stat(dst_path, &st) == 0 && (uint64_t)st.st_size == t->nbytes) {
        fprintf(stderr, "q38fn_stage shard=%u status=existing bytes=%llu\n",
                shard, (unsigned long long)t->nbytes);
        return 0;
    }
    int in = open(src_path, O_RDONLY);
    if (in < 0) return errno;
    /* Never expose a preallocated partial file as a valid shard.  A killed
     * parallel writer can otherwise leave the final path at the expected
     * size before its contents are complete. */
    int out = open(tmp_path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (out < 0) { int e = errno; close(in); return e; }
    const size_t chunk = 16 * 1024 * 1024;
    unsigned threads = 1;
#if defined(Q38FN_STAGE_PARALLEL)
    const char *threads_env = getenv("Q38FN_STAGE_THREADS");
    if (threads_env && *threads_env) {
        char *end = NULL;
        unsigned long value = strtoul(threads_env, &end, 10);
        if (end != threads_env && *end == '\0' && value >= 1 && value <= 8)
            threads = (unsigned)value;
    }
#endif
    void *buf = malloc(chunk);
    if (!buf) { close(out); unlink(tmp_path); close(in); return ENOMEM; }
    int rc = 0;
    if (threads == 1) {
        for (size_t off = 0; off < t->nbytes; off += chunk) {
            size_t n = t->nbytes - off < chunk ? t->nbytes - off : chunk;
            ssize_t got = pread(in, buf, n, (off_t)(owner->data_offset + t->offset + off));
            if (got != (ssize_t)n || (rc = write_all(out, buf, n)) != 0) {
                rc = rc ? rc : (errno ? errno : EIO); break;
            }
        }
    } else {
#if defined(Q38FN_STAGE_PARALLEL)
        if (ftruncate(out, (off_t)t->nbytes) != 0) rc = errno;
#pragma omp parallel num_threads(threads) shared(rc)
        {
            void *worker_buf = malloc(chunk);
            if (!worker_buf) {
                int expected = 0;
                __atomic_compare_exchange_n(&rc, &expected, ENOMEM, 0,
                                             __ATOMIC_RELAXED, __ATOMIC_RELAXED);
            }
#pragma omp for schedule(static)
            for (long index = 0; index < (long)((t->nbytes + chunk - 1) / chunk); ++index) {
                size_t off = (size_t)index * chunk;
                size_t n = t->nbytes - off < chunk ? t->nbytes - off : chunk;
                if (!worker_buf || __atomic_load_n(&rc, __ATOMIC_RELAXED)) continue;
                ssize_t got = pread(in, worker_buf, n,
                                    (off_t)(owner->data_offset + t->offset + off));
                if (got != (ssize_t)n) {
                    int expected = 0;
                    __atomic_compare_exchange_n(&rc, &expected,
                                                errno ? errno : EIO, 0,
                                                __ATOMIC_RELAXED, __ATOMIC_RELAXED);
                    continue;
                }
                size_t done = 0;
                while (done < n) {
                    ssize_t put = pwrite(out, (char *)worker_buf + done,
                                         n - done, (off_t)(off + done));
                    if (put <= 0) {
                        int expected = 0;
                        __atomic_compare_exchange_n(&rc, &expected,
                                                    errno ? errno : EIO, 0,
                                                    __ATOMIC_RELAXED, __ATOMIC_RELAXED);
                        break;
                    }
                    done += (size_t)put;
                }
            }
            free(worker_buf);
        }
#endif
    }
    if (!rc && fsync(out) != 0) rc = errno;
    if (close(out) != 0 && !rc) rc = errno;
    if (!rc && rename(tmp_path, dst_path) != 0) rc = errno;
    if (rc) unlink(tmp_path);
    free(buf); (void)posix_fadvise(in, 0, 0, POSIX_FADV_DONTNEED); close(in);
    fprintf(stderr, "q38fn_stage shard=%u status=%s bytes=%llu\n", shard,
            rc ? "failed" : "staged", (unsigned long long)t->nbytes);
    return rc;
}

static uint32_t rank_env(void)
{
    const char *v = getenv("OMPI_COMM_WORLD_RANK");
    if (!v) v = getenv("PMI_RANK");
    if (!v) v = getenv("PMIX_RANK");
#if defined(Q38FN_USE_MPI)
    int mpi_rank = 0;
    if (!v && MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank) == MPI_SUCCESS)
        return (uint32_t)mpi_rank;
#endif
    return v ? (uint32_t)strtoul(v, NULL, 10) : 0;
}

int main(int argc, char **argv)
{
#if defined(Q38FN_USE_MPI)
    MPI_Init(&argc, &argv);
    q38fn_mpi_active = 1;
    atexit(q38fn_mpi_finalize);
#endif
    if (argc < 3) {
        fprintf(stderr, "usage: %s MODEL_DIR STAGE_BASE [ranks=4]\n", argv[0]); return 2;
    }
    uint32_t rank = rank_env();
    uint32_t nranks = argc > 3 ? (uint32_t)strtoul(argv[3], NULL, 10) : 4;
    char dir[4096];
    if (snprintf(dir, sizeof dir, "%s-%u", argv[2], rank) >= (int)sizeof dir) return 2;
    int dir_rc = mkdir_p(dir);
    if (dir_rc) { errno = dir_rc; return perror(dir), 1; }
    glm53f_st_context *ctx = glm53f_st_open(argv[1]);
    if (!ctx) return fprintf(stderr, "rank %u: cannot open model metadata\n", rank), 1;
    int rc = 0;
    uint32_t shard_limit = Q38FN_NGRAM_SHARDS;
    const char *limit_env = getenv("Q38FN_NGRAM_STAGE_LIMIT");
    if (limit_env && *limit_env) {
        char *end = NULL;
        unsigned long value = strtoul(limit_env, &end, 10);
        if (end != limit_env && *end == '\0' && value >= 1 &&
            value <= Q38FN_NGRAM_SHARDS)
            shard_limit = (uint32_t)value;
    }
    for (uint32_t shard = rank; shard < shard_limit; shard += nranks)
        if ((rc = stage_one(ctx, shard, dir)) != 0) break;
    glm53f_st_close(ctx); return rc ? 1 : 0;
}
