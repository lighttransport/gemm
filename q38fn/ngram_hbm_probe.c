/* Resident-HBM n-gram lookup probe for one packed Qwen4-Exp split. */
#define _GNU_SOURCE
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "../common/glm53f_safetensors.h"
#include "../common/q38fn_arch.h"

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>
#include <omp.h>
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif
#include <sched.h>

#if defined(__ARM_FEATURE_SVE)
static inline uint64_t sve_stream_sum(const uint64_t *restrict p, size_t n) {
    p = (const uint64_t *)__builtin_assume_aligned(p, 64);
    svbool_t pg = svptrue_b64();
    size_t vl = svcntd(), step = vl * 8, i = 0;
    svuint64_t a0 = svdup_u64(0), a1 = svdup_u64(0);
    svuint64_t a2 = svdup_u64(0), a3 = svdup_u64(0);
    svuint64_t a4 = svdup_u64(0), a5 = svdup_u64(0);
    svuint64_t a6 = svdup_u64(0), a7 = svdup_u64(0);
    for (; i + step <= n; i += step) {
        a0 = svadd_u64_x(pg, a0, svld1_u64(pg, p + i + 0 * vl));
        a1 = svadd_u64_x(pg, a1, svld1_u64(pg, p + i + 1 * vl));
        a2 = svadd_u64_x(pg, a2, svld1_u64(pg, p + i + 2 * vl));
        a3 = svadd_u64_x(pg, a3, svld1_u64(pg, p + i + 3 * vl));
        a4 = svadd_u64_x(pg, a4, svld1_u64(pg, p + i + 4 * vl));
        a5 = svadd_u64_x(pg, a5, svld1_u64(pg, p + i + 5 * vl));
        a6 = svadd_u64_x(pg, a6, svld1_u64(pg, p + i + 6 * vl));
        a7 = svadd_u64_x(pg, a7, svld1_u64(pg, p + i + 7 * vl));
    }
    uint64_t sum = svaddv_u64(pg, svadd_u64_x(pg, svadd_u64_x(pg, a0, a1),
                                               svadd_u64_x(pg, a2, a3)));
    sum += svaddv_u64(pg, svadd_u64_x(pg, svadd_u64_x(pg, a4, a5),
                                      svadd_u64_x(pg, a6, a7)));
    for (; i < n; ++i) sum += p[i];
    return sum;
}
#else
static inline uint64_t sve_stream_sum(const uint64_t *p, size_t n) {
    uint64_t sum = 0;
    for (size_t i = 0; i < n; ++i) sum += p[i];
    return sum;
}
#endif

#if defined(__ARM_FEATURE_SVE)
static inline uint64_t sve_stream_sum_nt(const uint64_t *restrict p, size_t n) {
    p = (const uint64_t *)__builtin_assume_aligned(p, 64);
    svbool_t pg = svptrue_b64();
    size_t vl = svcntd(), step = vl * 8, i = 0;
    svuint64_t a0 = svdup_u64(0), a1 = svdup_u64(0);
    svuint64_t a2 = svdup_u64(0), a3 = svdup_u64(0);
    svuint64_t a4 = svdup_u64(0), a5 = svdup_u64(0);
    svuint64_t a6 = svdup_u64(0), a7 = svdup_u64(0);
    for (; i + step <= n; i += step) {
        a0 = svadd_u64_x(pg, a0, svldnt1_u64(pg, p + i + 0 * vl));
        a1 = svadd_u64_x(pg, a1, svldnt1_u64(pg, p + i + 1 * vl));
        a2 = svadd_u64_x(pg, a2, svldnt1_u64(pg, p + i + 2 * vl));
        a3 = svadd_u64_x(pg, a3, svldnt1_u64(pg, p + i + 3 * vl));
        a4 = svadd_u64_x(pg, a4, svldnt1_u64(pg, p + i + 4 * vl));
        a5 = svadd_u64_x(pg, a5, svldnt1_u64(pg, p + i + 5 * vl));
        a6 = svadd_u64_x(pg, a6, svldnt1_u64(pg, p + i + 6 * vl));
        a7 = svadd_u64_x(pg, a7, svldnt1_u64(pg, p + i + 7 * vl));
    }
    uint64_t sum = svaddv_u64(pg, svadd_u64_x(pg, svadd_u64_x(pg, a0, a1),
                                               svadd_u64_x(pg, a2, a3)));
    sum += svaddv_u64(pg, svadd_u64_x(pg, svadd_u64_x(pg, a4, a5),
                                      svadd_u64_x(pg, a6, a7)));
    for (; i < n; ++i) sum += p[i];
    return sum;
}
#endif

/* Interleave rows in vector-sized chunks.  Row-at-a-time copies expose one
 * HBM miss chain to the core; this keeps ns independent row chains in flight
 * while still writing each output row contiguously. */
__attribute__((noinline)) static void copy_ngram_rows_interleaved(
    uint16_t *restrict dst, const uint16_t *restrict resident,
    const uint64_t *restrict rows, int ns) {
#if defined(__ARM_FEATURE_SVE)
    dst = (uint16_t *)__builtin_assume_aligned(dst, 64);
    resident = (const uint16_t *)__builtin_assume_aligned(resident, 64);
    const size_t vl = svcnth();
    if (vl * 5 == Q38FN_NGRAM_HEAD_DIM) {
        const svbool_t all = svptrue_b16();
        for (size_t v = 0; v < 5; ++v) {
            const size_t offset = v * vl;
            for (int j = 0; j < ns; ++j) {
                const uint16_t *src = resident +
                    rows[j] * Q38FN_NGRAM_HEAD_DIM + offset;
                svst1_u16(all, dst + (size_t)j * Q38FN_NGRAM_HEAD_DIM + offset,
                          svld1_u16(all, src));
            }
        }
        return;
    }
    for (size_t i = 0; i < Q38FN_NGRAM_HEAD_DIM; i += vl) {
        const svbool_t pg = svwhilelt_b16((uint64_t)i,
                                           (uint64_t)Q38FN_NGRAM_HEAD_DIM);
        for (int j = 0; j < ns; ++j) {
            const uint16_t *src = resident + rows[j] * Q38FN_NGRAM_HEAD_DIM + i;
            svst1_u16(pg, dst + (size_t)j * Q38FN_NGRAM_HEAD_DIM + i,
                      svld1_u16(pg, src));
        }
    }
#else
    for (int j = 0; j < ns; ++j)
        memcpy(dst + (size_t)j * Q38FN_NGRAM_HEAD_DIM,
               resident + rows[j] * Q38FN_NGRAM_HEAD_DIM,
               Q38FN_NGRAM_ROW_BYTES);
#endif
}

/* Four token contexts is the best measured A64FX balance between independent
 * HBM misses and per-window setup.  Keep the larger values available for
 * workload-specific tuning via Q38FN_LOOKUP_WINDOW. */
enum { HBM_LOOKUP_WINDOW_DEFAULT = 4, HBM_LOOKUP_WINDOW_MAX = 16 };
enum { HBM_DEDUP_HASH_CAP = 512 };

static inline void prefetch_ngram_row(const uint16_t *row, int lines)
{
    /* A row is five 64-byte cache lines.  The default three hints are
     * deliberately spread across the row: fetching lines 0, 1, 2 leaves
     * the tail behind and consumes the same hint budget without advancing
     * the miss chain.  Keep the diagnostic knob deterministic for 0..5. */
    static const unsigned char line_index[6] = { 0, 2, 4, 1, 3, 0 };
    for (int i = 0; i < lines; ++i)
        __builtin_prefetch((const char *)row + (size_t)line_index[i] * 64,
                           0, 0);
}

static double seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static void *hbm_alloc(size_t bytes) {
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return NULL;
#ifdef MADV_HUGEPAGE
    /* The lookup tensor is resident and streamed repeatedly.  A64FX HBM
     * bandwidth is otherwise needlessly spent on 4-KiB page walks/TLB misses. */
    (void)madvise(p, bytes, MADV_HUGEPAGE);
#endif
    return p;
}

static int discover_cores(int *cores, int capacity) {
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return -1;
    int n = 0;
    for (int cpu = 0; cpu < CPU_SETSIZE && n < capacity; ++cpu)
        if (CPU_ISSET(cpu, &allowed)) cores[n++] = cpu;
    return n;
}

int main(int argc, char **argv) {
    const char *model = argc > 1 ? argv[1] : NULL;
    const char *storage = argc > 2 ? argv[2] : NULL;
    int iters = argc > 3 ? atoi(argv[3]) : 1000000;
    int partition = argc > 4 ? atoi(argv[4]) : 0;
    int duplicate_period = argc > 5 ? atoi(argv[5]) : 0;
    int stream_passes = argc > 6 ? atoi(argv[6]) : 16;
    char name[160], path[4096];
    glm53f_st_context *ctx = NULL;
    const st_context *owner = NULL;
    const st_tensor_info *tensor = NULL;
    uint16_t *resident = NULL;
    uint16_t *cmg[4] = {0};
    int fd = -1, sid = -1, logical = 0, unique = 0;
    int cores[48];
    uint64_t checksum = 0;
    int lookup_window = HBM_LOOKUP_WINDOW_DEFAULT;
    int prefetch_lines = 3;
    int prefetch_batch = 16;
    double min_eff_pct = 0.0;
    int stream_nt = getenv("Q38FN_STREAM_NT") && *getenv("Q38FN_STREAM_NT") != '0';
    const char *window_env = getenv("Q38FN_LOOKUP_WINDOW");
    if (window_env && *window_env) {
        char *end = NULL;
        long value = strtol(window_env, &end, 10);
        if (!end || *end || value < 1 || value > HBM_LOOKUP_WINDOW_MAX) {
            fprintf(stderr, "Q38FN_LOOKUP_WINDOW must be in [1,%d]\n",
                    HBM_LOOKUP_WINDOW_MAX);
            glm53f_st_close(ctx);
            return 2;
        }
        lookup_window = (int)value;
    }
    int interleave_rows = 8;
    const char *interleave_env = getenv("Q38FN_INTERLEAVE_ROWS");
    if (interleave_env && *interleave_env) {
        char *end = NULL;
        long value = strtol(interleave_env, &end, 10);
        if (end != interleave_env && *end == '\0' && value >= 1 && value <= 64)
            interleave_rows = (int)value;
    }
    const char *prefetch_env = getenv("Q38FN_PREFETCH_LINES");
    if (prefetch_env && *prefetch_env) {
        char *end = NULL;
        long value = strtol(prefetch_env, &end, 10);
        if (end != prefetch_env && *end == '\0' && value >= 0 && value <= 5)
            prefetch_lines = (int)value;
    }
    const char *prefetch_batch_env = getenv("Q38FN_PREFETCH_BATCH");
    if (prefetch_batch_env && *prefetch_batch_env) {
        char *end = NULL;
        long value = strtol(prefetch_batch_env, &end, 10);
        if (end != prefetch_batch_env && *end == '\0' && value >= 0 && value <= 256)
            prefetch_batch = (int)value;
    }
    const char *gate_env = getenv("Q38FN_HBM_MIN_EFF_PCT");
    if (gate_env && *gate_env) {
        char *end = NULL;
        double value = strtod(gate_env, &end);
        if (end != gate_env && *end == '\0' && value >= 0.0 && value <= 1000.0)
            min_eff_pct = value;
    }
    double t0, elapsed, load_elapsed;
#if defined(__ARM_FEATURE_SVE)
    uint64_t (*stream_fn)(const uint64_t *restrict, size_t) =
        stream_nt ? sve_stream_sum_nt : sve_stream_sum;
#else
    (void)stream_nt;
    uint64_t (*stream_fn)(const uint64_t *, size_t) = sve_stream_sum;
#endif

    if (!model || !storage || iters < 1 || partition < 0 ||
        partition >= Q38FN_NGRAM_SHARDS || duplicate_period < 0 ||
        duplicate_period > Q38FN_NGRAM_HEADS || stream_passes < 1 || stream_passes > 1024) {
        fprintf(stderr, "usage: %s MODEL_DIR STORAGE_DIR iterations partition duplicate_period stream_passes\n", argv[0]);
        return 2;
    }
    if (discover_cores(cores, 48) < 48) {
        fprintf(stderr, "need 48 allowed A64FX cores\n");
        return 2;
    }
    snprintf(name, sizeof(name),
             "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_%d.weight",
             partition);
    ctx = glm53f_st_open(model);
    tensor = ctx ? glm53f_st_find(ctx, name, &owner) : NULL;
    if (!tensor || !owner || tensor->nbytes != (uint64_t)Q38FN_NGRAM_ROWS_PER_SHARD *
                                      Q38FN_NGRAM_ROW_BYTES) {
        fprintf(stderr, "missing or unexpected tensor %s\n", name);
        goto fail;
    }
    for (int s = 0; s < ctx->n_shards; ++s)
        if (ctx->shards[s].st == owner) { sid = s; break; }
    if (sid < 0) goto fail;
    int raw_staged = 0;
    /* The resumable job stager writes validated payloads as shard-XXX.bin.
     * Prefer that node-local representation so the probe never needs a
     * second large-file copy or a safetensors header in /local.  Retain the
     * original filename as a convenient fallback for direct experiments. */
    if (snprintf(path, sizeof(path), "%s/shard-%03d.bin", storage, partition) >=
            (int)sizeof(path) || access(path, R_OK) != 0) {
        if (snprintf(path, sizeof(path), "%s/%s", storage,
                     ctx->shards[sid].name) >= (int)sizeof(path)) goto fail;
    } else {
        raw_staged = 1;
    }
    fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); goto fail; }
    resident = (uint16_t *)hbm_alloc((size_t)tensor->nbytes);
    if (!resident) {
        fprintf(stderr, "cannot allocate %llu-byte resident tensor\n",
                (unsigned long long)tensor->nbytes);
        goto fail;
    }
    const size_t chunk_bytes = 2 * 1024 * 1024;
    size_t chunks = ((size_t)tensor->nbytes + chunk_bytes - 1) / chunk_bytes;
    /* First-touch the destination before I/O, then read directly into it.
     * This avoids a second tensor-sized staging buffer and the subsequent
     * full memcpy while preserving the explicit per-CMG placement contract. */
#pragma omp parallel num_threads(48)
    {
        int cpu = cores[omp_get_thread_num()];
        cpu_set_t set;
        CPU_ZERO(&set); CPU_SET(cpu, &set);
        (void)sched_setaffinity(0, sizeof(set), &set);
#pragma omp for schedule(static)
        for (size_t c = 0; c < chunks; ++c) {
            size_t off = c * chunk_bytes;
            size_t want = (size_t)tensor->nbytes - off;
            if (want > chunk_bytes) want = chunk_bytes;
            memset((char *)resident + off, 0, want);
        }
    }
    t0 = seconds();
    size_t done = 0;
    while (done < (size_t)tensor->nbytes) {
        size_t want = (size_t)tensor->nbytes - done;
        if (want > (size_t)64 * 1024 * 1024) want = (size_t)64 * 1024 * 1024;
        size_t chunk_done = 0;
        while (chunk_done < want) {
            off_t source_offset = raw_staged ? 0 :
                (off_t)(owner->data_offset + tensor->offset);
            ssize_t got = pread(fd, (char *)resident + done + chunk_done,
                                want - chunk_done,
                                source_offset + (off_t)done + (off_t)chunk_done);
            if (got <= 0) {
                fprintf(stderr, "read resident tensor at %zu/%llu: %s\n",
                        done + chunk_done,
                        (unsigned long long)tensor->nbytes, strerror(errno));
                goto fail;
            }
            chunk_done += (size_t)got;
        }
        done += want;
    }
    load_elapsed = seconds() - t0;
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);

    size_t cmg_bytes = (size_t)tensor->nbytes / 4;
    for (int c = 0; c < 4; ++c) {
        cmg[c] = (uint16_t *)hbm_alloc(cmg_bytes);
        if (!cmg[c]) goto fail;
    }
#pragma omp parallel num_threads(48)
    {
        int t = omp_get_thread_num(), cpu = cores[t], c = t / 12, lane = t % 12;
        cpu_set_t set;
        CPU_ZERO(&set); CPU_SET(cpu, &set);
        (void)sched_setaffinity(0, sizeof(set), &set);
        size_t lane_bytes = (cmg_bytes / 12) & ~(size_t)63;
        size_t off = (size_t)lane * lane_bytes;
        if (lane == 11) lane_bytes = cmg_bytes - off;
        memcpy((char *)cmg[c] + off, (char *)resident + (size_t)c * cmg_bytes + off,
               lane_bytes);
    }

    /* Bind the worker team once, outside the timed region.  Repeating
     * sched_setaffinity() inside every warmup/trial measures scheduler syscall
     * overhead along with HBM bandwidth and makes the peak comparison depend
     * on trial count. */
#pragma omp parallel num_threads(48)
    {
        int cpu = cores[omp_get_thread_num()];
        cpu_set_t set;
        CPU_ZERO(&set); CPU_SET(cpu, &set);
        (void)sched_setaffinity(0, sizeof(set), &set);
    }

    unsigned long long stream_checksum = 0;
    /* Warm the exact stream and time repeated regions, matching the calibrated
     * A64FX ceiling probe. A single short pass is dominated by OpenMP wakeup. */
    for (int warm = 0; warm < 3; ++warm) {
#pragma omp parallel num_threads(48)
        {
            int c = omp_get_thread_num() / 12, lane = omp_get_thread_num() % 12;
            size_t lane_bytes = (cmg_bytes / 12) & ~(size_t)63;
            size_t begin = (size_t)lane * lane_bytes;
            size_t end = lane == 11 ? cmg_bytes : begin + lane_bytes;
            (void)stream_fn((const uint64_t *)cmg[c] + begin / 8,
                            (end - begin) / 8);
        }
    }
    double stream_elapsed = 1e30;
    for (int trial = 0; trial < 6; ++trial) {
        unsigned long long trial_checksum = 0;
        t0 = seconds();
#pragma omp parallel num_threads(48) reduction(+:trial_checksum)
        {
            int c = omp_get_thread_num() / 12, lane = omp_get_thread_num() % 12;
            size_t lane_bytes = (cmg_bytes / 12) & ~(size_t)63;
            size_t begin = (size_t)lane * lane_bytes;
            size_t end = lane == 11 ? cmg_bytes : begin + lane_bytes;
            for (int pass = 0; pass < stream_passes; ++pass)
                trial_checksum += stream_fn((const uint64_t *)cmg[c] + begin / 8,
                                            (end - begin) / 8);
        }
        double dt = seconds() - t0;
        if (dt < stream_elapsed) stream_elapsed = dt;
        stream_checksum += trial_checksum;
    }

    t0 = seconds();
#pragma omp parallel num_threads(48) reduction(+:logical,unique) reduction(^:checksum)
    {
        uint64_t seen[HBM_LOOKUP_WINDOW_MAX * Q38FN_NGRAM_HEADS];
        uint64_t seen_hash[HBM_DEDUP_HASH_CAP];
        uint32_t seen_tag[HBM_DEDUP_HASH_CAP];
        uint16_t thread_rows[HBM_LOOKUP_WINDOW_MAX * Q38FN_NGRAM_HEADS * Q38FN_NGRAM_HEAD_DIM]
            __attribute__((aligned(64)));
        uint32_t generation = 0;
        for (int i = 0; i < HBM_DEDUP_HASH_CAP; ++i) seen_tag[i] = 0;
#pragma omp for schedule(static)
    for (int n = 0; n < iters; n += lookup_window) {
        int window = iters - n;
        if (window > lookup_window) window = lookup_window;
        int ns = 0;
        if (++generation == 0) {
            for (int i = 0; i < HBM_DEDUP_HASH_CAP; ++i) seen_tag[i] = 0;
            generation = 1;
        }
        for (int b = 0; b < window; ++b) {
            for (int h = 0; h < Q38FN_NGRAM_HEADS; ++h) {
                /* Deduplicate the complete in-flight token window, not only
                 * the 16 heads of one token. */
                uint64_t row = duplicate_period ? (uint64_t)(h % duplicate_period) :
                    (uint64_t)(((n + b) * 7919 + h * 104729) % Q38FN_NGRAM_ROWS_PER_SHARD);
                uint32_t pos = (uint32_t)((row * UINT64_C(11400714819323198485)) >> 55) &
                                (HBM_DEDUP_HASH_CAP - 1);
                while (seen_tag[pos] == generation && seen_hash[pos] != row)
                    pos = (pos + 1) & (HBM_DEDUP_HASH_CAP - 1);
                if (seen_tag[pos] != generation) {
                    seen_tag[pos] = generation;
                    seen_hash[pos] = row;
                    seen[ns++] = row;
                }
                logical++;
            }
        }
        /* Expose the complete in-flight window of independent misses before consuming
         * any row. This is the lookup analogue of the stream kernel's eight
         * independent load chains and is effective for random HBM rows. */
        int pf_group = prefetch_batch > 0 ? prefetch_batch : ns;
        if (prefetch_batch > 0 && pf_group < interleave_rows)
            pf_group = interleave_rows;
        if (prefetch_batch > 0 && pf_group % interleave_rows)
            pf_group += interleave_rows - pf_group % interleave_rows;
        for (int base = 0; base < ns; base += interleave_rows) {
            int group = ns - base;
            if (group > interleave_rows) group = interleave_rows;
            /* Bounded mode issues hints just ahead of the consumer.  The
             * default (zero) preserves the all-at-once schedule. */
            if (prefetch_batch == 0 ? base == 0 : base % pf_group == 0) {
                int end = base + pf_group;
                if (end > ns) end = ns;
                for (int j = base; j < end; ++j)
                    prefetch_ngram_row(resident + seen[j] * Q38FN_NGRAM_HEAD_DIM,
                                       prefetch_lines);
            }
            copy_ngram_rows_interleaved(thread_rows +
                                            (size_t)base * Q38FN_NGRAM_HEAD_DIM,
                                        resident, seen + base, group);
        }
        for (int j = 0; j < ns; ++j) {
            checksum ^= (uint64_t)thread_rows[(size_t)j * Q38FN_NGRAM_HEAD_DIM];
            checksum ^= (uint64_t)thread_rows[(size_t)(j + 1) * Q38FN_NGRAM_HEAD_DIM - 1];
            unique++;
        }
    }
    }
    elapsed = seconds() - t0;
    double stream_gbps = (double)tensor->nbytes * stream_passes / stream_elapsed / 1e9;
    double logical_payload_gbps = (double)iters * Q38FN_NGRAM_HEADS * Q38FN_NGRAM_ROW_BYTES /
                                  elapsed / 1e9;
    double unique_payload_gbps = (double)unique * Q38FN_NGRAM_ROW_BYTES / elapsed / 1e9;
    printf("Q38FN_NGRAM_HBM partition=%d tensor_GB=%.3f load_s=%.3f stream_passes=%d "
           "stream_GB_s=%.2f stream_eff_pct=%.2f stream_checksum=%llu iterations=%d logical=%d unique=%d "
           "duplicate_period=%d lookup_window=%d interleave_rows=%d lookup_s=%.2f unique_s=%.2f logical_payload_GB_s=%.3f "
           "unique_payload_GB_s=%.3f prefetch_lines=%d prefetch_batch=%d stream_nt=%d checksum=%llu threads=%d\n",
           partition, (double)tensor->nbytes / 1e9, load_elapsed, stream_passes,
           stream_gbps, stream_gbps / 922.47 * 100.0,
           stream_checksum, iters, logical, unique, duplicate_period, lookup_window, interleave_rows,
           (double)iters / elapsed, (double)unique / elapsed,
           logical_payload_gbps, unique_payload_gbps, prefetch_lines, prefetch_batch, stream_nt,
           (unsigned long long)checksum, omp_get_max_threads());
    int gate_failed = min_eff_pct > 0.0 &&
                      stream_gbps / 922.47 * 100.0 < min_eff_pct;
    {
        const char *prefix = getenv("Q38FN_RESULT_PREFIX");
        if (prefix && *prefix) {
            const char *rank = getenv("OMPI_COMM_WORLD_RANK");
            char host[128], out_path[4096];
            if (!rank) rank = getenv("PMI_RANK");
            if (!rank) rank = "0";
            if (gethostname(host, sizeof(host)) != 0) strcpy(host, "unknown");
            host[sizeof(host) - 1] = '\0';
            if (snprintf(out_path, sizeof(out_path), "%s.%s.%s", prefix, rank, host)
                < (int)sizeof(out_path)) {
                FILE *rf = fopen(out_path, "w");
                if (rf) {
                    fprintf(rf, "Q38FN_NGRAM_HBM rank=%s host=%s partition=%d "
                            "tensor_GB=%.3f load_s=%.3f stream_passes=%d "
                            "stream_GB_s=%.2f stream_eff_pct=%.2f iterations=%d "
                            "logical=%d unique=%d duplicate_period=%d lookup_window=%d interleave_rows=%d lookup_s=%.2f "
                            "unique_s=%.2f logical_payload_GB_s=%.3f unique_payload_GB_s=%.3f prefetch_lines=%d prefetch_batch=%d stream_nt=%d "
                            "checksum=%llu threads=%d\n", rank, host,
                            partition, (double)tensor->nbytes / 1e9, load_elapsed,
                            stream_passes, stream_gbps, stream_gbps / 922.47 * 100.0,
                            iters, logical, unique, duplicate_period, lookup_window, interleave_rows,
                            (double)iters / elapsed, (double)unique / elapsed,
                            logical_payload_gbps, unique_payload_gbps,
                            prefetch_lines, prefetch_batch, stream_nt,
                            (unsigned long long)checksum, omp_get_max_threads());
                    fclose(rf);
                }
            }
        }
    }
    for (int c = 0; c < 4; ++c) munmap(cmg[c], cmg_bytes);
    munmap(resident, (size_t)tensor->nbytes);
    close(fd); glm53f_st_close(ctx);
    if (gate_failed)
        fprintf(stderr, "q38fn_hbm: efficiency gate failed: %.2f%% < %.2f%%\n",
                stream_gbps / 922.47 * 100.0, min_eff_pct);
    return gate_failed ? 3 : 0;
fail:
    for (int c = 0; c < 4; ++c) if (cmg[c]) munmap(cmg[c], (size_t)tensor->nbytes / 4);
    if (resident) munmap(resident, tensor ? (size_t)tensor->nbytes : 0);
    if (fd >= 0) close(fd);
    glm53f_st_close(ctx);
    return 1;
}
