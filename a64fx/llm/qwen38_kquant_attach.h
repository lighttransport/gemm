/* Attach a fully validated rank-local Q38KQC1 mapping to TP model tensors. */
#ifndef QWEN38_KQUANT_ATTACH_H
#define QWEN38_KQUANT_ATTACH_H

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "qwen38_kquant_load.h"

typedef struct {
    q38kc_loaded loaded;
    uint64_t materialized_bytes;
    uint32_t materialized_entries;
    uint32_t attached_entries;
    int enable_q5;
} q38kc_model_cache;

typedef struct {
    int fd;
    uint8_t *destination;
    const q38kc_header *header;
    int tid;
    int threads;
    int enable_q5;
    int failed;
} q38kc_materialize_task;

static int q38kc_model_error(char *error, size_t error_bytes,
                             const char *format, ...) {
    if (error && error_bytes) {
        va_list args;
        va_start(args, format);
        vsnprintf(error, error_bytes, format, args);
        va_end(args);
    }
    return -1;
}

static int q38kc_read_exact(int fd, void *buffer, size_t bytes,
                            uint64_t offset) {
    uint8_t *p = (uint8_t *)buffer;
    while (bytes) {
        ssize_t got = pread(fd, p, bytes, (off_t)offset);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) return -1;
        p += (size_t)got;
        bytes -= (size_t)got;
        offset += (uint64_t)got;
    }
    return 0;
}

static void *q38kc_materialize_worker(void *argument) {
    q38kc_materialize_task *task = (q38kc_materialize_task *)argument;
    const size_t chunk_max = 8u * 1024u * 1024u;
    for (uint32_t i = 0; i < task->header->n_entries; i++) {
        const q38kc_entry *entry = &task->header->entries[i];
        if (entry->source_type == GGML_TYPE_Q5_K && !task->enable_q5)
            continue;
        uint64_t groups = entry->local_rows / TF_KQUANT_CACHE_ROWS;
        uint64_t group_bytes = groups ? entry->byte_length / groups : 0;
        uint64_t group0 = groups * (uint64_t)task->tid /
                          (uint64_t)task->threads;
        uint64_t group1 = groups * (uint64_t)(task->tid + 1) /
                          (uint64_t)task->threads;
        uint64_t offset = entry->file_offset + group0 * group_bytes;
        uint64_t remaining = (group1 - group0) * group_bytes;
        while (remaining) {
            size_t chunk = remaining > chunk_max ? chunk_max : (size_t)remaining;
            if (q38kc_read_exact(task->fd, task->destination + offset,
                                  chunk, offset)) {
                task->failed = 1;
                return NULL;
            }
#ifdef POSIX_FADV_DONTNEED
            (void)posix_fadvise(task->fd, (off_t)offset, (off_t)chunk,
                                POSIX_FADV_DONTNEED);
#endif
            offset += chunk;
            remaining -= chunk;
        }
    }
    return NULL;
}

static int q38kc_model_materialize(q38kc_model_cache *cache,
                                   transformer_model *model,
                                   const char *path,
                                   char *error, size_t error_bytes) {
    size_t bytes = cache->loaded.mapping_bytes;
    uint64_t materialized_bytes = Q38KC_HEADER_BYTES;
    uint32_t materialized_entries = 0;
    for (uint32_t i = 0; i < cache->loaded.header->n_entries; i++) {
        const q38kc_entry *entry = &cache->loaded.header->entries[i];
        if (entry->source_type != GGML_TYPE_Q5_K || cache->enable_q5) {
            materialized_bytes += entry->byte_length;
            materialized_entries++;
        }
    }
    double available = tf_mem_available_gb();
    double required = (double)materialized_bytes / 1e9 + 3.0;
    if (available >= 0.0 && available < required)
        return q38kc_model_error(error, error_bytes,
                                  "insufficient HBM: %.2f GB available, %.2f GB required",
                                  available, required);
    int fd = open(path, O_RDONLY);
    if (fd < 0)
        return q38kc_model_error(error, error_bytes, "open %s: %s",
                                  path, strerror(errno));
    uint8_t *anonymous = (uint8_t *)mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (anonymous == MAP_FAILED) {
        close(fd);
        return q38kc_model_error(error, error_bytes,
                                  "allocate %.3f GB sidecar arena: %s",
                                  (double)bytes / 1e9, strerror(errno));
    }
    int failed = q38kc_read_exact(fd, anonymous, Q38KC_HEADER_BYTES, 0);
    int threads = model->n_threads > 1 && model->pool_alive ?
        model->n_threads : 1;
    q38kc_materialize_task *tasks =
        (q38kc_materialize_task *)alloca((size_t)threads * sizeof(*tasks));
    for (int tid = 0; tid < threads; tid++)
        tasks[tid] = (q38kc_materialize_task){
            fd, anonymous, cache->loaded.header, tid, threads,
            cache->enable_q5, 0
        };
    if (!failed) {
        if (threads > 1)
            tf_pool_dispatch(model, q38kc_materialize_worker, tasks,
                             sizeof(*tasks));
        else
            q38kc_materialize_worker(&tasks[0]);
        for (int tid = 0; tid < threads; tid++)
            if (tasks[tid].failed) failed = 1;
    }
    close(fd);
    if (failed) {
        munmap(anonymous, bytes);
        return q38kc_model_error(error, error_bytes,
                                  "materialize sidecar arena: read failed");
    }
    if (mprotect(anonymous, bytes, PROT_READ)) {
        int saved_errno = errno;
        munmap(anonymous, bytes);
        return q38kc_model_error(error, error_bytes,
                                  "protect sidecar arena: %s",
                                  strerror(saved_errno));
    }
    munmap(cache->loaded.mapping, cache->loaded.mapping_bytes);
    cache->loaded.mapping = anonymous;
    cache->materialized_bytes = materialized_bytes;
    cache->materialized_entries = materialized_entries;
    return 0;
}

static int q38kc_model_prepare(q38kc_model_cache *cache,
                               transformer_model *model,
                               const char *compact_dir,
                               const char *cache_dir,
                               int rank, int size, int enable_q5,
                               char *error, size_t error_bytes) {
    if (!cache || !model || !compact_dir || !*compact_dir ||
        !cache_dir || !*cache_dir || rank < 0 || size <= 1)
        return q38kc_model_error(error, error_bytes, "invalid cache arguments");
    memset(cache, 0, sizeof(*cache));
    cache->enable_q5 = enable_q5 != 0;

    char compact_path[PATH_MAX], cache_path[PATH_MAX];
    int n0 = snprintf(compact_path, sizeof(compact_path),
                      "%s/rank%02d.blob", compact_dir, rank);
    int n1 = snprintf(cache_path, sizeof(cache_path),
                      "%s/rank%02d.kquant", cache_dir, rank);
    if (n0 < 0 || (size_t)n0 >= sizeof(compact_path) ||
        n1 < 0 || (size_t)n1 >= sizeof(cache_path))
        return q38kc_model_error(error, error_bytes, "stage path too long");

    int fd = open(compact_path, O_RDONLY);
    if (fd < 0)
        return q38kc_model_error(error, error_bytes, "open %s: %s",
                                  compact_path, strerror(errno));
    q38tp_header *source = (q38tp_header *)malloc(sizeof(*source));
    struct stat st;
    int rc = -1;
    if (!source) {
        q38kc_model_error(error, error_bytes, "allocate compact header");
        goto done;
    }
    if (fstat(fd, &st) || st.st_size < 0 ||
        q38kc_read_exact(fd, source, sizeof(*source), 0)) {
        q38kc_model_error(error, error_bytes, "read compact header %s",
                          compact_path);
        goto done;
    }
    if (q38kc_load(&cache->loaded, cache_path, source, (uint64_t)st.st_size,
                   (uint32_t)rank, (uint32_t)size, error, error_bytes))
        goto done;

    for (uint32_t i = 0; i < cache->loaded.header->n_entries; i++) {
        const q38kc_entry *entry = &cache->loaded.header->entries[i];
        if (entry->source_type == GGML_TYPE_Q5_K && !cache->enable_q5)
            continue;
        qtensor *tensor = tf_tp_stage_tensor(model, entry->name);
        if (!tensor || !tensor->data || tensor->type != entry->source_type ||
            tensor->n_rows != (int)entry->local_rows ||
            tensor->n_cols != (int)entry->local_cols ||
            tensor->kquant_cache || tensor->kquant_cache_format) {
            q38kc_model_error(error, error_bytes,
                              "model tensor mismatch: %s", entry->name);
            q38kc_unload(&cache->loaded);
            goto done;
        }
    }
    if (q38kc_model_materialize(cache, model, cache_path,
                                error, error_bytes)) {
        q38kc_unload(&cache->loaded);
        goto done;
    }
    rc = 0;
done:
    free(source);
    close(fd);
    return rc;
}

static int q38kc_model_attach(q38kc_model_cache *cache,
                              transformer_model *model,
                              char *error, size_t error_bytes) {
    if (!cache || !cache->loaded.header || !cache->loaded.mapping || !model)
        return q38kc_model_error(error, error_bytes, "cache is not prepared");
    if (cache->attached_entries)
        return q38kc_model_error(error, error_bytes, "cache already attached");

    /* Validate the complete mapping first so attachment is all-or-nothing. */
    for (uint32_t i = 0; i < cache->loaded.header->n_entries; i++) {
        const q38kc_entry *entry = &cache->loaded.header->entries[i];
        if (entry->source_type == GGML_TYPE_Q5_K && !cache->enable_q5)
            continue;
        qtensor *tensor = tf_tp_stage_tensor(model, entry->name);
        if (!tensor || tensor->type != entry->source_type ||
            tensor->n_rows != (int)entry->local_rows ||
            tensor->n_cols != (int)entry->local_cols ||
            tensor->kquant_cache || tensor->kquant_cache_format)
            return q38kc_model_error(error, error_bytes,
                                      "attach tensor mismatch: %s", entry->name);
    }
    for (uint32_t i = 0; i < cache->loaded.header->n_entries; i++) {
        const q38kc_entry *entry = &cache->loaded.header->entries[i];
        if (entry->source_type == GGML_TYPE_Q5_K && !cache->enable_q5)
            continue;
        qtensor *tensor = tf_tp_stage_tensor(model, entry->name);
        tensor->kquant_cache =
            (uint8_t *)cache->loaded.mapping + entry->file_offset;
        tensor->kquant_cache_format = entry->cache_format;
    }
    cache->attached_entries = 0;
    for (uint32_t i = 0; i < cache->loaded.header->n_entries; i++)
        if (cache->enable_q5 ||
            cache->loaded.header->entries[i].source_type != GGML_TYPE_Q5_K)
            cache->attached_entries++;
    return 0;
}

static void q38kc_model_detach(q38kc_model_cache *cache,
                               transformer_model *model) {
    if (!cache || !cache->loaded.header || !cache->loaded.mapping || !model)
        return;
    for (uint32_t i = 0; i < cache->loaded.header->n_entries; i++) {
        const q38kc_entry *entry = &cache->loaded.header->entries[i];
        qtensor *tensor = tf_tp_stage_tensor(model, entry->name);
        void *payload = (uint8_t *)cache->loaded.mapping + entry->file_offset;
        if (tensor && tensor->kquant_cache == payload &&
            tensor->kquant_cache_format == entry->cache_format) {
            tensor->kquant_cache = NULL;
            tensor->kquant_cache_format = 0;
        }
    }
    cache->attached_entries = 0;
}

static void q38kc_model_close(q38kc_model_cache *cache,
                              transformer_model *model) {
    if (!cache) return;
    q38kc_model_detach(cache, model);
    q38kc_unload(&cache->loaded);
}

#endif
