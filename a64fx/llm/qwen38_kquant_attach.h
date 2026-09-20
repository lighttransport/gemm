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
    uint32_t attached_entries;
} q38kc_model_cache;

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

static int q38kc_model_prepare(q38kc_model_cache *cache,
                               transformer_model *model,
                               const char *compact_dir,
                               const char *cache_dir,
                               int rank, int size,
                               char *error, size_t error_bytes) {
    if (!cache || !model || !compact_dir || !*compact_dir ||
        !cache_dir || !*cache_dir || rank < 0 || size <= 1)
        return q38kc_model_error(error, error_bytes, "invalid cache arguments");
    memset(cache, 0, sizeof(*cache));

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
        qtensor *tensor = tf_tp_stage_tensor(model, entry->name);
        tensor->kquant_cache =
            (uint8_t *)cache->loaded.mapping + entry->file_offset;
        tensor->kquant_cache_format = entry->cache_format;
    }
    cache->attached_entries = cache->loaded.header->n_entries;
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
