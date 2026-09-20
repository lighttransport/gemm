/* Validate and map a rank-local Q5R/IQ4R decode sidecar. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "../../common/ggml_dequant.h"
#define TF_KQUANT_CACHE_LAYOUT_ONLY
#define TF_KQUANT_CACHE_PACK_ONLY
#include "kquant_decode_cache.h"
#include "qwen38_kquant_load.h"

#define Q38KC_HASH_BUFFER_BYTES (1u * 1024u * 1024u)

static int fail(char *error, size_t error_bytes, const char *format, ...) {
    if (error && error_bytes) {
        va_list args;
        va_start(args, format);
        vsnprintf(error, error_bytes, format, args);
        va_end(args);
    }
    return -1;
}

static int read_all_at(int fd, void *buffer, size_t bytes, uint64_t offset) {
    uint8_t *p = (uint8_t *)buffer;
    while (bytes) {
        ssize_t got = pread(fd, p, bytes, (off_t)offset);
        if (got <= 0) return -1;
        p += got;
        bytes -= (size_t)got;
        offset += (uint64_t)got;
    }
    return 0;
}

static uint64_t hash_continue(uint64_t hash, const void *data, size_t bytes) {
    const uint8_t *p = (const uint8_t *)data;
    for (size_t i = 0; i < bytes; i++) {
        hash ^= p[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static int hash_payload(int fd, const q38kc_entry *entry, uint8_t *buffer,
                        uint64_t *result) {
    uint64_t hash = UINT64_C(1469598103934665603);
    uint64_t offset = entry->file_offset;
    uint64_t remaining = entry->byte_length;
    while (remaining) {
        size_t chunk = remaining > Q38KC_HASH_BUFFER_BYTES ?
            Q38KC_HASH_BUFFER_BYTES : (size_t)remaining;
        if (read_all_at(fd, buffer, chunk, offset)) return -1;
        hash = hash_continue(hash, buffer, chunk);
        offset += chunk;
        remaining -= chunk;
    }
    *result = hash;
    return 0;
}

static int validate_source(const q38tp_header *source,
                           uint64_t source_file_bytes,
                           uint32_t tp_rank, uint32_t tp_size) {
    if (!source || memcmp(source->magic, Q38TP_MAGIC, 8) ||
        source->version != Q38TP_VERSION ||
        source->header_bytes != Q38TP_HEADER_BYTES ||
        source->tp_rank != tp_rank || source->tp_size != tp_size ||
        source->n_entries > Q38TP_MAX_ENTRIES ||
        source_file_bytes < Q38TP_HEADER_BYTES) return -1;
    uint64_t hash = q38tp_hash_update(
        0, source->entries,
        (size_t)source->n_entries * sizeof(source->entries[0]));
    if (hash != source->entries_checksum) return -1;
    for (uint32_t i = 0; i < source->n_entries; i++) {
        const q38tp_entry *entry = &source->entries[i];
        if (!memchr(entry->name, 0, sizeof(entry->name)) ||
            entry->file_offset < Q38TP_HEADER_BYTES ||
            entry->file_offset > source_file_bytes ||
            entry->byte_length > source_file_bytes - entry->file_offset)
            return -1;
        for (uint32_t j = 0; j < i; j++)
            if (!strcmp(source->entries[j].name, entry->name)) return -1;
    }
    return 0;
}

static const q38tp_entry *match_source(const q38tp_header *source,
                                       const q38kc_entry *cache) {
    const q38tp_entry *matched = NULL;
    for (uint32_t i = 0; i < source->n_entries; i++) {
        const q38tp_entry *candidate = &source->entries[i];
        if (strcmp(candidate->name, cache->name)) continue;
        if (matched) return NULL;
        matched = candidate;
    }
    if (!matched || matched->type != cache->source_type ||
        matched->local_rows != cache->local_rows ||
        matched->local_cols != cache->local_cols ||
        matched->checksum != cache->source_checksum) return NULL;
    return matched;
}

void q38kc_unload(q38kc_loaded *loaded) {
    if (!loaded) return;
    if (loaded->mapping && loaded->mapping != MAP_FAILED)
        munmap(loaded->mapping, loaded->mapping_bytes);
    free(loaded->header);
    memset(loaded, 0, sizeof(*loaded));
}

int q38kc_load(q38kc_loaded *loaded, const char *path,
               const q38tp_header *source, uint64_t source_file_bytes,
               uint32_t tp_rank, uint32_t tp_size,
               char *error, size_t error_bytes) {
    if (!loaded || !path) return fail(error, error_bytes, "invalid arguments");
    memset(loaded, 0, sizeof(*loaded));
    if (validate_source(source, source_file_bytes, tp_rank, tp_size))
        return fail(error, error_bytes, "invalid compact source identity");

    int fd = open(path, O_RDONLY);
    if (fd < 0) return fail(error, error_bytes, "open %s: %s", path,
                            strerror(errno));
    struct stat st;
    q38kc_header *header = calloc(1, sizeof(*header));
    uint8_t *buffer = NULL;
    void *mapping = MAP_FAILED;
    int rc = -1;
    if (!header) {
        fail(error, error_bytes, "allocate sidecar header");
        goto done;
    }
    if (fstat(fd, &st) || st.st_size < 0 ||
        read_all_at(fd, header, sizeof(*header), 0)) {
        fail(error, error_bytes, "read sidecar header");
        goto done;
    }
    uint64_t file_bytes = (uint64_t)st.st_size;
    if (memcmp(header->magic, Q38KC_MAGIC, 8) ||
        header->version != Q38KC_VERSION ||
        header->header_bytes != Q38KC_HEADER_BYTES ||
        header->layout_version != TF_KQUANT_CACHE_LAYOUT_VERSION ||
        header->tp_rank != tp_rank || header->tp_size != tp_size ||
        !header->n_entries || header->n_entries > Q38KC_MAX_ENTRIES ||
        header->source_stage_version != source->version ||
        header->source_file_bytes != source_file_bytes ||
        header->source_entries_checksum != source->entries_checksum ||
        file_bytes < Q38KC_HEADER_BYTES ||
        header->data_bytes != file_bytes - Q38KC_HEADER_BYTES) {
        fail(error, error_bytes, "incompatible sidecar header");
        goto done;
    }
    uint64_t entries_hash = q38kc_hash_update(
        0, header->entries,
        (size_t)header->n_entries * sizeof(header->entries[0]));
    if (entries_hash != header->entries_checksum) {
        fail(error, error_bytes, "sidecar entry-table checksum mismatch");
        goto done;
    }
    uint64_t previous_end = Q38KC_HEADER_BYTES;
    for (uint32_t i = 0; i < header->n_entries; i++) {
        const q38kc_entry *entry = &header->entries[i];
        if (!memchr(entry->name, 0, sizeof(entry->name))) {
            fail(error, error_bytes, "unterminated sidecar entry name %u", i);
            goto done;
        }
        for (uint32_t j = 0; j < i; j++)
            if (!strcmp(header->entries[j].name, entry->name)) {
                fail(error, error_bytes, "duplicate sidecar entry %s", entry->name);
                goto done;
            }
        uint32_t wanted_format = entry->source_type == GGML_TYPE_Q5_K ?
            Q38KC_FORMAT_Q5R : entry->source_type == GGML_TYPE_IQ4_XS ?
            Q38KC_FORMAT_IQ4R : 0;
        size_t wanted_bytes = wanted_format == Q38KC_FORMAT_Q5R ?
            packed_q5r_bytes((int)entry->local_rows, (int)entry->local_cols) :
            wanted_format == Q38KC_FORMAT_IQ4R ?
            packed_iq4r_bytes((int)entry->local_rows, (int)entry->local_cols) : 0;
        if (!wanted_format || entry->cache_format != wanted_format || !wanted_bytes ||
            entry->byte_length != wanted_bytes ||
            entry->file_offset % 256 || entry->file_offset < previous_end ||
            entry->file_offset > file_bytes ||
            entry->byte_length > file_bytes - entry->file_offset ||
            !match_source(source, entry)) {
            fail(error, error_bytes, "invalid sidecar entry %u", i);
            goto done;
        }
        previous_end = entry->file_offset + entry->byte_length;
    }
    if (previous_end != file_bytes) {
        fail(error, error_bytes, "sidecar payload extent mismatch");
        goto done;
    }
    if (posix_memalign((void **)&buffer, 256, Q38KC_HASH_BUFFER_BYTES)) {
        fail(error, error_bytes, "allocate sidecar hash buffer");
        goto done;
    }
    for (uint32_t i = 0; i < header->n_entries; i++) {
        uint64_t hash = 0;
        const q38kc_entry *entry = &header->entries[i];
        if (hash_payload(fd, entry, buffer, &hash) || hash != entry->checksum) {
            fail(error, error_bytes, "sidecar payload checksum mismatch: %s",
                 entry->name);
            goto done;
        }
#ifdef POSIX_FADV_DONTNEED
        posix_fadvise(fd, (off_t)entry->file_offset, (off_t)entry->byte_length,
                      POSIX_FADV_DONTNEED);
#endif
    }
    mapping = mmap(NULL, (size_t)file_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapping == MAP_FAILED) {
        fail(error, error_bytes, "map sidecar: %s", strerror(errno));
        goto done;
    }
    loaded->header = header;
    loaded->mapping = mapping;
    loaded->mapping_bytes = (size_t)file_bytes;
    header = NULL;
    mapping = MAP_FAILED;
    rc = 0;
done:
    if (mapping != MAP_FAILED) munmap(mapping, (size_t)st.st_size);
    free(buffer);
    free(header);
    close(fd);
    return rc;
}

const q38kc_entry *q38kc_find(const q38kc_loaded *loaded,
                              const q38tp_entry *source,
                              const void **payload) {
    if (payload) *payload = NULL;
    if (!loaded || !loaded->header || !loaded->mapping || !source ||
        !memchr(source->name, 0, sizeof(source->name))) return NULL;
    for (uint32_t i = 0; i < loaded->header->n_entries; i++) {
        const q38kc_entry *entry = &loaded->header->entries[i];
        if (!strcmp(entry->name, source->name) &&
            entry->source_type == source->type &&
            entry->local_rows == source->local_rows &&
            entry->local_cols == source->local_cols &&
            entry->source_checksum == source->checksum) {
            if (payload)
                *payload = (const uint8_t *)loaded->mapping + entry->file_offset;
            return entry;
        }
    }
    return NULL;
}
