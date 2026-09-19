/* Build rank-local Q5R/IQ4R decode sidecars from compact Q38TP stages. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"
#include "qwen38_tp_stage.h"
#define TF_KQUANT_CACHE_PACK_ONLY
#include "kquant_decode_cache.h"
#include "qwen38_kquant_stage.h"

_Static_assert(sizeof(q38kc_header) <= Q38KC_HEADER_BYTES,
               "kquant stage header exceeds reserved bytes");

static void die(const char *what) {
    perror(what);
    exit(1);
}

static long env_value(const char *explicit_name, const char *const *names,
                      size_t count) {
    const char *value = getenv(explicit_name);
    if (value && *value) return strtol(value, NULL, 10);
    for (size_t i = 0; i < count; i++) {
        value = getenv(names[i]);
        if (value && *value) return strtol(value, NULL, 10);
    }
    return -1;
}

static long env_rank(void) {
    static const char *const names[] = {
        "PMIX_RANK", "OMPI_COMM_WORLD_RANK", "PMI_RANK", "MV2_COMM_WORLD_RANK"
    };
    return env_value("Q38TP_RANK", names, sizeof(names) / sizeof(names[0]));
}

static long env_size(void) {
    static const char *const names[] = {
        "PMIX_SIZE", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE",
        "MV2_COMM_WORLD_SIZE", "PJM_MPI_PROC"
    };
    return env_value("Q38TP_SIZE", names, sizeof(names) / sizeof(names[0]));
}

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

static int mkdir_p(const char *path) {
    char copy[PATH_MAX];
    size_t length = strlen(path);
    if (!length || length >= sizeof(copy)) return -1;
    memcpy(copy, path, length + 1);
    for (char *p = copy + 1; *p; p++) {
        if (*p != '/') continue;
        *p = 0;
        if (mkdir(copy, 0755) && errno != EEXIST) return -1;
        *p = '/';
    }
    return mkdir(copy, 0755) && errno != EEXIST ? -1 : 0;
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

static int write_all_at(int fd, const void *buffer, size_t bytes,
                        uint64_t offset) {
    const uint8_t *p = (const uint8_t *)buffer;
    while (bytes) {
        ssize_t put = pwrite(fd, p, bytes, (off_t)offset);
        if (put <= 0) return -1;
        p += put;
        bytes -= (size_t)put;
        offset += (uint64_t)put;
    }
    return 0;
}

static size_t compact_row_bytes(uint32_t type, uint32_t cols) {
    if (!cols || cols % 256) return 0;
    if (type == GGML_TYPE_Q5_K) return (size_t)(cols / 256) * sizeof(block_q5_K);
    if (type == GGML_TYPE_IQ4_XS)
        return (size_t)(cols / 256) * sizeof(block_iq4_xs);
    return 0;
}

static int make_cache_entry(const q38tp_entry *source, q38kc_entry *cache) {
    uint32_t format = source->type == GGML_TYPE_Q5_K ? Q38KC_FORMAT_Q5R :
                      source->type == GGML_TYPE_IQ4_XS ? Q38KC_FORMAT_IQ4R : 0;
    if (!format || !source->local_rows || !source->local_cols ||
        source->local_rows % 8 || source->local_cols % 256) return 0;
    size_t row_bytes = compact_row_bytes(source->type, source->local_cols);
    if (!row_bytes || source->local_row_bytes != row_bytes ||
        source->byte_length != (uint64_t)source->local_rows * row_bytes) return -1;
    uint64_t row_groups = source->local_rows / 8;
    uint64_t col_blocks = source->local_cols / 256;
    uint64_t block_bytes = format == Q38KC_FORMAT_Q5R ?
        packed_q5r_block_bytes() : packed_iq4r_block_bytes();
    if (row_groups > UINT64_MAX / col_blocks ||
        row_groups * col_blocks > UINT64_MAX / block_bytes ||
        row_groups * col_blocks * block_bytes > SIZE_MAX) return -1;
    size_t bytes = (size_t)(row_groups * col_blocks * block_bytes);
    if (!bytes) return -1;
    memset(cache, 0, sizeof(*cache));
    memcpy(cache->name, source->name, sizeof(cache->name));
    cache->name[sizeof(cache->name) - 1] = 0;
    cache->source_type = source->type;
    cache->cache_format = format;
    cache->local_rows = source->local_rows;
    cache->local_cols = source->local_cols;
    cache->source_checksum = source->checksum;
    cache->byte_length = bytes;
    return 1;
}

static int compatible_source(const q38tp_header *source, uint64_t file_bytes,
                             long rank, long size) {
    if (memcmp(source->magic, Q38TP_MAGIC, 8) ||
        source->version != Q38TP_VERSION ||
        source->header_bytes != Q38TP_HEADER_BYTES ||
        source->tp_rank != (uint32_t)rank ||
        source->tp_size != (uint32_t)size ||
        source->n_entries > Q38TP_MAX_ENTRIES ||
        file_bytes < Q38TP_HEADER_BYTES) return 0;
    uint64_t entries_hash = q38tp_hash_update(
        0, source->entries, (size_t)source->n_entries * sizeof(source->entries[0]));
    if (entries_hash != source->entries_checksum) return 0;
    for (uint32_t i = 0; i < source->n_entries; i++) {
        const q38tp_entry *entry = &source->entries[i];
        if (!memchr(entry->name, 0, sizeof(entry->name)) ||
            entry->file_offset < Q38TP_HEADER_BYTES ||
            entry->file_offset > file_bytes ||
            entry->byte_length > file_bytes - entry->file_offset)
            return 0;
    }
    return 1;
}

static int compatible_cache(const q38kc_header *old, uint64_t file_bytes,
                            const q38kc_header *planned) {
    if (memcmp(old->magic, Q38KC_MAGIC, 8) ||
        old->version != Q38KC_VERSION ||
        old->header_bytes != Q38KC_HEADER_BYTES ||
        old->layout_version != TF_KQUANT_CACHE_LAYOUT_VERSION ||
        old->tp_rank != planned->tp_rank || old->tp_size != planned->tp_size ||
        old->source_stage_version != planned->source_stage_version ||
        old->source_file_bytes != planned->source_file_bytes ||
        old->source_entries_checksum != planned->source_entries_checksum ||
        old->n_entries != planned->n_entries || old->n_entries > Q38KC_MAX_ENTRIES ||
        old->data_bytes != planned->data_bytes ||
        file_bytes != Q38KC_HEADER_BYTES + old->data_bytes) return 0;
    uint64_t entries_hash = q38kc_hash_update(
        0, old->entries, (size_t)old->n_entries * sizeof(old->entries[0]));
    if (entries_hash != old->entries_checksum) return 0;
    for (uint32_t i = 0; i < old->n_entries; i++) {
        const q38kc_entry *have = &old->entries[i];
        const q38kc_entry *want = &planned->entries[i];
        if (!memchr(have->name, 0, sizeof(have->name)) ||
            strcmp(have->name, want->name) ||
            have->source_type != want->source_type ||
            have->cache_format != want->cache_format ||
            have->local_rows != want->local_rows ||
            have->local_cols != want->local_cols ||
            have->source_checksum != want->source_checksum ||
            have->file_offset != want->file_offset ||
            have->byte_length != want->byte_length ||
            have->file_offset < Q38KC_HEADER_BYTES ||
            have->file_offset > file_bytes ||
            have->byte_length > file_bytes - have->file_offset)
            return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    int plan = 0;
    int arg = 1;
    if (arg < argc && !strcmp(argv[arg], "--plan")) {
        plan = 1;
        arg++;
    }
    if (argc - arg != 2) {
        fprintf(stderr, "usage: %s [--plan] COMPACT_STAGE_DIR CACHE_STAGE_DIR\n",
                argv[0]);
        return 2;
    }
    const char *source_dir = argv[arg];
    const char *cache_dir = argv[arg + 1];
    long rank = env_rank(), size = env_size();
    if (rank < 0 || size < 2 || rank >= size) {
        fprintf(stderr, "qwen38_kquant_stage: invalid rank/size %ld/%ld\n", rank, size);
        return 2;
    }

    char source_path[PATH_MAX];
    snprintf(source_path, sizeof(source_path), "%s/rank%02ld.blob", source_dir, rank);
    int source_fd = open(source_path, O_RDONLY);
    if (source_fd < 0) die("open compact stage");
    struct stat source_stat;
    if (fstat(source_fd, &source_stat)) die("stat compact stage");
    q38tp_header *source = calloc(1, sizeof(*source));
    q38kc_header *cache = calloc(1, sizeof(*cache));
    if (!source || !cache) die("allocate headers");
    if (source_stat.st_size < 0 ||
        read_all_at(source_fd, source, sizeof(*source), 0) ||
        !compatible_source(source, (uint64_t)source_stat.st_size, rank, size)) {
        fprintf(stderr, "qwen38_kquant_stage: incompatible compact stage %s\n",
                source_path);
        return 3;
    }

    memcpy(cache->magic, Q38KC_MAGIC, 8);
    cache->version = Q38KC_VERSION;
    cache->header_bytes = Q38KC_HEADER_BYTES;
    cache->layout_version = TF_KQUANT_CACHE_LAYOUT_VERSION;
    cache->tp_rank = (uint32_t)rank;
    cache->tp_size = (uint32_t)size;
    cache->source_stage_version = source->version;
    cache->source_file_bytes = (uint64_t)source_stat.st_size;
    cache->source_entries_checksum = source->entries_checksum;
    uint64_t planned = Q38KC_HEADER_BYTES;
    for (uint32_t i = 0; i < source->n_entries; i++) {
        q38kc_entry entry;
        int made = make_cache_entry(&source->entries[i], &entry);
        if (made < 0) {
            fprintf(stderr, "qwen38_kquant_stage: invalid compact tensor %s\n",
                    source->entries[i].name);
            return 3;
        }
        if (!made) continue;
        if (cache->n_entries >= Q38KC_MAX_ENTRIES) {
            fprintf(stderr, "qwen38_kquant_stage: too many cache tensors\n");
            return 3;
        }
        for (uint32_t j = 0; j < cache->n_entries; j++)
            if (!strcmp(cache->entries[j].name, entry.name)) {
                fprintf(stderr, "qwen38_kquant_stage: duplicate tensor %s\n",
                        entry.name);
                return 3;
            }
        if (planned > UINT64_MAX - 255) {
            fprintf(stderr, "qwen38_kquant_stage: cache size overflow\n");
            return 3;
        }
        planned = align_up(planned, 256);
        if (planned > (uint64_t)INT64_MAX ||
            entry.byte_length > UINT64_MAX - planned ||
            entry.byte_length > (uint64_t)INT64_MAX - planned) {
            fprintf(stderr, "qwen38_kquant_stage: cache size overflow\n");
            return 3;
        }
        entry.file_offset = planned;
        planned += entry.byte_length;
        cache->entries[cache->n_entries++] = entry;
    }
    cache->data_bytes = planned - Q38KC_HEADER_BYTES;
    if (plan) {
        printf("qwen38_kquant_stage plan rank=%ld/%ld entries=%u compact=%.3fGB "
               "cache=%.3fGB combined=%.3fGB layout=%u\n",
               rank, size, cache->n_entries, (double)source->data_bytes / 1e9,
               (double)cache->data_bytes / 1e9,
               (double)(source->data_bytes + cache->data_bytes) / 1e9,
               cache->layout_version);
        free(cache);
        free(source);
        close(source_fd);
        return 0;
    }
    if (!cache->n_entries) {
        fprintf(stderr, "qwen38_kquant_stage: compact stage has no eligible tensors\n");
        return 3;
    }
    if (mkdir_p(cache_dir)) die("create cache stage directory");
    char final_path[PATH_MAX], partial_path[PATH_MAX];
    snprintf(final_path, sizeof(final_path), "%s/rank%02ld.kquant", cache_dir, rank);
    snprintf(partial_path, sizeof(partial_path), "%s.partial.%ld",
             final_path, (long)getpid());

    int old_fd = open(final_path, O_RDONLY);
    if (old_fd >= 0) {
        q38kc_header old;
        struct stat old_stat;
        if (!fstat(old_fd, &old_stat) && old_stat.st_size >= 0 &&
            read_all_at(old_fd, &old, sizeof(old), 0) == 0 &&
            compatible_cache(&old, (uint64_t)old_stat.st_size, cache)) {
            close(old_fd);
            printf("qwen38_kquant_stage reuse rank=%ld entries=%u bytes=%llu path=%s\n",
                   rank, old.n_entries, (unsigned long long)old.data_bytes, final_path);
            free(cache);
            free(source);
            close(source_fd);
            return 0;
        }
        close(old_fd);
    }

    int out = open(partial_path, O_CREAT | O_TRUNC | O_RDWR, 0644);
    if (out < 0) die("open cache partial");
    for (uint32_t i = 0; i < cache->n_entries; i++) {
        q38kc_entry *entry = &cache->entries[i];
        const q38tp_entry *source_entry = NULL;
        for (uint32_t j = 0; j < source->n_entries; j++)
            if (!strcmp(source->entries[j].name, entry->name)) {
                source_entry = &source->entries[j];
                break;
            }
        if (!source_entry) {
            fprintf(stderr, "qwen38_kquant_stage: lost source tensor %s\n", entry->name);
            return 4;
        }
        size_t compact_bytes = (size_t)source_entry->byte_length;
        size_t compact_alloc = (compact_bytes + 255) & ~(size_t)255;
        size_t cache_alloc = ((size_t)entry->byte_length + 255) & ~(size_t)255;
        void *compact = aligned_alloc(256, compact_alloc);
        uint8_t *packed = aligned_alloc(256, cache_alloc);
        if (!compact || !packed) die("allocate tensor buffers");
        if (read_all_at(source_fd, compact, compact_bytes, source_entry->file_offset))
            die("read compact tensor");
        uint64_t source_hash = q38tp_hash_update(0, compact, compact_bytes);
        if (source_hash != entry->source_checksum) {
            fprintf(stderr, "qwen38_kquant_stage: source checksum mismatch %s\n",
                    entry->name);
            return 4;
        }
        int rc = entry->cache_format == Q38KC_FORMAT_Q5R ?
            pack_q5r(packed, (const block_q5_K *)compact,
                     (int)entry->local_rows, (int)entry->local_cols) :
            pack_iq4r(packed, (const block_iq4_xs *)compact,
                      (int)entry->local_rows, (int)entry->local_cols);
        if (rc || write_all_at(out, packed, (size_t)entry->byte_length,
                               entry->file_offset)) {
            fprintf(stderr, "qwen38_kquant_stage: pack/write failed %s\n", entry->name);
            return 4;
        }
        entry->checksum = q38kc_hash_update(0, packed, (size_t)entry->byte_length);
#ifdef POSIX_FADV_DONTNEED
        posix_fadvise(source_fd, (off_t)source_entry->file_offset,
                      (off_t)source_entry->byte_length, POSIX_FADV_DONTNEED);
#endif
        free(packed);
        free(compact);
        if ((i & 15u) == 15u) {
            if (fdatasync(out)) die("sync cache stage");
#ifdef POSIX_FADV_DONTNEED
            posix_fadvise(out, 0, (off_t)(entry->file_offset + entry->byte_length),
                          POSIX_FADV_DONTNEED);
#endif
        }
        if (rank == 0 && ((i + 1) % 64 == 0 || i + 1 == cache->n_entries))
            fprintf(stderr, "qwen38_kquant_stage: %u/%u %.3fGB\n", i + 1,
                    cache->n_entries,
                    (double)(entry->file_offset + entry->byte_length -
                             Q38KC_HEADER_BYTES) / 1e9);
    }
    cache->entries_checksum = q38kc_hash_update(
        0, cache->entries, (size_t)cache->n_entries * sizeof(cache->entries[0]));
    if (write_all_at(out, cache, sizeof(*cache), 0) || fdatasync(out))
        die("write cache header");
#ifdef POSIX_FADV_DONTNEED
    posix_fadvise(out, 0, 0, POSIX_FADV_DONTNEED);
#endif
    close(out);
    if (rename(partial_path, final_path)) die("rename cache stage");
    char manifest_path[PATH_MAX];
    snprintf(manifest_path, sizeof(manifest_path), "%s/rank%02ld.kquant.manifest",
             cache_dir, rank);
    FILE *manifest = fopen(manifest_path, "w");
    if (manifest) {
        fprintf(manifest,
                "format=q38kc-v%u\nlayout=%u\nrank=%ld\nsize=%ld\nentries=%u\n"
                "bytes=%llu\nsource=%s\nsource_entries_checksum=%016llx\n",
                Q38KC_VERSION, cache->layout_version, rank, size, cache->n_entries,
                (unsigned long long)cache->data_bytes, source_path,
                (unsigned long long)cache->source_entries_checksum);
        fclose(manifest);
    }
    printf("qwen38_kquant_stage rank=%ld entries=%u bytes=%llu path=%s\n",
           rank, cache->n_entries, (unsigned long long)cache->data_bytes, final_path);
    free(cache);
    free(source);
    close(source_fd);
    return 0;
}
