/* End-to-end synthetic compact-stage to k-quant-sidecar test. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"
#include "qwen38_tp_stage.h"
#define TF_KQUANT_CACHE_PACK_ONLY
#include "kquant_decode_cache.h"
#include "qwen38_kquant_load.h"
#include "qwen38_kquant_stage.h"

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
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

static uint32_t next_random(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static void fill_compact(block_q5_K q5[8], block_iq4_xs iq4[8]) {
    uint32_t state = 0x38cace01u;
    for (int b = 0; b < 8; b++) {
        q5[b].d = ggml_fp32_to_fp16(0.001f + b * 0.0001f);
        q5[b].dmin = ggml_fp32_to_fp16(0.0004f + b * 0.00003f);
        for (size_t i = 0; i < sizeof(q5[b].scales); i++)
            q5[b].scales[i] = (uint8_t)next_random(&state);
        for (size_t i = 0; i < sizeof(q5[b].qh); i++)
            q5[b].qh[i] = (uint8_t)next_random(&state);
        for (size_t i = 0; i < sizeof(q5[b].qs); i++)
            q5[b].qs[i] = (uint8_t)next_random(&state);
        iq4[b].d = ggml_fp32_to_fp16(0.0008f + b * 0.0001f);
        iq4[b].scales_h = (uint16_t)next_random(&state);
        for (size_t i = 0; i < sizeof(iq4[b].scales_l); i++)
            iq4[b].scales_l[i] = (uint8_t)next_random(&state);
        for (size_t i = 0; i < sizeof(iq4[b].qs); i++)
            iq4[b].qs[i] = (uint8_t)next_random(&state);
    }
}

static int run_builder(const char *builder, const char *compact_dir,
                       const char *cache_dir, int plan) {
    pid_t pid = fork();
    if (pid < 0) return -1;
    if (!pid) {
        setenv("Q38TP_RANK", "0", 1);
        setenv("Q38TP_SIZE", "2", 1);
        if (plan)
            execl(builder, builder, "--plan", compact_dir, cache_dir, (char *)NULL);
        else
            execl(builder, builder, compact_dir, cache_dir, (char *)NULL);
        _exit(127);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) != pid) return -1;
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

static int write_variant(const char *path, const void *data, size_t bytes) {
    int fd = open(path, O_CREAT | O_TRUNC | O_RDWR, 0644);
    if (fd < 0) return -1;
    int rc = write_all_at(fd, data, bytes, 0) || fdatasync(fd);
    close(fd);
    return rc ? -1 : 0;
}

static int expect_load_failure(const char *label, const char *path,
                               const q38tp_header *source,
                               uint64_t source_file_bytes) {
    q38kc_loaded loaded = {0};
    char error[256] = {0};
    if (!q38kc_load(&loaded, path, source, source_file_bytes, 0, 2,
                    error, sizeof(error))) {
        fprintf(stderr, "loader accepted invalid %s sidecar\n", label);
        q38kc_unload(&loaded);
        return -1;
    }
    if (loaded.header || loaded.mapping || loaded.mapping_bytes) {
        fprintf(stderr, "loader retained state after invalid %s sidecar\n", label);
        q38kc_unload(&loaded);
        return -1;
    }
    if (!error[0]) {
        fprintf(stderr, "loader gave no error for invalid %s sidecar\n", label);
        return -1;
    }
    return 0;
}

static void refresh_entries_hash(q38kc_header *header) {
    header->entries_checksum = q38kc_hash_update(
        0, header->entries,
        (size_t)header->n_entries * sizeof(header->entries[0]));
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s KQUANT_STAGE_BUILDER LOCAL_WORK_DIR\n", argv[0]);
        return 2;
    }
    const char *builder = argv[1];
    char compact_dir[PATH_MAX], cache_dir[PATH_MAX];
    char compact_path[PATH_MAX], cache_path[PATH_MAX];
    snprintf(compact_dir, sizeof(compact_dir), "%s/compact", argv[2]);
    snprintf(cache_dir, sizeof(cache_dir), "%s/cache", argv[2]);
    snprintf(compact_path, sizeof(compact_path), "%s/rank00.blob", compact_dir);
    snprintf(cache_path, sizeof(cache_path), "%s/rank00.kquant", cache_dir);
    if ((mkdir(argv[2], 0755) && errno != EEXIST) ||
        (mkdir(compact_dir, 0755) && errno != EEXIST) ||
        (mkdir(cache_dir, 0755) && errno != EEXIST)) {
        perror("mkdir test stage");
        return 1;
    }
    unlink(cache_path);

    block_q5_K q5[8];
    block_iq4_xs iq4[8];
    fill_compact(q5, iq4);
    q38tp_header *source = calloc(1, sizeof(*source));
    if (!source) return 1;
    memcpy(source->magic, Q38TP_MAGIC, 8);
    source->version = Q38TP_VERSION;
    source->header_bytes = Q38TP_HEADER_BYTES;
    source->tp_rank = 0;
    source->tp_size = 2;
    source->n_entries = 2;
    source->n_layers = 64;
    source->n_embd = 5120;
    source->n_vocab = 248320;
    uint64_t offset = Q38TP_HEADER_BYTES;
    q38tp_entry *q5_entry = &source->entries[0];
    snprintf(q5_entry->name, sizeof(q5_entry->name), "blk.0.ffn_up.weight");
    q5_entry->type = GGML_TYPE_Q5_K;
    q5_entry->kind = Q38TP_SLICE_ROWS;
    q5_entry->local_rows = q5_entry->source_rows = 8;
    q5_entry->local_cols = q5_entry->source_cols = 256;
    q5_entry->row1 = 8;
    q5_entry->col1 = 256;
    q5_entry->source_row_bytes = q5_entry->local_row_bytes = sizeof(block_q5_K);
    q5_entry->file_offset = offset;
    q5_entry->byte_length = sizeof(q5);
    q5_entry->checksum = q38tp_hash_update(0, q5, sizeof(q5));
    offset = align_up(offset + sizeof(q5), 256);
    q38tp_entry *iq4_entry = &source->entries[1];
    snprintf(iq4_entry->name, sizeof(iq4_entry->name), "blk.0.ffn_gate.weight");
    iq4_entry->type = GGML_TYPE_IQ4_XS;
    iq4_entry->kind = Q38TP_SLICE_ROWS;
    iq4_entry->local_rows = iq4_entry->source_rows = 8;
    iq4_entry->local_cols = iq4_entry->source_cols = 256;
    iq4_entry->row1 = 8;
    iq4_entry->col1 = 256;
    iq4_entry->source_row_bytes = iq4_entry->local_row_bytes = sizeof(block_iq4_xs);
    iq4_entry->file_offset = offset;
    iq4_entry->byte_length = sizeof(iq4);
    iq4_entry->checksum = q38tp_hash_update(0, iq4, sizeof(iq4));
    offset += sizeof(iq4);
    source->data_bytes = offset - Q38TP_HEADER_BYTES;
    source->entries_checksum = q38tp_hash_update(
        0, source->entries, source->n_entries * sizeof(source->entries[0]));

    int compact_fd = open(compact_path, O_CREAT | O_TRUNC | O_RDWR, 0644);
    if (compact_fd < 0 || write_all_at(compact_fd, source, sizeof(*source), 0) ||
        write_all_at(compact_fd, q5, sizeof(q5), q5_entry->file_offset) ||
        write_all_at(compact_fd, iq4, sizeof(iq4), iq4_entry->file_offset) ||
        fdatasync(compact_fd)) {
        perror("write compact test stage");
        return 1;
    }
    close(compact_fd);
    if (run_builder(builder, compact_dir, cache_dir, 1) ||
        run_builder(builder, compact_dir, cache_dir, 0)) {
        fprintf(stderr, "kquant stage builder failed\n");
        return 1;
    }

    int cache_fd = open(cache_path, O_RDONLY);
    q38kc_header *cache = calloc(1, sizeof(*cache));
    if (cache_fd < 0 || !cache || read_all_at(cache_fd, cache, sizeof(*cache), 0)) {
        perror("read cache stage");
        return 1;
    }
    uint64_t entries_hash = q38kc_hash_update(
        0, cache->entries, cache->n_entries * sizeof(cache->entries[0]));
    if (memcmp(cache->magic, Q38KC_MAGIC, 8) || cache->version != Q38KC_VERSION ||
        cache->layout_version != TF_KQUANT_CACHE_LAYOUT_VERSION ||
        cache->tp_rank != 0 || cache->tp_size != 2 || cache->n_entries != 2 ||
        cache->source_entries_checksum != source->entries_checksum ||
        entries_hash != cache->entries_checksum) {
        fprintf(stderr, "cache header validation failed\n");
        return 1;
    }
    uint8_t expected_q5[8 * sizeof(packed_q5r_header) + 8 * 256];
    uint8_t expected_iq4[8 * sizeof(packed_iq4r_header) + 8 * 256];
    uint8_t actual_q5[sizeof(expected_q5)], actual_iq4[sizeof(expected_iq4)];
    if (pack_q5r(expected_q5, q5, 8, 256) || pack_iq4r(expected_iq4, iq4, 8, 256))
        return 1;
    const q38kc_entry *cq5 = &cache->entries[0], *ciq4 = &cache->entries[1];
    if (cq5->cache_format != Q38KC_FORMAT_Q5R ||
        ciq4->cache_format != Q38KC_FORMAT_IQ4R ||
        cq5->byte_length != sizeof(actual_q5) ||
        ciq4->byte_length != sizeof(actual_iq4) ||
        read_all_at(cache_fd, actual_q5, sizeof(actual_q5), cq5->file_offset) ||
        read_all_at(cache_fd, actual_iq4, sizeof(actual_iq4), ciq4->file_offset) ||
        memcmp(actual_q5, expected_q5, sizeof(actual_q5)) ||
        memcmp(actual_iq4, expected_iq4, sizeof(actual_iq4)) ||
        cq5->checksum != q38kc_hash_update(0, actual_q5, sizeof(actual_q5)) ||
        ciq4->checksum != q38kc_hash_update(0, actual_iq4, sizeof(actual_iq4))) {
        fprintf(stderr, "cache payload validation failed\n");
        return 1;
    }
    close(cache_fd);
    if (run_builder(builder, compact_dir, cache_dir, 0)) {
        fprintf(stderr, "cache reuse failed\n");
        return 1;
    }
    cache_fd = open(cache_path, O_RDWR);
    uint64_t bad_hash = cache->entries_checksum ^ UINT64_C(0x100);
    if (cache_fd < 0 ||
        write_all_at(cache_fd, &bad_hash, sizeof(bad_hash),
                     offsetof(q38kc_header, entries_checksum)) ||
        fdatasync(cache_fd)) {
        perror("corrupt cache header");
        return 1;
    }
    close(cache_fd);
    if (run_builder(builder, compact_dir, cache_dir, 0)) {
        fprintf(stderr, "cache rebuild failed\n");
        return 1;
    }
    cache_fd = open(cache_path, O_RDONLY);
    if (cache_fd < 0 || read_all_at(cache_fd, cache, sizeof(*cache), 0) ||
        cache->entries_checksum == bad_hash ||
        cache->entries_checksum != q38kc_hash_update(
            0, cache->entries,
            (size_t)cache->n_entries * sizeof(cache->entries[0]))) {
        fprintf(stderr, "cache header was not rebuilt\n");
        return 1;
    }
    close(cache_fd);

    q38kc_loaded loaded = {0};
    char load_error[256] = {0};
    const void *payload_q5 = NULL, *payload_iq4 = NULL;
    if (q38kc_load(&loaded, cache_path, source, offset, 0, 2,
                   load_error, sizeof(load_error)) ||
        !q38kc_find(&loaded, q5_entry, &payload_q5) ||
        !q38kc_find(&loaded, iq4_entry, &payload_iq4) ||
        memcmp(payload_q5, expected_q5, sizeof(expected_q5)) ||
        memcmp(payload_iq4, expected_iq4, sizeof(expected_iq4))) {
        fprintf(stderr, "valid cache load failed: %s\n", load_error);
        return 1;
    }
    q38kc_unload(&loaded);

    struct stat cache_stat;
    cache_fd = open(cache_path, O_RDONLY);
    if (cache_fd < 0 || fstat(cache_fd, &cache_stat) || cache_stat.st_size < 0)
        return 1;
    size_t cache_bytes = (size_t)cache_stat.st_size;
    uint8_t *pristine = malloc(cache_bytes);
    uint8_t *variant = malloc(cache_bytes);
    char variant_path[PATH_MAX];
    snprintf(variant_path, sizeof(variant_path), "%s/invalid.kquant", cache_dir);
    if (!pristine || !variant ||
        read_all_at(cache_fd, pristine, cache_bytes, 0)) return 1;
    close(cache_fd);

#define TEST_VARIANT(label, bytes) do { \
        if (write_variant(variant_path, variant, (bytes)) || \
            expect_load_failure((label), variant_path, source, offset)) return 1; \
    } while (0)

    memcpy(variant, pristine, cache_bytes);
    variant[0] ^= 1;
    TEST_VARIANT("magic", cache_bytes);
    memcpy(variant, pristine, cache_bytes);
    ((q38kc_header *)variant)->version++;
    TEST_VARIANT("version", cache_bytes);
    memcpy(variant, pristine, cache_bytes);
    ((q38kc_header *)variant)->layout_version++;
    TEST_VARIANT("layout", cache_bytes);
    memcpy(variant, pristine, cache_bytes);
    TEST_VARIANT("truncated", cache_bytes - 1);
    memcpy(variant, pristine, cache_bytes);
    ((q38kc_header *)variant)->entries[0].file_offset++;
    refresh_entries_hash((q38kc_header *)variant);
    TEST_VARIANT("offset", cache_bytes);
    memcpy(variant, pristine, cache_bytes);
    memcpy(((q38kc_header *)variant)->entries[1].name,
           ((q38kc_header *)variant)->entries[0].name, Q38KC_NAME_BYTES);
    refresh_entries_hash((q38kc_header *)variant);
    TEST_VARIANT("duplicate", cache_bytes);
    memcpy(variant, pristine, cache_bytes);
    ((q38kc_header *)variant)->entries[0].source_checksum++;
    refresh_entries_hash((q38kc_header *)variant);
    TEST_VARIANT("source", cache_bytes);
    memcpy(variant, pristine, cache_bytes);
    variant[((q38kc_header *)variant)->entries[0].file_offset] ^= 1;
    TEST_VARIANT("payload", cache_bytes);
    memcpy(variant, pristine, cache_bytes);
    ((q38kc_header *)variant)->entries[0].source_type = GGML_TYPE_F32;
    refresh_entries_hash((q38kc_header *)variant);
    TEST_VARIANT("unsupported", cache_bytes);
#undef TEST_VARIANT
    if (expect_load_failure("source-size", cache_path, source, offset + 1)) return 1;
    if (expect_load_failure("missing", "/nonexistent/q38kc-sidecar", source,
                            offset)) return 1;
    unlink(variant_path);
    free(variant);
    free(pristine);

    printf("SENTINEL qwen38_kquant_stage=OK entries=%u q5r=%zu iq4r=%zu "
           "reuse=1 corrupt_rebuild=1 loader_rejects=11 empty_on_reject=1\n",
           cache->n_entries, sizeof(actual_q5), sizeof(actual_iq4));
    free(cache);
    free(source);
    return 0;
}
