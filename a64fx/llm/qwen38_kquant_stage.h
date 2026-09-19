#ifndef QWEN38_KQUANT_STAGE_H
#define QWEN38_KQUANT_STAGE_H

#include <stddef.h>
#include <stdint.h>

#define Q38KC_MAGIC "Q38KQC1"
#define Q38KC_VERSION 1u
#define Q38KC_HEADER_BYTES (1u * 1024u * 1024u)
#define Q38KC_MAX_ENTRIES 512u
#define Q38KC_NAME_BYTES 96u

enum q38kc_format {
    Q38KC_FORMAT_Q5R = 1,
    Q38KC_FORMAT_IQ4R = 2,
};

typedef struct {
    char name[Q38KC_NAME_BYTES];
    uint32_t source_type;
    uint32_t cache_format;
    uint32_t local_rows;
    uint32_t local_cols;
    uint32_t reserved[4];
    uint64_t source_checksum;
    uint64_t file_offset;
    uint64_t byte_length;
    uint64_t checksum;
} q38kc_entry;

typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint32_t layout_version;
    uint32_t tp_rank;
    uint32_t tp_size;
    uint32_t n_entries;
    uint32_t source_stage_version;
    uint32_t reserved;
    uint64_t data_bytes;
    uint64_t source_file_bytes;
    uint64_t source_entries_checksum;
    uint64_t entries_checksum;
    q38kc_entry entries[Q38KC_MAX_ENTRIES];
} q38kc_header;

static inline uint64_t q38kc_hash_update(uint64_t h, const void *data, size_t n) {
    const uint8_t *p = (const uint8_t *)data;
    if (!h) h = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

#endif
