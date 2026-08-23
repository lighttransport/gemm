#ifndef QWEN38_TP_STAGE_H
#define QWEN38_TP_STAGE_H

#include <stdint.h>
#include <stddef.h>

#define Q38TP_MAGIC "Q38TPB1"
#define Q38TP_VERSION 2u
#define Q38TP_MAX_ENTRIES 768u
#define Q38TP_NAME_BYTES 96u
#define Q38TP_HEADER_BYTES (2u * 1024u * 1024u)

enum q38tp_slice_kind {
    Q38TP_SLICE_ROWS = 1,
    Q38TP_SLICE_COLS = 2,
    Q38TP_SLICE_SSM_ROWS = 3,
};

typedef struct {
    char name[Q38TP_NAME_BYTES];
    uint32_t type;
    uint32_t kind;
    uint32_t local_rows;
    uint32_t local_cols;
    uint32_t source_rows;
    uint32_t source_cols;
    uint32_t row0;
    uint32_t row1;
    uint32_t col0;
    uint32_t col1;
    uint32_t qk_rows;
    uint32_t reserved;
    uint64_t source_row_bytes;
    uint64_t local_row_bytes;
    uint64_t file_offset;
    uint64_t byte_length;
    uint64_t checksum;
} q38tp_entry;

typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint32_t tp_rank;
    uint32_t tp_size;
    uint32_t n_entries;
    uint32_t n_layers;
    uint32_t n_embd;
    uint32_t n_vocab;
    uint32_t reserved;
    uint64_t data_bytes;
    uint64_t source_bytes;
    uint64_t entries_checksum;
    q38tp_entry entries[Q38TP_MAX_ENTRIES];
} q38tp_header;

static inline uint64_t q38tp_hash_update(uint64_t h, const void *data, size_t n) {
    const unsigned char *p = (const unsigned char *)data;
    if (!h) h = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

#endif
