#ifndef QWEN38_KQUANT_LOAD_H
#define QWEN38_KQUANT_LOAD_H

#include <stddef.h>
#include <stdint.h>

#include "qwen38_kquant_stage.h"
#include "qwen38_tp_stage.h"

typedef struct {
    q38kc_header *header;
    void *mapping;
    size_t mapping_bytes;
} q38kc_loaded;

int q38kc_load(q38kc_loaded *loaded, const char *path,
               const q38tp_header *source, uint64_t source_file_bytes,
               uint32_t tp_rank, uint32_t tp_size,
               char *error, size_t error_bytes);

void q38kc_unload(q38kc_loaded *loaded);

const q38kc_entry *q38kc_find(const q38kc_loaded *loaded,
                              const q38tp_entry *source,
                              const void **payload);

#endif
