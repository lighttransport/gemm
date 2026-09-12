#ifndef DS41F_KV_H
#define DS41F_KV_H

#include <stddef.h>
#include <stdint.h>

#include "ds41f_model.h"

typedef struct {
    uint32_t max_position;
    uint32_t window_tokens;
    uint32_t index_topk;
    size_t compressed_value_bytes;
    size_t compressed_scale_bytes;
    size_t index_value_bytes;
    size_t index_scale_bytes;
    size_t window_value_bytes;
    size_t total_bytes;
} ds41f_kv_plan;

int ds41f_kv_plan_init(ds41f_kv_plan *plan, const ds41f_model_config *cfg,
                       uint32_t max_position, uint32_t window_tokens,
                       uint32_t index_topk);

#endif
