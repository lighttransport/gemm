#include "ds41f_kv.h"

#include <errno.h>
#include <string.h>

static size_t ceil_div(size_t a, size_t b) { return (a + b - 1) / b; }

int ds41f_kv_plan_init(ds41f_kv_plan *p, const ds41f_model_config *cfg,
                       uint32_t max_position, uint32_t window_tokens,
                       uint32_t index_topk)
{
    if (!p || !cfg || !ds41f_model_config_valid(cfg) || !max_position ||
        !window_tokens || !index_topk || window_tokens > max_position)
        return EINVAL;
    memset(p, 0, sizeof *p);
    p->max_position = max_position;
    p->window_tokens = window_tokens;
    p->index_topk = index_topk;

    /* Source layers store one compressed latent per ratio-sized token group.
     * The runtime representation is FP4 values plus one FP8/E8M0 scale per
     * 16 values (E4M3 scales). Four source layers are defined by the checkpoint. */
    for (int l = 0; l < cfg->layers; ++l) {
        if (!cfg->layer[l].is_kv_source) continue;
        if (cfg->layer[l].compress_ratio <= 0) return EINVAL;
        size_t tokens = ceil_div(max_position, (size_t)cfg->layer[l].compress_ratio);
        size_t values = tokens * (size_t)cfg->kv_lora;
        p->compressed_value_bytes += ceil_div(values, 2);
        p->compressed_scale_bytes += ceil_div(values, 16);
    }
    /* Only KV source layers own index keys. Later index sources recompute
     * query scores against the most recent KV source's shared key cache. */
    for (int l = 0; l < cfg->layers; ++l) {
        if (!cfg->layer[l].is_kv_source) continue;
        if (cfg->layer[l].compress_ratio <= 0) return EINVAL;
        size_t tokens = ceil_div(max_position, (size_t)cfg->layer[l].compress_ratio);
        size_t values = tokens * 128;
        p->index_value_bytes += ceil_div(values, 2);
        p->index_scale_bytes += ceil_div(values, 32);
    }
    /* Every layer keeps its own FP8 sliding window and group-32 scales. */
    p->window_value_bytes = (size_t)cfg->layers * window_tokens *
        (DS41F_HEAD_DIM + DS41F_HEAD_DIM / 32);
    p->total_bytes = p->compressed_value_bytes + p->compressed_scale_bytes +
                     p->index_value_bytes + p->index_scale_bytes +
                     p->window_value_bytes;
    return 0;
}
