#include "ds41f_model.h"

#include <stddef.h>

void ds41f_model_config_init(ds41f_model_config *cfg)
{
    static const int ratios[DS41F_LAYERS] = {
        0, 0, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    };
    static const int kv_sources[] = { 2, 8, 14, 20 };
    static const int index_sources[] = { 2, 8, 14, 20, 24, 28, 32, 36 };
    static const int engram_layers[] = { 1, 14 };
    if (!cfg) return;
    *cfg = (ds41f_model_config){
        .layers = DS41F_LAYERS, .hidden = DS41F_HIDDEN,
        .heads = DS41F_HEADS, .head_dim = DS41F_HEAD_DIM,
        .q_lora = DS41F_Q_LORA, .kv_lora = DS41F_KV_LORA,
        .o_lora = DS41F_O_LORA, .o_groups = DS41F_O_GROUPS,
        .experts = DS41F_EXPERTS, .active_experts = DS41F_ACTIVE_EXPERTS,
        .expert_inter = DS41F_EXPERT_INTER,
        .max_position = DS41F_MAX_POSITION
    };
    for (int i = 0; i < DS41F_LAYERS; ++i) {
        cfg->layer[i].layer = i;
        cfg->layer[i].compress_ratio = ratios[i];
    }
    for (size_t i = 0; i < sizeof kv_sources / sizeof kv_sources[0]; ++i)
        cfg->layer[kv_sources[i]].is_kv_source = 1;
    for (size_t i = 0; i < sizeof index_sources / sizeof index_sources[0]; ++i)
        cfg->layer[index_sources[i]].is_index_source = 1;
    for (size_t i = 0; i < sizeof engram_layers / sizeof engram_layers[0]; ++i)
        cfg->layer[engram_layers[i]].has_engram = 1;
}

int ds41f_model_config_valid(const ds41f_model_config *c)
{
    if (!c || c->layers != 40 || c->hidden != 5120 || c->heads != 64 ||
        c->head_dim != 512 || c->q_lora != 1280 || c->kv_lora != 512 ||
        c->o_lora != 1024 || c->o_groups != 8 || c->experts != 384 ||
        c->active_experts != 6 || c->expert_inter != 2304 ||
        c->max_position != 1048576)
        return 0;
    for (int i = 0; i < c->layers; ++i)
        if (c->layer[i].layer != i ||
            (c->layer[i].compress_ratio != 0 && c->layer[i].compress_ratio != 1 &&
             c->layer[i].compress_ratio != 2)) return 0;
    return 1;
}

int ds41f_expert_owner(int expert, int rank, int ranks)
{
    if (expert < 0 || expert >= DS41F_EXPERTS || rank < 0 || ranks <= 0 || rank >= ranks)
        return -1;
    return expert % ranks == rank ? rank : -1;
}

int ds41f_owned_expert_count(int rank, int ranks)
{
    if (rank < 0 || ranks <= 0 || rank >= ranks) return 0;
    int n = 0;
    for (int e = rank; e < DS41F_EXPERTS; e += ranks) ++n;
    return n;
}
