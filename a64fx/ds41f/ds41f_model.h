#ifndef DS41F_MODEL_H
#define DS41F_MODEL_H

#include <stdint.h>

/* V4.1-Flash text backbone contract.  Keep this separate from common/ds4f.h:
 * the older V4 runner has different layer, expert, MLA and cache geometry. */
#define DS41F_LAYERS              40
#define DS41F_HIDDEN              5120
#define DS41F_HEADS               64
#define DS41F_HEAD_DIM            512
#define DS41F_Q_LORA              1280
#define DS41F_KV_LORA             512
#define DS41F_O_LORA              1024
#define DS41F_O_GROUPS            8
#define DS41F_EXPERTS             384
#define DS41F_ACTIVE_EXPERTS      6
#define DS41F_EXPERT_INTER        2304
#define DS41F_EP_RANKS            12
#define DS41F_MAX_POSITION        1048576

typedef struct {
    int layer;
    int compress_ratio;
    int is_kv_source;
    int is_index_source;
    int has_engram;
} ds41f_layer_config;

typedef struct {
    int layers, hidden, heads, head_dim;
    int q_lora, kv_lora, o_lora, o_groups;
    int experts, active_experts, expert_inter;
    int max_position;
    ds41f_layer_config layer[DS41F_LAYERS];
} ds41f_model_config;

void ds41f_model_config_init(ds41f_model_config *cfg);
int ds41f_model_config_valid(const ds41f_model_config *cfg);
int ds41f_expert_owner(int expert, int rank, int ranks);
int ds41f_owned_expert_count(int rank, int ranks);

#endif
