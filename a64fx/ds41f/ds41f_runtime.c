#include "ds41f_runtime.h"

#include <errno.h>
#include <string.h>

int ds41f_runtime_init(ds41f_runtime *rt, int rank, int ranks, int threads,
                       const char *stage_dir, const char *engram_dir)
{
    if (!rt || rank < 0 || ranks != DS41F_EP_RANKS || rank >= ranks || threads <= 0 ||
        !stage_dir || !engram_dir) return EINVAL;
    memset(rt, 0, sizeof *rt);
    rt->rank = rank; rt->ranks = ranks; rt->threads = threads;
    rt->stage_dir = stage_dir; rt->engram_dir = engram_dir;
    ds41f_model_config_init(&rt->config);
    return 0;
}

int ds41f_runtime_validate(const ds41f_runtime *rt)
{
    if (!rt || rt->ranks != DS41F_EP_RANKS || rt->rank < 0 || rt->rank >= rt->ranks ||
        rt->threads <= 0 || !rt->stage_dir || !rt->engram_dir) return EINVAL;
    return ds41f_model_config_valid(&rt->config) ? 0 : EINVAL;
}
