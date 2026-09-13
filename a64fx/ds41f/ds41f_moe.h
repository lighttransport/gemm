#ifndef DS41F_MOE_H
#define DS41F_MOE_H

#include <stddef.h>

#include "ds41f_model.h"

typedef struct {
    int expert;
    float score;
    int slot;
} ds41f_route;

int ds41f_route_topk(const float *logits, int n, int k, ds41f_route *out);
int ds41f_route_owner(const ds41f_route *routes, int n, int rank, int ranks,
                      ds41f_route *out);
void ds41f_moe_weighted_add(float *dst, const float *expert_out,
                            const ds41f_route *routes, int n, size_t hidden);

#endif
