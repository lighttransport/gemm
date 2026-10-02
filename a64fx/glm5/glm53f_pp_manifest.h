#ifndef GLM53F_PP_MANIFEST_H
#define GLM53F_PP_MANIFEST_H
#include "glm53f_dist.h"
#include <stdio.h>
#include <string.h>

/* Check the layout/stage contract before reading tensor entries or allocating
 * their resident payloads. Tensor ranges, byte counts and hashes are validated
 * by the component loader. TP12 manifests have a separate namespace. */
static inline int glm53f_pp_manifest_check(const char *path, const char *component,
        const glm53f_dist *d, int first, int end) {
    if (!path || !component || !d || !d->initialized ||
        d->config.layout != GLM53F_PP3_TP4 || first < d->map.first_layer ||
        end > d->map.end_layer || first >= end) return -1;
    FILE *f = fopen(path, "r"); if (!f) return -1;
    char line[2048], expected[512];
    int n = snprintf(expected, sizeof(expected), "# GLM53F_PP_%s_V1 layout=pp3-tp4 "
        "world_rank=%d stage=%d tp_rank=%d tp_size=4 cuts=%d,%d layers=%d:%d",
        component, d->map.world_rank, d->map.stage, d->map.tp_rank,
        d->config.cuts[0], d->config.cuts[1], first, end);
    int valid = n > 0 && n < (int)sizeof(expected) && fgets(line, sizeof(line), f) &&
        !strncmp(line, expected, (size_t)n) && (line[n] == ' ' || line[n] == '\n');
    if (fclose(f)) return -1;
    return valid ? 0 : -1;
}
#endif
