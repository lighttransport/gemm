#ifndef GLM53F_MEMORY_BUDGET_H
#define GLM53F_MEMORY_BUDGET_H
#include <stdint.h>

/* Account before resident allocations. available is current MemAvailable,
 * so do not count already-resident bytes a second time. transient includes
 * source I/O, repacking copies and temporary kernel panels at their peak. */
typedef struct {
    uint64_t resident, transient;
} glm53f_memory_budget;
static inline int glm53f_memory_budget_add(glm53f_memory_budget *b,
        uint64_t resident, uint64_t transient) {
    if (!b || resident > UINT64_MAX - b->resident ||
        transient > UINT64_MAX - b->transient) return -1;
    b->resident += resident; b->transient += transient;
    return 0;
}
static inline int glm53f_memory_budget_fits(const glm53f_memory_budget *b,
        uint64_t available, uint64_t *headroom) {
    const uint64_t minimum = UINT64_C(6) * 1024 * 1024 * 1024;
    if (!b || !headroom || b->resident > available ||
        b->transient > available - b->resident) return -1;
    *headroom = available - b->resident - b->transient;
    return *headroom >= minimum ? 0 : -1;
}
#endif
