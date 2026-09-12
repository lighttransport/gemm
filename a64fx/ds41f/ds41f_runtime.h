#ifndef DS41F_RUNTIME_H
#define DS41F_RUNTIME_H

#include <stdint.h>

#include "ds41f_model.h"

typedef struct {
    int rank;
    int ranks;
    int threads;
    const char *stage_dir;
    const char *engram_dir;
    ds41f_model_config config;
} ds41f_runtime;

/* Initializes the rank-independent V4.1 contract.  uTofu registration and
 * tensor allocation are deliberately separate so this can be unit-tested on
 * x86 before launching a 12-node job. */
int ds41f_runtime_init(ds41f_runtime *rt, int rank, int ranks, int threads,
                       const char *stage_dir, const char *engram_dir);
int ds41f_runtime_validate(const ds41f_runtime *rt);

#endif
