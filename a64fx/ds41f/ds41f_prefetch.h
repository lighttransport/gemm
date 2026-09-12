#ifndef DS41F_PREFETCH_H
#define DS41F_PREFETCH_H
#include "ds41f_engram.h"
typedef struct ds41f_prefetch ds41f_prefetch;
/* One submitting/consuming thread; one worker performs local reads only.
 * A request contains both Engram layers and must finish before resubmission. */
int ds41f_prefetch_create(ds41f_prefetch **out,ds41f_engram *engram);
int ds41f_prefetch_submit(ds41f_prefetch *p,const uint64_t ids[2][24]);
int ds41f_prefetch_wait(ds41f_prefetch *p,int slot,float rows[24*256],double *read_seconds);
/* Finish all local reads and invalidate both saved results before rollback.
 * A later submit starts a new generation; stale waits return EINVAL. */
int ds41f_prefetch_drain(ds41f_prefetch *p);
uint64_t ds41f_prefetch_generation(ds41f_prefetch *p);
void ds41f_prefetch_destroy(ds41f_prefetch *p);
#endif
