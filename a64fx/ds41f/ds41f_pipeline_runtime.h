#ifndef DS41F_PIPELINE_RUNTIME_H
#define DS41F_PIPELINE_RUNTIME_H
#include "ds41f_pipeline.h"
typedef int (*ds41f_pipeline_stage_fn)(void *context, const ds41f_pipeline *pipeline,
                                       size_t tile, size_t position, size_t count, int slot);
typedef struct {
    int (*post_receive)(void *context,const ds41f_pipeline *pipeline,size_t tile,
                        size_t position,size_t count,int slot);
    int (*wait_receive)(void *context,const ds41f_pipeline *pipeline,size_t tile,int slot);
    int (*post_send)(void *context,const ds41f_pipeline *pipeline,size_t tile,
                     size_t position,size_t count,int slot);
    int (*wait_send)(void *context,const ds41f_pipeline *pipeline,size_t tile,int slot);
    void *context;
} ds41f_pipeline_transport;
/* All ranks call this function.  A rank executes its own stage callback on
 * wave W for tile W-stage; the other waves are bubbles for that rank. */
int ds41f_pipeline_run(size_t tiles, size_t tile_tokens, size_t total_tokens,
                       const ds41f_pipeline *pipeline, ds41f_pipeline_stage_fn callback,
                       void *context);
int ds41f_pipeline_run_ex(size_t tiles, size_t tile_tokens, size_t total_tokens,
                          const ds41f_pipeline *pipeline, ds41f_pipeline_stage_fn callback,
                          void *context, const ds41f_pipeline_transport *transport);
#endif
