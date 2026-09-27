#include "ds41f_pipeline_runtime.h"
#include <errno.h>

int ds41f_pipeline_run_ex(size_t tiles, size_t tile_tokens, size_t total_tokens,
                          const ds41f_pipeline *pipeline, ds41f_pipeline_stage_fn callback,
                          void *context, const ds41f_pipeline_transport *transport)
{
    if (!tiles || !tile_tokens || tile_tokens > DS41F_PIPELINE_TILE_MAX || !total_tokens || !pipeline || !pipeline->enabled ||
        !callback || pipeline->stage < 0 || pipeline->stage >= DS41F_PIPELINE_STAGES ||
        pipeline->rank < 0 || pipeline->rank >= 12) return EINVAL;
    if (tiles > SIZE_MAX - (DS41F_PIPELINE_STAGES - 1) ||
        tiles > SIZE_MAX / tile_tokens) return EOVERFLOW;
    unsigned char receive_posted[2] = {0, 0};
    for (size_t wave = 0; wave < tiles + DS41F_PIPELINE_STAGES - 1; ++wave) {
        if (wave < (size_t)pipeline->stage) continue;
        size_t tile = wave - (size_t)pipeline->stage;
        if (tile >= tiles) continue;
        size_t position = tile * tile_tokens;
        size_t count = total_tokens - position;
        if (count > tile_tokens) count = tile_tokens;
        int slot=(int)(tile&1),rc;
        if(transport&&tile>=2&&pipeline->stage<DS41F_PIPELINE_STAGES-1&&transport->wait_send){
            rc=transport->wait_send(transport->context,pipeline,tile-2,slot);
            if(rc)return rc;
        }
        if(transport&&pipeline->stage>0&&transport->post_receive){
            if(!receive_posted[slot]) {
                rc=transport->post_receive(transport->context,pipeline,tile,position,count,slot);
                if(rc)return rc;
                receive_posted[slot]=1;
            }
            if(transport->wait_receive){rc=transport->wait_receive(transport->context,pipeline,tile,slot);if(rc)return rc;}
            receive_posted[slot]=0;
        }
        rc = callback(context, pipeline, tile, position, count, slot);
        if (rc) return rc;
        if(transport&&pipeline->stage<DS41F_PIPELINE_STAGES-1&&transport->post_send){
            rc=transport->post_send(transport->context,pipeline,tile,position,count,slot);
            if(rc)return rc;
        }
        /* Post the next receive while this tile is being consumed by the
         * stage callback.  The next source send can then rendezvous without
         * adding a receive-control bubble to the wavefront. */
        if(transport&&pipeline->stage>0&&transport->post_receive&&tile+1<tiles){
            size_t next=tile+1, next_pos=next*tile_tokens;
            size_t next_count=total_tokens-next_pos;
            if(next_count>tile_tokens) next_count=tile_tokens;
            int next_slot=(int)(next&1);
            if(!receive_posted[next_slot]) {
                rc=transport->post_receive(transport->context,pipeline,next,next_pos,next_count,next_slot);
                if(rc)return rc;
                receive_posted[next_slot]=1;
            }
        }
    }
    if(transport&&pipeline->stage<DS41F_PIPELINE_STAGES-1&&transport->wait_send){
        size_t first=tiles>2?tiles-2:0;
        for(size_t tile=tiles;tile-- >first;){int rc=transport->wait_send(transport->context,pipeline,tile,(int)(tile&1));if(rc)return rc;}
    }
    return 0;
}
int ds41f_pipeline_run(size_t tiles, size_t tile_tokens, size_t total_tokens,
                       const ds41f_pipeline *pipeline, ds41f_pipeline_stage_fn callback,
                       void *context)
{ return ds41f_pipeline_run_ex(tiles,tile_tokens,total_tokens,pipeline,callback,context,NULL); }
