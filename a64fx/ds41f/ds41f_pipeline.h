#ifndef DS41F_PIPELINE_H
#define DS41F_PIPELINE_H
#include <stddef.h>
#include <stdint.h>
enum { DS41F_PIPELINE_STAGES=3, DS41F_PIPELINE_TP=4, DS41F_PIPELINE_LAYERS=40,
       DS41F_PIPELINE_SELECTED_MAX=512, DS41F_PIPELINE_CANDIDATE_MAX=2048,
       DS41F_PIPELINE_PUBLICATION_BYTES=356, DS41F_PIPELINE_TILE_MAX=64 };
typedef struct { int enabled,rank,stage,first_layer,last_layer,ranks,tp; size_t resident_bytes; } ds41f_pipeline;
typedef struct { uint32_t magic,version,tile,position,count; uint16_t selected_count,candidate_count,publication_bytes,reserved; } ds41f_pipeline_wire_header;
typedef struct { ds41f_pipeline_wire_header header; uint32_t selected[DS41F_PIPELINE_SELECTED_MAX]; uint32_t candidates[DS41F_PIPELINE_CANDIDATE_MAX]; uint8_t publication[DS41F_PIPELINE_PUBLICATION_BYTES]; } ds41f_pipeline_wire;
int ds41f_pipeline_open(ds41f_pipeline *,const char *,int);
int ds41f_pipeline_owner(int);
int ds41f_pipeline_stage_for_layer(int);
int ds41f_pipeline_layer_local(const ds41f_pipeline *,int);
int ds41f_pipeline_boundary(int,int *,int *);
int ds41f_pipeline_wire_init(ds41f_pipeline_wire *,uint32_t,uint32_t,uint32_t,const uint32_t *,size_t,const uint32_t *,size_t,const uint8_t *,size_t);
size_t ds41f_pipeline_wire_size(const ds41f_pipeline_wire *);
int ds41f_pipeline_wire_check(const ds41f_pipeline_wire *,size_t,uint32_t,uint32_t,uint32_t);
size_t ds41f_pipeline_envelope_bytes(void);
int ds41f_pipeline_envelope_pack(void *,size_t,uint32_t,uint32_t,uint32_t,
                                 const float *,size_t,const ds41f_pipeline_wire *,size_t);
int ds41f_pipeline_envelope_unpack(const void *,size_t,uint32_t,uint32_t,uint32_t,
                                   float *,size_t,ds41f_pipeline_wire *,size_t);
#endif
