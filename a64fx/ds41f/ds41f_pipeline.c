#define _POSIX_C_SOURCE 200809L
#include "ds41f_pipeline.h"
#include <errno.h>
#include <stdio.h>
#include <string.h>
#define PIPE_MAGIC 0x44533450u
typedef struct {uint32_t magic,version,tile,position,count;} pipeline_envelope_header;
_Static_assert(sizeof(pipeline_envelope_header)==20,"pipeline envelope header ABI");
static size_t envelope_stride(void){return 20480u*2u+4u*4u+sizeof(ds41f_pipeline_wire);}
size_t ds41f_pipeline_envelope_bytes(void){return sizeof(pipeline_envelope_header)+(size_t)DS41F_PIPELINE_TILE_MAX*envelope_stride();}
int ds41f_pipeline_stage_for_layer(int layer){return layer<0||layer>=40?-1:layer<13?0:layer<27?1:2;}
int ds41f_pipeline_owner(int layer){int s=ds41f_pipeline_stage_for_layer(layer);return s<0?-1:s*4+(layer&3);}
int ds41f_pipeline_layer_local(const ds41f_pipeline *p,int layer){return p&&p->enabled&&layer>=p->first_layer&&layer<=p->last_layer;}
int ds41f_pipeline_boundary(int stage,int *source,int *target){if(!source||!target||stage<0||stage>1)return EINVAL;*source=stage?6:0;*target=stage?11:5;return 0;}
int ds41f_pipeline_open(ds41f_pipeline *p,const char *stage,int rank){
    if(!p||!stage||rank<0||rank>=12)return EINVAL;
    memset(p,0,sizeof *p);char path[4096];
    int n=snprintf(path,sizeof path,"%s/pipeline.index",stage);if(n<0||(size_t)n>=sizeof path)return ENAMETOOLONG;
    FILE *f=fopen(path,"r");if(!f)return errno;int version,stages,tp,stored,stage_id,first,last;unsigned long long bytes;
    int fields=fscanf(f,"DS41FPIPE %d %d %d %d %d %d %d %llu",&version,&stages,&tp,&stored,&stage_id,&first,&last,&bytes);fclose(f);
    if(fields!=8||version!=1||stages!=3||tp!=4||stored!=rank||stage_id!=rank/4||first<0||last<first||last>=40||
       (stage_id==0&&(first!=0||last!=12))||(stage_id==1&&(first!=13||last!=26))||(stage_id==2&&(first!=27||last!=39)))return EINVAL;
    p->enabled=1;p->rank=rank;p->stage=stage_id;p->first_layer=first;p->last_layer=last;p->ranks=12;p->tp=4;p->resident_bytes=(size_t)bytes;return 0;
}
int ds41f_pipeline_wire_init(ds41f_pipeline_wire *w,uint32_t tile,uint32_t pos,uint32_t count,const uint32_t *s,size_t ns,const uint32_t *c,size_t nc,const uint8_t *pub,size_t np){
    if(!w||ns>512||nc>2048||np>356||(ns&&!s)||(nc&&!c)||(np&&!pub))return EINVAL;
    memset(w,0,sizeof *w);w->header.magic=PIPE_MAGIC;w->header.version=1;w->header.tile=tile;w->header.position=pos;w->header.count=count;w->header.selected_count=(uint16_t)ns;w->header.candidate_count=(uint16_t)nc;w->header.publication_bytes=(uint16_t)np;
    if(ns)memcpy(w->selected,s,ns*sizeof *s);
    if(nc)memcpy(w->candidates,c,nc*sizeof *c);
    if(np)memcpy(w->publication,pub,np);
    return 0;
}
size_t ds41f_pipeline_wire_size(const ds41f_pipeline_wire *w){
    if(!w||w->header.selected_count>512||w->header.candidate_count>2048||w->header.publication_bytes>356)return 0;
    return sizeof w->header+(size_t)w->header.selected_count*4+(size_t)w->header.candidate_count*4+w->header.publication_bytes;
}
int ds41f_pipeline_wire_check(const ds41f_pipeline_wire *w,size_t bytes,uint32_t tile,uint32_t pos,uint32_t count){
    if(!w||bytes<sizeof w->header||w->header.magic!=PIPE_MAGIC||w->header.version!=1||w->header.tile!=tile||w->header.position!=pos||w->header.count!=count||w->header.selected_count>512||w->header.candidate_count>2048||w->header.publication_bytes>356)return EINVAL;
    size_t need=ds41f_pipeline_wire_size(w);return bytes<need?EMSGSIZE:0;
}
int ds41f_pipeline_envelope_pack(void *dst,size_t bytes,uint32_t tile,uint32_t pos,uint32_t count,
                                 const float *residual,size_t residual_stride,
                                 const ds41f_pipeline_wire *metadata,size_t metadata_stride)
{
    if(!dst||bytes<ds41f_pipeline_envelope_bytes()||count>DS41F_PIPELINE_TILE_MAX||
       !residual||residual_stride<20484||!metadata||metadata_stride<sizeof(*metadata))return EINVAL;
    pipeline_envelope_header h={PIPE_MAGIC,1,tile,pos,count};memcpy(dst,&h,sizeof h);
    uint8_t *out=(uint8_t *)dst+sizeof h;
    for(uint32_t t=0;t<DS41F_PIPELINE_TILE_MAX;++t){
        /* Keep a fixed wire size, but never read beyond the valid tile count.
         * Full tiles do not need a redundant clear before being overwritten. */
        if(t<count){
            const float *src=residual+(size_t)t*residual_stride;uint16_t *bf=(uint16_t *)out;
            for(size_t i=0;i<20480;++i){uint32_t bits;memcpy(&bits,src+i,4);if(bits&0xffffu)return EINVAL;bf[i]=(uint16_t)(bits>>16);}
            memcpy(out+40960,src+20480,16);memcpy(out+40976,(const uint8_t *)metadata+(size_t)t*metadata_stride,sizeof(*metadata));
        } else memset(out,0,envelope_stride());
        out+=envelope_stride();
    }
    return 0;
}
int ds41f_pipeline_envelope_unpack(const void *src,size_t bytes,uint32_t tile,uint32_t pos,uint32_t count,
                                   float *residual,size_t residual_stride,ds41f_pipeline_wire *metadata,size_t metadata_stride)
{
    if(!src||bytes<ds41f_pipeline_envelope_bytes()||!residual||residual_stride<20484||!metadata||metadata_stride<sizeof(*metadata))return EINVAL;
    pipeline_envelope_header h;memcpy(&h,src,sizeof h);if(h.magic!=PIPE_MAGIC||h.version!=1||h.tile!=tile||h.position!=pos||h.count!=count||count>DS41F_PIPELINE_TILE_MAX)return EINVAL;
    const uint8_t *in=(const uint8_t *)src+sizeof h;
    for(uint32_t t=0;t<count;++t){float *dst=residual+(size_t)t*residual_stride;const uint16_t *bf=(const uint16_t *)in;
        for(size_t i=0;i<20480;++i){uint32_t bits=(uint32_t)bf[i]<<16;memcpy(dst+i,&bits,4);}memcpy(dst+20480,in+40960,16);memcpy((uint8_t *)metadata+(size_t)t*metadata_stride,in+40976,sizeof(*metadata));in+=envelope_stride();}
    return 0;
}
