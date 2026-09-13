#define _GNU_SOURCE
#include "ds41f_alloc.h"
#include "ds41f_fp4_sdot.h"
#include "ds41f_weights.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int check_packed_store(void)
{
    ds41f_weights store={0};store.count=2;store.items=calloc(2,sizeof *store.items);store.bytes=68;
    if(!store.items)return 1;
    ds41f_weight *scale=store.items,*weight=store.items+1;
    strcpy(scale->name,"layers.0.ffn.experts.0.w1.scale");strcpy(scale->dtype,"F8_E8M0");
    scale->rows=4;scale->cols=1;scale->bytes=4;
    strcpy(weight->name,"layers.0.ffn.experts.0.w1.weight");strcpy(weight->dtype,"I8");
    weight->rows=4;weight->cols=16;weight->bytes=64;
    if(ds41f_weights_pack_experts(&store,136)!=EINVAL)return 1;
    store.fresh_pages=1;
    if(ds41f_alloc_resident(&scale->data,4,1)||ds41f_alloc_resident(&weight->data,64,1))return 1;
    for(int i=0;i<4;++i)((uint8_t *)scale->data)[i]=(uint8_t)(125+i);
    for(int i=0;i<64;++i)((uint8_t *)weight->data)[i]=(uint8_t)(i*17);
    uint8_t expected[64],expected_scale[4];
    if(ds41f_mxfp4_pack_sdot(expected,expected_scale,weight->data,scale->data,4,32))return 1;
    if(ds41f_weights_pack_experts(&store,135)!=ENOMEM)return 1;
    if(ds41f_weights_pack_experts(&store,136)||!store.packed_experts||store.bytes!=68||
       memcmp(weight->data,expected,64)||memcmp(scale->data,expected_scale,4))return 1;
    if(ds41f_weights_pack_experts(&store,136)!=EINVAL)return 1;
    ds41f_weights_free(&store);puts("EXPERT_STORE PASS fresh_pages bounded_peak lossless_pack teardown");return 0;
}
int main(void)
{
    if(check_packed_store())return 1;
    ds41f_weights store={0};store.count=2;store.items=calloc(2,sizeof *store.items);if(!store.items)return 1;
    ds41f_weight *s=&store.items[0],*w=&store.items[1];
    strcpy(s->name,"unit.scale");strcpy(s->dtype,"F8_E8M0");s->rows=1;s->cols=160;s->bytes=160;s->data=malloc(160);
    strcpy(w->name,"unit.weight");strcpy(w->dtype,"F8_E4M3");w->rows=8;w->cols=5120;w->bytes=8*5120;
    uint8_t *raw=malloc(w->bytes);float *x=malloc(10240*4),ref[8],got[8];if(!raw||!s->data||!x)return 1;
    for(size_t i=0;i<w->bytes;++i)raw[i]=(uint8_t)((i*11+i/17)%112);
    for(size_t i=0;i<160;++i)((uint8_t *)s->data)[i]=(uint8_t)(122+i%8);
    for(size_t block=32;block<=128;block*=2){
        ds41f_int8_free(&w->int8);if(ds41f_int8_from_fp8(&w->int8,raw,s->data,8,5120,block))return 1;
        for(int trial=0;trial<12;++trial){
            for(size_t i=0;i<10240;++i)x[i]=sinf((float)i*.13f)*(trial%3);
            for(int raw_input=0;raw_input<2;++raw_input){
                void *saved=store.input_cache;store.input_cache=NULL;
                int rc=ds41f_linear(&store,"unit",ref,x,raw_input);store.input_cache=saved;if(rc)return 1;
                if(ds41f_weights_enable_input_cache(&store))return 1;
                for(int repeat=0;repeat<2;++repeat){if(ds41f_linear(&store,"unit",got,x,raw_input))return 1;
                    if(memcmp(ref,got,sizeof ref)){fprintf(stderr,"CACHE mismatch block=%zu trial=%d raw=%d\n",block,trial,raw_input);return 1;}}
            }
            if(ds41f_int8_matvec(ref,&w->int8,x,4,0)||ds41f_linear_int8_cached(&store,w,got,x,4,0)||memcmp(ref,got,sizeof ref))return 1;
        }
    }
    float *bx=malloc(6*10240*sizeof(float)),br[48],bg[48];if(!bx)return 1;
    for(size_t i=0;i<6*10240;++i)bx[i]=sinf((float)i*.087f)*.25f;
    ds41f_int8_free(&w->int8);if(ds41f_int8_from_fp8(&w->int8,raw,s->data,8,5120,32))return 1;
    for(size_t batch=1;batch<=6;++batch){
        for(int raw_input=0;raw_input<2;++raw_input){
            for(size_t i=0;i<batch;++i)if(ds41f_linear(&store,"unit",br+i*8,bx+i*5120,raw_input))return 1;
            if(ds41f_linear_batch(&store,"unit",bg,8,bx,5120,batch,raw_input)||memcmp(br,bg,batch*8*sizeof(float)))return 1;
        }
        for(size_t i=0;i<batch;++i)if(ds41f_int8_matvec(br+i*8,&w->int8,bx+i*10240,4,0))return 1;
        if(ds41f_int8_linear_batch(w,bg,8,bx,10240,batch,4,0)||memcmp(br,bg,batch*8*sizeof(float)))return 1;
    }
    free(bx);
    x[0]=NAN;
    if(ds41f_linear(&store,"unit",got,x,0)!=EDOM||ds41f_linear(&store,"unit",got,x,0)!=EDOM)return 1;
    ds41f_weights_free(&store);free(raw);free(x);
    puts("INPUT_CACHE PASS exact raw/FP8 boundaries grouped reused_addresses changing_blocks repeat_hits nonfinite");return 0;
}
