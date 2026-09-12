#include "ds41f_attention.h"
#include "ds41f_cache.h"
#include "ds41f_kernels.h"
#include "ds41f_sve.h"
#include "ds41f_ops.h"
#include "ds41f_profile.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(call) do {int check_rc=(call);if(check_rc)return check_rc;} while(0)
static int source(int layer){return layer<8?0:layer<14?1:layer<20?2:3;}
static int is_source(int l){return l==2||l==8||l==14||l==20;}
static int is_index(int l){return is_source(l)||l==24||l==28||l==32||l==36;}
static float bf(float x){return ds41f_bf16_to_f32(ds41f_f32_to_bf16(x));}
static int linear(const ds41f_weights *w,int layer,const char *part,float *out,const float *x,int raw)
{char name[192];snprintf(name,sizeof name,"layers.%d.attn.%s",layer,part);return ds41f_linear(w,name,out,x,raw);}
static int norm(const ds41f_weights *w,int layer,const char *part,float *out,const float *x)
{char name[192];snprintf(name,sizeof name,"layers.%d.attn.%s.weight",layer,part);return ds41f_norm(w,name,out,x);}
static void rope(float *x,size_t heads,size_t dim,int layer,size_t pos,int inverse)
{ds41f_rope(x,heads,dim,64,pos,layer<2?10000:160000,16,layer<2?0:65536,inverse);ds41f_round_bf16(x,heads*dim);}
int ds41f_attention_init(ds41f_attention *s,size_t capacity)
{
    if(!s||!capacity||capacity>1048576)return EINVAL;
    memset(s,0,sizeof *s);s->capacity=capacity;
    for(int i=0;i<4;++i){size_t rows=i==3?capacity:(capacity+1)/2;
        s->compressed[i]=calloc(rows,356);if(!s->compressed[i]){ds41f_attention_free(s);return ENOMEM;}
        /* Commit cache pages now: the admission check must cover the real
         * configured cache, not only a lazily reserved address range. */
        #pragma omp parallel for schedule(static)
        for(size_t page=0;page<(rows*356+4095)/4096;++page)
            ((volatile uint8_t *)s->compressed[i])[page*4096]=0;
    }
    s->window=calloc((size_t)40*128*512,sizeof(float));
    s->rows=malloc((size_t)640*512*sizeof(float));
    s->candidate_blocks=calloc((capacity+7)/8,1);
    if(!s->window||!s->rows||!s->candidate_blocks){ds41f_attention_free(s);return ENOMEM;}return 0;
}
void ds41f_attention_free(ds41f_attention *s)
{if(!s)return;for(int i=0;i<4;++i)free(s->compressed[i]);free(s->window);free(s->rows);free(s->candidate_blocks);memset(s,0,sizeof *s);}
int ds41f_attention_receive(ds41f_attention *s,int layer,size_t pos,const uint8_t row[356])
{
    if(!s||!row||pos>=s->capacity||layer<0||layer>=40)return EINVAL;
    if(!is_source(layer))return 0;
    int ratio=layer<20?2:1;if((pos+1)%ratio)return 0;
    memcpy(s->compressed[source(layer)]+(pos/ratio)*356,row,356);return 0;
}
static int update_source(ds41f_attention *s,const ds41f_weights *w,int layer,size_t pos,const float *x)
{
    int src=source(layer),ratio=layer<20?2:1;float latent[512],value[512],score[512],key[128];
    CHECK(linear(w,layer,"compressor.wkv",value,x,ratio==2));
    if(ratio==2){CHECK(linear(w,layer,"compressor.wgate",score,x,1));
        if(!(pos&1)){memcpy(s->pool_value[src],value,sizeof value);memcpy(s->pool_score[src],score,sizeof score);return 0;}
        ds41f_pool_pair(latent,s->pool_value[src],value,s->pool_score[src],score,512);ds41f_round_bf16(latent,512);
    }else memcpy(latent,value,sizeof latent);
    CHECK(norm(w,layer,"compressor.norm",latent,latent));
    CHECK(linear(w,layer,"indexer.wk",key,latent,0));
    CHECK(norm(w,layer,"indexer.k_norm",key,key));
    size_t group_pos=pos+1-ratio;
    rope(key,1,128,layer,group_pos,0);
    CHECK(ds41f_fp4_pack(s->publication+288,key,128,32,0));
    rope(latent,1,512,layer,group_pos,0);
    CHECK(ds41f_fp4_pack(s->publication,latent,512,16,1));
    return ds41f_attention_receive(s,layer,pos,s->publication);
}
static int select_positions(ds41f_attention *s,const ds41f_weights *w,int layer,size_t pos,const float *x,const float *qr)
{
    size_t count=(pos+1)/(layer<20?2:1);s->selected_count=0;if(!count)return 0;
    double pt=P_BEGIN();
    float q[32*128],weights[32];uint8_t packed[68];
    CHECK(linear(w,layer,"indexer.wq_b",q,qr,0));rope(q,32,128,layer,pos,0);
    for(int h=0;h<32;++h){CHECK(ds41f_fp4_pack(packed,q+h*128,128,32,0));CHECK(ds41f_fp4_unpack(q+h*128,packed,128,32,0));}
    CHECK(linear(w,layer,"indexer.weights_proj",weights,x,0));
    for(int h=0;h<32;++h)weights[h]=bf(weights[h]/64.f);
    float *scores=malloc(count*sizeof(float));if(!scores)return ENOMEM;
    const uint8_t *rows=s->compressed[source(layer)];
    P_END(INDEX_QUERY,pt);pt=P_BEGIN();
    #pragma omp parallel for schedule(static)
    for(size_t i=0;i<count;++i){
        /* Later query sources only score the candidate source's retained
         * blocks. Do not decode/score up to 1M rows merely to mask them later. */
        if(layer>20&&!s->candidate_blocks[i/8]){scores[i]=-INFINITY;continue;}
        float key[128],sum=0;ds41f_fp4_unpack(key,rows+i*356+288,128,32,0);
        for(int h=0;h<32;++h){float dot=0;
            for(int j=0;j<128;++j)dot+=q[h*128+j]*key[j];
            sum+=bf(fmaxf(bf(dot),0)*weights[h]);}scores[i]=bf(sum);}
    P_END(INDEX_SCORE,pt);pt=P_BEGIN();
    if(layer==20){size_t blocks=(count+7)/8;float *block_scores=malloc(blocks*sizeof(float));int *ids=malloc(2048*sizeof(int));
        if(!block_scores||!ids){free(block_scores);free(ids);free(scores);return ENOMEM;}
        for(size_t b=0;b<blocks;++b){float best=-INFINITY;for(size_t j=b*8;j<count&&j<b*8+8;++j)best=fmaxf(best,scores[j]);block_scores[b]=best;}
        block_scores[blocks-1]=INFINITY;
        size_t kept=ds41f_select_topk(block_scores,blocks,2048,ids);memset(s->candidate_blocks,0,(s->capacity+7)/8);
        for(size_t i=0;i<kept;++i)s->candidate_blocks[ids[i]]=1;
        free(block_scores);free(ids);
    }
    s->selected_count=ds41f_select_topk(scores,count,512,s->selected);free(scores);P_END(INDEX_SELECT,pt);return 0;
}
static int grouped_output(const ds41f_weights *w,int layer,float *out,const float *x)
{
    char name[192];snprintf(name,sizeof name,"layers.%d.attn.wo_a.weight",layer);
    const ds41f_weight *weight=ds41f_weight_find(w,name);
    snprintf(name,sizeof name,"layers.%d.attn.wo_a.scale",layer);
    const ds41f_weight *scale=ds41f_weight_find(w,name);
    if(!weight||!scale||strcmp(weight->dtype,"F8_E4M3")||weight->rows!=8192||weight->cols!=4096||scale->bytes!=256*128)return EINVAL;
    double pt=P_BEGIN();
    if(weight->int8.weight){P_VALUE(INT8_BYTES,weight->int8.bytes);
        CHECK(ds41f_int8_matvec(out,&weight->int8,x,1024,0));P_END(LINEAR_INT8,pt);}
    else{P_VALUE(FP8_BYTES,weight->bytes+scale->bytes);
        CHECK(ds41f_fp8_grouped_matvec(out,weight->data,scale->data,x,8,1024,4096));P_END(LINEAR_FP8,pt);}
    pt=P_BEGIN();ds41f_round_bf16(out,8192);P_END(LINEAR_ROUND,pt);return 0;
}
int ds41f_attention_step(ds41f_attention *s,const ds41f_weights *w,int layer,size_t pos,const float *x,float *out)
{
    if(!s||!w||!x||!out||!s->rows||layer<0||layer>=40||pos>=s->capacity)return EINVAL;
    float qr[1280],q[64*512],kv[512],attended[64*512],projected[8192];
    double pt=P_BEGIN();CHECK(linear(w,layer,"wq_a",qr,x,0));CHECK(norm(w,layer,"q_norm",qr,qr));P_END(ATTN_QA,pt);
    pt=P_BEGIN();CHECK(linear(w,layer,"wq_b",q,qr,0));P_END(ATTN_QB,pt);
    pt=P_BEGIN();rope(q,64,512,layer,pos,0);P_END(ATTN_Q_ROPE,pt);pt=P_BEGIN();
    CHECK(linear(w,layer,"wkv",kv,x,0));CHECK(norm(w,layer,"kv_norm",kv,kv));
    rope(kv,1,512,layer,pos,0);CHECK(ds41f_act_quant(kv,kv,512));
    float *window=s->window+(size_t)layer*128*512;
    memcpy(window+(pos%128)*512,kv,sizeof kv);
    P_END(ATTN_KV,pt);pt=P_BEGIN();
    if(is_source(layer))CHECK(update_source(s,w,layer,pos,x));
    P_END(ATTN_COMPRESS,pt);pt=P_BEGIN();
    if(is_index(layer))CHECK(select_positions(s,w,layer,pos,x,qr));
    P_END(ATTN_INDEX,pt);pt=P_BEGIN();
    size_t raw_count=pos<128?pos+1:128,extra=layer<2?0:s->selected_count;
    if(extra>512)return EINVAL;
    float *rows=s->rows;
    int ids[128+512];
    for(size_t i=0;i<raw_count;++i){size_t token=pos+1-raw_count+i;
        memcpy(rows+i*512,window+(token%128)*512,512*sizeof(float));ids[i]=(int)i;}
    #pragma omp parallel for schedule(static)
    for(size_t i=0;i<extra;++i){
        ds41f_fp4_unpack(rows+(raw_count+i)*512,s->compressed[source(layer)]+(size_t)s->selected[i]*356,512,16,1);
        ids[raw_count+i]=(int)(raw_count+i);}
    char name[192];snprintf(name,sizeof name,"layers.%d.attn.attn_sink",layer);
    const ds41f_weight *sink=ds41f_weight_find(w,name);
    if(!sink||sink->bytes!=64*4)return EINVAL;
    P_END(ATTN_ROWS,pt);pt=P_BEGIN();
    int rc=ds41f_sparse_attention(attended,q,rows,sink->data,ids,raw_count+extra,raw_count+extra,64,512);
    if(rc)return rc;P_END(ATTN_SPARSE,pt);pt=P_BEGIN();
    ds41f_round_bf16(attended,64*512);rope(attended,64,512,layer,pos,1);
    P_END(ATTN_INVERSE_ROPE,pt);pt=P_BEGIN();CHECK(grouped_output(w,layer,projected,attended));P_END(ATTN_WOA,pt);
    pt=P_BEGIN();rc=linear(w,layer,"wo_b",out,projected,0);P_END(ATTN_WOB,pt);return rc;
}
