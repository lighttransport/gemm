#include "ds41f_team.h"
#include "ds41f_attention.h"
#include "ds41f_alloc.h"
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
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

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
{
    /* Every caller supplies a BF16-rounded vector. Only the rotary suffix
     * changes, so rounding the untouched prefix again is redundant. */
    ds41f_rope(x,heads,dim,64,pos,layer<2?10000:160000,16,layer<2?0:65536,inverse);
    for(size_t h=0;h<heads;++h)ds41f_round_bf16(x+h*dim+dim-64,64);
}
int ds41f_attention_init(ds41f_attention *s,size_t capacity)
{
    if(!s||!capacity||capacity>1048576)return EINVAL;
    memset(s,0,sizeof *s);s->capacity=capacity;s->selected_limit=512;
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
{if(!s)return;for(int i=0;i<4;++i)free(s->compressed[i]);free(s->window);
    ds41f_free_resident(s->rows,(size_t)640*512*sizeof(float),s->rows_fresh_pages);
    ds41f_free_resident(s->decoded_rows,s->decoded_capacity*512*sizeof(float),s->decoded_fresh_pages);
    free(s->decoded_keys);free(s->candidate_blocks);memset(s,0,sizeof *s);}
int ds41f_attention_enable_row_cache(ds41f_attention *s,size_t rows)
{
    if(!s||!rows||rows>65536||s->decoded_rows)return EINVAL;
    if(rows>SIZE_MAX/512/sizeof(float))return EOVERFLOW;
    size_t bytes=rows*512*sizeof(float);float *values=NULL;uint64_t *keys=calloc(rows,sizeof *keys);
    if(!keys)return ENOMEM;
    int rc=ds41f_alloc_resident((void **)&values,bytes,1);
    if(rc){free(keys);return rc;}
    for(size_t i=0;i<rows;++i)keys[i]=UINT64_MAX;
    s->decoded_rows=values;s->decoded_keys=keys;s->decoded_capacity=rows;s->decoded_clock=0;s->decoded_fresh_pages=1;
    return 0;
}
void ds41f_attention_clear_row_cache(ds41f_attention *s)
{
    if(!s||!s->decoded_keys)return;
    for(size_t i=0;i<s->decoded_capacity;++i)s->decoded_keys[i]=UINT64_MAX;
    s->decoded_clock=0;
}
int ds41f_attention_place_workspace(ds41f_attention *s)
{
    if(!s||!s->rows)return EINVAL;
    if(s->rows_fresh_pages)return 0;
    size_t bytes=(size_t)640*512*sizeof(float);float *rows=NULL;
    int rc=ds41f_alloc_resident((void **)&rows,bytes,1);if(rc)return rc;
    #pragma omp parallel for schedule(static)
    for(size_t page=0;page<(bytes+4095)/4096;++page)((volatile char *)rows)[page*4096]=0;
    free(s->rows);s->rows=rows;s->rows_fresh_pages=1;
    if(s->decoded_rows&&!s->decoded_fresh_pages){
        size_t cache_bytes=s->decoded_capacity*512*sizeof(float);float *decoded=NULL;
        rc=ds41f_alloc_resident((void **)&decoded,cache_bytes,1);if(rc)return rc;
        #pragma omp parallel for schedule(static)
        for(size_t page=0;page<(cache_bytes+4095)/4096;++page)((volatile char *)decoded)[page*4096]=0;
        free(s->decoded_rows);s->decoded_rows=decoded;s->decoded_fresh_pages=1;
    }
    return 0;
}
int ds41f_attention_receive(ds41f_attention *s,int layer,size_t pos,const uint8_t row[356])
{
    if(!s||!row||pos>=s->capacity||layer<0||layer>=40)return EINVAL;
    if(!is_source(layer))return 0;
    int ratio=layer<20?2:1;if((pos+1)%ratio)return 0;
    int src=source(layer);size_t row_index=pos/(size_t)ratio;
    memcpy(s->compressed[src]+row_index*356,row,356);
    if(s->decoded_rows){
        size_t slot=row_index%s->decoded_capacity;
        double pt=P_BEGIN();
        int rc=ds41f_fp4_unpack(s->decoded_rows+slot*512,row,512,16,1);P_END(ATTN_PREPACK,pt);if(rc)return rc;
        s->decoded_keys[slot]=((uint64_t)(unsigned)src<<32)|(uint64_t)row_index;
    }
    return 0;
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
#if defined(__ARM_FEATURE_SVE)
static svfloat32_t index_bf16(svfloat32_t value)
{
    svbool_t pg=svptrue_b32();svuint32_t bits=svreinterpret_u32_f32(value);
    svuint32_t odd=svand_n_u32_x(pg,svlsr_n_u32_x(pg,bits,16),1);
    svuint32_t rounded=svadd_u32_x(pg,bits,svadd_n_u32_x(pg,odd,0x7fff));
    svbool_t nan=svcmpgt_n_u32(pg,svand_n_u32_x(pg,bits,0x7fffffffu),0x7f800000u);
    rounded=svsel_u32(nan,svorr_n_u32_x(pg,bits,0x00400000u),rounded);
    return svreinterpret_f32_u32(svand_n_u32_x(pg,rounded,0xffff0000u));
}
static float index_dot_heads(const float *packed,const float *key,const float *weights)
{
    #if defined(__clang__)
    #pragma STDC FP_CONTRACT OFF
    #endif
    svbool_t pg=svptrue_b32();svfloat32_t a=svdup_f32(0),b=a;
    /* SIMD lanes are heads. Each lane keeps all 128 ordered FP32 products
     * and additions, avoiding one ordered horizontal reduction per head. */
    for(size_t j=0;j<128;++j){
        svfloat32_t q0=svld1_f32(pg,packed+j*32),q1=svld1_f32(pg,packed+j*32+16);
        svfloat32_t p0=svmul_n_f32_x(pg,q0,key[j]),p1=svmul_n_f32_x(pg,q1,key[j]);
        a=svadd_f32_x(pg,a,p0);b=svadd_f32_x(pg,b,p1);
    }
    a=svmax_n_f32_x(pg,index_bf16(a),0);b=svmax_n_f32_x(pg,index_bf16(b),0);
    a=index_bf16(svmul_f32_x(pg,a,svld1_f32(pg,weights)));
    b=index_bf16(svmul_f32_x(pg,b,svld1_f32(pg,weights+16)));
    return bf(svadda_f32(pg,svadda_f32(pg,0,a),b));
}
#endif
typedef struct {
    float * scores;
    const float * q;
    const float * weights;
    const uint8_t * rows;
    const uint8_t * candidates;
    int head_tiles;
    const float * packed;
} index_scores_team_job;
static void index_scores_team_work(void *context,size_t first,size_t last)
{
    index_scores_team_job *job=context;
    float * scores=job->scores;
    const float * q=job->q;
    const float * weights=job->weights;
    const uint8_t * rows=job->rows;
    const uint8_t * candidates=job->candidates;
    int head_tiles=job->head_tiles;
    const float * packed=job->packed;
    (void)scores;
    (void)q;
    (void)weights;
    (void)rows;
    (void)candidates;
    (void)head_tiles;
    (void)packed;
    for(size_t task=first;task<last;++task){size_t i=task*(1);
        if(candidates&&!candidates[i/8]){scores[i]=-INFINITY;continue;}
        float key[128],sum=0;ds41f_fp4_unpack(key,rows+i*356+288,128,32,0);
        #if defined(__ARM_FEATURE_SVE)
        if(head_tiles&&svcntw()==16){scores[i]=index_dot_heads(packed,key,weights);continue;}
        #endif
        for(int h=0;h<32;++h){float dot=0;
            for(int j=0;j<128;++j)dot+=q[h*128+j]*key[j];
            sum+=bf(fmaxf(bf(dot),0)*weights[h]);}scores[i]=bf(sum);

    }
}
int ds41f_index_scores(float *scores,const float *q,const float *weights,const uint8_t *rows,
                       size_t count,const uint8_t *candidates,int head_tiles)
{
    if(!scores||!q||!weights||!rows||(head_tiles!=0&&head_tiles!=1))return EINVAL;
    float packed[128*32];
    #if defined(__ARM_FEATURE_SVE)
    if(head_tiles&&svcntw()==16)for(size_t j=0;j<128;++j)for(size_t h=0;h<32;++h)packed[j*32+h]=q[h*128+j];
    #else
    (void)packed;
    #endif
    if(ds41f_team_active()){
        index_scores_team_job job={scores, q, weights, rows, candidates, head_tiles, packed};
        (void)ds41f_team_for(count,index_scores_team_work,&job);
    }else
    #pragma omp parallel for schedule(static)
    for(size_t i=0;i<count;++i){
        if(candidates&&!candidates[i/8]){scores[i]=-INFINITY;continue;}
        float key[128],sum=0;ds41f_fp4_unpack(key,rows+i*356+288,128,32,0);
        #if defined(__ARM_FEATURE_SVE)
        if(head_tiles&&svcntw()==16){scores[i]=index_dot_heads(packed,key,weights);continue;}
        #endif
        for(int h=0;h<32;++h){float dot=0;
            for(int j=0;j<128;++j)dot+=q[h*128+j]*key[j];
            sum+=bf(fmaxf(bf(dot),0)*weights[h]);}scores[i]=bf(sum);
    }
    return 0;
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
    CHECK(ds41f_index_scores(scores,q,weights,rows,count,layer>20?s->candidate_blocks:NULL,s->index_head_tiles));
    P_END(INDEX_SCORE,pt);pt=P_BEGIN();
    if(layer==20){size_t blocks=(count+7)/8;float *block_scores=malloc(blocks*sizeof(float));int *ids=malloc(2048*sizeof(int));
        if(!block_scores||!ids){free(block_scores);free(ids);free(scores);return ENOMEM;}
        for(size_t b=0;b<blocks;++b){float best=-INFINITY;for(size_t j=b*8;j<count&&j<b*8+8;++j)best=fmaxf(best,scores[j]);block_scores[b]=best;}
        block_scores[blocks-1]=INFINITY;
        size_t kept=ds41f_select_topk(block_scores,blocks,2048,ids);memset(s->candidate_blocks,0,(s->capacity+7)/8);
        for(size_t i=0;i<kept;++i)s->candidate_blocks[ids[i]]=1;
        free(block_scores);free(ids);
    }
    s->selected_count=ds41f_select_topk(scores,count,s->selected_limit,s->selected);free(scores);P_END(INDEX_SELECT,pt);return 0;
}
int ds41f_attention_grouped_output(const ds41f_weights *w,int layer,float *out,const float *x,size_t groups)
{
    char name[192];snprintf(name,sizeof name,"layers.%d.attn.wo_a.weight",layer);
    const ds41f_weight *weight=ds41f_weight_find(w,name);
    snprintf(name,sizeof name,"layers.%d.attn.wo_a.scale",layer);
    const ds41f_weight *scale=ds41f_weight_find(w,name);
    if(!weight||!scale||strcmp(weight->dtype,"F8_E4M3")||weight->rows!=groups*1024||weight->cols!=4096||scale->bytes!=groups*32*128)return EINVAL;
    double pt=P_BEGIN();
    if(weight->int8.weight&&w->input_cache){CHECK(ds41f_linear_int8_cached(w,weight,out,x,1024,0));}
    else if(weight->int8.weight){P_VALUE(INT8_BYTES,weight->int8.bytes);
        CHECK(ds41f_int8_matvec(out,&weight->int8,x,1024,0));P_END(LINEAR_INT8,pt);}
    else{P_VALUE(FP8_BYTES,weight->bytes+scale->bytes);
        CHECK(ds41f_fp8_grouped_matvec(out,weight->data,scale->data,x,groups,1024,4096));P_END(LINEAR_FP8,pt);}
    pt=P_BEGIN();ds41f_round_bf16(out,groups*1024);P_END(LINEAR_ROUND,pt);return 0;
}
int ds41f_attention_prepare(ds41f_attention *s,const ds41f_weights *w,int layer,size_t pos,
                            const float *x,ds41f_attention_context *context)
{
    if(!s||!w||!x||!context||layer<0||layer>=40||pos>=s->capacity)return EINVAL;
    memset(context,0,sizeof *context);float *qr=context->qr,*kv=context->kv;
    double pt=P_BEGIN();CHECK(linear(w,layer,"wq_a",qr,x,0));CHECK(norm(w,layer,"q_norm",qr,qr));P_END(ATTN_QA,pt);
    pt=P_BEGIN();
    CHECK(linear(w,layer,"wkv",kv,x,0));CHECK(norm(w,layer,"kv_norm",kv,kv));
    rope(kv,1,512,layer,pos,0);CHECK(ds41f_act_quant(kv,kv,512));
    float *window=s->window+(size_t)layer*128*512;
    memcpy(window+(pos%128)*512,kv,512*sizeof(float));
    P_END(ATTN_KV,pt);pt=P_BEGIN();
    if(is_source(layer))CHECK(update_source(s,w,layer,pos,x));
    P_END(ATTN_COMPRESS,pt);pt=P_BEGIN();
    if(is_index(layer))CHECK(select_positions(s,w,layer,pos,x,qr));
    P_END(ATTN_INDEX,pt);pt=P_BEGIN();

    context->selected_count=s->selected_count;
    memcpy(context->selected,s->selected,s->selected_count*sizeof(int));
    memcpy(context->publication,s->publication,356);
    return 0;
}
int ds41f_attention_apply(ds41f_attention *s,int layer,size_t pos,const ds41f_attention_context *context)
{
    if(!s||!context||layer<0||layer>=40||pos>=s->capacity||context->selected_count>512)return EINVAL;
    memcpy(s->window+((size_t)layer*128+pos%128)*512,context->kv,512*sizeof(float));
    s->selected_count=context->selected_count;
    memcpy(s->selected,context->selected,s->selected_count*sizeof(int));
    memcpy(s->publication,context->publication,356);
    return ds41f_attention_receive(s,layer,pos,context->publication);
}
typedef struct {
    float * rows;
    ds41f_attention * s;
    int layer;
    size_t raw_count;
    int * ids;
} attention_project_team_job;
static void decode_attention_row(float *dst,const ds41f_attention *s,int src,size_t row)
{
    if(s->decoded_rows){
        size_t slot=row%s->decoded_capacity;
        uint64_t key=((uint64_t)(unsigned)src<<32)|(uint64_t)row;
        if(s->decoded_keys[slot]==key){memcpy(dst,s->decoded_rows+slot*512,512*sizeof *dst);return;}
    }
    (void)ds41f_fp4_unpack(dst,s->compressed[src]+row*356,512,16,1);
}
static void attention_project_team_work(void *context,size_t first,size_t last)
{
    attention_project_team_job *job=context;
    float * rows=job->rows;
    ds41f_attention * s=job->s;
    int layer=job->layer;
    size_t raw_count=job->raw_count;
    int * ids=job->ids;
    (void)rows;
    (void)s;
    (void)layer;
    (void)raw_count;
    (void)ids;
    int src=source(layer);
    for(size_t task=first;task<last;++task){size_t i=task*(1);
        decode_attention_row(rows+(raw_count+i)*512,s,src,(size_t)s->selected[i]);
        ids[raw_count+i]=(int)(raw_count+i);
    }
}
int ds41f_attention_attend(ds41f_attention *s,const ds41f_weights *w,int layer,size_t pos,
                           float *q,size_t first_head,size_t heads,float *attended)
{
    if(!s||!w||!q||!attended||!s->rows||layer<0||layer>=40||pos>=s->capacity||
       !heads||heads%8||first_head+heads>64)return EINVAL;
    double pt=P_BEGIN();rope(q,heads,512,layer,pos,0);P_END(ATTN_Q_ROPE,pt);pt=P_BEGIN();
    float *window=s->window+(size_t)layer*128*512;
    size_t raw_count=pos<128?pos+1:128,extra=layer<2?0:s->selected_count;
    if(extra>512)return EINVAL;
    float *rows=s->rows;
    int ids[128+512];
    for(size_t i=0;i<raw_count;++i){size_t token=pos+1-raw_count+i;
        memcpy(rows+i*512,window+(token%128)*512,512*sizeof(float));ids[i]=(int)i;}
    if(ds41f_team_active()){
        attention_project_team_job job={rows, s, layer, raw_count, ids};
        (void)ds41f_team_for(extra,attention_project_team_work,&job);
    }else
    #pragma omp parallel for schedule(static)
    for(size_t i=0;i<extra;++i){
        decode_attention_row(rows+(raw_count+i)*512,s,source(layer),(size_t)s->selected[i]);
        ids[raw_count+i]=(int)(raw_count+i);}
    char name[192];snprintf(name,sizeof name,"layers.%d.attn.attn_sink",layer);
    const ds41f_weight *sink=ds41f_weight_find(w,name);
    if(!sink||sink->bytes!=64*4)return EINVAL;
    P_END(ATTN_ROWS,pt);pt=P_BEGIN();
    const float *sinks=(const float *)sink->data+first_head;
    const uint8_t *packed_rows[512];
    if(s->sparse_sdot)for(size_t i=0;i<extra;++i)packed_rows[i]=s->compressed[source(layer)]+(size_t)s->selected[i]*356;
    int rc=s->sparse_sdot?ds41f_sparse_attention_sdot(attended,q,rows,sinks,packed_rows,raw_count,extra,heads,s->sparse_math,0):s->sparse_tile?ds41f_sparse_attention_tiled_math(attended,q,rows,sinks,ids,raw_count+extra,raw_count+extra,heads,512,s->sparse_tile,s->sparse_math):
        ds41f_sparse_attention(attended,q,rows,sinks,ids,raw_count+extra,raw_count+extra,heads,512);
    if(rc)return rc;P_END(ATTN_SPARSE,pt);pt=P_BEGIN();
    ds41f_round_bf16(attended,heads*512);rope(attended,heads,512,layer,pos,1);
    P_END(ATTN_INVERSE_ROPE,pt);return 0;
}
int ds41f_attention_project(ds41f_attention *s,const ds41f_weights *w,int layer,size_t pos,
                            const float *qr,size_t first_head,size_t heads,float *projected)
{
    if(!s||!w||!qr||!projected||!s->rows||layer<0||layer>=40||pos>=s->capacity||
       !heads||heads%8||first_head+heads>64)return EINVAL;
    float q[64*512],attended[64*512];double pt=P_BEGIN();
    CHECK(linear(w,layer,"wq_b",q,qr,0));P_END(ATTN_QB,pt);
    CHECK(ds41f_attention_attend(s,w,layer,pos,q,first_head,heads,attended));pt=P_BEGIN();
    CHECK(ds41f_attention_grouped_output(w,layer,projected,attended,heads/8));P_END(ATTN_WOA,pt);
    return 0;
}

int ds41f_attention_output(const ds41f_weights *w,int layer,const float *projected,float *out)
{
    double pt=P_BEGIN();int rc=linear(w,layer,"wo_b",out,projected,0);P_END(ATTN_WOB,pt);return rc;
}
int ds41f_attention_step(ds41f_attention *s,const ds41f_weights *w,int layer,size_t pos,const float *x,float *out)
{
    ds41f_attention_context context;float projected[8192];
    CHECK(ds41f_attention_prepare(s,w,layer,pos,x,&context));
    CHECK(ds41f_attention_project(s,w,layer,pos,context.qr,0,64,projected));
    return ds41f_attention_output(w,layer,projected,out);
}
