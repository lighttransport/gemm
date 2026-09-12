#define _POSIX_C_SOURCE 200809L
#include "ds41f_weights.h"
#include "ds41f_tensor.h"
#include "ds41f_sve.h"
#include "ds41f_kernels.h"
#include "ds41f_profile.h"
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Main-thread-only, bounded four-entry cache. Compare complete input bytes:
 * stack addresses are reused between layers and cannot serve as identities. */
typedef struct {float *original;ds41f_int8_input input;int fp8;size_t stamp;} input_entry;
typedef struct {input_entry entries[4];size_t clock;} input_cache;
int ds41f_weights_enable_input_cache(ds41f_weights *s)
{if(!s)return EINVAL;if(s->input_cache)return 0;s->input_cache=calloc(1,sizeof(input_cache));return s->input_cache?0:ENOMEM;}
static void free_input_cache(input_cache *cache)
{if(cache){for(int i=0;i<4;++i){free(cache->entries[i].original);ds41f_int8_input_free(&cache->entries[i].input);}free(cache);}}
int ds41f_linear_int8_cached(const ds41f_weights *s,const ds41f_weight *w,float *out,
                            const float *x,size_t group_rows,int fp8_quantize)
{
    if(!s||!w||!x||!out||!group_rows||w->rows%group_rows||!w->cols||
       w->rows/group_rows>SIZE_MAX/w->cols)return EINVAL;
    size_t n=(w->rows/group_rows)*w->cols;if(n>32768||!n)return EINVAL;
    input_cache *cache=s->input_cache;if(!cache)return EINVAL;
    input_entry *entry=NULL,*oldest=&cache->entries[0];
    for(int i=0;i<4;++i){input_entry *e=&cache->entries[i];
        if(e->input.elements==n&&e->input.block==w->int8.block&&e->fp8==fp8_quantize&&
           !memcmp(e->original,x,n*sizeof(float))){entry=e;break;}
        if(e->stamp<oldest->stamp)oldest=e;
    }
    if(entry)P_VALUE(INPUT_CACHE_HIT,1);
    else{
        P_VALUE(INPUT_CACHE_MISS,1);entry=oldest;free(entry->original);entry->original=NULL;
        ds41f_int8_input_free(&entry->input);entry->stamp=0;
        entry->original=malloc(n*sizeof(float));if(!entry->original)return ENOMEM;
        memcpy(entry->original,x,n*sizeof(float));float *quantized=NULL;int rc=0;double pt=P_BEGIN();
        if(fp8_quantize){quantized=malloc(n*sizeof(float));if(!quantized)return ENOMEM;rc=ds41f_act_quant(quantized,x,n);}
        P_END(LINEAR_QUANT,pt);pt=P_BEGIN();
        if(!rc)rc=ds41f_int8_prepare_input(&entry->input,quantized?quantized:x,n,w->int8.block);
        P_END(LINEAR_INT8,pt);free(quantized);if(rc)return rc;entry->fp8=fp8_quantize;
    }
    entry->stamp=++cache->clock;double pt=P_BEGIN();P_VALUE(INT8_BYTES,w->int8.bytes);
    int rc=ds41f_int8_matvec_prepared(out,&w->int8,&entry->input,group_rows,0);P_END(LINEAR_INT8,pt);return rc;
}
void ds41f_weights_free(ds41f_weights *s)
{
    if(!s)return;
    for(size_t i=0;i<s->count;++i){
        ds41f_tensor_free_local(s->items[i].data,s->items[i].bytes,s->fresh_pages);
        ds41f_int8_free(&s->items[i].int8);}
    free(s->items);free_input_cache(s->input_cache);memset(s,0,sizeof *s);
}
const ds41f_weight *ds41f_weight_find(const ds41f_weights *s,const char *name)
{
    size_t lo=0,hi=s->count;
    while(lo<hi){size_t m=lo+(hi-lo)/2;int c=strcmp(s->items[m].name,name);
        if(!c)return &s->items[m];
        if(c<0)lo=m+1;else hi=m;}
    return NULL;
}
int ds41f_weights_load(ds41f_weights *s,const char *stage,const char *prefix,size_t limit)
{
    return ds41f_weights_load_local(s,stage,prefix,limit,0);
}
int ds41f_weights_load_local(ds41f_weights *s,const char *stage,const char *prefix,size_t limit,int fresh_pages)
{
    if(!s||!stage||!limit)return EINVAL;
    memset(s,0,sizeof *s);s->fresh_pages=fresh_pages;char path[4096],line[1024];
    int length=snprintf(path,sizeof path,"%s/weights.index",stage);
    if(length<0||(size_t)length>=sizeof path)return ENAMETOOLONG;
    FILE *f=fopen(path,"r");if(!f)return errno;
    int rc=0;
    while(fgets(line,sizeof line,f)){
        ds41f_weight item={.rows=0};char extra;
        if(sscanf(line,"%191s %15s %zu %zu %zu %c",item.name,item.dtype,&item.rows,&item.cols,&item.bytes,&extra)!=5){rc=EINVAL;break;}
        if(prefix&&strncmp(item.name,prefix,strlen(prefix)))continue;
        item.global_rows=item.rows;
        if(!item.rows||!item.cols||!item.bytes||item.rows>SIZE_MAX/item.cols){rc=EINVAL;break;}
        size_t unit=!strcmp(item.dtype,"BF16")?2:!strcmp(item.dtype,"F32")?4:
            (!strcmp(item.dtype,"F8_E4M3")||!strcmp(item.dtype,"F8_E8M0")||!strcmp(item.dtype,"I8"))?1:0;
        if(!unit||item.rows*item.cols>SIZE_MAX/unit||item.rows*item.cols*unit!=item.bytes||
           item.bytes>limit-s->bytes){rc=EINVAL;break;}
        if(s->count&&strcmp(s->items[s->count-1].name,item.name)>=0){rc=EINVAL;break;}
        ds41f_weight *next=realloc(s->items,(s->count+1)*sizeof *next);
        if(!next){rc=ENOMEM;break;}s->items=next;s->items[s->count++]=item;s->bytes+=item.bytes;
    }
    if(ferror(f))rc=EIO;
    fclose(f);
    if(rc){ds41f_weights_free(s);return rc;}
    for(size_t i=0;i<s->count;++i){
        rc=ds41f_tensor_load_local(stage,s->items[i].name,s->items[i].bytes,&s->items[i].data,fresh_pages);
        if(rc){fprintf(stderr,"weight load failed %s rc=%d\n",s->items[i].name,rc);ds41f_weights_free(s);return rc;}
        if(i%500==0){fprintf(stderr,"LOAD tensor=%zu/%zu %s\n",i,s->count,s->items[i].name);fflush(stderr);}
    }
    return 0;
}
static int tp_sharded(const char *name)
{return !strcmp(name,"head.weight")||strstr(name,".attn.wq_b.")||strstr(name,".attn.wo_a.")||
    strstr(name,".attn.wo_b.")||strstr(name,".ffn.shared_experts.");}
int ds41f_weights_check_tp(ds41f_weights *s,const char *stage,int tp,int rank)
{
    if(!s||!stage||(tp!=1&&tp!=2&&tp!=4)||rank<0||rank>=12)return EINVAL;
    char path[4096],line[512];int n=snprintf(path,sizeof path,"%s/weights.tp",stage);
    if(n<0||(size_t)n>=sizeof path)return ENAMETOOLONG;
    FILE *f=fopen(path,"r");if(!f)return tp==1&&errno==ENOENT?0:errno;
    int version,stored_tp,stored_rank,ranks;char extra;
    int rc=0;
    if(!fgets(line,sizeof line,f)||sscanf(line,"DS41FTP %d %d %d %d %c",&version,&stored_tp,&stored_rank,&ranks,&extra)!=4||
       version!=1||tp!=stored_tp||rank!=stored_rank||ranks!=12||tp==1)rc=EINVAL;
    unsigned char *seen=calloc(s->count,1);if(!seen){fclose(f);return ENOMEM;}
    while(!rc&&fgets(line,sizeof line,f)){
        char name[192];size_t rows,cols,first,local;
        if(sscanf(line,"%191s %zu %zu %zu %zu %c",name,&rows,&cols,&first,&local,&extra)!=5){rc=EINVAL;break;}
        const ds41f_weight *found=ds41f_weight_find(s,name);
        if(!found||!tp_sharded(name)||found->rows!=local||found->cols!=cols||first>rows||local>rows-first){rc=EINVAL;break;}
        size_t index=(size_t)(found-s->items);
        if(seen[index]){rc=EINVAL;break;}seen[index]=1;
        if(!strcmp(name,"head.weight")){
            if(rows!=129280||first!=(rows/32*(size_t)rank/12)*32||
               local!=(rows/32*(size_t)(rank+1)/12)*32-first){rc=EINVAL;break;}
        }else if(rows%(size_t)tp||local!=rows/(size_t)tp||first!=(size_t)(rank%tp)*local){rc=EINVAL;break;}
        s->items[index].global_rows=rows;s->items[index].row_start=first;
    }
    if(ferror(f))rc=EIO;
    if(!rc)for(size_t i=0;i<s->count;++i)if(tp_sharded(s->items[i].name)&&!seen[i]){rc=EINVAL;break;}
    free(seen);fclose(f);return rc;
}
int ds41f_weights_requantize_fp8(ds41f_weights *s,size_t block,size_t limit,int projections_only)
{
    if(!s||block<32||block>256||block%32)return EINVAL;
    size_t converted=0,source_bytes=0,packed_bytes=0,original_bytes=s->bytes;
    for(size_t i=0;i<s->count;++i){ds41f_weight *w=&s->items[i];
        if(strcmp(w->dtype,"F8_E4M3"))continue;
        if(projections_only&&!strstr(w->name,".attn.wq_b.weight")&&!strstr(w->name,".attn.wo_a.weight")&&
           !strstr(w->name,".attn.wo_b.weight")&&!strstr(w->name,".ffn.shared_experts."))continue;
        size_t len=strlen(w->name);if(len<7||strcmp(w->name+len-7,".weight"))return EINVAL;
        char name[192];memcpy(name,w->name,len-7);strcpy(name+len-7,".scale");
        const ds41f_weight *scale=ds41f_weight_find(s,name);
        if(!scale||strcmp(scale->dtype,"F8_E8M0")||scale->rows!=(w->rows+31)/32||
           scale->cols!=(w->cols+31)/32||w->cols%block)return EINVAL;
        size_t n=((w->rows+3)/4*4)*w->cols,bytes=n+n/block*sizeof(float);
        /* A malloc pool can retain freed sources. Budget all replacements
         * until teardown unless the source mappings can be released directly. */
        size_t live=s->fresh_pages?s->bytes:original_bytes+packed_bytes;
        if(live>limit||bytes>limit-live)return ENOMEM;
        int rc=ds41f_int8_from_fp8(&w->int8,w->data,scale->data,w->rows,w->cols,block);
        if(rc)return rc;
        ds41f_tensor_free_local(w->data,w->bytes,s->fresh_pages);
        w->data=NULL;s->bytes+=w->int8.bytes-w->bytes;
        ++converted;source_bytes+=w->bytes;packed_bytes+=w->int8.bytes;
    }
    fprintf(stderr,"FP8_INT8 tensors=%zu block=%zu source_bytes=%zu packed_bytes=%zu\n",
        converted,block,source_bytes,packed_bytes);
    return 0;
}
int ds41f_linear(const ds41f_weights *s,const char *base,float *out,const float *x,int raw)
{
    char name[192];snprintf(name,sizeof name,"%s.weight",base);
    const ds41f_weight *w=ds41f_weight_find(s,name);if(!w||(!w->data&&!w->int8.weight))return ENOENT;
    double pt=P_BEGIN();
    if(!strcmp(w->dtype,"F8_E4M3")){
        snprintf(name,sizeof name,"%s.scale",base);const ds41f_weight *scale=ds41f_weight_find(s,name);
        if(!scale||strcmp(scale->dtype,"F8_E8M0")||scale->rows!=(w->rows+31)/32||scale->cols!=(w->cols+31)/32)return EINVAL;
        if(w->int8.weight&&s->input_cache){
            int rc=ds41f_linear_int8_cached(s,w,out,x,w->rows,!raw);if(rc)return rc;
        }else{
        float *input=NULL;int rc=0;
        if(!raw){input=malloc(w->cols*sizeof *input);if(!input)return ENOMEM;
            rc=ds41f_act_quant(input,x,w->cols);}
        P_END(LINEAR_QUANT,pt);pt=P_BEGIN();
        if(w->int8.weight){P_VALUE(INT8_BYTES,w->int8.bytes);
            if(!rc)rc=ds41f_int8_matvec(out,&w->int8,input?input:x,w->rows,0);
            P_END(LINEAR_INT8,pt);
        }else{P_VALUE(FP8_BYTES,w->bytes+scale->bytes);
            if(!rc)rc=ds41f_fp8_matvec(out,w->data,scale->data,input?input:x,w->rows,w->cols);
            P_END(LINEAR_FP8,pt);}
        free(input);if(rc)return rc;
        }
    }else if(!strcmp(w->dtype,"BF16")){
        P_VALUE(BF16_BYTES,w->bytes);
        ds41f_bf16_f32_matvec(out,w->data,x,w->rows,w->cols);
        P_END(LINEAR_BF16,pt);
    }else if(!strcmp(w->dtype,"F32")){
        P_VALUE(F32_BYTES,w->bytes);
        #pragma omp parallel for schedule(static)
        for(size_t r=0;r<w->rows;++r){float sum=0;
            #pragma omp simd reduction(+:sum)
            for(size_t c=0;c<w->cols;++c){size_t i=r*w->cols+c;
                float value=((float *)w->data)[i];
                sum+=value*x[c];}out[r]=sum;}P_END(LINEAR_F32,pt);
    }else return EINVAL;
    pt=P_BEGIN();if(!raw)ds41f_round_bf16(out,w->rows);P_END(LINEAR_ROUND,pt);
    return 0;
}
int ds41f_norm(const ds41f_weights *s,const char *name,float *out,const float *x)
{
    double pt=P_BEGIN();
    const ds41f_weight *w=ds41f_weight_find(s,name);
    if(!w||strcmp(w->dtype,"BF16")||w->rows!=1)return EINVAL;
    ds41f_rmsnorm_fast(out,x,w->data,w->cols,1e-20f);ds41f_round_bf16(out,w->cols);P_END(NORM,pt);return 0;
}
