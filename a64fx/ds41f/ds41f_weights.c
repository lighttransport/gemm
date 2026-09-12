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

void ds41f_weights_free(ds41f_weights *s)
{
    if(!s)return;
    for(size_t i=0;i<s->count;++i){
        ds41f_tensor_free_local(s->items[i].data,s->items[i].bytes,s->fresh_pages);
        ds41f_int8_free(&s->items[i].int8);}
    free(s->items);memset(s,0,sizeof *s);
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
