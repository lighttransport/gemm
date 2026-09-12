#define _POSIX_C_SOURCE 200809L
#include "ds41f_weights.h"
#include "ds41f_tensor.h"
#include "ds41f_sve.h"
#include "ds41f_kernels.h"
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void ds41f_round_bf16(float *x,size_t n)
{for(size_t i=0;i<n;++i)x[i]=ds41f_bf16_to_f32(ds41f_f32_to_bf16(x[i]));}
void ds41f_weights_free(ds41f_weights *s)
{
    if(!s)return;
    for(size_t i=0;i<s->count;++i)free(s->items[i].data);
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
    if(!s||!stage||!limit)return EINVAL;
    memset(s,0,sizeof *s);char path[4096],line[1024];
    int length=snprintf(path,sizeof path,"%s/weights.index",stage);
    if(length<0||(size_t)length>=sizeof path)return ENAMETOOLONG;
    FILE *f=fopen(path,"r");if(!f)return errno;
    int rc=0;
    while(fgets(line,sizeof line,f)){
        ds41f_weight item={0};char extra;
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
        rc=ds41f_tensor_load(stage,s->items[i].name,s->items[i].bytes,&s->items[i].data);
        if(rc){fprintf(stderr,"weight load failed %s rc=%d\n",s->items[i].name,rc);ds41f_weights_free(s);return rc;}
        if(i%500==0){fprintf(stderr,"LOAD tensor=%zu/%zu %s\n",i,s->count,s->items[i].name);fflush(stderr);}
    }
    return 0;
}
int ds41f_linear(const ds41f_weights *s,const char *base,float *out,const float *x,int raw)
{
    char name[192];snprintf(name,sizeof name,"%s.weight",base);
    const ds41f_weight *w=ds41f_weight_find(s,name);if(!w||!w->data)return ENOENT;
    if(!strcmp(w->dtype,"F8_E4M3")){
        snprintf(name,sizeof name,"%s.scale",base);const ds41f_weight *scale=ds41f_weight_find(s,name);
        if(!scale||strcmp(scale->dtype,"F8_E8M0")||scale->rows!=(w->rows+31)/32||scale->cols!=(w->cols+31)/32)return EINVAL;
        float *input=NULL;int rc=0;
        if(!raw){input=malloc(w->cols*sizeof *input);if(!input)return ENOMEM;
            rc=ds41f_act_quant(input,x,w->cols);}
        if(!rc)rc=ds41f_fp8_matvec(out,w->data,scale->data,input?input:x,w->rows,w->cols);
        free(input);if(rc)return rc;
    }else if(!strcmp(w->dtype,"BF16")){
        ds41f_bf16_f32_matvec(out,w->data,x,w->rows,w->cols);
    }else if(!strcmp(w->dtype,"F32")){
        #pragma omp parallel for schedule(static)
        for(size_t r=0;r<w->rows;++r){float sum=0;
            #pragma omp simd reduction(+:sum)
            for(size_t c=0;c<w->cols;++c){size_t i=r*w->cols+c;
                float value=((float *)w->data)[i];
                sum+=value*x[c];}out[r]=sum;}
    }else return EINVAL;
    if(!raw)ds41f_round_bf16(out,w->rows);
    return 0;
}
int ds41f_norm(const ds41f_weights *s,const char *name,float *out,const float *x)
{
    const ds41f_weight *w=ds41f_weight_find(s,name);
    if(!w||strcmp(w->dtype,"BF16")||w->rows!=1)return EINVAL;
    ds41f_rmsnorm_fast(out,x,w->data,w->cols,1e-20f);ds41f_round_bf16(out,w->cols);return 0;
}
