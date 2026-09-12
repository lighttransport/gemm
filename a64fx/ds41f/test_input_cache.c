#include "ds41f_weights.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(void)
{
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
    x[0]=NAN;
    if(ds41f_linear(&store,"unit",got,x,0)!=EDOM||ds41f_linear(&store,"unit",got,x,0)!=EDOM)return 1;
    ds41f_weights_free(&store);free(raw);free(x);
    puts("INPUT_CACHE PASS exact raw/FP8 boundaries grouped reused_addresses changing_blocks repeat_hits nonfinite");return 0;
}
