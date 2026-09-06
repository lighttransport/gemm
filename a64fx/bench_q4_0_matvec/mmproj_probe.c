/* Inspect mmproj-F32 tensors + vision KV config. */
#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#define GGUF_LOADER_IMPLEMENTATION
#include "gguf_loader.h"

static const char *tn(uint32_t t){
    switch(t){case 0:return"F32";case 1:return"F16";case 30:return"BF16";
    case 2:return"Q4_0";case 8:return"Q8_0";case 14:return"Q6_K";default:return"?";}
}
int main(int argc,char**argv){
    const char*p = argc>1?argv[1]:"/local/u14346/mmproj-F32.gguf";
    gguf_context*c = gguf_open(p,1);
    if(!c){fprintf(stderr,"open fail %s\n",p);return 1;}
    printf("n_tensors=%llu n_kv=%llu\n",(unsigned long long)c->n_tensors,(unsigned long long)c->n_kv);
    /* dump KV (vision config: image/patch size, n_embd, n_head, etc) */
    for(uint64_t i=0;i<c->n_kv;i++){
        gguf_kv*kv=&c->kv[i];
        if(!kv->key.str) continue;
        if(strstr(kv->key.str,"clip")||strstr(kv->key.str,"vision")||strstr(kv->key.str,"image")||
           strstr(kv->key.str,"patch")||strstr(kv->key.str,"proj")||strstr(kv->key.str,"block")||
           strstr(kv->key.str,"head")||strstr(kv->key.str,"embed")||strstr(kv->key.str,"feed")){
            printf("KV %-45s type=%u",kv->key.str,kv->type);
            switch(kv->type){
              case 4: printf(" = %u",kv->value.u32); break;   /* uint32 */
              case 5: printf(" = %d",kv->value.i32); break;   /* int32  */
              case 10:printf(" = %llu",(unsigned long long)kv->value.u64); break;
              case 11:printf(" = %lld",(long long)kv->value.i64); break;
              case 6: printf(" = %f",kv->value.f32); break;
              case 7: printf(" = %u",kv->value.b); break;
              case 8: if(kv->value.str.str) printf(" = %s",kv->value.str.str); break;
              default: break;
            }
            printf("\n");
        }
    }
    printf("--- tensors ---\n");
    for(uint64_t i=0;i<c->n_tensors;i++){
        gguf_tensor_info*t=&c->tensors[i];
        /* show conv/patch/first-few + any 4D */
        if(t->n_dims>=3 || strstr(t->name.str,"patch")||strstr(t->name.str,"conv")||
           strstr(t->name.str,"pos")|| i<6){
            printf("[%3llu] %-40s %s ndim=%u dims=[%llu,%llu,%llu,%llu]\n",
                (unsigned long long)i,t->name.str,tn(t->type),t->n_dims,
                (unsigned long long)t->dims[0],(unsigned long long)t->dims[1],
                (unsigned long long)t->dims[2],(unsigned long long)t->dims[3]);
        }
    }
    return 0;
}
