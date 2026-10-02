#ifndef GLM53F_PP_MEMORY_H
#define GLM53F_PP_MEMORY_H
#include "glm53f_memory_budget.h"
#include "glm53f_dist.h"
#include "glm53f_pp_images.h"
#include <stdio.h>
#include <stdint.h>
#include <sys/stat.h>
#include <inttypes.h>
/* Complete conservative inventory for the PP prototype's bounded shapes.
 * File sizes bound resident payloads, including alignment padding. Derived
 * panels/workspaces are upper bounds; unused copies are deliberately included.
 * Do not extend capacities, thread counts or kernel storage without updating
 * the corresponding bound and qualification. */
static inline uint64_t glm53f_pp_available(void) {
    FILE *f=fopen("/proc/meminfo","r");if(!f)return 0;
    char line[256];unsigned long long kb;uint64_t bytes=0;
    while(fgets(line,sizeof(line),f))if(sscanf(line,"MemAvailable: %llu kB",&kb)==1){bytes=kb*1024;break;}
    fclose(f);return bytes;
}
static inline int glm53f_pp_file_bytes(const char *dir,int rank,const char *suffix,uint64_t *bytes) {
    if(!dir||!bytes)return-1;
    char path[4096];struct stat st;
    int n=snprintf(path,sizeof(path),"%s/rank%02d.%s",dir,rank,suffix);
    if(n<0||n>=(int)sizeof(path)||stat(path,&st)||st.st_size<0)return-1;
    *bytes=(uint64_t)st.st_size;return 0;
}
static inline int glm53f_pp_inventory(const glm53f_dist *d,const glm53f_pp_images *images,
        int capacity,int threads,int max_tokens,uint64_t model_workspace,
        glm53f_memory_budget *budget) {
    const uint64_t mib=UINT64_C(1024)*1024;
    if(!d||!d->initialized||d->config.layout!=GLM53F_PP3_TP4||!images||!budget||capacity<1||capacity>32768||threads<1||threads>48||max_tokens<2048||max_tokens>4096)return-1;
    *budget=(glm53f_memory_budget){0};
    int nk=0,ns=0,nd=0,nr=0;
    for(int l=d->map.first_layer;l<d->map.end_layer;l++){if(l%4==3)ns++;else nk++;if(l<3)nd++;else nr++;}
    uint64_t core=0,routed=0,shared=0,dense=0,kda=0,sparse=0,vocab=0;
    int rank=d->map.world_rank;
    if(glm53f_pp_file_bytes(images->core,rank,"core.blob",&core)||
       (nr&&glm53f_pp_file_bytes(images->routed,rank,"blob",&routed))||
       (nr&&glm53f_pp_file_bytes(images->shared,rank,"blob",&shared))||
       (nd&&glm53f_pp_file_bytes(images->dense,rank,"blob",&dense))||
       (nk&&glm53f_pp_file_bytes(images->kda,rank,"blob",&kda))||
       (ns&&glm53f_pp_file_bytes(images->sparse,rank,"blob",&sparse))||
       (d->map.stage==0&&glm53f_pp_file_bytes(images->embed,rank,"f32",&vocab))||
       (d->map.stage==2&&glm53f_pp_file_bytes(images->head,rank,"f32",&vocab)))return-1;
    uint64_t native=0,parts[]={dense,shared,kda,sparse};
    for(size_t i=0;i<4;i++){if(parts[i]>UINT64_MAX-native)return-1;native+=parts[i];}
    if(native>UINT64_MAX/3||core>UINT64_MAX-routed)return-1;
    /* Native Q8_0 repacking grows34 bytes to36. Bound all formats by9/8;
     * up to two additional panel/converted copies are covered separately. */
    if(glm53f_memory_budget_add(budget,core+routed,0)||
       glm53f_memory_budget_add(budget,native+native/8,vocab+64*mib)||
       glm53f_memory_budget_add(budget,vocab,0)||
       glm53f_memory_budget_add(budget,2*native,0))return-1;
    uint64_t cache=(uint64_t)ns*((uint64_t)capacity*(512+2*128)*4+(uint64_t)(capacity/4+1)*128*4+(uint64_t)capacity*512*2);
    if(glm53f_memory_budget_add(budget,cache,0)||
       glm53f_memory_budget_add(budget,(uint64_t)nk*20*mib+(uint64_t)ns*16*mib+(uint64_t)nd*2*mib,0)||
       glm53f_memory_budget_add(budget,model_workspace,0)||
       glm53f_memory_budget_add(budget,16*mib+(uint64_t)max_tokens*4096*4,0))return-1;
    if(nr){
        uint64_t moe=(uint64_t)max_tokens*8*4096*4; /* routed output slots */
        moe+=(uint64_t)max_tokens*(3*4096+288)*4; /* local/router/shared storage */
        moe+=(uint64_t)threads*4*mib+160*mib; /* grouped workers, shared GEMM, quantization */
        moe+=(uint64_t)nr*288*4096*2; /* packed router duplicate */
        if(glm53f_memory_budget_add(budget,moe,0))return-1;
    }
    /* Metadata parsers, bounded I/O, conversion temporaries and allocator
     * overhead. The vocabulary duplicate above covers head first-touch. */
    return glm53f_memory_budget_add(budget,64*mib,512*mib);
}
#endif
