#ifndef GLM53F_PP_IMAGE_SOURCE_H
#define GLM53F_PP_IMAGE_SOURCE_H
#include "glm53f_pp_manifest.h"
#include "glm53f_pp_images.h"
#include <stdint.h>
#include <stdlib.h>
#include <ctype.h>
/* Reject mixed sources and foreign layouts before resident allocations. The
 * source stamp covers GGUF shard metadata; individual loaders verify bytes. */
static inline int glm53f_pp_image_stamp(const char *dir, const char *suffix,
        const char *component, const glm53f_dist *d, int first, int end,
        uint64_t *stamp) {
    if(!dir||!stamp)return-1;
    char path[4096],line[2048];
    int n=snprintf(path,sizeof(path),"%s/rank%02d.%s",dir,d->map.world_rank,suffix);
    if(n<0||n>=(int)sizeof(path)||glm53f_pp_manifest_check(path,component,d,first,end))return-1;
    FILE *f=fopen(path,"r");if(!f)return-1;
    int valid=fgets(line,sizeof(line),f)!=NULL;valid&=fclose(f)==0;
    if(!valid)return-1;
    const char *key="source_metadata_fnv1a=",*value=strstr(line,key);
    if(!value)return-1;
    value+=strlen(key);
    if(strlen(value)<16)return-1;
    for(int i=0;i<16;i++)if(!isxdigit((unsigned char)value[i]))return-1;
    if(value[16]&&value[16]!=' '&&value[16]!='\n'&&value[16]!='\r')return-1;
    char digest[17];memcpy(digest,value,16);digest[16]=0;
    *stamp=(uint64_t)strtoull(digest,NULL,16);return 0;
}
static inline int glm53f_pp_images_source(const glm53f_dist *d,
        const glm53f_pp_images *images,uint64_t *stamp) {
    if(!d||!images||!stamp)return-1;
    const int first=d->map.first_layer,end=d->map.end_layer;
    if(glm53f_pp_image_stamp(images->core,"core.manifest","CORE",d,first,end,stamp))return-1;
    uint64_t other;
#define CHECK_IMAGE(D,C,F,E) do { if(glm53f_pp_image_stamp(D,"manifest",C,d,F,E,&other)||other!=*stamp)return-1; } while(0)
    if(end>3){CHECK_IMAGE(images->routed,"ROUTED",first>3?first:3,end);CHECK_IMAGE(images->shared,"SHARED",first,end);}
    if(first<3)CHECK_IMAGE(images->dense,"DENSE",first,end);
    int nk=0,ns=0;for(int l=first;l<end;l++){if(l%4==3)ns++;else nk++;}
    if(nk)CHECK_IMAGE(images->kda,"KDA",first,end);
    if(ns)CHECK_IMAGE(images->sparse,"SPARSE",first,end);
    if(d->map.stage==0)CHECK_IMAGE(images->embed,"EMBED",first,end);
    if(d->map.stage==2)CHECK_IMAGE(images->head,"HEAD",first,end);
#undef CHECK_IMAGE
    return 0;
}
#endif
