#ifndef GLM53F_PP_CORE_H
#define GLM53F_PP_CORE_H
#include "../../common/glm53f_safetensors.h"
#include "glm53f_pp_manifest.h"
#include <stdint.h>
#include <stdlib.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
typedef struct {
    char name[512], kind;
    uint64_t a,b,c,offset,bytes,hash;
} glm53f_pp_core_entry;
typedef struct {
    int fd;
    size_t count;
    glm53f_pp_core_entry *entries;
} glm53f_pp_core_reader;
static inline void glm53f_pp_core_close(void *context) {
    glm53f_pp_core_reader *r=context;
    if(!r)return;
    if(r->fd>=0)close(r->fd);free(r->entries);free(r);
}
static inline int glm53f_pp_core_read(void *context,const char *kind,const char *name,
        size_t a,size_t b,size_t c,void *dst,size_t bytes) {
    glm53f_pp_core_reader *r=context;
    if(!r||!kind||!name||!dst)return-1;
    for(size_t i=0;i<r->count;i++){
        const glm53f_pp_core_entry *e=&r->entries[i];
        if(e->kind!=kind[0]||e->a!=a||e->b!=b||e->c!=c||e->bytes!=bytes||strcmp(e->name,name))continue;
        uint64_t hash=UINT64_C(1469598103934665603);size_t done=0;
        while(done<bytes){size_t count=bytes-done;if(count>(1u<<20))count=1u<<20;
            unsigned char *p=(unsigned char *)dst+done;
            ssize_t n=pread(r->fd,p,count,(off_t)(e->offset+done));
            if(n<0&&errno==EINTR)continue;if(n<=0)return-1;
            for(ssize_t j=0;j<n;j++){hash^=p[j];hash*=UINT64_C(1099511628211);}
            (void)posix_fadvise(r->fd,(off_t)(e->offset+done),n,POSIX_FADV_DONTNEED);done+=(size_t)n;
        }
        return hash==e->hash?0:-1;
    }
    return-1;
}
static inline glm53f_st_context *glm53f_pp_core_open(const glm53f_dist *d,const char *model) {
    if(!d||!d->initialized||!d->core_stage||d->config.layout!=GLM53F_PP3_TP4)return NULL;
    char manifest[4096],blob[4096],line[2048];
    int n=snprintf(manifest,sizeof(manifest),"%s/rank%02d.core.manifest",d->core_stage,d->map.world_rank);
    int b=snprintf(blob,sizeof(blob),"%s/rank%02d.core.blob",d->core_stage,d->map.world_rank);
    if(n<0||n>=(int)sizeof(manifest)||b<0||b>=(int)sizeof(blob)||glm53f_pp_manifest_check(manifest,"CORE",d,d->map.first_layer,d->map.end_layer))return NULL;
    glm53f_pp_core_reader *r=calloc(1,sizeof(*r));if(!r)return NULL;r->fd=-1;
    FILE *f=fopen(manifest,"r");struct stat st;
    if(!f||(r->fd=open(blob,O_RDONLY))<0||fstat(r->fd,&st)||st.st_size<1){if(f)fclose(f);glm53f_pp_core_close(r);return NULL;}
    int failed=0,pending=0,complete=0;uint64_t previous=0;glm53f_pp_core_entry e;
    while(fgets(line,sizeof(line),f)&&!failed){
        unsigned long long a,bb,c,off,bytes,hash;
        if(line[0]!='#'){
            if(pending){failed=1;break;}memset(&e,0,sizeof(e));
            int got=sscanf(line,"%c %511s %llu %llu %llu %llu",&e.kind,e.name,&a,&bb,&c,&off);
            if(e.kind=='R'&&got==5){e.a=a;e.b=bb;e.offset=c;}
            else if(e.kind=='C'&&got==6){e.a=a;e.b=bb;e.c=c;e.offset=off;}
            else{failed=1;break;}
            int layer;
            if(sscanf(e.name,"model.language_model.layers.%d.",&layer)==1){if(layer<d->map.first_layer||layer>=d->map.end_layer){failed=1;break;}}
            else if(d->map.stage!=2||strcmp(e.name,"model.language_model.norm.weight")){failed=1;break;}
            pending=1;
        }else if(sscanf(line,"# PAYLOAD offset=%llu bytes=%llu fnv1a=%llx",&off,&bytes,&hash)==3){
            if(!pending||off!=e.offset||off<previous||off>(uint64_t)st.st_size||!bytes||bytes>(uint64_t)st.st_size-off||(e.kind=='R'&&bytes!=e.b)){failed=1;break;}
            e.bytes=bytes;e.hash=hash;previous=off+bytes;
            for(size_t i=0;i<r->count;i++)if(r->entries[i].kind==e.kind&&r->entries[i].a==e.a&&r->entries[i].b==e.b&&r->entries[i].c==e.c&&!strcmp(r->entries[i].name,e.name)){failed=1;break;}
            if(failed)break;
            glm53f_pp_core_entry *p=realloc(r->entries,(r->count+1)*sizeof(*p));if(!p){failed=1;break;}
            r->entries=p;r->entries[r->count++]=e;pending=0;
        }else if(sscanf(line,"# COMPLETE bytes=%llu",&bytes)==1){if(complete||pending||bytes!=(uint64_t)st.st_size||previous!=bytes)failed=1;complete=1;}
    }
    failed|=ferror(f)||pending||!complete||!r->count;failed|=fclose(f)!=0;
    if(failed){glm53f_pp_core_close(r);return NULL;}
    glm53f_st_context *st_context=glm53f_st_open(model);
    if(!st_context){glm53f_pp_core_close(r);return NULL;}
    st_context->slice_reader=glm53f_pp_core_read;st_context->slice_close=glm53f_pp_core_close;st_context->slice_context=r;
    return st_context;
}
#endif
