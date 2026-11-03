#ifndef GLM53F_IQ_METADATA_H
#define GLM53F_IQ_METADATA_H
/* Internal to the IQ bridge. A model-owned, bounded, lazy metadata cache.
 * Caller serialization matches the owning MoE context; lookup is outside its
 * OpenMP region. Original packed weights are never copied or expanded. */
typedef struct { float d, dm; uint8_t scale[8], minimum[8]; } iq_metadata;
_Static_assert(sizeof(iq_metadata)==24,"metadata budget assumes 24-byte blocks");
typedef struct {
    const void *base;
    iq_metadata *metadata;
    int type, rows, blocks;
} iq_cache_entry;
enum { IQ_CACHE_SLOTS=16384 };
struct glm53f_iq_cache { size_t limit, used; iq_cache_entry *entry; };

glm53f_iq_cache *glm53f_iq_cache_create(size_t limit){
    size_t table_bytes=IQ_CACHE_SLOTS*sizeof(iq_cache_entry);
    if(limit<table_bytes+sizeof(glm53f_iq_cache)||limit>2UL*1024*1024*1024)return NULL;
    glm53f_iq_cache *c=calloc(1,sizeof(*c));
    if(!c)return NULL;
    c->entry=calloc(IQ_CACHE_SLOTS,sizeof(*c->entry));
    if(!c->entry){free(c);return NULL;}
    c->limit=limit;c->used=table_bytes+sizeof(*c);return c;
}
void glm53f_iq_cache_free(glm53f_iq_cache *c){
    if(!c)return;
    for(int i=0;i<IQ_CACHE_SLOTS;i++)free(c->entry[i].metadata);
    free(c->entry);free(c);
}
size_t glm53f_iq_cache_bytes(const glm53f_iq_cache *c){return c?c->used:0;}
static const iq_metadata *iq_cache_get(glm53f_iq_cache *c,const uint8_t *base,
                                      int type,int rows,int blocks){
    if(!c||(type!=GLM53F_GGML_Q4_K&&type!=GLM53F_GGML_Q5_K)||rows<1||blocks<1)return NULL;
    size_t hash=(((uintptr_t)base>>8)*UINT64_C(11400714819323198485))>>(64-14);
    iq_cache_entry *e=NULL;
    for(int probe=0;probe<IQ_CACHE_SLOTS;probe++){
        e=c->entry+((hash+(size_t)probe)&(IQ_CACHE_SLOTS-1));
        if(!e->base)break;
        if(e->base==base&&e->type==type&&e->rows==rows&&e->blocks==blocks)return e->metadata;
        e=NULL;
    }
    if(!e)return NULL;
    e->base=base;e->type=type;e->rows=rows;e->blocks=blocks;
    size_t n=(size_t)rows*blocks;
    if(n>SIZE_MAX/sizeof(iq_metadata)||n*sizeof(iq_metadata)>c->limit-c->used)return NULL;
    e->metadata=malloc(n*sizeof(*e->metadata));
    if(!e->metadata)return NULL;
    c->used+=n*sizeof(*e->metadata);
#pragma omp parallel for schedule(static)
    for(size_t b=0;b<n;b++){
        const uint8_t *scales;
        if(type==GLM53F_GGML_Q4_K){
            const block_q4_K *w=(const block_q4_K*)base+b;
            e->metadata[b].d=ggml_fp16_to_fp32(w->d);
            e->metadata[b].dm=ggml_fp16_to_fp32(w->dmin);scales=w->scales;
        }else{
            const block_q5_K *w=(const block_q5_K*)base+b;
            e->metadata[b].d=ggml_fp16_to_fp32(w->d);
            e->metadata[b].dm=ggml_fp16_to_fp32(w->dmin);scales=w->scales;
        }
        for(int g=0;g<8;g++)get_scale_min_k4(g,scales,&e->metadata[b].scale[g],&e->metadata[b].minimum[g]);
    }
    return e->metadata;
}
#endif
