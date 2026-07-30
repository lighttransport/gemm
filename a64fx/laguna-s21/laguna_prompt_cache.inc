/* Versioned, model-bound system-prompt KV snapshots.
 *
 * The file contains the exact prefix ids followed by the live K and V slots for
 * every layer.  Full-attention layers store prefix_tokens slots; sliding layers
 * store min(prefix_tokens, ring_capacity) slots.  An immutable in-memory copy is
 * restored into each HTTP sequence slot before its request suffix is prefilled.
 */
#define LPC_MAGIC "LGKVPC1\0"
#define LPC_VERSION 1u
#define LPC_HEADER_BYTES 128u
#if defined(LAGUNA_BF16)
#define LPC_BUILD_KIND 1u
#elif defined(LAGUNA_FP8)
#define LPC_BUILD_KIND 2u
#else
#define LPC_BUILD_KIND 3u
#endif

typedef struct {
    int n_tokens, n_layers, ep_rank, ep_size;
    int *ids;
    uint16_t *k, *v;
    size_t elems, layer_off[LAGUNA_LAYERS];
    int layer_slots[LAGUNA_LAYERS];
    uint64_t model_sig;
} laguna_prompt_cache;

static uint64_t lpc_hash(uint64_t h, const void *data, size_t n) {
    const unsigned char *p = (const unsigned char *)data;
    for (size_t i=0;i<n;i++) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}
static uint64_t lpc_model_sig(const laguna_model *m) {
    uint64_t h=1469598103934665603ull;
    uint32_t shape[8]={LAGUNA_HIDDEN,LAGUNA_VOCAB,LAGUNA_KV_HEADS,
                       LAGUNA_HEAD_DIM,LAGUNA_SLIDING_CAP,(uint32_t)m->n_layers,
#if defined(LAGUNA_KV_FP16)
                       1u,
#else
                       0u,
#endif
                       LPC_BUILD_KIND
    };
    h=lpc_hash(h,shape,sizeof shape);
    /* Replicated tensors provide a cheap checkpoint identity on every EP rank. */
    h=lpc_hash(h,m->final_norm,(size_t)LAGUNA_HIDDEN*sizeof(uint16_t));
    h=lpc_hash(h,m->embed,(size_t)LAGUNA_HIDDEN*sizeof(uint16_t));
    for(int L=0;L<m->n_layers;L++){
        h=lpc_hash(h,m->layers[L].in_ln,(size_t)LAGUNA_HIDDEN*sizeof(uint16_t));
        h=lpc_hash(h,m->layers[L].post_ln,(size_t)LAGUNA_HIDDEN*sizeof(uint16_t));
    }
    return h;
}
static void lpc_put32(unsigned char *p, uint32_t v) {
    p[0]=(unsigned char)v; p[1]=(unsigned char)(v>>8);
    p[2]=(unsigned char)(v>>16); p[3]=(unsigned char)(v>>24);
}
static void lpc_put64(unsigned char *p, uint64_t v) {
    for(int i=0;i<8;i++) p[i]=(unsigned char)(v>>(8*i));
}
static uint32_t lpc_get32(const unsigned char *p) {
    return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;
}
static uint64_t lpc_get64(const unsigned char *p) {
    uint64_t v=0; for(int i=0;i<8;i++)v|=(uint64_t)p[i]<<(8*i); return v;
}
static void lpc_free(laguna_prompt_cache *pc) {
    free(pc->ids); free(pc->k); free(pc->v); memset(pc,0,sizeof *pc);
}
static int lpc_layout(laguna_prompt_cache *pc, const laguna_model *m, int n_tokens,
                      char *err, size_t errn) {
    if(n_tokens<1 || n_tokens>m->max_pos){
        snprintf(err,errn,"cache prefix length %d is outside model maxpos %d",n_tokens,m->max_pos);
        return -1;
    }
    pc->n_tokens=n_tokens; pc->n_layers=m->n_layers; pc->elems=0;
    pc->ep_rank=m->ep_rank; pc->ep_size=m->ep_size>0?m->ep_size:1;
    for(int L=0;L<m->n_layers;L++){
        int slots=n_tokens<m->kv_cap[L]?n_tokens:m->kv_cap[L];
        pc->layer_off[L]=pc->elems; pc->layer_slots[L]=slots;
        pc->elems+=(size_t)slots*LAGUNA_KV_HEADS*LAGUNA_HEAD_DIM;
    }
    return 0;
}
static int lpc_capture(laguna_prompt_cache *pc, const laguna_model *m, int seq,
                       const int *ids, int n_tokens, char *err, size_t errn) {
    memset(pc,0,sizeof *pc);
    if(seq<0 || seq>=m->n_seq){snprintf(err,errn,"invalid cache sequence %d",seq);return -1;}
    if(lpc_layout(pc,m,n_tokens,err,errn))return -1;
    pc->ids=malloc((size_t)n_tokens*sizeof(int));
    pc->k=malloc(pc->elems*sizeof(uint16_t)); pc->v=malloc(pc->elems*sizeof(uint16_t));
    if(!pc->ids||!pc->k||!pc->v){snprintf(err,errn,"out of memory capturing prompt cache");lpc_free(pc);return -1;}
    memcpy(pc->ids,ids,(size_t)n_tokens*sizeof(int));
    size_t sb=(size_t)seq*m->kv_seq_stride;
    for(int L=0;L<m->n_layers;L++){
        size_t ne=(size_t)pc->layer_slots[L]*LAGUNA_KV_HEADS*LAGUNA_HEAD_DIM;
        memcpy(pc->k+pc->layer_off[L],m->kcache+sb+m->kv_off[L],ne*sizeof(uint16_t));
        memcpy(pc->v+pc->layer_off[L],m->vcache+sb+m->kv_off[L],ne*sizeof(uint16_t));
    }
    pc->model_sig=lpc_model_sig(m); return 0;
}
static int lpc_restore(const laguna_prompt_cache *pc, laguna_model *m, int seq,
                       char *err, size_t errn) {
    if(seq<0||seq>=m->n_seq){snprintf(err,errn,"invalid cache destination sequence %d",seq);return -1;}
    if(pc->n_layers!=m->n_layers||pc->model_sig!=lpc_model_sig(m)){
        snprintf(err,errn,"prompt cache model/KV format does not match this runner");return -1;
    }
    if(pc->ep_rank!=m->ep_rank||pc->ep_size!=(m->ep_size>0?m->ep_size:1)){
        snprintf(err,errn,"prompt cache belongs to a different EP rank/layout");return -1;
    }
    if(pc->n_tokens>m->max_pos){snprintf(err,errn,"cached prefix exceeds model maxpos");return -1;}
    size_t sb=(size_t)seq*m->kv_seq_stride;
    for(int L=0;L<m->n_layers;L++){
        int want=pc->n_tokens<m->kv_cap[L]?pc->n_tokens:m->kv_cap[L];
        if(pc->layer_slots[L]!=want){snprintf(err,errn,"cache layer %d layout mismatch",L);return -1;}
        size_t ne=(size_t)want*LAGUNA_KV_HEADS*LAGUNA_HEAD_DIM;
        memcpy(m->kcache+sb+m->kv_off[L],pc->k+pc->layer_off[L],ne*sizeof(uint16_t));
        memcpy(m->vcache+sb+m->kv_off[L],pc->v+pc->layer_off[L],ne*sizeof(uint16_t));
    }
    return 0;
}
static int lpc_matches(const laguna_prompt_cache *pc, const int *ids, int n) {
    return n>pc->n_tokens && !memcmp(ids,pc->ids,(size_t)pc->n_tokens*sizeof(int));
}
static int lpc_write_all(FILE *f, const void *p, size_t n) {
    return n==0||fwrite(p,1,n,f)==n?0:-1;
}
static int lpc_save(const char *path, const laguna_prompt_cache *pc,
                    char *err, size_t errn) {
    size_t ids_bytes=(size_t)pc->n_tokens*sizeof(int);
    size_t kv_bytes=pc->elems*sizeof(uint16_t);
    uint64_t ids_hash=lpc_hash(1469598103934665603ull,pc->ids,ids_bytes);
    uint64_t payload_hash=1469598103934665603ull;
    payload_hash=lpc_hash(payload_hash,pc->k,kv_bytes);
    payload_hash=lpc_hash(payload_hash,pc->v,kv_bytes);
    uint64_t payload_bytes=(uint64_t)ids_bytes+2u*(uint64_t)kv_bytes;
    unsigned char h[LPC_HEADER_BYTES]; memset(h,0,sizeof h);
    memcpy(h,LPC_MAGIC,8); lpc_put32(h+8,LPC_VERSION); lpc_put32(h+12,LPC_HEADER_BYTES);
    lpc_put32(h+16,LPC_BUILD_KIND<<8|
#if defined(LAGUNA_KV_FP16)
              1u
#else
              0u
#endif
    );
    lpc_put32(h+20,(uint32_t)pc->n_layers); lpc_put32(h+24,LAGUNA_KV_HEADS);
    lpc_put32(h+28,LAGUNA_HEAD_DIM); lpc_put32(h+32,LAGUNA_SLIDING_CAP);
    lpc_put32(h+36,(uint32_t)pc->n_tokens); lpc_put32(h+40,(uint32_t)pc->ep_rank);
    lpc_put32(h+44,(uint32_t)pc->ep_size); lpc_put64(h+48,pc->model_sig);
    lpc_put64(h+56,ids_hash); lpc_put64(h+64,payload_hash);
    lpc_put64(h+72,payload_bytes); lpc_put64(h+80,LPC_HEADER_BYTES+payload_bytes);
    char *tmp=malloc(strlen(path)+48); if(!tmp){snprintf(err,errn,"out of memory");return -1;}
    sprintf(tmp,"%s.tmp.%ld",path,(long)getpid());
    FILE *f=fopen(tmp,"wb");
    if(!f){snprintf(err,errn,"cannot create %s: %s",tmp,strerror(errno));free(tmp);return -1;}
    int bad=lpc_write_all(f,h,sizeof h)||lpc_write_all(f,pc->ids,ids_bytes)||
            lpc_write_all(f,pc->k,kv_bytes)||lpc_write_all(f,pc->v,kv_bytes);
    if(!bad && (fflush(f)||fsync(fileno(f))))bad=1;
    if(fclose(f))bad=1;
    if(bad||rename(tmp,path)){
        snprintf(err,errn,"cannot commit %s: %s",path,strerror(errno));unlink(tmp);free(tmp);return -1;
    }
    free(tmp); return 0;
}
static int lpc_read_all(FILE *f, void *p, size_t n) { return n==0||fread(p,1,n,f)==n?0:-1; }
static int lpc_load(const char *path, laguna_prompt_cache *pc, const laguna_model *m,
                    char *err, size_t errn) {
    memset(pc,0,sizeof *pc); FILE *f=fopen(path,"rb");
    if(!f){snprintf(err,errn,"cannot open %s: %s",path,strerror(errno));return -1;}
    unsigned char h[LPC_HEADER_BYTES]; struct stat st;
    if(fstat(fileno(f),&st)||lpc_read_all(f,h,sizeof h)){snprintf(err,errn,"truncated prompt cache header");fclose(f);return -1;}
    uint32_t flags=lpc_get32(h+16), nt=lpc_get32(h+36);
    uint32_t expected_flags=LPC_BUILD_KIND<<8|
#if defined(LAGUNA_KV_FP16)
        1u;
#else
        0u;
#endif
    if(memcmp(h,LPC_MAGIC,8)||lpc_get32(h+8)!=LPC_VERSION||lpc_get32(h+12)!=LPC_HEADER_BYTES){snprintf(err,errn,"unsupported prompt cache format");fclose(f);return -1;}
    if(flags!=expected_flags||lpc_get32(h+20)!=(uint32_t)m->n_layers||
       lpc_get32(h+24)!=LAGUNA_KV_HEADS||lpc_get32(h+28)!=LAGUNA_HEAD_DIM||
       lpc_get32(h+32)!=LAGUNA_SLIDING_CAP||lpc_get32(h+40)!=(uint32_t)m->ep_rank||
       lpc_get32(h+44)!=(uint32_t)(m->ep_size>0?m->ep_size:1)||
       lpc_get64(h+48)!=lpc_model_sig(m)){
        snprintf(err,errn,"prompt cache model/KV format does not match this runner");fclose(f);return -1;
    }
    if(lpc_get64(h+80)!=(uint64_t)st.st_size){snprintf(err,errn,"prompt cache size mismatch");fclose(f);return -1;}
    if(lpc_layout(pc,m,(int)nt,err,errn)){fclose(f);return -1;}
    size_t ib=(size_t)pc->n_tokens*sizeof(int), kb=pc->elems*sizeof(uint16_t);
    if(lpc_get64(h+72)!=(uint64_t)ib+2u*(uint64_t)kb){snprintf(err,errn,"prompt cache payload layout mismatch");fclose(f);return -1;}
    pc->ids=malloc(ib); pc->k=malloc(kb); pc->v=malloc(kb);
    if(!pc->ids||!pc->k||!pc->v){snprintf(err,errn,"out of memory loading prompt cache");fclose(f);lpc_free(pc);return -1;}
    if(lpc_read_all(f,pc->ids,ib)||lpc_read_all(f,pc->k,kb)||lpc_read_all(f,pc->v,kb)){
        snprintf(err,errn,"truncated prompt cache payload");fclose(f);lpc_free(pc);return -1;
    }
    fclose(f);
    uint64_t ih=lpc_hash(1469598103934665603ull,pc->ids,ib), ph=1469598103934665603ull;
    ph=lpc_hash(ph,pc->k,kb); ph=lpc_hash(ph,pc->v,kb);
    if(ih!=lpc_get64(h+56)||ph!=lpc_get64(h+64)){snprintf(err,errn,"prompt cache checksum mismatch");lpc_free(pc);return -1;}
    pc->model_sig=lpc_get64(h+48); return 0;
}
#undef LPC_BUILD_KIND

static int test_prompt_cache(void) {
    laguna_model m; memset(&m,0,sizeof m); m.n_layers=5; m.max_pos=16; m.n_seq=2;
    m.ep_rank=1; m.ep_size=2;
    static uint16_t norm[LAGUNA_HIDDEN], embed[LAGUNA_HIDDEN];
    m.final_norm=norm; m.embed=embed;
    for(int L=0;L<m.n_layers;L++){m.layers[L].in_ln=norm;m.layers[L].post_ln=norm;}
    size_t off=0; int slot=LAGUNA_KV_HEADS*LAGUNA_HEAD_DIM;
    for(int L=0;L<m.n_layers;L++){m.layers[L].is_sliding=(L%4)!=0;m.kv_cap[L]=m.layers[L].is_sliding?4:16;m.kv_off[L]=off;off+=(size_t)m.kv_cap[L]*slot;}
    m.kv_seq_stride=off; m.kcache=malloc(2*off*sizeof(uint16_t));m.vcache=malloc(2*off*sizeof(uint16_t));
    if(!m.kcache||!m.vcache){free(m.kcache);free(m.vcache);return 1;}
    for(size_t i=0;i<2*off;i++){m.kcache[i]=(uint16_t)(i*3+7);m.vcache[i]=(uint16_t)(i*5+11);}
    int ids[7]={2,11,22,33,44,55,66}; char err[256],path[]="/tmp/laguna-pcache-test-XXXXXX";
    int fd=mkstemp(path); if(fd<0){free(m.kcache);free(m.vcache);return 1;}close(fd);unlink(path);
    laguna_prompt_cache a,b; int rc=lpc_capture(&a,&m,1,ids,7,err,sizeof err)||lpc_save(path,&a,err,sizeof err)||lpc_load(path,&b,&m,err,sizeof err);
    if(!rc){int ext[8]={2,11,22,33,44,55,66,77};if(lpc_matches(&b,ids,7)||!lpc_matches(&b,ext,8))rc=1;}
    if(!rc){memset(m.kcache,0,off*sizeof(uint16_t));memset(m.vcache,0,off*sizeof(uint16_t));rc=lpc_restore(&b,&m,0,err,sizeof err);}
    if(!rc)for(int L=0;L<m.n_layers;L++){
        size_t ne=(size_t)b.layer_slots[L]*slot, src=off+m.kv_off[L], dst=m.kv_off[L];
        if(memcmp(m.kcache+dst,m.kcache+src,ne*sizeof(uint16_t))||memcmp(m.vcache+dst,m.vcache+src,ne*sizeof(uint16_t)))rc=1;
    }
    if(!rc){laguna_prompt_cache bad;m.n_layers=4;if(lpc_load(path,&bad,&m,err,sizeof err)==0){lpc_free(&bad);rc=1;}m.n_layers=5;}
    if(!rc){fd=open(path,O_RDWR);unsigned char q;if(fd<0||pread(fd,&q,1,LPC_HEADER_BYTES+4)!=1||pwrite(fd,&(unsigned char){(unsigned char)(q^1)},1,LPC_HEADER_BYTES+4)!=1)rc=1;if(fd>=0)close(fd);}
    if(!rc){laguna_prompt_cache bad;if(lpc_load(path,&bad,&m,err,sizeof err)==0){lpc_free(&bad);rc=1;}}
    unlink(path);lpc_free(&a);lpc_free(&b);free(m.kcache);free(m.vcache);
    if(rc)fprintf(stderr,"prompt cache self-test failed: %s\n",err); return rc;
}
