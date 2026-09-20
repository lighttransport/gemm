#define _GNU_SOURCE
#define SAFETENSORS_IMPLEMENTATION
#include "../../common/safetensors.h"
#include "dspark_internal.h"

#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static int ds_error(char *buf, size_t size, int code, const char *fmt, ...) {
    if (buf && size) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(buf, size, fmt, ap);
        va_end(ap);
    }
    return code;
}

static double ds_now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static char *ds_path(const char *dir, const char *name) {
    size_t a = strlen(dir), b = strlen(name);
    char *p = malloc(a + b + 2);
    if (!p) return NULL;
    memcpy(p, dir, a);
    p[a] = '/';
    memcpy(p + a + 1, name, b + 1);
    return p;
}

static char *ds_read_text(const char *path, size_t *size_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) || ftell(f) < 0) { fclose(f); return NULL; }
    long n = ftell(f);
    if (fseek(f, 0, SEEK_SET)) { fclose(f); return NULL; }
    char *s = malloc((size_t)n + 1);
    if (!s || fread(s, 1, (size_t)n, f) != (size_t)n) {
        free(s); fclose(f); return NULL;
    }
    fclose(f);
    s[n] = '\0';
    if (size_out) *size_out = (size_t)n;
    return s;
}

static int ds_json_number(const json_val *root, const char *key, int expect) {
    json_val *v = json_obj_get(root, key);
    return v && v->type == JSON_NUMBER && (int)v->num == expect;
}

static int ds_validate_config(const char *dir, char *error, size_t error_size) {
    char *path = ds_path(dir, "config.json");
    size_t n = 0;
    char *text = path ? ds_read_text(path, &n) : NULL;
    free(path);
    if (!text) return ds_error(error,error_size,DSPARK_EIO,"cannot read draft config.json");
    json_val *root = json_parse(text, (int)n);
    free(text);
    if (!root || root->type != JSON_OBJECT) {
        json_free(root);
        return ds_error(error,error_size,DSPARK_EFORMAT,"invalid draft config.json");
    }
    struct { const char *key; int value; } nums[] = {
        {"block_size",7},{"hidden_size",5120},{"intermediate_size",17408},
        {"num_hidden_layers",5},{"num_attention_heads",32},
        {"num_key_value_heads",8},{"head_dim",128},{"vocab_size",248320},
        {"markov_rank",256},{"mask_token_id",248070},
        {"max_position_embeddings",262144}
    };
    for (size_t i=0;i<sizeof(nums)/sizeof(nums[0]);++i) {
        if (!ds_json_number(root,nums[i].key,nums[i].value)) {
            int rc=ds_error(error,error_size,DSPARK_EUNSUPPORTED,
                            "unsupported config: %s must equal %d",nums[i].key,nums[i].value);
            json_free(root); return rc;
        }
    }
    json_val *ids=json_obj_get(root,"target_layer_ids");
    static const int want[5]={5,19,33,47,61};
    if(!ids||ids->type!=JSON_ARRAY||ids->arr.count!=5){json_free(root);return ds_error(error,error_size,DSPARK_EFORMAT,"target_layer_ids must contain five entries");}
    for(int i=0;i<5;i++)if(ids->arr.items[i].type!=JSON_NUMBER||(int)ids->arr.items[i].num!=want[i]){json_free(root);return ds_error(error,error_size,DSPARK_EUNSUPPORTED,"unsupported target_layer_ids");}
    json_free(root);
    return DSPARK_OK;
}

static int ds_check_tensor(const st_context *st, const char *name,
                           const char *dtype, int ndims, const uint64_t *shape,
                           int *index, char *error, size_t error_size) {
    int i=safetensors_find(st,name);
    if(i<0)return ds_error(error,error_size,DSPARK_EFORMAT,"missing tensor: %s",name);
    if(strcmp(safetensors_dtype(st,i),dtype)||safetensors_ndims(st,i)!=ndims)
        return ds_error(error,error_size,DSPARK_EFORMAT,"wrong dtype/rank for tensor: %s",name);
    const uint64_t *got=safetensors_shape(st,i);
    for(int d=0;d<ndims;d++)if(got[d]!=shape[d])
        return ds_error(error,error_size,DSPARK_EFORMAT,"wrong shape for tensor: %s",name);
    if(index)*index=i;
    return DSPARK_OK;
}

static int ds_validate_draft_header(const st_context *st, char *error, size_t error_size) {
    if(!st||st->n_tensors!=62)return ds_error(error,error_size,DSPARK_EFORMAT,"draft checkpoint must contain exactly 62 tensors");
    uint64_t h[1]={DS_HIDDEN}, hd[1]={DS_HEAD_DIM}, one[1]={1};
    uint64_t fc[2]={DS_HIDDEN,5*DS_HIDDEN}, conf[2]={1,DS_HIDDEN+DS_MARKOV_RANK};
    uint64_t mw[2]={DS_VOCAB,DS_MARKOV_RANK};
    int rc=0;
    if((rc=ds_check_tensor(st,"fc.weight","BF16",2,fc,NULL,error,error_size))||
       (rc=ds_check_tensor(st,"hidden_norm.weight","BF16",1,h,NULL,error,error_size))||
       (rc=ds_check_tensor(st,"norm.weight","BF16",1,h,NULL,error,error_size))||
       (rc=ds_check_tensor(st,"markov_head.markov_w1.weight","BF16",2,mw,NULL,error,error_size))||
       (rc=ds_check_tensor(st,"markov_head.markov_w2.weight","BF16",2,mw,NULL,error,error_size))||
       (rc=ds_check_tensor(st,"confidence_head.proj.weight","BF16",2,conf,NULL,error,error_size))||
       (rc=ds_check_tensor(st,"confidence_head.proj.bias","BF16",1,one,NULL,error,error_size)))return rc;
    for(int l=0;l<DS_LAYERS;l++){
        char n[128]; uint64_t q[2]={DS_Q_DIM,DS_HIDDEN},kv[2]={DS_KV_DIM,DS_HIDDEN};
        uint64_t o[2]={DS_HIDDEN,DS_Q_DIM},gu[2]={DS_INTERMEDIATE,DS_HIDDEN};
        uint64_t down[2]={DS_HIDDEN,DS_INTERMEDIATE};
        #define DS_CK(S,D,R,SH) do{snprintf(n,sizeof(n),"layers.%d.%s",l,(S));rc=ds_check_tensor(st,n,(D),(R),(SH),NULL,error,error_size);if(rc)return rc;}while(0)
        DS_CK("input_layernorm.weight","BF16",1,h);
        DS_CK("post_attention_layernorm.weight","BF16",1,h);
        DS_CK("self_attn.q_norm.weight","BF16",1,hd);
        DS_CK("self_attn.k_norm.weight","BF16",1,hd);
        DS_CK("self_attn.q_proj.weight","BF16",2,q);
        DS_CK("self_attn.k_proj.weight","BF16",2,kv);
        DS_CK("self_attn.v_proj.weight","BF16",2,kv);
        DS_CK("self_attn.o_proj.weight","BF16",2,o);
        DS_CK("mlp.gate_proj.weight","BF16",2,gu);
        DS_CK("mlp.up_proj.weight","BF16",2,gu);
        DS_CK("mlp.down_proj.weight","BF16",2,down);
        #undef DS_CK
    }
    return DSPARK_OK;
}

static int ds_target_file(const char *target_dir, const char *tensor,
                          char **path_out, char *error, size_t error_size) {
    char *index_path=ds_path(target_dir,"model.safetensors.index.json");
    size_t n=0; char *text=index_path?ds_read_text(index_path,&n):NULL; free(index_path);
    if(!text)return ds_error(error,error_size,DSPARK_EIO,"cannot read target safetensors index");
    json_val *root=json_parse(text,(int)n); free(text);
    json_val *map=root?json_obj_get(root,"weight_map"):NULL;
    json_val *v=map?json_obj_get(map,tensor):NULL;
    if(!v||v->type!=JSON_STRING){json_free(root);return ds_error(error,error_size,DSPARK_EFORMAT,"target index missing tensor: %s",tensor);}
    char *file=malloc((size_t)v->str.len+1);
    if(!file){json_free(root);return DSPARK_ENOMEM;}
    memcpy(file,v->str.ptr,(size_t)v->str.len);file[v->str.len]='\0';
    *path_out=ds_path(target_dir,file); free(file); json_free(root);
    if(!*path_out)return DSPARK_ENOMEM;
    return DSPARK_OK;
}

static int ds_validate_target(const char *target_dir, char *error, size_t error_size) {
    struct target_spec{const char*n,*d;int r;uint64_t s[2];}spec[]={
        {"model.language_model.embed_tokens.weight","BF16",2,{DS_VOCAB,DS_HIDDEN}},
        {"lm_head.weight","U8",2,{DS_VOCAB,DS_HIDDEN/2}},
        {"lm_head.weight_scale","F8_E4M3",2,{DS_VOCAB,DS_HIDDEN/16}},
        {"lm_head.weight_scale_2","F32",0,{0,0}},
        {"lm_head.input_scale","F32",0,{0,0}}
    };
    for(size_t j=0;j<sizeof(spec)/sizeof(spec[0]);j++){
        char *p=NULL;int rc=ds_target_file(target_dir,spec[j].n,&p,error,error_size);if(rc)return rc;
        st_context *st=safetensors_open_header(p);free(p);
        if(!st)return ds_error(error,error_size,DSPARK_EIO,"cannot read target shard header for %s",spec[j].n);
        rc=ds_check_tensor(st,spec[j].n,spec[j].d,spec[j].r,spec[j].s,NULL,error,error_size);
        safetensors_close(st);if(rc)return rc;
    }
    return DSPARK_OK;
}

int dspark_validate_files(const char *draft_dir,const char *target_dir,char *error,size_t error_size){
    if(!draft_dir||!target_dir)return ds_error(error,error_size,DSPARK_EINVAL,"model directories are required");
    int rc=ds_validate_config(draft_dir,error,error_size);if(rc)return rc;
    char *p=ds_path(draft_dir,"model.safetensors");st_context *st=p?safetensors_open_header(p):NULL;free(p);
    if(!st)return ds_error(error,error_size,DSPARK_EIO,"cannot read draft model.safetensors header");
    rc=ds_validate_draft_header(st,error,error_size);safetensors_close(st);if(rc)return rc;
    return ds_validate_target(target_dir,error,error_size);
}

static int ds_pread_all(int fd,void *dst,size_t bytes,off_t offset){
    uint8_t *p=dst;size_t done=0;
    while(done<bytes){ssize_t n=pread(fd,p+done,bytes-done,offset+(off_t)done);if(n<0&&errno==EINTR)continue;if(n<=0)return -1;done+=(size_t)n;}
    return 0;
}

static int ds_load_region(const char *path,off_t offset,size_t bytes,void **out,int threads,char *error,size_t error_size){
    void *src=ds_anon_alloc(bytes);if(!src)return ds_error(error,error_size,DSPARK_ENOMEM,"cannot allocate %.2f GiB load buffer",(double)bytes/(1u<<30));
    int fd=open(path,O_RDONLY);if(fd<0){ds_anon_free(src,bytes);return ds_error(error,error_size,DSPARK_EIO,"cannot open %s",path);}
    int failed=ds_pread_all(fd,src,bytes,offset);
    (void)posix_fadvise(fd,offset,(off_t)bytes,POSIX_FADV_DONTNEED);close(fd);
    if(failed){ds_anon_free(src,bytes);return ds_error(error,error_size,DSPARK_EIO,"short read from %s",path);}
    void *mem=ds_anon_alloc(bytes);if(!mem){ds_anon_free(src,bytes);return ds_error(error,error_size,DSPARK_ENOMEM,"cannot allocate %.2f GiB resident arena",(double)bytes/(1u<<30));}
    /* LLIO serializes concurrent pread into anonymous pages on A64FX.  Read
     * once, then first-touch the resident arena in parallel for HBM locality. */
    #pragma omp parallel num_threads(threads)
    {
        int tid=0,nth=1;
        #ifdef _OPENMP
        tid=omp_get_thread_num();nth=omp_get_num_threads();
        #endif
        size_t a=bytes*(size_t)tid/(size_t)nth,b=bytes*(size_t)(tid+1)/(size_t)nth;
        memcpy((uint8_t*)mem+a,(const uint8_t*)src+a,b-a);
    }
    ds_anon_free(src,bytes);
    *out=mem;return DSPARK_OK;
}

static const uint16_t *ds_tensor_ptr(const st_context *st,const void *base,const char *name){
    int i=safetensors_find(st,name);return i<0?NULL:(const uint16_t*)((const uint8_t*)base+st->tensors[i].offset);
}

static void ds_bind_matrix(ds_bf16_matrix *m,const st_context *st,const void*base,const char*n,size_t rows,size_t cols){m->data=ds_tensor_ptr(st,base,n);m->rows=rows;m->cols=cols;}

static void ds_init_rope(dspark_model *m){
    const double base=10000000.0,factor=32.0,max_pos=8192.0;
    const double beta_fast=32.0,beta_slow=1.0,dim=DS_HEAD_DIM;
    double low=floor(dim*log(max_pos/(beta_fast*2.0*M_PI))/(2.0*log(base)));
    double high=ceil(dim*log(max_pos/(beta_slow*2.0*M_PI))/(2.0*log(base)));
    if(low<0)low=0;
    if(high>dim/2-1)high=dim/2-1;
    for(int i=0;i<DS_HEAD_DIM/2;i++){
        double ordinary=1.0/pow(base,(double)(2*i)/dim),interp=ordinary/factor;
        double ramp=high==low?0.0:((double)i-low)/(high-low);
        if(ramp<0)ramp=0;
        if(ramp>1)ramp=1;
        double extrap=1.0-ramp;
        m->rope_inv_freq[i]=(float)(interp*(1.0-extrap)+ordinary*extrap);
    }
    m->rope_attention_factor=(float)(0.1*log(factor)+1.0);
}

static int ds_load_target_tensor(const char *target_dir,const char *name,const char*dtype,int ndims,const uint64_t*shape,void**mem,size_t*bytes,int threads,char*error,size_t error_size){
    char *p=NULL;int rc=ds_target_file(target_dir,name,&p,error,error_size);if(rc)return rc;
    st_context *st=safetensors_open_header(p);if(!st){free(p);return ds_error(error,error_size,DSPARK_EIO,"cannot open target shard header");}
    int ix=-1;rc=ds_check_tensor(st,name,dtype,ndims,shape,&ix,error,error_size);
    if(!rc){*bytes=st->tensors[ix].nbytes;rc=ds_load_region(p,(off_t)(st->data_offset+st->tensors[ix].offset),*bytes,mem,threads,error,error_size);}
    safetensors_close(st);free(p);return rc;
}

static int ds_load_lm_head(dspark_model*m,const char*target_dir,char*error,size_t error_size){
    uint64_t cs[2]={DS_VOCAB,DS_HIDDEN/2},ss[2]={DS_VOCAB,DS_HIDDEN/16};
    void *src_codes=NULL,*src_scales=NULL,*gs_mem=NULL;size_t cb=0,sb=0,gb=0;
    int rc=ds_load_target_tensor(target_dir,"lm_head.weight","U8",2,cs,&src_codes,&cb,m->threads,error,error_size);if(rc)return rc;
    rc=ds_load_target_tensor(target_dir,"lm_head.weight_scale","F8_E4M3",2,ss,&src_scales,&sb,m->threads,error,error_size);if(rc){ds_anon_free(src_codes,cb);return rc;}
    rc=ds_load_target_tensor(target_dir,"lm_head.weight_scale_2","F32",0,NULL,&gs_mem,&gb,1,error,error_size);if(rc){ds_anon_free(src_codes,cb);ds_anon_free(src_scales,sb);return rc;}
    memcpy(&m->lm_head.global_scale,gs_mem,4);ds_anon_free(gs_mem,gb);
    ds_nvfp4_matrix*w=&m->lm_head;w->n=DS_VOCAB;w->k=DS_HIDDEN;w->groups=DS_HIDDEN/16;w->code_bytes=cb;w->scale_bytes=sb;
    w->codes=ds_anon_alloc(cb);w->scales=ds_anon_alloc(sb);
    if(!w->codes||!w->scales){ds_anon_free(src_codes,cb);ds_anon_free(src_scales,sb);return ds_error(error,error_size,DSPARK_ENOMEM,"cannot allocate packed LM head");}
    /* Source rows are 2.5 KiB and contiguous.  Traverse rows, not panels, so
     * the shared-filesystem pages and source cache lines are consumed once.
     * Each worker owns complete rows; lane-disjoint panel writes cannot race. */
    #pragma omp parallel for num_threads(m->threads) schedule(static)
    for(size_t row=0;row<DS_VOCAB;row++){
        size_t p=row/16,lane=row&15;
        const uint8_t*cr=(const uint8_t*)src_codes+row*(DS_HIDDEN/2);
        const uint8_t*sr=(const uint8_t*)src_scales+row*w->groups;
        for(size_t g=0;g<w->groups;g++){
            w->scales[(p*w->groups+g)*16+lane]=sr[g];
            for(size_t q=0;q<8;q++)w->codes[((p*w->groups+g)*8+q)*16+lane]=cr[g*8+q];
        }
    }
    ds_anon_free(src_codes,cb);ds_anon_free(src_scales,sb);return DSPARK_OK;
}

int dspark_model_load(dspark_model **out,const char*draft_dir,const char*target_dir,const dspark_load_options*options,char*error,size_t error_size){
    if(!out)return ds_error(error,error_size,DSPARK_EINVAL,"output model pointer is required");
    *out=NULL;
    int rc=dspark_validate_files(draft_dir,target_dir,error,error_size);if(rc)return rc;
    dspark_model*m=calloc(1,sizeof(*m));if(!m)return DSPARK_ENOMEM;
    long cpus=sysconf(_SC_NPROCESSORS_ONLN);m->threads=options&&options->threads>0?options->threads:(int)(cpus>48?48:cpus);if(m->threads<1)m->threads=1;
    m->backend=options?options->backend:DSPARK_BACKEND_AUTO;
    if(m->backend==DSPARK_BACKEND_AUTO){
        #if defined(__ARM_FEATURE_SVE)
        m->backend=DSPARK_BACKEND_SVE;
        #else
        m->backend=DSPARK_BACKEND_SCALAR;
        #endif
    }
    #if !defined(__ARM_FEATURE_SVE)
    if(m->backend==DSPARK_BACKEND_SVE){free(m);return ds_error(error,error_size,DSPARK_EUNSUPPORTED,"SVE backend was not compiled");}
    #endif
    static const int ids[5]={5,19,33,47,61};memcpy(m->target_layer_ids,ids,sizeof(ids));m->mask_token_id=DS_MASK_TOKEN;
    char *p=ds_path(draft_dir,"model.safetensors");st_context*st=p?safetensors_open_header(p):NULL;
    if(!st){free(p);free(m);return ds_error(error,error_size,DSPARK_EIO,"cannot reopen draft header");}
    size_t end=0;for(int i=0;i<st->n_tensors;i++){size_t e=st->tensors[i].offset+st->tensors[i].nbytes;if(e>end)end=e;}
    rc=ds_load_region(p,(off_t)st->data_offset,end,&m->draft_arena,m->threads,error,error_size);free(p);if(rc){safetensors_close(st);dspark_model_free(m);return rc;}m->draft_bytes=end;
    ds_bind_matrix(&m->fc,st,m->draft_arena,"fc.weight",DS_HIDDEN,5*DS_HIDDEN);m->hidden_norm=ds_tensor_ptr(st,m->draft_arena,"hidden_norm.weight");m->norm=ds_tensor_ptr(st,m->draft_arena,"norm.weight");
    ds_bind_matrix(&m->markov_w1,st,m->draft_arena,"markov_head.markov_w1.weight",DS_VOCAB,DS_MARKOV_RANK);
    ds_bind_matrix(&m->markov_w2,st,m->draft_arena,"markov_head.markov_w2.weight",DS_VOCAB,DS_MARKOV_RANK);
    m->confidence_weight=ds_tensor_ptr(st,m->draft_arena,"confidence_head.proj.weight");m->confidence_bias=ds_bf16_to_f32(*ds_tensor_ptr(st,m->draft_arena,"confidence_head.proj.bias"));
    for(int l=0;l<DS_LAYERS;l++){
        char n[128];ds_layer*z=&m->layers[l];
        #define DS_PTR(F,S) do{snprintf(n,sizeof(n),"layers.%d.%s",l,(S));(F)=ds_tensor_ptr(st,m->draft_arena,n);}while(0)
        #define DS_MAT(F,S,R,C) do{snprintf(n,sizeof(n),"layers.%d.%s",l,(S));ds_bind_matrix(&(F),st,m->draft_arena,n,(R),(C));}while(0)
        DS_PTR(z->input_norm,"input_layernorm.weight");DS_PTR(z->post_norm,"post_attention_layernorm.weight");DS_PTR(z->q_norm,"self_attn.q_norm.weight");DS_PTR(z->k_norm,"self_attn.k_norm.weight");
        DS_MAT(z->q_proj,"self_attn.q_proj.weight",DS_Q_DIM,DS_HIDDEN);DS_MAT(z->k_proj,"self_attn.k_proj.weight",DS_KV_DIM,DS_HIDDEN);DS_MAT(z->v_proj,"self_attn.v_proj.weight",DS_KV_DIM,DS_HIDDEN);DS_MAT(z->o_proj,"self_attn.o_proj.weight",DS_HIDDEN,DS_Q_DIM);
        DS_MAT(z->gate_proj,"mlp.gate_proj.weight",DS_INTERMEDIATE,DS_HIDDEN);DS_MAT(z->up_proj,"mlp.up_proj.weight",DS_INTERMEDIATE,DS_HIDDEN);DS_MAT(z->down_proj,"mlp.down_proj.weight",DS_HIDDEN,DS_INTERMEDIATE);
        #undef DS_PTR
        #undef DS_MAT
    }
    safetensors_close(st);
    uint64_t es[2]={DS_VOCAB,DS_HIDDEN};void*emb=NULL;size_t eb=0;rc=ds_load_target_tensor(target_dir,"model.language_model.embed_tokens.weight","BF16",2,es,&emb,&eb,m->threads,error,error_size);if(rc){dspark_model_free(m);return rc;}m->embedding_arena=emb;m->embedding_bytes=eb;m->embedding=emb;
    rc=ds_load_lm_head(m,target_dir,error,error_size);if(rc){dspark_model_free(m);return rc;}
    ds_init_rope(m);*out=m;return DSPARK_OK;
}

void dspark_model_free(dspark_model*m){if(!m)return;ds_anon_free(m->draft_arena,m->draft_bytes);ds_anon_free(m->embedding_arena,m->embedding_bytes);ds_anon_free(m->lm_head.codes,m->lm_head.code_bytes);ds_anon_free(m->lm_head.scales,m->lm_head.scale_bytes);free(m);}

const char*dspark_backend_name(dspark_backend b){return b==DSPARK_BACKEND_SVE?"sve":b==DSPARK_BACKEND_SCALAR?"scalar":"auto";}
const int*dspark_target_layer_ids(const dspark_model*m){return m?m->target_layer_ids:NULL;}
size_t dspark_hidden_size(const dspark_model*m){return m?DS_HIDDEN:0;}
size_t dspark_vocab_size(const dspark_model*m){return m?DS_VOCAB:0;}
dspark_backend dspark_model_backend(const dspark_model*m){return m?m->backend:DSPARK_BACKEND_AUTO;}

static void ds_first_touch(void *ptr,size_t bytes,int threads){
    size_t pages=(bytes+4095)/4096;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(size_t p=0;p<pages;p++)((volatile uint8_t*)ptr)[p*4096]=0;
}

int dspark_state_create(dspark_state **out,const dspark_model*m,const dspark_state_options*options,char*error,size_t error_size){
    if(!out||!m)return ds_error(error,error_size,DSPARK_EINVAL,"model and output state are required");
    *out=NULL;
    size_t cap=options&&options->max_context_tokens?options->max_context_tokens:8192;
    if(cap>DS_MAX_POSITION)return ds_error(error,error_size,DSPARK_ERANGE,"context capacity exceeds 262144");
    dspark_state*s=calloc(1,sizeof(*s));if(!s)return DSPARK_ENOMEM;s->model=m;s->capacity=cap;
    size_t bytes=cap*DS_KV_DIM*sizeof(uint16_t);
    for(int l=0;l<DS_LAYERS;l++){
        s->key_cache[l]=ds_anon_alloc(bytes);s->value_cache[l]=ds_anon_alloc(bytes);
        if(!s->key_cache[l]||!s->value_cache[l]){dspark_state_free(s);return ds_error(error,error_size,DSPARK_ENOMEM,"cannot allocate DSpark KV cache");}
        ds_first_touch(s->key_cache[l],bytes,m->threads);ds_first_touch(s->value_cache[l],bytes,m->threads);
    }
    *out=s;return DSPARK_OK;
}

void dspark_state_free(dspark_state*s){if(!s)return;size_t bytes=s->capacity*DS_KV_DIM*sizeof(uint16_t);for(int l=0;l<DS_LAYERS;l++){ds_anon_free(s->key_cache[l],bytes);ds_anon_free(s->value_cache[l],bytes);}free(s);}
void dspark_state_reset(dspark_state*s){if(s)s->cursor=0;}
int dspark_state_truncate(dspark_state*s,size_t n){if(!s||n>s->cursor)return DSPARK_EINVAL;s->cursor=n;return DSPARK_OK;}
size_t dspark_state_context_tokens(const dspark_state*s){return s?s->cursor:0;}

int dspark_state_append_target(dspark_state*s,const float*const taps[DSPARK_TARGET_TAPS],size_t n_tokens,size_t stride,char*error,size_t error_size){
    if(!s||!taps||!n_tokens||stride<DS_HIDDEN)return ds_error(error,error_size,DSPARK_EINVAL,"invalid target-feature append");
    for(int i=0;i<DSPARK_TARGET_TAPS;i++)if(!taps[i])return ds_error(error,error_size,DSPARK_EINVAL,"target tap %d is NULL",i);
    if(n_tokens>s->capacity-s->cursor)return ds_error(error,error_size,DSPARK_ERANGE,"target append exceeds KV capacity");
    const dspark_model*m=s->model;
    float *cat=malloc(8u*5u*DS_HIDDEN*sizeof(float));
    float *projected=malloc(8u*DS_HIDDEN*sizeof(float));
    float *normed=malloc(8u*DS_HIDDEN*sizeof(float));
    float *k=malloc(8u*DS_KV_DIM*sizeof(float)),*v=malloc(8u*DS_KV_DIM*sizeof(float));
    if(!cat||!projected||!normed||!k||!v){free(v);free(k);free(normed);free(projected);free(cat);return ds_error(error,error_size,DSPARK_ENOMEM,"append scratch allocation failed");}
    for(size_t off=0;off<n_tokens;off+=8){
        size_t batch=n_tokens-off<8?n_tokens-off:8;
        for(size_t r=0;r<batch;r++)for(int t=0;t<5;t++)memcpy(cat+(r*5u+(size_t)t)*DS_HIDDEN,taps[t]+(off+r)*stride,DS_HIDDEN*sizeof(float));
        ds_gemm_bf16(m,m->fc.data,DS_HIDDEN,5u*DS_HIDDEN,cat,batch,projected);
        ds_rmsnorm(m->hidden_norm,projected,normed,batch,DS_HIDDEN,DS_RMS_EPS,m->threads);
        for(int l=0;l<DS_LAYERS;l++){
            const ds_layer*z=&m->layers[l];
            ds_gemm_bf16(m,z->k_proj.data,DS_KV_DIM,DS_HIDDEN,normed,batch,k);
            ds_gemm_bf16(m,z->v_proj.data,DS_KV_DIM,DS_HIDDEN,normed,batch,v);
            ds_head_rmsnorm(z->k_norm,k,batch,DS_KV_HEADS,DS_HEAD_DIM,DS_RMS_EPS,m->threads);
            ds_apply_rope(m,k,batch,DS_KV_HEADS,s->cursor+off,m->threads);
            size_t elems=batch*DS_KV_DIM,base=(s->cursor+off)*DS_KV_DIM;
            #pragma omp parallel for num_threads(m->threads) schedule(static)
            for(size_t i=0;i<elems;i++){s->key_cache[l][base+i]=ds_f32_to_bf16(k[i]);s->value_cache[l][base+i]=ds_f32_to_bf16(v[i]);}
        }
    }
    s->cursor+=n_tokens;free(v);free(k);free(normed);free(projected);free(cat);return DSPARK_OK;
}

static int ds_attention(const dspark_state*s,int layer,const float*q,const uint16_t*kn,const uint16_t*vn,float*out){
    const dspark_model*m=s->model;size_t total=s->cursor+DSPARK_BLOCK_SIZE,jobs=DSPARK_BLOCK_SIZE*DS_HEADS;
    const uint16_t*kc=s->key_cache[layer],*vc=s->value_cache[layer];
    float *score_mem=malloc((size_t)m->threads*total*sizeof(float));
    if(!score_mem)return DSPARK_ENOMEM;
    #pragma omp parallel num_threads(m->threads)
    {
        int tid=0;
        #ifdef _OPENMP
        tid=omp_get_thread_num();
        #endif
        float *scores=score_mem+(size_t)tid*total;
        #pragma omp for schedule(static)
        for(size_t job=0;job<jobs;job++){
            size_t row=job/DS_HEADS,head=job%DS_HEADS,kh=head/(DS_HEADS/DS_KV_HEADS);
            const float*qr=q+(row*DS_HEADS+head)*DS_HEAD_DIM;float mx=-FLT_MAX;
            for(size_t at=0;at<total;at++){
                const uint16_t*kp=at<s->cursor?kc+(at*DS_KV_HEADS+kh)*DS_HEAD_DIM:kn+(((at-s->cursor)*DS_KV_HEADS+kh)*DS_HEAD_DIM);
                float z=ds_dot_bf16(qr,kp,DS_HEAD_DIM,m->backend)*(1.0f/sqrtf((float)DS_HEAD_DIM));scores[at]=z;if(z>mx)mx=z;
            }
            double sum=0;for(size_t at=0;at<total;at++){scores[at]=expf(scores[at]-mx);sum+=scores[at];}
            float*dst=out+(row*DS_HEADS+head)*DS_HEAD_DIM;memset(dst,0,DS_HEAD_DIM*sizeof(float));
            for(size_t at=0;at<total;at++){
                float a=scores[at]/(float)sum;const uint16_t*vp=at<s->cursor?vc+(at*DS_KV_HEADS+kh)*DS_HEAD_DIM:vn+(((at-s->cursor)*DS_KV_HEADS+kh)*DS_HEAD_DIM);
                for(size_t d=0;d<DS_HEAD_DIM;d++)dst[d]=fmaf(a,ds_bf16_to_f32(vp[d]),dst[d]);
            }
        }
    }
    free(score_mem);return DSPARK_OK;
}

static int ds_argmax_markov(const dspark_model*m,const float*base,const float*latent,float*best_out){
    float best=-FLT_MAX;int best_id=0;
    #pragma omp parallel num_threads(m->threads)
    {
        float lb=-FLT_MAX;int li=0;
        #pragma omp for schedule(static) nowait
        for(int r=0;r<DS_VOCAB;r++){
            float v=base[r];
            const uint16_t*wr=m->markov_w2.data+(size_t)r*DS_MARKOV_RANK;
            float bias=ds_dot_bf16(latent,wr,DS_MARKOV_RANK,m->backend);
            v+=bias;if(v>lb||(v==lb&&r<li)){lb=v;li=r;}
        }
        #pragma omp critical(dspark_argmax)
        {if(lb>best||(lb==best&&li<best_id)){best=lb;best_id=li;}}
    }
    *best_out=best;return best_id;
}

static float ds_sigmoid(float x){if(x>=0){float z=expf(-x);return 1.0f/(1.0f+z);}float z=expf(x);return z/(1.0f+z);}

int dspark_state_propose(dspark_state*s,int32_t anchor,dspark_proposal*p,char*error,size_t error_size){
    if(!s||!p)return ds_error(error,error_size,DSPARK_EINVAL,"state and proposal are required");
    if(anchor<0||anchor>=DS_VOCAB)return ds_error(error,error_size,DSPARK_ERANGE,"anchor token is outside vocabulary");
    if(s->cursor+DSPARK_BLOCK_SIZE>DS_MAX_POSITION)return ds_error(error,error_size,DSPARK_ERANGE,"proposal positions exceed model limit");
    const dspark_model*m=s->model;const size_t B=DSPARK_BLOCK_SIZE;
    const int profile=getenv("DSPARK_PROFILE")!=NULL;
    double pt=profile?ds_now_sec():0.0,embed_ms=0,norm_ms=0,qkv_ms=0,qk_ms=0;
    double attn_ms=0,oproj_ms=0,ffn_proj_ms=0,ffn_act_ms=0,ffn_down_ms=0;
    double lm_norm_ms=0,lm_head_ms=0,markov_ms=0;
    float *h=malloc(B*DS_HIDDEN*sizeof(float)),*xn=malloc(B*DS_HIDDEN*sizeof(float));
    float *q=malloc(B*DS_Q_DIM*sizeof(float)),*k=malloc(B*DS_KV_DIM*sizeof(float)),*v=malloc(B*DS_KV_DIM*sizeof(float));
    uint16_t *kb=malloc(B*DS_KV_DIM*sizeof(uint16_t)),*vb=malloc(B*DS_KV_DIM*sizeof(uint16_t));
    float *att=malloc(B*DS_Q_DIM*sizeof(float)),*tmp=malloc(B*DS_HIDDEN*sizeof(float));
    float *gate=malloc(B*DS_INTERMEDIATE*sizeof(float)),*up=malloc(B*DS_INTERMEDIATE*sizeof(float));
    float *base=malloc(B*DS_VOCAB*sizeof(float));
    if(!h||!xn||!q||!k||!v||!kb||!vb||!att||!tmp||!gate||!up||!base){free(base);free(up);free(gate);free(tmp);free(att);free(vb);free(kb);free(v);free(k);free(q);free(xn);free(h);return ds_error(error,error_size,DSPARK_ENOMEM,"proposal scratch allocation failed");}
    for(size_t r=0;r<B;r++){
        int id=r?m->mask_token_id:anchor;const uint16_t*e=m->embedding+(size_t)id*DS_HIDDEN;
        for(int i=0;i<DS_HIDDEN;i++)h[r*DS_HIDDEN+i]=ds_bf16_to_f32(e[i]);
    }
    if(profile){double t=ds_now_sec();embed_ms=(t-pt)*1e3;pt=t;}
    for(int l=0;l<DS_LAYERS;l++){
        const ds_layer*z=&m->layers[l];ds_rmsnorm(z->input_norm,h,xn,B,DS_HIDDEN,DS_RMS_EPS,m->threads);
        if(profile){double t=ds_now_sec();norm_ms+=(t-pt)*1e3;pt=t;}
        ds_gemm_bf16(m,z->q_proj.data,DS_Q_DIM,DS_HIDDEN,xn,B,q);ds_gemm_bf16(m,z->k_proj.data,DS_KV_DIM,DS_HIDDEN,xn,B,k);ds_gemm_bf16(m,z->v_proj.data,DS_KV_DIM,DS_HIDDEN,xn,B,v);
        if(profile){double t=ds_now_sec();qkv_ms+=(t-pt)*1e3;pt=t;}
        ds_head_rmsnorm(z->q_norm,q,B,DS_HEADS,DS_HEAD_DIM,DS_RMS_EPS,m->threads);ds_head_rmsnorm(z->k_norm,k,B,DS_KV_HEADS,DS_HEAD_DIM,DS_RMS_EPS,m->threads);
        ds_apply_rope(m,q,B,DS_HEADS,s->cursor,m->threads);ds_apply_rope(m,k,B,DS_KV_HEADS,s->cursor,m->threads);
        #pragma omp parallel for num_threads(m->threads) schedule(static)
        for(size_t i=0;i<B*DS_KV_DIM;i++){kb[i]=ds_f32_to_bf16(k[i]);vb[i]=ds_f32_to_bf16(v[i]);}
        if(profile){double t=ds_now_sec();qk_ms+=(t-pt)*1e3;pt=t;}
        if(ds_attention(s,l,q,kb,vb,att)){
            free(base);free(up);free(gate);free(tmp);free(att);free(vb);free(kb);free(v);free(k);free(q);free(xn);free(h);
            return ds_error(error,error_size,DSPARK_ENOMEM,"attention score allocation failed");
        }
        if(profile){double t=ds_now_sec();attn_ms+=(t-pt)*1e3;pt=t;}
        ds_gemm_bf16(m,z->o_proj.data,DS_HIDDEN,DS_Q_DIM,att,B,tmp);
        for(size_t i=0;i<B*DS_HIDDEN;i++)h[i]+=tmp[i];
        ds_rmsnorm(z->post_norm,h,xn,B,DS_HIDDEN,DS_RMS_EPS,m->threads);
        if(profile){double t=ds_now_sec();oproj_ms+=(t-pt)*1e3;pt=t;}
        ds_gemm_bf16_pair(m,z->gate_proj.data,z->up_proj.data,DS_INTERMEDIATE,
                           DS_HIDDEN,xn,B,gate,up);
        if(profile){double t=ds_now_sec();ffn_proj_ms+=(t-pt)*1e3;pt=t;}
        #pragma omp parallel for num_threads(m->threads) schedule(static)
        for(size_t i=0;i<B*DS_INTERMEDIATE;i++){float g=gate[i];gate[i]=(g/(1.0f+expf(-g)))*up[i];}
        if(profile){double t=ds_now_sec();ffn_act_ms+=(t-pt)*1e3;pt=t;}
        ds_gemm_bf16(m,z->down_proj.data,DS_HIDDEN,DS_INTERMEDIATE,gate,B,tmp);for(size_t i=0;i<B*DS_HIDDEN;i++)h[i]+=tmp[i];
        if(profile){double t=ds_now_sec();ffn_down_ms+=(t-pt)*1e3;pt=t;}
    }
    ds_rmsnorm(m->norm,h,xn,B,DS_HIDDEN,DS_RMS_EPS,m->threads);
    if(profile){double t=ds_now_sec();lm_norm_ms=(t-pt)*1e3;pt=t;}
    ds_nvfp4_gemm(m,&m->lm_head,xn,B,base);
    if(profile){double t=ds_now_sec();lm_head_ms=(t-pt)*1e3;pt=t;}
    int prev=anchor;p->count=B;
    for(size_t r=0;r<B;r++){
        const uint16_t*latent_bf16=m->markov_w1.data+(size_t)prev*DS_MARKOV_RANK;
        float latent[DS_MARKOV_RANK];for(int i=0;i<DS_MARKOV_RANK;i++)latent[i]=ds_bf16_to_f32(latent_bf16[i]);
        int id=ds_argmax_markov(m,base+r*DS_VOCAB,latent,&p->selected_logits[r]);p->token_ids[r]=id;
        float c=m->confidence_bias;
        for(int i=0;i<DS_HIDDEN;i++)c=fmaf(xn[r*DS_HIDDEN+i],ds_bf16_to_f32(m->confidence_weight[i]),c);
        for(int i=0;i<DS_MARKOV_RANK;i++)c=fmaf(latent[i],ds_bf16_to_f32(m->confidence_weight[DS_HIDDEN+i]),c);
        p->confidence[r]=ds_sigmoid(c);prev=id;
    }
    if(profile){
        double t=ds_now_sec();markov_ms=(t-pt)*1e3;
        fprintf(stderr,"DSPARK_PROFILE threads=%d context=%zu embed=%.3f norm=%.3f qkv=%.3f qknorm_rope=%.3f attention=%.3f oproj_norm=%.3f ffn_up_gate=%.3f ffn_act=%.3f ffn_down=%.3f lm_norm=%.3f lm_head=%.3f markov_conf=%.3f total=%.3f ms\n",
                m->threads,s->cursor,embed_ms,norm_ms,qkv_ms,qk_ms,attn_ms,oproj_ms,
                ffn_proj_ms,ffn_act_ms,ffn_down_ms,lm_norm_ms,lm_head_ms,markov_ms,
                embed_ms+norm_ms+qkv_ms+qk_ms+attn_ms+oproj_ms+ffn_proj_ms+
                ffn_act_ms+ffn_down_ms+lm_norm_ms+lm_head_ms+markov_ms);
    }
    free(base);free(up);free(gate);free(tmp);free(att);free(vb);free(kb);free(v);free(k);free(q);free(xn);free(h);return DSPARK_OK;
}
