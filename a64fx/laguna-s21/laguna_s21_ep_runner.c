/* Laguna S 2.1 INT4 expert-parallel runner, A64FX / Fugaku.
 *
 * Distributed forward pass + greedy generation over pure uTofu (one rank/node).
 * Experts are EP-sharded (rank owns expert e where e % N == rank); attention,
 * dense layer-0 MLP, shared expert, router, embed, lm_head and norms are
 * replicated on every rank. The only per-MoE-layer communication is one
 * tp_allreduce_sum over the routed partial [hidden].
 *
 * Architecture (verified from config.json / safetensors index):
 *   48 layers, hidden 3072, vocab 100352, GQA 8x128, head_dim 128.
 *   full_attention layers (layer%4==0) = 48 q-heads + YaRN rope (rot 64);
 *   sliding_attention layers = 72 q-heads + default rope (rot 128), window 512.
 *   QK-norm per head before rope; softplus per-head attention gate before o_proj.
 *   MoE: sigmoid+bias top-10 of 256, shared expert, routed_scale 2.5, SwiGLU.
 *   Only routed experts are INT4 group-32; everything else bf16.
 *
 * Build:  make -C a64fx/laguna-s21 all CC=fcc
 * Stage:  mpiexec -np 12 build/laguna_s21_stage      (LAGUNA_EP_SIZE=12)
 * Run:    mpiexec -np 12 [-vcoordfile vc] build/laguna_s21_ep_runner \
 *             --generate --ids prompt.ids --max-new 64 --stage-dir /local/... \
 *             (after tofu_topo_helper produced tofu_topo.txt)
 */
#define _GNU_SOURCE
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "laguna_s21.h"

/* ============================ small helpers ============================ */
static FILE *g_log = NULL;
static int   g_rank = 0;
static void logmsg(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    if (g_log) { va_list a2; va_copy(a2, ap); vfprintf(g_log, fmt, a2); va_end(a2); fflush(g_log); }
    vfprintf(stderr, fmt, ap); va_end(ap);
}
static double now_sec(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static int envi(const char*k,int d){ const char*v=getenv(k); return (v&&*v)?atoi(v):d; }

/* ============================ ABI self-tests ============================ */
static int test_i4(void) {
    uint32_t w[4] = {0xFEDCBA98u,0x76543210u,0x01234567u,0x89ABCDEFu};
    uint16_t s[1] = {laguna_f32_to_bf16(0.25f)}; float x[32]; for(int i=0;i<32;i++)x[i]=(float)(i-15);
    float a=laguna_i4g32_dot(w,s,x,32), b=0; for(int i=0;i<32;i++)b+=(float)laguna_i4_at(w,i)*x[i]*0.25f;
    if(fabsf(a-b)>1e-5f){fprintf(stderr,"INT4 dot mismatch %.8g %.8g\n",a,b);return 1;} return 0;
}
static int test_fht(void) { float a[128],b[128]; for(int i=0;i<128;i++)a[i]=b[i]=(float)(i-63)*0.03125f; laguna_fht128(a);laguna_fht128(a);for(int i=0;i<128;i++)if(fabsf(a[i]-b[i])>2e-5f){fprintf(stderr,"FHT mismatch at %d\n",i);return 1;}return 0; }
static int test_route(void) { float l[256]={0},b[256]={0},w[10];int id[10];for(int i=0;i<256;i++)l[i]=(float)(i-128)*.01f;laguna_top10(l,b,id,w);float z=0;for(int i=0;i<10;i++){if(id[i]!=255-i){fprintf(stderr,"route id %d=%d\n",i,id[i]);return 1;}z+=w[i];}return fabsf(z-1)>1e-6f; }
/* cross-check the group-32 matvec against a scalar per-element reconstruction */
static int test_i4_matvec(void) {
    enum { R=5, C=64 };
    uint32_t packed[R*(C/8)]; uint16_t scales[R*(C/32)]; float x[C], y[R];
    uint64_t st=0x1234;
    for (int i=0;i<R*(C/8);i++){ st=st*6364136223846793005ull+1; packed[i]=(uint32_t)(st>>32); }
    for (int i=0;i<R*(C/32);i++) scales[i]=laguna_f32_to_bf16(0.1f+0.01f*i);
    for (int i=0;i<C;i++) x[i]=(float)((int)(i%7)-3)*0.5f;
    laguna_matvec_i4g32(y, packed, scales, x, R, C);
    for (int r=0;r<R;r++){
        float ref=0; const uint32_t*pw=packed+r*(C/8); const uint16_t*sc=scales+r*(C/32);
        for (int c=0;c<C;c++) ref += (float)laguna_i4_at(pw,c)*laguna_bf16_to_f32(sc[c/32])*x[c];
        if (fabsf(ref-y[r])>1e-4f){ fprintf(stderr,"i4 matvec row %d: %.6g vs %.6g\n",r,y[r],ref); return 1; }
    }
    return 0;
}

/* ============================ stage validation (unchanged) ============================ */
typedef struct { uint64_t off; size_t bytes; char dtype[8]; int nd; long shape[5]; char name[320]; } ent;
static int stage_check(const char *dir, int layers) {
    char path[1200], line[1024]; snprintf(path,sizeof path,"%s/rank00.manifest",dir);
    FILE *f=fopen(path,"r"); if(!f){perror(path);return 2;}
    long cap=4096,n=0,packed=0,scale=0; ent *v=calloc((size_t)cap,sizeof(*v));
    while(fgets(line,sizeof line,f)) { if(line[0]=='#')continue; if(n==cap){cap*=2;v=realloc(v,(size_t)cap*sizeof(*v));}
        ent *e=&v[n]; int pos=0,got=0; got=sscanf(line,"%" SCNu64 " %zu %7s %d%n",&e->off,&e->bytes,e->dtype,&e->nd,&pos);
        if (got != 4 || e->nd < 0 || e->nd > 5) continue;
        for (int d = 0; d < e->nd; d++) {
            int used = 0;
            if (sscanf(line + pos, " %ld%n", &e->shape[d], &used) != 1) { got = 0; break; }
            pos += used;
        }
        if (!got) continue;
        while (line[pos] == ' ') pos++;
        snprintf(e->name, sizeof e->name, "%s", line + pos);
        e->name[strcspn(e->name, "\n")] = 0;
        if (strstr(e->name, ".weight_packed")) packed++;
        if (strstr(e->name, ".weight_scale")) scale++;
        n++;
    }
    fclose(f); struct stat sb; char blob[1200];snprintf(blob,sizeof blob,"%s/rank00.blob",dir);if(stat(blob,&sb)){perror(blob);free(v);return 2;}
    int bad=0; for(long i=0;i<n;i++){ if(v[i].off+v[i].bytes>(uint64_t)sb.st_size){fprintf(stderr,"out of range: %s\n",v[i].name);bad=1;} }
    if(layers>1 && (!packed || packed!=scale)){fprintf(stderr,"packed/scale mismatch %ld/%ld\n",packed,scale);bad=1;}
    printf("stage rank00: %ld tensors, %ld packed INT4 tensors, %.3f GB, %s\n",n,packed,(double)sb.st_size/1e9,bad?"FAIL":"PASS"); free(v);return bad;
}
static int probe_stage(const char *dir) {
    static const char *need[] = {
        "model.embed_tokens.weight", "lm_head.weight", "model.norm.weight",
        "model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.gate_proj.weight",
        "model.layers.1.mlp.gate.weight", "model.layers.1.mlp.experts.0.gate_proj.weight_packed",
        "model.layers.1.mlp.experts.0.gate_proj.weight_scale", NULL};
    char mp[1200],bp[1200],line[1024]; snprintf(mp,sizeof mp,"%s/rank00.manifest",dir); snprintf(bp,sizeof bp,"%s/rank00.blob",dir);
    FILE *f=fopen(mp,"r"); if(!f){perror(mp);return 2;} int fd=open(bp,O_RDONLY);struct stat sb;if(fd<0||fstat(fd,&sb)){perror(bp);if(fd>=0)close(fd);fclose(f);return 2;}
    const unsigned char *base=mmap(NULL,(size_t)sb.st_size,PROT_READ,MAP_PRIVATE,fd,0);if(base==MAP_FAILED){perror("mmap");close(fd);fclose(f);return 2;}
    int found=0; while(fgets(line,sizeof line,f)){uint64_t off;size_t bytes;char dt[8],name[320];int nd,pos=0;if(line[0]=='#'||sscanf(line,"%" SCNu64 " %zu %7s %d%n",&off,&bytes,dt,&nd,&pos)!=4)continue;for(int d=0;d<nd;d++){long q;int used;if(sscanf(line+pos," %ld%n",&q,&used)!=1)break;pos+=used;}while(line[pos]==' ')pos++;snprintf(name,sizeof name,"%s",line+pos);name[strcspn(name,"\n")]=0;for(int i=0;need[i];i++)if(!strcmp(name,need[i])){if(off+bytes>(uint64_t)sb.st_size){fprintf(stderr,"bad tensor %s\n",name);found=-99;}else{printf("probe %s: %s %zu B first=0x%02x\n",name,dt,bytes,base[off]);found++;}}}
    munmap((void*)base,(size_t)sb.st_size);close(fd);fclose(f);if(found!=(int)(sizeof need/sizeof need[0]-1)){fprintf(stderr,"probe missing tensors (%d/8)\n",found);return 1;}return 0;
}

/* ============================ manifest -> model ============================ */
typedef struct { char name[320]; uint64_t off; size_t bytes; char dtype[8]; long shape[5]; int nd; } manifest_ent;

typedef struct {
    const unsigned char *blob; size_t blob_bytes;
    manifest_ent *ents; long n_ents;
} laguna_stage;

static const void *stage_find(const laguna_stage *s, const char *name) {
    for (long i = 0; i < s->n_ents; ++i)
        if (!strcmp(s->ents[i].name, name)) return s->blob + s->ents[i].off;
    return NULL;
}
static const void *stage_req(const laguna_stage *s, const char *name) {
    const void *p = stage_find(s, name);
    if (!p) { fprintf(stderr, "FATAL: missing tensor %s\n", name); exit(1); }
    return p;
}
static int stage_dtype_is_bf16(const laguna_stage *s, const char *name) {
    for (long i = 0; i < s->n_ents; ++i)
        if (!strcmp(s->ents[i].name, name)) return !strcmp(s->ents[i].dtype, "BF16");
    return 0;
}

static int stage_load(laguna_stage *s, const char *dir, int rank) {
    char mp[1200], bp[1200], line[1024];
    snprintf(mp,sizeof mp,"%s/rank%02d.manifest",dir,rank);
    snprintf(bp,sizeof bp,"%s/rank%02d.blob",dir,rank);
    FILE *f=fopen(mp,"r"); if(!f){perror(mp);return -1;}
    int fd=open(bp,O_RDONLY); struct stat sb;
    if(fd<0||fstat(fd,&sb)){perror(bp); if(fd>=0)close(fd); fclose(f); return -1;}
    const unsigned char *fmap=mmap(NULL,(size_t)sb.st_size,PROT_READ,MAP_PRIVATE,fd,0);
    if(fmap==MAP_FAILED){perror("mmap"); close(fd); fclose(f); return -1;}
    size_t nb=(size_t)sb.st_size;
    unsigned char *arena=NULL;
    if(posix_memalign((void**)&arena,256,nb)!=0){fprintf(stderr,"arena alloc %.1f GB failed\n",nb/1e9);munmap((void*)fmap,nb);close(fd);fclose(f);return -1;}
    s->blob=arena; s->blob_bytes=nb;
    /* Parse the manifest first so the arena copy can first-touch each tensor with
     * the SAME per-row static schedule the matvecs use.  Copying the whole blob in
     * arbitrary chunks lands each tensor's pages on a CMG unrelated to the thread
     * that later reads that row -> cross-CMG reads collapse bandwidth (~3x slower
     * decode).  Per-tensor parallel copy keeps weights CMG-local to their reader. */
    long cap=8192; s->n_ents=0; s->ents=malloc((size_t)cap*sizeof(manifest_ent));
    while (fgets(line,sizeof line,f)) {
        if (line[0]=='#') continue;
        if (s->n_ents==cap){ cap*=2; s->ents=realloc(s->ents,(size_t)cap*sizeof(manifest_ent)); }
        manifest_ent *e=&s->ents[s->n_ents]; int pos=0;
        if (sscanf(line,"%" SCNu64 " %zu %7s %d%n",&e->off,&e->bytes,e->dtype,&e->nd,&pos)!=4) continue;
        if (e->nd<0||e->nd>5) continue;
        int ok=1;
        for (int d=0; d<e->nd; d++){ int used=0; if(sscanf(line+pos," %ld%n",&e->shape[d],&used)!=1){ok=0;break;} pos+=used; }
        if(!ok) continue;
        while(line[pos]==' ')pos++;
        snprintf(e->name,sizeof e->name,"%s",line+pos);
        e->name[strcspn(e->name,"\n")]=0;
        if (e->off+e->bytes>s->blob_bytes){ fprintf(stderr,"tensor %s out of blob range\n",e->name); fclose(f); return -1; }
        s->n_ents++;
    }
    fclose(f);
    /* Per-tensor parallel copy for NUMA-local first-touch.  Large tensors are split
     * into page-chunks static across threads (thread t owns the same byte/row range
     * it later reads in the matvec).  Tiny tensors (norms/biases) don't matter for
     * bandwidth, so they're copied one-per-thread via a single dynamic loop -- this
     * avoids thousands of tiny omp-for dispatches that ballooned load time. */
    const size_t PG=(size_t)64<<10, BIG=(size_t)256<<10;
    long ne=s->n_ents;
#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        for (long i=0;i<ne;i++) {
            const manifest_ent *e=&s->ents[i];
            if (e->bytes < BIG) continue;
            long nchunk=(long)((e->bytes+PG-1)/PG);
#ifdef _OPENMP
            #pragma omp for schedule(static) nowait
#endif
            for (long c=0;c<nchunk;c++) {
                size_t o=(size_t)c*PG, len=e->bytes-o<PG?e->bytes-o:PG;
                memcpy(arena+e->off+o, fmap+e->off+o, len);
            }
        }
#ifdef _OPENMP
        #pragma omp for schedule(dynamic,16) nowait
#endif
        for (long i=0;i<ne;i++) {
            const manifest_ent *e=&s->ents[i];
            if (e->bytes >= BIG) continue;
            memcpy(arena+e->off, fmap+e->off, e->bytes);
        }
    }
    munmap((void*)fmap,nb); close(fd);
    return 0;
}

/* Load one linear weight from the stage.  int8 build: quantize bf16 -> per-row
 * W8 at load.  bf16 build (-DLAGUNA_BF16): point straight at the staged bf16. */
#ifdef LAGUNA_BF16
static laguna_lin stage_lin(const laguna_stage *s, const char *name, int rows, int cols) {
    (void)rows; (void)cols; return (laguna_lin)stage_req(s, name);
}
#else
static laguna_lin stage_lin(const laguna_stage *s, const char *name, int rows, int cols) {
    const uint16_t *w = stage_req(s, name);
    laguna_w8 r;
    if (posix_memalign((void**)&r.q, 256, (size_t)rows*cols) != 0 ||
        posix_memalign((void**)&r.s, 256, (size_t)rows*sizeof(float)) != 0) {
        fprintf(stderr, "FATAL: W8 alloc failed for %s\n", name); exit(1);
    }
    laguna_quant_w8(r.q, r.s, w, rows, cols);
    return r;
}
#endif

/* Fill the model from a rank-local stage. n_layers lets a bring-up run truncate.
 * Attention/dense/shared/router/lm_head are quantized to int8 (W8) at load to
 * halve the per-token weight bandwidth; norms/embed stay bf16, experts stay int4. */
static void model_build(laguna_model *m, const laguna_stage *s, int n_layers,
                        int max_pos, int rank, int ep_size) {
    memset(m,0,sizeof *m);
    m->n_layers=n_layers; m->max_pos=max_pos; m->ep_rank=rank; m->ep_size=ep_size;
    m->embed      = stage_req(s,"model.embed_tokens.weight");
    m->lm_head    = stage_lin(s,"lm_head.weight", LAGUNA_VOCAB, LAGUNA_HIDDEN);
    m->final_norm = stage_req(s,"model.norm.weight");
    char nm[320];
    for (int L=0; L<n_layers; ++L) {
        laguna_layer *ly=&m->layers[L];
        int full = (L % 4 == 0);            /* layer_types: layer%4==0 => full_attention */
        ly->is_sliding = !full;
        ly->num_heads  = full ? LAGUNA_FULL_HEADS : LAGUNA_SLIDING_HEADS;
        ly->is_moe     = (L != 0);          /* mlp_only_layers=[0] => layer 0 dense */
        int nh=ly->num_heads, H=LAGUNA_HIDDEN, hd=LAGUNA_HEAD_DIM;
        #define REQ(field,suffix) do{ snprintf(nm,sizeof nm,"model.layers.%d." suffix,L); ly->field=stage_req(s,nm);}while(0)
        #define QREQ(field,suffix,rows,cols) do{ snprintf(nm,sizeof nm,"model.layers.%d." suffix,L); ly->field=stage_lin(s,nm,(rows),(cols));}while(0)
        QREQ(q_proj,  "self_attn.q_proj.weight", nh*hd, H);
        QREQ(k_proj,  "self_attn.k_proj.weight", LAGUNA_KV_HEADS*hd, H);
        QREQ(v_proj,  "self_attn.v_proj.weight", LAGUNA_KV_HEADS*hd, H);
        QREQ(o_proj,  "self_attn.o_proj.weight", H, nh*hd);
        QREQ(g_proj,  "self_attn.g_proj.weight", nh, H);
        REQ(q_norm,  "self_attn.q_norm.weight");
        REQ(k_norm,  "self_attn.k_norm.weight");
        REQ(in_ln,   "input_layernorm.weight");
        REQ(post_ln, "post_attention_layernorm.weight");
        if (!ly->is_moe) {
            QREQ(dense_gate,"mlp.gate_proj.weight", LAGUNA_DENSE_INTER, H);
            QREQ(dense_up,  "mlp.up_proj.weight",   LAGUNA_DENSE_INTER, H);
            QREQ(dense_down,"mlp.down_proj.weight", H, LAGUNA_DENSE_INTER);
        } else {
            QREQ(shared_gate,"mlp.shared_expert.gate_proj.weight", LAGUNA_SHARED_INTER, H);
            QREQ(shared_up,  "mlp.shared_expert.up_proj.weight",   LAGUNA_SHARED_INTER, H);
            QREQ(shared_down,"mlp.shared_expert.down_proj.weight", H, LAGUNA_SHARED_INTER);
            QREQ(router_w,   "mlp.gate.weight", LAGUNA_EXPERTS, H);
            /* router bias: F32 in the int4 checkpoint, BF16 in the bf16 one.
             * Read the manifest dtype and convert to a f32 [256] so the router
             * kernel is uniform. */
            snprintf(nm,sizeof nm,"model.layers.%d.mlp.experts.e_score_correction_bias",L);
            { float *bias=malloc(LAGUNA_EXPERTS*sizeof(float));
              const void *bp=stage_req(s,nm);
              if (stage_dtype_is_bf16(s,nm)) { const uint16_t *b=bp; for(int e=0;e<LAGUNA_EXPERTS;++e) bias[e]=laguna_bf16_to_f32(b[e]); }
              else { memcpy(bias, bp, LAGUNA_EXPERTS*sizeof(float)); }
              ly->router_bias=bias; }
            for (int e=0;e<LAGUNA_EXPERTS;++e) {
                if (e % ep_size != rank) continue;   /* not owned by this rank */
                laguna_expert *ex=&ly->experts[e];
#ifdef LAGUNA_BF16
                snprintf(nm,sizeof nm,"model.layers.%d.mlp.experts.%d.gate_proj.weight",L,e); ex->gate=stage_req(s,nm);
                snprintf(nm,sizeof nm,"model.layers.%d.mlp.experts.%d.up_proj.weight",  L,e); ex->up  =stage_req(s,nm);
                snprintf(nm,sizeof nm,"model.layers.%d.mlp.experts.%d.down_proj.weight", L,e); ex->down=stage_req(s,nm);
#else
                snprintf(nm,sizeof nm,"model.layers.%d.mlp.experts.%d.gate_proj.weight_packed",L,e); ex->gp=stage_req(s,nm);
                snprintf(nm,sizeof nm,"model.layers.%d.mlp.experts.%d.gate_proj.weight_scale", L,e); ex->gs=stage_req(s,nm);
                snprintf(nm,sizeof nm,"model.layers.%d.mlp.experts.%d.up_proj.weight_packed",  L,e); ex->up=stage_req(s,nm);
                snprintf(nm,sizeof nm,"model.layers.%d.mlp.experts.%d.up_proj.weight_scale",   L,e); ex->us=stage_req(s,nm);
                snprintf(nm,sizeof nm,"model.layers.%d.mlp.experts.%d.down_proj.weight_packed",L,e); ex->dp=stage_req(s,nm);
                snprintf(nm,sizeof nm,"model.layers.%d.mlp.experts.%d.down_proj.weight_scale", L,e); ex->ds=stage_req(s,nm);
#endif
                ex->present=1;
            }
        }
        #undef REQ
        #undef QREQ
    }
    /* rope tables + KV cache */
    int hf=LAGUNA_ROPE_FULL_DIM/2, hs=LAGUNA_ROPE_SLIDING_DIM/2;
    m->full_cos=malloc((size_t)max_pos*hf*sizeof(float)); m->full_sin=malloc((size_t)max_pos*hf*sizeof(float));
    m->swa_cos =malloc((size_t)max_pos*hs*sizeof(float)); m->swa_sin =malloc((size_t)max_pos*hs*sizeof(float));
    laguna_build_rope_tables(m);
    m->kv_layer_stride=(size_t)max_pos*LAGUNA_KV_HEADS*LAGUNA_HEAD_DIM;
    size_t kv_elems=(size_t)n_layers*m->kv_layer_stride;
    m->kcache=malloc(kv_elems*sizeof(uint16_t));
    m->vcache=malloc(kv_elems*sizeof(uint16_t));
    if(!m->kcache||!m->vcache){ fprintf(stderr,"FATAL: KV cache alloc failed (%zu MB)\n",kv_elems*2*2/(1u<<20)); exit(1); }
}

/* ============================ forward pass ============================ */
/* per-token scratch (allocated once) */
typedef struct {
    float *n1, *n2;        /* [hidden] normed activations */
    float *qf, *kf, *vf;   /* q [maxheads*128], k/v [8*128] */
    float *gf;             /* [maxheads] attention gate */
    float *ao;             /* [maxheads*128] attention output pre-o_proj */
    float *attn_out;       /* [hidden] */
    float *inter_a, *inter_b; /* [dense_inter] swiglu scratch */
    float *partial;        /* [hidden] routed accumulator */
    float *shared;         /* [hidden] */
    float *logits;         /* [vocab] */
    float *scores;         /* [max_pos] attention scores */
} laguna_scratch;

static void scratch_alloc(laguna_scratch *sc, int max_pos) {
    sc->n1=malloc(LAGUNA_HIDDEN*sizeof(float));
    sc->n2=malloc(LAGUNA_HIDDEN*sizeof(float));
    sc->qf=malloc((size_t)LAGUNA_MAX_HEADS*LAGUNA_HEAD_DIM*sizeof(float));
    sc->kf=malloc(LAGUNA_KV_HEADS*LAGUNA_HEAD_DIM*sizeof(float));
    sc->vf=malloc(LAGUNA_KV_HEADS*LAGUNA_HEAD_DIM*sizeof(float));
    sc->gf=malloc(LAGUNA_MAX_HEADS*sizeof(float));
    sc->ao=malloc((size_t)LAGUNA_MAX_HEADS*LAGUNA_HEAD_DIM*sizeof(float));
    sc->attn_out=malloc(LAGUNA_HIDDEN*sizeof(float));
    sc->inter_a=malloc(LAGUNA_DENSE_INTER*sizeof(float));
    sc->inter_b=malloc(LAGUNA_DENSE_INTER*sizeof(float));
    sc->partial=malloc(LAGUNA_HIDDEN*sizeof(float));
    sc->shared=malloc(LAGUNA_HIDDEN*sizeof(float));
    sc->logits=malloc(LAGUNA_VOCAB*sizeof(float));
    sc->scores=malloc((size_t)max_pos*sizeof(float));
}

/* Attention for one token at `pos`. `x` is input_layernorm output (n1). Writes attn_out. */
static void attention(const laguna_model *m, const laguna_layer *ly, laguna_scratch *sc,
                      int layer, int pos) {
    int nh=ly->num_heads, hd=LAGUNA_HEAD_DIM;
    int rot = ly->is_sliding ? LAGUNA_ROPE_SLIDING_DIM : LAGUNA_ROPE_FULL_DIM;
    const float *cosp = ly->is_sliding ? m->swa_cos : m->full_cos;
    const float *sinp = ly->is_sliding ? m->swa_sin : m->full_sin;
    int half = rot/2;
    const float *rc = cosp + (size_t)pos*half, *rs = sinp + (size_t)pos*half;

    /* q/k/v/g all read n1 -> one fused parallel region */
    { float *ys[4]={sc->qf,sc->kf,sc->vf,sc->gf};
      const laguna_lin *Ws[4]={&ly->q_proj,&ly->k_proj,&ly->v_proj,&ly->g_proj};
      int rws[4]={nh*hd, LAGUNA_KV_HEADS*hd, LAGUNA_KV_HEADS*hd, nh};
      laguna_lin_mv_multi(ys, Ws, rws, 4, sc->n1, LAGUNA_HIDDEN); }

    /* q_norm + rope per query head */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int h=0; h<nh; ++h) {
        float *q=sc->qf+(size_t)h*hd; float tmp[LAGUNA_HEAD_DIM];
        laguna_rmsnorm(tmp, q, ly->q_norm, hd, LAGUNA_RMS_EPS);
        memcpy(q,tmp,sizeof tmp);
        laguna_rope_half(q, rc, rs, rot);
    }
    /* k_norm + rope per kv head, then store to cache at pos */
    uint16_t *kdst=m->kcache+(size_t)layer*m->kv_layer_stride+(size_t)pos*LAGUNA_KV_HEADS*hd;
    uint16_t *vdst=m->vcache+(size_t)layer*m->kv_layer_stride+(size_t)pos*LAGUNA_KV_HEADS*hd;
    for (int h=0; h<LAGUNA_KV_HEADS; ++h) {
        float *k=sc->kf+(size_t)h*hd; float tmp[LAGUNA_HEAD_DIM];
        laguna_rmsnorm(tmp, k, ly->k_norm, hd, LAGUNA_RMS_EPS);
        memcpy(k,tmp,sizeof tmp);
        laguna_rope_half(k, rc, rs, rot);
        for (int d=0; d<hd; ++d) kdst[h*hd+d]=laguna_f32_to_bf16(k[d]);
        for (int d=0; d<hd; ++d) vdst[h*hd+d]=laguna_f32_to_bf16(sc->vf[h*hd+d]);
    }
    /* attention per query head */
    float scale=1.0f/sqrtf((float)hd);
    int kv_groups=nh/LAGUNA_KV_HEADS;
    int lo = ly->is_sliding ? (pos-LAGUNA_SLIDING_WINDOW+1) : 0; if(lo<0)lo=0;
    const uint16_t *kbase=m->kcache+(size_t)layer*m->kv_layer_stride;
    const uint16_t *vbase=m->vcache+(size_t)layer*m->kv_layer_stride;
    /* Per-head attention, parallel over heads with a one-pass online softmax
     * (running max/sum + rescaled accumulator) so there is no shared score buffer. */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int h=0; h<nh; ++h) {
        const float *q=sc->qf+(size_t)h*hd;
        int kvh=h/kv_groups;
        float acc[LAGUNA_HEAD_DIM]; for(int d=0;d<hd;++d)acc[d]=0.0f;
        float m_i=-INFINITY, l_i=0.0f;
        for (int j=lo; j<=pos; ++j) {
            const uint16_t *kj=kbase+(size_t)j*LAGUNA_KV_HEADS*hd+(size_t)kvh*hd;
            float dot=0; for(int d=0;d<hd;++d) dot+=q[d]*laguna_bf16_to_f32(kj[d]);
            dot*=scale;
            float m_new = dot>m_i ? dot : m_i;
            float corr = expf(m_i - m_new);
            float p = expf(dot - m_new);
            l_i = l_i*corr + p;
            const uint16_t *vj=vbase+(size_t)j*LAGUNA_KV_HEADS*hd+(size_t)kvh*hd;
            for(int d=0;d<hd;++d) acc[d]=acc[d]*corr + p*laguna_bf16_to_f32(vj[d]);
            m_i=m_new;
        }
        float gate=laguna_softplus(sc->gf[h]);
        float s=gate/l_i; float *o=sc->ao+(size_t)h*hd;
        for(int d=0;d<hd;++d) o[d]=acc[d]*s;
    }
    /* o_proj: [hidden, nh*hd] * ao -> attn_out */
    laguna_lin_mv(sc->attn_out, &ly->o_proj, sc->ao, LAGUNA_HIDDEN, nh*hd);
}

/* dense/shared SwiGLU: out[hidden] = down( silu(gate(x)) * up(x) ). */
static void swiglu_lin(laguna_scratch *sc, const laguna_lin *gate_w, const laguna_lin *up_w,
                       const laguna_lin *down_w, const float *x, float *out, int inter) {
    float *ys[2]={sc->inter_a, sc->inter_b};
    const laguna_lin *Ws[2]={gate_w, up_w}; int rws[2]={inter, inter};
    laguna_lin_mv_multi(ys, Ws, rws, 2, x, LAGUNA_HIDDEN);   /* gate & up share x */
    for (int i=0;i<inter;++i) sc->inter_a[i]=laguna_silu(sc->inter_a[i])*sc->inter_b[i];
    laguna_lin_mv(out, down_w, sc->inter_a, LAGUNA_HIDDEN, inter);
}

/* One routed expert's SwiGLU into `out` (hidden). inter=1024. INT4 (production)
 * or bf16 (-DLAGUNA_BF16). */
static void expert_mv(laguna_scratch *sc, const laguna_expert *ex, const float *x, float *out) {
    int inter=LAGUNA_EXPERT_INTER;
#ifdef LAGUNA_BF16
    laguna_matvec_bf16(sc->inter_a, ex->gate, x, inter, LAGUNA_HIDDEN);
    laguna_matvec_bf16(sc->inter_b, ex->up,   x, inter, LAGUNA_HIDDEN);
    for (int i=0;i<inter;++i) sc->inter_a[i]=laguna_silu(sc->inter_a[i])*sc->inter_b[i];
    laguna_matvec_bf16(out, ex->down, sc->inter_a, LAGUNA_HIDDEN, inter);
#else
    laguna_matvec_i4g32(sc->inter_a, ex->gp, ex->gs, x, inter, LAGUNA_HIDDEN);
    laguna_matvec_i4g32(sc->inter_b, ex->up, ex->us, x, inter, LAGUNA_HIDDEN);
    for (int i=0;i<inter;++i) sc->inter_a[i]=laguna_silu(sc->inter_a[i])*sc->inter_b[i];
    laguna_matvec_i4g32(out, ex->dp, ex->ds, sc->inter_a, LAGUNA_HIDDEN, inter);
#endif
}

/* Async all-reduce interface: launch() posts the reduction (on a comm thread),
 * join() waits for it. Between them the caller runs independent compute (the
 * shared expert) so the per-MoE-layer allreduce overlaps it. NULL => single node
 * (no reduction needed). */
typedef struct {
    void (*launch)(void *ctx, float *buf, int count);
    void (*join)(void *ctx);
    void *ctx;
} laguna_async_ar;

/* Full forward for one token. `x` = residual stream (f32 [hidden]), updated in place.
 * If compute_logits, writes sc->logits. ar/ar_ctx do the routed allreduce. */
static int g_dbg=0;
static double vnorm(const float*v,int n){ double s=0; for(int i=0;i<n;i++)s+=(double)v[i]*v[i]; return sqrt(s); }
/* lightweight phase profiling (enabled by the bench) */
double g_t_attn=0, g_t_mlp=0, g_t_norm=0; int g_prof=0;
static double prof_now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static void forward_token(const laguna_model *m, laguna_scratch *sc, float *x, int pos,
                          laguna_async_ar *aar, int compute_logits) {
    for (int L=0; L<m->n_layers; ++L) {
        const laguna_layer *ly=&m->layers[L];
        double pt0=g_prof?prof_now():0;
        /* --- attention block --- */
        laguna_rmsnorm(sc->n1, x, ly->in_ln, LAGUNA_HIDDEN, LAGUNA_RMS_EPS);
        if(g_prof){double n=prof_now();g_t_norm+=n-pt0;pt0=n;}
        attention(m, ly, sc, L, pos);
        if(g_prof){double n=prof_now();g_t_attn+=n-pt0;pt0=n;}
        if(g_dbg&&compute_logits) logmsg("L%02d pos%d ||x_in||=%.3f ||attn||=%.3f\n",L,pos,vnorm(x,LAGUNA_HIDDEN),vnorm(sc->attn_out,LAGUNA_HIDDEN));
        for (int i=0;i<LAGUNA_HIDDEN;++i) x[i]+=sc->attn_out[i];
        double pm0=g_prof?prof_now():0;
        /* --- mlp block --- */
        laguna_rmsnorm(sc->n2, x, ly->post_ln, LAGUNA_HIDDEN, LAGUNA_RMS_EPS);
        if (!ly->is_moe) {
            swiglu_lin(sc, &ly->dense_gate, &ly->dense_up, &ly->dense_down, sc->n2,
                       sc->attn_out, LAGUNA_DENSE_INTER);
            for (int i=0;i<LAGUNA_HIDDEN;++i) x[i]+=sc->attn_out[i];
        } else {
            /* router: sigmoid + bias, top-10, normalized */
            laguna_lin_mv(sc->logits, &ly->router_w, sc->n2, LAGUNA_EXPERTS, LAGUNA_HIDDEN);
            int ids[LAGUNA_ACTIVE]; float rw[LAGUNA_ACTIVE];
            laguna_top10(sc->logits, ly->router_bias, ids, rw);
            for (int i=0;i<LAGUNA_HIDDEN;++i) sc->partial[i]=0.0f;
            for (int k=0;k<LAGUNA_ACTIVE;++k) {
                int e=ids[k]; const laguna_expert *ex=&ly->experts[e];
                if (!ex->present) continue;
                expert_mv(sc, ex, sc->n2, sc->attn_out);   /* reuse attn_out as expert out */
                float w=rw[k];
                for (int i=0;i<LAGUNA_HIDDEN;++i) sc->partial[i]+=w*sc->attn_out[i];
            }
            /* Launch the routed-partial allreduce on the comm thread, then compute
             * the (independent) shared expert on the OMP team so they overlap. */
            if (aar && aar->launch) aar->launch(aar->ctx, sc->partial, LAGUNA_HIDDEN);
            swiglu_lin(sc, &ly->shared_gate, &ly->shared_up, &ly->shared_down, sc->n2,
                       sc->shared, LAGUNA_SHARED_INTER);
            if (aar && aar->join) aar->join(aar->ctx);
            if(g_dbg&&compute_logits) logmsg("L%02d pos%d ||shared||=%.3f ||routed||=%.3f e0=%d\n",L,pos,vnorm(sc->shared,LAGUNA_HIDDEN),vnorm(sc->partial,LAGUNA_HIDDEN),ids[0]);
            for (int i=0;i<LAGUNA_HIDDEN;++i) x[i]+=sc->shared[i]+LAGUNA_ROUTED_SCALE*sc->partial[i];
        }
        if(g_prof) g_t_mlp+=prof_now()-pm0;
    }
    if (compute_logits) {
        laguna_rmsnorm(sc->n1, x, m->final_norm, LAGUNA_HIDDEN, LAGUNA_RMS_EPS);
        laguna_lin_mv(sc->logits, &m->lm_head, sc->n1, LAGUNA_VOCAB, LAGUNA_HIDDEN);
    }
}

static int argmax(const float *v, int n) {
    int best=0; float bv=v[0];
    for (int i=1;i<n;++i) if(v[i]>bv){bv=v[i];best=i;}
    return best;
}
static void embed_token(const laguna_model *m, int tok, float *x) {
    const uint16_t *row=m->embed+(size_t)tok*LAGUNA_HIDDEN;
    for (int i=0;i<LAGUNA_HIDDEN;++i) x[i]=laguna_bf16_to_f32(row[i]);
}
static int count_nan(const float *v, int n){ int c=0; for(int i=0;i<n;++i) if(!isfinite(v[i]))c++; return c; }

/* ============================ generate driver ============================ */
static int load_ids(const char *path, int **out) {
    FILE *f=fopen(path,"r"); if(!f){perror(path);return -1;}
    int cap=1024,n=0,*v=malloc((size_t)cap*sizeof(int)),t;
    while (fscanf(f,"%d",&t)==1){ if(n==cap){cap*=2;v=realloc(v,(size_t)cap*sizeof(int));} v[n++]=t; }
    fclose(f); *out=v; return n;
}

static void usage(const char *n){
    fprintf(stderr,
      "usage: %s --self-test | --describe | --check-stage DIR [LAYERS] | --probe-stage DIR\n"
      "       %s --generate --ids FILE --max-new N [--maxpos P] [--layers L]\n"
      "                     [--stage-dir DIR] [--gen-out FILE]\n", n, n);
}

#ifndef LAGUNA_BENCH
#include "laguna_tofu.inc"   /* uTofu bootstrap + allreduce glue (single EP group) */

int main(int argc, char **argv) {
    if (argc==2 && !strcmp(argv[1],"--self-test")) {
        int rc=test_i4()|test_fht()|test_route()|test_i4_matvec();
        if(!rc)puts("Laguna S21 ABI self-test: PASS"); return rc;
    }
    if (argc==2 && !strcmp(argv[1],"--describe")) {
        puts("Laguna S21: 48L H=3072 GQA=8x128, full=48h(YaRN rot64)/sliding=72h(rope rot128,win512), 256 experts top-10 INT4 g32, shared+dense0"); return 0;
    }
    if (argc>=3 && argc<=4 && !strcmp(argv[1],"--check-stage")) return stage_check(argv[2],argc==4?atoi(argv[3]):0);
    if (argc==3 && !strcmp(argv[1],"--probe-stage")) return probe_stage(argv[2]);
    if (argc>=2 && !strcmp(argv[1],"--generate")) return run_generate(argc,argv);
    usage(argv[0]); return 2;
}
#endif /* LAGUNA_BENCH */
