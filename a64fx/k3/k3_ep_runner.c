/* Kimi K3 A64FX intermediate-TP runner bring-up.
 *
 * This is the first executable distributed slice of the K3 network: every rank
 * owns a contiguous MXFP4 intermediate slice from each of the 16 selected
 * experts.  Routed latent and hidden partials are packed into one all-reduce,
 * matching the intended full-runner MoE collective.  Dummy and staged-real
 * modes deliberately share the same kernel and communication path.
 */
#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <utofu.h>

#include "k3_moe.h"
#include "k3_runtime.h"
#include "../utofu-tests/tofu_demo.h"
#include "../utofu-tests/tp_allreduce.h"

#define K3_RUNNER_MAX_NODES 512
#define K3_SELECTED 16
#define K3_WAIT_TIMEOUT 120.0
#define K3_RUN_STAG DEMO_STAG

typedef enum { K3_MODE_DUMMY, K3_MODE_REAL } k3_mode;
typedef struct {
    int nodes;
    int layers;
    int tokens;
    int threads;
    int layer;
    int profile;
    int kda_threads;
    int fused_threads;
    int fuse_kda_expert;
    k3_mode mode;
    const char *stage_dir;
    const char *status_dir;
    const char *topo_path;
} k3_options;
typedef struct { uint64_t offset,nbytes,rows,cols; char dtype[16],name[512]; } k3_entry;
typedef struct { uint8_t *blob; size_t size; k3_mxfp4_matrix w1,w2,w3; } k3_loaded_expert;

static int g_nodes, g_rank;
static char *g_region;
static size_t g_send_off, g_bar_base, g_slot_send, g_slot_bar;
static utofu_vcq_hdl_t g_vcq;
static utofu_stadd_t g_base;
static utofu_vcq_id_t g_peer_vcq[K3_RUNNER_MAX_NODES];
static utofu_stadd_t g_peer_base[K3_RUNNER_MAX_NODES];
static uint64_t g_bar_token=1;
static const unsigned long g_put_flags=UTOFU_ONESIDED_FLAG_TCQ_NOTICE;

static double now_sec(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void usage(const char *p){
    fprintf(stderr,
        "usage: %s [--mode dummy|real] [--nodes N] [--layers N] [--tokens N]\n"
        "          [--threads N] [--layer N] [--stage-dir DIR]\n"
        "          [--status-dir DIR] [--topo FILE] [--profile] [--kda-threads N]\n"
        "          [--fused-threads N] [--no-fused-team]\n",p);
}
static int parse_int(const char *flag,const char *s,int lo,int hi,int *out){
    char *end=NULL;errno=0;long v=strtol(s,&end,10);
    if(errno||!s[0]||!end||*end||v<lo||v>hi){
        fprintf(stderr,"k3_ep_runner: %s expects an integer in [%d,%d], got '%s'\n",flag,lo,hi,s);return-1;}
    *out=(int)v;return 0;
}
static int parse_options(int argc,char **argv,k3_options *o){
    *o=(k3_options){.nodes=96,.layers=1,.tokens=2,.threads=48,.layer=1,
        .fuse_kda_expert=1,.mode=K3_MODE_DUMMY,.stage_dir="/local/k3-runner",
        .status_dir=".",.topo_path="tofu_topo.txt"};
    for(int i=1;i<argc;++i){const char *a=argv[i];
#define VALUE() do{if(++i>=argc){fprintf(stderr,"k3_ep_runner: missing value for %s\n",a);usage(argv[0]);return-1;}}while(0)
        if(!strcmp(a,"--mode")){VALUE();if(!strcmp(argv[i],"dummy"))o->mode=K3_MODE_DUMMY;
            else if(!strcmp(argv[i],"real"))o->mode=K3_MODE_REAL;
            else{fprintf(stderr,"k3_ep_runner: --mode expects dummy or real, got '%s'\n",argv[i]);return-1;}}
        else if(!strcmp(a,"--nodes")){VALUE();if(parse_int(a,argv[i],1,K3_RUNNER_MAX_NODES,&o->nodes))return-1;}
        else if(!strcmp(a,"--layers")){VALUE();if(parse_int(a,argv[i],1,93,&o->layers))return-1;}
        else if(!strcmp(a,"--tokens")){VALUE();if(parse_int(a,argv[i],1,1048576,&o->tokens))return-1;}
        else if(!strcmp(a,"--threads")){VALUE();if(parse_int(a,argv[i],1,48,&o->threads))return-1;}
        else if(!strcmp(a,"--kda-threads")){VALUE();if(parse_int(a,argv[i],1,48,&o->kda_threads))return-1;}
        else if(!strcmp(a,"--fused-threads")){VALUE();if(parse_int(a,argv[i],1,48,&o->fused_threads))return-1;}
        else if(!strcmp(a,"--no-fused-team")){o->fuse_kda_expert=0;}
        else if(!strcmp(a,"--layer")){VALUE();if(parse_int(a,argv[i],0,92,&o->layer))return-1;}
        else if(!strcmp(a,"--stage-dir")){VALUE();o->stage_dir=argv[i];}
        else if(!strcmp(a,"--status-dir")){VALUE();o->status_dir=argv[i];}
        else if(!strcmp(a,"--topo")){VALUE();o->topo_path=argv[i];}
        else if(!strcmp(a,"--profile")){o->profile=1;}
        else if(!strcmp(a,"-h")||!strcmp(a,"--help")){usage(argv[0]);return 1;}
        else{fprintf(stderr,"k3_ep_runner: unknown argument '%s'\n",a);usage(argv[0]);return-1;}
#undef VALUE
    }
    if(o->nodes>96){
        fprintf(stderr,"k3_ep_runner: --nodes must be in [1,96] so every rank owns at least one native MXFP4 group\n");return-1;}
    if(o->mode==K3_MODE_REAL&&(!o->stage_dir||!o->stage_dir[0])){
        fprintf(stderr,"k3_ep_runner: real mode requires --stage-dir\n");return-1;}
    if(o->mode==K3_MODE_REAL&&(o->layer==0||o->layers!=1)){
        fprintf(stderr,"k3_ep_runner: bounded real mode requires --layer in [1,92] and --layers 1\n");return-1;}
    if(o->layer+o->layers>93){
        fprintf(stderr,"k3_ep_runner: layer range [%d,%d) exceeds the 93-layer network\n",o->layer,o->layer+o->layers);return-1;}
    if(!o->kda_threads)o->kda_threads=o->threads<8?o->threads:8;
    if(!o->fused_threads)o->fused_threads=o->threads;
    if(o->kda_threads>o->threads||o->fused_threads>o->threads){
        fprintf(stderr,"k3_ep_runner: phase team sizes cannot exceed --threads %d\n",o->threads);return-1;}
    return 0;
}

static int read_topology(const char *path,uint8_t coords[][TOFU_NCOORDS]){
    FILE *f=fopen(path,"r");if(!f){fprintf(stderr,"k3_ep_runner: cannot open topology '%s': %s\n",path,strerror(errno));return-1;}
    char line[256];int n=0;
    while(fgets(line,sizeof line,f)){if(line[0]=='#'||line[0]=='\n')continue;
        if(n>=K3_RUNNER_MAX_NODES){fprintf(stderr,"k3_ep_runner: topology exceeds %d ranks\n",K3_RUNNER_MAX_NODES);fclose(f);return-1;}
        unsigned r,c[TOFU_NCOORDS];
        if(sscanf(line,"%u %u %u %u %u %u %u",&r,&c[0],&c[1],&c[2],&c[3],&c[4],&c[5])!=7||(int)r!=n){
            fprintf(stderr,"k3_ep_runner: malformed/out-of-order topology line: %s",line);fclose(f);return-1;}
        for(int k=0;k<TOFU_NCOORDS;++k)if(c[k]>UINT8_MAX){
            fprintf(stderr,"k3_ep_runner: topology coordinate exceeds 255: %s",line);fclose(f);return-1;}
        for(int p=0;p<n;++p){int same=1;for(int k=0;k<TOFU_NCOORDS;++k)same&=coords[p][k]==c[k];
            if(same){fprintf(stderr,"k3_ep_runner: duplicate topology coordinates at ranks %d and %d\n",p,n);fclose(f);return-1;}}
        for(int k=0;k<TOFU_NCOORDS;++k)coords[n][k]=(uint8_t)c[k];++n;
    }
    fclose(f);return n;
}
static size_t bar_recv_off(int rank){return g_bar_base+(size_t)rank*g_slot_bar;}
static size_t bar_go_off(void){return g_bar_base+(size_t)g_nodes*g_slot_bar;}
static int put_issue(utofu_vcq_id_t peer,utofu_stadd_t src,utofu_stadd_t dst,size_t bytes){
    int rc;void *cb;
    do{rc=utofu_put(g_vcq,peer,src,dst,bytes,0,g_put_flags,NULL);if(rc==UTOFU_ERR_BUSY)(void)utofu_poll_tcq(g_vcq,0,&cb);}while(rc==UTOFU_ERR_BUSY);
    if(rc!=UTOFU_SUCCESS)return rc;
    do{rc=utofu_poll_tcq(g_vcq,0,&cb);}while(rc==UTOFU_ERR_NOT_FOUND);
    return rc;
}
static int wait_ge(volatile uint64_t *p,uint64_t want){double start=now_sec();while(*p<want)if(now_sec()-start>K3_WAIT_TIMEOUT)return-1;return 0;}
static void runner_barrier(void){
    uint64_t token=++g_bar_token;char *send=g_region+g_send_off;
    if(g_rank==0){
        for(int r=1;r<g_nodes;++r)if(wait_ge((volatile uint64_t*)(g_region+bar_recv_off(r)),token)){
            fprintf(stderr,"k3_ep_runner rank 0: barrier fan-in timeout from rank %d\n",r);exit(3);}
        for(int r=1;r<g_nodes;++r){*(volatile uint64_t*)send=token;
            int rc=put_issue(g_peer_vcq[r],g_base+g_send_off,g_peer_base[r]+bar_go_off(),8);
            if(rc!=UTOFU_SUCCESS){fprintf(stderr,"k3_ep_runner rank 0: barrier release rank %d rc=%d\n",r,rc);exit(3);}}
    }else{
        volatile uint64_t *go=(volatile uint64_t*)(g_region+bar_go_off());double start=now_sec();
        do{*(volatile uint64_t*)send=token;int rc=put_issue(g_peer_vcq[0],g_base+g_send_off,g_peer_base[0]+bar_recv_off(g_rank),8);
            if(rc!=UTOFU_SUCCESS){fprintf(stderr,"k3_ep_runner rank %d: barrier put rc=%d\n",g_rank,rc);exit(3);}
            for(int i=0;i<50&&*go<token;++i)usleep(2000);
            if(now_sec()-start>K3_WAIT_TIMEOUT){fprintf(stderr,"k3_ep_runner rank %d: barrier release timeout\n",g_rank);exit(3);}
        }while(*go<token);
    }
}

static int load_manifest(const char *path,k3_entry *entries,int cap){
    FILE *f=fopen(path,"r");if(!f){fprintf(stderr,"k3_ep_runner: open manifest '%s': %s\n",path,strerror(errno));return-1;}
    char line[1024];int n=0;
    while(fgets(line,sizeof line,f)){if(line[0]=='#')continue;k3_entry e;int nd;unsigned long long off,bytes,rows,cols;
        if(sscanf(line,"%llu %llu %15s %d %llu %llu %511s",&off,&bytes,e.dtype,&nd,&rows,&cols,e.name)!=7||nd!=2||n>=cap){
            fprintf(stderr,"k3_ep_runner: malformed manifest '%s' near entry %d\n",path,n);fclose(f);return-1;}
        e.offset=off;e.nbytes=bytes;e.rows=rows;e.cols=cols;entries[n++]=e;
    }
    fclose(f);return n;
}
static k3_entry *find_entry(k3_entry *entries,int n,const char *suffix){
    size_t sl=strlen(suffix);for(int i=0;i<n;++i){size_t nl=strlen(entries[i].name);
        if(nl>=sl&&!strcmp(entries[i].name+nl-sl,suffix))return&entries[i];}return NULL;
}
static int load_real_expert(k3_pool *pool,const char *stage_dir,int layer,int expert,
        int local,k3_loaded_expert *out){
    char blob[1024],manifest[1024];
    int nb=snprintf(blob,sizeof blob,"%s/expert%03d/layer%02d_expert%03d.blob",stage_dir,expert,layer,expert);
    int nm=snprintf(manifest,sizeof manifest,"%s/expert%03d/layer%02d_expert%03d.manifest",stage_dir,expert,layer,expert);
    if(nb<0||nm<0||(size_t)nb>=sizeof blob||(size_t)nm>=sizeof manifest){fprintf(stderr,"k3_ep_runner: staged path is too long\n");return-1;}
    k3_entry es[8];int ne=load_manifest(manifest,es,8);if(ne!=6){fprintf(stderr,"k3_ep_runner: '%s' has %d entries, expected 6\n",manifest,ne);return-1;}
    size_t size=0;uint8_t *base=k3_pool_load_blob(pool,blob,&size);if(!base){fprintf(stderr,"%s\n",k3_pool_error(pool));return-1;}
    char prefix[256];int np=snprintf(prefix,sizeof prefix,"language_model.model.layers.%d.block_sparse_moe.experts.%d.",layer,expert);
    size_t max_end=0;int valid=np>0&&(size_t)np<sizeof prefix;
    for(int i=0;i<ne&&valid;++i){size_t end=es[i].offset+es[i].nbytes;
        valid=!strcmp(es[i].dtype,"U8")&&!(es[i].offset%256)&&es[i].offset<=size&&es[i].nbytes<=size-es[i].offset&&
            !strncmp(es[i].name,prefix,(size_t)np);if(end>max_end)max_end=end;
        for(int j=0;j<i&&valid;++j){size_t other_end=es[j].offset+es[j].nbytes;
            valid=end<=es[j].offset||other_end<=es[i].offset;}}
    valid&=max_end==size;
    if(!valid){fprintf(stderr,"k3_ep_runner: identity/layout validation failed for manifest '%s' and blob '%s'\n",manifest,blob);
        k3_pool_free(pool,base);return-1;}
#define MATRIX(prefix,rows_,cols_) do{ \
    k3_entry *p=find_entry(es,ne,#prefix ".weight_packed"),*s=find_entry(es,ne,#prefix ".weight_scale"); \
    if(!p||!s||p->rows!=(uint64_t)(rows_)||p->cols!=(uint64_t)((cols_)/2)|| \
       s->rows!=(uint64_t)(rows_)||s->cols!=(uint64_t)((cols_)/32)){ \
        fprintf(stderr,"k3_ep_runner: invalid %s TP shape in '%s'\n",#prefix,manifest);k3_pool_free(pool,base);return-1;} \
    out->prefix=(k3_mxfp4_matrix){base+p->offset,base+s->offset,(rows_),(cols_)}; \
}while(0)
    MATRIX(w1,local,K3_LATENT);MATRIX(w2,K3_LATENT,local);MATRIX(w3,local,K3_LATENT);
#undef MATRIX
    out->blob=base;out->size=size;return 0;
}

static uint64_t mix64(uint64_t x){x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);return x^(x>>31);}
static int make_dummy_expert(k3_pool *pool,int rank,int expert,int local,k3_loaded_expert *out){
    size_t p13=(size_t)local*K3_LATENT/2,s13=(size_t)local*K3_LATENT/32;
    size_t p2=(size_t)K3_LATENT*local/2,s2=(size_t)K3_LATENT*local/32;
    size_t total=p13+s13+p2+s2+p13+s13;uint8_t *b=k3_pool_alloc(pool,total);if(!b)return-1;
    uint8_t *p=b;out->w1=(k3_mxfp4_matrix){p,p+p13,local,K3_LATENT};p+=p13+s13;
    out->w2=(k3_mxfp4_matrix){p,p+p2,K3_LATENT,local};p+=p2+s2;
    out->w3=(k3_mxfp4_matrix){p,p+p13,local,K3_LATENT};
    k3_mxfp4_matrix *m[3]={&out->w1,&out->w2,&out->w3};
    for(int q=0;q<3;++q){size_t pn=(size_t)m[q]->rows*m[q]->cols/2,sn=(size_t)m[q]->rows*m[q]->cols/32;
        uint64_t state=mix64(UINT64_C(0x4b33000000000000)^((uint64_t)rank<<24)^((uint64_t)expert<<8)^q);
        uint8_t *wp=(uint8_t*)m[q]->packed,*ws=(uint8_t*)m[q]->scale;
        for(size_t i=0;i<pn;++i){state=mix64(state+i+1);wp[i]=(uint8_t)((state&7)|(((state>>8)&7)<<4));}
        /* Small but non-zero synthetic weights avoid immediately saturating the
         * SiTU/tanh residual, keeping checksum drift useful as a smoke signal. */
        memset(ws,110,sn);}
    out->blob=b;out->size=total;return 0;
}
static void write_status(const k3_options *o,const char *state,double seconds,double checksum,size_t peak){
    char path[1024],tmp[1088];int n=snprintf(path,sizeof path,"%s/k3_rank%03d.status",o->status_dir,g_rank);
    if(n<0||(size_t)n>=sizeof path)return;
    n=snprintf(tmp,sizeof tmp,"%s.tmp.%ld",path,(long)getpid());if(n<0||(size_t)n>=sizeof tmp)return;
    FILE *f=fopen(tmp,"wx");if(!f){fprintf(stderr,"k3_ep_runner rank %d: create status '%s': %s\n",g_rank,tmp,strerror(errno));return;}
    int ok=fprintf(f,"rank=%d nodes=%d mode=%s state=%s layers=%d tokens=%d seconds=%.9f checksum=%+.9e peak_bytes=%zu\n",
        g_rank,g_nodes,o->mode==K3_MODE_REAL?"real":"dummy",state,o->layers,o->tokens,seconds,checksum,peak)>=0;
    if(ok)ok=fflush(f)==0;if(ok)ok=fsync(fileno(f))==0;if(fclose(f))ok=0;
    if(ok)ok=rename(tmp,path)==0;
    if(!ok){int saved=errno?errno:EIO;unlink(tmp);fprintf(stderr,"k3_ep_runner rank %d: publish status '%s': %s\n",g_rank,path,strerror(saved));}
}

/* Config lists layers one-based. KDA occupies three of each four through layer
 * 91; layer 92 (one-based) and the final layer 93 are MLA, for 69 KDA + 24 MLA. */
static int layer_is_kda(int layer){int one=layer+1;return one<=91&&(one&3)!=0;}
static int partition_count(int total,int rank,int ranks){int base=total/ranks,extra=total%ranks;return base+(rank<extra);}
static int partition_first(int total,int rank,int ranks){int base=total/ranks,extra=total%ranks;return rank*base+(rank<extra?rank:extra);}
static void residual_update(float *latent,const float *reduced){
    for(int i=0;i<K3_LATENT;++i)latent[i]+=reduced[i]+reduced[K3_LATENT+(i*2)%K3_HIDDEN];
    float inv=0.125f/sqrtf(k3_dot_sve(latent,latent,K3_LATENT)/K3_LATENT+1e-12f);
#if defined(__ARM_FEATURE_SVE)
    int vl=(int)svcntw();for(int i=0;i<K3_LATENT;i+=vl){svbool_t pg=svwhilelt_b32(i,K3_LATENT);
        svst1(pg,latent+i,svmul_n_f32_x(pg,svld1(pg,latent+i),inv));}
#else
    for(int i=0;i<K3_LATENT;++i)latent[i]*=inv;
#endif
}

static void synthetic_attention_prepare(float *shared,const float *latent,
        int global_layer,int token,int local_heads,int first_head,int is_kda,
        float *q,float *k,float *v,float *decay){
    size_t qstride=is_kda?K3_HEAD_DIM:192;
    memset(shared,0,K3_HIDDEN*sizeof(float));
    for(int h=0;h<local_heads;++h){int gh=first_head+h;
        for(size_t d=0;d<qstride;++d){
            float base=latent[(gh*131+(int)d*17+global_layer*29+token*7)%K3_LATENT];
            q[(size_t)h*192+d]=base+0.0001f*(float)(d+1);
            k[(size_t)h*192+d]=base*0.75f-0.00007f*(float)(d+1);
            if(d<K3_HEAD_DIM){v[(size_t)h*K3_HEAD_DIM+d]=latent[(gh*97+(int)d*11+token)%K3_LATENT]*0.5f;
                decay[(size_t)h*K3_HEAD_DIM+d]=0.995f;}}
        if(is_kda){k3_l2_normalize_sve(q+(size_t)h*192,K3_HEAD_DIM,1e-6f);
            k3_l2_normalize_sve(k+(size_t)h*192,K3_HEAD_DIM,1e-6f);}}
}

static void synthetic_attention_project(float *shared,const float *latent,
        const float *attn_out,int global_layer,int token,int local_heads,int first_head){
    for(int h=0;h<local_heads;++h){int gh=first_head+h;
        for(int d=0;d<K3_HEAD_DIM;++d){int out=(gh*K3_HEAD_DIM+d)%K3_HIDDEN;
            shared[out]+=attn_out[(size_t)h*K3_HEAD_DIM+d]*0.03125f;}}
    for(int i=g_rank;i<K3_HIDDEN;i+=g_nodes)
        shared[i]+=0.015625f*latent[(i+global_layer+token)%K3_LATENT];
}

static void synthetic_attention_partial(float *shared,const float *latent,
        int state_slot,int global_layer,int token,int cache_tokens,int local_heads,int first_head,
        float *kda_state,float *mla_keys,float *mla_values,
        float *q,float *k,float *v,float *decay,float *attn_out,
        float *mla_scratch,float *mla_stats,int threads){
    int is_kda=layer_is_kda(global_layer);
    omp_set_num_threads(threads);
    synthetic_attention_prepare(shared,latent,global_layer,token,local_heads,first_head,
        is_kda,q,k,v,decay);
    if(is_kda){
        float beta[96];for(int h=0;h<local_heads;++h)beta[h]=.5f;
        float *state=kda_state+(size_t)state_slot*local_heads*K3_HEAD_DIM*K3_HEAD_DIM;
        k3_kda_step_decay_parallel_sve(attn_out,q,k,v,decay,beta,state,
            local_heads,K3_HEAD_DIM,K3_HEAD_DIM,threads);
    }else{
        for(int h=0;h<local_heads;++h){
            size_t base=((size_t)state_slot*local_heads+h);
            float *kh=mla_keys+(base*(size_t)cache_tokens+(size_t)token)*192;
            float *vh=mla_values+(base*(size_t)cache_tokens+(size_t)token)*K3_HEAD_DIM;
            memcpy(kh,k+(size_t)h*192,192*sizeof(float));memcpy(vh,v+(size_t)h*K3_HEAD_DIM,K3_HEAD_DIM*sizeof(float));}
        const float *layer_keys=mla_keys+(size_t)state_slot*local_heads*cache_tokens*192;
        const float *layer_values=mla_values+(size_t)state_slot*local_heads*cache_tokens*K3_HEAD_DIM;
        if(token+1<128){
#pragma omp parallel for schedule(static)
            for(int h=0;h<local_heads;++h)k3_attention_sve(attn_out+(size_t)h*K3_HEAD_DIM,
                q+(size_t)h*192,layer_keys+(size_t)h*cache_tokens*192,
                layer_values+(size_t)h*cache_tokens*K3_HEAD_DIM,token+1,192,K3_HEAD_DIM);
        }else k3_attention_heads_parallel_sve(attn_out,q,layer_keys,layer_values,
            local_heads,token+1,cache_tokens,192,K3_HEAD_DIM,threads,mla_scratch,mla_stats);
    }
    /* Synthetic row projection: ownership is disjoint, so the following fused
     * allreduce reconstructs one replicated hidden contribution. */
    synthetic_attention_project(shared,latent,attn_out,global_layer,token,local_heads,first_head);
}

static int synthetic_kda_expert_partial(float *shared,float *partial,const float *latent,
        int state_slot,int global_layer,int token,int local_heads,int first_head,
        float *kda_state,float *q,float *k,float *v,float *decay,float *attn_out,
        const k3_mxfp4_matrix *w1,const k3_mxfp4_matrix *w2,const k3_mxfp4_matrix *w3,
        const float *route,float *gate,float *up,int threads,int profile,double timing[2]){
    if(!k3_expert_tp_selected_layout_valid(w1,w2,w3,K3_SELECTED))return-1;
    double start=profile?now_sec():0,expert_start=0;
    synthetic_attention_prepare(shared,latent,global_layer,token,local_heads,first_head,1,q,k,v,decay);
    float beta[96];for(int h=0;h<local_heads;++h)beta[h]=.5f;
    float *state=kda_state+(size_t)state_slot*local_heads*K3_HEAD_DIM*K3_HEAD_DIM;
    omp_set_num_threads(threads);
#pragma omp parallel shared(expert_start)
    {
        k3_kda_step_decay_team_sve(attn_out,q,k,v,decay,beta,state,
            local_heads,K3_HEAD_DIM,K3_HEAD_DIM);
#pragma omp single
        {synthetic_attention_project(shared,latent,attn_out,global_layer,token,local_heads,first_head);
            if(profile){timing[0]=now_sec()-start;expert_start=now_sec();}}
        k3_expert_tp_forward_selected_team_mxfp4(partial,w1,w2,w3,route,
            K3_SELECTED,latent,gate,up);
#pragma omp single
        {if(profile)timing[1]=now_sec()-expert_start;}
    }
    return 0;
}

int main(int argc,char **argv){
    k3_options opt;int parsed=parse_options(argc,argv,&opt);if(parsed)return parsed>0?0:2;
    omp_set_num_threads(opt.threads);
    static uint8_t topo[K3_RUNNER_MAX_NODES][TOFU_NCOORDS];int topo_n=read_topology(opt.topo_path,topo);
    if(topo_n<0)return 2;if(topo_n!=opt.nodes){fprintf(stderr,"k3_ep_runner: topology has %d ranks but --nodes is %d\n",topo_n,opt.nodes);return 2;}
    uint8_t mine[TOFU_NCOORDS]={0};int rc=utofu_query_my_coords(mine);if(rc!=UTOFU_SUCCESS){fprintf(stderr,"k3_ep_runner: utofu_query_my_coords rc=%d\n",rc);return 3;}
    g_nodes=topo_n;g_rank=-1;for(int r=0;r<g_nodes;++r)if(!memcmp(topo[r],mine,TOFU_NCOORDS)){g_rank=r;break;}
    if(g_rank<0){fprintf(stderr,"k3_ep_runner: local coordinates are absent from '%s'\n",opt.topo_path);return 3;}

    k3_pool pool;k3_pool_init(&pool,"k3-ep-runner");
    g_slot_send=DEMO_CACHE_LINE;g_slot_bar=DEMO_CACHE_LINE;g_send_off=0;g_bar_base=g_slot_send;
    size_t region_bytes=g_bar_base+(size_t)(g_nodes+1)*g_slot_bar;
    g_region=k3_pool_calloc(&pool,1,region_bytes);if(!g_region){fprintf(stderr,"%s\n",k3_pool_error(&pool));k3_pool_destroy(&pool);return 2;}

    utofu_tni_id_t *tnis=NULL;size_t ntni=0;rc=utofu_get_onesided_tnis(&tnis,&ntni);
    if(rc!=UTOFU_SUCCESS||ntni<1){fprintf(stderr,"k3_ep_runner rank %d: no one-sided TNI (rc=%d count=%zu)\n",g_rank,rc,ntni);k3_pool_destroy(&pool);return 3;}
    rc=utofu_create_vcq_with_cmp_id(tnis[0],DEMO_CMP_ID,0,&g_vcq);if(rc!=UTOFU_SUCCESS){fprintf(stderr,"k3_ep_runner rank %d: create VCQ rc=%d\n",g_rank,rc);free(tnis);k3_pool_destroy(&pool);return 3;}
    utofu_vcq_id_t self;rc=utofu_query_vcq_id(g_vcq,&self);if(rc!=UTOFU_SUCCESS){fprintf(stderr,"k3_ep_runner rank %d: query VCQ rc=%d\n",g_rank,rc);return 3;}
    rc=utofu_reg_mem_with_stag(g_vcq,g_region,region_bytes,K3_RUN_STAG,0,&g_base);if(rc!=UTOFU_SUCCESS){fprintf(stderr,"k3_ep_runner rank %d: register barrier rc=%d\n",g_rank,rc);return 3;}
    for(int r=0;r<g_nodes;++r){if(r==g_rank){g_peer_vcq[r]=self;g_peer_base[r]=g_base;continue;}
        rc=utofu_construct_vcq_id(topo[r],tnis[0],DEMO_CQ_ID,DEMO_CMP_ID,&g_peer_vcq[r]);
        if(rc==UTOFU_SUCCESS)utofu_set_vcq_id_path(&g_peer_vcq[r],NULL);
        if(rc==UTOFU_SUCCESS)rc=utofu_query_stadd(g_peer_vcq[r],K3_RUN_STAG,&g_peer_base[r]);
        if(rc!=UTOFU_SUCCESS){fprintf(stderr,"k3_ep_runner rank %d: peer %d bootstrap rc=%d\n",g_rank,r,rc);return 3;}}
    free(tnis);runner_barrier();
    tp_comm comm;if(tp_comm_init(&comm,g_vcq,g_peer_vcq,g_rank,g_nodes,K3_MOE_REDUCE_FLOATS,runner_barrier)){
        fprintf(stderr,"k3_ep_runner rank %d: all-reduce initialization failed\n",g_rank);return 3;}
    runner_barrier();

    int local=partition_count(96,g_rank,g_nodes)*K3_EXPERT_TP_BLOCK,ready=1;
    k3_loaded_expert experts[K3_SELECTED];memset(experts,0,sizeof experts);
    for(int e=0;e<K3_SELECTED;++e){int bad=opt.mode==K3_MODE_REAL?
        load_real_expert(&pool,opt.stage_dir,opt.layer,e,local,&experts[e]):
        make_dummy_expert(&pool,g_rank,e,local,&experts[e]);if(bad){ready=0;break;}}
    float ready_sum=(float)ready;tp_allreduce_sum(&comm,&ready_sum,1);
    if((int)lrintf(ready_sum)!=g_nodes){
        if(g_rank==0)fprintf(stderr,"k3_ep_runner: distributed weight readiness failed (%d/%d ranks ready)\n",(int)lrintf(ready_sum),g_nodes);
        write_status(&opt,"load-failed",0,0,pool.peak_active_bytes);runner_barrier();tp_comm_free(&comm);
        utofu_dereg_mem(g_vcq,g_base,0);utofu_free_vcq(g_vcq);k3_pool_destroy(&pool);return 4;}

    k3_mxfp4_matrix w1[K3_SELECTED],w2[K3_SELECTED],w3[K3_SELECTED];float route[K3_SELECTED];
    for(int e=0;e<K3_SELECTED;++e){w1[e]=experts[e].w1;w2[e]=experts[e].w2;w3[e]=experts[e].w3;route[e]=1.0f/K3_SELECTED;}
    float *latent=k3_pool_alloc(&pool,K3_LATENT*sizeof(float));
    float *partial=k3_pool_alloc(&pool,K3_LATENT*sizeof(float));
    float *shared=k3_pool_calloc(&pool,K3_HIDDEN,sizeof(float));
    float *reduce=k3_pool_alloc(&pool,K3_MOE_REDUCE_FLOATS*sizeof(float));
    float *gate=k3_pool_alloc(&pool,(size_t)K3_SELECTED*local*sizeof(float));
    float *up=k3_pool_alloc(&pool,(size_t)K3_SELECTED*local*sizeof(float));
    int local_heads=partition_count(96,g_rank,g_nodes),first_head=partition_first(96,g_rank,g_nodes);
    int state_slot[93],kda_layers=0,mla_layers=0;
    for(int l=0;l<opt.layers;++l)state_slot[l]=layer_is_kda(opt.layer+l)?kda_layers++:mla_layers++;
    size_t kda_elems=(size_t)kda_layers*local_heads*K3_HEAD_DIM*K3_HEAD_DIM;
    size_t mla_key_elems=(size_t)mla_layers*local_heads*opt.tokens*192;
    size_t mla_value_elems=(size_t)mla_layers*local_heads*opt.tokens*K3_HEAD_DIM;
    size_t state_bytes=(kda_elems+mla_key_elems+mla_value_elems)*sizeof(float);
    size_t available=k3_mem_available_bytes(),reserve=(size_t)6<<30;
    int capacity_ok=available>reserve&&state_bytes<=available-reserve;
    if(!capacity_ok)fprintf(stderr,"k3_ep_runner rank %d: state preflight failed: need=%zu MiB MemAvailable=%zu MiB reserve=%zu MiB\n",
        g_rank,state_bytes>>20,available>>20,reserve>>20);
    float *kda_state=capacity_ok&&kda_elems?k3_pool_calloc(&pool,kda_elems,sizeof(float)):NULL;
    float *mla_keys=capacity_ok&&mla_key_elems?k3_pool_calloc(&pool,mla_key_elems,sizeof(float)):NULL;
    float *mla_values=capacity_ok&&mla_value_elems?k3_pool_calloc(&pool,mla_value_elems,sizeof(float)):NULL;
    float *q=k3_pool_alloc(&pool,(size_t)local_heads*192*sizeof(float));
    float *k=k3_pool_alloc(&pool,(size_t)local_heads*192*sizeof(float));
    float *v=k3_pool_alloc(&pool,(size_t)local_heads*K3_HEAD_DIM*sizeof(float));
    float *decay=k3_pool_alloc(&pool,(size_t)local_heads*K3_HEAD_DIM*sizeof(float));
    float *attn_out=k3_pool_alloc(&pool,(size_t)local_heads*K3_HEAD_DIM*sizeof(float));
    float *mla_scratch=k3_pool_alloc(&pool,(size_t)local_heads*opt.threads*K3_HEAD_DIM*sizeof(float));
    float *mla_stats=k3_pool_alloc(&pool,((size_t)local_heads*opt.threads*2+(size_t)local_heads*2)*sizeof(float));
    ready=capacity_ok&&latent&&partial&&shared&&reduce&&gate&&up&&(!kda_elems||kda_state)&&
        (!mla_key_elems||mla_keys)&&(!mla_value_elems||mla_values)&&q&&k&&v&&decay&&attn_out&&mla_scratch&&mla_stats;
    ready_sum=(float)ready;tp_allreduce_sum(&comm,&ready_sum,1);
    if((int)lrintf(ready_sum)!=g_nodes){if(!ready&&pool.error[0])fprintf(stderr,"%s\n",k3_pool_error(&pool));
        if(g_rank==0)fprintf(stderr,"k3_ep_runner: distributed scratch allocation failed (%d/%d ranks ready)\n",(int)lrintf(ready_sum),g_nodes);
        write_status(&opt,"alloc-failed",0,0,pool.peak_active_bytes);runner_barrier();tp_comm_free(&comm);
        utofu_dereg_mem(g_vcq,g_base,0);utofu_free_vcq(g_vcq);k3_pool_destroy(&pool);return 4;}
    for(int i=0;i<K3_LATENT;++i)latent[i]=sinf((float)(i+1)*0.001f)*0.125f;

    runner_barrier();double start=now_sec(),phase[6]={0};int finite=1;
    for(int token=0;token<opt.tokens;++token)for(int layer=0;layer<opt.layers;++layer){int global_layer=opt.layer+layer;
        int fused=opt.fuse_kda_expert&&global_layer>0&&layer_is_kda(global_layer);double pt=0;
        if(fused){double ft[2]={0};if(synthetic_kda_expert_partial(shared,partial,latent,
                state_slot[layer],global_layer,token,local_heads,first_head,kda_state,q,k,v,decay,attn_out,
                w1,w2,w3,route,gate,up,opt.fused_threads,opt.profile,ft))finite=0;
            if(opt.profile){phase[0]+=ft[0];phase[2]+=ft[1];}}
        else{pt=opt.profile?now_sec():0;
            synthetic_attention_partial(shared,latent,state_slot[layer],global_layer,token,opt.tokens,local_heads,first_head,
                kda_state,mla_keys,mla_values,q,k,v,decay,attn_out,mla_scratch,mla_stats,
                layer_is_kda(global_layer)?opt.kda_threads:opt.threads);
            if(opt.profile)phase[layer_is_kda(global_layer)?0:1]+=now_sec()-pt;
            pt=opt.profile?now_sec():0;
            if(global_layer==0)memset(partial,0,K3_LATENT*sizeof(float));
            else if(k3_expert_tp_forward_selected_mxfp4(partial,w1,w2,w3,route,K3_SELECTED,latent,gate,up,NULL,opt.threads))finite=0;
            if(opt.profile)phase[2]+=now_sec()-pt;}
        pt=opt.profile?now_sec():0;
        k3_moe_pack_reduce(reduce,partial,shared);
        if(opt.profile)phase[3]+=now_sec()-pt;pt=opt.profile?now_sec():0;
        tp_allreduce_sum(&comm,reduce,K3_MOE_REDUCE_FLOATS);
        if(opt.profile)phase[4]+=now_sec()-pt;pt=opt.profile?now_sec():0;
        residual_update(latent,reduce);for(int i=0;i<K3_LATENT;++i)finite&=isfinite(latent[i]);
        if(opt.profile)phase[5]+=now_sec()-pt;
    }
    runner_barrier();double seconds=now_sec()-start,checksum=0,norm2=0;
    float latent_max=0,state_max=0,cache_max=0;
    for(int i=0;i<K3_LATENT;++i){float a=fabsf(latent[i]);checksum+=latent[i];norm2+=(double)latent[i]*latent[i];if(a>latent_max)latent_max=a;}
#pragma omp parallel for reduction(max:state_max)
    for(size_t i=0;i<kda_elems;++i){float a=fabsf(kda_state[i]);if(a>state_max)state_max=a;}
#pragma omp parallel for reduction(max:cache_max)
    for(size_t i=0;i<mla_key_elems;++i){float a=fabsf(mla_keys[i]);if(a>cache_max)cache_max=a;}
#pragma omp parallel for reduction(max:cache_max)
    for(size_t i=0;i<mla_value_elems;++i){float a=fabsf(mla_values[i]);if(a>cache_max)cache_max=a;}
    float health[3]={latent_max,state_max,cache_max};tp_allreduce_max(&comm,health,3);
    float checksum_sum=(float)checksum;tp_allreduce_sum(&comm,&checksum_sum,1);
    double disagreement=fabs((double)checksum_sum/g_nodes-checksum);
    finite&=isfinite(checksum)&&isfinite(norm2)&&disagreement<1e-4;
    write_status(&opt,finite?"pass":"fail",seconds,checksum,pool.peak_active_bytes);
    if(g_rank==0){double steps=(double)opt.layers*opt.tokens;
        int nkda=0;for(int l=0;l<opt.layers;++l)nkda+=layer_is_kda(opt.layer+l);
        printf("K3_RUN mode=%s nodes=%d local_channels=%d selected=%d layer_range=[%d,%d) KDA=%d MLA=%d tokens=%d threads=%d kda_threads=%d fused_team=%d fused_threads=%d\n",
            opt.mode==K3_MODE_REAL?"real":"dummy",g_nodes,local,K3_SELECTED,opt.layer,opt.layer+opt.layers,nkda,opt.layers-nkda,opt.tokens,opt.threads,opt.kda_threads,opt.fuse_kda_expert,opt.fused_threads);
        printf("K3_RESULT status=%s wall_s=%.6f layer_steps_per_s=%.3f checksum=%+.9e l2=%.9e disagreement=%.3e peak_MiB=%.2f\n",
            finite?"PASS":"FAIL",seconds,steps/seconds,checksum,sqrt(norm2),disagreement,pool.peak_active_bytes/1048576.0);
        printf("K3_HEALTH kda_slots=%d mla_slots=%d state_MiB=%.2f mla_cache_MiB=%.2f latent_max=%.6e kda_state_max=%.6e mla_cache_max=%.6e\n",
            kda_layers,mla_layers,kda_elems*sizeof(float)/1048576.0,
            (mla_key_elems+mla_value_elems)*sizeof(float)/1048576.0,health[0],health[1],health[2]);
        fflush(stdout);}
    if(opt.profile){const char *names[6]={"kda","mla","expert","pack","allreduce","residual"};double steps=(double)opt.layers*opt.tokens;
        for(int i=0;i<6;++i){float x=(float)phase[i];tp_allreduce_max(&comm,&x,1);phase[i]=x;}
        if(g_rank==0){double sum=0;for(int i=0;i<6;++i)sum+=phase[i];
            printf("K3_PROFILE rank_max_ms_per_layer");for(int i=0;i<6;++i)printf(" %s=%.4f",names[i],phase[i]*1e3/steps);
            printf(" measured_upper=%.4f upper_pct=%.1f\n",sum*1e3/steps,seconds>0?100.0*sum/seconds:0.0);
        fflush(stdout);}
    }
    runner_barrier();tp_comm_free(&comm);utofu_dereg_mem(g_vcq,g_base,0);utofu_free_vcq(g_vcq);k3_pool_destroy(&pool);
    return finite?0:1;
}
