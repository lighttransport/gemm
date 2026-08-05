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
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <signal.h>
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
#define K3_RUN_REDUCE_FLOATS (K3_MOE_REDUCE_FLOATS+2)
#define K3_CONTROL_SIGNAL K3_MOE_REDUCE_FLOATS
#define K3_CONTROL_NUMERIC (K3_MOE_REDUCE_FLOATS+1)
#define K3_RUN_PREFETCH_THREADS 32

#define K3_CACHE_VERSION 1
typedef struct __attribute__((packed)) {
    char magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint32_t layer;
    uint32_t layers;
    uint32_t local_heads;
    uint32_t kda_layers;
    uint32_t mla_layers;
    uint32_t cache_tokens;
    uint32_t cache_bf16;
    uint64_t kda_elems;
    uint64_t mla_key_elems;
    uint64_t mla_value_elems;
    uint32_t payload_crc;
    uint32_t saved_tokens;
    uint32_t world_rank;
    uint32_t world_nodes;
} k3_cache_header;

static const char k3_cache_magic[8] = {'K','3','C','A','C','H','E','1'};

typedef enum { K3_MODE_DUMMY, K3_MODE_REAL, K3_MODE_HYBRID } k3_mode;

static const char *k3_mode_name(k3_mode mode){
    return mode==K3_MODE_REAL?"real":mode==K3_MODE_HYBRID?"hybrid":"dummy";
}
typedef struct {
    int nodes;
    int tp_nodes;
    int layers;
    int tokens;
    int threads;
    int layer;
    int profile;
    int kda_threads;
    int fused_threads;
    int fuse_kda_expert;
    int mla_cache_bf16;
    int mla_cache_int8;
    int heartbeat_tokens;
    int min_available_mib;
    int ar_groups;
    int comm_robust;
    int comm_ack;
    int comm_deterministic;
    int comm_poll_spins;
    int prefetch_mib;
    int prefetch_threads;
    k3_mode mode;
    const char *cache_load;
    const char *cache_save;
    const char *stage_dir;
    const char *status_dir;
    const char *topo_path;
} k3_options;
typedef struct { uint64_t offset,nbytes,rows,cols; char dtype[16],name[512]; } k3_entry;
typedef struct { uint8_t *blob; size_t size; k3_mxfp4_matrix w1,w2,w3; } k3_loaded_expert;

static int g_nodes, g_rank;
static int g_world_nodes, g_world_rank, g_group, g_contexts;
static char *g_region;
static size_t g_send_off, g_bar_base, g_slot_send, g_slot_bar;
static utofu_vcq_hdl_t g_vcq;
static utofu_stadd_t g_base;
static utofu_vcq_id_t g_peer_vcq[K3_RUNNER_MAX_NODES];
static utofu_stadd_t g_peer_base[K3_RUNNER_MAX_NODES];
static uint64_t g_bar_token=1;
static const unsigned long g_put_flags=UTOFU_ONESIDED_FLAG_TCQ_NOTICE;
static volatile sig_atomic_t g_stop_signal;
static int g_spare_cpu=-1;

static void handle_stop_signal(int sig){g_stop_signal=sig;}
static void runner_drain_mrq(void){struct utofu_mrq_notice notice;
    while(utofu_poll_mrq(g_vcq,0,&notice)==UTOFU_SUCCESS){} }

static double now_sec(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void profile_add(double sum[6],double high[6],int phase,double seconds){
    sum[phase]+=seconds;if(seconds>high[phase])high[phase]=seconds;
}
static void usage(const char *p){
    fprintf(stderr,
        "usage: %s [--mode dummy|real|hybrid] [--nodes N] [--tp-nodes N] [--layers N] [--tokens N]\n"
        "          [--threads N] [--layer N] [--stage-dir DIR]\n"
        "          [--status-dir DIR] [--cache-load PATH] [--cache-save PATH]\n"
        "          [--topo FILE] [--profile] [--kda-threads N]\n"
        "          [--fused-threads N] [--no-fused-team]\n"
        "          [--mla-cache-bf16|--mla-cache-fp32|--mla-cache-int8] [--heartbeat-tokens N]\n"
        "          [--min-available-mib N]\n"
        "          [--ar-groups auto|N] [--comm-robust 1|2] [--comm-ack 0|1]\n"
        "          [--comm-deterministic 0|1]\n"
        "          [--comm-poll-spins N]\n"
        "          [--prefetch-mib N] [--prefetch-threads N]\n"
        "          (ar-groups: auto uses six-rank rows; 0 forces flat)\n",p);
}
static int parse_int(const char *flag,const char *s,int lo,int hi,int *out){
    char *end=NULL;errno=0;long v=strtol(s,&end,10);
    if(errno||!s[0]||!end||*end||v<lo||v>hi){
        fprintf(stderr,"k3_ep_runner: %s expects an integer in [%d,%d], got '%s'\n",flag,lo,hi,s);return-1;}
    *out=(int)v;return 0;
}
static int parse_options(int argc,char **argv,k3_options *o){
    *o=(k3_options){.nodes=96,.layers=1,.tokens=2,.threads=48,.layer=1,
        .mla_cache_bf16=1,.heartbeat_tokens=1024,.min_available_mib=2048,
        .ar_groups=-1,.comm_robust=2,.comm_poll_spins=4,
        .fuse_kda_expert=1,.mode=K3_MODE_DUMMY,.cache_load=NULL,.cache_save=NULL,
        .stage_dir="/local/k3-runner",.status_dir=".",.topo_path="tofu_topo.txt"};
    for(int i=1;i<argc;++i){const char *a=argv[i];
#define VALUE() do{if(++i>=argc){fprintf(stderr,"k3_ep_runner: missing value for %s\n",a);usage(argv[0]);return-1;}}while(0)
        if(!strcmp(a,"--mode")){VALUE();if(!strcmp(argv[i],"dummy"))o->mode=K3_MODE_DUMMY;
            else if(!strcmp(argv[i],"real"))o->mode=K3_MODE_REAL;
            else if(!strcmp(argv[i],"hybrid"))o->mode=K3_MODE_HYBRID;
            else{fprintf(stderr,"k3_ep_runner: --mode expects dummy, real, or hybrid, got '%s'\n",argv[i]);return-1;}}
        else if(!strcmp(a,"--nodes")){VALUE();if(parse_int(a,argv[i],1,K3_RUNNER_MAX_NODES,&o->nodes))return-1;}
        else if(!strcmp(a,"--tp-nodes")){VALUE();if(parse_int(a,argv[i],1,96,&o->tp_nodes))return-1;}
        else if(!strcmp(a,"--layers")){VALUE();if(parse_int(a,argv[i],1,93,&o->layers))return-1;}
        else if(!strcmp(a,"--tokens")){VALUE();if(parse_int(a,argv[i],1,1048576,&o->tokens))return-1;}
        else if(!strcmp(a,"--threads")){VALUE();if(parse_int(a,argv[i],1,48,&o->threads))return-1;}
        else if(!strcmp(a,"--kda-threads")){VALUE();if(parse_int(a,argv[i],1,48,&o->kda_threads))return-1;}
        else if(!strcmp(a,"--fused-threads")){VALUE();if(parse_int(a,argv[i],1,48,&o->fused_threads))return-1;}
        else if(!strcmp(a,"--no-fused-team")){o->fuse_kda_expert=0;}
        else if(!strcmp(a,"--mla-cache-bf16")){o->mla_cache_bf16=1;}
        else if(!strcmp(a,"--mla-cache-fp32")){o->mla_cache_bf16=0;}
        else if(!strcmp(a,"--mla-cache-int8")){o->mla_cache_int8=1;}
        else if(!strcmp(a,"--heartbeat-tokens")){VALUE();if(parse_int(a,argv[i],0,1048576,&o->heartbeat_tokens))return-1;}
        else if(!strcmp(a,"--min-available-mib")){VALUE();if(parse_int(a,argv[i],0,1048576,&o->min_available_mib))return-1;}
        else if(!strcmp(a,"--ar-groups")){VALUE();if(!strcmp(argv[i],"auto"))o->ar_groups=-1;
            else if(parse_int(a,argv[i],0,96,&o->ar_groups))return-1;}
        else if(!strcmp(a,"--comm-robust")){VALUE();if(parse_int(a,argv[i],1,2,&o->comm_robust))return-1;}
        else if(!strcmp(a,"--comm-ack")){VALUE();if(parse_int(a,argv[i],0,1,&o->comm_ack))return-1;}
        else if(!strcmp(a,"--comm-deterministic")){VALUE();if(parse_int(a,argv[i],0,1,&o->comm_deterministic))return-1;}
        else if(!strcmp(a,"--comm-poll-spins")){VALUE();if(parse_int(a,argv[i],1,1024,&o->comm_poll_spins))return-1;}
        else if(!strcmp(a,"--prefetch-mib")){VALUE();if(parse_int(a,argv[i],0,32,&o->prefetch_mib))return-1;}
        else if(!strcmp(a,"--prefetch-threads")){VALUE();if(parse_int(a,argv[i],0,48,&o->prefetch_threads))return-1;}
        else if(!strcmp(a,"--layer")){VALUE();if(parse_int(a,argv[i],0,92,&o->layer))return-1;}
        else if(!strcmp(a,"--stage-dir")){VALUE();o->stage_dir=argv[i];}
        else if(!strcmp(a,"--status-dir")){VALUE();o->status_dir=argv[i];}
        else if(!strcmp(a,"--cache-load")){VALUE();o->cache_load=argv[i];}
        else if(!strcmp(a,"--cache-save")){VALUE();o->cache_save=argv[i];}
        else if(!strcmp(a,"--topo")){VALUE();o->topo_path=argv[i];}
        else if(!strcmp(a,"--profile")){o->profile=1;}
        else if(!strcmp(a,"-h")||!strcmp(a,"--help")){usage(argv[0]);return 1;}
        else{fprintf(stderr,"k3_ep_runner: unknown argument '%s'\n",a);usage(argv[0]);return-1;}
#undef VALUE
    }
    if(!o->tp_nodes)o->tp_nodes=o->nodes<96?o->nodes:96;
    if(o->tp_nodes>96||o->nodes%o->tp_nodes){
        fprintf(stderr,"k3_ep_runner: --tp-nodes must be in [1,96] and divide --nodes (%d), got %d\n",
                o->nodes,o->tp_nodes);return-1;}
    if((o->mode==K3_MODE_REAL||o->mode==K3_MODE_HYBRID)&&(!o->stage_dir||!o->stage_dir[0])){
        fprintf(stderr,"k3_ep_runner: %s mode requires --stage-dir\n",k3_mode_name(o->mode));return-1;}
    if(o->mode==K3_MODE_REAL&&(o->layer==0||o->layers!=1)){
        fprintf(stderr,"k3_ep_runner: bounded real mode requires --layer in [1,92] and --layers 1\n");return-1;}
    if(o->mode==K3_MODE_HYBRID&&o->layer==0){
        fprintf(stderr,"k3_ep_runner: hybrid mode requires --layer in [1,92]\n");return-1;}
    int schedule_layer=o->mode==K3_MODE_HYBRID?0:o->layer;
    if(schedule_layer+o->layers>93){
        fprintf(stderr,"k3_ep_runner: layer range [%d,%d) exceeds the 93-layer network\n",schedule_layer,schedule_layer+o->layers);return-1;}
    if(o->ar_groups<0)o->ar_groups=o->tp_nodes>=12&&o->tp_nodes%6==0?o->tp_nodes/6:0;
    if(o->ar_groups>0&&(o->ar_groups==1||o->tp_nodes%o->ar_groups)){
        fprintf(stderr,"k3_ep_runner: --ar-groups must be 0 or a divisor in [2,--tp-nodes], got %d for TP%d\n",
                o->ar_groups,o->tp_nodes);return-1;}
    if(o->comm_poll_spins&(o->comm_poll_spins-1)){
        fprintf(stderr,"k3_ep_runner: --comm-poll-spins must be a power of two, got %d\n",o->comm_poll_spins);return-1;}
    if(!o->kda_threads)o->kda_threads=o->threads<8?o->threads:8;
    if(!o->fused_threads)o->fused_threads=o->threads;
    if(o->kda_threads>o->threads||o->fused_threads>o->threads){
        fprintf(stderr,"k3_ep_runner: phase team sizes cannot exceed --threads %d\n",o->threads);return-1;}
    if(o->prefetch_mib&&o->threads>47){
        fprintf(stderr,"k3_ep_runner: --prefetch-mib requires --threads <=47 to reserve one communication core\n");return-1;}
    if(o->prefetch_threads>o->threads){
        fprintf(stderr,"k3_ep_runner: --prefetch-threads cannot exceed --threads %d\n",o->threads);return-1;}
    if(o->mla_cache_int8&&(o->cache_load||o->cache_save)){
        fprintf(stderr,"k3_ep_runner: INT8 MLA cache does not support --cache-load/--cache-save yet (scales are not in cache format v1)\n");
        return -1;
    }
    return 0;
}

typedef struct {tp_comm row,col;int groups;} k3_runner_comm;

static uint64_t runner_comm_seq(const k3_runner_comm *c){return c->row.seq;}
static const char *runner_comm_error(const k3_runner_comm *c){
    return c->row.error_message[0]?tp_comm_error(&c->row):tp_comm_error(&c->col);
}
static int runner_allreduce_sum(k3_runner_comm *c,float *buf,int count){
    return c->groups?tp_allreduce_sum_2d_checked(&c->row,&c->col,buf,count):
        tp_allreduce_sum_checked(&c->row,buf,count);
}
static int runner_allreduce_max(k3_runner_comm *c,float *buf,int count){
    return c->groups?tp_allreduce_max_2d_checked(&c->row,&c->col,buf,count):
        tp_allreduce_max_checked(&c->row,buf,count);
}
static void runner_comm_set_robust(k3_runner_comm *c,int robust){
    c->row.robust=robust;if(c->groups)c->col.robust=robust;
}

typedef struct {
    pthread_t thread;
    k3_runner_comm *comm;
    float *buf;
    int count,rc,started,pin_rc;
    atomic_int state; /* 0 idle, 1 request, 2 complete, 3 stop, 4 starting */
} k3_async_reduce;
static inline void runner_event_wait(void){__asm__ __volatile__("wfe" ::: "memory");}
static inline void runner_event_send(void){__asm__ __volatile__("sev" ::: "memory");}
static void *runner_reduce_thread(void *arg){k3_async_reduce*w=arg;cpu_set_t one;
    if(g_spare_cpu>=0){CPU_ZERO(&one);CPU_SET(g_spare_cpu,&one);w->pin_rc=pthread_setaffinity_np(pthread_self(),sizeof one,&one);}
    else w->pin_rc=EINVAL;
    atomic_store_explicit(&w->state,0,memory_order_release);runner_event_send();
    for(;;){int state;while((state=atomic_load_explicit(&w->state,memory_order_acquire))==0||state==2)runner_event_wait();
        if(state==3)break;w->rc=runner_allreduce_sum(w->comm,w->buf,w->count);
        atomic_store_explicit(&w->state,2,memory_order_release);runner_event_send();}return NULL;}
static int runner_async_init(k3_async_reduce*w,k3_runner_comm*c){memset(w,0,sizeof*w);w->comm=c;
    atomic_init(&w->state,4);int rc=pthread_create(&w->thread,NULL,runner_reduce_thread,w);
    if(rc)return rc;w->started=1;while(atomic_load_explicit(&w->state,memory_order_acquire)==4)runner_event_wait();
    if(w->pin_rc){atomic_store_explicit(&w->state,3,memory_order_release);runner_event_send();pthread_join(w->thread,NULL);w->started=0;return w->pin_rc;}
    return 0;}
static void runner_async_submit(k3_async_reduce*w,float*buf,int count){w->buf=buf;w->count=count;
    atomic_store_explicit(&w->state,1,memory_order_release);runner_event_send();}
static int runner_async_wait(k3_async_reduce*w){while(atomic_load_explicit(&w->state,memory_order_acquire)!=2)runner_event_wait();
    int rc=w->rc;atomic_store_explicit(&w->state,0,memory_order_release);return rc;}
static void runner_async_destroy(k3_async_reduce*w){if(!w->started)return;
    while(atomic_load_explicit(&w->state,memory_order_acquire)==1){}
    atomic_store_explicit(&w->state,3,memory_order_release);runner_event_send();pthread_join(w->thread,NULL);w->started=0;}
static void runner_comm_free(k3_runner_comm *c){
    if(c->groups)tp_comm_free_2d(&c->row,&c->col);else tp_comm_free(&c->row);
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
    runner_drain_mrq();return rc;
}
static int wait_ge(volatile uint64_t *p,uint64_t want){double start=now_sec();unsigned spins=0;
    runner_drain_mrq();tp_ar_flag_inval(p);
    while(*p<want){if((++spins&7u)==0){runner_drain_mrq();tp_ar_flag_inval(p);}
        if(now_sec()-start>K3_WAIT_TIMEOUT)return-1;}
    runner_drain_mrq();return 0;}
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
            for(int i=0;i<50;++i){runner_drain_mrq();tp_ar_flag_inval(go);
                if(*go>=token)break;usleep(2000);}
            if(now_sec()-start>K3_WAIT_TIMEOUT){fprintf(stderr,"k3_ep_runner rank %d: barrier release timeout\n",g_rank);exit(3);}
        }while(*go<token);
    }
}

static int load_manifest(const char *path,k3_entry *entries,int cap,uint32_t *crc,int *has_crc){
    FILE *f=fopen(path,"r");if(!f){fprintf(stderr,"k3_ep_runner: open manifest '%s': %s\n",path,strerror(errno));return-1;}
    char line[1024];int n=0;*crc=0;*has_crc=0;
    while(fgets(line,sizeof line,f)){if(line[0]=='#'){unsigned value;
            if(!strncmp(line,"# K3EXPERTV2 ",13)){
                if(sscanf(line,"# K3EXPERTV2 layer=%*u expert=%*u tensors=%*u blob_bytes=%*u crc32=%x",&value)!=1){
                    fprintf(stderr,"k3_ep_runner: malformed V2 header in '%s'\n",path);fclose(f);return-1;}
                *crc=value;*has_crc=1;}
            continue;}k3_entry e;int nd;unsigned long long off,bytes,rows,cols;
        if(sscanf(line,"%llu %llu %15s %d %llu %llu %511s",&off,&bytes,e.dtype,&nd,&rows,&cols,e.name)!=7||nd!=2||n>=cap){
            fprintf(stderr,"k3_ep_runner: malformed manifest '%s' near entry %d\n",path,n);fclose(f);return-1;}
        e.offset=off;e.nbytes=bytes;e.rows=rows;e.cols=cols;entries[n++]=e;
    }
    fclose(f);return n;
}
static uint32_t crc32_bytes(const uint8_t *data,size_t size){
    static uint32_t table[256];static int initialized;
    if(!initialized){for(unsigned i=0;i<256;++i){uint32_t c=i;for(int b=0;b<8;++b)c=(c>>1)^((c&1)?UINT32_C(0xedb88320):0);table[i]=c;}initialized=1;}
    uint32_t crc=UINT32_MAX;for(size_t i=0;i<size;++i)crc=(crc>>8)^table[(crc^data[i])&255];return crc^UINT32_MAX;
}
static uint32_t crc32_init(void){
    return UINT32_MAX;
}
static uint32_t crc32_update(uint32_t crc,const uint8_t *data,size_t size){
    static uint32_t table[256];static int initialized;
    if(!initialized){for(unsigned i=0;i<256;++i){uint32_t c=i;for(int b=0;b<8;++b)c=(c>>1)^((c&1)?UINT32_C(0xedb88320):0);table[i]=c;}initialized=1;}
    for(size_t i=0;i<size;++i)crc=(crc>>8)^table[(crc^data[i])&255];
    return crc;
}
static uint32_t crc32_finalize(uint32_t crc){
    return crc^UINT32_MAX;
}
static int k3_read_all(FILE *f,void *dst,size_t n){
    uint8_t *p=(uint8_t*)dst;
    while(n){
        size_t nread=fread(p,1,n,f);
        if(nread==0){
            if(ferror(f))return errno?errno:EIO;
            return EOF;
        }
        p+=nread;n-=nread;
    }
    return 0;
}
static int k3_write_all(FILE *f,const void *src,size_t n){
    const uint8_t *p=(const uint8_t*)src;
    while(n){
        size_t nw=fwrite(p,1,n,f);
        if(nw==0)return errno?errno:EIO;
        p+=nw;n-=nw;
    }
    return 0;
}
static int k3_make_cache_path(const k3_options *o,char *out,size_t out_cap,const char *base){
    if(!base||!*base){errno=EINVAL;return -1;}
    if((size_t)snprintf(out,out_cap,"%s/k3_ep_cache_l%03d_%03d_n%03d_t%03d_g%03d_r%03d.bin",
            base,o->layer,o->layers,g_world_nodes,o->threads,g_group,g_world_rank)<out_cap) return 0;
    return -1;
}
static int k3_cache_ensure_dir(const char *path){
    if(!path||!*path)return -1;
    struct stat st;
    if(!stat(path,&st)){
        if(S_ISDIR(st.st_mode))return 0;
        errno=ENOTDIR;return -1;
    }
    if(errno!=ENOENT)return errno?errno:EIO;
    if(mkdir(path,0755)==-1&&errno!=EEXIST)return errno?errno:EIO;
    return 0;
}
static int k3_cache_load(const k3_options *o,int local_heads,int cache_tokens,size_t kda_elems,
    size_t mla_key_elems,size_t mla_value_elems,int cache_element_bytes,int *tokens_out,
    float *latent,float *kda_state,void *mla_keys,void *mla_values){
    if(!o||!o->cache_load||!latent||(kda_elems&&!kda_state)||(mla_key_elems&&!mla_keys)||(mla_value_elems&&!mla_values))
        return EINVAL;
    char base_path[1024];int rc=k3_make_cache_path(o,base_path,sizeof base_path,o->cache_load);
    if(rc) return errno?errno:EINVAL;
    FILE *f=fopen(base_path,"rb");if(!f)return errno?errno:EIO;
    k3_cache_header h;
    rc=k3_read_all(f,&h,sizeof h);
    if(!rc&&h.header_bytes>sizeof h)rc=EINVAL;
    if(!rc&&memcmp(h.magic,k3_cache_magic,sizeof h.magic)!=0)rc=EINVAL;
    if(!rc&&h.version!=K3_CACHE_VERSION)rc=EINVAL;
    if(!rc&&h.layer!= (uint32_t)o->layer)rc=EINVAL;
    if(!rc&&h.layers!=(uint32_t)o->layers)rc=EINVAL;
    if(!rc&&h.local_heads!=(uint32_t)local_heads)rc=EINVAL;
    uint32_t file_cache_tokens=h.cache_tokens?(uint32_t)h.cache_tokens:(uint32_t)cache_tokens;
    size_t file_mla_key_elems=0,file_mla_value_elems=0,file_mla_layers=0;
    if(!rc&&cache_tokens>0&&file_cache_tokens>(uint32_t)cache_tokens)rc=EINVAL;
    if(!rc&&h.cache_bf16!=(uint32_t)o->mla_cache_bf16)rc=EINVAL;
    if(!rc&&h.kda_elems!=kda_elems)rc=EINVAL;
    /* A serialized prefix may have a smaller cache capacity than the new
     * destination.  Validate its compact layout, then expand each head's
     * token stride while restoring it below. */
    if(!rc){
        size_t key_unit=(size_t)file_cache_tokens*(size_t)local_heads*192;
        size_t value_unit=(size_t)file_cache_tokens*(size_t)local_heads*K3_HEAD_DIM;
        if(!key_unit||!value_unit||h.mla_key_elems%key_unit||h.mla_value_elems%value_unit)
            rc=EINVAL;
        else{
            file_mla_key_elems=(size_t)h.mla_key_elems;
            file_mla_value_elems=(size_t)h.mla_value_elems;
            file_mla_layers=file_mla_key_elems/key_unit;
            if(file_mla_value_elems/value_unit!=file_mla_layers||
                    file_mla_value_elems!=file_mla_layers*value_unit||
                    mla_key_elems!=file_mla_layers*(size_t)cache_tokens*(size_t)local_heads*192||
                    mla_value_elems!=file_mla_layers*(size_t)cache_tokens*(size_t)local_heads*K3_HEAD_DIM)
                rc=EINVAL;
        }
    }
    if(!rc&&h.world_nodes!=(uint32_t)g_world_nodes)rc=EINVAL;
    if(!rc&&h.world_rank!=(uint32_t)g_world_rank)rc=EINVAL;
    if(!rc&&h.saved_tokens>(uint32_t)cache_tokens)rc=EINVAL;
    if(!rc&&fseek(f,(long)h.header_bytes,SEEK_SET))rc=errno?errno:EIO;
    void *file_mla_keys=file_mla_key_elems?malloc(file_mla_key_elems*(size_t)cache_element_bytes):NULL;
    void *file_mla_values=file_mla_value_elems?malloc(file_mla_value_elems*(size_t)cache_element_bytes):NULL;
    if(!rc&&((file_mla_key_elems&&!file_mla_keys)||(file_mla_value_elems&&!file_mla_values)))rc=ENOMEM;
    uint32_t payload_crc=crc32_init();
    if(!rc)rc=k3_read_all(f,latent,K3_LATENT*sizeof(float)),payload_crc=crc32_update(payload_crc,(uint8_t*)latent,K3_LATENT*sizeof(float));
    if(!rc&&kda_elems)rc=k3_read_all(f,kda_state,kda_elems*sizeof(float)),payload_crc=crc32_update(payload_crc,(uint8_t*)kda_state,kda_elems*sizeof(float));
    if(!rc&&file_mla_key_elems)rc=k3_read_all(f,file_mla_keys,file_mla_key_elems*cache_element_bytes),payload_crc=crc32_update(payload_crc,(uint8_t*)file_mla_keys,file_mla_key_elems*cache_element_bytes);
    if(!rc&&file_mla_value_elems)rc=k3_read_all(f,file_mla_values,file_mla_value_elems*cache_element_bytes),payload_crc=crc32_update(payload_crc,(uint8_t*)file_mla_values,file_mla_value_elems*cache_element_bytes);
    if(!rc&&crc32_finalize(payload_crc)!=h.payload_crc)rc=EIO;
    if(!rc&&file_mla_layers&&file_cache_tokens!=(uint32_t)cache_tokens){
        size_t src_head_stride=(size_t)file_cache_tokens;
        size_t dst_head_stride=(size_t)cache_tokens;
        size_t key_bytes=192*(size_t)cache_element_bytes,value_bytes=K3_HEAD_DIM*(size_t)cache_element_bytes;
        for(size_t layer=0;layer<file_mla_layers;++layer)for(int head=0;head<local_heads;++head){
            const uint8_t *src_k=(const uint8_t*)file_mla_keys+(layer*(size_t)local_heads+(size_t)head)*src_head_stride*key_bytes;
            uint8_t *dst_k=(uint8_t*)mla_keys+(layer*(size_t)local_heads+(size_t)head)*dst_head_stride*key_bytes;
            const uint8_t *src_v=(const uint8_t*)file_mla_values+(layer*(size_t)local_heads+(size_t)head)*src_head_stride*value_bytes;
            uint8_t *dst_v=(uint8_t*)mla_values+(layer*(size_t)local_heads+(size_t)head)*dst_head_stride*value_bytes;
            memcpy(dst_k,src_k,src_head_stride*key_bytes);memcpy(dst_v,src_v,src_head_stride*value_bytes);
        }
    }else if(!rc){
        if(file_mla_key_elems)memcpy(mla_keys,file_mla_keys,file_mla_key_elems*(size_t)cache_element_bytes);
        if(file_mla_value_elems)memcpy(mla_values,file_mla_values,file_mla_value_elems*(size_t)cache_element_bytes);
    }
    free(file_mla_keys);free(file_mla_values);
    if(!rc&&tokens_out)*tokens_out=(int)h.saved_tokens;
    if(fclose(f))rc=rc?rc:EIO;
    /* Keep a rejected/corrupt artifact for diagnosis and a possible retry.
     * Saves are published atomically through rename(), so load failure does
     * not need destructive cleanup here. */
    return rc;
}

static const char *k3_cache_load_reason(int rc){
    if(rc==EINVAL)return "cache-incompatible";
    if(rc==ENOENT)return "cache-missing";
    if(rc==EOF||rc==EIO||rc==EPIPE)return "cache-corrupt";
    return "cache-io";
}
static int k3_cache_save(const k3_options *o,int local_heads,int cache_tokens,size_t kda_elems,
    size_t mla_key_elems,size_t mla_value_elems,int cache_element_bytes,int saved_tokens,
    float *latent,float *kda_state,void *mla_keys,void *mla_values){
    if(!o||!o->cache_save||!latent||(kda_elems&&!kda_state)||(mla_key_elems&&!mla_keys)||(mla_value_elems&&!mla_values))
        return EINVAL;
    if(saved_tokens<0||saved_tokens>cache_tokens) return EINVAL;
    int rc=k3_cache_ensure_dir(o->cache_save);
    if(rc)return rc;
    char path[1024],tmp[1088],tmp_final[1024];
    rc=k3_make_cache_path(o,path,sizeof path,o->cache_save);
    if(rc) return errno?errno:EINVAL;
    if((size_t)snprintf(tmp,sizeof tmp,"%s.tmp.%ld",path,(long)getpid())>=sizeof tmp)return EINVAL;
    if((size_t)snprintf(tmp_final,sizeof tmp_final,"%s",path)>=sizeof tmp_final)return EINVAL;
    k3_cache_header h={.magic={'K','3','C','A','C','H','E','1'},.version=K3_CACHE_VERSION,.header_bytes=sizeof(k3_cache_header),
        .layer=(uint32_t)o->layer,.layers=(uint32_t)o->layers,.local_heads=(uint32_t)local_heads,
        .kda_layers=0,.mla_layers=0,.cache_tokens=(uint32_t)cache_tokens,.cache_bf16=(uint32_t)o->mla_cache_bf16,
        .kda_elems=kda_elems,.mla_key_elems=mla_key_elems,.mla_value_elems=mla_value_elems,.payload_crc=0,
        .saved_tokens=(uint32_t)(saved_tokens<0?0:saved_tokens),
        .world_rank=(uint32_t)g_world_rank,.world_nodes=(uint32_t)g_world_nodes};
    FILE *f=fopen(tmp,"wb");if(!f)return errno?errno:EIO;
    rc=k3_write_all(f,&h,sizeof h);
    uint32_t payload_crc=crc32_init();
    if(!rc)rc=k3_write_all(f,latent,K3_LATENT*sizeof(float)),payload_crc=crc32_update(payload_crc,(uint8_t*)latent,K3_LATENT*sizeof(float));
    if(!rc&&kda_elems)rc=k3_write_all(f,kda_state,kda_elems*sizeof(float)),payload_crc=crc32_update(payload_crc,(uint8_t*)kda_state,kda_elems*sizeof(float));
    if(!rc&&mla_key_elems)rc=k3_write_all(f,mla_keys,mla_key_elems*cache_element_bytes),payload_crc=crc32_update(payload_crc,(uint8_t*)mla_keys,mla_key_elems*cache_element_bytes);
    if(!rc&&mla_value_elems)rc=k3_write_all(f,mla_values,mla_value_elems*cache_element_bytes),payload_crc=crc32_update(payload_crc,(uint8_t*)mla_values,mla_value_elems*cache_element_bytes);
    h.payload_crc=crc32_finalize(payload_crc);
    if(!rc)rc=fseeko(f,0,SEEK_SET)?(errno?errno:EIO):0;
    if(!rc)rc=k3_write_all(f,&h,sizeof h);
    if(!rc&&fflush(f))rc=errno?errno:EIO;
    if(!rc&&fsync(fileno(f))!=0)rc=errno?errno:EIO;
    if(fclose(f))rc=rc?rc:EIO;
    if(rc||rename(tmp,tmp_final)!=0){if(!rc&&errno)rc=errno;unlink(tmp);return rc?rc:EIO;}
    return 0;
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
    k3_entry es[8];uint32_t expected_crc;int has_crc;
    int ne=load_manifest(manifest,es,8,&expected_crc,&has_crc);if(ne!=6){fprintf(stderr,"k3_ep_runner: '%s' has %d entries, expected 6\n",manifest,ne);return-1;}
    size_t size=0;uint8_t *base=k3_pool_load_blob(pool,blob,&size);if(!base){fprintf(stderr,"%s\n",k3_pool_error(pool));return-1;}
    if(has_crc){uint32_t actual=crc32_bytes(base,size);if(actual!=expected_crc){
        fprintf(stderr,"k3_ep_runner: CRC32 mismatch for '%s': expected=%08x actual=%08x\n",blob,expected_crc,actual);
        k3_pool_free(pool,base);return-1;}}
    else{static int warned;if(!warned){fprintf(stderr,"k3_ep_runner rank %d: legacy stage has no CRC32; restage before production use\n",g_rank);warned=1;}}
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
static void write_status(const k3_options *o,const char *state,const char *reason,
        int tokens_completed,int last_layer,uint64_t collective_seq,
        double seconds,double checksum,size_t peak){
    char path[1024],tmp[1088];int n=snprintf(path,sizeof path,"%s/k3_rank%03d.status",o->status_dir,g_world_rank);
    if(n<0||(size_t)n>=sizeof path)return;
    n=snprintf(tmp,sizeof tmp,"%s.tmp.%ld",path,(long)getpid());if(n<0||(size_t)n>=sizeof tmp)return;
    FILE *f=fopen(tmp,"wx");if(!f){fprintf(stderr,"k3_ep_runner rank %d: create status '%s': %s\n",g_rank,tmp,strerror(errno));return;}
    size_t rss=0,hwm=0;(void)k3_process_memory_bytes(&rss,&hwm);
    int ok=fprintf(f,"rank=%d nodes=%d group=%d contexts=%d tp_rank=%d tp_nodes=%d mode=%s state=%s reason=%s layers=%d tokens=%d tokens_completed=%d last_layer=%d collective_seq=%lu seconds=%.9f checksum=%+.9e peak_bytes=%zu rss_bytes=%zu hwm_bytes=%zu mem_available_bytes=%zu\n",
        g_world_rank,g_world_nodes,g_group,g_contexts,g_rank,g_nodes,k3_mode_name(o->mode),state,reason?reason:"none",
        o->layers,o->tokens,tokens_completed,last_layer,(unsigned long)collective_seq,seconds,checksum,peak,
        rss,hwm,k3_mem_available_bytes())>=0;
    if(ok)ok=fflush(f)==0;if(ok)ok=fsync(fileno(f))==0;if(fclose(f))ok=0;
    if(ok)ok=rename(tmp,path)==0;
    if(!ok){int saved=errno?errno:EIO;unlink(tmp);fprintf(stderr,"k3_ep_runner rank %d: publish status '%s': %s\n",g_rank,path,strerror(saved));}
}

static void write_progress(const k3_options *o,int tokens_completed,double seconds,
        float latent_max,float state_max,float cache_max,float min_available_mib,
        float max_rss_mib,float max_hwm_mib,uint64_t seq){
    if(g_rank)return;char path[1024],tmp[1088];
    int n=g_contexts==1?snprintf(path,sizeof path,"%s/k3_progress.status",o->status_dir):
        snprintf(path,sizeof path,"%s/k3_progress_group%03d.status",o->status_dir,g_group);
    if(n<0||(size_t)n>=sizeof path)return;
    n=snprintf(tmp,sizeof tmp,"%s.tmp.%ld",path,(long)getpid());if(n<0||(size_t)n>=sizeof tmp)return;
    FILE*f=fopen(tmp,"w");if(!f)return;
    int ok=fprintf(f,"state=running tokens_completed=%d seconds=%.6f tokens_per_s=%.3f latent_max=%.6e kda_state_max=%.6e mla_cache_max=%.6e min_available_mib=%.1f max_rss_mib=%.2f max_hwm_mib=%.2f collective_seq=%lu\n",
        tokens_completed,seconds,seconds>0?tokens_completed/seconds:0.0,latent_max,state_max,cache_max,
        min_available_mib,max_rss_mib,max_hwm_mib,(unsigned long)seq)>=0;
    /* Progress is advisory and replaced frequently. A shared-filesystem fsync
     * here stalls rank 0 long enough for every peer to wait in the next
     * collective; final per-rank status remains fully fsync-durable. */
    if(fclose(f))ok=0;
    if(ok)ok=rename(tmp,path)==0;if(!ok)unlink(tmp);
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
    for(int h=0;h<local_heads;++h){int gh=first_head+h;size_t qbase=is_kda?(size_t)h*K3_HEAD_DIM:(size_t)h*192;
        for(size_t d=0;d<qstride;++d){
            float base=latent[(gh*131+(int)d*17+global_layer*29+token*7)%K3_LATENT];
            q[qbase+d]=base+0.0001f*(float)(d+1);
            k[qbase+d]=base*0.75f-0.00007f*(float)(d+1);
            if(d<K3_HEAD_DIM){v[(size_t)h*K3_HEAD_DIM+d]=latent[(gh*97+(int)d*11+token)%K3_LATENT]*0.5f;
                decay[(size_t)h*K3_HEAD_DIM+d]=0.995f;}}
        if(is_kda){k3_l2_normalize_sve(q+qbase,K3_HEAD_DIM,1e-6f);
            k3_l2_normalize_sve(k+qbase,K3_HEAD_DIM,1e-6f);}}
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
        float *kda_state,void *mla_keys,void *mla_values,int mla_cache_bf16,int mla_cache_int8,
        float *mla_key_scales,float *mla_value_scales,
        float *q,float *k,float *v,float *decay,float *attn_out,
        float *mla_scratch,float *mla_stats,float *cache_running_max,int threads){
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
        if(mla_cache_int8){int8_t *keys=mla_keys,*values=mla_values;
            for(int h=0;h<local_heads;++h){size_t base=(size_t)state_slot*local_heads+h;
                int8_t *kh=keys+(base*(size_t)cache_tokens+token)*192;
                int8_t *vh=values+(base*(size_t)cache_tokens+token)*K3_HEAD_DIM;
                float *ks=mla_key_scales+base*(size_t)cache_tokens+token;
                float *vs=mla_value_scales+base*(size_t)cache_tokens+token;
                float kamax=0.0f,vamax=0.0f;
                for(int d=0;d<192;++d)kamax=fmaxf(kamax,fabsf(k[(size_t)h*192+d]));
                for(int d=0;d<K3_HEAD_DIM;++d)vamax=fmaxf(vamax,fabsf(v[(size_t)h*K3_HEAD_DIM+d]));
                *ks=kamax>0.0f?kamax/127.0f:0.0f;*vs=vamax>0.0f?vamax/127.0f:0.0f;
                for(int d=0;d<192;++d){float x=k[(size_t)h*192+d];kh[d]=*ks?(int8_t)lrintf(fmaxf(-127.0f,fminf(127.0f,x/ *ks))):0;}
                for(int d=0;d<K3_HEAD_DIM;++d){float x=v[(size_t)h*K3_HEAD_DIM+d];vh[d]=*vs?(int8_t)lrintf(fmaxf(-127.0f,fminf(127.0f,x/ *vs))):0;}
                *cache_running_max=fmaxf(*cache_running_max,fmaxf(kamax,vamax));
            }
            const int8_t *layer_keys=keys+(size_t)state_slot*local_heads*cache_tokens*192;
            const int8_t *layer_values=values+(size_t)state_slot*local_heads*cache_tokens*K3_HEAD_DIM;
            const float *layer_key_scales=mla_key_scales+(size_t)state_slot*local_heads*cache_tokens;
            const float *layer_value_scales=mla_value_scales+(size_t)state_slot*local_heads*cache_tokens;
#pragma omp parallel for schedule(static)
            for(int h=0;h<local_heads;++h)k3_attention_i8_sve(attn_out+(size_t)h*K3_HEAD_DIM,
                q+(size_t)h*192,layer_keys+(size_t)h*cache_tokens*192,
                layer_key_scales+(size_t)h*cache_tokens,layer_values+(size_t)h*cache_tokens*K3_HEAD_DIM,
                layer_value_scales+(size_t)h*cache_tokens,token+1,192,K3_HEAD_DIM);
        }else if(mla_cache_bf16){uint16_t *keys=mla_keys,*values=mla_values;
            for(int h=0;h<local_heads;++h){size_t base=(size_t)state_slot*local_heads+h;
                uint16_t *kh=keys+(base*(size_t)cache_tokens+token)*192;
                uint16_t *vh=values+(base*(size_t)cache_tokens+token)*K3_HEAD_DIM;
                for(int d=0;d<192;++d){kh[d]=k3_f32_to_bf16_rne(k[(size_t)h*192+d]);
                    float a=fabsf(k3_bf16_to_f32(kh[d]));if(a>*cache_running_max)*cache_running_max=a;}
                for(int d=0;d<K3_HEAD_DIM;++d){vh[d]=k3_f32_to_bf16_rne(v[(size_t)h*K3_HEAD_DIM+d]);
                    float a=fabsf(k3_bf16_to_f32(vh[d]));if(a>*cache_running_max)*cache_running_max=a;}}
            const uint16_t *layer_keys=keys+(size_t)state_slot*local_heads*cache_tokens*192;
            const uint16_t *layer_values=values+(size_t)state_slot*local_heads*cache_tokens*K3_HEAD_DIM;
            if(token+1<128){
#pragma omp parallel for schedule(static)
                for(int h=0;h<local_heads;++h)k3_attention_bf16_sve(attn_out+(size_t)h*K3_HEAD_DIM,
                    q+(size_t)h*192,layer_keys+(size_t)h*cache_tokens*192,
                    layer_values+(size_t)h*cache_tokens*K3_HEAD_DIM,token+1,192,K3_HEAD_DIM);
            }else k3_attention_heads_parallel_bf16_sve(attn_out,q,layer_keys,layer_values,
                local_heads,token+1,cache_tokens,192,K3_HEAD_DIM,threads,mla_scratch,mla_stats);
        }else{float *keys=mla_keys,*values=mla_values;
            for(int h=0;h<local_heads;++h){size_t base=(size_t)state_slot*local_heads+h;
                float *kh=keys+(base*(size_t)cache_tokens+token)*192;
                float *vh=values+(base*(size_t)cache_tokens+token)*K3_HEAD_DIM;
                memcpy(kh,k+(size_t)h*192,192*sizeof(float));memcpy(vh,v+(size_t)h*K3_HEAD_DIM,K3_HEAD_DIM*sizeof(float));
                for(int d=0;d<192;++d){float a=fabsf(kh[d]);if(a>*cache_running_max)*cache_running_max=a;}
                for(int d=0;d<K3_HEAD_DIM;++d){float a=fabsf(vh[d]);if(a>*cache_running_max)*cache_running_max=a;}}
            const float *layer_keys=keys+(size_t)state_slot*local_heads*cache_tokens*192;
            const float *layer_values=values+(size_t)state_slot*local_heads*cache_tokens*K3_HEAD_DIM;
            if(token+1<128){
#pragma omp parallel for schedule(static)
                for(int h=0;h<local_heads;++h)k3_attention_sve(attn_out+(size_t)h*K3_HEAD_DIM,
                q+(size_t)h*192,layer_keys+(size_t)h*cache_tokens*192,
                layer_values+(size_t)h*cache_tokens*K3_HEAD_DIM,token+1,192,K3_HEAD_DIM);
            }else k3_attention_heads_parallel_sve(attn_out,q,layer_keys,layer_values,
                local_heads,token+1,cache_tokens,192,K3_HEAD_DIM,threads,mla_scratch,mla_stats);}
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
    /* Dynamic teams changed the fused 32-thread recurrence result on libfjomp.
     * Fixed team sizes are an accuracy invariant, not only a tuning choice. */
    omp_set_dynamic(0);
    cpu_set_t initial_affinity;CPU_ZERO(&initial_affinity);
    if(!sched_getaffinity(0,sizeof initial_affinity,&initial_affinity))
        for(int i=0;i<CPU_SETSIZE;++i)if(CPU_ISSET(i,&initial_affinity))g_spare_cpu=i;
    struct sigaction sa;memset(&sa,0,sizeof sa);sa.sa_handler=handle_stop_signal;
    sigemptyset(&sa.sa_mask);sigaction(SIGINT,&sa,NULL);sigaction(SIGTERM,&sa,NULL);
    omp_set_num_threads(opt.threads);
    static uint8_t topo[K3_RUNNER_MAX_NODES][TOFU_NCOORDS];int topo_n=read_topology(opt.topo_path,topo);
    if(topo_n<0)return 2;if(topo_n!=opt.nodes){fprintf(stderr,"k3_ep_runner: topology has %d ranks but --nodes is %d\n",topo_n,opt.nodes);return 2;}
    uint8_t mine[TOFU_NCOORDS]={0};int rc=utofu_query_my_coords(mine);if(rc!=UTOFU_SUCCESS){fprintf(stderr,"k3_ep_runner: utofu_query_my_coords rc=%d\n",rc);return 3;}
    g_world_nodes=topo_n;g_world_rank=-1;
    for(int r=0;r<g_world_nodes;++r)if(!memcmp(topo[r],mine,TOFU_NCOORDS)){g_world_rank=r;break;}
    if(g_world_rank<0){fprintf(stderr,"k3_ep_runner: local coordinates are absent from '%s'\n",opt.topo_path);return 3;}
    g_nodes=opt.tp_nodes;g_contexts=g_world_nodes/g_nodes;
    g_group=g_world_rank/g_nodes;g_rank=g_world_rank%g_nodes;
    int group_first=g_group*g_nodes;

    k3_pool pool;k3_pool_init(&pool,"k3-ep-runner");
    g_slot_send=DEMO_CACHE_LINE;g_slot_bar=DEMO_CACHE_LINE;g_send_off=0;g_bar_base=g_slot_send;
    size_t region_bytes=g_bar_base+(size_t)(g_nodes+1)*g_slot_bar;
    g_region=k3_pool_calloc(&pool,1,region_bytes);if(!g_region){fprintf(stderr,"%s\n",k3_pool_error(&pool));k3_pool_destroy(&pool);return 2;}
    /* RDMA-polled slots must be clean before registration.  This prevents a
     * later dc civac from writing a dirty startup zero over an arrived Put. */
    for(size_t off=0;off<region_bytes;off+=DEMO_CACHE_LINE)
        __asm__ __volatile__("dc civac, %0"::"r"(g_region+off):"memory");
    __asm__ __volatile__("dsb sy":::"memory");

    utofu_tni_id_t *tnis=NULL;size_t ntni=0;rc=utofu_get_onesided_tnis(&tnis,&ntni);
    if(rc!=UTOFU_SUCCESS||ntni<1){fprintf(stderr,"k3_ep_runner rank %d: no one-sided TNI (rc=%d count=%zu)\n",g_rank,rc,ntni);k3_pool_destroy(&pool);return 3;}
    rc=utofu_create_vcq_with_cmp_id(tnis[0],DEMO_CMP_ID,0,&g_vcq);if(rc!=UTOFU_SUCCESS){fprintf(stderr,"k3_ep_runner rank %d: create VCQ rc=%d\n",g_rank,rc);free(tnis);k3_pool_destroy(&pool);return 3;}
    utofu_vcq_id_t self;rc=utofu_query_vcq_id(g_vcq,&self);if(rc!=UTOFU_SUCCESS){fprintf(stderr,"k3_ep_runner rank %d: query VCQ rc=%d\n",g_rank,rc);return 3;}
    rc=utofu_reg_mem_with_stag(g_vcq,g_region,region_bytes,K3_RUN_STAG,0,&g_base);if(rc!=UTOFU_SUCCESS){fprintf(stderr,"k3_ep_runner rank %d: register barrier rc=%d\n",g_rank,rc);return 3;}
    for(int r=0;r<g_nodes;++r){if(r==g_rank){g_peer_vcq[r]=self;g_peer_base[r]=g_base;continue;}
        rc=utofu_construct_vcq_id(topo[group_first+r],tnis[0],DEMO_CQ_ID,DEMO_CMP_ID,&g_peer_vcq[r]);
        if(rc==UTOFU_SUCCESS)utofu_set_vcq_id_path(&g_peer_vcq[r],NULL);
        if(rc==UTOFU_SUCCESS)rc=utofu_query_stadd(g_peer_vcq[r],K3_RUN_STAG,&g_peer_base[r]);
        if(rc!=UTOFU_SUCCESS){fprintf(stderr,"k3_ep_runner rank %d: peer %d bootstrap rc=%d\n",g_rank,r,rc);return 3;}}
    free(tnis);runner_barrier();
    tp_comm_config comm_config={.robust=opt.comm_robust,.poll_spins=opt.comm_poll_spins,.ack=opt.comm_ack,
        .deterministic=opt.comm_deterministic,
        .a2a_max=8192,.ack_retx=64,.ack_rtt=0.001,.timeout=120.0};
    if(opt.profile&&getenv("K3_DEBUG_COMM_DROP_N")){
        comm_config.drop_n=strtoul(getenv("K3_DEBUG_COMM_DROP_N"),NULL,10);
        if(getenv("K3_DEBUG_COMM_TIMEOUT_MS"))comm_config.timeout=atof(getenv("K3_DEBUG_COMM_TIMEOUT_MS"))*1e-3;}
    int row_nodes=opt.ar_groups?g_nodes/opt.ar_groups:g_nodes;
    int col_nodes=opt.ar_groups?opt.ar_groups:0;
    size_t comm_region_size=tp_comm_region_size(row_nodes,K3_RUN_REDUCE_FLOATS,&comm_config);
    size_t col_region_size=col_nodes?tp_comm_region_size(col_nodes,K3_RUN_REDUCE_FLOATS,&comm_config):0;
    void *comm_region=k3_pool_alloc(&pool,comm_region_size);
    void *col_region=col_region_size?k3_pool_alloc(&pool,col_region_size):NULL;
    if(!comm_region||(col_region_size&&!col_region)){
        fprintf(stderr,"k3_ep_runner rank %d: %s\n",g_rank,k3_pool_error(&pool));return 3;}
    k3_runner_comm comm;memset(&comm,0,sizeof comm);comm.groups=opt.ar_groups;
    int comm_init_rc=opt.ar_groups?
        tp_comm_init_2d_external(&comm.row,&comm.col,g_vcq,g_peer_vcq,g_rank,g_nodes,
            opt.ar_groups,K3_RUN_REDUCE_FLOATS,runner_barrier,&comm_config,
            comm_region,comm_region_size,col_region,col_region_size):
        tp_comm_init_external(&comm.row,g_vcq,g_peer_vcq,g_rank,g_nodes,
            K3_RUN_REDUCE_FLOATS,runner_barrier,&comm_config,comm_region,comm_region_size);
    if(comm_init_rc){
        fprintf(stderr,"k3_ep_runner rank %d: all-reduce initialization failed\n",g_rank);
        utofu_dereg_mem(g_vcq,g_base,0);utofu_free_vcq(g_vcq);k3_pool_destroy(&pool);return 3;}
    runner_barrier();

    int local=partition_count(96,g_rank,g_nodes)*K3_EXPERT_TP_BLOCK,ready=1;
    k3_loaded_expert experts[K3_SELECTED];memset(experts,0,sizeof experts);
    int use_real_experts=opt.mode==K3_MODE_REAL||opt.mode==K3_MODE_HYBRID;
    for(int e=0;e<K3_SELECTED;++e){int bad=use_real_experts?
        load_real_expert(&pool,opt.stage_dir,opt.layer,e,local,&experts[e]):
        make_dummy_expert(&pool,g_rank,e,local,&experts[e]);if(bad){ready=0;break;}}
    float ready_sum=(float)ready;int collective_rc=runner_allreduce_sum(&comm,&ready_sum,1);
    if(collective_rc){fprintf(stderr,"k3_ep_runner rank %d: readiness collective failed: %s\n",g_rank,runner_comm_error(&comm));
        write_status(&opt,"comm-failed","weight-readiness",0,-1,runner_comm_seq(&comm),0,0,pool.peak_active_bytes);
        runner_comm_free(&comm);utofu_dereg_mem(g_vcq,g_base,0);utofu_free_vcq(g_vcq);k3_pool_destroy(&pool);return 3;}
    if((int)lrintf(ready_sum)!=g_nodes){
        if(g_rank==0)fprintf(stderr,"k3_ep_runner: distributed weight readiness failed (%d/%d ranks ready)\n",(int)lrintf(ready_sum),g_nodes);
        write_status(&opt,"load-failed","weight-load",0,-1,runner_comm_seq(&comm),0,0,pool.peak_active_bytes);runner_barrier();runner_comm_free(&comm);
        utofu_dereg_mem(g_vcq,g_base,0);utofu_free_vcq(g_vcq);k3_pool_destroy(&pool);return 4;}

    k3_mxfp4_matrix w1[K3_SELECTED],w2[K3_SELECTED],w3[K3_SELECTED];float route[K3_SELECTED];
    for(int e=0;e<K3_SELECTED;++e){w1[e]=experts[e].w1;w2[e]=experts[e].w2;w3[e]=experts[e].w3;route[e]=1.0f/K3_SELECTED;}
    float *latent=k3_pool_alloc(&pool,K3_LATENT*sizeof(float));
    float *partial=k3_pool_alloc(&pool,K3_LATENT*sizeof(float));
    float *shared=k3_pool_calloc(&pool,K3_HIDDEN,sizeof(float));
    float *reduce=k3_pool_alloc(&pool,K3_RUN_REDUCE_FLOATS*sizeof(float));
    float *gate=k3_pool_alloc(&pool,(size_t)K3_SELECTED*local*sizeof(float));
    float *up=k3_pool_alloc(&pool,(size_t)K3_SELECTED*local*sizeof(float));
    int local_heads=partition_count(96,g_rank,g_nodes),first_head=partition_first(96,g_rank,g_nodes);
    /* Hybrid mode stages one real expert layer but walks the complete synthetic
     * 0..92 schedule, so its source layer and schedule layer are intentionally
     * different.  This keeps the 93-layer throughput attachment explicit. */
    int schedule_layer=opt.mode==K3_MODE_HYBRID?0:opt.layer;
    int state_slot[93],kda_layers=0,mla_layers=0;
    for(int l=0;l<opt.layers;++l)state_slot[l]=layer_is_kda(schedule_layer+l)?kda_layers++:mla_layers++;
    size_t kda_elems=(size_t)kda_layers*local_heads*K3_HEAD_DIM*K3_HEAD_DIM;
    size_t mla_key_elems=(size_t)mla_layers*local_heads*opt.tokens*192;
    size_t mla_value_elems=(size_t)mla_layers*local_heads*opt.tokens*K3_HEAD_DIM;
    size_t cache_element_bytes=opt.mla_cache_int8?sizeof(int8_t):(opt.mla_cache_bf16?sizeof(uint16_t):sizeof(float));
    size_t mla_scale_elems=(size_t)mla_layers*local_heads*opt.tokens;
    size_t state_bytes=kda_elems*sizeof(float)+(mla_key_elems+mla_value_elems)*cache_element_bytes+
        (opt.mla_cache_int8?2*mla_scale_elems*sizeof(float):0);
    size_t available=k3_mem_available_bytes(),reserve=(size_t)6<<30;
    int capacity_ok=available>reserve&&state_bytes<=available-reserve;
    if(!capacity_ok)fprintf(stderr,"k3_ep_runner rank %d: state preflight failed: need=%zu MiB MemAvailable=%zu MiB reserve=%zu MiB\n",
        g_rank,state_bytes>>20,available>>20,reserve>>20);
    float *kda_state=capacity_ok&&kda_elems?k3_pool_calloc(&pool,kda_elems,sizeof(float)):NULL;
    void *mla_keys=capacity_ok&&mla_key_elems?k3_pool_calloc(&pool,mla_key_elems,cache_element_bytes):NULL;
    void *mla_values=capacity_ok&&mla_value_elems?k3_pool_calloc(&pool,mla_value_elems,cache_element_bytes):NULL;
    float *mla_key_scales=capacity_ok&&opt.mla_cache_int8&&mla_scale_elems?k3_pool_calloc(&pool,mla_scale_elems,sizeof(float)):NULL;
    float *mla_value_scales=capacity_ok&&opt.mla_cache_int8&&mla_scale_elems?k3_pool_calloc(&pool,mla_scale_elems,sizeof(float)):NULL;
    float *q=k3_pool_alloc(&pool,(size_t)local_heads*192*sizeof(float));
    float *k=k3_pool_alloc(&pool,(size_t)local_heads*192*sizeof(float));
    float *v=k3_pool_alloc(&pool,(size_t)local_heads*K3_HEAD_DIM*sizeof(float));
    float *decay=k3_pool_alloc(&pool,(size_t)local_heads*K3_HEAD_DIM*sizeof(float));
    float *attn_out=k3_pool_alloc(&pool,(size_t)local_heads*K3_HEAD_DIM*sizeof(float));
    float *mla_scratch=k3_pool_alloc(&pool,(size_t)local_heads*opt.threads*K3_HEAD_DIM*sizeof(float));
    float *mla_stats=k3_pool_alloc(&pool,((size_t)local_heads*opt.threads*2+(size_t)local_heads*2)*sizeof(float));
    size_t prefetch_bytes=(size_t)opt.prefetch_mib<<20;
    uint8_t *prefetch_weights=prefetch_bytes?k3_pool_alloc(&pool,prefetch_bytes):NULL;
    ready=capacity_ok&&latent&&partial&&shared&&reduce&&gate&&up&&(!kda_elems||kda_state)&&
        (!mla_key_elems||mla_keys)&&(!mla_value_elems||mla_values)&&(!opt.mla_cache_int8||!mla_scale_elems||mla_key_scales)&&
        (!opt.mla_cache_int8||!mla_scale_elems||mla_value_scales)&&q&&k&&v&&decay&&attn_out&&mla_scratch&&mla_stats&&
        (!prefetch_bytes||prefetch_weights);
    ready_sum=(float)ready;collective_rc=runner_allreduce_sum(&comm,&ready_sum,1);
    if(collective_rc){fprintf(stderr,"k3_ep_runner rank %d: allocation collective failed: %s\n",g_rank,runner_comm_error(&comm));
        write_status(&opt,"comm-failed","allocation-readiness",0,-1,runner_comm_seq(&comm),0,0,pool.peak_active_bytes);
        runner_comm_free(&comm);utofu_dereg_mem(g_vcq,g_base,0);utofu_free_vcq(g_vcq);k3_pool_destroy(&pool);return 3;}
    if((int)lrintf(ready_sum)!=g_nodes){if(!ready&&pool.error[0])fprintf(stderr,"%s\n",k3_pool_error(&pool));
        if(g_rank==0)fprintf(stderr,"k3_ep_runner: distributed scratch allocation failed (%d/%d ranks ready)\n",(int)lrintf(ready_sum),g_nodes);
        write_status(&opt,"alloc-failed","state-allocation",0,-1,runner_comm_seq(&comm),0,0,pool.peak_active_bytes);runner_barrier();runner_comm_free(&comm);
        utofu_dereg_mem(g_vcq,g_base,0);utofu_free_vcq(g_vcq);k3_pool_destroy(&pool);return 4;}
    if(prefetch_bytes){
#pragma omp parallel for schedule(static)
        for(size_t i=0;i<prefetch_bytes;i+=256)prefetch_weights[i]=(uint8_t)(i/256+g_rank);
    }
    k3_async_reduce async_reduce;memset(&async_reduce,0,sizeof async_reduce);
    int async_rc=prefetch_bytes?runner_async_init(&async_reduce,&comm):0;
    int async_ok=!async_rc;if(async_rc)fprintf(stderr,"k3_ep_runner rank %d: async worker: %s\n",g_rank,strerror(async_rc));
    ready_sum=(float)async_ok;collective_rc=runner_allreduce_sum(&comm,&ready_sum,1);
    if(collective_rc||(int)lrintf(ready_sum)!=g_nodes){
        fprintf(stderr,"k3_ep_runner rank %d: post-allocation readiness failed (%d/%d ready)\n",
            g_rank,(int)lrintf(ready_sum),g_nodes);runner_async_destroy(&async_reduce);runner_comm_free(&comm);
        utofu_dereg_mem(g_vcq,g_base,0);utofu_free_vcq(g_vcq);k3_pool_destroy(&pool);return 3;}
    /* Pay OpenMP worker creation before the timed token loop. */
    int warm_threads=1;omp_set_num_threads(opt.threads);
#pragma omp parallel
    {
#pragma omp master
        warm_threads=omp_get_num_threads();
    }
    if(warm_threads!=opt.threads)fprintf(stderr,
        "k3_ep_runner rank %d: requested %d OpenMP workers but runtime created %d\n",
        g_rank,opt.threads,warm_threads);
    if(!opt.cache_load){
        for(int i=0;i<K3_LATENT;++i)
            latent[i]=sinf((float)(i+1+g_group*104729)*0.001f)*0.125f;
    }
    int start_token=0;
    float cache_running_max=0.0f;
    if(opt.cache_load){
        int cache_tokens=-1,cache_ready=1,cache_load_rc=0,cache_collective_ok=1;
        cache_load_rc=k3_cache_load(&opt,local_heads,opt.tokens,kda_elems,mla_key_elems,mla_value_elems,cache_element_bytes,&cache_tokens,
            latent,kda_state,mla_keys,mla_values);
        if(cache_load_rc||cache_tokens<0||cache_tokens>opt.tokens)cache_ready=0;
        /* Every rank must join this reduction, even when its local shard
         * failed.  Otherwise healthy ranks can enter the token reduction
         * while the failed rank exits, which is a collective deadlock. */
        /* The reduction must carry a positive errno-like value.  Short
         * reads return EOF (-1), and reducing raw errors with max would let
         * healthy ranks' zero hide that failure and enter token collectives.
         * Normalize negative read errors to EIO so every rank takes the same
         * diagnostic path. */
        int normalized_load_rc=cache_load_rc;
        if(normalized_load_rc<0)normalized_load_rc=EIO;
        float load_error=(float)normalized_load_rc;
        if(runner_allreduce_max(&comm,&load_error,1)){
            cache_collective_ok=0;
            cache_ready=0;
            cache_load_rc=EIO;
        }else if((int)lrintf(load_error)!=0){
            cache_load_rc=(int)lrintf(load_error);
            cache_ready=0;
        }
        if(cache_ready){
            float ready_sum=(float)cache_ready;
            float token_sum=(float)cache_tokens,token_max=(float)cache_tokens;
            if(runner_allreduce_sum(&comm,&ready_sum,1)||runner_allreduce_sum(&comm,&token_sum,1)||
                    runner_allreduce_max(&comm,&token_max,1)||(int)lrintf(token_sum)!=(int)lrintf(token_max*(float)g_nodes)){
                cache_collective_ok=0;
                cache_ready=0;
                cache_load_rc=EIO;
            }
            if(cache_ready&&(int)lrintf(ready_sum)!=g_nodes)cache_ready=0;
        }
        if(cache_ready){
            for(size_t i=0;i<kda_elems;++i){float a=fabsf(kda_state[i]);if(a>cache_running_max)cache_running_max=a;}
            if(opt.mla_cache_bf16){
                uint16_t *keys=mla_keys,*values=mla_values;
                for(size_t i=0;i<mla_key_elems;++i){float a=fabsf(k3_bf16_to_f32(keys[i]));if(a>cache_running_max)cache_running_max=a;}
                for(size_t i=0;i<mla_value_elems;++i){float a=fabsf(k3_bf16_to_f32(values[i]));if(a>cache_running_max)cache_running_max=a;}
            }else{
                float *keys=mla_keys,*values=mla_values;
                for(size_t i=0;i<mla_key_elems;++i){float a=fabsf(keys[i]);if(a>cache_running_max)cache_running_max=a;}
                for(size_t i=0;i<mla_value_elems;++i){float a=fabsf(values[i]);if(a>cache_running_max)cache_running_max=a;}
            }
            start_token=cache_tokens;
        }
        if(!cache_ready){
            const char *reason=cache_collective_ok?k3_cache_load_reason(cache_load_rc):"cache-collective";
            if(g_rank==0)fprintf(stderr,"k3_ep_runner: cache load failed rc=%d reason=%s\n",
                cache_load_rc,reason);
            write_status(&opt,"load-failed",reason,0,-1,runner_comm_seq(&comm),0,0,pool.peak_active_bytes);
            if(cache_collective_ok)runner_barrier();
            runner_async_destroy(&async_reduce);runner_comm_free(&comm);
            utofu_dereg_mem(g_vcq,g_base,0);utofu_free_vcq(g_vcq);k3_pool_destroy(&pool);
            return cache_collective_ok?5:3;
        }
    }
    int debug_rank=-1,debug_token=-1,debug_layer=-1;
    if(opt.profile&&getenv("K3_DEBUG_NAN_RANK")){debug_rank=atoi(getenv("K3_DEBUG_NAN_RANK"));
        debug_token=getenv("K3_DEBUG_NAN_TOKEN")?atoi(getenv("K3_DEBUG_NAN_TOKEN")):-1;
        debug_layer=getenv("K3_DEBUG_NAN_LAYER")?atoi(getenv("K3_DEBUG_NAN_LAYER")):-1;}
    runner_barrier();double start=now_sec(),phase[6]={0},phase_max[6]={0};int finite=1,stopped=0,comm_failed=0;
    int stop_numeric=0,stop_signal=0,stop_memory=0,tokens_completed=0,last_layer=-1;
    uint64_t prefetch_sink=0;
    for(int token=start_token;token<opt.tokens&&!stopped;++token){
        for(int layer=0;layer<opt.layers;++layer){int global_layer=schedule_layer+layer;
            if(!finite||g_stop_signal){memset(reduce,0,K3_RUN_REDUCE_FLOATS*sizeof(float));
                reduce[K3_CONTROL_SIGNAL]=g_stop_signal?1.0f:0.0f;
                reduce[K3_CONTROL_NUMERIC]=finite?0.0f:1.0f;
                if(runner_allreduce_sum(&comm,reduce,K3_RUN_REDUCE_FLOATS)){
                    fprintf(stderr,"k3_ep_runner rank %d: stop collective failed: %s\n",g_rank,runner_comm_error(&comm));comm_failed=stopped=1;break;}
                stop_signal=reduce[K3_CONTROL_SIGNAL]>0;stop_numeric=reduce[K3_CONTROL_NUMERIC]>0;stopped=1;break;}
            int fused=opt.fuse_kda_expert&&global_layer>0&&layer_is_kda(global_layer);double pt=0;
            if(fused){double ft[2]={0};if(synthetic_kda_expert_partial(shared,partial,latent,
                    state_slot[layer],global_layer,token,local_heads,first_head,kda_state,q,k,v,decay,attn_out,
                    w1,w2,w3,route,gate,up,opt.fused_threads,opt.profile,ft))finite=0;
                if(opt.profile){profile_add(phase,phase_max,0,ft[0]);profile_add(phase,phase_max,2,ft[1]);}}
            else{pt=opt.profile?now_sec():0;
                synthetic_attention_partial(shared,latent,state_slot[layer],global_layer,token,opt.tokens,local_heads,first_head,
                    kda_state,mla_keys,mla_values,opt.mla_cache_bf16,opt.mla_cache_int8,mla_key_scales,mla_value_scales,
                    q,k,v,decay,attn_out,mla_scratch,mla_stats,
                    &cache_running_max,layer_is_kda(global_layer)?opt.kda_threads:opt.threads);
                if(opt.profile)profile_add(phase,phase_max,layer_is_kda(global_layer)?0:1,now_sec()-pt);
                pt=opt.profile?now_sec():0;
                if(global_layer==0)memset(partial,0,K3_LATENT*sizeof(float));
                else if(k3_expert_tp_forward_selected_mxfp4(partial,w1,w2,w3,route,K3_SELECTED,latent,gate,up,NULL,opt.threads))finite=0;
                if(opt.profile)profile_add(phase,phase_max,2,now_sec()-pt);}
            pt=opt.profile?now_sec():0;k3_moe_pack_reduce(reduce,partial,shared);
            reduce[K3_CONTROL_SIGNAL]=g_stop_signal?1.0f:0.0f;
            reduce[K3_CONTROL_NUMERIC]=finite?0.0f:1.0f;
            if(opt.profile)profile_add(phase,phase_max,3,now_sec()-pt);pt=opt.profile?now_sec():0;
            if(prefetch_bytes){runner_comm_set_robust(&comm,1);
                runner_async_submit(&async_reduce,reduce,K3_RUN_REDUCE_FLOATS);
                int pth=opt.prefetch_threads?opt.prefetch_threads:
                    (opt.threads<K3_RUN_PREFETCH_THREADS?opt.threads:K3_RUN_PREFETCH_THREADS);
                prefetch_sink+=k3_prefetch_weight_window(prefetch_weights,prefetch_bytes,pth);
                collective_rc=runner_async_wait(&async_reduce);
                runner_comm_set_robust(&comm,opt.comm_robust);
            }else collective_rc=runner_allreduce_sum(&comm,reduce,K3_RUN_REDUCE_FLOATS);
            if(collective_rc){
                fprintf(stderr,"k3_ep_runner rank %d: layer collective failed at token=%d layer=%d: %s\n",
                    g_rank,token,global_layer,runner_comm_error(&comm));comm_failed=stopped=1;break;}
            if(opt.profile)profile_add(phase,phase_max,4,now_sec()-pt);
            if(reduce[K3_CONTROL_SIGNAL]>0||reduce[K3_CONTROL_NUMERIC]>0){
                stop_signal=reduce[K3_CONTROL_SIGNAL]>0;stop_numeric=reduce[K3_CONTROL_NUMERIC]>0;stopped=1;break;}
            pt=opt.profile?now_sec():0;residual_update(latent,reduce);last_layer=global_layer;
            if(g_world_rank==debug_rank&&token==debug_token&&global_layer==debug_layer)latent[0]=NAN;
            for(int i=0;i<K3_LATENT;++i)finite&=isfinite(latent[i]);
            if(opt.profile)profile_add(phase,phase_max,5,now_sec()-pt);
        }
        if(!stopped&&finite)tokens_completed=token+1;
        int heartbeat=opt.heartbeat_tokens>0&&((token+1)%opt.heartbeat_tokens==0||token+1==opt.tokens);
        if(!stopped&&heartbeat){float latent_now=0,state_now=0;
            for(int i=0;i<K3_LATENT;++i){float a=fabsf(latent[i]);if(a>latent_now)latent_now=a;}
#pragma omp parallel for reduction(max:state_now)
            for(size_t i=0;i<kda_elems;++i){float a=fabsf(kda_state[i]);if(a>state_now)state_now=a;}
            size_t rss=0,hwm=0;(void)k3_process_memory_bytes(&rss,&hwm);
            float available_mib=(float)(k3_mem_available_bytes()/1048576.0);
            float hb[9]={g_stop_signal?1.0f:0.0f,finite?0.0f:1.0f,latent_now,state_now,
                cache_running_max,-available_mib,(float)(rss/1048576.0),(float)(hwm/1048576.0),
                opt.min_available_mib>0&&available_mib<(float)opt.min_available_mib?1.0f:0.0f};
            if(runner_allreduce_max(&comm,hb,9)){
                fprintf(stderr,"k3_ep_runner rank %d: heartbeat collective failed: %s\n",g_rank,runner_comm_error(&comm));comm_failed=stopped=1;break;}
            stop_signal=hb[0]>0;stop_numeric=hb[1]>0;stop_memory=hb[8]>0;
            if(stop_signal||stop_numeric||stop_memory)stopped=1;
            double elapsed=now_sec()-start;write_progress(&opt,tokens_completed,elapsed,hb[2],hb[3],hb[4],-hb[5],hb[6],hb[7],runner_comm_seq(&comm));
            if(g_rank==0){printf("K3_PROGRESS group=%d contexts=%d tokens=%d/%d elapsed_s=%.3f tokens_per_s=%.3f min_available_MiB=%.1f max_rss_MiB=%.2f max_hwm_MiB=%.2f latent_max=%.6e state_max=%.6e cache_max=%.6e collective_seq=%lu\n",
                g_group,g_contexts,tokens_completed,opt.tokens,elapsed,elapsed>0?tokens_completed/elapsed:0.0,-hb[5],hb[6],hb[7],hb[2],hb[3],hb[4],(unsigned long)runner_comm_seq(&comm));}
        }
    }
    if(!comm_failed)runner_barrier();double seconds=now_sec()-start,checksum=0,norm2=0;
    float latent_max=0,state_max=0,cache_max=0;
    for(int i=0;i<K3_LATENT;++i){float a=fabsf(latent[i]);checksum+=latent[i];norm2+=(double)latent[i]*latent[i];if(a>latent_max)latent_max=a;}
#pragma omp parallel for reduction(max:state_max)
    for(size_t i=0;i<kda_elems;++i){float a=fabsf(kda_state[i]);if(a>state_max)state_max=a;}
    if(opt.mla_cache_int8){int8_t *keys=mla_keys,*values=mla_values;
#pragma omp parallel for reduction(max:cache_max)
        for(size_t i=0;i<mla_key_elems;++i){float a=fabsf((float)keys[i])*mla_key_scales[i/192];if(a>cache_max)cache_max=a;}
#pragma omp parallel for reduction(max:cache_max)
        for(size_t i=0;i<mla_value_elems;++i){float a=fabsf((float)values[i])*mla_value_scales[i/K3_HEAD_DIM];if(a>cache_max)cache_max=a;}
    }else if(opt.mla_cache_bf16){uint16_t *keys=mla_keys,*values=mla_values;
#pragma omp parallel for reduction(max:cache_max)
        for(size_t i=0;i<mla_key_elems;++i){float a=fabsf(k3_bf16_to_f32(keys[i]));if(a>cache_max)cache_max=a;}
#pragma omp parallel for reduction(max:cache_max)
        for(size_t i=0;i<mla_value_elems;++i){float a=fabsf(k3_bf16_to_f32(values[i]));if(a>cache_max)cache_max=a;}
    }else{float *keys=mla_keys,*values=mla_values;
#pragma omp parallel for reduction(max:cache_max)
        for(size_t i=0;i<mla_key_elems;++i){float a=fabsf(keys[i]);if(a>cache_max)cache_max=a;}
#pragma omp parallel for reduction(max:cache_max)
        for(size_t i=0;i<mla_value_elems;++i){float a=fabsf(values[i]);if(a>cache_max)cache_max=a;}}
    runner_async_destroy(&async_reduce);
    size_t final_rss=0,final_hwm=0;(void)k3_process_memory_bytes(&final_rss,&final_hwm);
    float health[7]={latent_max,state_max,cache_max,-(float)tokens_completed,-(float)last_layer,
        (float)(final_rss/1048576.0),(float)(final_hwm/1048576.0)};
    /* The reduction bounds are FP32 by design.  Compare them with the same
     * rounded local value; comparing against the original FP64 accumulation
     * creates a false one-ulp failure at checksums around 371. */
    double checksum_local=(double)(float)checksum;
    float checksum_bounds[2]={(float)checksum_local,-(float)checksum_local};
    int cache_save_rc=0;
    if(opt.cache_save&&!comm_failed&&tokens_completed>0){
        cache_save_rc=k3_cache_save(&opt,local_heads,opt.tokens,kda_elems,mla_key_elems,mla_value_elems,cache_element_bytes,
            tokens_completed,latent,kda_state,mla_keys,mla_values);
        if(cache_save_rc&&g_rank==0)fprintf(stderr,"k3_ep_runner rank %d: cache save failed: %s\n",g_rank,strerror(cache_save_rc));
    }
    if(!comm_failed&&(runner_allreduce_max(&comm,health,7)||
            runner_allreduce_max(&comm,checksum_bounds,2))){
        fprintf(stderr,"k3_ep_runner rank %d: final health collective failed: %s\n",g_rank,runner_comm_error(&comm));comm_failed=1;}
    if(!comm_failed){tokens_completed=(int)lrintf(-health[3]);last_layer=(int)lrintf(-health[4]);}
    if(opt.profile&&!comm_failed)for(int i=0;i<6;++i){float x[2]={(float)phase[i],(float)phase_max[i]};
        if(runner_allreduce_max(&comm,x,2)){fprintf(stderr,"k3_ep_runner rank %d: profile collective failed: %s\n",g_rank,runner_comm_error(&comm));comm_failed=1;break;}phase[i]=x[0];phase_max[i]=x[1];}
    double checksum_min=-(double)checksum_bounds[1],checksum_max=(double)checksum_bounds[0];
    double disagreement=comm_failed?INFINITY:fmax(fabs(checksum_max-checksum_local),fabs(checksum_local-checksum_min));
    /* A non-deterministic FP32 all-reduce can use a different reduction tree
     * on different ranks.  The resulting error is bounded by the magnitude of
     * the reduced value; treating every non-zero low bit as corruption made
     * the 96-rank dummy gate reject healthy results (3e-5 at checksum 371).
     * Fixed-root mode remains strict for diagnostic/release runs. */
    double checksum_scale=fmax(1.0,fmax(fabs(checksum_max),fabs(checksum_local)));
    double disagreement_tol=opt.comm_deterministic?1e-6:fmax(1e-6,2e-7*checksum_scale);
    int rank_diverged=!comm_failed&&disagreement>=disagreement_tol;
    finite&=!comm_failed&&isfinite(checksum)&&isfinite(norm2);
    if(!finite||rank_diverged)stop_numeric=1;
    const char *final_state=comm_failed?"comm-failed":stop_memory||stop_signal?"stopped":stop_numeric?"numeric-failed":"pass";
    const char *final_reason=comm_failed?"collective-error":stop_memory?"memory-pressure":stop_signal?(g_stop_signal==SIGINT?"signal-int":g_stop_signal==SIGTERM?"signal-term":"signal-peer"):
        stop_numeric?(rank_diverged&&finite?"rank-divergence":"non-finite"):"complete";
    write_status(&opt,final_state,final_reason,tokens_completed,last_layer,runner_comm_seq(&comm),seconds,checksum,pool.peak_active_bytes);
    if(g_rank==0){double steps=(double)opt.layers*tokens_completed;
        int nkda=0;for(int l=0;l<opt.layers;++l)nkda+=layer_is_kda(schedule_layer+l);
        printf("K3_RUN mode=%s model_coverage=%s expert_layer=%d nodes=%d tp_nodes=%d contexts=%d group=%d local_channels=%d selected=%d layer_range=[%d,%d) KDA=%d MLA=%d tokens=%d threads=%d kda_threads=%d fused_team=%d fused_threads=%d mla_cache=%s min_available_mib=%d allreduce=%s ar_groups=%d comm_robust=%d comm_ack=%d comm_deterministic=%d comm_poll_spins=%d prefetch_mib=%d prefetch_threads=%d prefetch_checksum=%llu\n",
            k3_mode_name(opt.mode),opt.mode==K3_MODE_HYBRID?"one_real_expert_layer_repeated":opt.mode==K3_MODE_REAL?"one_real_expert_layer":"synthetic",opt.layer,g_world_nodes,g_nodes,g_contexts,g_group,local,K3_SELECTED,schedule_layer,schedule_layer+opt.layers,nkda,opt.layers-nkda,opt.tokens,opt.threads,opt.kda_threads,opt.fuse_kda_expert,opt.fused_threads,opt.mla_cache_int8?"int8":opt.mla_cache_bf16?"bf16":"fp32",opt.min_available_mib,
            opt.ar_groups?"hierarchical":"flat",opt.ar_groups,opt.comm_robust,opt.comm_ack,opt.comm_deterministic,opt.comm_poll_spins,opt.prefetch_mib,
            opt.prefetch_threads?opt.prefetch_threads:(opt.threads<K3_RUN_PREFETCH_THREADS?opt.threads:K3_RUN_PREFETCH_THREADS),(unsigned long long)prefetch_sink);
        printf("K3_RESULT status=%s reason=%s tokens_completed=%d wall_s=%.6f decode_tok_s=%.3f layer_steps_per_s=%.3f checksum=%+.9e l2=%.9e disagreement=%.3e disagreement_tol=%.3e peak_MiB=%.2f collective_seq=%lu\n",
            !comm_failed&&!stop_signal&&!stop_memory&&!stop_numeric?"PASS":comm_failed?"COMM-FAILED":stop_signal||stop_memory?"STOPPED":"FAIL",final_reason,tokens_completed,
            seconds,seconds>0?(double)tokens_completed/seconds:0.0,seconds>0?steps/seconds:0.0,checksum,sqrt(norm2),disagreement,disagreement_tol,pool.peak_active_bytes/1048576.0,(unsigned long)runner_comm_seq(&comm));
        printf("K3_HEALTH kda_slots=%d mla_slots=%d state_MiB=%.2f mla_cache_MiB=%.2f pool_peak_MiB=%.2f rss_max_MiB=%.2f hwm_max_MiB=%.2f latent_max=%.6e kda_state_max=%.6e mla_cache_max=%.6e\n",
            kda_layers,mla_layers,kda_elems*sizeof(float)/1048576.0,
            (mla_key_elems+mla_value_elems)*cache_element_bytes/1048576.0,
            pool.peak_active_bytes/1048576.0,health[5],health[6],health[0],health[1],health[2]);
        fflush(stdout);}
    if(opt.profile&&!comm_failed){const char *names[6]={"kda","mla","expert","pack","allreduce","residual"};double steps=(double)opt.layers*tokens_completed;if(steps<1)steps=1;
        if(g_rank==0){double sum=0;for(int i=0;i<6;++i)sum+=phase[i];
            printf("K3_PROFILE rank_max_ms_per_layer");for(int i=0;i<6;++i)printf(" %s=%.4f",names[i],phase[i]*1e3/steps);
            printf(" measured_upper=%.4f upper_pct=%.1f\n",sum*1e3/steps,seconds>0?100.0*sum/seconds:0.0);
            printf("K3_PROFILE_MAX rank_max_single_ms");for(int i=0;i<6;++i)printf(" %s=%.4f",names[i],phase_max[i]*1e3);
            printf("\n");
        fflush(stdout);}
    }
    if(!comm_failed)runner_barrier();runner_comm_free(&comm);utofu_dereg_mem(g_vcq,g_base,0);utofu_free_vcq(g_vcq);k3_pool_destroy(&pool);
    return comm_failed?3:stop_signal||stop_memory?5:stop_numeric?1:0;
}
