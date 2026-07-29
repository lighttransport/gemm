#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "k3_kernels.h"
#include "k3_dense.h"
#include "k3_runtime.h"
#include "ggml_dequant.h"

typedef struct {
    uint64_t offset, nbytes, shape[4];
    int ndims;
    char dtype[16], name[512];
} entry;

static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec+ts.tv_nsec*1e-9;
}

static int load_manifest(const char *path, entry *out, int cap) {
    FILE *f=fopen(path,"r"); if(!f)return -1;
    char line[1024]; int nent=0;
    while(fgets(line,sizeof(line),f)){
        if(line[0]=='#')continue;
        char *save=NULL,*tok=strtok_r(line," \t\r\n",&save); entry e; memset(&e,0,sizeof(e));
        if(!tok)continue; e.offset=strtoull(tok,NULL,10);
        tok=strtok_r(NULL," \t\r\n",&save); if(!tok)goto bad; e.nbytes=strtoull(tok,NULL,10);
        tok=strtok_r(NULL," \t\r\n",&save); if(!tok)goto bad; snprintf(e.dtype,sizeof(e.dtype),"%s",tok);
        tok=strtok_r(NULL," \t\r\n",&save); if(!tok)goto bad; e.ndims=atoi(tok);
        if(e.ndims<1||e.ndims>4)goto bad;
        for(int d=0;d<e.ndims;++d){ tok=strtok_r(NULL," \t\r\n",&save); if(!tok)goto bad; e.shape[d]=strtoull(tok,NULL,10); }
        tok=strtok_r(NULL," \t\r\n",&save); if(!tok||nent>=cap)goto bad;
        snprintf(e.name,sizeof(e.name),"%s",tok); out[nent++]=e;
    }
    fclose(f); return nent;
bad:
    fclose(f); return -1;
}

static entry *find_suffix(entry *es,int n,const char *suffix){
    size_t sl=strlen(suffix);
    for(int i=0;i<n;++i){ size_t nl=strlen(es[i].name); if(nl>=sl&&!strcmp(es[i].name+nl-sl,suffix))return &es[i]; }
    return NULL;
}

static inline float bf16_to_f32(uint16_t b){ union{uint32_t u;float f;}v={(uint32_t)b<<16};return v.f; }
static uint64_t rng_state=UINT64_C(0x4b334b4441500001);
static uint64_t rng_next(void){uint64_t z=(rng_state+=UINT64_C(0x9e3779b97f4a7c15));z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);return z^(z>>31);}
static float rnd(void){return ((rng_next()>>40)/8388608.0f)-1.0f;}

static void bf16_mv128(float *y,const uint16_t *w,const float *x,int cols){
    for(int r=0;r<128;r+=8)
        matvec_bf16_8row(y+r,w+(size_t)r*cols,w+(size_t)(r+1)*cols,w+(size_t)(r+2)*cols,w+(size_t)(r+3)*cols,
                         w+(size_t)(r+4)*cols,w+(size_t)(r+5)*cols,w+(size_t)(r+6)*cols,w+(size_t)(r+7)*cols,x,cols);
}

static void bf16_mv128_parallel(float *y,const uint16_t *w,const float *x,int cols,int threads){
    omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
    for(int r=0;r<128;r+=8)
        matvec_bf16_8row(y+r,w+(size_t)r*cols,w+(size_t)(r+1)*cols,w+(size_t)(r+2)*cols,w+(size_t)(r+3)*cols,
                         w+(size_t)(r+4)*cols,w+(size_t)(r+5)*cols,w+(size_t)(r+6)*cols,w+(size_t)(r+7)*cols,x,cols);
}

static void projection4(float *out[4],const uint16_t *w[4],const float*x,int threads){
    omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
    for(int task=0;task<64;++task){
        int m=task/16,r=(task%16)*8; const uint16_t *b=w[m]+(size_t)r*K3_HIDDEN;
        matvec_bf16_8row(out[m]+r,b,b+K3_HIDDEN,b+2*K3_HIDDEN,b+3*K3_HIDDEN,
                         b+4*K3_HIDDEN,b+5*K3_HIDDEN,b+6*K3_HIDDEN,b+7*K3_HIDDEN,x,K3_HIDDEN);
    }
}
static void projection4_row4(float*out[4],const uint16_t*w[4],const float*x,int threads){
    omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
    for(int task=0;task<128;++task){int m=task/32,r=(task%32)*4;const uint16_t*b=w[m]+(size_t)r*K3_HIDDEN;
        matvec_bf16_4row(out[m]+r,b,b+K3_HIDDEN,b+2*K3_HIDDEN,b+3*K3_HIDDEN,x,K3_HIDDEN);}
}

static double projection_probe(float *out[4],const uint16_t*w[4],const float*x,int threads,int iters){
    for(int i=0;i<3;++i)projection4(out,w,x,threads);
    double t0=now_sec();
    for(int i=0;i<iters;++i)projection4(out,w,x,threads);
    double sec=now_sec()-t0;
    double bytes=(double)iters*4*128*K3_HIDDEN*2;
    printf("PROBE projection threads=%2d us/token=%8.3f resident_GB/s=%8.2f\n",
           threads,sec/iters*1e6,bytes/sec/1e9);
    return sec/iters;
}

static void evict_caches(float *buffer,size_t count,int threads){
    omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
    for(size_t i=0;i<count;i+=16) buffer[i]+=1.0f;
}

static double projection_probe_cold(float *out[4],const uint16_t*w[4],const float*x,int threads,int iters){
    size_t count=(size_t)64*1024*1024/sizeof(float);
    float *evict=calloc(count,sizeof(float));if(!evict)return 0.0;
    evict_caches(evict,count,threads);double sec=0.0;
    for(int i=0;i<iters;++i){evict_caches(evict,count,threads);double t0=now_sec();projection4(out,w,x,threads);sec+=now_sec()-t0;}
    double bytes=(double)iters*4*128*K3_HIDDEN*2;
    printf("PROBE projection_cold threads=%2d us/token=%8.3f HBM_GB/s=%8.2f\n",
           threads,sec/iters*1e6,bytes/sec/1e9);
    free(evict);return sec/iters;
}
static double projection4row_probe_cold(float*out[4],const uint16_t*w[4],const float*x,int threads,int iters){
    size_t count=(size_t)64*1024*1024/sizeof(float);float*evict=calloc(count,4);if(!evict)return 0;double sec=0;
    for(int i=0;i<iters;++i){evict_caches(evict,count,threads);double t=now_sec();projection4_row4(out,w,x,threads);sec+=now_sec()-t;}
    double bytes=(double)iters*4*128*K3_HIDDEN*2;printf("PROBE projection4row_cold threads=%2d us/token=%8.3f HBM_GB/s=%8.2f\n",threads,sec/iters*1e6,bytes/sec/1e9);free(evict);return sec/iters;
}
static double projection5_probe_cold(float*out[5],const uint16_t*w[5],const float*x,int threads,int iters){
    k3_bf16_matrix m[5];for(int i=0;i<5;++i)m[i]=(k3_bf16_matrix){w[i],128,K3_HIDDEN};size_t count=(size_t)64*1024*1024/4;float*evict=calloc(count,4);if(!evict)return 0;double sec=0;
    for(int i=0;i<iters;++i){evict_caches(evict,count,threads);double t=now_sec();k3_dense_many_bf16_row4(out,m,5,x,threads);sec+=now_sec()-t;}
    double bytes=(double)iters*5*128*K3_HIDDEN*2;printf("PROBE projection5row4_cold threads=%2d us/token=%8.3f HBM_GB/s=%8.2f\n",threads,sec/iters*1e6,bytes/sec/1e9);free(evict);return sec/iters;
}

static double kda_probe(float*out,const float*q,const float*k,const float*v,const float*decay,
                        float beta,float*state,int threads,int iters){
    memset(state,0,(size_t)K3_HEAD_DIM*K3_HEAD_DIM*4); omp_set_num_threads(threads);
    double t0=now_sec();
#pragma omp parallel
    {
        for(int it=0;it<iters;++it){
#pragma omp for schedule(static)
            for(int j=0;j<K3_HEAD_DIM;++j)
                k3_kda_step_decay_sve(out+j,q,k,v+j,decay,&beta,state+(size_t)j*K3_HEAD_DIM,
                                      1,K3_HEAD_DIM,1);
        }
    }
    double sec=now_sec()-t0;
    double ops=(double)iters*K3_HEAD_DIM*K3_HEAD_DIM*6;
    printf("PROBE recurrence threads=%2d us/step=%8.3f GOP/s=%7.3f\n",
           threads,sec/iters*1e6,ops/sec/1e9);
    return sec/iters;
}

int main(int argc,char**argv){
    if(argc!=3){fprintf(stderr,"usage: %s BLOB MANIFEST\n",argv[0]);return 2;}
    entry es[20];int ne=load_manifest(argv[2],es,20);if(ne!=13){fprintf(stderr,"expected 13 tensors, got %d\n",ne);return 2;}
    k3_apply_numa_interleave();size_t blob_size=0;uint8_t*blob=k3_load_blob_anon(argv[1],&blob_size);if(!blob){perror("blob");return 2;}
#define PTR(suf,type) ((type*)(blob+find_suffix(es,ne,suf)->offset))
    const uint16_t *qw=PTR("q_proj.weight",uint16_t),*kw=PTR("k_proj.weight",uint16_t);
    const uint16_t *vw=PTR("v_proj.weight",uint16_t),*gw=PTR("g_proj.weight",uint16_t);
    const uint16_t *faw=PTR("f_a_proj.weight",uint16_t),*fbw=PTR("f_b_proj.weight",uint16_t);
    const uint16_t *bw=PTR("b_proj.weight",uint16_t);
    const float *qcw=PTR("q_conv1d.weight",float),*kcw=PTR("k_conv1d.weight",float),*vcw=PTR("v_conv1d.weight",float);
    const float *dt=PTR("dt_bias",float),*alog=PTR("A_log",float),*onorm=PTR("o_norm.weight",float);
#undef PTR
    float*x=malloc(K3_HIDDEN*4),q[128],k[128],v[128],gout[128],fa[128],graw[128],decay[128],o[128];
    float qstate[128*3]={0},kstate[128*3]={0},vstate[128*3]={0};float*state=calloc(128*128,4);
    if(!x||!state)return 2;for(int i=0;i<K3_HIDDEN;++i)x[i]=rnd()*.125f;
    float*outs[4]={q,k,v,gout};const uint16_t*ws[4]={qw,kw,vw,gw};
    float*outs5[5]={q,k,v,gout,fa};const uint16_t*ws5[5]={qw,kw,vw,gw,faw};
    projection4(outs,ws,x,1);
    /* Independent scalar check of a real projected row. */
    double ref=0;for(int c=0;c<K3_HIDDEN;++c)ref+=(double)bf16_to_f32(qw[c])*x[c];
    printf("[real-qproj] row0 abs_err=%.3e %s\n",fabs((double)q[0]-ref),fabs((double)q[0]-ref)<2e-5?"OK":"FAIL");
    bf16_mv128(fa,faw,x,K3_HIDDEN);bf16_mv128(graw,fbw,fa,128);
    float btmp[8];matvec_bf16_8row(btmp,bw,bw,bw,bw,bw,bw,bw,bw,x,K3_HIDDEN);float beta_raw=btmp[0];
    float cq[128],ck[128],cv[128];k3_conv_step_sve(cq,q,qstate,qcw,NULL,128,4);k3_conv_step_sve(ck,k,kstate,kcw,NULL,128,4);k3_conv_step_sve(cv,v,vstate,vcw,NULL,128,4);
    for(int i=0;i<128;++i){cq[i]*=k3_sigmoidf(cq[i]);ck[i]*=k3_sigmoidf(ck[i]);cv[i]*=k3_sigmoidf(cv[i]);}
    k3_l2_normalize_sve(cq,128,1e-6f);k3_l2_normalize_sve(ck,128,1e-6f);
    k3_kda_log_decay(decay,graw,alog,dt,1,128);for(int i=0;i<128;++i)decay[i]=expf(decay[i]);
    float beta=k3_sigmoidf(beta_raw);k3_kda_step_decay_sve(o,cq,ck,cv,decay,&beta,state,1,128,128);
    k3_gated_rmsnorm_sve(o,o,gout,onorm,128,1e-6f);
    double sum=0,ss=0;int finite=1;for(int i=0;i<128;++i){sum+=o[i];ss+=(double)o[i]*o[i];finite&=isfinite(o[i]);}
    printf("[real-kda] beta=%.6f checksum=%+.9e l2=%.9e finite=%s\n",beta,sum,sqrt(ss),finite?"yes":"NO");
    printf("\nReal one-head BF16 projection scaling (four 128x7168 matrices):\n");
    int ts[]={1,4,8,12,16,24,32,48};double p1=0;
    for(int i=0;i<8;++i){double t=projection_probe(outs,ws,x,ts[i],40);if(i==0)p1=t;printf("PROBE projection_eff threads=%2d efficiency=%.3f\n",ts[i],p1/(t*ts[i]));}
    int cold_ts[]={16,24,32,40,44,47,48};
    for(int i=0;i<7;++i){projection_probe_cold(outs,ws,x,cold_ts[i],10);projection4row_probe_cold(outs,ws,x,cold_ts[i],10);projection5_probe_cold(outs5,ws5,x,cold_ts[i],10);}
    printf("\nReal-activation KDA recurrence scaling (128x128 FP32 state):\n");
    double r1=0;
    for(int i=0;i<8;++i){double t=kda_probe(o,cq,ck,cv,decay,beta,state,ts[i],300);if(i==0)r1=t;printf("PROBE recurrence_eff threads=%2d efficiency=%.3f\n",ts[i],r1/(t*ts[i]));}
    free(blob);free(x);free(state);
    return finite?0:1;
}
