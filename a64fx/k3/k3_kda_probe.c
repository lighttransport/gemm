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

typedef struct {
    char name[64];
    uint32_t shape[4], ndim;
    uint64_t count;
    const float *data;
} trace_tensor;

static int load_trace(const char *path, uint8_t **owner,
                      trace_tensor *ts, int cap, int *out_n) {
    FILE *f = fopen(path, "rb"); if (!f) return -1;
    if (fseek(f, 0, SEEK_END) || ftell(f) < 20 || fseek(f, 0, SEEK_SET)) { fclose(f); return -1; }
    long n = ftell(f); if (n <= 0 || fseek(f, 0, SEEK_SET)) { fclose(f); return -1; }
    uint8_t *b = (uint8_t *)malloc((size_t)n); if (!b) { fclose(f); return -1; }
    if (fread(b, 1, (size_t)n, f) != (size_t)n) { free(b); fclose(f); return -1; }
    fclose(f);
    if (memcmp(b, "K3TRC001", 8) != 0) { free(b); return -1; }
    size_t p = 8; uint32_t version, layer, count;
    memcpy(&version, b+p, 4); p += 4; memcpy(&layer, b+p, 4); p += 4; memcpy(&count, b+p, 4); p += 4;
    (void)version; (void)layer;
    if (count > (uint32_t)cap) { free(b); return -1; }
    for (uint32_t i = 0; i < count; ++i) {
        uint16_t nl, nd;
        if (p + 2 > (size_t)n) { free(b); return -1; } memcpy(&nl, b+p, 2); p += 2;
        if (!nl || nl >= sizeof ts[i].name || p + nl + 2 > (size_t)n) { free(b); return -1; }
        memcpy(ts[i].name, b+p, nl); ts[i].name[nl] = 0; p += nl;
        memcpy(&nd, b+p, 2); p += 2; if (!nd || nd > 4 || p + (size_t)nd*4 + 8 > (size_t)n) { free(b); return -1; }
        ts[i].ndim = nd; memset(ts[i].shape, 0, sizeof ts[i].shape);
        for (uint32_t d = 0; d < nd; ++d) { memcpy(&ts[i].shape[d], b+p, 4); p += 4; }
        memcpy(&ts[i].count, b+p, 8); p += 8;
        if (ts[i].count > ((size_t)n-p)/sizeof(float)) { free(b); return -1; }
        ts[i].data = (const float *)(b+p); p += (size_t)ts[i].count*sizeof(float);
    }
    *owner = b; *out_n = (int)count; return 0;
}

static const trace_tensor *trace_find(const trace_tensor *ts, int n, const char *name) {
    for (int i = 0; i < n; ++i) if (!strcmp(ts[i].name, name)) return ts+i;
    return NULL;
}

static int trace_compare(const char *label, const float *got,
                         const trace_tensor *ref, int offset, int count) {
    double se=0.0, sr=0.0; float maxe=0.0f; int maxi=-1;
    if (!ref || offset < 0 || (uint64_t)offset + (uint64_t)count > ref->count) {
        printf("K3TRACE %s missing-or-short\n", label); return 1;
    }
    for (int i=0; i<count; ++i) { double d=(double)got[i]-ref->data[offset+i];
        float a=fabsf((float)d); if(a>maxe){maxe=a;maxi=i;} se+=d*d; sr+=(double)ref->data[offset+i]*ref->data[offset+i]; }
    double rel=sqrt(se/(sr+1e-30)); int bad=maxe>2e-4f && rel>2e-4;
    printf("K3TRACE %s max_abs=%.3e rel_l2=%.3e first=%d %s\n",label,maxe,rel,maxi,bad?"FAIL":"PASS");
    return bad;
}

static k3_pool probe_pool;
static int probe_alloc_failed;
static void *probe_alloc(size_t bytes){void*p=k3_pool_alloc(&probe_pool,bytes);if(!p){probe_alloc_failed=1;fprintf(stderr,"%s\n",k3_pool_error(&probe_pool));}return p;}
static void *probe_calloc(size_t count,size_t size){void*p=k3_pool_calloc(&probe_pool,count,size);if(!p){probe_alloc_failed=1;fprintf(stderr,"%s\n",k3_pool_error(&probe_pool));}return p;}
static void probe_free(void*ptr){if(k3_pool_free(&probe_pool,ptr))fprintf(stderr,"%s\n",k3_pool_error(&probe_pool));}

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
    float *evict=probe_calloc(count,sizeof(float));if(!evict)return 0.0;
    evict_caches(evict,count,threads);double sec=0.0;
    for(int i=0;i<iters;++i){evict_caches(evict,count,threads);double t0=now_sec();projection4(out,w,x,threads);sec+=now_sec()-t0;}
    double bytes=(double)iters*4*128*K3_HIDDEN*2;
    printf("PROBE projection_cold threads=%2d us/token=%8.3f HBM_GB/s=%8.2f\n",
           threads,sec/iters*1e6,bytes/sec/1e9);
    probe_free(evict);return sec/iters;
}
static double projection4row_probe_cold(float*out[4],const uint16_t*w[4],const float*x,int threads,int iters){
    size_t count=(size_t)64*1024*1024/sizeof(float);float*evict=probe_calloc(count,4);if(!evict)return 0;double sec=0;
    for(int i=0;i<iters;++i){evict_caches(evict,count,threads);double t=now_sec();projection4_row4(out,w,x,threads);sec+=now_sec()-t;}
    double bytes=(double)iters*4*128*K3_HIDDEN*2;printf("PROBE projection4row_cold threads=%2d us/token=%8.3f HBM_GB/s=%8.2f\n",threads,sec/iters*1e6,bytes/sec/1e9);probe_free(evict);return sec/iters;
}
static double projection5_probe_cold(float*out[5],const uint16_t*w[5],const float*x,int threads,int iters){
    k3_bf16_matrix m[5];for(int i=0;i<5;++i)m[i]=(k3_bf16_matrix){w[i],128,K3_HIDDEN};size_t count=(size_t)64*1024*1024/4;float*evict=probe_calloc(count,4);if(!evict)return 0;double sec=0;
    for(int i=0;i<iters;++i){evict_caches(evict,count,threads);double t=now_sec();k3_dense_many_bf16_row4(out,m,5,x,threads);sec+=now_sec()-t;}
    double bytes=(double)iters*5*128*K3_HIDDEN*2;printf("PROBE projection5row4_cold threads=%2d us/token=%8.3f HBM_GB/s=%8.2f\n",threads,sec/iters*1e6,bytes/sec/1e9);probe_free(evict);return sec/iters;
}

static double projection5_q8p16_probe_cold(float *out[5], const uint16_t *w[5],
        const float *ref[5], const float *x, int threads, int iters) {
    size_t matrix_bytes=(size_t)128*K3_HIDDEN;
    int8_t *packed[5],*qx=probe_alloc(K3_HIDDEN);float *scale[5];
    const int8_t *pm[5];const float *sm[5];
    size_t count=(size_t)64*1024*1024/sizeof(float);float *evict=probe_calloc(count,sizeof(float));
    if(!qx||!evict)return 0.0;
    for(int m=0;m<5;++m){packed[m]=probe_alloc(matrix_bytes);scale[m]=probe_alloc(128*sizeof(float));
        if(!packed[m]||!scale[m])return 0.0;
        k3_q8p16_quantize_bf16(packed[m],scale[m],w[m],128,K3_HIDDEN);
        pm[m]=packed[m];sm[m]=scale[m];}
    k3_dense_many_q8p16(out,pm,sm,5,128,K3_HIDDEN,x,qx,threads);
    double se=0,sr=0,dot=0,so=0;
    for(int m=0;m<5;++m)for(int r=0;r<128;++r){double a=out[m][r],b=ref[m][r],d=a-b;
        se+=d*d;sr+=b*b;dot+=a*b;so+=a*a;}
    double rel=sqrt(se/(sr+1e-30)),cos=dot/sqrt((sr+1e-30)*(so+1e-30)),sec=0;
    for(int i=0;i<iters;++i){evict_caches(evict,count,threads);double t=now_sec();
        k3_dense_many_q8p16(out,pm,sm,5,128,K3_HIDDEN,x,qx,threads);sec+=now_sec()-t;}
    printf("PROBE projection5q8p16_cold threads=%2d us/token=%8.3f HBM_GB/s=%8.2f rel_l2=%.3e cosine=%.8f %s\n",
        threads,sec/iters*1e6,(double)iters*5*128*K3_HIDDEN/sec/1e9,rel,cos,
        rel<5e-3&&cos>=.99995?"GATE-PASS":"GATE-REJECT");
    for(int m=0;m<5;++m){probe_free(packed[m]);probe_free(scale[m]);}probe_free(qx);probe_free(evict);return sec/iters;
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
    if(argc!=3 && argc!=4){fprintf(stderr,"usage: %s BLOB MANIFEST [KTRACE]\n",argv[0]);return 2;}
    k3_pool_init(&probe_pool,"kda-probe");
    entry es[20];int ne=load_manifest(argv[2],es,20);if(ne!=13){fprintf(stderr,"k3_kda_probe: invalid manifest '%s': expected 13 tensors, got %d\n",argv[2],ne);k3_pool_destroy(&probe_pool);return 2;}
    size_t blob_size=0;uint8_t*blob=k3_pool_load_blob(&probe_pool,argv[1],&blob_size);if(!blob){fprintf(stderr,"%s\n",k3_pool_error(&probe_pool));k3_pool_destroy(&probe_pool);return 2;}
    for(int i=0;i<ne;++i)if(es[i].offset>blob_size||es[i].nbytes>blob_size-es[i].offset){fprintf(stderr,"k3_kda_probe: tensor '%s' exceeds blob (%llu+%llu > %zu)\n",es[i].name,(unsigned long long)es[i].offset,(unsigned long long)es[i].nbytes,blob_size);k3_pool_destroy(&probe_pool);return 2;}
    const char*required[]={"q_proj.weight","k_proj.weight","v_proj.weight","g_proj.weight","f_a_proj.weight","f_b_proj.weight","b_proj.weight","q_conv1d.weight","k_conv1d.weight","v_conv1d.weight","dt_bias","A_log","o_norm.weight"};
    for(size_t i=0;i<sizeof(required)/sizeof(required[0]);++i)if(!find_suffix(es,ne,required[i])){fprintf(stderr,"k3_kda_probe: manifest lacks '%s'\n",required[i]);k3_pool_destroy(&probe_pool);return 2;}
#define PTR(suf,type) ((type*)(blob+find_suffix(es,ne,suf)->offset))
    const uint16_t *qw=PTR("q_proj.weight",uint16_t),*kw=PTR("k_proj.weight",uint16_t);
    const uint16_t *vw=PTR("v_proj.weight",uint16_t),*gw=PTR("g_proj.weight",uint16_t);
    const uint16_t *faw=PTR("f_a_proj.weight",uint16_t),*fbw=PTR("f_b_proj.weight",uint16_t);
    const uint16_t *bw=PTR("b_proj.weight",uint16_t);
    const float *qcw=PTR("q_conv1d.weight",float),*kcw=PTR("k_conv1d.weight",float),*vcw=PTR("v_conv1d.weight",float);
    const float *dt=PTR("dt_bias",float),*alog=PTR("A_log",float),*onorm=PTR("o_norm.weight",float);
#undef PTR
    uint8_t *trace_blob=NULL; trace_tensor traces[32]; int ntrace=0;
    if (argc == 4 && load_trace(argv[3], &trace_blob, traces, 32, &ntrace)) {
        fprintf(stderr,"k3_kda_probe: invalid trace %s\n",argv[3]); k3_pool_destroy(&probe_pool); return 2;
    }
    const trace_tensor *tr_input=trace_blob?trace_find(traces,ntrace,"input"):NULL;
    if (trace_blob && (!tr_input || tr_input->count < K3_HIDDEN)) {
        fprintf(stderr,"k3_kda_probe: trace input must contain %d floats\n",K3_HIDDEN); free(trace_blob); k3_pool_destroy(&probe_pool); return 2;
    }
    float*x=probe_alloc(K3_HIDDEN*4),q[128],k[128],v[128],gout[128],fa[128],graw[128],decay[128],logdecay[128],o[128],recurrent[128];
    float qstate[128*3]={0},kstate[128*3]={0},vstate[128*3]={0};float*state=probe_calloc(128*128,4);
    if(!x||!state){free(trace_blob);k3_pool_destroy(&probe_pool);return 2;}
    if (tr_input) memcpy(x, tr_input->data, (size_t)K3_HIDDEN*sizeof(float));
    else for(int i=0;i<K3_HIDDEN;++i)x[i]=rnd()*.125f;
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
    k3_kda_log_decay(logdecay,graw,alog,dt,1,128);for(int i=0;i<128;++i)decay[i]=expf(logdecay[i]);
    float beta=k3_sigmoidf(beta_raw);k3_kda_step_decay_sve(recurrent,cq,ck,cv,decay,&beta,state,1,128,128);
    memcpy(o,recurrent,sizeof o);
    k3_gated_rmsnorm_sve(o,o,gout,onorm,128,1e-6f);
    int trace_failures=0;
    if (trace_blob) {
        int failures=0;
        failures += trace_compare("q_proj", q, trace_find(traces,ntrace,"q_proj"), 0, 128);
        failures += trace_compare("q_norm", cq, trace_find(traces,ntrace,"q_norm"), 0, 128);
        failures += trace_compare("k_norm", ck, trace_find(traces,ntrace,"k_norm"), 0, 128);
        failures += trace_compare("decay_log", logdecay, trace_find(traces,ntrace,"decay"), 0, 128);
        failures += trace_compare("recurrent", recurrent, trace_find(traces,ntrace,"recurrent"), 0, 128);
        failures += trace_compare("gated", o, trace_find(traces,ntrace,"gated"), 0, 128);
        trace_failures = failures;
        printf("K3TRACE summary failures=%d\n", failures);
    }
    double sum=0,ss=0;int finite=1;for(int i=0;i<128;++i){sum+=o[i];ss+=(double)o[i]*o[i];finite&=isfinite(o[i]);}
    printf("[real-kda] beta=%.6f checksum=%+.9e l2=%.9e finite=%s\n",beta,sum,sqrt(ss),finite?"yes":"NO");
    printf("\nReal one-head BF16 projection scaling (four 128x7168 matrices):\n");
    int ts[]={1,4,8,12,16,24,32,48};double p1=0;
    for(int i=0;i<8;++i){double t=projection_probe(outs,ws,x,ts[i],40);if(i==0)p1=t;printf("PROBE projection_eff threads=%2d efficiency=%.3f\n",ts[i],p1/(t*ts[i]));}
    int cold_ts[]={16,24,32,40,44,47,48};
    for(int i=0;i<7;++i){projection_probe_cold(outs,ws,x,cold_ts[i],10);projection4row_probe_cold(outs,ws,x,cold_ts[i],10);projection5_probe_cold(outs5,ws5,x,cold_ts[i],10);}
    float qref[128],kref[128],vref[128],gref[128],faref[128];
    float *refs[5]={qref,kref,vref,gref,faref};const float *crefs[5]={qref,kref,vref,gref,faref};
    k3_dense_many_bf16_row4(refs,(k3_bf16_matrix[]){{qw,128,K3_HIDDEN},{kw,128,K3_HIDDEN},{vw,128,K3_HIDDEN},{gw,128,K3_HIDDEN},{faw,128,K3_HIDDEN}},5,x,40);
    for(int i=0;i<7;++i)projection5_q8p16_probe_cold(outs5,ws5,crefs,x,cold_ts[i],10);
    printf("\nReal-activation KDA recurrence scaling (128x128 FP32 state):\n");
    double r1=0;
    for(int i=0;i<8;++i){double t=kda_probe(o,cq,ck,cv,decay,beta,state,ts[i],300);if(i==0)r1=t;printf("PROBE recurrence_eff threads=%2d efficiency=%.3f\n",ts[i],r1/(t*ts[i]));}
    probe_free(blob);probe_free(x);probe_free(state);free(trace_blob);
    int status=probe_alloc_failed?2:finite?0:1;
    if (trace_failures) status=1;
    fprintf(stderr,"k3 KDA pool: peak=%.2f MiB reserved=%.2f MiB\n",probe_pool.peak_active_bytes/1048576.0,probe_pool.reserved_bytes/1048576.0);k3_pool_destroy(&probe_pool);return status;
}
