/* Tier-B2 index_score MULTI-THREAD bandwidth probe (single-process, pthreads, no MPI).
 *
 * WHY: the committed idxscore_probe.c showed int8-svdot 1.95x faster than f32-svmla at
 * SINGLE thread, but wiring an on-the-fly-quant int8 scan into the model gave PARITY at 48
 * threads (ctx-warm A/B: tb2prep 125.9 vs 129.8 ms). The on-the-fly path reads idx_kv as
 * f32 (quantizes in-register) => same bytes moved => no win if the 48-thread scan is HBM-
 * bandwidth-bound. This probe settles it: does a RESIDENT int8 idx_kv (4x fewer bytes read)
 * beat f32 at production thread counts? And how much of the wall is NUMA placement?
 *
 * Two orthogonal levers, measured independently:
 *   (1) RESIDENT int8 storage  -> 4x less idx_kv HBM/L2 traffic per position.
 *   (2) NUMA layout of idx_kv  -> in the model idx_kv is written by the MAIN thread (the
 *       serial memcpy in ds4f_index_step), so all pages land on ONE CMG and 48 scan threads
 *       hammer one memory controller. FT_SINGLE models that; FT_SPREAD models a NUMA fix.
 *
 * Scan == production: score[t] = sum_h relu(q[h].kvc[t]) * w[h], H=64, hd=128, per-channel
 * static int8 (scheme A, sk folded into q so the dot is a clean svdot). Buffer is sized to
 * exceed aggregate L2 (4x8MB=32MB) so reads are cold from HBM, not L2-resident.
 *
 * Build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast \
 *        -o /tmp/idxscan_bw a64fx/llm/tools/idxscan_bw_probe.c -lm -lpthread
 * Run (cores 12-59 reserved for bench; 0-11 + mem left for the session):
 *        /tmp/idxscan_bw
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>
#include <arm_sve.h>

#define H     64
#define HD    128
static unsigned TT = (1u<<19);  /* positions (argv[1] overrides): 524288 => f32 256MB int8 64MB (>>L2).
                                 * Sweep small (e.g. 2560 = one prod CSA layer @10k => 1.3MB, L2-resident)
                                 * vs large to locate where resident int8's BW win turns on. */
#define IT    8            /* passes per timed measurement */
#define CAL   256
#define CORE0 12           /* first core to pin to (0-11 left for the session) */

static uint64_t rng=0x9e3779b97f4a7c15ULL;
static double urand(void){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (rng>>11)*(1.0/9007199254740992.0);}
static double nrand(void){ double u1=urand(),u2=urand(); if(u1<1e-12)u1=1e-12; return sqrt(-2*log(u1))*cos(6.283185307*u2);}

/* fp4 act-quant chain (matches common/ds4f_impl.h write dist) */
static inline float rs_pow2(float amax,float mi){ float t=amax*mi; uint32_t b; memcpy(&b,&t,4);
    int e=(int)((b>>23)&0xFFu)-127+((b&0x7FFFFFu)?1:0); uint32_t sb=(uint32_t)((e+127)&0xFF)<<23;
    float s; memcpy(&s,&sb,4); return s;}
static inline float clampf(float x,float lo,float hi){ return x<lo?lo:(x>hi?hi:x);}
static inline float bf16r(float f){ uint32_t u; memcpy(&u,&f,4);
    if((u&0x7FFFFFFFu)>=0x7F800000u) return f; uint32_t r=(u+0x7FFFu+((u>>16)&1u))&0xFFFF0000u; memcpy(&f,&r,4); return f;}
static inline float fp4_snap(float v){ static const float g[8]={0,.5f,1,1.5f,2,3,4,6};
    static const int ev[8]={1,0,1,0,1,0,1,0}; float sign=v<0?-1.f:1.f,a=sign*v,best=g[0]; float bd=a<0?-a:a; int bi=0;
    for(int i=1;i<8;i++){ float d=a-g[i]; if(d<0)d=-d; if(d<bd-1e-12f||(d<=bd+1e-12f&&ev[i]&&!ev[bi])){bd=d;bi=i;best=g[i];}} return sign*best;}
static inline void fp4_quant(float *x,int n,int block){
    for(int b0=0;b0<n;b0+=block){ int bn=(b0+block<=n)?block:n-b0; float amax=6.0f*1.1754944e-38f;
        for(int j=0;j<bn;j++){ float a=x[b0+j]<0?-x[b0+j]:x[b0+j]; if(a>amax)amax=a;}
        float s=rs_pow2(amax,1.0f/6.0f),inv=1.0f/s;
        for(int j=0;j<bn;j++) x[b0+j]=bf16r(fp4_snap(clampf(x[b0+j]*inv,-6.f,6.f))*s);}}
static inline void rotate(float *x,int n){ for(int h=1;h<n;h<<=1) for(int i=0;i<n;i+=(h<<1))
        for(int j=i;j<i+h;j++){ float a=x[j],b=x[j+h]; x[j]=a+b; x[j+h]=a-b;}
    float sc=1.0f/sqrtf((float)n); for(int i=0;i<n;i++)x[i]*=sc;}

static inline float fdot(const float *q,const float *k){
    svbool_t pg=svptrue_b32(); svfloat32_t d=svdup_f32(0.f);
    for(int x=0;x<HD;x+=(int)svcntw()){ svbool_t p=svwhilelt_b32(x,HD); d=svmla_f32_x(p,d,svld1(p,q+x),svld1(p,k+x)); }
    return svaddv_f32(svptrue_b32(),d);
}
static inline int32_t idot8(const int8_t *q,const int8_t *k){
    svbool_t pb=svptrue_b8(); svint32_t acc=svdup_s32(0);
    for(int d=0;d<HD;d+=(int)svcntb()) acc=svdot_s32(acc,svld1_s8(pb,q+d),svld1_s8(pb,k+d));
    return svaddv_s32(svptrue_b32(),acc);
}
static double nowsec(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9;}

/* ---- shared state ---- */
static float  *g_kv;      /* f32 resident idx_kv  [TT*HD] */
static int8_t *g_k8;      /* int8 resident idx_kv [TT*HD] */
static float  *g_q;       /* f32 query [H*HD] */
static int8_t *g_q8;      /* int8 query [H*HD] */
static float   g_sq[H], g_w[H];
static float  *g_sc;      /* scratch scores [TT] */

enum { M_F32, M_I8, M_RAWF, M_RAWI };
typedef struct { int tid,nthr,mode; double sec; } targ;

static void range(int nthr,int tid,unsigned *t0,unsigned *t1){
    unsigned per=TT/nthr, ex=TT%nthr;
    *t0=per*tid+(tid<(int)ex?tid:ex); *t1=*t0+per+(tid<(int)ex?1:0);
}
static void *worker(void *a){
    targ *T=(targ*)a;
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(CORE0+T->tid,&cs); sched_setaffinity(0,sizeof(cs),&cs);
    unsigned t0,t1; range(T->nthr,T->tid,&t0,&t1);
    double s=nowsec(); volatile float sink=0;
    if(T->mode==M_F32){
        for(int it=0;it<IT;it++) for(unsigned t=t0;t<t1;t++){ const float*kt=g_kv+(size_t)t*HD; float acc=0;
            for(int h=0;h<H;h++){ float d=fdot(g_q+(size_t)h*HD,kt); if(d<0)d=0; acc+=d*g_w[h]; } g_sc[t]=acc; }
    } else if(T->mode==M_I8){
        for(int it=0;it<IT;it++) for(unsigned t=t0;t<t1;t++){ const int8_t*kt=g_k8+(size_t)t*HD; float acc=0;
            for(int h=0;h<H;h++){ float d=g_sq[h]*(float)idot8(g_q8+(size_t)h*HD,kt); if(d<0)d=0; acc+=d*g_w[h]; } g_sc[t]=acc; }
    } else if(T->mode==M_RAWF){
        svbool_t pg=svptrue_b32(); svfloat32_t a=svdup_f32(0.f);
        for(int it=0;it<IT;it++) for(unsigned t=t0;t<t1;t++){ const float*kt=g_kv+(size_t)t*HD;
            for(int d=0;d<HD;d+=(int)svcntw()) a=svadd_f32_x(pg,a,svld1(pg,kt+d)); }
        sink+=svaddv_f32(pg,a);
    } else if(T->mode==M_RAWI){
        svbool_t pb=svptrue_b8(); svint32_t a=svdup_s32(0);
        for(int it=0;it<IT;it++) for(unsigned t=t0;t<t1;t++){ const int8_t*kt=g_k8+(size_t)t*HD;
            for(int d=0;d<HD;d+=(int)svcntb()) a=svdot_s32(a,svld1_s8(pb,kt+d),svdup_s8(1)); }
        sink+=(float)svaddv_s32(svptrue_b32(),a);
    }
    T->sec=nowsec()-s; g_sc[t0]+=sink;
    return NULL;
}

static double run(int nthr,int mode){
    pthread_t th[64]; targ ta[64];
    for(int i=0;i<nthr;i++){ ta[i]=(targ){i,nthr,mode,0}; pthread_create(&th[i],NULL,worker,&ta[i]); }
    double mx=0; for(int i=0;i<nthr;i++){ pthread_join(th[i],NULL); if(ta[i].sec>mx)mx=ta[i].sec; }
    return mx; /* wall = slowest thread */
}

int main(int argc,char**argv){
    if(argc>1){ unsigned v=(unsigned)strtoul(argv[1],NULL,10); if(v>0) TT=v; }
    printf("idx_score multithread BW probe  (H=%d hd=%d, TT=%u pos, f32buf=%.2fMB int8buf=%.2fMB, IT=%d)\n",
           H,HD,TT,(double)TT*HD*4/1e6,(double)TT*HD/1e6,IT);
    g_kv=malloc((size_t)TT*HD*4); g_k8=malloc((size_t)TT*HD); g_sc=malloc((size_t)TT*4);
    g_q=malloc((size_t)H*HD*4); g_q8=malloc((size_t)H*HD);
    /* build a representative tile (16384 positions) then tile it across TT */
    unsigned base=1u<<14;
    float sk[HD];
    { float *kv=malloc((size_t)base*HD*4);
      for(unsigned t=0;t<base;t++){ float*kt=kv+(size_t)t*HD; for(int d=0;d<HD;d++)kt[d]=(float)nrand(); rotate(kt,HD); fp4_quant(kt,HD,32);}
      for(int h=0;h<H;h++){ float*qh=g_q+(size_t)h*HD; for(int d=0;d<HD;d++)qh[d]=(float)nrand(); rotate(qh,HD); fp4_quant(qh,HD,32);
        g_w[h]=(float)nrand()*(1.0f/sqrtf((float)(HD*H))); }
      for(int d=0;d<HD;d++){ float mx=0; for(unsigned t=0;t<CAL;t++){ float a=kv[(size_t)t*HD+d]; a=a<0?-a:a; if(a>mx)mx=a;} sk[d]=mx/127.f; if(sk[d]<=0)sk[d]=1e-30f; }
      for(int h=0;h<H;h++){ const float*qh=g_q+(size_t)h*HD; float qp[HD],mx=0;
        for(int d=0;d<HD;d++){ qp[d]=qh[d]*sk[d]; float a=qp[d]<0?-qp[d]:qp[d]; if(a>mx)mx=a;} g_sq[h]=mx/127.f; float inv=g_sq[h]>0?1.f/g_sq[h]:0;
        for(int d=0;d<HD;d++){ int v=(int)lrintf(qp[d]*inv); if(v>127)v=127; if(v<-127)v=-127; g_q8[(size_t)h*HD+d]=(int8_t)v;} }
      /* fill the full TT buffers (single-thread = FT_SINGLE = main thread => one CMG, models prod) */
      for(unsigned t=0;t<TT;t++){ const float*src=kv+(size_t)(t%base)*HD; float*kt=g_kv+(size_t)t*HD;
        for(int d=0;d<HD;d++){ kt[d]=src[d]; int v=(int)lrintf(src[d]/sk[d]); if(v>127)v=127; if(v<-127)v=-127; g_k8[(size_t)t*HD+d]=(int8_t)v; } }
      free(kv);
    }

    int sweep[]={1,12,24,48};
    printf("\nlayout: FT_SINGLE (idx_kv first-touched by main thread => one CMG = production write path)\n");
    printf("%-5s | %-24s | %-26s | %-7s | %-8s\n","thr","f32 svmla","int8 svdot(resident)","i8/f32","rawI/rawF");
    for(unsigned i=0;i<sizeof(sweep)/sizeof(int);i++){ int n=sweep[i];
        double tf=1e9,ti=1e9,rf=1e9,ri=1e9;
        for(int rep=0;rep<2;rep++){ double a=run(n,M_F32); if(a<tf)tf=a; }
        for(int rep=0;rep<2;rep++){ double a=run(n,M_I8);  if(a<ti)ti=a; }
        for(int rep=0;rep<2;rep++){ double a=run(n,M_RAWF);if(a<rf)rf=a; }
        for(int rep=0;rep<2;rep++){ double a=run(n,M_RAWI);if(a<ri)ri=a; }
        double fb=(double)TT*HD*4*IT, ib=(double)TT*HD*1*IT;   /* bytes moved (kv read once/pos) */
        printf("%-5d | %7.1fms %8.1fGB/s | %7.1fms %8.1fGB/s | %6.2fx | %6.2fx\n",
               n, tf*1e3, fb/tf/1e9, ti*1e3, ib/ti/1e9, tf/ti, rf/ri);
    }
    printf("\nkey: i8/f32>1 => resident int8 scan is faster. If it grows from 1->48 thr, the scan is\n");
    printf("     bandwidth-bound and RESIDENT int8 (4x fewer bytes) is the real lever -- the on-the-fly\n");
    printf("     int8 wired into the model can't win because it still reads f32. rawI/rawF is the pure\n");
    printf("     read-traffic ratio ceiling (~4x if fully BW-bound on one CMG controller).\n");
    return 0;
}
