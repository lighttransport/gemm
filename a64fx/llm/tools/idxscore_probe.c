/* int8/SVE Tier-B2 index_score probe (single-process, no MPI) — de-risk the int8
 * indexer-scan lever before wiring it into ds4f_impl.h. The indexer scan
 *   score[t] = Σ_h relu( q[h] · kvc[t] ) * weights[h]      (H heads, hd=128, t<T=ctx/ratio)
 * is the long-ctx decode bottleneck (tb2prep ~38–43%): pure f32 svmla over a cache
 * (idx_kv) that is LINEAR in context, then a top-k=512 selection. Both operands are
 * already FP4-E2M1 block-quantized (rotate + fp4_act_quant, block=32) at write time and
 * merely held in f32 — so int8 (256 levels) is strictly MORE precise than the data
 * already is, and svdot does 4 int8 MACs/lane vs svmla's 1 f32 MAC/lane (and reads 4× fewer
 * bytes). This probe answers two questions BEFORE touching the model:
 *   (1) GO/NO-GO accuracy: does an int8 scan preserve the top-512 SELECTED SET (the only
 *       thing that feeds attention — score VALUES are discarded)? Metric = set overlap, not MSE.
 *   (2) SPEED: int8-svdot vs f32-svmla scan throughput (single-thread; the scan threads over
 *       positions so the per-core ratio is what scales — single-thread is the conservative win).
 *
 * Robust scale: per-channel STATIC scale sk[d] calibrated on the first CAL positions (the S5
 * winner from int8kv_probe — per-token int8 collapses on sink channels). To keep a CLEAN svdot
 * (a per-channel scale would otherwise live INSIDE the dot sum), fold sk[d] into the query:
 *   dot = Σ_d q[d]·kvc[t][d] = Σ_d (q[d]·sk[d]) · (kvc[t][d]/sk[d]) = sq[h] · svdot(q8'[h], k8[t])
 * where k8[t][d]=round(kvc/sk[d]) is computed ONCE at write (streaming-OK, sk is position-free),
 * and q8'[d]=round(q[d]·sk[d]/sq[h]) is quantized per-head per-token (cheap, hd=128).
 *
 * Build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast \
 *        -o /tmp/idxscore_probe a64fx/llm/tools/idxscore_probe.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <arm_sve.h>

#define H    64        /* index_n_heads */
#define HD   128       /* index_head_dim */
#define T    4096      /* compressed positions = ctx16k / ratio4 (accuracy run) */
#define CAL  256       /* calibration window for the static per-channel scale */
#define KPK  512       /* index_topk */
#define TTIME (65536)  /* long scan for cold-ish timing (32 MB f32 / 8 MB int8) */

static uint64_t rng = 0x9e3779b97f4a7c15ULL;
static double urand(void){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (rng>>11)*(1.0/9007199254740992.0); }
static double nrand(void){ double u1=urand(),u2=urand(); if(u1<1e-12)u1=1e-12; return sqrt(-2*log(u1))*cos(6.283185307*u2); }

/* ---- exact copies of the model's fp4 act-quant chain (common/ds4f_impl.h) ---- */
static inline float rs_pow2(float amax,float mi){ float t=amax*mi; uint32_t b; memcpy(&b,&t,4);
    int e=(int)((b>>23)&0xFFu)-127+((b&0x7FFFFFu)?1:0); uint32_t sb=(uint32_t)((e+127)&0xFF)<<23;
    float s; memcpy(&s,&sb,4); return s; }
static inline float clampf(float x,float lo,float hi){ return x<lo?lo:(x>hi?hi:x); }
static inline float bf16r(float f){ uint32_t u; memcpy(&u,&f,4);
    if((u&0x7FFFFFFFu)>=0x7F800000u) return f; uint32_t r=(u+0x7FFFu+((u>>16)&1u))&0xFFFF0000u;
    memcpy(&f,&r,4); return f; }
static inline float fp4_snap(float v){ static const float g[8]={0,.5f,1,1.5f,2,3,4,6};
    static const int ev[8]={1,0,1,0,1,0,1,0}; float sign=v<0?-1.f:1.f,a=sign*v,best=g[0];
    float bd=a<0?-a:a; int bi=0;
    for(int i=1;i<8;i++){ float d=a-g[i]; if(d<0)d=-d;
        if(d<bd-1e-12f||(d<=bd+1e-12f&&ev[i]&&!ev[bi])){bd=d;bi=i;best=g[i];} } return sign*best; }
static inline void fp4_quant(float *x,int n,int block){
    for(int b0=0;b0<n;b0+=block){ int bn=(b0+block<=n)?block:n-b0; float amax=6.0f*1.1754944e-38f;
        for(int j=0;j<bn;j++){ float a=x[b0+j]<0?-x[b0+j]:x[b0+j]; if(a>amax)amax=a; }
        float s=rs_pow2(amax,1.0f/6.0f),inv=1.0f/s;
        for(int j=0;j<bn;j++) x[b0+j]=bf16r(fp4_snap(clampf(x[b0+j]*inv,-6.f,6.f))*s); } }
static inline void rotate(float *x,int n){ for(int h=1;h<n;h<<=1) for(int i=0;i<n;i+=(h<<1))
        for(int j=i;j<i+h;j++){ float a=x[j],b=x[j+h]; x[j]=a+b; x[j+h]=a-b; }
    float sc=1.0f/sqrtf((float)n); for(int i=0;i<n;i++)x[i]*=sc; }

/* ---- the scan kernels ---- */
static inline float fdot(const float *q,const float *k){          /* current f32 svmla path */
    svbool_t pg=svptrue_b32(); svfloat32_t d=svdup_f32(0.f);
    for(int x=0;x<HD;x+=(int)svcntw()){ svbool_t p=svwhilelt_b32(x,HD);
        d=svmla_f32_x(p,d,svld1(p,q+x),svld1(p,k+x)); }
    return svaddv_f32(svptrue_b32(),d);
}
static inline int32_t idot8(const int8_t *q,const int8_t *k){     /* int8 svdot path (HD=128: 2 svdot) */
    svbool_t pb=svptrue_b8(); svint32_t acc=svdup_s32(0);
    for(int d=0; d<HD; d+=(int)svcntb()){
        acc=svdot_s32(acc, svld1_s8(pb,q+d), svld1_s8(pb,k+d));
    }
    return svaddv_s32(svptrue_b32(),acc);
}

/* top-k indices by score (selection of the k largest). idx[] filled, k entries. */
static void topk(const float *sc,int n,int k,int *idx){
    char *used=calloc(n,1);
    for(int i=0;i<k;i++){ int best=-1; float bv=-1e30f;
        for(int t=0;t<n;t++) if(!used[t]&&sc[t]>bv){bv=sc[t];best=t;}
        idx[i]=best; if(best>=0)used[best]=1; }
    free(used);
}
static int overlap(const int *a,const int *b,int k){ char *s=calloc(1<<22,1); /* T<4M */
    for(int i=0;i<k;i++) if(a[i]>=0) s[a[i]]=1;
    int o=0; for(int i=0;i<k;i++) if(b[i]>=0&&s[b[i]])o++; free(s); return o; }

static double nowsec(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec+ts.tv_nsec*1e-9; }

int main(void){
    printf("Tier-B2 index_score int8/SVE probe  (H=%d hd=%d, T=%d=ctx16k/4, topk=%d, calib=%d)\n",
           H,HD,T,KPK,CAL);
    printf("operands are already FP4-block (rotate+fp4_act_quant,block=32); ref = f64 over fp4 data.\n\n");

    /* ---- synthesize q[H][hd] and idx_kv[T][hd] at the real write distribution ---- */
    float *q   = malloc((size_t)H*HD*4);
    float *kv  = malloc((size_t)T*HD*4);
    float *w   = malloc((size_t)H*4);
    for(int h=0;h<H;h++){ float *qh=q+(size_t)h*HD;
        for(int d=0;d<HD;d++) qh[d]=(float)(nrand());
        rotate(qh,HD); fp4_quant(qh,HD,32);
        w[h]=(float)(nrand())*(1.0f/sqrtf((float)(HD*H))); }
    for(int t=0;t<T;t++){ float *kt=kv+(size_t)t*HD;
        for(int d=0;d<HD;d++) kt[d]=(float)(nrand());
        rotate(kt,HD); fp4_quant(kt,HD,32); }

    /* ---- f64 reference score + top-k ---- */
    float *sc_ref=malloc((size_t)T*4);
    for(int t=0;t<T;t++){ const float*kt=kv+(size_t)t*HD; double acc=0;
        for(int h=0;h<H;h++){ const float*qh=q+(size_t)h*HD; double dot=0;
            for(int d=0;d<HD;d++) dot+=(double)qh[d]*kt[d]; if(dot<0)dot=0; acc+=dot*w[h]; }
        sc_ref[t]=(float)acc; }
    int *tk_ref=malloc(KPK*4); topk(sc_ref,T,KPK,tk_ref);

    /* ---- f32 SVE kernel (current path) score + top-k: isolates kernel reassoc from quant ---- */
    float *sc_f32=malloc((size_t)T*4);
    for(int t=0;t<T;t++){ const float*kt=kv+(size_t)t*HD; float acc=0;
        for(int h=0;h<H;h++){ float dot=fdot(q+(size_t)h*HD,kt); if(dot<0)dot=0; acc+=dot*w[h]; }
        sc_f32[t]=acc; }
    int *tk_f32=malloc(KPK*4); topk(sc_f32,T,KPK,tk_f32);

    /* ============ int8 scheme A: per-channel STATIC scale (calib on first CAL) ============ */
    float sk[HD];                              /* per-channel scale, folded into the query */
    for(int d=0;d<HD;d++){ float mx=0; for(int t=0;t<CAL;t++){ float a=kv[(size_t)t*HD+d]; a=a<0?-a:a; if(a>mx)mx=a; }
        sk[d]=mx/127.f; if(sk[d]<=0)sk[d]=1e-30f; }
    int8_t *k8=malloc((size_t)T*HD);           /* int8 cache (1 B/elem) */
    for(int t=0;t<T;t++) for(int d=0;d<HD;d++){ float inv=1.f/sk[d];
        int v=(int)lrintf(kv[(size_t)t*HD+d]*inv); if(v>127)v=127; if(v<-127)v=-127; k8[(size_t)t*HD+d]=(int8_t)v; }
    int8_t *q8=malloc((size_t)H*HD); float sq[H];   /* per-head query: q'[d]=q[d]*sk[d], quant per head */
    for(int h=0;h<H;h++){ const float*qh=q+(size_t)h*HD; float qp[HD],mx=0;
        for(int d=0;d<HD;d++){ qp[d]=qh[d]*sk[d]; float a=qp[d]<0?-qp[d]:qp[d]; if(a>mx)mx=a; }
        sq[h]=mx/127.f; float inv=sq[h]>0?1.f/sq[h]:0;
        for(int d=0;d<HD;d++){ int v=(int)lrintf(qp[d]*inv); if(v>127)v=127; if(v<-127)v=-127; q8[(size_t)h*HD+d]=(int8_t)v; } }
    float *sc_i8=malloc((size_t)T*4);
    for(int t=0;t<T;t++){ const int8_t*kt=k8+(size_t)t*HD; float acc=0;
        for(int h=0;h<H;h++){ float dot=sq[h]*(float)idot8(q8+(size_t)h*HD,kt); if(dot<0)dot=0; acc+=dot*w[h]; }
        sc_i8[t]=acc; }
    int *tk_i8=malloc(KPK*4); topk(sc_i8,T,KPK,tk_i8);

    /* ============ int8 scheme B: per-POSITION scale (does the sink problem bite idx_kv?) ====== */
    int8_t *k8p=malloc((size_t)T*HD); float *stp=malloc((size_t)T*4);
    for(int t=0;t<T;t++){ const float*kt=kv+(size_t)t*HD; float mx=0;
        for(int d=0;d<HD;d++){ float a=kt[d]<0?-kt[d]:kt[d]; if(a>mx)mx=a; }
        stp[t]=mx/127.f; float inv=stp[t]>0?1.f/stp[t]:0;
        for(int d=0;d<HD;d++){ int v=(int)lrintf(kt[d]*inv); if(v>127)v=127; if(v<-127)v=-127; k8p[(size_t)t*HD+d]=(int8_t)v; } }
    int8_t *q8b=malloc((size_t)H*HD); float sqb[H];      /* query quantized directly (no sk fold) */
    for(int h=0;h<H;h++){ const float*qh=q+(size_t)h*HD; float mx=0;
        for(int d=0;d<HD;d++){ float a=qh[d]<0?-qh[d]:qh[d]; if(a>mx)mx=a; }
        sqb[h]=mx/127.f; float inv=sqb[h]>0?1.f/sqb[h]:0;
        for(int d=0;d<HD;d++){ int v=(int)lrintf(qh[d]*inv); if(v>127)v=127; if(v<-127)v=-127; q8b[(size_t)h*HD+d]=(int8_t)v; } }
    float *sc_i8p=malloc((size_t)T*4);
    for(int t=0;t<T;t++){ const int8_t*kt=k8p+(size_t)t*HD; float acc=0;
        for(int h=0;h<H;h++){ float dot=sqb[h]*stp[t]*(float)idot8(q8b+(size_t)h*HD,kt); if(dot<0)dot=0; acc+=dot*w[h]; }
        sc_i8p[t]=acc; }
    int *tk_i8p=malloc(KPK*4); topk(sc_i8p,T,KPK,tk_i8p);

    /* ---- accuracy report: top-k set overlap (the metric that matters) + score L1-rel ---- */
    double l1n=0,l1d=0; for(int t=0;t<T;t++){ l1n+=fabs((double)sc_i8[t]-sc_ref[t]); l1d+=fabs((double)sc_ref[t]); }
    double l1p_n=0; for(int t=0;t<T;t++) l1p_n+=fabs((double)sc_i8p[t]-sc_ref[t]);
    int ov_f32=overlap(tk_ref,tk_f32,KPK), ov_i8=overlap(tk_ref,tk_i8,KPK), ov_i8p=overlap(tk_ref,tk_i8p,KPK);
    printf("%-46s %10s %12s\n","scheme","top512_set","score_L1rel");
    printf("%-46s %9d/%d %12s\n","f32 SVE kernel (reassoc only, no quant)",ov_f32,KPK,"-");
    printf("%-46s %9d/%d %12.4f\n","int8 A: per-channel static (calib first 256)",ov_i8,KPK,l1n/l1d);
    printf("%-46s %9d/%d %12.4f\n","int8 B: per-position absmax",ov_i8p,KPK,l1p_n/l1d);
    printf("  (top512_set = |topk_ref ∩ topk_scheme|; only the selected SET feeds attention)\n\n");

    /* ---- timing: long cold-ish scan, f32 svmla vs int8 svdot, single-thread ---- */
    float  *kvT = malloc((size_t)TTIME*HD*4);
    int8_t *k8T = malloc((size_t)TTIME*HD);
    for(size_t i=0;i<(size_t)TTIME*HD;i++){ kvT[i]=kv[i%((size_t)T*HD)]; k8T[i]=k8[i%((size_t)T*HD)]; }
    float *scT=malloc((size_t)TTIME*4);
    volatile float sink=0;
    /* warm */
    for(int t=0;t<TTIME;t++){ const float*kt=kvT+(size_t)t*HD; float a=0; for(int h=0;h<H;h++){ float d=fdot(q+(size_t)h*HD,kt); if(d<0)d=0; a+=d*w[h]; } scT[t]=a; }
    int IT=4; double t0=nowsec();
    for(int it=0;it<IT;it++) for(int t=0;t<TTIME;t++){ const float*kt=kvT+(size_t)t*HD; float a=0;
        for(int h=0;h<H;h++){ float d=fdot(q+(size_t)h*HD,kt); if(d<0)d=0; a+=d*w[h]; } scT[t]=a; }
    double tf=(nowsec()-t0)/IT; sink+=scT[0];
    t0=nowsec();
    for(int it=0;it<IT;it++) for(int t=0;t<TTIME;t++){ const int8_t*kt=k8T+(size_t)t*HD; float a=0;
        for(int h=0;h<H;h++){ float d=sq[h]*(float)idot8(q8+(size_t)h*HD,kt); if(d<0)d=0; a+=d*w[h]; } scT[t]=a; }
    double ti=(nowsec()-t0)/IT; sink+=scT[0];
    double f32_bytes=(double)TTIME*HD*4, i8_bytes=(double)TTIME*HD*1;   /* kv traffic (q reused, L1) */
    printf("scan timing (single-thread, %d positions, mean of %d):\n",TTIME,IT);
    printf("  f32 svmla : %7.2f ms  %6.1f ns/pos  %6.1f GB/s\n", tf*1e3, tf*1e9/TTIME, f32_bytes/tf/1e9);
    printf("  int8 svdot: %7.2f ms  %6.1f ns/pos  %6.1f GB/s   => %.2fx faster\n",
           ti*1e3, ti*1e9/TTIME, i8_bytes/ti/1e9, tf/ti);
    printf("  (threads split positions => per-core ratio scales; single-thread is the conservative win.\n");
    printf("   int8 also stores idx_kv at 1 B/elem vs 4 => 4x less cache footprint AND HBM traffic.)\n");
    printf("[sink %.3e]\n",(double)sink);
    return 0;
}
