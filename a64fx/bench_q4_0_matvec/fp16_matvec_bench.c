/* fp16-FMA vs bf16-widen matvec on A64FX: BW + accuracy. 12B sizes, per-CMG-separate
 * placement (memory not the limit), FZ16 on for the fp16 path.
 *
 * bf16 path: widen bf16->fp32 (zip) + fp32 FMA (current decode kernel). Compute-bound
 *            ~300 GB/s in this regime.
 * fp16 path: weights bf16->fp16, x fp32->fp16, native fp16 FMLA (2x fp32 rate) with
 *            FZ16 (flush fp16 subnormals). Accumulates in fp16 -> precision risk over K.
 * Also: fp16-mul / fp32-acc (widen the fp16 product back to fp32 each step) for accuracy.
 *
 * Reports per-path GB/s and relL2 vs a double-precision reference (sampled rows).
 * build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast fp16_matvec_bench.c -lpthread -lm -o fp16_matvec_bench
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>
#include <arm_sve.h>

static double wall(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }
static float bf2f(uint16_t h){ uint32_t u=(uint32_t)h<<16; float f; memcpy(&f,&u,4); return f; }
static uint16_t f2bf(float f){ uint32_t u; memcpy(&u,&f,4); return (uint16_t)((u+0x8000)>>16); }
/* uncorrelated pseudo-random in [-scale,scale] (realistic LLM: weights~0.02, acts~1) */
static float hashf(uint64_t a,uint64_t b,float scale){ uint64_t h=a*0x9E3779B97F4A7C15ULL+b*0xC2B2AE3D27D4EB4FULL;
    h^=h>>29; h*=0xBF58476D1CE4E5B9ULL; h^=h>>32; return scale*((float)(h&0xFFFFF)/524288.0f-1.0f); }

/* set FPCR FZ (bit24) + FZ16 (bit19): flush fp32 AND fp16 subnormals to zero */
static void set_ftz(void){ uint64_t f; __asm__ volatile("mrs %0,fpcr":"=r"(f)); f|=(1ULL<<24)|(1ULL<<19); __asm__ volatile("msr fpcr,%0"::"r"(f)); }

static int NT,REPS,MODE; /* 0=bf16 widen+fp32, 1=fp16 fmla, 2=fp16mul/fp32acc, 3=int8 SDOT */
static uint16_t *Wbf[64];   /* per-thread bf16 weights */
static uint16_t *Wf16[64];  /* per-thread fp16 weights */
static int8_t *Wi8[64]; static float *Wsc[64];   /* per-thread int8 weights + per-row scale */
static float *Xf32; static uint16_t *Xf16; static int8_t *Xi8; static float Xsc; /* activation */
static size_t n_rows,K;
static pthread_barrier_t bar; static double t0g,t1g; static volatile double g_sink[64];

static void pin(int tid){ int cmg=tid%4,local=tid/4; if(local>=12)local%=12; int core=12+cmg*12+local;
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(core,&s); sched_setaffinity(0,sizeof(s),&s); }

static void *worker(void *arg){
    int tid=(int)(long)arg; pin(tid); set_ftz();
    size_t rp=n_rows/NT, r0=(size_t)tid*rp, r1=(tid==NT-1)?n_rows:r0+rp, myr=r1-r0;
    uint16_t *wb=(uint16_t*)malloc(myr*K*2), *wf=(uint16_t*)malloc(myr*K*2);
    int8_t *wi=(int8_t*)malloc(myr*K); float *ws=(float*)malloc(myr*sizeof(float));
    Wbf[tid]=wb; Wf16[tid]=wf; Wi8[tid]=wi; Wsc[tid]=ws;
    for(size_t r=0;r<myr;r++){
        float mx=0;
        for(size_t k=0;k<K;k++){ float v=hashf(r0+r,k,0.03f); wb[r*K+k]=f2bf(v);
            __fp16 hh=(__fp16)bf2f(wb[r*K+k]); memcpy(&wf[r*K+k],&hh,2);
            float a=v<0?-v:v; if(a>mx)mx=a; }
        float sc=mx/127.0f, inv=sc>0?1.0f/sc:0; ws[r]=sc;       /* per-row int8 scale */
        for(size_t k=0;k<K;k++){ float v=bf2f(wb[r*K+k]); int q=(int)lrintf(v*inv);
            wi[r*K+k]=(int8_t)(q<-127?-127:q>127?127:q); }
    }
    pthread_barrier_wait(&bar); if(tid==0)t0g=wall(); pthread_barrier_wait(&bar);
    svbool_t pt=svptrue_b32(), pth=svptrue_b16(); int vlh=svcnth(),vl=svcntw(); svuint16_t zero=svdup_u16(0);
    double sink=0;
    for(int rep=0;rep<REPS;rep++){
        for(size_t r=0;r<myr;r++){
            if(MODE==0){ const uint16_t*w=wb+r*K; svfloat32_t a=svdup_f32(0);
                for(size_t k=0;k+vlh<=K;k+=vlh){ svuint16_t raw=svldnt1_u16(pth,w+k);
                    a=svmla_f32_x(pt,a,svreinterpret_f32_u16(svzip1_u16(zero,raw)),svld1_f32(pt,Xf32+k));
                    a=svmla_f32_x(pt,a,svreinterpret_f32_u16(svzip2_u16(zero,raw)),svld1_f32(pt,Xf32+k+vl)); }
                sink+=svaddv_f32(pt,a);
            } else if(MODE==1){ const uint16_t*w=wf+r*K; svfloat16_t a=svdup_f16(0);
                for(size_t k=0;k+vlh<=K;k+=vlh) a=svmla_f16_x(pth,a,svld1_f16(pth,(const __fp16*)(w+k)),svld1_f16(pth,(const __fp16*)(Xf16+k)));
                svfloat32_t lo=svcvt_f32_f16_x(pt,svzip1_f16(a,svdup_f16(0))), hi=svcvt_f32_f16_x(pt,svzip2_f16(a,svdup_f16(0)));
                sink+=svaddv_f32(pt,svadd_x(pt,lo,hi));
            } else if(MODE==2){ const uint16_t*w=wf+r*K; svfloat32_t a=svdup_f32(0);
                for(size_t k=0;k+vlh<=K;k+=vlh){ svfloat16_t p=svmul_f16_x(pth,svld1_f16(pth,(const __fp16*)(w+k)),svld1_f16(pth,(const __fp16*)(Xf16+k)));
                    a=svadd_f32_x(pt,a,svcvt_f32_f16_x(pt,svzip1_f16(p,svdup_f16(0))));
                    a=svadd_f32_x(pt,a,svcvt_f32_f16_x(pt,svzip2_f16(p,svdup_f16(0)))); }
                sink+=svaddv_f32(pt,a);
            } else { /* int8 SDOT (per-row w scale, per-call x scale) */
                const int8_t*w=Wi8[tid]+r*K; svbool_t pb=svptrue_b8(); int vlb=svcntb();
                svint32_t a=svdup_s32(0);
                for(size_t k=0;k+vlb<=K;k+=vlb) a=svdot_s32(a,svld1_s8(pb,w+k),svld1_s8(pb,Xi8+k));
                sink += (double)svaddv_s32(pt,a)*(double)Wsc[tid][r]*(double)Xsc;
            }
        }
    }
    g_sink[tid]=sink; pthread_barrier_wait(&bar); if(tid==0)t1g=wall(); return NULL;
}

int main(int argc,char**argv){
    n_rows=argc>1?(size_t)atoll(argv[1]):15360; K=argc>2?(size_t)atoll(argv[2]):3840; REPS=argc>3?atoi(argv[3]):40;
    NT=getenv("NT")?atoi(getenv("NT")):48; MODE=getenv("MODE")?atoi(getenv("MODE")):0;
    Xf32=(float*)malloc(K*4); Xf16=(uint16_t*)malloc(K*2); Xi8=(int8_t*)malloc(K);
    { float mx=0; for(size_t k=0;k<K;k++){ float v=hashf(k,7,1.2f); Xf32[k]=v; __fp16 h=(__fp16)v; memcpy(&Xf16[k],&h,2); float a=v<0?-v:v; if(a>mx)mx=a; }
      Xsc=mx/127.0f; float inv=Xsc>0?1.0f/Xsc:0; for(size_t k=0;k<K;k++){ int q=(int)lrintf(Xf32[k]*inv); Xi8[k]=(int8_t)(q<-127?-127:q>127?127:q); } }
    pthread_barrier_init(&bar,NULL,NT);
    pthread_t th[64];
    for(int t=1;t<NT;t++) pthread_create(&th[t],NULL,worker,(void*)(long)t);
    worker((void*)0L);
    for(int t=1;t<NT;t++) pthread_join(th[t],NULL);
    double dt=t1g-t0g;
    int bytes=MODE==3?1:2;  /* int8 weight = 1 byte, bf16/fp16 = 2 */
    double gb=(double)n_rows*K*bytes*REPS/dt/1e9;
    double gmac=(double)n_rows*K*REPS/dt/1e9;   /* precision-independent compute throughput */
    /* accuracy of the CURRENT mode vs double ref (sampled rows) */
    double num=0,den=0; int ns=0;
    size_t rp=n_rows/NT;
    for(size_t r=0;r<rp && r<512;r+=8){ const uint16_t*wb=Wbf[0]+r*K,*wf=Wf16[0]+r*K; const int8_t*wi=Wi8[0]+r*K;
        double ref=0; for(size_t k=0;k<K;k++) ref+=(double)bf2f(wb[k])*(double)Xf32[k];
        double val;
        if(MODE==0){ double a=0; for(size_t k=0;k<K;k++) a+=(double)bf2f(wb[k])*(double)Xf32[k]; val=a; }           /* bf16->fp32 ~exact */
        else if(MODE==1){ __fp16 a=0; for(size_t k=0;k<K;k++){ __fp16 wk,xk; memcpy(&wk,&wf[k],2); memcpy(&xk,&Xf16[k],2); a=(__fp16)(a+wk*xk);} val=(double)a; } /* fp16 accum */
        else if(MODE==2){ double a=0; for(size_t k=0;k<K;k++){ __fp16 wk,xk; memcpy(&wk,&wf[k],2); memcpy(&xk,&Xf16[k],2); a+=(double)(__fp16)(wk*xk);} val=a; } /* fp16 mul, fp32 acc */
        else { long acc=0; for(size_t k=0;k<K;k++) acc+=(long)wi[k]*(long)Xi8[k]; val=(double)acc*(double)Wsc[0][r]*(double)Xsc; }   /* int8 SDOT */
        double e=val-ref; num+=e*e; den+=ref*ref; ns++;
    }
    const char*mn=MODE==0?"bf16 widen+fp32":MODE==1?"fp16 fmla (FZ16)":MODE==2?"fp16mul/fp32acc":"int8 SDOT (per-row)";
    printf("n_rows=%zu K=%zu reps=%d NT=%d MODE=%s\n",n_rows,K,REPS,NT,mn);
    printf("  %.1f ms  %.1f GB/s (weight)  %.1f GMAC/s  relL2=%.3e (vs fp64 ref, %d rows, sink=%.3e)\n",
        dt*1e3,gb,gmac,sqrt(num/den),ns,g_sink[0]);
    return 0;
}
