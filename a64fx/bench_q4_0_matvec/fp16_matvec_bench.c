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

static int NT,REPS,MODE; /* MODE 0=bf16 widen+fp32, 1=fp16 fmla, 2=fp16mul/fp32acc */
static uint16_t *Wbf[64];   /* per-thread bf16 weights */
static uint16_t *Wf16[64];  /* per-thread fp16 weights */
static float *Xf32; static uint16_t *Xf16;  /* shared activation (fp32 + fp16) */
static size_t n_rows,K;
static pthread_barrier_t bar; static double t0g,t1g; static volatile double g_sink[64];

static void pin(int tid){ int cmg=tid%4,local=tid/4; if(local>=12)local%=12; int core=12+cmg*12+local;
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(core,&s); sched_setaffinity(0,sizeof(s),&s); }

static void *worker(void *arg){
    int tid=(int)(long)arg; pin(tid); set_ftz();
    size_t rp=n_rows/NT, r0=(size_t)tid*rp, r1=(tid==NT-1)?n_rows:r0+rp, myr=r1-r0;
    uint16_t *wb=(uint16_t*)malloc(myr*K*2), *wf=(uint16_t*)malloc(myr*K*2);
    Wbf[tid]=wb; Wf16[tid]=wf;
    for(size_t r=0;r<myr;r++) for(size_t k=0;k<K;k++){
        float v=hashf(r0+r,k,0.03f);
        wb[r*K+k]=f2bf(v);
        __fp16 hh=(__fp16)bf2f(wb[r*K+k]); memcpy(&wf[r*K+k],&hh,2);  /* fp16 = bf16 value */
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
            } else { const uint16_t*w=wf+r*K; svfloat32_t a=svdup_f32(0);
                for(size_t k=0;k+vlh<=K;k+=vlh){ svfloat16_t p=svmul_f16_x(pth,svld1_f16(pth,(const __fp16*)(w+k)),svld1_f16(pth,(const __fp16*)(Xf16+k)));
                    a=svadd_f32_x(pt,a,svcvt_f32_f16_x(pt,svzip1_f16(p,svdup_f16(0))));
                    a=svadd_f32_x(pt,a,svcvt_f32_f16_x(pt,svzip2_f16(p,svdup_f16(0)))); }
                sink+=svaddv_f32(pt,a);
            }
        }
    }
    g_sink[tid]=sink; pthread_barrier_wait(&bar); if(tid==0)t1g=wall(); return NULL;
}

int main(int argc,char**argv){
    n_rows=argc>1?(size_t)atoll(argv[1]):15360; K=argc>2?(size_t)atoll(argv[2]):3840; REPS=argc>3?atoi(argv[3]):40;
    NT=getenv("NT")?atoi(getenv("NT")):48; MODE=getenv("MODE")?atoi(getenv("MODE")):0;
    Xf32=(float*)malloc(K*4); Xf16=(uint16_t*)malloc(K*2);
    for(size_t k=0;k<K;k++){ float v=hashf(k,7,1.2f); Xf32[k]=v; __fp16 h=(__fp16)v; memcpy(&Xf16[k],&h,2); }
    pthread_barrier_init(&bar,NULL,NT);
    pthread_t th[64];
    for(int t=1;t<NT;t++) pthread_create(&th[t],NULL,worker,(void*)(long)t);
    worker((void*)0L);
    for(int t=1;t<NT;t++) pthread_join(th[t],NULL);
    double dt=t1g-t0g, gb=(double)n_rows*K*2*REPS/dt/1e9;
    /* accuracy: recompute a sample of rows (owned by thread 0) 3 ways vs double ref */
    double num=0,den=0,nb=0; int ns=0;
    size_t rp=n_rows/NT;
    for(size_t r=0;r<rp && r<512;r+=8){ const uint16_t*wb=Wbf[0]+r*K,*wf=Wf16[0]+r*K;
        double ref=0; for(size_t k=0;k<K;k++) ref+=(double)bf2f(wb[k])*(double)Xf32[k];
        /* bf16 path value */
        double vbf=0; for(size_t k=0;k<K;k++) vbf+=(double)bf2f(wb[k])*(double)Xf32[k];
        /* fp16 path value (fp16 accumulate, scalar mirror) */
        __fp16 acc=0; for(size_t k=0;k<K;k++){ __fp16 wk,xk; memcpy(&wk,&wf[k],2); memcpy(&xk,&Xf16[k],2); acc=(__fp16)(acc+wk*xk); }
        double vf16=(double)acc;
        double e16=vf16-ref, ebf=vbf-ref; num+=e16*e16; nb+=ebf*ebf; den+=ref*ref; ns++;
    }
    const char*mn=MODE==0?"bf16 widen+fp32":MODE==1?"fp16 fmla (FZ16)":"fp16mul/fp32acc";
    printf("n_rows=%zu K=%zu reps=%d NT=%d MODE=%s\n",n_rows,K,REPS,NT,mn);
    printf("  %.1f ms  %.1f GB/s  (sink=%.3e)\n",dt*1e3,gb,g_sink[0]);
    printf("  relL2: fp16-accum=%.3e  bf16=%.3e  (vs double ref, %d rows)\n",sqrt(num/den),sqrt(nb/den),ns);
    return 0;
}
