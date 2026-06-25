/* attn_fused_prefill.c - WS3: fused fp16 prefill attention for Gemma-4 shapes.
 *
 * Validates the fused QK^T -> causal mask -> softmax -> P*V pipeline (fp16 GEMM,
 * fp32 accumulation) against the scalar fp32 reference matching
 * tf_gemma4_attention_batch semantics: NO 1/sqrt(hd) scale and NO softcap in the
 * attention loop (folded earlier), standard softmax, causal mask. One head
 * (multi-head is an outer loop; GQA shares K/V across query heads).
 *
 * QK^T:  scores[q,k] = sum_d Q[q,d]*K[k,d]   (contract d=hd)  -> micro_kernel_fp16_12x2_swp
 * P*V:   out[q,d]    = sum_k P[q,k]*V[k,d]    (contract k=seq) -> same kernel
 *
 * Build (native A64FX):
 *   fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -D_GNU_SOURCE \
 *       attn_fused_prefill.c ../fused-gemm/micro_kernel_fp16_12x2_swp.S \
 *       ../fused-gemm/micro_kernel_fp16_12x2_swp_accum.S -lm -o attn_fused_prefill
 *   OMP_NUM_THREADS=48 OMP_PROC_BIND=close OMP_PLACES=cores ./attn_fused_prefill [hd] [N] [window]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <arm_sve.h>
#include <omp.h>
#include <sys/mman.h>
#ifdef USE_FAPP
#include "fj_tool/fapp.h"
#endif

void micro_kernel_fp16_12x2_swp(const _Float16*A,const _Float16*B,float*C,int64_t K,int64_t u,int64_t ldc);
void micro_kernel_fp16_12x2_swp_accum(const _Float16*A,const _Float16*B,float*C,int64_t K,int64_t u,int64_t ldc);

static inline uint64_t rdcyc(void){ uint64_t v; __asm__ volatile("mrs %0, cntvct_el0":"=r"(v)); return v; }
static inline uint64_t rdfreq(void){ uint64_t v; __asm__ volatile("mrs %0, cntfrq_el0":"=r"(v)); return v; }
/* FPCR.FZ+FZ16: flush fp32/fp16 subnormals. CRITICAL for fp16 attention — softmax
 * probabilities and small QK/PV products go subnormal; without this the fp16 kernel
 * runs ~3x slower (A64FX subnormal microcode). Each OpenMP thread must set its own. */
static inline void set_fz(void){ uint64_t f; __asm__ volatile("mrs %0,fpcr":"=r"(f)); f|=(1ULL<<24)|(1ULL<<19); __asm__ volatile("msr fpcr,%0"::"r"(f)); }
static void*xmap(size_t n){ void*p=mmap(NULL,n,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0); if(p==MAP_FAILED){perror("mmap");exit(1);} return p; }

/* Vectorized FEXPA expf: exp(v) = 2^(v*log2e). The 2-insn core (add SH + fexpa) gives
 * 2^(round_{1/64}(u)) at ~0.5% (the doc's exp2_fexpa); we recover the sub-1/64 residual
 * r = u - round(u) and multiply by 2^r ~= 1 + r*ln2 -> ~1e-3, libm-class for softmax.
 * SH = 0x48481fc0 = 204927.0f is the FEXPA magic shift. Core chain (5 ops on the
 * critical path): mul(u) -> add(z) -> fexpa(sc) -> [sub(lg) -> sub(r)] -> mla -> mul. */
#define FEXPA_SH 204927.0f
static inline svfloat32_t sve_expf(svbool_t pg, svfloat32_t v){
    const float L2E=1.4426950408889634f, LN2=0.6931471805599453f;
    svfloat32_t u  = svmul_x(pg, v, svdup_f32(L2E));        /* u = v*log2e */
    svfloat32_t z  = svadd_x(pg, u, svdup_f32(FEXPA_SH));   /* z = u + SH (rounds low bits to 1/64) */
    svfloat32_t sc; __asm__("fexpa %0.s, %1.s":"=w"(sc):"w"(z)); /* sc = 2^round_{1/64}(u) */
    svfloat32_t lg = svsub_x(pg, z, svdup_f32(FEXPA_SH));   /* lg = round_{1/64}(u) */
    svfloat32_t r  = svsub_x(pg, u, lg);                    /* r = u - lg (|r| <= 1/128) */
    return svmul_x(pg, sc, svmla_x(pg, svdup_f32(1.0f), r, svdup_f32(LN2))); /* sc*(1+r*ln2) */
}
#define MR 12
#define NR 64

int main(int argc,char**argv){
    int hd = argc>1?atoi(argv[1]):256;     /* head_dim: 256 (SWA) or 512 (full) */
    int N  = argc>2?atoi(argv[2]):384;     /* query/seq length */
    int win= argc>3?atoi(argv[3]):0;       /* SWA window (0 = full causal) */
    int nt = getenv("OMP_NUM_THREADS")?atoi(getenv("OMP_NUM_THREADS")):48;
    if(hd%NR){ fprintf(stderr,"hd must be mult of 64\n"); return 1; }
    int seq=N;
    int Npad=(N+MR-1)/MR*MR;               /* pad queries to 12 */
    int Kpad=(seq+NR-1)/NR*NR;             /* pad keys to 64 for QK tiles */
    int hdt=hd/NR;                         /* d-tiles for PV */

    /* synthetic Q,K,V fp16 */
    _Float16*Q=(_Float16*)xmap((size_t)Npad*hd*sizeof(_Float16));
    _Float16*K=(_Float16*)xmap((size_t)Kpad*hd*sizeof(_Float16));
    _Float16*V=(_Float16*)xmap((size_t)Kpad*hd*sizeof(_Float16));
    memset(Q,0,(size_t)Npad*hd*sizeof(_Float16)); memset(K,0,(size_t)Kpad*hd*sizeof(_Float16)); memset(V,0,(size_t)Kpad*hd*sizeof(_Float16));
    for(int i=0;i<N;i++) for(int d=0;d<hd;d++) Q[(size_t)i*hd+d]=(_Float16)(0.05f*(float)(((i*3+d*7)&0x1f)-16));
    for(int j=0;j<seq;j++) for(int d=0;d<hd;d++){ K[(size_t)j*hd+d]=(_Float16)(0.05f*(float)(((j*5+d*3)&0x1f)-16));
                                                  V[(size_t)j*hd+d]=(_Float16)(0.04f*(float)(((j*7+d*2)&0x1f)-16)); }

    /* ---- scalar fp32 reference (causal, optional window) ---- */
    float*Oref=(float*)xmap((size_t)N*hd*sizeof(float));
    #pragma omp parallel for num_threads(nt) schedule(static)
    for(int i=0;i<N;i++){
        int start=0; if(win&&i+1>win) start=i-win+1;
        float*sc=(float*)malloc((size_t)(i+1)*sizeof(float)); float mx=-1e30f;
        for(int j=start;j<=i;j++){ float s=0; for(int d=0;d<hd;d++) s+=(float)Q[(size_t)i*hd+d]*(float)K[(size_t)j*hd+d]; sc[j]=s; if(s>mx)mx=s; }
        float sum=0; for(int j=start;j<=i;j++){ sc[j]=expf(sc[j]-mx); sum+=sc[j]; }
        float inv=1.0f/sum;
        for(int d=0;d<hd;d++){ float o=0; for(int j=start;j<=i;j++) o+=sc[j]*inv*(float)V[(size_t)j*hd+d]; Oref[(size_t)i*hd+d]=o; }
        free(sc);
    }

    /* ---- pack K for QK^T: Bqk[ktile][d*64+c] = K[ktile*64+c][d]  (B=[hd][64] per ktile) ---- */
    int KT=Kpad/NR;
    _Float16*Bqk=(_Float16*)xmap((size_t)KT*hd*NR*sizeof(_Float16));
    #pragma omp parallel for num_threads(nt) schedule(static)
    for(int kt=0;kt<KT;kt++) for(int d=0;d<hd;d++) for(int c=0;c<NR;c++){
        int kj=kt*NR+c; Bqk[((size_t)kt*hd+d)*NR+c]= kj<seq? K[(size_t)kj*hd+d] : (_Float16)0; }
    /* pack V for P*V: Bv[dtile][k*64+c] = V[k][dtile*64+c]  (B=[seq][64] per dtile, K-major=k) ---- */
    _Float16*Bv=(_Float16*)xmap((size_t)hdt*Kpad*NR*sizeof(_Float16));
    #pragma omp parallel for num_threads(nt) schedule(static)
    for(int dt=0;dt<hdt;dt++) for(int k=0;k<Kpad;k++) for(int c=0;c<NR;c++){
        Bv[((size_t)dt*Kpad+k)*NR+c]= k<seq? V[(size_t)k*hd+dt*NR+c] : (_Float16)0; }

    float*Ofu=(float*)xmap((size_t)Npad*hd*sizeof(float));
    int QT=Npad/MR;
    int scalar_exp=getenv("ATTN_SCALAR_EXP")?1:0;  /* A/B: libm expf vs FEXPA sve_expf */

    /* scratch per thread allocated inside loop (sized by hd,Kpad) */
    double freq=(double)rdfreq(); volatile uint64_t c0=0,c1=0; int reps=20;
    for(int rep=-1;rep<reps;rep++){
      if(rep==0){ c0=rdcyc();
#ifdef USE_FAPP
        fapp_start("attn_fused",1,0);
#endif
      }
      #pragma omp parallel num_threads(nt)
      {
        set_fz();   /* per-thread FPCR.FZ16 — softmax probs go subnormal otherwise */
        _Float16*Aq=(_Float16*)malloc((size_t)hd*MR*sizeof(_Float16));      /* [hd][12] */
        float*sc=(float*)malloc((size_t)MR*Kpad*sizeof(float));            /* scores [12][Kpad] fp32 */
        _Float16*Pp=(_Float16*)malloc((size_t)Kpad*MR*sizeof(_Float16));   /* P packed [seq][12] for PV */
        #pragma omp for schedule(dynamic)
        for(int qt=0;qt<QT;qt++){
            /* pack Q: Aq[d*12+r] = Q[qt*12+r][d] */
            for(int d=0;d<hd;d++) for(int r=0;r<MR;r++){ int qi=qt*MR+r; Aq[(size_t)d*MR+r]= qi<N? Q[(size_t)qi*hd+d]:(_Float16)0; }
            /* QK^T per key-tile -> sc[12][Kpad] */
            for(int kt=0;kt<KT;kt++){
                float Ctile[MR*NR];
                micro_kernel_fp16_12x2_swp(Aq, Bqk+(size_t)kt*hd*NR, Ctile, hd, 0, (int64_t)NR*4);
                for(int r=0;r<MR;r++) for(int c=0;c<NR;c++) sc[(size_t)r*Kpad + kt*NR + c]=Ctile[r*NR+c];
            }
            /* causal mask + softmax per row, then pack P transposed into Pp[k*12+r] */
            for(int r=0;r<MR;r++){
                int qi=qt*MR+r; if(qi>=N){ for(int k=0;k<Kpad;k++) Pp[(size_t)k*MR+r]=(_Float16)0; continue; }
                int start=0; if(win&&qi+1>win) start=qi-win+1;
                float*row=sc+(size_t)r*Kpad; int last=qi+1; float inv;
                if(scalar_exp){
                    float mx=-1e30f; for(int k=start;k<last;k++) if(row[k]>mx)mx=row[k];
                    float sum=0; for(int k=start;k<last;k++){ row[k]=expf(row[k]-mx); sum+=row[k]; }
                    inv=1.0f/sum;
                } else {
                    /* vectorized: max-reduce, FEXPA exp in place, sum-reduce over [start,last) */
                    int vl=(int)svcntw();
                    svfloat32_t vmax=svdup_f32(-1e30f);
                    for(int k=start;k<last;k+=vl){ svbool_t pg=svwhilelt_b32(k,last); vmax=svmax_m(pg,vmax,svld1(pg,row+k)); }
                    svfloat32_t vmx=svdup_f32(svmaxv(svptrue_b32(),vmax)); svfloat32_t vsum=svdup_f32(0.0f);
                    for(int k=start;k<last;k+=vl){ svbool_t pg=svwhilelt_b32(k,last);
                        svfloat32_t p=sve_expf(pg, svsub_x(pg, svld1(pg,row+k), vmx));
                        svst1(pg,row+k,p); vsum=svadd_m(pg,vsum,p); }
                    inv=1.0f/svaddv(svptrue_b32(),vsum);
                }
                for(int k=0;k<Kpad;k++){ float p=(k>=start&&k<last)? row[k]*inv : 0.0f; Pp[(size_t)k*MR+r]=(_Float16)p; }
            }
            /* P*V: out[12][hd], contract over k(seq) in blocks of KC=NR via init/accum */
            for(int dt=0;dt<hdt;dt++){
                float Otile[MR*NR];
                const _Float16*bv=Bv+(size_t)dt*Kpad*NR;
                micro_kernel_fp16_12x2_swp(Pp, bv, Otile, NR, 0, (int64_t)NR*4);
                for(int kb=1;kb<KT;kb++)
                    micro_kernel_fp16_12x2_swp_accum(Pp+(size_t)kb*NR*MR, bv+(size_t)kb*NR*NR, Otile, NR, 0, (int64_t)NR*4);
                for(int r=0;r<MR;r++){ int qi=qt*MR+r; if(qi<N) for(int c=0;c<NR;c++) Ofu[(size_t)qi*hd+dt*NR+c]=Otile[r*NR+c]; }
            }
        }
        free(Aq); free(sc); free(Pp);
      }
      if(rep==reps-1){ c1=rdcyc();
#ifdef USE_FAPP
        fapp_stop("attn_fused",1,0);
#endif
      }
    }
    double per=(double)(c1-c0)/freq/reps;

    /* accuracy */
    double num=0,den=0;
    for(int i=0;i<N;i++) for(int d=0;d<hd;d++){ double e=(double)Ofu[(size_t)i*hd+d]-(double)Oref[(size_t)i*hd+d]; num+=e*e; den+=(double)Oref[(size_t)i*hd+d]*(double)Oref[(size_t)i*hd+d]; }
    double rel=den>0?sqrt(num/den):0;
    /* QK^T flops = 2*N*seq*hd (causal ~half), PV same; count full for throughput upper bound */
    double flop=2.0*2.0*(double)N*seq*hd;
    printf("[attn-fused] hd=%d N=%d seq=%d win=%d  time=%.3f ms  %.1f GFLOP/s  relL2=%.5f\n",
           hd,N,seq,win,per*1000, flop/per/1e9, rel);
    return 0;
}
