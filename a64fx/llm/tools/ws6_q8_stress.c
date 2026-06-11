/* WS6 Q8 small-M NaN reproduction harness (single node, no alloc).
 * The bug: ds4f_forward_verify / DS4F_GEMM_DECODE under DS4F_Q8_DENSE NaNs nondeterministically
 * at small M (1-2 verify tokens) -- the doc hypothesis is the Q8 single-token remainder kernel
 * matvec_sdot_8row. ds4f_gemm_test already covers Q8 M=1,2 and passes, so this harness stresses
 * the SUSPECTED triggers gemm_test doesn't: (a) TLS xscratch reuse across decreasing K with M
 * varying, (b) interleaved single-token ds4f_matvec (xscratch(1,K)) between batched ds4f_gemm,
 * (c) adversarial X (huge / tiny-denormal / zero / mixed), (d) many threads + many iters.
 * Any NaN/inf in the Q8 output, or a relL2 blow-up vs the bf16 reference, is the repro.
 * Build:
 *   fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -D_GNU_SOURCE \
 *       -I../../common -o /tmp/ws6 tools/ws6_q8_stress.c -lm -lpthread -lhwb
 *   OMP_NUM_THREADS=12 taskset -c 12-23 /tmp/ws6
 */
#include "ds4f.h"
#include <math.h>

static uint32_t rng = 0xC0FFEEu;
static inline uint32_t nr(void){ rng = rng*1664525u + 1013904223u; return rng; }
static inline float frand(void){ return ((float)(nr()>>8)/(float)(1u<<24))*2.f-1.f; }

/* pack row-major bf16 -> DS4F_BF16_PV pair-interleaved (mirror gemm_test pack_pv) */
static void pack_pv(uint16_t *dst, const uint16_t *Wrm, int rows, int cols){
    for(int i=0;i<rows;i++){ size_t gb=(size_t)(i/8)*8*cols; int loc=i&7,pair=loc>>1,slot=loc&1;
        uint16_t *pb=dst+gb+(size_t)pair*2*cols; for(int j=0;j<cols;j++) pb[2*j+slot]=Wrm[(size_t)i*cols+j]; }
}

typedef struct { ds4f_tensor q8, bf; int rows, cols; float *Yref1; } shape_t;  /* Yref1 = bf16 ref for 1 token of Xref */

/* fill X[M][cols] with an adversarial pattern selected by `mode` */
static void fill_X(float *X, int M, int cols, int mode){
    for(int m=0;m<M;m++)for(int j=0;j<cols;j++){
        float v;
        switch(mode){
            case 0: v=frand(); break;                                   /* normal */
            case 1: v=frand()*1e6f; break;                              /* huge */
            case 2: v=frand()*1e-30f; break;                            /* tiny/denormal */
            case 3: v=(j%64==0)?frand()*1e8f:0.f; break;               /* sparse spike (one big per 64-block) */
            case 4: v=0.f; break;                                       /* all zero */
            default: v=(m&1)?frand()*1e7f:frand()*1e-20f; break;        /* mixed scale per token */
        }
        X[(size_t)m*cols+j]=v;
    }
}

static int has_bad(const float *Y, size_t n){
    int bad=0; for(size_t i=0;i<n;i++){ float a=Y[i]; if(!(a==a)||a>3e38f||a<-3e38f) bad++; } return bad;
}

int main(int argc,char**argv){
    int nthr=(argc>1)?atoi(argv[1]):12, ncmg=(argc>2)?atoi(argv[2]):1;
    ds4f_model m; memset(&m,0,sizeof(m)); m.n_threads=nthr; m.n_cmgs=ncmg;
    m.pool=ds4f_pool_start(nthr,ncmg); ds4f_init_fp8_e4m3_lut(m.fp8_lut);
    printf("ws6_q8_stress nthr=%d\n",nthr);

    /* shapes ordered to STRESS scratch reuse: large K first (grows xscratch), then small, then mid */
    struct{int r,c;} S[]={{4096,8192},{32768,1024},{1024,4096},{8192,4096},{512,4096},{4096,2048}};
    int ns=sizeof(S)/sizeof(S[0]);
    shape_t *sh=calloc(ns,sizeof(shape_t));
    for(int s=0;s<ns;s++){ int rows=S[s].r,cols=S[s].c; sh[s].rows=rows; sh[s].cols=cols;
        uint16_t *Wrm=malloc((size_t)rows*cols*2); for(size_t k=0;k<(size_t)rows*cols;k++) Wrm[k]=ds4f_f32_bf16(frand());
        /* bf16-pv tensor (kept) + q8 tensor (repacked from a copy) */
        ds4f_tensor bf; memset(&bf,0,sizeof(bf)); bf.type=DS4F_BF16_PV; bf.rows=rows; bf.cols=cols;
        size_t wb=ds4f_wbytes(DS4F_BF16_PV,rows,cols); bf.w=aligned_alloc(256,(wb+255)&~(size_t)255);
        pack_pv((uint16_t*)bf.w,Wrm,rows,cols);
        ds4f_tensor q8=bf; q8.w=aligned_alloc(256,(wb+255)&~(size_t)255); memcpy(q8.w,bf.w,wb);
        ds4f_repack_bf16pv_to_q8pv(&m,&q8);
        sh[s].bf=bf; sh[s].q8=q8; free(Wrm);
    }

    int ITERS=argc>3?atoi(argv[3]):2000;
    int Ms[]={1,2,3,1,2}; int totalbad=0; double maxrel=0;
    float *Xbig=aligned_alloc(256,(size_t)4*8192*4);    /* up to M=4, cols<=8192 */
    float *Yq=aligned_alloc(256,(size_t)4*32768*4);
    float *Yb=aligned_alloc(256,(size_t)4*32768*4);
    for(int it=0; it<ITERS; it++){
        int M=Ms[it%5];
        for(int s=0;s<ns;s++){
            int rows=sh[s].rows, cols=sh[s].cols;
            int mode = (it*7+s)%6;
            fill_X(Xbig,M,cols,mode);
            /* interleave a single-token decode matvec on the SAME thread-pool scratch */
            if((it+s)&1){ float yt[32768]; ds4f_matvec(&m,yt,&sh[s].q8,Xbig); (void)yt; }
            ds4f_gemm(&m,Yq,&sh[s].q8,Xbig,M,rows,cols);
            int bad=has_bad(Yq,(size_t)M*rows);
            if(bad){ totalbad+=bad;
                if(totalbad<=bad) printf("  !! NaN/inf: iter=%d shape=[%d,%d] M=%d mode=%d bad=%d\n",it,rows,cols,M,mode,bad);
            }
            /* correctness vs bf16 gemm (same K-tile reassoc family) for the NORMAL mode only */
            if(mode==0){
                ds4f_gemm(&m,Yb,&sh[s].bf,Xbig,M,rows,cols);
                double e=0,n=0; for(size_t i=0;i<(size_t)M*rows;i++){ double d=Yq[i]-Yb[i]; e+=d*d; n+=(double)Yb[i]*Yb[i]; }
                double rel=n>0?sqrt(e/n):0; if(rel>maxrel)maxrel=rel;
            }
        }
    }
    printf("DONE iters=%d  total NaN/inf outputs=%d  max relL2(q8 vs bf16, normal X)=%.3e\n",ITERS,totalbad,maxrel);
    printf(totalbad? "  -> REPRODUCED\n" : "  -> no NaN reproduced (Q8 small-M kernel robust under this stress)\n");
    ds4f_pool_stop(m.pool);
    return totalbad?2:0;
}
