/* pc_matvec.c - Producer/Consumer Q4_0 dequant+matvec on one A64FX CMG.
 *
 * Pattern (A): D producer cores stream repacked Q4_0 (dense nibbles WQS +
 * int8 scale DI, 0.53 B/wt) from HBM and dequant -> int8 into a small L2-
 * resident ring; M consumer cores SDOT the int8 from L2 (no HBM, no dequant).
 * The CMG then reads only the Q4_0 footprint from HBM; dequant compute is
 * parallelized off the matvec cores. Sweep D:M.
 *
 * Producers: NEON dequant (low-latency vand/vshr/vsub/vmul; A64FX has NO NEON
 * dotprod so the DOT must stay SVE) OR SVE dequant. Consumers: SVE svdot.
 *
 * Build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp \
 *           -D_GNU_SOURCE -I../../common pc_matvec.c -lm -o pc_matvec
 * Run:   OMP_PROC_BIND=close OMP_PLACES=cores ./pc_matvec [rows cols]
 */
#define TRANSFORMER_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION
#define GGUF_LOADER_IMPLEMENTATION
#include "transformer.h"
#include <arm_neon.h>
#include <omp.h>

static inline uint64_t rdcyc(void){ uint64_t v; __asm__ volatile("mrs %0, cntvct_el0":"=r"(v)); return v; }
static inline uint64_t rdfreq(void){ uint64_t v; __asm__ volatile("mrs %0, cntfrq_el0":"=r"(v)); return v; }
static uint16_t f32_to_f16(float f){ uint32_t b; memcpy(&b,&f,4); uint32_t s=(b>>16)&0x8000; int e=(int)((b>>23)&0xFF)-127+15; uint32_t m=b&0x7FFFFF;
    if(e<=0){ if(e<-10)return(uint16_t)s; m|=0x800000; return(uint16_t)(s|(m>>(14-e))); } if(e>=31)return(uint16_t)(s|0x7C00); return(uint16_t)(s|((uint32_t)e<<10)|(m>>13)); }

/* --- NEON dequant: 8 rows of one group -> int8 buf (row r at buf+r*cols,
 * block b: lo16 at b*32, hi16 at b*32+16; natural element order). --- */
static inline void dequant8_neon(int8_t *buf, const uint8_t *qs, const int8_t *di, int nb, int cols){
    uint8x16_t mask=vdupq_n_u8(0x0f); int8x16_t v8=vdupq_n_s8(8);
    for(int r=0;r<8;r++){
        const uint8_t *qp=qs+(size_t)r*nb*16; const int8_t *dp=di+(size_t)r*nb; int8_t *o=buf+(size_t)r*cols;
        for(int b=0;b<nb;b++){
            uint8x16_t q=vld1q_u8(qp+(size_t)b*16); int8x16_t vd=vdupq_n_s8(dp[b]);
            int8x16_t lo=vmulq_s8(vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q,mask)),v8),vd);
            int8x16_t hi=vmulq_s8(vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q,4)),v8),vd);
            vst1q_s8(o+(size_t)b*32, lo); vst1q_s8(o+(size_t)b*32+16, hi);
        }
    }
}
/* --- SVE dequant variant (same output) for comparison --- */
static inline void dequant8_sve(int8_t *buf, const uint8_t *qs, const int8_t *di, int nb, int cols){
    svbool_t pg16=svwhilelt_b8((uint32_t)0,(uint32_t)16);
    for(int r=0;r<8;r++){
        const uint8_t *qp=qs+(size_t)r*nb*16; const int8_t *dp=di+(size_t)r*nb; int8_t *o=buf+(size_t)r*cols;
        for(int b=0;b<nb;b++){
            svuint8_t q=svld1_u8(pg16,qp+(size_t)b*16);
            svint8_t lo=svmul_n_s8_x(pg16,svsub_n_s8_x(pg16,svreinterpret_s8_u8(svand_n_u8_x(pg16,q,0x0f)),8),dp[b]);
            svint8_t hi=svmul_n_s8_x(pg16,svsub_n_s8_x(pg16,svreinterpret_s8_u8(svlsr_n_u8_x(pg16,q,4)),8),dp[b]);
            svst1_s8(pg16,o+(size_t)b*32,lo); svst1_s8(pg16,o+(size_t)b*32+16,hi);
        }
    }
}
/* --- SVE consumer: 8-row int8 matvec (temporal loads won the A/B) --- */
static inline void consume8(int32_t acc[8], const int8_t *buf, int nbp, const int8_t *xi8){
    svbool_t pg=svptrue_b8();
    svint32_t a0=svdup_s32(0),a1=svdup_s32(0),a2=svdup_s32(0),a3=svdup_s32(0);
    svint32_t a4=svdup_s32(0),a5=svdup_s32(0),a6=svdup_s32(0),a7=svdup_s32(0);
    size_t rs=(size_t)nbp*64;
    const int8_t*w0=buf,*w1=buf+rs,*w2=buf+2*rs,*w3=buf+3*rs,*w4=buf+4*rs,*w5=buf+5*rs,*w6=buf+6*rs,*w7=buf+7*rs;
    for(int p=0;p<nbp;p++){ svint8_t xv=svld1_s8(pg,xi8+(size_t)p*64);
        a0=svdot_s32(a0,svld1_s8(pg,w0+(size_t)p*64),xv); a1=svdot_s32(a1,svld1_s8(pg,w1+(size_t)p*64),xv);
        a2=svdot_s32(a2,svld1_s8(pg,w2+(size_t)p*64),xv); a3=svdot_s32(a3,svld1_s8(pg,w3+(size_t)p*64),xv);
        a4=svdot_s32(a4,svld1_s8(pg,w4+(size_t)p*64),xv); a5=svdot_s32(a5,svld1_s8(pg,w5+(size_t)p*64),xv);
        a6=svdot_s32(a6,svld1_s8(pg,w6+(size_t)p*64),xv); a7=svdot_s32(a7,svld1_s8(pg,w7+(size_t)p*64),xv); }
    svbool_t p3=svptrue_b32();
    acc[0]=svaddv_s32(p3,a0);acc[1]=svaddv_s32(p3,a1);acc[2]=svaddv_s32(p3,a2);acc[3]=svaddv_s32(p3,a3);
    acc[4]=svaddv_s32(p3,a4);acc[5]=svaddv_s32(p3,a5);acc[6]=svaddv_s32(p3,a6);acc[7]=svaddv_s32(p3,a7);
}
static void ref_matvec(float*dst,const block_q4_0*w,size_t rb,const float*x,int rows,int cols){
    int nb=cols/32; for(int r=0;r<rows;r++){ const block_q4_0*row=(const block_q4_0*)((const uint8_t*)w+(size_t)r*rb);
        float s=0; for(int b=0;b<nb;b++){ float d=ggml_fp16_to_fp32(row[b].d);
            for(int j=0;j<16;j++){ s+=d*((row[b].qs[j]&0xf)-8)*x[b*32+j]; s+=d*((row[b].qs[j]>>4)-8)*x[b*32+j+16]; } } dst[r]=s; } }
static double relL2(const float*a,const float*b,int n){ double e=0,s=0; for(int i=0;i<n;i++){ double d=a[i]-b[i]; e+=d*d; s+=(double)b[i]*b[i]; } return sqrt(e/(s+1e-12)); }

/* ring state (file-scope so the parallel region sees it) */
#define NSLOT 48
static int8_t *slotbuf[NSLOT];
static volatile int prod_turn[NSLOT];   /* group allowed to produce into slot */
static volatile long slot_full[NSLOT];  /* group currently filled (-1 empty) */

int main(int argc,char**argv){
    int rows=argc>2?atoi(argv[1]):21504, cols=argc>2?atoi(argv[2]):5376;
    double freq=(double)rdfreq();
    int nb=cols/32, nbp=(cols+63)/64;
    size_t row_bytes=(size_t)nb*sizeof(block_q4_0);
    block_q4_0 *W=(block_q4_0*)aligned_alloc(256,(size_t)rows*row_bytes);
    float *x=(float*)aligned_alloc(256,(size_t)cols*4),*dref=(float*)aligned_alloc(256,(size_t)rows*4),*dP=(float*)aligned_alloc(256,(size_t)rows*4);
    srand(42); for(size_t i=0;i<(size_t)rows*nb;i++){ float d=0.01f+(rand()/(float)RAND_MAX)*0.5f; W[i].d=f32_to_f16(d); for(int j=0;j<16;j++) W[i].qs[j]=(unsigned char)(rand()&0xFF); }
    srand(137); for(int i=0;i<cols;i++) x[i]=(rand()/(float)RAND_MAX-0.5f)*2.0f;
    ref_matvec(dref,W,row_bytes,x,rows,cols);
    float max_d=0; for(size_t i=0;i<(size_t)rows*nb;i++){ float a=fabsf(ggml_fp16_to_fp32(W[i].d)); if(a>max_d)max_d=a; }
    float scale_w=max_d>0?127.0f/(8.0f*max_d):1.0f;
    /* repacked dense nibbles + int8 scale */
    uint8_t *WQS=(uint8_t*)aligned_alloc(256,(size_t)rows*nb*16); int8_t *DI=(int8_t*)aligned_alloc(256,(size_t)rows*nb);
    #pragma omp parallel for schedule(static)
    for(int r=0;r<rows;r++){ const block_q4_0*row=(const block_q4_0*)((const uint8_t*)W+(size_t)r*row_bytes);
        for(int b=0;b<nb;b++){ memcpy(WQS+((size_t)r*nb+b)*16,row[b].qs,16); int di=lrintf(ggml_fp16_to_fp32(row[b].d)*scale_w); if(di>127)di=127; if(di<-128)di=-128; DI[(size_t)r*nb+b]=(int8_t)di; } }
    int8_t *xi8=(int8_t*)aligned_alloc(256,(size_t)nbp*64); float x_inv; tf_quantize_f32_to_int8(x,xi8,cols,&x_inv);
    float inv=1.0f/(scale_w*x_inv);
    for(int s=0;s<NSLOT;s++) slotbuf[s]=(int8_t*)aligned_alloc(256,(size_t)8*nbp*64);
    int G=rows/8;
    size_t rbytes=(size_t)rows*nb*17;

    printf("Producer/Consumer Q4_0 matvec  rows=%d cols=%d  G=%d groups, %d L2 slots\n",rows,cols,G,NSLOT);
    printf("  reads %.1f MB Q4_0 (0.53 B/wt); ceilings 230 GB/s/CMG, 852/node\n",rbytes/1048576.0);
    printf("  %-12s %9s %9s %9s %8s %s\n","D:M (neon)","ms","GB/s","GIOPS","%CMG","ok");
    int cfgs[][2]={{4,8},{6,6},{8,4},{3,9},{9,3},{2,10},{10,2}};
    for(int useneon=1;useneon>=0;useneon--){
      printf(" -- producers: %s dequant --\n", useneon?"NEON":"SVE");
      for(int ci=0;ci<7;ci++){
        int D=cfgs[ci][0], M=cfgs[ci][1], nt=D+M;
        int reps=6; double best=1e30;
        for(int rep=0;rep<reps;rep++){
            for(int s=0;s<NSLOT;s++){ prod_turn[s]=s; slot_full[s]=-1; }
            int prod_seq=0, cons_seq=0;
            uint64_t t0=rdcyc();
            #pragma omp parallel num_threads(nt)
            {
                int tid=omp_get_thread_num();
                if(tid<D){ /* producer */
                    for(;;){ int g=__atomic_fetch_add(&prod_seq,1,__ATOMIC_RELAXED); if(g>=G)break; int s=g%NSLOT;
                        while(__atomic_load_n(&prod_turn[s],__ATOMIC_ACQUIRE)!=g) ;
                        const uint8_t*qs=WQS+(size_t)g*8*nb*16; const int8_t*di=DI+(size_t)g*8*nb;
                        if(useneon) dequant8_neon(slotbuf[s],qs,di,nb,cols); else dequant8_sve(slotbuf[s],qs,di,nb,cols);
                        __atomic_store_n(&slot_full[s],g,__ATOMIC_RELEASE);
                    }
                } else { /* consumer */
                    for(;;){ int c=__atomic_fetch_add(&cons_seq,1,__ATOMIC_RELAXED); if(c>=G)break; int s=c%NSLOT;
                        while(__atomic_load_n(&slot_full[s],__ATOMIC_ACQUIRE)!=c) ;
                        int32_t acc[8]; consume8(acc,slotbuf[s],nbp,xi8);
                        for(int i=0;i<8;i++) dP[c*8+i]=(float)acc[i]*inv;
                        __atomic_store_n(&prod_turn[s],c+NSLOT,__ATOMIC_RELEASE);
                    }
                }
            }
            double sec=(double)(rdcyc()-t0)/freq; if(sec<best)best=sec;
        }
        double gbps=(double)rbytes/best/1e9, gi=2.0*rows*cols/best/1e9;
        double mr=(ci==0)?relL2(dP,dref,rows):relL2(dP,dref,rows);
        printf("  %2d:%-9d %9.3f %9.1f %9.1f %7.0f%% %s\n",D,M,best*1000,gbps,gi,gbps/230.0*100, mr<0.06?"OK":"FAIL");
      }
    }
    return 0;
}
