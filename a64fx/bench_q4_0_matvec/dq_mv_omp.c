/* dq_mv_omp.c - Q4_0 dequant+matvec parallelization patterns vs HBM BW target.
 *
 * A64FX measured ceilings: ~57 GB/s/core, ~230 GB/s/CMG, ~852 GB/s/node.
 * Matvec is memory-bound, so HBM BYTES read decide throughput:
 *   - fused Q4_0  : read 0.5625 B/weight (d + nibbles), dequant in-core
 *   - int8 cache  : read 1.0 B/weight (pre-dequantized)
 * => fused should be ~1.78x fewer HBM bytes IF dequant keeps up with the load.
 *
 * Build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp \
 *           -D_GNU_SOURCE -I../../common dq_mv_omp.c -lm -o dq_mv_omp
 * Run:   OMP_PROC_BIND=close OMP_PLACES=cores ./dq_mv_omp [rows cols]
 */
#define TRANSFORMER_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION
#define GGUF_LOADER_IMPLEMENTATION
#include "transformer.h"
#include <omp.h>
#ifdef NONTEMPORAL_W
#define LDW(p) svldnt1_s8(pg,(p))
#else
#define LDW(p) svld1_s8(pg,(p))
#endif

static inline uint64_t rdcyc(void){ uint64_t v; __asm__ volatile("mrs %0, cntvct_el0":"=r"(v)); return v; }
static inline uint64_t rdfreq(void){ uint64_t v; __asm__ volatile("mrs %0, cntfrq_el0":"=r"(v)); return v; }

static uint16_t f32_to_f16(float f){
    uint32_t b; memcpy(&b,&f,4); uint32_t s=(b>>16)&0x8000; int e=(int)((b>>23)&0xFF)-127+15; uint32_t m=b&0x7FFFFF;
    if(e<=0){ if(e<-10)return(uint16_t)s; m|=0x800000; return(uint16_t)(s|(m>>(14-e))); }
    if(e>=31)return(uint16_t)(s|0x7C00); return(uint16_t)(s|((uint32_t)e<<10)|(m>>13));
}
static void init_q4_0(block_q4_0 *w,int nb,unsigned seed){
    srand(seed);
    for(int i=0;i<nb;i++){ float d=0.01f+(rand()/(float)RAND_MAX)*0.5f; w[i].d=f32_to_f16(d);
        for(int j=0;j<16;j++) w[i].qs[j]=(unsigned char)(rand()&0xFF); }
}
static void ref_matvec(float*dst,const block_q4_0*w,size_t rb,const float*x,int rows,int cols){
    int nb=cols/32;
    for(int r=0;r<rows;r++){ const block_q4_0*row=(const block_q4_0*)((const uint8_t*)w+(size_t)r*rb);
        float s=0; for(int b=0;b<nb;b++){ float d=ggml_fp16_to_fp32(row[b].d);
            for(int j=0;j<16;j++){ s+=d*((row[b].qs[j]&0xf)-8)*x[b*32+j];
                                   s+=d*((row[b].qs[j]>>4)-8)*x[b*32+j+16]; } }
        dst[r]=s; }
}
static double relL2(const float*a,const float*b,int n){
    double e=0,s=0; for(int i=0;i<n;i++){ double d=a[i]-b[i]; e+=d*d; s+=(double)b[i]*b[i]; } return sqrt(e/(s+1e-12));
}

/* Lean int8 8-row matvec: non-temporal weight loads + 8 independent
 * accumulators + 2-pair unroll. wi8[r] = WI8 + (g+r)*nbp*64 (contiguous).
 * This is the BW-saturation test: minimal compute, streaming loads. */
static inline void lean_i8_8row(int32_t acc[8], const int8_t *w0,int nbp,const int8_t *xi8){
    svbool_t pg=svptrue_b8();
    svint32_t a0=svdup_s32(0),a1=svdup_s32(0),a2=svdup_s32(0),a3=svdup_s32(0);
    svint32_t a4=svdup_s32(0),a5=svdup_s32(0),a6=svdup_s32(0),a7=svdup_s32(0);
    size_t rs=(size_t)nbp*64;
    const int8_t*w1=w0+rs,*w2=w0+2*rs,*w3=w0+3*rs,*w4=w0+4*rs,*w5=w0+5*rs,*w6=w0+6*rs,*w7=w0+7*rs;
    for(int p=0;p<nbp;p++){
        svint8_t xv=svldnt1_s8(pg,xi8+(size_t)p*64);
        a0=svdot_s32(a0,LDW(w0+(size_t)p*64),xv); a1=svdot_s32(a1,LDW(w1+(size_t)p*64),xv);
        a2=svdot_s32(a2,LDW(w2+(size_t)p*64),xv); a3=svdot_s32(a3,LDW(w3+(size_t)p*64),xv);
        a4=svdot_s32(a4,LDW(w4+(size_t)p*64),xv); a5=svdot_s32(a5,LDW(w5+(size_t)p*64),xv);
        a6=svdot_s32(a6,LDW(w6+(size_t)p*64),xv); a7=svdot_s32(a7,LDW(w7+(size_t)p*64),xv);
    }
    svbool_t p32=svptrue_b32();
    acc[0]=svaddv_s32(p32,a0); acc[1]=svaddv_s32(p32,a1); acc[2]=svaddv_s32(p32,a2); acc[3]=svaddv_s32(p32,a3);
    acc[4]=svaddv_s32(p32,a4); acc[5]=svaddv_s32(p32,a5); acc[6]=svaddv_s32(p32,a6); acc[7]=svaddv_s32(p32,a7);
}

/* Lean FUSED 8-row on a REPACKED layout: dense nibbles WQS (16 B/block, no
 * interleaved d) + precomputed int8 per-block scale DI (=round(d*sw)). Hot
 * loop has ZERO scalar fp math: ldnt(qs) -> unpack -> (v-8)*di -> assemble via
 * svsplice -> SDOT. Reads 0.5 (qs) + 0.03 (di) = 0.53 B/wt. xv shared/8 rows.
 * qsbase[r] = WQS + (g+r)*nb*16; dibase[r] = DI + (g+r)*nb.  cols mult of 64. */
static inline void lean_fused_8row(int32_t acc[8], const uint8_t *qsbase, const int8_t *dibase,
                                   int nb, int cols, const int8_t *xi8){
    svbool_t pg=svptrue_b8(), pg16=svwhilelt_b8((uint32_t)0,(uint32_t)16), pg32=svwhilelt_b8((uint32_t)0,(uint32_t)32);
    svint32_t a0=svdup_s32(0),a1=svdup_s32(0),a2=svdup_s32(0),a3=svdup_s32(0);
    svint32_t a4=svdup_s32(0),a5=svdup_s32(0),a6=svdup_s32(0),a7=svdup_s32(0);
    int pairs=cols/64;
#define DQ(A,R) do { \
    const uint8_t *qp=qsbase+(size_t)(R)*nb*16+(size_t)(2*p)*16; const int8_t *dp=dibase+(size_t)(R)*nb+2*p; \
    svuint8_t q0=svldnt1_u8(pg16,qp), q1=svldnt1_u8(pg16,qp+16); \
    svint8_t lo0=svreinterpret_s8_u8(svand_n_u8_x(pg16,q0,0x0f)); \
    svint8_t hi0=svreinterpret_s8_u8(svlsr_n_u8_x(pg16,q0,4)); \
    svint8_t lo1=svreinterpret_s8_u8(svand_n_u8_x(pg16,q1,0x0f)); \
    svint8_t hi1=svreinterpret_s8_u8(svlsr_n_u8_x(pg16,q1,4)); \
    svint8_t z0=svsplice_s8(pg16,lo0,hi0); z0=svmul_n_s8_x(pg,svsub_n_s8_x(pg,z0,8),dp[0]); \
    svint8_t z1=svsplice_s8(pg16,lo1,hi1); z1=svmul_n_s8_x(pg,svsub_n_s8_x(pg,z1,8),dp[1]); \
    svint8_t w=svsplice_s8(pg32,z0,z1); \
    A=svdot_s32(A,w,svldnt1_s8(pg,xi8+(size_t)p*64)); } while(0)
    for(int p=0;p<pairs;p++){
        DQ(a0,0);DQ(a1,1);DQ(a2,2);DQ(a3,3);DQ(a4,4);DQ(a5,5);DQ(a6,6);DQ(a7,7);
    }
#undef DQ
    svbool_t p3=svptrue_b32();
    acc[0]=svaddv_s32(p3,a0);acc[1]=svaddv_s32(p3,a1);acc[2]=svaddv_s32(p3,a2);acc[3]=svaddv_s32(p3,a3);
    acc[4]=svaddv_s32(p3,a4);acc[5]=svaddv_s32(p3,a5);acc[6]=svaddv_s32(p3,a6);acc[7]=svaddv_s32(p3,a7);
}

int main(int argc,char**argv){
    int rows=argc>2?atoi(argv[1]):21504;
    int cols=argc>2?atoi(argv[2]):5376;
    double freq=(double)rdfreq();
    int nbp=(cols+63)/64;
    size_t row_bytes=(size_t)(cols/32)*sizeof(block_q4_0);
    size_t q4bytes=(size_t)rows*row_bytes;
    size_t i8bytes=(size_t)rows*nbp*64;

    block_q4_0 *W=(block_q4_0*)aligned_alloc(256,q4bytes);
    int8_t *WI8=(int8_t*)aligned_alloc(256,i8bytes);
    float *x=(float*)aligned_alloc(256,(size_t)cols*sizeof(float));
    float *dref=(float*)aligned_alloc(256,(size_t)rows*sizeof(float));
    float *dA=(float*)aligned_alloc(256,(size_t)rows*sizeof(float));
    float *dI=(float*)aligned_alloc(256,(size_t)rows*sizeof(float));
    init_q4_0(W,(int)(q4bytes/sizeof(block_q4_0)),42);
    srand(137); for(int i=0;i<cols;i++) x[i]=(rand()/(float)RAND_MAX-0.5f)*2.0f;
    ref_matvec(dref,W,row_bytes,x,rows,cols);

    float max_d=0; { int n=(int)(q4bytes/sizeof(block_q4_0));
        for(int i=0;i<n;i++){ float a=fabsf(ggml_fp16_to_fp32(W[i].d)); if(a>max_d)max_d=a; } }
    float scale_w=max_d>0?127.0f/(8.0f*max_d):1.0f;
    #pragma omp parallel for schedule(static)
    for(int g=0;g<rows;g+=8)
        tf_dequant_q4_0_8row_strided_to_int8((const uint8_t*)W+(size_t)g*row_bytes,row_bytes,
                                             WI8+(size_t)g*nbp*64,cols,scale_w);
    /* Repacked fused layout: dense nibbles WQS (16 B/block) + int8 scale DI. */
    int nb=cols/32;
    uint8_t *WQS=(uint8_t*)aligned_alloc(256,(size_t)rows*nb*16);
    int8_t  *DI =(int8_t *)aligned_alloc(256,(size_t)rows*nb);
    size_t rbytes=(size_t)rows*nb*17;   /* HBM bytes for repacked fused */
    #pragma omp parallel for schedule(static)
    for(int r=0;r<rows;r++){ const block_q4_0*row=(const block_q4_0*)((const uint8_t*)W+(size_t)r*row_bytes);
        for(int b=0;b<nb;b++){ memcpy(WQS+((size_t)r*nb+b)*16,row[b].qs,16);
            int di=lrintf(ggml_fp16_to_fp32(row[b].d)*scale_w); if(di>127)di=127; if(di<-128)di=-128;
            DI[(size_t)r*nb+b]=(int8_t)di; } }
    int8_t *xi8=(int8_t*)aligned_alloc(256,(size_t)nbp*64);
    float x_inv; tf_quantize_f32_to_int8(x,xi8,cols,&x_inv);
    float inv=1.0f/(scale_w*x_inv);

    printf("Q4_0 dequant+matvec patterns vs HBM BW  (rows=%d cols=%d)\n",rows,cols);
    printf("  fused(repacked) reads %.1f MB (%.4f B/wt); int8 reads %.1f MB (%.4f B/wt)\n",
           rbytes/1048576.0,(double)rbytes/((double)rows*cols),
           i8bytes/1048576.0,(double)i8bytes/((double)rows*cols));
    printf("  ceilings: 57 GB/s/core, 230 GB/s/CMG, 852 GB/s/node\n\n");

    int threads[]={1,2,4,8,12,24,48};
    double opsf=2.0*rows*cols;
    int reps=(q4bytes>32u*1024*1024)?8:30;

    for(int pass=0;pass<2;pass++){
        printf("[%s] reads %s\n", pass==0?"A  FUSED Q4_0":"I8 int8 cache",
                                  pass==0?"Q4_0 (0.5625 B/wt)":"int8 (1.0 B/wt)");
        printf("    %-4s %9s %9s %9s %8s\n","thr","ms","GB/s","GIOPS","%node");
        for(int ti=0;ti<7;ti++){ int nt=threads[ti];
            int8_t **scr=(int8_t**)calloc(nt,sizeof(int8_t*));
            #pragma omp parallel num_threads(nt)
            { scr[omp_get_thread_num()]=(int8_t*)aligned_alloc(256,(size_t)8*nbp*64); }
            uint64_t t0=rdcyc();
            for(int rep=0;rep<reps;rep++){
                #pragma omp parallel for num_threads(nt) schedule(static)
                for(int g=0;g<rows;g+=8){
                    if(pass==0){
                        int32_t acc[8];
                        lean_fused_8row(acc,WQS+(size_t)g*nb*16,DI+(size_t)g*nb,nb,cols,xi8);
                        for(int i=0;i<8;i++) dA[g+i]=(float)acc[i]*inv;
                    } else {
                        int32_t acc[8]; lean_i8_8row(acc,WI8+(size_t)g*nbp*64,nbp,xi8);
                        for(int i=0;i<8;i++) dI[g+i]=(float)acc[i]*inv;
                    }
                }
            }
            uint64_t t1=rdcyc(); double sec=(double)(t1-t0)/freq;
            double bytes=(double)(pass==0?rbytes:i8bytes)*reps;
            double gbps=bytes/sec/1e9, gi=opsf*reps/sec/1e9, ms=sec/reps*1000;
            printf("    %-4d %9.3f %9.1f %9.1f %7.0f%%\n",nt,ms,gbps,gi,gbps/852.0*100);
            for(int t=0;t<nt;t++) free(scr[t]); free(scr);
        }
        double mr=relL2(pass==0?dA:dI,dref,rows);
        printf("    correctness maxrel vs fp32 ref = %.4f  %s\n\n",mr,mr<0.06?"OK":"FAIL");
    }
    free(W);free(WI8);free(WQS);free(DI);free(x);free(dref);free(dA);free(dI);free(xi8);
    return 0;
}
