/* gemm_shapes.c - int8 SDOT GEMM on a real weight shape, swept over M
 * (number of activation columns). M=1 is matvec (memory-bound); larger M
 * reuses each streamed weight M times => compute-bound, approaches peak.
 *
 *   out[r][m] = sum_k W_i8[r][k] * X_i8[m][k]     (k = contraction = cols)
 *
 * W (rows*cols int8) streams from HBM; X (M*cols int8) stays L2-resident,
 * so arithmetic intensity w.r.t. the HBM weight traffic = 2*M ops/byte.
 *
 * Build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast \
 *           -D_GNU_SOURCE gemm_shapes.c -lm -o gemm_shapes
 * Run:   ./gemm_shapes <rows> <cols>
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <arm_sve.h>

static inline uint64_t rdcyc(void){ uint64_t v; __asm__ volatile("mrs %0, cntvct_el0":"=r"(v)); return v; }
static inline uint64_t rdfreq(void){ uint64_t v; __asm__ volatile("mrs %0, cntfrq_el0":"=r"(v)); return v; }
#define OPS_PER_SDOT 128.0
#define LD(p,k) svld1_s8(svptrue_b8(),(p)+(size_t)(k)*64)

/* M=1: matvec. 8 accumulators down the contraction for ILP. */
static double gemm_m1(double freq,const int8_t*W,const int8_t*X,int rows,int cols,int reps){
    int kb=cols/64; uint64_t t0=rdcyc(); volatile long sink=0;
    for(int rep=0;rep<reps;rep++) for(int r=0;r<rows;r++){
        const int8_t*w=W+(size_t)r*cols;
        svint32_t a0=svdup_s32(0),a1=svdup_s32(0),a2=svdup_s32(0),a3=svdup_s32(0);
        svint32_t a4=svdup_s32(0),a5=svdup_s32(0),a6=svdup_s32(0),a7=svdup_s32(0);
        int k=0;
        for(;k+8<=kb;k+=8){
            a0=svdot_s32(a0,LD(w,k+0),LD(X,k+0)); a1=svdot_s32(a1,LD(w,k+1),LD(X,k+1));
            a2=svdot_s32(a2,LD(w,k+2),LD(X,k+2)); a3=svdot_s32(a3,LD(w,k+3),LD(X,k+3));
            a4=svdot_s32(a4,LD(w,k+4),LD(X,k+4)); a5=svdot_s32(a5,LD(w,k+5),LD(X,k+5));
            a6=svdot_s32(a6,LD(w,k+6),LD(X,k+6)); a7=svdot_s32(a7,LD(w,k+7),LD(X,k+7));
        }
        for(;k<kb;k++) a0=svdot_s32(a0,LD(w,k),LD(X,k));
        svbool_t pg=svptrue_b32();
        svint32_t s=svadd_s32_x(pg,svadd_s32_x(pg,svadd_s32_x(pg,a0,a1),svadd_s32_x(pg,a2,a3)),
                                   svadd_s32_x(pg,svadd_s32_x(pg,a4,a5),svadd_s32_x(pg,a6,a7)));
        sink+=svaddv_s32(pg,s);
    }
    uint64_t t1=rdcyc(); (void)sink;
    return (double)rows*kb*reps*OPS_PER_SDOT/((double)(t1-t0)/freq)/1e9;
}

/* M=4: weight vector loaded once per k, reused across 4 activation columns. */
static double gemm_m4(double freq,const int8_t*W,const int8_t*X,int rows,int cols,int reps){
    int kb=cols/64; uint64_t t0=rdcyc(); volatile long sink=0;
    const int8_t*x0=X,*x1=X+(size_t)1*cols,*x2=X+(size_t)2*cols,*x3=X+(size_t)3*cols;
    for(int rep=0;rep<reps;rep++) for(int r=0;r<rows;r++){
        const int8_t*w=W+(size_t)r*cols;
        svint32_t c0=svdup_s32(0),c1=svdup_s32(0),c2=svdup_s32(0),c3=svdup_s32(0);
        for(int k=0;k<kb;k++){ svint8_t wv=LD(w,k);
            c0=svdot_s32(c0,wv,LD(x0,k)); c1=svdot_s32(c1,wv,LD(x1,k));
            c2=svdot_s32(c2,wv,LD(x2,k)); c3=svdot_s32(c3,wv,LD(x3,k)); }
        svbool_t pg=svptrue_b32();
        sink+=svaddv_s32(pg,c0)+svaddv_s32(pg,c1)+svaddv_s32(pg,c2)+svaddv_s32(pg,c3);
    }
    uint64_t t1=rdcyc(); (void)sink;
    return (double)rows*kb*4*reps*OPS_PER_SDOT/((double)(t1-t0)/freq)/1e9;
}

/* M=8 */
static double gemm_m8(double freq,const int8_t*W,const int8_t*X,int rows,int cols,int reps){
    int kb=cols/64; uint64_t t0=rdcyc(); volatile long sink=0;
    const int8_t*x0=X,*x1=X+(size_t)1*cols,*x2=X+(size_t)2*cols,*x3=X+(size_t)3*cols;
    const int8_t*x4=X+(size_t)4*cols,*x5=X+(size_t)5*cols,*x6=X+(size_t)6*cols,*x7=X+(size_t)7*cols;
    for(int rep=0;rep<reps;rep++) for(int r=0;r<rows;r++){
        const int8_t*w=W+(size_t)r*cols;
        svint32_t c0=svdup_s32(0),c1=svdup_s32(0),c2=svdup_s32(0),c3=svdup_s32(0);
        svint32_t c4=svdup_s32(0),c5=svdup_s32(0),c6=svdup_s32(0),c7=svdup_s32(0);
        for(int k=0;k<kb;k++){ svint8_t wv=LD(w,k);
            c0=svdot_s32(c0,wv,LD(x0,k)); c1=svdot_s32(c1,wv,LD(x1,k));
            c2=svdot_s32(c2,wv,LD(x2,k)); c3=svdot_s32(c3,wv,LD(x3,k));
            c4=svdot_s32(c4,wv,LD(x4,k)); c5=svdot_s32(c5,wv,LD(x5,k));
            c6=svdot_s32(c6,wv,LD(x6,k)); c7=svdot_s32(c7,wv,LD(x7,k)); }
        svbool_t pg=svptrue_b32();
        sink+=svaddv_s32(pg,c0)+svaddv_s32(pg,c1)+svaddv_s32(pg,c2)+svaddv_s32(pg,c3)
             +svaddv_s32(pg,c4)+svaddv_s32(pg,c5)+svaddv_s32(pg,c6)+svaddv_s32(pg,c7);
    }
    uint64_t t1=rdcyc(); (void)sink;
    return (double)rows*kb*8*reps*OPS_PER_SDOT/((double)(t1-t0)/freq)/1e9;
}

/* M=16 */
static double gemm_m16(double freq,const int8_t*W,const int8_t*X,int rows,int cols,int reps){
    int kb=cols/64; uint64_t t0=rdcyc(); volatile long sink=0;
    const int8_t*xp[16]; for(int m=0;m<16;m++) xp[m]=X+(size_t)m*cols;
    for(int rep=0;rep<reps;rep++) for(int r=0;r<rows;r++){
        const int8_t*w=W+(size_t)r*cols;
        svint32_t c0=svdup_s32(0),c1=svdup_s32(0),c2=svdup_s32(0),c3=svdup_s32(0);
        svint32_t c4=svdup_s32(0),c5=svdup_s32(0),c6=svdup_s32(0),c7=svdup_s32(0);
        svint32_t c8=svdup_s32(0),c9=svdup_s32(0),cA=svdup_s32(0),cB=svdup_s32(0);
        svint32_t cC=svdup_s32(0),cD=svdup_s32(0),cE=svdup_s32(0),cF=svdup_s32(0);
        for(int k=0;k<kb;k++){ svint8_t wv=LD(w,k);
            c0=svdot_s32(c0,wv,LD(xp[ 0],k)); c1=svdot_s32(c1,wv,LD(xp[ 1],k));
            c2=svdot_s32(c2,wv,LD(xp[ 2],k)); c3=svdot_s32(c3,wv,LD(xp[ 3],k));
            c4=svdot_s32(c4,wv,LD(xp[ 4],k)); c5=svdot_s32(c5,wv,LD(xp[ 5],k));
            c6=svdot_s32(c6,wv,LD(xp[ 6],k)); c7=svdot_s32(c7,wv,LD(xp[ 7],k));
            c8=svdot_s32(c8,wv,LD(xp[ 8],k)); c9=svdot_s32(c9,wv,LD(xp[ 9],k));
            cA=svdot_s32(cA,wv,LD(xp[10],k)); cB=svdot_s32(cB,wv,LD(xp[11],k));
            cC=svdot_s32(cC,wv,LD(xp[12],k)); cD=svdot_s32(cD,wv,LD(xp[13],k));
            cE=svdot_s32(cE,wv,LD(xp[14],k)); cF=svdot_s32(cF,wv,LD(xp[15],k)); }
        svbool_t pg=svptrue_b32();
        sink+=svaddv_s32(pg,c0)+svaddv_s32(pg,c1)+svaddv_s32(pg,c2)+svaddv_s32(pg,c3)
             +svaddv_s32(pg,c4)+svaddv_s32(pg,c5)+svaddv_s32(pg,c6)+svaddv_s32(pg,c7)
             +svaddv_s32(pg,c8)+svaddv_s32(pg,c9)+svaddv_s32(pg,cA)+svaddv_s32(pg,cB)
             +svaddv_s32(pg,cC)+svaddv_s32(pg,cD)+svaddv_s32(pg,cE)+svaddv_s32(pg,cF);
    }
    uint64_t t1=rdcyc(); (void)sink;
    return (double)rows*kb*16*reps*OPS_PER_SDOT/((double)(t1-t0)/freq)/1e9;
}

/* 2D register tile: 4 weight-rows x 4 activation-cols = 16 accumulators.
 * Per k-block: 4 W-loads + 4 X-loads = 8 loads feed 16 SDOTs (vs the naive
 * 1xNR kernel's 1+NR loads). Amortizes the load port; W reuse = 4 cols. */
static double gemm_tile44(double freq,const int8_t*W,const int8_t*X,int rows,int cols,int reps){
    int kb=cols/64; uint64_t t0=rdcyc(); volatile long sink=0;
    const int8_t*x0=X,*x1=X+(size_t)1*cols,*x2=X+(size_t)2*cols,*x3=X+(size_t)3*cols;
    for(int rep=0;rep<reps;rep++) for(int r=0;r+3<rows;r+=4){
        const int8_t*w0=W+(size_t)(r+0)*cols,*w1=W+(size_t)(r+1)*cols;
        const int8_t*w2=W+(size_t)(r+2)*cols,*w3=W+(size_t)(r+3)*cols;
        svint32_t c00=svdup_s32(0),c01=svdup_s32(0),c02=svdup_s32(0),c03=svdup_s32(0);
        svint32_t c10=svdup_s32(0),c11=svdup_s32(0),c12=svdup_s32(0),c13=svdup_s32(0);
        svint32_t c20=svdup_s32(0),c21=svdup_s32(0),c22=svdup_s32(0),c23=svdup_s32(0);
        svint32_t c30=svdup_s32(0),c31=svdup_s32(0),c32=svdup_s32(0),c33=svdup_s32(0);
        for(int k=0;k<kb;k++){
            svint8_t a0=LD(w0,k),a1=LD(w1,k),a2=LD(w2,k),a3=LD(w3,k);
            svint8_t b0=LD(x0,k),b1=LD(x1,k),b2=LD(x2,k),b3=LD(x3,k);
            c00=svdot_s32(c00,a0,b0); c01=svdot_s32(c01,a0,b1); c02=svdot_s32(c02,a0,b2); c03=svdot_s32(c03,a0,b3);
            c10=svdot_s32(c10,a1,b0); c11=svdot_s32(c11,a1,b1); c12=svdot_s32(c12,a1,b2); c13=svdot_s32(c13,a1,b3);
            c20=svdot_s32(c20,a2,b0); c21=svdot_s32(c21,a2,b1); c22=svdot_s32(c22,a2,b2); c23=svdot_s32(c23,a2,b3);
            c30=svdot_s32(c30,a3,b0); c31=svdot_s32(c31,a3,b1); c32=svdot_s32(c32,a3,b2); c33=svdot_s32(c33,a3,b3);
        }
        svbool_t pg=svptrue_b32();
        sink+=svaddv_s32(pg,c00)+svaddv_s32(pg,c11)+svaddv_s32(pg,c22)+svaddv_s32(pg,c33);
    }
    uint64_t t1=rdcyc(); (void)sink;
    return (double)(rows/4*4)*kb*4*reps*OPS_PER_SDOT/((double)(t1-t0)/freq)/1e9;
}

int main(int argc,char**argv){
    int rows = argc>1?atoi(argv[1]):21504;
    int cols = argc>2?atoi(argv[2]):5376;
    double freq=(double)rdfreq();
    size_t wb=(size_t)rows*cols;
    int8_t *W=(int8_t*)aligned_alloc(256,wb);
    int8_t *X=(int8_t*)aligned_alloc(256,(size_t)16*cols);
    for(size_t i=0;i<wb;i++) W[i]=(int8_t)((i*131+7)&0x3f);
    for(size_t i=0;i<(size_t)16*cols;i++) X[i]=(int8_t)((i*17+1)&0x3f);

    int reps = wb>64L*1024*1024?3:8;
    printf("GEMM int8 SDOT, shape rows=%d cols=%d  (W=%.1f MB, > L2 -> streams HBM)\n",
           rows,cols,wb/1048576.0);
    printf("compute peak @2.0GHz=512 GIOPS/core; M=1 == matvec.\n\n");
    printf("  naive 1xM blocking (1 W-load + M X-loads per M SDOTs):\n");
    printf("    %-12s %-15s %9s %9s  %s\n","M(batch)","intensity(op/B)","GIOPS","%peak","note");
    struct{int M;double(*fn)(double,const int8_t*,const int8_t*,int,int,int);}t[]={
        {1,gemm_m1},{4,gemm_m4},{8,gemm_m8},{16,gemm_m16}};
    for(int i=0;i<4;i++){
        double g=t[i].fn(freq,W,X,rows,cols,reps);
        printf("    %-12d %-15d %9.1f %8.1f%%  %s\n",t[i].M,2*t[i].M,g,g/512*100,
               t[i].M==1?"matvec (decode)":(t[i].M==16?"load-port bound":""));
    }
    double g44=gemm_tile44(freq,W,X,rows,cols,reps);
    printf("  2D 4x4 register tile (8 loads per 16 SDOTs, batch=4):\n");
    printf("    %-12s %-15d %9.1f %8.1f%%  %s\n","4 (4x4)",8,g44,g44/512*100,"load-amortized");
    free(W);free(X);
    return 0;
}
