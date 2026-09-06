/* swar_fused.c - fused Q4_0 matvec with SCALAR (SWAR) dequant on the integer
 * pipes (EXA/EXB) overlapping SVE svdot on the FP/SIMD pipes (FLA/FLB).
 *
 * Hypothesis (user): SVE dequant competes with svdot for FLA/FLB -> serializes.
 * Scalar/SWAR dequant uses the integer pipes, so OoO can run dequant(block b+1)
 * in parallel with svdot(block b). Double-buffered to expose that ILP.
 *
 * Cheap scalar path via SWAR + the identity  sum (v-8)*x = sum v*x - 8*sum x :
 *   - dequant = just mask/shift (extract nibble v in 0..15), NO sub, NO mul.
 *   - svdot(v, xi8) -> S1 ; block dot (v-8)*x = S1 - 8*Sx[b] (Sx precomputed).
 *   - row += d[b] * (S1 - 8*Sx[b]) ; dst = row / x_inv.
 *
 * Build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp \
 *           -D_GNU_SOURCE -I../../common swar_fused.c -lm -o swar_fused
 */
#define TRANSFORMER_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION
#define GGUF_LOADER_IMPLEMENTATION
#include "transformer.h"
#include <omp.h>
#include <sys/mman.h>

static inline uint64_t rdcyc(void){ uint64_t v; __asm__ volatile("mrs %0, cntvct_el0":"=r"(v)); return v; }
static inline uint64_t rdfreq(void){ uint64_t v; __asm__ volatile("mrs %0, cntfrq_el0":"=r"(v)); return v; }
static uint16_t f32_to_f16(float f){ uint32_t b; memcpy(&b,&f,4); uint32_t s=(b>>16)&0x8000; int e=(int)((b>>23)&0xFF)-127+15; uint32_t m=b&0x7FFFFF;
    if(e<=0){ if(e<-10)return(uint16_t)s; m|=0x800000; return(uint16_t)(s|(m>>(14-e))); } if(e>=31)return(uint16_t)(s|0x7C00); return(uint16_t)(s|((uint32_t)e<<10)|(m>>13)); }

/* SWAR extract: 16 qs bytes -> 32 int8 v(0..15) in natural element order
 * (buf[0..15]=lo of bytes 0..15, buf[16..31]=hi). Pure scalar uint64 ops. */
static inline void swar_block(int8_t *buf, const uint8_t *qs){
    const uint64_t M=0x0f0f0f0f0f0f0f0fULL;
    uint64_t w0,w1; memcpy(&w0,qs,8); memcpy(&w1,qs+8,8);
    uint64_t lo0=w0&M, lo1=w1&M, hi0=(w0>>4)&M, hi1=(w1>>4)&M;
    memcpy(buf+0,&lo0,8); memcpy(buf+8,&lo1,8); memcpy(buf+16,&hi0,8); memcpy(buf+24,&hi1,8);
}
/* one row: SWAR dequant on EX pipes, svdot on FL pipes, double-buffered. */
static inline float swar_row(const uint8_t *qs, const float *d, int nb, const int8_t *xi8, const int32_t *Sx){
    svbool_t pg32=svwhilelt_b8((uint32_t)0,(uint32_t)32), p3=svptrue_b32();
    int8_t buf[2][32] __attribute__((aligned(64)));
    swar_block(buf[0], qs);
    float acc=0.0f;
    for(int b=0;b<nb;b++){
        int cur=b&1;
        svint8_t v=svld1_s8(pg32,buf[cur]);
        svint8_t xb=svld1_s8(pg32,xi8+(size_t)b*32);
        int32_t S1=svaddv_s32(p3, svdot_s32(svdup_s32(0), v, xb));
        if(b+1<nb) swar_block(buf[(b+1)&1], qs+(size_t)(b+1)*16);   /* EX pipe, overlaps the FL work above */
        acc += d[b]*(float)(S1 - 8*Sx[b]);
    }
    return acc;
}
static void ref_matvec(float*dst,const block_q4_0*w,size_t rb,const float*x,int rows,int cols){
    int nb=cols/32; for(int r=0;r<rows;r++){ const block_q4_0*row=(const block_q4_0*)((const uint8_t*)w+(size_t)r*rb);
        float s=0; for(int b=0;b<nb;b++){ float d=ggml_fp16_to_fp32(row[b].d);
            for(int j=0;j<16;j++){ s+=d*((row[b].qs[j]&0xf)-8)*x[b*32+j]; s+=d*((row[b].qs[j]>>4)-8)*x[b*32+j+16]; } } dst[r]=s; } }
static double relL2(const float*a,const float*b,int n){ double e=0,s=0; for(int i=0;i<n;i++){ double q=a[i]-b[i]; e+=q*q; s+=(double)b[i]*b[i]; } return sqrt(e/(s+1e-12)); }

int main(int argc,char**argv){
    int rows=argc>2?atoi(argv[1]):21504, cols=argc>2?atoi(argv[2]):5376;
    double freq=(double)rdfreq();
    int nb=cols/32; size_t row_bytes=(size_t)nb*sizeof(block_q4_0);
    block_q4_0 *W=(block_q4_0*)aligned_alloc(256,(size_t)rows*row_bytes);
    float *x=(float*)aligned_alloc(256,(size_t)cols*4),*dref=(float*)aligned_alloc(256,(size_t)rows*4),*dP=(float*)aligned_alloc(256,(size_t)rows*4);
    srand(42); for(size_t i=0;i<(size_t)rows*nb;i++){ float d=0.01f+(rand()/(float)RAND_MAX)*0.5f; W[i].d=f32_to_f16(d); for(int j=0;j<16;j++) W[i].qs[j]=(unsigned char)(rand()&0xFF); }
    srand(137); for(int i=0;i<cols;i++) x[i]=(rand()/(float)RAND_MAX-0.5f)*2.0f;
    ref_matvec(dref,W,row_bytes,x,rows,cols);
    int8_t *xi8=(int8_t*)aligned_alloc(256,(size_t)((cols+63)&~63)); float x_inv; tf_quantize_f32_to_int8(x,xi8,cols,&x_inv); float inv_x=1.0f/x_inv;
    int32_t *Sx=(int32_t*)aligned_alloc(256,(size_t)nb*4); for(int b=0;b<nb;b++){ int s=0; for(int j=0;j<32;j++) s+=xi8[b*32+j]; Sx[b]=s; }
    /* repacked: dense nibbles WQS (16 B/blk) + per-block d (fp32). 0.625 B/wt. */
    size_t qb=(size_t)rows*nb*16; size_t db=(size_t)rows*nb*4; size_t rbytes=qb+db;

    printf("SWAR-fused (scalar EX-pipe dequant + SVE FL-pipe svdot)  rows=%d cols=%d\n",rows,cols);
    printf("  reads %.1f MB (%.3f B/wt: qs 0.5 + d 0.125); int8 cache ref ~73%%/CMG=168 GB/s,336 GIOPS\n\n",rbytes/1048576.0,(double)rbytes/((double)rows*cols));
    printf("  %-4s %5s %9s %9s %9s %s\n","thr","CMGs","ms","GB/s","GIOPS","ok");
    int threads[]={1,12,24,48};
    for(int ti=0;ti<4;ti++){ int nt=threads[ti];
        uint8_t *WQS=(uint8_t*)mmap(NULL,qb,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        float   *D  =(float  *)mmap(NULL,db,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        #pragma omp parallel for num_threads(nt) schedule(static)
        for(int r=0;r<rows;r++){ const block_q4_0*row=(const block_q4_0*)((const uint8_t*)W+(size_t)r*row_bytes);
            for(int b=0;b<nb;b++){ memcpy(WQS+((size_t)r*nb+b)*16,row[b].qs,16); D[(size_t)r*nb+b]=ggml_fp16_to_fp32(row[b].d); } }
        #pragma omp parallel for num_threads(nt) schedule(static)
        for(int r=0;r<rows;r++) dP[r]=swar_row(WQS+(size_t)r*nb*16,D+(size_t)r*nb,nb,xi8,Sx)*inv_x;
        double best=1e30; int N=10;
        for(int rep=0;rep<N;rep++){ uint64_t t0=rdcyc();
            #pragma omp parallel for num_threads(nt) schedule(static)
            for(int r=0;r<rows;r++) dP[r]=swar_row(WQS+(size_t)r*nb*16,D+(size_t)r*nb,nb,xi8,Sx)*inv_x;
            double s=(double)(rdcyc()-t0)/freq; if(s<best)best=s; }
        int ncmg=(nt+11)/12; double gbps=(double)rbytes/best/1e9, gi=2.0*rows*cols/best/1e9;
        double mr=relL2(dP,dref,rows);
        printf("  %-4d %5d %9.3f %9.1f %9.1f %s\n",nt,ncmg,best*1000,gbps,gi,mr<0.06?"OK":"FAIL");
        munmap(WQS,qb); munmap(D,db);
    }
    return 0;
}
