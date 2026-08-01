/* numa_matvec.c - NUMA-clean int8 matvec BW measurement on A64FX.
 *
 * The cross-run variance in dq_mv_omp came from the shared weight matrix being
 * first-touched by setup threads, not the measurement threads -> remote HBM.
 * Here, for EACH thread count we (1) fresh-allocate the int8 weights, (2)
 * first-touch them with the SAME num_threads + static split the measurement
 * uses (so thread t's rows live on thread t's CMG), (3) run a warmup rep
 * (untimed), (4) time best-of-N. This gives the true per-CMG / node rate.
 *
 * Build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp \
 *           -D_GNU_SOURCE -I../../common numa_matvec.c -lm -o numa_matvec
 * Run:   OMP_PROC_BIND=close OMP_PLACES=cores ./numa_matvec [rows cols]
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

/* int8 8-row matvec, 8 independent accumulators (v1 — the A/B winner). */
static inline void mv8(int32_t acc[8], const int8_t *w0,int nbp,const int8_t *xi8){
    svbool_t pg=svptrue_b8();
    svint32_t a0=svdup_s32(0),a1=svdup_s32(0),a2=svdup_s32(0),a3=svdup_s32(0),a4=svdup_s32(0),a5=svdup_s32(0),a6=svdup_s32(0),a7=svdup_s32(0);
    size_t rs=(size_t)nbp*64; const int8_t*w1=w0+rs,*w2=w0+2*rs,*w3=w0+3*rs,*w4=w0+4*rs,*w5=w0+5*rs,*w6=w0+6*rs,*w7=w0+7*rs;
    for(int p=0;p<nbp;p++){ svint8_t xv=svld1_s8(pg,xi8+(size_t)p*64);
        a0=svdot_s32(a0,svld1_s8(pg,w0+(size_t)p*64),xv); a1=svdot_s32(a1,svld1_s8(pg,w1+(size_t)p*64),xv);
        a2=svdot_s32(a2,svld1_s8(pg,w2+(size_t)p*64),xv); a3=svdot_s32(a3,svld1_s8(pg,w3+(size_t)p*64),xv);
        a4=svdot_s32(a4,svld1_s8(pg,w4+(size_t)p*64),xv); a5=svdot_s32(a5,svld1_s8(pg,w5+(size_t)p*64),xv);
        a6=svdot_s32(a6,svld1_s8(pg,w6+(size_t)p*64),xv); a7=svdot_s32(a7,svld1_s8(pg,w7+(size_t)p*64),xv); }
    svbool_t p3=svptrue_b32();
    acc[0]=svaddv_s32(p3,a0);acc[1]=svaddv_s32(p3,a1);acc[2]=svaddv_s32(p3,a2);acc[3]=svaddv_s32(p3,a3);
    acc[4]=svaddv_s32(p3,a4);acc[5]=svaddv_s32(p3,a5);acc[6]=svaddv_s32(p3,a6);acc[7]=svaddv_s32(p3,a7);
}

int main(int argc,char**argv){
    int rows=argc>2?atoi(argv[1]):21504, cols=argc>2?atoi(argv[2]):5376;
    double freq=(double)rdfreq();
    int nb=cols/32, nbp=(cols+63)/64;
    size_t row_bytes=(size_t)nb*sizeof(block_q4_0);
    size_t i8bytes=(size_t)rows*nbp*64;

    block_q4_0 *W=(block_q4_0*)aligned_alloc(256,(size_t)rows*row_bytes);
    float *x=(float*)aligned_alloc(256,(size_t)cols*4), *dst=(float*)aligned_alloc(256,(size_t)rows*4);
    srand(42); for(size_t i=0;i<(size_t)rows*nb;i++){ float d=0.01f+(rand()/(float)RAND_MAX)*0.5f; W[i].d=f32_to_f16(d); for(int j=0;j<16;j++) W[i].qs[j]=(unsigned char)(rand()&0xFF); }
    srand(137); for(int i=0;i<cols;i++) x[i]=(rand()/(float)RAND_MAX-0.5f)*2.0f;
    float max_d=0; for(size_t i=0;i<(size_t)rows*nb;i++){ float a=fabsf(ggml_fp16_to_fp32(W[i].d)); if(a>max_d)max_d=a; }
    float scale_w=max_d>0?127.0f/(8.0f*max_d):1.0f;
    int8_t *xi8=(int8_t*)aligned_alloc(256,(size_t)nbp*64); float x_inv; tf_quantize_f32_to_int8(x,xi8,cols,&x_inv);
    float inv=1.0f/(scale_w*x_inv);

    printf("NUMA-clean int8 matvec  rows=%d cols=%d  int8=%.1f MB\n",rows,cols,i8bytes/1048576.0);
    printf("ceiling 230 GB/s/CMG (1 CMG=12 cores), 852/node; reports best-of-N, weights CMG-local\n\n");
    printf("  %-4s %5s %9s %9s %12s %9s\n","thr","CMGs","GB/s","GIOPS","GB/s/CMG","%ceil");
    int threads[]={1,2,4,8,12,24,36,48};
    for(int ti=0;ti<8;ti++){ int nt=threads[ti];
        /* fresh alloc -> fresh pages; first-touch (dequant) with the measurement split */
        int8_t *WI8=(int8_t*)mmap(NULL,i8bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        #pragma omp parallel for num_threads(nt) schedule(static)
        for(int g=0;g<rows;g+=8)
            tf_dequant_q4_0_8row_strided_to_int8((const uint8_t*)W+(size_t)g*row_bytes,row_bytes,WI8+(size_t)g*nbp*64,cols,scale_w);
        /* warmup (untimed) */
        #pragma omp parallel for num_threads(nt) schedule(static)
        for(int g=0;g<rows;g+=8){ int32_t a[8]; mv8(a,WI8+(size_t)g*nbp*64,nbp,xi8); for(int i=0;i<8;i++) dst[g+i]=(float)a[i]*inv; }
        /* timed: best of N */
        double best=1e30; int N=12;
        for(int rep=0;rep<N;rep++){ uint64_t t0=rdcyc();
            #pragma omp parallel for num_threads(nt) schedule(static)
            for(int g=0;g<rows;g+=8){ int32_t a[8]; mv8(a,WI8+(size_t)g*nbp*64,nbp,xi8); for(int i=0;i<8;i++) dst[g+i]=(float)a[i]*inv; }
            double s=(double)(rdcyc()-t0)/freq; if(s<best)best=s; }
        int ncmg=(nt+11)/12; double gbps=(double)i8bytes/best/1e9, gi=2.0*rows*cols/best/1e9;
        printf("  %-4d %5d %9.1f %9.1f %12.1f %8.0f%%\n",nt,ncmg,gbps,gi,gbps/ncmg,gbps/ncmg/230.0*100);
        munmap(WI8,i8bytes);
    }
    return 0;
}
