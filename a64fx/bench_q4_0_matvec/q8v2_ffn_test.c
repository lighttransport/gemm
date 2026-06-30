/* q8v2_ffn_test.c - FAST standalone unit test for the Q8v2 FFN gate/up GEMM,
 * validated against the production tf_gemm_q4_0_pair_gelu_tokenmajor. Synthetic
 * Q4_0 weights + activations, NO model load. Iterate here (seconds), then port
 * tf_gemm_q8v2_pair_gelu_tokenmajor into transformer.h.
 *
 * Build: fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -D_GNU_SOURCE \
 *   -I../../common q8v2_ffn_test.c ../gemma4-kernels/kernel_q8v2_3x4.S -lm -o q8v2_ffn_test
 */
#define TRANSFORMER_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION
#define GGUF_LOADER_IMPLEMENTATION
#include "transformer.h"
#include <math.h>
#include <omp.h>

void kernel_q8v2_3x4(const int8_t*,const float*,const int8_t*,const float*,long,float*,long);

/* ===== the function to port into transformer.h ===== */
static int tf_gemm_q8v2_pair_gelu_tokenmajor(float *Y, const qtensor *gate,
        const qtensor *up, const float *X, int n_rows, int N,
        int out_stride, int X_stride, int n_threads) {
    if (!gate || !up || gate->type!=GGML_TYPE_Q4_0 || up->type!=GGML_TYPE_Q4_0) return 0;
    if (gate->n_cols!=up->n_cols || gate->n_rows!=up->n_rows ||
        gate->n_cols!=X_stride || n_rows!=gate->n_rows) return 0;
    int K = gate->n_cols;
    if (K%256 || n_rows%64) return 0;          /* kernel needs K%256, NR=64 */
    const int MR=3, NR=64, BLK=32;
    int nb=K/BLK, NTn=n_rows/NR, MTn=(N+MR-1)/MR;
    int nt=n_threads<1?1:n_threads;

    /* 1. quantize activations -> Aq[MTn][nb*MR*32] int8, Ad[MTn][nb*MR] f32 (per-block per-row) */
    size_t aqt=(size_t)nb*MR*BLK, adt=(size_t)nb*MR;
    int8_t*Aq=(int8_t*)malloc((size_t)MTn*aqt);
    float *Ad=(float*)malloc((size_t)MTn*adt*sizeof(float));
    if(!Aq||!Ad){ free(Aq); free(Ad); return 0; }
    #pragma omp parallel for num_threads(nt) schedule(static)
    for(int m0=0;m0<MTn;m0++){ int8_t*aq=Aq+(size_t)m0*aqt; float*ad=Ad+(size_t)m0*adt;
        for(int b=0;b<nb;b++) for(int r=0;r<MR;r++){ int tok=m0*MR+r;
            float amax=0; if(tok<N) for(int k=0;k<BLK;k++){ float a=fabsf(X[(size_t)tok*X_stride+b*BLK+k]); if(a>amax)amax=a; }
            float d=amax>0?amax/127.0f:0.0f, inv=d>0?1.0f/d:0.0f;
            ad[b*MR+r]=d;
            for(int k=0;k<BLK;k++){ int q=0; if(tok<N){ q=(int)lrintf(X[(size_t)tok*X_stride+b*BLK+k]*inv); if(q>127)q=127; if(q<-127)q=-127; } aq[(size_t)(b*MR+r)*BLK+k]=(int8_t)q; }
        }
    }

    /* 2. per n-tile (64 out-features): dequant gate/up Q4_0 -> centered-nibble int8 (pack_B
     *    layout) + scales, then for each m-tile run two kernels and fuse fast-GELU. */
    const block_q4_0*G=(const block_q4_0*)gate->data;
    const block_q4_0*U=(const block_q4_0*)up->data;
    #pragma omp parallel num_threads(nt)
    {
        size_t bqt=(size_t)nb*8*4*64; /* = nb*NR*BLK */
        int8_t*bqg=(int8_t*)malloc(bqt), *bqu=(int8_t*)malloc(bqt);
        float *bdg=(float*)malloc((size_t)nb*NR*sizeof(float)), *bdu=(float*)malloc((size_t)nb*NR*sizeof(float));
        float Cg[MR*NR], Cu[MR*NR];
        #pragma omp for schedule(static)
        for(int n0=0;n0<NTn;n0++){
            /* pack gate & up n-tile */
            for(int b=0;b<nb;b++) for(int vec=0;vec<4;vec++) for(int col=0;col<16;col++){
                int n=n0*NR+vec*16+col;
                const block_q4_0*gr=G+(size_t)n*nb, *ur=U+(size_t)n*nb;
                bdg[b*NR+vec*16+col]=ggml_fp16_to_fp32(gr[b].d);
                bdu[b*NR+vec*16+col]=ggml_fp16_to_fp32(ur[b].d);
                for(int g3=0;g3<8;g3++) for(int kk=0;kk<4;kk++){ int k=g3*4+kk;
                    int qg=(k<16)?(gr[b].qs[k]&0xf):(gr[b].qs[k-16]>>4);
                    int qu=(k<16)?(ur[b].qs[k]&0xf):(ur[b].qs[k-16]>>4);
                    size_t off=((size_t)b*8*4+(size_t)g3*4+vec)*64 + (size_t)col*4 + kk;
                    bqg[off]=(int8_t)(qg-8); bqu[off]=(int8_t)(qu-8);
                }
            }
            if(getenv("SKIP_KERNEL")) continue;
            for(int m0=0;m0<MTn;m0++){
                const int8_t*aq=Aq+(size_t)m0*aqt; const float*ad=Ad+(size_t)m0*adt;
                kernel_q8v2_3x4(aq,ad,bqg,bdg,nb,Cg,(long)NR*4);
                kernel_q8v2_3x4(aq,ad,bqu,bdu,nb,Cu,(long)NR*4);
                for(int r=0;r<MR;r++){ int tok=m0*MR+r; if(tok>=N) continue;
                    for(int c=0;c<NR;c++) Y[(size_t)tok*out_stride + n0*NR + c]=tf_gelu_fast_scalar(Cg[r*NR+c])*Cu[r*NR+c];
                }
            }
        }
        free(bqg); free(bqu); free(bdg); free(bdu);
    }
    free(Aq); free(Ad);
    return 1;
}

/* ===== synthetic Q4_0 builder + test ===== */
static uint16_t f32_to_f16(float f){ uint32_t x; memcpy(&x,&f,4);
    uint32_t s=(x>>16)&0x8000; int e=((x>>23)&0xff)-112; uint32_t m=x&0x7fffff;
    if(e<=0) return (uint16_t)s; if(e>=31) return (uint16_t)(s|0x7c00);
    return (uint16_t)(s|(e<<10)|(m>>13)); }
static void synth_q4(block_q4_0*W,int n_rows,int K,int seed){ int nb=K/32;
    for(int r=0;r<n_rows;r++) for(int b=0;b<nb;b++){ block_q4_0*blk=&W[(size_t)r*nb+b];
        blk->d=f32_to_f16(0.01f+0.002f*((r*7+b*13+seed)%11));
        for(int j=0;j<16;j++){ int lo=(r*3+b*5+j*7+seed)%16, hi=(r*5+b*3+j*11+seed)%16; blk->qs[j]=(uint8_t)(lo|(hi<<4)); } }
}
int main(int argc,char**argv){
    int N=argc>1?atoi(argv[1]):128, K=argc>2?atoi(argv[2]):5376, n_rows=argc>3?atoi(argv[3]):21504;
    int nt=getenv("OMP_NUM_THREADS")?atoi(getenv("OMP_NUM_THREADS")):48;
    int nb=K/32;
    block_q4_0*Wg=(block_q4_0*)malloc((size_t)n_rows*nb*sizeof(block_q4_0));
    block_q4_0*Wu=(block_q4_0*)malloc((size_t)n_rows*nb*sizeof(block_q4_0));
    synth_q4(Wg,n_rows,K,1); synth_q4(Wu,n_rows,K,2);
    float*X=(float*)malloc((size_t)N*K*sizeof(float));
    for(int t=0;t<N;t++) for(int k=0;k<K;k++) X[(size_t)t*K+k]=0.04f*(float)(((t*3+k*7)&0x3f)-32);
    qtensor gate={Wg,GGML_TYPE_Q4_0,n_rows,K,2,{(uint64_t)K,(uint64_t)n_rows,0,0},0,0};
    qtensor up  ={Wu,GGML_TYPE_Q4_0,n_rows,K,2,{(uint64_t)K,(uint64_t)n_rows,0,0},0,0};

    float*Yref=(float*)calloc((size_t)N*n_rows,sizeof(float));
    float*Yq8 =(float*)calloc((size_t)N*n_rows,sizeof(float));
    struct timespec ta,tb; double tref,tq8;
    clock_gettime(CLOCK_MONOTONIC,&ta);
    int r1=tf_gemm_q4_0_pair_gelu_tokenmajor(Yref,&gate,&up,X,n_rows,N,n_rows,K,nt,0);
    clock_gettime(CLOCK_MONOTONIC,&tb); tref=(tb.tv_sec-ta.tv_sec)+(tb.tv_nsec-ta.tv_nsec)*1e-9;
    clock_gettime(CLOCK_MONOTONIC,&ta);
    int r2=tf_gemm_q8v2_pair_gelu_tokenmajor(Yq8,&gate,&up,X,n_rows,N,n_rows,K,nt);
    clock_gettime(CLOCK_MONOTONIC,&tb); tq8=(tb.tv_sec-ta.tv_sec)+(tb.tv_nsec-ta.tv_nsec)*1e-9;
    fprintf(stderr,"TIME ref=%.1f ms  q8=%.1f ms  speedup=%.2fx\n",tref*1e3,tq8*1e3,tref/tq8);
    fprintf(stderr,"ref=%d q8=%d  N=%d K=%d n_rows=%d\n",r1,r2,N,K,n_rows);
    if(!r1||!r2){ fprintf(stderr,"a gemm returned 0\n"); return 1; }
    double num=0,den=0,maxe=0; int amr=0,amq=0; float mr=-1e30f,mq=-1e30f;
    for(size_t i=0;i<(size_t)N*n_rows;i++){ double e=(double)Yq8[i]-Yref[i]; num+=e*e; den+=(double)Yref[i]*Yref[i]; if(fabs(e)>maxe)maxe=fabs(e);
        if(Yref[i]>mr){mr=Yref[i];amr=(int)i;} if(Yq8[i]>mq){mq=Yq8[i];amq=(int)i;} }
    printf("relL2=%.5f  maxabs=%.4g  argmax ref=%d q8=%d %s\n",sqrt(num/den),maxe,amr,amq,amr==amq?"MATCH":"DIFFER");
    return 0;
}
