/* q8v2_gemm.c - standalone MULTI-THREAD node-level Q8v2 int8 GEMM (no model).
 * The microkernel (q8v2_profile.c) is L1-resident; this measures the full
 * weight x activation GEMM where the weight (N*K int8, ~115 MB at gemma dims)
 * is streamed from HBM. Compares two loop orders:
 *   flat : collapse(2) over (n-tile, m-tile)            -- weight re-read per m-tile
 *   gepp : for n-panel(L2-resident) { for m-tile { kernel } }  -- weight reused
 *
 *   C[M_tok, N_out] = A[M,K](int8 acts) * W[N,K](centered-nibble int8) + per-block scales
 *
 * Build: make q8v2_gemm    Run: ./q8v2_gemm [N=21504] [K=5376] [NC=1536]
 *   NC = GEPP N-panel width (rows of weight kept L2-resident); 0 -> flat only.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>
#include <sys/mman.h>

void kernel_q8v2_3x4(const int8_t*aq,const float*ad,const int8_t*bq,const float*bd,long nb,float*C,long ldc);

static inline uint64_t rdcyc(void){ uint64_t v; __asm__ volatile("mrs %0, cntvct_el0":"=r"(v)); return v; }
static inline uint64_t rdfreq(void){ uint64_t v; __asm__ volatile("mrs %0, cntfrq_el0":"=r"(v)); return v; }
static void*xmap(size_t n){ void*p=mmap(NULL,n,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0); if(p==MAP_FAILED){perror("mmap");exit(1);} return p; }
#define MR 3
#define NR 64
#define BLK 32

int main(int argc,char**argv){
    int N  = argc>1?atoi(argv[1]):21504;   /* n_ff */
    int K  = argc>2?atoi(argv[2]):5376;    /* n_embd */
    int NC = argc>3?atoi(argv[3]):1536;    /* GEPP n-panel width (mult of 64) */
    int nt = getenv("OMP_NUM_THREADS")?atoi(getenv("OMP_NUM_THREADS")):48;
    int nb=K/BLK, NTn=N/NR; double peak=512.0*nt;
    if(N%NR||K%BLK){ fprintf(stderr,"need N%%64,K%%32\n"); return 1; }
    if(NC%NR) NC=(NC/NR)*NR;
    size_t bq_tile=(size_t)nb*NR*BLK, bd_tile=(size_t)nb*NR;
    fprintf(stderr,"[q8v2-node] N=%d K=%d nb=%d Ntiles=%d  weight=%.0f MB  NC=%d (%d n-tiles/panel)\n",
            N,K,nb,NTn,(double)NTn*bq_tile/1e6,NC,NC/NR);

    /* synthetic packed weights (centered nibbles [-8,7]) + per-block scales, NUMA-local */
    int8_t*Bq=(int8_t*)xmap((size_t)NTn*bq_tile);
    float *Bd=(float*)xmap((size_t)NTn*bd_tile*sizeof(float));
    #pragma omp parallel for num_threads(nt) schedule(static)
    for(int n0=0;n0<NTn;n0++){ int8_t*bq=Bq+(size_t)n0*bq_tile; float*bd=Bd+(size_t)n0*bd_tile;
        for(size_t i=0;i<bq_tile;i++) bq[i]=(int8_t)(((n0*7+i*13)%15)-8);
        for(size_t i=0;i<bd_tile;i++) bd[i]=0.01f+0.0003f*((n0+i)%11); }

    int Ms[]={48,96,192,384}; int nM=4;
    printf("\n=== Q8v2 node GEMM N=%d K=%d @ %d thr (peak %.0f GIOPS) ===\n",N,K,nt,peak);
    printf("  %-6s %10s %8s %10s %8s   %s\n","M","flat GI","%pk","gepp GI","%pk","speedup");
    for(int mi=0;mi<nM;mi++){ int M=Ms[mi]; int MTn=(M+MR-1)/MR;
        size_t aq_tile=(size_t)nb*MR*BLK, ad_tile=(size_t)nb*MR;
        int8_t*Aq=(int8_t*)xmap((size_t)MTn*aq_tile); float*Ad=(float*)xmap((size_t)MTn*ad_tile*sizeof(float));
        #pragma omp parallel for num_threads(nt) schedule(static)
        for(int m0=0;m0<MTn;m0++){ int8_t*aq=Aq+(size_t)m0*aq_tile; float*ad=Ad+(size_t)m0*ad_tile;
            for(size_t i=0;i<aq_tile;i++) aq[i]=(int8_t)((int)((m0*5+i*3)%127)-63);
            for(size_t i=0;i<ad_tile;i++) ad[i]=0.002f+0.0001f*((m0+i)%17); }
        float*C=(float*)xmap((size_t)MTn*MR*N*sizeof(float));

        #define FLAT() _Pragma("omp parallel for num_threads(nt) schedule(static) collapse(2)") \
            for(int n0=0;n0<NTn;n0++) for(int m0=0;m0<MTn;m0++) \
                kernel_q8v2_3x4(Aq+(size_t)m0*aq_tile,Ad+(size_t)m0*ad_tile,Bq+(size_t)n0*bq_tile,Bd+(size_t)n0*bd_tile,nb,C+(size_t)(m0*MR)*N+n0*NR,(long)N*4);
        /* GEPP: n-panels of NC rows kept L2-resident, swept by all m-tiles. Parallelize
         * over n-tiles within the (sequential) panel loop so each panel's weight is
         * read once into L2 and reused across all M token-tiles. */
        #define GEPP() \
            for(int nc=0;nc<NTn;nc+=NC/NR){ int ne=nc+NC/NR; if(ne>NTn)ne=NTn; \
              _Pragma("omp parallel for num_threads(nt) schedule(static) collapse(2)") \
              for(int n0=nc;n0<ne;n0++) for(int m0=0;m0<MTn;m0++) \
                kernel_q8v2_3x4(Aq+(size_t)m0*aq_tile,Ad+(size_t)m0*ad_tile,Bq+(size_t)n0*bq_tile,Bd+(size_t)n0*bd_tile,nb,C+(size_t)(m0*MR)*N+n0*NR,(long)N*4); }

        FLAT(); GEPP();   /* warm both */
        double freq=(double)rdfreq(); int reps=3; volatile double sink=0;
        uint64_t c0=rdcyc(); for(int r=0;r<reps;r++){ FLAT(); sink+=C[(r*48271)%((size_t)M*N)]; } uint64_t c1=rdcyc();
        double tf=(double)(c1-c0)/freq/reps; double gf=2.0*(double)M*N*K/tf/1e9;
        uint64_t c2=rdcyc(); for(int r=0;r<reps;r++){ GEPP(); sink+=C[(r*48271)%((size_t)M*N)]; } uint64_t c3=rdcyc();
        double tg=(double)(c3-c2)/freq/reps; double gg=2.0*(double)M*N*K/tg/1e9; (void)sink;
        printf("  %-6d %10.0f %7.0f%% %10.0f %7.0f%%   %.2fx\n",M,gf,gf/peak*100,gg,gg/peak*100,gf>0?gg/gf:0);
        #undef FLAT
        #undef GEPP
        munmap(Aq,(size_t)MTn*aq_tile); munmap(Ad,(size_t)MTn*ad_tile*sizeof(float)); munmap(C,(size_t)MTn*MR*N*sizeof(float));
    }
    return 0;
}
