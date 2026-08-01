/* bdecode_kern_bench — decode-shape bf16 matvec/GEMM bandwidth on a REAL A64FX node.
 *
 * Calibrates the two local simulators used for GLM-5.2 decode planning:
 *   decode_sim.py : BW_NODE (effective matvec GB/s; currently assumed 300 of 1024 peak)
 *                   and how the effective GB/s scales with the batch M (batched decode
 *                   re-reads weights ONCE per K-tile for all M streams).
 *   qlair         : per-kernel cycle cross-check for the A64FX pipeline model (run the
 *                   same shapes under `qlair -p` and compare GB/s within the 10% target).
 *
 * Times glm5_gemm_bf16 (the batched-decode kernel; M=1 uses the same matvec_bf16_8row
 * primitive as glm5_mv_bf16) on the REAL GLM-5.2 decode shapes, both replicated and the
 * 8-way-sharded variants that a 96n TP run sees. int8 kernels are covered separately by
 * glm5_int8_kernel_test.
 *
 * Build (native A64FX):
 *   fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -std=c11 -D_GNU_SOURCE \
 *       -fopenmp -I ../../common -o build/bdecode_kern_bench bdecode_kern_bench.c -lm
 * Run:
 *   OMP_NUM_THREADS=48 ./build/bdecode_kern_bench
 * Env: KB_REPS_MS target ms per measurement (default 200), KB_MMAX (default 32).
 *
 * Output: grep-able "KERNBENCH,shape,rows,cols,M,ms_per_call,GBs,GBs_per_stream" CSV.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "glm5.h"
#include "glm5_impl.h"

static double ksec(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }

typedef struct { const char*name; int rows, cols; } kb_shape;

int main(void){
    /* GLM-5.2 decode shapes (per layer / per token). _s8 = 8-way TP shard (96n-style run
     * has 96-way expert sharding but 8-way head/vocab groups are the qlair-comparable case;
     * scale rows for other shard counts — bandwidth per byte is shape-insensitive at these sizes). */
    const kb_shape shapes[] = {
        {"wq_a",      2048,  6144},   /* q down-proj, replicated */
        {"wq_b",     16384,  2048},   /* q up-proj, all 64 heads */
        {"wq_b_s8",   2048,  2048},   /* q up-proj, 8-way head shard */
        {"wkv_a",      576,  6144},   /* latent kv proj */
        {"wo",        6144, 16384},   /* o-proj, all heads */
        {"wo_s8",     6144,  2048},   /* o-proj, 8-way head shard */
        {"router",     256,  6144},   /* MoE gate (bf16 here; f32 in checkpoint) */
        {"ex_w13",    2048,  6144},   /* routed/shared expert gate/up */
        {"ex_w2",     6144,  2048},   /* routed/shared expert down */
        {"head_s8",  19360,  6144},   /* lm_head, 8-way vocab shard */
    };
    const int nsh = (int)(sizeof shapes / sizeof shapes[0]);
    int Mmax = 32; { const char*e=getenv("KB_MMAX"); if(e&&*e) Mmax=atoi(e); if(Mmax<1)Mmax=1; if(Mmax>64)Mmax=64; }
    double tgt_ms = 200; { const char*e=getenv("KB_REPS_MS"); if(e&&*e) tgt_ms=atof(e); }
    static const int Msweep[] = {1,2,4,8,16,32,64};

    printf("KERNBENCH,begin,threads=%s,Mmax=%d\n", getenv("OMP_NUM_THREADS")?getenv("OMP_NUM_THREADS"):"?", Mmax);
    for(int s=0;s<nsh;s++){
        const kb_shape*sh=&shapes[s];
        size_t wbytes=(size_t)sh->rows*sh->cols*2;
        uint16_t*W=glm5_amalloc(wbytes);
        float*X=glm5_amalloc((size_t)Mmax*sh->cols*4);
        float*Y=glm5_amalloc((size_t)Mmax*sh->rows*4);
        if(!W||!X||!Y){ fprintf(stderr,"alloc failed for %s\n",sh->name); return 1; }
        glm5_sm=0x6b62ULL+ (uint64_t)s;
        glm5_fill_bf16(W,(size_t)sh->rows*sh->cols,0.03f);
        for(size_t i=0;i<(size_t)Mmax*sh->cols;i++) X[i]=0.001f*(float)((i%97)+1);
        for(int mi=0;mi<(int)(sizeof Msweep/sizeof Msweep[0]);mi++){
            int M=Msweep[mi]; if(M>Mmax) break;
            glm5_gemm_bf16(Y,W,X,M,sh->rows,sh->cols);            /* warmup / page-in */
            double t1=ksec(); glm5_gemm_bf16(Y,W,X,M,sh->rows,sh->cols); double one=ksec()-t1;
            int reps=(int)(tgt_ms/1e3/(one>1e-6?one:1e-6)); if(reps<3)reps=3; if(reps>2000)reps=2000;
            double t0=ksec();
            for(int r=0;r<reps;r++) glm5_gemm_bf16(Y,W,X,M,sh->rows,sh->cols);
            double dt=(ksec()-t0)/reps;
            double gbs=(double)wbytes/dt/1e9;                     /* weight-stream rate (the decode-relevant number) */
            printf("KERNBENCH,%s,%d,%d,%d,%.3f,%.1f,%.1f\n",
                   sh->name,sh->rows,sh->cols,M,dt*1e3,gbs,gbs/M);
        }
        glm5_afree(W); glm5_afree(X); glm5_afree(Y);
    }
    printf("KERNBENCH,end\n");
    printf("KERNBENCH note: decode_sim.BW_NODE ~= weighted M=1 GBs over the per-token shape mix;\n");
    printf("KERNBENCH note: batched-decode compute time/token ~= sum(shape ms at M)/M.\n");
    return 0;
}
