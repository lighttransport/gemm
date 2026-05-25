/*
 * bench_cuda_fa2.c
 *
 * FlashAttention-2 forward benchmark for sm_120 (RTX 5060 Ti).
 * Follows the gau-nernst "fa-5090" recipe; mirrors cuda/fa/bench_cuda_fa.c.
 *
 * Math: O[b,h,i,d] = sum_j softmax(Q[i]·K[j] * scale) * V[j], scale=1/sqrt(D).
 * Q,K,V,O all [B*H, S, D]. dtype f16 | bf16 (fp8 = e4m3 QK + bf16 PV, later).
 * Causal optional (--causal 1).
 *
 * FLOP count: non-causal 4*B*H*S*S*D; causal 4*B*H*D*S*(S+1)/2.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <dlfcn.h>

#include "../cuew.h"

#define CUDA_RUNNER_COMMON_IMPLEMENTATION
#include "../cuda_runner_common.h"

#include "cuda_fa2_kernels.h"

typedef enum { DT_F16, DT_BF16, DT_FP8 } dtype_t;
typedef enum { MD_MMA, MD_REF, MD_ALL } bmode_t;

typedef struct { const char *name; int B, H, S, D; } shape_t;

static const shape_t g_shapes[] = {
    {"qwen3_512",  1, 16, 512,  128},
    {"qwen3_2k",   1, 16, 2048, 128},
    {"qwen3_4k",   1, 16, 4096, 128},
    {"dit_1k",     1, 24, 1024, 64 },
    {"sd_8k_d64",  1,  8, 8192, 64 },
    {"qwen35_2k",  1, 16, 2048, 256},   /* Qwen3.5 hybrid head_dim=256 */
    {"qwen35_512", 1, 16, 512,  256},
};
#define N_SHAPES ((int)(sizeof(g_shapes)/sizeof(g_shapes[0])))

/* ---------- dtype conversion (host) ---------- */
static uint16_t f32_to_f16(float f) { return cu_f32_to_f16(f); }
static float f16_to_f32(uint16_t h) {
    uint32_t sign=(h>>15)&1, exp=(h>>10)&0x1F, mant=h&0x3FF, fe, fm;
    if (exp==0 && mant==0) { uint32_t b=sign<<31; float v; memcpy(&v,&b,4); return v; }
    if (exp==0) { float v=(float)mant/(float)(1<<24); return sign?-v:v; }
    if (exp==31) { fe=255; fm=mant<<13; } else { fe=exp+(127-15); fm=mant<<13; }
    uint32_t b=(sign<<31)|(fe<<23)|fm; float v; memcpy(&v,&b,4); return v;
}
static uint16_t f32_to_bf16(float f) {
    uint32_t b; memcpy(&b,&f,4);
    uint32_t r = 0x7FFFu + ((b>>16)&1u);   /* round to nearest even */
    return (uint16_t)((b + r) >> 16);
}
static float bf16_to_f32(uint16_t h) {
    uint32_t b=((uint32_t)h)<<16; float v; memcpy(&v,&b,4); return v;
}
static uint16_t f32_to_store(float f, dtype_t dt) {
    return (dt==DT_BF16) ? f32_to_bf16(f) : f32_to_f16(f);
}
static float store_to_f32(uint16_t h, dtype_t dt) {
    return (dt==DT_BF16) ? bf16_to_f32(h) : f16_to_f32(h);
}
/* e4m3 (1s/4e bias7/3m) -> f32. Quantizer never emits the e15m7 NaN. */
static float fp8e4m3_to_f32(uint8_t b) {
    uint32_t s=(b>>7)&1u, e=(b>>3)&0xFu, m=b&0x7u; float v;
    if (e==0) v = (float)m * (1.0f/512.0f);            /* subnormal: m*2^-9 */
    else      v = (1.0f + (float)m/8.0f) * exp2f((float)e - 7.0f);
    return s ? -v : v;
}
/* O is f16 for f16, bf16 for bf16 *and* fp8 (V/PV stay bf16). */
static float out_to_f32(uint16_t h, dtype_t dt) {
    return (dt==DT_F16) ? f16_to_f32(h) : bf16_to_f32(h);
}

/* ---------- CPU FP32 reference (causal-aware) ---------- */
static void cpu_fa_ref(float *O, const float *Q, const float *K, const float *V,
                       int B, int H, int S, int D, int q_rows, int causal) {
    float scale = 1.0f / sqrtf((float)D);
    for (int bh = 0; bh < B*H; bh++) {
        for (int i = 0; i < q_rows; i++) {
            const float *q = Q + ((size_t)bh*S + i) * D;
            int klim = causal ? (i+1) : S;
            float *scores = (float*)malloc(sizeof(float)*S);
            float mx = -1e30f;
            for (int j = 0; j < klim; j++) {
                float s = 0; const float *k = K + ((size_t)bh*S + j)*D;
                for (int d = 0; d < D; d++) s += q[d]*k[d];
                s *= scale; scores[j] = s; if (s > mx) mx = s;
            }
            float sum = 0;
            for (int j = 0; j < klim; j++) { scores[j]=expf(scores[j]-mx); sum+=scores[j]; }
            float inv = 1.0f/sum;
            float *o = O + ((size_t)bh*q_rows + i)*D;
            for (int d = 0; d < D; d++) {
                float acc = 0;
                for (int j = 0; j < klim; j++) acc += scores[j]*V[((size_t)bh*S + j)*D + d];
                o[d] = acc*inv;
            }
            free(scores);
        }
    }
}

typedef struct { int ok; double cos_sim; float max_abs_err; } acc_result_t;
static acc_result_t validate(const float *got, const float *ref, int n, double cos_thresh) {
    acc_result_t r = {0,0,0}; double dot=0,ng=0,nr=0; float me=0;
    for (int i=0;i<n;i++){ float g=got[i],rr=ref[i],e=fabsf(g-rr); if(e>me)me=e;
        dot+=(double)g*rr; ng+=(double)g*g; nr+=(double)rr*rr; }
    r.cos_sim=(ng>0&&nr>0)?dot/(sqrt(ng)*sqrt(nr)):0; r.max_abs_err=me;
    r.ok=(r.cos_sim>=cos_thresh); return r;
}

/* ---------- source builder: prepend #define header ---------- */
static char *build_src(const char *body, int D, int BR, int BC, int causal, dtype_t dt) {
    char hdr[512];
    snprintf(hdr, sizeof(hdr),
        "#define FA2_D %d\n#define FA2_BR %d\n#define FA2_BC %d\n#define FA2_CAUSAL %d\n%s",
        D, BR, BC, causal, (dt==DT_BF16) ? "#define FA2_BF16 1\n" : "");
    size_t n = strlen(hdr) + strlen(body) + 1;
    char *s = (char*)malloc(n);
    strcpy(s, hdr); strcat(s, body);
    return s;
}

/* per-D block sizing: BR=64 (4 warps x 16 rows); BC=32, except D=256 -> 16 (smem/regs) */
static int pick_BC(int D) { return (D >= 256) ? 16 : 32; }

int main(int argc, char **argv) {
    dtype_t dt = DT_BF16;
    bmode_t md = MD_ALL;
    const char *shape_name = "all";
    int B=0,H=0,S=0,D=0, causal=0;
    int iters=50, warmup=5, verify=1, verify_qrows=8, verbose=0;

    for (int i=1;i<argc;i++){
        if (!strcmp(argv[i],"--dtype")&&i+1<argc){ const char*s=argv[++i];
            if(!strcmp(s,"f16"))dt=DT_F16; else if(!strcmp(s,"bf16"))dt=DT_BF16;
            else if(!strcmp(s,"fp8"))dt=DT_FP8; else {fprintf(stderr,"bad dtype %s\n",s);return 1;} }
        else if (!strcmp(argv[i],"--mode")&&i+1<argc){ const char*s=argv[++i];
            if(!strcmp(s,"mma")||!strcmp(s,"ptx"))md=MD_MMA; else if(!strcmp(s,"ref"))md=MD_REF;
            else if(!strcmp(s,"all"))md=MD_ALL; else {fprintf(stderr,"bad mode %s\n",s);return 1;} }
        else if (!strcmp(argv[i],"--shape")&&i+1<argc) shape_name=argv[++i];
        else if (!strcmp(argv[i],"--batch")&&i+1<argc) B=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--heads")&&i+1<argc) H=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--seqlen")&&i+1<argc) S=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--head-dim")&&i+1<argc) D=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--causal")&&i+1<argc) causal=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--iters")&&i+1<argc) iters=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--warmup")&&i+1<argc) warmup=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--verify")&&i+1<argc) verify=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--verify-qrows")&&i+1<argc) verify_qrows=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--verbose")&&i+1<argc) verbose=atoi(argv[++i]);
        else if (!strcmp(argv[i],"-h")||!strcmp(argv[i],"--help")){
            printf("Usage: %s [--dtype f16|bf16|fp8] [--mode mma|ref|all] [--shape NAME|all]\n",argv[0]);
            printf("  [--causal 0|1] [--batch B --heads H --seqlen S --head-dim D]\n");
            printf("  [--iters N] [--warmup N] [--verify 0|1] [--verify-qrows N] [--verbose N]\n");
            for(int k=0;k<N_SHAPES;k++) printf("  shape %-10s B=%d H=%2d S=%5d D=%3d\n",
                g_shapes[k].name,g_shapes[k].B,g_shapes[k].H,g_shapes[k].S,g_shapes[k].D);
            return 0; }
        else { fprintf(stderr,"unknown arg %s\n",argv[i]); return 1; }
    }

    if (cuewInit(CUEW_INIT_CUDA|CUEW_INIT_NVRTC)!=CUEW_SUCCESS){fprintf(stderr,"cuewInit failed\n");return 1;}
    if (cuInit(0)!=CUDA_SUCCESS){fprintf(stderr,"cuInit failed\n");return 1;}
    CUdevice dev; CUcontext ctx; cuDeviceGet(&dev,0); cuCtxCreate(&ctx,0,dev);
    char dn[256]; int sm_maj=0,sm_min=0,sm_count=0,clk=0;
    cuDeviceGetName(dn,sizeof(dn),dev);
    cuDeviceGetAttribute(&sm_maj,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,dev);
    cuDeviceGetAttribute(&sm_min,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,dev);
    cuDeviceGetAttribute(&sm_count,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,dev);
    cuDeviceGetAttribute(&clk,CU_DEVICE_ATTRIBUTE_CLOCK_RATE,dev);
    printf("Device: %s sm_%d%d SMs=%d clock=%d MHz | dtype=%s causal=%d\n",
           dn,sm_maj,sm_min,sm_count,clk/1000,
           dt==DT_BF16?"bf16":dt==DT_F16?"f16":"fp8", causal);
    CUstream stream=NULL; cuStreamCreate(&stream,0);
    srand(42);

    /* shape list */
    const shape_t *shapes[N_SHAPES]; int nsh=0;
    shape_t adhoc;
    if (B>0&&H>0&&S>0&&D>0){ adhoc=(shape_t){"adhoc",B,H,S,D}; shapes[nsh++]=&adhoc; }
    else if (!strcmp(shape_name,"all")) { for(int i=0;i<N_SHAPES;i++) shapes[nsh++]=&g_shapes[i]; }
    else { const shape_t*sh=NULL; for(int i=0;i<N_SHAPES;i++) if(!strcmp(g_shapes[i].name,shape_name)){sh=&g_shapes[i];break;}
           if(!sh){fprintf(stderr,"unknown shape %s\n",shape_name);return 1;} shapes[nsh++]=sh; }

    for (int si=0; si<nsh; si++) {
        const shape_t *sh = shapes[si];
        int b=sh->B,h=sh->H,s=sh->S,d=sh->D,bh=b*h;
        if (d!=64 && d!=128 && d!=256) { fprintf(stderr,"skip %s: D=%d unsupported (64/128/256)\n",sh->name,d); continue; }
        int BR=64, BC=pick_BC(d);

        size_t qn=(size_t)bh*s*d;
        int qk_es = (dt==DT_FP8) ? 1 : 2;   /* Q,K element size; V always bf16/f16 16-bit */
        float *Qf=malloc(qn*4),*Kf=malloc(qn*4),*Vf=malloc(qn*4);
        for(size_t i=0;i<qn;i++){ Qf[i]=((float)rand()/RAND_MAX-0.5f)*2; Kf[i]=((float)rand()/RAND_MAX-0.5f)*2; Vf[i]=((float)rand()/RAND_MAX-0.5f)*2; }
        void *Qs=malloc(qn*qk_es),*Ks=malloc(qn*qk_es); uint16_t *Vh=malloc(qn*2);
        float sQ=1.0f, sK=1.0f;
        if (dt==DT_FP8) {
            float mq=0,mk=0; for(size_t i=0;i<qn;i++){ float a=fabsf(Qf[i]); if(a>mq)mq=a; a=fabsf(Kf[i]); if(a>mk)mk=a; }
            sQ=(mq>0)?mq/448.0f:1.0f; sK=(mk>0)?mk/448.0f:1.0f;
            uint8_t *Qb=Qs,*Kb=Ks;
            for(size_t i=0;i<qn;i++){ Qb[i]=cu_f32_to_fp8_e4m3(Qf[i]/sQ); Kb[i]=cu_f32_to_fp8_e4m3(Kf[i]/sK); Vh[i]=f32_to_bf16(Vf[i]); }
            if (verify) for(size_t i=0;i<qn;i++){ Qf[i]=fp8e4m3_to_f32(Qb[i])*sQ; Kf[i]=fp8e4m3_to_f32(Kb[i])*sK; Vf[i]=bf16_to_f32(Vh[i]); }
        } else {
            uint16_t *Qh=Qs,*Kh=Ks;
            for(size_t i=0;i<qn;i++){ Qh[i]=f32_to_store(Qf[i],dt); Kh[i]=f32_to_store(Kf[i],dt); Vh[i]=f32_to_store(Vf[i],dt); }
            if (verify) for(size_t i=0;i<qn;i++){ Qf[i]=store_to_f32(Qh[i],dt); Kf[i]=store_to_f32(Kh[i],dt); Vf[i]=store_to_f32(Vh[i],dt); }
        }

        int vq = verify_qrows; if (vq>s) vq=s;
        float *Oref=NULL;
        if (verify){ Oref=malloc((size_t)bh*vq*d*4); cpu_fa_ref(Oref,Qf,Kf,Vf,b,h,s,d,vq,causal); }

        CUdeviceptr dQ=0,dK=0,dV=0,dO=0;
        cuMemAlloc(&dQ,qn*qk_es); cuMemAlloc(&dK,qn*qk_es); cuMemAlloc(&dV,qn*2); cuMemAlloc(&dO,qn*2);
        cuMemcpyHtoD(dQ,Qs,qn*qk_es); cuMemcpyHtoD(dK,Ks,qn*qk_es); cuMemcpyHtoD(dV,Vh,qn*2);

        uint16_t *Ohb=malloc(qn*2);
        float *Ogot=malloc((size_t)bh*vq*d*4);
        double pairs = causal ? ((double)s*(s+1)/2.0) : ((double)s*s);
        double flops = 4.0*(double)bh*pairs*(double)d;
        float scale = 1.0f/sqrtf((float)d);
        /* fp8: fold per-tensor descale (s_q*s_k) into the score scale */
        float launch_scale = (dt==DT_FP8) ? scale*sQ*sK : scale;

        bmode_t order[2]; int nmd=0;
        if (md==MD_MMA||md==MD_ALL) order[nmd++]=MD_MMA;
        if ((md==MD_REF||md==MD_ALL) && dt!=DT_FP8) order[nmd++]=MD_REF;  /* fp8: MMA only */

        for (int mi=0; mi<nmd; mi++) {
            bmode_t m = order[mi];
            cuMemsetD8(dO,0,qn*2);
            CUmodule mod; CUfunction fn; size_t smem; int block, gx;
            const char *body = (m==MD_MMA) ? (dt==DT_FP8 ? k_fa2_attn_fp8_src : k_fa2_attn_src) : k_fa2_ref_src;
            const char *kname = (m==MD_MMA) ? (dt==DT_FP8 ? "fa2_attn_fp8" : "fa2_attn") : "fa2_ref";
            char *src = build_src(body, d, BR, BC, causal, dt);
            if (cu_compile_kernels(&mod,dev,src,kname,verbose,"bench_cuda_fa2")<0){ free(src); continue; }
            free(src);
            if (cuModuleGetFunction(&fn,mod,kname)!=CUDA_SUCCESS){ fprintf(stderr,"getFunc %s failed\n",kname); cuModuleUnload(mod); continue; }

            if (m==MD_MMA){
                /* fp8: K e4m3 rows padded to D+16 bytes + V bf16 rows D+8 halves, x2 buffers */
                if (dt==DT_FP8) smem=(size_t)2*BC*(d+16) + (size_t)2*BC*(d+8)*sizeof(uint16_t);
                else            smem=(size_t)4*BC*(d+8)*sizeof(uint16_t);
                block=128; gx=(s+BR-1)/BR;
            }
            else { smem=(size_t)s*sizeof(float); block=d; gx=s; }
            if (smem > 48*1024) cuFuncSetAttribute(fn,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,(int)smem);

            void *args[]={ &dO,&dQ,&dK,&dV,&s,&d,&launch_scale };
            int launch_ok=1;
            for(int w=0;w<warmup;w++) if(cuLaunchKernel(fn,gx,bh,1,block,1,1,smem,stream,args,NULL)!=CUDA_SUCCESS){launch_ok=0;break;}
            cuStreamSynchronize(stream);
            if(!launch_ok){ fprintf(stderr,"launch failed (%s %s D=%d)\n",kname,sh->name,d); cuModuleUnload(mod); continue; }

            CUevent t0,t1; cuEventCreate(&t0,0); cuEventCreate(&t1,0);
            cuEventRecord(t0,stream);
            for(int it=0;it<iters;it++) cuLaunchKernel(fn,gx,bh,1,block,1,1,smem,stream,args,NULL);
            cuEventRecord(t1,stream); cuEventSynchronize(t1);
            float ms=0; cuEventElapsedTime(&ms,t0,t1); ms/=iters;
            cuEventDestroy(t0); cuEventDestroy(t1);

            double tflops = flops/(ms*1e9);
            acc_result_t acc={1,1.0,0};
            if (verify){
                cuMemcpyDtoH(Ohb,dO,qn*2);
                for(int x=0;x<bh;x++) for(int i=0;i<vq;i++) for(int dd=0;dd<d;dd++)
                    Ogot[((size_t)x*vq+i)*d+dd]=out_to_f32(Ohb[((size_t)x*s+i)*d+dd],dt);
                double thr = (dt==DT_FP8)?0.99:0.999;
                acc=validate(Ogot,Oref,bh*vq*d,thr);
            }
            printf("dtype=%s mode=%-3s shape=%-10s B=%d H=%2d S=%5d D=%3d causal=%d  ms=%7.3f  TFLOP/s=%6.2f  %s",
                   dt==DT_BF16?"bf16":dt==DT_F16?"f16":"fp8",
                   m==MD_MMA?"mma":"ref", sh->name, b,h,s,d,causal, ms,tflops, acc.ok?"ACC_OK":"ACC_FAIL");
            if (verify) printf("  cos=%.5f max_err=%.4g", acc.cos_sim, acc.max_abs_err);
            printf("\n");
            cuModuleUnload(mod);
        }

        cuMemFree(dQ);cuMemFree(dK);cuMemFree(dV);cuMemFree(dO);
        free(Qf);free(Kf);free(Vf);free(Qs);free(Ks);free(Vh);free(Ohb);free(Ogot);
        if(Oref)free(Oref);
    }

    cuStreamDestroy(stream); cuCtxDestroy(ctx);
    return 0;
}
