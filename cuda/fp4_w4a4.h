/* fp4_w4a4.h — shared NVFP4 W4A4 OMMA GEMM kernels for sm_120a (consumer
 * Blackwell, RTX 50-series). Hand-written `mma.sync m16n8k64 block_scale ue4m3`
 * because cuBLAS exposes no FP4 GEMM on sm_120 (datacenter Blackwell only).
 *
 * This is a verbatim lift of the qimg FP4 kernels (cuda/qimg/cuda_qimg_runner.h
 * lines 3367-3489) wrapped in a runner-agnostic API. Originally validated
 * against a CPU reference in cuda/fp4/fp4_w4a4_opt.c and against a real
 * Nunchaku SVDQuant tensor in cuda/fp4/fp4_w4a4_qkv_test.c (rel_L2 = 0.0947, at
 * the W4A4 activation-quant floor). The naive `w4a4_gemm` is kept as an A/B
 * oracle (~15 TOPS, traffic-bound); `w4a4_gemm_opt` is the production kernel
 * (BM=64 BN=128 BK=64 shared-mem tile, 256 threads, ~62-72 TOPS).
 *
 * Layout (matches the runtime; produced offline by *_fp4_repack tools):
 *   qw  int32  [n_out, n_in/8]   8 e2m1 codes / uint32 (low nib = low k)
 *   ws  uint8  [n_out, n_in/16]  raw e4m3 micro-scales (linear, NOT swizzled)
 *   wcwt f32   [n_out]           per-output-channel scale
 *
 * Dispatch computes: Y[m,n] = (Σ_k e2m1(code[n,k]) * e4m3(ws[n,k/16])
 *                              * e2m1(act_code[m,k]) * e4m3(act_scale[m,k/16]))
 *                              * wcwt[n]
 *                            + (low-rank, optional via pd/pu)
 *                            + bias
 * Activations are dynamically quantized per-token per-block-16 inside the
 * dispatch (fp4_quant_act). No checkpoint `input_scale` is consulted — dynamic
 * per-token amax is at least as accurate as ModelOpt's static calibration.
 *
 * Caller owns the ctx struct + scratch buffers (grown lazily on first call).
 * Requirements: K % 64 == 0 (block_scale tile size); n_out % 128 padded by
 * w4a4_gemm_opt's M/N guards (any n_out works).
 */
#ifndef FP4_W4A4_H
#define FP4_W4A4_H

#include <stdio.h>
#include <string.h>
#include "cuew.h"
#include "cuew_ptx_compiler.h"  /* nvrtcCreateProgram etc */

/* ----------------------------------------------------------------------------
 * Kernel source (verbatim from cuda/qimg/cuda_qimg_runner.h:3367-3489).
 * Compiled at --gpu-architecture=sm_120a (the block-scale MMA needs sm_120a;
 * plain sm_120 will not assemble the kind::mxf4nvf4.block_scale variant).
 * -------------------------------------------------------------------------- */
static const char *fp4_w4a4_kernel_src =
"__device__ __forceinline__ unsigned char qf_f32_to_e4m3(float v){\n"
"  unsigned short p; asm(\"cvt.rn.satfinite.e4m3x2.f32 %0,%1,%2;\":\"=h\"(p):\"f\"(v),\"f\"(v)); return p&0xFF; }\n"
"__device__ __forceinline__ float qf_e4m3_dec(unsigned char b){int e=(b>>3)&0xF,m=b&7;\n"
"  if(e==0) return (m/8.0f)*0.015625f; return (1.0f+m/8.0f)*exp2f((float)(e-7)); }\n"
"__device__ __forceinline__ unsigned char qf_near_e2m1(float q){\n"
"  unsigned char s=q<0?8:0; float a=fabsf(q); unsigned char c;\n"
"  if(a<0.25f)c=0;else if(a<0.75f)c=1;else if(a<1.25f)c=2;else if(a<1.75f)c=3;\n"
"  else if(a<2.5f)c=4;else if(a<3.5f)c=5;else if(a<5.0f)c=6;else c=7; return s|c; }\n"
"extern \"C\" __global__ void fp4_quant_act(const float* X, unsigned int* Ac, unsigned char* As, int M, int K){\n"
"  int ng=K>>4; long gi=(long)blockIdx.x*blockDim.x+threadIdx.x; if(gi>=(long)M*ng) return;\n"
"  int m=gi/ng, grp=gi%ng; long k0=(long)grp*16;\n"
"  float amax=0; for(int i=0;i<16;i++){float v=fabsf(X[(long)m*K+k0+i]); amax=fmaxf(amax,v);}\n"
"  float sc=amax*0.16666667f; unsigned char sb=qf_f32_to_e4m3(sc); float sd=qf_e4m3_dec(sb);\n"
"  float inv = sd>0.f? 1.0f/sd : 0.f; As[(long)m*ng+grp]=sb;\n"
"  for(int u=0;u<2;u++){ unsigned int v=0; for(int i=0;i<8;i++){ long k=k0+u*8+i;\n"
"      float q=X[(long)m*K+k]*inv; v|=((unsigned)qf_near_e2m1(q))<<(i*4);} Ac[(long)m*(K>>3)+grp*2+u]=v; } }\n"
"extern \"C\" __global__ void w4a4_gemm(const unsigned int* A, const unsigned int* B,\n"
"     const unsigned char* sA, const unsigned char* sB, float* D, int M, int N, int K){\n"
"  int warp=(blockIdx.x*blockDim.x+threadIdx.x)>>5; int lane=threadIdx.x&31;\n"
"  int ntn=N>>3; int tm=warp/ntn, tn=warp%ntn; int M0=tm*16, N0=tn*8; if(M0>=M) return;\n"
"  int g=lane>>2, t=lane&3; int Ku=K>>3, Kg=K>>4;\n"
"  long rA0=(long)(M0+g)*Ku, rA1=(long)(M0+g+8)*Ku, cB=(long)(N0+g)*Ku;\n"
"  int saRow=(t==0)?(M0+g):(t==1)?(M0+g+8):-1; int sbCol=(t==0)?(N0+g):-1;\n"
"  float d0=0.f,d1=0.f,d2=0.f,d3=0.f; int nkc=K>>6;\n"
"  for(int kc=0;kc<nkc;kc++){\n"
"    unsigned int a0=A[rA0+kc*8+t],a1=A[rA1+kc*8+t],a2=A[rA0+kc*8+t+4],a3=A[rA1+kc*8+t+4];\n"
"    unsigned int b0=B[cB+kc*8+t],b1=B[cB+kc*8+t+4];\n"
"    unsigned int sfa=0x38383838u,sfb=0x38383838u;\n"
"    if(saRow>=0){const unsigned char*p=&sA[(long)saRow*Kg+kc*4]; sfa=p[0]|(p[1]<<8)|(p[2]<<16)|(p[3]<<24);}\n"
"    if(sbCol>=0){const unsigned char*p=&sB[(long)sbCol*Kg+kc*4]; sfb=p[0]|(p[1]<<8)|(p[2]<<16)|(p[3]<<24);}\n"
"    asm volatile(\"mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 \"\n"
"      \"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};\"\n"
"      : \"+f\"(d0),\"+f\"(d1),\"+f\"(d2),\"+f\"(d3)\n"
"      : \"r\"(a0),\"r\"(a1),\"r\"(a2),\"r\"(a3),\"r\"(b0),\"r\"(b1),\"r\"(sfa),\"r\"(sfb));\n"
"  }\n"
"  D[(long)(M0+g)*N+N0+2*t]=d0; D[(long)(M0+g)*N+N0+2*t+1]=d1;\n"
"  D[(long)(M0+g+8)*N+N0+2*t]=d2; D[(long)(M0+g+8)*N+N0+2*t+1]=d3; }\n"
"extern \"C\" __global__ void w4a4_combine(float* Y, const float* D, const float* wcwt,\n"
"     const float* lo, const float* bias, int n_tok, int n_out){\n"
"  long i=(long)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(long)n_tok*n_out) return;\n"
"  int o=i%n_out; Y[i]=D[i]*wcwt[o]+(lo?lo[i]:0.f)+(bias?bias[o]:0.f); }\n"
/* ---- Optimized W4A4 GEMM: BM=64 BN=128 BK=64 tile, 256 threads (8 warps 2x4),
 * each warp computes 32x32 = 2 m-subtiles x 4 n-subtiles. A/B/scale tiles are
 * staged in shared memory once per K-tile and reused across the warp's subtiles
 * (A-frags reused across N, B-frags across M), cutting the redundant global
 * traffic of the naive 1-warp-per-16x8 kernel by ~8x. Bit-identical to the naive
 * w4a4_gemm (verified vs the naive oracle for full + partial M/N blocks and vs a
 * CPU reference in cuda/fp4/fp4_w4a4_opt.c); ~4-6x faster (15 -> 62-72 TOPS).
 * M/N guarded (any n_tok/n_out); requires K%64==0 (true for all qimg/flux2 linears). ---- */
"#define BM 64\n"
"#define BN 128\n"
"#define BK 64\n"
"#define WN_WARPS 4\n"
"#define MSUB 2\n"
"#define NSUB 4\n"
"extern \"C\" __global__ __launch_bounds__(256) void w4a4_gemm_opt(\n"
"    const unsigned int* __restrict__ A, const unsigned int* __restrict__ B,\n"
"    const unsigned char* __restrict__ sA, const unsigned char* __restrict__ sB,\n"
"    float* __restrict__ D, int M, int N, int K){\n"
"  __shared__ unsigned int  smA[BM][BK/8];\n"
"  __shared__ unsigned int  smB[BN][BK/8];\n"
"  __shared__ unsigned char smSA[BM][BK/16];\n"
"  __shared__ unsigned char smSB[BN][BK/16];\n"
"  int bm0=blockIdx.y*BM, bn0=blockIdx.x*BN;\n"
"  int tid=threadIdx.x, warp=tid>>5, lane=tid&31;\n"
"  int wm=warp/WN_WARPS, wn=warp%WN_WARPS;\n"
"  int g=lane>>2, t=lane&3;\n"
"  long Ku=K>>3, Kg=K>>4;\n"
"  float acc[MSUB][NSUB][4];\n"
"  #pragma unroll\n"
"  for(int i=0;i<MSUB;i++)for(int j=0;j<NSUB;j++)for(int e=0;e<4;e++)acc[i][j][e]=0.f;\n"
"  int nkt=K/BK;\n"
"  for(int kt=0; kt<nkt; kt++){\n"
"    long k0u=(long)kt*(BK/8), k0g=(long)kt*(BK/16);\n"
"    for(int idx=tid; idx<BM*(BK/8); idx+=256){ int r=idx/(BK/8), c=idx%(BK/8);\n"
"      int gr=bm0+r; smA[r][c]=(gr<M)?A[(long)gr*Ku+k0u+c]:0u; }\n"
"    for(int idx=tid; idx<BN*(BK/8); idx+=256){ int r=idx/(BK/8), c=idx%(BK/8);\n"
"      int gr=bn0+r; smB[r][c]=(gr<N)?B[(long)gr*Ku+k0u+c]:0u; }\n"
"    for(int idx=tid; idx<BM*(BK/16); idx+=256){ int r=idx/(BK/16), c=idx%(BK/16);\n"
"      int gr=bm0+r; smSA[r][c]=(gr<M)?sA[(long)gr*Kg+k0g+c]:0x38; }\n"
"    for(int idx=tid; idx<BN*(BK/16); idx+=256){ int r=idx/(BK/16), c=idx%(BK/16);\n"
"      int gr=bn0+r; smSB[r][c]=(gr<N)?sB[(long)gr*Kg+k0g+c]:0x38; }\n"
"    __syncthreads();\n"
"    unsigned int af[MSUB][4], sfa[MSUB], bf[NSUB][2], sfb[NSUB];\n"
"    #pragma unroll\n"
"    for(int i=0;i<MSUB;i++){ int mr=wm*(BM/2)+i*16;\n"
"      af[i][0]=smA[mr+g][t]; af[i][1]=smA[mr+g+8][t];\n"
"      af[i][2]=smA[mr+g][t+4]; af[i][3]=smA[mr+g+8][t+4];\n"
"      unsigned int s=0x38383838u;\n"
"      if(t==0){const unsigned char*p=smSA[mr+g]; s=p[0]|(p[1]<<8)|(p[2]<<16)|(p[3]<<24);}\n"
"      else if(t==1){const unsigned char*p=smSA[mr+g+8]; s=p[0]|(p[1]<<8)|(p[2]<<16)|(p[3]<<24);}\n"
"      sfa[i]=s; }\n"
"    #pragma unroll\n"
"    for(int j=0;j<NSUB;j++){ int nr=wn*(BN/WN_WARPS)+j*8;\n"
"      bf[j][0]=smB[nr+g][t]; bf[j][1]=smB[nr+g][t+4];\n"
"      unsigned int s=0x38383838u;\n"
"      if(t==0){const unsigned char*p=smSB[nr+g]; s=p[0]|(p[1]<<8)|(p[2]<<16)|(p[3]<<24);}\n"
"      sfb[j]=s; }\n"
"    #pragma unroll\n"
"    for(int i=0;i<MSUB;i++)\n"
"      #pragma unroll\n"
"      for(int j=0;j<NSUB;j++){\n"
"        asm volatile(\"mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 \"\n"
"          \"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};\"\n"
"          : \"+f\"(acc[i][j][0]),\"+f\"(acc[i][j][1]),\"+f\"(acc[i][j][2]),\"+f\"(acc[i][j][3])\n"
"          : \"r\"(af[i][0]),\"r\"(af[i][1]),\"r\"(af[i][2]),\"r\"(af[i][3]),\n"
"            \"r\"(bf[j][0]),\"r\"(bf[j][1]),\"r\"(sfa[i]),\"r\"(sfb[j]));\n"
"      }\n"
"    __syncthreads();\n"
"  }\n"
"  #pragma unroll\n"
"  for(int i=0;i<MSUB;i++)\n"
"    #pragma unroll\n"
"    for(int j=0;j<NSUB;j++){\n"
"      int mr=bm0+wm*(BM/2)+i*16, nc=bn0+wn*(BN/WN_WARPS)+j*8;\n"
"      int row0=mr+g, row1=mr+g+8, col=nc+2*t;\n"
"      if(col<N){\n"
"        if(row0<M){D[(long)row0*N+col]=acc[i][j][0]; D[(long)row0*N+col+1]=acc[i][j][1];}\n"
"        if(row1<M){D[(long)row1*N+col]=acc[i][j][2]; D[(long)row1*N+col+1]=acc[i][j][3];}\n"
"      }\n"
"    }\n"
"}\n"
"#undef BM\n#undef BN\n#undef BK\n#undef WN_WARPS\n#undef MSUB\n#undef NSUB\n";


/* ----------------------------------------------------------------------------
 * Context + dispatch API. Caller owns ctx + scratch buffers. The dispatch grows
 * scratch lazily on first call (or whenever a larger n_tok/n_out comes in).
 * -------------------------------------------------------------------------- */

/* One quantized W4A4 linear's resident weights (rhe layout produced by the
 * *_fp4_repack tools). pd/pu are the SVDQuant low-rank residuals — ModelOpt NVFP4
 * has none so they stay NULL; Nunchaku SVDQuant uses them via the optional
 * branch in fp4_w4a4_gemm. */
typedef struct { CUdeviceptr qw, ws, wcwt, pd, pu; } fp4_lin_t;

typedef struct {
    CUstream     stream;
    CUmodule     module;
    CUfunction   k_quant;       /* fp4_quant_act     */
    CUfunction   k_gemm;        /* w4a4_gemm  naive  */
    CUfunction   k_gemm_opt;    /* w4a4_gemm_opt     */
    CUfunction   k_combine;     /* w4a4_combine      */

    /* Scratch buffers, grown on demand. ac = quantized act codes, as = act scales,
     * d  = GEMM output (n_tok*n_out*4 bytes). */
    CUdeviceptr  d_ac, d_as, d_d;
    size_t       ac_n, as_n, d_n;

    int          use_naive;     /* 1 = w4a4_gemm (oracle), 0 = w4a4_gemm_opt (default) */
    int          verbose;
} fp4_w4a4_ctx;


static int fp4_w4a4_compile(fp4_w4a4_ctx *ctx, CUstream stream, int verbose) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->stream  = stream;
    ctx->verbose = verbose;
    nvrtcProgram prog;
    if (nvrtcCreateProgram(&prog, fp4_w4a4_kernel_src, "fp4_w4a4.cu", 0, NULL, NULL) != NVRTC_SUCCESS) {
        fprintf(stderr, "fp4_w4a4: nvrtcCreateProgram failed\n"); return -1;
    }
    const char *opts[] = { "--gpu-architecture=sm_120a" };
    nvrtcResult cr = nvrtcCompileProgram(prog, 1, opts);
    if (cr != NVRTC_SUCCESS) {
        size_t ls = 0; nvrtcGetProgramLogSize(prog, &ls);
        char *lg = (char *)malloc(ls + 1); nvrtcGetProgramLog(prog, lg); lg[ls] = 0;
        fprintf(stderr, "fp4_w4a4: kernel compile FAILED:\n%s\n", lg);
        free(lg); nvrtcDestroyProgram(&prog); return -1;
    }
    CUmodule m = NULL; size_t bs = 0;
    if (nvrtcGetCUBINSize && nvrtcGetCUBINSize(prog, &bs) == NVRTC_SUCCESS && bs > 0) {
        char *bl = (char *)malloc(bs); nvrtcGetCUBIN(prog, bl); nvrtcDestroyProgram(&prog);
        if (cuModuleLoadData(&m, bl) != CUDA_SUCCESS) m = NULL;
        free(bl);
    } else {
        size_t ps = 0; nvrtcGetPTXSize(prog, &ps);
        char *x = (char *)malloc(ps); nvrtcGetPTX(prog, x);
        nvrtcDestroyProgram(&prog);
        if (cuModuleLoadData(&m, x) != CUDA_SUCCESS) m = NULL;
        free(x);
    }
    if (!m) { fprintf(stderr, "fp4_w4a4: cuModuleLoadData failed\n"); return -1; }
    ctx->module = m;
    if (cuModuleGetFunction(&ctx->k_quant,    m, "fp4_quant_act")  != CUDA_SUCCESS) return -1;
    if (cuModuleGetFunction(&ctx->k_gemm,     m, "w4a4_gemm")      != CUDA_SUCCESS) return -1;
    if (cuModuleGetFunction(&ctx->k_gemm_opt, m, "w4a4_gemm_opt")  != CUDA_SUCCESS) return -1;
    if (cuModuleGetFunction(&ctx->k_combine,  m, "w4a4_combine")   != CUDA_SUCCESS) return -1;
    if (verbose) fprintf(stderr, "fp4_w4a4: kernels compiled (sm_120a)\n");
    return 0;
}


static void fp4_w4a4_free(fp4_w4a4_ctx *ctx) {
    if (ctx->d_ac) { cuMemFree(ctx->d_ac); ctx->d_ac = 0; ctx->ac_n = 0; }
    if (ctx->d_as) { cuMemFree(ctx->d_as); ctx->d_as = 0; ctx->as_n = 0; }
    if (ctx->d_d)  { cuMemFree(ctx->d_d);  ctx->d_d  = 0; ctx->d_n  = 0; }
    if (ctx->module) { cuModuleUnload(ctx->module); ctx->module = 0; }
}


/* Grow-on-demand: free + reallocate when need > capacity. */
#define FP4_W4A4_GROW(ptr, cap, need) do { \
    if ((need) > (cap)) { \
        if (ptr) cuMemFree(ptr); \
        if (cuMemAlloc(&(ptr), (need)) != CUDA_SUCCESS) { (ptr) = 0; (cap) = 0; } \
        else (cap) = (need); \
    } \
} while (0)


/* fp4_w4a4_gemm: Y[m,n] = (quant(X) @ W_fp4) * wcwt[n] + bias[n].
 *   X    f32 [n_tok, n_in]
 *   qw   i32 [n_out, n_in/8]   8 e2m1 codes / uint
 *   ws   u8  [n_out, n_in/16]  raw e4m3 micro-scales
 *   wcwt f32 [n_out]           per-output-channel scale
 *   bias f32 [n_out] or 0
 * Constraints: n_in % 64 == 0 (w4a4_gemm_opt's K-tile); any n_tok, n_out.
 * No low-rank (ModelOpt NVFP4 has none; combine zeros the lo term inline). */
static void fp4_w4a4_gemm(fp4_w4a4_ctx *ctx, CUdeviceptr Y,
                          CUdeviceptr qw, CUdeviceptr ws, CUdeviceptr wcwt,
                          CUdeviceptr X, CUdeviceptr bias,
                          int n_out, int n_in, int n_tok) {
    int Mpad = ((n_tok + 15) / 16) * 16;
    size_t ac = (size_t)Mpad * (n_in / 8) * 4;
    size_t as = (size_t)Mpad * (n_in / 16);
    size_t dsz = (size_t)Mpad * n_out * 4;
    FP4_W4A4_GROW(ctx->d_ac, ctx->ac_n, ac);
    FP4_W4A4_GROW(ctx->d_as, ctx->as_n, as);
    FP4_W4A4_GROW(ctx->d_d,  ctx->d_n,  dsz);

    /* 1. fp4_quant_act: per-token per-block-16 amax/6 -> e4m3 scale -> e2m1 codes.
     *    Note this IGNORES any static input_scale from the ModelOpt checkpoint;
     *    dynamic per-token quant is at least as accurate as static calibration. */
    long ng = (long)n_tok * (n_in / 16);
    void *qa[] = { &X, &ctx->d_ac, &ctx->d_as, &n_tok, &n_in };
    cuLaunchKernel(ctx->k_quant, (unsigned)((ng + 255) / 256), 1, 1,
                   256, 1, 1, 0, ctx->stream, qa, NULL);

    /* 2. W4A4 GEMM. Optimized tiled by default; naive (FLUX2_W4A4_NAIVE / QIMG_W4A4_NAIVE)
     *    for A/B verification. */
    if (ctx->use_naive) {
        int nw = (Mpad / 16) * (n_out / 8), th = 128;
        int bl = (nw * 32 + th - 1) / th;
        void *ga[] = { &ctx->d_ac, &qw, &ctx->d_as, &ws, &ctx->d_d, &Mpad, &n_out, &n_in };
        cuLaunchKernel(ctx->k_gemm, (unsigned)bl, 1, 1, th, 1, 1, 0, ctx->stream, ga, NULL);
    } else {
        unsigned bx = (unsigned)((n_out + 127) / 128);
        unsigned by = (unsigned)((n_tok + 63) / 64);
        void *ga[] = { &ctx->d_ac, &qw, &ctx->d_as, &ws, &ctx->d_d, &n_tok, &n_out, &n_in };
        cuLaunchKernel(ctx->k_gemm_opt, bx, by, 1, 256, 1, 1, 0, ctx->stream, ga, NULL);
    }

    /* 3. combine: Y = D*wcwt + bias  (lo skipped — pass NULL to w4a4_combine). */
    long tot = (long)n_tok * n_out;
    CUdeviceptr lo_null = 0;
    void *ca[] = { &Y, &ctx->d_d, &wcwt, &lo_null, &bias, &n_tok, &n_out };
    cuLaunchKernel(ctx->k_combine, (unsigned)((tot + 255) / 256), 1, 1,
                   256, 1, 1, 0, ctx->stream, ca, NULL);
}

#endif /* FP4_W4A4_H */
