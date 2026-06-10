/* Phase 3: mul_mat_id mechanics for the IQ2_XXS MMQ MoE path.
 *
 * Validates the full expert-routed pipeline against a multi-expert CPU oracle:
 *   1. compaction: group (token, slot) pairs by expert -> expert_bounds + ids_token
 *      (mirror of llama.cpp mm_ids_helper / the runner's host repack)
 *   2. gather+quantize activations to q8_1 over the compacted rows
 *   3. grouped MMQ: one dispatch over ALL experts (grid.z = n_experts); each block
 *      picks its expert's compact-row slice via expert_bounds and its own IQ2_XXS
 *      weight matrix -> out_compact[total_rows][N]
 *   4. weighted scatter: final[token] += wgt * out_compact (per compact row)
 *
 * This is the data-flow that replaces the per-expert dequant+cuBLAS loop in
 * cuda_llm_runner.c. Build: make -C cuda/llm/mmq mmq_iq2xxs_moe_test
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "iq2xxs_tables.h"

#define QK_K 256
#define IQ2XXS_BYTES 66
#define CK 32
#define WN 4
#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

__constant__ uint64_t c_grid[256];
__constant__ uint8_t  c_ksigns[128];
__constant__ uint64_t c_signmask[128];  /* per 7-bit index: 8 bytes, each 0x00 or 0xFF */
__device__ static float h2f_dev(uint16_t h){ return __half2float(*(const __half*)&h); }
__device__ __forceinline__ int pack4(const int8_t*p){ return (p[0]&0xff)|((p[1]&0xff)<<8)|((p[2]&0xff)<<16)|((p[3]&0xff)<<24); }
/* p must be 4-byte aligned; sW/sX rows are 32 bytes so [k*4] offsets are aligned. */
__device__ __forceinline__ int pack4a(const signed char*p){ return *(const int*)p; }

/* ---- host half<->float (finite) ---- */
static float h2f(uint16_t h){ uint32_t s=(h>>15)&1,e=(h>>10)&0x1f,m=h&0x3ff,o;
  if(!e){ if(!m)o=s<<31; else{ e=113; while(!(m&0x400)){m<<=1;e--;} m&=0x3ff; o=(s<<31)|(e<<23)|(m<<13);} }
  else if(e==0x1f)o=(s<<31)|(0xff<<23)|(m<<13); else o=(s<<31)|((e-15+127)<<23)|(m<<13);
  float f; __builtin_memcpy(&f,&o,4); return f; }
static uint16_t f2h(float f){ uint32_t x; __builtin_memcpy(&x,&f,4); uint32_t s=(x>>16)&0x8000;
  int32_t e=(int32_t)((x>>23)&0xff)-127+15; uint32_t m=x&0x7fffff;
  if(e<=0)return (uint16_t)s; if(e>=0x1f)return (uint16_t)(s|0x7bff); return (uint16_t)(s|(e<<10)|(m>>13)); }

static void decode_block_h(const uint8_t*bp,float*out){ float d=h2f(*(const uint16_t*)bp);
  const uint16_t*qs=(const uint16_t*)(bp+2);
  for(int ib=0;ib<8;ib++){ uint32_t a0=(uint32_t)qs[4*ib]|((uint32_t)qs[4*ib+1]<<16);
    uint32_t a1=(uint32_t)qs[4*ib+2]|((uint32_t)qs[4*ib+3]<<16); float db=d*(0.5f+(float)(a1>>28))*0.25f;
    for(int l=0;l<4;l++){ uint8_t idx=(a0>>(8*l))&255; const uint8_t*g=(const uint8_t*)&iq2xxs_grid[idx];
      uint8_t s=ksigns_iq2xs[(a1>>(7*l))&127];
      for(int j=0;j<8;j++) out[ib*32+l*8+j]=db*(float)g[j]*((s&(1<<j))?-1.f:1.f); } } }

/* ---- gather rows by token index, quantize to q8_1 (per-32 scale) ---- */
__global__ void gather_quant_q8_1(const float *X, const int *ids_token, int8_t *cxq8, float *cxs,
                                   int total_rows, int K) {
    int kb = blockIdx.x, c = blockIdx.y, t = threadIdx.x, k = kb*CK + t;
    if (c >= total_rows) return;
    int tok = ids_token[c];
    float v = (k < K) ? X[(size_t)tok * K + k] : 0.0f;
    float a = fabsf(v);
    for (int o=16;o>0;o>>=1) a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, o));
    float scale = a/127.0f, inv = scale>0?1.0f/scale:0.0f;
    int q=(int)rintf(v*inv); q=q<-127?-127:(q>127?127:q);
    cxq8[(size_t)c*K + k] = (int8_t)q;
    if (t==0) cxs[(size_t)c*(K/CK) + kb] = scale;
}

/* ---- grouped MMQ: one dispatch over all experts (grid.z = n_experts) ---- */
__global__ void mmq_iq2xxs_grouped(float *out, const uint8_t *W, size_t expert_stride_bytes,
                                    const int8_t *cxq8, const float *cxs, const int *expert_bounds,
                                    int N, int K) {
    int e  = blockIdx.z;
    int eb0 = expert_bounds[e], eb1 = expert_bounds[e+1];
    int m0 = eb0 + blockIdx.y * 8;
    if (m0 >= eb1) return;                       /* this expert has no tokens here */
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int gid = lane>>2, tid = lane&3;
    int n0 = blockIdx.x*(16*WN) + warp*16;
    int nb = K/QK_K, row_bytes = nb*IQ2XXS_BYTES, nsb = K/CK;
    const uint8_t *We = W + (size_t)e * expert_stride_bytes;

    __shared__ int8_t sX[8][CK]; __shared__ float sXs[8];
    __shared__ int8_t sW[16*WN][CK]; __shared__ float sWs[16*WN];
    float f0=0,f1=0,f2=0,f3=0;
    for (int sb=0; sb<nsb; sb++) {
        for (int i=threadIdx.x;i<8*CK;i+=blockDim.x){ int t=i/CK,kk=i%CK,m=m0+t;
            sX[t][kk]=(m<eb1)?cxq8[(size_t)m*K+sb*CK+kk]:0; }
        if (threadIdx.x<8){ int m=m0+threadIdx.x; sXs[threadIdx.x]=(m<eb1)?cxs[(size_t)m*nsb+sb]:0.0f; }
        if (lane<16){ int r=warp*16+lane, n=n0+lane;
            const uint8_t*bp=We+(size_t)n*row_bytes+(sb/8)*IQ2XXS_BYTES;
            float d=h2f_dev(*(const uint16_t*)bp); const uint16_t*qs=(const uint16_t*)(bp+2); int ib=sb&7;
            uint32_t a0=(uint32_t)qs[4*ib]|((uint32_t)qs[4*ib+1]<<16);
            uint32_t a1=(uint32_t)qs[4*ib+2]|((uint32_t)qs[4*ib+3]<<16);
            sWs[r]=d*(0.5f+(float)(a1>>28))*0.25f;
            for(int l=0;l<4;l++){ uint8_t idx=(a0>>(8*l))&255; const uint8_t*g=(const uint8_t*)&c_grid[idx];
                uint8_t s=c_ksigns[(a1>>(7*l))&127];
                for(int j=0;j<8;j++) sW[r][l*8+j]=(int8_t)((int)g[j]*((s&(1<<j))?-1:1)); } }
        __syncthreads();
        int wr=warp*16;
        int a0=pack4(&sW[wr+gid][tid*4]), a1=pack4(&sW[wr+gid+8][tid*4]);
        int a2=pack4(&sW[wr+gid][tid*4+16]), a3=pack4(&sW[wr+gid+8][tid*4+16]);
        int b0=pack4(&sX[gid][tid*4]), b1=pack4(&sX[gid][tid*4+16]);
        int c0=0,c1=0,c2=0,c3=0;
        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
            :"=r"(c0),"=r"(c1),"=r"(c2),"=r"(c3)
            :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(0),"r"(0),"r"(0),"r"(0));
        float wr0=sWs[wr+gid], wr8=sWs[wr+gid+8], xc0=sXs[tid*2], xc1=sXs[tid*2+1];
        f0+=wr0*xc0*(float)c0; f1+=wr0*xc1*(float)c1; f2+=wr8*xc0*(float)c2; f3+=wr8*xc1*(float)c3;
        __syncthreads();
    }
    int n_a=n0+gid, n_b=n0+gid+8, m_a=m0+tid*2, m_b=m0+tid*2+1;
    if (m_a<eb1){ out[(size_t)m_a*N+n_a]=f0; out[(size_t)m_a*N+n_b]=f2; }
    if (m_b<eb1){ out[(size_t)m_b*N+n_a]=f1; out[(size_t)m_b*N+n_b]=f3; }
}

/* ---- grouped MMQ v3: decode-amortized. Each block reuses one weight decode
   across up to TG=4 token-groups (32 tokens) of its expert. Occupancy comes from
   grid.z = n_experts. grid = (N/64, ceil(maxtok/32), n_experts), block=128 ---- */
#define TG 4
__global__ void mmq_iq2xxs_grouped_v3(float *out, const uint8_t *W, unsigned long long estride,
                                       const signed char *cxq8, const float *cxs, const int *ebounds,
                                       int N, int K) {
    int e = blockIdx.z; int eb0 = ebounds[e], eb1 = ebounds[e+1];
    int m_base = eb0 + blockIdx.y * (8*TG);
    if (m_base >= eb1) return;
    int ntg = (eb1 - m_base + 7) / 8; if (ntg > TG) ntg = TG;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int gid = lane>>2, tid = lane&3;
    int n0 = blockIdx.x*64 + warp*16;
    int nb = K/256, row_bytes = nb*66, nsb = K/32;
    const uint8_t *We = W + (size_t)e*estride;
    __shared__ signed char sX[8*TG][32]; __shared__ float sXs[8*TG];
    __shared__ signed char sW[64][32]; __shared__ float sWs[64];
    float f[TG][4]; for(int g=0;g<TG;g++){f[g][0]=f[g][1]=f[g][2]=f[g][3]=0;}
    for (int sb=0; sb<nsb; sb++) {
        if (lane<16){ int r=warp*16+lane, n=n0+lane;
            const uint8_t *bp=We+(size_t)n*row_bytes+(sb/8)*66;
            float d=h2f_dev(*(const uint16_t*)bp); const uint16_t *qs=(const uint16_t*)(bp+2); int ib=sb&7;
            uint32_t a0=(uint32_t)qs[4*ib]|((uint32_t)qs[4*ib+1]<<16);
            uint32_t a1=(uint32_t)qs[4*ib+2]|((uint32_t)qs[4*ib+3]<<16);
            sWs[r]=d*(0.5f+(float)(a1>>28))*0.25f;
            for(int l=0;l<4;l++){ uint8_t idx=(a0>>(8*l))&255; const uint8_t*g=(const uint8_t*)&c_grid[idx];
                uint8_t s=c_ksigns[(a1>>(7*l))&127];
                for(int j=0;j<8;j++) sW[r][l*8+j]=(signed char)((int)g[j]*((s&(1<<j))?-1:1)); } }
        for (int i=threadIdx.x;i<ntg*8*32;i+=blockDim.x){ int t=i/32,kk=i%32,m=m_base+t;
            sX[t][kk]=(m<eb1)?cxq8[(size_t)m*K+sb*32+kk]:0; }
        for (int t=threadIdx.x;t<ntg*8;t+=blockDim.x){ int m=m_base+t; sXs[t]=(m<eb1)?cxs[(size_t)m*nsb+sb]:0.0f; }
        __syncthreads();
        int wr=warp*16;
        int a0=pack4(&sW[wr+gid][tid*4]), a1=pack4(&sW[wr+gid+8][tid*4]);
        int a2=pack4(&sW[wr+gid][tid*4+16]), a3=pack4(&sW[wr+gid+8][tid*4+16]);
        float wr0=sWs[wr+gid], wr8=sWs[wr+gid+8];
        for (int g=0; g<ntg; g++) {
            int b0=pack4(&sX[g*8+gid][tid*4]), b1=pack4(&sX[g*8+gid][tid*4+16]);
            int c0=0,c1=0,c2=0,c3=0;
            asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                :"=r"(c0),"=r"(c1),"=r"(c2),"=r"(c3)
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(0),"r"(0),"r"(0),"r"(0));
            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];
            f[g][0]+=wr0*xc0*(float)c0; f[g][1]+=wr0*xc1*(float)c1;
            f[g][2]+=wr8*xc0*(float)c2; f[g][3]+=wr8*xc1*(float)c3;
        }
        __syncthreads();
    }
    int n_a=n0+gid, n_b=n0+gid+8;
    for (int g=0; g<ntg; g++) {
        int m_a=m_base+g*8+tid*2, m_b=m_a+1;
        if (m_a<eb1){ out[(size_t)m_a*N+n_a]=f[g][0]; out[(size_t)m_a*N+n_b]=f[g][2]; }
        if (m_b<eb1){ out[(size_t)m_b*N+n_a]=f[g][1]; out[(size_t)m_b*N+n_b]=f[g][3]; }
    }
}

/* grouped MMQ v4: v3 + 32-lane decode (all 32 lanes decode, 2 per row, each does
   half the row) to remove the lane<16 half-warp waste during the decode. */
__global__ void mmq_iq2xxs_grouped_v4(float *out, const uint8_t *W, unsigned long long estride,
                                       const signed char *cxq8, const float *cxs, const int *ebounds,
                                       int N, int K) {
    int e = blockIdx.z; int eb0 = ebounds[e], eb1 = ebounds[e+1];
    int m_base = eb0 + blockIdx.y * (8*TG);
    if (m_base >= eb1) return;
    int ntg = (eb1 - m_base + 7) / 8; if (ntg > TG) ntg = TG;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int gid = lane>>2, tid = lane&3;
    int n0 = blockIdx.x*64 + warp*16;
    int nb = K/256, row_bytes = nb*66, nsb = K/32;
    const uint8_t *We = W + (size_t)e*estride;
    __shared__ signed char sX[8*TG][32]; __shared__ float sXs[8*TG];
    __shared__ signed char sW[64][32]; __shared__ float sWs[64];
    float f[TG][4]; for(int g=0;g<TG;g++){f[g][0]=f[g][1]=f[g][2]=f[g][3]=0;}
    for (int sb=0; sb<nsb; sb++) {
        { int rl=lane>>1, half=lane&1, r=warp*16+rl, n=n0+rl;
            const uint8_t *bp=We+(size_t)n*row_bytes+(sb/8)*66;
            float d=h2f_dev(*(const uint16_t*)bp); const uint16_t *qs=(const uint16_t*)(bp+2); int ib=sb&7;
            uint32_t a0=(uint32_t)qs[4*ib]|((uint32_t)qs[4*ib+1]<<16);
            uint32_t a1=(uint32_t)qs[4*ib+2]|((uint32_t)qs[4*ib+3]<<16);
            if (half==0) sWs[r]=d*(0.5f+(float)(a1>>28))*0.25f;
            for(int l=half*2; l<half*2+2; l++){ uint8_t idx=(a0>>(8*l))&255; const uint8_t*g=(const uint8_t*)&c_grid[idx];
                uint8_t s=c_ksigns[(a1>>(7*l))&127];
                for(int j=0;j<8;j++) sW[r][l*8+j]=(signed char)((int)g[j]*((s&(1<<j))?-1:1)); } }
        for (int i=threadIdx.x;i<ntg*8*32;i+=blockDim.x){ int t=i/32,kk=i%32,m=m_base+t;
            sX[t][kk]=(m<eb1)?cxq8[(size_t)m*K+sb*32+kk]:0; }
        for (int t=threadIdx.x;t<ntg*8;t+=blockDim.x){ int m=m_base+t; sXs[t]=(m<eb1)?cxs[(size_t)m*nsb+sb]:0.0f; }
        __syncthreads();
        int wr=warp*16;
        int a0=pack4(&sW[wr+gid][tid*4]), a1=pack4(&sW[wr+gid+8][tid*4]);
        int a2=pack4(&sW[wr+gid][tid*4+16]), a3=pack4(&sW[wr+gid+8][tid*4+16]);
        float wr0=sWs[wr+gid], wr8=sWs[wr+gid+8];
        for (int g=0; g<ntg; g++) {
            int b0=pack4(&sX[g*8+gid][tid*4]), b1=pack4(&sX[g*8+gid][tid*4+16]);
            int c0=0,c1=0,c2=0,c3=0;
            asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                :"=r"(c0),"=r"(c1),"=r"(c2),"=r"(c3)
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(0),"r"(0),"r"(0),"r"(0));
            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];
            f[g][0]+=wr0*xc0*(float)c0; f[g][1]+=wr0*xc1*(float)c1;
            f[g][2]+=wr8*xc0*(float)c2; f[g][3]+=wr8*xc1*(float)c3;
        }
        __syncthreads();
    }
    int n_a=n0+gid, n_b=n0+gid+8;
    for (int g=0; g<ntg; g++) {
        int m_a=m_base+g*8+tid*2, m_b=m_a+1;
        if (m_a<eb1){ out[(size_t)m_a*N+n_a]=f[g][0]; out[(size_t)m_a*N+n_b]=f[g][2]; }
        if (m_b<eb1){ out[(size_t)m_b*N+n_a]=f[g][1]; out[(size_t)m_b*N+n_b]=f[g][3]; }
    }
}

/* grouped MMQ v5: v4 + flattened work-list grid. Instead of grid=(N/64, ceil(maxtok/32),
   n_experts) where ~80% of (y,z) blocks early-return and the few hot experts serialize a
   long tail, we launch grid=(N/64, n_work) where each blockIdx.y indexes a real (expert,
   group) pair from worklist[]. packed = (e<<16)|g. Eliminates empty blocks + balances the
   tail across SMs. Body identical to v4 once e/m_base are derived. */
__global__ void mmq_iq2xxs_grouped_v5(float *out, const uint8_t *W, unsigned long long estride,
                                       const signed char *cxq8, const float *cxs, const int *ebounds,
                                       const int *worklist, int N, int K) {
    int packed = worklist[blockIdx.y];
    int e = packed >> 16, g0 = packed & 0xffff;
    int eb0 = ebounds[e], eb1 = ebounds[e+1];
    int m_base = eb0 + g0 * (8*TG);
    int ntg = (eb1 - m_base + 7) / 8; if (ntg > TG) ntg = TG;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int gid = lane>>2, tid = lane&3;
    int n0 = blockIdx.x*64 + warp*16;
    int nb = K/256, row_bytes = nb*66, nsb = K/32;
    const uint8_t *We = W + (size_t)e*estride;
    __shared__ signed char sX[8*TG][32]; __shared__ float sXs[8*TG];
    __shared__ signed char sW[64][32]; __shared__ float sWs[64];
    float f[TG][4]; for(int g=0;g<TG;g++){f[g][0]=f[g][1]=f[g][2]=f[g][3]=0;}
    for (int sb=0; sb<nsb; sb++) {
        { int rl=lane>>1, half=lane&1, r=warp*16+rl, n=n0+rl;
            const uint8_t *bp=We+(size_t)n*row_bytes+(sb/8)*66;
            float d=h2f_dev(*(const uint16_t*)bp); const uint16_t *qs=(const uint16_t*)(bp+2); int ib=sb&7;
            uint32_t a0=(uint32_t)qs[4*ib]|((uint32_t)qs[4*ib+1]<<16);
            uint32_t a1=(uint32_t)qs[4*ib+2]|((uint32_t)qs[4*ib+3]<<16);
            if (half==0) sWs[r]=d*(0.5f+(float)(a1>>28))*0.25f;
            for(int l=half*2; l<half*2+2; l++){ uint8_t idx=(a0>>(8*l))&255; const uint8_t*g=(const uint8_t*)&c_grid[idx];
                uint8_t s=c_ksigns[(a1>>(7*l))&127];
                for(int j=0;j<8;j++) sW[r][l*8+j]=(signed char)((int)g[j]*((s&(1<<j))?-1:1)); } }
        for (int i=threadIdx.x;i<ntg*8*32;i+=blockDim.x){ int t=i/32,kk=i%32,m=m_base+t;
            sX[t][kk]=(m<eb1)?cxq8[(size_t)m*K+sb*32+kk]:0; }
        for (int t=threadIdx.x;t<ntg*8;t+=blockDim.x){ int m=m_base+t; sXs[t]=(m<eb1)?cxs[(size_t)m*nsb+sb]:0.0f; }
        __syncthreads();
        int wr=warp*16;
        int a0=pack4(&sW[wr+gid][tid*4]), a1=pack4(&sW[wr+gid+8][tid*4]);
        int a2=pack4(&sW[wr+gid][tid*4+16]), a3=pack4(&sW[wr+gid+8][tid*4+16]);
        float wr0=sWs[wr+gid], wr8=sWs[wr+gid+8];
        for (int g=0; g<ntg; g++) {
            int b0=pack4(&sX[g*8+gid][tid*4]), b1=pack4(&sX[g*8+gid][tid*4+16]);
            int c0=0,c1=0,c2=0,c3=0;
            asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                :"=r"(c0),"=r"(c1),"=r"(c2),"=r"(c3)
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(0),"r"(0),"r"(0),"r"(0));
            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];
            f[g][0]+=wr0*xc0*(float)c0; f[g][1]+=wr0*xc1*(float)c1;
            f[g][2]+=wr8*xc0*(float)c2; f[g][3]+=wr8*xc1*(float)c3;
        }
        __syncthreads();
    }
    int n_a=n0+gid, n_b=n0+gid+8;
    for (int g=0; g<ntg; g++) {
        int m_a=m_base+g*8+tid*2, m_b=m_a+1;
        if (m_a<eb1){ out[(size_t)m_a*N+n_a]=f[g][0]; out[(size_t)m_a*N+n_b]=f[g][2]; }
        if (m_b<eb1){ out[(size_t)m_b*N+n_a]=f[g][1]; out[(size_t)m_b*N+n_b]=f[g][3]; }
    }
}

/* grouped MMQ v6: v5 + codebook staged in shared memory. The IQ2_XXS decode gathers
   iq2xxs_grid[idx] (2KB, 256x u64) and ksigns[idx] (128B) with a per-lane random index,
   needed immediately for the sW write — in global/constant that is an exposed ~200-cycle
   latency repeated 64 rows x nsb sub-blocks. Stage both tables into shared once per block
   so every gather hits shared (~30 cyc). Decode is latency-bound on this gather, so this
   is the lever. +2176 B shared/block (fine at sm_120's 100KB/SM). */
__global__ void mmq_iq2xxs_grouped_v6(float *out, const uint8_t *W, unsigned long long estride,
                                       const signed char *cxq8, const float *cxs, const int *ebounds,
                                       const int *worklist, int N, int K) {
    __shared__ uint64_t sGrid[256]; __shared__ uint8_t sKs[128];
    for (int i=threadIdx.x;i<256;i+=blockDim.x) sGrid[i]=c_grid[i];
    for (int i=threadIdx.x;i<128;i+=blockDim.x) sKs[i]=c_ksigns[i];
    int packed = worklist[blockIdx.y];
    int e = packed >> 16, g0 = packed & 0xffff;
    int eb0 = ebounds[e], eb1 = ebounds[e+1];
    int m_base = eb0 + g0 * (8*TG);
    int ntg = (eb1 - m_base + 7) / 8; if (ntg > TG) ntg = TG;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int gid = lane>>2, tid = lane&3;
    int n0 = blockIdx.x*64 + warp*16;
    int nb = K/256, row_bytes = nb*66, nsb = K/32;
    const uint8_t *We = W + (size_t)e*estride;
    __shared__ signed char sX[8*TG][32]; __shared__ float sXs[8*TG];
    __shared__ signed char sW[64][32]; __shared__ float sWs[64];
    float f[TG][4]; for(int g=0;g<TG;g++){f[g][0]=f[g][1]=f[g][2]=f[g][3]=0;}
    __syncthreads();   /* codebook staged */
    for (int sb=0; sb<nsb; sb++) {
        { int rl=lane>>1, half=lane&1, r=warp*16+rl, n=n0+rl;
            const uint8_t *bp=We+(size_t)n*row_bytes+(sb/8)*66;
            float d=h2f_dev(*(const uint16_t*)bp); const uint16_t *qs=(const uint16_t*)(bp+2); int ib=sb&7;
            uint32_t a0=(uint32_t)qs[4*ib]|((uint32_t)qs[4*ib+1]<<16);
            uint32_t a1=(uint32_t)qs[4*ib+2]|((uint32_t)qs[4*ib+3]<<16);
            if (half==0) sWs[r]=d*(0.5f+(float)(a1>>28))*0.25f;
            for(int l=half*2; l<half*2+2; l++){ uint8_t idx=(a0>>(8*l))&255; const uint8_t*g=(const uint8_t*)&sGrid[idx];
                uint8_t s=sKs[(a1>>(7*l))&127];
                for(int j=0;j<8;j++) sW[r][l*8+j]=(signed char)((int)g[j]*((s&(1<<j))?-1:1)); } }
        for (int i=threadIdx.x;i<ntg*8*32;i+=blockDim.x){ int t=i/32,kk=i%32,m=m_base+t;
            sX[t][kk]=(m<eb1)?cxq8[(size_t)m*K+sb*32+kk]:0; }
        for (int t=threadIdx.x;t<ntg*8;t+=blockDim.x){ int m=m_base+t; sXs[t]=(m<eb1)?cxs[(size_t)m*nsb+sb]:0.0f; }
        __syncthreads();
        int wr=warp*16;
        int a0=pack4(&sW[wr+gid][tid*4]), a1=pack4(&sW[wr+gid+8][tid*4]);
        int a2=pack4(&sW[wr+gid][tid*4+16]), a3=pack4(&sW[wr+gid+8][tid*4+16]);
        float wr0=sWs[wr+gid], wr8=sWs[wr+gid+8];
        for (int g=0; g<ntg; g++) {
            int b0=pack4(&sX[g*8+gid][tid*4]), b1=pack4(&sX[g*8+gid][tid*4+16]);
            int c0=0,c1=0,c2=0,c3=0;
            asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                :"=r"(c0),"=r"(c1),"=r"(c2),"=r"(c3)
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(0),"r"(0),"r"(0),"r"(0));
            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];
            f[g][0]+=wr0*xc0*(float)c0; f[g][1]+=wr0*xc1*(float)c1;
            f[g][2]+=wr8*xc0*(float)c2; f[g][3]+=wr8*xc1*(float)c3;
        }
        __syncthreads();
    }
    int n_a=n0+gid, n_b=n0+gid+8;
    for (int g=0; g<ntg; g++) {
        int m_a=m_base+g*8+tid*2, m_b=m_a+1;
        if (m_a<eb1){ out[(size_t)m_a*N+n_a]=f[g][0]; out[(size_t)m_a*N+n_b]=f[g][2]; }
        if (m_b<eb1){ out[(size_t)m_b*N+n_a]=f[g][1]; out[(size_t)m_b*N+n_b]=f[g][3]; }
    }
}

/* grouped MMQ v7: v6 + COALESCED weight staging. The decode's global weight read is the
   real bottleneck: each lane reads bp=We+n*row_bytes+... so a warp's 16 rows are
   row_bytes (=528B for K=2048) apart -> 16 cache lines for ~160 useful bytes (~9-12x
   amplification, kernel sits ~9x above its DRAM floor). But the 64 rows a block owns are
   CONTIGUOUS in the expert matrix, so stage that 64*row_bytes tile into shared with a
   fully-coalesced cooperative load, then decode from shared. Dynamic shared = 64*row_bytes. */
__global__ void mmq_iq2xxs_grouped_v7(float *out, const uint8_t *W, unsigned long long estride,
                                       const signed char *cxq8, const float *cxs, const int *ebounds,
                                       const int *worklist, int N, int K) {
    extern __shared__ unsigned char sWt[];   /* [64*row_bytes] raw weight tile */
    __shared__ uint64_t sGrid[256]; __shared__ uint8_t sKs[128];
    for (int i=threadIdx.x;i<256;i+=blockDim.x) sGrid[i]=c_grid[i];
    for (int i=threadIdx.x;i<128;i+=blockDim.x) sKs[i]=c_ksigns[i];
    int packed = worklist[blockIdx.y];
    int e = packed >> 16, g0 = packed & 0xffff;
    int eb0 = ebounds[e], eb1 = ebounds[e+1];
    int m_base = eb0 + g0 * (8*TG);
    int ntg = (eb1 - m_base + 7) / 8; if (ntg > TG) ntg = TG;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int gid = lane>>2, tid = lane&3;
    int n0 = blockIdx.x*64 + warp*16;
    int nb = K/256, row_bytes = nb*66, nsb = K/32;
    int tile_n0 = blockIdx.x*64;                 /* first row of the 64-row tile */
    const uint8_t *We = W + (size_t)e*estride;
    /* coalesced cooperative load of the contiguous 64-row weight tile */
    int tile_bytes = 64*row_bytes;
    const uint8_t *tile_src = We + (size_t)tile_n0*row_bytes;
    for (int i=threadIdx.x;i<tile_bytes;i+=blockDim.x) sWt[i]=tile_src[i];
    __shared__ signed char sX[8*TG][32]; __shared__ float sXs[8*TG];
    __shared__ signed char sW[64][32]; __shared__ float sWs[64];
    float f[TG][4]; for(int g=0;g<TG;g++){f[g][0]=f[g][1]=f[g][2]=f[g][3]=0;}
    __syncthreads();
    for (int sb=0; sb<nsb; sb++) {
        { int rl=lane>>1, half=lane&1, r=warp*16+rl, n=n0+rl;
            const uint8_t *bp=sWt+(size_t)(n-tile_n0)*row_bytes+(sb/8)*66;
            float d=h2f_dev(*(const uint16_t*)bp); const uint16_t *qs=(const uint16_t*)(bp+2); int ib=sb&7;
            uint32_t a0=(uint32_t)qs[4*ib]|((uint32_t)qs[4*ib+1]<<16);
            uint32_t a1=(uint32_t)qs[4*ib+2]|((uint32_t)qs[4*ib+3]<<16);
            if (half==0) sWs[r]=d*(0.5f+(float)(a1>>28))*0.25f;
            for(int l=half*2; l<half*2+2; l++){ uint8_t idx=(a0>>(8*l))&255; const uint8_t*g=(const uint8_t*)&sGrid[idx];
                uint8_t s=sKs[(a1>>(7*l))&127];
                for(int j=0;j<8;j++) sW[r][l*8+j]=(signed char)((int)g[j]*((s&(1<<j))?-1:1)); } }
        for (int i=threadIdx.x;i<ntg*8*32;i+=blockDim.x){ int t=i/32,kk=i%32,m=m_base+t;
            sX[t][kk]=(m<eb1)?cxq8[(size_t)m*K+sb*32+kk]:0; }
        for (int t=threadIdx.x;t<ntg*8;t+=blockDim.x){ int m=m_base+t; sXs[t]=(m<eb1)?cxs[(size_t)m*nsb+sb]:0.0f; }
        __syncthreads();
        int wr=warp*16;
        int a0=pack4(&sW[wr+gid][tid*4]), a1=pack4(&sW[wr+gid+8][tid*4]);
        int a2=pack4(&sW[wr+gid][tid*4+16]), a3=pack4(&sW[wr+gid+8][tid*4+16]);
        float wr0=sWs[wr+gid], wr8=sWs[wr+gid+8];
        for (int g=0; g<ntg; g++) {
            int b0=pack4(&sX[g*8+gid][tid*4]), b1=pack4(&sX[g*8+gid][tid*4+16]);
            int c0=0,c1=0,c2=0,c3=0;
            asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                :"=r"(c0),"=r"(c1),"=r"(c2),"=r"(c3)
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(0),"r"(0),"r"(0),"r"(0));
            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];
            f[g][0]+=wr0*xc0*(float)c0; f[g][1]+=wr0*xc1*(float)c1;
            f[g][2]+=wr8*xc0*(float)c2; f[g][3]+=wr8*xc1*(float)c3;
        }
        __syncthreads();
    }
    int n_a=n0+gid, n_b=n0+gid+8;
    for (int g=0; g<ntg; g++) {
        int m_a=m_base+g*8+tid*2, m_b=m_a+1;
        if (m_a<eb1){ out[(size_t)m_a*N+n_a]=f[g][0]; out[(size_t)m_a*N+n_b]=f[g][2]; }
        if (m_b<eb1){ out[(size_t)m_b*N+n_a]=f[g][1]; out[(size_t)m_b*N+n_b]=f[g][3]; }
    }
}

/* grouped MMQ v8: v6 + BRANCHLESS vectorized sign application. The decode's per-byte
   sign-multiply `(int)g[j]*((s&(1<<j))?-1:1)` over j<8 (~4-5 inst/elem, 8 byte stores)
   is the decode-ALU bottleneck (kernel sits ~3% of int8-TC peak). Replace with a
   two's-complement negate via __vsub4: (g^0xFF)-0xFF = -g, (g^0)-0 = g, per byte, using
   a precomputed 8-byte 0x00/0xFF sign mask staged in shared. ~8 inst + 2 word stores for
   the same 8 bytes. Bit-exact. */
__global__ void mmq_iq2xxs_grouped_v8(float *out, const uint8_t *W, unsigned long long estride,
                                       const signed char *cxq8, const float *cxs, const int *ebounds,
                                       const int *worklist, int N, int K) {
    __shared__ uint64_t sGrid[256]; __shared__ uint64_t sSignMask[128];
    for (int i=threadIdx.x;i<256;i+=blockDim.x) sGrid[i]=c_grid[i];
    for (int i=threadIdx.x;i<128;i+=blockDim.x) sSignMask[i]=c_signmask[i];
    int packed = worklist[blockIdx.y];
    int e = packed >> 16, g0 = packed & 0xffff;
    int eb0 = ebounds[e], eb1 = ebounds[e+1];
    int m_base = eb0 + g0 * (8*TG);
    int ntg = (eb1 - m_base + 7) / 8; if (ntg > TG) ntg = TG;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int gid = lane>>2, tid = lane&3;
    int n0 = blockIdx.x*64 + warp*16;
    int nb = K/256, row_bytes = nb*66, nsb = K/32;
    const uint8_t *We = W + (size_t)e*estride;
    __shared__ signed char sX[8*TG][32]; __shared__ float sXs[8*TG];
    __shared__ signed char sW[64][32]; __shared__ float sWs[64];
    float f[TG][4]; for(int g=0;g<TG;g++){f[g][0]=f[g][1]=f[g][2]=f[g][3]=0;}
    __syncthreads();
    for (int sb=0; sb<nsb; sb++) {
        { int rl=lane>>1, half=lane&1, r=warp*16+rl, n=n0+rl;
            const uint8_t *bp=We+(size_t)n*row_bytes+(sb/8)*66;
            float d=h2f_dev(*(const uint16_t*)bp); const uint16_t *qs=(const uint16_t*)(bp+2); int ib=sb&7;
            uint32_t a0=(uint32_t)qs[4*ib]|((uint32_t)qs[4*ib+1]<<16);
            uint32_t a1=(uint32_t)qs[4*ib+2]|((uint32_t)qs[4*ib+3]<<16);
            if (half==0) sWs[r]=d*(0.5f+(float)(a1>>28))*0.25f;
            for(int l=half*2; l<half*2+2; l++){ uint8_t idx=(a0>>(8*l))&255;
                uint64_t gv=sGrid[idx], m=sSignMask[(a1>>(7*l))&127];
                uint32_t glo=(uint32_t)gv, ghi=(uint32_t)(gv>>32);
                uint32_t mlo=(uint32_t)m,  mhi=(uint32_t)(m>>32);
                *(uint32_t*)&sW[r][l*8]   = __vsub4(glo^mlo, mlo);
                *(uint32_t*)&sW[r][l*8+4] = __vsub4(ghi^mhi, mhi); } }
        for (int i=threadIdx.x;i<ntg*8*32;i+=blockDim.x){ int t=i/32,kk=i%32,m=m_base+t;
            sX[t][kk]=(m<eb1)?cxq8[(size_t)m*K+sb*32+kk]:0; }
        for (int t=threadIdx.x;t<ntg*8;t+=blockDim.x){ int m=m_base+t; sXs[t]=(m<eb1)?cxs[(size_t)m*nsb+sb]:0.0f; }
        __syncthreads();
        int wr=warp*16;
        int a0=pack4(&sW[wr+gid][tid*4]), a1=pack4(&sW[wr+gid+8][tid*4]);
        int a2=pack4(&sW[wr+gid][tid*4+16]), a3=pack4(&sW[wr+gid+8][tid*4+16]);
        float wr0=sWs[wr+gid], wr8=sWs[wr+gid+8];
        for (int g=0; g<ntg; g++) {
            int b0=pack4(&sX[g*8+gid][tid*4]), b1=pack4(&sX[g*8+gid][tid*4+16]);
            int c0=0,c1=0,c2=0,c3=0;
            asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                :"=r"(c0),"=r"(c1),"=r"(c2),"=r"(c3)
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(0),"r"(0),"r"(0),"r"(0));
            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];
            f[g][0]+=wr0*xc0*(float)c0; f[g][1]+=wr0*xc1*(float)c1;
            f[g][2]+=wr8*xc0*(float)c2; f[g][3]+=wr8*xc1*(float)c3;
        }
        __syncthreads();
    }
    int n_a=n0+gid, n_b=n0+gid+8;
    for (int g=0; g<ntg; g++) {
        int m_a=m_base+g*8+tid*2, m_b=m_a+1;
        if (m_a<eb1){ out[(size_t)m_a*N+n_a]=f[g][0]; out[(size_t)m_a*N+n_b]=f[g][2]; }
        if (m_b<eb1){ out[(size_t)m_b*N+n_a]=f[g][1]; out[(size_t)m_b*N+n_b]=f[g][3]; }
    }
}

/* grouped MMQ v10: v8 + direct aligned int loads for the MMA operands. sW is stored via
   *(uint32_t*) (4-aligned), so pack4's per-byte reassembly (4 loads + shifts/ors per operand,
   6 operands per MMA) is wasted — read the int directly. Pure hot-loop instruction cut. */
__global__ void mmq_iq2xxs_grouped_v10(float *out, const uint8_t *W, unsigned long long estride,
                                       const signed char *cxq8, const float *cxs, const int *ebounds,
                                       const int *worklist, int N, int K) {
    __shared__ uint64_t sGrid[256]; __shared__ uint64_t sSignMask[128];
    for (int i=threadIdx.x;i<256;i+=blockDim.x) sGrid[i]=c_grid[i];
    for (int i=threadIdx.x;i<128;i+=blockDim.x) sSignMask[i]=c_signmask[i];
    int packed = worklist[blockIdx.y];
    int e = packed >> 16, g0 = packed & 0xffff;
    int eb0 = ebounds[e], eb1 = ebounds[e+1];
    int m_base = eb0 + g0 * (8*TG);
    int ntg = (eb1 - m_base + 7) / 8; if (ntg > TG) ntg = TG;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int gid = lane>>2, tid = lane&3;
    int n0 = blockIdx.x*64 + warp*16;
    int nb = K/256, row_bytes = nb*66, nsb = K/32;
    const uint8_t *We = W + (size_t)e*estride;
    __shared__ signed char sX[8*TG][32]; __shared__ float sXs[8*TG];
    __shared__ signed char sW[64][32]; __shared__ float sWs[64];
    float f[TG][4]; for(int g=0;g<TG;g++){f[g][0]=f[g][1]=f[g][2]=f[g][3]=0;}
    __syncthreads();
    for (int sb=0; sb<nsb; sb++) {
        { int rl=lane>>1, half=lane&1, r=warp*16+rl, n=n0+rl;
            const uint8_t *bp=We+(size_t)n*row_bytes+(sb/8)*66;
            float d=h2f_dev(*(const uint16_t*)bp); const uint16_t *qs=(const uint16_t*)(bp+2); int ib=sb&7;
            uint32_t a0=(uint32_t)qs[4*ib]|((uint32_t)qs[4*ib+1]<<16);
            uint32_t a1=(uint32_t)qs[4*ib+2]|((uint32_t)qs[4*ib+3]<<16);
            if (half==0) sWs[r]=d*(0.5f+(float)(a1>>28))*0.25f;
            for(int l=half*2; l<half*2+2; l++){ uint8_t idx=(a0>>(8*l))&255;
                uint64_t gv=sGrid[idx], m=sSignMask[(a1>>(7*l))&127];
                uint32_t glo=(uint32_t)gv, ghi=(uint32_t)(gv>>32);
                uint32_t mlo=(uint32_t)m,  mhi=(uint32_t)(m>>32);
                *(uint32_t*)&sW[r][l*8]   = __vsub4(glo^mlo, mlo);
                *(uint32_t*)&sW[r][l*8+4] = __vsub4(ghi^mhi, mhi); } }
        for (int i=threadIdx.x;i<ntg*8*32;i+=blockDim.x){ int t=i/32,kk=i%32,m=m_base+t;
            sX[t][kk]=(m<eb1)?cxq8[(size_t)m*K+sb*32+kk]:0; }
        for (int t=threadIdx.x;t<ntg*8;t+=blockDim.x){ int m=m_base+t; sXs[t]=(m<eb1)?cxs[(size_t)m*nsb+sb]:0.0f; }
        __syncthreads();
        int wr=warp*16;
        int a0=pack4a(&sW[wr+gid][tid*4]), a1=pack4a(&sW[wr+gid+8][tid*4]);
        int a2=pack4a(&sW[wr+gid][tid*4+16]), a3=pack4a(&sW[wr+gid+8][tid*4+16]);
        float wr0=sWs[wr+gid], wr8=sWs[wr+gid+8];
        for (int g=0; g<ntg; g++) {
            int b0=pack4a(&sX[g*8+gid][tid*4]), b1=pack4a(&sX[g*8+gid][tid*4+16]);
            int c0=0,c1=0,c2=0,c3=0;
            asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                :"=r"(c0),"=r"(c1),"=r"(c2),"=r"(c3)
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(0),"r"(0),"r"(0),"r"(0));
            float xc0=sXs[g*8+tid*2], xc1=sXs[g*8+tid*2+1];
            f[g][0]+=wr0*xc0*(float)c0; f[g][1]+=wr0*xc1*(float)c1;
            f[g][2]+=wr8*xc0*(float)c2; f[g][3]+=wr8*xc1*(float)c3;
        }
        __syncthreads();
    }
    int n_a=n0+gid, n_b=n0+gid+8;
    for (int g=0; g<ntg; g++) {
        int m_a=m_base+g*8+tid*2, m_b=m_a+1;
        if (m_a<eb1){ out[(size_t)m_a*N+n_a]=f[g][0]; out[(size_t)m_a*N+n_b]=f[g][2]; }
        if (m_b<eb1){ out[(size_t)m_b*N+n_a]=f[g][1]; out[(size_t)m_b*N+n_b]=f[g][3]; }
    }
}

/* grouped MMQ v9: v8 + double-buffered software pipeline. Now that decode is cheap
   (v8 branchless), the per-sub-block chain decode->load->sync->MMA->sync is latency-bound:
   the MMA stalls waiting for the next sub-block's weight decode + activation load. Ping-pong
   sW/sX between two buffers so sub-block sb+1's decode+load is issued BEFORE sb's MMA and
   overlaps it (independent buffers), cutting to one __syncthreads per sub-block. Bit-exact. */
__device__ __forceinline__ void v9_decode_load(
        int sb, int nsb, int ntg, int m_base, int eb1, int K, int row_bytes, int nsb_,
        const uint8_t *We, const signed char *cxq8, const float *cxs,
        const uint64_t *sGrid, const uint64_t *sSignMask,
        signed char sW[64][32], float *sWs, signed char sX[8*TG][32], float *sXs,
        int lane, int warp, int n0) {
    int rl=lane>>1, half=lane&1, r=warp*16+rl, n=n0+rl;
    const uint8_t *bp=We+(size_t)n*row_bytes+(sb/8)*66;
    float d=h2f_dev(*(const uint16_t*)bp); const uint16_t *qs=(const uint16_t*)(bp+2); int ib=sb&7;
    uint32_t a0=(uint32_t)qs[4*ib]|((uint32_t)qs[4*ib+1]<<16);
    uint32_t a1=(uint32_t)qs[4*ib+2]|((uint32_t)qs[4*ib+3]<<16);
    if (half==0) sWs[r]=d*(0.5f+(float)(a1>>28))*0.25f;
    for(int l=half*2; l<half*2+2; l++){ uint8_t idx=(a0>>(8*l))&255;
        uint64_t gv=sGrid[idx], m=sSignMask[(a1>>(7*l))&127];
        uint32_t glo=(uint32_t)gv, ghi=(uint32_t)(gv>>32);
        uint32_t mlo=(uint32_t)m,  mhi=(uint32_t)(m>>32);
        *(uint32_t*)&sW[r][l*8]   = __vsub4(glo^mlo, mlo);
        *(uint32_t*)&sW[r][l*8+4] = __vsub4(ghi^mhi, mhi); }
    int bd=128;
    for (int i=threadIdx.x;i<ntg*8*32;i+=bd){ int t=i/32,kk=i%32,mm=m_base+t;
        sX[t][kk]=(mm<eb1)?cxq8[(size_t)mm*K+sb*32+kk]:0; }
    for (int t=threadIdx.x;t<ntg*8;t+=bd){ int mm=m_base+t; sXs[t]=(mm<eb1)?cxs[(size_t)mm*nsb_+sb]:0.0f; }
}
__global__ void mmq_iq2xxs_grouped_v9(float *out, const uint8_t *W, unsigned long long estride,
                                       const signed char *cxq8, const float *cxs, const int *ebounds,
                                       const int *worklist, int N, int K) {
    __shared__ uint64_t sGrid[256]; __shared__ uint64_t sSignMask[128];
    for (int i=threadIdx.x;i<256;i+=blockDim.x) sGrid[i]=c_grid[i];
    for (int i=threadIdx.x;i<128;i+=blockDim.x) sSignMask[i]=c_signmask[i];
    int packed = worklist[blockIdx.y];
    int e = packed >> 16, g0 = packed & 0xffff;
    int eb0 = ebounds[e], eb1 = ebounds[e+1];
    int m_base = eb0 + g0 * (8*TG);
    int ntg = (eb1 - m_base + 7) / 8; if (ntg > TG) ntg = TG;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int gid = lane>>2, tid = lane&3;
    int n0 = blockIdx.x*64 + warp*16;
    int nb = K/256, row_bytes = nb*66, nsb = K/32;
    const uint8_t *We = W + (size_t)e*estride;
    __shared__ signed char sX[2][8*TG][32]; __shared__ float sXs[2][8*TG];
    __shared__ signed char sW[2][64][32];   __shared__ float sWs[2][64];
    float f[TG][4]; for(int g=0;g<TG;g++){f[g][0]=f[g][1]=f[g][2]=f[g][3]=0;}
    __syncthreads();
    v9_decode_load(0,nsb,ntg,m_base,eb1,K,row_bytes,nsb,We,cxq8,cxs,sGrid,sSignMask,
                   sW[0],sWs[0],sX[0],sXs[0],lane,warp,n0);
    for (int sb=0; sb<nsb; sb++) {
        int cur=sb&1, nxt=(sb+1)&1;
        __syncthreads();
        if (sb+1<nsb) v9_decode_load(sb+1,nsb,ntg,m_base,eb1,K,row_bytes,nsb,We,cxq8,cxs,sGrid,sSignMask,
                                     sW[nxt],sWs[nxt],sX[nxt],sXs[nxt],lane,warp,n0);
        int wr=warp*16;
        int a0=pack4(&sW[cur][wr+gid][tid*4]), a1=pack4(&sW[cur][wr+gid+8][tid*4]);
        int a2=pack4(&sW[cur][wr+gid][tid*4+16]), a3=pack4(&sW[cur][wr+gid+8][tid*4+16]);
        float wr0=sWs[cur][wr+gid], wr8=sWs[cur][wr+gid+8];
        for (int g=0; g<ntg; g++) {
            int b0=pack4(&sX[cur][g*8+gid][tid*4]), b1=pack4(&sX[cur][g*8+gid][tid*4+16]);
            int c0=0,c1=0,c2=0,c3=0;
            asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                :"=r"(c0),"=r"(c1),"=r"(c2),"=r"(c3)
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(0),"r"(0),"r"(0),"r"(0));
            float xc0=sXs[cur][g*8+tid*2], xc1=sXs[cur][g*8+tid*2+1];
            f[g][0]+=wr0*xc0*(float)c0; f[g][1]+=wr0*xc1*(float)c1;
            f[g][2]+=wr8*xc0*(float)c2; f[g][3]+=wr8*xc1*(float)c3;
        }
    }
    int n_a=n0+gid, n_b=n0+gid+8;
    for (int g=0; g<ntg; g++) {
        int m_a=m_base+g*8+tid*2, m_b=m_a+1;
        if (m_a<eb1){ out[(size_t)m_a*N+n_a]=f[g][0]; out[(size_t)m_a*N+n_b]=f[g][2]; }
        if (m_b<eb1){ out[(size_t)m_b*N+n_a]=f[g][1]; out[(size_t)m_b*N+n_b]=f[g][3]; }
    }
}

/* ---- weighted scatter: final[token] += wgt[c] * out_compact[c] ---- */
__global__ void scatter_weighted(float *final_out, const float *out_compact, const int *ids_token,
                                  const float *cw, int total_rows, int N) {
    int c = blockIdx.x; if (c>=total_rows) return;
    int tok = ids_token[c]; float w = cw[c];
    const float *s = out_compact + (size_t)c*N; float *d = final_out + (size_t)tok*N;
    for (int n=threadIdx.x;n<N;n+=blockDim.x) atomicAdd(&d[n], w*s[n]);
}

static double rel_l2(const float*a,const float*b,int n){ double num=0,den=0;
  for(int i=0;i<n;i++){ double d=(double)a[i]-b[i]; num+=d*d; den+=(double)b[i]*b[i]; } return den>0?sqrt(num/den):sqrt(num); }

int main(int argc,char**argv){
    int n_experts = argc>1?atoi(argv[1]):8;
    int n_tokens  = argc>2?atoi(argv[2]):64;
    int n_used    = argc>3?atoi(argv[3]):2;
    int N = argc>4?atoi(argv[4]):512;     /* expert_ff */
    int K = argc>5?atoi(argv[5]):2048;    /* n_embd */
    printf("MoE mul_mat_id test: experts=%d tokens=%d top-%d  N=%d K=%d\n", n_experts,n_tokens,n_used,N,K);
    if (N%(16*WN)||K%QK_K){ fprintf(stderr,"need N%%%d==0, K%%256==0\n",16*WN); return 1; }
    srand(7);
    int nb=K/QK_K, row_bytes=nb*IQ2XXS_BYTES; size_t estride=(size_t)N*row_bytes;
    /* expert weights */
    uint8_t *W=(uint8_t*)malloc((size_t)n_experts*estride);
    for(size_t i=0;i<(size_t)n_experts*estride;i++) W[i]=rand()&0xff;
    for(int e=0;e<n_experts;e++)for(int n=0;n<N;n++)for(int b=0;b<nb;b++){
        float d=0.01f+(rand()/(float)RAND_MAX)*0.05f;
        *(uint16_t*)(W+e*estride+(size_t)n*row_bytes+b*IQ2XXS_BYTES)=f2h(d); }
    /* activations + routing */
    float *X=(float*)malloc((size_t)n_tokens*K*sizeof(float));
    for(int i=0;i<n_tokens*K;i++) X[i]=((rand()/(float)RAND_MAX)-0.5f)*2.0f;
    int *ids=(int*)malloc((size_t)n_tokens*n_used*sizeof(int));
    float *wgt=(float*)malloc((size_t)n_tokens*n_used*sizeof(float));
    for(int t=0;t<n_tokens;t++)for(int u=0;u<n_used;u++){ ids[t*n_used+u]=rand()%n_experts; wgt[t*n_used+u]=rand()/(float)RAND_MAX; }

    /* ---- compaction (host, mirrors mm_ids_helper) ---- */
    int total=n_tokens*n_used;
    int *count=(int*)calloc(n_experts,sizeof(int));
    for(int i=0;i<total;i++) count[ids[i]]++;
    int *ebounds=(int*)malloc((n_experts+1)*sizeof(int)); ebounds[0]=0;
    for(int e=0;e<n_experts;e++) ebounds[e+1]=ebounds[e]+count[e];
    int *cursor=(int*)malloc(n_experts*sizeof(int)); for(int e=0;e<n_experts;e++) cursor[e]=ebounds[e];
    int *ids_token=(int*)malloc(total*sizeof(int)); float *cw=(float*)malloc(total*sizeof(float));
    for(int t=0;t<n_tokens;t++)for(int u=0;u<n_used;u++){ int e=ids[t*n_used+u]; int c=cursor[e]++;
        ids_token[c]=t; cw[c]=wgt[t*n_used+u]; }

    /* ---- oracle ---- */
    float *ref=(float*)calloc((size_t)n_tokens*N,sizeof(float));
    float *wf=(float*)malloc((size_t)K*sizeof(float));
    for(int t=0;t<n_tokens;t++)for(int u=0;u<n_used;u++){ int e=ids[t*n_used+u]; float w=wgt[t*n_used+u];
        for(int n=0;n<N;n++){
            for(int b=0;b<nb;b++) decode_block_h(W+e*estride+(size_t)n*row_bytes+b*IQ2XXS_BYTES, wf+b*QK_K);
            double s=0; for(int k=0;k<K;k++) s+=(double)wf[k]*X[(size_t)t*K+k];
            ref[(size_t)t*N+n]+=w*(float)s; } }
    free(wf);

    /* ---- device pipeline ---- */
    CUDA_CHECK(cudaMemcpyToSymbol(c_grid,iq2xxs_grid,sizeof(iq2xxs_grid)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_ksigns,ksigns_iq2xs,sizeof(ksigns_iq2xs)));
    uint64_t signmask[128];
    for(int i=0;i<128;i++){ uint8_t s=ksigns_iq2xs[i]; uint64_t m=0;
        for(int j=0;j<8;j++) if(s&(1<<j)) m|=(uint64_t)0xFF<<(8*j); signmask[i]=m; }
    CUDA_CHECK(cudaMemcpyToSymbol(c_signmask,signmask,sizeof(signmask)));
    uint8_t *dW; float *dX,*dCxs,*dOutC,*dFinal,*dCw; int8_t *dCxq; int *dIdsTok,*dEbounds;
    CUDA_CHECK(cudaMalloc(&dW,(size_t)n_experts*estride));
    CUDA_CHECK(cudaMalloc(&dX,(size_t)n_tokens*K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dCxq,(size_t)total*K));
    CUDA_CHECK(cudaMalloc(&dCxs,(size_t)total*(K/CK)*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOutC,(size_t)total*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dFinal,(size_t)n_tokens*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dCw,total*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dIdsTok,total*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dEbounds,(n_experts+1)*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dW,W,(size_t)n_experts*estride,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dX,X,(size_t)n_tokens*K*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dCw,cw,total*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dIdsTok,ids_token,total*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dEbounds,ebounds,(n_experts+1)*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dFinal,0,(size_t)n_tokens*N*sizeof(float)));

    int max_tok=0; for(int e=0;e<n_experts;e++) if(count[e]>max_tok) max_tok=count[e];
    dim3 gq(K/CK,total); gather_quant_q8_1<<<gq,32>>>(dX,dIdsTok,dCxq,dCxs,total,K);
    CUDA_CHECK(cudaGetLastError());
    dim3 gg(N/(16*WN),(max_tok+7)/8,n_experts);
    mmq_iq2xxs_grouped<<<gg,32*WN>>>(dOutC,dW,estride,dCxq,dCxs,dEbounds,N,K);
    CUDA_CHECK(cudaGetLastError());
    scatter_weighted<<<total,256>>>(dFinal,dOutC,dIdsTok,dCw,total,N);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());

    float *out=(float*)malloc((size_t)n_tokens*N*sizeof(float));
    CUDA_CHECK(cudaMemcpy(out,dFinal,(size_t)n_tokens*N*sizeof(float),cudaMemcpyDeviceToHost));
    double e=rel_l2(out,ref,n_tokens*N);
    printf("P3 grouped MMQ + scatter vs CPU oracle: rel_L2 = %.6f\n",e);
    printf("  ref[0..3]=%.4f %.4f %.4f %.4f\n  gpu[0..3]=%.4f %.4f %.4f %.4f\n",
           ref[0],ref[1],ref[2],ref[3],out[0],out[1],out[2],out[3]);
    int ok=e<0.05;

    /* ---- validate + time grouped v1 vs v3 (decode-amortized), gate dispatch ---- */
    float *dOutC3; CUDA_CHECK(cudaMalloc(&dOutC3,(size_t)total*N*sizeof(float)));
    CUDA_CHECK(cudaMemset(dOutC3,0,(size_t)total*N*sizeof(float)));
    dim3 gg3(N/64,(max_tok+8*TG-1)/(8*TG),n_experts);
    mmq_iq2xxs_grouped_v3<<<gg3,128>>>(dOutC3,dW,estride,dCxq,dCxs,dEbounds,N,K);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    /* compare v3 vs v1 out_compact directly */
    float *oc1=(float*)malloc((size_t)total*N*sizeof(float)), *oc3=(float*)malloc((size_t)total*N*sizeof(float));
    CUDA_CHECK(cudaMemcpy(oc1,dOutC,(size_t)total*N*sizeof(float),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(oc3,dOutC3,(size_t)total*N*sizeof(float),cudaMemcpyDeviceToHost));
    double ev3=rel_l2(oc3,oc1,total*N);
    printf("P3 grouped_v3 vs grouped_v1: rel_L2 = %.6f  (expect ~0)\n", ev3);
    ok = ok && (ev3<1e-4);
    {
        int it=200; cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        dim3 g1(N/64,(max_tok+7)/8,n_experts);
        cudaEventRecord(t0); for(int i=0;i<it;i++) mmq_iq2xxs_grouped<<<g1,128>>>(dOutC,dW,estride,dCxq,dCxs,dEbounds,N,K);
        cudaEventRecord(t1); cudaEventSynchronize(t1); float m1; cudaEventElapsedTime(&m1,t0,t1); m1/=it;
        cudaEventRecord(t0); for(int i=0;i<it;i++) mmq_iq2xxs_grouped_v3<<<gg3,128>>>(dOutC3,dW,estride,dCxq,dCxs,dEbounds,N,K);
        cudaEventRecord(t1); cudaEventSynchronize(t1); float m3; cudaEventElapsedTime(&m3,t0,t1); m3/=it;
        /* v4: 32-lane decode */
        float *dOutC4; CUDA_CHECK(cudaMalloc(&dOutC4,(size_t)total*N*sizeof(float)));
        CUDA_CHECK(cudaMemset(dOutC4,0,(size_t)total*N*sizeof(float)));
        mmq_iq2xxs_grouped_v4<<<gg3,128>>>(dOutC4,dW,estride,dCxq,dCxs,dEbounds,N,K);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        float *oc4=(float*)malloc((size_t)total*N*sizeof(float));
        CUDA_CHECK(cudaMemcpy(oc4,dOutC4,(size_t)total*N*sizeof(float),cudaMemcpyDeviceToHost));
        double ev4=rel_l2(oc4,oc1,total*N);
        cudaEventRecord(t0); for(int i=0;i<it;i++) mmq_iq2xxs_grouped_v4<<<gg3,128>>>(dOutC4,dW,estride,dCxq,dCxs,dEbounds,N,K);
        cudaEventRecord(t1); cudaEventSynchronize(t1); float m4; cudaEventElapsedTime(&m4,t0,t1); m4/=it;
        printf("grouped gate dispatch: v1=%.3f  v3=%.3f  v4=%.3f ms  (v3 %.2fx, v4 %.2fx vs v1; v4 vs v3 %.2fx)\n",
               m1, m3, m4, m1/m3, m1/m4, m3/m4);
        printf("v4 vs v1 out_compact: rel_L2 = %.6f  (expect ~0)\n", ev4);
        ok = ok && (ev4<1e-4);

        /* ---- v5: flattened work-list grid ---- */
        int gy = (max_tok+8*TG-1)/(8*TG);
        int *worklist=(int*)malloc((size_t)n_experts*gy*sizeof(int)); int n_work=0;
        for(int ee=0;ee<n_experts;ee++){ int c=count[ee]; int ng=(c+8*TG-1)/(8*TG);
            for(int g=0;g<ng;g++) worklist[n_work++]=(ee<<16)|g; }
        int *dWork; CUDA_CHECK(cudaMalloc(&dWork,(size_t)n_work*sizeof(int)));
        CUDA_CHECK(cudaMemcpy(dWork,worklist,(size_t)n_work*sizeof(int),cudaMemcpyHostToDevice));
        float *dOutC5; CUDA_CHECK(cudaMalloc(&dOutC5,(size_t)total*N*sizeof(float)));
        CUDA_CHECK(cudaMemset(dOutC5,0,(size_t)total*N*sizeof(float)));
        dim3 gg5(N/64,n_work,1);
        mmq_iq2xxs_grouped_v5<<<gg5,128>>>(dOutC5,dW,estride,dCxq,dCxs,dEbounds,dWork,N,K);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        float *oc5=(float*)malloc((size_t)total*N*sizeof(float));
        CUDA_CHECK(cudaMemcpy(oc5,dOutC5,(size_t)total*N*sizeof(float),cudaMemcpyDeviceToHost));
        double ev5=rel_l2(oc5,oc1,total*N);
        cudaEventRecord(t0); for(int i=0;i<it;i++) mmq_iq2xxs_grouped_v5<<<gg5,128>>>(dOutC5,dW,estride,dCxq,dCxs,dEbounds,dWork,N,K);
        cudaEventRecord(t1); cudaEventSynchronize(t1); float m5; cudaEventElapsedTime(&m5,t0,t1); m5/=it;
        printf("grouped v5 (work-list): %.3f ms  (%.2fx vs v4, %.2fx vs v1)  n_work=%d (vs %d empty-padded)\n",
               m5, m4/m5, m1/m5, n_work, n_experts*gy);
        printf("v5 vs v1 out_compact: rel_L2 = %.6f  (expect ~0)\n", ev5);
        ok = ok && (ev5<1e-4);

        /* ---- v6: v5 + shared-memory codebook staging ---- */
        float *dOutC6; CUDA_CHECK(cudaMalloc(&dOutC6,(size_t)total*N*sizeof(float)));
        CUDA_CHECK(cudaMemset(dOutC6,0,(size_t)total*N*sizeof(float)));
        mmq_iq2xxs_grouped_v6<<<gg5,128>>>(dOutC6,dW,estride,dCxq,dCxs,dEbounds,dWork,N,K);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        float *oc6=(float*)malloc((size_t)total*N*sizeof(float));
        CUDA_CHECK(cudaMemcpy(oc6,dOutC6,(size_t)total*N*sizeof(float),cudaMemcpyDeviceToHost));
        double ev6=rel_l2(oc6,oc1,total*N);
        cudaEventRecord(t0); for(int i=0;i<it;i++) mmq_iq2xxs_grouped_v6<<<gg5,128>>>(dOutC6,dW,estride,dCxq,dCxs,dEbounds,dWork,N,K);
        cudaEventRecord(t1); cudaEventSynchronize(t1); float m6; cudaEventElapsedTime(&m6,t0,t1); m6/=it;
        printf("grouped v6 (shared codebook): %.3f ms  (%.2fx vs v5, %.2fx vs v1)\n", m6, m5/m6, m1/m6);
        printf("v6 vs v1 out_compact: rel_L2 = %.6f  (expect ~0)\n", ev6);
        ok = ok && (ev6<1e-4);

        /* ---- v7: v6 + coalesced contiguous weight-tile staging (dynamic shared) ---- */
        int row_bytes = (K/256)*66, smem7 = 64*row_bytes;
        cudaFuncSetAttribute(mmq_iq2xxs_grouped_v7, cudaFuncAttributeMaxDynamicSharedMemorySize, smem7);
        float *dOutC7; CUDA_CHECK(cudaMalloc(&dOutC7,(size_t)total*N*sizeof(float)));
        CUDA_CHECK(cudaMemset(dOutC7,0,(size_t)total*N*sizeof(float)));
        mmq_iq2xxs_grouped_v7<<<gg5,128,smem7>>>(dOutC7,dW,estride,dCxq,dCxs,dEbounds,dWork,N,K);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        float *oc7=(float*)malloc((size_t)total*N*sizeof(float));
        CUDA_CHECK(cudaMemcpy(oc7,dOutC7,(size_t)total*N*sizeof(float),cudaMemcpyDeviceToHost));
        double ev7=rel_l2(oc7,oc1,total*N);
        cudaEventRecord(t0); for(int i=0;i<it;i++) mmq_iq2xxs_grouped_v7<<<gg5,128,smem7>>>(dOutC7,dW,estride,dCxq,dCxs,dEbounds,dWork,N,K);
        cudaEventRecord(t1); cudaEventSynchronize(t1); float m7; cudaEventElapsedTime(&m7,t0,t1); m7/=it;
        printf("grouped v7 (coalesced tile, smem=%dB): %.3f ms  (%.2fx vs v6, %.2fx vs v1)\n", smem7, m7, m6/m7, m1/m7);
        printf("v7 vs v1 out_compact: rel_L2 = %.6f  (expect ~0)\n", ev7);
        ok = ok && (ev7<1e-4);

        /* ---- v8: v6 + branchless vectorized sign application (__vsub4) ---- */
        float *dOutC8; CUDA_CHECK(cudaMalloc(&dOutC8,(size_t)total*N*sizeof(float)));
        CUDA_CHECK(cudaMemset(dOutC8,0,(size_t)total*N*sizeof(float)));
        mmq_iq2xxs_grouped_v8<<<gg5,128>>>(dOutC8,dW,estride,dCxq,dCxs,dEbounds,dWork,N,K);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        float *oc8=(float*)malloc((size_t)total*N*sizeof(float));
        CUDA_CHECK(cudaMemcpy(oc8,dOutC8,(size_t)total*N*sizeof(float),cudaMemcpyDeviceToHost));
        double ev8=rel_l2(oc8,oc1,total*N);
        cudaEventRecord(t0); for(int i=0;i<it;i++) mmq_iq2xxs_grouped_v8<<<gg5,128>>>(dOutC8,dW,estride,dCxq,dCxs,dEbounds,dWork,N,K);
        cudaEventRecord(t1); cudaEventSynchronize(t1); float m8; cudaEventElapsedTime(&m8,t0,t1); m8/=it;
        printf("grouped v8 (branchless decode): %.3f ms  (%.2fx vs v6, %.2fx vs v1)\n", m8, m6/m8, m1/m8);
        printf("v8 vs v1 out_compact: rel_L2 = %.6f  (expect ~0)\n", ev8);
        ok = ok && (ev8<1e-4);

        /* ---- v9: v8 + double-buffered software pipeline ---- */
        float *dOutC9; CUDA_CHECK(cudaMalloc(&dOutC9,(size_t)total*N*sizeof(float)));
        CUDA_CHECK(cudaMemset(dOutC9,0,(size_t)total*N*sizeof(float)));
        mmq_iq2xxs_grouped_v9<<<gg5,128>>>(dOutC9,dW,estride,dCxq,dCxs,dEbounds,dWork,N,K);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        float *oc9=(float*)malloc((size_t)total*N*sizeof(float));
        CUDA_CHECK(cudaMemcpy(oc9,dOutC9,(size_t)total*N*sizeof(float),cudaMemcpyDeviceToHost));
        double ev9=rel_l2(oc9,oc1,total*N);
        cudaEventRecord(t0); for(int i=0;i<it;i++) mmq_iq2xxs_grouped_v9<<<gg5,128>>>(dOutC9,dW,estride,dCxq,dCxs,dEbounds,dWork,N,K);
        cudaEventRecord(t1); cudaEventSynchronize(t1); float m9; cudaEventElapsedTime(&m9,t0,t1); m9/=it;
        printf("grouped v9 (pipelined): %.3f ms  (%.2fx vs v8, %.2fx vs v1)\n", m9, m8/m9, m1/m9);
        printf("v9 vs v1 out_compact: rel_L2 = %.6f  (expect ~0)\n", ev9);
        ok = ok && (ev9<1e-4);

        /* ---- v10: v8 + direct aligned int loads (skip pack4 reassembly) ---- */
        float *dOutC10; CUDA_CHECK(cudaMalloc(&dOutC10,(size_t)total*N*sizeof(float)));
        CUDA_CHECK(cudaMemset(dOutC10,0,(size_t)total*N*sizeof(float)));
        mmq_iq2xxs_grouped_v10<<<gg5,128>>>(dOutC10,dW,estride,dCxq,dCxs,dEbounds,dWork,N,K);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        float *oc10=(float*)malloc((size_t)total*N*sizeof(float));
        CUDA_CHECK(cudaMemcpy(oc10,dOutC10,(size_t)total*N*sizeof(float),cudaMemcpyDeviceToHost));
        double ev10=rel_l2(oc10,oc1,total*N);
        cudaEventRecord(t0); for(int i=0;i<it;i++) mmq_iq2xxs_grouped_v10<<<gg5,128>>>(dOutC10,dW,estride,dCxq,dCxs,dEbounds,dWork,N,K);
        cudaEventRecord(t1); cudaEventSynchronize(t1); float m10; cudaEventElapsedTime(&m10,t0,t1); m10/=it;
        printf("grouped v10 (direct int loads): %.3f ms  (%.2fx vs v8, %.2fx vs v1)\n", m10, m8/m10, m1/m10);
        printf("v10 vs v1 out_compact: rel_L2 = %.6f  (expect ~0)\n", ev10);
        ok = ok && (ev10<1e-4);
    }
    printf("%s\n", ok?"PASS (mul_mat_id mechanics correct)":"FAIL");
    return ok?0:1;
}
