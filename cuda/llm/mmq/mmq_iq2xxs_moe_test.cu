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
__device__ static float h2f_dev(uint16_t h){ return __half2float(*(const __half*)&h); }
__device__ __forceinline__ int pack4(const int8_t*p){ return (p[0]&0xff)|((p[1]&0xff)<<8)|((p[2]&0xff)<<16)|((p[3]&0xff)<<24); }

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
    }
    printf("%s\n", ok?"PASS (mul_mat_id mechanics correct)":"FAIL");
    return ok?0:1;
}
