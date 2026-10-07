/* qimg f16 flash attention for gfx1201: verbatim fa / fa_db kernels from rdna4/fa2/bench_fa2.cpp
 * (HD=128, BQ=256, BKV=32, 16 waves, LDS XOR swizzle, double-buffered K/V; ~37 TF/s at S=4096 vs ~5 for
 * flash_attn_sa_wmma_pq_f32), plus a pack kernel from the runner's f32 token-major q/k/v.
 * Layouts: Q [N, H*HD] f16 token-major; Kt/Vt [H][N][HD] f16; out [N, H*HD] f32. Non-causal, no mask. */
#define SWZ 1

typedef unsigned int u32;
typedef unsigned char u8;
typedef float float8 __attribute__((ext_vector_type(8)));
#ifndef HD
#define HD 128
#endif
#define K_NB (HD/16)
#define LOG2E 1.4426950408889634f
__device__ __forceinline__ float exp2_fast(float x){if(x<=-24.0f)return 0.0f;if(x>8.0f)x=8.0f;union{int i;float f;}u;u.i=(int)((x+127.0f)*8388608.0f);return u.f;}
/* blog XOR swizzle: spread BKV rows of HD-stride K across LDS banks, 8-elem aligned */
#ifdef SWZ
#define SW(r) (((r)&7)<<4)
#else
#define SW(r) 0
#endif
#ifdef USE_BF16
  typedef unsigned short wt; typedef wt v8 __attribute__((ext_vector_type(8)));
  #define MMA(A,B,C) C=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(A,B,C)
  __device__ __forceinline__ wt CVT(float f){union{float f;u32 i;}u;u.f=f;return (wt)(u.i>>16);}
#else
  typedef _Float16 wt; typedef wt v8 __attribute__((ext_vector_type(8)));
  #define MMA(A,B,C) C=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(A,B,C)
  __device__ __forceinline__ wt CVT(float f){return (wt)f;}
#endif
extern "C" __global__ __launch_bounds__(512,1)
void fa(float*out,const wt*Q,const wt*Kt,const wt*Vt,int N,int H,float isd){
  enum{BQ=256,BKV=32,WC=16,CH=2}; int h=blockIdx.x,qb=blockIdx.y,q0=qb*BQ,tid=threadIdx.x,wid=tid>>5,lid=tid&31,hf=lid>>4,id=lid&15,dim=H*HD;
  __shared__ wt smK[BKV*HD]; __shared__ wt smV[HD*BKV]; __shared__ wt smP[WC*CH*256]; wt*sP=smP+wid*CH*256;
  v8 qr[K_NB/4][4]; for(int g=0;g<K_NB/4;g++)for(int i=0;i<4;i++){int kb=g*4+i,r=q0+wid*16+id; v8 t; for(int j=0;j<8;j++){int d=kb*16+hf*8+j;t[j]=(r<N)?Q[(size_t)r*dim+h*HD+d]:(wt)0;}qr[g][i]=t;}
  float8 O[K_NB]; for(int kb=0;kb<K_NB;kb++)for(int i=0;i<8;i++)O[kb][i]=0; float mi[8],li[8]; for(int i=0;i<8;i++){mi[i]=-1e30f;li[i]=0;}
  int nt=(N+BKV-1)/BKV,tot=BKV*HD,st=WC*32*16;
  for(int t=0;t<nt;t++){int kv0=t*BKV;
    for(int o=tid*16;o<tot;o+=st){int r=o/HD,d=o%HD,kv=kv0+r;size_t b=((size_t)h*N+kv)*HD+d;if(kv<N&&d+16<=HD){v8 k0=*(const v8*)(Kt+b),k1=*(const v8*)(Kt+b+8),v0=*(const v8*)(Vt+b),v1=*(const v8*)(Vt+b+8);*(v8*)(smK+r*HD+(d^SW(r)))=k0;*(v8*)(smK+r*HD+((d+8)^SW(r)))=k1;for(int i=0;i<8;i++)smV[(d+i)*BKV+r]=v0[i];for(int i=0;i<8;i++)smV[(d+8+i)*BKV+r]=v1[i];}else for(int i=0;i<16;i++){wt kk=0,vv=0;if(kv<N&&d+i<HD){kk=Kt[b+i];vv=Vt[b+i];}smK[r*HD+((d+i)^SW(r))]=kk;smV[(d+i)*BKV+r]=vv;}}
    __syncthreads(); float8 S[CH]; for(int c=0;c<CH;c++)for(int i=0;i<8;i++)S[c][i]=0;
    for(int g=0;g<K_NB/4;g++)for(int c=0;c<CH;c++){v8 kr[4];for(int i=0;i<4;i++){int kb=g*4+i,r=c*16+id,d=kb*16+hf*8;kr[i]=*(const v8*)(smK+r*HD+(d^SW(r)));}for(int i=0;i<4;i++)MMA(qr[g][i],kr[i],S[c]);}
    float rm[8];for(int i=0;i<8;i++){float mx=-1e30f;for(int c=0;c<CH;c++){int kv=kv0+c*16+id;float s=S[c][i]*isd*LOG2E;if(kv>=N)s=-1e30f;S[c][i]=s;mx=fmaxf(mx,s);}mx=fmaxf(mx,__shfl_xor(mx,1,32));mx=fmaxf(mx,__shfl_xor(mx,2,32));mx=fmaxf(mx,__shfl_xor(mx,4,32));mx=fmaxf(mx,__shfl_xor(mx,8,32));rm[i]=mx;}
    float al[8];for(int i=0;i<8;i++){float nm=fmaxf(mi[i],rm[i]);al[i]=exp2_fast(mi[i]-nm);float lo=0;for(int c=0;c<CH;c++){int kv=kv0+c*16+id;float p=exp2_fast(S[c][i]-nm+8.0f);if(kv>=N)p=0;S[c][i]=p;lo+=p;}li[i]=li[i]*al[i]+lo;mi[i]=nm;}
    for(int kb=0;kb<K_NB;kb++)for(int i=0;i<8;i++)O[kb][i]*=al[i];
    for(int c=0;c<CH;c++){wt*pc=sP+c*256;for(int i=0;i<8;i++)pc[(hf*8+i)*16+id]=CVT(S[c][i]);}__builtin_amdgcn_s_waitcnt(0);__syncthreads();
    for(int c=0;c<CH;c++){wt*pc=sP+c*256;v8 ap=*(const v8*)(pc+id*16+hf*8);for(int kb=0;kb<K_NB;kb++){int d=kb*16+id;MMA(ap,*(const v8*)(smV+d*BKV+c*16+hf*8),O[kb]);}}
  }
  for(int i=0;i<8;i++){float l=li[i];l+=__shfl_xor(l,1,32);l+=__shfl_xor(l,2,32);l+=__shfl_xor(l,4,32);l+=__shfl_xor(l,8,32);li[i]=l>0?1/l:0;}
  for(int kb=0;kb<K_NB;kb++)for(int i=0;i<8;i++){int r=q0+wid*16+hf*8+i,d=kb*16+id;if(r<N)out[(size_t)r*dim+h*HD+d]=O[kb][i]*li[i];}
}
/* double-buffered: prefetch tile t+1 into alt buffer while computing t */
#define LD(KK,VV,KV0) for(int o=tid*16;o<tot;o+=st){int r=o/HD,d=o%HD,kv=(KV0)+r;size_t b=((size_t)h*N+kv)*HD+d;if(kv<N&&d+16<=HD){v8 k0=*(const v8*)(Kt+b),k1=*(const v8*)(Kt+b+8),v0=*(const v8*)(Vt+b),v1=*(const v8*)(Vt+b+8);*(v8*)((KK)+r*HD+(d^SW(r)))=k0;*(v8*)((KK)+r*HD+((d+8)^SW(r)))=k1;for(int i=0;i<8;i++)(VV)[(d+i)*BKV+r]=v0[i];for(int i=0;i<8;i++)(VV)[(d+8+i)*BKV+r]=v1[i];}else for(int i=0;i<16;i++){wt kk=0,vv=0;if(kv<N&&d+i<HD){kk=Kt[b+i];vv=Vt[b+i];}(KK)[r*HD+((d+i)^SW(r))]=kk;(VV)[(d+i)*BKV+r]=vv;}}
extern "C" __global__ __launch_bounds__(512,1)
void fa_db(float*out,const wt*Q,const wt*Kt,const wt*Vt,int N,int H,float isd){
  enum{BQ=256,BKV=32,WC=16,CH=2}; int h=blockIdx.x,qb=blockIdx.y,q0=qb*BQ,tid=threadIdx.x,wid=tid>>5,lid=tid&31,hf=lid>>4,id=lid&15,dim=H*HD;
  __shared__ wt smK[2][BKV*HD]; __shared__ wt smV[2][HD*BKV]; __shared__ wt smP[WC*CH*256]; wt*sP=smP+wid*CH*256;
  v8 qr[K_NB/4][4]; for(int g=0;g<K_NB/4;g++)for(int i=0;i<4;i++){int kb=g*4+i,r=q0+wid*16+id; v8 t; for(int j=0;j<8;j++){int d=kb*16+hf*8+j;t[j]=(r<N)?Q[(size_t)r*dim+h*HD+d]:(wt)0;}qr[g][i]=t;}
  float8 O[K_NB]; for(int kb=0;kb<K_NB;kb++)for(int i=0;i<8;i++)O[kb][i]=0; float mi[8],li[8]; for(int i=0;i<8;i++){mi[i]=-1e30f;li[i]=0;}
  int nt=(N+BKV-1)/BKV,tot=BKV*HD,st=WC*32*16; LD(smK[0],smV[0],0); __syncthreads();
  for(int t=0;t<nt;t++){int kv0=t*BKV,cur=t&1,nx=cur^1; wt*cK=smK[cur],*cV=smV[cur];
    if(t+1<nt){LD(smK[nx],smV[nx],(t+1)*BKV);}
    float8 S[CH]; for(int c=0;c<CH;c++)for(int i=0;i<8;i++)S[c][i]=0;
    for(int g=0;g<K_NB/4;g++)for(int c=0;c<CH;c++){v8 kr[4];for(int i=0;i<4;i++){int kb=g*4+i,r=c*16+id,d=kb*16+hf*8;kr[i]=*(const v8*)(cK+r*HD+(d^SW(r)));}for(int i=0;i<4;i++)MMA(qr[g][i],kr[i],S[c]);}
    float rm[8];for(int i=0;i<8;i++){float mx=-1e30f;for(int c=0;c<CH;c++){int kv=kv0+c*16+id;float s=S[c][i]*isd*LOG2E;if(kv>=N)s=-1e30f;S[c][i]=s;mx=fmaxf(mx,s);}mx=fmaxf(mx,__shfl_xor(mx,1,32));mx=fmaxf(mx,__shfl_xor(mx,2,32));mx=fmaxf(mx,__shfl_xor(mx,4,32));mx=fmaxf(mx,__shfl_xor(mx,8,32));rm[i]=mx;}
    float al[8];for(int i=0;i<8;i++){float nm=fmaxf(mi[i],rm[i]);al[i]=exp2_fast(mi[i]-nm);float lo=0;for(int c=0;c<CH;c++){int kv=kv0+c*16+id;float p=exp2_fast(S[c][i]-nm+8.0f);if(kv>=N)p=0;S[c][i]=p;lo+=p;}li[i]=li[i]*al[i]+lo;mi[i]=nm;}
    for(int kb=0;kb<K_NB;kb++)for(int i=0;i<8;i++)O[kb][i]*=al[i];
    for(int c=0;c<CH;c++){wt*pc=sP+c*256;for(int i=0;i<8;i++)pc[(hf*8+i)*16+id]=CVT(S[c][i]);}__builtin_amdgcn_s_waitcnt(0);__syncthreads();
    for(int c=0;c<CH;c++){wt*pc=sP+c*256;v8 ap=*(const v8*)(pc+id*16+hf*8);for(int kb=0;kb<K_NB;kb++){int d=kb*16+id;MMA(ap,*(const v8*)(cV+d*BKV+c*16+hf*8),O[kb]);}}
    __syncthreads();
  }
  for(int i=0;i<8;i++){float l=li[i];l+=__shfl_xor(l,1,32);l+=__shfl_xor(l,2,32);l+=__shfl_xor(l,4,32);l+=__shfl_xor(l,8,32);li[i]=l>0?1/l:0;}
  for(int kb=0;kb<K_NB;kb++)for(int i=0;i<8;i++){int r=q0+wid*16+hf*8+i,d=kb*16+id;if(r<N)out[(size_t)r*dim+h*HD+d]=O[kb][i]*li[i];}
}

/* q/k/v f32 [N, H*HD] -> Qh f16 [N, H*HD], Kt/Vt f16 [H][N][HD]. One thread per 8 elements. */
extern "C" __global__ void fa16_pack(wt *Qh, wt *Kt, wt *Vt, const float *q, const float *k, const float *v,
                                     int N, int H) {
  long e = ((long)blockIdx.x * blockDim.x + threadIdx.x) * 8, total = (long)N * H * HD;
  if (e >= total) return;
  int t = (int)(e / (H * HD)), c = (int)(e % (H * HD)), h = c / HD, d = c % HD;
  long ht = ((long)h * N + t) * HD + d;
  for (int i = 0; i < 8; i++) { Qh[e + i] = CVT(q[e + i]); Kt[ht + i] = CVT(k[e + i]); Vt[ht + i] = CVT(v[e + i]); }
}
