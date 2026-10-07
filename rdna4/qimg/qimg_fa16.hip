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

/* fa_db_vt: V pre-transposed by the pack kernel to [H][HD][Np] (Np = N rounded up to 32, zero tail), so the
 * V tile is copied with 16-byte vectors like K instead of 16 strided 2-byte LDS stores per thread. */
#define LDVT(KK,VV,KV0) { for(int o=tid*16;o<tot;o+=st){int r=o/HD,d=o%HD,kv=(KV0)+r;size_t b=((size_t)h*N+kv)*HD+d; \
  if(kv<N&&d+16<=HD){v8 k0=*(const v8*)(Kt+b),k1=*(const v8*)(Kt+b+8);*(v8*)((KK)+r*HD+(d^SW(r)))=k0;*(v8*)((KK)+r*HD+((d+8)^SW(r)))=k1;} \
  else for(int i=0;i<16;i++){wt kk=0;if(kv<N&&d+i<HD)kk=Kt[b+i];(KK)[r*HD+((d+i)^SW(r))]=kk;}} \
  for(int o=tid*16;o<HD*BKV;o+=st){int d=o/BKV,r0=o%BKV;const wt*src=Vt+((size_t)h*HD+d)*Np+(KV0)+r0; \
  *(v8*)((VV)+d*BKV+r0)=*(const v8*)src;*(v8*)((VV)+d*BKV+r0+8)=*(const v8*)(src+8);} }
extern "C" __global__ __launch_bounds__(512,1)
void fa_db_vt(float*out,const wt*Q,const wt*Kt,const wt*Vt,int N,int H,float isd,int Np){
  enum{BQ=256,BKV=32,WC=16,CH=2}; int h=blockIdx.x,qb=blockIdx.y,q0=qb*BQ,tid=threadIdx.x,wid=tid>>5,lid=tid&31,hf=lid>>4,id=lid&15,dim=H*HD;
  __shared__ wt smK[2][BKV*HD]; __shared__ wt smV[2][HD*BKV]; __shared__ wt smP[WC*CH*256]; wt*sP=smP+wid*CH*256;
  v8 qr[K_NB/4][4]; for(int g=0;g<K_NB/4;g++)for(int i=0;i<4;i++){int kb=g*4+i,r=q0+wid*16+id; v8 t; for(int j=0;j<8;j++){int d=kb*16+hf*8+j;t[j]=(r<N)?Q[(size_t)r*dim+h*HD+d]:(wt)0;}qr[g][i]=t;}
  float8 O[K_NB]; for(int kb=0;kb<K_NB;kb++)for(int i=0;i<8;i++)O[kb][i]=0; float mi[8],li[8]; for(int i=0;i<8;i++){mi[i]=-1e30f;li[i]=0;}
  int nt=(N+BKV-1)/BKV,tot=BKV*HD,st=WC*32*16; LDVT(smK[0],smV[0],0); __syncthreads();
  for(int t=0;t<nt;t++){int kv0=t*BKV,cur=t&1,nx=cur^1; wt*cK=smK[cur],*cV=smV[cur];
    if(t+1<nt){LDVT(smK[nx],smV[nx],(t+1)*BKV);}
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

/* Fused per-head RMSNorm + 3-axis RoPE + f16 pack for the joint [txt | img] q/k/v (edit layout).
 * One wave per (token, head); lane l owns dims 4l..4l+3 (= rotary pairs 2l, 2l+1). Text tokens use the
 * norm_added_* weights and position txt_start+tok on all axes; image tokens use norm_* weights and the
 * per-token (frame, h, w) table. Same math order as rmsnorm_per_head -> rope_pos/rope_1d -> fa16_pack. */
extern "C" __global__ void fa16_norm_rope_pack(wt *Qh, wt *Kt, wt *Vt, const float *q, const float *k, const float *v,
        int N, int H, int n_txt, const float *wq_img, const float *wk_img, const float *wq_txt, const float *wk_txt,
        const int *pos_img, int txt_start, int t_dim, int h_dim, int w_dim, float theta) {
  int wave = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (wave >= N * H) return;
  int tok = wave / H, h = wave % H, dim = H * HD;
  long src = (long)tok * dim + h * HD + lane * 4;
  float4 qv = *(const float4 *)(q + src), kv = *(const float4 *)(k + src), vv = *(const float4 *)(v + src);
  int txt = tok < n_txt;
  const float *wq = (txt ? wq_txt : wq_img) + lane * 4, *wk = (txt ? wk_txt : wk_img) + lane * 4;
  float sq = qv.x*qv.x + qv.y*qv.y + qv.z*qv.z + qv.w*qv.w, sk = kv.x*kv.x + kv.y*kv.y + kv.z*kv.z + kv.w*kv.w;
  for (int m = 16; m > 0; m >>= 1) { sq += __shfl_xor(sq, m, 32); sk += __shfl_xor(sk, m, 32); }
  float iq = rsqrtf(sq / HD + 1e-6f), ik = rsqrtf(sk / HD + 1e-6f);
  float Q[4] = {qv.x*iq*wq[0], qv.y*iq*wq[1], qv.z*iq*wq[2], qv.w*iq*wq[3]};
  float K[4] = {kv.x*ik*wk[0], kv.y*ik*wk[1], kv.z*ik*wk[2], kv.w*ik*wk[3]};
  for (int pp = 0; pp < 2; pp++) {
    int p = lane * 2 + pp, axis, i, dimA;
    if (p < t_dim / 2) { axis = 0; i = p; dimA = t_dim; }
    else if (p < (t_dim + h_dim) / 2) { axis = 1; i = p - t_dim / 2; dimA = h_dim; }
    else { axis = 2; i = p - (t_dim + h_dim) / 2; dimA = w_dim; }
    float position = txt ? (float)(txt_start + tok) : (float)pos_img[(tok - n_txt) * 3 + axis];
    float ang = position / powf(theta, 2.0f * (float)i / (float)dimA), cs = cosf(ang), sn = sinf(ang);
    float q0 = Q[2*pp], q1 = Q[2*pp+1], k0 = K[2*pp], k1 = K[2*pp+1];
    Q[2*pp] = q0*cs - q1*sn; Q[2*pp+1] = q0*sn + q1*cs; K[2*pp] = k0*cs - k1*sn; K[2*pp+1] = k0*sn + k1*cs;
  }
  long ht = ((long)h * N + tok) * HD + lane * 4;
  float V[4] = {vv.x, vv.y, vv.z, vv.w};
  for (int j = 0; j < 4; j++) { Qh[src + j] = CVT(Q[j]); Kt[ht + j] = CVT(K[j]); Vt[ht + j] = CVT(V[j]); }
}

/* V transpose for fa_db_vt: Vt2[h][d][t] (stride Np), from the fa16 [H][N][HD] V; tail t >= N zero-filled. */
extern "C" __global__ void fa16_vt(wt *Vt2, const wt *Vt, int N, int H, int Np) {
  long e = (long)blockIdx.x * blockDim.x + threadIdx.x, total = (long)H * HD * Np;
  if (e >= total) return;
  int t = (int)(e % Np), d = (int)((e / Np) % HD), h = (int)(e / ((long)Np * HD));
  Vt2[e] = t < N ? Vt[((long)h * N + t) * HD + d] : (wt)0;
}
