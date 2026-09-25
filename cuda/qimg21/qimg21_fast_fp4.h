/* NVFP4 SVDQuant W4A4 kernels for the fast Qwen-Image 2.1 denoiser, compiled
 * for sm_120a (mma.sync m16n8k64 kind::mxf4nvf4 block_scale). The GEMM main
 * loop is cuda/fp4_w4a4.h's validated w4a4_gemm_opt (64x128x64 tile, 8 warps);
 * the epilogue applies the per-token and per-row F32 scales, adds the BF16
 * low-rank branch already stored in y, and writes BF16 with a row stride.
 * Activations use two-level scales like the weights: an F32 scale per token
 * that places the largest 16-element group scale at 448, and one E4M3 scale per
 * group, so no group saturates or underflows the E4M3 range. */
#ifndef QIMG21_FAST_FP4_H
#define QIMG21_FAST_FP4_H
static const char *q21f_fp4_src =
"typedef unsigned short bf;\n"
"__device__ __forceinline__ float b2f(bf x){return __uint_as_float(((unsigned)x)<<16);}\n"
"__device__ __forceinline__ bf f2b(float x){unsigned u=__float_as_uint(x);return (bf)((u+0x7fffu+((u>>16)&1u))>>16);}\n"
"__device__ __forceinline__ unsigned char e4m3(float v){unsigned short p;asm(\"cvt.rn.satfinite.e4m3x2.f32 %0,%1,%2;\":\"=h\"(p):\"f\"(v),\"f\"(v));return p&0xFF;}\n"
"__device__ __forceinline__ float e4m3_dec(unsigned char b){int e=(b>>3)&0xF,m=b&7;if(e==0)return (m/8.0f)*0.015625f;return (1.0f+m/8.0f)*exp2f((float)(e-7));}\n"
"__device__ __forceinline__ unsigned e2m1(float q){unsigned s=q<0?8:0;float a=fabsf(q);unsigned c;\n"
" if(a<0.25f)c=0;else if(a<0.75f)c=1;else if(a<1.25f)c=2;else if(a<1.75f)c=3;else if(a<2.5f)c=4;else if(a<3.5f)c=5;else if(a<5.0f)c=6;else c=7;return s|c;}\n"
"extern \"C\" {\n"
/* One 256-thread block per row: v = x*inv_s, ts = amax/(6*448), group scale
 * e4m3(gamax/6/ts), codes e2m1(v/(scale*ts)). codes [M,K/8] u32, gs [M,K/16]. */
"__global__ void fp4_act(unsigned*codes,unsigned char*gs,float*ts,const bf*x,int ld,const float*inv_s,int K){\n"
" int t=blockIdx.x,i=threadIdx.x;__shared__ float amx[8];const bf*r=x+(size_t)t*ld;float m=0.f;\n"
" for(int j=i;j<K;j+=256)m=fmaxf(m,fabsf(b2f(r[j])*inv_s[j]));\n"
" for(int s=16;s;s>>=1)m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,s));if(i%32==0)amx[i/32]=m;__syncthreads();\n"
" m=amx[0];for(int k=1;k<8;k++)m=fmaxf(m,amx[k]);float tsc=m>0.f?m/(6.f*448.f):1.f,itsc=1.f/tsc;\n"
" if(i==0)ts[t]=tsc;int ng=K>>4;\n"
" for(int g=i;g<ng;g+=256){float v[16],ga=0.f;for(int e=0;e<16;e++){v[e]=b2f(r[g*16+e])*inv_s[g*16+e]*itsc;ga=fmaxf(ga,fabsf(v[e]));}\n"
"  unsigned char sb=e4m3(ga*(1.f/6.f));float sd=e4m3_dec(sb),inv=sd>0.f?1.f/sd:0.f;gs[(size_t)t*ng+g]=sb;\n"
"  for(int u=0;u<2;u++){unsigned w=0;for(int e=0;e<8;e++)w|=e2m1(v[u*8+e]*inv)<<(e*4);codes[(size_t)t*(K>>3)+g*2+u]=w;}}\n"
"}\n"
/* CUTLASS path, pass 1: gmax = max over rows of |x*inv_s| (gmax zeroed). */
"__global__ void fp4_rowmax(float*gmax,const bf*x,int ld,const float*inv_s,int K){\n"
" int t=blockIdx.x,i=threadIdx.x;__shared__ float amx[8];const bf*r=x+(size_t)t*ld;float m=0.f;\n"
" for(int j=i;j<K;j+=256)m=fmaxf(m,fabsf(b2f(r[j])*inv_s[j]));\n"
" for(int s=16;s;s>>=1)m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,s));if(i%32==0)amx[i/32]=m;__syncthreads();\n"
" if(i==0){m=amx[0];for(int k=1;k<8;k++)m=fmaxf(m,amx[k]);atomicMax((int*)gmax,__float_as_int(m));}\n"
"}\n"
/* Pass 2: one F32 scale for the whole activation (largest group scale at
 * 448), E4M3 group scales in CUTLASS's interleaved layout (128-row x 4-group
 * atoms of 512 bytes), and alpha = activation scale * weight scale. */
"__global__ void fp4_act_cl(unsigned*codes,unsigned char*sf,float*alpha,const float*gmax,const float*wc,const bf*x,int ld,const float*inv_s,int K){\n"
" int t=blockIdx.x,i=threadIdx.x;const bf*r=x+(size_t)t*ld;float g0=*gmax;float tsc=g0>0.f?g0/(6.f*448.f):1.f,itsc=1.f/tsc;\n"
" if(t==0&&i==0)*alpha=tsc*wc[0];int ng=K>>4;size_t atom=((size_t)(t>>7)*(ng>>2))*512+(t&31)*16+((t&127)>>5)*4;\n"
" for(int g=i;g<ng;g+=256){float v[16],ga=0.f;for(int e=0;e<16;e++){v[e]=b2f(r[g*16+e])*inv_s[g*16+e]*itsc;ga=fmaxf(ga,fabsf(v[e]));}\n"
"  unsigned char sb=e4m3(ga*(1.f/6.f));float sd=e4m3_dec(sb),inv=sd>0.f?1.f/sd:0.f;sf[atom+(size_t)(g>>2)*512+(g&3)]=sb;\n"
"  for(int u=0;u<2;u++){unsigned w=0;for(int e=0;e<8;e++)w|=e2m1(v[u*8+e]*inv)<<(e*4);codes[(size_t)t*(K>>3)+g*2+u]=w;}}\n"
"}\n"
"#define BM 64\n#define BN 128\n#define BK 64\n#define WN_WARPS 4\n#define MSUB 2\n#define NSUB 4\n"
/* y[m, n] = bf16(acc * ts[m] * wc[n] + (lr ? y[m, n] : 0)). */
"__global__ __launch_bounds__(256) void w4a4_bf16(const unsigned*__restrict__ A,const unsigned*__restrict__ B,\n"
"  const unsigned char*__restrict__ sA,const unsigned char*__restrict__ sB,const float*__restrict__ ts,\n"
"  const float*__restrict__ wc,bf*y,int ldy,int lr,int M,int N,int K){\n"
"  __shared__ unsigned smA[BM][BK/8];__shared__ unsigned smB[BN][BK/8];\n"
"  __shared__ unsigned char smSA[BM][BK/16];__shared__ unsigned char smSB[BN][BK/16];\n"
"  int bm0=blockIdx.y*BM,bn0=blockIdx.x*BN,tid=threadIdx.x,warp=tid>>5,lane=tid&31;\n"
"  int wm=warp/WN_WARPS,wn=warp%WN_WARPS,g=lane>>2,t=lane&3;long Ku=K>>3,Kg=K>>4;\n"
"  float acc[MSUB][NSUB][4];\n"
"  #pragma unroll\n"
"  for(int i=0;i<MSUB;i++)for(int j=0;j<NSUB;j++)for(int e=0;e<4;e++)acc[i][j][e]=0.f;\n"
"  for(int kt=0;kt<K/BK;kt++){long k0u=(long)kt*(BK/8),k0g=(long)kt*(BK/16);\n"
"    for(int idx=tid;idx<BM*(BK/8);idx+=256){int r=idx/(BK/8),c=idx%(BK/8),gr=bm0+r;smA[r][c]=(gr<M)?A[(long)gr*Ku+k0u+c]:0u;}\n"
"    for(int idx=tid;idx<BN*(BK/8);idx+=256){int r=idx/(BK/8),c=idx%(BK/8),gr=bn0+r;smB[r][c]=(gr<N)?B[(long)gr*Ku+k0u+c]:0u;}\n"
"    for(int idx=tid;idx<BM*(BK/16);idx+=256){int r=idx/(BK/16),c=idx%(BK/16),gr=bm0+r;smSA[r][c]=(gr<M)?sA[(long)gr*Kg+k0g+c]:0x38;}\n"
"    for(int idx=tid;idx<BN*(BK/16);idx+=256){int r=idx/(BK/16),c=idx%(BK/16),gr=bn0+r;smSB[r][c]=(gr<N)?sB[(long)gr*Kg+k0g+c]:0x38;}\n"
"    __syncthreads();\n"
"    unsigned af[MSUB][4],sfa[MSUB],bfr[NSUB][2],sfb[NSUB];\n"
"    #pragma unroll\n"
"    for(int i=0;i<MSUB;i++){int mr=wm*(BM/2)+i*16;af[i][0]=smA[mr+g][t];af[i][1]=smA[mr+g+8][t];af[i][2]=smA[mr+g][t+4];af[i][3]=smA[mr+g+8][t+4];\n"
"      unsigned s=0x38383838u;if(t==0){const unsigned char*p=smSA[mr+g];s=p[0]|(p[1]<<8)|(p[2]<<16)|(p[3]<<24);}\n"
"      else if(t==1){const unsigned char*p=smSA[mr+g+8];s=p[0]|(p[1]<<8)|(p[2]<<16)|(p[3]<<24);}sfa[i]=s;}\n"
"    #pragma unroll\n"
"    for(int j=0;j<NSUB;j++){int nr=wn*(BN/WN_WARPS)+j*8;bfr[j][0]=smB[nr+g][t];bfr[j][1]=smB[nr+g][t+4];\n"
"      unsigned s=0x38383838u;if(t==0){const unsigned char*p=smSB[nr+g];s=p[0]|(p[1]<<8)|(p[2]<<16)|(p[3]<<24);}sfb[j]=s;}\n"
"    #pragma unroll\n"
"    for(int i=0;i<MSUB;i++)\n"
"      #pragma unroll\n"
"      for(int j=0;j<NSUB;j++)\n"
"        asm volatile(\"mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 \"\n"
"          \"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};\"\n"
"          :\"+f\"(acc[i][j][0]),\"+f\"(acc[i][j][1]),\"+f\"(acc[i][j][2]),\"+f\"(acc[i][j][3])\n"
"          :\"r\"(af[i][0]),\"r\"(af[i][1]),\"r\"(af[i][2]),\"r\"(af[i][3]),\"r\"(bfr[j][0]),\"r\"(bfr[j][1]),\"r\"(sfa[i]),\"r\"(sfb[j]));\n"
"    __syncthreads();}\n"
"  #pragma unroll\n"
"  for(int i=0;i<MSUB;i++)\n"
"    #pragma unroll\n"
"    for(int j=0;j<NSUB;j++){int mr=bm0+wm*(BM/2)+i*16,nc=bn0+wn*(BN/WN_WARPS)+j*8,col=nc+2*t;\n"
"      for(int h=0;h<2;h++){int row=mr+g+8*h;if(row>=M)continue;float s=ts[row];\n"
"        for(int c=0;c<2;c++){int cc=col+c;if(cc>=N)continue;bf*p=y+(size_t)row*ldy+cc;\n"
"          *p=f2b(acc[i][j][2*h+c]*s*wc[cc]+(lr?b2f(*p):0.f));}}}\n"
"}\n"
"}\n";
#endif
