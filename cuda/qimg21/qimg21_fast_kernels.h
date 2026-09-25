/* NVRTC kernels for the fast Qwen-Image 2.1 denoiser. Activations are stored
 * as BF16 (the parity harness keeps F32 buffers but rounds them to BF16 at the
 * same points), so every kernel reproduces the harness's exact-mode rounding
 * sequence: vector4 Welford normalization, host-table-exact RoPE, BF16 gates
 * and SwiGLU. Compiled without --use_fast_math. */
#ifndef QIMG21_FAST_KERNELS_H
#define QIMG21_FAST_KERNELS_H
static const char *q21f_kernel_src =
"typedef unsigned short bf;\n"
"__device__ __forceinline__ float b2f(bf x){return __uint_as_float(((unsigned)x)<<16);}\n"
"__device__ __forceinline__ float rb(float x){unsigned u=__float_as_uint(x);return __uint_as_float((u+0x7fffu+((u>>16)&1u))&0xffff0000u);}\n"
"__device__ __forceinline__ bf f2b(float x){unsigned u=__float_as_uint(x);return (bf)((u+0x7fffu+((u>>16)&1u))>>16);}\n"
"struct Stat{float m,v,n;};\n"
"__device__ Stat combine(Stat b,Stat a){float n=a.n+b.n,d=b.m-a.m,r=__frcp_rn(n),na=a.n*r,nb=b.n*r;Stat z;z.n=n;z.m=fmaf(na,a.m,nb*b.m);z.v=fmaf(d*d*a.n,nb,a.v+b.v);return z;}\n"
"extern \"C\" {\n"
/* F32 -> BF16 (round to nearest even). */
"__global__ void cast_bf16(bf*y,const float*x,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=f2b(x[i]);}\n"
/* Zero-centred text RMSNorm, F32 prompt in, BF16 out; same 256-thread tree as the harness. */
"__global__ void txt_norm(bf*y,const float*x,const float*w,int D,float eps){int t=blockIdx.x,i=threadIdx.x;__shared__ float s[256];float z=0;for(int j=i;j<D;j+=256){float v=x[(size_t)t*D+j];z+=v*v;}s[i]=z;__syncthreads();for(int q=128;q;q>>=1){if(i<q)s[i]+=s[i+q];__syncthreads();}float inv=rsqrtf(s[0]/D+eps);for(int j=i;j<D;j+=256)y[(size_t)t*D+j]=f2b(x[(size_t)t*D+j]*inv*(w[j]+1.f));}\n"
/* PyTorch GELU(tanh) with a double-precision tanh, in place on BF16. */
"__global__ void gelu_bf16(bf*x,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=b2f(x[i]),cube=v*v*v,inner=0.7978845608f*fmaf(0.044715f,cube,v);float t=(float)tanh((double)inner);x[i]=f2b((.5f*v)*(1.f+t));}}\n"
"__global__ void silu_bf16(bf*x,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=b2f(x[i]);x[i]=f2b(v/(1.f+expf(-v)));}}\n"
/* Per-step modulation constants shared by all 32 blocks. m is the BF16
 * modulation.1 output [2 rows][scale1,gate1,scale2,gate2 x D]; c gets F32
 * [row][rb(1+scale1),rb(tanh(gate1)),rb(1+scale2),rb(tanh(gate2))]. */
"__global__ void mod_prepare(float*c,const bf*m,int D){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=2*4*D)return;int k=(i/D)%4;float v=b2f(m[i]);c[i]=(k&1)?rb(tanhf(v)):rb(1.f+v);}\n"
/* Final-norm scale: c = rb(1+s) from norm_out.linear BF16 output. */
"__global__ void scale_prepare(float*c,const bf*s,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)c[i]=rb(1.f+b2f(s[i]));}\n"
/* Optional gated residual h=rb(h+rb(g*y)), then vector4 Welford LayerNorm
 * (the harness mod_ln_vector topology) and y=rb(rb(norm)*f). One 128-thread
 * block per row; D must be a multiple of 512. */
"__global__ void norm_mod(bf*out,bf*h,const bf*y,const float*g,const float*f,int D,float eps){\n"
" int t=blockIdx.x,i=threadIdx.x,l=i%32,w=i/32;__shared__ float mean[4],var[4],count[4];Stat a={0,0,0};\n"
" bf*row=h+(size_t)t*D;const bf*yr=y?y+(size_t)t*D:0;\n"
" for(int j=i*4;j<D;j+=512)for(int k=0;k<4;k++){float v=b2f(row[j+k]);if(yr){v=rb(v+rb(g[j+k]*b2f(yr[j+k])));row[j+k]=f2b(v);}float d=v-a.m;float nn=a.n+1.f,nm=fmaf(d,__frcp_rn(nn),a.m);a.v=fmaf(d,v-nm,a.v);a.m=nm;a.n=nn;}\n"
" for(int s=16;s;s>>=1){Stat b={__shfl_down_sync(0xffffffff,a.m,s),__shfl_down_sync(0xffffffff,a.v,s),__shfl_down_sync(0xffffffff,a.n,s)};a=combine(a,b);}\n"
" if(l==0){mean[w]=a.m;var[w]=a.v;count[w]=a.n;}__syncthreads();\n"
" for(int s=2;s;s>>=1){if(l==0&&w<s){Stat b={mean[w+s],var[w+s],count[w+s]};a=combine(a,b);mean[w]=a.m;var[w]=a.v;count[w]=a.n;}__syncthreads();}\n"
" float mu=mean[0],iv=rsqrtf(var[0]/D+eps);bf*o=out+(size_t)t*D;\n"
" for(int j=i*4;j<D;j+=512)for(int k=0;k<4;k++)o[j+k]=f2b(rb((b2f(row[j+k])-mu)*iv)*f[j+k]);\n"
"}\n"
/* norm_mod followed by SmoothQuant W8A8 activation quantization: v =
 * rb(norm)*f * inv_s (the BF16 activation times 1/s), then per-row symmetric
 * INT8 with scale amax/127. D must be 4096 (32 values per thread). */
"__global__ void norm_mod_q8(signed char*q,float*qs,bf*h,const bf*y,const float*g,const float*f,const float*inv_s,int D,float eps){\n"
" int t=blockIdx.x,i=threadIdx.x,l=i%32,w=i/32;__shared__ float mean[4],var[4],count[4],amx[4];Stat a={0,0,0};\n"
" bf*row=h+(size_t)t*D;const bf*yr=y?y+(size_t)t*D:0;\n"
" for(int j=i*4;j<D;j+=512)for(int k=0;k<4;k++){float v=b2f(row[j+k]);if(yr){v=rb(v+rb(g[j+k]*b2f(yr[j+k])));row[j+k]=f2b(v);}float d=v-a.m;float nn=a.n+1.f,nm=fmaf(d,__frcp_rn(nn),a.m);a.v=fmaf(d,v-nm,a.v);a.m=nm;a.n=nn;}\n"
" for(int s=16;s;s>>=1){Stat b={__shfl_down_sync(0xffffffff,a.m,s),__shfl_down_sync(0xffffffff,a.v,s),__shfl_down_sync(0xffffffff,a.n,s)};a=combine(a,b);}\n"
" if(l==0){mean[w]=a.m;var[w]=a.v;count[w]=a.n;}__syncthreads();\n"
" for(int s=2;s;s>>=1){if(l==0&&w<s){Stat b={mean[w+s],var[w+s],count[w+s]};a=combine(a,b);mean[w]=a.m;var[w]=a.v;count[w]=a.n;}__syncthreads();}\n"
" float mu=mean[0],iv=rsqrtf(var[0]/D+eps),v[32],m=0.f;int n=0;\n"
" for(int j=i*4;j<D;j+=512)for(int k=0;k<4;k++){float o=rb(rb((b2f(row[j+k])-mu)*iv)*f[j+k])*inv_s[j+k];v[n++]=o;m=fmaxf(m,fabsf(o));}\n"
" for(int s=16;s;s>>=1)m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,s));if(l==0)amx[w]=m;__syncthreads();\n"
" m=fmaxf(fmaxf(amx[0],amx[1]),fmaxf(amx[2],amx[3]));float r=m>0.f?127.f/m:0.f;\n"
" signed char*o=q+(size_t)t*D;n=0;for(int j=i*4;j<D;j+=512)for(int k=0;k<4;k++){int z=__float2int_rn(v[n++]*r);o[j+k]=(signed char)max(-127,min(127,z));}\n"
" if(i==0)qs[t]=m>0.f?m/127.f:1.f;\n"
"}\n"
/* Per-row SmoothQuant INT8 of a BF16 matrix (row stride ld): q = rint(x*inv_s/scale). */
"__global__ void quant_rows(signed char*q,float*qs,const bf*x,const float*inv_s,int cols,int ld){\n"
" int t=blockIdx.x,i=threadIdx.x;__shared__ float amx[8];const bf*r=x+(size_t)t*ld;float m=0.f;\n"
" for(int j=i;j<cols;j+=256)m=fmaxf(m,fabsf(b2f(r[j])*inv_s[j]));\n"
" for(int s=16;s;s>>=1)m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,s));if(i%32==0)amx[i/32]=m;__syncthreads();\n"
" m=amx[0];for(int k=1;k<8;k++)m=fmaxf(m,amx[k]);float rr=m>0.f?127.f/m:0.f;signed char*o=q+(size_t)t*cols;\n"
" for(int j=i;j<cols;j+=256){int z=__float2int_rn(b2f(r[j])*inv_s[j]*rr);o[j]=(signed char)max(-127,min(127,z));}\n"
" if(i==0)qs[t]=m>0.f?m/127.f:1.f;\n"
"}\n"
/* INT32 accumulator -> BF16 with per-row activation and per-column weight scales. */
"__global__ void dequant(bf*y,const int*acc,const float*xs,const float*ws,int rows,int cols,int ldy){size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=(size_t)rows*cols)return;size_t t=i/cols,o=i%cols;y[t*ldy+o]=f2b((float)acc[i]*xs[t]*ws[o]);}\n"
/* Per-head QK RMSNorm + RoPE from a fused [N,3D] QKV row (harness
 * q21_exact_qk_rope_kernel rounding). Writes Q to q[N,D] and K/V to kv rows
 * starting at kv_row (stride D). table holds F32 cos,sin per token pair. */
"__global__ void qk_norm_rope(bf*q,bf*k,bf*v,const bf*qkv,const float*qw,const float*kw,const float*table,int heads,int kv_row){\n"
" int token=blockIdx.x,head=blockIdx.y,lane=threadIdx.x,D=heads*128;\n"
" const bf*src=qkv+(size_t)token*3*D+head*128;__shared__ float qs[32],ks[32];float sq=0,sk=0;\n"
" for(int e=0;e<4;e++){float x=b2f(src[lane*4+e]);sq+=x*x;x=b2f(src[D+lane*4+e]);sk+=x*x;}\n"
" qs[lane]=sq;ks[lane]=sk;__syncthreads();\n"
" for(int s=16;s;s>>=1){if(lane<s){qs[lane]+=qs[lane+s];ks[lane]+=ks[lane+s];}__syncthreads();}\n"
" float qi=rsqrtf(qs[0]/128.0f+1e-6f),ki=rsqrtf(ks[0]/128.0f+1e-6f);\n"
" bf*qo=q+(size_t)token*D+head*128;size_t kr=((size_t)kv_row+token)*D+head*128;bf*ko=k+kr;bf*vo=v+kr;\n"
" for(int p=0;p<4;p+=2){int j=lane*4+p;\n"
"  float q0=rb(rb(b2f(src[j])*qi)*qw[j]),q1=rb(rb(b2f(src[j+1])*qi)*qw[j+1]);\n"
"  float k0=rb(rb(b2f(src[D+j])*ki)*kw[j]),k1=rb(rb(b2f(src[D+j+1])*ki)*kw[j+1]);\n"
"  float c=table[(size_t)token*128+j],s=table[(size_t)token*128+j+1];\n"
"  qo[j]=f2b(__fmaf_rn(q0,c,-__fmul_rn(q1,s)));qo[j+1]=f2b(__fmaf_rn(q1,c,__fmul_rn(q0,s)));\n"
"  ko[j]=f2b(__fmaf_rn(k0,c,-__fmul_rn(k1,s)));ko[j+1]=f2b(__fmaf_rn(k1,c,__fmul_rn(k0,s)));}\n"
" for(int e=0;e<4;e++)vo[lane*4+e]=src[2*D+lane*4+e];\n"
"}\n"
/* SwiGLU in place: row [gate(F) | proj(F)] -> row[0..F) = rb(rb(silu(gate))*proj). */
"__global__ void swiglu(bf*x,int rows,int F){size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=(size_t)rows*F)return;size_t t=i/F,j=i%F;bf*r=x+t*2*F;float a=b2f(r[j]);float z=rb(a/(1.f+expf(-a)));r[j]=f2b(z*b2f(r[F+j]));}\n"
/* Classifier-free guidance and the BF16 FlowMatch Euler update, matching the
 * harness host code. pred/neg are BF16 [n], sample F32 [n]; latent receives
 * the BF16 copy used as the next img_in input. */
"__global__ void euler(float*sample,bf*latent,const bf*pred,const bf*neg,int n,float scale,float dt,int cfg){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;float p=b2f(pred[i]);if(cfg){float u=b2f(neg[i]);p=rb(u+rb(scale*rb(p-u)));}float s=rb(sample[i]+rb(dt*rb(p)));sample[i]=s;latent[i]=f2b(s);}\n"
/* CFG combine only (single-step prediction dumps). */
"__global__ void cfg_combine(bf*pred,const bf*neg,int n,float scale){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float p=b2f(pred[i]),u=b2f(neg[i]);pred[i]=f2b(rb(u+rb(scale*rb(p-u))));}}\n"
/* Calibration: out[c] = max(out[c], max_r |x[r,c]|) for SmoothQuant; out is
 * non-negative F32 so its bits order like int32. */
"__global__ void colmax(float*out,const bf*x,int rows,int cols,int ld){int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=cols)return;float m=0.f;for(int r=0;r<rows;r++)m=fmaxf(m,fabsf(b2f(x[(size_t)r*ld+c])));atomicMax((int*)&out[c],__float_as_int(m));}\n"
/* Diagnostic: sampled sum of 32-bit words (every stride-th word). */
"__global__ void checksum(unsigned long long*out,const unsigned*p,unsigned long long words,unsigned stride){unsigned long long s=0;for(unsigned long long i=(unsigned long long)threadIdx.x*stride;i<words;i+=(unsigned long long)blockDim.x*stride)s+=p[i];atomicAdd(out,s);}\n"
"}\n";
#endif
