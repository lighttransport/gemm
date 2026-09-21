/* Vector-four, four-warp Welford diagnostic. Match the reduction topology
 * used by PyTorch CUDA vectorized layer norm; keep BF16 modulation boundaries. */
#ifndef QIMG21_NORM_VECTOR_KERNELS_H
#define QIMG21_NORM_VECTOR_KERNELS_H
static const char *q21_norm_vector_src =
"extern \"C\" {\n"
"struct Stat {float m,v,n;};\n"
"__device__ Stat combine(Stat b,Stat a){float n=a.n+b.n,d=b.m-a.m,r=__frcp_rn(n),na=a.n*r,nb=b.n*r;Stat z;z.n=n;z.m=fmaf(na,a.m,nb*b.m);z.v=fmaf(d*d*a.n,nb,a.v+b.v);return z;}\n"
"__device__ float rb(float x){unsigned u=__float_as_uint(x);return __uint_as_float((u+0x7fff+((u>>16)&1))&0xffff0000);}\n"
"__device__ void norm_impl(float*y,const float*x,const float*m,int N,int D,int prefix,int which,int stride){\n"
" int t=blockIdx.x,i=threadIdx.x,l=i%32,w=i/32;__shared__ float mean[4],var[4],count[4];Stat a={0,0,0};\n"
" for(int j=i*4;j<D;j+=512)for(int k=0;k<4;k++){float v=x[t*D+j+k],d=v-a.m;float nn=a.n+1.f,nm=fmaf(d,__frcp_rn(nn),a.m);a.v=fmaf(d,v-nm,a.v);a.m=nm;a.n=nn;}\n"
" for(int s=16;s;s>>=1){Stat b={__shfl_down_sync(0xffffffff,a.m,s),__shfl_down_sync(0xffffffff,a.v,s),__shfl_down_sync(0xffffffff,a.n,s)};a=combine(a,b);}\n"
" if(l==0){mean[w]=a.m;var[w]=a.v;count[w]=a.n;}__syncthreads();\n"
" for(int s=2;s;s>>=1){if(l==0&&w<s){Stat b={mean[w+s],var[w+s],count[w+s]};a=combine(a,b);mean[w]=a.m;var[w]=a.v;count[w]=a.n;}__syncthreads();}\n"
" float mu=mean[0],iv=rsqrtf(var[0]/D+1e-6f);int base=(t<prefix?1:0)*stride+which*2*D;\n"
" for(int j=i;j<D;j+=128)y[t*D+j]=rb(rb((x[t*D+j]-mu)*iv)*rb(1.f+m[base+j]));\n"
"}\n"
"__global__ void mod_ln_vector(float*y,const float*x,const float*m,int N,int D,int prefix,int which){norm_impl(y,x,m,N,D,prefix,which,4*D);}\n"
"__global__ void final_ln_vector(float*y,const float*x,const float*m,int N,int D,int prefix){norm_impl(y,x,m,N,D,prefix,0,D);}\n"
"}\n";
#endif
