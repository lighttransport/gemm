#ifndef QIMG21_ROPE_TABLE_KERNELS_H
#define QIMG21_ROPE_TABLE_KERNELS_H
/* Host-generated complex frequencies avoid CUDA fast-trig frequency drift. */
static const char *q21_rope_table_src =
"extern \"C\" {\n"
"__device__ float q21_rne(float x){unsigned u=__float_as_uint(x);return __uint_as_float((u+0x7fff+((u>>16)&1))&0xffff0000);}\n"
"__global__ void qk_rope_table(float*q,float*k,const float*qw,const float*kw,int N,int D,int nh,int hd,int prefix,int ih,int iw,const float*table){int t=blockIdx.x,h=blockIdx.y,j=threadIdx.x;if(t>=N||h>=nh)return;__shared__ float sq6[128],sk6[128];float aq=0,ak=0;for(int z=j;z<hd;z+=blockDim.x){float v=q[t*D+h*hd+z];aq+=v*v;v=k[t*D+h*hd+z];ak+=v*v;}sq6[j]=aq;sk6[j]=ak;__syncthreads();for(int z=64;z;z>>=1){if(j<z){sq6[j]+=sq6[j+z];sk6[j]+=sk6[j+z];}__syncthreads();}if(j&1)return;float iq=rsqrtf(sq6[0]/hd+1e-6f),ik=rsqrtf(sk6[0]/hd+1e-6f);int axis,off,pos;if(j<16){axis=16;off=0;pos=t<prefix?t:prefix;}else if(j<72){axis=56;off=16;pos=t<prefix?t:-(ih-ih/2)+(t-prefix)/iw;}else{axis=56;off=72;pos=t<prefix?t:-(iw-iw/2)+(t-prefix)%iw;}int pair=(j-off)&~1;float c=table[t*128+off+pair],sn=table[t*128+off+pair+1];int d0=h*hd+off+pair,d1=d0+1;float x0=q21_rne(q[t*D+d0]*iq)*qw[off+pair],x1=q21_rne(q[t*D+d1]*iq)*qw[off+pair+1];float y0=q21_rne(k[t*D+d0]*ik)*kw[off+pair],y1=q21_rne(k[t*D+d1]*ik)*kw[off+pair+1];unsigned u=__float_as_uint(x0),l=(u>>16)&1u;x0=__uint_as_float((u+0x7fffu+l)&0xffff0000u);u=__float_as_uint(x1);l=(u>>16)&1u;x1=__uint_as_float((u+0x7fffu+l)&0xffff0000u);u=__float_as_uint(y0);l=(u>>16)&1u;y0=__uint_as_float((u+0x7fffu+l)&0xffff0000u);u=__float_as_uint(y1);l=(u>>16)&1u;y1=__uint_as_float((u+0x7fffu+l)&0xffff0000u);q[t*D+d0]=x0*c-x1*sn;q[t*D+d1]=x0*sn+x1*c;k[t*D+d0]=y0*c-y1*sn;k[t*D+d1]=y0*sn+y1*c;}\n"
/* PyTorch mean(-1) on 128 contiguous F32 squares uses 32 lanes,
 * each combining four adjacent values, before the descending warp tree. */
"__global__ void qk_rope_table_vector4(float*q,float*k,const float*qw,const float*kw,int N,int D,int nh,int hd,int prefix,int ih,int iw,const float*table){\n"
" int t=blockIdx.x,h=blockIdx.y,j=threadIdx.x;if(t>=N||h>=nh)return;\n"
" __shared__ float sq[32],sk[32];int base=t*D+h*hd;\n"
" if(j<32){float aq=0,ak=0;for(int z=0;z<4;z++){float v=q[base+4*j+z];aq=__fadd_rn(aq,__fmul_rn(v,v));v=k[base+4*j+z];ak=__fadd_rn(ak,__fmul_rn(v,v));}sq[j]=aq;sk[j]=ak;}\n"
" __syncthreads();for(int z=16;z;z>>=1){if(j<z){sq[j]+=sq[j+z];sk[j]+=sk[j+z];}__syncthreads();}\n"
" if(j&1)return;float iq=rsqrtf(sq[0]/128.f+1e-6f),ik=rsqrtf(sk[0]/128.f+1e-6f);\n"
" float x0=q21_rne(q21_rne(q[base+j]*iq)*qw[j]),x1=q21_rne(q21_rne(q[base+j+1]*iq)*qw[j+1]);\n"
" float y0=q21_rne(q21_rne(k[base+j]*ik)*kw[j]),y1=q21_rne(q21_rne(k[base+j+1]*ik)*kw[j+1]);\n"
" float c=table[t*128+j],s=table[t*128+j+1];\n"
" q[base+j]=x0*c-x1*s;q[base+j+1]=x0*s+x1*c;k[base+j]=y0*c-y1*s;k[base+j+1]=y0*s+y1*c;\n"
"}\n"
"}\n"
"\n";
#endif
