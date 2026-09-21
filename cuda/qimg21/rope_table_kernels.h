#ifndef QIMG21_ROPE_TABLE_KERNELS_H
#define QIMG21_ROPE_TABLE_KERNELS_H
/* Host-generated complex frequencies avoid CUDA fast-trig frequency drift. */
static const char *q21_rope_table_src =
"extern \"C\" {\n"
"__device__ float q21_rne(float x){unsigned u=__float_as_uint(x);return __uint_as_float((u+0x7fff+((u>>16)&1))&0xffff0000);}\n"
"__global__ void qk_rope_table(float*q,float*k,const float*qw,const float*kw,int N,int D,int nh,int hd,int prefix,int ih,int iw,const float*table){int t=blockIdx.x,h=blockIdx.y,j=threadIdx.x;if(t>=N||h>=nh)return;__shared__ float sq6[128],sk6[128];float aq=0,ak=0;for(int z=j;z<hd;z+=blockDim.x){float v=q[t*D+h*hd+z];aq+=v*v;v=k[t*D+h*hd+z];ak+=v*v;}sq6[j]=aq;sk6[j]=ak;__syncthreads();for(int z=64;z;z>>=1){if(j<z){sq6[j]+=sq6[j+z];sk6[j]+=sk6[j+z];}__syncthreads();}if(j&1)return;float iq=rsqrtf(sq6[0]/hd+1e-6f),ik=rsqrtf(sk6[0]/hd+1e-6f);int axis,off,pos;if(j<16){axis=16;off=0;pos=t<prefix?t:prefix;}else if(j<72){axis=56;off=16;pos=t<prefix?t:-(ih-ih/2)+(t-prefix)/iw;}else{axis=56;off=72;pos=t<prefix?t:-(iw-iw/2)+(t-prefix)%iw;}int pair=(j-off)&~1;float c=table[t*128+off+pair],sn=table[t*128+off+pair+1];int d0=h*hd+off+pair,d1=d0+1;float x0=q21_rne(q[t*D+d0]*iq)*qw[off+pair],x1=q21_rne(q[t*D+d1]*iq)*qw[off+pair+1];float y0=q21_rne(k[t*D+d0]*ik)*kw[off+pair],y1=q21_rne(k[t*D+d1]*ik)*kw[off+pair+1];unsigned u=__float_as_uint(x0),l=(u>>16)&1u;x0=__uint_as_float((u+0x7fffu+l)&0xffff0000u);u=__float_as_uint(x1);l=(u>>16)&1u;x1=__uint_as_float((u+0x7fffu+l)&0xffff0000u);u=__float_as_uint(y0);l=(u>>16)&1u;y0=__uint_as_float((u+0x7fffu+l)&0xffff0000u);u=__float_as_uint(y1);l=(u>>16)&1u;y1=__uint_as_float((u+0x7fffu+l)&0xffff0000u);q[t*D+d0]=x0*c-x1*sn;q[t*D+d1]=x0*sn+x1*c;k[t*D+d0]=y0*c-y1*sn;k[t*D+d1]=y0*sn+y1*c;}\n"
"}\n"
"\n";
#endif
