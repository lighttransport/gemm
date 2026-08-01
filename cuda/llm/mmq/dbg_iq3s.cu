// Element-wise IQ3_S decode debug: host oracle (dequantize_row_iq3_s style) vs a
// device kernel mirroring vec_dot_iq3_s's exact reads.  One block (256 elems).
#include "mmvq.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>
extern "C" void ggml_abort(const char*f,int l,const char*fmt,...){fprintf(stderr,"abort %s:%d\n",f,l);abort();}
int ggml_cuda_get_device(){int d=0;cudaGetDevice(&d);return d;}
void ggml_cuda_error(const char*s,const char*fn,const char*f,int l,const char*m){fprintf(stderr,"err\n");abort();}
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

static float h2f(uint16_t h){uint32_t s=(h>>15)&1,e=(h>>10)&0x1f,m=h&0x3ff,o;
 if(e==0){if(m==0)o=s<<31;else{e=113;while(!(m&0x400)){m<<=1;e--;}m&=0x3ff;o=(s<<31)|(e<<23)|(m<<13);}}
 else if(e==0x1f)o=(s<<31)|(0xff<<23)|(m<<13);else o=(s<<31)|((e-15+127)<<23)|(m<<13);
 float f;memcpy(&f,&o,4);return f;}
static uint16_t f2h(float f){uint32_t b;memcpy(&b,&f,4);uint32_t sg=(b>>16)&0x8000,e=(b>>23)&0xff,m=b&0x7fffff;
 if(e<=112)return sg;if(e>=143)return sg|0x7c00;return sg|((e-112)<<10)|(m>>13);}

#define BSZ 110
static uint32_t hg[512];
static const uint8_t kmask[8]={1,2,4,8,16,32,64,128};
static void host_oracle(const uint8_t*bp,float*y){
 float d=h2f(*(const uint16_t*)bp);
 const uint8_t*qs=bp+2,*qh=bp+66,*signs=bp+74,*sc=bp+106;
 for(int ib=0;ib<8;ib+=2){
  float db1=d*(1+2*(sc[ib/2]&0xf)),db2=d*(1+2*(sc[ib/2]>>4));
  for(int l=0;l<4;l++){const uint8_t*g1=(const uint8_t*)(hg+(qs[2*l]|((qh[0]<<(8-2*l))&256)));
   const uint8_t*g2=(const uint8_t*)(hg+(qs[2*l+1]|((qh[0]<<(7-2*l))&256)));
   for(int j=0;j<4;j++){y[j]=db1*g1[j]*((signs[l]&kmask[j])?-1.f:1.f);y[j+4]=db1*g2[j]*((signs[l]&kmask[j+4])?-1.f:1.f);}y+=8;}
  qs+=8;signs+=4;
  for(int l=0;l<4;l++){const uint8_t*g1=(const uint8_t*)(hg+(qs[2*l]|((qh[1]<<(8-2*l))&256)));
   const uint8_t*g2=(const uint8_t*)(hg+(qs[2*l+1]|((qh[1]<<(7-2*l))&256)));
   for(int j=0;j<4;j++){y[j]=db2*g1[j]*((signs[l]&kmask[j])?-1.f:1.f);y[j+4]=db2*g2[j]*((signs[l]&kmask[j+4])?-1.f:1.f);}y+=8;}
  qh+=2;qs+=8;signs+=4;}
}
// device decode mirroring vec_dot_iq3_s reads; out[256] in superblock element order
__global__ void dev_decode(const void*vbq,float*out){
 const block_iq3_s*bq3=(const block_iq3_s*)vbq;
 float d=__half2float(bq3->d);
 for(int iqs=0;iqs<16;iqs+=2){
  int s=iqs/2;
  int2 qsp=make_int2(get_int_b2(bq3->qs,iqs+0),get_int_b2(bq3->qs,iqs+1));
  const uint8_t*qs=(const uint8_t*)&qsp;
  const int qh=bq3->qh[iqs/2];
  int sp=get_int_b2(bq3->signs,iqs/2);
  const uint8_t*s8=(const uint8_t*)&sp;
  int scale=1+2*((bq3->scales[iqs/4]>>((iqs<<1)&0x04))&0x0F);
  float db=d*scale;
  for(int l0=0;l0<8;l0+=2){
   int2 g=make_int2(iq3s_grid[qs[l0]|((qh<<(8-l0))&0x100)],iq3s_grid[qs[l0+1]|((qh<<(7-l0))&0x100)]);
   int s0=__vcmpne4(((s8[l0/2]&0x03)<<7)|((s8[l0/2]&0x0C)<<21),0);
   int s1=__vcmpne4(((s8[l0/2]&0x30)<<3)|((s8[l0/2]&0xC0)<<17),0);
   int gl=__vsub4(g.x^s0,s0), gh=__vsub4(g.y^s1,s1);
   signed char*glb=(signed char*)&gl,*ghb=(signed char*)&gh;
   for(int j=0;j<4;j++){out[s*32+(l0/2)*8+j]=db*glb[j]; out[s*32+(l0/2)*8+4+j]=db*ghb[j];}
  }
 }
}
// quantize X (256 floats) -> 8 block_q8_1
__global__ void qk(const float*x,void*vy){
 int wid=threadIdx.x/32, lane=threadIdx.x%32;
 const float xi=x[wid*32+lane]; float amax=fabsf(xi),sum=xi;
 amax=warp_reduce_max<32>(amax); sum=warp_reduce_sum<32>(sum);
 float d=amax/127.0f; int8_t q=amax==0?0:roundf(xi/d);
 block_q8_1*y=(block_q8_1*)vy; y[wid].qs[lane]=q; if(lane==0)y[wid].ds=make_half2(d,sum);
}
// call the REAL vec_dot_iq3_s_q8_1, summed over the 8 sub-blocks of block 0
__global__ void dev_vecdot(const void*vx,const block_q8_1*y,float*out){
 int kbx=0;
 for(int iqs=0;iqs<16;iqs+=2) out[iqs/2]=vec_dot_iq3_s_q8_1(vx,y,kbx,iqs);
}
// copy of vec_dot_iq3_s_q8_1 but with BY-VALUE kbx/iqs (vs const int& in the real one)
static __device__ __forceinline__ float my_vd_iq3s(
    const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, int kbx, int iqs){
    const block_iq3_s * bq3 = (const block_iq3_s *) vbq + kbx;
    const int2 qs_packed = make_int2(get_int_b2(bq3->qs, iqs+0), get_int_b2(bq3->qs, iqs+1));
    const uint8_t * qs = (const uint8_t *) &qs_packed;
    const int qh = bq3->qh[iqs/2];
    const int signs_packed_32 = get_int_b2(bq3->signs, iqs/2);
    const uint8_t * signs_packed_8 = (const uint8_t *) &signs_packed_32;
    int sumi = 0;
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(
            iq3s_grid[qs[l0+0] | ((qh << (8-l0)) & 0x100)],
            iq3s_grid[qs[l0+1] | ((qh << (7-l0)) & 0x100)]);
        const int signs0 = __vcmpne4(((signs_packed_8[l0/2] & 0x03) << 7) | ((signs_packed_8[l0/2] & 0x0C) << 21), 0);
        const int signs1 = __vcmpne4(((signs_packed_8[l0/2] & 0x30) << 3) | ((signs_packed_8[l0/2] & 0xC0) << 17), 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0+0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0+1);
        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }
    sumi *= 1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}
__global__ void dev_byval(const void*vx,const block_q8_1*y,float*out){
 for(int iqs=0;iqs<16;iqs+=2) out[iqs/2]=my_vd_iq3s(vx,y,0,iqs);
}
// MY inline full vec_dot (dp4a) per sub-block, same activation
__global__ void dev_inline(const void*vx,const block_q8_1*y,float*out){
 const block_iq3_s*bq3=(const block_iq3_s*)vx;
 for(int iqs=0;iqs<16;iqs+=2){
  int2 qsp=make_int2(get_int_b2(bq3->qs,iqs+0),get_int_b2(bq3->qs,iqs+1));
  const uint8_t*qs=(const uint8_t*)&qsp;
  const int qh=bq3->qh[iqs/2];
  int sp=get_int_b2(bq3->signs,iqs/2); const uint8_t*s8=(const uint8_t*)&sp;
  int sumi=0;
  for(int l0=0;l0<8;l0+=2){
   int2 g=make_int2(iq3s_grid[qs[l0]|((qh<<(8-l0))&0x100)],iq3s_grid[qs[l0+1]|((qh<<(7-l0))&0x100)]);
   int s0=__vcmpne4(((s8[l0/2]&0x03)<<7)|((s8[l0/2]&0x0C)<<21),0);
   int s1=__vcmpne4(((s8[l0/2]&0x30)<<3)|((s8[l0/2]&0xC0)<<17),0);
   int gl=__vsub4(g.x^s0,s0), gh=__vsub4(g.y^s1,s1);
   int u0=get_int_b4(y[iqs/2].qs,l0+0), u1=get_int_b4(y[iqs/2].qs,l0+1);
   sumi=__dp4a(gl,u0,sumi); sumi=__dp4a(gh,u1,sumi);
  }
  sumi*=1+2*((bq3->scales[iqs/4]>>((iqs<<1)&0x04))&0x0F);
  float d=__half2float(bq3->d)*__low2float(y[iqs/2].ds);
  out[iqs/2]=d*sumi;
 }
}
int main(){
 CK(cudaMemcpyFromSymbol(hg,iq3s_grid,sizeof(hg)));
 uint8_t W[BSZ]; srand(7); for(int i=0;i<BSZ;i++)W[i]=rand()&0xff;
 *(uint16_t*)W=f2h(0.03f);
 float wf[256]; host_oracle(W,wf);
 char*dW; float*dOut; CK(cudaMalloc(&dW,BSZ)); CK(cudaMalloc(&dOut,256*4));
 CK(cudaMemcpy(dW,W,BSZ,cudaMemcpyHostToDevice));
 dev_decode<<<1,1>>>(dW,dOut); CK(cudaDeviceSynchronize());
 float wd[256]; CK(cudaMemcpy(wd,dOut,256*4,cudaMemcpyDeviceToHost));
 int nbad=0; double nu=0,de=0;
 for(int i=0;i<256;i++){double dd=wf[i]-wd[i];nu+=dd*dd;de+=(double)wf[i]*wf[i];
  if(fabs(wf[i]-wd[i])>1e-3){if(nbad<20)printf("  [%3d] host=%.4f dev=%.4f  (sub=%d off=%d)\n",i,wf[i],wd[i],i/32,i%32);nbad++;}}
 printf("decode rel_L2(host-mirror vs dev-mirror)=%.6f  bad=%d/256\n",sqrt(nu/(de+1e-30)),nbad);

 // REAL vec_dot test: activation = all ones -> result == sum(weights)
 float Xones[256]; for(int i=0;i<256;i++)Xones[i]=1.0f;
 double host_sum=0; for(int i=0;i<256;i++)host_sum+=wf[i];
 float*dX; void*dY; CK(cudaMalloc(&dX,256*4)); CK(cudaMalloc(&dY,8*sizeof(block_q8_1)));
 CK(cudaMemcpy(dX,Xones,256*4,cudaMemcpyHostToDevice));
 qk<<<1,256>>>(dX,dY); CK(cudaDeviceSynchronize());
 dev_vecdot<<<1,1>>>(dW,(block_q8_1*)dY,dOut); CK(cudaDeviceSynchronize());
 float vd8[8]; CK(cudaMemcpy(vd8,dOut,8*4,cudaMemcpyDeviceToHost));
 dev_inline<<<1,1>>>(dW,(block_q8_1*)dY,dOut); CK(cudaDeviceSynchronize());
 float in8[8]; CK(cudaMemcpy(in8,dOut,8*4,cudaMemcpyDeviceToHost));
 dev_byval<<<1,1>>>(dW,(block_q8_1*)dY,dOut); CK(cudaDeviceSynchronize());
 float bv8[8]; CK(cudaMemcpy(bv8,dOut,8*4,cudaMemcpyDeviceToHost));
 double totbv=0; for(int s=0;s<8;s++) totbv+=bv8[s];
 printf("BY-VALUE copy sum=%.4f (oracle=%.4f)\n", totbv, host_sum);
 printf("per-sub-block: REAL vec_dot | MY inline dp4a | oracle sum(wf[s*32:+32]):\n");
 double tot=0,toti=0;
 for(int s=0;s<8;s++){double osum=0; for(int k=0;k<32;k++)osum+=wf[s*32+k]; tot+=vd8[s]; toti+=in8[s];
   printf("  sub %d: real=%.4f  inline=%.4f  oracle=%.4f  %s\n",s,vd8[s],in8[s],osum,
     fabs(vd8[s]-osum)>0.5?"<-- REAL DIFF":"");}
 printf("REAL sum=%.4f  INLINE sum=%.4f  oracle=%.4f\n",tot,toti,host_sum);
 return 0;
}
