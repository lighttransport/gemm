// Full-attention per-core benchmark (QK^T + AV), A64FX SVE. K/V sized to the
// real 96 KB/head (L2-resident) to mirror in-situ. Compares the standalone
// per-core rate to the in-situ ~36% to locate the penalty (kernel vs in-situ).
//   fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -std=c11 -D_GNU_SOURCE \
//       tools/bench_attn.c -lm -o tools/bench_attn
#include <arm_sve.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static inline double tnow(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}

static inline void qk_vert_8q_48k(const float *qh,const float *KT,int hd,int nps,float scale,float *dst,int qs){
 const svbool_t pg=svptrue_b32();const int VL=(int)svcntw();
 svfloat32_t a00=svdup_f32(0),a01=svdup_f32(0),a02=svdup_f32(0),a10=svdup_f32(0),a11=svdup_f32(0),a12=svdup_f32(0);
 svfloat32_t a20=svdup_f32(0),a21=svdup_f32(0),a22=svdup_f32(0),a30=svdup_f32(0),a31=svdup_f32(0),a32=svdup_f32(0);
 svfloat32_t a40=svdup_f32(0),a41=svdup_f32(0),a42=svdup_f32(0),a50=svdup_f32(0),a51=svdup_f32(0),a52=svdup_f32(0);
 svfloat32_t a60=svdup_f32(0),a61=svdup_f32(0),a62=svdup_f32(0),a70=svdup_f32(0),a71=svdup_f32(0),a72=svdup_f32(0);
 for(int d=0;d<hd;d++){const float*kp=KT+(size_t)d*nps;
  svfloat32_t v0=svld1_f32(pg,kp),v1=svld1_f32(pg,kp+VL),v2=svld1_f32(pg,kp+2*VL);
  float q0=qh[0*hd+d],q1=qh[1*hd+d],q2=qh[2*hd+d],q3=qh[3*hd+d],q4=qh[4*hd+d],q5=qh[5*hd+d],q6=qh[6*hd+d],q7=qh[7*hd+d];
  a00=svmla_n_f32_x(pg,a00,v0,q0);a01=svmla_n_f32_x(pg,a01,v1,q0);a02=svmla_n_f32_x(pg,a02,v2,q0);
  a10=svmla_n_f32_x(pg,a10,v0,q1);a11=svmla_n_f32_x(pg,a11,v1,q1);a12=svmla_n_f32_x(pg,a12,v2,q1);
  a20=svmla_n_f32_x(pg,a20,v0,q2);a21=svmla_n_f32_x(pg,a21,v1,q2);a22=svmla_n_f32_x(pg,a22,v2,q2);
  a30=svmla_n_f32_x(pg,a30,v0,q3);a31=svmla_n_f32_x(pg,a31,v1,q3);a32=svmla_n_f32_x(pg,a32,v2,q3);
  a40=svmla_n_f32_x(pg,a40,v0,q4);a41=svmla_n_f32_x(pg,a41,v1,q4);a42=svmla_n_f32_x(pg,a42,v2,q4);
  a50=svmla_n_f32_x(pg,a50,v0,q5);a51=svmla_n_f32_x(pg,a51,v1,q5);a52=svmla_n_f32_x(pg,a52,v2,q5);
  a60=svmla_n_f32_x(pg,a60,v0,q6);a61=svmla_n_f32_x(pg,a61,v1,q6);a62=svmla_n_f32_x(pg,a62,v2,q6);
  a70=svmla_n_f32_x(pg,a70,v0,q7);a71=svmla_n_f32_x(pg,a71,v1,q7);a72=svmla_n_f32_x(pg,a72,v2,q7);}
 svfloat32_t s=svdup_f32(scale);
 svst1_f32(pg,dst+0*qs,svmul_f32_x(pg,a00,s));svst1_f32(pg,dst+0*qs+VL,svmul_f32_x(pg,a01,s));svst1_f32(pg,dst+0*qs+2*VL,svmul_f32_x(pg,a02,s));
 svst1_f32(pg,dst+1*qs,svmul_f32_x(pg,a10,s));svst1_f32(pg,dst+1*qs+VL,svmul_f32_x(pg,a11,s));svst1_f32(pg,dst+1*qs+2*VL,svmul_f32_x(pg,a12,s));
 svst1_f32(pg,dst+2*qs,svmul_f32_x(pg,a20,s));svst1_f32(pg,dst+2*qs+VL,svmul_f32_x(pg,a21,s));svst1_f32(pg,dst+2*qs+2*VL,svmul_f32_x(pg,a22,s));
 svst1_f32(pg,dst+3*qs,svmul_f32_x(pg,a30,s));svst1_f32(pg,dst+3*qs+VL,svmul_f32_x(pg,a31,s));svst1_f32(pg,dst+3*qs+2*VL,svmul_f32_x(pg,a32,s));
 svst1_f32(pg,dst+4*qs,svmul_f32_x(pg,a40,s));svst1_f32(pg,dst+4*qs+VL,svmul_f32_x(pg,a41,s));svst1_f32(pg,dst+4*qs+2*VL,svmul_f32_x(pg,a42,s));
 svst1_f32(pg,dst+5*qs,svmul_f32_x(pg,a50,s));svst1_f32(pg,dst+5*qs+VL,svmul_f32_x(pg,a51,s));svst1_f32(pg,dst+5*qs+2*VL,svmul_f32_x(pg,a52,s));
 svst1_f32(pg,dst+6*qs,svmul_f32_x(pg,a60,s));svst1_f32(pg,dst+6*qs+VL,svmul_f32_x(pg,a61,s));svst1_f32(pg,dst+6*qs+2*VL,svmul_f32_x(pg,a62,s));
 svst1_f32(pg,dst+7*qs,svmul_f32_x(pg,a70,s));svst1_f32(pg,dst+7*qs+VL,svmul_f32_x(pg,a71,s));svst1_f32(pg,dst+7*qs+2*VL,svmul_f32_x(pg,a72,s));
}
static inline void attn_av_4q(const float*att,int aqs,const float*V,int np,int hd,float*out,int oqs){
 const svbool_t pg=svptrue_b32();const int VL=(int)svcntw();
 svfloat32_t a00=svdup_f32(0),a01=svdup_f32(0),a02=svdup_f32(0),a03=svdup_f32(0),a10=svdup_f32(0),a11=svdup_f32(0),a12=svdup_f32(0),a13=svdup_f32(0);
 svfloat32_t a20=svdup_f32(0),a21=svdup_f32(0),a22=svdup_f32(0),a23=svdup_f32(0),a30=svdup_f32(0),a31=svdup_f32(0),a32=svdup_f32(0),a33=svdup_f32(0);
 const float*a0=att;const float*a1=att+aqs;const float*a2=att+2*aqs;const float*a3=att+3*aqs;
 for(int vi=0;vi<np;vi++){const float*vh=V+(size_t)vi*hd;
  svfloat32_t v0=svld1_f32(pg,vh),v1=svld1_f32(pg,vh+VL),v2=svld1_f32(pg,vh+2*VL),v3=svld1_f32(pg,vh+3*VL);
  float w0=a0[vi],w1=a1[vi],w2=a2[vi],w3=a3[vi];
  a00=svmla_n_f32_x(pg,a00,v0,w0);a01=svmla_n_f32_x(pg,a01,v1,w0);a02=svmla_n_f32_x(pg,a02,v2,w0);a03=svmla_n_f32_x(pg,a03,v3,w0);
  a10=svmla_n_f32_x(pg,a10,v0,w1);a11=svmla_n_f32_x(pg,a11,v1,w1);a12=svmla_n_f32_x(pg,a12,v2,w1);a13=svmla_n_f32_x(pg,a13,v3,w1);
  a20=svmla_n_f32_x(pg,a20,v0,w2);a21=svmla_n_f32_x(pg,a21,v1,w2);a22=svmla_n_f32_x(pg,a22,v2,w2);a23=svmla_n_f32_x(pg,a23,v3,w2);
  a30=svmla_n_f32_x(pg,a30,v0,w3);a31=svmla_n_f32_x(pg,a31,v1,w3);a32=svmla_n_f32_x(pg,a32,v2,w3);a33=svmla_n_f32_x(pg,a33,v3,w3);}
 svst1_f32(pg,out,a00);svst1_f32(pg,out+VL,a01);svst1_f32(pg,out+2*VL,a02);svst1_f32(pg,out+3*VL,a03);
 svst1_f32(pg,out+oqs,a10);svst1_f32(pg,out+oqs+VL,a11);svst1_f32(pg,out+oqs+2*VL,a12);svst1_f32(pg,out+oqs+3*VL,a13);
 svst1_f32(pg,out+2*oqs,a20);svst1_f32(pg,out+2*oqs+VL,a21);svst1_f32(pg,out+2*oqs+2*VL,a22);svst1_f32(pg,out+2*oqs+3*VL,a23);
 svst1_f32(pg,out+3*oqs,a30);svst1_f32(pg,out+3*oqs+VL,a31);svst1_f32(pg,out+3*oqs+2*VL,a32);svst1_f32(pg,out+3*oqs+3*VL,a33);
}
int main(void){
 const int hd=64,np=384,Q32=32; // q_tile=32
 float*Q=malloc(32*hd*sizeof(float));float*KT=malloc((size_t)hd*np*sizeof(float));
 float*V=malloc((size_t)np*hd*sizeof(float));float*att=malloc(32*np*sizeof(float));
 float*o=malloc(32*hd*sizeof(float));
 for(int i=0;i<32*hd;i++)Q[i]=((i*7)%13)*0.01f;
 for(int i=0;i<hd*np;i++)KT[i]=((i*5)%11)*0.01f;
 for(int i=0;i<np*hd;i++)V[i]=((i*3)%7)*0.01f;
 // one q-tile (32 queries): 4 q-groups of 8 (QK^T) + 8 q-groups of 4 (AV). FLOPs = 32*np*hd*2*2
 int reps=3000;
 for(int r=0;r<100;r++){ for(int qi=0;qi<Q32;qi+=8) for(int ki=0;ki<np;ki+=48) qk_vert_8q_48k(Q+qi*hd,KT+ki,hd,np,0.125f,att+qi*np+ki,np);
                          for(int qi=0;qi<Q32;qi+=4) attn_av_4q(att+qi*np,np,V,np,hd,o+qi*hd,hd); }
 double t0=tnow();
 for(int r=0;r<reps;r++){ for(int qi=0;qi<Q32;qi+=8) for(int ki=0;ki<np;ki+=48) qk_vert_8q_48k(Q+qi*hd,KT+ki,hd,np,0.125f,att+qi*np+ki,np);
                          for(int qi=0;qi<Q32;qi+=4) attn_av_4q(att+qi*np,np,V,np,hd,o+qi*hd,hd); }
 double dt=tnow()-t0;
 double flops=(double)reps*Q32*np*hd*2*2.0; // QK^T + AV
 printf("VL=%d K=%zuKB V=%zuKB\n",(int)svcntw(),(size_t)hd*np*4/1024,(size_t)np*hd*4/1024);
 printf("full attn (QK^T+AV, 32q): %.1f us, %.1f GFLOP -> %.1f GFLOP/s per core\n",dt*1e6,flops*1e-9,flops/dt*1e-9);
 return 0;
}
