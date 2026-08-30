#define _GNU_SOURCE
#define Q38FN_TP_BLOB_IMPLEMENTATION
#define Q38FN_TP_RUNTIME_IMPLEMENTATION
#include "../common/q38fn_tp_runtime.h"
#include <math.h>
#include <stdio.h>

static void sum_stub(float *v,int n,void *p){(void)v;(void)n;(void)p;}
static void argmax_stub(float *v,int *i,void *p){(void)v;(void)i;(void)p;}
static void recurrent_reference(float*state,int heads,const float*qq,const float*kk,
 const float*v,const float*dec,const float*beta,float*core){
 for(int h=0;h<heads;h++){float del[128];double mv[128]={0},ov[128]={0};
  float*mat=state+(size_t)h*128*128;
  for(int i=0;i<128;i++)for(int j=0;j<128;j++){mat[(size_t)i*128+j]*=dec[h];mv[j]+=(double)mat[(size_t)i*128+j]*kk[(size_t)h*128+i];}
  for(int j=0;j<128;j++)del[j]=(v[(size_t)h*128+j]-(float)mv[j])*beta[h];
  for(int i=0;i<128;i++)for(int j=0;j<128;j++)mat[(size_t)i*128+j]+=kk[(size_t)h*128+i]*del[j];
  for(int i=0;i<128;i++)for(int j=0;j<128;j++)ov[j]+=(double)mat[(size_t)i*128+j]*qq[(size_t)h*128+i];
  for(int j=0;j<128;j++)core[(size_t)h*128+j]=(float)ov[j];
 }}
int main(void)
{
    q38fn_tp_model model;memset(&model,0,sizeof(model));model.sum=sum_stub;model.argmax=argmax_stub;
    uint16_t w[11*32];float x[32],got[11],want[11];
    for(int i=0;i<32;i++)x[i]=(float)(i-9)*0.03125f;
    for(int r=0;r<11;r++)for(int c=0;c<32;c++){
        float v=(float)((r*7+c)%19-9)*0.0625f;uint32_t bits;memcpy(&bits,&v,4);w[r*32+c]=(uint16_t)(bits>>16);
    }
    qtp_mv_rows(w,x,11,32,got);
    for(int r=0;r<11;r++){want[r]=qtp_dot(w+r*32,x,32);if(fabsf(got[r]-want[r])>1e-4f)return 1;}
    q38fn_q5_block q5[11];q38fn_tp_blob_entry entry={0};entry.q5_data=q5;
    if(q38fn_q5_quantize_bf16(q5,w,11,32)||qtp_entry_mv(&entry,x,11,32,got))return 1;
    double error2=0,want2=0;
    for(int r=0;r<11;r++){double d=got[r]-want[r];error2+=d*d;want2+=(double)want[r]*want[r];if(fabsf(got[r]-qtp_entry_row_dot(&entry,(size_t)r,x,32))>1e-5f)return 1;}
    if(sqrt(error2/want2)>0.15)return 1;
    enum{H=4,N=H*128,S=H*128*128};float *state=malloc(S*4),*reference=malloc(S*4);
    float qq[N],kk[N],v[N],dec[H],beta[H],actual[N],expected[N];if(!state||!reference)return 1;
    for(int i=0;i<S;i++)state[i]=(float)((i*17)%101-50)*1e-5f;
    memcpy(reference,state,S*4);
    for(int i=0;i<N;i++){qq[i]=(float)((i*7)%31-15)*0.002f;kk[i]=(float)((i*11)%37-18)*0.002f;v[i]=(float)((i*13)%41-20)*0.003f;}
    for(int h=0;h<H;h++){dec[h]=0.91f+0.01f*h;beta[h]=0.35f+0.07f*h;}
    for(int step=0;step<3;step++){qtp_delta_recurrent(state,0,H,qq,kk,v,dec,beta,actual);recurrent_reference(reference,H,qq,kk,v,dec,beta,expected);if(memcmp(state,reference,S*4)||memcmp(actual,expected,N*4))return 1;for(int i=0;i<N;i++)v[i]+=0.00001f*(float)((i+step)%5-2);}
    free(reference);free(state);
    if(!model.sum||!model.argmax)return 1;
    puts("Q38FN_TP_RUNTIME_TEST ok");return 0;
}
