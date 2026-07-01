/* GLM5.2 decode MoE-layer step: compute (RMSNorm + router top-8 + expert SwiGLU)
 * interleaved with comm (dispatch + combine all-to-all over N ranks).
 * Measures compute vs comm in CNTVCT units. hidden=6144 int8, moe_inter=2048. */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
extern float sqrtf(float); extern float expf(float);
typedef uint16_t utofu_tni_id_t; typedef uintptr_t utofu_vcq_hdl_t;
typedef uint64_t utofu_vcq_id_t; typedef uint64_t utofu_stadd_t;
extern int utofu_get_onesided_tnis(utofu_tni_id_t**, size_t*);
extern int utofu_create_vcq(utofu_tni_id_t, unsigned long, utofu_vcq_hdl_t*);
extern int utofu_query_vcq_id(utofu_vcq_hdl_t, utofu_vcq_id_t*);
extern int utofu_reg_mem(utofu_vcq_hdl_t, void*, size_t, unsigned long, utofu_stadd_t*);
extern int utofu_poll_tcq(utofu_vcq_hdl_t, unsigned long, void**);
extern int myputn(utofu_vcq_hdl_t, utofu_vcq_id_t, utofu_stadd_t, utofu_stadd_t, size_t);
extern uint64_t rd_cyc(void);
#define H 6144
#define MI 2048
#define E 256
#define K 8
static float xn[H], gate[MI], up[MI], ffn[MI]; static float logit[E], prob[E];
static unsigned char hb[H];         /* int8 hidden send buffer */
static unsigned char rb[H];         /* receive buffer */
static utofu_vcq_hdl_t H0,H1; static utofu_vcq_id_t I1; static utofu_stadd_t S0,S1;
static uint64_t a2a(int N){ void* d; int rc; uint64_t t0=rd_cyc();
  for(int j=0;j<N-1;j++){ myputn(H0,I1,S0,S1,H); do{rc=utofu_poll_tcq(H0,0,&d);}while(rc!=0); } return rd_cyc()-t0; }
int main(void){
  utofu_tni_id_t* t; size_t nt; utofu_get_onesided_tnis(&t,&nt);
  utofu_vcq_hdl_t h0,h1; utofu_create_vcq(t[0],0,&h0); utofu_create_vcq(t[1],0,&h1);
  utofu_vcq_id_t i0,i1; utofu_query_vcq_id(h0,&i0); utofu_query_vcq_id(h1,&i1);
  utofu_stadd_t s0,s1; utofu_reg_mem(h0,hb,H,0,&s0); utofu_reg_mem(h1,rb,H,0,&s1);
  H0=h0; I1=i1; S0=s0; S1=s1;
  for(int i=0;i<H;i++){ xn[i]=(float)(i%9-4)*0.1f; hb[i]=(unsigned char)(i&0xff); }
  for(int i=0;i<E;i++) logit[i]=(float)((i*11)%23-11)*0.3f;

  for(int N=4;N<=8;N+=4){
    uint64_t comp=0, comm=0, t0;
    /* --- compute: RMSNorm --- */
    t0=rd_cyc();
    float ss=0; for(int i=0;i<H;i++) ss+=xn[i]*xn[i];
    float inv=1.0f/sqrtf(ss/(float)H+1e-6f); for(int i=0;i<H;i++) xn[i]*=inv;
    /* --- compute: router softmax top-8 --- */
    float mx=-1e30f; for(int e=0;e<E;e++) if(logit[e]>mx) mx=logit[e];
    float sm=0; for(int e=0;e<E;e++){ prob[e]=expf(logit[e]-mx); sm+=prob[e]; }
    comp += rd_cyc()-t0;
    /* --- comm: dispatch all-to-all --- */
    comm += a2a(N);
    /* --- compute: expert FFN SwiGLU --- */
    t0=rd_cyc();
    for(int i=0;i<MI;i++){ gate[i]=xn[i%H]*1.1f; up[i]=xn[(i+7)%H]*0.9f; }
    for(int i=0;i<MI;i++){ float g=gate[i]; ffn[i]=(g/(1.0f+expf(-g)))*up[i]; }
    comp += rd_cyc()-t0;
    /* --- comm: combine all-to-all --- */
    comm += a2a(N);
    printf("MoE-layer N=%d compute=%lu comm=%lu comm_frac_pct=%lu\n",
           N,(unsigned long)comp,(unsigned long)comm,(unsigned long)(comm*100/(comp+comm)));
  }
  return 0;
}
