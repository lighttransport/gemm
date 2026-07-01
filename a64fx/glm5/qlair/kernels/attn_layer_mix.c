/* GLM5.2 decode attention-layer step: compute (RMSNorm + MLA q/kv projections +
 * attention) mixed with comm (attention all-reduce). kv_lora=512, qk_head=256,
 * n_heads=64. AR payload = kv_lora f32 = 2048 B. Measures compute vs comm.
 * Compile -O0; run with qlair --cores 2. */
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
#define KVL 512
#define QKH 256
#define NH 64
static float xn[H], kv[KVL], q[QKH], sc[NH];
static unsigned char sb[2048], rb[2048];   /* AR payload = 512 f32 */
static utofu_vcq_hdl_t H0; static utofu_vcq_id_t I1; static utofu_stadd_t S0,S1;
static uint64_t allreduce(int N){ void* d; int rc; uint64_t t0=rd_cyc();
  /* ring/recursive-doubling ~ (N-1) exchanges of the 2048B partial */
  for(int j=0;j<N-1;j++){ myputn(H0,I1,S0,S1,2048); do{rc=utofu_poll_tcq(H0,0,&d);}while(rc!=0);} return rd_cyc()-t0; }
int main(void){
  utofu_tni_id_t* t; size_t nt; utofu_get_onesided_tnis(&t,&nt);
  utofu_vcq_hdl_t h0,h1; utofu_create_vcq(t[0],0,&h0); utofu_create_vcq(t[1],0,&h1);
  utofu_vcq_id_t i0,i1; utofu_query_vcq_id(h0,&i0); utofu_query_vcq_id(h1,&i1);
  utofu_stadd_t s0,s1; utofu_reg_mem(h0,sb,2048,0,&s0); utofu_reg_mem(h1,rb,2048,0,&s1);
  H0=h0; I1=i1; S0=s0; S1=s1;
  for(int i=0;i<H;i++) xn[i]=(float)(i%9-4)*0.1f;
  for(int N=4;N<=8;N+=4){
    uint64_t comp=0, comm=0, t0=rd_cyc();
    /* RMSNorm */
    float ss=0; for(int i=0;i<H;i++) ss+=xn[i]*xn[i];
    float inv=1.0f/sqrtf(ss/(float)H+1e-6f);
    /* q_a/kv_a down-projection (matvec-ish reduce over hidden) */
    for(int k=0;k<KVL;k++){ float a=0; for(int i=0;i<H;i+=32) a+=xn[i]*xn[(i+k)%H]; /* down-scaled reduce */ kv[k]=a*inv; }
    for(int k=0;k<QKH;k++){ float a=0; for(int i=0;i<KVL;i++) a+=kv[i]*(float)((i+k)&7); q[k]=a; }
    /* per-head scaled score (softmax-ish) */
    for(int hh=0;hh<NH;hh++){ float a=0; for(int k=0;k<QKH;k++) a+=q[k]*q[(k+hh)%QKH]; sc[hh]=a/(float)QKH; }
    comp += rd_cyc()-t0;
    comm += allreduce(N);   /* attention all-reduce */
    printf("attn-layer N=%d compute=%lu comm=%lu comm_frac_pct=%lu\n",
           N,(unsigned long)comp,(unsigned long)comm,(unsigned long)(comm*100/(comp+comm)));
  }
  return 0;
}
