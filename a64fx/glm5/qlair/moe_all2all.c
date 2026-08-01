/* GLM5 MoE all-to-all communication cost on qlair's calibrated uTofu model.
 * Models each rank's all-to-all EGRESS as (N-1) puts of the per-pair size,
 * timed via the guest cycle counter (CNTVCT). qlair charges the Fugaku-
 * calibrated put cost (~1400 ns base + 6.3 GB/s per TNI) into that counter.
 * Reports serialized (1-TNI) and 6-TNI-parallel latency + effective BW.
 * GLM5: hidden=6144, int8 activations (1 B/elem), 256 experts, top-8. */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
typedef uint16_t  utofu_tni_id_t;
typedef uintptr_t utofu_vcq_hdl_t;
typedef uint64_t  utofu_vcq_id_t;
typedef uint64_t  utofu_stadd_t;
extern int utofu_get_onesided_tnis(utofu_tni_id_t**, size_t*);
extern int utofu_create_vcq(utofu_tni_id_t, unsigned long, utofu_vcq_hdl_t*);
extern int utofu_query_vcq_id(utofu_vcq_hdl_t, utofu_vcq_id_t*);
extern int utofu_reg_mem(utofu_vcq_hdl_t, void*, size_t, unsigned long, utofu_stadd_t*);
extern int utofu_poll_tcq(utofu_vcq_hdl_t, unsigned long, void**);
extern int myputn(utofu_vcq_hdl_t, utofu_vcq_id_t, utofu_stadd_t, utofu_stadd_t, size_t);
extern uint64_t rd_cyc(void);
#define HIDDEN 6144
#define BUFMAX 1048576
static unsigned char sbuf[BUFMAX] __attribute__((aligned(256)));
static unsigned char rbuf[BUFMAX] __attribute__((aligned(256)));
static utofu_vcq_hdl_t H0; static utofu_vcq_id_t ID1; static utofu_stadd_t S0,S1;
static uint64_t a2a(int N, size_t per){
  void* d; int rc; uint64_t t0=rd_cyc();
  for(int j=0;j<N-1;j++){ myputn(H0,ID1,S0,S1,per); do{rc=utofu_poll_tcq(H0,0,&d);}while(rc!=0); }
  return rd_cyc()-t0;
}
/* aligned table row: left-justified label, then N, per-pair, steps, latencies, BW */
static void rep(const char* label, int N, size_t per){
  uint64_t ns = a2a(N, per);
  int tni = (N-1) < 6 ? (N-1) : 6;
  uint64_t par = ns / (tni>0?tni:1);
  uint64_t bw100 = par ? (uint64_t)per*(N-1)*100/par : 0;
  printf("  %-22s N=%d  pair=%7luB  steps=%d  ser=%7lu ns  par=%7lu ns  %2lu.%02lu GB/s\n",
         label, N, (unsigned long)per, N-1,
         (unsigned long)ns, (unsigned long)par,
         (unsigned long)(bw100/100), (unsigned long)(bw100%100));
}
int main(void){
  utofu_tni_id_t* t; size_t nt; utofu_get_onesided_tnis(&t,&nt);
  utofu_vcq_hdl_t h0,h1; utofu_create_vcq(t[0],0,&h0); utofu_create_vcq(t[1],0,&h1);
  utofu_vcq_id_t id0,id1; utofu_query_vcq_id(h0,&id0); utofu_query_vcq_id(h1,&id1);
  utofu_stadd_t s0,s1; utofu_reg_mem(h0,sbuf,BUFMAX,0,&s0); utofu_reg_mem(h1,rbuf,BUFMAX,0,&s1);
  H0=h0; ID1=id1; S0=s0; S1=s1;
  printf("GLM5 MoE all-to-all comm cost (qlair uTofu, TNIs=%lu)\n",(unsigned long)nt);
  rep("decode dispatch", 4, HIDDEN);
  rep("decode dispatch", 8, HIDDEN);
  rep("decode combine", 4, HIDDEN);
  rep("decode combine", 8, HIDDEN);
  rep("prefill dispatch c64", 4, (size_t)64*8/4*HIDDEN);
  rep("prefill dispatch c64", 8, (size_t)64*8/8*HIDDEN);
  rep("attn all-reduce", 4, 2048);
  rep("attn all-reduce", 8, 2048);
  printf("DONE\n");
  return 0;
}
