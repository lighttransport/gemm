/* 2-endpoint uTofu comm test for qlair. qlair's uTofu is intra-node shared-mem
 * with a Fugaku-calibrated latency model. We create two VCQs (rank0/rank1
 * endpoints), register a buffer on each, then RDMA-put rank0->rank1 and poll
 * the TCQ for completion. Single thread => coherent verify + the marker region
 * captures the modeled transfer latency. */
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
extern int utofu_put(utofu_vcq_hdl_t, utofu_vcq_id_t, utofu_stadd_t, utofu_stadd_t,
                     size_t, uint64_t, unsigned long, void*);
extern int utofu_poll_tcq(utofu_vcq_hdl_t, unsigned long, void**);
extern int myput(utofu_vcq_hdl_t, utofu_vcq_id_t, utofu_stadd_t, utofu_stadd_t, size_t);
void __qlair_sim_start(void); void __qlair_sim_end(void);
#define TCQ_NOTICE (1UL<<14)
#define MSGLEN 4096
static unsigned char sbuf[MSGLEN] __attribute__((aligned(256)));
static unsigned char rbuf[MSGLEN] __attribute__((aligned(256)));

int main(void){
  utofu_tni_id_t* tnis; size_t ntni;
  utofu_get_onesided_tnis(&tnis, &ntni);
  printf("uTofu: %zu one-sided TNIs available\n",(size_t)ntni);
  /* rank0 (sender) endpoint */
  utofu_vcq_hdl_t h0; utofu_create_vcq(tnis[0], 0, &h0);
  utofu_vcq_id_t id0; utofu_query_vcq_id(h0, &id0);
  utofu_stadd_t s0; utofu_reg_mem(h0, sbuf, MSGLEN, 0, &s0);
  /* rank1 (receiver) endpoint, different TNI */
  utofu_vcq_hdl_t h1; utofu_create_vcq(tnis[1], 0, &h1);
  utofu_vcq_id_t id1; utofu_query_vcq_id(h1, &id1);
  utofu_stadd_t s1; utofu_reg_mem(h1, rbuf, MSGLEN, 0, &s1);
  printf("rank0 vcq_id=0x%llx stadd=0x%llx | rank1 vcq_id=0x%llx stadd=0x%llx\n",
         (unsigned long long)id0,(unsigned long long)s0,
         (unsigned long long)id1,(unsigned long long)s1);
  for(int i=0;i<MSGLEN;i++) sbuf[i]=(unsigned char)(i*7+1);
  printf("rank0 RDMA-put %d bytes -> rank1 ...\n", MSGLEN);
  __qlair_sim_start();
  void* cb=(void*)0xABCD;
  (void)cb;
  int prc = myput(h0, id1, s0, s1, MSGLEN);
  void* done=0; int rc=-1, spins=0;
  for(spins=0; spins<64; spins++){ rc=utofu_poll_tcq(h0,0,&done); if(rc==0) break; }
  __qlair_sim_end();
  /* verify */
  int ok=1; for(int i=0;i<MSGLEN;i++) if(rbuf[i]!=(unsigned char)(i*7+1)){ ok=0; break; }
  printf("put rc=%d | poll_tcq rc=%d spins=%d cb=%p\n", prc, rc, spins, done);
  printf("rank1 rbuf: [0]=%d [255]=%d [256]=%d [2048]=%d [4095]=%d -> %s\n",
         rbuf[0],rbuf[255],rbuf[256],rbuf[2048],rbuf[MSGLEN-1], ok?"ALL 4096 BYTES OK":"MISMATCH");
  printf("DONE\n");
  return 0;
}
