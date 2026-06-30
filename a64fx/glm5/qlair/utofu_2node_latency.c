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
#define MAXLEN 65536
static unsigned char sbuf[MAXLEN] __attribute__((aligned(256)));
static unsigned char rbuf[MAXLEN] __attribute__((aligned(256)));
static uint64_t measure(utofu_vcq_hdl_t h0, utofu_vcq_id_t id1, utofu_stadd_t s0,
                        utofu_stadd_t s1, size_t len, int reps){
  void* done; int rc;
  uint64_t t0 = rd_cyc();
  for(int n=0;n<reps;n++){
    myputn(h0, id1, s0, s1, len);
    do { rc = utofu_poll_tcq(h0,0,&done); } while(rc!=0);
  }
  uint64_t t1 = rd_cyc();
  return (t1 - t0);
}
int main(void){
  utofu_tni_id_t* tnis; size_t ntni; utofu_get_onesided_tnis(&tnis,&ntni);
  utofu_vcq_hdl_t h0,h1; utofu_create_vcq(tnis[0],0,&h0); utofu_create_vcq(tnis[1],0,&h1);
  utofu_vcq_id_t id0,id1; utofu_query_vcq_id(h0,&id0); utofu_query_vcq_id(h1,&id1);
  utofu_stadd_t s0,s1; utofu_reg_mem(h0,sbuf,MAXLEN,0,&s0); utofu_reg_mem(h1,rbuf,MAXLEN,0,&s1);
  printf("uTofu put+poll latency (qlair, %zu TNIs, model: 1400ns + 6.3GB/s/TNI):\n",(size_t)ntni);
  size_t sizes[5]={64,1024,4096,16384,65536}; int reps=16;
  for(int i=0;i<5;i++){
    uint64_t cyc = measure(h0,id1,s0,s1,sizes[i],reps);
    uint64_t per = cyc/reps;
    /* qlair CNTVCT delta == simulated nanoseconds */
    uint64_t ns = per;
    uint64_t bw_x100 = sizes[i]*100/ns;       /* bytes/ns = GB/s, *100 */
    printf("  %6zu B: %6llu ns/put  ( %2llu.%02llu us )   %2llu.%02llu GB/s\n",
           sizes[i],(unsigned long long)ns,
           (unsigned long long)(ns/1000),(unsigned long long)((ns%1000)/10),
           (unsigned long long)(bw_x100/100),(unsigned long long)(bw_x100%100));
  }
  printf("DONE\n");
  return 0;
}
