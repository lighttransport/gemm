/* Attribution benchmark for the attention inner loop.
 *
 * attention_full_flash sits at ~250 GFLOP/s, far below what its op count implies.
 * Rather than guess, time each primitive separately over the same key run, from
 * L2-resident KV so this measures issue rate, not DRAM:
 *
 *   qk_run          scores for n keys  (8 bf16 loads + 8 FMLA + 1 addv per key)
 *   exp_shift_sum   the softmax exp over n scores
 *   av_run          weighted V accumulate (8 bf16 loads + 8 FMLA per key)
 *
 * Per-key cycle counts tell us which one to attack.
 */
#define LAGUNA_BENCH
#include "laguna_s21_ep_runner.c"

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec+t.tv_nsec*1e-9; }

int main(int argc, char **argv) {
    int n    = argc>1 ? atoi(argv[1]) : 128;      /* keys per block (LAGUNA_KB) */
    int reps = argc>2 ? atoi(argv[2]) : 200000;
    int hd = LAGUNA_HEAD_DIM, kvstride = LAGUNA_KV_HEADS*hd;

    /* one kv head's worth of n keys: n*8*128*2 B; at n=128 that is 256 KB, L2 */
    uint16_t *K = aligned_alloc(256, (size_t)n*kvstride*sizeof(uint16_t));
    uint16_t *V = aligned_alloc(256, (size_t)n*kvstride*sizeof(uint16_t));
    float *q = aligned_alloc(256, hd*sizeof(float));
    float *sco = aligned_alloc(256, (size_t)(n+16)*sizeof(float));
    float *acc = aligned_alloc(256, hd*sizeof(float));
    uint64_t s=12345;
    for (size_t i=0;i<(size_t)n*kvstride;i++){ s=s*6364136223846793005ull+1;
        K[i]=laguna_f32_to_bf16((float)((int)((s>>33)%200)-100)/300.0f);
        V[i]=laguna_f32_to_bf16((float)((int)((s>>40)%200)-100)/300.0f); }
    for (int i=0;i<hd;i++) q[i]=(float)((i%13)-6)/7.0f;
    for (int i=0;i<hd;i++) acc[i]=0.0f;

    double t, dt; volatile float sink=0;
    const double GHZ = 2.0;

    laguna_qk_run(sco,q,K,kvstride,n,0.088f,hd);
    t=now_s(); for(int r=0;r<reps;r++) laguna_qk_run(sco,q,K,kvstride,n,0.088f,hd);
    dt=now_s()-t; sink+=sco[0];
    double qk_ns = dt/reps/n*1e9;
    printf("qk_run        %7.2f ns/key  %6.2f cyc/key  (%d bf16 ld + %d fmla + addv)\n",
           qk_ns, qk_ns*GHZ, hd/16, hd/16);

    for (int i=0;i<n;i++) sco[i]=(float)((i%17)-8)*0.1f;
    t=now_s(); for(int r=0;r<reps;r++){ sink+=laguna_exp_shift_sum(sco,n,1.0f);
        for(int i=0;i<n;i++) sco[i]=(float)((i%17)-8)*0.1f; }
    double dt_with_refill=now_s()-t;
    t=now_s(); for(int r=0;r<reps;r++){ for(int i=0;i<n;i++) sco[i]=(float)((i%17)-8)*0.1f; }
    double dt_refill=now_s()-t;
    double exp_ns=(dt_with_refill-dt_refill)/reps/n*1e9;
    printf("exp_shift_sum %7.2f ns/key  %6.2f cyc/key\n", exp_ns, exp_ns*GHZ);

    for (int i=0;i<n;i++) sco[i]=0.01f;
    laguna_av_run(acc,sco,V,kvstride,n,1.0f,hd);
    t=now_s(); for(int r=0;r<reps;r++) laguna_av_run(acc,sco,V,kvstride,n,1.0f,hd);
    dt=now_s()-t; sink+=acc[0];
    double av_ns=dt/reps/n*1e9;
    printf("av_run        %7.2f ns/key  %6.2f cyc/key  (%d bf16 ld + %d fmla)\n",
           av_ns, av_ns*GHZ, hd/16, hd/16);

    double tot=qk_ns+exp_ns+av_ns;
    printf("---\ntotal %7.2f ns/key => qk %.0f%%  exp %.0f%%  av %.0f%%\n",
           tot, 100*qk_ns/tot, 100*exp_ns/tot, 100*av_ns/tot);
    /* 512 flops per (query,key): 256 for qk, 256 for av */
    printf("implies %.0f GFLOP/s/core, %.0f GFLOP/s at 47 cores\n",
           512.0/tot, 512.0/tot*47);
    (void)sink;
    return 0;
}
