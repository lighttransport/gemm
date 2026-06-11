/* Pool-barrier dispatch-overhead microbench: counter (flagbar=0) vs per-worker flag (flagbar=1).
 * Quantifies the per-dispatch barrier cost that decode pays ~688x/token. Build:
 *   fcc -Nclang -O3 -march=armv8.2-a+sve -ffp-contract=fast -fopenmp -D_GNU_SOURCE \
 *       -I../../common -o /tmp/poolbar tools/ws_poolbar_bench.c -lm -lpthread -lhwb
 *   OMP_NUM_THREADS=48 taskset -c 12-59 /tmp/poolbar 48 4
 */
#include "ds4f.h"
#include <stdio.h>
#include <time.h>

static _Atomic long g_touch[64];
static void noop(void *a, int tid, int n){ (void)a;(void)n; atomic_fetch_add_explicit(&g_touch[tid], 1, memory_order_relaxed); }
/* ~tiny balanced work per thread, to expose barrier-vs-work ratio */
static void smallwork(void *a, int tid, int n){ (void)a;(void)n; double s=0; for(int i=0;i<3000;i++) s+=i*0.5; atomic_fetch_add_explicit(&g_touch[tid],(long)s,memory_order_relaxed); }
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

int main(int argc,char**argv){
    int nthr = argc>1?atoi(argv[1]):48, ncmg = argc>2?atoi(argv[2]):4;
    ds4f_pool *p = ds4f_pool_start(nthr, ncmg);
    printf("pool-barrier bench  nthr=%d n_cmgs=%d\n", nthr, ncmg);
    long N = 100000;
    for(int fb=0; fb<2; fb++){
        p->flagbar = fb;
        for(int i=0;i<2000;i++) ds4f_pool_run(p, noop, NULL);              /* warm */
        double t0=now(); for(long i=0;i<N;i++) ds4f_pool_run(p, noop, NULL); double dt=now()-t0;
        double empty_ns = dt/N*1e9;
        for(int i=0;i<200;i++) ds4f_pool_run(p, smallwork, NULL);
        t0=now(); long M=N/10; for(long i=0;i<M;i++) ds4f_pool_run(p, smallwork, NULL); dt=now()-t0;
        double work_ns = dt/M*1e9;
        printf("  flagbar=%d  empty=%.0f ns/dispatch   ~work=%.0f ns/dispatch\n", fb, empty_ns, work_ns);
    }
    ds4f_pool_stop(p);
    printf("(touch sink=%ld)\n", (long)g_touch[0]);
    return 0;
}
