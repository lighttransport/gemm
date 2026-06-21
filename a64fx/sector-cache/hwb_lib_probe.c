/* Drive the Fujitsu libhwb (vhbm_*) HW-barrier API from RAW pthreads.
 *
 * Discovery from disassembly of /lib64/libhwb.so.1:
 *   int  vhbm_bar_init(uint64_t core_bitmask);   // bit i = node-core i participates
 *   int  vhbm_bar_assign(int bd_mask, void *bb_hint); // per-thread join; returns bb (0..3)
 *   void vhbm_bar(long bb);                       // hot barrier on BST reg bb
 *   int  vhbm_bar_unassign(int bd_mask);
 *   int  vhbm_bar_fini(...);
 *
 * We probe the exact contract empirically and measure latency vs our flat barrier.
 * Watchdog alarm() guards against the unbounded WFE poll hanging (as a misconfig did before).
 *
 * Build:
 *   fcc -Nclang -O3 -march=armv8.2-a+sve -o hwb_lib_probe hwb_lib_probe.c -lhwb -lpthread -lm
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <unistd.h>

extern int  vhbm_bar_init(uint64_t core_bitmask);
extern int  vhbm_bar_assign(int bd_mask, void *bb_hint);
extern void vhbm_bar(long bb);
extern int  vhbm_bar_unassign(int bd_mask);

static inline uint64_t rdtsc(void){ uint64_t v; __asm__ volatile("mrs %0, cntvct_el0":"=r"(v)); return v; }
static inline uint64_t rdfreq(void){ uint64_t v; __asm__ volatile("mrs %0, cntfrq_el0":"=r"(v)); return v; }

#define NPER 12          /* threads in CMG0 */
#define CORE0 12
static int g_init_ret;
static volatile long shared[NPER];
static pthread_barrier_t startb;
static double g_lat_ns;
static long g_bad;

static void on_alarm(int s){ (void)s;
    const char *m="\n*** WATCHDOG: vhbm_bar HUNG (did not converge) — aborting ***\n";
    write(2,m,strlen(m)); _exit(3);
}

static void *worker(void *p){
    long tid=(long)p;
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(CORE0+tid,&s);
    pthread_setaffinity_np(pthread_self(),sizeof(s),&s);

    /* per-thread join: kernel assigns IMP_BARRIER_ASSIGN_EL1 for THIS core */
    int bb = vhbm_bar_assign(g_init_ret, NULL);
    if(tid==0) fprintf(stderr,"  [tid0] vhbm_bar_assign(%d,NULL) = %d (bb)\n", g_init_ret, bb);
    if(bb<0){ fprintf(stderr,"  [tid%ld] assign FAILED = %d\n",tid,bb); }

    pthread_barrier_wait(&startb);  /* all joined before any barrier */

    /* correctness + latency: each iter every thread bumps its slot, barrier,
       then checks all NPER slots equal it. */
    long bad=0, bbl=bb;
    for(int w=0; w<200; w++) vhbm_bar(bbl);   /* warmup */
    uint64_t c0=rdtsc();
    long iters=20000;
    for(long it=0; it<iters; it++){
        shared[tid]=it;
        vhbm_bar(bbl);
        for(int j=0;j<NPER;j++) if(shared[j]!=it){ bad++; break; }
        vhbm_bar(bbl);
    }
    uint64_t c1=rdtsc();
    if(tid==0){ double f=(double)rdfreq(); g_lat_ns=(double)(c1-c0)/f*1e9/(double)(iters*2); }
    __sync_add_and_fetch(&g_bad,bad);
    return NULL;
}

int main(void){
    setvbuf(stdout,NULL,_IONBF,0); setvbuf(stderr,NULL,_IONBF,0);
    signal(SIGALRM,on_alarm);

    /* core bitmask: cores 12..23 (CMG0) */
    uint64_t mask=0; for(int c=CORE0;c<CORE0+NPER;c++) mask |= (1ULL<<c);
    fprintf(stderr,"vhbm_bar_init(mask=0x%016llx)  [cores %d..%d]\n",
            (unsigned long long)mask, CORE0, CORE0+NPER-1);
    g_init_ret = vhbm_bar_init(mask);
    fprintf(stderr,"  vhbm_bar_init returned %d\n", g_init_ret);
    if(g_init_ret<0){ fprintf(stderr,"init failed (errno-ish %d) — abort\n",-g_init_ret); return 1; }

    memset((void*)shared,0xff,sizeof shared);
    pthread_barrier_init(&startb,NULL,NPER);
    pthread_t th[NPER];
    alarm(15);   /* watchdog: if any vhbm_bar hangs, kill in 15s */
    for(long t=1;t<NPER;t++) pthread_create(&th[t],NULL,worker,(void*)t);
    worker((void*)0);
    for(long t=1;t<NPER;t++) pthread_join(th[t],NULL);
    alarm(0);

    printf("\nRESULT (CMG0, %d threads):\n", NPER);
    printf("  vhbm HW barrier: %.1f ns/bar   sync_errors=%ld  %s\n",
           g_lat_ns, g_bad, g_bad? "<<< BROKEN":"CORRECT");
    return 0;
}
