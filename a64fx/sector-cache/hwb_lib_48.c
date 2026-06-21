/* 48-thread hierarchical HW barrier via libhwb (vhbm_*), from raw pthreads.
 *
 * The A64FX HW barrier is intra-CMG only, so a 48T (4-CMG) barrier needs:
 *   HW intra-CMG arrival  ->  4-way SW leader combine  ->  HW intra-CMG release.
 * Compare correctness + latency against our production flat SEV/WFE barrier.
 *
 * Build:
 *   fcc -Nclang -O3 -march=armv8.2-a+sve -o hwb_lib_48 hwb_lib_48.c -lhwb -lpthread -lm
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <unistd.h>

extern int  vhbm_bar_init(uint64_t core_bitmask);
extern int  vhbm_bar_assign(int bd_mask, void *bb_hint);
extern void vhbm_bar(long bb);

static inline uint64_t rdtsc(void){ uint64_t v; __asm__ volatile("mrs %0, cntvct_el0":"=r"(v)); return v; }
static inline uint64_t rdfreq(void){ uint64_t v; __asm__ volatile("mrs %0, cntfrq_el0":"=r"(v)); return v; }

#define NT 48
#define CORE0 12
#define TPC 12

static int g_init_ret;
static volatile long shared[NT];
static pthread_barrier_t startb;

/* ---- 4-way SW sense-reversing barrier for the 4 CMG leaders ---- */
typedef struct { volatile int cnt; char p[60]; volatile int sense; } sw4_t;
static sw4_t SW4;
static inline void sw4_barrier(int *ls){
    int my=!(*ls);
    if(__sync_add_and_fetch(&SW4.cnt,1)==4){ SW4.cnt=0; __sync_synchronize(); SW4.sense=my; }
    else { while(SW4.sense!=my) __asm__ volatile("yield"); }
    *ls=my;
}

/* ---- production flat single-atomic SEV/WFE barrier (tf_spin_barrier) ---- */
typedef struct { volatile int cnt; char p[60]; volatile int sense; } flat_t;
static flat_t FLAT;
static inline void flat_barrier(int *ls){
    int my=!(*ls);
    if(__sync_add_and_fetch(&FLAT.cnt,1)==NT){ FLAT.cnt=0; __sync_synchronize(); FLAT.sense=my; __asm__ volatile("sev"); }
    else { __asm__ volatile("sevl"); do{__asm__ volatile("wfe");}while(FLAT.sense!=my); }
    *ls=my;
}

typedef struct { long tid; long bb; int is_leader; int mode; double ns; long bad; } targ;
/* mode 0 = hier HW, mode 1 = flat */

static void watchdog(int s){ (void)s;
    const char *m="\n*** WATCHDOG: barrier HUNG — aborting ***\n"; write(2,m,strlen(m)); _exit(3);
}

static void *worker(void *p){
    targ *a=(targ*)p;
    long tid=a->tid;
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(CORE0+tid,&s);
    pthread_setaffinity_np(pthread_self(),sizeof(s),&s);

    int bb = vhbm_bar_assign(g_init_ret, NULL);
    a->bb = bb;
    pthread_barrier_wait(&startb);

    int ls4=0, lsf=0; long bad=0, iters=20000, bbl=bb;
    if(a->mode==0){
        for(int w=0;w<200;w++){ vhbm_bar(bbl); if(a->is_leader) sw4_barrier(&ls4); vhbm_bar(bbl); }
        uint64_t c0=rdtsc();
        for(long it=0;it<iters;it++){
            shared[tid]=it;
            vhbm_bar(bbl); if(a->is_leader) sw4_barrier(&ls4); vhbm_bar(bbl);
            for(int j=0;j<NT;j++) if(shared[j]!=it){ bad++; break; }
            vhbm_bar(bbl); if(a->is_leader) sw4_barrier(&ls4); vhbm_bar(bbl);
        }
        uint64_t c1=rdtsc();
        if(tid==0){ double f=(double)rdfreq(); a->ns=(double)(c1-c0)/f*1e9/(double)(iters*2); }
    } else {
        for(int w=0;w<200;w++) flat_barrier(&lsf);
        uint64_t c0=rdtsc();
        for(long it=0;it<iters;it++){
            shared[tid]=it;
            flat_barrier(&lsf);
            for(int j=0;j<NT;j++) if(shared[j]!=it){ bad++; break; }
            flat_barrier(&lsf);
        }
        uint64_t c1=rdtsc();
        if(tid==0){ double f=(double)rdfreq(); a->ns=(double)(c1-c0)/f*1e9/(double)(iters*2); }
    }
    a->bad=bad;
    return NULL;
}

static void run(int mode,const char*name){
    memset((void*)shared,0xff,sizeof shared);
    memset(&SW4,0,sizeof SW4); memset(&FLAT,0,sizeof FLAT);
    pthread_barrier_init(&startb,NULL,NT);
    pthread_t th[NT]; targ ar[NT];
    for(int t=0;t<NT;t++){ ar[t].tid=t; ar[t].is_leader=(t%TPC==0); ar[t].mode=mode; ar[t].ns=0; ar[t].bad=0; ar[t].bb=-1; }
    alarm(20);
    for(int t=1;t<NT;t++) pthread_create(&th[t],NULL,worker,&ar[t]);
    worker(&ar[0]);
    for(int t=1;t<NT;t++) pthread_join(th[t],NULL);
    alarm(0);
    pthread_barrier_destroy(&startb);
    long bad=0; for(int t=0;t<NT;t++) bad+=ar[t].bad;
    /* show the bb each CMG leader got */
    printf("  %-22s lat=%8.1f ns/bar  sync_errors=%ld %s",
           name, ar[0].ns, bad, bad?"<<< BROKEN":"OK");
    if(mode==0) printf("   (bb per CMG: %ld %ld %ld %ld)", ar[0].bb, ar[12].bb, ar[24].bb, ar[36].bb);
    printf("\n");
}

int main(void){
    setvbuf(stdout,NULL,_IONBF,0); setvbuf(stderr,NULL,_IONBF,0);
    signal(SIGALRM,watchdog);
    uint64_t mask=0; for(int c=CORE0;c<CORE0+NT;c++) mask|=(1ULL<<c);
    fprintf(stderr,"vhbm_bar_init(mask=0x%016llx) [cores %d..%d, all 4 CMGs]\n",
            (unsigned long long)mask, CORE0, CORE0+NT-1);
    g_init_ret = vhbm_bar_init(mask);
    fprintf(stderr,"  returned %d\n", g_init_ret);
    if(g_init_ret<0){ fprintf(stderr,"init failed %d\n",-g_init_ret); return 1; }
    printf("48-thread barrier comparison (cores 12-59):\n");
    run(0,"hier HW (vhbm)");
    run(1,"flat SEV/WFE");
    return 0;
}
