/* N=1 fused dequant/dot and matched byte scans. Setup/checks are untimed. */
#define _GNU_SOURCE
#include "qwen38_lowbit.h"
#include <arm_sve.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>
#ifdef Q38_FAPP
#include <fj_tool/fapp.h>
#endif

__attribute__((noinline)) void __qlair_sim_start(unsigned long id) {
    (void)id; __asm__ volatile("" ::: "memory");
}
__attribute__((noinline)) void __qlair_sim_end(unsigned long id) {
    (void)id; __asm__ volatile("" ::: "memory");
}
static uint64_t ticks(void) { uint64_t v; __asm__ volatile("isb; mrs %0,cntvct_el0":"=r"(v)); return v; }
static uint64_t frequency(void) { uint64_t v; __asm__ volatile("mrs %0,cntfrq_el0":"=r"(v)); return v; }
static int rows, cols, cores, passes, warmup, format, arithmetic, no_numa;
static size_t group_bytes;
static uint8_t *segments[4];
static q38_lowbit_act *activations[4];
static float expected[8];
static pthread_barrier_t barrier;
typedef struct { int id, error; uint64_t begin, end; float checksum; } worker;
static worker workers[48];

static int pin(int cpu) {
    if (no_numa) return 1;
    cpu_set_t mask; CPU_ZERO(&mask); CPU_SET(cpu, &mask);
    return sched_setaffinity(0, sizeof(mask), &mask) == 0;
}
static uint8_t scan(const uint8_t *src, size_t bytes) {
    svuint8_t sum = svdup_u8(0);
    size_t i = 0;
    for (; i + 64 <= bytes; i += 64)
        sum = sveor_u8_x(svptrue_b8(), sum, svld1_u8(svptrue_b8(), src+i));
    if (i < bytes) sum = sveor_u8_x(svptrue_b8(), sum,
                                  svld1_u8(svwhilelt_b8(i, bytes), src+i));
    return svorv_u8(svptrue_b8(), sum);
}
static void *run(void *arg) {
    worker *w = arg;
    int cmg = w->id / 12, lane = w->id % 12, cmgs = (cores+11)/12;
    int count = cores == 1 ? 1 : 12, groups = rows/8;
    int local_groups = groups*(cmg+1)/cmgs - groups*cmg/cmgs;
    int first = local_groups*lane/count, last = local_groups*(lane+1)/count;
    int nr = (last-first)*8;
    const uint8_t *weights = segments[cmg] + (size_t)first*group_bytes;
    w->error = !pin(12+w->id);
    float *y = calloc((size_t)nr, sizeof(float));
    if (!y) w->error = 1;
    volatile uint8_t checksum = 0;
    for (int rep = 0; rep < warmup && !w->error; rep++) {
        if (arithmetic) w->error |= !q38_lowbit_sve(y,weights,format,activations[cmg],arithmetic,nr,cols);
        else checksum ^= scan(weights,(size_t)(last-first)*group_bytes);
    }
    pthread_barrier_wait(&barrier);
#ifdef Q38_FAPP
    char region[64]; snprintf(region,sizeof(region),"lowbit_%d_a%d_t%d",format,arithmetic,w->id);
    fapp_start(region,0,0);
#endif
    __qlair_sim_start((unsigned long)w->id);
    w->begin = ticks();
    for (int rep = 0; rep < passes && !w->error; rep++) {
        if (arithmetic) w->error |= !q38_lowbit_sve(y,weights,format,activations[cmg],arithmetic,nr,cols);
        else checksum ^= scan(weights,(size_t)(last-first)*group_bytes);
    }
    w->end = ticks();
    __qlair_sim_end((unsigned long)w->id);
#ifdef Q38_FAPP
    fapp_stop(region,0,0);
#endif
    pthread_barrier_wait(&barrier);
    if (arithmetic && !w->error) {
        for (int r = 0; r < nr; r++)
            if (!isfinite(y[r]) || fabsf(y[r]-expected[r%8]) > 1e-4f*(1+fabsf(expected[r%8]))) w->error = 1;
    }
    w->checksum = arithmetic && y ? y[0] : checksum;
    free(y);
    return NULL;
}

static int number(const char *s, int *out) {
    char *end; errno=0; long v=strtol(s,&end,10);
    if (errno || *end || v<0 || v>INT_MAX) return 0;
    *out=(int)v; return 1;
}
int main(int argc, char **argv) {
    if ((argc != 8 && argc != 9) ||
        !number(argv[1],&format) || !number(argv[2],&arithmetic) ||
        !number(argv[3],&rows) || !number(argv[4],&cols) ||
        !number(argv[5],&cores) || !number(argv[6],&passes) ||
        !number(argv[7],&warmup) || (argc==9 && strcmp(argv[8],"--no-numa"))) {
        fprintf(stderr,"usage: %s FORMAT(1=NVFP4,2=FP6) ARITH(0=scan,8,16) ROWS COLS CORES PASSES WARMUP [--no-numa]\n",argv[0]);
        return 2;
    }
    no_numa=argc==9;
    if ((format!=1 && format!=2) || (arithmetic && arithmetic!=8 && arithmetic!=16) ||
        (cores!=1 && cores!=12 && cores!=48) || rows<cores*8 || rows%8 ||
        cols<64 || cols%64 || cols>65536 || rows>1048576 || passes<1 ||
        passes>10000 || warmup>100 || svcntb()!=64) return 2;
    group_bytes=q38_lowbit_bytes(format,8,cols);
    uint8_t *packed=malloc(group_bytes);
    float *src=malloc((size_t)8*cols*sizeof(float)), *x=malloc((size_t)cols*sizeof(float));
    uint8_t *raw=malloc((size_t)8*(cols/64)*36);
    q38_lowbit_act *act=malloc((size_t)(cols/16)*sizeof(*act));
    if (!packed || !src || !x || !raw || !act) return 1;
    for (int r=0;r<8;r++) for (int k=0;k<cols;k++)
        src[(size_t)r*cols+k]=q38_fp6_decode((uint8_t)((k+r*11)&63))*0x1p-8f;
    for (int k=0;k<cols;k++) x[k]=(float)((k*17)%101-50)*.015625f;
    for (int r=0;r<8;r++) for (int b=0;b<cols/64;b++) {
        uint8_t *p=raw+((size_t)r*(cols/64)+b)*36;
        for (int s=0;s<4;s++) p[s]=(uint8_t)(1+(r+b+s)%31);
        for (int j=0;j<32;j++) p[j+4]=(uint8_t)(r*37+b*13+j*7);
    }
    int ok=format==1 ? q38_lowbit_pack_nvfp4(packed,group_bytes,raw,(size_t)(cols/64)*36,8,cols) :
                      q38_lowbit_pack_fp6(packed,group_bytes,src,cols,8,cols);
    ok &= q38_lowbit_prepare(act,(size_t)cols/16,x,cols,arithmetic ? arithmetic : 8);
    ok &= q38_lowbit_dot(expected,packed,format,act,arithmetic ? arithmetic : 8,8,cols);
    if (!ok) return 1;
    int cmgs=(cores+11)/12, groups=rows/8;
    for (int c=0;c<cmgs;c++) {
        if (!pin(12+12*c)) { perror("pin init"); return 1; }
        size_t ng=(size_t)(groups*(c+1)/cmgs-groups*c/cmgs), bytes=ng*group_bytes;
        size_t alloc=(bytes+2097151)&~(size_t)2097151;
        if (posix_memalign((void **)&segments[c],2097152,alloc)) return 1;
        if (!no_numa) {
            unsigned long node=1UL<<(4+c);
            if (syscall(SYS_mbind,segments[c],alloc,2,&node,64UL,0UL)) { perror("mbind"); return 1; }
        }
        for (size_t g=0;g<ng;g++) memcpy(segments[c]+g*group_bytes,packed,group_bytes);
        size_t ab=(size_t)(cols/16)*sizeof(*act);
        if (posix_memalign((void **)&activations[c],256,ab)) return 1;
        memcpy(activations[c],act,ab);
        if (!no_numa) for (size_t off=0;off<bytes;off+=2097152) {
            int node=-1;
            if (syscall(SYS_get_mempolicy,&node,NULL,0UL,segments[c]+off,3UL) || node!=4+c) {
                fprintf(stderr,"placement mismatch cmg=%d node=%d offset=%zu\n",c,node,off); return 1;
            }
        }
    }
    pthread_barrier_init(&barrier,NULL,(unsigned)cores);
    pthread_t tids[48];
    for (int i=0;i<cores;i++) workers[i].id=i;
    for (int i=1;i<cores;i++) if (pthread_create(&tids[i],NULL,run,&workers[i])) return 1;
    run(&workers[0]);
    for (int i=1;i<cores;i++) pthread_join(tids[i],NULL);
    uint64_t first=UINT64_MAX,last=0,slow=0,latest=0,hz=frequency();
    int errors=0;
    for (int i=0;i<cores;i++) {
        worker *w=&workers[i]; errors+=w->error;
        if (w->begin<first) first=w->begin;
        if (w->begin>latest) latest=w->begin;
        if (w->end>last) last=w->end;
        if (w->end-w->begin>slow) slow=w->end-w->begin;
    }
    size_t bytes=(size_t)groups*group_bytes;
    printf("LOWBIT format=%d arithmetic=%d rows=%d cols=%d cores=%d passes=%d weight_bytes=%zu "
           "counter_hz=%llu worker_ticks=%llu makespan_ticks=%llu start_span_ticks=%llu "
           "us=%.6f source_gbps=%.6f worker_gbps=%.6f placement=%s correct=%d\n",
           format,arithmetic,rows,cols,cores,passes,bytes,(unsigned long long)hz,
           (unsigned long long)slow,(unsigned long long)(last-first),(unsigned long long)(latest-first),
           (double)(last-first)/hz*1e6/passes,(double)bytes*passes*hz/(last-first)*1e-9,
           (double)bytes*passes*hz/slow*1e-9,no_numa?"unchecked":"verified",!errors);
    for (int c=0;c<cmgs;c++) { free(segments[c]); free(activations[c]); }
    free(packed);free(src);free(x);free(raw);free(act);
    pthread_barrier_destroy(&barrier);
    return errors ? 1 : 0;
}
