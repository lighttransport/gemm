/* Bounded A64FX decode diagnostics: pool barriers and memory read throughput.
 * No model is loaded. At most 768 MiB is allocated for the stream tests.
 * Build as test_iq3_tokenmajor.c, with -lhwb for Fugaku huge pages.
 */
#include <omp.h>
#define GGUF_LOADER_IMPLEMENTATION
#include "../../common/gguf_loader.h"
#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/ggml_dequant.h"
#define TRANSFORMER_IMPLEMENTATION
#ifndef TF_TEST_TRANSFORMER_HEADER
#define TF_TEST_TRANSFORMER_HEADER "../../common/transformer.h"
#endif
#include TF_TEST_TRANSFORMER_HEADER

typedef struct {
    transformer_model *m;
    int tid, iterations;
    volatile int *values;
    int bad;
} barrier_task;

static void *barrier_worker(void *arg) {
    barrier_task *t = arg;
    tf_barrier_tid = t->tid;
    int sense = 0;
    for (int i = 0; i < 64; i++) {
        t->values[t->tid*64] = i+1;
        tf_spin_barrier(t->m, &sense, t->m->n_threads);
        for (int j = 0; j < t->m->n_threads; j++)
            if (t->values[j*64] != i+1) t->bad++;
        tf_spin_barrier(t->m, &sense, t->m->n_threads);
    }
    for (int i = 0; i < t->iterations; i++)
        tf_spin_barrier(t->m, &sense, t->m->n_threads);
    return NULL;
}

static int test_barrier(int nt, int mode) {
    transformer_model m = {0};
    m.n_threads = nt;
    m.numa.enabled = 1;
    m.numa.n_cmgs = 4;
    m.pool_core_striped = 1;
    transformer_set_decode_barrier(&m, mode);
    setenv("NUMA_INTERLEAVE", "1", 1); /* pool core binding, not allocation */
    tf_pool_start(&m);
    volatile int values[48*64] = {0};
    barrier_task tasks[48];
    const int iterations = 2000;
    for (int t = 0; t < nt; t++)
        tasks[t] = (barrier_task){&m,t,iterations,values,0};
    double start = omp_get_wtime();
    tf_pool_dispatch(&m, barrier_worker, tasks, sizeof(*tasks));
    double elapsed = omp_get_wtime()-start;
    int bad = 0;
    for (int t = 0; t < nt; t++) bad += tasks[t].bad;
    printf("BARRIER nt=%d mode=%d state_stride=%zu us=%.3f ordering=%s\n",
        nt,mode,sizeof(tf_cmg_barrier_state),elapsed*1e6/(iterations+128),bad?"FAIL":"PASS");
    tf_pool_shutdown(&m);
    return bad;
}

static float stream(const uint8_t *p, size_t n, int extend) {
    svbool_t pg = svptrue_b32();
    svfloat32_t a0=svdup_f32(0),a1=a0,a2=a0,a3=a0;
    if (extend) {
        for (size_t i=0; i<n; i+=64) {
            a0=svadd_f32_x(pg,a0,svcvt_f32_s32_x(pg,svld1sb_s32(pg,(const int8_t *)p+i)));
            a1=svadd_f32_x(pg,a1,svcvt_f32_s32_x(pg,svld1sb_s32(pg,(const int8_t *)p+i+16)));
            a2=svadd_f32_x(pg,a2,svcvt_f32_s32_x(pg,svld1sb_s32(pg,(const int8_t *)p+i+32)));
            a3=svadd_f32_x(pg,a3,svcvt_f32_s32_x(pg,svld1sb_s32(pg,(const int8_t *)p+i+48)));
        }
    } else {
        for (size_t i=0; i<n; i+=256) {
            a0=svadd_f32_x(pg,a0,svld1_f32(pg,(const float *)(p+i)));
            a1=svadd_f32_x(pg,a1,svld1_f32(pg,(const float *)(p+i+64)));
            a2=svadd_f32_x(pg,a2,svld1_f32(pg,(const float *)(p+i+128)));
            a3=svadd_f32_x(pg,a3,svld1_f32(pg,(const float *)(p+i+192)));
        }
    }
    return svaddv_f32(pg,svadd_f32_x(pg,svadd_f32_x(pg,a0,a1),svadd_f32_x(pg,a2,a3)));
}

static int test_stream(unsigned long mask) {
    if (syscall(SYS_set_mempolicy, mask ? 3 : 0, mask ? &mask : NULL, 8UL)) {
        perror("set_mempolicy"); return 1;
    }
    const size_t bytes = 512UL*1024*1024;
    uint8_t *p = tf_aligned_alloc_notouch(2097152,bytes);
    if (!p) return 1;
    #pragma omp parallel for num_threads(48) schedule(static)
    for (size_t i=0; i<bytes/sizeof(float); i++) ((float *)p)[i]=1.f;
    int cpus[48];
    #pragma omp parallel num_threads(48)
    { cpus[omp_get_thread_num()]=sched_getcpu(); }
    int unique=0;
    for (int i=0; i<48; i++) {
        int seen=0;
        for (int j=0; j<i; j++) if (cpus[i]==cpus[j]) seen=1;
        unique+=!seen;
    }
    printf("STREAM mask=%lx unique_cores=%d\n",mask,unique);
    for (int ext=0; ext<2; ext++) {
        float sum=0;
        double start=omp_get_wtime();
        #pragma omp parallel num_threads(48) reduction(+:sum)
        {
            int tid=omp_get_thread_num(),nt=omp_get_num_threads();
            size_t units=bytes/256, start_unit=units*tid/nt,end_unit=units*(tid+1)/nt;
            for (int r=0;r<5;r++) sum+=stream(p+start_unit*256,(end_unit-start_unit)*256,ext);
        }
        double elapsed=omp_get_wtime()-start;
        printf("STREAM mask=%lx extend=%d GB/s=%.2f checksum=%g\n",mask,ext,bytes*5/elapsed/1e9,sum);
    }
    free(p);
    return unique==48 ? 0 : 1;
}

static int test_worker_alloc(void) {
    const size_t bytes=16UL*1024*1024;
    uint8_t *p[48]={0};
    int bad=0;
    #pragma omp parallel num_threads(48) reduction(+:bad)
    {
        int tid=omp_get_thread_num();
        if (syscall(SYS_set_mempolicy,0,NULL,8UL)) bad++;
        p[tid]=tf_aligned_alloc_notouch(2097152,bytes);
        if (!p[tid]) bad++;
        else for (size_t i=0;i<bytes/4;i++) ((float *)p[tid])[i]=1.f;
    }
    if (bad) return bad;
    for (int ext=0;ext<2;ext++) {
        float sum=0;
        double start=omp_get_wtime();
        #pragma omp parallel num_threads(48) reduction(+:sum)
        {
            int tid=omp_get_thread_num();
            for (int r=0;r<5;r++) sum+=stream(p[tid],bytes,ext);
        }
        printf("STREAM worker-alloc extend=%d GB/s=%.2f checksum=%g\n",
            ext,bytes*48*5/(omp_get_wtime()-start)/1e9,sum);
    }
    for (int t=0;t<48;t++) free(p[t]);
    return 0;
}

int main(int argc, char **argv) {
    if (svcntb()!=64) return 2;
    int bad=0;
    if (argc>1 && !strcmp(argv[1],"stream")) {
        bad+=test_stream(0);
        bad+=test_stream(0xf0);
        bad+=test_stream(0xff);
        bad+=test_worker_alloc();
    } else {
        int mode=argc>1?atoi(argv[1]):TF_DECODE_BARRIER_DEFAULT;
        bad+=test_barrier(4,mode);
        bad+=test_barrier(12,mode);
        bad+=test_barrier(48,mode);
    }
    return bad?1:0;
}
