#define _POSIX_C_SOURCE 200809L
#include "ds41f_team.h"
#if defined(_OPENMP)
#include <errno.h>
#include <omp.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
/* A64FX cache lines are 256 bytes. Each worker publishes to its own line. */
typedef struct {atomic_uint_fast64_t epoch;unsigned char pad[256-sizeof(atomic_uint_fast64_t)];} completion;
static struct {
    atomic_uint_fast64_t epoch;
    ds41f_team_work work;
    void *context;
    size_t count;
    completion *done;
    int threads,active;
} team;
int ds41f_team_active(void){return team.active;}
int ds41f_team_thread_id(void){return omp_get_thread_num();}
static void relax(void)
{
#if defined(__aarch64__)
    __asm__ volatile("yield" ::: "memory");
#else
    atomic_signal_fence(memory_order_acq_rel);
#endif
}
static void execute(int id)
{
    size_t n=team.count,p=(size_t)team.threads;
    size_t step=n/p,rest=n%p;
    size_t begin=(size_t)id*step+((size_t)id<rest?(size_t)id:rest);
    size_t end=begin+step+((size_t)id<rest);
    if(begin<end)team.work(team.context,begin,end);
}
int ds41f_team_for(size_t count,ds41f_team_work work,void *context)
{
    if(!work)return EINVAL;
    if(!count)return 0;
    if(!team.active){
        #pragma omp parallel
        {size_t p=(size_t)omp_get_num_threads(),id=(size_t)omp_get_thread_num();
            size_t step=count/p,rest=count%p,begin=id*step+(id<rest?id:rest);
            size_t end=begin+step+(id<rest);if(begin<end)work(context,begin,end);}
        return 0;
    }
    if(omp_get_thread_num()!=0)return EPERM;
    team.work=work;team.context=context;team.count=count;
    uint64_t epoch=atomic_load_explicit(&team.epoch,memory_order_relaxed)+1;
    atomic_store_explicit(&team.epoch,epoch,memory_order_release);
    execute(0);
    for(int id=1;id<team.threads;++id)
        while(atomic_load_explicit(&team.done[id].epoch,memory_order_acquire)!=epoch)relax();
    return 0;
}
int ds41f_team_run(ds41f_team_body body,void *context)
{
    if(!body||team.active||omp_in_parallel())return EINVAL;
    int max_threads=omp_get_max_threads();if(max_threads<1||max_threads>48)return EINVAL;
    void *memory=NULL;if(posix_memalign(&memory,256,(size_t)max_threads*sizeof(completion)))return ENOMEM;
    team.done=memory;memset(team.done,0,(size_t)max_threads*sizeof(completion));
    atomic_init(&team.epoch,0);
    #pragma omp parallel
    {
        int id=omp_get_thread_num();atomic_init(&team.done[id].epoch,0);
        #pragma omp master
        {team.threads=omp_get_num_threads();team.active=1;}
        #pragma omp barrier
        if(id==0){body(context);team.work=NULL;
            atomic_fetch_add_explicit(&team.epoch,1,memory_order_release);
        }else{uint64_t previous=0;
            for(;;){uint64_t epoch;
                do{epoch=atomic_load_explicit(&team.epoch,memory_order_acquire);if(epoch==previous)relax();}while(epoch==previous);
                if(!team.work)break;
                execute(id);previous=epoch;
                atomic_store_explicit(&team.done[id].epoch,epoch,memory_order_release);
            }
        }
    }
    team.active=0;free(team.done);team.done=NULL;return 0;
}

#endif
