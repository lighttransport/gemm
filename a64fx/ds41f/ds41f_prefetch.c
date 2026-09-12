#define _POSIX_C_SOURCE 200809L
#include "ds41f_prefetch.h"
#include "ds41f_kernels.h"
#include <errno.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct ds41f_prefetch {
    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t request,complete;
    ds41f_engram *engram;
    int stop,pending,submitted,ready[2],error[2];
    uint64_t ids[2][24];
    float rows[2][24*256];
    double seconds[2];
};
static double now(void)
{struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void *worker(void *arg)
{
    ds41f_prefetch *p=arg;
    for(;;){
        pthread_mutex_lock(&p->mutex);
        while(!p->pending&&!p->stop)pthread_cond_wait(&p->request,&p->mutex);
        int stop=p->stop;pthread_mutex_unlock(&p->mutex);if(stop)break;
        for(int slot=0;slot<2;++slot){
            double start=now();int rc=0;uint16_t row[256];
            memset(p->rows[slot],0,sizeof p->rows[slot]);
            ds41f_engram_table *table=&p->engram->table[slot];
            for(int i=0;i<24;++i){uint64_t id=p->ids[slot][i];
                if(id<table->first||id>=table->first+table->owned_rows)continue;
                rc=ds41f_engram_read_local(p->engram,slot,id,row);if(rc)break;
                for(int j=0;j<256;++j)p->rows[slot][i*256+j]=ds41f_bf16_to_f32(row[j]);
            }
            pthread_mutex_lock(&p->mutex);
            p->seconds[slot]=now()-start;p->error[slot]=rc;p->ready[slot]=1;
            if(slot==1)p->pending=0;
            pthread_cond_broadcast(&p->complete);pthread_mutex_unlock(&p->mutex);
        }
    }
    return NULL;
}
int ds41f_prefetch_create(ds41f_prefetch **out,ds41f_engram *engram)
{
    if(!out||!engram)return EINVAL;
    *out=NULL;ds41f_prefetch *p=calloc(1,sizeof *p);if(!p)return ENOMEM;
    p->engram=engram;int rc=pthread_mutex_init(&p->mutex,NULL);if(rc){free(p);return rc;}
    rc=pthread_cond_init(&p->request,NULL);if(rc)goto mutex;
    rc=pthread_cond_init(&p->complete,NULL);if(rc)goto request;
    rc=pthread_create(&p->thread,NULL,worker,p);if(rc)goto complete;
    *out=p;return 0;
complete: pthread_cond_destroy(&p->complete);
request: pthread_cond_destroy(&p->request);
mutex: pthread_mutex_destroy(&p->mutex);free(p);return rc;
}
int ds41f_prefetch_submit(ds41f_prefetch *p,const uint64_t ids[2][24])
{
    if(!p||!ids)return EINVAL;
    pthread_mutex_lock(&p->mutex);
    if(p->pending){pthread_mutex_unlock(&p->mutex);return EBUSY;}
    memcpy(p->ids,ids,sizeof p->ids);p->pending=p->submitted=1;
    p->ready[0]=p->ready[1]=0;
    pthread_cond_signal(&p->request);pthread_mutex_unlock(&p->mutex);return 0;
}
int ds41f_prefetch_wait(ds41f_prefetch *p,int slot,float rows[24*256],double *read_seconds)
{
    if(!p||slot<0||slot>1||!rows)return EINVAL;
    pthread_mutex_lock(&p->mutex);
    if(!p->submitted){pthread_mutex_unlock(&p->mutex);return EINVAL;}
    while(!p->ready[slot])pthread_cond_wait(&p->complete,&p->mutex);
    int rc=p->error[slot];
    if(!rc)memcpy(rows,p->rows[slot],sizeof p->rows[slot]);
    if(read_seconds)*read_seconds=p->seconds[slot];
    pthread_mutex_unlock(&p->mutex);return rc;
}
void ds41f_prefetch_destroy(ds41f_prefetch *p)
{
    if(!p)return;
    pthread_mutex_lock(&p->mutex);p->stop=1;
    pthread_cond_signal(&p->request);pthread_mutex_unlock(&p->mutex);
    pthread_join(p->thread,NULL);pthread_cond_destroy(&p->complete);
    pthread_cond_destroy(&p->request);pthread_mutex_destroy(&p->mutex);free(p);
}
