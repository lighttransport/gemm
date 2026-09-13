#define _POSIX_C_SOURCE 200809L
#include "ds41f_team.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
typedef struct {size_t a[1025],generation;int failed;} state;
static void write_range(void *p,size_t first,size_t last)
{state *s=p;for(size_t i=first;i<last;++i)s->a[i]=s->generation+i;}
static void empty(void *p,size_t first,size_t last){(void)p;(void)first;(void)last;}
static void body(void *p)
{
    state *s=p;if(omp_get_thread_num()!=0)s->failed=1;
    for(size_t n=1;n<=1024;n=n<128?n*2: n+127){s->a[n]=12345;
        for(size_t it=0;it<20;++it){s->generation=it*1024;
            if(ds41f_team_for(n,write_range,s))s->failed=1;
            for(size_t i=0;i<n;++i)if(s->a[i]!=s->generation+i)s->failed=1;
            if(s->a[n]!=12345)s->failed=1;}}
    double start=now();for(int i=0;i<10000;++i)if(ds41f_team_for(48,empty,NULL))s->failed=1;
    printf("TEAM persistent=%d mean_dispatch_us=%.3f\n",omp_in_parallel(),(now()-start)*100);
}
int main(void)
{state s={.generation=0};body(&s);if(ds41f_team_run(body,&s))s.failed=1;
    if(s.failed){puts("TEAM FAIL");return 1;}puts("TEAM PASS repeated_dispatch partition canaries main_thread");return 0;}
