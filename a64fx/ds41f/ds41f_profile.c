#define _POSIX_C_SOURCE 200809L
#include "ds41f_profile.h"
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

_Thread_local double *ds41f_profile_current;
static double *records;
static size_t first,capacity,used;
static const size_t layers=41;

double ds41f_profile_clock(void)
{struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int ds41f_profile_init(size_t start,size_t count)
{
    if(records||count>4096)return EINVAL;
    first=start;capacity=count;used=0;
    if(!count)return 0;
    records=calloc(count*layers*DS41F_P_COUNT,sizeof(double));
    return records?0:ENOMEM;
}
void ds41f_profile_at(size_t position,int layer)
{
    ds41f_profile_current=NULL;
    if(!records||position<first||position-first>=capacity||layer<0||layer>40)return;
    size_t sample=position-first;
    if(sample+1>used)used=sample+1;
    ds41f_profile_current=records+(sample*layers+(size_t)layer)*DS41F_P_COUNT;
}
int ds41f_profile_write(int rank)
{
    ds41f_profile_current=NULL;
    if(!records)return 0;
    char path[128];snprintf(path,sizeof path,"profile.rank%02d.bin",rank);
    FILE *f=fopen(path,"wx");if(!f)return errno;
    size_t values=used*layers*DS41F_P_COUNT;
    int rc=fwrite(records,sizeof(double),values,f)==values?0:EIO;
    if(fclose(f))rc=EIO;
    if(rc)return rc;
    snprintf(path,sizeof path,"profile.rank%02d.json",rank);
    f=fopen(path,"wx");if(!f)return errno;
    fprintf(f,"{\"version\":1,\"rank\":%d,\"start\":%zu,\"positions\":%zu,\"layers\":%zu,"
              "\"dtype\":\"<f8\",\"layout\":\"position,layer,phase\",\"phases\":[",rank,first,used,layers);
    #define DS41F_PROFILE_NAME(name) #name,
    const char *names[]={DS41F_PROFILE_PHASES(DS41F_PROFILE_NAME)};
    #undef DS41F_PROFILE_NAME
    for(size_t i=0;i<DS41F_P_COUNT;++i)fprintf(f,"%s\"%s\"",i?",":"",names[i]);
    fputs("]}\n",f);
    rc=ferror(f)?EIO:0;if(fclose(f))rc=EIO;return rc;
}
void ds41f_profile_free(void)
{free(records);records=NULL;ds41f_profile_current=NULL;capacity=used=0;}
