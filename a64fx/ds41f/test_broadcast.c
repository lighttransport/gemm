#define _POSIX_C_SOURCE 200809L
#include "ds41f_comm.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static float value(size_t i,int owner)
{return i%17?(float)((int)(i%2048)-1024+owner)*.25f:-0.f;}
int main(int argc,char **argv)
{
    int rank,ranks;ds41f_comm_init(&argc,&argv,&rank,&ranks);
    const size_t sizes[]={0,1,3,513,5120,20484,32769};
    float *buffer=malloc(32770*sizeof(float));if(!buffer)ds41f_comm_abort("allocate",1);
    for(int mode=0;mode<2;++mode){ds41f_comm_use_mpi_broadcast(mode);
    for(int owner=0;owner<ranks;++owner)for(size_t shape=0;shape<7;++shape){
        size_t n=sizes[shape];for(size_t i=0;i<n;++i)buffer[i]=rank==owner?value(i,owner):123456.f;
        buffer[n]=98765.f;
        if(rank==(owner+1)%ranks&&(shape&1)){struct timespec delay={0,3000000};nanosleep(&delay,NULL);}
        ds41f_comm_broadcast(buffer,n,owner);
        for(size_t i=0;i<n;++i){float expected=value(i,owner);if(expected==0.f)expected=0.f;
            if(memcmp(buffer+i,&expected,4))ds41f_comm_abort("broadcast mismatch",1);}
        if(buffer[n]!=98765.f)ds41f_comm_abort("broadcast canary",1);
    }
    }
    free(buffer);if(!rank)puts("BROADCAST PASS modes=2 owners=12 sizes=7 chunk_boundary canary signed_zero delayed_receivers");
    ds41f_comm_free();return 0;
}
