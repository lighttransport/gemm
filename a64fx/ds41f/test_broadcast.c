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
    for(int owner=0;owner<ranks;++owner)for(int mode=0;mode<3;++mode){
        size_t n=mode==0?17:mode==1?5120:20480,tail=mode==1?12:4;
        int next=mode==0?owner:(owner+1)%ranks;
        for(int kind=0;kind<2;++kind){
            for(size_t i=0;i<n;++i){uint32_t bits=(uint32_t)((i*73+(size_t)owner*17)%65536)<<16;
                if((bits&0x7f800000)==0x7f800000)bits=0x80000000;
                if(i%17==0)bits=0x80000000;
                if(rank==owner)memcpy(buffer+i,&bits,4);else buffer[i]=123456.f;}
            for(size_t i=0;i<tail;++i)buffer[n+i]=rank==owner?1.f/(float)(i+3):123456.f;
            buffer[n+tail]=98765.f;
            if(rank==next&&next!=owner){struct timespec delay={0,3000000};nanosleep(&delay,NULL);}
            if(kind)ds41f_comm_bf16_handoff(buffer,n,tail,owner,next);
            else ds41f_comm_bf16_broadcast(buffer,n,tail,owner);
            for(size_t i=0;i<n+tail;++i){float expected=123456.f;
                if(!kind||rank==owner||rank==next){
                    if(i<n){uint32_t bits=(uint32_t)((i*73+(size_t)owner*17)%65536)<<16;
                        if((bits&0x7f800000)==0x7f800000||i%17==0)bits=0x80000000;
                        memcpy(&expected,&bits,4);if(expected==0.f)expected=0.f;
                    }else expected=1.f/(float)(i-n+3);}
                if(memcmp(buffer+i,&expected,4))ds41f_comm_abort("BF16 mixed packet mismatch",1);}
            if(buffer[n+tail]!=98765.f)ds41f_comm_abort("BF16 packet canary",1);
        }
        unsigned char bytes[357];
        for(size_t i=0;i<356;++i)bytes[i]=rank==owner?(unsigned char)(i+owner):0;
        bytes[356]=99;ds41f_comm_bytes(bytes,356,owner);
        for(size_t i=0;i<356;++i)if(bytes[i]!=(unsigned char)(i+owner))ds41f_comm_abort("byte packet mismatch",1);
        if(bytes[356]!=99)ds41f_comm_abort("byte packet canary",1);
    }
    if(!rank)puts("COMPACT_COMM PASS owners=12 broadcast/handoff/self BF16+FP32 tails bytes nonparticipant canary signed_zero delayed_receivers");
    free(buffer);if(!rank)puts("BROADCAST PASS modes=2 owners=12 sizes=7 chunk_boundary canary signed_zero delayed_receivers");
    ds41f_comm_free();return 0;
}
