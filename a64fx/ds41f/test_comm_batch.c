#define _POSIX_C_SOURCE 200809L
#include "ds41f_comm.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static float value(size_t i,size_t token,int owner)
{
    uint32_t bits=(uint32_t)((i*73+token*137+(size_t)owner*17)%65536)<<16;
    if((bits&0x7f800000)==0x7f800000||i%17==0)bits=0x80000000;
    float x;memcpy(&x,&bits,4);return x;
}
static void require(int ok,const char *s){if(!ok)ds41f_comm_abort(s,1);}
int main(int argc,char **argv)
{
    int rank,ranks;ds41f_comm_init(&argc,&argv,&rank,&ranks);int tp=argc==2?atoi(argv[1]):4;ds41f_comm_set_tp(tp);
    size_t ns[]={0,17,5120,20480},tails[]={4,0,12,4},max=6*20500;
    float *buffer=malloc(max*4),*part=malloc(max*4);require(buffer&&part,"batch buffers");
    for(int owner=0;owner<ranks;++owner)for(size_t batch=1;batch<=6;++batch)for(size_t shape=0;shape<4;++shape)for(int handoff=0;handoff<2;++handoff){
        size_t n=ns[shape],tail=tails[shape],stride=n+tail+7;int next=batch==1?owner:(owner+1)%ranks;
        for(size_t i=0;i<max;++i)buffer[i]=12345;
        if(rank==owner)for(size_t t=0;t<batch;++t){
            for(size_t i=0;i<n;++i)buffer[t*stride+i]=value(i,t,owner);
            for(size_t i=0;i<tail;++i)buffer[t*stride+n+i]=i?1.f/(float)(i+3+t):-0.f;
        }
        if(rank==next&&next!=owner&&(batch%2)){struct timespec delay={0,1000000};nanosleep(&delay,NULL);}
        if(handoff)ds41f_comm_bf16_handoff_batch(buffer,stride,n,tail,batch,owner,next);
        else ds41f_comm_bf16_broadcast_batch(buffer,stride,n,tail,batch,owner);
        for(size_t t=0;t<6;++t)for(size_t i=0;i<stride;++i){float expected=12345;
            if(t<batch&&i<n+tail&&(!handoff||rank==owner||rank==next)){
                expected=i<n?value(i,t,owner):i==n?-0.f:1.f/(float)(i-n+3+t);if(expected==0)expected=0;}
            require(!memcmp(buffer+t*stride+i,&expected,4),"batch packet value/stride/canary");}
    }
    size_t gather_ns[]={0,1,17,576,1280,8192/(size_t)tp};
    for(int owner=0;owner<12;++owner)if(owner/tp==rank/tp)
        for(size_t batch=1;batch<=6;++batch)for(size_t shape=0;shape<6;++shape)for(int all=0;all<2;++all){
            size_t n=gather_ns[shape],ps=n+7,os=n*tp+11;
            for(size_t i=0;i<max;++i)buffer[i]=part[i]=12345;
            for(size_t t=0;t<batch;++t)for(size_t i=0;i<n;++i)part[t*ps+i]=value(i,t,rank);
            if(rank==owner&&batch%2){struct timespec delay={0,1000000};nanosleep(&delay,NULL);}
            ds41f_comm_tp_gather_batch(buffer,os,part,ps,n,batch,owner,all);
            for(size_t t=0;t<6;++t)for(size_t i=0;i<os;++i){float expected=12345;
                if(t<batch&&i<n*(size_t)tp&&(all||rank==owner)){int source=rank/tp*tp+(int)(i/n);expected=value(i%n,t,source);if(expected==0)expected=0;}
                require(!memcmp(buffer+t*os+i,&expected,4),"batch TP order/stride/canary");}
            for(size_t t=0;t<batch;++t)for(size_t i=0;i<ps;++i){float expected=i<n?value(i,t,rank):12345;if(expected==0)expected=0;
                require(!memcmp(part+t*ps+i,&expected,4),"batch TP source/stride");}
        }
    ds41f_comm_ready();if(!rank)printf("COMM_BATCH PASS tp=%d owners=12 batches=1..6 broadcast handoff self TP_gather allgather zero_count signed_zero mixed_tail strides canaries delayed_receivers\n",tp);
    free(buffer);free(part);ds41f_comm_free();return 0;
}
