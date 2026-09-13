#define _POSIX_C_SOURCE 200809L
#include "ds41f_comm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static float value(size_t i,int rank){return (float)(i%16)*(rank+1)*.5f;}
int main(int argc,char **argv)
{
    int rank,ranks;ds41f_comm_init(&argc,&argv,&rank,&ranks);
    int tp=argc>1?atoi(argv[1]):2;ds41f_comm_set_tp(tp);
    float part[8192],all[8193];unsigned char bytes[513];
    for(int owner=0;owner<12;++owner){
        if(rank/tp==owner/tp){
            memset(bytes,rank==owner?37:99,512);bytes[512]=13;
            if(rank!=owner){struct timespec pause={0,1000000};nanosleep(&pause,NULL);}
            ds41f_comm_tp_bytes(bytes,512,owner);
            for(size_t i=0;i<512;++i)if(bytes[i]!=37)ds41f_comm_abort("TP bytes",1);
            if(bytes[512]!=13)ds41f_comm_abort("TP byte canary",1);
            size_t sizes[]={0,17,2304/(size_t)tp,5120/(size_t)tp,8192/(size_t)tp};
            for(size_t shape=0;shape<5;++shape)for(int gather=0;gather<2;++gather){size_t n=sizes[shape];
                for(size_t i=0;i<n;++i)part[i]=value(i,rank);
                for(size_t i=0;i<n*(size_t)tp+1;++i)all[i]=123456;
                if(gather)ds41f_comm_tp_gather(all,part,n,owner);else ds41f_comm_tp_allgather(all,part,n);
                for(size_t i=0;i<n*(size_t)tp;++i){float expected=(!gather||rank==owner)?value(i%n,(owner/tp)*tp+(int)(i/n)):123456;
                    if(memcmp(all+i,&expected,4))ds41f_comm_abort("TP gather reconstruction",1);}
                if(all[n*(size_t)tp]!=123456)ds41f_comm_abort("TP gather canary",1);
            }
        }
        ds41f_comm_ready();
    }
    /* Exercise the independent TP12 communicator used by the shared FFN.
     * The W1/W3 shape is evenly divisible, while W2 is deliberately split in
     * 32-row FP8 scale groups to cover the uneven aligned path. */
    ds41f_comm_set_shared_tp(12);
    size_t shared_first,shared_count;
    ds41f_comm_shared_range(2304,&shared_first,&shared_count);
    if(shared_count!=192||shared_first!=192*(size_t)rank)
        ds41f_comm_abort("shared W1 range",1);
    for(size_t i=0;i<shared_count;++i)part[i]=(float)((shared_first+i)%16);
    for(size_t i=0;i<2305;++i)all[i]=123456;
    ds41f_comm_shared_allgather(all,part,shared_count);
    for(size_t i=0;i<2304;++i)if(all[i]!=(float)(i%16))
        ds41f_comm_abort("shared allgather",1);
    if(all[2304]!=123456)ds41f_comm_abort("shared allgather canary",1);
    ds41f_comm_shared_range_aligned(5120,32,&shared_first,&shared_count);
    for(size_t i=0;i<shared_count;++i)part[i]=(float)((rank+1)*.5f);
    for(size_t i=0;i<5121;++i)all[i]=123456;
    ds41f_comm_shared_gather_aligned(all,part,shared_count,0,5120,32);
    if(!rank){for(size_t i=0;i<5120;++i){int source=-1;
            for(int r=0;r<12;++r){size_t begin=(160*(size_t)r/12)*32;
                size_t end=(160*(size_t)(r+1)/12)*32;
                if(i>=begin&&i<end){source=r;break;}}
            if(source<0||all[i]!=(float)(source+1)*.5f)ds41f_comm_abort("shared aligned gather",1);}
        if(all[5120]!=123456)ds41f_comm_abort("shared aligned gather canary",1);}
    float reduce_in[5120],reduce_out[5120];
    for(size_t i=0;i<5120;++i)reduce_in[i]=(float)(rank+1);
    ds41f_comm_shared_reduce_scatter_aligned(reduce_out,reduce_in,5120,32);
    ds41f_comm_shared_range_aligned(5120,32,&shared_first,&shared_count);
    for(size_t i=0;i<shared_count;++i)if(reduce_out[i]!=78.f)
        ds41f_comm_abort("shared reduce-scatter",1);
    ds41f_comm_ready();
    for(int trial=0;trial<12;++trial){float score=rank==trial?6:5;int id=12-rank;
        ds41f_comm_argmax(&score,&id);if(score!=6||id!=12-trial)ds41f_comm_abort("distributed argmax",1);}
    float score=5;int id=12-rank;ds41f_comm_argmax(&score,&id);
    if(score!=5||id!=1)ds41f_comm_abort("argmax tie",1);
    size_t first=(4040*(size_t)rank/12)*32,end=(4040*(size_t)(rank+1)/12)*32;
    float *logits=malloc((end-first)*4),*full=rank==11?malloc((129280+1)*4):NULL;
    if(!logits||(rank==11&&!full))ds41f_comm_abort("head test allocation",1);
    for(size_t i=first;i<end;++i)logits[i-first]=(float)i*.001f;
    if(full)full[129280]=123456;
    ds41f_comm_head_logits(full,logits,end-first);
    if(full){for(size_t i=0;i<129280;++i)if(full[i]!=(float)i*.001f)ds41f_comm_abort("head reconstruction",1);
        if(full[129280]!=123456)ds41f_comm_abort("head canary",1);}
    free(full);free(logits);
    if(!rank)printf("TP_COMM PASS tp=%d owners=12 sizes=5 delay BF16 gather/allgather shared_tp12 aligned_reduce head_reconstruction argmax_ties canaries\n",tp);
    ds41f_comm_free();return 0;
}
