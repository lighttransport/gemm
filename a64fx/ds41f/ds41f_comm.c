#define _GNU_SOURCE
#include "ds41f_comm.h"
#include <mpi.h>
#include <utofu.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif
/* The shared single-header transport defines entry points unused here. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "../utofu-tests/tp_allreduce.h"
#pragma GCC diagnostic pop
static tp_comm comm;
static utofu_vcq_hdl_t vcq;
static int my_rank;
static int mpi_broadcast,dense_tp=1,shared_tp=1;
static MPI_Comm dense_comm=MPI_COMM_NULL,shared_comm=MPI_COMM_NULL;
static void bootstrap_barrier(void){MPI_Barrier(MPI_COMM_WORLD);}
void ds41f_comm_ready(void){bootstrap_barrier();}
void ds41f_comm_use_mpi_broadcast(int enabled){mpi_broadcast=!!enabled;}
void ds41f_comm_abort(const char *message,int error)
{fprintf(stderr,"DS41F_ABORT rank=%d %s error=%d\n",my_rank,message,error);fflush(stderr);MPI_Abort(MPI_COMM_WORLD,error?error:1);exit(1);}
int ds41f_comm_init(int *argc,char ***argv,int *rank,int *ranks)
{
    int provided=0;MPI_Init_thread(argc,argv,MPI_THREAD_FUNNELED,&provided);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);MPI_Comm_size(MPI_COMM_WORLD,ranks);*rank=my_rank;
    if(provided<MPI_THREAD_FUNNELED)ds41f_comm_abort("MPI_THREAD_FUNNELED required",1);
    if(*ranks!=12)ds41f_comm_abort("requires one rank on each of 12 nodes",1);
    uint8_t coords[6],all[12][6];int rc=utofu_query_my_coords(coords);
    if(rc)ds41f_comm_abort("query coordinates",rc);
    MPI_Allgather(coords,6,MPI_BYTE,all,6,MPI_BYTE,MPI_COMM_WORLD);
    for(int i=0;i<12;++i)for(int j=0;j<i;++j)if(!memcmp(all[i],all[j],6))ds41f_comm_abort("duplicate node placement",1);
    utofu_tni_id_t *tnis=NULL;size_t count=0;rc=utofu_get_onesided_tnis(&tnis,&count);
    if(rc||!count)ds41f_comm_abort("query TNIs",rc);
    rc=utofu_create_vcq_with_cmp_id(tnis[0],0,0,&vcq);free(tnis);if(rc)ds41f_comm_abort("create VCQ",rc);
    utofu_vcq_id_t self,peers[12];rc=utofu_query_vcq_id(vcq,&self);if(rc)ds41f_comm_abort("VCQ id",rc);
    MPI_Allgather(&self,sizeof self,MPI_BYTE,peers,sizeof self,MPI_BYTE,MPI_COMM_WORLD);
    for(int i=0;i<12;++i){rc=utofu_set_vcq_id_path(&peers[i],NULL);if(rc)ds41f_comm_abort("VCQ path",rc);}
    tp_comm_config options={0};options.robust=1;options.poll_spins=8;options.timeout=120;
    /* A prefetch worker may preempt a receiver after its trailer arrives.
     * Confirm consumption before reusing the recursive-doubling receive slot. */
    options.ack=1;options.ack_retx=64;options.ack_rtt=0.001;
    rc=tp_comm_init_region_ex(&comm,vcq,peers,my_rank,12,32768,bootstrap_barrier,TP_AR_STAG,&options,NULL,0);
    if(rc)ds41f_comm_abort("comm registration",rc);return 0;
}
void ds41f_comm_sum(float *v,size_t n)
{for(size_t i=0;i<n;i+=32768){size_t count=n-i;if(count>32768)count=32768;tp_allreduce_sum(&comm,v+i,(int)count);}}
void ds41f_comm_broadcast(float *v,size_t n,int owner)
{
    if(!mpi_broadcast){if(my_rank!=owner)memset(v,0,n*sizeof(float));ds41f_comm_sum(v,n);return;}
    /* The sum-based broadcast normalizes signed zero. Keep that behavior
     * when copying the owner's values without arithmetic. */
    if(my_rank==owner)for(size_t i=0;i<n;++i)if(v[i]==0.f)v[i]=0.f;
    for(size_t i=0;i<n;i+=32768){size_t count=n-i;if(count>32768)count=32768;
        int rc=MPI_Bcast(v+i,(int)count,MPI_FLOAT,owner,MPI_COMM_WORLD);
        if(rc)ds41f_comm_abort("MPI_Bcast",rc);}
}
void ds41f_comm_bytes(void *v,size_t n,int owner)
{
    if(owner<0||owner>=12||n>1048576)ds41f_comm_abort("byte broadcast bounds",EINVAL);
    int rc=MPI_Bcast(v,(int)n,MPI_BYTE,owner,MPI_COMM_WORLD);
    if(rc)ds41f_comm_abort("byte broadcast",rc);
}
static void pack_bf16(uint16_t *wire,float *v,size_t n,size_t tail)
{
    size_t i=0;
    #if defined(__ARM_FEATURE_SVE)
    for(;i<n;i+=svcntw()){
        svbool_t pg=svwhilelt_b32(i,n);svfloat32_t f=svld1_f32(pg,v+i);
        f=svsel_f32(svcmpeq_n_f32(pg,f,0),svdup_f32(0),f);
        svuint32_t bits=svreinterpret_u32_f32(f);
        if(svptest_any(pg,svcmpne_n_u32(pg,svand_n_u32_x(pg,bits,65535),0)))
            ds41f_comm_abort("non-BF16 transport input",EINVAL);
        svst1_f32(pg,v+i,f);svst1h_u32(pg,wire+i,svlsr_n_u32_x(pg,bits,16));
    }
    #else
    for(;i<n;++i){uint32_t bits;if(v[i]==0.f)v[i]=0.f;memcpy(&bits,v+i,4);
        if(bits&65535)ds41f_comm_abort("non-BF16 transport input",EINVAL);
        wire[i]=(uint16_t)(bits>>16);}
    #endif
    for(i=0;i<tail;++i)if(v[n+i]==0.f)v[n+i]=0.f;
    memcpy(wire+n,v+n,tail*sizeof(float));
}
static void unpack_bf16(float *v,const uint16_t *wire,size_t n,size_t tail)
{
    size_t i=0;
    #if defined(__ARM_FEATURE_SVE)
    for(;i<n;i+=svcntw()){
        svbool_t pg=svwhilelt_b32(i,n);
        svst1_f32(pg,v+i,svreinterpret_f32_u32(svlsl_n_u32_x(pg,svld1uh_u32(pg,wire+i),16)));}
    #else
    for(;i<n;++i){uint32_t bits=(uint32_t)wire[i]<<16;memcpy(v+i,&bits,4);}
    #endif
    memcpy(v+n,wire+n,tail*sizeof(float));
}
static void bf16_bounds(size_t n,size_t tail,int owner,int next)
{if(n>20480||tail>12||owner<0||owner>=12||next<0||next>=12)
    ds41f_comm_abort("BF16 transport bounds",EINVAL);}
void ds41f_comm_bf16_broadcast(float *v,size_t n,size_t tail,int owner)
{
    bf16_bounds(n,tail,owner,owner);uint16_t wire[20480+24];
    if(my_rank==owner)pack_bf16(wire,v,n,tail);
    ds41f_comm_bytes(wire,n*2+tail*4,owner);
    if(my_rank!=owner)unpack_bf16(v,wire,n,tail);
}
void ds41f_comm_bf16_handoff(float *v,size_t n,size_t tail,int owner,int next)
{
    bf16_bounds(n,tail,owner,next);uint16_t wire[20480+24];int rc=0;
    /* Fixed tag is safe: a single main thread issues every handoff in layer
     * order, and each source/destination pair is FIFO. Nonparticipants may
     * enter the following collective while the next dense owner receives. */
    if(my_rank==owner){pack_bf16(wire,v,n,tail);
        if(next!=owner)rc=MPI_Send(wire,(int)(n*2+tail*4),MPI_BYTE,next,41,MPI_COMM_WORLD);}
    else if(my_rank==next){rc=MPI_Recv(wire,(int)(n*2+tail*4),MPI_BYTE,owner,41,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        if(!rc)unpack_bf16(v,wire,n,tail);}
    if(rc)ds41f_comm_abort("BF16 residual handoff",rc);
}
static void bf16_batch_bounds(size_t stride,size_t n,size_t tail,size_t batch,int owner,int next)
{
    bf16_bounds(n,tail,owner,next);
    if(!batch||batch>6||stride<n+tail||stride>SIZE_MAX/sizeof(float)/batch)
        ds41f_comm_abort("BF16 batch bounds",EINVAL);
}
void ds41f_comm_bf16_broadcast_batch(float *v,size_t stride,size_t n,size_t tail,size_t batch,int owner)
{
    bf16_batch_bounds(stride,n,tail,batch,owner,owner);uint16_t wire[6*(20480+24)];size_t step=n+tail*2;
    if(my_rank==owner)for(size_t t=0;t<batch;++t)pack_bf16(wire+t*step,v+t*stride,n,tail);
    ds41f_comm_bytes(wire,batch*step*2,owner);
    if(my_rank!=owner)for(size_t t=0;t<batch;++t)unpack_bf16(v+t*stride,wire+t*step,n,tail);
}
void ds41f_comm_bf16_handoff_batch(float *v,size_t stride,size_t n,size_t tail,size_t batch,int owner,int next)
{
    bf16_batch_bounds(stride,n,tail,batch,owner,next);uint16_t wire[6*(20480+24)];size_t step=n+tail*2;int rc=0;
    if(my_rank==owner){for(size_t t=0;t<batch;++t)pack_bf16(wire+t*step,v+t*stride,n,tail);
        if(next!=owner)rc=MPI_Send(wire,(int)(batch*step*2),MPI_BYTE,next,42,MPI_COMM_WORLD);}
    else if(my_rank==next){rc=MPI_Recv(wire,(int)(batch*step*2),MPI_BYTE,owner,42,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        if(!rc)for(size_t t=0;t<batch;++t)unpack_bf16(v+t*stride,wire+t*step,n,tail);}
    if(rc)ds41f_comm_abort("BF16 batch handoff",rc);
}
void ds41f_comm_set_tp(int tp)
{
    if((tp!=1&&tp!=2&&tp!=4)||dense_comm!=MPI_COMM_NULL)ds41f_comm_abort("TP configuration",EINVAL);
    dense_tp=tp;int rc=MPI_Comm_split(MPI_COMM_WORLD,my_rank/tp,my_rank,&dense_comm);
    if(rc)ds41f_comm_abort("TP communicator",rc);
}
void ds41f_comm_tp_bytes(void *v,size_t n,int owner)
{
    if(dense_comm==MPI_COMM_NULL||owner<0||owner>=12||owner/dense_tp!=my_rank/dense_tp||n>1048576)
        ds41f_comm_abort("TP broadcast bounds",EINVAL);
    int rc=MPI_Bcast(v,(int)n,MPI_BYTE,owner%dense_tp,dense_comm);
    if(rc)ds41f_comm_abort("TP broadcast",rc);
}
static void tp_collect(float *out,float *part,size_t n,int owner,int all)
{
    if(dense_comm==MPI_COMM_NULL||n*(size_t)dense_tp>8192||owner<0||owner>=12||owner/dense_tp!=my_rank/dense_tp)
        ds41f_comm_abort("TP gather bounds",EINVAL);
    uint16_t send[8192],recv[8192];pack_bf16(send,part,n,0);
    int rc=all?MPI_Allgather(send,(int)n*2,MPI_BYTE,recv,(int)n*2,MPI_BYTE,dense_comm):
        MPI_Gather(send,(int)n*2,MPI_BYTE,recv,(int)n*2,MPI_BYTE,owner%dense_tp,dense_comm);
    if(rc)ds41f_comm_abort("TP gather",rc);
    if(all||my_rank==owner)unpack_bf16(out,recv,n*(size_t)dense_tp,0);
}
void ds41f_comm_tp_allgather(float *out,float *part,size_t n)
{tp_collect(out,part,n,my_rank,1);}
void ds41f_comm_tp_gather(float *out,float *part,size_t n,int owner)
{tp_collect(out,part,n,owner,0);}
void ds41f_comm_tp_gather_batch(float *out,size_t os,float *part,size_t ps,size_t n,size_t batch,int owner,int all)
{
    if(dense_comm==MPI_COMM_NULL||owner<0||owner>=12||owner/dense_tp!=my_rank/dense_tp||
       !batch||batch>6||n>8192/(size_t)dense_tp||ps<n||os<n*(size_t)dense_tp||
       ps>SIZE_MAX/sizeof(float)/batch||os>SIZE_MAX/sizeof(float)/batch)
        ds41f_comm_abort("TP batch gather bounds",EINVAL);
    uint16_t send[6*8192],recv[6*8192];
    for(size_t t=0;t<batch;++t)pack_bf16(send+t*n,part+t*ps,n,0);
    int rc=all?MPI_Allgather(send,(int)(batch*n*2),MPI_BYTE,recv,(int)(batch*n*2),MPI_BYTE,dense_comm):
        MPI_Gather(send,(int)(batch*n*2),MPI_BYTE,recv,(int)(batch*n*2),MPI_BYTE,owner%dense_tp,dense_comm);
    if(rc)ds41f_comm_abort("TP batch gather",rc);
    if(all||my_rank==owner)for(size_t t=0;t<batch;++t)for(int r=0;r<dense_tp;++r)
        unpack_bf16(out+t*os+(size_t)r*n,recv+((size_t)r*batch+t)*n,n,0);
}
void ds41f_comm_argmax(float *value,int *index)
{
    struct {float value;int index;} in={*value,*index},out;
    int rc=MPI_Allreduce(&in,&out,1,MPI_FLOAT_INT,MPI_MAXLOC,MPI_COMM_WORLD);
    if(rc)ds41f_comm_abort("head argmax",rc);*value=out.value;*index=out.index;
}
void ds41f_comm_head_logits(float *out,const float *part,size_t n)
{
    int counts[12],offsets[12];
    for(int r=0;r<12;++r){offsets[r]=(4040*r/12)*32;counts[r]=(4040*(r+1)/12)*32-offsets[r];}
    if(n!=(size_t)counts[my_rank])ds41f_comm_abort("head shard geometry",EINVAL);
    int rc=MPI_Gatherv(part,(int)n,MPI_FLOAT,out,counts,offsets,MPI_FLOAT,11,MPI_COMM_WORLD);
    if(rc)ds41f_comm_abort("head logits gather",rc);
}
void ds41f_comm_set_shared_tp(int tp)
{
    if(tp!=1&&tp!=4&&tp!=12)ds41f_comm_abort("shared TP must be 1,4,12",EINVAL);
    if(shared_comm!=MPI_COMM_NULL){MPI_Comm_free(&shared_comm);shared_comm=MPI_COMM_NULL;}
    shared_tp=tp;
    if(tp>1){
        int rc=MPI_Comm_split(MPI_COMM_WORLD,my_rank/tp,my_rank,&shared_comm);
        if(rc)ds41f_comm_abort("MPI shared communicator",rc);
    }
}
int ds41f_comm_shared_member(int owner)
{
    if(owner<0||owner>=12)ds41f_comm_abort("shared owner bounds",EINVAL);
    return shared_tp==1?my_rank==owner:my_rank/shared_tp==owner/shared_tp;
}
void ds41f_comm_shared_range_aligned(size_t global_count,size_t alignment,size_t *first,size_t *count)
{
    if(!first||!count||!global_count||!alignment||shared_tp<1||global_count>INT_MAX||global_count%alignment)
        ds41f_comm_abort("shared range bounds",EINVAL);
    int local=shared_tp==1?0:my_rank%shared_tp;
    size_t blocks=global_count/alignment;
    /* Match stage_tp.py's floor-boundary partition.  This keeps the runtime
     * row_start/rows contract identical to every staged weight, including
     * the 160-block W2 tensor split over 12 ranks. */
    size_t first_block=blocks*(size_t)local/(size_t)shared_tp;
    size_t end_block=blocks*(size_t)(local+1)/(size_t)shared_tp;
    *first=first_block*alignment;*count=(end_block-first_block)*alignment;
}
void ds41f_comm_shared_range(size_t global_count,size_t *first,size_t *count)
{ds41f_comm_shared_range_aligned(global_count,1,first,count);}
static void shared_bounds(int owner,size_t global_count,size_t local_count,size_t alignment)
{
    if(shared_tp<=1||shared_comm==MPI_COMM_NULL||owner<0||owner>=12||
       !ds41f_comm_shared_member(owner)||global_count>8192||local_count>global_count)
        ds41f_comm_abort("shared collective bounds",EINVAL);
    size_t first,expected;ds41f_comm_shared_range_aligned(global_count,alignment,&first,&expected);
    (void)first;
    if(local_count!=expected)ds41f_comm_abort("shared shard range",EINVAL);
}
void ds41f_comm_shared_allgather(float *out,const float *part,size_t count)
{
    if(!out||!part||shared_tp<=1||shared_comm==MPI_COMM_NULL||!count||
       count>(size_t)4096||count>(size_t)INT_MAX/(size_t)shared_tp)
        ds41f_comm_abort("shared allgather bounds",EINVAL);
    uint16_t send[4096],recv[49152];pack_bf16(send,(float *)part,count,0);
    int rc=MPI_Allgather(send,(int)(count*2),MPI_BYTE,recv,(int)(count*2),MPI_BYTE,shared_comm);
    if(rc)ds41f_comm_abort("shared allgather",rc);
    unpack_bf16(out,recv,count*(size_t)shared_tp,0);
}
void ds41f_comm_shared_gather(float *out,const float *part,size_t count,
                              int owner,size_t global_count)
{ds41f_comm_shared_gather_aligned(out,part,count,owner,global_count,1);}
void ds41f_comm_shared_gather_aligned(float *out,const float *part,size_t count,
                                      int owner,size_t global_count,size_t alignment)
{
    if(!out||!part||global_count>8192||!alignment||global_count%alignment)
        ds41f_comm_abort("shared gather bounds",EINVAL);
    if(shared_tp==1){if(my_rank==owner)memcpy(out,part,global_count*sizeof(float));return;}
    shared_bounds(owner,global_count,count,alignment);
    uint16_t send[8192],recv[8192];int counts[12]={0},displs[12]={0};
    for(int r=0;r<shared_tp;++r){size_t blocks=global_count/alignment;
        size_t begin=blocks*(size_t)r/(size_t)shared_tp*alignment;
        size_t end=blocks*(size_t)(r+1)/(size_t)shared_tp*alignment;
        size_t rows=end-begin;
        counts[r]=(int)(rows*2);displs[r]=(int)(begin*2);}
    pack_bf16(send,(float *)part,count,0);
    int rc=MPI_Gatherv(send,(int)(count*2),MPI_BYTE,recv,counts,displs,
                       MPI_BYTE,owner%shared_tp,shared_comm);
    if(rc)ds41f_comm_abort("shared gather",rc);
    if(my_rank/shared_tp==owner/shared_tp&&my_rank%shared_tp==owner%shared_tp)
        unpack_bf16(out,recv,global_count,0);
}
void ds41f_comm_shared_reduce_scatter_aligned(float *out,const float *in,size_t global_count,size_t alignment)
{
    if(!out||!in||shared_tp<=1||shared_comm==MPI_COMM_NULL||global_count>INT_MAX||!alignment||global_count%alignment)
        ds41f_comm_abort("shared reduce-scatter bounds",EINVAL);
    int counts[12]={0};size_t blocks=global_count/alignment;
    for(int r=0;r<shared_tp;++r){size_t begin=blocks*(size_t)r/(size_t)shared_tp;
        size_t end=blocks*(size_t)(r+1)/(size_t)shared_tp;
        size_t rows=(end-begin)*alignment;
        if(rows>(size_t)INT_MAX)ds41f_comm_abort("shared reduce-scatter count",EINVAL);
        counts[r]=(int)rows;
    }
    int rc=MPI_Reduce_scatter(in,out,counts,MPI_FLOAT,MPI_SUM,shared_comm);
    if(rc)ds41f_comm_abort("shared reduce-scatter",rc);
}
void ds41f_comm_shared_reduce_scatter(float *out,const float *in,size_t global_count)
{ds41f_comm_shared_reduce_scatter_aligned(out,in,global_count,1);}
void ds41f_comm_free(void)
{bootstrap_barrier();if(shared_comm!=MPI_COMM_NULL)MPI_Comm_free(&shared_comm);
    if(dense_comm!=MPI_COMM_NULL)MPI_Comm_free(&dense_comm);tp_comm_free(&comm);utofu_free_vcq(vcq);MPI_Finalize();}
