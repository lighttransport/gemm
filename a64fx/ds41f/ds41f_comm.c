#define _GNU_SOURCE
#include "ds41f_comm.h"
#include <mpi.h>
#include <utofu.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../utofu-tests/tp_allreduce.h"
static tp_comm comm;
static utofu_vcq_hdl_t vcq;
static int my_rank;
static void bootstrap_barrier(void){MPI_Barrier(MPI_COMM_WORLD);}
void ds41f_comm_ready(void){bootstrap_barrier();}
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
{if(my_rank!=owner)memset(v,0,n*sizeof(float));ds41f_comm_sum(v,n);}
void ds41f_comm_free(void)
{bootstrap_barrier();tp_comm_free(&comm);utofu_free_vcq(vcq);MPI_Finalize();}
