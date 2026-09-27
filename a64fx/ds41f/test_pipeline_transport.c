#include "ds41f_pipeline_transport.h"
#include <mpi.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc,char **argv)
{
    MPI_Init(&argc,&argv); int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    assert(size==12);
    const size_t tiles=2, token_stride=20484, tile_stride=DS41F_PIPELINE_TILE_MAX*20484;
    const size_t metadata_stride=sizeof(ds41f_pipeline_wire), metadata_tile_stride=DS41F_PIPELINE_TILE_MAX*metadata_stride;
    float *sr=calloc(tiles*tile_stride,sizeof(*sr)), *rr=calloc(tiles*tile_stride,sizeof(*rr));
    uint8_t *sm=calloc(tiles*metadata_tile_stride,1), *rm=calloc(tiles*metadata_tile_stride,1); assert(sr&&rr&&sm&&rm);
    for(size_t i=0;i<20480;++i){uint32_t bits=0x3f800000u+(uint32_t)(i%13)*0x10000u;memcpy(sr+i,&bits,4);}
    ds41f_pipeline_wire *mw=(ds41f_pipeline_wire *)sm; assert(!ds41f_pipeline_wire_init(mw,0,0,1,NULL,0,NULL,0,NULL,0));
    ds41f_pipeline_transport_context c; assert(!ds41f_pipeline_transport_init(&c,rank,sr,tile_stride,token_stride,sm,metadata_tile_stride,metadata_stride,rr,tile_stride,token_stride,rm,metadata_tile_stride,metadata_stride));
    ds41f_pipeline_transport t; ds41f_pipeline_transport_bind(&t,&c);
    ds41f_pipeline source={1,0,0,0,12,12,4,0}, target={1,5,1,13,26,12,4,0};
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank==0){assert(!t.post_send(t.context,&source,0,0,1,0));MPI_Barrier(MPI_COMM_WORLD);assert(!t.wait_send(t.context,&source,0,0));}
    else if(rank==5){MPI_Barrier(MPI_COMM_WORLD);assert(!t.post_receive(t.context,&target,0,0,1,0));assert(!t.wait_receive(t.context,&target,0,0));assert(memcmp(sr,rr,20480*sizeof(float))==0);}
    else MPI_Barrier(MPI_COMM_WORLD);
    ds41f_pipeline_transport_destroy(&c);free(sr);free(rr);free(sm);free(rm);if(rank==0)puts("PIPELINE_TRANSPORT PASS delayed_receiver envelope");MPI_Finalize();return 0;
}
