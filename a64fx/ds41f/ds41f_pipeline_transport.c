#define _GNU_SOURCE
#include "ds41f_pipeline_transport.h"
#include <mpi.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

static int boundary_for_send(const ds41f_pipeline *p, int *src, int *dst)
{ return ds41f_pipeline_boundary(p->stage, src, dst); }
static int boundary_for_receive(const ds41f_pipeline *p, int *src, int *dst)
{ return ds41f_pipeline_boundary(p->stage-1, src, dst); }
static int participant(int rank, int src, int dst){return rank==src||rank==dst;}
static int tag_for(int boundary,int slot){return 80+boundary*2+slot;}
static int envelope_bounds(size_t tile,size_t pos,size_t count)
{ return tile>UINT32_MAX||pos>UINT32_MAX||count>UINT32_MAX||count>DS41F_PIPELINE_TILE_MAX?EINVAL:0; }

static int post_receive(void *opaque,const ds41f_pipeline *p,size_t tile,size_t pos,size_t count,int slot)
{
    ds41f_pipeline_transport_context *c=opaque; int src,dst;
    if(!c||slot<0||slot>1||envelope_bounds(tile,pos,count)||boundary_for_receive(p,&src,&dst))return EINVAL;
    if(!participant(c->rank,src,dst))return 0;
    if(c->rank!=dst||!c->recv_request[slot]||!c->recv_wire[slot]||c->recv_tile[slot]!=SIZE_MAX)return EBUSY;
    MPI_Request *r=c->recv_request[slot];
    int rc=MPI_Irecv(c->recv_wire[slot],(int)ds41f_pipeline_envelope_bytes(),MPI_BYTE,src,
                     tag_for(p->stage-1,slot),MPI_COMM_WORLD,r);
    if(rc)return rc;
    c->recv_tile[slot]=tile;c->recv_position[slot]=pos;c->recv_count[slot]=count;return 0;
}
static int wait_receive(void *opaque,const ds41f_pipeline *p,size_t tile,int slot)
{
    ds41f_pipeline_transport_context *c=opaque;int src,dst;
    if(!c||slot<0||slot>1||boundary_for_receive(p,&src,&dst))return EINVAL;
    if(!participant(c->rank,src,dst))return 0;
    if(c->rank!=dst||c->recv_tile[slot]!=tile)return EINVAL;
    MPI_Request *r=c->recv_request[slot];int rc=MPI_Wait(r,MPI_STATUS_IGNORE);if(rc)return rc;
    size_t tile_base=c->recv_tile[slot]*c->residual_tile_stride;
    float *residual=c->recv_residual+tile_base;
    uint8_t *metadata=c->recv_metadata+c->recv_tile[slot]*c->metadata_tile_stride;
    rc=ds41f_pipeline_envelope_unpack(c->recv_wire[slot],ds41f_pipeline_envelope_bytes(),
        (uint32_t)tile,(uint32_t)c->recv_position[slot],(uint32_t)c->recv_count[slot],residual,
        c->residual_token_stride,(ds41f_pipeline_wire *)metadata,c->metadata_token_stride);
    c->recv_tile[slot]=SIZE_MAX;return rc;
}
static int post_send(void *opaque,const ds41f_pipeline *p,size_t tile,size_t pos,size_t count,int slot)
{
    ds41f_pipeline_transport_context *c=opaque;int src,dst;
    if(!c||slot<0||slot>1||envelope_bounds(tile,pos,count)||boundary_for_send(p,&src,&dst))return EINVAL;
    if(!participant(c->rank,src,dst))return 0;
    if(c->rank!=src||!c->send_request[slot]||!c->send_wire[slot])return EINVAL;
    size_t tile_base=tile*c->residual_tile_stride;
    int rc=ds41f_pipeline_envelope_pack(c->send_wire[slot],ds41f_pipeline_envelope_bytes(),
        (uint32_t)tile,(uint32_t)pos,(uint32_t)count,c->send_residual+tile_base,
        c->residual_token_stride,(const ds41f_pipeline_wire *)(c->send_metadata+tile*c->metadata_tile_stride),
        c->metadata_token_stride);
    if(rc)return rc;
    return MPI_Isend(c->send_wire[slot],(int)ds41f_pipeline_envelope_bytes(),MPI_BYTE,dst,
                     tag_for(p->stage,slot),MPI_COMM_WORLD,(MPI_Request *)c->send_request[slot]);
}
static int wait_send(void *opaque,const ds41f_pipeline *p,size_t tile,int slot)
{
    ds41f_pipeline_transport_context *c=opaque;int src,dst;
    if(!c||slot<0||slot>1||boundary_for_send(p,&src,&dst))return EINVAL;
    if(!participant(c->rank,src,dst))return 0;
    if(c->rank!=src)return 0;
    return MPI_Wait((MPI_Request *)c->send_request[slot],MPI_STATUS_IGNORE);
}

int ds41f_pipeline_transport_init(ds41f_pipeline_transport_context *c,int rank,
                                  float *sr,size_t sts,size_t rts,uint8_t *sm,size_t smts,size_t mts,
                                  float *rr,size_t rrs,size_t rrt,uint8_t *rm,size_t rmts,size_t rmt)
{
    if(!c||rank<0||rank>=12||!sr||!sm||!rr||!rm||!sts||!smts||!rrs||!rmts||sts!=rrs||smts!=rmts||rts!=rrt||mts!=rmt||rts<20484||rrt<20484||mts<sizeof(ds41f_pipeline_wire)||rmt<sizeof(ds41f_pipeline_wire))return EINVAL;
    memset(c,0,sizeof *c);c->rank=rank;c->send_residual=sr;c->residual_tile_stride=sts;c->residual_token_stride=rts;c->send_metadata=sm;c->metadata_tile_stride=smts;c->metadata_token_stride=mts;c->recv_residual=rr;c->recv_metadata=rm;
    c->recv_tile[0]=c->recv_tile[1]=SIZE_MAX;
    size_t bytes=ds41f_pipeline_envelope_bytes();
    for(int i=0;i<2;++i){c->send_wire[i]=malloc(bytes);c->recv_wire[i]=malloc(bytes);c->send_request[i]=calloc(1,sizeof(MPI_Request));c->recv_request[i]=calloc(1,sizeof(MPI_Request));if(!c->send_wire[i]||!c->recv_wire[i]||!c->send_request[i]||!c->recv_request[i]){ds41f_pipeline_transport_destroy(c);return ENOMEM;}*(MPI_Request *)c->send_request[i]=MPI_REQUEST_NULL;*(MPI_Request *)c->recv_request[i]=MPI_REQUEST_NULL;}
    c->recv_residual=rr;c->recv_metadata=rm;return 0;
}
void ds41f_pipeline_transport_destroy(ds41f_pipeline_transport_context *c){if(!c)return;for(int i=0;i<2;++i){free(c->send_wire[i]);free(c->recv_wire[i]);free(c->send_request[i]);free(c->recv_request[i]);}memset(c,0,sizeof *c);}
void ds41f_pipeline_transport_bind(ds41f_pipeline_transport *t,ds41f_pipeline_transport_context *c){if(!t)return;memset(t,0,sizeof *t);t->post_receive=post_receive;t->wait_receive=wait_receive;t->post_send=post_send;t->wait_send=wait_send;t->context=c;}
