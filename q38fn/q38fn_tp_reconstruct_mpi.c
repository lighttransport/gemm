#define _GNU_SOURCE
#include <mpi.h>
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "../common/glm53f_safetensors.h"
#define Q38FN_TP_BLOB_IMPLEMENTATION
#include "../common/q38fn_tp_blob.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float bf16(uint16_t value)
{
    uint32_t bits=(uint32_t)value<<16;float result;memcpy(&result,&bits,4);return result;
}
static float dot(const uint16_t*w,const float*x,int n)
{
    float sum=0.0f;for(int i=0;i<n;++i)sum+=bf16(w[i])*x[i];return sum;
}
static int reference(const glm53f_st_context*ctx,const char*name,const float*x,
                     int rows,int cols,float*y,size_t element_offset)
{
    size_t count=(size_t)rows*cols;uint16_t*w=(uint16_t*)malloc(count*2);if(!w)return -1;
    if(glm53f_st_read(ctx,name,element_offset*2,w,count*2)){free(w);return -1;}
    for(int r=0;r<rows;++r)y[r]=dot(w+(size_t)r*cols,x,cols);
    free(w);return 0;
}
static float maximum_error(const float*a,const float*b,int n)
{
    float worst=0.0f;for(int i=0;i<n;++i){float e=fabsf(a[i]-b[i]);if(e>worst)worst=e;}return worst;
}
int main(int argc,char**argv)
{
    int rank,ranks,rc=1;MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    if(argc!=3||ranks!=Q38FN_TP_RANKS){if(!rank)fprintf(stderr,"usage: %s MODEL LOCAL_BASE\n",argv[0]);goto done;}
    char dir[4096];snprintf(dir,sizeof(dir),"%s/rank-%02d",argv[2],rank);
    q38fn_tp_blob blob;if(q38fn_tp_blob_open(&blob,dir,1)){fprintf(stderr,"rank %d blob load failed\n",rank);goto done;}
    const char*gu_name="model.language_model.layers.0.mlp.experts.gate_up_proj";
    const char*down_name="model.language_model.layers.0.mlp.experts.down_proj";
    const q38fn_tp_blob_entry*gu=q38fn_tp_blob_find(&blob,gu_name);
    const q38fn_tp_blob_entry*down=q38fn_tp_blob_find(&blob,down_name);
    int local_ok=gu&&down,all_ok=0;
    MPI_Allreduce(&local_ok,&all_ok,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
    if(!all_ok)goto close_blob;
    int local=(int)gu->range[0].count;
    int counts[Q38FN_TP_RANKS],displs[Q38FN_TP_RANKS];
    for(int r=0,off=0;r<ranks;++r){uint64_t s,c;q38fn_tp_split(Q38FN_EXPERT_INTERMEDIATE,r,ranks,&s,&c);counts[r]=(int)c;displs[r]=off;off+=(int)c;}
    float input[Q38FN_HIDDEN],local_gate[64],local_up[64];
    float gate[Q38FN_EXPERT_INTERMEDIATE],up[Q38FN_EXPERT_INTERMEDIATE];
    for(int i=0;i<Q38FN_HIDDEN;++i)input[i]=sinf((float)(i+1)*0.001f);
    const uint16_t*expert0=gu->data;
    for(int i=0;i<local;++i){local_gate[i]=dot(expert0+(size_t)i*Q38FN_HIDDEN,input,Q38FN_HIDDEN);local_up[i]=dot(expert0+(size_t)(local+i)*Q38FN_HIDDEN,input,Q38FN_HIDDEN);}
    MPI_Gatherv(local_gate,local,MPI_FLOAT,gate,counts,displs,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Gatherv(local_up,local,MPI_FLOAT,up,counts,displs,MPI_FLOAT,0,MPI_COMM_WORLD);
    float down_input[Q38FN_EXPERT_INTERMEDIATE],partial[Q38FN_HIDDEN],combined[Q38FN_HIDDEN];
    for(int i=0;i<Q38FN_EXPERT_INTERMEDIATE;++i)down_input[i]=cosf((float)(i+3)*0.002f);
    int down_cols=(int)down->range[0].count,down_start=(int)down->range[0].start;
    for(int row=0;row<Q38FN_HIDDEN;++row)partial[row]=dot(down->data+(size_t)row*down_cols,down_input+down_start,down_cols);
    MPI_Reduce(partial,combined,Q38FN_HIDDEN,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
    if(!rank){
        glm53f_st_context*ctx=glm53f_st_open(argv[1]);
        float ref_gu[2*Q38FN_EXPERT_INTERMEDIATE],ref_down[Q38FN_HIDDEN];
        if(!ctx||reference(ctx,gu_name,input,2*Q38FN_EXPERT_INTERMEDIATE,Q38FN_HIDDEN,ref_gu,0)||
           reference(ctx,down_name,down_input,Q38FN_HIDDEN,Q38FN_EXPERT_INTERMEDIATE,ref_down,0)) rc=1;
        else {
            float eg=maximum_error(gate,ref_gu,Q38FN_EXPERT_INTERMEDIATE);
            float eu=maximum_error(up,ref_gu+Q38FN_EXPERT_INTERMEDIATE,Q38FN_EXPERT_INTERMEDIATE);
            float ed=maximum_error(combined,ref_down,Q38FN_HIDDEN);
            fprintf(stderr,"Q38FN_TP_RECON gate=%.9g up=%.9g down=%.9g\n",eg,eu,ed);
            rc=(eg==0.0f&&eu==0.0f&&ed<2.0e-3f)?0:1;
        }
        glm53f_st_close(ctx);
    }
    MPI_Bcast(&rc,1,MPI_INT,0,MPI_COMM_WORLD);
close_blob:q38fn_tp_blob_close(&blob);
done:if(rc)MPI_Abort(MPI_COMM_WORLD,rc);MPI_Finalize();return rc;
}
