#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "glm53f_moe_stage_12n.h"

enum { HIDDEN = 4096, TOKENS = 4 };

int main(int argc, char **argv) {
    int rank,ranks,ok,local_ok=1,layer=argc>4?atoi(argv[4]):3;
    float*x=malloc((size_t)TOKENS*HIDDEN*4),*a=malloc((size_t)TOKENS*HIDDEN*4),*b=malloc((size_t)TOKENS*HIDDEN*4);
    MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    if(argc<4||ranks!=12||!x||!a||!b)MPI_Abort(MPI_COMM_WORLD,2);
    glm53f_moe_stage_context_12n*ca=glm53f_moe_stage_create_12n(argv[2],argv[3],argv[1],layer,1);
    glm53f_moe_stage_context_12n*cb=glm53f_moe_stage_create_12n(argv[2],argv[3],argv[1],layer,1);
    if(!ca||!cb)MPI_Abort(MPI_COMM_WORLD,2);
    for(int t=0;t<TOKENS;t++)for(int i=0;i<HIDDEN;i++)x[(size_t)t*HIDDEN+i]=(float)(((i*17+t*31+5)%251)-125)/125.0f;
    glm53f_moe_stage_sublayer_12n(ca,a,x);glm53f_moe_stage_sublayer_batch_12n(cb,b,x,1);
    MPI_Barrier(MPI_COMM_WORLD);double t0=MPI_Wtime();for(int t=0;t<TOKENS;t++)local_ok&=!glm53f_moe_stage_sublayer_12n(ca,a+(size_t)t*HIDDEN,x+(size_t)t*HIDDEN);double seq=MPI_Wtime()-t0;
    MPI_Barrier(MPI_COMM_WORLD);t0=MPI_Wtime();local_ok&=!glm53f_moe_stage_sublayer_batch_12n(cb,b,x,TOKENS);double bat=MPI_Wtime()-t0;
    double d2=0,r2=0;for(int i=0;i<TOKENS*HIDDEN;i++){double d=(double)a[i]-b[i];d2+=d*d;r2+=(double)a[i]*a[i];}double rel=sqrt(d2/(r2+1e-30));local_ok&=rel<3e-6;
    MPI_Allreduce(&local_ok,&ok,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);double sm,bm;MPI_Reduce(&seq,&sm,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);MPI_Reduce(&bat,&bm,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if(!rank)printf("GLM53F_MOE_BATCH layer=%d tokens=%d rel_l2=%.9g seq_ms=%.3f batch_ms=%.3f speedup=%.3f %s\n",layer,TOKENS,rel,sm*1e3,bm*1e3,sm/bm,ok?"PASS":"FAIL");
    glm53f_moe_stage_free_12n(cb);glm53f_moe_stage_free_12n(ca);free(b);free(a);free(x);MPI_Finalize();return ok?0:1;
}
