/* Both stored row types, all vocabulary owners, boundaries and large tails. */
#include "glm53f_embedding_12n.c"
#include <stdint.h>

int main(int argc,char **argv) {
    int rank,ranks;MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    if(ranks!=12)MPI_Abort(MPI_COMM_WORLD,2);
    glm53f_embedding_context_12n c={.rank=rank,.ranks=ranks};
    c.row0=(int)((long long)GLM53F_EMBED_VOCAB*rank/ranks);
    c.rows=(int)((long long)GLM53F_EMBED_VOCAB*(rank+1)/ranks)-c.row0;
    size_t count=(size_t)c.rows*GLM53F_EMBED_HIDDEN;
    c.q2_weight=malloc(count*sizeof(float));c.weight=malloc(count*sizeof(uint16_t));
    size_t output=(size_t)4096*GLM53F_EMBED_STREAMS*GLM53F_EMBED_HIDDEN;
    float *a=malloc((output+2)*4),*b=malloc((output+2)*4);int *ids=malloc(4096*sizeof(int));
    if(!c.q2_weight||!c.weight||!a||!b||!ids)MPI_Abort(MPI_COMM_WORLD,2);
    for(int r=0;r<c.rows;r++)for(int i=0;i<GLM53F_EMBED_HIDDEN;i++){
        float value=(float)(((c.row0+r)*17+i*31)%257-128)/128;
        uint32_t bits;memcpy(&bits,&value,4);
        c.q2_weight[(size_t)r*GLM53F_EMBED_HIDDEN+i]=value;
        c.weight[(size_t)r*GLM53F_EMBED_HIDDEN+i]=(uint16_t)(bits>>16);
    }
    const int sizes[]={1,3,4,63,64,65,128,129,512,4096};int ok=1,cases=0;
    float *f32=c.q2_weight;
    for(int format=0;format<2;format++){
        c.q2_weight=format?NULL:f32;
        for(size_t k=0;k<sizeof(sizes)/sizeof(sizes[0]);k++){
            int n=sizes[k];size_t bytes=(size_t)n*GLM53F_EMBED_STREAMS*GLM53F_EMBED_HIDDEN;
            for(int t=0;t<n;t++){
                int owner=t%ranks;
                int boundary=(int)((long long)GLM53F_EMBED_VOCAB*owner/ranks);
                ids[t]=(t%3==0)?boundary:(t%3==1?(boundary?boundary-1:154879):t*7919%154880);
            }
            a[0]=b[0]=123;a[bytes+1]=b[bytes+1]=456;
            setenv("GLM53F_EMBED_BATCH_PACKED","0",1);
            if(glm53f_embedding_streams_batch_12n(&c,ids,n,a+1)||glm53f_embedding_streams_packed_12n(&c,ids,n,b+1))MPI_Abort(MPI_COMM_WORLD,2);
            ok &= !memcmp(a,b,(bytes+2)*4);
            setenv("GLM53F_EMBED_BATCH_PACKED","1",1);
            if(glm53f_embedding_streams_batch_12n(&c,ids,n,b+1))MPI_Abort(MPI_COMM_WORLD,2);
            ok &= !memcmp(a,b,(bytes+2)*4);cases++;
        }
    }
    ids[0]=-1;ok &= glm53f_embedding_streams_packed_12n(&c,ids,1,b)==-1;
    ids[0]=154880;ok &= glm53f_embedding_streams_packed_12n(&c,ids,1,b)==-1;
    ok &= glm53f_embedding_streams_packed_12n(&c,ids,0,b)==-1;
    ok &= glm53f_embedding_streams_packed_12n(&c,ids,4097,b)==-1;
    int all;MPI_Allreduce(&ok,&all,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
    if(!rank)printf("GLM53F_EMBED_BATCH cases=%d formats=2 owners=12 guards=1 %s\n",cases,all?"BIT_EXACT PASS":"FAIL");
    free(f32);free(c.weight);free(a);free(b);free(ids);MPI_Finalize();return all?0:1;
}
