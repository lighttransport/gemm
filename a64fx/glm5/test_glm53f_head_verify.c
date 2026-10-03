#include "glm53f_head_verify_sve.h"
#include <mpi.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float reference_dot(const float *w,const float *x,int n) {
    svfloat32_t a=svdup_f32(0);
    for(int i=0;i<n;i+=(int)svcntw()){
        svbool_t p=svwhilelt_b32(i,n);
        a=svmla_x(p,a,svld1(p,w+i),svld1(p,x+i));
    }
    return svaddv_f32(svptrue_b32(),a);
}
static void project(float *out,const float *w,const float *x,int rows,int cols,int tokens,int shared) {
    int stride=rows+3;
    if(shared){
#pragma omp parallel for schedule(static)
        for(int r=0;r<rows/2;r++)glm53f_head_f32_pair_batch(out+2*r,stride,w+(size_t)2*r*cols,x,tokens,cols);
        if(rows%2)for(int t=0;t<tokens;t++)out[(size_t)t*stride+rows-1]=reference_dot(w+(size_t)(rows-1)*cols,x+(size_t)t*cols,cols);
    }else{
#pragma omp parallel for collapse(2) schedule(static)
        for(int t=0;t<tokens;t++)for(int r=0;r<rows;r++)out[(size_t)t*stride+r]=reference_dot(w+(size_t)r*cols,x+(size_t)t*cols,cols);
    }
}
int main(int argc,char **argv) {
    MPI_Init(&argc,&argv);int rank;MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    const int columns[]={16,31,32,127,128,4096}, row_counts[]={1,2,3,17,12907};
    int ok=1,cases=0;
    for(size_t c=0;c<sizeof(columns)/sizeof(columns[0]);c++)for(size_t r=0;r<sizeof(row_counts)/sizeof(row_counts[0]);r++){
        int cols=columns[c],rows=row_counts[r],stride=rows+3;
        size_t size=(size_t)rows*cols,output=(size_t)5*stride+2;
        float *w=malloc(size*4),*x=malloc((size_t)5*cols*4),*a=malloc(output*4),*b=malloc(output*4);
        if(!w||!x||!a||!b)MPI_Abort(MPI_COMM_WORLD,2);
        for(size_t i=0;i<size;i++)w[i]=(float)((int)(i*31%257)-128)/128;
        for(int i=0;i<5*cols;i++)x[i]=(float)((i*17+13)%251-125)/128;
        for(int tokens=1;tokens<=5;tokens++){
            for(size_t i=0;i<output;i++)a[i]=b[i]=12345;
            project(a+1,w,x,rows,cols,tokens,0);project(b+1,w,x,rows,cols,tokens,1);
            ok &= !memcmp(a,b,output*4);
            for(int t=0;t<tokens;t++)for(int j=0;j<rows;j++){
                uint32_t bits;memcpy(&bits,b+1+(size_t)t*stride+j,4);
                ok &= (bits&0x7f800000u)!=0x7f800000u;
            }
            cases++;
        }
        if(rows==12907&&cols==4096)for(int tokens=2;tokens<=5;tokens++)for(int pair=0;pair<5;pair++){
            double elapsed[2]={0,0},maximum[2];
            for(int turn=0;turn<2;turn++){
                int mode=(pair+turn)%2;
                MPI_Barrier(MPI_COMM_WORLD);double start=MPI_Wtime();
                for(int repeat=0;repeat<3;repeat++)project(b+1,w,x,rows,cols,tokens,mode);
                elapsed[mode]=(MPI_Wtime()-start)/3;
            }
            MPI_Reduce(elapsed,maximum,2,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
            if(!rank)printf("GLM53F_HEAD_VERIFY_TIMING tokens=%d pair=%d legacy_s=%.9f shared_s=%.9f speedup=%.9f\n",tokens,pair,maximum[0],maximum[1],maximum[0]/maximum[1]);
        }
        free(w);free(x);free(a);free(b);
    }
    int all;MPI_Allreduce(&ok,&all,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
    if(!rank)printf("GLM53F_HEAD_VERIFY cases=%d threads=%d bit_exact=%d finite_and_guards=%d %s\n",cases,omp_get_max_threads(),all,all,all?"PASS":"FAIL");
    MPI_Finalize();return all?0:1;
}
