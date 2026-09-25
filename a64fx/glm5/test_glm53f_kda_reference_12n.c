#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "glm53f_collective_12n.h"
#include "glm53f_kda_12n.h"

enum { HIDDEN=4096 };
static int load_f32(const char*path,float*p,size_t n){FILE*f=fopen(path,"rb");int ok=f&&fread(p,sizeof(*p),n,f)==n&&fgetc(f)==EOF;if(f)fclose(f);return ok?0:-1;}
int main(int argc,char**argv){int rank,nr,ok=1,all_ok;float*x,*out;glm53f_kda_context_12n*c;MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&nr);if(argc!=4||nr!=12)MPI_Abort(MPI_COMM_WORLD,2);if(getenv("GLM53F_UTOFU")&&glm53f_collective_init_12n(getenv("TOFU_TOPO_PATH"),HIDDEN))MPI_Abort(MPI_COMM_WORLD,2);x=malloc(HIDDEN*sizeof(*x));out=malloc(HIDDEN*sizeof(*out));if(!x||!out||load_f32(argv[2],x,HIDDEN))MPI_Abort(MPI_COMM_WORLD,2);c=glm53f_kda_create_12n(argv[1],atoi(argv[3]));if(!c||glm53f_kda_sublayer_12n(c,out,x))ok=0;MPI_Allreduce(&ok,&all_ok,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);if(!rank&&all_ok){const char*path=getenv("GLM53F_KDA_OUTPUT");if(path){FILE*f=fopen(path,"wb");if(!f||fwrite(out,sizeof(*out),HIDDEN,f)!=HIDDEN||fclose(f))all_ok=0;}printf("SENTINEL glm53f_kda_reference_12n=%s layer=%d\n",all_ok?"PASS":"FAIL",atoi(argv[3]));}glm53f_kda_free_12n(c);free(out);free(x);glm53f_collective_free_12n();MPI_Finalize();return all_ok?0:1;}
