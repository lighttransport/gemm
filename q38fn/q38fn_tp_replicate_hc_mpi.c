#define _GNU_SOURCE
#include <mpi.h>
#define Q38FN_TP_BLOB_IMPLEMENTATION
#include "../common/q38fn_tp_blob.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>

static int read_at(int fd,uint64_t off,void*ptr,size_t n){unsigned char*p=ptr;while(n){ssize_t z=pread(fd,p,n,(off_t)off);if(z<0&&errno==EINTR)continue;if(z<=0)return-1;p+=z;off+=(uint64_t)z;n-=(size_t)z;}return 0;}
static int write_at(int fd,uint64_t off,const void*ptr,size_t n){const unsigned char*p=ptr;while(n){ssize_t z=pwrite(fd,p,n,(off_t)off);if(z<0&&errno==EINTR)continue;if(z<=0)return-1;p+=z;off+=(uint64_t)z;n-=(size_t)z;}return 0;}
static uint64_t hash_bytes(const void*ptr,size_t n){const unsigned char*p=ptr;uint64_t h=UINT64_C(1469598103934665603);while(n--){h^=*p++;h*=UINT64_C(1099511628211);}return h;}
static int is_hc(const char*n){return strstr(n,"_hyper_connection.input_mix_weight_")!=NULL;}
static void write_full(FILE*f,const q38fn_tp_blob_entry*e,uint64_t off,uint64_t bytes,uint64_t hash){fprintf(f,"%llu %llu %016llx %d -1 %d",(unsigned long long)off,(unsigned long long)bytes,(unsigned long long)hash,(int)Q38FN_TP_FULL,e->ndims);for(int d=0;d<e->ndims;d++)fprintf(f," %llu",(unsigned long long)e->shape[d]);fprintf(f," 0 %s\n",e->name);}

int main(int argc,char**argv){int rank,ranks,rc=1,in_fd=-1,out_fd=-1,tensors=0;char in_dir[4096],out_dir[4096],in_blob[8192],in_manifest[8192],out_blob[8192],out_manifest[8192],partial[8192],line[8192];FILE*src=NULL,*dst=NULL;uint64_t append=0;
 MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&ranks);if(argc!=3||ranks!=Q38FN_TP_RANKS)goto done;
 snprintf(in_dir,sizeof(in_dir),"%s/rank-%02d",argv[1],rank);snprintf(out_dir,sizeof(out_dir),"%s/rank-%02d",argv[2],rank);mkdir(argv[2],0700);mkdir(out_dir,0700);
 snprintf(in_blob,sizeof(in_blob),"%s/tp12-v%d.blob",in_dir,Q38FN_TP_LAYOUT_VERSION);snprintf(in_manifest,sizeof(in_manifest),"%s/tp12-v%d.manifest",in_dir,Q38FN_TP_LAYOUT_VERSION);snprintf(out_blob,sizeof(out_blob),"%s/tp12-v%d.blob",out_dir,Q38FN_TP_LAYOUT_VERSION);snprintf(out_manifest,sizeof(out_manifest),"%s/tp12-v%d.manifest",out_dir,Q38FN_TP_LAYOUT_VERSION);snprintf(partial,sizeof(partial),"%s.partial",out_manifest);
 src=fopen(in_manifest,"r");dst=fopen(partial,"w");in_fd=open(in_blob,O_RDONLY);out_fd=open(out_blob,O_CREAT|O_TRUNC|O_RDWR,0600);if(!src||!dst||in_fd<0||out_fd<0||!fgets(line,sizeof(line),src))goto done;
 fprintf(dst,"# Q38FNTP layout=%d rank=%d ranks=%d layers=0 replicated_hc=1\n",Q38FN_TP_LAYOUT_VERSION,rank,ranks);
 while(fgets(line,sizeof(line),src)){if(!strncmp(line,"# COMPLETE",10))break;char parsed[8192];memcpy(parsed,line,sizeof(parsed));q38fn_tp_blob_entry e={0};if(q38fn_tp_blob_parse_entry(parsed,&e))goto done;if(!is_hc(e.name)){free(e.name);continue;}if(e.ndims!=2||e.n_ranges!=1||e.bytes>INT_MAX){free(e.name);goto done;}
  int lb=(int)e.bytes,bc[Q38FN_TP_RANKS],bd[Q38FN_TP_RANKS],gb=0;uint64_t starts[Q38FN_TP_RANKS],counts[Q38FN_TP_RANKS];MPI_Allgather(&lb,1,MPI_INT,bc,1,MPI_INT,MPI_COMM_WORLD);MPI_Allgather(&e.range[0].start,1,MPI_UINT64_T,starts,1,MPI_UINT64_T,MPI_COMM_WORLD);MPI_Allgather(&e.range[0].count,1,MPI_UINT64_T,counts,1,MPI_UINT64_T,MPI_COMM_WORLD);for(int r=0;r<ranks;r++){bd[r]=gb;gb+=bc[r];}
  unsigned char*local=malloc((size_t)lb),*gathered=malloc((size_t)gb),*full=NULL;if(!local||!gathered||read_at(in_fd,e.offset,local,(size_t)lb)){free(local);free(gathered);free(e.name);goto done;}MPI_Allgatherv(local,lb,MPI_BYTE,gathered,bc,bd,MPI_BYTE,MPI_COMM_WORLD);size_t fb;
  if(e.kind==Q38FN_TP_AXIS0){fb=(size_t)gb;full=gathered;gathered=NULL;}else if(e.kind==Q38FN_TP_AXIS1){uint64_t rows=e.shape[0],cols=e.shape[1];fb=(size_t)rows*(size_t)cols*2;if((uint64_t)gb!=rows*cols*2||(full=malloc(fb))==NULL){free(local);free(gathered);free(e.name);goto done;}for(int r=0;r<ranks;r++)for(uint64_t row=0;row<rows;row++)memcpy(full+(row*cols+starts[r])*2,gathered+bd[r]+row*counts[r]*2,(size_t)counts[r]*2);}else{free(local);free(gathered);free(e.name);goto done;}
  /* Preserve the source encoding while reconstructing the full matrix.
   * input_mix_weight_down is row-sharded Q5 and input_mix_weight_up is
   * column-sharded BF16.  The fused HC runtime relies on exactly that mix:
   * a compact Q5 down projection and a bandwidth-friendly BF16 up projection.
   * Converting this sidecar to Q8 disables that validated fast path. */
  append=(append+255u)&~UINT64_C(255);if(write_at(out_fd,append,full,fb)){free(local);free(gathered);free(full);free(e.name);goto done;}write_full(dst,&e,append,fb,hash_bytes(full,fb));append+=fb;tensors++;free(local);free(gathered);free(full);free(e.name);
 }
 if(tensors!=192||ftruncate(out_fd,(off_t)append))goto done;fprintf(dst,"# COMPLETE blob_bytes=%llu tensors=%d replicated_hc=1\n",(unsigned long long)append,tensors);if(fflush(dst)||fsync(fileno(dst))||fdatasync(out_fd))goto done;fclose(src);src=NULL;fclose(dst);dst=NULL;close(in_fd);in_fd=-1;close(out_fd);out_fd=-1;MPI_Barrier(MPI_COMM_WORLD);if(rename(partial,out_manifest))goto done;fprintf(stderr,"Q38FN_TP_REPLICATE_HC rank=%d bytes=%llu tensors=%d\n",rank,(unsigned long long)append,tensors);rc=0;
done:if(src)fclose(src);if(dst)fclose(dst);if(in_fd>=0)close(in_fd);if(out_fd>=0)close(out_fd);if(rc)MPI_Abort(MPI_COMM_WORLD,rc);MPI_Finalize();return rc;}
