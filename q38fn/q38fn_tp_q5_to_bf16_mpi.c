#define _GNU_SOURCE
#include <mpi.h>
#define Q38FN_TP_BLOB_IMPLEMENTATION
#include "../common/q38fn_tp_blob.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

static uint64_t hash_bytes(const void*p,size_t n){const unsigned char*s=p;uint64_t h=UINT64_C(1469598103934665603);while(n--){h^=*s++;h*=UINT64_C(1099511628211);}return h;}
static int write_all(int fd,const void*p,size_t n){const unsigned char*s=p;while(n){ssize_t z=write(fd,s,n);if(z<=0)return-1;s+=z;n-=(size_t)z;}return 0;}
static int selected(const char*n){const char*mode=getenv("Q38FN_TP_BF16_SELECT");int delta=strstr(n,".linear_attn.")!=NULL,delta_out=q38fn_tp_ends_with(n,".linear_attn.out_proj.weight"),gate=q38fn_tp_ends_with(n,".mlp.gate.weight"),hc_down=q38fn_tp_ends_with(n,".input_mix_weight_down.weight"),hc_inject=q38fn_tp_ends_with(n,".block_inject_weight.weight");if(!mode||!*mode||!strcmp(mode,"all"))return delta||gate;if(!strcmp(mode,"gate"))return gate;if(!strcmp(mode,"delta"))return delta;if(!strcmp(mode,"delta_out"))return delta_out;if(!strcmp(mode,"hc_down"))return hc_down;if(!strcmp(mode,"hc_inject"))return hc_inject;if(!strcmp(mode,"hc"))return hc_down||hc_inject;if(!strcmp(mode,"hc_down_delta_out"))return hc_down||delta_out;return 0;}
static uint16_t to_bf16(float x){uint32_t u;memcpy(&u,&x,4);u+=0x7fffu+((u>>16)&1u);return(uint16_t)(u>>16);}
static void write_entry(FILE*f,const q38fn_tp_blob_entry*e,uint64_t off,uint64_t bytes,uint64_t hash){fprintf(f,"%llu %llu %016llx %d %d %d",(unsigned long long)off,(unsigned long long)bytes,(unsigned long long)hash,(int)e->kind,e->axis,e->ndims);for(int d=0;d<e->ndims;d++)fprintf(f," %llu",(unsigned long long)e->shape[d]);fprintf(f," %d",e->n_ranges);for(int r=0;r<e->n_ranges;r++)fprintf(f," %llu %llu",(unsigned long long)e->range[r].start,(unsigned long long)e->range[r].count);fprintf(f," %s\n",e->name);}

int main(int argc,char**argv){int rank,ranks,rc=1,fd=-1,count=0;char srcdir[4096],dstdir[4096],bp[8192],mp[8192],partial[8192];FILE*mf=NULL;q38fn_tp_blob b={0};uint64_t off=0;MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&ranks);if(argc!=3||ranks!=Q38FN_TP_RANKS)goto done;setenv("Q38FN_TP_FILE_BACKED","1",1);snprintf(srcdir,sizeof(srcdir),"%s/rank-%02d",argv[1],rank);snprintf(dstdir,sizeof(dstdir),"%s/rank-%02d",argv[2],rank);mkdir(argv[2],0700);mkdir(dstdir,0700);snprintf(bp,sizeof(bp),"%s/tp12-v%d.blob",dstdir,Q38FN_TP_LAYOUT_VERSION);snprintf(mp,sizeof(mp),"%s/tp12-v%d.manifest",dstdir,Q38FN_TP_LAYOUT_VERSION);snprintf(partial,sizeof(partial),"%s.partial",mp);if(q38fn_tp_blob_open(&b,srcdir,0)||(fd=open(bp,O_CREAT|O_TRUNC|O_WRONLY,0600))<0||!(mf=fopen(partial,"w")))goto done;fprintf(mf,"# Q38FNTP layout=%d rank=%d ranks=%d layers=0 bf16_overlay=1\n",Q38FN_TP_LAYOUT_VERSION,rank,ranks);
 for(int ei=0;ei<b.n_entries;ei++){q38fn_tp_blob_entry*e=&b.entries[ei];if(!e->q5_data||!selected(e->name))continue;size_t cols=e->kind==Q38FN_TP_AXIS1?(size_t)e->range[0].count:(size_t)e->shape[e->ndims-1];if(!cols||cols%32)goto done;size_t rows=e->q5_bytes/(sizeof(q38fn_q5_block)*(cols/32)),bytes=rows*cols*sizeof(uint16_t);uint16_t*w=malloc(bytes);if(!w)goto done;int failed=0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(|:failed)
#endif
  for(size_t r=0;r<rows;r++){float*row=malloc(cols*sizeof(float));if(!row){failed=1;continue;}if(q38fn_q5_dequantize_row(row,e->q5_data+r*(cols/32),cols)){failed=1;free(row);continue;}for(size_t c=0;c<cols;c++)w[r*cols+c]=to_bf16(row[c]);free(row);}if(failed){free(w);goto done;}uint64_t aligned=(off+255u)&~UINT64_C(255);if(aligned>off){unsigned char pad[256]={0};if(write_all(fd,pad,(size_t)(aligned-off))){free(w);goto done;}off=aligned;}if(write_all(fd,w,bytes)){free(w);goto done;}write_entry(mf,e,off,bytes,hash_bytes(w,bytes));off+=bytes;count++;free(w);}
 fprintf(mf,"# COMPLETE blob_bytes=%llu tensors=%d bf16_overlay=1\n",(unsigned long long)off,count);if(!count||fflush(mf)||fsync(fileno(mf))||fsync(fd))goto done;fclose(mf);mf=NULL;close(fd);fd=-1;q38fn_tp_blob_close(&b);MPI_Barrier(MPI_COMM_WORLD);if(rename(partial,mp))goto done;fprintf(stderr,"Q38FN_TP_BF16_OVERLAY rank=%d bytes=%llu tensors=%d\n",rank,(unsigned long long)off,count);rc=0;
done:if(rc){fprintf(stderr,"Q38FN_TP_BF16_OVERLAY_ERROR rank=%d count=%d offset=%llu errno=%d\n",rank,count,(unsigned long long)off,errno);fflush(stderr);}if(mf)fclose(mf);if(fd>=0)close(fd);q38fn_tp_blob_close(&b);if(rc)MPI_Abort(MPI_COMM_WORLD,rc);MPI_Finalize();return rc;}
