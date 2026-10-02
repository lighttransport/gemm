/* Stage native GGUF MLA projections into one rank-local image.
 * Q5_K/Q8_0 blocks are preserved byte-for-byte so the sparse runtime can use
 * the same quantized arithmetic as llama.cpp instead of requantizing to FP8. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define GGUF_LOADER_IMPLEMENTATION
#include "../../common/gguf_loader.h"

#include <mpi.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef GLM53F_PP_SPARSE_STAGE
#include "glm53f_parallel.h"
#include "glm53f_pp_source.h"
#include "glm53f_pp_blob.h"
static glm53f_parallel_config pp_config;
static glm53f_parallel_map pp_map;
static uint64_t pp_hash=UINT64_C(1469598103934665603),pp_entry_hash;
#define SPARSE_RANKS 4
#else
#define SPARSE_RANKS 12
#endif
enum { RANKS=SPARSE_RANKS, HIDDEN=4096, HEADS=64, HD=256, QA=1536, LAT=512, QKV=16384 };

typedef struct { int fd; uint64_t base; const gguf_tensor_info *info; } tensor_ref;

static void die(int rank, const char *what) {
    fprintf(stderr, "rank=%d glm53f_q2_sparse_stage: %s: %s\n", rank, what,
            errno ? strerror(errno) : "contract failure");
    MPI_Abort(MPI_COMM_WORLD, 2);
}

static void slice(int rank, int *h0, int *hn) {
    int begin = HEADS * rank / RANKS;
    int end = HEADS * (rank + 1) / RANKS;
    *h0 = begin;
    *hn = end - begin;
}

static tensor_ref find_tensor(const gguf_context *g, const char *name) {
    tensor_ref t={-1,0,NULL};
    for (uint64_t i=0;i<g->n_tensors;i++) if (g->tensors[i].name.str &&
            !strcmp(g->tensors[i].name.str,name)) {
        t.info=&g->tensors[i];
        t.fd=g->tensor_fds?g->tensor_fds[i]:g->fd;
        t.base=g->tensor_file_offsets?g->tensor_file_offsets[i]:
               g->data_offset+g->tensors[i].offset;
        break;
    }
    return t;
}

static size_t row_bytes(const tensor_ref *t, int cols) {
    if(!t||!t->info)return 0;
    uint32_t type=t->info->type;
    if (type>=GGML_TYPE_COUNT || ggml_type_info[type].block_size<=0 || cols%ggml_type_info[type].block_size) return 0;
    return (size_t)(cols/ggml_type_info[type].block_size)*ggml_type_info[type].type_size;
}

static int read_exact(const tensor_ref *t, uint64_t rel, void *dst, size_t n) {
    unsigned char *p=dst;
#ifdef GLM53F_PP_SPARSE_STAGE
    uint64_t begin=rel;size_t bytes=n;
#endif
    while(n){ssize_t z=pread(t->fd,p,n,(off_t)(t->base+rel));
        if(z<0){if(errno==EINTR)continue;return-1;}if(!z){errno=EIO;return-1;}
        p+=z;rel+=(uint64_t)z;n-=(size_t)z;}
#ifdef GLM53F_PP_SPARSE_STAGE
    (void)posix_fadvise(t->fd,(off_t)(t->base+begin),bytes,POSIX_FADV_DONTNEED);
#endif
    return 0;
}

static int write_all(int fd,const void *src,size_t n){const unsigned char*p=src;
#ifdef GLM53F_PP_SPARSE_STAGE
    for(size_t i=0;i<n;i++){pp_hash^=p[i];pp_hash*=UINT64_C(1099511628211);pp_entry_hash^=p[i];pp_entry_hash*=UINT64_C(1099511628211);}
#endif
    while(n){ssize_t z=write(fd,p,n);if(z<0){if(errno==EINTR)continue;return-1;}
        if(!z){errno=EIO;return-1;}p+=z;n-=(size_t)z;}return 0;}

static int stage_rows(int fd,FILE*m,uint64_t*off,const tensor_ref*t,
        int expected_type,int src_rows,int cols,int row0,int rows,
        const char*name,void*buf){
    size_t rb=row_bytes(t,cols),bytes=(size_t)rows*rb;
    uint64_t flat_rows=1;
    if(t->info)for(uint32_t d=1;d<t->info->n_dims;d++)flat_rows*=t->info->dims[d];
    if(!t->info||t->info->type!=(uint32_t)expected_type||t->info->n_dims<2||
       t->info->dims[0]!=(uint64_t)cols||flat_rows!=(uint64_t)src_rows||!rb){
        if(t->info)fprintf(stderr,"stage_rows contract name=%s type=%u expected=%d dims=%"PRIu64",%"PRIu64" expected=%d,%d rb=%zu\n",name,t->info->type,expected_type,t->info->dims[0],t->info->dims[1],cols,src_rows,rb);
        return-1;
    }
#ifdef GLM53F_PP_SPARSE_STAGE
    if(row0<0||rows<1||row0>src_rows-rows)return-1;
    pp_entry_hash=UINT64_C(1469598103934665603);
    for(int start=0;start<rows;start+=64){int count=rows-start;if(count>64)count=64;
        if(read_exact(t,(uint64_t)(row0+start)*rb,buf,(size_t)count*rb)||write_all(fd,buf,(size_t)count*rb))return-1;}
#else
    if(read_exact(t,(uint64_t)row0*rb,buf,bytes)||write_all(fd,buf,bytes))return-1;
#endif
    fprintf(m,"%"PRIu64" %u %s %d %d %s\n",*off,t->info->type,
            ggml_type_name(t->info->type),rows,cols,name);
#ifdef GLM53F_PP_SPARSE_STAGE
    fprintf(m,"# PAYLOAD offset=%" PRIu64 " bytes=%zu fnv1a=%016" PRIx64 " source=%s rows=%d:%d columns=0:%d\n",*off,bytes,pp_entry_hash,name,row0,row0+rows,cols);
#endif
    *off+=bytes;return 0;
}

static int stage_columns(int fd,FILE*m,uint64_t*off,const tensor_ref*t,
        int expected_type,int rows,int src_cols,int col0,int cols,
        const char*name,void*full,void*local){
    size_t fr=row_bytes(t,src_cols),lr=row_bytes(t,cols);
    if(!t||!t->info||t->info->type>=GGML_TYPE_COUNT||!fr||!lr)return-1;
    int bs=ggml_type_info[t->info->type].block_size;
    if(bs<=0||col0<0||cols<1||col0>src_cols-cols)return-1;
    size_t byte0=(size_t)(col0/bs)*ggml_type_info[t->info->type].type_size;
    uint64_t begin=*off;
    if(!t->info||t->info->type!=(uint32_t)expected_type||t->info->n_dims!=2||
       t->info->dims[0]!=(uint64_t)src_cols||t->info->dims[1]!=(uint64_t)rows||
       col0%bs||cols%bs||!fr||!lr)return-1;
#ifdef GLM53F_PP_SPARSE_STAGE
    pp_entry_hash=UINT64_C(1469598103934665603);
#endif
    for(int r0=0;r0<rows;r0+=64){int nr=rows-r0<64?rows-r0:64;
        if(read_exact(t,(uint64_t)r0*fr,full,(size_t)nr*fr))return-1;
        for(int r=0;r<nr;r++)memcpy((unsigned char*)local+(size_t)r*lr,
            (unsigned char*)full+(size_t)r*fr+byte0,lr);
        if(write_all(fd,local,(size_t)nr*lr))return-1;}
    fprintf(m,"%"PRIu64" %u %s %d %d %s\n",begin,t->info->type,
            ggml_type_name(t->info->type),rows,cols,name);
#ifdef GLM53F_PP_SPARSE_STAGE
    fprintf(m,"# PAYLOAD offset=%" PRIu64 " bytes=%" PRIu64 " fnv1a=%016" PRIx64 " source=%s rows=0:%d columns=%d:%d\n",begin,(uint64_t)rows*lr,pp_entry_hash,name,rows,col0,col0+cols);
#endif
    *off+=(uint64_t)rows*lr;return 0;
}

int main(int argc,char**argv){(void)gguf_type_name;int rank,nr,h0,hn,fd=-1;uint64_t off=0;
    int first=3,end=45,tp_rank;
    char blob[4096],manifest[4096],bt[4096],mt[4096],name[128];
    gguf_context*g=NULL;FILE*m=NULL;void*rows=NULL,*full=NULL,*local=NULL;
    MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&nr);
    tp_rank=rank;
#ifdef GLM53F_PP_SPARSE_STAGE
    pp_config=glm53f_parallel_default();pp_config.layout=GLM53F_PP3_TP4;
    for(int i=3;i<argc;i++)if(glm53f_parallel_option(&pp_config,argc,argv,&i)!=1)die(rank,"pipeline option");
    if(pp_config.layout!=GLM53F_PP3_TP4||glm53f_parallel_map_rank(&pp_config,rank,nr,&pp_map))die(rank,"PP layout");
    int low[2],high[2];MPI_Allreduce(pp_config.cuts,low,2,MPI_INT,MPI_MIN,MPI_COMM_WORLD);MPI_Allreduce(pp_config.cuts,high,2,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
    if(memcmp(low,high,sizeof(low)))die(rank,"inconsistent cuts");
    if(argc<3||nr!=12||strncmp(argv[2],"/local/",7))die(rank,"PP usage: GGUF /local/STAGE");
    first=pp_map.first_layer;end=pp_map.end_layer;tp_rank=pp_map.tp_rank;
#else
    if(argc!=3||nr!=RANKS)die(rank,"usage: GGUF STAGE");
#endif
    if(mkdir(argv[2],0755)&&errno!=EEXIST)die(rank,"mkdir");
    slice(tp_rank,&h0,&hn);
    snprintf(blob,sizeof blob,"%s/rank%02d.blob",argv[2],rank);
    snprintf(manifest,sizeof manifest,"%s/rank%02d.manifest",argv[2],rank);
    snprintf(bt,sizeof bt,"%s/.rank%02d.blob.%ld",argv[2],rank,(long)getpid());
    snprintf(mt,sizeof mt,"%s/.rank%02d.manifest.%ld",argv[2],rank,(long)getpid());
    if(!(g=gguf_open_multi(argv[1],3))||g->n_tensors!=1412)die(rank,"GGUF metadata");
    if((fd=open(bt,O_CREAT|O_EXCL|O_WRONLY,0644))<0||!(m=fopen(mt,"wx")))die(rank,"create");
#ifdef GLM53F_PP_SPARSE_STAGE
    uint64_t stamp=0;if(glm53f_pp_source_stamp(g,argv[1],&stamp))die(rank,"source identity");
    fprintf(m,"# GLM53F_PP_SPARSE_V1 layout=pp3-tp4 world_rank=%d stage=%d tp_rank=%d tp_size=4 cuts=%d,%d layers=%d:%d source_metadata_fnv1a=%016" PRIx64 "\n",rank,pp_map.stage,pp_map.tp_rank,pp_config.cuts[0],pp_config.cuts[1],pp_map.first_layer,pp_map.end_layer,stamp);
#else
    {struct stat ms;if(stat(argv[1],&ms))ms.st_size=0;
     fprintf(m,"# GLM53F_Q2_SPARSE_V3 rank=%d ranks=12 model_bytes=%lld model=%s\n",rank,(long long)ms.st_size,argv[1]);}
#endif
    /* Q8_0 (8.5 bits/value) is the widest supported type; q_b/v_b head
     * slices (hn*HD rows of QA/LAT columns) are smaller than QA*HIDDEN. */
#ifdef GLM53F_PP_SPARSE_STAGE
    size_t max_rows=64u*
#else
    size_t max_rows=(size_t)QA*
#endif
    row_bytes(&(tensor_ref){.info=&(gguf_tensor_info){.type=GGML_TYPE_Q8_0}},HIDDEN);
    size_t max_full=64*row_bytes(&(tensor_ref){.info=&(gguf_tensor_info){.type=GGML_TYPE_Q8_0}},QKV);
    size_t max_local=64*row_bytes(&(tensor_ref){.info=&(gguf_tensor_info){.type=GGML_TYPE_Q8_0}},hn*HD);
    rows=malloc(max_rows);full=malloc(max_full);local=malloc(max_local);
    if(!rows||!full||!local)die(rank,"scratch");
    for(int layer=first;layer<end;layer++){if(layer%4!=3)continue;tensor_ref qa,qb,kva,vb,op;
        snprintf(name,sizeof name,"blk.%d.attn_q_a.weight",layer);qa=find_tensor(g,name);
        snprintf(name,sizeof name,"blk.%d.attn_q_b.weight",layer);qb=find_tensor(g,name);
        snprintf(name,sizeof name,"blk.%d.attn_kv_a_mqa.weight",layer);kva=find_tensor(g,name);
        snprintf(name,sizeof name,"blk.%d.attn_v_b.weight",layer);vb=find_tensor(g,name);
        snprintf(name,sizeof name,"blk.%d.attn_output.weight",layer);op=find_tensor(g,name);
        char qn[128],qbn[128],kvn[128],vbn[128],on[128];
        snprintf(qn,sizeof qn,"blk.%d.attn_q_a.weight",layer);
        snprintf(qbn,sizeof qbn,"blk.%d.attn_q_b.weight",layer);
        snprintf(kvn,sizeof kvn,"blk.%d.attn_kv_a_mqa.weight",layer);
        snprintf(vbn,sizeof vbn,"blk.%d.attn_v_b.weight",layer);
        snprintf(on,sizeof on,"blk.%d.attn_output.weight",layer);
        if(!qa.info||!qb.info||!kva.info||!vb.info||!op.info) die(rank,"missing tensor");
        if((qa.info->type!=GGML_TYPE_Q5_K&&qa.info->type!=GGML_TYPE_Q6_K&&qa.info->type!=GGML_TYPE_Q8_0)||
           (op.info->type!=GGML_TYPE_Q5_K&&op.info->type!=GGML_TYPE_Q6_K&&op.info->type!=GGML_TYPE_Q8_0))
            die(rank,"unsupported q_a/output type");
        if(stage_rows(fd,m,&off,&qa,qa.info->type,QA,HIDDEN,0,QA,qn,rows))
            die(rank,"q_a stage");
        if(stage_rows(fd,m,&off,&qb,GGML_TYPE_Q8_0,QKV,QA,h0*HD,hn*HD,qbn,rows))
            die(rank,"q_b stage");
        if(stage_rows(fd,m,&off,&kva,GGML_TYPE_Q8_0,LAT,HIDDEN,0,LAT,kvn,rows))
            die(rank,"kv_a stage");
        if(stage_rows(fd,m,&off,&vb,GGML_TYPE_Q8_0,QKV,LAT,h0*HD,hn*HD,vbn,rows))
            die(rank,"v_b stage");
        if(stage_columns(fd,m,&off,&op,op.info->type,HIDDEN,QKV,h0*HD,hn*HD,on,full,local))
            die(rank,"output stage");
    }
#ifdef GLM53F_PP_SPARSE_STAGE
    fprintf(m,"# COMPLETE bytes=%" PRIu64 " fnv1a=%016" PRIx64 "\n",off,pp_hash);
#else
    fprintf(m,"# COMPLETE bytes=%"PRIu64"\n",off);
#endif
    if(fflush(m)||fsync(fileno(m))||fsync(fd)||fclose(m)||close(fd)||
       rename(bt,blob)||rename(mt,manifest))die(rank,"publish");
    printf("SENTINEL glm53f_q2_sparse_stage=OK rank=%d bytes=%"PRIu64"\n",rank,off);
    free(local);free(full);free(rows);gguf_close(g);MPI_Finalize();return 0;
}
