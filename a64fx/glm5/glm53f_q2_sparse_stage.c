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

enum { RANKS=12, HIDDEN=4096, HEADS=64, HD=256, QA=1536, LAT=512, QKV=16384 };

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
    uint32_t type=t->info->type;
    if (type>=GGML_TYPE_COUNT || cols%ggml_type_info[type].block_size) return 0;
    return (size_t)(cols/ggml_type_info[type].block_size)*ggml_type_info[type].type_size;
}

static int read_exact(const tensor_ref *t, uint64_t rel, void *dst, size_t n) {
    unsigned char *p=dst;
    while(n){ssize_t z=pread(t->fd,p,n,(off_t)(t->base+rel));
        if(z<0){if(errno==EINTR)continue;return-1;}if(!z){errno=EIO;return-1;}
        p+=z;rel+=(uint64_t)z;n-=(size_t)z;}
    return 0;
}

static int write_all(int fd,const void *src,size_t n){const unsigned char*p=src;
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
    if(read_exact(t,(uint64_t)row0*rb,buf,bytes)||write_all(fd,buf,bytes))return-1;
    fprintf(m,"%"PRIu64" %u %s %d %d %s\n",*off,t->info->type,
            ggml_type_name(t->info->type),rows,cols,name);*off+=bytes;return 0;
}

static int stage_columns(int fd,FILE*m,uint64_t*off,const tensor_ref*t,
        int expected_type,int rows,int src_cols,int col0,int cols,
        const char*name,void*full,void*local){
    size_t fr=row_bytes(t,src_cols),lr=row_bytes(t,cols);
    int bs=ggml_type_info[t->info->type].block_size;
    size_t byte0=(size_t)(col0/bs)*ggml_type_info[t->info->type].type_size;
    uint64_t begin=*off;
    if(!t->info||t->info->type!=(uint32_t)expected_type||t->info->n_dims!=2||
       t->info->dims[0]!=(uint64_t)src_cols||t->info->dims[1]!=(uint64_t)rows||
       col0%bs||cols%bs||!fr||!lr)return-1;
    for(int r0=0;r0<rows;r0+=64){int nr=rows-r0<64?rows-r0:64;
        if(read_exact(t,(uint64_t)r0*fr,full,(size_t)nr*fr))return-1;
        for(int r=0;r<nr;r++)memcpy((unsigned char*)local+(size_t)r*lr,
            (unsigned char*)full+(size_t)r*fr+byte0,lr);
        if(write_all(fd,local,(size_t)nr*lr))return-1;}
    fprintf(m,"%"PRIu64" %u %s %d %d %s\n",begin,t->info->type,
            ggml_type_name(t->info->type),rows,cols,name);*off+=(uint64_t)rows*lr;return 0;
}

int main(int argc,char**argv){int rank,nr,h0,hn,fd=-1;uint64_t off=0;
    char blob[4096],manifest[4096],bt[4096],mt[4096],name[128];
    gguf_context*g=NULL;FILE*m=NULL;void*rows=NULL,*full=NULL,*local=NULL;
    MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&nr);
    if(argc!=3||nr!=RANKS)die(rank,"usage: GGUF STAGE");
    if(mkdir(argv[2],0755)&&errno!=EEXIST)die(rank,"mkdir");
    slice(rank,&h0,&hn);
    snprintf(blob,sizeof blob,"%s/rank%02d.blob",argv[2],rank);
    snprintf(manifest,sizeof manifest,"%s/rank%02d.manifest",argv[2],rank);
    snprintf(bt,sizeof bt,"%s/.rank%02d.blob.%ld",argv[2],rank,(long)getpid());
    snprintf(mt,sizeof mt,"%s/.rank%02d.manifest.%ld",argv[2],rank,(long)getpid());
    if(!(g=gguf_open_multi(argv[1],3))||g->n_tensors!=1412)die(rank,"GGUF metadata");
    if((fd=open(bt,O_CREAT|O_EXCL|O_WRONLY,0644))<0||!(m=fopen(mt,"wx")))die(rank,"create");
    {struct stat ms;if(stat(argv[1],&ms))ms.st_size=0;
     fprintf(m,"# GLM53F_Q2_SPARSE_V3 rank=%d ranks=12 model_bytes=%lld model=%s\n",rank,(long long)ms.st_size,argv[1]);}
    /* Q8_0 (8.5 bits/value) is the widest supported type; q_b/v_b head
     * slices (hn*HD rows of QA/LAT columns) are smaller than QA*HIDDEN. */
    size_t max_rows=(size_t)QA*row_bytes(&(tensor_ref){.info=&(gguf_tensor_info){.type=GGML_TYPE_Q8_0}},HIDDEN);
    size_t max_full=64*row_bytes(&(tensor_ref){.info=&(gguf_tensor_info){.type=GGML_TYPE_Q8_0}},QKV);
    size_t max_local=64*row_bytes(&(tensor_ref){.info=&(gguf_tensor_info){.type=GGML_TYPE_Q8_0}},hn*HD);
    rows=malloc(max_rows);full=malloc(max_full);local=malloc(max_local);
    if(!rows||!full||!local)die(rank,"scratch");
    for(int layer=3;layer<45;layer+=4){tensor_ref qa,qb,kva,vb,op;
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
    fprintf(m,"# COMPLETE bytes=%"PRIu64"\n",off);
    if(fflush(m)||fsync(fileno(m))||fsync(fd)||fclose(m)||close(fd)||
       rename(bt,blob)||rename(mt,manifest))die(rank,"publish");
    printf("SENTINEL glm53f_q2_sparse_stage=OK rank=%d bytes=%"PRIu64"\n",rank,off);
    free(local);free(full);free(rows);gguf_close(g);MPI_Finalize();return 0;
}
