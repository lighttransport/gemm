#define _GNU_SOURCE
#include <mpi.h>
#define SAFETENSORS_IMPLEMENTATION
#define GLM53F_SAFETENSORS_IMPLEMENTATION
#include "../common/glm53f_safetensors.h"
#include "../common/q38fn_arch.h"
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int copy_file(const char *source, const char *destination)
{
    int in=open(source,O_RDONLY),out=-1,rc=-1;char*buffer=NULL;struct stat st,dst;
    if(in<0||fstat(in,&st)!=0)goto done;
    if(stat(destination,&dst)==0&&dst.st_size==st.st_size){
        out=open(destination,O_RDONLY);
        if(out>=0)(void)posix_fadvise(out,0,dst.st_size,POSIX_FADV_DONTNEED);
        rc=0;goto done;
    }
    out=open(destination,O_WRONLY|O_CREAT,0600);if(out<0||ftruncate(out,st.st_size)!=0)goto done;
    buffer=(char*)malloc(16u<<20);if(!buffer)goto done;
    off_t offset=0;while(offset<st.st_size){size_t want=(size_t)(st.st_size-offset);if(want>(16u<<20))want=16u<<20;ssize_t got=pread(in,buffer,want,offset);if(got<=0)goto done;ssize_t put=pwrite(out,buffer,(size_t)got,offset);if(put!=got)goto done;offset+=got;posix_fadvise(in,offset-got,(off_t)got,POSIX_FADV_DONTNEED);}if(fsync(out)!=0)goto done;(void)posix_fadvise(out,0,st.st_size,POSIX_FADV_DONTNEED);rc=0;
done:free(buffer);if(in>=0)close(in);if(out>=0)close(out);return rc;
}
static int backbone_layer(const char *name)
{
    int layer=-1;if(sscanf(name,"model.language_model.layers.%d.",&layer)==1)return layer;return -1;
}
static int mkdir_p(const char *path)
{
    char *copy=strdup(path);if(!copy)return -1;
    for(char*p=copy+1;*p;++p)if(*p=='/'){*p='\0';if(mkdir(copy,0700)!=0&&errno!=EEXIST){free(copy);return -1;}*p='/';}
    int rc=(mkdir(copy,0700)==0||errno==EEXIST)?0:-1;free(copy);return rc;
}
int main(int argc,char**argv)
{
    int rank,ranks;MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    if(argc!=3||ranks!=12){if(!rank)fprintf(stderr,"usage: %s MODEL_DIR LOCAL_BASE (12 ranks)\n",argv[0]);MPI_Abort(MPI_COMM_WORLD,2);}
    glm53f_st_context*ctx=glm53f_st_open(argv[1]);if(!ctx)MPI_Abort(MPI_COMM_WORLD,1);
    unsigned char*needed=(unsigned char*)calloc((size_t)ctx->n_shards,1);if(!needed)MPI_Abort(MPI_COMM_WORLD,1);
    int l0=rank*4,l1=l0+4;
    for(int i=0;i<ctx->n_entries;++i){const char*n=ctx->entries[i].name;int layer=backbone_layer(n),take=layer>=l0&&layer<l1;
        const char*p=strstr(n,"ngram_embedding.shard_");if(p){int shard=-1;take=sscanf(p,"ngram_embedding.shard_%d.weight",&shard)==1&&q38fn_ngram_owner(shard)==rank;}
        if(rank==0&&!strcmp(n,"model.language_model.embed_tokens.weight"))take=1;
        if(rank==ranks-1&&(!strcmp(n,"lm_head.weight")||!strncmp(n,"model.language_model.hyper_connection_mixer.",strlen("model.language_model.hyper_connection_mixer."))))take=1;
        if(take)needed[ctx->entries[i].shard]=1;
    }
    char directory[4096];snprintf(directory,sizeof(directory),"%s/rank-%02d",argv[2],rank);
    if(mkdir_p(directory)!=0){fprintf(stderr,"rank %d mkdir failed %s: %s\n",rank,directory,strerror(errno));fflush(stderr);MPI_Abort(MPI_COMM_WORLD,1);}
    unsigned long long bytes=0;int files=0;
    for(int i=0;i<ctx->n_shards;++i)if(needed[i]){
        char *source=NULL,*destination=NULL;struct stat st;
        if(asprintf(&source,"%s/%s",argv[1],ctx->shards[i].name)<0||
           asprintf(&destination,"%s/%s",directory,ctx->shards[i].name)<0||
           stat(source,&st)!=0||copy_file(source,destination)!=0){
            fprintf(stderr,"rank %d stage failed %s\n",rank,source?source:"(path)");
            free(source);free(destination);MPI_Abort(MPI_COMM_WORLD,1);
        }
        bytes+=(unsigned long long)st.st_size;files++;free(source);free(destination);
    }
    fprintf(stderr,"Q38FN_STAGE rank=%d files=%d bytes=%llu dir=%s\n",rank,files,bytes,directory);free(needed);glm53f_st_close(ctx);MPI_Finalize();return 0;
}
