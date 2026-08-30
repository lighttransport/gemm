#define _GNU_SOURCE
#include <mpi.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int trim_rank(const char *base,int rank)
{
    char manifest[4096],temporary[4096],blob[4096],line[8192];
    if(snprintf(manifest,sizeof(manifest),"%s/rank-%02d/tp12-v1.manifest",base,rank)>=(int)sizeof(manifest)||
       snprintf(temporary,sizeof(temporary),"%s.trim",manifest)>=(int)sizeof(temporary)||
       snprintf(blob,sizeof(blob),"%s/rank-%02d/tp12-v1.blob",base,rank)>=(int)sizeof(blob))return-1;
    FILE*in=fopen(manifest,"r"),*out=NULL;if(!in)return-1;
    out=fopen(temporary,"w");if(!out){fclose(in);return-1;}
    uint64_t cutoff=0;int seen=0,entries=0,complete=0;
    while(fgets(line,sizeof(line),in)){
        if(line[0]=='#'){
            if(!strncmp(line,"# COMPLETE",10)){complete=1;break;}
            if(!seen&&fputs(line,out)==EOF)goto fail;
            continue;
        }
        char*last=strrchr(line,' ');if(!last)goto fail;last++;
        if(!strncmp(last,"model.visual.",13)||!strncmp(last,"mtp.",4)){
            if(!seen){char*end=NULL;errno=0;cutoff=strtoull(line,&end,10);if(errno||end==line)goto fail;seen=1;}
        }else{
            if(seen)goto fail;
            if(fputs(line,out)==EOF)goto fail;
            entries++;
        }
    }
    if(!complete){goto fail;}
    if(!seen){fclose(out);fclose(in);unlink(temporary);return 0;}
    if(fprintf(out,"# COMPLETE blob_bytes=%llu tensors=%d\n",(unsigned long long)cutoff,entries)<0||
       fflush(out)||fsync(fileno(out)))goto fail;
    fclose(out);out=NULL;fclose(in);in=NULL;
    int fd=open(blob,O_WRONLY);if(fd<0||ftruncate(fd,(off_t)cutoff)||fsync(fd)){if(fd>=0)close(fd);unlink(temporary);return-1;}close(fd);
    if(rename(temporary,manifest))return-1;
    fprintf(stderr,"Q38FN_TP_TRIM rank=%d bytes=%llu entries=%d\n",rank,(unsigned long long)cutoff,entries);return 0;
fail:
    if(out)fclose(out);
    if(in)fclose(in);
    unlink(temporary);
    return-1;
}

int main(int argc,char**argv)
{
    int rank,ranks,local,global;MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    local=argc==2&&ranks==12?trim_rank(argv[1],rank):-1;
    MPI_Allreduce(&local,&global,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
    if(!rank&&global)fprintf(stderr,"Q38FN_TP_TRIM failed\n");
    MPI_Finalize();
    return global?1:0;
}
