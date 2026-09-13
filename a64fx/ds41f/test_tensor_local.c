#define _POSIX_C_SOURCE 200809L
#include "ds41f_tensor.h"
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

static void require(int condition,const char *message)
{if(!condition){fprintf(stderr,"TENSOR_LOCAL FAIL %s\n",message);exit(1);}}
static uint8_t value(size_t i){return (uint8_t)(i*17+i/251);}
int main(int argc,char **argv)
{
    if(argc!=2)return 2;
    require(!mkdir(argv[1],0700)||errno==EEXIST,"fixture directory");
    char path[4096];int length=snprintf(path,sizeof path,"%s/fixture.bin",argv[1]);
    require(length>0&&(size_t)length<sizeof path,"fixture path");
    const size_t sizes[]={1,65537,2*1024*1024+257};size_t cases=0;
    for(int fresh=0;fresh<2;++fresh)for(size_t shape=0;shape<3;++shape){
        size_t bytes=sizes[shape];int fd=open(path,O_CREAT|O_TRUNC|O_RDWR,0600);
        require(fd>=0,"fixture open");
        for(size_t offset=0;offset<bytes;){uint8_t block[8192];size_t count=bytes-offset;
            if(count>sizeof block)count=sizeof block;
            for(size_t i=0;i<count;++i)block[i]=value(offset+i);
            require(write(fd,block,count)==(ssize_t)count,"fixture write");offset+=count;}
        void *data=NULL;
        require(!ds41f_tensor_load_local(argv[1],"fixture",bytes,&data,fresh),"load");
        require(!((uintptr_t)data%256),"alignment");
        for(size_t i=0;i<bytes;++i)require(((uint8_t *)data)[i]==value(i),"contents");
        uint8_t replacement=255;
        require(pwrite(fd,&replacement,1,0)==1&&!fsync(fd),"change source");
        require(((uint8_t *)data)[0]==value(0),"resident copy independent of source changes");
        ds41f_tensor_free_local(data,bytes,fresh);data=NULL;
        require(ds41f_tensor_load_local(argv[1],"fixture",bytes+1,&data,fresh)==EINVAL&&!data,"size rejection");
        require(!close(fd)&&!unlink(path),"fixture cleanup");++cases;
    }
    ds41f_tensor_free_local(NULL,0,0);ds41f_tensor_free_local(NULL,0,1);
    printf("TENSOR_LOCAL PASS cases=%zu malloc fresh_pages boundary_sizes source_independence size_rejection\n",cases);
    return 0;
}
