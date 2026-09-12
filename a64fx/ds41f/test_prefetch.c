#define _POSIX_C_SOURCE 200809L
#include "ds41f_prefetch.h"
#include "ds41f_kernels.h"
#include "ds41f_profile.h"
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

int main(int argc,char **argv)
{
    if(argc!=2){fprintf(stderr,"usage: %s NEW_FIXTURE_DIRECTORY\n",argv[0]);return 2;}
    if(mkdir(argv[1],0700))return 2;
    ds41f_engram *e=calloc(1,sizeof *e);if(!e)return 2;
    char files[4][4096];
    for(int slot=0;slot<2;++slot){ds41f_engram_table *t=&e->table[slot];
        t->first=7;t->owned_rows=32;
        for(int kind=0;kind<2;++kind){int index=slot*2+kind;
            snprintf(files[index],sizeof files[index],"%s/%d.bin",argv[1],index);
            int fd=open(files[index],O_RDWR|O_CREAT|O_EXCL,0600);if(fd<0)return 2;
            if(kind)t->scale_fd=fd;else t->weight_fd=fd;
            for(int row=0;row<32;++row){uint8_t bytes[256];size_t n=kind?8:256;
                for(size_t j=0;j<n;++j)bytes[j]=kind?(uint8_t)(121+(row+j+slot)%8):(uint8_t)(16+(row+j+slot)%96);
                if(write(fd,bytes,n)!=(ssize_t)n)return 2;}
        }
    }
    uint16_t golden[2][32][256];
    for(int slot=0;slot<2;++slot)for(int row=0;row<32;++row)
        if(ds41f_engram_read_local(e,slot,(uint64_t)row+7,golden[slot][row]))return 1;
    ds41f_prefetch *p=NULL;if(ds41f_prefetch_create(&p,e)||ds41f_profile_init(0,1))return 1;
    float expected[2][24*256],actual[24*256];uint64_t ids[2][24];
    const uint64_t (*request_ids)[24]=(const uint64_t (*)[24])ids;
    if(ds41f_prefetch_wait(p,0,actual,NULL)!=EINVAL)return 1;
    for(int generation=0;generation<96;++generation){
        if(generation==48){
            ds41f_prefetch_destroy(p);p=NULL;
            if(ds41f_engram_cache_scales(e,511)!=ENOMEM||e->table[0].scale_cache||e->table[1].scale_cache)return 1;
            if(ds41f_engram_cache_scales(e,512)||ds41f_engram_cache_scales(e,512)!=EINVAL)return 1;
            /* Cached reads must remain exact without either scale file descriptor. */
            for(int slot=0;slot<2;++slot){close(e->table[slot].scale_fd);e->table[slot].scale_fd=-1;}
            if(ds41f_prefetch_create(&p,e))return 1;
        }
        memset(expected,0,sizeof expected);
        for(int slot=0;slot<2;++slot)for(int i=0;i<24;++i){
            ids[slot][i]=(i%3)?7+(uint64_t)(generation+i+slot)%32:100;
            if(i%3)for(int j=0;j<256;++j)
                expected[slot][i*256+j]=ds41f_bf16_to_f32(golden[slot][ids[slot][i]-7][j]);
        }
        ds41f_profile_at(0,0);
        if(ds41f_prefetch_submit(p,request_ids))return 1;
        for(int slot=0;slot<2;++slot){double seconds=-1;
            if(ds41f_prefetch_wait(p,slot,actual,&seconds)||seconds<0||memcmp(actual,expected[slot],sizeof actual))return 1;}
        /* The worker must not record into the submitting thread's spans. */
        if(ds41f_profile_current[DS41F_P_ENGRAM_READ]||ds41f_profile_current[DS41F_P_ENGRAM_DECODE])return 1;
        ds41f_profile_at(1,0);
    }
    if(ftruncate(e->table[0].weight_fd,0)||ds41f_prefetch_submit(p,request_ids))return 1;
    if(ds41f_prefetch_wait(p,0,actual,NULL)!=EIO||ds41f_prefetch_wait(p,1,actual,NULL))return 1;
    /* Joining with work pending must not leave a thread using closed files. */
    if(ds41f_prefetch_submit(p,request_ids))return 1;
    ds41f_prefetch_destroy(p);ds41f_profile_free();ds41f_engram_close(e);free(e);
    for(int i=0;i<4;++i)unlink(files[i]);
    rmdir(argv[1]);
    puts("PREFETCH PASS bit_exact generations=96 remote_zeros short_read_error pending_close profiler_TLS scale_cache budget cache_only_reads");return 0;
}
