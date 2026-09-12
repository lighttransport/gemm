#define _POSIX_C_SOURCE 200809L
#include "ds41f_weights.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#define REQUIRE(x) do{if(!(x)){fprintf(stderr,"TP_WEIGHTS FAIL line=%d\n",__LINE__);return 1;}}while(0)
int main(int argc,char **argv)
{
    if(argc!=2)return 2;
    REQUIRE(!mkdir(argv[1],0700));char path[4096];snprintf(path,sizeof path,"%s/weights.tp",argv[1]);
    const char *names[]={"head.weight","mtp.0.main_proj.scale","mtp.0.main_proj.weight","mtp.2.markov_head.embed.weight","mtp.2.markov_head.head.weight"};
    const size_t global[]={129280,160,5120,129280,129280},cols[]={5120,480,15360,256,256};
    ds41f_weight items[5]={{.rows=0}};ds41f_weights s={.items=items,.count=5};size_t cases=0;
    for(int tp=2;tp<=4;tp*=2)for(int rank=0;rank<12;++rank)for(int fault=0;fault<4;++fault){
        FILE *f=fopen(path,"w");REQUIRE(f);fprintf(f,"DS41FTP 1 %d %d 12\n",tp,rank);
        for(int i=0;i<5;++i){int vocab=i==0||i>=3;size_t first=vocab?(global[i]/32*(size_t)rank/12)*32:global[i]/tp*(rank%tp);
            size_t local=vocab?(global[i]/32*(size_t)(rank+1)/12)*32-first:global[i]/tp;
            memset(items+i,0,sizeof items[i]);strcpy(items[i].name,names[i]);items[i].rows=local;items[i].cols=cols[i];
            if(fault==2&&i==4)continue;
            fprintf(f,"%s %zu %zu %zu %zu\n",names[i],global[i],cols[i],first+(fault==1&&i==0?1:0),local);
            if(fault==3&&i==4)fprintf(f,"%s %zu %zu %zu %zu\n",names[i],global[i],cols[i],first,local);
        }
        REQUIRE(!fclose(f));int rc=ds41f_weights_check_tp(&s,argv[1],tp,rank);REQUIRE(fault?rc==EINVAL:rc==0);
        if(!fault)for(int i=0;i<5;++i)REQUIRE(items[i].global_rows==global[i]);++cases;
    }
    REQUIRE(!unlink(path)&&!rmdir(argv[1]));printf("TP_WEIGHTS PASS cases=%zu backbone_head Markov_vocab main_projection ranges missing duplicate\n",cases);return 0;
}
