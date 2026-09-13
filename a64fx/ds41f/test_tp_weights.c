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
    REQUIRE(!mkdir(argv[1],0700));char path[4096],shared_path[4096];
    snprintf(path,sizeof path,"%s/weights.tp",argv[1]);
    snprintf(shared_path,sizeof shared_path,"%s/weights.shared.tp",argv[1]);
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
        if(!fault)for(int i=0;i<5;++i)REQUIRE(items[i].global_rows==global[i]);
        ++cases;
    }
    REQUIRE(!unlink(path)&&!rmdir(argv[1]));
    REQUIRE(!mkdir(argv[1],0700));
    /* weights.index is sorted lexicographically, so the scale precedes the
     * weight name. */
    const char *shared_names[]={"layers.0.ffn.shared_experts.w2.scale",
                                "layers.0.ffn.shared_experts.w2.weight"};
    const char *shared_dtype[]={"F8_E8M0","F8_E4M3"};
    const size_t shared_global[]={160,5120},shared_cols[]={72,2304};
    ds41f_weight shared_items[2]={0};ds41f_weights shared={.items=shared_items,.count=2};
    for(int rank=0;rank<12;++rank)for(int fault=0;fault<3;++fault){
        FILE *f=fopen(shared_path,"w");REQUIRE(f);fprintf(f,"DS41FSH 1 12 %d 12\n",rank);
        size_t first=(size_t)(160*rank/12)*32,end=(size_t)(160*(rank+1)/12)*32;
        for(int i=0;i<2;++i){int weight=i==1;size_t local_first=weight?first:first/32;
            size_t local=weight?end-first:(end-first)/32;
            memset(&shared_items[i],0,sizeof shared_items[i]);
            strcpy(shared_items[i].name,shared_names[i]);strcpy(shared_items[i].dtype,shared_dtype[i]);
            shared_items[i].rows=local;shared_items[i].cols=shared_cols[i];
            if(fault==2&&i==0)continue;
            if(fault==1&&i==0)++local_first;
            fprintf(f,"%s %zu %zu %zu %zu\n",shared_names[i],shared_global[i],shared_cols[i],local_first,local);
        }
        REQUIRE(!fclose(f));int rc=ds41f_weights_check_shared_tp(&shared,argv[1],12,rank);
        REQUIRE(fault?rc==EINVAL:rc==0);
        if(!fault){for(int i=0;i<2;++i){REQUIRE(shared_items[i].global_rows==shared_global[i]);
                REQUIRE(shared_items[i].row_start==(i?first:first/32));}}
        ++cases;
    }
    REQUIRE(!unlink(shared_path)&&!rmdir(argv[1]));
    REQUIRE(!mkdir(argv[1],0700));
    char attention_path[4096];snprintf(attention_path,sizeof attention_path,"%s/weights.attention.tp",argv[1]);
    const char *attention_names[]={"layers.0.attn.wo_a.scale","layers.0.attn.wo_a.weight",
                                   "layers.0.attn.wo_b.scale","layers.0.attn.wo_b.weight",
                                   "layers.0.attn.wq_b.scale","layers.0.attn.wq_b.weight"};
    const char *attention_dtype[]={"F8_E8M0","F8_E4M3","F8_E8M0","F8_E4M3","F8_E8M0","F8_E4M3"};
    const size_t attention_global[]={256,8192,160,5120,1024,32768};
    const size_t attention_cols[]={128,4096,256,8192,40,1280};
    ds41f_weight attention_items[6]={0};ds41f_weights attention={.items=attention_items,.count=6};
    for(int rank=0;rank<12;++rank)for(int fault=0;fault<3;++fault){
        FILE *f=fopen(attention_path,"w");REQUIRE(f);fprintf(f,"DS41FA 1 8 %d 12\n",rank);
        for(int i=0;i<6;++i){int head=strstr(attention_names[i],"wq_b")||strstr(attention_names[i],"wo_a");int weight=i&1;int want=!head||rank<8;
            memset(&attention_items[i],0,sizeof attention_items[i]);strcpy(attention_items[i].name,attention_names[i]);
            strcpy(attention_items[i].dtype,attention_dtype[i]);
            if(!want){attention_items[i].rows=attention_items[i].cols=0;continue;}
            size_t align=(!head&&weight)?32:1,degree=head?8:12;
            size_t blocks=attention_global[i]/align,first=(blocks*(size_t)rank/degree)*align;
            size_t end=(blocks*(size_t)(rank+1)/degree)*align;
            size_t local=end-first;if(!weight&&align==32){first/=32;local/=32;}
            attention_items[i].rows=local;attention_items[i].cols=attention_cols[i];
            if(fault==1&&i==2)++first;
            if(fault==2&&i==3)continue;
            fprintf(f,"%s %zu %zu %zu %zu\n",attention_names[i],attention_global[i],attention_cols[i],first,local);
        }
        REQUIRE(!fclose(f));int rc=ds41f_weights_check_attention_tp(&attention,argv[1],rank);
        REQUIRE(fault?rc==EINVAL:rc==0);++cases;
    }
    REQUIRE(!unlink(attention_path)&&!rmdir(argv[1]));
    printf("TP_WEIGHTS PASS cases=%zu backbone_head Markov_vocab main_projection shared_fp8_scale_ranges attention_tp12 missing duplicate\n",cases);return 0;
}
