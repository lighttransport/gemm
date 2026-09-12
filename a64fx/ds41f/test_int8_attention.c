#include "ds41f_attention.h"
#include "ds41f_weights.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int input(const char *prefix,int layer,int pos,float x[5120])
{
    char path[4096],magic[8],label[32];
    snprintf(path,sizeof path,"%s.pos%d.layer%d.bin",prefix,pos,layer);
    FILE *f=fopen(path,"rb");if(!f)return 1;
    if(fread(magic,1,8,f)!=8||memcmp(magic,"DS41FD1\0",8)){fclose(f);return 1;}
    while(fread(label,1,32,f)==32){uint64_t count;
        if(fread(&count,8,1,f)!=1||count>20480){fclose(f);return 1;}
        if(!memcmp(label,"attn_input\0",11)){
            int rc=count!=5120||fread(x,4,5120,f)!=5120;fclose(f);return rc;}
        if(fseek(f,(long)count*4,SEEK_CUR)){fclose(f);return 1;}
    }
    fclose(f);return 1;
}
int main(int argc,char **argv)
{
    if(argc!=6){fprintf(stderr,"usage: %s STAGE DUMP_PREFIX LAYER POSITIONS BLOCK\n",argv[0]);return 2;}
    int layer=atoi(argv[3]),positions=atoi(argv[4]);size_t block=strtoul(argv[5],NULL,10);
    if((layer!=0&&layer!=1&&layer!=2&&layer!=8&&layer!=14&&layer!=20)||positions<1||positions>64)return 2;
    ds41f_weights w;char prefix[96];snprintf(prefix,sizeof prefix,"layers.%d.attn.",layer);
    int rc=ds41f_weights_load(&w,argv[1],prefix,512*1024*1024);
    if(rc){fprintf(stderr,"weights rc=%d\n",rc);return 1;}
    float *reference=malloc((size_t)positions*5120*sizeof(float));if(!reference)return 2;
    int failures=0;
    for(int mode=0;mode<2;++mode){
        if(mode&&(rc=ds41f_weights_requantize_fp8(&w,block,512*1024*1024,0))){fprintf(stderr,"quant rc=%d\n",rc);return 1;}
        ds41f_attention a;if(ds41f_attention_init(&a,(size_t)positions))return 1;
        for(int pos=0;pos<positions;++pos){float x[5120],out[5120];
            if(input(argv[2],layer,pos,x)||(rc=ds41f_attention_step(&a,&w,layer,(size_t)pos,x,out))){fprintf(stderr,"attention rc=%d\n",rc);return 1;}
            if(!mode)memcpy(reference+(size_t)pos*5120,out,sizeof out);
            else{double aa=0,bb=0,ab=0,error=0;
                for(int i=0;i<5120;++i){double u=reference[(size_t)pos*5120+i],v=out[i];
                    aa+=u*u;bb+=v*v;ab+=u*v;error+=(u-v)*(u-v);}
                double cosine=ab/sqrt(aa*bb),relative=sqrt(error/aa);
                int ok=isfinite(cosine)&&cosine>=.999;
                printf("INT8_ATTENTION %s layer=%d pos=%d block=%zu cosine=%.9f relative_rms=%.9g\n",ok?"PASS":"FAIL",layer,pos,block,cosine,relative);
                failures+=!ok;
            }
        }
        ds41f_attention_free(&a);
    }
    free(reference);ds41f_weights_free(&w);return failures?1:0;
}
