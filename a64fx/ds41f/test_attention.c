#include "ds41f_attention.h"
#include "ds41f_weights.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc,char **argv)
{
    if(argc<3||argc>5){fprintf(stderr,"usage: %s stage output.bin [source_layer [positions]]\n",argv[0]);return 2;}
    int layer=argc>3?atoi(argv[3]):0,positions=argc>4?atoi(argv[4]):3;
    if((layer!=0&&layer!=2&&layer!=8&&layer!=14&&layer!=20)||positions<1||positions>2048)return 2;
    ds41f_weights weights;
    char prefix[80];snprintf(prefix,sizeof prefix,"layers.%d.attn.",layer);
    int rc=ds41f_weights_load(&weights,argv[1],prefix,512*1024*1024);
    if(rc){fprintf(stderr,"load rc=%d\n",rc);return 1;}
    ds41f_attention state;if(ds41f_attention_init(&state,(size_t)positions))return 1;
    FILE *f=fopen(argv[2],"wb");if(!f)return 1;
    for(int token=0;token<positions;++token){float x[5120],out[5120];
        for(int i=0;i<5120;++i)x[i]=sinf((float)i*.013f+(float)token*.07f);
        ds41f_round_bf16(x,5120);
        rc=ds41f_attention_step(&state,&weights,layer,token,x,out);
        if(rc){fprintf(stderr,"attention rc=%d\n",rc);return 1;}
        double norm=0;for(int i=0;i<5120;++i){if(!isfinite(out[i]))return 1;norm+=(double)out[i]*out[i];}
        if(fwrite(out,sizeof(float),5120,f)!=5120)return 1;
        printf("ATTENTION_SMOKE layer=%d token=%d norm=%g selected=%zu\n",layer,token,sqrt(norm),state.selected_count);
    }
    fclose(f);ds41f_attention_free(&state);ds41f_weights_free(&weights);return 0;
}
