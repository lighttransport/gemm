#include "ds41f_attention.h"
#include "ds41f_cache.h"
#include "ds41f_weights.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Real query projections against synthetic history: exercise >2048 blocks
 * without claiming that a full model has processed a 32K-token context. */
int main(int argc,char **argv)
{
    if(argc!=3)return 2;
    int layer=atoi(argv[2]);if(layer!=20&&layer!=24)return 2;
    char prefix[80];snprintf(prefix,sizeof prefix,"layers.%d.attn.",layer);
    ds41f_weights weights;
    if(ds41f_weights_load(&weights,argv[1],prefix,512*1024*1024))return 1;
    ds41f_attention state;
    const size_t count=32768;
    if(ds41f_attention_init(&state,count))return 1;
    uint8_t prototypes[16][356];float values[512],keys[128];
    for(int p=0;p<16;++p){
        for(int j=0;j<512;++j)values[j]=(float)((j+p*7)%17-8)/8;
        for(int j=0;j<128;++j)keys[j]=(float)((j+p*3)%13-6)/8;
        if(ds41f_fp4_pack(prototypes[p],values,512,16,1)||
           ds41f_fp4_pack(prototypes[p]+288,keys,128,32,0))return 1;
    }
    for(size_t i=0;i<count;++i)memcpy(state.compressed[3]+i*356,prototypes[i%16],356);
    if(layer==24)for(size_t b=0;b<8;++b)state.candidate_blocks[b*503]=1;
    float x[5120],out[5120];
    for(int j=0;j<5120;++j)x[j]=sinf(j*.013f);
    ds41f_round_bf16(x,5120);
    if(ds41f_attention_step(&state,&weights,layer,count-1,x,out))return 1;
    size_t kept=0;for(size_t b=0;b<count/8;++b)kept+=state.candidate_blocks[b]!=0;
    if(layer==20&&(kept!=2048||!state.candidate_blocks[count/8-1]||state.selected_count!=512))return 1;
    if(layer==24&&(kept!=8||state.selected_count!=64))return 1;
    for(size_t i=0;i<state.selected_count;++i){int id=state.selected[i];
        if(id<0||(size_t)id>=count||(i&&id<=state.selected[i-1]))return 1;
        if(layer==24&&!state.candidate_blocks[(size_t)id/8])return 1;
    }
    for(int j=0;j<5120;++j)if(!isfinite(out[j]))return 1;
    printf("INDEX_CANDIDATES PASS layer=%d synthetic_history=%zu blocks=%zu selected=%zu\n",layer,count,kept,state.selected_count);
    ds41f_attention_free(&state);ds41f_weights_free(&weights);return 0;
}
