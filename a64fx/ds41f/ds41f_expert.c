#include "ds41f_expert.h"
#include "ds41f_fp4_sdot.h"
#include "ds41f_tensor.h"
#include "ds41f_sve.h"
#include "ds41f_kernels.h"
#include "ds41f_ops.h"
#include "ds41f_profile.h"
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

void ds41f_expert_free(ds41f_expert *e)
{
    if (!e) return;
    for (int i=0;i<3;++i) {free(e->weight[i]);free(e->scale[i]);}
    memset(e,0,sizeof *e);
}
int ds41f_expert_load(ds41f_expert *e,const char *stage,int layer,int id)
{
    if (!e || !stage || layer<0 || layer>=40 || id<0 || id>=384) return EINVAL;
    memset(e,0,sizeof *e);
    const size_t values=(size_t)5120*2304;
    for (int i=0;i<3;++i) {
        char name[256]; void *data=NULL;
        snprintf(name,sizeof name,"layers.%d.ffn.experts.%d.w%d.weight",layer,id,i+1);
        int rc=ds41f_tensor_load(stage,name,values/2,&data);
        e->weight[i]=data;
        if (rc) {ds41f_expert_free(e);return rc;}
        snprintf(name,sizeof name,"layers.%d.ffn.experts.%d.w%d.scale",layer,id,i+1);
        rc=ds41f_tensor_load(stage,name,values/32,&data);
        e->scale[i]=data;
        if (rc) {ds41f_expert_free(e);return rc;}
    }
    return 0;
}
int ds41f_expert_forward_fused(const ds41f_expert *e,float *out,const float *x,
                          float route_weight,float *scratch,int reference,int fused)
{
    if (!e || !out || !x || !scratch||fused<0||fused>2) return EINVAL;
    for (int i=0;i<3;++i) if (!e->weight[i] || !e->scale[i]) return EINVAL;
    int (*mv)(float *,const uint8_t *,const uint8_t *,const float *,size_t,size_t)=
        reference?ds41f_mxfp4_matvec_ref:ds41f_mxfp4_matvec;
    float *gate=scratch,*up=scratch+2304,*hidden=scratch+4608;
    float *input=scratch+6912;
    double pt=P_BEGIN();P_VALUE(EXPERT_COUNT,1);P_VALUE(FP4_BYTES,3*((size_t)5120*2304/2+(size_t)5120*2304/32));
    int rc=ds41f_act_quant(input,x,5120);
    if (rc) return rc;
    ds41f_int8_input prepared={0};
    if(e->packed_sdot){rc=ds41f_int8_prepare_input(&prepared,input,5120,32);if(rc)return rc;}
    P_END(EXPERT_QUANT,pt);pt=P_BEGIN();
    if(e->packed_sdot){
        if(fused&&!reference)rc=ds41f_mxfp4_sdot_pair_prepared(gate,up,e->weight[0],e->scale[0],e->weight[2],e->scale[2],&prepared,2304,5120);
        else{rc=ds41f_mxfp4_sdot_prepared(gate,e->weight[0],e->scale[0],&prepared,2304,5120,reference);
            rc|=ds41f_mxfp4_sdot_prepared(up,e->weight[2],e->scale[2],&prepared,2304,5120,reference);}
        ds41f_int8_input_free(&prepared);
    }else if(fused&&!reference)rc=ds41f_mxfp4_matvec_pair(gate,up,e->weight[0],e->scale[0],e->weight[2],e->scale[2],input,2304,5120,fused);
    else{rc=mv(gate,e->weight[0],e->scale[0],input,2304,5120);
        rc|=mv(up,e->weight[2],e->scale[2],input,2304,5120);}
    if (rc) return rc;
    P_END(EXPERT_W13,pt);pt=P_BEGIN();
    ds41f_round_bf16(gate,2304);ds41f_round_bf16(up,2304);
    P_END(EXPERT_ROUND,pt);pt=P_BEGIN();ds41f_swiglu(hidden,gate,up,2304,10);
    for (int i=0;i<2304;++i) hidden[i]*=route_weight;
    P_END(EXPERT_SWIGLU,pt);pt=P_BEGIN();rc=ds41f_act_quant(hidden,hidden,2304);
    if (rc) return rc;
    if(e->packed_sdot){rc=ds41f_int8_prepare_input(&prepared,hidden,2304,32);if(rc)return rc;}
    P_END(EXPERT_QUANT,pt);pt=P_BEGIN();
    if(e->packed_sdot){rc=ds41f_mxfp4_sdot_prepared(out,e->weight[1],e->scale[1],&prepared,5120,2304,reference);ds41f_int8_input_free(&prepared);}
    else rc=mv(out,e->weight[1],e->scale[1],hidden,5120,2304);
    P_END(EXPERT_W2,pt);pt=P_BEGIN();
    ds41f_round_bf16(out,5120);
    P_END(EXPERT_ROUND,pt);return rc;
}

int ds41f_expert_forward(const ds41f_expert *e,float *out,const float *x,
                          float route_weight,float *scratch,int reference)
{return ds41f_expert_forward_fused(e,out,x,route_weight,scratch,reference,0);}
