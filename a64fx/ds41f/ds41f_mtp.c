#include "ds41f_mtp.h"
#include "ds41f_comm.h"
#include "ds41f_expert.h"
#include "ds41f_kernels.h"
#include "ds41f_ops.h"
#include "ds41f_sve.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do{int rc_=(x);if(rc_)return rc_;}while(0)
#define DIM 5120
#define STREAMS 20480
#define BLOCK DS41F_DRAFT_BLOCK
static const ds41f_weight *find(const ds41f_mtp *m,int stage,const char *part)
{char name[192];snprintf(name,sizeof name,"mtp.%d.%s",stage,part);return ds41f_weight_find(&m->weights,name);}
static int linear(const ds41f_mtp *m,int stage,const char *part,float *out,const float *x,int raw)
{char name[192];snprintf(name,sizeof name,"mtp.%d.%s",stage,part);return ds41f_linear(&m->weights,name,out,x,raw);}
static int norm(const ds41f_mtp *m,int stage,const char *part,float *out,const float *x)
{char name[192];snprintf(name,sizeof name,"mtp.%d.%s.weight",stage,part);return ds41f_norm(&m->weights,name,out,x);}
static int member(const ds41f_mtp *m,int stage){return m->rank/4==stage;}
static void rope(float *x,size_t heads,size_t pos,int inverse)
{
    /* DSpark is uncompressed: theta=10000, no long-context correction. */
    ds41f_rope(x,heads,512,64,pos,10000,16,0,inverse);
    for(size_t h=0;h<heads;++h)ds41f_round_bf16(x+h*512+448,64);
}
static int mixes(const ds41f_mtp *m,int stage,const char *kind,const float *h,
                 float *pre,float *post,float *comb)
{
    char part[96];snprintf(part,sizeof part,"hc_%s_fn",kind);const ds41f_weight *fn=find(m,stage,part);
    snprintf(part,sizeof part,"hc_%s_base",kind);const ds41f_weight *base=find(m,stage,part);
    snprintf(part,sizeof part,"hc_%s_scale",kind);const ds41f_weight *scale=find(m,stage,part);
    if(!fn||!base||!scale||strcmp(fn->dtype,"F32")||fn->rows!=24||fn->cols!=STREAMS||base->bytes!=96||scale->bytes!=12)return EINVAL;
    float value[24];CHECK(ds41f_hc_matvec(value,fn->data,h,ds41f_hc_inverse_rms(h,STREAMS),m->hc_mode));
    ds41f_hc_split(value,scale->data,base->data,20,1e-6f,pre,post,comb);return 0;
}
static int grouped(const ds41f_mtp *m,int stage,float *out,const float *x)
{
    const ds41f_weight *w=find(m,stage,"attn.wo_a.weight"),*s=find(m,stage,"attn.wo_a.scale");
    if(!w||!s||w->rows!=2048||w->cols!=4096||s->bytes!=8192)return EINVAL;
    if(w->int8.weight)CHECK(ds41f_linear_int8_cached(&m->weights,w,out,x,1024,0));
    else CHECK(ds41f_fp8_grouped_matvec(out,w->data,s->data,x,2,1024,4096));
    ds41f_round_bf16(out,2048);return 0;
}
void ds41f_mtp_free(ds41f_mtp *m)
{if(m){ds41f_weights_free(&m->weights);free(m->window);free(m->workspace);memset(m,0,sizeof *m);}}
int ds41f_mtp_admission(const char *stage,int int8,size_t *bytes)
{
    if(!stage||!bytes)return EINVAL;
    char path[4096],line[1024];int n=snprintf(path,sizeof path,"%s/weights.index",stage);
    if(n<0||(size_t)n>=sizeof path)return ENAMETOOLONG;
    FILE *f=fopen(path,"r");if(!f)return errno;
    size_t total=0,peak=0,count=0;int rc=0;
    while(fgets(line,sizeof line,f)){char name[192],type[16],extra;size_t rows,cols,size;
        if(sscanf(line,"%191s %15s %zu %zu %zu %c",name,type,&rows,&cols,&size,&extra)!=5||
           strncmp(name,"mtp.",4)||!rows||!cols||!size||size>(size_t)1024*1024*1024){rc=EINVAL;break;}
        size_t converted=size;
        if(int8&&!strcmp(type,"F8_E4M3"))converted=size+size/8;
        if(converted>(size_t)1024*1024*1024){rc=EINVAL;break;}
        if(converted>peak)peak=converted;
        if(total>(size_t)1024*1024*1024-converted){rc=EINVAL;break;}
        total+=converted;++count;
    }
    if(ferror(f))rc=EIO;
    fclose(f);if(!count)rc=EINVAL;
    /* Include draft buffers, input cache, stack work and journal headroom. */
    if(!rc)*bytes=total+peak+(size_t)8*1024*1024;
    return rc;
}
int ds41f_mtp_load(ds41f_mtp *m,const ds41f_weights *backbone,const char *stage,
                   int rank,size_t budget,int int8,int expert_sdot,int hc_mode)
{
    if(!m||!backbone||!stage||rank<0||rank>=12||hc_mode<0||hc_mode>2)return EINVAL;
    memset(m,0,sizeof *m);m->rank=rank;m->backbone=backbone;m->hc_mode=hc_mode;m->expert_fused=1;
    /* Resident weights plus bounded draft work; never duplicate backbone head. */
    const size_t scratch=(size_t)(3*128*512+2*BLOCK*STREAMS+BLOCK*8192+BLOCK*DIM)*sizeof(float);
    if(budget<=scratch)return ENOMEM;
    int rc=ds41f_weights_load_local(&m->weights,stage,NULL,budget-scratch,1);
    if(!rc)rc=ds41f_weights_check_tp(&m->weights,stage,4,rank);
    if(!rc&&int8)rc=ds41f_weights_requantize_fp8(&m->weights,32,budget-scratch,0);
    if(!rc&&expert_sdot)rc=ds41f_weights_pack_experts(&m->weights,budget-scratch);
    if(!rc)rc=ds41f_weights_enable_input_cache(&m->weights);
    if(!rc){m->window=calloc((size_t)3*128*512,sizeof(float));
        m->workspace=calloc((size_t)2*BLOCK*STREAMS+BLOCK*8192+BLOCK*DIM,sizeof(float));
        if(!m->window||!m->workspace)rc=ENOMEM;}
    if(rc)ds41f_mtp_free(m);return rc;
}
int ds41f_mtp_commit(ds41f_mtp *m,const float *taps,size_t pos)
{
    if(!m||!m->window||!taps||pos!=m->committed||pos>=1048576)return EINVAL;
    float main_x[DIM],part[DIM/4];
    if(member(m,0)){
        CHECK(linear(m,0,"main_proj",part,taps,0));ds41f_comm_tp_gather(main_x,part,DIM/4,0);
        if(m->rank==0)CHECK(norm(m,0,"main_norm",main_x,main_x));}
    ds41f_comm_bf16_broadcast(main_x,DIM,0,0);
    for(int stage=0;stage<3;++stage)if(member(m,stage)){
        float kv[512];if(m->rank==stage*4){CHECK(linear(m,stage,"attn.wkv",kv,main_x,0));
            CHECK(norm(m,stage,"attn.kv_norm",kv,kv));rope(kv,1,pos,0);CHECK(ds41f_act_quant(kv,kv,512));}
        ds41f_comm_tp_bytes(kv,sizeof kv,stage*4);
        memcpy(m->window+((size_t)stage*128+pos%128)*512,kv,sizeof kv);
    }
    ++m->committed;return 0;
}
static int attention(const ds41f_mtp *m,int stage,const float *x,float *out)
{
    if(!member(m,stage))return 0;
    const int owner=stage*4;size_t start=m->committed;
    struct {float qr[BLOCK][1280],kv[BLOCK][512];} context;
    if(m->rank==owner)for(int i=0;i<BLOCK;++i){
        CHECK(linear(m,stage,"attn.wq_a",context.qr[i],x+i*DIM,0));CHECK(norm(m,stage,"attn.q_norm",context.qr[i],context.qr[i]));
        CHECK(linear(m,stage,"attn.wkv",context.kv[i],x+i*DIM,0));CHECK(norm(m,stage,"attn.kv_norm",context.kv[i],context.kv[i]));
        rope(context.kv[i],1,start+i,0);CHECK(ds41f_act_quant(context.kv[i],context.kv[i],512));}
    ds41f_comm_tp_bytes(&context,sizeof context,owner);
    /* Preserve the checkpoint's physical window order, followed by every
     * draft key. All five queries can attend all five noise-block positions. */
    size_t count=m->committed<128?m->committed:128;
    float rows[(128+BLOCK)*512];int ids[128+BLOCK];
    memcpy(rows,m->window+(size_t)stage*128*512,count*512*sizeof(float));
    memcpy(rows+count*512,context.kv,sizeof context.kv);
    for(size_t i=0;i<count+BLOCK;++i)ids[i]=(int)i;
    const ds41f_weight *sink=find(m,stage,"attn.attn_sink");if(!sink||sink->bytes!=64*4)return EINVAL;
    for(int i=0;i<BLOCK;++i){float q[16*512],attended[16*512],local[2048],projected[8192],part[DIM/4];
        CHECK(linear(m,stage,"attn.wq_b",q,context.qr[i],0));rope(q,16,start+i,0);
        CHECK(ds41f_sparse_attention_tiled_math(attended,q,rows,(float *)sink->data+(m->rank%4)*16,
            ids,count+BLOCK,count+BLOCK,16,512,4,0));
        ds41f_round_bf16(attended,16*512);rope(attended,16,start+i,1);CHECK(grouped(m,stage,local,attended));
        ds41f_comm_tp_allgather(projected,local,2048);CHECK(linear(m,stage,"attn.wo_b",part,projected,0));
        ds41f_comm_tp_gather(out+i*DIM,part,DIM/4,owner);
    }
    return 0;
}
static int experts(const ds41f_mtp *m,int stage,const float *x,const float *route,float *out)
{
    memset(out,0,DIM*sizeof(float));float scratch[3*2304+DIM],value[DIM];
    for(int k=0;k<3;++k){int id=(int)route[k];if(id<0||id>=128)return EINVAL;if(id%12!=m->rank)continue;
        ds41f_expert e={{0},{0},m->weights.packed_experts};char part[128];
        for(int i=0;i<3;++i){snprintf(part,sizeof part,"ffn.experts.%d.w%d.weight",id,i+1);const ds41f_weight *w=find(m,stage,part);
            snprintf(part,sizeof part,"ffn.experts.%d.w%d.scale",id,i+1);const ds41f_weight *s=find(m,stage,part);
            if(!w||!s)return ENOENT;e.weight[i]=w->data;e.scale[i]=s->data;}
        CHECK(ds41f_expert_forward_fused(&e,value,x,route[k+3],scratch,0,m->expert_fused));
        for(int j=0;j<DIM;++j)out[j]+=value[j];}
    return 0;
}
static int shared(const ds41f_mtp *m,int stage,const float *x,float *out)
{
    if(!member(m,stage))return 0;
    float gate[576],up[576],local[576],hidden[2304],part[DIM/4];
    CHECK(linear(m,stage,"ffn.shared_experts.w1",gate,x,0));CHECK(linear(m,stage,"ffn.shared_experts.w3",up,x,0));
    ds41f_swiglu(local,gate,up,576,10);ds41f_round_bf16(local,576);ds41f_comm_tp_allgather(hidden,local,576);
    CHECK(linear(m,stage,"ffn.shared_experts.w2",part,hidden,0));ds41f_comm_tp_gather(out,part,DIM/4,stage*4);return 0;
}
int ds41f_mtp_draft(ds41f_mtp *m,int seed,int output[6],float confidence[5])
{
    if(!m||!m->window||!m->workspace||!output||!confidence||seed<0||seed>=129280||m->committed<2||m->committed+BLOCK>1048576)return EINVAL;
    float *h=m->workspace,*x=h+BLOCK*STREAMS,*y=x+BLOCK*STREAMS;float pre[BLOCK][4]={{0}};
    for(int i=0;i<BLOCK;++i)pre[i][0]=1;
    if(m->rank==0){const ds41f_weight *embed=ds41f_weight_find(m->backbone,"embed.weight");
        if(!embed||strcmp(embed->dtype,"BF16")||embed->rows!=129280||embed->cols!=DIM)return EINVAL;
        for(int i=0;i<BLOCK;++i){const uint16_t *row=(uint16_t *)embed->data+(size_t)(i?128799:seed)*DIM;
            for(int c=0;c<4;++c)for(int j=0;j<DIM;++j)h[i*STREAMS+c*DIM+j]=ds41f_bf16_to_f32(row[j]);}}
    for(int stage=0;stage<3;++stage){int owner=stage*4;
        float attn_pre[BLOCK][4],attn_post[BLOCK][4],attn_comb[BLOCK][16];
        if(m->rank==owner)for(int i=0;i<BLOCK;++i){
            CHECK(mixes(m,stage,"attn",h+i*STREAMS,attn_pre[i],attn_post[i],attn_comb[i]));
            ds41f_hc_pre(x+i*DIM,h+i*STREAMS,pre[i],DIM);ds41f_round_bf16(x+i*DIM,DIM);CHECK(norm(m,stage,"attn_norm",x+i*DIM,x+i*DIM));}
        CHECK(attention(m,stage,x,y));
        for(int i=0;i<BLOCK;++i){float post[4],comb[16],next_pre[4],packet[DIM+6];
            if(m->rank==owner){
                ds41f_hc_post(h+i*STREAMS,y+i*DIM,h+i*STREAMS,attn_post[i],attn_comb[i],DIM);ds41f_round_bf16(h+i*STREAMS,STREAMS);
                CHECK(mixes(m,stage,"ffn",h+i*STREAMS,next_pre,post,comb));
                ds41f_hc_pre(packet,h+i*STREAMS,attn_pre[i],DIM);ds41f_round_bf16(packet,DIM);CHECK(norm(m,stage,"ffn_norm",packet,packet));
                float logits[128],prob[3];int ids[3];CHECK(linear(m,stage,"ffn.gate",logits,packet,1));
                const ds41f_weight *bias=find(m,stage,"ffn.gate.bias");if(!bias)return ENOENT;
                CHECK(ds41f_gate(logits,bias->data,128,3,1,1.5f,ids,prob));
                for(int k=0;k<3;++k){packet[DIM+k]=(float)ids[k];packet[DIM+k+3]=prob[k];}}
            ds41f_comm_bf16_broadcast(packet,DIM,6,owner);float combined[DIM],shared_value[DIM];
            CHECK(experts(m,stage,packet,packet+DIM,combined));CHECK(shared(m,stage,packet,shared_value));ds41f_comm_sum(combined,DIM);
            if(m->rank==owner){for(int j=0;j<DIM;++j)combined[j]+=shared_value[j];ds41f_round_bf16(combined,DIM);
                ds41f_hc_post(h+i*STREAMS,combined,h+i*STREAMS,post,comb,DIM);ds41f_round_bf16(h+i*STREAMS,STREAMS);memcpy(pre[i],next_pre,sizeof next_pre);}
        }
        if(stage<2)for(int i=0;i<BLOCK;++i){ds41f_comm_bf16_handoff(h+i*STREAMS,STREAMS,0,owner,owner+4);
            ds41f_comm_bf16_handoff(pre[i],0,4,owner,owner+4);}
    }
    const ds41f_weight *head=ds41f_weight_find(m->backbone,"head.weight"),*markov=find(m,2,"markov_head.head.weight");
    if(!head||!markov||head->rows!=markov->rows||head->row_start!=markov->row_start)return EINVAL;
    float *logits=malloc(2*head->rows*sizeof(float));if(!logits)return ENOMEM;
    output[0]=seed;
    for(int i=0;i<BLOCK;++i){float collapsed[DIM],normalized[DIM],markov_x[256];
        if(m->rank==8){ds41f_hc_pre(collapsed,h+i*STREAMS,pre[i],DIM);ds41f_round_bf16(collapsed,DIM);CHECK(norm(m,2,"norm",normalized,collapsed));}
        ds41f_comm_bf16_broadcast(normalized,DIM,0,8);CHECK(ds41f_linear(m->backbone,"head",logits,normalized,1));
        int token=output[i],owner=0;while(owner<11&&token>=(129280/32*(owner+1)/12)*32)++owner;
        if(m->rank==owner){const ds41f_weight *embed=find(m,2,"markov_head.embed.weight");
            if(!embed||token<(int)embed->row_start||(size_t)token>=embed->row_start+embed->rows){free(logits);return EINVAL;}
            const uint16_t *row=(uint16_t *)embed->data+((size_t)token-embed->row_start)*256;
            for(int j=0;j<256;++j)markov_x[j]=ds41f_bf16_to_f32(row[j]);}
        ds41f_comm_bf16_broadcast(markov_x,256,0,owner);CHECK(linear(m,2,"markov_head.head",logits+head->rows,markov_x,1));
        for(size_t j=0;j<head->rows;++j)logits[j]+=logits[head->rows+j];
        size_t best;CHECK(ds41f_argmax_finite(logits,head->rows,&best));float value=logits[best];int id=(int)(best+head->row_start);
        ds41f_comm_argmax(&value,&id);output[i+1]=id;
        if(m->logits_prefix){float *full=m->rank==11?malloc(129280*sizeof(float)):NULL;
            if(m->rank==11&&!full){free(logits);return ENOMEM;}
            ds41f_comm_head_logits(full,logits,head->rows);
            if(m->rank==11){char path[4096];int n=snprintf(path,sizeof path,"%s.pos%zu.step%d.bin",m->logits_prefix,m->committed-1,i);
                if(n<0||(size_t)n>=sizeof path){free(full);free(logits);return ENAMETOOLONG;}
                FILE *f=fopen(path,"wx");if(!f){free(full);free(logits);return errno;}
                int bad=fwrite(full,sizeof(float),129280,f)!=129280;if(fclose(f))bad=1;
                if(bad){free(full);free(logits);return EIO;}}
            free(full);}

        if(m->rank==8){float confidence_x[DIM+256];memcpy(confidence_x,collapsed,sizeof collapsed);memcpy(confidence_x+DIM,markov_x,sizeof markov_x);
            CHECK(linear(m,2,"confidence_head.proj",confidence+i,confidence_x,1));if(!isfinite(confidence[i])){free(logits);return EDOM;}}
    }
    ds41f_comm_broadcast(confidence,BLOCK,8);free(logits);return 0;
}
