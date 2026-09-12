#define _GNU_SOURCE
#include "ds41f_attention.h"
#include "ds41f_cache.h"
#include "ds41f_comm.h"
#include "ds41f_team.h"
#include "ds41f_mtp.h"
#include "ds41f_journal.h"
#include "ds41f_engram.h"
#include "ds41f_expert.h"
#include "ds41f_kernels.h"
#include "ds41f_ops.h"
#include "ds41f_profile.h"
#include "ds41f_prefetch.h"
#include "ds41f_weights.h"
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <omp.h>
#include <sched.h>

static int rank,ranks,dense_tp=1,expert_fused,verify_expert_batch,verify_timing,expert_input_cache,verify_comm_batch;
static ds41f_weights weights;
static ds41f_mtp mtp;
static int mtp_enabled,mtp_probe,speculate,spec_force_reject=-1,verify_batch,verify_check,verify_replay_batch,dump_state_hash;
static const char *mtp_dump;
static float main_taps[15360];
static ds41f_attention attention;
static ds41f_engram engram;
static ds41f_prefetch *prefetch;
static int hc_matvec_mode,hc_mix_sve, shared_overlap,compact_comm;
static double profile_attention,profile_expert,profile_shared,profile_head;
static const char *dump_prefix;
static size_t dump_count=1;
#define CHECK(call) do {int rc_=(call);if(rc_)ds41f_comm_abort(#call,rc_);} while(0)
/* Bounded, owner-only FP32 records for independent same-input replay. */
static void dump_record(FILE *f,const char *name,const float *x,size_t n)
{
    if(!f)return;
    char label[32]={0};uint64_t count=n;
    if(strlen(name)>=sizeof label)ds41f_comm_abort("dump label",EINVAL);
    memcpy(label,name,strlen(name));
    if(fwrite(label,1,sizeof label,f)!=sizeof label||
       fwrite(&count,sizeof count,1,f)!=1||fwrite(x,sizeof(float),n,f)!=n)
        ds41f_comm_abort("write intermediate dump",EIO);
}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static size_t available(void)
{
    FILE *f=fopen("/proc/meminfo","r");if(!f)return 0;
    char line[256];unsigned long long kb=0;
    while(fgets(line,sizeof line,f))if(sscanf(line,"MemAvailable: %llu kB",&kb)==1)break;
    fclose(f);return (size_t)kb*1024;
}
static size_t weight_prepare_limit(void)
{
    /* KV, Engram scales and MTP are allocated after backbone preparation.
     * Their future reservation must not also consume the transient budget
     * while one source tensor and its replacement coexist. Keep the actual
     * 2 GiB floor here, then recheck each later allocation and final residency. */
    size_t memory=available(),floor=(size_t)2*1024*1024*1024;
    if(memory<=floor||weights.bytes>SIZE_MAX-(memory-floor))
        ds41f_comm_abort("weight preparation memory guard",ENOMEM);
    size_t limit=weights.bytes+memory-floor;
    fprintf(stderr,"WEIGHT_PREP_ADMISSION rank=%d resident=%zu available=%zu limit=%zu floor=%zu\n",
        rank,weights.bytes,memory,limit,floor);
    return limit;
}
static const ds41f_weight *tensor(const char *name)
{
    const ds41f_weight *w=ds41f_weight_find(&weights,name);
    if(!w||!w->data)ds41f_comm_abort(name,ENOENT);
    return w;
}
static void named_linear(int layer,const char *part,float *out,const float *x,int raw)
{char name[192];snprintf(name,sizeof name,"layers.%d.%s",layer,part);CHECK(ds41f_linear(&weights,name,out,x,raw));}
static void named_norm(int layer,const char *part,float *out,const float *x)
{char name[192];snprintf(name,sizeof name,"layers.%d.%s.weight",layer,part);CHECK(ds41f_norm(&weights,name,out,x));}
static void mixes(int layer,const char *kind,const float *h,float pre[4],float post[4],float comb[16])
{
    char name[192];snprintf(name,sizeof name,"layers.%d.hc_%s_fn",layer,kind);
    const ds41f_weight *fn=tensor(name);
    if(strcmp(fn->dtype,"F32")||fn->rows!=24||fn->cols!=20480)ds41f_comm_abort("HC fn geometry",EINVAL);
    double pt=P_BEGIN();float inv;
    if(hc_mix_sve)inv=ds41f_hc_inverse_rms(h,20480);
    else{double ss=0;for(int i=0;i<20480;++i)ss+=(double)h[i]*h[i];inv=(float)(1/sqrt(ss/20480+1e-20));}
    P_END(HC_NORM,pt);pt=P_BEGIN();P_VALUE(F32_BYTES,fn->bytes);
    float mix[24];CHECK(ds41f_hc_matvec(mix,fn->data,h,inv,hc_matvec_mode));
    P_END(HC_MATVEC,pt);P_END(LINEAR_F32,pt);pt=P_BEGIN();
    snprintf(name,sizeof name,"layers.%d.hc_%s_base",layer,kind);const float *base=tensor(name)->data;
    snprintf(name,sizeof name,"layers.%d.hc_%s_scale",layer,kind);const float *scale=tensor(name)->data;
    ds41f_hc_split(mix,scale,base,20,1e-6f,pre,post,comb);P_END(HC_SPLIT,pt);
}
static void engram_step(int layer,int slot,float *h,const uint64_t ids[24])
{
    double pt=P_BEGIN();
    float rows[24*256]={0};uint16_t row[256];
    if(prefetch){double read_seconds=0;CHECK(ds41f_prefetch_wait(prefetch,slot,rows,&read_seconds));
        P_VALUE(ENGRAM_PREFETCH,read_seconds);
    }else for(int i=0;i<24;++i){ds41f_engram_table *table=&engram.table[slot];
        if(ids[i]>=table->first&&ids[i]<table->first+table->owned_rows){
            CHECK(ds41f_engram_read_local(&engram,slot,ids[i],row));
            for(int j=0;j<256;++j)rows[i*256+j]=ds41f_bf16_to_f32(row[j]);}}
    P_END(ENGRAM_IO,pt);pt=P_BEGIN();
    ds41f_comm_sum(rows,24*256);P_END(ENGRAM_SUM,pt);pt=P_BEGIN();
    if(rank==layer%12){float kv[5*5120];named_linear(layer,"engram.wkv",kv,rows,0);
        char name[192];snprintf(name,sizeof name,"layers.%d.engram.q_weight",layer);const uint16_t *qw=tensor(name)->data;
        snprintf(name,sizeof name,"layers.%d.engram.k_weight",layer);const uint16_t *kw=tensor(name)->data;
        float q[4*5120],k[4*5120];for(int i=0;i<4*5120;++i){q[i]=ds41f_bf16_to_f32(qw[i]);k[i]=ds41f_bf16_to_f32(kw[i]);}
        ds41f_engram_fuse(h,kv,kv+4*5120,q,k,5120,1e-20f);ds41f_round_bf16(h,20480);P_END(ENGRAM_PROJECT,pt);}
}
static void sync_attention(int layer,size_t pos)
{
    int owner=layer%12,is_source=layer==2||layer==8||layer==14||layer==20;
    int ratio=layer<20?2:1;
    if(is_source&&(pos+1)%ratio==0){
        if(compact_comm){
            ds41f_comm_bytes(attention.publication,356,owner);
            if(rank!=owner)CHECK(ds41f_attention_receive(&attention,layer,pos,attention.publication));
        }else{float row[356];
        if(rank==owner)for(int i=0;i<356;++i)row[i]=attention.publication[i];
        ds41f_comm_broadcast(row,356,owner);
        if(rank!=owner){uint8_t bytes[356];for(int i=0;i<356;++i)bytes[i]=(uint8_t)row[i];CHECK(ds41f_attention_receive(&attention,layer,pos,bytes));}}}
    if(is_source||layer==24||layer==28||layer==32||layer==36){float selection[513]={0};
        if(rank==owner){selection[0]=(float)attention.selected_count;for(size_t i=0;i<attention.selected_count;++i)selection[i+1]=(float)attention.selected[i];}
        ds41f_comm_broadcast(selection,513,owner);attention.selected_count=(size_t)selection[0];
        if(attention.selected_count>512)ds41f_comm_abort("selection count",EINVAL);
        for(size_t i=0;i<attention.selected_count;++i)attention.selected[i]=(int)selection[i+1];}
    if(layer==20){float blocks[2049]={0};size_t count=0;
        if(rank==owner){for(size_t b=0;b<(pos+8)/8;++b)if(attention.candidate_blocks[b]){
            if(count>=2048)ds41f_comm_abort("candidate count",EINVAL);
            blocks[++count]=(float)b;}blocks[0]=(float)count;}
        ds41f_comm_broadcast(blocks,2049,owner);
        if(rank!=owner){memset(attention.candidate_blocks,0,(attention.capacity+7)/8);
            for(size_t i=0;i<(size_t)blocks[0];++i)attention.candidate_blocks[(size_t)blocks[i+1]]=1;}}
}
static void local_experts(int layer,const float *input,const float route[12],float out[5120])
{
    memset(out,0,5120*sizeof(float));float scratch[3*2304+5120],value[5120],quantized[5120];
    ds41f_int8_input prepared={0};int needed=0;
    for(int k=0;k<6;++k)if((int)route[k]%12==rank)needed=1;
    if(expert_input_cache&&needed){double pt=P_BEGIN();CHECK(ds41f_act_quant(quantized,input,5120));
        if(weights.packed_experts)CHECK(ds41f_int8_prepare_input(&prepared,quantized,5120,32));P_END(EXPERT_QUANT,pt);}
    for(int k=0;k<6;++k){int id=(int)route[k];if(id%12!=rank)continue;
        ds41f_expert e={{0},{0},0};e.packed_sdot=weights.packed_experts;char name[192];
        for(int i=0;i<3;++i){snprintf(name,sizeof name,"layers.%d.ffn.experts.%d.w%d.weight",layer,id,i+1);e.weight[i]=tensor(name)->data;
            snprintf(name,sizeof name,"layers.%d.ffn.experts.%d.w%d.scale",layer,id,i+1);e.scale[i]=tensor(name)->data;}
        if(expert_input_cache)CHECK(ds41f_expert_forward_prepared(&e,value,quantized,weights.packed_experts?&prepared:NULL,route[k+6],scratch,0,expert_fused));
        else CHECK(ds41f_expert_forward_fused(&e,value,input,route[k+6],scratch,0,expert_fused));
        for(int i=0;i<5120;++i)out[i]+=value[i];}
    ds41f_int8_input_free(&prepared);
}
static void shared_expert(int layer,const float *input,float *out)
{
    float gate[2304],up[2304],hidden[2304];
    named_linear(layer,"ffn.shared_experts.w1",gate,input,0);named_linear(layer,"ffn.shared_experts.w3",up,input,0);
    ds41f_swiglu(hidden,gate,up,2304,10);ds41f_round_bf16(hidden,2304);
    named_linear(layer,"ffn.shared_experts.w2",out,hidden,0);
}
static int tp_member(int owner){return rank/dense_tp==owner/dense_tp;}
static void parallel_attention(int layer,size_t pos,const float *x,float *out)
{
    int owner=layer%12;if(!tp_member(owner))return;
    ds41f_attention_context context;float local[8192],projected[8192],part[5120];
    if(rank==owner)CHECK(ds41f_attention_prepare(&attention,&weights,layer,pos,x,&context));
    double pt=P_BEGIN();ds41f_comm_tp_bytes(&context,sizeof context,owner);P_END(TP_COMM,pt);
    if(rank!=owner)CHECK(ds41f_attention_apply(&attention,layer,pos,&context));
    CHECK(ds41f_attention_project(&attention,&weights,layer,pos,context.qr,
        (size_t)(rank%dense_tp)*(64/dense_tp),64/dense_tp,local));
    pt=P_BEGIN();ds41f_comm_tp_allgather(projected,local,8192/dense_tp);P_END(TP_COMM,pt);
    CHECK(ds41f_attention_output(&weights,layer,projected,part));
    pt=P_BEGIN();ds41f_comm_tp_gather(out,part,5120/dense_tp,owner);P_END(TP_COMM,pt);
}
static void parallel_shared(int layer,const float *input,float *out)
{
    int owner=layer%12;if(!tp_member(owner))return;
    size_t width=2304/dense_tp;float gate[2304],up[2304],local[2304],hidden[2304],part[5120];
    named_linear(layer,"ffn.shared_experts.w1",gate,input,0);
    named_linear(layer,"ffn.shared_experts.w3",up,input,0);
    ds41f_swiglu(local,gate,up,width,10);ds41f_round_bf16(local,width);
    double pt=P_BEGIN();ds41f_comm_tp_allgather(hidden,local,width);P_END(TP_COMM,pt);
    named_linear(layer,"ffn.shared_experts.w2",part,hidden,0);
    pt=P_BEGIN();ds41f_comm_tp_gather(out,part,5120/dense_tp,owner);P_END(TP_COMM,pt);
}
static int forward(int token,size_t pos,int trace,const char *logits_path)
{
    ds41f_profile_at(pos,0);double token_start=P_BEGIN(),pt=P_BEGIN();
    /* These pairs share an owner and a lifetime. Publish each contiguous
     * packet once, avoiding two tiny all-reduces per layer. */
    float residual_packet[20484],*h=residual_packet,*pre_mix=residual_packet+20480;
    pre_mix[0]=1;pre_mix[1]=pre_mix[2]=pre_mix[3]=0;
    if(rank==0){const ds41f_weight *embedding=tensor("embed.weight");
        if(token<0||token>=129280||strcmp(embedding->dtype,"BF16"))ds41f_comm_abort("token/embedding",EINVAL);
        const uint16_t *row=(uint16_t *)embedding->data+(size_t)token*5120;
        for(int c=0;c<4;++c)for(int j=0;j<5120;++j)h[c*5120+j]=ds41f_bf16_to_f32(row[j]);P_END(EMBED,pt);}
    pt=P_BEGIN();if(!compact_comm)ds41f_comm_broadcast(residual_packet,20484,0);P_END(EMBED_BCAST,pt);
    uint64_t hashes[2][24];CHECK(ds41f_engram_hash_ids(&engram,(uint32_t)token,hashes));
    if(prefetch)CHECK(ds41f_prefetch_submit(prefetch,(const uint64_t (*)[24])hashes));
    for(int layer=0;layer<40;++layer){int owner=layer%12;
        ds41f_profile_at(pos,layer);
        FILE *dump=NULL;
        if(rank==owner&&dump_prefix&&pos<dump_count){char path[4096];
            int n=snprintf(path,sizeof path,"%s.pos%zu.layer%d.bin",dump_prefix,pos,layer);
            if(n<0||(size_t)n>=sizeof path)ds41f_comm_abort("dump path",ENAMETOOLONG);
            dump=fopen(path,"wx");if(!dump)ds41f_comm_abort("open intermediate dump",errno);
            if(fwrite("DS41FD1\0",1,8,dump)!=8)ds41f_comm_abort("dump header",EIO);
            dump_record(dump,"residual",h,20480);dump_record(dump,"pre_mix",pre_mix,4);}
        float ffn_packet[5132],*ffn_input=ffn_packet,*route=ffn_packet+5120;
        float post[4],comb[16],next_pre[4];
        if(layer==1||layer==14)engram_step(layer,layer==1?0:1,h,hashes[layer==1?0:1]);
        if(mtp_enabled&&layer>=37&&rank==owner){float *tap=main_taps+(layer-37)*5120;
            for(int j=0;j<5120;++j){float sum=0;for(int c=0;c<4;++c)sum+=h[c*5120+j];tap[j]=sum*.25f;}
            ds41f_round_bf16(tap,5120);}
        double attention_start=now();float attn_pre[4],attn_post[4],attn_comb[16],x[5120],y[5120];
        if(rank==owner){
            if(layer==1||layer==14)dump_record(dump,"engram",h,20480);
            pt=P_BEGIN();mixes(layer,"attn",h,attn_pre,attn_post,attn_comb);P_END(HC_ATTN_MIX,pt);
            dump_record(dump,"attn_pre",attn_pre,4);dump_record(dump,"attn_post",attn_post,4);dump_record(dump,"attn_comb",attn_comb,16);
            pt=P_BEGIN();ds41f_hc_pre(x,h,pre_mix,5120);ds41f_round_bf16(x,5120);named_norm(layer,"attn_norm",x,x);P_END(HC_ATTN_PRE,pt);
            dump_record(dump,"attn_input",x,5120);
        }
        pt=P_BEGIN();
        if(dense_tp>1)parallel_attention(layer,pos,x,y);
        else if(rank==owner)CHECK(ds41f_attention_step(&attention,&weights,layer,pos,x,y));
        if(rank==owner){P_END(ATTENTION,pt);P_VALUE(TP_GROUP,dense_tp);
            dump_record(dump,"attn_output",y,5120);
            pt=P_BEGIN();ds41f_hc_post(h,y,h,attn_post,attn_comb,5120);ds41f_round_bf16(h,20480);P_END(HC_ATTN_POST,pt);
            dump_record(dump,"attn_residual",h,20480);
            pt=P_BEGIN();mixes(layer,"ffn",h,next_pre,post,comb);P_END(HC_FFN_MIX,pt);
            dump_record(dump,"ffn_pre",next_pre,4);dump_record(dump,"ffn_post",post,4);dump_record(dump,"ffn_comb",comb,16);
            pt=P_BEGIN();ds41f_hc_pre(ffn_input,h,attn_pre,5120);ds41f_round_bf16(ffn_input,5120);named_norm(layer,"ffn_norm",ffn_input,ffn_input);P_END(HC_FFN_PRE,pt);
            dump_record(dump,"ffn_input",ffn_input,5120);
            pt=P_BEGIN();float logits[384],prob[6];int ids[6];named_linear(layer,"ffn.gate",logits,ffn_input,1);
            dump_record(dump,"gate_logits",logits,384);
            char name[192];snprintf(name,sizeof name,"layers.%d.ffn.gate.bias",layer);
            CHECK(ds41f_gate(logits,tensor(name)->data,384,6,1,1.5f,ids,prob));
            for(int i=0;i<6;++i){route[i]=(float)ids[i];route[i+6]=prob[i];}P_END(GATE,pt);
            dump_record(dump,"route",route,12);
            profile_attention+=now()-attention_start;}
        pt=P_BEGIN();sync_attention(layer,pos);P_END(ATTN_SYNC,pt);
        pt=P_BEGIN();if(compact_comm)ds41f_comm_bf16_broadcast(ffn_packet,5120,12,owner);
        else ds41f_comm_broadcast(ffn_packet,5132,owner);P_END(FFN_BCAST,pt);
        float combined[5120];double phase_start=now();pt=P_BEGIN();local_experts(layer,ffn_input,route,combined);P_END(EXPERTS,pt);profile_expert+=now()-phase_start;
        float shared[5120];
        /* Use the same owner thread team before the rendezvous. The shared
         * output is still added after the routed sum, preserving its rounding. */
        if(shared_overlap){pt=P_BEGIN();
            if(dense_tp>1){if(tp_member(owner)){parallel_shared(layer,ffn_input,shared);P_END(SHARED_OVERLAP,pt);}}
            else if(rank==owner){shared_expert(layer,ffn_input,shared);P_END(SHARED_OVERLAP,pt);}}
        pt=P_BEGIN();ds41f_comm_sum(combined,5120);P_END(EXPERT_SUM,pt);
        if(rank==owner){phase_start=now();pt=P_BEGIN();
            if(!shared_overlap)shared_expert(layer,ffn_input,shared);
            for(int j=0;j<5120;++j)combined[j]+=shared[j];
            ds41f_round_bf16(combined,5120);P_END(SHARED_EXPERT,pt);
            dump_record(dump,"ffn_output",combined,5120);
            pt=P_BEGIN();ds41f_hc_post(h,combined,h,post,comb,5120);ds41f_round_bf16(h,20480);memcpy(pre_mix,next_pre,sizeof next_pre);P_END(HC_FFN_POST,pt);
            dump_record(dump,"output",h,20480);
            if(dump&&fclose(dump))ds41f_comm_abort("close intermediate dump",EIO);
            profile_shared+=now()-phase_start;
            if(trace){double norm=0;for(int j=0;j<20480;++j){if(!isfinite(h[j]))ds41f_comm_abort("nonfinite residual",EDOM);norm+=(double)h[j]*h[j];}
                fprintf(stderr,"LAYER pos=%zu layer=%d norm=%g experts=%d,%d,%d,%d,%d,%d\n",pos,layer,sqrt(norm),(int)route[0],(int)route[1],(int)route[2],(int)route[3],(int)route[4],(int)route[5]);}}
        pt=P_BEGIN();if(compact_comm)ds41f_comm_bf16_handoff(residual_packet,20480,4,owner,layer==39?11:(layer+1)%12);
        else ds41f_comm_broadcast(residual_packet,20484,owner);P_END(RESIDUAL_BCAST,pt);
    }
    ds41f_profile_at(pos,40);float next=0,x[5120];double head_start=now();
    if(rank==11){pt=P_BEGIN();ds41f_hc_pre(x,h,pre_mix,5120);ds41f_round_bf16(x,5120);CHECK(ds41f_norm(&weights,"norm.weight",x,x));P_END(HEAD_PRE,pt);}
    pt=P_BEGIN();
    if(dense_tp>1)ds41f_comm_bf16_broadcast(x,5120,0,11);
    if(rank==11||dense_tp>1){
        const ds41f_weight *head=tensor("head.weight");size_t count=head->rows;
        float *logits=malloc(count*sizeof(float));if(!logits)ds41f_comm_abort("logits",ENOMEM);
        CHECK(ds41f_linear(&weights,"head",logits,x,1));if(rank==11)P_END(HEAD_LINEAR,pt);
        pt=P_BEGIN();size_t local_best;CHECK(ds41f_argmax_finite(logits,count,&local_best));
        float value=logits[local_best];int best=(int)(local_best+head->row_start);
        if(dense_tp>1)ds41f_comm_argmax(&value,&best);
        if(rank==11)P_END(HEAD_SELECT,pt);
        float *full=NULL;
        if(logits_path&&dense_tp>1){if(rank==11){full=malloc(129280*sizeof(float));if(!full)ds41f_comm_abort("full logits",ENOMEM);}
            ds41f_comm_head_logits(full,logits,count);}
        if(rank==11){
            if(logits_path){char path[4096];snprintf(path,sizeof path,"%s.pos%zu.bin",logits_path,pos);FILE *f=fopen(path,"wb");
                if(!f||fwrite(full?full:logits,sizeof(float),129280,f)!=129280)ds41f_comm_abort("write logits",EIO);
                fclose(f);}
            next=(float)best;fprintf(stderr,"LOGITS pos=%zu argmax=%d value=%g\n",pos,best,value);profile_head+=now()-head_start;}
        free(full);free(logits);
    }
    pt=P_BEGIN();ds41f_comm_broadcast(&next,1,11);P_END(NEXT_BCAST,pt);P_END(TOKEN,token_start);
    if(mtp_enabled){ds41f_profile_at(SIZE_MAX,0);
        for(int i=0;i<3;++i)ds41f_comm_bf16_broadcast(main_taps+i*5120,5120,0,(37+i)%12);
    }
    return (int)next;
}
#include "ds41f_verify.h"
static uint64_t state_hash(size_t end)
{
    if(prefetch)CHECK(ds41f_prefetch_drain(prefetch));
    uint64_t hash=UINT64_C(14695981039346656037);
    #define HASH_BYTES(pointer,length) do{const unsigned char *p_=(const unsigned char *)(pointer);size_t n_=(length); \
        for(size_t j_=0;j_<n_;++j_){hash^=p_[j_];hash*=UINT64_C(1099511628211);}}while(0)
    HASH_BYTES(attention.window,(size_t)40*128*512*sizeof(float));
    for(int i=0;i<4;++i)HASH_BYTES(attention.compressed[i],(i==3?end:end/2)*356);
    HASH_BYTES(attention.pool_value,sizeof attention.pool_value);HASH_BYTES(attention.pool_score,sizeof attention.pool_score);
    HASH_BYTES(attention.selected,sizeof attention.selected);HASH_BYTES(&attention.selected_count,sizeof attention.selected_count);
    HASH_BYTES(attention.publication,sizeof attention.publication);HASH_BYTES(attention.candidate_blocks,(attention.capacity+7)/8);
    HASH_BYTES(engram.history,sizeof engram.history);HASH_BYTES(&engram.history_len,sizeof engram.history_len);
    for(int i=0;i<2;++i){HASH_BYTES(&engram.table[i].lookups,sizeof(uint64_t));HASH_BYTES(&engram.table[i].local_rows,sizeof(uint64_t));HASH_BYTES(&engram.table[i].remote_rows,sizeof(uint64_t));}
    #undef HASH_BYTES
    return hash;
}
static void commit_mtp_taps(const float *taps,size_t pos)
{
    CHECK(ds41f_mtp_commit(&mtp,taps,pos));
    if(mtp_dump&&rank==0){char path[4096];int n=snprintf(path,sizeof path,"%s.pos%zu.bin",mtp_dump,pos);
        if(n<0||(size_t)n>=sizeof path)ds41f_comm_abort("MTP tap path",ENAMETOOLONG);
        FILE *f=fopen(path,"wx");if(!f)ds41f_comm_abort("MTP tap open",errno);
        int bad=fwrite(taps,sizeof(float),15360,f)!=15360;if(fclose(f))bad=1;
        if(bad)ds41f_comm_abort("MTP tap dump",EIO);}
}
typedef struct {int *tokens;size_t count;int generate,trace,ignore_eos;size_t logits_start,logits_count;
    const char *logits_path;int produced;} inference_job;
static void inference_loop(void *context)
{
    inference_job *job=context;int next=0;
    for(size_t pos=0;pos<job->count+(size_t)job->generate-1;++pos){
        int input=pos<job->count?job->tokens[pos]:next;double t=now();
        next=forward(input,pos,job->trace,pos>=job->logits_start&&pos-job->logits_start<job->logits_count?job->logits_path:NULL);
        if(mtp_enabled){double commit_start=now();commit_mtp_taps(main_taps,pos);
            double commit_seconds=now()-commit_start;
            if(pos+1>=job->count&&pos+1-job->count<(size_t)mtp_probe&&mtp.committed>=2){
                int draft[6];float confidence[5];double draft_start=now();CHECK(ds41f_mtp_draft(&mtp,next,draft,confidence));
                if(rank==0)fprintf(stderr,"MTP_DRAFT pos=%zu seed=%d proposed=%d,%d,%d,%d,%d confidence=%g,%g,%g,%g,%g commit_seconds=%.6f draft_seconds=%.6f\n",
                    pos,next,draft[1],draft[2],draft[3],draft[4],draft[5],confidence[0],confidence[1],confidence[2],confidence[3],confidence[4],commit_seconds,now()-draft_start);
            }}
        if(pos+1>=job->count)++job->produced;
        if(rank==0){fprintf(stderr,"TOKEN pos=%zu input=%d next=%d seconds=%.6f\n",pos,input,next,now()-t);
            if(pos+1>=job->count){printf("%d\n",next);fflush(stdout);}}
        if(!job->ignore_eos&&next==1&&pos+1>=job->count)break;
    }
}
static const char *logits_at(const inference_job *job,size_t pos)
{return pos>=job->logits_start&&pos-job->logits_start<job->logits_count?job->logits_path:NULL;}
static void report_token(inference_job *job,size_t pos,int input,int next,double seconds)
{
    if(pos+1>=job->count)++job->produced;
    if(rank==0){fprintf(stderr,"TOKEN pos=%zu input=%d next=%d seconds=%.6f\n",pos,input,next,seconds);
        if(pos+1>=job->count){printf("%d\n",next);fflush(stdout);}}
}
static void speculative_loop(void *context)
{
    inference_job *job=context;int seed=0;
    for(size_t pos=0;pos<job->count;++pos){double start=now();
        seed=forward(job->tokens[pos],pos,job->trace,logits_at(job,pos));commit_mtp_taps(main_taps,pos);
        report_token(job,pos,job->tokens[pos],seed,now()-start);
    }
    if(!job->ignore_eos&&seed==1)return;
    size_t pos=job->count,cycles=0,accepted_total=0,verified_total=0,emitted_total=0;
    float *taps=malloc((size_t)6*15360*sizeof(float));if(!taps)ds41f_comm_abort("verifier tap allocation",ENOMEM);
    while(job->produced<job->generate){double cycle_start=now();
        size_t remaining=(size_t)(job->generate-job->produced),draft_count=(size_t)speculate;
        if(draft_count>=remaining)draft_count=remaining-1;
        /* A final partial block can use sequential verification at the model
         * position limit, where five temporary draft positions do not fit. */
        if(mtp.committed<2||mtp.committed>1048576-DS41F_DRAFT_BLOCK)draft_count=0;
        int draft[6]={seed,0,0,0,0,0},prediction[6],input[6];float confidence[5];
        if(draft_count)CHECK(ds41f_mtp_draft(&mtp,seed,draft,confidence));
        double draft_seconds=now()-cycle_start,verify_start=now();size_t n=draft_count+1;
        ds41f_journal *journal=NULL;uint64_t expected_hash=0;int expected_prediction[6];
        if(!verify_batch||verify_check||spec_force_reject>=0){
            CHECK(ds41f_journal_create(&journal,&attention,&engram,prefetch,
                pos,n,ds41f_journal_bytes(attention.capacity,n)));
            for(size_t i=0;i<n;++i){input[i]=draft[i];CHECK(ds41f_journal_record(journal,pos+i));
                prediction[i]=forward(input[i],pos+i,job->trace,logits_at(job,pos+i));
                memcpy(taps+i*15360,main_taps,15360*sizeof(float));
                if(spec_force_reject>=0&&i<draft_count){
                    if(i<(size_t)spec_force_reject)draft[i+1]=prediction[i];
                    else if(i==(size_t)spec_force_reject)draft[i+1]=(prediction[i]+1)%129280;}
            }
            if(verify_batch){expected_hash=state_hash(pos+n);memcpy(expected_prediction,prediction,n*sizeof(int));
                CHECK(ds41f_journal_finish(journal,0));ds41f_journal_free(journal);journal=NULL;}
        }else for(size_t i=0;i<n;++i)input[i]=draft[i];
        if(verify_batch){
            float *batch_taps=(verify_check||spec_force_reject>=0)?malloc(n*15360*sizeof(float)):taps;
            if(!batch_taps)ds41f_comm_abort("batch tap check",ENOMEM);
            forward_batch(input,n,pos,prediction,batch_taps,job->logits_path,job->logits_start,job->logits_count);
            if(verify_check||spec_force_reject>=0){
                /* Compare the complete committed state by temporarily selecting
                 * the final causal view, before rolling back any rejected tail. */
                ds41f_attention saved=attention;attention=verifier->state[n-1];uint64_t got_hash=state_hash(pos+n);attention=saved;
                if(got_hash!=expected_hash||memcmp(expected_prediction,prediction,n*sizeof(int))||memcmp(taps,batch_taps,n*15360*sizeof(float)))
                    ds41f_comm_abort("causal verifier differs from sequential state/taps/tokens",EDOM);
                memcpy(taps,batch_taps,n*15360*sizeof(float));free(batch_taps);
                if(rank==0)fprintf(stderr,"VERIFY_CHECK PASS pos=%zu inputs=%zu state taps tokens\n",pos,n);
            }
        }
        double verify_seconds=now()-verify_start,commit_start=now();size_t accepted=0;
        while(accepted<draft_count&&prediction[accepted]==draft[accepted+1])++accepted;
        size_t keep=accepted+1;int stopped=0;
        if(!job->ignore_eos)for(size_t i=0;i<keep;++i)if(prediction[i]==1){keep=i+1;stopped=1;break;}
        if(verify_batch)verify_finish(keep);
        else{CHECK(ds41f_journal_finish(journal,keep));ds41f_journal_free(journal);}
        for(size_t i=0;i<keep;++i)commit_mtp_taps(taps+i*15360,pos+i);
        double commit_seconds=now()-commit_start,total=now()-cycle_start;
        for(size_t i=0;i<keep;++i)report_token(job,pos+i,input[i],prediction[i],total/keep);
        if(rank==0)fprintf(stderr,"SPEC_CYCLE pos=%zu drafted=%zu verified=%zu accepted=%zu emitted=%zu draft_seconds=%.6f verify_seconds=%.6f commit_seconds=%.6f seconds=%.6f forced=%d\n",
            pos,draft_count,n,accepted,keep,draft_seconds,verify_seconds,commit_seconds,total,spec_force_reject);
        ++cycles;accepted_total+=accepted;verified_total+=n;emitted_total+=keep;
        seed=prediction[keep-1];pos+=keep;if(stopped)break;
    }
    if(rank==0)fprintf(stderr,"SPEC_FINISHED cycles=%zu accepted=%zu verified=%zu emitted=%zu verifier=%s\n",cycles,accepted_total,verified_total,emitted_total,verify_batch?"batched":"sequential");
    free(taps);
}
static void batched_replay_loop(void *context)
{
    inference_job *job=context;float *taps=malloc((size_t)6*15360*sizeof(float));if(!taps)ds41f_comm_abort("replay taps",ENOMEM);
    for(size_t pos=0;pos<job->count;){size_t n=job->count-pos;if(n>(size_t)verify_replay_batch)n=(size_t)verify_replay_batch;
        int prediction[6];double start=now();forward_batch(job->tokens+pos,n,pos,prediction,taps,job->logits_path,job->logits_start,job->logits_count);
        verify_finish(n);if(mtp_enabled)for(size_t i=0;i<n;++i)commit_mtp_taps(taps+i*15360,pos+i);
        double seconds=now()-start;for(size_t i=0;i<n;++i)report_token(job,pos+i,job->tokens[pos+i],prediction[i],seconds/n);
        pos+=n;
    }
    free(taps);
}
int main(int argc,char **argv)
{
    CHECK(ds41f_comm_init(&argc,&argv,&rank,&ranks));
    char logfile[80];snprintf(logfile,sizeof logfile,"inference.rank%02d.log",rank);
    if(!freopen(logfile,"w",stderr))ds41f_comm_abort("rank log",errno);
    setvbuf(stderr,NULL,_IOLBF,0);
    int cpu[48];for(int i=0;i<48;++i)cpu[i]=-1;
    #pragma omp parallel
    {int tid=omp_get_thread_num();if(tid<48)cpu[tid]=sched_getcpu();}
    int unique=0;for(int i=0;i<48;++i){int seen=0;for(int j=0;j<i;++j)if(cpu[j]==cpu[i])seen=1;if(cpu[i]>=0&&!seen)++unique;}
    fprintf(stderr,"THREADS max=%d distinct_cpus=%d\n",omp_get_max_threads(),unique);
    const char *root=NULL,*prompt=NULL,*logits_path=NULL,*mtp_root=NULL,*mtp_logits=NULL;int mtp_quant=1,mtp_expert_sdot=1,mtp_hc=1;size_t capacity=4096,logits_count=SIZE_MAX,logits_start=0,profile_start=SIZE_MAX,profile_count=0,int8_block=0;int generate=1,trace=0,ignore_eos=0,prefetch_engram=0,engram_row_cache_mib=0,int8_projections=0,engram_scale_cache=0,weights_local_pages=0,sparse_tile=0,sparse_math=0,attention_local_pages=0,index_head_tiles=0,linear_input_cache=0,expert_sdot=0,sparse_sdot=0,persistent_team=0;
    for(int i=1;i<argc;++i){
        if(!strcmp(argv[i],"--stage-root")&&i+1<argc)root=argv[++i];
        else if(!strcmp(argv[i],"--prompt-ids")&&i+1<argc)prompt=argv[++i];
        else if(!strcmp(argv[i],"--max-context")&&i+1<argc)capacity=strtoul(argv[++i],NULL,10);
        else if(!strcmp(argv[i],"--generate")&&i+1<argc)generate=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--logits-prefix")&&i+1<argc)logits_path=argv[++i];
        else if(!strcmp(argv[i],"--logits-start")&&i+1<argc)logits_start=strtoul(argv[++i],NULL,10);
        else if(!strcmp(argv[i],"--logits-count")&&i+1<argc)logits_count=strtoul(argv[++i],NULL,10);
        else if(!strcmp(argv[i],"--dump-prefix")&&i+1<argc)dump_prefix=argv[++i];
        else if(!strcmp(argv[i],"--dump-count")&&i+1<argc)dump_count=strtoul(argv[++i],NULL,10);
        else if(!strcmp(argv[i],"--profile-start")&&i+1<argc)profile_start=strtoul(argv[++i],NULL,10);
        else if(!strcmp(argv[i],"--profile-count")&&i+1<argc)profile_count=strtoul(argv[++i],NULL,10);
        else if(!strcmp(argv[i],"--fp8-int8-block")&&i+1<argc)int8_block=strtoul(argv[++i],NULL,10);
        else if(!strcmp(argv[i],"--fp8-int8-scope")&&i+1<argc){const char *scope=argv[++i];
            if(!strcmp(scope,"all"))int8_projections=0;
            else if(!strcmp(scope,"projections"))int8_projections=1;
            else ds41f_comm_abort("FP8 INT8 scope must be all or projections",EINVAL);}
        else if(!strcmp(argv[i],"--sparse-tile")&&i+1<argc)sparse_tile=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--hc-matvec")&&i+1<argc)hc_matvec_mode=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--dense-tp")&&i+1<argc)dense_tp=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--sparse-math")&&i+1<argc)sparse_math=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--expert-fused")&&i+1<argc)expert_fused=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--attention-local-pages"))attention_local_pages=1;
        else if(!strcmp(argv[i],"--index-head-tiles"))index_head_tiles=1;
        else if(!strcmp(argv[i],"--linear-input-cache"))linear_input_cache=1;
        else if(!strcmp(argv[i],"--quant-parallel"))ds41f_set_quant_parallel(1);
        else if(!strcmp(argv[i],"--expert-sdot"))expert_sdot=1;
        else if(!strcmp(argv[i],"--sparse-sdot"))sparse_sdot=1;
        else if(!strcmp(argv[i],"--persistent-team"))persistent_team=1;
        else if(!strcmp(argv[i],"--cache-sve"))ds41f_set_cache_sve(1);
        else if(!strcmp(argv[i],"--rope-cache"))ds41f_set_rope_cache(1);
        else if(!strcmp(argv[i],"--mtp-stage-root")&&i+1<argc)mtp_root=argv[++i];
        else if(!strcmp(argv[i],"--engram-row-cache-mib")&&i+1<argc)engram_row_cache_mib=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--expert-input-cache"))expert_input_cache=1;
        else if(!strcmp(argv[i],"--verify-comm-batch"))verify_comm_batch=1;
        else if(!strcmp(argv[i],"--verify-timing"))verify_timing=1;
        else if(!strcmp(argv[i],"--verify-expert-batch"))verify_expert_batch=1;
        else if(!strcmp(argv[i],"--state-hash"))dump_state_hash=1;
        else if(!strcmp(argv[i],"--verify-batch"))verify_batch=1;
        else if(!strcmp(argv[i],"--verify-check"))verify_check=1;
        else if(!strcmp(argv[i],"--verify-replay-batch")&&i+1<argc)verify_replay_batch=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--speculate")&&i+1<argc)speculate=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--spec-force-reject")&&i+1<argc)spec_force_reject=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--mtp-probe")&&i+1<argc)mtp_probe=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--mtp-quant")&&i+1<argc)mtp_quant=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--mtp-expert-sdot")&&i+1<argc)mtp_expert_sdot=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--mtp-hc-matvec")&&i+1<argc)mtp_hc=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--mtp-logits-prefix")&&i+1<argc)mtp_logits=argv[++i];
        else if(!strcmp(argv[i],"--mtp-dump-prefix")&&i+1<argc)mtp_dump=argv[++i];
        else if(!strcmp(argv[i],"--trace"))trace=1;
        else if(!strcmp(argv[i],"--ignore-eos"))ignore_eos=1;
        else if(!strcmp(argv[i],"--engram-prefetch"))prefetch_engram=1;
        else if(!strcmp(argv[i],"--engram-scale-cache"))engram_scale_cache=1;
        else if(!strcmp(argv[i],"--hc-mix-sve"))hc_mix_sve=1;
        else if(!strcmp(argv[i],"--shared-overlap"))shared_overlap=1;
        else if(!strcmp(argv[i],"--weights-local-pages"))weights_local_pages=1;
        else if(!strcmp(argv[i],"--compact-comm"))compact_comm=1;
        else if(!strcmp(argv[i],"--mpi-broadcast"))ds41f_comm_use_mpi_broadcast(1);
        else ds41f_comm_abort("unknown/missing option",EINVAL);}
    if(!root||!prompt||!capacity||capacity>1048576||generate<1)ds41f_comm_abort("required --stage-root --prompt-ids; valid context/generate",EINVAL);
    if(!dump_count||dump_count>64)ds41f_comm_abort("dump count must be in [1,64]",EINVAL);
    if(int8_block&&int8_block!=32&&int8_block!=64&&int8_block!=128&&int8_block!=256)
        ds41f_comm_abort("FP8 INT8 block must be 0,32,64,128,256",EINVAL);
    if(sparse_tile!=0&&sparse_tile!=1&&sparse_tile!=2&&sparse_tile!=4&&sparse_tile!=6)
        ds41f_comm_abort("sparse tile must be 0,1,2,4,6",EINVAL);
    if(hc_matvec_mode<0||hc_matvec_mode>2)ds41f_comm_abort("HC matvec must be 0,1,2",EINVAL);
    if((dense_tp!=1&&dense_tp!=2&&dense_tp!=4)||(dense_tp>1&&!shared_overlap))
        ds41f_comm_abort("dense TP must be 1,2,4; TP2/TP4 require --shared-overlap",EINVAL);
    mtp_enabled=mtp_root!=NULL;
    if((mtp_enabled&&dense_tp!=4)||mtp_probe<0||mtp_probe>64||mtp_quant<0||mtp_quant>1||
       mtp_expert_sdot<0||mtp_expert_sdot>1||mtp_hc<0||mtp_hc>2||((mtp_probe||mtp_dump||mtp_logits)&&!mtp_enabled))
        ds41f_comm_abort("MTP requires TP4, probe 0..64, quant/expert 0..1, HC 0..2",EINVAL);
    if((speculate&&speculate!=2&&speculate!=4&&speculate!=5)||
       (speculate&&(!mtp_enabled||mtp_probe||profile_count))||spec_force_reject<-1||
       (spec_force_reject>=0&&(!speculate||spec_force_reject>=speculate)))
        ds41f_comm_abort("speculate 2/4/5 requires MTP, no probe/ordinary profile; forced index must be within draft",EINVAL);
    if((verify_batch&&!speculate)||verify_replay_batch<0||verify_replay_batch>6||
       (verify_check&&!verify_batch)||(verify_replay_batch&&(speculate||generate!=1))||
       ((verify_batch||verify_replay_batch)&&(dense_tp!=4||!compact_comm||profile_count)))
        ds41f_comm_abort("batched verifier requires TP4 compact transport, no ordinary profile; replay is fixed-input generate=1",EINVAL);
    if(engram_row_cache_mib<0||engram_row_cache_mib>64)ds41f_comm_abort("Engram row cache budget must be 0..64 MiB",EINVAL);
    ds41f_comm_set_tp(dense_tp);
    if(sparse_math<0||sparse_math>3||(sparse_math&&!sparse_tile))ds41f_comm_abort("sparse math 0..3 requires a tile when nonzero",EINVAL);
    if((verify_expert_batch||verify_timing||verify_comm_batch)&&!verify_batch&&!verify_replay_batch)ds41f_comm_abort("expert batch requires batched verifier",EINVAL);
    if(expert_fused<0||expert_fused>2)ds41f_comm_abort("expert fused must be 0,1,2",EINVAL);
    if(persistent_team&&(!sparse_tile||sparse_sdot))ds41f_comm_abort("persistent team requires tiled FP32 attention",EINVAL);
    if(expert_sdot&&!weights_local_pages)ds41f_comm_abort("expert SDOT requires fresh weight pages",EINVAL);
    int *tokens=malloc(capacity*sizeof(int));if(!tokens)ds41f_comm_abort("prompt allocation",ENOMEM);
    FILE *f=fopen(prompt,"r");if(!f)ds41f_comm_abort("prompt open",errno);
    size_t count=0;int token;
    while(fscanf(f,"%d",&token)==1){if(count>=capacity||token<0||token>=129280)ds41f_comm_abort("prompt bounds",EINVAL);tokens[count++]=token;}
    if(!feof(f)||!count||count+(size_t)generate-1>capacity)ds41f_comm_abort("prompt/context",EINVAL);
    fclose(f);
    if(mtp_dump&&count+(size_t)generate-1>2048)ds41f_comm_abort("MTP tap dump limited to 2048 positions",EINVAL);
    if(profile_start==SIZE_MAX)profile_start=count;
    if(profile_count>4096||profile_start>count+(size_t)generate-1)ds41f_comm_abort("profile bounds",EINVAL);
    CHECK(ds41f_profile_init(profile_start,profile_count));
    char stage[4096];snprintf(stage,sizeof stage,"%s/rank%d",root,rank);
    size_t memory=available(),reserve=(size_t)4*1024*1024*1024,mtp_budget=0;
    char mtp_stage[4096];
    if(mtp_enabled){int n=snprintf(mtp_stage,sizeof mtp_stage,"%s/rank%d",mtp_root,rank);
        if(n<0||(size_t)n>=sizeof mtp_stage)ds41f_comm_abort("MTP stage path",ENAMETOOLONG);
        CHECK(ds41f_mtp_admission(mtp_stage,mtp_quant,&mtp_budget));
        /* Reserve the configured KV, Engram scales, scratch and 2 GiB floor.
         * A 1M cache uses about 0.9 GiB; Engram scales fit below 0.5 GiB/rank. */
        size_t cache=capacity*890+(size_t)32*1024*1024;
        size_t planned=mtp_budget+cache+(engram_scale_cache?(size_t)520*1024*1024:0)+(size_t)2*1024*1024*1024;
        if(planned>reserve)reserve=planned;
        fprintf(stderr,"MTP_ADMISSION rank=%d budget=%zu reserve=%zu available=%zu\n",rank,mtp_budget,reserve,memory);}

    reserve+=(size_t)engram_row_cache_mib*1024*1024;
    if(verify_batch||verify_replay_batch)reserve+=(size_t)80*1024*1024;
    if(memory<=reserve)ds41f_comm_abort("insufficient MemAvailable",ENOMEM);
    double start=now();CHECK(ds41f_weights_load_local(&weights,stage,NULL,memory-reserve,weights_local_pages));
    CHECK(ds41f_weights_check_tp(&weights,stage,dense_tp,rank));
    if(int8_block){double quant_start=now();CHECK(ds41f_weights_requantize_fp8(&weights,int8_block,weight_prepare_limit(),int8_projections));
        fprintf(stderr,"FP8_INT8_READY rank=%d seconds=%.6f available=%zu\n",rank,now()-quant_start,available());}
    if(expert_sdot){double pack_start=now();CHECK(ds41f_weights_pack_experts(&weights,weight_prepare_limit()));
        fprintf(stderr,"EXPERT_PACKED_READY rank=%d seconds=%.6f available=%zu\n",rank,now()-pack_start,available());}
    if(linear_input_cache)CHECK(ds41f_weights_enable_input_cache(&weights));
    CHECK(ds41f_attention_init(&attention,capacity));attention.sparse_sdot=sparse_sdot;attention.sparse_tile=sparse_tile;attention.sparse_math=sparse_math;attention.index_head_tiles=index_head_tiles;
    if(attention_local_pages)CHECK(ds41f_attention_place_workspace(&attention));
    CHECK(ds41f_engram_open(&engram,stage,rank,12));
    if(engram_scale_cache){size_t headroom=available();double cache_start=now();
        if(headroom<=(size_t)2*1024*1024*1024)ds41f_comm_abort("Engram scale cache memory guard",ENOMEM);
        CHECK(ds41f_engram_cache_scales(&engram,headroom-(size_t)2*1024*1024*1024));
        fprintf(stderr,"ENGRAM_SCALE_CACHE_READY rank=%d bytes=%llu seconds=%.6f available=%zu\n",rank,
            (unsigned long long)((engram.table[0].owned_rows+engram.table[1].owned_rows)*8),now()-cache_start,available());}
    if(mtp_enabled){size_t headroom=available();
        if(headroom<=(size_t)2*1024*1024*1024||mtp_budget>headroom-(size_t)2*1024*1024*1024)
            ds41f_comm_abort("MTP memory guard",ENOMEM);
        CHECK(ds41f_mtp_load(&mtp,&weights,mtp_stage,rank,mtp_budget,mtp_quant,mtp_expert_sdot,mtp_hc));
        mtp.logits_prefix=mtp_logits;
        fprintf(stderr,"MTP_READY rank=%d bytes=%zu available=%zu\n",rank,mtp.weights.bytes,available());}
    if(verify_batch||verify_replay_batch){verifier_allocate();
        fprintf(stderr,"VERIFY_READY rank=%d bounded_window_bytes=%zu available=%zu\n",rank,(size_t)6*40*128*512*sizeof(float),available());}
    if(engram_row_cache_mib){size_t bytes=0;CHECK(ds41f_engram_cache_rows(&engram,(size_t)engram_row_cache_mib*1024*1024,&bytes));
        fprintf(stderr,"ENGRAM_ROW_CACHE_READY rank=%d bytes=%zu available=%zu\n",rank,bytes,available());}
    if(prefetch_engram)CHECK(ds41f_prefetch_create(&prefetch,&engram));
    fprintf(stderr,"RESIDENT_READY rank=%d bytes=%zu seconds=%.3f available=%zu\n",rank,weights.bytes,now()-start,available());
    if(available()<(size_t)2*1024*1024*1024)ds41f_comm_abort("post-load memory guard",ENOMEM);
    ds41f_comm_ready();start=now();
    inference_job job={tokens,count,generate,trace,ignore_eos,logits_start,logits_count,logits_path,0};
    ds41f_team_body loop=verify_replay_batch?batched_replay_loop:speculate?speculative_loop:inference_loop;
    if(persistent_team)CHECK(ds41f_team_run(loop,&job));else loop(&job);
    int produced=job.produced;
    if(dump_state_hash){uint64_t mtp_hash=UINT64_C(14695981039346656037);
        if(mtp_enabled){const unsigned char *bytes=(const unsigned char *)mtp.window;
            for(size_t i=0;i<(size_t)3*128*512*sizeof(float);++i){mtp_hash^=bytes[i];mtp_hash*=UINT64_C(1099511628211);}}
        fprintf(stderr,"STATE_HASH rank=%d inputs=%zu backbone=%016llx mtp_inputs=%zu mtp=%016llx\n",rank,count+(size_t)produced-1,
            (unsigned long long)state_hash(count+(size_t)produced-1),mtp.committed,(unsigned long long)mtp_hash);}
    if(engram_row_cache_mib)for(int slot=0;slot<2;++slot)fprintf(stderr,"ENGRAM_ROW_CACHE rank=%d slot=%d hits=%llu misses=%llu\n",rank,slot,
        (unsigned long long)engram.table[slot].row_cache_hits,(unsigned long long)engram.table[slot].row_cache_misses);
    fprintf(stderr,"INFERENCE_FINISHED rank=%d prompt=%zu generated=%d seconds=%.3f available=%zu\n",rank,count,produced,now()-start,available());
    fprintf(stderr,"PROFILE rank=%d attention_hc_gate=%.6f local_experts=%.6f shared_hc=%.6f head=%.6f\n",rank,profile_attention,profile_expert,profile_shared,profile_head);
    verifier_write_timing();CHECK(ds41f_profile_write(rank));ds41f_profile_free();
    verifier_free();ds41f_mtp_free(&mtp);ds41f_prefetch_destroy(prefetch);ds41f_engram_close(&engram);ds41f_attention_free(&attention);ds41f_weights_free(&weights);free(tokens);ds41f_comm_free();return 0;
}
