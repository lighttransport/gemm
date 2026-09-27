#ifndef DS41F_VERIFY_H
#define DS41F_VERIFY_H
/* Included after runner helpers. One main thread owns MPI and mutable state.
 * Token-major window snapshots keep future writes invisible to earlier queries;
 * compressed appends are shared, but every selector is bounded by its position.
 * A full 1M compressed cache is never cloned. */
enum { VERIFY_BATCH_MAX=64, VERIFY_KERNEL_BATCH=6, VERIFY_TIMING_RECORDS=256 };
typedef struct {size_t position,inputs;double summary[15],layer[40][7];} verifier_timing_record;
typedef struct {
    ds41f_attention state[VERIFY_BATCH_MAX],base;
    uint8_t *candidates;float overwritten[VERIFY_BATCH_MAX][512];
    ds41f_attention_context contexts[VERIFY_BATCH_MAX];
    ds41f_journal *journal;
    size_t count,start;
    uint32_t history[VERIFY_BATCH_MAX][4],history_len[VERIFY_BATCH_MAX];uint64_t counters[VERIFY_BATCH_MAX][2][3];
    float engram_rows[VERIFY_BATCH_MAX][2][24*256];
    float residual[VERIFY_BATCH_MAX][20484],input[VERIFY_BATCH_MAX][5120],output[VERIFY_BATCH_MAX][5120],qr[VERIFY_BATCH_MAX][1280];
    float q[VERIFY_BATCH_MAX][8192],attended[VERIFY_BATCH_MAX][8192],projected[VERIFY_BATCH_MAX][8192],local[VERIFY_BATCH_MAX][2048],part[VERIFY_BATCH_MAX][1280];
    float attn_pre[VERIFY_BATCH_MAX][4],attn_post[VERIFY_BATCH_MAX][4],attn_comb[VERIFY_BATCH_MAX][16],ffn_pre[VERIFY_BATCH_MAX][4],ffn_post[VERIFY_BATCH_MAX][4],ffn_comb[VERIFY_BATCH_MAX][16];
    float ffn[VERIFY_BATCH_MAX][5132],combined[VERIFY_BATCH_MAX][5120],shared[VERIFY_BATCH_MAX][5120],gate[VERIFY_BATCH_MAX][2304],up[VERIFY_BATCH_MAX][2304],hidden[VERIFY_BATCH_MAX][2304];
    float expert_values[VERIFY_BATCH_MAX][6][5120],expert_input[VERIFY_BATCH_MAX][5120],expert_gather[VERIFY_KERNEL_BATCH][5120],expert_output[VERIFY_KERNEL_BATCH][5120];
    float expert_scratch[VERIFY_KERNEL_BATCH*3*2304];
    /* Bounded recent diagnostic records; never write shared-storage logs
     * between verification and commit. */
    verifier_timing_record timing[VERIFY_TIMING_RECORDS];size_t timing_count;
} verifier_work;
static verifier_work *verifier;
static void verifier_allocate(void)
{
    verifier=calloc(1,sizeof *verifier);if(!verifier)ds41f_comm_abort("verifier work",ENOMEM);
    verifier->candidates=malloc((size_t)VERIFY_BATCH_MAX*((attention.capacity+7)/8));
    if(!verifier->candidates)ds41f_comm_abort("verifier causal windows",ENOMEM);
    /* Charge physical pages before the post-load MemAvailable guard. */
    memset(verifier->candidates,0,(size_t)VERIFY_BATCH_MAX*((attention.capacity+7)/8));
}
static void verifier_free(void)
{if(verifier){free(verifier->candidates);free(verifier);verifier=NULL;}}
static void verify_linear(int layer,const char *part,float *out,size_t out_stride,
                          const float *x,size_t x_stride,size_t count,int raw)
{char name[192];snprintf(name,sizeof name,"layers.%d.%s",layer,part);
    for(size_t first=0;first<count;first+=VERIFY_KERNEL_BATCH){size_t n=count-first;if(n>VERIFY_KERNEL_BATCH)n=VERIFY_KERNEL_BATCH;
        CHECK(ds41f_linear_batch(&weights,name,out+first*out_stride,out_stride,x+first*x_stride,x_stride,n,raw));}}
static void verify_begin(const int *tokens,size_t count,size_t pos)
{
    verifier_work *v=verifier;if(!v||!count||count>VERIFY_BATCH_MAX||v->journal)ds41f_comm_abort("verifier lifetime",EINVAL);
    v->base=attention;v->count=count;v->start=pos;size_t candidate_bytes=(attention.capacity+7)/8;
    CHECK(ds41f_journal_create(&v->journal,&attention,&engram,prefetch,pos,count,
        ds41f_journal_bytes(attention.capacity,count)));
    for(size_t i=0;i<count;++i){
        CHECK(ds41f_journal_record(v->journal,pos+i));v->state[i]=attention;
        v->state[i].candidate_blocks=v->candidates+i*candidate_bytes;
        memcpy(v->state[i].candidate_blocks,attention.candidate_blocks,candidate_bytes);
        memset(v->residual[i]+20480,0,4*sizeof(float));v->residual[i][20480]=1;
        if(rank==0){const ds41f_weight *embedding=tensor("embed.weight");
            const uint16_t *row=(uint16_t *)embedding->data+(size_t)tokens[i]*5120;
            for(int c=0;c<4;++c)for(int j=0;j<5120;++j)v->residual[i][c*5120+j]=ds41f_bf16_to_f32(row[j]);}
    }
    /* Resolve the bounded Engram batch before mutating layer state. Both slot
     * results are consumed before the next generation can replace them. */
    for(size_t i=0;i<count;++i){uint64_t ids[2][24];CHECK(ds41f_engram_hash_ids(&engram,(uint32_t)tokens[i],ids));
        if(prefetch)CHECK(ds41f_prefetch_submit(prefetch,(const uint64_t (*)[24])ids));
        for(int slot=0;slot<2;++slot){float *rows=v->engram_rows[i][slot];memset(rows,0,24*256*sizeof(float));
            if(prefetch)CHECK(ds41f_prefetch_wait(prefetch,slot,rows,NULL));
            else for(int k=0;k<24;++k){ds41f_engram_table *table=&engram.table[slot];
                if(ids[slot][k]>=table->first&&ids[slot][k]<table->first+table->owned_rows){uint16_t row[256];
                    CHECK(ds41f_engram_read_local(&engram,slot,ids[slot][k],row));
                    for(int j=0;j<256;++j)rows[k*256+j]=ds41f_bf16_to_f32(row[j]);}}
        }
        memcpy(v->history[i],engram.history,sizeof engram.history);v->history_len[i]=engram.history_len;
        for(int slot=0;slot<2;++slot){v->counters[i][slot][0]=engram.table[slot].lookups;
            v->counters[i][slot][1]=engram.table[slot].local_rows;v->counters[i][slot][2]=engram.table[slot].remote_rows;}
    }
}
static void __attribute__((unused)) verify_begin_stage(const float *incoming,size_t count,size_t pos)
{
    verifier_work *v=verifier;
    if(!v||!incoming||!count||count>VERIFY_BATCH_MAX||v->journal||pos+count>attention.capacity)
        ds41f_comm_abort("pipeline verifier lifetime",EINVAL);
    v->base=attention;v->count=count;v->start=pos;
    size_t candidate_bytes=(attention.capacity+7)/8;
    CHECK(ds41f_journal_create(&v->journal,&attention,&engram,prefetch,pos,count,
                               ds41f_journal_bytes(attention.capacity,count)));
    for(size_t i=0;i<count;++i){
        CHECK(ds41f_journal_record(v->journal,pos+i));v->state[i]=attention;
        v->state[i].candidate_blocks=v->candidates+i*candidate_bytes;
        memcpy(v->state[i].candidate_blocks,attention.candidate_blocks,candidate_bytes);
        memcpy(v->residual[i],incoming+i*20484,20484*sizeof(float));
        /* Engram rows are populated by the stage-local feeder before layer
         * 14. Clear them here so a missing feeder cannot consume stale heap
         * data; the staged path remains gated until history is transported. */
        memset(v->engram_rows[i],0,sizeof v->engram_rows[i]);
        memset(v->counters[i],0,sizeof v->counters[i]);
    }
}
static void verify_engram(int layer,float *h,float *rows)
{
    ds41f_comm_sum(rows,24*256);
    if(rank==pipeline_owner_for_layer(layer)){float kv[5*5120];named_linear(layer,"engram.wkv",kv,rows,0);
        char name[192];snprintf(name,sizeof name,"layers.%d.engram.q_weight",layer);const uint16_t *qw=tensor(name)->data;
        snprintf(name,sizeof name,"layers.%d.engram.k_weight",layer);const uint16_t *kw=tensor(name)->data;
        float q[4*5120],k[4*5120];for(int j=0;j<4*5120;++j){q[j]=ds41f_bf16_to_f32(qw[j]);k[j]=ds41f_bf16_to_f32(kw[j]);}
        ds41f_engram_fuse(h,kv,kv+4*5120,q,k,5120,1e-20f);ds41f_round_bf16(h,20480);}
}
static void verify_attention(int layer,double detail[6])
{
    verifier_work *v=verifier;size_t n=v->count;int owner=pipeline_owner_for_layer(layer);double mark=verify_timing?now():0;
    for(size_t i=0;i<n;++i)memcpy(v->overwritten[i],v->base.window+
        ((size_t)layer*128+(v->start+i)%128)*512,512*sizeof(float));
    if(tp_member(owner)){
        for(size_t i=0;i<n;++i)if(rank==owner){attention=v->state[i];
            if(i&&(layer==2||layer==8||layer==14)){int src=layer==2?0:layer==8?1:2;
                memcpy(attention.pool_value[src],v->state[i-1].pool_value[src],512*sizeof(float));
                memcpy(attention.pool_score[src],v->state[i-1].pool_score[src],512*sizeof(float));}
            CHECK(ds41f_attention_prepare(&attention,&weights,layer,v->start+i,v->input[i],v->contexts+i));
            v->state[i]=attention;}
        ds41f_comm_tp_bytes(v->contexts,n*sizeof *v->contexts,owner);
        if(rank!=owner)for(size_t i=0;i<n;++i){attention=v->state[i];
            if(i&&(layer==2||layer==8||layer==14)){int src=layer==2?0:layer==8?1:2;
                memcpy(attention.pool_value[src],v->state[i-1].pool_value[src],512*sizeof(float));
                memcpy(attention.pool_score[src],v->state[i-1].pool_score[src],512*sizeof(float));}
            CHECK(ds41f_attention_apply(&attention,layer,v->start+i,v->contexts+i));v->state[i]=attention;}
        for(size_t i=0;i<n;++i)memcpy(v->qr[i],v->contexts[i].qr,sizeof v->contexts[i].qr);
    }
    for(size_t i=0;i<n;++i){attention=v->state[i];sync_attention(layer,v->start+i);v->state[i]=attention;
    }
    if(verify_timing){detail[0]+=now()-mark;mark=now();}if(!tp_member(owner))return;
    verify_linear(layer,"attn.wq_b",(float *)v->q,8192,(float *)v->qr,1280,n,0);
    if(verify_timing){detail[1]+=now()-mark;mark=now();}
    for(size_t i=0;i<n;++i)CHECK(ds41f_attention_attend_causal_tile(v->state+i,&weights,layer,v->start+i,
        v->q[i],(rank%4)*16,16,v->attended[i],v->start,n,(float *)v->overwritten));
    if(verify_timing){detail[2]+=now()-mark;mark=now();}
    char name[192];snprintf(name,sizeof name,"layers.%d.attn.wo_a.weight",layer);
    const ds41f_weight *w=ds41f_weight_find(&weights,name);
    if(w&&w->int8.weight){for(size_t first=0;first<n;first+=VERIFY_KERNEL_BATCH){size_t m=n-first;if(m>VERIFY_KERNEL_BATCH)m=VERIFY_KERNEL_BATCH;
        CHECK(ds41f_int8_linear_batch(w,(float *)v->local+first*2048,2048,(float *)v->attended+first*8192,8192,m,1024,0));}
        for(size_t i=0;i<n;++i)ds41f_round_bf16(v->local[i],2048);}
    else for(size_t i=0;i<n;++i)CHECK(ds41f_attention_grouped_output(&weights,layer,v->local[i],v->attended[i],2));
    if(verify_timing){detail[3]+=now()-mark;mark=now();}
    if(verify_comm_batch)ds41f_comm_tp_gather_batch((float *)v->projected,8192,(float *)v->local,2048,2048,n,owner,1);
    else for(size_t i=0;i<n;++i)ds41f_comm_tp_allgather(v->projected[i],v->local[i],2048);
    if(verify_timing){detail[4]+=now()-mark;mark=now();}
    verify_linear(layer,"attn.wo_b",(float *)v->part,1280,(float *)v->projected,8192,n,0);
    if(verify_comm_batch)ds41f_comm_tp_gather_batch((float *)v->output,5120,(float *)v->part,1280,1280,n,owner,0);
    else for(size_t i=0;i<n;++i)ds41f_comm_tp_gather(v->output[i],v->part[i],1280,owner);
    if(verify_timing)detail[5]+=now()-mark;
}
static void verify_shared(int layer)
{
    verifier_work *v=verifier;int owner=pipeline_owner_for_layer(layer);if(!tp_member(owner))return;
    size_t n=v->count;
    verify_linear(layer,"ffn.shared_experts.w1",(float *)v->gate,2304,(float *)v->ffn,5132,n,0);
    verify_linear(layer,"ffn.shared_experts.w3",(float *)v->up,2304,(float *)v->ffn,5132,n,0);
    for(size_t i=0;i<n;++i){ds41f_swiglu(v->local[i],v->gate[i],v->up[i],576,10);ds41f_round_bf16(v->local[i],576);
        if(!verify_comm_batch)ds41f_comm_tp_allgather(v->hidden[i],v->local[i],576);}
    if(verify_comm_batch)ds41f_comm_tp_gather_batch((float *)v->hidden,2304,(float *)v->local,2048,576,n,owner,1);
    verify_linear(layer,"ffn.shared_experts.w2",(float *)v->part,1280,(float *)v->hidden,2304,n,0);
    if(verify_comm_batch)ds41f_comm_tp_gather_batch((float *)v->shared,5120,(float *)v->part,1280,1280,n,owner,0);
    else for(size_t i=0;i<n;++i)ds41f_comm_tp_gather(v->shared[i],v->part[i],1280,owner);
}
static void verify_experts(int layer)
{
    verifier_work *v=verifier;size_t n=v->count;
    if(!verify_expert_batch||n==1){for(size_t t=0;t<n;++t)local_experts(layer,v->ffn[t],v->ffn[t]+5120,v->combined[t]);return;}
    for(size_t t=0;t<n;++t){int needed=0;for(int k=0;k<6;++k)if(pipeline_expert_owner(layer,(int)v->ffn[t][5120+k])==rank)needed=1;
        if(needed)CHECK(ds41f_act_quant(v->expert_input[t],v->ffn[t],5120));}
    for(int id=0;id<384;++id)if(pipeline_expert_owner(layer,id)==rank){size_t count=0,token[VERIFY_BATCH_MAX];int route_index[VERIFY_BATCH_MAX];float route[VERIFY_BATCH_MAX];
        for(size_t t=0;t<n;++t)for(int k=0;k<6;++k)if((int)v->ffn[t][5120+k]==id){
            if(count==VERIFY_BATCH_MAX)ds41f_comm_abort("duplicate expert route",EINVAL);
            token[count]=t;route_index[count]=k;route[count]=v->ffn[t][5126+k];++count;}
        if(!count)continue;
        ds41f_expert e={{0},{0},0};e.packed_sdot=weights.packed_experts;char name[192];
        for(int i=0;i<3;++i){snprintf(name,sizeof name,"layers.%d.ffn.experts.%d.w%d.weight",layer,id,i+1);e.weight[i]=tensor(name)->data;
            snprintf(name,sizeof name,"layers.%d.ffn.experts.%d.w%d.scale",layer,id,i+1);e.scale[i]=tensor(name)->data;}
        for(size_t first=0;first<count;first+=VERIFY_KERNEL_BATCH){size_t m=count-first;if(m>VERIFY_KERNEL_BATCH)m=VERIFY_KERNEL_BATCH;
            for(size_t i=0;i<m;++i)memcpy(v->expert_gather[i],v->expert_input[token[first+i]],5120*sizeof(float));
            CHECK(ds41f_expert_batch_prepared(&e,(float *)v->expert_output,5120,(float *)v->expert_gather,5120,route+first,v->expert_scratch,m));
            for(size_t i=0;i<m;++i)memcpy(v->expert_values[token[first+i]][route_index[first+i]],v->expert_output[i],5120*sizeof(float));}
    }
    /* Expert scheduling may change; the per-token sum order must not. */
    for(size_t t=0;t<n;++t){memset(v->combined[t],0,5120*sizeof(float));
        for(int k=0;k<6;++k)if(pipeline_expert_owner(layer,(int)v->ffn[t][5120+k])==rank)
            for(int j=0;j<5120;++j)v->combined[t][j]+=v->expert_values[t][k][j];}
}
static void forward_batch_range(const int *tokens,size_t count,size_t pos,int *prediction,size_t head_first,float *taps,
                                const char *logits_path,size_t logits_start,size_t logits_count,
                                int first_layer,int last_layer,int include_head,
                                int stage_input,const float *incoming)
{
    if(head_first>count||first_layer<0||last_layer<first_layer||last_layer>=40)
        ds41f_comm_abort("batch layer range",EINVAL);
    ds41f_profile_at(SIZE_MAX,0);double timing[8]={0},attention_detail[6]={0},layer_timing[40][7],mark=verify_timing?now():0;
    if(stage_input){
        /* A staged callback may install one metadata record per token before
         * entering this layer range. Preserve that journal instead of
         * reinitializing it and discarding the received state. */
        if(!verifier||!verifier->journal)verify_begin_stage(incoming,count,pos);
    }
    else verify_begin(tokens,count,pos);
    verifier_work *v=verifier;
    if(verify_timing){timing[0]=now()-mark;mark=now();}
    for(int layer=first_layer;layer<=last_layer;++layer){int owner=pipeline_owner_for_layer(layer);
        if(layer==1||layer==14)for(size_t i=0;i<count;++i)verify_engram(layer,v->residual[i],v->engram_rows[i][layer==1?0:1]);
        if(rank==owner)for(size_t i=0;i<count;++i){float *h=v->residual[i];
            if(layer>=37){float *tap=taps+i*15360+(layer-37)*5120;
                for(int j=0;j<5120;++j){float sum=0;for(int c=0;c<4;++c)sum+=h[c*5120+j];tap[j]=sum*.25f;}
                ds41f_round_bf16(tap,5120);}
            mixes(layer,"attn",h,v->attn_pre[i],v->attn_post[i],v->attn_comb[i]);
            ds41f_hc_pre(v->input[i],h,h+20480,5120);ds41f_round_bf16(v->input[i],5120);named_norm(layer,"attn_norm",v->input[i],v->input[i]);}
        if(verify_timing){double elapsed=now()-mark;timing[1]+=elapsed;layer_timing[layer][0]=elapsed;mark=now();}
        verify_attention(layer,attention_detail);
        if(verify_timing){double elapsed=now()-mark;timing[2]+=elapsed;layer_timing[layer][1]=elapsed;mark=now();}
        if(rank==owner)for(size_t i=0;i<count;++i){float *h=v->residual[i],*ffn=v->ffn[i];
            ds41f_hc_post(h,v->output[i],h,v->attn_post[i],v->attn_comb[i],5120);ds41f_round_bf16(h,20480);
            mixes(layer,"ffn",h,v->ffn_pre[i],v->ffn_post[i],v->ffn_comb[i]);
            ds41f_hc_pre(ffn,h,v->attn_pre[i],5120);ds41f_round_bf16(ffn,5120);named_norm(layer,"ffn_norm",ffn,ffn);
            float logits[384],prob[6];int ids[6];named_linear(layer,"ffn.gate",logits,ffn,1);char name[192];
            snprintf(name,sizeof name,"layers.%d.ffn.gate.bias",layer);CHECK(ds41f_gate(logits,tensor(name)->data,384,6,1,1.5f,ids,prob));
            for(int k=0;k<6;++k){ffn[5120+k]=(float)ids[k];ffn[5126+k]=prob[k];}}
        if(verify_timing){double elapsed=now()-mark;timing[3]+=elapsed;layer_timing[layer][2]=elapsed;mark=now();}
        if(verify_comm_batch&&pipeline_runtime.enabled)ds41f_comm_tp_bf16_broadcast_batch((float *)v->ffn,5132,5120,12,count,owner);
        else if(verify_comm_batch)ds41f_comm_bf16_broadcast_batch((float *)v->ffn,5132,5120,12,count,owner);
        else for(size_t i=0;i<count;++i)ds41f_comm_bf16_broadcast(v->ffn[i],5120,12,owner);
        if(verify_timing){double elapsed=now()-mark;timing[4]+=elapsed;layer_timing[layer][3]=elapsed;mark=now();}
        verify_experts(layer);
        if(verify_timing){double elapsed=now()-mark;timing[5]+=elapsed;layer_timing[layer][4]=elapsed;mark=now();}
        verify_shared(layer);
        if(prefill_batch&&pipeline_runtime.enabled)ds41f_comm_tp_reduce_owner((float *)v->combined,count*5120,owner);
        else if(prefill_batch)ds41f_comm_reduce_owner((float *)v->combined,count*5120,owner);
        else ds41f_comm_sum((float *)v->combined,count*5120);
        if(verify_timing){double elapsed=now()-mark;timing[6]+=elapsed;layer_timing[layer][5]=elapsed;mark=now();}
        if(rank==owner)for(size_t i=0;i<count;++i){float *h=v->residual[i];
            for(int j=0;j<5120;++j)v->combined[i][j]+=v->shared[i][j];ds41f_round_bf16(v->combined[i],5120);
            ds41f_hc_post(h,v->combined[i],h,v->ffn_post[i],v->ffn_comb[i],5120);ds41f_round_bf16(h,20480);
            memcpy(h+20480,v->ffn_pre[i],4*sizeof(float));}
        int next_owner;
        if(pipeline_runtime.enabled)next_owner=layer<last_layer?pipeline_owner_for_layer(layer+1):owner;
        else next_owner=layer==39?11:(layer+1)%12;
        if(verify_comm_batch)ds41f_comm_bf16_handoff_batch((float *)v->residual,20484,20480,4,count,owner,next_owner);
        else for(size_t i=0;i<count;++i)ds41f_comm_bf16_handoff(v->residual[i],20480,4,owner,next_owner);
        if(verify_timing){double elapsed=now()-mark;timing[7]+=elapsed;layer_timing[layer][6]=elapsed;mark=now();}
    }
    if(!include_head){attention=v->base;return;}
    const ds41f_weight *head=tensor("head.weight");float *logits=malloc(head->rows*sizeof(float));if(!logits)ds41f_comm_abort("batch logits",ENOMEM);
    for(size_t i=head_first;i<count;++i){float *x=v->input[i];
        if(rank==11){ds41f_hc_pre(x,v->residual[i],v->residual[i]+20480,5120);ds41f_round_bf16(x,5120);CHECK(ds41f_norm(&weights,"norm.weight",x,x));}
        ds41f_comm_bf16_broadcast(x,5120,0,11);CHECK(ds41f_linear(&weights,"head",logits,x,1));
        size_t best;CHECK(ds41f_argmax_finite(logits,head->rows,&best));int id=(int)(best+head->row_start);float value=logits[best];
        ds41f_comm_argmax(&value,&id);prediction[i]=id;
        if(logits_path&&pos+i>=logits_start&&pos+i-logits_start<logits_count){float *full=rank==11?malloc(129280*sizeof(float)):NULL;
            if(rank==11&&!full)ds41f_comm_abort("batch full logits",ENOMEM);ds41f_comm_head_logits(full,logits,head->rows);
            if(rank==11){char path[4096];snprintf(path,sizeof path,"%s.pos%zu.bin",logits_path,pos+i);FILE *f=fopen(path,"wb");
                if(!f||fwrite(full,sizeof(float),129280,f)!=129280||fclose(f))ds41f_comm_abort("batch logit dump",EIO);}free(full);}
        for(int j=0;j<3;++j)ds41f_comm_bf16_broadcast(taps+i*15360+j*5120,5120,0,(37+j)%12);
    }
    free(logits);attention=v->base;
    if(verify_timing){verifier_timing_record *r=v->timing+v->timing_count++%VERIFY_TIMING_RECORDS;
        memcpy(r->summary,timing,sizeof timing);r->summary[8]=now()-mark;
        memcpy(r->summary+9,attention_detail,sizeof attention_detail);
        memcpy(r->layer,layer_timing,sizeof layer_timing);r->position=pos;r->inputs=count;}

}
static void forward_batch(const int *tokens,size_t count,size_t pos,int *prediction,size_t head_first,float *taps,
                          const char *logits_path,size_t logits_start,size_t logits_count)
{
    forward_batch_range(tokens,count,pos,prediction,head_first,taps,logits_path,logits_start,logits_count,0,39,1,0,NULL);
}
typedef struct { const int *tokens; int *prediction; float *taps;
    const char *logits_path; size_t logits_start, logits_count;
    const float *incoming_residual; size_t incoming_stride;
    const ds41f_pipeline_wire *incoming_metadata; size_t metadata_stride, metadata_token_stride, metadata_bytes;
    const uint32_t *incoming_history; size_t incoming_history_len;
    float *outgoing_residual; size_t outgoing_stride;
    uint8_t *outgoing_metadata; size_t outgoing_metadata_stride, outgoing_metadata_token_stride; } pipeline_forward_context;
static void verify_finish(size_t keep);
/* Not selected by legacy dispatch until stage-local state setup replaces the
 * global verifier initialization. */
static int __attribute__((unused)) pipeline_forward_stage_callback(void *opaque,const ds41f_pipeline *pipeline,
                                           size_t tile,size_t position,size_t count,int slot)
{
    (void)tile;(void)slot; pipeline_forward_context *context=opaque;
    if(!context||!pipeline||!pipeline->enabled||!context->tokens||!context->prediction||
       !context->taps||position>SIZE_MAX-count||(pipeline->stage&&
       (!context->incoming_residual||context->incoming_stride<20484||!context->incoming_metadata||
        !context->incoming_history||context->incoming_history_len>4||
        !context->metadata_bytes||context->metadata_stride<context->metadata_bytes)))return EINVAL;
    if(pipeline->stage){
        memcpy(engram.history,context->incoming_history,context->incoming_history_len*sizeof(*engram.history));
        engram.history_len= context->incoming_history_len;
        const uint8_t *base=(const uint8_t *)context->incoming_metadata+tile*context->metadata_stride;
        verify_begin_stage(context->incoming_residual+tile*context->incoming_stride,count,position);
        for(size_t i=0;i<count;++i){
            const ds41f_pipeline_wire *wire=(const ds41f_pipeline_wire *)(base+i*context->metadata_token_stride);
            if(ds41f_attention_receive_pipeline_metadata(&verifier->state[i],wire,context->metadata_bytes,
                                                          (uint32_t)tile,(uint32_t)position,(uint32_t)count))return EINVAL;
        }
    }
    size_t head_first=pipeline->stage==DS41F_PIPELINE_STAGES-1?0:count;
    const float *incoming=pipeline->stage?context->incoming_residual+tile*context->incoming_stride:NULL;
    forward_batch_range(context->tokens+position,count,position,context->prediction+position,
                        head_first,context->taps+position*15360,context->logits_path,
                        context->logits_start,context->logits_count,
                        pipeline->first_layer,pipeline->last_layer,
                        pipeline->stage==DS41F_PIPELINE_STAGES-1,
                        pipeline->stage!=0,
                        incoming);
    if(pipeline->stage<DS41F_PIPELINE_STAGES-1){
        if(!context->outgoing_residual||context->outgoing_stride<20484||
           !context->outgoing_metadata||context->outgoing_metadata_stride<count*context->outgoing_metadata_token_stride)
            return EINVAL;
        for(size_t i=0;i<count;++i){
            memcpy(context->outgoing_residual+tile*context->outgoing_stride+i*20484,
                   verifier->residual[i],20484*sizeof(float));
            ds41f_attention *s=&verifier->state[i]; uint32_t ids[DS41F_PIPELINE_SELECTED_MAX];
            uint32_t candidates[DS41F_PIPELINE_CANDIDATE_MAX]; size_t nc=0;
            for(size_t j=0;j<s->selected_count;++j)ids[j]=(uint32_t)s->selected[j];
            size_t cb=(s->capacity+7)/8; for(size_t j=0;j<cb&&nc<DS41F_PIPELINE_CANDIDATE_MAX;++j)if(s->candidate_blocks[j])candidates[nc++]=(uint32_t)j;
            ds41f_pipeline_wire *w=(ds41f_pipeline_wire *)(context->outgoing_metadata+tile*context->outgoing_metadata_stride+i*context->outgoing_metadata_token_stride);
            if(ds41f_pipeline_wire_init(w,(uint32_t)tile,(uint32_t)position,(uint32_t)count,ids,s->selected_count,candidates,nc,s->publication,sizeof s->publication))return EINVAL;
        }
    }
    verify_finish(count);
    return 0;
}
static int __attribute__((unused)) pipeline_forward_replay(const ds41f_pipeline *pipeline,
                                                           const int *tokens,size_t total_tokens,
                                                           size_t tile_tokens,int *prediction,float *taps,
                                                           const float *incoming_residual,size_t incoming_stride,
                                                           const ds41f_pipeline_wire *incoming_metadata,
                                                           size_t metadata_stride,size_t metadata_bytes,
                                                           const char *logits_path,size_t logits_start,size_t logits_count)
{
    if(!pipeline||!tokens||!total_tokens||!tile_tokens||!prediction||!taps||
       (pipeline->stage&&( !incoming_residual||!incoming_metadata||!metadata_bytes)))return EINVAL;
    size_t tiles=(total_tokens+tile_tokens-1)/tile_tokens;
    pipeline_forward_context context={
        .tokens=tokens,.prediction=prediction,.taps=taps,.logits_path=logits_path,
        .logits_start=logits_start,.logits_count=logits_count,
        .incoming_residual=incoming_residual,.incoming_stride=incoming_stride,
        .incoming_metadata=incoming_metadata,.metadata_stride=metadata_stride,
        .metadata_token_stride=sizeof(ds41f_pipeline_wire),.metadata_bytes=metadata_bytes,
        .incoming_history=NULL,.incoming_history_len=0};
    return ds41f_pipeline_run(tiles,tile_tokens,total_tokens,pipeline,
                              pipeline_forward_stage_callback,&context);
}
static void verifier_write_timing(void)
{
    if(!verifier||!verify_timing)return;
    verifier_work *v=verifier;size_t first=v->timing_count>VERIFY_TIMING_RECORDS?v->timing_count-VERIFY_TIMING_RECORDS:0;
    char path[80];snprintf(path,sizeof path,"verify-timing.rank%02d.log",rank);
    FILE *f=fopen(path,"w");if(!f)ds41f_comm_abort("verifier timing open",errno);
    for(size_t i=first;i<v->timing_count;++i){const verifier_timing_record *r=v->timing+i%VERIFY_TIMING_RECORDS;const double *t=r->summary;
        fprintf(f,"VERIFY_TIMING rank=%d pos=%zu inputs=%zu begin=%.6f pre=%.6f attention=%.6f gate=%.6f broadcast=%.6f experts=%.6f shared_reduce=%.6f post=%.6f head=%.6f\n",
            rank,r->position,r->inputs,t[0],t[1],t[2],t[3],t[4],t[5],t[6],t[7],t[8]);
        fprintf(f,"VERIFY_ATTENTION rank=%d pos=%zu inputs=%zu prepare=%.6f wqb=%.6f attend=%.6f woa=%.6f project_gather=%.6f wob_output_gather=%.6f\n",
            rank,r->position,r->inputs,t[9],t[10],t[11],t[12],t[13],t[14]);
        for(int layer=0;layer<40;++layer){t=r->layer[layer];
            fprintf(f,"VERIFY_LAYER rank=%d pos=%zu inputs=%zu layer=%d owner=%d pre=%.6f attention=%.6f gate=%.6f broadcast=%.6f experts=%.6f shared_reduce=%.6f post=%.6f\n",
                rank,r->position,r->inputs,layer,pipeline_owner_for_layer(layer),t[0],t[1],t[2],t[3],t[4],t[5],t[6]);}
    }
    int bad=ferror(f);if(fclose(f))bad=1;if(bad)ds41f_comm_abort("verifier timing write",EIO);
}
static void verify_finish(size_t keep)
{
    verifier_work *v=verifier;if(!v||!v->journal||keep>v->count)ds41f_comm_abort("batch commit",EINVAL);
    CHECK(ds41f_journal_finish(v->journal,keep));ds41f_journal_free(v->journal);v->journal=NULL;
    if(keep){ds41f_attention *chosen=v->state+keep-1;
        attention=*chosen;attention.window=v->base.window;attention.candidate_blocks=v->base.candidate_blocks;
        if(attention.window!=chosen->window)memcpy(attention.window,chosen->window,(size_t)40*128*512*sizeof(float));
        memcpy(attention.candidate_blocks,chosen->candidate_blocks,(attention.capacity+7)/8);
        memcpy(engram.history,v->history[keep-1],sizeof engram.history);engram.history_len=v->history_len[keep-1];
        for(int slot=0;slot<2;++slot){engram.table[slot].lookups=v->counters[keep-1][slot][0];
            engram.table[slot].local_rows=v->counters[keep-1][slot][1];engram.table[slot].remote_rows=v->counters[keep-1][slot][2];}}
}
#endif
