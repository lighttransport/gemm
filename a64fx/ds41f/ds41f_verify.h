#ifndef DS41F_VERIFY_H
#define DS41F_VERIFY_H
/* Included after runner helpers. One main thread owns MPI and mutable state.
 * Token-major window snapshots keep future writes invisible to earlier queries;
 * compressed appends are shared, but every selector is bounded by its position.
 * A full 1M compressed cache is never cloned. */
enum { VERIFY_TIMING_RECORDS=256 };
typedef struct {size_t position,inputs;double summary[9],layer[40][7];} verifier_timing_record;
typedef struct {
    ds41f_attention state[6],base;
    float *windows;uint8_t *candidates;
    ds41f_journal *journal;
    size_t count,start;
    uint32_t history[6][4],history_len[6];uint64_t counters[6][2][3];
    float engram_rows[6][2][24*256];
    float residual[6][20484],input[6][5120],output[6][5120],qr[6][1280];
    float q[6][8192],attended[6][8192],projected[6][8192],local[6][2048],part[6][1280];
    float attn_pre[6][4],attn_post[6][4],attn_comb[6][16],ffn_pre[6][4],ffn_post[6][4],ffn_comb[6][16];
    float ffn[6][5132],combined[6][5120],shared[6][5120],gate[6][2304],up[6][2304],hidden[6][2304];
    float expert_values[6][6][5120],expert_input[6][5120],expert_gather[6][5120],expert_output[6][5120];
    float expert_scratch[6*3*2304];
    /* Bounded recent diagnostic records; never write shared-storage logs
     * between verification and commit. */
    verifier_timing_record timing[VERIFY_TIMING_RECORDS];size_t timing_count;
} verifier_work;
static verifier_work *verifier;
static void verifier_allocate(void)
{
    verifier=calloc(1,sizeof *verifier);if(!verifier)ds41f_comm_abort("verifier work",ENOMEM);
    verifier->windows=malloc((size_t)6*40*128*512*sizeof(float));
    verifier->candidates=malloc((size_t)6*((attention.capacity+7)/8));
    if(!verifier->windows||!verifier->candidates)ds41f_comm_abort("verifier causal windows",ENOMEM);
    /* Charge physical pages before the post-load MemAvailable guard. */
    memset(verifier->windows,0,(size_t)6*40*128*512*sizeof(float));
    memset(verifier->candidates,0,(size_t)6*((attention.capacity+7)/8));
}
static void verifier_free(void)
{if(verifier){free(verifier->windows);free(verifier->candidates);free(verifier);verifier=NULL;}}
static void verify_linear(int layer,const char *part,float *out,size_t out_stride,
                          const float *x,size_t x_stride,size_t count,int raw)
{char name[192];snprintf(name,sizeof name,"layers.%d.%s",layer,part);
    CHECK(ds41f_linear_batch(&weights,name,out,out_stride,x,x_stride,count,raw));}
static void verify_begin(const int *tokens,size_t count,size_t pos)
{
    verifier_work *v=verifier;if(!v||!count||count>6||v->journal)ds41f_comm_abort("verifier lifetime",EINVAL);
    v->base=attention;v->count=count;v->start=pos;size_t candidate_bytes=(attention.capacity+7)/8;
    CHECK(ds41f_journal_create(&v->journal,&attention,&engram,prefetch,pos,count,
        ds41f_journal_bytes(attention.capacity,count)));
    for(size_t i=0;i<count;++i){
        CHECK(ds41f_journal_record(v->journal,pos+i));v->state[i]=attention;
        v->state[i].window=v->windows+i*40*128*512;v->state[i].candidate_blocks=v->candidates+i*candidate_bytes;
        memcpy(v->state[i].window,attention.window,(size_t)40*128*512*sizeof(float));
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
static void verify_engram(int layer,float *h,float *rows)
{
    ds41f_comm_sum(rows,24*256);
    if(rank==layer%12){float kv[5*5120];named_linear(layer,"engram.wkv",kv,rows,0);
        char name[192];snprintf(name,sizeof name,"layers.%d.engram.q_weight",layer);const uint16_t *qw=tensor(name)->data;
        snprintf(name,sizeof name,"layers.%d.engram.k_weight",layer);const uint16_t *kw=tensor(name)->data;
        float q[4*5120],k[4*5120];for(int j=0;j<4*5120;++j){q[j]=ds41f_bf16_to_f32(qw[j]);k[j]=ds41f_bf16_to_f32(kw[j]);}
        ds41f_engram_fuse(h,kv,kv+4*5120,q,k,5120,1e-20f);ds41f_round_bf16(h,20480);}
}
static void verify_attention(int layer)
{
    verifier_work *v=verifier;size_t n=v->count;int owner=layer%12;
    for(size_t i=0;i<n;++i){attention=v->state[i];
        if(i&&(layer==2||layer==8||layer==14)){int src=layer==2?0:layer==8?1:2;
            memcpy(attention.pool_value[src],v->state[i-1].pool_value[src],512*sizeof(float));
            memcpy(attention.pool_score[src],v->state[i-1].pool_score[src],512*sizeof(float));}
        if(tp_member(owner)){ds41f_attention_context c;
            if(rank==owner)CHECK(ds41f_attention_prepare(&attention,&weights,layer,v->start+i,v->input[i],&c));
            ds41f_comm_tp_bytes(&c,sizeof c,owner);
            if(rank!=owner)CHECK(ds41f_attention_apply(&attention,layer,v->start+i,&c));
            memcpy(v->qr[i],c.qr,sizeof c.qr);
            for(size_t future=i+1;future<n;++future)
                memcpy(v->state[future].window+((size_t)layer*128+(v->start+i)%128)*512,c.kv,sizeof c.kv);
        }
        sync_attention(layer,v->start+i);v->state[i]=attention;
    }
    if(!tp_member(owner))return;
    verify_linear(layer,"attn.wq_b",(float *)v->q,8192,(float *)v->qr,1280,n,0);
    for(size_t i=0;i<n;++i)CHECK(ds41f_attention_attend(v->state+i,&weights,layer,v->start+i,
        v->q[i],(rank%4)*16,16,v->attended[i]));
    char name[192];snprintf(name,sizeof name,"layers.%d.attn.wo_a.weight",layer);
    const ds41f_weight *w=ds41f_weight_find(&weights,name);
    if(w&&w->int8.weight){CHECK(ds41f_int8_linear_batch(w,(float *)v->local,2048,(float *)v->attended,8192,n,1024,0));
        for(size_t i=0;i<n;++i)ds41f_round_bf16(v->local[i],2048);}
    else for(size_t i=0;i<n;++i)CHECK(ds41f_attention_grouped_output(&weights,layer,v->local[i],v->attended[i],2));
    if(verify_comm_batch)ds41f_comm_tp_gather_batch((float *)v->projected,8192,(float *)v->local,2048,2048,n,owner,1);
    else for(size_t i=0;i<n;++i)ds41f_comm_tp_allgather(v->projected[i],v->local[i],2048);
    verify_linear(layer,"attn.wo_b",(float *)v->part,1280,(float *)v->projected,8192,n,0);
    if(verify_comm_batch)ds41f_comm_tp_gather_batch((float *)v->output,5120,(float *)v->part,1280,1280,n,owner,0);
    else for(size_t i=0;i<n;++i)ds41f_comm_tp_gather(v->output[i],v->part[i],1280,owner);
}
static void verify_shared(int layer)
{
    verifier_work *v=verifier;int owner=layer%12;if(!tp_member(owner))return;
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
    for(size_t t=0;t<n;++t){int needed=0;for(int k=0;k<6;++k)if((int)v->ffn[t][5120+k]%12==rank)needed=1;
        if(needed)CHECK(ds41f_act_quant(v->expert_input[t],v->ffn[t],5120));}
    for(int id=rank;id<384;id+=12){size_t count=0,token[6];int route_index[6];float route[6];
        for(size_t t=0;t<n;++t)for(int k=0;k<6;++k)if((int)v->ffn[t][5120+k]==id){
            if(count==6)ds41f_comm_abort("duplicate expert route",EINVAL);
            token[count]=t;route_index[count]=k;route[count]=v->ffn[t][5126+k];
            memcpy(v->expert_gather[count],v->expert_input[t],5120*sizeof(float));++count;}
        if(!count)continue;
        ds41f_expert e={{0},{0},0};e.packed_sdot=weights.packed_experts;char name[192];
        for(int i=0;i<3;++i){snprintf(name,sizeof name,"layers.%d.ffn.experts.%d.w%d.weight",layer,id,i+1);e.weight[i]=tensor(name)->data;
            snprintf(name,sizeof name,"layers.%d.ffn.experts.%d.w%d.scale",layer,id,i+1);e.scale[i]=tensor(name)->data;}
        CHECK(ds41f_expert_batch_prepared(&e,(float *)v->expert_output,5120,(float *)v->expert_gather,5120,route,v->expert_scratch,count));
        for(size_t i=0;i<count;++i)memcpy(v->expert_values[token[i]][route_index[i]],v->expert_output[i],5120*sizeof(float));
    }
    /* Expert scheduling may change; the per-token sum order must not. */
    for(size_t t=0;t<n;++t){memset(v->combined[t],0,5120*sizeof(float));
        for(int k=0;k<6;++k)if((int)v->ffn[t][5120+k]%12==rank)
            for(int j=0;j<5120;++j)v->combined[t][j]+=v->expert_values[t][k][j];}
}
static void forward_batch(const int *tokens,size_t count,size_t pos,int *prediction,float *taps,
                           const char *logits_path,size_t logits_start,size_t logits_count)
{
    ds41f_profile_at(SIZE_MAX,0);double timing[8]={0},layer_timing[40][7],mark=verify_timing?now():0;
    verify_begin(tokens,count,pos);verifier_work *v=verifier;
    if(verify_timing){timing[0]=now()-mark;mark=now();}
    for(int layer=0;layer<40;++layer){int owner=layer%12;
        if(layer==1||layer==14)for(size_t i=0;i<count;++i)verify_engram(layer,v->residual[i],v->engram_rows[i][layer==1?0:1]);
        if(rank==owner)for(size_t i=0;i<count;++i){float *h=v->residual[i];
            if(layer>=37){float *tap=taps+i*15360+(layer-37)*5120;
                for(int j=0;j<5120;++j){float sum=0;for(int c=0;c<4;++c)sum+=h[c*5120+j];tap[j]=sum*.25f;}
                ds41f_round_bf16(tap,5120);}
            mixes(layer,"attn",h,v->attn_pre[i],v->attn_post[i],v->attn_comb[i]);
            ds41f_hc_pre(v->input[i],h,h+20480,5120);ds41f_round_bf16(v->input[i],5120);named_norm(layer,"attn_norm",v->input[i],v->input[i]);}
        if(verify_timing){double elapsed=now()-mark;timing[1]+=elapsed;layer_timing[layer][0]=elapsed;mark=now();}
        verify_attention(layer);
        if(verify_timing){double elapsed=now()-mark;timing[2]+=elapsed;layer_timing[layer][1]=elapsed;mark=now();}
        if(rank==owner)for(size_t i=0;i<count;++i){float *h=v->residual[i],*ffn=v->ffn[i];
            ds41f_hc_post(h,v->output[i],h,v->attn_post[i],v->attn_comb[i],5120);ds41f_round_bf16(h,20480);
            mixes(layer,"ffn",h,v->ffn_pre[i],v->ffn_post[i],v->ffn_comb[i]);
            ds41f_hc_pre(ffn,h,v->attn_pre[i],5120);ds41f_round_bf16(ffn,5120);named_norm(layer,"ffn_norm",ffn,ffn);
            float logits[384],prob[6];int ids[6];named_linear(layer,"ffn.gate",logits,ffn,1);char name[192];
            snprintf(name,sizeof name,"layers.%d.ffn.gate.bias",layer);CHECK(ds41f_gate(logits,tensor(name)->data,384,6,1,1.5f,ids,prob));
            for(int k=0;k<6;++k){ffn[5120+k]=(float)ids[k];ffn[5126+k]=prob[k];}}
        if(verify_timing){double elapsed=now()-mark;timing[3]+=elapsed;layer_timing[layer][2]=elapsed;mark=now();}
        if(verify_comm_batch)ds41f_comm_bf16_broadcast_batch((float *)v->ffn,5132,5120,12,count,owner);
        else for(size_t i=0;i<count;++i)ds41f_comm_bf16_broadcast(v->ffn[i],5120,12,owner);
        if(verify_timing){double elapsed=now()-mark;timing[4]+=elapsed;layer_timing[layer][3]=elapsed;mark=now();}
        verify_experts(layer);
        if(verify_timing){double elapsed=now()-mark;timing[5]+=elapsed;layer_timing[layer][4]=elapsed;mark=now();}
        verify_shared(layer);ds41f_comm_sum((float *)v->combined,count*5120);
        if(verify_timing){double elapsed=now()-mark;timing[6]+=elapsed;layer_timing[layer][5]=elapsed;mark=now();}
        if(rank==owner)for(size_t i=0;i<count;++i){float *h=v->residual[i];
            for(int j=0;j<5120;++j)v->combined[i][j]+=v->shared[i][j];ds41f_round_bf16(v->combined[i],5120);
            ds41f_hc_post(h,v->combined[i],h,v->ffn_post[i],v->ffn_comb[i],5120);ds41f_round_bf16(h,20480);
            memcpy(h+20480,v->ffn_pre[i],4*sizeof(float));}
        if(verify_comm_batch)ds41f_comm_bf16_handoff_batch((float *)v->residual,20484,20480,4,count,owner,layer==39?11:(layer+1)%12);
        else for(size_t i=0;i<count;++i)ds41f_comm_bf16_handoff(v->residual[i],20480,4,owner,layer==39?11:(layer+1)%12);
        if(verify_timing){double elapsed=now()-mark;timing[7]+=elapsed;layer_timing[layer][6]=elapsed;mark=now();}
    }
    const ds41f_weight *head=tensor("head.weight");float *logits=malloc(head->rows*sizeof(float));if(!logits)ds41f_comm_abort("batch logits",ENOMEM);
    for(size_t i=0;i<count;++i){float *x=v->input[i];
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
        memcpy(r->layer,layer_timing,sizeof layer_timing);r->position=pos;r->inputs=count;}

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
        for(int layer=0;layer<40;++layer){t=r->layer[layer];
            fprintf(f,"VERIFY_LAYER rank=%d pos=%zu inputs=%zu layer=%d owner=%d pre=%.6f attention=%.6f gate=%.6f broadcast=%.6f experts=%.6f shared_reduce=%.6f post=%.6f\n",
                rank,r->position,r->inputs,layer,layer%12,t[0],t[1],t[2],t[3],t[4],t[5],t[6]);}
    }
    int bad=ferror(f);if(fclose(f))bad=1;if(bad)ds41f_comm_abort("verifier timing write",EIO);
}
static void verify_finish(size_t keep)
{
    verifier_work *v=verifier;if(!v||!v->journal||keep>v->count)ds41f_comm_abort("batch commit",EINVAL);
    CHECK(ds41f_journal_finish(v->journal,keep));ds41f_journal_free(v->journal);v->journal=NULL;
    if(keep){ds41f_attention *chosen=v->state+keep-1;
        attention=*chosen;attention.window=v->base.window;attention.candidate_blocks=v->base.candidate_blocks;
        memcpy(attention.window,chosen->window,(size_t)40*128*512*sizeof(float));
        memcpy(attention.candidate_blocks,chosen->candidate_blocks,(attention.capacity+7)/8);
        memcpy(engram.history,v->history[keep-1],sizeof engram.history);engram.history_len=v->history_len[keep-1];
        for(int slot=0;slot<2;++slot){engram.table[slot].lookups=v->counters[keep-1][slot][0];
            engram.table[slot].local_rows=v->counters[keep-1][slot][1];engram.table[slot].remote_rows=v->counters[keep-1][slot][2];}}
}
#endif
