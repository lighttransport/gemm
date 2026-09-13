/* Included after transformer pool/allocation helpers. A64FX only. */
#ifndef A64FX_IQ4_CACHE_BUILD_H
#define A64FX_IQ4_CACHE_BUILD_H

typedef struct {
    qtensor *weight;
    tf_iq4_cache_view *view;
    size_t offset[48];
} tf_iq4_cache_record;

typedef struct {
    tf_iq4_cache_record *records;
    int count, tid, bad;
    size_t bytes;
    uint8_t *slab;
} tf_iq4_cache_build_task;

static void *tf_iq4_cache_build_worker(void *arg) {
    tf_iq4_cache_build_task *t=arg;
    unsigned long saved_mask=0;
    int saved_mode=0;
    if (syscall(SYS_get_mempolicy,&saved_mode,&saved_mask,8*sizeof(saved_mask),NULL,0) ||
        syscall(SYS_set_mempolicy,0,NULL,0)) { t->bad=1; return NULL; }
    if (t->bytes) {
        t->slab=tf_aligned_alloc_notouch(256,t->bytes);
        if (!t->slab) t->bad=1;
    }
    for (int i=0;t->slab && i<t->count;i++) {
        tf_iq4_cache_record *r=t->records+i;
        tf_iq4_cache_view *v=r->view;
        int first=v->start[t->tid],last=v->start[t->tid+1];
        size_t rb=tf_row_bytes(v->type,v->n_cols);
        tf_iq4_cache_block *dst=(tf_iq4_cache_block *)(t->slab+r->offset[t->tid]);
        v->slice[t->tid]=dst;
        for (int row=first;row<last;row++)
            t->bad+=tf_iq4_cache_pack_row(dst+(size_t)(row-first)*(v->n_cols/256),
                (const uint8_t *)r->weight->data+(size_t)row*rb,v->type,v->n_cols)!=0;
    }
    if (syscall(SYS_set_mempolicy,saved_mode,saved_mode?&saved_mask:NULL,8*sizeof(saved_mask)))
        t->bad++;
    return NULL;
}

static size_t tf_iq4_build_decode_cache(transformer_model *m,size_t budget) {
    if (!m || m->tp_size>1 || !budget || m->n_threads<1 || m->n_threads>48 ||
        svcntb()!=64 || (m->n_threads>1 && !m->pool_alive)) return 0;
    const int nt=m->n_threads;
    const long reserve_kb=6L*1024*1024;
    long available=tf_mem_available_kb();
    /* Allow one huge page of allocator rounding per slab, plus descriptors. */
    const size_t overhead=(size_t)nt*2097152+2097152;
    if (available<=reserve_kb || (size_t)(available-reserve_kb)*1024<=overhead) return 0;
    size_t limit=(size_t)(available-reserve_kb)*1024-overhead;
    if (limit>budget) limit=budget;
    size_t cap=(size_t)m->n_layers*12+1;
    tf_iq4_cache_record *records=calloc(cap,sizeof(*records));
    if (!records) return 0;
    int count=0,skipped=0;
    size_t total=0,slice_bytes[48]={0};
    for (int l=0;l<m->n_layers;l++) {
        transformer_layer *L=m->layers+l;
        qtensor *weights[]={&L->ffn_gate,&L->ffn_up,&L->ffn_down,
            &L->ssm_qkv,&L->ssm_gate,&L->ssm_out,&L->ssm_alpha,&L->ssm_beta,
            &L->attn_q,&L->attn_k,&L->attn_v,&L->attn_output};
        for (int j=0;j<12;j++) {
            qtensor *w=weights[j];
            if (!w->data || w->mixed_iq_cache || w->iq4_cache || w->n_rows<1 ||
                w->n_cols<1 || w->n_cols%256 || !tf_iq4_cache_palette(w->type)) continue;
            size_t nb=w->n_cols/256;
            if ((size_t)w->n_rows>SIZE_MAX/nb/sizeof(tf_iq4_cache_block)) continue;
            size_t bytes=(size_t)w->n_rows*nb*sizeof(tf_iq4_cache_block);
            if (bytes>limit-total) { skipped++;continue; }
            tf_iq4_cache_view *v=calloc(1,sizeof(*v));
            if (!v) { skipped++;continue; }
            v->n_slices=nt;v->n_cols=w->n_cols;v->type=w->type;
            int rp=w->n_rows/nt,extra=w->n_rows%nt,first=0;
            for (int t=0;t<nt;t++) {
                int rc=rp+(t<extra);
                v->start[t]=first;
                records[count].offset[t]=slice_bytes[t];
                slice_bytes[t]+=(size_t)rc*nb*sizeof(tf_iq4_cache_block);
                first+=rc;
            }
            v->start[nt]=w->n_rows;
            records[count].weight=w;records[count].view=v;
            count++;total+=bytes;
        }
    }
    if (!count) { free(records);return 0; }
    int need=m->decode_owned_count+count+nt;
    if (need>m->decode_owned_cap) {
        void **owned=realloc(m->decode_owned,(size_t)need*sizeof(*owned));
        if (!owned) {
            for (int i=0;i<count;i++) free(records[i].view);
            free(records);return 0;
        }
        m->decode_owned=owned;m->decode_owned_cap=need;
    }
    tf_iq4_cache_build_task tasks[48];
    for (int t=0;t<nt;t++) tasks[t]=(tf_iq4_cache_build_task){records,count,t,0,slice_bytes[t],NULL};
    if (nt>1) tf_pool_dispatch(m,tf_iq4_cache_build_worker,tasks,sizeof(*tasks));
    else tf_iq4_cache_build_worker(tasks);
    int bad=0;
    for (int t=0;t<nt;t++) bad+=tasks[t].bad;
    available=tf_mem_available_kb();
    if (bad || available<reserve_kb) {
        for (int t=0;t<nt;t++) free(tasks[t].slab);
        for (int i=0;i<count;i++) free(records[i].view);
        free(records);
        fprintf(stderr,"IQ4 decode cache: rejected (errors=%d, available=%.3f GiB)\n",bad,available/1048576.);
        return 0;
    }
    for (int t=0;t<nt;t++) m->decode_owned[m->decode_owned_count++]=tasks[t].slab;
    for (int i=0;i<count;i++) {
        m->decode_owned[m->decode_owned_count++]=records[i].view;
        records[i].weight->iq4_cache=records[i].view;
    }
    free(records);
    fprintf(stderr,"IQ4 decode cache: %d tensors %.3f GiB in %d worker-local slabs, skipped=%d, MemAvailable=%.3f GiB\n",
        count,total/1073741824.,nt,skipped,available/1048576.);
    return total;
}
#endif
