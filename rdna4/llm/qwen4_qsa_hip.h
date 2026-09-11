/* Explicit QSA state, independent of the trunk KV pointer arrays. */
static void hllm_qwen4_qsa_free(hip_llm_runner *r) {
    if(r->layers)for(int l=0;l<r->n_layers;++l) {
        if(r->layers[l].d_index_cache)hipFree(r->layers[l].d_index_cache);
        r->layers[l].d_index_cache=NULL;
    }
#define QSA_FREE(p) do {if(r->p)hipFree(r->p);r->p=NULL;} while(0)
    QSA_FREE(d_qsa_q); QSA_FREE(d_qsa_k); QSA_FREE(d_qsa_scores); QSA_FREE(d_qsa_ids); QSA_FREE(d_qsa_marks); QSA_FREE(d_qsa_freq);
#undef QSA_FREE
    free(r->h_qsa_scores);r->h_qsa_scores=NULL;
    free(r->h_qsa_ids);r->h_qsa_ids=NULL;
    free(r->h_qsa_blocks);r->h_qsa_blocks=NULL;
}

int hip_llm_qwen4_exact_enable(hip_llm_runner *r) {
    if(!r || !r->weights_loaded || !r->is_qwen4exp || r->indexer_dim>256 ||
       r->head_dim>256)return -1;
    if(r->qwen4_exact)return 0;
    size_t n=(size_t)r->max_seq_len;
    if(hipMalloc(&r->d_qsa_q,(size_t)r->indexer_heads*r->indexer_dim*sizeof(float)) ||
       hipMalloc(&r->d_qsa_k,(size_t)r->indexer_dim*sizeof(float)) ||
       hipMalloc(&r->d_qsa_scores,n*sizeof(float)) ||
       hipMalloc(&r->d_qsa_ids,n*sizeof(int)) ||
       hipMalloc(&r->d_qsa_marks,n*sizeof(int)) ||
       hipMalloc(&r->d_qsa_freq,(size_t)(r->indexer_dim/2)*sizeof(float)))goto fail;
    {
        int pairs=r->indexer_dim/2;
        int rope_pairs=r->use_mrope ? r->mrope_sections[0]+r->mrope_sections[1]+
            r->mrope_sections[2]+r->mrope_sections[3] : pairs;
        float *freq=(float *)malloc((size_t)pairs*sizeof(float));
        if(!freq || !r->rope_freq_base) { free(freq); goto fail; }
        for(int d=0;d<pairs;++d)
            freq[d]=powf(r->rope_freq_base,-(float)d/(float)rope_pairs);
        if(hipMemcpy(r->d_qsa_freq,freq,(size_t)pairs*sizeof(float),hipMemcpyHostToDevice)) {
            free(freq); goto fail;
        }
        free(freq);
    }
    r->h_qsa_scores=malloc(n*sizeof(float));
    r->h_qsa_ids=malloc(n*sizeof(int));
    r->h_qsa_blocks=malloc(n*sizeof(qwen4_qsa_block));
    if(!r->h_qsa_scores || !r->h_qsa_ids || !r->h_qsa_blocks)goto fail;
    for(int l=0;l<r->n_layers;++l) {
        hip_layer *cl=&r->layers[l];
        if(cl->is_ssm || cl->qsa_ratio==0)continue;
        if(hipMalloc(&cl->d_index_cache,n*r->indexer_dim*sizeof(uint16_t)) ||
           hipMemset(cl->d_index_cache,0,n*r->indexer_dim*sizeof(uint16_t)))goto fail;
    }
    r->qwen4_exact=1;
    r->qwen4_coding_profile=0;
    hip_llm_set_batched_path(r,0);
    return 0;
fail:
    hllm_qwen4_qsa_free(r);
    return -1;
}

static int hllm_qwen4_qsa_attention(hip_llm_runner *r, hip_layer *cl,
                                    void *keys, void *values) {
    if(!r->qwen4_exact || !cl->qsa_ratio)return 0;
    int dim=r->indexer_dim, nh=r->indexer_heads, ratio=cl->qsa_ratio;
    int n=r->cur_position+1;
    launch_matvec_auto(r,r->d_qsa_k,cl->index_k_w,r->d_xb,dim,r->n_embd,cl->index_k_type);
    launch_kv_store_f16_devp(r,cl->d_index_cache,cl->d_index_cache,
                            r->d_qsa_k,r->d_qsa_k,dim);
    int width = ratio-1>n-r->indexer_top_k || r->indexer_top_k>n ? n : r->indexer_top_k+ratio-1;
    /* Selected attention is slower than the tuned dense kernel when it saves
     * only a small tail.  Keep the exact index-cache update above, but avoid
     * replacing dense attention until at least 25% of the context is removed.
     * This is especially important around 2K, where top_k is 2048. */
    if(n<=r->indexer_top_k+ratio-1 || (long long)width*4 >= (long long)n*3)return 0;
    launch_matvec_auto(r,r->d_qsa_q,cl->index_q_w,r->d_xb,dim*nh,r->n_embd,cl->index_q_type);
    launch_qknorm(r,r->d_qsa_q,cl->index_q_norm_w,nh,dim,r->rms_norm_eps);
    launch_rope_devp(r,r->d_qsa_q,nh,dim,r->rope_freq_base);
    int blocks=n/ratio, pairs=r->use_mrope ? r->mrope_sections[0]+r->mrope_sections[1]+r->mrope_sections[2]+r->mrope_sections[3] : dim/2;
    int active=r->use_mrope ? r->mrope_sections[0]+r->mrope_sections[1]+r->mrope_sections[2] : dim/2;
    void *a[]={&r->d_qsa_scores,&cl->d_index_cache,&r->d_qsa_q,&cl->index_k_norm_w,
        &dim,&nh,&ratio,&blocks,&pairs,&active,&r->d_qsa_freq,&r->rms_norm_eps};
    int qsa_threads=dim<=64 ? 64 : (dim<=128 ? 128 : 256);
    LAUNCH(r->fn_qwen4_qsa_scores,blocks,1,1,qsa_threads,1,1,0,r->stream,a);
    const char *device_select=getenv("LLM_QWEN4_QSA_DEVICE_SELECT");
    int use_device_select=device_select && atoi(device_select)!=0;
    int count;
    if(use_device_select) {
        int chunks=(blocks+255)/256;
        void *s[]={&r->d_qsa_scores,&r->d_qsa_marks,&blocks};
        LAUNCH(r->fn_qwen4_qsa_sort_blocks,chunks,1,1,1,1,1,0,r->stream,s);
        void *m[]={&r->d_qsa_scores,&r->d_qsa_ids,&r->d_qsa_marks,
                   &n,&ratio,&r->indexer_top_k};
        LAUNCH(r->fn_qwen4_qsa_merge_ids,1,1,1,1,1,1,0,r->stream,m);
        count=width;
    } else {
        if(hipMemcpyAsync(r->h_qsa_scores,r->d_qsa_scores,(size_t)blocks*sizeof(float),hipMemcpyDeviceToHost,r->stream) ||
           hipStreamSynchronize(r->stream))return -1;
        count=qwen4_qsa_select(r->h_qsa_scores,n,ratio,r->indexer_top_k,r->h_qsa_blocks,r->h_qsa_ids);
        if(count<0 || hipMemcpyAsync(r->d_qsa_ids,r->h_qsa_ids,(size_t)count*sizeof(int),hipMemcpyHostToDevice,r->stream))return -1;
    }
    int first=0;
    int layer = (int)(cl - r->layers);
    const char *warp_env=getenv("LLM_QWEN4_QSA_WARP_ATTN");
    int warp_attn=warp_env && atoi(warp_env)!=0;
    if (r->qwen4_kv_i8) {
        void *b[]={&r->d_xb2,&r->d_q,&keys,&values,
            &r->d_key_cache_scale[layer],&r->d_value_cache_scale[layer],
            &r->d_qsa_ids,&count,&first,&r->d_position,
            &r->n_heads,&r->n_kv_heads,&r->head_dim};
        LAUNCH(warp_attn ? r->fn_qwen4_selected_attn_i8_warp : r->fn_qwen4_selected_attn_i8,
               r->n_heads,1,1,warp_attn?32:256,1,1,0,r->stream,b);
    } else {
        void *b[]={&r->d_xb2,&r->d_q,&keys,&values,&r->d_qsa_ids,&count,&first,&r->d_position,
            &r->n_heads,&r->n_kv_heads,&r->head_dim};
        LAUNCH(warp_attn ? r->fn_qwen4_selected_attn_warp : r->fn_qwen4_selected_attn,
               r->n_heads,1,1,warp_attn?32:256,1,1,0,r->stream,b);
    }
    return 1;
}
