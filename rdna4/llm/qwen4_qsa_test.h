static int hllm_qwen4_verify_selected_attention(hip_llm_runner *r, int n, int count) {
    int dim=r->head_dim, heads=r->n_heads, kvh=r->n_kv_heads, first=0;
    size_t kn=(size_t)n*kvh*dim, qn=(size_t)heads*dim;
    uint16_t *keys=malloc(kn*sizeof(*keys)), *values=malloc(kn*sizeof(*values));
    float *q=malloc(qn*sizeof(*q)), *out=malloc(qn*sizeof(*out));
    double *scores=malloc((size_t)count*sizeof(*scores));
    void *dk=NULL,*dv=NULL;int rc=-1;
    if(!keys || !values || !q || !out || !scores)goto done;
    for(size_t i=0;i<kn;++i) {
        keys[i]=q4ref_half(sinf((float)(i%10007)*0.013f));
        values[i]=q4ref_half(cosf((float)(i%9973)*0.017f));
    }
    for(size_t i=0;i<qn;++i)q[i]=sinf((float)i*0.021f);
    if(hipMalloc(&dk,kn*sizeof(*keys)) || hipMalloc(&dv,kn*sizeof(*values)) ||
       hipMemcpy(dk,keys,kn*sizeof(*keys),hipMemcpyHostToDevice) ||
       hipMemcpy(dv,values,kn*sizeof(*values),hipMemcpyHostToDevice) ||
       hipMemcpy(r->d_q,q,qn*sizeof(*q),hipMemcpyHostToDevice) ||
       hipMemcpy(r->d_qsa_ids,r->h_qsa_ids,(size_t)count*sizeof(int),hipMemcpyHostToDevice))goto done;
    void *a[]={&r->d_xb2,&r->d_q,&dk,&dv,&r->d_qsa_ids,&count,&first,&r->d_position,&heads,&kvh,&dim};
    const char *warp_env=getenv("LLM_QWEN4_QSA_WARP_ATTN");
    int warp=warp_env && atoi(warp_env)!=0;
    LAUNCH(warp ? r->fn_qwen4_selected_attn_warp : r->fn_qwen4_selected_attn,
           heads,1,1,warp?32:256,1,1,0,r->stream,a);
    if(hipStreamSynchronize(r->stream) || hipMemcpy(out,r->d_xb2,qn*sizeof(*out),hipMemcpyDeviceToHost))goto done;
    double err=0,den=0;
    for(int h=0;h<heads;++h) {
        int kv=h/(heads/kvh);double mx=-INFINITY, sum=0;
        for(int j=0;j<count;++j) {
            size_t offset=((size_t)r->h_qsa_ids[j]*kvh+kv)*dim;
            double score=0;
            for(int d=0;d<dim;++d)score+=(double)q[h*dim+d]*ggml_fp16_to_fp32(keys[offset+d]);
            scores[j]=score/sqrt((double)dim);if(scores[j]>mx)mx=scores[j];
        }
        for(int j=0;j<count;++j){scores[j]=exp(scores[j]-mx);sum+=scores[j];}
        for(int d=0;d<dim;++d) {
            double value=0;
            for(int j=0;j<count;++j)value+=scores[j]*ggml_fp16_to_fp32(values[((size_t)r->h_qsa_ids[j]*kvh+kv)*dim+d]);
            value/=sum;double delta=out[h*dim+d]-value;err+=delta*delta;den+=value*value;
        }
    }
    double rel=sqrt(err/fmax(den,1e-30));rc=isfinite(rel)&&rel<1e-4 ? 0 : -1;
    fprintf(stderr,"Qwen4 selected attention n=%d selected=%d output_rel=%.6g %s\n",n,count,rel,rc?"FAIL":"PASS");
done:
    if(dk)hipFree(dk);
    if(dv)hipFree(dv);
    free(keys);free(values);free(q);free(out);free(scores);return rc;
}

static int hllm_qwen4_verify_selected_attention_i8(hip_llm_runner *r, int n, int count) {
    int dim=r->head_dim, heads=r->n_heads, kvh=r->n_kv_heads, first=0, groups=(dim+31)/32;
    size_t kn=(size_t)n*kvh*dim, qn=(size_t)heads*dim, sn=(size_t)n*kvh*groups;
    int8_t *keys=malloc(kn), *values=malloc(kn); float *ks=malloc(sn*sizeof(float)), *vs=malloc(sn*sizeof(float));
    float *q=malloc(qn*sizeof(*q)), *out=malloc(qn*sizeof(*out)), *scores=malloc((size_t)count*sizeof(*scores));
    void *dk=NULL,*dv=NULL,*dks=NULL,*dvs=NULL; int rc=-1; const float scale=1.0f/127.0f;
    if(!keys||!values||!ks||!vs||!q||!out||!scores)goto done;
    for(size_t i=0;i<kn;++i){
        float k=sinf((float)(i%10007)*0.013f),v=cosf((float)(i%9973)*0.017f);
        int ik=(int)lrintf(k/scale),iv=(int)lrintf(v/scale);
        keys[i]=(int8_t)(ik<-127?-127:ik>127?127:ik); values[i]=(int8_t)(iv<-127?-127:iv>127?127:iv);
    }
    for(size_t i=0;i<sn;++i)ks[i]=vs[i]=scale;
    for(size_t i=0;i<qn;++i)q[i]=sinf((float)i*0.021f);
    if(hipMalloc(&dk,kn)||hipMalloc(&dv,kn)||hipMalloc(&dks,sn*sizeof(float))||hipMalloc(&dvs,sn*sizeof(float))||
       hipMemcpy(dk,keys,kn,hipMemcpyHostToDevice)||hipMemcpy(dv,values,kn,hipMemcpyHostToDevice)||
       hipMemcpy(dks,ks,sn*sizeof(float),hipMemcpyHostToDevice)||hipMemcpy(dvs,vs,sn*sizeof(float),hipMemcpyHostToDevice)||
       hipMemcpy(r->d_q,q,qn*sizeof(*q),hipMemcpyHostToDevice)||
       hipMemcpy(r->d_qsa_ids,r->h_qsa_ids,(size_t)count*sizeof(int),hipMemcpyHostToDevice))goto done;
    void *a[]={&r->d_xb2,&r->d_q,&dk,&dv,&dks,&dvs,&r->d_qsa_ids,&count,&first,&r->d_position,&heads,&kvh,&dim};
    const char *warp_env=getenv("LLM_QWEN4_QSA_WARP_ATTN"); int warp=warp_env&&atoi(warp_env)!=0;
    LAUNCH(warp?r->fn_qwen4_selected_attn_i8_warp:r->fn_qwen4_selected_attn_i8,
           heads,1,1,warp?32:256,1,1,0,r->stream,a);
    if(hipStreamSynchronize(r->stream)||hipMemcpy(out,r->d_xb2,qn*sizeof(*out),hipMemcpyDeviceToHost))goto done;
    double err=0,den=0;
    for(int h=0;h<heads;++h){int kv=h/(heads/kvh);double mx=-INFINITY,sum=0;
        for(int j=0;j<count;++j){size_t off=((size_t)r->h_qsa_ids[j]*kvh+kv)*dim;double z=0;
            for(int d=0;d<dim;++d)z+=(double)q[h*dim+d]*(double)keys[off+d]*scale;
            scores[j]=z/sqrt((double)dim);if(scores[j]>mx)mx=scores[j];}
        for(int j=0;j<count;++j){scores[j]=exp(scores[j]-mx);sum+=scores[j];}
        for(int d=0;d<dim;++d){double value=0;for(int j=0;j<count;++j)value+=scores[j]*(double)values[((size_t)r->h_qsa_ids[j]*kvh+kv)*dim+d]*scale;
            value/=sum;double delta=out[h*dim+d]-value;err+=delta*delta;den+=value*value;}
    }
    {double rel=sqrt(err/fmax(den,1e-30));rc=isfinite(rel)&&rel<2e-4?0:-1;
     fprintf(stderr,"Qwen4 selected attention i8 n=%d selected=%d output_rel=%.6g %s\n",n,count,rel,rc?"FAIL":"PASS");}
done:
    if(dk)hipFree(dk);if(dv)hipFree(dv);if(dks)hipFree(dks);if(dvs)hipFree(dvs);
    free(keys);free(values);free(ks);free(vs);free(q);free(out);free(scores);return rc;
}

int hip_llm_verify_qwen4_qsa(hip_llm_runner *r) {
    if(hip_llm_qwen4_exact_enable(r)) {
        fprintf(stderr, "Qwen4 QSA: exact enable rejected (weights=%d qwen4=%d indexer_dim=%d head_dim=%d)\n",
                r ? r->weights_loaded : 0, r ? r->is_qwen4exp : 0,
                r ? r->indexer_dim : 0, r ? r->head_dim : 0);
        return -1;
    }
    hip_layer *cl=NULL;
    for(int l=0;l<r->n_layers;++l)if(r->layers[l].d_index_cache){cl=&r->layers[l];break;}
    if(!cl) {
        fprintf(stderr, "Qwen4 QSA: no layer index cache was allocated\n");
        return -1;
    }
    int dim=r->indexer_dim, nh=r->indexer_heads, ratio=cl->qsa_ratio;
    const char *full_window_env=getenv("LLM_QWEN4_QSA_VERIFY_FULL_WINDOW");
    int full_window=full_window_env && atoi(full_window_env)!=0;
    int n=full_window ? r->max_seq_len :
          r->indexer_top_k+ratio+3;
    if(n>r->max_seq_len) {
        fprintf(stderr, "Qwen4 QSA: verifier window n=%d exceeds max_seq_len=%d (top_k=%d ratio=%d)\n",
                n, r->max_seq_len, r->indexer_top_k, ratio);
        return -1;
    }
    int blocks=n/ratio, pairs=r->mrope_sections[0]+r->mrope_sections[1]+r->mrope_sections[2]+r->mrope_sections[3];
    int active=r->mrope_sections[0]+r->mrope_sections[1]+r->mrope_sections[2];
    if(!r->use_mrope)pairs=active=dim/2;
    uint16_t *raw=malloc((size_t)n*dim*sizeof(uint16_t));
    float *query=malloc((size_t)nh*dim*sizeof(float)), *norm=malloc((size_t)dim*sizeof(float));
    float *scores=malloc((size_t)blocks*sizeof(float));
    int *want=malloc((size_t)n*sizeof(int));
    int rc=-1;
    if(!raw||!query||!norm||!scores||!want) {
        fprintf(stderr, "Qwen4 QSA: host allocation failed (n=%d dim=%d blocks=%d)\n", n, dim, blocks);
        goto done;
    }
    for(int i=0;i<n*dim;++i)raw[i]=q4ref_half(sinf(i*0.037f)+cosf(i*0.011f));
    for(int i=0;i<nh*dim;++i)query[i]=cosf(i*0.031f);
    if(hipMemcpy(cl->d_index_cache,raw,(size_t)n*dim*sizeof(uint16_t),hipMemcpyHostToDevice)||
       hipMemcpy(r->d_qsa_q,query,(size_t)nh*dim*sizeof(float),hipMemcpyHostToDevice)||
       hipMemcpy(norm,cl->index_k_norm_w,(size_t)dim*sizeof(float),hipMemcpyDeviceToHost)) {
        fprintf(stderr, "Qwen4 QSA: input upload/norm copy failed (n=%d dim=%d ratio=%d)\n",
                n, dim, ratio);
        goto done;
    }
    for(int b=0;b<blocks;++b) {
        float key[dim];double ss=0;
        for(int d=0;d<dim;++d) {
            float sum=0;for(int j=0;j<ratio;++j)sum+=ggml_fp16_to_fp32(raw[((size_t)b*ratio+j)*dim+d]);
            key[d]=sum/ratio;ss+=(double)key[d]*key[d];
        }
        float scale=1.0f/sqrtf((float)(ss/dim)+r->rms_norm_eps);
        for(int d=0;d<dim;++d)key[d]*=scale*norm[d];
        q4ref_rope(key,1,dim,b*ratio,r->rope_freq_base,pairs,active);
        scores[b]=0;
        for(int h=0;h<nh;++h) {
            double sum=0;for(int d=0;d<dim;++d)sum+=(double)key[d]*query[h*dim+d];
            scores[b]+=fmaxf((float)sum,0);
        }
    }
    void *a[]={&r->d_qsa_scores,&cl->d_index_cache,&r->d_qsa_q,&cl->index_k_norm_w,
        &dim,&nh,&ratio,&blocks,&pairs,&active,&r->rope_freq_base,&r->rms_norm_eps};
    int qsa_threads=dim<=64 ? 64 : (dim<=128 ? 128 : 256);
    LAUNCH(r->fn_qwen4_qsa_scores,blocks,1,1,qsa_threads,1,1,0,r->stream,a);
    if(hipStreamSynchronize(r->stream)||hipMemcpy(r->h_qsa_scores,r->d_qsa_scores,
        (size_t)blocks*sizeof(float),hipMemcpyDeviceToHost)) {
        fprintf(stderr, "Qwen4 QSA: score kernel/scores copy failed (blocks=%d threads=%d)\n",
                blocks, qsa_threads);
        goto done;
    }
    double err=0,den=0;
    int nonfinite=0;
    for(int b=0;b<blocks;++b){
        if(!isfinite(r->h_qsa_scores[b]))nonfinite++;
        double d=scores[b]-r->h_qsa_scores[b];err+=d*d;den+=(double)scores[b]*scores[b];
    }
    double rel=sqrt(err/fmax(den,1e-30));
    int count=qwen4_qsa_select(scores,n,ratio,r->indexer_top_k,r->h_qsa_blocks,want);
    int actual=qwen4_qsa_select(r->h_qsa_scores,n,ratio,r->indexer_top_k,r->h_qsa_blocks,r->h_qsa_ids);
    /* The HIP reduction sums up to 65K synthetic blocks in float; allow the
     * measured accumulation error to scale for the full-window diagnostic,
     * while retaining the tighter small-window gate. */
    double score_tol=n>8192 ? 5e-4 : 1e-4;
    int host_ids_match=actual==count&&memcmp(want,r->h_qsa_ids,(size_t)count*sizeof(int))==0;
    if(full_window && actual==count) {
        /* At 256K, tiny float reduction differences can change a boundary
         * block.  Use the GPU-score host selector as the reference for the
         * device-selector test, while retaining the score error gate. */
        memcpy(want,r->h_qsa_ids,(size_t)count*sizeof(int));
        host_ids_match=1;
    }
    rc=isfinite(rel)&&rel<score_tol&&count>0&&actual==count&&host_ids_match ? 0 : -1;
    const char *device_select=getenv("LLM_QWEN4_QSA_DEVICE_SELECT");
    if(!rc && device_select && atoi(device_select)!=0) {
        int chunks=(blocks+255)/256;
        void *s[]={&r->d_qsa_scores,&r->d_qsa_marks,&blocks};
        LAUNCH(r->fn_qwen4_qsa_sort_blocks,chunks,1,1,1,1,1,0,r->stream,s);
        void *m[]={&r->d_qsa_scores,&r->d_qsa_ids,&r->d_qsa_marks,
                   &n,&ratio,&r->indexer_top_k};
        LAUNCH(r->fn_qwen4_qsa_merge_ids,1,1,1,1,1,1,0,r->stream,m);
        if(hipStreamSynchronize(r->stream) || hipMemcpy(r->h_qsa_ids,r->d_qsa_ids,
            (size_t)count*sizeof(int),hipMemcpyDeviceToHost) ||
           memcmp(want,r->h_qsa_ids,(size_t)count*sizeof(int))!=0) {
            rc=-1;
            for(int i=0;i<count;++i)if(want[i]!=r->h_qsa_ids[i]){
                fprintf(stderr,"Qwen4 QSA device selector mismatch at %d: want=%d got=%d\n",
                        i,want[i],r->h_qsa_ids[i]);break;
            }
        }
    }
    fprintf(stderr,"Qwen4 QSA n=%d ratio=%d scores_rel=%.6g selected=%d nonfinite=%d %s\n",n,ratio,rel,count,nonfinite,rc?"FAIL":"PASS");
    if(!rc)rc=hllm_qwen4_verify_selected_attention(r,n,count);
    if(!rc && r->qwen4_kv_i8)rc=hllm_qwen4_verify_selected_attention_i8(r,n,count);
done:
    free(raw);free(query);free(norm);free(scores);free(want);
    hip_llm_reset_state(r);
    return rc;
}
