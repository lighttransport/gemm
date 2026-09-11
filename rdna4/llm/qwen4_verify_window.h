/* Layer-major exact verifier. Reuses scalar arithmetic and stores recurrent
 * checkpoints on device, so rejection selects a prefix without trunk replay. */
static int hllm_qwen4_window_alloc(hip_llm_runner *r, int rows) {
    if(r->qwen4_verify_capacity>=rows)return 0;
    size_t hc=(size_t)r->hc_count*r->n_embd*sizeof(float);
    size_t conv=(size_t)(r->ssm_conv_kernel-1)*r->ssm_qkv_dim*sizeof(float);
    size_t rec=(size_t)r->ssm_dt_rank*r->ssm_d_state*r->ssm_d_state*sizeof(float);
    size_t ple=(size_t)(r->ple_conv_kernel-1)*r->ple_ngram*hc;
    /* Grow once for the configured K. A failed growth invalidates the window
     * capacity; scalar mode remains available after the request is reset. */
    r->qwen4_verify_capacity=0;
#define WINDOW_ALLOC(ptr,bytes) do { \
        if(ptr) {hipFree(ptr);} ptr=NULL; if(hipMalloc(&(ptr),(bytes)))return -1; \
    } while(0)
    WINDOW_ALLOC(r->d_verify_hc,(size_t)rows*hc);
    WINDOW_ALLOC(r->d_verify_base_hc,hc);
    if(r->d_ple_conv_state)WINDOW_ALLOC(r->d_verify_ple,(size_t)rows*ple);
    for(int l=0;l<r->n_layers;++l)if(r->layers[l].is_ssm) {
        WINDOW_ALLOC(r->d_verify_conv[l],(size_t)rows*conv);
        WINDOW_ALLOC(r->d_verify_rec[l],(size_t)rows*rec);
    }
#undef WINDOW_ALLOC
    r->qwen4_verify_capacity=rows;
    return 0;
}

static void hllm_qwen4_grouped_cache_snapshot_clear(hip_llm_runner *r) {
    for (int l = 0; l < r->n_layers && l < 128; ++l) {
        free(r->qwen4_grouped_cache_snapshot[l]);
        r->qwen4_grouped_cache_snapshot[l] = NULL;
        r->qwen4_grouped_cache_next[l] = 0;
    }
}

/* Save the recurrent target state before an experimental grouped window.  KV
 * entries written for rejected rows are harmless because subsequent attention
 * only consumes positions up to the committed cursor; SSM/PLE/HC state is
 * the part that must be transactional. */
static int hllm_qwen4_grouped_save_base(hip_llm_runner *r) {
    hllm_qwen4_grouped_cache_snapshot_clear(r);
    size_t hc=(size_t)r->hc_count*r->n_embd*sizeof(float);
    if (hipMemcpyAsync(r->d_verify_base_hc,r->d_hc,hc,
                       hipMemcpyDeviceToDevice,r->stream)!=hipSuccess)return -1;
    size_t conv=(size_t)(r->ssm_conv_kernel-1)*r->ssm_qkv_dim*sizeof(float);
    size_t rec=(size_t)r->ssm_dt_rank*r->ssm_d_state*r->ssm_d_state*sizeof(float);
    for(int l=0;l<r->n_layers;++l)if(r->layers[l].is_ssm) {
        if(hipMemcpyAsync(r->d_verify_conv[l],r->layers[l].d_conv_state,conv,
                          hipMemcpyDeviceToDevice,r->stream)!=hipSuccess)return -1;
        if(hipMemcpyAsync(r->d_verify_rec[l],r->layers[l].d_recurrent_state,rec,
                          hipMemcpyDeviceToDevice,r->stream)!=hipSuccess)return -1;
    }
    if(r->d_ple_conv_state) {
        size_t ple=(size_t)(r->ple_conv_kernel-1)*r->ple_ngram*hc;
        if(hipMemcpyAsync(r->d_verify_ple,r->d_ple_conv_state,ple,
                          hipMemcpyDeviceToDevice,r->stream)!=hipSuccess)return -1;
    }
    for (int l = 0; l < r->n_layers && l < 128; ++l) {
        hip_layer *cl = &r->layers[l];
        if (!cl->moe_cache_ids || cl->moe_cache_slots <= 0) continue;
        r->qwen4_grouped_cache_snapshot[l] = malloc(
            (size_t)cl->moe_cache_slots * sizeof(int));
        if (!r->qwen4_grouped_cache_snapshot[l]) return -1;
        memcpy(r->qwen4_grouped_cache_snapshot[l], cl->moe_cache_ids,
               (size_t)cl->moe_cache_slots * sizeof(int));
        r->qwen4_grouped_cache_next[l] = cl->moe_cache_next;
    }
    return 0;
}

static void hllm_qwen4_grouped_restore_caches(hip_llm_runner *r) {
    /* Grouped prefill may have replaced cache slots before a rejection was
     * discovered. Preserve slots whose identity and payload were unchanged;
     * invalidate only overwritten slots so scalar fallback does not pair an
     * expert id with another expert's bytes while retaining useful residency. */
    for(int l=0;l<r->n_layers && l<128;++l) {
        hip_layer *cl=&r->layers[l];
        if(!cl->moe_cache_ids)continue;
        int *base = r->qwen4_grouped_cache_snapshot[l];
        if (base) {
            for(int s=0;s<cl->moe_cache_slots;++s) {
                if (cl->moe_cache_ids[s] != base[s]) {
                    cl->moe_cache_ids[s]=-1;
                    if(cl->moe_cache_age)cl->moe_cache_age[s]=0;
                    if(cl->moe_cache_freq)cl->moe_cache_freq[s]=0;
                }
            }
            cl->moe_cache_next = r->qwen4_grouped_cache_next[l];
        }
        if(cl->d_moe_cache_map) {
            int *map = malloc((size_t)r->n_experts*sizeof(int));
            if (map) {
                for (int e=0;e<r->n_experts;++e) map[e]=-1;
                for (int s=0;s<cl->moe_cache_slots;++s) {
                    int e=cl->moe_cache_ids[s];
                    if (e>=0 && e<r->n_experts) map[e]=s;
                }
                hipMemcpyAsync(cl->d_moe_cache_map,map,
                               (size_t)r->n_experts*sizeof(int),
                               hipMemcpyHostToDevice,r->stream);
                free(map);
            } else {
                hipMemsetAsync(cl->d_moe_cache_map,0xff,
                               (size_t)r->n_experts*sizeof(int),r->stream);
            }
        }
        cl->moe_pending_slot=-1;cl->moe_pending_expert=-1;
    }
    hllm_qwen4_grouped_cache_snapshot_clear(r);
}

static void hllm_qwen4_grouped_restore_base(hip_llm_runner *r,
                                             int32_t h0,int32_t h1) {
    /* The grouped MoE path may have queued host-to-device expert promotions.
     * Publish those copies before scalar fallback so its cache-map metadata
     * cannot observe a half-completed slot transition. */
    int old_decode=r->decode_mode;
    hip_llm_set_decode_mode(r,0);
    size_t hc=(size_t)r->hc_count*r->n_embd*sizeof(float);
    hipMemcpyAsync(r->d_hc,r->d_verify_base_hc,hc,hipMemcpyDeviceToDevice,r->stream);
    size_t conv=(size_t)(r->ssm_conv_kernel-1)*r->ssm_qkv_dim*sizeof(float);
    size_t rec=(size_t)r->ssm_dt_rank*r->ssm_d_state*r->ssm_d_state*sizeof(float);
    for(int l=0;l<r->n_layers;++l)if(r->layers[l].is_ssm) {
        hipMemcpyAsync(r->layers[l].d_conv_state,r->d_verify_conv[l],conv,
                       hipMemcpyDeviceToDevice,r->stream);
        hipMemcpyAsync(r->layers[l].d_recurrent_state,r->d_verify_rec[l],rec,
                       hipMemcpyDeviceToDevice,r->stream);
    }
    if(r->d_ple_conv_state) {
        size_t ple=(size_t)(r->ple_conv_kernel-1)*r->ple_ngram*hc;
        hipMemcpyAsync(r->d_ple_conv_state,r->d_verify_ple,ple,
                       hipMemcpyDeviceToDevice,r->stream);
    }
    hllm_qwen4_grouped_restore_caches(r);
    r->ple_history[0]=h0;r->ple_history[1]=h1;
    r->qwen4_grouped_window_rows=0;
    hipStreamSynchronize(r->stream);
    hip_llm_set_decode_mode(r,old_decode);
}

/* Grouped exact target verification.  This deliberately remains opt-in until
 * the transaction path has been exercised across a wider prompt corpus.  The
 * batched dispatcher retains qwen4_exact=1, so the validated host top-k and
 * exact expert-miss policy remain active; only the row-wise GEMM scheduling is
 * changed.  A single batched lm-head GEMM replaces one 248k-row matvec per
 * verifier position. */
static int hllm_qwen4_window_forward_grouped(hip_llm_runner *r,
                                             const int32_t *tokens,int rows,
                                             int position,int32_t *predictions,
                                             int32_t h0,int32_t h1) {
    if(!r->d_x_batch || !r->d_qwen4_logits_batch || !r->d_moe_out_batch ||
       !r->moe_prefill_batched)return -1;
    if(hllm_qwen4_grouped_save_base(r))return -1;
    if(embed_tokens_batch(r,tokens,rows))return -1;
    r->qwen4_grouped_tokens=tokens;
    r->qwen4_grouped_history[0]=h0;
    r->qwen4_grouped_history[1]=h1;
    r->qwen4_grouped_verify=1;
    int old_batch=r->batch_path_ok;
    r->batch_path_ok=1;
    int rc=forward_block_batched_dense(r,rows,position,tokens);
    r->batch_path_ok=old_batch;
    r->qwen4_grouped_verify=0;
    r->qwen4_grouped_tokens=NULL;
    if(rc)return -1;
    /* d_x_batch contains the post-head-normalization hidden rows.  Convert the
     * output matrix once and run the existing one-vector argmax kernel over
     * each row; all copies are stream ordered and synchronize only once. */
    launch_pack_bf16_from_f32(r,r->d_hc_norm_batch_bf16,r->d_x_batch,
                              rows*r->n_embd);
    size_t hc_bytes=(size_t)r->hc_count*r->n_embd*sizeof(float);
    hipMemcpyAsync(r->d_verify_hc,r->d_hc_batch,
                   (size_t)rows*hc_bytes,hipMemcpyDeviceToDevice,r->stream);
    size_t conv_bytes=(size_t)(r->ssm_conv_kernel-1)*r->ssm_qkv_dim*sizeof(float);
    size_t rec_bytes=(size_t)r->ssm_dt_rank*r->ssm_d_state*r->ssm_d_state*sizeof(float);
    for(int l=0;l<r->n_layers;++l)if(r->layers[l].is_ssm) {
        hipMemcpyAsync((char *)r->d_verify_conv[l]+(size_t)(rows-1)*conv_bytes,
                       r->layers[l].d_conv_state,conv_bytes,
                       hipMemcpyDeviceToDevice,r->stream);
        hipMemcpyAsync((char *)r->d_verify_rec[l]+(size_t)(rows-1)*rec_bytes,
                       r->layers[l].d_recurrent_state,rec_bytes,
                       hipMemcpyDeviceToDevice,r->stream);
    }
    if(r->d_ple_conv_state) {
        size_t ple=(size_t)(r->ple_conv_kernel-1)*r->ple_ngram*hc_bytes;
        hipMemcpyAsync((char *)r->d_verify_ple+(size_t)(rows-1)*ple,
                       r->d_ple_conv_state,ple,hipMemcpyDeviceToDevice,r->stream);
    }
    const char *scalar_lm_env=getenv("LLM_QWEN4_GROUPED_SCALAR_LMHEAD");
    /* A 2048-column tiled lm-head turns a two-row window into ~122 tiny
     * hipBLASLt launches.  The existing Q8_0 matvec is substantially faster
     * for these small windows and preserves the exact row-wise argmax.  Keep
     * the tiled path available only as an explicit diagnostic override. */
    if(!scalar_lm_env || atoi(scalar_lm_env)!=0) {
        /* Qwen4's lm-head is padded Q8_0.  The native batched kernel uses the
         * same warp-per-row accumulation order as the scalar path, but covers
         * the whole verification window in one launch.  Keep the generic
         * per-row fallback for alternate output-head formats. */
        if(r->output_w_type==GGML_TYPE_Q8_0) {
            launch_matmul_q8_batch_f32(r,r->d_qwen4_logits_batch,
                                       r->d_output_w,r->d_x_batch,rows,
                                       r->n_vocab,r->n_embd);
        } else for(int i=0;i<rows;++i) {
            float *row=(float *)r->d_x_batch+(size_t)i*r->n_embd;
            float *logits=(float *)r->d_qwen4_logits_batch+(size_t)i*r->n_vocab;
            launch_matvec_auto(r,logits,r->d_output_w,row,r->n_vocab,
                               r->n_embd,r->output_w_type);
        }
        /* The per-row matvecs remain exact, but reduce their completed logits
         * with one deterministic block per row instead of one launch/copy per
         * row. This only changes launch scheduling, not tie-breaking. */
        {
            int n = r->n_vocab;
            void *a[]={&r->d_qwen4_logits_batch,&n,&rows,&r->d_qwen4_nextn_tokens};
            LAUNCH(r->fn_qwen4_argmax_batch,rows,1,1,256,1,1,0,r->stream,a);
        }
        if(hipStreamSynchronize(r->stream)!=hipSuccess ||
           hipMemcpy(predictions,r->d_qwen4_nextn_tokens,(size_t)rows*sizeof(*predictions),
                     hipMemcpyDeviceToHost)!=hipSuccess)return -1;
        r->qwen4_grouped_window_rows=rows;
        return 0;
    }
    /* The Qwen4 lm-head is Q8_0 and too large for the shared dequant staging
     * buffer.  Dequantize 2048 output rows at a time, run a batched GEMM, and
     * scatter the compact tile into the full row-major logits matrix. */
    if(r->output_w_type!=GGML_TYPE_Q8_0)return -1;
    /* upload_q8_0_raw pads each 32-value block to 36 bytes (2-byte d,
     * 2-byte alignment pad, 32 int8 values). */
    const int tile=2048, row_bytes=(r->n_embd/32)*36;
    for(int col=0;col<r->n_vocab;col+=tile) {
        int cc=r->n_vocab-col;if(cc>tile)cc=tile;
        void *raw=(unsigned char *)r->d_output_w+(size_t)col*row_bytes;
        if(launch_dequant_q8_0_to_bf16(r,r->d_qwen4_lm_wbuf,raw,cc,r->n_embd)!=0)return -1;
        if(gemm_run_bf16_w(r,r->d_qwen4_lm_outbuf,r->d_qwen4_lm_wbuf,
                           r->d_hc_norm_batch_bf16,rows,cc,r->n_embd,
                           r->stream)!=0)return -1;
        for(int i=0;i<rows;++i)
            hipMemcpyAsync((float *)r->d_qwen4_logits_batch+(size_t)i*r->n_vocab+col,
                           (float *)r->d_qwen4_lm_outbuf+(size_t)i*cc,
                           (size_t)cc*sizeof(float),hipMemcpyDeviceToDevice,r->stream);
    }
    for(int i=0;i<rows;++i) {
        void *a[]={&(r->d_qwen4_logits_batch),&r->n_vocab,&r->d_moe_idx};
        /* The argmax kernel takes a base pointer, so pass the row address via a
         * temporary device pointer-sized argument. */
        float *row=(float *)r->d_qwen4_logits_batch+(size_t)i*r->n_vocab;
        a[0]=&row;
        LAUNCH(r->fn_qwen4_argmax,1,1,1,256,1,1,0,r->stream,a);
        hipMemcpyAsync((int *)r->d_qwen4_nextn_tokens+i,r->d_moe_idx,
                       sizeof(int),hipMemcpyDeviceToDevice,r->stream);
    }
    if(hipStreamSynchronize(r->stream)!=hipSuccess ||
       hipMemcpy(predictions,r->d_qwen4_nextn_tokens,(size_t)rows*sizeof(*predictions),
                 hipMemcpyDeviceToHost)!=hipSuccess)return -1;
    r->qwen4_grouped_window_rows=rows;
    return 0;
}

static int hllm_qwen4_window_forward(hip_llm_runner *r, const int32_t *tokens,
                                    int rows, int position, int32_t *predictions) {
    if(rows<1 || rows>33 || position+rows>r->max_seq_len || hllm_qwen4_window_alloc(r,rows))return -1;
    int ne=r->n_embd,ns=r->hc_count;
    size_t hc=(size_t)ne*ns*sizeof(float);
    size_t conv=(size_t)(r->ssm_conv_kernel-1)*r->ssm_qkv_dim*sizeof(float);
    size_t rec=(size_t)r->ssm_dt_rank*r->ssm_d_state*r->ssm_d_state*sizeof(float);
    size_t ple=(size_t)(r->ple_conv_kernel-1)*r->ple_ngram*hc;
    int32_t history[2]={r->ple_history[0],r->ple_history[1]};
    int positions[33];for(int i=0;i<rows;++i)positions[i]=position+i;
    const char *grouped_env=getenv("LLM_QWEN4_GROUPED_VERIFY");
    /* A grouped window pays snapshot/restore and batched-plan setup costs.
     * For one or two rows this is consistently slower than the scalar path,
     * especially after a rejection, so keep those tiny windows scalar.  A
     * diagnostic override can lower the threshold when measuring plan setup. */
    int grouped_min_rows=3;
    const char *grouped_min_env=getenv("LLM_QWEN4_GROUPED_MIN_ROWS");
    if(grouped_min_env && *grouped_min_env) {
        int v=atoi(grouped_min_env);
        if(v>=1 && v<=33) grouped_min_rows=v;
    }
    int grouped = !r->qwen4_grouped_force_scalar && grouped_env &&
                  atoi(grouped_env)!=0 && r->batch_max >= rows &&
                  r->moe_prefill_batched && rows >= grouped_min_rows;
    if(grouped) {
        for(int i=0;i<rows;++i)
            if(tokens[i]<0 || tokens[i]>=r->n_vocab)return -1;
        int grouped_rc=hllm_qwen4_window_forward_grouped(r,tokens,rows,position,
                                             predictions,history[0],history[1]);
        if (getenv("LLM_QWEN4_GROUPED_TRACE"))
            fprintf(stderr,"qwen4 grouped attempt: rows=%d batch=%d moe_batch=%d rc=%d\n",
                    rows,r->batch_max,r->moe_prefill_batched,grouped_rc);
        if(grouped_rc==0)
            return 0;
        /* A failed grouped launch must not leak a partially advanced recurrent
         * state into scalar verification. */
        hllm_qwen4_grouped_restore_base(r,history[0],history[1]);
        hipStreamSynchronize(r->stream);
    }
    void *target_hc=r->d_hc;
    hipMemcpyAsync(r->d_verify_base_hc,target_hc,hc,hipMemcpyDeviceToDevice,r->stream);
    for(int i=0;i<rows;++i) {
        int token=tokens[i];
        if(token<0 || token>=r->n_vocab)return -1;
        if(r->token_embd_type==GGML_TYPE_Q8_0)launch_embed_q8_0(r,r->d_x,r->d_token_embd,token,ne);
        else if(r->token_embd_type==GGML_TYPE_Q4_0)launch_embed_q4_0(r,r->d_x,r->d_token_embd,token,ne);
        else if(r->token_embd_type==GGML_TYPE_Q2_K)launch_embed_q2_K(r,r->d_x,r->d_token_embd,token,ne);
        else launch_embed(r,r->d_x,r->d_token_embd,token,ne);
        void *dst=(char *)r->d_verify_hc+(size_t)i*hc;
        void *a[]={&dst,&r->d_x,&ne,&ns};
        LAUNCH(r->fn_hc_repeat_f32,(ne*ns+255)/256,1,1,256,1,1,0,r->stream,a);
    }
    for(int l=0;l<r->n_layers;++l) {
        hip_layer *cl=&r->layers[l];
        for(int i=0;i<rows;++i) {
            r->cur_position=position+i;r->ple_token_id=tokens[i];
            r->ple_history[0]=i<2 ? history[i] : tokens[i-2];
            r->ple_history[1]=i==0 ? history[1] : tokens[i-1];
            r->d_hc=(char *)r->d_verify_hc+(size_t)i*hc;
            hipMemcpyAsync(r->d_position,&positions[i],sizeof(int),hipMemcpyHostToDevice,r->stream);
            forward_layer_state(r,cl,l,r->d_key_cache[l],r->d_value_cache[l],2);
            if(r->qwen4_forward_error) {r->d_hc=target_hc;return -1;}
            if(cl->is_ssm) {
                hipMemcpyAsync((char *)r->d_verify_conv[l]+(size_t)i*conv,cl->d_conv_state,conv,hipMemcpyDeviceToDevice,r->stream);
                hipMemcpyAsync((char *)r->d_verify_rec[l]+(size_t)i*rec,cl->d_recurrent_state,rec,hipMemcpyDeviceToDevice,r->stream);
            }
            if(l==1 && r->d_ple_conv_state)
                hipMemcpyAsync((char *)r->d_verify_ple+(size_t)i*ple,r->d_ple_conv_state,ple,hipMemcpyDeviceToDevice,r->stream);
        }
    }
    for(int i=0;i<rows;++i) {
        r->d_hc=(char *)r->d_verify_hc+(size_t)i*hc;
        forward_hc_mix(r,r->hc_head_norm_w,r->hc_head_down_w,r->hc_head_down_type,
                       r->hc_head_up_w,r->hc_head_up_type,NULL,0,r->d_x,-1);
        launch_matvec_auto(r,r->d_logits,r->d_output_w,r->d_x,r->n_vocab,ne,r->output_w_type);
        void *a[]={&r->d_logits,&r->n_vocab,&r->d_moe_idx};
        LAUNCH(r->fn_qwen4_argmax,1,1,1,256,1,1,0,r->stream,a);
        hipMemcpyAsync((int *)r->d_qwen4_nextn_tokens+i,r->d_moe_idx,
                       sizeof(int),hipMemcpyDeviceToDevice,r->stream);
    }
    r->d_hc=target_hc;
    r->ple_history[0]=history[0];r->ple_history[1]=history[1];
    if(hipStreamSynchronize(r->stream)!=hipSuccess ||
       hipMemcpy(predictions,r->d_qwen4_nextn_tokens,(size_t)rows*sizeof(*predictions),
                 hipMemcpyDeviceToHost)!=hipSuccess)return -1;
    return 0;
}

static int hllm_qwen4_window_commit(hip_llm_runner *r, const int32_t *tokens,
                                   int processed, int position,
                                   int replay_sidecar) {
    if(processed<1)return -1;
    size_t hc=(size_t)r->hc_count*r->n_embd*sizeof(float);
    size_t conv=(size_t)(r->ssm_conv_kernel-1)*r->ssm_qkv_dim*sizeof(float);
    size_t rec=(size_t)r->ssm_dt_rank*r->ssm_d_state*r->ssm_d_state*sizeof(float);
    size_t ple=(size_t)(r->ple_conv_kernel-1)*r->ple_ngram*hc;
    int last=processed-1;
    for(int l=0;l<r->n_layers;++l)if(r->layers[l].is_ssm) {
        hipMemcpyAsync(r->layers[l].d_conv_state,(char *)r->d_verify_conv[l]+(size_t)last*conv,conv,hipMemcpyDeviceToDevice,r->stream);
        hipMemcpyAsync(r->layers[l].d_recurrent_state,(char *)r->d_verify_rec[l]+(size_t)last*rec,rec,hipMemcpyDeviceToDevice,r->stream);
    }
    if(r->d_ple_conv_state)hipMemcpyAsync(r->d_ple_conv_state,(char *)r->d_verify_ple+(size_t)last*ple,ple,hipMemcpyDeviceToDevice,r->stream);
    if (replay_sidecar) {
        /* Forced-draft transactions replace the generated tokens, so their
         * cached sidecar checkpoints do not describe this prefix.  Retain
         * the reference replay path for those checks. */
        for(int i=0;i<processed;++i) {
            const void *prior=i ? (char *)r->d_verify_hc+(size_t)(i-1)*hc : r->d_verify_base_hc;
            if(hllm_qwen4_nextn_forward(r,tokens[i],NULL,position+i,prior))return -1;
        }
    } else {
        /* Draft generation already advanced NextN through the accepted draft
         * prefix.  Resume at that checkpoint and process only the final
         * emitted token; replaying all accepted tokens doubled sidecar work. */
        int sidecar_index=processed-1;
        const void *prior=(const char *)r->d_qwen4_nextn_hc_checkpoints+
                          (size_t)sidecar_index*hc;
        if(hllm_qwen4_nextn_forward(r,tokens[sidecar_index],NULL,
                                    position+sidecar_index,prior))return -1;
    }
    hipMemcpyAsync(r->d_hc,(char *)r->d_verify_hc+(size_t)last*hc,hc,hipMemcpyDeviceToDevice,r->stream);
    r->ple_history[0]=processed==1 ? r->ple_history[1] : tokens[processed-2];
    r->ple_history[1]=tokens[last];r->cur_position=position+last;
    hipMemcpyAsync(r->d_position,&r->cur_position,sizeof(int),hipMemcpyHostToDevice,r->stream);
    /* Restore the target distribution after teacher-forcing the draft state. */
    forward_hc_mix(r,r->hc_head_norm_w,r->hc_head_down_w,r->hc_head_down_type,
                   r->hc_head_up_w,r->hc_head_up_type,NULL,0,r->d_x,-1);
    launch_matvec_auto(r,r->d_logits,r->d_output_w,r->d_x,r->n_vocab,r->n_embd,r->output_w_type);
    hipMemcpyAsync(r->h_output,r->d_logits,(size_t)r->n_vocab*sizeof(float),hipMemcpyDeviceToHost,r->stream);
    r->qwen4_grouped_window_rows=0;
    return hipStreamSynchronize(r->stream) ? -1 : 0;
}
