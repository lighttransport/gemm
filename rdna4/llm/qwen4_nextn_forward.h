/* Included after the scalar HIP layer executor. All scratch is stream ordered. */
static int hllm_qwen4_nextn_forward(hip_llm_runner *r, int32_t token,
                                   const int *device_token, int position,
                                   const void *hidden) {
    if (!r || !r->qwen4_nextn_fusion_loaded || !hidden || token < 0 ||
        (!device_token && token >= r->n_vocab) ||
        position < 0 || position >= r->max_seq_len) return -1;
    int ne = r->n_embd, ns = r->hc_count, old_position = r->cur_position;
    float eps = r->rms_norm_eps;
    void *target_hc = r->d_hc;
    if (r->qwen4_nextn_start < 0) r->qwen4_nextn_start = position;
    if (hipMemcpyAsync(r->d_position, &position, sizeof(position), hipMemcpyHostToDevice, r->stream)) return -1;
    if (device_token) {
        if (r->token_embd_type == GGML_TYPE_Q8_0)
            launch_embed_q8_0_devtoken(r, r->d_qwen4_nextn_token,
                                       r->d_token_embd, device_token, ne);
        else if (r->token_embd_type == GGML_TYPE_Q4_0)
            launch_embed_q4_0_devtoken(r, r->d_qwen4_nextn_token,
                                       r->d_token_embd, device_token, ne);
        else if (r->token_embd_type == GGML_TYPE_Q2_K)
            launch_embed_q2_K_devtoken(r, r->d_qwen4_nextn_token,
                                       r->d_token_embd, device_token, ne);
        else
            launch_embed_devtoken(r, r->d_qwen4_nextn_token,
                                  r->d_token_embd, device_token, ne);
    } else if (r->token_embd_type == GGML_TYPE_Q8_0)
        launch_embed_q8_0(r, r->d_qwen4_nextn_token, r->d_token_embd, token, ne);
    else if (r->token_embd_type == GGML_TYPE_Q4_0)
        launch_embed_q4_0(r, r->d_qwen4_nextn_token, r->d_token_embd, token, ne);
    else if (r->token_embd_type == GGML_TYPE_Q2_K)
        launch_embed_q2_K(r, r->d_qwen4_nextn_token, r->d_token_embd, token, ne);
    else
        launch_embed(r, r->d_qwen4_nextn_token, r->d_token_embd, token, ne);
    launch_rmsnorm(r, r->d_qwen4_nextn_token, r->d_qwen4_nextn_token,
                   r->qwen4_nextn_enorm_w, ne, eps);
    void *a[] = { &r->d_hc_norm, &hidden, &r->qwen4_nextn_hnorm_w, &ne, &ns, &eps };
    LAUNCH(r->fn_hc_norm_f32, ns, 1, 1, 256, 1, 1, 256*sizeof(float), r->stream, a);
    for (int s = 0; s < ns; ++s) {
        float *f = (float *)r->d_qwen4_nextn_fusion + (size_t)s*2*ne;
        hipMemcpyAsync(f, r->d_qwen4_nextn_token, (size_t)ne*sizeof(float), hipMemcpyDeviceToDevice, r->stream);
        hipMemcpyAsync(f+ne, (float *)r->d_hc_norm+(size_t)s*ne, (size_t)ne*sizeof(float), hipMemcpyDeviceToDevice, r->stream);
        launch_matvec_auto(r, (float *)r->d_qwen4_nextn_hc+(size_t)s*ne,
                           r->qwen4_nextn_eh_w, f, ne, 2*ne, r->qwen4_nextn_eh_type);
    }
    r->d_hc = r->d_qwen4_nextn_hc;
    r->qwen4_nextn_active = 1;
    forward_layer_state(r, r->qwen4_nextn_layer, r->n_layers,
                        r->d_qwen4_nextn_key_cache, r->d_qwen4_nextn_value_cache, 0);
    forward_hc_mix(r, r->qwen4_nextn_hc_head_norm_w, r->qwen4_nextn_hc_head_down_w,
                   r->qwen4_nextn_hc_head_down_type, r->qwen4_nextn_hc_head_up_w,
                   r->qwen4_nextn_hc_head_up_type, NULL, 0, r->d_x, -1);
    launch_matvec_auto(r, r->d_logits, r->d_output_w, r->d_x, r->n_vocab, ne, r->output_w_type);
    r->d_hc = target_hc;
    r->qwen4_nextn_active = 0;
    /* The MTP draft loop writes the next absolute position at the start of
     * every step, and target verification writes its own position before
     * consuming the state. Avoid a redundant 4-byte H2D restore between
     * draft steps; retain it for the public/non-MTP path. */
    if (!r->qwen4_mtp_draft_active)
        hipMemcpyAsync(r->d_position, &old_position, sizeof(old_position),
                       hipMemcpyHostToDevice, r->stream);
    /* Trusted MTP immediately launches GPU argmax, whose D2H token copy is
     * already the required synchronization point.  Avoid a redundant stream
     * wait here; all work remains ordered on r->stream.  Keep the wait for
     * exact MTP and the public logits API, where the caller may consume state
     * without an argmax immediately following this function. */
    const char *trust_env = getenv("LLM_QWEN4_MTP_TRUST_DRAFT");
    if (r->qwen4_mtp_draft_active ||
        (r->qwen4_mtp_enabled && trust_env && atoi(trust_env) != 0))
        return r->qwen4_forward_error ? -1 : 0;
    return hipStreamSynchronize(r->stream) == hipSuccess && !r->qwen4_forward_error ? 0 : -1;
}

static int hllm_qwen4_argmax(hip_llm_runner *r) {
    int token = -1;
    void *args[] = { &r->d_logits, &r->n_vocab, &r->d_moe_idx };
    LAUNCH(r->fn_qwen4_argmax, 1, 1, 1, 256, 1, 1, 0, r->stream, args);
    if (hipMemcpyAsync(&token, r->d_moe_idx, sizeof(token), hipMemcpyDeviceToHost, r->stream) ||
        hipStreamSynchronize(r->stream)) return -1;
    return token;
}

int hip_llm_qwen4_nextn_logits(hip_llm_runner *r, int32_t token, int position,
                               const float *hidden, float *out_hidden, float *logits) {
    if (!r || !hidden || !out_hidden || !logits || !r->qwen4_nextn_fusion_loaded) return -1;
    size_t bytes = (size_t)r->hc_count*r->n_embd*sizeof(float);
    if (hipMemcpy(r->d_qwen4_nextn_hc, hidden, bytes, hipMemcpyHostToDevice) ||
        hllm_qwen4_nextn_forward(r, token, NULL, position, r->d_qwen4_nextn_hc) ||
        hipMemcpy(out_hidden, r->d_qwen4_nextn_hc, bytes, hipMemcpyDeviceToHost) ||
        hipMemcpy(logits, r->d_logits, (size_t)r->n_vocab*sizeof(float), hipMemcpyDeviceToHost)) return -1;
    return 0;
}

int hip_llm_qwen4_mtp_configure(hip_llm_runner *r, size_t cache_bytes, int draft) {
    if(!r || r->weights_loaded || draft<1 || draft>32)return -1;
    r->qwen4_nextn_cache_bytes=cache_bytes ? cache_bytes : (size_t)128<<20;
    r->qwen4_mtp_reserve=1;
    r->qwen4_mtp_max_draft=draft;
    /* Preserve an explicit --qwen4-batched-prefill request made before the
     * MTP sidecar is configured.  The old unconditional reset silently
     * discarded that request, so exact MTP could never exercise the parity-
     * checked grouped Q6_K prefill path.  Zero still means the conservative
     * scalar default; the loader may opt in through LLM_QWEN4_BATCH=1. */
    return 0;
}

int hip_llm_qwen4_mtp_enable(hip_llm_runner *r) {
    if (!r || !r->qwen4_nextn_fusion_loaded || r->n_layers != 48) return -1;
    const char *approx_env=getenv("LLM_QWEN4_MTP_APPROX");
    int approx=approx_env && atoi(approx_env)!=0;
    if (!approx && hip_llm_qwen4_exact_enable(r)) return -1;
    r->qwen4_mtp_enabled = 1;
    r->qwen4_mtp_approx = approx;
    r->qwen4_coding_profile = approx ? 1 : 0;
    /* Do not clear an explicit batched-prefill selection.  Exact MTP keeps
     * scalar verification, but its initial prompt prefill may use the grouped
     * path when the caller requested it (or set LLM_QWEN4_BATCH=1). */
    if (!approx) {
        /* Exact scalar MTP is transaction-safe with the fused router when
         * its cold experts still use the exact CPU miss path.  Keep these
         * scoped to MTP: ordinary exact decode retains its legacy defaults,
         * and either switch remains opt-out for diagnostics. */
        if (!getenv("LLM_QWEN4_EXACT_GPU_TOPK"))
            setenv("LLM_QWEN4_EXACT_GPU_TOPK", "1", 0);
        if (!getenv("LLM_QWEN4_EXACT_CPU_MISSES"))
            setenv("LLM_QWEN4_EXACT_CPU_MISSES", "1", 0);
    }
    /* The scalar target is the reference until verifier GEMM parity passes. */
    if (!approx) hip_llm_set_batched_path(r, 0);
    return 0;
}

int hip_llm_qwen4_mtp_set_verify(hip_llm_runner *r, int window) {
    if(!r || (window!=0 && window!=1))return -1;
    if(r->weights_loaded && window && !r->qwen4_mtp_verify_mode)return -1;
    r->qwen4_mtp_verify_mode=window;
    return 0;
}

static int hllm_qwen4_is_stop(int32_t token, const int32_t *ids, int n) {
    for (int i=0;i<n;++i) if(token==ids[i]) return 1;
    return 0;
}

static int hllm_qwen4_mtp_step(hip_llm_runner *r, int32_t anchor, int position,
                           int draft, int max_emit, const int32_t *stop_ids,
                           int n_stop, hip_llm_qwen4_mtp_result *out,
                           const int32_t *test_drafts) {
    if (!r || !out || !r->qwen4_mtp_enabled || draft<1 || draft>32 ||
        (r->qwen4_mtp_max_draft>0 && draft>r->qwen4_mtp_max_draft) ||
        max_emit<1 || position<0 || position>=r->max_seq_len ||
        anchor<0 || anchor>=r->n_vocab || n_stop<0 || (n_stop && !stop_ids)) return -1;
    memset(out,0,sizeof(*out)); out->pending=anchor;
    if (max_emit>r->max_seq_len-position) max_emit=r->max_seq_len-position;
    if (draft>max_emit-1) draft=max_emit-1;
    int32_t drafts[32], token=anchor;
    double start=hllm_monotonic_ms();
    const char *trust_env = getenv("LLM_QWEN4_MTP_TRUST_DRAFT");
    if (!hllm_qwen4_is_stop(anchor,stop_ids,n_stop)) {
        /* After the first trusted batch the target trunk is intentionally not
         * advanced. Continue the sidecar from its own recurrent HC state;
         * reusing target d_hc here restarts every batch from the same prefix
         * and produces repetitive, incoherent text. */
        /* d_hc is used as the serialized trusted-chain scratch buffer.  The
         * NextN output buffer cannot be fed back in place: hnorm reads the
         * input while the following step overwrites that same allocation. */
        const size_t hc_bytes = (size_t)r->hc_count*r->n_embd*sizeof(float);
        /* Scalar verification advances the target token-by-token and never
         * enters hllm_qwen4_window_commit(), so its sidecar HC checkpoints
         * are dead data.  The NextN forward writes its result to the
         * dedicated d_qwen4_nextn_hc buffer; it does not modify target d_hc.
         * Keep the sidecar chain in that buffer instead of copying the full
         * HC tensor into/out of target storage for every draft token. */
        const int need_checkpoints = r->qwen4_mtp_verify_mode != 0;
        const int device_chain = !getenv("LLM_QWEN4_MTP_HOST_CHAIN");
        /* Keep the sidecar HC at each prefix boundary.  Scalar verification
         * may reject a suffix; commit can then resume from the accepted
         * prefix instead of replaying every accepted sidecar token. */
        if (need_checkpoints &&
            hipMemcpyAsync(r->d_qwen4_nextn_hc_checkpoints, r->d_hc,
                           hc_bytes, hipMemcpyDeviceToDevice, r->stream))
            return -1;
        const void *hidden = r->d_hc;
        const int *device_token = NULL;
        int draft_error = 0;
        r->qwen4_mtp_draft_active = 1;
        for (int i=0;i<draft;++i) {
            if (hllm_qwen4_nextn_forward(r,token,device_token,position+i,hidden)) {
                draft_error = 1;
                break;
            }
            int *device_out = (int *)r->d_qwen4_nextn_tokens + i;
            if (device_chain) {
                void *argmax_args[] = { &r->d_logits, &r->n_vocab, &device_out };
                if (LAUNCH(r->fn_qwen4_argmax, 1, 1, 1, 256, 1, 1, 0,
                           r->stream, argmax_args) != hipSuccess) {
                    draft_error = 1;
                    break;
                }
            } else {
                token=hllm_qwen4_argmax(r);
                if (token<0) {
                    draft_error = 1;
                    break;
                }
                drafts[out->drafted++]=token;
            }
            if (need_checkpoints &&
                /* Slot 0 is the pre-anchor state; slot i+1 is the state
                 * after consuming the token at draft step i.  Keeping the
                 * initial boundary distinct lets commit resume the sidecar
                 * from the accepted-prefix boundary without replaying or
                 * skipping one token. */
                hipMemcpyAsync((char *)r->d_qwen4_nextn_hc_checkpoints +
                               (size_t)(i + 1) * hc_bytes,
                               r->d_qwen4_nextn_hc, hc_bytes,
                               hipMemcpyDeviceToDevice, r->stream)) {
                draft_error = 1;
                break;
            }
            hidden=r->d_qwen4_nextn_hc;
            if (device_chain) device_token=device_out;
            else if (hllm_qwen4_is_stop(token,stop_ids,n_stop)) break;
        }
        r->qwen4_mtp_draft_active = 0;
        /* Trusted mode carries the independent sidecar recurrent state into
         * the next request.  Publish it once per draft batch; exact mode
         * leaves target d_hc untouched and therefore needs no large copy. */
        if (r->qwen4_mtp_approx && trust_env && atoi(trust_env) != 0 &&
            (device_chain ? draft : out->drafted) > 0 &&
            hipMemcpyAsync(r->d_hc, r->d_qwen4_nextn_hc, hc_bytes,
                           hipMemcpyDeviceToDevice, r->stream))
            draft_error = 1;
        if (draft_error) return -1;
        if (device_chain) out->drafted=draft;
        if (device_chain && out->drafted &&
            (hipMemcpyAsync(drafts, r->d_qwen4_nextn_tokens,
                            (size_t)out->drafted*sizeof(*drafts),
                            hipMemcpyDeviceToHost, r->stream) != hipSuccess ||
             hipStreamSynchronize(r->stream) != hipSuccess)) return -1;
        for (int i=0; i<out->drafted; ++i) {
            if (hllm_qwen4_is_stop(drafts[i],stop_ids,n_stop)) {
                out->drafted=i+1;
                break;
            }
        }
    }
    out->draft_ms=hllm_monotonic_ms()-start;
    if(test_drafts)memcpy(drafts,test_drafts,(size_t)out->drafted*sizeof(int32_t));
    /* Explicit approximate ceiling mode: trust the sidecar's recurrent
     * predictions and keep advancing its independent state without replaying
     * the expensive target verifier.  This is never enabled by exact MTP and
     * is intentionally opt-in because it does not provide target parity. */
    if (trust_env && atoi(trust_env) != 0 && getenv("LLM_QWEN4_MTP_TRACE"))
        fprintf(stderr, "qwen4 mtp: approx=%d drafted=%d\n", r->qwen4_mtp_approx, out->drafted);
    if (r->qwen4_mtp_approx && trust_env && atoi(trust_env) != 0 && out->drafted > 0) {
        out->accepted = out->drafted;
        out->emitted = out->drafted + 1;
        out->processed = out->drafted;
        out->tokens[0] = anchor;
        memcpy(out->tokens + 1, drafts, (size_t)out->drafted * sizeof(int32_t));
        out->pending = drafts[out->drafted - 1];
        for (int i = 0; i < out->emitted; ++i) {
            if (hllm_qwen4_is_stop(out->tokens[i], stop_ids, n_stop)) {
                out->stopped = 1;
                out->accepted = i;
                out->emitted = i + 1;
                out->processed = i;
                out->pending = -1;
                break;
            }
        }
        out->verify_ms = 0.0;
        return 0;
    }
    start=hllm_monotonic_ms(); token=anchor;
    if(r->qwen4_mtp_verify_mode && out->drafted) {
        int32_t inputs[33],predictions[33];inputs[0]=anchor;
        memcpy(inputs+1,drafts,(size_t)out->drafted*sizeof(int32_t));
        /* Forced transaction vectors are the rollback oracle.  Do not let an
         * opt-in grouped experiment mutate that oracle: grouped state/commit
         * parity is still under audit, while the scalar window is the exact
         * reference used to validate every reject position. */
        const char *grouped_env = getenv("LLM_QWEN4_GROUPED_VERIFY");
        int forced_scalar_tx = test_drafts && grouped_env &&
                               atoi(grouped_env) != 0;
        if (forced_scalar_tx) r->qwen4_grouped_force_scalar = 1;
        int window_rc = hllm_qwen4_window_forward(r,inputs,out->drafted+1,
                                                   position,predictions);
        if (forced_scalar_tx) r->qwen4_grouped_force_scalar = 0;
        if(window_rc)return -1;
        while(out->accepted<out->drafted && predictions[out->accepted]==drafts[out->accepted])out->accepted++;
        if (r->qwen4_grouped_window_rows && getenv("LLM_QWEN4_GROUPED_TRACE"))
            fprintf(stderr,"qwen4 grouped: rows=%d pred0=%d draft0=%d accepted=%d/%d\n",
                    r->qwen4_grouped_window_rows,predictions[0],drafts[0],
                    out->accepted,out->drafted);
        out->emitted=out->processed=out->accepted+1;
        memcpy(out->tokens,inputs,(size_t)out->emitted*sizeof(int32_t));
        out->pending=predictions[out->emitted-1];
        if(hllm_qwen4_is_stop(inputs[out->emitted-1],stop_ids,n_stop)) {
            out->stopped=1;out->processed--;out->pending=-1;
        }
        /* A grouped window has already advanced the target through every row.
         * Commit it directly only when the whole draft was accepted.  A
         * partial rejection restores the transaction snapshot and reruns the
         * scalar window, preserving exact rollback semantics. */
        if (r->qwen4_grouped_window_rows) {
            /* Batched recurrent/MoE arithmetic is still under parity audit.
             * Never publish its state by default: even a grouped PASS can
             * have a different greedy sequence from scalar exact mode.  The
             * explicit commit knob is reserved for controlled parity runs. */
            const char *grouped_commit_env = getenv("LLM_QWEN4_GROUPED_COMMIT");
            int grouped_commit = grouped_commit_env &&
                                 atoi(grouped_commit_env) != 0;
            const char *partial_env = getenv("LLM_QWEN4_GROUPED_PARTIAL_COMMIT");
            int grouped_partial = partial_env && atoi(partial_env) != 0;
            int full_grouped_commit = grouped_commit &&
                                      out->accepted == out->drafted &&
                                      !out->stopped;
            const char *batch_conv_env = getenv("LLM_SSM_BATCH_CONV");
            const char *batch_recur_env = getenv("LLM_SSM_BATCH_RECURRENCE");
            int grouped_row_state_safe =
                !(batch_conv_env && atoi(batch_conv_env) != 0) &&
                !(batch_recur_env && atoi(batch_recur_env) != 0);
            int partial_grouped_commit = grouped_commit && grouped_partial &&
                                         out->accepted > 0 &&
                                         out->accepted < out->drafted &&
                                         grouped_row_state_safe &&
                                         !out->stopped;
            if (full_grouped_commit || partial_grouped_commit) {
                /* The grouped verifier filled the standard checkpoint slots;
                 * use the canonical transaction commit.  For a partial
                 * prefix, restore only expert-cache identities first; the
                 * per-row HC/SSM/PLE checkpoints then publish the accepted
                 * target state without replaying the rejected suffix. */
                if (partial_grouped_commit) {
                    hllm_qwen4_grouped_restore_caches(r);
                    hipStreamSynchronize(r->stream);
                }
                if(hllm_qwen4_window_commit(r,inputs,out->processed,position,
                                             test_drafts != NULL))return -1;
                hllm_qwen4_grouped_cache_snapshot_clear(r);
            } else {
                /* Restore the history captured before the grouped window.
                 * forward_block_batched_dense updates the live PLE history as
                 * it walks rows, so reading r->ple_history here would restore
                 * the rejected suffix rather than the transaction base. */
                int32_t h0=r->qwen4_grouped_history[0];
                int32_t h1=r->qwen4_grouped_history[1];
                hllm_qwen4_grouped_restore_base(r,h0,h1);
                r->qwen4_grouped_force_scalar=1;
                int32_t scalar_pred[33];
                /* The grouped result already identifies the first candidate
                 * mismatch.  Replaying the entire draft suffix here doubled
                 * target work on the common reject path (especially when the
                 * first draft token was wrong).  Replay only through that
                 * mismatch plus one row for the committed pending token.  If
                 * all replayed rows agree, committing that verified prefix is
                 * exact; the unverified suffix is simply redrafted next
                 * round. */
                /* Forced transaction vectors replace the generated drafts
                 * below, so the grouped acceptance count is not an upper
                 * bound for those tests; retain the full replay there. */
                int scalar_rows = test_drafts ? out->drafted + 1 : out->accepted + 2;
                if (scalar_rows > out->drafted + 1) scalar_rows = out->drafted + 1;
                if (hllm_qwen4_window_forward(r,inputs,scalar_rows,position,scalar_pred))return -1;
                r->qwen4_grouped_force_scalar=0;
                if (getenv("LLM_QWEN4_GROUPED_TRACE"))
                    fprintf(stderr,"qwen4 grouped fallback: rows=%d scalar_pred0=%d draft0=%d\n",
                            scalar_rows,scalar_pred[0],drafts[0]);
                out->accepted=0;
                int scalar_checked = scalar_rows - 1;
                while(out->accepted<scalar_checked &&
                      scalar_pred[out->accepted]==drafts[out->accepted])out->accepted++;
                out->emitted=out->processed=out->accepted+1;
                memcpy(out->tokens,inputs,(size_t)out->emitted*sizeof(int32_t));
                out->pending=scalar_pred[out->emitted-1];
                out->stopped=0;
                if (hllm_qwen4_is_stop(inputs[out->emitted-1],stop_ids,n_stop)) {
                    out->stopped=1; out->processed--; out->pending=-1;
                }
                if(hllm_qwen4_window_commit(r,inputs,out->processed,position,
                                             test_drafts != NULL))return -1;
            }
        } else if(hllm_qwen4_window_commit(r,inputs,out->processed,position,
                                           test_drafts != NULL))return -1;
        out->verify_ms=hllm_monotonic_ms()-start;
        return 0;
    }
    /* Verification consumes only accepted tokens. This scalar reference never
     * writes a rejected target suffix, including SSM/PLE recurrent state. */
    for (int i=0;;++i) {
        out->tokens[out->emitted++]=token;
        if(hllm_qwen4_is_stop(token,stop_ids,n_stop)) {out->stopped=1;out->pending=-1;break;}
        /* The target verifier must not run the sidecar prefix again.  Draft
         * generation already consumed NextN for this position; re-entering
         * it here doubles the expensive sidecar work per target token. */
        r->qwen4_mtp_verify_active = 1;
        int next=hip_llm_forward_argmax(r,token,position+out->processed);
        r->qwen4_mtp_verify_active = 0;
        if(next<0)return -1;
        out->processed++;
        out->pending=next;
        if(i>=out->drafted || next!=drafts[i])break;
        out->accepted++; token=next;
    }
    out->verify_ms=hllm_monotonic_ms()-start;
    return 0;
}

int hip_llm_qwen4_mtp_step(hip_llm_runner *r, int32_t anchor, int position,
                           int draft, int max_emit, const int32_t *stop_ids,
                           int n_stop, hip_llm_qwen4_mtp_result *out) {
    return hllm_qwen4_mtp_step(r,anchor,position,draft,max_emit,stop_ids,n_stop,out,NULL);
}

int hip_llm_verify_qwen4_mtp(hip_llm_runner *r, int32_t anchor, int position, int draft) {
    if(!r || draft<1 || draft>32 || position+draft+5>r->max_seq_len)return -1;
    hip_llm_state_snapshot *state=hip_llm_snapshot_state_window(r,position,draft+5);
    if(!state)return -1;
    int32_t reference[38], forced[32];
    int token=anchor, rc=0;
    for(int i=0;i<draft+5;++i) {
        reference[i]=token;
        token=hip_llm_forward_argmax(r,token,position+i);
        if(token<0){rc=-1;goto done;}
    }
    /* Includes every rejection slot and all acceptance. No rejected token may
     * reach the target; restoring a checkpoint must also restore NextN/QSA. */
    for(int reject=0;reject<=draft;++reject) {
        if(hip_llm_restore_state(r,state)){rc=-1;break;}
        for(int i=0;i<draft;++i)forced[i]=reference[i+1];
        if(reject<draft)forced[reject]=(forced[reject]+1)%r->n_vocab;
        hip_llm_qwen4_mtp_result out;
        if(hllm_qwen4_mtp_step(r,anchor,position,draft,draft+1,NULL,0,&out,forced) ||
           out.accepted!=reject || out.emitted!=reject+1){rc=-1;break;}
        for(int i=0;i<out.emitted;++i)if(out.tokens[i]!=reference[i])rc=-1;
        token=out.pending;
        for(int i=out.emitted;i<out.emitted+3;++i) {
            if(token!=reference[i])rc=-1;
            token=hip_llm_forward_argmax(r,token,position+i);
            if(token<0){rc=-1;break;}
        }
        fprintf(stderr,"Qwen4 MTP transaction reject=%d/%d continuation=%s\n",reject,draft,rc?"FAIL":"PASS");
        if(rc)break;
    }
    if (!rc) {
        hip_llm_qwen4_mtp_result out;
        /* EOS at the anchor must not mutate target or NextN state. */
        if (hip_llm_restore_state(r,state) ||
            hip_llm_qwen4_mtp_step(r,anchor,position,draft,draft+1,&anchor,1,&out) ||
            !out.stopped || out.emitted!=1 || out.processed!=0 || out.drafted!=0)
            rc=-1;
        /* A one-token output budget suppresses drafting entirely. */
        if (hip_llm_restore_state(r,state) ||
            hip_llm_qwen4_mtp_step(r,anchor,position,draft,1,NULL,0,&out) ||
            out.emitted!=1 || out.processed!=1 || out.drafted!=0 ||
            out.pending!=reference[1]) rc=-1;
        /* Accepted EOS is emitted but never part of the committed cache. */
        if (reference[1]!=anchor) {
            forced[0]=reference[1];
            if (hip_llm_restore_state(r,state) ||
                hllm_qwen4_mtp_step(r,anchor,position,1,2,forced,1,&out,forced) ||
                !out.stopped || out.emitted!=2 || out.processed!=1 ||
                out.accepted!=1) rc=-1;
        }
        fprintf(stderr,"Qwen4 MTP EOS/output-limit=%s\n",rc?"FAIL":"PASS");
    }
done:
    if(hip_llm_restore_state(r,state))rc=-1;
    hip_llm_free_state_snapshot(state);
    return rc;
}
