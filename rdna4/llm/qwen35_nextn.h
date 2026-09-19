/* Dense NextN draft path. Included after the scalar layer executor.
 * Draft state is independent; no unverified token may reach the caller. */
typedef struct hllm_qwen35_mtp {
    gguf_shards *source;
    qtensor embedding;
    hip_layer layer;
    void *enorm, *hnorm, *eh, *head_norm, *head;
    int eh_type, head_type, origin, trace_records, kv_end;
    int32_t pending_token;
    void *x, *fusion, *key, *value, *argmax, *logits;
    float *host_embedding;
    void *verify_x, *verify_logits, *verify_positions;
    void *verify_norm, *verify_gate, *verify_up;
    void *verify_q, *verify_scales;
    void *verify_ssm_qkv, *verify_ssm_z, *verify_ssm_alpha, *verify_ssm_beta, *verify_ssm_out;
    void *verify_conv[128], *verify_rec[128];
    float *host_logits;
    hipGraph_t graphs[17];
    hipGraphExec_t executions[17];
    int verify_capacity, verify_rows, verify_position;
} hllm_qwen35_mtp;

static void hllm_free_qwen35_mtp(hip_llm_runner *r) {
    hllm_qwen35_mtp *m = r->qwen35_mtp;
    if (!m) return;
    hipStreamSynchronize(r->stream);
    for (int i = 0; i < 17; ++i) {
        if (m->executions[i]) hipGraphExecDestroy(m->executions[i]);
        if (m->graphs[i]) hipGraphDestroy(m->graphs[i]);
    }
#define DENSE_FREE(p) do { if (p) hipFree(p); } while (0)
    DENSE_FREE(m->verify_x); DENSE_FREE(m->verify_logits); DENSE_FREE(m->verify_positions);
    DENSE_FREE(m->verify_norm); DENSE_FREE(m->verify_gate); DENSE_FREE(m->verify_up);
    DENSE_FREE(m->verify_q); DENSE_FREE(m->verify_scales);
    DENSE_FREE(m->verify_ssm_qkv); DENSE_FREE(m->verify_ssm_z);
    DENSE_FREE(m->verify_ssm_alpha); DENSE_FREE(m->verify_ssm_beta); DENSE_FREE(m->verify_ssm_out);
    for (int i = 0; i < 128; ++i) {
        DENSE_FREE(m->verify_conv[i]); DENSE_FREE(m->verify_rec[i]);
    }
    DENSE_FREE(m->enorm); DENSE_FREE(m->hnorm); DENSE_FREE(m->eh);
    DENSE_FREE(m->head_norm); DENSE_FREE(m->head);
    DENSE_FREE(m->x); DENSE_FREE(m->fusion); DENSE_FREE(m->key);
    DENSE_FREE(m->value); DENSE_FREE(m->argmax);
    DENSE_FREE(m->logits);
    DENSE_FREE(m->layer.attn_norm_w); DENSE_FREE(m->layer.ffn_norm_w);
    DENSE_FREE(m->layer.attn_q_norm_w); DENSE_FREE(m->layer.attn_k_norm_w);
    DENSE_FREE(m->layer.attn_q_w); DENSE_FREE(m->layer.attn_k_w);
    DENSE_FREE(m->layer.attn_v_w); DENSE_FREE(m->layer.attn_output_w);
    DENSE_FREE(m->layer.ffn_gate_w); DENSE_FREE(m->layer.ffn_up_w);
    DENSE_FREE(m->layer.ffn_down_w);
#undef DENSE_FREE
    free(m->host_embedding);
    free(m->host_logits);
    if (m->source) gguf_close_shards(m->source);
    free(m);
    r->qwen35_mtp = NULL;
}

/* DFlash2 shares the exact multi-row target verifier with dense NextN but
 * does not own a Qwen3.5 NextN layer.  Keep verifier allocation lazy; this
 * small shell only establishes the transaction state used by verify/commit. */
static int hllm_qwen35_verify_workspace_create(hip_llm_runner *r) {
    if (!r || r->qwen35_mtp) return -1;
    hllm_qwen35_mtp *m = calloc(1, sizeof(*m));
    if (!m) return -1;
    m->origin = -1;
    r->qwen35_mtp = m;
    return 0;
}

int hip_llm_qwen35_mtp_load(hip_llm_runner *r, const char *path,
                           char *error, size_t error_cap) {
    const gguf_shards *saved = hllm_active_shards;
    if (error && error_cap) snprintf(error, error_cap, "Unsupported dense NextN target/configuration");
    if (!r || !r->weights_loaded || !r->is_hybrid || r->is_moe ||
        r->is_qwen4exp || r->n_layers > 128 || r->qwen35_mtp || !path) return -1;
    if (hllm_qwen35_verify_workspace_create(r)) return -1;
    hllm_qwen35_mtp *m = r->qwen35_mtp;
    m->source = gguf_open_shards(path, 2);
    if (!m->source) goto fail;
    hllm_active_shards = m->source;
    gguf_context *g = m->source->metadata;
    if (hllm_get_int(g, "qwen35.block_count", 0) != r->n_layers + 1 ||
        hllm_get_int(g, "qwen35.nextn_predict_layers", 0) != 1 ||
        hllm_get_int(g, "qwen35.embedding_length", 0) != r->n_embd ||
        hllm_get_int(g, "qwen35.feed_forward_length", 0) != r->n_ff ||
        hllm_get_int(g, "qwen35.attention.head_count", 0) != r->n_heads ||
        hllm_get_int(g, "qwen35.attention.head_count_kv", 0) != r->n_kv_heads ||
        hllm_get_int(g, "qwen35.attention.key_length", 0) != r->head_dim ||
        hllm_get_int(g, "qwen35.attention.value_length", 0) != r->head_dim ||
        hllm_get_int(g, "qwen35.rope.dimension_count", 0) != 2*r->n_rope_pairs ||
        hllm_get_float(g, "qwen35.attention.layer_norm_rms_epsilon", 1e-6f) != r->rms_norm_eps ||
        hllm_get_float(g, "qwen35.rope.freq_base", 5000000.0f) != r->rope_freq_base)
        goto fail;
    char name[128];
    qtensor t;
    hip_layer *cl = &m->layer;
    cl->has_qk_norm = 1;
    cl->shared_kv_source = -1;
    cl->local_head_dim = r->head_dim;
    cl->local_kv_heads = r->n_kv_heads;
#define DENSE_TENSOR(suffix) \
    snprintf(name, sizeof(name), "blk.%d." suffix ".weight", r->n_layers); \
    t = hllm_load_tensor(g, name, 1)
#define DENSE_NORM(dst, suffix, n) do { DENSE_TENSOR(suffix); \
    if (!t.data || (size_t)t.n_rows*t.n_cols != (size_t)(n) || \
        upload_norm_f32(&(dst), &t, (n))) goto fail; } while (0)
#define DENSE_MAT(dst, type, suffix, nr, nc) do { DENSE_TENSOR(suffix); \
    if (!t.data || t.n_rows != (nr) || t.n_cols != (nc) || \
        upload_weight_matrix(&(dst), &t, &(type))) goto fail; } while (0)
#define DENSE_PROJ(field, suffix, nr, nc) do { \
    DENSE_MAT(cl->field##_w, cl->field##_type, suffix, nr, nc); \
    cl->field##_rows = nr; cl->field##_cols = nc; } while (0)
    int ne = r->n_embd, qd = r->n_heads*r->head_dim;
    int kd = r->n_kv_heads*r->head_dim;
    DENSE_PROJ(attn_q, "attn_q", 2*qd, ne);
    DENSE_PROJ(attn_k, "attn_k", kd, ne);
    DENSE_PROJ(attn_v, "attn_v", kd, ne);
    DENSE_PROJ(attn_output, "attn_output", ne, qd);
    DENSE_PROJ(ffn_gate, "ffn_gate", r->n_ff, ne);
    DENSE_PROJ(ffn_up, "ffn_up", r->n_ff, ne);
    DENSE_PROJ(ffn_down, "ffn_down", ne, r->n_ff);
    DENSE_NORM(cl->attn_norm_w, "attn_norm", ne);
    DENSE_NORM(cl->ffn_norm_w, "post_attention_norm", ne);
    DENSE_NORM(cl->attn_q_norm_w, "attn_q_norm", r->head_dim);
    DENSE_NORM(cl->attn_k_norm_w, "attn_k_norm", r->head_dim);
    DENSE_NORM(m->enorm, "nextn.enorm", ne);
    DENSE_NORM(m->hnorm, "nextn.hnorm", ne);
    DENSE_NORM(m->head_norm, "nextn.shared_head_norm", ne);
    DENSE_MAT(m->eh, m->eh_type, "nextn.eh_proj", ne, 2*ne);
#undef DENSE_PROJ
#undef DENSE_MAT
#undef DENSE_NORM
#undef DENSE_TENSOR
    t = hllm_load_tensor(g, "output.weight", 1);
    if (!t.data || t.n_rows != r->n_vocab || t.n_cols != ne ||
        upload_weight_matrix(&m->head, &t, &m->head_type)) goto fail;
    m->embedding = hllm_load_tensor(g, "token_embd.weight", 1);
    if (!m->embedding.data || m->embedding.n_rows != r->n_vocab ||
        m->embedding.n_cols != ne || !dequant_row_size(m->embedding.type, ne)) goto fail;
    m->host_embedding = malloc((size_t)ne*sizeof(float));
    if (!m->host_embedding || hipMalloc(&m->x, (size_t)ne*sizeof(float)) ||
        hipMalloc(&m->fusion, (size_t)2*ne*sizeof(float)) ||
        hipMalloc(&m->key, (size_t)r->max_seq_len*kd*2) ||
        hipMalloc(&m->value, (size_t)r->max_seq_len*kd*2) ||
        hipMalloc(&m->logits, (size_t)r->n_vocab*sizeof(float)) ||
        hipMalloc(&m->argmax, sizeof(int))) goto fail;
    hllm_active_shards = saved;
    if (error && error_cap) error[0] = 0;
    return 0;
fail:
    hllm_active_shards = saved;
    if (error && error_cap) snprintf(error, error_cap, "Invalid dense NextN sidecar or allocation failure");
    hllm_free_qwen35_mtp(r);
    return -1;
}

int hip_llm_qwen35_mtp_propose(hip_llm_runner *r, int32_t anchor, int position,
                              int count, int32_t *drafts) {
    hllm_qwen35_mtp *m = r ? r->qwen35_mtp : NULL;
    if (!m || !m->source || !m->head || m->verify_rows || !drafts || count < 1 || count > 16 || anchor < 0 ||
        anchor >= r->n_vocab || position < 0 || position > r->max_seq_len-count) return -1;
    if (m->origin < 0) { m->origin = position; m->kv_end = 0; }
    if (position < m->origin) return -1;
    int gap = position-m->origin-m->kv_end;
    if (gap > 1) return -1;
    int32_t requested_anchor = anchor;
    void *saved_x = r->d_x;
    int saved_pos = r->cur_position, saved_quant = r->kv_quantized;
    int saved_layer = r->active_layer;
    hip_llm_kv_cache_type saved_kv = r->kv_cache_type;
    int rc = 0;
    r->kv_quantized = 0;
    r->active_layer = r->n_layers;
    r->kv_cache_type = HIP_LLM_KV_F16;
    /* A fully accepted K+1-row window advances one token beyond the draft
     * cache. Fill that position before attending to it on the next round. */
    for (int j = gap > 0 ? -1 : 0; j < count; ++j) {
        if (j < 0) anchor = m->pending_token;
        else if (j == 0) anchor = requested_anchor;
        int p = position - m->origin + j;
        const char *trace = getenv("LLM_QWEN35_MTP_TRACE");
        char trace_path[4096];
        int traced = j >= 0 && trace && m->trace_records < 3;
        if (traced) {
            int32_t header[] = {r->n_embd, r->n_vocab, anchor, p};
            if (snprintf(trace_path, sizeof(trace_path), "%s.%d.input", trace, m->trace_records) >= (int)sizeof(trace_path) ||
                hipStreamSynchronize(r->stream) ||
                hipMemcpy(m->host_embedding, j ? m->x : saved_x, (size_t)r->n_embd*sizeof(float), hipMemcpyDeviceToHost)) { rc=-1; break; }
            FILE *f = fopen(trace_path, "wb");
            if (!f) { rc=-1; break; }
            int ok = fwrite(header, sizeof(header), 1, f) == 1 &&
                fwrite(m->host_embedding, sizeof(float), r->n_embd, f) == (size_t)r->n_embd;
            if (fclose(f) || !ok) { rc=-1; break; }
        }
        const void *embedding = (const char *)m->embedding.data +
            (size_t)anchor * dequant_row_size(m->embedding.type, r->n_embd);
        if (dequant_row(m->embedding.type, embedding, m->host_embedding, r->n_embd)) { rc=-1; break; }
        if (hipMemcpy(m->fusion, m->host_embedding, (size_t)r->n_embd*sizeof(float), hipMemcpyHostToDevice)) { rc=-1; break; }
        launch_rmsnorm(r, m->fusion, m->fusion, m->enorm, r->n_embd, r->rms_norm_eps);
        void *hn = (float *)m->fusion + r->n_embd;
        launch_rmsnorm(r, hn, j ? m->x : saved_x, m->hnorm, r->n_embd, r->rms_norm_eps);
        launch_matvec_auto(r, m->x, m->eh, m->fusion, r->n_embd, 2*r->n_embd, m->eh_type);
        r->d_x = m->x;
        r->cur_position = p;
        if (hipMemcpyAsync(r->d_position, &p, sizeof(p), hipMemcpyHostToDevice, r->stream)) { rc=-1; break; }
        if (j < 0 && !r->ssm_fused_decode) {
            /* Only the missing K/V entry is needed; no draft prediction is
             * consumed for the target's bonus token. */
            hip_layer *cl = &m->layer;
            launch_rmsnorm(r, r->d_xb, m->x, cl->attn_norm_w, r->n_embd, r->rms_norm_eps);
            begin_q8x2_reuse(r);
            launch_matvec_auto(r, r->d_k, cl->attn_k_w, r->d_xb,
                cl->attn_k_rows, cl->attn_k_cols, cl->attn_k_type);
            launch_matvec_auto(r, r->d_v, cl->attn_v_w, r->d_xb,
                cl->attn_v_rows, cl->attn_v_cols, cl->attn_v_type);
            end_q8x2_reuse(r);
            launch_qknorm(r, r->d_k, cl->attn_k_norm_w, r->n_kv_heads, r->head_dim, r->rms_norm_eps);
            launch_rope_devp(r, r->d_k, r->n_kv_heads, r->head_dim, r->rope_freq_base);
            launch_kv_store_f16_devp(r, m->key, m->value, r->d_k, r->d_v,
                r->n_kv_heads*r->head_dim);
            /* The next iteration rewrites host_embedding/fusion. Complete
             * this nonblocking stream before the next host upload. */
            if (hipStreamSynchronize(r->stream) || r->qwen4_forward_error) { rc=-1; break; }
            m->kv_end = p+1;
            continue;
        }
        forward_layer_state(r, &m->layer, 0, m->key, m->value, 0);
        launch_rmsnorm(r, m->x, m->x, m->head_norm, r->n_embd, r->rms_norm_eps);
        launch_matvec_auto(r, m->logits, m->head, m->x, r->n_vocab, r->n_embd, m->head_type);
        launch_qwen35_argmax(r, m->logits, m->argmax);
        if (hipMemcpyAsync(&anchor, m->argmax, sizeof(anchor), hipMemcpyDeviceToHost, r->stream) ||
            hipStreamSynchronize(r->stream) || r->qwen4_forward_error) { rc=-1; break; }
        m->kv_end = p+1;
        m->pending_token = anchor;
        if (j >= 0) drafts[j] = anchor;
        if (traced) {
            float *logits = malloc((size_t)r->n_vocab*sizeof(float));
            if (!logits) { rc=-1; break; }
            int ok = !hipMemcpy(logits, m->logits, (size_t)r->n_vocab*sizeof(float), hipMemcpyDeviceToHost);
            snprintf(trace_path, sizeof(trace_path), "%s.%d.logits", trace, m->trace_records++);
            FILE *f = ok ? fopen(trace_path, "wb") : NULL;
            if (f) { ok = fwrite(logits, sizeof(float), r->n_vocab, f) == (size_t)r->n_vocab; if (fclose(f)) ok=0; }
            else ok=0;
            free(logits);
            if (!ok) { rc=-1; break; }
        }
    }
    r->d_x = saved_x;
    r->cur_position = saved_pos;
    r->active_layer = saved_layer;
    r->kv_quantized = saved_quant;
    r->kv_cache_type = saved_kv;
    r->q8x2_reuse_valid = r->iq1_q8_valid = 0;
    if (hipMemcpyAsync(r->d_position, &saved_pos, sizeof(saved_pos), hipMemcpyHostToDevice, r->stream) ||
        hipStreamSynchronize(r->stream)) rc=-1;
    return rc;
}

static void hllm_dense_mtp_projection(hip_llm_runner *r, void *dst, void *w,
        void *x, int rows, int nr, int nc, int type) {
    hllm_qwen35_mtp *m = r->qwen35_mtp;
    hipFunction_t fn = NULL;
    switch (type) {
        case GGML_TYPE_Q2_K: if (rows <= 4 && nc <= 6144) fn=r->fn_qwen35_matvec_q2k_multi4; break;
        case GGML_TYPE_IQ2_XXS: fn=r->fn_qwen35_matvec_iq2xxs; break;
        case GGML_TYPE_IQ2_XS: fn=r->fn_qwen35_matvec_iq2xs; break;
        case GGML_TYPE_IQ2_S: fn=r->fn_qwen35_matvec_iq2s; break;
        case GGML_TYPE_IQ3_XXS: fn=r->fn_qwen35_matvec_iq3xxs; break;
        case GGML_TYPE_IQ3_S: fn=r->fn_qwen35_matvec_iq3s; break;
        case GGML_TYPE_IQ4_XS: fn=r->fn_qwen35_matvec_iq4xs; break;
    }
    if (fn && nc % 256 == 0 && nc <= r->n_ff) {
        int total = rows*nc;
        void *qa[] = { &m->verify_q, &m->verify_scales, &x, &total };
        LAUNCH(r->fn_qwen35_quantize_q81, total/32, 1, 1, 32, 1, 1, 0, r->stream, qa);
        void *a[] = { &dst, &w, &m->verify_q, &m->verify_scales, &nr, &nc };
        if (type == GGML_TYPE_Q2_K) {
            void *ma[] = { &dst, &w, &m->verify_q, &m->verify_scales, &nr, &nc, &rows };
            LAUNCH(fn, (nr+3)/4, 1, 1, 128, 1, 1, 0, r->stream, ma);
        } else if (rows <= 4 && type != GGML_TYPE_IQ4_XS &&
                   !(type == GGML_TYPE_IQ2_S && nc > 6144)) {
            int kind = type==GGML_TYPE_IQ2_XXS ? 0 : type==GGML_TYPE_IQ2_XS ? 1 :
                type==GGML_TYPE_IQ2_S ? 2 : type==GGML_TYPE_IQ3_XXS ? 3 : 4;
            void *ma[] = { &dst, &w, &m->verify_q, &m->verify_scales, &nr, &nc, &kind, &rows };
            LAUNCH(r->fn_qwen35_matvec_iq_multi4, (nr+3)/4, 1, 1, 128, 1, 1, 0, r->stream, ma);
        } else LAUNCH(fn, (nr+7)/8, rows, 1, 256, 1, 1, 0, r->stream, a);
    } else {
        for (int i = 0; i < rows; ++i)
            launch_matvec_ffn_auto(r, (float *)dst+(size_t)i*nr, w,
                (float *)x+(size_t)i*nc, nr, nc, type, 0);
    }
    r->q8x2_reuse_valid = r->iq1_q8_valid = 0;
}

static int hllm_dense_mtp_native_iq(int type) {
    return type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS ||
        type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_XXS ||
        type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ4_XS;
}

static void hllm_dense_mtp_ssm(hip_llm_runner *r, hip_layer *cl, int l, int rows,
                              size_t conv, size_t rec) {
    hllm_qwen35_mtp *m = r->qwen35_mtp;
    int ne = r->n_embd, dt = r->ssm_dt_rank, ds = r->ssm_d_state;
    float eps = r->rms_norm_eps;
    for (int i = 0; i < rows; ++i)
        launch_rmsnorm(r, (float *)m->verify_norm+(size_t)i*ne,
            (float *)m->verify_x+(size_t)i*ne, cl->attn_norm_w, ne, eps);
    hllm_dense_mtp_projection(r, m->verify_ssm_qkv, cl->ssm_qkv_w, m->verify_norm,
        rows, cl->ssm_qkv_rows, cl->ssm_qkv_cols, cl->ssm_qkv_type);
    hllm_dense_mtp_projection(r, m->verify_ssm_z, cl->ssm_gate_w, m->verify_norm,
        rows, cl->ssm_gate_rows, cl->ssm_gate_cols, cl->ssm_gate_type);
    void *qkv = r->d_ssm_qkv, *z = r->d_ssm_z, *alpha = r->d_ssm_alpha;
    void *beta = r->d_ssm_beta, *out = r->d_ssm_out;
    for (int i = 0; i < rows; ++i) {
        r->d_ssm_qkv = (float *)m->verify_ssm_qkv+(size_t)i*r->ssm_qkv_dim;
        r->d_ssm_z = (float *)m->verify_ssm_z+(size_t)i*r->ssm_d_inner;
        r->d_ssm_alpha = (float *)m->verify_ssm_alpha+(size_t)i*dt;
        r->d_ssm_beta = (float *)m->verify_ssm_beta+(size_t)i*dt;
        r->d_ssm_out = (float *)m->verify_ssm_out+(size_t)i*r->ssm_d_inner;
        void *x = (float *)m->verify_norm+(size_t)i*ne;
        if (cl->ssm_alpha_type == GGML_TYPE_BF16)
            launch_matvec_llama_bf16(r, r->d_ssm_alpha, cl->ssm_alpha_w, x, dt, ne);
        else launch_matvec_llama_f16(r, r->d_ssm_alpha, cl->ssm_alpha_w, x, dt, ne);
        if (cl->ssm_beta_type == GGML_TYPE_BF16)
            launch_matvec_llama_bf16(r, r->d_ssm_beta, cl->ssm_beta_w, x, dt, ne);
        else launch_matvec_llama_f16(r, r->d_ssm_beta, cl->ssm_beta_w, x, dt, ne);
        forward_dense_ssm_core(r, cl, l);
        const char *separate = getenv("LLM_QWEN35_SSM_NORM_SEPARATE");
        if (separate && atoi(separate)) {
            launch_rmsnorm_heads_inplace(r, r->d_ssm_out, cl->ssm_norm_w, dt, ds, eps);
            launch_silu_gate_mul(r, r->d_ssm_out, r->d_ssm_z, dt*ds);
        } else launch_gated_rmsnorm_silu(r, r->d_ssm_out, r->d_ssm_z, cl->ssm_norm_w, dt, ds, eps);
        hipMemcpyAsync((char *)m->verify_conv[l]+(size_t)i*conv,
            cl->d_conv_state, conv, hipMemcpyDeviceToDevice, r->stream);
        hipMemcpyAsync((char *)m->verify_rec[l]+(size_t)i*rec,
            cl->d_recurrent_state, rec, hipMemcpyDeviceToDevice, r->stream);
    }
    r->d_ssm_qkv=qkv; r->d_ssm_z=z; r->d_ssm_alpha=alpha; r->d_ssm_beta=beta; r->d_ssm_out=out;
    hllm_dense_mtp_projection(r, m->verify_norm, cl->ssm_out_w, m->verify_ssm_out,
        rows, cl->ssm_out_rows, cl->ssm_out_cols, cl->ssm_out_type);
    for (int i = 0; i < rows; ++i) {
        void *x = (float *)m->verify_x+(size_t)i*ne;
        void *y = (float *)m->verify_norm+(size_t)i*ne;
        const char *split = getenv("LLM_QWEN35_SPLIT_RES_RMSNORM");
        if (split && atoi(split)) {
            launch_add(r, x, y, ne);
            launch_rmsnorm(r, y, x, cl->ffn_norm_w, ne, eps);
        } else {
            void *a[] = { &x, &y, &y, &cl->ffn_norm_w, &ne, &eps };
            LAUNCH(r->fn_res_rmsnorm_f32, 1, 1, 1, 256, 1, 1, 256*sizeof(float), r->stream, a);
        }
    }
}

/* Capture a layer-major window using precisely the scalar target kernels.
 * Each row owns its hidden vector; recurrent state is checkpointed after
 * every row. Positions are device inputs, so graphs remain valid as the
 * context grows. Rejected KV rows are masked by the committed position. */
float *hip_llm_qwen35_mtp_verify(hip_llm_runner *r, const int32_t *tokens,
                                int rows, int position) {
    hllm_qwen35_mtp *m = r ? r->qwen35_mtp : NULL;
    if (!m || m->verify_rows || !tokens || rows < 1 || rows > 16 || position < 0 ||
        position > r->max_seq_len-rows || !r->decode_mode || r->debug_layers ||
        !r->requested_qwen35_decode_graph ||
        r->kv_cache_type != HIP_LLM_KV_Q8_0_Q8_0 || !r->fn_q8_attention_decode) return NULL;
    for (int i = 0; i < rows; ++i) if (tokens[i] < 0 || tokens[i] >= r->n_vocab) return NULL;
    size_t conv = (size_t)(r->ssm_conv_kernel-1)*r->ssm_qkv_dim*sizeof(float);
    size_t rec = (size_t)r->ssm_dt_rank*r->ssm_d_state*r->ssm_d_state*sizeof(float);
    if (!m->verify_capacity) {
        /* Fixed capacity keeps graph pointers stable; four rows cover K<=3. */
        int capacity = rows;
        m->verify_capacity = -1; /* A partial allocation cannot be retried over live pointers. */
        if (hipMalloc(&m->verify_x, (size_t)capacity*r->n_embd*sizeof(float)) ||
            hipMalloc(&m->verify_norm, (size_t)capacity*r->n_embd*sizeof(float)) ||
            hipMalloc(&m->verify_gate, (size_t)capacity*r->n_ff*sizeof(float)) ||
            hipMalloc(&m->verify_up, (size_t)capacity*r->n_ff*sizeof(float)) ||
            hipMalloc(&m->verify_q, (size_t)capacity*r->n_ff) ||
            hipMalloc(&m->verify_scales, (size_t)capacity*r->n_ff/32*sizeof(float)) ||
            hipMalloc(&m->verify_ssm_qkv, (size_t)capacity*r->ssm_qkv_dim*sizeof(float)) ||
            hipMalloc(&m->verify_ssm_z, (size_t)capacity*r->ssm_d_inner*sizeof(float)) ||
            hipMalloc(&m->verify_ssm_out, (size_t)capacity*r->ssm_d_inner*sizeof(float)) ||
            hipMalloc(&m->verify_ssm_alpha, (size_t)capacity*r->ssm_dt_rank*sizeof(float)) ||
            hipMalloc(&m->verify_ssm_beta, (size_t)capacity*r->ssm_dt_rank*sizeof(float)) ||
            hipMalloc(&m->verify_logits, (size_t)capacity*r->n_vocab*sizeof(float)) ||
            hipMalloc(&m->verify_positions, (size_t)capacity*sizeof(int))) return NULL;
        m->host_logits = malloc((size_t)capacity*r->n_vocab*sizeof(float));
        if (!m->host_logits) return NULL;
        for (int l = 0; l < r->n_layers; ++l) if (r->layers[l].is_ssm) {
            if (hipMalloc(&m->verify_conv[l], (size_t)capacity*conv) ||
                hipMalloc(&m->verify_rec[l], (size_t)capacity*rec)) return NULL;
        }
        m->verify_capacity = capacity;
        hllm_vram_sample(r);
    }
    if (rows > m->verify_capacity) return NULL;
    void *saved_x = r->d_x;
    int saved_position = r->cur_position;
    int saved_layer = r->active_layer;
    if (!m->executions[rows]) {
        if (hipStreamSynchronize(r->stream) ||
            hipStreamBeginCapture(r->stream, hipStreamCaptureModeThreadLocal)) return NULL;
        for (int l = 0; l < r->n_layers; ++l) {
            r->active_layer = l;
            hip_layer *cl = &r->layers[l];
            hllm_qwen35_dflash2_capture(r, l, m->verify_x, rows);
            /* Group each projection so its weights remain hot across rows.
             * Fused Q6/Q8 FFNs retain their original scalar execution. */
            int grouped = cl->ffn_gate_type != GGML_TYPE_Q6_K &&
                cl->ffn_up_type != GGML_TYPE_Q6_K && cl->ffn_down_type != GGML_TYPE_Q6_K &&
                cl->ffn_gate_type != GGML_TYPE_Q8_0 && cl->ffn_up_type != GGML_TYPE_Q8_0 &&
                r->fn_qwen35_matvec_iq3xxs;
            int grouped_ssm = grouped && cl->is_ssm &&
                hllm_dense_mtp_native_iq(cl->ssm_qkv_type) &&
                hllm_dense_mtp_native_iq(cl->ssm_gate_type) &&
                hllm_dense_mtp_native_iq(cl->ssm_out_type) &&
                (cl->ssm_alpha_type == GGML_TYPE_F16 || cl->ssm_alpha_type == GGML_TYPE_BF16) &&
                (cl->ssm_beta_type == GGML_TYPE_F16 || cl->ssm_beta_type == GGML_TYPE_BF16);
            if (grouped_ssm) hllm_dense_mtp_ssm(r, cl, l, rows, conv, rec);
            else for (int i = 0; i < rows; ++i) {
                r->d_x = (float *)m->verify_x + (size_t)i*r->n_embd;
                r->cur_position = position+i;
                hipMemcpyAsync(r->d_position, (int *)m->verify_positions+i, sizeof(int),
                               hipMemcpyDeviceToDevice, r->stream);
                forward_layer_state_phase(r, cl, l, r->d_key_cache[l], r->d_value_cache[l], 1, grouped);
                if (grouped) hipMemcpyAsync((float *)m->verify_norm+(size_t)i*r->n_embd,
                    r->d_xb, (size_t)r->n_embd*sizeof(float), hipMemcpyDeviceToDevice, r->stream);
                if (cl->is_ssm) {
                    hipMemcpyAsync((char *)m->verify_conv[l]+(size_t)i*conv,
                        cl->d_conv_state, conv, hipMemcpyDeviceToDevice, r->stream);
                    hipMemcpyAsync((char *)m->verify_rec[l]+(size_t)i*rec,
                        cl->d_recurrent_state, rec, hipMemcpyDeviceToDevice, r->stream);
                }
            }
            if (grouped) {
                hllm_dense_mtp_projection(r, m->verify_gate, cl->ffn_gate_w, m->verify_norm,
                    rows, cl->ffn_gate_rows, cl->ffn_gate_cols, cl->ffn_gate_type);
                hllm_dense_mtp_projection(r, m->verify_up, cl->ffn_up_w, m->verify_norm,
                    rows, cl->ffn_up_rows, cl->ffn_up_cols, cl->ffn_up_type);
                for (int i = 0; i < rows; ++i)
                    launch_silu_mul(r, (float *)m->verify_gate+(size_t)i*r->n_ff,
                        (float *)m->verify_up+(size_t)i*r->n_ff, r->n_ff);
                hllm_dense_mtp_projection(r, m->verify_norm, cl->ffn_down_w, m->verify_gate,
                    rows, cl->ffn_down_rows, cl->ffn_down_cols, cl->ffn_down_type);
                launch_add(r, m->verify_x, m->verify_norm, rows*r->n_embd);
            }
        }
        for (int i = 0; i < rows; ++i) {
            void *x = (float *)m->verify_x + (size_t)i*r->n_embd;
            launch_rmsnorm(r, x, x, r->d_output_norm, r->n_embd, r->rms_norm_eps);
            launch_matvec_auto(r, (float *)m->verify_logits+(size_t)i*r->n_vocab,
                r->d_output_w, x, r->n_vocab, r->n_embd, r->output_w_type);
        }
        r->d_x = saved_x;
        r->cur_position = saved_position;
        r->active_layer = saved_layer;
        if (hipStreamEndCapture(r->stream, &m->graphs[rows]) ||
            hipGraphInstantiate(&m->executions[rows], m->graphs[rows], NULL, NULL, 0)) return NULL;
    }
    int positions[16];
    for (int i = 0; i < rows; ++i) {
        positions[i] = position+i;
        void *x = (float *)m->verify_x + (size_t)i*r->n_embd;
        int t = tokens[i];
        if (r->token_embd_type == GGML_TYPE_Q8_0) launch_embed_q8_0(r,x,r->d_token_embd,t,r->n_embd);
        else if (r->token_embd_type == GGML_TYPE_Q4_0) launch_embed_q4_0(r,x,r->d_token_embd,t,r->n_embd);
        else if (r->token_embd_type == GGML_TYPE_Q2_K) launch_embed_q2_K(r,x,r->d_token_embd,t,r->n_embd);
        else if (r->token_embd_type == GGML_TYPE_IQ1_M) launch_embed_iq1_m(r,x,r->d_token_embd,t,r->n_embd);
        else if (r->token_embd_type == GGML_TYPE_F32) launch_embed_f32(r,x,r->d_token_embd,t,r->n_embd);
        else launch_embed(r,x,r->d_token_embd,t,r->n_embd);
    }
    if (hipMemcpyAsync(m->verify_positions, positions, (size_t)rows*sizeof(int), hipMemcpyHostToDevice,r->stream) ||
        hipGraphLaunch(m->executions[rows],r->stream) ||
        hipMemcpyAsync(m->host_logits,m->verify_logits,(size_t)rows*r->n_vocab*sizeof(float),hipMemcpyDeviceToHost,r->stream) ||
        hipStreamSynchronize(r->stream) || r->qwen4_forward_error) return NULL;
    m->verify_rows = rows;
    m->verify_position = position;
    return m->host_logits;
}

int hip_llm_qwen35_mtp_commit(hip_llm_runner *r, int processed) {
    hllm_qwen35_mtp *m = r ? r->qwen35_mtp : NULL;
    if (!m || processed < 1 || processed > m->verify_rows) return -1;
    int last = processed-1;
    size_t conv = (size_t)(r->ssm_conv_kernel-1)*r->ssm_qkv_dim*sizeof(float);
    size_t rec = (size_t)r->ssm_dt_rank*r->ssm_d_state*r->ssm_d_state*sizeof(float);
    for (int l = 0; l < r->n_layers; ++l) if (r->layers[l].is_ssm) {
        if (hipMemcpyAsync(r->layers[l].d_conv_state,(char *)m->verify_conv[l]+(size_t)last*conv,conv,hipMemcpyDeviceToDevice,r->stream) ||
            hipMemcpyAsync(r->layers[l].d_recurrent_state,(char *)m->verify_rec[l]+(size_t)last*rec,rec,hipMemcpyDeviceToDevice,r->stream)) return -1;
    }
    r->cur_position = m->verify_position+last;
    if (hipMemcpyAsync(r->d_x,(float *)m->verify_x+(size_t)last*r->n_embd,(size_t)r->n_embd*sizeof(float),hipMemcpyDeviceToDevice,r->stream) ||
        hipMemcpyAsync(r->d_logits,(float *)m->verify_logits+(size_t)last*r->n_vocab,(size_t)r->n_vocab*sizeof(float),hipMemcpyDeviceToDevice,r->stream) ||
        hipMemcpyAsync(r->d_position,&r->cur_position,sizeof(int),hipMemcpyHostToDevice,r->stream) ||
        hipStreamSynchronize(r->stream)) return -1;
    m->verify_rows = 0;
    r->q8x2_reuse_valid = r->iq1_q8_valid = 0;
    return 0;
}
