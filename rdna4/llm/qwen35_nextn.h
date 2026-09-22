/* Dense NextN draft path. Included after the scalar layer executor.
 * Draft state is independent; no unverified token may reach the caller. */
enum {
    HLLM_DENSE_MTP_MAX_ROWS = 16,
    HLLM_DENSE_MTP_REUSE_ROWS = 8,
    HLLM_DENSE_MTP_SMALL_REUSE_ROWS = 4,
};

static int hllm_qwen35_attention_splits(const hip_llm_runner *r, int length) {
    const int occupancy = 11;
    int tiles = (length + 255) / 256;
    if (length >= 16384) {
        int splits = (length + 511) / 512;
        /* Keep graph selection identical to the native Q8 attention kernel.
         * The validated 64K--<96K serving window uses 64 larger partitions;
         * without this cap a reuse graph could span the 96K boundary while
         * the device kernel changes from 64 to 128 partitions. */
        if (length >= 65536 && length < 98304)
            splits = splits < 64 ? splits : 64;
        if (splits < occupancy) splits = occupancy;
        return splits > 128 ? 128 : splits;
    }
    int splits = occupancy < tiles ? occupancy : tiles;
    int best = 0, waves_best = 0;
    int blocks_per_wave = r->q8_attention_nsm * occupancy;
    for (int trial = splits; trial <= tiles; ++trial) {
        int blocks = r->n_heads * trial;
        int waves = (blocks + blocks_per_wave - 1) / blocks_per_wave;
        int efficiency = 100 * blocks / (waves * blocks_per_wave);
        if (best >= 95 && waves > waves_best) break;
        if (efficiency > best) {
            best = efficiency;
            waves_best = waves;
            splits = trial;
        }
    }
    return splits;
}

typedef struct hllm_qwen35_mtp {
    gguf_shards *source;
    qtensor embedding;
    hip_layer layer;
    void *enorm, *hnorm, *eh, *head_norm, *head;
    int eh_type, head_type, origin, trace_records, kv_end;
    int32_t pending_token;
    void *x, *fusion, *key, *value, *argmax, *logits;
    float *host_embedding;
    void *verify_x, *verify_logits, *verify_positions, *verify_argmax;
    void *verify_norm, *verify_gate, *verify_up;
    void *verify_q, *verify_scales;
    void *verify_ssm_qkv, *verify_ssm_z, *verify_ssm_alpha, *verify_ssm_beta, *verify_ssm_out;
    void *verify_attn_parts, *verify_attn_meta;
    void *verify_conv[128], *verify_rec[128];
    void *verify_conv_dst_ptrs, *verify_conv_src_ptrs;
    void *verify_rec_dst_ptrs, *verify_rec_src_ptrs;
    float *host_logits;
    hipGraph_t graphs[HLLM_DENSE_MTP_MAX_ROWS + 1];
    hipGraphExec_t executions[HLLM_DENSE_MTP_MAX_ROWS + 1];
    hipGraph_t reuse_graphs[HLLM_DENSE_MTP_MAX_ROWS + 1];
    hipGraphExec_t reuse_executions[HLLM_DENSE_MTP_MAX_ROWS + 1];
    int verify_capacity, verify_rows, verify_position, verify_ssm_layers;
    int verify_attention_reuse;
} hllm_qwen35_mtp;

static void hllm_free_qwen35_mtp(hip_llm_runner *r) {
    hllm_qwen35_mtp *m = r->qwen35_mtp;
    if (!m) return;
    hipStreamSynchronize(r->stream);
    for (int i = 0; i <= HLLM_DENSE_MTP_MAX_ROWS; ++i) {
        if (m->executions[i]) hipGraphExecDestroy(m->executions[i]);
        if (m->graphs[i]) hipGraphDestroy(m->graphs[i]);
        if (m->reuse_executions[i]) hipGraphExecDestroy(m->reuse_executions[i]);
        if (m->reuse_graphs[i]) hipGraphDestroy(m->reuse_graphs[i]);
    }
#define DENSE_FREE(p) do { if (p) hipFree(p); } while (0)
    DENSE_FREE(m->verify_x); DENSE_FREE(m->verify_logits); DENSE_FREE(m->verify_positions);
    DENSE_FREE(m->verify_argmax);
    DENSE_FREE(m->verify_norm); DENSE_FREE(m->verify_gate); DENSE_FREE(m->verify_up);
    DENSE_FREE(m->verify_q); DENSE_FREE(m->verify_scales);
    DENSE_FREE(m->verify_ssm_qkv); DENSE_FREE(m->verify_ssm_z);
    DENSE_FREE(m->verify_ssm_alpha); DENSE_FREE(m->verify_ssm_beta); DENSE_FREE(m->verify_ssm_out);
    DENSE_FREE(m->verify_attn_parts); DENSE_FREE(m->verify_attn_meta);
    DENSE_FREE(m->verify_conv_dst_ptrs); DENSE_FREE(m->verify_conv_src_ptrs);
    DENSE_FREE(m->verify_rec_dst_ptrs); DENSE_FREE(m->verify_rec_src_ptrs);
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
    if (!m || !m->source || !m->head || m->verify_rows || !drafts ||
        count < 1 || count > HLLM_DENSE_MTP_MAX_ROWS || anchor < 0 ||
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

static int hllm_dense_mtp_iq_kind(int type) {
    switch (type) {
        case GGML_TYPE_IQ2_XXS: return 0;
        case GGML_TYPE_IQ2_XS: return 1;
        case GGML_TYPE_IQ2_S: return 2;
        case GGML_TYPE_IQ3_XXS: return 3;
        case GGML_TYPE_IQ3_S: return 4;
        default: return -1;
    }
}

static hipFunction_t hllm_dense_mtp_iq_multi8(hip_llm_runner *r, int type) {
    switch (type) {
        case GGML_TYPE_IQ2_XXS: return r->fn_qwen35_matvec_iq2xxs_multi8;
        case GGML_TYPE_IQ2_XS: return r->fn_qwen35_matvec_iq2xs_multi8;
        case GGML_TYPE_IQ2_S: return r->fn_qwen35_matvec_iq2s_multi8;
        case GGML_TYPE_IQ3_XXS: return r->fn_qwen35_matvec_iq3xxs_multi8;
        case GGML_TYPE_IQ3_S: return r->fn_qwen35_matvec_iq3s_multi8;
        default: return NULL;
    }
}

static hipFunction_t hllm_dense_mtp_iq_fixed8(hip_llm_runner *r, int type) {
    switch (type) {
        case GGML_TYPE_IQ2_XXS: return r->fn_qwen35_matvec_iq2xxs_fixed8;
        case GGML_TYPE_IQ2_XS: return r->fn_qwen35_matvec_iq2xs_fixed8;
        case GGML_TYPE_IQ2_S: return r->fn_qwen35_matvec_iq2s_fixed8;
        case GGML_TYPE_IQ3_XXS: return r->fn_qwen35_matvec_iq3xxs_fixed8;
        case GGML_TYPE_IQ3_S: return r->fn_qwen35_matvec_iq3s_fixed8;
        default: return NULL;
    }
}

static void hllm_dense_mtp_res_rmsnorm_batch(hip_llm_runner *r, void *x,
        void *res, void *normalized, void *weight, int width, int rows) {
    float eps = r->rms_norm_eps;
    void *args[] = { &x, &res, &normalized, &weight, &width, &rows, &eps };
    LAUNCH(r->fn_res_rmsnorm_batch_f32, rows, 1, 1, 256, 1, 1,
           256*sizeof(float), r->stream, args);
}

static void hllm_dense_mtp_projection(hip_llm_runner *r, void *dst, void *w,
        void *x, int rows, int nr, int nc, int type) {
    hllm_qwen35_mtp *m = r->qwen35_mtp;
    hip_layer *active = r->active_layer >= 0 && r->active_layer < r->n_layers ?
        &r->layers[r->active_layer] : NULL;
    if (rows <= HLLM_DENSE_MTP_REUSE_ROWS && nc % 256 == 0 && nc <= r->n_ff &&
        (type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M)) {
        /* Gate and up projections consume the same normalized rows.  Reuse
         * the exact Q8_1 bytes and block sums when the immediately preceding
         * IQ1 projection used the same source shape; a different source or
         * row count still forces a fresh quantization before graph replay. */
        const char *reuse_env = getenv("LLM_QWEN35_MTP_IQ1_Q81_REUSE");
        int reuse_enabled = !reuse_env || atoi(reuse_env) != 0;
        int reuse_q81 = reuse_enabled && r->iq1_q8_valid &&
                        r->iq1_q8_source == x &&
                        r->iq1_q8_n == nc && r->iq1_q8_rows == rows;
        if (!reuse_q81) {
            launch_quantize_q81_iq1_batch(r, x, nc, rows, nc);
            r->iq1_q8_source = x;
            r->iq1_q8_n = nc;
            r->iq1_q8_rows = rows;
            r->iq1_q8_valid = 1;
        }
        hipFunction_t iq1_fn = type == GGML_TYPE_IQ1_S ?
            r->fn_matvec_iq1_s_q81_reuse8 : r->fn_matvec_iq1_m_q81_reuse8;
        void *a[] = { &dst, &w, &r->d_act_q8_batch,
                      &r->d_act_scale_batch, &r->d_act_scale_batch_b,
                      &nr, &nc, &rows };
        const char *mmq_env = type == GGML_TYPE_IQ1_S ?
            getenv("LLM_IQ1S_MMQ_SCALES") : NULL;
        int ffn_role = active && (w == active->ffn_gate_w ||
            w == active->ffn_up_w || w == active->ffn_down_w);
        if (type == GGML_TYPE_IQ1_S && !ffn_role && mmq_env &&
            atoi(mmq_env) != 0) {
            int threads = nr < nc ? 64 : 256;
            int rows_per_block = threads/32;
            LAUNCH(r->fn_matvec_iq1_s_mmq_scales,
                   (nr+rows_per_block-1)/rows_per_block, rows, 1,
                   threads, 1, 1, 0, r->stream, a);
        } else {
            LAUNCH(iq1_fn, (nr+7)/8, 1, 1, 256, 1, 1, 0, r->stream, a);
        }
        r->q8x2_reuse_valid = r->batch_q8_valid = 0;
        return;
    }
    hipFunction_t fn = NULL;
    switch (type) {
        case GGML_TYPE_Q2_K:
            if (rows <= HLLM_DENSE_MTP_REUSE_ROWS && nc <= 6144) {
                fn = r->fn_qwen35_matvec_q2k_multi4;
            }
            break;
        case GGML_TYPE_IQ2_XXS: fn = r->fn_qwen35_matvec_iq2xxs; break;
        case GGML_TYPE_IQ2_XS: fn = r->fn_qwen35_matvec_iq2xs; break;
        case GGML_TYPE_IQ2_S: fn = r->fn_qwen35_matvec_iq2s; break;
        case GGML_TYPE_IQ3_XXS: fn = r->fn_qwen35_matvec_iq3xxs; break;
        case GGML_TYPE_IQ3_S: fn = r->fn_qwen35_matvec_iq3s; break;
        case GGML_TYPE_IQ4_XS: fn = r->fn_qwen35_matvec_iq4xs; break;
    }
    if (fn && nc % 256 == 0 && nc <= r->n_ff) {
        /* The captured graph reuses one quantized scratch allocation for all
         * projections.  Always restage it: pointer identity does not imply
         * that an earlier graph node's contents survive later scratch writes. */
        int total = rows*nc;
        void *qa[] = { &m->verify_q, &m->verify_scales, &x, &total };
        LAUNCH(r->fn_qwen35_quantize_q81, total/32, 1, 1, 32, 1, 1, 0,
               r->stream, qa);
        void *a[] = { &dst, &w, &m->verify_q, &m->verify_scales, &nr, &nc };
        if (type == GGML_TYPE_Q2_K) {
            if (rows == HLLM_DENSE_MTP_REUSE_ROWS) {
                void *ma[] = { &dst, &w, &m->verify_q, &m->verify_scales,
                               &nr, &nc, &rows };
                LAUNCH(r->fn_qwen35_matvec_q2k_fixed8, (nr+7)/8, 1, 1,
                       256, 1, 1, 0, r->stream, ma);
            } else {
                int first = rows <= HLLM_DENSE_MTP_SMALL_REUSE_ROWS ? rows :
                    HLLM_DENSE_MTP_SMALL_REUSE_ROWS;
                void *ma[] = { &dst, &w, &m->verify_q, &m->verify_scales,
                               &nr, &nc, &first };
                LAUNCH(fn, (nr+3)/4, 1, 1, 128, 1, 1, 0, r->stream, ma);
                if (rows > first) {
                    int tail = rows - first;
                    void *tail_dst = (float *)dst + (size_t)first*nr;
                    void *tail_q = (signed char *)m->verify_q +
                        (size_t)first*nc;
                    void *tail_scales = (float *)m->verify_scales +
                        (size_t)first*(nc/32);
                    void *ta[] = { &tail_dst, &w, &tail_q, &tail_scales,
                                   &nr, &nc, &tail };
                    LAUNCH(fn, (nr+3)/4, 1, 1, 128, 1, 1, 0, r->stream, ta);
                }
            }
        } else if (rows > HLLM_DENSE_MTP_SMALL_REUSE_ROWS &&
                   type == GGML_TYPE_IQ4_XS) {
            void *ma[] = { &dst, &w, &m->verify_q, &m->verify_scales,
                           &nr, &nc, &rows };
            hipFunction_t iq4_fn = nc == 5120 ?
                r->fn_qwen35_matvec_iq4xs_5120_multi8 :
                r->fn_qwen35_matvec_iq4xs_multi8;
            LAUNCH(iq4_fn, (nr+7)/8, 1, 1,
                   256, 1, 1, 0, r->stream, ma);
        } else if (rows <= HLLM_DENSE_MTP_REUSE_ROWS &&
                   type != GGML_TYPE_IQ4_XS &&
                   !(type == GGML_TYPE_IQ2_S && nc > 6144)) {
            int kind = hllm_dense_mtp_iq_kind(type);
            if (rows <= HLLM_DENSE_MTP_SMALL_REUSE_ROWS) {
                void *ma[] = { &dst, &w, &m->verify_q, &m->verify_scales,
                               &nr, &nc, &kind, &rows };
                LAUNCH(r->fn_qwen35_matvec_iq_multi4, (nr+3)/4, 1, 1,
                       128, 1, 1, 0, r->stream, ma);
            } else {
                hipFunction_t multi = rows == HLLM_DENSE_MTP_REUSE_ROWS ?
                    hllm_dense_mtp_iq_fixed8(r, type) :
                    hllm_dense_mtp_iq_multi8(r, type);
                void *ma[] = { &dst, &w, &m->verify_q, &m->verify_scales,
                               &nr, &nc, &rows };
                LAUNCH(multi, (nr+7)/8, 1, 1, 256, 1, 1, 0,
                       r->stream, ma);
            }
        } else LAUNCH(fn, (nr+7)/8, rows, 1, 256, 1, 1, 0, r->stream, a);
    } else {
        for (int i = 0; i < rows; ++i)
            launch_matvec_ffn_auto(r, (float *)dst+(size_t)i*nr, w,
                (float *)x+(size_t)i*nc, nr, nc, type, 0);
    }
    r->q8x2_reuse_valid = r->iq1_q8_valid = r->batch_q8_valid = 0;
}

static void hllm_dense_mtp_ssm(hip_llm_runner *r, hip_layer *cl, int l, int rows,
                              size_t conv, size_t rec) {
    hllm_qwen35_mtp *m = r->qwen35_mtp;
    int ne = r->n_embd, dt = r->ssm_dt_rank, ds = r->ssm_d_state;
    int qkv_dim = r->ssm_qkv_dim, d_inner = r->ssm_d_inner;
    int n_group = r->ssm_n_group, conv_k = r->ssm_conv_kernel;
    float eps = r->rms_norm_eps;
    (void)conv;
    (void)rec;
    /* verify_norm is reused for every layer.  Its contents change at the
     * norm below, so an IQ1 cache from the previous layer cannot survive
     * into this projection pair merely because the pointer is unchanged. */
    r->iq1_q8_valid = 0;
    launch_rmsnorm_batch(r, m->verify_norm, m->verify_x, cl->attn_norm_w,
                         ne, rows, ne, eps);
    hllm_dense_mtp_projection(r, m->verify_ssm_qkv, cl->ssm_qkv_w, m->verify_norm,
        rows, cl->ssm_qkv_rows, cl->ssm_qkv_cols, cl->ssm_qkv_type);
    hllm_dense_mtp_projection(r, m->verify_ssm_z, cl->ssm_gate_w, m->verify_norm,
        rows, cl->ssm_gate_rows, cl->ssm_gate_cols, cl->ssm_gate_type);
    if (cl->ssm_alpha_type == GGML_TYPE_BF16)
        launch_matvec_llama_bf16_batch(r,m->verify_ssm_alpha,cl->ssm_alpha_w,
                                       m->verify_norm,dt,ne,rows);
    else launch_matvec_llama_f16_batch(r,m->verify_ssm_alpha,cl->ssm_alpha_w,
                                       m->verify_norm,dt,ne,rows);
    if (cl->ssm_beta_type == GGML_TYPE_BF16)
        launch_matvec_llama_bf16_batch(r,m->verify_ssm_beta,cl->ssm_beta_w,
                                       m->verify_norm,dt,ne,rows);
    else launch_matvec_llama_f16_batch(r,m->verify_ssm_beta,cl->ssm_beta_w,
                                       m->verify_norm,dt,ne,rows);

    launch_ssm_alpha_beta_prepare_batch(r, m->verify_ssm_alpha,
        m->verify_ssm_beta, cl->ssm_dt_bias, cl->ssm_a, dt, rows);
    launch_conv1d_batch(r, r->d_ssm_conv_out_batch, cl->d_conv_state,
                        m->verify_conv[l], m->verify_ssm_qkv, cl->ssm_conv1d_w,
                        qkv_dim, conv_k, qkv_dim, rows);

    /* Normalize Q/K and broadcast the group heads to dt_rank in one launch.
     * The fused kernel keeps the original per-head reduction and rsqrt order,
     * while avoiding three intermediate launches and round trips. */
    launch_l2_norm_repeat_qk_batch(r, r->d_ssm_Q_exp_batch,
        r->d_ssm_K_exp_batch, r->d_ssm_conv_out_batch,
        n_group, ds, dt, qkv_dim, d_inner, rows, eps);

    float *v = (float *)r->d_ssm_conv_out_batch+(size_t)2*n_group*ds;
    launch_deltanet_step_batch_verify(r, cl->d_recurrent_state, m->verify_rec[l],
        m->verify_ssm_out, r->d_ssm_Q_exp_batch, r->d_ssm_K_exp_batch, v,
        m->verify_ssm_alpha, m->verify_ssm_beta, dt, ds, qkv_dim, rows);
    launch_gated_rmsnorm_silu_batch(r, m->verify_ssm_out, m->verify_ssm_z,
                                    cl->ssm_norm_w, dt, ds, d_inner, rows, eps);

    hllm_dense_mtp_projection(r, m->verify_norm, cl->ssm_out_w, m->verify_ssm_out,
        rows, cl->ssm_out_rows, cl->ssm_out_cols, cl->ssm_out_type);
    const char *split = getenv("LLM_QWEN35_SPLIT_RES_RMSNORM");
    if (split && atoi(split)) {
        for (int i = 0; i < rows; ++i) {
            void *x = (float *)m->verify_x+(size_t)i*ne;
            void *y = (float *)m->verify_norm+(size_t)i*ne;
            launch_add(r, x, y, ne);
            launch_rmsnorm(r, y, x, cl->ffn_norm_w, ne, eps);
        }
    } else {
        hllm_dense_mtp_res_rmsnorm_batch(r, m->verify_x, m->verify_norm,
            m->verify_norm, cl->ffn_norm_w, ne, rows);
    }
}

/* Exact gated-attention verifier with grouped weight projections.  Positions
 * stay device-resident so the captured graph can be replayed at every decode
 * offset.  The per-row RoPE/KV/attention sequence deliberately matches scalar
 * decode; only the four IQ projections share target weight decoding. */
static void hllm_dense_mtp_attention(hip_llm_runner *r, hip_layer *cl,
                                     int l, int rows) {
    hllm_qwen35_mtp *m = r->qwen35_mtp;
    int ne = r->n_embd, qd = r->n_heads*r->head_dim;
    int kd = r->n_kv_heads*r->head_dim;
    float eps = r->rms_norm_eps;
    r->iq1_q8_valid = 0;
    launch_rmsnorm_batch(r, r->d_xnorm_batch, m->verify_x,
                         cl->attn_norm_w, ne, rows, ne, eps);

    hllm_dense_mtp_projection(r, r->d_qfull_batch, cl->attn_q_w,
        r->d_xnorm_batch, rows, cl->attn_q_rows, cl->attn_q_cols,
        cl->attn_q_type);
    launch_deinterleave_qgate_batch(r, r->d_q_batch, r->d_attn_gate_batch,
                                    r->d_qfull_batch, r->n_heads,
                                    r->head_dim, rows);
    hllm_dense_mtp_projection(r, r->d_k_batch, cl->attn_k_w,
        r->d_xnorm_batch, rows, cl->attn_k_rows, cl->attn_k_cols,
        cl->attn_k_type);
    hllm_dense_mtp_projection(r, r->d_v_batch, cl->attn_v_w,
        r->d_xnorm_batch, rows, cl->attn_v_rows, cl->attn_v_cols,
        cl->attn_v_type);
    launch_qknorm_pair_batch(r, r->d_q_batch, cl->attn_q_norm_w,
        r->n_heads, r->d_k_batch, cl->attn_k_norm_w, r->n_kv_heads,
        r->head_dim, qd, kd, rows, eps);

    if (r->use_mrope) {
        int s0=r->mrope_sections[0],s1=r->mrope_sections[1];
        int s2=r->mrope_sections[2],s3=r->mrope_sections[3];
        void *qa[]={&r->d_q_batch,&r->n_heads,&r->head_dim,
            &m->verify_positions,&r->rope_freq_base,&s0,&s1,&s2,&s3,&qd};
        void *ka[]={&r->d_k_batch,&r->n_kv_heads,&r->head_dim,
            &m->verify_positions,&r->rope_freq_base,&s0,&s1,&s2,&s3,&kd};
        LAUNCH(r->fn_rope_mrope_batch_devpos_f32,r->n_heads,rows,1,
               r->head_dim/2,1,1,0,r->stream,qa);
        LAUNCH(r->fn_rope_mrope_batch_devpos_f32,r->n_kv_heads,rows,1,
               r->head_dim/2,1,1,0,r->stream,ka);
        void *sa[]={&r->d_key_cache[l],&r->d_value_cache[l],
            &r->d_key_cache_scale[l],&r->d_value_cache_scale[l],
            &r->d_k_batch,&r->d_v_batch,&r->n_kv_heads,&r->head_dim,
            &m->verify_positions,&rows};
        LAUNCH(r->fn_kv_cache_store_q8q8_positions,rows*r->n_kv_heads,1,1,
               256,1,1,0,r->stream,sa);
    } else for (int i = 0; i < rows; ++i) {
        void *q = (float *)r->d_q_batch+(size_t)i*qd;
        void *k = (float *)r->d_k_batch+(size_t)i*kd;
        void *v = (float *)r->d_v_batch+(size_t)i*kd;
        hipMemcpyAsync(r->d_position, (int *)m->verify_positions+i,
                       sizeof(int), hipMemcpyDeviceToDevice, r->stream);
        launch_rope_devp(r, q, r->n_heads, r->head_dim, r->rope_freq_base);
        launch_rope_devp(r, k, r->n_kv_heads, r->head_dim, r->rope_freq_base);
        void *a[] = { &r->d_key_cache[l], &r->d_value_cache[l],
            &r->d_key_cache_scale[l], &r->d_value_cache_scale[l], &k, &v,
            &r->n_kv_heads, &r->head_dim, &r->d_position };
        LAUNCH(r->fn_kv_cache_store_q8q8_devp, r->n_kv_heads, 1, 1,
               256, 1, 1, 0, r->stream, a);
    }
    /* The verifier chooses this graph only when every causal row has the same
     * adaptive split count. Selector boundaries use the ordinary query grid. */
    launch_attn_verify_native_q8(r, r->d_attn_out_batch,
        m->verify_attn_parts, m->verify_attn_meta, r->d_q_batch,
        r->d_key_cache[l], r->d_value_cache[l], r->d_key_cache_scale[l],
        r->d_value_cache_scale[l], m->verify_positions, r->d_attn_gate_batch,
        rows, m->verify_attention_reuse);
    hllm_dense_mtp_projection(r, r->d_attn_proj_batch,
        cl->attn_output_w, r->d_attn_out_batch, rows,
        cl->attn_output_rows, cl->attn_output_cols, cl->attn_output_type);
    hllm_dense_mtp_res_rmsnorm_batch(r, m->verify_x,
        r->d_attn_proj_batch, m->verify_norm, cl->ffn_norm_w, ne, rows);
}

/* Capture a layer-major window with the target's exact arithmetic contract.
 * Grouped kernels share weights without changing each row's accumulation
 * order. Recurrent state is checkpointed after every row, and positions stay
 * device-resident so graphs remain valid at later context offsets. */
static float *hllm_qwen35_mtp_verify_impl(hip_llm_runner *r,
                                         const int32_t *tokens, int rows,
                                         int position, int32_t *argmax) {
    hllm_qwen35_mtp *m = r ? r->qwen35_mtp : NULL;
    if (!m || m->verify_rows || !tokens || rows < 1 ||
        rows > HLLM_DENSE_MTP_MAX_ROWS || position < 0 ||
        position > r->max_seq_len-rows || !r->decode_mode || r->debug_layers ||
        !r->requested_qwen35_decode_graph || (argmax && !r->fn_qwen4_argmax_batch) ||
        r->kv_cache_type != HIP_LLM_KV_Q8_0_Q8_0 || !r->fn_q8_attention_decode) return NULL;
    for (int i = 0; i < rows; ++i) if (tokens[i] < 0 || tokens[i] >= r->n_vocab) return NULL;
    size_t conv = (size_t)(r->ssm_conv_kernel-1)*r->ssm_qkv_dim*sizeof(float);
    size_t rec = (size_t)r->ssm_dt_rank*r->ssm_d_state*r->ssm_d_state*sizeof(float);
    if (!m->verify_capacity) {
        /* Fixed capacity keeps captured graph pointers stable.  A loaded
         * draft backend uses one fixed verification width for its lifetime. */
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
            hipMalloc(&m->verify_attn_parts, (size_t)capacity*r->n_heads*
                r->q8_attention_max_splits*r->head_dim*sizeof(float)) ||
            hipMalloc(&m->verify_attn_meta, (size_t)capacity*r->n_heads*
                r->q8_attention_max_splits*2*sizeof(float)) ||
            hipMalloc(&m->verify_logits, (size_t)capacity*r->n_vocab*sizeof(float)) ||
            hipMalloc(&m->verify_positions, (size_t)capacity*sizeof(int)) ||
            hipMalloc(&m->verify_argmax, (size_t)capacity*sizeof(int32_t))) return NULL;
        m->host_logits = malloc((size_t)capacity*r->n_vocab*sizeof(float));
        if (!m->host_logits) return NULL;
        void *conv_dst[128], *conv_src[128], *rec_dst[128], *rec_src[128];
        int ssm_layers = 0;
        for (int l = 0; l < r->n_layers; ++l) if (r->layers[l].is_ssm) {
            if (hipMalloc(&m->verify_conv[l], (size_t)capacity*conv) ||
                hipMalloc(&m->verify_rec[l], (size_t)capacity*rec)) return NULL;
            conv_dst[ssm_layers] = r->layers[l].d_conv_state;
            conv_src[ssm_layers] = m->verify_conv[l];
            rec_dst[ssm_layers] = r->layers[l].d_recurrent_state;
            rec_src[ssm_layers] = m->verify_rec[l];
            ++ssm_layers;
        }
        size_t ptr_bytes = (size_t)ssm_layers*sizeof(void *);
        if (hipMalloc(&m->verify_conv_dst_ptrs, ptr_bytes) ||
            hipMalloc(&m->verify_conv_src_ptrs, ptr_bytes) ||
            hipMalloc(&m->verify_rec_dst_ptrs, ptr_bytes) ||
            hipMalloc(&m->verify_rec_src_ptrs, ptr_bytes) ||
            hipMemcpyAsync(m->verify_conv_dst_ptrs, conv_dst, ptr_bytes,
                           hipMemcpyHostToDevice, r->stream) ||
            hipMemcpyAsync(m->verify_conv_src_ptrs, conv_src, ptr_bytes,
                           hipMemcpyHostToDevice, r->stream) ||
            hipMemcpyAsync(m->verify_rec_dst_ptrs, rec_dst, ptr_bytes,
                           hipMemcpyHostToDevice, r->stream) ||
            hipMemcpyAsync(m->verify_rec_src_ptrs, rec_src, ptr_bytes,
                           hipMemcpyHostToDevice, r->stream)) return NULL;
        m->verify_ssm_layers = ssm_layers;
        m->verify_capacity = capacity;
        hllm_vram_sample(r);
    }
    if (rows > m->verify_capacity) return NULL;
    int reuse_attention = rows > 1 && rows <= HLLM_DENSE_MTP_REUSE_ROWS &&
                          r->fn_q8_attention_decode_reuse8;
    if (reuse_attention) {
        int shared_splits = hllm_qwen35_attention_splits(r, position + 1);
        for (int i = 1; reuse_attention && i < rows; ++i) {
            if (hllm_qwen35_attention_splits(r, position + i + 1) !=
                shared_splits) reuse_attention = 0;
        }
    }
    hipGraph_t *selected_graph = reuse_attention ?
        &m->reuse_graphs[rows] : &m->graphs[rows];
    hipGraphExec_t *selected_execution = reuse_attention ?
        &m->reuse_executions[rows] : &m->executions[rows];
    void *saved_x = r->d_x;
    int saved_position = r->cur_position;
    int saved_layer = r->active_layer;
    if (!*selected_execution) {
        m->verify_attention_reuse = reuse_attention;
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
                (cl->ssm_alpha_type == GGML_TYPE_F16 || cl->ssm_alpha_type == GGML_TYPE_BF16) &&
                (cl->ssm_beta_type == GGML_TYPE_F16 || cl->ssm_beta_type == GGML_TYPE_BF16) &&
                r->d_ssm_conv_out_batch && r->d_ssm_Q_exp_batch &&
                r->d_ssm_K_exp_batch && r->fn_deltanet_step_batch_gda_verify_f32;
            int grouped_attn = grouped && !cl->is_ssm &&
                r->d_qfull_batch && r->d_attn_gate_batch &&
                r->d_q_batch && r->d_k_batch && r->d_v_batch &&
                r->d_attn_out_batch && r->d_attn_proj_batch &&
                m->verify_attn_parts && m->verify_attn_meta;
            if (grouped_ssm) hllm_dense_mtp_ssm(r, cl, l, rows, conv, rec);
            else if (grouped_attn) hllm_dense_mtp_attention(r, cl, l, rows);
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
                launch_silu_mul(r,m->verify_gate,m->verify_up,rows*r->n_ff);
                hllm_dense_mtp_projection(r, m->verify_norm, cl->ffn_down_w, m->verify_gate,
                    rows, cl->ffn_down_rows, cl->ffn_down_cols, cl->ffn_down_type);
                launch_add(r, m->verify_x, m->verify_norm, rows*r->n_embd);
            }
        }
        launch_rmsnorm_batch(r, m->verify_x, m->verify_x,
            r->d_output_norm, r->n_embd, rows, r->n_embd,
            r->rms_norm_eps);
        hllm_dense_mtp_projection(r, m->verify_logits, r->d_output_w,
            m->verify_x, rows, r->n_vocab, r->n_embd, r->output_w_type);
        r->d_x = saved_x;
        r->cur_position = saved_position;
        r->active_layer = saved_layer;
        if (hipStreamEndCapture(r->stream, selected_graph) ||
            hipGraphInstantiate(selected_execution, *selected_graph, NULL, NULL, 0)) return NULL;
        m->verify_attention_reuse = 0;
    }
    int positions[HLLM_DENSE_MTP_MAX_ROWS];
    for (int i = 0; i < rows; ++i) positions[i] = position+i;
    if (r->token_embd_type == GGML_TYPE_IQ1_M && r->fn_embed_iq1_m_batch) {
        if (hipMemcpyAsync(m->verify_argmax, tokens, (size_t)rows*sizeof(int32_t),
                           hipMemcpyHostToDevice, r->stream)) return NULL;
        launch_embed_iq1_m_batch(r, m->verify_x, r->d_token_embd,
                                 m->verify_argmax, r->n_embd, rows);
    } else for (int i = 0; i < rows; ++i) {
        void *x = (float *)m->verify_x + (size_t)i*r->n_embd;
        int t = tokens[i];
        if (r->token_embd_type == GGML_TYPE_Q8_0) launch_embed_q8_0(r,x,r->d_token_embd,t,r->n_embd);
        else if (r->token_embd_type == GGML_TYPE_Q4_0) launch_embed_q4_0(r,x,r->d_token_embd,t,r->n_embd);
        else if (r->token_embd_type == GGML_TYPE_Q2_K) launch_embed_q2_K(r,x,r->d_token_embd,t,r->n_embd);
        else if (r->token_embd_type == GGML_TYPE_IQ1_M) launch_embed_iq1_m(r,x,r->d_token_embd,t,r->n_embd);
        else if (r->token_embd_type == GGML_TYPE_F32) launch_embed_f32(r,x,r->d_token_embd,t,r->n_embd);
        else launch_embed(r,x,r->d_token_embd,t,r->n_embd);
    }
    if (hipMemcpyAsync(m->verify_positions, positions, (size_t)rows*sizeof(int),
                       hipMemcpyHostToDevice, r->stream) ||
        hipGraphLaunch(*selected_execution, r->stream)) return NULL;
    if (argmax) {
        void *a[] = { &m->verify_logits, &r->n_vocab, &rows, &m->verify_argmax };
        LAUNCH(r->fn_qwen4_argmax_batch, rows, 1, 1, 256, 1, 1, 0, r->stream, a);
        if (hipMemcpyAsync(argmax, m->verify_argmax, (size_t)rows*sizeof(int32_t),
                           hipMemcpyDeviceToHost, r->stream)) return NULL;
    } else if (hipMemcpyAsync(m->host_logits, m->verify_logits,
                              (size_t)rows*r->n_vocab*sizeof(float),
                              hipMemcpyDeviceToHost, r->stream)) return NULL;
    if (hipStreamSynchronize(r->stream) || r->qwen4_forward_error) return NULL;
    m->verify_rows = rows;
    m->verify_position = position;
    return m->host_logits;
}

float *hip_llm_qwen35_mtp_verify(hip_llm_runner *r, const int32_t *tokens,
                                int rows, int position) {
    return hllm_qwen35_mtp_verify_impl(r, tokens, rows, position, NULL);
}

int hip_llm_qwen35_mtp_verify_argmax(hip_llm_runner *r, const int32_t *tokens,
                                    int rows, int position, int32_t *argmax) {
    if (!argmax) return -1;
    return hllm_qwen35_mtp_verify_impl(r, tokens, rows, position, argmax) ? 0 : -1;
}

int hip_llm_qwen35_mtp_commit(hip_llm_runner *r, int processed) {
    hllm_qwen35_mtp *m = r ? r->qwen35_mtp : NULL;
    if (!m || processed < 1 || processed > m->verify_rows) return -1;
    int last = processed-1;
    size_t conv = (size_t)(r->ssm_conv_kernel-1)*r->ssm_qkv_dim*sizeof(float);
    size_t rec = (size_t)r->ssm_dt_rank*r->ssm_d_state*r->ssm_d_state*sizeof(float);
    size_t conv_elements = conv/sizeof(float), rec_elements = rec/sizeof(float);
    /* Verify snapshots are row-major [token][state], so each row advances
     * by one state vector.  The destination is the live single-row state. */
    size_t conv_stride = conv_elements;
    size_t rec_stride = rec_elements;
    size_t state_vectors = rec_elements > conv_elements ?
                           rec_elements/4 : conv_elements/4;
    const char *fused_env = getenv("LLM_QWEN35_COMMIT_FUSED_COPY");
    int fused_copy = fused_env && atoi(fused_env) != 0 &&
                     r->fn_copy_state_rows_commit_f32;
    if (fused_copy) {
        size_t x_elements = (size_t)r->n_embd;
        size_t logits_elements = (size_t)r->n_vocab;
        size_t x_stride = x_elements, logits_stride = logits_elements;
        void *a[] = { &m->verify_conv_dst_ptrs, &m->verify_conv_src_ptrs,
            &conv_elements, &conv_stride, &m->verify_rec_dst_ptrs,
            &m->verify_rec_src_ptrs, &rec_elements, &rec_stride, &last,
            &m->verify_ssm_layers, &r->d_x, &m->verify_x, &x_elements,
            &x_stride, &r->d_logits, &m->verify_logits, &logits_elements,
            &logits_stride };
        LAUNCH(r->fn_copy_state_rows_commit_f32, (state_vectors+255)/256,
               m->verify_ssm_layers, 1, 256, 1, 1, 0, r->stream, a);
    } else {
        void *state_args[] = { &m->verify_conv_dst_ptrs, &m->verify_conv_src_ptrs,
            &conv_elements, &conv_stride, &m->verify_rec_dst_ptrs,
            &m->verify_rec_src_ptrs, &rec_elements, &rec_stride, &last,
            &m->verify_ssm_layers };
        LAUNCH(r->fn_copy_state_rows_f32, (state_vectors+255)/256,
               m->verify_ssm_layers, 1, 256, 1, 1, 0, r->stream, state_args);
        if (hipMemcpyAsync(r->d_x,(float *)m->verify_x+(size_t)last*r->n_embd,
                           (size_t)r->n_embd*sizeof(float),hipMemcpyDeviceToDevice,r->stream) ||
            hipMemcpyAsync(r->d_logits,(float *)m->verify_logits+(size_t)last*r->n_vocab,
                           (size_t)r->n_vocab*sizeof(float),hipMemcpyDeviceToDevice,r->stream)) return -1;
    }
    r->cur_position = m->verify_position+last;
    if (hipMemcpyAsync(r->d_position,&r->cur_position,sizeof(int),hipMemcpyHostToDevice,r->stream)) return -1;
    /* Commit only enqueues device-to-device publication.  All following
     * decode/propose work uses the same stream, while request reset and the
     * verifier's host-logit path still synchronize at their boundaries. */
    m->verify_rows = 0;
    r->q8x2_reuse_valid = r->iq1_q8_valid = 0;
    return 0;
}

void hip_llm_qwen35_mtp_reset(hip_llm_runner *r) {
    hllm_qwen35_mtp *m = r ? r->qwen35_mtp : NULL;
    if (!m || !m->source) return;
    if (r->stream) hipStreamSynchronize(r->stream);
    m->origin = -1;
    m->kv_end = 0;
    m->pending_token = -1;
    m->verify_rows = 0;
    m->verify_position = 0;
}
