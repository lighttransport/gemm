/* Qwen3.8-27B DFlash2 draft model.
 *
 * The target verifier owns correctness: this sidecar only proposes a path.
 * Target layer-input features are captured into an interleaved row, fused by
 * fc.weight and committed to the draft's private sliding-window KV cache only
 * after the corresponding target rows have been accepted. */

#define HLLM_DFLASH_LAYERS 5
#define HLLM_DFLASH_WINDOW 2048
#define HLLM_DFLASH_MAX_BLOCK 8
#define HLLM_DFLASH_GROUPS 320
#define HLLM_DFLASH_DYNAMIC (HLLM_DFLASH_GROUPS * 4)
#define HLLM_DFLASH_RANK 256
#define HLLM_DFLASH_TOPK 16
#define HLLM_DFLASH_HEADS 32
#define HLLM_DFLASH_KV_HEADS 8
#define HLLM_DFLASH_HEAD_DIM 128
#define HLLM_DFLASH_ATTN_ROWS_PER_WAVE 4
#define HLLM_DFLASH_ATTN_MAX_SPLITS 16

typedef struct hllm_dflash_layer {
    void *attn_norm, *q_norm, *k_norm, *ffn_norm;
    void *q, *k, *v, *o, *gate, *up, *down;
    int q_type, k_type, v_type, o_type, gate_type, up_type, down_type;
    void *attn_conv_base, *attn_conv_proj;
    void *ffn_conv_base, *ffn_conv_proj;
    int attn_conv_proj_type, ffn_conv_proj_type;
    void *key_cache, *value_cache;
    /* Prompt injection is a large-M operation.  Keep only its K/V weights in
     * BF16; draft decode continues to use the authoritative quantized
     * matrices above. */
    void *inject_k_bf16, *inject_v_bf16;
} hllm_dflash_layer;

typedef struct hllm_qwen35_dflash2 {
    gguf_shards *source;
    hipModule_t module;
    hipFunction_t fn_capture, fn_conv, fn_attention, fn_attention_combine;
    hipFunction_t fn_topk, fn_select;
    void *fc, *fc_bf16, *enc_norm, *out_norm, *selector_hidden;
    int fc_type, selector_hidden_type;
    qtensor selector_prev, selector_next;
    void *selector_prev_w, *selector_next_w;
    int selector_prev_type, selector_next_type;
    int target_layers[HLLM_DFLASH_LAYERS];
    int mask_token, feature_rows, kv_end;
    hllm_dflash_layer layers[HLLM_DFLASH_LAYERS];

    /* All activation storage is row-major F32. */
    void *features, *features_bf16, *x, *x_bf16, *norm, *dynamic, *conv;
    void *q, *k, *v, *attn, *attn_partial, *proj, *gate, *up;
    void *logits, *selector_gate;
    void *selector_candidates, *selector_drafts;
    void *q81_source;
    int q81_rows, q81_cols;
} hllm_qwen35_dflash2;

static void hllm_qwen35_dflash2_free(hip_llm_runner *r) {
    hllm_qwen35_dflash2 *d = r ? r->qwen35_dflash2 : NULL;
    if (!d) return;
    if (r->stream) hipStreamSynchronize(r->stream);
#define DFLASH_FREE(p) do { if (p) hipFree(p); } while (0)
    DFLASH_FREE(d->fc); DFLASH_FREE(d->fc_bf16);
    DFLASH_FREE(d->enc_norm); DFLASH_FREE(d->out_norm);
    DFLASH_FREE(d->selector_hidden);
    DFLASH_FREE(d->selector_prev_w); DFLASH_FREE(d->selector_next_w);
    DFLASH_FREE(d->features); DFLASH_FREE(d->features_bf16);
    DFLASH_FREE(d->x); DFLASH_FREE(d->x_bf16); DFLASH_FREE(d->norm);
    DFLASH_FREE(d->dynamic); DFLASH_FREE(d->conv); DFLASH_FREE(d->q);
    DFLASH_FREE(d->k); DFLASH_FREE(d->v); DFLASH_FREE(d->attn);
    DFLASH_FREE(d->attn_partial);
    DFLASH_FREE(d->proj); DFLASH_FREE(d->gate); DFLASH_FREE(d->up);
    DFLASH_FREE(d->logits); DFLASH_FREE(d->selector_gate);
    DFLASH_FREE(d->selector_candidates); DFLASH_FREE(d->selector_drafts);
    for (int l = 0; l < HLLM_DFLASH_LAYERS; ++l) {
        hllm_dflash_layer *cl = &d->layers[l];
        DFLASH_FREE(cl->attn_norm); DFLASH_FREE(cl->q_norm);
        DFLASH_FREE(cl->k_norm); DFLASH_FREE(cl->ffn_norm);
        DFLASH_FREE(cl->q); DFLASH_FREE(cl->k); DFLASH_FREE(cl->v);
        DFLASH_FREE(cl->o); DFLASH_FREE(cl->gate); DFLASH_FREE(cl->up);
        DFLASH_FREE(cl->down); DFLASH_FREE(cl->attn_conv_base);
        DFLASH_FREE(cl->attn_conv_proj); DFLASH_FREE(cl->ffn_conv_base);
        DFLASH_FREE(cl->ffn_conv_proj); DFLASH_FREE(cl->key_cache);
        DFLASH_FREE(cl->value_cache); DFLASH_FREE(cl->inject_k_bf16);
        DFLASH_FREE(cl->inject_v_bf16);
    }
#undef DFLASH_FREE
    if (d->module) hipModuleUnload(d->module);
    if (d->source) gguf_close_shards(d->source);
    free(d);
    r->qwen35_dflash2 = NULL;
}

static int hllm_dflash_raw_f32(void **dst, const qtensor *t, size_t elements) {
    if (!t->data || t->type != GGML_TYPE_F32 ||
        (size_t)t->n_rows * t->n_cols != elements) return -1;
    size_t bytes = elements * sizeof(float);
    if (hipMalloc(dst, bytes) != hipSuccess) return -1;
    if (hipMemcpy(*dst, t->data, bytes, hipMemcpyHostToDevice) != hipSuccess) {
        hipFree(*dst); *dst = NULL; return -1;
    }
    return 0;
}

static int hllm_dflash_tap(const hllm_qwen35_dflash2 *d, int layer) {
    for (int i = 0; i < HLLM_DFLASH_LAYERS; ++i)
        if (d->target_layers[i] == layer) return i;
    return -1;
}

static void hllm_qwen35_dflash2_capture(hip_llm_runner *r, int layer,
                                        void *hidden, int rows) {
    hllm_qwen35_dflash2 *d = r ? r->qwen35_dflash2 : NULL;
    if (!d || !hidden || rows < 1 || rows > r->batch_max) return;
    int tap = hllm_dflash_tap(d, layer);
    if (tap < 0) return;
    int total = rows * r->n_embd;
    int taps = HLLM_DFLASH_LAYERS;
    void *a[] = { &d->features, &hidden, &rows, &r->n_embd, &tap, &taps };
    LAUNCH(d->fn_capture, (total + 255) / 256, 1, 1, 256, 1, 1, 0,
           r->stream, a);
    d->feature_rows = rows;
}

static void hllm_dflash_project(hip_llm_runner *r, void *dst, void *weight,
        void *x, int rows, int nr, int nc, int stride, int type) {
    hllm_qwen35_dflash2 *d=r->qwen35_dflash2;
    int q4_q81=type==GGML_TYPE_Q4_K && rows>1 &&
        rows<=HLLM_DFLASH_MAX_BLOCK && stride==nc && nc%256==0 &&
        r->fn_qwen35_quantize_q81 && r->fn_qwen35_matvec_q4k_q81_multi8;
    if(!q4_q81)d->q81_source=NULL;
    if (type == GGML_TYPE_Q2_K && rows > 1 &&
        rows <= HLLM_DENSE_MTP_REUSE_ROWS && stride == nc && nc <= 6144 &&
        nc % 256 == 0 && r->fn_qwen35_matvec_q2k_multi4) {
        hllm_dense_mtp_projection(r,dst,weight,x,rows,nr,nc,type);
    } else if (type == GGML_TYPE_Q6_K && rows > 1 && stride == nc &&
               r->fn_matvec_q6_K_batch_reuse8) {
        void *a[]={&dst,&weight,&x,&rows,&nr,&nc,&stride};
        LAUNCH(r->fn_matvec_q6_K_batch_reuse8,nr,(rows+7)/8,1,
               64,1,1,0,r->stream,a);
    } else if (type == GGML_TYPE_IQ4_XS &&
        rows >= HLLM_DFLASH_ATTN_ROWS_PER_WAVE &&
        rows <= HLLM_DFLASH_MAX_BLOCK && stride == nc &&
        nc % 256 == 0 && r->fn_qwen35_quantize_q81 &&
        r->fn_qwen35_matvec_iq4xs_multi8) {
        int total = rows*nc;
        void *qa[] = { &r->d_act_q8_batch, &r->d_act_scale_batch, &x, &total };
        LAUNCH(r->fn_qwen35_quantize_q81, total/32, 1, 1, 32, 1, 1, 0,
               r->stream, qa);
        void *ma[] = { &dst, &weight, &r->d_act_q8_batch,
                       &r->d_act_scale_batch, &nr, &nc, &rows };
        hipFunction_t iq4_fn = nc == 5120 ?
            r->fn_qwen35_matvec_iq4xs_5120_multi8 :
            r->fn_qwen35_matvec_iq4xs_multi8;
        LAUNCH(iq4_fn, (nr+7)/8, 1, 1,
               256, 1, 1, 0, r->stream, ma);
    } else if (q4_q81) {
        if(d->q81_source!=x||d->q81_rows!=rows||d->q81_cols!=nc){
            int total=rows*nc;
            void *qa[]={&r->d_act_q8_batch,&r->d_act_scale_batch,&x,&total};
            LAUNCH(r->fn_qwen35_quantize_q81,total/32,1,1,32,1,1,0,
                   r->stream,qa);
            d->q81_source=x;d->q81_rows=rows;d->q81_cols=nc;
        }
        void *ma[]={&dst,&weight,&r->d_act_q8_batch,
                    &r->d_act_scale_batch,&nr,&nc,&rows};
        LAUNCH(r->fn_qwen35_matvec_q4k_q81_multi8,(nr+7)/8,1,1,
               256,1,1,0,r->stream,ma);
    } else if (type == GGML_TYPE_Q4_K && rows > 1) {
        launch_matvec_qwen35_native_batch(r,dst,weight,x,rows,nr,nc,
                                          stride,type);
    } else {
        for (int i = 0; i < rows; ++i)
            launch_matvec_auto(r, (float *)dst + (size_t)i * nr, weight,
                (float *)x + (size_t)i * stride, nr, nc, type);
    }
    r->q8x2_reuse_valid = r->iq1_q8_valid = r->batch_q8_valid = 0;
}

static int hllm_qwen35_dflash2_inject(hip_llm_runner *r, int position,
                                      int rows) {
    hllm_qwen35_dflash2 *d = r ? r->qwen35_dflash2 : NULL;
    if (!d || rows < 1 || rows > d->feature_rows) return -1;
    int ne = r->n_embd, kd = HLLM_DFLASH_KV_HEADS * HLLM_DFLASH_HEAD_DIM;
    if (d->fc_bf16 && d->features_bf16) {
        int feature_dim = HLLM_DFLASH_LAYERS * ne;
        launch_pack_bf16_from_f32(r, d->features_bf16, d->features,
                                  rows * feature_dim);
        if (gemm_run_bf16_w(r, d->x, d->fc_bf16, d->features_bf16,
                            rows, ne, feature_dim, r->stream) != 0) return -1;
    } else {
        hllm_dflash_project(r, d->x, d->fc, d->features, rows, ne,
                            HLLM_DFLASH_LAYERS * ne,
                            HLLM_DFLASH_LAYERS * ne, d->fc_type);
    }
    launch_rmsnorm_batch(r, d->x, d->x, d->enc_norm, ne, rows, ne,
                         r->rms_norm_eps);
    if (d->x_bf16)
        launch_pack_bf16_from_f32(r, d->x_bf16, d->x, rows * ne);
    for (int l = 0; l < HLLM_DFLASH_LAYERS; ++l) {
        hllm_dflash_layer *cl = &d->layers[l];
        if (d->x_bf16 && cl->inject_k_bf16 && cl->inject_v_bf16) {
            if (gemm_run_bf16_w(r, d->k, cl->inject_k_bf16, d->x_bf16,
                                rows, kd, ne, r->stream) != 0 ||
                gemm_run_bf16_w(r, d->v, cl->inject_v_bf16, d->x_bf16,
                                rows, kd, ne, r->stream) != 0) return -1;
        } else {
            hllm_dflash_project(r, d->k, cl->k, d->x, rows, kd, ne, ne,
                                cl->k_type);
            hllm_dflash_project(r, d->v, cl->v, d->x, rows, kd, ne, ne,
                                cl->v_type);
        }
        launch_qknorm_batch(r, d->k, cl->k_norm, HLLM_DFLASH_KV_HEADS,
                            HLLM_DFLASH_HEAD_DIM, rows, kd, r->rms_norm_eps);
        launch_rope_mrope_batch(r, d->k, HLLM_DFLASH_KV_HEADS, HLLM_DFLASH_HEAD_DIM,
            position, r->rope_freq_base, HLLM_DFLASH_HEAD_DIM / 2, 0, 0, 0,
            kd, rows);
        launch_kv_store_batch_strided(r, cl->key_cache, cl->value_cache,
            d->k, d->v, position, rows, kd, kd, HLLM_DFLASH_WINDOW);
    }
    d->kv_end = position + rows;
    return r->qwen4_forward_error ? -1 : 0;
}

static int hllm_dflash_upload_matrix(const gguf_context *g, const char *name,
        void **dst, int *type, int rows, int cols) {
    qtensor t = hllm_load_tensor(g, name, 1);
    return (!t.data || t.n_rows != rows || t.n_cols != cols ||
            upload_weight_matrix(dst, &t, type)) ? -1 : 0;
}

static int hllm_dflash_upload_norm(const gguf_context *g, const char *name,
        void **dst, int n) {
    qtensor t = hllm_load_tensor(g, name, 1);
    return (!t.data || (size_t)t.n_rows * t.n_cols != (size_t)n ||
            upload_norm_f32(dst, &t, n)) ? -1 : 0;
}

static int hllm_dflash_dequant_bf16(hip_llm_runner *r, void *dst, void *src,
                                    int type, int rows, int cols) {
    if (type == GGML_TYPE_Q4_K)
        return launch_dequant_q4_K_to_bf16(r, dst, src, rows, cols);
    if (type == GGML_TYPE_Q6_K)
        return launch_dequant_q6_K_to_bf16(r, dst, src, rows, cols);
    return -1;
}

int hip_llm_qwen35_dflash2_load(hip_llm_runner *r, const char *path,
                                char *error, size_t error_cap) {
    const gguf_shards *saved = hllm_active_shards;
    if (error && error_cap) snprintf(error, error_cap, "unsupported target/configuration");
    if (!r || !path || !r->weights_loaded || !r->is_hybrid || r->is_moe ||
        r->is_qwen4exp || r->qwen35_dflash2 || r->qwen35_mtp ||
        r->n_embd != 5120 || r->n_ff != 17408 || r->n_vocab != 248320 ||
        r->batch_max < HLLM_DFLASH_MAX_BLOCK) return -1;
    hllm_qwen35_dflash2 *d = calloc(1, sizeof(*d));
    if (!d) return -1;
    r->qwen35_dflash2 = d;
    d->source = gguf_open_shards(path, 2);
    if (!d->source) goto fail;
    hllm_active_shards = d->source;
    gguf_context *g = d->source->metadata;
    if (hllm_get_int(g, "dflash.block_count", 0) != HLLM_DFLASH_LAYERS ||
        hllm_get_int(g, "dflash.embedding_length", 0) != r->n_embd ||
        hllm_get_int(g, "dflash.feed_forward_length", 0) != r->n_ff ||
        hllm_get_int(g, "dflash.attention.head_count", 0) != HLLM_DFLASH_HEADS ||
        hllm_get_int(g, "dflash.attention.head_count_kv", 0) != HLLM_DFLASH_KV_HEADS ||
        hllm_get_int(g, "dflash.attention.key_length", 0) != HLLM_DFLASH_HEAD_DIM ||
        hllm_get_int(g, "dflash.attention.value_length", 0) != HLLM_DFLASH_HEAD_DIM ||
        hllm_get_int(g, "dflash.block_size", 0) != HLLM_DFLASH_MAX_BLOCK ||
        hllm_get_int(g, "dflash.conv_kernel_size", 0) != 2 ||
        hllm_get_int(g, "dflash.conv_group_size", 0) != 16 ||
        hllm_get_int(g, "dflash.selector_rank", 0) != HLLM_DFLASH_RANK ||
        hllm_get_int(g, "dflash.selector_top_k", 0) != HLLM_DFLASH_TOPK ||
        hllm_get_int(g, "dflash.attention.sliding_window", 0) != HLLM_DFLASH_WINDOW)
        goto fail;
    int target_idx = gguf_find_key(g, "dflash.target_layers");
    if (target_idx < 0 || g->kv[target_idx].type != GGUF_TYPE_ARRAY ||
        g->kv[target_idx].value.arr.type != GGUF_TYPE_INT32 ||
        g->kv[target_idx].value.arr.n != HLLM_DFLASH_LAYERS) goto fail;
    memcpy(d->target_layers, g->kv[target_idx].value.arr.data,
           sizeof(d->target_layers));
    d->mask_token = hllm_get_int(g, "tokenizer.ggml.mask_token_id", -1);
    if (d->mask_token < 0 || d->mask_token >= r->n_vocab) goto fail;

    if (hip_compile_kernels_ex(&d->module, r->device, qwen35_dflash2_source,
            "qwen35_dflash2.hip", r->verbose, "qwen35_dflash2", 0) <= 0 ||
        hipModuleGetFunction(&d->fn_capture, d->module,
                             "qwen35_dflash2_capture") != hipSuccess ||
        hipModuleGetFunction(&d->fn_conv, d->module,
                             "qwen35_dflash2_conv") != hipSuccess ||
        hipModuleGetFunction(&d->fn_attention, d->module,
                             "qwen35_dflash2_attention") != hipSuccess ||
        hipModuleGetFunction(&d->fn_attention_combine, d->module,
                             "qwen35_dflash2_attention_combine") != hipSuccess ||
        hipModuleGetFunction(&d->fn_topk, d->module,
                             "qwen35_dflash2_topk") != hipSuccess ||
        hipModuleGetFunction(&d->fn_select, d->module,
                             "qwen35_dflash2_select") != hipSuccess) goto fail;

    if (hllm_dflash_upload_matrix(g, "fc.weight", &d->fc, &d->fc_type,
            r->n_embd, HLLM_DFLASH_LAYERS*r->n_embd) ||
        hllm_dflash_upload_norm(g, "enc.output_norm.weight", &d->enc_norm,
                               r->n_embd) ||
        hllm_dflash_upload_norm(g, "output_norm.weight", &d->out_norm,
                               r->n_embd) ||
        hllm_dflash_upload_matrix(g, "selector_hidden.weight",
            &d->selector_hidden, &d->selector_hidden_type,
            HLLM_DFLASH_RANK, r->n_embd)) goto fail;
    d->selector_prev = hllm_load_tensor(g, "selector_predecessor.weight", 1);
    d->selector_next = hllm_load_tensor(g, "selector_successor.weight", 1);
    if (!d->selector_prev.data || !d->selector_next.data ||
        d->selector_prev.n_rows != r->n_vocab ||
        d->selector_next.n_rows != r->n_vocab ||
        d->selector_prev.n_cols != HLLM_DFLASH_RANK ||
        d->selector_next.n_cols != HLLM_DFLASH_RANK) goto fail;
    if (upload_weight_matrix(&d->selector_prev_w, &d->selector_prev,
                             &d->selector_prev_type) ||
        upload_weight_matrix(&d->selector_next_w, &d->selector_next,
                             &d->selector_next_type) ||
        d->selector_prev_type != GGML_TYPE_Q4_K ||
        d->selector_next_type != GGML_TYPE_Q4_K) goto fail;

    char name[128];
    for (int l = 0; l < HLLM_DFLASH_LAYERS; ++l) {
        hllm_dflash_layer *cl = &d->layers[l];
#define DNORM(field, suffix, n) do { snprintf(name,sizeof(name),"blk.%d." suffix,l); \
        if (hllm_dflash_upload_norm(g,name,&cl->field,(n))) goto fail; } while (0)
#define DMAT(field, suffix, nr, nc) do { snprintf(name,sizeof(name),"blk.%d." suffix ".weight",l); \
        if (hllm_dflash_upload_matrix(g,name,&cl->field,&cl->field##_type,(nr),(nc))) goto fail; } while (0)
        DNORM(attn_norm, "attn_norm.weight", r->n_embd);
        DNORM(q_norm, "attn_q_norm.weight", HLLM_DFLASH_HEAD_DIM);
        DNORM(k_norm, "attn_k_norm.weight", HLLM_DFLASH_HEAD_DIM);
        DNORM(ffn_norm, "ffn_norm.weight", r->n_embd);
        DMAT(q, "attn_q", HLLM_DFLASH_HEADS*HLLM_DFLASH_HEAD_DIM, r->n_embd);
        DMAT(k, "attn_k", HLLM_DFLASH_KV_HEADS*HLLM_DFLASH_HEAD_DIM, r->n_embd);
        DMAT(v, "attn_v", HLLM_DFLASH_KV_HEADS*HLLM_DFLASH_HEAD_DIM, r->n_embd);
        DMAT(o, "attn_output", r->n_embd, HLLM_DFLASH_HEADS*HLLM_DFLASH_HEAD_DIM);
        DMAT(gate, "ffn_gate", r->n_ff, r->n_embd);
        DMAT(up, "ffn_up", r->n_ff, r->n_embd);
        DMAT(down, "ffn_down", r->n_embd, r->n_ff);
        DMAT(attn_conv_proj, "attn_conv_proj", HLLM_DFLASH_DYNAMIC, r->n_embd);
        DMAT(ffn_conv_proj, "ffn_conv_proj", HLLM_DFLASH_DYNAMIC, r->n_embd);
#undef DMAT
#undef DNORM
        snprintf(name,sizeof(name),"blk.%d.attn_conv_base",l);
        qtensor cb = hllm_load_tensor(g,name,1);
        if (hllm_dflash_raw_f32(&cl->attn_conv_base,&cb,(size_t)r->n_embd*4)) goto fail;
        snprintf(name,sizeof(name),"blk.%d.ffn_conv_base",l);
        cb = hllm_load_tensor(g,name,1);
        if (hllm_dflash_raw_f32(&cl->ffn_conv_base,&cb,(size_t)r->n_embd*4)) goto fail;
        size_t cache = (size_t)HLLM_DFLASH_WINDOW*HLLM_DFLASH_KV_HEADS*HLLM_DFLASH_HEAD_DIM*sizeof(float);
        if (hipMalloc(&cl->key_cache,cache) || hipMalloc(&cl->value_cache,cache) ||
            hipMemset(cl->key_cache,0,cache) || hipMemset(cl->value_cache,0,cache)) goto fail;
    }

    size_t cap = r->batch_max, ne = r->n_embd, ff = r->n_ff;
#define DALLOC(field, count) do { if (hipMalloc(&d->field,(size_t)(count)*sizeof(float))) goto fail; } while (0)
    DALLOC(features, cap*HLLM_DFLASH_LAYERS*ne);
    DALLOC(x, cap*ne); DALLOC(norm, cap*ne); DALLOC(dynamic, cap*HLLM_DFLASH_DYNAMIC);
    DALLOC(conv, cap*ne); DALLOC(q, cap*HLLM_DFLASH_HEADS*HLLM_DFLASH_HEAD_DIM);
    DALLOC(k, cap*HLLM_DFLASH_KV_HEADS*HLLM_DFLASH_HEAD_DIM); DALLOC(v, cap*HLLM_DFLASH_KV_HEADS*HLLM_DFLASH_HEAD_DIM);
    DALLOC(attn, cap*HLLM_DFLASH_HEADS*HLLM_DFLASH_HEAD_DIM);
    DALLOC(attn_partial, HLLM_DFLASH_ATTN_MAX_SPLITS*HLLM_DFLASH_MAX_BLOCK*
            HLLM_DFLASH_HEADS*
            (HLLM_DFLASH_HEAD_DIM+2));
    DALLOC(proj, cap*ne);
    DALLOC(gate, cap*ff); DALLOC(up, cap*ff);
    DALLOC(logits, HLLM_DFLASH_MAX_BLOCK*r->n_vocab);
    DALLOC(selector_gate, HLLM_DFLASH_MAX_BLOCK*HLLM_DFLASH_RANK);
#undef DALLOC
    /* The DFlash prompt cache is populated in 512-row tiles.  Its fusion
     * projection is 5x wider than an ordinary model projection, and the
     * decode-oriented Q4_K reuse kernel rereads that matrix for every tile
     * row.  Dequantize the prompt-only fusion and K/V weights once so the
     * existing large-M BF16 GEMM path can reuse them.  Proposal decode keeps
     * the compact quantized weights and exact small-row kernels. */
    if (d->fc_type == GGML_TYPE_Q4_K) {
        size_t fc_elems = (size_t)ne * HLLM_DFLASH_LAYERS * ne;
        if (hipMalloc(&d->fc_bf16, fc_elems * sizeof(uint16_t)) ||
            launch_dequant_q4_K_to_bf16(r, d->fc_bf16, d->fc, ne,
                                        HLLM_DFLASH_LAYERS * ne) ||
            hipMalloc(&d->features_bf16,
                      cap * HLLM_DFLASH_LAYERS * ne * sizeof(uint16_t)) ||
            hipMalloc(&d->x_bf16, cap * ne * sizeof(uint16_t))) goto fail;
        for (int l = 0; l < HLLM_DFLASH_LAYERS; ++l) {
            hllm_dflash_layer *cl = &d->layers[l];
            size_t kv_elems = (size_t)HLLM_DFLASH_KV_HEADS *
                              HLLM_DFLASH_HEAD_DIM * ne;
            if (hipMalloc(&cl->inject_k_bf16, kv_elems * sizeof(uint16_t)) ||
                hipMalloc(&cl->inject_v_bf16, kv_elems * sizeof(uint16_t)) ||
                hllm_dflash_dequant_bf16(r, cl->inject_k_bf16, cl->k,
                    cl->k_type, HLLM_DFLASH_KV_HEADS * HLLM_DFLASH_HEAD_DIM,
                    ne) ||
                hllm_dflash_dequant_bf16(r, cl->inject_v_bf16, cl->v,
                    cl->v_type, HLLM_DFLASH_KV_HEADS * HLLM_DFLASH_HEAD_DIM,
                    ne)) goto fail;
        }
        if (hipStreamSynchronize(r->stream) != hipSuccess) goto fail;
        if (r->verbose >= 1)
            fprintf(stderr,
                "DFlash2 prompt injection: cached %.1f MiB BF16 weights\n",
                (double)(fc_elems + 2 * HLLM_DFLASH_LAYERS *
                    (size_t)HLLM_DFLASH_KV_HEADS * HLLM_DFLASH_HEAD_DIM * ne) *
                    sizeof(uint16_t) / (1024.0 * 1024.0));
    }
    if (hipMalloc(&d->selector_candidates,
            HLLM_DFLASH_MAX_BLOCK*HLLM_DFLASH_TOPK*sizeof(int)) ||
        hipMalloc(&d->selector_drafts,
            (HLLM_DFLASH_MAX_BLOCK-1)*sizeof(int)) ||
        hllm_qwen35_verify_workspace_create(r)) goto fail;
    hllm_active_shards = saved;
    if (error && error_cap) error[0] = 0;
    hllm_vram_sample(r);
    return 0;
fail:
    hllm_active_shards = saved;
    if (error && error_cap) snprintf(error,error_cap,
        "invalid DFlash2 sidecar schema or allocation failure");
    hllm_qwen35_dflash2_free(r);
    if (r && r->qwen35_mtp && !r->qwen35_mtp->source) hllm_free_qwen35_mtp(r);
    return -1;
}

static void hllm_dflash_conv(hip_llm_runner *r, hllm_qwen35_dflash2 *d,
        void *out, void *hidden, void *dynamic, void *base, int rows, int side) {
    int total = rows*r->n_embd, groups = HLLM_DFLASH_GROUPS;
    void *a[] = { &out, &hidden, &dynamic, &base, &rows, &r->n_embd,
                  &groups, &side };
    LAUNCH(d->fn_conv,(total+255)/256,1,1,256,1,1,0,r->stream,a);
    d->q81_source=NULL;
}

int hip_llm_qwen35_dflash2_propose(hip_llm_runner *r, int32_t anchor,
        int position, int count, int32_t *drafts) {
    hllm_qwen35_dflash2 *d = r ? r->qwen35_dflash2 : NULL;
    hllm_qwen35_mtp *m = r ? r->qwen35_mtp : NULL;
    if (!d || !m || m->verify_rows || !drafts || count < 1 || count > 7 ||
        anchor < 0 || anchor >= r->n_vocab || position != d->kv_end) {
        fprintf(stderr,
            "DFlash2 propose rejected: d=%d verifier=%d drafts=%d count=%d "
            "anchor=%d position=%d kv_end=%d\n",
            d != NULL, m ? m->verify_rows : -1, drafts != NULL, count,
            anchor, position, d ? d->kv_end : -1);
        return -1;
    }
    int rows=count+1, ne=r->n_embd, qd=HLLM_DFLASH_HEADS*HLLM_DFLASH_HEAD_DIM;
    int kd=HLLM_DFLASH_KV_HEADS*HLLM_DFLASH_HEAD_DIM;
    d->q81_source=NULL;
    launch_embed_iq1_m(r,d->x,r->d_token_embd,anchor,ne);
    for (int i=1;i<rows;++i)
        launch_embed_iq1_m(r,(float *)d->x+(size_t)i*ne,r->d_token_embd,d->mask_token,ne);
    for (int l=0;l<HLLM_DFLASH_LAYERS;++l) {
        hllm_dflash_layer *cl=&d->layers[l];
        launch_rmsnorm_batch(r,d->norm,d->x,cl->attn_norm,ne,rows,ne,r->rms_norm_eps);
        d->q81_source=NULL;
        hllm_dflash_project(r,d->dynamic,cl->attn_conv_proj,d->norm,rows,
            HLLM_DFLASH_DYNAMIC,ne,ne,cl->attn_conv_proj_type);
        hllm_dflash_conv(r,d,d->conv,d->norm,d->dynamic,cl->attn_conv_base,rows,0);
        hllm_dflash_project(r,d->q,cl->q,d->conv,rows,qd,ne,ne,cl->q_type);
        hllm_dflash_project(r,d->k,cl->k,d->conv,rows,kd,ne,ne,cl->k_type);
        hllm_dflash_project(r,d->v,cl->v,d->conv,rows,kd,ne,ne,cl->v_type);
        launch_qknorm_batch(r,d->q,cl->q_norm,HLLM_DFLASH_HEADS,HLLM_DFLASH_HEAD_DIM,rows,qd,r->rms_norm_eps);
        launch_qknorm_batch(r,d->k,cl->k_norm,HLLM_DFLASH_KV_HEADS,HLLM_DFLASH_HEAD_DIM,rows,kd,r->rms_norm_eps);
        launch_rope_mrope_batch(r,d->q,HLLM_DFLASH_HEADS,HLLM_DFLASH_HEAD_DIM,position,
            r->rope_freq_base,HLLM_DFLASH_HEAD_DIM/2,0,0,0,qd,rows);
        launch_rope_mrope_batch(r,d->k,HLLM_DFLASH_KV_HEADS,HLLM_DFLASH_HEAD_DIM,position,
            r->rope_freq_base,HLLM_DFLASH_HEAD_DIM/2,0,0,0,kd,rows);
        launch_kv_store_batch_strided(r,cl->key_cache,cl->value_cache,d->k,d->v,
            position,rows,kd,kd,HLLM_DFLASH_WINDOW);
        int window=HLLM_DFLASH_WINDOW;
        int attention_length=position+rows < window ? position+rows : window;
        int splits=attention_length >= 1024 ? 16 : attention_length >= 512 ? 4 : 1;
        void *aa[]={&d->attn_partial,&d->q,&cl->key_cache,&cl->value_cache,&rows,
            &position,&(int){HLLM_DFLASH_HEADS},&(int){HLLM_DFLASH_KV_HEADS},
            &(int){HLLM_DFLASH_HEAD_DIM},&window,&splits};
        LAUNCH(d->fn_attention, HLLM_DFLASH_HEADS,
               (rows + HLLM_DFLASH_ATTN_ROWS_PER_WAVE - 1) /
                   HLLM_DFLASH_ATTN_ROWS_PER_WAVE, splits,
               32,1,1,0,r->stream,aa);
        void *ac[]={&d->attn,&d->attn_partial,&rows,
            &(int){HLLM_DFLASH_HEADS},&(int){HLLM_DFLASH_HEAD_DIM},&splits};
        LAUNCH(d->fn_attention_combine,HLLM_DFLASH_HEADS,rows,1,
               32,1,1,0,r->stream,ac);
        hllm_dflash_project(r,d->proj,cl->o,d->attn,rows,ne,qd,qd,cl->o_type);
        hllm_dflash_conv(r,d,d->conv,d->proj,d->dynamic,cl->attn_conv_base,rows,1);
        launch_add(r,d->x,d->conv,rows*ne);
        launch_rmsnorm_batch(r,d->norm,d->x,cl->ffn_norm,ne,rows,ne,r->rms_norm_eps);
        d->q81_source=NULL;
        hllm_dflash_project(r,d->dynamic,cl->ffn_conv_proj,d->norm,rows,
            HLLM_DFLASH_DYNAMIC,ne,ne,cl->ffn_conv_proj_type);
        hllm_dflash_conv(r,d,d->conv,d->norm,d->dynamic,cl->ffn_conv_base,rows,0);
        hllm_dflash_project(r,d->gate,cl->gate,d->conv,rows,r->n_ff,ne,ne,cl->gate_type);
        hllm_dflash_project(r,d->up,cl->up,d->conv,rows,r->n_ff,ne,ne,cl->up_type);
        launch_silu_mul(r,d->gate,d->up,rows*r->n_ff);
        hllm_dflash_project(r,d->proj,cl->down,d->gate,rows,ne,r->n_ff,r->n_ff,cl->down_type);
        hllm_dflash_conv(r,d,d->conv,d->proj,d->dynamic,cl->ffn_conv_base,rows,1);
        launch_add(r,d->x,d->conv,rows*ne);
    }
    launch_rmsnorm_batch(r,d->norm,d->x,d->out_norm,ne,rows,ne,r->rms_norm_eps);
    d->q81_source=NULL;
    /* The anchor row seeds the selector's predecessor but never contributes
     * logits or a selector gate.  Keep row-indexed buffers so the top-k and
     * selector kernels retain their established indexing, and project only
     * the K mask rows that they consume. */
    int selected_rows=rows-1;
    void *selected_norm=(float *)d->norm+ne;
    void *selected_logits=(float *)d->logits+r->n_vocab;
    void *selected_gate=(float *)d->selector_gate+HLLM_DFLASH_RANK;
    hllm_dflash_project(r,selected_logits,r->d_output_w,selected_norm,
        selected_rows,r->n_vocab,ne,ne,r->output_w_type);
    hllm_dflash_project(r,selected_gate,d->selector_hidden,selected_norm,
        selected_rows,HLLM_DFLASH_RANK,ne,ne,d->selector_hidden_type);
    void *ta[]={&d->logits,&d->selector_candidates,&rows,&r->n_vocab};
    LAUNCH(d->fn_topk,rows-1,1,1,256,1,1,0,r->stream,ta);
    void *sa[]={&d->logits,&d->selector_gate,&d->selector_prev_w,
        &d->selector_next_w,&d->selector_candidates,&d->selector_drafts,
        &anchor,&rows,&r->n_vocab};
    LAUNCH(d->fn_select,1,1,1,32,1,1,0,r->stream,sa);
    hipError_t copy_error = hipMemcpyAsync(drafts, d->selector_drafts,
        (size_t)count * sizeof(int), hipMemcpyDeviceToHost, r->stream);
    hipError_t sync_error = copy_error == hipSuccess ?
        hipStreamSynchronize(r->stream) : hipSuccess;
    if (copy_error != hipSuccess || sync_error != hipSuccess ||
        r->qwen4_forward_error) {
        const char *copy_string = "unknown", *sync_string = "unknown";
        if (hipGetErrorString) {
            hipGetErrorString(copy_error, &copy_string);
            hipGetErrorString(sync_error, &sync_string);
        }
        fprintf(stderr,
            "DFlash2 propose failed: copy=%s sync=%s forward_error=%d "
            "position=%d count=%d\n",
            copy_string, sync_string,
            r->qwen4_forward_error, position, count);
        return -1;
    }
    return 0;
}

int hip_llm_qwen35_dflash2_commit(hip_llm_runner *r, int position,
                                  int processed) {
    hllm_qwen35_dflash2 *d = r ? r->qwen35_dflash2 : NULL;
    hllm_qwen35_mtp *m = r ? r->qwen35_mtp : NULL;
    if (!d || !m || position != m->verify_position || processed < 1 ||
        processed > m->verify_rows || processed > d->feature_rows) return -1;
    if (hllm_qwen35_dflash2_inject(r,position,processed)) return -1;
    return hip_llm_qwen35_mtp_commit(r,processed);
}
