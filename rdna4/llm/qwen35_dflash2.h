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

typedef struct hllm_dflash_layer {
    void *attn_norm, *q_norm, *k_norm, *ffn_norm;
    void *q, *k, *v, *o, *gate, *up, *down;
    int q_type, k_type, v_type, o_type, gate_type, up_type, down_type;
    void *attn_conv_base, *attn_conv_proj;
    void *ffn_conv_base, *ffn_conv_proj;
    int attn_conv_proj_type, ffn_conv_proj_type;
    void *key_cache, *value_cache;
} hllm_dflash_layer;

typedef struct hllm_qwen35_dflash2 {
    gguf_shards *source;
    hipModule_t module;
    hipFunction_t fn_capture, fn_conv, fn_attention;
    void *fc, *enc_norm, *out_norm, *selector_hidden;
    int fc_type, selector_hidden_type;
    qtensor selector_prev, selector_next;
    int target_layers[HLLM_DFLASH_LAYERS];
    int mask_token, feature_rows, kv_end;
    hllm_dflash_layer layers[HLLM_DFLASH_LAYERS];

    /* All activation storage is row-major F32. */
    void *features, *x, *norm, *dynamic, *conv;
    void *q, *k, *v, *attn, *proj, *gate, *up;
    void *logits, *selector_gate;
    float *host_logits, *host_selector_gate;
    float *selector_prev_row, *selector_next_rows;
} hllm_qwen35_dflash2;

static void hllm_qwen35_dflash2_free(hip_llm_runner *r) {
    hllm_qwen35_dflash2 *d = r ? r->qwen35_dflash2 : NULL;
    if (!d) return;
    if (r->stream) hipStreamSynchronize(r->stream);
#define DFLASH_FREE(p) do { if (p) hipFree(p); } while (0)
    DFLASH_FREE(d->fc); DFLASH_FREE(d->enc_norm); DFLASH_FREE(d->out_norm);
    DFLASH_FREE(d->selector_hidden);
    DFLASH_FREE(d->features); DFLASH_FREE(d->x); DFLASH_FREE(d->norm);
    DFLASH_FREE(d->dynamic); DFLASH_FREE(d->conv); DFLASH_FREE(d->q);
    DFLASH_FREE(d->k); DFLASH_FREE(d->v); DFLASH_FREE(d->attn);
    DFLASH_FREE(d->proj); DFLASH_FREE(d->gate); DFLASH_FREE(d->up);
    DFLASH_FREE(d->logits); DFLASH_FREE(d->selector_gate);
    for (int l = 0; l < HLLM_DFLASH_LAYERS; ++l) {
        hllm_dflash_layer *cl = &d->layers[l];
        DFLASH_FREE(cl->attn_norm); DFLASH_FREE(cl->q_norm);
        DFLASH_FREE(cl->k_norm); DFLASH_FREE(cl->ffn_norm);
        DFLASH_FREE(cl->q); DFLASH_FREE(cl->k); DFLASH_FREE(cl->v);
        DFLASH_FREE(cl->o); DFLASH_FREE(cl->gate); DFLASH_FREE(cl->up);
        DFLASH_FREE(cl->down); DFLASH_FREE(cl->attn_conv_base);
        DFLASH_FREE(cl->attn_conv_proj); DFLASH_FREE(cl->ffn_conv_base);
        DFLASH_FREE(cl->ffn_conv_proj); DFLASH_FREE(cl->key_cache);
        DFLASH_FREE(cl->value_cache);
    }
#undef DFLASH_FREE
    free(d->host_logits); free(d->host_selector_gate);
    free(d->selector_prev_row); free(d->selector_next_rows);
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
    if (type == GGML_TYPE_Q4_K && rows > 1) {
        launch_matvec_qwen35_native_batch(r, dst, weight, x, rows, nr, nc,
                                          stride, type);
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
    hllm_dflash_project(r, d->x, d->fc, d->features, rows, ne,
                        HLLM_DFLASH_LAYERS * ne,
                        HLLM_DFLASH_LAYERS * ne, d->fc_type);
    launch_rmsnorm_batch(r, d->x, d->x, d->enc_norm, ne, rows, ne,
                         r->rms_norm_eps);
    for (int l = 0; l < HLLM_DFLASH_LAYERS; ++l) {
        hllm_dflash_layer *cl = &d->layers[l];
        hllm_dflash_project(r, d->k, cl->k, d->x, rows, kd, ne, ne,
                            cl->k_type);
        hllm_dflash_project(r, d->v, cl->v, d->x, rows, kd, ne, ne,
                            cl->v_type);
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
                             "qwen35_dflash2_attention") != hipSuccess) goto fail;

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
    DALLOC(attn, cap*HLLM_DFLASH_HEADS*HLLM_DFLASH_HEAD_DIM); DALLOC(proj, cap*ne);
    DALLOC(gate, cap*ff); DALLOC(up, cap*ff);
    DALLOC(logits, HLLM_DFLASH_MAX_BLOCK*r->n_vocab);
    DALLOC(selector_gate, HLLM_DFLASH_MAX_BLOCK*HLLM_DFLASH_RANK);
#undef DALLOC
    d->host_logits = malloc((size_t)HLLM_DFLASH_MAX_BLOCK*r->n_vocab*sizeof(float));
    d->host_selector_gate = malloc((size_t)HLLM_DFLASH_MAX_BLOCK*HLLM_DFLASH_RANK*sizeof(float));
    d->selector_prev_row = malloc(HLLM_DFLASH_RANK*sizeof(float));
    d->selector_next_rows = malloc((size_t)HLLM_DFLASH_TOPK*HLLM_DFLASH_RANK*sizeof(float));
    if (!d->host_logits || !d->host_selector_gate || !d->selector_prev_row ||
        !d->selector_next_rows || hllm_qwen35_verify_workspace_create(r)) goto fail;
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
}

static void hllm_dflash_topk(const float *logits, int n, int *ids) {
    float scores[HLLM_DFLASH_TOPK];
    for (int k=0;k<HLLM_DFLASH_TOPK;++k) { scores[k]=-INFINITY; ids[k]=0; }
    for (int i=0;i<n;++i) if (logits[i] > scores[HLLM_DFLASH_TOPK-1]) {
        int k=HLLM_DFLASH_TOPK-1;
        while (k>0 && logits[i] > scores[k-1]) {
            scores[k]=scores[k-1]; ids[k]=ids[k-1]; --k;
        }
        scores[k]=logits[i]; ids[k]=i;
    }
}

int hip_llm_qwen35_dflash2_propose(hip_llm_runner *r, int32_t anchor,
        int position, int count, int32_t *drafts) {
    hllm_qwen35_dflash2 *d = r ? r->qwen35_dflash2 : NULL;
    hllm_qwen35_mtp *m = r ? r->qwen35_mtp : NULL;
    if (!d || !m || m->verify_rows || !drafts || count < 1 || count > 7 ||
        anchor < 0 || anchor >= r->n_vocab || position != d->kv_end) return -1;
    int rows=count+1, ne=r->n_embd, qd=HLLM_DFLASH_HEADS*HLLM_DFLASH_HEAD_DIM;
    int kd=HLLM_DFLASH_KV_HEADS*HLLM_DFLASH_HEAD_DIM;
    launch_embed_iq1_m(r,d->x,r->d_token_embd,anchor,ne);
    for (int i=1;i<rows;++i)
        launch_embed_iq1_m(r,(float *)d->x+(size_t)i*ne,r->d_token_embd,d->mask_token,ne);
    for (int l=0;l<HLLM_DFLASH_LAYERS;++l) {
        hllm_dflash_layer *cl=&d->layers[l];
        launch_rmsnorm_batch(r,d->norm,d->x,cl->attn_norm,ne,rows,ne,r->rms_norm_eps);
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
        void *aa[]={&d->attn,&d->q,&cl->key_cache,&cl->value_cache,&rows,
            &position,&(int){HLLM_DFLASH_HEADS},&(int){HLLM_DFLASH_KV_HEADS},
            &(int){HLLM_DFLASH_HEAD_DIM},&window};
        LAUNCH(d->fn_attention,HLLM_DFLASH_HEADS,rows,1,32,1,1,0,r->stream,aa);
        hllm_dflash_project(r,d->proj,cl->o,d->attn,rows,ne,qd,qd,cl->o_type);
        hllm_dflash_conv(r,d,d->conv,d->proj,d->dynamic,cl->attn_conv_base,rows,1);
        launch_add(r,d->x,d->conv,rows*ne);
        launch_rmsnorm_batch(r,d->norm,d->x,cl->ffn_norm,ne,rows,ne,r->rms_norm_eps);
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
    hllm_dflash_project(r,d->logits,r->d_output_w,d->norm,rows,r->n_vocab,ne,ne,r->output_w_type);
    hllm_dflash_project(r,d->selector_gate,d->selector_hidden,d->norm,rows,
        HLLM_DFLASH_RANK,ne,ne,d->selector_hidden_type);
    if (hipMemcpyAsync(d->host_logits,d->logits,(size_t)rows*r->n_vocab*sizeof(float),
            hipMemcpyDeviceToHost,r->stream) ||
        hipMemcpyAsync(d->host_selector_gate,d->selector_gate,
            (size_t)rows*HLLM_DFLASH_RANK*sizeof(float),hipMemcpyDeviceToHost,r->stream) ||
        hipStreamSynchronize(r->stream) || r->qwen4_forward_error) return -1;

    int candidate[HLLM_DFLASH_MAX_BLOCK][HLLM_DFLASH_TOPK];
    int predecessor=0, predecessor_token=anchor;
    size_t row_bytes_prev=dequant_row_size(d->selector_prev.type,HLLM_DFLASH_RANK);
    size_t row_bytes_next=dequant_row_size(d->selector_next.type,HLLM_DFLASH_RANK);
    if (!row_bytes_prev || !row_bytes_next) return -1;
    for (int p=1;p<rows;++p) {
        const float *logit=d->host_logits+(size_t)p*r->n_vocab;
        hllm_dflash_topk(logit,r->n_vocab,candidate[p]);
        if (p>1) predecessor_token=candidate[p-1][predecessor];
        if (dequant_row(d->selector_prev.type,
                (const char *)d->selector_prev.data+(size_t)predecessor_token*row_bytes_prev,
                d->selector_prev_row,HLLM_DFLASH_RANK)) return -1;
        for (int k=0;k<HLLM_DFLASH_TOPK;++k)
            if (dequant_row(d->selector_next.type,
                (const char *)d->selector_next.data+(size_t)candidate[p][k]*row_bytes_next,
                d->selector_next_rows+(size_t)k*HLLM_DFLASH_RANK,
                HLLM_DFLASH_RANK)) return -1;
        const float *sg=d->host_selector_gate+(size_t)p*HLLM_DFLASH_RANK;
        float best=-INFINITY; int best_k=0;
        for (int k=0;k<HLLM_DFLASH_TOPK;++k) {
            const float *sn=d->selector_next_rows+(size_t)k*HLLM_DFLASH_RANK;
            float score=logit[candidate[p][k]];
            for (int j=0;j<HLLM_DFLASH_RANK;++j)
                score += d->selector_prev_row[j]*sg[j]*sn[j];
            if (score>best) { best=score; best_k=k; }
        }
        predecessor=best_k;
        drafts[p-1]=candidate[p][best_k];
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
