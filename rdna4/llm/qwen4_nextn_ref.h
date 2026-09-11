/* Scalar Qwen4 HC + MoE NextN oracle. Included by hip_llm_runner.c.
 * Tensor views borrow live GGUF mappings; no HIP operations occur here. */
typedef struct {
    qtensor q, k, v, qnorm, knorm, out;
    qtensor an, ad, au, ai, fn, fd, fu, fi;
    qtensor router, gate, up, down, sg, su, sd, sr;
    qtensor eh, en, hn, headn, headd, headu, emb, head;
    int ne, ns, lr, nh, nk, hd, ff, shared_ff, experts, used, capacity, first;
    float eps, rope;
    int rope_pairs, rope_active;
    float *keys, *values;
} qwen4_nextn_ref;

/* HIP __float2half uses round-to-nearest-even, unlike the legacy weight
 * uploader's truncation. Match the cache format, including subnormals. */
static uint16_t q4ref_half(float x) {
    _Float16 h = (_Float16)x;
    uint16_t bits; memcpy(&bits, &h, sizeof(bits)); return bits;
}

static void q4ref_mv(float *out, const qtensor *w, const float *in) {
    size_t stride = dequant_row_size(w->type, w->n_cols);
#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
        float row[w->n_cols];
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (int i = 0; i < w->n_rows; ++i) {
            dequant_row(w->type, (const unsigned char *)w->data + (size_t)i*stride, row, w->n_cols);
            double sum = 0;
            for (int j = 0; j < w->n_cols; ++j) sum += (double)row[j]*in[j];
            out[i] = (float)sum;
        }
    }
}

static void q4ref_norm(float *out, const float *in, const qtensor *weight,
                       int width, int streams, float eps, int shared) {
    float w[width*streams];
    dequant_row(weight->type, weight->data, w, shared ? width : width*streams);
    for (int s = 0; s < streams; ++s) {
        double sum = 0;
        for (int j = 0; j < width; ++j) sum += (double)in[s*width+j]*in[s*width+j];
        float scale = 1.0f/sqrtf((float)(sum/width)+eps);
        for (int j = 0; j < width; ++j) out[s*width+j] = in[s*width+j]*scale*w[(shared ? 0 : s*width)+j];
    }
}

static void q4ref_mix(qwen4_nextn_ref *c, const float *hc, float *mixed, float *inject,
                      const qtensor *norm, const qtensor *down, const qtensor *up, const qtensor *iw) {
    int ne = c->ne, ns = c->ns;
    float xn[ne*ns], lo[c->lr], gate[ne*ns];
    q4ref_norm(xn, hc, norm, ne, ns, c->eps, 0);
    q4ref_mv(lo, down, xn);
    for (int i = 0; i < c->lr; ++i) { float x = lo[i]/ns; lo[i] = x/(1.0f+expf(-x)); }
    q4ref_mv(gate, up, lo);
    for (int j = 0; j < ne; ++j) {
        float sum = 0;
        for (int s = 0; s < ns; ++s) sum += xn[s*ne+j]/(1.0f+expf(-gate[s*ne+j]));
        mixed[j] = sum/ns;
    }
    if (iw) q4ref_mv(inject, iw, xn);
}

static void q4ref_combine(qwen4_nextn_ref *c, float *hc, const float *block, const float *inject) {
    for (int s = 0; s < c->ns; ++s) {
        float scale = 2.0f/(1.0f+expf(-inject[s]/c->ns));
        for (int i = 0; i < c->ne; ++i) hc[s*c->ne+i] += scale*block[i];
    }
}

static int q4ref_init(qwen4_nextn_ref *c, hip_llm_runner *r, const gguf_shards *sidecar,
                      const gguf_shards *target, int capacity) {
    memset(c, 0, sizeof(*c));
    c->ne=r->n_embd; c->ns=r->hc_count; c->lr=r->hc_low_rank;
    c->nh=r->n_heads; c->nk=r->n_kv_heads; c->hd=r->head_dim;
    c->ff=r->expert_ff; c->shared_ff=r->shared_expert_ff;
    c->experts=r->n_experts; c->used=r->n_experts_used;
    c->eps=r->rms_norm_eps; c->rope=r->rope_freq_base; c->capacity=capacity; c->first=-1;
    c->rope_pairs=r->use_mrope ? r->mrope_sections[0]+r->mrope_sections[1]+r->mrope_sections[2]+r->mrope_sections[3] : r->head_dim/2;
    c->rope_active=r->use_mrope ? r->mrope_sections[0]+r->mrope_sections[1]+r->mrope_sections[2] : c->rope_pairs;
    const gguf_shards *saved = hllm_active_shards;
    hllm_active_shards = sidecar;
    char name[96];
#define REF_LOAD(field, suffix) do { \
        snprintf(name, sizeof(name), "blk.%d." suffix ".weight", r->n_layers); \
        c->field=hllm_load_tensor(sidecar->metadata,name,1); \
        if (!c->field.data) { hllm_active_shards=saved; return -1; } \
    } while (0)
    REF_LOAD(q,"attn_q"); REF_LOAD(k,"attn_k"); REF_LOAD(v,"attn_v");
    REF_LOAD(qnorm,"attn_q_norm"); REF_LOAD(knorm,"attn_k_norm"); REF_LOAD(out,"attn_output");
    REF_LOAD(an,"hc_attn_norm"); REF_LOAD(ad,"hc_attn_down"); REF_LOAD(au,"hc_attn_up"); REF_LOAD(ai,"hc_attn_inject");
    REF_LOAD(fn,"hc_ffn_norm"); REF_LOAD(fd,"hc_ffn_down"); REF_LOAD(fu,"hc_ffn_up"); REF_LOAD(fi,"hc_ffn_inject");
    REF_LOAD(router,"ffn_gate_inp"); REF_LOAD(gate,"ffn_gate_exps"); REF_LOAD(up,"ffn_up_exps"); REF_LOAD(down,"ffn_down_exps");
    REF_LOAD(sg,"ffn_gate_shexp"); REF_LOAD(su,"ffn_up_shexp"); REF_LOAD(sd,"ffn_down_shexp"); REF_LOAD(sr,"ffn_gate_inp_shexp");
    REF_LOAD(eh,"nextn.eh_proj"); REF_LOAD(en,"nextn.enorm"); REF_LOAD(hn,"nextn.hnorm");
    REF_LOAD(headn,"nextn.hc_head_norm"); REF_LOAD(headd,"nextn.hc_head_down"); REF_LOAD(headu,"nextn.hc_head_up");
#undef REF_LOAD
    hllm_active_shards=target;
    c->emb=hllm_load_tensor(target->metadata,"token_embd.weight",1);
    c->head=hllm_load_tensor(target->metadata,"output.weight",0);
    if (!c->head.data) c->head=c->emb;
    hllm_active_shards=saved;
    if (!c->emb.data) return -1;
    c->keys=calloc((size_t)capacity*c->nk*c->hd,sizeof(float));
    c->values=calloc((size_t)capacity*c->nk*c->hd,sizeof(float));
    if (!c->keys || !c->values) { free(c->keys); free(c->values); return -1; }
    return 0;
}

static void q4ref_rope(float *x, int heads, int dim, int position, float base, int pairs, int active) {
    for (int h = 0; h < heads; ++h) for (int j = 0; j < dim/2; ++j) {
        float theta = j<active ? position*powf(base,-(float)j/pairs) : 0.0f;
        float a=x[h*dim+j], b=x[h*dim+j+dim/2];
        x[h*dim+j]=a*cosf(theta)-b*sinf(theta);
        x[h*dim+j+dim/2]=a*sinf(theta)+b*cosf(theta);
    }
}

static void q4ref_expert(float *out, const qtensor *g, const qtensor *u,
                         const qtensor *d, int expert, int ff, const float *x) {
    qtensor gate=*g, up=*u, down=*d;
    gate.n_rows=up.n_rows=ff; down.n_rows=g->n_cols;
    gate.data=(unsigned char *)g->data+(size_t)expert*ff*dequant_row_size(g->type,g->n_cols);
    up.data=(unsigned char *)u->data+(size_t)expert*ff*dequant_row_size(u->type,u->n_cols);
    down.data=(unsigned char *)d->data+(size_t)expert*down.n_rows*dequant_row_size(d->type,d->n_cols);
    float gv[ff], uv[ff]; q4ref_mv(gv,&gate,x); q4ref_mv(uv,&up,x);
    for (int j=0;j<ff;++j) gv[j]=gv[j]/(1.0f+expf(-gv[j]))*uv[j];
    q4ref_mv(out,&down,gv);
}

static int q4ref_forward(qwen4_nextn_ref *c, int token, int position,
                         const float *hidden, float *hc, float *logits) {
    if (token<0 || token>=c->emb.n_rows || position<0 || position>=c->capacity) return -1;
    int ne=c->ne, ns=c->ns, qd=c->nh*c->hd, kd=c->nk*c->hd;
    float emb[ne], hn[ns*ne], fusion[2*ne], x[ne], block[ne], inject[ns];
    dequant_row(c->emb.type,(const unsigned char *)c->emb.data+(size_t)token*dequant_row_size(c->emb.type,ne),emb,ne);
    q4ref_norm(emb,emb,&c->en,ne,1,c->eps,0);
    q4ref_norm(hn,hidden,&c->hn,ne,ns,c->eps,0);
    for(int s=0;s<ns;++s) {
        memcpy(fusion,emb,sizeof(emb)); memcpy(fusion+ne,hn+s*ne,sizeof(emb));
        q4ref_mv(hc+s*ne,&c->eh,fusion);
    }
    q4ref_mix(c,hc,x,inject,&c->an,&c->ad,&c->au,&c->ai);
    float qg[2*qd], q[qd], gate[qd], k[kd], v[kd], attn[qd];
    q4ref_mv(qg,&c->q,x); q4ref_mv(k,&c->k,x); q4ref_mv(v,&c->v,x);
    for(int h=0;h<c->nh;++h) {
        memcpy(q+h*c->hd,qg+2*h*c->hd,(size_t)c->hd*sizeof(float));
        memcpy(gate+h*c->hd,qg+(2*h+1)*c->hd,(size_t)c->hd*sizeof(float));
    }
    q4ref_norm(q,q,&c->qnorm,c->hd,c->nh,c->eps,1);
    q4ref_norm(k,k,&c->knorm,c->hd,c->nk,c->eps,1);
    q4ref_rope(q,c->nh,c->hd,position,c->rope,c->rope_pairs,c->rope_active);
    q4ref_rope(k,c->nk,c->hd,position,c->rope,c->rope_pairs,c->rope_active);
    if(c->first<0)c->first=position;
    for(int j=0;j<kd;++j) {
        c->keys[(size_t)position*kd+j]=ggml_fp16_to_fp32(q4ref_half(k[j]));
        c->values[(size_t)position*kd+j]=ggml_fp16_to_fp32(q4ref_half(v[j]));
    }
    for(int h=0;h<c->nh;++h) {
        int kv=h/(c->nh/c->nk); float scores[position+1], max=-INFINITY;
        for(int t=c->first;t<=position;++t) {
            double sum=0;
            for(int j=0;j<c->hd;++j) sum+=(double)q[h*c->hd+j]*c->keys[(size_t)t*kd+kv*c->hd+j];
            scores[t]=(float)sum/sqrtf(c->hd); if(scores[t]>max)max=scores[t];
        }
        double den=0; for(int t=c->first;t<=position;++t) {scores[t]=expf(scores[t]-max);den+=scores[t];}
        for(int j=0;j<c->hd;++j) {
            double sum=0; for(int t=c->first;t<=position;++t)sum+=scores[t]*c->values[(size_t)t*kd+kv*c->hd+j];
            attn[h*c->hd+j]=(float)(sum/den)/(1.0f+expf(-gate[h*c->hd+j]));
        }
    }
    q4ref_mv(block,&c->out,attn); q4ref_combine(c,hc,block,inject);
    q4ref_mix(c,hc,x,inject,&c->fn,&c->fd,&c->fu,&c->fi);
    float router[c->experts], weights[c->used]; int ids[c->used];
    q4ref_mv(router,&c->router,x);
    moe_topk_softmax(router,c->experts,c->used,ids,weights);
    memset(block,0,sizeof(block));
    for(int i=0;i<c->used;++i) {
        float out[ne]; q4ref_expert(out,&c->gate,&c->up,&c->down,ids[i],c->ff,x);
        for(int j=0;j<ne;++j)block[j]+=weights[i]*out[j];
    }
    float shared[ne], scale; q4ref_mv(&scale,&c->sr,x);
    q4ref_expert(shared,&c->sg,&c->su,&c->sd,0,c->shared_ff,x);
    for(int j=0;j<ne;++j)block[j]+=shared[j]/(1.0f+expf(-scale));
    q4ref_combine(c,hc,block,inject);
    q4ref_mix(c,hc,x,NULL,&c->headn,&c->headd,&c->headu,NULL);
    q4ref_mv(logits,&c->head,x);
    return 0;
}
