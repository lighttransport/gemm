/* Compare native router/shared batching against the scalar GPU kernels.
 * This oracle excludes routed expert execution and its weighted reduction. */
int hip_llm_verify_moe_native(hip_llm_runner *r, int M) {
    if (!r || !r->is_qwen4exp || M<1 || M>r->batch_max ||
        !r->d_router_logits_batch || r->n_experts>512) return -1;
    int rc=-1, checked=0, ne=r->n_embd, nx=r->n_experts, eff=r->shared_expert_ff;
    int dims[4]={nx,1,eff,ne}, K=r->n_experts_used;
    size_t bytes=(size_t)M*ne*sizeof(float);
    size_t count=(size_t)M*(nx+1+eff+ne);
    float *input=malloc(bytes), *scalar=malloc(count*sizeof(float)), *batch=malloc(count*sizeof(float));
    void *din=NULL, *dacc=NULL;
    if (!input || !scalar || !batch || hipMalloc(&din,bytes)!=hipSuccess ||
        hipMalloc(&dacc,bytes)!=hipSuccess) goto done;
    uint32_t seed=97;
    for (int i=0;i<M*ne;i++) {
        seed=seed*1664525u+1013904223u;
        input[i]=((seed>>8)*(1.0f/16777216.0f)-0.5f)*0.17f;
    }
#define MOE_CHECK(call) do { if ((call)!=hipSuccess) goto done; } while (0)
    MOE_CHECK(hipMemcpyAsync(din,input,bytes,hipMemcpyHostToDevice,r->stream));
    MOE_CHECK(hipMemsetAsync(r->d_router_counter,0,sizeof(unsigned),r->stream));
    MOE_CHECK(hipStreamSynchronize(r->stream));
    for (int l=0;l<r->n_layers;l++) {
        hip_layer *cl=&r->layers[l];
        if (!cl->is_moe || !qwen4_shared_native_supported(cl)) continue;
        int q6=cl->moe_shared_gate_type==GGML_TYPE_Q6_K;
        int rows=q6?8:4, threads=q6?256:128;
        hipFunction_t gu=q6?r->fn_shexp_gateup_silu_q6k:r->fn_shexp_gateup_silu_q8;
        hipFunction_t down=q6?r->fn_shexp_down_accum_q6k:r->fn_shexp_down_accum_q8;
        MOE_CHECK(hipMemcpyAsync(dacc,din,bytes,hipMemcpyDeviceToDevice,r->stream));
        for (int m=0;m<M;m++) {
            void *x=(float*)din+(size_t)m*ne, *out=(float*)dacc+(size_t)m*ne;
            void *slot_map=NULL, *cache_valid=NULL;
            void *a[]={&cl->moe_gate_w_bf16,&cl->moe_shared_gate_w_bf16,&x,&nx,&K,&ne,
                &r->d_router_logits,&r->d_moe_idx,&r->d_moe_w,&r->d_shared_scale,
                &r->d_router_counter,&slot_map,&cache_valid};
            MOE_CHECK(LAUNCH(r->fn_moe_router_fused,nx+1,1,1,256,1,1,0,r->stream,a));
            void *b[]={&r->d_gate,&cl->moe_shared_ffn_gate_w,&cl->moe_shared_ffn_up_w,&x,&eff,&ne};
            MOE_CHECK(LAUNCH(gu,(eff+rows-1)/rows,1,1,threads,1,1,0,r->stream,b));
            void *c[]={&out,&cl->moe_shared_ffn_down_w,&r->d_gate,&ne,&eff,&r->d_shared_scale};
            MOE_CHECK(LAUNCH(down,(ne+rows-1)/rows,1,1,threads,1,1,0,r->stream,c));
            MOE_CHECK(hipStreamSynchronize(r->stream));
            void *src[4]={r->d_router_logits,r->d_shared_scale,r->d_gate,out};
            size_t off=0;
            for (int k=0;k<4;k++) {
                MOE_CHECK(hipMemcpy(scalar+off+(size_t)m*dims[k],src[k],
                          (size_t)dims[k]*sizeof(float),hipMemcpyDeviceToHost));
                off+=(size_t)M*dims[k];
            }
        }
        MOE_CHECK(hipMemcpyAsync(dacc,din,bytes,hipMemcpyDeviceToDevice,r->stream));
        if (launch_qwen4_router_native(r,cl,M,din,r->d_router_logits_batch,r->d_shared_scale_batch) ||
            launch_qwen4_shared_native(r,cl,M,din,r->d_moe_eg,dacc,r->d_shared_scale_batch)) goto done;
        MOE_CHECK(hipStreamSynchronize(r->stream));
        void *src[4]={r->d_router_logits_batch,r->d_shared_scale_batch,r->d_moe_eg,dacc};
        size_t off=0;
        for (int k=0;k<4;k++) {
            size_t n=(size_t)M*dims[k];
            MOE_CHECK(hipMemcpy(batch+off,src[k],n*sizeof(float),hipMemcpyDeviceToHost));
            for (size_t i=0;i<n;i++) {
                if (!isfinite(scalar[off+i]) || !isfinite(batch[off+i]) ||
                    memcmp(scalar+off+i,batch+off+i,sizeof(float))) {
                    fprintf(stderr,"native MoE mismatch layer=%d stage=%d index=%zu scalar=%.9g batch=%.9g\n",
                            l,k,i,scalar[off+i],batch[off+i]); goto done;
                }
            }
            off+=n;
        }
        checked++;
    }
    rc=checked ? 0 : -1;
done:
    hipStreamSynchronize(r->stream);
    if (din) hipFree(din);
    if (dacc) hipFree(dacc);
    free(input);free(scalar);free(batch);
    fprintf(stderr,"Native router/shared bitwise %s: %d layers x %d rows\n",
            rc ? "FAIL" : "PASS",checked,M);
    return rc;
#undef MOE_CHECK
}
