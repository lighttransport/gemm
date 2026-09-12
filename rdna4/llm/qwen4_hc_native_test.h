/* Real-weight bitwise oracle for native batched HC, including injection. */
static int qwen4_verify_hc_native(hip_llm_runner *r, int M, double *rel, double *mx) {
    int ne=r->n_embd, ns=r->hc_count, hcd=ne*ns, rc=-1;
    size_t input_bytes=(size_t)M*hcd*sizeof(float);
    size_t output_bytes=(size_t)M*ne*sizeof(float);
    size_t inject_bytes=(size_t)M*ns*sizeof(float);
    float *input=malloc(input_bytes), *scalar=malloc(output_bytes);
    float *batch=malloc(output_bytes), *si=malloc(inject_bytes), *bi=malloc(inject_bytes);
    void *out=NULL;
    if (!input || !scalar || !batch || !si || !bi ||
        hipMalloc(&out,output_bytes)!=hipSuccess) goto done;
    uint32_t seed=19;
    for (int i=0;i<M*hcd;i++) {
        seed=seed*1664525u+1013904223u;
        input[i]=((seed>>8)*(1.0f/16777216.0f)-0.5f)*0.31f;
    }
#define HC_CHECK(call) do { if ((call)!=hipSuccess) goto done; } while (0)
    for (int l=0;l<r->n_layers;l++) for (int ffn=0;ffn<2;ffn++) {
        hip_layer *cl=&r->layers[l];
        void *nw=ffn?cl->hc_ffn_norm_w:cl->hc_attn_norm_w;
        void *dw=ffn?cl->hc_ffn_down_w:cl->hc_attn_down_w;
        void *uw=ffn?cl->hc_ffn_up_w:cl->hc_attn_up_w;
        void *iw=ffn?cl->hc_ffn_inject_w:cl->hc_attn_inject_w;
        int dt=ffn?cl->hc_ffn_down_type:cl->hc_attn_down_type;
        int ut=ffn?cl->hc_ffn_up_type:cl->hc_attn_up_type;
        int it=ffn?cl->hc_ffn_inject_type:cl->hc_attn_inject_type;
        for (int m=0;m<M;m++) {
            HC_CHECK(hipMemcpyAsync(r->d_hc,input+(size_t)m*hcd,
                     (size_t)hcd*sizeof(float),hipMemcpyHostToDevice,r->stream));
            forward_hc_mix(r,nw,dw,dt,uw,ut,iw,it,r->d_xb,-1);
            HC_CHECK(hipStreamSynchronize(r->stream));
            HC_CHECK(hipMemcpy(scalar+(size_t)m*ne,r->d_xb,
                     (size_t)ne*sizeof(float),hipMemcpyDeviceToHost));
            if (iw) HC_CHECK(hipMemcpy(si+(size_t)m*ns,r->d_hc_inject,
                     (size_t)ns*sizeof(float),hipMemcpyDeviceToHost));
        }
        HC_CHECK(hipMemcpyAsync(r->d_hc_batch,input,input_bytes,
                 hipMemcpyHostToDevice,r->stream));
        if (forward_hc_mix_batched(r,M,nw,dw,NULL,dt,uw,NULL,ut,iw,it,out)) goto done;
        HC_CHECK(hipStreamSynchronize(r->stream));
        HC_CHECK(hipMemcpy(batch,out,output_bytes,hipMemcpyDeviceToHost));
        if (iw) HC_CHECK(hipMemcpy(bi,r->d_hc_inject_batch,inject_bytes,hipMemcpyDeviceToHost));
        for (int i=0;i<M*ne;i++) {
            if (!isfinite(scalar[i]) || !isfinite(batch[i]) ||
                memcmp(&scalar[i],&batch[i],sizeof(float))) {
                fprintf(stderr,"native HC mismatch layer=%d ffn=%d output=%d scalar=%.9g batch=%.9g\n",
                        l,ffn,i,scalar[i],batch[i]); goto done;
            }
        }
        if (iw) for (int i=0;i<M*ns;i++) {
            if (!isfinite(si[i]) || !isfinite(bi[i]) || memcmp(&si[i],&bi[i],sizeof(float))) {
                fprintf(stderr,"native HC injection mismatch layer=%d ffn=%d index=%d\n",l,ffn,i);
                goto done;
            }
        }
    }
    fprintf(stderr,"native HC bitwise PASS: %d layers x 2 phases x %d rows\n",r->n_layers,M);
    if (rel) *rel=0;
    if (mx) *mx=0;
    rc=0;
done:
    hipStreamSynchronize(r->stream);
    if (out) hipFree(out);
    free(input);free(scalar);free(batch);free(si);free(bi);
    return rc;
#undef HC_CHECK
}
