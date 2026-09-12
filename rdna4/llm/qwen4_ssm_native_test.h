/* Real-weight SSM projection oracle: batched vs scalar fused Q8 projections.
 * Recurrence, convolution and gate normalization are deliberately separate. */
int hip_llm_verify_ssm_projections(hip_llm_runner *r, int M) {
    if (!r || !r->is_qwen4exp || M<1 || M>r->batch_max ||
        !r->d_ssm_qkv_batch) return -1;
    int rc=-1, checked=0, ne=r->n_embd, di=r->ssm_d_inner;
    size_t input_bytes=(size_t)M*(ne+di)*sizeof(float);
    size_t max_bytes=(size_t)M*(r->ssm_qkv_dim+di+2*r->ssm_dt_rank+ne)*sizeof(float);
    float *input=malloc(input_bytes), *scalar=malloc(max_bytes), *batch=malloc(max_bytes);
    void *din=NULL, *dout=NULL;
    if (!input || !scalar || !batch || hipMalloc(&din,input_bytes)!=hipSuccess ||
        hipMalloc(&dout,(size_t)M*ne*sizeof(float))!=hipSuccess) goto done;
    uint32_t seed=73;
    for (size_t i=0;i<input_bytes/sizeof(float);i++) {
        seed=seed*1664525u+1013904223u;
        input[i]=((seed>>8)*(1.0f/16777216.0f)-0.5f)*0.29f;
    }
    /* Include signed-zero inputs so empty-warp elision cannot hide a sign change. */
    for (int i=0;i<ne;i++) input[i]=(i&1)?-0.0f:0.0f;
    for (int i=0;i<di;i++) input[(size_t)M*ne+i]=(i&1)?-0.0f:0.0f;
#define SSM_CHECK(call) do { if ((call)!=hipSuccess) goto done; } while (0)
    SSM_CHECK(hipMemcpyAsync(din,input,input_bytes,hipMemcpyHostToDevice,r->stream));
    SSM_CHECK(hipStreamSynchronize(r->stream));
    for (int l=0;l<r->n_layers;l++) {
        hip_layer *cl=&r->layers[l];
        if (!cl->is_ssm || !qwen4_ssm_native_supported(cl)) continue;
        int dims[5]={cl->ssm_qkv_rows,cl->ssm_gate_rows,r->ssm_dt_rank,r->ssm_dt_rank,ne};
        void *src[5]={r->d_ssm_qkv,r->d_ssm_z,r->d_ssm_alpha,r->d_ssm_beta,r->d_xb};
        void *bsrc[5]={r->d_ssm_qkv_batch,r->d_ssm_z_batch,r->d_ssm_alpha_batch,r->d_ssm_beta_batch,dout};
        int f16=cl->ssm_alpha_type==GGML_TYPE_F16;
        for (int m=0;m<M;m++) {
            void *x=(float*)din+(size_t)m*ne;
            launch_ssm_matvec4_q8_f32(r,src[0],src[1],src[2],src[3],
                cl->ssm_qkv_w,cl->ssm_gate_w,cl->ssm_alpha_w,cl->ssm_beta_w,x,
                dims[0],dims[1],dims[2],ne,ne,f16);
            launch_matvec_auto(r,src[4],cl->ssm_out_w,
                (float*)din+(size_t)M*ne+(size_t)m*di,ne,di,cl->ssm_out_type);
            SSM_CHECK(hipStreamSynchronize(r->stream));
            size_t off=0;
            for (int k=0;k<5;k++) {
                SSM_CHECK(hipMemcpy(scalar+off+(size_t)m*dims[k],src[k],
                          (size_t)dims[k]*sizeof(float),hipMemcpyDeviceToHost));
                off+=(size_t)M*dims[k];
            }
        }
        if (launch_ssm_matvec4_q8_batch(r,cl,M,din,bsrc[0],bsrc[1],bsrc[2],bsrc[3])) goto done;
        if (launch_matmul_q8_batch_f32(r,dout,cl->ssm_out_w,
            (float*)din+(size_t)M*ne,M,ne,di)) goto done;
        SSM_CHECK(hipStreamSynchronize(r->stream));
        size_t off=0;
        for (int k=0;k<5;k++) {
            size_t count=(size_t)M*dims[k];
            SSM_CHECK(hipMemcpy(batch+off,bsrc[k],count*sizeof(float),hipMemcpyDeviceToHost));
            for (size_t i=0;i<count;i++) {
                if (!isfinite(scalar[off+i]) || !isfinite(batch[off+i]) ||
                    memcmp(scalar+off+i,batch+off+i,sizeof(float))) {
                    fprintf(stderr,"SSM projection mismatch layer=%d projection=%d index=%zu scalar=%.9g batch=%.9g\n",
                            l,k,i,scalar[off+i],batch[off+i]);
                    goto done;
                }
            }
            off+=count;
        }
        checked++;
    }
    rc=checked ? 0 : -1;
done:
    hipStreamSynchronize(r->stream);
    if (din) hipFree(din);
    if (dout) hipFree(dout);
    free(input);free(scalar);free(batch);
    fprintf(stderr,"SSM native projection bitwise %s: %d layers x 5 projections x %d rows\n",
            rc ? "FAIL" : "PASS",checked,M);
    return rc;
#undef SSM_CHECK
}
