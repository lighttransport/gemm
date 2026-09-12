/* Real-weight phase-order oracle. Included after the scalar layer routines. */
int hip_llm_verify_qwen4_ple_split(hip_llm_runner *r, int rows) {
    if (!r || !r->is_qwen4exp || r->n_layers < 2 || rows < 2 || rows > 32)
        return -1;
    hip_layer *cl = &r->layers[1];
    if (!cl->is_ssm || !cl->is_moe || !r->d_ple_conv_state) return -1;
    size_t hcd = (size_t)r->hc_count * r->n_embd;
    size_t hc_bytes = (size_t)rows * hcd * sizeof(float);
    size_t conv_bytes = (size_t)(r->ssm_conv_kernel - 1) * r->ssm_qkv_dim * sizeof(float);
    size_t rec_bytes = (size_t)r->ssm_dt_rank * r->ssm_d_state * r->ssm_d_state * sizeof(float);
    size_t ple_bytes = (size_t)(r->ple_conv_kernel - 1) * r->ple_ngram * hcd * sizeof(float);
    size_t bytes = hc_bytes + conv_bytes + rec_bytes + ple_bytes;
    float *input = malloc(hc_bytes);
    unsigned char *ref = malloc(bytes), *got = malloc(bytes);
    void *device = NULL, *saved_hc = r->d_hc;
    int rc = -1, positions[32];
    if (!input || !ref || !got || hipMalloc(&device, hc_bytes) != hipSuccess) goto done;
    for (size_t i = 0; i < hc_bytes / sizeof(float); ++i)
        input[i] = sinf((float)i * 0.017f) + 0.25f * cosf((float)i * 0.003f);
    for (int i = 0; i < rows; ++i) positions[i] = i;
    for (int split = 0; split < 2; ++split) {
        hip_llm_reset_state(r);
        /* Match cold-cache state even when the caller did not request resets. */
        cl->moe_cache_next = 0;
        for (int i = 0; i < cl->moe_cache_slots; ++i) {
            cl->moe_cache_ids[i] = -1;
            if (cl->moe_cache_age) cl->moe_cache_age[i] = 0;
            if (cl->moe_cache_freq) cl->moe_cache_freq[i] = 0;
        }
        if (hipMemsetAsync(cl->d_moe_cache_map, 0xff, (size_t)r->n_experts * sizeof(int), r->stream) != hipSuccess ||
            hipMemcpyAsync(device, input, hc_bytes, hipMemcpyHostToDevice, r->stream) != hipSuccess) goto done;
        for (int m = 0; m < rows; ++m) {
            r->d_hc = (char *)device + (size_t)m * hcd * sizeof(float);
            r->ple_token_id = m + 1;
            r->ple_history[0] = m < 2 ? r->ple_eos_token : m - 1;
            r->ple_history[1] = m < 1 ? r->ple_eos_token : m;
            r->cur_position = m;
            if (hipMemcpyAsync(r->d_position, &positions[m], sizeof(int), hipMemcpyHostToDevice, r->stream) != hipSuccess) goto done;
            forward_layer_state_phase(r, cl, 1, r->d_key_cache[1], r->d_value_cache[1], 1, split);
            if (hipStreamSynchronize(r->stream) != hipSuccess || r->qwen4_forward_error) goto done;
        }
        if (split) for (int m = 0; m < rows; ++m) {
            r->d_hc = (char *)device + (size_t)m * hcd * sizeof(float);
            r->cur_position = m;
            forward_hc_mix(r, cl->hc_ffn_norm_w, cl->hc_ffn_down_w,
                cl->hc_ffn_down_type, cl->hc_ffn_up_w, cl->hc_ffn_up_type,
                cl->hc_ffn_inject_w, cl->hc_ffn_inject_type, r->d_xb, 3);
            forward_moe_ffn(r, cl);
            forward_hc_combine(r, r->d_moe_accum);
            if (hipStreamSynchronize(r->stream) != hipSuccess || r->qwen4_forward_error) goto done;
        }
        unsigned char *out = split ? got : ref;
        if (hipMemcpy(out, device, hc_bytes, hipMemcpyDeviceToHost) != hipSuccess ||
            hipMemcpy(out + hc_bytes, cl->d_conv_state, conv_bytes, hipMemcpyDeviceToHost) != hipSuccess ||
            hipMemcpy(out + hc_bytes + conv_bytes, cl->d_recurrent_state, rec_bytes, hipMemcpyDeviceToHost) != hipSuccess ||
            hipMemcpy(out + hc_bytes + conv_bytes + rec_bytes, r->d_ple_conv_state, ple_bytes, hipMemcpyDeviceToHost) != hipSuccess) goto done;
    }
    rc = 0;
    for (size_t i = 0; i < bytes / sizeof(float); ++i) {
        float a, b;
        memcpy(&a, ref + i * sizeof(float), sizeof(float));
        memcpy(&b, got + i * sizeof(float), sizeof(float));
        if (!isfinite(a) || !isfinite(b) || memcmp(&a, &b, sizeof(float))) {
            fprintf(stderr, "PLE phase-order mismatch float=%zu: %.9g / %.9g\n", i, a, b);
            rc = 1; break;
        }
    }
done:
    hipStreamSynchronize(r->stream);
    r->d_hc = saved_hc;
    if (device) hipFree(device);
    free(input); free(ref); free(got);
    hip_llm_reset_state(r);
    return rc;
}
