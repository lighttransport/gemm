/* Qwen4 HC, SSM, full attention, and MoE execute in disjoint phases on
 * r->stream. Only their temporary intermediates share this allocation.
 * Persistent residuals, HC injection, normalized input, attention projection,
 * dequantized weights, and asynchronous staging banks remain separate.
 * Included after hip_llm_runner's definition. */
static int qwen4_phase_scratch_init(hip_llm_runner *r) {
    size_t bm = (size_t)r->batch_max, hcd = (size_t)r->hc_count*r->n_embd;
    size_t lr = (size_t)r->hc_low_rank, ta = bm*r->n_experts_used;
    size_t qd = (size_t)r->n_heads*r->head_dim, kd = (size_t)r->n_kv_heads*r->head_dim;
    size_t max_q = qd;
    for (int l = 0; l < r->n_layers; ++l)
        if (r->layers[l].attn_q_rows > 0 && (size_t)r->layers[l].attn_q_rows > max_q)
            max_q = (size_t)r->layers[l].attn_q_rows;
    size_t sizes[4] = {0};
    #define ADD_SIZE(phase, bytes) sizes[phase] += ((bytes) + 255) & ~(size_t)255
    ADD_SIZE(0, bm*hcd*4);
    ADD_SIZE(0, bm*hcd*2);
    ADD_SIZE(0, bm*hcd*4);
    ADD_SIZE(0, bm*hcd*2);
    ADD_SIZE(0, bm*lr*4);
    ADD_SIZE(0, bm*lr*2);
    ADD_SIZE(1, bm*r->ssm_qkv_dim*4);
    ADD_SIZE(1, bm*r->ssm_d_inner*4);
    ADD_SIZE(1, bm*r->ssm_dt_rank*4);
    ADD_SIZE(1, bm*r->ssm_dt_rank*4);
    ADD_SIZE(1, bm*r->ssm_qkv_dim*4);
    ADD_SIZE(1, bm*r->ssm_dt_rank*r->ssm_d_state*4);
    ADD_SIZE(1, bm*r->ssm_dt_rank*r->ssm_d_state*4);
    ADD_SIZE(1, bm*r->ssm_d_inner*4);
    ADD_SIZE(2, bm*qd*4);
    ADD_SIZE(2, bm*kd*4);
    ADD_SIZE(2, bm*kd*4);
    ADD_SIZE(2, bm*qd*4);
    ADD_SIZE(2, bm*qd*2);
    ADD_SIZE(2, bm*max_q*4);
    ADD_SIZE(2, bm*qd*4);
    ADD_SIZE(3, ta*r->n_embd*4);
    ADD_SIZE(3, ta*r->n_embd*2);
    ADD_SIZE(3, ta*r->expert_ff*4);
    ADD_SIZE(3, ta*r->expert_ff*2);
    ADD_SIZE(3, bm*r->n_embd*4);
    ADD_SIZE(3, bm*r->n_embd*2);
    ADD_SIZE(3, bm*4);
    #undef ADD_SIZE
    size_t bytes = 0, total = 0;
    for (int i = 0; i < 4; ++i) { total += sizes[i]; if (sizes[i] > bytes) bytes = sizes[i]; }
    if (!bytes || r->d_qwen4_phase_scratch) return -1;
    if (hipMalloc(&r->d_qwen4_phase_scratch, bytes) != hipSuccess) return -1;
    size_t offset = 0;
    #define BIND(field, bytes) do { \
        r->field = (char *)r->d_qwen4_phase_scratch + offset; \
        offset += ((bytes) + 255) & ~(size_t)255; \
    } while (0)
    offset = 0;
    BIND(d_hc_norm_batch, bm*hcd*4);
    BIND(d_hc_norm_batch_bf16, bm*hcd*2);
    BIND(d_hc_gate_batch, bm*hcd*4);
    BIND(d_hc_gate_batch_bf16, bm*hcd*2);
    BIND(d_hc_low_batch, bm*lr*4);
    BIND(d_hc_low_batch_bf16, bm*lr*2);
    offset = 0;
    BIND(d_ssm_qkv_batch, bm*r->ssm_qkv_dim*4);
    BIND(d_ssm_z_batch, bm*r->ssm_d_inner*4);
    BIND(d_ssm_alpha_batch, bm*r->ssm_dt_rank*4);
    BIND(d_ssm_beta_batch, bm*r->ssm_dt_rank*4);
    BIND(d_ssm_conv_out_batch, bm*r->ssm_qkv_dim*4);
    BIND(d_ssm_Q_exp_batch, bm*r->ssm_dt_rank*r->ssm_d_state*4);
    BIND(d_ssm_K_exp_batch, bm*r->ssm_dt_rank*r->ssm_d_state*4);
    BIND(d_ssm_out_batch, bm*r->ssm_d_inner*4);
    offset = 0;
    BIND(d_q_batch, bm*qd*4);
    BIND(d_k_batch, bm*kd*4);
    BIND(d_v_batch, bm*kd*4);
    BIND(d_attn_out_batch, bm*qd*4);
    BIND(d_attn_out_batch_bf16, bm*qd*2);
    BIND(d_qfull_batch, bm*max_q*4);
    BIND(d_attn_gate_batch, bm*qd*4);
    offset = 0;
    BIND(d_moe_gather_in, ta*r->n_embd*4);
    BIND(d_moe_gather_in_bf16, ta*r->n_embd*2);
    BIND(d_moe_eg, ta*r->expert_ff*4);
    BIND(d_moe_esilu_bf16, ta*r->expert_ff*2);
    BIND(d_moe_out_batch, bm*r->n_embd*4);
    BIND(d_xnorm_batch_bf16_moe, bm*r->n_embd*2);
    BIND(d_shared_scale_batch, bm*4);
    #undef BIND
    fprintf(stderr, "hip_llm: Qwen4 phase scratch %.1f MiB, saved %.1f MiB\n",
            bytes/1048576.0, (total-bytes)/1048576.0);
    return 0;
}

/* Called after streams drain, before the ordinary per-field cleanup. */
static void qwen4_phase_scratch_free(hip_llm_runner *r) {
    if (!r->d_qwen4_phase_scratch) return;
    hipFree(r->d_qwen4_phase_scratch);
    r->d_qwen4_phase_scratch = NULL;
    if (r->d_moe_eout_alias_gather) {
        if (r->d_moe_eu_alias_eout) r->d_moe_eu = NULL;
        r->d_moe_eout = NULL;
    }
    r->d_hc_norm_batch = NULL;
    r->d_hc_norm_batch_bf16 = NULL;
    r->d_hc_gate_batch = NULL;
    r->d_hc_gate_batch_bf16 = NULL;
    r->d_hc_low_batch = NULL;
    r->d_hc_low_batch_bf16 = NULL;
    r->d_ssm_qkv_batch = NULL;
    r->d_ssm_z_batch = NULL;
    r->d_ssm_alpha_batch = NULL;
    r->d_ssm_beta_batch = NULL;
    r->d_ssm_conv_out_batch = NULL;
    r->d_ssm_Q_exp_batch = NULL;
    r->d_ssm_K_exp_batch = NULL;
    r->d_ssm_out_batch = NULL;
    r->d_q_batch = NULL;
    r->d_k_batch = NULL;
    r->d_v_batch = NULL;
    r->d_attn_out_batch = NULL;
    r->d_attn_out_batch_bf16 = NULL;
    r->d_qfull_batch = NULL;
    r->d_attn_gate_batch = NULL;
    r->d_moe_gather_in = NULL;
    r->d_moe_gather_in_bf16 = NULL;
    r->d_moe_eg = NULL;
    r->d_moe_esilu_bf16 = NULL;
    r->d_moe_out_batch = NULL;
    r->d_xnorm_batch_bf16_moe = NULL;
    r->d_shared_scale_batch = NULL;
}
