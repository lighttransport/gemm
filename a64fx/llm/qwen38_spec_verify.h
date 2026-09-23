/* Exact batched target verification for single-node Qwen3.8 NextN decode.
 * Include after transformer.h; all emitted IDs come from target logits. */
#ifndef QWEN38_SPEC_VERIFY_H
#define QWEN38_SPEC_VERIFY_H

static int q38_spec_verify(transformer_model *m, bpe_vocab *v, int pos,
                           int32_t cur, float cur_logit, int max_gen, int k,
                           int dump_tokens) {
    const int layers = m->n_layers, ne = m->n_embd, vocab = m->n_vocab;
    const size_t conv = (size_t)(m->ssm_conv_kernel - 1) * m->ssm_qkv_dim;
    const size_t rec = (size_t)m->ssm_dt_rank * m->ssm_d_state * m->ssm_d_state;
    const size_t layer_stride = conv + rec, slot_stride = (size_t)layers * layer_stride;
    float *arena = NULL, *all_logits = NULL, *seed = NULL, *batch_hidden = NULL;
    float **slots = NULL, **orig_conv = NULL, **orig_rec = NULL;
    int *conv_pos = NULL;
    int32_t batch[4], target[4];
    int rc = 1, initialized = 0, generated = 0, rounds = 0, accepted_total = 0;
    double draft_s = 0.0, verify_s = 0.0, commit_s = 0.0;
    if (k < 2 || k > 4 || m->kv_cache_type != 0 || !m->nextn.loaded)
        return 2;
    if (posix_memalign((void **)&arena, 256, (size_t)k * slot_stride * sizeof(float)))
        arena = NULL;
    all_logits = malloc((size_t)k * vocab * sizeof(float));
    seed = malloc((size_t)ne * sizeof(float));
    slots = malloc((size_t)k * sizeof(*slots));
    orig_conv = malloc((size_t)layers * sizeof(*orig_conv));
    orig_rec = malloc((size_t)layers * sizeof(*orig_rec));
    conv_pos = malloc((size_t)layers * sizeof(*conv_pos));
    if (!arena || !all_logits || !seed || !slots || !orig_conv || !orig_rec ||
        !conv_pos) {
        fprintf(stderr, "qwen38: speculative verifier allocation failed\n");
        goto done;
    }
    for (int j = 0; j < k - 1; j++)
        slots[j] = arena + (size_t)j * slot_stride;
    slots[k - 1] = NULL;
    float *current = arena + (size_t)(k - 1) * slot_stride;
    for (int l = 0; l < layers; l++) {
        orig_conv[l] = m->conv_state ? m->conv_state[l] : NULL;
        orig_rec[l] = m->recurrent_state ? m->recurrent_state[l] : NULL;
        if (!m->layers[l].is_ssm) continue;
        float *dst = current + (size_t)l * layer_stride;
        memcpy(dst, orig_conv[l], conv * sizeof(float));
        memcpy(dst + conv, orig_rec[l], rec * sizeof(float));
        m->conv_state[l] = dst;
        m->recurrent_state[l] = dst + conv;
    }
    initialized = 1;
    tf_batch_ssm_snapshots = arena;
    tf_batch_ssm_snapshot_slots = slots;
    tf_batch_ssm_layer_stride = layer_stride;
    tf_batch_ssm_slot_stride = slot_stride;
    tf_batch_keep_pool = 1;
    tf_batch_quiet = 1;
    transformer_prefill_profile_reset();
    tf_rmsnorm(seed, transformer_get_hidden(m), &m->output_norm,
               ne, m->rms_norm_eps, m->matvec_tmp);
    double t0 = now_sec();
    while (generated < max_gen) {
        double round_t0 = now_sec();
        if (pos + k >= m->max_seq_len) {
            fprintf(stderr, "qwen38: speculative verifier exceeded context\n");
            goto done;
        }
        batch[0] = cur;
        const float *draft_h = seed;
        int32_t prev = cur;
        for (int j = 1; j < k; j++) {
            float *dl = transformer_nextn_logits(m, prev, draft_h,
                                                 pos - 1 + j - 1);
            if (!dl) goto done;
            batch[j] = argmax(dl, vocab);
            prev = batch[j];
            draft_h = transformer_nextn_hidden(m);
        }
        double draft_end = now_sec();
        for (int l = 0; l < layers; l++)
            conv_pos[l] = m->conv_state_pos ? m->conv_state_pos[l] : 0;
        tf_batch_all_logits = all_logits;
        tf_batch_hidden_out = &batch_hidden;
        float *ok = transformer_prefill_gemm(m, batch, k, pos);
        double verify_end = now_sec();
        tf_batch_all_logits = NULL;
        tf_batch_hidden_out = NULL;
        if (!ok || !batch_hidden) {
            fprintf(stderr, "qwen38: exact target batch failed at position %d\n", pos);
            goto done;
        }
        for (int j = 0; j < k; j++)
            target[j] = argmax(all_logits + (size_t)j * vocab, vocab);
        int accepted = 0;
        while (accepted < k - 1 && batch[accepted + 1] == target[accepted])
            accepted++;
        if (getenv("TF_SPEC_TRACE") && rounds < 12 && k == 4)
            fprintf(stderr, "qwen38: spec_round pos=%d input=%d drafts=%d,%d,%d target=%d,%d,%d,%d accepted=%d\n",
                    pos, batch[0], batch[1], batch[2], batch[3],
                    target[0], target[1], target[2], target[3], accepted);
        int committed = accepted + 1;
        if (accepted < k - 1) {
            float *old = current;
            current = slots[accepted];
            slots[accepted] = old;
            for (int l = 0; l < layers; l++) {
                if (!m->layers[l].is_ssm) continue;
                float *state = current + (size_t)l * layer_stride;
                m->conv_state[l] = state;
                m->recurrent_state[l] = state + conv;
                m->conv_state_pos[l] =
                    (conv_pos[l] + committed) % (m->ssm_conv_kernel - 1);
            }
        }
        const float *hidden = batch_hidden + (size_t)accepted * ne;
        tf_rmsnorm(seed, hidden, &m->output_norm,
                   ne, m->rms_norm_eps, m->matvec_tmp);
        for (int j = 0; j < committed && generated < max_gen; j++) {
            int32_t emitted = j == 0 ? cur : target[j - 1];
            float selected = j == 0 ? cur_logit :
                all_logits[(size_t)(j - 1) * vocab + emitted];
            if (dump_tokens)
                fprintf(stderr, "qwen38: token n=%d pos=%d id=%d logit=%a\n",
                        generated, pos + j, emitted, selected);
            const char *piece = bpe_token_to_str(v, emitted);
            if (piece) fputs(piece, stdout);
            generated++;
            if (emitted == v->eos_id || emitted == v->eot_id) {
                rc = 0;
                goto done;
            }
        }
        cur = target[accepted];
        cur_logit = all_logits[(size_t)accepted * vocab + cur];
        pos += committed;
        accepted_total += accepted;
        rounds++;
        draft_s += draft_end - round_t0;
        verify_s += verify_end - draft_end;
        commit_s += now_sec() - verify_end;
    }
    rc = 0;
done:
    if (initialized) {
        for (int l = 0; l < layers; l++) {
            if (!m->layers[l].is_ssm) continue;
            float *state = current + (size_t)l * layer_stride;
            memcpy(orig_conv[l], state, conv * sizeof(float));
            memcpy(orig_rec[l], state + conv, rec * sizeof(float));
            m->conv_state[l] = orig_conv[l];
            m->recurrent_state[l] = orig_rec[l];
        }
    }
    tf_batch_ssm_snapshots = NULL;
    tf_batch_ssm_snapshot_slots = NULL;
    tf_batch_ssm_layer_stride = tf_batch_ssm_slot_stride = 0;
    tf_batch_keep_pool = 0;
    tf_batch_quiet = 0;
    tf_batch_all_logits = NULL;
    tf_batch_hidden_out = NULL;
    if (initialized) {
        double seconds = now_sec() - t0;
        fprintf(stderr, "qwen38: spec_verify=%d tokens %.3fs %.3f tok/s rounds=%d accepted=%d/%d\n",
                generated, seconds, seconds > 0 ? generated / seconds : 0.0,
                rounds, accepted_total, rounds * (k - 1));
        transformer_prefill_profile profile;
        transformer_prefill_profile_get(&profile);
        fprintf(stderr, "qwen38: spec_profile proj=%.1f out=%.1f ffn_proj=%.1f down=%.1f ms\n",
                profile.proj_ms, profile.out_proj_ms, profile.ffn_proj_ms,
                profile.ffn_down_ms);
        fprintf(stderr, "qwen38: spec_profile_other norm=%.1f ssm_prepare=%.1f "
                        "ssm_scan=%.1f attn_prepare=%.1f attn_kernel=%.1f "
                        "ffn_act=%.1f collective=%.1f ms layers=%d calls=%d\n",
                profile.norm_ms, profile.ssm_prepare_ms, profile.ssm_scan_ms,
                profile.attn_prepare_ms, profile.attn_kernel_ms,
                profile.ffn_act_ms, profile.collective_ms,
                profile.layers, profile.calls);
        fprintf(stderr, "qwen38: spec_stage draft=%.1f verify=%.1f commit=%.1f ms\n",
                draft_s * 1000.0, verify_s * 1000.0, commit_s * 1000.0);
        fputc('\n', stdout);
    }
    free(conv_pos); free(orig_rec); free(orig_conv); free(slots);
    free(seed); free(all_logits); free(arena);
    return rc;
}

#endif
