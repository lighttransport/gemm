/* Exact predecoded Q6_K vocabulary head for one A64FX node. */
#ifndef QWEN38_Q6_HEAD_H
#define QWEN38_Q6_HEAD_H

#if defined(__ARM_FEATURE_SVE)
static int q38_q6_pack_head(transformer_model *m) {
    qtensor *head = &m->output;
    if (head->type != GGML_TYPE_Q6_K || !head->data ||
        head->n_cols % 256 || head->tp_owned_data) return -1;
    size_t nb = (size_t)head->n_cols / 256;
    size_t count = (size_t)head->n_rows * nb;
    size_t bytes = count * sizeof(tf_q6_exact_block);
    tf_q6_exact_block *out = NULL;
    if (posix_memalign((void **)&out, 2u * 1024 * 1024, bytes)) return -1;
    const size_t page = 2u * 1024 * 1024;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(48)
#endif
    for (size_t p = 0; p < bytes; p += page)
        ((volatile uint8_t *)out)[p] = 0;
    const block_q6_K *src = (const block_q6_K *)head->data;
    for (size_t i = 0; i < count; i++) {
        const block_q6_K *b = &src[i];
        tf_q6_exact_block *d = &out[i];
        float base_scale = ggml_fp16_to_fp32(b->d);
        for (int si = 0; si < 16; si++)
            d->scale[si] = base_scale * (float)b->scales[si];
        for (int half = 0; half < 2; half++)
            for (int part = 0; part < 4; part++)
                for (int k = 0; k < 32; k++) {
                    int lo_off = half * 64 + ((part & 1) ? 32 : 0) + k;
                    int hi_off = half * 32 + k;
                    int lo_shift = part >= 2 ? 4 : 0;
                    int hi_shift = part * 2;
                    int lo = (b->ql[lo_off] >> lo_shift) & 15;
                    int hi = (b->qh[hi_off] >> hi_shift) & 3;
                    d->q[half * 128 + part * 32 + k] =
                        (int8_t)((lo | (hi << 4)) - 32);
                }
    }
    head->tp_owned_data = out;
    head->data = (uint8_t *)out;
    head->q6_decoded = 1;
    fprintf(stderr, "qwen38: exact Q6_K head %.3fGB decoded from %.3fGB\n",
            bytes / 1e9, count * sizeof(block_q6_K) / 1e9);
    return 0;
}
#endif

#endif
