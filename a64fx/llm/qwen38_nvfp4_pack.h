/* Qwen3.8 single-node NVFP4 decode repack. Include after transformer.h. */
#ifndef QWEN38_NVFP4_PACK_H
#define QWEN38_NVFP4_PACK_H

#if defined(__ARM_FEATURE_SVE)

typedef struct {
    uint64_t idx;
    size_t old_off, new_off, src_bytes, dst_bytes;
    size_t rows, cols;
    int packed;
} q38_nvfp4_layout;

static int q38_nvfp4_layout_cmp(const void *ap, const void *bp) {
    const q38_nvfp4_layout *a = ap, *b = bp;
    return (a->old_off > b->old_off) - (a->old_off < b->old_off);
}

static int q38_nvfp4_plan(const gguf_context *g, q38_nvfp4_layout **out,
                           size_t *out_bytes, int *out_count) {
    if (!g || g->n_shards || !g->data || !g->data_alloc_size) return -1;
    size_t n = (size_t)g->n_tensors;
    q38_nvfp4_layout *ds = calloc(n, sizeof(*ds));
    if (!ds) return -1;
    for (size_t i = 0; i < n; i++) {
        const gguf_tensor_info *ti = &g->tensors[i];
        q38_nvfp4_layout *d = &ds[i];
        d->idx = i;
        d->old_off = (size_t)ti->offset;
        d->src_bytes = gguf_tensor_size(g, (int)i);
        d->dst_bytes = d->src_bytes;
        d->cols = ti->dims[0];
        d->rows = 1;
        for (uint32_t k = 1; k < ti->n_dims; k++) d->rows *= ti->dims[k];
        if (ti->type == GGML_TYPE_NVFP4 && ti->n_dims >= 2 &&
            d->cols % 64 == 0 && d->rows % 8 == 0) {
            d->packed = 1;
            d->dst_bytes = (d->rows / 8) * (d->cols / 64) *
                           sizeof(tf_nvfp4_packed_block);
        }
    }
    qsort(ds, n, sizeof(*ds), q38_nvfp4_layout_cmp);
    size_t cursor = 0;
    int count = 0;
    for (size_t i = 0; i < n; i++) {
        q38_nvfp4_layout *d = &ds[i];
        size_t align = g->alignment;
        size_t off = (cursor + align - 1) / align * align;
        if (off < d->old_off) off = d->old_off;
        if (off > g->data_alloc_size ||
            d->dst_bytes > g->data_alloc_size - off) {
            free(ds); return -1;
        }
        d->new_off = off;
        cursor = off + d->dst_bytes;
        count += d->packed;
    }
    *out = ds; *out_bytes = cursor; *out_count = count;
    return 0;
}

static int q38_nvfp4_plan_tiled(const gguf_context *g, int tile_nextn_ffn,
                                q38_nvfp4_layout **out, int *out_count) {
    if (!g || g->n_shards || !g->data) return -1;
    size_t n = (size_t)g->n_tensors;
    q38_nvfp4_layout *ds = calloc(n, sizeof(*ds));
    if (!ds) return -1;
    int count = 0;
    _Static_assert(sizeof(tf_nvfp4_tiled_block) ==
                   8 * sizeof(block_nvfp4), "exact tile must preserve bytes");
    for (size_t i = 0; i < n; i++) {
        const gguf_tensor_info *ti = &g->tensors[i];
        q38_nvfp4_layout *d = &ds[i];
        d->idx = i;
        d->old_off = d->new_off = (size_t)ti->offset;
        d->src_bytes = d->dst_bytes = gguf_tensor_size(g, (int)i);
        d->cols = ti->dims[0];
        d->rows = 1;
        for (uint32_t k = 1; k < ti->n_dims; k++) d->rows *= ti->dims[k];
        /* The NextN FFN can use exact tiles when its fused gate/up worker is
         * tile-aware. Keep all other draft tensors in their GGUF row layout. */
        int nextn_tensor = strstr(ti->name.str, "nextn") != NULL ||
                           strncmp(ti->name.str, "blk.64.", 7) == 0;
        int nextn_ffn = tile_nextn_ffn &&
            (!strcmp(ti->name.str, "blk.64.ffn_gate.weight") ||
             !strcmp(ti->name.str, "blk.64.ffn_up.weight") ||
             !strcmp(ti->name.str, "blk.64.ffn_down.weight"));
        d->packed = (!nextn_tensor || nextn_ffn) &&
                    ti->type == GGML_TYPE_NVFP4 && ti->n_dims >= 2 &&
                    d->cols % 64 == 0 && d->rows % 8 == 0;
        count += d->packed;
    }
    qsort(ds, n, sizeof(*ds), q38_nvfp4_layout_cmp);
    *out = ds; *out_count = count;
    return 0;
}

/* First-touch final packed locations from their future decode workers. This
 * preserves CMG-local HBM after the serial reverse-order repack. A volatile
 * read/write preserves any small tensors already eagerly loaded by GGUF. */
static void q38_nvfp4_touch(gguf_context *g, const q38_nvfp4_layout *ds,
                             size_t n, int threads) {
    const size_t page = 2u * 1024 * 1024;
#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
#endif
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num(), nt = omp_get_num_threads();
#else
        int tid = 0, nt = 1;
#endif
        volatile uint8_t *base = g->data;
        for (size_t i = 0; i < n; i++) {
            const q38_nvfp4_layout *d = &ds[i];
            size_t start = d->new_off + d->dst_bytes * (size_t)tid / (size_t)nt;
            size_t end = d->new_off + d->dst_bytes * (size_t)(tid + 1) / (size_t)nt;
            start = start / page * page;
            for (size_t p = start; p < end; p += page) base[p] = base[p];
        }
    }
}

static void q38_nvfp4_pack_tile(tf_nvfp4_packed_block *dst,
                                  const block_nvfp4 *src, size_t nb) {
    for (size_t b = 0; b < nb; b++) {
        for (int s = 0; s < 4; s++) {
            tf_nvfp4_packed_subblock *p = &dst[b].s[s];
            for (int r = 0; r < 8; r++) {
                const block_nvfp4 *q = src + (size_t)r * nb + b;
                p->d[r] = tf_nvfp4_scale_fast(q->d[s]);
                memcpy(p->qs + r * 8, q->qs + s * 8, 8);
            }
        }
    }
}

static int q38_nvfp4_rebind_one(qtensor *t, const q38_nvfp4_layout *ds,
                                 size_t n, uint8_t *base, size_t old_bytes,
                                 int tiled) {
    if (!t || !t->data || (uintptr_t)t->data < (uintptr_t)base ||
        (uintptr_t)t->data >= (uintptr_t)base + old_bytes) return 0;
    size_t off = (size_t)((uint8_t *)t->data - base);
    for (size_t i = 0; i < n; i++) {
        const q38_nvfp4_layout *d = &ds[i];
        if (off >= d->old_off && off < d->old_off + d->src_bytes) {
            size_t delta = off - d->old_off;
            if (d->packed && delta) return -1;
            t->data = base + d->new_off + delta;
            t->nvfp4_packed = d->packed && !tiled;
            t->nvfp4_tiled = d->packed && tiled;
            return 0;
        }
    }
    return -1;
}

static int q38_nvfp4_rebind_layer(transformer_layer *l,
                                   const q38_nvfp4_layout *ds, size_t n,
                                   uint8_t *base, size_t old_bytes, int tiled) {
    /* The leading fields of transformer_layer are consecutive qtensors. */
    _Static_assert(offsetof(transformer_layer, is_ssm) -
                   offsetof(transformer_layer, attn_norm) ==
                   34 * sizeof(qtensor), "transformer_layer tensor layout");
    uint8_t *p = (uint8_t *)&l->attn_norm;
    for (int i = 0; i < 34; i++)
        if (q38_nvfp4_rebind_one((qtensor *)(p + (size_t)i * sizeof(qtensor)),
                                  ds, n, base, old_bytes, tiled)) return -1;
    return 0;
}

static int q38_nvfp4_rebind_model(transformer_model *m,
                                   const q38_nvfp4_layout *ds, size_t n,
                                   uint8_t *base, size_t old_bytes, int tiled) {
    qtensor *globals[] = {&m->per_layer_token_embd, &m->per_layer_model_proj,
        &m->per_layer_proj_norm, &m->token_embd, &m->output_norm, &m->output,
        &m->nextn.eh_proj, &m->nextn.enorm, &m->nextn.hnorm,
        &m->nextn.shared_head_norm, &m->nextn.embed_tokens,
        &m->nextn.shared_head_head};
    for (size_t i = 0; i < sizeof(globals)/sizeof(globals[0]); i++)
        if (q38_nvfp4_rebind_one(globals[i], ds, n, base, old_bytes, tiled)) return -1;
    for (int l = 0; l < m->n_layers; l++)
        if (q38_nvfp4_rebind_layer(&m->layers[l], ds, n, base, old_bytes, tiled)) return -1;
    if (m->nextn.loaded &&
        q38_nvfp4_rebind_layer(&m->nextn.layer, ds, n, base, old_bytes, tiled)) return -1;
    return 0;
}

static int q38_nvfp4_pack_model(gguf_context *g, transformer_model *m,
                                 const q38_nvfp4_layout *ds, size_t n,
                                 size_t new_bytes) {
    size_t max_tile = 0;
    for (size_t i = 0; i < n; i++)
        if (ds[i].packed) {
            size_t tile = (ds[i].cols / 64) * sizeof(tf_nvfp4_packed_block);
            if (tile > max_tile) max_tile = tile;
        }
    uint8_t *scratch = malloc(max_tile);
    if (!scratch) return -1;
    uint8_t *base = g->data;
    size_t old_bytes = g->data_size;
    for (size_t i = n; i-- > 0;) {
        const q38_nvfp4_layout *d = &ds[i];
        uint8_t *src = base + d->old_off, *dst = base + d->new_off;
        if (d->packed) {
            size_t nb = d->cols / 64;
            size_t src_tile = 8 * nb * sizeof(block_nvfp4);
            size_t dst_tile = nb * sizeof(tf_nvfp4_packed_block);
            for (size_t tile = d->rows / 8; tile-- > 0;) {
                q38_nvfp4_pack_tile((tf_nvfp4_packed_block *)scratch,
                    (const block_nvfp4 *)(src + tile * src_tile), nb);
                memcpy(dst + tile * dst_tile, scratch, dst_tile);
            }
        } else if (dst != src) {
            memmove(dst, src, d->src_bytes);
        }
    }
    free(scratch);
    if (q38_nvfp4_rebind_model(m, ds, n, base, old_bytes, 0)) return -1;
    for (size_t i = 0; i < n; i++)
        g->tensors[ds[i].idx].offset = ds[i].new_off;
    g->data_size = new_bytes;
    return 0;
}

static int q38_nvfp4_tile_model(gguf_context *g, transformer_model *m,
                                 const q38_nvfp4_layout *ds, size_t n) {
    size_t max_tile = 0;
    for (size_t i = 0; i < n; i++)
        if (ds[i].packed) {
            size_t tile = (ds[i].cols / 64) * sizeof(tf_nvfp4_tiled_block);
            if (tile > max_tile) max_tile = tile;
        }
    uint8_t *scratch = malloc(max_tile);
    if (!scratch) return -1;
    uint8_t *base = g->data;
    for (size_t i = 0; i < n; i++) {
        const q38_nvfp4_layout *d = &ds[i];
        if (!d->packed) continue;
        size_t nb = d->cols / 64;
        size_t tile_bytes = nb * sizeof(tf_nvfp4_tiled_block);
        uint8_t *tensor = base + d->old_off;
        for (size_t tile = 0; tile < d->rows / 8; tile++) {
            const block_nvfp4 *src = (const block_nvfp4 *)(tensor + tile * tile_bytes);
            tf_nvfp4_tiled_block *dst = (tf_nvfp4_tiled_block *)scratch;
            for (size_t b = 0; b < nb; b++)
                for (int s = 0; s < 4; s++) {
                    tf_nvfp4_tiled_subblock *p = &dst[b].s[s];
                    for (int r = 0; r < 8; r++) {
                        const block_nvfp4 *q = src + (size_t)r * nb + b;
                        p->d[r] = q->d[s];
                        memcpy(p->qs + r * 8, q->qs + s * 8, 8);
                    }
                }
            memcpy(tensor + tile * tile_bytes, scratch, tile_bytes);
        }
    }
    free(scratch);
    return q38_nvfp4_rebind_model(m, ds, n, base, g->data_size, 1);
}

#endif /* __ARM_FEATURE_SVE */
#endif /* QWEN38_NVFP4_PACK_H */
