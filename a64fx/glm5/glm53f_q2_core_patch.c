/* Patch node-local compact-core layers with weights dequantized from GGUF.
 *
 * The compact image keeps the production kernel layouts: BF16 for mHC/KDA,
 * F32 for scalar vectors, and block-128 FP8 plus F32 scales for the dense FFN.
 * Only the rank-local /local copy is modified; the shared source image is not.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define GGUF_LOADER_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION
#include "../../common/gguf_loader.h"
#include "../../common/ggml_dequant.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

enum { RANKS = 12, HIDDEN = 4096, INTER = 12288, BLOCK = 128 };

typedef struct {
    char kind;
    char name[512];
    uint64_t a, b, c, blob;
} image_entry;

typedef struct {
    const gguf_tensor_info *info;
    int fd;
    uint64_t base;
} tensor_ref;

static int read_exact(int fd, uint64_t off, void *dst, size_t bytes) {
    unsigned char *p = dst;
    while (bytes) {
        ssize_t got = pread(fd, p, bytes, (off_t)off);
        if (got < 0) { if (errno == EINTR) continue; return -1; }
        if (!got) { errno = EIO; return -1; }
        p += got; off += (uint64_t)got; bytes -= (size_t)got;
    }
    return 0;
}

static int write_exact(int fd, uint64_t off, const void *src, size_t bytes) {
#ifdef GLM53F_PP_CORE_STAGE
    static uint64_t pending;
    const size_t initial_bytes=bytes;
#endif
    const unsigned char *p = src;
    while (bytes) {
        ssize_t put = pwrite(fd, p, bytes, (off_t)off);
        if (put < 0) { if (errno == EINTR) continue; return -1; }
        if (!put) { errno = EIO; return -1; }
        p += put; off += (uint64_t)put; bytes -= (size_t)put;
    }
#ifdef GLM53F_PP_CORE_STAGE
    pending+=initial_bytes;
    if(pending>=(1u<<20)){
        if(fdatasync(fd))return-1;
        (void)posix_fadvise(fd,0,0,POSIX_FADV_DONTNEED);pending=0;
    }
#endif
    return 0;
}

static size_t tensor_row_bytes(const tensor_ref *t) {
    uint32_t type = t->info->type;
    uint64_t cols = t->info->dims[0];
    if (type >= GGML_TYPE_COUNT || ggml_type_info[type].block_size <= 0 ||
        cols % (uint64_t)ggml_type_info[type].block_size) return 0;
    return (size_t)(cols / (uint64_t)ggml_type_info[type].block_size) *
           (size_t)ggml_type_info[type].type_size;
}

static void release_tensor_cache(const tensor_ref *t) {
#ifdef GLM53F_PP_CORE_STAGE
    uint64_t bytes = tensor_row_bytes(t);
    for (uint32_t d = 1; d < t->info->n_dims; ++d) {
        if (!t->info->dims[d] || bytes > UINT64_MAX / t->info->dims[d]) return;
        bytes *= t->info->dims[d];
    }
    /* Row advice alone leaves partial pages cached for quantized rows.
     * Release the complete tensor, including its boundary pages, after use. */
    long page = sysconf(_SC_PAGESIZE);
    if (page <= 0 || t->base > UINT64_MAX - bytes ||
        t->base + bytes > INT64_MAX - (uint64_t)page) return;
    uint64_t begin = t->base / (uint64_t)page * (uint64_t)page;
    uint64_t end = (t->base + bytes + (uint64_t)page - 1) /
                   (uint64_t)page * (uint64_t)page;
    (void)posix_fadvise(t->fd, (off_t)begin, (off_t)(end - begin), POSIX_FADV_DONTNEED);
#else
    (void)t;
#endif
}

static tensor_ref find_tensor(const gguf_context *g, const char *name) {
    tensor_ref t = {0};
    t.fd = -1;
    for (uint64_t i = 0; i < g->n_tensors; ++i) {
        if (!g->tensors[i].name.str || strcmp(g->tensors[i].name.str, name)) continue;
        t.info = &g->tensors[i];
        t.fd = g->tensor_fds ? g->tensor_fds[i] : g->fd;
        t.base = g->tensor_file_offsets ? g->tensor_file_offsets[i] :
                 g->data_offset + g->tensors[i].offset;
        break;
    }
    return t;
}

static int read_row(const tensor_ref *t, uint64_t row, void *raw, float *values) {
    size_t bytes = tensor_row_bytes(t);
    uint64_t rows = 1;
    for (uint32_t d = 1; d < t->info->n_dims; ++d) rows *= t->info->dims[d];
    if (!bytes || row >= rows || read_exact(t->fd, t->base + row * bytes, raw, bytes))
        return -1;
#ifdef GLM53F_PP_CORE_STAGE
    (void)posix_fadvise(t->fd,(off_t)(t->base+row*bytes),bytes,POSIX_FADV_DONTNEED);
#endif
    return dequant_row(t->info->type, raw, values, (int)t->info->dims[0]);
}

static int find_entry(const char *manifest, const char *name, image_entry *out) {
    FILE *f = fopen(manifest, "r");
    char line[2048], got_name[512], kind;
    unsigned long long a, b, c, blob;
    if (!f) return -1;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, " %c %511s %llu %llu %llu %llu",
                   &kind, got_name, &a, &b, &c, &blob) == 6 && kind == 'C') {
            if (strcmp(got_name, name)) continue;
            *out = (image_entry){kind, "", a, b, c, blob};
            snprintf(out->name, sizeof(out->name), "%s", got_name);
            fclose(f); return 0;
        }
        if (sscanf(line, " %c %511s %llu %llu %llu",
                   &kind, got_name, &a, &b, &blob) == 5 && kind == 'R') {
            if (strcmp(got_name, name)) continue;
            *out = (image_entry){kind, "", a, b, 0, blob};
            snprintf(out->name, sizeof(out->name), "%s", got_name);
            fclose(f); return 0;
        }
    }
    fclose(f); errno = ENOENT; return -1;
}

static uint16_t bf16_rne(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += UINT32_C(0x7fff) + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

static int patch_linear(int blob_fd, const image_entry *e, const tensor_ref *t,
                        int element_bytes, int a_log) {
    size_t raw_bytes = tensor_row_bytes(t);
    uint64_t cols = t->info->dims[0], start, count, col0, coln, rows;
    unsigned char *raw = malloc(raw_bytes);
    float *values = malloc((size_t)cols * sizeof(float));
    void *out = malloc((size_t)cols * (size_t)element_bytes);
    if (!raw || !values || !out) goto fail;
    if (e->kind == 'R') {
        if (e->a % (uint64_t)element_bytes || e->b % (uint64_t)element_bytes) goto fail;
        start = e->a / (uint64_t)element_bytes;
        count = e->b / (uint64_t)element_bytes;
        uint64_t done = 0;
        while (done < count) {
            uint64_t at = start + done, row = at / cols;
            col0 = at % cols;
            coln = cols - col0 < count - done ? cols - col0 : count - done;
            if (read_row(t, row, raw, values)) goto fail;
            if (element_bytes == 2) {
                uint16_t *dst = out;
                for (uint64_t i = 0; i < coln; ++i) dst[i] = bf16_rne(values[col0 + i]);
            } else {
                float *dst = out;
                for (uint64_t i = 0; i < coln; ++i) {
                    float v = values[col0 + i];
                    dst[i] = a_log ? logf(-v) : v;
                }
            }
            if (write_exact(blob_fd, e->blob + done * (uint64_t)element_bytes,
                            out, (size_t)coln * (size_t)element_bytes)) goto fail;
            done += coln;
        }
    } else {
        if (e->a != cols * (uint64_t)element_bytes ||
            e->b % (uint64_t)element_bytes || e->c % (uint64_t)element_bytes) goto fail;
        col0 = e->b / (uint64_t)element_bytes;
        coln = e->c / (uint64_t)element_bytes;
        rows = 1;
        for (uint32_t d = 1; d < t->info->n_dims; ++d) rows *= t->info->dims[d];
        if (col0 + coln > cols) goto fail;
        for (uint64_t row = 0; row < rows; ++row) {
            if (read_row(t, row, raw, values)) goto fail;
            if (element_bytes == 2) {
                uint16_t *dst = out;
                for (uint64_t i = 0; i < coln; ++i) dst[i] = bf16_rne(values[col0 + i]);
            } else {
                float *dst = out;
                for (uint64_t i = 0; i < coln; ++i) dst[i] = values[col0 + i];
            }
            if (write_exact(blob_fd, e->blob + row * coln * (uint64_t)element_bytes,
                            out, (size_t)coln * (size_t)element_bytes)) goto fail;
        }
    }
    release_tensor_cache(t);
    free(out); free(values); free(raw); return 0;
fail:
    free(out); free(values); free(raw); return -1;
}

static int patch_named(int blob_fd, const char *manifest, const gguf_context *g,
                       int layer, const char *safe_suffix, const char *gguf_suffix,
                       int element_bytes, int a_log) {
    char safe[768], gguf[256];
    image_entry e;
    snprintf(safe, sizeof(safe), "model.language_model.layers.%d.%s", layer, safe_suffix);
    snprintf(gguf, sizeof(gguf), "blk.%d.%s", layer, gguf_suffix);
    tensor_ref t = find_tensor(g, gguf);
    if (!t.info || find_entry(manifest, safe, &e) ||
        patch_linear(blob_fd, &e, &t, element_bytes, a_log)) {
        fprintf(stderr, "patch failed layer=%d dst=%s src=%s\n", layer, safe, gguf);
        return -1;
    }
    return 0;
}

static float fp8_positive(int code) {
    int exponent = (code >> 3) & 15, mantissa = code & 7;
    if (!exponent) return (float)mantissa / 512.0f;
    return ldexpf(1.0f + (float)mantissa / 8.0f, exponent - 7);
}

static uint8_t fp8_nearest(float value) {
    int sign = signbit(value), lo = 0, hi = 126;
    float x = fabsf(value);
    if (x >= 448.0f) return (uint8_t)((sign << 7) | 126);
    while (lo + 1 < hi) {
        int mid = (lo + hi) / 2;
        if (fp8_positive(mid) <= x) lo = mid; else hi = mid;
    }
    if (fabsf(fp8_positive(hi) - x) < fabsf(x - fp8_positive(lo))) lo = hi;
    return (uint8_t)((sign << 7) | lo);
}

static int quantize_dense_slice(int blob_fd, const image_entry *weight,
                                const image_entry *scale, const tensor_ref *t,
                                uint64_t row0, uint64_t rows,
                                uint64_t col0, uint64_t cols) {
    size_t raw_bytes = tensor_row_bytes(t);
    unsigned char *raw = malloc(raw_bytes);
    float *row = malloc((size_t)t->info->dims[0] * sizeof(float));
    float *tile = malloc((size_t)BLOCK * cols * sizeof(float));
    uint8_t *quant = malloc((size_t)BLOCK * cols);
    float *scales = malloc((size_t)(cols / BLOCK) * sizeof(float));
    uint64_t source_rows = 1;
    for (uint32_t d = 1; d < t->info->n_dims; ++d)
        source_rows *= t->info->dims[d];
    if (!raw || !row || !tile || !quant || !scales ||
        row0 % BLOCK || rows % BLOCK || col0 % BLOCK || cols % BLOCK ||
        row0 + rows > source_rows || col0 + cols > t->info->dims[0]) goto fail;
    for (uint64_t rb = 0; rb < rows; rb += BLOCK) {
        for (uint64_t r = 0; r < BLOCK; ++r) {
            if (read_row(t, row0 + rb + r, raw, row)) goto fail;
            memcpy(tile + r * cols, row + col0, (size_t)cols * sizeof(float));
        }
        for (uint64_t cb = 0; cb < cols; cb += BLOCK) {
            float max_abs = 0.0f;
            for (uint64_t r = 0; r < BLOCK; ++r)
                for (uint64_t c = 0; c < BLOCK; ++c) {
                    float a = fabsf(tile[r * cols + cb + c]);
                    if (a > max_abs) max_abs = a;
                }
            float s = max_abs > 0.0f ? max_abs / 448.0f : 1.0f;
            scales[cb / BLOCK] = s;
            for (uint64_t r = 0; r < BLOCK; ++r)
                for (uint64_t c = 0; c < BLOCK; ++c)
                    quant[r * cols + cb + c] = fp8_nearest(tile[r * cols + cb + c] / s);
        }
        if (write_exact(blob_fd, weight->blob + rb * cols, quant,
                        (size_t)BLOCK * cols) ||
            write_exact(blob_fd, scale->blob + (rb / BLOCK) * (cols / BLOCK) * sizeof(float),
                        scales, (size_t)(cols / BLOCK) * sizeof(float))) goto fail;
    }
    release_tensor_cache(t);
    free(scales); free(quant); free(tile); free(row); free(raw); return 0;
fail:
    free(scales); free(quant); free(tile); free(row); free(raw); return -1;
}

static int patch_fp8_named(int blob_fd, const char *manifest, const gguf_context *g,
                            int layer, const char *safe_weight,
                            const char *safe_scale, const char *gguf_suffix) {
    char wn[768], sn[768], gn[256];
    image_entry w, s;
    snprintf(wn, sizeof(wn), "model.language_model.layers.%d.%s", layer, safe_weight);
    snprintf(sn, sizeof(sn), "model.language_model.layers.%d.%s", layer, safe_scale);
    snprintf(gn, sizeof(gn), "blk.%d.%s", layer, gguf_suffix);
    tensor_ref t = find_tensor(g, gn);
    if (!t.info || find_entry(manifest, wn, &w) || find_entry(manifest, sn, &s)) return -1;
    uint64_t row0, rows, col0, cols;
    if (w.kind == 'R') {
        cols = t.info->dims[0];
        if (!cols || w.a % cols || w.b % cols) return -1;
        row0 = w.a / cols; rows = w.b / cols; col0 = 0;
        if (s.kind != 'R' || s.b != (rows / BLOCK) * (cols / BLOCK) * sizeof(float))
            return -1;
    } else {
        uint64_t source_cols = t.info->dims[0];
        rows = 1;
        for (uint32_t d = 1; d < t.info->n_dims; ++d) rows *= t.info->dims[d];
        row0 = 0; col0 = w.b; cols = w.c;
        if (w.a != source_cols || s.kind != 'C' ||
            s.b != (col0 / BLOCK) * sizeof(float) ||
            s.c != (cols / BLOCK) * sizeof(float)) return -1;
    }
    if (quantize_dense_slice(blob_fd, &w, &s, &t, row0, rows, col0, cols)) {
        fprintf(stderr, "dense patch failed layer=%d dst=%s src=%s\n", layer, wn, gn);
        return -1;
    }
    return 0;
}

static int patch_common(int blob_fd, const char *manifest,
                        const gguf_context *g, int layer) {
#define BF(SAFE, GGUF) if (patch_named(blob_fd, manifest, g, layer, SAFE, GGUF, 2, 0)) return -1
#define F32(SAFE, GGUF) if (patch_named(blob_fd, manifest, g, layer, SAFE, GGUF, 4, 0)) return -1
    BF("hc_attn_fn", "hc_attn_fn.weight");
    F32("hc_attn_base", "hc_attn_base.weight");
    F32("hc_attn_scale", "hc_attn_scale.weight");
    BF("hc_ffn_fn", "hc_ffn_fn.weight");
    F32("hc_ffn_base", "hc_ffn_base.weight");
    F32("hc_ffn_scale", "hc_ffn_scale.weight");
    BF("input_layernorm.weight", "attn_norm.weight");
    BF("post_attention_layernorm.weight", "ffn_norm.weight");
#undef F32
#undef BF
    return 0;
}

static int patch_kda(int blob_fd, const char *manifest,
                     const gguf_context *g, int layer) {
#define BF(SAFE, GGUF) if (patch_named(blob_fd, manifest, g, layer, SAFE, GGUF, 2, 0)) return -1
#define F32(SAFE, GGUF) if (patch_named(blob_fd, manifest, g, layer, SAFE, GGUF, 4, 0)) return -1
    if (patch_named(blob_fd, manifest, g, layer,
                    "self_attn.A_log", "ssm_a", 4, 1)) return -1;
    F32("self_attn.dt_bias", "ssm_dt.bias");
    BF("self_attn.q_proj.weight", "attn_q.weight");
    BF("self_attn.k_proj.weight", "attn_k.weight");
    BF("self_attn.v_proj.weight", "attn_v.weight");
    BF("self_attn.q_conv1d.weight", "ssm_conv1d_q.weight");
    BF("self_attn.k_conv1d.weight", "ssm_conv1d_k.weight");
    BF("self_attn.v_conv1d.weight", "ssm_conv1d_v.weight");
    BF("self_attn.f_a_proj.weight", "ssm_f_a.weight");
    BF("self_attn.f_b_proj.weight", "ssm_f_b.weight");
    BF("self_attn.b_proj.weight", "ssm_beta.weight");
    BF("self_attn.g_a_proj.weight", "ssm_g_a.weight");
    BF("self_attn.g_b_proj.weight", "ssm_g_b.weight");
    BF("self_attn.o_norm.weight", "ssm_norm.weight");
    BF("self_attn.o_proj.weight", "attn_output.weight");
#undef F32
#undef BF
#ifndef GLM53F_PP_CORE_STAGE
    if (layer < 3 &&
        (patch_fp8_named(blob_fd, manifest, g, layer,
             "mlp.gate_proj.weight", "mlp.gate_proj.weight_scale_inv",
             "ffn_gate.weight") ||
         patch_fp8_named(blob_fd, manifest, g, layer,
             "mlp.up_proj.weight", "mlp.up_proj.weight_scale_inv",
             "ffn_up.weight") ||
         patch_fp8_named(blob_fd, manifest, g, layer,
             "mlp.down_proj.weight", "mlp.down_proj.weight_scale_inv",
             "ffn_down.weight"))) return -1;
#endif
    return 0;
}

static int patch_sparse_kv_b(int blob_fd, const char *manifest,
                             const gguf_context *g, int layer) {
    enum { KD = 256, VD = 256, LATENT = 512 };
    char safe[768], kn[256], vn[256];
    image_entry e;
    snprintf(safe, sizeof(safe),
             "model.language_model.layers.%d.self_attn.kv_b_proj.weight", layer);
    snprintf(kn, sizeof(kn), "blk.%d.attn_k_b.weight", layer);
    snprintf(vn, sizeof(vn), "blk.%d.attn_v_b.weight", layer);
    tensor_ref kt = find_tensor(g, kn), vt = find_tensor(g, vn);
    if (!kt.info || !vt.info || find_entry(manifest, safe, &e) || e.kind != 'R' ||
        e.a % ((uint64_t)(KD + VD) * LATENT * sizeof(uint16_t)) ||
        e.b % ((uint64_t)(KD + VD) * LATENT * sizeof(uint16_t)) ||
        kt.info->dims[0] != KD || kt.info->dims[1] != LATENT ||
        vt.info->dims[0] != LATENT || vt.info->dims[1] != VD) return -1;
    int h0 = (int)(e.a / ((uint64_t)(KD + VD) * LATENT * sizeof(uint16_t)));
    int heads = (int)(e.b / ((uint64_t)(KD + VD) * LATENT * sizeof(uint16_t)));
    size_t krb = tensor_row_bytes(&kt), vrb = tensor_row_bytes(&vt);
    unsigned char *kraw = malloc(krb), *vraw = malloc(vrb);
    float *row = malloc(LATENT * sizeof(float));
    float *k = malloc((size_t)LATENT * KD * sizeof(float));
    uint16_t *out = malloc((size_t)(KD + VD) * LATENT * sizeof(uint16_t));
    if (!kraw || !vraw || !row || !k || !out) goto fail;
    for (int lh = 0; lh < heads; ++lh) {
        int h = h0 + lh;
        for (int d = 0; d < LATENT; ++d) {
            if (read_row(&kt, (uint64_t)h * LATENT + d, kraw, row)) goto fail;
            memcpy(k + (size_t)d * KD, row, KD * sizeof(float));
        }
        for (int j = 0; j < KD; ++j)
            for (int d = 0; d < LATENT; ++d)
                out[(size_t)j * LATENT + d] = bf16_rne(k[(size_t)d * KD + j]);
        for (int j = 0; j < VD; ++j) {
            if (read_row(&vt, (uint64_t)h * VD + j, vraw, row)) goto fail;
            for (int d = 0; d < LATENT; ++d)
                out[(size_t)(KD + j) * LATENT + d] = bf16_rne(row[d]);
        }
        if (write_exact(blob_fd,
                e.blob + (uint64_t)lh * (KD + VD) * LATENT * sizeof(uint16_t),
                out, (size_t)(KD + VD) * LATENT * sizeof(uint16_t))) goto fail;
    }
    release_tensor_cache(&kt);
    release_tensor_cache(&vt);
    free(out); free(k); free(row); free(vraw); free(kraw); return 0;
fail:
    free(out); free(k); free(row); free(vraw); free(kraw); return -1;
}

static int patch_sparse(int blob_fd, const char *manifest,
                        const gguf_context *g, int layer) {
#define BF(SAFE, GGUF) if (patch_named(blob_fd, manifest, g, layer, SAFE, GGUF, 2, 0)) return -1
    if (patch_fp8_named(blob_fd, manifest, g, layer,
            "self_attn.q_a_proj.weight", "self_attn.q_a_proj.weight_scale_inv",
            "attn_q_a.weight") ||
        patch_fp8_named(blob_fd, manifest, g, layer,
            "self_attn.q_b_proj.weight", "self_attn.q_b_proj.weight_scale_inv",
            "attn_q_b.weight") ||
        patch_fp8_named(blob_fd, manifest, g, layer,
            "self_attn.kv_a_proj_with_mqa.weight",
            "self_attn.kv_a_proj_with_mqa.weight_scale_inv",
            "attn_kv_a_mqa.weight") ||
        patch_fp8_named(blob_fd, manifest, g, layer,
            "self_attn.o_proj.weight", "self_attn.o_proj.weight_scale_inv",
            "attn_output.weight")) return -1;
    BF("self_attn.q_a_layernorm.weight", "attn_q_a_norm.weight");
    BF("self_attn.kv_a_layernorm.weight", "attn_kv_a_norm.weight");
    if (patch_sparse_kv_b(blob_fd, manifest, g, layer)) return -1;
    BF("self_attn.indexer.wk.weight", "indexer.attn_k.weight");
    BF("self_attn.indexer.k_norm.weight", "indexer.k_norm.weight");
    BF("self_attn.indexer.k_norm.bias", "indexer.k_norm.bias");
    BF("self_attn.indexer.index_kpool_compress_gate", "indexer_compressor_gate.weight");
    BF("self_attn.indexer.index_kpool_compress_ape", "indexer_compressor_ape.weight");
    BF("self_attn.indexer.wq_b.weight", "indexer.attn_q_b.weight");
    BF("self_attn.indexer.weights_proj.weight", "indexer.proj.weight");
#undef BF
    return 0;
}

static int patch_router_if_present(int blob_fd, const char *manifest,
                                   const gguf_context *g, int layer) {
    char safe[768];
    image_entry ignored;
    snprintf(safe, sizeof(safe),
             "model.language_model.layers.%d.mlp.gate.weight", layer);
    if (find_entry(manifest, safe, &ignored)) { errno = 0; return 0; }
    if (patch_named(blob_fd, manifest, g, layer, "mlp.gate.weight",
                    "ffn_gate_inp.weight", 2, 0) ||
        patch_named(blob_fd, manifest, g, layer,
                    "mlp.gate.e_score_correction_bias",
                    "exp_probs_b.bias", 4, 0)) return -1;
    return 0;
}

int main(int argc, char **argv) {
    char manifest[4096], blob[4096];
    int rank = -1, first = -1, last = -1, blob_fd = -1, rc = 1;
    gguf_context *g = NULL;
    if (argc != 5 || strncmp(argv[2], "/local/", 7) ||
        (rank = atoi(argv[3])) < 0 || rank >= RANKS) {
usage:
        fprintf(stderr, "usage: %s MODEL-00001-of-00004.gguf /local/CORE_DIR RANK LAYER[0..44]|all\n", argv[0]);
        return 2;
    }
    if (!strcmp(argv[4], "all")) { first = 0; last = 45; }
    else {
        char *end = NULL;
        long layer = strtol(argv[4], &end, 10);
        if (!end || *end || layer < 0 || layer >= 45) goto usage;
        first = (int)layer; last = first + 1;
    }
    snprintf(manifest, sizeof(manifest), "%s/rank%02d.core.manifest", argv[2], rank);
    snprintf(blob, sizeof(blob), "%s/rank%02d.core.blob", argv[2], rank);
    if (!(g = gguf_open_multi(argv[1], 3)) || (blob_fd = open(blob, O_RDWR)) < 0) goto done;
    for (int layer = first; layer < last; ++layer) {
        char type_name[128];
        snprintf(type_name, sizeof(type_name), "blk.%d.ssm_a", layer);
        if (patch_common(blob_fd, manifest, g, layer) ||
            (find_tensor(g, type_name).info ? patch_kda(blob_fd, manifest, g, layer)
                                           : patch_sparse(blob_fd, manifest, g, layer)) ||
            (layer >= 3 && patch_router_if_present(blob_fd, manifest, g, layer)))
            goto done;
        printf("GLM53F_Q2_CORE_PATCH rank=%d layer=%d\n", rank, layer);
        fflush(stdout);
    }
    if (fsync(blob_fd)) goto done;
    printf("SENTINEL glm53f_q2_core_patch=OK rank=%d layers=%d:%d\n",
           rank, first, last);
    rc = 0;
done:
    if (rc) fprintf(stderr, "glm53f_q2_core_patch failed rank=%d layers=%d:%d: %s\n",
                    rank, first, last,
                    errno ? strerror(errno) : "contract failure");
    if (blob_fd >= 0) close(blob_fd);
    if (g) gguf_close(g);
    return rc;
}
