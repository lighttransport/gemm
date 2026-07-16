/*
 * glm52_gguf_convert.c - streaming GLM-5.2 GGUF -> A64FX EP12 rank blob.
 *
 * This is deliberately a host-side tool. It mmaps GGUF shards without
 * MAP_POPULATE, touches one tensor row at a time, and writes one rank only.
 * Routed experts stay in their original GGML block format; dense tensors are
 * dequantized to BF16 after applying the rank's TP slice.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define GGUF_LOADER_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION
#include "gguf_loader.h"
#include "ggml_dequant.h"

#include <errno.h>
#include <glob.h>
#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

enum { GLM52_LAYERS = 78, GLM52_EXPERTS = 256, GLM52_HEADS = 64 };
enum { HIDDEN = 6144, Q_LORA = 2048, KV_LORA = 512, QK_NOPE = 192,
       QK_ROPE = 64, V_HEAD = 256, MOE_INTER = 2048,
       DENSE_INTER = 12288, VOCAB = 154880 };

typedef struct {
    gguf_context **shard;
    char **path;
    int n;
} source_set;

typedef struct {
    FILE *blob;
    FILE *manifest;
    uint64_t off;
    uint64_t hash;
    int dry_run;
    int rank;
} output;

typedef struct {
    gguf_context *ctx;
    gguf_tensor_info *ti;
    int index;
} tensor_ref;

static void fatal(const char *fmt, ...) {
    va_list ap;
    fprintf(stderr, "glm52-convert: FATAL: ");
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    exit(2);
}

static void pathf(char *dst, size_t cap, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(dst, cap, fmt, ap);
    va_end(ap);
    if (n < 0 || (size_t)n >= cap) fatal("path is too long");
}

static uint16_t f32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

static void shard_even(int total, int rank, int size, int *r0, int *rows) {
    int base = total / size, rem = total % size;
    *r0 = rank < rem ? rank * (base + 1)
                     : rem * (base + 1) + (rank - rem) * base;
    *rows = base + (rank < rem);
}

static void shard_blocks(int total, int block, int rank, int size, int *r0, int *rows) {
    int nb = (total + block - 1) / block, b0, bn;
    shard_even(nb, rank, size, &b0, &bn);
    *r0 = b0 * block;
    int hi = (b0 + bn) * block;
    if (hi > total) hi = total;
    if (*r0 > total) *r0 = total;
    *rows = hi - *r0;
}

static int has_full_indexer(int l) {
    return l >= 6 && ((l - 6) % 4) == 0;
}

static void drop_pages(const void *p, size_t n) {
    if (!p || !n) return;
    long ps = sysconf(_SC_PAGESIZE);
    if (ps <= 0) ps = 4096;
    uintptr_t a = (uintptr_t)p & ~(uintptr_t)(ps - 1);
    uintptr_t b = ((uintptr_t)p + n + (uintptr_t)ps - 1) & ~(uintptr_t)(ps - 1);
    if (b > a) madvise((void *)a, b - a, MADV_DONTNEED);
}

static void hash_bytes(output *o, const void *buf, size_t n) {
    const uint8_t *p = (const uint8_t *)buf;
    uint64_t h = o->hash;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= UINT64_C(1099511628211);
    }
    o->hash = h;
}

static void write_bytes(output *o, const void *buf, size_t n) {
    if (!o->dry_run) {
        if (fwrite(buf, 1, n, o->blob) != n)
            fatal("blob write failed: %s", strerror(errno));
        hash_bytes(o, buf, n);
    }
    o->off += n;
}

static void align_blob(output *o) {
    static const uint8_t zero[64] = {0};
    size_t pad = (size_t)((64 - (o->off & 63u)) & 63u);
    if (pad) write_bytes(o, zero, pad);
}

static uint64_t begin_tensor(output *o) {
    align_blob(o);
    return o->off;
}

static void record_tensor(output *o, uint64_t start, const char *dtype,
                          int nd, const long *shape, const char *name) {
    if (o->dry_run) return;
    fprintf(o->manifest, "%" PRIu64 " %" PRIu64 " %s %d",
            start, o->off - start, dtype, nd);
    for (int i = 0; i < nd; ++i) fprintf(o->manifest, " %ld", shape[i]);
    fprintf(o->manifest, " %s\n", name);
}

static tensor_ref find_tensor(source_set *s, const char *name) {
    tensor_ref z = {0};
    for (int j = 0; j < s->n; ++j) {
        gguf_context *c = s->shard[j];
        for (uint64_t i = 0; i < c->n_tensors; ++i) {
            if (strcmp(c->tensors[i].name.str, name) == 0) {
                z.ctx = c;
                z.ti = &c->tensors[i];
                z.index = (int)i;
                return z;
            }
        }
    }
    fatal("missing GGUF tensor %s", name);
    return z;
}

static size_t row_bytes(uint32_t type, int cols) {
    if (type >= GGML_TYPE_COUNT || ggml_type_info[type].block_size <= 0)
        fatal("unsupported GGML type %u", type);
    int bs = ggml_type_info[type].block_size;
    if (cols % bs) fatal("column count %d is not divisible by block %d", cols, bs);
    return (size_t)(cols / bs) * ggml_type_info[type].type_size;
}

static void require_matrix(const tensor_ref *r, int cols, int rows, int depth,
                           const char *name) {
    gguf_tensor_info *t = r->ti;
    uint64_t got0 = t->n_dims > 0 ? t->dims[0] : 0;
    uint64_t got1 = t->n_dims > 1 ? t->dims[1] : 1;
    uint64_t got2 = t->n_dims > 2 ? t->dims[2] : 1;
    if ((int)got0 != cols || (int)got1 != rows || (int)got2 != depth)
        fatal("shape mismatch %s: got [%" PRIu64 ",%" PRIu64 ",%" PRIu64
              "] expected [%d,%d,%d]", name, got0, got1, got2,
              cols, rows, depth);
}

/* mode: 0 full, 1 row slice, 2 column slice. GGUF ne0 is row columns. */
static void emit_bf16_matrix(output *o, source_set *s, const char *src_name,
                             const char *dst_name, int full_rows, int full_cols,
                             int mode, int start, int count) {
    tensor_ref r = find_tensor(s, src_name);
    require_matrix(&r, full_cols, full_rows, 1, src_name);
    int out_rows = mode == 1 ? count : full_rows;
    int out_cols = mode == 2 ? count : full_cols;
    long shape[2] = {out_rows, out_cols};
    uint64_t at = begin_tensor(o);
    if (o->dry_run) {
        o->off += (uint64_t)out_rows * out_cols * 2;
        return;
    }
    const uint8_t *base = (const uint8_t *)gguf_tensor_data(r.ctx, r.index);
    size_t rb = row_bytes(r.ti->type, full_cols);
    float *tmp = malloc((size_t)full_cols * sizeof(float));
    uint16_t *bf = malloc((size_t)out_cols * sizeof(uint16_t));
    if (!tmp || !bf) fatal("out of memory converting %s", src_name);
    int r0 = mode == 1 ? start : 0;
    for (int rr = 0; rr < out_rows; ++rr) {
        const void *sp = base + (size_t)(r0 + rr) * rb;
        if (dequant_row(r.ti->type, sp, tmp, full_cols))
            fatal("cannot dequantize %s (%s)", src_name,
                  ggml_type_name(r.ti->type));
        int c0 = mode == 2 ? start : 0;
        for (int c = 0; c < out_cols; ++c)
            bf[c] = f32_to_bf16(tmp[c0 + c]);
        write_bytes(o, bf, (size_t)out_cols * 2);
    }
    drop_pages(base + (size_t)r0 * rb, (size_t)out_rows * rb);
    free(tmp);
    free(bf);
    record_tensor(o, at, "BF16", 2, shape, dst_name);
}

static void emit_bf16_vector(output *o, source_set *s, const char *src_name,
                             const char *dst_name, int n) {
    tensor_ref r = find_tensor(s, src_name);
    require_matrix(&r, n, 1, 1, src_name);
    long shape[1] = {n};
    uint64_t at = begin_tensor(o);
    if (o->dry_run) {
        o->off += (uint64_t)n * 2;
        return;
    }
    const void *src = gguf_tensor_data(r.ctx, r.index);
    float *tmp = malloc((size_t)n * sizeof(float));
    uint16_t *bf = malloc((size_t)n * sizeof(uint16_t));
    if (!tmp || !bf) fatal("out of memory converting %s", src_name);
    if (dequant_row(r.ti->type, src, tmp, n))
        fatal("cannot dequantize %s", src_name);
    for (int i = 0; i < n; ++i) bf[i] = f32_to_bf16(tmp[i]);
    write_bytes(o, bf, (size_t)n * 2);
    drop_pages(src, gguf_tensor_size(r.ctx, r.index));
    free(tmp);
    free(bf);
    record_tensor(o, at, "BF16", 1, shape, dst_name);
}

static void emit_f32(output *o, source_set *s, const char *src_name,
                     const char *dst_name, int rows, int cols) {
    tensor_ref r = find_tensor(s, src_name);
    require_matrix(&r, cols, rows, 1, src_name);
    if (r.ti->type != GGML_TYPE_F32) fatal("%s must be F32", src_name);
    long shape[2] = {rows, cols};
    uint64_t at = begin_tensor(o);
    size_t n = (size_t)rows * cols * 4;
    if (o->dry_run) {
        o->off += n;
        return;
    }
    const void *src = gguf_tensor_data(r.ctx, r.index);
    write_bytes(o, src, n);
    drop_pages(src, n);
    record_tensor(o, at, "F32", rows == 1 ? 1 : 2, shape, dst_name);
}

static void emit_raw_expert(output *o, source_set *s, const char *src_name,
                            const char *dst_name, int expert,
                            int rows, int cols) {
    tensor_ref r = find_tensor(s, src_name);
    require_matrix(&r, cols, rows, GLM52_EXPERTS, src_name);
    size_t rb = row_bytes(r.ti->type, cols);
    size_t bytes = (size_t)rows * rb;
    long shape[2] = {rows, cols};
    uint64_t at = begin_tensor(o);
    if (o->dry_run) {
        o->off += bytes;
        return;
    }
    const uint8_t *src = (const uint8_t *)gguf_tensor_data(r.ctx, r.index)
                       + (size_t)expert * bytes;
    write_bytes(o, src, bytes);
    drop_pages(src, bytes);
    record_tensor(o, at, ggml_type_name(r.ti->type), 2, shape, dst_name);
}

/* Normalize split llama.cpp K/V tensors to legacy [head*(192+256),512]. */
static void emit_combined_kv_b(output *o, source_set *s, int layer,
                               int h0, int hn, const char *dst_name) {
    char kn[128], vn[128];
    snprintf(kn, sizeof kn, "blk.%d.attn_k_b.weight", layer);
    snprintf(vn, sizeof vn, "blk.%d.attn_v_b.weight", layer);
    tensor_ref kr = find_tensor(s, kn), vr = find_tensor(s, vn);
    require_matrix(&kr, QK_NOPE, KV_LORA, GLM52_HEADS, kn);
    require_matrix(&vr, KV_LORA, V_HEAD, GLM52_HEADS, vn);
    int rows = hn * (QK_NOPE + V_HEAD);
    long shape[2] = {rows, KV_LORA};
    uint64_t at = begin_tensor(o);
    if (o->dry_run) {
        o->off += (uint64_t)rows * KV_LORA * 2;
        return;
    }
    size_t krb = row_bytes(kr.ti->type, QK_NOPE);
    size_t vrb = row_bytes(vr.ti->type, KV_LORA);
    size_t khead = (size_t)KV_LORA * krb;
    size_t vhead = (size_t)V_HEAD * vrb;
    const uint8_t *kb = gguf_tensor_data(kr.ctx, kr.index);
    const uint8_t *vb = gguf_tensor_data(vr.ctx, vr.index);
    float *kt = malloc((size_t)KV_LORA * QK_NOPE * sizeof(float));
    float *tmp = malloc((size_t)KV_LORA * sizeof(float));
    uint16_t *bf = malloc((size_t)KV_LORA * sizeof(uint16_t));
    if (!kt || !tmp || !bf)
        fatal("out of memory converting layer %d K/V", layer);
    for (int hh = h0; hh < h0 + hn; ++hh) {
        const uint8_t *kh = kb + (size_t)hh * khead;
        for (int r = 0; r < KV_LORA; ++r) {
            if (dequant_row(kr.ti->type, kh + (size_t)r * krb,
                            kt + (size_t)r * QK_NOPE, QK_NOPE))
                fatal("cannot dequantize %s", kn);
        }
        for (int r = 0; r < QK_NOPE; ++r) {
            for (int c = 0; c < KV_LORA; ++c)
                bf[c] = f32_to_bf16(kt[(size_t)c * QK_NOPE + r]);
            write_bytes(o, bf, (size_t)KV_LORA * 2);
        }
        const uint8_t *vh = vb + (size_t)hh * vhead;
        for (int r = 0; r < V_HEAD; ++r) {
            if (dequant_row(vr.ti->type, vh + (size_t)r * vrb,
                            tmp, KV_LORA))
                fatal("cannot dequantize %s", vn);
            for (int c = 0; c < KV_LORA; ++c)
                bf[c] = f32_to_bf16(tmp[c]);
            write_bytes(o, bf, (size_t)KV_LORA * 2);
        }
        drop_pages(kh, khead);
        drop_pages(vh, vhead);
    }
    free(kt);
    free(tmp);
    free(bf);
    record_tensor(o, at, "BF16", 2, shape, dst_name);
}

static source_set open_sources(const char *dir) {
    char pattern[PATH_MAX];
    pathf(pattern, sizeof pattern, "%s/*.gguf", dir);
    glob_t g;
    memset(&g, 0, sizeof g);
    if (glob(pattern, 0, NULL, &g) || g.gl_pathc == 0)
        fatal("no GGUF shards under %s", dir);
    source_set s = {0};
    s.n = (int)g.gl_pathc;
    s.shard = calloc((size_t)s.n, sizeof *s.shard);
    s.path = calloc((size_t)s.n, sizeof *s.path);
    if (!s.shard || !s.path) fatal("out of memory opening shards");
    setenv("NUMA_DISTRIBUTE", "1", 1); /* suppress gguf_loader MAP_POPULATE */
    setenv("TF_FORCE_MMAP", "1", 1);
    for (int i = 0; i < s.n; ++i) {
        s.path[i] = strdup(g.gl_pathv[i]);
        s.shard[i] = gguf_open(g.gl_pathv[i], 1);
        if (!s.shard[i]) fatal("cannot parse %s", g.gl_pathv[i]);
    }
    globfree(&g);
    return s;
}

static void close_sources(source_set *s) {
    for (int i = 0; i < s->n; ++i) {
        gguf_close(s->shard[i]);
        free(s->path[i]);
    }
    free(s->shard);
    free(s->path);
}

static void emit_model(output *o, source_set *s, int ep_size, int layers) {
    int vr0, vrows, h0, hcount, sh0, shrows, ff0, ffrows;
    shard_even(VOCAB, o->rank, ep_size, &vr0, &vrows);
    shard_even(GLM52_HEADS, o->rank, ep_size, &h0, &hcount);
    shard_blocks(MOE_INTER, 128, o->rank, ep_size, &sh0, &shrows);
    shard_even(DENSE_INTER, o->rank, ep_size, &ff0, &ffrows);

    emit_bf16_matrix(o, s, "token_embd.weight", "model.embed_tokens.weight",
                     VOCAB, HIDDEN, 1, vr0, vrows);
    emit_bf16_matrix(o, s, "output.weight", "lm_head.weight",
                     VOCAB, HIDDEN, 1, vr0, vrows);
    emit_bf16_vector(o, s, "output_norm.weight", "model.norm.weight", HIDDEN);

    for (int l = 0; l < layers; ++l) {
        char src[192], dst[256];
#define EMIT_VEC(SRC, DST, N) do { \
        snprintf(src,sizeof src,"blk.%d.%s",l,(SRC)); \
        snprintf(dst,sizeof dst,"model.layers.%d.%s",l,(DST)); \
        emit_bf16_vector(o,s,src,dst,(N)); \
    } while (0)
#define EMIT_MAT(SRC, DST, R, C, MODE, START, COUNT) do { \
        snprintf(src,sizeof src,"blk.%d.%s",l,(SRC)); \
        snprintf(dst,sizeof dst,"model.layers.%d.%s",l,(DST)); \
        emit_bf16_matrix(o,s,src,dst,(R),(C),(MODE),(START),(COUNT)); \
    } while (0)
        EMIT_VEC("attn_norm.weight", "input_layernorm.weight", HIDDEN);
        EMIT_VEC("ffn_norm.weight", "post_attention_layernorm.weight", HIDDEN);
        EMIT_MAT("attn_q_a.weight", "self_attn.q_a_proj.weight",
                 Q_LORA, HIDDEN, 0, 0, 0);
        EMIT_MAT("attn_q_b.weight", "self_attn.q_b_proj.weight",
                 GLM52_HEADS * (QK_NOPE + QK_ROPE), Q_LORA,
                 1, h0 * (QK_NOPE + QK_ROPE),
                 hcount * (QK_NOPE + QK_ROPE));
        EMIT_MAT("attn_kv_a_mqa.weight",
                 "self_attn.kv_a_proj_with_mqa.weight",
                 KV_LORA + QK_ROPE, HIDDEN, 0, 0, 0);
        snprintf(dst,sizeof dst,
                 "model.layers.%d.self_attn.kv_b_proj.weight",l);
        emit_combined_kv_b(o,s,l,h0,hcount,dst);
        EMIT_MAT("attn_output.weight", "self_attn.o_proj.weight",
                 HIDDEN, GLM52_HEADS * V_HEAD,
                 2, h0 * V_HEAD, hcount * V_HEAD);
        EMIT_VEC("attn_q_a_norm.weight",
                 "self_attn.q_a_layernorm.weight", Q_LORA);
        EMIT_VEC("attn_kv_a_norm.weight",
                 "self_attn.kv_a_layernorm.weight", KV_LORA);

        if (l < 3) {
            EMIT_MAT("ffn_gate.weight", "mlp.gate_proj.weight",
                     DENSE_INTER, HIDDEN, 1, ff0, ffrows);
            EMIT_MAT("ffn_up.weight", "mlp.up_proj.weight",
                     DENSE_INTER, HIDDEN, 1, ff0, ffrows);
            EMIT_MAT("ffn_down.weight", "mlp.down_proj.weight",
                     HIDDEN, DENSE_INTER, 2, ff0, ffrows);
        } else {
            if (has_full_indexer(l)) {
                EMIT_MAT("indexer.attn_q_b.weight",
                         "self_attn.indexer.wq_b.weight",
                         4096, Q_LORA, 0, 0, 0);
                EMIT_MAT("indexer.attn_k.weight",
                         "self_attn.indexer.wk.weight",
                         128, HIDDEN, 0, 0, 0);
                EMIT_MAT("indexer.proj.weight",
                         "self_attn.indexer.weights_proj.weight",
                         32, HIDDEN, 0, 0, 0);
                EMIT_VEC("indexer.k_norm.weight",
                         "self_attn.indexer.k_norm.weight", 128);
                EMIT_VEC("indexer.k_norm.bias",
                         "self_attn.indexer.k_norm.bias", 128);
            }
            snprintf(src,sizeof src,"blk.%d.ffn_gate_inp.weight",l);
            snprintf(dst,sizeof dst,"model.layers.%d.mlp.gate.weight",l);
            emit_f32(o,s,src,dst,GLM52_EXPERTS,HIDDEN);
            snprintf(src,sizeof src,"blk.%d.exp_probs_b.bias",l);
            snprintf(dst,sizeof dst,
                     "model.layers.%d.mlp.gate.e_score_correction_bias",l);
            emit_f32(o,s,src,dst,1,GLM52_EXPERTS);
            EMIT_MAT("ffn_gate_shexp.weight",
                     "mlp.shared_experts.gate_proj.weight",
                     MOE_INTER, HIDDEN, 1, sh0, shrows);
            EMIT_MAT("ffn_up_shexp.weight",
                     "mlp.shared_experts.up_proj.weight",
                     MOE_INTER, HIDDEN, 1, sh0, shrows);
            EMIT_MAT("ffn_down_shexp.weight",
                     "mlp.shared_experts.down_proj.weight",
                     HIDDEN, MOE_INTER, 2, sh0, shrows);
            for (int e = o->rank; e < GLM52_EXPERTS; e += ep_size) {
                snprintf(src,sizeof src,"blk.%d.ffn_gate_exps.weight",l);
                snprintf(dst,sizeof dst,
                         "model.layers.%d.mlp.experts.%d.gate_proj.weight",l,e);
                emit_raw_expert(o,s,src,dst,e,MOE_INTER,HIDDEN);
                snprintf(src,sizeof src,"blk.%d.ffn_up_exps.weight",l);
                snprintf(dst,sizeof dst,
                         "model.layers.%d.mlp.experts.%d.up_proj.weight",l,e);
                emit_raw_expert(o,s,src,dst,e,MOE_INTER,HIDDEN);
                snprintf(src,sizeof src,"blk.%d.ffn_down_exps.weight",l);
                snprintf(dst,sizeof dst,
                         "model.layers.%d.mlp.experts.%d.down_proj.weight",l,e);
                emit_raw_expert(o,s,src,dst,e,HIDDEN,MOE_INTER);
            }
        }
#undef EMIT_MAT
#undef EMIT_VEC
        fprintf(stderr, "glm52-convert: rank %d layer %d/%d %.3f GiB\n",
                o->rank, l + 1, layers,
                (double)o->off / (1024.0*1024.0*1024.0));
    }
}

static void usage(const char *argv0) {
    fprintf(stderr,
        "usage: %s [--source DIR] [--output DIR] --rank R [--ep-size 12]\n"
        "          [--layers 78] [--maxpos 2304] [--dry-run] [--force]\n",
        argv0);
    exit(2);
}

int main(int argc, char **argv) {
    const char *home = getenv("HOME");
    char source[PATH_MAX], outdir[PATH_MAX];
    pathf(source,sizeof source,"%s/models/glm52-2bit",home?home:".");
    outdir[0] = 0;
    int rank = -1, ep_size = 12, layers = GLM52_LAYERS;
    int maxpos = 2304, dry = 0, force = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i],"--source") && ++i < argc)
            pathf(source,sizeof source,"%s",argv[i]);
        else if (!strcmp(argv[i],"--output") && ++i < argc)
            pathf(outdir,sizeof outdir,"%s",argv[i]);
        else if (!strcmp(argv[i],"--rank") && ++i < argc) rank=atoi(argv[i]);
        else if (!strcmp(argv[i],"--ep-size") && ++i < argc) ep_size=atoi(argv[i]);
        else if (!strcmp(argv[i],"--layers") && ++i < argc) layers=atoi(argv[i]);
        else if (!strcmp(argv[i],"--maxpos") && ++i < argc) maxpos=atoi(argv[i]);
        else if (!strcmp(argv[i],"--dry-run")) dry=1;
        else if (!strcmp(argv[i],"--force")) force=1;
        else usage(argv[0]);
    }
    if(rank<0){
        const char*keys[]={"GLM52_RANK","PMIX_RANK","OMPI_COMM_WORLD_RANK",
                           "PMI_RANK","MV2_COMM_WORLD_RANK",NULL};
        for(int i=0;keys[i]&&rank<0;i++){ const char*v=getenv(keys[i]); if(v&&*v)rank=atoi(v); }
    }
    if (!outdir[0])
        pathf(outdir,sizeof outdir,"%s/a64fx-ep12-v1",source);
    if (rank < 0 || rank >= ep_size || ep_size != 12 ||
        layers < 1 || layers > GLM52_LAYERS) usage(argv[0]);

    source_set sources = open_sources(source);
    output o = {.off=0,.hash=UINT64_C(1469598103934665603),
                .dry_run=dry,.rank=rank};
    char blob[PATH_MAX], mani[PATH_MAX], btmp[PATH_MAX], mtmp[PATH_MAX];
    pathf(blob,sizeof blob,"%s/rank%02d.blob",outdir,rank);
    pathf(mani,sizeof mani,"%s/rank%02d.manifest",outdir,rank);
    pathf(btmp,sizeof btmp,"%s/.rank%02d.blob.tmp.%ld",outdir,rank,(long)getpid());
    pathf(mtmp,sizeof mtmp,"%s/.rank%02d.manifest.tmp.%ld",outdir,rank,(long)getpid());
    if (!dry) {
        if (mkdir(outdir,0755) && errno!=EEXIST)
            fatal("mkdir %s: %s",outdir,strerror(errno));
        struct statvfs pre;
        if (!statvfs(outdir,&pre)) {
            uint64_t freeb=(uint64_t)pre.f_bavail*pre.f_frsize;
            /* Largest observed rank is 24.968 GiB. Leave at least two GiB
             * beyond the temporary blob so an ENOSPC cannot strand a nearly
             * complete 25 GiB conversion. */
            if (freeb < UINT64_C(27)*1024*1024*1024)
                fatal("need at least 27 GiB free under %s (have %.2f GiB)",
                      outdir,(double)freeb/(1024.0*1024.0*1024.0));
        }
        if (!force && access(blob,F_OK)==0 && access(mani,F_OK)==0)
            fatal("rank %d output exists (use --force)",rank);
        o.blob=fopen(btmp,"wb");
        o.manifest=fopen(mtmp,"w");
        if(!o.blob||!o.manifest)
            fatal("cannot create output under %s: %s",outdir,strerror(errno));
        fprintf(o.manifest,
                "# glm52-a64fx-ep12-v1\n# rank %d\n# ep_size %d\n"
                "# layers %d\n# maxpos %d\n",
                rank,ep_size,layers,maxpos);
    }
    emit_model(&o,&sources,ep_size,layers);
    uint64_t reserve = UINT64_C(3)*1024*1024*1024;
    uint64_t predicted = o.off + reserve;
    fprintf(stderr,
            "glm52-convert: rank %d blob=%.3f GiB predicted_peak=%.3f GiB\n",
            rank,(double)o.off/(1024.0*1024.0*1024.0),
            (double)predicted/(1024.0*1024.0*1024.0));
    if (predicted > UINT64_C(29)*1024*1024*1024)
        fatal("rank %d predicted peak exceeds 29 GiB",rank);
    if (!dry) {
        fprintf(o.manifest,
                "# blob_bytes %" PRIu64 "\n# runtime_reserve_bytes %" PRIu64
                "\n# fnv1a64 %016" PRIx64 "\n",o.off,reserve,o.hash);
        if (fflush(o.blob)||fsync(fileno(o.blob))||fclose(o.blob))
            fatal("cannot sync %s",btmp);
        if (fflush(o.manifest)||fsync(fileno(o.manifest))||
            fclose(o.manifest)) fatal("cannot sync %s",mtmp);
        struct statvfs sv;
        if (!statvfs(outdir,&sv)) {
            uint64_t freeb=(uint64_t)sv.f_bavail*sv.f_frsize;
            if (freeb < o.off/20)
                fatal("less than 5%% blob-size free space remains in %s",outdir);
        }
        if (rename(btmp,blob)||rename(mtmp,mani))
            fatal("atomic publish failed: %s",strerror(errno));
    }
    close_sources(&sources);
    return 0;
}
