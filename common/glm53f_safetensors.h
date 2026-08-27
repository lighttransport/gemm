/* GLM5.3F sharded safetensors metadata/lookup layer.
 *
 * This is deliberately a metadata-first boundary for the future A64FX graph:
 * it opens the shard mappings but does not copy or dequantize weights.  A
 * graph loader can resolve a tensor by its Hugging Face name, inspect dtype and
 * shape, then stream only the rank-owned payload into its arena.
 *
 * Usage in one translation unit:
 *   #define SAFETENSORS_IMPLEMENTATION
 *   #define GLM53F_SAFETENSORS_IMPLEMENTATION
 *   #include "glm53f_safetensors.h"
 */
#ifndef GLM53F_SAFETENSORS_H
#define GLM53F_SAFETENSORS_H

#include <stddef.h>
#include <stdint.h>
#include "safetensors.h"

typedef struct {
    char *name;
    st_context *st;
} glm53f_st_shard;

typedef struct {
    char *name;
    int shard;
    int tensor;
} glm53f_st_entry;

typedef struct {
    char *model_dir;
    glm53f_st_shard *shards;
    int n_shards;
    glm53f_st_entry *entries;
    int n_entries;
} glm53f_st_context;

glm53f_st_context *glm53f_st_open(const char *model_dir);
void glm53f_st_close(glm53f_st_context *ctx);
const st_tensor_info *glm53f_st_find(const glm53f_st_context *ctx, const char *name,
                                     const st_context **owner);
int glm53f_st_validate_contract(const glm53f_st_context *ctx, int verbose);

/* Validate one role before binding it to a graph buffer.  Shapes are in
 * safetensors row-major order (the checkpoint's output/input convention). */
int glm53f_st_expect(const glm53f_st_context *ctx, const char *name,
                     const char *dtype, int n_dims, const uint64_t *shape);
/* Read a bounded tensor slice without mapping the shard payload. */
int glm53f_st_read(const glm53f_st_context *ctx, const char *name,
                   size_t offset, void *dst, size_t nbytes);

#ifdef GLM53F_SAFETENSORS_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *glm53f_st_dup(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = (char *)malloc(n);
    if (p) memcpy(p, s, n);
    return p;
}

static int glm53f_st_read_file(const char *path, char **out, int *out_len) {
    FILE *f = fopen(path, "rb");
    long n;
    char *p;
    if (!f) return -1;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    n = ftell(f);
    if (n < 0 || n > 128L * 1024L * 1024L) { fclose(f); return -1; }
    rewind(f);
    p = (char *)malloc((size_t)n + 1);
    if (!p || fread(p, 1, (size_t)n, f) != (size_t)n) {
        free(p); fclose(f); return -1;
    }
    fclose(f); p[n] = '\0'; *out = p; *out_len = (int)n; return 0;
}

static int glm53f_st_shard_id(glm53f_st_context *ctx, const char *name) {
    int i;
    for (i = 0; i < ctx->n_shards; ++i)
        if (!strcmp(ctx->shards[i].name, name)) return i;
    return -1;
}

static int glm53f_st_add_shard(glm53f_st_context *ctx, const char *dir, const char *name) {
    char path[4096];
    int id = glm53f_st_shard_id(ctx, name);
    if (id >= 0) return id;
    id = ctx->n_shards++;
    ctx->shards = (glm53f_st_shard *)realloc(ctx->shards,
                                             (size_t)ctx->n_shards * sizeof(*ctx->shards));
    if (!ctx->shards) return -1;
    ctx->shards[id].name = glm53f_st_dup(name);
    if (!ctx->shards[id].name) return -1;
    if (snprintf(path, sizeof(path), "%s/%s", dir, name) >= (int)sizeof(path)) return -1;
    /* Metadata-only open is essential here: the 62 payloads are far larger
     * than a login node's address-space/mapping budget.  The future graph
     * loader can reopen an owned shard with safetensors_open when needed. */
    ctx->shards[id].st = safetensors_open_header(path);
    if (!ctx->shards[id].st) return -1;
    return id;
}

glm53f_st_context *glm53f_st_open(const char *model_dir) {
    char path[4096], *src = NULL;
    int len = 0, i;
    json_val *root = NULL, *wm;
    glm53f_st_context *ctx = NULL;
    if (snprintf(path, sizeof(path), "%s/model.safetensors.index.json", model_dir) >= (int)sizeof(path) ||
        glm53f_st_read_file(path, &src, &len) != 0) return NULL;
    root = json_parse(src, len); free(src);
    if (!root || root->type != JSON_OBJECT || !(wm = json_obj_get(root, "weight_map")) ||
        wm->type != JSON_OBJECT) { json_free(root); return NULL; }
    ctx = (glm53f_st_context *)calloc(1, sizeof(*ctx));
    if (!ctx) { json_free(root); return NULL; }
    ctx->model_dir = glm53f_st_dup(model_dir);
    if (!ctx->model_dir) goto fail;
    for (i = 0; i < wm->obj.count; ++i) {
        json_val *v = &wm->obj.vals[i];
        int sid, tid;
        if (v->type != JSON_STRING) goto fail;
        sid = glm53f_st_add_shard(ctx, model_dir, v->str.ptr);
        if (sid < 0) goto fail;
        tid = safetensors_find(ctx->shards[sid].st, wm->obj.keys[i]);
        if (tid < 0) goto fail;
        ctx->entries = (glm53f_st_entry *)realloc(ctx->entries,
                          (size_t)(ctx->n_entries + 1) * sizeof(*ctx->entries));
        if (!ctx->entries) goto fail;
        ctx->entries[ctx->n_entries].name = glm53f_st_dup(wm->obj.keys[i]);
        ctx->entries[ctx->n_entries].shard = sid;
        ctx->entries[ctx->n_entries].tensor = tid;
        if (!ctx->entries[ctx->n_entries].name) goto fail;
        ctx->n_entries++;
    }
    json_free(root);
    return ctx;
fail:
    json_free(root); glm53f_st_close(ctx); return NULL;
}

void glm53f_st_close(glm53f_st_context *ctx) {
    int i;
    if (!ctx) return;
    for (i = 0; i < ctx->n_entries; ++i) free(ctx->entries[i].name);
    for (i = 0; i < ctx->n_shards; ++i) {
        free(ctx->shards[i].name);
        safetensors_close(ctx->shards[i].st);
    }
    free(ctx->model_dir); free(ctx->entries); free(ctx->shards); free(ctx);
}

const st_tensor_info *glm53f_st_find(const glm53f_st_context *ctx, const char *name,
                                     const st_context **owner) {
    int i;
    if (owner) *owner = NULL;
    if (!ctx || !name) return NULL;
    for (i = 0; i < ctx->n_entries; ++i)
        if (!strcmp(ctx->entries[i].name, name)) {
            if (owner) *owner = ctx->shards[ctx->entries[i].shard].st;
            return &ctx->shards[ctx->entries[i].shard].st->tensors[ctx->entries[i].tensor];
        }
    return NULL;
}

int glm53f_st_expect(const glm53f_st_context *ctx, const char *name,
                     const char *dtype, int n_dims, const uint64_t *shape) {
    const st_tensor_info *t = glm53f_st_find(ctx, name, NULL);
    int d;
    if (!t || (dtype && strcmp(t->dtype_str, dtype)) || t->n_dims != n_dims) return -1;
    for (d = 0; d < n_dims; ++d) if (t->shape[d] != shape[d]) return -1;
    return 0;
}

int glm53f_st_read(const glm53f_st_context *ctx, const char *name,
                   size_t offset, void *dst, size_t nbytes) {
    const st_context *owner = NULL;
    const st_tensor_info *t = glm53f_st_find(ctx, name, &owner);
    int i, fd, rc = -1;
    char path[4096];
    if (!t || !owner || offset > t->nbytes || nbytes > t->nbytes - offset || !dst) return -1;
    for (i = 0; i < ctx->n_shards; ++i) if (ctx->shards[i].st == owner) break;
    if (i == ctx->n_shards || snprintf(path, sizeof(path), "%s/%s", ctx->model_dir,
                                        ctx->shards[i].name) >= (int)sizeof(path)) return -1;
    fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    if (pread(fd, dst, nbytes, (off_t)(owner->data_offset + t->offset + offset)) == (ssize_t)nbytes)
        rc = 0;
    close(fd);
    return rc;
}

int glm53f_st_validate_contract(const glm53f_st_context *ctx, int verbose) {
    const char *required[] = {
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
        "model.language_model.layers.0.input_layernorm.weight",
        "model.language_model.layers.3.self_attn.indexer.k_norm.weight",
        "model.language_model.layers.3.mlp.gate.weight"
    };
    int i, missing = 0, layers = 0, experts = 0, shape_errors = 0;
    char name[128];
    static const uint64_t s_embed[] = {154880, 4096};
    static const uint64_t s_hidden[] = {4096};
    static const uint64_t s_l0_q[] = {8192, 4096};
    static const uint64_t s_l0_ffn[] = {12288, 4096};
    static const uint64_t s_l3_qa[] = {1536, 4096};
    static const uint64_t s_l3_qb[] = {16384, 1536};
    static const uint64_t s_l3_kvb[] = {32768, 512};
    static const uint64_t s_l3_gate[] = {288, 4096};
    static const uint64_t s_mtp[] = {4096, 8192};
    struct spec { const char *n, *dt; int nd; const uint64_t *sh; };
    static const struct spec specs[] = {
        {"model.language_model.embed_tokens.weight", "BF16", 2, s_embed},
        {"model.language_model.norm.weight", "BF16", 1, s_hidden},
        {"lm_head.weight", "BF16", 2, s_embed},
        {"model.language_model.layers.0.self_attn.q_proj.weight", "BF16", 2, s_l0_q},
        {"model.language_model.layers.0.mlp.gate_proj.weight", "F8_E4M3", 2, s_l0_ffn},
        {"model.language_model.layers.3.self_attn.q_a_proj.weight", "F8_E4M3", 2, s_l3_qa},
        {"model.language_model.layers.3.self_attn.q_b_proj.weight", "F8_E4M3", 2, s_l3_qb},
        {"model.language_model.layers.3.self_attn.kv_b_proj.weight", "BF16", 2, s_l3_kvb},
        {"model.language_model.layers.3.mlp.gate.weight", "BF16", 2, s_l3_gate},
        {"model.language_model.layers.45.eh_proj.weight", "BF16", 2, s_mtp}
    };
    for (i = 0; i < (int)(sizeof(required) / sizeof(required[0])); ++i)
        if (!glm53f_st_find(ctx, required[i], NULL)) { missing++; if (verbose) fprintf(stderr, "missing %s\n", required[i]); }
    for (i = 0; i < (int)(sizeof(specs) / sizeof(specs[0])); ++i)
        if (glm53f_st_expect(ctx, specs[i].n, specs[i].dt, specs[i].nd, specs[i].sh) != 0) {
            shape_errors++;
            if (verbose) fprintf(stderr, "shape/dtype mismatch %s (expected %s)\n", specs[i].n, specs[i].dt);
        }
    for (i = 0; i <= 45; ++i) {
        snprintf(name, sizeof(name), "model.language_model.layers.%d.input_layernorm.weight", i);
        if (glm53f_st_find(ctx, name, NULL)) layers++;
    }
    for (i = 3; i <= 45; ++i) {
        snprintf(name, sizeof(name), "model.language_model.layers.%d.mlp.experts.0.gate_proj.weight", i);
        if (glm53f_st_find(ctx, name, NULL)) experts++;
    }
    if (verbose) fprintf(stderr, "glm53f_st: entries=%d shards=%d layers=%d moe_layers=%d\n",
                         ctx ? ctx->n_entries : 0, ctx ? ctx->n_shards : 0, layers, experts);
    return ctx && missing == 0 && shape_errors == 0 && layers == 46 && experts == 43 ? 0 : -1;
}

#endif
#endif
