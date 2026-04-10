/*
 * flux2_klein_text_encoder.h - Text encoder wrapper for Flux.2 Klein
 *
 * Wraps transformer.h (Qwen3-4B arch) to extract per-token hidden states
 * as conditioning for the Flux.2 Klein DiT.
 *
 * Usage:
 *   #define FLUX2_TEXT_ENCODER_IMPLEMENTATION
 *   #include "flux2_klein_text_encoder.h"
 *
 * Dependencies: transformer.h, bpe_tokenizer.h, safetensors.h, gguf_loader.h
 *
 * Weight format: qwen_3_4b.safetensors (BF16, ComfyUI distribution)
 * Tokenizer: any Qwen3 GGUF (vocab/merges only, not weights), e.g.:
 *   /mnt/disk01/models/Qwen3-VL-4B-Instruct-GGUF/Qwen3VL-4B-Instruct-Q8_0.gguf
 *
 * API:
 *   flux2_text_enc *flux2_text_enc_load_safetensors(const char *st_path,
 *                                                    const char *tok_gguf_path);
 *   flux2_text_enc *flux2_text_enc_load(const char *gguf_path);   // GGUF fallback
 *   flux2_text_enc *flux2_text_enc_load_gpu(const char *model_path,
 *                                           const char *tok_gguf_path,
 *                                           int device);
 *   void            flux2_text_enc_free(flux2_text_enc *enc);
 *   float          *flux2_text_enc_encode(flux2_text_enc *enc, const char *text,
 *                                         int *out_n_tokens);
 */
#ifndef FLUX2_KLEIN_TEXT_ENCODER_H
#define FLUX2_KLEIN_TEXT_ENCODER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void *model;        /* transformer_model* (CPU) or cuda_llm_runner* (GPU) */
    void *vocab;        /* bpe_vocab* */
    void *gguf;         /* gguf_context* for mmap lifetime (NULL if safetensors) */
    int   n_embd;       /* output dim per token (n_layers_concat * model_hidden) */
    int   n_embd_inner; /* model internal hidden dim (e.g. 2560 for Qwen3-4B) */
    int   n_vocab;
    int   n_layers;     /* total transformer layers */
    int   use_gpu;
    int   owns_resources;
} flux2_text_enc;

/* Load from a single merged safetensors file (ComfyUI qwen_3_4b.safetensors).
 * tok_gguf_path: any Qwen3 GGUF used only for its tokenizer vocab. */
flux2_text_enc *flux2_text_enc_load_safetensors(const char *st_path,
                                                 const char *tok_gguf_path);

/* Load from a directory of shards (HuggingFace BFL format).
 * dir_path should contain model-00001-of-00002.safetensors etc. */
flux2_text_enc *flux2_text_enc_load_safetensors_dir(const char *dir_path,
                                                     const char *tok_gguf_path);

/* Fallback: load from GGUF (requires convert_hf_to_gguf conversion). */
flux2_text_enc *flux2_text_enc_load(const char *gguf_path);

/* GPU text encoder via CUDA LLM runner (requires cuda_llm_runner.h).
 * model_path may be a GGUF file or safetensors file/directory.
 * tok_gguf_path provides the tokenizer when model_path is safetensors. */
flux2_text_enc *flux2_text_enc_load_gpu(const char *model_path,
                                        const char *tok_gguf_path,
                                        int gpu_device);

void   flux2_text_enc_free(flux2_text_enc *enc);

/* Encode text → per-token hidden states [n_tokens, n_embd] (caller frees). */
float *flux2_text_enc_encode(flux2_text_enc *enc, const char *text,
                             int *out_n_tokens);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef FLUX2_TEXT_ENCODER_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include "transformer.h"
#include "bpe_tokenizer.h"
#include "safetensors.h"

#ifdef CUDA_LLM_RUNNER_H
typedef struct {
    char model_path[512];
    char tok_path[512];
    int gpu_device;
    int ref_count;
    cuda_llm_runner *model;
    bpe_vocab *vocab;
    gguf_context *tok_gguf;
    int n_embd_inner;
    int n_embd;
    int n_vocab;
    int n_layers;
} flux2_enc_gpu_cache;

static flux2_enc_gpu_cache g_flux2_enc_gpu_cache = {0};
#endif

/* ---- BF16 dequant helper ---- */
static float flux2_enc_bf16_to_f32(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float f; memcpy(&f, &bits, 4); return f;
}

/* Load a BF16 or F32 1-D/2-D tensor from one of the given safetensors shards.
 * Returns NULL if not found in any shard. */
static float *flux2_enc_st_load_f32_multi(st_context **shards, int n_shards,
                                           const char *name, int *rows, int *cols) {
    st_context *st = NULL;
    int idx = -1;
    for (int s = 0; s < n_shards && idx < 0; s++) {
        idx = safetensors_find(shards[s], name);
        if (idx >= 0) st = shards[s];
    }
    if (idx < 0) { *rows = *cols = 0; return NULL; }

    const uint64_t *sh = safetensors_shape(st, idx);
    int nd = safetensors_ndims(st, idx);
    *rows = (nd >= 2) ? (int)sh[0] : 1;
    *cols = (nd >= 2) ? (int)sh[1] : (int)sh[0];
    int n = (*rows) * (*cols);

    float *out = (float *)malloc((size_t)n * sizeof(float));
    const char *dtype = safetensors_dtype(st, idx);

    if (strcmp(dtype, "BF16") == 0) {
        const uint16_t *src = (const uint16_t *)safetensors_data(st, idx);
        for (int i = 0; i < n; i++) out[i] = flux2_enc_bf16_to_f32(src[i]);
    } else if (strcmp(dtype, "F32") == 0) {
        memcpy(out, safetensors_data(st, idx), (size_t)n * sizeof(float));
    } else if (strcmp(dtype, "F16") == 0) {
        const uint16_t *src = (const uint16_t *)safetensors_data(st, idx);
        for (int i = 0; i < n; i++) {
            uint32_t sv = src[i];
            uint32_t s = (sv >> 15) & 1;
            uint32_t e = (sv >> 10) & 0x1f;
            uint32_t m = sv & 0x3ff;
            uint32_t b;
            if (e == 0)       b = (s << 31) | ((m ? 0x70 : 0x00) << 23) | (m << 13);
            else if (e == 31) b = (s << 31) | (0xff << 23) | (m << 13);
            else              b = (s << 31) | ((e + 112) << 23) | (m << 13);
            memcpy(&out[i], &b, 4);
        }
    } else {
        fprintf(stderr, "flux2_text_enc: unsupported dtype '%s' for %s\n", dtype, name);
        free(out); *rows = *cols = 0; return NULL;
    }
    return out;
}

/* Single-shard convenience wrapper */
static float *flux2_enc_st_load_f32(st_context *st, const char *name,
                                     int *rows, int *cols) {
    return flux2_enc_st_load_f32_multi(&st, 1, name, rows, cols);
}


static qtensor flux2_enc_make_qt(float *data, int rows, int cols) {
    qtensor t = {0};
    t.data   = data;
    t.type   = GGML_TYPE_F32;
    t.n_rows = rows;
    t.n_cols = (rows == 1) ? cols : cols;
    t.n_dims = (rows == 1) ? 1 : 2;
    t.dims[0] = (uint64_t)(rows == 1 ? cols : cols);
    if (rows > 1) t.dims[1] = (uint64_t)rows;
    return t;
}

/* ---- Internal: build transformer_model from multi-shard safetensors ---- */

static flux2_text_enc *flux2_enc_build_from_shards(st_context **shards, int n_shards,
                                                    gguf_context *tok_gguf,
                                                    bpe_vocab *vocab) {
    /* Auto-detect architecture: count layers across all shards */
    int n_layers = 0;
    for (int s = 0; s < n_shards; s++) {
        for (int i = 0; i < shards[s]->n_tensors; i++) {
            const char *nm = safetensors_name(shards[s], i);
            if (strncmp(nm, "model.layers.", 13) == 0) {
                int l = atoi(nm + 13);
                if (l + 1 > n_layers) n_layers = l + 1;
            }
        }
    }

    /* Detect dims */
    int r, c;
    float *emb_probe = flux2_enc_st_load_f32_multi(shards, n_shards, "model.embed_tokens.weight", &r, &c);
    int n_vocab = r, n_embd = c;

    char wn[256];
    snprintf(wn, sizeof(wn), "model.layers.0.self_attn.q_proj.weight");
    float *q_probe = flux2_enc_st_load_f32_multi(shards, n_shards, wn, &r, &c);
    int q_dim = r;   /* n_heads * head_dim — may differ from n_embd in Qwen3 */
    if (q_probe) free(q_probe);

    snprintf(wn, sizeof(wn), "model.layers.0.self_attn.k_proj.weight");
    float *k_probe = flux2_enc_st_load_f32_multi(shards, n_shards, wn, &r, &c);
    int kv_dim = r;  /* n_kv_heads * head_dim */
    if (k_probe) free(k_probe);

    snprintf(wn, sizeof(wn), "model.layers.0.self_attn.q_norm.weight");
    float *qnorm_probe = flux2_enc_st_load_f32_multi(shards, n_shards, wn, &r, &c);
    int head_dim = (r == 1) ? c : 0;  /* q_norm has shape [head_dim] */
    if (qnorm_probe) free(qnorm_probe);

    snprintf(wn, sizeof(wn), "model.layers.0.mlp.gate_proj.weight");
    float *ff_probe = flux2_enc_st_load_f32_multi(shards, n_shards, wn, &r, &c);
    int n_ff = r;
    if (ff_probe) free(ff_probe);

    /* Fallback to Qwen3-4B defaults if detection fails */
    if (n_embd == 0)  n_embd  = 2560;
    if (head_dim == 0) head_dim = 128;
    if (n_ff == 0)    n_ff    = 9728;
    if (n_vocab == 0) n_vocab = 151936;
    /* Detect n_heads from q_proj shape (Qwen3: q_dim=4096, n_embd=2560, so n_heads=32) */
    int n_heads    = (q_dim > 0 && head_dim > 0) ? q_dim / head_dim : n_embd / head_dim;
    int n_kv_heads = (kv_dim > 0) ? kv_dim / head_dim : 8;

    fprintf(stderr, "flux2_text_enc: %d layers, n_embd=%d n_heads=%d/%d head_dim=%d n_ff=%d\n",
            n_layers, n_embd, n_heads, n_kv_heads, head_dim, n_ff);

    /* Build transformer_model struct. */
    transformer_model *m = (transformer_model *)calloc(1, sizeof(transformer_model));
    m->n_layers    = n_layers;
    m->n_embd      = n_embd;
    m->n_heads     = n_heads;
    m->n_kv_heads  = n_kv_heads;
    m->head_dim    = head_dim;
    m->n_ff        = n_ff;
    m->n_vocab     = n_vocab;
    m->max_seq_len = 2048;
    m->rms_norm_eps  = 1e-6f;
    m->rope_freq_base = 1000000.0f;  /* Qwen3 rope_theta=1000000 */
    m->has_lm_head   = 0;
    m->use_mrope     = 0;

    int kv_d  = n_kv_heads * head_dim;
    int attn_q_dim = n_heads * head_dim;  /* may exceed n_embd in Qwen3 (4096 vs 2560) */
    int xb2_dim = attn_q_dim > n_embd ? attn_q_dim : n_embd;
    m->x        = (float *)calloc(n_embd, sizeof(float));
    m->xb       = (float *)calloc(n_embd, sizeof(float));
    m->xb2      = (float *)calloc(xb2_dim, sizeof(float));
    m->q        = (float *)calloc(attn_q_dim, sizeof(float));
    m->k        = (float *)calloc(kv_d, sizeof(float));
    m->v        = (float *)calloc(kv_d, sizeof(float));
    m->att      = (float *)calloc((size_t)n_heads * m->max_seq_len, sizeof(float));
    m->ffn_buf1 = (float *)calloc(n_ff, sizeof(float));
    m->ffn_buf2 = (float *)calloc(n_ff, sizeof(float));
    m->ffn_buf3 = (float *)calloc(n_ff, sizeof(float));
    int scratch_dim = n_ff > n_embd ? n_ff : n_embd;
    m->matvec_tmp = (float *)calloc(scratch_dim, sizeof(float));

    {
        int n_cpu = 4;
#ifdef _SC_NPROCESSORS_ONLN
        int hw = (int)sysconf(_SC_NPROCESSORS_ONLN);
        if (hw > 0) n_cpu = hw > 8 ? hw / 2 : hw;
#endif
        m->n_threads  = n_cpu;
        m->thread_tmp = (float **)calloc(n_cpu, sizeof(float *));
        for (int t = 0; t < n_cpu; t++)
            m->thread_tmp[t] = (float *)calloc(scratch_dim, sizeof(float));
        transformer_set_threads(m, n_cpu);
        fprintf(stderr, "flux2_text_enc: using %d CPU threads\n", n_cpu);
    }

    m->key_cache   = (float **)calloc(n_layers, sizeof(float *));
    m->value_cache = (float **)calloc(n_layers, sizeof(float *));
    for (int l = 0; l < n_layers; l++) {
        m->key_cache[l]   = (float *)calloc((size_t)m->max_seq_len * kv_d, sizeof(float));
        m->value_cache[l] = (float *)calloc((size_t)m->max_seq_len * kv_d, sizeof(float));
    }

    int rope_len = head_dim / 2;
    m->rope_inv_freq     = (float *)malloc(rope_len * sizeof(float));
    m->rope_inv_freq_len = rope_len;
    for (int j = 0; j < rope_len; j++)
        m->rope_inv_freq[j] = 1.0f / powf(m->rope_freq_base, (float)(2*j) / (float)head_dim);

    m->token_embd  = flux2_enc_make_qt(emb_probe, n_vocab, n_embd);

    float *norm_data = flux2_enc_st_load_f32_multi(shards, n_shards, "model.norm.weight", &r, &c);
    m->output_norm = flux2_enc_make_qt(norm_data, 1, n_embd);

    m->layers = (transformer_layer *)calloc(n_layers, sizeof(transformer_layer));
    fprintf(stderr, "flux2_text_enc: loading %d layers...\n", n_layers);

    for (int l = 0; l < n_layers; l++) {
        transformer_layer *ly = &m->layers[l];

        #define MLOAD(field, suffix) do { \
            snprintf(wn, sizeof(wn), "model.layers.%d." suffix, l); \
            float *_d = flux2_enc_st_load_f32_multi(shards, n_shards, wn, &r, &c); \
            if (_d) ly->field = flux2_enc_make_qt(_d, r, c); \
        } while(0)

        MLOAD(attn_norm,   "input_layernorm.weight");
        MLOAD(attn_q,      "self_attn.q_proj.weight");
        MLOAD(attn_k,      "self_attn.k_proj.weight");
        MLOAD(attn_v,      "self_attn.v_proj.weight");
        MLOAD(attn_output, "self_attn.o_proj.weight");
        MLOAD(attn_q_norm, "self_attn.q_norm.weight");
        MLOAD(attn_k_norm, "self_attn.k_norm.weight");
        MLOAD(ffn_norm,    "post_attention_layernorm.weight");
        MLOAD(ffn_gate,    "mlp.gate_proj.weight");
        MLOAD(ffn_up,      "mlp.up_proj.weight");
        MLOAD(ffn_down,    "mlp.down_proj.weight");

        #undef MLOAD

        if ((l + 1) % 8 == 0 || l == n_layers - 1)
            fprintf(stderr, "\r  layer %d/%d", l + 1, n_layers);
    }
    fprintf(stderr, "\n");

    flux2_text_enc *enc = (flux2_text_enc *)calloc(1, sizeof(flux2_text_enc));
    enc->model        = m;
    enc->vocab        = vocab;
    enc->gguf         = tok_gguf;
    enc->n_embd_inner = n_embd;
    enc->n_embd       = 3 * n_embd;  /* last 3 layers concatenated: 3×2560=7680 */
    enc->n_vocab      = n_vocab;
    enc->n_layers     = n_layers;
    enc->use_gpu      = 0;
    enc->owns_resources = 1;
    return enc;
}

/* ---- Safetensors loader (single file) ---- */

flux2_text_enc *flux2_text_enc_load_safetensors(const char *st_path,
                                                 const char *tok_gguf_path) {
    /* Auto-detect: if st_path is a directory, delegate to sharded loader */
    struct stat sb;
    if (stat(st_path, &sb) == 0 && S_ISDIR(sb.st_mode))
        return flux2_text_enc_load_safetensors_dir(st_path, tok_gguf_path);

    fprintf(stderr, "flux2_text_enc: loading %s\n", st_path);

    st_context *st = safetensors_open(st_path);
    if (!st) { fprintf(stderr, "flux2_text_enc: failed to open %s\n", st_path); return NULL; }

    gguf_context *tok_gguf = gguf_open(tok_gguf_path, 1);
    if (!tok_gguf) {
        fprintf(stderr, "flux2_text_enc: failed to open tokenizer GGUF %s\n", tok_gguf_path);
        safetensors_close(st); return NULL;
    }
    bpe_vocab *vocab = bpe_vocab_load(tok_gguf);
    if (!vocab) {
        fprintf(stderr, "flux2_text_enc: failed to load tokenizer from %s\n", tok_gguf_path);
        gguf_close(tok_gguf); safetensors_close(st); return NULL;
    }

    flux2_text_enc *enc = flux2_enc_build_from_shards(&st, 1, tok_gguf, vocab);
    safetensors_close(st);
    return enc;
}

/* ---- Safetensors loader (sharded directory: contains model-*-of-*.safetensors) ---- */

flux2_text_enc *flux2_text_enc_load_safetensors_dir(const char *dir_path,
                                                     const char *tok_gguf_path) {
    /* Enumerate shards: try model-00001-of-NNNNN.safetensors pattern */
    char path[512];
    struct stat sb;
    st_context *shards[16];
    int n_shards = 0;

    for (int i = 1; i <= 16 && n_shards < 16; i++) {
        for (int total = i; total <= 16; total++) {
            snprintf(path, sizeof(path), "%s/model-%05d-of-%05d.safetensors",
                     dir_path, i, total);
            if (stat(path, &sb) != 0) continue;
            st_context *s = safetensors_open(path);
            if (s) {
                shards[n_shards++] = s;
                break;
            }
        }
        if (n_shards == 0 && i == 1) break;  /* no shard 1 found */
        if (n_shards < i) break;             /* gap in sequence */
    }

    if (n_shards == 0) {
        fprintf(stderr, "flux2_text_enc: no safetensors shards found in %s\n", dir_path);
        return NULL;
    }
    fprintf(stderr, "flux2_text_enc: opened %d shard(s) from %s\n", n_shards, dir_path);

    gguf_context *tok_gguf = gguf_open(tok_gguf_path, 1);
    if (!tok_gguf) {
        fprintf(stderr, "flux2_text_enc: failed to open tokenizer GGUF %s\n", tok_gguf_path);
        for (int s = 0; s < n_shards; s++) safetensors_close(shards[s]);
        return NULL;
    }
    bpe_vocab *vocab = bpe_vocab_load(tok_gguf);
    if (!vocab) {
        fprintf(stderr, "flux2_text_enc: failed to load tokenizer from %s\n", tok_gguf_path);
        gguf_close(tok_gguf);
        for (int s = 0; s < n_shards; s++) safetensors_close(shards[s]);
        return NULL;
    }

    flux2_text_enc *enc = flux2_enc_build_from_shards(shards, n_shards, tok_gguf, vocab);
    for (int s = 0; s < n_shards; s++) safetensors_close(shards[s]);
    return enc;
}

/* ---- GGUF loader (fallback if GGUF is available) ---- */

flux2_text_enc *flux2_text_enc_load(const char *gguf_path) {
    fprintf(stderr, "flux2_text_enc: loading GGUF %s\n", gguf_path);

    gguf_context *gguf = gguf_open(gguf_path, 1);
    if (!gguf) { fprintf(stderr, "flux2_text_enc: failed to open GGUF\n"); return NULL; }

    bpe_vocab *vocab = bpe_vocab_load(gguf);
    if (!vocab) {
        fprintf(stderr, "flux2_text_enc: failed to load tokenizer\n");
        gguf_close(gguf); return NULL;
    }

    transformer_model *model = transformer_load(gguf, 2048);
    if (!model) {
        fprintf(stderr, "flux2_text_enc: failed to load model\n");
        bpe_vocab_free(vocab); gguf_close(gguf); return NULL;
    }

    {
        int n_cpu = 4;
#ifdef _SC_NPROCESSORS_ONLN
        int hw = (int)sysconf(_SC_NPROCESSORS_ONLN);
        if (hw > 0) n_cpu = hw > 8 ? hw / 2 : hw;
#endif
        transformer_set_threads(model, n_cpu);
    }

    flux2_text_enc *enc = (flux2_text_enc *)calloc(1, sizeof(flux2_text_enc));
    enc->model        = model;
    enc->vocab        = vocab;
    enc->gguf         = gguf;
    enc->n_embd_inner = model->n_embd;
    enc->n_embd       = 3 * model->n_embd;
    enc->n_vocab      = model->n_vocab;
    enc->n_layers     = model->n_layers;
    enc->use_gpu      = 0;
    enc->owns_resources = 1;
    fprintf(stderr, "flux2_text_enc: n_embd_inner=%d output_dim=%d n_vocab=%d\n",
            enc->n_embd_inner, enc->n_embd, enc->n_vocab);
    return enc;
}

/* ---- GPU loader ---- */

flux2_text_enc *flux2_text_enc_load_gpu(const char *model_path,
                                        const char *tok_gguf_path,
                                        int gpu_device) {
#ifdef CUDA_LLM_RUNNER_H
    if (g_flux2_enc_gpu_cache.model &&
        g_flux2_enc_gpu_cache.gpu_device == gpu_device &&
        strcmp(g_flux2_enc_gpu_cache.model_path, model_path) == 0 &&
        strcmp(g_flux2_enc_gpu_cache.tok_path, tok_gguf_path ? tok_gguf_path : "") == 0) {
        flux2_text_enc *enc = (flux2_text_enc *)calloc(1, sizeof(flux2_text_enc));
        if (!enc) return NULL;
        g_flux2_enc_gpu_cache.ref_count++;
        enc->model = g_flux2_enc_gpu_cache.model;
        enc->vocab = g_flux2_enc_gpu_cache.vocab;
        enc->gguf = g_flux2_enc_gpu_cache.tok_gguf;
        enc->n_embd_inner = g_flux2_enc_gpu_cache.n_embd_inner;
        enc->n_embd = g_flux2_enc_gpu_cache.n_embd;
        enc->n_vocab = g_flux2_enc_gpu_cache.n_vocab;
        enc->n_layers = g_flux2_enc_gpu_cache.n_layers;
        enc->use_gpu = 1;
        enc->owns_resources = 0;
        fprintf(stderr, "flux2_text_enc: reusing cached GPU encoder\n");
        return enc;
    }

    fprintf(stderr, "flux2_text_enc: GPU %s (device %d)\n", model_path, gpu_device);

    int is_gguf = 0;
    size_t model_len = strlen(model_path);
    if (model_len >= 5 && strcmp(model_path + model_len - 5, ".gguf") == 0) is_gguf = 1;

    gguf_context *tok_gguf = NULL;
    bpe_vocab *vocab = NULL;
    if (tok_gguf_path) tok_gguf = gguf_open(tok_gguf_path, 1);
    if (!tok_gguf && is_gguf) tok_gguf = gguf_open(model_path, 1);
    if (!tok_gguf) { fprintf(stderr, "flux2_text_enc: failed to open tokenizer GGUF\n"); return NULL; }

    vocab = bpe_vocab_load(tok_gguf);
    if (!vocab) { gguf_close(tok_gguf); return NULL; }

    cuda_llm_runner *gpu_model = cuda_llm_init(gpu_device, 1);
    if (!gpu_model) { bpe_vocab_free(vocab); gguf_close(tok_gguf); return NULL; }

    if (is_gguf) {
        gguf_context *model_gguf = gguf_open(model_path, 1);
        if (!model_gguf) {
            cuda_llm_free(gpu_model); bpe_vocab_free(vocab); gguf_close(tok_gguf); return NULL;
        }
        if (cuda_llm_load_weights(gpu_model, model_gguf, 2048) != 0) {
            gguf_close(model_gguf);
            cuda_llm_free(gpu_model); bpe_vocab_free(vocab); gguf_close(tok_gguf); return NULL;
        }
        gguf_close(model_gguf);
    } else {
        if (cuda_llm_load_weights_qwen3_safetensors(gpu_model, model_path, 2048) != 0) {
            cuda_llm_free(gpu_model); bpe_vocab_free(vocab); gguf_close(tok_gguf); return NULL;
        }
    }

    {
        const int hs_layers[3] = {8, 17, 26};
        if (cuda_llm_set_hidden_snapshot_layers(gpu_model, hs_layers, 3) != 0) {
            cuda_llm_free(gpu_model); bpe_vocab_free(vocab); gguf_close(tok_gguf); return NULL;
        }
    }

    flux2_text_enc *enc = (flux2_text_enc *)calloc(1, sizeof(flux2_text_enc));
    enc->model        = gpu_model;
    enc->vocab        = vocab;
    enc->gguf         = tok_gguf;
    enc->n_embd_inner = cuda_llm_n_embd(gpu_model);
    enc->n_embd       = 3 * cuda_llm_n_embd(gpu_model);
    enc->n_vocab      = cuda_llm_n_vocab(gpu_model);
    enc->n_layers     = cuda_llm_n_layers(gpu_model);
    enc->use_gpu      = 1;
    enc->owns_resources = 0;

    strncpy(g_flux2_enc_gpu_cache.model_path, model_path, sizeof(g_flux2_enc_gpu_cache.model_path) - 1);
    strncpy(g_flux2_enc_gpu_cache.tok_path, tok_gguf_path ? tok_gguf_path : "",
            sizeof(g_flux2_enc_gpu_cache.tok_path) - 1);
    g_flux2_enc_gpu_cache.gpu_device = gpu_device;
    g_flux2_enc_gpu_cache.ref_count = 1;
    g_flux2_enc_gpu_cache.model = gpu_model;
    g_flux2_enc_gpu_cache.vocab = vocab;
    g_flux2_enc_gpu_cache.tok_gguf = tok_gguf;
    g_flux2_enc_gpu_cache.n_embd_inner = enc->n_embd_inner;
    g_flux2_enc_gpu_cache.n_embd = enc->n_embd;
    g_flux2_enc_gpu_cache.n_vocab = enc->n_vocab;
    g_flux2_enc_gpu_cache.n_layers = enc->n_layers;
    return enc;
#else
    (void)model_path; (void)tok_gguf_path; (void)gpu_device;
    fprintf(stderr, "flux2_text_enc: GPU path requires CUDA_LLM_RUNNER_H\n");
    return NULL;
#endif
}

/* ---- Free ---- */

void flux2_text_enc_free(flux2_text_enc *enc) {
    if (!enc) return;
    if (enc->use_gpu) {
#ifdef CUDA_LLM_RUNNER_H
        if (g_flux2_enc_gpu_cache.model == (cuda_llm_runner *)enc->model &&
            g_flux2_enc_gpu_cache.ref_count > 0) {
            g_flux2_enc_gpu_cache.ref_count--;
            if (g_flux2_enc_gpu_cache.ref_count == 0) {
                cuda_llm_free(g_flux2_enc_gpu_cache.model);
                bpe_vocab_free(g_flux2_enc_gpu_cache.vocab);
                if (g_flux2_enc_gpu_cache.tok_gguf) gguf_close(g_flux2_enc_gpu_cache.tok_gguf);
                memset(&g_flux2_enc_gpu_cache, 0, sizeof(g_flux2_enc_gpu_cache));
            }
        } else if (enc->owns_resources) {
            cuda_llm_free((cuda_llm_runner *)enc->model);
            bpe_vocab_free((bpe_vocab *)enc->vocab);
            if (enc->gguf) gguf_close((gguf_context *)enc->gguf);
        }
#endif
    } else {
        transformer_free((transformer_model *)enc->model);
        bpe_vocab_free((bpe_vocab *)enc->vocab);
        if (enc->gguf) gguf_close((gguf_context *)enc->gguf);
    }
    free(enc);
}

/* ---- Encode ---- */

float *flux2_text_enc_encode(flux2_text_enc *enc, const char *text,
                             int *out_n_tokens) {
    if (!enc || !text) return NULL;

    bpe_vocab *vocab = (bpe_vocab *)enc->vocab;

    /* Apply Qwen3 chat template matching diffusers pipeline_flux2_klein.py:
     *   messages = [{"role":"user","content":prompt}]
     *   text = tokenizer.apply_chat_template(messages, tokenize=False,
     *              add_generation_prompt=True, enable_thinking=False)
     * With enable_thinking=False the template is:
     *   <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
     */
    const char *prefix = "<|im_start|>user\n";
    const char *suffix = "<|im_end|>\n<|im_start|>assistant\n";
    size_t chat_len = strlen(prefix) + strlen(text) + strlen(suffix);
    char *chat_text = (char *)malloc(chat_len + 1);
    if (!chat_text) return NULL;
    strcpy(chat_text, prefix);
    strcat(chat_text, text);
    strcat(chat_text, suffix);

    /* Tokenize; diffusers pads/truncates to 512 tokens */
    const int MAX_SEQ = 512;
    int32_t toks[512];
    int n_tok = bpe_tokenize(vocab, chat_text, (int)strlen(chat_text), toks, MAX_SEQ);
    free(chat_text);
    if (n_tok <= 0) {
        fprintf(stderr, "flux2_text_enc: tokenization failed\n");
        return NULL;
    }
    fprintf(stderr, "flux2_text_enc: '%s' → %d tokens (chat template)\n", text, n_tok);

    /* Do NOT pad: feeding only real tokens to the DiT is equivalent to using
     * attention_mask (as diffusers does) since padding tokens carry no content.
     * Padding to MAX_SEQ=512 makes the DiT forward ~17× slower for short prompts. */

    /* Output dim = 3 * n_embd_inner (last 3 layers concatenated) */
    int n_inner = enc->n_embd_inner;
    int n_out   = enc->n_embd;  /* = 3 * n_inner */
    float *hidden = (float *)malloc((size_t)n_tok * n_out * sizeof(float));
    if (!hidden) return NULL;

    if (enc->use_gpu) {
#ifdef CUDA_LLM_RUNNER_H
        /* GPU path: capture the same intermediate hidden states as the CPU
         * reference after layers 8, 17, and 26. */
        cuda_llm_runner *gpu = (cuda_llm_runner *)enc->model;
        cuda_llm_reset_state(gpu);
        for (int i = 0; i < n_tok; i++) {
            float *dst = hidden + (size_t)i * n_out;
            if (!cuda_llm_forward(gpu, toks[i], i)) {
                fprintf(stderr, "flux2_text_enc: cuda_llm_forward failed at token %d/%d\n", i + 1, n_tok);
                free(hidden);
                return NULL;
            }
            if (cuda_llm_read_hidden_snapshots(gpu, dst, 3, n_inner) != 0) {
                fprintf(stderr, "flux2_text_enc: cuda_llm_read_hidden_snapshots failed at token %d/%d\n",
                        i + 1, n_tok);
                free(hidden);
                return NULL;
            }
        }
#endif
    } else {
        transformer_model *mdl = (transformer_model *)enc->model;
        int nl = mdl->n_layers;
        /* Diffusers Flux2KleinPipeline uses hidden_states at indices (9, 18, 27).
         * HuggingFace convention: hidden_states[k] = output of transformer layer k-1.
         * So we extract after layers 8, 17, 26 (0-indexed) by running partial segments.
         * After each segment, model->x holds the output of that layer range. */
        for (int i = 0; i < n_tok; i++) {
            float *dst = hidden + (size_t)i * n_out;
            transformer_embed_token(mdl, toks[i]);
            transformer_forward_partial(mdl, i, 0, 9);
            memcpy(dst, mdl->x, n_inner * sizeof(float));             /* hs[9]  */
            transformer_forward_partial(mdl, i, 9, 18);
            memcpy(dst + n_inner, mdl->x, n_inner * sizeof(float));   /* hs[18] */
            transformer_forward_partial(mdl, i, 18, 27);
            memcpy(dst + 2 * n_inner, mdl->x, n_inner * sizeof(float)); /* hs[27] */
            transformer_forward_partial(mdl, i, 27, nl); /* run remaining layers for KV cache */
        }
    }

    if (out_n_tokens) *out_n_tokens = n_tok;
    return hidden;
}

#endif /* FLUX2_TEXT_ENCODER_IMPLEMENTATION */
#endif /* FLUX2_KLEIN_TEXT_ENCODER_H */
