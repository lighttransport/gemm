/*
 * qwen_image_text_encoder.h - Text encoder for Qwen-Image pipeline
 *
 * Wraps transformer.h (Qwen2.5-VL LLM) to extract per-token hidden states
 * as conditioning for the diffusion model.
 *
 * Usage:
 *   #define QIMG_TEXT_ENCODER_IMPLEMENTATION
 *   #include "qwen_image_text_encoder.h"
 *
 * Dependencies: transformer.h, bpe_tokenizer.h, gguf_loader.h
 *
 * API:
 *   qimg_text_enc *qimg_text_enc_load(const char *gguf_path);
 *   void           qimg_text_enc_free(qimg_text_enc *enc);
 *   float         *qimg_text_enc_encode(qimg_text_enc *enc, const char *text,
 *                                       int *out_n_tokens);
 */
#ifndef QIMG_TEXT_ENCODER_H
#define QIMG_TEXT_ENCODER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void *model;     /* transformer_model* */
    void *vocab;     /* bpe_vocab* */
    void *gguf;      /* gguf_context* (kept for mmap lifetime) */
    int   n_embd;    /* 3584 for Qwen2.5-VL-7B */
    int   n_vocab;
} qimg_text_enc;

qimg_text_enc *qimg_text_enc_load(const char *gguf_path);
/* Load GGUF model + inject Q/K/V biases from FP8 safetensors (for Qwen2.5-VL) */
qimg_text_enc *qimg_text_enc_load_gguf_with_biases(const char *gguf_path,
                                                     const char *bias_st_path);
qimg_text_enc *qimg_text_enc_load_safetensors(const char *st_path,
                                               const char *tokenizer_gguf_path);
void           qimg_text_enc_free(qimg_text_enc *enc);

/* Encode text prompt to hidden states.
 * Returns [n_tokens, n_embd] float array (caller must free).
 * Sets *out_n_tokens to the number of tokens. */
float *qimg_text_enc_encode(qimg_text_enc *enc, const char *text,
                            int *out_n_tokens);

#ifdef __cplusplus
}
#endif

/* ======================================================================== */
#ifdef QIMG_TEXT_ENCODER_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "transformer.h"
#include "bpe_tokenizer.h"

qimg_text_enc *qimg_text_enc_load(const char *gguf_path) {
    fprintf(stderr, "qimg_text_enc: loading %s\n", gguf_path);
    gguf_context *gguf = gguf_open(gguf_path, 1);
    if (!gguf) {
        fprintf(stderr, "qimg_text_enc: failed to open GGUF\n");
        return NULL;
    }

    /* Load tokenizer from GGUF metadata */
    bpe_vocab *vocab = bpe_vocab_load(gguf);
    if (!vocab) {
        fprintf(stderr, "qimg_text_enc: failed to load tokenizer\n");
        gguf_close(gguf);
        return NULL;
    }

    /* Load transformer model */
    transformer_model *model = transformer_load(gguf, 2048);
    if (!model) {
        fprintf(stderr, "qimg_text_enc: failed to load model\n");
        bpe_vocab_free(vocab);
        gguf_close(gguf);
        return NULL;
    }

    { /* Use half of available cores for text encoder (matvec is memory-bound) */
        int n_cpu = 4;
#ifdef _SC_NPROCESSORS_ONLN
        int hw = (int)sysconf(_SC_NPROCESSORS_ONLN);
        if (hw > 0) n_cpu = hw > 8 ? hw / 2 : hw;
#endif
        transformer_set_threads(model, n_cpu);
    }

    qimg_text_enc *enc = (qimg_text_enc *)calloc(1, sizeof(qimg_text_enc));
    enc->model = model;
    enc->vocab = vocab;
    enc->gguf = gguf;
    enc->n_embd = model->n_embd;
    enc->n_vocab = model->n_vocab;

    fprintf(stderr, "qimg_text_enc: loaded (n_embd=%d, n_vocab=%d, n_layers=%d)\n",
            model->n_embd, model->n_vocab, model->n_layers);
    return enc;
}

/* ---- Scaled FP8 safetensors loader ---- */

/* Dequant scaled FP8: actual = fp8_to_f32(byte) * scale_weight */
static float *st_dequant_scaled_fp8(st_context *st, const char *weight_name,
                                     const char *scale_name,
                                     int *out_rows, int *out_cols) {
    int widx = safetensors_find(st, weight_name);
    int sidx = safetensors_find(st, scale_name);
    if (widx < 0) { *out_rows = *out_cols = 0; return NULL; }

    const uint64_t *shape = safetensors_shape(st, widx);
    int ndims = safetensors_ndims(st, widx);
    *out_rows = (ndims >= 2) ? (int)shape[0] : 1;
    *out_cols = (ndims >= 2) ? (int)shape[1] : (int)shape[0];
    int n = (*out_rows) * (*out_cols);

    const uint8_t *raw = (const uint8_t *)safetensors_data(st, widx);
    float scale = 1.0f;
    if (sidx >= 0) {
        /* scale_weight is a scalar F32 */
        const float *sp = (const float *)safetensors_data(st, sidx);
        scale = *sp;
    }

    /* Use LUT for FP8→F32, then multiply by scale */
    static float fp8_lut[256];
    static int fp8_lut_done = 0;
    if (!fp8_lut_done) {
        for (int i = 0; i < 256; i++) {
            uint8_t b = (uint8_t)i;
            uint32_t sign = (b >> 7) & 1, exp = (b >> 3) & 0xF, mant = b & 0x7;
            if (exp == 0 && mant == 0) fp8_lut[i] = 0.0f;
            else if (exp == 0) fp8_lut[i] = (sign ? -1.0f : 1.0f) * ((float)mant / 8.0f) * ldexpf(1.0f, -6);
            else if (exp == 15 && mant == 7) fp8_lut[i] = 0.0f;
            else fp8_lut[i] = (sign ? -1.0f : 1.0f) * (1.0f + (float)mant / 8.0f) * ldexpf(1.0f, (int)exp - 7);
        }
        fp8_lut_done = 1;
    }

    float *f32 = (float *)malloc((size_t)n * sizeof(float));
    const char *dtype = safetensors_dtype(st, widx);
    if (strcmp(dtype, "F8_E4M3") == 0 || strcmp(dtype, "F8_E4M3FN") == 0) {
        for (int i = 0; i < n; i++)
            f32[i] = fp8_lut[raw[i]] * scale;
    } else if (strcmp(dtype, "BF16") == 0) {
        const uint16_t *bf = (const uint16_t *)safetensors_data(st, widx);
        for (int i = 0; i < n; i++) {
            uint32_t bits = (uint32_t)bf[i] << 16;
            memcpy(&f32[i], &bits, 4);
        }
    } else if (strcmp(dtype, "F32") == 0) {
        memcpy(f32, safetensors_data(st, widx), (size_t)n * sizeof(float));
    }
    return f32;
}

/* Build a qtensor from F32 data (PyTorch shape [rows, cols]) */
static qtensor make_qt_f32(float *data, int rows, int cols) {
    qtensor t = {0};
    t.data = data;
    t.type = GGML_TYPE_F32;
    t.n_rows = rows;
    t.n_cols = cols;
    t.n_dims = (rows == 1) ? 1 : 2;
    t.dims[0] = (uint64_t)cols;
    if (rows > 1) t.dims[1] = (uint64_t)rows;
    return t;
}

/* Load GGUF model + inject Q/K/V biases from FP8 safetensors.
 * GGUF has quantized weights (fast CPU inference) but no biases.
 * The safetensors file has BF16 biases (~200KB total, loads in <0.1s). */
qimg_text_enc *qimg_text_enc_load_gguf_with_biases(const char *gguf_path,
                                                     const char *bias_st_path) {
    qimg_text_enc *enc = qimg_text_enc_load(gguf_path);
    if (!enc || !bias_st_path) return enc;

    st_context *st = safetensors_open(bias_st_path);
    if (!st) {
        fprintf(stderr, "qimg_text_enc: no bias file, continuing without biases\n");
        return enc;
    }

    transformer_model *m = (transformer_model *)enc->model;
    int n_loaded = 0;
    for (int l = 0; l < m->n_layers; l++) {
        char wn[256]; int r, c;
        snprintf(wn, sizeof(wn), "model.layers.%d.self_attn.q_proj.bias", l);
        float *qb = st_dequant_scaled_fp8(st, wn, "", &r, &c);
        if (qb) { m->layers[l].attn_q_bias = make_qt_f32(qb, 1, r*c); n_loaded++; }
        snprintf(wn, sizeof(wn), "model.layers.%d.self_attn.k_proj.bias", l);
        float *kb = st_dequant_scaled_fp8(st, wn, "", &r, &c);
        if (kb) { m->layers[l].attn_k_bias = make_qt_f32(kb, 1, r*c); n_loaded++; }
        snprintf(wn, sizeof(wn), "model.layers.%d.self_attn.v_proj.bias", l);
        float *vb = st_dequant_scaled_fp8(st, wn, "", &r, &c);
        if (vb) { m->layers[l].attn_v_bias = make_qt_f32(vb, 1, r*c); n_loaded++; }
    }
    safetensors_close(st);
    fprintf(stderr, "qimg_text_enc: injected %d biases from %s\n", n_loaded, bias_st_path);
    return enc;
}

qimg_text_enc *qimg_text_enc_load_safetensors(const char *st_path,
                                               const char *tokenizer_gguf_path) {
    fprintf(stderr, "qimg_text_enc: loading FP8 safetensors %s\n", st_path);
    st_context *st = safetensors_open(st_path);
    if (!st) return NULL;

    /* Load tokenizer from GGUF (need vocab + merges) */
    gguf_context *tok_gguf = gguf_open(tokenizer_gguf_path, 1);
    if (!tok_gguf) { safetensors_close(st); return NULL; }
    bpe_vocab *vocab = bpe_vocab_load(tok_gguf);
    if (!vocab) { gguf_close(tok_gguf); safetensors_close(st); return NULL; }

    /* Count layers */
    int n_layers = 0;
    for (int i = 0; i < st->n_tensors; i++) {
        const char *nm = safetensors_name(st, i);
        if (strstr(nm, "model.layers.")) {
            int l = atoi(strstr(nm, "layers.") + 7);
            if (l + 1 > n_layers) n_layers = l + 1;
        }
    }

    int n_embd = 3584, n_heads = 28, n_kv_heads = 4, head_dim = 128;
    int n_ff = 18944, n_vocab = 152064;
    fprintf(stderr, "qimg_text_enc: %d layers, n_embd=%d\n", n_layers, n_embd);

    /* Allocate transformer model */
    transformer_model *m = (transformer_model *)calloc(1, sizeof(transformer_model));
    m->n_embd = n_embd;
    m->n_heads = n_heads;
    m->n_kv_heads = n_kv_heads;
    m->head_dim = head_dim;
    m->n_ff = n_ff;
    m->n_vocab = n_vocab;
    m->n_layers = n_layers;
    m->max_seq_len = 2048;
    m->rms_norm_eps = 1e-6f;
    m->rope_freq_base = 1000000.0f;
    m->has_lm_head = 0;  /* no lm_head needed for hidden state extraction */

    /* Allocate scratch buffers */
    int kv_dim = n_kv_heads * head_dim;
    m->x = (float *)calloc(n_embd, sizeof(float));
    m->xb = (float *)calloc(n_embd, sizeof(float));
    m->xb2 = (float *)calloc(n_embd, sizeof(float));
    m->q = (float *)calloc(n_embd, sizeof(float));
    m->k = (float *)calloc(kv_dim, sizeof(float));
    m->v = (float *)calloc(kv_dim, sizeof(float));
    m->att = (float *)calloc((size_t)n_heads * m->max_seq_len, sizeof(float));
    m->ffn_buf1 = (float *)calloc(n_ff, sizeof(float));
    m->ffn_buf2 = (float *)calloc(n_ff, sizeof(float));
    m->ffn_buf3 = (float *)calloc(n_ff, sizeof(float));
    m->matvec_tmp = (float *)calloc(n_ff > n_embd ? n_ff : n_embd, sizeof(float));
    m->n_threads = 1;
    /* thread_tmp: needed even for single-thread (matvec scratch per thread) */
    m->thread_tmp = (float **)calloc(1, sizeof(float *));
    m->thread_tmp[0] = (float *)calloc(n_ff > n_embd ? n_ff : n_embd, sizeof(float));

    /* KV cache */
    m->key_cache = (float **)calloc(n_layers, sizeof(float *));
    m->value_cache = (float **)calloc(n_layers, sizeof(float *));
    for (int l = 0; l < n_layers; l++) {
        m->key_cache[l] = (float *)calloc((size_t)m->max_seq_len * kv_dim, sizeof(float));
        m->value_cache[l] = (float *)calloc((size_t)m->max_seq_len * kv_dim, sizeof(float));
    }

    /* Load global weights */
    int er, ec;
    float *emb_data = st_dequant_scaled_fp8(st, "model.embed_tokens.weight", "", &er, &ec);
    m->token_embd = make_qt_f32(emb_data, er, ec);

    float *norm_data = st_dequant_scaled_fp8(st, "model.norm.weight", "", &er, &ec);
    m->output_norm = make_qt_f32(norm_data, 1, n_embd);

    /* Allocate layers */
    m->layers = (transformer_layer *)calloc(n_layers, sizeof(transformer_layer));

    fprintf(stderr, "qimg_text_enc: loading %d layers...\n", n_layers);
    for (int l = 0; l < n_layers; l++) {
        transformer_layer *ly = &m->layers[l];
        char wn[256], sn[256];
        int r, c;

        #define LOAD_W(field, w_suffix, s_suffix) do { \
            snprintf(wn, sizeof(wn), "model.layers.%d." w_suffix, l); \
            snprintf(sn, sizeof(sn), "model.layers.%d." s_suffix, l); \
            float *d = st_dequant_scaled_fp8(st, wn, sn, &r, &c); \
            ly->field = make_qt_f32(d, r, c); \
        } while(0)

        #define LOAD_NORM(field, suffix) do { \
            snprintf(wn, sizeof(wn), "model.layers.%d." suffix, l); \
            float *d = st_dequant_scaled_fp8(st, wn, "", &r, &c); \
            ly->field = make_qt_f32(d, 1, n_embd); \
        } while(0)

        LOAD_NORM(attn_norm, "input_layernorm.weight");
        LOAD_W(attn_q, "self_attn.q_proj.weight", "self_attn.q_proj.scale_weight");
        LOAD_W(attn_k, "self_attn.k_proj.weight", "self_attn.k_proj.scale_weight");
        LOAD_W(attn_v, "self_attn.v_proj.weight", "self_attn.v_proj.scale_weight");
        LOAD_W(attn_output, "self_attn.o_proj.weight", "self_attn.o_proj.scale_weight");

        /* Q/K/V biases (BF16 in safetensors) — critical for correct attention */
        #define LOAD_BIAS(field, suffix) do { \
            snprintf(wn, sizeof(wn), "model.layers.%d." suffix, l); \
            float *d = st_dequant_scaled_fp8(st, wn, "", &r, &c); \
            if (d) ly->field = make_qt_f32(d, 1, r * c); \
        } while(0)
        LOAD_BIAS(attn_q_bias, "self_attn.q_proj.bias");
        LOAD_BIAS(attn_k_bias, "self_attn.k_proj.bias");
        LOAD_BIAS(attn_v_bias, "self_attn.v_proj.bias");
        #undef LOAD_BIAS

        LOAD_NORM(ffn_norm, "post_attention_layernorm.weight");
        LOAD_W(ffn_gate, "mlp.gate_proj.weight", "mlp.gate_proj.scale_weight");
        LOAD_W(ffn_up, "mlp.up_proj.weight", "mlp.up_proj.scale_weight");
        LOAD_W(ffn_down, "mlp.down_proj.weight", "mlp.down_proj.scale_weight");

        #undef LOAD_W
        #undef LOAD_NORM
        #undef LOAD_BIAS

        if (l % 7 == 0)
            fprintf(stderr, "\r  layer %d/%d (%.0f MB)",
                    l+1, n_layers, (float)(l+1)*26*n_embd*n_embd*4/1024/1024);
    }
    fprintf(stderr, "\n");

    /* RoPE */
    m->use_mrope = 0;  /* standard RoPE for text-only */
    int rope_len = head_dim / 2;
    m->rope_inv_freq = (float *)malloc(rope_len * sizeof(float));
    m->rope_inv_freq_len = rope_len;
    for (int i = 0; i < rope_len; i++)
        m->rope_inv_freq[i] = 1.0f / powf(m->rope_freq_base, (float)(2*i) / (float)head_dim);

    safetensors_close(st);

    { int n_cpu = 4;
#ifdef _SC_NPROCESSORS_ONLN
        int hw = (int)sysconf(_SC_NPROCESSORS_ONLN);
        if (hw > 0) n_cpu = hw > 8 ? hw / 2 : hw;
#endif
        transformer_set_threads(m, n_cpu);
    }

    qimg_text_enc *enc = (qimg_text_enc *)calloc(1, sizeof(qimg_text_enc));
    enc->model = m;
    enc->vocab = vocab;
    enc->gguf = tok_gguf;
    enc->n_embd = n_embd;
    enc->n_vocab = n_vocab;

    fprintf(stderr, "qimg_text_enc: loaded FP8 scaled encoder (%d layers, %d vocab)\n",
            n_layers, n_vocab);
    return enc;
}

void qimg_text_enc_free(qimg_text_enc *enc) {
    if (!enc) return;
    if (enc->model) transformer_free((transformer_model *)enc->model);
    if (enc->vocab) bpe_vocab_free((bpe_vocab *)enc->vocab);
    if (enc->gguf) gguf_close((gguf_context *)enc->gguf);
    free(enc);
}

float *qimg_text_enc_encode(qimg_text_enc *enc, const char *text,
                            int *out_n_tokens) {
    transformer_model *model = (transformer_model *)enc->model;
    bpe_vocab *vocab = (bpe_vocab *)enc->vocab;
    int n_embd = enc->n_embd;

    /* Wrap prompt in Qwen-Image chat template (matching ComfyUI's QwenImageTokenizer).
     * Template: <|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n
     * Special tokens: <|im_start|>=151644, <|im_end|>=151645
     * Without this template, the LLM produces wrong hidden states. */
    #define IM_START 151644
    #define IM_END   151645

    /* Tokenize the system prompt text (without special tokens) */
    static const char sys_text[] =
        "system\nDescribe the image by detailing the color, shape, size, texture, "
        "quantity, text, spatial relationships of the objects and background:";
    static const char user_text[] = "user\n";
    static const char asst_text[] = "assistant\n";

    int n_sys = bpe_tokenize(vocab, sys_text, (int)strlen(sys_text), NULL, 0);
    int text_len = (int)strlen(text);
    int n_user_prompt = bpe_tokenize(vocab, text, text_len, NULL, 0);
    int n_user_prefix = bpe_tokenize(vocab, user_text, (int)strlen(user_text), NULL, 0);
    int n_asst = bpe_tokenize(vocab, asst_text, (int)strlen(asst_text), NULL, 0);

    /* Total: <im_start> sys_tokens <im_end> \n <im_start> user_prefix user_tokens <im_end> \n <im_start> asst_tokens */
    int n_tokens = 1 + n_sys + 1 + 1 + n_user_prefix + n_user_prompt + 1 + 1 + n_asst;
    /* Actually: [IM_START] [sys_tokens...] [IM_END] [\n] [IM_START] [user\n] [prompt...] [IM_END] [\n] [IM_START] [assistant\n] */
    /* \n = token 198 in Qwen tokenizer */

    int32_t *tokens = (int32_t *)malloc((size_t)(n_tokens + 10) * sizeof(int32_t));
    int pos = 0;

    /* <|im_start|>system\n...system_prompt...<|im_end|>\n */
    tokens[pos++] = IM_START;
    pos += bpe_tokenize(vocab, sys_text, (int)strlen(sys_text), tokens + pos, n_sys);
    tokens[pos++] = IM_END;
    tokens[pos++] = 198;  /* \n */

    /* <|im_start|>user\n...prompt...<|im_end|>\n */
    tokens[pos++] = IM_START;
    pos += bpe_tokenize(vocab, user_text, (int)strlen(user_text), tokens + pos, n_user_prefix);
    pos += bpe_tokenize(vocab, text, text_len, tokens + pos, n_user_prompt);
    tokens[pos++] = IM_END;
    tokens[pos++] = 198;  /* \n */

    /* <|im_start|>assistant\n */
    tokens[pos++] = IM_START;
    pos += bpe_tokenize(vocab, asst_text, (int)strlen(asst_text), tokens + pos, n_asst);

    n_tokens = pos;

    #undef IM_START
    #undef IM_END

    fprintf(stderr, "qimg_text_enc: tokenized \"%s\" -> %d tokens:\n  [",
            text, n_tokens);
    for (int i = 0; i < n_tokens; i++)
        fprintf(stderr, "%d%s", tokens[i], i < n_tokens - 1 ? ", " : "");
    fprintf(stderr, "]\n");

    /* Run each token through transformer and collect hidden states */
    float *hidden_states = (float *)malloc((size_t)n_tokens * n_embd * sizeof(float));

    for (int i = 0; i < n_tokens; i++) {
        /* Forward pass: returns hidden state after final RMSNorm, before lm_head */
        float *h = transformer_forward(model, tokens[i], i);
        if (!h) {
            fprintf(stderr, "qimg_text_enc: forward failed at token %d\n", i);
            free(tokens);
            free(hidden_states);
            *out_n_tokens = 0;
            return NULL;
        }
        memcpy(hidden_states + (size_t)i * n_embd, h, (size_t)n_embd * sizeof(float));

        if (i == 0 || (i + 1) % 10 == 0 || i == n_tokens - 1)
            fprintf(stderr, "\r  qimg_text_enc: token %d/%d", i + 1, n_tokens);
    }
    fprintf(stderr, "\n");

    /* Strip template prefix (matching ComfyUI's encode_token_weights).
     * ComfyUI finds the 2nd <|im_start|> (151644), then if followed by
     * "user" (872) + "\n" (198), skips past them.
     * Result: only user prompt + assistant tokens remain. */
    int template_end = 0;
    {
        int count_im_start = 0;
        for (int i = 0; i < n_tokens; i++) {
            if (tokens[i] == 151644 && count_im_start < 2) {
                template_end = i;
                count_im_start++;
            }
        }
        /* Advance past "user\n" if present */
        if (n_tokens > template_end + 3 &&
            tokens[template_end + 1] == 872 &&   /* "user" */
            tokens[template_end + 2] == 198) {    /* "\n" */
            template_end += 3;
        }
    }

    int out_count = n_tokens - template_end;
    fprintf(stderr, "qimg_text_enc: stripping template prefix (%d tokens), returning %d tokens\n",
            template_end, out_count);

    /* Shift hidden states to remove template prefix */
    float *out_hidden = (float *)malloc((size_t)out_count * n_embd * sizeof(float));
    memcpy(out_hidden, hidden_states + (size_t)template_end * n_embd,
           (size_t)out_count * n_embd * sizeof(float));
    free(hidden_states);
    free(tokens);

    *out_n_tokens = out_count;
    return out_hidden;
}

#endif /* QIMG_TEXT_ENCODER_IMPLEMENTATION */
#endif /* QIMG_TEXT_ENCODER_H */
