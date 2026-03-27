/*
 * test_qwen_image.c - Qwen-Image end-to-end text-to-image test
 *
 * Modes:
 *   --test-vae     : Load VAE, decode random latent, save output
 *   --test-dit     : Load DiT, run one denoising step, save output
 *   --test-sched   : Test scheduler timestep generation
 *   --test-enc     : Load text encoder, encode a prompt
 *   --generate     : Full pipeline with text encoder
 *
 * Build:
 *   cc -O2 -o test_qwen_image test_qwen_image.c -lm -lpthread
 */

#define SAFETENSORS_IMPLEMENTATION
#define GGUF_LOADER_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION
#define BPE_TOKENIZER_IMPLEMENTATION
#define TRANSFORMER_IMPLEMENTATION
#define QIMG_SCHEDULER_IMPLEMENTATION
#define QIMG_DIT_IMPLEMENTATION
#define QIMG_VAE_IMPLEMENTATION
#define QIMG_TEXT_ENCODER_IMPLEMENTATION

#include "../../common/safetensors.h"
#include "../../common/gguf_loader.h"
#include "../../common/ggml_dequant.h"
#include "../../common/bpe_tokenizer.h"
#include "../../common/transformer.h"
#include "../../common/qwen_image_scheduler.h"
#include "../../common/qwen_image_dit.h"
#include "../../common/qwen_image_vae.h"
#include "../../common/qwen_image_text_encoder.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ---- Save to .npy (float32) ---- */

static void save_npy_f32(const char *path, const float *data,
                         int ndims, const int *shape) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot open %s for writing\n", path); return; }

    /* NPY format header */
    char shape_str[128] = "(";
    for (int d = 0; d < ndims; d++) {
        char tmp[32];
        snprintf(tmp, sizeof(tmp), "%d%s", shape[d], d < ndims - 1 ? ", " : "");
        strcat(shape_str, tmp);
    }
    if (ndims == 1) strcat(shape_str, ",");
    strcat(shape_str, ")");

    /* Build header dict */
    char dict[256];
    int dlen = snprintf(dict, sizeof(dict),
        "{'descr': '<f4', 'fortran_order': False, 'shape': %s, }", shape_str);

    /* Pad to multiple of 64 (10 bytes for magic+version+header_len) */
    int total_hdr = 10 + dlen + 1;  /* +1 for newline */
    int pad = 64 - (total_hdr % 64);
    if (pad == 64) pad = 0;
    int header_data_len = dlen + pad + 1;

    uint8_t magic[10] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0, 0, 0};
    magic[8] = (uint8_t)(header_data_len & 0xFF);
    magic[9] = (uint8_t)((header_data_len >> 8) & 0xFF);
    fwrite(magic, 1, 10, f);

    /* Write dict + padding + newline */
    fwrite(dict, 1, (size_t)dlen, f);
    for (int i = 0; i < pad; i++) fputc(' ', f);
    fputc('\n', f);

    /* Write data */
    size_t total_elems = 1;
    for (int d = 0; d < ndims; d++) total_elems *= (size_t)shape[d];
    fwrite(data, sizeof(float), total_elems, f);
    fclose(f);
    fprintf(stderr, "Saved %s (%zu elements)\n", path, total_elems);
}

/* ---- Simple PRNG ---- */

static uint64_t rng_state = 42;

static float randn(void) {
    /* Box-Muller */
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u1 = (double)(rng_state >> 11) / (double)(1ULL << 53);
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u2 = (double)(rng_state >> 11) / (double)(1ULL << 53);
    if (u1 < 1e-10) u1 = 1e-10;
    return (float)(sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2));
}

/* ---- Test modes ---- */

static int test_scheduler(void) {
    fprintf(stderr, "\n=== Scheduler Test ===\n");
    qimg_scheduler sched;
    qimg_sched_init(&sched);

    /* Test with different image sizes */
    int sizes[] = {256, 1024, 4096};
    for (int s = 0; s < 3; s++) {
        qimg_sched_set_timesteps(&sched, 30, sizes[s]);
        fprintf(stderr, "img_seq_len=%d, n_steps=%d:\n", sizes[s], sched.n_steps);
        fprintf(stderr, "  sigma[0]=%.4f sigma[%d]=%.4f\n",
                sched.sigmas[0], sched.n_steps, sched.sigmas[sched.n_steps]);
        fprintf(stderr, "  timesteps: [%.1f", sched.timesteps[0]);
        for (int i = 1; i < 5 && i < sched.n_steps; i++)
            fprintf(stderr, ", %.1f", sched.timesteps[i]);
        fprintf(stderr, ", ... , %.1f]\n", sched.timesteps[sched.n_steps - 1]);
    }
    fprintf(stderr, "Scheduler test passed.\n");
    return 0;
}

/* Layer-by-layer VAE decode with .npy dumps at each stage */
static int test_vae(const char *vae_path, int lat_h, int lat_w, uint64_t seed) {
    fprintf(stderr, "\n=== VAE Layer-by-Layer Decode ===\n");
    qimg_vae_model *vae = qimg_vae_load(vae_path);
    if (!vae) return 1;

    /* Use PyTorch-compatible randn (seed via torch.manual_seed in ref) */
    /* Our PRNG differs from torch, so load from npy if available, else use ours */
    int z_dim = 16;
    size_t lat_size = (size_t)z_dim * lat_h * lat_w;
    float *x = (float *)malloc(lat_size * sizeof(float));

    /* Try to load reference input latent */
    FILE *lat_f = fopen("../../ref/qwen_image/output/vae_00_input.npy", "rb");
    if (lat_f) {
        /* Skip npy header: find first \n after shape dict */
        int ch;
        int newlines = 0;
        while ((ch = fgetc(lat_f)) != EOF) {
            if (ch == '\n') { newlines++; if (newlines >= 1) break; }
        }
        /* Header is exactly 10 + header_len bytes; just seek to where data starts */
        /* Re-open and parse properly */
        fclose(lat_f);
        /* Simple approach: read raw npy */
        lat_f = fopen("../../ref/qwen_image/output/vae_00_input.npy", "rb");
        uint8_t magic[10];
        fread(magic, 1, 10, lat_f);
        int hdr_len = (int)magic[8] | ((int)magic[9] << 8);
        fseek(lat_f, 10 + hdr_len, SEEK_SET);
        size_t got = fread(x, sizeof(float), lat_size, lat_f);
        fclose(lat_f);
        if (got == lat_size) {
            fprintf(stderr, "Loaded ref input latent (%zu floats)\n", got);
        } else {
            fprintf(stderr, "Partial read (%zu/%zu), using own PRNG\n", got, lat_size);
            rng_state = seed;
            for (size_t i = 0; i < lat_size; i++) x[i] = randn();
        }
    } else {
        fprintf(stderr, "No ref input found, using own PRNG (seed=%lu)\n", (unsigned long)seed);
        rng_state = seed;
        for (size_t i = 0; i < lat_size; i++) x[i] = randn();
    }

    int h = lat_h, w = lat_w, c = z_dim;
    int stage = 0;
    char fname[256];

    /* Stage 0: input */
    {
        int sh[] = {c, h, w};
        snprintf(fname, sizeof(fname), "vae_%02d_input.npy", stage);
        save_npy_f32(fname, x, 3, sh);
        stage++;
    }

    /* Stage 1: post_quant_conv */
    if (vae->pqc_weight) {
        int spatial = h * w;
        float *out = (float *)malloc((size_t)c * spatial * sizeof(float));
        for (int s = 0; s < spatial; s++)
            for (int o = 0; o < c; o++) {
                float sum = vae->pqc_bias ? vae->pqc_bias[o] : 0.0f;
                for (int i = 0; i < c; i++)
                    sum += vae->pqc_weight[(size_t)o * c + i] *
                           x[(size_t)i * spatial + s];
                out[(size_t)o * spatial + s] = sum;
            }
        free(x); x = out;
    }
    {
        int sh[] = {c, h, w};
        snprintf(fname, sizeof(fname), "vae_%02d_post_quant.npy", stage);
        save_npy_f32(fname, x, 3, sh);
        stage++;
    }

    /* Stage 2: decoder.conv1 */
    c = 384;
    {
        float *out = (float *)malloc((size_t)c * h * w * sizeof(float));
        qimg_vae_conv2d(out, x, vae->dec_conv1_weight, vae->dec_conv1_bias,
                        16, h, w, c, 3, 3, 1);
        free(x); x = out;
    }
    {
        int sh[] = {c, h, w};
        snprintf(fname, sizeof(fname), "vae_%02d_dec_conv1.npy", stage);
        save_npy_f32(fname, x, 3, sh);
        stage++;
    }

    /* Stage 3: middle.0 resblock */
    {
        float *out = (float *)malloc((size_t)c * h * w * sizeof(float));
        qimg_vae_resblock_forward(out, x, &vae->mid_res0, h, w);
        free(x); x = out;
    }
    {
        int sh[] = {c, h, w};
        snprintf(fname, sizeof(fname), "vae_%02d_mid_res0.npy", stage);
        save_npy_f32(fname, x, 3, sh);
        stage++;
    }

    /* Stage 4: middle attention */
    {
        float *out = (float *)malloc((size_t)c * h * w * sizeof(float));
        qimg_vae_spatial_attn(out, x, vae->mid_attn.norm_gamma,
                              vae->mid_attn.qkv_weight, vae->mid_attn.qkv_bias,
                              vae->mid_attn.proj_weight, vae->mid_attn.proj_bias,
                              c, h, w);
        free(x); x = out;
    }
    {
        int sh[] = {c, h, w};
        snprintf(fname, sizeof(fname), "vae_%02d_mid_attn.npy", stage);
        save_npy_f32(fname, x, 3, sh);
        stage++;
    }

    /* Stage 5: middle.2 resblock */
    {
        float *out = (float *)malloc((size_t)c * h * w * sizeof(float));
        qimg_vae_resblock_forward(out, x, &vae->mid_res2, h, w);
        free(x); x = out;
    }
    {
        int sh[] = {c, h, w};
        snprintf(fname, sizeof(fname), "vae_%02d_mid_res2.npy", stage);
        save_npy_f32(fname, x, 3, sh);
        stage++;
    }

    /* Upsample blocks 0-14 */
    for (int i = 0; i < vae->n_up_blocks; i++) {
        qimg_vae_resblock *rb = &vae->up_res[i];
        if (rb->conv1_weight) {
            int new_c = rb->c_out;
            float *out = (float *)malloc((size_t)new_c * h * w * sizeof(float));
            qimg_vae_resblock_forward(out, x, rb, h, w);
            free(x); x = out;
            c = new_c;
            int sh[] = {c, h, w};
            snprintf(fname, sizeof(fname), "vae_%02d_up%d_res.npy", stage, i);
            save_npy_f32(fname, x, 3, sh);
            stage++;
        }
        if (vae->up_has_sample[i]) {
            qimg_vae_upsample *us = &vae->up_sample[i];
            float *up = qimg_vae_nn_upsample(x, c, h, w);
            h *= 2; w *= 2;
            int new_c2 = us->c_out;
            float *conv_out = (float *)malloc((size_t)new_c2 * h * w * sizeof(float));
            qimg_vae_conv2d_zero(conv_out, up, us->conv_weight, us->conv_bias,
                                 c, h, w, new_c2, 3, 3, 1);
            free(up); free(x);
            x = conv_out;
            c = new_c2;
            int sh[] = {c, h, w};
            snprintf(fname, sizeof(fname), "vae_%02d_up%d_resample.npy", stage, i);
            save_npy_f32(fname, x, 3, sh);
            stage++;
        }
    }

    /* Head: GroupNorm → SiLU → Conv */
    {
        float *normed = (float *)malloc((size_t)c * h * w * sizeof(float));
        qimg_vae_groupnorm(normed, x, vae->head_norm_gamma, c, h, w);
        qimg_vae_silu(normed, c * h * w);
        float *rgb = (float *)malloc((size_t)3 * h * w * sizeof(float));
        qimg_vae_conv2d(rgb, normed, vae->head_conv_weight, vae->head_conv_bias,
                        c, h, w, 3, 3, 3, 1);
        free(normed); free(x);
        x = rgb;
        c = 3;
    }
    {
        int sh[] = {c, h, w};
        snprintf(fname, sizeof(fname), "vae_%02d_output.npy", stage);
        save_npy_f32(fname, x, 3, sh);
    }

    /* Save PPM */
    {
        FILE *fp = fopen("vae_output.ppm", "wb");
        if (fp) {
            fprintf(fp, "P6\n%d %d\n255\n", w, h);
            for (int y = 0; y < h; y++)
                for (int px = 0; px < w; px++) {
                    uint8_t p[3];
                    for (int ch = 0; ch < 3; ch++) {
                        float v = x[(size_t)ch * h * w + y * w + px];
                        v = v * 0.5f + 0.5f;
                        if (v < 0.0f) v = 0.0f;
                        if (v > 1.0f) v = 1.0f;
                        p[ch] = (uint8_t)(v * 255.0f);
                    }
                    fwrite(p, 1, 3, fp);
                }
            fclose(fp);
            fprintf(stderr, "Saved vae_output.ppm (%dx%d)\n", w, h);
        }
    }

    free(x);
    qimg_vae_free(vae);
    return 0;
}

static void dit_dump_cb(const char *name, const float *data, int n, void *ctx) {
    (void)ctx;
    char path[256];
    snprintf(path, sizeof(path), "dit_blk0_%s.npy", name);
    int sh[] = {n};
    save_npy_f32(path, data, 1, sh);
}

static int test_dit(const char *dit_path, int lat_h, int lat_w, uint64_t seed) {
    fprintf(stderr, "\n=== DiT Layer-by-Layer Verify ===\n");
    qimg_dit_model *dit = qimg_dit_load_gguf(dit_path);
    if (!dit) return 1;

    int ps = dit->patch_size;
    int hp = lat_h / ps, wp = lat_w / ps;
    int n_img = hp * wp;
    int n_txt = 10;
    int in_ch = dit->in_channels;
    int txt_dim = dit->txt_dim;
    int dim = dit->hidden_dim;

    fprintf(stderr, "n_img=%d, n_txt=%d, in_ch=%d, txt_dim=%d, dim=%d\n",
            n_img, n_txt, in_ch, txt_dim, dim);

    /* Try to load reference inputs (from Python np.random.seed(42)) */
    float *img_tokens = (float *)malloc((size_t)n_img * in_ch * sizeof(float));
    float *txt_tokens = (float *)malloc((size_t)n_txt * txt_dim * sizeof(float));
    FILE *f;
    int use_ref = 0;
    f = fopen("../../ref/qwen_image/output/dit_img_input.npy", "rb");
    if (f) {
        uint8_t hdr[10]; fread(hdr, 1, 10, f);
        int hl = (int)hdr[8] | ((int)hdr[9] << 8);
        fseek(f, 10 + hl, SEEK_SET);
        fread(img_tokens, sizeof(float), (size_t)n_img * in_ch, f);
        fclose(f);
        f = fopen("../../ref/qwen_image/output/dit_txt_input.npy", "rb");
        if (f) {
            fread(hdr, 1, 10, f);
            hl = (int)hdr[8] | ((int)hdr[9] << 8);
            fseek(f, 10 + hl, SEEK_SET);
            fread(txt_tokens, sizeof(float), (size_t)n_txt * txt_dim, f);
            fclose(f);
            use_ref = 1;
            fprintf(stderr, "Loaded ref inputs\n");
        }
    }
    if (!use_ref) {
        fprintf(stderr, "Using own PRNG for inputs\n");
        rng_state = seed;
        for (int i = 0; i < n_img * in_ch; i++) img_tokens[i] = randn() * 0.1f;
        for (int i = 0; i < n_txt * txt_dim; i++) txt_tokens[i] = randn() * 0.1f;
    }

    float timestep = 500.0f;
    char fname[256];

    /* Save inputs */
    { int sh[] = {n_img, in_ch}; save_npy_f32("dit_img_input.npy", img_tokens, 2, sh); }
    { int sh[] = {n_txt, txt_dim}; save_npy_f32("dit_txt_input.npy", txt_tokens, 2, sh); }

    /* ---- Step-by-step DiT forward with dumps ---- */

    /* 1. Timestep embedding */
    float t_sin[256];
    qimg_timestep_embed(t_sin, timestep, 256);

    float *t_emb = (float *)malloc((size_t)dim * sizeof(float));
    qimg_batch_gemm(t_emb, &dit->t_fc1_w, &dit->t_fc1_b, t_sin, 1, dim, 256, 1);
    qimg_silu(t_emb, dim);
    float *t_emb2 = (float *)malloc((size_t)dim * sizeof(float));
    qimg_batch_gemm(t_emb2, &dit->t_fc2_w, &dit->t_fc2_b, t_emb, 1, dim, dim, 1);
    free(t_emb); t_emb = t_emb2;
    { int sh[] = {dim}; save_npy_f32("dit_t_emb.npy", t_emb, 1, sh); }

    /* 2. Text input: RMSNorm -> Linear */
    float *txt = (float *)malloc((size_t)n_txt * dim * sizeof(float));
    {
        float *txt_normed = (float *)malloc((size_t)n_txt * txt_dim * sizeof(float));
        memcpy(txt_normed, txt_tokens, (size_t)n_txt * txt_dim * sizeof(float));
        if (dit->txt_norm_w.data) {
            float *nw = qimg_dequant_full(&dit->txt_norm_w);
            qimg_rmsnorm(txt_normed, n_txt, txt_dim, nw, dit->ln_eps);
            free(nw);
        }
        qimg_batch_gemm(txt, &dit->txt_in_w, &dit->txt_in_b,
                        txt_normed, n_txt, dim, txt_dim, 1);
        free(txt_normed);
    }
    { int sh[] = {n_txt, dim}; save_npy_f32("dit_txt_projected.npy", txt, 2, sh); }

    /* 3. Image input: Linear */
    float *img = (float *)malloc((size_t)n_img * dim * sizeof(float));
    qimg_batch_gemm(img, &dit->img_in_w, &dit->img_in_b,
                    img_tokens, n_img, dim, in_ch, 1);
    { int sh[] = {n_img, dim}; save_npy_f32("dit_img_projected.npy", img, 2, sh); }

    /* 4. Block 0: modulation */
    {
        float *t_silu = (float *)malloc((size_t)dim * sizeof(float));
        memcpy(t_silu, t_emb, (size_t)dim * sizeof(float));
        qimg_silu(t_silu, dim);
        float *img_mod = (float *)malloc((size_t)6 * dim * sizeof(float));
        qimg_batch_gemm(img_mod, &dit->blocks[0].img_mod_w,
                        &dit->blocks[0].img_mod_b,
                        t_silu, 1, 6 * dim, dim, 1);
        { int sh[] = {6 * dim}; save_npy_f32("dit_blk0_img_mod.npy", img_mod, 1, sh); }
        free(img_mod);
        free(t_silu);
    }

    fprintf(stderr, "Intermediates saved. Run full forward with block 0 dumps...\n");

    /* Enable block 0 dump */
    dit->dump_block = 0;
    dit->dump_fn = dit_dump_cb;
    dit->dump_ctx = NULL;

    /* Full forward for final output */
    float *out = (float *)malloc((size_t)n_img * in_ch * sizeof(float));
    clock_t t0 = clock();
    qimg_dit_forward(out, img_tokens, n_img, txt_tokens, n_txt, timestep,
                     dit, 1);
    clock_t t1 = clock();
    fprintf(stderr, "DiT forward: %.2f s\n", (double)(t1 - t0) / CLOCKS_PER_SEC);

    { int sh[] = {n_img, in_ch}; save_npy_f32("dit_output.npy", out, 2, sh); }

    float min_v = out[0], max_v = out[0], sum_v = 0;
    for (int i = 0; i < n_img * in_ch; i++) {
        if (out[i] < min_v) min_v = out[i];
        if (out[i] > max_v) max_v = out[i];
        sum_v += out[i];
    }
    fprintf(stderr, "Output: min=%.6f max=%.6f mean=%.6f\n",
            min_v, max_v, sum_v / (n_img * in_ch));

    free(out); free(img); free(txt); free(t_emb);
    free(img_tokens); free(txt_tokens);
    qimg_dit_free(dit);
    return 0;
}

static int test_text_encoder(const char *enc_path, const char *prompt) {
    fprintf(stderr, "\n=== Text Encoder Test ===\n");
    qimg_text_enc *enc = qimg_text_enc_load(enc_path);
    if (!enc) return 1;

    int n_tokens = 0;
    float *hidden = qimg_text_enc_encode(enc, prompt, &n_tokens);
    if (!hidden) { qimg_text_enc_free(enc); return 1; }

    fprintf(stderr, "Text encoder output: [%d, %d]\n", n_tokens, enc->n_embd);

    /* Stats */
    float min_v = hidden[0], max_v = hidden[0], sum_v = 0;
    int total = n_tokens * enc->n_embd;
    for (int i = 0; i < total; i++) {
        if (hidden[i] < min_v) min_v = hidden[i];
        if (hidden[i] > max_v) max_v = hidden[i];
        sum_v += hidden[i];
    }
    fprintf(stderr, "  min=%.4f max=%.4f mean=%.4f\n",
            min_v, max_v, sum_v / total);

    /* Save */
    int sh[] = {n_tokens, enc->n_embd};
    save_npy_f32("text_hidden_states.npy", hidden, 2, sh);

    free(hidden);
    qimg_text_enc_free(enc);
    return 0;
}

static int test_full_pipeline(const char *dit_path, const char *vae_path,
                              const char *enc_path, const char *prompt,
                              int out_h, int out_w, int n_steps, uint64_t seed) {
    fprintf(stderr, "\n=== Full Pipeline ===\n");

    /* Load models */
    qimg_dit_model *dit = qimg_dit_load_gguf(dit_path);
    if (!dit) return 1;
    qimg_vae_model *vae = qimg_vae_load(vae_path);
    if (!vae) return 1;

    int ps = dit->patch_size;
    int lat_h = out_h / 8;  /* VAE downscale factor */
    int lat_w = out_w / 8;
    int hp = lat_h / ps, wp = lat_w / ps;
    int n_img = hp * wp;
    int in_ch = dit->in_channels;  /* 64 */
    int lat_ch = 16;               /* latent channels */

    /* Text conditioning — load from .npy if available for reproducibility */
    int n_txt = 0;
    int txt_dim = dit->txt_dim;
    float *txt_tokens = NULL;
    {
        const char *th_path = "../../ref/qwen_image/output/../../cpu/qwen_image/text_hidden_states.npy";
        /* Try the same text_hidden_states.npy that PyTorch reference uses */
        FILE *tf = fopen("text_hidden_states.npy", "rb");
        if (tf) {
            uint8_t hdr[10];
            if (fread(hdr, 1, 10, tf) == 10) {
                int hl = (int)hdr[8] | ((int)hdr[9] << 8);
                /* Parse shape from header to get n_txt */
                char hbuf[512];
                fseek(tf, 10, SEEK_SET);
                if (fread(hbuf, 1, (size_t)hl, tf) == (size_t)hl) {
                    hbuf[hl] = 0;
                    /* Extract first shape dim: 'shape': (N, D) */
                    char *sp = strstr(hbuf, "shape");
                    if (sp) {
                        sp = strchr(sp, '(');
                        if (sp) n_txt = atoi(sp + 1);
                    }
                }
                if (n_txt > 0 && txt_dim > 0) {
                    fseek(tf, 10 + hl, SEEK_SET);
                    txt_tokens = (float *)malloc((size_t)n_txt * txt_dim * sizeof(float));
                    size_t got = fread(txt_tokens, sizeof(float), (size_t)n_txt * txt_dim, tf);
                    if ((int)got == n_txt * txt_dim) {
                        fprintf(stderr, "Loaded text_hidden_states.npy (%d tokens)\n", n_txt);
                    } else {
                        free(txt_tokens); txt_tokens = NULL; n_txt = 0;
                    }
                }
            }
            fclose(tf);
        }
    }
    if (!txt_tokens && enc_path && prompt) {
        qimg_text_enc *enc = qimg_text_enc_load(enc_path);
        if (enc) {
            txt_tokens = qimg_text_enc_encode(enc, prompt, &n_txt);
            qimg_text_enc_free(enc);
        }
    }
    if (!txt_tokens) {
        fprintf(stderr, "Using random text conditioning\n");
        n_txt = 20;
        rng_state = seed + 1;
        txt_tokens = (float *)malloc((size_t)n_txt * txt_dim * sizeof(float));
        for (int i = 0; i < n_txt * txt_dim; i++) txt_tokens[i] = randn() * 0.1f;
    }

    /* Initialize noise — try loading from ref .npy for comparison */
    size_t lat_total = (size_t)lat_ch * lat_h * lat_w;
    float *latent = (float *)malloc(lat_total * sizeof(float));
    {
        char noise_path[512];
        snprintf(noise_path, sizeof(noise_path),
                 "../../ref/qwen_image/output/ground_truth_%d_noise.npy", out_h);
        FILE *nf = fopen(noise_path, "rb");
        int loaded = 0;
        if (nf) {
            uint8_t hdr[10];
            if (fread(hdr, 1, 10, nf) == 10) {
                int hl = (int)hdr[8] | ((int)hdr[9] << 8);
                fseek(nf, 10 + hl, SEEK_SET);
                /* Skip batch dim: [1, 16, H, W] → read 16*H*W */
                if (fread(latent, sizeof(float), lat_total, nf) == lat_total) {
                    loaded = 1;
                    fprintf(stderr, "Loaded noise from %s\n", noise_path);
                }
            }
            fclose(nf);
        }
        if (!loaded) {
            rng_state = seed;
            for (size_t i = 0; i < lat_total; i++) latent[i] = randn();
            fprintf(stderr, "Using own PRNG noise (seed=%lu)\n", (unsigned long)seed);
        }
    }

    /* Setup scheduler */
    qimg_scheduler sched;
    qimg_sched_init(&sched);
    qimg_sched_set_timesteps(&sched, n_steps, n_img);

    fprintf(stderr, "Pipeline: %dx%d image, %d latent, %d patches, %d steps\n",
            out_w, out_h, lat_h, n_img, n_steps);

    /* Patchify */
    float *img_tokens = (float *)malloc((size_t)n_img * in_ch * sizeof(float));

    /* Denoising loop */
    clock_t total_start = clock();
    for (int step = 0; step < n_steps; step++) {
        float t = sched.timesteps[step];
        fprintf(stderr, "\nStep %d/%d (t=%.1f)\n", step + 1, n_steps, t);

        /* Patchify current latent */
        qimg_dit_patchify(img_tokens, latent, lat_ch, lat_h, lat_w, ps);

        /* DiT forward: predict velocity */
        float *velocity = (float *)malloc((size_t)n_img * in_ch * sizeof(float));
        qimg_dit_forward(velocity, img_tokens, n_img, txt_tokens, n_txt, t,
                         dit, 1);

        /* Unpatchify velocity */
        float *vel_latent = (float *)malloc((size_t)lat_ch * lat_h * lat_w * sizeof(float));
        qimg_dit_unpatchify(vel_latent, velocity, n_img, lat_ch, lat_h, lat_w, ps);
        free(velocity);

        /* Euler step */
        qimg_sched_step(latent, vel_latent, lat_ch * lat_h * lat_w, step, &sched);
        free(vel_latent);
    }
    clock_t total_end = clock();
    fprintf(stderr, "\nDenoising: %.2f s total\n",
            (double)(total_end - total_start) / CLOCKS_PER_SEC);

    free(img_tokens);
    free(txt_tokens);

    /* VAE decode */
    float *rgb = (float *)malloc((size_t)3 * out_h * out_w * sizeof(float));
    qimg_vae_decode(rgb, latent, lat_h, lat_w, vae);

    /* Save PPM */
    {
        FILE *fp = fopen("qwen_image_output.ppm", "wb");
        if (fp) {
            fprintf(fp, "P6\n%d %d\n255\n", out_w, out_h);
            for (int y = 0; y < out_h; y++)
                for (int x = 0; x < out_w; x++) {
                    uint8_t px[3];
                    for (int c = 0; c < 3; c++) {
                        float v = rgb[(size_t)c * out_h * out_w + y * out_w + x];
                        v = v * 0.5f + 0.5f;
                        if (v < 0) v = 0; if (v > 1) v = 1;
                        px[c] = (uint8_t)(v * 255.0f);
                    }
                    fwrite(px, 1, 3, fp);
                }
            fclose(fp);
            fprintf(stderr, "Saved qwen_image_output.ppm (%dx%d)\n", out_w, out_h);
        }
    }

    /* Save latent */
    int lat_shape[] = {1, lat_ch, lat_h, lat_w};
    save_npy_f32("pipeline_latent_final.npy", latent, 4, lat_shape);

    free(rgb);
    free(latent);
    qimg_vae_free(vae);
    qimg_dit_free(dit);
    return 0;
}

/* ---- Main ---- */

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s <mode> [options]\n", prog);
    fprintf(stderr, "Modes:\n");
    fprintf(stderr, "  --test-sched                   Test scheduler\n");
    fprintf(stderr, "  --test-vae <vae.safetensors>    Test VAE decode\n");
    fprintf(stderr, "  --test-dit <dit.gguf>           Test DiT single step\n");
    fprintf(stderr, "  --test-enc <llm.gguf>           Test text encoder\n");
    fprintf(stderr, "  --generate <dit> <vae> <llm>    Full pipeline\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --height <h>     Output height (default 256)\n");
    fprintf(stderr, "  --width <w>      Output width (default 256)\n");
    fprintf(stderr, "  --steps <n>      Denoising steps (default 20)\n");
    fprintf(stderr, "  --seed <s>       Random seed (default 42)\n");
    fprintf(stderr, "  --prompt <text>  Text prompt (default: 'a red apple')\n");
}

int main(int argc, char **argv) {
    int out_h = 256, out_w = 256, n_steps = 20;
    uint64_t seed = 42;
    const char *dit_path = NULL, *vae_path = NULL, *enc_path = NULL;
    const char *prompt = "a red apple on a white table";
    const char *mode = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--test-sched") == 0) mode = "sched";
        else if (strcmp(argv[i], "--test-vae") == 0 && i + 1 < argc) {
            mode = "vae"; vae_path = argv[++i];
        }
        else if (strcmp(argv[i], "--test-dit") == 0 && i + 1 < argc) {
            mode = "dit"; dit_path = argv[++i];
        }
        else if (strcmp(argv[i], "--test-enc") == 0 && i + 1 < argc) {
            mode = "enc"; enc_path = argv[++i];
        }
        else if (strcmp(argv[i], "--generate") == 0 && i + 3 < argc) {
            mode = "gen"; dit_path = argv[++i]; vae_path = argv[++i]; enc_path = argv[++i];
        }
        else if (strcmp(argv[i], "--height") == 0 && i + 1 < argc)
            out_h = atoi(argv[++i]);
        else if (strcmp(argv[i], "--width") == 0 && i + 1 < argc)
            out_w = atoi(argv[++i]);
        else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc)
            n_steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
            seed = (uint64_t)atoll(argv[++i]);
        else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc)
            prompt = argv[++i];
    }

    if (!mode) { usage(argv[0]); return 1; }

    if (strcmp(mode, "sched") == 0) return test_scheduler();
    if (strcmp(mode, "vae") == 0) return test_vae(vae_path, out_h / 8, out_w / 8, seed);
    if (strcmp(mode, "dit") == 0) return test_dit(dit_path, out_h / 8, out_w / 8, seed);
    if (strcmp(mode, "enc") == 0) return test_text_encoder(enc_path, prompt);
    if (strcmp(mode, "gen") == 0) return test_full_pipeline(dit_path, vae_path,
                                      enc_path, prompt, out_h, out_w, n_steps, seed);
    return 1;
}
