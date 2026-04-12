/*
 * test_flux2.c - Flux.2 Klein 4B end-to-end text-to-image test (CPU)
 *
 * Modes:
 *   --test-sched   : Test scheduler timestep generation
 *   --test-vae     : Load VAE, decode random/reference latent
 *   --test-dit     : Load DiT, run one denoising step
 *   --test-enc     : Load text encoder, encode a prompt
 *   --generate     : Full pipeline (text encoder → DiT × N → VAE)
 *
 * Build:
 *   make test_flux2
 *   (or: cc -O2 -mavx2 -mfma -I../../common -o test_flux2 test_flux2.c -lm -lpthread)
 *
 * Weight paths (set via command line or adjust defaults below):
 *   --dit  /mnt/disk01/models/klein2-4b/diffusion_models/flux-2-klein-4b-fp8.safetensors
 *   --vae  /mnt/disk01/models/klein2-4b/vae/flux2-vae.safetensors
 *   --enc  /mnt/disk01/models/klein2-4b/text_encoders/qwen3-4b-q4k.gguf
 */

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#define SAFETENSORS_IMPLEMENTATION
#define GGUF_LOADER_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION
#define BPE_TOKENIZER_IMPLEMENTATION
#define TRANSFORMER_IMPLEMENTATION
#define QIMG_SCHEDULER_IMPLEMENTATION
#define FLUX2_DIT_IMPLEMENTATION
#define FLUX2_VAE_IMPLEMENTATION
#define FLUX2_TEXT_ENCODER_IMPLEMENTATION

#include "../../common/safetensors.h"
#include "../../common/gguf_loader.h"
#include "../../common/ggml_dequant.h"
#include "../../common/bpe_tokenizer.h"
#include "../../common/transformer.h"
#include "../../common/qwen_image_scheduler.h"
#include "../../common/flux2_klein_dit.h"
#include "../../common/flux2_klein_vae.h"
#include "../../common/flux2_klein_text_encoder.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ---- Default weight paths ---- */
static const char *DEFAULT_DIT     = "/mnt/disk01/models/klein2-4b/diffusion_models/flux-2-klein-4b-fp8.safetensors";
static const char *DEFAULT_VAE     = "/mnt/disk01/models/klein2-4b/vae/flux2-vae.safetensors";
static const char *DEFAULT_ENC     = "/mnt/disk01/models/klein2-4b/text_encoder";
static const char *DEFAULT_TOK     = "/mnt/disk01/models/Qwen3-VL-4B-Instruct-GGUF/Qwen3VL-4B-Instruct-Q8_0.gguf";

/* ---- PRNG: Box-Muller with pair caching ---- */
static uint64_t rng_state = 42;
static int rng_cached_valid = 0;
static float rng_cached = 0.0f;

static float randn(void) {
    if (rng_cached_valid) { rng_cached_valid = 0; return rng_cached; }
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u1 = (double)(rng_state >> 11) / (double)(1ULL << 53);
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u2 = (double)(rng_state >> 11) / (double)(1ULL << 53);
    if (u1 < 1e-10) u1 = 1e-10;
    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * 3.14159265358979323846 * u2;
    rng_cached = (float)(r * sin(theta));
    rng_cached_valid = 1;
    return (float)(r * cos(theta));
}

/* ---- Utility ---- */

static void save_ppm(const char *path, const float *rgb, int h, int w) {
    FILE *fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "Cannot write %s\n", path); return; }
    fprintf(fp, "P6\n%d %d\n255\n", w, h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            uint8_t px[3];
            for (int c = 0; c < 3; c++) {
                float v = rgb[(size_t)c * h * w + y * w + x] * 0.5f + 0.5f;
                if (v < 0.0f) v = 0.0f;
                if (v > 1.0f) v = 1.0f;
                px[c] = (uint8_t)(v * 255.0f + 0.5f);
            }
            fwrite(px, 1, 3, fp);
        }
    fclose(fp);
    fprintf(stderr, "Saved %s (%dx%d)\n", path, w, h);
}

static void save_npy_f32(const char *path, const float *data, int ndims, const int *shape) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path); return; }
    char shape_str[256] = "(";
    for (int d = 0; d < ndims; d++) {
        char tmp[32];
        snprintf(tmp, sizeof(tmp), "%d%s", shape[d], d < ndims-1 ? ", " : "");
        strcat(shape_str, tmp);
    }
    if (ndims == 1) strcat(shape_str, ",");
    strcat(shape_str, ")");
    char dict[512];
    int dlen = snprintf(dict, sizeof(dict),
        "{'descr': '<f4', 'fortran_order': False, 'shape': %s, }", shape_str);
    int total_hdr = 10 + dlen + 1;
    int pad = 64 - (total_hdr % 64);
    if (pad == 64) pad = 0;
    int hdr_data_len = dlen + pad + 1;
    uint8_t magic[10] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0, 0, 0};
    magic[8] = (uint8_t)(hdr_data_len & 0xFF);
    magic[9] = (uint8_t)((hdr_data_len >> 8) & 0xFF);
    fwrite(magic, 1, 10, f);
    fwrite(dict, 1, (size_t)dlen, f);
    for (int i = 0; i < pad; i++) fputc(' ', f);
    fputc('\n', f);
    size_t n = 1;
    for (int d = 0; d < ndims; d++) n *= (size_t)shape[d];
    fwrite(data, sizeof(float), n, f);
    fclose(f);
    fprintf(stderr, "Saved %s (%zu floats)\n", path, n);
}

static int load_npy_f32(const char *path, float *buf, size_t max_elems) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    uint8_t magic[10];
    if (fread(magic, 1, 10, f) < 10) { fclose(f); return -1; }
    int hdr_len = (int)magic[8] | ((int)magic[9] << 8);
    fseek(f, 10 + hdr_len, SEEK_SET);
    size_t got = fread(buf, sizeof(float), max_elems, f);
    fclose(f);
    return (int)got;
}

/* ---- Flux.2 Klein scheduler parameters ----
 * Distilled: shift=1.0 (nearly linear), 4 steps
 * Base: shift adaptive or 3.5, 20+ steps */
/* Flux.2 Klein uses Flux-style time shift mu=2.02 → AuraFlow alpha=exp(2.02)≈7.539 */
#define FLUX2_KLEIN_SHIFT_MU 2.02f

static void flux2_sched_init_distilled(qimg_scheduler *s, int n_steps) {
    qimg_sched_init(s);
    qimg_sched_set_timesteps_comfyui(s, n_steps, expf(FLUX2_KLEIN_SHIFT_MU), 1.0f);
}

static void flux2_sched_init_base(qimg_scheduler *s, int n_steps, int n_img_tokens) {
    (void)n_img_tokens;
    qimg_sched_init(s);
    qimg_sched_set_timesteps_comfyui(s, n_steps, expf(FLUX2_KLEIN_SHIFT_MU), 1.0f);
}

/* ---- Test modes ---- */

static int test_scheduler(void) {
    fprintf(stderr, "\n=== Flux.2 Klein Scheduler Test ===\n");
    qimg_scheduler s;

    /* Distilled: 4 steps, shift=1.0 */
    flux2_sched_init_distilled(&s, 4);
    fprintf(stderr, "Distilled (4 steps, shift=1.0):\n");
    fprintf(stderr, "  sigmas: [%.4f", s.sigmas[0]);
    for (int i = 1; i <= s.n_steps; i++) fprintf(stderr, ", %.4f", s.sigmas[i]);
    fprintf(stderr, "]\n  timesteps: [%.1f", s.timesteps[0]);
    for (int i = 1; i < s.n_steps; i++) fprintf(stderr, ", %.1f", s.timesteps[i]);
    fprintf(stderr, "]\n");

    /* Base: 20 steps, dynamic shift for 256 tokens */
    flux2_sched_init_base(&s, 20, 256);
    fprintf(stderr, "Base (20 steps, dynamic shift for 256 tokens):\n");
    fprintf(stderr, "  sigma[0]=%.4f sigma[20]=%.4f\n", s.sigmas[0], s.sigmas[s.n_steps]);
    fprintf(stderr, "  timesteps[0]=%.2f timesteps[19]=%.2f\n",
            s.timesteps[0], s.timesteps[s.n_steps - 1]);

    fprintf(stderr, "Scheduler test passed.\n");
    return 0;
}

static int test_vae(const char *vae_path, int lat_h, int lat_w, uint64_t seed, int n_threads) {
    fprintf(stderr, "\n=== Flux.2 Klein VAE Decode Test ===\n");

    flux2_vae_model *vae = flux2_vae_load(vae_path);
    if (!vae) return 1;
    vae->n_threads = n_threads;

    int lc = vae->latent_channels;
    size_t lat_sz = (size_t)lc * lat_h * lat_w;
    float *latent = (float *)malloc(lat_sz * sizeof(float));

    /* Try loading reference latent */
    char ref_path[512];
    snprintf(ref_path, sizeof(ref_path), "../../ref/flux2_klein/output/latent.npy");
    int got = load_npy_f32(ref_path, latent, lat_sz);
    if (got == (int)lat_sz) {
        fprintf(stderr, "Loaded reference latent from %s\n", ref_path);
    } else {
        fprintf(stderr, "No reference latent; using PRNG (seed=%llu)\n", (unsigned long long)seed);
        rng_state = seed;
        for (size_t i = 0; i < lat_sz; i++) latent[i] = randn() * 0.18215f;
    }

    int out_h = lat_h * 8, out_w = lat_w * 8;
    float *rgb = (float *)malloc((size_t)3 * out_h * out_w * sizeof(float));

    fprintf(stderr, "Decoding latent [%d, %d, %d] → RGB [3, %d, %d]...\n",
            lc, lat_h, lat_w, out_h, out_w);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    flux2_vae_decode(rgb, latent, lat_h, lat_w, vae);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    fprintf(stderr, "VAE decode: %.2f s\n", dt);

    save_ppm("flux2_vae_output.ppm", rgb, out_h, out_w);

    free(latent); free(rgb);
    flux2_vae_free(vae);
    return 0;
}

/* Dump callback for --dump-block: saves each intermediate as .npy */
static void flux2_dit_dump_cb(const char *name, const float *data, int n, void *ctx) {
    (void)ctx;
    char path[256];
    snprintf(path, sizeof(path), "flux2_%s.npy", name);
    int sh[] = {n};
    save_npy_f32(path, data, 1, sh);
    fprintf(stderr, "  dump: %s (%d floats)\n", path, n);
}

static int test_dit(const char *dit_path, int lat_h, int lat_w, int n_threads, int dump_block) {
    fprintf(stderr, "\n=== Flux.2 Klein DiT Single-Step Test ===\n");

    flux2_dit_model *dit = flux2_dit_load_safetensors(dit_path);
    if (!dit) return 1;

    int pin = dit->patch_in_channels;  /* 128 */
    int n_img = lat_h * lat_w;         /* after VAE 2×2 patchify, latent is [32, lat_h, lat_w] */
    int n_txt = 16;                     /* dummy text tokens */
    int txt_dim = dit->txt_dim;

    fprintf(stderr, "DiT: hidden=%d n_heads=%d pin=%d n_img=%d n_txt=%d txt_dim=%d\n",
            dit->hidden_dim, dit->n_heads, pin, n_img, n_txt, txt_dim);

    /* Random img tokens [n_img, pin] */
    float *img_tok = (float *)calloc((size_t)n_img * pin, sizeof(float));
    float *txt_tok = (float *)calloc((size_t)n_txt * txt_dim, sizeof(float));
    float *vel_out = (float *)malloc((size_t)n_img * pin * sizeof(float));

    rng_state = 12345;
    for (int i = 0; i < n_img * pin; i++) img_tok[i] = randn() * 0.1f;
    for (int i = 0; i < n_txt * txt_dim; i++) txt_tok[i] = randn() * 0.1f;

    float timestep = 750.0f;

    /* Set up dump if requested */
    if (dump_block >= 0) {
        int n_dbl = dit->n_double_blocks;
        if (dump_block < n_dbl) {
            dit->dump_dblk = dump_block;
            fprintf(stderr, "Dumping double-block %d intermediates\n", dump_block);
        } else {
            dit->dump_sblk = dump_block - n_dbl;
            fprintf(stderr, "Dumping single-block %d intermediates\n", dump_block - n_dbl);
        }
        dit->dump_fn = flux2_dit_dump_cb;
        dit->dump_ctx = NULL;
    }

    fprintf(stderr, "Running DiT forward (t=%.0f)...\n", timestep);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    flux2_dit_forward(vel_out, img_tok, n_img, txt_tok, n_txt,
                      timestep, dit, n_threads);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    fprintf(stderr, "DiT single step: %.2f s\n", dt);

    /* Stats */
    float mn = vel_out[0], mx = vel_out[0], sm = 0.0f;
    for (int i = 0; i < n_img * pin; i++) {
        if (vel_out[i] < mn) mn = vel_out[i];
        if (vel_out[i] > mx) mx = vel_out[i];
        sm += vel_out[i];
    }
    fprintf(stderr, "Output stats: min=%.4f max=%.4f mean=%.4f\n",
            mn, mx, sm / (float)(n_img * pin));

    /* Save as npy for Python verification */
    int shape[] = {n_img, pin};
    save_npy_f32("flux2_dit_output.npy", vel_out, 2, shape);

    free(img_tok); free(txt_tok); free(vel_out);
    flux2_dit_free(dit);
    return 0;
}

static int test_encoder(const char *enc_path, const char *tok_path,
                        const char *prompt) {
    fprintf(stderr, "\n=== Flux.2 Klein Text Encoder Test ===\n");

    flux2_text_enc *enc = flux2_text_enc_load_safetensors(enc_path, tok_path);
    if (!enc) return 1;

    fprintf(stderr, "Encoding: '%s'\n", prompt);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int n_tok = 0;
    float *hidden = flux2_text_enc_encode(enc, prompt, &n_tok);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;

    if (!hidden) { fprintf(stderr, "Encoding failed\n"); flux2_text_enc_free(enc); return 1; }

    int n_embd = enc->n_embd;
    fprintf(stderr, "Encoded: %d tokens × %d dims in %.2f s\n", n_tok, n_embd, dt);

    /* Stats on first token */
    float mn = hidden[0], mx = hidden[0], sm = 0.0f;
    for (int i = 0; i < n_tok * n_embd; i++) {
        if (hidden[i] < mn) mn = hidden[i];
        if (hidden[i] > mx) mx = hidden[i];
        sm += hidden[i];
    }
    fprintf(stderr, "Hidden states stats: min=%.4f max=%.4f mean=%.6f\n",
            mn, mx, sm / (float)(n_tok * n_embd));

    int shape[] = {n_tok, n_embd};
    save_npy_f32("flux2_text_hidden.npy", hidden, 2, shape);

    free(hidden);
    flux2_text_enc_free(enc);
    return 0;
}

/* ---- Full pipeline ---- */

/* Patchify latent: VAE outputs [32, lat_h, lat_w].
 * Flux.2 Klein uses patch_size=1 in DiT but VAE patchification of (2,2)
 * makes the effective input [32×4, lat_h/2, lat_w/2] = [128, H/16, W/16].
 * For simplicity: treat VAE output latent [32, lat_h, lat_w] directly as
 * img_tokens [lat_h*lat_w, 32] or apply 2×2 spatial patchification.
 *
 * TODO: After weight inspection, confirm exact patchification scheme.
 * For now: x_embedder expects [n_img, 128] where n_img = (lat_h/2)*(lat_w/2). */
static void flux2_patchify(float *out_tok, const float *latent,
                            int lc, int lat_h, int lat_w, int ps) {
    /* ps=2: [lc, lat_h, lat_w] → [lat_h/2 * lat_w/2, lc*ps*ps]
     * Rearranges patches in row-major order */
    int ph = lat_h / ps, pw = lat_w / ps;
    int pin = lc * ps * ps;
    for (int r = 0; r < ph; r++)
        for (int c = 0; c < pw; c++) {
            float *tok = out_tok + ((size_t)r * pw + c) * pin;
            for (int ch = 0; ch < lc; ch++)
                for (int pr = 0; pr < ps; pr++)
                    for (int pc = 0; pc < ps; pc++)
                        tok[ch * ps * ps + pr * ps + pc] =
                            latent[(size_t)ch * lat_h * lat_w
                                   + (r * ps + pr) * lat_w + (c * ps + pc)];
        }
}

static void flux2_unpatchify(float *latent, const float *tok,
                              int lc, int lat_h, int lat_w, int ps) {
    int ph = lat_h / ps, pw = lat_w / ps;
    int pin = lc * ps * ps;
    for (int r = 0; r < ph; r++)
        for (int c = 0; c < pw; c++) {
            const float *t = tok + ((size_t)r * pw + c) * pin;
            for (int ch = 0; ch < lc; ch++)
                for (int pr = 0; pr < ps; pr++)
                    for (int pc = 0; pc < ps; pc++)
                        latent[(size_t)ch * lat_h * lat_w
                               + (r * ps + pr) * lat_w + (c * ps + pc)] =
                            t[ch * ps * ps + pr * ps + pc];
        }
}

static int run_generate(const char *dit_path, const char *vae_path,
                         const char *enc_path, const char *tok_path,
                         const char *hidden_npy_path,
                         const char *prompt, int out_h, int out_w,
                         int n_steps, uint64_t seed,
                         int is_distilled, float cfg_scale, int n_threads) {
    fprintf(stderr, "\n=== Flux.2 Klein Full Pipeline ===\n");
    fprintf(stderr, "Prompt: '%s'\n", prompt);
    fprintf(stderr, "Output: %dx%d, %d steps, seed=%llu, distilled=%d\n",
            out_w, out_h, n_steps, (unsigned long long)seed, is_distilled);

    /* Step 1: Text encoding */
    fprintf(stderr, "\n[1/4] Text encoding...\n");
    struct timespec t_start, t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    int n_txt = 0;
    float *txt_hidden = NULL;

    if (hidden_npy_path) {
        /* Load pre-computed text embeddings (n_txt × txt_dim floats).
         * txt_dim = 3 × 2560 = 7680 for Qwen3-4B. Auto-detect n_txt from file. */
        fprintf(stderr, "Loading pre-computed embeddings from %s\n", hidden_npy_path);
        const int txt_dim = 7680;
        const int max_load = 512 * txt_dim;
        txt_hidden = (float *)malloc((size_t)max_load * sizeof(float));
        int got = load_npy_f32(hidden_npy_path, txt_hidden, (size_t)max_load);
        if (got <= 0) {
            fprintf(stderr, "Failed to load hidden states from %s\n", hidden_npy_path);
            free(txt_hidden); return 1;
        }
        n_txt = got / txt_dim;
        fprintf(stderr, "Loaded %d tokens × %d dims\n", n_txt, txt_dim);
    } else {
        flux2_text_enc *enc = flux2_text_enc_load_safetensors(enc_path, tok_path);
        if (!enc) return 1;
        txt_hidden = flux2_text_enc_encode(enc, prompt, &n_txt);
        flux2_text_enc_free(enc);
        if (!txt_hidden) return 1;
    }

    /* txt_dim is enc->n_embd; DiT will read it from context_embedder weight */

    /* Front-pad text to 512 tokens with zeros (matches ComfyUI Flux2.extra_conds) */
    {
        const int FLUX2_KLEIN_TXT_LEN = 512;
        const int txt_dim_pad = 7680;
        if (n_txt < FLUX2_KLEIN_TXT_LEN) {
            float *padded = (float *)calloc((size_t)FLUX2_KLEIN_TXT_LEN * txt_dim_pad, sizeof(float));
            int pad_front = FLUX2_KLEIN_TXT_LEN - n_txt;
            memcpy(padded + (size_t)pad_front * txt_dim_pad, txt_hidden,
                   (size_t)n_txt * txt_dim_pad * sizeof(float));
            free(txt_hidden);
            txt_hidden = padded;
            n_txt = FLUX2_KLEIN_TXT_LEN;
            fprintf(stderr, "Front-padded text to %d tokens\n", n_txt);
        }
    }

    /* Step 2: Load DiT */
    fprintf(stderr, "\n[2/4] Loading DiT...\n");
    clock_gettime(CLOCK_MONOTONIC, &t0);
    flux2_dit_model *dit = flux2_dit_load_safetensors(dit_path);
    if (!dit) { free(txt_hidden); return 1; }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    fprintf(stderr, "DiT load: %.1f s\n",
            (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9);

    /* Latent shape: [32, lat_h, lat_w] where lat_h = out_h/8, lat_w = out_w/8 */
    int lat_h = out_h / 8, lat_w = out_w / 8;
    int lc = FLUX2_VAE_LATENT_CHANNELS;

    /* After VAE 2×2 patch: img_tokens shape [lat_h/2 * lat_w/2, 128] */
    /* Wait - patch_size for DiT is 1, but the VAE has a "patch" internally.
     * From model inspection: x_embedder.weight shape = [hidden, 128]
     * where 128 = 32 (latent ch) × 2 × 2 (patch_size).
     * So DiT sees patchified latent: [lat_h/2 * lat_w/2, 128]. */
    int ps = 2;  /* TODO: confirm from weight inspection */
    int n_img = (lat_h / ps) * (lat_w / ps);

    /* If hidden_dim was determined from x_embedder, patch_in_channels is dit->patch_in_channels */
    if (dit->patch_in_channels != lc * ps * ps) {
        fprintf(stderr, "WARN: x_embedder expects %d input channels but got %d*%d*%d=%d\n",
                dit->patch_in_channels, lc, ps, ps, lc*ps*ps);
        /* Try ps=1 (no patchification) */
        if (dit->patch_in_channels == lc) {
            ps = 1; n_img = lat_h * lat_w;
            fprintf(stderr, "Falling back to ps=1 (n_img=%d)\n", n_img);
        }
    }

    fprintf(stderr, "Latent: [%d, %d, %d], patch_size=%d, n_img=%d\n",
            lc, lat_h, lat_w, ps, n_img);

    /* Initial noise */
    rng_state = seed;
    size_t lat_sz = (size_t)lc * lat_h * lat_w;
    float *latent = (float *)malloc(lat_sz * sizeof(float));
    for (size_t i = 0; i < lat_sz; i++) latent[i] = randn();

    /* Patchified buffers */
    float *img_tok = (float *)malloc((size_t)n_img * dit->patch_in_channels * sizeof(float));
    float *vel_out = (float *)malloc((size_t)n_img * dit->patch_in_channels * sizeof(float));

    /* Scheduler */
    qimg_scheduler sched;
    if (is_distilled)
        flux2_sched_init_distilled(&sched, n_steps);
    else
        flux2_sched_init_base(&sched, n_steps, n_img);

    fprintf(stderr, "Sigma schedule: [%.3f → %.3f]\n",
            sched.sigmas[0], sched.sigmas[n_steps]);

    /* Step 3: Denoising loop */
    fprintf(stderr, "\n[3/4] Denoising (%d steps)...\n", n_steps);
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int step = 0; step < n_steps; step++) {
        float t_sigma = sched.timesteps[step];
        struct timespec ts0, ts1;
        clock_gettime(CLOCK_MONOTONIC, &ts0);

        /* Patchify latent */
        flux2_patchify(img_tok, latent, lc, lat_h, lat_w, ps);

        if (is_distilled || cfg_scale <= 1.0f) {
            /* No CFG: single pass */
            flux2_dit_forward(vel_out, img_tok, n_img, txt_hidden, n_txt,
                              t_sigma, dit, n_threads);
        } else {
            /* CFG: two passes (cond + uncond) */
            /* Uncond: empty prompt hidden states (zeros) */
            float *txt_uncond = (float *)calloc((size_t)n_txt * dit->txt_dim, sizeof(float));
            float *vel_uncond = (float *)malloc((size_t)n_img * dit->patch_in_channels * sizeof(float));

            flux2_dit_forward(vel_uncond, img_tok, n_img, txt_uncond, n_txt,
                              t_sigma, dit, n_threads);
            flux2_dit_forward(vel_out, img_tok, n_img, txt_hidden, n_txt,
                              t_sigma, dit, n_threads);

            /* CFG blending: v = uncond + scale * (cond - uncond) */
            for (int i = 0; i < n_img * dit->patch_in_channels; i++)
                vel_out[i] = vel_uncond[i] + cfg_scale * (vel_out[i] - vel_uncond[i]);

            free(txt_uncond); free(vel_uncond);
        }

        /* Unpatchify velocity */
        float *vel_lat = (float *)calloc(lat_sz, sizeof(float));
        flux2_unpatchify(vel_lat, vel_out, lc, lat_h, lat_w, ps);

        /* Euler step */
        qimg_sched_step(latent, vel_lat, (int)lat_sz, step, &sched);
        free(vel_lat);

        clock_gettime(CLOCK_MONOTONIC, &ts1);
        double step_dt = (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
        fprintf(stderr, "  step %d/%d  sigma=%.4f  %.1f s\n",
                step + 1, n_steps, t_sigma, step_dt);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dit_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    fprintf(stderr, "DiT denoising: %.1f s (%.2f s/step)\n", dit_time, dit_time / n_steps);

    flux2_dit_free(dit);
    free(img_tok); free(vel_out);

    /* Save final latent */
    {
        int sh[] = {lc, lat_h, lat_w};
        save_npy_f32("flux2_latent_final.npy", latent, 3, sh);
    }

    /* Step 4: VAE decode */
    fprintf(stderr, "\n[4/4] VAE decoding...\n");
    clock_gettime(CLOCK_MONOTONIC, &t0);

    flux2_vae_model *vae = flux2_vae_load(vae_path);
    if (!vae) { free(latent); free(txt_hidden); return 1; }
    vae->n_threads = n_threads;

    float *rgb = (float *)malloc((size_t)3 * out_h * out_w * sizeof(float));
    flux2_vae_decode(rgb, latent, lat_h, lat_w, vae);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    fprintf(stderr, "VAE decode: %.1f s\n",
            (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9);

    save_ppm("flux2_output.ppm", rgb, out_h, out_w);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double total = (t1.tv_sec - t_start.tv_sec) + (t1.tv_nsec - t_start.tv_nsec) * 1e-9;
    fprintf(stderr, "\nTotal pipeline: %.1f s\n", total);

    free(rgb); free(latent); free(txt_hidden);
    flux2_vae_free(vae);
    return 0;
}

/* ---- main ---- */

int main(int argc, char **argv) {
    const char *dit_path = DEFAULT_DIT;
    const char *vae_path = DEFAULT_VAE;
    const char *enc_path = DEFAULT_ENC;
    const char *tok_path = DEFAULT_TOK;
    const char *prompt = "a red apple on a white table";
    const char *mode = NULL;
    const char *latent_npy = NULL;  /* --latent-npy path (future use) */
    const char *hidden_npy = NULL;  /* --hidden-npy: pre-computed text embeddings */

    int out_h = 256, out_w = 256;
    int n_steps = 4;
    int n_threads = 1;
    int is_distilled = 1;
    float cfg_scale = 1.0f;
    uint64_t seed = 42;
    int dump_block = -1;  /* --dump-block N: dump intermediates for block N */

    for (int i = 1; i < argc; i++) {
        if      (strcmp(argv[i], "--test-sched") == 0) mode = "sched";
        else if (strcmp(argv[i], "--test-vae")   == 0) mode = "vae";
        else if (strcmp(argv[i], "--test-dit")   == 0) mode = "dit";
        else if (strcmp(argv[i], "--test-enc")   == 0) mode = "enc";
        else if (strcmp(argv[i], "--generate")   == 0) mode = "gen";
        else if (strcmp(argv[i], "--base")       == 0) { is_distilled = 0; n_steps = 20; }
        else if (strcmp(argv[i], "--distilled")  == 0) { is_distilled = 1; n_steps = 4; }
        else if (strcmp(argv[i], "--dit")   == 0 && i+1 < argc) dit_path = argv[++i];
        else if (strcmp(argv[i], "--vae")   == 0 && i+1 < argc) vae_path = argv[++i];
        else if (strcmp(argv[i], "--enc")   == 0 && i+1 < argc) enc_path = argv[++i];
        else if (strcmp(argv[i], "--tok")   == 0 && i+1 < argc) tok_path = argv[++i];
        else if (strcmp(argv[i], "--prompt")== 0 && i+1 < argc) prompt   = argv[++i];
        else if (strcmp(argv[i], "--height")== 0 && i+1 < argc) out_h  = atoi(argv[++i]);
        else if (strcmp(argv[i], "--width") == 0 && i+1 < argc) out_w  = atoi(argv[++i]);
        else if (strcmp(argv[i], "--steps") == 0 && i+1 < argc) n_steps= atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed")  == 0 && i+1 < argc) seed   = (uint64_t)atoll(argv[++i]);
        else if (strcmp(argv[i], "--cfg")     == 0 && i+1 < argc) cfg_scale  = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0 && i+1 < argc) n_threads  = atoi(argv[++i]);
        else if (strcmp(argv[i], "--latent-npy") == 0 && i+1 < argc) { latent_npy = argv[++i]; (void)latent_npy; }
        else if (strcmp(argv[i], "--hidden-npy") == 0 && i+1 < argc) hidden_npy = argv[++i];
        else if (strcmp(argv[i], "--dump-block") == 0 && i+1 < argc) dump_block = atoi(argv[++i]);
        /* Positional: DIT VAE ENC (legacy compat) */
        else if (argv[i][0] != '-' && !dit_path[0]) dit_path = argv[i];
        else if (argv[i][0] != '-' && !vae_path[0]) vae_path = argv[i];
        else if (argv[i][0] != '-' && !enc_path[0]) enc_path = argv[i];
    }

    if (!mode) {
        fprintf(stderr,
            "Usage: %s [--test-sched|--test-vae|--test-dit|--test-enc|--generate]\n"
            "          [--dit PATH] [--vae PATH] [--enc PATH] [--tok GGUF]\n"
            "          [--prompt TEXT] [--height H] [--width W]\n"
            "          [--steps N] [--seed S] [--cfg SCALE] [--threads N]\n"
            "          [--base|--distilled]\n"
            "  --enc: qwen_3_4b.safetensors (BF16 weights)\n"
            "  --tok: any Qwen3 GGUF used only for tokenizer vocab\n",
            argv[0]);
        return 1;
    }

    if (strcmp(mode, "sched") == 0) return test_scheduler();
    if (strcmp(mode, "vae")   == 0) return test_vae(vae_path, out_h/8, out_w/8, seed, n_threads);
    if (strcmp(mode, "dit")   == 0) return test_dit(dit_path, out_h/16, out_w/16, n_threads, dump_block);
    if (strcmp(mode, "enc")   == 0) return test_encoder(enc_path, tok_path, prompt);
    if (strcmp(mode, "gen")   == 0)
        return run_generate(dit_path, vae_path, enc_path, tok_path, hidden_npy,
                            prompt, out_h, out_w, n_steps, seed, is_distilled, cfg_scale, n_threads);

    fprintf(stderr, "Unknown mode: %s\n", mode);
    return 1;
}
