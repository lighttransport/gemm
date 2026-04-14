/*
 * test_hip_flux2.c - Flux.2 Klein 4B end-to-end text-to-image test (HIP/RDNA4)
 *
 * Modes:
 *   --test-dit   : single DiT step with random inputs (diagnostics)
 *   --generate   : full denoising pipeline (scheduler + VAE + PPM output)
 *   (default)    : existing N-step denoising with dummy text (legacy)
 *
 * Build:
 *   make
 *
 * Generate usage:
 *   ./test_hip_flux2 --generate \
 *       --dit <dit.safetensors> --vae <vae.safetensors> \
 *       [--enc <text_encoder_dir_or_st> --tok <tok.gguf>] \
 *       [--txt-bin <raw F32 [n_tok,7680]>] \
 *       [--prompt "..."] [--steps 4] [--size 256] [--seed 42] \
 *       [-o out.ppm] [-d device_id] [-v]
 *
 * If neither --enc nor --txt-bin is supplied, a seeded pseudo-random text
 * hidden tensor is used (for pipeline shakeout only — output will not be a
 * recognizable image).
 */

#define SAFETENSORS_IMPLEMENTATION
#define GGML_DEQUANT_IMPLEMENTATION
#define QIMG_SCHEDULER_IMPLEMENTATION
#define FLUX2_TEXT_ENCODER_IMPLEMENTATION
#define TRANSFORMER_IMPLEMENTATION
#define BPE_TOKENIZER_IMPLEMENTATION
#define GGUF_LOADER_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "../../common/safetensors.h"
#include "../../common/ggml_dequant.h"
#include "../../common/qwen_image_scheduler.h"
#include "../../common/flux2_klein_text_encoder.h"
#include "../../common/stb_image_write.h"
#include "hip_flux2_runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>

/* ---- PRNG: Box-Muller ---- */
static uint64_t rng_state = 42;
static int rng_cached_valid = 0;
static float rng_cached = 0.0f;

static void rng_seed(uint64_t s) {
    rng_state = s ? s : 42;
    rng_cached_valid = 0;
}

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

/* CHW float [-1,1] -> HWC uint8 [0,255] */
static unsigned char *rgb_to_u8(const float *rgb, int h, int w) {
    unsigned char *u8 = (unsigned char *)malloc((size_t)h * w * 3);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            for (int c = 0; c < 3; c++) {
                float v = rgb[c * h * w + y * w + x];
                v = 0.5f * v + 0.5f;
                v = v < 0 ? 0 : (v > 1 ? 1 : v);
                u8[(y * w + x) * 3 + c] = (unsigned char)(v * 255.0f + 0.5f);
            }
    return u8;
}

static int ends_with_icase(const char *s, const char *suf) {
    size_t ls = strlen(s), lf = strlen(suf);
    if (lf > ls) return 0;
    for (size_t i = 0; i < lf; i++) {
        char a = s[ls - lf + i], b = suf[i];
        if (a >= 'A' && a <= 'Z') a += 32;
        if (b >= 'A' && b <= 'Z') b += 32;
        if (a != b) return 0;
    }
    return 1;
}

static void save_image(const char *path, const float *rgb, int h, int w) {
    unsigned char *u8 = rgb_to_u8(rgb, h, w);
    int ok = 0;
    if (ends_with_icase(path, ".png")) {
        ok = stbi_write_png(path, w, h, 3, u8, w * 3);
    } else if (ends_with_icase(path, ".jpg") || ends_with_icase(path, ".jpeg")) {
        ok = stbi_write_jpg(path, w, h, 3, u8, 95);
    } else {
        /* Default: PPM via stdio */
        FILE *fp = fopen(path, "wb");
        if (fp) {
            fprintf(fp, "P6\n%d %d\n255\n", w, h);
            fwrite(u8, 1, (size_t)h * w * 3, fp);
            fclose(fp);
            ok = 1;
        }
    }
    free(u8);
    if (ok) fprintf(stderr, "Saved %s (%dx%d)\n", path, w, h);
    else    fprintf(stderr, "Failed to write %s\n", path);
}

/* Back-compat alias */
static void save_ppm(const char *path, const float *rgb, int h, int w) {
    save_image(path, rgb, h, w);
}

static void patchify(const float *latent, float *img_tok, int lat_h, int lat_w) {
    const int ps = 2, C = 32, pin = C * ps * ps;
    int ph = lat_h / ps, pw = lat_w / ps;
    for (int py = 0; py < ph; py++)
        for (int px = 0; px < pw; px++) {
            int pidx = py * pw + px;
            for (int c = 0; c < C; c++)
                for (int dy = 0; dy < ps; dy++)
                    for (int dx = 0; dx < ps; dx++) {
                        int y = py * ps + dy, x = px * ps + dx;
                        int dst = pidx * pin + c * ps * ps + dy * ps + dx;
                        img_tok[dst] = latent[c * lat_h * lat_w + y * lat_w + x];
                    }
        }
}

static void unpatchify(const float *img_tok, float *latent, int lat_h, int lat_w) {
    const int ps = 2, C = 32, pin = C * ps * ps;
    int ph = lat_h / ps, pw = lat_w / ps;
    for (int py = 0; py < ph; py++)
        for (int px = 0; px < pw; px++) {
            int pidx = py * pw + px;
            for (int c = 0; c < C; c++)
                for (int dy = 0; dy < ps; dy++)
                    for (int dx = 0; dx < ps; dx++) {
                        int y = py * ps + dy, x = px * ps + dx;
                        int src = pidx * pin + c * ps * ps + dy * ps + dx;
                        latent[c * lat_h * lat_w + y * lat_w + x] = img_tok[src];
                    }
        }
}

/* Load a raw F32 binary of known element count. Fails if size mismatches. */
static float *load_f32_bin(const char *path, size_t n_elems) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", path); return NULL; }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    size_t want = n_elems * sizeof(float);
    if ((size_t)sz != want) {
        fprintf(stderr, "%s: size %ld != expected %zu (%zu elems)\n", path, sz, want, n_elems);
        fclose(fp); return NULL;
    }
    float *buf = (float *)malloc(want);
    if (fread(buf, 1, want, fp) != want) { free(buf); fclose(fp); return NULL; }
    fclose(fp);
    return buf;
}

/* Load text hidden states from a raw F32 file (shape inferred from size given txt_dim). */
static float *load_txt_bin(const char *path, int *n_tok, int txt_dim) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", path); return NULL; }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz <= 0 || sz % (txt_dim * (int)sizeof(float)) != 0) {
        fprintf(stderr, "txt-bin %s size %ld not multiple of %d*%zu\n",
                path, sz, txt_dim, sizeof(float));
        fclose(fp);
        return NULL;
    }
    int n = (int)(sz / (txt_dim * (int)sizeof(float)));
    float *buf = (float *)malloc((size_t)n * txt_dim * sizeof(float));
    if (fread(buf, 1, sz, fp) != (size_t)sz) { free(buf); fclose(fp); return NULL; }
    fclose(fp);
    *n_tok = n;
    fprintf(stderr, "Loaded %s: %d tokens × %d dim\n", path, n, txt_dim);
    return buf;
}

/* Front-pad text to N tokens (ComfyUI convention: zeros in front, real at back). */
static float *front_pad_text(const float *src, int n_src, int txt_dim, int n_pad) {
    if (n_src >= n_pad) return (float *)src;
    float *dst = (float *)calloc((size_t)n_pad * txt_dim, sizeof(float));
    memcpy(dst + (size_t)(n_pad - n_src) * txt_dim, src,
           (size_t)n_src * txt_dim * sizeof(float));
    return dst;
}

static int path_is_dir(const char *p) {
    struct stat sb;
    return (stat(p, &sb) == 0 && S_ISDIR(sb.st_mode));
}

/* diffusers' compute_empirical_mu for Flux.2 Klein.
 * Port of diffusers/pipelines/flux2/pipeline_flux2_klein.py::compute_empirical_mu. */
static float flux2_compute_mu(int image_seq_len, int num_steps) {
    const float a1 = 8.73809524e-05f, b1 = 1.89833333f;
    const float a2 = 0.00016927f,     b2 = 0.45666666f;
    if (image_seq_len > 4300) return a2 * image_seq_len + b2;
    float m_200 = a2 * image_seq_len + b2;
    float m_10  = a1 * image_seq_len + b1;
    float a = (m_200 - m_10) / 190.0f;
    float b = m_200 - 200.0f * a;
    return a * num_steps + b;
}

/* Build sigmas with FlowMatchEulerDiscreteScheduler + exponential dynamic shift.
 * Matches diffusers Flux2KleinPipeline exactly:
 *   sigmas = linspace(1.0, 1/N, N)
 *   sigmas = exp(mu) / (exp(mu) + (1/t - 1))   with sigma=1.0 in the formula
 *   append 0.0
 * out must hold n_steps+1 floats. */
static void flux2_flow_sigmas(float *out, int n_steps, int image_seq_len) {
    float mu = flux2_compute_mu(image_seq_len, n_steps);
    float em = expf(mu);
    for (int i = 0; i < n_steps; i++) {
        float t = 1.0f + ((float)i * (1.0f / (float)n_steps - 1.0f) / (float)(n_steps - 1));
        /* linspace(1.0, 1/N, N): t_i = 1 + i*(1/N - 1)/(N-1); for N=4 -> [1.0, 0.75, 0.5, 0.25] */
        if (n_steps == 1) t = 1.0f;
        out[i] = em / (em + (1.0f / t - 1.0f));
    }
    out[n_steps] = 0.0f;
}

/* Unpack a [n_patches, 128] F32 packed latent (the same layout diffusers feeds to DiT)
 * back into [32, lat_h, lat_w] standard CHW latent. Applies BN de-normalization per
 * the diffusers pipeline: packed[c_packed, ph, pw] = packed * sqrt(var+eps) + mean,
 * where bn has 128 channels ordered as [c*4 + pr*2 + pc]. */
static void unpack_latent_for_vae(const float *packed,
                                   float *chw32,
                                   int lat_h, int lat_w,
                                   const float *bn_mean, const float *bn_var, float bn_eps) {
    const int C = 32, ps = 2;
    int ph = lat_h / ps, pw = lat_w / ps;
    /* packed is [ph*pw, C*ps*ps]. Each token row has 128 channels ordered:
     * c*(ps*ps) + pr*ps + pc. */
    for (int py = 0; py < ph; py++)
        for (int px = 0; px < pw; px++) {
            int pidx = py * pw + px;
            const float *row = packed + (size_t)pidx * (C * ps * ps);
            for (int c = 0; c < C; c++)
                for (int pr = 0; pr < ps; pr++)
                    for (int pc = 0; pc < ps; pc++) {
                        int bn_ch = c * (ps * ps) + pr * ps + pc;
                        float v = row[bn_ch];
                        if (bn_mean && bn_var) {
                            float sd = sqrtf(bn_var[bn_ch] + bn_eps);
                            v = v * sd + bn_mean[bn_ch];
                        }
                        int y = py * ps + pr, x = px * ps + pc;
                        chw32[c * lat_h * lat_w + y * lat_w + x] = v;
                    }
        }
}

/* ---- VAE-only mode: feed diffusers' final_latent_packed.bin through HIP VAE ---- */

static int run_vae_only(hip_flux2_runner *r,
                        const char *vae_path, const char *latent_packed_bin,
                        const char *bn_mean_bin, const char *bn_var_bin, float bn_eps,
                        int img_size, const char *out_path,
                        const char *dump_rgb_path) {
    const int ps = 2, C = 32;
    int lat_h = img_size / 8, lat_w = img_size / 8;
    int ph = lat_h / ps, pw = lat_w / ps;
    int n_patches = ph * pw;
    int pin = C * ps * ps;

    if (hip_flux2_load_vae(r, vae_path) < 0) { fprintf(stderr, "VAE load fail\n"); return -1; }

    float *packed = load_f32_bin(latent_packed_bin, (size_t)n_patches * pin);
    if (!packed) return -1;
    fprintf(stderr, "vae-only: loaded %s (%d patches × %d)\n",
            latent_packed_bin, n_patches, pin);

    /* Optional BN stats from diffusers dump. Use these to apply the same
     * denormalization the diffusers pipeline does on its final latent. */
    float *bn_mean = NULL, *bn_var = NULL;
    if (bn_mean_bin && bn_var_bin) {
        bn_mean = load_f32_bin(bn_mean_bin, 128);
        bn_var  = load_f32_bin(bn_var_bin,  128);
        if (!bn_mean || !bn_var) {
            fprintf(stderr, "BN stats load failed\n");
            free(packed); free(bn_mean); free(bn_var); return -1;
        }
        fprintf(stderr, "vae-only: BN denorm enabled (eps=%g)\n", bn_eps);
    }

    float *latent = (float *)calloc((size_t)C * lat_h * lat_w, sizeof(float));
    unpack_latent_for_vae(packed, latent, lat_h, lat_w, bn_mean, bn_var, bn_eps);
    free(bn_mean); free(bn_var); free(packed);

    /* diagnostic stats */
    {
        double mn = 0, sq = 0;
        size_t n = (size_t)C * lat_h * lat_w;
        for (size_t i = 0; i < n; i++) { mn += latent[i]; sq += (double)latent[i] * latent[i]; }
        double mean = mn / n, var = sq / n - mean * mean;
        fprintf(stderr, "vae-only: unpacked latent mean=%.4f std=%.4f\n", mean, sqrt(var));
    }

    float *rgb = (float *)calloc((size_t)3 * img_size * img_size, sizeof(float));
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    /* Tell the HIP VAE decode NOT to re-apply BN (we did it in unpack above).
     * The existing runner applies BN if m->bn_mean && m->bn_var are set. Hack:
     * temporarily null them out via environment var handled by runner? Not yet.
     * Simpler: leave runner BN alone (it's disabled when bn_mean==NULL; and when
     * we load ref flux2-vae.safetensors it may or may not have BN. For now call
     * hip_flux2_vae_decode anyway and accept duplicated BN only if both paths
     * have valid stats — we disable via env FLUX2_SKIP_VAE_BN=1 below). */
    setenv("FLUX2_SKIP_VAE_BN", "1", 1);
    int rc = hip_flux2_vae_decode(r, latent, lat_h, lat_w, rgb);
    unsetenv("FLUX2_SKIP_VAE_BN");
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double vae_t = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    fprintf(stderr, "vae-only: VAE decode %.2f s (rc=%d)\n", vae_t, rc);

    if (dump_rgb_path) {
        FILE *fp = fopen(dump_rgb_path, "wb");
        if (fp) {
            fwrite(rgb, sizeof(float), (size_t)3 * img_size * img_size, fp);
            fclose(fp);
            fprintf(stderr, "vae-only: dumped raw RGB F32 to %s\n", dump_rgb_path);
        }
    }
    save_image(out_path, rgb, img_size, img_size);

    free(rgb); free(latent);
    return rc;
}

/* ---- Generate mode ---- */

static int run_generate(hip_flux2_runner *r,
                         const char *prompt, const char *enc_path, const char *tok_path,
                         const char *txt_bin_path,
                         const char *init_bin_path, const char *sigmas_bin_path,
                         const char *dump_final_path,
                         const char *bn_mean_bin, const char *bn_var_bin, float bn_eps,
                         const char *vae_path, const char *out_path,
                         int img_size, int n_steps, uint64_t seed) {
    const int ps = 2, C = 32, pin = C * ps * ps;
    int lat_h = img_size / 8, lat_w = img_size / 8;
    int n_img = (lat_h / ps) * (lat_w / ps);
    int txt_dim = 7680;

    if (hip_flux2_load_vae(r, vae_path) < 0) {
        fprintf(stderr, "generate: VAE load failed\n");
        return -1;
    }

    /* ---- Text encoding ---- */
    float *txt_hidden = NULL;
    int n_txt_real = 0;
    if (txt_bin_path) {
        txt_hidden = load_txt_bin(txt_bin_path, &n_txt_real, txt_dim);
        if (!txt_hidden) return -1;
    } else if (enc_path && tok_path) {
        fprintf(stderr, "generate: loading CPU text encoder from %s\n", enc_path);
        flux2_text_enc *enc = flux2_text_enc_load_safetensors(enc_path, tok_path);
        if (!enc) {
            fprintf(stderr, "generate: text encoder load FAILED\n");
            return -1;
        }
        fprintf(stderr, "generate: encoding \"%s\"\n", prompt);
        txt_hidden = flux2_text_enc_encode(enc, prompt, &n_txt_real);
        flux2_text_enc_free(enc);
        if (!txt_hidden) {
            fprintf(stderr, "generate: text encode FAILED\n");
            return -1;
        }
        fprintf(stderr, "generate: encoded %d real tokens × %d dim\n", n_txt_real, txt_dim);
    } else {
        fprintf(stderr, "generate: WARNING no --enc/--tok or --txt-bin supplied. "
                        "Using seeded random text hidden (output will not be a real image).\n");
        n_txt_real = 16;
        txt_hidden = (float *)calloc((size_t)n_txt_real * txt_dim, sizeof(float));
        rng_seed(seed ^ 0xBEEFCAFEULL);
        for (int i = 0; i < n_txt_real * txt_dim; i++) txt_hidden[i] = 0.01f * randn();
    }

    /* Front-pad text to 512 tokens (ComfyUI convention). */
    const int N_TXT = 512;
    float *txt_padded = front_pad_text(txt_hidden, n_txt_real, txt_dim, N_TXT);
    int n_txt = N_TXT;

    /* ---- Sigmas (pinned from --sigmas-bin, else diffusers Flux2 dynamic shift) ---- */
    float *sigmas = (float *)calloc((size_t)(n_steps + 1), sizeof(float));
    if (sigmas_bin_path) {
        float *loaded = load_f32_bin(sigmas_bin_path, (size_t)(n_steps + 1));
        if (!loaded) { free(sigmas); return -1; }
        memcpy(sigmas, loaded, (size_t)(n_steps + 1) * sizeof(float));
        free(loaded);
        fprintf(stderr, "generate: sigmas (pinned) = ");
    } else {
        /* diffusers Flux2KleinPipeline scheduler: dynamic exponential shift
         * mu = empirical fn of (image_seq_len=n_img, num_steps). */
        flux2_flow_sigmas(sigmas, n_steps, n_img);
        fprintf(stderr, "generate: sigmas (flux2 dynamic, mu=%.4f) = ",
                flux2_compute_mu(n_img, n_steps));
    }
    for (int i = 0; i <= n_steps; i++) fprintf(stderr, "%.4f ", sigmas[i]);
    fprintf(stderr, "\n");

    /* ---- Initial packed latent [n_img, pin] ----
     * We work in packed space throughout the denoising loop (matches diffusers).
     * At the end, unpack to [C, lat_h, lat_w] for VAE decode. */
    float *img_tok  = (float *)malloc((size_t)n_img * pin * sizeof(float));
    float *velocity = (float *)malloc((size_t)n_img * pin * sizeof(float));

    if (init_bin_path) {
        float *loaded = load_f32_bin(init_bin_path, (size_t)n_img * pin);
        if (!loaded) {
            free(sigmas); free(img_tok); free(velocity);
            if (txt_padded != txt_hidden) free(txt_padded);
            free(txt_hidden);
            return -1;
        }
        memcpy(img_tok, loaded, (size_t)n_img * pin * sizeof(float));
        free(loaded);
        fprintf(stderr, "generate: loaded pinned init latent from %s\n", init_bin_path);
    } else {
        /* Box-Muller randn into packed layout. */
        rng_seed(seed);
        for (int i = 0; i < n_img * pin; i++) img_tok[i] = randn();
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int step = 0; step < n_steps; step++) {
        float sigma = sigmas[step];
        float dt    = sigmas[step + 1] - sigmas[step];
        if (hip_flux2_dit_step(r, img_tok, n_img, txt_padded, n_txt,
                                sigma, 0.0f, velocity) != 0) {
            fprintf(stderr, "generate: dit step %d failed\n", step);
            break;
        }

        /* Euler step directly in packed space: img_tok += dt * velocity */
        for (int i = 0; i < n_img * pin; i++) img_tok[i] += dt * velocity[i];

        fprintf(stderr, "  step %d/%d (sigma=%.4f dt=%.4f)\n",
                step + 1, n_steps, sigma, dt);
    }

    if (dump_final_path) {
        FILE *fp = fopen(dump_final_path, "wb");
        if (fp) {
            fwrite(img_tok, sizeof(float), (size_t)n_img * pin, fp);
            fclose(fp);
            fprintf(stderr, "generate: dumped final packed latent to %s\n", dump_final_path);
        } else {
            fprintf(stderr, "generate: failed to open %s for writing\n", dump_final_path);
        }
    }

    /* Unpack once (with optional BN denorm from diffusers stats) for VAE. */
    size_t lat_n = (size_t)C * lat_h * lat_w;
    float *latent = (float *)calloc(lat_n, sizeof(float));
    float *bnm = NULL, *bnv = NULL;
    if (bn_mean_bin && bn_var_bin) {
        bnm = load_f32_bin(bn_mean_bin, 128);
        bnv = load_f32_bin(bn_var_bin, 128);
    }
    unpack_latent_for_vae(img_tok, latent, lat_h, lat_w, bnm, bnv, bn_eps);
    free(bnm); free(bnv);
    free(sigmas);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dit_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    fprintf(stderr, "generate: DiT total %.2f s (%.2f s/step)\n",
            dit_time, dit_time / n_steps);

    /* ---- VAE decode ---- */
    float *rgb = (float *)calloc((size_t)3 * img_size * img_size, sizeof(float));
    clock_gettime(CLOCK_MONOTONIC, &t0);
    /* If we already applied BN during unpack, tell VAE to skip internal BN. */
    if (bn_mean_bin && bn_var_bin) setenv("FLUX2_SKIP_VAE_BN", "1", 1);
    if (hip_flux2_vae_decode(r, latent, lat_h, lat_w, rgb) != 0) {
        fprintf(stderr, "generate: VAE decode FAILED\n");
        free(rgb); free(latent); free(img_tok); free(velocity);
        if (txt_padded != txt_hidden) free(txt_padded);
        free(txt_hidden);
        return -1;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double vae_time = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    fprintf(stderr, "generate: VAE %.2f s\n", vae_time);

    if (bn_mean_bin && bn_var_bin) unsetenv("FLUX2_SKIP_VAE_BN");
    save_image(out_path, rgb, img_size, img_size);

    free(rgb); free(latent); free(img_tok); free(velocity);
    if (txt_padded != txt_hidden) free(txt_padded);
    free(txt_hidden);
    return 0;
}

int main(int argc, char **argv) {
    const char *dit_path = NULL;
    const char *vae_path = NULL;
    const char *enc_path = NULL;
    const char *tok_path = NULL;
    const char *txt_bin_path = NULL;
    const char *init_bin_path = NULL;     /* --init-bin: pinned init packed latent */
    const char *sigmas_bin_path = NULL;   /* --sigmas-bin: pinned sigmas */
    const char *dump_final_path = NULL;   /* --dump-final: write HIP final packed latent */
    const char *dump_rgb_path = NULL;     /* --dump-rgb: write HIP raw VAE RGB F32 */
    const char *latent_bin_path = NULL;   /* --vae-only: final packed latent */
    const char *bn_mean_bin = NULL;
    const char *bn_var_bin  = NULL;
    float bn_eps = 1e-4f;
    const char *prompt = "a red apple on a white table";
    const char *out_path = "hip_flux2_out.ppm";
    int device_id = 0;
    int verbose = 1;
    int n_steps = 4;
    int img_size = 256;
    int test_dit_only = 0;
    int generate = 0;
    int vae_only = 0;
    uint64_t seed = 42;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--dit")     && i+1 < argc) dit_path = argv[++i];
        else if (!strcmp(argv[i], "--vae")     && i+1 < argc) vae_path = argv[++i];
        else if (!strcmp(argv[i], "--enc")     && i+1 < argc) enc_path = argv[++i];
        else if (!strcmp(argv[i], "--tok")     && i+1 < argc) tok_path = argv[++i];
        else if (!strcmp(argv[i], "--txt-bin") && i+1 < argc) txt_bin_path = argv[++i];
        else if (!strcmp(argv[i], "--prompt")  && i+1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "-o")        && i+1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "-d")        && i+1 < argc) device_id = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) verbose = 2;
        else if (!strcmp(argv[i], "--steps")   && i+1 < argc) n_steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--size")    && i+1 < argc) img_size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed")    && i+1 < argc) seed = (uint64_t)strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--test-dit")) test_dit_only = 1;
        else if (!strcmp(argv[i], "--generate")) generate = 1;
        else if (!strcmp(argv[i], "--vae-only")) vae_only = 1;
        else if (!strcmp(argv[i], "--init-bin") && i+1 < argc) init_bin_path = argv[++i];
        else if (!strcmp(argv[i], "--sigmas-bin") && i+1 < argc) sigmas_bin_path = argv[++i];
        else if (!strcmp(argv[i], "--dump-final") && i+1 < argc) dump_final_path = argv[++i];
        else if (!strcmp(argv[i], "--dump-rgb") && i+1 < argc) dump_rgb_path = argv[++i];
        else if (!strcmp(argv[i], "--latent-bin") && i+1 < argc) latent_bin_path = argv[++i];
        else if (!strcmp(argv[i], "--bn-mean-bin") && i+1 < argc) bn_mean_bin = argv[++i];
        else if (!strcmp(argv[i], "--bn-var-bin")  && i+1 < argc) bn_var_bin  = argv[++i];
        else if (!strcmp(argv[i], "--bn-eps") && i+1 < argc) bn_eps = (float)atof(argv[++i]);
        else {
            fprintf(stderr,
                    "Usage: %s [--generate|--test-dit] --dit <st> [--vae <st>]\n"
                    "  [--enc <dir_or_st> --tok <tok.gguf>] [--txt-bin <bin>]\n"
                    "  [--prompt <text>] [--steps N] [--size S] [--seed N]\n"
                    "  [-o out.ppm] [-d dev] [-v]\n", argv[0]);
            return 1;
        }
    }

    if (!dit_path) {
        fprintf(stderr, "Error: --dit <path> is required\n");
        return 1;
    }

    rng_seed(seed);

    hip_flux2_runner *r = hip_flux2_init(device_id, verbose);
    if (!r) { fprintf(stderr, "hip_flux2_init failed\n"); return 1; }

    if (hip_flux2_load_dit(r, dit_path) < 0) {
        fprintf(stderr, "Failed to load DiT\n");
        hip_flux2_free(r);
        return 1;
    }

    if (vae_only) {
        if (!vae_path || !latent_bin_path) {
            fprintf(stderr, "Error: --vae-only requires --vae and --latent-bin\n");
            hip_flux2_free(r); return 1;
        }
        int rc = run_vae_only(r, vae_path, latent_bin_path,
                              bn_mean_bin, bn_var_bin, bn_eps,
                              img_size, out_path, dump_rgb_path);
        hip_flux2_free(r);
        return rc ? 1 : 0;
    }

    if (generate) {
        if (!vae_path) { fprintf(stderr, "Error: --generate requires --vae\n"); hip_flux2_free(r); return 1; }
        if (enc_path && !tok_path) {
            fprintf(stderr, "Error: --enc requires --tok\n"); hip_flux2_free(r); return 1;
        }
        (void)path_is_dir;
        int rc = run_generate(r, prompt, enc_path, tok_path, txt_bin_path,
                              init_bin_path, sigmas_bin_path, dump_final_path,
                              bn_mean_bin, bn_var_bin, bn_eps,
                              vae_path, out_path, img_size, n_steps, seed);
        hip_flux2_free(r);
        return rc ? 1 : 0;
    }

    /* ---- Legacy modes (random dummy text) ---- */
    if (vae_path) {
        if (hip_flux2_load_vae(r, vae_path) < 0)
            fprintf(stderr, "Warning: Failed to load VAE, will skip decode\n");
    }

    int patch_size = 2;
    int lat_h = img_size / 8;
    int lat_w = img_size / 8;
    int n_img = (lat_h / patch_size) * (lat_w / patch_size);
    int patch_in_ch = 32 * patch_size * patch_size;
    int n_txt = 128;
    int txt_dim = 7680;

    fprintf(stderr, "Image: %dx%d -> lat %dx%d -> %d patches (patch_in=%d)\n",
            img_size, img_size, lat_h, lat_w, n_img, patch_in_ch);
    fprintf(stderr, "Text: %d tokens x %d dim\n", n_txt, txt_dim);

    float *img_tok  = (float *)calloc((size_t)n_img * patch_in_ch, sizeof(float));
    float *txt_tok  = (float *)calloc((size_t)n_txt * txt_dim, sizeof(float));
    float *velocity = (float *)calloc((size_t)n_img * patch_in_ch, sizeof(float));

    for (int i = 0; i < n_img * patch_in_ch; i++) img_tok[i] = randn();
    for (int i = 0; i < n_txt * txt_dim; i++)      txt_tok[i] = randn() * 0.01f;

    if (test_dit_only) {
        fprintf(stderr, "\n--- Single DiT step test ---\n");
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        int ret = hip_flux2_dit_step(r, img_tok, n_img, txt_tok, n_txt, 1.0f, 0.0f, velocity);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        fprintf(stderr, "DiT step: %s (%.3f s)\n", ret == 0 ? "OK" : "FAIL", dt);

        float sum = 0, mx = 0;
        for (int i = 0; i < n_img * patch_in_ch; i++) {
            sum += velocity[i];
            float a = fabsf(velocity[i]);
            if (a > mx) mx = a;
        }
        fprintf(stderr, "velocity: mean=%.6f max_abs=%.6f\n",
                sum / (n_img * patch_in_ch), mx);
    } else {
        fprintf(stderr, "\n--- %d-step denoising (dummy text) ---\n", n_steps);
        qimg_scheduler sched; qimg_sched_init(&sched);
        qimg_sched_set_timesteps_comfyui(&sched, n_steps, expf(2.02f), 1.0f);

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int step = 0; step < n_steps; step++) {
            int ret = hip_flux2_dit_step(r, img_tok, n_img, txt_tok, n_txt,
                                         sched.sigmas[step], 0.0f, velocity);
            if (ret != 0) { fprintf(stderr, "DiT step %d failed\n", step); break; }
            for (int i = 0; i < n_img * patch_in_ch; i++)
                img_tok[i] += sched.dt[step] * velocity[i];
            fprintf(stderr, "  step %d/%d (sigma=%.4f)\n",
                    step+1, n_steps, sched.sigmas[step]);
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double total = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        fprintf(stderr, "Denoising: %.3f s total (%.3f s/step)\n", total, total / n_steps);

        if (vae_path) {
            float *latent = (float *)calloc((size_t)32 * lat_h * lat_w, sizeof(float));
            unpatchify(img_tok, latent, lat_h, lat_w);

            float *rgb = (float *)calloc((size_t)3 * img_size * img_size, sizeof(float));
            fprintf(stderr, "VAE decode...\n");
            clock_gettime(CLOCK_MONOTONIC, &t0);
            hip_flux2_vae_decode(r, latent, lat_h, lat_w, rgb);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double vae_t = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
            fprintf(stderr, "VAE decode: %.3f s\n", vae_t);

            save_ppm(out_path, rgb, img_size, img_size);
            free(latent); free(rgb);
        }
    }

    free(img_tok); free(txt_tok); free(velocity);
    hip_flux2_free(r);
    return 0;
}
