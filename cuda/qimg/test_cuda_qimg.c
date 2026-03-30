/*
 * test_cuda_qimg.c - CUDA Qwen-Image end-to-end test
 *
 * Modes:
 *   --test-init    : Initialize CUDA, compile kernels
 *   --test-load    : Load DiT + VAE weights to GPU
 *   --test-dit     : Run single DiT step on GPU
 *   --generate     : Full text-to-image pipeline
 *
 * Build:
 *   cc -O2 -mavx2 -mfma -I../../common -I.. -o test_cuda_qimg test_cuda_qimg.c ../cuew.c -lm -ldl -lpthread
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
#define QIMG_DIT_IMPLEMENTATION
#define QIMG_VAE_IMPLEMENTATION
#define QIMG_TEXT_ENCODER_IMPLEMENTATION
#define CUDA_QIMG_RUNNER_IMPLEMENTATION

#include "../../common/gguf_loader.h"
#include "../../common/ggml_dequant.h"
#include "../../common/bpe_tokenizer.h"
#include "../../common/transformer.h"
#include "../../common/qwen_image_scheduler.h"
#include "../../common/qwen_image_dit.h"
#include "../../common/qwen_image_vae.h"
#include "../llm/cuda_llm_runner.h"  /* must be before text_encoder.h for GPU path */
#include "../../common/qwen_image_text_encoder.h"
#include "cuda_qimg_runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* PRNG with Box-Muller pair caching (uses both cos and sin outputs) */
static uint64_t rng_state = 42;
static int rng_has_cached = 0;
static float rng_cached = 0.0f;
static float randn(void) {
    if (rng_has_cached) { rng_has_cached = 0; return rng_cached; }
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u1 = (double)(rng_state >> 11) / (double)(1ULL << 53);
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u2 = (double)(rng_state >> 11) / (double)(1ULL << 53);
    if (u1 < 1e-10) u1 = 1e-10;
    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * 3.14159265358979323846 * u2;
    rng_cached = (float)(r * sin(theta));
    rng_has_cached = 1;
    return (float)(r * cos(theta));
}

static void save_ppm(const char *path, const float *rgb, int h, int w) {
    FILE *fp = fopen(path, "wb");
    if (!fp) return;
    fprintf(fp, "P6\n%d %d\n255\n", w, h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            uint8_t px[3];
            for (int c = 0; c < 3; c++) {
                float v = rgb[(size_t)c * h * w + y * w + x];
                v = v * 0.5f + 0.5f;
                if (v < 0.0f) v = 0.0f;
                if (v > 1.0f) v = 1.0f;
                px[c] = (uint8_t)(v * 255.0f);
            }
            fwrite(px, 1, 3, fp);
        }
    fclose(fp);
    fprintf(stderr, "Saved %s (%dx%d)\n", path, w, h);
}

int main(int argc, char **argv) {
    const char *dit_path = "/mnt/disk01/models/qwen-image-st/diffusion_models/qwen_image_fp8_e4m3fn.safetensors";
    const char *vae_path = "/mnt/disk01/models/qwen-image-st/vae/qwen_image_vae.safetensors";
    const char *enc_path = "/mnt/disk01/models/qwen-image/text-encoder/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf";
    const char *prompt = "a red apple on a white table";
    int custom_prompt = 0;
    const char *mode = NULL;
    int out_h = 256, out_w = 256, n_steps = 20;
    int force_f16 = 0, no_cfg = 0;
    uint64_t seed = 42;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--test-init") == 0) mode = "init";
        else if (strcmp(argv[i], "--test-load") == 0) mode = "load";
        else if (strcmp(argv[i], "--test-dit") == 0) mode = "dit";
        else if (strcmp(argv[i], "--test-vae") == 0) mode = "vae";
        else if (strcmp(argv[i], "--generate") == 0) mode = "gen";
        else if (strcmp(argv[i], "--no-fp8") == 0) force_f16 = 1;
        else if (strcmp(argv[i], "--f16") == 0) force_f16 = 2;  /* F16 MMA path */
        else if (strcmp(argv[i], "--bf16") == 0) force_f16 = 3;  /* BF16 truncation */
        else if (strcmp(argv[i], "--no-cfg") == 0) no_cfg = 1;
        else if (strcmp(argv[i], "--old-gemm") == 0) force_f16 = 4;
        else if (strcmp(argv[i], "--dit") == 0 && i+1 < argc) dit_path = argv[++i];
        else if (strcmp(argv[i], "--vae") == 0 && i+1 < argc) vae_path = argv[++i];
        else if (strcmp(argv[i], "--enc") == 0 && i+1 < argc) enc_path = argv[++i];
        else if (strcmp(argv[i], "--prompt") == 0 && i+1 < argc) { prompt = argv[++i]; custom_prompt = 1; }
        else if (strcmp(argv[i], "--height") == 0 && i+1 < argc) out_h = atoi(argv[++i]);
        else if (strcmp(argv[i], "--width") == 0 && i+1 < argc) out_w = atoi(argv[++i]);
        else if (strcmp(argv[i], "--steps") == 0 && i+1 < argc) n_steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i+1 < argc) seed = (uint64_t)atoll(argv[++i]);
    }

    if (!mode) {
        fprintf(stderr, "Usage: %s --test-init|--test-load|--test-dit|--generate\n"
                "  --dit <st>  --vae <st>  --enc <gguf>  --prompt <text>\n"
                "  --height <h>  --width <w>  --steps <n>  --seed <s>  --no-fp8\n", argv[0]);
        return 1;
    }

    int lat_h = out_h / 8, lat_w = out_w / 8;

    fprintf(stderr, "=== CUDA Qwen-Image: %s ===\n", mode);

    /* ---- Init CUDA (deferred for gen mode — GPU text encoder needs VRAM first) ---- */
    cuda_qimg_runner *r = NULL;
    int verbose_level = 1;
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], "--verbose") == 0 && i+1 < argc) verbose_level = atoi(argv[++i]);

    if (strcmp(mode, "gen") != 0) {
        r = cuda_qimg_init(0, 1);
        r->verbose = verbose_level;
        if (r && force_f16 == 1) { r->use_fp8_gemm = 0; }
        if (r && force_f16 == 2) { r->use_fp8_gemm = 0; r->use_f16_gemm = 1; }
        if (r && force_f16 == 3) { r->use_bf16_trunc = 1; }
        if (r && force_f16 == 4) { r->use_old_gemm = 1; }
        if (!r) { fprintf(stderr, "Init failed\n"); return 1; }
    }

    if (strcmp(mode, "init") == 0) { cuda_qimg_free(r); return 0; }

    /* ---- Load DiT (skip for gen mode — deferred after GPU text encoding) ---- */
    clock_t t0;
    if (r) {
        t0 = clock();
        if (cuda_qimg_load_dit(r, dit_path) != 0) { cuda_qimg_free(r); return 1; }
        fprintf(stderr, "DiT loaded in %.1fs\n", (double)(clock()-t0)/CLOCKS_PER_SEC);
    }

    if (strcmp(mode, "load") == 0) {
        cuda_qimg_load_vae(r, vae_path);
        cuda_qimg_free(r); return 0;
    }

    /* ---- test-vae: decode a .npy latent ---- */
    if (strcmp(mode, "vae") == 0) {
        const char *lat_npy = "../../ref/qwen_image/output/ground_truth_comfyui_512_latent.npy";
        for (int i = 1; i < argc; i++)
            if (strcmp(argv[i], "--latent") == 0 && i+1 < argc) lat_npy = argv[++i];

        FILE *fp = fopen(lat_npy, "rb");
        if (!fp) { fprintf(stderr, "Cannot open %s\n", lat_npy); return 1; }
        uint8_t hdr[10]; fread(hdr,1,10,fp);
        int hl = (int)hdr[8] | ((int)hdr[9]<<8);
        char hbuf[512]; fseek(fp,10,SEEK_SET); fread(hbuf,1,(size_t)hl,fp); hbuf[hl]=0;
        fprintf(stderr, "Latent header: %s\n", hbuf);

        /* Parse shape — support both (1,16,H,W) and (1,16,1,H,W) */
        int dims[5]={0}, ndims=0;
        char *sp = strchr(hbuf, '('); if (sp) { sp++;
            while(*sp && *sp != ')' && ndims < 5) {
                while(*sp == ' ') sp++;
                if (*sp >= '0' && *sp <= '9') { dims[ndims++] = atoi(sp); while(*sp>='0'&&*sp<='9') sp++; }
                if (*sp == ',') sp++;
            }
        }
        int lc, lh, lw;
        if (ndims == 5) { lc=dims[1]; lh=dims[3]; lw=dims[4]; }
        else if (ndims == 4) { lc=dims[1]; lh=dims[2]; lw=dims[3]; }
        else { fprintf(stderr, "Unexpected shape\n"); return 1; }
        fprintf(stderr, "Latent: [%d, %d, %d]\n", lc, lh, lw);

        fseek(fp, 10+hl, SEEK_SET);
        size_t lat_n = (size_t)lc * lh * lw;
        float *latent_raw = (float *)malloc(lat_n * sizeof(float));
        /* For 5D [1,16,1,H,W], read all and reshape to [16,H,W] */
        fread(latent_raw, sizeof(float), lat_n, fp);
        fclose(fp);

        cuda_qimg_load_vae(r, vae_path);
        int oh = lh * 8, ow = lw * 8;
        float *rgb = (float *)malloc((size_t)3 * oh * ow * sizeof(float));
        t0 = clock();
        cuda_qimg_vae_decode(r, latent_raw, lh, lw, rgb);
        fprintf(stderr, "VAE decode: %.1fs\n", (double)(clock()-t0)/CLOCKS_PER_SEC);

        save_ppm("cuda_qimg_vae_test.ppm", rgb, oh, ow);
        free(latent_raw); free(rgb);
        cuda_qimg_free(r); return 0;
    }

    /* ---- test-dit: single step ---- */
    if (strcmp(mode, "dit") == 0) {
        int ps = 2, hp = lat_h/ps, wp = lat_w/ps, n_img = hp*wp, n_txt = 7;
        int in_ch = 64, txt_dim = 3584;
        rng_state = seed;
        float *img = (float *)malloc((size_t)n_img * in_ch * sizeof(float));
        float *txt = (float *)malloc((size_t)n_txt * txt_dim * sizeof(float));
        float *out = (float *)malloc((size_t)n_img * in_ch * sizeof(float));
        for (int i = 0; i < n_img*in_ch; i++) img[i] = randn() * 0.1f;
        for (int i = 0; i < n_txt*txt_dim; i++) txt[i] = randn() * 0.1f;
        t0 = clock();
        cuda_qimg_dit_step(r, img, n_img, txt, n_txt, 500.0f, out);
        fprintf(stderr, "DiT step: %.2fs\n", (double)(clock()-t0)/CLOCKS_PER_SEC);
        { float mn=out[0],mx=out[0],sm=0; int nn=n_img*in_ch, nc=0;
          for(int i=0;i<nn;i++){if(out[i]!=out[i])nc++;else{if(out[i]<mn)mn=out[i];if(out[i]>mx)mx=out[i];sm+=out[i];}}
          fprintf(stderr, "Output: min=%.4f max=%.4f mean=%.4f nan=%d/%d\n",mn,mx,sm/(nn-nc),nc,nn); }
        free(img); free(txt); free(out);
        cuda_qimg_free(r); return 0;
    }

    /* ---- generate: full text-to-image pipeline ---- */
    if (strcmp(mode, "gen") == 0) {
        int ps = 2;
        int hp = lat_h / ps, wp = lat_w / ps;
        int n_img = hp * wp;
        int in_ch = 64, lat_ch = 16;
        int txt_dim = 3584;

        /* 1. Text encoder — try loading ComfyUI .npy first, fall back to GGUF */
        clock_t enc_t0 = clock();
        fprintf(stderr, "\n[1/3] Text conditioning...\n");
        int n_txt = 0, n_txt_neg = 0;
        float *txt_hidden = NULL, *txt_neg_hidden = NULL;

        /* Try ComfyUI pre-encoded hidden states (only for default prompt) */
        if (!custom_prompt) {
            /* Try loading from several locations */
            const char *pos_paths[] = {
                "../../ref/qwen_image/output/comfyui_text_hidden.npy",
                "../../../diffusion/ref/qwen_image/output/comfyui_text_hidden.npy",
                "comfyui_text_hidden.npy", NULL};
            const char *pos_path = NULL, *neg_path = NULL;
            for (int pi = 0; pos_paths[pi]; pi++) {
                FILE *tf = fopen(pos_paths[pi], "rb");
                if (tf) { fclose(tf); pos_path = pos_paths[pi]; break; }
            }
            if (!pos_path) pos_path = "comfyui_text_hidden.npy";
            {
                /* Derive neg path from pos path */
                static char neg_buf[512];
                const char *dot = strrchr(pos_path, '.');
                if (dot) {
                    size_t base_len = (size_t)(dot - pos_path);
                    memcpy(neg_buf, pos_path, base_len);
                    snprintf(neg_buf + base_len, sizeof(neg_buf) - base_len, "_neg.npy");
                    neg_path = neg_buf;
                }
            }
            FILE *fp = fopen(pos_path, "rb");
            if (fp) {
                uint8_t hdr[10]; if(fread(hdr,1,10,fp)==10) {
                    int hl = (int)hdr[8] | ((int)hdr[9]<<8);
                    char hbuf[512]; fseek(fp,10,SEEK_SET);
                    if(fread(hbuf,1,(size_t)hl,fp)==(size_t)hl) {
                        hbuf[hl]=0;
                        /* Parse shape (1, N, 3584) */
                        char *sp = strstr(hbuf, "shape");
                        if (sp) { sp = strchr(sp, '(');
                            if (sp) { sp++; /* skip batch dim */
                                while(*sp && *sp!=',') sp++; if(*sp==',') sp++;
                                n_txt = atoi(sp);
                            }
                        }
                    }
                    if (n_txt > 0) {
                        fseek(fp, 10+hl, SEEK_SET);
                        txt_hidden = (float*)malloc((size_t)n_txt*txt_dim*sizeof(float));
                        fread(txt_hidden, sizeof(float), (size_t)n_txt*txt_dim, fp);
                        fprintf(stderr, "  Loaded ComfyUI pos hidden [%d, %d]\n", n_txt, txt_dim);
                    }
                }
                fclose(fp);
                /* Load negative */
                fp = fopen(neg_path, "rb");
                if (fp) {
                    uint8_t hdr2[10]; fread(hdr2,1,10,fp);
                    int hl2 = (int)hdr2[8] | ((int)hdr2[9]<<8);
                    char hb2[512]; fseek(fp,10,SEEK_SET);
                    if(fread(hb2,1,(size_t)hl2,fp)==(size_t)hl2) {
                        hb2[hl2]=0;
                        char *sp2 = strstr(hb2, "shape");
                        if (sp2) { sp2 = strchr(sp2, '(');
                            if (sp2) { sp2++; while(*sp2&&*sp2!=',') sp2++; if(*sp2==',') sp2++;
                                n_txt_neg = atoi(sp2); } }
                    }
                    if (n_txt_neg > 0) {
                        fseek(fp, 10+hl2, SEEK_SET);
                        txt_neg_hidden = (float*)malloc((size_t)n_txt_neg*txt_dim*sizeof(float));
                        fread(txt_neg_hidden, sizeof(float), (size_t)n_txt_neg*txt_dim, fp);
                        fprintf(stderr, "  Loaded ComfyUI neg hidden [%d, %d]\n", n_txt_neg, txt_dim);
                    }
                    fclose(fp);
                }
            }
        } /* if !custom_prompt */

        /* Try GPU text encoder (GGUF + biases, ~500× faster than CPU).
         * Uses primary CUDA context shared with DiT runner. */
        if (!txt_hidden) {
            const char *bias_st = "/mnt/disk01/models/qwen-image-st/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors";
            qimg_text_enc *enc = qimg_text_enc_load_gpu(enc_path, bias_st, 0);
            if (enc) {
                txt_hidden = qimg_text_enc_encode(enc, prompt, &n_txt);
                fprintf(stderr, "  Encoding negative prompt...\n");
                txt_neg_hidden = qimg_text_enc_encode(enc, " ", &n_txt_neg);
                qimg_text_enc_free(enc);
            }
        }
        /* CPU GGUF fallback (slow but works without GPU LLM runner) */
        if (!txt_hidden) {
            const char *bias_st = "/mnt/disk01/models/qwen-image-st/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors";
            qimg_text_enc *enc = qimg_text_enc_load_gguf_with_biases(enc_path, bias_st);
            if (enc) {
                txt_hidden = qimg_text_enc_encode(enc, prompt, &n_txt);
                fprintf(stderr, "  Encoding negative prompt...\n");
                txt_neg_hidden = qimg_text_enc_encode(enc, " ", &n_txt_neg);
                qimg_text_enc_free(enc);
            }
        }
        /* FP8 safetensors fallback (dequants to F32 — 30GB, slow on CPU) */
        if (!txt_hidden) {
            const char *st_enc = "/mnt/disk01/models/qwen-image-st/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors";
            FILE *tf = fopen(st_enc, "rb");
            if (tf) {
                fclose(tf);
                fprintf(stderr, "  Loading FP8 scaled text encoder (slow fallback)...\n");
                qimg_text_enc *enc = qimg_text_enc_load_safetensors(st_enc, enc_path);
                if (enc) {
                    txt_hidden = qimg_text_enc_encode(enc, prompt, &n_txt);
                    fprintf(stderr, "  Encoding negative prompt...\n");
                    txt_neg_hidden = qimg_text_enc_encode(enc, " ", &n_txt_neg);
                    qimg_text_enc_free(enc);
                }
            }
        }
        if (!txt_hidden) {
            fprintf(stderr, "Text encoder failed\n");
            free(txt_neg_hidden);
            cuda_qimg_free(r); return 1;
        }
        /* ComfyUI processes cond/uncond separately on 16GB GPUs (not enough VRAM
         * to batch). Each pass uses its ORIGINAL token count.
         * Cond: 12 tokens, Uncond: 6 tokens — different attention patterns. */
        fprintf(stderr, "  Positive: %d tokens, Negative: %d tokens (separate passes)\n", n_txt, n_txt_neg);
        fprintf(stderr, "  Text encoding: %.1fs\n", (double)(clock() - enc_t0) / CLOCKS_PER_SEC);

        /* Init qimg + load DiT AFTER text encoding frees GPU memory */
        r = cuda_qimg_init(0, verbose_level);
        if (!r) { fprintf(stderr, "CUDA qimg init failed\n"); return 1; }
        r->verbose = verbose_level;
        if (force_f16 == 1) { r->use_fp8_gemm = 0; }
        if (force_f16 == 2) { r->use_fp8_gemm = 0; r->use_f16_gemm = 1; }
        if (force_f16 == 4) { r->use_old_gemm = 1; }
        t0 = clock();
        if (cuda_qimg_load_dit(r, dit_path) != 0) { cuda_qimg_free(r); return 1; }
        fprintf(stderr, "DiT loaded in %.1fs\n", (double)(clock()-t0)/CLOCKS_PER_SEC);

        /* 2. DiT denoising loop (CUDA) */
        fprintf(stderr, "\n[2/3] DiT denoising (%d steps, %dx%d, %s)...\n",
                n_steps, out_w, out_h, r->use_fp8_gemm ? "FP8 native" : "F16 fallback");

        /* Initialize noise latent with ComfyUI's noise_scaling.
         * CONST.noise_scaling(sigma, noise, latent_image, max_denoise=True):
         *   noise = noise * sqrt(1.0 + sigma^2) + latent_image
         * For sigma[0]=1.0, latent_image=0: noise *= sqrt(2) */
        rng_state = seed;
        float *latent = (float *)malloc((size_t)lat_ch * lat_h * lat_w * sizeof(float));
        for (size_t i = 0; i < (size_t)lat_ch * lat_h * lat_w; i++)
            latent[i] = randn();

        /* Scheduler — match ComfyUI ground truth exactly:
         * shift=3.1, multiplier=1000 (our embedding uses angle=t*freq,
         * ComfyUI's Timesteps has internal scale=1000 so t=sigma*1000) */
        qimg_scheduler sched;
        qimg_sched_init(&sched);
        qimg_sched_set_timesteps_comfyui(&sched, n_steps, 3.1f, 1000.0f);

        /* CONST.noise_scaling: sigma * noise + (1-sigma) * latent_image
         * For sigma=1.0, latent_image=0: noise unchanged. No sqrt(2) scaling. */

        float cfg_scale = 2.5f;
        for (int i = 1; i < argc; i++)
            if (strcmp(argv[i], "--cfg-scale") == 0 && i+1 < argc)
                cfg_scale = (float)atof(argv[++i]);

        /* Patchify + denoising loop with CFG */
        float *img_tokens = (float *)malloc((size_t)n_img * in_ch * sizeof(float));
        float *vel_cond = (float *)malloc((size_t)n_img * in_ch * sizeof(float));
        float *vel_uncond = (float *)malloc((size_t)n_img * in_ch * sizeof(float));

        clock_t denoise_t0 = clock();
        for (int step = 0; step < n_steps; step++) {
            float t_val = sched.timesteps[step];
            clock_t step_t0 = clock();

            /* Patchify latent */
            qimg_dit_patchify(img_tokens, latent, lat_ch, lat_h, lat_w, ps);

            /* Conditional DiT forward (with text) */
            cuda_qimg_dit_step(r, img_tokens, n_img, txt_hidden, n_txt,
                               t_val, vel_cond);

            int lat_n = lat_ch * lat_h * lat_w;
            float *vel_latent = (float *)malloc((size_t)lat_n * sizeof(float));

            /* Euler step: x += v * dt (model output is velocity) */
            if (no_cfg) {
                qimg_dit_unpatchify(vel_latent, vel_cond, n_img, lat_ch, lat_h, lat_w, ps);
                /* Debug: save step 1 model output (verbose >= 3) */
                if (step == 0 && r->verbose >= 3) {
                    FILE *vf = fopen("dit_output_step1.bin", "wb");
                    if (vf) { fwrite(vel_cond, sizeof(float), (size_t)n_img*in_ch, vf); fclose(vf);
                        fprintf(stderr, "  Saved dit_output_step1.bin [%d, %d]\n", n_img, in_ch); }
                    FILE *lf2 = fopen("dit_vel_latent_step1.bin", "wb");
                    if (lf2) { fwrite(vel_latent, sizeof(float), (size_t)lat_n, lf2); fclose(lf2); }
                }
            } else {
                cuda_qimg_dit_step(r, img_tokens, n_img, txt_neg_hidden, n_txt_neg,
                                   t_val, vel_uncond);

                float *vc_lat = (float *)malloc((size_t)lat_n * sizeof(float));
                float *vu_lat = (float *)malloc((size_t)lat_n * sizeof(float));
                qimg_dit_unpatchify(vc_lat, vel_cond, n_img, lat_ch, lat_h, lat_w, ps);
                qimg_dit_unpatchify(vu_lat, vel_uncond, n_img, lat_ch, lat_h, lat_w, ps);

                /* Debug: save separate velocities at step 0 */
                if (step == 0 && r->verbose >= 2) {
                    float vc_std=0,vu_std=0,diff_std=0;
                    for(int i=0;i<lat_n;i++){vc_std+=vc_lat[i]*vc_lat[i];vu_std+=vu_lat[i]*vu_lat[i];
                        float d=vc_lat[i]-vu_lat[i];diff_std+=d*d;}
                    fprintf(stderr, "    step0: v_cond_std=%.4f  v_uncond_std=%.4f  diff_std=%.4f\n",
                            sqrtf(vc_std/lat_n), sqrtf(vu_std/lat_n), sqrtf(diff_std/lat_n));
                }

                for (int i = 0; i < lat_n; i++)
                    vel_latent[i] = vu_lat[i] + cfg_scale * (vc_lat[i] - vu_lat[i]);
                free(vc_lat); free(vu_lat);
            }

            /* Track velocity stats (verbose >= 2) */
            if (r->verbose >= 2) {
                float vs=0,vs2=0; int vn=lat_ch*lat_h*lat_w;
                for(int i=0;i<vn;i++){vs+=vel_latent[i];vs2+=vel_latent[i]*vel_latent[i];}
                float vm=vs/vn, vstd=sqrtf(vs2/vn-vm*vm);
                fprintf(stderr, "    vel: mean=%.4f std=%.4f dt=%.6f\n", vm, vstd, sched.dt[step]);
            }

            /* Save per-step latent (before Euler step) */
            if (r->verbose >= 3) {
                char fn[64]; snprintf(fn, sizeof(fn), "cuda_perstep_%02d.bin", step);
                FILE *sf = fopen(fn, "wb");
                if (sf) { fwrite(latent, sizeof(float), (size_t)lat_n, sf); fclose(sf); }
            }

            /* Euler step */
            qimg_sched_step(latent, vel_latent, lat_n, step, &sched);
            free(vel_latent);

            /* Track latent stats */
            { double step_s = (double)(clock() - step_t0) / CLOCKS_PER_SEC;
              if (r->verbose >= 2) {
                  float lmn=latent[0],lmx=latent[0],ls=0,ls2=0;
                  int ln=lat_ch*lat_h*lat_w;
                  for(int i=0;i<ln;i++){if(latent[i]<lmn)lmn=latent[i];if(latent[i]>lmx)lmx=latent[i];ls+=latent[i];ls2+=latent[i]*latent[i];}
                  float mean=ls/ln, std=sqrtf(ls2/ln - mean*mean);
                  fprintf(stderr, "  step %d/%d  t=%.3f  lat=[%.2f,%.2f] mean=%.3f std=%.3f  %.2fs\n",
                          step+1, n_steps, t_val, lmn, lmx, mean, std, step_s);
              } else if (r->verbose >= 1) {
                  fprintf(stderr, "  step %d/%d  %.2fs\n", step+1, n_steps, step_s);
              } }
        }
        double denoise_s = (double)(clock() - denoise_t0) / CLOCKS_PER_SEC;
        fprintf(stderr, "Denoising: %.1fs total (%.2fs/step)\n", denoise_s, denoise_s / n_steps);

        free(img_tokens); free(vel_cond); free(vel_uncond);
        free(txt_hidden); free(txt_neg_hidden);

        /* Wan21 process_latent_out: latent = latent * latents_std + latents_mean
         * The DiT operates in normalized space. Denormalize before VAE decode. */
        {
            static const float wan21_mean[16] = {
                -0.7571f, -0.7089f, -0.9113f,  0.1075f, -0.1745f,  0.9653f, -0.1517f,  1.5508f,
                 0.4134f, -0.0715f,  0.5517f, -0.3632f, -0.1922f, -0.9497f,  0.2503f, -0.2921f
            };
            static const float wan21_std[16] = {
                2.8184f, 1.4541f, 2.3275f, 2.6558f, 1.2196f, 1.7708f, 2.6052f, 2.0743f,
                3.2687f, 2.1526f, 2.8652f, 1.5579f, 1.6382f, 1.1253f, 2.8251f, 1.9160f
            };
            int spatial = lat_h * lat_w;
            for (int c = 0; c < lat_ch; c++)
                for (int s = 0; s < spatial; s++)
                    latent[c * spatial + s] = latent[c * spatial + s] * wan21_std[c] + wan21_mean[c];
            fprintf(stderr, "  Applied Wan21 latent denormalization\n");
        }

        /* Save denoised latent for debugging */
        {
            int lat_n = lat_ch * lat_h * lat_w;
            FILE *lf = fopen("cuda_latent.bin", "wb");
            if (lf) { fwrite(latent, sizeof(float), (size_t)lat_n, lf); fclose(lf);
                fprintf(stderr, "Saved latent [%d,%d,%d] to cuda_latent.bin\n", lat_ch, lat_h, lat_w); }
        }

        /* 3. VAE decode */
        fprintf(stderr, "\n[3/3] VAE decode (CPU)...\n");
        cuda_qimg_load_vae(r, vae_path);
        float *rgb = (float *)malloc((size_t)3 * out_h * out_w * sizeof(float));
        t0 = clock();
        cuda_qimg_vae_decode(r, latent, lat_h, lat_w, rgb);
        fprintf(stderr, "VAE decode: %.1fs\n", (double)(clock()-t0)/CLOCKS_PER_SEC);
        
        free(latent);

        /* Save */
        save_ppm("cuda_qimg_output.ppm", rgb, out_h, out_w);
        free(rgb);
    }

    cuda_qimg_free(r);
    return 0;
}
